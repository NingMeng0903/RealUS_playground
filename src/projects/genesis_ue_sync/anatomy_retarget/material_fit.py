"""Shape-preserving articulated rest fitting for anatomy schema v5.

Rigid anatomy is fitted from semantic joints and material groups.  Soft
materials follow the finalized authored driver frames through their original
sparse weights.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms, with_source_driver_coupling
from .rigged_asset import AnatomyRiggedAsset
from .source_rebind import rebind_source_rig


_CRANIAL_TOKENS = (
    "skull",
    "cranium",
    "brain",
    "cerebr",
    "cerebell",
    "midbrain",
    "amygdala",
    "basal_ganglia",
    "corpus_callosum",
    "occipital_lobe",
    "temporal_lobe",
    "frontal_lobe",
    "parietal_lobe",
    "thalam",
    "hypothalam",
    "pituitary",
    "pineal",
    "fornix",
    "upper_teeth",
)
_PELVIS_TOKENS = ("ilium", "sacrum", "ischium", "pubis", "pelvis")
_LONG_BONE_TOKENS = (
    "clavicle",
    "humerus",
    "radius",
    "ulna",
    "femur",
    "tibia",
    "fibula",
    "metacarpal",
    "phalanx_hand",
    "phalanges_hand",
    "finger_",
)
_FOOT_TOKENS = (
    "calcaneus",
    "talus",
    "navicular",
    "cuboid",
    "cuneiform",
    "metatarsal",
    "phalanx_foot",
    "phalanges_foot",
)


def _load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
    return np.asarray(vertices, dtype=np.float64)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = np.array(source, dtype=np.float64, copy=True)
    b = np.array(target, dtype=np.float64, copy=True)
    a /= max(float(np.linalg.norm(a)), 1.0e-12)
    b /= max(float(np.linalg.norm(b)), 1.0e-12)
    cross = np.cross(a, b)
    cosine = float(np.clip(a @ b, -1.0, 1.0))
    norm = float(np.linalg.norm(cross))
    if norm < 1.0e-10:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis -= a * float(axis @ a)
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)
    skew = np.asarray(
        ((0.0, -cross[2], cross[1]), (cross[2], 0.0, -cross[0]), (-cross[1], cross[0], 0.0)),
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (norm * norm))


def _vector_set_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Best proper rotation from two or more corresponding direction vectors."""
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    valid = (np.linalg.norm(src, axis=1) > 1.0e-8) & (np.linalg.norm(dst, axis=1) > 1.0e-8)
    src, dst = src[valid], dst[valid]
    if len(src) < 2:
        return _rotation_between(src[0], dst[0]) if len(src) else np.eye(3)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    dst /= np.linalg.norm(dst, axis=1, keepdims=True)
    u, _singular, vt = np.linalg.svd(src.T @ dst)
    row_rotation = u @ vt
    if np.linalg.det(row_rotation) < 0.0:
        u[:, -1] *= -1.0
        row_rotation = u @ vt
    return row_rotation.T


def shaft_preserving_segment_map(
    points: np.ndarray,
    *,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    end_fraction: float = 0.20,
) -> np.ndarray:
    """Fit segment length in the shaft while keeping both epiphyses rigid."""
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sa = np.asarray(source_a, dtype=np.float64).reshape(3)
    sb = np.asarray(source_b, dtype=np.float64).reshape(3)
    ta = np.asarray(target_a, dtype=np.float64).reshape(3)
    tb = np.asarray(target_b, dtype=np.float64).reshape(3)
    source_vector = sb - sa
    target_vector = tb - ta
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1.0e-6 or target_length < 1.0e-6:
        return source.copy()
    source_axis = source_vector / source_length
    target_axis = target_vector / target_length
    rotation = _rotation_between(source_axis, target_axis)
    rigid = (source - sa) @ rotation.T + ta
    parameter = np.clip(((source - sa) @ source_axis) / source_length, 0.0, 1.0)
    lo = float(np.clip(end_fraction, 0.0, 0.45))
    hi = 1.0 - lo
    t = np.clip((parameter - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return rigid + smooth[:, None] * (target_length - source_length) * target_axis


def uniform_segment_similarity(
    points: np.ndarray,
    *,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit a compound by one rotation and one scale; never stretch one axis."""
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sa = np.asarray(source_a, dtype=np.float64).reshape(3)
    sb = np.asarray(source_b, dtype=np.float64).reshape(3)
    ta = np.asarray(target_a, dtype=np.float64).reshape(3)
    tb = np.asarray(target_b, dtype=np.float64).reshape(3)
    source_vector = sb - sa
    target_vector = tb - ta
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length <= 1.0e-8 or target_length <= 1.0e-8:
        raise ValueError("uniform segment similarity requires nondegenerate landmarks")
    rotation = _rotation_between(source_vector, target_vector)
    scale = target_length / source_length
    return ta + scale * ((source - sa) @ rotation.T), float(scale), rotation


def _anatomical_frame(
    *, origin: np.ndarray, lateral: np.ndarray, superior: np.ndarray
) -> np.ndarray:
    """Build a stable right-handed local anatomical frame."""
    x = np.asarray(lateral, dtype=np.float64).reshape(3).copy()
    y_hint = np.asarray(superior, dtype=np.float64).reshape(3).copy()
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    y = y_hint - x * float(x @ y_hint)
    y /= max(float(np.linalg.norm(y)), 1.0e-10)
    z = np.cross(x, y)
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    y = np.cross(z, x)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = np.stack((x, y, z), axis=1)
    frame[:3, 3] = np.asarray(origin, dtype=np.float64).reshape(3)
    return frame


def _frame_coordinates(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) - frame[:3, 3]) @ frame[:3, :3]


def _from_frame_coordinates(points: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ frame[:3, :3].T + frame[:3, 3]


def freeze_soft_material(
    fitted_vertices: np.ndarray,
    source_vertices: np.ndarray,
    asset: AnatomyRiggedAsset,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore non-bone vertices after a bone-only material operation."""
    fitted = np.asarray(fitted_vertices, dtype=np.float64).copy()
    source = np.asarray(source_vertices, dtype=np.float64)
    soft = _soft_material_mask(asset)
    fitted[soft] = source[soft]
    return fitted, soft


def _spine_anchor_specs(
    asset: AnatomyRiggedAsset,
    target_joints: np.ndarray,
    pelvis_l5_interface: np.ndarray,
    skull_c1_interface: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Final connection anchors for the authored L5-to-C1 chain."""
    joint_id = {name: index for index, name in enumerate(asset.joint_names)}
    targets = np.asarray(target_joints, dtype=np.float64)
    specs: list[tuple[str, np.ndarray]] = [
        ("Spine_L5", np.asarray(pelvis_l5_interface, dtype=np.float64).reshape(3))
    ]
    for bone_name, joint_name in (
        ("Spine_L2", "spine2"),
        ("Spine_T8", "spine3"),
        ("Spine_C7", "neck"),
    ):
        if joint_name in joint_id:
            specs.append((bone_name, targets[joint_id[joint_name]].copy()))
    if skull_c1_interface is not None:
        specs.append(("Spine_C1", np.asarray(skull_c1_interface, dtype=np.float64).reshape(3)))
    elif "head" in joint_id:
        specs.append(("Head_Bone", targets[joint_id["head"]].copy()))
    return specs


def _scapula_side_axes(
    *,
    shoulder: np.ndarray,
    collar: np.ndarray,
    spine3: np.ndarray,
) -> np.ndarray:
    """Unit medial axis (toward collar/spine, away from GH joint)."""
    shoulder = np.asarray(shoulder, dtype=np.float64).reshape(3)
    collar = np.asarray(collar, dtype=np.float64).reshape(3)
    spine3 = np.asarray(spine3, dtype=np.float64).reshape(3)
    medial = 0.65 * (collar - shoulder) + 0.35 * (spine3 - shoulder)
    medial_norm = float(np.linalg.norm(medial))
    if medial_norm < 1.0e-8:
        medial = np.array([-np.sign(shoulder[0] or 1.0), 0.0, 0.0], dtype=np.float64)
        medial_norm = 1.0
    return medial / medial_norm


def _scapula_thorax_transform(
    *,
    source_shoulder: np.ndarray,
    source_collar: np.ndarray,
    source_spine3: np.ndarray,
    target_shoulder: np.ndarray,
    target_collar: np.ndarray,
    target_spine3: np.ndarray,
) -> np.ndarray:
    """Same shoulder/collar/spine3 frame as anatomy_lbs scapula pose driver."""
    from .anatomy_lbs import _rigid_frame

    source = _rigid_frame(source_shoulder, source_collar, source_spine3)
    target = _rigid_frame(target_shoulder, target_collar, target_spine3)
    return target @ np.linalg.inv(source)


def _fit_scapula_mesh(
    points: np.ndarray,
    *,
    side: str,
    source_shoulder: np.ndarray,
    source_collar: np.ndarray,
    source_spine3: np.ndarray,
    target_shoulder: np.ndarray,
    target_collar: np.ndarray,
    target_spine3: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bounded uniform similarity preserving glenoid/acromion/blade geometry."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        return pts.copy(), {"vertices": int(len(pts))}
    transform = _scapula_thorax_transform(
        source_shoulder=source_shoulder,
        source_collar=source_collar,
        source_spine3=source_spine3,
        target_shoulder=target_shoulder,
        target_collar=target_collar,
        target_spine3=target_spine3,
    )
    source_lengths = np.asarray(
        (
            np.linalg.norm(source_collar - source_shoulder),
            np.linalg.norm(source_spine3 - source_shoulder),
        ),
        dtype=np.float64,
    )
    target_lengths = np.asarray(
        (
            np.linalg.norm(target_collar - target_shoulder),
            np.linalg.norm(target_spine3 - target_shoulder),
        ),
        dtype=np.float64,
    )
    similarity_scale = float(
        np.clip(np.median(target_lengths / np.maximum(source_lengths, 1.0e-8)), 0.85, 1.15)
    )
    fitted = np.asarray(target_shoulder, dtype=np.float64) + similarity_scale * (
        (pts - np.asarray(source_shoulder, dtype=np.float64)) @ transform[:3, :3].T
    )
    distance = np.linalg.norm(fitted - target_shoulder, axis=1)
    glenoid_candidates = fitted[distance <= np.quantile(distance, 0.05)]
    glenoid = np.mean(glenoid_candidates, axis=0)
    glenoid_translation = np.asarray(target_shoulder) - glenoid
    fitted += glenoid_translation
    return fitted, {
        "vertices": int(len(pts)),
        "uniform_scale": similarity_scale,
        "glenoid_to_shoulder_m": float(
            np.linalg.norm((glenoid + glenoid_translation) - target_shoulder)
        ),
        "glenoid_translation_m": float(np.linalg.norm(glenoid_translation)),
        "aspect_ratio_change": _aspect_ratio_change(pts, fitted),
    }


def _dominant_bone(asset: AnatomyRiggedAsset, start: int, stop: int) -> int | None:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        return None
    indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
    weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
    mass = np.bincount(indices, weights=weights, minlength=len(asset.source_bone_names))
    return int(np.argmax(mass)) if mass.size and float(mass.max()) > 0.0 else None


def _is_follow_mode(mode: str) -> bool:
    return str(mode) in {"bind_follow", "parent_follow"}


def _is_joint_local_mode(mode: str) -> bool:
    return str(mode) in {"joint_local", "direct_joint"}


def _direct_smplx_hand_controllers(asset: AnatomyRiggedAsset) -> list[int]:
    """Independent wrist/finger controls must rotate at their fitted joints."""
    modes = list(asset.source_bone_driver_types or [])
    mapped = np.asarray(asset.source_bone_smplx_a, dtype=np.int64)
    direct: list[int] = []
    for bone, mode in enumerate(modes):
        if not _is_joint_local_mode(mode):
            continue
        joint = str(asset.joint_names[int(mapped[bone])])
        side, separator, part = joint.partition("_")
        if side not in {"left", "right"} or not separator:
            continue
        if part == "wrist" or part.startswith(
            ("thumb", "index", "middle", "ring", "pinky")
        ):
            direct.append(int(bone))
    return direct


def _controller(bone: int, parents: np.ndarray, modes: list[str]) -> int:
    current = int(bone)
    while current >= 0 and _is_follow_mode(modes[current]):
        current = int(parents[current])
    return int(bone if current < 0 else current)


def _source_joint_anchors(
    asset: AnatomyRiggedAsset,
    *,
    bind_global: np.ndarray | None = None,
) -> np.ndarray:
    target = np.asarray(asset.rest_joints, dtype=np.float64)
    anchors = target.copy()
    assigned = np.zeros(len(target), dtype=bool)
    modes = list(asset.source_bone_driver_types or [])
    global_bind = np.asarray(
        asset.target_bind_global if bind_global is None else bind_global,
        dtype=np.float64,
    )
    for bone, mode in enumerate(modes):
        if _is_follow_mode(mode):
            continue
        joint = int(asset.source_bone_smplx_a[bone])
        if not assigned[joint]:
            anchors[joint] = global_bind[bone, :3, 3]
            assigned[joint] = True
    return anchors


def _joint_child(joint: int, parents: np.ndarray) -> int | None:
    children = np.flatnonzero(np.asarray(parents, dtype=np.int64) == int(joint))
    return int(children[0]) if len(children) else None


def _correct_clavicle_segment_joints(asset: AnatomyRiggedAsset) -> AnatomyRiggedAsset:
    """Force clavicle drivers onto collar→shoulder (not spine3→collar)."""
    modes = list(asset.source_bone_driver_types or [])
    if not modes or asset.source_bone_smplx_a is None or asset.source_bone_smplx_b is None:
        return asset
    joint_id = {name: i for i, name in enumerate(asset.joint_names)}
    a = np.asarray(asset.source_bone_smplx_a, dtype=np.int32).copy()
    b = np.asarray(asset.source_bone_smplx_b, dtype=np.int32).copy()
    changed = False
    for bi, mode in enumerate(modes):
        mode_s = str(mode)
        if not mode_s.startswith("clavicle_segment_"):
            continue
        side = "left" if mode_s.endswith("left") else "right"
        collar = joint_id.get(f"{side}_collar")
        shoulder = joint_id.get(f"{side}_shoulder")
        if collar is None or shoulder is None:
            continue
        if int(a[bi]) != int(collar) or int(b[bi]) != int(shoulder):
            a[bi] = int(collar)
            b[bi] = int(shoulder)
            changed = True
    if not changed:
        return asset
    return type(asset)(
        **{
            **asset.__dict__,
            "source_bone_smplx_a": a.astype(np.int16),
            "source_bone_smplx_b": b.astype(np.int16),
        }
    )


def _three_joint_frame(points: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """Anatomical frame from origin, secondary landmark and distal landmark."""
    ids = np.asarray(joints, dtype=np.int64).reshape(3)
    origin = np.asarray(points[ids[0]], dtype=np.float64)
    x = np.asarray(points[ids[1]], dtype=np.float64) - origin
    y_hint = np.asarray(points[ids[2]], dtype=np.float64) - origin
    x /= max(float(np.linalg.norm(x)), 1.0e-10)
    z = np.cross(x, y_hint)
    z /= max(float(np.linalg.norm(z)), 1.0e-10)
    y = np.cross(z, x)
    y /= max(float(np.linalg.norm(y)), 1.0e-10)
    frame = np.eye(4, dtype=np.float64)
    frame[:3, :3] = np.stack((x, y, z), axis=1)
    frame[:3, 3] = origin
    return frame


def _fit_source_frames(
    asset: AnatomyRiggedAsset,
    *,
    preserve_same_semantic_offset: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    old_local = np.asarray(asset.target_bind_local, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64).copy()
    source_anchors = _source_joint_anchors(asset)
    raw_frame_joints = getattr(asset, "source_bone_frame_joints", None)
    frame_joints = (
        np.asarray(raw_frame_joints, dtype=np.int64)
        if raw_frame_joints is not None
        else np.full((len(modes), 3), -1, dtype=np.int64)
    )
    new_global = np.empty_like(old_global)
    for bone, mode in enumerate(modes):
        parent = int(source_parents[bone])
        if _is_follow_mode(mode) and parent >= 0:
            new_global[bone] = new_global[parent] @ old_local[bone]
            continue
        a = int(asset.source_bone_smplx_a[bone])
        b = int(asset.source_bone_smplx_b[bone])
        if a == b and _is_joint_local_mode(mode):
            child = _joint_child(a, asset.parents)
            if child is not None:
                b = child
        rotation = old_global[bone, :3, :3].copy()
        joint_name = asset.joint_names[a]
        explicit = frame_joints[bone]
        explicit_desired: np.ndarray | None = None
        if np.all(explicit >= 0) and len(np.unique(explicit)) == 3:
            source_frame = _three_joint_frame(source_anchors, explicit)
            target_frame = _three_joint_frame(target_joints, explicit)
            explicit_desired = target_frame @ np.linalg.inv(source_frame) @ old_global[bone]
            rotation = explicit_desired[:3, :3]
        elif joint_name in {"left_wrist", "right_wrist"}:
            side = joint_name.split("_", 1)[0]
            roots = [
                asset.joint_names.index(f"{side}_{finger}1")
                for finger in ("thumb", "index", "middle", "ring", "pinky")
                if f"{side}_{finger}1" in asset.joint_names
            ]
            if roots:
                rotation = _vector_set_rotation(
                    source_anchors[roots] - source_anchors[a],
                    target_joints[roots] - target_joints[a],
                ) @ rotation
        elif joint_name in {"left_ankle", "right_ankle"} and a != b:
            side = joint_name.split("_", 1)[0]
            knee = asset.joint_names.index(f"{side}_knee")
            rotation = _vector_set_rotation(
                np.stack((source_anchors[b] - source_anchors[a], source_anchors[knee] - source_anchors[a])),
                np.stack((target_joints[b] - target_joints[a], target_joints[knee] - target_joints[a])),
            ) @ rotation
        elif str(mode).startswith("scapula_"):
            side = "left" if str(mode).endswith("left") else "right"
            joint_id = {name: idx for idx, name in enumerate(asset.joint_names)}
            spine3 = joint_id["spine3"]
            collar = joint_id[f"{side}_collar"]
            shoulder = joint_id[f"{side}_shoulder"]
            scapula_delta = _scapula_thorax_transform(
                source_shoulder=source_anchors[shoulder],
                source_collar=source_anchors[collar],
                source_spine3=source_anchors[spine3],
                target_shoulder=target_joints[shoulder],
                target_collar=target_joints[collar],
                target_spine3=target_joints[spine3],
            )
            desired = scapula_delta @ old_global[bone]
            if parent < 0:
                new_global[bone] = desired
            else:
                new_global[bone] = new_global[parent] @ (np.linalg.inv(new_global[parent]) @ desired)
            continue
        elif a != b:
            source_vector = source_anchors[b] - source_anchors[a]
            target_vector = target_joints[b] - target_joints[a]
            if float(np.linalg.norm(source_vector)) > 1.0e-6 and float(np.linalg.norm(target_vector)) > 1.0e-6:
                rotation = _rotation_between(source_vector, target_vector) @ rotation
        if explicit_desired is not None:
            desired = explicit_desired
        else:
            desired = np.eye(4, dtype=np.float64)
            desired[:3, :3] = rotation
            desired[:3, 3] = target_joints[a]
        if (
            preserve_same_semantic_offset
            and parent >= 0
            and int(asset.source_bone_smplx_a[parent]) == a
        ):
            # Several authored helper/twist controllers share one SMPL-X
            # semantic joint with their parent but occupy distinct points
            # along the physical segment.  Collapsing every such controller
            # onto target_joints[a] creates zero local translations and makes
            # full-local FK fold the tibia, forearm and ankle chains.
            desired[:3, 3] = (
                new_global[parent, :3, :3] @ old_local[bone, :3, 3]
                + new_global[parent, :3, 3]
            )
        if parent < 0:
            new_global[bone] = desired
        else:
            local = np.linalg.inv(new_global[parent]) @ desired
            new_global[bone] = new_global[parent] @ local

    # Retarget the *entire* authored pelvis-to-head chain, including discs.
    # The previous four independent vertebra-only spans left their disc parents
    # behind and even reversed Disc42/Spine_L1.  Use the actual parent path and
    # authored cumulative arc length so every element remains ordered.
    joint_id = {name: asset.joint_names.index(name) for name in asset.joint_names}
    source_names = list(asset.source_bone_names or [])
    spine_chain: list[int] = []
    if "Hip_bone" in source_names and "Head_Bone" in source_names:
        current = source_names.index("Head_Bone")
        while current >= 0:
            spine_chain.append(current)
            if current == source_names.index("Hip_bone"):
                break
            current = int(source_parents[current])
        spine_chain.reverse()
    anchors = (
        ("Hip_bone", "pelvis"),
        ("Spine_L5", "spine1"),
        ("Spine_L2", "spine2"),
        ("Spine_T8", "spine3"),
        ("Spine_C7", "neck"),
        ("Head_Bone", "head"),
    )
    anchor_positions: list[tuple[int, np.ndarray]] = []
    for bone_name, joint_name in anchors:
        if bone_name in source_names and joint_name in joint_id:
            anchor_positions.append((source_names.index(bone_name), target_joints[joint_id[joint_name]]))
    chain_set = set(spine_chain)
    for (start_bone, start_target), (stop_bone, stop_target) in zip(anchor_positions, anchor_positions[1:]):
        try:
            start_at = spine_chain.index(start_bone)
            stop_at = spine_chain.index(stop_bone)
        except ValueError:
            continue
        if stop_at <= start_at:
            continue
        segment = spine_chain[start_at : stop_at + 1]
        authored = old_global[segment, :3, 3]
        lengths = np.linalg.norm(np.diff(authored, axis=0), axis=1)
        fractions = np.r_[0.0, np.cumsum(lengths)]
        fractions /= max(float(fractions[-1]), 1.0e-8)
        rotation = _rotation_between(authored[-1] - authored[0], stop_target - start_target)
        for fraction, bone in zip(fractions.tolist(), segment):
            new_global[bone, :3, :3] = rotation @ old_global[bone, :3, :3]
            new_global[bone, :3, 3] = (1.0 - fraction) * start_target + fraction * stop_target

    # Each rib pair belongs to its authored thoracic level.  Retain the exact
    # rib-to-vertebra bind offset instead of collapsing every rib at spine2.
    rib_bones: set[int] = set()
    for bone, name in enumerate(source_names):
        match = re.fullmatch(r"Rib_(?:Bone|Name)_[LR](\d+)", name)
        if match is None:
            continue
        level_name = f"Spine_T{int(match.group(1))}"
        if level_name not in source_names:
            continue
        level = source_names.index(level_name)
        new_global[bone] = new_global[level] @ np.linalg.inv(old_global[level]) @ old_global[bone]
        rib_bones.add(bone)

    # Re-evaluate authored helper descendants after the spine/rib overrides.
    # Chain elements and rib roots are fixed roots for this pass.
    for bone, mode in enumerate(modes):
        parent = int(source_parents[bone])
        if bone in chain_set or bone in rib_bones or parent < 0:
            continue
        if _is_follow_mode(mode):
            new_global[bone] = new_global[parent] @ old_local[bone]
    new_local = new_global.copy()
    for bone, parent in enumerate(source_parents.tolist()):
        if int(parent) >= 0:
            new_local[bone] = np.linalg.inv(new_global[int(parent)]) @ new_global[bone]
    delta = new_global @ np.linalg.inv(old_global)
    return new_global, new_local, delta


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def _aspect_ratio_change(source: np.ndarray, fitted: np.ndarray) -> float:
    # Singular-value extents are invariant to the rigid frame rotation applied
    # to pelvis/head compounds.  A world-axis AABB falsely reported a 15.8%
    # pelvis shape change for a mathematically uniform similarity transform.
    source_centered = np.asarray(source, dtype=np.float64) - np.mean(source, axis=0)
    fitted_centered = np.asarray(fitted, dtype=np.float64) - np.mean(fitted, axis=0)
    source_extent = np.sort(np.linalg.svd(source_centered, compute_uv=False))
    fitted_extent = np.sort(np.linalg.svd(fitted_centered, compute_uv=False))
    source_ratio = source_extent / max(float(source_extent[-1]), 1.0e-8)
    fitted_ratio = fitted_extent / max(float(fitted_extent[-1]), 1.0e-8)
    return float(np.max(np.abs(fitted_ratio - source_ratio) / np.maximum(source_ratio, 1.0e-8)))


def _robust_sphere_center(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        return np.mean(pts, axis=0) if len(pts) else np.zeros(3)
    centered = pts - np.mean(pts, axis=0)
    radii = np.linalg.norm(centered, axis=1)
    median = float(np.median(radii))
    keep = radii <= max(2.5 * median, 1.0e-4)
    if np.count_nonzero(keep) >= 8:
        pts = pts[keep]
    system = np.concatenate((2.0 * pts, np.ones((len(pts), 1))), axis=1)
    rhs = np.sum(pts * pts, axis=1)
    try:
        solution, *_unused = np.linalg.lstsq(system, rhs, rcond=None)
        return solution[:3]
    except np.linalg.LinAlgError:
        return np.mean(pts, axis=0)


def _femur_head_and_acetabulum(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    side: str,
    target_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit hip landmarks from bone surfaces inside the SMPL-X search region."""
    suffix = "_l" if side == "left" else "_r"
    hip = target_joints[asset.joint_names.index(f"{side}_hip")]
    knee = target_joints[asset.joint_names.index(f"{side}_knee")]
    axis = hip - knee
    axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
    femur = _mesh_mask(
        asset,
        lambda name, tissue, suffix=suffix: tissue == "bone"
        and "femur" in name
        and (name.endswith(suffix) or f"{suffix}_" in name),
    )
    pelvis = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone"
        and any(token in name for token in ("ilium", "ischium", "pubis", "acetabul", "pelvis")),
    )
    if not np.any(femur) or not np.any(pelvis):
        return None
    femur_points = np.asarray(vertices[femur], dtype=np.float64)
    parameter = (femur_points - knee) @ axis
    head_points = femur_points[parameter >= np.quantile(parameter, 0.85)]
    head = _robust_sphere_center(head_points)
    pelvis_points = np.asarray(vertices[pelvis], dtype=np.float64)
    distance = np.linalg.norm(pelvis_points - hip, axis=1)
    near = pelvis_points[distance <= max(float(np.quantile(distance, 0.08)), 0.025)]
    if len(near) < 8:
        near = pelvis_points[np.argsort(distance)[: min(32, len(pelvis_points))]]
    acetabulum = 0.70 * np.mean(near, axis=0) + 0.30 * hip
    return head, acetabulum


def _protected_end_edge_change(
    asset: AnatomyRiggedAsset,
    *,
    start: int,
    stop: int,
    source: np.ndarray,
    fitted: np.ndarray,
    source_a: np.ndarray,
    source_b: np.ndarray,
) -> float:
    faces = np.asarray(asset.faces, dtype=np.int64)
    local = faces[np.all((faces >= int(start)) & (faces < int(stop)), axis=1)] - int(start)
    if not len(local):
        return 0.0
    edges = np.unique(
        np.sort(np.concatenate((local[:, [0, 1]], local[:, [1, 2]], local[:, [2, 0]])), axis=1),
        axis=0,
    )
    axis = np.asarray(source_b, dtype=np.float64) - np.asarray(source_a, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-8:
        return 0.0
    axis /= length
    parameter = np.clip((np.asarray(source) - source_a) @ axis / length, 0.0, 1.0)
    protected = ((parameter[edges[:, 0]] <= 0.2) & (parameter[edges[:, 1]] <= 0.2)) | (
        (parameter[edges[:, 0]] >= 0.8) & (parameter[edges[:, 1]] >= 0.8)
    )
    edges = edges[protected]
    if not len(edges):
        return 0.0
    before = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    after = np.linalg.norm(fitted[edges[:, 0]] - fitted[edges[:, 1]], axis=1)
    valid = before > 1.0e-8
    return float(np.max(np.abs(after[valid] / before[valid] - 1.0))) if np.any(valid) else 0.0


def _mesh_mask(asset: AnatomyRiggedAsset, predicate) -> np.ndarray:
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    if asset.source_vertex_ranges is None:
        return mask
    tissues = list(
        getattr(asset, "source_tissues", None)
        or [""] * len(asset.source_mesh_names)
    )
    for (start, stop), name, tissue in zip(asset.source_vertex_ranges, asset.source_mesh_names, tissues):
        if predicate(str(name).lower(), str(tissue).lower()):
            mask[int(start) : int(stop)] = True
    return mask


def cranial_material_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    """All Head_Bone material except the independently articulated jaw.

    Object names are not a reliable anatomical hierarchy: upper teeth and a
    number of intracranial structures have generic names.  Their Blender
    controller is the source of truth, so use the exported rig hierarchy when
    it is available and retain the token fallback only for tiny test assets.
    """
    names = list(asset.source_bone_names or [])
    if not names or asset.source_vertex_ranges is None:
        return _mesh_mask(asset, lambda name, _tissue: any(token in name for token in _CRANIAL_TOKENS))
    try:
        head = names.index("Head_Bone")
    except ValueError:
        return _mesh_mask(asset, lambda name, _tissue: any(token in name for token in _CRANIAL_TOKENS))
    jaw = names.index("Jaw_Bone_tip") if "Jaw_Bone_tip" in names else -1
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)

    def descends_from(bone: int, ancestor: int) -> bool:
        while bone >= 0:
            if bone == ancestor:
                return True
            bone = int(parents[bone])
        return False

    if asset.driver_indices is None or asset.driver_weights is None:
        return np.zeros(len(asset.vertices_rest), dtype=bool)
    cranial_bone = np.asarray(
        [
            descends_from(bone, head)
            and not (jaw >= 0 and descends_from(bone, jaw))
            for bone in range(len(names))
        ],
        dtype=bool,
    )
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    return np.sum(weights * cranial_bone[indices], axis=1) >= 0.5


def jaw_material_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    """Meshes driven by the authored jaw subtree, including lower teeth."""
    names = list(asset.source_bone_names or [])
    if "Jaw_Bone_tip" not in names or asset.source_vertex_ranges is None:
        return _mesh_mask(asset, lambda name, _tissue: "mandible" in name)
    jaw = names.index("Jaw_Bone_tip")
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    if asset.driver_indices is None or asset.driver_weights is None:
        return np.zeros(len(asset.vertices_rest), dtype=bool)

    def descends_from_jaw(bone: int) -> bool:
        while bone >= 0:
            if bone == jaw:
                return True
            bone = int(parents[bone])
        return False

    jaw_bone = np.asarray(
        [descends_from_jaw(bone) for bone in range(len(names))], dtype=bool
    )
    indices = np.asarray(asset.driver_indices, dtype=np.int64)
    weights = np.asarray(asset.driver_weights, dtype=np.float64)
    return np.sum(weights * jaw_bone[indices], axis=1) >= 0.5


def bone_material_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    return _mesh_mask(asset, lambda _name, tissue: tissue == "bone")


def _surface_region(
    canonical_dir: Path,
    joint_names: list[str],
    names: tuple[str, ...],
    *,
    subject: bool,
) -> np.ndarray:
    weights = np.load(canonical_dir / "smpl_canonical_weights.npz", allow_pickle=True)
    surface = _load_obj_vertices(
        canonical_dir / ("smpl_canonical_tpose.obj" if subject else "smpl_canonical_tpose_neutral.obj")
    )
    ids = [joint_names.index(name) for name in names if name in joint_names]
    if not ids:
        return surface
    mass = np.asarray(weights["lbs_weights"], dtype=np.float64)[:, ids].sum(axis=1)
    threshold = max(0.15, float(np.quantile(mass[mass > 0.0], 0.35))) if np.any(mass > 0.0) else 0.15
    selected = surface[mass >= threshold]
    return selected if len(selected) >= 32 else surface[np.argsort(-mass)[: max(32, len(surface) // 50)]]


def _uniform_envelope_fit(
    points: np.ndarray,
    target: np.ndarray,
    *,
    reference_points: np.ndarray | None = None,
    scale_multiplier: float,
    center_offset: np.ndarray,
    margin: float,
    maximum_scale: float = 1.5,
    minimum_scale: float = 0.5,
    scale_mode: str = "median",
    source_center: np.ndarray | None = None,
    target_center: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, float | bool | str]]:
    source = np.asarray(points, dtype=np.float64)
    reference = np.asarray(
        source if reference_points is None else reference_points, dtype=np.float64
    )
    destination = np.asarray(target, dtype=np.float64)
    source_lo, source_hi = np.quantile(reference, (0.01, 0.99), axis=0)
    target_lo, target_hi = np.quantile(destination, (0.01, 0.99), axis=0)
    resolved_source_center = (
        np.asarray(source_center, dtype=np.float64).reshape(3)
        if source_center is not None
        else 0.5 * (source_lo + source_hi)
    )
    resolved_target_center = (
        np.asarray(target_center, dtype=np.float64).reshape(3)
        if target_center is not None
        else 0.5 * (target_lo + target_hi) + np.asarray(center_offset, dtype=np.float64)
    )
    source_extent = 0.5 * (source_hi - source_lo)
    target_extent = 0.5 * (target_hi - target_lo)
    valid = source_extent > 1.0e-5
    ratios = target_extent[valid] / source_extent[valid] if np.any(valid) else np.asarray([1.0])
    mode = str(scale_mode).strip().lower()
    if mode == "min":
        base_scale = float(np.min(ratios))
    elif mode == "mean":
        base_scale = float(np.mean(ratios))
    else:
        mode = "median"
        base_scale = float(np.median(ratios))
    raw_scale = float(margin) * base_scale * float(scale_multiplier)
    scale = float(np.clip(raw_scale, float(minimum_scale), float(maximum_scale)))
    return resolved_target_center + scale * (source - resolved_source_center), scale, {
        "base_scale": base_scale,
        "raw_scale": raw_scale,
        "scale": scale,
        "saturated": bool(scale != raw_scale),
        "scale_mode": mode,
    }


def _contained_uniform_candidate(
    source: np.ndarray,
    *,
    source_center: np.ndarray,
    target_center: np.ndarray,
    target_surface: np.ndarray,
    target_faces: np.ndarray,
    desired_scale: float,
    minimum_scale: float,
    clearance_m: float = 0.0005,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Choose a contained uniform scale; never project individual vertices."""
    import igl

    points = np.asarray(source, dtype=np.float64)
    source_origin = np.asarray(source_center, dtype=np.float64).reshape(3)
    target_origin = np.asarray(target_center, dtype=np.float64).reshape(3)

    def candidate(scale: float) -> tuple[np.ndarray, float]:
        value = target_origin + float(scale) * (points - source_origin)
        signed, _face, _closest, _normal = igl.signed_distance(
            value,
            np.asarray(target_surface, dtype=np.float64),
            np.asarray(target_faces, dtype=np.int32),
        )
        return value, float(np.max(np.asarray(signed, dtype=np.float64) + clearance_m))

    desired_vertices, desired_violation = candidate(float(desired_scale))
    if desired_violation <= 0.0:
        return desired_vertices, float(desired_scale), {
            "candidate_contained": True,
            "desired_scale": float(desired_scale),
            "contained_scale": float(desired_scale),
            "maximum_clearance_violation_m": desired_violation,
        }
    low = float(minimum_scale)
    low_vertices, low_violation = candidate(low)
    if low_violation > 0.0:
        return desired_vertices, float(desired_scale), {
            "candidate_contained": False,
            "desired_scale": float(desired_scale),
            "contained_scale": None,
            "minimum_scale": low,
            "minimum_scale_violation_m": low_violation,
            "maximum_clearance_violation_m": desired_violation,
        }
    high = float(desired_scale)
    best_vertices = low_vertices
    best_scale = low
    best_violation = low_violation
    for _ in range(20):
        mid = 0.5 * (low + high)
        mid_vertices, violation = candidate(mid)
        if violation <= 0.0:
            low = mid
            best_vertices = mid_vertices
            best_scale = mid
            best_violation = violation
        else:
            high = mid
    return best_vertices, float(best_scale), {
        "candidate_contained": True,
        "desired_scale": float(desired_scale),
        "contained_scale": float(best_scale),
        "maximum_clearance_violation_m": float(best_violation),
    }


def _sample_spine_centerline(
    control_points: np.ndarray,
    fractions: np.ndarray,
    *,
    control_fractions: np.ndarray | None = None,
) -> np.ndarray:
    """Sample an order-preserving C1 curve through anatomical anchors."""
    controls = np.asarray(control_points, dtype=np.float64).reshape(-1, 3)
    query = np.clip(np.asarray(fractions, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if not len(controls):
        raise ValueError("spine centerline requires control points")
    if len(controls) == 1:
        return np.repeat(controls, len(query), axis=0)
    if len(controls) == 2:
        return (1.0 - query)[:, None] * controls[0] + query[:, None] * controls[1]
    if control_fractions is None:
        chord = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        param = np.r_[0.0, np.cumsum(np.maximum(chord, 1.0e-8))]
    else:
        param = np.asarray(control_fractions, dtype=np.float64).reshape(-1)
        if len(param) != len(controls):
            raise ValueError("control_fractions must match control_points")
        if np.any(~np.isfinite(param)) or np.any(np.diff(param) <= 0.0):
            raise ValueError("control_fractions must be finite and strictly increasing")
        param = param - param[0]
    param /= max(float(param[-1]), 1.0e-8)
    try:
        from scipy.interpolate import PchipInterpolator

        return np.asarray(PchipInterpolator(param, controls, axis=0)(query), dtype=np.float64)
    except Exception:
        return np.stack([np.interp(query, param, controls[:, axis]) for axis in range(3)], axis=1)


def _fit_final_spine_interfaces(
    asset: AnatomyRiggedAsset,
    *,
    old_global: np.ndarray,
    new_global: np.ndarray,
    pelvis_l5_interface: np.ndarray,
    skull_c1_interface: np.ndarray,
    target_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refit the authored L5--C1 path to a monotone C1 anatomical curve."""
    names = list(asset.source_bone_names or [])
    required = ("Spine_L5", "Spine_C1")
    if not all(name in names for name in required):
        return new_global, np.asarray(asset.target_bind_local), {
            "available": False,
            "reason": "l5_or_c1_missing",
        }
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    start = names.index("Spine_L5")
    stop = names.index("Spine_C1")
    chain: list[int] = []
    current = stop
    while current >= 0:
        chain.append(current)
        if current == start:
            break
        current = int(parents[current])
    if not chain or chain[-1] != start:
        return new_global, np.asarray(asset.target_bind_local), {
            "available": False,
            "reason": "authored_path_missing",
        }
    chain.reverse()
    authored = np.asarray(old_global, dtype=np.float64)[chain, :3, 3]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(authored, axis=0), axis=1))]
    fractions = arc / max(float(arc[-1]), 1.0e-8)
    joint_id = {name: index for index, name in enumerate(asset.joint_names)}
    control_points = [np.asarray(pelvis_l5_interface, dtype=np.float64)]
    control_fractions = [0.0]
    for fraction, joint_name in ((0.30, "spine2"), (0.62, "spine3"), (0.86, "neck")):
        if joint_name in joint_id:
            control_fractions.append(float(fraction))
            control_points.append(np.asarray(target_joints[joint_id[joint_name]], dtype=np.float64))
    control_fractions.append(1.0)
    control_points.append(np.asarray(skull_c1_interface, dtype=np.float64))
    controls = np.asarray(control_points, dtype=np.float64)
    # Guard against a noisy SMPL-X control point reversing the cranio-caudal
    # order. Project controls monotonically along the hard endpoint axis while
    # retaining their transverse curvature.
    axis = controls[-1] - controls[0]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-10)
    scalar = (controls - controls[0]) @ axis
    total = max(float(scalar[-1]), 1.0e-8)
    minimum_step = total * 1.0e-4
    scalar = np.maximum.accumulate(scalar + minimum_step * np.arange(len(scalar)))
    scalar *= total / max(float(scalar[-1]), 1.0e-8)
    controls += (scalar - (controls - controls[0]) @ axis)[:, None] * axis
    centerline = _sample_spine_centerline(
        controls,
        fractions,
        control_fractions=np.asarray(control_fractions, dtype=np.float64),
    )
    fitted = np.asarray(new_global, dtype=np.float64).copy()
    for position, bone in enumerate(chain):
        lo = max(0, position - 1)
        hi = min(len(chain) - 1, position + 1)
        authored_tangent = authored[hi] - authored[lo]
        fitted_tangent = centerline[hi] - centerline[lo]
        fitted[bone, :3, :3] = (
            _rotation_between(authored_tangent, fitted_tangent)
            @ np.asarray(old_global[bone, :3, :3], dtype=np.float64)
        )
        fitted[bone, :3, 3] = centerline[position]
    local = fitted.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            local[bone] = np.linalg.inv(fitted[int(parent)]) @ fitted[bone]
    projected = (centerline - centerline[0]) @ axis
    gaps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    return fitted, local, {
        "available": True,
        "chain_bones": int(len(chain)),
        "monotonic": bool(np.all(np.diff(projected) > 0.0)),
        "minimum_center_gap_m": float(np.min(gaps)) if len(gaps) else 0.0,
        "pelvis_l5_gap_m": 0.0,
        "skull_c1_gap_m": 0.0,
    }


def _midline_envelope_centers(
    *,
    reference_points: np.ndarray,
    target_points: np.ndarray,
    source_anchors: np.ndarray,
    target_joints: np.ndarray,
    joint_names: list[str],
    center_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Center cranial envelopes on the eye midline for left-right alignment."""
    left_name, right_name = "left_eye_smplhf", "right_eye_smplhf"
    if left_name not in joint_names or right_name not in joint_names:
        source_lo, source_hi = np.quantile(reference_points, (0.01, 0.99), axis=0)
        target_lo, target_hi = np.quantile(target_points, (0.01, 0.99), axis=0)
        return 0.5 * (source_lo + source_hi), 0.5 * (target_lo + target_hi) + center_offset
    left_id = joint_names.index(left_name)
    right_id = joint_names.index(right_name)
    source_lateral = source_anchors[right_id] - source_anchors[left_id]
    target_lateral = target_joints[right_id] - target_joints[left_id]
    source_lateral /= max(float(np.linalg.norm(source_lateral)), 1.0e-8)
    target_lateral /= max(float(np.linalg.norm(target_lateral)), 1.0e-8)
    source_eye_mid = 0.5 * (source_anchors[left_id] + source_anchors[right_id])
    target_eye_mid = 0.5 * (target_joints[left_id] + target_joints[right_id])
    source_lo, source_hi = np.quantile(reference_points, (0.01, 0.99), axis=0)
    target_lo, target_hi = np.quantile(target_points, (0.01, 0.99), axis=0)
    source_aabb = 0.5 * (source_lo + source_hi)
    target_aabb = 0.5 * (target_lo + target_hi) + center_offset
    source_center = source_aabb + (
        float((source_eye_mid - source_aabb) @ source_lateral) * source_lateral
    )
    target_center = target_aabb + (
        float((target_eye_mid - target_aabb) @ target_lateral) * target_lateral
    )
    return source_center, target_center


def _transport_soft_material(
    vertices: np.ndarray,
    old_vertices: np.ndarray,
    soft_material: np.ndarray,
    *,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    bone_delta: np.ndarray,
) -> None:
    if not np.any(soft_material):
        return
    indices = np.asarray(driver_indices, dtype=np.int64)[soft_material]
    weights = np.asarray(driver_weights, dtype=np.float64)[soft_material]
    transforms = bone_delta[indices]
    blended = np.sum(transforms * weights[..., None, None], axis=1)
    homogeneous = np.concatenate(
        (old_vertices[soft_material], np.ones((int(np.count_nonzero(soft_material)), 1))),
        axis=1,
    )
    vertices[soft_material] = np.matmul(blended, homogeneous[..., None])[:, :3, 0]


def _soft_material_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue).lower() != "bone":
            mask[int(start) : int(stop)] = True
    return mask


def _tissue_mask(asset: AnatomyRiggedAsset, *tissues: str) -> np.ndarray:
    wanted = {str(t).lower() for t in tissues}
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue).lower() in wanted:
            mask[int(start) : int(stop)] = True
    return mask


def _nearest_skeleton_segment(
    points: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    batch_size: int = 50000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    children = np.flatnonzero(np.asarray(parents, dtype=np.int64) >= 0)
    starts = joints[parents[children]]
    vectors = joints[children] - starts
    length2 = np.einsum("ij,ij->i", vectors, vectors)
    valid = length2 > 1.0e-10
    children, starts, vectors, length2 = (
        children[valid],
        starts[valid],
        vectors[valid],
        length2[valid],
    )
    assignment = np.empty(len(points), dtype=np.int32)
    centers = np.empty_like(points, dtype=np.float64)
    for begin in range(0, len(points), int(batch_size)):
        end = min(len(points), begin + int(batch_size))
        query = np.asarray(points[begin:end], dtype=np.float64)
        rel = query[:, None, :] - starts[None, :, :]
        parameter = np.clip(
            np.einsum("nsi,si->ns", rel, vectors) / length2[None, :], 0.0, 1.0
        )
        projected = starts[None, :, :] + parameter[:, :, None] * vectors[None, :, :]
        distance2 = np.sum((query[:, None, :] - projected) ** 2, axis=2)
        selected = np.argmin(distance2, axis=1)
        rows = np.arange(len(query))
        assignment[begin:end] = selected.astype(np.int32)
        centers[begin:end] = projected[rows, selected]
    return assignment, centers, children


def _smooth_material_displacement(
    desired: np.ndarray, faces: np.ndarray, *, iterations: int = 30
) -> np.ndarray:
    from scipy.sparse import coo_matrix, diags

    triangles = np.asarray(faces, dtype=np.int64)
    if not len(triangles):
        return np.asarray(desired, dtype=np.float64)
    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(desired), len(desired)),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    average = diags(1.0 / np.maximum(degree, 1.0)) @ adjacency
    target = np.asarray(desired, dtype=np.float64)
    output = target.copy()
    for _ in range(int(iterations)):
        output = 0.1 * target + 0.9 * (average @ output)
    return output


def _follow_foot_soft_scale(
    vertices: np.ndarray,
    asset: AnatomyRiggedAsset,
    *,
    ankle: np.ndarray,
    forward: np.ndarray,
    scale: float,
    rigid_offset: np.ndarray,
) -> dict[str, float]:
    """Apply ankle-centered *axial* foot scale to local vessel/nerve (width preserved)."""
    soft = _tissue_mask(asset, "vessel", "nerve")
    if not np.any(soft):
        return {"soft_vertices": 0, "axial_scale": float(scale)}
    ankle = np.asarray(ankle, dtype=np.float64).reshape(3)
    forward = np.asarray(forward, dtype=np.float64).reshape(3)
    forward /= max(float(np.linalg.norm(forward)), 1.0e-8)
    offset = np.asarray(rigid_offset, dtype=np.float64).reshape(3)
    relative = vertices[soft] - ankle
    axial = relative @ forward
    radial = np.linalg.norm(relative - axial[:, None] * forward, axis=1)
    local = (axial > -0.03) & (axial < 0.50) & (radial < 0.14)
    if not np.any(local):
        return {"soft_vertices": 0, "axial_scale": float(scale)}
    soft_idx = np.flatnonzero(soft)[local]
    weight = np.clip((axial[local] + 0.03) / 0.08, 0.0, 1.0)
    effective = 1.0 + weight * (float(scale) - 1.0)
    rel = vertices[soft_idx] - ankle
    ax = (rel @ forward)[:, None] * forward
    rad = rel - ax
    vertices[soft_idx] = ankle + effective[:, None] * ax + rad
    vertices[soft_idx] += offset[None, :] * weight[:, None]
    return {
        "soft_vertices": int(len(soft_idx)),
        "axial_scale": float(scale),
        "mean_blend": float(np.mean(weight)),
    }


def _axial_scale_about_axis(
    points: np.ndarray,
    *,
    origin: np.ndarray,
    axis: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Scale only the component along ``axis``; keep transverse offsets."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
    rel = np.asarray(points, dtype=np.float64) - origin
    axial = (rel @ axis)[:, None] * axis
    return origin + float(scale) * axial + (rel - axial)


def _bone_mesh_centroids(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    bone_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-source-bone mesh centroids (optionally after a rigid bone_delta)."""
    n_bones = len(asset.source_bone_names or [])
    centroids = np.zeros((n_bones, 3), dtype=np.float64)
    counts = np.zeros(n_bones, dtype=np.int64)
    if asset.source_vertex_ranges is None or asset.source_tissues is None or n_bones == 0:
        return centroids, counts > 0
    verts = np.asarray(vertices, dtype=np.float64)
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue).lower() != "bone":
            continue
        start_i, stop_i = int(start), int(stop)
        bone = _dominant_bone(asset, start_i, stop_i)
        if bone is None:
            continue
        block = verts[start_i:stop_i]
        if bone_delta is not None:
            block = _transform_points(block, bone_delta[int(bone)])
        centroids[int(bone)] += block.sum(axis=0)
        counts[int(bone)] += int(block.shape[0])
    valid = counts > 0
    centroids[valid] /= counts[valid, None]
    return centroids, valid


def _skin_depth_attenuation(
    points: np.ndarray,
    *,
    subject_surface: np.ndarray,
    subject_faces: np.ndarray,
    near_m: float = 0.004,
    far_m: float = 0.028,
) -> np.ndarray:
    """1 deep inside, 0 near/outside skin. SDF used only as a weight, never a push."""
    import igl

    signed, _face_index, _closest, _normal = igl.signed_distance(
        np.asarray(points, dtype=np.float64), subject_surface, subject_faces
    )
    depth = np.maximum(0.0, -np.asarray(signed, dtype=np.float64))
    span = max(float(far_m) - float(near_m), 1.0e-6)
    return np.clip((depth - float(near_m)) / span, 0.0, 1.0)


def _apply_soft_bone_translation_field(
    vertices: np.ndarray,
    asset: AnatomyRiggedAsset,
    soft_mask: np.ndarray,
    *,
    blender_centroids: np.ndarray,
    material_centroids: np.ndarray,
    valid_bones: np.ndarray,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    subject_surface: np.ndarray,
    subject_faces: np.ndarray,
) -> dict[str, float]:
    """Push soft tissue by residual bone translation (Blender linkage → material).

    Translation-only: keeps vessel topology. Near-skin attenuation prevents the
    field from dragging subcutaneous vessels through the envelope. No SDF push.
    """
    if not np.any(soft_mask):
        return {"soft_vertices": 0, "active_bones": 0}
    residual = np.asarray(material_centroids, dtype=np.float64) - np.asarray(
        blender_centroids, dtype=np.float64
    )
    residual[~np.asarray(valid_bones, dtype=bool)] = 0.0
    active = int(np.count_nonzero(np.asarray(valid_bones, dtype=bool) & (np.linalg.norm(residual, axis=1) > 1.0e-7)))
    if active == 0:
        return {"soft_vertices": int(np.count_nonzero(soft_mask)), "active_bones": 0}

    idx = np.flatnonzero(soft_mask)
    indices = np.asarray(driver_indices, dtype=np.int64)[idx]
    weights = np.asarray(driver_weights, dtype=np.float64)[idx]
    weight_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(weight_sum, 1.0e-12)
    desired = np.sum(residual[indices] * weights[..., None], axis=1)

    alpha = _skin_depth_attenuation(
        vertices[idx],
        subject_surface=subject_surface,
        subject_faces=subject_faces,
    )
    desired *= alpha[:, None]

    # Smooth per soft mesh so thin tubes do not shear at weight seams.
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    if ranges is not None and tissues is not None:
        for (start, stop), tissue in zip(ranges, tissues):
            if str(tissue).lower() == "bone":
                continue
            start_i, stop_i = int(start), int(stop)
            local_soft = soft_mask[start_i:stop_i]
            if not np.any(local_soft):
                continue
            local_faces = all_faces[
                (all_faces[:, 0] >= start_i)
                & (all_faces[:, 0] < stop_i)
                & (all_faces[:, 1] >= start_i)
                & (all_faces[:, 1] < stop_i)
                & (all_faces[:, 2] >= start_i)
                & (all_faces[:, 2] < stop_i)
            ]
            if not len(local_faces):
                continue
            local_faces = local_faces - start_i
            block = np.zeros((stop_i - start_i, 3), dtype=np.float64)
            # Map global soft indices into this mesh block.
            global_in_mesh = idx[(idx >= start_i) & (idx < stop_i)]
            if not len(global_in_mesh):
                continue
            block[global_in_mesh - start_i] = desired[np.searchsorted(idx, global_in_mesh)]
            smoothed = _smooth_material_displacement(block, local_faces, iterations=12)
            vertices[global_in_mesh] += smoothed[global_in_mesh - start_i]
    else:
        vertices[idx] += desired

    disp = np.linalg.norm(desired, axis=1)
    return {
        "soft_vertices": int(len(idx)),
        "active_bones": active,
        "mean_translation_m": float(np.mean(disp)),
        "max_translation_m": float(np.max(disp)) if len(disp) else 0.0,
        "mean_skin_attenuation": float(np.mean(alpha)),
    }


def _soft_section_inward_scale(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    subject_surface: np.ndarray,
    subject_faces: np.ndarray,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """0a21d7b-style section radial shrink for soft tissue only (bones untouched).

    Scales are estimated from soft points versus the subject SMPL surface, then
    applied uniformly within each skeleton segment so vessel topology is kept.
    """
    import igl

    soft = _soft_material_mask(asset)
    if exclude is not None:
        soft = soft & ~np.asarray(exclude, dtype=bool)
    if not np.any(soft):
        return vertices, {"minimum_section_scale": 1.0, "soft_vertices": 0}

    joints = np.asarray(asset.rest_joints, dtype=np.float64)
    parents = np.asarray(asset.parents, dtype=np.int64)
    output = np.asarray(vertices, dtype=np.float64).copy()
    assignment, centers, children = _nearest_skeleton_segment(output, joints, parents)
    soft_points = output[soft]
    _signed, _face_index, closest, _normal = igl.signed_distance(
        soft_points, subject_surface, subject_faces
    )
    soft_centers = centers[soft]
    source_radius = np.linalg.norm(soft_points - soft_centers, axis=1)
    target_radius = np.linalg.norm(np.asarray(closest) - soft_centers, axis=1)
    # Only trust near-surface / outside samples for inward shrink.
    usable = source_radius > 1.0e-4
    ratios = np.ones(len(soft_points), dtype=np.float64)
    ratios[usable] = np.clip(
        target_radius[usable] / source_radius[usable], 0.70, 1.0
    )

    scales = np.ones(len(children), dtype=np.float64)
    soft_assignment = assignment[soft]
    for segment in np.unique(soft_assignment):
        local = ratios[soft_assignment == segment]
        if len(local) >= 8:
            scales[int(segment)] = min(1.0, float(np.quantile(local, 0.05)))

    child_to_segment = {int(child): idx for idx, child in enumerate(children.tolist())}
    for _ in range(3):
        previous = scales.copy()
        for idx, child in enumerate(children.tolist()):
            neighbours = [idx]
            parent = int(parents[child])
            if parent in child_to_segment:
                neighbours.append(child_to_segment[parent])
            neighbours.extend(
                child_to_segment[int(other)]
                for other in children
                if int(parents[int(other)]) == int(child)
            )
            scales[idx] = min(previous[idx], float(np.mean(previous[neighbours])))

    local_scale = scales[assignment]
    desired = (local_scale[:, None] - 1.0) * (output - centers)
    ranges = asset.source_vertex_ranges
    tissues = asset.source_tissues
    if ranges is None or tissues is None:
        output[~soft] = vertices[~soft]
        return output, {"minimum_section_scale": 1.0, "soft_vertices": int(np.count_nonzero(soft))}
    all_faces = np.asarray(asset.faces, dtype=np.int64)
    for (start, stop), tissue in zip(ranges, tissues):
        if str(tissue).lower() == "bone":
            continue
        start_i, stop_i = int(start), int(stop)
        range_soft = soft[start_i:stop_i]
        if not np.any(range_soft):
            continue
        local_faces = all_faces[
            (all_faces[:, 0] >= start_i)
            & (all_faces[:, 0] < stop_i)
            & (all_faces[:, 1] >= start_i)
            & (all_faces[:, 1] < stop_i)
            & (all_faces[:, 2] >= start_i)
            & (all_faces[:, 2] < stop_i)
        ] - start_i
        delta = _smooth_material_displacement(desired[start_i:stop_i], local_faces)
        moved = output[start_i:stop_i] + delta
        block = output[start_i:stop_i].copy()
        block[range_soft] = moved[range_soft]
        output[start_i:stop_i] = block

    output[~soft] = vertices[~soft]
    displacement = np.linalg.norm(output[soft] - vertices[soft], axis=1)
    return output, {
        "minimum_section_scale": float(np.min(scales)) if len(scales) else 1.0,
        "mean_soft_displacement_m": float(np.mean(displacement)),
        "max_soft_displacement_m": float(np.max(displacement)) if len(displacement) else 0.0,
        "soft_vertices": int(np.count_nonzero(soft)),
    }


def _hand_mesh_segment(
    name: str,
    *,
    joint_names: list[str],
    source_anchors: np.ndarray,
    target_joints: np.ndarray,
    finger_tips: dict[tuple[str, str], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve every authored hand bone to its own SMPL-X finger segment."""
    lower = str(name).lower()
    if not any(token in lower for token in ("metacarpal", "phalanx_hand", "phalanges_hand")):
        return None
    side = "left" if lower.endswith("_l") or "_hand_l" in lower else "right" if lower.endswith("_r") or "_hand_r" in lower else None
    digit_match = re.search(r"(?:^|_)([1-5])(?:st|nd|rd|th)?_", lower)
    if side is None or digit_match is None:
        return None
    finger = {1: "thumb", 2: "index", 3: "middle", 4: "ring", 5: "pinky"}[int(digit_match.group(1))]
    if "metacarpal" in lower:
        if finger == "thumb":
            # SMPL-X thumb1 is the CMC pivot.  The first metacarpal lies
            # distal to it and is driven by thumb1 in the Blender source rig;
            # fitting it wrist->thumb1 puts the whole mesh proximal to its
            # runtime pivot and makes it sweep outside the palm when flexed.
            a_name, b_name = f"{side}_thumb1", f"{side}_thumb2"
        else:
            a_name, b_name = f"{side}_wrist", f"{side}_{finger}1"
    elif "proximal" in lower:
        if finger == "thumb":
            a_name, b_name = f"{side}_thumb2", f"{side}_thumb3"
        else:
            a_name, b_name = f"{side}_{finger}1", f"{side}_{finger}2"
    elif "intermediate" in lower or "middle" in lower:
        a_name, b_name = f"{side}_{finger}2", f"{side}_{finger}3"
    elif "distal" in lower:
        # SMPL-X has three finger joints and no fingertip joint.  Continue the
        # last authored segment direction without inventing a shared hand scale.
        j2 = joint_names.index(f"{side}_{finger}2")
        j3 = joint_names.index(f"{side}_{finger}3")
        target_tip = finger_tips.get((side, finger))
        if target_tip is None:
            target_tip = target_joints[j3] + (target_joints[j3] - target_joints[j2])
        return (
            source_anchors[j3],
            source_anchors[j3] + (source_anchors[j3] - source_anchors[j2]),
            target_joints[j3],
            target_tip,
        )
    else:
        return None
    if a_name not in joint_names or b_name not in joint_names:
        return None
    a, b = joint_names.index(a_name), joint_names.index(b_name)
    return source_anchors[a], source_anchors[b], target_joints[a], target_joints[b]


def _finger_tip_targets(
    canonical_dir: Path,
    *,
    joint_names: list[str],
    target_joints: np.ndarray,
    subject: bool,
) -> dict[tuple[str, str], np.ndarray]:
    """Locate each fingertip from that finger's own SMPL-X skin weights."""
    weights = np.load(canonical_dir / "smpl_canonical_weights.npz", allow_pickle=True)
    surface = _load_obj_vertices(
        canonical_dir / ("smpl_canonical_tpose.obj" if subject else "smpl_canonical_tpose_neutral.obj")
    )
    lbs = np.asarray(weights["lbs_weights"], dtype=np.float64)
    result: dict[tuple[str, str], np.ndarray] = {}
    for side in ("left", "right"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            ids = [joint_names.index(f"{side}_{finger}{level}") for level in (1, 2, 3)]
            mass = lbs[:, ids].sum(axis=1)
            selected = surface[mass > 0.05]
            if len(selected) < 8:
                selected = surface[np.argsort(-mass)[:32]]
            j2, j3 = target_joints[ids[1]], target_joints[ids[2]]
            axis = j3 - j2
            axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
            reach = float(np.quantile((selected - j3) @ axis, 0.99))
            # Keep the authored distal epiphysis just behind the skin front.
            result[(side, finger)] = j3 + 0.95 * max(0.0, reach) * axis
    return result


def _override(config: dict[str, Any], group: str) -> tuple[float, np.ndarray]:
    section = dict((config.get("fit_overrides", {}) or {}).get(group, {}) or {})
    scale = float(section.get("scale_multiplier", 1.0))
    offset = np.asarray(section.get("center_offset_local_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return scale, offset


def fit_source_bind_hands(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    stage: str = "source_bind_hand_refit",
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Refit authored Blender hands without changing the accepted body bake."""
    direct = _direct_smplx_hand_controllers(asset)
    direct_set = set(direct)
    joint_names = list(asset.joint_names)
    wrist_roots = {
        bone
        for bone in direct
        if joint_names[int(asset.source_bone_smplx_a[bone])].endswith("_wrist")
    }
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    hand_subtree: set[int] = set()
    for bone in range(len(parents)):
        current = int(bone)
        while current >= 0:
            if current in wrist_roots:
                hand_subtree.add(int(bone))
                break
            current = int(parents[current])

    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    source_geometry = (
        np.asarray(asset.source_bind_vertices, dtype=np.float64)
        if asset.source_bind_vertices is not None
        else np.asarray(asset.vertices_rest, dtype=np.float64)
    )
    source_anchors = _source_joint_anchors(
        asset,
        bind_global=np.asarray(asset.source_rest_global, dtype=np.float64),
    )
    hand_geometry_mask = np.zeros(len(vertices), dtype=bool)
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    finger_tips = _finger_tip_targets(
        Path(canonical_dir),
        joint_names=joint_names,
        target_joints=target_joints,
        subject=True,
    )
    fitted_meshes: list[str] = []
    for (start, stop), name, tissue in zip(
        asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
    ):
        if str(tissue).lower() != "bone":
            continue
        segment = _hand_mesh_segment(
            str(name),
            joint_names=joint_names,
            source_anchors=source_anchors,
            target_joints=target_joints,
            finger_tips=finger_tips,
        )
        if segment is None:
            continue
        start_i, stop_i = int(start), int(stop)
        source_a, source_b, target_a, target_b = segment
        target_a = np.asarray(target_a, dtype=np.float64)
        target_b = np.asarray(target_b, dtype=np.float64)
        vertices[start_i:stop_i] = shaft_preserving_segment_map(
            source_geometry[start_i:stop_i],
            source_a=source_a,
            source_b=source_b,
            target_a=target_a,
            target_b=target_b,
        )
        hand_geometry_mask[start_i:stop_i] = True
        fitted_meshes.append(str(name))

    from scipy.spatial import cKDTree

    carpal_tokens = (
        "capitate",
        "hamate",
        "lunate",
        "pisiform",
        "scaphoid",
        "trapezium",
        "trapezoid",
        "triquetrum",
    )
    carpal_report: dict[str, Any] = {}
    baseline_vertices = source_geometry
    for side, suffix in (("left", "_l"), ("right", "_r")):
        ranges = [
            (int(start), int(stop), str(name))
            for (start, stop), name, tissue in zip(
                asset.source_vertex_ranges,
                asset.source_mesh_names,
                asset.source_tissues,
            )
            if str(tissue).lower() == "bone"
            and str(name).lower().endswith(suffix)
            and any(token in str(name).lower() for token in carpal_tokens)
        ]
        if not ranges:
            continue
        carpal_indices = np.concatenate(
            [np.arange(start, stop, dtype=np.int64) for start, stop, _name in ranges]
        )
        source_carpal = baseline_vertices[carpal_indices]
        # A thumb-only contact fit rotates the whole carpal block around one
        # local point.  That leaves the radial/ulnar carpals outside the hand
        # when the beta changes.  Use all five metacarpal interfaces instead;
        # the fitted metacarpals already carry their exact Blender source-pivot
        # correspondence, so this is one shared hand transform, not a per-
        # carpal correction.
        source_contacts: list[np.ndarray] = []
        target_contacts: list[np.ndarray] = []
        source_contact_gaps: list[np.ndarray] = []
        mapped_metacarpals: list[np.ndarray] = []
        metacarpal_count = 0
        side_code = "L" if side == "left" else "R"
        for digit, ordinal in ((1, "st"), (2, "nd"), (3, "rd"), (4, "th"), (5, "th")):
            metacarpal_name = f"_{digit}{ordinal}_Metacarpal_{side_code}"
            metacarpal_range = next(
                (
                    (int(start), int(stop))
                    for (start, stop), mesh_name in zip(
                        asset.source_vertex_ranges, asset.source_mesh_names
                    )
                    if str(mesh_name) == metacarpal_name
                ),
                None,
            )
            if metacarpal_range is None:
                continue
            metacarpal_count += 1
            met_start, met_stop = metacarpal_range
            source_metacarpal = baseline_vertices[met_start:met_stop]
            mapped_metacarpal = vertices[met_start:met_stop]
            mapped_metacarpals.append(mapped_metacarpal)
            contact_distance, contact_metacarpal = cKDTree(source_metacarpal).query(
                source_carpal, k=1
            )
            source_contact_gaps.append(contact_distance)
            contact_count = min(24, len(source_carpal))
            contact_carpal = np.argpartition(
                contact_distance, contact_count - 1
            )[:contact_count]
            source_contact = source_carpal[contact_carpal]
            target_contact = source_contact + (
                mapped_metacarpal[contact_metacarpal[contact_carpal]]
                - source_metacarpal[contact_metacarpal[contact_carpal]]
            )
            source_contacts.append(source_contact)
            target_contacts.append(target_contact)
        if not source_contacts:
            raise RuntimeError(f"missing metacarpal contacts for {side} carpal fit")
        source_contact = np.concatenate(source_contacts, axis=0)
        target_contact = np.concatenate(target_contacts, axis=0)
        source_center = np.mean(source_contact, axis=0)
        target_center = np.mean(target_contact, axis=0)
        u, _singular, vt = np.linalg.svd(
            (source_contact - source_center).T @ (target_contact - target_center)
        )
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        denominator = float(np.sum((source_contact - source_center) ** 2))
        raw_scale = (
            float(
                np.sum(
                    (target_contact - target_center)
                    * ((source_contact - source_center) @ rotation.T)
                )
            )
            / max(denominator, 1.0e-10)
        )
        compound_scale = float(np.clip(raw_scale, 0.85, 1.05))
        translation = target_center - compound_scale * (rotation @ source_center)
        mapped_compound = compound_scale * (source_carpal @ rotation.T) + translation
        cursor = 0
        for start, stop, name in ranges:
            count = stop - start
            vertices[start:stop] = mapped_compound[cursor : cursor + count]
            cursor += count
            hand_geometry_mask[start:stop] = True
            fitted_meshes.append(name)
        carpal_report[side] = {
            "translation_m": translation.tolist(),
            "source_metacarpal_contact_gap_m": float(
                np.min(np.concatenate(source_contact_gaps, axis=0))
            ),
            "mapped_metacarpal_contact_gap_m": float(
                np.min(
                    cKDTree(np.concatenate(mapped_metacarpals, axis=0)).query(
                        mapped_compound, k=1
                    )[0]
                )
            ),
            "contact_samples": int(len(source_contact)),
            "metacarpal_count": int(metacarpal_count),
            "raw_uniform_scale": float(raw_scale),
            "uniform_scale": float(compound_scale),
        }

    interim = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": source_geometry.astype(np.float32),
            "target_rest_global": np.asarray(asset.source_bind_global, dtype=np.float32),
            "target_rest_local": np.asarray(asset.source_rest_local, dtype=np.float32),
            "target_inverse_bind": np.asarray(asset.source_inverse_bind, dtype=np.float32),
            "target_bone_head": np.asarray(asset.source_bone_head, dtype=np.float32),
            "target_bone_tail": np.asarray(asset.source_bone_tail, dtype=np.float32),
        }
    )
    rebound, local_rebind_report = rebind_source_rig(
        interim,
        source_vertices=source_geometry,
        target_vertices=vertices,
        stage=str(stage),
        bone_mask=hand_geometry_mask,
        fallback_to_soft=False,
        anchor_joint_local=True,
    )
    rebound_global = np.asarray(rebound.target_bind_global, dtype=np.float64)
    target_global = np.asarray(asset.target_bind_global, dtype=np.float64).copy()
    for bone in hand_subtree:
        target_global[bone] = rebound_global[bone]
    target_local = target_global.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            target_local[bone] = np.linalg.inv(target_global[int(parent)]) @ target_global[bone]
    target_head = np.asarray(asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(asset.target_bone_tail, dtype=np.float64).copy()
    rebound_head = np.asarray(rebound.target_bone_head, dtype=np.float64)
    rebound_tail = np.asarray(rebound.target_bone_tail, dtype=np.float64)
    for bone in hand_subtree:
        target_head[bone] = rebound_head[bone]
        target_tail[bone] = rebound_tail[bone]

    metadata = dict(asset.metadata or {})
    metadata["source_direct_driver_bones_v1"] = direct
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "target_rest_global": target_global.astype(np.float32),
            "target_rest_local": target_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(target_global).astype(np.float32),
            "target_bone_head": target_head.astype(np.float32),
            "target_bone_tail": target_tail.astype(np.float32),
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    if not np.array_equal(result.driver_indices, asset.driver_indices):
        raise RuntimeError("source-bind hand refit changed Blender driver indices")
    if not np.array_equal(result.driver_weights, asset.driver_weights):
        raise RuntimeError("source-bind hand refit changed Blender driver weights")
    if not np.array_equal(result.source_bone_parents, asset.source_bone_parents):
        raise RuntimeError("source-bind hand refit changed Blender hierarchy")
    report = {
        "stage": str(stage),
        "available": True,
        "backend": "blender_source_pivot_shaft_map_plus_direct_joint_coupling",
        "fitted_mesh_count": int(len(fitted_meshes)),
        "fitted_meshes": fitted_meshes,
        "wrist_subtree_bone_count": int(len(hand_subtree)),
        "direct_controller_count": int(len(direct_set)),
        "local_weighted_rebind": local_rebind_report,
        "hand_bone_radial_scale": 1.0,
        "distal_tip_inset_ratio": 0.0,
        "carpal_compound": carpal_report,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }
    result.validate()
    return result, report


def merge_fitted_hand_reference(
    body_asset: AnatomyRiggedAsset,
    hand_asset: AnatomyRiggedAsset,
    *,
    fitted_meshes: list[str],
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Merge a same-beta hand correspondence without touching the body rig."""
    for field in ("faces", "driver_indices", "driver_weights", "source_bone_parents"):
        if not np.array_equal(getattr(body_asset, field), getattr(hand_asset, field)):
            raise ValueError(f"hand correspondence {field} differs from body asset")
    if list(body_asset.source_mesh_names or []) != list(hand_asset.source_mesh_names or []):
        raise ValueError("hand correspondence mesh order differs from body asset")
    if not np.allclose(body_asset.rest_joints, hand_asset.rest_joints, atol=1.0e-7):
        raise ValueError("hand correspondence was baked for a different SMPL-X beta")

    selected_names = set(str(value) for value in fitted_meshes)
    vertices = np.asarray(body_asset.vertices_rest, dtype=np.float32).copy()
    merged_vertices = 0
    for (start, stop), name in zip(
        body_asset.source_vertex_ranges, body_asset.source_mesh_names
    ):
        if str(name) not in selected_names:
            continue
        start_i, stop_i = int(start), int(stop)
        vertices[start_i:stop_i] = np.asarray(hand_asset.vertices_rest)[start_i:stop_i]
        merged_vertices += stop_i - start_i

    direct = _direct_smplx_hand_controllers(hand_asset)
    wrist_roots = {
        bone
        for bone in direct
        if str(hand_asset.joint_names[int(hand_asset.source_bone_smplx_a[bone])]).endswith(
            "_wrist"
        )
    }
    parents = np.asarray(body_asset.source_bone_parents, dtype=np.int64)
    subtree: list[int] = []
    for bone in range(len(parents)):
        current = int(bone)
        while current >= 0:
            if current in wrist_roots:
                subtree.append(int(bone))
                break
            current = int(parents[current])

    target_global = np.asarray(body_asset.target_bind_global, dtype=np.float64).copy()
    target_head = np.asarray(body_asset.target_bone_head, dtype=np.float64).copy()
    target_tail = np.asarray(body_asset.target_bone_tail, dtype=np.float64).copy()
    target_global[subtree] = np.asarray(hand_asset.target_bind_global)[subtree]
    target_head[subtree] = np.asarray(hand_asset.target_bone_head)[subtree]
    target_tail[subtree] = np.asarray(hand_asset.target_bone_tail)[subtree]
    target_local = target_global.copy()
    for bone, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            target_local[bone] = np.linalg.inv(target_global[int(parent)]) @ target_global[bone]
    metadata = dict(body_asset.metadata or {})
    metadata["source_direct_driver_bones_v1"] = direct
    result = type(body_asset)(
        **{
            **body_asset.__dict__,
            "vertices_rest": vertices,
            "target_rest_global": target_global.astype(np.float32),
            "target_rest_local": target_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(target_global).astype(np.float32),
            "target_bone_head": target_head.astype(np.float32),
            "target_bone_tail": target_tail.astype(np.float32),
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    result.validate()
    return result, {
        "backend": "same_beta_hand_mesh_and_wrist_subtree_merge",
        "fitted_mesh_count": int(len(selected_names)),
        "merged_vertex_count": int(merged_vertices),
        "wrist_subtree_bone_count": int(len(subtree)),
        "direct_controller_count": int(len(direct)),
        "body_vertices_outside_hand_unchanged": True,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }


def fit_stage1_rigid_regions(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Fit Blender bone compounds without breaking authored joint contacts."""
    if asset.source_bind_vertices is None:
        raise ValueError("compound bone fit requires immutable Blender bind vertices")

    from scipy.spatial import cKDTree

    source = np.asarray(asset.source_bind_vertices, dtype=np.float64)
    target = np.asarray(asset.vertices_rest, dtype=np.float64)
    vertices = target.copy()
    hand_fitted, hand_report = fit_source_bind_hands(
        asset,
        canonical_dir=canonical_dir,
        stage="stage1_blender_source_pivot_hands",
    )
    ranges = np.asarray(asset.source_vertex_ranges, dtype=np.int64)
    names = [str(value) for value in asset.source_mesh_names]
    tissues = [str(value).lower() for value in asset.source_tissues]
    for mesh_name in hand_report["fitted_meshes"]:
        mesh_index = names.index(str(mesh_name))
        start, stop = (int(value) for value in ranges[mesh_index])
        vertices[start:stop] = np.asarray(hand_fitted.vertices_rest)[start:stop]

    def mesh_indices(predicate: Any) -> np.ndarray:
        selected: list[np.ndarray] = []
        for (start, stop), name, tissue in zip(ranges, names, tissues):
            if tissue == "bone" and bool(predicate(name.lower())):
                selected.append(np.arange(int(start), int(stop), dtype=np.int64))
        return np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)

    foot_tokens = (
        "calcaneus", "talus", "navicular", "cuboid", "cuneiform",
        "metatarsal", "phalanx_foot",
    )
    groups: dict[str, np.ndarray] = {
        "pelvis": mesh_indices(
            lambda name: "sacrum" in name or "ilium" in name
        ),
    }
    for side, suffix in (("left", "_l"), ("right", "_r")):
        side_match = lambda name, suffix=suffix: name.endswith(suffix)
        groups.update(
            {
                f"{side}_shoulder": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and ("clavicle" in name or "scapula" in name)
                ),
                f"{side}_upper_arm": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and "humerus" in name
                ),
                f"{side}_forearm": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and ("radius" in name or "ulna" in name)
                ),
                f"{side}_thigh": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and "femur" in name
                ),
                f"{side}_lower_leg": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and any(token in name for token in ("tibia", "fibula"))
                ),
                f"{side}_patella": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and "patella" in name
                ),
                f"{side}_foot": mesh_indices(
                    lambda name, side_match=side_match: side_match(name)
                    and any(token in name for token in foot_tokens)
                ),
            }
        )
    groups = {name: indices for name, indices in groups.items() if len(indices)}
    def source_contact(
        parent_name: str, child_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        parent_points = source[groups[parent_name]]
        child_points = source[groups[child_name]]
        distance, nearest = cKDTree(parent_points).query(child_points, k=1)
        count = min(32, len(child_points))
        selected = np.argpartition(distance, count - 1)[:count]
        parent_indices = groups[parent_name][nearest[selected]]
        child_indices = groups[child_name][selected]
        return (
            parent_indices,
            child_indices,
            np.mean(source[parent_indices], axis=0),
            np.mean(source[child_indices], axis=0),
        )

    chains: list[tuple[str, str, str]] = []
    for side in ("left", "right"):
        chains.extend(
            (
                ("pelvis", f"{side}_thigh", f"{side}_hip"),
                (f"{side}_thigh", f"{side}_lower_leg", f"{side}_knee"),
                (f"{side}_lower_leg", f"{side}_foot", f"{side}_ankle"),
                (f"{side}_shoulder", f"{side}_upper_arm", f"{side}_shoulder"),
                (f"{side}_upper_arm", f"{side}_forearm", f"{side}_elbow"),
            )
        )
    interfaces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in groups
    }
    contact_samples: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    shared_targets: dict[str, np.ndarray] = {}
    authority_groups = {name for name in ("pelvis",) if name in groups}
    authority_fit: dict[str, tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
    for group_name in authority_groups:
        indices = groups[group_name]
        source_points = source[indices]
        target_points = target[indices]
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        u_authority, _singular_authority, vt_authority = np.linalg.svd(
            (source_points - source_center).T @ (target_points - target_center)
        )
        rotation_authority = vt_authority.T @ u_authority.T
        if np.linalg.det(rotation_authority) < 0.0:
            vt_authority[-1] *= -1.0
            rotation_authority = vt_authority.T @ u_authority.T
        rotated_authority = (source_points - source_center) @ rotation_authority.T
        denominator_authority = float(np.sum((source_points - source_center) ** 2))
        scale_authority = float(
            np.clip(
                np.sum((target_points - target_center) * rotated_authority)
                / max(denominator_authority, 1.0e-10),
                0.90,
                1.05,
            )
        )
        authority_fit[group_name] = (
            rotation_authority,
            scale_authority,
            source_center,
            target_center,
        )

    def map_authority_contact(group_name: str, point: np.ndarray) -> np.ndarray:
        rotation, scale, source_center, target_center = authority_fit[group_name]
        return target_center + scale * (rotation @ (point - source_center))

    for parent_name, child_name, joint_name in chains:
        if parent_name not in groups or child_name not in groups:
            continue
        parent_indices, child_indices, source_parent, source_child = source_contact(
            parent_name, child_name
        )
        target_parent = np.mean(target[parent_indices], axis=0)
        target_child = np.mean(target[child_indices], axis=0)
        if parent_name in authority_groups:
            shared_contact = map_authority_contact(parent_name, source_parent)
        elif child_name in authority_groups:
            shared_contact = map_authority_contact(child_name, source_child)
        else:
            shared_contact = 0.5 * (target_parent + target_child)
        if parent_name not in authority_groups:
            interfaces[parent_name].append((source_parent, shared_contact))
        if child_name not in authority_groups:
            interfaces[child_name].append((source_child, shared_contact))
        shared_targets[joint_name] = shared_contact
        contact_samples[joint_name] = (
            parent_indices,
            child_indices,
            source_parent,
            source_child,
            shared_contact,
        )

    source_joint_anchors = _source_joint_anchors(
        asset,
        bind_global=np.asarray(asset.source_rest_global, dtype=np.float64),
    )
    target_joint_anchors = np.asarray(asset.rest_joints, dtype=np.float64)
    joint_lookup = {str(name): index for index, name in enumerate(asset.joint_names)}
    segment_joints: dict[str, tuple[int, int]] = {}
    single_joint_compounds: dict[str, int] = {}
    for side in ("left", "right"):
        segment_joints.update(
            {
                f"{side}_upper_arm": (
                    joint_lookup[f"{side}_shoulder"], joint_lookup[f"{side}_elbow"]
                ),
                f"{side}_forearm": (
                    joint_lookup[f"{side}_elbow"], joint_lookup[f"{side}_wrist"]
                ),
                f"{side}_thigh": (
                    joint_lookup[f"{side}_hip"], joint_lookup[f"{side}_knee"]
                ),
                f"{side}_lower_leg": (
                    joint_lookup[f"{side}_knee"], joint_lookup[f"{side}_ankle"]
                ),
            }
        )
        single_joint_compounds[f"{side}_shoulder"] = joint_lookup[
            f"{side}_shoulder"
        ]
        single_joint_compounds[f"{side}_foot"] = joint_lookup[f"{side}_ankle"]
        single_joint_compounds[f"{side}_patella"] = joint_lookup[f"{side}_knee"]

    # These meshes are already represented by the same beta harmonic field as
    # the target body.  A single correspondence similarity fixes their global
    # placement while preserving the authored source cross-section; unlike the
    # old endpoint fit it does not force a left/right source-pivot asymmetry.
    correspondence_groups = {
        name
        for name in groups
        if name.endswith("_patella")
    }
    group_fit: dict[str, dict[str, Any]] = {}
    for group_name, indices in groups.items():
        src = source[indices]
        dst = target[indices]
        src_center = np.mean(src, axis=0)
        dst_center = np.mean(dst, axis=0)
        u, _singular, vt = np.linalg.svd(
            (src - src_center).T @ (dst - dst_center)
        )
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0.0:
            vt[-1] *= -1.0
            rotation = vt.T @ u.T
        rotated = (src - src_center) @ rotation.T
        denominator = float(np.sum((src - src_center) ** 2))
        raw_scale = (
            float(np.sum((dst - dst_center) * rotated)) / denominator
            if denominator > 1.0e-12
            else 1.0
        )
        cross_scale = float(np.clip(raw_scale, 0.90, 1.05))
        mapped = dst_center + cross_scale * rotated
        anchors = interfaces[group_name]
        if group_name in correspondence_groups:
            correspondence_u, _correspondence_s, correspondence_vt = np.linalg.svd(
                (src - src_center).T @ (dst - dst_center)
            )
            correspondence_rotation = correspondence_vt.T @ correspondence_u.T
            if np.linalg.det(correspondence_rotation) < 0.0:
                correspondence_vt[-1] *= -1.0
                correspondence_rotation = correspondence_vt.T @ correspondence_u.T
            correspondence_rotated = (src - src_center) @ correspondence_rotation.T
            correspondence_denominator = float(
                np.sum((src - src_center) ** 2)
            )
            correspondence_scale = float(
                np.clip(
                    np.sum((dst - dst_center) * correspondence_rotated)
                    / max(correspondence_denominator, 1.0e-10),
                    0.90,
                    1.05,
                )
            )
            mapped = dst_center + correspondence_scale * correspondence_rotated
            rotation = correspondence_rotation
            cross_scale = correspondence_scale
        elif group_name in segment_joints:
            joint_a, joint_b = segment_joints[group_name]
            source_a = source_joint_anchors[joint_a]
            source_b = source_joint_anchors[joint_b]
            target_a = target_joint_anchors[joint_a]
            target_b = target_joint_anchors[joint_b]
            scaled = source_a + cross_scale * (src - source_a)
            scaled_b = source_a + cross_scale * (source_b - source_a)
            mapped = shaft_preserving_segment_map(
                scaled,
                source_a=source_a,
                source_b=scaled_b,
                target_a=target_a,
                target_b=target_b,
            )
            rotation = _rotation_between(scaled_b - source_a, target_b - target_a)
        elif group_name in single_joint_compounds:
            joint = single_joint_compounds[group_name]
            source_anchor = source_joint_anchors[joint]
            target_anchor = target_joint_anchors[joint]
            mapped_anchor = dst_center + cross_scale * (
                rotation @ (source_anchor - src_center)
            )
            mapped += target_anchor - mapped_anchor
        elif len(anchors) >= 2:
            source_anchors = np.stack([value[0] for value in anchors], axis=0)
            target_anchors = np.stack([value[1] for value in anchors], axis=0)
            source_midpoint = np.mean(source_anchors, axis=0)
            target_midpoint = np.mean(target_anchors, axis=0)
            u_anchor, _singular_anchor, vt_anchor = np.linalg.svd(
                (source_anchors - source_midpoint).T
                @ (target_anchors - target_midpoint)
            )
            rotation = vt_anchor.T @ u_anchor.T
            if np.linalg.det(rotation) < 0.0:
                vt_anchor[-1] *= -1.0
                rotation = vt_anchor.T @ u_anchor.T
            rotated_anchors = (source_anchors - source_midpoint) @ rotation.T
            anchor_denominator = float(
                np.sum((source_anchors - source_midpoint) ** 2)
            )
            anchor_scale = float(
                np.clip(
                    np.sum(
                        (target_anchors - target_midpoint) * rotated_anchors
                    )
                    / max(anchor_denominator, 1.0e-10),
                    0.85,
                    1.05,
                )
            )
            mapped = target_midpoint + anchor_scale * (
                (src - source_midpoint) @ rotation.T
            )
            cross_scale = anchor_scale
        elif len(anchors) == 1:
            source_anchor, target_anchor = anchors[0]
            mapped_anchor = dst_center + cross_scale * (
                rotation @ (source_anchor - src_center)
            )
            mapped += target_anchor - mapped_anchor
        vertices[indices] = mapped
        residual = np.linalg.norm(mapped - dst, axis=1)
        group_fit[group_name] = {
            "vertex_count": int(len(indices)),
            "raw_uniform_scale": float(raw_scale),
            "cross_section_scale": cross_scale,
            "target_residual_rms_m": float(np.sqrt(np.mean(residual**2))),
            "rotation": rotation,
        }

    # The harmonic body can be narrower than the authored head while the
    # mandible and cranial vault must remain one anatomical compound.  Fit the
    # complete source skull once, then choose the largest uniformly scaled
    # candidate contained by the subject surface.  This preserves jaw/teeth
    # proportions and never translates the mandible independently.
    head_indices = mesh_indices(
        lambda name: name in {"upper_skull", "skull", "cranium", "mandible", "jaw"}
    )
    head_fit_report: dict[str, Any] = {"available": False}
    if len(head_indices):
        canonical_root = Path(canonical_dir)
        target_surface = _load_obj_vertices(
            canonical_root / "smpl_canonical_tpose.obj"
        )
        target_faces = np.asarray(
            np.load(canonical_root / "smpl_canonical_weights.npz", allow_pickle=True)["faces"],
            dtype=np.int32,
        )
        target_head_surface = _surface_region(
            canonical_root,
            list(asset.joint_names),
            ("head",),
            subject=True,
        )
        source_head_center, target_head_center = _midline_envelope_centers(
            reference_points=source[head_indices],
            target_points=target_head_surface,
            source_anchors=source_joint_anchors,
            target_joints=target_joint_anchors,
            joint_names=list(asset.joint_names),
            center_offset=np.zeros(3, dtype=np.float64),
        )
        head_fitted, head_scale, envelope_report = _uniform_envelope_fit(
            source[head_indices],
            target_head_surface,
            reference_points=source[head_indices],
            scale_multiplier=1.0,
            center_offset=np.zeros(3, dtype=np.float64),
            margin=1.0,
            maximum_scale=1.25,
            minimum_scale=0.70,
            scale_mode="median",
            source_center=source_head_center,
            target_center=target_head_center,
        )
        head_candidate, contained_scale, containment_report = _contained_uniform_candidate(
            source[head_indices],
            source_center=source_head_center,
            target_center=target_head_center,
            target_surface=target_surface,
            target_faces=target_faces,
            desired_scale=head_scale,
            minimum_scale=0.70,
            # A zero clearance is intentional here.  The skull is a rigid
            # compound; a millimetre envelope margin would reject an otherwise
            # contained candidate at the SMPL-X facial openings and leave the
            # original, much larger head in place.
            clearance_m=0.0,
        )
        if bool(containment_report.get("candidate_contained")):
            head_fitted = head_candidate
            head_scale = contained_scale
        vertices[head_indices] = head_fitted
        head_fit_report = {
            "available": True,
            "mesh_vertex_count": int(len(head_indices)),
            "uniform_scale": float(head_scale),
            "envelope": envelope_report,
            "contained_candidate": containment_report,
            "independent_jaw_translation": False,
            "source_proportions_preserved": True,
        }
    contact_report: dict[str, Any] = {}
    for joint_name, sample in contact_samples.items():
        parent_contact, child_contact, source_parent, source_child, _target_joint = sample
        target_joint = shared_targets[joint_name]
        mapped_parent = np.mean(vertices[parent_contact], axis=0)
        mapped_child = np.mean(vertices[child_contact], axis=0)
        contact_report[joint_name] = {
            "source_contact_sample_centroid_gap_m": float(
                np.linalg.norm(source_child - source_parent)
            ),
            "mapped_contact_sample_centroid_gap_m": float(
                np.linalg.norm(mapped_child - mapped_parent)
            ),
            "parent_to_shared_anchor_m": float(
                np.linalg.norm(mapped_parent - target_joint)
            ),
            "child_to_shared_anchor_m": float(
                np.linalg.norm(mapped_child - target_joint)
            ),
        }

    bone_mask = np.zeros(len(vertices), dtype=bool)
    for (start, stop), tissue in zip(ranges, tissues):
        if tissue == "bone":
            bone_mask[int(start) : int(stop)] = True
    source_frame_asset = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": source.astype(np.float32),
            "target_rest_global": np.asarray(asset.source_bind_global, dtype=np.float32),
            "target_rest_local": np.asarray(asset.source_rest_local, dtype=np.float32),
            "target_inverse_bind": np.asarray(asset.source_inverse_bind, dtype=np.float32),
            "target_bone_head": np.asarray(asset.source_bone_head, dtype=np.float32),
            "target_bone_tail": np.asarray(asset.source_bone_tail, dtype=np.float32),
        }
    )
    rebound, rebind_report = rebind_source_rig(
        source_frame_asset,
        source_vertices=source,
        target_vertices=vertices,
        stage="stage1_blender_compound_chains",
        bone_mask=bone_mask,
        fallback_to_soft=False,
        anchor_joint_local=True,
    )
    rebound = type(rebound)(
        **{**rebound.__dict__, "vertices_rest": vertices.astype(np.float32)}
    )
    metadata = dict(rebound.metadata or {})
    metadata["source_full_local_fk_v2"] = False
    metadata["source_direct_driver_bones_v1"] = _direct_smplx_hand_controllers(asset)
    result = type(rebound)(**{**rebound.__dict__, "metadata": metadata})
    result = with_source_driver_coupling(result)
    result.validate()
    return result, {
        "backend": "blender_source_compound_chain_fit_v1",
        "hand": hand_report,
        "groups": {
            name: {key: value for key, value in report.items() if key != "rotation"}
            for name, report in group_fit.items()
        },
        "shared_contact_anchors": contact_report,
        "head_compound": head_fit_report,
        "contact_centroid_metrics_are_not_surface_clearance": True,
        "minimum_cross_section_scale": 0.90,
        "independent_mesh_fits": False,
        "changed_vertex_count": int(np.count_nonzero(np.linalg.norm(vertices - target, axis=1))),
        "source_rig_rebind": rebind_report,
        "source_full_local_fk_v2": False,
        "source_weights_preserved": True,
        "source_hierarchy_preserved": True,
    }


def fit_articulated_rest(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    config: dict[str, Any] | None = None,
    subject: bool,
    stage: str,
    preserve_source_binding: bool = False,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Fit rigid anatomy, rebind source frames, then transport soft tissue once."""
    asset.validate()
    asset = _correct_clavicle_segment_joints(asset)
    cfg = dict(config or {})
    root = Path(canonical_dir)
    # Ribs must follow their authored thoracic parent.  Older source caches still
    # mark them as rigid_group(spine2→spine3), which explodes the cage under pose.
    modes = list(asset.source_bone_driver_types or [])
    source_names = list(asset.source_bone_names or [])
    if modes and source_names and len(modes) == len(source_names):
        patched = False
        for bi, name in enumerate(source_names):
            lower = str(name).lower()
            if lower.startswith("rib_bone_") or lower.startswith("rib_name_"):
                if not _is_follow_mode(modes[bi]):
                    modes[bi] = "bind_follow"
                    patched = True
        if patched:
            asset = type(asset)(**{**asset.__dict__, "source_bone_driver_types": modes})
    old_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    vertices = old_vertices.copy()
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    new_global, new_local, bone_delta = _fit_source_frames(asset)
    # Pure Blender/SMPL joint-linkage bone centroids (before material mesh surgery).
    blender_bone_centroids, blender_bone_valid = _bone_mesh_centroids(
        asset, old_vertices, bone_delta=bone_delta
    )
    source_anchors = _source_joint_anchors(asset)
    authored_hand_vertices = (
        np.asarray(asset.source_bind_vertices, dtype=np.float64)
        if asset.source_bind_vertices is not None
        else old_vertices
    )
    authored_hand_anchors = (
        _source_joint_anchors(
            asset,
            bind_global=np.asarray(asset.source_rest_global, dtype=np.float64),
        )
        if asset.source_bind_vertices is not None
        else source_anchors
    )
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64).copy()
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    shaft_meshes = 0
    protected_end_edge_change = 0.0
    finger_tips = _finger_tip_targets(
        root,
        joint_names=asset.joint_names,
        target_joints=target_joints,
        subject=subject,
    )

    cranial = cranial_material_mask(asset)
    jaw = jaw_material_mask(asset)
    pelvis_material = _mesh_mask(
        asset, lambda name, tissue: tissue == "bone" and any(t in name for t in _PELVIS_TOKENS)
    )
    thorax_material = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone"
        and any(token in name for token in ("sternum", "rib_")),
    )
    foot_material = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone" and any(token in name for token in _FOOT_TOKENS),
    )
    bone_material = bone_material_mask(asset)
    fit_driver_indices = np.asarray(asset.driver_indices, dtype=np.int32)
    fit_driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
    scapula_report: dict[str, Any] = {}

    if asset.source_vertex_ranges is not None and asset.source_tissues is not None:
        for (start, stop), name, tissue in zip(
            asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
        ):
            start_i, stop_i = int(start), int(stop)
            if str(tissue) != "bone":
                continue
            bone = _dominant_bone(asset, start_i, stop_i)
            if bone is None:
                continue
            if (
                np.any(cranial[start_i:stop_i])
                or np.any(jaw[start_i:stop_i])
                or np.any(pelvis_material[start_i:stop_i])
                or np.any(thorax_material[start_i:stop_i])
                or np.any(foot_material[start_i:stop_i])
            ):
                continue
            control = _controller(bone, source_parents, modes)
            a = int(asset.source_bone_smplx_a[control])
            b = int(asset.source_bone_smplx_b[control])
            lower = str(name).lower()
            if "clavicle" in lower and a != b:
                # Authored clavicle bone was tagged spine3→collar; remap the
                # mesh along its authored head→tail onto subject collar→shoulder.
                fitted = shaft_preserving_segment_map(
                    old_vertices[start_i:stop_i],
                    source_a=np.asarray(
                        asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
                        dtype=np.float64,
                    )[bone],
                    source_b=np.asarray(
                        asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
                        dtype=np.float64,
                    )[bone],
                    target_a=target_joints[a],
                    target_b=target_joints[b],
                )
                vertices[start_i:stop_i] = fitted
                protected_end_edge_change = max(
                    protected_end_edge_change,
                    _protected_end_edge_change(
                        asset,
                        start=start_i,
                        stop=stop_i,
                        source=old_vertices[start_i:stop_i],
                        fitted=fitted,
                        source_a=np.asarray(
                            asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
                            dtype=np.float64,
                        )[bone],
                        source_b=np.asarray(
                            asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
                            dtype=np.float64,
                        )[bone],
                    ),
                )
                shaft_meshes += 1
                continue
            if "scapula" in lower:
                side = "left" if str(name).endswith(("_L", "_l")) else "right"
                collar_i = asset.joint_names.index(f"{side}_collar")
                shoulder_i = asset.joint_names.index(f"{side}_shoulder")
                spine3_i = asset.joint_names.index("spine3")
                fitted, scap_info = _fit_scapula_mesh(
                    old_vertices[start_i:stop_i],
                    side=side,
                    source_shoulder=source_anchors[shoulder_i],
                    source_collar=source_anchors[collar_i],
                    source_spine3=source_anchors[spine3_i],
                    target_shoulder=target_joints[shoulder_i],
                    target_collar=target_joints[collar_i],
                    target_spine3=target_joints[spine3_i],
                )
                vertices[start_i:stop_i] = fitted
                scapula_report[side] = scap_info
                continue
            hand_segment = _hand_mesh_segment(
                str(name),
                joint_names=asset.joint_names,
                source_anchors=authored_hand_anchors,
                target_joints=target_joints,
                finger_tips=finger_tips,
            )
            if hand_segment is not None:
                source_a, source_b, target_a, target_b = hand_segment
                fitted = shaft_preserving_segment_map(
                    authored_hand_vertices[start_i:stop_i],
                    source_a=source_a,
                    source_b=source_b,
                    target_a=target_a,
                    target_b=target_b,
                )
                vertices[start_i:stop_i] = fitted
                protected_end_edge_change = max(
                    protected_end_edge_change,
                    _protected_end_edge_change(
                        asset,
                        start=start_i,
                        stop=stop_i,
                        source=old_vertices[start_i:stop_i],
                        fitted=fitted,
                        source_a=source_a,
                        source_b=source_b,
                    ),
                )
                shaft_meshes += 1
                continue
            if a == b and _is_joint_local_mode(modes[control]):
                child = _joint_child(a, asset.parents)
                if child is not None:
                    b = child
            if a != b and any(token in lower for token in _LONG_BONE_TOKENS):
                fitted = shaft_preserving_segment_map(
                    old_vertices[start_i:stop_i],
                    source_a=source_anchors[a],
                    source_b=source_anchors[b],
                    target_a=target_joints[a],
                    target_b=target_joints[b],
                )
                vertices[start_i:stop_i] = fitted
                protected_end_edge_change = max(
                    protected_end_edge_change,
                    _protected_end_edge_change(
                        asset,
                        start=start_i,
                        stop=stop_i,
                        source=old_vertices[start_i:stop_i],
                        fitted=fitted,
                        source_a=source_anchors[a],
                        source_b=source_anchors[b],
                    ),
                )
                shaft_meshes += 1
            else:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[bone]
                )

    skull_reference = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone" and ("skull" in name or "cranium" in name),
    )
    mandible_reference = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone"
        and ("mandible" in name or "jaw" in name),
    )
    cranial_scale = 1.0
    cranial_scale_report: dict[str, Any] = {}
    cranial_aspect_ratio_change = 0.0
    brain_skull_center_drift_m = 0.0
    cranial_envelope_center_before: np.ndarray | None = None
    cranial_envelope_center_after: np.ndarray | None = None
    cranial_soft_moved = np.zeros(len(vertices), dtype=bool)
    if np.any(cranial) or np.any(jaw):
        source_names = list(asset.source_bone_names or [])
        # Rest fit treats mandible as part of one closed skull compound.
        # Jaw_Bone_tip is for pose articulation only — using it here tears the
        # mandible off the cranial vault (upper/lower teeth no longer meet).
        head_compound = cranial | jaw
        if "Head_Bone" in source_names and np.any(head_compound):
            vertices[head_compound] = _transform_points(
                old_vertices[head_compound], bone_delta[source_names.index("Head_Bone")]
            )
        old_cranial = vertices[cranial].copy() if np.any(cranial) else np.zeros((0, 3))
        old_skull_center = (
            np.mean(vertices[skull_reference | mandible_reference], axis=0)
            if np.any(skull_reference | mandible_reference)
            else np.mean(vertices[head_compound], axis=0)
        )
        old_brain_center = (
            np.mean(vertices[cranial & ~skull_reference], axis=0)
            if np.any(cranial & ~skull_reference)
            else old_skull_center
        )
        # Envelope against the full bony head (skull + mandible).
        bone_reference = skull_reference | mandible_reference
        if not np.any(bone_reference):
            bone_reference = head_compound
        target_head = _surface_region(
            root,
            asset.joint_names,
            ("head",),
            subject=subject,
        )
        multiplier, local_offset = _override(cfg, "skull")
        head_index = asset.joint_names.index("head")
        head_frame = joint_global_transforms(
            pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )[head_index]
        offset_world = head_frame[:3, :3] @ local_offset
        cranial_reference = vertices[bone_reference]
        cranial_envelope_center_before, cranial_envelope_center_after = _midline_envelope_centers(
            reference_points=cranial_reference,
            target_points=target_head,
            source_anchors=source_anchors,
            target_joints=target_joints,
            joint_names=list(asset.joint_names),
            center_offset=offset_world,
        )
        compound_before_envelope = vertices[head_compound].copy()
        vertices[head_compound], cranial_scale, cranial_scale_report = _uniform_envelope_fit(
            compound_before_envelope,
            target_head,
            reference_points=cranial_reference,
            scale_multiplier=multiplier,
            center_offset=offset_world,
            # The old 0.88 margin made the current skull 11.4% smaller.  Use
            # the robust isotropic envelope and leave containment to the
            # candidate-scale diagnostic rather than pre-shrinking the head.
            margin=1.0,
            maximum_scale=1.25,
            minimum_scale=0.70,
            scale_mode="median",
            source_center=cranial_envelope_center_before,
            target_center=cranial_envelope_center_after,
        )
        full_surface = _load_obj_vertices(
            root
            / (
                "smpl_canonical_tpose.obj"
                if subject
                else "smpl_canonical_tpose_neutral.obj"
            )
        )
        full_surface_faces = np.asarray(
            np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)["faces"],
            dtype=np.int32,
        )
        contained_vertices, contained_scale, containment_candidate = _contained_uniform_candidate(
            compound_before_envelope,
            source_center=cranial_envelope_center_before,
            target_center=cranial_envelope_center_after,
            target_surface=full_surface,
            target_faces=full_surface_faces,
            desired_scale=cranial_scale,
            minimum_scale=0.70,
        )
        cranial_scale_report["containment_candidate"] = containment_candidate
        if bool(containment_candidate.get("candidate_contained")):
            vertices[head_compound] = contained_vertices
            cranial_scale = contained_scale
        cranial_soft_moved = (cranial | jaw) & ~bone_material
        if np.any(cranial) and len(old_cranial):
            cranial_aspect_ratio_change = _aspect_ratio_change(old_cranial, vertices[cranial])
        new_skull_center = (
            np.mean(vertices[skull_reference | mandible_reference], axis=0)
            if np.any(skull_reference | mandible_reference)
            else np.mean(vertices[head_compound], axis=0)
        )
        new_brain_center = (
            np.mean(vertices[cranial & ~skull_reference], axis=0)
            if np.any(cranial & ~skull_reference)
            else new_skull_center
        )
        brain_skull_center_drift_m = float(
            np.linalg.norm(
                (new_brain_center - new_skull_center)
                - cranial_scale * (old_brain_center - old_skull_center)
            )
        )

    # Mandible stays in the closed-skull compound above (Head_Bone + envelope).
    pelvis = pelvis_material
    pelvis_scale = 1.0
    pelvis_scale_report: dict[str, Any] = {}
    pelvis_aspect_ratio_change = 0.0
    if np.any(pelvis):
        # Uniform envelope against the subject pelvic *surface*, centered on the
        # pelvis joint (spine base).  Scaling by the SMPL-X hip-joint span
        # collapsed the ilium (~0.5x) because those joints sit far more medial
        # than the authored iliac width.
        old_pelvis = old_vertices[pelvis].copy()
        multiplier, local_offset = _override(cfg, "pelvis")
        pelvis_id = asset.joint_names.index("pelvis")
        target_pelvis = _surface_region(
            root,
            asset.joint_names,
            ("pelvis", "left_hip", "right_hip", "spine1"),
            subject=subject,
        )
        rest_global_pelvis = joint_global_transforms(
            pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
            rest_joints=asset.rest_joints,
            parents=asset.parents,
        )[pelvis_id]
        offset_world = rest_global_pelvis[:3, :3] @ local_offset
        source_center = np.asarray(source_anchors[pelvis_id], dtype=np.float64)
        target_center = np.asarray(target_joints[pelvis_id], dtype=np.float64) + offset_world
        # Align lateral hip axis before the isotropic envelope so the sacrum
        # stays on the spine chain after scaling about the pelvis joint.
        left_hip_id = asset.joint_names.index("left_hip")
        right_hip_id = asset.joint_names.index("right_hip")
        source_axis = source_anchors[right_hip_id] - source_anchors[left_hip_id]
        target_axis = target_joints[right_hip_id] - target_joints[left_hip_id]
        rotation = _rotation_between(source_axis, target_axis)
        rotated = (old_vertices[pelvis] - source_center) @ rotation.T + target_center
        vertices[pelvis], pelvis_scale, pelvis_scale_report = _uniform_envelope_fit(
            rotated,
            target_pelvis,
            reference_points=rotated,
            scale_multiplier=multiplier,
            center_offset=np.zeros(3, dtype=np.float64),
            margin=1.0,
            maximum_scale=1.60,
            minimum_scale=0.80,
            scale_mode="median",
            source_center=target_center,
            target_center=target_center,
        )
        # Floor so a narrow subject hip surface cannot collapse the ilium.
        if pelvis_scale < 0.80:
            vertices[pelvis] = target_center + 0.80 / max(pelvis_scale, 1.0e-8) * (
                vertices[pelvis] - target_center
            )
            pelvis_scale = 0.80
        pelvis_surface = _load_obj_vertices(
            root
            / (
                "smpl_canonical_tpose.obj"
                if subject
                else "smpl_canonical_tpose_neutral.obj"
            )
        )
        pelvis_surface_faces = np.asarray(
            np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)["faces"],
            dtype=np.int32,
        )
        contained_pelvis, contained_pelvis_scale, pelvis_candidate = _contained_uniform_candidate(
            rotated,
            source_center=target_center,
            target_center=target_center,
            target_surface=pelvis_surface,
            target_faces=pelvis_surface_faces,
            desired_scale=pelvis_scale,
            minimum_scale=0.80,
            clearance_m=0.0005,
        )
        pelvis_scale_report["containment_candidate"] = pelvis_candidate
        if bool(pelvis_candidate.get("candidate_contained")):
            vertices[pelvis] = contained_pelvis
            pelvis_scale = contained_pelvis_scale
            pelvis_scale_report["envelope_saturated"] = bool(
                pelvis_scale_report.get("saturated")
            )
            pelvis_scale_report["scale"] = float(pelvis_scale)
            pelvis_scale_report["saturated"] = False
            pelvis_scale_report["selection"] = "maximum_contained_uniform"
        pelvis_aspect_ratio_change = _aspect_ratio_change(old_pelvis, vertices[pelvis])

    hip_report: dict[str, Any] = {}
    for side in ("left", "right"):
        pair = _femur_head_and_acetabulum(
            asset, vertices, side=side, target_joints=target_joints
        )
        suffix = "_l" if side == "left" else "_r"
        femur = _mesh_mask(
            asset,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and "femur" in name
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        if pair is None or not np.any(femur):
            continue
        femoral_head, acetabulum = pair
        # The acetabular surface is the fixed member of the pelvis compound;
        # the femoral epiphysis is mapped rigidly onto that shared center.
        shared_center = acetabulum.copy()
        knee_id = asset.joint_names.index(f"{side}_knee")
        hip_id = asset.joint_names.index(f"{side}_hip")
        knee = target_joints[knee_id]
        femur_points = vertices[femur].copy()
        shaft_axis = knee - femoral_head
        shaft_axis /= max(float(np.linalg.norm(shaft_axis)), 1.0e-8)
        axial = (femur_points - femoral_head) @ shaft_axis
        distal = np.mean(
            femur_points[axial >= np.quantile(axial, 0.85)], axis=0
        )
        vertices[femur] = shaft_preserving_segment_map(
            femur_points,
            source_a=femoral_head,
            source_b=distal,
            target_a=shared_center,
            target_b=knee,
        )
        rotation = _rotation_between(distal - femoral_head, knee - shared_center)
        frame_delta = np.eye(4, dtype=np.float64)
        frame_delta[:3, :3] = rotation
        frame_delta[:3, 3] = shared_center - rotation @ femoral_head
        for bone, bone_name in enumerate(asset.source_bone_names or []):
            lower = str(bone_name).lower()
            if "femur" in lower and (lower.endswith(suffix) or f"{suffix}_" in lower):
                new_global[bone] = frame_delta @ new_global[bone]
        target_joints[hip_id] = shared_center
        hip_report[side] = {
            "search_prior": "smplx_hip_soft_constraint",
            "pre_surface_gap_m": float(np.linalg.norm(femoral_head - acetabulum)),
            "shared_center_m": shared_center.tolist(),
            "femoral_head_to_shared_center_m": float(
                np.linalg.norm(femoral_head - shared_center)
            ),
            "acetabulum_to_shared_center_m": float(
                np.linalg.norm(acetabulum - shared_center)
            ),
            "femoral_head_to_acetabulum_m": 0.0,
        }
    if hip_report:
        new_local = new_global.copy()
        for bone, parent in enumerate(source_parents.tolist()):
            if int(parent) >= 0:
                new_local[bone] = np.linalg.inv(new_global[int(parent)]) @ new_global[bone]
        bone_delta = new_global @ np.linalg.inv(old_global)

    spine_interface_report: dict[str, Any] = {
        "available": False,
        "reason": "interface_geometry_missing",
    }
    if np.any(pelvis) and np.any(skull_reference):
        pelvis_id = asset.joint_names.index("pelvis")
        spine1_id = asset.joint_names.index("spine1")
        neck_id = asset.joint_names.index("neck")
        head_id = asset.joint_names.index("head")
        inferior_axis = target_joints[spine1_id] - target_joints[pelvis_id]
        inferior_axis /= max(float(np.linalg.norm(inferior_axis)), 1.0e-10)
        sacrum = _mesh_mask(
            asset,
            lambda name, tissue: tissue == "bone" and "sacrum" in name,
        )
        pelvis_interface_points = vertices[sacrum] if np.any(sacrum) else vertices[pelvis]
        pelvis_projection = pelvis_interface_points @ inferior_axis
        pelvis_l5_interface = np.median(
            pelvis_interface_points[
                pelvis_projection >= np.quantile(pelvis_projection, 0.97)
            ],
            axis=0,
        )
        superior_axis = target_joints[head_id] - target_joints[neck_id]
        superior_axis /= max(float(np.linalg.norm(superior_axis)), 1.0e-10)
        skull_points = vertices[skull_reference]
        skull_projection = skull_points @ superior_axis
        skull_c1_interface = np.median(
            skull_points[skull_projection <= np.quantile(skull_projection, 0.03)],
            axis=0,
        )
        new_global, new_local, spine_interface_report = _fit_final_spine_interfaces(
            asset,
            old_global=old_global,
            new_global=new_global,
            pelvis_l5_interface=pelvis_l5_interface,
            skull_c1_interface=skull_c1_interface,
            target_joints=target_joints,
        )
        bone_delta = new_global @ np.linalg.inv(old_global)
        for (start, stop), mesh_name, tissue in zip(
            asset.source_vertex_ranges, asset.source_mesh_names, asset.source_tissues
        ):
            lower = str(mesh_name).lower()
            if str(tissue).lower() != "bone" or not any(
                token in lower for token in ("spine_", "vertebra", "disc")
            ):
                continue
            start_i, stop_i = int(start), int(stop)
            controller = _dominant_bone(asset, start_i, stop_i)
            if controller is not None:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[controller]
                )

    thorax = thorax_material
    thorax_scale = 1.0
    thorax_axis_scale = np.ones(3, dtype=np.float64)
    if np.any(thorax):
        for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
            start_i, stop_i = int(start), int(stop)
            if str(tissue).lower() != "bone" or not np.any(thorax[start_i:stop_i]):
                continue
            controller = _dominant_bone(asset, start_i, stop_i)
            if controller is not None:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[controller]
                )

    foot_report: dict[str, Any] = {}
    soft_follow_report: dict[str, Any] = {}
    surface_faces = np.asarray(
        np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)["faces"], dtype=np.int32
    )
    subject_surface = _load_obj_vertices(
        root / ("smpl_canonical_tpose.obj" if subject else "smpl_canonical_tpose_neutral.obj")
    )
    for side in ("left", "right"):
        suffix = "_l" if side == "left" else "_r"
        foot = _mesh_mask(
            asset,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and any(token in name for token in _FOOT_TOKENS)
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        if not np.any(foot):
            continue
        target_foot = _surface_region(
            root,
            asset.joint_names,
            (f"{side}_ankle", f"{side}_foot"),
            subject=subject,
        )
        ankle = target_joints[asset.joint_names.index(f"{side}_ankle")]
        forward = target_joints[asset.joint_names.index(f"{side}_foot")] - ankle
        forward /= max(float(np.linalg.norm(forward)), 1.0e-8)
        root_name = f"Ankle_Rot_{'L' if side == 'left' else 'R'}"
        source_names = list(asset.source_bone_names or [])
        # Foot compound follows the Ankle_Rot bind delta rigidly so the ankle
        # joint stays connected.  Forefoot reach is then scaled about that
        # ankle; no independent per-toe translation is introduced here.
        if root_name in source_names:
            vertices[foot] = _transform_points(
                old_vertices[foot], bone_delta[source_names.index(root_name)]
            )
        source_reach = float(np.quantile((vertices[foot] - ankle) @ forward, 0.995))
        target_reach = float(np.quantile((target_foot - ankle) @ forward, 0.995))
        scale = float(np.clip(0.95 * target_reach / max(source_reach, 1.0e-5), 0.5, 1.05))
        # Length-only scale about ankle: preserve foot width / vessel wrap.
        vertices[foot] = _axial_scale_about_axis(
            vertices[foot], origin=ankle, axis=forward, scale=scale
        )
        import igl

        rigid_offset = np.zeros(3, dtype=np.float64)
        for _iteration in range(4):
            signed, _face_index, closest, _normal = igl.signed_distance(
                vertices[foot], subject_surface, surface_faces
            )
            outside = np.asarray(signed) > 0.0
            if not np.any(outside):
                break
            step = np.median(np.asarray(closest)[outside] - vertices[foot][outside], axis=0)
            length = float(np.linalg.norm(step))
            if length <= 1.0e-6:
                break
            step *= min(1.0, 0.003 / length)
            vertices[foot] += step
            rigid_offset += step
        foot_report[side] = {
            # Preserve the ef58024 hand/foot material fit as requested.  The
            # signed-distance query above selects one bounded rigid offset for
            # the whole foot compound; it never projects individual vertices.
            "fit_policy": "ef58024_ankle_axial_material_fit",
            "post_projection_applied": False,
            "rigid_containment_offset_applied": bool(
                np.linalg.norm(rigid_offset) > 0.0
            ),
            "axial_scale": float(scale),
            "source_reach_m": source_reach,
            "target_reach_m": target_reach,
            "surface_center_offset_m": rigid_offset.tolist(),
            "forefoot_gap_before_m": 0.0,
            "forefoot_rigid_shift_m": 0.0,
        }
        if asset.harmonic_reference_vertices is not None:
            soft_follow_report[side] = {
                "disabled": True,
                "reason": "station_soft_follow_authoritative",
            }
        else:
            soft_follow_report[side] = _follow_foot_soft_scale(
                vertices,
                asset,
                ankle=ankle,
                forward=forward,
                scale=scale,
                rigid_offset=rigid_offset,
            )
        foot_report[side]["soft_follow"] = soft_follow_report[side]

    soft_section_report = {"disabled": True, "reason": "knee_popliteal_pierce"}

    if preserve_source_binding:
        # Discard legacy regional soft edits first.  The final v71-preserving
        # rest transport is evaluated once below from the exact Blender
        # weights and the explicit source-bone material-fit transforms.
        vertices[~bone_material] = old_vertices[~bone_material]

    # Snapshot articulated bind frames before rebind.  Weighted vertex rebind
    # may improve local orientation, but must not drag bind origins off the
    # SMPL-X joints (that caused 13-32 cm anchor drift and detached chains).
    articulated_global = new_global.copy()
    articulated_local = new_local.copy()

    interim = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "target_rest_global": new_global.astype(np.float32),
            "target_rest_local": new_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
        }
    )
    if preserve_source_binding:
        # The articulated frames above are constructed from fixed SMPL-X joint
        # endpoints while retaining the mapped Blender transverse axes and
        # local hierarchy.  A second fit from moved mesh vertices is ambiguous
        # for thin/near-planar bones and previously changed rotation centers
        # after the geometry was already correct.  Keep these explicit frames
        # authoritative and never infer the rig back from material-fit meshes.
        rebound = interim
        new_global = articulated_global.copy()
        rebind_report = {
            "stage": str(stage),
            "backend": "v71_explicit_articulated_frames",
            "weighted_vertex_rebind_skipped": True,
            "source_weights_preserved": True,
            "source_hierarchy_preserved": True,
            "source_driver_modes_preserved": True,
        }
    else:
        rebound, rebind_report = rebind_source_rig(
            interim,
            source_vertices=old_vertices,
            target_vertices=vertices,
            stage=stage,
            bone_mask=bone_material,
        )
        new_global = np.asarray(rebound.target_bind_global, dtype=np.float64)
        # Rotation-only rebind for independent drivers: keep articulated bind
        # translation on the SMPL-X joint, adopt rebind orientation only.
        for bone, mode in enumerate(modes):
            if _is_follow_mode(mode):
                continue
            new_global[bone, :3, 3] = articulated_global[bone, :3, 3]
            if (
                mode in {"segment_root", "twist", "rigid_group"}
                or _is_joint_local_mode(mode)
                or str(mode).startswith("scapula_")
            ):
                # Prefer articulated orientation when the frame was set explicitly
                # from joint endpoints; rebind only supplies residual spin about it.
                new_global[bone, :3, :3] = articulated_global[bone, :3, :3]
        # Re-derive bind_follow children from the corrected parents so toes/patella
        # stay attached to ankle/knee without independent translation.
        for bone, mode in enumerate(modes):
            parent = int(source_parents[bone])
            if _is_follow_mode(mode) and parent >= 0:
                new_global[bone] = new_global[parent] @ articulated_local[bone]
    new_local = new_global.copy()
    for bone, parent in enumerate(source_parents.tolist()):
        if int(parent) >= 0:
            new_local[bone] = np.linalg.inv(new_global[int(parent)]) @ new_global[bone]
    bone_delta = new_global @ np.linalg.inv(old_global)
    rebind_report = {
        **dict(rebind_report),
        "anchor_translation_restored": True,
        "bind_follow_rederived": True,
    }

    if preserve_source_binding:
        # Bone material fitting changed the rest placement of the source rig.
        # Move thin anatomy through that exact same fixed Blender-weight field;
        # otherwise the bones move into vessels that remain at the harmonic
        # location.  This is an offline rest bake using all original sparse
        # influences—not SMPL-X weights, a station approximation, or an online
        # collision correction.
        inherited_soft = _tissue_mask(asset, "vessel")
        selected_delta = bone_delta[fit_driver_indices[inherited_soft]]
        blended_delta = np.sum(
            fit_driver_weights[inherited_soft, :, None, None] * selected_delta,
            axis=1,
        )
        source_soft = old_vertices[inherited_soft]
        homogeneous = np.concatenate(
            (source_soft, np.ones((len(source_soft), 1), dtype=np.float64)),
            axis=1,
        )
        vertices[inherited_soft] = np.matmul(
            blended_delta, homogeneous[:, :, None]
        )[:, :3, 0]

    # Soft already sits on the Blender harmonic field. Material mesh surgery
    # moves bones off that linkage — close the gap with a translation-only field
    # (no SE(3) LBS, no vessel SDF push). Near-skin attenuation keeps topology.
    soft_material = _soft_material_mask(asset)
    soft_material &= ~(cranial_soft_moved | jaw)
    vessel_nerve = _tissue_mask(asset, "vessel", "nerve") & soft_material
    material_bone_centroids, material_bone_valid = _bone_mesh_centroids(asset, vertices)
    bone_valid = blender_bone_valid & material_bone_valid
    if asset.harmonic_reference_vertices is not None:
        # The explicit station bake performed after final endpoints are known
        # supersedes the legacy centroid field and starts from the exact
        # all-harmonic reference, avoiding compounded residuals.
        soft_translation_report = {
            "disabled": True,
            "reason": "station_soft_follow_authoritative",
        }
    else:
        soft_translation_report = _apply_soft_bone_translation_field(
            vertices,
            asset,
            vessel_nerve,
            blender_centroids=blender_bone_centroids,
            material_centroids=material_bone_centroids,
            valid_bones=bone_valid,
            driver_indices=fit_driver_indices,
            driver_weights=fit_driver_weights,
            subject_surface=subject_surface,
            subject_faces=surface_faces,
        )

    endpoints_delta = bone_delta
    head = np.asarray(
        asset.target_bone_head if asset.target_bone_head is not None else asset.source_bone_head,
        dtype=np.float64,
    )
    tail = np.asarray(
        asset.target_bone_tail if asset.target_bone_tail is not None else asset.source_bone_tail,
        dtype=np.float64,
    )
    new_head = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], head) + endpoints_delta[:, :3, 3]
    new_tail = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], tail) + endpoints_delta[:, :3, 3]
    for bone, mode in enumerate(modes):
        a = int(asset.source_bone_smplx_a[bone])
        b = int(asset.source_bone_smplx_b[bone])
        if mode in {"segment_root", "twist", "rigid_group"} and a != b:
            new_head[bone] = target_joints[a]
            new_tail[bone] = target_joints[b]
        elif _is_joint_local_mode(mode):
            joint_name = asset.joint_names[a]
            new_head[bone] = target_joints[a]
            if joint_name in {"left_wrist", "right_wrist"}:
                side = joint_name.split("_", 1)[0]
                index_name = f"{side}_index1"
                if index_name in asset.joint_names:
                    new_tail[bone] = target_joints[asset.joint_names.index(index_name)]
            else:
                child = _joint_child(a, asset.parents)
                if child is not None:
                    new_tail[bone] = target_joints[child]
    anchor_error = np.asarray(
        [
            np.linalg.norm(new_global[bone, :3, 3] - target_joints[int(asset.source_bone_smplx_a[bone])])
            for bone, mode in enumerate(modes)
            if not _is_follow_mode(mode)
        ],
        dtype=np.float64,
    )
    metadata = dict(asset.metadata or {})
    direct_hand_controllers = _direct_smplx_hand_controllers(asset)
    metadata["source_direct_driver_bones_v1"] = direct_hand_controllers
    history = list(metadata.get("articulated_rest_fit", []))
    report = {
        "stage": str(stage),
        "backend": "articulated_material_fit_v5",
        "shaft_meshes": int(shaft_meshes),
        "cranial_uniform_scale": float(cranial_scale),
        "cranial_scale_report": cranial_scale_report,
        "cranial_aspect_ratio_change": float(cranial_aspect_ratio_change),
        "brain_skull_center_drift_m": float(brain_skull_center_drift_m),
        "pelvis_uniform_scale": float(pelvis_scale),
        "pelvis_scale_report": pelvis_scale_report,
        "hip_geometry": hip_report,
        "pelvis_aspect_ratio_change": float(pelvis_aspect_ratio_change),
        "thorax_uniform_scale": float(thorax_scale),
        "thorax_axis_scale": thorax_axis_scale.tolist(),
        "long_bone_end_edge_change": float(protected_end_edge_change),
        "maximum_digit_rigid_offset_m": 0.0,
        "feet": foot_report,
        "foot_soft_follow": soft_follow_report,
        "scapula_thorax_fit": scapula_report,
        "spine_interfaces": spine_interface_report,
        "soft_section_inward": soft_section_report,
        "soft_bone_translation_field": soft_translation_report,
        "source_rig_rebind": rebind_report,
        "anchor_rms_m": float(np.sqrt(np.mean(anchor_error * anchor_error))) if len(anchor_error) else 0.0,
        "anchor_max_m": float(np.max(anchor_error)) if len(anchor_error) else 0.0,
        "direct_hand_controller_count": int(len(direct_hand_controllers)),
    }
    history.append(report)
    metadata["articulated_rest_fit"] = history
    result = type(asset)(
        **{
            **rebound.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "target_rest_global": new_global.astype(np.float32),
            "target_rest_local": new_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
            "target_bone_head": new_head.astype(np.float32),
            "target_bone_tail": new_tail.astype(np.float32),
            "driver_indices": fit_driver_indices,
            "driver_weights": fit_driver_weights,
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    if preserve_source_binding:
        if not np.array_equal(result.driver_indices, asset.driver_indices):
            raise RuntimeError("material fit changed Blender driver indices")
        if not np.array_equal(result.driver_weights, asset.driver_weights):
            raise RuntimeError("material fit changed Blender driver weights")
        if not np.array_equal(result.source_bone_parents, asset.source_bone_parents):
            raise RuntimeError("material fit changed Blender source hierarchy")
        if list(result.source_bone_driver_types or []) != list(
            asset.source_bone_driver_types or []
        ):
            raise RuntimeError("material fit changed source driver modes")
        report["v71_source_binding_preserved"] = True
        untouched_nonbone = (~bone_material) & ~_tissue_mask(asset, "vessel")
        report["v71_unfitted_nonbone_rest_vertices_preserved"] = bool(
            np.array_equal(
                np.asarray(result.vertices_rest)[untouched_nonbone],
                np.asarray(asset.vertices_rest)[untouched_nonbone],
            )
        )
        if not report["v71_unfitted_nonbone_rest_vertices_preserved"]:
            raise RuntimeError("material fit changed unfitted v71 non-bone geometry")
        report["v71_thin_anatomy_rest_transport"] = (
            "vessel_full_blender_weighted_source_bone_delta"
        )
    result.validate()
    return result, report

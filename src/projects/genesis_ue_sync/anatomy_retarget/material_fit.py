"""Shape-preserving articulated rest fitting for anatomy schema v6.

Rigid anatomy is fitted from semantic joints and material groups.  Soft
materials follow the finalized authored driver frames through their original
sparse weights.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms
from .rigged_asset import AnatomyRiggedAsset
from .anatomy_lbs import with_source_driver_coupling


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


def _load_obj_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            polygon = [
                int(value.split("/", 1)[0]) - 1
                for value in line.split()[1:]
            ]
            for index in range(1, len(polygon) - 1):
                faces.append([polygon[0], polygon[index], polygon[index + 1]])
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int32),
    )


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


def _principal_cap_centers(
    points: np.ndarray,
    *,
    reference_a: np.ndarray,
    reference_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract ordered epiphysis centers from authored geometry."""
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(source) < 8:
        return (
            np.asarray(reference_a, dtype=np.float64).reshape(3),
            np.asarray(reference_b, dtype=np.float64).reshape(3),
        )
    centered = source - np.mean(source, axis=0, keepdims=True)
    _u, _singular, vt = np.linalg.svd(centered, full_matrices=False)
    parameter = centered @ vt[0]
    low, high = np.quantile(parameter, (0.10, 0.90))
    low_center = np.mean(source[parameter <= low], axis=0)
    high_center = np.mean(source[parameter >= high], axis=0)
    a = np.asarray(reference_a, dtype=np.float64).reshape(3)
    b = np.asarray(reference_b, dtype=np.float64).reshape(3)
    forward = np.linalg.norm(low_center - a) + np.linalg.norm(high_center - b)
    reverse = np.linalg.norm(high_center - a) + np.linalg.norm(low_center - b)
    return (
        (low_center, high_center)
        if forward <= reverse
        else (high_center, low_center)
    )


def uniform_segment_similarity(
    points: np.ndarray,
    *,
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Uniformly fit a rigid compound between two anatomical landmarks."""
    source = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    source_a = np.asarray(source_a, dtype=np.float64).reshape(3)
    source_b = np.asarray(source_b, dtype=np.float64).reshape(3)
    target_a = np.asarray(target_a, dtype=np.float64).reshape(3)
    target_b = np.asarray(target_b, dtype=np.float64).reshape(3)
    source_vector = source_b - source_a
    target_vector = target_b - target_a
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length <= 1.0e-8 or target_length <= 1.0e-8:
        raise ValueError("uniform segment similarity requires nondegenerate landmarks")
    rotation = _rotation_between(source_vector, target_vector)
    scale = target_length / source_length
    mapped = target_a + scale * ((source - source_a) @ rotation.T)
    return mapped, float(scale), rotation


def _dominant_bone(asset: AnatomyRiggedAsset, start: int, stop: int) -> int | None:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        return None
    indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
    weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
    mass = np.bincount(indices, weights=weights, minlength=len(asset.source_bone_names))
    return int(np.argmax(mass)) if mass.size and float(mass.max()) > 0.0 else None


def _controller(bone: int, parents: np.ndarray, modes: list[str]) -> int:
    current = int(bone)
    while current >= 0 and modes[current] == "bind_follow":
        current = int(parents[current])
    return int(bone if current < 0 else current)


def _source_joint_anchors(asset: AnatomyRiggedAsset) -> np.ndarray:
    target = np.asarray(asset.rest_joints, dtype=np.float64)
    anchors = target.copy()
    assigned = np.zeros(len(target), dtype=bool)
    modes = list(asset.source_bone_driver_types or [])
    global_bind = np.asarray(asset.target_bind_global, dtype=np.float64)
    for bone, mode in enumerate(modes):
        if mode == "bind_follow":
            continue
        joint = int(asset.source_bone_smplx_a[bone])
        if not assigned[joint]:
            anchors[joint] = global_bind[bone, :3, 3]
            assigned[joint] = True
    return anchors


def _joint_child(joint: int, parents: np.ndarray) -> int | None:
    children = np.flatnonzero(np.asarray(parents, dtype=np.int64) == int(joint))
    return int(children[0]) if len(children) else None


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


def _anatomical_frame(
    *,
    origin: np.ndarray,
    lateral: np.ndarray,
    superior: np.ndarray,
) -> np.ndarray:
    """Build a right-handed local frame without assuming world up/forward."""
    # ``np.asarray`` may return a view into caller-owned landmark arrays.
    # Normalising such a view used to overwrite the measured hip span before
    # the pelvis similarity scale was computed.
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


def _fit_source_frames(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    old_local = np.asarray(asset.target_bind_local, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    source_anchors = _source_joint_anchors(asset)
    frame_joints = (
        np.asarray(asset.source_bone_frame_joints, dtype=np.int64)
        if asset.source_bone_frame_joints is not None
        else np.full((len(modes), 3), -1, dtype=np.int64)
    )
    new_global = np.empty_like(old_global)
    for bone, mode in enumerate(modes):
        parent = int(source_parents[bone])
        if mode == "bind_follow" and parent >= 0:
            new_global[bone] = new_global[parent] @ old_local[bone]
            continue
        a = int(asset.source_bone_smplx_a[bone])
        b = int(asset.source_bone_smplx_b[bone])
        if a == b and mode == "joint_local":
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
    if len(spine_chain) >= 2 and len(anchor_positions) >= 2:
        # Map the full authored pelvis→head arc onto a C1 monotone cubic.  Each
        # target control retains the exact authored bone's arc parameter.
        authored = old_global[spine_chain, :3, 3]
        lengths = np.linalg.norm(np.diff(authored, axis=0), axis=1)
        fractions = np.r_[0.0, np.cumsum(lengths)]
        fractions /= max(float(fractions[-1]), 1.0e-8)
        indexed_controls = sorted(
            (
                (spine_chain.index(bone), np.asarray(position, dtype=np.float64))
                for bone, position in anchor_positions
                if bone in chain_set
            ),
            key=lambda item: item[0],
        )
        control = np.stack([position for _index, position in indexed_controls], axis=0)
        control_fractions = fractions[
            np.asarray([index for index, _position in indexed_controls], dtype=np.int64)
        ]
        sampled = _sample_spine_centerline(
            control,
            fractions,
            control_fractions=control_fractions,
        )
        authored_tangents = np.gradient(authored, fractions, axis=0)
        target_tangents = np.gradient(sampled, fractions, axis=0)
        for index, bone in enumerate(spine_chain):
            source_tangent = authored_tangents[index]
            target_tangent = target_tangents[index]
            if (
                float(np.linalg.norm(target_tangent)) > 1.0e-8
                and float(np.linalg.norm(source_tangent)) > 1.0e-8
            ):
                new_global[bone, :3, :3] = (
                    _rotation_between(source_tangent, target_tangent)
                    @ old_global[bone, :3, :3]
                )
            else:
                new_global[bone, :3, :3] = old_global[bone, :3, :3]
            new_global[bone, :3, 3] = sampled[index]
    else:
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
        if mode == "bind_follow":
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
    driven = np.sum(weights * cranial_bone[indices], axis=1) >= 0.5
    # A number of whole-body nerve and vessel meshes inherit a Head_Bone
    # influence through their authored hierarchy.  They are continuous soft
    # material and must stay on the volume field; treating that influence as
    # cranial compound membership teleported the complete network when the
    # skull similarity was applied.
    explicit_compound = np.zeros(len(asset.vertices_rest), dtype=bool)
    compound_ids = getattr(asset, "source_compound_ids", None)
    if compound_ids is not None and asset.source_vertex_ranges is not None:
        for (start, stop), compound_id in zip(
            asset.source_vertex_ranges,
            compound_ids,
        ):
            if str(compound_id) == "cranial":
                explicit_compound[int(start) : int(stop)] = True
    # Bone descendants include upper teeth and other rigid skull pieces.  Soft
    # cranial membership is mesh-semantic and therefore never splits one organ
    # according to per-vertex weight noise.
    driven_bone = driven & _tissue_mask(asset, {"bone"})
    return explicit_compound | driven_bone


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
    driven = np.sum(weights * jaw_bone[indices], axis=1) >= 0.5
    return driven & _tissue_mask(asset, {"bone"})


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
) -> tuple[np.ndarray, float, dict[str, float | bool]]:
    """Uniform similarity about an explicit pivot.

    ``scale_mode`` chooses how axis extent ratios collapse to one isotropic
    scale.  ``min`` (legacy) actively shrinks to the tightest axis and caused
    cranial under-scale; ``median`` / ``mean`` follow PLAN §4.4/4.5.
    """
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
        base_scale = float(np.median(ratios))
    raw_scale = float(margin) * base_scale * float(scale_multiplier)
    lo = float(minimum_scale)
    hi = float(maximum_scale)
    saturated = bool(raw_scale < lo or raw_scale > hi)
    scale = max(lo, min(hi, raw_scale))
    mapped = resolved_target_center + scale * (source - resolved_source_center)
    return mapped, scale, {
        "base_scale": float(base_scale),
        "raw_scale": float(raw_scale),
        "scale": float(scale),
        "saturated": saturated,
        "scale_mode": mode,
    }


def _maximum_contained_similarity_scale(
    reference_points: np.ndarray,
    *,
    source_center: np.ndarray,
    target_center: np.ndarray,
    proposed_scale: float,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    clearance_m: float,
    optimize_translation: bool = False,
) -> tuple[float, np.ndarray]:
    """Largest isotropic scale whose reference geometry remains under skin."""
    import igl

    reference = np.asarray(reference_points, dtype=np.float64)
    source_origin = np.asarray(source_center, dtype=np.float64).reshape(3)
    target_origin = np.asarray(target_center, dtype=np.float64).reshape(3)
    surface = np.asarray(surface_vertices, dtype=np.float64)
    faces = np.asarray(surface_faces, dtype=np.int32)

    def distances(scale: float) -> tuple[np.ndarray, np.ndarray]:
        candidate = target_origin + float(scale) * (reference - source_origin)
        signed, _face, closest, _normal = igl.signed_distance(
            candidate,
            surface,
            faces,
        )
        values = np.asarray(signed, dtype=np.float64)
        return values, np.asarray(closest, dtype=np.float64)

    def feasible(scale: float) -> bool:
        values, _closest = distances(scale)
        return bool(
            np.all(np.isfinite(values))
            # A single facial suture/outlier vertex must not shrink the whole
            # cranial compound.  The 99.9th percentile still constrains all but
            # a handful of source vertices; the final signed-distance gate
            # reports any actual protrusion separately.
            and float(np.quantile(values, 0.999))
            <= -float(clearance_m)
        )

    upper = float(proposed_scale)
    if optimize_translation:
        for _iteration in range(16):
            values, closest = distances(upper)
            violation = values + float(clearance_m)
            active = violation > 0.0
            if not np.any(active):
                break
            candidate = target_origin + upper * (reference - source_origin)
            correction = closest[active] - candidate[active]
            weights = violation[active]
            shift = np.average(
                correction,
                axis=0,
                weights=np.maximum(weights, 1.0e-12),
            )
            if float(np.linalg.norm(shift)) <= 1.0e-7:
                break
            target_origin += 0.5 * shift
    if feasible(upper):
        return upper, target_origin
    lower = 0.0
    if not feasible(lower):
        raise RuntimeError("similarity target center is outside the SMPL-X skin")
    for _iteration in range(24):
        middle = 0.5 * (lower + upper)
        if feasible(middle):
            lower = middle
        else:
            upper = middle
    return float(lower), target_origin


def _maximum_contained_radial_scale(
    points: np.ndarray,
    *,
    axis_start: np.ndarray,
    axis_end: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, float]:
    import igl

    source = np.asarray(points, dtype=np.float64)
    start = np.asarray(axis_start, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_end, dtype=np.float64).reshape(3) - start
    length2 = float(axis @ axis)
    if length2 <= 1.0e-12:
        return source.copy(), 1.0
    parameter = np.clip(((source - start) @ axis) / length2, 0.0, 1.0)
    centerline = start + parameter[:, None] * axis
    radial = source - centerline

    def candidate(scale: float) -> np.ndarray:
        return centerline + float(scale) * radial

    def feasible(scale: float) -> bool:
        signed, _face, _closest, _normal = igl.signed_distance(
            candidate(scale),
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_faces, dtype=np.int32),
        )
        values = np.asarray(signed, dtype=np.float64)
        return bool(
            np.all(np.isfinite(values))
            and float(np.max(values)) <= -float(clearance_m)
        )

    if feasible(1.0):
        return source.copy(), 1.0
    if not feasible(0.0):
        return source.copy(), 1.0
    lower, upper = 0.0, 1.0
    for _iteration in range(24):
        middle = 0.5 * (lower + upper)
        if feasible(middle):
            lower = middle
        else:
            upper = middle
    return candidate(lower), float(lower)


def _contained_rigid_translation(
    points: np.ndarray,
    *,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    import igl

    source = np.asarray(points, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    for _iteration in range(24):
        candidate = source + translation
        signed, _face, closest, _normal = igl.signed_distance(
            candidate,
            np.asarray(surface_vertices, dtype=np.float64),
            np.asarray(surface_faces, dtype=np.int32),
        )
        values = np.asarray(signed, dtype=np.float64)
        violation = values + float(clearance_m)
        active = violation > 0.0
        if not np.any(active):
            return candidate, translation
        correction = np.asarray(closest, dtype=np.float64)[active] - candidate[active]
        step = np.average(
            correction,
            axis=0,
            weights=np.maximum(violation[active], 1.0e-12),
        )
        if float(np.linalg.norm(step)) <= 1.0e-8:
            break
        translation += 0.5 * step
    return source + translation, translation


def _sample_spine_centerline(
    control_points: np.ndarray,
    fractions: np.ndarray,
    *,
    control_fractions: np.ndarray | None = None,
) -> np.ndarray:
    """Sample an order-preserving cubic through anatomically indexed anchors.

    ``control_fractions`` locates each SMPL-X control on the authored spine
    chain.  Treating controls as uniformly/chord spaced loses the L5, L2 and T8
    correspondence and can leave the lumbar chain detached from the sacrum.
    """
    controls = np.asarray(control_points, dtype=np.float64).reshape(-1, 3)
    query = np.clip(np.asarray(fractions, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if len(controls) == 0:
        raise ValueError("spine centerline requires control points")
    if len(controls) == 1:
        return np.repeat(controls, len(query), axis=0)
    if len(controls) == 2:
        return (1.0 - query)[:, None] * controls[0] + query[:, None] * controls[1]
    if control_fractions is None:
        chords = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        param = np.r_[0.0, np.cumsum(np.maximum(chords, 1.0e-8))]
        param = param / max(float(param[-1]), 1.0e-8)
    else:
        param = np.asarray(control_fractions, dtype=np.float64).reshape(-1)
        if len(param) != len(controls):
            raise ValueError("control_fractions must match control_points")
        if np.any(~np.isfinite(param)) or np.any(np.diff(param) <= 0.0):
            raise ValueError("control_fractions must be finite and strictly increasing")
        param = (param - param[0]) / max(float(param[-1] - param[0]), 1.0e-8)
    for index in range(1, len(param)):
        if param[index] <= param[index - 1]:
            param[index] = param[index - 1] + 1.0e-6
    param = param / param[-1]
    try:
        from scipy.interpolate import PchipInterpolator

        # PCHIP avoids the overshoot of an unconstrained natural cubic while
        # retaining C1 continuity and exact interpolation of every anchor.
        spline = PchipInterpolator(param, controls, axis=0)
        return np.asarray(spline(query), dtype=np.float64)
    except Exception:
        out = np.empty((len(query), 3), dtype=np.float64)
        for i, value in enumerate(query.tolist()):
            right = int(np.searchsorted(param, value, side="right"))
            left = max(0, right - 1)
            right = min(len(controls) - 1, max(right, left + 1))
            span = max(float(param[right] - param[left]), 1.0e-8)
            alpha = (value - float(param[left])) / span
            out[i] = (1.0 - alpha) * controls[left] + alpha * controls[right]
        return out


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
    target_lateral = target_joints[right_id] - target_joints[left_id]
    target_lateral /= max(float(np.linalg.norm(target_lateral)), 1.0e-8)
    target_eye_mid = 0.5 * (target_joints[left_id] + target_joints[right_id])
    source_lo, source_hi = np.quantile(reference_points, (0.01, 0.99), axis=0)
    target_lo, target_hi = np.quantile(target_points, (0.01, 0.99), axis=0)
    source_aabb = 0.5 * (source_lo + source_hi)
    target_aabb = 0.5 * (target_lo + target_hi) + center_offset
    # Source eye joints are often unassigned in the authored rig; the previous
    # fallback copied target eye joints and made the source/target comparison
    # self-referential.  Use the fitted skull geometry as the source pivot.
    source_center = source_aabb
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
    alpha: float = 1.0,
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
    lbs_target = np.matmul(blended, homogeneous[..., None])[:, :3, 0]
    blend = float(np.clip(alpha, 0.0, 1.0))
    if blend >= 1.0 - 1.0e-12:
        vertices[soft_material] = lbs_target
    else:
        vertices[soft_material] = (1.0 - blend) * vertices[soft_material] + blend * lbs_target


def _attach_soft_by_bone_translation(
    vertices: np.ndarray,
    soft_material: np.ndarray,
    *,
    driver_indices: np.ndarray,
    driver_weights: np.ndarray,
    old_global: np.ndarray,
    new_global: np.ndarray,
    alpha: float = 1.0,
    bone_mask: set[int] | None = None,
) -> None:
    """Pull soft tissue with bone-origin translations only (no rotational shear)."""
    if not np.any(soft_material):
        return
    blend = float(np.clip(alpha, 0.0, 1.0))
    if blend <= 1.0e-12:
        return
    indices = np.asarray(driver_indices, dtype=np.int64)[soft_material]
    weights = np.asarray(driver_weights, dtype=np.float64)[soft_material]
    bone_shift = (
        np.asarray(new_global, dtype=np.float64)[:, :3, 3]
        - np.asarray(old_global, dtype=np.float64)[:, :3, 3]
    )
    if bone_mask is not None:
        keep = np.zeros(len(bone_shift), dtype=bool)
        for bone in bone_mask:
            if 0 <= int(bone) < len(keep):
                keep[int(bone)] = True
        bone_shift = np.where(keep[:, None], bone_shift, 0.0)
    if indices.ndim == 1:
        shift = bone_shift[indices]
    else:
        shift = np.sum(bone_shift[indices] * weights[..., None], axis=1)
    vertices[soft_material] = vertices[soft_material] + blend * shift


def _tissue_mask(asset: AnatomyRiggedAsset, tissues: set[str]) -> np.ndarray:
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    allowed = {str(t).lower() for t in tissues}
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue).lower() in allowed:
            mask[int(start) : int(stop)] = True
    return mask


def _soft_material_mask(asset: AnatomyRiggedAsset) -> np.ndarray:
    mask = np.zeros(len(asset.vertices_rest), dtype=bool)
    if asset.source_vertex_ranges is None or asset.source_tissues is None:
        return mask
    for (start, stop), tissue in zip(asset.source_vertex_ranges, asset.source_tissues):
        if str(tissue).lower() != "bone":
            mask[int(start) : int(stop)] = True
    return mask


def _robust_sphere_center(points: np.ndarray) -> np.ndarray:
    """Approximate sphere center via algebraic fit; fall back to centroid."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 8:
        return np.mean(pts, axis=0) if len(pts) else np.zeros(3)
    centered = pts - np.mean(pts, axis=0)
    # Drop outliers beyond 2.5 median radii for femoral-head robustness.
    radii = np.linalg.norm(centered, axis=1)
    med = float(np.median(radii))
    keep = radii <= max(2.5 * med, 1.0e-4)
    pts = pts[keep] if np.count_nonzero(keep) >= 8 else pts
    a = np.concatenate((2.0 * pts, np.ones((len(pts), 1))), axis=1)
    b = np.sum(pts * pts, axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        return sol[:3]
    except np.linalg.LinAlgError:
        return np.mean(pts, axis=0)


def _femur_head_and_acetabulum(
    asset: AnatomyRiggedAsset,
    vertices: np.ndarray,
    *,
    side: str,
    target_joints: np.ndarray,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (femoral_head_center, acetabulum_center) in current vertex space."""
    suffix = "_l" if side == "left" else "_r"
    hip = target_joints[joint_names.index(f"{side}_hip")]
    knee = target_joints[joint_names.index(f"{side}_knee")]
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
        and any(token in name for token in ("ilium", "ischium", "pubis", "acetabul", "pelvis", "sacrum")),
    )
    if not np.any(femur) or not np.any(pelvis):
        return None
    femur_pts = vertices[femur]
    # Proximal head candidates along the hip←knee axis.
    param = (femur_pts - knee) @ axis
    hi = float(np.quantile(param, 0.85))
    head_pts = femur_pts[param >= hi]
    if len(head_pts) < 16:
        head_pts = femur_pts[np.argsort(-param)[: max(32, len(femur_pts) // 10)]]
    head = _robust_sphere_center(head_pts)
    pelvis_pts = vertices[pelvis]
    # Socket is defined from the pelvis relative to the SMPL-X hip controller,
    # not relative to the current femoral head — otherwise snap→remeasure is a
    # moving target and residual gaps stay at several millimetres.
    dist = np.linalg.norm(pelvis_pts - hip, axis=1)
    near = pelvis_pts[dist <= max(float(np.quantile(dist, 0.08)), 0.025)]
    if len(near) < 8:
        near = pelvis_pts[np.argsort(dist)[:32]]
    socket = 0.70 * np.mean(near, axis=0) + 0.30 * hip
    return head, socket


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
        a_name, b_name = f"{side}_wrist", f"{side}_{finger}1"
    elif "proximal" in lower:
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
            # Keep a fixed metric clearance from the skin front.  A fractional
            # 0.95 multiplier makes the result depend on finger length and was
            # effectively tuning the geometry to the old reach gate.
            result[(side, finger)] = j3 + max(0.0, reach - 0.0015) * axis
    return result


def _override(config: dict[str, Any], group: str) -> tuple[float, np.ndarray]:
    section = dict((config.get("fit_overrides", {}) or {}).get(group, {}) or {})
    scale = float(section.get("scale_multiplier", 1.0))
    offset = np.asarray(section.get("center_offset_local_m", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    return scale, offset


def fit_articulated_rest(
    asset: AnatomyRiggedAsset,
    *,
    canonical_dir: Path | str,
    config: dict[str, Any] | None = None,
    subject: bool,
    stage: str,
) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Fit rigid anatomy, rebind source frames, then transport soft tissue once."""
    asset.validate()
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
                if modes[bi] != "bind_follow":
                    modes[bi] = "bind_follow"
                    patched = True
        if patched:
            asset = type(asset)(**{**asset.__dict__, "source_bone_driver_types": modes})
    old_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    vertices = old_vertices.copy()
    old_global = np.asarray(asset.target_bind_global, dtype=np.float64)
    new_global, new_local, bone_delta = _fit_source_frames(asset)
    source_anchors = _source_joint_anchors(asset)
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
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
            if "scapula" in lower or "clavicle" in lower:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[bone]
                )
                continue
            hand_segment = _hand_mesh_segment(
                str(name),
                joint_names=asset.joint_names,
                source_anchors=source_anchors,
                target_joints=target_joints,
                finger_tips=finger_tips,
            )
            if "1st_metacarpal" in lower:
                # Authored thumb opposition: SMPL-X thumb1 is not collinear with
                # the palm rays, so a straight shaft map rotates through skin.
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[bone]
                )
                continue
            if hand_segment is not None:
                source_a, source_b, target_a, target_b = hand_segment
                source_a, source_b = _principal_cap_centers(
                    old_vertices[start_i:stop_i],
                    reference_a=source_a,
                    reference_b=source_b,
                )
                fitted = shaft_preserving_segment_map(
                    old_vertices[start_i:stop_i],
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
            if a == b and modes[control] == "joint_local":
                child = _joint_child(a, asset.parents)
                if child is not None:
                    b = child
            if a != b and any(token in lower for token in _LONG_BONE_TOKENS):
                geometry_reference = old_vertices[start_i:stop_i]
                paired_tokens: tuple[str, ...] | None = None
                if "radius" in lower or "ulna" in lower:
                    paired_tokens = ("radius", "ulna")
                elif "tibia" in lower or "fibula" in lower:
                    paired_tokens = ("tibia", "fibula")
                if paired_tokens is not None:
                    side = (
                        "left"
                        if lower.endswith("_l") or "_l_" in lower
                        else (
                            "right"
                            if lower.endswith("_r") or "_r_" in lower
                            else ""
                        )
                    )
                    paired_mask = _mesh_mask(
                        asset,
                        lambda mesh_name, mesh_tissue: (
                            mesh_tissue == "bone"
                            and any(
                                token in mesh_name
                                for token in paired_tokens or ()
                            )
                            and (
                                not side
                                or (
                                    side == "left"
                                    and (
                                        mesh_name.endswith("_l")
                                        or "_l_" in mesh_name
                                    )
                                )
                                or (
                                    side == "right"
                                    and (
                                        mesh_name.endswith("_r")
                                        or "_r_" in mesh_name
                                    )
                                )
                            )
                        ),
                    )
                    if np.any(paired_mask):
                        geometry_reference = old_vertices[paired_mask]
                geometry_a, geometry_b = _principal_cap_centers(
                    geometry_reference,
                    reference_a=source_anchors[a],
                    reference_b=source_anchors[b],
                )
                fitted = shaft_preserving_segment_map(
                    old_vertices[start_i:stop_i],
                    source_a=geometry_a,
                    source_b=geometry_b,
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
                        source_a=geometry_a,
                        source_b=geometry_b,
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
    cranial_scale = 1.0
    cranial_scale_report: dict[str, float | bool | str] = {
        "saturated": False,
        "raw_scale": 1.0,
        "scale": 1.0,
        "scale_mode": "median",
    }
    cranial_aspect_ratio_change = 0.0
    brain_skull_center_drift_m = 0.0
    cranial_envelope_center_before: np.ndarray | None = None
    cranial_envelope_center_after: np.ndarray | None = None
    cranial_soft_moved = np.zeros(len(vertices), dtype=bool)
    if np.any(cranial):
        source_names = list(asset.source_bone_names or [])
        if "Head_Bone" in source_names:
            vertices[cranial] = _transform_points(
                old_vertices[cranial], bone_delta[source_names.index("Head_Bone")]
            )
        old_cranial = vertices[cranial].copy()
        old_skull_center = (
            np.mean(vertices[skull_reference], axis=0)
            if np.any(skull_reference)
            else np.mean(old_cranial, axis=0)
        )
        old_brain_center = (
            np.mean(vertices[cranial & ~skull_reference], axis=0)
            if np.any(cranial & ~skull_reference)
            else old_skull_center
        )
        target_head = _surface_region(
            root,
            asset.joint_names,
            ("head", "left_eye_smplhf", "right_eye_smplhf"),
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
        cranial_reference = (
            vertices[skull_reference] if np.any(skull_reference) else vertices[cranial]
        )
        # PLAN §4.4: pivot on skull centroid / eye midline, never the SMPL-X
        # head joint.  Median axis ratio avoids the legacy min*0.96 shrink.
        cranial_envelope_center_before, cranial_envelope_center_after = _midline_envelope_centers(
            reference_points=cranial_reference,
            target_points=target_head,
            source_anchors=source_anchors,
            target_joints=target_joints,
            joint_names=list(asset.joint_names),
            center_offset=offset_world,
        )
        vertices[cranial], cranial_scale, cranial_scale_report = _uniform_envelope_fit(
            vertices[cranial],
            target_head,
            reference_points=cranial_reference,
            scale_multiplier=multiplier,
            center_offset=offset_world,
            margin=1.0,
            maximum_scale=10.0,
            minimum_scale=0.1,
            scale_mode="median",
            source_center=cranial_envelope_center_before,
            target_center=cranial_envelope_center_after,
        )
        surface_vertices, surface_faces = _load_obj_surface(
            root
            / (
                "smpl_canonical_tpose.obj"
                if subject
                else "smpl_canonical_tpose_neutral.obj"
            )
        )
        contained_scale, contained_center = _maximum_contained_similarity_scale(
            cranial_reference,
            source_center=cranial_envelope_center_before,
            target_center=cranial_envelope_center_after,
            proposed_scale=cranial_scale,
            surface_vertices=surface_vertices,
            surface_faces=surface_faces,
            clearance_m=0.001,
            optimize_translation=True,
        )
        center_shift = float(
            np.linalg.norm(contained_center - cranial_envelope_center_after)
        )
        if contained_scale < cranial_scale or center_shift > 1.0e-9:
            cranial_envelope_center_after = contained_center
            vertices[cranial] = contained_center + contained_scale * (
                old_cranial - cranial_envelope_center_before
            )
            cranial_scale = contained_scale
            cranial_scale_report = {
                **cranial_scale_report,
                "scale": float(cranial_scale),
                "surface_constrained": True,
                "clearance_m": 0.001,
                "center_shift_m": center_shift,
            }
        else:
            cranial_scale_report = {
                **cranial_scale_report,
                "surface_constrained": False,
                "clearance_m": 0.001,
                "center_shift_m": center_shift,
            }
        if "Head_Bone" in source_names:
            head_bone = source_names.index("Head_Bone")
            jaw_bone = source_names.index("Jaw_Bone_tip") if "Jaw_Bone_tip" in source_names else -1
            for bone in range(len(source_names)):
                current = bone
                follows_head = False
                follows_jaw = False
                while current >= 0:
                    follows_head = follows_head or current == head_bone
                    follows_jaw = follows_jaw or (jaw_bone >= 0 and current == jaw_bone)
                    current = int(source_parents[current])
                if not follows_head or follows_jaw:
                    continue
                new_global[bone, :3, 3] = cranial_envelope_center_after + cranial_scale * (
                    new_global[bone, :3, 3] - cranial_envelope_center_before
                )
        cranial_soft_moved = cranial & ~bone_material
        cranial_aspect_ratio_change = _aspect_ratio_change(old_cranial, vertices[cranial])
        new_skull_center = (
            np.mean(vertices[skull_reference], axis=0)
            if np.any(skull_reference)
            else np.mean(vertices[cranial], axis=0)
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

    if np.any(jaw):
        source_names = list(asset.source_bone_names or [])
        if "Jaw_Bone_tip" in source_names:
            jaw_base = _transform_points(
                old_vertices[jaw], bone_delta[source_names.index("Jaw_Bone_tip")]
            )
            if cranial_envelope_center_before is not None and cranial_envelope_center_after is not None:
                vertices[jaw] = cranial_envelope_center_after + cranial_scale * (
                    jaw_base - cranial_envelope_center_before
                )
                jaw_root = source_names.index("Jaw_Bone_tip")
                for bone in range(len(source_names)):
                    current = bone
                    follows_jaw = False
                    while current >= 0:
                        follows_jaw = follows_jaw or current == jaw_root
                        current = int(source_parents[current])
                    if follows_jaw:
                        new_global[bone, :3, 3] = (
                            cranial_envelope_center_after
                            + cranial_scale
                            * (new_global[bone, :3, 3] - cranial_envelope_center_before)
                        )
            else:
                vertices[jaw] = jaw_base

    pelvis = pelvis_material
    pelvis_scale = 1.0
    pelvis_aspect_ratio_change = 0.0
    pelvis_scale_report: dict[str, float | bool | str] = {
        "saturated": False,
        "raw_scale": 1.0,
        "scale": 1.0,
        "scale_mode": "median",
    }
    if np.any(pelvis):
        # A single anatomical Sim(3) for the complete pelvic compound.  Its
        # origin is the bilateral hip midpoint (not the rig root), and its
        # forward axis follows from the lateral/superior landmark plane.
        old_pelvis = old_vertices[pelvis].copy()
        multiplier, local_offset = _override(cfg, "pelvis")
        left_hip_id = asset.joint_names.index("left_hip")
        right_hip_id = asset.joint_names.index("right_hip")
        spine1_id = asset.joint_names.index("spine1")
        source_axis = source_anchors[right_hip_id] - source_anchors[left_hip_id]
        target_axis = target_joints[right_hip_id] - target_joints[left_hip_id]
        source_origin = 0.5 * (
            source_anchors[left_hip_id] + source_anchors[right_hip_id]
        )
        target_origin = 0.5 * (
            target_joints[left_hip_id] + target_joints[right_hip_id]
        )
        source_frame = _anatomical_frame(
            origin=source_origin,
            lateral=source_axis,
            superior=source_anchors[spine1_id] - source_origin,
        )
        target_frame = _anatomical_frame(
            origin=target_origin,
            lateral=target_axis,
            superior=target_joints[spine1_id] - target_origin,
        )
        source_span = float(np.linalg.norm(source_axis))
        hip_span = float(np.linalg.norm(target_axis))
        raw_scale = float(multiplier) * hip_span / max(source_span, 1.0e-6)
        if not np.isfinite(raw_scale) or raw_scale <= 0.0:
            raise ValueError(f"invalid pelvis similarity scale: {raw_scale}")
        pelvis_scale = raw_scale
        local = _frame_coordinates(old_vertices[pelvis], source_frame)
        target_frame = target_frame.copy()
        target_frame[:3, 3] += target_frame[:3, :3] @ local_offset
        vertices[pelvis] = _from_frame_coordinates(pelvis_scale * local, target_frame)
        pelvis_rotation = target_frame[:3, :3] @ source_frame[:3, :3].T
        for bone, name in enumerate(asset.source_bone_names or []):
            lower = str(name).lower()
            if not (
                any(token in lower for token in _PELVIS_TOKENS)
                or "hip_organ_hold" in lower
            ):
                continue
            bind_local = _frame_coordinates(
                old_global[bone, :3, 3][None, :], source_frame
            )[0]
            new_global[bone, :3, 3] = _from_frame_coordinates(
                (pelvis_scale * bind_local)[None, :], target_frame
            )[0]
            new_global[bone, :3, :3] = pelvis_rotation @ old_global[bone, :3, :3]
        pelvis_scale_report = {
            "base_scale": float(hip_span / max(source_span, 1.0e-6)),
            "raw_scale": float(raw_scale),
            "scale": float(pelvis_scale),
            "saturated": False,
            "scale_mode": "bilateral_hip_similarity",
            "source_hip_span_m": source_span,
            "target_hip_span_m": hip_span,
        }
        pelvis_aspect_ratio_change = _aspect_ratio_change(old_pelvis, vertices[pelvis])

    hip_report: dict[str, Any] = {}
    for side in ("left", "right"):
        suffix = "_l" if side == "left" else "_r"
        femur = _mesh_mask(
            asset,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and "femur" in name
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        pair = _femur_head_and_acetabulum(
            asset,
            vertices,
            side=side,
            target_joints=target_joints,
            joint_names=list(asset.joint_names),
        )
        if pair is None or not np.any(femur):
            continue
        head, socket = pair
        pre_err = float(np.linalg.norm(socket - head))
        # Solve proximal and distal constraints together.  Translating the
        # complete femur to close the socket also translated the knee away from
        # its controller.  The shaft map keeps both epiphyses rigid and absorbs
        # the length change only in the diaphysis.
        knee = target_joints[asset.joint_names.index(f"{side}_knee")]
        femur_points = vertices[femur].copy()
        shaft_axis = knee - head
        shaft_axis /= max(float(np.linalg.norm(shaft_axis)), 1.0e-8)
        axial = (femur_points - head) @ shaft_axis
        distal_points = femur_points[axial >= float(np.quantile(axial, 0.85))]
        distal = (
            np.mean(distal_points, axis=0)
            if len(distal_points)
            else femur_points[int(np.argmax(axial))]
        )
        vertices[femur] = shaft_preserving_segment_map(
            femur_points,
            source_a=head,
            source_b=distal,
            target_a=socket,
            target_b=knee,
        )
        frame_rotation = _rotation_between(distal - head, knee - socket)
        frame_delta = np.eye(4, dtype=np.float64)
        frame_delta[:3, :3] = frame_rotation
        frame_delta[:3, 3] = socket - frame_rotation @ head
        for bone, name in enumerate(asset.source_bone_names or []):
            lower = str(name).lower()
            if "femur" in lower and (lower.endswith(suffix) or f"{suffix}_" in lower):
                new_global[bone] = frame_delta @ new_global[bone]
        pair_after = _femur_head_and_acetabulum(
            asset,
            vertices,
            side=side,
            target_joints=target_joints,
            joint_names=list(asset.joint_names),
        )
        post_err = (
            float(np.linalg.norm(pair_after[0] - pair_after[1]))
            if pair_after is not None
            else pre_err
        )
        hip_report[side] = {
            "femoral_head_to_acetabulum_m": post_err,
            "pre_correct_gap_m": pre_err,
            "proximal_delta_m": (socket - head).tolist(),
            "distal_to_knee_m": float(
                np.linalg.norm(
                    np.mean(
                        vertices[femur][
                            ((vertices[femur] - socket) @ (knee - socket))
                            >= float(
                                np.quantile(
                                    (vertices[femur] - socket) @ (knee - socket),
                                    0.85,
                                )
                            )
                        ],
                        axis=0,
                    )
                    - knee
                )
            ),
        }

    thorax = thorax_material
    thorax_scale = 1.0
    thorax_local_frame_scale = np.ones(3, dtype=np.float64)
    if np.any(thorax):
        # Measure beta-induced torso change against the neutral SMPL-X surface,
        # then apply it in a target anatomical frame about each rib-head bind.
        # This preserves the vertebral attachment and avoids world X/Z scaling.
        torso_subject = _surface_region(
            root,
            asset.joint_names,
            ("spine1", "spine2", "spine3", "left_shoulder", "right_shoulder"),
            subject=subject,
        )
        torso_neutral = _surface_region(
            root,
            asset.joint_names,
            ("spine1", "spine2", "spine3", "left_shoulder", "right_shoulder"),
            subject=False,
        )
        left_hip = target_joints[asset.joint_names.index("left_hip")]
        right_hip = target_joints[asset.joint_names.index("right_hip")]
        spine1 = target_joints[asset.joint_names.index("spine1")]
        spine2 = target_joints[asset.joint_names.index("spine2")]
        spine3 = target_joints[asset.joint_names.index("spine3")]
        thorax_frame = _anatomical_frame(
            origin=spine2,
            lateral=right_hip - left_hip,
            superior=spine3 - spine1,
        )
        subject_local = _frame_coordinates(torso_subject, thorax_frame)
        neutral_local = _frame_coordinates(torso_neutral, thorax_frame)
        subject_lo, subject_hi = np.quantile(subject_local, (0.05, 0.95), axis=0)
        neutral_lo, neutral_hi = np.quantile(neutral_local, (0.05, 0.95), axis=0)
        subject_extent = subject_hi - subject_lo
        neutral_extent = neutral_hi - neutral_lo
        sx = float(subject_extent[0] / max(float(neutral_extent[0]), 1.0e-6))
        sz = float(subject_extent[2] / max(float(neutral_extent[2]), 1.0e-6))
        if not np.isfinite(sx) or not np.isfinite(sz) or sx <= 0.0 or sz <= 0.0:
            raise ValueError(f"invalid thorax local scale: {(sx, sz)}")
        for (start, stop), tissue, name in zip(
            asset.source_vertex_ranges, asset.source_tissues, asset.source_mesh_names
        ):
            start_i, stop_i = int(start), int(stop)
            if str(tissue).lower() != "bone" or not np.any(thorax[start_i:stop_i]):
                continue
            controller = _dominant_bone(asset, start_i, stop_i)
            if controller is not None and subject:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[controller]
                )
            if controller is None:
                continue
            if not subject:
                # The globally aligned source ribs/sternum are already inside
                # the neutral skin.  Re-applying descendant bind-follow
                # translations moved the sternum 8 cm anteriorly and detached
                # rib heads.  Preserve their authored neutral geometry/pivots;
                # their target locals are recomputed against the fitted spine.
                vertices[start_i:stop_i] = old_vertices[start_i:stop_i]
                new_global[controller] = old_global[controller]
                continue
            pts = vertices[start_i:stop_i]
            pivot_local = _frame_coordinates(
                new_global[controller, :3, 3][None, :], thorax_frame
            )[0]
            local = _frame_coordinates(pts, thorax_frame) - pivot_local
            local[:, 0] *= sx
            local[:, 2] *= sz
            vertices[start_i:stop_i] = _from_frame_coordinates(
                local + pivot_local, thorax_frame
            )
        if not subject:
            for bone, bone_name in enumerate(asset.source_bone_names or []):
                lower = str(bone_name).lower()
                if (
                    lower.startswith("rib_bone_")
                    or lower.startswith("rib_name_")
                    or "sternum" in lower
                ):
                    new_global[bone] = old_global[bone]
        thorax_scale = float(np.sqrt(sx * sz))
        thorax_local_frame_scale = np.asarray((sx, 1.0, sz), dtype=np.float64)

    foot_report: dict[str, Any] = {}
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
        ankle_id = asset.joint_names.index(f"{side}_ankle")
        foot_id = asset.joint_names.index(f"{side}_foot")
        source_ankle = source_anchors[ankle_id]
        source_forward = source_anchors[foot_id] - source_ankle
        target_ankle = target_joints[ankle_id]
        target_forward = target_joints[foot_id] - target_ankle
        source_length = float(np.linalg.norm(source_forward))
        target_length = float(np.linalg.norm(target_forward))
        target_foot = _surface_region(
            root,
            asset.joint_names,
            (f"{side}_ankle", f"{side}_foot"),
            subject=subject,
        )
        source_direction = source_forward / max(source_length, 1.0e-8)
        target_direction = target_forward / max(target_length, 1.0e-8)
        source_reach = float(
            np.quantile(
                (old_vertices[foot] - source_ankle) @ source_direction,
                0.995,
            )
        )
        skin_reach = float(
            np.quantile(
                (target_foot - target_ankle) @ target_direction,
                0.995,
            )
        )
        if (
            not np.isfinite(source_reach)
            or not np.isfinite(skin_reach)
            or source_reach <= 1.0e-6
            or skin_reach <= 1.0e-6
        ):
            raise ValueError(
                f"{side} foot has invalid geometry reach "
                f"(source={source_reach}, target={skin_reach})"
            )
        geometry_scale = skin_reach / source_reach
        knee_id = asset.joint_names.index(f"{side}_knee")
        source_lateral = (
            source_anchors[asset.joint_names.index("right_ankle")]
            - source_anchors[asset.joint_names.index("left_ankle")]
        )
        target_lateral = (
            target_joints[asset.joint_names.index("right_ankle")]
            - target_joints[asset.joint_names.index("left_ankle")]
        )
        source_foot_frame = _anatomical_frame(
            origin=source_ankle,
            lateral=source_lateral,
            superior=source_anchors[knee_id] - source_ankle,
        )
        target_foot_frame = _anatomical_frame(
            origin=target_ankle,
            lateral=target_lateral,
            superior=target_joints[knee_id] - target_ankle,
        )
        foot_rotation = (
            target_foot_frame[:3, :3] @ source_foot_frame[:3, :3].T
        )
        rotated_foot = source_ankle + (
            old_vertices[foot] - source_ankle
        ) @ foot_rotation.T
        body_vertices, body_faces = _load_obj_surface(
            root
            / (
                "smpl_canonical_tpose.obj"
                if subject
                else "smpl_canonical_tpose_neutral.obj"
            )
        )
        geometry_scale, _contained_foot_center = _maximum_contained_similarity_scale(
            rotated_foot,
            source_center=source_ankle,
            target_center=target_ankle,
            proposed_scale=geometry_scale,
            surface_vertices=body_vertices,
            surface_faces=body_faces,
            clearance_m=0.001,
        )
        scale = float(geometry_scale)
        rotation = foot_rotation
        vertices[foot] = target_ankle + scale * (
            old_vertices[foot] - source_ankle
        ) @ rotation.T

        # Scale and rotate every authored foot bind around the same ankle pivot.
        # Toes remain bind-follow because SMPL-X has no independent toe chain.
        for bone, name in enumerate(asset.source_bone_names or []):
            lower = str(name).lower()
            if not (
                any(token in lower for token in _FOOT_TOKENS)
                and (lower.endswith(suffix) or f"{suffix}_" in lower)
            ):
                continue
            new_global[bone, :3, :3] = rotation @ old_global[bone, :3, :3]
            new_global[bone, :3, 3] = target_ankle + scale * (
                rotation @ (old_global[bone, :3, 3] - source_ankle)
            )

        fitted_reach = float(
            np.quantile(
                (vertices[foot] - target_ankle) @ target_direction,
                0.995,
            )
        )
        foot_report[side] = {
            "uniform_scale": float(scale),
            "source_reach_m": source_reach,
            "fitted_reach_m": fitted_reach,
            "skin_reach_m": skin_reach,
            "reach_ratio": fitted_reach / max(skin_reach, 1.0e-8),
            "fit_policy": "ankle_foot_similarity_compound",
            "post_projection_applied": False,
        }

    body_vertices, body_faces = _load_obj_surface(
        root
        / (
            "smpl_canonical_tpose.obj"
            if subject
            else "smpl_canonical_tpose_neutral.obj"
        )
    )
    bone_surface_constraints: dict[str, Any] = {}
    if asset.source_vertex_ranges is not None:
        import igl

        for (start, stop), mesh_name, tissue in zip(
            asset.source_vertex_ranges,
            asset.source_mesh_names,
            asset.source_tissues,
        ):
            if str(tissue) != "bone":
                continue
            start_i, stop_i = int(start), int(stop)
            points = vertices[start_i:stop_i]
            signed, _face, _closest, _normal = igl.signed_distance(
                points,
                body_vertices,
                body_faces,
            )
            maximum = float(np.max(np.asarray(signed, dtype=np.float64)))
            if maximum <= -0.001:
                continue
            bone = _dominant_bone(asset, start_i, stop_i)
            if bone is None:
                continue
            control = _controller(bone, source_parents, modes)
            a = int(asset.source_bone_smplx_a[control])
            b = int(asset.source_bone_smplx_b[control])
            if a == b:
                child = _joint_child(a, asset.parents)
                if child is not None:
                    b = child
            lower = str(mesh_name).lower()
            if a != b and any(
                token in lower
                for token in (
                    "humerus",
                    "radius",
                    "ulna",
                    "femur",
                    "tibia",
                    "fibula",
                    "metacarpal",
                    "phalang",
                )
            ):
                fitted, radial_scale = _maximum_contained_radial_scale(
                    points,
                    axis_start=target_joints[a],
                    axis_end=target_joints[b],
                    surface_vertices=body_vertices,
                    surface_faces=body_faces,
                    clearance_m=0.001,
                )
                vertices[start_i:stop_i] = fitted
                bone_surface_constraints[str(mesh_name)] = {
                    "policy": "joint_axis_radial_similarity",
                    "radial_scale": float(radial_scale),
                    "initial_max_outside_m": maximum,
                }
            elif "scapula" in lower:
                fitted, translation = _contained_rigid_translation(
                    points,
                    surface_vertices=body_vertices,
                    surface_faces=body_faces,
                    clearance_m=0.001,
                )
                vertices[start_i:stop_i] = fitted
                new_global[bone, :3, 3] += translation
                bone_surface_constraints[str(mesh_name)] = {
                    "policy": "rigid_clearance_translation",
                    "translation_m": translation.tolist(),
                    "initial_max_outside_m": maximum,
                }

    from scipy.spatial import cKDTree

    mesh_range = {
        str(name): tuple(int(value) for value in start_stop)
        for name, start_stop in zip(
            asset.source_mesh_names,
            np.asarray(asset.source_vertex_ranges, dtype=np.int64),
        )
    }
    for side, suffix in (("left", "_L"), ("right", "_R")):
        scapula_name = f"Scapula{suffix}"
        humerus_name = f"Humerus{suffix}"
        if scapula_name not in mesh_range or humerus_name not in mesh_range:
            continue
        shoulder = target_joints[
            asset.joint_names.index(f"{side}_shoulder")
        ]
        scapula_start, scapula_stop = mesh_range[scapula_name]
        humerus_start, humerus_stop = mesh_range[humerus_name]
        scapula_indices = np.arange(scapula_start, scapula_stop)
        humerus_indices = np.arange(humerus_start, humerus_stop)
        scapula_distance = np.linalg.norm(
            vertices[scapula_indices] - shoulder,
            axis=1,
        )
        humerus_distance = np.linalg.norm(
            vertices[humerus_indices] - shoulder,
            axis=1,
        )
        scapula_local = scapula_indices[
            scapula_distance <= np.quantile(scapula_distance, 0.25)
        ]
        humerus_local = humerus_indices[
            humerus_distance <= np.quantile(humerus_distance, 0.25)
        ]
        nearest, nearest_index = cKDTree(
            vertices[humerus_local]
        ).query(vertices[scapula_local], k=1)
        scapula_closest = int(np.argmin(nearest))
        gap = float(nearest[scapula_closest])
        target_gap = 0.0015
        if gap <= target_gap:
            continue
        proximal = int(scapula_local[scapula_closest])
        distal = int(humerus_local[int(nearest_index[scapula_closest])])
        direction = vertices[distal] - vertices[proximal]
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        translation = (gap - target_gap) * direction
        vertices[scapula_start:scapula_stop] += translation
        scapula_bone = _dominant_bone(
            asset,
            scapula_start,
            scapula_stop,
        )
        if scapula_bone is not None:
            new_global[scapula_bone, :3, 3] += translation
        bone_surface_constraints[scapula_name] = {
            **bone_surface_constraints.get(scapula_name, {}),
            "joint_surface_gap_before_m": gap,
            "joint_surface_gap_target_m": target_gap,
            "joint_surface_translation_m": translation.tolist(),
        }

    bone_delta = new_global @ np.linalg.inv(old_global)
    # Hip snap may have translated femur binds; refresh locals for bind_follow kids.
    new_local = new_global.copy()
    for bone, parent in enumerate(np.asarray(asset.source_bone_parents, dtype=np.int64).tolist()):
        if int(parent) >= 0:
            new_local[bone] = np.linalg.inv(new_global[int(parent)]) @ new_global[bone]
    rebind_report = {
        "stage": str(stage),
        "backend": "authoritative_driver_coupling_v6",
        "weighted_mesh_rebind": False,
        "bind_follow_preserved": int(sum(mode == "bind_follow" for mode in modes)),
    }

    # Soft rest positions stay on the harmonic field (335f59f / PLAN §4.2).
    # Full rotational LBS and even bone-origin translation after aggressive bone
    # moves shear thin vessels (Artery p99.9 blew past 10×).  Bone-adjacent
    # attachment handles belong in the joint volume-field stage (PLAN §4.3/4.8),
    # not as a post-hoc rest teleport here.
    vessel_nerve = _tissue_mask(asset, {"vessel", "nerve"}) & ~(cranial_soft_moved | jaw)
    organ_soft = (
        _soft_material_mask(asset)
        & ~vessel_nerve
        & ~(cranial_soft_moved | jaw)
    )
    soft_attachment_mode = "harmonic_only"
    vessel_alpha = 0.0
    organ_alpha = 0.0
    # Hip-snap soft follow is deferred to the joint volume-field stage: pulling
    # femur-weighted soft by bind translation improved neither pose stretch nor
    # vessel containment on this subject, and raised source_to_final p99.9.
    hip_follow = np.zeros(len(vertices), dtype=bool)

    endpoints_delta = bone_delta
    head = np.asarray(
        asset.target_bone_head
        if asset.target_bone_head is not None
        else asset.source_bone_head,
        dtype=np.float64,
    )
    tail = np.asarray(
        asset.target_bone_tail
        if asset.target_bone_tail is not None
        else asset.source_bone_tail,
        dtype=np.float64,
    )
    new_head = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], head) + endpoints_delta[:, :3, 3]
    new_tail = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], tail) + endpoints_delta[:, :3, 3]
    # These remain authored geometry probes transformed by the fitted bind.
    # SMPL-X controller endpoints are reported separately and must never
    # overwrite the data used by geometry diagnostics.
    anchor_error = np.asarray(
        [
            np.linalg.norm(new_global[bone, :3, 3] - target_joints[int(asset.source_bone_smplx_a[bone])])
            for bone, mode in enumerate(modes)
            if mode != "bind_follow"
        ],
        dtype=np.float64,
    )
    metadata = dict(asset.metadata or {})
    history = list(metadata.get("articulated_rest_fit", []))
    report = {
        "stage": str(stage),
        "backend": "articulated_material_fit_v6",
        "shaft_meshes": int(shaft_meshes),
        "cranial_uniform_scale": float(cranial_scale),
        "cranial_scale_report": cranial_scale_report,
        "cranial_aspect_ratio_change": float(cranial_aspect_ratio_change),
        "brain_skull_center_drift_m": float(brain_skull_center_drift_m),
        "pelvis_uniform_scale": float(pelvis_scale),
        "pelvis_scale_report": pelvis_scale_report,
        "pelvis_aspect_ratio_change": float(pelvis_aspect_ratio_change),
        "hip_geometry": hip_report,
        "thorax_uniform_scale": float(thorax_scale),
        "thorax_local_frame_scale": thorax_local_frame_scale.tolist(),
        "long_bone_end_edge_change": float(protected_end_edge_change),
        "feet": foot_report,
        "bone_surface_constraints": bone_surface_constraints,
        "target_bind_update": rebind_report,
        "soft_attachment": {
            "mode": soft_attachment_mode,
            "vessel_nerve_alpha": float(vessel_alpha),
            "organ_alpha": float(organ_alpha),
            "vessel_nerve_vertices": int(np.count_nonzero(vessel_nerve)),
            "organ_vertices": int(np.count_nonzero(organ_soft)),
            "hip_follow_vertices": int(np.count_nonzero(hip_follow)),
        },
        "anchor_rms_m": float(np.sqrt(np.mean(anchor_error * anchor_error))) if len(anchor_error) else 0.0,
        "anchor_max_m": float(np.max(anchor_error)) if len(anchor_error) else 0.0,
    }
    history.append(report)
    metadata["articulated_rest_fit"] = history
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "target_rest_global": new_global.astype(np.float32),
            "target_rest_local": new_local.astype(np.float32),
            "target_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
            "target_bone_head": new_head.astype(np.float32),
            "target_bone_tail": new_tail.astype(np.float32),
            "source_driver_coupling": None,
            "registration_reference": (
                None
                if asset.registration_reference is None
                else np.asarray(asset.registration_reference, dtype=np.float32)
            ),
            "driver_indices": fit_driver_indices,
            "driver_weights": fit_driver_weights,
            "metadata": metadata,
        }
    )
    result = with_source_driver_coupling(result)
    result.validate()
    return result, report

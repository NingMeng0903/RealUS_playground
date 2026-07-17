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

from .anatomy_lbs import joint_global_transforms
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


def _controller(bone: int, parents: np.ndarray, modes: list[str]) -> int:
    current = int(bone)
    while current >= 0 and _is_follow_mode(modes[current]):
        current = int(parents[current])
    return int(bone if current < 0 else current)


def _source_joint_anchors(asset: AnatomyRiggedAsset) -> np.ndarray:
    target = np.asarray(asset.rest_joints, dtype=np.float64)
    anchors = target.copy()
    assigned = np.zeros(len(target), dtype=bool)
    modes = list(asset.source_bone_driver_types or [])
    global_bind = np.asarray(asset.source_rest_global, dtype=np.float64)
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


def _fit_source_frames(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    old_local = np.asarray(asset.source_rest_local, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
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
    tissues = list(asset.source_tissues or [""] * len(asset.source_mesh_names))
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
    source_center: np.ndarray | None = None,
    target_center: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
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
    base_scale = float(np.min(target_extent[valid] / source_extent[valid])) if np.any(valid) else 1.0
    scale = max(0.5, min(float(maximum_scale), margin * base_scale * float(scale_multiplier)))
    return resolved_target_center + scale * (source - resolved_source_center), scale


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
            # Keep the authored distal epiphysis just behind the skin front.
            result[(side, finger)] = j3 + 0.95 * max(0.0, reach) * axis
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
                if not _is_follow_mode(modes[bi]):
                    modes[bi] = "parent_follow"
                    patched = True
        if patched:
            asset = type(asset)(**{**asset.__dict__, "source_bone_driver_types": modes})
    old_vertices = np.asarray(asset.vertices_rest, dtype=np.float64)
    vertices = old_vertices.copy()
    old_global = np.asarray(asset.source_rest_global, dtype=np.float64)
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
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[bone]
                )
                continue
            if hand_segment is not None and "metacarpal" in lower:
                vertices[start_i:stop_i] = _transform_points(
                    old_vertices[start_i:stop_i], bone_delta[bone]
                )
                continue
            if hand_segment is not None:
                source_a, source_b, target_a, target_b = hand_segment
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
    cranial_scale = 1.0
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
        # Place about the head joint (not an eye-midline AABB recenter).  The
        # Head_Bone delta already put the compound on the subject head frame;
        # the envelope only supplies a single isotropic scale about that joint.
        cranial_envelope_center_before = np.asarray(target_joints[head_index], dtype=np.float64)
        cranial_envelope_center_after = cranial_envelope_center_before + offset_world
        cranial_reference = (
            vertices[skull_reference] if np.any(skull_reference) else vertices[cranial]
        )
        vertices[cranial], cranial_scale = _uniform_envelope_fit(
            vertices[cranial],
            target_head,
            reference_points=cranial_reference,
            scale_multiplier=multiplier,
            center_offset=offset_world,
            margin=0.96,
            source_center=cranial_envelope_center_before,
            target_center=cranial_envelope_center_after,
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
            else:
                vertices[jaw] = jaw_base

    pelvis = pelvis_material
    pelvis_scale = 1.0
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
        vertices[pelvis], pelvis_scale = _uniform_envelope_fit(
            rotated,
            target_pelvis,
            reference_points=rotated,
            scale_multiplier=multiplier,
            center_offset=np.zeros(3, dtype=np.float64),
            margin=0.96,
            maximum_scale=1.20,
            source_center=target_center,
            target_center=target_center,
        )
        # Floor so a narrow subject hip surface cannot collapse the ilium.
        if pelvis_scale < 0.80:
            vertices[pelvis] = target_center + 0.80 / max(pelvis_scale, 1.0e-8) * (
                vertices[pelvis] - target_center
            )
            pelvis_scale = 0.80
        pelvis_aspect_ratio_change = _aspect_ratio_change(old_pelvis, vertices[pelvis])

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
        vertices[foot] = ankle + scale * (vertices[foot] - ankle)
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
            "uniform_scale": float(scale),
            "source_reach_m": source_reach,
            "target_reach_m": target_reach,
            "surface_center_offset_m": rigid_offset.tolist(),
            "forefoot_gap_before_m": 0.0,
            "forefoot_rigid_shift_m": 0.0,
        }

    # Snapshot articulated bind frames before rebind.  Weighted vertex rebind
    # may improve local orientation, but must not drag bind origins off the
    # SMPL-X joints (that caused 13-32 cm anchor drift and detached chains).
    articulated_global = new_global.copy()
    articulated_local = new_local.copy()

    interim = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "source_rest_global": new_global.astype(np.float32),
            "source_rest_local": new_local.astype(np.float32),
            "source_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
        }
    )
    rebound, rebind_report = rebind_source_rig(
        interim,
        source_vertices=old_vertices,
        target_vertices=vertices,
        stage=stage,
        bone_mask=bone_material,
    )
    new_global = np.asarray(rebound.source_rest_global, dtype=np.float64)
    # Rotation-only rebind for independent drivers: keep articulated bind
    # translation on the SMPL-X joint, adopt rebind orientation only.
    for bone, mode in enumerate(modes):
        if _is_follow_mode(mode):
            continue
        new_global[bone, :3, 3] = articulated_global[bone, :3, 3]
        if mode in {"segment_root", "twist", "rigid_group"} or _is_joint_local_mode(mode):
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

    # Soft tissue rest shape comes from the harmonic volume field in
    # shape_volume.py.  Do not LBS-blend soft through articulated bone deltas
    # here — that re-introduces thin-structure explosions across rib/spine and
    # wrist/finger driver boundaries.  Cranial soft already moved with the skull.
    soft_material = _soft_material_mask(asset)
    soft_material &= ~(cranial_soft_moved | jaw)
    # Soft vertices already hold the field-warped positions from shape_volume.

    endpoints_delta = bone_delta
    head = np.asarray(asset.source_bone_head, dtype=np.float64)
    tail = np.asarray(asset.source_bone_tail, dtype=np.float64)
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
    history = list(metadata.get("articulated_rest_fit", []))
    report = {
        "stage": str(stage),
        "backend": "articulated_material_fit_v5",
        "shaft_meshes": int(shaft_meshes),
        "cranial_uniform_scale": float(cranial_scale),
        "cranial_aspect_ratio_change": float(cranial_aspect_ratio_change),
        "brain_skull_center_drift_m": float(brain_skull_center_drift_m),
        "pelvis_uniform_scale": float(pelvis_scale),
        "pelvis_aspect_ratio_change": float(pelvis_aspect_ratio_change),
        "thorax_uniform_scale": float(thorax_scale),
        "thorax_axis_scale": thorax_axis_scale.tolist(),
        "long_bone_end_edge_change": float(protected_end_edge_change),
        "maximum_digit_rigid_offset_m": 0.0,
        "feet": foot_report,
        "source_rig_rebind": rebind_report,
        "anchor_rms_m": float(np.sqrt(np.mean(anchor_error * anchor_error))) if len(anchor_error) else 0.0,
        "anchor_max_m": float(np.max(anchor_error)) if len(anchor_error) else 0.0,
    }
    history.append(report)
    metadata["articulated_rest_fit"] = history
    result = type(asset)(
        **{
            **rebound.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "source_rest_global": new_global.astype(np.float32),
            "source_rest_local": new_local.astype(np.float32),
            "source_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
            "source_bone_head": new_head.astype(np.float32),
            "source_bone_tail": new_tail.astype(np.float32),
            "registration_reference": vertices.astype(np.float32),
            "driver_indices": fit_driver_indices,
            "driver_weights": fit_driver_weights,
            "metadata": metadata,
        }
    )
    result.validate()
    return result, report

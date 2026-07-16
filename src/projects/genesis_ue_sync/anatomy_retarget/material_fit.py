"""Shape-preserving articulated rest fitting for anatomy schema v4.

Rigid anatomy is fitted from semantic joints and material groups.  Soft
tissues are deliberately left for the volumetric registration stages.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms
from .rigged_asset import AnatomyRiggedAsset


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
    global_bind = np.asarray(asset.source_rest_global, dtype=np.float64)
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


def _fit_source_frames(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_global = np.asarray(asset.source_rest_global, dtype=np.float64)
    old_local = np.asarray(asset.source_rest_local, dtype=np.float64)
    source_parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    modes = list(asset.source_bone_driver_types or [])
    target_joints = np.asarray(asset.rest_joints, dtype=np.float64)
    source_anchors = _source_joint_anchors(asset)
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
        if joint_name in {"left_wrist", "right_wrist"}:
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
        desired = np.eye(4, dtype=np.float64)
        desired[:3, :3] = rotation
        desired[:3, 3] = target_joints[a]
        if parent < 0:
            new_global[bone] = desired
        else:
            local = np.linalg.inv(new_global[parent]) @ desired
            new_global[bone] = new_global[parent] @ local
    new_local = new_global.copy()
    for bone, parent in enumerate(source_parents.tolist()):
        if int(parent) >= 0:
            new_local[bone] = np.linalg.inv(new_global[int(parent)]) @ new_global[bone]
    delta = new_global @ np.linalg.inv(old_global)
    return new_global, new_local, delta


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ transform[:3, :3].T + transform[:3, 3]


def _aspect_ratio_change(source: np.ndarray, fitted: np.ndarray) -> float:
    source_extent = np.sort(np.ptp(np.asarray(source, dtype=np.float64), axis=0))
    fitted_extent = np.sort(np.ptp(np.asarray(fitted, dtype=np.float64), axis=0))
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
    return _mesh_mask(asset, lambda name, _tissue: any(token in name for token in _CRANIAL_TOKENS))


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
) -> tuple[np.ndarray, float]:
    source = np.asarray(points, dtype=np.float64)
    reference = np.asarray(
        source if reference_points is None else reference_points, dtype=np.float64
    )
    destination = np.asarray(target, dtype=np.float64)
    source_lo, source_hi = np.quantile(reference, (0.01, 0.99), axis=0)
    target_lo, target_hi = np.quantile(destination, (0.01, 0.99), axis=0)
    source_center = 0.5 * (source_lo + source_hi)
    target_center = 0.5 * (target_lo + target_hi) + np.asarray(center_offset, dtype=np.float64)
    source_extent = 0.5 * (source_hi - source_lo)
    target_extent = 0.5 * (target_hi - target_lo)
    valid = source_extent > 1.0e-5
    base_scale = float(np.min(target_extent[valid] / source_extent[valid])) if np.any(valid) else 1.0
    scale = max(0.5, min(float(maximum_scale), margin * base_scale * float(scale_multiplier)))
    return target_center + scale * (source - source_center), scale


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
    """Fit rigid anatomy and source frames without deforming soft tissues."""
    asset.validate()
    cfg = dict(config or {})
    root = Path(canonical_dir)
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

    # The authored sparse source-rig weights provide the regional skin
    # correspondence for soft anatomy.  Apply the articulated rest deltas once
    # before the residual harmonic field; this prevents a whole-body nearest
    # surface fit from pairing an arm vessel with the torso or opposite limb.
    protected_compound = bone_material_mask(asset) | cranial_material_mask(asset)
    if asset.driver_indices is not None and asset.driver_weights is not None:
        driver_indices = np.asarray(asset.driver_indices, dtype=np.int64)
        driver_weights = np.asarray(asset.driver_weights, dtype=np.float64)
        selected = bone_delta[driver_indices]
        driven = np.einsum(
            "nkij,nkj->nki", selected[:, :, :3, :3], old_vertices[:, None, :]
        ) + selected[:, :, :3, 3]
        articulated = np.sum(driven * driver_weights[:, :, None], axis=1)
        vertices[~protected_compound] = articulated[~protected_compound]
        source_skin_vertices = None
        if asset.source_skin_vertices is not None:
            from scipy.spatial import cKDTree

            skin = np.asarray(asset.source_skin_vertices, dtype=np.float64)
            distance, nearest = cKDTree(old_vertices).query(skin, k=min(8, len(old_vertices)))
            distance = np.asarray(distance, dtype=np.float64).reshape(len(skin), -1)
            nearest = np.asarray(nearest, dtype=np.int64).reshape(len(skin), -1)
            spatial = 1.0 / (distance * distance + 1.0e-6)
            spatial /= np.maximum(np.sum(spatial, axis=1, keepdims=True), 1.0e-12)
            skin_bones = driver_indices[nearest].reshape(len(skin), -1)
            skin_weights = (
                spatial[:, :, None] * driver_weights[nearest]
            ).reshape(len(skin), -1)
            skin_weights /= np.maximum(np.sum(skin_weights, axis=1, keepdims=True), 1.0e-12)
            skin_delta = bone_delta[skin_bones]
            skin_driven = np.einsum(
                "nkij,nkj->nki", skin_delta[:, :, :3, :3], skin[:, None, :]
            ) + skin_delta[:, :, :3, 3]
            source_skin_vertices = np.sum(skin_driven * skin_weights[:, :, None], axis=1)
    else:
        source_skin_vertices = None

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
            control = _controller(bone, source_parents, modes)
            a = int(asset.source_bone_smplx_a[control])
            b = int(asset.source_bone_smplx_b[control])
            lower = str(name).lower()
            hand_segment = _hand_mesh_segment(
                str(name),
                joint_names=asset.joint_names,
                source_anchors=source_anchors,
                target_joints=target_joints,
                finger_tips=finger_tips,
            )
            if "1st_metacarpal" in lower:
                # The authored first metacarpal already encodes the thumb's
                # opposition angle.  SMPL-X thumb1 is not collinear with the
                # four palm rays, so treating it as a straight shaft rotates
                # the epiphysis through the palm surface.
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
            if a == b and modes[control] == "joint_local":
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

    # Each complete digit may receive a sub-2 mm rigid skin-centering offset.
    # This preserves every bone shape and stays within the accepted mesh-chain
    # gap budget while avoiding transverse scaling of tiny phalanges.
    surface_faces_for_fit = np.asarray(
        np.load(root / "smpl_canonical_weights.npz", allow_pickle=True)["faces"], dtype=np.int32
    )
    surface_for_fit = _load_obj_vertices(
        root / ("smpl_canonical_tpose.obj" if subject else "smpl_canonical_tpose_neutral.obj")
    )
    import igl

    maximum_digit_offset = 0.0
    for side_suffix in ("_l", "_r"):
        for digit in range(1, 6):
            digit_mask = _mesh_mask(
                asset,
                lambda name, tissue, digit=digit, suffix=side_suffix: tissue == "bone"
                and any(token in name for token in ("metacarpal", "phalanx_hand", "phalanges_hand"))
                and name.endswith(suffix)
                and re.search(rf"(?:^|_){digit}(?:st|nd|rd|th)?_", name) is not None,
            )
            if not np.any(digit_mask):
                continue
            total_offset = np.zeros(3, dtype=np.float64)
            for _iteration in range(2):
                signed, _face_index, closest, _normal = igl.signed_distance(
                    vertices[digit_mask], surface_for_fit, surface_faces_for_fit
                )
                outside = np.asarray(signed) > 0.0
                if not np.any(outside):
                    break
                step = np.median(
                    np.asarray(closest)[outside] - vertices[digit_mask][outside], axis=0
                )
                length = float(np.linalg.norm(step))
                if length <= 1.0e-7:
                    break
                step *= min(1.0, 0.001 / length)
                vertices[digit_mask] += step
                total_offset += step
            maximum_digit_offset = max(maximum_digit_offset, float(np.linalg.norm(total_offset)))

    cranial = cranial_material_mask(asset)
    skull_reference = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone" and ("skull" in name or "cranium" in name),
    )
    cranial_scale = 1.0
    cranial_aspect_ratio_change = 0.0
    brain_skull_center_drift_m = 0.0
    if np.any(cranial):
        old_cranial = vertices[cranial].copy()
        old_skull_center = np.mean(vertices[skull_reference], axis=0) if np.any(skull_reference) else np.mean(old_cranial, axis=0)
        old_brain_center = np.mean(vertices[cranial & ~skull_reference], axis=0) if np.any(cranial & ~skull_reference) else old_skull_center
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
        vertices[cranial], cranial_scale = _uniform_envelope_fit(
            vertices[cranial],
            target_head,
            reference_points=vertices[skull_reference] if np.any(skull_reference) else vertices[cranial],
            scale_multiplier=multiplier,
            center_offset=offset_world,
            margin=0.96,
        )
        cranial_aspect_ratio_change = _aspect_ratio_change(old_cranial, vertices[cranial])
        new_skull_center = np.mean(vertices[skull_reference], axis=0) if np.any(skull_reference) else np.mean(vertices[cranial], axis=0)
        new_brain_center = np.mean(vertices[cranial & ~skull_reference], axis=0) if np.any(cranial & ~skull_reference) else new_skull_center
        brain_skull_center_drift_m = float(
            np.linalg.norm((new_brain_center - new_skull_center) - cranial_scale * (old_brain_center - old_skull_center))
        )

    pelvis = _mesh_mask(asset, lambda name, tissue: tissue == "bone" and any(t in name for t in _PELVIS_TOKENS))
    pelvis_scale = 1.0
    pelvis_aspect_ratio_change = 0.0
    if np.any(pelvis):
        old_pelvis = vertices[pelvis].copy()
        target_pelvis = _surface_region(
            root,
            asset.joint_names,
            ("pelvis", "left_hip", "right_hip", "spine1"),
            subject=subject,
        )
        multiplier, local_offset = _override(cfg, "pelvis")
        pelvis_id = asset.joint_names.index("pelvis")
        left_hip = target_joints[asset.joint_names.index("left_hip")] - target_joints[pelvis_id]
        right_hip = target_joints[asset.joint_names.index("right_hip")] - target_joints[pelvis_id]
        spine = target_joints[asset.joint_names.index("spine1")] - target_joints[pelvis_id]
        lateral = right_hip - left_hip
        lateral /= max(float(np.linalg.norm(lateral)), 1.0e-8)
        vertical = spine - lateral * float(spine @ lateral)
        vertical /= max(float(np.linalg.norm(vertical)), 1.0e-8)
        depth = np.cross(lateral, vertical)
        pelvis_frame = np.stack((lateral, vertical, depth), axis=1)
        vertices[pelvis], pelvis_scale = _uniform_envelope_fit(
            vertices[pelvis],
            target_pelvis,
            scale_multiplier=multiplier,
            center_offset=pelvis_frame @ local_offset,
            margin=0.94,
            maximum_scale=1.0,
        )
        pelvis_aspect_ratio_change = _aspect_ratio_change(old_pelvis, vertices[pelvis])

    # The sternum, ribs and scapulae are a single authored thoracic shell.
    # Fitting them independently moves their joints apart; one rigid/uniform
    # compound transform preserves their proportions and relative layout.
    thorax = _mesh_mask(
        asset,
        lambda name, tissue: tissue == "bone"
        and any(token in name for token in ("sternum", "scapula", "rib_", "clavicle")),
    )
    thorax_scale = 1.0
    if np.any(thorax):
        target_thorax = _surface_region(
            root,
            asset.joint_names,
            ("spine2", "spine3", "left_collar", "right_collar", "left_shoulder", "right_shoulder"),
            subject=subject,
        )
        vertices[thorax], thorax_scale = _uniform_envelope_fit(
            vertices[thorax],
            target_thorax,
            scale_multiplier=1.0,
            center_offset=np.zeros(3, dtype=np.float64),
            margin=0.82,
            maximum_scale=1.0,
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
        source_reach = float(np.quantile((vertices[foot] - ankle) @ forward, 0.995))
        target_reach = float(np.quantile((target_foot - ankle) @ forward, 0.995))
        scale = min(1.0, 0.90 * target_reach / max(source_reach, 1.0e-5))
        vertices[foot] = ankle + scale * (vertices[foot] - ankle)
        forefoot = _mesh_mask(
            asset,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and any(token in name for token in ("metatarsal", "phalanx_foot", "phalanges_foot"))
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        midfoot = _mesh_mask(
            asset,
            lambda name, tissue, suffix=suffix: tissue == "bone"
            and any(token in name for token in ("navicular", "cuboid", "cuneiform"))
            and (name.endswith(suffix) or f"{suffix}_" in name),
        )
        forefoot_gap = 0.0
        forefoot_shift = 0.0
        if np.any(forefoot) and np.any(midfoot):
            fore_projection = (vertices[forefoot] - ankle) @ forward
            mid_projection = (vertices[midfoot] - ankle) @ forward
            forefoot_gap = float(np.quantile(fore_projection, 0.01) - np.quantile(mid_projection, 0.99))
            forefoot_shift = max(0.0, forefoot_gap - 0.003)
            if forefoot_shift > 0.0:
                # Preserve the authored metatarsal/toe subtree exactly while
                # closing only the excessive midfoot-to-forefoot rest gap.
                vertices[forefoot] -= forefoot_shift * forward
        # Surface-derived rigid centering corrects asymmetric feet without a
        # world-space offset or per-bone deformation.  Apply at most 5 mm per
        # iteration and retain the complete authored foot subtree together.
        import igl

        rigid_offset = np.zeros(3, dtype=np.float64)
        for _iteration in range(8):
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
            step *= min(1.0, 0.005 / length)
            vertices[foot] += step
            rigid_offset += step
        for _proximal_iteration in range(3):
            proximal = ((vertices[foot] - ankle) @ forward) <= 0.30 * max(target_reach, 1.0e-5)
            signed, _face_index, closest, _normal = igl.signed_distance(
                vertices[foot][proximal], subject_surface, surface_faces
            )
            outside = np.asarray(signed) > 0.0
            if not np.any(outside):
                break
            step = np.median(
                np.asarray(closest)[outside] - vertices[foot][proximal][outside], axis=0
            )
            # Keep forward reach unchanged; this is the missing surface-PCA
            # transverse centering term for the hindfoot.
            step -= forward * float(step @ forward)
            length = float(np.linalg.norm(step))
            if length <= 1.0e-7:
                break
            step *= min(1.0, 0.005 / length)
            vertices[foot] += step
            rigid_offset += step
        foot_report[side] = {
            "uniform_scale": float(scale),
            "source_reach_m": source_reach,
            "target_reach_m": target_reach,
            "surface_center_offset_m": rigid_offset.tolist(),
            "forefoot_gap_before_m": float(forefoot_gap),
            "forefoot_rigid_shift_m": float(forefoot_shift),
        }

    endpoints_delta = bone_delta
    head = np.asarray(asset.source_bone_head, dtype=np.float64)
    tail = np.asarray(asset.source_bone_tail, dtype=np.float64)
    new_head = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], head) + endpoints_delta[:, :3, 3]
    new_tail = np.einsum("bij,bj->bi", endpoints_delta[:, :3, :3], tail) + endpoints_delta[:, :3, 3]
    finger_roots = {
        side: [
            asset.joint_names.index(f"{side}_{finger}1")
            for finger in ("thumb", "index", "middle", "ring", "pinky")
            if f"{side}_{finger}1" in asset.joint_names
        ]
        for side in ("left", "right")
    }
    for bone, mode in enumerate(modes):
        a = int(asset.source_bone_smplx_a[bone])
        b = int(asset.source_bone_smplx_b[bone])
        if mode in {"segment_root", "twist"} and a != b:
            new_head[bone] = target_joints[a]
            new_tail[bone] = target_joints[b]
        elif mode == "rigid_group" and a != b:
            new_head[bone] = target_joints[a]
            new_tail[bone] = target_joints[b]
        elif mode == "joint_local":
            joint_name = asset.joint_names[a]
            new_head[bone] = target_joints[a]
            if joint_name in {"left_wrist", "right_wrist"}:
                side = joint_name.split("_", 1)[0]
                index_name = f"{side}_index1"
                if index_name in asset.joint_names:
                    # The frame orientation uses all five MCP vectors above;
                    # the stored tail is only an unambiguous chain probe.
                    new_tail[bone] = target_joints[asset.joint_names.index(index_name)]
            else:
                child = _joint_child(a, asset.parents)
                if child is not None:
                    new_tail[bone] = target_joints[child]
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
        "backend": "articulated_material_fit_v4",
        "shaft_meshes": int(shaft_meshes),
        "cranial_uniform_scale": float(cranial_scale),
        "cranial_aspect_ratio_change": float(cranial_aspect_ratio_change),
        "brain_skull_center_drift_m": float(brain_skull_center_drift_m),
        "pelvis_uniform_scale": float(pelvis_scale),
        "pelvis_aspect_ratio_change": float(pelvis_aspect_ratio_change),
        "thorax_uniform_scale": float(thorax_scale),
        "long_bone_end_edge_change": float(protected_end_edge_change),
        "maximum_digit_rigid_offset_m": float(maximum_digit_offset),
        "feet": foot_report,
        "anchor_rms_m": float(np.sqrt(np.mean(anchor_error * anchor_error))) if len(anchor_error) else 0.0,
        "anchor_max_m": float(np.max(anchor_error)) if len(anchor_error) else 0.0,
    }
    history.append(report)
    metadata["articulated_rest_fit"] = history
    result = type(asset)(
        **{
            **asset.__dict__,
            "vertices_rest": vertices.astype(np.float32),
            "source_rest_global": new_global.astype(np.float32),
            "source_rest_local": new_local.astype(np.float32),
            "source_inverse_bind": np.linalg.inv(new_global).astype(np.float32),
            "source_bone_head": new_head.astype(np.float32),
            "source_bone_tail": new_tail.astype(np.float32),
            "source_skin_vertices": (
                source_skin_vertices.astype(np.float32)
                if source_skin_vertices is not None
                else asset.source_skin_vertices
            ),
            "registration_reference": vertices.astype(np.float32),
            "metadata": metadata,
        }
    )
    result.validate()
    return result, report

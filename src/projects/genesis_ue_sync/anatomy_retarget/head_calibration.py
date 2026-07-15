"""Rest-space head/skull calibration that preserves zero-pose source LBS."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anatomy_lbs import joint_global_transforms
from .rigged_asset import AnatomyRiggedAsset


def _has_ancestor(bone: int, ancestor: int, parents: np.ndarray) -> bool:
    current = int(bone)
    visited = 0
    while current >= 0 and visited <= len(parents):
        if current == int(ancestor):
            return True
        current = int(parents[current])
        visited += 1
    return False


def _head_subtree_bones(asset: AnatomyRiggedAsset, head_index: int) -> set[int]:
    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    return {
        int(bi)
        for bi in range(len(parents))
        if _has_ancestor(bi, int(head_index), parents) or int(bi) == int(head_index)
    }


def _dominant_source_bone(asset: AnatomyRiggedAsset, start: int, stop: int) -> int | None:
    if asset.driver_indices is None or asset.driver_weights is None or asset.source_bone_names is None:
        return None
    indices = np.asarray(asset.driver_indices[start:stop], dtype=np.int64).reshape(-1)
    weights = np.asarray(asset.driver_weights[start:stop], dtype=np.float64).reshape(-1)
    mass = np.bincount(indices, weights=weights, minlength=len(asset.source_bone_names))
    return int(np.argmax(mass)) if mass.size and float(mass.max()) > 0.0 else None


def _compute_head_rest_offset(asset: AnatomyRiggedAsset) -> np.ndarray:
    if asset.source_bone_names is None or "Head_Bone" not in asset.source_bone_names:
        return np.zeros(3, dtype=np.float64)
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    if "head" not in joint_index:
        return np.zeros(3, dtype=np.float64)
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    smpl_head = rest_global[joint_index["head"], :3, 3]
    skull_centroid = None
    if asset.source_vertex_ranges is not None and asset.source_mesh_names is not None:
        for (start, stop), mesh_name in zip(asset.source_vertex_ranges, asset.source_mesh_names):
            if "skull" not in str(mesh_name).lower():
                continue
            block = np.asarray(asset.vertices_rest[int(start) : int(stop)], dtype=np.float64)
            if len(block) >= 3:
                skull_centroid = block.mean(axis=0)
                break
    if skull_centroid is None:
        return np.zeros(3, dtype=np.float64)
    offset = np.zeros(3, dtype=np.float64)
    vertical_gap = float(skull_centroid[1] - smpl_head[1])
    if vertical_gap < -0.002:
        offset[1] = float(-vertical_gap * 0.85)
    return offset


def calibrate_head_rest_offset(asset: AnatomyRiggedAsset) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Lift the head subtree in rest space while keeping zero-pose LBS exact."""
    if asset.source_bone_names is None or asset.source_rest_global is None:
        return asset, {"applied": False, "reason": "legacy_asset"}
    names = list(asset.source_bone_names)
    if "Head_Bone" not in names:
        return asset, {"applied": False, "reason": "missing_head_bone"}
    head_index = int(names.index("Head_Bone"))
    offset = _compute_head_rest_offset(asset)
    if float(np.linalg.norm(offset)) < 1.0e-4:
        return asset, {"applied": False, "reason": "within_tolerance", "offset_m": [0.0, 0.0, 0.0]}

    head_bones = _head_subtree_bones(asset, head_index)
    vertices = np.asarray(asset.vertices_rest, dtype=np.float64).copy()
    mask = np.zeros(len(vertices), dtype=bool)
    if asset.source_vertex_ranges is not None:
        parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
        for (start, stop) in np.asarray(asset.source_vertex_ranges, dtype=np.int64):
            bone = _dominant_source_bone(asset, int(start), int(stop))
            if bone is not None and int(bone) in head_bones:
                mask[int(start) : int(stop)] = True
    if not np.any(mask):
        return asset, {"applied": False, "reason": "no_head_meshes", "offset_m": offset.tolist()}

    vertices[mask] += offset
    rest_global = np.asarray(asset.source_rest_global, dtype=np.float64).copy()
    for bi in head_bones:
        rest_global[int(bi), :3, 3] += offset

    parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
    rest_local = rest_global.copy()
    for bi, parent in enumerate(parents.tolist()):
        if int(parent) >= 0:
            rest_local[bi] = np.linalg.inv(rest_global[int(parent)]) @ rest_global[bi]

    updates: dict[str, Any] = {
        "vertices_rest": vertices.astype(np.float32),
        "source_rest_global": rest_global.astype(np.float32),
        "source_rest_local": rest_local.astype(np.float32),
        "source_inverse_bind": np.linalg.inv(rest_global).astype(np.float32),
    }
    if asset.registration_reference is not None:
        reference = np.asarray(asset.registration_reference, dtype=np.float64).copy()
        reference[mask] += offset
        updates["registration_reference"] = reference.astype(np.float32)
    for field_name in ("source_bone_head", "source_bone_tail"):
        value = getattr(asset, field_name, None)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float64).copy()
        for bi in head_bones:
            points[int(bi)] += offset
        updates[field_name] = points.astype(np.float32)

    meta = dict(asset.metadata or {})
    meta["head_rest_calibration"] = {
        "offset_m": [float(v) for v in offset.tolist()],
        "head_bones": int(len(head_bones)),
        "vertex_count": int(np.count_nonzero(mask)),
    }
    updates["metadata"] = meta
    return type(asset)(**{**asset.__dict__, **updates}), dict(meta["head_rest_calibration"], applied=True)

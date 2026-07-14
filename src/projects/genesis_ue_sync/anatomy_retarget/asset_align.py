"""Scale and align Blender-exported anatomy vertices to canonical SMPL-X rest space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .rigged_asset import AnatomyRiggedAsset, load_rigged_asset, save_rigged_asset

DEFAULT_BLENDER_UNIT_SCALE = 0.01
DEFAULT_PELVIS_JOINT = "pelvis"
DRAW_SPAN_MAX_M = 10.0


def _pelvis_index(joint_names: list[str], *, pelvis_joint: str) -> int:
    name = str(pelvis_joint)
    if name in joint_names:
        return int(joint_names.index(name))
    return 0


def normalize_vertices_to_canonical_rest(
    vertices_rest: np.ndarray,
    lbs_weights: np.ndarray,
    rest_joints: np.ndarray,
    *,
    unit_scale: float = DEFAULT_BLENDER_UNIT_SCALE,
    pelvis_index: int = 0,
) -> np.ndarray:
    """Map Blender world units (~cm) into canonical SMPL-X rest joint space (meters)."""
    verts = np.asarray(vertices_rest, dtype=np.float32) * float(unit_scale)
    weights = np.asarray(lbs_weights, dtype=np.float32)
    joints = np.asarray(rest_joints, dtype=np.float32).reshape(-1, 3)
    idx = int(np.clip(pelvis_index, 0, joints.shape[0] - 1))
    pelvis_w = weights[:, idx]
    total = float(pelvis_w.sum())
    if total > 1.0e-8:
        anchor = (verts * pelvis_w[:, None]).sum(axis=0) / total
    else:
        anchor = verts.mean(axis=0)
    target = joints[idx]
    return (verts - anchor + target).astype(np.float32)


def asset_rest_span_m(asset: AnatomyRiggedAsset) -> float:
    return float(np.max(np.ptp(np.asarray(asset.vertices_rest, dtype=np.float32), axis=0)))


def needs_vertex_rest_normalize(asset: AnatomyRiggedAsset, *, max_span_m: float = DRAW_SPAN_MAX_M) -> bool:
    return asset_rest_span_m(asset) > float(max_span_m)


def normalize_rigged_asset(
    asset: AnatomyRiggedAsset,
    config: dict[str, Any] | None = None,
) -> AnatomyRiggedAsset:
    if asset.source_bone_names is not None:
        raise ValueError("source-rig v2 must be exported in metric canonical coordinates; post-hoc scaling is forbidden")
    cfg = dict(config or {})
    pelvis_joint = str(cfg.get("fallback_joint", DEFAULT_PELVIS_JOINT))
    pelvis_idx = _pelvis_index(asset.joint_names, pelvis_joint=pelvis_joint)
    unit_scale = float(cfg.get("blender_unit_scale", DEFAULT_BLENDER_UNIT_SCALE))
    verts = normalize_vertices_to_canonical_rest(
        asset.vertices_rest,
        asset.lbs_weights,
        asset.rest_joints,
        unit_scale=unit_scale,
        pelvis_index=pelvis_idx,
    )
    meta = dict(asset.metadata or {})
    meta["vertex_rest_normalized"] = True
    meta["blender_unit_scale"] = unit_scale
    meta["align_pelvis_joint"] = pelvis_joint
    return type(asset)(**{**asset.__dict__, "vertices_rest": verts, "metadata": meta})


def normalize_rigged_asset_file(
    path: Path | str,
    *,
    config: dict[str, Any] | None = None,
    force: bool = False,
) -> AnatomyRiggedAsset:
    asset = load_rigged_asset(path, validate=True)
    if force or needs_vertex_rest_normalize(asset):
        asset = normalize_rigged_asset(asset, config)
        save_rigged_asset(path, asset)
    return asset

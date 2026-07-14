"""Bake Blender segment-local coupling matrices for source_template_v4."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anatomy_lbs import _endpoint_segment_delta, _rigid_frame, _segment_frame, joint_global_transforms
from .rigged_asset import AnatomyRiggedAsset

_SEGMENT_PREFIXES = (
    "forearm_segment_",
    "shin_segment_",
    "knee_chain_",
    "foot_chain_",
    "head_segment",
    "rib_segment",
)


def _segment_rest_frame(
    asset: AnatomyRiggedAsset,
    bi: int,
    driver_type: str,
    *,
    rest_points: np.ndarray,
    joint_index: dict[str, int],
) -> np.ndarray:
    a = int(asset.source_bone_smplx_a[bi])
    b = int(asset.source_bone_smplx_b[bi])
    reference_x = np.asarray(asset.source_rest_global[bi], dtype=np.float64)[:3, 0]
    if driver_type.startswith("forearm_segment_") or driver_type.startswith("shin_segment_") or driver_type.startswith("knee_chain_") or driver_type.startswith("foot_chain_"):
        return _segment_frame(rest_points[a], rest_points[b], reference_x)
    if driver_type == "head_segment" or driver_type == "rib_segment":
        return _segment_frame(rest_points[a], rest_points[b], reference_x)
    if driver_type.startswith("scapula_"):
        side = "left" if driver_type.endswith("left") else "right"
        s, c, h = (joint_index["spine3"], joint_index[f"{side}_collar"], joint_index[f"{side}_shoulder"])
        return _rigid_frame(rest_points[h], rest_points[c], rest_points[s])
    return np.asarray(asset.source_rest_global[bi], dtype=np.float64)


def bake_segment_coupling(asset: AnatomyRiggedAsset) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute ``M_couple[bone] = inv(F_seg_rest) @ T_blender_rest`` for segment-driven bones."""
    if asset.source_bone_names is None or asset.source_rest_global is None:
        raise ValueError("segment coupling requires source-rig v2 arrays")
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    rest_points = rest_global[:, :3, 3]
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    types = asset.source_bone_driver_types or []
    coupling = np.tile(np.eye(4, dtype=np.float32), (len(asset.source_bone_names), 1, 1))
    fitted = 0
    for bi, driver_type in enumerate(types):
        is_segment = any(
            driver_type.startswith(prefix) if prefix.endswith("_") else driver_type == prefix
            for prefix in _SEGMENT_PREFIXES
        )
        if not is_segment and not driver_type.startswith("scapula_"):
            continue
        F_seg = _segment_rest_frame(asset, bi, driver_type, rest_points=rest_points, joint_index=joint_index)
        T_bone = np.asarray(asset.source_rest_global[bi], dtype=np.float64)
        M = np.linalg.inv(F_seg) @ T_bone
        coupling[bi] = M.astype(np.float32)
        fitted += 1
    return coupling, {"fitted_bones": int(fitted), "backend": "segment_coupling_v4"}


def segment_coupling_roundtrip_error(asset: AnatomyRiggedAsset, coupling: np.ndarray) -> float:
    """Max translation error when recomposing rest frames from coupling."""
    if asset.source_bone_names is None:
        return 0.0
    rest_global = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=asset.rest_joints,
        parents=asset.parents,
    ).astype(np.float64)
    rest_points = rest_global[:, :3, 3]
    joint_index = {name: idx for idx, name in enumerate(asset.joint_names)}
    types = asset.source_bone_driver_types or []
    errors: list[float] = []
    for bi, driver_type in enumerate(types):
        if float(np.max(np.abs(coupling[bi] - np.eye(4)))) < 1.0e-8:
            continue
        F_seg = _segment_rest_frame(asset, bi, driver_type, rest_points=rest_points, joint_index=joint_index)
        predicted = (F_seg @ np.asarray(coupling[bi], dtype=np.float64))[:3, 3]
        actual = np.asarray(asset.source_rest_global[bi], dtype=np.float64)[:3, 3]
        errors.append(float(np.linalg.norm(predicted - actual)))
    return float(max(errors)) if errors else 0.0


def refresh_segment_coupling(asset: AnatomyRiggedAsset) -> tuple[AnatomyRiggedAsset, dict[str, Any]]:
    """Rebake coupling after any ``rebind_source_rig`` rest-space warp."""
    coupling, report = bake_segment_coupling(asset)
    report["roundtrip_error_m"] = segment_coupling_roundtrip_error(asset, coupling)
    return type(asset)(**{**asset.__dict__, "source_segment_coupling": coupling}), report

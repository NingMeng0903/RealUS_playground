"""Release contract for a reusable Stage-1 harmonic runtime asset."""

from __future__ import annotations

from typing import Any

import numpy as np

from .anatomy_lbs import skin_vertices, source_bone_skinning_transforms
from .rigged_asset import AnatomyRiggedAsset


def stage1_runtime_contract(asset: AnatomyRiggedAsset) -> dict[str, Any]:
    """Verify that the asset is driven at runtime, not cached for one pose."""
    asset.validate()
    required = {
        "source_bone_names": asset.source_bone_names,
        "source_rest_global": asset.source_rest_global,
        "source_rest_local": asset.source_rest_local,
        "target_rest_global": asset.target_rest_global,
        "target_rest_local": asset.target_rest_local,
        "target_inverse_bind": asset.target_inverse_bind,
        "source_driver_coupling": asset.source_driver_coupling,
        "driver_indices": asset.driver_indices,
        "driver_weights": asset.driver_weights,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    zero_pose = np.zeros((55, 3), dtype=np.float32)
    transforms = source_bone_skinning_transforms(asset, zero_pose)
    vertices = skin_vertices(asset, zero_pose)
    identity = np.eye(4, dtype=np.float64)[None]
    transform_error = float(
        np.max(np.abs(np.asarray(transforms, dtype=np.float64) - identity))
    )
    vertex_error = float(
        np.max(
            np.linalg.norm(
                np.asarray(vertices, dtype=np.float64)
                - np.asarray(asset.vertices_rest, dtype=np.float64),
                axis=1,
            )
        )
    )
    pose_cache_absent = asset.pose_cache_vertices is None or not np.asarray(
        asset.pose_cache_vertices
    ).size
    passed = bool(
        not missing
        and pose_cache_absent
        and transform_error <= 1.0e-5
        and vertex_error <= 1.0e-5
    )
    return {
        "contract_version": 1,
        "runtime_backend": "source_rig_smplx55_lbs",
        "requires_blender_at_runtime": False,
        "requires_pose_rebake": False,
        "required_runtime_fields_missing": missing,
        "pose_cache_absent": bool(pose_cache_absent),
        "zero_pose_source_transform_error": transform_error,
        "zero_pose_vertex_error_m": vertex_error,
        "passed": passed,
    }

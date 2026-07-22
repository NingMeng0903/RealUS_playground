from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset
from projects.genesis_ue_sync.anatomy_retarget.material_fit import _fit_source_frames
from projects.genesis_ue_sync.anatomy_retarget.source_rebind import rebind_source_rig
from projects.genesis_ue_sync.anatomy_retarget.source_skin_volume import (
    _semantic_skin_surface_correspondence,
)


def _controller_follower_asset() -> AnatomyRiggedAsset:
    parents = np.asarray((-1, 0), dtype=np.int32)
    joints = np.asarray(((0, 0, 0), (0, 1, 0)), dtype=np.float32)
    bind = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    bind[1, :3, 3] = joints[1]
    local = bind.copy()
    vertices = np.asarray(
        ((-0.1, 1.0, 0.0), (0.1, 1.0, 0.0), (0.0, 1.2, 0.1)),
        dtype=np.float32,
    )
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
        faces=np.asarray(((0, 1, 2),), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root", "child"],
        parents=parents,
        rest_joints=joints,
        inverse_bind=np.linalg.inv(bind).astype(np.float32),
        source_mesh_names=["FollowerMesh"],
        source_vertex_ranges=np.asarray(((0, 3),), dtype=np.int32),
        source_tissues=["bone"],
        driver_indices=np.ones((3, 1), dtype=np.int16),
        driver_weights=np.ones((3, 1), dtype=np.float32),
        source_bone_names=["Controller", "DeformFollower"],
        source_bone_parents=parents,
        source_rest_global=bind,
        source_rest_local=local,
        source_inverse_bind=np.linalg.inv(bind).astype(np.float32),
        source_bone_head=joints.copy(),
        source_bone_tail=np.asarray(((0, 1, 0), (0, 1.3, 0)), dtype=np.float32),
        source_bone_roll=np.zeros(2, dtype=np.float32),
        source_bone_use_connect=np.asarray((0, 1), dtype=np.uint8),
        source_bone_inherit_scale=np.zeros(2, dtype=np.uint8),
        source_bone_smplx_a=np.asarray((0, 1), dtype=np.int32),
        source_bone_smplx_b=np.asarray((0, 1), dtype=np.int32),
        source_bone_blend=np.zeros(2, dtype=np.float32),
        source_bone_driver_types=["joint_local", "bind_follow"],
        source_bone_frame_joints=np.asarray(((0, -1, -1), (1, -1, -1)), dtype=np.int32),
        target_rest_global=bind.copy(),
        target_rest_local=local.copy(),
        target_inverse_bind=np.linalg.inv(bind).astype(np.float32),
        target_bone_head=joints.copy(),
        target_bone_tail=np.asarray(((0, 1, 0), (0, 1.3, 0)), dtype=np.float32),
    )


def test_rebind_infers_unweighted_controller_from_weighted_follower() -> None:
    asset = _controller_follower_asset()
    target = np.asarray(asset.vertices_rest) + np.asarray((0.5, 0.0, 0.0), dtype=np.float32)

    rebound, report = rebind_source_rig(
        asset,
        source_vertices=asset.vertices_rest,
        target_vertices=target,
        stage="test",
    )

    assert report["fitted_bones"] == 1
    assert report["controllers_inferred_from_weighted_children"] == 1
    assert report["connected_shared_anchors_enforced"] == 1
    np.testing.assert_allclose(rebound.target_rest_global[:, 0, 3], 0.5, atol=1.0e-6)
    np.testing.assert_allclose(
        rebound.target_bone_tail[0], rebound.target_bone_head[1], atol=1.0e-6
    )


def test_fit_source_frames_preserves_same_semantic_controller_offset() -> None:
    asset = _controller_follower_asset()
    asset = AnatomyRiggedAsset(
        **{
            **asset.__dict__,
            "source_bone_driver_types": ["joint_local", "joint_local"],
            "source_bone_smplx_a": np.asarray((0, 0), dtype=np.int32),
            "source_bone_smplx_b": np.asarray((1, 1), dtype=np.int32),
            "source_bone_frame_joints": np.asarray(
                ((0, -1, -1), (0, -1, -1)), dtype=np.int32
            ),
        }
    )

    fitted_global, fitted_local, _delta = _fit_source_frames(asset)

    assert np.linalg.norm(fitted_global[1, :3, 3] - fitted_global[0, :3, 3]) > 0.5
    np.testing.assert_allclose(
        fitted_local[1, :3, 3], asset.target_bind_local[1, :3, 3], atol=1.0e-6
    )


def test_fit_source_frames_uses_persisted_driver_rest_joint_authority() -> None:
    asset = _controller_follower_asset()
    driver_rest = np.asarray(asset.rest_joints).copy()
    driver_rest[0] = np.asarray((0.25, -0.1, 0.05), dtype=np.float32)
    asset = AnatomyRiggedAsset(
        **{**asset.__dict__, "source_driver_rest_joints": driver_rest}
    )

    fitted_global, _fitted_local, _delta = _fit_source_frames(asset)

    np.testing.assert_allclose(fitted_global[0, :3, 3], driver_rest[0], atol=1.0e-6)


def test_semantic_skin_surface_correspondence_does_not_cross_joint_branch() -> None:
    source = np.asarray(((0.0, 0.0, 0.0), (0.0, 0.1, 0.0)), dtype=np.float64)
    guide = np.asarray(((0.9, 0.0, 0.0), (0.1, 0.1, 0.0)), dtype=np.float64)
    source_weights = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    target = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64)
    target_weights = np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float64)

    mapped, report = _semantic_skin_surface_correspondence(
        source, guide, source_weights, target, target_weights
    )

    np.testing.assert_allclose(mapped, target)
    assert report["fallback_vertex_count"] == 0

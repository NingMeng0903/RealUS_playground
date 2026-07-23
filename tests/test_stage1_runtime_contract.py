from __future__ import annotations

from dataclasses import replace

import numpy as np

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import with_source_driver_coupling
from projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_retarget import (
    _runtime_publication_asset,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset
from projects.genesis_ue_sync.anatomy_retarget.stage1_contract import stage1_runtime_contract


def _chain_asset() -> AnatomyRiggedAsset:
    joints = np.asarray(
        ((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)), dtype=np.float32
    )
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    global_bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    global_bind[:, :3, 3] = joints
    local_bind = global_bind.copy()
    for index in range(1, 4):
        local_bind[index] = np.linalg.inv(global_bind[index - 1]) @ global_bind[index]
    heads = joints.copy()
    return AnatomyRiggedAsset(
        vertices_rest=heads + np.asarray((0.01, 0.0, 0.0), dtype=np.float32),
        faces=np.asarray(((0, 1, 2), (1, 2, 3)), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root", "left_elbow", "left_wrist", "left_index1"],
        parents=parents,
        rest_joints=joints,
        inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_mesh_names=["Upper", "Lower", "Palm", "Finger"],
        source_vertex_ranges=np.asarray(((0, 1), (1, 2), (2, 3), (3, 4)), dtype=np.int32),
        source_tissues=["bone"] * 4,
        source_mesh_controller_bones=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_mesh_material_groups=["skeletal"] * 4,
        source_mesh_roles=["authored_mesh"] * 4,
        driver_indices=np.arange(4, dtype=np.int16)[:, None],
        driver_weights=np.ones((4, 1), dtype=np.float32),
        source_bone_names=["Upper", "Lower", "Palm", "Finger"],
        source_bone_parents=parents.copy(),
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_bone_head=heads,
        source_bone_tail=heads + np.asarray((0, 0.8, 0), dtype=np.float32),
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 2, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=["segment_root", "segment_root", "joint_local", "joint_local"],
        source_bone_frame_joints=np.asarray(
            ((0, 1, -1), (1, 2, -1), (2, 2, -1), (3, 3, -1)), dtype=np.int32
        ),
    )


def _stage1_asset():
    base = _chain_asset()
    target = replace(
        base,
        target_rest_global=np.asarray(base.source_rest_global).copy(),
        target_rest_local=np.asarray(base.source_rest_local).copy(),
        target_inverse_bind=np.asarray(base.source_inverse_bind).copy(),
        target_bone_head=np.asarray(base.source_bone_head).copy(),
        target_bone_tail=np.asarray(base.source_bone_tail).copy(),
    )
    return with_source_driver_coupling(target)


def test_stage1_contract_requires_runtime_rig_and_no_pose_cache() -> None:
    asset = _stage1_asset()
    report = stage1_runtime_contract(asset)

    assert report["passed"] is True
    assert report["requires_blender_at_runtime"] is False
    assert report["requires_pose_rebake"] is False
    assert report["pose_cache_absent"] is True


def test_stage1_contract_rejects_pose_cache() -> None:
    asset = _stage1_asset()
    cached = replace(
        asset,
        pose_cache_vertices=np.asarray(asset.vertices_rest).copy(),
        pose_cache_hash="diagnostic-only-cache",
    )

    report = stage1_runtime_contract(cached)

    assert report["passed"] is False
    assert report["pose_cache_absent"] is False


def test_runtime_publication_strips_offline_pose_cache() -> None:
    asset = _stage1_asset()
    cached = replace(
        asset,
        pose_cache_vertices=np.asarray(asset.vertices_rest).copy(),
        pose_cache_hash="diagnostic-only-cache",
    )

    published = _runtime_publication_asset(cached)

    assert cached.pose_cache_vertices is not None
    assert cached.pose_cache_hash == "diagnostic-only-cache"
    assert published.pose_cache_vertices is None
    assert published.pose_cache_hash == ""
    assert stage1_runtime_contract(published)["passed"] is True

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    axis_angle_to_matrix,
    build_source_driver_coupling,
    skin_vertices,
    source_bone_driver_frames,
    source_bone_skinning_transforms,
    with_source_driver_coupling,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset


def _chain_asset() -> AnatomyRiggedAsset:
    joints = np.asarray(((0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)), dtype=np.float32)
    parents = np.asarray((-1, 0, 1, 2), dtype=np.int32)
    global_bind = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    global_bind[:, :3, 3] = joints
    local_bind = global_bind.copy()
    for index in range(1, 4):
        local_bind[index] = np.linalg.inv(global_bind[index - 1]) @ global_bind[index]
    heads = joints.copy()
    tails = joints + np.asarray((0, 0.8, 0), dtype=np.float32)
    vertices = heads + np.asarray((0.01, 0.0, 0.0), dtype=np.float32)
    return AnatomyRiggedAsset(
        vertices_rest=vertices,
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
        source_bone_tail=tails,
        source_bone_smplx_a=np.asarray((0, 1, 2, 3), dtype=np.int32),
        source_bone_smplx_b=np.asarray((1, 2, 2, 3), dtype=np.int32),
        source_bone_blend=np.zeros(4, dtype=np.float32),
        source_bone_driver_types=["segment_root", "segment_root", "joint_local", "joint_local"],
        source_bone_frame_joints=np.asarray(((0, 1, -1), (1, 2, -1), (2, 2, -1), (3, 3, -1)), dtype=np.int32),
    )


def _with_bind(asset: AnatomyRiggedAsset, global_bind: np.ndarray) -> AnatomyRiggedAsset:
    global_bind = np.asarray(global_bind, dtype=np.float32)
    local_bind = global_bind.copy()
    for index, parent in enumerate(np.asarray(asset.source_bone_parents).tolist()):
        if int(parent) >= 0:
            local_bind[index] = np.linalg.inv(global_bind[int(parent)]) @ global_bind[index]
    return replace(
        asset,
        source_rest_global=global_bind,
        source_rest_local=local_bind,
        source_inverse_bind=np.linalg.inv(global_bind).astype(np.float32),
        source_driver_coupling=None,
    )


def _root_pose_delta(rotvec: np.ndarray) -> np.ndarray:
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = axis_angle_to_matrix(np.asarray(rotvec, dtype=np.float32))[0]
    return delta


def _assert_all_bones_follow_delta(asset: AnatomyRiggedAsset, rotvec: np.ndarray) -> None:
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[0] = rotvec
    actual = source_bone_skinning_transforms(asset, pose)
    expected = np.repeat(_root_pose_delta(rotvec)[None], len(asset.source_bone_names or []), axis=0)
    np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=0.0)


def test_neutral_coupling_recovers_bind_frames() -> None:
    asset = with_source_driver_coupling(_chain_asset())
    frames = source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float32))
    coupling = np.asarray(asset.source_driver_coupling, dtype=np.float64)
    bind = np.asarray(asset.source_rest_global, dtype=np.float64)
    for bone, mode in enumerate(asset.source_bone_driver_types or []):
        if mode == "bind_follow":
            continue
        recovered = frames[bone] @ coupling[bone]
        np.testing.assert_allclose(recovered, bind[bone], atol=1.0e-6)


def test_build_source_driver_coupling_matches_inv_f_times_b() -> None:
    asset = _chain_asset()
    frames = source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float32))
    bind = np.asarray(asset.source_rest_global, dtype=np.float64)
    coupling = build_source_driver_coupling(asset)
    for bone, mode in enumerate(asset.source_bone_driver_types or []):
        if mode == "bind_follow" and int(asset.source_bone_parents[bone]) >= 0:
            np.testing.assert_allclose(coupling[bone], np.eye(4), atol=1.0e-7)
            continue
        expected = np.linalg.inv(frames[bone]) @ bind[bone]
        np.testing.assert_allclose(coupling[bone], expected, atol=1.0e-6)


def test_zero_pose_skinning_is_identity_with_persisted_coupling() -> None:
    asset = with_source_driver_coupling(_chain_asset())
    transforms = source_bone_skinning_transforms(asset, np.zeros((55, 3), dtype=np.float32))
    expected = np.repeat(np.eye(4, dtype=np.float32)[None], len(transforms), axis=0)
    np.testing.assert_array_equal(transforms, expected)


def test_target_bind_is_runtime_authority_without_mutating_source_bind() -> None:
    base = _chain_asset()
    source_before = np.asarray(base.source_bind_global).copy()
    target_global = np.asarray(base.source_bind_global).copy()
    target_global[:, :3, 3] += np.asarray((0.25, -0.1, 0.4), dtype=np.float32)
    target_local = target_global.copy()
    for index, parent in enumerate(np.asarray(base.source_bone_parents).tolist()):
        if int(parent) >= 0:
            target_local[index] = (
                np.linalg.inv(target_global[int(parent)]) @ target_global[index]
            )
    target = with_source_driver_coupling(
        replace(
            base,
            target_rest_global=target_global,
            target_rest_local=target_local,
            target_inverse_bind=np.linalg.inv(target_global).astype(np.float32),
            target_bone_head=np.asarray(base.source_bone_head)
            + np.asarray((0.25, -0.1, 0.4), dtype=np.float32),
            target_bone_tail=np.asarray(base.source_bone_tail)
            + np.asarray((0.25, -0.1, 0.4), dtype=np.float32),
            source_driver_coupling=None,
        )
    )
    np.testing.assert_array_equal(target.source_bind_global, source_before)
    np.testing.assert_array_equal(target.target_bind_global, target_global)
    zero = source_bone_skinning_transforms(
        target, np.zeros((55, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        zero, np.repeat(np.eye(4, dtype=np.float32)[None], len(zero), axis=0)
    )
    frames = source_bone_driver_frames(
        target, np.zeros((55, 3), dtype=np.float32)
    )
    for bone, mode in enumerate(target.source_bone_driver_types or []):
        if mode != "bind_follow":
            np.testing.assert_allclose(
                frames[bone] @ target.source_driver_coupling[bone],
                target_global[bone],
                atol=2.0e-6,
            )


@pytest.mark.parametrize("mode", ["segment_root", "rigid_group"])
def test_segment_and_rigid_group_pure_twist_rotates_attached_bone(mode: str) -> None:
    base = _chain_asset()
    modes = list(base.source_bone_driver_types or [])
    modes[0] = mode
    asset = with_source_driver_coupling(
        replace(base, source_bone_driver_types=modes)
    )
    rest_pose = np.zeros((55, 3), dtype=np.float32)
    twist_pose = rest_pose.copy()
    twist_pose[0, 1] = np.float32(0.73)
    rest_frames = source_bone_driver_frames(asset, rest_pose)
    posed_frames = source_bone_driver_frames(asset, twist_pose)
    delta = _root_pose_delta(twist_pose[0])

    # The joint endpoints lie on Y and do not move under this twist.  Only a
    # transported transverse axis can recover the rotation.
    np.testing.assert_allclose(
        posed_frames[0],
        delta @ rest_frames[0],
        atol=2.0e-6,
        rtol=0.0,
    )
    transforms = source_bone_skinning_transforms(asset, twist_pose)
    np.testing.assert_allclose(transforms[0], delta, atol=2.0e-6, rtol=0.0)
    attached = np.asarray((0.2, 0.4, -0.1, 1.0), dtype=np.float32)
    np.testing.assert_allclose(
        transforms[0] @ attached,
        delta @ attached,
        atol=2.0e-6,
        rtol=0.0,
    )


def test_collinear_three_joint_root_frame_preserves_axial_twist() -> None:
    base = _chain_asset()
    modes = list(base.source_bone_driver_types or [])
    modes[0] = "rigid_group"
    frame_joints = np.asarray(base.source_bone_frame_joints).copy()
    frame_joints[0] = np.asarray((0, 1, 2), dtype=np.int32)
    asset = with_source_driver_coupling(
        replace(
            base,
            source_bone_driver_types=modes,
            source_bone_frame_joints=frame_joints,
        )
    )
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[0, 1] = np.float32(-0.81)
    expected = _root_pose_delta(pose[0])
    transforms = source_bone_skinning_transforms(asset, pose)
    np.testing.assert_allclose(transforms[0], expected, atol=2.0e-6, rtol=0.0)


def test_nonroot_segment_twist_is_not_lost_when_endpoints_are_stationary() -> None:
    asset = with_source_driver_coupling(_chain_asset())
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[1, 1] = np.float32(0.67)
    expected = _root_pose_delta(pose[1])
    transforms = source_bone_skinning_transforms(asset, pose)
    np.testing.assert_allclose(transforms[0], np.eye(4), atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(transforms[1], expected, atol=2.0e-6, rtol=0.0)


def test_segment_root_combined_swing_and_twist_preserves_full_rotation() -> None:
    asset = with_source_driver_coupling(_chain_asset())
    _assert_all_bones_follow_delta(
        asset,
        np.asarray((0.41, -0.62, 0.27), dtype=np.float32),
    )


def test_bind_follow_preserves_authored_parent_child_local_fk() -> None:
    base = _chain_asset()
    followed = replace(
        base,
        source_bone_driver_types=[
            "segment_root",
            "bind_follow",
            "bind_follow",
            "bind_follow",
        ],
    )
    asset = with_source_driver_coupling(followed)
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[0] = np.asarray((0.31, 0.52, -0.23), dtype=np.float32)
    transforms = source_bone_skinning_transforms(asset, pose)
    bind = np.asarray(asset.source_bind_global, dtype=np.float64)
    posed_global = np.asarray(transforms, dtype=np.float64) @ bind
    local_bind = np.asarray(asset.source_bind_local, dtype=np.float64)

    for child in range(1, len(posed_global)):
        parent = int(asset.source_bone_parents[child])
        actual_local = np.linalg.inv(posed_global[parent]) @ posed_global[child]
        np.testing.assert_allclose(
            actual_local,
            local_bind[child],
            atol=3.0e-6,
            rtol=0.0,
        )
    _assert_all_bones_follow_delta(asset, pose[0])


def test_mirrored_source_bind_does_not_change_segment_motion() -> None:
    base = _chain_asset()
    mirrored = np.asarray(base.source_bind_global, dtype=np.float32).copy()
    reflection = np.diag(np.asarray((-1.0, 1.0, 1.0), dtype=np.float32))
    mirrored[:, :3, :3] = reflection
    mirrored[:, 0, 3] = np.asarray((0.2, -0.1, 0.3, -0.25), dtype=np.float32)
    asset = with_source_driver_coupling(_with_bind(base, mirrored))

    _assert_all_bones_follow_delta(
        asset,
        np.asarray((-0.37, 0.58, 0.19), dtype=np.float32),
    )


def test_random_source_binds_and_root_poses_recover_rigid_motion() -> None:
    rng = np.random.default_rng(1947)
    base = _chain_asset()
    for _ in range(20):
        bind = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
        bind[:, :3, :3] = axis_angle_to_matrix(rng.normal(0.0, 0.9, size=(4, 3)))
        bind[:, :3, 3] = rng.normal(0.0, 0.5, size=(4, 3))
        asset = with_source_driver_coupling(_with_bind(base, bind))
        _assert_all_bones_follow_delta(
            asset,
            rng.normal(0.0, 0.8, size=3).astype(np.float32),
        )


def test_authored_bind_and_endpoints_are_detached_immutable_authority() -> None:
    base = _chain_asset()
    bind_input = np.asarray(base.source_bind_global).copy()
    head_input = np.asarray(base.source_bone_head).copy() + np.asarray((0.13, -0.07, 0.04))
    tail_input = np.asarray(base.source_bone_tail).copy() + np.asarray((0.13, -0.07, 0.04))
    asset = replace(
        base,
        source_rest_global=bind_input,
        source_bone_head=head_input,
        source_bone_tail=tail_input,
    )
    expected_bind = np.asarray(asset.source_bind_global).copy()
    expected_head = np.asarray(asset.source_bone_head).copy()
    expected_tail = np.asarray(asset.source_bone_tail).copy()

    bind_input[0, 0, 3] = 99.0
    head_input[0] = 99.0
    tail_input[0] = 100.0
    source_bone_driver_frames(asset, np.zeros((55, 3), dtype=np.float32))

    np.testing.assert_array_equal(asset.source_bind_global, expected_bind)
    np.testing.assert_array_equal(asset.source_bone_head, expected_head)
    np.testing.assert_array_equal(asset.source_bone_tail, expected_tail)
    assert not np.asarray(asset.source_bind_global).flags.writeable
    assert not np.asarray(asset.source_bind_local).flags.writeable
    assert not np.asarray(asset.source_bone_head).flags.writeable
    assert not np.asarray(asset.source_bone_tail).flags.writeable
    with pytest.raises(ValueError):
        asset.source_bind_global[0, 0, 0] = 2.0


@pytest.mark.parametrize("failure", ["unmapped", "degenerate", "bad_coupling"])
def test_invalid_driver_authority_fails_closed(failure: str) -> None:
    asset = _chain_asset()
    if failure == "unmapped":
        invalid_b = np.asarray(asset.source_bone_smplx_b).copy()
        invalid_b[0] = -1
        invalid = replace(asset, source_bone_smplx_b=invalid_b)
        with pytest.raises(ValueError, match="unmapped or invalid"):
            source_bone_driver_frames(invalid, np.zeros((55, 3), dtype=np.float32))
    elif failure == "degenerate":
        invalid_joints = np.asarray(asset.rest_joints).copy()
        invalid_joints[1] = invalid_joints[0]
        invalid = replace(asset, rest_joints=invalid_joints)
        with pytest.raises(ValueError, match="degenerate"):
            source_bone_driver_frames(invalid, np.zeros((55, 3), dtype=np.float32))
    else:
        coupled = with_source_driver_coupling(asset)
        invalid_coupling = np.asarray(coupled.source_driver_coupling).copy()
        invalid_coupling[0, 0, 3] += 0.1
        invalid = replace(coupled, source_driver_coupling=invalid_coupling)
        with pytest.raises(ValueError, match="does not recover"):
            source_bone_skinning_transforms(
                invalid,
                np.zeros((55, 3), dtype=np.float32),
            )


def _mixed_weight_organ_asset() -> AnatomyRiggedAsset:
    base = _chain_asset()
    indices = np.asarray(((0, 1), (1, 1), (2, 2), (3, 3)), dtype=np.int16)
    weights = np.asarray(((0.5, 0.5), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)), dtype=np.float32)
    return with_source_driver_coupling(
        replace(
            base,
            source_tissues=["organ"] * 4,
            driver_indices=indices,
            driver_weights=weights,
        )
    )


def test_matrix_lbs_remains_default_with_dqs_explicitly_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _mixed_weight_organ_asset()
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[1, 1] = np.float32(1.1)
    monkeypatch.setenv("AMONGUS_ANATOMY_LBS_DEVICE", "cpu")
    monkeypatch.delenv("AMONGUS_ANATOMY_DQS", raising=False)
    transforms = source_bone_skinning_transforms(asset, pose)
    selected = transforms[np.asarray(asset.driver_indices, dtype=np.int64)]
    blended = np.sum(
        selected * np.asarray(asset.driver_weights)[..., None, None],
        axis=1,
    )
    vertices = np.asarray(asset.vertices_rest, dtype=np.float32)
    homogeneous = np.concatenate(
        (vertices, np.ones((len(vertices), 1), dtype=np.float32)),
        axis=1,
    )
    expected_lbs = (blended @ homogeneous[..., None])[:, :3, 0]
    default_result = skin_vertices(asset, pose)
    np.testing.assert_allclose(default_result, expected_lbs, atol=1.0e-7, rtol=0.0)

    monkeypatch.setenv("AMONGUS_ANATOMY_DQS", "1")
    opted_in = skin_vertices(asset, pose)
    assert np.linalg.norm(opted_in[0] - default_result[0]) > 1.0e-5


def test_matrix_lbs_cpu_gpu_equivalence_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    asset = _mixed_weight_organ_asset()
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[0] = np.asarray((0.29, -0.41, 0.17), dtype=np.float32)
    pose[1, 2] = np.float32(0.63)
    monkeypatch.delenv("AMONGUS_ANATOMY_DQS", raising=False)
    monkeypatch.setenv("AMONGUS_ANATOMY_LBS_DEVICE", "cpu")
    cpu = skin_vertices(asset, pose, transl=np.asarray((0.2, -0.3, 0.1)))
    monkeypatch.setenv("AMONGUS_ANATOMY_LBS_DEVICE", "cuda")
    gpu = skin_vertices(asset, pose, transl=np.asarray((0.2, -0.3, 0.1)))
    np.testing.assert_allclose(gpu, cpu, atol=2.0e-5, rtol=0.0)

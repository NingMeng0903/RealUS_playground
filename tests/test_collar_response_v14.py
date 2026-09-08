from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import (
    joint_global_transforms,
    source_bone_posed_global,
)
from projects.genesis_ue_sync.anatomy_retarget.collar_response_v14 import (
    COLLAR_BONE_ID_V14,
    COLLAR_RESPONSE_DRIVER_A_V14,
    COLLAR_RESPONSE_DRIVER_B_V14,
    COLLAR_RESPONSE_FRAME_JOINTS_V14,
    COLLAR_RESPONSE_MODE_V14,
    COLLAR_RESPONSE_SCHEMA_V14,
    collar_response_runtime_arrays_v14,
    make_collar_pivot_response_asset_v14,
    make_collar_response_asset_v14,
    restore_collar_response_asset_v14,
    source_bone_local_correction_v14,
    source_bone_local_matrices_collar_v14,
    source_bone_posed_global_collar_v14,
)
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject,
)


ROOT = Path(__file__).resolve().parents[1]
COMPILED = ROOT / "outputs/anatomy_retarget/v14_arm_fit_20260908_001/compiled"
CAPTURE_ROOT = COMPILED.parent
COLLAR = COLLAR_BONE_ID_V14


@pytest.fixture(scope="module")
def source_asset():
    if not COMPILED.is_dir():
        pytest.skip("the captured V14 compiled fixture is not available")
    return load_compiled_subject(COMPILED).source_asset


def _local_matrices(globals_: np.ndarray, parents: np.ndarray) -> np.ndarray:
    global_matrices = np.asarray(globals_, dtype=np.float64)
    result = global_matrices.copy()
    for bone, parent in enumerate(np.asarray(parents, dtype=np.int64).tolist()):
        if parent >= 0:
            result[bone] = np.linalg.inv(global_matrices[parent]) @ global_matrices[bone]
    return result


def _pose(file_name: str) -> np.ndarray:
    return np.asarray(np.load(CAPTURE_ROOT / file_name)["pose"], dtype=np.float32)


def _collar_probe(axis: int, sign: int) -> np.ndarray:
    pose = np.zeros((55, 3), dtype=np.float32)
    pose[13, axis] = np.float32(sign * np.pi / 6.0)
    return pose


def test_effective_copy_changes_only_calibrated_collar_driver(source_asset) -> None:
    original = source_asset
    original_a = np.asarray(original.source_bone_smplx_a).copy()
    original_b = np.asarray(original.source_bone_smplx_b).copy()
    original_frame = np.asarray(original.source_bone_frame_joints).copy()
    original_modes = list(original.source_bone_driver_types)
    original_coupling = np.asarray(original.source_driver_coupling).copy()
    original_metadata = dict(original.metadata or {})

    effective = make_collar_response_asset_v14(original)
    assert effective is not original
    assert original.source_bone_driver_types[COLLAR] == "segment_root"
    assert int(original.source_bone_smplx_a[COLLAR]) == 9
    assert int(original.source_bone_smplx_b[COLLAR]) == 13
    assert tuple(original.source_bone_frame_joints[COLLAR]) == (9, 13, 16)
    assert "source_collar_response_v14" not in original.metadata

    assert effective.source_bone_driver_types[COLLAR] == COLLAR_RESPONSE_MODE_V14
    assert int(effective.source_bone_smplx_a[COLLAR]) == COLLAR_RESPONSE_DRIVER_A_V14
    assert int(effective.source_bone_smplx_b[COLLAR]) == COLLAR_RESPONSE_DRIVER_B_V14
    assert tuple(effective.source_bone_frame_joints[COLLAR]) == COLLAR_RESPONSE_FRAME_JOINTS_V14
    record = effective.metadata["source_collar_response_v14"]
    assert record["schema"] == COLLAR_RESPONSE_SCHEMA_V14
    assert record["world_space_wrist_restore"] is False
    assert record["topology_weights_mesh_immutable"] is True

    # The response declaration changes at one controller only.  The coupling
    # row is rebaked at neutral pose, so row 129 is the only changed coupling.
    changed_a = np.flatnonzero(original_a != effective.source_bone_smplx_a)
    changed_b = np.flatnonzero(original_b != effective.source_bone_smplx_b)
    changed_frame = np.flatnonzero(np.any(original_frame != effective.source_bone_frame_joints, axis=1))
    assert np.array_equal(changed_a, np.asarray([COLLAR]))
    # The original and response declarations intentionally share b=13; only
    # a, mode and explicit frame differ at controller 129.
    assert changed_b.size == 0
    assert np.array_equal(changed_frame, np.asarray([COLLAR]))
    assert [i for i, (a, b) in enumerate(zip(original_modes, effective.source_bone_driver_types)) if a != b] == [COLLAR]
    coupling_delta = np.max(np.abs(original_coupling - effective.source_driver_coupling), axis=(1, 2))
    assert np.array_equal(np.flatnonzero(coupling_delta > 1.0e-8), np.asarray([COLLAR]))
    other_rows = np.arange(len(original_coupling)) != COLLAR
    np.testing.assert_array_equal(
        effective.source_driver_coupling[other_rows], original_coupling[other_rows]
    )

    # The fitted geometry, sparse weights, hierarchy and both bind frames are
    # shared by the effective copy; this is a response rebake, not a rebind.
    for field in (
        "vertices_rest",
        "faces",
        "driver_indices",
        "driver_weights",
        "source_influence_offsets",
        "source_influence_group_indices",
        "source_influence_values",
        "source_group_bone_indices",
        "source_rest_global",
        "source_rest_local",
        "source_inverse_bind",
        "target_bind_global",
        "target_bind_local",
        "target_inverse_bind",
    ):
        np.testing.assert_array_equal(getattr(effective, field), getattr(original, field))
    assert original.metadata == original_metadata

    effective.validate()
    assert make_collar_response_asset_v14(effective) is effective


def test_baked_array_view_is_detached_and_restore_is_inverse(source_asset) -> None:
    effective = make_collar_response_asset_v14(source_asset)
    arrays = collar_response_runtime_arrays_v14(source_asset)
    assert set(arrays) == {
        "source_bone_smplx_a",
        "source_bone_smplx_b",
        "source_bone_frame_joints",
        "source_driver_coupling",
        "source_bone_driver_types",
    }
    arrays["source_bone_smplx_a"][COLLAR] = 0
    arrays["source_driver_coupling"][COLLAR, 0, 0] = 123.0
    assert int(effective.source_bone_smplx_a[COLLAR]) == COLLAR_RESPONSE_DRIVER_A_V14
    assert float(effective.source_driver_coupling[COLLAR, 0, 0]) != 123.0

    restored = restore_collar_response_asset_v14(effective)
    assert restored.metadata == source_asset.metadata
    np.testing.assert_array_equal(restored.source_bone_smplx_a, source_asset.source_bone_smplx_a)
    np.testing.assert_array_equal(restored.source_bone_smplx_b, source_asset.source_bone_smplx_b)
    np.testing.assert_array_equal(restored.source_bone_frame_joints, source_asset.source_bone_frame_joints)
    assert restored.source_bone_driver_types == source_asset.source_bone_driver_types
    np.testing.assert_allclose(restored.source_driver_coupling, source_asset.source_driver_coupling, atol=2.0e-6, rtol=0.0)
    restored.validate()
    assert restore_collar_response_asset_v14(source_asset) is source_asset


def test_j13_pivot_rebuilds_target_bind_triple_without_mesh_or_weight_edits(source_asset) -> None:
    original = source_asset
    original_target_global = np.asarray(original.target_bind_global).copy()
    original_target_local = np.asarray(original.target_bind_local).copy()
    original_target_inverse = np.asarray(original.target_inverse_bind).copy()
    original_coupling = np.asarray(original.source_driver_coupling).copy()
    original_vertices = np.asarray(original.vertices_rest).copy()
    original_weights = np.asarray(original.driver_weights).copy()
    original_faces = np.asarray(original.faces).copy()

    pivot = make_collar_pivot_response_asset_v14(original)
    assert pivot is not original
    assert pivot.metadata["source_collar_response_v14"]["variant"] == "joint13_pivot"
    assert pivot.metadata["source_collar_response_v14"]["rest_refit_performed"] is False

    target_global = np.asarray(pivot.target_bind_global, dtype=np.float64)
    target_local = np.asarray(pivot.target_bind_local, dtype=np.float64)
    target_inverse = np.asarray(pivot.target_inverse_bind, dtype=np.float64)
    parents = np.asarray(pivot.source_bone_parents, dtype=np.int64)
    expected_local = _local_matrices(target_global, parents)
    np.testing.assert_allclose(target_local, expected_local, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(
        target_global @ target_inverse,
        np.repeat(np.eye(4)[None], len(target_global), axis=0),
        atol=2.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        target_global[COLLAR, :3, 3],
        np.asarray(original.rest_joints, dtype=np.float64)[13],
        atol=2.0e-6,
        rtol=0.0,
    )
    np.testing.assert_array_equal(target_global[130], original_target_global[130])
    np.testing.assert_array_equal(target_global[np.arange(235) != COLLAR], original_target_global[np.arange(235) != COLLAR])
    np.testing.assert_array_equal(target_global[COLLAR, :3, :3], original_target_global[COLLAR, :3, :3])

    # The response/coupling change is scoped to controller 129.  Target local
    # binds for its descendants necessarily change because their parent global
    # bind is now J13; source geometry and sparse material ownership do not.
    np.testing.assert_array_equal(
        pivot.source_driver_coupling[np.arange(235) != COLLAR],
        original_coupling[np.arange(235) != COLLAR],
    )
    assert not np.array_equal(pivot.source_driver_coupling[COLLAR], original_coupling[COLLAR])
    np.testing.assert_array_equal(pivot.vertices_rest, original_vertices)
    np.testing.assert_array_equal(pivot.driver_weights, original_weights)
    np.testing.assert_array_equal(pivot.faces, original_faces)
    np.testing.assert_array_equal(original.target_bind_global, original_target_global)
    np.testing.assert_array_equal(original.target_bind_local, original_target_local)
    np.testing.assert_array_equal(original.target_inverse_bind, original_target_inverse)

    neutral = np.zeros((55, 3), dtype=np.float32)
    posed = source_bone_posed_global(pivot, neutral)
    np.testing.assert_allclose(posed, target_global, atol=2.0e-6, rtol=0.0)
    skinning = posed @ target_inverse
    np.testing.assert_allclose(
        skinning,
        np.repeat(np.eye(4)[None], len(skinning), axis=0),
        atol=3.0e-6,
        rtol=0.0,
    )
    assert make_collar_pivot_response_asset_v14(pivot) is pivot

    malformed_metadata = dict(pivot.metadata)
    malformed_record = dict(malformed_metadata["source_collar_response_v14"])
    malformed_record["response_mode"] = "segment_root"
    malformed_metadata["source_collar_response_v14"] = malformed_record
    malformed = replace(pivot, metadata=malformed_metadata)
    with pytest.raises(ValueError, match="metadata does not match"):
        make_collar_pivot_response_asset_v14(malformed)


def test_neutral_pose_preserves_fitted_bind_and_local_delta_identity(source_asset) -> None:
    pose = np.zeros((55, 3), dtype=np.float32)
    original = source_bone_posed_global(source_asset, pose)
    corrected = source_bone_posed_global_collar_v14(source_asset, pose)
    np.testing.assert_allclose(original, source_asset.target_bind_global, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(corrected, source_asset.target_bind_global, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(corrected, original, atol=2.0e-6, rtol=0.0)

    parents = np.asarray(source_asset.source_bone_parents, dtype=np.int64)
    original_local = _local_matrices(original, parents)
    corrected_local = _local_matrices(corrected, parents)
    delta = source_bone_local_correction_v14(source_asset, pose)
    np.testing.assert_allclose(original_local @ delta, corrected_local, atol=3.0e-12, rtol=0.0)
    np.testing.assert_allclose(
        source_bone_local_matrices_collar_v14(source_asset, pose),
        corrected_local,
        atol=3.0e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize("sign", (-1, 1))
def test_collar_probe_uses_joint13_rotation_but_keeps_parent_anchor(source_asset, axis: int, sign: int) -> None:
    pose = _collar_probe(axis, sign)
    effective = make_collar_response_asset_v14(source_asset)
    original = source_bone_posed_global(source_asset, pose)
    corrected = source_bone_posed_global_collar_v14(source_asset, pose)
    target = joint_global_transforms(
        pose_axis_angle=pose,
        rest_joints=source_asset.rest_joints,
        parents=source_asset.parents,
    ).astype(np.float64)
    rest_target = joint_global_transforms(
        pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
        rest_joints=source_asset.rest_joints,
        parents=source_asset.parents,
    ).astype(np.float64)
    expected = (target[13] @ np.linalg.inv(rest_target[13])) @ np.asarray(
        source_asset.target_bind_global[COLLAR], dtype=np.float64
    )

    # The calibrated response corrects orientation from local joint 13.  Its
    # origin still comes from source parent-local FK, so it stays at the old
    # sternoclavicular-side anchor rather than taking the detached Delta*B
    # translation from the diagnostic world-space construction.
    np.testing.assert_allclose(corrected[COLLAR, :3, :3], expected[:3, :3], atol=3.0e-6, rtol=0.0)
    np.testing.assert_allclose(corrected[COLLAR, :3, 3], original[COLLAR, :3, 3], atol=3.0e-9, rtol=0.0)
    assert np.linalg.norm(corrected[COLLAR, :3, :3] - original[COLLAR, :3, :3]) > 1.0e-4

    # 130 bind-follow inherits the collar through the same local bind offset,
    # while 131 and its hand chain retain their original desired world
    # rotations.  No independent wrist position is injected.
    np.testing.assert_allclose(
        np.linalg.inv(corrected[COLLAR]) @ corrected[130],
        np.linalg.inv(original[COLLAR]) @ original[130],
        atol=3.0e-6,
        rtol=0.0,
    )
    for bone in (131, 132, 133, 134, 135):
        np.testing.assert_allclose(corrected[bone, :3, :3], original[bone, :3, :3], atol=3.0e-6, rtol=0.0)
    excluded = {COLLAR, 130}
    for bone in range(len(corrected)):
        if bone not in excluded:
            np.testing.assert_allclose(corrected[bone, :3, :3], original[bone, :3, :3], atol=5.0e-6, rtol=0.0)

    parents = np.asarray(source_asset.source_bone_parents, dtype=np.int64)
    original_local = _local_matrices(original, parents)
    corrected_local = _local_matrices(corrected, parents)
    local_delta = source_bone_local_correction_v14(source_asset, pose)
    np.testing.assert_allclose(original_local @ local_delta, corrected_local, atol=3.0e-12, rtol=0.0)
    rotations = np.asarray(local_delta[:, :3, :3], dtype=np.float64)
    np.testing.assert_allclose(
        np.einsum("bij,bkj->bik", rotations, rotations),
        np.repeat(np.eye(3)[None], len(rotations), axis=0),
        atol=5.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=5.0e-6, rtol=0.0)
    assert effective is make_collar_response_asset_v14(effective)


@pytest.mark.parametrize(
    "file_name",
    (
        "subject_213328_tpose.npz",
        "subject_213328_pose_213328.npz",
        "subject_213328_pose_213712.npz",
        "subject_213328_heldout_sitting.npz",
        "subject_213328_heldout_kicking.npz",
    ),
)
def test_captured_and_heldout_fk_preserves_hand_chain_rotation_and_sc_anchor(source_asset, file_name: str) -> None:
    pose = _pose(file_name)
    original = source_bone_posed_global(source_asset, pose)
    corrected = source_bone_posed_global_collar_v14(source_asset, pose)
    target = joint_global_transforms(
        pose_axis_angle=pose,
        rest_joints=source_asset.rest_joints,
        parents=source_asset.parents,
    ).astype(np.float64)

    # 129 remains on its parent-107 path near SMPL-X J9 in every capture.
    anchor_error_mm = np.linalg.norm(
        corrected[COLLAR, :3, 3] - target[9, :3, 3]
    ) * 1000.0
    assert anchor_error_mm < 1.0
    np.testing.assert_allclose(
        corrected[COLLAR, :3, 3], original[COLLAR, :3, 3], atol=3.0e-9, rtol=0.0
    )

    # The whole hand is evaluated from the same 235-bone FK.  Its positions
    # can move with the collar, but downstream desired orientations are not
    # replaced by a world-space wrist/finger driver.
    for bone in (131, 132, 133, 134, 135):
        np.testing.assert_allclose(
            corrected[bone, :3, :3], original[bone, :3, :3], atol=5.0e-6, rtol=0.0
        )
    keep = np.asarray([i not in {COLLAR, 130} for i in range(len(corrected))])
    np.testing.assert_allclose(
        corrected[keep, :3, :3],
        original[keep, :3, :3],
        atol=8.0e-6,
        rtol=0.0,
    )

    # This captures the useful but bounded geometric effect: the candidate
    # improves the original 131-to-J16 origin drift in 213328 and the two
    # held-out actions, while 213712 remains an allowed regression requiring
    # later calibration rather than being hidden by a pass/fail claim.
    error_original_mm = np.linalg.norm(original[131, :3, 3] - target[16, :3, 3]) * 1000.0
    error_corrected_mm = np.linalg.norm(corrected[131, :3, 3] - target[16, :3, 3]) * 1000.0
    if file_name in {
        "subject_213328_pose_213328.npz",
        "subject_213328_heldout_sitting.npz",
        "subject_213328_heldout_kicking.npz",
    }:
        assert error_corrected_mm < error_original_mm
    assert np.isfinite(error_original_mm)
    assert np.isfinite(error_corrected_mm)


def test_noncanonical_collar_contract_fails_closed(source_asset) -> None:
    modes = list(source_asset.source_bone_driver_types)
    modes[COLLAR] = "joint_local"
    malformed = replace(source_asset, source_bone_driver_types=modes)
    with pytest.raises(ValueError, match="expected 9->13 explicit frame"):
        make_collar_response_asset_v14(malformed)

    frame = np.asarray(source_asset.source_bone_frame_joints).copy()
    frame[COLLAR] = (9, 13, -1)
    malformed = replace(source_asset, source_bone_frame_joints=frame)
    with pytest.raises(ValueError, match="expected 9->13 explicit frame"):
        make_collar_response_asset_v14(malformed)

    effective = make_collar_response_asset_v14(source_asset)
    metadata = dict(effective.metadata)
    record = dict(metadata["source_collar_response_v14"])
    record["bone_id"] = 128
    metadata["source_collar_response_v14"] = record
    malformed = replace(effective, metadata=metadata)
    with pytest.raises(ValueError, match="metadata does not match"):
        make_collar_response_asset_v14(malformed)

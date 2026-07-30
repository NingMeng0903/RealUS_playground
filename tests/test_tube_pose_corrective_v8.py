from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from projects.genesis_ue_sync.anatomy_retarget.tube_pose_corrective_v8 import (
    TubePoseCorrectivePackV1,
    apply_tube_pose_corrective_v1,
    bake_tube_pose_corrective_v1,
    evaluate_tube_pose_corrective_local_v1,
    tube_pose_corrective_pack_from_runtime_fields_v1,
    tube_pose_corrective_pack_to_runtime_fields_v1,
)
from projects.genesis_ue_sync.anatomy_retarget.tube_frames_v8 import (
    TubeCouplingPackV8,
    _rest_digest,
    _weight_digest,
    tube_coupling_pack_to_runtime_fields_v8,
)
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    ResidentPoseEvaluatorV8,
    SubjectRuntimePackV8,
    _subject_cache_key_for_solver_version,
    _runtime_tube_packs_v8,
    save_subject_runtime,
)
from projects.genesis_ue_sync.anatomy_retarget.rigged_asset import AnatomyRiggedAsset
from projects.genesis_ue_sync.anatomy_retarget.version_v8 import (
    SUBJECT_SOLVER_VERSION,
)


def _pack() -> TubePoseCorrectivePackV1:
    basis = np.zeros((2, 3, 1), dtype=np.float32)
    basis[0, :, 0] = (0.010, 0.000, 0.000)
    basis[1, :, 0] = (0.000, 0.020, 0.000)
    return TubePoseCorrectivePackV1(
        vertex_ids=np.asarray((1, 3), dtype=np.int32),
        local_displacement_basis=basis,
        driver_joint_ids=np.asarray((4,), dtype=np.int16),
        rbf_centers=np.zeros((1, 1, 3), dtype=np.float32),
        rbf_widths=np.asarray((1.0,), dtype=np.float32),
        center_coefficients=np.asarray(((1.0,),), dtype=np.float32),
        maximum_displacement_m=0.1,
    )


def _transforms() -> np.ndarray:
    result = np.tile(np.eye(4, dtype=np.float64), (8, 1, 1))
    # The translation is deliberately nonzero: local correction vectors must
    # only see the linear part of the matrices.
    result[2, :3, 3] = (9.0, 8.0, 7.0)
    result[4, :3, 3] = (-3.0, 2.0, 1.0)
    angle = np.pi / 2.0
    result[2, :3, :3] = np.asarray(
        ((np.cos(angle), -np.sin(angle), 0.0), (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return result


def _selected_weights() -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros((2, 14), dtype=np.int16)
    weights = np.zeros((2, 14), dtype=np.float32)
    # First correction uses a genuine two-bone linear blend, not a polar or
    # dual-quaternion frame.  Second is driven by one source bone.
    indices[0, :2] = (2, 4)
    weights[0, :2] = (0.25, 0.75)
    indices[1, 0] = 4
    weights[1, 0] = 1.0
    return indices, weights


def _tube_pack() -> TubeCouplingPackV8:
    vertex_ids = np.asarray((1, 3, 5), dtype=np.int32)
    rest = np.asarray(
        ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)),
        dtype=np.float32,
    )
    indices = np.zeros((3, 14), dtype=np.int16)
    weights = np.zeros((3, 14), dtype=np.float32)
    weights[:, 0] = 1.0
    return TubeCouplingPackV8(
        vertex_count=6,
        vertex_ids=vertex_ids,
        rest_vertices_m=rest,
        driver_indices=indices,
        driver_weights=weights,
        material_edges=np.empty((0, 2), dtype=np.int32),
        mesh_indices=np.asarray((0,), dtype=np.int32),
        mesh_vertex_ranges=np.asarray(((0, 3),), dtype=np.int32),
        topology_digest="a" * 64,
        domain_digest="b" * 64,
        rest_digest=_rest_digest(vertex_ids, rest),
        weight_digest=_weight_digest(vertex_ids, indices, weights),
    )


def _legacy_subject() -> SubjectRuntimePackV8:
    identity = np.eye(4, dtype=np.float32)[None]
    rig = AnatomyRiggedAsset(
        vertices_rest=np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32),
        faces=np.empty((0, 3), dtype=np.int32),
        lbs_weights=None,
        joint_names=["root"],
        parents=np.asarray((-1,), dtype=np.int32),
        rest_joints=np.zeros((1, 3), dtype=np.float32),
        inverse_bind=identity,
        source_mesh_names=["Bone"],
        source_vertex_ranges=np.asarray(((0, 1),), dtype=np.int32),
        source_tissues=["bone"],
        source_mesh_controller_bones=np.asarray((0,), dtype=np.int32),
        source_mesh_material_groups=["skeletal"],
        source_mesh_roles=["authored_mesh"],
        source_fit_policies=["rigid"],
        source_driver_policies=["source_rig"],
        source_compound_ids=["root"],
        source_sides=["center"],
        source_landmarks=[tuple()],
        target_landmark_recipes=["none"],
        source_quality_profiles=["bone"],
        driver_indices=np.zeros((1, 1), dtype=np.int16),
        driver_weights=np.ones((1, 1), dtype=np.float32),
        source_bone_names=["root"],
        source_bone_parents=np.asarray((-1,), dtype=np.int32),
        source_rest_global=identity,
        source_rest_local=identity,
        source_inverse_bind=identity,
        source_bone_head=np.zeros((1, 3), dtype=np.float32),
        source_bone_tail=np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32),
        source_bone_smplx_a=np.asarray((0,), dtype=np.int32),
        source_bone_smplx_b=np.asarray((0,), dtype=np.int32),
        source_bone_blend=np.zeros(1, dtype=np.float32),
        source_bone_driver_types=["joint_local"],
        source_bone_frame_joints=np.asarray(((0, -1, -1),), dtype=np.int32),
        source_driver_coupling=identity,
        source_driver_rest_joints=np.zeros((1, 3), dtype=np.float32),
        metadata={"source_full_local_fk_v2": True, "disable_soft_follow": True},
    )
    old_solver = "contact-first-leg-v8.10-frozen-route"
    identity_digest = "a" * 64
    reference_digest = "b" * 64
    return SubjectRuntimePackV8(
        rigged_asset=rig,
        operator_runtime_digest=identity_digest,
        reference_digest=reference_digest,
        betas=np.zeros(10, dtype=np.float32),
        gender="male",
        algorithm_version="algorithm-v8.10",
        oracle_version="oracle-v8.10",
        correction_version="correction-v8.10",
        cache_key=_subject_cache_key_for_solver_version(
            operator_runtime_digest=identity_digest,
            betas=np.zeros(10, dtype=np.float32),
            gender="male",
            algorithm_version="algorithm-v8.10",
            oracle_version="oracle-v8.10",
            correction_version="correction-v8.10",
            reference_digest=reference_digest,
            subject_solver_version=old_solver,
        ),
        internal_handle_displacements=np.zeros((1, 3), dtype=np.float32),
        runtime_coefficients={},
        skinning_csr_offsets=np.asarray((0, 1), dtype=np.int64),
        skinning_csr_indices=np.asarray((0,), dtype=np.int32),
        skinning_csr_weights=np.asarray((1.0,), dtype=np.float32),
        audit_report={"publishable": False},
        cache_solver_version=old_solver,
    )


def test_apply_scatter_adds_lbs_linear_local_offsets() -> None:
    pack = _pack()
    indices, weights = _selected_weights()
    posed = np.zeros((5, 3), dtype=np.float32)
    pose = np.zeros((55, 3), dtype=np.float32)

    result = apply_tube_pose_corrective_v1(
        posed,
        pack,
        pose_axis_angle=pose,
        source_transforms=_transforms(),
        driver_indices=indices,
        driver_weights=weights,
    )

    rotation = _transforms()[2, :3, :3]
    expected_first = (0.25 * rotation + 0.75 * np.eye(3)) @ np.asarray(
        (0.010, 0.0, 0.0)
    )
    np.testing.assert_allclose(result[1], expected_first, atol=1.0e-7, rtol=0.0)
    np.testing.assert_allclose(result[3], (0.0, 0.020, 0.0), atol=1.0e-7, rtol=0.0)
    np.testing.assert_array_equal(result[[0, 2, 4]], posed[[0, 2, 4]])


def test_flat_runtime_fields_authenticate_every_array() -> None:
    pack = _pack()
    fields = tube_pose_corrective_pack_to_runtime_fields_v1(pack)
    assert all(isinstance(value, np.ndarray) for value in fields.values())
    assert fields["tube_pose_corrective_v1.content_digest"].dtype == np.uint8
    assert fields["tube_pose_corrective_v1.maximum_displacement_m"].shape == ()

    restored = tube_pose_corrective_pack_from_runtime_fields_v1(fields)
    assert restored.content_digest() == pack.content_digest()
    np.testing.assert_array_equal(restored.vertex_ids, pack.vertex_ids)
    assert restored.local_displacement_basis.flags.writeable is False

    tampered = {name: value.copy() for name, value in fields.items()}
    tampered["tube_pose_corrective_v1.center_coefficients"][0, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match="content digest mismatch"):
        tube_pose_corrective_pack_from_runtime_fields_v1(tampered)

    unknown = dict(fields)
    unknown["tube_pose_corrective_v1.unreviewed_override"] = np.asarray(
        1, dtype=np.int32
    )
    with pytest.raises(ValueError, match="unknown fields"):
        tube_pose_corrective_pack_from_runtime_fields_v1(unknown)


def test_runtime_fields_restore_from_mmap_and_require_tube_subset(tmp_path) -> None:
    pack = _pack()
    fields = tube_pose_corrective_pack_to_runtime_fields_v1(pack)
    mmap_fields = {}
    for index, (name, value) in enumerate(fields.items()):
        path = tmp_path / f"field-{index}.npy"
        np.save(path, value, allow_pickle=False)
        mmap_fields[name] = np.load(path, allow_pickle=False, mmap_mode="r")
    restored = tube_pose_corrective_pack_from_runtime_fields_v1(mmap_fields)
    assert restored.content_digest() == pack.content_digest()

    runtime_fields = tube_coupling_pack_to_runtime_fields_v8(_tube_pack())
    runtime_fields.update(fields)
    tube, corrective = _runtime_tube_packs_v8(
        runtime_fields, label="test runtime coefficients"
    )
    assert tube is not None
    assert corrective is not None
    np.testing.assert_array_equal(corrective.vertex_ids, (1, 3))

    invalid = TubePoseCorrectivePackV1(
        vertex_ids=np.asarray((1, 4), dtype=np.int32),
        local_displacement_basis=pack.local_displacement_basis,
        driver_joint_ids=pack.driver_joint_ids,
        rbf_centers=pack.rbf_centers,
        rbf_widths=pack.rbf_widths,
        center_coefficients=pack.center_coefficients,
        maximum_displacement_m=pack.maximum_displacement_m,
    )
    runtime_fields.update(tube_pose_corrective_pack_to_runtime_fields_v1(invalid))
    with pytest.raises(ValueError, match="outside the frozen tube domain"):
        _runtime_tube_packs_v8(runtime_fields, label="test runtime coefficients")


def test_bake_rbf_pca_reconstructs_local_training_samples() -> None:
    poses = np.zeros((3, 55, 3), dtype=np.float32)
    poses[1, 8, 0] = 0.5
    poses[2, 8, 0] = 1.0
    direction = np.asarray(
        ((0.002, -0.001, 0.0), (0.0, 0.003, 0.001)), dtype=np.float32
    )
    samples = np.stack((np.zeros_like(direction), 0.5 * direction, direction), axis=0)

    pack, report = bake_tube_pose_corrective_v1(
        np.asarray((2, 5), dtype=np.int32),
        poses,
        samples,
        np.asarray((8,), dtype=np.int16),
        maximum_components=1,
        explained_variance_ratio=1.0,
        rbf_ridge=1.0e-10,
    )

    assert pack.component_count == 1
    assert report["runtime_spatial_query"] is False
    for expected, pose in zip(samples, poses):
        actual = evaluate_tube_pose_corrective_local_v1(pack, pose)
        np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=0.0)


def test_apply_rejects_non_14_slot_weight_arrays() -> None:
    pack = _pack()
    with pytest.raises(ValueError, match="exactly 14"):
        apply_tube_pose_corrective_v1(
            np.zeros((4, 3), dtype=np.float32),
            pack,
            pose_axis_angle=np.zeros((55, 3), dtype=np.float32),
            source_transforms=_transforms(),
            driver_indices=np.zeros((2, 2), dtype=np.int16),
            driver_weights=np.ones((2, 2), dtype=np.float32) / 2.0,
        )


def test_legacy_full_fk_subject_keeps_its_authenticated_cache_key_for_playback_only(
    tmp_path,
) -> None:
    subject = _legacy_subject()
    subject.validate()
    posed = ResidentPoseEvaluatorV8(subject).apply_pose(
        np.zeros((55, 3), dtype=np.float32)
    )
    np.testing.assert_array_equal(posed, subject.rigged_asset.vertices_rest)

    with pytest.raises(ValueError):
        save_subject_runtime(tmp_path / "legacy-subject", subject)

    with pytest.raises(ValueError, match="cache_key"):
        replace(
            subject, cache_solver_version=SUBJECT_SOLVER_VERSION
        ).validate()

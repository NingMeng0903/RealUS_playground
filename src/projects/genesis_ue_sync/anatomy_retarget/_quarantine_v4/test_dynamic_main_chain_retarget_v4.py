from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    JOINT_SPECS,
    _measure_frames,
    load_anatomical_calibration_v1,
)
from src.projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_retarget_v4 import (
    EXPECTED_POSE_LABELS_V4,
    _carry_target_rest_frames,
    _channel_basis_active_mask,
    _channel_basis_controller_indices,
    _pose_with_target_local_bind,
    _rotation_between_vectors,
    _solve_multi_pose_main_chain,
    _source_baked_parent_local_pose,
    build_dynamic_main_chain_retarget_v4,
)
from src.projects.genesis_ue_sync.anatomy_retarget.dynamic_main_chain_validation_v4 import (
    AXIS_LIMIT_DEG_V4,
    PIVOT_LIMIT_M_V4,
    ZERO_MATRIX_LIMIT_V4,
    ZERO_VERTEX_LIMIT_M_V4,
    _external_target_frames_v4,
    _transport_contract,
    check_dynamic_main_chain_retarget_v4,
)
from src.projects.genesis_ue_sync.anatomy_retarget.pose_adapter import (
    easymocap_fit_to_smplx55,
)
from src.projects.genesis_ue_sync.anatomy_retarget.smplx_body_surface_v7 import (
    load_smplx_model_v7,
)
from src.projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import (
    load_source_operator,
    materialize_subject,
)


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = ROOT / "outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8"
CALIBRATION = (
    ROOT
    / "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006"
    / "anatomical_calibration_v1"
)
ORACLE = (
    ROOT
    / "outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001"
    / "blender_link_oracle_v7.npz"
)
MODEL = ROOT / "ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl"
CAPTURES = {
    label: ROOT / f"smplx_outputs/20260713_{label}/moment_0000/smplx_result.npz"
    for label in ("213328", "213712")
}


def _test_rotation(axis: int, angle: float) -> np.ndarray:
    first = (axis + 1) % 3
    second = (axis + 2) % 3
    result = np.eye(3, dtype=np.float64)
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result[first, first] = cosine
    result[first, second] = -sine
    result[second, first] = sine
    result[second, second] = cosine
    return result


def test_target_frames_use_parent_roots_and_proximal_male_segment_motion() -> None:
    source = inspect.getsource(_carry_target_rest_frames)
    assert "source_parents" not in source
    assert "parents[int(controllers[row])]" in source
    assert "_smplx_joint_kinematics_v7" in source
    assert "station_rotation[proximal_row] @ rest_vector" in source
    assert "station_rotation @ fixed[:, :3, :3]" in source


def test_external_target_checker_does_not_call_candidate_target_helper() -> None:
    source = inspect.getsource(_external_target_frames_v4)
    assert "_carry_target_rest_frames" not in source
    assert "_external_source_baked_parent_local_pose" in source
    assert "source_parents[int(controllers[row])]" in source
    assert "station_rotation[proximal_row] @ rest_vector" in source
    assert "station_rotation @ target_rest[:, :3, :3]" in source
    assert "_smplx_joint_kinematics_v7" in source
    assert "source_bone_driver_frames" in source


def test_identity_target_bind_preserves_authored_parent_local_motion() -> None:
    parents = np.asarray((-1, 0, 1), dtype=np.int64)
    rest_local = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    rest_local[1, :3, 3] = np.asarray((0.2, -0.1, 0.03))
    rest_local[2, :3, 3] = np.asarray((0.4, 0.02, -0.05))
    basis = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    basis[1, :3, :3] = _test_rotation(2, 0.73)
    basis[1, :3, 3] = np.asarray((0.004, -0.002, 0.001))
    basis[2, :3, :3] = _test_rotation(0, -0.41)
    basis[2, :3, 3] = np.asarray((-0.003, 0.006, 0.002))

    def fk(local: np.ndarray) -> np.ndarray:
        global_frames = np.empty_like(local)
        for row, parent in enumerate(parents.tolist()):
            global_frames[row] = (
                local[row]
                if parent < 0
                else global_frames[parent] @ local[row]
            )
        return global_frames

    rest_global = fk(rest_local)
    posed_global = fk(rest_local @ basis)
    actual = _pose_with_target_local_bind(
        B_prefit=rest_global,
        B_final=rest_global,
        source_posed_global=posed_global,
        parents=parents,
        calibration=None,
        controller_local_pivots=np.empty((0, 3), dtype=np.float64),
    )
    np.testing.assert_allclose(actual, posed_global, atol=1.0e-12, rtol=0.0)


def test_rotation_between_vectors_maps_direction_without_scale() -> None:
    source = np.asarray((0.1, -0.9, 0.3), dtype=np.float64)
    target = np.asarray((-0.4, -0.2, 0.8), dtype=np.float64)
    rotation = _rotation_between_vectors(source, target)
    np.testing.assert_allclose(
        rotation @ (source / np.linalg.norm(source)),
        target / np.linalg.norm(target),
        atol=1.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-12)


def test_arch_channel_basis_slots_are_frozen() -> None:
    active = _channel_basis_active_mask()
    assert active.shape == (18,)
    assert np.array_equal(np.flatnonzero(active == 0.0), np.asarray((4, 9)))


def test_multi_pose_solver_uses_zero_correction_142_baseline() -> None:
    source = inspect.getsource(_solve_multi_pose_main_chain)
    assert "raw_segment_rotation = torch.zeros" in source
    assert "raw_axial_handle = torch.zeros" in source
    assert "raw_root_translation = torch.zeros" in source
    assert "raw_segment_rotation.copy_" not in source
    assert "raw_axial_handle.copy_" not in source
    assert "raw_root_translation.copy_" not in source
    assert "zero_correction_142_baseline_v4" in source
    assert "candidate_raw_score[0]" in source
    assert "candidate_score[2] == 0" in source
    assert "trial_regressed == 0" in source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def v4_matrix():
    required = (OPERATOR, CALIBRATION, ORACLE, MODEL, *CAPTURES.values())
    if not all(path.exists() for path in required):
        pytest.skip("frozen V4 matrix inputs are unavailable")
    operator = load_source_operator(OPERATOR, mmap=True)
    calibration = load_anatomical_calibration_v1(
        CALIBRATION, operator=operator, required_scope="full_main_chain"
    )
    model = load_smplx_model_v7(MODEL)
    model_sha = _sha256(MODEL)
    betas: dict[str, np.ndarray] = {}
    poses: dict[str, np.ndarray] = {}
    for label, capture_path in CAPTURES.items():
        with np.load(capture_path, allow_pickle=False) as capture:
            betas[label] = np.asarray(
                capture["shapes"], dtype=np.float64
            ).reshape(-1)[:10]
            poses[f"pose_{label}"] = np.asarray(
                easymocap_fit_to_smplx55(
                    capture["Rh"], capture["poses"], model_path=MODEL
                ),
                dtype=np.float64,
            )
    values = {}
    reports = {}
    for label in ("213328", "213712"):
        value = build_dynamic_main_chain_retarget_v4(
            operator,
            calibration,
            betas=betas[label],
            subject_label=label,
            capture_sha256=_sha256(CAPTURES[label]),
            smplx_model=model,
            smplx_model_sha256=model_sha,
            recorded_poses=poses,
            gender="male",
        )
        values[label] = value
        reports[label] = check_dynamic_main_chain_retarget_v4(
            value,
            operator=operator,
            calibration=calibration,
            smplx_model=model,
            smplx_model_path=MODEL,
            capture_paths=CAPTURES,
            oracle_path=ORACLE,
        )
    return operator, calibration, model, values, reports


def test_two_beta_checker_rebuilds_all_external_trust_roots(v4_matrix) -> None:
    _operator, _calibration, _model, _values, reports = v4_matrix
    for report in reports.values():
        assert report["passed"] is True, json.dumps(
            report, indent=2, sort_keys=True
        )
        assert report["provenance"]["passed"] is True, json.dumps(
            report["provenance"], indent=2, sort_keys=True
        )
        assert report["oracle"]["passed"] is True
        assert report["rig_contract"]["passed"] is True
        rig = report["rig_contract"]
        assert rig["tube_mesh_count"] == 17
        assert rig["tube_vertex_count"] == 55_337
        assert all(rig["checks"].values())


def test_single_bind_and_baked_weight_transport_are_hard_gates(v4_matrix) -> None:
    _operator, _calibration, _model, _values, reports = v4_matrix
    for report in reports.values():
        assert report["bind_contract"]["passed"] is True
        assert report["bind_contract"]["rigid"]["passed"] is True
        assert report["transport"]["passed"] is True
        assert report["transport"]["tube_application_count"] == 1
        assert report["transport"]["soft_application_count"] == 1
        assert report["transport"]["max_error_m"] <= ZERO_VERTEX_LIMIT_M_V4


def test_terminal_hand_and_foot_subtrees_preserve_local_topology(v4_matrix) -> None:
    _operator, _calibration, _model, _values, reports = v4_matrix
    for report in reports.values():
        terminal = report["terminal_subtrees"]
        assert terminal["passed"] is True, json.dumps(
            terminal, indent=2, sort_keys=True
        )
        assert set(terminal["roots"]) == {
            "Ankle_Rot_L",
            "Ankle_Rot_R",
            "Wrist_Rotate_L",
            "Wrist_Rotate_R1",
        }
        for root in terminal["roots"].values():
            assert root["internal_local_bind_max_error"] <= 1.0e-6
            assert root["shared_correction_max_error"] <= 1.0e-6
            assert root["posed_relative_max_error"] <= 1.0e-6
        assert terminal["arch_basis_identity_max_error"] <= 1.0e-10


def test_dynamic_matrix_uses_apply_v4_for_all_two_beta_three_pose_cells(
    v4_matrix,
) -> None:
    _operator, _calibration, _model, _values, reports = v4_matrix
    for report in reports.values():
        dynamic = report["dynamic_matrix"]
        assert dynamic["target_segment_length_invariance"]["passed"] is True
        assert (
            dynamic["target_segment_length_invariance"]["max_drift_m"]
            <= dynamic["target_segment_length_invariance"]["limit_m"]
        )
        assert tuple(dynamic["cells"]) == EXPECTED_POSE_LABELS_V4
        for cell in dynamic["cells"].values():
            assert len(cell["joints"]) == 12
            for joint in cell["joints"].values():
                assert joint["passed"] == (
                    joint["pivot_error_m"] <= PIVOT_LIMIT_M_V4
                    and joint["bone_attachment_pivot_error_m"]
                    <= joint["bone_attachment_limit_m"]
                    and joint["axis_error_deg"] <= AXIS_LIMIT_DEG_V4
                    and joint["rigid_cap_passed"]
                )
        zero = report["zero_pose"]
        assert zero["passed"] == (
            zero["matrix_max_error"] <= ZERO_MATRIX_LIMIT_V4
            and zero["vertex_max_m"] <= ZERO_VERTEX_LIMIT_M_V4
        )
        assert report["passed"] == all(report["sections"].values())


def test_identity_target_bind_exactly_reproduces_142_baked_motion(v4_matrix) -> None:
    operator, calibration, _model, values, _reports = v4_matrix
    for value in values.values():
        asset = materialize_subject(
            operator, betas=value.betas, gender="male"
        ).rigged_asset
        bind = np.asarray(asset.target_bind_global, dtype=np.float64)
        parents = np.asarray(asset.source_bone_parents, dtype=np.int64)
        names = list(asset.source_bone_names)
        channel_ids = _channel_basis_controller_indices(names)
        identity = np.repeat(np.eye(3, dtype=np.float64)[None], len(channel_ids), axis=0)
        controllers = np.asarray(calibration.controller_indices, dtype=np.int64)
        source_frames, _widths, _details = _measure_frames(
            np.asarray(asset.vertices_rest, dtype=np.float64),
            calibration.domains,
            calibration.joint_domain_bases,
            partition="fit",
        )
        inverse = np.linalg.inv(bind[controllers])
        pivot_local = (
            np.einsum(
                "bij,bj->bi", inverse[:, :3, :3], source_frames[:, :3, 3]
            )
            + inverse[:, :3, 3]
        )
        for pose in np.asarray(value.validation_pose_axis_angle)[1:]:
            expected = _source_baked_parent_local_pose(asset, pose)
            actual = _pose_with_target_local_bind(
                B_prefit=bind,
                B_final=bind,
                source_posed_global=expected,
                parents=parents,
                calibration=calibration,
                controller_local_pivots=pivot_local,
                channel_basis_controller_indices=channel_ids,
                channel_basis_change=identity,
            )
            np.testing.assert_allclose(actual, expected, atol=1.0e-10, rtol=0.0)


def test_strict_containment_is_per_bone_for_every_matrix_cell(v4_matrix) -> None:
    _operator, _calibration, _model, _values, reports = v4_matrix
    for report in reports.values():
        containment = report["strict_bone_containment"]
        assert tuple(containment["cells"]) == EXPECTED_POSE_LABELS_V4
        for cell in containment["cells"].values():
            assert cell["failed_mesh_count"] == 0, json.dumps(
                cell, indent=2, sort_keys=True
            )
            assert all(mesh["passed"] for mesh in cell["meshes"].values())


def test_transport_count_tampering_is_rejected_without_rebuilding(v4_matrix) -> None:
    operator, _calibration, _model, values, _reports = v4_matrix
    value = values["213328"]
    tampered = replace(
        value,
        build_report={**value.build_report, "tube_transport_application_count": 2},
    )
    asset = materialize_subject(
        operator, betas=tampered.betas, gender="male"
    ).rigged_asset
    report = _transport_contract(tampered, asset)
    assert report["passed"] is False
    assert report["checks"]["tube_application_count_one"] is False

"""Schema and serialization checks for the fixed single-shot QPIK telemetry."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger


def _step() -> JointIkStep:
    return JointIkStep(
        q_send=np.arange(8, dtype=float),
        qdot=np.linspace(-0.2, 0.2, 8),
        twist_base=np.zeros(6),
        sigma_min=0.1,
        manip=0.2,
        slack_norm=0.0,
        n_cbf_active=1,
        follow_err_rad=0.01,
        qp_backend="proxqp",
        qp_solver_status="solved",
        qp_solver_iterations=7,
        qp_solver_solve_ms=0.21,
        qp_solver_call_count=1,
        qp_solver_overrun=True,
        qpik_alpha=0.75,
        qpik_beta=0.90,
        qpik_authority=0.70,
        qpik_equality_residual_max=2.0e-7,
        qpik_hard_residual_max=3.0e-8,
        qpik_anchor_valid=True,
        qpik_protected_nominal_overflow=np.array([0.01, 0.0, 0.0, 0.0]),
        qpik_recovery_caps=np.arange(14, dtype=float) * 0.01,
        qpik_recovery_overflow_indices=(2, 9),
        qpik_working_slack=np.linspace(0.0, 0.007, 8),
        qpik_collision_slack=np.linspace(0.0, 0.003, 4),
        qpik_dexterity_slack=0.004,
        qpik_branch_slack=0.005,
        rail_macro_pref_v=0.03,
        rail_center_pref_v=-0.004,
        arm_risk_pref_norm=0.12,
        arm_risk_pref=np.linspace(0.0, 0.07, 8),
        risk_direction_cosine=0.95,
        path_velocity_xy=np.array([0.002, 0.0]),
        feedback_xy_raw=np.array([0.02, -0.03]),
        feedback_xy_filtered=np.array([0.01, -0.015]),
        rail_xy_contribution=np.array([0.0, 0.02]),
        arm_xy_contribution=np.array([0.01, -0.03]),
        rail_task_projection=0.04,
        rail_arm_cancel=-0.01,
        rail_decomposition_error=2.0e-8,
        arm_health=0.08,
        joint_margin_rad=0.3,
        wrist_margin_rad=0.2,
        wrist_singularity=0.17,
        hard_active_constraint_ids=("joint_lower:2", "self_collision:arm"),
        protected_target=np.array([1.0, np.nan]),
        protected_achieved=np.array([0.9, 0.2]),
        protected_residual=np.array([-0.1, 0.2]),
        scan_target=np.array([0.02, -0.01]),
        scan_achieved=np.array([0.015, -0.012]),
        scan_residual=np.array([-0.005, -0.002]),
        accepted_reference_lag_s=0.125,
        pre_solve_feedback_age_s=0.004,
        post_solve_feedback_age_s=0.006,
        fallback_level="none",
        fallback_reason="",
    )


def test_generic_qpik_columns_are_aligned_and_json_is_strict(tmp_path):
    path = tmp_path / "generic_qpik.csv"
    logger = _TickLogger(str(path))
    try:
        logger.write(
            0.0,
            "application_phase",
            0.0,
            _step(),
            np.zeros(8),
            np.zeros(6),
            np.zeros(6),
        )
    finally:
        logger.close()

    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    header, row = rows
    assert len(header) == len(row) == len(_TickLogger._HEADER)
    assert "scan_phase" not in " ".join(name for name in header if name.startswith("qpik_"))

    values = dict(zip(header, row, strict=True))
    assert values["phase"] == "application_phase"
    assert values["qpik_backend"] == "proxqp"
    assert values["qpik_solver_status"] == "solved"
    assert values["qpik_solver_iterations"] == "7"
    assert values["qpik_solver_call_count"] == "1"
    assert values["qpik_solver_overrun"] == "1"
    assert values["qpik_alpha"] == "0.75000000"
    assert values["qpik_beta"] == "0.90000000"
    assert values["qpik_authority"] == "0.70000000"
    assert json.loads(values["qpik_scan_residual_json"]) == [-0.005, -0.002]
    assert json.loads(values["qpik_working_slack_json"])[-1] == pytest.approx(0.007)
    assert json.loads(values["qpik_recovery_overflow_indices_json"]) == [2, 9]
    assert json.loads(values["qpik_feedback_xy_filtered_json"]) == [0.01, -0.015]
    assert json.loads(values["qpik_path_velocity_xy_json"]) == [0.002, 0.0]
    assert json.loads(values["qpik_rail_xy_contribution_json"]) == [0.0, 0.02]
    assert json.loads(values["qpik_arm_xy_contribution_json"]) == [0.01, -0.03]
    assert values["qpik_rail_task_projection"] == "0.04000000"
    assert values["qpik_rail_arm_cancel"] == "-0.01000000"
    assert values["qpik_rail_decomposition_error"] == "2.000000000e-08"
    assert values["qpik_q_cmd_q_meas_norm"] == f"{np.linalg.norm(np.arange(8)):.8f}"
    assert values["qpik_final_sent_qdot_json"] == _TickLogger._json_compact(_step().qdot)

    # Every structured cell must be valid strict JSON; NaN is encoded null.
    structured = [name for name in header if name.endswith("_json")]
    assert structured
    for name in structured:
        decoded = json.loads(values[name])
        assert decoded is not None
    assert json.loads(values["qpik_protected_target_json"]) == [1.0, None]


def test_json_compact_is_deterministic_for_numpy_and_nonfinite_values():
    value = {"z": np.float64(np.inf), "a": np.array([1.0, -np.inf])}
    assert _TickLogger._json_compact(value) == '{"a":[1.0,null],"z":null}'

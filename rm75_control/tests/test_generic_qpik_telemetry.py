"""Schema and serialization checks for generic two-level QPIK telemetry."""

from __future__ import annotations

import csv
import json

import numpy as np

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
        qp1_status="solved",
        qp2_status="solved",
        qp1_iterations=7,
        qp2_iterations=9,
        qp1_solve_ms=0.21,
        qp2_solve_ms=0.31,
        hard_active_constraint_ids=("joint_lower:2", "self_collision:arm"),
        protected_target=np.array([1.0, np.nan]),
        protected_achieved=np.array([0.9, 0.2]),
        protected_residual=np.array([-0.1, 0.2]),
        scalable_group_targets={"z": np.array([1.0]), "a": np.array([2.0, 3.0])},
        scalable_group_achieved={"z": np.array([0.8]), "a": np.array([1.9, 3.1])},
        scalable_group_alphas={"z": 0.8, "a": 1.0},
        scalable_group_residuals={"z": np.array([-0.2]), "a": np.array([-0.1, 0.1])},
        scalable_group_residual_norms={"z": 0.2, "a": 0.1},
        accepted_reference_lag_s=0.125,
        accepted_reference_error=np.array([0.01, -0.02, 0.0, 0.1, 0.0, -0.1]),
        planner_type="SrsRailPosturePlanner",
        planner_branch=3,
        planner_winding=-1,
        planner_state="valid",
        planner_age_s=0.02,
        planner_quality=0.75,
        planner_reason="",
        rail_guide_position_m=0.42,
        rail_guide_velocity_m_s=-0.03,
        rail_guide_acceleration_m_s2=0.11,
        psi_guide_position_rad=0.4,
        psi_guide_velocity_rad_s=0.2,
        psi_guide_acceleration_rad_s2=-0.1,
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
    assert values["qpik_qp1_status"] == "solved"
    assert values["qpik_qp2_iterations"] == "9"
    assert values["qpik_planner_branch"] == "3"
    assert values["qpik_planner_winding"] == "-1"
    assert json.loads(values["qpik_accepted_reference_error_json"]) == [
        0.01,
        -0.02,
        0.0,
        0.1,
        0.0,
        -0.1,
    ]
    assert values["qpik_q_cmd_q_meas_norm"] == f"{np.linalg.norm(np.arange(8)):.8f}"
    assert values["qpik_final_sent_qdot_json"] == _TickLogger._json_compact(_step().qdot)

    # Every structured cell must be valid strict JSON; NaN is encoded null.
    structured = [name for name in header if name.endswith("_json")]
    assert structured
    for name in structured:
        decoded = json.loads(values[name])
        assert decoded is not None
    assert json.loads(values["qpik_protected_target_json"]) == [1.0, None]
    # Mapping keys are sorted for deterministic replay/diff output.
    assert values["qpik_scalable_group_alphas_json"] == '{"a":1.0,"z":0.8}'


def test_json_compact_is_deterministic_for_numpy_and_nonfinite_values():
    value = {"z": np.float64(np.inf), "a": np.array([1.0, -np.inf])}
    assert _TickLogger._json_compact(value) == '{"a":[1.0,null],"z":null}'

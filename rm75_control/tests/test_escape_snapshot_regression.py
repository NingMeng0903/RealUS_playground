"""The generic controller must not revive retired rail-escape telemetry."""

from __future__ import annotations

import csv

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger


def test_episode_columns_are_removed_and_generic_fields_remain(tmp_path):
    path = tmp_path / "geometry.csv"
    logger = _TickLogger(str(path))
    try:
        qdot = np.zeros(8)
        qdot[0] = 0.012
        step = JointIkStep(
            q_send=np.zeros(8),
            qdot=qdot,
            twist_base=np.zeros(6),
            sigma_min=0.1,
            manip=0.0,
            slack_norm=0.0,
            n_cbf_active=0,
            follow_err_rad=0.0,
            qp_backend="scipy",
            qp1_status="solved",
            qp2_status="solved",
        )
        logger.write(0.0, "4d", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6), v_max=np.ones(8))
    finally:
        logger.close()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    retired = {
        "escape_active",
        "rail_escape_active",
        "rail_escape_sign",
        "rail_escape_stopped",
        "rail_escape_travel_m",
        "rail_escape_v_des_m_s",
        "rail_escape_qdot_cmd_m_s",
        "rail_ext_w",
        "rail_ext_w_raw",
        "rail_ext_w_capped",
        "rail_ext_w_effective",
    }
    assert retired.isdisjoint(row)
    assert row["qpik_backend"] == "scipy"
    assert row["qpik_qp1_status"] == "solved"
    assert row["qpik_qp2_status"] == "solved"
    assert len(row) == len(_TickLogger._HEADER)

"""Append-compatible geometry telemetry for the restored controller."""

from __future__ import annotations

import csv

import numpy as np

from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger


def test_episode_columns_are_retired_and_zero_by_default(tmp_path):
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
            rail_ext_weight=3.25,
            rail_ext_weight_raw=3.25,
            rail_ext_weight_capped=3.25,
            rail_ext_weight_effective=3.25,
            rail_escape_v_des_m_s=0.03,
        )
        logger.write(0.0, "4d", 0.0, step, np.zeros(8), np.zeros(6), np.zeros(6), v_max=np.ones(8))
    finally:
        logger.close()
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    row = rows[0]
    assert row["escape_active"] == "0"
    assert row["rail_escape_active"] == "0"
    assert row["rail_escape_sign"] == "0.000000"
    assert row["rail_escape_stopped"] == "0"
    assert row["rail_escape_travel_m"] == "0.00000000"
    assert row["rail_escape_v_des_m_s"] == "0.03000000"
    assert row["rail_escape_qdot_cmd_m_s"] == "0.01200000"
    assert row["rail_ext_w"] == "3.2500"
    assert row["rail_ext_w_raw"] == "3.2500"
    assert row["rail_ext_w_capped"] == "3.2500"
    assert row["rail_ext_w_effective"] == "3.2500"
    assert len(row) == len(_TickLogger._HEADER)

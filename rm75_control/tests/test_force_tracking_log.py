"""CSV schema checks for force-stability telemetry."""

from __future__ import annotations

import csv

import numpy as np

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.joint_admittance_8dof.loop import JointIkStep, _TickLogger


def test_force_log_has_new_fields_and_preserves_row_alignment(tmp_path) -> None:
    path = tmp_path / "force.csv"
    logger = _TickLogger(str(path))
    step = JointIkStep(
        q_send=np.zeros(8),
        qdot=np.zeros(8),
        twist_base=np.zeros(6),
        sigma_min=0.2,
        manip=0.1,
        slack_norm=0.0,
        n_cbf_active=0,
        follow_err_rad=0.0,
    )
    controller = AdmittanceController(0.005, AdmittanceConfig())
    controller.instability_index_raw = 0.456
    controller.instability_index = 0.123
    controller.force_pred_z = 3.9
    controller.force_dot_z = 12.5
    controller.cap_press_z = 0.007
    controller.cap_retract_z = 0.031
    controller._ke_estimator._update_gated = True
    controller._ke_estimator.last_dx_m = 0.0002
    controller._ke_estimator.last_df_n = 0.4
    controller._ke_estimator.update_count = 7

    class Outer:
        pass

    outer = Outer()
    outer.controller = controller
    logger.write(
        0.0,
        "scan",
        0.0,
        step,
        np.zeros(8),
        np.zeros(6),
        np.zeros(6),
        outer=outer,
        dt_actual_s=0.005,
        sensor_age_s=0.001,
        f_ext_raw=np.zeros(6),
        twist_achieved_base=np.zeros(6),
        v_tcp_z_actual=0.002,
    )
    logger.close()

    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1])
    values = dict(zip(rows[0], rows[1], strict=True))
    assert values["instability_idx"] == "0.1230"
    assert values["instability_idx_raw"] == "0.4560"
    assert values["instability_idx_active"] == "0.1230"
    assert values["force_pred_z"] == "3.9000"
    assert values["force_dot_z"] == "12.5000"
    assert values["cap_press_z"] == "0.007000"
    assert values["cap_retract_z"] == "0.031000"
    assert values["ke_update_gated"] == "1"
    assert values["ke_dx_m"] == "0.00020000"
    assert values["ke_df_n"] == "0.40000"
    assert values["ke_update_count"] == "7"

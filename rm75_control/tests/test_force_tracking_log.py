"""CSV schema checks for the single stable force controller."""

from __future__ import annotations

import csv

import numpy as np

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkStep,
    _TickLogger,
)


def test_force_log_has_energy_aware_reference_and_barrier_telemetry(tmp_path):
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

    class Outer:
        pass

    outer = Outer()
    outer.controller = controller
    controller.damping_trend_z = 18.5
    controller.cap_press_z = 0.012
    controller.cap_retract_z = 0.034
    controller.f_dot_z = -12.5
    controller.contact_present = True
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
    header = rows[0]
    assert "force_reference_scale_n" in header
    assert "force_reference_drive" in header
    assert "force_reference_gate_scale" in header
    assert "force_reference_accel_m_s2" in header
    assert "force_reference_reversal_reset" in header
    assert "damping_trend_z" in header
    assert "cap_press_z" in header
    assert "cap_retract_z" in header
    assert "f_dot_z" in header
    assert "contact_present" in header
    assert "vz_achieved_tool" in header
    values = dict(zip(header, rows[1]))
    assert values["damping_trend_z"] == "18.50"
    assert values["cap_press_z"] == "0.012000"
    assert values["cap_retract_z"] == "0.034000"
    assert values["f_dot_z"] == "-12.5000"
    assert values["contact_present"] == "1"
    assert values["vz_achieved_tool"] == "0.002000"

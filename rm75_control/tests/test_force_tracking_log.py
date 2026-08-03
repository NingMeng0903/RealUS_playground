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


def test_force_log_has_energy_aware_reference_and_actual_tcp_velocity(tmp_path):
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
    controller.force_reference_fast_clear = True
    controller.force_fast_z = 1.234
    controller.retract_guard_armed = True
    controller.retract_fast_hold = True
    controller.retract_fast_stop_count = 2
    controller.retract_fast_rearm_count = 3
    controller.force_task_latched = True
    controller.physical_contact_state = "suspect_loss"
    controller.physical_contact_acquire_event = True
    controller.physical_contact_loss_event = False
    controller.physical_contact_reacquire_event = True
    controller.physical_contact_low_timer_s = 0.012
    controller.physical_contact_high_timer_s = 0.034
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
    assert "force_reference_fast_clear" in header
    assert "force_fast_z" in header
    assert "retract_guard_armed" in header
    assert "retract_fast_hold" in header
    assert "retract_fast_stop_count" in header
    assert "retract_fast_rearm_count" in header
    assert "force_task_latched" in header
    assert "physical_contact_state" in header
    assert "physical_contact_acquire_event" in header
    assert "physical_contact_loss_event" in header
    assert "physical_contact_reacquire_event" in header
    assert "physical_contact_low_timer_s" in header
    assert "physical_contact_high_timer_s" in header
    assert "mass_z_eff" in header
    assert "damping_ke_z" in header
    assert "damping_dimeas_z" in header
    assert "vz_achieved_tool" in header

    values = dict(zip(header, rows[1], strict=True))
    assert values["force_reference_fast_clear"] == "1"
    assert values["force_fast_z"] == "1.234"
    assert values["retract_guard_armed"] == "1"
    assert values["retract_fast_hold"] == "1"
    assert values["retract_fast_stop_count"] == "2"
    assert values["retract_fast_rearm_count"] == "3"
    assert values["force_task_latched"] == "1"
    assert values["physical_contact_state"] == "suspect_loss"
    assert values["physical_contact_acquire_event"] == "1"
    assert values["physical_contact_loss_event"] == "0"
    assert values["physical_contact_reacquire_event"] == "1"
    assert values["physical_contact_low_timer_s"] == "0.012000"
    assert values["physical_contact_high_timer_s"] == "0.034000"

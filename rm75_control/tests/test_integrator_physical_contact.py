"""v_r / DOB integrate only while physical contact is present."""

from __future__ import annotations

import numpy as np

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.force_dob import ForceDobConfig
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


DT = 0.005


def test_loss_event_clears_vr_and_dob() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        desired_force_ramp_s=0.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        proactive_ff=ProactiveFfConfig(enabled=True, gain=0.3, retract_gain=0.3),
        force_dob=ForceDobConfig(enabled=True, ki=8.0, leak_s=0.4),
    )
    cfg.adaptive_ke.enabled = False
    cfg.force_barrier.enabled = False
    cfg.var_damping_enabled = False
    cfg.physical_contact.enabled = False
    ctrl = AdmittanceController(DT, cfg)
    # This isolated legacy-loop test deliberately opens the first-contact
    # slow latch; production CDYOB shadow keeps both loops disabled.
    ctrl._first_contact_slow_latched = False
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    f_ext = np.zeros(6)
    f_des = np.zeros(6)
    f_des[2] = 2.0
    pose = np.zeros(6)
    for _ in range(80):
        f_ext[2] = 0.8
        ctrl.compute_velocity_command(pose, pose, np.zeros(6), f_ext, f_des, in_contact=True)
    assert ctrl.v_r_z > 0.0
    assert ctrl.u_dob_z > 0.0
    f_ext[2] = 0.0
    ctrl.compute_velocity_command(pose, pose, np.zeros(6), f_ext, f_des, in_contact=False)
    assert ctrl.v_r_z == 0.0
    assert ctrl.u_dob_z == 0.0
    assert ctrl.force_task_latched

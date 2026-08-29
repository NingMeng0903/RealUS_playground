"""Kc stores F* as potential; outer xd integrator trims steady force."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


def test_kc_keeps_equilibrium_off_origin() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        admittance_stiffness_z=10.0,
        xd_gain_m_s_per_n=0.0,
        xd_rate_max_m_s=0.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.05,
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_barrier.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_ext = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(80):
        ctrl.compute_velocity_command(
            pose, pose, np.zeros(6), f_ext, f_des, in_contact=True
        )
    assert abs(ctrl.x_tilde_z) < 0.01
    assert abs(ctrl.v_force_cmd_z) < 0.01


def test_xd_integrator_is_rate_bounded() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=40.0,
        admittance_stiffness_z=10.0,
        xd_gain_m_s_per_n=1.0,
        xd_rate_max_m_s=0.002,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.05,
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_barrier.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des, in_contact=True
    )
    assert abs(ctrl.x_d_z) <= 0.002 * 0.005 + 1e-12

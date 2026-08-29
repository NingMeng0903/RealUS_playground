"""R1: LegacyForceLaw must emit the clamped command, not v_force_z."""

from __future__ import annotations

import numpy as np
import pytest

from peirastic.realman8dof.force.legacy import LegacyForceLaw
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig


def test_legacy_emits_u_sent_when_barrier_clamps() -> None:
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=8.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.08,
        max_velocity=np.array([0.2, 0.2, 0.08, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
    )
    cfg.physical_contact.enabled = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    cfg.force_dob.enabled = False
    cfg.force_barrier.enabled = True
    cfg.force_barrier.stiffness_cap_enabled = True
    cfg.force_barrier.budget_min_n = 0.2
    cfg.force_barrier.budget_frac = 0.0
    cfg.cdyob.mode = "off"
    ctrl = AdmittanceController(0.005, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    ctrl.contact_present = True
    ctrl.ke_est = 4000.0
    ctrl.ke_cap_n_m = 4000.0
    pose = np.zeros(6)
    law = LegacyForceLaw(ctrl)
    fout = law.update(
        dt_s=0.005,
        pose=pose,
        f_ext=np.zeros(6),
        f_des=np.array([0.0, 0.0, 4.0, 0.0, 0.0, 0.0]),
        path_twist=np.zeros(6),
        contact=True,
    )
    assert fout.v_force_z == pytest.approx(ctrl.v_force_cmd_z)
    assert fout.v_force_z == pytest.approx(ctrl.u_sent_z)
    assert abs(ctrl.u_sent_z) < abs(ctrl.v_force_z) - 1e-6

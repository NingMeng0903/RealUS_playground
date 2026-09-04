"""Smooth constant-force chase: DOB, no Is gate on under-force, short ΔD_hf."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.force_dob import (
    ForceDobConfig,
    ForceDisturbanceObserver,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)


DT = 0.005


def test_force_dob_removes_steady_bias():
    dob = ForceDisturbanceObserver(
        ForceDobConfig(enabled=True, ki=10.0, leak_s=0.5, u_max_n=2.0, freeze_is=0.9)
    )
    u = 0.0
    for _ in range(400):
        u = dob.update(0.4, dt_eff=DT, in_contact=True, instability_index=0.0)
    assert u > 0.3
    assert u <= 2.0 + 1e-9


def test_underforce_press_not_gated_by_is():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            enabled=True,
            gain=0.2,
            retract_gain=0.2,
            gate_press_on_is=False,
            press_is_gate=0.5,
            press_is_gate_start=0.2,
            leak_s=0.3,
            v_r_max_m_s=0.06,
        )
    )
    v = 0.0
    for _ in range(80):
        v = ff.update(
            0.8,
            in_contact=True,
            dt_eff=DT,
            instability_index=1.0,
            v_force_z=0.0,
            v_z_cap=0.1,
            desired_force_n=2.0,
        )
    assert v > 0.01


def test_yaml_smooth_chase_defaults_load():
    import yaml
    from pathlib import Path

    raw = yaml.safe_load(
        Path("configs/joint_admittance_8dof.yaml").read_text(encoding="utf-8")
    )
    cfg = AdmittanceConfig.from_dict(raw)
    assert cfg.force_dob.enabled is True
    assert cfg.proactive_ff.enabled is False
    assert cfg.proactive_ff.retract_only is False
    assert cfg.force_barrier.enabled is False
    assert cfg.track_axes[2] == pytest.approx(0.0)
    assert cfg.kp_pos[2] == pytest.approx(0.0)
    assert cfg.adaptive_ke.drive_damping is False
    assert cfg.proactive_ff.gate_press_on_is is False
    assert cfg.var_damping_d_u == pytest.approx(0.0)
    assert cfg.var_damping_m_u == pytest.approx(0.0)
    assert cfg.ke_schedule.enabled is True
    assert cfg.energy_tank.enabled is True
    assert cfg.cdyob.mode == "off"
    assert cfg.cdyob.applies() is False
    assert cfg.cdyob.omega_q_hz == pytest.approx(0.75)
    assert cfg.cdyob.t0_s == pytest.approx(0.028)
    assert cfg.cdyob.tp_s == pytest.approx(0.014)
    assert cfg.cdyob.v_corr_max_m_s == pytest.approx(0.015)
    assert cfg.cdyob.active_press_max_m_s == pytest.approx(0.010)
    assert cfg.cdyob.active_retract_max_m_s == pytest.approx(0.015)
    assert cfg.cdyob.active_force_ratio == pytest.approx(0.90)
    assert cfg.cdyob.active_settle_speed_m_s == pytest.approx(0.010)
    assert cfg.cdyob.active_settle_hold_s == pytest.approx(0.05)
    assert cfg.force_dob.ki == pytest.approx(8.0)
    assert cfg.force_dob.leak_s == pytest.approx(0.4)
    assert cfg.proactive_ff.v_r_max_m_s == pytest.approx(0.06)
    assert cfg.force_barrier.v_underforce_press_m_s == pytest.approx(0.010)
    assert cfg.cdyob.active_model_validated is False
    assert cfg.tdpa.enabled is True
    assert cfg.tdpa.apply is True
    assert cfg.tdpa.alpha_max == pytest.approx(20.0)
    assert cfg.safety_shield.u_retract_m_s == pytest.approx(0.080)
    assert cfg.force_corridor.enabled is False
    assert cfg.press_envelope.max_force_axis_m_s == pytest.approx(0.0)
    assert cfg.press_envelope.first_touch_m_s == pytest.approx(0.0)
    assert cfg.press_envelope.soft_approach_m_s == pytest.approx(0.0)
    assert cfg.admittance_stiffness_z == pytest.approx(0.0)
    assert cfg.var_damping_omega_c_hz == pytest.approx(2.5)
    assert cfg.max_vz_tool_m_s == pytest.approx(0.08)
    assert cfg.safety_shield.mode == "observe"
    assert cfg.safety_shield.k_ub_n_m == pytest.approx(8000.0)
    assert cfg.admittance_mass_z == pytest.approx(1.0)
    assert cfg.admittance_damping_z == pytest.approx(25.0)
    ctrl = AdmittanceController(DT, cfg)
    ctrl._first_contact_slow_latched = False
    ctrl._recontact_slow_latched = False
    ctrl._in_contact_latched = True
    ctrl._episode_seen = True
    ctrl.contact_present = True
    f_ext = np.zeros(6)
    f_des = np.zeros(6)
    f_des[2] = 2.0
    for _ in range(40):
        f_ext[2] = 0.9
        ctrl.compute_velocity_command(
            np.zeros(6), np.zeros(6), np.zeros(6), f_ext, f_des, in_contact=True
        )
    assert ctrl.v_force_z > 0.0
    assert abs(ctrl.v_r_z) <= 1e-6
    assert abs(ctrl.u_dob_z) > 1e-4


def test_hf_delta_d_releases_after_hold():
    cfg = AdmittanceConfig(
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        desired_force_ramp_s=0.0,
        var_damping_enabled=True,
        var_damping_d_u=80.0,
        var_damping_m_u=1.0,
        var_damping_hf_attack_s=0.02,
        var_damping_hf_hold_s=0.10,
        var_damping_hf_release_s=0.05,
        var_damping_hf_on=0.2,
        var_damping_hf_off=0.1,
        var_damping_hf_err_n=1.0,
    )
    cfg.adaptive_ke.enabled = False
    cfg.adaptive_ke.drive_damping = False
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.force_dob = ForceDobConfig(enabled=False)
    ctrl = AdmittanceController(DT, cfg)
    ctrl._contact_time_s = 1.0
    ctrl.instability_index = 0.8
    for _ in range(10):
        d = ctrl._update_delta_d_hf(DT, abs_eff_n=0.2)
    assert d > 10.0
    ctrl.instability_index = 0.0
    for _ in range(80):
        d = ctrl._update_delta_d_hf(DT, abs_eff_n=0.2)
    assert d < 5.0

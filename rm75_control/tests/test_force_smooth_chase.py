"""Smooth constant-force chase: DOB, no Is gate on under-force, short ΔD_hf."""

from __future__ import annotations

import numpy as np

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
    # e85c9ab's press path is back: bidirectional proactive feedforward plus
    # DOB.  1bfe98b had switched both off to fight bounce, at the cost of all
    # under-force chase speed; the force barrier is the brake now.
    assert cfg.force_dob.enabled
    assert cfg.proactive_ff.retract_only is False
    assert cfg.force_barrier.enabled
    assert cfg.adaptive_ke.drive_damping is False
    assert cfg.proactive_ff.gate_press_on_is is False
    assert cfg.var_damping_d_u >= 50.0
    ctrl = AdmittanceController(DT, cfg)
    f_ext = np.zeros(6)
    f_des = np.zeros(6)
    f_des[2] = 2.0
    for _ in range(40):
        f_ext[2] = 0.9
        ctrl.compute_velocity_command(
            np.zeros(6), np.zeros(6), np.zeros(6), f_ext, f_des
        )
    # Sustained under-force must now recruit both auxiliary terms, not just
    # the passive admittance: DOB removes the steady offset and the proactive
    # reference chases press.  Both were off in the 1bfe98b anti-bounce
    # baseline, which is what made the descent feel damped.
    assert ctrl.v_force_z > 0.0
    assert ctrl.u_dob_z > 0.0
    assert ctrl.v_r_z > 0.0


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

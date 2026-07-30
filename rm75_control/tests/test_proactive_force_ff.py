"""Ke-normalized proactive force feedforward (7dde980 feel restore)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)


DT = 0.005
POSE = np.zeros(6)


def test_press_uses_ke_floor_not_impact_ke():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain_mode="ke_normalized",
            tau_ff_s=0.20,
            ke_floor_ff=80.0,
            tau_track_s=0.02,
            v_r_max_m_s=0.20,
            leak_s=1e6,
            alpha_leak=0.0,
            reset_on_reversal=False,
        )
    )
    for _ in range(100):
        ff.update(
            1.0,
            in_contact=True,
            dt_eff=DT,
            ke_hat=1500.0,
            instability_index=0.0,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
    assert ff.last_force_scale_n == pytest.approx(80.0 * 0.20)
    assert ff.v_r > 0.02


def test_press_gate_fades_chase_when_unstable():
    kw = dict(
        gain_mode="ke_normalized",
        tau_ff_s=0.20,
        ke_floor_ff=80.0,
        tau_track_s=0.02,
        leak_s=1e6,
        alpha_leak=0.0,
        reset_on_reversal=False,
        press_is_gate_start=0.2,
        press_is_gate=0.6,
    )
    ff_lo = ProactiveForceIntegrator(ProactiveFfConfig(**kw))
    ff_hi = ProactiveForceIntegrator(ProactiveFfConfig(**kw))
    for _ in range(80):
        ff_lo.update(
            2.0,
            in_contact=True,
            dt_eff=DT,
            ke_hat=80.0,
            instability_index=0.0,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
        ff_hi.update(
            2.0,
            in_contact=True,
            dt_eff=DT,
            ke_hat=80.0,
            instability_index=0.6,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
    assert ff_hi.v_r < ff_lo.v_r * 0.5


def test_v_r_respects_saturation():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(
            gain_mode="ke_normalized",
            tau_ff_s=0.05,
            v_r_max_m_s=0.01,
            tau_track_s=0.02,
            leak_s=1e6,
            alpha_leak=0.0,
        )
    )
    for _ in range(200):
        ff.update(
            5.0,
            in_contact=True,
            dt_eff=DT,
            ke_hat=80.0,
            instability_index=0.0,
            v_force_z=0.0,
            v_z_cap=0.10,
        )
    assert abs(ff.v_r) <= 0.01 + 1e-9


def test_controller_builds_v_r_in_contact():
    cfg = AdmittanceConfig(
        contact_delta_n=0.5,
        deadband_n=0.0,
        deadband_width_n=0.0,
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
        proactive_ff=ProactiveFfConfig(
            enabled=True,
            gain_mode="ke_normalized",
            tau_ff_s=0.20,
            ke_floor_ff=80.0,
            leak_s=1e6,
            alpha_leak=0.0,
        ),
    )
    cfg.adaptive_ke.enabled = False
    ctrl = AdmittanceController(DT, cfg)
    for _ in range(100):
        force = np.zeros(6)
        force[2] = 0.5
        target = np.zeros(6)
        target[2] = 2.0
        ctrl.compute_velocity_command(
            POSE,
            POSE,
            np.zeros(6),
            force,
            target,
            f_ext_raw=force,
            dt_actual=DT,
            in_contact=True,
        )
    assert abs(ctrl.v_r_z) > 1e-4


def test_yaml_has_7dde980_ff_feel_knobs():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    hm = raw["hybrid_motion"]
    cfg = AdmittanceConfig.from_dict(raw)
    assert cfg.proactive_ff.gain_mode == "ke_normalized"
    assert cfg.proactive_ff.tau_track_s == pytest.approx(0.08)
    assert cfg.proactive_ff.leak_s == pytest.approx(0.3)
    assert cfg.proactive_ff.reset_on_reversal is True
    assert hm["proactive_press_is_gate"] > hm["proactive_press_is_gate_start"]

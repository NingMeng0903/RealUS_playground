"""Leaky ∫F_err proactive reference (bidirectional chase + bounce guards)."""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from pathlib import Path

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import (
    ProactiveFfConfig,
    ProactiveForceIntegrator,
)

DT = 0.005


def test_integrator_bidirectional_press_and_retract():
    ff = ProactiveForceIntegrator(
        ProactiveFfConfig(enabled=True, retract_only=False, gain=0.10, leak_s=10.0)
    )
    for _ in range(200):
        ff.update(1.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff.v_r > 0.01
    ff.reset()
    for _ in range(200):
        ff.update(-1.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff.v_r < -0.01


def test_press_side_fades_with_instability_index():
    ff_lo = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    ff_hi = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.5))
    for _ in range(300):
        ff_lo.update(2.0, in_contact=True, dt_eff=DT, instability_index=0.0, v_force_z=0.0, v_z_cap=0.10)
        ff_hi.update(2.0, in_contact=True, dt_eff=DT, instability_index=0.5, v_force_z=0.0, v_z_cap=0.10)
    assert ff_hi.v_r < ff_lo.v_r - 1e-6


def test_retract_unaffected_by_press_is_gate():
    ff = ProactiveForceIntegrator(ProactiveFfConfig(gain=0.10, leak_s=10.0, press_is_gate=0.1))
    for _ in range(300):
        ff.update(-2.0, in_contact=True, dt_eff=DT, instability_index=1.0, v_force_z=0.0, v_z_cap=0.10)
    assert ff.v_r < -0.02


def test_rising_edge_clears_press_v_r():
    ctrl = AdmittanceController(
        DT,
        AdmittanceConfig(
            proactive_ff=ProactiveFfConfig(enabled=True),
            adaptive_ke=AdmittanceConfig().adaptive_ke,
        ),
    )
    ctrl.cfg.adaptive_ke.enabled = False
    ctrl._proactive_ff.v_r = 0.04
    ctrl._update_proactive_v_r(0.0, True, DT, rising_edge=True)
    assert ctrl.v_r_z == pytest.approx(0.0)


def _controller(**over) -> AdmittanceController:
    kw = dict(
        contact_threshold_n=0.8,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.10,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        var_damping_enabled=False,
        proactive_ff=ProactiveFfConfig(
            enabled=True,
            retract_only=False,
            gain=0.10,
            leak_s=0.3,
            v_r_max_m_s=0.06,
        ),
    )
    kw.update(over)
    cfg = AdmittanceConfig(**kw)
    cfg.adaptive_ke.enabled = False
    return AdmittanceController(DT, cfg)


def test_proactive_boosts_velocity_under_sustained_error():
    ctrl = _controller()
    ctrl._in_contact_latched = True
    for _ in range(400):
        ctrl._admittance_z(
            2.0,
            True,
            dt_eff=DT,
            rising_edge=False,
        )
    assert ctrl.v_r_z > 0.015
    assert ctrl.v_force_z > 0.08


def test_yaml_proactive_bidirectional_and_headroom():
    raw = yaml.safe_load(Path("configs/joint_admittance_8dof.yaml").read_text())
    hm = raw["hybrid_motion"]
    assert hm["proactive_feedforward"] is True
    assert hm["proactive_retract_only"] is False
    assert hm["v_r_max_m_s"] < hm["max_vz_tool_m_s"]
    assert "li2022" not in hm

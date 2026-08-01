"""Bounce guards on hard / high-elasticity surfaces."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.admittance_common.adaptive_ke import (
    AdaptiveKeConfig,
    EnvironmentStiffnessEstimator,
)
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.proactive_force_ff import ProactiveFfConfig

DT = 0.005
POSE = np.zeros(6)


def test_var_damping_lambda_matches_paper_bandwidth():
    cfg = AdmittanceConfig()
    tau = -DT / math.log(cfg.var_damping_lambda)
    assert 0.06 <= tau <= 0.16, f"var_damping_lambda tau={tau:.3f}s, expected ~0.1s"


def test_var_damping_m_max_is_finite_safety_cap():
    cfg = AdmittanceConfig()
    assert cfg.admittance_mass_z < cfg.var_damping_m_max < math.inf


def test_eq5_instability_index_is_unbounded_leaky_accumulator():
    cfg = AdmittanceConfig(var_damping_enabled=True)
    cfg.proactive_ff = ProactiveFfConfig(enabled=False)
    cfg.adaptive_ke.enabled = False
    dt = 0.001
    ctrl = AdmittanceController(dt, cfg)
    for k in range(6000):
        f = 10.0 * math.sin(2.0 * math.pi * 8.0 * k * dt)
        ctrl._update_instability_index(f)
    assert ctrl.instability_index > 1.0


def _estimator(**over) -> EnvironmentStiffnessEstimator:
    kw = dict(
        enabled=True,
        ke_initial=80.0,
        ke_impact_initial=1500.0,
        ke_idle_decay_s=2.0,
        settle_ticks=0,
        gate_lateral_velocity=False,
        gate_df_spike=False,
        bd_slew_max=1e6,
        ke_slew_max=1e6,
    )
    kw.update(over)
    return EnvironmentStiffnessEstimator(AdaptiveKeConfig(**kw), dt=DT, mass_z=1.0)


def _run_contact_ticks(est: EnvironmentStiffnessEstimator, n: int) -> float:
    ke = est.ke_est
    for _ in range(n):
        ke, _bd = est.update(
            3.0,
            POSE,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.0,
            f_err_z=0.0,
            f_des_z=3.0,
            instability_index=0.0,
        )
    return ke


def test_idle_decay_active_when_contact_quiet():
    est = _estimator()
    ke = _run_contact_ticks(est, 400)
    assert ke < 0.6 * 1500.0


def test_instability_does_not_freeze_idle_ke_decay() -> None:
    est = _estimator()
    ke = est.ke_est
    for _ in range(400):
        ke, _ = est.update(
            3.0,
            POSE,
            in_contact=True,
            mass_z=1.0,
            v_force_z=0.0,
            f_err_z=0.0,
            f_des_z=3.0,
            instability_index=0.16,
        )
    assert ke < 0.6 * 1500.0


def _controller(**over) -> AdmittanceController:
    kw = dict(
        contact_threshold_n=0.8,
        contact_use_fz_only=True,
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        deadband_n=0.0,
        deadband_width_n=0.0,
        max_vz_tool_m_s=0.05,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
        proactive_ff=ProactiveFfConfig(enabled=False),
    )
    kw.update(over)
    cfg = AdmittanceConfig(**kw)
    cfg.adaptive_ke.enabled = False
    return AdmittanceController(DT, cfg)


def test_v_z_cap_unified_and_independent_of_instability():
    ctrl = _controller(
        max_vz_tool_m_s=0.10,
        max_velocity=np.array([0.2, 0.2, 0.10, 0.5, 0.5, 0.5]),
        var_damping_enabled=True,
    )
    ctrl.instability_index = 0.0
    assert ctrl._v_z_cap() == pytest.approx(0.10)
    ctrl.instability_index = 2.5
    assert ctrl._v_z_cap() == pytest.approx(0.10)


def test_idle_decay_respects_ke_soft_floor():
    est = _estimator(ke_initial=80.0, ke_soft_floor=300.0, ke_idle_decay_s=2.0)
    ke = _run_contact_ticks(est, 400)
    assert ke >= 300.0 - 1e-3
    assert ke < 1500.0


def test_idle_decay_soft_floor_disabled_when_zero():
    est = _estimator(ke_initial=80.0, ke_soft_floor=0.0, ke_idle_decay_s=2.0)
    ke = _run_contact_ticks(est, 400)
    assert ke < 0.6 * 1500.0


def test_var_damping_d_target_respects_bd_max_when_is_unbounded():
    ctrl = _controller(
        var_damping_enabled=True,
        var_damping_d_u=2.0,
        admittance_damping_z=25.0,
    )
    ctrl.cfg.adaptive_ke.bd_max = 50.0
    ctrl._in_contact_latched = True
    ctrl.adaptive_bd = 25.0
    ctrl._m_z_now = 1.0
    ctrl.ke_est = 80.0
    ctrl.instability_index = 100.0
    for _ in range(50):
        ctrl._admittance_z(
            0.0,
            True,
            dt_eff=DT,
            rising_edge=False,
        )
    assert ctrl.damping_z_eff <= 50.0 + 1e-6

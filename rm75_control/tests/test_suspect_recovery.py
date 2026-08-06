"""Suspect-loss recovery + zero-centered impact + Teff press budget."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.contact_state import (
    PhysicalContactConfig,
)
from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.press_energy_tank import (
    PortPassivityConfig,
    PressEnergyTankConfig,
)


def _ctrl(**kwargs) -> AdmittanceController:
    base = dict(
        admittance_mass_z=1.0,
        admittance_damping_z=25.0,
        max_vz_tool_m_s=0.08,
        desired_force_ramp_s=0.0,
        var_damping_enabled=False,
        delay_damping_enabled=True,
        delay_damping_mode="impact_only",
        delay_press_budget_enabled=True,
        t_eff_s=0.070,
        suspect_recovery_enabled=True,
        suspect_recovery_vz_cap_m_s=0.012,
        suspect_recovery_vr_press_max_m_s=0.003,
        low_force_press_cap_m_s=0.010,
        low_force_press_enter_n=1.80,
        impact_damping_hold_s=0.12,
        impact_damping_release_s=0.10,
        impact_damping_zeta=0.9,
        impact_damping_d_min=60.0,
        impact_damping_d_max=100.0,
        impact_ke_floor=1300.0,
        v_force_aw_enabled=False,
        press_energy_tank=PressEnergyTankConfig(enabled=False),
        port_passivity=PortPassivityConfig(enabled=False),
        d_extra_attack_s=0.005,
        physical_contact=PhysicalContactConfig(
            enabled=True,
            enter_n=0.5,
            hard_enter_n=0.5,
            exit_n=0.35,
            enter_confirm_s=0.0,
            exit_confirm_s=0.100,
        ),
    )
    base.update(kwargs)
    cfg = AdmittanceConfig(**base)
    cfg.force_barrier.enabled = False
    cfg.force_barrier.t_pred_s = 0.070
    cfg.force_dob.enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.proactive_ff.enabled = False
    return AdmittanceController(0.005, cfg)


def _step(ctrl, fz, fdes=2.0, vz_tcp=0.0):
    pose = np.zeros(6)
    f_ext = np.array([0.0, 0.0, fz, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, fdes, 0.0, 0.0, 0.0])
    return ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        f_ext,
        f_des,
        v_tcp_z_actual=vz_tcp,
    )


def test_critical_impact_damping_on_acquire():
    ctrl = _ctrl()
    for _ in range(3):
        _step(ctrl, 1.0)
    assert ctrl.physical_contact_acquire_event or ctrl.contact_present
    assert ctrl.damping_z_eff >= 55.0
    assert ctrl.damping_impact_z >= 30.0


def test_steady_contact_returns_to_d0():
    ctrl = _ctrl()
    for _ in range(5):
        _step(ctrl, 2.0)
    for _ in range(80):
        _step(ctrl, 2.0)
    assert ctrl.damping_z_eff == pytest.approx(25.0, abs=2.0)
    assert ctrl.damping_impact_z == pytest.approx(0.0, abs=1.0)


def test_suspect_loss_caps_repress():
    ctrl = _ctrl()
    for _ in range(10):
        _step(ctrl, 2.0)
    for _ in range(5):
        v = _step(ctrl, 0.1)
    assert ctrl.suspect_recovery_active
    assert ctrl.cap_press_z <= 0.012 + 1e-9
    assert v[2] <= 0.012 + 1e-6


def test_teff_budget_limits_overforce_press():
    ctrl = _ctrl()
    for _ in range(10):
        _step(ctrl, 2.0)
    for _ in range(80):
        _step(ctrl, 2.0)
    ctrl.ke_est = 1300.0
    ctrl.ke_barrier = 1300.0
    ctrl._press_budget_filt = None
    # budget≈0.5 N → raw ≈ (2.5-2.36)/(1300*0.07) floored at 3 mm/s.
    v = _step(ctrl, 2.36)
    assert ctrl.cap_press_z < 0.015
    assert v[2] < 0.015


def test_zero_center_extra_does_not_amplify_vr():
    """Di must not multiply v_r (old bug: ΔD·vr injects press force)."""
    ctrl = _ctrl(
        delay_press_budget_enabled=False,
        low_force_press_cap_m_s=0.0,
        suspect_recovery_enabled=False,
    )
    for _ in range(3):
        _step(ctrl, 1.2, vz_tcp=0.02)
    ctrl._impact_timer_s = 0.20
    ctrl._impact_rearm_ready = False  # don't re-trigger mid-step
    ctrl._d_extra_smooth = 0.0
    ctrl.v_force_z = 0.0
    ctrl.v_r_z = 0.02
    ctrl._proactive_ff.v_r = 0.02
    ctrl.ke_est = 1300.0
    ctrl.ke_barrier = 1300.0
    v = _step(ctrl, 2.0, vz_tcp=0.0)
    d0 = 25.0
    di = float(ctrl.damping_impact_z)
    assert di > 20.0
    m, dt = 1.0, 0.005
    vr = 0.02
    # Ideal zero-center with drive≈0, v_old=0:
    v_zero = (d0 * vr) / (m / dt + d0 + di)
    v_old = ((d0 + di) * vr) / (m / dt + d0 + di)
    assert v_zero < v_old - 1e-4
    assert v[2] < v_old - 1e-4
    assert abs(v[2] - v_zero) < 0.008


def test_compress_rise_arms_impact_without_suspect():
    ctrl = _ctrl(
        impact_fdot_arm_n_s=12.0,
        impact_fpred_over_n=0.15,
        impact_arm_confirm_s=0.010,
    )
    for _ in range(5):
        _step(ctrl, 2.0, vz_tcp=0.01)
    # Expire acquire impact and re-arm latch.
    for _ in range(80):
        _step(ctrl, 1.2, vz_tcp=0.0)
    assert ctrl.damping_impact_z == pytest.approx(0.0, abs=1.0)
    ctrl._impact_rearm_ready = True
    # Rising force with TCP still retracting (delayed) must still arm Di.
    fz = 1.2
    armed = False
    for _ in range(16):
        fz += 0.25  # ~50 N/s at dt=5 ms
        _step(ctrl, fz, vz_tcp=-0.02)
        if ctrl._impact_timer_s > 0.0 and ctrl.damping_impact_z > 10.0:
            armed = True
            break
    assert armed
    assert fz > 0.7  # never needed a deep trough


def test_slow_press_not_blocked_without_overshoot_episode():
    """Interlock must not lock normal under-force press (65.95 s bug)."""
    ctrl = _ctrl(
        delay_press_budget_enabled=False,
        low_force_press_cap_m_s=0.0,
        suspect_recovery_enabled=False,
        free_seek_vz_m_s=0.0,
        contact_press_cap_m_s=0.08,
        reverse_interlock_enter_m_s=0.004,
        reverse_interlock_exit_m_s=0.0015,
        reverse_interlock_enter_confirm_s=0.010,
        reverse_interlock_exit_confirm_s=0.010,
        v_gate_window_s=0.030,
        press_energy_tank=PressEnergyTankConfig(enabled=False),
        port_passivity=PortPassivityConfig(enabled=False),
    )
    for _ in range(10):
        _step(ctrl, 2.0, vz_tcp=0.0)
    ctrl.free_seek_active = False
    ctrl._overshoot_episode_s = 0.0
    ctrl.v_force_z = 0.0
    v = None
    for _ in range(8):
        v = _step(ctrl, 0.8, vz_tcp=-0.04)
    assert not ctrl.reverse_interlock_active
    assert ctrl.reverse_interlock_gate == pytest.approx(1.0)
    assert v is not None and v[2] > 1e-4


def test_overshoot_episode_gates_press_drive_not_hard_zero_when_idle():
    ctrl = _ctrl(
        delay_press_budget_enabled=False,
        low_force_press_cap_m_s=0.0,
        suspect_recovery_enabled=False,
        free_seek_vz_m_s=0.0,
        contact_press_cap_m_s=0.08,
        reverse_interlock_enter_m_s=0.004,
        reverse_interlock_enter_confirm_s=0.010,
        v_gate_window_s=0.030,
        press_energy_tank=PressEnergyTankConfig(enabled=False),
        port_passivity=PortPassivityConfig(enabled=False),
    )
    for _ in range(10):
        _step(ctrl, 2.0, vz_tcp=0.0)
    ctrl.free_seek_active = False
    # Simulate retract-overshoot episode with TCP still retracting.
    ctrl._overshoot_episode_s = 0.20
    ctrl._retract_brake_timer_s = 0.10
    ctrl.v_force_z = 0.0
    for _ in range(8):
        _step(ctrl, 0.8, vz_tcp=-0.05)
    assert ctrl.reverse_interlock_active or ctrl.reverse_interlock_gate < 0.5


def test_retract_brake_raises_damping_on_hold_edge():
    ctrl = _ctrl(
        retract_brake_damping_ns_m=70.0,
        retract_brake_hold_s=0.060,
        retract_brake_release_s=0.040,
    )
    for _ in range(5):
        _step(ctrl, 2.0)
    ctrl.v_force_z = -0.05
    # Simulate predictive hold rising edge.
    ctrl._prev_retract_fast_hold = False
    ctrl.retract_fast_hold = True
    ctrl._retract_brake_timer_s = 0.10
    ctrl._impact_timer_s = 0.0
    ctrl._d_extra_smooth = 0.0
    v = _step(ctrl, 1.9, vz_tcp=-0.03)
    assert ctrl.damping_retract_brake_z > 30.0
    assert ctrl.damping_z_eff >= 55.0
    # Brake dissipates retract proxy speed (not a hard zero reset).
    assert abs(v[2]) < 0.05


def test_force_pred_one_sided_ignores_negative_fdot():
    ctrl = _ctrl()
    for _ in range(5):
        _step(ctrl, 2.0)
    for _ in range(80):
        _step(ctrl, 2.0)
    ctrl.ke_est = 900.0
    ctrl.ke_barrier = 900.0
    # Build negative ḟ then check pred does not undershoot f_ext.
    _step(ctrl, 3.0, vz_tcp=0.0)
    _step(ctrl, 2.5, vz_tcp=0.0)
    _step(ctrl, 2.0, vz_tcp=0.0)
    assert ctrl.force_dot_z < 0.0
    # pred == f_ext when ḟ < 0 (horizon term zeroed via max(fdot,0)).
    assert ctrl.force_pred_z == pytest.approx(2.0, abs=0.05)


def test_aw_disabled_does_not_weaken_retract():
    ctrl = _ctrl(v_force_aw_enabled=False)
    for _ in range(5):
        _step(ctrl, 2.0)
    ctrl.v_force_z = -0.08
    v = _step(ctrl, 5.0, vz_tcp=0.02)  # over-force, TCP still pressing
    assert v[2] < 0.0


def test_impact_danger_renews_while_force_still_rising():
    """Di must not decay to 0 while Fz is still climbing (hard-surface fix)."""
    ctrl = _ctrl(
        press_energy_tank=PressEnergyTankConfig(enabled=False),
        impact_safe_confirm_s=0.025,
        impact_pred_span_n=1.5,
    )
    for _ in range(5):
        _step(ctrl, 2.0, vz_tcp=0.0)
    for _ in range(80):
        _step(ctrl, 2.0, vz_tcp=0.0)
    ctrl._impact_rearm_ready = True
    fz = 2.2
    di_at_peak = []
    for _ in range(40):
        fz += 0.15  # sustained rise ~30 N/s
        _step(ctrl, fz, vz_tcp=0.01)
        if fz > 4.0:
            di_at_peak.append(ctrl.damping_impact_z)
    assert di_at_peak
    assert min(di_at_peak) > 10.0
    assert ctrl.impact_danger or ctrl.damping_impact_z > 5.0


def test_press_tank_scales_only_in_energy_limit():
    """Tank γ must not starve under-force tracking (run_233408 regression)."""
    ctrl = _ctrl(
        press_energy_tank=PressEnergyTankConfig(
            enabled=True,
            e_initial_j=1e-6,
            e_max_j=1e-6,
            e_min_j=0.0,
            seed_on_acquire=False,
        ),
        port_passivity=PortPassivityConfig(enabled=False),
        delay_press_budget_enabled=False,
        low_force_press_cap_m_s=0.0,
        suspect_recovery_enabled=False,
        reverse_interlock_enter_m_s=0.0,
        free_seek_vz_m_s=0.0,
        contact_press_cap_m_s=0.08,
        impact_damping_hold_s=0.0,
    )
    for _ in range(5):
        _step(ctrl, 2.0, vz_tcp=0.0)
    # Clear acquire/impact leftovers so under-force is not energy-limited.
    ctrl._impact_timer_s = 0.0
    ctrl._overshoot_episode_s = 0.0
    ctrl._retract_brake_timer_s = 0.0
    ctrl.free_seek_active = False
    ctrl.v_force_z = 0.02
    # Under Fd: energy-limit off → γ=1 (tracking must stay alive).
    _step(ctrl, 0.5, vz_tcp=0.01)
    assert ctrl.tank_gamma > 0.99
    assert not ctrl._energy_limit_active
    # Live overshoot episode with press drive → empty tank scales.
    ctrl._overshoot_episode_s = 0.2
    ctrl._press_tank.energy_j = 1e-6
    ctrl.v_force_z = 0.02
    v = _step(ctrl, 0.5, vz_tcp=0.01)
    assert ctrl._tank_pc_active
    assert ctrl.tank_gamma < 0.5
    assert v[2] < 0.02


def test_tank_dx_residual_accumulates_slow_motion():
    from rm75_control.control.admittance_common.press_energy_tank import (
        PressEnergyTank,
    )

    tank = PressEnergyTank(
        PressEnergyTankConfig(
            enabled=True,
            e_max_j=0.004,
            e_initial_j=0.001,
            credit_gain=0.2,
            dx_deadband_m=2.0e-5,
        )
    )
    # 1 mm/s × 5 ms = 5 µm < 20 µm — must accumulate, not discard forever.
    for _ in range(5):
        tank.observe_and_scale(
            f_ext_z=2.0,
            dx_m=-5.0e-6,  # retract under load → credit
            u_press=0.0,
            v_press_est_m_s=0.0,
            dt_s=0.005,
        )
    assert tank.energy_j > 0.001  # got credit after residual flushed

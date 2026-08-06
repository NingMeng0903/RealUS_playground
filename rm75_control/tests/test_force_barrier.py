"""Unit tests for stiffness-scheduled two-sided force barrier + 1-D bounce sim."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)
from tests.sim_contact_1d import SimConfig, run_sim


def _barrier(**kwargs) -> ForceSpaceVelocityDamper:
    # Explicit press_only=False when testing legacy two-sided retract.
    kwargs.setdefault("press_only", True)
    kwargs.setdefault("cap_lpf_tau_s", 0.0)  # deterministic unit asserts
    cfg = ForceBarrierConfig(enabled=True, **kwargs)
    return ForceSpaceVelocityDamper(cfg)


def test_stiff_surface_press_cap_tight():
    b = _barrier(
        t_dead_s=0.04,
        budget_min_n=1.5,
        f_keep_n=0.3,
        v_floor_press_m_s=0.015,
        ke_max=20000.0,
    )
    b.ke_barrier = 9000.0
    b.f_dot_z = 0.0
    cap_p, _ = b.caps(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    # g = 1/(9000*0.04) → raw ≈ 4.2 mm/s, floored at 15 mm/s
    assert cap_p == pytest.approx(0.015)


def test_soft_surface_press_cap_open():
    b = _barrier(t_dead_s=0.04, budget_min_n=1.5)
    b.ke_barrier = 400.0
    b.f_dot_z = 0.0
    cap_p, _ = b.caps(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_p >= 0.07


def test_press_only_keeps_retract_open():
    b = _barrier(press_only=True, t_dead_s=0.04, f_keep_n=0.3)
    b.ke_barrier = 9000.0
    b.f_dot_z = 0.0
    _, cap_r = b.caps(
        f_z=0.3,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_r == pytest.approx(0.08)


def test_retract_collapses_at_f_keep():
    b = _barrier(
        press_only=False,
        t_dead_s=0.04,
        f_keep_n=0.3,
        yield_overforce_n=10.0,
    )
    b.ke_barrier = 9000.0
    b.f_dot_z = 0.0
    _, cap_r = b.caps(
        f_z=0.3,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_r <= 1e-6


def test_yield_overforce_opens_retract():
    b = _barrier(
        press_only=False,
        t_dead_s=0.04,
        yield_overforce_n=1.5,
        yield_fdot_max_n_s=60.0,
        ke_max=20000.0,
    )
    b.ke_barrier = 9000.0
    b.f_dot_z = 10.0  # mild — hand push
    _, cap_r = b.caps(
        f_z=4.0,  # 2 N over f_des
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_r == pytest.approx(0.08)


def test_bounce_overforce_keeps_retract_barrier():
    b = _barrier(
        press_only=False,
        t_dead_s=0.04,
        yield_overforce_n=1.5,
        yield_fdot_max_n_s=60.0,
        f_keep_n=0.3,
        f_panic_n=12.0,
        ke_max=20000.0,
    )
    b.ke_barrier = 9000.0
    b.f_dot_z = 100.0  # fast — bounce transient (f_z itself below panic)
    _, cap_r = b.caps(
        f_z=6.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    # Not fully open; barrier still schedules retract from f_pred.
    assert cap_r < 0.08


def test_retract_fast_hold_bypasses():
    b = _barrier(press_only=False, t_dead_s=0.04, f_keep_n=0.3)
    b.ke_barrier = 9000.0
    b.f_dot_z = 0.0
    _, cap_r = b.caps(
        f_z=0.5,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        retract_fast_hold=True,
    )
    assert cap_r == pytest.approx(0.08)


def test_panic_bypasses_retract():
    b = _barrier(press_only=False, t_dead_s=0.04, f_panic_n=12.0)
    b.ke_barrier = 9000.0
    b.f_dot_z = 0.0
    _, cap_r = b.caps(
        f_z=13.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_r == pytest.approx(0.08)


def test_free_space_keeps_seek():
    b = _barrier(ke_seek_default=300.0)
    cap_p, _ = b.caps(
        f_z=0.0,
        f_des_z=2.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
    )
    assert cap_p == pytest.approx(0.08)


def test_ke_barrier_converges_on_impact_ramp():
    b = _barrier(
        ke_attack_s=0.030,
        ke_seek_default=300.0,
        ke_max=20000.0,
        ke_v_press_min_m_s=0.005,
        ke_f_err_gate_n=50.0,  # don't freeze during ramp
        ke_slew_up_max=0.0,
    )
    dt = 0.005
    ke_true = 9000.0
    v_press = 0.02
    f = 0.5
    for _ in range(40):
        f += ke_true * v_press * dt
        b.update_fdot(f, dt)
        b.update_ke(
            f_z=f,
            v_tcp_z=v_press,
            in_contact=True,
            dt_eff=dt,
            f_des_z=f,  # near setpoint so learning allowed
        )
    assert b.ke_barrier > 3000.0
    assert b.ke_barrier <= 20000.0


def test_hand_push_freezes_and_relaxes_ke():
    b = _barrier(
        ke_seek_default=300.0,
        ke_f_err_gate_n=1.5,
        ke_release_s=0.2,
    )
    b.ke_barrier = 5000.0
    dt = 0.005
    before = b.ke_barrier
    for _ in range(40):
        b.f_dot_z = 200.0  # would inflate Ke if learned
        b.update_ke(
            f_z=8.0,
            v_tcp_z=0.02,
            in_contact=True,
            dt_eff=dt,
            f_des_z=2.0,  # over-force 6 N ≫ gate
        )
    # Must not climb; should relax toward seek.
    assert b.ke_barrier <= before + 1.0
    assert b.ke_barrier < 4000.0


def test_ke_barrier_holds_across_short_loss():
    b = _barrier(ke_free_hold_s=2.0, ke_seek_default=300.0, ke_max=20000.0)
    b.ke_barrier = 5000.0
    dt = 0.005
    for _ in range(int(0.2 / dt)):
        b.update_ke(
            f_z=0.0,
            v_tcp_z=0.0,
            in_contact=False,
            dt_eff=dt,
            f_des_z=2.0,
        )
    assert b.ke_barrier == pytest.approx(5000.0, rel=0.01)


def test_caps_monotone_in_force():
    b = _barrier(t_dead_s=0.04, yield_overforce_n=20.0)
    b.ke_barrier = 5000.0
    b.f_dot_z = 0.0
    prev_p = 1.0
    for fz in (0.5, 1.5, 2.5, 3.5, 4.5):
        cap_p, _ = b.caps(
            f_z=fz,
            f_des_z=2.0,
            in_contact=True,
            v_z_cap=0.08,
            seek_vz_m_s=0.08,
        )
        assert cap_p <= prev_p + 1e-9
        prev_p = cap_p


def test_sim_baseline_bounces_on_stiff():
    r = run_sim(SimConfig(ke=9000.0, use_barrier=False, t_end_s=3.0))
    assert r.peak_f > 6.0
    assert r.frac_low > 0.05
    assert r.n_loss >= 1


def test_sim_barrier_kills_limit_cycle():
    base = run_sim(SimConfig(ke=9000.0, use_barrier=False, t_end_s=4.0))
    fixed = run_sim(
        SimConfig(ke=9000.0, use_barrier=True, press_only=False, t_end_s=4.0)
    )
    assert fixed.peak_f < base.peak_f
    assert fixed.peak_f <= 8.0  # impact-only D; barrier + seed cut 33→~6
    # Steady window should not stay in free-flight slam cycle.
    assert fixed.peak_f < 0.35 * base.peak_f


def test_sim_soft_surface_still_tracks():
    r = run_sim(SimConfig(ke=400.0, use_barrier=True, t_end_s=3.0))
    mask = r.t >= 1.5
    assert abs(float(np.median(r.f[mask])) - 2.0) < 0.8
    assert r.peak_f < 6.0


def test_steady_contact_damping_is_d0():
    """Regression for 160926: continuous D_delay made D≈214 and feel sticky."""
    from rm75_control.control.admittance_common.controller import (
        AdmittanceConfig,
        AdmittanceController,
    )
    from rm75_control.control.admittance_common.contact_state import (
        PhysicalContactConfig,
    )

    cfg = AdmittanceConfig(
        admittance_damping_z=25.0,
        delay_damping_enabled=True,
        delay_damping_mode="impact_only",
        var_damping_enabled=False,
        desired_force_ramp_s=0.0,
        physical_contact=PhysicalContactConfig(
            enabled=True,
            enter_n=0.5,
            hard_enter_n=0.5,
            exit_n=0.2,
            enter_confirm_s=0.0,
            exit_confirm_s=0.0,
        ),
    )
    cfg.force_barrier.enabled = True
    cfg.adaptive_ke.enabled = False
    cfg.proactive_ff.enabled = False
    ctrl = AdmittanceController(0.005, cfg)
    # Seed an inflated ke_barrier like 160926
    ctrl._force_barrier.ke_barrier = 6000.0
    pose = np.zeros(6)
    f_ext = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        ctrl.compute_velocity_command(
            pose, pose, np.zeros(6), f_ext, f_des,
            v_tcp_z_actual=0.0,
        )
    # After impact timer expires, D must return to D0.
    assert ctrl.damping_z_eff == pytest.approx(25.0, abs=1.0)
    assert ctrl.damping_delay_z == pytest.approx(0.0)

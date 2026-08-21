"""Delay-aware shield: modes, energy, terminal set, pipeline debt."""

from __future__ import annotations

import math

from collections import deque

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
)
from rm75_control.control.admittance_common.delay_safety_shield import (
    DelaySafetyShield,
    SafetyShieldConfig,
    ShieldCertificationError,
    StopDxBin,
    inf_minus_fv,
    measured_power_lb,
)
from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
)


def _wide_cover_bins() -> list[StopDxBin]:
    return [
        StopDxBin(
            v0_m_s=1.0,
            a0_m_s2=50.0,
            q_remain_m=1.0,
            u_prev_m_s=1.0,
            a_cmd_m_s2=50.0,
            q_front_m_s=1.0,
            dx_ub_m=0.0002,
            n_b=40,
        )
    ]


def _cert_kwargs() -> dict:
    return dict(
        stop_dx_certified=True,
        stop_dx_bins=_wide_cover_bins(),
        pose_domain_declared=True,
        payload_domain_declared=True,
        pose_min=[-2.0] * 6,
        pose_max=[2.0] * 6,
        payload_min_kg=0.0,
        payload_max_kg=10.0,
        payload_kg=1.0,
    )


def _shield(mode: str = "observe", **kwargs) -> DelaySafetyShield:
    params = dict(
        enabled=True,
        mode=mode,
        horizon_steps=30,
        t0_s=0.005,
        tp_s=0.020,
        k_ub_n_m=800.0,
        r_f_n_s=0.0,
        r_f_window_steps=0,
        f_release_n=0.8,
        v_hold_m_s=0.015,
        u_retract_m_s=0.030,
        queue_clear_m_s=0.025,
        a_max_m_s2=2.0,
        j_max_m_s3=0.0,
        e0_j=0.004,
        eps_j=0.0005,
        e_f_n=0.0,
        bar_f_n=0.0,
        rho=0.2,
        solver_budget_us=4000.0,
    )
    params.update(kwargs)
    return DelaySafetyShield(SafetyShieldConfig(**params), 0.005)


def _update(sh: DelaySafetyShield, u_nom: float, **kwargs):
    kwargs.setdefault("f_csv", 1.0)
    kwargs.setdefault("v_actual", 0.0)
    kwargs.setdefault("f_max_n", 3.0)
    kwargs.setdefault("a_actual", 0.0)
    kwargs.setdefault("feedback_age_s", 0.0)
    kwargs.setdefault("pose_in_domain", True)
    kwargs.setdefault("payload_in_domain", True)
    return sh.update(u_nom, **kwargs)


def test_observe_sends_nom_and_records_lambda() -> None:
    sh = _shield("observe")
    out = sh.update(0.02, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
    assert out.u_sent == pytest.approx(0.02)
    assert out.shield_applied is False
    assert math.isfinite(out.lambda_obs)


def test_force_rollout_rejects_continued_press() -> None:
    sh = _shield(
        "force",
        k_ub_n_m=2000.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.0] * 40,
        position_error_ub_plus_m=[0.0] * 40,
    )
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    sh._v_plant = 0.0
    ok_press, f_press, *_ = sh._rollout(
        0.08,
        f0=2.0,
        energy0=0.004,
        enforce_force=True,
        enforce_energy=False,
        rho=0.0,
        f_max=2.15,
    )
    ok_stop, f_stop, *_ = sh._rollout(
        0.0,
        f0=2.0,
        energy0=0.004,
        enforce_force=True,
        enforce_energy=False,
        rho=0.0,
        f_max=2.15,
    )
    assert ok_press is False
    assert ok_stop is True
    assert f_press > f_stop


def test_infeasible_does_not_forge_lambda_zero() -> None:
    sh = _shield(
        "force",
        k_ub_n_m=50000.0,
        horizon_steps=4,
        r_f_n_s=200.0,
        r_f_window_steps=4,
        f_release_n=0.1,
        u_retract_m_s=0.001,
        a_max_m_s2=0.05,
        j_max_m_s3=0.05,
        **_cert_kwargs(),
    )
    out = _update(sh, 0.08, f_csv=2.9, v_actual=0.06, f_max_n=3.0)
    if not out.shield_feasible:
        assert math.isnan(out.lambda_obs)


def test_energy_not_reset_on_zero_force() -> None:
    sh = _shield("observe")
    sh.update(0.02, f_csv=2.0, v_actual=0.03, f_max_n=3.0)
    drained = float(sh.energy_lb_j)
    assert drained < sh.cfg.e0_j
    sh.update(-0.01, f_csv=0.0, v_actual=0.0, f_max_n=3.0)
    assert sh.energy_lb_j == pytest.approx(drained, abs=0.002)
    assert sh.energy_lb_j != pytest.approx(sh.cfg.e0_j)


def test_press_positive_power_is_negative() -> None:
    p = measured_power_lb(2.0, 0.03, 0.0, 0.0)
    assert p < 0.0
    assert inf_minus_fv(1.5, 2.5, 0.01, 0.04) < 0.0


def test_terminal_hold_is_invariant() -> None:
    sh = _shield(
        "force",
        r_f_n_s=0.0,
        r_f_window_steps=0,
        velocity_error_ub_m_s=[0.0] * 40,
        velocity_error_persistent_m_s=0.0,
    )
    assert sh.terminal_set_invariant(require_energy=False) is False
    d_t = sh.terminal_indent_ub()
    assert math.isfinite(d_t)
    assert d_t > 0.0
    sh.cfg.x_detach_m = d_t
    assert sh.terminal_set_invariant(require_energy=False) is True


def test_terminal_hold_not_invariant_with_residual_ev() -> None:
    sh = _shield(
        "force",
        r_f_n_s=0.0,
        r_f_window_steps=0,
        velocity_error_ub_m_s=[0.003] * 40,
        velocity_error_persistent_m_s=0.003,
        x_detach_m=1.0,
    )
    assert sh.terminal_set_invariant(require_energy=False) is False


def test_finite_table_zero_is_not_ev_infinity() -> None:
    sh = _shield(
        "force",
        r_f_n_s=0.0,
        r_f_window_steps=0,
        velocity_error_ub_m_s=[0.0] * 40,
        x_detach_m=1.0,
    )
    assert math.isinf(sh.terminal_indent_ub())
    assert sh.terminal_set_invariant(require_energy=False) is False


def test_terminal_gap_covers_box_not_origin() -> None:
    ev = [0.003] * 5 + [0.0] * 35
    sh = _shield(
        "force",
        r_f_n_s=0.0,
        r_f_window_steps=0,
        velocity_error_ub_m_s=ev,
        velocity_error_persistent_m_s=0.0,
        x_detach_m=0.0,
    )
    assert sh.terminal_set_invariant(require_energy=False) is False
    sh.cfg.x_detach_m = 5 * sh.dt_s * 0.003
    assert sh.terminal_set_invariant(require_energy=False) is False
    sh.cfg.x_detach_m = sh.terminal_indent_ub()
    assert sh.terminal_set_invariant(require_energy=False) is True


def test_backup_tail_shift_stays_feasible() -> None:
    sh = _shield("force", k_ub_n_m=400.0, r_f_n_s=0.0, **_cert_kwargs())
    first = _update(sh, 0.01, f_csv=1.0, v_actual=0.0, f_max_n=4.0)
    assert first.shield_feasible
    second = _update(sh, 0.0, f_csv=1.0, v_actual=0.0, f_max_n=4.0)
    assert second.shield_feasible


def test_pipeline_dx_positive_when_only_accel() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(enabled=True, t_react_s=0.055, tau_stop_s=0.08)
    )
    dx0 = damper.pipeline_dx_ub(v_tcp_z_actual=0.0, a_tcp_z_actual=0.0)
    dx_a = damper.pipeline_dx_ub(v_tcp_z_actual=0.0, a_tcp_z_actual=1.5)
    assert dx0 == pytest.approx(0.0)
    assert dx_a > 0.0


def test_v_act_tightens_press_cap() -> None:
    cfg = ForceBarrierConfig(
        enabled=True,
        t_react_s=0.055,
        tau_stop_s=0.08,
        v_min_press_m_s=0.0,
        budget_min_n=0.5,
        budget_frac=0.0,
        bar_f_n=0.0,
        e_f_n=0.0,
        e_x_m=0.0,
        stiffness_cap_enabled=True,
    )
    rest = ForceSpaceVelocityDamper(cfg)
    moving = ForceSpaceVelocityDamper(cfg)
    kwargs = dict(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=2000.0,
        tau_s=0.055,
        mass_eq_kg=1.0,
        a_tcp_z_actual=0.0,
    )
    cap0, _ = rest.caps(v_tcp_z_actual=0.0, **kwargs)
    cap_v, _ = moving.caps(v_tcp_z_actual=0.05, **kwargs)
    assert cap_v < cap0


def test_negative_velocity_does_not_open_press() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(
            enabled=True,
            t_react_s=0.055,
            v_min_press_m_s=0.0,
            budget_min_n=0.5,
            budget_frac=0.0,
            bar_f_n=0.0,
            e_f_n=0.0,
            e_x_m=0.0,
            stiffness_cap_enabled=True,
        )
    )
    damper.f_dot_z = -80.0
    cap, _ = damper.caps(
        f_z=2.0,
        f_des_z=2.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=2000.0,
        tau_s=0.055,
        mass_eq_kg=1.0,
        v_tcp_z_actual=-0.08,
        a_tcp_z_actual=-2.0,
    )
    assert cap == pytest.approx(0.5 / (2000.0 * 0.055), rel=0.05)


def test_cap_press_zero_keeps_retract() -> None:
    damper = ForceSpaceVelocityDamper(ForceBarrierConfig(enabled=True))
    damper.cap_press_z = 0.0
    damper.cap_retract_z = 0.08
    assert damper.clamp_velocity(0.04) == pytest.approx(0.0)
    assert damper.clamp_velocity(-0.05) == pytest.approx(-0.05)


def test_controller_observe_sent_equals_capped() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.enabled = True
    cfg.force_axis_slew_press_m_s2 = 1.2
    cfg.force_axis_jerk_max_m_s3 = 40.0
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_ext = np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    v = ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.0, dt_actual=0.005
    )
    assert v[2] == pytest.approx(ctrl.u_sent_z)
    assert ctrl.u_sent_z == pytest.approx(ctrl.u_nom_capped_z)
    assert ctrl.shield_applied is False


def test_observe_records_energy_below_eps() -> None:
    sh = _shield("observe", e0_j=0.0006, eps_j=0.0005)
    for _ in range(8):
        sh.update(0.04, f_csv=4.0, v_actual=0.05, f_max_n=3.0)
    assert sh.energy_lb_j < sh.cfg.eps_j


def test_force_mode_cuts_lambda_when_already_pressing() -> None:
    sh = _shield(
        "force",
        k_ub_n_m=1500.0,
        enforce_terminal=False,
        j_max_m_s3=0.0,
        a_max_m_s2=8.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        bar_f_n=0.0,
        tp_s=0.015,
        t0_s=0.0,
        velocity_error_ub_m_s=[0.001] * 30,
        **_cert_kwargs(),
    )
    sh._v_plant = 0.012
    sh._u_prev = 0.012
    sh._u_prev2 = 0.012
    out = _update(sh, 0.08, f_csv=2.05, v_actual=0.012, f_max_n=2.20)
    if out.shield_feasible:
        assert out.lambda_obs < 1.0 - 1e-6
        assert out.u_sent < out.u_nom
    else:
        assert math.isnan(out.lambda_obs)
        assert out.u_sent != pytest.approx(out.u_nom)


def test_passive_rho_is_zero_ospf_uses_rho() -> None:
    passive = _shield("passive", rho=0.4)
    ospf = _shield("ospf", rho=0.4)
    assert passive.cfg.rho_used() == pytest.approx(0.0)
    assert ospf.cfg.rho_used() == pytest.approx(0.4)


def test_shift_tail_recursive_feasibility() -> None:
    sh = _shield("force", k_ub_n_m=400.0, r_f_n_s=0.0, enforce_terminal=False)
    assert sh.shift_tail_feasible(
        0.01,
        f0=1.0,
        energy0=0.004,
        enforce_force=True,
        enforce_energy=False,
        rho=0.0,
        f_max=4.0,
    )


def test_tube_violation_uses_predicted_plant() -> None:
    sh = _shield("observe")
    sh._v_plant = 0.0
    out = sh.update(0.0, f_csv=1.0, v_actual=0.05, f_max_n=3.0)
    assert out.tube_violation is True


def test_pipeline_without_measured_speed_is_zero() -> None:
    sh = _shield("observe")
    sh._v_plant = 0.04
    assert sh.pipeline_penetration_ub(v_actual=None) == pytest.approx(0.0)


def test_pipeline_keeps_pending_press_queue() -> None:
    sh = _shield("observe", t0_s=0.050, tp_s=0.020, e_x_m=0.0, r_f_n_s=0.0)
    delay_n = max(sh._delay_steps(), 1)
    pending = deque([0.04] * delay_n, maxlen=sh._delay.maxlen)
    sh._delay = pending
    sh._v_plant = 0.0
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    dx = sh.pipeline_penetration_ub(v_actual=0.0)
    assert dx > 0.0
    assert list(sh._delay) == list(pending)


def test_stop_dx_lookup_is_monotonic_covering() -> None:
    sh = _shield("observe")
    sh.cfg.stop_dx_certified = True
    sh.cfg.stop_dx_bins = [
        StopDxBin(v0_m_s=0.010, dx_ub_m=0.0004),
        StopDxBin(v0_m_s=0.020, dx_ub_m=0.0008),
    ]
    assert sh.lookup_stop_dx(0.009, 0.0, 0.0) == pytest.approx(0.0004)
    assert sh.lookup_stop_dx(0.015, 0.0, 0.0) == pytest.approx(0.0008)
    assert math.isinf(sh.lookup_stop_dx(0.030, 0.0, 0.0))
    assert math.isinf(sh.lookup_stop_dx(0.009, 0.0, 0.0, u_prev=0.02))
    sh.cfg.k_ub_n_m = 1000.0
    assert sh.max_safe_approach_m_s(room_n=0.5, a0=0.0, q0=0.0) == pytest.approx(0.010)
    assert sh.max_safe_approach_m_s(room_n=0.3, a0=0.0, q0=0.0) == pytest.approx(0.0)


def test_rollout_uses_error_x_plus_every_tick() -> None:
    sh = _shield(
        "observe",
        t0_s=0.0,
        k_ub_n_m=1000.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.0] * 40,
        position_error_ub_m=[0.0] * 40,
        position_error_ub_plus_m=[0.001] * 40,
    )
    sh._v_plant = 0.02
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    _ok, f_ub, _e, _n, _t, dx, _reason = sh._rollout(
        0.0,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    assert dx >= 0.0
    assert f_ub >= 1000.0 * 0.001 - 1e-9


def test_error_x_plus_updates_when_predicted_speed_is_negative() -> None:
    sh = _shield(
        "observe",
        t0_s=0.0,
        k_ub_n_m=1000.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.0] * 40,
        position_error_ub_plus_m=[0.002] * 40,
    )
    sh._v_plant = -0.01
    sh._u_prev = -0.01
    sh._u_prev2 = -0.01
    _ok, f_ub, *_rest = sh._rollout(
        -0.01,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    assert f_ub >= 1000.0 * 0.002 - 1e-9


def test_signed_position_error_is_not_the_force_indent() -> None:
    sh = _shield(
        "observe",
        t0_s=0.0,
        k_ub_n_m=1000.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.0] * 40,
        position_error_ub_m=[0.010] * 40,
        position_error_ub_plus_m=[],
    )
    sh._v_plant = -0.02
    sh._u_prev = -0.02
    sh._u_prev2 = -0.02
    _ok, f_ub, *_rest = sh._rollout(
        -0.02,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    assert f_ub < 1.0


def test_lookup_uses_measured_a_plus_not_command_accel() -> None:
    sh = _shield("observe")
    sh.cfg.stop_dx_certified = True
    sh.cfg.stop_dx_bins = [
        StopDxBin(v0_m_s=0.040, a0_m_s2=0.10, q_remain_m=0.001, dx_ub_m=0.0003),
        StopDxBin(v0_m_s=0.040, a0_m_s2=8.0, q_remain_m=0.001, dx_ub_m=0.0020),
    ]
    sh._v_plant = 0.02
    sh._a_plus = 0.0
    sh._u_prev = 0.04
    sh._u_prev2 = 0.0
    assert sh.dt_s == pytest.approx(0.005)
    cmd_a = abs(sh._u_prev - sh._u_prev2) / sh.dt_s
    assert cmd_a > 1.0
    assert sh.lookup_stop_dx(0.02, sh._a_plus, 0.0) == pytest.approx(0.0003)
    assert sh.lookup_stop_dx(0.02, cmd_a, 0.0) == pytest.approx(0.0020)


def test_certified_indent_is_first_tick_plus_backup_tail() -> None:
    sh = _shield(
        "observe",
        t0_s=0.0,
        k_ub_n_m=1000.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.0] * 40,
        position_error_ub_plus_m=[0.0] * 40,
        u_retract_m_s=0.0,
        a_max_m_s2=50.0,
        j_max_m_s3=0.0,
    )
    sh._v_plant = 0.02
    sh._a_plus = 0.0
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    _ok_m, f_model, *_r = sh._rollout(
        0.02,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    sh.cfg.stop_dx_certified = True
    sh.cfg.stop_dx_bins = [
        StopDxBin(
            v0_m_s=0.080,
            a0_m_s2=50.0,
            q_remain_m=1.0,
            u_prev_m_s=0.08,
            a_cmd_m_s2=50.0,
            q_front_m_s=1.0,
            dx_ub_m=0.0002,
        ),
    ]
    _ok_t, f_table, *_r2 = sh._rollout(
        0.02,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    assert f_model > f_table + 0.05
    dt = sh.dt_s
    v1 = math.exp(-dt / sh.cfg.tp_s) * 0.02 + (1.0 - math.exp(-dt / sh.cfg.tp_s)) * 0.02
    dx1 = dt * max(v1, 0.0)
    assert f_table == pytest.approx(1000.0 * (dx1 + 0.0002), abs=1e-6)


def test_certified_tail_queries_worst_successor() -> None:
    sh = _shield(
        "observe",
        t0_s=0.0,
        k_ub_n_m=1000.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        velocity_error_ub_m_s=[0.012] * 40,
        position_error_ub_plus_m=[0.0] * 40,
        u_retract_m_s=0.0,
        a_max_m_s2=50.0,
        j_max_m_s3=0.0,
    )
    sh._v_plant = 0.02
    sh._a_plus = 0.0
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    sh.cfg.stop_dx_certified = True
    sh.cfg.stop_dx_bins = [
        StopDxBin(
            v0_m_s=0.025,
            a0_m_s2=0.50,
            q_remain_m=1.0,
            u_prev_m_s=0.08,
            a_cmd_m_s2=50.0,
            q_front_m_s=1.0,
            dx_ub_m=0.0002,
        ),
        StopDxBin(
            v0_m_s=0.040,
            a0_m_s2=4.00,
            q_remain_m=1.0,
            u_prev_m_s=0.08,
            a_cmd_m_s2=50.0,
            q_front_m_s=1.0,
            dx_ub_m=0.0015,
        ),
    ]
    _ok, f_ub, *_r = sh._rollout(
        0.02,
        f0=0.0,
        energy0=0.004,
        enforce_force=False,
        enforce_energy=False,
        rho=0.0,
        f_max=1e9,
    )
    dt = sh.dt_s
    v1 = math.exp(-dt / sh.cfg.tp_s) * 0.02 + (1.0 - math.exp(-dt / sh.cfg.tp_s)) * 0.02
    ev = 0.012
    v_q, a_q = sh._worst_successor(0.02, v1, 0.0, ev, float("nan"))
    assert v_q == pytest.approx(abs(v1) + ev)
    assert a_q == pytest.approx(max((v1 + ev - 0.02) / dt, 0.0))
    assert v_q > 0.025
    assert a_q > 0.50
    dx1 = dt * max(v1, 0.0)
    assert f_ub == pytest.approx(1000.0 * (dx1 + 0.0015), abs=1e-6)


def test_shield_pipeline_is_the_cap_source() -> None:
    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(enabled=True, tau_stop_s=0.08, e_x_m=0.0004)
    )
    closed = damper.pipeline_dx_ub(v_tcp_z_actual=0.03, a_tcp_z_actual=0.0)
    sourced = damper.pipeline_dx_ub(
        v_tcp_z_actual=0.03,
        a_tcp_z_actual=0.0,
        shield_dx_m=0.0012,
    )
    assert sourced == pytest.approx(0.0012)
    assert closed > sourced


def test_observe_runs_same_force_certificate() -> None:
    kwargs = dict(
        k_ub_n_m=50000.0,
        horizon_steps=8,
        e_f_n=0.0,
        r_f_n_s=0.0,
        t0_s=0.0,
        tp_s=0.020,
        enforce_terminal=False,
    )
    obs = _shield("observe", **kwargs)
    frc = _shield("force", **kwargs, **_cert_kwargs())
    for sh in (obs, frc):
        sh._v_plant = 0.04
        sh._u_prev = 0.04
        sh._u_prev2 = 0.04
    out_obs = obs.update(0.08, f_csv=4.0, v_actual=0.04, f_max_n=2.4)
    out_frc = _update(frc, 0.08, f_csv=4.0, v_actual=0.04, f_max_n=2.4)
    assert out_obs.shield_feasible is False
    assert out_frc.shield_feasible is False
    assert out_obs.infeasible_reason == "force"
    assert out_frc.infeasible_reason == "force"
    assert out_obs.u_sent == pytest.approx(0.08)
    assert out_frc.u_sent < out_frc.u_nom
    assert math.isnan(out_obs.lambda_obs)
    assert math.isnan(out_frc.lambda_obs)


def test_force_recovery_stays_on_backup() -> None:
    sh = _shield(
        "force",
        k_ub_n_m=50000.0,
        e_f_n=0.0,
        r_f_n_s=0.0,
        recovery_hold_s=0.05,
        t0_s=0.0,
        enforce_terminal=False,
        **_cert_kwargs(),
    )
    bad = _update(sh, 0.08, f_csv=5.0, v_actual=0.0, f_max_n=2.4)
    assert bad.recovery_latched is True
    assert bad.u_sent != pytest.approx(0.08)
    good = _update(sh, 0.08, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
    assert good.recovery_latched is True
    assert good.u_sent != pytest.approx(0.08)


def test_suspect_loss_does_not_release_same_tick() -> None:
    from rm75_control.control.admittance_common.contact_state import (
        PhysicalContactConfig,
        PhysicalContactTracker,
    )

    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.recontact_settle_m_s = 0.003
    cfg.recontact_settle_hold_s = 0.050
    cfg.physical_contact = PhysicalContactConfig(
        enabled=True,
        enter_n=0.8,
        hard_enter_n=1.5,
        exit_n=0.50,
        enter_confirm_s=0.010,
        exit_confirm_s=0.050,
    )
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl.recontact_slow_latched is True
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.array([0.0, 0.0, 0.30, 0.0, 0.0, 0.0]),
        f_des,
        v_tcp_z_actual=0.0,
        dt_actual=0.005,
    )
    assert ctrl.physical_contact_state == PhysicalContactTracker.SUSPECT_LOSS
    assert ctrl.contact_present
    assert ctrl.recontact_slow_latched is True
    assert ctrl.recontact_detached_seen is False


def test_air_seek_uses_v_seek_free() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.recontact_vz_cap_m_s = 0.012
    cfg.force_barrier.v_seek_free_m_s = 0.020
    ctrl = AdmittanceController(0.005, cfg)
    # 022208 reproduced this state: an impact-conservative 2000 N/m cap
    # must not schedule virgin free space down to 1/(2000*0.055)=9.091 mm/s.
    ctrl.ke_cap_n_m = 2000.0
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.zeros(6),
        f_des,
        v_tcp_z_actual=0.0,
        dt_actual=0.005,
    )
    assert ctrl.contact_present is False
    assert ctrl.force_task_armed is False
    assert ctrl._use_delay_safe_press() is False
    assert ctrl._press_vz_cap() == pytest.approx(0.020, abs=1e-9)
    assert ctrl.cap_press_z == pytest.approx(0.020, abs=1e-9)
    assert ctrl.u_nom_capped_z <= 0.020 + 1e-6
    assert ctrl.u_nom_capped_z >= 0.0


def test_first_contact_uses_delay_safe_not_seek() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.recontact_vz_cap_m_s = 0.012
    cfg.force_barrier.v_seek_free_m_s = 0.020
    cfg.recontact_release_force_frac = 0.70
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.zeros(6),
        f_des,
        v_tcp_z_actual=0.0,
        dt_actual=0.005,
    )
    assert ctrl._press_vz_cap() == pytest.approx(0.020, abs=1e-9)
    for _ in range(6):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    v_safe = ctrl._v_delay_safe()
    assert v_safe < 0.010
    assert v_safe > 0.0
    assert ctrl._use_delay_safe_press() is True
    assert ctrl._press_vz_cap() == pytest.approx(v_safe, abs=1e-9)
    assert ctrl.cap_press_z <= v_safe + 1e-9
    assert ctrl.u_nom_capped_z <= v_safe + 1e-6


def test_confirmed_contact_uses_full_chase_cap() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.force_barrier.v_seek_free_m_s = 0.020
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(16):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl.recontact_slow_latched is False
    assert ctrl._press_vz_cap() == pytest.approx(0.08)
    assert ctrl.v_recontact_cap_m_s == pytest.approx(0.0)


def test_recovery_requires_cleared_delay_queue() -> None:
    from collections import deque

    sh = _shield(
        "force",
        t0_s=0.050,
        tp_s=0.020,
        recovery_hold_s=0.020,
        queue_clear_m_s=0.015,
        v_hold_m_s=0.015,
        a_hold_m_s2=0.15,
        k_ub_n_m=400.0,
        r_f_n_s=0.0,
        e_f_n=0.0,
        enforce_terminal=False,
        **_cert_kwargs(),
    )
    n = max(sh._delay_steps(), 1)
    sh._delay = deque([0.08] * n, maxlen=n)
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    sh._v_plant = 0.0
    sh._recovery_latched = True
    sh._recovery_ok_s = 0.0
    dirty = _update(sh, 0.0, f_csv=1.0, v_actual=0.0, f_max_n=3.0, a_actual=0.0)
    assert dirty.recovery_latched is True
    sh._delay = deque([0.0] * n, maxlen=n)
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    out = None
    for _ in range(8):
        out = _update(sh, 0.0, f_csv=1.0, v_actual=0.0, f_max_n=3.0, a_actual=0.0)
    assert out is not None
    assert out.recovery_latched is False


def test_recontact_sequence_releases_only_after_settled_hold() -> None:
    from rm75_control.control.admittance_common.contact_state import (
        PhysicalContactConfig,
        PhysicalContactTracker,
    )

    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.recontact_settle_m_s = 0.003
    cfg.recontact_settle_hold_s = 0.050
    cfg.physical_contact = PhysicalContactConfig(
        enabled=True,
        enter_n=0.8,
        hard_enter_n=1.5,
        exit_n=0.50,
        enter_confirm_s=0.010,
        exit_confirm_s=0.025,
    )
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

    def tick(fz: float, raw: float, v: float) -> None:
        f_ext = np.array([0.0, 0.0, fz, 0.0, 0.0, 0.0])
        f_raw = np.array([0.0, 0.0, raw, 0.0, 0.0, 0.0])
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            f_ext_raw=f_raw,
            v_tcp_z_actual=v,
            dt_actual=0.005,
        )

    for _ in range(10):
        tick(1.2, 1.2, 0.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.recontact_slow_latched is True

    for _ in range(3):
        tick(0.30, 0.30, 0.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.SUSPECT_LOSS
    assert ctrl.recontact_slow_latched is True
    assert ctrl.recontact_detached_seen is False

    for _ in range(4):
        tick(0.30, 0.30, 0.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    assert ctrl.contact_present is False
    assert ctrl.recontact_detached_seen is True
    assert ctrl.recontact_slow_latched is True

    for _ in range(4):
        tick(0.10, 0.10, 0.040)
    assert ctrl.recontact_slow_latched is True

    tick(0.90, 1.90, 0.060)
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    assert ctrl.recontact_slow_latched is True
    for _ in range(4):
        tick(1.2, 1.2, 0.050)
        assert ctrl.recontact_slow_latched is True

    for i in range(9):
        tick(1.6, 1.6, 0.0)
        assert ctrl.recontact_slow_latched is True, i
    tick(1.6, 1.6, 0.0)
    assert ctrl.recontact_slow_latched is False


def test_retract_speed_does_not_release_recontact_latch() -> None:
    from rm75_control.control.admittance_common.contact_state import (
        PhysicalContactConfig,
        PhysicalContactTracker,
    )

    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.recontact_settle_m_s = 0.003
    cfg.recontact_settle_hold_s = 0.050
    cfg.physical_contact = PhysicalContactConfig(
        enabled=True,
        enter_n=0.8,
        hard_enter_n=1.5,
        exit_n=0.50,
        enter_confirm_s=0.010,
        exit_confirm_s=0.025,
    )
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])

    def tick(fz: float, raw: float, v: float) -> None:
        f_ext = np.array([0.0, 0.0, fz, 0.0, 0.0, 0.0])
        f_raw = np.array([0.0, 0.0, raw, 0.0, 0.0, 0.0])
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            f_ext,
            f_des,
            f_ext_raw=f_raw,
            v_tcp_z_actual=v,
            dt_actual=0.005,
        )

    for _ in range(10):
        tick(1.2, 1.2, 0.0)
    for _ in range(8):
        tick(0.30, 0.30, 0.0)
    assert ctrl.physical_contact_state == PhysicalContactTracker.LOST
    tick(0.90, 1.90, 0.060)
    assert ctrl.physical_contact_reacquire_event
    assert ctrl.physical_contact_state == PhysicalContactTracker.CONTACT
    for _ in range(12):
        tick(1.2, 1.2, -0.040)
        assert ctrl.recontact_slow_latched is True
    for i in range(9):
        tick(1.6, 1.6, 0.0)
        assert ctrl.recontact_slow_latched is True, i
    tick(1.6, 1.6, 0.0)
    assert ctrl.recontact_slow_latched is False


def test_zero_press_cap_does_not_reopen_positive_clip() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_ext = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.0, dt_actual=0.005
    )
    assert ctrl.cap_press_z == pytest.approx(0.0, abs=1e-9)
    assert ctrl.u_nom_capped_z <= 0.0 + 1e-9


def test_contact_loss_latches_slow_reapproach() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.recontact_vz_cap_m_s = 0.012
    cfg.physical_contact.exit_n = 0.35
    cfg.physical_contact.exit_confirm_s = 0.020
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.2, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    for _ in range(8):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.zeros(6),
            f_des,
            v_tcp_z_actual=0.020,
            dt_actual=0.005,
        )
    assert ctrl.recontact_slow_latched
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.zeros(6),
        f_des,
        v_tcp_z_actual=0.060,
        dt_actual=0.005,
    )
    assert ctrl.recontact_slow_latched
    assert ctrl.contact_present is False
    assert ctrl._use_delay_safe_press() is False
    assert ctrl.v_recontact_cap_m_s == pytest.approx(0.0)
    assert ctrl._press_vz_cap() == pytest.approx(ctrl._v_air_seek(), abs=1e-9)
    assert ctrl.cap_press_z == pytest.approx(ctrl._v_air_seek(), abs=1e-9)
    assert ctrl.u_nom_capped_z > 0.005


def test_controller_no_post_shield_zero_trap() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.force_axis_slew_press_m_s2 = 1.2
    cfg.force_axis_slew_retract_m_s2 = 1.2
    cfg.force_axis_jerk_max_m_s3 = 40.0
    ctrl = AdmittanceController(0.005, cfg)
    ctrl.last_v_cmd[2] = -0.04
    pose = np.zeros(6)
    f_ext = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    v = ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), f_ext, f_des, v_tcp_z_actual=0.01, dt_actual=0.005
    )
    assert v[2] <= 0.0 + 1e-9


def test_controller_refuses_uncertified_force_mode() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.mode = "force"
    with pytest.raises(ShieldCertificationError, match="certified"):
        AdmittanceController(0.005, cfg)


def test_controller_refuses_force_when_payload_is_outside_declared_range() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.mode = "force"
    cfg.safety_shield.stop_dx_certified = True
    cfg.safety_shield.stop_dx_bins = _wide_cover_bins()
    cfg.safety_shield.pose_domain_declared = True
    cfg.safety_shield.pose_min = [-2.0] * 6
    cfg.safety_shield.pose_max = [2.0] * 6
    cfg.safety_shield.payload_domain_declared = True
    cfg.safety_shield.payload_min_kg = 0.0
    cfg.safety_shield.payload_max_kg = 1.0
    cfg.safety_shield.payload_kg = 1.5
    with pytest.raises(ShieldCertificationError, match="payload_kg"):
        AdmittanceController(0.005, cfg)


def test_controller_refuses_passive_without_terminal_proof() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.mode = "passive"
    cfg.safety_shield.stop_dx_certified = True
    cfg.safety_shield.stop_dx_bins = [
        StopDxBin(v0_m_s=0.02, dx_ub_m=0.0002),
    ]
    cfg.safety_shield.velocity_error_persistent_m_s = 0.0
    cfg.safety_shield.x_detach_m = 1.0
    with pytest.raises(ShieldCertificationError, match="terminal_invariance_proven"):
        AdmittanceController(0.005, cfg)


def test_writing_ev_inf_zero_is_not_a_terminal_proof() -> None:
    sh = _shield(
        "force",
        velocity_error_ub_m_s=[0.0] * 40,
        velocity_error_persistent_m_s=0.0,
        x_detach_m=1.0,
    )
    sh.cfg.x_detach_m = sh.terminal_indent_ub()
    assert sh.terminal_set_invariant() is True
    assert sh.cfg.terminal_invariance_proven is False
    sh.cfg.mode = "passive"
    assert "terminal_invariance_proven" in " ".join(sh.enforcement_blockers())


def test_domain_ok_is_fail_closed_without_certificate_inputs() -> None:
    sh = _shield("observe")
    out = sh.update(0.02, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
    assert out.domain_ok is False
    assert out.u_sent == pytest.approx(0.02)
    ok, reasons = sh.evaluate_domain(
        v_actual=0.0,
        a_actual=0.0,
        feedback_age_s=0.0,
        pose_in_domain=True,
        payload_in_domain=True,
    )
    assert ok is False
    assert "lookup" in reasons


def test_tube_violation_latches_backup_in_force() -> None:
    sh = _shield("force", **_cert_kwargs())
    sh._v_plant = 0.0
    out = _update(sh, 0.02, f_csv=1.0, v_actual=0.05, f_max_n=3.0, a_actual=0.0)
    assert out.tube_violation is True
    assert out.recovery_latched is True
    assert out.u_sent != pytest.approx(0.02)
    assert abs(out.u_sent) <= abs(out.u_b) + 1e-9
    assert out.uncertified_brake is False


def test_tube_violation_uncovered_is_uncertified_brake() -> None:
    sh = _shield(
        "force",
        stop_dx_certified=True,
        stop_dx_bins=[
            StopDxBin(
                v0_m_s=0.001,
                a0_m_s2=0.001,
                q_remain_m=0.0,
                dx_ub_m=0.0001,
            )
        ],
        pose_domain_declared=True,
        payload_domain_declared=True,
        pose_min=[-2.0] * 6,
        pose_max=[2.0] * 6,
        payload_min_kg=0.0,
        payload_max_kg=10.0,
        payload_kg=1.0,
    )
    sh._v_plant = 0.0
    out = _update(sh, 0.02, f_csv=1.0, v_actual=0.05, f_max_n=3.0, a_actual=0.0)
    assert out.tube_violation is True
    assert out.uncertified_brake is True
    assert out.recovery_latched is True
    limited = sh._limit_increment(0.0, 0.0, 0.0)
    assert out.u_sent == pytest.approx(limited)


def test_domain_violation_brakes_even_when_stop_lookup_covers_state() -> None:
    sh = _shield("force", **_cert_kwargs())
    out = _update(
        sh,
        0.02,
        pose_in_domain=False,
        payload_in_domain=True,
    )
    assert sh.lookup_covers_state(0.0, 0.0) is True
    assert out.domain_ok is False
    assert out.infeasible_reason.startswith("domain:")
    assert out.uncertified_brake is True


def test_uncertified_apply_sends_limited_backup_not_zero() -> None:
    sh = _shield("force", a_max_m_s2=1.2, j_max_m_s3=40.0)
    sh._u_prev = 0.02
    sh._u_prev2 = 0.02
    out = sh.update(0.08, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
    assert out.infeasible_reason.startswith("uncertified:")
    assert out.u_sent != pytest.approx(0.0)
    assert out.u_sent == pytest.approx(
        sh._limit_increment(out.u_b, 0.02, 0.02)
    ) or out.uncertified_brake
    assert out.aj_ok is True


def test_observe_mode_mutation_to_force_is_refused() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.mode = "observe"
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des,
        v_tcp_z_actual=0.0, dt_actual=0.005, feedback_age_s=0.0,
    )
    assert ctrl.shield_applied is False
    ctrl.cfg.safety_shield.mode = "force"
    ctrl.compute_velocity_command(
        pose, pose, np.zeros(6), np.zeros(6), f_des,
        v_tcp_z_actual=0.0, dt_actual=0.005, feedback_age_s=0.0,
    )
    assert ctrl.shield_applied is True
    assert ctrl.shield_infeasible_reason.startswith("mode_changed:")
    assert ctrl.shield_uncertified_brake is True
    assert ctrl.u_sent_z != pytest.approx(ctrl.u_nom_capped_z)


def test_first_contact_stays_delay_safe_until_settled() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.recontact_settle_m_s = 0.003
    cfg.recontact_settle_hold_s = 0.050
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for i in range(4):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
        if ctrl.contact_present:
            assert ctrl._press_vz_cap() == pytest.approx(
                ctrl._v_delay_safe(), abs=1e-9
            ), i
        else:
            assert ctrl._press_vz_cap() == pytest.approx(
                ctrl._v_air_seek(), abs=1e-9
            ), i
    assert ctrl.contact_present
    assert ctrl.recontact_slow_latched is True
    for _ in range(12):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.recontact_slow_latched is False
    assert ctrl._press_vz_cap() == pytest.approx(0.08)


def test_underforce_does_not_release_first_contact_slow() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.recontact_settle_m_s = 0.003
    cfg.recontact_settle_hold_s = 0.050
    cfg.recontact_release_force_frac = 0.70
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 0.90, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl.recontact_slow_latched is True
    assert ctrl._press_vz_cap() == pytest.approx(ctrl._v_delay_safe(), abs=1e-9)


def test_controller_observe_domain_ok_is_false_without_declared_set() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.safety_shield.mode = "observe"
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    ctrl.compute_velocity_command(
        pose,
        pose,
        np.zeros(6),
        np.zeros(6),
        np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        v_tcp_z_actual=0.0,
        dt_actual=0.005,
        feedback_age_s=0.0,
    )
    assert ctrl.shield_domain_ok is False
    assert ctrl.u_sent_z == pytest.approx(ctrl.u_nom_capped_z)


def test_pose_declared_without_bounds_is_out_of_domain() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.pose_domain_declared = True
    ctrl = AdmittanceController(0.005, cfg)
    assert ctrl._pose_in_certificate_domain(np.zeros(6)) is False
    cfg.safety_shield.pose_min = [-1.0] * 6
    cfg.safety_shield.pose_max = [1.0] * 6
    assert ctrl._pose_in_certificate_domain(np.zeros(6)) is True
    pose = np.zeros(6)
    pose[2] = 2.0
    assert ctrl._pose_in_certificate_domain(pose) is False


def test_payload_declared_without_range_is_out_of_domain() -> None:
    cfg = AdmittanceConfig()
    cfg.safety_shield.payload_domain_declared = True
    ctrl = AdmittanceController(0.005, cfg)
    assert ctrl._payload_in_certificate_domain() is False
    cfg.safety_shield.payload_min_kg = 0.2
    cfg.safety_shield.payload_max_kg = 1.5
    cfg.safety_shield.payload_kg = 0.8
    assert ctrl._payload_in_certificate_domain() is True
    cfg.safety_shield.payload_kg = 2.0
    assert ctrl._payload_in_certificate_domain() is False


def test_passive_to_observe_mutation_refuses() -> None:
    sh = _shield("observe")
    sh.cfg.mode = "passive"
    out = sh.update(0.02, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
    assert out.infeasible_reason.startswith("mode_changed:")
    assert out.uncertified_brake is True
    assert out.u_sent != pytest.approx(0.02)


def test_armed_flight_keeps_air_seek_chase() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.force_barrier.v_seek_free_m_s = 0.020
    cfg.physical_contact.exit_confirm_s = 0.025
    cfg.physical_contact.exit_n = 0.50
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.contact_present
    assert ctrl._use_delay_safe_press() is False
    for _ in range(8):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.zeros(6),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl.force_task_armed
    assert ctrl.contact_present is False
    assert ctrl._use_delay_safe_press() is False
    assert ctrl._press_vz_cap() == pytest.approx(0.020, abs=1e-9)
    assert ctrl.cap_press_z == pytest.approx(0.020, abs=1e-9)
    assert ctrl.u_nom_capped_z > 0.005


def test_suspect_trough_does_not_rearm_delay_safe() -> None:
    from rm75_control.control.admittance_common.contact_state import (
        PhysicalContactTracker,
    )

    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.max_vz_tool_m_s = 0.08
    cfg.max_velocity[2] = 0.08
    cfg.physical_contact.exit_n = 0.50
    cfg.physical_contact.exit_confirm_s = 0.100
    ctrl = AdmittanceController(0.005, cfg)
    pose = np.zeros(6)
    f_des = np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    for _ in range(20):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 1.6, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
    assert ctrl._use_delay_safe_press() is False
    for _ in range(10):
        ctrl.compute_velocity_command(
            pose,
            pose,
            np.zeros(6),
            np.array([0.0, 0.0, 0.20, 0.0, 0.0, 0.0]),
            f_des,
            v_tcp_z_actual=0.0,
            dt_actual=0.005,
        )
        assert ctrl.physical_contact_state == PhysicalContactTracker.SUSPECT_LOSS
        assert ctrl._use_delay_safe_press() is False
        assert ctrl._press_vz_cap() == pytest.approx(0.08)

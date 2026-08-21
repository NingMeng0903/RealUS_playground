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
    StopDxBin,
    inf_minus_fv,
    measured_power_lb,
)
from rm75_control.control.admittance_common.force_barrier import (
    ForceBarrierConfig,
    ForceSpaceVelocityDamper,
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
    )
    out = sh.update(0.08, f_csv=2.9, v_actual=0.06, f_max_n=3.0)
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
    sh = _shield("force", r_f_n_s=0.0, r_f_window_steps=0)
    assert sh.terminal_set_invariant(require_energy=False) is True


def test_backup_tail_shift_stays_feasible() -> None:
    sh = _shield("force", k_ub_n_m=400.0, r_f_n_s=0.0)
    first = sh.update(0.01, f_csv=1.0, v_actual=0.0, f_max_n=4.0)
    assert first.shield_feasible
    second = sh.update(0.0, f_csv=1.0, v_actual=0.0, f_max_n=4.0)
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
    )
    sh._v_plant = 0.012
    sh._u_prev = 0.012
    sh._u_prev2 = 0.012
    out = sh.update(0.08, f_csv=2.05, v_actual=0.012, f_max_n=2.20)
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
        StopDxBin(v0_m_s=0.080, a0_m_s2=50.0, q_remain_m=1.0, dx_ub_m=0.0002),
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
    frc = _shield("force", **kwargs)
    for sh in (obs, frc):
        sh._v_plant = 0.04
        sh._u_prev = 0.04
        sh._u_prev2 = 0.04
    out_obs = obs.update(0.08, f_csv=4.0, v_actual=0.04, f_max_n=2.4)
    out_frc = frc.update(0.08, f_csv=4.0, v_actual=0.04, f_max_n=2.4)
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
    )
    bad = sh.update(0.08, f_csv=5.0, v_actual=0.0, f_max_n=2.4)
    assert bad.recovery_latched is True
    assert bad.u_sent != pytest.approx(0.08)
    good = sh.update(0.08, f_csv=1.0, v_actual=0.0, f_max_n=3.0)
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
    assert ctrl.recontact_slow_latched is False
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


def test_first_contact_uses_delay_safe_cap() -> None:
    cfg = AdmittanceConfig()
    cfg.var_damping_enabled = False
    cfg.adaptive_ke.enabled = False
    cfg.safety_shield.mode = "observe"
    cfg.safety_shield.k_ub_n_m = 8000.0
    cfg.recontact_vz_cap_m_s = 0.012
    cfg.force_barrier.v_seek_free_m_s = 0.030
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
    assert ctrl.contact_present is False
    assert ctrl.recontact_slow_latched is False
    v_safe = ctrl._v_delay_safe()
    assert v_safe > 0.0
    assert v_safe < 0.010
    assert ctrl.v_recontact_cap_m_s == pytest.approx(v_safe)
    assert ctrl.cap_press_z <= v_safe + 1e-9
    assert ctrl.u_nom_capped_z <= v_safe + 1e-6


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
    )
    n = max(sh._delay_steps(), 1)
    sh._delay = deque([0.08] * n, maxlen=n)
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    sh._v_plant = 0.0
    sh._recovery_latched = True
    sh._recovery_ok_s = 0.0
    dirty = sh.update(0.0, f_csv=1.0, v_actual=0.0, f_max_n=3.0, a_actual=0.0)
    assert dirty.recovery_latched is True
    sh._delay = deque([0.0] * n, maxlen=n)
    sh._u_prev = 0.0
    sh._u_prev2 = 0.0
    out = None
    for _ in range(8):
        out = sh.update(0.0, f_csv=1.0, v_actual=0.0, f_max_n=3.0, a_actual=0.0)
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
    assert ctrl.recontact_slow_latched is False

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
        tick(1.2, 1.2, 0.0)
        assert ctrl.recontact_slow_latched is True, i
    tick(1.2, 1.2, 0.0)
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
        tick(1.2, 1.2, 0.0)
        assert ctrl.recontact_slow_latched is True, i
    tick(1.2, 1.2, 0.0)
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
    assert ctrl.v_recontact_cap_m_s > 0.0
    assert ctrl.cap_press_z <= ctrl.v_recontact_cap_m_s + 1e-9
    assert ctrl.u_nom_capped_z <= ctrl.v_recontact_cap_m_s + 1e-6


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

"""Gates for coarse/fine rail allocation, reference model, and wall envelope."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    collapse_interval,
    stopping_velocity,
    wall_cap,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    MidrangingController,
    RailReferenceModel,
    RailStateObserver,
    allocate_rail,
    lpf_tau_from_fc,
    margin_weight_from_activation,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


def test_wall_cap_monotonic_and_zero_at_wall() -> None:
    lo, hi = 0.015, 0.77
    a_max, tau = 0.8, 0.06
    xs = np.linspace(lo, 0.20, 40)
    caps = [wall_cap(float(x), lo=lo, hi=hi, a_max=a_max, reaction_s=tau)[0] for x in xs]
    assert caps[0] == pytest.approx(0.0, abs=1e-12)
    for prev, cur in zip(caps, caps[1:]):
        assert cur <= prev + 1e-12
    far = wall_cap(0.40, lo=lo, hi=hi, a_max=a_max, reaction_s=tau)
    v_stop = float(stopping_velocity(0.40 - lo, a_max, tau))
    assert far[0] == pytest.approx(-v_stop)
    assert far[1] == pytest.approx(float(stopping_velocity(hi - 0.40, a_max, tau)))


def test_wall_cap_after_slew_not_just_dv_max() -> None:
    """Envelope must bind when slew would only take one a_max step."""
    v_max, a_max, dt, tau = 0.12, 0.8, 0.005, 0.06
    prev_v = -v_max
    v_des = 0.0
    dv_max = a_max * dt
    v_slew = max(prev_v - dv_max, min(prev_v + dv_max, v_des))
    lo_cap, hi_cap = wall_cap(0.016, lo=0.015, hi=0.77, a_max=a_max, reaction_s=tau)
    v_cmd = max(lo_cap, min(hi_cap, v_slew))
    assert v_slew < -0.10
    assert v_cmd > v_slew
    assert v_cmd >= lo_cap - 1e-12


def test_collapse_interval_preserves_true_empty_box() -> None:
    lo, hi = collapse_interval(
        np.array([0.02, 0.200000000001]), np.array([0.01, 0.2])
    )
    # A 10 mm/s crossing is a real infeasibility, not roundoff.  The tiny
    # second crossing is safely closed to its midpoint.
    assert lo[0] == pytest.approx(0.02)
    assert hi[0] == pytest.approx(0.01)
    assert lo[1] == pytest.approx(0.2000000000005)
    assert hi[1] == pytest.approx(0.2000000000005)


def test_allocate_rail_uses_translation_and_can_be_zero_on_pure_rotation() -> None:
    J = np.zeros((6, 8))
    J[1, 0] = 1.0
    J[0, 1] = 0.4
    J[3, 5] = 0.8
    s = np.array([0.12, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
    mw = np.ones(8)
    u_trans, _ = allocate_rail(J, np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0]), qdot_scale=s, margin_weight=mw, lam=0.05)
    assert u_trans > 0.01
    u_rot, _ = allocate_rail(J, np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0]), qdot_scale=s, margin_weight=mw, lam=0.05)
    assert abs(u_rot) < abs(u_trans)


def test_margin_weight_pushes_task_to_rail() -> None:
    J = np.zeros((6, 8))
    J[1, 0] = 1.0
    J[1, 1] = 1.0
    s = np.ones(8)
    v = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    u_eq, q_eq = allocate_rail(J, v, qdot_scale=s, margin_weight=np.ones(8), lam=0.02)
    mw = np.ones(8)
    mw[1] = 20.0
    u_rail, q_w = allocate_rail(J, v, qdot_scale=s, margin_weight=mw, lam=0.02)
    assert abs(u_rail) > abs(u_eq)
    assert abs(q_w[1]) < abs(q_eq[1])


def test_reference_model_uses_real_dt_and_committed_history() -> None:
    model = RailReferenceModel(f_c_hz=5.0, a_max=0.60, j_max=60.0, v_max=0.12, reaction_s=0.06)
    model.reset(0.0)
    v1 = model.step(0.12, 0.005, x_m=0.40)
    v2 = model.step(-0.12, 0.005, x_m=0.40)
    assert v1 > 0.0
    assert v2 >= 0.0
    # Jerk box can hold a=0 for one reverse tick, so v2 may equal v1.
    assert v2 <= v1 + 1e-12
    v_later = v2
    for _ in range(8):
        v_later = model.step(-0.12, 0.005, x_m=0.40)
    assert v_later < v1
    assert abs(model.state.a) <= 0.60 + 1e-9


def test_reference_model_does_not_reverse_in_one_tick() -> None:
    model = RailReferenceModel(f_c_hz=5.0, a_max=0.60, j_max=60.0, v_max=0.12)
    model.reset(0.0)
    for _ in range(40):
        model.step(0.02, 0.005, x_m=0.40)
    before = model.state.v
    after = model.step(-0.02, 0.005, x_m=0.40)
    assert before > 0.0
    assert after >= 0.0


def test_lpf_rise_time_matches_fc() -> None:
    tau = lpf_tau_from_fc(5.0)
    assert tau == pytest.approx(1.0 / (2.0 * math.pi * 5.0))
    t_1090 = 2.197 * tau
    assert t_1090 == pytest.approx(0.070, abs=0.005)


def test_legacy_ff_is_recorded_but_not_applied() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=0.0,
            k_esc=0.0,
            k_ff=1.0,
            e0_m=0.0,
            e1_m=0.01,
            v_reach_cap_m_s=0.0,
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    vel_ff = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    v_out, _ = task(q, sigma_scale=1.0, dt_s=0.005, vel_ff=vel_ff)
    assert abs(task.last_v_ff) > 0.05
    assert task.last_k_ff_scale == pytest.approx(1.0)
    assert abs(v_out) < abs(task.last_v_ff)


def test_midranging_error_stays_live_during_feedforward() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=5.0,
            k_esc=0.0,
            k_ff=0.0,
            e0_m=0.0,
            e1_m=0.01,
            v_lpf_tau_s=0.0,
            d_star_err0_m=1.0,
            d_band_m=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    y_des = float(kin.fk_placement(q).translation[1]) + 0.040
    task(q, sigma_scale=1.0, dt_s=0.005, y_tcp_d=y_des)
    assert abs(task.last_e_mid_m) > 1e-4
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert task.last_rail_ff_m == pytest.approx(y_des - float(task.d_pref_m), abs=1e-9)


def test_observer_tracks_low_frequency_motion() -> None:
    obs = RailStateObserver(pos_gain=0.5, vel_gain=4.0, vel_lpf_hz=8.0)
    obs.reset(0.40, 0.0)
    t = 0.0
    q = 0.40
    v_ref = 0.03
    for i in range(200):
        t += 0.005
        q += v_ref * 0.005
        sample_t = t if i % 4 == 0 else t - 0.012
        q_hat, v_hat = obs.update(
            now_s=t, dt_s=0.005, v_r_ref=v_ref, q_meas=q, sample_t=sample_t, v_meas=v_ref
        )
    assert abs(q_hat - q) < 0.003
    assert abs(v_hat - v_ref) < 0.01


def test_wall_crash_both_ends_no_chatter() -> None:
    for x0, u in ((0.10, -0.12), (0.70, 0.12)):
        model = RailReferenceModel(
            f_c_hz=5.0,
            a_max=0.80,
            j_max=60.0,
            v_max=0.12,
            reaction_s=0.06,
            soft_min_m=0.015,
            soft_max_m=0.77,
        )
        model.reset(0.0)
        x = x0
        v = 0.0
        zeros = 0
        last_sign = 0
        for _ in range(400):
            v = model.step(u, 0.005, x_m=x)
            x = float(np.clip(x + v * 0.005, 0.0, 0.80))
            sign = 1 if v > 1e-6 else (-1 if v < -1e-6 else 0)
            if last_sign != 0 and sign != 0 and sign != last_sign:
                zeros += 1
            if sign != 0:
                last_sign = sign
        assert 0.015 - 1e-6 <= x <= 0.77 + 1e-6
        assert zeros <= 1


def test_margin_weight_from_activation_grows_near_limit() -> None:
    q = np.array([0.4, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mid = np.zeros(8)
    half = np.full(8, 1.6)
    w = margin_weight_from_activation(q, mid, half, k_margin=4.0, activation=0.7)
    assert w[1] > w[2]


def test_midranging_pi_integrates_and_freezes_on_wall() -> None:
    ctrl = MidrangingController(kp=0.40, ki=0.80, v_max=0.03)
    u0 = ctrl.step(0.02, 0.005, freeze=False)
    for _ in range(20):
        u1 = ctrl.step(0.02, 0.005, freeze=False)
    assert u1 > u0 + 1.0e-4
    integ = ctrl.integ
    ctrl.step(0.02, 0.005, freeze=True)
    assert ctrl.integ == pytest.approx(integ)
    sat = MidrangingController(kp=4.0, ki=20.0, v_max=0.03)
    for _ in range(200):
        sat.step(0.10, 0.005, freeze=False)
    assert abs(sat.step(0.10, 0.005)) <= 0.03 + 1e-12
    integ_sat = sat.integ
    sat.step(0.10, 0.005, freeze=True)
    assert sat.integ == pytest.approx(integ_sat)


def test_midranging_is_sign_symmetric_in_y() -> None:
    plus = MidrangingController(kp=1.2, ki=0.80, v_max=0.12)
    minus = MidrangingController(kp=1.2, ki=0.80, v_max=0.12)
    u_plus = plus.step(0.04, 0.005)
    u_minus = minus.step(-0.04, 0.005)
    assert u_plus == pytest.approx(-u_minus, abs=1e-12)
    assert u_plus > 0.0


def test_allocate_rail_share_rises_with_e_mid() -> None:
    J = np.zeros((6, 8))
    J[1, 0] = 1.0
    J[1, 1] = 1.0
    s = np.array([0.12, 1.5, 1.5, 1.5, 1.5, 2.0, 2.0, 2.0])
    mw = np.ones(8)
    v = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    shares = []
    for e_mid in (0.0, 0.02, 0.04, 0.08):
        u_r, qdot = allocate_rail(
            J,
            v,
            qdot_scale=s,
            margin_weight=mw,
            lam=0.05,
            e_mid=e_mid,
            k_err=4.0,
            e_ref=0.08,
        )
        tot = abs(float(u_r)) + abs(float(qdot[1]))
        shares.append(abs(float(u_r)) / max(tot, 1e-12))
    for prev, cur in zip(shares, shares[1:]):
        assert cur >= prev - 1e-9
    assert shares[-1] > shares[0]


def test_wall_leave_only_sign_at_hard_band() -> None:
    from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
        wall_leave_only_sign,
    )

    assert wall_leave_only_sign(0.40, hard_min_m=0.005, hard_max_m=0.78, band_m=0.025) == 0.0
    assert wall_leave_only_sign(0.76, hard_min_m=0.005, hard_max_m=0.78, band_m=0.025) == 1.0
    assert wall_leave_only_sign(0.02, hard_min_m=0.005, hard_max_m=0.78, band_m=0.025) == -1.0


def test_midranging_projects_into_wall_and_does_not_windup() -> None:
    ctrl = MidrangingController(kp=1.2, ki=0.80, v_max=0.12)
    for _ in range(80):
        u = ctrl.step(0.05, 0.005, leave_only_sign=1.0, u_committed=0.0)
        assert u <= 1.0e-9
    assert ctrl.integ < 0.02


def test_observer_ignores_optimistic_v_r_ref() -> None:
    obs = RailStateObserver(pos_gain=0.35, vel_gain=2.0, vel_lpf_hz=8.0)
    obs.reset(0.40, 0.0)
    t = 0.0
    q = 0.40
    v_actual = 0.0
    for i in range(80):
        t += 0.005
        q += v_actual * 0.005
        sample_t = t if i % 4 == 0 else t - 0.012
        _q_hat, v_hat = obs.update(
            now_s=t,
            dt_s=0.005,
            v_r_ref=0.12,
            q_meas=q,
            sample_t=sample_t,
            v_meas=v_actual,
            v_written=0.0,
        )
    assert abs(v_hat) < 0.02


def test_secondary_rate_filter_holds_target_between_15hz_samples() -> None:
    from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
        SecondaryRateFilter,
    )

    filt = SecondaryRateFilter(2, target_hz=15.0)
    j_max = np.array([3.0, 40.0])
    first = filt.step(np.array([0.06, 0.0]), 0.005, j_max)
    second = filt.step(np.array([0.0, 0.4]), 0.005, j_max)
    # 15 Hz period is ~67 ms; the second 5 ms tick must not retarget.
    assert filt.target[0] == pytest.approx(0.06, abs=1e-12)
    assert first[0] != pytest.approx(second[0], abs=1e-15)
    zeroed = None
    for _ in range(80):
        zeroed = filt.step(np.zeros(2), 0.005, j_max, force_target=True)
    assert zeroed is not None
    assert abs(float(zeroed[0])) < 1.0e-3

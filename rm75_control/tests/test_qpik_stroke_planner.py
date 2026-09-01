"""One-shot (d*, ψ*) stroke planner + rail feedforward.

The force-side phase-2 work was reverted to 55e261d after hardware showed a
contact-loss / impact regression; only the rail-side planner survived, so this
file covers just that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    StrokeInfeasibleError,
    arm_respects_floor,
    joint_margin_frac,
    nearest_planar_psi,
    stroke_score,
    wrist_band_frac,
)
from rm75_control.kinematics.srs_ik import psi_from_q
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


DT = 0.005
_SEED_Q = np.array(
    [0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370]
)


def test_minmax_rejects_empty_stroke() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3, rail_margin_m=0.02),
    )
    rt.reset(_SEED_Q)
    with pytest.raises(StrokeInfeasibleError, match="reduce amplitude"):
        rt.plan_stroke(
            _SEED_Q,
            y_center_m=0.4,
            amplitude_m=2.0,
            rail_lo=0.005,
            rail_hi=0.78,
        )


def test_minmax_does_not_park_rail_at_a_stop() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=5, n_d=6, n_psi=5, w_sigma=0.5),
    )
    rt.reset(_SEED_Q)
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d_star, psi_star = rt.plan_stroke(
        _SEED_Q,
        y_center_m=y_c,
        amplitude_m=0.05,
        rail_lo=0.005,
        rail_hi=0.78,
    )
    y_lo, y_hi = y_c - 0.05, y_c + 0.05
    assert y_lo - d_star >= 0.02 - 1e-9
    assert y_hi - d_star <= 0.76 + 1e-9
    assert np.isfinite(psi_star)


def test_psi_star_is_wrapped_for_telemetry() -> None:
    """Chosen ψ* stays in (−π, π] for CSV / rate-limit wrap."""
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=4, n_psi=8, w_sigma=0.5),
    )
    rt.reset(_SEED_Q)
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    _d, psi_star = rt.plan_stroke(
        _SEED_Q,
        y_center_m=y_c,
        amplitude_m=0.05,
        rail_lo=0.005,
        rail_hi=0.78,
    )
    assert -np.pi - 1e-9 <= psi_star <= np.pi + 1e-9
    assert -np.pi - 1e-9 <= rt.psi_star_rad <= np.pi + 1e-9


def test_plan_stroke_prefers_taught_family() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=5, n_d=6, n_psi=5, w_sigma=0.5),
    )
    q = np.array(
        [0.774, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    assert abs(nearest_planar_psi(psi_from_q(q))) == pytest.approx(np.pi, abs=1e-6)
    y_c = float(kin.fk_placement(q).translation[1])
    _d, psi_star = rt.plan_stroke(
        q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.005, rail_hi=0.78
    )
    assert not rt.last_psi_family_degraded
    err0 = abs(float(psi_star))
    err_pi = min(abs(float(psi_star) - np.pi), abs(float(psi_star) + np.pi))
    assert err_pi < err0
    assert err_pi < np.deg2rad(45.0)


def test_plan_stroke_degrades_to_opposite_family_when_taught_empty() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3, w_sigma=0.5),
    )
    rt.reset(_SEED_Q)
    rt._planned = True
    rt._psi_star = 0.0
    rt.psi_star_rad = 0.0

    class _OppositeOnly:
        def evaluate(self, pose, psi, branch, y_rail):
            del pose, branch
            a = float(psi)
            a = (a + np.pi) % (2.0 * np.pi) - np.pi
            if abs(a) < 0.5 * np.pi:
                return None
            q_arm = np.array([0.1, -0.4, 0.1, 1.0, 0.1, np.deg2rad(45.0), 0.1])
            q_full = np.concatenate([[float(y_rail)], q_arm])
            return q_arm, q_full, 0.2

    rt._eval = _OppositeOnly()
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    _d, psi_star = rt.plan_stroke(
        _SEED_Q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.025, rail_hi=0.78
    )
    assert rt.last_psi_family_degraded
    err_pi = min(abs(float(psi_star) - np.pi), abs(float(psi_star) + np.pi))
    assert err_pi < np.deg2rad(45.0)


def test_psi_rate_limit_no_lpf() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin, PsiRetargetConfig(enabled=True, psi_rate_rad_s=np.deg2rad(20.0))
    )
    rt.reset(_SEED_Q)
    rt._psi_cmd = 0.0
    rt._psi_star = np.deg2rad(90.0)
    out = rt._rate_limit_psi(0.005)
    assert abs(out) <= np.deg2rad(20.0) * 0.005 + 1e-9
    assert abs(out) > 0.0


def test_wrist_term_prefers_open_j6_when_margin_ties() -> None:
    """J6=0 is the ±128° midpoint; without w_wrist the planner likes it."""
    q_closed = np.zeros(7)
    q_closed[3] = 1.0  # J4 sets the worst margin for both
    q_closed[5] = 0.05
    q_open = q_closed.copy()
    q_open[5] = 0.80
    assert joint_margin_frac(q_closed) == pytest.approx(joint_margin_frac(q_open), abs=1e-9)
    s_old_closed = stroke_score(q_closed, 0.15, w_sigma=0.5, w_wrist=0.0)
    s_old_open = stroke_score(q_open, 0.15, w_sigma=0.5, w_wrist=0.0)
    assert s_old_closed == pytest.approx(s_old_open, abs=1e-9)
    s_new_closed = stroke_score(q_closed, 0.15, w_sigma=0.5, w_wrist=0.5)
    s_new_open = stroke_score(q_open, 0.15, w_sigma=0.5, w_wrist=0.5)
    assert s_new_open > s_new_closed


def test_wrist_band_peaks_near_attractor_not_the_stop() -> None:
    assert wrist_band_frac(0.0) == pytest.approx(0.0)
    assert wrist_band_frac(np.deg2rad(60.0)) == pytest.approx(1.0)
    assert wrist_band_frac(np.deg2rad(60.0)) > wrist_band_frac(np.deg2rad(120.0))


def test_margin_floor_rejects_a_joint_on_the_stop() -> None:
    q = np.zeros(7)
    q[1] = -2.2689  # J2 at the URDF stop
    assert arm_respects_floor(q, np.deg2rad(15.0)) is False
    q[1] = -0.8
    assert arm_respects_floor(q, np.deg2rad(15.0)) is True


def test_step_does_not_replan_when_z_moves() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3),
    )
    rt.reset(_SEED_Q)
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d0, psi0 = rt.plan_stroke(
        _SEED_Q,
        y_center_m=y_c,
        amplitude_m=0.05,
        rail_lo=0.005,
        rail_hi=0.78,
    )
    q2 = _SEED_Q.copy()
    q2[2] -= 0.25
    _psi1, d1 = rt.step(q2, 0.005, rail_lo=0.005, rail_hi=0.78)
    assert d1 == pytest.approx(d0, abs=1e-9)
    assert rt.d_star_m == pytest.approx(d0, abs=1e-9)
    assert rt.psi_star_rad == pytest.approx(psi0, abs=1e-9)


def test_planner_rejects_wrist_on_barrier_floor() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(
            enabled=True, n_y=3, n_d=3, n_psi=3, wrist_min_rad=np.deg2rad(25.0)
        ),
    )
    rt.reset(_SEED_Q)

    class _FloorWrist:
        def evaluate(self, pose, psi, branch, y_rail):
            del pose, psi, branch
            q_arm = np.array([0.1, -0.4, 0.1, 1.0, 0.1, np.deg2rad(15.0), 0.1])
            q_full = np.concatenate([[float(y_rail)], q_arm])
            return q_arm, q_full, 0.2

    rt._eval = _FloorWrist()
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    with pytest.raises(StrokeInfeasibleError, match="no feasible"):
        rt.plan_stroke(
            _SEED_Q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.025, rail_hi=0.78
        )


def test_planner_accepts_open_wrist_cells() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(
            enabled=True, n_y=3, n_d=3, n_psi=3, wrist_min_rad=np.deg2rad(25.0)
        ),
    )
    rt.reset(_SEED_Q)

    class _OpenWrist:
        def evaluate(self, pose, psi, branch, y_rail):
            del pose, psi, branch
            q_arm = np.array([0.1, -0.4, 0.1, 1.0, 0.1, np.deg2rad(45.0), 0.1])
            q_full = np.concatenate([[float(y_rail)], q_arm])
            return q_arm, q_full, 0.2

    rt._eval = _OpenWrist()
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d_star, psi_star = rt.plan_stroke(
        _SEED_Q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.025, rail_hi=0.78
    )
    assert np.isfinite(d_star)
    assert np.isfinite(psi_star)
    assert rt.last_wrist_open_rad >= np.deg2rad(25.0) - 1e-9


def test_ird_d_star_that_hits_the_limit_band_is_discarded() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin, PsiRetargetConfig(enabled=True, n_y=3, n_d=5, n_psi=5)
    )
    rt.reset(_SEED_Q)

    class _BadIrd:
        available = True

        def tcp_ird_from_q(self, _kin, _q):
            return np.eye(4)

        def query_d_star(self, *_a, **_k):
            return 5.0

    rt._ird = _BadIrd()
    y_c = float(kin.fk_placement(_SEED_Q).translation[1])
    d_star, _psi = rt.plan_stroke(
        _SEED_Q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.025, rail_hi=0.78
    )
    assert abs(d_star) < 1.0


def test_split_error_exposes_e_mid_and_fades_k_ff() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=2.0,
            k_esc=0.0,
            k_ff=1.0,
            e0_m=0.0,
            e1_m=0.01,
            v_lpf_tau_s=0.0,
            v_max_m_s=0.08,
            d_star_err0_m=0.01,
            d_star_err1_m=0.04,
            d_star_reg_mult=20.0,
            w_max=2.0,
            w_ext_cap=40.0,
            d_band_m=0.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    task.set_d_pref(0.10)
    y_on = float(q[0]) + 0.10
    vel_ff = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on, vel_ff=vel_ff)
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    assert task.last_k_ff_scale == pytest.approx(1.0)
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert abs(task.last_e_mid_m) < 1e-9

    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.025, vel_ff=vel_ff)
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    assert task.last_k_ff_scale == pytest.approx(1.0)
    assert abs(task.last_v_ff) > 0.04
    assert abs(task.last_e_mid_m) > 0.02
    assert abs(task.last_track_err_m) > 0.02

    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on - 0.08, vel_ff=vel_ff)
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    assert task.last_k_ff_scale == pytest.approx(1.0)
    assert abs(task.last_v_ff) > 0.04
    assert abs(task.last_e_mid_m) > 0.04

    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on - 0.08, vel_ff=None)
    assert task.last_k_ff_scale < 1.0
    assert task.last_v_ff == pytest.approx(0.0, abs=1e-12)


def test_e_mid_deadzone_inside_d_band() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=2.0,
            k_esc=0.0,
            k_ff=0.0,
            e0_m=0.0,
            e1_m=0.01,
            v_lpf_tau_s=0.0,
            v_max_m_s=0.08,
            d_band_m=0.08,
            d_star_err0_m=0.01,
            d_star_err1_m=0.04,
            d_star_reg_mult=20.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    d_live = task.extension(q)
    task.set_d_pref(d_live)
    y_on = float(q[0]) + d_live
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.04, stroke_limiters=False)
    assert task.last_e_mid_m == pytest.approx(0.0, abs=1e-12)
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.16, stroke_limiters=False)
    assert abs(task.last_e_mid_m) > 1e-6
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    assert task.last_k_ff_scale < 1.0


def test_e_mid_alive_with_5mm_d_band() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=2.0,
            k_esc=0.0,
            k_ff=0.0,
            e0_m=0.0,
            e1_m=0.01,
            v_lpf_tau_s=0.0,
            v_max_m_s=0.08,
            d_band_m=0.005,
            d_star_err0_m=0.01,
            d_star_err1_m=0.04,
            d_star_reg_mult=20.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    d_live = task.extension(q)
    task.set_d_pref(d_live)
    y_on = float(q[0]) + d_live
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.008, apply_d_band=True)
    assert abs(task.last_e_mid_m) == pytest.approx(0.003, abs=1e-9)
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.080, apply_d_band=True)
    assert abs(task.last_e_mid_m) == pytest.approx(0.075, abs=1e-9)


def test_rail_ff_tracks_desired_y_minus_d_star() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=5.0,
            k_esc=0.0,
            k_ff=1.0,
            e0_m=0.0,
            e1_m=0.01,
            v_ff_thr_m_s=0.005,
            v_reach_cap_m_s=0.08,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    task.capture_reference(q)
    d_star = float(task.d_pref_m)
    y_tcp_d = float(kin.fk_placement(q).translation[1]) + 0.04
    vel_ff = np.array([0.0, 0.03, 0.0, 0.0, 0.0, 0.0])
    v, _w = task(q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, y_tcp_d=y_tcp_d)
    assert np.isfinite(task.last_rail_ff_m)
    assert task.last_rail_ff_m == pytest.approx(y_tcp_d - d_star, abs=1e-9)
    assert abs(task.last_v_ff) > 0.02
    assert abs(float(v)) < abs(task.last_v_ff)


def test_stroke_limiters_flag_the_limit_band_without_a_reach_velocity() -> None:
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
            v_ff_thr_m_s=0.005,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
            limit_margin_m=0.15,
            escape_leave_m=0.04,
            soft_min_m=0.005,
            soft_max_m=0.78,
            d_star_err0_m=1.0,
            d_band_m=0.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    q[0] = 0.75
    y_des = float(kin.fk_placement(q).translation[1]) + 0.08
    v_open, _ = task(
        q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_des, stroke_limiters=False
    )
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_fade, _ = task(
        q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_des, stroke_limiters=True
    )
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert abs(task.last_e_mid_m) > 0.05
    assert abs(float(v_open)) < 1e-9
    assert abs(float(v_fade)) < 1e-9
    assert task.last_in_limit_band

    q[0] = 0.74
    y_leave = float(kin.fk_placement(q).translation[1]) + 0.08
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_leave_open, _ = task(
        q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_leave, stroke_limiters=False
    )
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_leave_scan, _ = task(
        q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_leave, stroke_limiters=True
    )
    assert abs(task.last_e_mid_m) > 0.05
    assert abs(float(v_leave_open)) < 1e-9
    assert abs(float(v_leave_scan)) < 1e-9
    assert task._in_plus_leave(float(q[0]))


def test_anti_cancel_term_is_gone() -> None:
    """The λ (J_y,arm q̇)² term traded tracking for arm-Y suppression."""
    assert not hasattr(QpConfig(), "anti_cancel_weight")


def _wln_core() -> tuple[QpIkController, SafetyLimits]:
    kin = RobotKinematics()
    a_max = np.full(kin.nv, 3.0)
    a_max[0] = 0.30
    lim = SafetyLimits.from_kinematics(
        kin, v_scale=0.8, a_max=a_max, position_margin=0.0
    )
    lim.q_lower[0], lim.q_upper[0] = 0.005, 0.78
    cfg = QpConfig()
    cfg.collision.enabled = False
    return QpIkController(kin, lim, cfg), lim


def test_barrier_press_cap_closes_on_overforce() -> None:
    from rm75_control.control.admittance_common.force_barrier import (
        ForceBarrierConfig,
        ForceSpaceVelocityDamper,
    )

    cfg = ForceBarrierConfig(enabled=True, v_min_press_m_s=0.0, v_ref_m_s=0.08)
    damper = ForceSpaceVelocityDamper(cfg)
    cap_press, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=20000.0,
        mass_eq_kg=1.0,
        tau_s=0.055,
    )
    assert cap_press <= 1e-9

    cap_small, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.001,
        seek_vz_m_s=0.001,
        contact_enter_n=0.5,
        tau_s=0.055,
    )
    assert cap_small <= 0.001 + 1e-12


def test_barrier_caps_the_free_space_approach() -> None:
    """Impact ~ Ke*v*T_delay, so the gap-closing speed sets the first peak."""
    from rm75_control.control.admittance_common.force_barrier import (
        ForceBarrierConfig,
        ForceSpaceVelocityDamper,
    )

    damper = ForceSpaceVelocityDamper(
        ForceBarrierConfig(enabled=True, v_seek_free_m_s=0.030)
    )
    cap_free, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
    )
    assert cap_free == pytest.approx(0.030)

    # A tighter external sleeve still wins.
    cap_sleeve, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=False,
        v_z_cap=0.08,
        seek_vz_m_s=0.008,
        contact_enter_n=0.5,
    )
    assert cap_sleeve == pytest.approx(0.008)

    # In contact the cap is the admittance's business, not this one.
    cap_contact, _ = damper.caps(
        f_z=0.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
    )
    assert cap_contact > 0.030


def test_admittance_error_is_not_low_passed() -> None:
    """The force-axis slew limiter already bounds command jitter to ~4.9 mm/s
    per tick (measured v_force_z step: 2.8 mm/s p95), so a low-pass on the
    admittance error bought nothing and cost twice — stiff-surface impact
    8 N -> 12.2 N, and proactive v_r 6.97 -> 5.89 mm/s on a receding surface.
    """
    from rm75_control.control.admittance_common.controller import AdmittanceConfig

    cfg = AdmittanceConfig()
    assert not hasattr(cfg, "force_lpf_tau_s")
    assert not hasattr(cfg, "force_lpf_snap_n")


def test_qpik_yaml_keeps_rail_planner_and_baseline_force() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1].joinpath(
            "configs", "joint_admittance_8dof.yaml"
        ).read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    assert cfg.qp.near_arm_margin_rad == pytest.approx(0.08)
    assert cfg.psi_retarget.enabled
    assert cfg.psi_retarget.n_y >= 3
    assert cfg.psi_retarget.w_wrist == pytest.approx(0.5)
    assert cfg.psi_retarget.wrist_min_rad == pytest.approx(np.deg2rad(30.0))
    assert cfg.psi_retarget.margin_floor_rad == pytest.approx(np.deg2rad(20.0))
    qn = np.asarray(raw["inner"]["nullspace"]["q_nominal_deg"], dtype=float)
    assert qn[1] == pytest.approx(-89.5)
    assert qn[4] == pytest.approx(96.0)
    assert qn[5] == pytest.approx(89.3)
    assert qn[6] == pytest.approx(61.0)
    assert cfg.rail.soft_min_m == pytest.approx(0.030)
    assert cfg.rail_extension.d_star_reg_mult == pytest.approx(20.0)
    assert cfg.rail_allocator.u_mid_max_m_s == pytest.approx(0.03)
    assert cfg.rail_allocator.kp_mid == pytest.approx(1.2)
    assert cfg.rail_allocator.k_err_rail == pytest.approx(4.0)
    assert cfg.rail_allocator.e_ref_m == pytest.approx(0.08)
    assert cfg.rail_allocator.f_c_hz == pytest.approx(4.0)
    assert cfg.rail_allocator.leave_exit_eps_m == pytest.approx(0.008)
    assert cfg.rail_allocator.kaw_mid == pytest.approx(8.0)
    assert cfg.rail_allocator.rho_mirror_a == pytest.approx(0.50)
    assert cfg.rail_allocator.rho_mirror_j == pytest.approx(0.30)
    assert cfg.rail_extension.d_band_m == pytest.approx(0.005)
    assert cfg.cartesian_track.k_task_lin == pytest.approx(12.0)
    assert cfg.cartesian_track.fb_lpf_tau_s == pytest.approx(0.0)
    assert cfg.cartesian_track.k_task_rot == pytest.approx(2.0)
    assert cfg.cartesian_track.max_pos_err_m == pytest.approx(0.05)
    assert raw["hybrid_motion"]["kp_pos"][0] == pytest.approx(10.0)
    assert raw["hybrid_motion"]["kp_pos"][2] == pytest.approx(0.0)
    assert cfg.collision.d_safe == pytest.approx(0.01)
    assert cfg.collision.d_activate == pytest.approx(0.04)
    assert raw["hw"]["lw100"]["soft_min_m"] == pytest.approx(0.030)
    assert raw["hw"]["lw100"]["vel_kp"] == pytest.approx(14.0)
    assert raw["hw"]["lw100"]["vel_kd"] == pytest.approx(0.22)
    assert raw["hw"]["lw100"]["target_stale_coast_s"] == pytest.approx(0.35)
    assert cfg.qp.branch_barrier.eps_rad == pytest.approx(0.35)
    assert cfg.qp.branch_barrier.j4_limit_eps_rad == pytest.approx(
        math.radians(5.0), abs=1e-6
    )
    assert cfg.qp.branch_barrier.j4_limit_activate_rad == pytest.approx(
        math.radians(25.0), abs=1e-6
    )
    assert cfg.qp.branch_barrier.j1_overfold_abs_rad == pytest.approx(
        math.radians(140.0), abs=1e-6
    )
    assert cfg.qp.branch_barrier.j1_overfold_activate_rad == pytest.approx(
        math.radians(25.0), abs=1e-6
    )
    assert cfg.qp.branch_barrier.activate_rad == pytest.approx(0.52)
    assert cfg.qp.branch_barrier.box_activate_rad == pytest.approx(0.87)
    assert cfg.qp.branch_barrier.slack_weight == pytest.approx(80.0)
    assert cfg.qp.branch_barrier.dwell_free_s == pytest.approx(0.3)
    assert cfg.qp.branch_barrier.dwell_ramp_s == pytest.approx(1.0)
    assert cfg.qp.branch_barrier.dwell_scale_max == pytest.approx(5.0)
    assert cfg.qp.aniso_task_damping is True
    assert not hasattr(cfg.qp, "wln")
    assert not hasattr(cfg.qp, "sns_retry_scales")
    assert not hasattr(cfg.qp, "twist_scale_lpf_tau_s")
    assert cfg.rail_extension.enabled
    assert cfg.qp.twist_sigma_floor == pytest.approx(0.02)
    assert cfg.qp.task_weight_min_frac == pytest.approx(0.05)
    assert "anti_cancel_weight" not in raw["inner"]["qp"]
    assert cfg.ird.enabled
    assert cfg.ird.device == "cpu"
    assert cfg.qp.joint_comfort.enabled
    assert cfg.qp.near_arm_margin_rad == pytest.approx(0.08)
    assert cfg.qp.joint_comfort.m_comfort_rad == pytest.approx(math.radians(15.0))
    assert cfg.rail_extension.healthy_sigma_mute == pytest.approx(0.08)
    assert cfg.rail_extension.press_v_force_min_m_s == pytest.approx(0.02)
    assert cfg.rail_extension.press_dz_max_m == pytest.approx(0.002)
    assert cfg.rail_extension.press_y_err_m == pytest.approx(0.005)
    assert cfg.rail_extension.press_stall_s == pytest.approx(0.5)
    assert not hasattr(cfg.psi_retarget, "z_replan_m")
    assert cfg.psi_retarget.d_center_rate_m_s == pytest.approx(0.02)
    assert cfg.psi_retarget.psi_cmd_lead_rad == pytest.approx(np.deg2rad(18.0))
    assert cfg.psi_retarget.psi_attr_rad == pytest.approx(np.deg2rad(68.0))
    assert cfg.psi_retarget.d_attr_m == pytest.approx(-0.185)
    assert cfg.psi_retarget.psi_envelope_lo_rad == pytest.approx(np.deg2rad(40.0))
    assert cfg.psi_retarget.psi_envelope_hi_rad == pytest.approx(np.deg2rad(110.0))
    assert cfg.rail_extension.escape_sign_policy == "auto"
    assert not hasattr(cfg.psi_retarget, "d_band_m")
    assert cfg.psi_retarget.psi_replan_period_s == pytest.approx(0.1)
    assert cfg.arm_angle.k_psi == pytest.approx(1.5)
    assert cfg.nullspace.weights[1] == pytest.approx(1.0)
    assert cfg.qp.j_max_arm_rad_s3 == pytest.approx(300.0)
    assert cfg.qp.j_max_rail_m_s3 > 0.0
    # Rail command shaping must not come back: it fed core.qdot_prev and so
    # multiplied every acceleration limit by its own alpha.
    assert not hasattr(cfg.rail, "cmd_lpf_tau_s")
    hm = raw["hybrid_motion"]
    assert hm["proactive_retract_only"] is False
    assert hm["proactive_feedforward"] is True
    assert hm["force_dob"]["enabled"] is True
    assert hm["cdyob"]["mode"] == "off"
    assert float(hm["force_axis_slew_press_m_s2"]) >= 0.8
    assert hm["force_barrier"]["enabled"] is False
    # The blunt phase-2 instruments stay out.
    assert "vel_dob" not in hm
    assert "v_air_m_s" not in hm
    assert "admittance_damping_retract_z" not in hm
    assert "tafac_scan_ff" not in hm
    # BEFM stays fail-closed until befm_calibrate.py passes on hardware.
    assert hm["bidirectional_flow"]["mode"] == "observe"
    assert hm["bidirectional_flow"]["sign_verified"] is False
    assert hm["bidirectional_flow"]["feedback_delay_verified"] is False


def test_analyzer_rejects_empty_scan(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("phase,t_wall_s\n", encoding="utf-8")
    assert mod.analyze(csv_path) == 2
    assert "waste_ratio" in mod.GATES
    assert "accel_reversals_per_s" in mod.GATES
    assert "dt_on_time_frac" in mod.GATES
    assert "step_ripple_p999" in mod.GATES
    assert "deadline_slack_pos_frac" in mod.GATES
    assert mod.GATES["dt_nominal_s"] == pytest.approx(0.005)
    assert mod.PERIOD_LADDER_MS == (7.0, 6.0, 5.0)
    assert mod.next_period_ms(7.0, 0.50) == pytest.approx(7.0)
    assert mod.GATES["rail_min_m"] == pytest.approx(0.02)
    assert mod.GATES["tick_inner_max_ms"] == pytest.approx(20.0)
    assert mod.GATES["track_err_p95_mm"] == pytest.approx(1.0)
    assert mod.GATES["rail_share_p50"] == pytest.approx(0.60)
    assert mod.GATES["psi_err_p95_deg"] == pytest.approx(15.0)
    assert mod.GATES["fa24_write_hz"] == pytest.approx(40.0)
    assert mod.GATES["vpc_track_err_p95_mm"] == pytest.approx(5.0)


def test_analyzer_accepts_ellipse_track_phase(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    n = 40
    cols = ["phase", "t_wall_s", "q_meas_0", "tool_y_err_mm"]
    lines = [",".join(cols)]
    for k in range(n):
        lines.append(
            ",".join(["ellipse_track", f"{0.005 * k:.4f}", "0.20", "0.4"])
        )
    csv_path = tmp_path / "ellipse.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.analyze(csv_path)
    out = buf.getvalue()
    assert rc != 2
    assert "phase=ellipse_track" in out


def test_wbc_log_header_has_rail_cmd_meas_err() -> None:
    from rm75_control.control.joint_admittance_8dof.loop import _TickLogger

    header = _TickLogger._HEADER
    assert "deadline_slack_s" in header
    assert "rail_cmd_meas_err_m" in header
    assert header.count("rail_cmd_meas_err_m") == 1
    assert "rail_posture_err_m" in header
    assert "rail_track_err_m" not in header
    for name in (
        "qpik_nullspace_centering_norm",
        "qpik_nullspace_manip_norm",
        "qpik_nullspace_arm_angle_norm",
        "qpik_nullspace_damping_norm",
        "qpik_nullspace_rail_lock_norm",
        "qpik_sat_scale",
        "qpik_sec_target_norm",
        "post_qp_step_clamp_enabled",
        "post_step_would_clamp",
        "post_step_clamp_applied",
        "dt_nom_s",
        "dt_int_s",
        "box_h1_s",
        "box_h2_s",
        "qpik_qdot_raw_json",
        "qpik_qdot_pre_commit_json",
        "qpik_qdot_committed_json",
        "qpik_qdot_prev_used_json",
        "qpik_qdot_prev2_used_json",
        "qpik_box_lo_json",
        "qpik_box_hi_json",
        "post_step_shadow_q_json",
        "q_cmd_json",
        "arm_send_mono_ns",
        "rail_target_publish_mono_ns",
        "rail_fa24_write_mono_ns",
        "rail_encoder_sample_mono_ns",
        "arm_qdot_target_wall_json",
        "e_shape_norm",
        "e_qp_norm",
        "e_exec_norm",
        "quiescent",
        "secondary_suppressed",
        "command_stale",
        "joint_limited",
        "rail_limited",
        "wall_active",
        "rail_q_hat_m",
        "rail_goal_err_m",
    ):
        assert name in header
    assert len(header) == len(set(header))


def test_analyzer_flags_resync_guard_binding(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    n = 80
    cols = [
        "phase", "t_wall_s", "q_cmd_0", "q_meas_0", "rail_cmd_meas_err_m",
        "tool_y_err_mm",
    ]
    lines = [",".join(cols)]
    for k in range(n):
        lines.append(
            ",".join(
                [
                    "scan",
                    f"{0.005 * k:.4f}",
                    "0.420",
                    "0.400",
                    "0.020",
                    "0.4",
                ]
            )
        )
    csv_path = tmp_path / "bind.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.analyze(csv_path)
    out = buf.getvalue()
    assert rc == 1
    assert "lead clamp" in out
    assert "[FAIL]" in out


def test_analyzer_recovers_known_axis_lead() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    dt = 0.005
    t = np.arange(0.0, 4.0, dt)
    omega = 2.0 * np.pi / 2.0
    tau = -0.19
    ref = np.sin(omega * t)
    meas = np.sin(omega * (t - tau))
    got, resid = mod.best_axis_time_shift(ref, meas, dt)
    assert got == pytest.approx(tau, abs=dt)
    assert resid < 0.02
    err = (ref - meas) * 1000.0
    vel = omega * np.cos(omega * t)
    corr = mod.err_vel_correlation(err, vel)
    assert corr < -0.5


def test_analyzer_wall_uses_tool_y_err_and_reports_posture(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    n = 80
    cols = [
        "phase", "t_wall_s", "q_meas_0", "q_meas_2", "q_meas_4", "q_meas_5",
        "q_meas_6", "q_meas_7", "tool_y_err_mm", "motion_err_rms_mm",
    ]
    lines = [",".join(cols)]
    for k in range(n):
        lines.append(
            ",".join(
                [
                    "scan",
                    f"{0.005 * k:.4f}",
                    "0.030",
                    "0.0",
                    "2.094",  # 120°
                    "-0.262",  # -15°
                    "0.384",  # 22°
                    "0.0",
                    "1.5",
                    "40.0",
                ]
            )
        )
    csv_path = tmp_path / "wall.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.analyze(csv_path)
    out = buf.getvalue()
    assert "tool_y_err p95" in out
    assert "J5 vs nominal" in out
    assert "J4 comfort park" in out


def test_analyzer_jerk_metric_ignores_loop_period_jitter(tmp_path) -> None:
    """A perfectly smooth command must not read as jerk just because the
    scheduler jittered.  Dividing by wall dt inflated the reversal rate 3-4x
    and sent three rounds of tuning after a metric artefact."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(1)
    n = 3000
    cols = ["phase", "t_wall_s"] + [f"q_cmd_{i}" for i in range(8)]
    lines = [",".join(cols)]
    t = 0.0
    for k in range(n):
        # Smooth sinusoid in sample index; wall clock jitters +-20% like the
        # real loop (measured 5.6-6.9 ms against a 5.0 ms budget).
        t += 0.0061 * float(rng.uniform(0.8, 1.2))
        q = 0.3 * np.sin(2.0 * np.pi * 0.2 * k * 0.0061)
        lines.append(
            ",".join(["scan", f"{t:.6f}"] + [f"{q:.9f}"] * 8)
        )
    csv_path = tmp_path / "jitter.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.analyze(csv_path)
    out = buf.getvalue()
    line = next(
        ln for ln in out.splitlines() if "accel sign reversals" in ln
    )
    assert "[PASS]" in line, line


def test_keep_task_weight_skips_sigma_scale() -> None:
    core, _lim = _wln_core()
    q = np.zeros(8)
    q[0] = 0.20
    core.reset()
    core._task_scale_lpf = 0.5
    core._s_lpf = None
    twist = np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    core.step(q, twist, DT, keep_task_weight=True)
    assert core._task_scale_lpf == pytest.approx(0.5)
    assert np.allclose(core.last_s_sigma, 1.0)
    core.step(q, twist, DT, keep_task_weight=False)
    assert not np.allclose(core.last_s_sigma, 1.0)


def test_rail_sat_is_not_workspace_saturation() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = 0.025
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    assert inner.rail_ext_task is not None
    inner._rail_ext_active = False
    inner.rail_ext_task.last_limit_saturated = True
    twist = np.zeros(6)
    twist[1] = 0.03
    step = inner.update(twist, dt=cfg.dt, q_meas=q)
    assert step.physical_saturated is False


def test_rail_soft_min_is_one_way() -> None:
    """Hard 5 mm zeros into-wall; 30 mm is the Faverjon inner edge, not a park."""
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = float(cfg.rail.soft_min_m)
    lo_soft, hi_soft = inner.core.constraints.bounds(q, cfg.dt, qdot_prev=None)
    assert float(lo_soft[0]) < -1.0e-4
    assert float(hi_soft[0]) > 1.0e-4

    q[0] = float(cfg.rail.hard_min_m)
    lo_hard, hi_hard = inner.core.constraints.bounds(q, cfg.dt, qdot_prev=None)
    assert float(lo_hard[0]) == pytest.approx(0.0, abs=1e-9)
    assert float(hi_hard[0]) > 1.0e-4


def test_rail_does_not_pin_in_fade_band_only() -> None:
    """150 mm fade is not a wall: q0=100 mm must stay free for handoff."""
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = 0.10
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    pose_d = kin.fk_pose(q)
    step = inner.update(np.zeros(6), dt=cfg.dt, q_meas=q, pose_d=pose_d)
    assert inner.rail_ext_task is not None
    assert inner.rail_ext_task._rail_in_limit_band(0.10)
    assert not inner.rail_ext_task._rail_end_blocks(0.10, -1.0)
    assert not step.rail_sat


def test_plus_leave_band_does_not_freeze_rail() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = 0.75
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    pose = kin.fk_pose(q)
    away = np.zeros(6)
    away[1] = -0.04
    step = inner.update(
        away,
        dt=cfg.dt,
        q_meas=q,
        pose_d=pose,
        vel_ff=away,
        task_rotation_base=np.eye(3),
    )
    assert inner.rail_ext_task is not None
    assert inner.rail_ext_task._in_plus_leave(0.75)
    assert not step.rail_sat
    assert float(step.qdot[0]) < -1.0e-4


def test_allocation_leave_band_uses_measured_rail() -> None:
    """Command in the plus leave band must not park a mid-stroke encoder."""
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q_meas = _SEED_Q.copy()
    q_meas[0] = 0.40
    inner.reset(q_meas)
    inner.q_cmd[0] = 0.75
    if inner.posture_retarget is not None:
        inner.posture_retarget._planned = True
    toward = np.zeros(6)
    toward[1] = 0.05
    step = inner.update(toward, dt=cfg.dt, q_meas=q_meas, vel_ff=toward)
    assert inner.rail_ext_task is not None
    assert inner.rail_ext_task._in_plus_leave(0.75)
    assert not inner.rail_ext_task._in_plus_leave(0.40)
    assert float(inner.last_v_r_ref) > 1.0e-4 or float(step.qdot[0]) > 1.0e-4

    q_leave = q_meas.copy()
    q_leave[0] = 0.75
    inner.reset(q_leave)
    inner.q_cmd[0] = 0.40
    if inner.posture_retarget is not None:
        inner.posture_retarget._planned = True
    parked = inner.update(toward, dt=cfg.dt, q_meas=q_leave, vel_ff=toward)
    assert inner.rail_ext_task._in_plus_leave(float(q_leave[0]))
    assert float(inner.last_v_r_ref) <= 1.0e-4
    assert float(parked.rail_task_vel) == pytest.approx(0.0, abs=1.0e-4)


def test_retired_inner_keys_are_rejected() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    raw["inner"]["post_qp_step_clamp"] = True
    with pytest.raises(ValueError, match="post_qp_step_clamp"):
        build_joint_ik_config(raw)
    raw["inner"].pop("post_qp_step_clamp")
    raw["inner"]["qp"]["slack_probe"] = True
    with pytest.raises(ValueError, match="slack_probe"):
        build_joint_ik_config(raw)


def test_d_star_nudge_stays_inside_soft_travel() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3),
    )
    rt.reset(_SEED_Q)
    y_des = float(kin.fk_placement(_SEED_Q).translation[1])
    d0 = float(rt.d_star_m)
    d1 = rt.nudge_d_star(-0.01, y_des_m=y_des, rail_lo=0.025, rail_hi=0.78)
    assert d1 != pytest.approx(d0)
    rail_ff = y_des - d1
    assert 0.025 - 1e-9 <= rail_ff <= 0.78 + 1e-9
    d_hi = rt.nudge_d_star(10.0, y_des_m=y_des, rail_lo=0.025, rail_hi=0.78)
    assert y_des - 0.78 - 1e-9 <= d_hi <= y_des - 0.025 + 1e-9


def test_press_stall_nudges_d_star_on_the_controller() -> None:
    from pathlib import Path

    import yaml

    from rm75_control.control.joint_admittance_8dof.config import build_joint_ik_config
    from rm75_control.control.joint_admittance_8dof.loop import JointIkController

    raw = yaml.safe_load(
        Path(__file__).resolve().parents[1]
        .joinpath("configs", "joint_admittance_8dof.yaml")
        .read_text(encoding="utf-8")
    )
    cfg = build_joint_ik_config(raw)
    cfg.backend = "python"
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    inner.reset(q)
    inner.begin_hybrid_episode(q, np.zeros(8))
    assert inner.posture_retarget is not None
    d0 = float(inner.posture_retarget.d_star_m)
    pose_d = kin.fk_pose(q)
    pose_d[1] += 0.02
    z = float(pose_d[2])
    inner._press_z_mark = z
    inner._press_stall_s = 0.6
    inner._d_star_nudge_cool_s = 0.0
    inner.update(
        np.zeros(6),
        dt=cfg.dt,
        q_meas=q,
        pose_d=pose_d,
        v_force_z=0.04,
    )
    assert float(inner.posture_retarget.d_star_m) != pytest.approx(d0)


def _load_analyze_qpik_quality():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_rail_stop_csv(path, *, reverse_frac: float) -> None:
    import csv

    n = 80
    dt = 1.0 / 60.0
    header = [
        "t_wall_s",
        "follow",
        "target_age_ms",
        "a_cmd_m_s2",
        "e_track_mm",
        "v_goal_est_m_s",
        "v_enc_m_s",
        "x_meas_m",
    ]
    entry = 0.030
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        x = 0.40
        for k in range(n):
            t = k * dt
            if k < 40:
                v_goal = entry
                v_enc = entry
            elif k == 40:
                v_goal = 0.0
                v_enc = entry
            elif k < 48:
                # 8 samples (~130 ms) after v_goal→0.
                phase = (k - 41) / 6.0
                v_enc = entry * (1.0 - phase) - entry * reverse_frac * max(
                    0.0, phase
                )
                v_goal = 0.0
            else:
                v_goal = 0.0
                v_enc = 0.0
            x += v_enc * dt
            writer.writerow(
                {
                    "t_wall_s": f"{t:.6f}",
                    "follow": "1",
                    "target_age_ms": "8.0",
                    "a_cmd_m_s2": "0.0",
                    "e_track_mm": "0.05",
                    "v_goal_est_m_s": f"{v_goal:.6f}",
                    "v_enc_m_s": f"{v_enc:.6f}",
                    "x_meas_m": f"{x:.6f}",
                }
            )


def test_analyzer_stop_reverse_gate_fails_on_plugging(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000000.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    servo = tmp_path / "rail_servo" / "rail_20260101_000000.csv"
    _write_rail_stop_csv(servo, reverse_frac=0.55)
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "stop reverse" in r[0])
    assert gate[1] is False
    assert "55" in gate[2] or float(gate[2].split("%")[0]) > 15.0


def test_analyzer_stop_reverse_gate_passes_without_reverse(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000001.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    servo = tmp_path / "rail_servo" / "rail_20260101_000001.csv"
    _write_rail_stop_csv(servo, reverse_frac=0.0)
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "stop reverse" in r[0])
    assert gate[1] is True


def test_analyzer_stop_reverse_gate_survives_a_near_zero_entry_sample(tmp_path) -> None:
    """Entry speed comes from the pre-stop peak, not one backtracked sample.

    Run 225941 printed 2632175% because the sample at the backtrack index
    was a quantisation zero.
    """
    import csv

    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000002.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    servo = tmp_path / "rail_servo" / "rail_20260101_000002.csv"
    servo.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t_wall_s", "follow", "target_age_ms", "a_cmd_m_s2", "e_track_mm",
        "v_goal_est_m_s", "v_enc_m_s", "x_meas_m",
    ]
    dt = 1.0 / 60.0
    with servo.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        x = 0.40
        for k in range(80):
            if k < 40:
                v_goal, v_enc = 0.030, 0.030
            elif k == 40:
                # The one sample the old backtrack landed on.
                v_goal, v_enc = 0.0, 1.0e-7
            elif k < 46:
                v_goal, v_enc = 0.0, -0.002
            else:
                v_goal, v_enc = 0.0, 0.0
            x += v_enc * dt
            writer.writerow(
                {
                    "t_wall_s": f"{k * dt:.6f}", "follow": "1",
                    "target_age_ms": "8.0", "a_cmd_m_s2": "0.0",
                    "e_track_mm": "0.05", "v_goal_est_m_s": f"{v_goal:.9f}",
                    "v_enc_m_s": f"{v_enc:.9f}", "x_meas_m": f"{x:.6f}",
                }
            )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "stop reverse" in r[0])
    # 2 mm/s against a 30 mm/s entry, not a percentage in the millions.
    assert float(gate[2].split("%")[0]) < 20.0
    assert gate[1] is True


def _write_idle_rows(
    *, rail_travel_m: float, tcp_drift_m: float, slack: float, n: int = 400
) -> list[dict]:
    """Drive for 1 s, then hold the stick at zero for 2 s."""
    dt = 1.0 / 200.0
    rows: list[dict] = []
    n_drive = n // 3
    rail = 0.40
    y = 0.19
    for k in range(n):
        idle = k >= n_drive
        if idle:
            phase = min(1.0, (k - n_drive) / max(n - n_drive - 1, 1))
            rail_k = 0.40 + rail_travel_m * phase
            y_k = 0.19 + tcp_drift_m * phase
            slack_k = slack
        else:
            rail_k = rail
            y_k = y
            slack_k = 0.0
        rows.append(
            {
                "t_wall_s": f"{k * dt:.6f}",
                "twist_requested_vx": "0.0",
                "twist_requested_vy": "0.0" if idle else "0.12",
                "twist_requested_vz": "0.0",
                "twist_requested_wx": "0.0",
                "twist_requested_wy": "0.0",
                "twist_requested_wz": "0.0",
                "rail_meas_m": f"{rail_k:.6f}",
                "pose_meas_x": "0.5",
                "pose_meas_y": f"{y_k:.6f}",
                "pose_meas_z": "0.3",
                "slack_norm": f"{slack_k:.6f}",
                # tool_y_err is not the idle-drift gate; pose_meas is.
                "tool_y_err_mm": "0.0",
            }
        )
    return rows


def test_analyzer_idle_gates_fail_on_the_release_slide() -> None:
    mod = _load_analyze_qpik_quality()
    rows = _write_idle_rows(rail_travel_m=0.050, tcp_drift_m=0.022, slack=0.05)
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._idle_hold_checks(rows, results, info)
    by_name = {name: (ok, detail) for name, ok, detail in results}
    assert by_name["idle rail travel p95 < 8 mm"][0] is False
    assert by_name["idle TCP drift p95 < 1 mm (pose_meas latched)"][0] is False
    assert by_name["idle QP1 task slack < 5% of ticks"][0] is False


def test_analyzer_idle_gates_pass_when_the_release_is_clean() -> None:
    mod = _load_analyze_qpik_quality()
    rows = _write_idle_rows(rail_travel_m=0.001, tcp_drift_m=0.0005, slack=0.0)
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._idle_hold_checks(rows, results, info)
    assert results
    assert all(ok for _name, ok, _detail in results)


def _write_debt_rows(track_err_m: float, n: int = 200) -> list[dict]:
    return [
        {
            "rail_qdot_ff": "0.120",
            "rail_track_err_m": f"{track_err_m:.6f}",
        }
        for _ in range(n)
    ]


def test_analyzer_posture_debt_gate_catches_a_starved_reach() -> None:
    mod = _load_analyze_qpik_quality()
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._posture_debt_check(_write_debt_rows(0.094), results, info)
    gate = next(r for r in results if "rail_posture_err" in r[0])
    assert gate[1] is False

    results.clear()
    mod._posture_debt_check(_write_debt_rows(0.010), results, info)
    gate = next(r for r in results if "rail_posture_err" in r[0])
    assert gate[1] is True


def test_analyzer_posture_debt_gate_skips_ticks_without_hard_drive() -> None:
    mod = _load_analyze_qpik_quality()
    rows = [
        {"rail_qdot_ff": "0.001", "rail_track_err_m": "0.200"} for _ in range(200)
    ]
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._posture_debt_check(rows, results, info)
    assert not results
    assert any("posture debt" in name for name, _detail in info)


def _write_rail_source_csv(path, *, reg_frac: float, n: int = 200) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t_wall_s", "follow", "target_age_ms", "a_cmd_m_s2", "e_track_mm",
        "v_goal_est_m_s", "v_enc_m_s", "v_enc_source", "x_meas_m",
    ]
    dt = 1.0 / 60.0
    n_reg = int(round(reg_frac * n))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for k in range(n):
            writer.writerow(
                {
                    "t_wall_s": f"{k * dt:.6f}", "follow": "1",
                    "target_age_ms": "8.0", "a_cmd_m_s2": "0.0",
                    "e_track_mm": "0.05", "v_goal_est_m_s": "0.030",
                    "v_enc_m_s": "0.030",
                    "v_enc_source": "reg" if k < n_reg else "lsq",
                    "x_meas_m": f"{0.40 + 0.030 * k * dt:.6f}",
                }
            )


def test_analyzer_flags_v_enc_falling_back_to_the_drive_register(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000003.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_source_csv(
        tmp_path / "rail_servo" / "rail_20260101_000003.csv", reg_frac=0.112
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "register fallback" in r[0])
    assert gate[1] is False
    assert "reg 11.0%" in gate[2]


def test_analyzer_accepts_v_enc_without_register_fallback(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000004.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_source_csv(
        tmp_path / "rail_servo" / "rail_20260101_000004.csv", reg_frac=0.0
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "register fallback" in r[0])
    assert gate[1] is True


def test_analyzer_stop_reverse_ignores_a_through_zero_command(tmp_path) -> None:
    """A stick that sweeps through zero is a new command, not a brake reverse."""
    import csv

    mod = _load_analyze_qpik_quality()
    t = np.arange(0.0, 1.2, 1.0 / 60.0)
    v_goal = np.where(t < 0.40, -0.084, np.where(t < 0.42, 0.0, 0.135))
    v_enc = np.where(t < 0.40, -0.084, np.where(t < 0.45, -0.010, 0.135))
    frac = mod.rail_stop_reverse_frac(t, v_goal, v_enc)
    assert not np.isfinite(frac) or frac < 0.15

    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000005.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    servo = tmp_path / "rail_servo" / "rail_20260101_000005.csv"
    servo.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t_wall_s", "follow", "target_age_ms", "a_cmd_m_s2", "e_track_mm",
        "v_goal_est_m_s", "v_enc_m_s", "x_meas_m",
    ]
    with servo.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        x = 0.40
        dt = 1.0 / 60.0
        for k, (vg, ve) in enumerate(zip(v_goal, v_enc)):
            x += ve * dt
            writer.writerow(
                {
                    "t_wall_s": f"{k * dt:.6f}", "follow": "1",
                    "target_age_ms": "8.0", "a_cmd_m_s2": "0.0",
                    "e_track_mm": "0.05", "v_goal_est_m_s": f"{vg:.6f}",
                    "v_enc_m_s": f"{ve:.6f}", "x_meas_m": f"{x:.6f}",
                }
            )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    stop = [r for r in results if "stop reverse" in r[0]]
    assert not stop or stop[0][1] is True


def _write_rail_shape_csv(path, *, e_shape_mm: float, v_enc=0.03, a_cmd=0.10, n: int = 80) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t_wall_s", "follow", "target_age_ms", "a_cmd_m_s2", "e_track_mm",
        "e_shape_mm", "v_goal_est_m_s", "v_enc_m_s", "x_meas_m",
    ]
    dt = 1.0 / 60.0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for k in range(n):
            writer.writerow(
                {
                    "t_wall_s": f"{k * dt:.6f}", "follow": "1",
                    "target_age_ms": "8.0", "a_cmd_m_s2": f"{a_cmd:.6f}",
                    "e_track_mm": "0.05", "e_shape_mm": f"{e_shape_mm:.4f}",
                    "v_goal_est_m_s": f"{v_enc:.6f}", "v_enc_m_s": f"{v_enc:.6f}",
                    "x_meas_m": f"{0.40 + v_enc * k * dt:.6f}",
                }
            )


def test_analyzer_eshape_gate_fails_on_open_loop_drift(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000006.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_shape_csv(
        tmp_path / "rail_servo" / "rail_20260101_000006.csv", e_shape_mm=15.93
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "e_shape" in r[0])
    assert gate[1] is False


def test_analyzer_eshape_gate_passes_when_the_loop_is_closed(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000007.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_shape_csv(
        tmp_path / "rail_servo" / "rail_20260101_000007.csv", e_shape_mm=0.40
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    gate = next(r for r in results if "e_shape" in r[0])
    assert gate[1] is True


def test_analyzer_box_gates_fail_when_servo_outruns_qp(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000008.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_shape_csv(
        tmp_path / "rail_servo" / "rail_20260101_000008.csv",
        e_shape_mm=0.2,
        v_enc=0.154,
        a_cmd=1.06,
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    v_gate = next(r for r in results if "|v_enc| over QP box" in r[0])
    a_gate = next(r for r in results if "|a_cmd| over QP box" in r[0])
    assert v_gate[1] is False
    assert a_gate[1] is False


def test_analyzer_box_gates_pass_inside_the_qp_box(tmp_path) -> None:
    mod = _load_analyze_qpik_quality()
    scan = tmp_path / "gamepad_vcmd" / "run_20260101_000009.csv"
    scan.parent.mkdir(parents=True)
    scan.write_text("phase,t_wall_s\nscan,0.0\n", encoding="utf-8")
    _write_rail_shape_csv(
        tmp_path / "rail_servo" / "rail_20260101_000009.csv",
        e_shape_mm=0.2,
        v_enc=0.10,
        a_cmd=0.40,
    )
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_servo_checks(scan, results, info)
    v_gate = next(r for r in results if "|v_enc| over QP box" in r[0])
    a_gate = next(r for r in results if "|a_cmd| over QP box" in r[0])
    assert v_gate[1] is True
    assert a_gate[1] is True


def test_analyzer_rail_task_dropout_fails_when_ff_is_silenced() -> None:
    mod = _load_analyze_qpik_quality()
    rows = [
        {
            "t_wall_s": f"{k * 0.006:.4f}",
            "rail_task_vel": "",
            "v_ff_rail": "0.0023",
        }
        for k in range(200)
    ]
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_task_dropout_check(rows, results, info)
    gate = next(r for r in results if "dropout" in r[0])
    assert gate[1] is False
    assert "longest" in gate[2]


def test_analyzer_rail_task_dropout_passes_when_vel_is_issued() -> None:
    mod = _load_analyze_qpik_quality()
    rows = [
        {
            "t_wall_s": f"{k * 0.006:.4f}",
            "rail_task_vel": "0.0023",
            "v_ff_rail": "0.0023",
        }
        for k in range(200)
    ]
    results: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []
    mod._rail_task_dropout_check(rows, results, info)
    gate = next(r for r in results if "dropout" in r[0])
    assert gate[1] is True


def test_vpc_midrange_gates_pass_on_symmetric_synthetic() -> None:
    mod = _load_analyze_qpik_quality()
    rows = []
    for k in range(80):
        t = k * 0.005
        plus = k < 40
        vy = 0.05 if plus else -0.05
        rows.append(
            {
                "t_wall_s": f"{t:.4f}",
                "rail_motion_share": "0.70",
                "v_cmd_vy": f"{vy:.4f}",
                "v_r_ref": f"{0.04 if plus else -0.04:.4f}",
                "psi_deg": "68.0",
                "psi_ref_deg": "68.0",
                "track_err_mm": "1.5",
            }
        )
    results: list[tuple[str, bool, str]] = []
    mod._vpc_midrange_checks(rows, results)
    by_name = {r[0]: r for r in results}
    assert by_name["rail_motion_share p50 ≥ 0.60 (|v_cmd_vy| > 20 mm/s)"][1]
    assert by_name["+Y/−Y rail share ratio ≤ 1.25"][1]
    assert by_name["|ψ − ψ_ref| p95 ≤ 15°"][1]
    assert by_name["sign(v_r_ref)==sign(v_cmd_vy) ≥ 85% (|vy|>10 mm/s)"][1]
    assert by_name["track_err p95 ≤ 5 mm"][1]


def test_fa24_vpc_gates_pass_on_dense_writes() -> None:
    mod = _load_analyze_qpik_quality()
    rows = []
    rpm = 200
    for k in range(80):
        rpm += 4
        rows.append(
            {
                "t_wall_s": f"{k / 50.0:.4f}",
                "rpm_cmd": str(rpm),
                "t_write_ms": "3.0",
            }
        )
    results: list[tuple[str, bool, str]] = []
    mod._rail_servo_vpc_checks(rows, results)
    by_name = {r[0]: r for r in results}
    assert by_name["FA24 write ≥ 40 Hz (active window)"][1]
    assert by_name["FA24 |Δrpm| p95 ≤ 20"][1]

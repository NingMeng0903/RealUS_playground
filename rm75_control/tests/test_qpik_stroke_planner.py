"""One-shot (d*, ψ*) stroke planner + rail feedforward.

The force-side phase-2 work was reverted to 55e261d after hardware showed a
contact-loss / impact regression; only the rail-side planner survived, so this
file covers just that.
"""

from __future__ import annotations

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
        PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3, z_replan_m=0.03),
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


def test_split_error_raises_rail_reg_and_fades_k_ff() -> None:
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
            v_reach_cap_m_s=0.04,
            v_max_m_s=0.08,
            d_star_err0_m=0.01,
            d_star_err1_m=0.04,
            d_star_reg_mult=20.0,
            w_max=2.0,
            w_ext_cap=40.0,
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
    assert abs(task.last_v_reach) < 1e-9

    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.025, vel_ff=vel_ff)
    assert task.last_d_star_reg_scale > 1.0
    assert task.last_k_ff_scale < 1.0
    assert abs(task.last_v_ff) > 1e-4
    assert abs(task.last_track_err_m) > 0.02

    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on - 0.08, vel_ff=vel_ff)
    assert task.last_d_star_reg_scale > 1.0
    assert task.last_k_ff_scale < 1.0


def test_v_reach_deadzone_inside_d_band() -> None:
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
            v_reach_cap_m_s=0.02,
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
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert task.last_d_star_reg_scale == pytest.approx(1.0)
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    task(q, sigma_scale=1.0, dt_s=DT, y_tcp_d=y_on + 0.16, stroke_limiters=False)
    assert abs(task.last_v_reach) > 1e-6
    assert task.last_d_star_reg_scale > 1.0
    assert task.last_k_ff_scale < 1.0


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
    assert v * 0.03 > 0.0


def test_stroke_limiters_only_when_planned() -> None:
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
            v_ff_thr_m_s=0.005,
            v_reach_cap_m_s=0.08,
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
            limit_margin_m=0.15,
            escape_leave_m=0.04,
            soft_min_m=0.005,
            soft_max_m=0.78,
            d_star_err0_m=1.0,
        ),
    )
    task.set_mode("reach")
    q = _SEED_Q.copy()
    q[0] = 0.70
    task.set_d_pref(float(kin.fk_placement(q).translation[1]) - float(q[0]))
    vel_ff = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    v_open, _ = task(
        q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, stroke_limiters=False
    )
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_fade, _ = task(
        q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, stroke_limiters=True
    )
    assert v_open > 0.05
    assert v_fade < v_open
    assert v_fade < 0.055
    assert not task.last_limit_saturated or v_fade < 0.02

    q[0] = 0.74
    task.set_d_pref(float(kin.fk_placement(q).translation[1]) - float(q[0]))
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_leave_open, _ = task(
        q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, stroke_limiters=False
    )
    task._v_lpf = 0.0
    task._v_lpf_initialized = False
    v_leave_scan, _ = task(
        q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=DT, stroke_limiters=True
    )
    assert v_leave_open > 0.04
    assert v_leave_scan == pytest.approx(0.0, abs=1e-4)


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


def test_wln_prices_the_rail_only_toward_its_stop() -> None:
    """band_rail_m=0: rail WLN stays 1 toward a stop, away, and mid-stroke."""
    core, lim = _wln_core()
    nv = core.kin.nv
    q_mid = 0.5 * (lim.q_lower + lim.q_upper)

    q_near = q_mid.copy()
    q_near[0] = lim.q_lower[0] + 0.04
    toward = np.zeros(nv)
    toward[0] = -0.05
    assert core._wln_reg_scale(q_near, toward)[0] == pytest.approx(1.0)

    away = np.zeros(nv)
    away[0] = +0.05
    core._wln_scale_prev[:] = 1.0
    assert core._wln_reg_scale(q_near, away)[0] == pytest.approx(1.0)

    core._wln_scale_prev[:] = 1.0
    assert core._wln_reg_scale(q_mid, toward)[0] == pytest.approx(1.0)


def test_wln_scale_does_not_jump_on_reverse() -> None:
    """Rail WLN is off, so a sign flip is a no-op on rail reg."""
    core, lim = _wln_core()
    nv = core.kin.nv
    q_mid = 0.5 * (lim.q_lower + lim.q_upper)
    q_near = q_mid.copy()
    q_near[0] = lim.q_lower[0] + 0.04
    toward = np.zeros(nv)
    toward[0] = -0.05
    away = np.zeros(nv)
    away[0] = +0.05
    core._wln_scale_prev[:] = 1.0
    s0 = float(core._wln_reg_scale(q_near, toward)[0])
    s1 = float(core._wln_reg_scale(q_near, away)[0])
    assert s0 == pytest.approx(1.0)
    assert s1 == pytest.approx(1.0)


def test_wln_never_touches_arm_joints() -> None:
    """J4 sits inside any useful band most of a scan; pricing it buys slack."""
    core, lim = _wln_core()
    nv = core.kin.nv
    for joint in range(1, nv):
        q = 0.5 * (lim.q_lower + lim.q_upper)
        q[joint] = lim.q_upper[joint] - 0.002  # 0.1 deg off the stop
        qdot = np.zeros(nv)
        qdot[joint] = +0.5  # driving straight at it
        assert core._wln_reg_scale(q, qdot)[joint] == pytest.approx(1.0)


def test_barrier_press_cap_has_a_floor_in_contact() -> None:
    from rm75_control.control.admittance_common.force_barrier import (
        ForceBarrierConfig,
        ForceSpaceVelocityDamper,
    )

    cfg = ForceBarrierConfig(enabled=True, v_min_press_m_s=0.003, v_ref_m_s=0.08)
    damper = ForceSpaceVelocityDamper(cfg)
    # Massively over force: prediction and stiffness terms both collapse.
    cap_press, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.08,
        seek_vz_m_s=0.08,
        contact_enter_n=0.5,
        ke_est_n_m=20000.0,
        mass_eq_kg=1.0,
    )
    assert cap_press >= 0.003 - 1e-12

    # A tighter v_z_cap (recontact sleeve) still wins over the floor.
    cap_small, _ = damper.caps(
        f_z=40.0,
        f_des_z=3.0,
        in_contact=True,
        v_z_cap=0.001,
        seek_vz_m_s=0.001,
        contact_enter_n=0.5,
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
    assert cfg.qp.twist_scale_lpf_tau_s == pytest.approx(0.08)
    assert cfg.psi_retarget.enabled
    assert cfg.psi_retarget.n_y >= 3
    assert cfg.psi_retarget.w_wrist == pytest.approx(0.5)
    assert cfg.psi_retarget.wrist_min_rad == pytest.approx(np.deg2rad(30.0))
    assert cfg.psi_retarget.margin_floor_rad == pytest.approx(np.deg2rad(20.0))
    qn = np.asarray(raw["inner"]["nullspace"]["q_nominal_deg"], dtype=float)
    assert qn[1] == pytest.approx(-90.4)
    assert qn[4] == pytest.approx(104.4)
    assert qn[5] == pytest.approx(94.5)
    assert qn[6] == pytest.approx(60.3)
    assert cfg.rail.soft_min_m == pytest.approx(0.030)
    assert cfg.rail_extension.d_star_reg_mult == pytest.approx(20.0)
    assert cfg.rail_extension.v_reach_cap_m_s == pytest.approx(0.02)
    assert cfg.rail_extension.d_band_m == pytest.approx(0.08)
    assert cfg.collision.d_safe == pytest.approx(0.01)
    assert cfg.collision.d_activate == pytest.approx(0.04)
    assert raw["hw"]["lw100"]["soft_min_m"] == pytest.approx(0.030)
    assert raw["hw"]["lw100"]["vel_kp"] == pytest.approx(14.0)
    assert raw["hw"]["lw100"]["vel_kd"] == pytest.approx(0.22)
    assert raw["hw"]["lw100"]["target_stale_coast_s"] == pytest.approx(0.35)
    assert cfg.qp.branch_barrier.eps_rad == pytest.approx(0.35)
    assert cfg.qp.branch_barrier.activate_rad == pytest.approx(0.52)
    assert cfg.qp.branch_barrier.slack_weight == pytest.approx(80.0)
    assert cfg.qp.branch_barrier.dwell_free_s == pytest.approx(0.3)
    assert cfg.qp.branch_barrier.dwell_ramp_s == pytest.approx(1.0)
    assert cfg.qp.branch_barrier.dwell_scale_max == pytest.approx(5.0)
    assert cfg.qp.aniso_task_damping is True
    assert cfg.qp.wln.max_delta == pytest.approx(3.0)
    assert cfg.rail_extension.enabled
    assert cfg.qp.twist_sigma_floor == pytest.approx(0.02)
    assert cfg.qp.task_weight_min_frac == pytest.approx(0.05)
    assert "anti_cancel_weight" not in raw["inner"]["qp"]
    # Rail/jerk work: limit handoff on the rail only, third-order box.
    assert cfg.qp.wln.enabled
    assert cfg.qp.wln.band_rail_m == pytest.approx(0.0)
    # The arm has no spare joint to hand the stroke to; weighting it only
    # bought slack, so its band must stay disabled.
    assert cfg.qp.wln.band_rad == 0.0
    assert cfg.ird.enabled
    assert cfg.ird.device == "cpu"
    assert cfg.qp.joint_comfort.enabled
    assert cfg.psi_retarget.z_replan_m == pytest.approx(0.0)
    assert cfg.psi_retarget.d_center_rate_m_s == pytest.approx(0.02)
    assert cfg.psi_retarget.psi_attr_rad == pytest.approx(np.deg2rad(70.0))
    assert cfg.psi_retarget.d_attr_m == pytest.approx(-0.22)
    assert cfg.psi_retarget.psi_envelope_lo_rad == pytest.approx(np.deg2rad(40.0))
    assert cfg.psi_retarget.psi_envelope_hi_rad == pytest.approx(np.deg2rad(110.0))
    assert cfg.rail_extension.escape_sign_policy == "minus"
    assert cfg.psi_retarget.d_band_m == pytest.approx(0.08)
    assert cfg.psi_retarget.psi_replan_period_s == pytest.approx(0.1)
    assert cfg.arm_angle.k_psi == pytest.approx(1.5)
    assert cfg.nullspace.weights[1] == pytest.approx(1.0)
    assert cfg.qp.j_max_arm_rad_s3 == pytest.approx(300.0)
    assert cfg.qp.j_max_rail_m_s3 > 0.0
    # Rail command shaping must not come back: it fed core.qdot_prev and so
    # multiplied every acceleration limit by its own alpha.
    assert not hasattr(cfg.rail, "cmd_lpf_tau_s")
    # e85c9ab press speed restored; the barrier stays as the brake.
    hm = raw["hybrid_motion"]
    assert hm["proactive_retract_only"] is False
    assert hm["force_dob"]["enabled"] is True
    assert float(hm["force_axis_slew_press_m_s2"]) >= 0.8
    assert hm["force_barrier"]["enabled"] is True
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
    csv_path.write_text("phase,t_wall_s\nmove,0\n", encoding="utf-8")
    assert mod.analyze(csv_path) == 2
    assert "waste_ratio" in mod.GATES
    assert "accel_reversals_per_s" in mod.GATES
    assert "dt_on_time_frac" in mod.GATES
    assert mod.GATES["rail_min_m"] == pytest.approx(0.02)
    assert mod.GATES["tick_inner_max_ms"] == pytest.approx(20.0)


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


def test_wln_still_prices_rail_on_the_stop() -> None:
    """band_rail_m=0: sitting on the stop does not raise rail WLN."""
    core, lim = _wln_core()
    q = 0.5 * (lim.q_lower + lim.q_upper)
    q[0] = lim.q_lower[0]
    toward = np.zeros(core.kin.nv)
    toward[0] = -0.05
    assert float(core._wln_reg_scale(q, toward)[0]) == pytest.approx(1.0)


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
    """30 mm is the full-speed edge; one-way into-wall block is at hard 5 mm."""
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
    cfg.collision.enabled = False
    cfg.qp.collision.enabled = False
    cfg.ird.enabled = False
    cfg.qp.joint_comfort.enabled = False
    kin = RobotKinematics()
    inner = JointIkController(kin, cfg)
    q = _SEED_Q.copy()
    q[0] = float(cfg.rail.soft_min_m)
    lo_soft, hi_soft = inner.core.constraints.bounds(q, cfg.dt, qdot_prev=None)
    assert float(lo_soft[0]) < 0.0
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
    assert not inner.rail_ext_task._rail_in_pin_band(0.10)
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
        away, dt=cfg.dt, q_meas=q, pose_d=pose, task_rotation_base=np.eye(3)
    )
    assert inner.rail_ext_task is not None
    assert inner.rail_ext_task._in_plus_leave(0.75)
    assert not step.rail_sat
    assert float(step.qdot[0]) < -1.0e-4


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

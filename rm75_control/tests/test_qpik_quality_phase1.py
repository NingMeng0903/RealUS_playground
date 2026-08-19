"""Phase-1 QPIK quality: ψ retarget, rail soft-limit fade, capped reach, uniform scale."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import (
    hold_setpoint_from_vel_ff,
    scale_qdot_into_box,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
    SecondaryComposer,
)
from rm75_control.control.joint_admittance_8dof.solver.branch_barrier import (
    BranchBarrierBuilder,
    BranchBarrierConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import (
    QpConfig,
    QpIkController,
)
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    clamp_psi_to_envelope,
    d_from_q,
    fold_psi_to_positive,
    nearest_planar_psi,
    psi_err_avoiding_zero,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.kinematics.srs_ik import psi_from_q
from rm75_control.control.joint_admittance_8dof.tasks.rail_allocator import (
    MidrangingController,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)


def test_limit_saturation_uses_soft_band_not_urdf() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(
            enabled=True,
            k_ext=0.0,
            k_esc=0.0,
            v_lpf_tau_s=0.0,
            limit_margin_m=0.10,
            soft_min_m=0.10,
            soft_max_m=0.70,
        ),
    )
    # 5 cm past the *soft* max but still inside URDF 0.8 → must already be faded.
    scale = task._limit_saturation(0.70, v=0.05)
    assert scale == 0.0
    assert task.last_limit_saturated
    scale_mid = task._limit_saturation(0.40, v=0.05)
    assert scale_mid == 1.0


def test_ff_owns_still_exposes_e_mid() -> None:
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
            v_max_m_s=0.08,
            v_lpf_tau_s=0.0,
            d_star_err0_m=1.0,
            d_band_m=0.0,
        ),
    )
    task.set_mode("reach")
    q = 0.5 * (kin.q_lower + kin.q_upper)
    task.capture_reference(q)
    task.d_pref_m = task.extension(q) + 0.30
    j_rail = kin.jacobian(q)[:3, 0]
    n = float(np.linalg.norm(j_rail))
    vel_ff = np.zeros(6)
    vel_ff[:3] = 0.05 * (j_rail / n)
    task(q, sigma_scale=1.0, vel_ff=vel_ff, dt_s=0.005)
    assert abs(task.last_e_mid_m) > 0.20
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    assert abs(task.last_v_ff) > 1e-4


def test_midranging_owns_idle_posture_not_reach_cap() -> None:
    mid = MidrangingController(kp=1.2, ki=0.0, v_max=0.12)
    u_idle = mid.step(0.008, 0.005)
    mid.reset()
    u_drive = mid.step(0.10, 0.005)
    assert abs(u_idle) < abs(u_drive)
    assert abs(u_drive) <= 0.12 + 1e-9


def test_scale_qdot_into_box_preserves_direction() -> None:
    qdot = np.array([0.2, 0.4, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    lo = np.full(8, -1.0)
    hi = np.full(8, 1.0)
    hi[1] = 0.1
    out = scale_qdot_into_box(qdot, lo, hi)
    assert out[1] <= 0.1 + 1e-12
    assert np.sign(out[0]) == np.sign(qdot[0])
    assert np.sign(out[2]) == np.sign(qdot[2])
    ratio = out[0] / qdot[0]
    assert np.isclose(out[2] / qdot[2], ratio, atol=1e-9)
    clipped = np.clip(qdot, lo, hi)
    assert abs(out[0] - clipped[0]) > 1e-6


def test_psi_rate_limit_caps_step() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(
            enabled=True,
            psi_rate_rad_s=np.deg2rad(20.0),
        ),
    )
    q = 0.5 * (kin.q_lower + kin.q_upper)
    rt.reset(q)
    start = 0.0
    rt._psi_cmd = start
    rt._psi_star = np.deg2rad(90.0)
    out = rt._rate_limit_psi(0.005)
    assert abs(out - start) <= np.deg2rad(20.0) * 0.005 + 1e-9
    assert abs(out - start) > 0.0


def test_planned_step_does_not_climb_d_star() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True, n_y=3, n_d=3, n_psi=3))
    q = np.array([0.375, 0.194, -0.503, -0.069, 1.979, -0.776, 0.547, -4.370])
    rt.reset(q)
    y_c = float(kin.fk_placement(q).translation[1])
    d0, _psi0 = rt.plan_stroke(
        q, y_center_m=y_c, amplitude_m=0.04, rail_lo=0.05, rail_hi=0.75
    )
    for _ in range(20):
        _psi, d = rt.step(q, 0.005, rail_lo=0.05, rail_hi=0.75)
    assert d == pytest.approx(d0)


def test_nearest_planar_psi_snaps_to_sew_planes() -> None:
    assert nearest_planar_psi(0.0) == pytest.approx(0.0)
    assert nearest_planar_psi(np.deg2rad(10.0)) == pytest.approx(0.0)
    assert nearest_planar_psi(np.deg2rad(96.0)) == pytest.approx(np.pi)
    assert nearest_planar_psi(np.pi) == pytest.approx(np.pi)
    assert nearest_planar_psi(-np.pi) == pytest.approx(np.pi)
    assert nearest_planar_psi(np.deg2rad(-170.0)) == pytest.approx(np.pi)


def test_unplanned_step_holds_taught_plane_not_q_nominal() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.array(
        [0.774, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    q_star = np.array([0.0, 0.0, -0.785, 0.0, 1.571, 0.698, 0.785, 0.0])
    rt.reset(q)
    d_live = d_from_q(kin, q)
    psi_taught = nearest_planar_psi(psi_from_q(q))
    assert psi_taught == pytest.approx(np.pi, abs=1e-6)
    assert abs(psi_from_q(q_star)) < 0.05
    last_d = float("nan")
    last_psi = float("nan")
    for _ in range(8):
        last_psi, last_d = rt.step(
            q, 0.005, rail_lo=0.005, rail_hi=0.78, q_nominal=q_star
        )
    d_yaml = d_from_q(kin, q_star)
    assert abs(last_d - d_yaml) > 1.0e-3
    assert last_d == pytest.approx(d_live, abs=0.08)
    assert abs(last_d - float(rt.cfg.d_attr_m)) > 0.05
    assert rt.psi_star_rad == pytest.approx(float(rt.cfg.psi_attr_rad), abs=1e-6)
    assert min(abs(last_psi - np.pi), abs(last_psi + np.pi)) < np.deg2rad(2.0)
    assert last_psi > 0.0
    assert not rt.planned


def test_hold_setpoint_freezes_d_star() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.array(
        [0.50, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    d_live = d_from_q(kin, q)
    rt._d_star = d_live
    rt.d_star_m = d_live
    rt._d_center_target = float(rt.cfg.d_attr_m)
    for _ in range(20):
        _psi, d = rt.step(
            q, 0.02, rail_lo=0.005, rail_hi=0.78, hold_setpoint=True
        )
    assert d == pytest.approx(d_live, abs=1e-9)


def test_hold_release_does_not_jump_d_star(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window 8: s=1 and a 132 mm d_goal step must still rate-limit Δd*."""
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, d_center_rate_m_s=0.02)
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.40, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    d0 = -0.185
    d_goal = -0.317
    rt._d_star = d0
    rt.d_star_m = d0
    rt._d0 = d0
    rt._s = 1.0
    rt.homotopy_s = 1.0
    rt._held_prev = True
    rt._planned = False
    monkeypatch.setattr(rt, "_select_d_for_elbow", lambda *a, **k: d_goal)
    monkeypatch.setattr(
        rt, "_eval_at_split", lambda pose, psi, d: (q[1:], q.copy(), 0.2)
    )
    monkeypatch.setattr(rt, "_q_star_acceptable", lambda *a, **k: True)
    monkeypatch.setattr(rt, "_maybe_retarget_psi", lambda *a, **k: None)
    monkeypatch.setattr(rt, "_clip_d_to_travel", lambda d_try, **k: float(d_try))
    dt = 0.005
    _psi, d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78, hold_setpoint=False)
    assert abs(d - d0) <= cfg.d_center_rate_m_s * dt + 1e-9
    assert abs(d - d_goal) > 0.10


def test_hold_setpoint_includes_rotation() -> None:
    assert hold_setpoint_from_vel_ff(np.zeros(6), lin_thr_m_s=0.01) is False
    assert hold_setpoint_from_vel_ff(
        np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), lin_thr_m_s=0.01
    )
    assert hold_setpoint_from_vel_ff(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.2]), lin_thr_m_s=0.01
    )
    assert hold_setpoint_from_vel_ff(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.01]), lin_thr_m_s=0.01
    ) is False


def test_unplanned_infeasible_rail_keeps_split_and_envelope() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True, rail_margin_m=0.02))
    q = np.array(
        [0.40, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    psi0 = float(rt.psi_star_rad)
    q0 = float(q[0])
    last_d = float("nan")
    for _ in range(12):
        _psi, last_d = rt.step(q, 0.005, rail_lo=q0 - 0.005, rail_hi=q0 + 0.005)
    assert last_d == pytest.approx(d_from_q(kin, q), abs=1e-9)
    assert rt.psi_star_rad == pytest.approx(psi0, abs=1e-9)


def test_search_psi_at_collapsed_wrist_opens_j6() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True, psi_replan_period_s=0.0))
    # Logged 171944 t=56.7: J6=18.9° on the locked 180° plane.
    q = np.array(
        [0.360018, 2.534646, -0.341951, -2.812693, 2.084567, 2.844237, 0.329491, -1.621615]
    )
    rt.reset(q)
    found = rt.search_psi_at_pose(q, rail_lo=0.005, rail_hi=0.78)
    assert found is not None
    assert found == pytest.approx(
        clamp_psi_to_envelope(found, rt.cfg.psi_envelope_lo_rad, rt.cfg.psi_envelope_hi_rad),
        abs=1e-9,
    )
    assert 0.0 < found < np.pi
    assert rt.last_search_j6_rad >= np.deg2rad(45.0)


def test_reset_folds_negative_pi_and_keeps_attr() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, psi_rate_rad_s=np.deg2rad(25.0))
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.396, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    assert rt._psi_cmd == pytest.approx(fold_psi_to_positive(psi_from_q(q)), abs=1e-9)
    assert rt._psi_cmd >= 0.0
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)
    assert rt.d_star_m == pytest.approx(d_from_q(kin, q), abs=1e-6)
    rt._psi_cmd = -np.pi
    dt = 0.02
    prev = float("nan")
    for _ in range(50):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert psi >= -1.0e-9
        assert psi <= np.pi + 1.0e-9
        if np.isfinite(prev):
            assert psi <= prev + 1.0e-9
        prev = psi
    assert rt.psi_star_rad == pytest.approx(float(cfg.psi_attr_rad), abs=1e-9)
    assert prev < np.deg2rad(175.0)


def test_design_pose_releases_rail_split() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.deg2rad([0.0, -89.5, -94.5, 65.2, 96.0, 89.3, 61.0, 94.6])
    q[0] = 0.40
    rt.reset(q)
    assert rt.d_star_m == pytest.approx(d_from_q(kin, q), abs=1e-6)
    _psi, d = rt.step(q, 0.005, rail_lo=0.005, rail_hi=0.78)
    assert d == pytest.approx(float(rt.cfg.d_attr_m), abs=0.08)


def test_unplanned_d_and_psi_move_together() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, d_center_rate_m_s=0.02)
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    d0 = float(rt.d_star_m)
    psi0 = float(rt._psi_cmd)
    psi, d = rt.step(q, 0.02, rail_lo=0.005, rail_hi=0.78)
    assert abs(d - d0) > 1.0e-6
    assert abs(psi_err_avoiding_zero(psi0, psi)) > 1.0e-6
    assert abs(d - d0) <= cfg.d_center_rate_m_s * 0.02 + 1e-9


def test_q_star_does_not_latch_yaml_at_reset() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, d_center_rate_m_s=0.02)
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    assert rt.q_star_rad is not None
    assert abs(float(rt.q_star_rad[1])) < np.deg2rad(5.0)
    yaml_j1 = np.deg2rad(-89.5)
    j1_hist = [float(rt.q_star_rad[1])]
    d_hist = [float(rt.d_star_m)]
    for _ in range(80):
        rt.step(q, 0.05, rail_lo=0.005, rail_hi=0.78)
        assert rt.q_star_rad is not None
        j1_hist.append(float(rt.q_star_rad[1]))
        d_hist.append(float(rt.d_star_m))
    assert abs(j1_hist[4] - yaml_j1) > np.deg2rad(20.0)
    assert j1_hist[-1] < j1_hist[0] - np.deg2rad(5.0)
    assert d_hist[-1] > d_hist[0] + 1.0e-4
    assert abs(d_hist[-1] - float(cfg.d_attr_m)) < abs(d_hist[0] - float(cfg.d_attr_m))


def test_homotopy_does_not_freeze_when_live_j1_leads() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True, d_center_rate_m_s=0.02))
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    real_eval = rt._eval.evaluate

    def _ik_j1_slightly_positive(pose, psi, branch, y_rail):
        pack = real_eval(pose, psi, branch, y_rail)
        q_arm: np.ndarray
        q_full: np.ndarray
        sig: float
        if pack is None:
            q_arm = np.array(
                [
                    np.deg2rad(2.0),
                    -0.5,
                    0.0,
                    np.deg2rad(90.0),
                    0.0,
                    np.pi / 2.0,
                    np.pi / 2.0,
                ]
            )
            q_full = np.concatenate([[float(y_rail)], q_arm])
            sig = 1.0
        else:
            q_arm, q_full, sig = pack
            q_arm = np.asarray(q_arm, dtype=float).copy()
            q_full = np.asarray(q_full, dtype=float).copy()
        # Homotopy IK still near the planar start (even slightly +) while
        # live J1 is already folded.  That used to freeze s.
        q_arm[0] = np.deg2rad(2.0)
        q_full[1] = np.deg2rad(2.0)
        return q_arm, q_full, sig

    rt._eval.evaluate = _ik_j1_slightly_positive  # type: ignore[method-assign]
    q_lead = q.copy()
    q_lead[1] = np.deg2rad(-90.0)
    q_ik = np.array(
        [np.deg2rad(2.0), -0.5, 0.0, np.deg2rad(90.0), 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    assert rt._q_star_acceptable(q_ik, q_lead, 0.005, 0.78)
    q_ik_stop = q_ik.copy()
    q_ik_stop[3] = np.deg2rad(132.0)
    assert not rt._q_star_acceptable(q_ik_stop, q_lead, 0.005, 0.78)
    d0 = float(rt.d_star_m)
    s_hist = [float(rt.homotopy_s)]
    for _ in range(40):
        rt.step(q_lead, 0.05, rail_lo=0.005, rail_hi=0.78)
        s_hist.append(float(rt.homotopy_s))
    assert s_hist[-1] > s_hist[0] + 0.05
    assert float(rt.d_star_m) > d0 + 1.0e-4


def test_homotopy_ik_failure_freezes_s() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True))
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    rt.step(q, 0.02, rail_lo=0.005, rail_hi=0.78)
    s0 = float(rt.homotopy_s)
    d0 = float(rt.d_star_m)
    assert s0 > 0.0

    def _boom(*_a, **_k):
        return None

    rt._eval.evaluate = _boom  # type: ignore[method-assign]
    rt.step(q, 0.02, rail_lo=0.005, rail_hi=0.78)
    assert float(rt.homotopy_s) == pytest.approx(s0)
    assert float(rt.d_star_m) == pytest.approx(d0)


def test_d_star_keeps_j4_in_band_when_attr_folds() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, d_attr_m=-0.185)
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    real_eval = rt._eval.evaluate

    def _fold_at_attr(pose, psi, branch, y_rail):
        pack = real_eval(pose, psi, branch, y_rail)
        if pack is None:
            return None
        d = float(pose[1]) - float(y_rail)
        q_arm, q_full, sig = pack
        if abs(d - float(cfg.d_attr_m)) < 0.02:
            q_arm = np.asarray(q_arm, dtype=float).copy()
            q_full = np.asarray(q_full, dtype=float).copy()
            q_arm[3] = np.deg2rad(134.0)
            q_full[4] = np.deg2rad(134.0)
            return q_arm, q_full, sig
        return pack

    rt._eval.evaluate = _fold_at_attr  # type: ignore[method-assign]
    last_d = float(rt.d_star_m)
    for _ in range(40):
        _psi, last_d = rt.step(q, 0.05, rail_lo=0.005, rail_hi=0.78)
    assert abs(last_d - float(cfg.d_attr_m)) > 0.02
    assert rt.q_star_rad is not None
    assert float(rt.q_star_rad[4]) < np.deg2rad(122.0)
    d_live = d_from_q(kin, q)
    unload = float(np.sign(d_live - last_d))
    assert unload != 0.0


def test_zero_vel_ff_does_not_freeze_e_mid() -> None:
    kin = RobotKinematics()
    task = RailExtensionTask(
        kin,
        RailExtensionConfig(enabled=True, d_band_m=0.0, v_lpf_tau_s=0.0),
    )
    q = np.array(
        [0.40, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    task.set_d_pref(-0.185)
    v, _w = task(
        q,
        sigma_scale=1.0,
        sigma_grad_rail=0.0,
        vel_ff=np.zeros(6),
        dt_s=0.02,
    )
    assert abs(float(task.last_e_mid_m)) > 1.0e-4
    assert task.last_v_reach == pytest.approx(0.0, abs=1e-12)
    del v


def test_unplanned_d_star_slews_without_step() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, d_center_rate_m_s=0.02)
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    rt._psi_cmd = float(cfg.psi_attr_rad)
    d0 = float(rt.d_star_m)
    dt = 0.02
    prev = d0
    for _ in range(20):
        _psi, d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(d - prev) <= cfg.d_center_rate_m_s * dt + 1e-9
        prev = d
    assert abs(prev - d0) > 1e-4
    assert abs(prev - float(cfg.d_attr_m)) < abs(d0 - float(cfg.d_attr_m))


def test_psi_cmd_does_not_lead_live() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(
        enabled=True,
        psi_rate_rad_s=np.deg2rad(25.0),
        psi_cmd_lead_rad=np.deg2rad(18.0),
    )
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.31, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    live = fold_psi_to_positive(psi_from_q(q))
    dt = 0.02
    for _ in range(80):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(live, psi)) <= cfg.psi_cmd_lead_rad + 1e-9
    assert abs(psi_err_avoiding_zero(live, float(rt._psi_cmd))) == pytest.approx(
        cfg.psi_cmd_lead_rad, abs=np.deg2rad(1.0)
    )


def test_stretched_start_does_not_search_off_attr() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(kin, PsiRetargetConfig(enabled=True, psi_replan_period_s=0.1))
    q = np.array(
        [0.396, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    for _ in range(20):
        rt.step(q, 0.02, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count == 0
    assert rt.psi_star_rad == pytest.approx(float(rt.cfg.psi_attr_rad), abs=1e-9)


def test_arm_angle_reset_folds_negative_pi_reference() -> None:
    kin = RobotKinematics()
    arm = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True, k_psi=1.5))
    q = np.array(
        [0.396, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    arm.psi_ref = -np.pi
    arm.reset(q)
    assert arm._psi_ref_unwrapped == pytest.approx(np.pi, abs=1e-9)
    arm.set_reference(np.deg2rad(70.0))
    assert arm._psi_ref_unwrapped == pytest.approx(np.deg2rad(70.0), abs=1e-9)
    assert float(arm._psi_ref_unwrapped) > 0.0


def test_psi_step_is_rate_limited_and_stays_in_envelope() -> None:
    kin = RobotKinematics()
    cfg = PsiRetargetConfig(enabled=True, psi_rate_rad_s=np.deg2rad(25.0))
    rt = PostureRetarget(kin, cfg)
    q = np.array(
        [0.50, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.pi / 2.0, np.pi / 2.0]
    )
    rt.reset(q)
    rt._psi_star = np.deg2rad(90.0)
    rt.psi_star_rad = float(rt._psi_star)
    dt = 0.02
    prev = float(rt._psi_cmd)
    for _ in range(8):
        psi, _d = rt.step(q, dt, rail_lo=0.005, rail_hi=0.78)
        assert abs(psi_err_avoiding_zero(prev, psi)) <= cfg.psi_rate_rad_s * dt + 1e-9
        assert psi * prev >= -1e-9 or abs(prev) > 0.5 * np.pi
        prev = psi
    assert 0.0 < float(rt._psi_star) < np.pi
    assert not (prev > 0.0 and float(rt._psi_star) < 0.0)


def test_straight_elbow_does_not_search_psi() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, psi_replan_period_s=0.0, psi_wrist_ok_rad=2.0),
    )
    q = np.array(
        [0.16, -2.1, -1.55, 1.2, np.deg2rad(8.0), 0.9, np.deg2rad(15.0), 1.3]
    )
    rt.reset(q)
    for _ in range(8):
        rt.step(q, 0.05, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count == 0
    assert rt.psi_star_rad == pytest.approx(float(rt.cfg.psi_attr_rad), abs=1e-9)


def test_psi_search_is_throttled() -> None:
    kin = RobotKinematics()
    rt = PostureRetarget(
        kin,
        PsiRetargetConfig(enabled=True, psi_replan_period_s=0.5, psi_wrist_ok_rad=2.0),
    )
    q = np.array(
        [0.40, 0.0, np.deg2rad(-30.0), 0.0, np.pi / 2.0, 0.0, np.deg2rad(15.0), np.pi / 2.0]
    )
    rt.reset(q)
    for _ in range(80):
        rt.step(q, 0.005, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count == 0
    for _ in range(40):
        rt.step(q, 0.005, rail_lo=0.005, rail_hi=0.78)
    assert rt.last_psi_search_count == 1


def test_j4_near_damper_does_not_kill_plane_attractor() -> None:
    kin = RobotKinematics()
    centering = JointCenteringTask.from_kinematics(
        kin, NullspaceTaskConfig(k_center=1.0, k_limit=2.0, activation=0.75)
    )
    arm = ArmAngleTask(kin, ArmAngleTaskConfig(enabled=True, k_psi=2.0))
    arm.set_reference(float(np.pi))
    comp = SecondaryComposer.from_controller_parts(
        centering, arm, centering.cfg, v_max=kin.v_max, max_qdot_frac=0.2
    )
    q = np.array(
        [0.55, 0.0, np.deg2rad(-30.0), 0.0, np.deg2rad(120.0), 0.0, np.deg2rad(56.0), 0.0]
    )
    qdot = comp.compose(q, None, np.zeros(8), arm_suppressed=False)
    assert comp.last_limit_activation > 0.5
    assert comp.last_arm_smooth > 0.5
    assert float(np.linalg.norm(qdot[1:])) > 0.0


def test_aniso_task_weight_spares_xy_when_umin_is_wz() -> None:
    kin = RobotKinematics()
    limits = SafetyLimits(
        q_lower=kin.q_lower,
        q_upper=kin.q_upper,
        v_max=kin.v_max,
        a_max=np.full(kin.nv, 10.0),
        position_margin=np.full(kin.nv, 0.01),
    )
    cfg = QpConfig(
        task_weight=np.array([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]),
        aniso_task_damping=True,
        task_weight_lpf_tau_s=0.0,
        sr_damping=cfg_sr(),
        collision=kin_collision_off(),
    )
    core = QpIkController(kin, limits, cfg)
    # Weighted J whose last left singular vector is task-wz.
    w = np.asarray(cfg.task_weight, dtype=float)
    w_sqrt = np.sqrt(w)
    s_j = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.057])
    u = np.eye(6)
    v = np.eye(kin.nv)[:, :6]
    jw = u @ np.diag(s_j) @ v.T
    j = jw / w_sqrt[:, None]
    mat = core._task_weight_matrix(j, dt=0.005, keep_task_weight=False)
    assert mat[0, 0] == pytest.approx(100.0, rel=1e-6)
    assert mat[1, 1] == pytest.approx(100.0, rel=1e-6)
    s_min = (0.057 / 0.08) ** 2
    assert core.last_s_sigma[-1] == pytest.approx(s_min, rel=1e-6)
    umin = u[:, -1]
    assert float(umin @ mat @ umin) == pytest.approx(50.0 * s_min, rel=1e-5)


def cfg_sr():
    from rm75_control.control.joint_admittance_8dof.ik_types import SrDampingConfig

    return SrDampingConfig(lam0=0.05, sigma_ref=0.08, sigma_floor=1e-6)


def kin_collision_off():
    from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig

    return CollisionConfig(enabled=False)


def test_branch_barrier_dwell_keeps_crossing_cheap() -> None:
    cfg = BranchBarrierConfig(
        enabled=True,
        activate_rad=0.52,
        eps_rad=0.35,
        slack_weight=80.0,
        dwell_free_s=0.3,
        dwell_ramp_s=1.0,
        dwell_scale_max=5.0,
    )
    b = BranchBarrierBuilder(cfg)
    q_star = np.array([0.0, 0.0, -0.8, 0.0, 1.6, 0.7, 1.05, 0.0])
    q = q_star.copy()
    q[6] = 0.20  # ~11.5°, inside the 30° band
    b.build_rows(q, q_star, dt_s=0.05)
    for _ in range(3):  # 0.20 s total
        b.build_rows(q, q_star, dt_s=0.05)
    assert b.last_dwell_scale == pytest.approx(1.0)
    assert cfg.slack_weight * b.last_dwell_scale == pytest.approx(80.0)
    for _ in range(26):  # +1.30 s → 1.50 s total
        b.build_rows(q, q_star, dt_s=0.05)
    assert b.last_dwell_scale == pytest.approx(5.0)
    assert cfg.slack_weight * b.last_dwell_scale == pytest.approx(400.0)
    q[6] = 0.80
    b.build_rows(q, q_star, dt_s=0.05)
    assert b.last_dwell_scale == pytest.approx(1.0)


def test_qp_smoothness_weight_is_wired() -> None:
    cfg = QpConfig(smoothness_weight=0.15)
    assert cfg.smoothness_weight == 0.15


def test_governor_floor_and_physical_gate() -> None:
    from rm75_control.control.joint_admittance_8dof.loop import (
        Phase,
        _reference_governor_scale,
    )

    class _Outer:
        pass

    phase = Phase(
        outer=_Outer(),
        governor_err_ok_mm=5.0,
        governor_err_max_mm=25.0,
        governor_scale_min=0.25,
        governor_joint_err_max_deg=0.0,
    )
    raw_free = _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=False
    )
    assert raw_free == 1.0
    raw_sat = _reference_governor_scale(
        phase, outer_err_mm=80.0, joint_err_deg=None, physical_saturated=True
    )
    assert raw_sat == 0.25

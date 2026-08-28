"""P0 rail exact compensation, slack latch, posture-only fail-slow, σ barrier."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    SigmaSetBasedConfig,
    SigmaSetBasedTracker,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_command import (
    RailCommandMixer,
    allocate_rail_shares,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import RailMode


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller() -> JointIkController:
    collision = CollisionConfig(enabled=False)
    qp = QpConfig(
        backend="proxqp",
        collision=collision,
        smoothness_weight=np.r_[0.0, np.full(7, 0.15)],
    )
    cfg = JointIkConfig(control_frame="base", qp=qp, collision=collision)
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def _step_core(core, q, v_cmd, rail_exec, *, jacobian=None):
    kin = core.kin
    J = kin.jacobian(q) if jacobian is None else jacobian
    return core.step(
        q,
        np.asarray(v_cmd, dtype=float),
        0.005,
        q_meas=q,
        rail_exec_vel_m_s=float(rail_exec),
        zero_secondary_rail=True,
        jacobian=J,
        sigma=kin.singular_values(J),
    )


def _assert_exact_b_task(core, q, v_cmd, rail_exec) -> np.ndarray:
    J = core.kin.jacobian(q)
    _step_core(core, q, v_cmd, rail_exec, jacobian=J)
    expected = np.asarray(v_cmd, dtype=float) - J[:, 0] * float(rail_exec)
    b_task = core.last_task_target - core.last_rail_exec_contrib
    np.testing.assert_allclose(b_task, expected, atol=1.0e-12)
    return b_task


def test_exact_compensation_target_is_continuous_across_j4_threshold() -> None:
    controller = _controller()
    core = controller.core
    q_lo = Q_SAFE.copy()
    q_hi = Q_SAFE.copy()
    q_lo[4] = np.deg2rad(107.5)
    q_hi[4] = np.deg2rad(108.5)
    v_cmd = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    b_lo = _assert_exact_b_task(core, q_lo, v_cmd, 0.04)
    b_hi = _assert_exact_b_task(core, q_hi, v_cmd, 0.04)
    np.testing.assert_allclose(b_lo[3:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(b_hi[3:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(b_lo, b_hi, atol=5.0e-3)


def test_exact_compensation_target_is_continuous_across_j2_threshold() -> None:
    controller = _controller()
    core = controller.core
    q_lo = Q_SAFE.copy()
    q_hi = Q_SAFE.copy()
    q_lo[2] = np.deg2rad(-103.0)
    q_hi[2] = np.deg2rad(-105.0)
    v_cmd = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    b_lo = _assert_exact_b_task(core, q_lo, v_cmd, 0.06)
    b_hi = _assert_exact_b_task(core, q_hi, v_cmd, 0.06)
    np.testing.assert_allclose(b_lo[3:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(b_hi[3:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(b_lo, b_hi, atol=5.0e-3)


def test_infeasible_compensation_grows_explicit_task_slack() -> None:
    controller = _controller()
    core = controller.core
    v_cmd = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    rail_exec = 0.12
    _step_core(core, Q_SAFE, v_cmd, rail_exec)
    physical = (
        core.last_rail_exec_contrib + core.last_arm_contrib - core.last_task_target
    )
    np.testing.assert_allclose(
        physical, -core.last_task_residual, atol=1.0e-9, rtol=0.0
    )
    assert float(core.last_task_residual_norm) >= 0.0


def test_slack_latch_hysteresis() -> None:
    ctrl = _controller()
    enter = 0.15
    exit_ = 0.03
    seq = [0.14, 0.151, 0.149, 0.10, 0.031, 0.029]
    expect = [False, True, True, True, True, False]
    got = []
    for value in seq:
        if (not ctrl._slack_hold_latched) and value >= enter:
            ctrl._slack_hold_latched = True
        elif ctrl._slack_hold_latched and value <= exit_:
            ctrl._slack_hold_latched = False
        got.append(bool(ctrl._slack_hold_latched))
    assert got == expect


def test_posture_hold_does_not_clear_task_share() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    mix.xi = 0.04
    tel = mix.step(
        d_live=-0.03,
        d_star_target=0.0,
        u_task_raw=0.05,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        posture_hold=True,
        quiescent=False,
    )
    expected = allocate_rail_shares(
        u_task_raw=0.05,
        u_post_raw=0.0,
        u_escape_raw=0.0,
        escape_dir=0,
        u_lo=-0.12,
        u_hi=0.12,
    )
    assert tel.u_post_feasible == pytest.approx(0.0, abs=1e-12)
    assert tel.u_task_feasible == pytest.approx(expected["u_task_feasible"], abs=1e-12)
    assert mix.xi == pytest.approx(-mix.kp * tel.e_d, abs=1e-12)


def test_posture_hold_keeps_nonzero_task_when_interval_is_open() -> None:
    mix = RailCommandMixer(kp=1.2, ki=0.8, u_mid_max=0.12, kaw=8.0)
    mix.d_star.init_from_live(0.0)
    tel = mix.step(
        d_live=0.0,
        d_star_target=0.0,
        u_task_raw=0.04,
        u_escape_raw=0.0,
        escape_explicit=False,
        dt=0.005,
        u_max=0.12,
        posture_hold=True,
        quiescent=False,
    )
    assert tel.u_task_feasible == pytest.approx(0.04, abs=1e-12)
    assert tel.u_post_feasible == pytest.approx(0.0, abs=1e-12)


def test_slack_hold_latches_but_still_steps_planner() -> None:
    ctrl = _controller()
    ctrl.set_rail_mode(RailMode.COUPLED)
    twist = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
    ctrl.update(twist, 0.005, q_meas=Q_SAFE, rail_exec_vel_m_s=0.0)
    seeded = 0.42
    ctrl.posture_retarget._s = seeded
    ctrl.posture_retarget.homotopy_s = seeded
    ctrl.posture_retarget._held_prev = False
    ctrl.last_slack_norm = 0.20
    ctrl.update(twist, 0.005, q_meas=Q_SAFE, rail_exec_vel_m_s=0.0)
    assert ctrl._slack_hold_latched is True
    assert ctrl.posture_retarget.homotopy_s != pytest.approx(seeded, abs=1e-12)
    ctrl.last_slack_norm = 0.02
    ctrl.update(twist, 0.005, q_meas=Q_SAFE, rail_exec_vel_m_s=0.0)
    assert ctrl._slack_hold_latched is False
    assert ctrl.posture_retarget.homotopy_s != pytest.approx(0.0, abs=1e-9)


def test_reset_and_stop_clear_slack_latch() -> None:
    ctrl = _controller()
    ctrl._slack_hold_latched = True
    ctrl.stop()
    assert ctrl._slack_hold_latched is False
    ctrl._slack_hold_latched = True
    ctrl.reset(Q_SAFE)
    assert ctrl._slack_hold_latched is False
    ctrl._slack_hold_latched = True
    ctrl.begin_hybrid_episode(Q_SAFE, np.zeros(8))
    assert ctrl._slack_hold_latched is False


def _diverging_sigma_q() -> np.ndarray:
    # Logged fly-window pose: σ_min(6×8) and σ_arm split by construction.
    return np.array(
        [
            0.412,
            -0.21,
            -2.265,  # J2 near the −130° stop
            0.55,
            1.90,  # J4 past 108°
            1.05,
            1.10,
            0.20,
        ]
    )


def test_sigma_barrier_uses_arm_sigma_not_full_jacobian() -> None:
    kin = RobotKinematics()
    q = _diverging_sigma_q()
    J = kin.jacobian(q)
    sigma_full = float(np.linalg.svd(J, compute_uv=False).min())
    sigma_arm = float(np.linalg.svd(J[:, 1:], compute_uv=False).min())
    assert abs(sigma_full - sigma_arm) > 1.0e-3
    tr = SigmaSetBasedTracker(
        SigmaSetBasedConfig(activate=0.20, safe=0.045, exit=0.30, enabled=True)
    )
    row = tr.build_row(kin, q)
    assert tr.last_sigma == pytest.approx(sigma_arm, abs=1e-9)
    assert tr.last_sigma != pytest.approx(sigma_full, abs=1e-4)
    native_rhs = -8.0 * (sigma_arm - 0.045)
    python_rhs = 8.0 * (0.045 - sigma_arm)
    assert native_rhs == pytest.approx(python_rhs, abs=1e-12)
    assert row.active
    assert row.lower[0] == pytest.approx(native_rhs, rel=1e-6, abs=1e-9)
    np.testing.assert_allclose(row.jacobian[0], tr.last_grad, atol=1e-12)


def test_sigma_barrier_exit_is_reachable_on_logged_range() -> None:
    tr = SigmaSetBasedTracker(
        SigmaSetBasedConfig(activate=0.09, safe=0.045, exit=0.13, enabled=True)
    )
    values = [0.08, 0.07, 0.11, 0.125, 0.131, 0.12]
    states = [tr.update_hysteresis(v) for v in values]
    assert True in states
    assert states[-2] is False
    old = SigmaSetBasedTracker(
        SigmaSetBasedConfig(activate=0.12, safe=0.06, exit=0.16, enabled=True)
    )
    logged_max = 0.1413
    old.update_hysteresis(0.08)
    assert old.update_hysteresis(logged_max) is True


def test_sigma_gradient_refreshes_on_first_tick_and_activation_edge() -> None:
    kin = RobotKinematics()
    tr = SigmaSetBasedTracker(
        SigmaSetBasedConfig(
            activate=0.20,
            safe=0.05,
            exit=0.30,
            enabled=True,
            grad_period_ticks=10,
        )
    )
    tr.reset()
    assert tr.last_grad is None
    row = tr.build_row(kin, Q_SAFE)
    assert tr.last_grad is not None
    assert row.active
    held = tr.last_grad.copy()
    q2 = Q_SAFE.copy()
    q2[4] += 0.05
    row2 = tr.build_row(kin, q2)
    assert row2.active
    np.testing.assert_allclose(tr.last_grad, held)
    tr.active = False
    tr.last_grad = None
    tr._grad_tick = 3
    row3 = tr.build_row(kin, Q_SAFE)
    assert row3.active
    assert tr.last_grad is not None
    assert tr._grad_tick == 0


def test_regression_script_points_at_this_repo() -> None:
    path = Path(__file__).resolve().parents[1] / "tests" / "controller_fix_regression.py"
    if not path.exists():
        path = Path(__file__).resolve().parent / "controller_fix_regression.py"
    text = path.read_text(encoding="utf-8")
    assert "reconstructed/rm75_control" not in text
    assert "secondary_alpha=float(secondary_alpha)" in text
    assert "b_task = v_cmd - rail_contrib" in text
    assert "slack_now <= slack_exit" in text

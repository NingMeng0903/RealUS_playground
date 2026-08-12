"""The solver velocity is the only Cartesian command sent/integrated."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    ProtectedTask,
)
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _guard_qpik_step_before_send,
    _publish_rail_target_before_arm,
    _rail_m_for_feedback,
)
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    full_q_from_arm,
)
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import Watchdog


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller() -> JointIkController:
    cfg = JointIkConfig(
        generic_qpik=GenericQpikRuntimeConfig(
            solver=TwoLevelQpikConfig(
                backend="scipy",
            max_solve_ms=500.0,
                max_rows=96,
                max_scalable_groups=4,
                max_iter=100,
            )
        ),
        collision=CollisionConfig(enabled=False),
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def test_successful_final_send_is_exact_solver_velocity() -> None:
    controller = _controller()
    captured: list[np.ndarray] = []
    real_solve = controller.core.solve

    def recording_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        captured.append(result.qdot.copy())
        return result

    controller.core.solve = recording_solve  # type: ignore[method-assign]
    q_before = controller.q_cmd.copy()
    step = controller.update(
        np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]),
        q_meas=Q_SAFE,
    )
    np.testing.assert_allclose(step.qdot, captured[-1], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * captured[-1], atol=1e-12, rtol=0.0
    )


def test_qp2_failure_final_send_is_exact_same_tick_qp1() -> None:
    controller = _controller()
    backend = controller.core.solver.backend_qp2
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        q_before = controller.q_cmd.copy()
        step = controller.update(
            np.array([0.008, 0.003, -0.002, 0.0, 0.0, 0.0]),
            q_meas=Q_SAFE,
        )
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]
    assert step.fallback_level == "qp1"
    np.testing.assert_allclose(
        step.protected_achieved,
        step.protected_target + step.protected_residual,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        step.qdot, (step.q_send - q_before) / controller.cfg.dt, atol=1e-12
    )


def test_qp1_failure_never_reintroduces_previous_limiter_velocity() -> None:
    controller = _controller()
    # Seed both the solver and the direct-PTP limiter histories with motion.
    previous = np.full(8, 0.05)
    controller.core.sync_applied(previous)
    controller.safety.sync_applied_delta(previous * controller.cfg.dt, controller.cfg.dt)
    backend = controller.core.solver.backend_qp1
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    q_before = controller.q_cmd.copy()
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]
    # Soft P0-safe fallback: no latch, and never reintroduce the full previous
    # limiter velocity (accel-compatible box projection may keep a residual).
    assert not step.solver_fault_latched
    assert step.fallback_level == "p0_safe"
    assert step.fallback_reason.startswith("qp1_failed_p0_")
    assert not np.allclose(step.qdot, previous)
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * step.qdot, atol=1e-12
    )


def test_dense_one_sided_row_survives_final_send_without_axis_clipping() -> None:
    controller = _controller()
    row = np.array([0.2, -0.5, 0.3, 0.1, 0.4, -0.2, 0.25, -0.15])
    protected = ProtectedTask(np.zeros((0, 8)), np.zeros(0), name="none")
    safety = HardConstraintRow(row, lower=0.012, name="dense_application_row")
    step = controller.update_tasks(
        protected,
        q_meas=Q_SAFE,
        application_hard_rows=(safety,),
    )
    assert float(row @ step.qdot) >= 0.012 - 2e-6
    assert not step.solver_fault_latched


def test_cartesian_qpik_rejects_missing_measured_state() -> None:
    controller = _controller()
    try:
        controller.update(np.zeros(6))
    except ValueError as exc:
        assert "q_meas is required" in str(exc)
    else:  # pragma: no cover - safety regression guard
        raise AssertionError("missing measured state was accepted")


def test_locked_hold_never_teleports_rail_reference() -> None:
    controller = _controller()
    rail_before = float(controller.q_cmd[0])
    with np.testing.assert_raises_regex(ValueError, "cannot move rail"):
        controller.set_rail_mode(
            RailMode.LOCKED,
            locked_style=LockedStyle.HOLD,
            q_ref_m=rail_before + 0.05,
        )
    assert float(controller.q_cmd[0]) == rail_before

    controller.set_rail_mode(
        RailMode.LOCKED,
        locked_style=LockedStyle.HOLD,
        q_ref_m=rail_before,
    )
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    assert float(step.q_send[0]) == rail_before


def test_qp1_soft_fallback_remains_sendable() -> None:
    controller = _controller()
    backend = controller.core.solver.backend_qp1
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]

    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(
        step, lambda _reason: events.append("stop")
    )
    if sendable:
        events.extend(["rail_send", "arm_send"])
    assert sendable
    assert reason == ""
    assert step.fallback_level == "p0_safe"
    assert not step.solver_fault_latched
    assert events == ["rail_send", "arm_send"]


def test_latched_fault_still_stops_before_send() -> None:
    controller = _controller()
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    step.solver_fault_latched = True
    step.fallback_level = "fault"
    step.fallback_reason = "forced"
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(
        step, lambda _reason: events.append("stop")
    )
    if sendable:
        events.extend(["rail_send", "arm_send"])
    assert not sendable
    assert reason.startswith("qpik_fault:")
    assert events == ["stop"]


def test_rail_rejection_stops_before_arm_half_of_8d_tick() -> None:
    class RejectingRail:
        enabled = True
        calibrated = True
        armed = True
        panicked = False
        panic_reason = ""

        def set_target_m(self, _target):
            return False

    events: list[str] = []
    accepted, reason = _publish_rail_target_before_arm(
        RejectingRail(), 0.4, lambda value: events.append(value)
    )
    if accepted:
        events.append("arm_send")
    assert not accepted
    assert reason == "rail_target_rejected:bridge_declined"
    assert events == [reason]


def test_invalid_enabled_rail_feedback_never_falls_back_to_q_cmd() -> None:
    rail = SimpleNamespace(enabled=True, measured_m=float("nan"))
    inner = SimpleNamespace(q_cmd=np.array([0.4]))
    with np.testing.assert_raises_regex(RuntimeError, "non-finite"):
        _rail_m_for_feedback(rail, inner)


def test_direct_mode_cannot_bypass_latched_solver_fault_and_reset_clears_mode() -> None:
    controller = _controller()
    controller.set_direct_joint_ptp(True)
    controller.core.solver.fault_latched = True
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, qdot_ff=np.full(8, 0.05)
    )
    assert step.solver_fault_latched
    np.testing.assert_allclose(step.qdot, np.zeros(8), atol=0.0)
    controller.reset(Q_SAFE)
    assert not controller._direct_joint_ptp
    assert not controller._plan_drives_rail


def test_watchdog_fault_is_latched_until_explicit_phase_arm() -> None:
    import threading

    fired = threading.Event()
    watchdog = Watchdog(0.01, fired.set, poll_s=0.001)
    watchdog.start()
    try:
        assert fired.wait(0.5)
        assert watchdog.fired
        assert not watchdog.beat()
        assert watchdog.fired
        watchdog.arm()
        assert not watchdog.fired
        assert watchdog.beat()
    finally:
        watchdog.stop()

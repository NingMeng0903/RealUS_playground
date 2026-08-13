"""The slack-QP velocity is the Cartesian command sent; QP fail never latches stop."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _guard_qpik_step_before_send,
    _publish_rail_target_before_arm,
    _rail_m_for_feedback,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.utils.safety import Watchdog


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller() -> JointIkController:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
    )
    cfg.rail.mode = RailMode.COUPLED
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def test_successful_final_send_is_exact_qp_velocity() -> None:
    controller = _controller()
    q_before = controller.q_cmd.copy()
    step = controller.update(
        np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]), q_meas=Q_SAFE
    )
    assert step.qp_solver_call_count == 1
    assert not step.solver_fault_latched
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * step.qdot, atol=1e-9, rtol=0.0
    )


def test_backend_failure_decays_qdot_and_stays_sendable() -> None:
    controller = _controller()
    previous = np.full(8, 0.05)
    controller.core.sync_applied(previous)
    backend = controller.core.backend
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    q_before = controller.q_cmd.copy()
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]

    assert step.qp_solver_call_count == 1
    assert step.fallback_level == "decay"
    assert step.qpik_authority == 1.0
    assert not step.solver_fault_latched
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert events == []
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * step.qdot, atol=1e-12
    )


def test_empty_velocity_box_does_not_latch_stop() -> None:
    controller = _controller()
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert not step.solver_fault_latched
    assert events == []


def test_numerical_fallback_remains_sendable() -> None:
    controller = _controller()
    backend = controller.core.backend
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: None  # type: ignore[method-assign]
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]

    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert step.fallback_level == "decay"
    assert step.qpik_authority == 1.0
    assert events == []


def test_direct_joint_ptp_calls_no_cartesian_backend() -> None:
    controller = _controller()
    controller.set_direct_joint_ptp(True)
    calls_before = controller.core.solve_count
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, qdot_ff=np.full(8, 0.05)
    )
    assert controller.core.solve_count == calls_before
    assert step.controller_mode == "direct_joint_ptp"
    assert step.qp_solver_call_count == 0
    assert not step.solver_fault_latched
    controller.reset(Q_SAFE)
    assert not controller._direct_joint_ptp
    assert not controller._plan_drives_rail


def test_cartesian_qpik_rejects_missing_measured_state() -> None:
    controller = _controller()
    with np.testing.assert_raises_regex(ValueError, "q_meas is required"):
        controller.update(np.zeros(6))


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
        RailMode.LOCKED, locked_style=LockedStyle.HOLD, q_ref_m=rail_before
    )
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    assert float(step.q_send[0]) == rail_before


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
        RejectingRail(), 0.4, events.append
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


def test_watchdog_fault_is_latched_until_explicit_phase_arm() -> None:
    import threading

    fired = threading.Event()
    watchdog = Watchdog(0.01, fired.set, poll_s=0.001)
    watchdog.start()
    try:
        assert fired.wait(0.5)
        assert watchdog.fired
        assert not watchdog.beat()
        watchdog.arm()
        assert not watchdog.fired
        assert watchdog.beat()
    finally:
        watchdog.stop()

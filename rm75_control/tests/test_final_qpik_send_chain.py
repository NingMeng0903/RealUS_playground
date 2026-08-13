"""The validated single-QPIK velocity is the only Cartesian command sent."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import GenericQpikRuntimeConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _guard_qpik_step_before_send,
    _publish_rail_target_before_arm,
    _rail_m_for_feedback,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import VelocityBoxInfeasible
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import SingleQpikConfig
from rm75_control.control.joint_admittance_8dof.task_adapter import TaskSpaceConstraintRow
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.utils.safety import Watchdog


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def _controller() -> JointIkController:
    cfg = JointIkConfig(
        control_frame="base",
        generic_qpik=GenericQpikRuntimeConfig(
            solver=SingleQpikConfig(
                backend="scipy", max_solve_ms=500.0, max_iter=200,
                authority_rise_per_s=1000.0,
            )
        ),
        collision=CollisionConfig(enabled=False),
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    return controller


def _failed_backend_result():
    return SimpleNamespace(
        x=None,
        success=False,
        status="forced_failure",
        iterations=0,
        elapsed_s=0.001,
        message="forced",
    )


def test_successful_final_send_is_exact_single_solver_velocity() -> None:
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
        np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]), q_meas=Q_SAFE
    )
    assert step.qp_solver_call_count == 1
    np.testing.assert_allclose(step.qdot, captured[-1], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * captured[-1], atol=1e-12, rtol=0.0
    )


def test_full_cartesian_tick_uses_one_jacobian_snapshot() -> None:
    controller = _controller()
    real_jacobian = controller.kin.jacobian
    calls = 0

    def counted(q):
        nonlocal calls
        calls += 1
        return real_jacobian(q)

    controller.kin.jacobian = counted  # type: ignore[method-assign]
    controller.core._arm_gradient[:] = 1.0
    controller.core._arm_gradient_valid = True
    controller.update(np.zeros(6), q_meas=Q_SAFE)
    assert calls == 1


def test_backend_failure_publishes_exact_certified_anchor_and_freezes_authority() -> None:
    controller = _controller()
    previous = np.full(8, 0.05)
    controller.core.sync_applied(previous)
    backend = controller.core.solver.backend
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: _failed_backend_result()  # type: ignore[method-assign]
    q_before = controller.q_cmd.copy()
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]

    assert step.qp_solver_call_count == 1
    assert step.fallback_level == "hard_anchor"
    assert step.fallback_reason == "main_qp_forced_failure"
    assert step.qpik_authority == 0.0
    assert not step.solver_fault_latched
    assert not np.allclose(step.qdot, previous)
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * step.qdot, atol=1e-12
    )


def test_task_space_hard_row_survives_final_send_without_axis_clipping() -> None:
    controller = _controller()
    coefficient = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    safety = TaskSpaceConstraintRow(
        coefficient, lower=0.004, name="application_x_min"
    )
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, task_safety_rows=(safety,)
    )
    jacobian = controller.kin.jacobian(Q_SAFE)
    assert float(coefficient @ jacobian @ step.qdot) >= 0.004 - 1.0e-5
    assert not step.solver_fault_latched


def test_hard_construction_failure_stops_before_send_and_skips_backend() -> None:
    controller = _controller()
    calls_before = controller.core.solver.solve_count

    def fail_build(*args, **kwargs):
        del args, kwargs
        raise VelocityBoxInfeasible(
            "acceleration", np.array([4]), np.ones(8), -np.ones(8)
        )

    controller.core.p0_builder.build = fail_build  # type: ignore[method-assign]
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)

    assert controller.core.solver.solve_count == calls_before
    assert step.qp_solver_call_count == 0
    assert step.solver_fault_latched
    assert not sendable
    assert reason.startswith("qpik_fault:fault:VelocityBoxInfeasible")
    assert events == [reason]


def test_numerical_fallback_remains_sendable() -> None:
    controller = _controller()
    backend = controller.core.solver.backend
    real_solve = backend.solve
    backend.solve = lambda *args, **kwargs: _failed_backend_result()  # type: ignore[method-assign]
    try:
        step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    finally:
        backend.solve = real_solve  # type: ignore[method-assign]

    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert step.fallback_level == "hard_anchor"
    assert step.qpik_anchor_valid
    assert step.qpik_authority == 0.0
    assert events == []


def test_direct_joint_ptp_calls_no_cartesian_backend() -> None:
    controller = _controller()
    controller.set_direct_joint_ptp(True)
    calls_before = controller.core.solver.solve_count
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, qdot_ff=np.full(8, 0.05)
    )
    assert controller.core.solver.solve_count == calls_before
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

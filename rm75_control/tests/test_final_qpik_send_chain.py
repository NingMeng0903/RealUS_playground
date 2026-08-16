"""Only a current hard-feasible QPIK result may reach rail/CANFD publication."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _guard_qpik_step_before_send,
    _publish_rail_target_before_arm,
    _qpik_rail_v_ff_m_s,
    _rail_m_for_feedback,
    _wall_clock_rail_target,
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


def test_qp1_backend_failure_is_stopped_before_send() -> None:
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
    assert step.fallback_level == "stop"
    assert step.qpik_authority == 1.0
    assert step.solver_fault_latched
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert not sendable
    assert reason == "qpik_fault:stop:qp_failed"
    assert events == [reason]
    np.testing.assert_allclose(
        step.q_send, q_before + controller.cfg.dt * step.qdot, atol=1e-12
    )
    np.testing.assert_allclose(step.qdot, np.zeros(8), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        controller.core.qdot_prev, previous, atol=0.0, rtol=0.0
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


def test_numerical_qp1_fallback_is_not_sendable() -> None:
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
    assert not sendable
    assert reason == "qpik_fault:stop:qp_failed"
    assert step.fallback_level == "stop"
    assert step.qpik_authority == 1.0
    assert step.solver_fault_latched
    assert events == [reason]


def test_qp2_failure_keeps_same_tick_qp1_sendable() -> None:
    controller = _controller()
    solve_qp2 = controller.core._backend_qp2.solve
    controller.core._backend_qp2.solve = lambda *args, **kwargs: None
    try:
        step = controller.update(
            np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]),
            q_meas=Q_SAFE,
            rail_exec_vel_m_s=0.0,
        )
    finally:
        controller.core._backend_qp2.solve = solve_qp2

    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert step.qp2_fallback
    assert not step.solver_fault_latched
    assert sendable
    assert reason == ""
    assert events == []


def test_post_qp_certificate_failure_is_stopped_and_not_integrated() -> None:
    controller = _controller()
    q_before = controller.q_cmd.copy()
    qdot_before = controller.core.qdot_prev.copy()
    real_validate = controller.core.validate_final_qdot
    controller.core.validate_final_qdot = lambda _qdot: (2.0e-3, 0.0)
    try:
        step = controller.update(
            np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]),
            q_meas=Q_SAFE,
            rail_exec_vel_m_s=0.0,
        )
    finally:
        controller.core.validate_final_qdot = real_validate

    assert step.solver_fault_latched
    assert "final_publication_certificate_failed" in step.fallback_reason
    np.testing.assert_allclose(step.q_send, q_before, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(step.qdot, np.zeros(8), atol=0.0, rtol=0.0)
    np.testing.assert_allclose(
        controller.core.qdot_prev, qdot_before, atol=0.0, rtol=0.0
    )
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert not sendable
    assert reason.startswith("qpik_fault:stop:final_publication_certificate_failed")


def test_send_guard_independently_rejects_bad_hard_certificate() -> None:
    controller = _controller()
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    step.solver_fault_latched = False
    step.qpik_hard_residual_max = 2.0e-3
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert not sendable
    assert reason == "qpik_fault:hard_certificate:violation=2.000e-03"
    assert events == [reason]


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


def test_rail_panic_is_reported_before_not_armed() -> None:
    class PanickedRail:
        enabled = True
        calibrated = True
        armed = False
        panicked = True
        panic_reason = "home_di"

    events: list[str] = []
    accepted, reason = _publish_rail_target_before_arm(
        PanickedRail(), 0.4, events.append
    )
    assert not accepted
    assert "home_di" in reason
    assert "re-arm" in reason
    assert "not_armed" not in reason
    assert events == [reason]


def test_rail_rejection_stops_before_arm_half_of_8d_tick() -> None:
    class RejectingRail:
        enabled = True
        calibrated = True
        armed = True
        panicked = False
        panic_reason = ""

        def set_target_m(self, _target, v_ff_m_s=None):
            del v_ff_m_s
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


def test_publish_forwards_v_ff_to_the_bridge() -> None:
    class RecordingRail:
        enabled = True
        calibrated = True
        armed = True
        panicked = False
        panic_reason = ""
        seen: tuple[float, float | None] | None = None

        def set_target_m(self, target, v_ff_m_s=None):
            self.seen = (float(target), v_ff_m_s)
            return True

    rail = RecordingRail()
    events: list[str] = []
    accepted, reason = _publish_rail_target_before_arm(
        rail, 0.41, events.append, v_ff_m_s=0.08
    )
    assert accepted
    assert reason == ""
    assert events == []
    assert rail.seen == (0.41, 0.08)


def test_publish_rejects_an_explicit_mode_mismatch() -> None:
    class WrongModeRail:
        enabled = True
        calibrated = True
        armed = True
        panicked = False
        panic_reason = ""
        command_mode = "position"

        def set_target_m(self, _target, v_ff_m_s=None, *, mode=None):
            del v_ff_m_s, mode
            return True

    events: list[str] = []
    accepted, reason = _publish_rail_target_before_arm(
        WrongModeRail(),
        0.41,
        events.append,
        v_ff_m_s=0.08,
        command_mode="coupled_velocity",
    )
    assert not accepted
    assert "mode_mismatch" in reason
    assert events == [reason]


def test_qpik_rail_v_ff_is_ik_qdot_not_pad_bypass() -> None:
    assert _qpik_rail_v_ff_m_s(0.079) == pytest.approx(0.079)
    assert _qpik_rail_v_ff_m_s(0.0) == 0.0
    assert _qpik_rail_v_ff_m_s(5.0e-4) == 0.0
    assert _qpik_rail_v_ff_m_s(float("nan")) == 0.0
    assert _qpik_rail_v_ff_m_s(float("inf")) == 0.0


def test_wall_clock_rail_target_one_tick_extra_not_accumulator() -> None:
    pub = _wall_clock_rail_target(
        0.4004,
        0.08,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
    )
    assert pub == pytest.approx(0.4004 + 0.08 * 0.0015)
    again = _wall_clock_rail_target(
        0.4008,
        0.08,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
    )
    assert again == pytest.approx(0.4008 + 0.08 * 0.0015)
    # A persistent integrator would have been prev_pub + qdot * dt_wall.
    assert again != pytest.approx(pub + 0.08 * 0.0065)


def test_wall_clock_idle_publishes_q_send_without_lead_chase() -> None:
    first = _wall_clock_rail_target(
        0.40,
        0.0,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
        meas_m=0.42,
        lead_max_m=0.020,
    )
    assert first == pytest.approx(0.40)
    walked = 0.40
    for _ in range(200):
        walked = _wall_clock_rail_target(
            0.40,
            -0.033,
            0.0065,
            0.005,
            soft_lo=0.025,
            soft_hi=0.78,
            meas_m=walked,
            lead_max_m=0.020,
        )
    # Residual qdot without a persistent integrator cannot walk 20 mm.
    assert abs(walked - 0.40) < 0.020


def test_zero_v_cmd_does_not_invent_rail_task() -> None:
    controller = _controller()
    q = Q_SAFE.copy()
    controller.reset(q)
    controller.centering_task.set_q_target(q)
    if controller.rail_ext_task is not None:
        d_now = float(controller.kin.fk_placement(q).translation[1]) - float(q[0])
        controller.rail_ext_task.set_d_pref(d_now)
    step = controller.update(np.zeros(6), q_meas=q)
    assert not np.isfinite(step.rail_task_vel) or abs(float(step.rail_task_vel)) < 1e-9
    assert abs(float(step.qdot[0])) < 0.01
    assert _qpik_rail_v_ff_m_s(float(step.qdot[0])) == 0.0


def test_lead_clamp_does_not_invent_qdot_above_vmax() -> None:
    controller = _controller()
    q_meas = Q_SAFE.copy()
    q_meas[0] = 0.40
    controller.reset(q_meas)
    controller.q_cmd[0] = 0.50
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    step = controller.update(twist, q_meas=q_meas)
    v_max = float(controller.limits.v_max[0])
    assert abs(float(step.qdot[0])) <= v_max + 1e-9
    assert abs(float(step.q_send[0]) - 0.40) <= float(controller.cfg.resync_err_rail_m) + 1e-9


def test_nonzero_v_cmd_publishes_qpik_qdot_not_v_ff_rail() -> None:
    controller = _controller()
    q = Q_SAFE.copy()
    controller.reset(q)
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    step = None
    for _ in range(40):
        step = controller.update(twist, q_meas=controller.q_cmd.copy(), vel_ff=twist)
    assert step is not None
    qdot0 = float(step.qdot[0])
    published = _qpik_rail_v_ff_m_s(qdot0)
    assert abs(qdot0) > 1.0e-3
    assert published == pytest.approx(qdot0)
    pad_proj = float(step.v_ff_rail)
    # Servo v_ff is the IK rail velocity.  Pad/path projection is not substituted.
    if math.isfinite(pad_proj) and abs(pad_proj - qdot0) > 1.0e-3:
        assert published != pytest.approx(pad_proj)

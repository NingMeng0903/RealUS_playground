"""The slack-QP velocity is the Cartesian command sent; QP1 fail latches stop."""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _guard_qpik_step_before_send,
    _guard_uncertified_brake_before_inner,
    _publish_rail_target_before_arm,
    _qpik_rail_v_ff_m_s,
    _rail_m_for_feedback,
    run_joint_admittance_phases,
    _wall_clock_rail_target,
)
from rm75_control.control.joint_admittance_8dof.wbc_rt import protocol as P
from rm75_control.control.joint_admittance_8dof.wbc_rt.client import (
    NativeWbcClient,
    classify_native_fallback,
    failed_timeout_out,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.utils import safety as safety_mod
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


def test_backend_failure_latches_stop_without_advancing_history() -> None:
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
    assert step.solver_fault_latched
    assert step.fallback_reason == "qp_failed"
    np.testing.assert_allclose(step.qdot, 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(step.q_send, q_before, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(controller.core.qdot_prev, previous, atol=1e-12, rtol=0.0)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert not sendable
    assert "qpik_fault:stop:qp_failed" == reason
    assert events == [reason]


def test_certified_overrun_is_sendable_and_does_not_stop() -> None:
    controller = _controller()
    previous = np.full(8, 0.05)
    controller.core.sync_applied(previous)
    q_before = controller.q_cmd.copy()
    controller.cfg.qp.max_solve_ms = 1.0e-9
    step = controller.update(
        np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]), q_meas=Q_SAFE
    )
    assert step.fallback_level != "stop"
    assert not step.solver_fault_latched
    assert step.qp_solver_overrun
    assert np.linalg.norm(step.qdot) > 0.0
    assert not np.allclose(step.q_send, q_before, atol=1e-12)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert events == []

    controller.cfg.qp.max_solve_ms = 50.0
    step2 = controller.update(
        np.array([0.01, -0.006, 0.004, 0.0, 0.0, 0.0]), q_meas=Q_SAFE
    )
    assert step2.fallback_level == "none"
    assert not step2.solver_fault_latched
    sendable2, reason2 = _guard_qpik_step_before_send(step2, events.append)
    assert sendable2
    assert reason2 == ""
    assert events == []


def test_zero_twist_remains_sendable() -> None:
    controller = _controller()
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert sendable
    assert reason == ""
    assert not step.solver_fault_latched
    assert events == []


def test_empty_velocity_box_latches_stop_without_publish() -> None:
    controller = _controller()
    previous = np.full(8, 50.0)
    controller.core.sync_applied(previous)
    q_before = controller.q_cmd.copy()
    step = controller.update(np.zeros(6), q_meas=Q_SAFE)
    assert step.fallback_level == "stop"
    assert step.solver_fault_latched
    assert step.fallback_reason == "qp_failed"
    assert str(step.qp1_status) == "box_infeasible"
    assert int(step.failure_code) == P.FAILURE_BOX_INFEASIBLE
    np.testing.assert_allclose(step.qdot, 0.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(step.q_send, q_before, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(controller.core.qdot_prev, previous, atol=1e-12, rtol=0.0)
    events: list[str] = []
    sendable, reason = _guard_qpik_step_before_send(step, events.append)
    assert not sendable
    assert reason == "qpik_fault:stop:qp_failed"
    assert events == [reason]


def test_native_fallback_uncertified_and_timeout_stay_stop() -> None:
    for fallback_u, failure_code in (
        (P.FALLBACK_STOP, P.FAILURE_QP1_STATUS),
        (P.FALLBACK_STOP, P.FAILURE_QP1_CERTIFICATE),
        (P.FALLBACK_STOP, P.FAILURE_BOX_INFEASIBLE),
        (P.FALLBACK_NONE, P.FAILURE_FINAL_CERTIFICATE),
        (P.FALLBACK_NONE, P.FAILURE_NONE),
        (P.FALLBACK_STOP, P.FAILURE_SOLVE_OVERRUN),
    ):
        level, reason, latched = classify_native_fallback(
            fallback_u, failure_code, native_failed=True
        )
        assert level == "stop"
        assert latched
        assert reason
    level, _, latched = classify_native_fallback(
        P.FALLBACK_NONE, P.FAILURE_SOLVE_OVERRUN, native_failed=False
    )
    assert (level, latched) == ("none", False)
    o = failed_timeout_out()
    assert int(o["status"]) == P.STATUS_FAIL
    assert int(o["flags"]) & P.OUT_FAILED
    update_src = inspect.getsource(NativeWbcClient.update)
    assert "failed_timeout_out()" in update_src


def test_no_qdot_decay_publish_regression() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    inner = (root / "native" / "wbc_rt" / "src" / "inner.cpp").read_text()
    builder = (
        root
        / "rm75_control"
        / "control"
        / "joint_admittance_8dof"
        / "solver"
        / "qp_builder.py"
    ).read_text()
    loop = (
        root / "rm75_control" / "control" / "joint_admittance_8dof" / "loop.py"
    ).read_text()
    assert "fail_qdot_decay" not in inner
    assert "fail_qdot_decay" not in builder
    assert "fail_qdot_decay" not in loop
    assert "qp1_decay" not in loop


def test_numerical_failure_is_not_sendable() -> None:
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
    assert step.fallback_reason == "qp_failed"
    assert step.fallback_level == "stop"
    assert step.solver_fault_latched
    assert events == [reason]


def test_direct_joint_ptp_uses_certified_qp_backend() -> None:
    controller = _controller()
    controller.set_direct_joint_ptp(True)
    calls_before = controller.core.solve_count
    target = np.full(8, 0.001)
    target[0] = 5.0e-5
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, qdot_ff=target
    )
    assert controller.core.solve_count == calls_before + 1
    assert step.controller_mode == "direct_joint_ptp"
    assert step.qp_solver_call_count == calls_before + 1
    assert not step.solver_fault_latched
    np.testing.assert_allclose(step.qdot, target, atol=1.0e-6, rtol=0.0)
    controller.reset(Q_SAFE)
    assert not controller._direct_joint_ptp
    assert not controller._plan_drives_rail


def test_direct_joint_ptp_out_of_box_fails_closed() -> None:
    controller = _controller()
    controller.set_direct_joint_ptp(True)
    previous = np.full(8, 0.002)
    controller.core.sync_applied(previous)
    q_before = controller.q_cmd.copy()
    # The first-tick acceleration/jerk box cannot reach this target.
    step = controller.update(
        np.zeros(6), q_meas=Q_SAFE, qdot_ff=np.full(8, 0.05)
    )
    assert step.solver_fault_latched
    assert step.fallback_level == "stop"
    assert step.fallback_reason == "qp_failed"
    np.testing.assert_allclose(step.qdot, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(step.q_send, q_before, atol=1.0e-12)
    np.testing.assert_allclose(controller.core.qdot_prev, previous, atol=1.0e-12)


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


def test_uncertified_brake_same_tick_publishes_no_new_axis_target() -> None:
    """The brake boundary precedes IK, rail target, and arm CANFD send."""
    outer = SimpleNamespace(
        controller=SimpleNamespace(shield_uncertified_brake=True)
    )
    events: list[str] = []
    sendable, reason = _guard_uncertified_brake_before_inner(
        outer,
        lambda why: events.append(f"fault_stop:{why}"),
    )
    if sendable:
        events.extend(("inner.update", "rail.set_target_m", "arm.canfd"))

    assert not sendable
    assert reason == "uncertified_brake"
    assert events == ["fault_stop:uncertified_brake"]


def test_uncertified_brake_guard_precedes_all_normal_tick_publication() -> None:
    source = inspect.getsource(run_joint_admittance_phases)
    guard = source.index("_guard_uncertified_brake_before_inner(")
    inner_update = source.index("step = inner.update(")
    rail_publish = source.index("_publish_rail_target_before_arm(", inner_update)
    arm_publish = source.index("_send_joint_canfd_cmd(", inner_update)
    assert guard < inner_update < rail_publish < arm_publish


def test_invalid_enabled_rail_feedback_never_falls_back_to_q_cmd() -> None:
    rail = SimpleNamespace(enabled=True, measured_m=float("nan"))
    inner = SimpleNamespace(q_cmd=np.array([0.4]))
    with np.testing.assert_raises_regex(RuntimeError, "non-finite"):
        _rail_m_for_feedback(rail, inner)


def test_post_solve_safety_limiter_is_gone_watchdog_remains() -> None:
    assert not hasattr(safety_mod, "SafetyLimiter")
    assert not hasattr(safety_mod, "SafetyReport")
    assert hasattr(safety_mod, "Watchdog")
    assert hasattr(safety_mod, "SafetyLimits")


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


def test_qpik_rail_v_ff_is_ik_qdot_not_pad_bypass() -> None:
    assert _qpik_rail_v_ff_m_s(0.079) == pytest.approx(0.079)
    assert _qpik_rail_v_ff_m_s(0.0) == 0.0
    assert _qpik_rail_v_ff_m_s(5.0e-4) == pytest.approx(5.0e-4)
    assert _qpik_rail_v_ff_m_s(float("nan")) == 0.0
    assert _qpik_rail_v_ff_m_s(float("inf")) == 0.0


def test_wall_clock_rail_target_does_not_add_one_tick_lead() -> None:
    pub = _wall_clock_rail_target(
        0.4004,
        0.08,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
    )
    assert pub == pytest.approx(0.4004)
    clamped = _wall_clock_rail_target(
        0.50,
        0.08,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
        meas_m=0.40,
        lead_max_m=0.020,
    )
    assert clamped == pytest.approx(0.42)


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


def test_wall_clock_idle_still_clamps_far_command() -> None:
    """Stationary publish must clamp meas±lead (0.28 m idle skip is gone)."""
    parked = _wall_clock_rail_target(
        0.70,
        0.0,
        0.0065,
        0.005,
        soft_lo=0.025,
        soft_hi=0.78,
        meas_m=0.40,
        lead_max_m=0.020,
    )
    assert parked == pytest.approx(0.42)


def test_arm_and_rail_integrate_on_wall_dt() -> None:
    dt = 0.005
    dt_wall = 0.010
    dt_int = min(dt_wall, 1.25 * dt)
    qdot_ff = np.zeros(8)
    qdot_ff[0] = 0.001
    qdot_ff[2] = 0.001

    wall = _controller()
    wall.set_direct_joint_ptp(True)
    q0 = wall.q_cmd.copy()
    step = wall.update(
        np.zeros(6), dt, q_meas=q0, qdot_ff=qdot_ff, dt_wall_s=dt_wall
    )
    # Direct FF is a tracked QP preference, not an unchecked velocity write.
    # The published command integrates the certified qdot on the clipped wall
    # period and remains inside the same hard box used by the solver.
    np.testing.assert_allclose(wall.q_cmd, q0 + step.qdot * dt_int, atol=1.0e-9)
    assert np.all(step.qdot >= wall.core.last_lo_box - 1.0e-9)
    assert np.all(step.qdot <= wall.core.last_hi_box + 1.0e-9)
    assert step.qp_solver_call_count == 1
    assert np.all(wall.q_cmd >= wall.limits.q_lower - 1.0e-9)
    assert np.all(wall.q_cmd <= wall.limits.q_upper + 1.0e-9)


def test_qp_tick_integrates_on_wall_dt_and_stays_in_position_box() -> None:
    controller = _controller()
    q0 = controller.q_cmd.copy()
    dt_nom = float(controller.cfg.dt)
    dt_wall = 1.6 * dt_nom
    dt_int = min(dt_wall, 1.25 * dt_nom)
    twist = np.array([0.0, 0.04, 0.0, 0.0, 0.0, 0.0])
    step = controller.update(
        twist, dt_nom, q_meas=q0, vel_ff=twist, dt_wall_s=dt_wall
    )
    np.testing.assert_allclose(
        controller.q_cmd,
        q0 + np.asarray(step.qdot, dtype=float) * dt_int,
        atol=1.0e-8,
        rtol=0.0,
    )
    margin = np.asarray(controller.limits.position_margin, dtype=float)
    assert np.all(controller.q_cmd >= controller.limits.q_lower + margin - 1.0e-6)
    assert np.all(controller.q_cmd <= controller.limits.q_upper - margin + 1.0e-6)


def test_zero_v_cmd_does_not_invent_rail_task() -> None:
    controller = _controller()
    q = Q_SAFE.copy()
    controller.reset(q)
    controller.centering_task.set_q_target(q)
    d_now = float(controller.kin.fk_placement(q).translation[1]) - float(q[0])
    if controller.posture_retarget is not None:
        controller.posture_retarget._d_star = d_now
        controller.posture_retarget.d_star_m = d_now
        controller.posture_retarget._d_center_target = d_now
        controller.posture_retarget.cfg.d_attr_m = d_now
    if controller.rail_ext_task is not None:
        controller.rail_ext_task.set_d_pref(d_now)
    step = controller.update(np.zeros(6), q_meas=q)
    assert not np.isfinite(step.rail_task_vel) or abs(float(step.rail_task_vel)) < 1e-3
    assert abs(float(step.qdot[0])) < 0.01
    assert _qpik_rail_v_ff_m_s(float(step.qdot[0])) == pytest.approx(
        float(step.qdot[0])
    )


def test_lead_clamp_does_not_invent_qdot_above_vmax() -> None:
    """QP velocity box keeps ``qdot`` inside ``v_max``; publish clamp holds 20 mm.

    Post-solve ``q_cmd`` is no longer lead-clamped.  The remaining bound is
    ``_wall_clock_rail_target``, including idle ticks.
    """
    controller = _controller()
    q_meas = Q_SAFE.copy()
    q_meas[0] = 0.40
    controller.reset(q_meas)
    controller.q_cmd[0] = 0.50
    twist = np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0])
    step = controller.update(twist, q_meas=q_meas)
    v_max = float(controller.limits.v_max[0])
    assert abs(float(step.qdot[0])) <= v_max + 1e-9
    lead = float(controller.cfg.resync_err_rail_m)
    published = _wall_clock_rail_target(
        0.50,
        0.08,
        0.0065,
        0.005,
        soft_lo=0.0,
        soft_hi=0.8,
        meas_m=0.40,
        lead_max_m=lead,
    )
    assert published == pytest.approx(0.40 + lead)


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

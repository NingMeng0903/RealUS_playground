"""Deterministic integration timebase and slim command-step clamp."""

from __future__ import annotations

import os

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.loop import (
    JointIkConfig,
    JointIkController,
    _CStateGuard,
    _pin_control_cpu,
    _set_gil_switch_interval,
    _set_realtime_priority,
    filter_q_meas,
    reference_time_step,
)
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics, full_q_from_arm
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    clamp_command_step,
    integration_period,
)


Q_SAFE = full_q_from_arm(
    np.deg2rad([5.0, -30.0, 10.0, 60.0, -5.0, 45.0, 0.0]), 0.40
)


def test_integration_period_clips_overrun_and_short_ticks() -> None:
    dt_nom = 0.007
    assert integration_period(dt_nom, None) == pytest.approx(dt_nom)
    assert integration_period(dt_nom, 0.006) == pytest.approx(dt_nom)
    assert integration_period(dt_nom, 0.008) == pytest.approx(0.008)
    assert integration_period(dt_nom, 0.020) == pytest.approx(1.25 * dt_nom)


def test_command_step_clamp_bounds_ddq() -> None:
    q_prev = np.zeros(3)
    dq_prev = np.array([0.001, 0.0, 0.0])
    a_max = np.array([3.0, 3.0, 3.0])
    dt_nom = 0.007
    q_desired = q_prev + np.array([0.050, 0.0, 0.0])
    q_safe, dq, acc_clamped = clamp_command_step(
        q_prev, q_desired, dq_prev, a_max, dt_nom
    )
    assert acc_clamped
    assert abs(dq[0] - dq_prev[0]) <= a_max[0] * dt_nom * dt_nom + 1.0e-15
    assert q_safe[0] == pytest.approx(dq[0])


def _ptp_controller() -> JointIkController:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.005,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        a_max_arm_rad_s2=3.0,
        a_max_rail_m_s2=0.60,
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    controller.set_direct_joint_ptp(True)
    return controller


def test_jittering_dt_wall_keeps_integration_timebase_constant() -> None:
    controller = _ptp_controller()
    dt_nom = float(controller.cfg.dt)
    qdot_ff = np.zeros(8)
    qdot_ff[2] = 0.010
    walls = [0.004, 0.006, 0.012, 0.005, 0.009]
    expected = [integration_period(dt_nom, w) for w in walls]
    q = controller.q_cmd.copy()
    travelled = 0.0
    for dt_wall, dt_int in zip(walls, expected):
        q_before = controller.q_cmd.copy()
        step = controller.update(
            np.zeros(6),
            dt_nom,
            q_meas=q,
            qdot_ff=qdot_ff,
            dt_wall_s=dt_wall,
        )
        dq2 = float(controller.q_cmd[2] - q_before[2])
        assert dq2 == pytest.approx(qdot_ff[2] * dt_int, abs=1.0e-9)
        travelled += dq2
        q = controller.q_cmd.copy()
        assert step.qdot[2] == pytest.approx(qdot_ff[2], abs=1.0e-9)
    assert travelled == pytest.approx(qdot_ff[2] * sum(expected), abs=1.0e-9)
    assert travelled < qdot_ff[2] * sum(walls) - 1.0e-9


def test_box_periods_use_nominal_dt_not_wall() -> None:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.007,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        a_max_arm_rad_s2=3.0,
        a_max_rail_m_s2=0.60,
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    seen: list[float] = []
    real = controller._measure_box_periods

    def _wrap(dt: float) -> tuple[float, float | None]:
        seen.append(float(dt))
        return real(dt)

    controller._measure_box_periods = _wrap  # type: ignore[method-assign]
    controller.update(
        np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0]),
        0.007,
        q_meas=Q_SAFE,
        dt_wall_s=0.014,
    )
    assert seen == [0.007]


def test_rt_helpers_are_best_effort() -> None:
    assert _set_realtime_priority() in (True, False)
    assert _pin_control_cpu(None) is False
    try:
        before = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        before = None
    assert _pin_control_cpu(0) in (True, False)
    if before is not None:
        try:
            os.sched_setaffinity(0, before)
        except (OSError, PermissionError, AttributeError):
            pass
    with _CStateGuard() as guard:
        assert isinstance(guard.active, bool)


def test_yaml_dt_ms_is_5ms() -> None:
    from pathlib import Path

    import yaml

    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "joint_admittance_8dof.yaml").read_text()
    )
    assert float(raw["timing"]["dt_ms"]) == pytest.approx(5.0)
    assert raw["inner"]["post_qp_step_clamp"] is True
    assert raw["timing"]["control_cpu"] is None
    assert float(raw["inner"]["qp"]["nullspace_vel_damp"]) == pytest.approx(0.0)
    assert float(raw["hw"]["lw100"]["poll_hz"]) == pytest.approx(60.0)
    assert float(raw["inner"]["arm_angle"]["engage_s"]) == pytest.approx(0.0)
    assert float(raw["inner"]["nullspace"]["engage_s"]) == pytest.approx(0.0)
    assert raw["inner"]["qmeas_filter"] == "raw"
    assert raw["inner"]["qp_geometry_source"] == "cmd"
    assert float(raw["timing"]["gil_switch_interval_ms"]) == pytest.approx(0.5)


def test_reference_time_step_follows_elapsed_wall() -> None:
    assert reference_time_step(0.00664, 1.0) == pytest.approx(0.00664)
    assert reference_time_step(0.00664, 0.5) == pytest.approx(0.00332)
    assert reference_time_step(0.0, 1.0) == pytest.approx(0.0)
    assert reference_time_step(float("nan"), 1.0) == pytest.approx(0.0)


def test_post_qp_step_clamp_off_does_not_rewrite_command() -> None:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.005,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        a_max_arm_rad_s2=3.0,
        a_max_rail_m_s2=0.60,
        post_qp_step_clamp=False,
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    controller.set_direct_joint_ptp(True)
    dt_nom = float(controller.cfg.dt)
    q = controller.q_cmd.copy()
    slow = np.zeros(8)
    slow[2] = 0.002
    controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=slow)
    q = controller.q_cmd.copy()
    jump = np.zeros(8)
    jump[2] = 0.80
    step = controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=jump)
    assert not step.acc_clamped
    assert step.qdot[2] == pytest.approx(0.80, abs=1.0e-9)


def test_period_ladder_holds_until_slack_passes() -> None:
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
    assert mod.next_period_ms(7.0, 0.98) == pytest.approx(7.0)
    assert mod.next_period_ms(7.0, 0.99) == pytest.approx(6.0)
    assert mod.next_period_ms(6.0, 0.99) == pytest.approx(5.0)
    assert mod.next_period_ms(5.0, 1.0) == pytest.approx(5.0)
    assert mod.raise_period_ms(5.0) == pytest.approx(7.0)
    assert mod.PERIOD_LADDER_MS == (7.0, 6.0, 5.0)


def test_step_clamp_flags_acc_clamped_on_large_qdot_jump() -> None:
    controller = _ptp_controller()
    dt_nom = float(controller.cfg.dt)
    q = controller.q_cmd.copy()
    slow = np.zeros(8)
    slow[2] = 0.002
    controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=slow)
    q = controller.q_cmd.copy()
    jump = np.zeros(8)
    jump[2] = 0.80
    step = controller.update(np.zeros(6), dt_nom, q_meas=q, qdot_ff=jump)
    assert step.acc_clamped
    ddq = abs(step.qdot[2] * dt_nom - slow[2] * dt_nom)
    assert ddq <= controller.limits.a_max[2] * dt_nom * dt_nom + 1.0e-9


def test_box_period_does_not_open_on_long_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _ptp_controller()
    controller._box_dt_last_t = 0.0
    controller._box_h1_last = 0.005
    monkeypatch.setattr(
        "rm75_control.control.joint_admittance_8dof.loop.time.monotonic",
        lambda: 0.0109,
    )
    h1, prev = controller._measure_box_periods(0.005)
    assert prev == pytest.approx(0.005)
    assert h1 == pytest.approx(0.005)


def test_set_gil_switch_interval_accepts_positive_ms() -> None:
    import sys

    before = sys.getswitchinterval()
    try:
        assert _set_gil_switch_interval(0.5) is True
        assert sys.getswitchinterval() == pytest.approx(0.0005)
        assert _set_gil_switch_interval(0.0) is False
        assert _set_gil_switch_interval(-1.0) is False
        assert _set_gil_switch_interval(float("nan")) is False
    finally:
        sys.setswitchinterval(before)


def test_qp_geometry_source_cmd_uses_command_not_measurement() -> None:
    qp = QpConfig(backend="proxqp", collision=CollisionConfig(enabled=False))
    cfg = JointIkConfig(
        dt=0.005,
        control_frame="base",
        qp=qp,
        collision=CollisionConfig(enabled=False),
        qp_use_cpp_kernel=False,
        qp_geometry_source="cmd",
    )
    controller = JointIkController(RobotKinematics(), cfg)
    controller.reset(Q_SAFE)
    seen: list[np.ndarray] = []
    real = controller.kin.jacobian

    def _wrap(q):
        seen.append(np.asarray(q, dtype=float).copy())
        return real(q)

    controller.kin.jacobian = _wrap  # type: ignore[method-assign]
    q_before = controller.q_cmd.copy()
    q_meas = q_before.copy()
    q_meas[4] += 0.05
    controller.update(
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        0.005,
        q_meas=q_meas,
    )
    assert seen
    np.testing.assert_allclose(seen[0], q_before)
    controller.cfg.qp_geometry_source = "meas"
    seen.clear()
    q_before = controller.q_cmd.copy()
    q_meas = q_before.copy()
    q_meas[4] += 0.05
    controller.update(
        np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        0.005,
        q_meas=q_meas,
    )
    assert seen
    np.testing.assert_allclose(seen[0], q_meas)
    controller.cfg.qp_geometry_source = "nope"
    with pytest.raises(ValueError, match="qp_geometry_source"):
        controller._qp_geometry_state(q_before, q_meas)


def test_filter_q_meas_hold_keeps_stale_frame() -> None:
    q0 = np.linspace(0.1, 0.8, 8)
    q1 = q0.copy()
    q1[4] += 1.7e-5
    filt, raw, kept = filter_q_meas(
        q0, mode="hold", dt=0.005, prev_raw=None, prev_filt=None
    )
    np.testing.assert_allclose(filt, q0)
    filt2, _, _ = filter_q_meas(
        q0, mode="hold", dt=0.005, prev_raw=raw, prev_filt=kept
    )
    np.testing.assert_allclose(filt2, q0)
    filt3, _, kept3 = filter_q_meas(
        q1, mode="hold", dt=0.005, prev_raw=q0, prev_filt=q0
    )
    np.testing.assert_allclose(filt3, q1)
    stale, _, _ = filter_q_meas(
        q1, mode="hold", dt=0.005, prev_raw=q1, prev_filt=kept3
    )
    np.testing.assert_allclose(stale, q1)


def test_filter_q_meas_lowpass_lags_a_step() -> None:
    q0 = np.zeros(8)
    q1 = np.ones(8)
    filt, _, kept = filter_q_meas(
        q0, mode="lowpass", dt=0.005, prev_raw=None, prev_filt=None, lowpass_hz=25.0
    )
    np.testing.assert_allclose(filt, q0)
    filt2, _, _ = filter_q_meas(
        q1, mode="lowpass", dt=0.005, prev_raw=q0, prev_filt=kept, lowpass_hz=25.0
    )
    assert float(filt2[0]) > 0.0
    assert float(filt2[0]) < 1.0


def test_latency_histogram_splits_the_write_tail() -> None:
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "apps"
        / "joint_admittance_8dof"
        / "analyze_qpik_quality.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_qpik_quality_hist", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    text = mod._latency_histogram(np.array([0.0, 1.0, 6.0, 17.0, 22.0]))
    assert "0-2:40%" in text
    assert ">=20:20%" in text
    t = np.linspace(0.0, 4.0, 800)
    q = 0.01 * np.sin(2.0 * np.pi * 4.0 * t)
    spec = mod._cmd_accel_spectrum(q, 200.0)
    assert spec is not None
    peak, hf = spec
    assert 2.0 < peak < 10.0
    assert hf < 0.20

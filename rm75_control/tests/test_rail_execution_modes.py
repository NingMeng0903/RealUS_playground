"""Unit tests for explicit rail command semantics and feedback snapshots."""

from __future__ import annotations

import csv
import math
import time

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    _RailCsvLogger,
    RailCommandMode,
    RailExecutionFeedback,
    RailServoBridge,
    RailServoConfig,
    RailServoSample,
)


def test_rail_csv_schema_keeps_feedback_and_mode_columns_aligned(tmp_path) -> None:
    """Every queued rail event must match the append-only CSV schema."""

    path = tmp_path / "rail.csv"
    logger = _RailCsvLogger(str(path))
    logger.write(
        event="sample",
        motion_seq=17,
        feedback_valid=True,
        command_mode=RailCommandMode.COUPLED_VELOCITY.value,
    )
    logger.close()

    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    header, values = rows
    assert len(header) == len(values)
    assert len(header) == len(_RailCsvLogger._HEADER)  # noqa: SLF001
    assert len(set(header)) == len(header)
    fields = dict(zip(header, values, strict=True))
    assert fields["motion_seq"] == "17"
    assert fields["feedback_valid"] == "1"
    assert fields["command_mode"] == RailCommandMode.COUPLED_VELOCITY.value


def _armed_bridge() -> RailServoBridge:
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    bridge._calibrated = True  # noqa: SLF001
    bridge._armed = True  # noqa: SLF001
    bridge._measured_m = 0.40  # noqa: SLF001
    bridge._measured_mono_s = time.monotonic()  # noqa: SLF001
    return bridge


def test_velocity_command_is_explicit_and_position_is_backward_compatible() -> None:
    bridge = _armed_bridge()

    assert bridge.set_target_m(0.45, v_ff_m_s=0.08)
    command = bridge.command
    assert command.mode is RailCommandMode.COUPLED_VELOCITY
    assert command.v_ff_m_s == pytest.approx(0.08)
    assert bridge.command_mode is RailCommandMode.COUPLED_VELOCITY

    # The old two-argument position API remains a position command.
    assert bridge.set_target_m(0.46)
    assert bridge.command_mode is RailCommandMode.POSITION
    assert math.isnan(bridge.target_v_ff_m_s)


def test_explicit_velocity_mode_without_feedforward_means_zero_velocity() -> None:
    bridge = _armed_bridge()
    assert bridge.set_target_m(0.40, mode="coupled_velocity")
    command = bridge.command
    assert command.mode is RailCommandMode.COUPLED_VELOCITY
    assert command.v_ff_m_s == pytest.approx(0.0)


def test_velocity_reference_does_not_position_catch_up() -> None:
    # A stale integrated target at 0.55 m must not move a stopped coupled
    # reference at 0.40 m when the authoritative velocity is zero.
    x_new, v_new, a_new = RailServoBridge._step_velocity_reference(
        0.40,
        0.0,
        0.0,
        dt=0.02,
        v_max=0.30,
        a_max=0.80,
        x_min=0.005,
        x_max=0.78,
    )
    assert x_new == pytest.approx(0.40)
    assert v_new == pytest.approx(0.0)
    assert a_new == pytest.approx(0.0)


@pytest.mark.parametrize(
    "v_des,v_prev,v_meas,expected",
    [
        (-0.012, 0.006, 0.020, 0.0),
        (0.012, -0.006, -0.020, 0.0),
        (0.004, 0.006, 0.020, 0.004),
        (-0.004, -0.006, -0.020, -0.004),
    ],
)
def test_zero_velocity_brake_never_commands_a_reversal(
    v_des: float, v_prev: float, v_meas: float, expected: float
) -> None:
    got = RailServoBridge._clamp_zero_target_brake(
        v_des,
        v_goal=0.0,
        v_ref=0.0,
        v_meas=v_meas,
        v_prev_cmd=v_prev,
    )
    assert got == pytest.approx(expected)


def test_explicit_reverse_velocity_is_not_blocked_by_stop_brake_guard() -> None:
    got = RailServoBridge._clamp_zero_target_brake(
        -0.012,
        v_goal=-0.020,
        v_ref=0.004,
        v_meas=0.020,
        v_prev_cmd=0.006,
    )
    assert got == pytest.approx(-0.012)


def test_execution_feedback_is_immutable_and_reports_freshness() -> None:
    bridge = _armed_bridge()
    stamp = time.monotonic() - 0.001
    bridge._servo_sample = RailServoSample(  # noqa: SLF001
        sample_mono_s=stamp,
        motion_seq=7,
        feedback_valid=True,
        x_meas_m=0.401,
        v_meas_m_s=0.012,
        v_cmd_m_s=0.020,
        a_cmd_m_s2=0.30,
        command_mode=RailCommandMode.COUPLED_VELOCITY.value,
        follow=True,
        armed=True,
    )
    feedback = bridge.execution_feedback
    assert isinstance(feedback, RailExecutionFeedback)
    assert feedback.position_m == pytest.approx(0.401)
    assert feedback.v_meas_m_s == pytest.approx(0.012)
    assert feedback.v_cmd_m_s == pytest.approx(0.020)
    assert feedback.a_cmd_m_s2 == pytest.approx(0.30)
    assert feedback.command_mode is RailCommandMode.COUPLED_VELOCITY
    assert feedback.motion_seq == 7
    assert feedback.is_fresh(0.05)
    with pytest.raises(AttributeError):
        feedback.v_cmd_m_s = 0.0  # type: ignore[misc]

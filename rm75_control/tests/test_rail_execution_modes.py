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
    assert "v_enc_m_s" in fields
    assert "v_reg_m_s" in fields
    assert "v_meas_m_s" not in fields


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


def test_brake_guard_blocks_leaked_reverse_while_goal_still_nonzero() -> None:
    """Replay t=6.54s +48 ms: reverse leaked while v_goal was still 9.4 mm/s.

    After that leak, v_prev_cmd was already negative so the old guard
    (direction from previous command, only when |v_goal|<1e-3) let the
    reverse through.  Motion is still +32.5 mm/s.
    """
    got = RailServoBridge._clamp_zero_target_brake(
        -0.0169,
        v_goal=0.0094,
        v_ref=0.0,
        v_meas=0.0325,
        v_prev_cmd=-0.0023,
    )
    assert got == pytest.approx(0.0)


def test_brake_guard_uses_motion_not_prev_cmd_after_goal_reaches_zero() -> None:
    got = RailServoBridge._clamp_zero_target_brake(
        -0.0169,
        v_goal=0.0,
        v_ref=0.0,
        v_meas=0.0325,
        v_prev_cmd=-0.0023,
    )
    assert got == pytest.approx(0.0)


def test_encoder_velocity_matches_bounded_difference() -> None:
    hz = 60.0
    dt = 1.0 / hz
    samples = [(k * dt, 0.030 * k * dt) for k in range(4)]
    got, source = RailServoBridge._encoder_velocity(
        samples, poll_hz=hz, fallback_m_s=0.99
    )
    assert got == pytest.approx(0.030, abs=1e-9)
    assert source == "lsq"


def test_encoder_velocity_repeated_samples_are_zero_not_a_spike() -> None:
    hz = 60.0
    dt = 1.0 / hz
    samples = [(0.0, 0.40), (dt, 0.40), (2.0 * dt, 0.40)]
    got, source = RailServoBridge._encoder_velocity(
        samples, poll_hz=hz, fallback_m_s=0.99
    )
    assert got == pytest.approx(0.0)
    assert source == "lsq"


def test_encoder_velocity_out_of_range_interval_falls_back() -> None:
    hz = 60.0
    period = 1.0 / hz
    stale = [(0.0, 0.40), (8.0 * period, 0.41)]
    assert RailServoBridge._encoder_velocity(
        stale, poll_hz=hz, fallback_m_s=0.012
    ) == (pytest.approx(0.012), "reg")
    bunched = [(0.0, 0.40), (0.1 * period, 0.41)]
    assert RailServoBridge._encoder_velocity(
        bunched, poll_hz=hz, fallback_m_s=0.012
    ) == (pytest.approx(0.012), "reg")
    assert RailServoBridge._encoder_velocity(
        [(0.0, 0.40)], poll_hz=hz, fallback_m_s=0.012
    ) == (pytest.approx(0.012), "reg")


def test_encoder_velocity_survives_56hz_jitter_against_a_60hz_config() -> None:
    # The worker asked for 60 Hz and got 56 with 18/26 ms jitter.  A fixed
    # 3/poll_hz window rejected 11.2% of ticks and hard-switched the D term
    # back to the 157 ms-lagged register.
    dt_pattern = [0.018, 0.026, 0.018, 0.026, 0.018]
    t = 0.0
    samples = [(0.0, 0.40)]
    for dt in dt_pattern:
        t += dt
        samples.append((t, 0.40 + 0.030 * t))
    measured_period = 1.0 / 56.0
    for k in range(2, len(samples) + 1):
        got, source = RailServoBridge._encoder_velocity(
            samples[:k],
            poll_hz=60.0,
            fallback_m_s=0.99,
            period_s=measured_period,
        )
        assert source == "lsq"
        assert got == pytest.approx(0.030, abs=1e-6)


def test_encoder_velocity_holds_through_a_dropped_poll_before_the_register() -> None:
    hz = 60.0
    period = 1.0 / hz
    gap = [(0.0, 0.40), (9.0 * period, 0.41)]
    held, source = RailServoBridge._encoder_velocity(
        gap, poll_hz=hz, fallback_m_s=0.012, hold_m_s=0.031, hold_budget=2
    )
    assert held == pytest.approx(0.031)
    assert source == "hold"
    # Budget spent, or nothing to hold: the register value is the last resort.
    assert RailServoBridge._encoder_velocity(
        gap, poll_hz=hz, fallback_m_s=0.012, hold_m_s=0.031, hold_budget=0
    ) == (pytest.approx(0.012), "reg")
    assert RailServoBridge._encoder_velocity(
        gap, poll_hz=hz, fallback_m_s=0.012, hold_m_s=float("nan"), hold_budget=2
    ) == (pytest.approx(0.012), "reg")


def test_encoder_velocity_slope_rejects_timestamp_jitter() -> None:
    # Same constant 30 mm/s, but every other stamp is late by 4 ms.  A
    # two-point difference alternates 15/60 mm/s on this; the slope must not.
    hz = 60.0
    period = 1.0 / hz
    samples = []
    for k in range(6):
        skew = 0.004 if k % 2 else 0.0
        t = k * period + skew
        samples.append((t, 0.030 * (k * period)))
    got, source = RailServoBridge._encoder_velocity(
        samples, poll_hz=hz, fallback_m_s=0.99, period_s=period
    )
    assert source == "lsq"
    assert got == pytest.approx(0.030, rel=0.15)


def test_encoder_velocity_tracks_a_constant_acceleration_ramp() -> None:
    hz = 60.0
    period = 1.0 / hz
    accel = 0.8
    samples = [(k * period, 0.5 * accel * (k * period) ** 2) for k in range(5)]
    got, source = RailServoBridge._encoder_velocity(
        samples, poll_hz=hz, fallback_m_s=0.99, period_s=period
    )
    assert source == "lsq"
    # Least squares over the window reports the mid-window velocity.
    t_mid = 0.5 * (samples[0][0] + samples[-1][0])
    assert got == pytest.approx(accel * t_mid, rel=1e-6)


def test_brake_position_term_was_merged_into_zero_target_brake() -> None:
    assert not hasattr(RailServoBridge, "_clamp_brake_position_term")


def test_zero_target_brake_covers_leaked_position_reverse() -> None:
    """``_clamp_zero_target_brake`` is the remaining stop-stack reverse guard.

    The old P-only clamp is gone; the same leak (x_ref stopped, measured
    already past it) must still be zeroed, while same-sign trim and an
    explicit reverse goal stay open.
    """
    v_des = 14.0 * (-0.00069)
    got = RailServoBridge._clamp_zero_target_brake(
        v_des, v_goal=0.0, v_ref=0.0, v_meas=0.018, v_prev_cmd=0.0
    )
    assert got == pytest.approx(0.0)
    assert RailServoBridge._clamp_zero_target_brake(
        0.004, v_goal=0.0, v_ref=0.0, v_meas=0.018, v_prev_cmd=0.0
    ) == pytest.approx(0.004)
    assert RailServoBridge._clamp_zero_target_brake(
        -0.010, v_goal=-0.020, v_ref=0.0, v_meas=0.018, v_prev_cmd=0.0
    ) == pytest.approx(-0.010)


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


def test_velocity_reference_does_not_catch_up_when_goal_is_parked() -> None:
    x_new, v_new, a_new = RailServoBridge._step_velocity_reference(
        0.40,
        0.0,
        0.0,
        dt=0.02,
        v_max=0.12,
        a_max=0.60,
        x_goal=0.55,
        catch_v_max=0.02,
        k_catch=5.0,
        x_min=0.005,
        x_max=0.78,
    )
    assert x_new == pytest.approx(0.40)
    assert v_new == pytest.approx(0.0)
    assert a_new == pytest.approx(0.0)


def test_velocity_reference_catch_up_is_bounded_while_moving() -> None:
    x_new, v_new, _a_new = RailServoBridge._step_velocity_reference(
        0.40,
        0.10,
        0.10,
        dt=0.05,
        v_max=0.12,
        a_max=0.60,
        x_goal=0.55,
        catch_v_max=0.02,
        k_catch=5.0,
        x_min=0.005,
        x_max=0.78,
    )
    # 5.0 * 0.15 = 0.75, capped at catch_v_max 0.02 → v_target = 0.12
    assert v_new == pytest.approx(0.12)
    assert x_new == pytest.approx(0.40 + 0.12 * 0.05)


def test_velocity_reference_catch_up_is_zero_when_x_ref_matches_goal() -> None:
    x_new, v_new, _a_new = RailServoBridge._step_velocity_reference(
        0.40,
        0.08,
        0.08,
        dt=0.02,
        v_max=0.12,
        a_max=0.60,
        x_goal=0.40,
        catch_v_max=0.02,
        k_catch=5.0,
    )
    assert v_new == pytest.approx(0.08)
    assert x_new == pytest.approx(0.40 + 0.08 * 0.02)


def test_parked_reanchor_snaps_x_ref_to_measured_and_does_not_move() -> None:
    x_ref, v_ref, a_ref, parked = RailServoBridge._parked_reanchor(
        0.3114,
        0.0,
        0.0,
        measured=0.2917,
        v_goal=0.0,
        v_meas=0.0,
    )
    assert parked is True
    assert x_ref == pytest.approx(0.2917)
    assert v_ref == pytest.approx(0.0)
    assert a_ref == pytest.approx(0.0)


def test_velocity_reference_catch_up_bounds_repeated_start_stop_loss() -> None:
    x_ref = 0.40
    x_goal = 0.40
    v_ref = 0.0
    dt = 0.02
    for _cycle in range(30):
        for _ in range(20):
            x_goal += 0.08 * dt
            x_ref, v_ref, _a = RailServoBridge._step_velocity_reference(
                x_ref,
                v_ref,
                0.08,
                dt=dt,
                v_max=0.12,
                a_max=0.60,
                x_goal=x_goal,
                catch_v_max=0.02,
                k_catch=5.0,
            )
        x_ref -= 0.001
        for _ in range(10):
            x_ref, v_ref, _a = RailServoBridge._step_velocity_reference(
                x_ref,
                v_ref,
                0.0,
                dt=dt,
                v_max=0.12,
                a_max=0.60,
                x_goal=x_goal,
                catch_v_max=0.02,
                k_catch=5.0,
            )
            v_ref = 0.0
    assert abs(x_goal - x_ref) < 0.005


def test_turn_catch_up_cannot_overspeed_a_1mm_s_goal() -> None:
    """Window-8 turn kick: e_shape 1.06 mm at v_goal 1.4 mm/s must stay < 1.5x."""
    v_goal = 0.0014
    x_ref = 0.40
    x_goal = x_ref + 0.00106
    _x_new, v_new, _a_new = RailServoBridge._step_velocity_reference(
        x_ref,
        0.0,
        v_goal,
        dt=0.02,
        v_max=0.12,
        a_max=0.60,
        x_goal=x_goal,
        catch_v_max=0.02,
        k_catch=5.0,
        catch_frac=0.3,
    )
    assert abs(v_new) < 1.5 * abs(v_goal)


def test_decel_request_covers_encoder_noise_gap() -> None:
    """t=5.390 cruise leak: |v_goal| 19.66 vs |v_enc| 18.81 mm/s is still brake."""
    assert RailServoBridge._is_decel_request(0.01966, 0.01881) is True
    assert RailServoBridge._is_decel_request(-0.01966, -0.01881) is True
    assert RailServoBridge._is_decel_request(0.01966, -0.01881) is False


def test_parked_reanchor_leaves_a_moving_stream_alone() -> None:
    x_ref, v_ref, a_ref, parked = RailServoBridge._parked_reanchor(
        0.40,
        0.08,
        0.0,
        measured=0.399,
        v_goal=0.08,
        v_meas=0.07,
    )
    assert parked is False
    assert x_ref == pytest.approx(0.40)
    assert v_ref == pytest.approx(0.08)

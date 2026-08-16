"""Rail online-reference tests; no hardware."""

from __future__ import annotations

import math
from collections import deque

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
    live_host_accel_m_s2,
    parse_rail_servo_config,
)
from rm75_control.hw.lw100.drive import apply_fa24_rpm_deadband


DT = 0.02
V_MAX = 0.15
A_MAX = 0.8


def _step(
    x_ref: float,
    v_ref: float,
    goal: float,
    v_goal: float = 0.0,
    *,
    stationary: bool = False,
) -> tuple[float, float, float]:
    return RailServoBridge._step_reference(
        x_ref,
        v_ref,
        goal,
        v_goal,
        stationary=stationary,
        dt=DT,
        v_max=V_MAX,
        a_max=A_MAX,
    )


def test_goal_motion_estimates_and_time_aligns_a_ramp() -> None:
    samples = [(1.000, 0.40000), (1.004, 0.40008), (1.040, 0.40080)]
    goal, velocity, stationary = RailServoBridge._estimate_goal_motion(
        samples,
        now_s=1.060,
        max_age_s=0.040,
    )
    assert goal == pytest.approx(0.40120)
    assert velocity == pytest.approx(0.020)
    assert not stationary


def test_repeated_step_becomes_a_static_goal_without_projection() -> None:
    samples = [
        (1.000, 0.400),
        (1.020, 0.410),
        (1.040, 0.410),
        (1.060, 0.410),
        (1.080, 0.410),
    ]
    goal, velocity, stationary = RailServoBridge._estimate_goal_motion(
        samples,
        now_s=1.080,
        max_age_s=0.040,
    )
    assert goal == pytest.approx(0.410)
    assert velocity == pytest.approx(0.0)
    assert stationary


@pytest.mark.parametrize("distance_m", [0.0001, 0.0005, 0.002, 0.040, 0.390])
def test_static_step_stops_without_overshoot(distance_m: float) -> None:
    x_ref = 0.0
    v_ref = 0.0
    max_x = x_ref
    signs: list[int] = []
    for _ in range(1000):
        x_ref, v_ref, _ = _step(
            x_ref,
            v_ref,
            distance_m,
            stationary=True,
        )
        max_x = max(max_x, x_ref)
        if abs(v_ref) > 1.0e-5:
            sign = 1 if v_ref > 0.0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
        if x_ref == distance_m and v_ref == 0.0:
            break
    else:
        raise AssertionError(f"reference did not settle: x={x_ref} v={v_ref}")
    assert max_x <= distance_m + 1.0e-12
    assert signs == [1]


def test_reference_respects_velocity_and_acceleration_limits() -> None:
    x_ref = 0.400
    v_ref = 0.0
    schedule = [0.400] * 5 + [0.700] * 100 + [0.200] * 120 + [0.400] * 100
    for goal in schedule:
        previous_v = v_ref
        x_ref, v_ref, a_ref = _step(x_ref, v_ref, goal, stationary=True)
        assert abs(v_ref) <= V_MAX + 1.0e-12
        assert abs(v_ref - previous_v) <= A_MAX * DT + 1.0e-12
        assert abs(a_ref) <= A_MAX + 1.0e-12


def test_200hz_ramp_is_smooth_at_the_50hz_worker() -> None:
    history: deque[tuple[float, float]] = deque(maxlen=64)
    x_ref = 0.400
    v_ref = 0.0
    worker_v: list[float] = []
    for caller_i in range(600):
        now_s = caller_i * 0.005
        history.append((now_s, 0.400 + 0.020 * now_s))
        if caller_i % 4:
            continue
        goal, v_goal, stationary = RailServoBridge._estimate_goal_motion(
            history,
            now_s=now_s,
            max_age_s=0.04,
        )
        x_ref, v_ref, _ = _step(
            x_ref,
            v_ref,
            goal,
            v_goal,
            stationary=stationary,
        )
        worker_v.append(v_ref)
    assert min(worker_v[20:]) > 0.018
    assert (
        max(abs(worker_v[i] - worker_v[i - 1]) for i in range(1, len(worker_v)))
        <= A_MAX * DT
    )


def test_irregular_target_timestamps_do_not_create_velocity_steps() -> None:
    intervals = (0.020, 0.020, 0.004, 0.036)
    events: list[tuple[float, float]] = []
    event_t = 0.0
    while event_t < 5.0:
        events.append((event_t, 0.400 + 0.020 * event_t))
        event_t += intervals[len(events) % len(intervals)]

    history: deque[tuple[float, float]] = deque(maxlen=64)
    next_event = 0
    x_ref = 0.400
    v_ref = 0.0
    for worker_i in range(250):
        now_s = worker_i * DT
        while next_event < len(events) and events[next_event][0] <= now_s + 1.0e-12:
            history.append(events[next_event])
            next_event += 1
        goal, v_goal, stationary = RailServoBridge._estimate_goal_motion(
            history,
            now_s=now_s,
            max_age_s=0.04,
        )
        previous_v = v_ref
        x_ref, v_ref, _ = _step(
            x_ref,
            v_ref,
            goal,
            v_goal,
            stationary=stationary,
        )
        assert abs(v_ref - previous_v) <= A_MAX * DT + 1.0e-12
    assert v_ref == pytest.approx(0.020, abs=0.001)


def test_scan_has_exactly_one_reference_reversal_at_each_endpoint() -> None:
    history: deque[tuple[float, float]] = deque(maxlen=64)
    x_ref = 0.400
    v_ref = 0.0
    samples: list[tuple[float, float, float]] = []
    for caller_i in range(int(45.0 / 0.005)):
        now_s = caller_i * 0.005
        goal = 0.400 + 0.040 * math.sin(2.0 * math.pi * 0.08 * now_s)
        history.append((now_s, goal))
        if caller_i % 4:
            continue
        goal_eval, v_goal, stationary = RailServoBridge._estimate_goal_motion(
            history,
            now_s=now_s,
            max_age_s=0.04,
        )
        x_ref, v_ref, _ = _step(
            x_ref,
            v_ref,
            goal_eval,
            v_goal,
            stationary=stationary,
        )
        samples.append((now_s, v_goal, v_ref))

    wrong_way = [
        row
        for row in samples
        if abs(row[1]) > 0.002
        and row[1] * row[2] < 0.0
        and abs(row[2]) > 0.0005
    ]
    assert not wrong_way
    for center_s in [3.125 + 6.25 * i for i in range(7)]:
        signs: list[int] = []
        for t_s, _, velocity in samples:
            if abs(t_s - center_s) > 0.65 or abs(velocity) <= 0.0005:
                continue
            sign = 1 if velocity > 0.0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
        assert len(signs) == 2


def test_full_scan_and_return_remain_bounded() -> None:
    def smoothstep(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3.0 - 2.0 * value)

    def target_at(t_s: float) -> float:
        if t_s < 5.0:
            return 0.400
        if t_s < 41.0:
            tr = t_s - 5.0
            fade = smoothstep(tr / 1.5) if tr < 1.5 else 1.0
            return 0.400 + 0.040 * fade * math.sin(2.0 * math.pi * 0.08 * tr)
        tr_end = 41.0 - 5.0 - 1.0e-4
        x_end = 0.400 + 0.040 * math.sin(2.0 * math.pi * 0.08 * tr_end)
        weight = smoothstep((t_s - 41.0) / 4.0)
        return x_end * (1.0 - weight) + 0.400 * weight

    history: deque[tuple[float, float]] = deque(maxlen=64)
    x_ref = x_meas = 0.400
    v_ref = v_cmd = v_meas = 0.0
    shape_errors: list[float] = []
    track_errors: list[float] = []
    goal_signs: list[int] = []
    ref_signs: list[int] = []
    for caller_i in range(int(45.0 / 0.005)):
        now_s = caller_i * 0.005
        history.append((now_s, target_at(now_s)))
        if caller_i % 4:
            continue
        goal, v_goal, stationary = RailServoBridge._estimate_goal_motion(
            history,
            now_s=now_s,
            max_age_s=0.04,
        )
        x_ref, v_ref, _ = _step(
            x_ref,
            v_ref,
            goal,
            v_goal,
            stationary=stationary,
        )
        if abs(v_goal) > 0.002:
            sign = 1 if v_goal > 0.0 else -1
            if not goal_signs or goal_signs[-1] != sign:
                goal_signs.append(sign)
        if abs(v_ref) > 0.0005:
            sign = 1 if v_ref > 0.0 else -1
            if not ref_signs or ref_signs[-1] != sign:
                ref_signs.append(sign)
        v_des = v_ref + 14.0 * (x_ref - x_meas) + 0.22 * (v_ref - v_meas)
        v_des = max(-V_MAX, min(V_MAX, v_des))
        v_cmd = max(v_cmd - A_MAX * DT, min(v_cmd + A_MAX * DT, v_des))
        v_meas += 0.5 * (v_cmd - v_meas)
        x_meas += v_meas * DT
        shape_errors.append(abs(goal - x_ref))
        track_errors.append(abs(x_ref - x_meas))

    assert len(ref_signs) <= len(goal_signs)
    assert max(shape_errors) < 0.002
    assert sorted(track_errors)[int(0.95 * (len(track_errors) - 1))] < 0.005
    assert max(track_errors) < 0.010
    assert x_ref == pytest.approx(0.400, abs=0.001)


def test_defaults_match_the_hardware_baseline() -> None:
    for cfg in (RailServoConfig(), parse_rail_servo_config({})):
        assert cfg.vel_kp == 14.0
        assert cfg.vel_kd == 0.22
        assert cfg.target_stale_coast_s == pytest.approx(0.35)
        assert cfg.soft_min_m == pytest.approx(0.015)
        assert cfg.vel_max_m_s == 0.30
        assert cfg.max_speed_rpm == 1800
        assert cfg.vel_amax_m_s2 == 0.8
        assert cfg.vel_ff_kp == 4.0
        assert cfg.vel_ff_p_trim_m_s == pytest.approx(0.010)
        assert cfg.match_drive_accel is True
        assert cfg.fa24_rpm_deadband == 12
        assert cfg.vel_deadband_mm == 0.05
        assert cfg.standstill_enter_mm == 0.05
        assert cfg.standstill_exit_mm == 0.25
        assert cfg.standstill_dwell_s == 0.08
        assert cfg.encoder_freeze_min_move_mm == 0.15
        assert cfg.accel_ms == 150
        assert cfg.decel_ms == 150
        assert cfg.scurve_ms == 30


def test_single_hitch_does_not_end_the_target_stream() -> None:
    """A 127 ms QPIK hitch must coast; only timeout+coast ends the stream."""
    cfg = RailServoConfig(target_timeout_s=0.10, target_stale_coast_s=0.35)
    dead = cfg.stream_dead_s()
    assert dead == pytest.approx(0.45)
    assert 0.127 < dead
    assert 0.50 > dead
    yaml_cfg = parse_rail_servo_config(
        {
            "hw": {
                "lw100": {
                    "target_timeout_s": 0.25,
                    "target_stale_coast_s": 0.35,
                    "vel_kp": 14.0,
                    "vel_kd": 0.22,
                }
            }
        }
    )
    assert yaml_cfg.vel_kp == 14.0
    assert yaml_cfg.vel_kd == 0.22
    assert yaml_cfg.stream_dead_s() == pytest.approx(0.60)


def test_begin_tracking_session_clears_stale_target_and_hold() -> None:
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    bridge._calibrated = True  # noqa: SLF001
    bridge._armed = True  # noqa: SLF001
    bridge._measured_m = 0.40  # noqa: SLF001
    bridge._target_m = 0.10  # noqa: SLF001
    bridge._commanded_m = 0.10  # noqa: SLF001
    bridge._last_target_rx_mono = 123.0  # noqa: SLF001
    bridge._follow_enabled = True  # noqa: SLF001
    bridge._hold_count = 7  # noqa: SLF001
    bridge._target_v_ff_m_s = 0.08  # noqa: SLF001
    bridge.begin_tracking_session()
    assert bridge._follow_enabled is False  # noqa: SLF001
    assert bridge._last_target_rx_mono == 0.0  # noqa: SLF001
    assert bridge._hold_count == 0  # noqa: SLF001
    assert bridge._target_m == pytest.approx(0.40)  # noqa: SLF001
    assert bridge._commanded_m == pytest.approx(0.40)  # noqa: SLF001
    assert math.isnan(bridge.target_v_ff_m_s)


def test_continuous_follow_does_not_freeze_tiny_v_ref() -> None:
    """COUPLED tracking must keep PD live for tiny nonzero goals (no deadband 0)."""

    # Mirror the worker gate: fresh follow + not settling ⇒ no standstill/deadband.
    follow, settling, target_stale = True, False, False
    continuous_tracking = bool(follow) and not settling and not target_stale
    assert continuous_tracking
    v_ref = 0.0005  # < 1 mm/s
    err_x = 0.00002
    deadband_m = 0.00005
    v_raw = v_ref + 14.0 * err_x
    if not continuous_tracking and abs(v_ref) < 0.001 and abs(err_x) <= deadband_m:
        v_raw = 0.0
    assert v_raw != 0.0


def test_standstill_hysteresis_enters_tight_and_wakes_wide() -> None:
    held, since = False, None
    # Quiet inside enter band starts the dwell timer.
    held, since = RailServoBridge._standstill_hold_update(
        held=held,
        enter_since_s=since,
        now_s=1.0,
        err_m=0.00004,
        v_ref_m_s=0.0,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
        enter_m=0.00005,
        exit_m=0.00025,
        dwell_s=0.08,
    )
    assert held is False
    assert since == pytest.approx(1.0)
    # After dwell → latch.
    held, since = RailServoBridge._standstill_hold_update(
        held=held,
        enter_since_s=since,
        now_s=1.09,
        err_m=0.00004,
        v_ref_m_s=0.0,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
        enter_m=0.00005,
        exit_m=0.00025,
        dwell_s=0.08,
    )
    assert held is True
    # Residual between enter and exit must NOT wake (no hum re-entry).
    held, since = RailServoBridge._standstill_hold_update(
        held=held,
        enter_since_s=since,
        now_s=1.20,
        err_m=0.00012,
        v_ref_m_s=0.0,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
        enter_m=0.00005,
        exit_m=0.00025,
        dwell_s=0.08,
    )
    assert held is True
    # Disturbance past exit wakes the loop.
    held, since = RailServoBridge._standstill_hold_update(
        held=held,
        enter_since_s=since,
        now_s=1.30,
        err_m=0.00030,
        v_ref_m_s=0.0,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
        enter_m=0.00005,
        exit_m=0.00025,
        dwell_s=0.08,
    )
    assert held is False
    # Motion reference also wakes immediately.
    held, since = RailServoBridge._standstill_hold_update(
        held=True,
        enter_since_s=None,
        now_s=2.0,
        err_m=0.0,
        v_ref_m_s=0.02,
        v_cmd_m_s=0.0,
        v_meas_m_s=0.0,
        enter_m=0.00005,
        exit_m=0.00025,
        dwell_s=0.08,
    )
    assert held is False
    assert since is None


def test_late_tick_position_stream_under_reads_without_v_ff() -> None:
    """Nominal 5 ms integrate on a 6.5 ms wall tick looks like ~0.061 m/s."""
    history: deque[tuple[float, float]] = deque(maxlen=64)
    x = 0.400
    qdot = 0.079
    t = 0.0
    for _ in range(24):
        t += 0.0065
        x += qdot * 0.005
        history.append((t, x))
    _, v_est, _ = RailServoBridge._estimate_goal_motion(
        history,
        now_s=t,
        max_age_s=0.04,
    )
    assert v_est == pytest.approx(0.0608, abs=0.004)


def test_v_ff_overrides_slow_nominal_dt_position_stream() -> None:
    history: deque[tuple[float, float]] = deque(maxlen=64)
    x = 0.400
    qdot = 0.079
    t = 0.0
    for _ in range(24):
        t += 0.0065
        x += qdot * 0.005
        history.append((t, x))
    goal, v_goal, stationary = RailServoBridge._resolve_stream_goal(
        history,
        now_s=t + 0.002,
        max_age_s=0.04,
        target_m=x,
        last_rx_s=t,
        v_ff_m_s=qdot,
    )
    assert v_goal == pytest.approx(0.079)
    assert not stationary
    assert goal == pytest.approx(x + qdot * 0.002, abs=1.0e-9)


def test_v_ff_cruise_does_not_chop_host_velocity() -> None:
    """Live v_ff + FA40-matched a_max + trim P must not reverse FA24 on cruise."""
    a_max = live_host_accel_m_s2(
        vel_max_m_s=0.15,
        accel_ms=200.0,
        configured_m_s2=0.8,
    )
    assert a_max < 0.70
    x_ref = 0.400
    v_ref = 0.0
    v_cmd = 0.0
    v_meas = 0.0
    x_meas = 0.400
    cmds: list[float] = []
    for i in range(80):
        now_s = i * DT
        target = 0.400 + 0.080 * now_s
        goal, v_goal, stationary = RailServoBridge._resolve_stream_goal(
            ((now_s, target),),
            now_s=now_s,
            max_age_s=0.04,
            target_m=target,
            last_rx_s=now_s,
            v_ff_m_s=0.080,
        )
        x_ref, v_ref, _ = RailServoBridge._step_reference(
            x_ref,
            v_ref,
            goal,
            v_goal,
            stationary=stationary,
            dt=DT,
            v_max=V_MAX,
            a_max=a_max,
        )
        err_x = x_ref - x_meas
        v_p = max(-0.010, min(0.010, 4.0 * err_x))
        v_des = max(-V_MAX, min(V_MAX, v_ref + v_p + 0.22 * (v_ref - v_meas)))
        v_cmd = max(v_cmd - a_max * DT, min(v_cmd + a_max * DT, v_des))
        v_meas += 0.35 * (v_cmd - v_meas)
        x_meas += v_meas * DT
        cmds.append(v_cmd)
    cruise = cmds[25:]
    assert min(cruise) > 0.04
    assert max(cruise) < 0.12
    flips = sum(
        1
        for i in range(1, len(cruise))
        if cruise[i] * cruise[i - 1] < 0.0
        and abs(cruise[i]) > 0.005
        and abs(cruise[i - 1]) > 0.005
    )
    assert flips == 0
    assert max(abs(cruise[i] - cruise[i - 1]) for i in range(1, len(cruise))) <= (
        a_max * DT + 1.0e-12
    )


def test_set_target_m_stores_v_ff() -> None:
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    bridge._calibrated = True  # noqa: SLF001
    bridge._armed = True  # noqa: SLF001
    assert bridge.set_target_m(0.40, v_ff_m_s=0.08)
    assert bridge.target_v_ff_m_s == pytest.approx(0.08)
    assert bridge.set_target_m(0.41)
    assert math.isnan(bridge.target_v_ff_m_s)


def test_fa24_deadband_skips_small_nonzero_dither() -> None:
    assert apply_fa24_rpm_deadband(360, 360, 12) == 360
    assert apply_fa24_rpm_deadband(368, 360, 12) == 360
    assert apply_fa24_rpm_deadband(380, 360, 12) == 380
    assert apply_fa24_rpm_deadband(0, 360, 12) == 0
    assert apply_fa24_rpm_deadband(12, 0, 12) == 12
    assert apply_fa24_rpm_deadband(368, 360, 12, force=True) == 368

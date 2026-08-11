"""Rail online-reference tests; no hardware."""

from __future__ import annotations

import math
from collections import deque

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
    parse_rail_servo_config,
)


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
        assert cfg.vel_max_m_s == 0.20
        assert cfg.vel_amax_m_s2 == 0.8
        assert cfg.vel_deadband_mm == 0.05
        assert cfg.encoder_freeze_min_move_mm == 0.15
        assert cfg.accel_ms == 150
        assert cfg.decel_ms == 150
        assert cfg.scurve_ms == 30

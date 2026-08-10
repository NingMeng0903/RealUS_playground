"""Unit tests for RailServoBridge online trapezoid reference (no hardware)."""

from __future__ import annotations

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import RailServoBridge


def test_trapezoid_reaches_goal_without_overshoot() -> None:
    x, v = 0.0, 0.0
    goal = 0.040  # 40 mm
    dt = 0.01
    v_max = 0.05
    a_max = 0.5
    for _ in range(5000):
        x, v = RailServoBridge._step_trapezoid_ref(
            x, v, goal, dt=dt, v_max=v_max, a_max=a_max
        )
        assert x <= goal + 1e-9
        if abs(x - goal) < 1e-6 and abs(v) < 1e-6:
            break
    else:
        raise AssertionError(f"did not settle: x={x} v={v}")
    assert abs(x - goal) < 1e-6
    assert abs(v) < 1e-6


def test_trapezoid_respects_vmax() -> None:
    x, v = 0.0, 0.0
    goal = 1.0
    dt = 0.01
    v_max = 0.02
    a_max = 2.0
    peak = 0.0
    for _ in range(200):
        x, v = RailServoBridge._step_trapezoid_ref(
            x, v, goal, dt=dt, v_max=v_max, a_max=a_max
        )
        peak = max(peak, abs(v))
    assert peak <= v_max + 1e-9


def test_trapezoid_brakes_on_goal_reversal() -> None:
    x, v = 0.02, 0.05
    goal = 0.0
    dt = 0.01
    x2, v2 = RailServoBridge._step_trapezoid_ref(
        x, v, goal, dt=dt, v_max=0.1, a_max=0.5
    )
    # Was moving +x away from goal at 0 → must brake (v decreases).
    assert v2 < v
    assert x2 < x or abs(v2) < abs(v)

"""Execution-side FA24 velocity/acceleration/jerk box contracts."""

from __future__ import annotations

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailCommandReceipt,
    RailServoBridge,
    RailServoConfig,
    RailServoSample,
    intersect_rail_velocity_box,
)


@pytest.mark.parametrize("dt", [0.005, 0.006, 0.007])
def test_execution_box_limits_acceleration_and_jerk(dt: float) -> None:
    v, a, j, feasible = intersect_rail_velocity_box(
        0.12,
        0.0,
        0.0,
        dt_s=dt,
        v_max_m_s=0.12,
        a_max_m_s2=0.60,
        j_max_m_s3=60.0,
        rpm_per_m_s=6000.0,
    )
    assert feasible
    assert abs(v) <= 0.12 + 1e-12
    assert abs(a) <= 0.60 + 1e-12
    assert abs(j) <= 60.0 + 1e-9


def test_execution_box_recomputes_after_wall_cap() -> None:
    # Moving toward the upper end: wall cap forces a negative acceleration,
    # while the jerk interval prevents a one-tick sign flip.
    v, a, j, feasible = intersect_rail_velocity_box(
        0.10,
        0.06,
        0.20,
        dt_s=0.006,
        v_max_m_s=0.12,
        a_max_m_s2=0.60,
        j_max_m_s3=60.0,
        wall_lo_m_s=-0.12,
        wall_hi_m_s=0.062,
        rpm_per_m_s=6000.0,
    )
    assert feasible
    assert v <= 0.062 + 1e-12
    assert abs(a) <= 0.60 + 1e-12
    assert abs(j) <= 60.0 + 1e-9


def test_execution_box_reports_empty_intersection() -> None:
    # A wall interval that excludes the reachable continuity interval must
    # fail closed instead of collapsing one hard constraint silently.
    _v, _a, _j, feasible = intersect_rail_velocity_box(
        0.0,
        0.08,
        0.0,
        dt_s=0.005,
        v_max_m_s=0.12,
        a_max_m_s2=0.60,
        j_max_m_s3=60.0,
        wall_lo_m_s=-0.01,
        wall_hi_m_s=-0.005,
    )
    assert not feasible


def test_command_receipt_is_bool_compatible_and_exposes_sequence() -> None:
    accepted = RailCommandReceipt(True, command_seq=17)
    rejected = RailCommandReceipt(False, reason="not_armed")
    assert accepted
    assert accepted.command_seq == 17
    assert accepted.motion_seq == 17
    assert not rejected
    assert rejected.reason == "not_armed"


def test_execution_feedback_can_be_time_aligned_from_ring() -> None:
    bridge = RailServoBridge(RailServoConfig(enabled=False))
    first = RailServoSample(
        sample_mono_s=10.0,
        x_meas_m=0.40,
        v_meas_m_s=0.0,
        v_cmd_m_s=0.0,
        feedback_valid=True,
    )
    second = RailServoSample(
        sample_mono_s=10.01,
        x_meas_m=0.401,
        v_meas_m_s=0.1,
        v_cmd_m_s=0.1,
        feedback_valid=True,
        command_seq=2,
    )
    with bridge._lock:  # noqa: SLF001 - deterministic ring fixture
        bridge._servo_sample = second
        bridge._execution_history.clear()
        bridge._execution_history.extend((first, second))
    aligned = bridge.execution_feedback_at(10.005)
    assert aligned.sample_mono_s == pytest.approx(10.005)
    assert aligned.position_m == pytest.approx(0.4005)
    assert aligned.v_meas_m_s == pytest.approx(0.05)
    assert aligned.valid

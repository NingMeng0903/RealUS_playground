"""Unit tests for health-state hysteresis."""

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.health_monitor import (
    HealthMonitor,
    HealthState,
    HealthThresholds,
)


def test_default_health_thresholds_are_the_safety_bands() -> None:
    t = HealthThresholds()
    assert t.arm_warn == pytest.approx(0.08)
    assert t.arm_danger == pytest.approx(0.04)
    assert t.arm_exit == pytest.approx(0.10)
    assert t.joint_danger_deg == pytest.approx(15.0)
    assert t.joint_warn_deg == pytest.approx(20.0)
    assert t.wrist_danger_deg == pytest.approx(20.0)
    assert t.wrist_warn_deg == pytest.approx(25.0)


def test_danger_recovery_and_exit_settling_hysteresis() -> None:
    monitor = HealthMonitor(settling_s=0.10)
    assert monitor.update(arm_rho=0.03, dt=0.01).state is HealthState.RECOVERY
    # Exit requires all available metrics to be above their exit bands.
    assert monitor.update(arm_rho=0.11, dt=0.05).state is HealthState.SETTLING
    assert monitor.update(arm_rho=0.11, dt=0.05).state is HealthState.NORMAL
    assert monitor.update(arm_rho=0.03, dt=0.01).state is HealthState.RECOVERY


def test_joint_and_wrist_metrics_can_be_derived_from_q() -> None:
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    monitor = HealthMonitor(q_lower=lower, q_upper=upper, wrist_indices=(5,))
    q = np.zeros(7)
    q[5] = 0.1
    result = monitor.update(arm_rho=0.2, q=q, dt=0.01)
    assert result.wrist_margin_rad == pytest.approx(0.9)
    assert result.joint_margin_rad == pytest.approx(0.9)
    q[5] = -1.0
    # Joint/wrist danger warns but must not enter RECOVERY (no α freeze).
    near_limit = monitor.update(arm_rho=0.2, q=q, dt=0.01)
    assert near_limit.state is HealthState.NORMAL
    assert near_limit.warning
    assert "margin" in near_limit.reason


def test_only_arm_health_danger_enters_recovery() -> None:
    lower = np.full(7, -1.0)
    upper = np.full(7, 1.0)
    monitor = HealthMonitor(q_lower=lower, q_upper=upper, wrist_indices=(5,))
    q = np.zeros(7)
    q[5] = -1.0  # wrist on limit
    assert monitor.update(arm_rho=0.2, q=q, dt=0.01).state is HealthState.NORMAL
    assert monitor.update(arm_rho=0.03, q=q, dt=0.01).state is HealthState.RECOVERY


def test_invalid_sample_latches_fault_until_cleared() -> None:
    monitor = HealthMonitor()
    assert monitor.update(arm_rho=0.1, dt=0.01).state is HealthState.NORMAL
    assert monitor.update(arm_rho=np.nan, dt=0.01).state is HealthState.FAULT
    monitor.clear_fault()
    assert monitor.state is HealthState.NORMAL
    assert monitor.update(arm_rho=0.1, dt=0.01).state is HealthState.NORMAL

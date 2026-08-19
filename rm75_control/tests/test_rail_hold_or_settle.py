"""Task-end rail hold: never re-open follow; drift must rewrite FA24=0."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
)
from rm75_control.hw.lw100.drive import LW100Drive
from rm75_control.hw.lw100.modbus_rtu_tcp import ModbusRtuError


def _bridge(*, target_m: float, measured_m: float) -> RailServoBridge:
    cfg = RailServoConfig(enabled=True, settle_tol_mm=0.05)
    bridge = RailServoBridge(cfg)
    bridge._drive = MagicMock()  # mark as connected
    bridge._armed = True
    bridge._calibrated = True
    bridge._panic = False
    bridge._measured_m = measured_m
    bridge._target_m = target_m
    bridge._commanded_m = target_m
    bridge._follow_enabled = True
    bridge.kill_motion = MagicMock()
    bridge._encoder_sane = lambda m: True  # type: ignore[method-assign]
    return bridge


def test_hold_or_settle_holds_when_residual_under_2mm() -> None:
    bridge = _bridge(target_m=0.400, measured_m=0.4001)  # 0.1 mm
    ok = bridge.hold_or_settle_after_task()
    assert ok is True
    assert bridge._follow_enabled is False
    assert bridge._hold_active is True
    assert bridge._target_m == pytest.approx(0.4001)
    bridge.kill_motion.assert_called()


def test_hold_or_settle_holds_when_residual_large() -> None:
    """10 mm leftover must not re-open follow (125211 hit 900 r/min)."""
    bridge = _bridge(target_m=0.410, measured_m=0.400)
    bridge.settle_and_hold = MagicMock(return_value=True)  # type: ignore[method-assign]
    ok = bridge.hold_or_settle_after_task()
    assert ok is True
    assert bridge._follow_enabled is False
    assert bridge._hold_active is True
    assert bridge._target_m == pytest.approx(0.400)
    bridge.settle_and_hold.assert_not_called()
    bridge.kill_motion.assert_called()


def test_hold_watchdog_rewrites_fa24_after_2mm_drift() -> None:
    bridge = _bridge(target_m=0.300, measured_m=0.300)
    drive = MagicMock()
    bridge._drive = drive
    bridge._hold_active = True
    bridge._hold_origin_m = 0.300
    bridge._hold_anchor_m = 0.300
    bridge._last_hold_zero_mono = time.monotonic()
    bridge._hold_watchdog(0.303, time.monotonic())
    drive.set_velocity_rpm.assert_called_with(0, force=True)
    assert bridge._hold_anchor_m == pytest.approx(0.303)
    assert bridge._panic is False


def test_hold_watchdog_does_not_panic_after_5mm_from_origin() -> None:
    bridge = _bridge(target_m=0.300, measured_m=0.300)
    drive = MagicMock()
    drive.kill_velocity_hard.return_value = True
    drive._client = MagicMock()
    drive._client._sock = None
    bridge._drive = drive
    bridge._hold_active = True
    bridge._hold_origin_m = 0.300
    bridge._hold_anchor_m = 0.304
    bridge._last_hold_zero_mono = time.monotonic()
    bridge._last_hold_drift_log_mono = 0.0
    bridge._hold_watchdog(0.306, time.monotonic())
    assert bridge._panic is False
    drive.set_velocity_rpm.assert_called_with(0, force=True)
    assert bridge._hold_anchor_m == pytest.approx(0.306)


def test_hold_watchdog_force_zeros_every_second() -> None:
    bridge = _bridge(target_m=0.300, measured_m=0.300)
    drive = MagicMock()
    bridge._drive = drive
    bridge._hold_active = True
    bridge._hold_origin_m = 0.300
    bridge._hold_anchor_m = 0.300
    now = time.monotonic()
    bridge._last_hold_zero_mono = now - 1.1
    bridge._hold_watchdog(0.300, now)
    drive.set_velocity_rpm.assert_called_with(0, force=True)


def test_kill_velocity_hard_preserves_latch_on_write_fail() -> None:
    class _Stub:
        _last_rpm_cmd = 90

        def set_velocity_rpm(self, rpm: float, *, force: bool = False) -> int:
            raise ModbusRtuError("nope")

        class _client:
            @staticmethod
            def recover() -> None:
                return None

    stub = _Stub()
    ok = LW100Drive.kill_velocity_hard(stub, attempts=2)
    assert ok is False
    assert stub._last_rpm_cmd == 90

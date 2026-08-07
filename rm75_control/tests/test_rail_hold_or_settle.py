"""Task-end rail hold: sub-mm residual must not re-open follow crawl."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rm75_control.control.joint_admittance_8dof.hw.rail_servo import (
    RailServoBridge,
    RailServoConfig,
)


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
    ok = bridge.hold_or_settle_after_task(settle_if_err_mm=2.0)
    assert ok is True
    assert bridge._follow_enabled is False
    assert bridge._target_m == pytest.approx(0.4001)
    bridge.kill_motion.assert_called()


def test_hold_or_settle_settles_when_residual_large() -> None:
    bridge = _bridge(target_m=0.410, measured_m=0.400)  # 10 mm
    bridge.settle_and_hold = MagicMock(return_value=True)  # type: ignore[method-assign]
    ok = bridge.hold_or_settle_after_task(settle_if_err_mm=2.0)
    assert ok is True
    bridge.settle_and_hold.assert_called_once()

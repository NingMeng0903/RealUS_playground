"""Control-loop schedule helpers (no robot required)."""

from __future__ import annotations

import pytest

from rm75_control.control.joint_admittance.loop import _resync_late_tick


def test_resync_late_tick_within_period():
    dt = 0.005
    next_tick, late_ms = _resync_late_tick(1.0, 1.003, dt)
    assert next_tick == 1.0
    assert late_ms == pytest.approx(3.0)


def test_resync_late_tick_skips_burst_when_whole_period_missed():
    dt = 0.005
    next_tick, late_ms = _resync_late_tick(1.0, 1.012, dt)
    assert next_tick == 1.012
    assert late_ms == pytest.approx(12.0)

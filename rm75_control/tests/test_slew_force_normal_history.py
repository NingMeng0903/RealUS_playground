"""Force-axis slew history, monotone deadband, and DOB reversal through zero."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rm75_control.control.admittance_common.controller import (
    AdmittanceConfig,
    AdmittanceController,
    smooth_deadband_eff,
)
DT = 0.005


def _slew_ctrl() -> AdmittanceController:
    cfg = AdmittanceConfig(
        force_axis_slew_press_m_s2=1.2,
        force_axis_slew_retract_m_s2=1.2,
        force_axis_slew_reverse_m_s2=2.0,
        force_axis_jerk_max_m_s3=40.0,
    )
    return AdmittanceController(DT, cfg)


def test_slew_zero_hold_does_not_store_press_accel() -> None:
    ctrl = _slew_ctrl()
    for _ in range(10):
        out = ctrl._slew_force_normal(0.0, DT)
        assert out == 0.0
    assert ctrl._u_force_slew_dot == 0.0


def test_slew_small_retract_after_zero_does_not_invert() -> None:
    ctrl = _slew_ctrl()
    for _ in range(10):
        ctrl._slew_force_normal(0.0, DT)
    out = ctrl._slew_force_normal(-0.0001, DT)
    assert out <= 0.0
    assert out == -0.0001
    assert ctrl._u_force_slew_dot == (out - 0.0) / DT


def test_slew_history_matches_emitted_delta() -> None:
    ctrl = _slew_ctrl()
    prev = 0.0
    for target in (0.02, 0.04, 0.04, 0.01, 0.0):
        out = ctrl._slew_force_normal(target, DT)
        assert ctrl._u_force_slew_dot == (out - prev) / DT
        prev = out


def test_smooth_deadband_is_c1_and_monotone() -> None:
    d, w = 0.08, 0.10
    xs = np.linspace(-0.5, 0.5, 2001)
    ys = np.array([smooth_deadband_eff(float(x), d, w) for x in xs])
    edge = d + w
    left = smooth_deadband_eff(edge - 1e-9, d, w)
    at = smooth_deadband_eff(edge, d, w)
    right = smooth_deadband_eff(edge + 1e-9, d, w)
    assert abs(left - at) < 1e-8
    assert abs(right - at) < 1e-8
    assert at == pytest.approx(math.copysign(w / 2.0, edge), abs=1e-12)
    h = 1e-6
    d_left = (
        smooth_deadband_eff(edge, d, w) - smooth_deadband_eff(edge - h, d, w)
    ) / h
    d_right = (
        smooth_deadband_eff(edge + h, d, w) - smooth_deadband_eff(edge, d, w)
    ) / h
    assert abs(d_left - d_right) < 1e-3
    pos = ys[xs >= 0.0]
    assert np.all(np.diff(pos) >= -1e-12)

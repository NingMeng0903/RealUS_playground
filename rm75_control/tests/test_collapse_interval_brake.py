"""collapse_interval must pick the brake side of an empty velocity box."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    collapse_interval,
)


def _old_rule(lo, hi, qdot_prev=None, a_max=None, dt=None):
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()
    crossed = lo > hi
    keep_zero = crossed & (hi < 0.0) & (lo > 0.0)
    if qdot_prev is None:
        pick_lo = np.abs(lo) <= np.abs(hi)
        collapsed = np.where(pick_lo, lo, hi)
    else:
        prev = np.asarray(qdot_prev, dtype=float)
        collapsed = np.where(prev >= 0.0, lo, hi)
    collapsed = np.where(keep_zero, 0.0, collapsed)
    if qdot_prev is not None and a_max is not None and dt is not None and float(dt) > 0.0:
        prev = np.asarray(qdot_prev, dtype=float)
        a_step = np.asarray(a_max, dtype=float) * float(dt)
        collapsed = np.clip(collapsed, prev - a_step, prev + a_step)
    return np.where(crossed, collapsed, lo)


def test_positive_travel_picks_lower_gap_not_lo() -> None:
    lo = np.array([0.104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.101, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev)
    assert out_lo[0] == pytest.approx(0.101)
    assert out_hi[0] == pytest.approx(0.101)
    assert _old_rule(lo, hi, qdot_prev=prev)[0] == pytest.approx(0.104)


def test_negative_travel_is_the_mirror() -> None:
    lo = np.array([-0.101, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([-0.104, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev = np.array([-0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev)
    assert out_lo[0] == pytest.approx(-0.101)
    assert out_hi[0] == pytest.approx(-0.101)
    assert _old_rule(lo, hi, qdot_prev=prev)[0] == pytest.approx(-0.104)


def test_gap_straddling_zero_stays_between_zero_and_prev() -> None:
    lo = np.full(8, 0.02)
    hi = np.full(8, -0.03)
    prev = np.zeros(8)
    prev[0] = 0.10
    a_max = np.full(8, 0.60)
    dt = 0.005
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev, a_max=a_max, dt=dt)
    v = float(out_lo[0])
    assert v == pytest.approx(float(out_hi[0]))
    assert 0.0 <= v <= float(prev[0]) + 1.0e-12
    assert v == pytest.approx(0.10 - 0.60 * dt)


def test_result_is_brake_projection_and_not_farther_from_zero() -> None:
    rng = np.random.default_rng(4)
    for _ in range(40):
        prev = rng.uniform(-0.12, 0.12, size=8)
        gap_a = rng.uniform(-0.15, 0.15, size=8)
        gap_b = gap_a + rng.uniform(0.001, 0.02, size=8)
        # Force a crossed box: lo = max, hi = min.
        lo = np.maximum(gap_a, gap_b)
        hi = np.minimum(gap_a, gap_b)
        a_max = np.full(8, 0.60)
        dt = 0.005
        out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev, a_max=a_max, dt=dt)
        np.testing.assert_allclose(out_lo, out_hi, atol=1e-12)
        old = _old_rule(lo, hi, qdot_prev=prev, a_max=a_max, dt=dt)
        step = a_max * dt
        brake = np.where(prev > 0.0, np.maximum(0.0, prev - step), prev)
        brake = np.where(prev < 0.0, np.minimum(0.0, prev + step), brake)
        projected = np.clip(brake, hi, lo)
        projected = np.clip(projected, prev - step, prev + step)
        np.testing.assert_allclose(out_lo, projected, atol=1e-12)
        assert np.all(np.abs(out_lo) <= np.abs(old) + 1.0e-12)


def test_non_crossed_box_is_unchanged() -> None:
    lo = -0.05 * np.ones(8)
    hi = 0.05 * np.ones(8)
    prev = np.full(8, 0.02)
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev, a_max=np.full(8, 0.6), dt=0.005)
    np.testing.assert_allclose(out_lo, lo)
    np.testing.assert_allclose(out_hi, hi)


def test_cpp_kernel_matches_python() -> None:
    if not cpp_kernel.available() or not hasattr(
        cpp_kernel, "collapse_interval"
    ):
        pytest.skip("qpik_kernel collapse_interval not built")
    rng = np.random.default_rng(7)
    lo = rng.uniform(-0.1, 0.1, size=8)
    hi = lo - rng.uniform(0.001, 0.01, size=8)
    prev = rng.uniform(-0.12, 0.12, size=8)
    a_max = np.full(8, 0.60)
    dt = 0.005
    py_lo, py_hi = collapse_interval(lo, hi, qdot_prev=prev, a_max=a_max, dt=dt)
    cxx_lo, cxx_hi = cpp_kernel.collapse_interval(
        lo, hi, qdot_prev=prev, a_max=a_max, dt=dt
    )
    np.testing.assert_allclose(cxx_lo, py_lo, atol=1e-12)
    np.testing.assert_allclose(cxx_hi, py_hi, atol=1e-12)
    cxx2_lo, cxx2_hi = cpp_kernel.collapse_interval(lo, hi, qdot_prev=prev)
    py2_lo, py2_hi = collapse_interval(lo, hi, qdot_prev=prev)
    np.testing.assert_allclose(cxx2_lo, py2_lo, atol=1e-12)
    np.testing.assert_allclose(cxx2_hi, py2_hi, atol=1e-12)

"""collapse_interval keeps the old prev>=0 → lo singleton (nonempty box unchanged)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.control.joint_admittance_8dof.solver import cpp_kernel
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    collapse_interval,
)


def test_positive_travel_picks_lo() -> None:
    lo = np.array([0.104, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([0.101, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev)
    assert out_lo[0] == pytest.approx(0.104)
    assert out_hi[0] == pytest.approx(0.104)


def test_negative_travel_picks_hi() -> None:
    lo = np.array([-0.101, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hi = np.array([-0.104, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    prev = np.array([-0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev)
    assert out_lo[0] == pytest.approx(-0.104)
    assert out_hi[0] == pytest.approx(-0.104)


def test_gap_straddling_zero_collapses_to_zero_then_accel_clip() -> None:
    lo = np.full(8, 0.02)
    hi = np.full(8, -0.03)
    prev = np.zeros(8)
    prev[0] = 0.10
    a_max = np.full(8, 0.60)
    dt = 0.005
    out_lo, out_hi = collapse_interval(lo, hi, qdot_prev=prev, a_max=a_max, dt=dt)
    v = float(out_lo[0])
    assert v == pytest.approx(float(out_hi[0]))
    # hi<0<lo → 0, then clip into [prev - a·dt, prev + a·dt].
    assert v == pytest.approx(0.10 - 0.60 * dt)


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

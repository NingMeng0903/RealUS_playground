"""Faverjon/Tournassoud joint-limit velocity damper tests (constraint_mgr)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance.solver.constraint_mgr import (
    VelocityBoxConstraints,
)
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


def _limits() -> SafetyLimits:
    return SafetyLimits(
        q_lower=np.full(7, -2.0),
        q_upper=np.full(7, 2.0),
        v_max=np.full(7, 1.0),
        a_max=None,
        position_margin=0.017,
    )


def test_damper_ramps_continuously_toward_limit():
    """hi(q) toward the upper limit is continuous and monotonically shrinks to
    ~0 at the margin (no binary flip at a threshold)."""
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.15)
    dt = 0.005
    qs = np.linspace(1.6, 2.0 - 0.017, 200)
    his = []
    for qj in qs:
        q = np.zeros(7)
        q[3] = qj
        _lo, hi = box.bounds(q, dt)
        his.append(hi[3])
    his = np.asarray(his)
    assert float(np.max(np.abs(np.diff(his)))) < 0.05  # continuous
    assert his[0] == 1.0                                # far: full v_max
    assert his[-1] <= 1e-9                              # at margin: no motion toward limit


def test_damper_never_blocks_motion_away_from_limit():
    box = VelocityBoxConstraints(_limits(), damper_band_rad=0.15)
    q = np.zeros(7)
    q[3] = 2.0 - 0.02  # just inside the margin-backed upper limit
    lo, hi = box.bounds(q, 0.005)
    assert lo[3] <= -0.9   # full speed away from the limit
    assert hi[3] >= 0.0    # box always contains 0 (never infeasible)
    assert lo[3] <= hi[3]


def test_resync_damper_is_continuous_and_feasible():
    """Command-lead anti-windup must ramp smoothly, never collapse lo==hi."""
    lim = SafetyLimits(
        q_lower=np.full(7, -3.0),
        q_upper=np.full(7, 3.0),
        v_max=np.full(7, 1.0),
        a_max=None,
        position_margin=0.017,
    )
    box = VelocityBoxConstraints(lim)
    dt = 0.005
    resync = 0.10
    q_meas = np.zeros(7)
    prev_hi = None
    for lead_frac in np.linspace(0.0, 1.5, 200):
        q = q_meas + lead_frac * resync
        lo, hi = box.bounds(q, dt, np.zeros(7), q_meas=q_meas, resync_err=resync)
        assert np.all(lo <= hi + 1e-12)
        if prev_hi is not None:
            assert float(np.max(np.abs(hi - prev_hi))) < 0.2
        prev_hi = hi.copy()

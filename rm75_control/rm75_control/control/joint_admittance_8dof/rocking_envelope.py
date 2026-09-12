"""Fixed tool-y command envelope, subordinate to existing mechanical limits.

Bounds contain rate, acceleration and jerk pairs. They are intersected in that
order. A conflicting interval is deliberately allowed so admission can relax
jerk, then acceleration, then the task rate restriction without relaxing P0.
"""
import numpy as np


def validate_rocking(axis, bounds):
    if axis is None and bounds is None:
        return None, None
    if axis is None or bounds is None:
        raise ValueError("rocking axis and bounds must be supplied together")
    axis = np.asarray(axis, dtype=float).reshape(3)
    bounds = np.asarray(bounds, dtype=float).reshape(6)
    if not np.isfinite(axis).all() or abs(float(axis @ axis) - 1.) > 1e-8:
        raise ValueError("rocking axis must be a finite unit base-frame axis")
    if not np.isfinite(bounds).all():
        raise ValueError("rocking bounds must be finite")
    if bounds[0] > bounds[1]:
        raise ValueError("rocking rate interval must be ordered")
    return axis.copy(), bounds.copy()


def rocking_interval(bounds, tier):
    """1 jerk, 2 acceleration, 3 rate, 4 mechanical-only, 0 disabled."""
    if tier in (0, 4):
        return -float("inf"), float("inf")
    pairs = np.asarray(bounds, dtype=float).reshape(3, 2)[:4-tier]
    return float(pairs[:, 0].max()), float(pairs[:, 1].min())

"""Strict opt-in wire validation for generic TCP/base velocity rows."""
from dataclasses import dataclass
import math
import numpy as np

from .protocol import MAX_CARTESIAN_ROWS


@dataclass(frozen=True)
class CartesianCertificate:
    A: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sequence: int
    stop_epoch: int
    valid_until_s: float


def validate_cartesian_constraints(value, *, now_s):
    if value is None:
        return None
    if getattr(value, "frame", None) != "tcp_base":
        raise ValueError("Cartesian constraints require frame='tcp_base' at the original TCP")
    A = np.array(value.A, dtype=float, copy=True)
    lower = np.array(value.lower, dtype=float, copy=True)
    upper = np.array(value.upper, dtype=float, copy=True)
    if A.ndim != 2 or A.shape[1] != 6 or len(A) > MAX_CARTESIAN_ROWS:
        raise ValueError("Cartesian constraints require at most 96 six-component rows")
    if lower.shape != (len(A),) or upper.shape != lower.shape or not np.isfinite(A).all():
        raise ValueError("invalid Cartesian row dimensions or coefficients")
    if (np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower > upper)
            or np.isposinf(lower).any() or np.isneginf(upper).any()):
        raise ValueError("invalid Cartesian row bounds")
    ids = []
    for name in ("sequence", "stop_epoch"):
        item = getattr(value, name)
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)) or not 0 <= item < 2**64:
            raise ValueError(f"{name} must be an unsigned 64-bit integer")
        ids.append(int(item))
    expiry = float(value.valid_until_s)
    if not math.isfinite(expiry) or not math.isfinite(now_s) or now_s >= expiry:
        raise ValueError("Cartesian certificate must have a finite future expiry")
    for array in (A, lower, upper):
        array.setflags(write=False)
    return CartesianCertificate(A, lower, upper, *ids, expiry)

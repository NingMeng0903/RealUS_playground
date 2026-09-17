"""Time-average Welleweerd C_ij maps on ultrasound frames that have contact force."""

from __future__ import annotations

import numpy as np


def contact_us_mask(us_t, force_t, force_fz, *, fz_min=0.8):
    """US samples whose interpolated tool-z force is at or above contact."""
    us_t = np.asarray(us_t, dtype=np.float64).reshape(-1)
    force_t = np.asarray(force_t, dtype=np.float64).reshape(-1)
    force_fz = np.asarray(force_fz, dtype=np.float64).reshape(-1)
    if us_t.size == 0 or force_t.size == 0:
        return np.zeros(us_t.shape, dtype=bool)
    order = np.argsort(force_t)
    fz = np.interp(us_t, force_t[order], force_fz[order], left=np.nan, right=np.nan)
    return np.isfinite(fz) & (fz >= float(fz_min))


def mean_confidence_map(maps):
    """Pixelwise mean of finite C_ij maps, ignoring empty input."""
    stack = np.asarray(maps, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.size == 0:
        raise ValueError("need at least one confidence map")
    return np.nanmean(stack, axis=0)

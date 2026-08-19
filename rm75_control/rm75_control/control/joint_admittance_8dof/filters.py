"""Shared C1 fades and first-order filters (one copy each)."""

from __future__ import annotations

import numpy as np


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lpf_tau_from_fc(f_c_hz: float) -> float:
    """First-order time constant such that |H(f_c)| = 1/√2."""
    fc = float(f_c_hz)
    if fc <= 1.0e-9:
        return 0.0
    return 1.0 / (2.0 * np.pi * fc)


def first_order_lpf(prev: float, target: float, dt: float, tau: float) -> float:
    """Scalar first-order LPF.  ``tau<=0`` or ``dt<=0`` snaps to ``target``."""
    dt = float(dt)
    tau = float(tau)
    if tau <= 1.0e-9 or dt <= 0.0:
        return float(target)
    alpha = dt / (tau + dt)
    return (1.0 - alpha) * float(prev) + alpha * float(target)


def first_order_lpf_vec(
    prev: np.ndarray, target: np.ndarray, dt: float, tau: float
) -> np.ndarray:
    tgt = np.asarray(target, dtype=float)
    dt = float(dt)
    tau = float(tau)
    if tau <= 1.0e-9 or dt <= 0.0:
        return tgt.copy()
    prev_a = np.asarray(prev, dtype=float)
    alpha = dt / (tau + dt)
    return (1.0 - alpha) * prev_a + alpha * tgt


__all__ = (
    "first_order_lpf",
    "first_order_lpf_vec",
    "lpf_tau_from_fc",
    "smoothstep01",
)

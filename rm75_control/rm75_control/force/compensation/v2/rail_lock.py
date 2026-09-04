"""Rail lock measurement gates before the first static hold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RailLockLimits:
    pos_err_max_m: float = 5.0e-4
    vel_p95_max_m_s: float = 5.0e-4
    settle_s: float = 0.4


@dataclass
class RailLockStatus:
    ok: bool
    pos_err_m: float
    vel_p95_m_s: float
    reason: str = ""


def evaluate_rail_lock(
    q_cmd_m: float,
    q_meas_m: np.ndarray,
    qdot_m_s: np.ndarray,
    lim: RailLockLimits | None = None,
) -> RailLockStatus:
    lim = lim or RailLockLimits()
    meas = np.asarray(q_meas_m, dtype=float).reshape(-1)
    vel = np.asarray(qdot_m_s, dtype=float).reshape(-1)
    err = float(np.max(np.abs(meas - float(q_cmd_m))))
    p95 = float(np.percentile(np.abs(vel), 95)) if vel.size else float("inf")
    ok = err < lim.pos_err_max_m and p95 < lim.vel_p95_max_m_s
    reason = ""
    if err >= lim.pos_err_max_m:
        reason = "rail_pos"
    elif p95 >= lim.vel_p95_max_m_s:
        reason = "rail_vel"
    return RailLockStatus(ok, err, p95, reason)

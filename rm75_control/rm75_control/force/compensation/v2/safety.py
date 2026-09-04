"""Collection abort gates using raw wrench, not the model under identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyLimits:
    m_max_kg: float = 2.5
    f_margin_n: float = 8.0
    f_sat_n: float = 140.0
    m_sat_nm: float = 8.0
    g: float = 9.80665
    cmd_stale_s: float = 0.05
    workspace_p_min: tuple[float, float, float] = (0.15, -0.25, 0.15)
    workspace_p_max: tuple[float, float, float] = (0.55, 0.25, 0.65)
    joint_margin_rad: float = 0.08


@dataclass
class SafetyStatus:
    ok: bool
    reason: str = ""


def raw_contact_abort(force_raw: np.ndarray, lim: SafetyLimits) -> SafetyStatus:
    f = np.asarray(force_raw, dtype=float).reshape(-1)
    if f.size < 6 or not np.all(np.isfinite(f[:6])):
        return SafetyStatus(False, "raw_nan")
    if float(np.linalg.norm(f[:3])) > lim.m_max_kg * lim.g + lim.f_margin_n:
        return SafetyStatus(False, "unexpected_contact")
    if np.any(np.abs(f[:3]) > lim.f_sat_n) or np.any(np.abs(f[3:6]) > lim.m_sat_nm):
        return SafetyStatus(False, "raw_saturation")
    return SafetyStatus(True)


def workspace_abort(p_L: np.ndarray, lim: SafetyLimits) -> SafetyStatus:
    p = np.asarray(p_L, dtype=float).reshape(3)
    lo = np.asarray(lim.workspace_p_min)
    hi = np.asarray(lim.workspace_p_max)
    if np.any(p < lo) or np.any(p > hi):
        return SafetyStatus(False, "workspace")
    return SafetyStatus(True)


def joint_margin_detail(
    q: np.ndarray,
    q_lo: np.ndarray,
    q_hi: np.ndarray,
    lim: SafetyLimits,
) -> tuple[bool, int, float]:
    """Return (ok, worst_joint_index_0based, worst_margin_rad)."""

    q = np.asarray(q, dtype=float).reshape(-1)
    lo = np.asarray(q_lo, dtype=float).reshape(-1) + lim.joint_margin_rad
    hi = np.asarray(q_hi, dtype=float).reshape(-1) - lim.joint_margin_rad
    margin = np.minimum(q - lo, hi - q)
    i = int(np.argmin(margin))
    return bool(margin[i] >= 0.0), i, float(margin[i])


def joint_margin_abort(q: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, lim: SafetyLimits) -> SafetyStatus:
    ok, _, _ = joint_margin_detail(q, q_lo, q_hi, lim)
    return SafetyStatus(ok, "" if ok else "joint_margin")


def command_stale(now_s: float, cmd_t_s: float, lim: SafetyLimits) -> bool:
    return (float(now_s) - float(cmd_t_s)) > lim.cmd_stale_s


def deadman_twist(prev: np.ndarray, *, dt: float, j_lin: float = 16.0, j_ang: float = 32.0) -> np.ndarray:
    """Jerk-limited decay toward zero (Window B deadman)."""
    v = np.asarray(prev, dtype=float).reshape(6).copy()
    j = np.array([j_lin, j_lin, j_lin, j_ang, j_ang, j_ang], dtype=float)
    dv = np.clip(-v, -j * dt, j * dt)
    # if remaining is small, snap to 0 after one more clip
    out = v + dv
    out[np.abs(out) < 1e-5] = 0.0
    return out

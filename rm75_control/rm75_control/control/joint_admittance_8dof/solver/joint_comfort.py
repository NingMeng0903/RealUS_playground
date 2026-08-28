"""Per-joint set-based comfort inequalities (Moe / Kanoun).

Hard URDF boxes and the Faverjon damper only act in the last few degrees.
This layer opens a recoverable preference inequality for every arm joint
while it is still 15–25° from a stop, each with its own slack so the QP
cannot spend J2 to save J4.

Activation is a C1 smoothstep of the margin — no enter/exit hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.filters import smoothstep01
from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    PrefInequalityRows,
)

# Pref-slack layout: 0=sigma, 1=branch, 2=J4 design band, 3..8 leftover, J4 physical=5.
COMFORT_SLACK0 = 2
J4_DESIGN_SLACK = 2


@dataclass
class JointComfortConfig:
    enabled: bool = True
    m_comfort_rad: float = 0.26  # ~15°
    activate_rad: float = 0.44  # ~25° from the stop
    gamma: float = 6.0
    slack_weight: float = 80.0


class JointComfortBuilder:
    def __init__(self, cfg: JointComfortConfig | None = None) -> None:
        self.cfg = cfg or JointComfortConfig()
        self.last_n_active: int = 0
        self.last_min_margin_rad: float = float("nan")

    def reset(self) -> None:
        self.last_n_active = 0
        self.last_min_margin_rad = float("nan")

    def build_rows(
        self,
        q_rad: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
    ) -> PrefInequalityRows:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        lo = np.asarray(q_lower, dtype=float).reshape(-1)
        hi = np.asarray(q_upper, dtype=float).reshape(-1)
        nv = int(q.size)
        empty = PrefInequalityRows(
            jacobian=np.zeros((0, nv)),
            slack_col=np.zeros(0, dtype=int),
            lower=np.zeros(0),
            active=False,
        )
        if not self.cfg.enabled or nv < 2:
            return empty
        m_c = float(self.cfg.m_comfort_rad)
        act = max(float(self.cfg.activate_rad), m_c + 1.0e-6)
        band = act - m_c
        gamma = float(self.cfg.gamma)
        rows_j = []
        rows_s = []
        rows_lo = []
        min_m = float("inf")
        # J4 only — other arm comfort rows never bound (max slack 2e-6).
        for i in range(4, min(5, nv)):
            d_hi = float(hi[i] - q[i])
            d_lo = float(q[i] - lo[i])
            margin = min(d_hi, d_lo)
            min_m = min(min_m, margin)
            h = margin - m_c
            w = smoothstep01((act - margin) / band)
            if w <= 1.0e-6:
                continue
            jac = np.zeros(nv, dtype=float)
            # ∇h points away from the nearer stop.
            jac[i] = -w if d_hi <= d_lo else w
            rows_j.append(jac)
            rows_s.append(COMFORT_SLACK0 + (i - 1))
            rows_lo.append(-gamma * h * w)
        self.last_min_margin_rad = float(min_m) if np.isfinite(min_m) else float("nan")
        if not rows_j:
            self.last_n_active = 0
            return empty
        self.last_n_active = len(rows_j)
        return PrefInequalityRows(
            jacobian=np.vstack(rows_j),
            slack_col=np.asarray(rows_s, dtype=int),
            lower=np.asarray(rows_lo, dtype=float),
            active=True,
        )


@dataclass
class J4DesignComfortConfig:
    """J4 design band (70–115°) as a hard Faverjon–Tournassoud velocity box.

    Slacked pref rows are not built; the band is intersected into ``lo_box/hi_box``.
    """

    enabled: bool = True
    lower_rad: float = np.deg2rad(70.0)
    upper_rad: float = np.deg2rad(115.0)
    gamma: float = 4.0
    slack_weight: float = 60.0


def j4_joint_index(nv: int) -> int:
    """8-DoF uses q[4]; a 7-axis arm vector uses q[3]."""
    n = int(nv)
    if n >= 8:
        return 4
    if n == 7:
        return 3
    return -1


def j4_design_qdot_bounds(
    q4: float,
    *,
    lower_rad: float,
    upper_rad: float,
    gamma: float,
) -> tuple[float, float]:
    """Hard velocity box for the J4 design band (Haviland/FT, ρ_s = 0)."""
    return (
        -float(gamma) * (float(q4) - float(lower_rad)),
        float(gamma) * (float(upper_rad) - float(q4)),
    )


class J4DesignComfortBuilder:
    def __init__(self, cfg: J4DesignComfortConfig | None = None) -> None:
        self.cfg = cfg or J4DesignComfortConfig()
        self.last_n_rows: int = 0
        self.last_q4: float = float("nan")

    def reset(self) -> None:
        self.last_n_rows = 0
        self.last_q4 = float("nan")

    def build_rows(self, q_rad: np.ndarray) -> PrefInequalityRows:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        nv = int(q.size)
        idx = j4_joint_index(nv)
        if idx >= 0:
            self.last_q4 = float(q[idx])
        self.last_n_rows = 0
        return PrefInequalityRows(
            jacobian=np.zeros((0, nv)),
            slack_col=np.zeros(0, dtype=int),
            lower=np.zeros(0),
            active=False,
        )


__all__ = [
    "COMFORT_SLACK0",
    "J4_DESIGN_SLACK",
    "J4DesignComfortBuilder",
    "J4DesignComfortConfig",
    "JointComfortBuilder",
    "JointComfortConfig",
    "j4_design_qdot_bounds",
    "j4_joint_index",
]

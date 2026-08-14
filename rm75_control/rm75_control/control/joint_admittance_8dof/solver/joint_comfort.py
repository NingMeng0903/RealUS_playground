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

from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    PrefInequalityRows,
)

# Pref-slack layout: 0=sigma, 1=branch, 2..8=J1..J7.
COMFORT_SLACK0 = 2


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


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
        # Arm only — rail has WLN + its own travel box.
        for i in range(1, nv):
            d_hi = float(hi[i] - q[i])
            d_lo = float(q[i] - lo[i])
            margin = min(d_hi, d_lo)
            min_m = min(min_m, margin)
            h = margin - m_c
            w = _smoothstep01((act - margin) / band)
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


__all__ = [
    "COMFORT_SLACK0",
    "JointComfortBuilder",
    "JointComfortConfig",
]

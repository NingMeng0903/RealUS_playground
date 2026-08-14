"""Near-zero soft barriers for latched attractor branch signs.

For every arm joint with a nonzero latched q*_i, when |q_i| is small the QP
gets a recoverable unilateral inequality that blocks crossing through 0 to the
opposite sign (prevents irreversible elbow/wrist flips).  Not joint-index
business logic: any attractor component with |q*| > eps participates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.solver.sigma_setbased import (
    PrefInequalityRows,
)


@dataclass
class BranchBarrierConfig:
    enabled: bool = True
    activate_rad: float = 0.35  # ~20°
    eps_rad: float = 0.26  # ~15° floor; never an attractor (planner rejects this band)
    gamma: float = 6.0
    slack_weight: float = 80.0
    # Skip rail (index 0); only arm joints with |q*| > target_eps.
    target_eps_rad: float = 1.0e-3


class BranchBarrierBuilder:
    def __init__(self, cfg: BranchBarrierConfig | None = None) -> None:
        self.cfg = cfg or BranchBarrierConfig()
        self.last_slack: float = 0.0
        self.last_n_active: int = 0

    def reset(self) -> None:
        self.last_slack = 0.0
        self.last_n_active = 0

    def build_rows(
        self, q_rad: np.ndarray, q_star: np.ndarray
    ) -> PrefInequalityRows:
        q = np.asarray(q_rad, dtype=float).reshape(-1)
        q_star = np.asarray(q_star, dtype=float).reshape(-1)
        nv = int(q.size)
        empty = PrefInequalityRows(
            jacobian=np.zeros((0, nv)),
            slack_col=np.zeros(0, dtype=int),
            lower=np.zeros(0),
            active=False,
        )
        if not self.cfg.enabled or q_star.size != nv:
            return empty
        act = float(self.cfg.activate_rad)
        eps = float(self.cfg.eps_rad)
        gamma = float(self.cfg.gamma)
        target_eps = float(self.cfg.target_eps_rad)
        rows_j = []
        rows_lo = []
        # Arm joints only (skip rail index 0).
        for i in range(1, nv):
            qs = float(q_star[i])
            if abs(qs) <= target_eps:
                continue
            qi = float(q[i])
            if abs(qi) >= act:
                continue
            sign = 1.0 if qs >= 0.0 else -1.0
            # sign * qdot_i + s_b ≥ −γ (sign * q_i − eps)
            # → blocks motion that would decrease signed margin below eps.
            jac = np.zeros(nv, dtype=float)
            jac[i] = sign
            rhs = -gamma * (sign * qi - eps)
            rows_j.append(jac)
            rows_lo.append(rhs)
        if not rows_j:
            self.last_n_active = 0
            return empty
        self.last_n_active = len(rows_j)
        return PrefInequalityRows(
            jacobian=np.vstack(rows_j),
            slack_col=np.full(len(rows_j), 1, dtype=int),  # shared branch slack
            lower=np.asarray(rows_lo, dtype=float),
            active=True,
        )


def latch_q_star_signs(
    q_nominal: np.ndarray, q_meas: np.ndarray, *, target_eps: float = 1.0e-3
) -> np.ndarray:
    """Keep |q*| magnitudes; set signs from measurement for nonzero targets."""
    qn = np.asarray(q_nominal, dtype=float).reshape(-1).copy()
    qm = np.asarray(q_meas, dtype=float).reshape(-1)
    if qm.size != qn.size:
        return qn
    for i in range(qn.size):
        if abs(float(qn[i])) <= float(target_eps):
            continue
        # Rail (0) usually q*=0; still allow if configured nonzero.
        sign = 1.0 if float(qm[i]) >= 0.0 else -1.0
        qn[i] = sign * abs(float(qn[i]))
    return qn


__all__ = [
    "BranchBarrierConfig",
    "BranchBarrierBuilder",
    "latch_q_star_signs",
]

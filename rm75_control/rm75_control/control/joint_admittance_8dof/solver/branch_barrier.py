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
    activate_rad: float = 0.52  # ~30° soft preference
    # Hard velocity-box damper starts earlier so J4 cannot blast through 0.
    box_activate_rad: float = 0.87  # ~50°
    eps_rad: float = 0.35  # ~20° floor; crossing stays cheap via slack
    gamma: float = 6.0
    slack_weight: float = 80.0
    # Skip rail (index 0); only arm joints with |q*| > target_eps.
    target_eps_rad: float = 1.0e-3
    # Dwell upgrade: brief crossings keep slack_weight; parking ramps it.
    dwell_free_s: float = 0.3
    dwell_ramp_s: float = 1.0
    dwell_scale_max: float = 5.0


class BranchBarrierBuilder:
    def __init__(self, cfg: BranchBarrierConfig | None = None) -> None:
        self.cfg = cfg or BranchBarrierConfig()
        self.last_slack: float = 0.0
        self.last_n_active: int = 0
        self.last_dwell_scale: float = 1.0
        self._dwell_s: float = 0.0

    def reset(self) -> None:
        self.last_slack = 0.0
        self.last_n_active = 0
        self.last_dwell_scale = 1.0
        self._dwell_s = 0.0

    def _joint_in_band(
        self, q: np.ndarray, q_star: np.ndarray | None, index: int
    ) -> bool:
        qs = float(q_star[index]) if q_star is not None and q_star.size > index else 0.0
        if abs(qs) <= float(self.cfg.target_eps_rad):
            if q_star is not None:
                return False
            # Legacy J6-only path when q* is unknown.
            return index in {5, 6} and abs(float(q[index])) < float(self.cfg.activate_rad)
        sign = 1.0 if qs >= 0.0 else -1.0
        return sign * float(q[index]) < float(self.cfg.activate_rad)

    def _any_in_band(self, q: np.ndarray, q_star: np.ndarray | None) -> bool:
        nv = int(q.size)
        for i in range(1, nv):
            if self._joint_in_band(q, q_star, i):
                return True
        return False

    def _update_dwell(
        self,
        q: np.ndarray,
        dt_s: float,
        q_star: np.ndarray | None = None,
    ) -> float:
        """Scale in [1, dwell_scale_max] from how long a branch joint stays in-band."""
        in_band = self._any_in_band(q, q_star)
        if in_band:
            self._dwell_s += max(float(dt_s), 0.0)
        else:
            self._dwell_s = 0.0
        free = max(float(self.cfg.dwell_free_s), 0.0)
        ramp = max(float(self.cfg.dwell_ramp_s), 1.0e-9)
        hi = max(float(self.cfg.dwell_scale_max), 1.0)
        if self._dwell_s <= free:
            scale = 1.0
        else:
            u = min((self._dwell_s - free) / ramp, 1.0)
            scale = 1.0 + u * (hi - 1.0)
        self.last_dwell_scale = float(scale)
        return float(scale)

    def build_rows(
        self, q_rad: np.ndarray, q_star: np.ndarray, *, dt_s: float = 0.0
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
        if dt_s > 0.0:
            self._update_dwell(q, dt_s, q_star=q_star)
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
            sign = 1.0 if qs >= 0.0 else -1.0
            margin = sign * qi
            # Stay on the wrong side too — |q| > activate after a flip used
            # to drop the row and let the elbow keep going.
            if margin >= act:
                continue
            # sign * qdot_i + s_b ≥ −γ (sign * q_i − eps)
            # → blocks motion that would decrease signed margin below eps.
            jac = np.zeros(nv, dtype=float)
            jac[i] = sign
            rhs = -gamma * (margin - eps)
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

    def tighten_box(
        self,
        lo: np.ndarray,
        hi: np.ndarray,
        q: np.ndarray,
        q_star: np.ndarray,
        v_max: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hard Faverjon damper toward 0 for latched branch signs.

        Motion *away* from 0 is never restricted, so a planar J1≈0 start
        can still fold toward q*=−90°.  Crossing through 0 is closed at
        ``eps`` so QP1 cannot buy an elbow flip with TCP slack.
        """
        lo = np.asarray(lo, dtype=float).copy()
        hi = np.asarray(hi, dtype=float).copy()
        if not self.cfg.enabled:
            return lo, hi
        q = np.asarray(q, dtype=float).reshape(-1)
        q_star = np.asarray(q_star, dtype=float).reshape(-1)
        v_max = np.asarray(v_max, dtype=float).reshape(-1)
        act = float(self.cfg.box_activate_rad)
        if act <= 1.0e-9:
            act = float(self.cfg.activate_rad)
        eps = float(self.cfg.eps_rad)
        target_eps = float(self.cfg.target_eps_rad)
        band = max(act - eps, 1.0e-6)
        nv = min(q.size, q_star.size, lo.size, hi.size, v_max.size)
        for i in range(1, nv):
            qs = float(q_star[i])
            if abs(qs) <= target_eps:
                continue
            sign = 1.0 if qs >= 0.0 else -1.0
            margin = sign * float(q[i])
            d = float(np.clip((margin - eps) / band, 0.0, 1.0))
            vmax = abs(float(v_max[i]))
            if sign > 0.0:
                lo[i] = max(float(lo[i]), -vmax * d)
            else:
                hi[i] = min(float(hi[i]), vmax * d)
        return lo, hi


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

"""Nullspace secondary task: ascend Yoshikawa manipulability ∇μ(q).

During a large joint-space move near a kinematic singularity, Liegeois centering
pulls toward q_nominal (often a straight arm) and fights the plan.  This task
instead commands joint velocity along +∇μ so the redundant DOF bends away from
singular postures while the primary Cartesian / joint tracking task runs in the
task space.  The gradient is computed by central finite differences on
``RobotKinematics.manipulability`` — cheap enough at 200 Hz for nv=7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class ManipulabilityTaskConfig:
    k_mu: float = 0.8          # rad/s scale for ∂μ/∂q direction
    eps_rad: float = 1e-4      # finite-difference step per joint
    # Fade manipulability ascent when σ is already healthy (avoid fighting scan).
    sigma_fade_ref: float = 0.12
    # Soft-normalize floor: hard unit-norm turns tiny FD noise into ±k_mu bangs
    # ("move a bit, stop, move a bit").  Below this |∇μ|, command shrinks.
    grad_norm_floor: float = 0.02
    # EMA on commanded qdot (s); 0 disables.
    qdot_lpf_tau_s: float = 0.08


class ManipulabilityTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) along +∇μ."""

    def __init__(self, kin: RobotKinematics, cfg: ManipulabilityTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ManipulabilityTaskConfig()
        self.last_mu: float = 0.0
        self.last_grad_norm: float = 0.0
        self._qdot_filt: np.ndarray | None = None

    def reset(self) -> None:
        self._qdot_filt = None
        self.last_mu = 0.0
        self.last_grad_norm = 0.0

    def gradient(self, q_rad: np.ndarray, *, exclude_rail: bool = False) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        eps = max(float(self.cfg.eps_rad), 1e-6)
        mu0 = self.kin.manipulability(self.kin.jacobian(q))
        grad = np.zeros(self.kin.nv, dtype=float)
        for i in range(self.kin.nv):
            qp = q.copy()
            qm = q.copy()
            qp[i] += eps
            qm[i] -= eps
            mu_p = self.kin.manipulability(self.kin.jacobian(qp))
            mu_m = self.kin.manipulability(self.kin.jacobian(qm))
            grad[i] = (mu_p - mu_m) / (2.0 * eps)
        if exclude_rail:
            grad[0] = 0.0
        self.last_mu = mu0
        self.last_grad_norm = float(np.linalg.norm(grad))
        return grad

    def __call__(
        self,
        q_rad: np.ndarray,
        *,
        sigma_min: float = 1.0,
        exclude_rail: bool = False,
        dt_s: float = 0.005,
    ) -> np.ndarray:
        grad = self.gradient(q_rad, exclude_rail=exclude_rail)
        if self.last_grad_norm < 1e-12:
            qdot0 = np.zeros(self.kin.nv, dtype=float)
        else:
            # Soft normalize: full k_mu only when |∇μ| is well above the floor.
            soft = max(self.last_grad_norm, float(self.cfg.grad_norm_floor))
            qdot0 = self.cfg.k_mu * grad / soft
        ref = max(float(self.cfg.sigma_fade_ref), 1e-6)
        if sigma_min >= ref:
            fade = max(0.0, 1.0 - (sigma_min - ref) / ref)
            qdot0 = qdot0 * fade

        tau = float(self.cfg.qdot_lpf_tau_s)
        if tau <= 1e-6 or dt_s <= 1e-9:
            self._qdot_filt = np.asarray(qdot0, dtype=float).copy()
            return self._qdot_filt
        if self._qdot_filt is None or self._qdot_filt.shape != qdot0.shape:
            self._qdot_filt = np.asarray(qdot0, dtype=float).copy()
            return self._qdot_filt.copy()
        a = float(np.clip(dt_s / tau, 0.0, 1.0))
        self._qdot_filt = self._qdot_filt + a * (qdot0 - self._qdot_filt)
        return self._qdot_filt.copy()

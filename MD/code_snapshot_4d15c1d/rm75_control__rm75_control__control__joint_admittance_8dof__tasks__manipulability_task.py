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
    k_mu: float = 0.8          # rad/s per unit ∂μ/∂q (scaled by typical |∇μ|)
    eps_rad: float = 1e-4      # finite-difference step per joint
    # Fade manipulability ascent when σ is already healthy (avoid fighting scan).
    sigma_fade_ref: float = 0.12


class ManipulabilityTask:
    """Callable secondary task: q (rad) -> qdot0 (rad/s) along +∇μ."""

    def __init__(self, kin: RobotKinematics, cfg: ManipulabilityTaskConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or ManipulabilityTaskConfig()
        self.last_mu: float = 0.0
        self.last_grad_norm: float = 0.0

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

    def __call__(self, q_rad: np.ndarray, *, sigma_min: float = 1.0, exclude_rail: bool = False) -> np.ndarray:
        grad = self.gradient(q_rad, exclude_rail=exclude_rail)
        if self.last_grad_norm < 1e-12:
            return np.zeros(self.kin.nv, dtype=float)
        # Unit direction × gain; typical |∇μ| is O(0.01–0.1) near singularities.
        qdot0 = self.cfg.k_mu * grad / self.last_grad_norm
        ref = max(float(self.cfg.sigma_fade_ref), 1e-6)
        if sigma_min >= ref:
            fade = max(0.0, 1.0 - (sigma_min - ref) / ref)
            qdot0 = qdot0 * fade
        return qdot0

"""Lillo-style set-based σ inequality for online velocity IK.

Soft manipulability (nullspace) always prefers high σ.  Near a threshold this
module activates a recoverable unilateral inequality

    ∇σᵀ q̇ + s_σ ≥ −γ (σ − σ_safe)     when σ < σ_act (hysteresis)

so the QP cannot silently drive further into singularity; s_σ ≥ 0 is a
dedicated high-cost slack (Escande: prefer task slack over penetrating σ_safe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RobotKinematics


@dataclass
class SigmaSetBasedConfig:
    enabled: bool = True
    activate: float = 0.14
    safe: float = 0.06
    exit: float = 0.18
    gamma: float = 8.0
    slack_weight: float = 200.0
    grad_eps: float = 1.0e-4


@dataclass
class PrefInequalityRows:
    """Packed preference inequalities: J_pref qdot + S s_pref >= lower."""

    jacobian: np.ndarray  # (n_rows, nv)
    slack_col: np.ndarray  # (n_rows,) which pref-slack index (0=sigma, 1=branch)
    lower: np.ndarray  # (n_rows,)
    active: bool = False


class SigmaSetBasedTracker:
    """Hysteresis activation + arm-σ gradient for the set-based row."""

    def __init__(self, cfg: SigmaSetBasedConfig | None = None) -> None:
        self.cfg = cfg or SigmaSetBasedConfig()
        self.active: bool = False
        self.last_sigma: float = float("nan")
        self.last_grad: np.ndarray | None = None
        self.last_slack: float = 0.0

    def reset(self) -> None:
        self.active = False
        self.last_sigma = float("nan")
        self.last_grad = None
        self.last_slack = 0.0

    def update_hysteresis(self, sigma: float) -> bool:
        cfg = self.cfg
        if not cfg.enabled:
            self.active = False
            return False
        sig = float(sigma)
        self.last_sigma = sig
        enter = float(cfg.activate)
        exit_ = max(float(cfg.exit), enter)
        if self.active:
            if sig >= exit_:
                self.active = False
        else:
            if sig < enter:
                self.active = True
        return self.active

    def arm_sigma_and_grad(
        self, kin: RobotKinematics, q_rad: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """σ_min(J_arm) and finite-difference ∇_q σ_arm."""
        q = np.asarray(q_rad, dtype=float)
        nv = int(q.size)
        eps = float(self.cfg.grad_eps)
        J0 = kin.jacobian(q)
        sig0 = float(np.linalg.svd(J0[:, 1:], compute_uv=False).min())
        g = np.zeros(nv, dtype=float)
        for i in range(nv):
            dq = np.zeros(nv, dtype=float)
            dq[i] = eps
            Jp = kin.jacobian(q + dq)
            sig_p = float(np.linalg.svd(Jp[:, 1:], compute_uv=False).min())
            g[i] = (sig_p - sig0) / eps
        self.last_grad = g
        return sig0, g

    def build_row(
        self, kin: RobotKinematics, q_rad: np.ndarray
    ) -> PrefInequalityRows:
        nv = int(np.asarray(q_rad).size)
        empty = PrefInequalityRows(
            jacobian=np.zeros((0, nv)),
            slack_col=np.zeros(0, dtype=int),
            lower=np.zeros(0),
            active=False,
        )
        if not self.cfg.enabled:
            return empty
        sigma, grad = self.arm_sigma_and_grad(kin, q_rad)
        if not self.update_hysteresis(sigma):
            return empty
        # ∇σᵀ q̇ + s ≥ −γ(σ − σ_safe)  ⇔  ∇σᵀ q̇ + s ≥ γ(σ_safe − σ)
        rhs = float(self.cfg.gamma) * (float(self.cfg.safe) - float(sigma))
        return PrefInequalityRows(
            jacobian=grad.reshape(1, nv),
            slack_col=np.array([0], dtype=int),
            lower=np.array([rhs], dtype=float),
            active=True,
        )


__all__ = [
    "SigmaSetBasedConfig",
    "SigmaSetBasedTracker",
    "PrefInequalityRows",
]

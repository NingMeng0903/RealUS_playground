"""Shared IK types and utilities for the WBC inner loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IkStepResult:
    """One WBC QP velocity-IK step (all joint quantities in rad, rad/s)."""

    q_next: np.ndarray
    qdot: np.ndarray
    sigma_min: float
    manip: float
    slack_norm: float = 0.0
    n_cbf_active: int = 0


@dataclass
class SrDampingConfig:
    """Singularity-robust (SR) damping for nullspace projection (Chiaverini 1997).

    ``lam0`` is the baseline damped-least-squares λ when the arm is well-
    conditioned (``sigma_min >= sigma_ref``).  Below ``sigma_ref``, λ ramps up
    as ``lam0 * (sigma_ref / sigma)^2`` so the task Jacobian pseudoinverse
    contribution vanishes and the nullspace projector N → I — secondary tasks
    and joint feedforward regain control of directions the primary task cannot
    use near a kinematic singularity.
    """

    lam0: float = 0.05
    sigma_ref: float = 0.08
    sigma_floor: float = 1e-6


def sr_damping_lambda(sigma_min: float, cfg: SrDampingConfig | None = None) -> float:
    """Return SR damping λ(σ) for ``project_onto_task_nullspace`` / DLS."""
    cfg = cfg or SrDampingConfig()
    sigma = max(float(sigma_min), cfg.sigma_floor)
    if sigma >= cfg.sigma_ref:
        return cfg.lam0
    return cfg.lam0 * (cfg.sigma_ref / sigma) ** 2


def saturate_error(err: np.ndarray, max_pos: float, max_rot: float) -> np.ndarray:
    """Norm-clamp a 6D pose error (linear part to max_pos, angular to max_rot)."""
    out = np.asarray(err, dtype=float).copy()
    pos_n = float(np.linalg.norm(out[:3]))
    if max_pos > 0.0 and pos_n > max_pos:
        out[:3] *= max_pos / pos_n
    rot_n = float(np.linalg.norm(out[3:6]))
    if max_rot > 0.0 and rot_n > max_rot:
        out[3:6] *= max_rot / rot_n
    return out


def project_onto_task_nullspace(
    J: np.ndarray,
    qdot0: np.ndarray,
    *,
    sigma_min: float | None = None,
    damping: float | None = None,
    sr_cfg: SrDampingConfig | None = None,
    M: np.ndarray | None = None,
    use_dyn: bool = False,
    m_floor: float = 0.05,
) -> np.ndarray:
    """Liegeois (kinematic) or Khatib (dynamics-consistent) nullspace projection.

    When ``use_dyn`` and ``M`` are supplied, uses
    ``N_dyn = I - M^{-1} J^T (J M^{-1} J^T + λI)^{-1} J`` so secondary motion
    does not produce task-space wrenches at the acceleration level.
    """
    if use_dyn and M is not None:
        return project_onto_task_nullspace_dyn(
            J, M, qdot0, sigma_min=sigma_min, damping=damping, sr_cfg=sr_cfg,
            m_floor=m_floor,
        )
    qdot0 = np.asarray(qdot0, dtype=float)
    if damping is None:
        if sigma_min is not None:
            damping = sr_damping_lambda(sigma_min, sr_cfg)
        else:
            damping = 1e-4
    m = J.shape[0]
    lam2I = (damping * damping) * np.eye(m)
    Jd = J.T @ np.linalg.solve(J @ J.T + lam2I, np.eye(m))
    N = np.eye(J.shape[1]) - Jd @ J
    return N @ qdot0


def project_onto_task_nullspace_dyn(
    J: np.ndarray,
    M: np.ndarray,
    qdot0: np.ndarray,
    *,
    sigma_min: float | None = None,
    damping: float | None = None,
    sr_cfg: SrDampingConfig | None = None,
    m_floor: float = 0.05,
) -> np.ndarray:
    """Dynamically consistent nullspace projector (Khatib 1987).

    ``m_floor`` regularizes the joint-space inertia (``M + m_floor*I``): the
    RM75 URDF's wrist inertias are ~1e-4 kg m^2, so the raw ``M^{-1}`` blows
    those rows up ~1e4x and the oblique projector then AMPLIFIES the small
    out-of-nullspace residue of a damped secondary task instead of removing it
    - observed as the projected task pointing the WRONG way (nullspace
    twist-oscillation on hardware, arm-angle divergence offline).  Vectors
    exactly in ker(J) are untouched by the floor.
    """
    qdot0 = np.asarray(qdot0, dtype=float)
    J = np.asarray(J, dtype=float)
    M = np.asarray(M, dtype=float)
    nv = J.shape[1]
    if damping is None:
        if sigma_min is not None:
            damping = sr_damping_lambda(sigma_min, sr_cfg)
        else:
            damping = 1e-4
    m = J.shape[0]
    if m_floor > 0.0:
        M = M + m_floor * np.eye(nv)
    Minv = np.linalg.inv(M)
    JMinv = J @ Minv
    lam2I = (damping * damping) * np.eye(m)
    Jbar = Minv @ J.T @ np.linalg.solve(JMinv @ J.T + lam2I, np.eye(m))
    N = np.eye(nv) - Jbar @ J
    return N @ qdot0

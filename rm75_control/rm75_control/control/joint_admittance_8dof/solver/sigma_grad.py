"""TCP-preserving directional derivative of σ_min w.r.t. rail translation.

World-frame J is independent of q_rail here, so ∂σ/∂q_rail is zero.  Instead
move rail by δy with arm δq_arm = -J_arm⁺ e_rail δy (hold TCP); σ under that
coordinated move is what σ-escape needs.  Central difference, 2 Jacobians.
"""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.model import RAIL_INDEX, RobotKinematics


def _sigma_min(J: np.ndarray) -> float:
    return float(np.linalg.svd(J, compute_uv=False).min())


def sigma_min_grad_rail(
    kin: RobotKinematics,
    q_rad: np.ndarray,
    eps: float = 1.0e-3,
) -> float:
    """Directional derivative ``d σ_min / d y_rail`` under TCP-preservation.

    Positive value → moving the rail in +Y increases the arm's conditioning
    (helps escape a singularity); negative → −Y direction helps instead.
    Returns 0.0 when ``J_arm`` is itself rank-deficient (rare — happens only
    at deep singularities where the whole task is already infeasible).
    """
    q = np.asarray(q_rad, dtype=float)
    J = kin.jacobian(q)
    # J_arm: columns 1..7 (the 7-DOF arm), J_rail: column 0.
    J_arm = np.delete(J, RAIL_INDEX, axis=1)
    e_rail = J[:, RAIL_INDEX]
    # Damped least-squares pseudoinverse (small damping keeps this smooth
    # near singularities — the analytical J_arm^+ blows up right where we
    # want the escape term most).
    lam = 5.0e-3
    try:
        dq_arm = -np.linalg.solve(
            J_arm.T @ J_arm + lam * lam * np.eye(J_arm.shape[1]),
            J_arm.T @ e_rail,
        )
    except np.linalg.LinAlgError:
        return 0.0
    # Central difference under the coordinated move.
    q_p = q.copy()
    q_m = q.copy()
    q_p[RAIL_INDEX] += eps
    q_m[RAIL_INDEX] -= eps
    # scatter dq_arm into the non-rail slots
    arm_slots = [i for i in range(q.shape[0]) if i != RAIL_INDEX]
    for k, slot in enumerate(arm_slots):
        q_p[slot] += eps * dq_arm[k]
        q_m[slot] -= eps * dq_arm[k]
    sig_p = _sigma_min(kin.jacobian(q_p))
    sig_m = _sigma_min(kin.jacobian(q_m))
    return float((sig_p - sig_m) / (2.0 * eps))

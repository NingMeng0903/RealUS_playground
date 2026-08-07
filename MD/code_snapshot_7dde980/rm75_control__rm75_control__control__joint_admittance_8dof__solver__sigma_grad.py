"""Analytical / semi-analytical σ_min gradient for the rail coordinator.

The plan (Bug 2) asks for ``∂σ_min/∂y_rail`` so the rail-extension task can
add a *σ-escape* velocity component that kicks in inside the reach dead zone
whenever the arm approaches a singularity.

A subtlety: for our 8-DOF ``J = [J_rail | J_arm]`` the rail is a pure y-translation
of the base and pinocchio's world-frame Jacobian is **exactly independent of
``q_rail``** (verified empirically: ``‖J(q)-J(q + δ·e_rail)‖ = 0``).  So the naive
``∂σ_min/∂q_rail = u_min^T ∂J/∂q_rail v_min`` is identically zero and would leave
the σ-escape term inert.

The physically meaningful quantity is a *directional* derivative under
TCP-preservation: if the rail moves by ``δy``, the arm must move by
``δq_arm = -J_arm^+ · e_rail · δy`` to keep the TCP fixed in world.  Under that
coordinated move the full-configuration ``σ_min`` DOES change, and its slope in
that direction is a well-defined "if I recruit the rail, how much does the
arm's conditioning improve?" quantity — exactly what the σ-escape term wants.

We compute it by central-difference on the coordinated move (2 Jacobians per
sample) rather than an 8-column analytical Hessian.  Cache callers should
re-evaluate at a modest rate (e.g. every 10 ticks — the RM75 tick is 200 Hz,
so the gradient updates at 20 Hz which is way above the rail acceleration
bandwidth).
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

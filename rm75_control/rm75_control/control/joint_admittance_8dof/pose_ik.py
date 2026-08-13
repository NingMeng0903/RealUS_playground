"""One-shot pose IK for the 8-DOF stack (no vendor rm_algo_inverse_kinematics).

``resolve_pose_ik_srs`` is the preferred closed-form SRS + ψ enum + path
check planner.  ``solve_pose_ik`` is the compatibility entry point for the
few offline callers that still need a general numerical solve; it delegates
to the bounded fixed-rail solver in :mod:`numerical_pose_ik` and never owns an
online QPIK controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
    pose_error,
)
from rm75_control.control.joint_admittance_8dof.numerical_pose_ik import (
    CollisionCheck,
    NumericalPoseIkConfig,
    NumericalPoseIkResult,
    solve_numerical_pose_ik,
)
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    flange_tcp_from_kin,
    psi_from_q,
    shoulder_y_from_q_rail,
    srs_ik,
)


class UnreachablePathError(RuntimeError):
    """No ψ candidate yields a globally reachable path (fail loud; re-teach)."""


@dataclass
class PoseIkReport:
    """Convergence diagnostics from :func:`resolve_pose_ik_srs`."""
    pos_err_mm: float
    rot_err_deg: float
    sigma_min: float
    iters: int
    within_limits: bool
    psi_deg: float = float("nan")
    psi_home_deg: float = float("nan")
    path_ok: bool = True


@dataclass
class PlannerGoalWeights:
    """Weights for the SRS planner's goal_score (higher = better posture).

    The score is
        s = -w_home · ((ψ − ψ_home)/π)²
            -w_sigma_floor · max(0, sigma_safe − sigma_min)
            -w_limit · Σ ((q_i − q_mid_i)/q_range_i)²
            -w_wrist · exp(-8 · sin²(q5))
            -w_elbow · max(0, 0.3 − sin(q4))

    ψ_home is the PRIMARY attractor; sigma / limit / wrist / elbow are
    thresholds that keep the candidate feasible / comfortable but do not
    compete with ψ_home unless ψ_home itself lands in trouble.
    """
    w_home: float = 1.0
    sigma_safe: float = 0.08
    w_sigma_floor: float = 100.0
    w_limit: float = 0.5
    w_wrist: float = 0.3
    w_elbow: float = 0.5


def _wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _slerp_pose(p0: np.ndarray, p1: np.ndarray, s: float, euler_order: str = "xyz") -> np.ndarray:
    """Constant-speed SE(3) interpolation: position lerp + rotation SLERP.

    Both endpoints are 6-vec ``[x, y, z, rx, ry, rz]`` (matches fk_pose).
    ``s`` in [0, 1].
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    R_stack = Rsc.from_euler(euler_order, np.stack([p0[3:6], p1[3:6]]), degrees=False)
    key_times = [0.0, 1.0]
    slerp = Slerp(key_times, R_stack)
    R_s = slerp([float(np.clip(s, 0.0, 1.0))])[0]
    pos = (1.0 - s) * p0[:3] + s * p1[:3]
    out = np.zeros(6, dtype=float)
    out[:3] = pos
    out[3:6] = R_s.as_euler(euler_order, degrees=False)
    return out


def goal_score(
    q_arm: np.ndarray,
    q_full: np.ndarray,
    psi: float,
    psi_home: float,
    sigma_min: float,
    kin: RobotKinematics,
    weights: PlannerGoalWeights,
) -> float:
    """Higher = more desirable posture.  See PlannerGoalWeights docstring."""
    d_home = _wrap_pi(psi - psi_home) / np.pi          # ∈ [-1, 1]
    home_penalty = weights.w_home * d_home * d_home

    sigma_penalty = weights.w_sigma_floor * max(0.0, weights.sigma_safe - sigma_min)

    q_range = Q_UPPER - Q_LOWER
    q_mid = 0.5 * (Q_UPPER + Q_LOWER)
    u = (q_arm - q_mid) / np.maximum(q_range, 1e-6)
    limit_penalty = weights.w_limit * float(np.sum(u * u))

    # Wrist singularity proxy: exp(-8·sin²(q5)) is ~1 at q5 ≈ 0 / ±π, ~0 elsewhere.
    # RM75's Z-Y-Z wrist loses rank at J6, arm-vector index 5.
    wrist_penalty = weights.w_wrist * float(np.exp(-8.0 * np.sin(q_arm[5]) ** 2))

    # Straight-elbow penalty: sin(q4) < 0.3 means elbow bent < ~17.5°, i.e.
    # near-straight arm — dangerous (approaches the SRS shoulder-arm-wrist
    # collinear singularity used by the arm_angle observability decay).
    elbow_penalty = weights.w_elbow * max(0.0, 0.3 - float(np.sin(q_arm[3])))

    return -(home_penalty + sigma_penalty + limit_penalty + wrist_penalty + elbow_penalty)


def _path_reachable(
    kin: RobotKinematics,
    pose_seed: np.ndarray,
    pose_target: np.ndarray,
    psi_seed: float,
    psi_target: float,
    branch: int,
    y_rail_seed: float,
    y_rail_target: float,
    *,
    n_samples: int = 10,
    euler_order: str = "xyz",
    R_flange_tcp: np.ndarray | None = None,
    t_flange_tcp: np.ndarray | None = None,
) -> bool:
    """True iff srs_ik succeeds at every interior sample of the (pose, ψ, y_rail)
    interpolation.  Endpoints are excluded: they are guaranteed by the seed
    (feasibility already verified for the seed) and by the enumeration itself.
    """
    # Unwrap ψ so linear interpolation goes the short way and does not cross ±π.
    psi_target_unwrapped = psi_seed + _wrap_pi(psi_target - psi_seed)
    for i in range(1, n_samples + 1):
        s = i / (n_samples + 1)                           # 1/(n+1) ... n/(n+1)
        pose_s = _slerp_pose(pose_seed, pose_target, s, euler_order)
        psi_s = psi_seed + s * (psi_target_unwrapped - psi_seed)
        y_rail_s = y_rail_seed + s * (y_rail_target - y_rail_seed)
        q_arm = srs_ik(
            pose_s,
            psi_s,
            branch,
            y_rail=shoulder_y_from_q_rail(y_rail_s),
            euler_order=euler_order,
            R_flange_tcp=R_flange_tcp,
            t_flange_tcp=t_flange_tcp,
        )
        if q_arm is None:
            return False
    return True


def resolve_pose_ik_srs(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    q_branch_seed: np.ndarray | None = None,
    y_rail_target: float | None = None,
    psi_home_rad: float | None = None,
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0,
    psi_hard_lower_rad: float | None = None,
    psi_hard_upper_rad: float | None = None,
    planner_weights: PlannerGoalWeights | None = None,
    psi_grid_step_rad: float = 5.0 * np.pi / 180.0,
    path_check_samples: int = 10,
    top_k_for_path_check: int = 5,
    require_path: bool = True,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """SRS closed-form IK + 1-D ψ grid enumeration + path reachability check.

    Returns ``(q_target_full_rad, ok, report)`` where ``q_target_full_rad``
    is an 8-vec with the rail entry set to ``y_rail_target`` (or
    ``q_seed[0]`` if the caller left it None).

    Enumeration rules (in priority order):

    1. Reject ψ candidates outside ``[psi_hard_lower_rad, psi_hard_upper_rad]``
       (if provided) and outside ``|wrap(ψ − ψ_seed)| ≤ max_psi_swing_rad``.
    2. Reject candidates whose srs_ik is None (branch unreachable / hits
       shoulder or wrist singularity / violates URDF joint limits).
    3. Rank surviving candidates by :func:`goal_score` and take the top-K.
    4. For each top-K candidate, verify the whole interpolation path
       ``(pose_seed, ψ_seed) → (pose_target, ψ_candidate)`` is srs_ik-solvable
       at ``path_check_samples`` interior points.
    5. Return the highest-scoring candidate whose path check passes.

    Raises
    ------
    UnreachablePathError
        If no candidate survives the path check.  The caller must re-teach
        the target pose or the seed rather than silently accepting a plan
        that will stall mid-move.
    """
    weights = planner_weights or PlannerGoalWeights()
    q_seed = np.asarray(q_seed, dtype=float).copy()
    if q_seed.size != 8:
        raise ValueError(f"q_seed must be 8-vec, got size {q_seed.size}")
    q_arm_seed = q_seed[1:]
    q_branch_src = (
        np.asarray(q_branch_seed, dtype=float).copy()
        if q_branch_seed is not None
        else q_seed
    )
    if q_branch_src.size != 8:
        raise ValueError(f"q_branch_seed must be 8-vec, got size {q_branch_src.size}")
    y_rail_seed = float(q_seed[RAIL_INDEX])
    y_rail_target = float(q_seed[RAIL_INDEX] if y_rail_target is None else y_rail_target)

    pose_seed = kin.fk_pose(q_seed)
    psi_seed = psi_from_q(q_arm_seed)
    branch_seed = branch_from_q(q_branch_src[1:])
    psi_home = float(psi_seed if psi_home_rad is None else psi_home_rad)
    R_flange_tcp, t_flange_tcp = flange_tcp_from_kin(kin)

    # Candidate ψ grid on (-π, π].  max_psi_swing is measured from ψ_home
    # (the posture attractor), NOT from ψ_seed — so a live q0 at ψ≈72° can
    # still pick a ψ near 72° even when the taught slot branch differs.
    psi_grid = np.arange(-np.pi, np.pi, float(psi_grid_step_rad))
    scored: list[tuple[float, float, np.ndarray, float]] = []  # (score, psi, q_arm, sigma_min)
    for psi in psi_grid:
        d_home = abs(_wrap_pi(float(psi) - psi_home))
        if d_home > float(max_psi_swing_rad):
            continue
        # Hard bounds (cable-carrier / cabin envelope):
        if psi_hard_lower_rad is not None and float(psi) < float(psi_hard_lower_rad):
            continue
        if psi_hard_upper_rad is not None and float(psi) > float(psi_hard_upper_rad):
            continue

        q_arm = srs_ik(
            pose_target, float(psi), branch_seed,
            y_rail=shoulder_y_from_q_rail(y_rail_target),
            euler_order=euler_order,
            R_flange_tcp=R_flange_tcp,
            t_flange_tcp=t_flange_tcp,
        )
        if q_arm is None:
            continue
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        J = kin.jacobian(q_full)
        sigma_min = float(kin.singular_values(J).min())
        score = goal_score(q_arm, q_full, float(psi), psi_home, sigma_min, kin, weights)
        scored.append((score, float(psi), q_arm, sigma_min))

    if not scored:
        raise UnreachablePathError(
            "SRS IK found no reachable ψ candidate for pose_target — "
            "check max_psi_swing_rad, psi_hard_*, or re-teach the target pose."
        )

    scored.sort(key=lambda x: x[0], reverse=True)     # highest score first
    top_k = scored[: max(1, int(top_k_for_path_check))]

    def _report_from(
        psi: float,
        q_arm: np.ndarray,
        sigma_min: float,
        *,
        path_ok: bool,
    ) -> tuple[np.ndarray, bool, PoseIkReport]:
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        pose_ach = kin.fk_pose(q_full)
        err = pose_error(pose_target, pose_ach, euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        within = bool(
            np.all(q_full[1:] >= Q_LOWER - 1e-6)
            and np.all(q_full[1:] <= Q_UPPER + 1e-6)
        )
        report = PoseIkReport(
            pos_err_mm=pos_err_m * 1000.0,
            rot_err_deg=float(np.degrees(rot_err_rad)),
            sigma_min=sigma_min,
            iters=0,
            within_limits=within,
            psi_deg=float(np.degrees(psi)),
            psi_home_deg=float(np.degrees(psi_home)),
            path_ok=path_ok,
        )
        ok = path_ok and within and pos_err_m <= 0.005 and rot_err_rad <= np.deg2rad(2.0)
        return q_full, ok, report

    if not require_path:
        # Path check skipped (e.g. MoveJ).  Mark path_ok=True so ``ok`` reflects
        # pose accuracy / limits only — not a false reject with 0 mm / 0 deg error.
        score, psi, q_arm, sigma_min = top_k[0]
        return _report_from(psi, q_arm, sigma_min, path_ok=True)

    # Path reachability check on the top-K candidates.
    for score, psi, q_arm, sigma_min in top_k:
        if _path_reachable(
            kin,
            pose_seed=pose_seed,
            pose_target=pose_target,
            psi_seed=psi_seed,
            psi_target=psi,
            branch=branch_seed,
            y_rail_seed=y_rail_seed,
            y_rail_target=y_rail_target,
            n_samples=int(path_check_samples),
            euler_order=euler_order,
            R_flange_tcp=R_flange_tcp,
            t_flange_tcp=t_flange_tcp,
        ):
            return _report_from(psi, q_arm, sigma_min, path_ok=True)

    # None of the top-K candidates has a fully reachable path.
    _, psi_best, q_arm_best, sigma_best = top_k[0]
    raise UnreachablePathError(
        f"pose IK: top-{len(top_k)} ψ candidates all fail path reachability. "
        f"Best ψ={np.degrees(psi_best):.1f}° from ψ_seed={np.degrees(psi_seed):.1f}° "
        f"(branch from {'branch_seed' if q_branch_seed is not None else 'q_seed'}) — "
        f"either the pose is too far from the seed or ψ_home is unreachable at this pose. "
        f"Please re-teach the target pose or adjust psi_home_deg / max_psi_swing_deg."
    )


def solve_pose_ik(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    rail_m: float | None = None,
    config: NumericalPoseIkConfig | None = None,
    collision_check: CollisionCheck | None = None,
) -> NumericalPoseIkResult:
    """Solve ``q_seed`` → ``pose_target`` with the rail held fixed.

    This small compatibility wrapper deliberately exposes only the numerical
    solver's safety boundary.  The default rail is the seed rail for a full
    eight-joint state (or zero for an arm-only seed); callers that need a
    different fixed coordinate must pass ``rail_m`` explicitly.  The returned
    :class:`NumericalPoseIkResult` remains tuple-unpackable as
    ``(q_target, ok, report)`` for existing offline callers.
    """
    q_seed_arr = np.asarray(q_seed, dtype=float).reshape(-1)
    if rail_m is None:
        rail_m = float(q_seed_arr[RAIL_INDEX]) if q_seed_arr.size == kin.nq else 0.0
    return solve_numerical_pose_ik(
        kin,
        q_seed_arr,
        np.asarray(pose_target, dtype=float),
        rail_m=float(rail_m),
        config=config,
        collision_check=collision_check,
    )


__all__ = [
    "PlannerGoalWeights",
    "PoseIkReport",
    "UnreachablePathError",
    "goal_score",
    "resolve_pose_ik_srs",
    "solve_pose_ik",
]

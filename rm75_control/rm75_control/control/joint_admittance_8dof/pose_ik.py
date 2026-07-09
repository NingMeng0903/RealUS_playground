"""One-shot pose inverse kinematics for the 8-DOF stack.

Two entry points:

* :func:`resolve_pose_ik_srs` — SRS closed-form IK + 1-D ψ enumeration + path
  reachability check.  Preferred: analytical, no QP iterations, jointly
  selects the swivel branch that is closest to a global ``psi_home`` while
  avoiding singularities/limits/wrist locks.  Fails loud on unreachable
  poses (``UnreachablePathError``) so the caller re-teaches instead of
  silently degrading.

* :func:`solve_pose_ik` — legacy iterative WBC IK (retained for
  backward-compat; still used by tools/reachability scripts).  This is the
  "gradient-descent-of-pose-error via slack QP" path; when ``attractor_q``
  is ``None`` (its new default) it uses ``q_seed`` as the centering target
  so the resolved posture stays on the teach branch rather than being
  pulled toward a yaml zero.

Planning-only helpers: they resolve ``q_target`` for a desired TCP pose
without any vendor ``rm_algo_inverse_kinematics`` call — this is the ONLY
IK path allowed for large point-to-point moves (see MD/debug.md
architecture constraint: no black-box vendor IK, ever).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import Slerp

from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.model import (
    RAIL_INDEX,
    RobotKinematics,
    full_q_from_arm,
    pose_error,
)
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits
from rm75_control.kinematics.srs_ik import (
    Q_LOWER,
    Q_UPPER,
    branch_from_q,
    psi_from_q,
    srs_ik,
)


class UnreachablePathError(RuntimeError):
    """Raised when no ψ candidate produces a globally reachable path from
    (pose_seed, ψ_seed) to (pose_target, ψ_target).  This is deliberately a
    hard failure: silently accepting an "almost feasible" plan is what caused
    the mid-move singularity stalls we are trying to eliminate.
    """


@dataclass
class PoseIkReport:
    """Convergence diagnostics from ``solve_pose_ik`` / ``resolve_pose_ik_srs``."""
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


def _goal_score(
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
    wrist_penalty = weights.w_wrist * float(np.exp(-8.0 * np.sin(q_arm[4]) ** 2))

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
        q_arm = srs_ik(pose_s, psi_s, branch, y_rail=y_rail_s, euler_order=euler_order)
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
    3. Rank surviving candidates by :func:`_goal_score` and take the top-K.
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
            y_rail=y_rail_target, euler_order=euler_order,
        )
        if q_arm is None:
            continue
        q_full = full_q_from_arm(q_arm, rail_m=y_rail_target)
        J = kin.jacobian(q_full)
        sigma_min = float(kin.singular_values(J).min())
        score = _goal_score(q_arm, q_full, float(psi), psi_home, sigma_min, kin, weights)
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
        score, psi, q_arm, sigma_min = top_k[0]
        return _report_from(psi, q_arm, sigma_min, path_ok=False)

    # Bug 7a: path reachability check on the top-K candidates.
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


def resolve_pose_ik_for_move(
    kin: RobotKinematics,
    q0_rad: np.ndarray,
    q_slot_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    y_rail_target: float | None = None,
    psi_home_rad: float | None = None,
    max_psi_swing_rad: float = 150.0 * np.pi / 180.0,
    psi_hard_lower_rad: float | None = None,
    psi_hard_upper_rad: float | None = None,
    planner_weights: PlannerGoalWeights | None = None,
    euler_order: str = "xyz",
) -> tuple[np.ndarray, bool, PoseIkReport, bool]:
    """Move-aware SRS IK: live q0 path + taught slot branch.

    Returns ``(q_target, ok, report, use_srs_move_ref)``.

    * ``q_seed=q0`` for path reachability (actual move start).
    * ``q_branch_seed=q_slot`` for elbow/wrist branch at pose D.
    * ``psi_home`` defaults to ψ(q0) unless yaml overrides.

    If the full path check fails (common when q0 is far from the taught
    slot, e.g. home → D), falls back to goal-only IK and signals
    ``use_srs_move_ref=False`` so the caller uses joint interpolation
    instead of :class:`SrsSmoothMoveReference`.
    """
    q0 = np.asarray(q0_rad, dtype=float)
    q_slot = np.asarray(q_slot_rad, dtype=float)
    psi_live = float(psi_from_q(q0[1:]))
    psi_home = float(psi_live if psi_home_rad is None else psi_home_rad)
    common = dict(
        pose_target=pose_target,
        y_rail_target=y_rail_target,
        psi_home_rad=psi_home,
        max_psi_swing_rad=max_psi_swing_rad,
        psi_hard_lower_rad=psi_hard_lower_rad,
        psi_hard_upper_rad=psi_hard_upper_rad,
        planner_weights=planner_weights,
        euler_order=euler_order,
        q_branch_seed=q_slot,
    )
    try:
        q_tgt, ok, rep = resolve_pose_ik_srs(kin, q_seed=q0, require_path=True, **common)
        return q_tgt, ok, rep, True
    except UnreachablePathError:
        q_tgt, ok, rep = resolve_pose_ik_srs(
            kin, q_seed=q0, require_path=False, **common
        )
        return q_tgt, ok, rep, False


def solve_pose_ik(
    kin: RobotKinematics,
    q_seed: np.ndarray,
    pose_target: np.ndarray,
    *,
    max_iters: int = 500,
    pos_tol_m: float = 1e-3,
    rot_tol_rad: float = 0.02,
    dt: float = 0.02,
    k_gain: float = 3.0,
    max_pos_err_m: float = 0.05,
    max_rot_err_rad: float = 0.20,
    qp_cfg: QpConfig | None = None,
    nullspace_cfg: NullspaceTaskConfig | None = None,
    attractor_q: np.ndarray | None = None,
    trace: list[dict] | None = None,
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """Iterative WBC IK (legacy path): ``q_seed`` -> ``q`` with fk(q) ≈ pose_target.

    Each iteration feeds ``v_cmd = k_gain · saturate(pose_error)`` to the QP.

    ``attractor_q`` sets the ``JointCenteringTask`` target for the nullspace
    pull.  When ``None`` (the new default), we use ``q_seed`` itself — this
    matches Bug 4 of the SRS+Rail fix: the old default read
    ``nullspace_cfg.q_nominal_rad`` which was all-zeros in yaml and pulled the
    IK toward a straight arm (J4 → 0, σ_min → 0).  Prefer
    :func:`resolve_pose_ik_srs` when you have SRS geometry (all 8-DOF-stack
    call sites do).
    """
    cfg = qp_cfg or QpConfig()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.9, a_max=50.0)
    ctrl = QpIkController(kin, limits, cfg)

    task: JointCenteringTask | None = None
    if nullspace_cfg is not None:
        # Attractor selection (Bug 4 Step A):
        #   attractor_q explicit    → use it verbatim
        #   attractor_q None        → use q_seed (the teach posture)
        # Only fall through to nullspace_cfg.q_nominal_rad if the caller
        # cleared attractor_q AND the config still has an explicit q_nominal.
        # yaml default of ``q_nominal_deg: null`` (Bug 4) means q_seed wins.
        target = np.asarray(
            attractor_q if attractor_q is not None else q_seed,
            dtype=float,
        )
        cfg_used = NullspaceTaskConfig(
            k_center=nullspace_cfg.k_center,
            k_limit=nullspace_cfg.k_limit,
            activation=nullspace_cfg.activation,
            weights=nullspace_cfg.weights,
            q_nominal_rad=target,
        )
        task = JointCenteringTask.from_kinematics(kin, cfg_used)

    q = np.clip(np.asarray(q_seed, dtype=float).copy(), kin.q_lower, kin.q_upper)
    pose_target = np.asarray(pose_target, dtype=float)
    ctrl.reset(q)

    sigma_last = float("nan")
    pos_err_m = float("nan")
    rot_err_rad = float("nan")
    for it in range(max_iters):
        err = pose_error(pose_target, kin.fk_pose(q), cfg.euler_order)
        pos_err_m = float(np.linalg.norm(err[:3]))
        rot_err_rad = float(np.linalg.norm(err[3:6]))
        if pos_err_m < pos_tol_m and rot_err_rad < rot_tol_rad:
            report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, it, sigma_last)
            if trace is not None:
                trace.append(
                    {
                        "iter": it,
                        "pos_err_mm": pos_err_m * 1000.0,
                        "rot_err_deg": np.degrees(rot_err_rad),
                        "v_cmd_norm": 0.0,
                        "slack_norm": None,
                        "n_cbf_active": None,
                        "sigma_min": report.sigma_min,
                        "converged": True,
                    }
                )
            return q, True, report
        err_sat = saturate_error(err, max_pos_err_m, max_rot_err_rad)
        v_cmd = k_gain * err_sat
        secondary = task(q) if task is not None else None
        r = ctrl.step(q, v_cmd, dt, secondary_qdot=secondary)
        sigma_last = r.sigma_min
        if trace is not None:
            trace.append(
                {
                    "iter": it,
                    "pos_err_mm": pos_err_m * 1000.0,
                    "rot_err_deg": np.degrees(rot_err_rad),
                    "v_cmd_norm": float(np.linalg.norm(v_cmd)),
                    "slack_norm": r.slack_norm,
                    "n_cbf_active": r.n_cbf_active,
                    "sigma_min": r.sigma_min,
                    "converged": False,
                }
            )
        q = np.clip(r.q_next, kin.q_lower, kin.q_upper)

    report = _make_report(q, kin, ctrl, pos_err_m, rot_err_rad, max_iters, sigma_last)
    return q, False, report


def _make_report(
    q: np.ndarray,
    kin: RobotKinematics,
    ctrl: QpIkController,
    pos_err_m: float,
    rot_err_rad: float,
    iters: int,
    sigma_last: float,
) -> PoseIkReport:
    try:
        sigma_min = float(kin.singular_values(kin.jacobian(q)).min())
    except Exception:
        sigma_min = float(sigma_last)
    margin = float(ctrl.constraints.lim.position_margin)
    lo = kin.q_lower + margin
    hi = kin.q_upper - margin
    within = bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
    return PoseIkReport(
        pos_err_mm=pos_err_m * 1000.0,
        rot_err_deg=float(np.degrees(rot_err_rad)),
        sigma_min=sigma_min,
        iters=int(iters),
        within_limits=within,
    )


__all__ = [
    "PlannerGoalWeights",
    "PoseIkReport",
    "UnreachablePathError",
    "resolve_pose_ik_srs",
    "solve_pose_ik",
]

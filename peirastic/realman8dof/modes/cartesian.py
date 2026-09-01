"""Cartesian trajectory planning: pose-to-pose (joint PTP) and MOVES (polyline)."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import CompileContext
from rm75_control.control.joint_admittance_8dof.loop import Phase
from rm75_control.control.joint_admittance_8dof.pose_ik import (
    PlannerGoalWeights,
    UnreachablePathError,
    goal_score,
    resolve_pose_ik_srs,
)
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import d_from_q
from rm75_control.kinematics.srs_ik import (
    flange_tcp_from_kin,
    is_reachable,
    psi_from_q,
    shoulder_y_from_q_rail,
)
from peirastic.realman8dof.modes.joint import build_movej_phase, speed_frac
from peirastic.realman8dof.modes.track import build_track_cartesian_phase


def _as_pose(pose) -> np.ndarray:
    return np.asarray(pose, dtype=float).reshape(6)


def reachable_rails(
    kin,
    pose: np.ndarray,
    *,
    euler_order: str = "xyz",
    n_rail: int = 17,
) -> list[float]:
    """Rail joint values where the TCP pose lies in the SRS annulus."""

    pose_a = _as_pose(pose)
    R_flange, t_flange = flange_tcp_from_kin(kin)
    lo = float(kin.q_lower[0])
    hi = float(kin.q_upper[0])
    hits: list[float] = []
    for rail in np.linspace(lo, hi, int(max(n_rail, 3))):
        y = shoulder_y_from_q_rail(float(rail))
        if is_reachable(
            pose_a,
            y_rail=y,
            euler_order=euler_order,
            R_flange_tcp=R_flange,
            t_flange_tcp=t_flange,
        ):
            hits.append(float(rail))
    return hits


def assert_poses_reachable(kin, poses, *, euler_order: str = "xyz") -> None:
    arr = np.asarray(poses, dtype=float).reshape(-1, 6)
    bad: list[int] = []
    for i, pose in enumerate(arr):
        if not reachable_rails(kin, pose, euler_order=euler_order):
            bad.append(i)
    if bad:
        raise ValueError(f"unreachable Cartesian pose(s) at index {bad}")


def _finite_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def ns_attractor(ctx: CompileContext) -> tuple[float, float]:
    """Live ``(ψ*, d*)`` if finite, else yaml ``psi_attr`` / ``d_attr``."""

    psi_home = None
    d_star = None
    pr = getattr(ctx.inner, "posture_retarget", None)
    if pr is not None:
        psi_home = _finite_or_none(getattr(pr, "psi_star_rad", None))
        d_star = _finite_or_none(getattr(pr, "d_star_m", None))
        cfg = getattr(pr, "cfg", None)
        if psi_home is None and cfg is not None:
            psi_home = _finite_or_none(getattr(cfg, "psi_attr_rad", None))
        if d_star is None and cfg is not None:
            d_star = _finite_or_none(getattr(cfg, "d_attr_m", None))
    inner_cfg = getattr(ctx.inner, "cfg", None)
    pr_cfg = getattr(inner_cfg, "psi_retarget", None) if inner_cfg is not None else None
    if psi_home is None and pr_cfg is not None:
        psi_home = _finite_or_none(getattr(pr_cfg, "psi_attr_rad", None))
    if d_star is None and pr_cfg is not None:
        d_star = _finite_or_none(getattr(pr_cfg, "d_attr_m", None))
    if psi_home is None:
        psi_home = float(np.deg2rad(68.0))
    if d_star is None:
        d_star = -0.185
    return float(psi_home), float(d_star)


def _ns_pose_cost(
    kin,
    q: np.ndarray,
    *,
    psi_home: float,
    d_star: float,
    weights: PlannerGoalWeights,
) -> float:
    """Lower is better. Matches online NS: ``goal_score(ψ)`` plus ``|d − d*|``."""

    q = np.asarray(q, dtype=float).reshape(-1)
    q_arm = q[1:]
    psi = float(psi_from_q(q_arm))
    sigma_min = float(kin.singular_values(kin.jacobian(q)).min())
    score = float(goal_score(q_arm, q, psi, float(psi_home), sigma_min, kin, weights))
    d_err = abs(float(d_from_q(kin, q)) - float(d_star))
    return -score + 12.0 * d_err


def resolve_pose_q(
    ctx: CompileContext,
    pose,
    *,
    q_seed: np.ndarray | None = None,
    rail_m: float | None = None,
    require_path: bool = True,
) -> np.ndarray:
    """8-DOF pose IK: sweep rail unless ``rail_m`` locks it.

    Rank candidates by the same ``(ψ, d*)`` secondary used online. Hold may
    still catch remaining d*; this only stops planning from preferring
    ``min |Δrail|`` against that attractor.
    """

    pose_a = _as_pose(pose)
    q0 = np.asarray(ctx.inner.q_cmd if q_seed is None else q_seed, dtype=float).reshape(-1)
    if q0.size != 8:
        raise ValueError(f"q_seed must be 8-vec, got {q0.size}")
    psi_home, d_star = ns_attractor(ctx)
    weights = PlannerGoalWeights()
    if rail_m is not None:
        rails = [float(rail_m)]
    else:
        rails = list(reachable_rails(ctx.kin, pose_a, euler_order=ctx.euler_order))
        pref = float(np.clip(float(pose_a[1]) - d_star, float(ctx.kin.q_lower[0]), float(ctx.kin.q_upper[0])))
        R_flange, t_flange = flange_tcp_from_kin(ctx.kin)
        if is_reachable(
            pose_a,
            y_rail=shoulder_y_from_q_rail(pref),
            euler_order=ctx.euler_order,
            R_flange_tcp=R_flange,
            t_flange_tcp=t_flange,
        ):
            rails.append(pref)
        if not rails:
            raise ValueError("pose unreachable: no rail puts the wrist in the SRS annulus")
    best: np.ndarray | None = None
    best_cost = float("inf")
    for rail in rails:
        try:
            q, ok, _rep = resolve_pose_ik_srs(
                ctx.kin,
                q0,
                pose_a,
                y_rail_target=float(rail),
                psi_home_rad=psi_home,
                require_path=bool(require_path),
                euler_order=ctx.euler_order,
                planner_weights=weights,
            )
        except UnreachablePathError:
            continue
        if not ok or q is None:
            continue
        q = np.asarray(q, dtype=float).reshape(-1)
        cost = _ns_pose_cost(
            ctx.kin, q, psi_home=psi_home, d_star=d_star, weights=weights
        )
        if cost < best_cost:
            best = q
            best_cost = cost
    if best is None:
        raise ValueError("pose unreachable: SRS IK found no path-valid solution")
    return best


def _v_frac(payload: dict) -> float:
    return speed_frac(payload.get("v"))


def build_cartesian_ptp_phase(
    ctx: CompileContext,
    payload: dict,
    *,
    dt: float = 0.005,
) -> Phase:
    """Pose-to-pose: IK the goal, then the same joint-space smooth PTP as MOVEJ.

    TCP is not forced onto a straight line. Joint interpolation is the usual
    industrial move-to-pose; MOVES is the polyline if a Cartesian path is needed.
    """

    del dt
    pose = payload.get("pose")
    if pose is None:
        poses = payload.get("poses")
        if poses is None:
            raise ValueError("cartesian plan needs pose")
        pose = np.asarray(poses, dtype=float).reshape(-1, 6)[0]
    pose_a = _as_pose(pose)
    q0 = payload.get("q_start")
    q_start = None if q0 is None else np.asarray(q0, dtype=float)
    q_seed = np.asarray(ctx.inner.q_cmd if q_start is None else q_start, dtype=float)
    if payload.get("q_target") is not None:
        qt = np.asarray(payload["q_target"], dtype=float).reshape(-1)
    else:
        if not reachable_rails(ctx.kin, pose_a, euler_order=ctx.euler_order):
            raise ValueError("pose unreachable: no rail puts the wrist in the SRS annulus")
        qt = resolve_pose_q(
            ctx,
            pose_a,
            q_seed=q_seed,
            rail_m=payload.get("rail_m"),
            require_path=False,
        )
    dur = payload.get("duration_s")
    return build_movej_phase(
        ctx,
        qt,
        q_start=q_start,
        duration_s=None if dur is None else float(dur),
        v=_v_frac(payload),
        label=str(payload.get("label", "cartesian")),
    )


def build_moves_phase(
    ctx: CompileContext,
    payload: dict,
    *,
    dt: float = 0.005,
) -> Phase:
    del dt
    poses = payload.get("poses")
    if poses is None and payload.get("pose") is not None:
        poses = [payload["pose"]]
    if poses is None:
        raise ValueError("MOVES needs poses")
    arr = np.asarray(poses, dtype=float).reshape(-1, 6)
    assert_poses_reachable(ctx.kin, arr, euler_order=ctx.euler_order)
    speed = payload.get("speed_m_s")
    if speed is None:
        v = _v_frac(payload)
        max_lin = payload.get("max_lin_vel_m_s")
        speed = float(max_lin) if max_lin is not None else 0.4 * v
    ref = WorldPolylineReference(
        arr[:, :3],
        rpy=arr[:, 3:6],
        speed_m_s=float(speed),
        soft_start=bool(payload.get("soft_start", True)),
        ramp_s=float(payload.get("ramp_s", 0.4)),
        euler_order=ctx.euler_order,
    )
    duration = payload.get("duration_s")
    if duration is None:
        duration = ref.duration_s()
    return build_track_cartesian_phase(
        ctx,
        ref,
        duration_s=float(duration),
        label=str(payload.get("label", "moves")),
        max_lin_vel_m_s=payload.get("max_lin_vel_m_s"),
        move_kp=payload.get("move_kp"),
    )

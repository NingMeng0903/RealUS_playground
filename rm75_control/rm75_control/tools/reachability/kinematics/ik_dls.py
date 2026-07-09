"""Fast Damped Least-Squares (SR-inverse) inverse kinematics for the 7-DOF arm.

We deliberately do NOT reuse ``pose_ik.solve_pose_ik`` here:

* The QP-based controller IK is ~5-15 ms per iteration × up to 500 iterations –
  too slow when we need millions of "does this pose have any IK?" checks during
  capability-map refinement.
* The QP couples in CBF collision avoidance, nullspace tasks, rail relief etc.
  Those are online-safety features; for reachability we only want a *bool*
  answer plus optionally the manipulability at the solution.

The algorithm below is the textbook damped Newton update using the
LOCAL_WORLD_ALIGNED spatial-velocity Jacobian::

    e = [p_target - p, alpha * a]           # a = axis-angle of R_target R^T
    JJt = J J^T + λ² I                       # SR damping
    dq  = J^T @ solve(JJt, e)
    q  <- clip(q + dq, q_lower, q_upper)

Convergence checks (position ≤ ``tol_pos_m``, rotation ≤ ``tol_rot_rad``) are
applied after every clip so an early success bails out with a real limit-safe
configuration. Single call target ≤ 200 μs at 40 iterations on the RM75.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from rm75_control.tools.reachability.kinematics.model_locked_rail import LockedRailModel


@dataclass
class IkDlsReport:
    ok: bool
    iters: int
    pos_err_m: float
    rot_err_rad: float
    sigma_min: float


@dataclass
class IkDlsResult:
    q: np.ndarray
    report: IkDlsReport


def _axis_angle_from_matrix(R: np.ndarray) -> np.ndarray:
    """Return world-frame rotation vector (axis * angle) for a 3x3 rotation.

    Uses the pinocchio ``log3`` implementation for numerical robustness near
    identity and π.
    """
    return pin.log3(R)


def _pose_error(M_target: pin.SE3, M_current: pin.SE3) -> np.ndarray:
    """6-vector ``[dp_world, dw_world]``.

    ``dp = p_target - p_current`` and ``dw = log(R_target R_current^T)``.
    Aligned with the LOCAL_WORLD_ALIGNED Jacobian convention.
    """
    dp = M_target.translation - M_current.translation
    dR = M_target.rotation @ M_current.rotation.T
    dw = _axis_angle_from_matrix(dR)
    return np.concatenate([dp, dw])


def ik_dls(
    lm: LockedRailModel,
    pose_target: pin.SE3,
    q_seed: np.ndarray,
    *,
    max_iter: int = 40,
    tol_pos_m: float = 5e-4,
    tol_rot_rad: float = 5e-3,
    lam: float = 0.08,
    alpha_rot: float = 1.0,
    step_gain: float = 1.0,
    margin_rad: float = 1e-3,
) -> IkDlsResult:
    """Solve 7-DOF IK to ``pose_target`` starting from ``q_seed``.

    Parameters
    ----------
    lm
        Locked-rail model (see :func:`build_locked_rail_model`). ``lm.data`` is
        mutated in place.
    pose_target
        SE3 in the same frame as ``lm.model``'s TCP (i.e. ``rail_base``).
    q_seed
        Initial (7,) configuration in rad; will be clipped into limits first.
    max_iter, tol_pos_m, tol_rot_rad
        Convergence controls; the loop bails as soon as both tolerances hit.
    lam
        SR-inverse damping ``λ``. 0.05–0.10 works well on the RM75.
    alpha_rot
        Scalar weight on the rotational part of the pose error. ``1.0`` matches
        the unit convention (1 rad ~ 1 m of angular equivalent) and is
        appropriate for tolerances of the same numerical size.
    step_gain
        Scale on the DLS step; keep 1.0 for standard behaviour.
    margin_rad
        Extra clearance from ``q_lower``/``q_upper`` after clipping.
    """
    if q_seed.shape != (lm.model.nq,):
        raise ValueError(f"q_seed must be shape ({lm.model.nq},), got {q_seed.shape}")

    lo = lm.q_lower + margin_rad
    hi = lm.q_upper - margin_rad
    q = np.clip(np.asarray(q_seed, dtype=np.float64).copy(), lo, hi)

    model, data, fid = lm.model, lm.data, lm.tcp_id
    sigma_min = float("inf")
    err_pos = float("inf")
    err_rot = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, fid)
        err = _pose_error(pose_target, data.oMf[fid])
        if alpha_rot != 1.0:
            err[3:6] *= alpha_rot
        err_pos = float(np.linalg.norm(err[:3]))
        err_rot = float(np.linalg.norm(err[3:6]) / max(alpha_rot, 1e-12))
        if err_pos < tol_pos_m and err_rot < tol_rot_rad:
            break

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J = pin.getFrameJacobian(
            model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )  # (6, 7)
        if alpha_rot != 1.0:
            J = J.copy()
            J[3:6, :] *= alpha_rot

        JJt = J @ J.T + (lam * lam) * np.eye(6)
        try:
            y = np.linalg.solve(JJt, err)
        except np.linalg.LinAlgError:
            break
        dq = step_gain * (J.T @ y)
        q_new = np.clip(q + dq, lo, hi)
        # if clipping killed all progress, damp further and retry once
        if np.max(np.abs(q_new - q)) < 1e-9:
            lam *= 1.5
            continue
        q = q_new
        # cheap conditioning estimate every 4 iters
        if it % 4 == 0:
            try:
                sigma_min = float(np.linalg.svd(J, compute_uv=False).min())
            except np.linalg.LinAlgError:
                sigma_min = 0.0

    # final FK for reporting
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, fid)
    err_final = _pose_error(pose_target, data.oMf[fid])
    err_pos = float(np.linalg.norm(err_final[:3]))
    err_rot = float(np.linalg.norm(err_final[3:6]))
    ok = (err_pos < tol_pos_m) and (err_rot < tol_rot_rad)
    if not np.isfinite(sigma_min):
        try:
            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(
                model, data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            sigma_min = float(np.linalg.svd(J, compute_uv=False).min())
        except Exception:
            sigma_min = 0.0

    report = IkDlsReport(
        ok=bool(ok), iters=int(it), pos_err_m=err_pos, rot_err_rad=err_rot, sigma_min=sigma_min
    )
    return IkDlsResult(q=q, report=report)


def ik_dls_multiseed(
    lm: LockedRailModel,
    pose_target: pin.SE3,
    seeds: np.ndarray,
    *,
    keep_best: bool = False,
    **kwargs,
) -> IkDlsResult:
    """Try ``ik_dls`` from every seed row of ``seeds`` (shape (S, 7)).

    Default returns the first successful solve (fast reachability test); with
    ``keep_best=True`` scores all solves by manipulability and returns the max.
    """
    if seeds.ndim != 2 or seeds.shape[1] != lm.model.nq:
        raise ValueError(f"seeds must be (S, {lm.model.nq}), got {seeds.shape}")
    best: IkDlsResult | None = None
    best_mu = -1.0
    for row in seeds:
        res = ik_dls(lm, pose_target, row, **kwargs)
        if res.report.ok:
            if not keep_best:
                return res
            mu = _manipulability(lm)
            if mu > best_mu:
                best_mu = mu
                best = res
    if best is not None:
        return best
    # nothing converged: return the last attempt as-is (ok=False)
    return res  # noqa: F821  (res is defined by the loop, seeds is non-empty)


def _manipulability(lm: LockedRailModel) -> float:
    """Yoshikawa μ = sqrt(det(J J^T)) at the current ``lm.data`` state."""
    J = pin.getFrameJacobian(
        lm.model, lm.data, lm.tcp_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    JJt = J @ J.T
    d = float(np.linalg.det(JJt))
    return float(np.sqrt(max(d, 0.0)))

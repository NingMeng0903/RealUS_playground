"""One-shot pose inverse kinematics using WBC slack-QP iterations.

Planning-only helper: resolves ``q_target`` for a desired TCP pose without any
vendor ``rm_algo_inverse_kinematics`` call - this is the ONLY IK path allowed
for large point-to-point moves (see MD/debug.md architecture constraint: no
black-box vendor IK, ever).

The WBC QP core consumes a given task twist verbatim (position feedback is
owned exactly once, by the caller) - this iterative planner is that caller and
builds an explicit P-controller twist from the pose error each iteration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance.ik_types import saturate_error
from rm75_control.control.joint_admittance.model import RobotKinematics, pose_error
from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance.utils.safety import SafetyLimits


@dataclass
class PoseIkReport:
    """Convergence diagnostics from ``solve_pose_ik``.

    Sciavicco & Siciliano (1988) §7: a numerical IK acceptance decision needs
    (a) task-space residual, (b) an ill-conditioning indicator (σ_min at the
    solution) and (c) joint-limit feasibility.  Callers previously had only
    the boolean ``ok`` flag and silently drove toward a wrong pose_d on
    graceful degradation - the report closes that loop.
    """

    pos_err_mm: float
    rot_err_deg: float
    sigma_min: float
    iters: int
    within_limits: bool


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
    trace: list[dict] | None = None,
) -> tuple[np.ndarray, bool, PoseIkReport]:
    """Iterative WBC IK: ``q_seed`` -> ``q`` with ``fk(q) ≈ pose_target``.

    Each iteration feeds ``v_cmd = k_gain * saturate(pose_error)`` to the QP
    core.  The error saturation bounds the twist per iteration so the "virtual
    dt" integration stays well-conditioned even far from the target.

    ``nullspace_cfg`` (typically ``JointIkConfig.nullspace`` straight from the
    YAML) drives the SAME ``JointCenteringTask`` the online loop uses as the
    QP's secondary task, so the resolved ``q_target`` lands biased toward the
    comfortable posture (``q_nominal_deg``) instead of the QP's default
    minimum-joint-motion solution.

    ``trace``, if given a list, is appended one dict per iteration with the
    convergence diagnostics that ``IkStepResult`` already computes but which
    the caller normally discards (``slack_norm``, ``n_cbf_active``,
    ``sigma_min``) plus the pose error and commanded twist norm - debugging
    aid for "IK looks stuck / lands on a weird branch" investigations, zero
    cost/behavior change when ``trace`` is ``None``.
    """
    cfg = qp_cfg or QpConfig()
    limits = SafetyLimits.from_kinematics(kin, v_scale=0.9, a_max=50.0)
    ctrl = QpIkController(kin, limits, cfg)
    task = JointCenteringTask.from_kinematics(kin, nullspace_cfg) if nullspace_cfg is not None else None
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
    # Recompute σ_min at the final q so a converged solve (which exits the loop
    # before ctrl.step ran that iter) still reports a real conditioning value.
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

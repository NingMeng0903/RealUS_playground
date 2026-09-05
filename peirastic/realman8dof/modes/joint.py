"""Joint-space modes: snap-to-angle and planned MoveJ. Both use the inner PTP bypass."""

from __future__ import annotations

import numpy as np

from rm75_control.control.joint_admittance_8dof.api import (
    CompileContext,
    attach_joint_move_rail,
    compile_phase,
    compute_move_plan,
    phase_cartesian_goto,
)
from rm75_control.control.joint_admittance_8dof.loop import JointTrackOuterLoop, Phase
from rm75_control.control.joint_admittance_8dof.model import auto_move_duration_s
from rm75_control.control.joint_admittance_8dof.reference import JointSmoothMoveReference

DEFAULT_V = 0.4


def speed_frac(v) -> float:
    """API ``v`` in (0, 1]; missing or non-positive falls back to ``DEFAULT_V``."""

    if v is None:
        return DEFAULT_V
    v = float(v)
    if v <= 0.0:
        return DEFAULT_V
    return min(v, 1.0)


def _duration(
    kin,
    q0,
    q_target,
    duration_s: float | None,
    *,
    v_scale: float,
    v: float,
) -> float:
    if duration_s is not None and float(duration_s) > 1e-6:
        return float(duration_s)
    pose = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
    v = speed_frac(v)
    T, _meta = auto_move_duration_s(
        kin,
        q0,
        q_target,
        pose,
        v_scale=float(v_scale) * v,
        v_max_rad_s=kin.v_max,
        peak_joint_v_frac=0.80,
        max_lin_vel_m_s=0.4 * v,
        duration_min_s=0.4 / v,
        duration_max_s=max(20.0, 20.0 / v),
    )
    return float(T)


def build_movej_phase(
    ctx: CompileContext,
    q_target: np.ndarray,
    *,
    q_start: np.ndarray | None = None,
    duration_s: float | None = None,
    v: float | None = None,
    label: str = "movej",
    secondary: str | None = None,
) -> Phase:
    if q_start is not None:
        q0 = np.asarray(q_start, dtype=float)
    else:
        # Native auto_commit=False can leave q_cmd at enable/mid while the
        # arm is still at the last encoder sample.  Duration must use that
        # sample or MOVEJ is planned as if it were already at the target.
        meas = getattr(ctx.inner, "_last_q_meas", None)
        q0 = np.asarray(
            ctx.inner.q_cmd if meas is None else meas, dtype=float
        )
    qt = np.asarray(q_target, dtype=float).reshape(-1)
    v_frac = speed_frac(v)
    T = _duration(ctx.kin, q0, qt, duration_s, v_scale=float(ctx.v_scale), v=v_frac)
    pose_t = np.asarray(ctx.kin.fk_pose(qt), dtype=float).reshape(6)
    plan = compute_move_plan(
        ctx.kin,
        q0,
        qt,
        pose_t,
        v_scale=float(ctx.v_scale) * v_frac,
        duration_s=T,
        move_mode="joint",
        max_lin_vel_m_s=0.4 * v_frac,
        duration_min_s=0.4 / v_frac,
        duration_max_s=max(20.0, 20.0 / v_frac),
    )
    move_ref = JointSmoothMoveReference(ctx.kin, q0, qt, plan.duration_s)
    spec = phase_cartesian_goto(
        move_ref,
        label=label,
        pose_target=pose_t,
        q_target_rad=qt,
        move_mode="joint",
        gov_joint_max_deg=plan.gov_joint_max_deg,
        secondary_preset=str(secondary) if secondary else "move",
    )
    compiled = compile_phase(spec, ctx)
    attach_joint_move_rail(compiled.phase, ctx.inner, move_ref=move_ref)
    return compiled.phase


def build_goto_joints_phase(
    ctx: CompileContext,
    q_target: np.ndarray,
    *,
    q_start: np.ndarray | None = None,
    duration_s: float | None = None,
    v: float | None = None,
) -> Phase:
    """Direct joint-angle command. Inner ``set_direct_joint_ptp(True)``.

    Still a planned PTP (limits / jerk stay on). Differs from MoveJ only in
    label and the default auto duration; both skip the Cartesian QP.
    """
    return build_movej_phase(
        ctx,
        q_target,
        q_start=q_start,
        duration_s=duration_s,
        v=v,
        label="goto_joints",
    )


# Aliases so tests can talk about outers even though compile yields Phase.
GotoJointsOuter = JointTrackOuterLoop
MoveJOuter = JointTrackOuterLoop

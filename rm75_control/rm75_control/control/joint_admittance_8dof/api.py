"""Phase factories + compile: JointPhaseSpec → runtime Phase for the 8-DOF loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal

import numpy as np

from rm75_control.control.admittance_common.controller import AdmittanceController
from rm75_control.control.admittance_common.reference import MotionReferenceSource
from rm75_control.control.joint_admittance_8dof.loop import (
    AdmittanceOuterLoop,
    CartesianTrackConfig,
    CartesianTrackOuterLoop,
    JointIkController,
    JointTrackConfig,
    JointTrackOuterLoop,
    Phase,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import _wrap_pi
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import LockedStyle, RailMode
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    auto_move_duration_s,
    pose_distance,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.reference import (
    HoldReference,
    JointSmoothMoveReference,
    RailSmoothMoveReference,
    SrsSmoothMoveReference,
    auto_rail_move_duration_s,
    srs_move_duration_s,
)


class TaskMode(str, Enum):
    JOINT_RESET = "joint_reset"
    CARTESIAN_GOTO = "cartesian_goto"
    CARTESIAN_TRACK = "cartesian_track"
    HYBRID_TRACK = "hybrid_track"
    # LOCKED_MOVE == plan drives the rail while the top-level mode is LOCKED;
    # the substyle (RAIL_ONLY vs TCP_FIXED) is carried on JointPhaseSpec.
    LOCKED_MOVE = "locked_move"


@dataclass
class ArmAngleSpec:
    """Arm-angle nullspace target applied on phase entry (scan/handoff)."""

    psi_rad: float | None = None


@dataclass
class SecondaryPolicy:
    """Nullspace / secondary-task preset: off | move | track | hold."""

    preset: Literal["off", "move", "track", "hold"] = "track"
    arm_angle: ArmAngleSpec | None = None
    qdot_ff: Literal["off", "plan", "plan_joint"] = "off"

    def _set_arm_angle_reference(
        self,
        inner: JointIkController,
        psi_rad: float | None,
    ) -> None:
        if psi_rad is None or inner.arm_task is None:
            return
        psi_live = float(inner.arm_task.arm_angle(inner.q_cmd))
        psi_set = float(psi_live + _wrap_pi(float(psi_rad) - psi_live))
        inner.arm_task.set_reference(psi_set)

    def apply(self, inner: JointIkController, *, psi_rad: float | None = None) -> None:
        psi = psi_rad
        if self.arm_angle is not None and self.arm_angle.psi_rad is not None:
            psi = self.arm_angle.psi_rad

        if self.preset == "move":
            # Plan owns posture; suppress secondary fights with the planner.
            inner.set_coupled()
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_mode("pose_attract")
            inner.set_rail_extension_active(True)
        elif self.preset == "track":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            inner.set_centering_suppressed(False)
            inner.set_arm_task_suppressed(False)
            # Restore yaml rail mode (live cfg.rail.mode is mutated by locks).
            if inner.configured_rail_mode == RailMode.COUPLED:
                inner.set_coupled()
                inner.set_rail_extension_mode("reach")
                inner.capture_rail_extension_ref()
                inner.set_rail_extension_active(True)
            else:
                inner.set_locked(
                    LockedStyle.HOLD, q_ref_m=float(inner.q_cmd[0])
                )
                inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "hold":
            inner.set_plan_drives_rail(False)
            inner.set_manipulability_active(False)
            # Suppress centering (fights teach/force hold); keep arm_angle branch.
            inner.set_centering_suppressed(True)
            inner.set_arm_task_suppressed(False)
            # Pin rail at current q_cmd[0] (never yaml q_ref_m=0.0).
            inner.set_locked(LockedStyle.HOLD, q_ref_m=float(inner.q_cmd[0]))
            inner.set_rail_extension_active(False)
            if psi is not None and inner.arm_task is not None:
                self._set_arm_angle_reference(inner, psi)
        elif self.preset == "off":
            inner.set_arm_task_suppressed(True)
            inner.set_centering_suppressed(True)
            inner.set_manipulability_active(False)
            inner.set_rail_extension_active(False)

    def make_qdot_ff_provider(
        self,
        inner: JointIkController,
        move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None,
    ) -> Callable[[float], np.ndarray] | None:
        if self.qdot_ff == "off" or move_ref is None:
            return None
        if self.qdot_ff == "plan":
            return lambda t: move_ref.sample_q(t)[1]
        if self.qdot_ff == "plan_joint":

            def _joint_ff(t: float) -> np.ndarray:
                q_plan, dq_plan = move_ref.sample_q(t)
                return dq_plan + 1.0 * wrap_joint_delta(inner.q_cmd, q_plan)

            return _joint_ff
        return None


@dataclass
class GovernorSpec:
    err_ok_mm: float = 5.0
    err_max_mm: float = 25.0
    joint_err_ok_deg: float = 3.0
    joint_err_max_deg: float = 0.0
    tau_s: float = 0.2
    freeze_below: float = 0.02
    release_above: float = 0.10


@dataclass
class JointPhaseSpec:
    mode: TaskMode
    label: str = ""
    secondary: SecondaryPolicy = field(default_factory=SecondaryPolicy)
    governor: GovernorSpec = field(default_factory=GovernorSpec)
    duration_s: float | None = None
    max_duration_s: float | None = None
    wait_until: Callable[..., bool] | None = None
    require_arrival: bool = False
    force_observer: Any = None
    scale_qdot_ff_with_governor: bool = True
    # Move / goto
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None = None
    pose_target: np.ndarray | None = None
    q_target_rad: np.ndarray | None = None
    move_kp: float = 2.0
    move_mode: Literal["joint", "cartesian"] = "cartesian"
    max_lin_vel_m_s: float = 0.4
    sigma_ref: float = 0.08
    # Track / hybrid
    reference: MotionReferenceSource | None = None
    controller: AdmittanceController | None = None
    desired_force: np.ndarray | None = None
    # Locked-move (LOCKED + RAIL_ONLY / TCP_FIXED): external plan drives rail
    rail_ref: RailSmoothMoveReference | None = None
    locked_style: LockedStyle = LockedStyle.RAIL_ONLY
    q_rail_target_m: float | None = None


@dataclass
class CompileContext:
    kin: RobotKinematics
    inner: JointIkController
    euler_order: str = "xyz"
    control_frame: str = "tool"
    v_scale: float = 0.5


@dataclass
class CompiledPhase:
    phase: Phase
    label: str
    outer: Any = None
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference | None = None
    rail_ref: RailSmoothMoveReference | None = None
    reference: MotionReferenceSource | None = None


def make_srs_move_reference(
    kin: RobotKinematics,
    q_start_rad: np.ndarray,
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    duration_s: float,
    *,
    euler_order: str = "xyz",
) -> SrsSmoothMoveReference:
    """Build a branch-locked SRS move reference (plan pose = FK(q_target))."""
    from rm75_control.kinematics.srs_ik import d_wt_from_kin, psi_from_q

    q_start = np.asarray(q_start_rad, dtype=float)
    q_target = np.asarray(q_target_rad, dtype=float)
    # Plan uses live FK(q_target), not a cached pose_target.
    pose_from_q = np.asarray(kin.fk_pose(q_target), dtype=float).reshape(6)
    v_max = kin.v_max * 0.5  # match inner v_scale default
    T_rate = srs_move_duration_s(q_start, q_target, max_qdot_rad_s=v_max)
    T = max(float(duration_s), T_rate)
    d_wt = float(d_wt_from_kin(kin))
    return SrsSmoothMoveReference(
        kin,
        q_start,
        pose_from_q,
        y_rail_target_m=float(q_target[0]),
        psi_target_rad=float(psi_from_q(q_target[1:])),
        duration_s=T,
        euler_order=euler_order,
        d_wt=d_wt,
    )


def attach_joint_move_rail(
    phase: Phase,
    inner: JointIkController,
) -> None:
    """Pin rail to the joint plan and enable direct joint PTP (no Cartesian QP)."""
    prev_on_enter = phase.on_enter
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        inner.set_rail_extension_active(False)
        inner.set_plan_drives_rail(True)
        inner.set_direct_joint_ptp(True)

    def _exit() -> None:
        inner.set_direct_joint_ptp(False)
        inner.set_plan_drives_rail(False)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_exit = _exit


def attach_srs_move_tracking(
    phase: Phase,
    inner: JointIkController,
    move_ref: SrsSmoothMoveReference,
    q_target_rad: np.ndarray,
) -> None:
    """Wire ψ_ref(t) + centering for SRS move; pin rail to y_rail(t) plan."""
    q_target = np.asarray(q_target_rad, dtype=float)
    prev_on_enter = phase.on_enter
    prev_on_tick = phase.on_tick
    prev_on_exit = phase.on_exit

    def _enter() -> None:
        if prev_on_enter is not None:
            prev_on_enter()
        if not inner._centering_suppressed:
            inner.centering_task.set_q_target(q_target)
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.psi_start)
        inner.set_plan_drives_rail(True)

    def _tick(t_ref: float, step, q_meas: np.ndarray) -> None:
        if inner.arm_task is not None and not inner._arm_task_suppressed:
            inner.arm_task.set_reference(move_ref.sample_psi(t_ref))
        if prev_on_tick is not None:
            prev_on_tick(t_ref, step, q_meas)

    def _exit() -> None:
        inner.set_plan_drives_rail(False)
        # Restore yaml posture attractor (do not keep D-point as scan target).
        inner.centering_task.set_q_target(None)
        if prev_on_exit is not None:
            prev_on_exit()

    phase.on_enter = _enter
    phase.on_tick = _tick
    phase.on_exit = _exit


@dataclass
class MovePlan:
    duration_s: float
    move_mode: Literal["joint", "cartesian"]
    gov_joint_max_deg: float
    meta: dict


def compute_move_plan(
    kin: RobotKinematics,
    q0_rad: np.ndarray,
    q_target_rad: np.ndarray,
    pose_target: np.ndarray,
    *,
    v_scale: float,
    duration_s: float | None = None,
    move_mode: Literal["joint", "cartesian"] = "joint",
    peak_joint_v_frac: float = 0.80,
    max_lin_vel_m_s: float = 0.4,
    duration_min_s: float = 2.5,
    duration_max_s: float = 20.0,
    approach_dz_m: float | None = None,
    sigma_ref: float = 0.08,
    euler_order: str = "xyz",
) -> MovePlan:
    """Duration and joint governor cap for an explicit PTP mode (no auto-switch)."""
    auto_duration, meta = auto_move_duration_s(
        kin,
        q0_rad,
        q_target_rad,
        pose_target,
        v_scale=v_scale,
        v_max_rad_s=kin.v_max,
        peak_joint_v_frac=peak_joint_v_frac,
        max_lin_vel_m_s=max_lin_vel_m_s,
        duration_min_s=duration_min_s,
        duration_max_s=duration_max_s,
        approach_dz_m=approach_dz_m,
        sigma_ref=sigma_ref,
        euler_order=euler_order,
    )
    max_dq_deg = float(meta["max_dq_deg"])
    gov_joint_max_deg = float(np.clip(1.15 * max_dq_deg, 25.0, 90.0))
    duration = float(duration_s) if duration_s is not None else auto_duration
    meta["user_override"] = duration_s is not None
    return MovePlan(
        duration_s=duration,
        move_mode=move_mode,
        gov_joint_max_deg=gov_joint_max_deg,
        meta=meta,
    )


def make_move_arrived(
    pose_target: np.ndarray,
    q_target_rad: np.ndarray,
    *,
    tol_mm: float = 3.0,
    tol_deg: float = 1.5,
    joint_tol_deg: float = 3.0,
    rail_tol_mm: float = 5.0,
    joint_only: bool = False,
    require_joints: bool = True,
    euler_order: str = "xyz",
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Arrival gate for move→D (pose and/or joint tolerances)."""

    def _joints_ok(q_meas: np.ndarray) -> bool:
        qa = np.asarray(q_meas, dtype=float).reshape(-1)
        qt = np.asarray(q_target_rad, dtype=float).reshape(-1)
        n = int(min(qa.size, qt.size))
        if n < 1:
            return False
        if abs(float(qa[0]) - float(qt[0])) * 1000.0 > float(rail_tol_mm):
            return False
        if n > 1:
            from rm75_control.control.joint_admittance_8dof.model import wrap_joint_delta

            d = wrap_joint_delta(qa[:n], qt[:n])
            arm_err = float(np.rad2deg(np.max(np.abs(d[1:]))))
            if arm_err > float(joint_tol_deg):
                return False
        return True

    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        if joint_only:
            return _joints_ok(q_meas)
        d_mm, d_deg = pose_distance(pose_meas, pose_target, euler_order)
        if d_mm > tol_mm or d_deg > tol_deg:
            return False
        if not require_joints:
            return True
        return _joints_ok(q_meas)

    return _fn


def make_rail_arrived(
    q_target_m: float,
    *,
    tol_mm: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    def _fn(pose_meas: np.ndarray, q_meas: np.ndarray) -> bool:
        del pose_meas
        return abs(float(q_meas[0]) - float(q_target_m)) * 1000.0 <= tol_mm

    return _fn


def phase_rail_reposition(
    q_target_m: float,
    q_start_rad: np.ndarray,
    kin: RobotKinematics,
    *,
    label: str = "rail_reposition",
    style: LockedStyle | str = LockedStyle.RAIL_ONLY,
    duration_s: float | None = None,
    max_duration_s: float | None = None,
    require_arrival: bool = True,
    force_observer: Any = None,
    v_max_m_s: float | None = None,
) -> JointPhaseSpec:
    """Smoothstep rail_y to ``q_target_m`` (RAIL_ONLY or TCP_FIXED)."""
    if isinstance(style, str):
        style = LockedStyle(style)
    if style not in (LockedStyle.RAIL_ONLY, LockedStyle.TCP_FIXED):
        raise ValueError(
            f"phase_rail_reposition style must be RAIL_ONLY or TCP_FIXED, got {style}"
        )
    q_start = np.asarray(q_start_rad, dtype=float)
    rail_v = float(v_max_m_s if v_max_m_s is not None else kin.v_max[0])
    if duration_s is None:
        duration_s = auto_rail_move_duration_s(
            float(q_start[0]),
            float(q_target_m),
            v_max_m_s=rail_v,
            peak_v_frac=1.0,
        )
    rail_ref = RailSmoothMoveReference(q_start, float(q_target_m), float(duration_s))
    # Suppress secondary posture tasks during rail reposition.
    sec = SecondaryPolicy(preset="off", qdot_ff="plan")
    return JointPhaseSpec(
        mode=TaskMode.LOCKED_MOVE,
        label=label,
        rail_ref=rail_ref,
        q_rail_target_m=float(q_target_m),
        locked_style=style,
        duration_s=float(duration_s),
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=GovernorSpec(err_max_mm=0.0),
        scale_qdot_ff_with_governor=False,
        wait_until=make_rail_arrived(q_target_m),
        move_kp=2.0 if style == LockedStyle.TCP_FIXED else 0.0,
        max_lin_vel_m_s=0.10 if style == LockedStyle.TCP_FIXED else 0.4,
    )


def phase_hold_at_pose(
    duration_s: float,
    *,
    label: str = "hold",
    move_kp: float = 1.0,
    force_observer: Any = None,
) -> JointPhaseSpec:
    """Hold current TCP pose for ``duration_s`` (rail locked via preset hold)."""
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_TRACK,
        label=label,
        reference=HoldReference(),
        duration_s=float(duration_s),
        move_kp=float(move_kp),
        force_observer=force_observer,
        secondary=SecondaryPolicy(preset="hold", qdot_ff="off"),
        governor=GovernorSpec(err_ok_mm=15.0, err_max_mm=80.0),
    )


def phase_cartesian_goto(
    move_ref: JointSmoothMoveReference | SrsSmoothMoveReference,
    *,
    label: str = "cartesian_goto",
    pose_target: np.ndarray | None = None,
    q_target_rad: np.ndarray | None = None,
    move_kp: float = 2.0,
    move_mode: Literal["joint", "cartesian"] = "cartesian",
    max_lin_vel_m_s: float = 0.4,
    max_duration_s: float | None = None,
    gov_joint_max_deg: float = 25.0,
    require_arrival: bool = True,
    force_observer: Any = None,
) -> JointPhaseSpec:
    sec = SecondaryPolicy(
        preset="move",
        qdot_ff="plan_joint",
    )
    gov = (
        GovernorSpec(
            err_max_mm=0.0,
            joint_err_ok_deg=12.0,
            joint_err_max_deg=max(float(gov_joint_max_deg), 60.0),
        )
        if move_mode == "joint"
        else GovernorSpec(
            err_ok_mm=10.0,
            err_max_mm=60.0,
            joint_err_ok_deg=5.0,
            joint_err_max_deg=0.0,
        )
    )
    return JointPhaseSpec(
        mode=TaskMode.CARTESIAN_GOTO if move_mode == "cartesian" else TaskMode.JOINT_RESET,
        label=label,
        move_ref=move_ref,
        pose_target=pose_target,
        q_target_rad=q_target_rad,
        move_kp=move_kp,
        move_mode=move_mode,
        max_lin_vel_m_s=max_lin_vel_m_s,
        max_duration_s=max_duration_s,
        require_arrival=require_arrival,
        force_observer=force_observer,
        secondary=sec,
        governor=gov,
        scale_qdot_ff_with_governor=False,
        wait_until=(
            make_move_arrived(
                pose_target,
                q_target_rad,
                joint_only=(move_mode == "joint"),
                # Cartesian/SRS: TCP pose is the goal; joint residual is OK.
                require_joints=(move_mode == "joint"),
                tol_mm=5.0 if move_mode == "cartesian" else 3.0,
                tol_deg=3.0 if move_mode == "cartesian" else 1.5,
            )
            if pose_target is not None and q_target_rad is not None
            else None
        ),
    )


def phase_hybrid_track(
    reference: MotionReferenceSource,
    controller: AdmittanceController,
    *,
    desired_force: np.ndarray,
    label: str = "hybrid_track",
    duration_s: float | None = None,
    force_observer: Any = None,
    psi_rad_on_enter: float | None = None,
    governor: GovernorSpec | None = None,
    secondary: SecondaryPolicy | None = None,
) -> JointPhaseSpec:
    sec = secondary or SecondaryPolicy(preset="track", qdot_ff="off")
    if psi_rad_on_enter is not None and sec.arm_angle is None:
        sec.arm_angle = ArmAngleSpec(psi_rad=psi_rad_on_enter)
    return JointPhaseSpec(
        mode=TaskMode.HYBRID_TRACK,
        label=label,
        reference=reference,
        controller=controller,
        desired_force=np.asarray(desired_force, dtype=float),
        duration_s=duration_s,
        force_observer=force_observer,
        secondary=sec,
        governor=governor or GovernorSpec(err_ok_mm=10.0, err_max_mm=40.0),
    )


def _make_on_enter(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    psi = None
    if spec.secondary.arm_angle is not None:
        psi = spec.secondary.arm_angle.psi_rad

    def _enter() -> None:
        spec.secondary.apply(ctx.inner, psi_rad=psi)
        # move→D: soft-attract rail to the target pose's rail coordinate.
        if (
            spec.secondary.preset == "move"
            and spec.q_target_rad is not None
            and len(np.asarray(spec.q_target_rad).reshape(-1)) > 0
        ):
            y_tgt = float(np.asarray(spec.q_target_rad, dtype=float).reshape(-1)[0])
            ctx.inner.set_rail_pose_target(y_tgt)
            ctx.inner.set_rail_extension_mode("pose_attract")
            ctx.inner.set_rail_extension_active(True)
        if spec.mode == TaskMode.LOCKED_MOVE and spec.q_rail_target_m is not None:
            ctx.inner.set_locked(spec.locked_style, q_ref_m=spec.q_rail_target_m)

    return _enter


def _make_on_exit(spec: JointPhaseSpec, ctx: CompileContext) -> Callable[[], None] | None:
    if spec.mode != TaskMode.LOCKED_MOVE:
        return None

    def _exit() -> None:
        ctx.inner.set_locked(LockedStyle.HOLD, q_ref_m=float(ctx.inner.q_cmd[0]))

    return _exit


def compile_phase(spec: JointPhaseSpec, ctx: CompileContext) -> CompiledPhase:
    """Build a runtime ``Phase`` from a ``JointPhaseSpec``."""
    gov = spec.governor
    on_enter = _make_on_enter(spec, ctx)
    on_exit = _make_on_exit(spec, ctx)
    ff_ref = spec.rail_ref if spec.mode == TaskMode.LOCKED_MOVE else spec.move_ref
    qdot_ff = spec.secondary.make_qdot_ff_provider(ctx.inner, ff_ref)

    if spec.mode in (TaskMode.JOINT_RESET, TaskMode.CARTESIAN_GOTO):
        if spec.move_ref is None:
            raise ValueError(f"{spec.mode}: move_ref is required")
        v_max_scaled = ctx.kin.v_max * ctx.v_scale
        if spec.move_mode == "joint":
            outer = JointTrackOuterLoop(
                spec.move_ref,
                ctx.kin,
                JointTrackConfig(
                    k_joint=float(spec.move_kp),
                    max_joint_err_rad=0.35,
                    sigma_ref=spec.sigma_ref,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
                v_max_rad_s=v_max_scaled,
            )
        else:
            outer = CartesianTrackOuterLoop(
                spec.move_ref,
                CartesianTrackConfig(
                    k_task=np.full(6, spec.move_kp),
                    max_pos_err_m=0.05,
                    max_rot_err_rad=0.35,
                    max_lin_vel_m_s=spec.max_lin_vel_m_s,
                    control_frame=ctx.control_frame,
                    euler_order=ctx.euler_order,
                ),
            )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            soft_start_ramp_s=(0.3 if spec.secondary.preset == "move" else 0.0),
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        if spec.move_mode == "joint":
            attach_joint_move_rail(phase, ctx.inner)
            phase.scale_qdot_ff_with_governor = True
        # Wire ψ_ref(t) + centering when the move plan is SRS.
        if (
            isinstance(spec.move_ref, SrsSmoothMoveReference)
            and spec.q_target_rad is not None
        ):
            attach_srs_move_tracking(
                phase, ctx.inner, spec.move_ref, spec.q_target_rad
            )
            # Governor must scale the rail pin (avoid full-rate y_dot at soft-start).
            phase.scale_qdot_ff_with_governor = True
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            move_ref=spec.move_ref,
        )

    if spec.mode == TaskMode.LOCKED_MOVE:
        if spec.rail_ref is None:
            raise ValueError("locked_move: rail_ref is required")
        hold = HoldReference()
        kp = (
            float(spec.move_kp)
            if spec.locked_style == LockedStyle.TCP_FIXED
            else 0.0
        )
        outer = CartesianTrackOuterLoop(
            hold,
            CartesianTrackConfig(
                k_task=np.full(6, kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            qdot_ff_provider=qdot_ff,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            rail_ref=spec.rail_ref,
        )

    if spec.mode == TaskMode.CARTESIAN_TRACK:
        if spec.reference is None:
            raise ValueError(f"{spec.mode}: reference is required")
        outer = CartesianTrackOuterLoop(
            spec.reference,
            CartesianTrackConfig(
                k_task=np.full(6, spec.move_kp),
                max_pos_err_m=0.05,
                max_rot_err_rad=0.35,
                max_lin_vel_m_s=spec.max_lin_vel_m_s,
                control_frame=ctx.control_frame,
                euler_order=ctx.euler_order,
            ),
        )
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
            scale_qdot_ff_with_governor=spec.scale_qdot_ff_with_governor,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    if spec.mode == TaskMode.HYBRID_TRACK:
        if spec.reference is None or spec.controller is None:
            raise ValueError("hybrid_track: reference and controller are required")
        desired = spec.desired_force if spec.desired_force is not None else np.zeros(6)
        outer = AdmittanceOuterLoop(spec.controller, spec.reference, desired_force=desired)
        phase = Phase(
            outer=outer,
            label=spec.label or spec.mode.value,
            duration_s=spec.duration_s,
            max_duration_s=spec.max_duration_s,
            wait_until=spec.wait_until,
            on_enter=on_enter,
            on_exit=on_exit,
            require_arrival=spec.require_arrival,
            governor_err_ok_mm=gov.err_ok_mm,
            governor_err_max_mm=gov.err_max_mm,
            governor_joint_err_ok_deg=gov.joint_err_ok_deg,
            governor_joint_err_max_deg=gov.joint_err_max_deg,
            governor_tau_s=gov.tau_s,
            governor_freeze_below=gov.freeze_below,
            governor_release_above=gov.release_above,
            force_observer=spec.force_observer,
        )
        return CompiledPhase(
            phase=phase,
            label=phase.label,
            outer=outer,
            reference=spec.reference,
        )

    raise ValueError(f"unknown TaskMode: {spec.mode}")


def compile_phases(
    specs: list[JointPhaseSpec],
    ctx: CompileContext,
) -> list[CompiledPhase]:
    return [compile_phase(s, ctx) for s in specs]


from rm75_control.control.admittance_common.scaling import scale_admittance_for_desired_z

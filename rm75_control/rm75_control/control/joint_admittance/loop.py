"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

Two layers:

* ``JointIkController`` - the reusable, hardware-free inner loop.  Given the
  last commanded joint state, the measured joint state and a Cartesian twist,
  it runs slack-variable WBC QP IK, integrates and safety-clamps, and returns
  the next joint command.  There is deliberately NO low-pass filter on the send
  path: the QP velocity/acceleration box plus the SafetyLimiter already emit a
  C1-continuous stream, and any extra filtering here adds phase lag the outer
  loops would have to fight (a per-tick filter+sync stage on this path once
  attenuated every commanded velocity by ~6.7x - the 200mm move lag).

* ``run_joint_admittance_phases`` - the on-robot orchestration.  It feeds an
  outer loop's twist into ``JointIkController`` every tick and streams the
  result through ``rm_movej_canfd`` (mode 0, no driver-side filtering) on an
  absolute perf_counter schedule.  The Cartesian loop closes on the ENCODERS
  (Siciliano 1990 CLIK): the outer loop's pose feedback and the phase origin
  both come from FK(q_meas), and the reference clock is governed by tracking
  error so the reference can never run away from the physical arm.
"""

from __future__ import annotations

import csv
import inspect
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance.model import (
    RobotKinematics,
    deg2rad,
    max_joint_err_deg,
    pose_distance,
    pose_error,
    pose_track_error_mm_deg,
    rad2deg,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance.tasks.secondary_composer import SecondaryComposer
from rm75_control.control.joint_admittance.ik_types import saturate_error
from rm75_control.control.joint_admittance.utils.safety import (
    SafetyLimiter,
    SafetyLimits,
    Watchdog,
)


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class JointIkConfig:
    dt: float = 0.005
    control_frame: str = "tool"        # frame the incoming twist is expressed in
    euler_order: str = "xyz"
    qp: QpConfig = field(default_factory=QpConfig)
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    manipulability: ManipulabilityTaskConfig = field(default_factory=ManipulabilityTaskConfig)
    arm_angle: ArmAngleTaskConfig = field(default_factory=ArmAngleTaskConfig)
    # safety
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    a_max: float = 20.0                # rad/s^2 acceleration clamp (per joint)
    position_margin_rad: float = 0.017
    # Command-lead anti-windup: an extra QP velocity bound (never a position
    # jump) that stops q_cmd from leading the measured q by more than this
    # much per joint - the integrator is simply not allowed to command any
    # further motion in the direction that would grow the lead. 0 disables it.
    resync_err_rad: float = 0.10
    nullspace_d_null: float = 0.0          # viscous damping on secondary qdot (1/s)
    nullspace_d_null_adaptive: float = 1.0 # scale d_null up near joint limits
    # Per-joint cap on the composed soft secondary tasks (centering/arm/damping)
    # as a fraction of the URDF velocity limit.  Near a singularity the SR
    # projector passes secondary velocity straight through (N -> I); without a
    # cap the centering gradient from a far-from-nominal posture (straight arm)
    # commanded rad/s-scale self-motion while the Cartesian task was soft.
    nullspace_max_qdot_frac: float = 0.2


@dataclass
class JointIkStep:
    q_send: np.ndarray          # commanded joint position (rad) after clamp
    qdot: np.ndarray            # joint velocity (rad/s)
    twist_base: np.ndarray      # twist actually applied (base frame)
    sigma_min: float
    manip: float
    slack_norm: float
    n_cbf_active: int
    follow_err_rad: float       # max |q_meas - q_cmd| this tick (0 if no q_meas)
    cart_err_mm: float = 0.0    # outer-loop tracking error, filled by the caller
    qdot_ff_norm: float = 0.0
    arm_singularity_smooth: float = 1.0
    limit_activation: float = 0.0
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False


class JointIkController:
    """Reusable inner loop: (q_cmd, q_meas, twist) -> next joint command (rad)."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        self.cfg.qp.euler_order = self.cfg.euler_order
        self.centering_task = JointCenteringTask.from_kinematics(kin, self.cfg.nullspace)
        self.manipulability_task = (
            ManipulabilityTask(kin, self.cfg.manipulability)
            if self.cfg.manipulability.k_mu > 0.0
            else None
        )
        self.arm_task = (
            ArmAngleTask(kin, self.cfg.arm_angle) if self.cfg.arm_angle.enabled else None
        )
        self.limits = SafetyLimits.from_kinematics(
            kin,
            v_scale=self.cfg.v_scale,
            a_max=self.cfg.a_max,
            position_margin=self.cfg.position_margin_rad,
        )
        self.core = QpIkController(self.kin, self.limits, self.cfg.qp)
        self.safety = SafetyLimiter(self.limits)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self._arm_task_suppressed = False
        self._centering_suppressed = False
        self._manipulability_active = False
        self.secondary = SecondaryComposer.from_controller_parts(
            self.centering_task,
            self.arm_task,
            self.cfg.nullspace,
            manipulability=self.manipulability_task,
            d_null=self.cfg.nullspace_d_null,
            adaptive_d_null_gain=self.cfg.nullspace_d_null_adaptive,
            v_max=kin.v_max,
            max_qdot_frac=self.cfg.nullspace_max_qdot_frac,
        )
        self.last_secondary_norm: float = 0.0
        self.last_sigma_min: float = float(self.cfg.qp.sr_damping.sigma_ref)

    def set_arm_task_suppressed(self, suppressed: bool) -> None:
        """Pause the S-R-S arm-angle nullspace task (e.g. during a joint-space move).

        Pinning ``psi_ref`` to the IK target while the arm is still at ``q0`` with
        a different swivel angle fights the joint plan and can stall the move near
        singularities — re-enable at the scan/handoff pose once redundancy branch
        selection matters again.
        """
        self._arm_task_suppressed = bool(suppressed)

    def set_centering_suppressed(self, suppressed: bool) -> None:
        """Pause joint-centering nullspace (e.g. during a joint-space move).

        Centering pulls toward q_mid; near a kinematic singularity with a weak
        or frozen Cartesian task it can collapse the arm to a nominal posture
        instead of following the joint plan.
        """
        self._centering_suppressed = bool(suppressed)

    def set_manipulability_active(self, active: bool) -> None:
        """Use ∇μ ascent in the nullspace instead of Liegeois centering.

        Enable during large joint-space moves near singularities; disable at
        scan/handoff when centering and arm-angle branch selection matter again.
        """
        self._manipulability_active = bool(active) and self.manipulability_task is not None

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset(self.q_cmd)
        self.safety.reset(self.q_cmd)
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)

    def _twist_to_base(self, twist: np.ndarray, q_for_rot: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=float)
        if self.cfg.control_frame != "tool":
            return twist
        R = self.kin.fk_placement(q_for_rot).rotation
        out = np.zeros(6, dtype=float)
        out[:3] = R @ twist[:3]
        out[3:6] = R @ twist[3:6]
        return out

    def _secondary(self, q: np.ndarray, qdot_ff: np.ndarray | None) -> np.ndarray:
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            sigma_min=self.last_sigma_min,
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            manipulability_active=self._manipulability_active,
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
    ) -> JointIkStep:
        """One Cartesian-tracking WBC step.

        ``q_meas`` (encoder, rad) is used for the tool->base twist rotation (so
        the twist the outer loop computed against the MEASURED pose is rotated
        with the same orientation) and bounds the command integrator's lead
        over the physical arm - as a VELOCITY constraint inside the QP
        (``resync_err_rad``), never as a position teleport: capping the lead by
        directly reassigning ``q_cmd`` bypasses the velocity/acceleration box
        and can command a multi-degree joint step in one tick (rm_movej_canfd
        treats that as a discontinuity - visible as violent shake/jerk on
        hardware). Some follow lag during a fast move is normal servo
        behaviour, not a fault; the QP bound just stops it from growing
        further, at the normal velocity-limited rate.  ``qdot_ff`` is a
        joint-space feedforward projected onto the task nullspace together
        with the centering / arm-angle tasks.
        """
        dt = self.cfg.dt if dt is None else dt
        q_prev = self.q_cmd
        follow_err = 0.0 if q_meas is None else float(np.max(np.abs(q_prev - q_meas)))
        q_rot = q_meas if q_meas is not None else q_prev
        twist_base = self._twist_to_base(twist, q_rot)

        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(q_prev, qdot_ff),
            q_meas=q_meas,
            resync_err=self.cfg.resync_err_rad,
        )

        rep = self.safety.clamp(q_prev, r.q_next, dt)
        self.q_cmd = rep.q_safe
        # Keep the QP's velocity memory consistent with what was ACTUALLY
        # commanded: if the SafetyLimiter clipped the step, next tick's QP
        # acceleration box must be centered on the sent velocity, not on the
        # unclipped QP solution (that mismatch let consecutive ticks alternate
        # between an optimistic solve and a hard clamp - a chatter source).
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            self.core.qdot_prev = rep.dq / dt
        self.last_sigma_min = r.sigma_min
        return JointIkStep(
            q_send=rep.q_safe.copy(),
            qdot=r.qdot,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
            slack_norm=r.slack_norm,
            n_cbf_active=r.n_cbf_active,
            follow_err_rad=follow_err,
            qdot_ff_norm=float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0,
            arm_singularity_smooth=self.secondary.last_arm_smooth,
            limit_activation=self.secondary.last_limit_activation,
            vel_clamped=rep.vel_clamped,
            acc_clamped=rep.acc_clamped,
            pos_clamped=rep.pos_clamped,
        )


# ---------------------------------------------------------------------------
# Outer loops
# ---------------------------------------------------------------------------
class OuterLoop(Protocol):
    """Task-space controller producing a Cartesian twist each tick."""

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        """Return a 6D twist in the inner loop's control_frame."""
        ...


class AdmittanceOuterLoop:
    """Wrap AdmittanceController + a MotionReferenceSource.

    Force-position hybrid: tool-frame PBAC on the tracking axes, second-order
    admittance on the force axes (task-frame formalism, De Schutter 1988 /
    Bruyninckx 1996).  ``control_frame`` matches the AdmittanceController
    config (tool by default).
    """

    def __init__(self, controller, reference_source, *, desired_force: np.ndarray | None = None):
        self.controller = controller
        self.reference = reference_source
        self.desired_force = (
            np.zeros(6) if desired_force is None else np.asarray(desired_force, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_track_rot_deg: float = 0.0

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Reference-clock governor scale (0=frozen..1=realtime), forwarded to
        the admittance controller so its force integrator (v_force_z) pauses
        together with the reference instead of winding up against a frozen
        pose_d and shoving on resume."""
        if hasattr(self.controller, "set_time_scale"):
            self.controller.set_time_scale(scale)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        f_ext_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        ref = self.reference.sample(t_s)
        # Track-axis-only error (tool X/Y + attitude); the force axis (tool-Z)
        # is excluded - compliance there is not a tracking failure.
        tr_mm, tr_deg = pose_track_error_mm_deg(
            ref.pose_d,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
        return self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
        )


@dataclass
class CartesianTrackConfig:
    """PD + feedforward Cartesian tracking (no force axis)."""

    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 2.0))
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35
    max_lin_vel_m_s: float = 0.4
    max_ang_vel_rad_s: float = 1.5
    euler_order: str = "xyz"
    # MUST match the consuming JointIkConfig.control_frame: the PD+ff twist is
    # computed in base frame and rotated INTO tool axes when "tool", because
    # the inner loop rotates a "tool" twist back out with R @ twist.
    control_frame: str = "tool"


class CartesianTrackOuterLoop:
    """Point-to-point / trajectory tracking outer loop (no force).

    Wraps any MotionReferenceSource (typically ``JointSmoothMoveReference``, so
    point-to-point moves stay planned in joint space) and turns (pose_d, vel_ff)
    into a twist via PD + feedforward against the MEASURED pose.  Pair with
    ``Phase.qdot_ff_provider`` so the planned path's own redundancy resolution
    is kept alive in the QP nullspace while the primary task tracks FK(q_ref).
    """

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()
        self.last_err_mm: float = 0.0
        self.time_scale: float = 1.0

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Governor scale (0..1): scale trajectory vel_ff only, not the PD term."""
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self.reference.sample(t_s)
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        err_sat = saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_ff = np.asarray(ref.vel_ff, dtype=float) * self.time_scale
        v = v_ff + cfg.k_task * err_sat  # base-frame twist

        lin_n = float(np.linalg.norm(v[:3]))
        if cfg.max_lin_vel_m_s > 0.0 and lin_n > cfg.max_lin_vel_m_s:
            v[:3] *= cfg.max_lin_vel_m_s / lin_n
        ang_n = float(np.linalg.norm(v[3:6]))
        if cfg.max_ang_vel_rad_s > 0.0 and ang_n > cfg.max_ang_vel_rad_s:
            v[3:6] *= cfg.max_ang_vel_rad_s / ang_n

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            return out
        return v


@dataclass
class JointTrackConfig:
    """Joint-space PD + feedforward tracking for point-to-point moves.

    Unlike ``CartesianTrackOuterLoop``, the primary task twist is built from
    ``J(q_meas) @ (qdot_plan + k_joint * (q_ref - q_meas))`` — the same resolved-
    rate structure vendor ``rm_movej`` uses (pure joint interpolation with no
    Cartesian feedback loop to stall on near kinematic singularities).  WBC
    nullspace centering, CBF and velocity/acceleration boxes still run every
    tick on top.
    """

    k_joint: float = 2.0
    max_joint_err_rad: float = 0.35
    sigma_ref: float = 0.08
    # σ-adaptive k_eff floor: k_eff = k_joint * max(σ/σ_ref, floor).  0.2 lets
    # k_eff track σ/σ_ref continuously below σ≈0.016 instead of pinning at 1.0
    # through σ∈[0.02,0.04] (which drove infeasible v_cmd and slack chatter).
    k_joint_sigma_min_frac: float = 0.2
    control_frame: str = "tool"
    euler_order: str = "xyz"
    # Rise-only slew on k_eff: exiting a singular dip spreads accumulated
    # q_err discharge over ~1s instead of a one-tick TCP overshoot.
    k_joint_rise_per_s: float = 1.2
    # LPF on nullspace fb (see last_qdot_fb); kills ~20 Hz QP dual oscillation.
    fb_lpf_tau_s: float = 0.015
    # Scale nullspace fb pull; primary v_cmd still uses full k_eff·q_err.
    fb_secondary_gain: float = 0.4


class JointTrackOuterLoop:
    """MoveJ-like outer loop: track ``JointSmoothMoveReference`` in joint space.

    Requires ``q_meas`` (rad) passed into ``sample`` — the orchestration loop
    detects this via the ``q_meas`` keyword and supplies encoder feedback.
    """

    def __init__(
        self,
        reference,
        kin: RobotKinematics,
        cfg: JointTrackConfig | None = None,
        *,
        v_max_rad_s: np.ndarray | None = None,
    ) -> None:
        self.reference = reference
        self.kin = kin
        self.cfg = cfg or JointTrackConfig()
        self.v_max = (
            np.asarray(v_max_rad_s, dtype=float)
            if v_max_rad_s is not None
            else np.asarray(kin.v_max, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_joint_err_deg: float = 0.0
        self.last_sigma_min: float = 0.0
        # k_eff·q_err fed into QP secondary (plan ff stays governor-scaled).
        self.last_qdot_fb: np.ndarray | None = None
        self._qdot_fb_lpf: np.ndarray | None = None
        self._k_eff_prev: float | None = None
        self._t_prev: float | None = None

    def set_origin(self, pose0: np.ndarray) -> None:
        if hasattr(self.reference, "set_origin"):
            self.reference.set_origin(pose0)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        *,
        q_meas: np.ndarray | None = None,
    ) -> np.ndarray:
        del f_ext
        if q_meas is None:
            raise RuntimeError("JointTrackOuterLoop.sample requires q_meas")
        cfg = self.cfg
        q_ref, qdot_plan = self.reference.sample_q(t_s)
        q_meas = np.asarray(q_meas, dtype=float)
        q_err = np.clip(
            wrap_joint_delta(q_meas, q_ref),
            -cfg.max_joint_err_rad,
            cfg.max_joint_err_rad,
        )
        self.last_joint_err_deg = max_joint_err_deg(q_meas, q_ref)
        J = self.kin.jacobian(q_meas)
        sigma = self.kin.singular_values(J)
        sigma_min = float(sigma.min())
        self.last_sigma_min = sigma_min
        if cfg.sigma_ref > 1e-9:
            k_target = cfg.k_joint * float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        else:
            k_target = cfg.k_joint
        if (
            self._k_eff_prev is None
            or self._t_prev is None
            or cfg.k_joint_rise_per_s <= 0.0
            or k_target <= self._k_eff_prev
        ):
            k_eff = k_target
        else:
            dt_eff = max(0.0, t_s - self._t_prev)
            k_eff = min(k_target, self._k_eff_prev + cfg.k_joint_rise_per_s * dt_eff)
        dt_eff_lpf = 0.005 if self._t_prev is None else max(1e-4, t_s - self._t_prev)
        self._k_eff_prev = k_eff
        self._t_prev = t_s
        qdot_fb_raw = k_eff * q_err
        if self._qdot_fb_lpf is None or cfg.fb_lpf_tau_s <= 0.0:
            self._qdot_fb_lpf = qdot_fb_raw.copy()
        else:
            alpha = dt_eff_lpf / (cfg.fb_lpf_tau_s + dt_eff_lpf)
            self._qdot_fb_lpf = self._qdot_fb_lpf + alpha * (qdot_fb_raw - self._qdot_fb_lpf)
        self.last_qdot_fb = self._qdot_fb_lpf * float(cfg.fb_secondary_gain)
        qdot_cmd = qdot_plan + qdot_fb_raw
        v_lim = np.asarray(self.v_max, dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -v_lim, v_lim)
        v_base = J @ qdot_cmd
        q_err_deg = float(np.max(np.abs(np.rad2deg(q_err))))
        feas = 1.0
        if cfg.sigma_ref > 1e-9 and sigma_min < cfg.sigma_ref:
            feas = float(
                np.clip(sigma_min / cfg.sigma_ref, cfg.k_joint_sigma_min_frac, 1.0)
            )
        if q_err_deg > 8.0 and sigma_min < cfg.sigma_ref * 1.5:
            feas *= min(1.0, 8.0 / q_err_deg)
        if feas < 1.0:
            v_base = feas * v_base
        pose_ref = self.kin.fk_pose(q_ref)
        err = pose_error(pose_ref, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v_base[:3]
            out[3:6] = R.T @ v_base[3:6]
            return out
        return v_base


def arrived(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    *,
    tol_mm: float = 1.0,
    tol_deg: float = 0.5,
    euler_order: str = "xyz",
) -> bool:
    """Convenience Phase.wait_until predicate: True once within tolerance of target."""
    d_mm, d_deg = pose_distance(current_pose, target_pose, euler_order)
    return d_mm <= tol_mm and d_deg <= tol_deg


# ---------------------------------------------------------------------------
# On-robot orchestration
# ---------------------------------------------------------------------------
def _set_realtime_priority(priority: int = 80) -> bool:
    """Best-effort SCHED_FIFO for the control thread (needs CAP_SYS_NICE / root)."""
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return True
    except (PermissionError, OSError, AttributeError):
        return False


# Linux ``time.sleep`` often wakes 1–3 ms late; spin the last slice for tighter
# 200 Hz CANFD pacing (reduces the "sudden stall" feel when a tick oversleeps).
_SPIN_MARGIN_S = 0.001


def _wait_until(deadline: float) -> None:
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > _SPIN_MARGIN_S:
            time.sleep(remaining - _SPIN_MARGIN_S)


def _resync_late_tick(next_tick: float, now: float, dt: float) -> tuple[float, float]:
    """If we missed a whole period, jump the schedule forward instead of bursting.

    Returns ``(next_tick, late_ms)`` where ``late_ms`` is how far ``now`` was
    past the scheduled tick start (always >= 0).
    """
    late_s = now - next_tick
    if late_s > dt:
        return now, late_s * 1000.0
    return next_tick, max(0.0, late_s * 1000.0)


@dataclass
class LoopResult:
    ticks: int
    duration_s: float
    max_jitter_ms: float
    stalled: bool
    stutter_count: int = 0


@dataclass
class Phase:
    """One leg of a multi-phase on-robot run, e.g. "walk to D" then "sin scan at D".

    All phases share the SAME inner loop, async state reader and watchdog -
    there is no MoveJ/MoveV switch and no gap in the joint-command stream at
    the phase boundary, only ``outer`` (and optionally ``force_observer``)
    changing.

    Reference-clock governor: the phase reference time ``t_ref`` (what the
    outer loop's reference is sampled at) advances by ``dt * scale`` each tick,
    where the raw ``scale`` fades 1 -> 0 as the outer loop's tracking error
    grows from ``governor_err_ok_mm`` to ``governor_err_max_mm`` (and/or the
    joint-space band).  The reference therefore waits for the physical arm
    instead of running away on the wall clock.  Set ``governor_err_max_mm`` to
    0 to disable the Cartesian governor (e.g. MoveJ-like joint moves, where
    Cartesian deviation through a singular region is expected, not a fault).

    The raw scale is passed through a first-order low-pass + freeze hysteresis
    (``GovernorFilter``) before it multiplies ``dt``: the raw error->scale map
    is a static gain inside the tracking loop and, applied directly, forms a
    limit cycle with the outer PD (err grows -> reference slows -> err shrinks
    -> reference accelerates -> err grows...).  The filter breaks that loop;
    the hysteresis keeps a hard freeze from chattering on/off at the max-error
    threshold.  ``qdot_ff_provider`` is ALWAYS sampled at the same governed
    ``t_ref`` as the pose reference, so the plan feedforward and the tracking
    reference can never diverge (the old dual-clock "self-motion escape"
    replayed the feedforward on a separate clock against a frozen reference -
    the two fought each other and shook the arm).
    """

    outer: OuterLoop
    label: str = ""
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # wall-clock safety cap
    wait_until: object | None = None         # Callable pose or (pose, q_meas) -> bool
    qdot_ff_provider: object | None = None   # Callable[[float], qdot_ff_rad_s] sampled at t_ref
    scale_qdot_ff_with_governor: bool = True # False keeps plan-anchor alive when t_ref frozen
    require_arrival: bool = False            # abort later phases if wait_until never fires
    governor_err_ok_mm: float = 5.0
    governor_err_max_mm: float = 25.0
    # Joint-space governor (JointTrackOuterLoop): scale t_ref from max joint
    # tracking error in deg.  Set ``governor_joint_err_max_deg > 0`` to enable.
    governor_joint_err_ok_deg: float = 3.0
    governor_joint_err_max_deg: float = 0.0
    # GovernorFilter tuning: low-pass time constant and freeze hysteresis band.
    governor_tau_s: float = 0.2
    governor_freeze_below: float = 0.02
    governor_release_above: float = 0.10
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin
    on_tick: object | None = None            # Callable[[float, JointIkStep, np.ndarray], None]


class _TickLogger:
    """Per-tick CSV telemetry (q_cmd/q_meas/twist/slack/clamp flags/force).

    Rows are queued and written on a background thread so disk I/O cannot stall
    the 200 Hz control loop (sync flush was a common source of 10+ ms hitches).
    """

    _HEADER = (
        ["t_wall_s", "phase", "t_ref_s"]
        + [f"q_cmd_{i}" for i in range(1, 8)]
        + [f"q_meas_{i}" for i in range(1, 8)]
        + [f"pose_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        + [f"twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + ["track_err_mm", "follow_err_deg", "slack_norm", "n_cbf",
           "vel_clamped", "acc_clamped", "pos_clamped", "fx", "fy", "fz",
           "instability_idx", "instability_idx_raw", "instability_idx_active",
           "damping_z_eff", "v_force_z", "ke_est",
           "f_des_z_eff", "v_r_z", "takeover",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_update_gated", "ke_dx_m", "ke_df_n", "ke_update_count",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "arm_singularity_smooth", "limit_activation"]
    )

    def __init__(self, path: str) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            args=(path,),
            name="joint-admittance-csv",
            daemon=True,
        )
        self._worker.start()

    def _run(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._HEADER)
            n = 0
            while True:
                if self._stop.is_set() and self._q.empty():
                    break
                try:
                    row = self._q.get(timeout=0.05)
                except queue.Empty:
                    continue
                if row is None:
                    break
                w.writerow(row)
                n += 1
                if n % 200 == 0:
                    f.flush()

    def write(
        self,
        t_wall,
        label,
        t_ref,
        step: JointIkStep,
        q_meas,
        pose,
        f_ext,
        outer=None,
        *,
        governor_scale: float = float("nan"),
        governor_scale_raw: float = float("nan"),
        v_max: np.ndarray | None = None,
    ) -> None:
        qm = q_meas if q_meas is not None else np.full(7, np.nan)
        ctrl = getattr(outer, "controller", None)
        is_idx = getattr(ctrl, "instability_index", float("nan"))
        is_idx_raw = getattr(ctrl, "instability_index_raw", float("nan"))
        d_eff = getattr(ctrl, "damping_z_eff", float("nan"))
        v_fz = getattr(ctrl, "v_force_z", float("nan"))
        ke_est = getattr(ctrl, "ke_est", float("nan"))
        f_des_eff = getattr(ctrl, "f_des_z_eff", float("nan"))
        v_r_z = getattr(ctrl, "v_r_z", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        force_pred_z = getattr(ctrl, "force_pred_z", float("nan"))
        force_dot_z = getattr(ctrl, "force_dot_z", float("nan"))
        cap_press_z = getattr(ctrl, "cap_press_z", float("nan"))
        cap_retract_z = getattr(ctrl, "cap_retract_z", float("nan"))
        ke_tracker = getattr(ctrl, "_ke_estimator", None)
        ke_update_gated = getattr(ke_tracker, "update_gated", False)
        ke_dx_m = getattr(ke_tracker, "last_dx_m", float("nan"))
        ke_df_n = getattr(ke_tracker, "last_df_n", float("nan"))
        ke_update_count = getattr(ke_tracker, "update_count", 0)
        qdot_norm = float(np.linalg.norm(step.qdot))
        # Fraction of the per-joint velocity box actually used (1.0 = saturated
        # on at least one joint) - the clearest signal for "CBF/limits are
        # strangling the commanded twist" vs "the twist itself is just small".
        if v_max is not None and np.any(v_max > 1e-9):
            qdot_max_frac = float(np.max(np.abs(step.qdot) / np.maximum(v_max, 1e-9)))
        else:
            qdot_max_frac = float("nan")
        self._q.put(
            [f"{t_wall:.4f}", label, f"{t_ref:.4f}"]
            + [f"{v:.6f}" for v in step.q_send]
            + [f"{v:.6f}" for v in qm]
            + [f"{v:.6f}" for v in pose]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{step.cart_err_mm:.3f}", f"{np.degrees(step.follow_err_rad):.4f}",
               f"{step.slack_norm:.5f}", step.n_cbf_active,
               int(step.vel_clamped), int(step.acc_clamped), int(step.pos_clamped),
               f"{f_ext[0]:.3f}", f"{f_ext[1]:.3f}", f"{f_ext[2]:.3f}",
               f"{is_idx:.4f}", f"{is_idx_raw:.4f}", f"{is_idx:.4f}",
               f"{d_eff:.2f}", f"{v_fz:.5f}", f"{ke_est:.1f}",
               f"{f_des_eff:.3f}", f"{v_r_z:.5f}", int(bool(takeover)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               int(bool(ke_update_gated)), f"{ke_dx_m:.8f}", f"{ke_df_n:.5f}",
               int(ke_update_count),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.arm_singularity_smooth:.4f}",
               f"{step.limit_activation:.4f}"]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=10.0)


def _joint_plan_err_deg(outer: OuterLoop, t_ref: float, q_meas: np.ndarray) -> float | None:
    """Max |q_ref(t_ref) - q_meas| in deg from the outer loop's joint reference."""
    ref = getattr(outer, "reference", None)
    if ref is None or not hasattr(ref, "sample_q"):
        return None
    q_ref, _ = ref.sample_q(t_ref)
    return max_joint_err_deg(q_meas, q_ref)


def _reference_governor_scale(
    phase: Phase,
    *,
    outer_err_mm: float | None,
    joint_err_deg: float | None,
) -> float:
    """Raw reference-clock scale in [0, 1] from the tracking-error bands.

    Multiple active governors combine with min (most conservative).  This is
    the STATIC error->scale map only; smoothing/hysteresis live in
    ``GovernorFilter`` (applying this raw gain directly closed a limit cycle
    with the outer tracking PD).
    """
    scales: list[float] = []

    if phase.governor_joint_err_max_deg > 0.0 and joint_err_deg is not None:
        e0, e1 = phase.governor_joint_err_ok_deg, phase.governor_joint_err_max_deg
        if e1 > e0:
            scales.append(float(np.clip((e1 - joint_err_deg) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if phase.governor_err_max_mm > 0.0 and outer_err_mm is not None:
        e0, e1 = phase.governor_err_ok_mm, phase.governor_err_max_mm
        if e1 > e0:
            scales.append(float(np.clip((e1 - outer_err_mm) / (e1 - e0), 0.0, 1.0)))

    return min(scales) if scales else 1.0


class GovernorFilter:
    """First-order low-pass + freeze hysteresis on the governor scale.

    The filtered state keeps integrating even while frozen, so on release the
    output resumes from a continuous value instead of stepping - the reference
    clock rate is C0-continuous everywhere except the (intentional) hard
    freeze, which only engages/disengages through the hysteresis band.
    """

    def __init__(
        self,
        tau_s: float = 0.2,
        freeze_below: float = 0.02,
        release_above: float = 0.10,
    ) -> None:
        self.tau_s = float(tau_s)
        self.freeze_below = float(freeze_below)
        self.release_above = float(release_above)
        self.scale = 1.0
        self.frozen = False

    def update(self, raw: float, dt: float) -> float:
        raw = float(np.clip(raw, 0.0, 1.0))
        alpha = 1.0 if self.tau_s <= 0.0 else min(1.0, dt / self.tau_s)
        self.scale += alpha * (raw - self.scale)
        if self.frozen:
            if raw >= self.release_above and self.scale >= self.release_above:
                self.frozen = False
        elif self.scale <= self.freeze_below:
            self.frozen = True
        return 0.0 if self.frozen else self.scale


def run_joint_admittance_phases(
    session,
    phases: list[Phase],
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
) -> LoopResult:
    """Run a sequence of ``Phase`` objects on the real robot, one continuous stream.

    Sequence:
      1. move_j to q_start (single planned motion; the only non-CANFD command).
      2. Start the async state reader; read q0 and reset the inner loop at it.
      3. For each phase, at fixed dt (perf_counter absolute schedule):
           outer.sample(t_ref, FK(q_meas), f_ext) -> inner.update -> rm_movej_canfd,
         with t_ref governed by tracking error (see Phase).  A phase ends when
         t_ref >= duration_s, wait_until(pose_meas) is True, or the wall-clock
         cap max_duration_s is hit.
    """
    from rm75_control.control.admittance_common.async_state import create_state_observer
    from rm75_control.motion.canfd import send_joint_canfd

    dt = inner.cfg.dt if dt is None else dt
    robot = session.robot

    if q_start_deg is not None:
        session.move_joints(list(np.asarray(q_start_deg, dtype=float)), velocity_percent=move_speed, block=1)
        time.sleep(0.5)

    async_obs = create_state_observer(robot, session.config, robot_ip=session.ip)
    async_obs.start()
    if verbose:
        print(
            f"  feedback: UDP push {async_obs.push_period_ms:.0f}ms "
            f"port={async_obs.config.port} ip={async_obs._target_ip}",
            flush=True,
        )
    ticks = 0
    max_jitter_ms = 0.0
    stutter_count = 0
    stalled = False
    total_t0 = time.perf_counter()
    logger = _TickLogger(log_csv) if log_csv else None
    try:
        _pose0_rm = async_obs.wait_first_pose(timeout_s=5.0)
        snap0 = async_obs.read()
        if snap0.q_deg is None:
            raise RuntimeError("no joint feedback from robot")
        q0_rad = deg2rad(snap0.q_deg)
        # The whole Cartesian loop (inner and outer) uses the Pinocchio tcp
        # frame; Realman FK for the active tool may differ.
        pose0 = inner.kin.fk_pose(q0_rad)
        inner.reset(q0_rad)
        if snap0.pose is not None:
            d_mm, _ = pose_distance(snap0.pose, pose0, inner.cfg.euler_order)
            if d_mm > 5.0 and verbose:
                print(
                    f"  FK note: Realman vs Pinocchio tcp {d_mm:.1f}mm "
                    "(Cartesian loop uses Pinocchio)",
                    flush=True,
                )

        if realtime and not _set_realtime_priority():
            if verbose:
                print("  (SCHED_FIFO unavailable - running at normal priority)", flush=True)

        def _hold() -> None:
            # watchdog stall action: hold at the last commanded joint state
            try:
                send_joint_canfd(robot, rad2deg(inner.q_cmd), follow=False)
            except Exception:
                try:
                    robot.rm_set_arm_slow_stop()
                except Exception:
                    pass

        wd = Watchdog(watchdog_timeout_s, _hold)
        wd.start()
        try:
            pose_rm = _pose0_rm
            q_meas = q0_rad
            pose_pin = pose0
            jump_warn_t = 0.0
            try:
                for phase in phases:
                    if verbose:
                        print(f"-- phase: {phase.label or phase.outer.__class__.__name__} --", flush=True)
                    # Phase origin from the ENCODERS, never from the command integrator.
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        q_meas = deg2rad(snap.q_deg)
                    pose_pin = inner.kin.fk_pose(q_meas)
                    if hasattr(phase.outer, "set_origin"):
                        phase.outer.set_origin(pose_pin)
                    if phase.on_enter is not None:
                        phase.on_enter()

                    obs = phase.force_observer if phase.force_observer is not None else force_observer
                    phase_t0 = time.perf_counter()
                    next_tick = phase_t0
                    t_ref = 0.0
                    gov_filter = GovernorFilter(
                        tau_s=phase.governor_tau_s,
                        freeze_below=phase.governor_freeze_below,
                        release_above=phase.governor_release_above,
                    )
                    scale = 1.0
                    phase_arrived = False
                    while True:
                        now = time.perf_counter()
                        next_tick, late_ms = _resync_late_tick(next_tick, now, dt)
                        if late_ms > dt * 1000.0:
                            stutter_count += 1
                        max_jitter_ms = max(max_jitter_ms, late_ms)
                        t_wall = now - phase_t0
                        if phase.duration_s is not None and t_ref >= phase.duration_s:
                            break
                        if phase.max_duration_s is not None and t_wall >= phase.max_duration_s:
                            break
    
                        snap = async_obs.read()
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            q_meas = deg2rad(snap.q_deg)
                            pose_pin = inner.kin.fk_pose(q_meas)
                        f_ext = np.zeros(6)
                        f_ext_raw = None
                        if obs is not None:
                            pose_l7 = inner.kin.frame_pose(q_meas, "link_7")
                            _signed, f_ext = obs.update(now - total_t0, pose_l7, snap.force_raw)
                            f_ext_raw = getattr(obs, "f_ext_raw_last", None)
    
                        q_prev = inner.q_cmd.copy()
                        # Forward the previous tick's governed scale to the outer
                        # loop BEFORE sampling: an admittance outer freezes its
                        # force-integrator together with the reference clock, so a
                        # frozen t_ref cannot wind up v_force_z and shove on resume.
                        if hasattr(phase.outer, "set_time_scale"):
                            phase.outer.set_time_scale(scale)
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered compensated wrench for the Dimeas index
                            # (the 6 Hz control LPF hides the instability band).
                            sample_kwargs["f_ext_raw"] = f_ext_raw
                        twist = np.asarray(
                            phase.outer.sample(t_ref, pose_pin, f_ext, **sample_kwargs),
                            dtype=float,
                        )
                        qdot_ff = (
                            phase.qdot_ff_provider(t_ref)
                            if phase.qdot_ff_provider is not None
                            else None
                        )
                        if qdot_ff is not None:
                            qdot_ff = np.asarray(qdot_ff, dtype=float)
                            if phase.scale_qdot_ff_with_governor:
                                qdot_ff = qdot_ff * scale
                        # Joint-space nullspace fb from JointTrackOuterLoop
                        # (k_eff·q_err LPF); additive to plan ff, not governor-scaled.
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        step = inner.update(twist, dt, q_meas=q_meas, qdot_ff=qdot_ff)
                        outer_err_mm = getattr(phase.outer, "last_err_mm", None)
                        if outer_err_mm is not None:
                            step.cart_err_mm = outer_err_mm
                        send_joint_canfd(robot, rad2deg(step.q_send), follow=follow)
                        wd.beat()
    
                        # Reference-clock governor: reference waits for the arm.
                        joint_err_deg = getattr(phase.outer, "last_joint_err_deg", None)
                        if joint_err_deg is None:
                            joint_err_deg = _joint_plan_err_deg(phase.outer, t_ref, q_meas)
                        raw_scale = _reference_governor_scale(
                            phase,
                            outer_err_mm=outer_err_mm,
                            joint_err_deg=joint_err_deg,
                        )
                        scale = gov_filter.update(raw_scale, dt)
                        t_ref += dt * scale
    
                        if phase.on_tick is not None:
                            phase.on_tick(t_ref, step, q_meas)
    
                        dq_deg = np.abs(rad2deg(step.q_send - q_prev))
                        if verbose and now - jump_warn_t >= 1.0 and np.any(dq_deg > 1.5):
                            jump_warn_t = now
                            j = int(np.argmax(dq_deg)) + 1
                            print(
                                f"  warn: joint jump J{j} {dq_deg.max():.2f}deg/tick "
                                f"(>{1.5:.1f} @ {dt*1000:.0f}ms)",
                                flush=True,
                            )
    
                        if logger is not None:
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                            )
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)
    
                        if phase.wait_until is not None:
                            n_wait = len(inspect.signature(phase.wait_until).parameters)
                            if n_wait >= 2:
                                phase_arrived = bool(phase.wait_until(pose_pin, q_meas))
                            else:
                                phase_arrived = bool(phase.wait_until(pose_pin))
                            if phase_arrived:
                                break
    
                        ticks += 1
                        next_tick += dt
                        _wait_until(next_tick)
    
                    if phase.require_arrival and not phase_arrived:
                        err_mm = getattr(phase.outer, "last_err_mm", float("nan"))
                        jq = getattr(phase.outer, "last_joint_err_deg", float("nan"))
                        if verbose:
                            print(
                                f"  ERROR: phase {phase.label!r} did not reach target "
                                f"(t_ref={t_ref:.2f}s, wall={t_wall:.1f}s, "
                                f"track={err_mm:.0f}mm, jq={jq:.1f}deg) "
                                f"— skipping remaining phases",
                                flush=True,
                            )
                        break
            except KeyboardInterrupt:
                if verbose:
                    print("\nStopped.", flush=True)
        finally:
            wd.stop()
            stalled = wd.fired
    finally:
        async_obs.stop()
        if logger is not None:
            logger.close()

    total_s = time.perf_counter() - total_t0
    if verbose:
        stutter_note = f", {stutter_count} stutter(s)" if stutter_count else ""
        print(
            f"  joint-admittance loop: {ticks} ticks, {total_s:.1f}s, "
            f"max jitter {max_jitter_ms:.2f} ms{stutter_note}"
            f"{' [WATCHDOG FIRED]' if stalled else ''}",
            flush=True,
        )
    return LoopResult(
        ticks=ticks,
        duration_s=total_s,
        max_jitter_ms=max_jitter_ms,
        stalled=stalled,
        stutter_count=stutter_count,
    )


def run_joint_admittance_loop(
    session,
    outer: OuterLoop,
    inner: JointIkController,
    *,
    q_start_deg: np.ndarray | None = None,
    duration_s: float = 10.0,
    dt: float | None = None,
    force_observer=None,
    follow: bool = True,
    move_speed: int = 20,
    realtime: bool = False,
    watchdog_timeout_s: float = 0.1,
    on_step=None,
    log_csv: str | None = None,
    verbose: bool = True,
) -> LoopResult:
    """Single-phase convenience wrapper around ``run_joint_admittance_phases``."""
    phase = Phase(outer=outer, label="run", duration_s=duration_s)
    on_step_1 = None if on_step is None else (lambda label, t, step, pose, f_ext: on_step(t, step, pose, f_ext))
    return run_joint_admittance_phases(
        session,
        [phase],
        inner,
        q_start_deg=q_start_deg,
        dt=dt,
        force_observer=force_observer,
        follow=follow,
        move_speed=move_speed,
        realtime=realtime,
        watchdog_timeout_s=watchdog_timeout_s,
        on_step=on_step_1,
        log_csv=log_csv,
        verbose=verbose,
    )

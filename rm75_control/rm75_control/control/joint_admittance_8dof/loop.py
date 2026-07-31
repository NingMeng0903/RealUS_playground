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

from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
from rm75_control.control.joint_admittance_8dof.model import (
    RobotKinematics,
    arm_q_from_full,
    deg2rad,
    full_q_from_arm,
    max_joint_err_deg,
    pose_distance,
    pose_error,
    pose_track_error_mm_deg,
    rad2deg,
    wrap_joint_delta,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import (
    RailLockConfig,
    RailLockTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.tasks.arm_angle import (
    ArmAngleTask,
    ArmAngleTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.manipulability_task import (
    ManipulabilityTask,
    ManipulabilityTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.nullspace_task import (
    JointCenteringTask,
    NullspaceTaskConfig,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_extension import (
    RailExtensionConfig,
    RailExtensionTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import SecondaryComposer
from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.utils.safety import (
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
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    # Preferred-extension rail coordination (COUPLED mode only): the rail
    # proactively follows the TCP when the arm reaches beyond its comfortable
    # extension, keeping the arm away from stretched-singular postures.
    rail_extension: RailExtensionConfig = field(default_factory=RailExtensionConfig)
    # safety
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    # Acceleration limits are UNIT-SEPARATED: rail is m/s^2, arm is rad/s^2.
    # A single scalar mixed the two and gave the prismatic joint a de-facto
    # 20 m/s^2 limit (0 -> 0.2 m/s in 10 ms — no accel limit at all).
    a_max_arm_rad_s2: float = 20.0     # rad/s^2 per arm joint (1..7)
    a_max_rail_m_s2: float = 0.30      # m/s^2 for prismatic rail (0)
    position_margin_rad: float = 0.017
    # Rail position margin in METRES: the scalar rad margin applied to the
    # prismatic joint stole 2 deg = 35 mm of rail travel.
    position_margin_rail_m: float = 0.0
    # Command-lead anti-windup: an extra QP velocity bound (never a position
    # jump) that stops q_cmd from leading the measured q by more than this
    # much per joint - the integrator is simply not allowed to command any
    # further motion in the direction that would grow the lead. 0 disables it.
    resync_err_rad: float = 0.10       # arm joints 1..7 (radians)
    resync_err_rail_m: float = 0.020   # rail joint 0 (metres; 20 mm)
    nullspace_d_null: float = 0.0          # viscous damping on secondary qdot (1/s)
    nullspace_d_null_adaptive: float = 1.0 # scale d_null up near joint limits
    # Per-joint cap on the composed soft secondary tasks (centering/arm/damping)
    # as a fraction of the URDF velocity limit.  Near a singularity the SR
    # projector passes secondary velocity straight through (N -> I); without a
    # cap the centering gradient from a far-from-nominal posture (straight arm)
    # commanded rad/s-scale self-motion while the Cartesian task was soft.
    nullspace_max_qdot_frac: float = 0.2
    # After an actual singularity escape, keep a stronger posture pull until
    # the arm is close to the active YAML/runtime target.  This is latched by
    # the escape event, rather than being a short pulse that can expire before
    # the Cartesian nullspace has enough freedom to recover the posture.
    centering_recovery_gain: float = 3.0
    centering_recovery_max_qdot_frac: float = 0.35
    centering_recovery_tol: float = 0.12


@dataclass
class JointIkStep:
    q_send: np.ndarray          # commanded joint position (rad) after clamp
    qdot: np.ndarray            # joint velocity (rad/s)
    twist_base: np.ndarray      # requested task twist in the base frame
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
    tcp_jump_mm: float = 0.0
    # Preferred-extension rail task telemetry (COUPLED mode).
    rail_ext_err_m: float = 0.0
    rail_ext_weight: float = 0.0
    # Debug: how the rail was driven this tick (plan pin vs free QP).
    rail_vel_pin: float = float("nan")      # m/s hard pin, or NaN if free
    rail_qdot_ff: float = float("nan")      # plan qdot_ff[0] before strip
    plan_drives_rail: bool = False


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
        self.rail_task = RailLockTask(self.cfg.rail)
        self.rail_ext_task = (
            RailExtensionTask(kin, self.cfg.rail_extension)
            if self.cfg.rail_extension.enabled
            else None
        )
        # Preset-gated (api.py): pose_attract during move→D; reach during
        # track/scan; off during hold (rail is pinned anyway).
        self._rail_ext_active = True
        # Bug 2: σ-escape gradient cache — updated every ``_sigma_grad_period``
        # ticks (default 10 → 20 Hz at dt=5 ms).  The gradient is smooth on
        # this timescale (way slower than rail acceleration bandwidth).
        # Sourced via the pluggable RailGoodness (default: SigmaMinGoodness).
        from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
            CachedRailGoodness,
            SigmaMinGoodness,
        )

        self._rail_goodness = CachedRailGoodness(
            SigmaMinGoodness(kin), period_ticks=10
        )
        self._sigma_grad_rail_cached: float = 0.0
        self._sigma_grad_tick: int = 0
        self._sigma_grad_period: int = 10
        # Build an 8-vector a_max: rail is m/s^2, arm joints 1..7 are rad/s^2.
        a_max_vec = np.full(kin.nv, float(self.cfg.a_max_arm_rad_s2))
        a_max_vec[0] = float(self.cfg.a_max_rail_m_s2)
        # Position margin is unit-separated too: arm rad, rail metres.
        margin_vec = np.full(kin.nv, float(self.cfg.position_margin_rad))
        margin_vec[0] = float(self.cfg.position_margin_rail_m)
        self.limits = SafetyLimits.from_kinematics(
            kin,
            v_scale=self.cfg.v_scale,
            a_max=a_max_vec,
            position_margin=margin_vec,
        )
        if self.cfg.rail.v_max_m_s is not None:
            self.limits.v_max[0] = min(
                float(self.limits.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
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
            rail_lock=self.rail_task,
            d_null=self.cfg.nullspace_d_null,
            adaptive_d_null_gain=self.cfg.nullspace_d_null_adaptive,
            v_max=kin.v_max,
            max_qdot_frac=self.cfg.nullspace_max_qdot_frac,
        )
        self.last_secondary_norm: float = 0.0
        self.last_sigma_min: float = float(self.cfg.qp.sr_damping.sigma_ref)
        self._singularity_escape_seen: bool = False
        self._centering_recovery_active: bool = False
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        # Snapshot of the CONFIGURED (yaml) rail mode.  _apply_rail_mode_side_
        # effects() writes the live mode back into cfg.rail (shared with
        # RailLockTask), so cfg.rail.mode is destroyed by the first hold/lock
        # phase — presets that want to restore "what the yaml asked for"
        # (e.g. track re-coupling after hold@D) must consult this snapshot.
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        # When True, SRS (or other) plan owns rail velocity via qdot_ff pin —
        # prevents the arm alone from absorbing tool-Y when the carriage stalls.
        self._plan_drives_rail: bool = False
        # Industrial MoveJ: integrate joint plan (+fb) with safety boxes only —
        # skip Cartesian ProxQP equality (near-σ that path freezes the GIL).
        self._direct_joint_ptp: bool = False
        self._apply_rail_mode_side_effects()

    @property
    def rail_mode(self) -> RailMode:
        return self._rail_mode

    def set_plan_drives_rail(self, enabled: bool) -> None:
        """Pin rail to plan qdot_ff[0] (SRS move→D); clear on scan/hold exit."""
        self._plan_drives_rail = bool(enabled)

    def set_direct_joint_ptp(self, enabled: bool) -> None:
        """Enable joint-space PTP (no Cartesian ProxQP primary)."""
        self._direct_joint_ptp = bool(enabled)

    @property
    def configured_rail_mode(self) -> RailMode:
        """Rail mode as configured in yaml (immutable), NOT the live mode.

        cfg.rail.mode is mutated by _apply_rail_mode_side_effects() every mode
        switch, so after a LOCKED phase it no longer reflects the yaml intent.
        Phase presets that restore the configured behaviour (e.g. scan/track
        re-coupling after hold@D) must use this property — reading
        cfg.rail.mode kept the rail LOCKED for the whole scan whenever a hold
        phase ran first.
        """
        return self._configured_rail_mode

    @property
    def locked_style(self) -> LockedStyle:
        """Active LockedStyle (only meaningful when rail_mode == LOCKED)."""
        return self._locked_style

    @property
    def is_locked_hold(self) -> bool:
        return (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.HOLD
        )

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

    def set_rail_extension_active(self, active: bool) -> None:
        """Gate the preferred-extension / pose-attract rail task (COUPLED).

        On during move→D (pose_attract → q_target[0]) and Cartesian track/scan
        (reach → d_pref); off during hold (rail is LOCKED+HOLD anyway).
        """
        self._rail_ext_active = bool(active)

    def set_rail_extension_mode(self, mode: str) -> None:
        """Select ``reach`` (scan) or ``pose_attract`` (move→D)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_mode(mode)  # type: ignore[arg-type]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        """Soft-attract target for pose_attract mode (metres on the rail)."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_rail_pose_target(y_rail_m)

    def capture_rail_extension_ref(self) -> None:
        """Capture preferred rail extension from the current scan-entry posture."""
        if self.rail_ext_task is not None:
            self.rail_ext_task.capture_reference(self.q_cmd)

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset()
        self.safety.reset(self.q_cmd)
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)
        self.rail_task.reset(self.q_cmd)
        if self.rail_ext_task is not None:
            self.rail_ext_task.reset(self.q_cmd)
        self._singularity_escape_seen = False
        self._centering_recovery_active = False
        self._apply_rail_mode_side_effects()

    def _centering_recovery_scale(
        self,
        q: np.ndarray,
        sigma_min: float,
        sigma_escape_ref: float,
    ) -> float:
        """Latch strong posture recovery after leaving the singularity zone."""
        if sigma_escape_ref > 1e-9 and sigma_min < sigma_escape_ref:
            self._singularity_escape_seen = True
            self._centering_recovery_active = False
            return 1.0

        if not self._singularity_escape_seen:
            self._centering_recovery_active = False
            return 1.0

        target_error = np.abs(
            (np.asarray(q, dtype=float) - self.centering_task.q_target)
            / self.centering_task.half
        )
        weighted_error = float(np.max(target_error * self.centering_task.w))
        if weighted_error <= max(float(self.cfg.centering_recovery_tol), 0.0):
            self._singularity_escape_seen = False
            self._centering_recovery_active = False
            return 1.0

        self._centering_recovery_active = True
        return max(float(self.cfg.centering_recovery_gain), 1.0)

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
        """Set rail top-level mode + (optionally) locked substyle.

        - ``COUPLED``: rail is a normal QP joint.  ``locked_style`` is ignored.
        - ``LOCKED``:  set ``locked_style`` to HOLD (hold position), RAIL_ONLY (plan drives
          rail, arm frozen) or TCP_FIXED (plan drives rail, arm compensates TCP).
        """
        if isinstance(mode, str):
            mode = RailMode(mode)
        self._rail_mode = mode
        if locked_style is not None:
            if isinstance(locked_style, str):
                locked_style = LockedStyle(locked_style)
            self._locked_style = locked_style
        if q_ref_m is not None:
            self.rail_task.set_reference(q_ref_m)
        elif mode == RailMode.LOCKED and self._locked_style == LockedStyle.HOLD:
            # HOLD without explicit ref = pin at current command (never yaml 0.0).
            self.rail_task.set_reference(float(self.q_cmd[0]))
        self._apply_rail_mode_side_effects()

    def set_coupled(self) -> None:
        """Convenience: switch to RailMode.COUPLED (rail participates in QP)."""
        self.set_rail_mode(RailMode.COUPLED)

    def set_locked(
        self,
        style: LockedStyle | str = LockedStyle.HOLD,
        *,
        q_ref_m: float | None = None,
    ) -> None:
        """Convenience: switch to RailMode.LOCKED with a specific style."""
        self.set_rail_mode(RailMode.LOCKED, q_ref_m=q_ref_m, locked_style=style)

    def _apply_rail_mode_side_effects(self) -> None:
        # Push the resolved (mode, style) into the RailLockTask config so
        # ``rail_task.active`` reflects the composed state (HOLD-only truth).
        self.rail_task.cfg.mode = self._rail_mode
        self.rail_task.cfg.locked_style = self._locked_style

    def _pin_rail_if_locked_hold(self) -> None:
        """Freeze rail_y in the 8-DOF command when LOCKED+HOLD.

        Only HOLD pins the rail position; RAIL_ONLY / TCP_FIXED explicitly
        drive it via qdot_ff; COUPLED lets the QP resolve it.
        """
        if not self.is_locked_hold or not self.cfg.rail.lock_hard_pin:
            return
        if self.rail_task.q_ref is None:
            return
        self.q_cmd[0] = float(self.rail_task.q_ref)
        self.core.qdot_prev[0] = 0.0

    def _twist_to_base(self, twist: np.ndarray, q_for_rot: np.ndarray) -> np.ndarray:
        twist = np.asarray(twist, dtype=float)
        if self.cfg.control_frame != "tool":
            return twist
        R = self.kin.fk_placement(q_for_rot).rotation
        out = np.zeros(6, dtype=float)
        out[:3] = R @ twist[:3]
        out[3:6] = R @ twist[3:6]
        return out

    def _secondary(
        self,
        q: np.ndarray,
        qdot_ff: np.ndarray | None,
        *,
        manipulability_active: bool | None = None,
        centering_sigma_fade: bool = True,
        sigma_min: float | None = None,
        centering_gain_scale: float = 1.0,
        max_qdot_frac_override: float | None = None,
    ) -> np.ndarray:
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            # Prefer this tick's σ when the caller already computed it: the
            # ∇μ fade is the fastest escape channel and a one-tick-stale σ
            # delays it by exactly the interval it is meant to win.
            sigma_min=(
                self.last_sigma_min if sigma_min is None else float(sigma_min)
            ),
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
            centering_gain_scale=centering_gain_scale,
            max_qdot_frac_override=max_qdot_frac_override,
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
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

        # Two-threshold singularity policy.  ``sigma_ref`` is the *brake*:
        # below it the Cartesian (incl. force) twist is attenuated so the
        # slack QP stops force-hybrid from driving the elbow straight.
        # ``sigma_escape_ref`` (default 2·σ_ref) is the *avoidance* onset, and
        # it must lead the brake — the rail is accel-limited to 0.3 m/s² and
        # needs ~0.17 s of lead to reach a useful escape speed, so an escape
        # that starts at the same σ as the brake always arrives after the arm
        # has already gone stiff.  Raising σ_ref itself instead was tried and
        # regressed (braking on 100 % of ticks, σ hovers ~0.08-0.12 at D).
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        sigma_escape_ref = sigma_ref * float(
            getattr(self.cfg.qp, "sigma_escape_ref_scale", 2.0)
        )
        J_pre = self.kin.jacobian(q_prev)
        sigma_pre = float(self.kin.singular_values(J_pre).min())
        centering_gain_scale = self._centering_recovery_scale(
            q_prev, sigma_pre, sigma_escape_ref
        )
        if sigma_ref > 1e-9 and sigma_pre < sigma_ref:
            floor = float(getattr(self.cfg.qp, "twist_sigma_floor", 0.08))
            twist_scale = max(float(sigma_pre / sigma_ref), floor)
            # Below half σ_ref, square the scale so force retract cannot
            # keep collapsing posture while σ→0.
            if sigma_pre < 0.5 * sigma_ref:
                twist_scale = max(twist_scale * twist_scale, 0.5 * floor)
            twist_base = twist_base * twist_scale

        # Rail mode dispatch (top-level + substyle)
        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        # (A) Command-magnitude safety: the joint feedforward is
        # ``dq_plan + k·(q_plan − q_cmd)``.  The anchor term is unbounded, and on
        # the prismatic rail it drove 0.64 m/s commands into a 0.10 m/s joint
        # (hardware log: rail cmd ran 6.4× v_max → 900 rpm → Er-01 overspeed).
        # Clamp EVERY feedforward channel to the same v_max the QP box and the
        # safety layer already enforce, so no plan/anchor can ever request a
        # velocity the hardware cannot execute — an IK "go to D" now approaches
        # at the joint speed limit instead of a lurch.
        if qdot_ff is not None:
            v_lim_ff = np.asarray(self.safety.lim.v_max, dtype=float)
            qdot_ff = np.clip(np.asarray(qdot_ff, dtype=float), -v_lim_ff, v_lim_ff)

        # Industrial MoveJ: integrate joint plan (+fb); skip Cartesian ProxQP.
        if self._direct_joint_ptp and qdot_ff is not None:
            qdot_cmd = np.asarray(qdot_ff, dtype=float).copy()
            q_next = q_prev + qdot_cmd * dt
            rep = self.safety.clamp(q_prev, q_next, dt)
            self.q_cmd = rep.q_safe
            if dt > 1e-9:
                self.core.qdot_prev = (self.q_cmd - q_prev) / dt
            else:
                self.core.qdot_prev = qdot_cmd
            if q_meas is not None:
                lead_max = float(self.cfg.resync_err_rail_m)
                if lead_max > 0.0:
                    q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                    q0_cmd = float(self.q_cmd[0])
                    if q0_cmd > q0_meas + lead_max:
                        self.q_cmd[0] = q0_meas + lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                    elif q0_cmd < q0_meas - lead_max:
                        self.q_cmd[0] = q0_meas - lead_max
                        if dt > 1e-9:
                            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            J = self.kin.jacobian(q_prev)
            sigma = self.kin.singular_values(J)
            sigma_min = float(sigma.min())
            self.last_sigma_min = sigma_min
            qdot_out = self.core.qdot_prev.copy()
            return JointIkStep(
                q_send=self.q_cmd.copy(),
                qdot=qdot_out,
                twist_base=twist_base,
                sigma_min=sigma_min,
                manip=float(np.prod(sigma)),
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err_rad=follow_err,
                qdot_ff_norm=float(np.linalg.norm(qdot_ff)),
                arm_singularity_smooth=1.0,
                limit_activation=0.0,
                vel_clamped=rep.vel_clamped,
                acc_clamped=rep.acc_clamped,
                pos_clamped=rep.pos_clamped,
                rail_ext_err_m=0.0,
                rail_ext_weight=0.0,
                rail_vel_pin=float(qdot_ff[0]),
                rail_qdot_ff=float(qdot_ff[0]),
                plan_drives_rail=True,
            )

        # (B) Pin the rail velocity ONLY when the rail is LOCKED (RAIL_ONLY /
        # TCP_FIXED), or when an SRS move explicitly requests plan ownership
        # (``set_plan_drives_rail(True)``).  In free COUPLED scan the rail is a
        # normal QP joint — the plan's rail intent already rides the primary
        # twist (J·qdot_cmd), so the QP freely allocates tool-Y across rail +
        # arm.  The old rule pinned the rail whenever a qdot_ff was present,
        # silently overriding set_coupled() AND bypassing v_max via the QP box.
        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)

        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        rail_qdot_ff_val = float("nan")
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            rail_qdot_ff_val = v_rail
            # Secondary tasks (centering / arm-angle / manipulability) act on the
            # arm portion only; the rail is either pinned (LOCKED) or freely
            # allocated by the QP (COUPLED).
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            if plan_drives_rail:
                rail_vel_pin = v_rail

        # Vectorized command-lead anti-windup: arm rad, rail m (units matter).
        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        # Preferred-extension rail coordination (COUPLED only): the rail
        # proactively follows the TCP when the arm reaches past its
        # comfortable extension — early, smooth singularity avoidance
        # instead of reactive last-moment recruitment.
        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        # Once sigma has recovered from an escape, posture recovery takes the
        # nullspace slot even if a move preset left the manipulability flag on.
        manip_for_saturation = (
            self._manipulability_active and not self._centering_recovery_active
        )
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_pre)
            # Two σ-health scalars, both 1.0 when healthy and → 0 at σ→0:
            #   sig_scale  vs σ_ref        — fades the scan feedforward, i.e.
            #                                "stop scanning, we are in trouble"
            #   sig_escape vs σ_escape_ref — drives the escape velocity, the
            #                                w_sigma_floor baseline and the
            #                                w-boost, i.e. "start getting out"
            # Keeping them separate is what makes avoidance proactive without
            # also throttling a healthy scan.  The escape scalar is NOT floored
            # at 0.25 any more: that capped the rail's authority at 75 % of
            # k_esc / 2.5x of w_max exactly at σ→0, where the invariant
            # w_max·(1 + k_sigma_boost) = 6 ≪ W_task = 100 already guarantees
            # the QP preference order slack > rail > free-arm.
            sig_scale = 1.0
            if sigma_ref > 1e-9 and sigma_now < sigma_ref:
                sig_scale = max(sigma_now / sigma_ref, 0.25)
            sig_escape = 1.0
            if sigma_escape_ref > 1e-9 and sigma_now < sigma_escape_ref:
                sig_escape = max(sigma_now / sigma_escape_ref, 0.0)
            # Bug 2: refresh the σ-escape / guardrail gradient every
            # ``_sigma_grad_period`` ticks via the pluggable RailGoodness
            # (default σ_min).  Passing 0 means the σ-escape v-component
            # collapses to the reach/pose term (safe fallback).
            self._sigma_grad_tick += 1
            if (
                self._sigma_grad_tick % self._sigma_grad_period == 0
                or self._sigma_grad_tick == 1
            ):
                _g, self._sigma_grad_rail_cached = self._rail_goodness.refresh(
                    q_prev, force=True
                )
                del _g
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_escape_scale=sig_escape,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            # Escape arm singularities in the nullspace whenever σ is
            # depressed — not only when the rail hits a travel stop.  Force
            # retract with ext_err≈0 still collapsed the elbow while rail
            # recruitment alone was too weak (hardware: σ→0, 4.7 s freeze).
            # Uses the escape threshold: ∇μ ascent is the one channel that can
            # act instantly, so it must be armed before the brake bites.
            if sigma_escape_ref > 1e-9 and sigma_now < sigma_escape_ref:
                manip_for_saturation = True

        r = self.core.step(
            q_prev,
            twist_base,
            dt,
            secondary_qdot=self._secondary(
                q_prev,
                qdot_ff_sec,
                manipulability_active=manip_for_saturation,
                centering_sigma_fade=not (
                    self._rail_ext_active and self._rail_mode == RailMode.COUPLED
                ),
                sigma_min=sigma_pre,
                centering_gain_scale=centering_gain_scale,
                max_qdot_frac_override=(
                    self.cfg.centering_recovery_max_qdot_frac
                    if self._centering_recovery_active
                    else None
                ),
            ),
            q_meas=q_meas,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight,
        )

        rep = self.safety.clamp(q_prev, r.q_next, dt)
        self.q_cmd = rep.q_safe
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            self.core.qdot_prev = rep.dq / dt
        # Hard command-lead cap vs encoder (belt-and-suspenders after the
        # safety margin-teleport bug).  Rail may not run more than
        # resync_err_rail_m ahead of the measured carriage — otherwise the
        # motor at 0.15 m/s is chasing a 1 m/s phantom and the governor dies.
        if q_meas is not None:
            lead_max = float(self.cfg.resync_err_rail_m)
            if lead_max > 0.0:
                q0_meas = float(np.asarray(q_meas, dtype=float)[0])
                q0_cmd = float(self.q_cmd[0])
                if q0_cmd > q0_meas + lead_max:
                    self.q_cmd[0] = q0_meas + lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
                elif q0_cmd < q0_meas - lead_max:
                    self.q_cmd[0] = q0_meas - lead_max
                    if dt > 1e-9:
                        self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
        # Plan-owned rail (SRS move→D / RAIL_ONLY): integrate q_cmd[0] from
        # qdot_ff.  Relying on the QP box pin alone was NOT enough — near-zero
        # pins lost to Cartesian slack and pose_attract, so q_cmd raced to
        # ~20 mm then plan_anchor yanked it back → soft-PD hunting at start.
        if plan_drives_rail and qdot_ff is not None and dt > 1e-9:
            v_rail = float(np.asarray(qdot_ff)[0])
            y = float(q_prev[0] + v_rail * dt)
            y_lo = float(self.limits.q_lower[0])
            y_hi = float(self.limits.q_upper[0])
            self.q_cmd[0] = float(np.clip(y, y_lo, y_hi))
            self.core.qdot_prev[0] = (self.q_cmd[0] - q_prev[0]) / dt
            if rail_only:
                self.q_cmd[1:] = q_prev[1:]
                self.core.qdot_prev[1:] = 0.0
        else:
            self._pin_rail_if_locked_hold()
        qdot_out = r.qdot.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0
        self.last_sigma_min = r.sigma_min
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=qdot_out,
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
            rail_ext_err_m=rail_ext_err,
            rail_ext_weight=rail_task_weight,
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=rail_qdot_ff_val,
            plan_drives_rail=bool(plan_drives_rail),
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
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
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
        dt_actual: float | None = None,
        v_tcp_z_actual: float | None = None,
        sensor_age_s: float | None = None,
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
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        return self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            v_tcp_z_actual=v_tcp_z_actual,
            sensor_age_s=sensor_age_s,
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
        self.last_vel_ff: np.ndarray | None = None

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def set_time_scale(self, scale: float) -> None:
        """Governor scale (0..1): scale trajectory vel_ff only, not the PD term."""
        self.time_scale = float(np.clip(scale, 0.0, 1.0))

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self.reference.sample(t_s)
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
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
    # σ-adaptive P-gain floor: k_eff = k_joint * max(σ/σ_ref, floor).  The old
    # 0.2 floor (k_eff → 0.4, τ = 2.5s) let the joint error grow to >12° when
    # a long move dipped through σ_min ≈ 0.03; on singular exit the P gain
    # snapped back and discharged that error as a TCP overshoot.  0.5 keeps
    # τ ≤ 1s through the dip — combined with ``k_joint_rise_per_s`` (rise-
    # only slew), the accumulated error decays smoothly instead of stepping.
    # Safe: k_eff acts on q_err only (not v_cmd), so a moderately higher
    # floor does NOT amplify pseudo-inverse tension at low σ.
    # σ-adaptive k_eff floor.  Debug H14: at σ∈[0.02,0.04] with floor=0.5,
    # k_eff stayed at 1.0 while q_err clipped at 20° → v_cmd infeasible →
    # slack≈0.2 with qp_fail=0 (solver converged but task was soft).  0.2
    # lets k_eff track σ/σ_ref continuously below σ≈0.016.
    k_joint_sigma_min_frac: float = 0.2
    control_frame: str = "tool"
    euler_order: str = "xyz"
    # Slew-rate limit on the σ-adaptive P gain (per second).  When the arm
    # crosses through a near-singular region during a long move, k_eff drops
    # to floor (k_joint * k_joint_sigma_min_frac) so tracking lags by up to
    # governor_joint_err_max_deg.  Without a rate limit, exiting the singular
    # region snaps k_eff back to k_joint in ONE tick and the accumulated
    # joint error is discharged as a Cartesian velocity spike → visible TCP
    # overshoot + pull-back at the end of the move.  Limiting the *rise*
    # rate spreads that discharge over ~1s so the QP box (a_max) can absorb
    # it; the *fall* rate stays instantaneous so entering a singular region
    # still triggers immediate protection.
    k_joint_rise_per_s: float = 1.2
    # First-order LPF time-constant on last_qdot_fb (s).  Debug logs showed
    # tick-to-tick qn_norm swings of 20-30% and slack_norm spikes to 0.10
    # while fb_signs stayed stable — the jitter came from QP dual-variable
    # oscillation between two near-optimal solutions when secondary (plan_ff
    # + fb) reached the same scale as slack·W_task.  Smoothing fb over ~15ms
    # kills the ~20Hz component driving the QP into this bimodal regime.
    fb_lpf_tau_s: float = 0.015
    # Additional scaling on the fb secondary pull (0..1).  Full fb (α=1.0)
    # made secondary dominate the QP reg block and let SR-damping's
    # imprecise N leak into J-row → slack chatter.  A ~0.4 gain keeps
    # nullspace closure alive with the QP well-conditioned.
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
        # Joint-space feedback term k_eff · q_err (NO plan feedforward, NO
        # governor scaling).  The phase loop feeds this in addition to the
        # governor-scaled qdot_plan into the QP's secondary channel so q_err
        # components in the Jacobian nullspace also get driven to zero.
        # Feeding the full qdot_cmd = qdot_plan + k_eff·q_err was tried and
        # caused divergence: qdot_plan bypassing the governor scale meant the
        # QP drove qdot at 2x q_ref's actual wall-clock rate during a
        # governor-throttled pass, overshooting and never recovering in deep
        # singular regions.
        self.last_qdot_fb: np.ndarray | None = None
        self._qdot_fb_lpf: np.ndarray | None = None  # LPF state, unscaled
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
        # Rise-only slew limit on k_eff: dropping into a singular region is
        # immediate (protection); climbing out is rate-limited so the built-
        # up q_err releases smoothly instead of a one-tick TCP kick.
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
        # First-order LPF on the fb term fed to QP secondary (see cfg).
        if self._qdot_fb_lpf is None or cfg.fb_lpf_tau_s <= 0.0:
            self._qdot_fb_lpf = qdot_fb_raw.copy()
        else:
            alpha = dt_eff_lpf / (cfg.fb_lpf_tau_s + dt_eff_lpf)
            self._qdot_fb_lpf = self._qdot_fb_lpf + alpha * (qdot_fb_raw - self._qdot_fb_lpf)
        # Scale down secondary contribution to prevent it dominating the QP
        # reg block and inducing dual-variable oscillation (see cfg comment).
        # v_cmd = J·(qdot_plan + qdot_fb_raw) still carries full fb into
        # primary, so J-row-space correction is unchanged; only nullspace
        # pull is scaled.
        self.last_qdot_fb = self._qdot_fb_lpf * float(cfg.fb_secondary_gain)
        qdot_cmd = qdot_plan + qdot_fb_raw
        v_lim = np.asarray(self.v_max, dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -v_lim, v_lim)
        v_base = J @ qdot_cmd
        # Near-singular: soften primary twist to what J can support (H14).
        # Also soften when σ has recovered but q_err is still large (H15:
        # Run 1 slack=0.81 at σ≈0.09, q_err≈16° — k_eff slew had ramped up
        # while v_feas was already 1.0).
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


def _print_move_plan_summary(
    phase: Phase,
    *,
    inner: JointIkController,
    q_meas: np.ndarray,
    rail_bridge=None,
    verbose: bool = True,
) -> None:
    """One-line move→D plan summary at phase enter (debug; no control change)."""
    if not verbose:
        return
    label = str(phase.label or "")
    if not label.startswith("move"):
        return
    ref = getattr(phase.outer, "reference", None)
    if ref is None or not hasattr(ref, "sample_q"):
        return
    q0 = np.asarray(getattr(ref, "q_start", q_meas), dtype=float).reshape(-1)
    qT = np.asarray(getattr(ref, "q_target", q_meas), dtype=float).reshape(-1)
    dur = float(getattr(ref, "duration_s", 0.0) or 0.0)
    rail0 = float(q0[0]) if q0.size > 0 else 0.0
    railT = float(qT[0]) if qT.size > 0 else 0.0
    dq_rail = abs(railT - rail0)
    # Quintic smoothstep peak |qdot| = 1.875 · |dq| / T
    peak_v = (1.875 * dq_rail / dur) if dur > 1e-9 else float("nan")
    motor_vmax = 0.15
    if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
        try:
            motor_vmax = float(rail_bridge.config.vel_max_m_s)
        except Exception:
            pass
    arm_dq_deg = float(np.rad2deg(np.max(np.abs(wrap_joint_delta(q0, qT)[1:])))) if q0.size >= 8 else float("nan")
    try:
        J0 = inner.kin.jacobian(q_meas)
        sigma0 = float(inner.kin.singular_values(J0).min())
    except Exception:
        sigma0 = float("nan")
    mode = "cartesian"
    if type(phase.outer).__name__ == "JointTrackOuterLoop":
        mode = "joint"
    elif type(phase.outer).__name__ == "CartesianTrackOuterLoop":
        mode = "cartesian"
    over = " OVER_MOTOR" if (np.isfinite(peak_v) and peak_v > motor_vmax + 1e-6) else ""
    y_attr = getattr(getattr(inner, "rail_ext_task", None), "y_rail_target_m", None)
    ext_mode = getattr(getattr(inner, "rail_ext_task", None), "mode", "?")
    print(
        f"  move plan: mode={mode} dur={dur:.2f}s | "
        f"rail {rail0 * 1000:.1f}→{railT * 1000:.1f} mm "
        f"peak_v={peak_v:.3f} m/s vs motor {motor_vmax:.2f} m/s{over} | "
        f"arm max|dq|={arm_dq_deg:.1f}deg sigma0={sigma0:.3f} | "
        f"COUPLED pose_attract→"
        f"{(float(y_attr) * 1000.0 if y_attr is not None else railT * 1000.0):.1f}mm "
        f"(mode={ext_mode}; σ guardrail only)",
        flush=True,
    )


def _print_tcp_frame_diagnose(
    inner: JointIkController,
    *,
    q_meas: np.ndarray,
    q_target: np.ndarray | None,
    phase_label: str,
    verbose: bool = True,
) -> None:
    """Read-only: gripper-TCP fk_pose vs link_7 vs (optional) q_target FK.

    Catches the ~220 mm flange-vs-gripper offset regression: if pose_d / scan
    origin was built on link_7 instead of the synced gripper TCP, the print
    shows a ~220 mm Z (or tool-Z) gap between fk_pose and frame_pose(link_7).
    """
    if not verbose:
        return
    label = str(phase_label or "").lower()
    if not (
        label.startswith("move")
        or "scan" in label
        or "hybrid" in label
    ):
        return
    q = np.asarray(q_meas, dtype=float).reshape(-1)
    try:
        pose_tcp = np.asarray(inner.kin.fk_pose(q), dtype=float).reshape(6)
        pose_l7 = np.asarray(inner.kin.frame_pose(q, "link_7"), dtype=float).reshape(6)
    except Exception as exc:
        print(f"  tcp diagnose: FK failed ({exc})", flush=True)
        return
    d_mm = (pose_tcp[:3] - pose_l7[:3]) * 1000.0
    off = getattr(inner.kin, "tcp_offset_pose", None)
    off_note = ""
    if off is not None:
        try:
            o = np.asarray(off, dtype=float).reshape(6)
            off_note = (
                f" | tool_offset xyz(mm)={np.round(o[:3] * 1000.0, 1).tolist()} "
                f"rpy(deg)={np.round(np.degrees(o[3:6]), 1).tolist()}"
            )
        except Exception:
            pass
    print(
        f"  tcp diagnose [{phase_label}]: "
        f"gripper-TCP xyz={np.round(pose_tcp[:3] * 1000.0, 1).tolist()} mm | "
        f"link_7 xyz={np.round(pose_l7[:3] * 1000.0, 1).tolist()} mm | "
        f"Δ(tcp-l7)={np.round(d_mm, 1).tolist()} mm "
        f"(|Δ|={float(np.linalg.norm(d_mm)):.1f} mm){off_note}",
        flush=True,
    )
    # Tool offset cache / sync sanity: |Δ| should be ~gripper Z (~220 mm), not ~0.
    if float(np.linalg.norm(d_mm)) < 5.0:
        print(
            "  tcp diagnose WARN: gripper-TCP ≈ link_7 — tool offset may be "
            "missing/unsynced (force-hybrid will look ~220 mm behind).",
            flush=True,
        )
    print(
        "  tcp diagnose: position loop uses kin.fk_pose (gripper TCP); "
        "force uses link_7 → wrench_link7_to_tcp (keep as-is).",
        flush=True,
    )
    if q_target is not None:
        qt = np.asarray(q_target, dtype=float).reshape(-1)
        if qt.size == q.size:
            try:
                pose_d = np.asarray(inner.kin.fk_pose(qt), dtype=float).reshape(6)
                print(
                    f"  tcp diagnose: pose_d=fk_pose(q_target) "
                    f"xyz={np.round(pose_d[:3] * 1000.0, 1).tolist()} mm "
                    f"(should be gripper TCP, not flange)",
                    flush=True,
                )
            except Exception:
                pass


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
    stop_reason: str = ""


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
    # Soft-start ramp on governor scale at phase entry (seconds).  Kill tick-0
    # speed spikes when a large Cartesian / joint plan error is present.
    soft_start_ramp_s: float = 0.0
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin
    on_exit: object | None = None            # Callable[[], None], fired when phase completes
    on_tick: object | None = None            # Callable[[float, JointIkStep, np.ndarray], None]


class _TickLogger:
    """Per-tick CSV telemetry (q_cmd/q_meas/twist/slack/clamp flags/force).

    Rows are queued and written on a background thread so disk I/O cannot stall
    the 200 Hz control loop (sync flush was a common source of 10+ ms hitches).
    """

    _HEADER = (
        ["t_wall_s", "phase", "controller_mode", "t_ref_s"]
        + [f"q_cmd_{i}" for i in range(0, 8)]
        + [f"q_meas_{i}" for i in range(0, 8)]
        + [f"pose_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        # ``twist_*`` is retained as a deprecated requested-twist alias for
        # existing analysis scripts.  The explicit columns remove the old
        # ambiguity: achieved twist is encoder J(q)qdot, not the QP request.
        + [f"twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_requested_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"twist_achieved_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + ["track_err_mm", "follow_err_deg", "slack_norm", "n_cbf",
           "vel_clamped", "acc_clamped", "pos_clamped", "fx", "fy", "fz",
           "instability_idx", "instability_idx_raw", "instability_idx_active",
           "damping_z_eff",
           "damping_ke_z", "damping_dimeas_z",
           "v_force_z", "ke_est",
           "f_des_z_eff", "v_r_z",
           "force_reference_scale_n", "force_reference_drive",
           "force_reference_gate_scale",
           "force_reference_accel_m_s2",
           "force_reference_reversal_reset",
           "mass_z_eff", "takeover",
           "dt_actual_s", "sensor_age_s",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_update_gated", "ke_dx_m", "ke_df_n", "ke_update_count",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "arm_singularity_smooth", "limit_activation",
           "tcp_jump_mm",
           "rail_ext_err_m", "rail_ext_w",
           "rail_target_sent_m", "rail_meas_m",
           "rail_vel_pin", "plan_drives_rail", "rail_qdot_ff"]
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
        rail_meas_m: float = float("nan"),
        dt_actual_s: float = float("nan"),
        sensor_age_s: float = float("nan"),
        f_ext_raw: np.ndarray | None = None,
        twist_achieved_base: np.ndarray | None = None,
        v_tcp_z_actual: float = float("nan"),
    ) -> None:
        qm = q_meas if q_meas is not None else np.full(8, np.nan)
        ctrl = getattr(outer, "controller", None)
        is_idx = getattr(ctrl, "instability_index", float("nan"))
        is_idx_raw = getattr(ctrl, "instability_index_raw", float("nan"))
        d_eff = getattr(ctrl, "damping_z_eff", float("nan"))
        d_ke = getattr(ctrl, "damping_ke_z", float("nan"))
        d_dimeas = getattr(ctrl, "damping_dimeas_z", float("nan"))
        v_fz = getattr(ctrl, "v_force_z", float("nan"))
        ke_est = getattr(ctrl, "ke_est", float("nan"))
        f_des_eff = getattr(ctrl, "f_des_z_eff", float("nan"))
        v_r_z = getattr(ctrl, "v_r_z", float("nan"))
        force_reference_scale = getattr(
            ctrl, "force_reference_scale_n", float("nan")
        )
        force_reference_drive = getattr(
            ctrl, "force_reference_drive", float("nan")
        )
        force_reference_gate = getattr(
            ctrl, "force_reference_gate_scale", float("nan")
        )
        force_reference_accel = getattr(
            ctrl, "force_reference_accel_m_s2", float("nan")
        )
        force_reference_reversal_reset = getattr(
            ctrl, "force_reference_reversal_reset", False
        )
        mass_z_eff = getattr(ctrl, "mass_z_eff", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        contact_present = getattr(ctrl, "contact_present", False)
        cap_press_z = getattr(ctrl, "cap_press_z", float("nan"))
        cap_retract_z = getattr(ctrl, "cap_retract_z", float("nan"))
        force_pred_z = getattr(ctrl, "force_pred_z", float("nan"))
        force_dot_z = getattr(ctrl, "force_dot_z", float("nan"))
        ke_tracker = getattr(ctrl, "_ke_estimator", None)
        ke_update_gated = getattr(ke_tracker, "update_gated", False)
        ke_dx_m = getattr(ke_tracker, "last_dx_m", float("nan"))
        ke_df_n = getattr(ke_tracker, "last_df_n", float("nan"))
        ke_update_count = getattr(ke_tracker, "update_count", 0)
        raw_comp = (
            np.asarray(f_ext_raw, dtype=float)
            if f_ext_raw is not None
            else np.full(6, np.nan)
        )
        twist_achieved = (
            np.asarray(twist_achieved_base, dtype=float)
            if twist_achieved_base is not None
            else np.full(6, np.nan)
        )
        qdot_norm = float(np.linalg.norm(step.qdot))
        # Fraction of the per-joint velocity box actually used (1.0 = saturated
        # on at least one joint) - the clearest signal for "CBF/limits are
        # strangling the commanded twist" vs "the twist itself is just small".
        if v_max is not None and np.any(v_max > 1e-9):
            qdot_max_frac = float(np.max(np.abs(step.qdot) / np.maximum(v_max, 1e-9)))
        else:
            qdot_max_frac = float("nan")
        rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        self._q.put(
            [
                f"{t_wall:.4f}",
                label,
                str(getattr(ctrl, "controller_mode", "none")),
                f"{t_ref:.4f}",
            ]
            + [f"{v:.6f}" for v in step.q_send]
            + [f"{v:.6f}" for v in qm]
            + [f"{v:.6f}" for v in pose]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in step.twist_base]
            + [f"{v:.5f}" for v in twist_achieved]
            + [f"{step.cart_err_mm:.3f}", f"{np.degrees(step.follow_err_rad):.4f}",
               f"{step.slack_norm:.5f}", step.n_cbf_active,
               int(step.vel_clamped), int(step.acc_clamped), int(step.pos_clamped),
               f"{f_ext[0]:.3f}", f"{f_ext[1]:.3f}", f"{f_ext[2]:.3f}",
               f"{is_idx:.4f}", f"{is_idx_raw:.4f}", f"{is_idx:.4f}",
               f"{d_eff:.2f}",
               f"{d_ke:.2f}", f"{d_dimeas:.2f}",
               f"{v_fz:.5f}", f"{ke_est:.1f}",
               f"{f_des_eff:.3f}", f"{v_r_z:.5f}",
               f"{force_reference_scale:.4f}",
               f"{force_reference_drive:.6f}",
               f"{force_reference_gate:.4f}",
               f"{force_reference_accel:.6f}",
               int(bool(force_reference_reversal_reset)),
               f"{mass_z_eff:.4f}",
               int(bool(takeover)),
               f"{dt_actual_s:.6f}", f"{sensor_age_s:.6f}",
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               int(bool(ke_update_gated)), f"{ke_dx_m:.8f}", f"{ke_df_n:.5f}",
               int(ke_update_count),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.arm_singularity_smooth:.4f}",
               f"{step.limit_activation:.4f}",
               f"{step.tcp_jump_mm:.3f}",
               f"{step.rail_ext_err_m:.5f}", f"{step.rail_ext_weight:.4f}",
               f"{rail_sent:.6f}",
               f"{rail_meas_m:.6f}" if np.isfinite(rail_meas_m) else "",
               f"{step.rail_vel_pin:.6f}" if np.isfinite(step.rail_vel_pin) else "",
               int(bool(step.plan_drives_rail)),
               f"{step.rail_qdot_ff:.6f}" if np.isfinite(step.rail_qdot_ff) else ""]
        )

    def close(self) -> None:
        self._q.put(None)
        self._stop.set()
        self._worker.join(timeout=1.0)


def _expand_q_meas(q_deg_or_rad: np.ndarray, rail_m: float) -> np.ndarray:
    """Realman feedback is 7 arm joints; prepend rail position for 8-DOF FK."""
    q = np.asarray(q_deg_or_rad, dtype=float)
    if q.size >= 8:
        return q[:8]
    if q.size == 7:
        return full_q_from_arm(q, rail_m)
    raise ValueError(f"expected 7 or 8 joint values, got {q.size}")


def _rail_m_for_init(rail_bridge, inner: JointIkController) -> float:
    """Seed WBC ``q_cmd[0]`` at task/phase start from encoder (measured).

    Use measured (encoder), not a stale plan value, so the first
    ``set_target_m`` is near the true carriage and the soft loop does not
    slam toward an old q_cmd.
    """
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.measured_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(rail_bridge, inner: JointIkController) -> float:
    """Rail component of ``q_meas`` inside the WBC tick: **encoder**, not ``q_cmd``.

    Cascade matches ``apps/lw100_vel_pos_follow_demo.py`` (manual §5.2 host
    soft-position / drive FA24 speed):

    * outer: WBC Cartesian / QP issues a rail *target* ``q_cmd[0]``;
    * inner: ``RailServoBridge`` soft PD closes ``target − encoder → FA24``
      (same kp/kd/ff as the tuned demo);
    * measurement: this helper returns the encoder so FK / tracking error /
      nullspace see the *real* carriage.  When the motor lags or reverses,
      the arm can compensate — the old ``q_meas[0]=q_cmd[0]`` open-loop lie
      made the controller "happy" while the viewer showed the rail hunting.

    Garbage / OOB encoder readings fall back to ``q_cmd[0]`` for one tick
    (never feed -1474 mm into FK).  No rail bridge → virtual rail = ``q_cmd``.
    """
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return float(inner.q_cmd[0])
    try:
        meas = float(rail_bridge.measured_m)
    except Exception:
        return float(inner.q_cmd[0])
    sane = getattr(rail_bridge, "_encoder_sane", None)
    if callable(sane):
        if not sane(meas):
            return float(inner.q_cmd[0])
    elif not (np.isfinite(meas)):
        return float(inner.q_cmd[0])
    return meas


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


def _send_joint_canfd_cmd(robot, q_deg, follow: bool, canfd_proxy=None) -> None:
    from rm75_control.motion.canfd import send_joint_canfd

    q = np.asarray(q_deg, dtype=float).reshape(-1)[:7]
    if canfd_proxy is not None:
        canfd_proxy.write(q, follow=follow)
        return
    if robot is None:
        raise RuntimeError("no robot handle and no CANFD proxy configured")
    send_joint_canfd(robot, list(q), follow=follow)


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
    state_bus=None,
    canfd_proxy=None,
    stop_check=None,
    rail_bridge=None,
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
    from rm75_control.control.admittance_common.state_bus import RobotStateBus

    dt = inner.cfg.dt if dt is None else dt
    robot = session.robot

    if q_start_deg is not None:
        if robot is None:
            raise RuntimeError("q_start_deg move_j requires a local robot session")
        session.move_joints(list(np.asarray(q_start_deg, dtype=float)), velocity_percent=move_speed, block=1)
        time.sleep(0.5)

    own_bus = state_bus is None
    if own_bus:
        state_bus = RobotStateBus(robot, session.config, robot_ip=session.ip)
        state_bus.start()
    async_obs = state_bus.observer
    if verbose and own_bus:
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
        q0_rad = _expand_q_meas(
            deg2rad(snap0.q_deg),
            _rail_m_for_init(rail_bridge, inner),
        )
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
                _send_joint_canfd_cmd(
                    robot,
                    rad2deg(arm_q_from_full(inner.q_cmd)),
                    False,
                    canfd_proxy,
                )
            except Exception:
                if robot is not None:
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
            phase_stopped = False
            stop_reason = ""
            try:
                for phase_idx, phase in enumerate(phases):
                    if stop_check is not None and stop_check():
                        phase_stopped = True
                        if verbose:
                            print("  stopped by external request", flush=True)
                        break
                    if verbose:
                        print(f"-- phase: {phase.label or phase.outer.__class__.__name__} --", flush=True)
                    # Phase origin from the ENCODERS, never from the command integrator.
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        # Soft-start reseed wants the *encoder* rail, not q_cmd[0].
                        rail_seed = _rail_m_for_init(rail_bridge, inner)
                        q_meas = _expand_q_meas(deg2rad(snap.q_deg), rail_seed)
                    pose_pin = inner.kin.fk_pose(q_meas)
                    # Soft-start: reseed plan start from live encoders so
                    # tick-0 Cartesian / joint error is ≈0 (no lurch), then ramp
                    # governor scale over soft_start_ramp_s.
                    ref = getattr(phase.outer, "reference", None)
                    if ref is not None:
                        try:
                            q_live = np.asarray(q_meas, dtype=float).reshape(-1)
                            if hasattr(ref, "reseed_start"):
                                ref.reseed_start(q_live)
                                if verbose and str(phase.label or "").startswith("move"):
                                    print(
                                        f"  soft-start: reseeded SRS start from encoders "
                                        f"(rail={q_live[0] * 1000:.1f} mm)",
                                        flush=True,
                                    )
                            elif hasattr(ref, "q_start") and hasattr(ref, "q_target"):
                                if q_live.size == int(np.asarray(ref.q_start).size):
                                    ref.q_start = q_live.copy()
                                    if verbose and str(phase.label or "").startswith("move"):
                                        print(
                                            f"  soft-start: reseeded plan q_start from encoders "
                                            f"(rail={q_live[0] * 1000:.1f} mm)",
                                            flush=True,
                                        )
                        except Exception:
                            pass
                    if hasattr(phase.outer, "set_origin"):
                        phase.outer.set_origin(pose_pin)
                    if phase.on_enter is not None:
                        phase.on_enter()
                    _print_move_plan_summary(
                        phase,
                        inner=inner,
                        q_meas=q_meas,
                        rail_bridge=rail_bridge,
                        verbose=verbose,
                    )
                    _print_tcp_frame_diagnose(
                        inner,
                        q_meas=q_meas,
                        q_target=getattr(getattr(phase.outer, "reference", None), "q_target", None),
                        phase_label=str(phase.label or ""),
                        verbose=verbose,
                    )

                    obs = phase.force_observer if phase.force_observer is not None else force_observer
                    phase_t0 = time.perf_counter()
                    next_tick = phase_t0
                    last_tick_time = phase_t0
                    t_ref = 0.0
                    gov_filter = GovernorFilter(
                        tau_s=phase.governor_tau_s,
                        freeze_below=phase.governor_freeze_below,
                        release_above=phase.governor_release_above,
                    )
                    scale = 1.0
                    phase_arrived = False
                    prev_pose_cmd = inner.kin.fk_pose(inner.q_cmd)
                    # Scan-phase debug: throttled state dump for tuning force-hybrid.
                    _is_scan = bool(phase.label) and (
                        "scan" in str(phase.label) or "hybrid" in str(phase.label)
                    )
                    _scan_log_t = 0.0
                    _scan_origin_pose = None
                    # Encoder-derived TCP velocity for diagnostics. Only
                    # update on a fresh UDP sequence; reusing a frame must not
                    # create a fake zero-velocity sample.
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    twist_achieved_base = np.zeros(6, dtype=float)
                    v_tcp_z_actual = 0.0
                    phase_ctrl = getattr(phase.outer, "controller", None)
                    if verbose and phase_ctrl is not None:
                        mode = str(
                            getattr(phase_ctrl, "controller_mode", "legacy_symmetric")
                        )
                        print(
                            f"  force controller: {mode} "
                            "(fixed-dt 2965fea+energy-aware tracking)",
                            flush=True,
                        )
                    while True:
                        if stop_check is not None and stop_check():
                            phase_stopped = True
                            break
                        now = time.perf_counter()
                        dt_raw = now - last_tick_time
                        last_tick_time = now
                        # The first phase tick occurs immediately after setup;
                        # use the nominal period rather than a near-zero dt.
                        if dt_raw < 0.002:
                            dt_raw = dt
                        dt_actual = float(np.clip(dt_raw, 0.002, 0.015))
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
                            q_new = _expand_q_meas(
                                deg2rad(snap.q_deg),
                                _rail_m_for_feedback(rail_bridge, inner),
                            )
                            snap_seq = int(getattr(snap, "seq", 0))
                            snap_t = float(getattr(snap, "t_s", 0.0))
                            if (
                                snap_seq != last_feedback_seq
                                and snap_t > last_feedback_t
                            ):
                                dt_feedback = snap_t - last_feedback_t
                                if 0.001 <= dt_feedback <= 0.050:
                                    qdot_meas = (
                                        wrap_joint_delta(last_feedback_q, q_new)
                                        / dt_feedback
                                    )
                                    twist_achieved_base = (
                                        inner.kin.jacobian(q_new) @ qdot_meas
                                    )
                                    pose_for_velocity = inner.kin.fk_pose(q_new)
                                    r_velocity = Rsc.from_euler(
                                        inner.cfg.euler_order,
                                        pose_for_velocity[3:6],
                                        degrees=False,
                                    ).as_matrix()
                                    v_tcp_z_actual = float(
                                        (r_velocity.T @ twist_achieved_base[:3])[2]
                                    )
                                last_feedback_seq = snap_seq
                                last_feedback_t = snap_t
                                last_feedback_q = q_new.copy()
                            q_meas = q_new
                            pose_pin = inner.kin.fk_pose(q_meas)

                        sensor_age_s = (
                            max(0.0, time.monotonic() - float(snap.t_s))
                            if float(getattr(snap, "t_s", 0.0)) > 0.0
                            else float("inf")
                        )

                        f_ext = np.zeros(6)
                        f_ext_raw = None
                        if obs is not None:
                            pose_l7 = inner.kin.frame_pose(q_meas, "link_7")
                            _signed, f_ext = obs.update(now - total_t0, pose_l7, snap.force_raw)
                            f_ext_raw = getattr(obs, "f_ext_raw_last", None)
                            f_ext = inner.kin.wrench_link7_to_tcp(f_ext)
                            if f_ext_raw is not None:
                                f_ext_raw = inner.kin.wrench_link7_to_tcp(f_ext_raw)
    
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
                        if "dt_actual" in sample_params:
                            sample_kwargs["dt_actual"] = dt_actual
                        if "v_tcp_z_actual" in sample_params:
                            sample_kwargs["v_tcp_z_actual"] = v_tcp_z_actual
                        if "sensor_age_s" in sample_params:
                            sample_kwargs["sensor_age_s"] = sensor_age_s
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
                        # Joint-space feedback k_eff·q_err from
                        # JointTrackOuterLoop: closes the arm's 8-DOF null on
                        # the joint target (qdot_ff plan-only sees just
                        # J-row-space corrections through v_cmd = J·qdot_cmd,
                        # so q_err in the Jacobian nullspace stalls at
                        # multi-degree residuals even after track_xy → 0).
                        # NOT governor-scaled — feedback throttled by
                        # governor would defeat the very tracking the
                        # governor is waiting for.  Kept as an ADDITIVE
                        # nullspace pull so centering / arm_angle / rail_lock
                        # / manipulability-ascent stay active (compose adds,
                        # then N projects — orthogonal components survive).
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        vel_ff_ref = getattr(phase.outer, "last_vel_ff", None)
                        # Keep the hardware-proven fixed timing law throughout
                        # controller, IK limits, governor and reference clock.
                        control_dt = dt
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                        )
                        if rail_bridge is not None:
                            rail_bridge.set_target_m(float(inner.q_cmd[0]))
                        # Throttled rail follow debug (move→D + scan) for C iteration.
                        if (
                            verbose
                            and rail_bridge is not None
                            and rail_bridge.enabled
                            and now - jump_warn_t >= 0.5
                        ):
                            jump_warn_t = now
                            print(
                                f"  rail follow tgt={inner.q_cmd[0]*1000:.1f} "
                                f"meas={rail_bridge.measured_m*1000:.1f} mm "
                                f"phase={phase.label} t_ref={t_ref:.2f}s",
                                flush=True,
                            )
                        outer_err_mm = getattr(phase.outer, "last_err_mm", None)
                        if outer_err_mm is not None:
                            step.cart_err_mm = outer_err_mm
                        pose_cmd = inner.kin.fk_pose(step.q_send)
                        step.tcp_jump_mm = float(
                            np.linalg.norm(pose_cmd[:3] - prev_pose_cmd[:3]) * 1000.0
                        )
                        if verbose and step.tcp_jump_mm > 8.0 and now - jump_warn_t >= 1.0:
                            jump_warn_t = now
                            print(
                                f"  warn: TCP jump {step.tcp_jump_mm:.1f}mm/tick",
                                flush=True,
                            )
                        prev_pose_cmd = pose_cmd
                        _send_joint_canfd_cmd(
                            robot,
                            rad2deg(arm_q_from_full(step.q_send)),
                            follow,
                            canfd_proxy,
                        )
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
                        scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6:
                            scale *= float(np.clip(t_wall / ramp_s, 0.0, 1.0))
                        t_ref += control_dt * scale
    
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
                            rail_meas = float("nan")
                            if rail_bridge is not None and rail_bridge.enabled:
                                try:
                                    rail_meas = float(rail_bridge.measured_m)
                                except Exception:
                                    rail_meas = float("nan")
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                                rail_meas_m=rail_meas,
                                dt_actual_s=dt_actual,
                                sensor_age_s=sensor_age_s,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                            )
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)

                        # Scan-phase debug log (throttled ~1 Hz): tool-Y sweep, rail, force.
                        if _is_scan and (t_wall - _scan_log_t) >= 1.0:
                            _scan_log_t = t_wall
                            if _scan_origin_pose is None:
                                _scan_origin_pose = pose_pin.copy()
                            dy_cmd_mm = float((pose_cmd[1] - _scan_origin_pose[1]) * 1000.0)
                            dy_meas_mm = float((pose_pin[1] - _scan_origin_pose[1]) * 1000.0)
                            rail_cmd_mm = float(inner.q_cmd[0] * 1000.0)
                            rail_meas_mm = (
                                float(rail_bridge.measured_m * 1000.0)
                                if rail_bridge is not None and rail_bridge.enabled
                                else rail_cmd_mm
                            )
                            fz = float(f_ext[2])
                            print(
                                f"  [scan t={t_ref:5.1f}s] toolY cmd={dy_cmd_mm:+7.1f} "
                                f"meas={dy_meas_mm:+7.1f} mm | rail cmd={rail_cmd_mm:6.1f} "
                                f"meas={rail_meas_mm:6.1f} mm | Fz={fz:+5.2f}N "
                                f"| track={step.cart_err_mm:5.1f}mm gov={scale:.2f} "
                                f"σ={step.sigma_min:.3f}",
                                flush=True,
                            )

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

                    if phase.on_exit is not None:
                        phase.on_exit()

                    if phase_stopped:
                        break

                    if phase.require_arrival and not phase_arrived:
                        err_mm = getattr(phase.outer, "last_err_mm", float("nan"))
                        jq = getattr(phase.outer, "last_joint_err_deg", float("nan"))
                        d_mm = d_deg = float("nan")
                        try:
                            pt = getattr(phase, "pose_target", None)
                            if pt is None:
                                ref = getattr(phase.outer, "reference", None)
                                pt = getattr(ref, "pose_d", None) or getattr(ref, "pose_target", None)
                            if pt is not None and q_meas is not None:
                                d_mm, d_deg = pose_distance(
                                    pose_pin, pt, inner.cfg.euler_order
                                )
                        except Exception:
                            pass
                        print(
                            f"  ERROR: phase {phase.label!r} did not reach target "
                            f"(t_ref={t_ref:.2f}s, wall={t_wall:.1f}s, "
                            f"track={err_mm:.0f}mm, poseΔ={d_mm:.1f}mm/{d_deg:.1f}deg, "
                            f"jq={jq:.1f}deg) "
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
        if own_bus:
            state_bus.stop()
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
        stop_reason=stop_reason,
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
    state_bus=None,
    rail_bridge=None,
) -> LoopResult:
    """Single-phase convenience wrapper around ``run_joint_admittance_phases``."""
    phase = Phase(
        outer=outer,
        label="run",
        duration_s=duration_s,
    )
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
        state_bus=state_bus,
        rail_bridge=rail_bridge,
    )

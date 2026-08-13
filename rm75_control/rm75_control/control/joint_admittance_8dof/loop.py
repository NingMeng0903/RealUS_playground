"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

``JointIkController``: hardware-free WBC QP IK + safety clamp (no send-path LPF).
``run_joint_admittance_phases``: on-robot orchestration closing on FK(q_meas).
"""

from __future__ import annotations

import csv
import inspect
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation as Rsc

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_runtime import (
    GenericQpikRuntime,
    GenericQpikRuntimeConfig,
)
from rm75_control.control.joint_admittance_8dof.solver.constraint_mgr import (
    VelocityBoxInfeasible,
)
from rm75_control.control.joint_admittance_8dof.solver.p0_safety import (
    CollisionHardCapacityExceeded,
)
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import (
    HardConstraintCapacityExceeded,
)
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    TaskSpaceConstraintRow,
)
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
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
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
    generic_qpik: GenericQpikRuntimeConfig = field(
        default_factory=GenericQpikRuntimeConfig
    )
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    limit_damper_band_rad: float = 0.15
    limit_damper_band_rail_m: float = 0.05
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    v_scale: float = 0.5               # fraction of URDF joint velocity limit allowed
    # Accel limits are unit-separated: rail m/s^2, arm rad/s^2.
    a_max_arm_rad_s2: float = 20.0     # rad/s^2 per arm joint (1..7)
    a_max_rail_m_s2: float = 0.30      # m/s^2 for prismatic rail (0)
    position_margin_rad: float = 0.017
    position_margin_rail_m: float = 0.0  # metres (do not reuse arm rad margin)
    # QP velocity bound: stop q_cmd leading q_meas (0 disables; never a teleport).
    resync_err_rad: float = 0.10       # arm joints 1..7 (radians)
    resync_err_rail_m: float = 0.020   # rail joint 0 (metres; 20 mm)
    # Cartesian geometry/CBF must never run on an indefinitely retained UDP
    # sample.  This is a transport-age limit, not a requirement that every
    # 200 Hz tick receive a new packet.
    feedback_timeout_s: float = 0.050


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
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False
    tcp_jump_mm: float = 0.0
    rail_vel_pin: float = float("nan")      # m/s hard pin, or NaN if free
    rail_qdot_ff: float = float("nan")      # plan qdot_ff[0] before strip
    plan_drives_rail: bool = False
    controller_mode: str = "single_qpik"
    qp_backend: str = ""
    qp_solver_status: str = "not_run"
    qp_solver_iterations: int = 0
    qp_solver_solve_ms: float = 0.0
    qp_solver_call_count: int = 0
    qp_solver_overrun: bool = False
    qpik_alpha: float = 1.0
    qpik_beta: float = 1.0
    qpik_authority: float = 1.0
    qpik_equality_residual_max: float = float("nan")
    qpik_hard_residual_max: float = float("nan")
    qpik_anchor_valid: bool = True
    qpik_recovery_overflow: bool = False
    qpik_protected_nominal_overflow: np.ndarray = field(default_factory=lambda: np.zeros(4))
    qpik_recovery_caps: np.ndarray = field(default_factory=lambda: np.zeros(14))
    qpik_recovery_overflow_indices: tuple[int, ...] = ()
    qpik_working_slack: np.ndarray = field(default_factory=lambda: np.zeros(8))
    qpik_collision_slack: np.ndarray = field(default_factory=lambda: np.zeros(4))
    qpik_dexterity_slack: float = 0.0
    qpik_branch_slack: float = 0.0
    rail_macro_pref_v: float = 0.0
    rail_center_pref_v: float = 0.0
    arm_risk_pref_norm: float = 0.0
    arm_risk_pref: np.ndarray = field(default_factory=lambda: np.zeros(8))
    risk_direction_cosine: float = float("nan")
    path_velocity_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feedback_xy_raw: np.ndarray = field(default_factory=lambda: np.zeros(2))
    feedback_xy_filtered: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rail_xy_contribution: np.ndarray = field(default_factory=lambda: np.zeros(2))
    arm_xy_contribution: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rail_task_projection: float = float("nan")
    rail_arm_cancel: float = float("nan")
    rail_decomposition_error: float = float("nan")
    wrist_singularity: float = float("nan")
    hard_active_constraint_ids: tuple[str, ...] = ()
    protected_target: np.ndarray = field(default_factory=lambda: np.zeros(0))
    protected_achieved: np.ndarray = field(default_factory=lambda: np.zeros(0))
    protected_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scan_target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scan_achieved: np.ndarray = field(default_factory=lambda: np.zeros(2))
    scan_residual: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fallback_level: str = "none"
    fallback_reason: str = ""
    solver_fault_latched: bool = False
    health_state: str = "NORMAL"
    arm_health: float = float("nan")
    joint_margin_rad: float = float("nan")
    wrist_margin_rad: float = float("nan")
    accepted_reference_lag_s: float = 0.0
    pre_solve_feedback_age_s: float = float("nan")
    post_solve_feedback_age_s: float = float("nan")


class JointIkController:
    """Reusable inner loop: (q_cmd, q_meas, twist) -> next joint command (rad)."""

    def __init__(self, kin: RobotKinematics, cfg: JointIkConfig | None = None) -> None:
        self.kin = kin
        self.cfg = cfg or JointIkConfig()
        if (
            not np.isfinite(float(self.cfg.feedback_timeout_s))
            or float(self.cfg.feedback_timeout_s) <= 0.0
        ):
            raise ValueError("feedback_timeout_s must be finite and > 0")
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
            # The rail has its own absolute cap. Keep the arm-wide v_scale on
            # joints 1..7, while still respecting the URDF rail limit.
            self.limits.v_max[0] = min(
                float(self.kin.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
            )
        # Canonical host soft rail band.  SafetyLimiter and the QP share this
        # same SafetyLimits object, so both command clamps and velocity boxes
        # stop at 1--78 cm while the kinematic model retains mechanical travel.
        soft_lo = float(getattr(self.cfg.rail, "soft_min_m", 0.01))
        soft_hi = float(getattr(self.cfg.rail, "soft_max_m", 0.78))
        if not (
            np.isfinite(soft_lo)
            and np.isfinite(soft_hi)
            and float(self.kin.q_lower[0]) <= soft_lo < soft_hi
            and soft_hi <= float(self.kin.q_upper[0])
        ):
            raise ValueError(
                "invalid canonical rail soft limits: "
                f"[{soft_lo:.6f}, {soft_hi:.6f}]"
            )
        self.limits.q_lower[0] = max(float(self.limits.q_lower[0]), soft_lo)
        self.limits.q_upper[0] = min(float(self.limits.q_upper[0]), soft_hi)
        # Expose the resolved band on the common safety object for telemetry.
        self.limits.rail_soft_min_m = float(self.limits.q_lower[0])
        self.limits.rail_soft_max_m = float(self.limits.q_upper[0])
        # The production Cartesian path is the fixed single-shot QPIK.
        # Reuse dbb's collision and velocity-damper configuration in the
        # common measured-state P0 layer; no legacy weighted/nullspace solver
        # participates in a Cartesian tick.
        damper_band = np.full(
            kin.nv, float(self.cfg.limit_damper_band_rad), dtype=float
        )
        damper_band[0] = float(self.cfg.limit_damper_band_rail_m)
        self.core = GenericQpikRuntime(
            self.kin,
            self.limits,
            self.cfg.generic_qpik,
            collision_config=self.cfg.collision,
            damper_band=damper_band,
        )
        self.safety = SafetyLimiter(self.limits)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self.last_sigma_min: float = float("nan")
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        # Immutable yaml rail mode (live cfg.rail.mode is mutated by locks).
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        # When True, plan owns rail velocity via qdot_ff pin.
        self._plan_drives_rail: bool = False
        # Direct joint PTP: integrate plan (+fb); skip Cartesian ProxQP.
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
        """Yaml rail mode (immutable); live cfg.rail.mode is mutated by locks."""
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

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset(self.q_cmd)
        self.safety.reset(self.q_cmd)
        # Direct joint/rail ownership is phase-scoped.  An exception that
        # skipped the previous phase's on_exit must never leak a P0-bypassing
        # mode into the next task.
        self._direct_joint_ptp = False
        self._plan_drives_rail = False
        self._comfort_arm_q = np.asarray(q0_rad, dtype=float)[1:].copy()
        self._apply_rail_mode_side_effects()

    def begin_hybrid_episode(
        self,
        q_meas: np.ndarray,
        qdot_applied: np.ndarray | None = None,
    ) -> None:
        """Start a Cartesian episode from the actually applied joint velocity."""

        applied = self.core.qdot_prev if qdot_applied is None else qdot_applied
        self.core.begin_hybrid_episode(q_meas, applied)

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
        """Set rail mode (COUPLED / LOCKED) and optional locked_style."""
        if isinstance(mode, str):
            mode = RailMode(mode)
        self._rail_mode = mode
        if locked_style is not None:
            if isinstance(locked_style, str):
                locked_style = LockedStyle(locked_style)
            self._locked_style = locked_style
        if q_ref_m is not None:
            if (
                mode == RailMode.LOCKED
                and self._locked_style == LockedStyle.HOLD
                and abs(float(q_ref_m) - float(self.q_cmd[0])) > 1.0e-9
            ):
                raise ValueError(
                    "locked HOLD cannot move rail to a different reference; "
                    "use a continuous RAIL_ONLY/TCP_FIXED phase first"
                )
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
        self.cfg.rail.mode = self._rail_mode
        self.cfg.rail.locked_style = self._locked_style

    def _finish_single_qpik_result(
        self,
        result,
        *,
        q_prev: np.ndarray,
        period: float,
        twist_base: np.ndarray,
        sigma_min: float,
        manip: float,
        follow_err: float,
        qdot_ff_norm: float = 0.0,
        rail_vel_pin: float | None = None,
        rail_qdot_ff: float = float("nan"),
        plan_drives_rail: bool = False,
    ) -> JointIkStep:
        """Validate in physical units and preserve a certified anchor fallback."""

        applied = np.asarray(result.qdot, dtype=float).reshape(-1).copy()
        tolerance = max(
            float(self.cfg.generic_qpik.solver.feasibility_tolerance),
            np.finfo(float).eps,
        )
        integration_fault = ""
        violated: list[str] = []
        if applied.shape != q_prev.shape or not np.all(np.isfinite(applied)):
            integration_fault = "final_qdot_nonfinite_or_bad_shape"
        else:
            values = result.p0.C @ applied
            bad = np.flatnonzero(
                (values < result.p0.lower - tolerance)
                | (values > result.p0.upper + tolerance)
            )
            if bad.size:
                integration_fault = "final_qdot_violates_hard_constraints"
                violated.extend(result.p0.names[int(index)] for index in bad)
            q_candidate = q_prev + applied * period
            margin = np.asarray(self.limits.position_margin, dtype=float)
            if np.any(q_candidate < self.limits.q_lower + margin - tolerance) or np.any(
                q_candidate > self.limits.q_upper - margin + tolerance
            ):
                integration_fault = "final_command_violates_position_band"

        hard_failure = bool(result.solver.hard_failure or integration_fault)
        if hard_failure:
            # This velocity is never published; keep the software command at
            # the last sent value so a later operator reset has no hidden lead.
            applied = np.zeros_like(q_prev)
            self.q_cmd = q_prev.copy()
        else:
            self.q_cmd = q_prev + applied * period
            self.safety.sync_applied_delta(applied * period, period)
            self.core.sync_applied(applied)
        self.last_sigma_min = sigma_min

        protected_achieved = result.protected_jacobian @ applied
        protected_residual = (
            protected_achieved - np.asarray(result.protected_target, dtype=float)
        )
        solver = result.solver
        diagnostics = solver.diagnostics
        alpha = 0.0 if hard_failure else float(solver.alpha)
        beta = 0.0 if hard_failure else float(solver.beta)
        authority = 0.0 if hard_failure else float(solver.authority)
        scan_achieved = result.scan_jacobian @ applied
        scan_residual = np.asarray(solver.scan_residual, dtype=float)
        rail_column = np.asarray(result.scan_jacobian[:, 0], dtype=float)
        rail_xy_contribution = rail_column * float(applied[0])
        arm_xy_contribution = result.scan_jacobian[:, 1:] @ applied[1:]
        rail_norm2 = float(rail_column @ rail_column)
        if rail_norm2 > 1.0e-12:
            anchor_xy = result.scan_jacobian @ solver.anchor
            actual_scan_target = (
                (1.0 - beta) * anchor_xy
                + beta * result.command.feedback_velocity
                + alpha * result.command.path_velocity
            )
            rail_task_projection = float(
                rail_column @ actual_scan_target / rail_norm2
            )
            rail_arm_cancel = float(
                -rail_column @ arm_xy_contribution / rail_norm2
            )
            rail_decomposition_error = abs(
                float(applied[0]) - rail_task_projection - rail_arm_cancel
            )
        else:
            rail_task_projection = float("nan")
            rail_arm_cancel = float("nan")
            rail_decomposition_error = float("nan")
        active_ids = tuple(
            dict.fromkeys((*solver.active_constraint_ids, *violated))
        )
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=applied.copy(),
            twist_base=np.asarray(twist_base, dtype=float).copy(),
            sigma_min=float(sigma_min),
            manip=float(manip),
            slack_norm=float(
                np.linalg.norm(
                    np.concatenate(
                        (
                            protected_residual,
                            solver.working_slack,
                            solver.collision_slack,
                            [solver.dexterity_slack, solver.branch_slack],
                        )
                    )
                )
            ),
            n_cbf_active=sum(
                name.startswith("self_collision:") for name in active_ids
            ),
            follow_err_rad=float(follow_err),
            qdot_ff_norm=float(qdot_ff_norm),
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=float(rail_qdot_ff),
            plan_drives_rail=bool(plan_drives_rail),
            controller_mode="single_qpik",
            qp_backend=self.core.backend_name,
            qp_solver_status=diagnostics.status,
            qp_solver_iterations=int(diagnostics.iterations),
            qp_solver_solve_ms=float(diagnostics.solve_time_ms),
            qp_solver_call_count=int(diagnostics.call_count),
            qp_solver_overrun=bool(diagnostics.overrun),
            qpik_alpha=alpha,
            qpik_beta=beta,
            qpik_authority=authority,
            qpik_equality_residual_max=float(
                diagnostics.equality_residual_max
            ),
            qpik_hard_residual_max=float(solver.hard_residual_max),
            qpik_anchor_valid=bool(solver.anchor_valid),
            qpik_recovery_overflow=bool(solver.recovery_overflow),
            qpik_protected_nominal_overflow=np.asarray(
                solver.protected_nominal_overflow, dtype=float
            ).copy(),
            qpik_recovery_caps=np.asarray(solver.recovery_caps, dtype=float).copy(),
            qpik_recovery_overflow_indices=tuple(
                int(index) for index in solver.recovery_overflow_indices
            ),
            qpik_working_slack=np.asarray(solver.working_slack, dtype=float).copy(),
            qpik_collision_slack=np.asarray(solver.collision_slack, dtype=float).copy(),
            qpik_dexterity_slack=float(solver.dexterity_slack),
            qpik_branch_slack=float(solver.branch_slack),
            rail_macro_pref_v=float(result.rail_macro_preference),
            rail_center_pref_v=float(result.rail_center_preference),
            arm_risk_pref_norm=float(result.arm_risk_preference_norm),
            arm_risk_pref=np.asarray(result.risk_preference, dtype=float).copy(),
            risk_direction_cosine=float(result.risk_direction_cosine),
            path_velocity_xy=np.asarray(
                result.command.path_velocity, dtype=float
            ).copy(),
            feedback_xy_raw=np.asarray(result.feedback_xy_raw, dtype=float).copy(),
            feedback_xy_filtered=np.asarray(
                result.feedback_xy_filtered, dtype=float
            ).copy(),
            rail_xy_contribution=rail_xy_contribution.copy(),
            arm_xy_contribution=arm_xy_contribution.copy(),
            rail_task_projection=rail_task_projection,
            rail_arm_cancel=rail_arm_cancel,
            rail_decomposition_error=rail_decomposition_error,
            wrist_singularity=float(result.wrist_singularity),
            hard_active_constraint_ids=active_ids,
            protected_target=np.asarray(result.protected_target, dtype=float).copy(),
            protected_achieved=protected_achieved.copy(),
            protected_residual=protected_residual.copy(),
            scan_target=np.asarray(result.scan_target, dtype=float).copy(),
            scan_achieved=scan_achieved.copy(),
            scan_residual=scan_residual.copy(),
            fallback_level=(
                "fault"
                if hard_failure
                else ("hard_anchor" if solver.fallback else "none")
            ),
            fallback_reason=(
                integration_fault if integration_fault else solver.fallback_reason
            ),
            solver_fault_latched=hard_failure,
            health_state=result.health.state.value,
            arm_health=(
                float(result.health.arm_rho)
                if result.health.arm_rho is not None
                else float("nan")
            ),
            joint_margin_rad=(
                float(result.health.joint_margin_rad)
                if result.health.joint_margin_rad is not None
                else float("nan")
            ),
            wrist_margin_rad=(
                float(result.health.wrist_margin_rad)
                if result.health.wrist_margin_rad is not None
                else float("nan")
            ),
        )

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
        f_ext_z: float | None = None,
        f_des_z: float | None = None,
        contact_active: bool = False,
        task_rotation_base: np.ndarray | None = None,
        task_safety_rows: tuple[TaskSpaceConstraintRow, ...] = (),
        path_twist: np.ndarray | None = None,
        feedback_twist: np.ndarray | None = None,
    ) -> JointIkStep:
        """Run one fixed Z/orientation-protected, XY-scalable QPIK tick."""

        del vel_ff  # trajectory-specific rail feed-forward is retired
        period = float(self.cfg.dt if dt is None else dt)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("dt must be finite and > 0")
        q_prev = np.asarray(self.q_cmd, dtype=float).copy()
        if q_meas is None:
            raise ValueError("q_meas is required for every Cartesian QPIK tick")
        q_state = np.asarray(q_meas, dtype=float).copy()
        if q_state.shape != (self.kin.nv,) or not np.isfinite(q_state).all():
            raise ValueError(f"q_meas must be a finite {(self.kin.nv,)} vector")
        follow_err = float(np.max(np.abs(q_prev - q_state)))
        twist_task = np.asarray(twist, dtype=float).reshape(-1)
        if twist_task.size != 6 or not np.isfinite(twist_task).all():
            raise ValueError("twist must be a finite 6-vector")

        if task_rotation_base is not None:
            rotation_base_task = np.asarray(task_rotation_base, dtype=float)
        elif self.cfg.control_frame == "tool":
            rotation_base_task = np.asarray(
                self.kin.fk_placement(q_state).rotation, dtype=float
            )
        elif self.cfg.control_frame == "base":
            rotation_base_task = np.eye(3)
        else:
            raise ValueError(
                "an explicit task_rotation_base is required for a custom control frame"
            )
        twist_base = np.concatenate(
            (
                rotation_base_task @ twist_task[:3],
                rotation_base_task @ twist_task[3:],
            )
        )
        J_snapshot = np.asarray(self.kin.jacobian(q_state), dtype=float)
        singular = np.asarray(self.kin.singular_values(J_snapshot), dtype=float)
        sigma_min = float(np.min(singular))
        manip = float(self.kin.manipulability(J_snapshot))

        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)

        qdot_ff_arr: np.ndarray | None = None
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float).reshape(-1)
            if qdot_ff_arr.shape != (self.kin.nv,) or not np.isfinite(qdot_ff_arr).all():
                raise ValueError(f"qdot_ff must be a finite {(self.kin.nv,)} vector")
            qdot_ff_arr = np.clip(
                qdot_ff_arr, -self.safety.lim.v_max, self.safety.lim.v_max
            )

        # Explicit joint/rail PTP remains outside Cartesian QPIK.  It still
        # passes through the common downstream SafetyLimiter and reports the
        # actually integrated velocity.
        direct = bool(self._direct_joint_ptp) or bool(
            rail_only and qdot_ff_arr is not None
        )
        if direct and qdot_ff_arr is not None:
            qdot_direct = qdot_ff_arr.copy()
            if rail_only:
                qdot_direct[1:] = 0.0
            report = self.safety.clamp(
                q_prev, q_prev + qdot_direct * period, period
            )
            self.q_cmd = report.q_safe
            applied = self.safety.sync_applied_delta(
                self.q_cmd - q_prev, period
            ) / period
            self.core.sync_applied(applied)
            self.last_sigma_min = sigma_min
            return JointIkStep(
                q_send=self.q_cmd.copy(),
                qdot=applied.copy(),
                twist_base=twist_base,
                sigma_min=sigma_min,
                manip=manip,
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err_rad=follow_err,
                qdot_ff_norm=float(np.linalg.norm(qdot_ff_arr)),
                vel_clamped=report.vel_clamped,
                acc_clamped=report.acc_clamped,
                pos_clamped=report.pos_clamped,
                rail_vel_pin=float(qdot_ff_arr[0]),
                rail_qdot_ff=float(qdot_ff_arr[0]),
                plan_drives_rail=True,
                controller_mode="direct_joint_ptp",
                qp_backend=self.core.backend_name,
            )

        timestamp = time.monotonic()
        state = RobotState(
            q_meas=q_state,
            q_cmd=q_prev,
            qdot_applied_prev=self.core.qdot_prev,
            dt=period,
            contact_active=bool(contact_active),
            timestamp=timestamp,
        )

        safety_rows = list(task_safety_rows)
        force_is_over_target = (
            f_ext_z is not None
            and f_des_z is not None
            and np.isfinite(float(f_ext_z))
            and np.isfinite(float(f_des_z))
            and float(f_ext_z) > float(f_des_z)
        )
        if force_is_over_target:
            coefficients = np.zeros(6, dtype=float)
            coefficients[2] = 1.0
            safety_rows.append(
                TaskSpaceConstraintRow(
                    coefficients, upper=0.0, name="overforce_do_not_advance"
                )
            )

        rail_vel_pin: float | None = None
        rail_qdot_ff = float("nan")
        if qdot_ff_arr is not None:
            rail_qdot_ff = float(qdot_ff_arr[0])
            if plan_drives_rail:
                rail_vel_pin = rail_qdot_ff
        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)
        path_task = np.zeros(6) if path_twist is None else np.asarray(path_twist, dtype=float).reshape(6)
        feedback_task = (
            twist_task.copy()
            if feedback_twist is None
            else np.asarray(feedback_twist, dtype=float).reshape(6)
        )
        protected_task = twist_task.copy()
        # Only XY are split into path/feedback.  Z and orientation remain the
        # protected command produced by the force/pose outer loop.
        protected_task[:2] = 0.0
        path_task[2:] = 0.0
        feedback_task[2:] = 0.0
        try:
            result = self.core.solve(
                state,
                protected_twist_task=protected_task,
                path_twist_task=path_task,
                feedback_twist_task=feedback_task,
                rotation_base_task=rotation_base_task,
                task_safety_rows=tuple(safety_rows),
                resync_err=resync_vec,
                rail_locked=locked_hold,
                rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
                rail_vel_pin_m_s=rail_vel_pin,
                jacobian_base=J_snapshot,
            )
        except (
            VelocityBoxInfeasible,
            CollisionHardCapacityExceeded,
            HardConstraintCapacityExceeded,
        ) as exc:
            # No hard-feasible velocity was certified.  Keep q_cmd unchanged
            # and return a non-sendable step; the runner invokes the dedicated
            # stop path before either rail or CANFD publication.
            return JointIkStep(
                q_send=q_prev.copy(),
                qdot=np.zeros_like(q_prev),
                twist_base=twist_base,
                sigma_min=sigma_min,
                manip=manip,
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err_rad=follow_err,
                qdot_ff_norm=(
                    float(np.linalg.norm(qdot_ff_arr))
                    if qdot_ff_arr is not None
                    else 0.0
                ),
                rail_vel_pin=(
                    float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
                ),
                rail_qdot_ff=rail_qdot_ff,
                plan_drives_rail=plan_drives_rail,
                controller_mode="single_qpik",
                qp_backend=self.core.backend_name,
                qp_solver_status="not_run_hard_construction_failure",
                qpik_alpha=0.0,
                qpik_beta=0.0,
                qpik_authority=0.0,
                qpik_anchor_valid=False,
                fallback_level="fault",
                fallback_reason=f"{type(exc).__name__}:{exc}",
                solver_fault_latched=True,
            )
        if result.health.arm_rho is not None:
            self.last_arm_rho = float(result.health.arm_rho)

        return self._finish_single_qpik_result(
            result,
            q_prev=q_prev,
            period=period,
            twist_base=twist_base,
            sigma_min=sigma_min,
            manip=manip,
            follow_err=follow_err,
            qdot_ff_norm=(
                float(np.linalg.norm(qdot_ff_arr)) if qdot_ff_arr is not None else 0.0
            ),
            rail_vel_pin=rail_vel_pin,
            rail_qdot_ff=rail_qdot_ff,
            plan_drives_rail=plan_drives_rail,
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
    """Wrap AdmittanceController + a MotionReferenceSource (force-position hybrid)."""

    def __init__(self, controller, reference_source, *, desired_force: np.ndarray | None = None):
        self.controller = controller
        self.reference = reference_source
        self.desired_force = (
            np.zeros(6) if desired_force is None else np.asarray(desired_force, dtype=float)
        )
        self.last_err_mm: float = 0.0
        self.last_track_rot_deg: float = 0.0
        self.last_vel_ff: np.ndarray | None = None
        self.last_path_twist = np.zeros(6)
        self.last_feedback_twist = np.zeros(6)
        self._reference_override = None

    def begin_hybrid_episode(
        self,
        applied_twist_base: np.ndarray,
        current_pose: np.ndarray,
    ) -> None:
        """Reset force-task transients and seed the output from applied motion."""

        seed = np.asarray(applied_twist_base, dtype=float).reshape(6).copy()
        if self.controller.cfg.control_frame == "tool":
            rotation = Rsc.from_euler(
                self.controller.cfg.euler_order,
                np.asarray(current_pose, dtype=float)[3:6],
                degrees=False,
            ).as_matrix()
            seed[:3] = rotation.T @ seed[:3]
            seed[3:] = rotation.T @ seed[3:]
        self.controller.begin_hybrid_episode(seed)
        self.last_path_twist.fill(0.0)
        self.last_feedback_twist.fill(0.0)

    def set_reference_override(self, reference) -> None:
        self._reference_override = reference

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def sample(
        self,
        t_s: float,
        current_pose: np.ndarray,
        f_ext: np.ndarray,
        f_ext_raw: np.ndarray | None = None,
        dt_actual: float | None = None,
        sensor_age_s: float | None = None,
        feedback_age_s: float | None = None,
        feedback_fresh_tick: bool | None = None,
        feedback_velocity_valid: bool | None = None,
        v_tcp_z_actual: float | None = None,
    ) -> np.ndarray:
        ref = self._reference_override
        self._reference_override = None
        if ref is None:
            ref = self.reference.sample(t_s)
        # Track-axis-only error (force axis excluded).
        tr_mm, tr_deg = pose_track_error_mm_deg(
            ref.pose_d,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        # ``feedback_fresh_tick`` is a per-cycle telemetry edge, not a
        # validity gate: when one UDP frame is missed, retain the last valid
        # velocity and let ``feedback_age_s`` decide staleness.  Before the
        # first successful finite-difference estimate, pass no velocity so
        # BEFM remains fail-closed.
        velocity_valid = (
            bool(feedback_velocity_valid)
            if feedback_velocity_valid is not None
            else v_tcp_z_actual is not None
        )
        v_actual = v_tcp_z_actual if velocity_valid else None
        command = self.controller.compute_velocity_command(
            current_pose,
            ref.pose_d,
            ref.vel_ff,
            f_ext,
            self.desired_force,
            f_ext_raw=f_ext_raw,
            dt_actual=dt_actual,
            sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s,
            feedback_fresh=None,
            v_tcp_z_actual=v_actual,
        )
        path = np.asarray(ref.vel_ff, dtype=float).reshape(6).copy()
        if self.controller.cfg.control_frame == "tool":
            rotation = Rsc.from_euler(
                self.controller.cfg.euler_order,
                current_pose[3:6],
                degrees=False,
            ).as_matrix()
            path[:3] = rotation.T @ path[:3]
            path[3:] = rotation.T @ path[3:]
        self.last_path_twist = np.asarray(
            self.controller.last_path_twist, dtype=float
        ).copy()
        self.last_feedback_twist = np.asarray(
            self.controller.last_feedback_twist, dtype=float
        ).copy()
        return command


@dataclass
class CartesianTrackConfig:
    """PD + feedforward Cartesian tracking (no force axis)."""

    k_task: np.ndarray = field(default_factory=lambda: np.full(6, 2.0))
    max_pos_err_m: float = 0.05
    max_rot_err_rad: float = 0.35
    max_lin_vel_m_s: float = 0.4
    max_ang_vel_rad_s: float = 1.5
    euler_order: str = "xyz"
    # Must match JointIkConfig.control_frame (tool twist is rotated by R @ twist).
    control_frame: str = "tool"
    path_feedforward: bool = True


class CartesianTrackOuterLoop:
    """PD + feedforward Cartesian tracking against measured pose (no force)."""

    def __init__(self, reference, cfg: CartesianTrackConfig | None = None) -> None:
        self.reference = reference
        self.cfg = cfg or CartesianTrackConfig()
        self.last_err_mm: float = 0.0
        self.last_vel_ff: np.ndarray | None = None
        self.last_path_twist = np.zeros(6)
        self.last_feedback_twist = np.zeros(6)
        self._reference_override = None

    def set_reference_override(self, reference) -> None:
        self._reference_override = reference

    def set_origin(self, pose0: np.ndarray, *, t_s: float | None = None) -> None:
        if hasattr(self.reference, "set_origin"):
            try:
                self.reference.set_origin(pose0, t_s=t_s)
            except TypeError:
                self.reference.set_origin(pose0)

    def sample(self, t_s: float, current_pose: np.ndarray, f_ext: np.ndarray) -> np.ndarray:
        del f_ext
        cfg = self.cfg
        ref = self._reference_override
        self._reference_override = None
        if ref is None:
            ref = self.reference.sample(t_s)
        self.last_vel_ff = np.asarray(ref.vel_ff, dtype=float).copy()
        err = pose_error(ref.pose_d, current_pose, cfg.euler_order)
        self.last_err_mm = float(np.linalg.norm(err[:3]) * 1000.0)
        err_sat = saturate_error(err, cfg.max_pos_err_m, cfg.max_rot_err_rad)
        v_ff = np.asarray(ref.vel_ff, dtype=float)
        path_base = v_ff.copy() if cfg.path_feedforward else np.zeros(6)
        feedback_base = cfg.k_task * err_sat

        def cap_twist(value: np.ndarray) -> np.ndarray:
            capped = np.asarray(value, dtype=float).copy()
            lin_norm = float(np.linalg.norm(capped[:3]))
            if cfg.max_lin_vel_m_s > 0.0 and lin_norm > cfg.max_lin_vel_m_s:
                capped[:3] *= cfg.max_lin_vel_m_s / lin_norm
            ang_norm = float(np.linalg.norm(capped[3:6]))
            if cfg.max_ang_vel_rad_s > 0.0 and ang_norm > cfg.max_ang_vel_rad_s:
                capped[3:6] *= cfg.max_ang_vel_rad_s / ang_norm
            return capped

        path_base = cap_twist(path_base)
        feedback_base = cap_twist(feedback_base)
        v = cap_twist(path_base + feedback_base)  # base-frame legacy output

        if cfg.control_frame == "tool":
            R = Rsc.from_euler(cfg.euler_order, current_pose[3:6], degrees=False).as_matrix()
            out = np.zeros(6, dtype=float)
            out[:3] = R.T @ v[:3]
            out[3:6] = R.T @ v[3:6]
            path = np.zeros(6)
            path[:3] = R.T @ path_base[:3]
            path[3:6] = R.T @ path_base[3:6]
            feedback = np.zeros(6)
            feedback[:3] = R.T @ feedback_base[:3]
            feedback[3:6] = R.T @ feedback_base[3:6]
            self.last_path_twist = path
            self.last_feedback_twist = feedback
            return out
        self.last_path_twist = path_base
        self.last_feedback_twist = feedback_base
        return v


@dataclass
class JointTrackConfig:
    """Joint-space PD + feedforward tracking (MoveJ-like; no Cartesian stall)."""

    k_joint: float = 2.0
    max_joint_err_rad: float = 0.35
    sigma_ref: float = 0.08
    # σ-adaptive floor: k_eff = k_joint * max(σ/σ_ref, floor).
    k_joint_sigma_min_frac: float = 0.2
    control_frame: str = "tool"
    euler_order: str = "xyz"
    # Rise-only slew on k_eff (1/s); fall is immediate for singularity protection.
    k_joint_rise_per_s: float = 1.2
    # LPF on last_qdot_fb (s); damps QP dual chatter when secondary ≈ slack·W_task.
    fb_lpf_tau_s: float = 0.015
    # Scale fb secondary pull (0..1); keeps QP reg well-conditioned.
    fb_secondary_gain: float = 0.4


class JointTrackOuterLoop:
    """MoveJ-like outer loop: track joint plan via J(q)·(qdot_plan + k·q_err)."""

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
        # Feedback-only term for QP secondary (plan ff is governor-scaled separately).
        self.last_qdot_fb: np.ndarray | None = None
        self.last_qdot_command: np.ndarray | None = None
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
        # Rise-only slew on k_eff (fall is immediate).
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
        # Scale secondary fb only; primary v_cmd still uses full qdot_fb_raw.
        self.last_qdot_fb = self._qdot_fb_lpf * float(cfg.fb_secondary_gain)
        qdot_cmd = qdot_plan + qdot_fb_raw
        v_lim = np.asarray(self.v_max, dtype=float)
        qdot_cmd = np.clip(qdot_cmd, -v_lim, v_lim)
        self.last_qdot_command = qdot_cmd.copy()
        v_base = J @ qdot_cmd
        # Soften primary twist near σ or with large residual q_err.
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


# Spin the last ~1 ms of the period (sleep often wakes 1–3 ms late at 200 Hz).
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
    """One leg of a multi-phase on-robot run (shared inner loop / watchdog).

    ``t_ref`` advances by ``dt * governor_scale``; qdot_ff is sampled at the
    same governed ``t_ref``. Set ``governor_err_max_mm=0`` to disable Cartesian
    governor (typical for MoveJ-like joint moves).
    """

    outer: OuterLoop
    label: str = ""
    duration_s: float | None = None          # None -> run until wait_until (or max_duration_s)
    max_duration_s: float | None = None      # wall-clock safety cap
    wait_until: object | None = None         # Callable pose or (pose, q_meas) -> bool
    qdot_ff_provider: object | None = None   # Callable[[float], qdot_ff_rad_s] sampled at t_ref
    scale_qdot_ff_with_governor: bool = True # False keeps plan-anchor alive when t_ref frozen
    require_arrival: bool = False            # abort later phases if wait_until never fires
    arrival_plan_duration_s: float | None = None
    arrival_dwell_s: float = 0.0
    arrival_arm_speed_rad_s: float = 0.02
    arrival_rail_speed_m_s: float = 0.003
    governor_err_ok_mm: float = 5.0
    governor_err_max_mm: float = 25.0
    # Joint-space governor: enable with governor_joint_err_max_deg > 0.
    governor_joint_err_ok_deg: float = 3.0
    governor_joint_err_max_deg: float = 0.0
    governor_tau_s: float = 0.2
    governor_freeze_below: float = 0.02
    governor_release_above: float = 0.10
    soft_start_ramp_s: float = 0.0           # governor soft-start at phase entry (s)
    force_observer: object | None = None     # None -> reuse the loop-level force_observer
    on_enter: object | None = None           # Callable[[], None], fired right after set_origin
    on_exit: object | None = None            # Callable[[], None], fired when phase completes
    on_tick: object | None = None            # Callable[[float, JointIkStep, np.ndarray], None]


@dataclass
class _ArrivalDwellGate:
    """Require plan completion, geometric arrival, and settled sent velocity."""

    plan_duration_s: float | None
    dwell_required_s: float
    arm_speed_rad_s: float
    rail_speed_m_s: float
    dwell_s: float = 0.0

    def update(
        self,
        *,
        geometric_arrival: bool,
        t_ref_s: float,
        qdot_applied: np.ndarray,
        dt_s: float,
        rail_settled: bool | None = None,
    ) -> bool:
        qdot = np.asarray(qdot_applied, dtype=float).reshape(-1)
        if qdot.size != 8 or not np.all(np.isfinite(qdot)):
            self.dwell_s = 0.0
            return False
        plan_complete = bool(
            self.plan_duration_s is None
            or float(t_ref_s) >= float(self.plan_duration_s) - 1.0e-12
        )
        rail_speed_ok = (
            abs(float(qdot[0])) <= max(float(self.rail_speed_m_s), 0.0)
            if rail_settled is None
            else bool(rail_settled)
        )
        speed_ok = bool(
            rail_speed_ok
            and np.max(np.abs(qdot[1:]), initial=0.0)
            <= max(float(self.arm_speed_rad_s), 0.0)
        )
        candidate = bool(geometric_arrival and plan_complete and speed_ok)
        if candidate:
            self.dwell_s += max(float(dt_s), 0.0)
        else:
            self.dwell_s = 0.0
        return bool(candidate and (
            self.dwell_s + 1.0e-12 >= max(float(self.dwell_required_s), 0.0)
        ))


class _TickLogger:
    """Async per-tick CSV telemetry (background writer; no sync flush in the RT loop)."""

    @staticmethod
    def _json_compact(value) -> str:
        """Encode structured telemetry as deterministic, strict JSON.

        CSV remains the transport for compatibility with existing replay
        tools.  Variable-length task rows/groups are kept in one compact JSON
        cell; non-finite floats become ``null`` instead of invalid JSON NaN.
        """

        def normalize(item):
            if isinstance(item, np.ndarray):
                return normalize(item.tolist())
            if isinstance(item, np.generic):
                return normalize(item.item())
            if isinstance(item, dict):
                return {
                    str(key): normalize(item[key])
                    for key in sorted(item, key=lambda key: str(key))
                }
            if isinstance(item, (tuple, list)):
                return [normalize(entry) for entry in item]
            if isinstance(item, (float, np.floating)):
                return float(item) if np.isfinite(item) else None
            if isinstance(item, (int, np.integer, bool, str)) or item is None:
                return item
            value = getattr(item, "value", None)
            if value is not None and value is not item:
                return normalize(value)
            return str(item)

        return json.dumps(
            normalize(value),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    _HEADER = (
        ["t_wall_s", "phase", "controller_mode", "t_ref_s"]
        + [f"q_cmd_{i}" for i in range(0, 8)]
        + [f"q_meas_{i}" for i in range(0, 8)]
        + [f"pose_{a}" for a in ("x", "y", "z", "rx", "ry", "rz")]
        # twist_* = deprecated alias of twist_requested_*; achieved = J(q)qdot.
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
           "force_reference_fast_clear",
           "force_fast_z",
           "retract_guard_armed", "retract_fast_hold",
           "retract_fast_stop_count", "retract_fast_rearm_count",
           "force_task_latched",
           "physical_contact_state",
           "physical_contact_acquire_event", "physical_contact_loss_event",
           "physical_contact_reacquire_event",
           "physical_contact_low_timer_s", "physical_contact_high_timer_s",
           "mass_z_eff", "takeover",
           "dt_actual_s", "sensor_age_s", "feedback_age_s",
           "feedback_fresh_tick",
           "fx_raw_comp", "fy_raw_comp", "fz_raw_comp",
           "vz_achieved_tool", "contact_present",
           "force_pred_z", "force_dot_z", "cap_press_z", "cap_retract_z",
           "ke_update_gated", "ke_dx_m", "ke_df_n", "ke_update_count",
           "governor_scale", "governor_scale_raw", "sigma_min",
           "qdot_norm", "qdot_max_frac_vmax",
           "qdot_ff_norm", "tcp_jump_mm",
           "rail_target_sent_m", "rail_meas_m",
           "rail_vel_pin", "plan_drives_rail", "rail_qdot_ff",
           # Append-only normal-axis BEFM/audit schema.
           "flow_x_p", "flow_v_p", "flow_v_aux", "flow_x_a", "flow_v_a",
           "flow_e", "flow_edot", "flow_F_c", "flow_v_track",
           "flow_P_e", "flow_P_c", "flow_alpha_target", "flow_alpha",
           "flow_alpha_case", "flow_T", "flow_psi", "flow_S_n",
           "flow_S_r_hat", "flow_P_phys", "flow_P_mismatch",
           "flow_E_phys", "flow_E_mismatch", "flow_gamma_active",
           "flow_sign_fault", "flow_feedback_stale", "flow_blocked_reason",
           "contact_episode_rearm_event", "contact_episode_release_s",
           "surface_force_scale", "surface_force_alpha", "surface_xy_error_m",
           "force_barrier_contact_active",
           # Fixed single-shot QPIK telemetry.
           "qpik_backend", "qpik_solver_status", "qpik_solver_iterations",
           "qpik_solver_solve_ms", "qpik_solver_call_count",
           "qpik_solver_overrun",
           "qpik_alpha", "qpik_beta", "qpik_authority",
           "qpik_equality_residual_max", "qpik_hard_residual_max",
           "qpik_anchor_valid", "qpik_recovery_overflow",
           "qpik_protected_nominal_overflow_json",
           "qpik_recovery_caps_json",
           "qpik_recovery_overflow_indices_json",
           "qpik_hard_active_constraint_ids_json",
           "qpik_protected_target_json", "qpik_protected_achieved_json",
           "qpik_protected_residual_json",
           "qpik_scan_target_json", "qpik_scan_achieved_json",
           "qpik_scan_residual_json", "qpik_working_slack_json",
           "qpik_collision_slack_json", "qpik_dexterity_slack",
           "qpik_branch_slack", "qpik_rail_macro_pref_v",
           "qpik_rail_center_pref_v",
           "qpik_rail_final_qdot", "qpik_arm_risk_pref_norm",
           "qpik_arm_risk_pref_json", "qpik_risk_direction_cosine",
           "qpik_path_velocity_xy_json",
           "qpik_feedback_xy_raw_json", "qpik_feedback_xy_filtered_json",
           "qpik_rail_xy_contribution_json", "qpik_arm_xy_contribution_json",
           "qpik_rail_task_projection", "qpik_rail_arm_cancel",
           "qpik_rail_decomposition_error",
           "qpik_arm_rho", "qpik_joint_margin_rad",
           "qpik_wrist_margin_rad", "qpik_wrist_singularity",
           "qpik_accepted_reference_lag_s",
           "qpik_pre_solve_feedback_age_s", "qpik_post_solve_feedback_age_s",
           "qpik_q_cmd_q_meas_norm", "qpik_fallback_level",
           "qpik_fallback_reason", "qpik_solver_fault_latched",
           "qpik_final_sent_qdot_json"]
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
        feedback_age_s: float = float("nan"),
        feedback_fresh_tick: bool = False,
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
        force_reference_fast_clear = getattr(
            ctrl, "force_reference_fast_clear", False
        )
        force_fast_z = getattr(ctrl, "force_fast_z", float("nan"))
        retract_guard_armed = getattr(ctrl, "retract_guard_armed", False)
        retract_fast_hold = getattr(ctrl, "retract_fast_hold", False)
        retract_fast_stop_count = getattr(
            ctrl, "retract_fast_stop_count", 0
        )
        retract_fast_rearm_count = getattr(
            ctrl, "retract_fast_rearm_count", 0
        )
        force_task_latched = getattr(ctrl, "force_task_latched", False)
        physical_contact_state = getattr(
            ctrl, "physical_contact_state", ""
        )
        physical_contact_acquire_event = getattr(
            ctrl, "physical_contact_acquire_event", False
        )
        physical_contact_loss_event = getattr(
            ctrl, "physical_contact_loss_event", False
        )
        physical_contact_reacquire_event = getattr(
            ctrl, "physical_contact_reacquire_event", False
        )
        physical_contact_tracker = getattr(ctrl, "_physical_contact", None)
        physical_contact_low_timer = getattr(
            ctrl,
            "physical_contact_low_timer_s",
            getattr(physical_contact_tracker, "low_timer_s", float("nan")),
        )
        physical_contact_high_timer = getattr(
            ctrl,
            "physical_contact_high_timer_s",
            getattr(physical_contact_tracker, "high_timer_s", float("nan")),
        )
        mass_z_eff = getattr(ctrl, "mass_z_eff", float("nan"))
        takeover = getattr(ctrl, "takeover_active", False)
        contact_present = getattr(ctrl, "contact_present", False)
        cap_press_z = getattr(ctrl, "cap_press_z", float("nan"))
        cap_retract_z = getattr(ctrl, "cap_retract_z", float("nan"))
        force_pred_z = getattr(ctrl, "force_pred_z", float("nan"))
        force_dot_z = getattr(ctrl, "force_dot_z", float("nan"))
        force_barrier_contact_active = getattr(
            ctrl, "force_barrier_contact_active", False
        )
        ke_tracker = getattr(ctrl, "_ke_estimator", None)
        ke_update_gated = getattr(ke_tracker, "update_gated", False)
        ke_dx_m = getattr(ke_tracker, "last_dx_m", float("nan"))
        ke_df_n = getattr(ke_tracker, "last_df_n", float("nan"))
        ke_update_count = getattr(ke_tracker, "update_count", 0)
        flow = getattr(ctrl, "bidirectional_flow", None)
        flow_xp = getattr(flow, "xp", float("nan"))
        flow_vp = getattr(flow, "vp", float("nan"))
        flow_v_aux = getattr(flow, "v_aux", float("nan"))
        flow_xa = getattr(flow, "xa", float("nan"))
        flow_va = getattr(flow, "va", float("nan"))
        flow_e = getattr(flow, "e", float("nan"))
        flow_edot = getattr(flow, "edot", float("nan"))
        flow_fc = getattr(flow, "fc", float("nan"))
        flow_v_track = getattr(flow, "v_track", float("nan"))
        flow_pe = getattr(flow, "Pe", float("nan"))
        flow_pc = getattr(flow, "Pc", float("nan"))
        flow_alpha_target = getattr(flow, "alpha_raw", float("nan"))
        flow_alpha = getattr(flow, "alpha", float("nan"))
        flow_alpha_case = getattr(flow, "alpha_case", "")
        flow_tank = getattr(flow, "tank_energy", float("nan"))
        flow_psi = getattr(flow, "psi", float("nan"))
        flow_sn = getattr(flow, "Sn", float("nan"))
        flow_sr = getattr(flow, "Sr_hat", float("nan"))
        flow_p_phys = getattr(flow, "P_phys", float("nan"))
        flow_p_mismatch = getattr(flow, "P_mismatch", float("nan"))
        flow_e_phys = getattr(flow, "energy_phys_j", float("nan"))
        flow_e_mismatch = getattr(flow, "energy_mismatch_j", float("nan"))
        flow_gamma = getattr(flow, "gamma_effective", float("nan"))
        flow_sign_fault = getattr(flow, "sign_fault", True)
        flow_stale = getattr(flow, "feedback_stale", True)
        flow_blocked = getattr(flow, "blocked_reason", "")
        episode_rearm = getattr(ctrl, "contact_episode_rearm_event", False)
        episode_release_s = getattr(
            ctrl, "contact_episode_release_s", float("nan")
        )
        surface_force_scale = getattr(ctrl, "surface_force_scale", float("nan"))
        surface_force_alpha = getattr(ctrl, "surface_force_alpha", float("nan"))
        surface_xy_error_m = getattr(ctrl, "surface_xy_error_m", float("nan"))
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
        # Max |qdot|/v_max (1.0 = saturated on at least one joint).
        if v_max is not None and np.any(v_max > 1e-9):
            qdot_max_frac = float(np.max(np.abs(step.qdot) / np.maximum(v_max, 1e-9)))
        else:
            qdot_max_frac = float("nan")
        rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        try:
            q_cmd_q_meas_norm = float(
                np.linalg.norm(np.asarray(step.q_send, dtype=float) - qm)
            )
        except (TypeError, ValueError):
            q_cmd_q_meas_norm = float("nan")
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
               int(bool(force_reference_fast_clear)),
               f"{force_fast_z:.3f}",
               int(bool(retract_guard_armed)),
               int(bool(retract_fast_hold)),
               int(retract_fast_stop_count),
               int(retract_fast_rearm_count),
               int(bool(force_task_latched)),
               str(physical_contact_state),
               int(bool(physical_contact_acquire_event)),
               int(bool(physical_contact_loss_event)),
               int(bool(physical_contact_reacquire_event)),
               f"{float(physical_contact_low_timer):.6f}",
               f"{float(physical_contact_high_timer):.6f}",
               f"{mass_z_eff:.4f}",
               int(bool(takeover)),
               f"{dt_actual_s:.6f}", f"{sensor_age_s:.6f}",
               f"{feedback_age_s:.6f}", int(bool(feedback_fresh_tick)),
               f"{raw_comp[0]:.3f}", f"{raw_comp[1]:.3f}", f"{raw_comp[2]:.3f}",
               f"{v_tcp_z_actual:.6f}", int(bool(contact_present)),
               f"{force_pred_z:.4f}", f"{force_dot_z:.4f}",
               f"{cap_press_z:.6f}", f"{cap_retract_z:.6f}",
               int(bool(ke_update_gated)), f"{ke_dx_m:.8f}", f"{ke_df_n:.5f}",
               int(ke_update_count),
               f"{governor_scale:.4f}", f"{governor_scale_raw:.4f}",
               f"{step.sigma_min:.5f}",
               f"{qdot_norm:.5f}", f"{qdot_max_frac:.4f}",
               f"{step.qdot_ff_norm:.5f}", f"{step.tcp_jump_mm:.3f}",
               f"{rail_sent:.6f}",
               f"{rail_meas_m:.6f}" if np.isfinite(rail_meas_m) else "",
               f"{step.rail_vel_pin:.6f}" if np.isfinite(step.rail_vel_pin) else "",
               int(bool(step.plan_drives_rail)),
               f"{step.rail_qdot_ff:.6f}" if np.isfinite(step.rail_qdot_ff) else "",
               f"{flow_xp:.8f}", f"{flow_vp:.8f}", f"{flow_v_aux:.8f}",
               f"{flow_xa:.8f}", f"{flow_va:.8f}", f"{flow_e:.8f}",
               f"{flow_edot:.8f}", f"{flow_fc:.8f}", f"{flow_v_track:.8f}",
               f"{flow_pe:.8f}", f"{flow_pc:.8f}",
               f"{flow_alpha_target:.8f}", f"{flow_alpha:.8f}",
               str(flow_alpha_case), f"{flow_tank:.9f}", f"{flow_psi:.8f}",
               f"{flow_sn:.9f}", f"{flow_sr:.9f}", f"{flow_p_phys:.8f}",
               f"{flow_p_mismatch:.8f}", f"{flow_e_phys:.9f}",
               f"{flow_e_mismatch:.9f}", f"{flow_gamma:.8f}",
               int(bool(flow_sign_fault)), int(bool(flow_stale)),
               str(flow_blocked), int(bool(episode_rearm)),
               f"{episode_release_s:.6f}", f"{surface_force_scale:.6f}",
               f"{surface_force_alpha:.6f}", f"{surface_xy_error_m:.8f}",
               int(bool(force_barrier_contact_active)),
               str(step.qp_backend), str(step.qp_solver_status),
               int(step.qp_solver_iterations),
               f"{step.qp_solver_solve_ms:.6f}",
               int(step.qp_solver_call_count),
               int(bool(step.qp_solver_overrun)),
               f"{step.qpik_alpha:.8f}", f"{step.qpik_beta:.8f}",
               f"{step.qpik_authority:.8f}",
               f"{step.qpik_equality_residual_max:.9e}",
               f"{step.qpik_hard_residual_max:.9e}",
               int(bool(step.qpik_anchor_valid)),
               int(bool(step.qpik_recovery_overflow)),
               self._json_compact(step.qpik_protected_nominal_overflow),
               self._json_compact(step.qpik_recovery_caps),
               self._json_compact(step.qpik_recovery_overflow_indices),
               self._json_compact(step.hard_active_constraint_ids),
               self._json_compact(step.protected_target),
               self._json_compact(step.protected_achieved),
               self._json_compact(step.protected_residual),
               self._json_compact(step.scan_target),
               self._json_compact(step.scan_achieved),
               self._json_compact(step.scan_residual),
               self._json_compact(step.qpik_working_slack),
               self._json_compact(step.qpik_collision_slack),
               f"{step.qpik_dexterity_slack:.9e}",
               f"{step.qpik_branch_slack:.9e}",
               f"{step.rail_macro_pref_v:.8f}",
               f"{step.rail_center_pref_v:.8f}", f"{step.qdot[0]:.8f}",
               f"{step.arm_risk_pref_norm:.8f}",
               self._json_compact(step.arm_risk_pref),
               f"{step.risk_direction_cosine:.8f}",
               self._json_compact(step.path_velocity_xy),
               self._json_compact(step.feedback_xy_raw),
               self._json_compact(step.feedback_xy_filtered),
               self._json_compact(step.rail_xy_contribution),
               self._json_compact(step.arm_xy_contribution),
               f"{step.rail_task_projection:.8f}",
               f"{step.rail_arm_cancel:.8f}",
               f"{step.rail_decomposition_error:.9e}",
               f"{step.arm_health:.8f}",
               f"{step.joint_margin_rad:.8f}", f"{step.wrist_margin_rad:.8f}",
               f"{step.wrist_singularity:.8f}",
               f"{step.accepted_reference_lag_s:.6f}",
               f"{step.pre_solve_feedback_age_s:.6f}",
               f"{step.post_solve_feedback_age_s:.6f}",
               f"{q_cmd_q_meas_norm:.8f}",
               str(step.fallback_level), str(step.fallback_reason),
               int(bool(step.solver_fault_latched)),
               self._json_compact(step.qdot)]
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
    """Seed WBC ``q_cmd[0]`` from encoder so the first set_target is near reality."""
    if rail_bridge is not None and rail_bridge.enabled:
        return float(rail_bridge.measured_m)
    return float(inner.q_cmd[0])


def _rail_m_for_feedback(rail_bridge, inner: JointIkController) -> float:
    """Return measured rail position; enabled-rail faults must stop 8D QPIK."""
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return float(inner.q_cmd[0])
    try:
        meas = float(rail_bridge.measured_m)
    except Exception as exc:
        raise RuntimeError(f"rail feedback unavailable: {exc}") from exc
    sane = getattr(rail_bridge, "_encoder_sane", None)
    if callable(sane):
        if not sane(meas):
            raise RuntimeError(f"rail encoder value is invalid: {meas!r}")
    elif not (np.isfinite(meas)):
        raise RuntimeError(f"rail encoder value is non-finite: {meas!r}")
    return meas


def _rail_settled_for_arrival(
    rail_bridge,
    *,
    speed_limit_m_s: float,
    now_s: float,
    freshness_s: float,
) -> bool | None:
    """Return worker-aligned rail standstill, or None when no rail is active."""

    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return None
    try:
        sample = rail_bridge.servo_sample
        sample_time = float(sample.sample_mono_s)
        v_cmd = float(sample.v_cmd_m_s)
        v_meas = float(sample.v_meas_m_s)
    except Exception:
        return False
    if not all(np.isfinite(value) for value in (sample_time, v_cmd, v_meas)):
        return False
    if max(0.0, float(now_s) - sample_time) > max(float(freshness_s), 0.0):
        return False
    limit = max(float(speed_limit_m_s), 0.0)
    return bool(abs(v_cmd) <= limit and abs(v_meas) <= limit)


def _publish_rail_target_before_arm(
    rail_bridge,
    target_m: float,
    fault_stop,
) -> tuple[bool, str]:
    """Require the rail to accept this 8D tick before publishing the arm half."""

    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return True, ""
    if not bool(getattr(rail_bridge, "calibrated", False)):
        reason = "rail_target_rejected:not_calibrated"
    elif not bool(getattr(rail_bridge, "armed", False)):
        reason = "rail_target_rejected:not_armed"
    elif bool(getattr(rail_bridge, "panicked", False)):
        detail = str(getattr(rail_bridge, "panic_reason", "") or "panic")
        reason = f"rail_target_rejected:{detail}"
    else:
        try:
            accepted = rail_bridge.set_target_m(float(target_m))
        except Exception as exc:
            reason = f"rail_target_exception:{type(exc).__name__}:{exc}"
        else:
            if accepted is True:
                return True, ""
            reason = "rail_target_rejected:bridge_declined"
    fault_stop(reason)
    return False, reason


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
    """Raw governor scale in [0, 1] (min of active bands); filter in GovernorFilter."""
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
    """First-order LPF + freeze hysteresis on the governor scale."""

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


def _guard_qpik_step_before_send(step: JointIkStep, fault_stop) -> tuple[bool, str]:
    """Gate every rail/CANFD publication on final solver validation.

    Keeping this decision outside the hardware writers makes the critical
    stop-before-send ordering directly testable.  A fault never becomes a
    decaying trajectory command.
    """

    if not bool(step.solver_fault_latched):
        return True, ""
    reason = f"qpik_fault:{step.fallback_level}:{step.fallback_reason}"
    fault_stop(reason)
    return False, reason


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
    """Run ``Phase`` objects on the robot as one continuous CANFD stream."""
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
        # Cartesian loop uses Pinocchio TCP (may differ from RealMan FK).
        pose0 = inner.kin.fk_pose(q0_rad)
        inner.reset(q0_rad)

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

        def _fault_stop(reason: str) -> None:
            """Stop both axes without publishing another trajectory target."""

            if verbose:
                print(f"  QPIK SAFETY STOP: {reason}", flush=True)
            if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
                try:
                    rail_bridge.hold_current()
                except Exception:
                    try:
                        rail_bridge.kill_motion()
                    except Exception:
                        pass
            if robot is not None:
                try:
                    robot.rm_set_arm_slow_stop()
                except Exception:
                    pass

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
                    # Phase origin from encoders (never from the command integrator).
                    snap = async_obs.read()
                    if snap.q_deg is not None:
                        rail_seed = _rail_m_for_init(rail_bridge, inner)
                        q_meas = _expand_q_meas(deg2rad(snap.q_deg), rail_seed)
                    pose_pin = inner.kin.fk_pose(q_meas)
                    # Soft-start: reseed plan from live encoders (no tick-0 lurch).
                    ref = getattr(phase.outer, "reference", None)
                    if ref is not None:
                        try:
                            q_live = np.asarray(q_meas, dtype=float).reshape(-1)
                            if hasattr(ref, "reseed_start"):
                                ref.reseed_start(q_live)
                            elif hasattr(ref, "q_start") and hasattr(ref, "q_target"):
                                if q_live.size == int(np.asarray(ref.q_start).size):
                                    ref.q_start = q_live.copy()
                        except Exception:
                            pass
                    if hasattr(phase.outer, "set_origin"):
                        phase.outer.set_origin(pose_pin)
                    if phase.on_enter is not None:
                        phase.on_enter()
                    if hasattr(phase.outer, "begin_hybrid_episode"):
                        applied_qdot = inner.core.qdot_prev
                        applied_twist = inner.kin.jacobian(q_meas) @ applied_qdot
                        inner.begin_hybrid_episode(q_meas, applied_qdot)
                        phase.outer.begin_hybrid_episode(applied_twist, pose_pin)

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
                    arrival_gate = _ArrivalDwellGate(
                        plan_duration_s=phase.arrival_plan_duration_s,
                        dwell_required_s=phase.arrival_dwell_s,
                        arm_speed_rad_s=phase.arrival_arm_speed_rad_s,
                        rail_speed_m_s=phase.arrival_rail_speed_m_s,
                    )
                    prev_pose_cmd = inner.kin.fk_pose(inner.q_cmd)
                    # Encoder TCP velocity: update only on a fresh UDP sequence.
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    # ``feedback_age_s`` tracks the last sample from which a
                    # finite-difference TCP velocity was actually computed;
                    # sensor transport age is a separate diagnostic.
                    last_feedback_velocity_t = last_feedback_t
                    twist_achieved_base = np.zeros(6, dtype=float)
                    v_tcp_z_actual = 0.0
                    feedback_velocity_valid = False
                    feedback_fresh_tick = False
                    first_tick = True
                    wd.arm()
                    while True:
                        if stop_check is not None and stop_check():
                            phase_stopped = True
                            break
                        if not wd.fired:
                            wd.beat()
                        now = time.perf_counter()
                        dt_raw = now - last_tick_time
                        last_tick_time = now
                        # The first phase tick occurs immediately after setup;
                        # subsequent wall periods are only sanity-clamped so
                        # >15 ms stalls remain visible to the force/proxy
                        # dynamics.  Inner QP integration stays on ``dt``.
                        if first_tick:
                            dt_wall_actual = float(dt)
                            first_tick = False
                        else:
                            dt_wall_actual = float(
                                np.clip(
                                    dt_raw if np.isfinite(dt_raw) else dt,
                                    1.0e-4,
                                    0.10,
                                )
                            )
                        next_tick, late_ms = _resync_late_tick(next_tick, now, dt)
                        if late_ms > dt * 1000.0:
                            stutter_count += 1
                        max_jitter_ms = max(max_jitter_ms, late_ms)
                        t_wall = now - phase_t0
                        if (
                            phase.duration_s is not None
                            and phase.wait_until is None
                            and t_ref >= phase.duration_s
                        ):
                            break
                        if phase.max_duration_s is not None and t_wall >= phase.max_duration_s:
                            break
    
                        feedback_fresh_tick = False
                        snap = async_obs.read()
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            try:
                                rail_measured_m = _rail_m_for_feedback(
                                    rail_bridge, inner
                                )
                            except RuntimeError as exc:
                                phase_stopped = True
                                stop_reason = f"rail_feedback_fault:{exc}"
                                _fault_stop(stop_reason)
                                break
                            q_new = _expand_q_meas(
                                deg2rad(snap.q_deg), rail_measured_m
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
                                    feedback_fresh_tick = True
                                    feedback_velocity_valid = True
                                    last_feedback_velocity_t = snap_t
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
                        feedback_age_s = (
                            max(0.0, time.monotonic() - last_feedback_velocity_t)
                            if last_feedback_velocity_t > 0.0
                            else float("inf")
                        )

                        if (
                            not np.isfinite(sensor_age_s)
                            or sensor_age_s > float(inner.cfg.feedback_timeout_s)
                        ):
                            phase_stopped = True
                            stop_reason = (
                                "feedback_stale: "
                                f"age={sensor_age_s:.6f}s > "
                                f"{inner.cfg.feedback_timeout_s:.6f}s"
                            )
                            _fault_stop(stop_reason)
                            break

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
                        sample_params = inspect.signature(phase.outer.sample).parameters
                        sample_kwargs: dict = {}
                        if "q_meas" in sample_params:
                            sample_kwargs["q_meas"] = q_meas
                        if "f_ext_raw" in sample_params and f_ext_raw is not None:
                            # Unfiltered wrench for Dimeas (LPF hides the band).
                            sample_kwargs["f_ext_raw"] = f_ext_raw
                        if "dt_actual" in sample_params:
                            sample_kwargs["dt_actual"] = dt_wall_actual
                        if "v_tcp_z_actual" in sample_params:
                            sample_kwargs["v_tcp_z_actual"] = v_tcp_z_actual
                        if "sensor_age_s" in sample_params:
                            sample_kwargs["sensor_age_s"] = sensor_age_s
                        if "feedback_age_s" in sample_params:
                            sample_kwargs["feedback_age_s"] = feedback_age_s
                        if "feedback_fresh_tick" in sample_params:
                            sample_kwargs["feedback_fresh_tick"] = feedback_fresh_tick
                        if "feedback_velocity_valid" in sample_params:
                            sample_kwargs["feedback_velocity_valid"] = (
                                feedback_velocity_valid
                            )
                        twist = np.asarray(
                            phase.outer.sample(t_ref, pose_pin, f_ext, **sample_kwargs),
                            dtype=float,
                        )
                        qdot_ff = (
                            phase.qdot_ff_provider(t_ref)
                            if phase.qdot_ff_provider is not None
                            else None
                        )
                        qdot_command = getattr(
                            phase.outer, "last_qdot_command", None
                        )
                        if qdot_command is not None:
                            qdot_ff = np.asarray(qdot_command, dtype=float).copy()
                        if qdot_ff is not None:
                            qdot_ff = np.asarray(qdot_ff, dtype=float)
                            if phase.scale_qdot_ff_with_governor:
                                qdot_ff = qdot_ff * scale
                        # Additive joint fb (not governor-scaled) closes nullspace q_err.
                        qdot_fb = getattr(phase.outer, "last_qdot_fb", None)
                        if qdot_fb is not None and qdot_command is None:
                            qdot_fb = np.asarray(qdot_fb, dtype=float)
                            qdot_ff = qdot_fb if qdot_ff is None else (qdot_ff + qdot_fb)
                        vel_ff_ref = getattr(phase.outer, "last_vel_ff", None)
                        path_twist = getattr(phase.outer, "last_path_twist", None)
                        feedback_twist = getattr(
                            phase.outer, "last_feedback_twist", None
                        )
                        control_dt = dt
                        ctrl = getattr(phase.outer, "controller", None)
                        f_des_z = float(
                            getattr(ctrl, "f_des_z_eff", float("nan"))
                        ) if ctrl is not None else float("nan")
                        f_ext_z = (
                            float(f_ext[2])
                            if f_ext is not None and len(f_ext) > 2
                            else float("nan")
                        )
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                            f_ext_z=f_ext_z if math.isfinite(f_ext_z) else None,
                            f_des_z=f_des_z if math.isfinite(f_des_z) else None,
                            contact_active=bool(
                                getattr(ctrl, "contact_present", False)
                                if ctrl is not None
                                else False
                            ),
                            path_twist=path_twist,
                            feedback_twist=feedback_twist,
                        )
                        step.pre_solve_feedback_age_s = sensor_age_s
                        # A hard-construction/final-validation fault is acted on before the
                        # rail target or CANFD joint command can be published.
                        sendable, qpik_stop_reason = _guard_qpik_step_before_send(
                            step, _fault_stop
                        )
                        if not sendable:
                            phase_stopped = True
                            stop_reason = qpik_stop_reason
                            break
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
                        publication_reason = ""
                        if stop_check is not None and stop_check():
                            publication_reason = "external_stop_before_send"
                        elif wd.fired:
                            publication_reason = "watchdog_fired_before_send"
                        else:
                            publish_snap = async_obs.read()
                            snap_time = float(getattr(publish_snap, "t_s", 0.0))
                            post_solve_sensor_age_s = (
                                max(0.0, time.monotonic() - snap_time)
                                if snap_time > 0.0
                                else float("inf")
                            )
                            step.post_solve_feedback_age_s = post_solve_sensor_age_s
                            if (
                                not np.isfinite(post_solve_sensor_age_s)
                                or post_solve_sensor_age_s
                                > float(inner.cfg.feedback_timeout_s)
                            ):
                                publication_reason = (
                                    "feedback_stale_before_send:"
                                    f"age={post_solve_sensor_age_s:.6f}s"
                                )
                            elif not wd.beat():
                                publication_reason = "watchdog_latched_before_send"
                        if publication_reason:
                            phase_stopped = True
                            stop_reason = publication_reason
                            _fault_stop(stop_reason)
                            break
                        rail_ok, rail_reason = _publish_rail_target_before_arm(
                            rail_bridge,
                            float(step.q_send[0]),
                            _fault_stop,
                        )
                        if not rail_ok:
                            phase_stopped = True
                            stop_reason = rail_reason
                            break
                        try:
                            _send_joint_canfd_cmd(
                                robot,
                                rad2deg(arm_q_from_full(step.q_send)),
                                follow,
                                canfd_proxy,
                            )
                        except Exception as exc:
                            phase_stopped = True
                            stop_reason = (
                                "arm_send_fault:"
                                f"{type(exc).__name__}:{exc}"
                            )
                            _fault_stop(stop_reason)
                            break
    
                        if step.controller_mode == "single_qpik":
                            raw_scale = float(step.qpik_authority)
                            scale = raw_scale
                        else:
                            joint_err_deg = getattr(
                                phase.outer, "last_joint_err_deg", None
                            )
                            if joint_err_deg is None:
                                joint_err_deg = _joint_plan_err_deg(
                                    phase.outer, t_ref, q_meas
                                )
                            raw_scale = _reference_governor_scale(
                                phase,
                                outer_err_mm=outer_err_mm,
                                joint_err_deg=joint_err_deg,
                            )
                            scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6 and step.controller_mode != "single_qpik":
                            scale *= float(np.clip(t_wall / ramp_s, 0.0, 1.0))
                        t_ref += control_dt * scale
                        step.accepted_reference_lag_s = max(0.0, t_wall - t_ref)
    
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
                                dt_actual_s=dt_wall_actual,
                                sensor_age_s=sensor_age_s,
                                feedback_age_s=feedback_age_s,
                                feedback_fresh_tick=feedback_fresh_tick,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                            )
                        if on_step is not None:
                            on_step(phase.label, t_ref, step, pose_pin, f_ext, t_wall)

                        if phase.wait_until is not None:
                            n_wait = len(inspect.signature(phase.wait_until).parameters)
                            if n_wait >= 2:
                                phase_arrived = bool(phase.wait_until(pose_pin, q_meas))
                            else:
                                phase_arrived = bool(phase.wait_until(pose_pin))
                            if arrival_gate.update(
                                geometric_arrival=phase_arrived,
                                t_ref_s=t_ref,
                                qdot_applied=step.qdot,
                                dt_s=control_dt,
                                rail_settled=_rail_settled_for_arrival(
                                    rail_bridge,
                                    speed_limit_m_s=phase.arrival_rail_speed_m_s,
                                    now_s=time.monotonic(),
                                    freshness_s=inner.cfg.feedback_timeout_s,
                                ),
                            ):
                                phase_arrived = True
                                break
                            phase_arrived = False
    
                        ticks += 1
                        next_tick += dt
                        _wait_until(next_tick)

                    if phase.on_exit is not None:
                        phase.on_exit()

                    if phase_stopped:
                        break

                    if phase.require_arrival and not phase_arrived:
                        stop_reason = f"arrival_timeout:{phase.label or phase_idx}"
                        phase_stopped = True
                        _fault_stop(stop_reason)
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
                            f"— safety stop",
                            flush=True,
                        )
                        break
            except KeyboardInterrupt:
                if verbose:
                    print("\nStopped.", flush=True)
        finally:
            inner.set_direct_joint_ptp(False)
            inner.set_plan_drives_rail(False)
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

"""Joint-space inner loop: Cartesian twist -> absolute joint angles (rm_movej_canfd).

``JointIkController``: hardware-free WBC slack-QP IK + safety clamp (no send-path LPF).
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

from rm75_control.control.admittance_common.async_state import arm_qdot_rad_s_from_snap
from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.ik_types import saturate_error
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
from rm75_control.control.joint_admittance_8dof.solver.qp_builder import QpConfig, QpIkController
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
from rm75_control.control.joint_admittance_8dof.tasks.ird_adapter import (
    IrdConfig,
    try_load_ird,
)
from rm75_control.control.joint_admittance_8dof.tasks.psi_retarget import (
    PostureRetarget,
    PsiRetargetConfig,
    design_family_ok,
    fold_psi_to_positive,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_lock import (
    RailLockConfig,
    RailLockTask,
)
from rm75_control.control.joint_admittance_8dof.tasks.rail_mode import (
    LockedStyle,
    RailMode,
)
from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import SecondaryComposer
from rm75_control.control.joint_admittance_8dof.utils.safety import (
    SafetyLimiter,
    SafetyLimits,
    Watchdog,
)

# Below this, QPIK rail speed is treated as idle (publish 0, no 20 mm lead).
_RAIL_V_IDLE_M_S = 1.0e-3


# ---------------------------------------------------------------------------
# Inner loop (hardware-free)
# ---------------------------------------------------------------------------
@dataclass
class JointIkConfig:
    dt: float = 0.005
    control_frame: str = "tool"
    euler_order: str = "xyz"
    qp: QpConfig = field(default_factory=QpConfig)
    nullspace: NullspaceTaskConfig = field(default_factory=NullspaceTaskConfig)
    manipulability: ManipulabilityTaskConfig = field(default_factory=ManipulabilityTaskConfig)
    arm_angle: ArmAngleTaskConfig = field(default_factory=ArmAngleTaskConfig)
    psi_retarget: PsiRetargetConfig = field(default_factory=PsiRetargetConfig)
    ird: IrdConfig = field(default_factory=IrdConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    rail: RailLockConfig = field(default_factory=RailLockConfig)
    rail_extension: RailExtensionConfig = field(default_factory=RailExtensionConfig)
    v_scale: float = 0.5
    a_max_arm_rad_s2: float = 20.0
    a_max_rail_m_s2: float = 0.30
    position_margin_rad: float = 0.017
    position_margin_rail_m: float = 0.0
    resync_err_rad: float = 0.10
    resync_err_rail_m: float = 0.020
    feedback_timeout_s: float = 0.050
    nullspace_d_null: float = 0.0
    nullspace_d_null_adaptive: float = 1.0
    nullspace_max_qdot_frac: float = 0.2


@dataclass
class JointIkStep:
    q_send: np.ndarray
    qdot: np.ndarray
    twist_base: np.ndarray
    sigma_min: float
    manip: float
    slack_norm: float
    n_cbf_active: int
    follow_err_rad: float
    cart_err_mm: float = 0.0
    qdot_ff_norm: float = 0.0
    vel_clamped: bool = False
    acc_clamped: bool = False
    pos_clamped: bool = False
    tcp_jump_mm: float = 0.0
    rail_vel_pin: float = float("nan")
    rail_qdot_ff: float = float("nan")
    plan_drives_rail: bool = False
    arm_singularity_smooth: float = 1.0
    limit_activation: float = 0.0
    rail_ext_err_m: float = 0.0
    rail_ext_weight: float = 0.0
    rail_escape_active: bool = False
    psi_deg: float = float("nan")
    psi_ref_deg: float = float("nan")
    psi_retarget_score: float = float("nan")
    d_pref_m: float = float("nan")
    elbow_margin_rad: float = float("nan")
    wrist_open_rad: float = float("nan")
    family_ok: bool = True
    physical_saturated: bool = False
    rail_contrib_m_s: float = float("nan")
    arm_contrib_m_s: float = float("nan")
    rail_motion_share: float = float("nan")
    wln_scale_rail: float = float("nan")
    wln_scale_arm_max: float = float("nan")
    waste_ratio: float = float("nan")
    rail_ff_m: float = float("nan")
    rail_track_err_m: float = float("nan")
    d_star_m: float = float("nan")
    psi_star_deg: float = float("nan")
    minmax_margin: float = float("nan")
    controller_mode: str = "qpik"
    qp_backend: str = ""
    qp_solver_status: str = "not_run"
    qp_solver_iterations: int = 0
    qp_solver_solve_ms: float = 0.0
    qp_solver_call_count: int = 0
    qp_solver_overrun: bool = False
    qp1_status: str = "not_run"
    qp2_status: str = "not_run"
    qp1_solve_ms: float = 0.0
    qp2_solve_ms: float = 0.0
    qp_assembly_ms: float = 0.0
    qp_fallback_ms: float = 0.0
    qpik_total_ms: float = 0.0
    qp2_fallback: bool = False
    # Coarse per-stage tick profile (ms).  The loop budgets 5.0 ms but
    # measured 6.16 ms mean with only 2.1% of ticks on time, and the log had
    # no way to attribute the overrun.
    tick_inner_ms: float = float("nan")
    tick_send_ms: float = float("nan")
    tick_log_ms: float = float("nan")
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
    rail_decomposition_error: float = 0.0
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
    rail_sat: bool = False
    rail_exec_velocity_m_s: float = float("nan")
    rail_measured_velocity_m_s: float = float("nan")
    rail_commanded_velocity_m_s: float = float("nan")
    rail_commanded_acceleration_m_s2: float = float("nan")
    rail_feedback_age_s: float = float("nan")
    a_mirror_frac: float = float("nan")
    j_mirror_frac: float = float("nan")
    last_limit_saturated: bool = False
    keep_task_weight: bool = False
    pref_slack_scale: float = 1.0
    rail_task_vel: float = float("nan")
    v_escape: float = float("nan")
    v_reach: float = float("nan")
    v_ff_rail: float = float("nan")
    sigma_arm: float = float("nan")
    sns_scale: float = 1.0
    qdot_meas: np.ndarray = field(default_factory=lambda: np.full(8, np.nan))
    v_cmd: np.ndarray = field(default_factory=lambda: np.zeros(6))
    path_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    feedback_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    comfort_slack: np.ndarray = field(default_factory=lambda: np.zeros(7))
    cbf_min_dist: float = float("nan")
    cbf_pair: str = ""
    nullspace_norm: float = float("nan")


def scale_qdot_into_box(
    qdot: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Uniform task-scaling into [lo, hi]: preserve Cartesian direction.

    Per-joint ``np.clip`` would break the twist direction (Flacco SNS vs naive
    saturation).  A single ``s ∈ [0, 1]`` is applied to the whole vector.
    """
    qdot = np.asarray(qdot, dtype=float).reshape(-1).copy()
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    if qdot.shape != lo.shape or qdot.shape != hi.shape:
        return np.clip(qdot, lo, hi) if lo.shape == qdot.shape else qdot
    s = 1.0
    eps = 1.0e-12
    for i, v in enumerate(qdot):
        if v > hi[i] + eps:
            if v > eps:
                s = min(s, float(hi[i] / v))
            else:
                s = 0.0
        elif v < lo[i] - eps:
            if v < -eps:
                s = min(s, float(lo[i] / v))
            else:
                s = 0.0
    s = float(np.clip(s, 0.0, 1.0))
    return qdot * s


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
        self.cfg.qp.euler_order = self.cfg.euler_order
        self.cfg.qp.collision = self.cfg.collision
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
        if self.rail_ext_task is not None:
            self.rail_ext_task.cfg.soft_min_m = float(self.cfg.rail.hard_min_m)
            self.rail_ext_task.cfg.soft_max_m = float(self.cfg.rail.hard_max_m)
        self.posture_retarget = (
            PostureRetarget(kin, self.cfg.psi_retarget, euler_order=self.cfg.euler_order)
            if self.cfg.psi_retarget.enabled
            else None
        )
        self._rail_ext_active = True
        from rm75_control.control.joint_admittance_8dof.tasks.rail_goodness import (
            CachedRailGoodness,
            SigmaMinGoodness,
        )

        ird_cfg = self.cfg.ird if self.cfg.ird is not None else IrdConfig()
        self._ird = (
            try_load_ird(ird_cfg) if bool(getattr(ird_cfg, "enabled", False)) else None
        )
        if self.posture_retarget is not None:
            self.posture_retarget._ird = self._ird
        # IRD is one-shot d* at plan_scan_stroke only.  Autograd goodness
        # on this thread caused 127 ms hitches → servo FA24=0.
        self._rail_goodness = CachedRailGoodness(
            SigmaMinGoodness(kin), period_ticks=10
        )
        self._sigma_grad_rail_cached: float = 0.0
        a_max_vec = np.full(kin.nv, float(self.cfg.a_max_arm_rad_s2))
        a_max_vec[0] = float(self.cfg.a_max_rail_m_s2)
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
                float(self.kin.v_max[0]),
                float(self.cfg.rail.v_max_m_s),
            )
        hard_lo = float(getattr(self.cfg.rail, "hard_min_m", 0.005))
        hard_hi = float(getattr(self.cfg.rail, "hard_max_m", 0.78))
        if not (
            np.isfinite(hard_lo)
            and np.isfinite(hard_hi)
            and float(self.kin.q_lower[0]) <= hard_lo < hard_hi
            and hard_hi <= float(self.kin.q_upper[0])
        ):
            raise ValueError(
                "invalid rail hard limits: "
                f"[{hard_lo:.6f}, {hard_hi:.6f}]"
            )
        self.limits.q_lower[0] = max(float(self.limits.q_lower[0]), hard_lo)
        self.limits.q_upper[0] = min(float(self.limits.q_upper[0]), hard_hi)
        self.limits.rail_soft_min_m = float(self.cfg.rail.soft_min_m)
        self.limits.rail_soft_max_m = float(self.cfg.rail.soft_max_m)
        self.core = QpIkController(self.kin, self.limits, self.cfg.qp)
        self.core.set_q_star(self.centering_task.q_target)
        self.safety = SafetyLimiter(self.limits)
        self.q_cmd = np.zeros(kin.nv, dtype=float)
        self._arm_task_suppressed = False
        self._centering_suppressed = False
        self._manipulability_active = False
        self._twist_scale_filt = 1.0
        self._box_dt_last_t: float | None = None
        self._box_h1_last: float | None = None
        self._rail_dv_filt: float = 0.0
        self._rail_dv_tau_s: float = 0.025
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
        self.last_arm_rho: float = float("nan")
        self._press_z_mark: float = float("nan")
        self._press_stall_s: float = 0.0
        self._d_star_nudge_cool_s: float = 0.0
        self._family_ok: bool = True
        self._rail_mode: RailMode = self.cfg.rail.mode
        self._locked_style: LockedStyle = self.cfg.rail.locked_style
        self._configured_rail_mode: RailMode = self.cfg.rail.mode
        self._plan_drives_rail: bool = False
        self._direct_joint_ptp: bool = False
        self._apply_rail_mode_side_effects()

    @property
    def rail_mode(self) -> RailMode:
        return self._rail_mode

    def set_plan_drives_rail(self, enabled: bool) -> None:
        self._plan_drives_rail = bool(enabled)

    def set_direct_joint_ptp(self, enabled: bool) -> None:
        self._direct_joint_ptp = bool(enabled)

    @property
    def configured_rail_mode(self) -> RailMode:
        return self._configured_rail_mode

    @property
    def locked_style(self) -> LockedStyle:
        return self._locked_style

    @property
    def is_locked_hold(self) -> bool:
        return (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.HOLD
        )

    def set_arm_task_suppressed(self, suppressed: bool) -> None:
        self._arm_task_suppressed = bool(suppressed)

    def set_centering_suppressed(self, suppressed: bool) -> None:
        self._centering_suppressed = bool(suppressed)

    def set_manipulability_active(self, active: bool) -> None:
        self._manipulability_active = bool(active) and self.manipulability_task is not None

    def set_rail_extension_active(self, active: bool) -> None:
        self._rail_ext_active = bool(active)

    def set_rail_extension_mode(self, mode: str) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_mode(mode)  # type: ignore[arg-type]

    def set_rail_pose_target(self, y_rail_m: float | None) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_rail_pose_target(y_rail_m)

    def capture_rail_extension_ref(self) -> None:
        if self.rail_ext_task is not None:
            self.rail_ext_task.capture_reference(self.q_cmd)

    def _measure_box_dt(self, dt: float) -> float:
        """Backward-compatible single-period wall dt (``h1``)."""
        h1, _h2 = self._measure_box_periods(dt)
        return h1

    def _measure_box_periods(self, dt: float) -> tuple[float, float | None]:
        """Two most recent wall periods for the unequal-sample third-order box.

        Each period is clamped to ``[0.8, 2.0] × dt`` so one stalled tick
        cannot open the acceleration/jerk boxes.
        """
        now = time.monotonic()
        prev = self._box_dt_last_t
        prev_h1 = self._box_h1_last
        self._box_dt_last_t = now
        nominal = max(float(dt), 1.0e-6)
        if prev is None:
            self._box_h1_last = nominal
            return nominal, None
        measured = now - prev
        if not math.isfinite(measured) or measured <= 0.0:
            h1 = nominal
        else:
            h1 = float(np.clip(measured, 0.8 * nominal, 2.0 * nominal))
        self._box_h1_last = h1
        return h1, prev_h1

    def plan_scan_stroke(
        self,
        y_center_m: float,
        amplitude_m: float,
        q_rad: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """One-shot min-max (d*, ψ*) at scan start.  Raises if infeasible."""
        q = self.q_cmd if q_rad is None else np.asarray(q_rad, dtype=float)
        if self.posture_retarget is None:
            y_tcp = float(self.kin.fk_placement(q).translation[1])
            d_star = y_tcp - float(q[0])
            if self.rail_ext_task is not None:
                self.rail_ext_task.set_d_pref(d_star)
            return d_star, float("nan")
        d_star, psi_star = self.posture_retarget.plan_stroke(
            q,
            y_center_m=float(y_center_m),
            amplitude_m=float(amplitude_m),
            rail_lo=float(self.limits.q_lower[0]),
            rail_hi=float(self.limits.q_upper[0]),
        )
        if self.arm_task is not None:
            self.arm_task.set_reference(float(psi_star))
        if self.rail_ext_task is not None:
            self.rail_ext_task.set_d_pref(float(d_star))
        return float(d_star), float(psi_star)

    def _check_design_family(self, q_meas: np.ndarray) -> None:
        qn = np.asarray(self.centering_task._q_target_default, dtype=float)
        ok = design_family_ok(q_meas, qn)
        self._family_ok = bool(ok)
        if ok:
            return
        from rm75_control.kinematics.srs_ik import branch_from_q, psi_from_q

        q = np.asarray(q_meas, dtype=float).reshape(-1)
        psi_m = math.degrees(fold_psi_to_positive(psi_from_q(q)))
        psi_n = math.degrees(fold_psi_to_positive(psi_from_q(qn)))
        msg = (
            "DESIGN FAMILY MISMATCH: measured "
            f"ψ={psi_m:.1f}° branch={int(branch_from_q(q))} "
            f"J1={math.degrees(float(q[1])):+.1f}° vs design "
            f"ψ={psi_n:.1f}° branch={int(branch_from_q(qn))} "
            f"J1={math.degrees(float(qn[1])):+.1f}°"
        )
        print(f"[joint_ik] {msg}", flush=True)
        if bool(getattr(self.cfg.psi_retarget, "require_design_family", False)):
            raise ValueError(msg)

    def _latch_attractor_from_q(self, q_meas: np.ndarray) -> None:
        """Keep yaml ``q_nominal`` signs for both centering and q*.

        Mirroring J1 from a planar start (J1≈0⁺) made q*=+90° and the
        branch barrier then forbade the design fold to J1=−90°.
        """
        q = np.asarray(q_meas, dtype=float).reshape(-1)
        if q.size != self.kin.nv or not np.all(np.isfinite(q)):
            return
        q_nominal = np.asarray(self.centering_task._q_target_default, dtype=float)
        self.centering_task.set_q_target(q_nominal)
        self.core.set_q_star(q_nominal.copy())
        if self.arm_task is not None:
            self.arm_task.reset(q)
        self._check_design_family(q)

    def reset(self, q0_rad: np.ndarray) -> None:
        self.q_cmd = np.asarray(q0_rad, dtype=float).copy()
        self.core.reset()
        self.safety.reset(self.q_cmd)
        if self.arm_task is not None:
            self.arm_task.reset(self.q_cmd)
        self.rail_task.reset(self.q_cmd)
        if self.rail_ext_task is not None:
            self.rail_ext_task.reset(self.q_cmd)
        if self.posture_retarget is not None:
            self.posture_retarget.reset(self.q_cmd)
            if self.rail_ext_task is not None and np.isfinite(
                self.posture_retarget.d_star_m
            ):
                self.rail_ext_task.set_d_pref(float(self.posture_retarget.d_star_m))
            if self.arm_task is not None and self.posture_retarget._psi_cmd is not None:
                self.arm_task.set_reference(float(self.posture_retarget._psi_cmd))
        self._twist_scale_filt = 1.0
        self._box_dt_last_t = None
        self._box_h1_last = None
        self._rail_dv_filt = 0.0
        self._direct_joint_ptp = False
        self._plan_drives_rail = False
        self._press_z_mark = float("nan")
        self._press_stall_s = 0.0
        self._d_star_nudge_cool_s = 0.0
        self._apply_rail_mode_side_effects()
        self._latch_attractor_from_q(self.q_cmd)

    def begin_hybrid_episode(
        self,
        q_meas: np.ndarray,
        qdot_applied: np.ndarray | None = None,
    ) -> None:
        """Preserve velocity continuity and latch yaml q* branch signs.

        ``q_nominal`` keeps magnitudes from yaml; signs follow the measured
        pose so the attractor never pulls through 0 to the reverse branch.
        """
        applied = self.core.qdot_prev if qdot_applied is None else qdot_applied
        self.core.sync_applied(applied)
        self._latch_attractor_from_q(q_meas)

    def set_rail_mode(
        self,
        mode: RailMode | str,
        *,
        q_ref_m: float | None = None,
        locked_style: LockedStyle | str | None = None,
    ) -> None:
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
            self.rail_task.set_reference(q_ref_m)
        elif mode == RailMode.LOCKED and self._locked_style == LockedStyle.HOLD:
            self.rail_task.reset(self.q_cmd)
        self._apply_rail_mode_side_effects()

    def set_coupled(self) -> None:
        self.set_rail_mode(RailMode.COUPLED)

    def set_locked(
        self,
        style: LockedStyle | str = LockedStyle.HOLD,
        *,
        q_ref_m: float | None = None,
    ) -> None:
        self.set_rail_mode(RailMode.LOCKED, q_ref_m=q_ref_m, locked_style=style)

    def _apply_rail_mode_side_effects(self) -> None:
        self.rail_task.cfg.mode = self._rail_mode
        self.rail_task.cfg.locked_style = self._locked_style
        self.cfg.rail.mode = self._rail_mode
        self.cfg.rail.locked_style = self._locked_style

    def _pin_rail_if_locked_hold(self) -> None:
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
    ) -> np.ndarray:
        qdot0 = self.secondary.compose(
            q,
            qdot_ff,
            self.core.qdot_prev,
            arm_suppressed=self._arm_task_suppressed,
            sigma_min=self.last_sigma_min,
            sigma_ref=self.cfg.qp.sr_damping.sigma_ref,
            centering_suppressed=self._centering_suppressed,
            centering_sigma_fade=centering_sigma_fade,
            manipulability_active=(
                self._manipulability_active
                if manipulability_active is None
                else manipulability_active
            ),
        )
        self.last_secondary_norm = float(np.linalg.norm(qdot0))
        return qdot0

    def _clip_qdot_to_box(
        self,
        q_prev: np.ndarray,
        qdot: np.ndarray,
        dt: float,
        q_meas: np.ndarray | None,
        resync_vec: np.ndarray,
        *,
        rail_locked: bool,
        rail_vel_pin: float | None,
        box_h1: float | None = None,
        box_h2: float | None = None,
        rail_lead_exempt: bool = False,
    ) -> np.ndarray:
        qdot = np.asarray(qdot, dtype=float).reshape(-1).copy()
        if qdot.shape != q_prev.shape or not np.all(np.isfinite(qdot)):
            qdot = np.zeros_like(q_prev)
        q_geom = q_meas if q_meas is not None else q_prev
        lo, hi = self.core.constraints.bounds(
            q_geom,
            dt,
            self.core.qdot_prev,
            q_meas=q_meas,
            q_cmd=q_prev,
            resync_err=resync_vec,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin,
            qdot_prev2=self.core.qdot_prev2,
            j_max=self.core._j_max,
            box_h1=box_h1,
            box_h2=box_h2,
            rail_lead_exempt=rail_lead_exempt,
        )
        return scale_qdot_into_box(qdot, lo, hi)

    def _make_step(
        self,
        *,
        qdot: np.ndarray,
        twist_base: np.ndarray,
        sigma_min: float,
        manip: float,
        slack_norm: float,
        n_cbf_active: int,
        follow_err: float,
        qdot_ff_norm: float,
        vel_clamped: bool = False,
        acc_clamped: bool = False,
        pos_clamped: bool = False,
        rail_vel_pin: float | None = None,
        rail_qdot_ff: float = float("nan"),
        plan_drives_rail: bool = False,
        rail_ext_err_m: float = 0.0,
        rail_ext_weight: float = 0.0,
        mode: str = "qpik",
        failed: bool = False,
        fallback_reason: str = "",
        rail_macro_pref_v: float = 0.0,
        rail_escape_active: bool = False,
        rail_contrib_m_s: float = float("nan"),
        arm_contrib_m_s: float = float("nan"),
        rail_motion_share: float = float("nan"),
        scan_target: np.ndarray | None = None,
        scan_achieved: np.ndarray | None = None,
        scan_residual: np.ndarray | None = None,
        physical_saturated: bool = False,
    ) -> JointIkStep:
        slack = float(slack_norm)
        alpha = 0.0 if failed else float(np.clip(1.0 - slack, 0.0, 1.0))
        qp_total_ms = float(getattr(self.core, "last_qp_total_ms", 0.0))
        qp2_fallback = bool(getattr(self.core, "last_qp2_fallback", False))
        return JointIkStep(
            q_send=self.q_cmd.copy(),
            qdot=np.asarray(qdot, dtype=float).copy(),
            twist_base=np.asarray(twist_base, dtype=float).copy(),
            sigma_min=float(sigma_min),
            manip=float(manip),
            slack_norm=slack,
            n_cbf_active=int(n_cbf_active),
            follow_err_rad=float(follow_err),
            qdot_ff_norm=float(qdot_ff_norm),
            vel_clamped=bool(vel_clamped),
            acc_clamped=bool(acc_clamped),
            pos_clamped=bool(pos_clamped),
            rail_vel_pin=(
                float(rail_vel_pin) if rail_vel_pin is not None else float("nan")
            ),
            rail_qdot_ff=float(rail_qdot_ff),
            plan_drives_rail=bool(plan_drives_rail),
            arm_singularity_smooth=self.secondary.last_arm_smooth,
            limit_activation=self.secondary.last_limit_activation,
            rail_ext_err_m=float(rail_ext_err_m),
            rail_ext_weight=float(rail_ext_weight),
            rail_escape_active=bool(rail_escape_active),
            psi_deg=(
                float(np.degrees(self.arm_task.arm_angle(self.q_cmd)))
                if self.arm_task is not None
                else float("nan")
            ),
            psi_ref_deg=(
                float(np.degrees(self.arm_task.psi_ref))
                if self.arm_task is not None and self.arm_task.psi_ref is not None
                else float("nan")
            ),
            psi_retarget_score=(
                float(self.posture_retarget.last_psi_score)
                if self.posture_retarget is not None
                else float("nan")
            ),
            d_pref_m=(
                float(self.rail_ext_task.d_pref_m)
                if self.rail_ext_task is not None and self.rail_ext_task.d_pref_m is not None
                else float("nan")
            ),
            elbow_margin_rad=(
                float(self.posture_retarget.last_elbow_margin_rad)
                if self.posture_retarget is not None
                else float("nan")
            ),
            wrist_open_rad=(
                float(self.posture_retarget.last_wrist_open_rad)
                if self.posture_retarget is not None
                else float("nan")
            ),
            family_ok=bool(self._family_ok),
            physical_saturated=bool(physical_saturated),
            rail_contrib_m_s=float(rail_contrib_m_s),
            arm_contrib_m_s=float(arm_contrib_m_s),
            rail_motion_share=float(rail_motion_share),
            wln_scale_rail=float(self.core.last_wln_scale[0]),
            wln_scale_arm_max=float(np.max(self.core.last_wln_scale[1:])),
            waste_ratio=(
                (abs(float(rail_contrib_m_s)) + abs(float(arm_contrib_m_s)))
                / max(abs(float(rail_contrib_m_s) + float(arm_contrib_m_s)), 1.0e-9)
                if np.isfinite(rail_contrib_m_s) and np.isfinite(arm_contrib_m_s)
                else float("nan")
            ),
            rail_ff_m=(
                float(getattr(self.rail_ext_task, "last_rail_ff_m", float("nan")))
                if self.rail_ext_task is not None
                else float("nan")
            ),
            rail_track_err_m=(
                float(getattr(self.rail_ext_task, "last_track_err_m", float("nan")))
                if self.rail_ext_task is not None
                else float("nan")
            ),
            d_star_m=(
                float(self.posture_retarget.d_star_m)
                if self.posture_retarget is not None
                else float("nan")
            ),
            psi_star_deg=(
                float(np.degrees(self.posture_retarget.psi_star_rad))
                if self.posture_retarget is not None
                and np.isfinite(self.posture_retarget.psi_star_rad)
                else float("nan")
            ),
            minmax_margin=(
                float(self.posture_retarget.last_minmax_margin)
                if self.posture_retarget is not None
                else float("nan")
            ),
            controller_mode=mode,
            qp_backend=self.core.backend_name,
            qp_solver_status=self.core.last_status if mode == "qpik" else "not_run",
            qp_solver_call_count=int(self.core.solve_count) if mode == "qpik" else 0,
            qp_solver_solve_ms=qp_total_ms if mode == "qpik" else 0.0,
            qp_solver_overrun=bool(
                mode == "qpik"
                and qp_total_ms > float(getattr(self.cfg.qp, "max_solve_ms", 5.0))
            ),
            qp1_status=(
                str(getattr(self.core, "last_qp1_status", "not_run"))
                if mode == "qpik"
                else "not_run"
            ),
            qp2_status=(
                str(getattr(self.core, "last_qp2_status", "not_run"))
                if mode == "qpik"
                else "not_run"
            ),
            qp1_solve_ms=(
                float(getattr(self.core, "last_qp1_solve_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qp2_solve_ms=(
                float(getattr(self.core, "last_qp2_solve_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qp_assembly_ms=(
                max(
                    qp_total_ms
                    - float(getattr(self.core, "last_qp1_solve_ms", 0.0))
                    - float(getattr(self.core, "last_qp2_solve_ms", 0.0))
                    - float(getattr(self.core, "last_fallback_ms", 0.0)),
                    0.0,
                )
                if mode == "qpik"
                else 0.0
            ),
            qp_fallback_ms=(
                float(getattr(self.core, "last_fallback_ms", 0.0))
                if mode == "qpik"
                else 0.0
            ),
            qpik_total_ms=qp_total_ms if mode == "qpik" else 0.0,
            qp2_fallback=qp2_fallback if mode == "qpik" else False,
            qpik_alpha=alpha,
            qpik_beta=1.0,
            qpik_authority=1.0,
            qpik_hard_residual_max=0.0,
            qpik_dexterity_slack=float(getattr(self.core, "last_dexterity_slack", 0.0)),
            qpik_branch_slack=float(getattr(self.core, "last_branch_slack", 0.0)),
            rail_macro_pref_v=float(rail_macro_pref_v),
            rail_decomposition_error=0.0,
            scan_target=(
                np.asarray(scan_target, dtype=float).copy()
                if scan_target is not None
                else np.zeros(2)
            ),
            scan_achieved=(
                np.asarray(scan_achieved, dtype=float).copy()
                if scan_achieved is not None
                else np.zeros(2)
            ),
            scan_residual=(
                np.asarray(scan_residual, dtype=float).copy()
                if scan_residual is not None
                else np.zeros(2)
            ),
            fallback_level="stop" if failed else "none",
            fallback_reason=fallback_reason,
            solver_fault_latched=bool(mode == "qpik" and failed),
            arm_health=float(sigma_min),
            a_mirror_frac=float(getattr(self.core, "last_a_mirror_frac", float("nan"))),
            j_mirror_frac=float(getattr(self.core, "last_j_mirror_frac", float("nan"))),
        )

    def update(
        self,
        twist: np.ndarray,
        dt: float | None = None,
        q_meas: np.ndarray | None = None,
        qdot_ff: np.ndarray | None = None,
        *,
        vel_ff: np.ndarray | None = None,
        pose_d: np.ndarray | None = None,
        f_ext_z: float | None = None,
        f_des_z: float | None = None,
        contact_active: bool = False,
        task_rotation_base: np.ndarray | None = None,
        task_safety_rows: tuple = (),
        path_twist: np.ndarray | None = None,
        feedback_twist: np.ndarray | None = None,
        v_force_z: float | None = None,
        rail_exec_vel_m_s: float | None = None,
        rail_exec_smooth_m_s: float | None = None,
    ) -> JointIkStep:
        del f_ext_z, f_des_z, contact_active, task_safety_rows
        path_twist_arr = (
            np.asarray(path_twist, dtype=float).reshape(6)
            if path_twist is not None
            else np.zeros(6)
        )
        feedback_twist_arr = (
            np.asarray(feedback_twist, dtype=float).reshape(6)
            if feedback_twist is not None
            else np.zeros(6)
        )
        dt = self.cfg.dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
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
        if rail_exec_vel_m_s is not None and not np.isfinite(float(rail_exec_vel_m_s)):
            raise ValueError("rail_exec_vel_m_s must be finite when supplied")
        if rail_exec_smooth_m_s is not None and not np.isfinite(
            float(rail_exec_smooth_m_s)
        ):
            raise ValueError("rail_exec_smooth_m_s must be finite when supplied")
        # Hardware supplies the time-stamped worker estimate.  Offline callers
        # have no independent actuator, so the last applied rail command is
        # the least-surprising zero-order execution estimate.
        if rail_exec_vel_m_s is not None and rail_exec_smooth_m_s is not None:
            delta_v = float(rail_exec_vel_m_s) - float(rail_exec_smooth_m_s)
            tau = max(float(self._rail_dv_tau_s), 1.0e-6)
            alpha = min(1.0, float(dt) / tau)
            self._rail_dv_filt += alpha * (delta_v - self._rail_dv_filt)
            rail_exec_for_qp = float(rail_exec_smooth_m_s) + float(self._rail_dv_filt)
        elif rail_exec_vel_m_s is not None:
            rail_exec_for_qp = float(rail_exec_vel_m_s)
        else:
            rail_exec_for_qp = float(self.core.qdot_prev[0])

        if task_rotation_base is not None:
            rotation_base_task = np.asarray(task_rotation_base, dtype=float)
            twist_base = np.concatenate(
                (
                    rotation_base_task @ twist_task[:3],
                    rotation_base_task @ twist_task[3:],
                )
            )
        else:
            twist_base = self._twist_to_base(twist_task, q_state)

        J_pre = self.kin.jacobian(q_state)
        sigma_values_pre = self.kin.singular_values(J_pre)
        mass_pre = (
            self.kin.mass_matrix(q_state)
            if self.cfg.qp.use_dyn_nullspace or self.cfg.qp.use_mass_weighted_reg
            else None
        )
        sigma_pre = float(sigma_values_pre.min())
        sigma_arm = float(np.linalg.svd(J_pre[:, 1:], compute_uv=False).min())
        sigma_ref = float(self.cfg.qp.sr_damping.sigma_ref)
        # The protected QP target is exactly the outer-loop v_cmd.  Sigma is
        # still used for SR/null-space conditioning and diagnostics, but it
        # must not silently scale the Cartesian command before QP1.
        self._twist_scale_filt = 1.0

        locked_hold = self.is_locked_hold
        rail_only = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.RAIL_ONLY
        )
        tcp_fixed = (
            self._rail_mode == RailMode.LOCKED
            and self._locked_style == LockedStyle.TCP_FIXED
        )
        if qdot_ff is not None:
            v_lim_ff = np.asarray(self.safety.lim.v_max, dtype=float)
            qdot_ff = np.clip(np.asarray(qdot_ff, dtype=float), -v_lim_ff, v_lim_ff)

        if self._direct_joint_ptp and qdot_ff is not None:
            qdot_cmd = np.asarray(qdot_ff, dtype=float).copy()
            if rail_only:
                qdot_cmd[1:] = 0.0
            q_next = q_prev + qdot_cmd * dt
            rep = self.safety.clamp(q_prev, q_next, dt)
            self.q_cmd = rep.q_safe
            if dt > 1e-9:
                applied = (self.q_cmd - q_prev) / dt
            else:
                applied = qdot_cmd
            self.core.sync_applied(applied)
            self.last_sigma_min = sigma_pre
            J = J_pre
            sigma = sigma_values_pre
            return self._make_step(
                qdot=applied,
                twist_base=twist_base,
                sigma_min=float(sigma.min()),
                manip=float(np.prod(sigma)),
                slack_norm=0.0,
                n_cbf_active=0,
                follow_err=follow_err,
                qdot_ff_norm=float(np.linalg.norm(qdot_ff)),
                vel_clamped=rep.vel_clamped,
                acc_clamped=rep.acc_clamped,
                pos_clamped=rep.pos_clamped,
                rail_vel_pin=float(qdot_ff[0]),
                rail_qdot_ff=float(qdot_ff[0]),
                plan_drives_rail=True,
                mode="direct_joint_ptp",
            )

        plan_drives_rail = rail_only or tcp_fixed or bool(self._plan_drives_rail)
        qdot_ff_sec = qdot_ff
        rail_vel_pin: float | None = None
        rail_qdot_ff_val = float("nan")
        if qdot_ff is not None:
            qdot_ff_arr = np.asarray(qdot_ff, dtype=float)
            v_rail = float(qdot_ff_arr[0])
            rail_qdot_ff_val = v_rail
            qdot_ff_sec = qdot_ff_arr.copy()
            qdot_ff_sec[0] = 0.0
            if plan_drives_rail:
                rail_vel_pin = v_rail

        resync_vec = np.full(self.kin.nv, float(self.cfg.resync_err_rad))
        resync_vec[0] = float(self.cfg.resync_err_rail_m)

        pose_now = self.kin.fk_pose(q_prev)
        z_now = float(pose_now[2])
        y_tcp_d = None
        tool_y_err_m = 0.0
        if pose_d is not None:
            pose_d_arr = np.asarray(pose_d, dtype=float).reshape(-1)
            if pose_d_arr.size >= 2 and np.isfinite(pose_d_arr[1]):
                y_tcp_d = float(pose_d_arr[1])
                tool_y_err_m = y_tcp_d - float(pose_now[1])

        ext_cfg = self.rail_ext_task.cfg if self.rail_ext_task is not None else None
        v_force = (
            float(v_force_z)
            if v_force_z is not None and np.isfinite(float(v_force_z))
            else float("nan")
        )
        v_min = float(getattr(ext_cfg, "press_v_force_min_m_s", 0.02))
        dz_max = float(getattr(ext_cfg, "press_dz_max_m", 0.002))
        y_thr = float(getattr(ext_cfg, "press_y_err_m", 0.005))
        stall_need = float(getattr(ext_cfg, "press_stall_s", 0.5))
        v_z_demand = (
            v_force if np.isfinite(v_force) else float(twist_base[2])
        )
        demanding = bool(abs(v_z_demand) >= v_min)
        # Windowed stall: |Δz| over the timer, not per 5 ms tick (2 mm/tick
        # is 400 mm/s — every real press looked "stuck").
        if demanding:
            if not np.isfinite(self._press_z_mark):
                self._press_z_mark = z_now
            z_progress = abs(z_now - self._press_z_mark)
            if z_progress > dz_max:
                self._press_z_mark = z_now
                self._press_stall_s = 0.0
                z_stuck = False
            else:
                self._press_stall_s += float(dt)
                z_stuck = True
        else:
            self._press_z_mark = float("nan")
            self._press_stall_s = 0.0
            z_stuck = False
        press_stalled_timer = self._press_stall_s + 1.0e-12 >= stall_need
        in_band = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._rail_in_limit_band(float(q_prev[0]))
        )
        in_pin = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._rail_in_pin_band(float(q_prev[0]))
        )
        has_travel = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._rail_has_open_travel(float(q_prev[0]))
        )
        policy_leave = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task._in_leave_band(
                float(q_prev[0]), self.rail_ext_task._policy_escape_sign()
            )
        )
        arm_starved = bool(abs(tool_y_err_m) >= y_thr)
        comfort_m = 0.26
        j4_blocked = bool(
            (float(self.limits.q_upper[4]) - float(q_prev[4])) <= comfort_m
            or (float(q_prev[4]) - float(self.limits.q_lower[4])) <= comfort_m
        )
        allow_press_escape = bool(
            demanding
            and has_travel
            and (
                press_stalled_timer
                or (j4_blocked and not policy_leave)
                or (policy_leave and arm_starved)
            )
        )

        hold_d_star = False
        if vel_ff is not None:
            vff = np.asarray(vel_ff, dtype=float).reshape(-1)
            thr = (
                float(self.rail_ext_task.cfg.v_ff_thr_m_s)
                if self.rail_ext_task is not None
                else 0.01
            )
            hold_d_star = (
                vff.size >= 3 and float(np.linalg.norm(vff[:3])) > thr
            )

        if (
            self.posture_retarget is not None
            and self._rail_mode == RailMode.COUPLED
        ):
            psi_ref, d_pref = self.posture_retarget.step(
                q_prev,
                float(dt),
                rail_lo=float(self.limits.q_lower[0]),
                rail_hi=float(self.limits.q_upper[0]),
                hold_setpoint=hold_d_star,
            )
            if self.arm_task is not None:
                self.arm_task.set_reference(float(psi_ref))
            if self.rail_ext_task is not None:
                self.rail_ext_task.set_d_pref(float(d_pref))

        if (
            press_stalled_timer
            and allow_press_escape
            and self.posture_retarget is not None
            and self.rail_ext_task is not None
            and self._d_star_nudge_cool_s <= 0.0
        ):
            y_des = y_tcp_d if y_tcp_d is not None else float(pose_now[1])
            lo, hi = self.rail_ext_task._soft_travel()
            away = self.rail_ext_task._preferred_escape_sign(float(q_prev[0]))
            delta = -away * float(self.rail_ext_task.cfg.d_star_nudge_m)
            d_new = self.posture_retarget.nudge_d_star(
                delta, y_des_m=y_des, rail_lo=lo, rail_hi=hi
            )
            if np.isfinite(d_new):
                self.rail_ext_task.set_d_pref(float(d_new))
            self._d_star_nudge_cool_s = stall_need
        else:
            self._d_star_nudge_cool_s = max(
                0.0, self._d_star_nudge_cool_s - float(dt)
            )

        rail_task_vel: float | None = None
        rail_task_weight = 0.0
        rail_ext_err = 0.0
        rail_escape_active = False
        manip_for_saturation = self._manipulability_active
        if (
            self.rail_ext_task is not None
            and self._rail_ext_active
            and self._rail_mode == RailMode.COUPLED
        ):
            sigma_now = float(sigma_arm)
            sig_scale = 1.0
            sigma_esc_ref = max(
                sigma_ref,
                float(self.cfg.manipulability.sigma_fade_ref),
            )
            # No artificial floor: deep singularity must raise escape authority.
            if sigma_esc_ref > 1e-9 and sigma_now < sigma_esc_ref:
                sig_scale = max(sigma_now / sigma_esc_ref, 0.0)
            _g, self._sigma_grad_rail_cached = self._rail_goodness.refresh(q_prev)
            del _g
            # General near-limit metric from centering activation (any joint).
            from rm75_control.control.joint_admittance_8dof.tasks.secondary_composer import (
                max_limit_activation,
            )

            u_max = max_limit_activation(
                q_prev,
                self.centering_task.q_mid,
                self.centering_task.half,
                activation=self.centering_task.cfg.activation,
            )
            joint_margin_frac = float(np.clip(1.0 - u_max, 0.0, 1.0))
            stroke_planned = bool(
                self.posture_retarget is not None and self.posture_retarget.planned
            )
            homing_split = False
            if self.posture_retarget is not None and not stroke_planned:
                homing_split = abs(
                    float(self.posture_retarget.d_star_m)
                    - float(self.posture_retarget.cfg.d_attr_m)
                ) > 0.01
            elbow_floor = float(self.cfg.qp.branch_barrier.box_activate_rad)
            if elbow_floor <= 1.0e-9:
                elbow_floor = float(self.cfg.qp.branch_barrier.activate_rad)
            block_escape = abs(float(q_prev[4])) < elbow_floor
            v_ext, w_ext = self.rail_ext_task(
                q_prev,
                sigma_scale=sig_scale,
                sigma_grad_rail=self._sigma_grad_rail_cached,
                vel_ff=vel_ff,
                dt_s=float(dt),
                joint_margin_frac=joint_margin_frac,
                sigma_raw=sigma_now,
                y_tcp_d=y_tcp_d,
                press_stalled=allow_press_escape,
                tool_y_err_m=tool_y_err_m,
                stroke_limiters=stroke_planned,
                apply_d_band=not homing_split,
                block_escape=block_escape,
            )
            rail_ext_err = self.rail_ext_task.last_err_m
            rail_escape_active = bool(self.rail_ext_task._escape_active)
            # Prefer projected MotionReference FF over joint-plan rail FF.
            if np.isfinite(getattr(self.rail_ext_task, "last_v_ff", float("nan"))):
                rail_qdot_ff_val = float(self.rail_ext_task.last_v_ff)
            if w_ext > 0.0:
                rail_task_vel = v_ext
                rail_task_weight = w_ext
            if sigma_esc_ref > 1e-9 and sigma_now < sigma_esc_ref:
                manip_for_saturation = True

        rail_reg_scale = 1.0
        if self.rail_ext_task is not None:
            rail_reg_scale = float(
                getattr(self.rail_ext_task, "last_d_star_reg_scale", 1.0) or 1.0
            )

        # Hard box is 5/780 mm.  Do not freeze q0 in a leave/fade band.
        rail_sat_now = bool(
            self.rail_ext_task is not None
            and bool(getattr(self.rail_ext_task, "last_limit_saturated", False))
        )
        rail_vel_pin_eff = rail_vel_pin
        rail_task_weight_eff = rail_task_weight
        keep_task_weight = False
        pref_slack_scale = 1.0

        box_h1, box_h2 = self._measure_box_periods(dt)
        qdot_history_before_solve = np.asarray(self.core.qdot_prev, dtype=float).copy()
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
            ),
            q_meas=q_state,
            resync_err=resync_vec,
            rail_locked=locked_hold,
            rail_lock_reg_scale=self.cfg.rail.lock_reg_scale,
            rail_reg_scale=rail_reg_scale,
            rail_lock_vel_eps_m_s=self.cfg.rail.lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_eff,
            zero_secondary_rail=not locked_hold,
            rail_task_vel_m_s=rail_task_vel,
            rail_task_weight=rail_task_weight_eff,
            box_dt=box_h1,
            box_h1=box_h1,
            box_h2=box_h2,
            keep_task_weight=keep_task_weight,
            pref_slack_scale=pref_slack_scale,
            rail_exec_vel_m_s=rail_exec_for_qp,
            jacobian=J_pre,
            sigma=sigma_values_pre,
            mass_matrix=mass_pre,
            kinematics_ready=True,
        )

        qdot_out = np.asarray(r.qdot, dtype=float).copy()
        failed = bool(self.core.last_failed)
        fallback_reason = "qp_failed" if failed else ""
        if qdot_out.shape != q_prev.shape or not np.all(np.isfinite(qdot_out)):
            qdot_out = np.zeros_like(q_prev)
            failed = True
            fallback_reason = "final_qdot_nonfinite_or_bad_shape"
        if failed:
            # One infeasible / max-iter tick must not kill the session.
            # Brake with the certified previous command, same as 3d095f2.
            decay = float(getattr(self.cfg.qp, "fail_qdot_decay", 0.85))
            qdot_out = np.asarray(qdot_history_before_solve, dtype=float) * decay
            v_lim = np.asarray(self.limits.v_max, dtype=float)
            qdot_out = np.clip(qdot_out, -v_lim, v_lim)
            self.q_cmd = q_prev + qdot_out * float(dt)
            self.core.qdot_prev = qdot_out.copy()
            self.safety.sync_applied_delta(qdot_out * float(dt), float(dt))
            return self._make_step(
                qdot=qdot_out,
                twist_base=twist_base,
                sigma_min=r.sigma_min,
                manip=r.manip,
                slack_norm=r.slack_norm,
                n_cbf_active=r.n_cbf_active,
                follow_err=follow_err,
                qdot_ff_norm=(
                    float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0
                ),
                rail_vel_pin=rail_vel_pin_eff,
                rail_qdot_ff=rail_qdot_ff_val,
                plan_drives_rail=bool(plan_drives_rail),
                rail_ext_err_m=rail_ext_err,
                rail_ext_weight=rail_task_weight,
                failed=False,
                fallback_reason="qp1_decay",
                rail_macro_pref_v=(
                    float(rail_task_vel) if rail_task_vel is not None else 0.0
                ),
                rail_escape_active=rail_escape_active,
            )
        else:
            qdot_certified = qdot_out.copy()
            q_candidate = q_prev + qdot_out * dt
            margin = np.asarray(self.limits.position_margin, dtype=float)
            if np.any(q_candidate < self.limits.q_lower + margin - 1.0e-9) or np.any(
                q_candidate > self.limits.q_upper - margin + 1.0e-9
            ):
                qdot_out = self._clip_qdot_to_box(
                    q_prev, qdot_out, dt, q_state, resync_vec,
                    rail_locked=locked_hold, rail_vel_pin=rail_vel_pin_eff,
                    box_h1=box_h1, box_h2=box_h2,
                    rail_lead_exempt=True,
                )
                fallback_reason = fallback_reason or "projected_into_velocity_box"

        # Do NOT shape qdot_out[0] here to "match the rail servo bandwidth".
        # Whatever is written here becomes core.qdot_prev below, which is the
        # base of the QP acceleration box, the jerk box and the SafetyLimiter
        # acceleration clamp — a first-order filter therefore multiplies all
        # three limits by its own alpha instead of just smoothing.  A 40 ms
        # filter measured 7x less rail travel than the task asked for
        # (0.0137 vs 0.0997 m/s) and pushed the residual into slack.  Any
        # future shaping must act only on the value handed to the servo, and
        # must leave the state that feeds the boxes untouched.
        q_next = q_prev + qdot_out * dt
        rep = self.safety.clamp(q_prev, q_next, dt)
        self.q_cmd = rep.q_safe
        if dt > 1e-9 and (rep.vel_clamped or rep.acc_clamped or rep.pos_clamped):
            applied = (self.q_cmd - q_prev) / dt
            if rep.pos_clamped:
                applied[0] = float(qdot_out[0])
                lo0 = float(self.limits.q_lower[0])
                hi0 = float(self.limits.q_upper[0])
                if self.q_cmd[0] <= lo0 + 1.0e-9 and applied[0] < 0.0:
                    applied[0] = 0.0
                elif self.q_cmd[0] >= hi0 - 1.0e-9 and applied[0] > 0.0:
                    applied[0] = 0.0
            self.core.qdot_prev = applied
        else:
            self.core.qdot_prev = qdot_out.copy()

        if q_state is not None:
            lead_max = float(self.cfg.resync_err_rail_m)
            if lead_max > 0.0:
                q0_meas = float(q_state[0])
                q0_cmd = float(self.q_cmd[0])
                if q0_cmd > q0_meas + lead_max:
                    self.q_cmd[0] = q0_meas + lead_max
                elif q0_cmd < q0_meas - lead_max:
                    self.q_cmd[0] = q0_meas - lead_max
                v_rail_max = float(self.limits.v_max[0])
                self.core.qdot_prev[0] = float(
                    np.clip(float(self.core.qdot_prev[0]), -v_rail_max, v_rail_max)
                )
        # At the command floor, refuse to jog back into the switch.
        lo0 = float(self.limits.q_lower[0])
        hi0 = float(self.limits.q_upper[0])
        self.q_cmd[0] = float(np.clip(self.q_cmd[0], lo0, hi0))
        if self.q_cmd[0] <= lo0 + 1.0e-4 and self.core.qdot_prev[0] < 0.0:
            self.q_cmd[0] = lo0
            self.core.qdot_prev[0] = 0.0
        elif self.q_cmd[0] >= hi0 - 1.0e-4 and self.core.qdot_prev[0] > 0.0:
            self.q_cmd[0] = hi0
            self.core.qdot_prev[0] = 0.0
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
        qdot_out = self.core.qdot_prev.copy()
        if locked_hold and self.cfg.rail.lock_hard_pin:
            qdot_out[0] = 0.0
        elif plan_drives_rail and qdot_ff is not None:
            qdot_out[0] = float(np.asarray(qdot_ff)[0])
            if rail_only:
                qdot_out[1:] = 0.0

        final_hard_violation, final_task_lock_violation = (
            self.core.validate_final_qdot(qdot_out)
        )
        self.core.last_final_hard_violation = float(final_hard_violation)
        self.core.last_final_task_lock_violation = float(final_task_lock_violation)
        final_tol = max(
            10.0 * float(getattr(self.cfg.qp, "eps_abs", 1.0e-6)),
            1.0e-5,
        )
        if (
            not np.isfinite(final_hard_violation)
            or not np.isfinite(final_task_lock_violation)
            or final_hard_violation > final_tol
            or final_task_lock_violation > final_tol
        ):
            # A limiter/lead rewrite is not allowed to break QP1.  If this
            # tick already has a certified QP command, publish that instead
            # of stopping; stop only when no certified command exists.
            hard_qp, lock_qp = self.core.validate_final_qdot(qdot_certified)
            if (
                np.isfinite(hard_qp)
                and np.isfinite(lock_qp)
                and hard_qp <= final_tol
                and lock_qp <= final_tol
            ):
                fallback_reason = (
                    fallback_reason or "limiter_rewrite_rejected_keep_qp"
                )
                qdot_out = qdot_certified.copy()
                self.q_cmd = q_prev + qdot_out * dt
                self.core.qdot_prev = qdot_out.copy()
                self.safety.sync_applied_delta(qdot_out * float(dt), float(dt))
                self.core.last_final_hard_violation = float(hard_qp)
                self.core.last_final_task_lock_violation = float(lock_qp)
            else:
                failed = True
                fallback_reason = (
                    "final_publication_certificate_failed:"
                    f"hard={final_hard_violation:.3e},"
                    f"task_lock={final_task_lock_violation:.3e}"
                )
                self.q_cmd = q_prev.copy()
                self.core.qdot_prev = qdot_history_before_solve.copy()
                self.safety.sync_applied_delta(
                    qdot_history_before_solve * float(dt), float(dt)
                )
                qdot_out = np.zeros_like(q_prev)
        self.last_sigma_min = r.sigma_min
        self.last_arm_rho = float(r.sigma_min)
        # Decompose achieved linear velocity into rail vs arm along primary motion.
        J_fin = J_pre
        qdot_arr = np.asarray(qdot_out, dtype=float)
        twist_rail = J_fin[:, 0] * float(rail_exec_for_qp)
        twist_arm = J_fin[:, 1:] @ qdot_arr[1:]
        motion_dir = np.asarray(twist_base[:3], dtype=float)
        if vel_ff is not None:
            vff = np.asarray(vel_ff, dtype=float).reshape(-1)
            if vff.size >= 3 and float(np.linalg.norm(vff[:3])) > 1e-6:
                motion_dir = vff[:3].astype(float)
        n_dir = float(np.linalg.norm(motion_dir))
        if n_dir > 1e-9:
            u = motion_dir / n_dir
            rail_contrib = float(np.dot(twist_rail[:3], u))
            arm_contrib = float(np.dot(twist_arm[:3], u))
            denom = abs(rail_contrib) + abs(arm_contrib)
            rail_share = (abs(rail_contrib) / denom) if denom > 1e-9 else float("nan")
        else:
            rail_contrib = float("nan")
            arm_contrib = float("nan")
            rail_share = float("nan")
        # Keep qpik_scan_* alive: primary linear motion des/achieved/residual (m).
        scan_t = np.array(
            [float(twist_base[0]), float(twist_base[1])], dtype=float
        )
        scan_a = np.array(
            [
                float(twist_rail[0] + twist_arm[0]),
                float(twist_rail[1] + twist_arm[1]),
            ],
            dtype=float,
        )
        q_now = np.asarray(self.q_cmd, dtype=float)
        near_arm = bool(
            np.any(q_now[1:] < self.limits.q_lower[1:] + 0.08)
            or np.any(q_now[1:] > self.limits.q_upper[1:] - 0.08)
        )
        physical_saturated = bool(rep.pos_clamped or near_arm)
        step = self._make_step(
            qdot=qdot_out,
            twist_base=twist_base,
            sigma_min=r.sigma_min,
            manip=r.manip,
            slack_norm=r.slack_norm,
            n_cbf_active=r.n_cbf_active,
            follow_err=follow_err,
            qdot_ff_norm=float(np.linalg.norm(qdot_ff)) if qdot_ff is not None else 0.0,
            vel_clamped=rep.vel_clamped,
            acc_clamped=rep.acc_clamped,
            pos_clamped=rep.pos_clamped,
            rail_vel_pin=rail_vel_pin_eff,
            rail_qdot_ff=rail_qdot_ff_val,
            plan_drives_rail=bool(plan_drives_rail),
            rail_ext_err_m=rail_ext_err,
            rail_ext_weight=rail_task_weight,
            failed=failed,
            fallback_reason=fallback_reason,
            rail_macro_pref_v=(
                float(rail_task_vel) if rail_task_vel is not None else 0.0
            ),
            rail_escape_active=rail_escape_active,
            rail_contrib_m_s=rail_contrib,
            arm_contrib_m_s=arm_contrib,
            rail_motion_share=rail_share,
            scan_target=scan_t,
            scan_achieved=scan_a,
            scan_residual=scan_t - scan_a,
            physical_saturated=physical_saturated,
        )
        actual_task_twist = twist_rail + twist_arm
        actual_task_residual = np.asarray(twist_base, dtype=float) - actual_task_twist
        step.protected_target = np.asarray(twist_base, dtype=float).copy()
        step.protected_achieved = np.asarray(actual_task_twist, dtype=float).copy()
        step.protected_residual = np.asarray(actual_task_residual, dtype=float).copy()
        step.qpik_working_slack = np.asarray(actual_task_residual, dtype=float).copy()
        step.qpik_equality_residual_max = float(np.max(np.abs(actual_task_residual)))
        step.qpik_hard_residual_max = float(
            getattr(self.core, "last_final_hard_violation", 0.0)
        )
        step.rail_xy_contribution = np.asarray(twist_rail[:2], dtype=float).copy()
        step.arm_xy_contribution = np.asarray(twist_arm[:2], dtype=float).copy()
        step.rail_exec_velocity_m_s = float(rail_exec_for_qp)
        if rail_exec_vel_m_s is not None:
            step.rail_measured_velocity_m_s = float(rail_exec_vel_m_s)
        if rail_exec_smooth_m_s is not None:
            step.rail_commanded_velocity_m_s = float(rail_exec_smooth_m_s)
        step.qp_solver_overrun = bool(getattr(self.core, "last_qp_overrun", False))
        step.qp1_status = str(getattr(self.core, "last_qp1_status", step.qp1_status))
        step.qp2_status = str(getattr(self.core, "last_qp2_status", step.qp2_status))
        step.qp2_fallback = bool(getattr(self.core, "last_qp2_fallback", False))
        step.rail_sat = bool(rail_sat_now)
        step.last_limit_saturated = bool(
            self.rail_ext_task is not None
            and self.rail_ext_task.last_limit_saturated
        )
        step.keep_task_weight = bool(keep_task_weight)
        step.pref_slack_scale = float(pref_slack_scale)
        step.rail_task_vel = (
            float(rail_task_vel) if rail_task_vel is not None else float("nan")
        )
        if self.rail_ext_task is not None:
            step.v_escape = float(self.rail_ext_task.last_v_escape)
            step.v_reach = float(self.rail_ext_task.last_v_reach)
            step.v_ff_rail = float(self.rail_ext_task.last_v_ff)
        step.sigma_arm = float(sigma_arm)
        step.sns_scale = float(getattr(self.core, "last_sns_scale", 1.0))
        step.v_cmd = np.asarray(twist_base, dtype=float).reshape(6).copy()
        step.path_twist = np.asarray(path_twist_arr, dtype=float).reshape(6).copy()
        step.feedback_twist = np.asarray(
            feedback_twist_arr, dtype=float
        ).reshape(6).copy()
        comfort = getattr(self.core, "last_comfort_slack", None)
        if comfort is not None:
            step.comfort_slack = np.asarray(comfort, dtype=float).reshape(-1)[:7]
        step.cbf_min_dist = float(
            getattr(self.core, "last_cbf_min_dist", float("nan"))
        )
        step.cbf_pair = str(getattr(self.core, "last_cbf_pair", "") or "")
        step.nullspace_norm = float(self.last_secondary_norm)
        return step

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
        self.last_pose_d: np.ndarray | None = None
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
        self.last_pose_d = np.asarray(ref.pose_d, dtype=float).copy()
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
        pose_track = np.asarray(
            getattr(self.controller, "last_pose_d_combined", ref.pose_d),
            dtype=float,
        ).reshape(-1)
        if pose_track.size != 6 or not np.isfinite(pose_track).all():
            pose_track = np.asarray(ref.pose_d, dtype=float)
        tr_mm, tr_deg = pose_track_error_mm_deg(
            pose_track,
            current_pose,
            track_axes=self.controller.cfg.track_axes,
            euler_order=self.controller.cfg.euler_order,
        )
        self.last_err_mm = tr_mm
        self.last_track_rot_deg = tr_deg
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
        self.last_pose_d: np.ndarray | None = None
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
        self.last_pose_d = np.asarray(ref.pose_d, dtype=float).copy()
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
    governor_scale_min: float = 0.25
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
           # Motion-subspace accuracy (force axes excluded from "准" metrics).
           "pose_d_x", "pose_d_y", "pose_d_z", "pose_d_rx", "pose_d_ry", "pose_d_rz",
           "pose_meas_x", "pose_meas_y", "pose_meas_z",
           "pose_meas_rx", "pose_meas_ry", "pose_meas_rz",
           "motion_err_lin_x_mm", "motion_err_lin_y_mm", "motion_err_lin_z_mm",
           "motion_err_rot_x_deg", "motion_err_rot_y_deg", "motion_err_rot_z_deg",
           "motion_err_rms_mm", "motion_axis_peak_mm",
           "vel_ff_vx", "vel_ff_vy", "vel_ff_vz", "vel_ff_wx", "vel_ff_wy", "vel_ff_wz",
           "rail_contrib_m_s", "arm_contrib_m_s", "arm_y_qdot", "rail_motion_share",
           # Chan-Dubey reg multipliers: rail first, then the worst arm joint.
           "wln_scale_rail", "wln_scale_arm_max",
           "waste_ratio", "rail_ff_m", "rail_track_err_m",
           "rail_escape_active",
           "psi_deg", "psi_ref_deg", "psi_retarget_score", "d_pref_m",
           "d_star_m", "psi_star_deg", "minmax_margin",
           "elbow_margin_rad", "wrist_open_rad", "family_ok",
           "tool_y_des_m", "tool_y_err_mm",
           "contact_phase", "v_air_cmd", "ke_hat", "dob_v", "barrier_cap_floor",
           # Append-only normal-axis BEFM/audit schema.
           "flow_x_p", "flow_v_p", "flow_v_aux", "flow_x_a", "flow_v_a",
           "flow_e", "flow_edot", "flow_F_c", "flow_v_track",
           "flow_P_e", "flow_P_c", "flow_alpha_target", "flow_alpha",
           "flow_alpha_case", "flow_T", "flow_psi", "flow_S_n",
           "flow_S_r_hat", "flow_P_phys", "flow_P_mismatch",
           "flow_E_phys", "flow_E_mismatch", "flow_gamma_active",
           # Observe-mode evidence: press (m/s) the gate would have removed.
           "flow_alpha_would_gate", "flow_edot_aligned",
           "flow_sign_fault", "flow_feedback_stale", "flow_blocked_reason",
           "contact_episode_rearm_event", "contact_episode_release_s",
           "surface_force_scale", "surface_force_alpha", "surface_xy_error_m",
           "force_barrier_contact_active",
           # Fixed single-shot QPIK telemetry.
           "qpik_backend", "qpik_solver_status", "qpik_solver_iterations",
           "qpik_solver_solve_ms", "qpik_solver_call_count",
           "qpik_solver_overrun",
           "qpik_qp1_status", "qpik_qp2_status",
           "qpik_qp1_solve_ms", "qpik_qp2_solve_ms",
           "qpik_assembly_ms", "qpik_fallback_ms",
           "qpik_total_ms", "qpik_qp2_fallback",
           "tick_inner_ms", "tick_send_ms", "tick_log_ms",
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
           "qpik_final_sent_qdot_json",
           "rail_sat",
           "rail_exec_velocity_m_s", "rail_measured_velocity_m_s",
           "rail_commanded_velocity_m_s", "rail_commanded_acceleration_m_s2",
           "rail_feedback_age_s", "a_mirror_frac", "j_mirror_frac",
           "last_limit_saturated", "keep_task_weight",
           "pref_slack_scale", "rail_task_vel",
           "v_escape", "v_reach", "v_ff_rail", "sigma_arm", "sns_scale",
           "qpik_nullspace_norm", "cbf_min_dist", "cbf_pair"]
        + [f"qdot_meas_{i}" for i in range(8)]
        + [f"v_cmd_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"path_twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"feedback_twist_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
        + [f"comfort_slack_j{i}" for i in range(1, 8)]
        + [
            "pad_connected",
            "pad_lx", "pad_ly", "pad_lt", "pad_rx", "pad_ry", "pad_rt",
            "pad_lb", "pad_rb",
            "pad_vx", "pad_vy", "pad_vz",
            "pad_wx", "pad_wy", "pad_wz",
        ]
        + [f"pad_vcmd_base_{a}" for a in ("vx", "vy", "vz", "wx", "wy", "wz")]
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

    @staticmethod
    def _fmt_pad_fields(outer) -> list[str]:
        """Stick + mapped v_cmd extras; empty when the outer loop is not a pad."""

        def _fmt_n(arr, n: int, prec: int = 6) -> list[str]:
            if arr is None:
                return [""] * n
            vals = np.asarray(arr, dtype=float).reshape(-1)
            out = []
            for i in range(n):
                if i >= vals.size or not np.isfinite(vals[i]):
                    out.append("")
                else:
                    out.append(f"{float(vals[i]):.{prec}f}")
            return out

        axes = getattr(outer, "last_pad_axes", None)
        buttons = getattr(outer, "last_pad_buttons", None)
        if axes is None and buttons is None:
            return [""] * 21
        connected = getattr(outer, "last_pad_connected", False)
        btn = (
            np.asarray(buttons, dtype=float).reshape(-1)
            if buttons is not None
            else np.zeros(8)
        )
        lb = 1 if (btn.size > 4 and float(btn[4]) > 0.5) else 0
        rb = 1 if (btn.size > 5 and float(btn[5]) > 0.5) else 0
        return (
            [str(int(bool(connected)))]
            + _fmt_n(axes, 6, 4)
            + [str(lb), str(rb)]
            + _fmt_n(getattr(outer, "last_v_world", None), 3, 6)
            + _fmt_n(getattr(outer, "last_w_tool", None), 3, 6)
            + _fmt_n(getattr(outer, "last_twist_base", None), 6, 6)
        )

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
        qdot_meas: np.ndarray | None = None,
        rail_target_sent_m: float | None = None,
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
        contact_phase = getattr(ctrl, "contact_phase", "")
        v_air_cmd = getattr(ctrl, "v_air_cmd", float("nan"))
        ke_hat = getattr(ctrl, "ke_hat", getattr(ctrl, "ke_est", float("nan")))
        dob_v = getattr(ctrl, "dob_v", float("nan"))
        barrier_cap_floor = getattr(ctrl, "barrier_cap_floor", float("nan"))
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
        flow_would_gate = getattr(flow, "alpha_would_gate_m_s", float("nan"))
        flow_edot_aligned = getattr(
            flow, "mismatch_velocity_aligned", float("nan")
        )
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
        if rail_target_sent_m is not None and np.isfinite(float(rail_target_sent_m)):
            rail_sent = float(rail_target_sent_m)
        else:
            rail_sent = float(step.q_send[0]) if step.q_send is not None else float("nan")
        try:
            q_cmd_q_meas_norm = float(
                np.linalg.norm(np.asarray(step.q_send, dtype=float) - qm)
            )
        except (TypeError, ValueError):
            q_cmd_q_meas_norm = float("nan")

        # Desired pose + motion-subspace errors (force axes zeroed / ignored).
        pose_d = getattr(outer, "last_pose_d", None)
        pose_d_arr = (
            np.asarray(pose_d, dtype=float).reshape(-1)
            if pose_d is not None
            else np.full(6, np.nan)
        )
        if pose_d_arr.size < 6:
            pose_d_arr = np.full(6, np.nan)
        pose_meas_arr = np.asarray(pose, dtype=float).reshape(-1)
        if pose_meas_arr.size < 6:
            pose_meas_arr = np.full(6, np.nan)
        vel_ff = getattr(outer, "last_vel_ff", None)
        vel_ff_arr = (
            np.asarray(vel_ff, dtype=float).reshape(-1)
            if vel_ff is not None
            else np.full(6, np.nan)
        )
        if vel_ff_arr.size < 6:
            vel_ff_arr = np.full(6, np.nan)
        motion_err_lin_mm = np.full(3, np.nan)
        motion_err_rot_deg = np.full(3, np.nan)
        motion_err_rms_mm = float("nan")
        motion_axis_peak_mm = float("nan")
        tool_y_des_m = float("nan")
        tool_y_err_mm = float("nan")
        euler_order = "xyz"
        track_axes = np.ones(6)
        ctrl_cfg = getattr(ctrl, "cfg", None) if ctrl is not None else None
        if ctrl_cfg is not None:
            euler_order = str(getattr(ctrl_cfg, "euler_order", "xyz"))
            ta = getattr(ctrl_cfg, "track_axes", None)
            if ta is not None:
                track_axes = np.asarray(ta, dtype=float).reshape(-1)
                if track_axes.size < 6:
                    track_axes = np.ones(6)
        if np.all(np.isfinite(pose_d_arr)) and np.all(np.isfinite(pose_meas_arr)):
            err_base = pose_error(pose_d_arr, pose_meas_arr, euler_order)
            r_cur = Rsc.from_euler(
                euler_order, pose_meas_arr[3:6], degrees=False
            ).as_matrix()
            err_tool = np.zeros(6, dtype=float)
            err_tool[:3] = r_cur.T @ err_base[:3]
            err_tool[3:6] = r_cur.T @ err_base[3:6]
            ta6 = np.asarray(track_axes, dtype=float)[:6]
            # Force-axis components excluded from accuracy (NaN in per-axis cols).
            for i in range(3):
                if ta6[i] > 0.5:
                    motion_err_lin_mm[i] = float(err_tool[i] * 1000.0)
                else:
                    motion_err_lin_mm[i] = float("nan")
            for i in range(3):
                if ta6[3 + i] > 0.5:
                    motion_err_rot_deg[i] = float(np.degrees(err_tool[3 + i]))
                else:
                    motion_err_rot_deg[i] = float("nan")
            lin_tracked = motion_err_lin_mm[np.isfinite(motion_err_lin_mm)]
            if lin_tracked.size:
                motion_err_rms_mm = float(np.sqrt(np.mean(np.square(lin_tracked))))
                motion_axis_peak_mm = float(np.max(np.abs(lin_tracked)))
            # SIN fixture alias: tool-Y from general des/meas (control frame).
            tool_y_des_m = float((r_cur.T @ pose_d_arr[:3])[1])
            if np.isfinite(motion_err_lin_mm[1]):
                tool_y_err_mm = float(motion_err_lin_mm[1])
            elif ta6[1] > 0.5:
                tool_y_meas = float((r_cur.T @ pose_meas_arr[:3])[1])
                tool_y_err_mm = float((tool_y_des_m - tool_y_meas) * 1000.0)

        def _fmt6(arr: np.ndarray | None) -> list[str]:
            if arr is None:
                return [""] * 6
            vals = np.asarray(arr, dtype=float).reshape(-1)
            if vals.size < 6:
                vals = np.pad(vals, (0, 6 - int(vals.size)), constant_values=np.nan)
            return [
                f"{float(v):.6f}" if np.isfinite(v) else ""
                for v in vals[:6]
            ]

        def _fmt8(arr: np.ndarray | None) -> list[str]:
            if arr is None:
                return [""] * 8
            vals = np.asarray(arr, dtype=float).reshape(-1)
            if vals.size < 8:
                vals = np.pad(vals, (0, 8 - int(vals.size)), constant_values=np.nan)
            return [
                f"{float(v):.6f}" if np.isfinite(v) else ""
                for v in vals[:8]
            ]

        def _fmt3(arr: np.ndarray, prec: int = 4) -> list[str]:
            return [
                f"{float(v):.{prec}f}" if np.isfinite(v) else ""
                for v in np.asarray(arr, dtype=float).reshape(-1)[:3]
            ]

        comfort = np.asarray(
            getattr(step, "comfort_slack", np.zeros(7)), dtype=float
        ).reshape(-1)
        if comfort.size < 7:
            comfort = np.pad(comfort, (0, 7 - int(comfort.size)))
        comfort = comfort[:7]

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
               *_fmt6(pose_d_arr),
               *_fmt6(pose_meas_arr),
               *_fmt3(motion_err_lin_mm, prec=3),
               *_fmt3(motion_err_rot_deg, prec=4),
               f"{motion_err_rms_mm:.3f}" if np.isfinite(motion_err_rms_mm) else "",
               (
                   f"{motion_axis_peak_mm:.3f}"
                   if np.isfinite(motion_axis_peak_mm)
                   else ""
               ),
               *_fmt6(vel_ff_arr),
               (
                   f"{step.rail_contrib_m_s:.6f}"
                   if np.isfinite(step.rail_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.arm_contrib_m_s:.6f}"
                   if np.isfinite(step.arm_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.arm_contrib_m_s:.6f}"
                   if np.isfinite(step.arm_contrib_m_s)
                   else ""
               ),
               (
                   f"{step.rail_motion_share:.4f}"
                   if np.isfinite(step.rail_motion_share)
                   else ""
               ),
               (
                   f"{step.wln_scale_rail:.4f}"
                   if np.isfinite(getattr(step, "wln_scale_rail", float("nan")))
                   else ""
               ),
               (
                   f"{step.wln_scale_arm_max:.4f}"
                   if np.isfinite(getattr(step, "wln_scale_arm_max", float("nan")))
                   else ""
               ),
               (
                   f"{step.waste_ratio:.4f}"
                   if np.isfinite(getattr(step, "waste_ratio", float("nan")))
                   else ""
               ),
               (
                   f"{step.rail_ff_m:.6f}"
                   if np.isfinite(getattr(step, "rail_ff_m", float("nan")))
                   else ""
               ),
               (
                   f"{step.rail_track_err_m:.6f}"
                   if np.isfinite(getattr(step, "rail_track_err_m", float("nan")))
                   else ""
               ),
               int(bool(step.rail_escape_active)),
               f"{step.psi_deg:.4f}" if np.isfinite(step.psi_deg) else "",
               f"{step.psi_ref_deg:.4f}" if np.isfinite(step.psi_ref_deg) else "",
               (
                   f"{step.psi_retarget_score:.6f}"
                   if np.isfinite(step.psi_retarget_score)
                   else ""
               ),
               f"{step.d_pref_m:.6f}" if np.isfinite(step.d_pref_m) else "",
               (
                   f"{step.d_star_m:.6f}"
                   if np.isfinite(getattr(step, "d_star_m", float("nan")))
                   else ""
               ),
               (
                   f"{step.psi_star_deg:.4f}"
                   if np.isfinite(getattr(step, "psi_star_deg", float("nan")))
                   else ""
               ),
               (
                   f"{step.minmax_margin:.6f}"
                   if np.isfinite(getattr(step, "minmax_margin", float("nan")))
                   else ""
               ),
               (
                   f"{step.elbow_margin_rad:.6f}"
                   if np.isfinite(step.elbow_margin_rad)
                   else ""
               ),
               (
                   f"{step.wrist_open_rad:.6f}"
                   if np.isfinite(step.wrist_open_rad)
                   else ""
               ),
               "1" if bool(getattr(step, "family_ok", True)) else "0",
               f"{tool_y_des_m:.6f}" if np.isfinite(tool_y_des_m) else "",
               f"{tool_y_err_mm:.3f}" if np.isfinite(tool_y_err_mm) else "",
               str(contact_phase),
               f"{float(v_air_cmd):.6f}" if np.isfinite(v_air_cmd) else "",
               f"{float(ke_hat):.4f}" if np.isfinite(ke_hat) else "",
               f"{float(dob_v):.6f}" if np.isfinite(dob_v) else "",
               (
                   f"{float(barrier_cap_floor):.6f}"
                   if np.isfinite(barrier_cap_floor)
                   else ""
               ),
               f"{flow_xp:.8f}", f"{flow_vp:.8f}", f"{flow_v_aux:.8f}",
               f"{flow_xa:.8f}", f"{flow_va:.8f}", f"{flow_e:.8f}",
               f"{flow_edot:.8f}", f"{flow_fc:.8f}", f"{flow_v_track:.8f}",
               f"{flow_pe:.8f}", f"{flow_pc:.8f}",
               f"{flow_alpha_target:.8f}", f"{flow_alpha:.8f}",
               str(flow_alpha_case), f"{flow_tank:.9f}", f"{flow_psi:.8f}",
               f"{flow_sn:.9f}", f"{flow_sr:.9f}", f"{flow_p_phys:.8f}",
               f"{flow_p_mismatch:.8f}", f"{flow_e_phys:.9f}",
               f"{flow_e_mismatch:.9f}", f"{flow_gamma:.8f}",
               f"{float(flow_would_gate):.8f}",
               f"{float(flow_edot_aligned):.8f}",
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
               str(getattr(step, "qp1_status", "not_run")),
               str(getattr(step, "qp2_status", "not_run")),
               f"{float(getattr(step, 'qp1_solve_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp2_solve_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp_assembly_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qp_fallback_ms', 0.0)):.6f}",
               f"{float(getattr(step, 'qpik_total_ms', 0.0)):.6f}",
               int(bool(getattr(step, "qp2_fallback", False))),
               *(
                   f"{v:.4f}" if np.isfinite(v) else ""
                   for v in (
                       getattr(step, "tick_inner_ms", float("nan")),
                       getattr(step, "tick_send_ms", float("nan")),
                       getattr(step, "tick_log_ms", float("nan")),
                   )
               ),
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
               self._json_compact(step.qdot),
               int(bool(step.rail_sat)),
               (
                   f"{float(step.rail_exec_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_exec_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_measured_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_measured_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_commanded_velocity_m_s):.8f}"
                   if np.isfinite(step.rail_commanded_velocity_m_s) else ""
               ),
               (
                   f"{float(step.rail_commanded_acceleration_m_s2):.8f}"
                   if np.isfinite(step.rail_commanded_acceleration_m_s2) else ""
               ),
               (
                   f"{float(step.rail_feedback_age_s):.6f}"
                   if np.isfinite(step.rail_feedback_age_s) else ""
               ),
               (
                   f"{float(step.a_mirror_frac):.6f}"
                   if np.isfinite(step.a_mirror_frac) else ""
               ),
               (
                   f"{float(step.j_mirror_frac):.6f}"
                   if np.isfinite(step.j_mirror_frac) else ""
               ),
               int(bool(step.last_limit_saturated)),
               int(bool(step.keep_task_weight)),
               f"{float(step.pref_slack_scale):.4f}",
               (
                   f"{float(step.rail_task_vel):.6f}"
                   if np.isfinite(step.rail_task_vel)
                   else ""
               ),
               f"{float(step.v_escape):.6f}" if np.isfinite(step.v_escape) else "",
               f"{float(step.v_reach):.6f}" if np.isfinite(step.v_reach) else "",
               f"{float(step.v_ff_rail):.6f}" if np.isfinite(step.v_ff_rail) else "",
               f"{float(step.sigma_arm):.5f}" if np.isfinite(step.sigma_arm) else "",
               f"{float(step.sns_scale):.4f}",
               (
                   f"{float(step.nullspace_norm):.6f}"
                   if np.isfinite(step.nullspace_norm)
                   else ""
               ),
               (
                   f"{float(step.cbf_min_dist):.6f}"
                   if np.isfinite(step.cbf_min_dist)
                   else ""
               ),
               str(step.cbf_pair),
               *_fmt8(
                   qdot_meas
                   if qdot_meas is not None
                   else getattr(step, "qdot_meas", None)
               ),
               *_fmt6(getattr(step, "v_cmd", step.twist_base)),
               *_fmt6(getattr(step, "path_twist", None)),
               *_fmt6(getattr(step, "feedback_twist", None)),
               *(f"{float(v):.9e}" for v in comfort),
               *self._fmt_pad_fields(outer)]
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


def _rail_qdot_m_s(
    rail_bridge,
    q_new: np.ndarray,
    last_q: np.ndarray | None,
    dt_feedback: float | None,
) -> float | None:
    """Rail is not in the RealMan UDP frame; use FA24 speed, else Δq0/Δt."""
    meas = getattr(rail_bridge, "measured_speed_m_s", None)
    if meas is not None:
        try:
            v_rail = float(meas)
        except (TypeError, ValueError):
            v_rail = float("nan")
        if np.isfinite(v_rail):
            return v_rail
    if (
        last_q is not None
        and dt_feedback is not None
        and 0.001 <= float(dt_feedback) <= 0.050
    ):
        return float(q_new[0] - last_q[0]) / float(dt_feedback)
    return None


@dataclass
class RailExecutionEstimate:
    position_m: float
    velocity_m_s: float
    measured_velocity_m_s: float
    commanded_velocity_m_s: float
    commanded_acceleration_m_s2: float
    sample_mono_s: float
    age_s: float
    extrapolation_age_s: float
    command_mode: str


def _rail_execution_velocity_estimate(
    rail_bridge,
    *,
    now_s: float | None = None,
    freshness_s: float,
    feedback=None,
    require_fresh: bool = True,
) -> RailExecutionEstimate | None:
    """Bounded one-poll rail execution estimate for strict QPIK.

    Between two FA24 samples the measured velocity is propagated with the
    worker's latest commanded acceleration, but never farther than one
    configured rail poll.  After a fresh sample has been seen, stale or
    non-finite feedback is a control fault.  Before that first sample the
    caller should pass ``require_fresh=False`` so a cold Modbus poll can
    finish; the QP then uses its existing command-velocity ZOH.
    """
    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return None
    now = time.monotonic() if now_s is None else float(now_s)
    try:
        if feedback is None:
            feedback = rail_bridge.execution_feedback
        position = float(feedback.position_m)
        sample_t = float(feedback.sample_mono_s)
        v_meas = float(feedback.v_meas_m_s)
        v_cmd = float(feedback.v_cmd_m_s)
        a_cmd = float(feedback.a_cmd_m_s2)
        feedback_valid = bool(getattr(feedback, "valid", True))
        mode_obj = feedback.command_mode
    except Exception as exc:
        if not require_fresh:
            return None
        raise RuntimeError(f"rail execution feedback unavailable: {exc}") from exc
    values = (now, position, sample_t, v_meas, v_cmd, a_cmd)
    if not all(np.isfinite(value) for value in values):
        if not require_fresh:
            return None
        raise RuntimeError(
            "rail execution feedback is non-finite: "
            f"x={position!r}, t={sample_t!r}, v_meas={v_meas!r}, "
            f"v_cmd={v_cmd!r}, a_cmd={a_cmd!r}"
        )
    if not feedback_valid:
        if not require_fresh:
            return None
        raise RuntimeError("rail execution feedback rejected by encoder gate")
    age = max(0.0, now - sample_t)
    max_age = max(float(freshness_s), 0.0)
    if age > max_age:
        if not require_fresh:
            return None
        raise RuntimeError(
            f"rail execution feedback stale: age={age:.6f}s > {max_age:.6f}s"
        )
    cfg = getattr(rail_bridge, "config", None)
    poll_hz = max(float(getattr(cfg, "poll_hz", 50.0)), 1.0)
    extrap_age = min(age, 1.0 / poll_hz)
    v_est = v_meas + a_cmd * extrap_age
    v_cap = abs(float(getattr(cfg, "vel_max_m_s", float("inf"))))
    if np.isfinite(v_cap):
        v_est = float(np.clip(v_est, -v_cap, v_cap))
    mode = str(getattr(mode_obj, "value", mode_obj) or "")
    return RailExecutionEstimate(
        position_m=position,
        velocity_m_s=float(v_est),
        measured_velocity_m_s=v_meas,
        commanded_velocity_m_s=v_cmd,
        commanded_acceleration_m_s2=a_cmd,
        sample_mono_s=sample_t,
        age_s=age,
        extrapolation_age_s=extrap_age,
        command_mode=mode,
    )


def _qdot_meas_8dof(
    q_new: np.ndarray,
    last_q: np.ndarray | None,
    dt_feedback: float | None,
    snap,
    rail_bridge=None,
    rail_velocity_m_s: float | None = None,
) -> np.ndarray | None:
    """8-vector qdot: SDK arm speed + rail encoder/servo. Finite-diff only as fallback."""
    arm = arm_qdot_rad_s_from_snap(snap)
    if arm is not None:
        qdot = np.zeros(8, dtype=float)
        qdot[1:] = arm
        v_rail = (
            float(rail_velocity_m_s)
            if rail_velocity_m_s is not None
            and np.isfinite(float(rail_velocity_m_s))
            else _rail_qdot_m_s(rail_bridge, q_new, last_q, dt_feedback)
        )
        if v_rail is not None:
            qdot[0] = v_rail
        return qdot
    if last_q is None or dt_feedback is None:
        return None
    if not (0.001 <= float(dt_feedback) <= 0.050):
        return None
    qdot = wrap_joint_delta(last_q, q_new) / float(dt_feedback)
    if rail_velocity_m_s is not None and np.isfinite(float(rail_velocity_m_s)):
        qdot[0] = float(rail_velocity_m_s)
    return qdot


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


def _qpik_rail_v_ff_m_s(
    qdot0: float, *, idle_eps: float = _RAIL_V_IDLE_M_S
) -> float:
    """Servo ``v_ff`` is the QPIK rail velocity, never a pad/path bypass."""
    v = float(qdot0)
    if not math.isfinite(v) or abs(v) < float(idle_eps):
        return 0.0
    return v


def _wall_clock_rail_target(
    q_send0: float,
    qdot0: float,
    dt_wall: float,
    dt_nom: float,
    *,
    soft_lo: float,
    soft_hi: float,
    meas_m: float | None = None,
    lead_max_m: float = 0.0,
) -> float:
    """One-tick wall alignment of QPIK ``q_send[0]``; no persistent integrator.

    QPIK already integrated at ``dt_nom``.  Add only
    ``qdot * max(0, dt_wall - dt_nom)`` for this tick.  Idle (``|qdot|≈0``)
    publishes ``q_send[0]`` and does not keep a 20 mm lead chase.
    """
    v = _qpik_rail_v_ff_m_s(qdot0)
    x = float(q_send0)
    if abs(v) >= _RAIL_V_IDLE_M_S:
        extra = max(0.0, float(dt_wall) - max(float(dt_nom), 0.0))
        x = x + v * extra
    lo = float(soft_lo)
    hi = float(soft_hi)
    if hi < lo:
        lo, hi = hi, lo
    x = max(lo, min(hi, x))
    lead = max(float(lead_max_m), 0.0)
    if (
        abs(v) >= _RAIL_V_IDLE_M_S
        and lead > 0.0
        and meas_m is not None
        and math.isfinite(float(meas_m))
    ):
        meas = float(meas_m)
        x = max(meas - lead, min(meas + lead, x))
    return x


def _publish_rail_target_before_arm(
    rail_bridge,
    target_m: float,
    fault_stop,
    v_ff_m_s: float | None = None,
) -> tuple[bool, str]:
    """Require the rail to accept this 8D tick before publishing the arm half."""

    if rail_bridge is None or not getattr(rail_bridge, "enabled", False):
        return True, ""
    if not bool(getattr(rail_bridge, "calibrated", False)):
        reason = "rail_target_rejected:not_calibrated"
    elif bool(getattr(rail_bridge, "panicked", False)):
        detail = str(getattr(rail_bridge, "panic_reason", "") or "panic")
        reason = (
            f"rail_target_rejected:{detail}; "
            "restart Window A to re-arm (panic latches)"
        )
    elif not bool(getattr(rail_bridge, "armed", False)):
        reason = "rail_target_rejected:not_armed; restart Window A to re-arm"
    else:
        try:
            try:
                accepted = rail_bridge.set_target_m(
                    float(target_m), v_ff_m_s=v_ff_m_s
                )
            except TypeError:
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
    physical_saturated: bool = False,
) -> float:
    """Raw governor scale in [0, 1] (min of active bands); filter in GovernorFilter.

    Cartesian error only slows the clock when a joint/rail is physically
    saturated; otherwise tracking lag from a bad IK posture must not crawl
    the reference to ~4% speed.  A floor (``governor_scale_min``) still
    applies whenever a band is active.
    """
    scales: list[float] = []
    floor = float(getattr(phase, "governor_scale_min", 0.0) or 0.0)

    if phase.governor_joint_err_max_deg > 0.0 and joint_err_deg is not None:
        e0, e1 = phase.governor_joint_err_ok_deg, phase.governor_joint_err_max_deg
        if e1 > e0:
            scales.append(float(np.clip((e1 - joint_err_deg) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if phase.governor_err_max_mm > 0.0 and outer_err_mm is not None:
        if physical_saturated:
            e0, e1 = phase.governor_err_ok_mm, phase.governor_err_max_mm
            if e1 > e0:
                scales.append(float(np.clip((e1 - outer_err_mm) / (e1 - e0), 0.0, 1.0)))
        else:
            scales.append(1.0)

    if not scales:
        return 1.0
    return max(float(min(scales)), floor)


class GovernorFilter:
    """First-order LPF + freeze hysteresis on the governor scale."""

    def __init__(
        self,
        tau_s: float = 0.2,
        freeze_below: float = 0.02,
        release_above: float = 0.10,
        scale_min: float = 0.0,
    ) -> None:
        self.tau_s = float(tau_s)
        self.freeze_below = float(freeze_below)
        self.release_above = float(release_above)
        self.scale_min = float(scale_min)
        self.scale = 1.0
        self.frozen = False

    def update(self, raw: float, dt: float) -> float:
        floor = float(getattr(self, "scale_min", 0.0) or 0.0)
        raw = float(np.clip(raw, floor, 1.0))
        alpha = 1.0 if self.tau_s <= 0.0 else min(1.0, dt / self.tau_s)
        self.scale += alpha * (raw - self.scale)
        freeze_below = float(self.freeze_below)
        if floor >= freeze_below:
            self.frozen = False
            return float(np.clip(self.scale, floor, 1.0))
        if self.frozen:
            if raw >= self.release_above and self.scale >= self.release_above:
                self.frozen = False
        elif self.scale <= freeze_below:
            self.frozen = True
        if self.frozen:
            return 0.0
        return float(np.clip(self.scale, floor, 1.0))


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
    """Gate rail/CANFD publication.  A failed QP1 has no certified command."""
    if bool(step.solver_fault_latched) or str(step.fallback_level) == "stop":
        reason = f"qpik_fault:{step.fallback_level}:{step.fallback_reason}"
        fault_stop(reason)
        return False, reason
    return True, ""


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
            rail_feedback_ready = False
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
                        scale_min=float(getattr(phase, "governor_scale_min", 0.25)),
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
                    # TCP velocity from SDK joint_speed (rail from servo / Δq0).
                    last_feedback_seq = int(getattr(snap, "seq", 0))
                    last_feedback_t = float(getattr(snap, "t_s", 0.0))
                    last_feedback_q = np.asarray(q_meas, dtype=float).copy()
                    # ``feedback_age_s`` tracks the last sample from which a
                    # finite-difference TCP velocity was actually computed;
                    # sensor transport age is a separate diagnostic.
                    last_feedback_velocity_t = last_feedback_t
                    twist_achieved_base = np.zeros(6, dtype=float)
                    qdot_meas = None
                    v_tcp_z_actual = 0.0
                    feedback_velocity_valid = False
                    feedback_fresh_tick = False
                    first_tick = True
                    last_log_ms = float("nan")
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
                        try:
                            rail_exec_estimate = _rail_execution_velocity_estimate(
                                rail_bridge,
                                now_s=time.monotonic(),
                                freshness_s=float(inner.cfg.feedback_timeout_s),
                                require_fresh=rail_feedback_ready,
                            )
                        except RuntimeError as exc:
                            phase_stopped = True
                            stop_reason = f"rail_feedback_fault:{exc}"
                            _fault_stop(stop_reason)
                            break
                        if rail_exec_estimate is not None:
                            rail_feedback_ready = True
                        if snap.pose is not None:
                            pose_rm = snap.pose
                        if snap.q_deg is not None:
                            try:
                                rail_measured_m = (
                                    float(rail_exec_estimate.position_m)
                                    if rail_exec_estimate is not None
                                    else _rail_m_for_feedback(rail_bridge, inner)
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
                                qdot_meas = _qdot_meas_8dof(
                                    q_new,
                                    last_feedback_q,
                                    dt_feedback,
                                    snap,
                                    rail_bridge,
                                    rail_velocity_m_s=(
                                        rail_exec_estimate.velocity_m_s
                                        if rail_exec_estimate is not None
                                        else None
                                    ),
                                )
                                if qdot_meas is not None:
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
                        pose_d_ref = getattr(phase.outer, "last_pose_d", None)
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
                        _t_inner0 = time.perf_counter()
                        step = inner.update(
                            twist,
                            control_dt,
                            q_meas=q_meas,
                            qdot_ff=qdot_ff,
                            vel_ff=vel_ff_ref,
                            pose_d=pose_d_ref,
                            f_ext_z=f_ext_z if math.isfinite(f_ext_z) else None,
                            f_des_z=f_des_z if math.isfinite(f_des_z) else None,
                            contact_active=bool(
                                getattr(ctrl, "contact_present", False)
                                if ctrl is not None
                                else False
                            ),
                            path_twist=path_twist,
                            feedback_twist=feedback_twist,
                            v_force_z=(
                                float(getattr(ctrl, "v_force_z", float("nan")))
                                if ctrl is not None
                                else None
                            ),
                            rail_exec_vel_m_s=(
                                rail_exec_estimate.velocity_m_s
                                if rail_exec_estimate is not None
                                else None
                            ),
                            rail_exec_smooth_m_s=(
                                rail_exec_estimate.commanded_velocity_m_s
                                if rail_exec_estimate is not None
                                else None
                            ),
                        )
                        if rail_exec_estimate is not None:
                            step.rail_exec_velocity_m_s = float(
                                rail_exec_estimate.velocity_m_s
                            )
                            step.rail_measured_velocity_m_s = float(
                                rail_exec_estimate.measured_velocity_m_s
                            )
                            step.rail_commanded_velocity_m_s = float(
                                rail_exec_estimate.commanded_velocity_m_s
                            )
                            step.rail_commanded_acceleration_m_s2 = float(
                                rail_exec_estimate.commanded_acceleration_m_s2
                            )
                            step.rail_feedback_age_s = float(rail_exec_estimate.age_s)
                        step.tick_inner_ms = (
                            time.perf_counter() - _t_inner0
                        ) * 1000.0
                        step.pre_solve_feedback_age_s = sensor_age_s
                        # A hard-construction/final-validation fault is acted on before the
                        # rail target or CANFD joint command can be published.
                        sendable, qpik_stop_reason = _guard_qpik_step_before_send(
                            step, _fault_stop
                        )
                        if not sendable:
                            phase_stopped = True
                            stop_reason = qpik_stop_reason
                            if logger is not None:
                                rail_meas = float("nan")
                                if (
                                    rail_bridge is not None
                                    and rail_bridge.enabled
                                ):
                                    try:
                                        rail_meas = float(rail_bridge.measured_m)
                                    except Exception:
                                        rail_meas = float("nan")
                                step.tick_log_ms = last_log_ms
                                logger.write(
                                    now - total_t0,
                                    phase.label,
                                    t_ref,
                                    step,
                                    q_meas,
                                    pose_pin,
                                    f_ext,
                                    outer=phase.outer,
                                    rail_meas_m=rail_meas,
                                    dt_actual_s=dt_wall_actual,
                                    sensor_age_s=sensor_age_s,
                                    feedback_age_s=feedback_age_s,
                                    feedback_fresh_tick=feedback_fresh_tick,
                                    f_ext_raw=f_ext_raw,
                                    twist_achieved_base=twist_achieved_base,
                                    v_tcp_z_actual=v_tcp_z_actual,
                                    qdot_meas=qdot_meas,
                                )
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
                        _t_send0 = time.perf_counter()
                        qdot0_pub = _qpik_rail_v_ff_m_s(
                            float(np.asarray(step.qdot, dtype=float).reshape(-1)[0])
                        )
                        rail_meas_pub = float("nan")
                        if rail_bridge is not None and getattr(rail_bridge, "enabled", False):
                            try:
                                rail_meas_pub = float(rail_bridge.measured_m)
                            except Exception:
                                rail_meas_pub = float("nan")
                        rail_pub_m = _wall_clock_rail_target(
                            float(step.q_send[0]),
                            qdot0_pub,
                            dt_wall_actual,
                            float(inner.cfg.dt),
                            soft_lo=float(inner.limits.q_lower[0]),
                            soft_hi=float(inner.limits.q_upper[0]),
                            meas_m=rail_meas_pub,
                            lead_max_m=float(inner.cfg.resync_err_rail_m),
                        )
                        rail_ok, rail_reason = _publish_rail_target_before_arm(
                            rail_bridge,
                            float(rail_pub_m),
                            _fault_stop,
                            v_ff_m_s=qdot0_pub,
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
                        step.tick_send_ms = (
                            time.perf_counter() - _t_send0
                        ) * 1000.0
    
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
                            physical_saturated=bool(step.physical_saturated),
                        )
                        scale = gov_filter.update(raw_scale, control_dt)
                        # Soft-start ramp: first ~0.3s cannot command near-vmax.
                        ramp_s = float(getattr(phase, "soft_start_ramp_s", 0.0) or 0.0)
                        if ramp_s > 1e-6 and step.controller_mode != "direct_joint_ptp":
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
                            # The write cannot time itself into its own row, so
                            # carry the previous tick's cost; over a run the
                            # statistics are the same.
                            step.tick_log_ms = last_log_ms
                            _t_log0 = time.perf_counter()
                            logger.write(
                                now - total_t0, phase.label, t_ref, step, q_meas, pose_pin, f_ext,
                                outer=phase.outer,
                                governor_scale=scale,
                                governor_scale_raw=raw_scale,
                                v_max=inner.limits.v_max,
                                rail_meas_m=rail_meas,
                                rail_target_sent_m=rail_pub_m,
                                dt_actual_s=dt_wall_actual,
                                sensor_age_s=sensor_age_s,
                                feedback_age_s=feedback_age_s,
                                feedback_fresh_tick=feedback_fresh_tick,
                                f_ext_raw=f_ext_raw,
                                twist_achieved_base=twist_achieved_base,
                                v_tcp_z_actual=v_tcp_z_actual,
                                qdot_meas=qdot_meas,
                            )
                            last_log_ms = (
                                time.perf_counter() - _t_log0
                            ) * 1000.0
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

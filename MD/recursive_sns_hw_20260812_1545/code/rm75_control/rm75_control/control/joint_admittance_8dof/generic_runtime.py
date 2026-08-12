"""Trajectory-agnostic orchestration of measured-state P0 + two-level QPIK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    LinearConstraintSet,
    PostureGuide,
    ProtectedTask,
    RobotState,
    ScalableTask,
)
from rm75_control.control.joint_admittance_8dof.health_metrics import (
    compute_health_metrics,
)
from rm75_control.control.joint_admittance_8dof.health_monitor import (
    HealthMonitor,
    HealthReport,
    HealthState,
    HealthThresholds,
)
from rm75_control.control.joint_admittance_8dof.solver.p0_safety import P0SafetyBuilder
from rm75_control.control.joint_admittance_8dof.solver.two_level_qpik import (
    TwoLevelQpikConfig,
    TwoLevelQpikController,
    TwoLevelQpikResult,
)
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    CartesianTaskProfile,
    TaskSpaceConstraintRow,
    build_cartesian_tasks,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


@dataclass
class GenericQpikRuntimeConfig:
    solver: TwoLevelQpikConfig = field(default_factory=TwoLevelQpikConfig)
    task_profile: CartesianTaskProfile = field(
        default_factory=CartesianTaskProfile.all_protected
    )
    health: HealthThresholds = field(default_factory=HealthThresholds)
    rail_indices: tuple[int, ...] = (0,)
    wrist_indices: tuple[int, ...] = (5, 6, 7)
    # Characteristic allowable task velocities used only for dimensionless
    # health normalisation, not as task semantics.
    task_velocity_scales: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.10, 0.10, 0.50, 0.50, 0.50])
    )
    # Compatibility adapter only: applications that still pass a scalar
    # over-force signal may declare which task-frame velocity row becomes a
    # one-sided safety row.  ``None`` disables the adapter.
    overforce_task_row: int | None = None
    overforce_positive_is_unsafe: bool = True
    # Arm-dexterity CBF (velocity-normalized arm Jacobian condition ratio).
    dexterity_d_safe: float = 0.04
    dexterity_gamma: float = 5.0
    dexterity_d_activate: float = 0.08
    # Proactive dexterity pull in P3 comfort guide (not P0 barrier).
    dexterity_k_d: float = 0.15
    # Joint working-set CBF (inner envelope inside hard limits).
    working_arm_margin_rad: float = 0.30  # ~17° — inside health danger 15°
    working_rail_margin_m: float = 0.02
    working_gamma: float = 8.0


@dataclass(frozen=True)
class GenericQpikRuntimeResult:
    qdot: np.ndarray
    solver: TwoLevelQpikResult
    p0: LinearConstraintSet
    protected: ProtectedTask
    scalable: tuple[ScalableTask, ...]
    health: HealthReport
    jacobian_base: np.ndarray


class GenericQpikRuntime:
    """Pure one-tick runtime used by hardware, simulation and log replay."""

    def __init__(
        self,
        kin,
        limits: SafetyLimits,
        config: GenericQpikRuntimeConfig | None = None,
        *,
        collision=None,
        collision_config: CollisionConfig | None = None,
        damper_band: float | np.ndarray = 0.15,
    ) -> None:
        self.kin = kin
        self.limits = limits
        self.config = config or GenericQpikRuntimeConfig()
        self.solver = TwoLevelQpikController(int(kin.nv), self.config.solver)
        self.p0_builder = P0SafetyBuilder(
            kin,
            limits,
            collision_config=collision_config,
            collision=collision,
            damper_band=damper_band,
        )
        self._damper_band = np.broadcast_to(
            np.asarray(damper_band, dtype=float), (int(kin.nv),)
        ).astype(float).copy()
        arm_indices = tuple(
            i for i in range(int(kin.nv)) if i not in set(self.config.rail_indices)
        )
        self.health_monitor = HealthMonitor(
            self.config.health,
            q_lower=limits.q_lower,
            q_upper=limits.q_upper,
            joint_indices=arm_indices,
            wrist_indices=self.config.wrist_indices,
        )
        self._last_solver_fault = False

    @property
    def backend_name(self) -> str:
        return self.solver.backend_name

    @property
    def qdot_prev(self) -> np.ndarray:
        return self.solver.qdot_prev

    def reset(self) -> None:
        self.solver.reset()
        self.health_monitor.reset()
        self._last_solver_fault = False

    def sync_applied(self, qdot: np.ndarray) -> None:
        self.solver.sync_applied(qdot)

    def set_collision_enabled(self, enabled: bool) -> None:
        self.p0_builder.set_collision_enabled(enabled)

    def solve(
        self,
        state: RobotState,
        *,
        twist_task: np.ndarray,
        rotation_base_task: np.ndarray,
        profile: CartesianTaskProfile | None = None,
        posture_guide: PostureGuide | object | None = None,
        task_safety_rows: Sequence[TaskSpaceConstraintRow] = (),
        application_hard_rows: Sequence[HardConstraintRow] = (),
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        psi_soft: object | None = None,
        heartbeat=None,
    ) -> GenericQpikRuntimeResult:
        # This is the only Jacobian evaluation used for task rows and health.
        J_base = np.asarray(self.kin.jacobian(state.q_meas), dtype=float)
        health = self._update_health(state, J_base)
        active_profile = profile or self.config.task_profile
        protected, scalable = build_cartesian_tasks(
            J_base,
            twist_task,
            rotation_base_task,
            active_profile,
            one_sided_rows=task_safety_rows,
            # Keep recovery slack active through SETTLING; changing back to
            # tight limits is then governed by the health dwell rather than a
            # one-tick threshold crossing.
            recovery=health.state in (HealthState.RECOVERY, HealthState.SETTLING),
        )
        return self._solve_prepared(
            state,
            protected=protected,
            scalable=scalable,
            posture_guide=posture_guide,
            application_hard_rows=application_hard_rows,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            health=health,
            jacobian_base=J_base,
            psi_soft=psi_soft,
            heartbeat=heartbeat,
        )

    def solve_tasks(
        self,
        state: RobotState,
        *,
        protected: ProtectedTask,
        scalable: Sequence[ScalableTask] = (),
        posture_guide: PostureGuide | object | None = None,
        application_hard_rows: Sequence[HardConstraintRow] = (),
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        psi_soft: object | None = None,
        heartbeat=None,
    ) -> GenericQpikRuntimeResult:
        """Solve caller-provided generic rows without Cartesian assumptions.

        Task producers may supply arbitrary linear motion rows.  The runtime
        still computes one measured-state arm-health snapshot for scheduling
        and telemetry, but it does not reinterpret or rotate these tasks.
        """

        if protected.n_vars != int(self.kin.nv):
            raise ValueError(
                f"protected task has {protected.n_vars} variables, expected {self.kin.nv}"
            )
        task_tuple = tuple(scalable)
        if any(task.n_vars != int(self.kin.nv) for task in task_tuple):
            raise ValueError("all scalable tasks must match the robot velocity dimension")
        J_base = np.asarray(self.kin.jacobian(state.q_meas), dtype=float)
        health = self._update_health(state, J_base)
        return self._solve_prepared(
            state,
            protected=protected,
            scalable=task_tuple,
            posture_guide=posture_guide,
            application_hard_rows=application_hard_rows,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            health=health,
            jacobian_base=J_base,
            psi_soft=psi_soft,
            heartbeat=heartbeat,
        )

    def _update_health(
        self,
        state: RobotState,
        jacobian_base: np.ndarray,
    ) -> HealthReport:
        metrics = compute_health_metrics(
            jacobian_base=jacobian_base,
            q_meas=state.q_meas,
            q_lower=self.limits.q_lower,
            q_upper=self.limits.q_upper,
            velocity_limits=self.limits.v_max,
            rail_indices=self.config.rail_indices,
            wrist_indices=self.config.wrist_indices,
            task_velocity_scales=self.config.task_velocity_scales,
            solver_ok=not self._last_solver_fault,
        )
        return self.health_monitor.update(
            arm_rho=metrics.arm_health,
            joint_margin_rad=metrics.joint_margin,
            wrist_margin_rad=metrics.wrist_margin,
            dt=state.dt,
            solver_fault=self._last_solver_fault,
        )

    def _solve_prepared(
        self,
        state: RobotState,
        *,
        protected: ProtectedTask,
        scalable: Sequence[ScalableTask],
        posture_guide: PostureGuide | object | None,
        application_hard_rows: Sequence[HardConstraintRow],
        resync_err: float | np.ndarray,
        rail_locked: bool,
        rail_lock_vel_eps_m_s: float,
        rail_vel_pin_m_s: float | None,
        health: HealthReport,
        jacobian_base: np.ndarray,
        psi_soft: object | None = None,
        heartbeat=None,
    ) -> GenericQpikRuntimeResult:
        # α is decided by QP2 feasibility (SNS), never by arm_health.
        # arm_rho only scales regularization for numeric stability near singular.
        _, reg_scale = self._reg_scale_from_arm_health(health.arm_rho)
        app_rows = list(application_hard_rows)
        app_rows.extend(self._joint_working_cbf_rows(state))
        app_rows.extend(
            self._dexterity_and_branch_rows(state, health, jacobian_base)
        )
        p0 = self.p0_builder.build(
            state,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            application_rows=app_rows,
            measured_kinematics_ready=True,
        )
        solved = self.solver.solve(
            state,
            protected,
            scalable_tasks=tuple(scalable),
            posture_guide=posture_guide,
            hard_constraints=p0,
            alpha_cap=1.0,
            reg_scale=reg_scale,
            q_lower=self.limits.q_lower,
            q_upper=self.limits.q_upper,
            margin_band=self._damper_band,
            psi_soft=psi_soft,
            heartbeat=heartbeat,
            velocity_scale=self.limits.v_max,
        )
        # Soft QP1 fallbacks are not latched faults; only true P0 impossibility is.
        self._last_solver_fault = bool(solved.fault_latched)
        if self._last_solver_fault:
            health = self.health_monitor.update(
                arm_rho=health.arm_rho,
                joint_margin_rad=health.joint_margin_rad,
                wrist_margin_rad=health.wrist_margin_rad,
                dt=state.dt,
                solver_fault=True,
                reason=solved.fallback_reason,
            )
        return GenericQpikRuntimeResult(
            qdot=np.asarray(solved.qdot, dtype=float).copy(),
            solver=solved,
            p0=p0,
            protected=protected,
            scalable=tuple(scalable),
            health=health,
            jacobian_base=np.asarray(jacobian_base, dtype=float).copy(),
        )

    def _reg_scale_from_arm_health(
        self, arm_rho: float | None, *, reg_max: float = 50.0
    ) -> tuple[float, float]:
        """Return ``(alpha_cap=1.0, reg_scale)`` from arm dexterity.

        α is never capped by health — only regularization grows as ρ drops
        (numeric damping near singularity).  Missing ρ → healthy defaults.
        """

        if arm_rho is None or not np.isfinite(float(arm_rho)):
            return 1.0, 1.0
        danger = float(self.config.health.arm_danger)
        full = float(self.config.health.arm_warn)
        span = max(full - danger, 1.0e-9)
        s = float(np.clip((float(arm_rho) - danger) / span, 0.0, 1.0))
        reg_max = float(max(reg_max, 1.0))
        return 1.0, 1.0 + (reg_max - 1.0) * (1.0 - s)

    # Backward-compatible alias (tests / older callers).
    _authority_from_arm_health = _reg_scale_from_arm_health

    def _joint_working_cbf_rows(self, state: RobotState) -> list[HardConstraintRow]:
        """P0 working-set CBF: q̇_i ≤ γ (q_work⁺ − q_i) (and lower mirror)."""

        n = int(self.kin.nv)
        q = np.asarray(state.q_meas, dtype=float).reshape(-1)
        if q.size != n:
            return []
        arm_m = float(getattr(self.config, "working_arm_margin_rad", 0.30))
        rail_m = float(getattr(self.config, "working_rail_margin_m", 0.02))
        gamma = float(getattr(self.config, "working_gamma", 8.0))
        rail_set = {int(i) for i in self.config.rail_indices}
        lo = np.asarray(self.limits.q_lower, dtype=float).reshape(-1)
        hi = np.asarray(self.limits.q_upper, dtype=float).reshape(-1)
        # Clamp to this tick's accel-aware velocity box so CBF never empties P0.
        try:
            box_lo, box_hi = self.p0_builder.velocity_box.bounds(
                state.q_meas,
                state.dt,
                state.qdot_applied_prev,
                q_meas=state.q_meas,
                q_cmd=state.q_cmd,
            )
            box_lo = np.asarray(box_lo, dtype=float).reshape(-1)
            box_hi = np.asarray(box_hi, dtype=float).reshape(-1)
        except Exception:
            v_max = np.asarray(self.limits.v_max, dtype=float).reshape(-1)
            box_hi = np.maximum(np.abs(v_max), 1.0e-6)
            box_lo = -box_hi
        rows: list[HardConstraintRow] = []
        for i in range(n):
            m = rail_m if i in rail_set else arm_m
            if m <= 0.0:
                continue
            q_lo_w = float(lo[i] + m)
            q_hi_w = float(hi[i] - m)
            if q_lo_w >= q_hi_w:
                continue
            a = np.zeros(n, dtype=float)
            a[i] = 1.0
            # q̇ ≤ γ (q_work+ − q)  and  q̇ ≥ −γ (q − q_work−)
            cbf_lo = -gamma * (float(q[i]) - q_lo_w)
            cbf_hi = gamma * (q_hi_w - float(q[i]))
            v_lo = float(box_lo[i]) if i < box_lo.size else -1.0
            v_hi = float(box_hi[i]) if i < box_hi.size else 1.0
            if not np.isfinite(v_lo):
                v_lo = -1.0e6
            if not np.isfinite(v_hi):
                v_hi = 1.0e6
            # Outside the working envelope: block further intrusion (bound at 0)
            # but keep q̇=0 feasible. Forced leave rates made ProxQP timeouts
            # cascade into final_qdot_violates_p0 hard stops mid-scan; leave
            # recovery to P3 soft margin / dexterity guide instead.
            if cbf_hi < 0.0:
                cbf_hi = 0.0
            if cbf_lo > 0.0:
                cbf_lo = 0.0
            cbf_lo = max(cbf_lo, v_lo)
            cbf_hi = min(cbf_hi, v_hi)
            if cbf_lo > cbf_hi + 1.0e-12:
                continue
            rows.append(
                HardConstraintRow(
                    a,
                    lower=cbf_lo,
                    upper=cbf_hi,
                    name=f"joint_working_cbf:{i}",
                )
            )
        return rows

    def _dexterity_and_branch_rows(
        self,
        state: RobotState,
        health: HealthReport,
        jacobian_base: np.ndarray,
    ) -> list[HardConstraintRow]:
        """P0 set rows: arm-dexterity CBF + optional elbow-branch separatrix."""

        from rm75_control.control.joint_admittance_8dof.health_metrics import (
            arm_dexterity_gradient,
        )

        rows: list[HardConstraintRow] = []
        n = int(self.kin.nv)
        d_safe = float(getattr(self.config, "dexterity_d_safe", 0.04))
        gamma = float(getattr(self.config, "dexterity_gamma", 5.0))
        d_activate = float(
            getattr(self.config, "dexterity_d_activate", max(2.0 * d_safe, d_safe + 0.02))
        )
        d_arm = health.arm_rho
        # Skip FD gradient when well above the barrier (CBF inactive).
        if (
            d_arm is not None
            and np.isfinite(float(d_arm))
            and float(d_arm) < d_activate
        ):
            grad = arm_dexterity_gradient(
                self.kin,
                state.q_meas,
                velocity_limits=self.limits.v_max,
                rail_indices=self.config.rail_indices,
                task_velocity_scales=self.config.task_velocity_scales,
            )
            if grad is not None and float(np.dot(grad, grad)) > 1.0e-12:
                h = float(d_arm) - d_safe
                # ∇d^T q̇ ≥ −γ h.  When already below d_safe (h<0), keep q̇=0
                # feasible (∇d^T q̇ ≥ 0) so ProxQP timeouts cannot hard-stop.
                # Active recovery remains P3 k_d ∇d.
                lower = -gamma * h
                if lower > 0.0:
                    lower = 0.0
                rows.append(
                    HardConstraintRow(
                        grad,
                        lower=lower,
                        upper=None,
                        name="arm_dexterity_cbf",
                    )
                )

        # Elbow branch latch: prevent undeclared crossing of q4≈0 separatrix.
        elbow = 4  # full-q index (rail=0, arm J1..=1..)
        if 0 <= elbow < n:
            q4 = float(state.q_meas[elbow])
            sign = getattr(self, "_elbow_branch_sign", None)
            if sign is None:
                if abs(q4) > np.deg2rad(8.0):
                    self._elbow_branch_sign = 1.0 if q4 > 0.0 else -1.0
                    sign = self._elbow_branch_sign
            if sign is not None and abs(q4) < np.deg2rad(25.0):
                # Keep sign(q4) consistent: sign * qdot4 >= -γ * max(sign*q4, 0)
                # Equivalent: when sign>0, qdot4 >= -γ q4 (don't drive through 0 fast)
                a = np.zeros(n, dtype=float)
                a[elbow] = float(sign)
                h_br = max(float(sign) * q4, 0.0)
                rows.append(
                    HardConstraintRow(
                        a,
                        lower=-gamma * h_br,
                        upper=None,
                        name="elbow_branch_separatrix",
                    )
                )
        return rows

    # Stable spelling used by instrumentation that wraps the inner solve.
    step = solve


__all__ = [
    "GenericQpikRuntime",
    "GenericQpikRuntimeConfig",
    "GenericQpikRuntimeResult",
]

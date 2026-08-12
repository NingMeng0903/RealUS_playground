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
    ) -> GenericQpikRuntimeResult:
        p0 = self.p0_builder.build(
            state,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            application_rows=application_hard_rows,
            measured_kinematics_ready=True,
        )
        solved = self.solver.solve(
            state,
            protected,
            scalable_tasks=tuple(scalable),
            posture_guide=posture_guide,
            hard_constraints=p0,
        )
        self._last_solver_fault = bool(solved.fault_latched or not solved.qp1.success)
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

    # Stable spelling used by instrumentation that wraps the inner solve.
    step = solve


__all__ = [
    "GenericQpikRuntime",
    "GenericQpikRuntimeConfig",
    "GenericQpikRuntimeResult",
]

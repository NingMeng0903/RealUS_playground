"""Fixed RM75 Cartesian runtime around the single-shot 28-variable QPIK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from rm75_control.control.joint_admittance_8dof.collision_model import CollisionConfig
from rm75_control.control.joint_admittance_8dof.generic_tasks import (
    HardConstraintRow,
    LinearConstraintSet,
    RobotState,
)
from rm75_control.control.joint_admittance_8dof.health_metrics import (
    arm_dexterity_gradient,
    compute_health_metrics,
)
from rm75_control.control.joint_admittance_8dof.health_monitor import (
    HealthMonitor,
    HealthReport,
    HealthThresholds,
)
from rm75_control.control.joint_admittance_8dof.solver.p0_safety import P0SafetyBuilder
from rm75_control.control.joint_admittance_8dof.solver.single_qpik import (
    CartesianQpCommand,
    SingleQpikConfig,
    SingleQpikController,
    SingleQpikResult,
)
from rm75_control.control.joint_admittance_8dof.task_adapter import (
    TaskSpaceConstraintRow,
    task_rotation_map,
)
from rm75_control.control.joint_admittance_8dof.utils.safety import SafetyLimits


@dataclass
class GenericQpikRuntimeConfig:
    solver: SingleQpikConfig = field(default_factory=SingleQpikConfig)
    health: HealthThresholds = field(default_factory=HealthThresholds)
    rail_indices: tuple[int, ...] = (0,)
    wrist_indices: tuple[int, ...] = (5, 6, 7)
    task_velocity_scales: np.ndarray = field(
        default_factory=lambda: np.array([0.10, 0.10, 0.10, 0.50, 0.50, 0.50])
    )
    dexterity_d_safe: float = 0.04
    dexterity_gamma: float = 5.0
    dexterity_d_activate: float = 0.08
    dexterity_k_d: float = 0.15
    collision_k_d: float = 0.10
    working_arm_margin_rad: float = 0.30
    working_rail_margin_m: float = 0.02
    working_gamma: float = 8.0
    arm_nominal_k: float = 0.25
    arm_nominal_qdot_max: float = 0.30
    risk_attack_s: float = 0.05
    risk_release_s: float = 0.40
    risk_exit_dwell_s: float = 0.20
    gradient_period_ticks: int = 10
    gradient_lpf_tau_s: float = 0.10
    wrist_danger_deg: float = 10.0
    wrist_warn_deg: float = 20.0
    wrist_exit_deg: float = 25.0
    rail_macro_tau_s: float = 0.15
    rail_macro_v_max_m_s: float = 0.12
    rail_macro_a_max_m_s2: float = 0.30
    rail_macro_jerk_max_m_s3: float = 2.0
    rail_center_k: float = 0.04
    rail_center_v_max_m_s: float = 0.025
    feedback_lpf_tau_s: float = 0.05
    feedback_accel_max_m_s2: float = 0.30


@dataclass(frozen=True)
class GenericQpikRuntimeResult:
    qdot: np.ndarray
    solver: SingleQpikResult
    p0: LinearConstraintSet
    health: HealthReport
    jacobian_base: np.ndarray
    command: CartesianQpCommand
    protected_target: np.ndarray
    protected_jacobian: np.ndarray
    scan_target: np.ndarray
    scan_jacobian: np.ndarray
    wrist_singularity: float
    rail_macro_preference: float
    rail_center_preference: float
    arm_risk_preference_norm: float
    feedback_xy_raw: np.ndarray
    feedback_xy_filtered: np.ndarray
    risk_preference: np.ndarray
    risk_direction_cosine: float


class GenericQpikRuntime:
    """One measured snapshot, one fixed QP, and one validated publication candidate."""

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
        if int(kin.nv) != 8:
            raise ValueError("fixed RM75 QPIK requires exactly 8 velocity DOFs")
        self.kin = kin
        self.limits = limits
        self.config = config or GenericQpikRuntimeConfig()
        self.solver = SingleQpikController(limits.v_max, self.config.solver)
        self.p0_builder = P0SafetyBuilder(
            kin,
            limits,
            collision_config=collision_config,
            collision=collision,
            damper_band=damper_band,
        )
        arm_indices = tuple(
            index
            for index in range(int(kin.nv))
            if index not in set(self.config.rail_indices)
        )
        self.health_monitor = HealthMonitor(
            self.config.health,
            q_lower=limits.q_lower,
            q_upper=limits.q_upper,
            joint_indices=arm_indices,
            wrist_indices=self.config.wrist_indices,
        )
        self._arm_nominal: np.ndarray | None = None
        self._elbow_branch_sign: float | None = None
        self._wrist_branch_sign: float | None = None
        self._arm_gradient = np.zeros(8)
        self._arm_gradient_valid = False
        self._tick = 0
        self._risk_level = 0.0
        self._healthy_dwell_s = 0.0
        self._rail_macro = 0.0
        self._rail_macro_acceleration = 0.0
        self._feedback_xy = np.zeros(2)
        self._risk_preference_prev = np.zeros(8)

    @property
    def backend_name(self) -> str:
        return self.solver.backend_name

    @property
    def qdot_prev(self) -> np.ndarray:
        return self.solver.qdot_prev

    def reset(self, q_seed: np.ndarray | None = None) -> None:
        self.solver.reset()
        self.health_monitor.reset()
        self._arm_nominal = None
        self._elbow_branch_sign = None
        self._wrist_branch_sign = None
        self._arm_gradient.fill(0.0)
        self._arm_gradient_valid = False
        self._tick = 0
        self._risk_level = 0.0
        self._healthy_dwell_s = 0.0
        self._rail_macro = 0.0
        self._rail_macro_acceleration = 0.0
        self._feedback_xy.fill(0.0)
        self._risk_preference_prev.fill(0.0)
        if q_seed is not None:
            self._initialize_branch_signs(q_seed)

    def _initialize_branch_signs(self, q_seed: np.ndarray) -> None:
        """Capture the two analytic branch signs from a phase-independent pose."""

        q = np.asarray(q_seed, dtype=float).reshape(8)
        if not np.all(np.isfinite(q)):
            raise ValueError("branch seed must be finite")
        self._elbow_branch_sign = 1.0 if q[4] >= 0.0 else -1.0
        self._wrist_branch_sign = 1.0 if q[6] >= 0.0 else -1.0

    def begin_hybrid_episode(
        self,
        q_meas: np.ndarray,
        qdot_applied: np.ndarray,
    ) -> None:
        """Reset phase-local policy state while preserving velocity continuity."""

        q = np.asarray(q_meas, dtype=float).reshape(8)
        applied = np.asarray(qdot_applied, dtype=float).reshape(8)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(applied)):
            raise ValueError("hybrid episode seed must be finite")
        previous_wrist_sign = self._wrist_branch_sign
        self.health_monitor.reset()
        self._arm_nominal = None
        previous_elbow_sign = self._elbow_branch_sign
        self._elbow_branch_sign = previous_elbow_sign
        self._wrist_branch_sign = previous_wrist_sign
        if self._elbow_branch_sign is None:
            self._elbow_branch_sign = 1.0 if q[4] >= 0.0 else -1.0
        wrist_metric = abs(float(np.sin(q[6])))
        if self._wrist_branch_sign is None:
            self._wrist_branch_sign = 1.0 if q[6] > 0.0 else -1.0
        self._arm_gradient.fill(0.0)
        self._arm_gradient_valid = False
        self._tick = 0
        self._risk_level = 0.0
        self._healthy_dwell_s = 0.0
        self._rail_macro = 0.0
        self._rail_macro_acceleration = 0.0
        self._feedback_xy.fill(0.0)
        self._risk_preference_prev.fill(0.0)
        self.solver.reset()
        self.solver.sync_applied(applied)

    def sync_applied(self, qdot: np.ndarray) -> None:
        self.solver.sync_applied(qdot)

    def set_collision_enabled(self, enabled: bool) -> None:
        self.p0_builder.set_collision_enabled(enabled)

    @staticmethod
    def _task_jacobian(jacobian_base: np.ndarray, rotation_base_task: np.ndarray) -> np.ndarray:
        return task_rotation_map(rotation_base_task) @ np.asarray(
            jacobian_base, dtype=float
        )

    def _update_health(self, state: RobotState, jacobian_base: np.ndarray) -> HealthReport:
        metrics = compute_health_metrics(
            jacobian_base=jacobian_base,
            q_meas=state.q_meas,
            q_lower=self.limits.q_lower,
            q_upper=self.limits.q_upper,
            velocity_limits=self.limits.v_max,
            rail_indices=self.config.rail_indices,
            wrist_indices=self.config.wrist_indices,
            task_velocity_scales=self.config.task_velocity_scales,
            solver_ok=True,
        )
        return self.health_monitor.update(
            arm_rho=metrics.arm_health,
            joint_margin_rad=metrics.joint_margin,
            wrist_margin_rad=metrics.wrist_margin,
            dt=state.dt,
            solver_fault=False,
        )

    def _working_bounds(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(8)
        hi = np.empty(8)
        gamma = max(float(self.config.working_gamma), 0.0)
        for index in range(8):
            margin = (
                float(self.config.working_rail_margin_m)
                if index in set(self.config.rail_indices)
                else float(self.config.working_arm_margin_rad)
            )
            lower = float(self.limits.q_lower[index] + margin)
            upper = float(self.limits.q_upper[index] - margin)
            lo[index] = -gamma * (float(q[index]) - lower)
            hi[index] = gamma * (upper - float(q[index]))
        return lo, hi

    @staticmethod
    def _smooth_level(current: float, target: float, dt: float, attack: float, release: float) -> float:
        tau = attack if target > current else release
        if tau <= 0.0:
            return float(target)
        gain = 1.0 - float(np.exp(-max(float(dt), 0.0) / tau))
        return float(current + gain * (target - current))

    def _risk_preference(
        self,
        state: RobotState,
        health: HealthReport,
        collision_warning_jacobian: np.ndarray,
        collision_warning_lower: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]:
        q = np.asarray(state.q_meas, dtype=float)
        cfg = self.config
        arm_rho = float(health.arm_rho) if health.arm_rho is not None else 1.0
        rho_span = max(float(cfg.dexterity_d_activate - cfg.dexterity_d_safe), 1.0e-6)
        rho_risk = float(
            np.clip((cfg.dexterity_d_activate - arm_rho) / rho_span, 0.0, 1.0)
        )
        wrist_index = 6
        wrist_metric = float(np.sqrt(np.sin(q[wrist_index]) ** 2 + 1.0e-8))
        wrist_danger = float(np.sin(np.deg2rad(cfg.wrist_danger_deg)))
        wrist_warn = float(np.sin(np.deg2rad(cfg.wrist_warn_deg)))
        wrist_exit = float(np.sin(np.deg2rad(cfg.wrist_exit_deg)))
        wrist_span = max(wrist_warn - wrist_danger, 1.0e-6)
        if self._wrist_branch_sign is None and wrist_metric >= wrist_warn:
            self._wrist_branch_sign = 1.0 if q[wrist_index] >= 0.0 else -1.0
        wrist_active = (
            self._wrist_branch_sign is not None
            and (wrist_metric < wrist_exit or self._risk_level > 1.0e-3)
        )
        wrist_risk = (
            float(np.clip((wrist_warn - wrist_metric) / wrist_span, 0.0, 1.0))
            if wrist_active
            else 0.0
        )
        margins = np.minimum(q - self.limits.q_lower, self.limits.q_upper - q)
        arm_margin = float(np.min(margins[1:]))
        margin_risk = float(
            np.clip(
                (float(cfg.working_arm_margin_rad) - arm_margin)
                / max(float(cfg.working_arm_margin_rad), 1.0e-6),
                0.0,
                1.0,
            )
        )
        collision_gradient = np.zeros(8)
        collision_risk = 0.0
        collision_span = max(
            float(self.p0_builder.collision_config.d_activate)
            - float(self.p0_builder.collision_config.d_safe),
            1.0e-6,
        )
        collision_gamma = max(
            float(self.p0_builder.collision_config.gamma), 1.0e-6
        )
        for gradient, lower in zip(
            np.asarray(collision_warning_jacobian, dtype=float),
            np.asarray(collision_warning_lower, dtype=float),
        ):
            norm = float(np.linalg.norm(gradient))
            if not np.isfinite(lower) or norm <= 1.0e-12:
                continue
            distance = (
                float(self.p0_builder.collision_config.d_safe)
                - float(lower) / collision_gamma
            )
            level = float(
                np.clip(
                    (float(self.p0_builder.collision_config.d_activate) - distance)
                    / collision_span,
                    0.0,
                    1.0,
                )
            )
            collision_risk = max(collision_risk, level)
            collision_gradient += level * gradient / norm

        raw_risk = max(rho_risk, wrist_risk, margin_risk, collision_risk)
        self._risk_level = self._smooth_level(
            self._risk_level,
            raw_risk,
            state.dt,
            float(cfg.risk_attack_s),
            float(cfg.risk_release_s),
        )
        if raw_risk <= 1.0e-6 and wrist_metric >= wrist_exit:
            self._healthy_dwell_s += state.dt
        else:
            self._healthy_dwell_s = 0.0

        self._tick += 1
        gradient_period = max(int(cfg.gradient_period_ticks), 1)
        gradient_due = self._tick % gradient_period == 0
        gradient_needed_now = (
            arm_rho < float(cfg.dexterity_d_activate)
            and not self._arm_gradient_valid
        )
        if gradient_due or gradient_needed_now:
            gradient = arm_dexterity_gradient(
                self.kin,
                q,
                velocity_limits=self.limits.v_max,
                rail_indices=cfg.rail_indices,
                task_velocity_scales=cfg.task_velocity_scales,
            )
            if gradient is not None:
                gradient = np.asarray(gradient, dtype=float)
                gradient_valid = bool(
                    np.all(np.isfinite(gradient))
                    and float(gradient @ gradient) > 1.0e-16
                )
            else:
                gradient_valid = False
            if gradient_valid:
                tau = max(float(cfg.gradient_lpf_tau_s), 0.0)
                if not self._arm_gradient_valid:
                    self._arm_gradient = gradient.copy()
                else:
                    gradient_dt = gradient_period * state.dt
                    gain = 1.0 if tau <= 0.0 else gradient_dt / (tau + gradient_dt)
                    self._arm_gradient += gain * (gradient - self._arm_gradient)
                self._arm_gradient_valid = True

        risk_gradient = np.zeros(8)
        risk_gradient += float(cfg.dexterity_k_d) * rho_risk * self._arm_gradient
        risk_gradient += float(cfg.collision_k_d) * collision_gradient
        wrist_sign = self._wrist_branch_sign
        if wrist_sign is None:
            wrist_sign = 1.0 if q[wrist_index] >= 0.0 else -1.0
        wrist_grad = wrist_sign * abs(np.cos(q[wrist_index]))
        risk_gradient[wrist_index] += float(cfg.dexterity_k_d) * wrist_risk * wrist_grad
        for index in range(1, 8):
            lower_distance = max(float(q[index] - self.limits.q_lower[index]), 1.0e-4)
            upper_distance = max(float(self.limits.q_upper[index] - q[index]), 1.0e-4)
            joint_gradient = 1.0 / lower_distance**2 - 1.0 / upper_distance**2
            risk_gradient[index] += (
                0.0025 * margin_risk * np.clip(joint_gradient, -40.0, 40.0)
            )

        dex_lower = -np.inf
        if (
            arm_rho < float(cfg.dexterity_d_activate)
            and self._arm_gradient_valid
            and float(self._arm_gradient @ self._arm_gradient) > 1.0e-16
        ):
            dex_lower = -float(cfg.dexterity_gamma) * (
                arm_rho - float(cfg.dexterity_d_safe)
            )
        branch_jacobian = np.zeros((2, 8))
        branch_lower = np.full(2, -np.inf)
        q4 = float(q[4])
        if self._elbow_branch_sign is None and abs(q4) > np.deg2rad(25.0):
            self._elbow_branch_sign = 1.0 if q4 > 0.0 else -1.0
        if self._elbow_branch_sign is not None and abs(q4) < np.deg2rad(25.0):
            branch_jacobian[0, 4] = self._elbow_branch_sign
            branch_lower[0] = -float(cfg.dexterity_gamma) * (
                self._elbow_branch_sign * q4
            )
        # J4 and J6 use independent fixed rows so simultaneous risk cannot
        # switch the active branch barrier.  The rows share one recovery slack.
        if self._wrist_branch_sign is not None and wrist_metric < wrist_exit:
            wrist_signed = self._wrist_branch_sign * float(q[wrist_index])
            branch_jacobian[1, wrist_index] = self._wrist_branch_sign
            branch_lower[1] = -float(cfg.dexterity_gamma) * wrist_signed
        return (
            risk_gradient,
            self._arm_gradient.copy(),
            float(dex_lower),
            branch_jacobian,
            branch_lower,
            wrist_metric,
        )

    def _whole_body_preference(
        self,
        state: RobotState,
        risk_gradient: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(state.q_meas, dtype=float)
        healthy = (
            self._risk_level <= 1.0e-3
            and self._healthy_dwell_s >= float(self.config.risk_exit_dwell_s)
        )
        if self._arm_nominal is None and healthy:
            self._arm_nominal = q[1:].copy()
        preference = np.zeros(8)
        if self._arm_nominal is not None:
            preference[1:] = float(self.config.arm_nominal_k) * (
                self._arm_nominal - q[1:]
            ) * max(1.0 - self._risk_level, 0.0)
        risk_preference = self._risk_level * np.asarray(risk_gradient, dtype=float)
        preference += risk_preference
        cap = max(float(self.config.arm_nominal_qdot_max), 0.0)
        arm_norm = float(np.linalg.norm(preference[1:]))
        if cap > 0.0 and arm_norm > cap:
            scale = cap / arm_norm
            preference[1:] *= scale
            risk_preference[1:] *= scale
        preference[0] = 0.0
        risk_preference[0] = 0.0
        return preference, risk_preference

    def _rail_preferences(
        self,
        scan_jacobian: np.ndarray,
        path_velocity: np.ndarray,
        state: RobotState,
    ) -> tuple[float, float]:
        rail_column = np.asarray(scan_jacobian[:, 0], dtype=float)
        denominator = float(rail_column @ rail_column + 1.0e-8)
        raw = float(rail_column @ path_velocity / denominator)
        raw = float(
            np.clip(
                raw,
                -self.config.rail_macro_v_max_m_s,
                self.config.rail_macro_v_max_m_s,
            )
        )
        tau = max(float(self.config.rail_macro_tau_s), 0.0)
        filtered = raw if tau <= 0.0 else self._rail_macro + state.dt / (tau + state.dt) * (raw - self._rail_macro)
        desired_acceleration = (filtered - self._rail_macro) / state.dt
        acceleration_delta = float(self.config.rail_macro_jerk_max_m_s3) * state.dt
        acceleration = np.clip(
            desired_acceleration,
            self._rail_macro_acceleration - acceleration_delta,
            self._rail_macro_acceleration + acceleration_delta,
        )
        acceleration = float(
            np.clip(
                acceleration,
                -self.config.rail_macro_a_max_m_s2,
                self.config.rail_macro_a_max_m_s2,
            )
        )
        self._rail_macro += acceleration * state.dt
        self._rail_macro = float(
            np.clip(
                self._rail_macro,
                -self.config.rail_macro_v_max_m_s,
                self.config.rail_macro_v_max_m_s,
            )
        )
        self._rail_macro_acceleration = acceleration
        rail_center = 0.5 * (
            float(self.limits.q_lower[0]) + float(self.limits.q_upper[0])
        )
        center_velocity = float(self.config.rail_center_k) * (
            rail_center - float(state.q_meas[0])
        )
        center_velocity = float(
            np.clip(
                center_velocity,
                -self.config.rail_center_v_max_m_s,
                self.config.rail_center_v_max_m_s,
            )
        )
        center_gate = 0.0
        if self._healthy_dwell_s >= float(self.config.risk_exit_dwell_s):
            center_gate = float(
                np.clip(
                    (self._healthy_dwell_s - float(self.config.risk_exit_dwell_s))
                    / max(float(self.config.risk_release_s), 1.0e-6),
                    0.0,
                    1.0,
                )
            )
        center_gate *= max(1.0 - self._risk_level, 0.0)
        return self._rail_macro, center_gate * center_velocity

    def _filter_feedback_xy(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=float).reshape(2)
        tau = max(float(self.config.feedback_lpf_tau_s), 0.0)
        filtered = target if tau <= 0.0 else (
            self._feedback_xy + dt / (tau + dt) * (target - self._feedback_xy)
        )
        max_delta = max(float(self.config.feedback_accel_max_m_s2), 0.0) * dt
        delta = np.clip(filtered - self._feedback_xy, -max_delta, max_delta)
        self._feedback_xy += delta
        return self._feedback_xy.copy()

    def solve(
        self,
        state: RobotState,
        *,
        protected_twist_task: np.ndarray,
        path_twist_task: np.ndarray,
        feedback_twist_task: np.ndarray,
        rotation_base_task: np.ndarray,
        task_safety_rows: Sequence[TaskSpaceConstraintRow] = (),
        application_hard_rows: Sequence[HardConstraintRow] = (),
        resync_err: float | np.ndarray = 0.0,
        rail_locked: bool = False,
        rail_lock_vel_eps_m_s: float = 0.0,
        rail_vel_pin_m_s: float | None = None,
        jacobian_base: np.ndarray | None = None,
    ) -> GenericQpikRuntimeResult:
        if jacobian_base is None:
            J_base = np.asarray(self.kin.jacobian(state.q_meas), dtype=float)
        else:
            J_base = np.asarray(jacobian_base, dtype=float)
            if J_base.shape != (6, 8) or not np.all(np.isfinite(J_base)):
                raise ValueError("jacobian_base must be a finite (6, 8) snapshot")
        J_task = self._task_jacobian(J_base, rotation_base_task)
        protected_twist = np.asarray(protected_twist_task, dtype=float).reshape(6)
        path_twist = np.asarray(path_twist_task, dtype=float).reshape(6)
        feedback_twist = np.asarray(feedback_twist_task, dtype=float).reshape(6)
        if not all(
            np.all(np.isfinite(value))
            for value in (protected_twist, path_twist, feedback_twist)
        ):
            raise ValueError("all task twists must be finite six-vectors")
        protected_indices = np.array([2, 3, 4, 5])
        scan_indices = np.array([0, 1])
        feedback_xy_raw = feedback_twist[scan_indices].copy()
        feedback_twist = feedback_twist.copy()
        feedback_twist[scan_indices] = self._filter_feedback_xy(
            feedback_twist[scan_indices], state.dt
        )
        protected_J = J_task[protected_indices]
        scan_J = J_task[scan_indices]

        hard_rows = list(application_hard_rows)
        for row in task_safety_rows:
            coefficients = np.asarray(row.coefficients, dtype=float) @ J_task
            hard_rows.append(
                HardConstraintRow(
                    coefficients,
                    lower=row.lower,
                    upper=row.upper,
                    name=row.name,
                )
            )
        p0 = self.p0_builder.build(
            state,
            resync_err=resync_err,
            rail_locked=rail_locked,
            rail_lock_vel_eps_m_s=rail_lock_vel_eps_m_s,
            rail_vel_pin_m_s=rail_vel_pin_m_s,
            application_rows=hard_rows,
            measured_kinematics_ready=True,
        )
        health = self._update_health(state, J_base)
        (
            risk_gradient,
            dexterity_gradient,
            dexterity_lower,
            branch_jacobian,
            branch_lower,
            wrist_metric,
        ) = self._risk_preference(
            state,
            health,
            self.p0_builder.last_collision_warning_C,
            self.p0_builder.last_collision_warning_lower,
        )
        preference, risk_preference = self._whole_body_preference(
            state, risk_gradient
        )
        previous_norm = float(np.linalg.norm(self._risk_preference_prev[1:]))
        current_norm = float(np.linalg.norm(risk_preference[1:]))
        risk_direction_cosine = float("nan")
        if previous_norm > 1.0e-9 and current_norm > 1.0e-9:
            risk_direction_cosine = float(
                np.clip(
                    np.dot(
                        self._risk_preference_prev[1:], risk_preference[1:]
                    )
                    / (previous_norm * current_norm),
                    -1.0,
                    1.0,
                )
            )
        self._risk_preference_prev = risk_preference.copy()
        working_lower, working_upper = self._working_bounds(state.q_meas)
        rail_macro, rail_center = self._rail_preferences(
            scan_J, path_twist[scan_indices], state
        )
        command = CartesianQpCommand(
            protected_jacobian=protected_J,
            protected_velocity=protected_twist[protected_indices],
            scan_jacobian=scan_J,
            path_velocity=path_twist[scan_indices],
            feedback_velocity=feedback_twist[scan_indices],
            qdot_preference=preference,
            working_lower=working_lower,
            working_upper=working_upper,
            dexterity_gradient=dexterity_gradient,
            dexterity_lower=dexterity_lower,
            branch_jacobian=branch_jacobian,
            branch_lower=branch_lower,
            collision_warning_jacobian=(
                self.p0_builder.last_collision_warning_C.copy()
            ),
            collision_warning_lower=(
                self.p0_builder.last_collision_warning_lower.copy()
            ),
            rail_macro_velocity=rail_macro,
            rail_center_velocity=rail_center,
            arm_risk_preference_norm=float(np.linalg.norm(risk_preference[1:])),
        )
        solved = self.solver.solve(
            command,
            p0.C,
            p0.lower,
            p0.upper,
            dt=state.dt,
            hard_names=p0.names,
        )
        return GenericQpikRuntimeResult(
            qdot=solved.qdot.copy(),
            solver=solved,
            p0=p0,
            health=health,
            jacobian_base=J_base.copy(),
            command=command,
            protected_target=protected_twist[protected_indices].copy(),
            protected_jacobian=protected_J.copy(),
            scan_target=(
                (1.0 - solved.beta) * (scan_J @ solved.anchor)
                + solved.beta * command.feedback_velocity
                + solved.alpha * command.path_velocity
            ).copy(),
            scan_jacobian=scan_J.copy(),
            wrist_singularity=wrist_metric,
            rail_macro_preference=rail_macro,
            rail_center_preference=rail_center,
            arm_risk_preference_norm=float(np.linalg.norm(risk_preference[1:])),
            feedback_xy_raw=feedback_xy_raw,
            feedback_xy_filtered=feedback_twist[scan_indices].copy(),
            risk_preference=risk_preference,
            risk_direction_cosine=risk_direction_cosine,
        )

__all__ = [
    "GenericQpikRuntime",
    "GenericQpikRuntimeConfig",
    "GenericQpikRuntimeResult",
]

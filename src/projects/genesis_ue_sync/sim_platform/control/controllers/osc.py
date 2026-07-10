from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
    OperationalSpaceControllerBase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import clip_norm, pose_error_wxyz


def _as_gain_vector(value: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 6)
    if arr.size != 6:
        raise ValueError(f"Expected scalar or 6 gains, got shape {arr.shape}.")
    return arr


@dataclass
class OSCControllerConfig:
    dt: float = 0.01
    task_stiffness: np.ndarray = field(default_factory=lambda: np.array([220.0, 220.0, 260.0, 60.0, 60.0, 60.0], dtype=np.float32))
    task_damping: np.ndarray = field(default_factory=lambda: np.array([35.0, 35.0, 40.0, 12.0, 12.0, 12.0], dtype=np.float32))
    task_integral: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float32))
    nullspace_stiffness: float = 20.0
    nullspace_damping: float = 8.0
    task_force_limit: float = 150.0
    max_joint_torque: float = 20.0
    max_joint_torque_delta: float = 3.0
    integral_limit: float = 0.08
    lambda_damping: float = 1e-4
    project_nullspace: bool = False

    def __post_init__(self) -> None:
        self.task_stiffness = _as_gain_vector(self.task_stiffness)
        self.task_damping = _as_gain_vector(self.task_damping)
        self.task_integral = _as_gain_vector(self.task_integral)


class OSCController(OperationalSpaceControllerBase):
    """Operational-space impedance controller with nullspace posture torque."""

    def __init__(self, motion, config: OSCControllerConfig | None = None, *, link_name: str | None = None) -> None:
        super().__init__(motion, link_name=link_name)
        self.config = config or OSCControllerConfig()
        self._integral_error = np.zeros(6, dtype=np.float32)
        self._last_torque_command: np.ndarray | None = None

    def reset(self) -> None:
        self._integral_error = np.zeros(6, dtype=np.float32)
        self._last_torque_command = None

    def step(self, target: CartesianControlTarget) -> ControllerStepResult:
        observation = self.observe(wrench_source="external_injected", include_mass_matrix=True)
        if observation.mass_matrix is None:
            raise RuntimeError("OSCController requires a mass matrix observation.")

        pose_error = pose_error_wxyz(target.pose, observation.tcp_pose)
        twist_error = np.asarray(target.twist - observation.tcp_twist, dtype=np.float32)
        self._integral_error = np.clip(
            self._integral_error + pose_error * float(target.metadata.get("dt", self.config.dt)),
            -self.config.integral_limit,
            self.config.integral_limit,
        )

        stiffness_term = self.config.task_stiffness * pose_error
        damping_term = self.config.task_damping * twist_error
        integral_term = self.config.task_integral * self._integral_error
        desired_wrench = stiffness_term + damping_term + integral_term + target.wrench

        mass_matrix = np.asarray(observation.mass_matrix, dtype=np.float32)
        jacobian = np.asarray(observation.jacobian, dtype=np.float32)
        mass_matrix_inv = np.linalg.inv(mass_matrix + 1e-6 * np.eye(mass_matrix.shape[0], dtype=np.float32))
        lambda_inv = jacobian @ mass_matrix_inv @ jacobian.T + self.config.lambda_damping * np.eye(6, dtype=np.float32)
        lambda_matrix = np.linalg.inv(lambda_inv)
        dynamically_consistent_pinv = mass_matrix_inv @ jacobian.T @ lambda_matrix
        task_wrench = clip_norm(desired_wrench, self.config.task_force_limit)
        torque_command = jacobian.T @ task_wrench

        nullspace_target = target.nullspace_target
        if nullspace_target is not None:
            posture_error = np.asarray(nullspace_target - observation.joint_position, dtype=np.float32)
            nullspace_torque = self.config.nullspace_stiffness * posture_error - self.config.nullspace_damping * observation.joint_velocity
            if self.config.project_nullspace:
                nullspace_projector = np.eye(mass_matrix.shape[0], dtype=np.float32) - jacobian.T @ dynamically_consistent_pinv.T
                torque_command = torque_command + nullspace_projector @ nullspace_torque
            else:
                torque_command = torque_command + nullspace_torque

        torque_command = np.clip(
            torque_command,
            -self.config.max_joint_torque,
            self.config.max_joint_torque,
        )
        if self._last_torque_command is not None and self.config.max_joint_torque_delta > 0.0:
            lo = self._last_torque_command - float(self.config.max_joint_torque_delta)
            hi = self._last_torque_command + float(self.config.max_joint_torque_delta)
            torque_command = np.clip(torque_command, lo, hi)
        self._last_torque_command = np.asarray(torque_command, dtype=np.float32).copy()

        self.motion.control_joint_forces(torque_command)
        return ControllerStepResult(
            control_mode="joint_force",
            command=np.asarray(torque_command, dtype=np.float32),
            observation=observation,
            target=target,
            pose_error=pose_error,
            metadata={
                "desired_wrench": desired_wrench.tolist(),
                "task_wrench": np.asarray(task_wrench, dtype=np.float32).tolist(),
            },
        )

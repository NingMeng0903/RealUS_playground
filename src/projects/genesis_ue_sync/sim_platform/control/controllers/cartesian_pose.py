from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
    OperationalSpaceControllerBase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import clip_norm, damped_pseudoinverse, pose_error_wxyz


def _as_numpy_f32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32).reshape(-1)


@dataclass
class CartesianPoseControllerConfig:
    dt: float = 0.01
    linear_gain: float = 4.0
    angular_gain: float = 3.0
    linear_damping: float = 0.0
    angular_damping: float = 0.0
    damping: float = 0.05
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.5
    max_joint_speed: float = 1.5
    output_mode: str = "ik_joint_position"
    ik_target_tolerance: float = 1e-6
    nullspace_stiffness: float = 0.0
    nullspace_damping: float = 0.0


class CartesianPoseController(OperationalSpaceControllerBase):
    """Cartesian tracking without explicit torque output.

    ``output_mode="ik_joint_position"`` runs Genesis ``inverse_kinematics`` each step and sends
    joint-position commands (underlying PD tracks them). Other modes integrate Cartesian velocity
    through the Jacobian (resolved-rate).
    """

    def __init__(self, motion, config: CartesianPoseControllerConfig | None = None, *, link_name: str | None = None) -> None:
        super().__init__(motion, link_name=link_name)
        self.config = config or CartesianPoseControllerConfig()
        self._last_ik_target_pose: np.ndarray | None = None
        self._last_ik_command: np.ndarray | None = None

    def reset(self) -> None:
        self._last_ik_target_pose = None
        self._last_ik_command = None

    def step(self, target: CartesianControlTarget) -> ControllerStepResult:
        observation = self.observe(wrench_source="external_injected", include_mass_matrix=False)
        pose_error = pose_error_wxyz(target.pose, observation.tcp_pose)

        if self.config.output_mode == "ik_joint_position":
            embodiment = self.motion.runtime.embodiments[self.motion.entity_name]
            ik_link = self.motion.entity.get_link(self.link_name or embodiment.end_effector.tcp_frame)
            target_pose = np.asarray(target.pose, dtype=np.float32).reshape(7)
            can_reuse_command = (
                self._last_ik_target_pose is not None
                and self._last_ik_command is not None
                and np.allclose(
                    target_pose,
                    self._last_ik_target_pose,
                    atol=float(self.config.ik_target_tolerance),
                    rtol=0.0,
                )
            )
            if can_reuse_command:
                command = self._last_ik_command.copy()
                solver = "inverse_kinematics_cached"
            else:
                init_qpos = self._last_ik_command if self._last_ik_command is not None else observation.joint_position
                command = _as_numpy_f32(
                    self.motion.entity.inverse_kinematics(
                        link=ik_link,
                        pos=target_pose[:3],
                        quat=target_pose[3:],
                        init_qpos=init_qpos,
                        damping=self.config.damping,
                        respect_joint_limit=True,
                    ),
                )
                self._last_ik_target_pose = target_pose.copy()
                self._last_ik_command = command.copy()
                solver = "inverse_kinematics"
            self.motion.control_joint_positions(command)
            return ControllerStepResult(
                control_mode="joint_position",
                command=command,
                observation=observation,
                target=target,
                pose_error=pose_error,
                metadata={"solver": solver},
            )

        desired_twist = target.twist.copy()
        desired_twist[:3] += self.config.linear_gain * pose_error[:3]
        desired_twist[3:] += self.config.angular_gain * pose_error[3:]
        desired_twist[:3] -= self.config.linear_damping * observation.tcp_twist[:3]
        desired_twist[3:] -= self.config.angular_damping * observation.tcp_twist[3:]
        desired_twist[:3] = clip_norm(desired_twist[:3], self.config.max_linear_speed)
        desired_twist[3:] = clip_norm(desired_twist[3:], self.config.max_angular_speed)

        jacobian_pinv = damped_pseudoinverse(observation.jacobian, self.config.damping)
        joint_velocity = jacobian_pinv @ desired_twist
        if target.nullspace_target is not None and self.config.nullspace_stiffness > 0.0:
            nullspace_projector = np.eye(observation.jacobian.shape[1], dtype=np.float32) - jacobian_pinv @ observation.jacobian
            nullspace_velocity = (
                self.config.nullspace_stiffness * (target.nullspace_target - observation.joint_position)
                - self.config.nullspace_damping * observation.joint_velocity
            )
            joint_velocity = joint_velocity + nullspace_projector @ nullspace_velocity
        joint_velocity = clip_norm(joint_velocity, self.config.max_joint_speed)

        if self.config.output_mode == "joint_velocity":
            self.motion.control_joint_velocities(joint_velocity)
            command = joint_velocity
            control_mode = "joint_velocity"
        elif self.config.output_mode == "joint_position":
            command = observation.joint_position + joint_velocity * self.config.dt
            self.motion.control_joint_positions(command)
            control_mode = "joint_position"
        else:
            raise ValueError(f"Unsupported CartesianPoseController output_mode: {self.config.output_mode}")

        return ControllerStepResult(
            control_mode=control_mode,
            command=_as_numpy_f32(command),
            observation=observation,
            target=target,
            pose_error=pose_error,
            metadata={"desired_twist": desired_twist.tolist()},
        )

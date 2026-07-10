from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
    OperationalSpaceControllerBase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import clip_norm, damped_pseudoinverse, pose_error_wxyz


@dataclass
class CartesianVelocityControllerConfig:
    damping: float = 0.05
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.5
    max_joint_speed: float = 1.5
    nullspace_stiffness: float = 0.0
    nullspace_damping: float = 0.0


class CartesianVelocityController(OperationalSpaceControllerBase):
    """Direct Cartesian twist controller that solves joint velocities from a desired TCP twist."""

    def __init__(
        self,
        motion,
        config: CartesianVelocityControllerConfig | None = None,
        *,
        link_name: str | None = None,
    ) -> None:
        super().__init__(motion, link_name=link_name)
        self.config = config or CartesianVelocityControllerConfig()

    def step(self, target: CartesianControlTarget) -> ControllerStepResult:
        observation = self.observe(wrench_source="external_injected", include_mass_matrix=False)
        pose_error = pose_error_wxyz(target.pose, observation.tcp_pose)
        desired_twist = np.asarray(target.twist, dtype=np.float32).reshape(6)
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
        self.motion.control_joint_velocities(joint_velocity)
        return ControllerStepResult(
            control_mode="joint_velocity",
            command=np.asarray(joint_velocity, dtype=np.float32),
            observation=observation,
            target=target,
            pose_error=pose_error,
            metadata={"desired_twist": desired_twist.tolist()},
        )

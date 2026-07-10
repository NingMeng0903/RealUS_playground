from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.common import as_pose_array
from projects.genesis_ue_sync.sim_platform.control.motion import MotionInterface


@dataclass
class CartesianControlTarget:
    pose: np.ndarray
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    wrench: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    nullspace_target: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pose = as_pose_array(self.pose)
        self.twist = np.asarray(self.twist, dtype=np.float32).reshape(6)
        self.wrench = np.asarray(self.wrench, dtype=np.float32).reshape(6)
        if self.nullspace_target is not None:
            self.nullspace_target = np.asarray(self.nullspace_target, dtype=np.float32).reshape(-1)


@dataclass
class ControllerObservation:
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_effort: np.ndarray
    tcp_pose: np.ndarray
    tcp_twist: np.ndarray
    jacobian: np.ndarray
    wrench: np.ndarray
    mass_matrix: np.ndarray | None = None


@dataclass
class ControllerStepResult:
    control_mode: str
    command: np.ndarray
    observation: ControllerObservation
    target: CartesianControlTarget
    pose_error: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class OperationalSpaceControllerBase:
    def __init__(self, motion: MotionInterface, *, link_name: str | None = None) -> None:
        self.motion = motion
        self.link_name = link_name

    def observe(
        self,
        *,
        wrench_source: str = "external_injected",
        include_mass_matrix: bool = False,
    ) -> ControllerObservation:
        tcp_pose = self.motion.get_tcp_pose() if self.link_name is None else self.motion.get_link_pose(self.link_name)
        tcp_twist = self.motion.get_tcp_twist() if self.link_name is None else self.motion.get_link_twist(self.link_name)
        jacobian = self.motion.get_jacobian(link_name=self.link_name)
        return ControllerObservation(
            joint_position=self.motion.get_joint_positions(),
            joint_velocity=self.motion.get_joint_velocities(),
            joint_effort=self.motion.get_joint_efforts(),
            tcp_pose=np.asarray(tcp_pose, dtype=np.float32).reshape(7),
            tcp_twist=np.asarray(tcp_twist, dtype=np.float32).reshape(6),
            jacobian=np.asarray(jacobian, dtype=np.float32),
            wrench=self.motion.get_wrench(source=wrench_source, link_name=self.link_name),
            mass_matrix=self.motion.get_mass_matrix() if include_mass_matrix else None,
        )

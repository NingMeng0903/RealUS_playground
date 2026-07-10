from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.common import quat_wxyz_to_rotation_matrix


@dataclass
class MotionCommand:
    control_mode: str
    values: list[float]
    dofs_idx: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MotionInterface:
    """Generic articulated robot motion API for simulation, replay, and deployment."""

    def __init__(self, runtime, entity_name: str) -> None:
        self.runtime = runtime
        self.entity_name = entity_name

    @property
    def entity(self):
        if self.entity_name not in self.runtime.entities:
            raise KeyError(f"Unknown runtime entity: {self.entity_name}")
        return self.runtime.entities[self.entity_name]

    def set_joint_positions(self, joint_positions: list[float] | np.ndarray) -> None:
        self.runtime.set_robot_joint_positions(self.entity_name, joint_positions)

    def control_joint_positions(self, joint_positions: list[float] | np.ndarray, dofs_idx: list[int] | None = None) -> None:
        values = np.asarray(joint_positions, dtype=np.float32)
        if dofs_idx is None:
            self.entity.control_dofs_position(values)
        else:
            self.entity.control_dofs_position(values, dofs_idx)

    def control_joint_velocities(self, joint_velocities: list[float] | np.ndarray, dofs_idx: list[int] | None = None) -> None:
        values = np.asarray(joint_velocities, dtype=np.float32)
        if dofs_idx is None:
            self.entity.control_dofs_velocity(values)
        else:
            self.entity.control_dofs_velocity(values, dofs_idx)

    def control_joint_forces(self, joint_forces: list[float] | np.ndarray, dofs_idx: list[int] | None = None) -> None:
        values = np.asarray(joint_forces, dtype=np.float32)
        if dofs_idx is None:
            self.entity.control_dofs_force(values)
        else:
            self.entity.control_dofs_force(values, dofs_idx)

    def get_joint_positions(self) -> np.ndarray:
        return np.asarray(self.runtime.get_robot_joint_positions(self.entity_name), dtype=np.float32).reshape(-1)

    def get_joint_velocities(self) -> np.ndarray:
        return np.asarray(self.runtime.get_robot_joint_velocities(self.entity_name), dtype=np.float32).reshape(-1)

    def get_joint_efforts(self) -> np.ndarray:
        return np.asarray(self.runtime.get_robot_joint_efforts(self.entity_name), dtype=np.float32).reshape(-1)

    def get_tcp_pose(self) -> np.ndarray:
        return np.asarray(self.runtime.get_tcp_pose(self.entity_name), dtype=np.float32).reshape(-1)

    def get_tcp_twist(self) -> np.ndarray:
        return np.asarray(self.runtime.get_tcp_twist(self.entity_name), dtype=np.float32).reshape(-1)

    def get_link_pose(self, link_name: str) -> np.ndarray:
        return np.asarray(self.runtime.get_link_pose(self.entity_name, link_name), dtype=np.float32).reshape(-1)

    def get_link_twist(self, link_name: str) -> np.ndarray:
        return np.asarray(self.runtime.get_link_twist(self.entity_name, link_name), dtype=np.float32).reshape(-1)

    def get_jacobian(self, link_name: str | None = None, local_point: list[float] | np.ndarray | None = None) -> np.ndarray:
        return np.asarray(
            self.runtime.get_robot_jacobian(self.entity_name, link_name=link_name, local_point=local_point),
            dtype=np.float32,
        )

    def get_mass_matrix(self) -> np.ndarray:
        return np.asarray(self.runtime.get_robot_mass_matrix(self.entity_name), dtype=np.float32)

    def resolve_tcp_link(self, link_name: str | None) -> str:
        emb = self.runtime.embodiments.get(self.entity_name)
        name = getattr(getattr(emb, "end_effector", None), "tcp_frame", None) if emb is not None else None
        return str(link_name or name or "TCP")

    def get_link_point_pose_wxyz(self, link_name: str | None, local_point: list[float] | np.ndarray | None) -> np.ndarray:
        if local_point is None:
            if link_name is None:
                return np.asarray(self.get_tcp_pose(), dtype=np.float32).reshape(7)
            return np.asarray(self.get_link_pose(str(link_name)), dtype=np.float32).reshape(7)

        link_eff = self.resolve_tcp_link(link_name)
        pose_link = np.asarray(self.get_link_pose(str(link_eff)), dtype=np.float32).reshape(7)
        lp = np.asarray(local_point, dtype=np.float32).reshape(3)
        R = quat_wxyz_to_rotation_matrix(pose_link[3:7]).astype(np.float32)
        out = pose_link.copy()
        out[:3] = pose_link[:3].astype(np.float32) + R @ lp.astype(np.float32)
        return out

    def get_link_point_twist(self, link_name: str | None, local_point: list[float] | np.ndarray | None) -> np.ndarray:
        link_eff = self.resolve_tcp_link(link_name)
        twist = np.asarray(self.get_link_twist(link_eff), dtype=np.float32).reshape(6)
        if local_point is None:
            return twist
        lp = np.asarray(local_point, dtype=np.float32).reshape(3)
        pose_link = np.asarray(self.get_link_pose(link_eff), dtype=np.float32).reshape(7)
        R = quat_wxyz_to_rotation_matrix(pose_link[3:7]).astype(np.float32)
        r_world = R @ lp
        v = twist[:3]
        omega = twist[3:]
        v_point = v + np.cross(omega.astype(np.float64), r_world.astype(np.float64)).astype(np.float32)
        return np.concatenate([v_point, omega], dtype=np.float32)

    def try_get_joint_coriolis_gravity_torques(self) -> tuple[np.ndarray, np.ndarray]:
        entity = self.entity
        n = int(self.get_joint_positions().size)
        c = np.zeros(n, dtype=np.float32)
        g = np.zeros(n, dtype=np.float32)

        coriolis_candidates = ("get_dof_coriolis_force", "get_coriolis", "compute_coriolis")
        for name in coriolis_candidates:
            fn = getattr(entity, name, None)
            if not callable(fn):
                continue
            try:
                vec = np.asarray(fn(), dtype=np.float32).reshape(-1)
                if vec.size == n:
                    c = vec
                    break
            except Exception:
                continue

        gravity_candidates = ("get_dof_gravity_force", "get_gravity", "compute_gravity")
        for name in gravity_candidates:
            fn = getattr(entity, name, None)
            if not callable(fn):
                continue
            try:
                vec = np.asarray(fn(), dtype=np.float32).reshape(-1)
                if vec.size == n:
                    g = vec
                    break
            except Exception:
                continue

        return c, g

    def update_external_wrench(self, wrench: list[float] | np.ndarray) -> None:
        self.runtime.set_external_wrench(self.entity_name, wrench)

    def set_gravity_compensation(self, value: float) -> None:
        self.runtime.set_robot_gravity_compensation(self.entity_name, value)

    def get_external_wrench(self) -> np.ndarray:
        return np.asarray(self.runtime.get_external_wrench(self.entity_name), dtype=np.float32)

    def get_wrench(self, source: str = "external_injected", link_name: str | None = None) -> np.ndarray:
        return np.asarray(self.runtime.get_wrench(self.entity_name, source=source, link_name=link_name), dtype=np.float32)

    def apply(self, command: MotionCommand) -> None:
        if command.control_mode == "joint_position":
            self.control_joint_positions(command.values, dofs_idx=command.dofs_idx)
            return
        if command.control_mode == "joint_velocity":
            self.control_joint_velocities(command.values, dofs_idx=command.dofs_idx)
            return
        if command.control_mode == "joint_force":
            self.control_joint_forces(command.values, dofs_idx=command.dofs_idx)
            return
        raise ValueError(f"Unsupported control mode: {command.control_mode}")

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
    OperationalSpaceControllerBase,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import apply_pose_delta_wxyz


def _as_axis_vector(value: float | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 6)
    if arr.size != 6:
        raise ValueError(f"Expected scalar or 6-vector, got shape {arr.shape}.")
    return arr


@dataclass
class AdmittanceControllerConfig:
    dt: float = 0.01
    virtual_mass: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 3.0, 0.4, 0.4, 0.4], dtype=np.float32))
    virtual_damping: np.ndarray = field(default_factory=lambda: np.array([45.0, 45.0, 45.0, 10.0, 10.0, 10.0], dtype=np.float32))
    virtual_stiffness: np.ndarray = field(default_factory=lambda: np.array([80.0, 80.0, 80.0, 20.0, 20.0, 20.0], dtype=np.float32))
    compliant_axes: np.ndarray = field(default_factory=lambda: np.array([1, 1, 1, 0, 0, 0], dtype=np.float32))
    wrench_source: str = "external_injected"
    max_offset: np.ndarray = field(default_factory=lambda: np.array([0.05, 0.05, 0.05, 0.25, 0.25, 0.25], dtype=np.float32))

    def __post_init__(self) -> None:
        self.virtual_mass = np.maximum(_as_axis_vector(self.virtual_mass), 1e-4)
        self.virtual_damping = _as_axis_vector(self.virtual_damping)
        self.virtual_stiffness = _as_axis_vector(self.virtual_stiffness)
        self.compliant_axes = _as_axis_vector(self.compliant_axes)
        self.max_offset = np.abs(_as_axis_vector(self.max_offset))


class AdmittanceController(OperationalSpaceControllerBase):
    """Outer-loop admittance controller that warps the Cartesian target pose."""

    def __init__(self, motion, config: AdmittanceControllerConfig | None = None, *, link_name: str | None = None) -> None:
        super().__init__(motion, link_name=link_name)
        self.config = config or AdmittanceControllerConfig()
        self._offset = np.zeros(6, dtype=np.float32)
        self._offset_velocity = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        self._offset = np.zeros(6, dtype=np.float32)
        self._offset_velocity = np.zeros(6, dtype=np.float32)

    def update_target(self, target: CartesianControlTarget) -> tuple[CartesianControlTarget, ControllerStepResult]:
        observation = self.observe(wrench_source=self.config.wrench_source, include_mass_matrix=False)
        wrench_error = np.asarray(observation.wrench - target.wrench, dtype=np.float32) * self.config.compliant_axes
        acceleration = (
            wrench_error
            - self.config.virtual_damping * self._offset_velocity
            - self.config.virtual_stiffness * self._offset
        ) / self.config.virtual_mass
        self._offset_velocity += acceleration * self.config.dt
        self._offset += self._offset_velocity * self.config.dt
        self._offset = np.clip(self._offset, -self.config.max_offset, self.config.max_offset)

        adjusted_target = CartesianControlTarget(
            pose=apply_pose_delta_wxyz(target.pose, self._offset),
            twist=target.twist + self._offset_velocity,
            wrench=target.wrench,
            nullspace_target=target.nullspace_target,
            metadata={
                **target.metadata,
                "wrench_source": self.config.wrench_source,
                "admittance_offset": self._offset.tolist(),
            },
        )
        result = ControllerStepResult(
            control_mode="admittance_target",
            command=self._offset.copy(),
            observation=observation,
            target=adjusted_target,
            pose_error=self._offset.copy(),
            metadata={
                "measured_wrench": observation.wrench.tolist(),
                "wrench_error": wrench_error.tolist(),
                "offset_velocity": self._offset_velocity.tolist(),
            },
        )
        return adjusted_target, result

    def step(self, target: CartesianControlTarget, inner_controller) -> tuple[CartesianControlTarget, ControllerStepResult, ControllerStepResult]:
        adjusted_target, outer_result = self.update_target(target)
        inner_result = inner_controller.step(adjusted_target)
        return adjusted_target, outer_result, inner_result

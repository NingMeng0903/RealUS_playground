from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_force_sensor import (
    read_scene_virtual_force_sensor_wrench,
)


def _zeros6() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@dataclass
class rm_force_data_t:
    force_data: list[float] = field(default_factory=_zeros6)
    zero_force_data: list[float] = field(default_factory=_zeros6)
    work_zero_force_data: list[float] = field(default_factory=_zeros6)
    tool_zero_force_data: list[float] = field(default_factory=_zeros6)

    def to_dict(self, recurse: bool = True) -> dict[str, list[float]]:
        del recurse
        return {
            "force_data": list(self.force_data),
            "zero_force_data": list(self.zero_force_data),
            "work_zero_force_data": list(self.work_zero_force_data),
            "tool_zero_force_data": list(self.tool_zero_force_data),
        }


class VirtualRmForceSensor:
    def __init__(self, runtime: Any, robot_name: str, spec: Any | None = None) -> None:
        self.runtime = runtime
        self.robot_name = str(robot_name)
        self.spec = spec
        self._bias_world = np.zeros(6, dtype=np.float32)
        self._bias_sensor = np.zeros(6, dtype=np.float32)
        self.calibrated = False

    def read_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        if self.spec is None:
            z = np.zeros(6, dtype=np.float32)
            return z, z
        return read_scene_virtual_force_sensor_wrench(self.runtime, self.robot_name, self.spec)

    def read_data(self) -> rm_force_data_t:
        world, sensor = self.read_wrench()
        world_zero = np.asarray(world, dtype=np.float32).reshape(6) - self._bias_world
        sensor_zero = np.asarray(sensor, dtype=np.float32).reshape(6) - self._bias_sensor
        return rm_force_data_t(
            force_data=np.asarray(sensor, dtype=np.float32).reshape(6).tolist(),
            zero_force_data=sensor_zero.tolist(),
            work_zero_force_data=world_zero.tolist(),
            tool_zero_force_data=sensor_zero.tolist(),
        )

    def clear(self) -> None:
        world, sensor = self.read_wrench()
        self._bias_world = np.asarray(world, dtype=np.float32).reshape(6)
        self._bias_sensor = np.asarray(sensor, dtype=np.float32).reshape(6)
        self.calibrated = True

    def set_force_sensor(self, block: bool) -> int:
        del block
        self.clear()
        return 0

    def manual_set_force(self, point_num: int, joint: list[float], block: bool) -> int:
        del joint, block
        if int(point_num) not in {1, 2, 3, 4}:
            return 1
        if int(point_num) == 4:
            self.clear()
        return 0

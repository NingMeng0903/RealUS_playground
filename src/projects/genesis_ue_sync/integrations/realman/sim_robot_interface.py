from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from projects.genesis_ue_sync.integrations.realman.virtual_force_sensor import (
    VirtualRmForceSensor,
    rm_force_data_t,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.base import CartesianControlTarget
from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import CartesianPoseController
from projects.genesis_ue_sync.sim_platform.control.controllers.common import (
    as_pose_array,
    normalize_quaternion_wxyz,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.force_position_hybrid import (
    ForcePositionHybridController,
    ForcePositionHybridControllerConfig,
    ForcePositionHybridParams,
)
from projects.genesis_ue_sync.sim_platform.control.teleop import cartesian_follow_controller_config


def _as_float_list(values: list[float] | tuple[float, ...] | np.ndarray, length: int, name: str) -> list[float]:
    out = np.asarray(values, dtype=np.float32).reshape(-1)
    if out.size != int(length):
        raise ValueError(f"{name} must contain {length} values, got {out.size}.")
    return [float(v) for v in out.tolist()]


def _quat_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cy, sy = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)
    return normalize_quaternion_wxyz(
        np.asarray(
            [
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
            ],
            dtype=np.float32,
        )
    )


@dataclass
class rm_position_t:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [float(self.x), float(self.y), float(self.z)]


@dataclass
class rm_quat_t:
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [float(self.w), float(self.x), float(self.y), float(self.z)]


@dataclass
class rm_euler_t:
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0

    def to_list(self) -> list[float]:
        return [float(self.rx), float(self.ry), float(self.rz)]


@dataclass
class rm_pose_t:
    position: rm_position_t = field(default_factory=rm_position_t)
    quaternion: rm_quat_t = field(default_factory=rm_quat_t)
    euler: rm_euler_t = field(default_factory=rm_euler_t)

    @classmethod
    def from_list(cls, pose: list[float] | tuple[float, ...] | np.ndarray) -> "rm_pose_t":
        values = np.asarray(pose, dtype=np.float32).reshape(-1)
        if values.size == 6:
            quat = _quat_from_euler_xyz(float(values[3]), float(values[4]), float(values[5]))
            return cls(
                position=rm_position_t(*[float(v) for v in values[:3]]),
                quaternion=rm_quat_t(*[float(v) for v in quat]),
                euler=rm_euler_t(*[float(v) for v in values[3:6]]),
            )
        if values.size == 7:
            quat = normalize_quaternion_wxyz(values[3:7])
            return cls(
                position=rm_position_t(*[float(v) for v in values[:3]]),
                quaternion=rm_quat_t(*[float(v) for v in quat]),
                euler=rm_euler_t(),
            )
        raise ValueError(f"pose must contain 6 or 7 values, got {values.size}.")

    def to_pose_wxyz(self) -> np.ndarray:
        return as_pose_array(self.position.to_list() + self.quaternion.to_list())


@dataclass
class rm_force_position_t:
    sensor: int = 1
    mode: int = 1
    control_mode: list[int] = field(default_factory=lambda: [3, 3, 4, 0, 0, 0])
    desired_force: list[float] = field(default_factory=lambda: [0.0] * 6)
    limit_vel: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3])

    def to_hybrid_params(self) -> ForcePositionHybridParams:
        return ForcePositionHybridParams(
            sensor=int(self.sensor),
            mode=int(self.mode),
            control_mode=[int(v) for v in self.control_mode],
            desired_force=_as_float_list(self.desired_force, 6, "desired_force"),
            limit_vel=_as_float_list(self.limit_vel, 6, "limit_vel"),
        )


@dataclass
class rm_force_position_move_t:
    flag: int = 1
    pose: rm_pose_t | list[float] | None = None
    joint: list[float] | None = None
    sensor: int = 1
    mode: int = 1
    follow: bool = True
    control_mode: list[int] = field(default_factory=lambda: [3, 3, 4, 0, 0, 0])
    desired_force: list[float] = field(default_factory=lambda: [0.0] * 6)
    limit_vel: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3])
    trajectory_mode: int = 0
    radio: int = 0

    def to_hybrid_params(self) -> ForcePositionHybridParams:
        return ForcePositionHybridParams(
            sensor=int(self.sensor),
            mode=int(self.mode),
            control_mode=[int(v) for v in self.control_mode],
            desired_force=_as_float_list(self.desired_force, 6, "desired_force"),
            limit_vel=_as_float_list(self.limit_vel, 6, "limit_vel"),
        )

    def pose_wxyz(self) -> np.ndarray:
        if self.pose is None:
            raise ValueError("pose is required when flag == 1.")
        if isinstance(self.pose, rm_pose_t):
            return self.pose.to_pose_wxyz()
        return rm_pose_t.from_list(self.pose).to_pose_wxyz()


class RealManForceScanSession:
    def __init__(self, robot: "RealManSimRobotInterface") -> None:
        self.robot = robot

    def step_pose(
        self,
        pose: list[float] | tuple[float, ...] | np.ndarray,
        *,
        desired_force: list[float] | None = None,
        control_mode: list[int] | None = None,
        limit_vel: list[float] | None = None,
    ) -> int:
        base = self.robot.default_force_position_move_param(pose=pose)
        if desired_force is not None:
            base.desired_force = list(desired_force)
        if control_mode is not None:
            base.control_mode = [int(v) for v in control_mode]
        if limit_vel is not None:
            base.limit_vel = list(limit_vel)
        return self.robot.rm_force_position_move(base)

    def step_joint(
        self,
        joint: list[float] | tuple[float, ...] | np.ndarray,
        *,
        desired_force: list[float] | None = None,
    ) -> int:
        del joint, desired_force
        return 1

    def stop(self) -> int:
        return self.robot.rm_stop_force_position_move()


class RealManSimRobotInterface:
    """Simulation-only subset of the RealMan Python API2 robot interface.

    Force-position streaming supports ``rm_force_position_move`` with ``flag=1`` (pose target) only.
    ``flag=0`` (joint-angle target) and ``rm_force_position_move_joint`` are reserved for API
    compatibility and return error code ``1``.
    """

    def __init__(
        self,
        runtime: Any,
        robot_name: str,
        *,
        robot_spec: Any | None = None,
        dt: float | None = None,
    ) -> None:
        self.runtime = runtime
        self.robot_name = str(robot_name)
        self.robot_spec = robot_spec
        self.motion = runtime.get_motion_interface(self.robot_name)
        self.dt = float(dt if dt is not None else getattr(runtime.config, "dt", 0.01))
        self.max_line_speed = 0.25
        self.max_line_acc = 1.0
        self.max_angular_speed = 0.6
        self.max_angular_acc = 1.0
        self._active_mode = "idle"
        self._paused_mode: str | None = None
        self._current_trajectory = "idle"
        self.force_sensor = VirtualRmForceSensor(runtime, self.robot_name, getattr(robot_spec, "force_sensor", None))
        inner_cfg = cartesian_follow_controller_config(self.dt)
        self.cartesian = CartesianPoseController(self.motion, inner_cfg)
        self.hybrid = ForcePositionHybridController(
            self.motion,
            ForcePositionHybridControllerConfig(dt=self.dt, inner=inner_cfg),
            wrench_reader=self.force_sensor.read_wrench,
        )
        self._planned_force_params: ForcePositionHybridParams | None = None
        self._scan_defaults: dict[str, Any] = {
            "sensor": 1,
            "mode": 1,
            "follow": True,
            "control_mode": [3, 3, 4, 0, 0, 0],
            "desired_force": [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            "limit_vel": [0.05, 0.05, 0.05, 0.3, 0.3, 0.3],
            "trajectory_mode": 0,
            "radio": 0,
        }
        self._scan_overrides: dict[str, Any] = {}

    def _can_enter(self, mode: str) -> bool:
        return self._active_mode in {"idle", mode}

    def _pose_from_any(self, pose: rm_pose_t | list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        if isinstance(pose, rm_pose_t):
            return pose.to_pose_wxyz()
        return rm_pose_t.from_list(pose).to_pose_wxyz()

    def _step_runtime(self, steps: int = 1) -> None:
        for _ in range(max(int(steps), 1)):
            self.runtime.step()

    def _run_cartesian_pose(self, target_pose: np.ndarray, *, v: int, block: int, force_params: ForcePositionHybridParams | None = None) -> int:
        target = as_pose_array(target_pose)
        current = np.asarray(self.motion.get_tcp_pose(), dtype=np.float32).reshape(7)
        speed = max(float(self.max_line_speed) * max(float(v), 1.0) / 100.0, 1e-4)
        distance = float(np.linalg.norm(target[:3] - current[:3]))
        steps = 1 if not bool(block) else max(1, int(math.ceil(distance / max(speed * self.dt, 1e-6))))
        steps = min(steps, 2000)
        for idx in range(1, steps + 1):
            alpha = float(idx) / float(steps)
            pose_i = current.copy()
            pose_i[:3] = (1.0 - alpha) * current[:3] + alpha * target[:3]
            pose_i[3:] = target[3:]
            if force_params is None:
                self.cartesian.step(CartesianControlTarget(pose=pose_i))
            else:
                self.hybrid.step_pose(pose_i, force_params)
            self._step_runtime(1)
        return 0

    def default_force_position_move_param(
        self,
        *,
        pose: list[float] | tuple[float, ...] | np.ndarray | None = None,
        joint: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> rm_force_position_move_t:
        defaults = {**self._scan_defaults, **self._scan_overrides}
        if pose is not None:
            return rm_force_position_move_t(flag=1, pose=rm_pose_t.from_list(pose), **defaults)
        if joint is not None:
            return rm_force_position_move_t(flag=0, joint=_as_float_list(joint, 7, "joint"), **defaults)
        return rm_force_position_move_t(**defaults)

    def move_joints(self, joint_radians: list[float] | tuple[float, ...] | np.ndarray) -> int:
        values = _as_float_list(joint_radians, 7, "joint_radians")
        if not self._can_enter("position"):
            return 1
        self._active_mode = "position"
        self._current_trajectory = "move_joints"
        self.motion.control_joint_positions(values)
        self._step_runtime(1)
        self._active_mode = "idle"
        self._current_trajectory = "idle"
        return 0

    def stop_all(self) -> int:
        self.rm_set_arm_slow_stop()
        self.rm_stop_force_position()
        self.rm_stop_force_position_move()
        self.rm_set_arm_delete_trajectory()
        return 0

    def start_force_scan(self, **overrides: Any) -> RealManForceScanSession:
        tag = self.rm_start_force_position_move()
        if tag != 0:
            raise RuntimeError(f"rm_start_force_position_move failed with code {tag}")
        self._scan_overrides = dict(overrides)
        return RealManForceScanSession(self)

    def stop_force_scan(self) -> int:
        return self.rm_stop_force_position_move()

    def rm_movej(self, joint: list[float], v: int, r: int, connect: int, block: int) -> int:
        del v, r, connect
        if not self._can_enter("position"):
            return 1
        values_deg = np.asarray(_as_float_list(joint, 7, "joint"), dtype=np.float32)
        values_rad = np.deg2rad(values_deg).astype(np.float32)
        self._active_mode = "position"
        self._current_trajectory = "rm_movej"
        self.motion.control_joint_positions(values_rad)
        self._step_runtime(1)
        if bool(block):
            self._active_mode = "idle"
            self._current_trajectory = "idle"
        return 0

    def rm_movel(self, pose: list[float], v: int, r: int, connect: int, block: int) -> int:
        del r, connect
        if self._active_mode == "planned_force":
            if self._planned_force_params is None:
                return 1
            return self._run_cartesian_pose(self._pose_from_any(pose), v=int(v), block=int(block), force_params=self._planned_force_params)
        if not self._can_enter("position"):
            return 1
        self._active_mode = "position"
        self._current_trajectory = "rm_movel"
        tag = self._run_cartesian_pose(self._pose_from_any(pose), v=int(v), block=int(block), force_params=None)
        if bool(block):
            self._active_mode = "idle"
            self._current_trajectory = "idle"
        return tag

    def rm_movej_p(self, pose: list[float], v: int, r: int, connect: int, block: int) -> int:
        return self.rm_movel(pose, v, r, connect, block)

    def rm_set_arm_slow_stop(self) -> int:
        if self._active_mode in {"position", "paused"}:
            self._active_mode = "idle"
            self._paused_mode = None
            self._current_trajectory = "slow_stop"
            return 0
        return 0 if self._active_mode == "idle" else 1

    def rm_set_arm_stop(self) -> int:
        if self._active_mode in {"planned_force", "streaming_force"}:
            return 1
        self._active_mode = "idle"
        self._paused_mode = None
        self._current_trajectory = "stop"
        return 0

    def rm_set_arm_pause(self) -> int:
        if self._active_mode not in {"position"}:
            return 1
        self._paused_mode = self._active_mode
        self._active_mode = "paused"
        return 0

    def rm_set_arm_continue(self) -> int:
        if self._active_mode != "paused" or self._paused_mode is None:
            return 1
        self._active_mode = self._paused_mode
        self._paused_mode = None
        return 0

    def rm_set_arm_delete_trajectory(self) -> int:
        self._current_trajectory = "idle"
        return 0

    def rm_get_arm_current_trajectory(self) -> dict[str, Any]:
        return {
            "active_mode": self._active_mode,
            "trajectory": self._current_trajectory,
        }

    def rm_set_arm_max_line_speed(self, speed: float) -> int:
        self.max_line_speed = max(float(speed), 1e-6)
        return 0

    def rm_set_arm_max_line_acc(self, acc: float) -> int:
        self.max_line_acc = max(float(acc), 1e-6)
        return 0

    def rm_set_arm_max_angular_speed(self, speed: float) -> int:
        self.max_angular_speed = max(float(speed), 1e-6)
        return 0

    def rm_set_arm_max_angular_acc(self, acc: float) -> int:
        self.max_angular_acc = max(float(acc), 1e-6)
        return 0

    def rm_set_force_position(self, sensor: int, mode: int, direction: int, force: float) -> int:
        control_mode = [3, 3, 3, 0, 0, 0]
        direction_i = int(direction)
        if not 0 <= direction_i < 6:
            return 1
        control_mode[direction_i] = 4
        desired_force = [0.0] * 6
        desired_force[direction_i] = float(force)
        return self.rm_set_force_position_new(
            rm_force_position_t(
                sensor=int(sensor),
                mode=int(mode),
                control_mode=control_mode,
                desired_force=desired_force,
                limit_vel=[0.05, 0.05, 0.05, 0.3, 0.3, 0.3],
            )
        )

    def rm_set_force_position_new(self, param: rm_force_position_t) -> int:
        if self._active_mode not in {"idle", "planned_force"}:
            return 1
        self._planned_force_params = param.to_hybrid_params()
        self._active_mode = "planned_force"
        self.hybrid.reset()
        return 0

    def rm_stop_force_position(self) -> int:
        if self._active_mode == "planned_force":
            self._active_mode = "idle"
        self._planned_force_params = None
        self.hybrid.reset()
        return 0

    def step_planned_force_pose(self, pose: list[float] | tuple[float, ...] | np.ndarray) -> int:
        if self._active_mode != "planned_force" or self._planned_force_params is None:
            return 1
        self.hybrid.step_pose(self._pose_from_any(pose), self._planned_force_params)
        self._step_runtime(1)
        return 0

    def rm_start_force_position_move(self) -> int:
        if self._active_mode != "idle":
            return 1
        self._active_mode = "streaming_force"
        self.hybrid.reset()
        return 0

    def rm_stop_force_position_move(self) -> int:
        if self._active_mode == "streaming_force":
            self._active_mode = "idle"
        self._scan_overrides = {}
        self.hybrid.reset()
        return 0

    def rm_force_position_move_joint(
        self,
        joint: list[float],
        sensor: int,
        mode: int,
        dir: int,
        force: float,
        follow: bool,
    ) -> int:
        del joint, sensor, mode, dir, force, follow
        return 1

    def rm_force_position_move_pose(
        self,
        pose: list[float],
        sensor: int,
        mode: int,
        dir: int,
        force: float,
        follow: bool,
    ) -> int:
        control_mode = [3, 3, 3, 0, 0, 0]
        if 0 <= int(dir) < 6:
            control_mode[int(dir)] = 4
        desired = [0.0] * 6
        if 0 <= int(dir) < 6:
            desired[int(dir)] = float(force)
        return self.rm_force_position_move(
            rm_force_position_move_t(
                flag=1,
                pose=rm_pose_t.from_list(pose),
                sensor=int(sensor),
                mode=int(mode),
                follow=bool(follow),
                control_mode=control_mode,
                desired_force=desired,
            )
        )

    def rm_force_position_move(self, param: rm_force_position_move_t) -> int:
        if self._active_mode != "streaming_force":
            return 1
        if int(param.flag) != 1:
            return 1
        if param.pose is None:
            return 1
        self.hybrid.step_pose(param.pose_wxyz(), param.to_hybrid_params())
        self._step_runtime(1)
        return 0

    def rm_get_force_data(self) -> tuple[int, dict[str, Any]]:
        data = self.force_sensor.read_data()
        return 0, data.to_dict()

    def rm_clear_force_data(self) -> int:
        self.force_sensor.clear()
        return 0

    def rm_set_force_sensor(self, block: bool) -> int:
        return self.force_sensor.set_force_sensor(bool(block))

    def rm_manual_set_force(self, point_num: int, joint: list[float], block: bool) -> int:
        return self.force_sensor.manual_set_force(int(point_num), joint, bool(block))


def build_realman_sim_robot(runtime: Any, robot_name: str, robot_spec: Any | None = None) -> RealManSimRobotInterface:
    return RealManSimRobotInterface(runtime, robot_name, robot_spec=robot_spec)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.base import (
    CartesianControlTarget,
    ControllerStepResult,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseController,
    CartesianPoseControllerConfig,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.common import (
    as_pose_array,
    apply_pose_delta_wxyz,
    quat_wxyz_to_rotation_matrix,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.realman_control_modes import (
    RM_CTRL_ADAPTIVE,
    RM_CTRL_FIXED,
    RM_CTRL_FLOAT,
    RM_CTRL_FLOAT_MOTION,
    RM_CTRL_FORCE_MOTION,
    RM_CTRL_FORCE_TRACK,
    RM_CTRL_MOTION,
    RM_CTRL_SPRING,
    RM_CTRL_SPRING_MOTION,
)


@dataclass
class ForcePositionHybridParams:
    sensor: int = 1
    mode: int = 1
    control_mode: list[int] = field(default_factory=lambda: [3, 3, 4, 0, 0, 0])
    desired_force: list[float] = field(default_factory=lambda: [0.0] * 6)
    limit_vel: list[float] = field(default_factory=lambda: [0.05, 0.05, 0.05, 0.3, 0.3, 0.3])

    def __post_init__(self) -> None:
        if len(self.control_mode) != 6:
            raise ValueError("control_mode must contain 6 values.")
        if len(self.desired_force) != 6:
            raise ValueError("desired_force must contain 6 values.")
        if len(self.limit_vel) != 6:
            raise ValueError("limit_vel must contain 6 values.")
        self.control_mode = [int(v) for v in self.control_mode]
        self.desired_force = [float(v) for v in self.desired_force]
        self.limit_vel = [float(v) for v in self.limit_vel]


@dataclass
class MBKAxisGains:
    """Mass-spring-damper outer-loop gains (RealMan documents M/B/K tuning; sim defaults are approximate)."""

    mass: float = 2.0
    damping: float = 35.0
    stiffness: float = 0.0


@dataclass
class ForcePositionHybridControllerConfig:
    dt: float = 0.01
    force_kp: float = 0.003
    force_ki: float = 0.0005
    torque_kp: float = 0.01
    torque_ki: float = 0.001
    float_mbk: MBKAxisGains = field(default_factory=lambda: MBKAxisGains(mass=2.0, damping=40.0, stiffness=0.0))
    spring_mbk: MBKAxisGains = field(default_factory=lambda: MBKAxisGains(mass=2.0, damping=28.0, stiffness=90.0))
    adaptive_tilt_gain: float = 0.0025
    adaptive_tilt_limit_rad: float = 0.12
    inner: CartesianPoseControllerConfig = field(default_factory=CartesianPoseControllerConfig)


@dataclass
class _AxisOuterState:
    mbk_offset: float = 0.0
    mbk_velocity: float = 0.0
    force_offset: float = 0.0
    force_integral: float = 0.0


WrenchReader = Callable[[], tuple[np.ndarray, np.ndarray]]


class ForcePositionHybridController:
    """RealMan-style position-command force-position hybrid (simulation outer loop).

    Maps RM_API2 ``control_mode[6]`` to documented outer-loop semantics:

    - 0 fixed: lock axis at activation pose.
    - 1 float: admittance ``M*a + B*v = F`` (K=0).
    - 2 spring: impedance ``M*a + B*v + K*x = F - F_des``.
    - 3 motion: follow commanded pose on axis (inner position loop).
    - 4 force track: ``x = x_fixed + PI(F_des - F)`` (no pose track on axis).
    - 5 float+motion: ``x = x_des + float_admittance``.
    - 6 spring+motion: ``x = x_des + spring_impedance``.
    - 7 force+motion: ``x = x_des + PI(F_des - F)``.
    - 8 adaptive: tool-frame Fz uses mode 7 + small Rx/Ry tilt from Fx/Fy.

    Inner loop remains joint-position Cartesian tracking (RM75 sim path, not hardware torque).
    """

    def __init__(
        self,
        motion: Any,
        config: ForcePositionHybridControllerConfig | None = None,
        *,
        link_name: str | None = None,
        wrench_reader: WrenchReader | None = None,
    ) -> None:
        self.motion = motion
        self.config = config or ForcePositionHybridControllerConfig()
        self.inner = CartesianPoseController(motion, self.config.inner, link_name=link_name)
        self.link_name = link_name
        self.wrench_reader = wrench_reader
        self._fixed_pose: np.ndarray | None = None
        self._axis_state = [_AxisOuterState() for _ in range(6)]

    def reset(self) -> None:
        self.inner.reset()
        self._fixed_pose = None
        self._axis_state = [_AxisOuterState() for _ in range(6)]

    def current_pose(self) -> np.ndarray:
        if self.link_name is None:
            return np.asarray(self.motion.get_tcp_pose(), dtype=np.float32).reshape(7)
        return np.asarray(self.motion.get_link_pose(self.link_name), dtype=np.float32).reshape(7)

    def _read_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        if self.wrench_reader is not None:
            world, sensor = self.wrench_reader()
            return (
                np.asarray(world, dtype=np.float32).reshape(6),
                np.asarray(sensor, dtype=np.float32).reshape(6),
            )
        wrench = np.asarray(
            self.motion.get_wrench(source="sim_contact", link_name=self.link_name),
            dtype=np.float32,
        ).reshape(6)
        return wrench, wrench

    @staticmethod
    def _project_scalar(position: np.ndarray, axis_world: np.ndarray) -> float:
        return float(np.dot(position[:3], axis_world))

    @staticmethod
    def _set_axis_scalar(adjusted: np.ndarray, axis_world: np.ndarray, scalar: float) -> None:
        adjusted[:3] = adjusted[:3] - axis_world * float(np.dot(adjusted[:3], axis_world)) + axis_world * float(scalar)

    def _mbk_step(
        self,
        state: _AxisOuterState,
        *,
        wrench: float,
        desired_wrench: float,
        mbk: MBKAxisGains,
        dt: float,
        limit_vel: float,
    ) -> float:
        acc = (float(wrench) - float(desired_wrench) - mbk.damping * state.mbk_velocity - mbk.stiffness * state.mbk_offset) / max(
            float(mbk.mass), 1e-6
        )
        state.mbk_velocity += acc * dt
        state.mbk_velocity = float(np.clip(state.mbk_velocity, -limit_vel, limit_vel))
        state.mbk_offset += state.mbk_velocity * dt
        return state.mbk_offset

    def _force_step(
        self,
        state: _AxisOuterState,
        *,
        wrench: float,
        desired_wrench: float,
        kp: float,
        ki: float,
        dt: float,
        limit_vel: float,
    ) -> float:
        err = float(desired_wrench) - float(wrench)
        state.force_integral += err * dt
        vel = kp * err + ki * state.force_integral
        vel = float(np.clip(vel, -limit_vel, limit_vel))
        state.force_offset += vel * dt
        return state.force_offset

    def _axis_scalar_cmd(
        self,
        axis: int,
        mode: int,
        *,
        s_des: float,
        s_fixed: float,
        wrench: float,
        desired_wrench: float,
        limit_vel: float,
        dt: float,
        is_torque: bool,
    ) -> float:
        state = self._axis_state[axis]
        kp = self.config.torque_kp if is_torque else self.config.force_kp
        ki = self.config.torque_ki if is_torque else self.config.force_ki

        if mode == RM_CTRL_FIXED:
            return s_fixed
        if mode == RM_CTRL_MOTION:
            return s_des
        if mode == RM_CTRL_FLOAT:
            off = self._mbk_step(state, wrench=wrench, desired_wrench=0.0, mbk=self.config.float_mbk, dt=dt, limit_vel=limit_vel)
            return s_fixed + off
        if mode == RM_CTRL_SPRING:
            off = self._mbk_step(
                state,
                wrench=wrench,
                desired_wrench=desired_wrench,
                mbk=self.config.spring_mbk,
                dt=dt,
                limit_vel=limit_vel,
            )
            return s_fixed + off
        if mode == RM_CTRL_FLOAT_MOTION:
            off = self._mbk_step(state, wrench=wrench, desired_wrench=0.0, mbk=self.config.float_mbk, dt=dt, limit_vel=limit_vel)
            return s_des + off
        if mode == RM_CTRL_SPRING_MOTION:
            off = self._mbk_step(
                state,
                wrench=wrench,
                desired_wrench=desired_wrench,
                mbk=self.config.spring_mbk,
                dt=dt,
                limit_vel=limit_vel,
            )
            return s_des + off
        if mode == RM_CTRL_FORCE_TRACK:
            off = self._force_step(state, wrench=wrench, desired_wrench=desired_wrench, kp=kp, ki=ki, dt=dt, limit_vel=limit_vel)
            return s_fixed + off
        if mode in {RM_CTRL_FORCE_MOTION, RM_CTRL_ADAPTIVE}:
            off = self._force_step(state, wrench=wrench, desired_wrench=desired_wrench, kp=kp, ki=ki, dt=dt, limit_vel=limit_vel)
            return s_des + off
        return s_des

    def adjusted_pose(
        self,
        target_pose: list[float] | tuple[float, ...] | np.ndarray,
        params: ForcePositionHybridParams,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        target = as_pose_array(target_pose)
        current = self.current_pose()
        if self._fixed_pose is None:
            self._fixed_pose = current.copy()

        wrench_world, wrench_sensor = self._read_wrench()
        measured = wrench_sensor if int(params.mode) == 1 else wrench_world
        basis = quat_wxyz_to_rotation_matrix(target[3:7]) if int(params.mode) == 1 else np.eye(3, dtype=np.float32)

        adjusted = target.copy()
        dt = float(self.config.dt)
        desired = np.asarray(params.desired_force, dtype=np.float32).reshape(6)
        limits = np.maximum(np.asarray(params.limit_vel, dtype=np.float32).reshape(6), 0.0)
        axis_scalars: list[float] = []

        for axis in range(3):
            axis_world = np.asarray(basis[:, axis], dtype=np.float32).reshape(3)
            mode = int(params.control_mode[axis])
            s_des = self._project_scalar(target, axis_world)
            s_fixed = self._project_scalar(self._fixed_pose, axis_world)
            s_cmd = self._axis_scalar_cmd(
                axis,
                mode,
                s_des=s_des,
                s_fixed=s_fixed,
                wrench=float(measured[axis]),
                desired_wrench=float(desired[axis]),
                limit_vel=float(limits[axis]),
                dt=dt,
                is_torque=False,
            )
            self._set_axis_scalar(adjusted, axis_world, s_cmd)
            axis_scalars.append(s_cmd)

        delta_rot = np.zeros(3, dtype=np.float32)
        for axis in range(3, 6):
            local_idx = axis - 3
            mode = int(params.control_mode[axis])
            if mode == RM_CTRL_FIXED:
                continue
            s_des = float(target[3 + local_idx])
            s_fixed = float(self._fixed_pose[3 + local_idx])
            s_cmd = self._axis_scalar_cmd(
                axis,
                mode,
                s_des=s_des,
                s_fixed=s_fixed,
                wrench=float(measured[axis]),
                desired_wrench=float(desired[axis]),
                limit_vel=float(limits[axis]),
                dt=dt,
                is_torque=True,
            )
            delta_rot[local_idx] = s_cmd - float(adjusted[3 + local_idx])
            adjusted[3 + local_idx] = s_cmd

        if int(params.mode) == 1 and int(params.control_mode[2]) == RM_CTRL_ADAPTIVE:
            tilt_lim = float(self.config.adaptive_tilt_limit_rad)
            gain = float(self.config.adaptive_tilt_gain)
            delta_rot[0] += float(np.clip(gain * float(measured[0]), -tilt_lim, tilt_lim))
            delta_rot[1] += float(np.clip(gain * float(measured[1]), -tilt_lim, tilt_lim))

        if float(np.linalg.norm(delta_rot)) > 0.0:
            adjusted = apply_pose_delta_wxyz(adjusted, np.concatenate([np.zeros(3, dtype=np.float32), delta_rot], dtype=np.float32))

        return adjusted, {
            "requested_pose": target.tolist(),
            "adjusted_pose": adjusted.tolist(),
            "measured_wrench": measured.tolist(),
            "wrench_world": wrench_world.tolist(),
            "wrench_sensor": wrench_sensor.tolist(),
            "control_mode": list(params.control_mode),
            "desired_force": list(params.desired_force),
            "limit_vel": list(params.limit_vel),
            "axis_scalars": axis_scalars,
            "sim_model": "realman_mbk_outer_loop_v2",
        }

    def step_pose(
        self,
        target_pose: list[float] | tuple[float, ...] | np.ndarray,
        params: ForcePositionHybridParams,
        *,
        nullspace_target: np.ndarray | None = None,
    ) -> ControllerStepResult:
        adjusted, metadata = self.adjusted_pose(target_pose, params)
        result = self.inner.step(
            CartesianControlTarget(
                pose=adjusted,
                nullspace_target=nullspace_target,
                metadata={"force_position_hybrid": metadata},
            )
        )
        result.metadata = {**result.metadata, "force_position_hybrid": metadata}
        return result

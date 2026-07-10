"""
Xbox / pygame gamepad -> desired TCP pose increments -> Cartesian controller.

This module only wires ``MotionInterface`` + controller + stepping; it does **not** load scenes or URDFs.

**Linux + ``linux_xbox`` axis profile (default on Linux)** — SDL/pygame mapping aligned with PEIRASTIC Franka teleop:

- Left stick: planar translation (world X/Y after code sign mapping).
- LT / RT: TCP translation along world Z (up / down).
- Right stick: tool-frame rotation about local X and Y.
- ``linux_xbox_hybrid``: D-pad left/right adds tool-frame rotation about local Z.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.cartesian_pose import (
    CartesianPoseControllerConfig,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.base import CartesianControlTarget
from projects.genesis_ue_sync.sim_platform.control.controllers.common import (
    apply_pose_delta_wxyz,
    normalize_quaternion_wxyz,
)
from projects.genesis_ue_sync.sim_platform.control.controllers.registry import build_cartesian_teleop_controller
from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_contact import (
    read_virtual_contact_force_world,
    read_virtual_contact_wrench,
)
from projects.genesis_ue_sync.sim_platform.control.teleop.xbox_gamepad import XboxGamepad


@dataclass
class RuckigLinearVelocityPlanner:
    """Ruckig OTG for Cartesian translation velocity commands."""

    dt: float
    initial_position: np.ndarray
    max_velocity: float = 0.20
    max_acceleration: float = 1.50
    max_jerk: float = 8.0

    def __post_init__(self) -> None:
        try:
            from ruckig import ControlInterface, InputParameter, OutputParameter, Ruckig  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Ruckig is required for ruckig teleop. Install it in the Genesis env, e.g. "
                "pip install /media/camp/EXT_DRIVE/Among_US/ref_code_library/PEIRASTIC_control/third_party/ruckig"
            ) from exc
        self._control_interface = ControlInterface
        self._otg = Ruckig(3, float(self.dt))
        self._inp = InputParameter(3)
        self._out = OutputParameter(3)
        self._inp.control_interface = ControlInterface.Velocity
        pos = np.asarray(self.initial_position, dtype=np.float64).reshape(3)
        self._inp.current_position = pos.tolist()
        self._inp.current_velocity = [0.0, 0.0, 0.0]
        self._inp.current_acceleration = [0.0, 0.0, 0.0]
        self._inp.target_velocity = [0.0, 0.0, 0.0]
        self._inp.target_acceleration = [0.0, 0.0, 0.0]
        self._inp.max_velocity = [float(self.max_velocity)] * 3
        self._inp.max_acceleration = [float(self.max_acceleration)] * 3
        self._inp.max_jerk = [float(self.max_jerk)] * 3

    def reset_position(self, position: np.ndarray, *, velocity: np.ndarray | None = None) -> None:
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        vel = np.zeros(3, dtype=np.float64) if velocity is None else np.asarray(velocity, dtype=np.float64).reshape(3)
        self._inp.current_position = pos.tolist()
        self._inp.current_velocity = vel.tolist()
        self._inp.current_acceleration = [0.0, 0.0, 0.0]
        self._inp.target_velocity = vel.tolist()
        self._inp.target_acceleration = [0.0, 0.0, 0.0]

    def update(self, target_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vel = np.asarray(target_velocity, dtype=np.float64).reshape(3)
        self._inp.target_velocity = vel.tolist()
        self._inp.target_acceleration = [0.0, 0.0, 0.0]
        self._otg.update(self._inp, self._out)
        pos = np.asarray(self._out.new_position, dtype=np.float32).reshape(3)
        new_vel = np.asarray(self._out.new_velocity, dtype=np.float32).reshape(3)
        self._out.pass_to_input(self._inp)
        return pos, new_vel


def cartesian_follow_controller_config(dt: float) -> CartesianPoseControllerConfig:
    """Resolved-rate Cartesian tracking → joint-position targets (stable vs ``control_joint_velocities``)."""
    return CartesianPoseControllerConfig(
        dt=float(dt),
        output_mode="joint_position",
        linear_gain=24.0,
        angular_gain=16.0,
        linear_damping=1.8,
        angular_damping=1.2,
        damping=0.07,
        max_linear_speed=0.85,
        max_angular_speed=3.0,
        max_joint_speed=3.5,
        nullspace_stiffness=0.9,
        nullspace_damping=0.25,
    )


def teleop_hybrid_limit_vel(trans_scale: float, rot_scale: float) -> list[float]:
    """Map teleop stick scales to RealMan ``limit_vel[6]`` for hybrid outer loops."""

    ts = max(float(trans_scale), 1e-4)
    rs = max(float(rot_scale), 1e-4)
    return [ts, ts, ts, rs, rs, rs]


def integrate_gamepad_pose_target(
    *,
    target_pose: np.ndarray,
    pad: XboxGamepad,
    trans_scale: float,
    rot_scale: float,
    dt: float,
    origin_pose: np.ndarray | None = None,
    relative_limits: tuple[float, float, float] | None = None,
    measured_pose: np.ndarray | None = None,
    sync_orientation_when_idle: bool = True,
    max_angular_step_rad: float = 0.08,
) -> np.ndarray:
    """Integrate stick deltas onto a commanded TCP pose (tool-frame rotation, world translation)."""

    vec = pad.read_action_vector()
    tx, ty, tz, rx, ry, rz = vec.tolist()
    linear_velocity_cmd = np.array(
        [-ty * float(trans_scale), -tx * float(trans_scale), -tz * float(trans_scale)],
        dtype=np.float32,
    )
    angular_delta = np.array(
        [rx * float(rot_scale) * dt, ry * float(rot_scale) * dt, rz * float(rot_scale) * dt],
        dtype=np.float32,
    )
    ang_norm = float(np.linalg.norm(angular_delta))
    if ang_norm > float(max_angular_step_rad) > 0.0:
        angular_delta *= float(max_angular_step_rad) / ang_norm
    rotating = ang_norm > 1e-6
    delta = np.concatenate([linear_velocity_cmd * float(dt), angular_delta], dtype=np.float32)
    next_pose = apply_pose_delta_wxyz(target_pose, delta)
    next_pose[3:7] = normalize_quaternion_wxyz(next_pose[3:7])
    if sync_orientation_when_idle and not rotating and measured_pose is not None:
        next_pose[3:7] = normalize_quaternion_wxyz(np.asarray(measured_pose, dtype=np.float32).reshape(7)[3:7])
    if origin_pose is not None and relative_limits is not None:
        origin = np.asarray(origin_pose, dtype=np.float32).reshape(-1)
        rel = np.asarray(relative_limits, dtype=np.float32).reshape(3)
        next_pose[:3] = np.clip(next_pose[:3], origin[:3] - rel, origin[:3] + rel)
    return np.asarray(next_pose, dtype=np.float32).reshape(-1)


def teleop_cartesian_step(
    *,
    motion: Any,
    pad: XboxGamepad,
    cart: Any,
    trans_scale: float,
    rot_scale: float,
    dt: float,
    nullspace_target: np.ndarray | None = None,
    workspace_limits: dict[str, tuple[float, float]] | None = None,
    origin_pose: np.ndarray | None = None,
    relative_limits: tuple[float, float, float] | None = None,
    feedforward_twist: bool = True,
    ruckig_planner: RuckigLinearVelocityPlanner | None = None,
    measured_pose: np.ndarray | None = None,
    max_tracking_error_m: float | None = None,
    sync_orientation_when_idle: bool = True,
) -> np.ndarray:
    """One teleop command: sticks → desired TCP delta → controller step.

    ``dt`` should match how often this is called (e.g. ``runtime.config.dt`` once per ``runtime.step``).
    """
    target_pose = np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(-1)
    return teleop_cartesian_step_from_target(
        target_pose=target_pose,
        pad=pad,
        cart=cart,
        trans_scale=trans_scale,
        rot_scale=rot_scale,
        dt=dt,
        nullspace_target=nullspace_target,
        workspace_limits=workspace_limits,
        origin_pose=origin_pose,
        relative_limits=relative_limits,
        feedforward_twist=feedforward_twist,
        ruckig_planner=ruckig_planner,
        measured_pose=measured_pose,
        max_tracking_error_m=max_tracking_error_m,
        sync_orientation_when_idle=sync_orientation_when_idle,
    )


def teleop_cartesian_step_from_target(
    *,
    target_pose: np.ndarray,
    pad: XboxGamepad,
    cart: Any,
    trans_scale: float,
    rot_scale: float,
    dt: float,
    nullspace_target: np.ndarray | None = None,
    workspace_limits: dict[str, tuple[float, float]] | None = None,
    origin_pose: np.ndarray | None = None,
    relative_limits: tuple[float, float, float] | None = None,
    feedforward_twist: bool = True,
    ruckig_planner: RuckigLinearVelocityPlanner | None = None,
    measured_pose: np.ndarray | None = None,
    max_tracking_error_m: float | None = None,
    sync_orientation_when_idle: bool = True,
) -> np.ndarray:
    """Integrate gamepad increments onto the desired TCP pose, not the measured TCP pose."""
    if measured_pose is not None and max_tracking_error_m is not None:
        measured = np.asarray(measured_pose, dtype=np.float32).reshape(7)
        current_target = np.asarray(target_pose, dtype=np.float32).reshape(7)
        if float(np.linalg.norm(current_target[:3] - measured[:3])) > float(max_tracking_error_m):
            target_pose = measured.copy()
            if ruckig_planner is not None:
                ruckig_planner.reset_position(measured[:3])
    vec = pad.read_action_vector()
    tx, ty, tz, rx, ry, rz = vec.tolist()
    linear_velocity_cmd = np.array(
        [-ty * float(trans_scale), -tx * float(trans_scale), -tz * float(trans_scale)],
        dtype=np.float32,
    )
    angular_delta = np.array(
        [rx * float(rot_scale) * dt, ry * float(rot_scale) * dt, rz * float(rot_scale) * dt],
        dtype=np.float32,
    )
    max_angular_step = 0.08
    ang_norm = float(np.linalg.norm(angular_delta))
    if ang_norm > max_angular_step:
        angular_delta *= max_angular_step / ang_norm
    rotating = ang_norm > 1e-6
    if ruckig_planner is not None:
        next_xyz, linear_velocity = ruckig_planner.update(linear_velocity_cmd)
        next_pose = np.asarray(target_pose, dtype=np.float32).reshape(7).copy()
        next_pose[:3] = next_xyz
        next_pose = apply_pose_delta_wxyz(next_pose, np.concatenate([np.zeros(3, dtype=np.float32), angular_delta]))
    else:
        delta = np.concatenate([linear_velocity_cmd * float(dt), angular_delta], dtype=np.float32)
        next_pose = apply_pose_delta_wxyz(target_pose, delta)
        linear_velocity = linear_velocity_cmd
    next_pose[3:7] = normalize_quaternion_wxyz(next_pose[3:7])
    if sync_orientation_when_idle and not rotating and measured_pose is not None:
        next_pose[3:7] = normalize_quaternion_wxyz(np.asarray(measured_pose, dtype=np.float32).reshape(7)[3:7])
    unclipped_xyz = next_pose[:3].copy()
    if workspace_limits:
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            if axis_name in workspace_limits:
                lo, hi = workspace_limits[axis_name]
                next_pose[axis_idx] = float(np.clip(next_pose[axis_idx], float(lo), float(hi)))
    if origin_pose is not None and relative_limits is not None:
        origin = np.asarray(origin_pose, dtype=np.float32).reshape(-1)
        rel = np.asarray(relative_limits, dtype=np.float32).reshape(3)
        next_pose[:3] = np.clip(next_pose[:3], origin[:3] - rel, origin[:3] + rel)
    if ruckig_planner is not None and not np.allclose(unclipped_xyz, next_pose[:3], atol=1e-7, rtol=0.0):
        ruckig_planner.reset_position(next_pose[:3])
        linear_velocity = np.zeros(3, dtype=np.float32)
    angular_velocity = angular_delta / max(float(dt), 1e-6)
    twist = np.concatenate([linear_velocity, angular_velocity], dtype=np.float32) if feedforward_twist else np.zeros(6, dtype=np.float32)
    cart.step(
        CartesianControlTarget(
            pose=next_pose,
            twist=twist,
            nullspace_target=nullspace_target,
            metadata={"dt": float(dt)},
        )
    )
    return np.asarray(next_pose, dtype=np.float32).reshape(-1)


def run_gamepad_cartesian_teleop_loop(
    *,
    runtime: Any,
    motion: Any,
    robot_name: str,
    tcp_link: str,
    pad: XboxGamepad,
    duration_s: float,
    rate_hz: float,
    trans_scale: float,
    rot_scale: float,
    print_interval: float,
    print_contact: bool,
    control_mode: str = "cartesian",
    use_ruckig: bool = True,
    profile_label: str = "",
    viewer_alive: Callable[[], bool] | None = None,
    pre_step_callback: Callable[[int, float], None] | None = None,
    force_sensor_reader: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None,
    nullspace_target: np.ndarray | None = None,
) -> None:
    """Teleop using Cartesian position following by default.

    ``control_mode="cartesian"`` uses ``CartesianPoseController(output_mode="joint_position")``: resolved-rate
    Cartesian tracking (integrates joint targets each step). This avoids Genesis velocity-actuator drift.

    ``control_mode="osc"`` uses operational-space torque control. ``control_mode="ik"`` is kept for
    debugging only and uses IK → joint-position targets.

    Idle is zero increment on the desired TCP pose, so the Cartesian target remains fixed instead of
    following measured drift.

    If ``viewer_alive`` is set (e.g. ``lambda: runtime.scene.visualizer.viewer.is_alive()``), the
    loop also exits when it returns False.
    """
    dt = 1.0 / max(float(rate_hz), 1e-6)
    mode = str(control_mode).strip().lower()
    cart = build_cartesian_teleop_controller(mode=mode, motion=motion, dt=float(dt), link_name=tcp_link)
    t_end = time.perf_counter() + float(duration_s)
    last_print = time.perf_counter()

    def _measured_control_pose() -> np.ndarray:
        if mode == "osc_impedance" and hasattr(cart, "current_pose"):
            return np.asarray(cart.current_pose(), dtype=np.float32).reshape(7)
        return np.asarray(motion.get_tcp_pose(), dtype=np.float32).reshape(7)

    target_pose = np.asarray(_measured_control_pose(), dtype=np.float32).reshape(-1)
    ruckig_planner = (
        RuckigLinearVelocityPlanner(dt=float(dt), initial_position=target_pose[:3])
        if mode == "cartesian" and bool(use_ruckig)
        else None
    )
    msg = (
        f"Gamepad Cartesian teleop (mode={mode}). "
        "Left stick XY, LT/RT vertical Z, right stick rotation (linux_xbox profile). Ctrl+C to exit."
    )
    if profile_label:
        msg += f" Profile: {profile_label}."
    if str(os.environ.get("AMONGUS_GAMEPAD_QUIET", "") or "").strip().lower() not in ("1", "true", "yes", "on"):
        print(msg)

    step_index = 0
    try:
        while True:
            if time.perf_counter() >= t_end:
                break
            if viewer_alive is not None and not viewer_alive():
                break
            t0 = time.perf_counter()
            target_pose = teleop_cartesian_step_from_target(
                target_pose=target_pose,
                pad=pad,
                cart=cart,
                trans_scale=float(trans_scale),
                rot_scale=float(rot_scale),
                dt=float(dt),
                nullspace_target=nullspace_target,
                workspace_limits=None,
                origin_pose=target_pose,
                relative_limits=None,
                feedforward_twist=True,
                ruckig_planner=ruckig_planner,
                measured_pose=_measured_control_pose(),
                max_tracking_error_m=0.08,
            )
            if pre_step_callback is not None:
                pre_step_callback(step_index, time.perf_counter())
            runtime.step()
            step_index += 1

            if print_contact:
                now = time.perf_counter()
                if now - last_print >= float(print_interval):
                    if force_sensor_reader is not None:
                        w_world, w_sensor = force_sensor_reader()
                        print(f"FT_world={np.round(w_world, 3)} | FT_sensor={np.round(w_sensor, 3)}")
                    else:
                        f_world = read_virtual_contact_force_world(runtime, robot_name, link_name=tcp_link)
                        w6 = read_virtual_contact_wrench(runtime, robot_name, link_name=tcp_link)
                        print(
                            f"F_world={np.round(f_world, 3)} N | wrench6={np.round(w6, 3)} "
                            f"(moments placeholder zeros)"
                        )
                    last_print = now

            elapsed = time.perf_counter() - t0
            sleep_remain = dt - elapsed
            if sleep_remain > 0.0:
                time.sleep(sleep_remain)
    except KeyboardInterrupt:
        pass

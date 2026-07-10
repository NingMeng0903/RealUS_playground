from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.sim_platform.control.controllers.common import quat_wxyz_to_rotation_matrix
from projects.genesis_ue_sync.sim_platform.control.teleop.virtual_contact import read_virtual_contact_force_world
from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SceneRobotForceSensorSpec


def _pose_wxyz_to_homogeneous(pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64).reshape(7)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_wxyz_to_rotation_matrix(p[3:7]).astype(np.float64)
    out[:3, 3] = p[:3]
    return out


def read_scene_virtual_force_sensor_wrench(
    runtime: Any,
    robot_name: str,
    spec: SceneRobotForceSensorSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Return configured virtual F/T wrench in world and sensor frames."""

    try:
        force_contact_world = read_virtual_contact_force_world(runtime, robot_name, link_name=str(spec.contact_link))
        mount_pose = np.asarray(runtime.get_link_pose(robot_name, str(spec.mount_link)), dtype=np.float32).reshape(7)
    except Exception:
        z = np.zeros(6, dtype=np.float32)
        return z, z

    T_world_link = _pose_wxyz_to_homogeneous(mount_pose)
    T_world_sensor = T_world_link @ np.asarray(spec.link_T_sensor, dtype=np.float64).reshape(4, 4)
    rot_ws = np.asarray(T_world_sensor[:3, :3], dtype=np.float64)

    f_c_sensor = rot_ws.T @ np.asarray(force_contact_world, dtype=np.float64).reshape(3)
    lever_contact = np.asarray(spec.sensor_T_contact, dtype=np.float64).reshape(4, 4)[:3, 3]
    tau_c_sensor = np.cross(lever_contact.reshape(3), f_c_sensor.reshape(3))
    wrench_contact_sensor = np.concatenate([f_c_sensor, tau_c_sensor], dtype=np.float64)

    wrench_tool_sensor = np.zeros(6, dtype=np.float64)
    if float(spec.tool_mass_kg) > 1e-9:
        gravity_world = np.asarray(spec.gravity_world_m_s2, dtype=np.float64).reshape(3)
        force_tool_world = float(spec.tool_mass_kg) * gravity_world
        force_tool_sensor = rot_ws.T @ force_tool_world
        com_sensor = np.asarray(spec.tool_com_sensor_m, dtype=np.float64).reshape(3)
        tau_tool_sensor = np.cross(com_sensor, force_tool_sensor)
        wrench_tool_sensor = np.concatenate([force_tool_sensor, tau_tool_sensor], dtype=np.float64)

    wrench_sensor = (
        wrench_contact_sensor
        if bool(spec.subtract_static_wrench_from_output)
        else wrench_contact_sensor + wrench_tool_sensor
    )
    wrench_world = np.concatenate(
        [
            rot_ws @ wrench_sensor[:3].reshape(3),
            rot_ws @ wrench_sensor[3:].reshape(3),
        ],
        dtype=np.float64,
    )
    return (
        np.asarray(wrench_world, dtype=np.float32).reshape(6),
        np.asarray(wrench_sensor, dtype=np.float32).reshape(6),
    )

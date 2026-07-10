"""Resolve a base scene yaml + robot model_id into the payload Genesis and UE both consume."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.scenes.common_scene import (
    SceneRobotForceSensorSpec,
    SceneRobotProbeCollisionSpec,
    SceneRobotSpec,
    SceneRobotTcpControlSpec,
    SyncSceneSpec,
    load_sync_scene_payload,
    load_sync_scene_spec,
)
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import select_robot_specs


def effective_robot_model_id(*, cli_model: str = "", env_key: str = "AMONGUS_ROBOT_MODEL") -> str:
    return str(cli_model or os.environ.get(env_key, "") or "").strip()


def robot_spec_to_scene_dict(
    robot_spec: SceneRobotSpec,
    *,
    for_ue_spawn: bool = True,
) -> dict[str, Any]:
    """Serialize a merged SceneRobotSpec back to a scene-spec robot mapping."""

    payload: dict[str, Any] = {
        "name": str(robot_spec.name),
        "model_id": str(robot_spec.model_id),
        "base_pos": [float(v) for v in robot_spec.base_pos],
        "joint_positions": [float(v) for v in robot_spec.joint_positions],
        "use_visual_mesh": bool(robot_spec.use_visual_mesh),
        "allow_collision_fallback": bool(robot_spec.allow_collision_fallback),
        "visual_mesh_format": str(robot_spec.visual_mesh_format),
        "visual_mesh_scale": float(robot_spec.visual_mesh_scale),
        "ue_visual_asset_root": str(robot_spec.ue_visual_asset_root),
        "color": [float(v) for v in robot_spec.color],
    }
    if robot_spec.base_quat_xyzw is not None:
        payload["base_quat_xyzw"] = [float(v) for v in robot_spec.base_quat_xyzw]
    if robot_spec.genesis_link_visual_urdf_rgba:
        payload["genesis_link_visual_urdf_rgba"] = {
            k: [float(c) for c in v] for k, v in robot_spec.genesis_link_visual_urdf_rgba.items()
        }
    payload["use_collision_geometry"] = (
        False
        if for_ue_spawn
        else bool(robot_spec.use_collision_geometry)
    )
    pc: SceneRobotProbeCollisionSpec | None = robot_spec.probe_collision
    if pc is not None:
        payload["probe_collision"] = {
            "enabled": bool(pc.enabled),
            "link_name": str(pc.link_name),
            "shape": str(pc.shape),
            "radius": float(pc.radius),
            "length": float(pc.length),
            "origin_xyz": None if pc.origin_xyz is None else [float(v) for v in pc.origin_xyz],
            "origin_rpy": None if pc.origin_rpy is None else [float(v) for v in pc.origin_rpy],
            "mesh_filename": pc.mesh_filename,
        }
    fs: SceneRobotForceSensorSpec | None = robot_spec.force_sensor
    if fs is not None:
        payload["force_sensor"] = {
            "mount_link": fs.mount_link,
            "contact_link": fs.contact_link,
            "link_T_sensor": fs.link_T_sensor,
            "sensor_T_contact": fs.sensor_T_contact,
            "tool_mass_kg": float(fs.tool_mass_kg),
            "tool_com_sensor_m": [float(v) for v in fs.tool_com_sensor_m],
            "gravity_world_m_s2": [float(v) for v in fs.gravity_world_m_s2],
            "subtract_static_wrench_from_output": bool(fs.subtract_static_wrench_from_output),
        }
    tcp: SceneRobotTcpControlSpec | None = robot_spec.tcp_control
    if tcp is not None and tcp.link_name:
        payload["tcp_control"] = {
            "link_name": tcp.link_name,
            "local_point_m": (
                None if tcp.local_point_m is None else [float(v) for v in tcp.local_point_m]
            ),
        }
    return payload


def apply_robot_model_to_scene_payload(
    payload: dict[str, Any],
    scene_spec: SyncSceneSpec,
    robot_spec: SceneRobotSpec,
    *,
    for_ue_spawn: bool = True,
) -> dict[str, Any]:
    out = dict(payload)
    if "robots" in out and isinstance(out["robots"], list) and out["robots"]:
        robots = [dict(item) if isinstance(item, dict) else {} for item in out["robots"]]
        robots[0] = robot_spec_to_scene_dict(robot_spec, for_ue_spawn=for_ue_spawn)
        out["robots"] = robots
    else:
        out["robot"] = robot_spec_to_scene_dict(robot_spec, for_ue_spawn=for_ue_spawn)
    return out


def resolve_scene_spec_payload(
    scene_spec_path: str | Path,
    *,
    robot_model: str = "",
    for_ue_spawn: bool = True,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Load base scene yaml and merge ``assets/robots/<model_id>/robot.yaml`` into the robot block."""

    path = Path(scene_spec_path).expanduser().resolve()
    payload = load_sync_scene_payload(path)
    scene_spec = load_sync_scene_spec(path)
    model_id = effective_robot_model_id(cli_model=robot_model)
    specs = select_robot_specs(scene_spec, robots_mode="scene", robot_model=model_id)
    if not specs:
        return payload
    robot_spec = specs[0]
    return apply_robot_model_to_scene_payload(
        payload,
        scene_spec,
        robot_spec,
        for_ue_spawn=for_ue_spawn,
    )


def write_resolved_scene_selection(
    *,
    repo_root: Path,
    scene_spec_path: Path,
    robot_spec: SceneRobotSpec,
    resolved_robot: dict[str, Any],
) -> Path:
    out_path = repo_root / "outputs" / "genesis_viz" / "last_robot_scene_selection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "scene_spec": str(scene_spec_path),
                "model_id": str(robot_spec.model_id),
                "robot_name": str(robot_spec.name),
                "robot": dict(resolved_robot),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path

"""Spawn articulated arms from scene specs + ``assets/robots/<model_id>/robot.yaml`` manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.project import project_paths

from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SceneRobotSpec, SyncSceneSpec, _load_robot_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_probe_urdf import resolved_robot_urdf_for_robot_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry, robot_capabilities_for_spec


@dataclass(frozen=True)
class SpawnedRobots:
    names: tuple[str, ...]
    primary_name: str
    home_q: dict[str, np.ndarray]
    specs: tuple[SceneRobotSpec, ...]


def _robot_metadata(robot_spec: SceneRobotSpec) -> dict[str, Any]:
    meta = dict(getattr(robot_spec, "asset_metadata", None) or {})
    meta.update(dict(getattr(robot_spec, "metadata", None) or {}))
    return meta


def assert_robot_model_ready(robot_spec: SceneRobotSpec) -> None:
    meta = _robot_metadata(robot_spec)
    model_id = str(getattr(robot_spec, "model_id", "") or "").strip()
    if str(meta.get("status", "")).strip().lower() == "placeholder":
        note = str(meta.get("note", "") or "").strip()
        raise SystemExit(
            f"Robot model {model_id!r} is a placeholder (URDF/meshes not installed). "
            f"{note} Use --robots none for human-only Genesis, or pick panda_urdf / rm75_6f."
        )
    if not robot_spec.resolved_urdf_path.is_file():
        raise SystemExit(
            f"Robot {robot_spec.name!r} (model_id={model_id!r}) URDF missing: {robot_spec.resolved_urdf_path}"
        )


def robot_spec_with_model_id(
    robot_spec: SceneRobotSpec,
    model_id: str,
    *,
    model_overrides: dict[str, Any] | None = None,
) -> SceneRobotSpec:
    """Re-merge ``assets/robots/<model_id>/robot.yaml`` while keeping scene pose/instance name."""

    target_mid = str(model_id).strip()
    source_mid = str(robot_spec.model_id or "").strip()
    payload: dict[str, Any] = {
        "model_id": target_mid,
        "name": robot_spec.name,
        "instance_id": robot_spec.instance_id or robot_spec.name,
        "base_pos": [float(v) for v in robot_spec.base_pos],
    }
    if robot_spec.base_quat_xyzw is not None:
        payload["base_quat_xyzw"] = [float(v) for v in robot_spec.base_quat_xyzw]
    overrides = dict(model_overrides or {})
    if overrides.get("joint_positions"):
        payload["joint_positions"] = [float(v) for v in overrides["joint_positions"]]
    elif (
        source_mid
        and target_mid.lower() == source_mid.lower()
        and robot_spec.joint_positions
    ):
        payload["joint_positions"] = [float(v) for v in robot_spec.joint_positions]
    if robot_spec.color:
        payload["color"] = [float(v) for v in robot_spec.color]
    if overrides.get("genesis_link_visual_urdf_rgba"):
        payload["genesis_link_visual_urdf_rgba"] = {
            k: [float(c) for c in v]
            for k, v in dict(overrides["genesis_link_visual_urdf_rgba"]).items()
        }
    elif robot_spec.genesis_link_visual_urdf_rgba and target_mid.lower() == source_mid.lower():
        payload["genesis_link_visual_urdf_rgba"] = {
            k: [float(c) for c in v] for k, v in robot_spec.genesis_link_visual_urdf_rgba.items()
        }
    for key in (
        "use_collision_geometry",
        "use_visual_mesh",
        "allow_collision_fallback",
        "ue_visual_asset_root",
        "visual_mesh_format",
        "visual_mesh_scale",
    ):
        if key in overrides:
            payload[key] = overrides[key]
    mid = target_mid.lower()
    if mid in {"panda_urdf", "panda"} and robot_spec.probe_collision is not None:
        pc = robot_spec.probe_collision
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
        if robot_spec.tcp_control is not None and robot_spec.tcp_control.link_name:
            payload["tcp_control"] = {
                "link_name": robot_spec.tcp_control.link_name,
                "local_point_m": (
                    None
                    if robot_spec.tcp_control.local_point_m is None
                    else [float(v) for v in robot_spec.tcp_control.local_point_m]
                ),
            }
        if robot_spec.force_sensor is not None:
            fs = robot_spec.force_sensor
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
    return _load_robot_spec(payload)


def select_robot_specs(
    scene_spec: SyncSceneSpec,
    *,
    robots_mode: str,
    no_robot: bool = False,
    robot_model: str = "",
    robot_instances: str = "",
) -> list[SceneRobotSpec]:
    if no_robot:
        return []
    mode = str(robots_mode or "scene").strip().lower()
    if mode in {"", "none", "off", "human-only"}:
        return []
    specs = list(scene_spec.iter_robot_specs())
    if str(robot_model or "").strip():
        if not specs:
            raise SystemExit("select_robot_specs: scene has no robot entry to apply --robot-model.")
        mid = str(robot_model).strip()
        overrides = dict(scene_spec.robot_model_overrides.get(mid) or {})
        specs[0] = robot_spec_with_model_id(specs[0], mid, model_overrides=overrides)
    if str(robot_instances or "").strip():
        wanted = {s.strip() for s in str(robot_instances).split(",") if s.strip()}
        specs = [s for s in specs if s.name in wanted]
        if not specs:
            raise SystemExit(f"No scene robot instances matched --robot-instances={robot_instances!r}.")
        return specs
    if mode in {"scene", "default", "all"}:
        return specs
    if mode == "none":
        return []
    wanted = {s.strip() for s in mode.split(",") if s.strip()}
    filtered = [s for s in specs if s.name in wanted]
    if not filtered:
        raise SystemExit(
            f"--robots {robots_mode!r}: no instance names in scene ({[s.name for s in scene_spec.iter_robot_specs()]})."
        )
    return filtered


@dataclass(frozen=True)
class GamepadTeleopProfile:
    mode: str
    profile_key: str
    osc_config_path: Path | None = None
    use_ruckig: bool = False


_PROFILE_TYPE_TO_TELEOP_MODE: dict[str, str] = {
    "osc_impedance": "osc_impedance",
    "osc": "osc",
    "cartesian_pose": "cartesian",
    "sim_admittance_outer_loop": "sim_admittance_outer_loop",
    "admittance": "sim_admittance_outer_loop",
    "cartesian_admittance": "sim_admittance_outer_loop",
}


def _simulation_profile(controllers: dict[str, Any], profile_key: str) -> dict[str, Any]:
    profiles = dict(controllers.get("simulation_profiles", {}) or {})
    return dict(profiles.get(str(profile_key).strip(), {}) or {})


def resolve_gamepad_teleop_profile(
    robot_spec: SceneRobotSpec,
    *,
    cli_mode: str = "",
    profile_key: str = "",
) -> GamepadTeleopProfile:
    """Resolve one gamepad inner loop from ``assets/robots/<model_id>/robot.yaml``."""

    explicit_key = str(profile_key or "").strip()
    if explicit_key:
        return _gamepad_profile_from_key(robot_spec, explicit_key)

    cli = str(cli_mode or "").strip().lower()
    controllers = dict(getattr(robot_spec, "controllers", None) or {})
    profiles = dict(controllers.get("simulation_profiles", {}) or {})
    caps = robot_capabilities_for_spec(robot_spec)

    if cli and cli not in {"auto", "default"}:
        mode = cli
        profile_key = str(controllers.get("gamepad_profile") or controllers.get("default") or cli)
        prof = _simulation_profile(controllers, profile_key)
        if mode == "osc_impedance" and not prof:
            prof = next(
                (dict(v) for k, v in profiles.items() if str(v.get("type", "")).lower() == "osc_impedance"),
                {},
            )
            if prof:
                profile_key = next(k for k, v in profiles.items() if str(v.get("type", "")).lower() == "osc_impedance")
        osc_path = default_osc_impedance_config_path(robot_spec) if mode == "osc_impedance" else None
        if mode == "osc_impedance" and osc_path is None:
            raise SystemExit(
                f"No osc_impedance config for model_id={robot_spec.model_id!r}; "
                "use --teleop-control-mode cartesian or add simulation_profiles.osc_impedance."
            )
        use_ruckig = bool(prof.get("use_ruckig", mode == "cartesian"))
        return GamepadTeleopProfile(
            mode=mode,
            profile_key=str(profile_key),
            osc_config_path=osc_path,
            use_ruckig=use_ruckig,
        )

    profile_key = str(controllers.get("gamepad_profile") or controllers.get("default") or "").strip()
    if not profile_key:
        profile_key = next(iter(profiles.keys()), "")

    prof = _simulation_profile(controllers, profile_key)
    ptype = str(prof.get("type", "cartesian_pose")).strip().lower()
    if not prof and caps.supports_torque_control:
        for key, candidate in profiles.items():
            if str(candidate.get("type", "")).strip().lower() == "osc_impedance":
                profile_key = str(key)
                prof = dict(candidate)
                ptype = "osc_impedance"
                break

    mode = _PROFILE_TYPE_TO_TELEOP_MODE.get(ptype, "cartesian")
    if mode == "osc_impedance" and not caps.supports_torque_control:
        raise SystemExit(
            f"model_id={robot_spec.model_id!r} gamepad profile {profile_key!r} requires torque control; "
            "set controllers.gamepad_profile to a cartesian_pose profile."
        )

    osc_path: Path | None = None
    if mode == "osc_impedance":
        rel = str(prof.get("config_path", "")).strip()
        if rel:
            osc_path = (project_paths(__file__).root / rel).resolve()
        else:
            osc_path = default_osc_impedance_config_path(robot_spec)
        if osc_path is None:
            raise SystemExit(
                f"model_id={robot_spec.model_id!r}: osc_impedance profile {profile_key!r} missing config_path."
            )

    use_ruckig = bool(prof.get("use_ruckig", mode == "cartesian"))
    return GamepadTeleopProfile(
        mode=mode,
        profile_key=str(profile_key),
        osc_config_path=osc_path,
        use_ruckig=use_ruckig,
    )


def _gamepad_profile_from_key(robot_spec: SceneRobotSpec, profile_key: str) -> GamepadTeleopProfile:
    key = str(profile_key).strip()
    if not key:
        raise ValueError("profile_key must be non-empty.")
    controllers = dict(getattr(robot_spec, "controllers", None) or {})
    prof = _simulation_profile(controllers, key)
    if not prof:
        raise KeyError(
            f"Unknown gamepad profile {key!r} for model_id={robot_spec.model_id!r}. "
            f"Known: {list(dict(controllers.get('simulation_profiles', {}) or {}).keys())!r}"
        )
    caps = robot_capabilities_for_spec(robot_spec)
    ptype = str(prof.get("type", "cartesian_pose")).strip().lower()
    mode = _PROFILE_TYPE_TO_TELEOP_MODE.get(ptype)
    if mode is None:
        raise ValueError(
            f"Profile {key!r} type {ptype!r} is not supported for Xbox Cartesian teleop "
            f"(model_id={robot_spec.model_id!r})."
        )
    if mode == "osc_impedance" and not caps.supports_torque_control:
        raise ValueError(
            f"Profile {key!r} requires torque control; unavailable on model_id={robot_spec.model_id!r}."
        )
    osc_path: Path | None = None
    if mode == "osc_impedance":
        rel = str(prof.get("config_path", "")).strip()
        if rel:
            osc_path = (project_paths(__file__).root / rel).resolve()
        else:
            osc_path = default_osc_impedance_config_path(robot_spec)
        if osc_path is None:
            raise ValueError(f"Profile {key!r} on {robot_spec.model_id!r} missing osc_impedance config_path.")
    use_ruckig = bool(prof.get("use_ruckig", mode == "cartesian"))
    return GamepadTeleopProfile(
        mode=mode,
        profile_key=key,
        osc_config_path=osc_path,
        use_ruckig=use_ruckig,
    )


def list_gamepad_teleop_profile_keys(robot_spec: SceneRobotSpec) -> list[str]:
    """Ordered profile keys the user may cycle at runtime (from robot.yaml)."""

    controllers = dict(getattr(robot_spec, "controllers", None) or {})
    cycle = controllers.get("gamepad_cycle_profiles")
    if isinstance(cycle, list) and cycle:
        keys = [str(item).strip() for item in cycle if str(item).strip()]
    else:
        keys = [str(k) for k in dict(controllers.get("simulation_profiles", {}) or {}).keys()]

    caps = robot_capabilities_for_spec(robot_spec)
    out: list[str] = []
    for key in keys:
        prof = _simulation_profile(controllers, key)
        if not prof:
            continue
        ptype = str(prof.get("type", "")).strip().lower()
        mode = _PROFILE_TYPE_TO_TELEOP_MODE.get(ptype)
        if mode is None:
            continue
        if mode == "osc_impedance" and not caps.supports_torque_control:
            continue
        out.append(key)
    return out


def list_gamepad_teleop_profiles(robot_spec: SceneRobotSpec) -> list[GamepadTeleopProfile]:
    return [_gamepad_profile_from_key(robot_spec, key) for key in list_gamepad_teleop_profile_keys(robot_spec)]


def robot_probe_collision_enabled(robot_specs: list[SceneRobotSpec]) -> bool:
    return any(
        getattr(spec, "probe_collision", None) is not None and bool(spec.probe_collision.enabled)
        for spec in robot_specs
    )


def default_osc_impedance_config_path(robot_spec: SceneRobotSpec) -> Path | None:
    controllers = dict(getattr(robot_spec, "controllers", None) or {})
    profiles = dict(controllers.get("simulation_profiles", {}))
    default_key = str(controllers.get("default", "")).strip()
    ordered_keys = ([default_key] if default_key else []) + list(profiles.keys())
    root = project_paths(__file__).root
    for key in ordered_keys:
        prof = dict(profiles.get(key, {}) or {})
        if str(prof.get("type", "")).strip().lower() != "osc_impedance":
            continue
        rel = str(prof.get("config_path", "")).strip()
        if rel:
            return (root / rel).resolve()
    return None


def add_robots_to_runtime(
    runtime: Any,
    robot_specs: list[SceneRobotSpec],
    registry: RobotRegistry,
    *,
    enable_collision: bool,
    repo_root: Path | None = None,
    suppress_probe_physics_collision: bool = False,
) -> list[str]:
    names: list[str] = []
    for idx, robot_spec in enumerate(robot_specs):
        assert_robot_model_ready(robot_spec)
        suppress = bool(suppress_probe_physics_collision and idx == 0)
        robot_urdf = resolved_robot_urdf_for_robot_spec(
            robot_spec,
            enable_collision=enable_collision,
            repo_root=repo_root,
            suppress_probe_physics_collision=suppress,
        )
        embodiment = registry.build_embodiment(robot_spec, robot_urdf)
        robot_name = str(robot_spec.name)
        runtime.add_articulated_entity(
            embodiment,
            name=robot_name,
            pos=robot_spec.base_pos,
            quat_xyzw=robot_spec.base_quat_xyzw,
        )
        runtime.set_robot_gravity_compensation(robot_name, 1.0)
        names.append(robot_name)
    return names


def init_robots_after_build(
    runtime: Any,
    registry: RobotRegistry,
    robot_specs: list[SceneRobotSpec],
    robot_names: list[str],
) -> SpawnedRobots:
    home: dict[str, np.ndarray] = {}
    for robot_spec, robot_name in zip(robot_specs, robot_names, strict=True):
        registry.apply_pd_gains(runtime, robot_name, robot_spec)
    runtime.reset()
    for robot_spec, robot_name in zip(robot_specs, robot_names, strict=True):
        motion = runtime.get_motion_interface(robot_name)
        jq = np.asarray(registry.initial_joint_positions(robot_spec), dtype=np.float32).reshape(-1)
        motion.set_joint_positions(jq)
        motion.control_joint_positions(jq)
        home[robot_name] = jq.copy()
    primary = robot_names[0] if robot_names else ""
    return SpawnedRobots(
        names=tuple(robot_names),
        primary_name=primary,
        home_q=home,
        specs=tuple(robot_specs),
    )


def resolve_primary_robot_name(
    spawned: SpawnedRobots,
    *,
    teleop_robot: str = "",
) -> str:
    if not spawned.names:
        return ""
    want = str(teleop_robot or "").strip()
    if want:
        if want not in spawned.names:
            raise SystemExit(
                f"--teleop-robot {want!r} not among spawned robots {list(spawned.names)}."
            )
        return want
    return spawned.primary_name

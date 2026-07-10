from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.scenes.robot_assets import resolve_robot_model_payload

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class SceneBoxSpec:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] | None = None
    color: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)


@dataclass
class SceneSupportSurfaceSpec:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] | None = None
    color: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    semantic_role: str = "bed"
    spawn_in_genesis: bool = True
    spawn_in_ue: bool = True
    #: Normal stiffness (N/m) for soft contact against the support plane in offline refit.
    #: Higher values resist penetration more (firmer mattress); sink depth emerges from pen/comp balance.
    contact_stiffness_n_per_m: float = 1.5e6

    @property
    def top_z(self) -> float:
        return float(self.pos[2] + 0.5 * self.size[2])


@dataclass
class SceneRobotProbeCollisionSpec:
    enabled: bool
    link_name: str
    radius: float
    length: float
    origin_xyz: tuple[float, float, float] | None = None
    origin_rpy: tuple[float, float, float] | None = None
    shape: str = "cylinder"
    mesh_filename: str | None = None


@dataclass
class SceneRobotForceSensorSpec:
    mount_link: str
    contact_link: str
    link_T_sensor: list[list[float]]
    sensor_T_contact: list[list[float]]
    tool_mass_kg: float = 0.0
    tool_com_sensor_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity_world_m_s2: tuple[float, float, float] = (0.0, 0.0, -9.81)
    subtract_static_wrench_from_output: bool = False


@dataclass
class SceneRobotTcpControlSpec:
    link_name: str | None = None
    local_point_m: tuple[float, float, float] | None = None


@dataclass
class SceneRobotSpec:
    name: str
    urdf_path: str
    base_pos: tuple[float, float, float]
    model_id: str = ""
    instance_id: str = ""
    base_quat_xyzw: tuple[float, float, float, float] | None = None
    joint_positions: list[float] = field(default_factory=list)
    use_collision_geometry: bool = True
    use_visual_mesh: bool = True
    allow_collision_fallback: bool = False
    mesh_root: str = ""
    visual_mesh_format: str = "fbx"
    ue_visual_asset_root: str = "/Game/Bedlam/Generated/PandaVisual"
    visual_mesh_scale: float = 1.0
    color: tuple[float, float, float, float] = (0.55, 0.55, 0.6, 1.0)
    genesis_link_visual_urdf_rgba: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    controllers: dict[str, Any] = field(default_factory=dict)
    asset_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    probe_collision: SceneRobotProbeCollisionSpec | None = None
    force_sensor: SceneRobotForceSensorSpec | None = None
    tcp_control: SceneRobotTcpControlSpec | None = None

    @property
    def resolved_urdf_path(self) -> Path:
        candidate = Path(self.urdf_path)
        if candidate.is_absolute():
            return candidate
        return project_paths(__file__).resolve_from_root(candidate)


@dataclass
class SceneHumanSpec:
    anchor_pos: tuple[float, float, float]
    support_margin_m: float = 0.015
    support_reference: str = "support_surface_top"
    align_floor: bool = True
    display_vertical_sink_m: float = 0.0
    display_vertical_offset_m: float = 0.0
    display_pitch_forward_deg: float = 0.0
    #: Extra (dx,dy,dz) in meters, Genesis RH Z-up, added only to ``root_translation`` in
    #: ``amongus_canonical_human`` for UE sync. Does not move the Genesis SMPL debug mesh.
    #: Use to align Bedlam/retarget skeleton pelvis vs SMPL visualization (e.g. negative Z if UE looks too high).
    ue_root_offset_genesis_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def bed_contact_margin_m(self) -> float:
        return self.support_margin_m


@dataclass
class SceneEnvironmentSpec:
    ground_plane_color: tuple[float, float, float, float] = (0.92, 0.92, 0.92, 1.0)
    ground_plane_z_m: float = 0.0


@dataclass
class SceneCameraSpec:
    name: str
    res: tuple[int, int] = (1280, 720)
    pos: tuple[float, float, float] = (1.5, -1.5, 1.2)
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 45.0
    near: float = 0.05
    far: float = 100.0
    gui: bool = False
    mount_entity: str | None = None
    mount_link: str | None = None
    pose_rel: list[list[float]] | None = None
    follow_entity: bool = False
    roll_deg: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneMotionSpec:
    source_id: str = ""
    source_path: str = ""
    sequence_npz_path: str = ""
    mesh_manifest_path: str = ""
    fps: float = 30.0
    frame_count: int = 0
    start_frame: int = 0
    frame_step: int = 1

    def _resolve_optional_path(self, path_value: str) -> Optional[Path]:
        if not path_value:
            return None
        candidate = Path(path_value)
        if candidate.is_absolute():
            return candidate
        return project_paths(__file__).resolve_from_root(candidate)

    @property
    def resolved_source_path(self) -> Optional[Path]:
        return self._resolve_optional_path(self.source_path)

    @property
    def resolved_sequence_npz_path(self) -> Optional[Path]:
        return self._resolve_optional_path(self.sequence_npz_path)

    @property
    def resolved_mesh_manifest_path(self) -> Optional[Path]:
        return self._resolve_optional_path(self.mesh_manifest_path)


@dataclass
class SceneRenderSpec:
    fps: float = 8.0
    frame_limit: int = 120
    genesis_backend: str = "cuda"
    ue_frame_count: int = 240
    ue_frame_step: int = 4
    ue_render_now: bool = False
    ue_spawn_robot: bool = True
    ue_spawn_human: bool = True


@dataclass
class SceneLevelAssetBindingSpec:
    adapter_name: str = "ue_scene"
    map_path: str = "/Game/Bedlam/IBLMap"
    hdri_name: str = "autumn_park"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneCharacterVisualBindingSpec:
    adapter_name: str = "ue_character"
    body_mode: str = "retargeted_overlay"
    body_name: str = "it_4375_M_2400"
    texture_body: str | None = "Male_Skin_Preset_0027"
    texture_clothing: str | None = None
    texture_clothing_overlay: str | None = "gr_ben_004_M_texture_01"
    skeletal_mesh_path: str = "/Engine/PS/Bedlam/SMPLX_LH_animations/it_4375_M/it_4375_M_2400"
    animation_asset_root: str = "/Game/Bedlam/Generated/RetargetedAnimations"
    imported_fbx_root: str = "/Game/Bedlam/Generated/ImportedSMPLMotion"
    fallback_animation_path: str = ""
    hidden_material_path: str = "/Engine/PS/Bedlam/Core/Materials/M_SMPLX_Hidden.M_SMPLX_Hidden"
    fbx_global_scale: float = 100.0


@dataclass
class SceneBindingsSpec:
    level: SceneLevelAssetBindingSpec | None = None
    character_visual: SceneCharacterVisualBindingSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


SceneUeAvatarSpec = SceneCharacterVisualBindingSpec


@dataclass
class SyncSceneSpec:
    name: str
    environment: SceneEnvironmentSpec
    support_surface: SceneSupportSurfaceSpec | None
    robot: SceneRobotSpec
    human: SceneHumanSpec
    cameras: list[SceneCameraSpec]
    motion: SceneMotionSpec = field(default_factory=SceneMotionSpec)
    render: SceneRenderSpec = field(default_factory=SceneRenderSpec)
    bindings: SceneBindingsSpec = field(default_factory=SceneBindingsSpec)
    metadata: dict[str, Any] = field(default_factory=dict)
    robots: list[SceneRobotSpec] = field(default_factory=list)
    robot_model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def support_surface_top_z(self) -> float:
        if self.support_surface is None:
            raise RuntimeError("Scene does not define a support_surface.")
        return float(self.support_surface.top_z)

    @property
    def bed_top_z(self) -> float:
        return self.support_surface_top_z

    def resolved_human_anchor(self) -> tuple[float, float, float]:
        anchor_x, anchor_y, anchor_z = self.human.anchor_pos
        if self.human.support_reference == "world":
            return (float(anchor_x), float(anchor_y), float(anchor_z))
        if self.human.support_reference == "ground_plane":
            return (
                float(anchor_x),
                float(anchor_y),
                float(self.environment.ground_plane_z_m + self.human.support_margin_m),
            )
        if self.human.support_reference == "support_surface_top":
            if self.support_surface is None:
                return (
                    float(anchor_x),
                    float(anchor_y),
                    float(self.environment.ground_plane_z_m + self.human.support_margin_m),
                )
            return (
                float(anchor_x),
                float(anchor_y),
                float(self.support_surface_top_z + self.human.support_margin_m),
            )
        raise ValueError(f"Unsupported human.support_reference: {self.human.support_reference}")

    def human_anchor_on_bed(self) -> tuple[float, float, float]:
        return self.resolved_human_anchor()

    @property
    def bed(self) -> SceneSupportSurfaceSpec:
        if self.support_surface is None:
            raise RuntimeError("Legacy 'bed' access requires support_surface to be defined.")
        return self.support_surface

    @property
    def ue_avatar(self) -> SceneCharacterVisualBindingSpec:
        return self.bindings.character_visual or SceneCharacterVisualBindingSpec()

    @property
    def scene_level_binding(self) -> SceneLevelAssetBindingSpec:
        return self.bindings.level or SceneLevelAssetBindingSpec()

    def iter_robot_specs(self) -> list[SceneRobotSpec]:
        return list(self.robots) if self.robots else [self.robot]


def _as_tuple(payload: Any, *, length: int, cast=float) -> tuple:
    values = tuple(cast(item) for item in payload)
    if len(values) != length:
        raise ValueError(f"Expected tuple of length {length}, got {payload!r}")
    return values


def _as_matrix4(payload: Any) -> list[list[float]]:
    rows = [[float(v) for v in row] for row in payload]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"Expected 4x4 matrix, got {payload!r}")
    return rows


def _parse_genesis_link_visual_urdf_rgba(raw: Any) -> dict[str, tuple[float, float, float, float]]:
    if raw is None or raw == {}:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("genesis_link_visual_urdf_rgba must map link names to [r,g,b,a]")
    out: dict[str, tuple[float, float, float, float]] = {}
    for key, val in raw.items():
        lk = str(key).strip()
        if not lk:
            continue
        out[lk] = _as_tuple(val, length=4)
    return out


def _load_camera_specs(payload: list[dict[str, Any]]) -> list[SceneCameraSpec]:
    cameras: list[SceneCameraSpec] = []
    for item in payload:
        cameras.append(
            SceneCameraSpec(
                name=str(item["name"]),
                res=_as_tuple(item.get("res", (1280, 720)), length=2, cast=int),
                pos=_as_tuple(item["pos"], length=3),
                lookat=_as_tuple(item["lookat"], length=3),
                up=_as_tuple(item.get("up", (0.0, 0.0, 1.0)), length=3),
                fov=float(item.get("fov", 45.0)),
                near=float(item.get("near", 0.05)),
                far=float(item.get("far", 100.0)),
                gui=bool(item.get("gui", False)),
                mount_entity=item.get("mount_entity"),
                mount_link=item.get("mount_link"),
                pose_rel=item.get("pose_rel"),
                follow_entity=bool(item.get("follow_entity", False)),
                roll_deg=float(item.get("roll_deg", 0.0)),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return cameras


def _quat_wxyz_to_xyzw(quat_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return (x, y, z, w)


def _load_robot_spec(robot_payload_raw: dict[str, Any]) -> SceneRobotSpec:
    robot_payload = resolve_robot_model_payload(dict(robot_payload_raw))
    probe_collision_payload = robot_payload.get("probe_collision")
    force_sensor_payload = robot_payload.get("force_sensor")
    tcp_control_payload = robot_payload.get("tcp_control")
    if robot_payload.get("base_quat_wxyz") is not None:
        base_quat_xyzw = _quat_wxyz_to_xyzw(_as_tuple(robot_payload["base_quat_wxyz"], length=4))
    elif robot_payload.get("base_quat_xyzw") is not None:
        base_quat_xyzw = _as_tuple(robot_payload["base_quat_xyzw"], length=4)
    else:
        base_quat_xyzw = None
    return SceneRobotSpec(
        name=str(robot_payload.get("name", robot_payload.get("instance_id", "robot_main"))),
        model_id=str(robot_payload.get("model_id", "")),
        instance_id=str(robot_payload.get("instance_id", robot_payload.get("name", ""))),
        urdf_path=str(robot_payload["urdf_path"]),
        base_pos=_as_tuple(robot_payload["base_pos"], length=3),
        base_quat_xyzw=base_quat_xyzw,
        joint_positions=[float(item) for item in robot_payload.get("joint_positions", [])],
        use_collision_geometry=bool(robot_payload.get("use_collision_geometry", True)),
        use_visual_mesh=bool(robot_payload.get("use_visual_mesh", True)),
        allow_collision_fallback=bool(robot_payload.get("allow_collision_fallback", False)),
        mesh_root=str(robot_payload.get("mesh_root", "")),
        visual_mesh_format=str(robot_payload.get("visual_mesh_format", "fbx")).strip().lower() or "fbx",
        ue_visual_asset_root=str(robot_payload.get("ue_visual_asset_root", "/Game/Bedlam/Generated/PandaVisual")),
        visual_mesh_scale=float(robot_payload.get("visual_mesh_scale", 1.0)),
        color=_as_tuple(robot_payload.get("color", (0.55, 0.55, 0.6, 1.0)), length=4),
        genesis_link_visual_urdf_rgba=_parse_genesis_link_visual_urdf_rgba(
            robot_payload.get("genesis_link_visual_urdf_rgba")
        ),
        capabilities=dict(robot_payload.get("capabilities", {})),
        controllers=dict(robot_payload.get("controllers", {})),
        asset_metadata=dict(robot_payload.get("asset_metadata", {})),
        metadata=dict(robot_payload.get("metadata", {})),
        probe_collision=(
            None
            if probe_collision_payload is None
            else SceneRobotProbeCollisionSpec(
                enabled=bool(probe_collision_payload.get("enabled", True)),
                link_name=str(probe_collision_payload["link_name"]),
                radius=float(probe_collision_payload.get("radius", 0.024)),
                length=float(probe_collision_payload.get("length", 0.14)),
                origin_xyz=(
                    _as_tuple(probe_collision_payload["origin_xyz"], length=3)
                    if probe_collision_payload.get("origin_xyz") is not None
                    else None
                ),
                origin_rpy=(
                    _as_tuple(probe_collision_payload["origin_rpy"], length=3)
                    if probe_collision_payload.get("origin_rpy") is not None
                    else None
                ),
                shape=str(probe_collision_payload.get("shape", "cylinder")).strip().lower(),
                mesh_filename=(
                    str(probe_collision_payload["mesh_filename"]).strip()
                    if probe_collision_payload.get("mesh_filename") not in (None, "")
                    else None
                ),
            )
        ),
        force_sensor=(
            None
            if force_sensor_payload is None
            else SceneRobotForceSensorSpec(
                mount_link=str(force_sensor_payload["mount_link"]),
                contact_link=str(force_sensor_payload.get("contact_link", force_sensor_payload["mount_link"])),
                link_T_sensor=_as_matrix4(force_sensor_payload["link_T_sensor"]),
                sensor_T_contact=_as_matrix4(force_sensor_payload["sensor_T_contact"]),
                tool_mass_kg=float(force_sensor_payload.get("tool_mass_kg", 0.0)),
                tool_com_sensor_m=_as_tuple(force_sensor_payload.get("tool_com_sensor_m", [0.0, 0.0, 0.0]), length=3),
                gravity_world_m_s2=_as_tuple(
                    force_sensor_payload.get("gravity_world_m_s2", [0.0, 0.0, -9.81]),
                    length=3,
                ),
                subtract_static_wrench_from_output=bool(
                    force_sensor_payload.get("subtract_static_wrench_from_output", False),
                ),
            )
        ),
        
        tcp_control=(
            None
            if tcp_control_payload is None
            else SceneRobotTcpControlSpec(
                link_name=(
                    None
                    if tcp_control_payload.get("link_name") in (None, "")
                    else str(tcp_control_payload["link_name"])
                ),
                local_point_m=(
                    None
                    if tcp_control_payload.get("local_point_m") is None
                    else _as_tuple(tcp_control_payload["local_point_m"], length=3)
                ),
            )
        ),
    )


def load_sync_scene_payload(path: Path) -> dict[str, Any]:
    raw_text = Path(path).read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"Scene config is empty: {path}")
    if yaml is not None:
        try:
            payload = yaml.safe_load(raw_text)
        except Exception:
            payload = json.loads(raw_text)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected top-level mapping in scene config: {path}")
        return payload
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected top-level mapping in scene config: {path}")
    return payload


def load_sync_scene_spec(path: Path) -> SyncSceneSpec:
    payload = load_sync_scene_payload(path)
    environment_payload = dict(payload["environment"])
    bindings_payload = dict(payload.get("bindings", {}))
    legacy_ue_avatar_payload = dict(payload.get("ue_avatar", {}))
    environment = SceneEnvironmentSpec(
        ground_plane_color=_as_tuple(environment_payload.get("ground_plane_color", (0.92, 0.92, 0.92, 1.0)), length=4),
        ground_plane_z_m=float(environment_payload.get("ground_plane_z_m", 0.0)),
    )
    support_surface_payload = payload.get("support_surface")
    bed_payload = payload.get("bed")
    if "robots" in payload:
        raw_robots = payload["robots"]
        if not isinstance(raw_robots, list) or not raw_robots:
            raise ValueError("scene.robots must be a non-empty list when provided.")
        robots = [_load_robot_spec(dict(item)) for item in raw_robots]
    else:
        robots = [_load_robot_spec(dict(payload["robot"]))]
    primary_robot = robots[0]
    human_payload = payload["human"]
    motion_payload = dict(payload.get("motion", {}))
    render_payload = dict(payload.get("render", {}))
    level_binding_payload = dict(bindings_payload.get("level", {}))
    character_binding_payload = dict(bindings_payload.get("character_visual", legacy_ue_avatar_payload))
    return SyncSceneSpec(
        name=str(payload["name"]),
        environment=environment,
        support_surface=(
            None
            if support_surface_payload is None and bed_payload is None
            else SceneSupportSurfaceSpec(
                name=str((support_surface_payload or bed_payload).get("name", "support_surface")),
                pos=_as_tuple((support_surface_payload or bed_payload)["pos"], length=3),
                size=_as_tuple((support_surface_payload or bed_payload)["size"], length=3),
                quat_xyzw=(
                    _as_tuple((support_surface_payload or bed_payload)["quat_xyzw"], length=4)
                    if (support_surface_payload or bed_payload).get("quat_xyzw") is not None
                    else None
                ),
                color=_as_tuple((support_surface_payload or bed_payload).get("color", (0.75, 0.75, 0.78, 1.0)), length=4),
                semantic_role=str((support_surface_payload or bed_payload).get("semantic_role", "bed")),
                spawn_in_genesis=bool((support_surface_payload or bed_payload).get("spawn_in_genesis", True)),
                spawn_in_ue=bool((support_surface_payload or bed_payload).get("spawn_in_ue", True)),
                contact_stiffness_n_per_m=float(
                    (support_surface_payload or bed_payload).get("contact_stiffness_n_per_m", 1.5e6)
                ),
            )
        ),
        robot=primary_robot,
        human=SceneHumanSpec(
            anchor_pos=_as_tuple(human_payload["anchor_pos"], length=3),
            support_margin_m=float(human_payload.get("support_margin_m", human_payload.get("bed_contact_margin_m", 0.015))),
            support_reference=str(human_payload.get("support_reference", "support_surface_top")),
            align_floor=bool(human_payload.get("align_floor", True)),
            display_vertical_sink_m=float(human_payload.get("display_vertical_sink_m", 0.0)),
            display_vertical_offset_m=float(human_payload.get("display_vertical_offset_m", 0.0)),
            display_pitch_forward_deg=float(human_payload.get("display_pitch_forward_deg", 0.0)),
            ue_root_offset_genesis_m=_as_tuple(human_payload.get("ue_root_offset_genesis_m", (0.0, 0.0, 0.0)), length=3),
        ),
        cameras=_load_camera_specs(payload["cameras"]),
        motion=SceneMotionSpec(
            source_id=str(motion_payload.get("source_id", "")),
            source_path=str(motion_payload.get("source_path", "")),
            sequence_npz_path=str(motion_payload.get("sequence_npz_path", "")),
            mesh_manifest_path=str(motion_payload.get("mesh_manifest_path", "")),
            fps=float(motion_payload.get("fps", 30.0)),
            frame_count=int(motion_payload.get("frame_count", 0)),
            start_frame=int(motion_payload.get("start_frame", 0)),
            frame_step=int(motion_payload.get("frame_step", 1)),
        ),
        render=SceneRenderSpec(
            fps=float(render_payload.get("fps", 8.0)),
            frame_limit=int(render_payload.get("frame_limit", 120)),
            genesis_backend=str(render_payload.get("genesis_backend", "cuda")),
            ue_frame_count=int(render_payload.get("ue_frame_count", 240)),
            ue_frame_step=int(render_payload.get("ue_frame_step", 4)),
            ue_render_now=bool(render_payload.get("ue_render_now", False)),
            ue_spawn_robot=bool(render_payload.get("ue_spawn_robot", True)),
            ue_spawn_human=bool(render_payload.get("ue_spawn_human", True)),
        ),
        bindings=SceneBindingsSpec(
            level=SceneLevelAssetBindingSpec(
                map_path=str(level_binding_payload.get("map_path", environment_payload.get("ue_map", "/Game/Bedlam/IBLMap"))),
                hdri_name=str(level_binding_payload.get("hdri_name", environment_payload.get("ue_hdri_name", "autumn_park"))),
                metadata=dict(level_binding_payload.get("metadata", {})),
            ),
            character_visual=SceneCharacterVisualBindingSpec(
                body_mode=str(character_binding_payload.get("body_mode", "retargeted_overlay")),
                body_name=str(character_binding_payload.get("body_name", "it_4375_M_2400")),
                texture_body=character_binding_payload.get("texture_body"),
                texture_clothing=character_binding_payload.get("texture_clothing"),
                texture_clothing_overlay=character_binding_payload.get("texture_clothing_overlay"),
                skeletal_mesh_path=str(
                    character_binding_payload.get(
                        "skeletal_mesh_path",
                        "/Engine/PS/Bedlam/SMPLX_LH_animations/it_4375_M/it_4375_M_2400",
                    )
                ),
                animation_asset_root=str(
                    character_binding_payload.get("animation_asset_root", "/Game/Bedlam/Generated/RetargetedAnimations")
                ),
                imported_fbx_root=str(
                    character_binding_payload.get("imported_fbx_root", "/Game/Bedlam/Generated/ImportedSMPLMotion")
                ),
                fallback_animation_path=str(character_binding_payload.get("fallback_animation_path", "")),
                hidden_material_path=str(
                    character_binding_payload.get(
                        "hidden_material_path",
                        "/Engine/PS/Bedlam/Core/Materials/M_SMPLX_Hidden.M_SMPLX_Hidden",
                    )
                ),
                fbx_global_scale=float(character_binding_payload.get("fbx_global_scale", 100.0)),
            ),
            metadata=dict(bindings_payload.get("metadata", {})),
        ),
        metadata=dict(payload.get("metadata", {})),
        robots=robots,
        robot_model_overrides={
            str(key): dict(value)
            for key, value in (payload.get("robot_model_overrides") or {}).items()
            if isinstance(value, dict)
        },
    )


def default_sync_scene_spec_path() -> Path:
    return project_paths(__file__).default_scene_spec_path

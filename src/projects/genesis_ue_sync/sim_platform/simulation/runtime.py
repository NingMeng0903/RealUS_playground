from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import site
import sys
from typing import Any

import numpy as np

from bridge.adapters.genesis import genesis_quat_wxyz_from_xyzw
from projects.genesis_ue_sync.sim_platform.core.messages import (
    CameraExtrinsics,
    CameraIntrinsics,
    MessageHeader,
    ObservationBundle,
    RewardSignal,
    RobotState,
    ScenarioState,
    SensorFrame,
    StepResult,
)
from projects.genesis_ue_sync.sim_platform.control.motion import MotionInterface
from projects.genesis_ue_sync.sim_platform.embodiments.profiles import EmbodimentProfile, SensorProfile
from projects.genesis_ue_sync.sim_platform.scenes.entity_commands import SceneEntityCommand, SceneEntityRegistry


def genesis_morph_quat_wxyz_from_xyzw(quat_xyzw: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Map layout (x, y, z, w) to Genesis ``gs.morphs.*.quat`` layout (w, x, y, z)."""
    return genesis_quat_wxyz_from_xyzw(quat_xyzw)


@dataclass
class GenesisRuntimeConfig:
    dt: float = 0.01
    substeps: int = 1
    backend: str = "cuda"
    precision: str = "32"
    coupler_kind: str = "none"
    fem_enabled: bool = False
    fem_use_implicit_solver: bool = True
    sap_enable_rigid_fem_contact: bool = True
    sap_enable_fem_self_tet_contact: bool = False
    sap_hydroelastic_stiffness: float = 1.0e8
    sap_point_contact_stiffness: float = 1.0e8
    show_viewer: bool = False
    show_fps: bool = False
    enable_collision: bool = True
    enable_self_collision: bool = False
    enable_adjacent_collision: bool = False
    # Passed to ``Scene.step(update_visualizer=..., refresh_visualizer=...)``. Debug meshes + pyrender can still fail;
    # Genesis ``scene.step`` catches viewer errors after ``_sim.step`` so physics is not rolled back.
    sim_update_visualizer: bool = True
    sim_refresh_visualizer: bool = True
    viewer_camera_pos: tuple[float, float, float] = (2.5, -2.0, 1.5)
    viewer_camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    viewer_camera_fov: float = 40.0
    ambient_light: tuple[float, float, float] = (0.25, 0.25, 0.25)
    plane_reflection: bool = True
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    # Genesis URDF morph defaults decimate to ~500 faces per link — fine for primitives, destroys arm meshes.
    urdf_visual_decimate: bool = False
    clock_domain: str = "genesis_sim"


@dataclass
class BoxEntityConfig:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    # World rotation as Hamilton quaternion (x, y, z, w); converted to Genesis (w, x, y, z) in ``add_box``.
    quat_xyzw: tuple[float, float, float, float] | None = None
    color: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    fixed: bool = True
    visualization: bool = True
    collision: bool = True
    rigid_friction: float | None = None
    rigid_contact_resistance: float | None = None
    rigid_coup_restitution: float | None = None
    rigid_coup_softness: float | None = None


@dataclass
class SphereEntityConfig:
    name: str
    pos: tuple[float, float, float]
    radius: float
    color: tuple[float, float, float, float] = (0.85, 0.85, 0.85, 1.0)
    fixed: bool = False
    visualization: bool = True
    collision: bool = True


@dataclass
class MeshEntityConfig:
    name: str
    file: Path
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat_xyzw: tuple[float, float, float, float] | None = None
    scale: float = 1.0
    color: tuple[float, float, float, float] = (0.75, 0.8, 0.95, 1.0)
    fixed: bool = True
    visualization: bool = True
    collision: bool = True

    def __post_init__(self) -> None:
        self.file = Path(self.file)


@dataclass
class FemBoxEntityConfig:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] | None = None
    color: tuple[float, float, float, float] = (0.70, 0.78, 0.86, 0.9)
    youngs_modulus: float = 3.0e4
    poissons_ratio: float = 0.45
    density: float = 250.0
    friction_mu: float = 0.6
    model: str = "stable_neohookean"
    maxvolume: float = -1.0
    nobisect: bool = True
    verbose: int = 0


@dataclass
class StaticCameraConfig:
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


class GenesisPlatformRuntime:
    """Reusable Genesis runtime for simulation, rendering, and policy I/O."""

    def __init__(self, config: GenesisRuntimeConfig | None = None) -> None:
        self.config = config or GenesisRuntimeConfig()
        self._gs = None
        self.scene = None
        self.entities: dict[str, Any] = {}
        self.embodiments: dict[str, EmbodimentProfile] = {}
        self.cameras: dict[str, Any] = {}
        self.camera_sensor_profiles: dict[str, SensorProfile] = {}
        self._camera_mounts: dict[str, StaticCameraConfig] = {}
        self._external_wrenches: dict[str, np.ndarray] = {}
        self.dynamic_entities = SceneEntityRegistry()
        self._injected_observation: ObservationBundle | None = None
        self._built = False
        self._sim_tick_observers: list[Any] = []

    def register_sim_tick_observer(self, callback: Any) -> None:
        """Callable(runtime: GenesisPlatformRuntime) invoked after each physics sub-step."""
        self._sim_tick_observers.append(callback)

    @staticmethod
    def _drop_user_site_packages() -> None:
        raw_user_site = site.getusersitepackages()
        if not raw_user_site:
            return
        user_site = str(Path(raw_user_site).expanduser().resolve())
        sys.path = [
            path
            for path in sys.path
            if not str(Path(path).expanduser().resolve()).startswith(user_site)
        ]
        for module_name in list(sys.modules):
            if module_name == "OpenGL" or module_name.startswith("OpenGL."):
                sys.modules.pop(module_name, None)

    def _require_genesis(self):
        if self._gs is None:
            try:
                self._drop_user_site_packages()
                import genesis as gs
            except ImportError as exc:
                raise ImportError("Genesis is required to use GenesisPlatformRuntime.") from exc
            self._gs = gs
        return self._gs

    def initialize(self) -> Any:
        gs = self._require_genesis()
        backend = gs.cuda if self.config.backend == "cuda" else gs.cpu
        if not bool(getattr(gs, "_initialized", False)):
            gs.init(backend=backend, precision=str(self.config.precision), logging_level="warning")
        else:
            current_backend = getattr(gs, "backend", None)
            if current_backend is not None and current_backend != backend:
                raise RuntimeError(
                    f"Genesis already initialized with backend={current_backend}, "
                    f"cannot reinitialize with backend={backend}."
                )
        scene_kwargs: dict[str, Any] = {
            "sim_options": gs.options.SimOptions(dt=self.config.dt, substeps=self.config.substeps),
            "rigid_options": gs.options.RigidOptions(
                gravity=self.config.gravity,
                enable_collision=self.config.enable_collision,
                enable_self_collision=self.config.enable_self_collision,
                enable_adjacent_collision=self.config.enable_adjacent_collision,
            ),
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=self.config.viewer_camera_pos,
                camera_lookat=self.config.viewer_camera_lookat,
                camera_fov=self.config.viewer_camera_fov,
                max_FPS=60,
            ),
            "vis_options": gs.options.VisOptions(
                show_world_frame=True,
                plane_reflection=self.config.plane_reflection,
                ambient_light=self.config.ambient_light,
            ),
            "show_viewer": self.config.show_viewer,
            "renderer": gs.renderers.Rasterizer(),
        }
        if bool(self.config.fem_enabled):
            scene_kwargs["fem_options"] = gs.options.FEMOptions(
                use_implicit_solver=bool(self.config.fem_use_implicit_solver),
            )
        coupler_kind = str(self.config.coupler_kind or "none").strip().lower()
        if coupler_kind == "sap":
            scene_kwargs["coupler_options"] = gs.options.SAPCouplerOptions(
                enable_rigid_fem_contact=bool(self.config.sap_enable_rigid_fem_contact),
                enable_fem_self_tet_contact=bool(self.config.sap_enable_fem_self_tet_contact),
                hydroelastic_stiffness=float(self.config.sap_hydroelastic_stiffness),
                point_contact_stiffness=float(self.config.sap_point_contact_stiffness),
            )
        elif coupler_kind == "ipc":
            scene_kwargs["coupler_options"] = gs.options.IPCCouplerOptions()
        profiling_options_cls = getattr(gs.options, "ProfilingOptions", None)
        if profiling_options_cls is not None:
            scene_kwargs["profiling_options"] = profiling_options_cls(show_FPS=bool(self.config.show_fps))
        else:
            scene_kwargs["show_FPS"] = bool(self.config.show_fps)
        self.scene = gs.Scene(
            **scene_kwargs,
        )
        if profiling_options_cls is not None and hasattr(self.scene, "profiling_options"):
            self.scene.profiling_options.show_FPS = bool(self.config.show_fps)
        return self.scene

    def add_ground_plane(
        self,
        *,
        name: str = "ground",
        color: tuple[float, float, float, float] = (0.92, 0.92, 0.92, 1.0),
    ) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        entity = self.scene.add_entity(
            gs.morphs.Plane(),
            surface=gs.surfaces.Default(color=color),
            name=name,
        )
        self.entities[name] = entity
        return entity

    def add_box(self, config: BoxEntityConfig) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        kwargs: dict[str, Any] = {
            "pos": config.pos,
            "size": config.size,
            "fixed": config.fixed,
            "visualization": config.visualization,
            "collision": config.collision,
        }
        if config.quat_xyzw is not None:
            kwargs["quat"] = genesis_morph_quat_wxyz_from_xyzw(config.quat_xyzw)
        mat_kwargs: dict[str, Any] = {}
        if config.rigid_friction is not None:
            mat_kwargs["friction"] = float(config.rigid_friction)
        if config.rigid_contact_resistance is not None:
            mat_kwargs["contact_resistance"] = float(config.rigid_contact_resistance)
        if config.rigid_coup_restitution is not None:
            mat_kwargs["coup_restitution"] = float(config.rigid_coup_restitution)
        if config.rigid_coup_softness is not None:
            mat_kwargs["coup_softness"] = float(config.rigid_coup_softness)
        material = gs.materials.Rigid(**mat_kwargs) if mat_kwargs else gs.materials.Rigid()
        entity = self.scene.add_entity(
            gs.morphs.Box(**kwargs),
            material=material,
            surface=gs.surfaces.Default(color=config.color),
            name=config.name,
        )
        self.entities[config.name] = entity
        return entity

    def add_sphere(self, config: SphereEntityConfig) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        entity = self.scene.add_entity(
            gs.morphs.Sphere(
                pos=config.pos,
                radius=float(config.radius),
                fixed=config.fixed,
                visualization=config.visualization,
                collision=config.collision,
            ),
            surface=gs.surfaces.Default(color=config.color),
            name=config.name,
        )
        self.entities[config.name] = entity
        return entity

    def add_mesh_entity(self, config: MeshEntityConfig) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        kwargs: dict[str, Any] = {
            "file": str(config.file),
            "pos": config.pos,
            "scale": config.scale,
            "fixed": config.fixed,
            "visualization": bool(config.visualization),
            "collision": bool(config.collision),
        }
        if config.quat_xyzw is not None:
            kwargs["quat"] = genesis_morph_quat_wxyz_from_xyzw(config.quat_xyzw)
        entity = self.scene.add_entity(
            gs.morphs.Mesh(**kwargs),
            surface=gs.surfaces.Default(color=config.color),
            name=config.name,
        )
        self.entities[config.name] = entity
        return entity

    def apply_dynamic_entity_command(self, command: SceneEntityCommand | dict[str, Any]) -> dict[str, Any]:
        cmd = command if isinstance(command, SceneEntityCommand) else SceneEntityCommand.from_mapping(command)
        result = self.dynamic_entities.apply(cmd)
        if cmd.op in {"delete", "remove", "rename"}:
            self.entities.pop(cmd.entity_id, None)
            return result
        if cmd.entity_id in self.entities:
            return result

        pose = dict(cmd.pose or {})
        payload = dict(cmd.payload or {})
        pos = tuple(float(v) for v in pose.get("pos_m", pose.get("pos", (0.0, 0.0, 0.0))))
        color = tuple(float(v) for v in payload.get("color_rgba", (0.2, 0.6, 1.0, 0.35)))
        kind = str(cmd.entity_type or payload.get("entity_type", "")).strip().lower()
        if kind == "box":
            size = tuple(float(v) for v in payload.get("size_m", payload.get("size", (0.05, 0.05, 0.05))))
            self.add_box(
                BoxEntityConfig(
                    name=cmd.entity_id,
                    pos=pos,  # type: ignore[arg-type]
                    size=size,  # type: ignore[arg-type]
                    quat_xyzw=pose.get("quat_xyzw"),
                    color=color,  # type: ignore[arg-type]
                    fixed=bool(payload.get("fixed", True)),
                    visualization=bool(payload.get("visualization", True)),
                    collision=bool(payload.get("collision", False)),
                )
            )
        elif kind == "sphere":
            self.add_sphere(
                SphereEntityConfig(
                    name=cmd.entity_id,
                    pos=pos,  # type: ignore[arg-type]
                    radius=float(payload.get("radius_m", payload.get("radius", 0.05))),
                    color=color,  # type: ignore[arg-type]
                    fixed=bool(payload.get("fixed", True)),
                    visualization=bool(payload.get("visualization", True)),
                    collision=bool(payload.get("collision", False)),
                )
            )
        elif kind == "mesh" and payload.get("file"):
            self.add_mesh_entity(
                MeshEntityConfig(
                    name=cmd.entity_id,
                    file=Path(str(payload["file"])),
                    pos=pos,  # type: ignore[arg-type]
                    quat_xyzw=pose.get("quat_xyzw"),
                    scale=float(payload.get("scale", 1.0)),
                    color=color,  # type: ignore[arg-type]
                    fixed=bool(payload.get("fixed", True)),
                    visualization=bool(payload.get("visualization", True)),
                    collision=bool(payload.get("collision", False)),
                )
            )
        return result

    def add_fem_box(self, config: FemBoxEntityConfig) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        kwargs: dict[str, Any] = {
            "pos": config.pos,
            "size": config.size,
            "nobisect": bool(config.nobisect),
            "verbose": int(config.verbose),
        }
        if config.quat_xyzw is not None:
            kwargs["quat"] = genesis_morph_quat_wxyz_from_xyzw(config.quat_xyzw)
        if float(config.maxvolume) > 0.0:
            kwargs["maxvolume"] = float(config.maxvolume)
        entity = self.scene.add_entity(
            gs.morphs.Box(**kwargs),
            material=gs.materials.FEM.Elastic(
                E=float(config.youngs_modulus),
                nu=float(config.poissons_ratio),
                rho=float(config.density),
                friction_mu=float(config.friction_mu),
                model=str(config.model),
            ),
            surface=gs.surfaces.Default(color=config.color),
            name=config.name,
        )
        self.entities[config.name] = entity
        return entity

    def add_robot(
        self,
        embodiment: EmbodimentProfile,
        *,
        name: str | None = None,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: tuple[float, float, float, float] | None = None,
        visual_surface_rgba: tuple[float, float, float, float] | None = None,
    ) -> Any:
        gs = self._require_genesis()
        if self.scene is None:
            self.initialize()
        robot_name = name or embodiment.robot.name
        morph_path = embodiment.robot.genesis_morph_path or embodiment.robot.urdf_path
        morph_format = str(embodiment.robot.genesis_morph_format or "urdf").strip().lower()
        kwargs: dict[str, Any] = {
            "file": str(morph_path),
            "pos": pos,
            "fixed": embodiment.robot.fixed_base,
            "merge_fixed_links": False,
            "decimate": bool(self.config.urdf_visual_decimate),
        }
        if quat_xyzw is not None:
            kwargs["quat"] = genesis_morph_quat_wxyz_from_xyzw(quat_xyzw)
        if morph_format == "mjcf":
            kwargs.pop("decimate", None)
            kwargs.pop("merge_fixed_links", None)
            kwargs.pop("fixed", None)
        if morph_format != "mjcf" and embodiment.metadata.get("genesis_urdf_prioritize_material"):
            kwargs["prioritize_urdf_material"] = True
        surface = gs.surfaces.Default(color=visual_surface_rgba) if visual_surface_rgba is not None else gs.surfaces.Default()
        if morph_format == "mjcf":
            morph = gs.morphs.MJCF(**kwargs)
        else:
            morph = gs.morphs.URDF(**kwargs)
        entity = self.scene.add_entity(
            morph=morph,
            material=gs.materials.Rigid(),
            surface=surface,
            name=robot_name,
        )
        self.entities[robot_name] = entity
        self.embodiments[robot_name] = embodiment
        self._external_wrenches[robot_name] = np.zeros(6, dtype=np.float32)
        return entity

    def add_articulated_entity(
        self,
        embodiment: EmbodimentProfile,
        *,
        name: str | None = None,
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        quat_xyzw: tuple[float, float, float, float] | None = None,
        visual_surface_rgba: tuple[float, float, float, float] | None = None,
    ) -> Any:
        return self.add_robot(
            embodiment,
            name=name,
            pos=pos,
            quat_xyzw=quat_xyzw,
            visual_surface_rgba=visual_surface_rgba,
        )

    def add_camera(self, config: StaticCameraConfig, *, sensor_profile: SensorProfile | None = None) -> Any:
        if self.scene is None:
            self.initialize()
        fov_deg = float(config.fov)
        metadata = dict(getattr(config, "metadata", {}) or {})
        fov_axis = str(metadata.get("fov_axis", "horizontal" if metadata.get("synced_from_calibration") else "vertical")).strip().lower()
        if fov_axis == "horizontal":
            width, height = (int(config.res[0]), int(config.res[1]))
            if width > 0 and height > 0:
                fov_deg = math.degrees(
                    2.0 * math.atan(math.tan(math.radians(float(config.fov)) * 0.5) * float(height) / float(width))
                )
        camera = self.scene.add_camera(
            res=config.res,
            pos=config.pos,
            lookat=config.lookat,
            up=config.up,
            fov=fov_deg,
            near=config.near,
            far=config.far,
            GUI=config.gui,
        )
        self.cameras[config.name] = camera
        self._camera_mounts[config.name] = config
        if sensor_profile is not None:
            self.camera_sensor_profiles[config.name] = sensor_profile
        return camera

    def add_camera_rig(
        self,
        *,
        camera_configs: list[StaticCameraConfig],
        sensor_profiles: dict[str, SensorProfile] | None = None,
    ) -> None:
        for camera_config in camera_configs:
            sensor_profile = sensor_profiles.get(camera_config.name) if sensor_profiles is not None else None
            self.add_camera(camera_config, sensor_profile=sensor_profile)

    def build(self, *, n_envs: int = 0, env_spacing: tuple[float, float] = (2.0, 2.0)) -> None:
        if self.scene is None:
            self.initialize()
        self.scene.build(n_envs=n_envs, env_spacing=env_spacing)
        self._built = True
        self._apply_camera_mounts()

    def apply_rm75_6f_arm_pd_gains(self, entity_name: str) -> None:
        """PD + torque clamps aligned with vendor RM75-6F ``effort`` limits (Genesis 7-DOF URDF ordering)."""

        if not self._built:
            raise RuntimeError("GenesisPlatformRuntime.build() must be called first.")
        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        entity = self.entities[entity_name]
        n = int(self._to_numpy(entity.get_dofs_position()).reshape(-1).size)
        if n != 7:
            return
        effort = np.array([60.0, 60.0, 30.0, 30.0, 10.0, 10.0, 10.0], dtype=np.float32)
        kp = np.array([3400.0, 3400.0, 2600.0, 2600.0, 1100.0, 850.0, 850.0], dtype=np.float32)
        kv = np.array([380.0, 380.0, 290.0, 290.0, 125.0, 95.0, 95.0], dtype=np.float32)
        entity.set_dofs_kp(kp)
        entity.set_dofs_kv(kv)
        entity.set_dofs_force_range(-effort, effort)

    def apply_franka_like_arm_pd_gains(
        self,
        entity_name: str,
        *,
        effort_limit: float = 87.0,
    ) -> None:
        """Set kp/kv/torque limits similar to Genesis Franka integration tests (7-DOF arm only).

        Default URDF import often leaves PD gains too low to hold pose under gravity, so the arm sags.
        """
        if not self._built:
            raise RuntimeError("GenesisPlatformRuntime.build() must be called first.")
        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        entity = self.entities[entity_name]
        n = int(self._to_numpy(entity.get_dofs_position()).reshape(-1).size)
        if n != 7:
            return
        kp = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000], dtype=np.float32)
        kv = np.array([450, 450, 350, 350, 200, 200, 200], dtype=np.float32)
        entity.set_dofs_kp(kp)
        entity.set_dofs_kv(kv)
        entity.set_dofs_force_range(
            np.full(7, -effort_limit, dtype=np.float32),
            np.full(7, effort_limit, dtype=np.float32),
        )

    def _apply_camera_mounts(self) -> None:
        for camera_name, mount in self._camera_mounts.items():
            if mount.mount_entity is None:
                continue
            if mount.mount_entity not in self.entities:
                raise KeyError(f"Unknown mount entity for camera '{camera_name}': {mount.mount_entity}")
            entity = self.entities[mount.mount_entity]
            camera = self.cameras[camera_name]
            if mount.follow_entity:
                camera.follow_entity(entity)
                continue
            if mount.mount_link is None:
                raise ValueError(f"Camera '{camera_name}' mount_link is required for link attachment.")
            link = entity.get_link(mount.mount_link)
            pose_rel = np.asarray(mount.pose_rel if mount.pose_rel is not None else np.eye(4), dtype=np.float32)
            camera.attach(link, pose_rel)
            camera.move_to_attach()

    def reset(self) -> None:
        if not self._built:
            raise RuntimeError("GenesisPlatformRuntime.build() must be called before reset().")
        self.scene.reset()
        for camera_name, mount in self._camera_mounts.items():
            if mount.mount_entity is not None and not mount.follow_entity:
                self.cameras[camera_name].move_to_attach()

    def close(self) -> None:
        if self.scene is None:
            return
        try:
            self.scene.destroy()
        except Exception:
            pass
        self.scene = None
        self.entities = {}
        self.embodiments = {}
        self.cameras = {}
        self.camera_sensor_profiles = {}
        self._camera_mounts = {}
        self._external_wrenches = {}
        self._injected_observation = None
        self._built = False
        self._sim_tick_observers = []

    def step(self, commands: dict[str, list[float]] | None = None, *, n: int = 1) -> None:
        if not self._built:
            raise RuntimeError("GenesisPlatformRuntime.build() must be called before step().")
        if commands:
            for entity_name, joint_positions in commands.items():
                self.set_robot_joint_positions(entity_name, joint_positions)
        for _ in range(n):
            self.scene.step(
                update_visualizer=bool(self.config.sim_update_visualizer),
                refresh_visualizer=bool(self.config.sim_refresh_visualizer),
            )
            for observer in self._sim_tick_observers:
                try:
                    observer(self)
                except Exception:
                    pass
            for camera_name, mount in self._camera_mounts.items():
                if mount.mount_entity is not None and not mount.follow_entity:
                    self.cameras[camera_name].move_to_attach()

    def set_robot_joint_positions(self, entity_name: str, joint_positions: list[float] | np.ndarray) -> None:
        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        entity = self.entities[entity_name]
        entity.set_dofs_position(np.asarray(joint_positions, dtype=np.float32))

    def apply_dof_forces(self, entity_name: str, tau: list[float] | np.ndarray) -> None:
        """Inject generalized forces into Genesis dofs (one-shot per step)."""

        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        entity = self.entities[entity_name]
        values = np.asarray(tau, dtype=np.float32).reshape(-1)
        entity.control_dofs_force(values)

    def get_robot_joint_positions(self, entity_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        return self._to_numpy(entity.get_dofs_position()).astype(np.float32).reshape(-1)

    def get_robot_joint_velocities(self, entity_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "get_dofs_velocity"):
            return np.zeros_like(self.get_robot_joint_positions(entity_name))
        return self._to_numpy(entity.get_dofs_velocity()).astype(np.float32).reshape(-1)

    def get_robot_joint_efforts(self, entity_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "get_dofs_force"):
            return np.zeros_like(self.get_robot_joint_positions(entity_name))
        return self._to_numpy(entity.get_dofs_force()).astype(np.float32).reshape(-1)

    def get_robot_control_efforts(self, entity_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "get_dofs_control_force"):
            return np.zeros_like(self.get_robot_joint_positions(entity_name))
        return self._to_numpy(entity.get_dofs_control_force()).astype(np.float32).reshape(-1)

    def get_link_pose(self, entity_name: str, link_name: str) -> np.ndarray:
        link = self._get_link(entity_name, link_name)
        pos = self._to_numpy(link.get_pos()).astype(np.float32).reshape(-1)
        quat = self._to_numpy(link.get_quat()).astype(np.float32).reshape(-1)
        return np.concatenate([pos, quat], dtype=np.float32)

    def get_link_twist(self, entity_name: str, link_name: str) -> np.ndarray:
        link = self._get_link(entity_name, link_name)
        linear = self._to_numpy(link.get_vel()).astype(np.float32).reshape(-1)
        angular = self._to_numpy(link.get_ang()).astype(np.float32).reshape(-1)
        return np.concatenate([linear, angular], dtype=np.float32)

    def get_tcp_pose(self, entity_name: str) -> np.ndarray:
        embodiment = self._require_embodiment(entity_name)
        return self.get_link_pose(entity_name, embodiment.end_effector.tcp_frame)

    def get_tcp_twist(self, entity_name: str) -> np.ndarray:
        embodiment = self._require_embodiment(entity_name)
        return self.get_link_twist(entity_name, embodiment.end_effector.tcp_frame)

    def get_robot_jacobian(
        self,
        entity_name: str,
        *,
        link_name: str | None = None,
        local_point: list[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        embodiment = self._require_embodiment(entity_name)
        target_link = entity.get_link(link_name or embodiment.end_effector.tcp_frame)
        local_point_arr = None if local_point is None else np.asarray(local_point, dtype=np.float32).reshape(3)
        return self._to_numpy(entity.get_jacobian(target_link, local_point=local_point_arr)).astype(np.float32)

    def get_robot_mass_matrix(self, entity_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        return self._to_numpy(entity.get_mass_mat()).astype(np.float32)

    def get_link_contact_force(self, entity_name: str, link_name: str) -> np.ndarray:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "get_links_net_contact_force"):
            return np.zeros(3, dtype=np.float32)
        link = entity.get_link(link_name)
        contact_forces = self._to_numpy(entity.get_links_net_contact_force()).astype(np.float32)
        return np.asarray(contact_forces[link.idx_local], dtype=np.float32).reshape(3)

    def get_entity_contacts(
        self,
        entity_name: str,
        *,
        with_entity_name: str | None = None,
        exclude_self_contact: bool = False,
    ) -> list[dict[str, Any]]:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "get_contacts"):
            return []
        other_entity = None if with_entity_name is None else self.entities.get(with_entity_name)
        raw = entity.get_contacts(with_entity=other_entity, exclude_self_contact=bool(exclude_self_contact))
        if not isinstance(raw, dict) or not raw:
            return []

        def _flat(value: Any) -> np.ndarray:
            arr = self._to_numpy(value)
            if arr.ndim >= 2 and arr.shape[0] == 1:
                arr = arr[0]
            return np.asarray(arr)

        valid_mask = raw.get("valid_mask")
        if valid_mask is None:
            penetration = _flat(raw.get("penetration", np.zeros((0,), dtype=np.float32))).reshape(-1)
            valid = np.ones((penetration.shape[0],), dtype=bool)
        else:
            valid = _flat(valid_mask).reshape(-1).astype(bool)

        link_name_by_idx: dict[int, str] = {}
        for target_name in (entity_name, with_entity_name):
            if target_name is None or target_name not in self.entities:
                continue
            target_entity = self.entities[target_name]
            for link in getattr(target_entity, "links", []):
                link_idx = getattr(link, "idx", None)
                link_name = getattr(link, "name", None)
                if link_idx is None or link_name is None:
                    continue
                link_name_by_idx[int(link_idx)] = str(link_name)

        link_a = _flat(raw.get("link_a", np.zeros((0,), dtype=np.int32))).reshape(-1)
        link_b = _flat(raw.get("link_b", np.zeros((0,), dtype=np.int32))).reshape(-1)
        geom_a = _flat(raw.get("geom_a", np.zeros((0,), dtype=np.int32))).reshape(-1)
        geom_b = _flat(raw.get("geom_b", np.zeros((0,), dtype=np.int32))).reshape(-1)
        penetration = _flat(raw.get("penetration", np.zeros((0,), dtype=np.float32))).reshape(-1)
        position = _flat(raw.get("position", np.zeros((0, 3), dtype=np.float32))).reshape(-1, 3)
        normal = _flat(raw.get("normal", np.zeros((0, 3), dtype=np.float32))).reshape(-1, 3)
        force_a = _flat(raw.get("force_a", np.zeros((0, 3), dtype=np.float32))).reshape(-1, 3)
        force_b = _flat(raw.get("force_b", np.zeros((0, 3), dtype=np.float32))).reshape(-1, 3)
        count = min(
            len(valid),
            link_a.shape[0],
            link_b.shape[0],
            geom_a.shape[0],
            geom_b.shape[0],
            penetration.shape[0],
            position.shape[0],
            normal.shape[0],
            force_a.shape[0],
            force_b.shape[0],
        )
        contacts: list[dict[str, Any]] = []
        for idx in range(count):
            if not bool(valid[idx]):
                continue
            contacts.append(
                {
                    "entity_name": str(entity_name),
                    "with_entity_name": None if with_entity_name is None else str(with_entity_name),
                    "link_a_idx": int(link_a[idx]),
                    "link_b_idx": int(link_b[idx]),
                    "link_a_name": link_name_by_idx.get(int(link_a[idx])),
                    "link_b_name": link_name_by_idx.get(int(link_b[idx])),
                    "geom_a_idx": int(geom_a[idx]),
                    "geom_b_idx": int(geom_b[idx]),
                    "position": np.asarray(position[idx], dtype=np.float32).reshape(3),
                    "normal": np.asarray(normal[idx], dtype=np.float32).reshape(3),
                    "penetration_depth_m": float(penetration[idx]),
                    "force_a": np.asarray(force_a[idx], dtype=np.float32).reshape(3),
                    "force_b": np.asarray(force_b[idx], dtype=np.float32).reshape(3),
                }
            )
        return contacts

    def get_wrench(
        self,
        entity_name: str,
        *,
        source: str = "external_injected",
        link_name: str | None = None,
    ) -> np.ndarray:
        if source == "external_injected":
            return self.get_external_wrench(entity_name).astype(np.float32).reshape(6)
        if source == "sim_contact":
            embodiment = self._require_embodiment(entity_name)
            target_link = link_name or embodiment.end_effector.tcp_frame
            force = self.get_link_contact_force(entity_name, target_link)
            return np.concatenate([force, np.zeros(3, dtype=np.float32)], dtype=np.float32)
        raise ValueError(f"Unsupported wrench source: {source}")

    def get_motion_interface(self, entity_name: str) -> MotionInterface:
        return MotionInterface(self, entity_name)

    def set_external_wrench(self, entity_name: str, wrench: list[float] | np.ndarray) -> None:
        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        self._external_wrenches[entity_name] = np.asarray(wrench, dtype=np.float32).reshape(-1)

    def set_robot_gravity_compensation(self, entity_name: str, value: float) -> None:
        entity = self._require_robot_entity(entity_name)
        if not hasattr(entity, "material") or not hasattr(entity.material, "gravity_compensation"):
            raise AttributeError(f"Entity '{entity_name}' does not expose gravity_compensation.")
        entity.material.gravity_compensation = float(value)

    def get_external_wrench(self, entity_name: str) -> np.ndarray:
        if entity_name not in self._external_wrenches:
            return np.zeros(6, dtype=np.float32)
        return self._external_wrenches[entity_name]

    def inject_observation(self, observation: ObservationBundle) -> None:
        self._injected_observation = observation

    def get_robot_state(self, entity_name: str) -> RobotState:
        embodiment = self._require_embodiment(entity_name)
        joint_position = self.get_robot_joint_positions(entity_name).tolist()
        joint_velocity = self.get_robot_joint_velocities(entity_name).tolist()
        joint_effort = self.get_robot_joint_efforts(entity_name).tolist()
        tcp_pose = self.get_tcp_pose(entity_name).tolist()
        tcp_twist = self.get_tcp_twist(entity_name).tolist()
        eef_wrench = self.get_external_wrench(entity_name).astype(np.float32).reshape(-1).tolist()
        return RobotState(
            robot_name=entity_name,
            joint_names=embodiment.robot.joint_names,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            joint_effort=joint_effort,
            eef_pose=tcp_pose,
            tcp_pose=tcp_pose,
            eef_twist=tcp_twist,
            eef_wrench=eef_wrench,
            control_mode=embodiment.robot.default_control_space,
            robot_mode="simulated",
            metadata={
                "tcp_frame": embodiment.end_effector.tcp_frame,
                "available_wrench_sources": ["external_injected", "sim_contact"],
            },
        )

    def _require_robot_entity(self, entity_name: str) -> Any:
        if entity_name not in self.entities:
            raise KeyError(f"Unknown robot entity: {entity_name}")
        return self.entities[entity_name]

    def _require_embodiment(self, entity_name: str) -> EmbodimentProfile:
        embodiment = self.embodiments.get(entity_name)
        if embodiment is None:
            raise KeyError(f"No embodiment registered for robot entity: {entity_name}")
        return embodiment

    def _get_link(self, entity_name: str, link_name: str) -> Any:
        entity = self._require_robot_entity(entity_name)
        return entity.get_link(link_name)

    def render_camera(
        self,
        camera_name: str,
        *,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        normal: bool = False,
        force_render: bool = False,
    ) -> dict[str, Any]:
        if camera_name not in self.cameras:
            raise KeyError(f"Unknown camera: {camera_name}")
        camera = self.cameras[camera_name]
        rgb_arr, depth_arr, seg_arr, normal_arr = camera.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            normal=normal,
            force_render=force_render,
        )
        payload = {
            "rgb": rgb_arr,
            "depth": depth_arr,
            "segmentation": seg_arr,
            "normal": normal_arr,
        }
        return {
            key: self._to_numpy(value)
            for key, value in payload.items()
            if value is not None
        }

    def _to_numpy(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def render_all_cameras(
        self,
        *,
        modalities: tuple[str, ...] = ("rgb",),
        force_render: bool = False,
    ) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for camera_name in self.cameras:
            outputs[camera_name] = self.render_camera(
                camera_name,
                rgb="rgb" in modalities,
                depth="depth" in modalities,
                segmentation="segmentation" in modalities,
                normal="normal" in modalities,
                force_render=force_render,
            )
        return outputs

    def capture_sensor_frames(
        self,
        *,
        camera_names: list[str] | None = None,
        include_modalities: tuple[str, ...] = ("rgb",),
        force_render: bool = False,
    ) -> dict[str, SensorFrame]:
        selected = camera_names or list(self.cameras.keys())
        sensor_frames: dict[str, SensorFrame] = {}
        for camera_name in selected:
            rendered = self.render_camera(
                camera_name,
                rgb="rgb" in include_modalities,
                depth="depth" in include_modalities,
                segmentation="segmentation" in include_modalities,
                normal="normal" in include_modalities,
                force_render=force_render,
            )
            sensor_profile = self.camera_sensor_profiles.get(camera_name)
            if "rgb" in rendered:
                sensor_frames[camera_name] = SensorFrame(
                    sensor_name=camera_name,
                    modality="rgb",
                    frame_id=sensor_profile.frame_id if sensor_profile is not None else f"camera_frame/{camera_name}",
                    timestamp_ns=MessageHeader.create("SensorFrame").timestamp_ns,
                    encoding=sensor_profile.encoding if sensor_profile is not None else "rgb8",
                    transport="inline",
                    shape=tuple(rendered["rgb"].shape),
                    data=rendered["rgb"],
                    intrinsics=self._camera_intrinsics(camera_name, sensor_profile),
                    extrinsics=self._camera_extrinsics(camera_name),
                    metadata={"render_modalities": list(include_modalities)},
                )
        return sensor_frames

    def _camera_intrinsics(
        self,
        camera_name: str,
        sensor_profile: SensorProfile | None,
    ) -> CameraIntrinsics | None:
        if sensor_profile is not None and sensor_profile.intrinsics is not None:
            return sensor_profile.intrinsics
        camera = self.cameras[camera_name]
        intrinsics = self._to_numpy(camera.intrinsics)
        if intrinsics.shape != (3, 3):
            return None
        width, height = sensor_profile.resolution if sensor_profile and sensor_profile.resolution else (0, 0)
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=float(intrinsics[0, 0]),
            fy=float(intrinsics[1, 1]),
            cx=float(intrinsics[0, 2]),
            cy=float(intrinsics[1, 2]),
            skew=float(intrinsics[0, 1]),
        )

    def _camera_extrinsics(self, camera_name: str) -> CameraExtrinsics:
        camera = self.cameras[camera_name]
        extrinsics = self._to_numpy(camera.extrinsics)
        world_from_camera = np.linalg.inv(extrinsics)
        return CameraExtrinsics(
            world_from_camera=world_from_camera.tolist(),
            camera_from_world=extrinsics.tolist(),
            parent_frame="world",
            child_frame=f"camera_frame/{camera_name}",
        )

    def get_state(self) -> dict[str, Any]:
        robot_states = {
            entity_name: self.get_robot_state(entity_name).to_dict()
            for entity_name in self.embodiments
        }
        return {
            "robots": robot_states,
            "cameras": sorted(self.cameras.keys()),
            "entities": sorted(self.entities.keys()),
            "has_injected_observation": self._injected_observation is not None,
        }

    def make_step_result(
        self,
        *,
        robot_entity: str | None = None,
        scenario_state: ScenarioState | None = None,
        reward: RewardSignal | None = None,
        language: dict[str, Any] | None = None,
    ) -> StepResult:
        robot_state = self.get_robot_state(robot_entity) if robot_entity is not None else None
        observation = ObservationBundle(
            header=MessageHeader.create("ObservationBundle", source_id="genesis_runtime"),
            robot_state=robot_state,
            sensors=self.capture_sensor_frames(),
            scenario_state=scenario_state,
            language=language or {},
            meta={"runtime": "genesis"},
        )
        return StepResult(
            observation=observation,
            reward=reward or RewardSignal(),
            info=self.get_state(),
        )

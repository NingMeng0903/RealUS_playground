from __future__ import annotations

from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.sim_platform.scenes.common_scene import SyncSceneSpec, load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import add_robots_to_runtime, init_robots_after_build
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


def build_genesis_runtime_for_scene(
    *,
    scene_spec: SyncSceneSpec | None = None,
    scene_spec_path: str | Path | None = None,
    backend: str = "cuda",
    show_viewer: bool = False,
    physics_dt: float = 0.01,
    enable_self_collision: bool = True,
    enable_adjacent_collision: bool = True,
) -> tuple[GenesisPlatformRuntime, str]:
    if scene_spec is None:
        if scene_spec_path is None:
            raise ValueError("Provide scene_spec or scene_spec_path.")
        scene_spec = load_sync_scene_spec(Path(scene_spec_path))
    registry = RobotRegistry()
    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend,
            show_viewer=show_viewer,
            show_fps=False,
            enable_collision=True,
            enable_self_collision=bool(enable_self_collision),
            enable_adjacent_collision=bool(enable_adjacent_collision),
            dt=float(physics_dt),
        )
    )
    runtime.initialize()
    runtime.add_ground_plane(color=scene_spec.environment.ground_plane_color)
    if scene_spec.support_surface is not None and scene_spec.support_surface.spawn_in_genesis:
        runtime.add_box(
            BoxEntityConfig(
                name=scene_spec.support_surface.name,
                pos=scene_spec.support_surface.pos,
                size=scene_spec.support_surface.size,
                quat_xyzw=scene_spec.support_surface.quat_xyzw,
                color=scene_spec.support_surface.color,
            )
        )

    robot_specs = list(scene_spec.iter_robot_specs())
    robot_names = add_robots_to_runtime(
        runtime,
        robot_specs,
        registry,
        enable_collision=bool(runtime.config.enable_collision),
    )

    for camera in scene_spec.cameras:
        runtime.add_camera(camera)
    runtime.build()

    spawned = init_robots_after_build(runtime, registry, robot_specs, robot_names)
    return runtime, spawned.primary_name


def build_genesis_runtime_from_scene_file(scene_spec_path: str | Path, **kwargs: Any) -> tuple[GenesisPlatformRuntime, str]:
    return build_genesis_runtime_for_scene(scene_spec_path=Path(scene_spec_path), **kwargs)

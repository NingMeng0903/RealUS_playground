from __future__ import annotations

from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.config import GenesisOverlayConfig, MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import MultiviewTrackFrame
from projects.genesis_ue_sync.multiview_realtime.viz.track_skeleton_drawer import TrackSkeletonDrawer
from projects.genesis_ue_sync.sim_platform.scenes import load_sync_scene_spec
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import add_robots_to_runtime, init_robots_after_build
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import select_robot_specs
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
)


class GenesisTrackOverlay:
    """Minimal Genesis scene (bed optional) + orange tracked Body25 3D skeleton."""

    def __init__(self, config: MultiviewRealtimeConfig) -> None:
        self.config = config
        self.genesis_cfg = config.genesis
        self._runtime: GenesisPlatformRuntime | None = None
        self._drawer: TrackSkeletonDrawer | None = None
        self._live_frame_counter = 0

    def setup(self) -> GenesisPlatformRuntime:
        if self.config.scene_spec_path is None:
            raise ValueError("scene_spec_path is required for Genesis overlay.")
        scene_spec = load_sync_scene_spec(self.config.scene_spec_path)
        runtime = GenesisPlatformRuntime(
            GenesisRuntimeConfig(
                backend=str(self.genesis_cfg.backend),
                show_viewer=bool(self.genesis_cfg.show_viewer),
                show_fps=bool(self.genesis_cfg.show_fps),
                enable_collision=False,
                enable_self_collision=False,
                dt=0.01,
            )
        )
        runtime.initialize()
        runtime.add_ground_plane(color=scene_spec.environment.ground_plane_color)
        if self.genesis_cfg.spawn_bed and scene_spec.support_surface is not None:
            surf = scene_spec.support_surface
            if surf.spawn_in_genesis:
                runtime.add_box(
                    BoxEntityConfig(
                        name=surf.name,
                        pos=surf.pos,
                        size=surf.size,
                        quat_xyzw=surf.quat_xyzw,
                        color=surf.color,
                    )
                )
        if self.genesis_cfg.spawn_robot and scene_spec.robot is not None:
            registry = RobotRegistry()
            specs = select_robot_specs(scene_spec, robots_mode="primary")
            names = add_robots_to_runtime(runtime, specs, registry, enable_collision=False)
            runtime.build()
            init_robots_after_build(runtime, registry, specs, names)
        else:
            runtime.build()
        self._runtime = runtime
        return runtime

    def _ensure_drawer(self) -> TrackSkeletonDrawer:
        if self._drawer is None:
            rgba = self.genesis_cfg.track_mesh_rgba
            self._drawer = TrackSkeletonDrawer(
                self._runtime,
                joint_rgba=(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
            )
        return self._drawer

    def draw_track_frame(self, track: MultiviewTrackFrame) -> None:
        if self._runtime is None:
            raise RuntimeError("GenesisTrackOverlay.setup() was not called.")
        drawer = self._ensure_drawer()
        drawer.draw(track.keypoints3d, track.keypoints3d_schema)
        self._live_frame_counter += 1

    @property
    def runtime(self) -> GenesisPlatformRuntime | None:
        return self._runtime

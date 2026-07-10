from __future__ import annotations

import importlib
import importlib.util
import json
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh

from common.project import project_paths
from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_debug_log
from projects.genesis_ue_sync.sim_platform.datasets import build_trimesh_sequence
from projects.genesis_ue_sync.sim_platform.embodiments import build_panda_ultrasound_preset
from projects.genesis_ue_sync.sim_platform.scenes import SyncSceneSpec
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
    MeshEntityConfig,
)


@dataclass(frozen=True)
class GenesisMaskRendererConfig:
    backend: str = "cpu"
    robot_enabled: bool = True
    human_anchor_override: tuple[float, float, float] | None = None
    segmentation_threshold: float = 1e-6
    cache_mesh_dir: Path | None = None
    export_png: bool = True


@dataclass
class GenesisMaskSequence:
    masks: dict[str, list[np.ndarray]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def by_camera(self) -> dict[str, list[np.ndarray]]:
        return self.masks


def _log_opengl_runtime_state(*, location: str, message: str, hypothesis_id: str) -> None:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    try:
        egl_spec = importlib.util.find_spec("OpenGL.EGL")
        egl_origin = None if egl_spec is None else egl_spec.origin
    except Exception as exc:
        egl_origin = f"find_spec_error:{exc}"
    opengl_mod = sys.modules.get("OpenGL.EGL")
    opengl_loaded = None if opengl_mod is None else getattr(opengl_mod, "__file__", None)
    append_debug_log(
        location=location,
        message=message,
        data={
            "python_executable": sys.executable,
            "user_site": user_site,
            "enable_user_site": getattr(site, "ENABLE_USER_SITE", None),
            "user_site_present_in_syspath": bool(user_site and any(str(p) == str(user_site) for p in sys.path)),
            "syspath_entries_with_local": [str(p) for p in sys.path if ".local" in str(p)],
            "opengl_egl_spec_origin": egl_origin,
            "opengl_egl_loaded_file": opengl_loaded,
        },
        run_id="genesis_mask",
        hypothesis_id=hypothesis_id,
    )


def _sanitize_opengl_import_path() -> dict[str, Any]:
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = None
    removed_paths: list[str] = []
    if user_site:
        keep: list[str] = []
        for entry in sys.path:
            if str(entry) == str(user_site):
                removed_paths.append(str(entry))
                continue
            keep.append(entry)
        sys.path[:] = keep
    purged_modules: list[str] = []
    for name in list(sys.modules.keys()):
        if name == "OpenGL" or name.startswith("OpenGL."):
            purged_modules.append(name)
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return {
        "removed_paths": removed_paths,
        "purged_opengl_modules": purged_modules,
    }


def _ensure_mesh_cache(meshes: list[trimesh.Trimesh], cache_dir: Path) -> list[Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for frame_idx, mesh in enumerate(meshes):
        path = cache_dir / f"frame_{frame_idx:05d}.obj"
        if not path.is_file():
            mesh.export(path)
        out.append(path)
    return out


def _default_human_anchor(scene_spec: SyncSceneSpec) -> tuple[float, float, float]:
    return scene_spec.resolved_human_anchor()


def render_genesis_masks(
    *,
    motion_sequence,
    calibration: CalibrationBundle,
    scene_spec: SyncSceneSpec,
    output_dir: Path,
    config: GenesisMaskRendererConfig,
    robot_joint_positions: list[float] | None = None,
) -> GenesisMaskSequence:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_opengl_runtime_state(
        location="src/projects/genesis_ue_sync/human_recovery/genesis_mask_renderer.py:render_genesis_masks:pre_sanitize",
        message="Genesis OpenGL import state before sanitize",
        hypothesis_id="G1",
    )
    sanitize_info = _sanitize_opengl_import_path()
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/genesis_mask_renderer.py:render_genesis_masks:sanitize",
        message="Sanitized user site OpenGL path",
        data=sanitize_info,
        run_id="genesis_mask",
        hypothesis_id="G1",
    )
    _log_opengl_runtime_state(
        location="src/projects/genesis_ue_sync/human_recovery/genesis_mask_renderer.py:render_genesis_masks:post_sanitize",
        message="Genesis OpenGL import state after sanitize",
        hypothesis_id="G1",
    )
    cache_dir = config.cache_mesh_dir or (output_dir / "mesh_cache")
    human_anchor = config.human_anchor_override or _default_human_anchor(scene_spec)
    meshes = build_trimesh_sequence(
        motion_sequence,
        world_offset=human_anchor,
        align_floor=False,
        color=(240, 240, 240, 255),
    )
    mesh_paths = _ensure_mesh_cache(meshes, cache_dir)
    masks: dict[str, list[np.ndarray]] = {camera_id: [] for camera_id in calibration.ordered_camera_ids()}
    per_camera_png_dir: dict[str, Path] = {}
    if config.export_png:
        for camera_id in calibration.ordered_camera_ids():
            camera_dir = output_dir / "png" / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            per_camera_png_dir[camera_id] = camera_dir
    for frame_idx, mesh_path in enumerate(mesh_paths):
        runtime = GenesisPlatformRuntime(
            GenesisRuntimeConfig(
                backend=str(config.backend),
                show_viewer=False,
                show_fps=False,
                enable_collision=False,
                gravity=(0.0, 0.0, 0.0),
            )
        )
        runtime.initialize()
        runtime.add_mesh_entity(
            MeshEntityConfig(
                name="patient_mesh",
                file=mesh_path,
                pos=(0.0, 0.0, 0.0),
                scale=1.0,
                fixed=True,
                color=(0.95, 0.95, 0.95, 1.0),
            )
        )
        robot_name = None
        if config.robot_enabled:
            robot_name = scene_spec.robot.name
            embodiment = build_panda_ultrasound_preset(camera_names=())
            runtime.add_articulated_entity(
                embodiment,
                name=robot_name,
                pos=scene_spec.robot.base_pos,
                quat_xyzw=scene_spec.robot.base_quat_xyzw,
            )
        runtime.add_camera_rig(
            camera_configs=calibration.static_camera_configs(),
            sensor_profiles=calibration.sensor_profiles(),
        )
        runtime.build()
        if robot_name is not None:
            runtime.reset()
            joints = robot_joint_positions if robot_joint_positions is not None else scene_spec.robot.joint_positions
            if joints:
                runtime.set_robot_joint_positions(robot_name, np.asarray(joints, dtype=np.float32))
        rendered = runtime.render_all_cameras(modalities=("segmentation",), force_render=True)
        for camera_id, payload in rendered.items():
            segmentation = np.asarray(payload["segmentation"])
            if segmentation.ndim == 3:
                segmentation = segmentation[..., 0]
            mask = np.asarray(np.abs(segmentation) > float(config.segmentation_threshold), dtype=bool)
            masks[camera_id].append(mask)
            if config.export_png:
                imageio.imwrite(per_camera_png_dir[camera_id] / f"mask_{frame_idx:05d}.png", (mask.astype(np.uint8) * 255))
        runtime.close()
    meta = {
        "frame_count": int(len(mesh_paths)),
        "camera_ids": calibration.ordered_camera_ids(),
        "mesh_cache_dir": str(cache_dir),
        "human_anchor": [float(v) for v in human_anchor],
    }
    (output_dir / "mask_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return GenesisMaskSequence(masks=masks, metadata=meta)


__all__ = [
    "GenesisMaskRendererConfig",
    "GenesisMaskSequence",
    "render_genesis_masks",
]

from __future__ import annotations

import json
import math
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from bridge.core.rotation import axis_angle_rotation, rotation_matrix_to_quaternion_xyzw
from common.project import project_paths
from projects.genesis_ue_sync.sim_platform.datasets import HumanMotionSequence, build_trimesh_sequence, load_amass_sequence
from projects.genesis_ue_sync.sim_platform.scenes import resolve_scene_spec_with_augmentation
from projects.genesis_ue_sync.sim_platform.scenes.human_bed_fit import fit_human_sequence_to_bed
from projects.genesis_ue_sync.sim_platform.scenes.robot_registry import RobotRegistry
from projects.genesis_ue_sync.sim_platform.scenes.robot_spawn import (
    add_robots_to_runtime,
    init_robots_after_build,
    select_robot_specs,
)
from projects.genesis_ue_sync.sim_platform.embodiments.smpl_capsule_runtime import (
    DEFAULT_SMPL_PROXY_VISUAL_RGBA,
    prepare_smpl_capsule_runtime_asset,
)
from projects.genesis_ue_sync.sim_platform.simulation.runtime import (
    BoxEntityConfig,
    GenesisPlatformRuntime,
    GenesisRuntimeConfig,
    MeshEntityConfig,
)


@dataclass(frozen=True)
class GenesisRenderQualitySpec:
    profile: str = "standard"
    ambient_light: tuple[float, float, float] = (0.3, 0.3, 0.3)
    plane_reflection: bool = True


def load_scene_motion_sequence(scene_spec) -> HumanMotionSequence:
    if scene_spec.motion.resolved_sequence_npz_path and scene_spec.motion.resolved_sequence_npz_path.is_file():
        return HumanMotionSequence.load(scene_spec.motion.resolved_sequence_npz_path)
    if scene_spec.motion.resolved_source_path and scene_spec.motion.resolved_source_path.is_file():
        return load_amass_sequence(scene_spec.motion.resolved_source_path)
    raise FileNotFoundError(
        "Scene motion is missing both sequence_npz_path and source_path. "
        "Provide one valid path in the SyncSceneSpec."
    )


def _trim_sequence_to_first_frame(sequence: HumanMotionSequence) -> HumanMotionSequence:
    image_names = list(sequence.image_names[:1]) if sequence.image_names else []
    cam_int = sequence.cam_int[:1] if sequence.cam_int is not None else None
    cam_ext = sequence.cam_ext[:1] if sequence.cam_ext is not None else None
    return replace(
        sequence,
        poses=np.asarray(sequence.poses[:1], dtype=np.float32),
        trans=np.asarray(sequence.trans[:1], dtype=np.float32),
        image_names=image_names,
        cam_int=cam_int,
        cam_ext=cam_ext,
    )


def render_sync_scene_genesis_frame0(
    *,
    scene_spec_path: str | Path,
    output_root: str | Path,
    augmentation_spec_path: str | Path | None = None,
    backend: str = "cuda",
    include_robot: bool = True,
    quality: GenesisRenderQualitySpec | None = None,
    fit_samples: int = 1,
    support_band_m: float = 0.03,
    center_margin_m: float = 0.05,
    robot_model: str = "",
) -> dict[str, Any]:
    quality = quality or GenesisRenderQualitySpec()
    repo = project_paths(__file__).root
    scene_spec, augmentation_summary = resolve_scene_spec_with_augmentation(scene_spec_path, augmentation_spec_path)
    sequence = _trim_sequence_to_first_frame(load_scene_motion_sequence(scene_spec))

    capsule_asset = prepare_smpl_capsule_runtime_asset(
        sequence,
        cache_dir=repo / "outputs" / "genesis_capsule_urdf_cache",
        device="cpu",
        visual_rgba=tuple(float(x) for x in DEFAULT_SMPL_PROXY_VISUAL_RGBA),
        force_rewrite=False,
    )
    placement = fit_human_sequence_to_bed(
        sequence,
        scene_spec=scene_spec,
        proxy_geometry=capsule_asset.proxy_geometry,
        device="cpu",
        sample_count=int(fit_samples),
        support_band_m=float(support_band_m),
        center_margin_m=float(center_margin_m),
    )
    world_offset = np.asarray(placement.world_offset, dtype=np.float32).reshape(3) + np.asarray(
        (0.0, 0.0, float(scene_spec.human.display_vertical_offset_m)),
        dtype=np.float32,
    )
    pitch_deg = float(scene_spec.human.display_pitch_forward_deg)
    if abs(pitch_deg) > 1e-9:
        r_pitch = axis_angle_rotation((1.0, 0.0, 0.0), math.radians(pitch_deg))
        human_mesh_quat_xyzw = tuple(float(x) for x in rotation_matrix_to_quaternion_xyzw(r_pitch).tolist())
    else:
        human_mesh_quat_xyzw = None
    human_mesh = build_trimesh_sequence(
        sequence,
        world_offset=tuple(float(v) for v in world_offset.tolist()),
        align_floor=False,
        color=(184, 209, 245, 255),
    )[0]
    human_cache_dir = repo / "outputs" / "genesis_frame0_human_cache"
    human_cache_dir.mkdir(parents=True, exist_ok=True)
    human_obj_path = human_cache_dir / f"{scene_spec.name}_frame0_human.obj"
    human_mesh.export(str(human_obj_path))

    viewer_cam = scene_spec.cameras[0] if scene_spec.cameras else None
    runtime = GenesisPlatformRuntime(
        GenesisRuntimeConfig(
            backend=backend,
            show_viewer=False,
            show_fps=False,
            gravity=(0.0, 0.0, 0.0),
            enable_collision=False,
            viewer_camera_pos=tuple(viewer_cam.pos) if viewer_cam is not None else (2.5, -2.0, 1.5),
            viewer_camera_lookat=tuple(viewer_cam.lookat) if viewer_cam is not None else (0.0, 0.0, 0.5),
            viewer_camera_fov=float(viewer_cam.fov) if viewer_cam is not None else 40.0,
            ambient_light=tuple(float(v) for v in quality.ambient_light),
            plane_reflection=bool(quality.plane_reflection),
        )
    )
    try:
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
        robot_specs = select_robot_specs(
            scene_spec,
            robots_mode="scene",
            no_robot=not bool(include_robot),
            robot_model=str(robot_model or ""),
        )
        registry = RobotRegistry()
        robot_names = add_robots_to_runtime(
            runtime,
            robot_specs,
            registry,
            enable_collision=False,
            repo_root=repo,
        )
        for cam in scene_spec.cameras:
            runtime.add_camera(cam)
        runtime.add_mesh_entity(
            MeshEntityConfig(
                name="human_smpl_preview",
                file=human_obj_path,
                pos=(0.0, 0.0, 0.0),
                quat_xyzw=human_mesh_quat_xyzw,
                scale=1.0,
                fixed=True,
                visualization=True,
                collision=False,
                color=(184.0 / 255.0, 209.0 / 255.0, 245.0 / 255.0, 1.0),
            )
        )
        runtime.build()
        if robot_specs:
            init_robots_after_build(runtime, registry, robot_specs, robot_names)
        else:
            runtime.reset()

        root = Path(output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        camera_outputs: dict[str, str] = {}
        for camera in scene_spec.cameras:
            rgb = runtime.render_camera(camera.name, rgb=True, force_render=True)["rgb"]
            out_path = root / f"{camera.name}_frame0.png"
            imageio.imwrite(out_path, np.asarray(rgb, dtype=np.uint8))
            camera_outputs[str(camera.name)] = str(out_path)

        report = {
            "scene_spec": str(Path(scene_spec_path).expanduser()),
            "augmentation": augmentation_summary,
            "motion_source": str(
                scene_spec.motion.resolved_sequence_npz_path
                or scene_spec.motion.resolved_source_path
                or ""
            ),
            "backend": backend,
            "quality": {
                "profile": quality.profile,
                "ambient_light": [float(v) for v in quality.ambient_light],
                "plane_reflection": bool(quality.plane_reflection),
            },
            "include_robot": bool(include_robot),
            "robot_model": str(robot_model or ""),
            "world_offset_m": [float(v) for v in world_offset.tolist()],
            "genesis_human_display_vertical_sink_m": float(scene_spec.human.display_vertical_sink_m),
            "genesis_human_display_vertical_offset_m": float(scene_spec.human.display_vertical_offset_m),
            "genesis_human_display_pitch_forward_deg": float(pitch_deg),
            "support_plane_z_m": float(placement.support_plane_z),
            "support_contact_ratio": float(placement.support_contact_ratio),
            "bed_fit_sample_indices": [int(i) for i in placement.sample_indices],
            "bed_fit_lower_shell_snap_dz_m": float(placement.lower_shell_snap_dz_m),
            "human_preview_obj": str(human_obj_path),
            "camera_outputs": camera_outputs,
        }
        (root / "genesis_frame0_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        runtime.close()

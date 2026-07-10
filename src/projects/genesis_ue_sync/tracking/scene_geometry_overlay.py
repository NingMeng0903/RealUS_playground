from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.debug_runtime import append_debug_log
from projects.genesis_ue_sync.tracking.feature_video_renderer import write_mp4_streaming
from projects.genesis_ue_sync.tracking.tracking_skeleton_overlay import project_world_points_to_pixels


_CUBOID_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _support_surface_corners_world(scene_spec) -> np.ndarray:
    surf = scene_spec.support_surface
    if surf is None:
        raise RuntimeError("Scene does not define support_surface.")
    cx, cy, cz = (float(v) for v in surf.pos)
    sx, sy, sz = (float(v) for v in surf.size)
    hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz
    return np.asarray(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
        ],
        dtype=np.float32,
    )


def render_support_surface_overlays_on_rgb(
    *,
    scene_spec,
    sequence_result,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    export_png: bool = True,
    export_mp4: bool = False,
    line_rgb: tuple[int, int, int] = (255, 0, 255),
    line_width: int = 3,
) -> dict[str, Any]:
    corners_world = _support_surface_corners_world(scene_spec)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    png_dirs: dict[str, Path] = {}
    mp4_paths: dict[str, Path | None] = {}
    debug_by_camera: dict[str, dict[str, Any]] = {}
    for camera_id in calibration.ordered_camera_ids():
        cam = calibration.camera(camera_id)
        uv, valid = project_world_points_to_pixels(corners_world, cam.camera_from_world, cam.intrinsics)
        cam_out = output_root / camera_id
        cam_out.mkdir(parents=True, exist_ok=True)
        png_dirs[camera_id] = cam_out
        frames_for_mp4: list[np.ndarray] = []
        for fr in sequence_result.frame_results:
            rgb = fr.rgb_frames.get(camera_id)
            if rgb is None:
                continue
            img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            for a, b in _CUBOID_EDGES:
                if not (bool(valid[a]) and bool(valid[b])):
                    continue
                pa = (float(uv[a, 0]), float(uv[a, 1]))
                pb = (float(uv[b, 0]), float(uv[b, 1]))
                draw.line([pa, pb], fill=tuple(int(v) for v in line_rgb), width=int(line_width))
            out = np.asarray(img, dtype=np.uint8)
            if export_png:
                stem = f"frame_{fr.frame_idx:05d}"
                Image.fromarray(out).save(cam_out / f"{stem}.png")
            if export_mp4:
                frames_for_mp4.append(out)
        mp4_paths[camera_id] = None
        if export_mp4 and frames_for_mp4:
            p = cam_out / f"{camera_id}_support_surface_overlay.mp4"
            write_mp4_streaming(p, frames_for_mp4, fps=float(fps))
            mp4_paths[camera_id] = p
        finite = np.all(np.isfinite(uv), axis=1) & valid
        if np.any(finite):
            bbox_min = np.min(uv[finite], axis=0)
            bbox_max = np.max(uv[finite], axis=0)
        else:
            bbox_min = np.array([np.nan, np.nan], dtype=np.float64)
            bbox_max = np.array([np.nan, np.nan], dtype=np.float64)
        debug_by_camera[camera_id] = {
            "visible_corner_count": int(np.sum(finite)),
            "uv_bbox_min": [float(bbox_min[0]), float(bbox_min[1])],
            "uv_bbox_max": [float(bbox_max[0]), float(bbox_max[1])],
        }
    append_debug_log(
        location="src/projects/genesis_ue_sync/human_recovery/scene_geometry_overlay.py:render_support_surface_overlays_on_rgb:summary",
        message="Support-surface overlay projection summary",
        data={
            "support_surface_pos": [float(v) for v in scene_spec.support_surface.pos],
            "support_surface_size": [float(v) for v in scene_spec.support_surface.size],
            "line_rgb": [int(v) for v in line_rgb],
            "per_camera": debug_by_camera,
        },
        run_id="debug-triage",
        hypothesis_id="H32",
    )
    return {
        "output_root": str(output_root.resolve()),
        "per_camera_png_dirs": {k: str(v.resolve()) for k, v in png_dirs.items()},
        "per_camera_mp4": {k: str(v.resolve()) if v is not None else None for k, v in mp4_paths.items()},
        "per_camera_debug": debug_by_camera,
    }


__all__ = ["render_support_surface_overlays_on_rgb"]

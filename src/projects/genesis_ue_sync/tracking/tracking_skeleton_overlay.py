from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle
from projects.genesis_ue_sync.tracking.uhmr_backend import UhmrSequenceResult
from projects.genesis_ue_sync.sim_platform.datasets.human_sequence import evaluate_smpl_sequence
from projects.genesis_ue_sync.tracking.feature_video_renderer import write_mp4_streaming

_SMPL24_PARENTS: tuple[int, ...] = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)


def _smpl24_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for j, p in enumerate(_SMPL24_PARENTS):
        if p >= 0:
            edges.append((p, j))
    return edges


def project_world_points_to_pixels(
    points_world: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, :3]
    t = np.asarray(camera_from_world, dtype=np.float64).reshape(4, 4)[:3, 3].reshape(3)
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    Xc = (R @ X.T).T + t
    z = Xc[:, 2]
    valid = z > 1e-5
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = np.full(X.shape[0], np.nan, dtype=np.float64)
    v = np.full(X.shape[0], np.nan, dtype=np.float64)
    u[valid] = fx * Xc[valid, 0] / z[valid] + cx
    v[valid] = fy * Xc[valid, 1] / z[valid] + cy
    return np.stack([u, v], axis=-1), valid


def _overlay_skeleton_on_rgb(
    rgb: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    *,
    edges: list[tuple[int, int]],
    line_width: int,
    pred_cam_t: np.ndarray | None,
) -> np.ndarray:
    img = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = img.size[1], img.size[0]
    green = (0, 220, 80)
    red = (255, 60, 60)
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        pa = (float(uv[a, 0]), float(uv[a, 1]))
        pb = (float(uv[b, 0]), float(uv[b, 1]))
        if not (0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h):
            continue
        draw.line([pa, pb], fill=green, width=int(line_width))
    r = max(2, int(line_width))
    for i in range(uv.shape[0]):
        if not valid[i]:
            continue
        x, y = float(uv[i, 0]), float(uv[i, 1])
        if not (0 <= x < w and 0 <= y < h):
            continue
        draw.ellipse((x - r, y - r, x + r, y + r), outline=red, width=2)
    if pred_cam_t is not None:
        t = np.asarray(pred_cam_t, dtype=np.float64).reshape(-1)
        text = "pred_cam_t " + " ".join(f"{v:.3f}" for v in t[:3])
        try:
            font = ImageFont.load_default()
        except OSError:  # pragma: no cover
            font = None  # type: ignore[assignment]
        draw.text((8, 8), text, fill=(255, 255, 0), font=font)
    return np.asarray(img, dtype=np.uint8)


def render_tracking_skeleton_overlays(
    *,
    sequence_result: UhmrSequenceResult,
    calibration: CalibrationBundle,
    output_root: Path,
    fps: float,
    smpl_device: str | None = None,
    export_png: bool = True,
    export_mp4: bool = False,
    line_width: int = 3,
    joint_count: int = 24,
) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    motion = sequence_result.motion_sequence
    _, joints = evaluate_smpl_sequence(
        motion,
        device=smpl_device,
        include_vertices=False,
        include_joints=True,
    )
    if joints is None:
        raise RuntimeError("SMPL joint evaluation failed for tracking overlay.")
    jn = min(int(joint_count), int(joints.shape[1]))
    edges = _smpl24_edges()
    edges = [(a, b) for a, b in edges if a < jn and b < jn]

    camera_ids = calibration.ordered_camera_ids()
    png_dirs: dict[str, Path] = {}
    mp4_paths: dict[str, Path | None] = {}
    for camera_id in camera_ids:
        cam = calibration.camera(camera_id)
        K = cam.intrinsics
        ext = cam.camera_from_world
        cam_out = output_root / camera_id
        cam_out.mkdir(parents=True, exist_ok=True)
        png_dirs[camera_id] = cam_out
        frames_for_mp4: list[np.ndarray] = []
        for fr in sequence_result.frame_results:
            rgb = fr.rgb_frames.get(camera_id)
            if rgb is None:
                continue
            J = joints[fr.frame_idx, :jn, :]
            uv, valid = project_world_points_to_pixels(J, ext, K)
            pct = fr.pred_cam_t.get(camera_id)
            out = _overlay_skeleton_on_rgb(
                rgb,
                uv,
                valid,
                edges=edges,
                line_width=int(line_width),
                pred_cam_t=pct,
            )
            if export_png:
                stem = f"frame_{fr.frame_idx:05d}"
                Image.fromarray(out).save(cam_out / f"{stem}.png")
            if export_mp4:
                frames_for_mp4.append(out)
        mp4_paths[camera_id] = None
        if export_mp4 and frames_for_mp4:
            p = cam_out / f"{camera_id}_skeleton_overlay.mp4"
            write_mp4_streaming(p, frames_for_mp4, fps=float(fps))
            mp4_paths[camera_id] = p

    return {
        "output_root": str(output_root.resolve()),
        "per_camera_png_dirs": {k: str(v.resolve()) for k, v in png_dirs.items()},
        "per_camera_mp4": {k: str(v.resolve()) if v is not None else None for k, v in mp4_paths.items()},
        "joint_count": jn,
    }


__all__ = [
    "project_world_points_to_pixels",
    "render_tracking_skeleton_overlays",
]

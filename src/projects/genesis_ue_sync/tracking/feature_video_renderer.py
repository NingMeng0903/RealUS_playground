from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import imageio.v2 as imageio
import numpy as np

from projects.genesis_ue_sync.tracking.heatmap_ops import overlay_heatmap


@dataclass(frozen=True)
class FeatureVideoOutputs:
    per_camera_mp4: dict[str, Path]
    strip_mp4: Path | None


def _sanitize_video_frame(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _mp4_writer(path: Path, *, fps: float):
    return imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=max(float(fps), 1e-6),
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )


def write_mp4_streaming(path: Path, frames: Iterable[np.ndarray], *, fps: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _mp4_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(_sanitize_video_frame(frame))
    return path


def write_mp4(path: Path, frames: list[np.ndarray], *, fps: float) -> Path:
    return write_mp4_streaming(path, iter(frames), fps=fps)


def hstack_mp4_videos(input_paths: list[Path], output_path: Path, *, fps: float) -> Path:
    """Horizontally stack same-length mp4s using ffmpeg (disk-friendly)."""
    _ffmpeg_hstack_strip(input_paths=input_paths, output_path=output_path, fps=fps)
    return Path(output_path)


def _ffmpeg_hstack_strip(
    *,
    input_paths: list[Path],
    output_path: Path,
    fps: float,
) -> None:
    if not input_paths:
        raise ValueError("hstack requires at least one input video.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(input_paths)
    if n == 1:
        shutil.copyfile(input_paths[0], output_path)
        return
    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]")
    fc = f"{''.join(parts)}hstack=inputs={n}[outv]"
    cmd = ["ffmpeg", "-y"]
    for p in input_paths:
        cmd.extend(["-i", str(p)])
    cmd.extend(
        [
            "-filter_complex",
            fc,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(max(float(fps), 1e-6)),
            str(output_path),
        ]
    )
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found; install ffmpeg or set vit_video.enable_strip=false "
            "to skip strip encoding."
        ) from exc


def render_feature_videos(
    *,
    rgb_frames: dict[str, list[np.ndarray]],
    heatmaps: dict[str, list[np.ndarray]],
    output_dir: Path,
    fps: float,
    alpha: float = 0.45,
    strip_name: str = "multiview_vit_overlay.mp4",
    enable_strip: bool = False,
) -> FeatureVideoOutputs:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_camera_paths: dict[str, Path] = {}
    ordered_ids = list(rgb_frames.keys())
    tmp_paths: list[Path] = []
    tmp_dir = output_dir / "_vit_strip_tmp"
    try:
        for camera_id in ordered_ids:
            frames = rgb_frames[camera_id]
            camera_heatmaps = heatmaps.get(camera_id)
            if camera_heatmaps is None:
                raise KeyError(f"Missing heatmaps for camera: {camera_id}")
            if len(frames) != len(camera_heatmaps):
                raise ValueError(
                    f"RGB/heatmap length mismatch for {camera_id}: {len(frames)} vs {len(camera_heatmaps)}"
                )

            def overlay_iter() -> Iterator[np.ndarray]:
                for frame, heatmap in zip(frames, camera_heatmaps):
                    yield overlay_heatmap(frame, heatmap, alpha=float(alpha))

            out_path = output_dir / f"{camera_id}_vit_overlay.mp4"
            if enable_strip:
                tmp_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = tmp_dir / f"{camera_id}_strip_src.mp4"
                write_mp4_streaming(tmp_path, overlay_iter(), fps=fps)
                tmp_paths.append(tmp_path)
            else:
                write_mp4_streaming(out_path, overlay_iter(), fps=fps)
            per_camera_paths[camera_id] = out_path

        strip_path: Path | None = None
        if enable_strip and tmp_paths:
            for i, camera_id in enumerate(ordered_ids):
                shutil.move(str(tmp_paths[i]), str(per_camera_paths[camera_id]))
            strip_path = output_dir / strip_name
            ordered_files = [per_camera_paths[cid] for cid in ordered_ids]
            _ffmpeg_hstack_strip(input_paths=ordered_files, output_path=strip_path, fps=fps)
    finally:
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return FeatureVideoOutputs(per_camera_mp4=per_camera_paths, strip_mp4=strip_path)


def slice_frame_dicts(
    rgb_frames: dict[str, list[np.ndarray]],
    heatmaps: dict[str, list[np.ndarray]],
    *,
    start: int | None,
    end: int | None,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]:
    n = len(next(iter(rgb_frames.values())))
    s = 0 if start is None else max(0, int(start))
    e = n if end is None else int(end)
    if e < s:
        raise ValueError(f"Invalid vit slice [{s}, {e}) for length {n}.")
    e = min(e, n)
    rgb_out = {k: v[s:e] for k, v in rgb_frames.items()}
    hm_out = {k: v[s:e] for k, v in heatmaps.items()}
    return rgb_out, hm_out


__all__ = [
    "FeatureVideoOutputs",
    "hstack_mp4_videos",
    "render_feature_videos",
    "write_mp4",
    "write_mp4_streaming",
    "slice_frame_dicts",
]

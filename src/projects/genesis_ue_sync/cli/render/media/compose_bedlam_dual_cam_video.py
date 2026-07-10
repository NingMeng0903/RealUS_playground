from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def collect_positive_frames(seq_dir: Path) -> list[Path]:
    pattern = re.compile(r"_(\d+)\.png$")
    ordered: list[tuple[int, Path]] = []
    for path in seq_dir.glob("*.png"):
        if "_-" in path.name:
            continue
        match = pattern.search(path.name)
        if match is None:
            continue
        ordered.append((int(match.group(1)), path))
    ordered.sort()
    return [path for _, path in ordered]


def _write_mp4(path: Path, frames: list[np.ndarray], *, fps: int) -> None:
    out: list[np.ndarray] = []
    for im in frames:
        arr = np.asarray(im)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        out.append(arr)
    imageio.mimsave(
        path,
        out,
        fps=fps,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/media/camp/PEI_T7/Among_US/outputs/bedlam_ue_dual_cam_sync")
    png_root = root / "png"
    seq_names = sys.argv[2:] if len(sys.argv) > 2 else ["amass_gc_cam_a", "amass_gc_cam_b"]
    fps = int(os.environ.get("AMONGUS_COMPOSE_FPS", "8"))

    frames: dict[str, list[np.ndarray]] = {}
    for name in seq_names:
        paths = collect_positive_frames(png_root / name)
        if not paths:
            raise FileNotFoundError(f"No PNG frames found for sequence: {png_root / name}")
        frames[name] = [imageio.imread(path) for path in paths]

    frame_count = min(len(images) for images in frames.values())
    trimmed_frames = {name: images[:frame_count] for name, images in frames.items()}

    for name, images in trimmed_frames.items():
        out_path = root / f"{name}.mp4"
        _write_mp4(out_path, images, fps=fps)

    multiview_strip = [
        np.concatenate([trimmed_frames[name][idx] for name in seq_names], axis=1)
        for idx in range(frame_count)
    ]
    multiview_path = root / "multiview_strip.mp4"
    _write_mp4(multiview_path, multiview_strip, fps=fps)

    side_path = None
    if len(seq_names) >= 2:
        side_by_side = [
            np.concatenate([trimmed_frames[seq_names[0]][idx], trimmed_frames[seq_names[1]][idx]], axis=1)
            for idx in range(frame_count)
        ]
        side_path = root / "side_by_side.mp4"
        _write_mp4(side_path, side_by_side, fps=fps)

    meta = {
        "source": "UE BEDLAM MRQ",
        "sequence_names": list(seq_names),
        "sequence_png_dirs": {name: str(png_root / name) for name in seq_names},
        "sequence_mp4s": {name: str(root / f"{name}.mp4") for name in seq_names},
        "multiview_strip_mp4": str(multiview_path),
        "side_by_side_mp4": str(side_path) if side_path is not None else None,
        "positive_frame_count": frame_count,
        "body_asset": seq_names[0].split("_cam_")[0],
        "body_texture": None,
    }
    (root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

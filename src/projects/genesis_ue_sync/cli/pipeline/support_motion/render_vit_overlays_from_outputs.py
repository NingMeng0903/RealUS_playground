#!/usr/bin/env python3
"""Re-render ViT heatmap overlay mp4s from a prior tracking output (no U-HMR rerun).

Reads heatmaps saved as <output-root>/<heatmaps-subdir>/<cam>/frame_XXXXX.npy (default:
heatmaps; use heatmaps_mid for mid-layer) and RGB paths rebuilt from run_meta.json.
Jet-colored overlay matches the full pipeline (see heatmap_ops.overlay_heatmap).

Normally uses <output-root>/tracking_result.json for run_meta_path, input_fps,
pipeline_sampling, and camera_ids. If the full pipeline was killed before that file
was written (e.g. OOM during ViT encoding), pass --run-meta and optional sampling flags
to match the original tracking YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from projects.genesis_ue_sync.tracking.feature_video_renderer import (
    hstack_mp4_videos,
    write_mp4_streaming,
)
from projects.genesis_ue_sync.tracking.heatmap_ops import overlay_heatmap
from projects.genesis_ue_sync.tracking.multiview_io import build_multiview_request_from_run_meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-root",
        type=Path,
        help="Tracking output directory (contains heatmaps/ and tracking_result.json).",
    )
    p.add_argument(
        "--tracking-result",
        type=Path,
        default=None,
        help="Path to tracking_result.json (default: <output-root>/tracking_result.json).",
    )
    p.add_argument("--frame-start", type=int, default=0, help="First heatmap index (inclusive, 0-based).")
    p.add_argument("--frame-end", type=int, default=None, help="Last heatmap index (exclusive). Default: all frames.")
    p.add_argument("--fps", type=float, default=None, help="Override fps (default: from tracking_result input_fps).")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--enable-strip", action="store_true", help="Also write multiview strip via ffmpeg hstack.")
    p.add_argument("--strip-name", type=str, default="multiview_vit_overlay.mp4")
    p.add_argument(
        "--run-meta",
        type=Path,
        default=None,
        help="run_meta.json path when tracking_result.json is missing (repo-relative or absolute).",
    )
    p.add_argument(
        "--pipeline-frame-start",
        type=int,
        default=0,
        help="With --run-meta only: same as tracking YAML frame_start (default 0).",
    )
    p.add_argument(
        "--pipeline-frame-step",
        type=int,
        default=1,
        help="With --run-meta only: same as tracking YAML frame_step (default 1).",
    )
    p.add_argument(
        "--pipeline-frame-limit",
        type=int,
        default=None,
        help="With --run-meta only: same as tracking YAML frame_limit (omit for full sequence).",
    )
    p.add_argument(
        "--camera-ids",
        type=str,
        default=None,
        help="Comma-separated order, e.g. cam_left,cam_right,cam_top. Default: sorted subdirs under heatmaps/.",
    )
    p.add_argument(
        "--heatmaps-subdir",
        type=str,
        default="heatmaps",
        help="Subdirectory under output-root: heatmaps (final ViT) or heatmaps_mid (mid-layer).",
    )
    p.add_argument(
        "--vit-out-subdir",
        type=str,
        default=None,
        help="Output folder under output-root (default: vit_videos or vit_videos_mid when heatmaps-subdir is heatmaps_mid).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        raise SystemExit("--output-root is required.")
    raw_root = str(args.output_root)
    if "..." in raw_root or raw_root.strip() in {".", ".."}:
        raise SystemExit(
            "Replace the placeholder in --output-root with a real directory path, e.g.\n"
            "  --output-root outputs/tracking/ue_exec2_multiview_tracking\n"
            "Do not pass literal '...' from the docs."
        )
    output_root = Path(args.output_root).resolve()
    hm_sub = str(args.heatmaps_subdir).strip().strip("/") or "heatmaps"
    heat_root = output_root / hm_sub
    if not heat_root.is_dir():
        raise FileNotFoundError(f"Missing heatmaps directory: {heat_root}")

    tr_path = Path(args.tracking_result).resolve() if args.tracking_result else (output_root / "tracking_result.json")
    tr_path = tr_path.resolve()
    result: dict | None = None
    if tr_path.is_file():
        result = json.loads(tr_path.read_text(encoding="utf-8"))

    if result is not None:
        run_meta = Path(result["run_meta_path"]).resolve()
        sampling = result.get("pipeline_sampling") or {}
        fs = int(sampling.get("frame_start", 0))
        fst = int(sampling.get("frame_step", 1))
        fl = sampling.get("frame_limit")
        fl = int(fl) if fl not in {None, "", 0} else None
        fps = float(args.fps) if args.fps is not None else float(result.get("input_fps", 30.0))
        camera_ids = list(result.get("camera_ids") or [])
    else:
        if args.run_meta is None:
            inferred = sorted(
                p.name
                for p in heat_root.iterdir()
                if p.is_dir() and any(p.glob("frame_*.npy"))
            )
            raise SystemExit(
                f"Missing tracking summary: {tr_path}\n"
                "The full pipeline writes this file only at the very end. If the process was killed "
                "during ViT video or later, pass the same run_meta as your tracking config, e.g.\n"
                "  --run-meta dataset/demo_video/ue_render_exec2/ue_render/run_meta.json \\\n"
                "  --fps 30 \\\n"
                "  --pipeline-frame-start 0 --pipeline-frame-step 1\n"
                f"Heatmap camera subdirs found: {inferred}"
            )
        run_meta = Path(args.run_meta)
        if not run_meta.is_file():
            raise FileNotFoundError(f"--run-meta not found: {run_meta}")
        run_meta = run_meta.resolve()
        fs = int(args.pipeline_frame_start)
        fst = int(args.pipeline_frame_step)
        fl = args.pipeline_frame_limit
        fl = int(fl) if fl is not None and fl > 0 else None
        fps = float(args.fps) if args.fps is not None else 30.0
        if args.camera_ids:
            camera_ids = [s.strip() for s in str(args.camera_ids).split(",") if s.strip()]
        else:
            camera_ids = sorted(
                p.name
                for p in heat_root.iterdir()
                if p.is_dir() and any(p.glob("frame_*.npy"))
            )
        if not camera_ids:
            raise SystemExit(f"No camera subdirs with frame_*.npy under {heat_root}")

    frame_set = build_multiview_request_from_run_meta(
        run_meta,
        fps=fps,
        max_frames=fl,
        start_frame=fs,
        frame_step=fst,
    )
    request = frame_set.to_request()
    if result is not None and not camera_ids:
        camera_ids = list(request.views.keys())
    n = len(request.views[camera_ids[0]])
    end = n if args.frame_end is None else int(args.frame_end)
    start = max(0, int(args.frame_start))
    if end <= start or end > n:
        raise ValueError(f"Invalid slice [{start}, {end}) for {n} frames.")
    indices = range(start, end)
    if args.vit_out_subdir:
        vit_out = output_root / str(args.vit_out_subdir).strip().strip("/")
    else:
        vit_out = output_root / ("vit_videos_mid" if hm_sub == "heatmaps_mid" else "vit_videos")
    vit_out.mkdir(parents=True, exist_ok=True)
    is_mid = hm_sub == "heatmaps_mid"
    mp4_suffix = "_vit_mid_overlay" if is_mid else "_vit_overlay"

    per_paths: list[Path] = []
    for camera_id in camera_ids:
        out_mp4 = vit_out / f"{camera_id}{mp4_suffix}.mp4"

        def overlay_iter():
            for i in indices:
                hm_path = heat_root / camera_id / f"frame_{i:05d}.npy"
                if not hm_path.is_file():
                    raise FileNotFoundError(hm_path)
                hm = np.load(hm_path).astype(np.float32)
                png = Path(request.views[camera_id][i].image_path)
                rgb = np.asarray(Image.open(png).convert("RGB"))
                yield overlay_heatmap(rgb, hm, alpha=float(args.alpha))

        write_mp4_streaming(out_mp4, overlay_iter(), fps=fps)
        per_paths.append(out_mp4)
        print(f"Wrote {out_mp4}")

    if args.enable_strip:
        strip_name = args.strip_name
        if is_mid and strip_name.lower().endswith(".mp4"):
            strip_name = strip_name[:-4] + "_mid.mp4"
        strip_path = vit_out / strip_name
        hstack_mp4_videos(per_paths, strip_path, fps=fps)
        print(f"Wrote {strip_path}")


if __name__ == "__main__":
    main()

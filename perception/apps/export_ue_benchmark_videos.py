#!/usr/bin/env python3
"""Export one aligned MP4 per camera from a strict UE recording manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_videos(recording_dir: Path, *, fps: float = 30.0, output_dir: Path | None = None) -> dict:
    import cv2

    if fps <= 0:
        raise ValueError("fps must be positive")
    root = recording_dir.expanduser().resolve()
    manifest = root / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("Recording must contain complete successful capture groups")
    cameras = sorted(rows[0]["image_paths"])
    if any(sorted(row["image_paths"]) != cameras for row in rows):
        raise ValueError("Camera set changes within recording")
    destination = output_dir or root / "videos"
    destination.mkdir(parents=True, exist_ok=True)
    writers = {}
    shapes = {}
    frame_mapping = []
    try:
        for index, row in enumerate(rows):
            canonical = json.loads((root / row["canonical_path"]).read_text())
            human = canonical.get("human") or {}
            frame_mapping.append({
                "video_frame_index": index,
                "video_time_ns": round(index * 1e9 / fps),
                "capture_group_id": row["capture_group_id"],
                "loop_id": row.get("loop_id"),
                "canonical_frame_index": row["frame_index"],
                "source_sim_time_ns": row.get("sim_time_ns"),
                "motion_frame_index": human.get("motion_frame_index"),
                "motion_time_ns": human.get("motion_time_ns"),
                "motion_fps": human.get("motion_fps"),
                "benchmark_time_ns": canonical.get("benchmark_time_ns"),
                "benchmark_phase": canonical.get("benchmark_phase"),
                "canonical_path": row["canonical_path"],
                "gt_path": row["gt_path"],
            })
            for camera in cameras:
                path = root / row["image_paths"][camera]
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f"Cannot decode recording image: {path}")
                height, width = frame.shape[:2]
                if camera not in writers:
                    video = destination / f"{camera}.mp4"
                    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot open MP4 encoder for {video}")
                    writers[camera] = writer
                    shapes[camera] = (width, height)
                if (width, height) != shapes[camera]:
                    raise ValueError(f"Image resolution changed for {camera}")
                writers[camera].write(frame)
    finally:
        for writer in writers.values():
            writer.release()
    summary = {
        "recording_dir": str(root), "fps": fps, "frame_count": len(rows),
        "camera_names": cameras,
        "videos": {camera: str(destination / f"{camera}.mp4") for camera in cameras},
        # Keep an explicit mapping: the MP4 frame number is not the original
        # simulation tick or AMASS source frame number.
        "frames": frame_mapping,
    }
    (destination / "video_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    summary = export_videos(args.recording_dir, fps=args.fps, output_dir=args.output_dir)
    print(json.dumps({key: value for key, value in summary.items() if key != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

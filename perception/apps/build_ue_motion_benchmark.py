#!/usr/bin/env python3
"""Prepend a static shape-initialization interval to a Genesis canonical log.

The first source pose is held for ``warmup_s``. The source motion frame and
clock remain unchanged; ``benchmark_time_ns`` is the unique video/replay
clock. Use the recorder's ``--keep-duplicate-motion-frames`` option.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def build_benchmark_log(source: Path, output: Path, *, warmup_s: float = 0.5, fps: float = 30.0) -> dict:
    if fps <= 0 or warmup_s < 0:
        raise ValueError("fps must be positive and warmup_s nonnegative")
    if output.exists():
        raise FileExistsError(f"Benchmark log already exists: {output}")
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("human"), dict) for row in rows):
        raise ValueError("Source must contain canonical human motion frames")
    warmup_count = round(warmup_s * fps)
    frames = [(rows[0], "shape_initialization")] * warmup_count + [(row, "motion") for row in rows]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        for index, (source_row, phase) in enumerate(frames):
            row = copy.deepcopy(source_row)
            row["source_canonical_frame_index"] = row.get("frame_index")
            row["frame_index"] = index
            row["benchmark_time_ns"] = round(index * 1e9 / fps)
            row["benchmark_phase"] = phase
            row["benchmark_fps"] = fps
            row["human"]["benchmark_time_ns"] = row["benchmark_time_ns"]
            row["human"]["benchmark_phase"] = phase
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "source_canonical_jsonl": str(source.resolve()), "canonical_jsonl": str(output.resolve()),
        "fps": fps, "warmup_s": warmup_count / fps, "warmup_frames": warmup_count,
        "motion_frames": len(rows), "total_frames": len(frames),
        "motion_source_start": rows[0]["human"].get("motion_frame_index"),
        "shape_initialization": "static first pose; RGB-derived betas, no ground-truth betas",
        "recorder_args": ["--keep-duplicate-motion-frames"],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(build_benchmark_log(args.canonical_jsonl, args.output_jsonl, warmup_s=args.warmup_s, fps=args.fps), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

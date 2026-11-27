#!/usr/bin/env python3
"""Score reconstructed native SMPL-X joints against recorded UE bones."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perception.benchmark.evaluation import evaluate_predictions


def sequence_latency(row: dict):
    # A fresh moment now includes RGB inference/fusion in addition to the
    # DWPose frontend and geometric backend. Do not silently drop that stage.
    for key in ("latency_s", "total_elapsed_s", "algorithm_elapsed_s"):
        if row.get(key) is not None:
            return float(row[key]), key
    frontend = row.get("frontend_elapsed_s", row.get("frontend_wall_s"))
    backend = row.get("backend_elapsed_s", row.get("fit_elapsed_s"))
    if frontend is not None and backend is not None:
        rgb = row.get("hybrik_rgb") or {}
        if row.get("frontend_includes_rgb"):
            return float(frontend) + float(backend), "overlapped_component_sum"
        if rgb and rgb.get("wall_elapsed_s") is None:
            return None, "rgb_timing_missing"
        return float(frontend) + float(backend) + float(rgb.get("wall_elapsed_s", 0)), "component_sum"
    return None, "unavailable"


def sequence_predictions(root: Path) -> list[dict]:
    index = json.loads((root / "index.json").read_text())
    result = []
    for row in index.get("index", []):
        moment = Path(row["moment_dir"])
        if not moment.is_absolute():
            moment = root / moment
        with np.load(moment / "smplx_result.npz", allow_pickle=False) as data:
            key = next((k for k in ("native55_joints_world_m", "joints_world_m55", "joints_world_m") if k in data), None)
            if key is None:
                raise ValueError(f"{moment}: true native SMPL-X55 joints are required")
            joints = np.asarray(data[key], dtype=np.float64)
        frontend = row.get("frontend_elapsed_s", row.get("frontend_wall_s"))
        backend = row.get("backend_elapsed_s", row.get("fit_elapsed_s"))
        elapsed, timing_source = sequence_latency(row)
        result.append({"frame_index": int(row["frame_index"]), "status": "ok",
                       "joints_world_m": joints.tolist(), "latency_elapsed_s": elapsed,
                       "frontend_elapsed_s": frontend, "backend_elapsed_s": backend,
                       "latency_measurement_source": timing_source})
    for row in index.get("rejected_index", []):
        frame = row.get("frame_index", row.get("source_frame_index"))
        if frame is None:
            raise ValueError("Rejected sequence row is missing its recorded source frame")
        elapsed = row.get("algorithm_elapsed_s")
        if elapsed is None:
            elapsed = row.get("latency_s", row.get("frontend_elapsed_s", row.get("frontend_wall_s")))
        result.append({"frame_index": int(frame), "status": "rejected",
                       "reason": row.get("reason", row.get("reject_reason", row.get("skip_reason", row.get("error", "frontend quality gate")))),
                       "latency_elapsed_s": elapsed})
    return sorted(result, key=lambda row: row["frame_index"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-dir", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sequence-dir", type=Path)
    source.add_argument("--predictions", type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = (sequence_predictions(args.sequence_dir.resolve()) if args.sequence_dir else
                   [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()])
    predictions = [row for row in predictions if row.get("phase", "motion") == "motion"]
    summary = evaluate_predictions(args.recording_dir, predictions, method=args.method)
    if args.sequence_dir:
        summary["latency_measurement_sources"] = sorted({row.get("latency_measurement_source", "rejection") for row in predictions})
        summary["latency_measurement_policy"] = (
            "Prefer recorded whole-moment wall time (includes RGB when enabled); otherwise explicit algorithm time or complete component sum. "
            "Moment wall excludes source image decode/normalization, cold initialization and later visualization/publication. Not live end-to-end latency."
        )
        if set(summary["latency_measurement_sources"]).issubset({"latency_s", "total_elapsed_s", "rejection"}):
            for key in ("steady_state_latency", "successful_frame_latency"):
                if key in summary:
                    summary[key]["timing_scope"] = (
                        "Whole moment from synchronized RGB ready through reconstruction; includes RGB inference, fusion and in-function output writes. "
                        "Excludes source decode/normalization, cold initialization, later overlays and publication."
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("per_frame", "rejected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

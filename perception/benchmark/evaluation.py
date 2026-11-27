"""Compare native SMPL-X joints with actual UE bones and measure throughput.

No ground-truth pose/shape is passed to a reconstruction backend. Metrics are
absolute and pelvis-relative joint errors, without Procrustes alignment.
Surface PVE is deliberately absent: recordings contain UE bones, not a
ground-truth skinned vertex array.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def evaluate_predictions(recording_dir: Path, predictions: list[dict], *, method: str) -> dict:
    root = recording_dir.resolve()
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    by_frame = {int(row["frame_index"]): row for row in manifest if row.get("status") == "ok"}
    errors, latency, successful_latency, rejected = [], [], [], []
    for prediction in predictions:
        frame = int(prediction["frame_index"])
        row = by_frame.get(frame)
        if row is None:
            raise ValueError(f"Prediction frame {frame} has no recording GT")
        elapsed = prediction.get("latency_elapsed_s")
        if elapsed is not None:
            elapsed = float(elapsed)
            if not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Latency must be a finite positive measured wall time")
            latency.append(elapsed)
        if prediction.get("status", "ok") != "ok":
            rejected.append({"frame_index": frame, "reason": prediction.get("reason")})
            continue
        if elapsed is not None:
            successful_latency.append(elapsed)
        gt = json.loads((root / row["gt_path"]).read_text())
        predicted = np.asarray(prediction["joints_world_m"], dtype=np.float64)
        if predicted.shape != (55, 3) or not np.all(np.isfinite(predicted)):
            raise ValueError(f"Prediction {frame} must contain finite native SMPL-X55 joints")
        bones = {b["joint_name"]: b for b in gt["bones"]}
        if gt["missing_joints"]:
            raise ValueError("Cannot score a recording with missing GT joints")
        target = np.array([bones[name]["world_location_cm"] for name in gt["joint_names"]], dtype=np.float64)
        if target.shape != (55, 3):
            raise ValueError("Recording GT must contain the complete ordered SMPL-X55 skeleton")
        target *= np.array([0.01, -0.01, 0.01])  # UE LH centimetres -> Genesis RH metres
        world = np.linalg.norm(predicted - target, axis=1) * 1000
        relative = np.linalg.norm((predicted - predicted[0]) - (target - target[0]), axis=1) * 1000
        errors.append({"frame_index": frame, "world_body22_mpjpe_mm": float(world[:22].mean()),
                       "pelvis_relative_body22_mpjpe_mm": float(relative[:22].mean()),
                       "world_hands30_mpjpe_mm": float(world[25:].mean()),
                       "pelvis_error_mm": float(world[0])})
    summary = {"method": method, "recording_dir": str(root), "frames_requested": len(predictions),
               "frames_scored": len(errors), "frames_rejected": len(rejected),
               "accuracy_reference": "actual UE native bones; no Procrustes alignment; no surface PVE",
               "per_frame": errors, "rejected": rejected}
    if errors:
        summary["accuracy"] = {key: float(np.mean([row[key] for row in errors]))
                               for key in errors[0] if key != "frame_index"}
    if latency:
        mean, median, p95 = np.mean(latency), np.median(latency), np.percentile(latency, 95)
        summary["steady_state_latency"] = {
            "samples": len(latency), "mean_s": float(mean), "median_s": float(median), "p95_s": float(p95),
            "latency_complete": len(latency) == len(predictions),
            "latency_coverage_fraction": float(len(latency) / len(predictions)),
            "mean_throughput_hz": float(1.0 / mean),
            "p95_meets_2hz": bool(len(latency) == len(predictions) and p95 <= 0.5),
            "successful_output_hz": (float(len(errors) / np.sum(latency))
                                     if len(latency) == len(predictions) else None),
            "success_fraction": float(len(errors) / len(predictions)) if predictions else 0.0,
            "throughput_definition": "accepted frames / summed measured compute latency; not source-time output cadence",
            "timing_scope": "backend-defined measured preprocessing/inference/fusion; exclude cold init and file writes",
        }
    if successful_latency:
        summary["successful_frame_latency"] = {
            "samples": len(successful_latency),
            "mean_s": float(np.mean(successful_latency)),
            "p95_s": float(np.percentile(successful_latency, 95)),
            "p95_meets_2hz": bool(np.percentile(successful_latency, 95) <= 0.5),
            "timing_scope": "successful reconstructions only; coverage is reported separately",
        }
    return summary

#!/usr/bin/env python3
"""Benchmark SMPL-X refinement with saved DWPose and optional HybrIK-X RGB priors."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from perception.benchmark.hybrik_backend import InitializedHybrIKX
from perception.benchmark.orientation_constraints import orientation_policy
from perception.benchmark.evaluation import evaluate_predictions
from perception.benchmark.rgb_twist_fusion import fuse_twist_predictions


def load_rgb_cache_frame(cache_root: Path, row: dict | None):
    if row is None or row.get("status") not in {"ok", "partial"} or not row.get("npz_path"):
        return None, None
    with np.load(cache_root / row["npz_path"], allow_pickle=False) as data:
        if int(data["frame_index"]) != int(row["frame_index"]):
            raise ValueError("RGB cache frame index mismatch")
        values = {key: np.array(data[key]) for key in ("pred_phi", "pred_rot_mats")}
        timing_keys = ("rgb_decode_elapsed_s", "crop_elapsed_s", "rgb_forward_elapsed_s")
        elapsed = sum(float(np.sum(data[key])) for key in timing_keys)
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid RGB cache timings")
        return values, elapsed


def load_points(moment: Path, *, preserved: Path | None = None) -> np.ndarray:
    with np.load(preserved or moment / "triangulation.npz") as data:
        for key in ("bodyhand3d", "points67"):
            if key in data:
                points = np.asarray(data[key], dtype=np.float32)
                return points.reshape(-1, 67, 4)[0]
        body = np.asarray(data["keypoints3d"], dtype=np.float32)
        if body.shape == (67, 4):
            return body
        return np.vstack([body.reshape(25, 4), data["handl3d"].reshape(21, 4), data["handr3d"].reshape(21, 4)])


def frontend_rejection_reason(row: dict) -> str | None:
    """A backend ablation cannot override the recorded observation quality gate."""
    if row.get("skip_reason") in {"bodyhand3d_quality", "pose2d_quality"}:
        return str(row["skip_reason"])
    for key in ("pose2d_quality", "bodyhand3d_quality"):
        quality = row.get(key)
        if isinstance(quality, dict) and quality.get("ok") is False:
            return key
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--recording-dir", type=Path, required=True)
    parser.add_argument("--init-npz", type=Path, required=True, help="RGB-fitted warmup SMPL-X; never use GT betas")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--gender", choices=["male", "female", "neutral"], default="male",
                        help="Must match the RGB-fitted initialization model")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--pose-initialization", choices=["hybrik", "previous"], default="hybrik",
                        help="Pose seed for refinement; previous requires --refine-steps > 0")
    parser.add_argument("--temporal-weight", type=float, default=0.0,
                        help="SO(3) chordal prior weight for refinement (default: 0)")
    parser.add_argument("--orientation-weight", type=float, default=0.0,
                        help="Optional normalized palm/foot/head direction weighting of existing landmarks (default: 0)")
    parser.add_argument("--rgb-cache-dir", type=Path)
    parser.add_argument("--rgb-hint", choices=["twist", "local-pose"])
    parser.add_argument("--rgb-pose-weight", type=float, default=0.0)
    parser.add_argument("--observation-geometry-filter", action="store_true",
                        help="Reject impossible hand observations using frozen rest-shape reach")
    parser.add_argument("--occlusion-rotation-weight", type=float, default=0.0)
    args = parser.parse_args()
    if bool(args.rgb_cache_dir) != bool(args.rgb_hint):
        parser.error("--rgb-cache-dir and --rgb-hint must be used together")
    if args.rgb_hint == "twist" and args.pose_initialization != "hybrik":
        parser.error("RGB twist is used only by analytical IK; select --pose-initialization hybrik")
    if (args.rgb_hint == "local-pose") != bool(args.rgb_pose_weight):
        parser.error("--rgb-hint local-pose requires positive --rgb-pose-weight; other modes require zero")
    root = args.sequence_dir.resolve()
    index = json.loads((root / "index.json").read_text())
    rows = list(index.get("index", [])) + list(index.get("rejected_index", []))
    rows.sort(key=lambda row: (int(row.get("timestamp_ns", 0)), int(row.get("frame_index", 0))))
    rgb_rows = {}
    if args.rgb_cache_dir:
        rgb_index = json.loads((args.rgb_cache_dir / "index.json").read_text())
        if Path(rgb_index["recording_dir"]).resolve() != args.recording_dir.resolve():
            raise ValueError("RGB cache belongs to a different recording")
        for row in rgb_index["index"]:
            if int(row["frame_index"]) in rgb_rows:
                raise ValueError("Duplicate frame in RGB cache")
            rgb_rows[int(row["frame_index"])] = row
    model = InitializedHybrIKX(
        args.init_npz, device=args.device, gender=args.gender,
        refine_steps=args.refine_steps, pose_initialization=args.pose_initialization,
        temporal_weight=args.temporal_weight,
        orientation_weight=args.orientation_weight,
        rgb_pose_weight=args.rgb_pose_weight,
        observation_geometry_filter=args.observation_geometry_filter,
        occlusion_rotation_weight=args.occlusion_rotation_weight,
    )
    twist_fallback = model.phis[0].detach().cpu().numpy().copy()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions, exported = [], []
    for row in rows:
        if args.max_frames and len(predictions) >= args.max_frames:
            break
        frame = int(row["frame_index"])
        rejection = frontend_rejection_reason(row)
        if rejection:
            predictions.append({"frame_index": frame, "status": "rejected", "reason": rejection,
                                "source_frontend_rejection_preserved": True,
                                "latency_elapsed_s": row.get("frontend_elapsed_s", row.get("frontend_wall_s"))})
            continue
        moment = Path(row["moment_dir"])
        if not moment.is_absolute():
            moment = root / moment
        preserved = row.get("triangulation_npz_preserved")
        preserved = root / preserved if preserved else None
        if not (preserved or moment / "triangulation.npz").is_file():
            predictions.append({"frame_index": frame, "status": "rejected",
                                "reason": "saved_triangulation_missing"})
            continue
        timestamp_value = row.get("timestamp_ns")
        timestamp_ns = int(timestamp_value) if timestamp_value is not None else None
        target = args.output_dir / f"moment_{len(exported):04d}"
        try:
            hints, rgb_meta, rgb_elapsed, fusion_elapsed = {}, {}, None, 0.0
            if args.rgb_cache_dir:
                rgb_row = rgb_rows.get(frame)
                if rgb_row and int(rgb_row["timestamp_ns"]) != timestamp_ns:
                    raise ValueError("RGB cache timestamp mismatch")
                values, rgb_elapsed = load_rgb_cache_frame(args.rgb_cache_dir, rgb_row)
                rgb_meta["rgb_cache_status"] = "missing" if rgb_row is None else rgb_row["status"]
                started = time.perf_counter()
                if values is not None and args.rgb_hint == "twist":
                    phis, mask, consensus = fuse_twist_predictions(values["pred_phi"], twist_fallback)
                    hints["rgb_phis"] = phis
                    rgb_meta.update(rgb_accepted_mask=mask.tolist(), rgb_consensus=consensus.tolist())
                elif values is not None and args.rgb_hint == "local-pose":
                    from perception.benchmark.rgb_pose_fusion import fuse_local_rotation_predictions
                    rotations, mask, consensus = fuse_local_rotation_predictions(values["pred_rot_mats"])
                    hints.update(rgb_rotations=rotations, rgb_agreement=consensus)
                    rgb_meta.update(rgb_accepted_mask=mask.tolist(), rgb_consensus=consensus.tolist())
                fusion_elapsed = time.perf_counter() - started
            result = model.step(load_points(moment, preserved=preserved), timestamp_ns=timestamp_ns, **hints)
            if "rgb_phis" in hints:
                twist_fallback = hints["rgb_phis"].copy()  # accepted poses only
            frontend = row.get("frontend_elapsed_s", row.get("frontend_wall_s"))
            frontend = float(frontend) if frontend is not None else None
            elapsed = frontend + result["backend_elapsed_s"] if frontend is not None else None
            if args.rgb_cache_dir:
                elapsed = elapsed + rgb_elapsed + fusion_elapsed if elapsed is not None and rgb_elapsed is not None else None
            prediction = {"frame_index": frame, "status": "ok", "joints_world_m": result["joints_world_m"].tolist(),
                          "timestamp_ns": timestamp_ns, "latency_elapsed_s": elapsed,
                          "backend_elapsed_s": result["backend_elapsed_s"],
                          "frontend_elapsed_s": frontend, "observed_points": result["observed_points"],
                          "pose_initialization": args.pose_initialization,
                          "temporal_weight": float(args.temporal_weight)}
            prediction["orientation_weight"] = float(args.orientation_weight)
            prediction["orientation_observed_group_mask"] = result["orientation_observed_group_mask"].tolist()
            if args.rgb_cache_dir:
                prediction.update(rgb_meta, rgb_elapsed_s=rgb_elapsed, rgb_fusion_elapsed_s=fusion_elapsed)
                result["rgb_hint_metadata"] = json.dumps(rgb_meta)
                result.update({key: value for key, value in hints.items()})
            target.mkdir()
            np.savez_compressed(target / "smplx_result.npz", **{key: value for key, value in result.items()
                                 if key not in ("backend_elapsed_s", "observed_points")})
            exported.append({"moment_dir": str(target.resolve()), "frame_index": frame,
                             "timestamp_ns": int(row.get("timestamp_ns", 0)), "fit_ok": True})
        except ValueError as exc:
            prediction = {"frame_index": frame, "status": "rejected", "reason": str(exc)}
        predictions.append(prediction)
    if not predictions:
        raise ValueError("Sequence has no saved triangulation.npz; run updated DWPose sequence frontend first")
    method = ("DWPose + temporal SMPL-X refinement" if args.pose_initialization == "previous"
              else "DWPose + initialized analytical HybrIK-X")
    if args.rgb_hint:
        method += f" + official HybrIK-X RGB {args.rgb_hint} consensus"
    summary = evaluate_predictions(args.recording_dir, predictions, method=method)
    summary["latency_measurement_policy"] = (
        "Backend timed in this replay; frontend timings reused from the source sequence. "
        "Their sum is a component estimate, not a fresh end-to-end or live measurement."
    )
    summary["frontend_gate_policy"] = "Preserve source RGB/triangulation rejections; count every requested source row."
    summary["initialization_npz"] = str(args.init_npz.resolve())
    summary["gender"] = args.gender
    summary["face_policy"] = "fixed jaw/eyes/expression; no face iterations"
    summary["pose_initialization"] = args.pose_initialization
    summary["temporal_weight"] = float(args.temporal_weight)
    summary["orientation_weight"] = float(args.orientation_weight)
    summary["rgb_hint"] = args.rgb_hint
    summary["rgb_pose_weight"] = args.rgb_pose_weight
    summary["observation_geometry_filter"] = args.observation_geometry_filter
    summary["occlusion_rotation_weight"] = args.occlusion_rotation_weight
    if args.rgb_cache_dir:
        summary["rgb_cache_dir"] = str(args.rgb_cache_dir.resolve())
        summary["rgb_model"] = rgb_index.get("rgb_model")
        summary["rgb_policy"] = (
            "Official RGB per-view predictions; fixed fitted beta; no GT used in fusion. "
            "Cross-view agreement is consistency, not calibrated confidence. "
            "Missing hints fall back to the existing tracker and are recorded. "
            "Twist disagreement uses the previous accepted twist (initial static twist at start). "
            "Local rotations exclude global root and fixed jaw/eyes."
        )
        summary["latency_measurement_policy"] += " RGB decode/crop/full forward reused from cache, fusion timed here."
    summary["orientation_policy"] = orientation_policy()
    summary["orientation_policy"]["enabled"] = bool(args.orientation_weight)
    summary["temporal_reference_dt_s"] = 1.0 / 30.0
    summary["twist_policy"] = (
        "previous accepted pose + landmark refinement; no per-frame analytical IK reset"
        if args.pose_initialization == "previous"
        else "fixed RGB initialization; no pretrained RGB twist network"
    )
    if args.rgb_hint == "twist":
        summary["twist_policy"] = "per-frame official RGB twist circular consensus + analytical IK + refinement"
    elif args.rgb_hint == "local-pose":
        summary["twist_policy"] += "; official RGB full local pose consensus prior"
    summary["refine_steps"] = args.refine_steps
    summary["refine_loss_policy"] = (
        "linear confidence; body2 hand2; Huber delta 0.03m; fixed shape/face; "
        "SO(3) chordal temporal prior normalized to nominal 30Hz dt"
        if args.temporal_weight
        else "linear confidence; body2 hand2; Huber delta 0.03m; fixed shape/face"
    )
    if args.orientation_weight:
        summary["refine_loss_policy"] += "; normalized three-axis palm/foot/head direction error from eligible existing observations"
    summary["complete_pipeline_latency_available"] = False
    summary["component_latency_sum_available"] = all(p.get("latency_elapsed_s") is not None for p in predictions if p.get("status") == "ok")
    summary["backend_latency_mean_s"] = float(np.mean([p["backend_elapsed_s"] for p in predictions if p.get("status") == "ok"])) if exported else None
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions))
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "index.json").write_text(json.dumps({"index": exported}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("per_frame", "rejected")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

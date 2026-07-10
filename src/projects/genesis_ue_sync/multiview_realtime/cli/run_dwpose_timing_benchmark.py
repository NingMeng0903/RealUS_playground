#!/usr/bin/env python3
"""Benchmark YOLO vs pose ONNX latency on live UE camera frames."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.tracking.camera_image_correction import correct_views_rgb_for_calibration
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--wait-timeout-s", type=float, default=30.0)
    p.add_argument("--output-json", type=Path, default=Path("outputs/tracking_debug/dwpose_timing.json"))
    return p.parse_args()


def _percentiles(arr: np.ndarray, ps: tuple[float, ...]) -> dict[str, float]:
    if arr.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    return {f"p{int(p)}": round(float(np.percentile(arr, p)), 3) for p in ps}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = MultiviewRealtimeConfig.load(args.config)
    if args.connect:
        ing = cfg.ingress
        cfg = MultiviewRealtimeConfig(
            calibration_path=cfg.calibration_path,
            scene_spec_path=cfg.scene_spec_path,
            camera_ids=cfg.camera_ids,
            primary_camera_id=cfg.primary_camera_id,
            ingress=type(ing)(
                connect=str(args.connect),
                topic=ing.topic,
                recv_timeout_ms=ing.recv_timeout_ms,
                sync_tolerance_frames=ing.sync_tolerance_frames,
                max_buffer_per_camera=ing.max_buffer_per_camera,
            ),
            pose_backend=cfg.pose_backend,
            world_reconstruction=cfg.world_reconstruction,
            robot_kinematic_mask=cfg.robot_kinematic_mask,
            genesis=cfg.genesis,
        )

    calibration = load_calibration_bundle(cfg.calibration_path, scene_spec_path=cfg.scene_spec_path)
    camera_ids = list(cfg.camera_ids)
    dw_cfg = DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {}))
    detector = DwposeOnnxDetector(dw_cfg)
    detector.preload()

    stream = MultiviewCameraStream(cfg.ingress, camera_ids=camera_ids)
    stream.connect()
    stream.start_ingest()

    mode = str(cfg.pose_backend.get("image_correction_mode") or "ingress")
    overrides = {str(k): dict(v or {}) for k, v in (cfg.pose_backend.get("image_correction_overrides") or {}).items()}

    synced = None
    t_wait = time.time()
    while synced is None and time.time() - t_wait < float(args.wait_timeout_s):
        synced = stream.pop_latest_synced()
        if synced is None:
            time.sleep(0.01)
    if synced is None:
        logging.error("no synced camera frame within %.1fs", args.wait_timeout_s)
        stream.close()
        return 1

    views_rgb, _ = correct_views_rgb_for_calibration(
        synced.views_rgb,
        calibration=calibration,
        camera_ids=camera_ids,
        mode=mode,
        overrides=overrides,
        metadata_by_camera=synced.metadata_by_camera,
    )

    views = {cid: np.asarray(views_rgb[cid], dtype=np.uint8) for cid in camera_ids}

    yolo_all: list[float] = []
    pose_all: list[float] = []
    total_all: list[float] = []
    per_cam: dict[str, dict[str, list[float]]] = {cid: {"yolo": [], "pose": [], "total": []} for cid in camera_ids}
    batch_wall: list[float] = []
    batch_yolo: list[float] = []
    batch_pose: list[float] = []

    n_runs = int(args.warmup) + int(args.iters)
    for run in range(n_runs):
        for cid in camera_ids:
            _, diag = detector.infer_body25(views[cid])
            tms = dict(diag.get("timing_ms") or {})
            y = float(tms.get("yolo_det_ms", float("nan")))
            p = float(tms.get("pose_onnx_ms", float("nan")))
            tot = float(tms.get("total_ms", float("nan")))
            if run >= int(args.warmup):
                yolo_all.append(y)
                pose_all.append(p)
                total_all.append(tot)
                per_cam[cid]["yolo"].append(y)
                per_cam[cid]["pose"].append(p)
                per_cam[cid]["total"].append(tot)

        t0 = time.perf_counter()
        _, _, batch_timing = detector.infer_body25_multiview(views, camera_ids)
        wall = (time.perf_counter() - t0) * 1000.0
        if run >= int(args.warmup):
            batch_wall.append(wall)
            batch_yolo.append(float(batch_timing.get("yolo_det_ms_total", float("nan"))))
            batch_pose.append(float(batch_timing.get("pose_onnx_ms_batch", float("nan"))))

    def _stats(vals: list[float]) -> dict[str, float | dict[str, float]]:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "min_ms": round(float(np.min(arr)), 3),
            "max_ms": round(float(np.max(arr)), 3),
            **_percentiles(arr, (50, 90, 95)),
        }

    multiview_yolo = float(np.sum([np.mean(per_cam[c]["yolo"]) for c in camera_ids]))
    multiview_pose = float(np.sum([np.mean(per_cam[c]["pose"]) for c in camera_ids]))
    multiview_total = float(np.sum([np.mean(per_cam[c]["total"]) for c in camera_ids]))

    report = {
        "frame_index": int(synced.frame_index),
        "camera_ids": camera_ids,
        "dwpose": {
            "detector_onnx": str(dw_cfg.detector_onnx_path),
            "pose_onnx": str(dw_cfg.pose_onnx_path),
        },
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "single_view_ms": {
            "yolo_det": _stats(yolo_all),
            "pose_onnx": _stats(pose_all),
            "total_infer_body25": _stats(total_all),
        },
        "per_camera_mean_ms": {
            cid: {
                "yolo_det": round(float(np.mean(per_cam[cid]["yolo"])), 3),
                "pose_onnx": round(float(np.mean(per_cam[cid]["pose"])), 3),
                "total": round(float(np.mean(per_cam[cid]["total"])), 3),
            }
            for cid in camera_ids
        },
        "multiview_triplet_ms": {
            "mode": "serial_per_camera",
            "yolo_det_total": round(multiview_yolo, 3),
            "pose_onnx_total": round(multiview_pose, 3),
            "infer_body25_total": round(multiview_total, 3),
            "note": "Legacy loop: infer_body25() once per camera.",
        },
        "multiview_batched_pose_ms": {
            "mode": "batched_yolo_pose",
            "wall_ms": _stats(batch_wall),
            "yolo_det_total": _stats(batch_yolo),
            "pose_onnx_batch": _stats(batch_pose),
            "note": "YOLO + pose ONNX each run once with batch=N when dynbatch/TRT exports are enabled.",
        },
    }

    out_path = project_paths(__file__).resolve_from_root(str(args.output_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    stream.close()
    detector.close()

    logging.info(
        "single-view mean: yolo=%.2fms pose=%.2fms total=%.2fms",
        report["single_view_ms"]["yolo_det"]["mean_ms"],
        report["single_view_ms"]["pose_onnx"]["mean_ms"],
        report["single_view_ms"]["total_infer_body25"]["mean_ms"],
    )
    logging.info(
        "serial triplet mean: yolo=%.2fms pose=%.2fms total=%.2fms",
        multiview_yolo,
        multiview_pose,
        multiview_total,
    )
    logging.info(
        "batched triplet mean: wall=%.2fms yolo=%.2fms pose_batch=%.2fms -> %s",
        report["multiview_batched_pose_ms"]["wall_ms"]["mean_ms"],
        report["multiview_batched_pose_ms"]["yolo_det_total"]["mean_ms"],
        report["multiview_batched_pose_ms"]["pose_onnx_batch"]["mean_ms"],
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

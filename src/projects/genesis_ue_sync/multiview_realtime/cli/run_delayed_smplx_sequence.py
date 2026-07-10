#!/usr/bin/env python3
"""Sample UE multiview at a fixed interval; offline EasyMocap SMPL-X per moment.

No terminal 8 / Genesis. Each moment writes UE RGB + 2D skeleton + 3D repro + SMPL-X mesh.
"""

from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.easymocap.moment_pipeline import (
    _jsonable,
    load_fixed_betas,
    process_one_moment,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.ingress.motion_frame_gate import (
    CanonicalMotionIndexClient,
    motion_window_from_scene_spec,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.synced_frame_acquire import wait_pop_next_synced
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.dwpose_onnx import DwposeOnnxConfig, DwposeOnnxDetector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None)
    p.add_argument("--output-root", type=Path, default=Path("outputs/offline_compare"))
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--interval-s", type=float, default=0.5, help="Minimum wall time between captured moments.")
    p.add_argument("--duration-s", type=float, default=30.0, help="Total capture window (0 = until --max-moments).")
    p.add_argument("--max-moments", type=int, default=0, help="Stop after N moments (0 = only duration).")
    p.add_argument("--sync-wait-s", type=float, default=2.0, help="Max wait for next synced frame each tick.")
    p.add_argument("--gender", type=str, default="male", choices=["male", "female", "neutral"])
    p.add_argument(
        "--fit-model",
        type=str,
        default="smplx",
        choices=["smplh", "smplx"],
        help="EasyMocap body model (smplx male; needs SMPLX_MALE.pkl).",
    )
    p.add_argument("--thres2d", type=float, default=0.15)
    p.add_argument("--max-repro-error", type=float, default=50.0)
    p.add_argument("--mesh-alpha", type=float, default=0.82)
    p.add_argument("--mesh-rgb", type=str, default="255,128,32")
    p.add_argument("--face-stride", type=int, default=1)
    p.add_argument("--max-triangle-px", type=float, default=520.0)
    p.add_argument("--skip-fit", action="store_true")
    p.add_argument(
        "--betas-path",
        type=Path,
        default=None,
        help="Fixed SMPL betas (.npy/.json). Skips per-frame shape optimization when set.",
    )
    p.add_argument("--bed-sdf", action="store_true", help="Refine fit with bed-plane SDF anti-penetration loss.")
    p.add_argument("--scene-spec-path", type=Path, default=None, help="Scene yaml for bed top_z (defaults to config).")
    p.add_argument(
        "--motion-fraction",
        type=float,
        default=None,
        help="Optional canonical motion index window (first fraction of scene clip). Default: no motion gate, 2D quality only.",
    )
    p.add_argument("--motion-end", type=int, default=None, help="Exclusive motion_frame_index cap (overrides --motion-fraction).")
    p.add_argument("--canonical-connect", type=str, default=None, help="Genesis canonical ZMQ for motion_frame_index gating.")
    p.add_argument("--keep-rejected", action="store_true", help="Keep rejected/failed moment folders for debugging.")
    p.add_argument(
        "--max-output-frame-span",
        type=int,
        default=2,
        help="Max allowed frame-index spread across cameras in an output moment.",
    )
    p.add_argument(
        "--sync-tolerance-frames",
        type=int,
        default=16,
        help="Max frame index spread across cameras for one synced moment (6-cam offline needs larger than live 2).",
    )
    return p.parse_args()


def _parse_rgb(raw: str) -> tuple[int, int, int]:
    parts = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected RGB as r,g,b, got: {raw}")
    return tuple(max(0, min(255, v)) for v in parts)  # type: ignore[return-value]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    mesh_rgb = _parse_rgb(args.mesh_rgb)
    cfg = MultiviewRealtimeConfig.load(args.config)
    ingress = type(cfg.ingress)(
        connect=str(args.connect or cfg.ingress.connect),
        topic=cfg.ingress.topic,
        recv_timeout_ms=cfg.ingress.recv_timeout_ms,
        sync_tolerance_frames=max(int(args.sync_tolerance_frames), int(cfg.ingress.sync_tolerance_frames)),
        max_buffer_per_camera=max(16, int(cfg.ingress.max_buffer_per_camera)),
    )
    cfg = MultiviewRealtimeConfig(
        calibration_path=cfg.calibration_path,
        scene_spec_path=cfg.scene_spec_path,
        camera_ids=cfg.camera_ids,
        primary_camera_id=cfg.primary_camera_id,
        ingress=ingress,
        pose_backend=cfg.pose_backend,
        world_reconstruction=cfg.world_reconstruction,
        robot_kinematic_mask=cfg.robot_kinematic_mask,
        genesis=cfg.genesis,
    )

    run_name = str(args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_root = project_paths(__file__).resolve_from_root(str(args.output_root))
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration_bundle(project_paths(__file__).resolve_from_root(cfg.calibration_path))
    detector = DwposeOnnxDetector(DwposeOnnxConfig.from_dict(dict(cfg.pose_backend.get("dwpose") or {})))
    body_model_cache: dict[str, Any] = {}
    fixed_betas = load_fixed_betas(args.betas_path) if args.betas_path else None
    scene_spec_path = (
        project_paths(__file__).resolve_from_root(str(args.scene_spec_path))
        if args.scene_spec_path
        else cfg.scene_spec_path
    )

    motion_window = None
    if args.motion_fraction is not None or args.motion_end is not None:
        motion_window = motion_window_from_scene_spec(
            scene_spec_path,
            fraction=float(args.motion_fraction if args.motion_fraction is not None else 0.5),
            motion_end_exclusive=args.motion_end,
        )
    canonical_connect = str(
        args.canonical_connect
        or (cfg.robot_kinematic_mask or {}).get("canonical_connect")
        or "tcp://127.0.0.1:5599"
    )
    canonical = CanonicalMotionIndexClient(connect=canonical_connect)
    canonical.open()

    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)
    stream.connect()
    stream.start_ingest()

    interval_s = max(0.05, float(args.interval_s))
    duration_s = float(args.duration_s)
    max_moments = int(args.max_moments)
    deadline = time.perf_counter() + duration_s if duration_s > 0.0 else None

    index: list[dict[str, Any]] = []
    rejected_index: list[dict[str, Any]] = []
    moment_id = 0
    attempts = 0
    next_capture_at = time.perf_counter()

    if motion_window is not None:
        logging.info(
            "sequence capture on %s interval=%.2fs max_moments=%d motion_window=[%d,%d) + pose2d quality",
            cfg.ingress.connect,
            interval_s,
            max_moments,
            motion_window.motion_start,
            motion_window.motion_end_exclusive,
        )
    else:
        logging.info(
            "sequence capture on %s interval=%.2fs max_moments=%d (FIFO + pose2d quality only)",
            cfg.ingress.connect,
            interval_s,
            max_moments,
        )

    try:
        detector.preload()
        while True:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if max_moments > 0 and moment_id >= max_moments:
                break

            now = time.perf_counter()
            if now < next_capture_at:
                stream.poll_once()
                time.sleep(min(0.01, next_capture_at - now))
                continue

            synced, motion_fi, skip_reason = wait_pop_next_synced(
                stream,
                motion_window=motion_window,
                canonical=canonical,
                wait_timeout_s=float(args.sync_wait_s),
                max_frame_span=int(args.max_output_frame_span),
            )
            next_capture_at = time.perf_counter() + interval_s
            if synced is None:
                logging.warning(
                    "no accepted synced frame at moment %d (%s) buffer=%s",
                    moment_id,
                    skip_reason,
                    stream.buffer_status(),
                )
                continue

            attempts += 1
            moment_dir = run_dir / f"moment_{moment_id:04d}"
            logging.info(
                "moment %d ue_frame=%s motion_fi=%s -> %s",
                moment_id,
                synced.frame_index,
                motion_fi,
                moment_dir.name,
            )
            try:
                summary = process_one_moment(
                    moment_dir=moment_dir,
                    synced=synced,
                    cfg=cfg,
                    calibration=calibration,
                    detector=detector,
                    camera_ids=list(cfg.camera_ids),
                    gender=str(args.gender),
                    fit_model=str(args.fit_model),
                    thres2d=float(args.thres2d),
                    max_repro_error=float(args.max_repro_error),
                    mesh_alpha=float(args.mesh_alpha),
                    mesh_rgb=mesh_rgb,
                    face_stride=int(args.face_stride),
                    max_triangle_px=float(args.max_triangle_px),
                    body_model_cache=body_model_cache,
                    skip_fit=bool(args.skip_fit),
                    fixed_betas=fixed_betas,
                    bed_sdf=bool(args.bed_sdf),
                    scene_spec_path=scene_spec_path,
                    motion_frame_index=motion_fi,
                    keep_rejected=bool(args.keep_rejected),
                )
                summary["moment_id"] = int(moment_id)
                summary["attempt_id"] = int(attempts - 1)
                summary["motion_frame_index"] = int(motion_fi) if motion_fi is not None else None
                if summary.get("skip_reason") or not summary.get("fit_ok"):
                    rejected_index.append(summary)
                    if not bool(args.keep_rejected):
                        shutil.rmtree(moment_dir, ignore_errors=True)
                    logging.info(
                        "rejected attempt=%d ue_frame=%s reason=%s fit_ok=%s",
                        attempts - 1,
                        synced.frame_index,
                        summary.get("skip_reason") or summary.get("fit_error") or "fit_failed",
                        summary.get("fit_ok"),
                    )
                    continue
                index.append(summary)
                moment_id += 1
            except Exception as exc:
                logging.exception("moment %d failed: %s", moment_id, exc)
                rejected_index.append(
                    {"moment_id": moment_id, "attempt_id": attempts - 1, "fit_ok": False, "error": str(exc)}
                )
                if not bool(args.keep_rejected):
                    shutil.rmtree(moment_dir, ignore_errors=True)
    finally:
        canonical.close()
        detector.close()
        stream.close()

    run_summary = {
        "run_dir": str(run_dir.resolve()),
        "moments_attempted": int(attempts),
        "moments_produced": int(moment_id),
        "moments_ok": int(sum(1 for row in index if row.get("fit_ok"))),
        "interval_s": interval_s,
        "duration_s": duration_s,
        "bed_sdf": bool(args.bed_sdf),
        "max_output_frame_span": int(args.max_output_frame_span),
        "betas_path": str(args.betas_path) if args.betas_path else None,
        "fixed_betas": [float(v) for v in fixed_betas.tolist()] if fixed_betas is not None else None,
        "motion_window": (
            {
                "motion_start": int(motion_window.motion_start),
                "motion_end_exclusive": int(motion_window.motion_end_exclusive),
            }
            if motion_window is not None
            else None
        ),
        "index": index,
        "rejected_index": rejected_index,
    }
    (run_dir / "index.json").write_text(json.dumps(_jsonable(run_summary), ensure_ascii=True, indent=2), encoding="utf-8")
    logging.info(
        "done: %d produced, %d ok, %d rejected -> %s",
        moment_id,
        run_summary["moments_ok"],
        len(rejected_index),
        run_dir,
    )
    return 0 if run_summary["moments_ok"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

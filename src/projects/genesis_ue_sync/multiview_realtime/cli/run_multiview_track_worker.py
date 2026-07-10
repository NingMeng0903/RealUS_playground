#!/usr/bin/env python3
"""Dedicated process: camera ZMQ -> multiview pose backend -> track pose ZMQ.

Run in its own terminal (GPU). Genesis amass_bed_capsule_demo only subscribes to poses when
AMONGUS_GENESIS_TRACK_SUBSCRIBE is set; it does not load the pose backend.

Example:

  PYTHONNOUSERSITE=1 PYTHONPATH=src python -m projects.genesis_ue_sync.multiview_realtime.cli.run_multiview_track_worker \\
    --config configs/tracking/multiview_realtime_dwpose_triangulation.yaml \\
    --pub-bind tcp://127.0.0.1:5598
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.egress.track_pose_publisher import TrackPosePublisher
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import MultiviewTrackerSession
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.track_stream import DEFAULT_TRACK_PUB_BIND
from common.project import project_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None, help="Override camera ingress ZMQ (default from yaml).")
    p.add_argument("--pub-bind", type=str, default=DEFAULT_TRACK_PUB_BIND)
    p.add_argument("--max-track-fps", type=float, default=0.0, help="Cap inference rate (0 = no cap).")
    p.add_argument(
        "--betas-amass-npz",
        type=Path,
        default=None,
        help="Debug fallback for SMPL betas from AMASS npz; smpl_fit.betas_path remains preferred.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    cfg = MultiviewRealtimeConfig.load(args.config)
    if args.connect:
        ingress = cfg.ingress
        cfg = MultiviewRealtimeConfig(
            calibration_path=cfg.calibration_path,
            scene_spec_path=cfg.scene_spec_path,
            camera_ids=cfg.camera_ids,
            primary_camera_id=cfg.primary_camera_id,
            ingress=type(ingress)(
                connect=str(args.connect),
                topic=ingress.topic,
                recv_timeout_ms=ingress.recv_timeout_ms,
                sync_tolerance_frames=ingress.sync_tolerance_frames,
                max_buffer_per_camera=ingress.max_buffer_per_camera,
            ),
            pose_backend=cfg.pose_backend,
            world_reconstruction=cfg.world_reconstruction,
            robot_kinematic_mask=cfg.robot_kinematic_mask,
            genesis=cfg.genesis,
        )

    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)
    session = MultiviewTrackerSession(cfg, betas_amass_npz_override=args.betas_amass_npz)
    smpl_fit_enabled = bool(dict(cfg.pose_backend.get("smpl_fit") or {}).get("enable", False))
    publisher = TrackPosePublisher(
        bind=str(args.pub_bind),
        publish_keypoints_fallback=not smpl_fit_enabled,
    )
    min_dt = 0.0
    max_fps = float(args.max_track_fps or cfg.genesis.max_track_fps or 0.0)
    if max_fps > 0.0:
        min_dt = 1.0 / max_fps
    infer_stride = max(1, int(cfg.genesis.inference_every_n_synced_frames))
    synced_seen = 0

    logging.info("multiview track worker: preload pose backend type=%s …", cfg.pose_backend.get("type", ""))
    session.preload()
    debug_on = bool(cfg.pose_backend.get("debug_tracking", False)) or str(
        __import__("os").environ.get("AMONGUS_CURSOR_DEBUG_TRACKING", "") or ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if debug_on:
        debug_root = project_paths(__file__).root / "outputs" / "tracking_debug" / "62415c"
        logging.info("debug tracking ON -> %s", debug_root)
    logging.info(
        "multiview pose live: backend=%s preprocess=%s translation=%s max_track_fps=%.1f robot_mask=%s",
        cfg.pose_backend.get("type", ""),
        cfg.pose_backend.get("preprocess", {}).get("mode", "native"),
        cfg.world_reconstruction.get("live_translation_mode", "scene_bed_anchor"),
        float(args.max_track_fps or cfg.genesis.max_track_fps or 0.0),
        bool(cfg.robot_kinematic_mask.get("enable", False)),
    )
    if smpl_fit_enabled:
        logging.info("track publisher: SMPL-only display payloads; keypoints3d capsule fallback disabled")
    mask_export = dict(cfg.robot_kinematic_mask.get("export") or {})
    if bool(cfg.robot_kinematic_mask.get("enable", False)) and bool(mask_export.get("enable", True)):
        export_root = mask_export.get("output_root", "outputs/tracking_debug/62415c/robot_mask")
        logging.info("robot_kinematic_mask export -> %s", export_root)
    logging.info("multiview track worker: waiting for synced cameras on %s", cfg.ingress.connect)

    stream.connect()
    stream.start_ingest()
    wait_start = time.perf_counter()
    last_wait_log = wait_start
    try:
        while True:
            synced = stream.pop_latest_synced()
            if synced is None:
                now = time.perf_counter()
                if now - last_wait_log >= 5.0:
                    logging.info(
                        "still waiting for synced cameras on %s status=%s (elapsed %.0fs)",
                        cfg.ingress.connect,
                        stream.buffer_status(),
                        now - wait_start,
                    )
                    last_wait_log = now
                time.sleep(0.002)
                continue
            synced_seen += 1
            if synced_seen == 1:
                logging.info(
                    "first synced multiview frame=%s (waited %.1fs)",
                    synced.frame_index,
                    time.perf_counter() - wait_start,
                )
            if synced_seen % infer_stride != 0:
                continue
            t0 = time.perf_counter()
            if synced_seen <= 3 or synced_seen % 30 == 0:
                logging.info("infer frame=%s (synced_seen=%s) …", synced.frame_index, synced_seen)
            track = session.track_synced_frame(synced)
            publisher.publish(track)
            dt = time.perf_counter() - t0
            recon = dict(getattr(track, "reconstruction", {}) or {})
            if dt >= 0.12 or synced_seen <= 3 or synced_seen % 30 == 0:
                timing = dict(recon.get("timing_s") or {})
                logging.info(
                    "published track frame=%s dt=%.3fs smpl_fit=%s held=%s init=%s cold=%s reason=%s rms3d_cm=%s root=%s iters=%s lbfgs=%s lbfgs_reason=%s body25=%s bones=%s vposer=%s timing=%s",
                    track.frame_index,
                    dt,
                    recon.get("smpl_fit_ok"),
                    recon.get("held"),
                    recon.get("init_source"),
                    recon.get("cold_start"),
                    recon.get("reason"),
                    round(float(recon.get("rms3d_m", float("nan"))) * 100.0, 2)
                    if recon.get("rms3d_m") is not None
                    else None,
                    recon.get("root_stage_steps"),
                    recon.get("iterations"),
                    recon.get("lbfgs_steps"),
                    recon.get("lbfgs_reason"),
                    dict(recon.get("body25_regressor") or {}).get("available"),
                    recon.get("n_bone_terms"),
                    dict(recon.get("vposer") or {}).get("available"),
                    {
                        key: round(float(timing.get(key, 0.0)), 3)
                        for key in ("ensure_model", "root_stage", "body_adam", "lbfgs", "final_forward", "total_fit")
                    }
                    if timing
                    else None,
                )
                if dt >= 0.25:
                    logging.warning(
                        "slow infer frame=%s dt=%.3fs timing=%s",
                        track.frame_index,
                        dt,
                        timing,
                    )
            if min_dt > 0.0:
                sleep_s = min_dt - (time.perf_counter() - t0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        logging.info("multiview track worker: stopped")
    finally:
        stream.close()
        session.close()
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

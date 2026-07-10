#!/usr/bin/env python3
"""Live robot kinematic mask preview; exports masked camera PNGs for alignment checks.

Example:

  PYTHONNOUSERSITE=1 PYTHONPATH=src python -m projects.genesis_ue_sync.multiview_realtime.cli.run_robot_kinematic_mask_preview \\
    --config configs/tracking/multiview_realtime_dwpose_triangulation.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream, SyncedMultiviewFrame
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.camera_image_correction import correct_views_rgb_for_calibration
from projects.genesis_ue_sync.tracking.robot_kinematic_mask import RobotKinematicMaskConfig, RobotKinematicMaskStage


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--connect", type=str, default=None, help="Override camera ingress ZMQ endpoint.")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N exported frames (0 = unlimited).")
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

    mask_cfg = RobotKinematicMaskConfig.from_dict(cfg.robot_kinematic_mask)
    mask_cfg = RobotKinematicMaskConfig(
        enable=True,
        fill_value=mask_cfg.fill_value,
        margin_px=mask_cfg.margin_px,
        face_stride=mask_cfg.face_stride,
        max_triangle_px=mask_cfg.max_triangle_px,
        canonical_connect=mask_cfg.canonical_connect,
        canonical_topic=mask_cfg.canonical_topic,
        robot_entity_name=mask_cfg.robot_entity_name,
        visual_basis_rpy_deg=mask_cfg.visual_basis_rpy_deg,
        fov_tolerance_deg=mask_cfg.fov_tolerance_deg,
        fx_tolerance_px=mask_cfg.fx_tolerance_px,
        export=mask_cfg.export,
    )

    calibration = load_calibration_bundle(cfg.calibration_path, scene_spec_path=cfg.scene_spec_path)
    stage = RobotKinematicMaskStage(calibration=calibration, config=mask_cfg)
    stream = MultiviewCameraStream(cfg.ingress, camera_ids=cfg.camera_ids)

    logging.info("robot kinematic mask preview: export_root=%s", stage.export_output_root)
    logging.info("waiting for synced cameras on %s", cfg.ingress.connect)

    stream.connect()
    exported = 0
    wait_start = time.perf_counter()
    last_wait_log = wait_start
    try:
        while True:
            stream.poll_once()
            synced = stream.try_pop_synced()
            if synced is None:
                now = time.perf_counter()
                if now - last_wait_log >= 5.0:
                    logging.info(
                        "still waiting status=%s (elapsed %.0fs)",
                        stream.buffer_status(),
                        now - wait_start,
                    )
                    last_wait_log = now
                time.sleep(0.002)
                continue
            t0 = time.perf_counter()
            views_rgb = synced.views_rgb
            if bool(mask_cfg.precorrect_views_rgb):
                views_rgb, _corrections = correct_views_rgb_for_calibration(
                    views_rgb,
                    calibration=calibration,
                    camera_ids=list(cfg.camera_ids),
                    mode=str(mask_cfg.image_correction_mode),
                )
                synced = SyncedMultiviewFrame(
                    frame_index=int(synced.frame_index),
                    views_rgb=views_rgb,
                    metadata_by_camera=synced.metadata_by_camera,
                    timestamp_ns=int(synced.timestamp_ns),
                )
            result = stage.apply(synced)
            dt = time.perf_counter() - t0
            if result.export_paths:
                exported += 1
                logging.info(
                    "exported frame=%s mask_pixels=%s dt=%.3fs",
                    synced.frame_index,
                    {cid: int((result.masks[cid] > 0).sum()) for cid in result.masks},
                    dt,
                )
            else:
                logging.info("masked frame=%s (export quota reached) dt=%.3fs", synced.frame_index, dt)
            if args.max_frames > 0 and exported >= int(args.max_frames):
                logging.info("reached max exported frames=%s", exported)
                break
    except KeyboardInterrupt:
        logging.info("robot kinematic mask preview: stopped")
    finally:
        stream.close()
        stage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

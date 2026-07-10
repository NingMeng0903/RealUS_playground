#!/usr/bin/env python3
"""Realtime three-view human tracking with a configurable pose backend and Genesis overlay.

Ingress defaults to ZMQ JPEG (amongus camera mux). The module name is camera-agnostic so
the same pipeline can subscribe to physical camera publishers later.

Example (Genesis viewer + ZMQ cameras from terminal 2 mux):

  PYTHONNOUSERSITE=1 PYTHONPATH=src python -m projects.genesis_ue_sync.multiview_realtime.cli.run_multiview_realtime_track \\
    --config configs/tracking/multiview_realtime_dwpose_triangulation.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.pipeline import MultiviewRealtimeTracker


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"),
        help="YAML config (calibration, pose backend, ingress, Genesis overlay).",
    )
    p.add_argument("--connect", type=str, default=None, help="Override ingress.connect ZMQ endpoint.")
    p.add_argument("--no-genesis", action="store_true", help="Run inference only (no Genesis viewer).")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N tracked frames (0 = unlimited).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
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
    if args.no_genesis:
        genesis = cfg.genesis
        cfg = MultiviewRealtimeConfig(
            calibration_path=cfg.calibration_path,
            scene_spec_path=cfg.scene_spec_path,
            camera_ids=cfg.camera_ids,
            primary_camera_id=cfg.primary_camera_id,
            ingress=cfg.ingress,
            pose_backend=cfg.pose_backend,
            world_reconstruction=cfg.world_reconstruction,
            robot_kinematic_mask=cfg.robot_kinematic_mask,
            genesis=type(genesis)(
                backend=genesis.backend,
                show_viewer=False,
                show_fps=False,
                spawn_bed=genesis.spawn_bed,
                spawn_robot=genesis.spawn_robot,
                track_mesh_rgba=genesis.track_mesh_rgba,
                inference_every_n_synced_frames=genesis.inference_every_n_synced_frames,
                max_track_fps=genesis.max_track_fps,
            ),
        )

    tracker = MultiviewRealtimeTracker(cfg)
    if args.max_frames <= 0:
        tracker.run()
        return 0

    if cfg.genesis.show_viewer:
        tracker.enable_genesis_overlay()
    tracker.stream.connect()
    n = 0
    try:
        for synced in tracker.stream.iter_synced():
            track = tracker.process_synced_frame(synced)
            if tracker.genesis_overlay is not None:
                tracker.genesis_overlay.draw_track_frame(track)
            n += 1
            if n >= int(args.max_frames):
                break
    finally:
        tracker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

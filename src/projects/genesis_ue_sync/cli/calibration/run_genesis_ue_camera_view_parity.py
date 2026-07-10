#!/usr/bin/env python3
"""Render Genesis three-view cameras and compare with UE to diagnose image-axis flips.

Genesis uses the SyncSceneSpec / calibration lookat basis (OpenCV). UE JPEGs come from
SceneCapture2D. This tool renders Genesis frame0 for each camera, pairs it with UE frames,
scores all four flip combinations, and writes side-by-side panels plus a JSON report.

Examples:

  # Live UE + Genesis render (requires genesis env + GPU):
  PYTHONNOUSERSITE=1 PYTHONPATH=src python -m projects.genesis_ue_sync.cli.calibration.run_genesis_ue_camera_view_parity \\
    --config configs/tracking/multiview_realtime_dwpose_triangulation.yaml \\
    --robot-model rm75_6f \\
    --ue-connect tcp://127.0.0.1:17356

  # Offline using exported robot_mask originals:
  PYTHONNOUSERSITE=1 PYTHONPATH=src python -m projects.genesis_ue_sync.cli.calibration.run_genesis_ue_camera_view_parity \\
    --config configs/tracking/multiview_realtime_dwpose_triangulation.yaml \\
    --ue-robot-mask-dir outputs/tracking_debug/62415c/robot_mask \\
    --skip-genesis-render --genesis-frame-dir outputs/camera_parity/genesis_frame0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_SRC_ROOT = next(parent for parent in (_THIS_FILE.parent, *_THIS_FILE.parents) if parent.name == "src")
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from common.project import project_paths
from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.tracking.calibration import load_calibration_bundle
from projects.genesis_ue_sync.tracking.camera_view_parity import (
    capture_ue_views_from_zmq,
    load_genesis_views,
    load_ue_views_from_dir,
    load_ue_views_from_robot_mask_debug,
    render_genesis_calibration_views,
    run_camera_view_parity,
)

PROJECT_PATHS = project_paths(__file__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/tracking/multiview_realtime_dwpose_triangulation.yaml"))
    p.add_argument("--calibration-path", type=Path, default=None)
    p.add_argument("--scene-spec-path", type=Path, default=None)
    p.add_argument("--camera-ids", type=str, nargs="*", default=None)
    p.add_argument("--output-root", type=Path, default=PROJECT_PATHS.tmp_root / "genesis_ue_camera_parity")
    p.add_argument("--genesis-backend", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--skip-genesis-render", action="store_true")
    p.add_argument("--genesis-frame-dir", type=Path, default=None, help="Directory with cam_*_frame0.png renders.")
    p.add_argument("--no-robot", action="store_true", help="Render Genesis without the robot arm.")
    p.add_argument("--robot-model", type=str, default="", help="Override Genesis robot model_id, e.g. rm75_6f.")
    p.add_argument("--image-correction-mode", type=str, default="scene_layout")
    p.add_argument("--ue-connect", type=str, default=None, help="ZMQ endpoint for live UE JPEG frames.")
    p.add_argument("--ue-topic", type=str, default=None)
    p.add_argument("--ue-timeout-s", type=float, default=30.0)
    p.add_argument("--ue-frame-dir", type=Path, default=None, help="Directory with per-camera UE PNG/JPG files.")
    p.add_argument(
        "--ue-robot-mask-dir",
        type=Path,
        default=None,
        help="robot_mask export root containing frame_XXXXXX/<camera>/01_original.png",
    )
    p.add_argument("--ue-frame-index", type=int, default=None, help="Pick a specific robot_mask frame index.")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, list[str], str | None, str]:
    cfg = MultiviewRealtimeConfig.load(args.config)
    calibration_path = Path(args.calibration_path or cfg.calibration_path)
    scene_spec_path = Path(args.scene_spec_path or cfg.scene_spec_path)
    camera_ids = list(args.camera_ids or cfg.camera_ids)
    ue_connect = args.ue_connect or cfg.ingress.connect
    ue_topic = args.ue_topic or cfg.ingress.topic
    return calibration_path, scene_spec_path, camera_ids, ue_connect, ue_topic


def main() -> int:
    args = parse_args()
    calibration_path, scene_spec_path, camera_ids, ue_connect, ue_topic = _resolve_paths(args)
    output_root = Path(args.output_root).expanduser().resolve()
    genesis_dir = Path(args.genesis_frame_dir or output_root / "genesis_frame0").expanduser().resolve()

    calibration = load_calibration_bundle(calibration_path, scene_spec_path=scene_spec_path)

    if not args.skip_genesis_render:
        print(f"Rendering Genesis frame0 to {genesis_dir} ...")
        camera_outputs = render_genesis_calibration_views(
            scene_spec_path=scene_spec_path,
            output_root=genesis_dir,
            backend=str(args.genesis_backend),
            include_robot=not bool(args.no_robot),
            robot_model=str(args.robot_model or ""),
        )
        for camera_id, path in sorted(camera_outputs.items()):
            print(f"  genesis {camera_id}: {path}")
    else:
        print(f"Using existing Genesis renders in {genesis_dir}")

    genesis_views = load_genesis_views(genesis_dir, camera_ids)

    ue_capture_meta: dict = {}
    if args.ue_robot_mask_dir is not None:
        ue_views = load_ue_views_from_robot_mask_debug(
            args.ue_robot_mask_dir,
            camera_ids,
            frame_index=args.ue_frame_index,
        )
        ue_capture_meta = {
            "source": "robot_mask_debug",
            "root": str(Path(args.ue_robot_mask_dir).expanduser().resolve()),
            "frame_index": args.ue_frame_index,
        }
    elif args.ue_frame_dir is not None:
        ue_views = load_ue_views_from_dir(args.ue_frame_dir, camera_ids)
        ue_capture_meta = {
            "source": "frame_dir",
            "root": str(Path(args.ue_frame_dir).expanduser().resolve()),
        }
    else:
        print(f"Capturing synced UE frames from {ue_connect} topic={ue_topic} ...")
        ue_views, ue_capture_meta = capture_ue_views_from_zmq(
            connect=str(ue_connect),
            camera_ids=camera_ids,
            topic=str(ue_topic),
            timeout_s=float(args.ue_timeout_s),
        )
        ue_capture_meta["source"] = "zmq"
        ue_capture_meta["connect"] = str(ue_connect)
        ue_capture_meta["topic"] = str(ue_topic)

    report = run_camera_view_parity(
        calibration=calibration,
        camera_ids=camera_ids,
        output_root=output_root,
        genesis_views=genesis_views,
        ue_views=ue_views,
        image_correction_mode=str(args.image_correction_mode),
        ue_capture_meta=ue_capture_meta,
    )

    print(f"\nGenesis vs UE camera parity report: {report['paths']['report_json']}")
    print(f"Tiled panel: {report['paths']['tiled_panel']}")
    for line in report.get("summary_lines") or []:
        print(line)

    mismatches = [
        cid
        for cid in camera_ids
        if report["cameras"][cid]["scene_layout"]["flip_u"] != report["cameras"][cid]["recommended_flip_u"]
        or report["cameras"][cid]["scene_layout"]["flip_v"] != report["cameras"][cid]["recommended_flip_v"]
    ]
    if mismatches:
        print(f"\nWARNING: scene_layout disagrees with empirical best for: {', '.join(mismatches)}")
        print("Inspect per_camera/*_flip_panel.png before changing yaml flip overrides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

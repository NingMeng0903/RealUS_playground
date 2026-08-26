#!/usr/bin/env python3
"""Publish Orbbec D2C-aligned colored point cloud (camera frame) over ZMQ.

Occupies the USB camera (v1 Gemini goes through the 3.9 bridge). The twin
viewer only subscribes and must be started with ``--orbbec-cloud``.

  python perception/apps/run_orbbec_cloud_publisher.py
  python apps/joint_admittance_8dof/run_with_twin.py --orbbec-cloud
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(os.environ.get("REALUS_PROJECT_ROOT") or Path(__file__).resolve().parents[2]).resolve()
for _p in (_REPO / "rm75_control", _REPO / "camera_calibration" / "src"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import (  # noqa: E402
    DEFAULT_CLOUD_STRIDE,
    DEFAULT_ORBBEC_CLOUD_BIND,
    DEFAULT_ORBBEC_CLOUD_TOPIC,
    load_T_link7_cam,
    pack_cloud_multipart,
    rgb_uint8_to_float,
)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pub-bind", type=str, default=DEFAULT_ORBBEC_CLOUD_BIND)
    ap.add_argument("--topic", type=str, default=DEFAULT_ORBBEC_CLOUD_TOPIC)
    ap.add_argument("--orbbec-yaml", type=Path, default=None, help="orbbec.yaml (default: camera_calibration/configs)")
    ap.add_argument("--handeye", type=Path, default=None, help="orbbec_handeye.yaml for T_link7_cam in meta")
    ap.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_CLOUD_STRIDE,
        help="Fixed pixel stride (default 6 → ~8k on 640×480)",
    )
    ap.add_argument("--fps", type=int, default=30, help="Color stream fps (overrides orbbec.yaml)")
    ap.add_argument("--depth-fps", type=int, default=30, help="Depth stream fps (overrides orbbec.yaml)")
    ap.add_argument("--session-id", type=str, default="realus_orbbec")
    ap.add_argument(
        "--d2c-offset",
        type=Path,
        default=None,
        help="orbbec_d2c_offset.yaml (default: camera_calibration/calibration_results)",
    )
    ap.add_argument(
        "--no-d2c-offset",
        action="store_true",
        help="Skip depth-to-color residual rotation (A/B). Stage 5 yaml is never written.",
    )
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def _cloud_rgb(rgb: np.ndarray | None, source: str) -> np.ndarray:
    if rgb is None or np.asarray(rgb).size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return rgb_uint8_to_float(rgb, bgr=str(source).startswith("unproject"))


def main() -> int:
    args = _parse_args()
    T_link7_cam = load_T_link7_cam(args.handeye)
    print(
        f"bind={args.pub_bind} topic={args.topic} fps={args.fps} depth_fps={args.depth_fps} "
        f"stride={args.stride}",
        flush=True,
    )

    try:
        import zmq
        from multicam_calib.calib.orbbec_d2c_offset import apply_R_depth_to_color, load_R_depth_to_color
        from multicam_calib.calib.orbbec_rgbd import unproject_aligned_depth
        from multicam_calib.devices.orbbec import OrbbecRGBDSession
        from multicam_calib.io.config import load_orbbec
    except ImportError as exc:
        print(
            f"missing dependency: {exc}\n"
            "Use the camera_calib env (has multicam_calib + v1 bridge).",
            file=sys.stderr,
        )
        return 1

    if args.no_d2c_offset:
        R_d2c = np.eye(3, dtype=np.float64)
        d2c_meta = {"source": "disabled", "R_depth_to_color_rpy_xyz_deg": [0.0, 0.0, 0.0]}
    else:
        R_d2c, d2c_meta = load_R_depth_to_color(args.d2c_offset)
    d2c_rpy_deg = np.asarray(d2c_meta.get("R_depth_to_color_rpy_xyz_deg") or [0.0, 0.0, 0.0], dtype=np.float64)
    print(
        f"d2c_offset rpy_xyz_deg=[{d2c_rpy_deg[0]:+.4f}, {d2c_rpy_deg[1]:+.4f}, {d2c_rpy_deg[2]:+.4f}] "
        f"source={d2c_meta.get('source')}",
        flush=True,
    )
    if args.dry_run:
        return 0

    cfg = load_orbbec(args.orbbec_yaml)
    cfg.fps = int(args.fps)
    cfg.depth_fps = int(args.depth_fps)
    session = OrbbecRGBDSession(cfg)
    sock = None
    try:
        params = session.open()
        emitter = session.emitter
        print(
            f"backend={session.backend} serial={params.serial!r} model={params.model} "
            f"has_depth={session.has_depth} laser={emitter.get('laser')} ldp={emitter.get('ldp')}",
            flush=True,
        )
        if not session.has_depth:
            print("Orbbec opened without depth (v4l2 RGB-only). Cannot publish a cloud.", file=sys.stderr)
            return 1

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.SNDTIMEO, 0)
        sock.bind(str(args.pub_bind))

        K = np.asarray(params.color.K, dtype=np.float64).reshape(3, 3)
        frame_index = 0
        misses = 0
        empty = 0
        while True:
            try:
                frame = session.read(timeout_ms=2000)
            except Exception as exc:
                misses += 1
                if misses in (1, 5, 15) or misses % 30 == 0:
                    print(f"WARN read failed x{misses}: {exc}", flush=True)
                continue
            misses = 0
            if getattr(frame, "depth_m", None) is None:
                continue
            xyz, rgb = unproject_aligned_depth(
                frame.depth_m,
                K,
                color_bgr=frame.color_bgr,
                stride=int(max(args.stride, 1)),
                min_m=cfg.min_depth_m,
                max_m=cfg.max_depth_m,
            )
            if xyz is None or xyz.size == 0:
                empty += 1
                if empty in (1, 10) or empty % 30 == 0:
                    zmax = float(np.nanmax(frame.depth_m)) if frame.depth_m is not None else -1.0
                    print(
                        f"empty cloud x{empty} z_max={zmax:.4f} "
                        f"color={getattr(frame.color_bgr, 'shape', None)}",
                        flush=True,
                    )
                continue
            empty = 0
            xyz = apply_R_depth_to_color(xyz, R_d2c)
            source = "unproject"
            rgb_f = _cloud_rgb(rgb, source)
            if rgb_f.shape[0] != xyz.shape[0]:
                rgb_f = np.full((int(xyz.shape[0]), 3), 0.7, dtype=np.float32)
            meta: dict[str, Any] = {
                "schema_version": 1,
                "session_id": str(args.session_id),
                "source_id": "realus.orbbec",
                "frame_index": int(frame_index),
                "timestamp_ns": int(frame.timestamp_ns),
                "wall_time_ns": int(time.time_ns()),
                "n": int(xyz.shape[0]),
                "K": K.tolist(),
                "T_link7_cam": np.asarray(T_link7_cam, dtype=np.float64).tolist(),
                "cloud_source": str(source),
                "color_size": [int(frame.color_size[0]), int(frame.color_size[1])],
                "d2c_offset_rpy_xyz_deg": [float(v) for v in d2c_rpy_deg],
                "d2c_offset_applied": bool(d2c_meta.get("source") not in {"identity", "identity_bad_yaml", "identity_bad_R", "disabled"}),
            }
            payload = pack_cloud_multipart(args.topic, meta, xyz, rgb_f)
            try:
                sock.send_multipart(payload, flags=zmq.NOBLOCK)
            except Exception:
                pass
            frame_index += 1
            if frame_index == 1 or frame_index % 30 == 0:
                print(f"published n={xyz.shape[0]} source={source} frame={frame_index}", flush=True)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        return 0
    finally:
        if sock is not None:
            try:
                sock.close(0)
            except Exception:
                pass
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

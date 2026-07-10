#!/usr/bin/env python3
"""Publish N RealSense RGB streams as amongus_camera_frame_v1 ZMQ frames.

Reads camera list + K/dist/extrinsics from genesis_bundle.yaml (not hard-coded to 4).
Device serials come from camera_calibration/configs/cameras.yaml when present.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def _serial_map(cameras_yaml: Path | None) -> dict[str, str]:
    if cameras_yaml is None or not cameras_yaml.is_file():
        return {}
    payload = _load_yaml(cameras_yaml)
    out: dict[str, str] = {}
    for row in payload.get("cameras", []) or []:
        alias = str(row.get("alias", "")).strip()
        serial = str(row.get("serial", "")).strip()
        if alias and serial:
            out[alias] = serial
    return out


def _enabled_cameras(bundle: dict[str, Any], only: list[str] | None) -> list[str]:
    cams = bundle.get("cameras") or {}
    if not isinstance(cams, dict) or not cams:
        raise ValueError("genesis_bundle has no cameras block")
    names = [str(k) for k in cams.keys()]
    if only:
        missing = [c for c in only if c not in cams]
        if missing:
            raise ValueError(f"Requested cameras not in bundle: {missing}")
        return list(only)
    return names


def _meta_for_camera(
    *,
    camera_name: str,
    cam: dict[str, Any],
    frame_index: int,
    wall_time_ns: int,
    session_id: str,
) -> dict[str, Any]:
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cam.get("distortion", [0.0] * 5), dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    image_size = [int(v) for v in cam["image_size"]]
    cfw = np.asarray(cam["camera_from_world"], dtype=np.float64).reshape(4, 4)
    wfc = np.asarray(cam["world_from_camera"], dtype=np.float64).reshape(4, 4)
    return {
        "schema_version": 1,
        "session_id": session_id,
        "source_id": "realus.realsense",
        "camera_name": camera_name,
        "frame_index": int(frame_index),
        "sim_time_ns": int(wall_time_ns),
        "wall_time_ns": int(wall_time_ns),
        "source_time_ns": int(wall_time_ns),
        "encoding": "jpeg",
        "width": int(image_size[0]),
        "height": int(image_size[1]),
        "intrinsics": {
            "K": K.tolist(),
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "distortion": dist[:5].tolist(),
        },
        "extrinsics": {
            "camera_from_world": cfw.tolist(),
            "world_from_camera": wfc.tolist(),
        },
        "scene_capture_flip_u": False,
        "scene_capture_flip_v": False,
    }


def _open_realsense(serial: str, width: int, height: int, fps: int):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 required. Use camera_calib env or install into genesis env."
        ) from exc
    cfg = rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)
    return rs, pipeline, profile


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="genesis_bundle.yaml (default: REALUS_CAMERA_CALIB_BUNDLE / CAMERA_CALIB_BUNDLE)",
    )
    ap.add_argument(
        "--cameras-yaml",
        type=Path,
        default=None,
        help="serial→alias map (default: REALUS_CAMERAS_YAML)",
    )
    ap.add_argument("--camera-ids", nargs="*", default=None, help="Subset of aliases; default=all in bundle")
    ap.add_argument("--pub-bind", type=str, default="tcp://127.0.0.1:17356")
    ap.add_argument("--topic", type=str, default="amongus_camera_frame_v1")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--undistort", action="store_true", help="Undistort with bundle K/dist before publish")
    ap.add_argument("--session-id", type=str, default="realus_realsense")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import os

    bundle_path = args.bundle
    if bundle_path is None:
        raw = os.environ.get("REALUS_CAMERA_CALIB_BUNDLE") or os.environ.get("CAMERA_CALIB_BUNDLE") or ""
        bundle_path = Path(raw) if raw else Path("camera_calibration/calibration_results/genesis_bundle.yaml")
    cameras_yaml = args.cameras_yaml
    if cameras_yaml is None:
        raw = os.environ.get("REALUS_CAMERAS_YAML", "")
        cameras_yaml = Path(raw) if raw else Path("camera_calibration/configs/cameras.yaml")

    repo = Path(os.environ.get("REALUS_PROJECT_ROOT", ".")).resolve()
    if not bundle_path.is_absolute():
        bundle_path = (repo / bundle_path).resolve()
    if not cameras_yaml.is_absolute():
        cameras_yaml = (repo / cameras_yaml).resolve()

    bundle = _load_yaml(bundle_path)
    cam_ids = _enabled_cameras(bundle, args.camera_ids)
    serials = _serial_map(cameras_yaml)
    for cid in cam_ids:
        hw = (bundle["cameras"][cid].get("hardware") or {})
        if cid not in serials and hw.get("serial"):
            serials[cid] = str(hw["serial"])
        if cid not in serials:
            raise ValueError(f"No serial for {cid}; set cameras.yaml or hardware.serial in bundle")

    print(f"publishing {len(cam_ids)} cameras: {cam_ids}", flush=True)
    print(f"bind={args.pub_bind} topic={args.topic}", flush=True)
    if args.dry_run:
        return 0

    try:
        import cv2
        import zmq
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 1

    pipelines: dict[str, Any] = {}
    maps: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    try:
        for cid in cam_ids:
            cam = bundle["cameras"][cid]
            w, h = [int(v) for v in cam["image_size"]]
            _rs, pipe, _prof = _open_realsense(serials[cid], w, h, args.fps)
            pipelines[cid] = pipe
            if args.undistort:
                K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
                dist = np.asarray(cam.get("distortion", [0.0] * 5), dtype=np.float64).reshape(-1)[:5]
                maps[cid] = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
            else:
                maps[cid] = None
            print(f"  opened {cid} serial={serials[cid]} {w}x{h}@{args.fps}", flush=True)

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.bind(str(args.pub_bind))
        topic = str(args.topic).encode("utf-8")
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
        frame_index = 0
        period = 1.0 / max(float(args.fps), 1.0)
        while True:
            t0 = time.perf_counter()
            wall_ns = time.time_ns()
            for cid in cam_ids:
                frames = pipelines[cid].wait_for_frames(timeout_ms=2000)
                color = frames.get_color_frame()
                if color is None:
                    continue
                bgr = np.asanyarray(color.get_data())
                m = maps[cid]
                if m is not None:
                    bgr = cv2.remap(bgr, m[0], m[1], interpolation=cv2.INTER_LINEAR)
                ok, buf = cv2.imencode(".jpg", bgr, encode_params)
                if not ok:
                    continue
                meta = _meta_for_camera(
                    camera_name=cid,
                    cam=bundle["cameras"][cid],
                    frame_index=frame_index,
                    wall_time_ns=wall_ns,
                    session_id=str(args.session_id),
                )
                sock.send_multipart([topic, json.dumps(meta, ensure_ascii=True).encode("utf-8"), buf.tobytes()])
            frame_index += 1
            delay = period - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        return 0
    finally:
        for pipe in pipelines.values():
            try:
                pipe.stop()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

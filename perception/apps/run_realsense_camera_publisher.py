#!/usr/bin/env python3
"""Publish N RealSense RGB streams over ZMQ (capture + preview topics).

Capture topic (``amongus_camera_frame_v1``): full resolution for calibration / SMPL-X.
Preview topic (``amongus_camera_preview_v1``): downscaled JPEG for live UI (drop-friendly).

Timestamps follow ROS semantics: ``source_time_ns`` from RealSense global-time (hardware),
``wall_time_ns`` host receipt time, ``sim_time_ns`` = ``source_time_ns``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
if str(_PERCEPTION_ROOT) not in sys.path:
    sys.path.insert(0, str(_PERCEPTION_ROOT))
from realsense_open import open_color_pipeline  # noqa: E402
from realsense_timestamps import frame_timing_ns  # noqa: E402

DEFAULT_CAPTURE_TOPIC = "amongus_camera_frame_v1"
DEFAULT_PREVIEW_TOPIC = "amongus_camera_preview_v1"


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
    source_time_ns: int,
    wall_time_ns: int,
    session_id: str,
    width: int,
    height: int,
    image_is_undistorted: bool,
) -> dict[str, Any]:
    K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cam.get("distortion", [0.0] * 5), dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    image_size = [int(v) for v in cam["image_size"]]
    cfw = np.asarray(cam["camera_from_world"], dtype=np.float64).reshape(4, 4)
    wfc = np.asarray(cam["world_from_camera"], dtype=np.float64).reshape(4, 4)
    full_w, full_h = int(image_size[0]), int(image_size[1])
    sx = float(width) / float(full_w) if full_w else 1.0
    sy = float(height) / float(full_h) if full_h else 1.0
    K_out = K.copy()
    K_out[0, 0] *= sx
    K_out[1, 1] *= sy
    K_out[0, 2] *= sx
    K_out[1, 2] *= sy
    return {
        "schema_version": 1,
        "session_id": session_id,
        "source_id": "realus.realsense",
        "camera_name": camera_name,
        "frame_index": int(frame_index),
        "source_time_ns": int(source_time_ns),
        "sim_time_ns": int(source_time_ns),
        "wall_time_ns": int(wall_time_ns),
        "encoding": "jpeg",
        "width": int(width),
        "height": int(height),
        "intrinsics": {
            "K": K_out.tolist(),
            "fx": float(K_out[0, 0]),
            "fy": float(K_out[1, 1]),
            "cx": float(K_out[0, 2]),
            "cy": float(K_out[1, 2]),
            "distortion": dist[:5].tolist(),
        },
        # Downstream calibrated DLT must not mix raw distorted images with an
        # undistorted K/zero-distortion projection model.  Keep both the
        # original calibration distortion and the explicit image contract.
        "image_geometry": {
            "undistorted": bool(image_is_undistorted),
            "effective_K": K_out.tolist(),
            "projection_distortion_model": "zero" if image_is_undistorted else "calibration",
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
        import pyrealsense2 as rs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 required. Use camera_calib env or install into genesis env."
        ) from exc
    pipeline, profile, usb = open_color_pipeline(
        str(serial), int(width), int(height), int(fps)
    )
    return pipeline, profile, usb


def _encode_send(
    *,
    sock: Any,
    send_lock: threading.Lock,
    topic: bytes,
    meta: dict[str, Any],
    bgr: np.ndarray,
    encode_params: list[int],
) -> None:
    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, encode_params)
    if not ok:
        return
    payload = [topic, json.dumps(meta, ensure_ascii=True).encode("utf-8"), buf.tobytes()]
    with send_lock:
        try:
            import zmq

            sock.send_multipart(payload, flags=zmq.NOBLOCK)
        except Exception:
            return


def _camera_publish_loop(
    *,
    cid: str,
    pipe: Any,
    undistort_map: tuple[np.ndarray, np.ndarray] | None,
    cam: dict[str, Any],
    sock: Any,
    capture_topic: bytes,
    preview_topic: bytes | None,
    capture_encode: list[int],
    preview_encode: list[int],
    preview_max_width: int,
    session_id: str,
    send_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    import cv2

    frame_index = 0
    misses = 0
    while not stop_event.is_set():
        try:
            frames = pipe.wait_for_frames(timeout_ms=1000)
        except Exception:
            misses += 1
            if misses in (1, 5, 15) or misses % 30 == 0:
                print(f"WARN {cid}: wait_for_frames timeout x{misses}", flush=True)
            continue
        misses = 0
        color = frames.get_color_frame()
        if color is None:
            continue
        source_ns, wall_ns = frame_timing_ns(color)
        # RealSense ring buffer is reused; copy before remap/JPEG.
        bgr = np.ascontiguousarray(color.get_data()).copy()
        full_h, full_w = bgr.shape[:2]
        capture_bgr = bgr
        if undistort_map is not None:
            capture_bgr = cv2.remap(bgr, undistort_map[0], undistort_map[1], interpolation=cv2.INTER_LINEAR)
        if preview_topic is not None and preview_max_width > 0:
            pw = min(int(preview_max_width), int(full_w))
            ph = max(1, int(round(full_h * pw / max(full_w, 1))))
            preview_bgr = cv2.resize(capture_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
            meta_preview = _meta_for_camera(
                camera_name=cid,
                cam=cam,
                frame_index=frame_index,
                source_time_ns=source_ns,
                wall_time_ns=wall_ns,
                session_id=session_id,
                width=pw,
                height=ph,
                image_is_undistorted=undistort_map is not None,
            )
            _encode_send(
                sock=sock,
                send_lock=send_lock,
                topic=preview_topic,
                meta=meta_preview,
                bgr=preview_bgr,
                encode_params=preview_encode,
            )
        meta_capture = _meta_for_camera(
            camera_name=cid,
            cam=cam,
            frame_index=frame_index,
            source_time_ns=source_ns,
            wall_time_ns=wall_ns,
            session_id=session_id,
            width=full_w,
            height=full_h,
            image_is_undistorted=undistort_map is not None,
        )
        _encode_send(
            sock=sock,
            send_lock=send_lock,
            topic=capture_topic,
            meta=meta_capture,
            bgr=capture_bgr,
            encode_params=capture_encode,
        )
        frame_index += 1


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
    ap.add_argument("--topic", type=str, default=DEFAULT_CAPTURE_TOPIC, help="Full-res capture topic")
    ap.add_argument(
        "--preview-topic",
        type=str,
        default=DEFAULT_PREVIEW_TOPIC,
        help="Low-res preview topic (empty to disable)",
    )
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument(
        "--width",
        type=int,
        default=0,
        help="Color width (0 = genesis_bundle image_size). Set 1920 when recalibrating at 1080p.",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=0,
        help="Color height (0 = genesis_bundle image_size). Set 1080 when recalibrating at 1080p.",
    )
    ap.add_argument("--jpeg-quality", type=int, default=90, help="Capture JPEG quality")
    ap.add_argument("--preview-jpeg-quality", type=int, default=72)
    ap.add_argument("--preview-max-width", type=int, default=960, help="Preview stream width (height scaled)")
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

    preview_topic_str = str(args.preview_topic or "").strip()
    print(f"publishing {len(cam_ids)} cameras: {cam_ids}", flush=True)
    print(f"bind={args.pub_bind} capture={args.topic} preview={preview_topic_str or '(off)'}", flush=True)
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
    threads: list[threading.Thread] = []
    stop_event = threading.Event()
    try:
        for i, cid in enumerate(cam_ids):
            cam = bundle["cameras"][cid]
            bw, bh = [int(v) for v in cam["image_size"]]
            w = int(args.width) if int(args.width) > 0 else bw
            h = int(args.height) if int(args.height) > 0 else bh
            if i:
                time.sleep(0.3)
            pipe, _prof, usb = _open_realsense(serials[cid], w, h, args.fps)
            pipelines[cid] = pipe
            if args.undistort:
                if (w, h) != (bw, bh):
                    raise ValueError(
                        f"{cid}: --undistort needs stream {w}x{h} to match bundle "
                        f"image_size {bw}x{bh}. Recalibrating? omit --undistort until Stage 0 is redone."
                    )
                K = np.asarray(cam["intrinsics"], dtype=np.float64).reshape(3, 3)
                dist = np.asarray(cam.get("distortion", [0.0] * 5), dtype=np.float64).reshape(-1)[:5]
                maps[cid] = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)
            else:
                maps[cid] = None
            print(
                f"  opened {cid} serial={serials[cid]} usb={usb} {w}x{h}@{args.fps} global_time=on",
                flush=True,
            )

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.setsockopt(zmq.SNDHWM, 64)
        sock.setsockopt(zmq.SNDTIMEO, 0)
        sock.bind(str(args.pub_bind))
        capture_topic = str(args.topic).encode("utf-8")
        preview_topic = preview_topic_str.encode("utf-8") if preview_topic_str else None
        capture_encode = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
        preview_encode = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.preview_jpeg_quality)]
        send_lock = threading.Lock()
        stop_event.clear()
        threads = []
        for cid in cam_ids:
            t = threading.Thread(
                target=_camera_publish_loop,
                kwargs={
                    "cid": cid,
                    "pipe": pipelines[cid],
                    "undistort_map": maps[cid],
                    "cam": bundle["cameras"][cid],
                    "sock": sock,
                    "capture_topic": capture_topic,
                    "preview_topic": preview_topic,
                    "capture_encode": capture_encode,
                    "preview_encode": preview_encode,
                    "preview_max_width": int(args.preview_max_width),
                    "session_id": str(args.session_id),
                    "send_lock": send_lock,
                    "stop_event": stop_event,
                },
                name=f"realsense-{cid}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        print(f"  parallel capture: {len(threads)} threads, source_time_ns=RS global time", flush=True)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopped", flush=True)
        return 0
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)
        for pipe in pipelines.values():
            try:
                pipe.stop()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

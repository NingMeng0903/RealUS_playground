"""ZMQ CameraFrame JPEG publisher (same 3-part contract as RealSense)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

from us_framegrab.config import FrameGrabConfig

log = logging.getLogger("us_framegrab.zmq")

DEFAULT_PUB_BIND = "tcp://127.0.0.1:17359"


def downscale_preview(image: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return image
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    import cv2

    new_h = max(1, int(round(height * float(max_width) / float(width))))
    return cv2.resize(image, (int(max_width), new_h), interpolation=cv2.INTER_AREA)


def pack_jpeg_parts(
    topic: str | bytes,
    meta: dict[str, Any],
    image: np.ndarray,
    quality: int,
) -> list[bytes] | None:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(min(100, max(1, quality)))],
    )
    if not ok:
        return None
    topic_b = topic.encode("utf-8") if isinstance(topic, str) else topic
    return [
        topic_b,
        json.dumps(meta, ensure_ascii=True).encode("utf-8"),
        buf.tobytes(),
    ]


def camera_frame_meta(
    *,
    cfg: FrameGrabConfig,
    frame_index: int,
    width: int,
    height: int,
    source_time_ns: int,
    wall_time_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "session_id": str(cfg.session_id),
        "source_id": str(cfg.source_id),
        "camera_name": str(cfg.camera_name),
        "frame_id": str(cfg.frame_id),
        "frame_index": int(frame_index),
        "source_time_ns": int(source_time_ns),
        "sim_time_ns": int(source_time_ns),
        "wall_time_ns": int(wall_time_ns),
        "encoding": "jpeg",
        "width": int(width),
        "height": int(height),
    }


class UsImagePublisher:
    """PUB bind + optional preview topic. Drops on HWM (NOBLOCK)."""

    def __init__(self, cfg: FrameGrabConfig) -> None:
        self._cfg = cfg
        self._sock: Any = None
        self._ctx: Any = None

    def bind(self) -> None:
        import zmq

        self._ctx = zmq.Context.instance()
        sock = self._ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.LINGER, 200)
        sock.setsockopt(zmq.SNDHWM, 2)
        sock.setsockopt(zmq.SNDTIMEO, 0)
        sock.bind(str(self._cfg.pub_bind))
        self._sock = sock
        log.info(
            "ZMQ PUB bind=%s capture=%s preview=%s",
            self._cfg.pub_bind,
            self._cfg.capture_topic,
            self._cfg.preview_topic or "(off)",
        )

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        try:
            sock.close(0)
        except Exception:
            pass

    def send(self, image: np.ndarray, frame_index: int) -> None:
        if self._sock is None:
            return
        import zmq

        now = time.time_ns()
        offset_ns = int(round(float(self._cfg.time_offset) * 1e9))
        source_ns = now + offset_ns
        height, width = image.shape[:2]
        meta = camera_frame_meta(
            cfg=self._cfg,
            frame_index=frame_index,
            width=width,
            height=height,
            source_time_ns=source_ns,
            wall_time_ns=now,
        )
        parts = pack_jpeg_parts(
            self._cfg.capture_topic,
            meta,
            image,
            self._cfg.compressed_quality,
        )
        if parts is None:
            return
        try:
            self._sock.send_multipart(parts, flags=zmq.NOBLOCK)
        except Exception:
            return

        preview_topic = str(self._cfg.preview_topic or "").strip()
        if not preview_topic:
            return
        preview = downscale_preview(image, int(self._cfg.preview_max_width))
        ph, pw = preview.shape[:2]
        preview_meta = dict(meta)
        preview_meta["width"] = int(pw)
        preview_meta["height"] = int(ph)
        preview_parts = pack_jpeg_parts(
            preview_topic,
            preview_meta,
            preview,
            self._cfg.preview_jpeg_quality,
        )
        if preview_parts is None:
            return
        try:
            self._sock.send_multipart(preview_parts, flags=zmq.NOBLOCK)
        except Exception:
            return

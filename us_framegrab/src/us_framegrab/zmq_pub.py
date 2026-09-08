"""ZMQ CameraFrame JPEG publisher (same 3-part contract as RealSense)."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import numpy as np

from us_framegrab.config import FrameGrabConfig
from realus_clock import get_clock

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
    capture_monotonic_ns: int | None = None,
    capture_wall_time_ns: int | None = None,
    time_offset_ns: int = 0,
    timestamp_source: str = "host_frame_read_complete",
    clock_domain: str = "host_monotonic",
    publisher_instance_id: str = "",
    crop_box: list[int] | tuple[int, int, int, int] | None = None,
    hflip: bool | None = None,
    color: bool | None = None,
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
        "capture_monotonic_ns": (
            None if capture_monotonic_ns is None else int(capture_monotonic_ns)
        ),
        "capture_wall_time_ns": (
            None if capture_wall_time_ns is None else int(capture_wall_time_ns)
        ),
        "time_offset_ns": int(time_offset_ns),
        "timestamp_source": str(timestamp_source),
        "clock_domain": str(clock_domain),
        "publisher_instance_id": str(publisher_instance_id),
        "crop_box": None if crop_box is None else [int(v) for v in crop_box],
        "hflip": None if hflip is None else bool(hflip),
        "color": None if color is None else bool(color),
        "encoding": "jpeg",
        "width": int(width),
        "height": int(height),
    }


class UsImagePublisher:
    """PUB bind + optional preview topic. Drops on HWM (NOBLOCK)."""

    def __init__(self, cfg: FrameGrabConfig) -> None:
        self.clock = get_clock()
        self._cfg = cfg
        self._sock: Any = None
        self._ctx: Any = None
        # Stable for this publisher process and new after every restart. This
        # lets a recorder distinguish a restarted source from a frame-index
        # reset without changing the multipart/topic contract.
        self._publisher_instance_id = uuid.uuid4().hex

    @property
    def publisher_instance_id(self) -> str:
        return self._publisher_instance_id

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

    def send(
        self,
        image: np.ndarray,
        frame_index: int,
        *,
        capture_monotonic_ns: int | None = None,
        capture_wall_time_ns: int | None = None,
        crop_box: list[int] | tuple[int, int, int, int] | None = None,
        hflip: bool | None = None,
        color: bool | None = None,
    ) -> None:
        if self._sock is None:
            return
        import zmq

        # ``wall_time_ns`` keeps its historical meaning: the publisher-side
        # wall-clock timestamp taken immediately before JPEG encoding. The
        # source timestamp is instead tied to the successful frame read, which
        # is supplied by FrameGrabSession before crop/encode work begins.
        now = time.time_ns()
        if capture_wall_time_ns is None:
            capture_wall_time_ns = now
        if capture_monotonic_ns is None:
            capture_monotonic_ns = time.monotonic_ns()
        offset_ns = int(round(float(self._cfg.time_offset) * 1e9))
        source_ns = int(capture_wall_time_ns) + offset_ns
        height, width = image.shape[:2]
        meta = camera_frame_meta(
            cfg=self._cfg,
            frame_index=frame_index,
            width=width,
            height=height,
            source_time_ns=source_ns,
            wall_time_ns=now,
            capture_monotonic_ns=capture_monotonic_ns,
            capture_wall_time_ns=capture_wall_time_ns,
            time_offset_ns=offset_ns,
            timestamp_source="host_frame_read_complete",
            clock_domain="host_monotonic",
            publisher_instance_id=self._publisher_instance_id,
            crop_box=crop_box,
            hflip=hflip,
            color=color,
        )
        meta.update(self.clock.metadata(
            str(self._cfg.frame_id),
            timestamp_ns=self.clock.from_monotonic_ns(capture_monotonic_ns) + offset_ns,
        ))
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

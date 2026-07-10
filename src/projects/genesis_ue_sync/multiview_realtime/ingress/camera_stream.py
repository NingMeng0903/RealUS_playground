from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.config import IngressConfig

# Matches integrations.controller_bus.stream_schemas.TOPIC_CAMERA_FRAME_V1 (avoid heavy package __init__).
DEFAULT_CAMERA_FRAME_TOPIC = "amongus_camera_frame_v1"


@dataclass(frozen=True)
class SyncedMultiviewFrame:
    frame_index: int
    views_rgb: dict[str, np.ndarray]
    metadata_by_camera: dict[str, dict[str, Any]]
    timestamp_ns: int


def decode_jpeg_bgr(image_bytes: bytes) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for camera JPEG decode.") from exc
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame.")
    return frame


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(np.asarray(frame_bgr), cv2.COLOR_BGR2RGB)


class MultiviewCameraStream:
    """Subscribe to multiplexed camera JPEG frames and emit synchronized triplets."""

    def __init__(
        self,
        config: IngressConfig,
        *,
        camera_ids: tuple[str, ...],
        topic: str | None = None,
    ) -> None:
        self.config = config
        self.camera_ids = tuple(camera_ids)
        self.topic = str(topic or config.topic or DEFAULT_CAMERA_FRAME_TOPIC)
        self._buffers: dict[str, deque[tuple[int, dict[str, Any], np.ndarray]]] = {
            cid: deque(maxlen=max(int(config.max_buffer_per_camera), 2)) for cid in self.camera_ids
        }
        self._ctx = None
        self._sock = None
        self._total_messages = 0
        self._dropped_messages = 0
        self._unknown_camera_hits: dict[str, int] = {}
        self._lock = threading.Lock()
        self._socket_lock = threading.Lock()
        self._ingest_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def ingest_running(self) -> bool:
        thread = self._ingest_thread
        return thread is not None and thread.is_alive()

    def buffer_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {}
        with self._lock:
            for camera_id in self.camera_ids:
                buf = self._buffers[camera_id]
                if buf:
                    status[camera_id] = {
                        "depth": len(buf),
                        "head_frame_index": int(buf[0][0]),
                        "tail_frame_index": int(buf[-1][0]),
                    }
                else:
                    status[camera_id] = {"depth": 0, "head_frame_index": None, "tail_frame_index": None}
            status["_ingress"] = {
                "total_messages": int(self._total_messages),
                "dropped_messages": int(self._dropped_messages),
                "unknown_cameras": dict(self._unknown_camera_hits),
            }
        return status

    def connect(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for camera ingress.") from exc
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, int(max(self.config.recv_timeout_ms, 1)))
        # Keep a generous receive buffer so a momentarily busy consumer does not lose
        # messages at the PUB high-water mark; the background drain thread keeps it empty.
        self._sock.setsockopt(zmq.RCVHWM, 200)
        self._sock.connect(str(self.config.connect))
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))

    def start_ingest(self) -> None:
        """Drain the socket continuously on a background thread (decouples ingest from inference)."""
        if self._ingest_thread is not None:
            return
        if self._sock is None:
            self.connect()
        self._stop_event.clear()
        thread = threading.Thread(target=self._ingest_loop, name="camera-ingest", daemon=True)
        self._ingest_thread = thread
        thread.start()

    def _ingest_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self._recv_one_message():
                time.sleep(0.001)

    def close(self) -> None:
        self._stop_event.set()
        thread = self._ingest_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._ingest_thread = None
        with self._socket_lock:
            if self._sock is not None:
                try:
                    self._sock.close(0)
                except Exception:
                    pass
                self._sock = None

    def poll_once(self) -> bool:
        """Ingest one ZMQ message if available. Returns True when a message was consumed."""
        if self.ingest_running:
            return False
        return self._recv_one_message()

    def _recv_one_message(self) -> bool:
        if self._sock is None:
            raise RuntimeError("MultiviewCameraStream.connect() was not called.")
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for camera ingress.") from exc
        with self._socket_lock:
            if self._sock is None:
                return False
            try:
                parts = self._sock.recv_multipart()
            except zmq.Again:
                return False
            except zmq.ZMQError:
                return False
        if len(parts) < 3:
            return False
        try:
            meta = json.loads(parts[1].decode("utf-8"))
            name = str(meta.get("camera_name") or meta.get("camera_frame_id") or "")
            if name not in self._buffers:
                with self._lock:
                    self._total_messages += 1
                    if name:
                        self._unknown_camera_hits[name] = int(self._unknown_camera_hits.get(name, 0)) + 1
                return True
            frame_index = int(meta.get("frame_index", 0))
            rgb = bgr_to_rgb(decode_jpeg_bgr(parts[2]))
            with self._lock:
                self._total_messages += 1
                self._buffers[name].append((frame_index, meta, rgb))
        except Exception:
            with self._lock:
                self._dropped_messages += 1
            return True
        return True

    def try_pop_synced(self) -> SyncedMultiviewFrame | None:
        """Return the oldest frame index present in all camera buffers (within tolerance)."""
        if not self.camera_ids:
            return None
        with self._lock:
            heads: list[tuple[str, int, dict[str, Any], np.ndarray]] = []
            for camera_id in self.camera_ids:
                buf = self._buffers[camera_id]
                if not buf:
                    return None
                fi, meta, rgb = buf[0]
                heads.append((camera_id, int(fi), meta, rgb))
            target = min(fi for _cid, fi, _meta, _rgb in heads)
            tol = max(int(self.config.sync_tolerance_frames), 0)
            for camera_id, fi, _meta, _rgb in heads:
                if abs(int(fi) - target) > tol:
                    if int(fi) < target:
                        self._buffers[camera_id].popleft()
                    return None
            views_rgb: dict[str, np.ndarray] = {}
            metadata_by_camera: dict[str, dict[str, Any]] = {}
            timestamps: list[int] = []
            for camera_id in self.camera_ids:
                fi, meta, rgb = self._buffers[camera_id].popleft()
                views_rgb[camera_id] = rgb
                metadata_by_camera[camera_id] = dict(meta)
                timestamps.append(int(meta.get("sim_time_ns") or meta.get("wall_time_ns") or fi))
        return SyncedMultiviewFrame(
            frame_index=int(target),
            views_rgb=views_rgb,
            metadata_by_camera=metadata_by_camera,
            timestamp_ns=min(timestamps) if timestamps else int(target),
        )

    def try_pop_synced_strict(self, *, max_frame_span: int = 2) -> SyncedMultiviewFrame | None:
        """Return the oldest synchronized frame with a bounded cross-camera frame span.

        This drops stale heads until all camera heads are close enough. It is stricter than
        ``try_pop_synced`` and avoids mixing visibly different hand poses across cameras.
        """
        if not self.camera_ids:
            return None
        max_span = max(0, int(max_frame_span))
        with self._lock:
            while True:
                heads: list[tuple[str, int, dict[str, Any], np.ndarray]] = []
                for camera_id in self.camera_ids:
                    buf = self._buffers[camera_id]
                    if not buf:
                        return None
                    fi, meta, rgb = buf[0]
                    heads.append((camera_id, int(fi), meta, rgb))
                min_f = min(fi for _cid, fi, _meta, _rgb in heads)
                max_f = max(fi for _cid, fi, _meta, _rgb in heads)
                if max_f - min_f <= max_span:
                    views_rgb: dict[str, np.ndarray] = {}
                    metadata_by_camera: dict[str, dict[str, Any]] = {}
                    timestamps: list[int] = []
                    for camera_id in self.camera_ids:
                        fi, meta, rgb = self._buffers[camera_id].popleft()
                        views_rgb[camera_id] = rgb
                        metadata_by_camera[camera_id] = dict(meta)
                        timestamps.append(int(meta.get("sim_time_ns") or meta.get("wall_time_ns") or fi))
                    return SyncedMultiviewFrame(
                        frame_index=int(min_f),
                        views_rgb=views_rgb,
                        metadata_by_camera=metadata_by_camera,
                        timestamp_ns=min(timestamps) if timestamps else int(min_f),
                    )
                for camera_id, fi, _meta, _rgb in heads:
                    if int(fi) == int(min_f):
                        self._buffers[camera_id].popleft()

    def pop_latest_synced(self) -> SyncedMultiviewFrame | None:
        """Drain all currently buffered synced frames and return only the newest one.

        Keeps inference real-time by discarding stale backlog instead of chasing it.
        """
        latest: SyncedMultiviewFrame | None = None
        while True:
            synced = self.try_pop_synced()
            if synced is None:
                return latest
            latest = synced

    def iter_synced(self, *, idle_sleep_s: float = 0.002):
        """Yield synchronized frames until interrupted."""
        while True:
            self.poll_once()
            synced = self.try_pop_synced()
            if synced is not None:
                yield synced
            else:
                time.sleep(float(idle_sleep_s))

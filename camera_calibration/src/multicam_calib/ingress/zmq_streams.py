"""ZMQ camera preview + capture ingress (shared RealSense publisher)."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from multicam_calib.devices.base import Frame
from multicam_calib.recording.sync import MultiCamSnapshot


def _decode_jpeg(image_bytes: bytes) -> np.ndarray:
    import cv2

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame.")
    return frame


def _meta_to_frame(meta: dict[str, Any], image_bgr: np.ndarray, *, frame_index: int) -> Frame:
    source_ns = int(meta.get("source_time_ns") or meta.get("sim_time_ns") or meta.get("wall_time_ns") or 0)
    wall_ns = int(meta.get("wall_time_ns") or source_ns)
    return Frame(
        image=np.ascontiguousarray(image_bgr),
        timestamp_ns=wall_ns,
        device_timestamp_ns=source_ns,
        frame_index=int(frame_index),
        metadata={"camera_name": str(meta.get("camera_name", ""))},
    )


@dataclass
class ZmqCameraStreamThread:
    """CameraStreamThread-compatible wrapper over a shared ZMQ latest-frame hub."""

    alias: str
    _hub: "ZmqMulticamHub"
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self, on_frame=None) -> None:
        self._hub.start()
        self._thread = threading.current_thread()

    def stop(self, timeout_s: float = 2.0) -> None:
        return

    def latest(self) -> Frame | None:
        return self._hub.preview_latest(self.alias)

    def last_error(self) -> BaseException | None:
        return self._hub.last_error(self.alias)


class ZmqMulticamHub:
    """Subscribe to preview (live UI) and capture (synced snapshot) topics from one publisher."""

    def __init__(
        self,
        *,
        connect: str,
        aliases: list[str],
        preview_topic: str = "amongus_camera_preview_v1",
        capture_topic: str = "amongus_camera_frame_v1",
    ) -> None:
        self.connect = str(connect)
        self.aliases = list(aliases)
        self.preview_topic = str(preview_topic)
        self.capture_topic = str(capture_topic)
        self._preview_latest: dict[str, Frame] = {}
        self._capture_latest: dict[str, Frame] = {}
        self._preview_index: dict[str, int] = {a: 0 for a in self.aliases}
        self._capture_index: dict[str, int] = {a: 0 for a in self.aliases}
        self._errors: dict[str, BaseException] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._capture_thread: threading.Thread | None = None
        self._started = False

    def stream_threads(self) -> dict[str, ZmqCameraStreamThread]:
        return {alias: ZmqCameraStreamThread(alias=alias, _hub=self) for alias in self.aliases}

    def start(self) -> None:
        if self._started:
            return
        self._stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_loop, name="zmq-preview", daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, name="zmq-capture", daemon=True)
        self._preview_thread.start()
        self._capture_thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        for t in (self._preview_thread, self._capture_thread):
            if t is not None and t.is_alive():
                t.join(timeout=1.5)
        self._preview_thread = None
        self._capture_thread = None
        self._started = False

    def preview_latest(self, alias: str) -> Frame | None:
        with self._lock:
            return self._preview_latest.get(alias)

    def last_error(self, alias: str) -> BaseException | None:
        return self._errors.get(alias)

    def capture_snapshot_best_effort(
        self,
        *,
        attempts: int = 30,
        poll_interval_ms: float = 2.0,
        use_device_timestamp: bool = True,
    ) -> MultiCamSnapshot | None:
        """Pick the tightest hardware-timestamp-aligned capture set (same as local snapshot_best_effort)."""
        best: MultiCamSnapshot | None = None
        n = max(1, int(attempts))
        sleep_s = max(0.0, float(poll_interval_ms)) / 1000.0
        for _ in range(n):
            with self._lock:
                frames = {a: self._capture_latest.get(a) for a in self.aliases}
            if any(v is None for v in frames.values()):
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue
            ts = []
            for f in frames.values():
                assert f is not None
                if use_device_timestamp and f.device_timestamp_ns is not None:
                    ts.append(int(f.device_timestamp_ns))
                else:
                    ts.append(int(f.timestamp_ns))
            spread = max(ts) - min(ts)
            snap = MultiCamSnapshot(
                frames={k: v for k, v in frames.items() if v is not None},
                host_timestamp_spread_ns=int(spread),
            )
            if best is None or snap.host_timestamp_spread_ns < best.host_timestamp_spread_ns:
                best = snap
            if sleep_s > 0:
                time.sleep(sleep_s)
        return best

    def _ingest_loop(self, *, topic: str, store: dict[str, Frame], index: dict[str, int]) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 50)
        sock.setsockopt(zmq.RCVHWM, 8)
        sock.connect(self.connect)
        sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        while not self._stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if len(parts) < 3:
                continue
            name = ""
            try:
                meta = json.loads(parts[1].decode("utf-8"))
                name = str(meta.get("camera_name") or "")
                if name not in self.aliases:
                    continue
                bgr = _decode_jpeg(parts[2])
                index[name] = index.get(name, 0) + 1
                frame = _meta_to_frame(meta, bgr, frame_index=index[name])
            except Exception as exc:
                if name:
                    self._errors[name] = exc
                continue
            with self._lock:
                store[name] = frame
                self._errors.pop(name, None)
            # Drain backlog — keep only newest per camera on this socket.
            while True:
                try:
                    parts = sock.recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) < 3:
                    continue
                try:
                    meta = json.loads(parts[1].decode("utf-8"))
                    name = str(meta.get("camera_name") or "")
                    if name not in self.aliases:
                        continue
                    bgr = _decode_jpeg(parts[2])
                    index[name] = index.get(name, 0) + 1
                    frame = _meta_to_frame(meta, bgr, frame_index=index[name])
                    with self._lock:
                        store[name] = frame
                except Exception:
                    continue
        sock.close(0)

    def _preview_loop(self) -> None:
        self._ingest_loop(topic=self.preview_topic, store=self._preview_latest, index=self._preview_index)

    def _capture_loop(self) -> None:
        self._ingest_loop(topic=self.capture_topic, store=self._capture_latest, index=self._capture_index)

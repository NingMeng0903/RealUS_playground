"""Gate live capture to a motion-sequence window (e.g. first half on-bed segment)."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projects.genesis_ue_sync.integrations.controller_bus.stream_schemas import TOPIC_CANONICAL_SCENE_V1
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream, SyncedMultiviewFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionFrameWindow:
    motion_start: int = 0
    motion_end_exclusive: int = 435
    require_canonical: bool = True

    def contains(self, motion_frame_index: int) -> bool:
        fi = int(motion_frame_index)
        return int(self.motion_start) <= fi < int(self.motion_end_exclusive)


def motion_window_from_scene_spec(
    scene_spec_path: str | Path,
    *,
    fraction: float = 0.5,
    motion_end_exclusive: int | None = None,
) -> MotionFrameWindow:
    from projects.genesis_ue_sync.sim_platform.scenes.common_scene import load_sync_scene_spec

    spec = load_sync_scene_spec(Path(scene_spec_path))
    start = int(getattr(spec.motion, "start_frame", 0) or 0)
    limit = int(
        motion_end_exclusive
        if motion_end_exclusive is not None
        else (
            getattr(spec.render, "frame_limit", None)
            or getattr(spec.motion, "frame_count", None)
            or 870
        )
    )
    span = max(1, int(limit) - int(start))
    end = int(start + max(1, int(round(span * float(fraction)))))
    return MotionFrameWindow(motion_start=start, motion_end_exclusive=end, require_canonical=True)


class CanonicalMotionIndexClient:
    """Non-blocking subscriber to Genesis canonical human.motion_frame_index."""

    def __init__(
        self,
        *,
        connect: str = "tcp://127.0.0.1:5599",
        topic: str = TOPIC_CANONICAL_SCENE_V1,
        recv_timeout_ms: int = 5,
    ) -> None:
        self.zmq_connect = str(connect)
        self.topic = str(topic)
        self.recv_timeout_ms = int(recv_timeout_ms)
        self._ctx = None
        self._sock = None
        self._latest_motion_frame_index: int | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for canonical motion gate.") from exc
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.RCVTIMEO, max(1, self.recv_timeout_ms))
        self._sock.setsockopt(zmq.RCVHWM, 8)
        self._sock.connect(self.zmq_connect)
        self._sock.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close(0)
                except Exception:
                    pass
                self._sock = None

    def poll(self, *, max_messages: int = 32) -> int | None:
        with self._lock:
            if self._sock is None:
                return self._latest_motion_frame_index
            try:
                import zmq

                for _ in range(max(1, int(max_messages))):
                    try:
                        parts = self._sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    except zmq.ZMQError:
                        break
                    if len(parts) < 2:
                        continue
                    payload = json.loads(parts[1].decode("utf-8"))
                    human = dict(payload.get("human") or {})
                    if "motion_frame_index" in human:
                        self._latest_motion_frame_index = int(human["motion_frame_index"])
            except Exception as exc:
                logger.debug("canonical poll failed: %s", exc)
            return self._latest_motion_frame_index

    @property
    def latest(self) -> int | None:
        return self._latest_motion_frame_index


def resolve_motion_frame_index(
    synced: SyncedMultiviewFrame,
    canonical: CanonicalMotionIndexClient | None,
) -> int | None:
    if canonical is not None:
        canonical.poll()
        if canonical.latest is not None:
            return int(canonical.latest)
    for meta in synced.metadata_by_camera.values():
        raw = meta.get("motion_frame_index")
        if raw is not None:
            return int(raw)
    return None


def wait_synced_in_motion_window(
    stream: MultiviewCameraStream,
    window: MotionFrameWindow,
    canonical: CanonicalMotionIndexClient | None,
    *,
    wait_timeout_s: float = 2.0,
) -> tuple[SyncedMultiviewFrame | None, int | None, str | None]:
    import time

    deadline = time.perf_counter() + float(wait_timeout_s)
    last_reason = "no_synced_frame"
    while time.perf_counter() < deadline:
        if not stream.ingest_running:
            stream.poll_once()
        if canonical is not None:
            canonical.poll()
        synced = stream.try_pop_synced()
        if synced is None:
            time.sleep(0.002)
            last_reason = "no_synced_frame"
            continue
        motion_fi = resolve_motion_frame_index(synced, canonical)
        if motion_fi is None:
            last_reason = "missing_motion_frame_index"
            if window.require_canonical:
                continue
            return synced, None, None
        if not window.contains(motion_fi):
            last_reason = f"outside_motion_window:{motion_fi}"
            continue
        return synced, int(motion_fi), None
    return None, None, last_reason

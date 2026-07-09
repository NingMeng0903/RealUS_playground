"""Multi-camera frame synchronizer.

For static calibration we only need "were these frames captured close enough
in host time to be considered simultaneous". Each camera streams in its own
thread; when the user presses "capture" we poll the latest frame from every
alias over a short window and keep the set with the smallest host-timestamp
spread (see ``snapshot_best_effort``).
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Callable

from multicam_calib.devices.base import CameraDevice, Frame


@dataclass
class CameraStreamThread:
    """Background thread that continuously polls a single camera.

    The most recent frame is always available via `latest()`; older frames are
    dropped. This suits the "press button when board is static" workflow: we
    don't need a queue, just the freshest frame from each camera at click time.
    """

    alias: str
    device: CameraDevice
    _latest: Frame | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _on_frame: Callable[[str, Frame], None] | None = field(default=None, init=False)
    _last_error: BaseException | None = field(default=None, init=False)

    def start(self, on_frame: Callable[[str, Frame], None] | None = None) -> None:
        self._on_frame = on_frame
        self._stop.clear()
        t = threading.Thread(target=self._run, name=f"cam-{self.alias}", daemon=True)
        self._thread = t
        t.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout_s)
        self._thread = None

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def last_error(self) -> BaseException | None:
        return self._last_error

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                f = self.device.read(timeout_ms=1000)
            except Exception as exc:  # noqa: BLE001
                self._last_error = exc
                continue
            with self._lock:
                self._latest = f
            if self._on_frame is not None:
                try:
                    self._on_frame(self.alias, f)
                except Exception:  # noqa: BLE001 — never let a callback kill the stream
                    pass


@dataclass
class MultiCamSnapshot:
    """A synchronized set of frames — one Frame per alias."""

    frames: dict[str, Frame]
    host_timestamp_spread_ns: int

    @property
    def spread_ms(self) -> float:
        return self.host_timestamp_spread_ns / 1e6


def snapshot(streams: dict[str, CameraStreamThread]) -> MultiCamSnapshot | None:
    """Grab the most recent frame from every stream. Returns None if any is missing."""
    frames: dict[str, Frame] = {}
    for alias, s in streams.items():
        f = s.latest()
        if f is None:
            return None
        frames[alias] = f
    if not frames:
        return None
    ts = [f.timestamp_ns for f in frames.values()]
    spread = max(ts) - min(ts)
    return MultiCamSnapshot(frames=frames, host_timestamp_spread_ns=int(spread))


def snapshot_best_effort(
    streams: dict[str, CameraStreamThread],
    *,
    attempts: int = 30,
    poll_interval_ms: float = 2.0,
) -> MultiCamSnapshot | None:
    """Poll ``snapshot`` several times and return the tightest-sync frame set.

    Four independent USB streams rarely expose frames at the same host instant;
    polling over ~60 ms usually finds a moment when all four ``latest()`` frames
    were received within a few tens of milliseconds of each other.
    """
    best: MultiCamSnapshot | None = None
    n = max(1, int(attempts))
    sleep_s = max(0.0, float(poll_interval_ms)) / 1000.0
    for _ in range(n):
        snap = snapshot(streams)
        if snap is None:
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue
        if best is None or snap.host_timestamp_spread_ns < best.host_timestamp_spread_ns:
            best = snap
        if sleep_s > 0:
            time.sleep(sleep_s)
    return best

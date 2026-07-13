"""Acquire synced multiview frames: always FIFO (try_pop_synced), never pop_latest."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream, SyncedMultiviewFrame
from projects.genesis_ue_sync.multiview_realtime.ingress.motion_frame_gate import (
    CanonicalMotionIndexClient,
    MotionFrameWindow,
    resolve_motion_frame_index,
)

if TYPE_CHECKING:
    pass


def try_pop_next_synced(
    stream: MultiviewCameraStream,
    *,
    motion_window: MotionFrameWindow | None = None,
    canonical: CanonicalMotionIndexClient | None = None,
    max_frame_span: int | None = 2,
    sync_mode: str | None = None,
    max_hardware_spread_ms: float | None = None,
) -> tuple[SyncedMultiviewFrame | None, int | None, str | None]:
    """Return the oldest synchronized frame (FIFO). Optionally drop frames outside motion window."""
    if not stream.ingest_running:
        stream.poll_once()
    if canonical is not None:
        canonical.poll()
    mode = str(sync_mode or stream.config.sync_mode or "hardware_timestamp").strip().lower()
    if mode == "hardware_timestamp":
        spread_ms = float(max_hardware_spread_ms if max_hardware_spread_ms is not None else stream.config.max_hardware_spread_ms)
        synced = stream.try_pop_synced_hardware(max_spread_ns=int(max(spread_ms, 1.0) * 1_000_000))
    elif max_frame_span is not None:
        synced = stream.try_pop_synced_strict(max_frame_span=int(max_frame_span))
    else:
        synced = stream.try_pop_synced()
    if synced is None:
        return None, None, "no_synced_frame"
    motion_fi = resolve_motion_frame_index(synced, canonical)
    if motion_window is not None:
        if motion_fi is None:
            return None, None, "missing_motion_frame_index"
        if not motion_window.contains(motion_fi):
            return None, motion_fi, f"outside_motion_window:{motion_fi}"
    return synced, motion_fi, None


def wait_pop_next_synced(
    stream: MultiviewCameraStream,
    *,
    motion_window: MotionFrameWindow | None = None,
    canonical: CanonicalMotionIndexClient | None = None,
    wait_timeout_s: float = 0.5,
    max_frame_span: int | None = 2,
    sync_mode: str | None = None,
    max_hardware_spread_ms: float | None = None,
) -> tuple[SyncedMultiviewFrame | None, int | None, str | None]:
    deadline = time.perf_counter() + float(wait_timeout_s)
    last_reason = "no_synced_frame"
    while time.perf_counter() < deadline:
        if canonical is not None:
            canonical.poll()
        if not stream.ingest_running:
            stream.poll_once()
        drained = 0
        while True:
            synced, motion_fi, reason = try_pop_next_synced(
                stream,
                motion_window=motion_window,
                canonical=canonical,
                max_frame_span=max_frame_span,
                sync_mode=sync_mode,
                max_hardware_spread_ms=max_hardware_spread_ms,
            )
            if synced is None:
                last_reason = reason or last_reason
                break
            drained += 1
            if motion_window is None or motion_fi is None or motion_window.contains(int(motion_fi)):
                return synced, motion_fi, None
            last_reason = reason or f"outside_motion_window:{motion_fi}"
        if drained == 0:
            time.sleep(0.002)
    return None, None, last_reason


def collect_synced_burst(
    stream: MultiviewCameraStream,
    *,
    duration_s: float = 0.5,
    wait_timeout_s: float = 2.0,
    motion_window: MotionFrameWindow | None = None,
    canonical: CanonicalMotionIndexClient | None = None,
    max_frame_span: int | None = 2,
    sync_mode: str | None = None,
    max_hardware_spread_ms: float | None = None,
) -> tuple[list[SyncedMultiviewFrame], list[int | None], str | None]:
    """Collect a producer-timestamped synchronized burst after its first group."""
    deadline = time.perf_counter() + max(float(wait_timeout_s), 0.05)
    duration_ns = int(max(float(duration_s), 0.0) * 1_000_000_000)
    frames: list[SyncedMultiviewFrame] = []
    motion_indices: list[int | None] = []
    first_timestamp_ns: int | None = None
    reason: str | None = "no_synced_frame"
    while time.perf_counter() < deadline:
        synced, motion_fi, reason = wait_pop_next_synced(
            stream,
            motion_window=motion_window,
            canonical=canonical,
            wait_timeout_s=min(0.2, max(0.01, deadline - time.perf_counter())),
            max_frame_span=max_frame_span,
            sync_mode=sync_mode,
            max_hardware_spread_ms=max_hardware_spread_ms,
        )
        if synced is None:
            continue
        if first_timestamp_ns is None:
            first_timestamp_ns = int(synced.timestamp_ns)
        if int(synced.timestamp_ns) < first_timestamp_ns:
            continue
        frames.append(synced)
        motion_indices.append(motion_fi)
        if int(synced.timestamp_ns) - first_timestamp_ns >= duration_ns:
            return frames, motion_indices, None
    return frames, motion_indices, (None if frames else reason)

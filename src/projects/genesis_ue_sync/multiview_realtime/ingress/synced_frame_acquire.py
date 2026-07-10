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
) -> tuple[SyncedMultiviewFrame | None, int | None, str | None]:
    """Return the oldest synchronized frame (FIFO). Optionally drop frames outside motion window."""
    if not stream.ingest_running:
        stream.poll_once()
    if canonical is not None:
        canonical.poll()
    synced = (
        stream.try_pop_synced_strict(max_frame_span=int(max_frame_span))
        if max_frame_span is not None
        else stream.try_pop_synced()
    )
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

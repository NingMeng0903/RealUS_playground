from __future__ import annotations

import numpy as np
import pytest

from projects.genesis_ue_sync.multiview_realtime.config import IngressConfig
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.ingress.motion_frame_gate import MotionFrameWindow
from projects.genesis_ue_sync.multiview_realtime.ingress.synced_frame_acquire import wait_pop_next_synced


CAMERAS = tuple(f"cam{i}" for i in range(1, 6))


def _stream(monkeypatch, *, mode="frame_index", capacity=8):
    stream = MultiviewCameraStream(
        IngressConfig(sync_mode=mode, max_buffer_per_camera=capacity), camera_ids=CAMERAS,
    )
    # Exercise the real synchronization buffers without opening a network socket.
    monkeypatch.setattr(stream, "poll_once", lambda: False)
    return stream


def _append(stream, camera, frame, *, timestamp=None, motion=20):
    meta = {"camera_name": camera, "frame_index": frame,
            "source_time_ns": int(timestamp if timestamp is not None else (frame + 1) * 100_000_000)}
    if motion is not None:
        meta["motion_frame_index"] = motion
    stream._buffers[camera].append((frame, meta, np.full((1, 1, 3), frame, dtype=np.uint8)))


def _group(stream, frame, **kwargs):
    for camera in CAMERAS:
        _append(stream, camera, frame, **kwargs)


def test_default_fifo_returns_oldest_and_preserves_newer_groups(monkeypatch):
    stream = _stream(monkeypatch)
    for frame in (10, 11, 12):
        _group(stream, frame)
    synced, _, reason = wait_pop_next_synced(stream, max_frame_span=0)
    assert reason is None and synced.frame_index == 10
    assert all(status["depth"] == 2 for name, status in stream.buffer_status().items() if name in CAMERAS)


def test_latest_returns_newest_complete_five_camera_group_not_individual_tails(monkeypatch):
    stream = _stream(monkeypatch)
    for frame in (10, 11, 12):
        _group(stream, frame)
    # Only four views of 13 have arrived.  Returning independent tails would
    # mix these with cam5 at 12, corrupting triangulation.
    for camera in CAMERAS[:-1]:
        _append(stream, camera, 13)
    synced, _, reason = wait_pop_next_synced(stream, max_frame_span=0, prefer_latest=True)
    assert reason is None and synced.frame_index == 12
    assert set(synced.views_rgb) == set(CAMERAS)
    assert {meta["frame_index"] for meta in synced.metadata_by_camera.values()} == {12}
    assert all(int(image[0, 0, 0]) == 12 for image in synced.views_rgb.values())
    assert [stream._buffers[camera][0][0] for camera in CAMERAS[:-1]] == [13] * 4


@pytest.mark.parametrize("tail_motion", [None, 90])
def test_invalid_motion_tail_does_not_replace_last_accepted_group(monkeypatch, tail_motion):
    stream = _stream(monkeypatch)
    _group(stream, 10, motion=10)
    _group(stream, 11, motion=11)
    _group(stream, 12, motion=tail_motion)
    synced, motion, reason = wait_pop_next_synced(
        stream, max_frame_span=0, prefer_latest=True,
        motion_window=MotionFrameWindow(motion_start=10, motion_end_exclusive=20),
    )
    assert reason is None and synced.frame_index == 11 and motion == 11


def test_latest_can_skip_rejected_head_to_find_newer_valid_group(monkeypatch):
    stream = _stream(monkeypatch)
    _group(stream, 10, motion=90)
    _group(stream, 11, motion=11)
    synced, motion, reason = wait_pop_next_synced(
        stream, max_frame_span=0, prefer_latest=True,
        motion_window=MotionFrameWindow(motion_start=10, motion_end_exclusive=20),
    )
    assert reason is None and synced.frame_index == 11 and motion == 11


def test_latest_preserves_hardware_timestamp_spread_gate(monkeypatch):
    stream = _stream(monkeypatch, mode="hardware_timestamp")
    _group(stream, 10, timestamp=1_000_000_000)
    _group(stream, 11, timestamp=1_100_000_000)
    # IDs agree at 12, but one camera is 50 ms away: reject this group with a
    # 10 ms timestamp gate, even though frame-index synchronization would pass.
    for index, camera in enumerate(CAMERAS):
        _append(stream, camera, 12, timestamp=1_200_000_000 + (50_000_000 if index == 4 else 0))
    synced, _, reason = wait_pop_next_synced(
        stream, prefer_latest=True, max_hardware_spread_ms=10.,
    )
    assert reason is None and synced.frame_index == 11
    assert synced.timestamp_ns == 1_100_000_000


def test_latest_drain_is_bounded_under_continuous_production(monkeypatch):
    stream = _stream(monkeypatch, capacity=3)
    next_frame = 1
    calls = 0
    original = stream.try_pop_synced_strict

    def refill(*, max_frame_span):
        nonlocal next_frame, calls
        _group(stream, next_frame)
        next_frame += 1
        calls += 1
        assert calls <= 3, "consumer must not chase a producer indefinitely"
        return original(max_frame_span=max_frame_span)

    monkeypatch.setattr(stream, "try_pop_synced_strict", refill)
    synced, _, reason = wait_pop_next_synced(stream, max_frame_span=0, prefer_latest=True)
    assert reason is None and synced.frame_index == 3 and calls == 3

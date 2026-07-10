from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import (
    MultiviewTrackFrame,
    MultiviewTrackerSession,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import MultiviewCameraStream
from projects.genesis_ue_sync.multiview_realtime.viz.track_skeleton_drawer import TrackSkeletonDrawer
from projects.genesis_ue_sync.sim_platform.simulation.runtime import GenesisPlatformRuntime

logger = logging.getLogger(__name__)


class LiveTrackOverlayOnRuntime:
    """Draw live track mesh on an existing Genesis runtime."""

    def __init__(
        self,
        config: MultiviewRealtimeConfig,
        runtime: GenesisPlatformRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.stream = MultiviewCameraStream(config.ingress, camera_ids=config.camera_ids)
        self.session = MultiviewTrackerSession(config)
        self._drawer: TrackSkeletonDrawer | None = None
        self._live_frame_counter = 0
        self._last_track: MultiviewTrackFrame | None = None
        self._infer_stride = max(1, int(config.genesis.inference_every_n_synced_frames))
        self._synced_seen = 0
        self._preload_error: str | None = None

    @classmethod
    def from_config_path(cls, path: Path, runtime: GenesisPlatformRuntime) -> "LiveTrackOverlayOnRuntime":
        return cls(MultiviewRealtimeConfig.load(path), runtime)

    def connect_cameras(self) -> None:
        self.stream.connect()

    def start_background_model_preload(self) -> None:
        """Load pose backend weights on a daemon thread so the Genesis viewer can open first."""

        def _work() -> None:
            try:
                self.session.preload()
                logger.info("live_track: pose backend ready")
            except Exception as exc:
                self._preload_error = repr(exc)
                logger.error("live_track: pose backend preload failed: %s", exc)

        threading.Thread(target=_work, daemon=True, name="pose-backend-preload").start()

    def close(self) -> None:
        self.stream.close()
        self.session.close()
        if self._drawer is not None:
            try:
                self._drawer._clear()
            except Exception:
                pass

    def _ensure_drawer(self) -> TrackSkeletonDrawer:
        if self._drawer is None:
            rgba = self.config.genesis.track_mesh_rgba
            self._drawer = TrackSkeletonDrawer(
                self.runtime,
                joint_rgba=(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])),
            )
        return self._drawer

    def draw_track_frame(self, track: MultiviewTrackFrame) -> None:
        drawer = self._ensure_drawer()
        drawer.draw(track.keypoints3d, track.keypoints3d_schema)
        self._live_frame_counter += 1
        self._last_track = track

    def poll_and_draw(self, *, max_ingest: int = 32) -> bool:
        """Ingest camera ZMQ messages; run pose backend when a synced triplet is ready."""
        if self._preload_error is not None:
            return False
        if self.stream._sock is None:
            self.connect_cameras()
        for _ in range(max(1, int(max_ingest))):
            self.stream.poll_once()
        synced = self.stream.try_pop_synced()
        if synced is None:
            return False
        self._synced_seen += 1
        if self._synced_seen % self._infer_stride != 0:
            return False
        track = self.session.track_synced_frame(synced)
        self.draw_track_frame(track)
        return True

    def redraw_last(self) -> bool:
        if self._last_track is None:
            return False
        self.draw_track_frame(self._last_track)
        return True

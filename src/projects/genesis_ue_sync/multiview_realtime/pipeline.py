from __future__ import annotations

import logging
import time
from typing import Iterator

from projects.genesis_ue_sync.multiview_realtime.config import MultiviewRealtimeConfig
from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import (
    MultiviewTrackFrame,
    MultiviewTrackerSession,
)
from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import (
    MultiviewCameraStream,
    SyncedMultiviewFrame,
)
from projects.genesis_ue_sync.multiview_realtime.viz.genesis_track_overlay import GenesisTrackOverlay

logger = logging.getLogger(__name__)


class MultiviewRealtimeTracker:
    """End-to-end live tracker: camera ingress -> pose backend -> optional Genesis overlay."""

    def __init__(self, config: MultiviewRealtimeConfig) -> None:
        self.config = config
        self.stream = MultiviewCameraStream(config.ingress, camera_ids=config.camera_ids)
        self.session = MultiviewTrackerSession(config)
        self._genesis: GenesisTrackOverlay | None = None

    def enable_genesis_overlay(self) -> GenesisTrackOverlay:
        overlay = GenesisTrackOverlay(self.config)
        overlay.setup()
        self._genesis = overlay
        return overlay

    @property
    def genesis_overlay(self) -> GenesisTrackOverlay | None:
        return self._genesis

    def close(self) -> None:
        self.stream.close()
        self.session.close()

    def iter_synced_frames(self) -> Iterator[SyncedMultiviewFrame]:
        self.stream.connect()
        try:
            yield from self.stream.iter_synced()
        finally:
            self.stream.close()

    def process_synced_frame(self, synced: SyncedMultiviewFrame) -> MultiviewTrackFrame:
        return self.session.track_synced_frame(synced)

    def run(self) -> None:
        if self.config.genesis.show_viewer:
            self.enable_genesis_overlay()
        min_dt = 0.0
        if float(self.config.genesis.max_track_fps) > 0.0:
            min_dt = 1.0 / float(self.config.genesis.max_track_fps)
        infer_stride = max(1, int(self.config.genesis.inference_every_n_synced_frames))
        self.stream.connect()
        synced_count = 0
        try:
            for synced in self.stream.iter_synced():
                synced_count += 1
                if synced_count % infer_stride != 0:
                    continue
                t0 = time.perf_counter()
                track = self.session.track_synced_frame(synced)
                if self._genesis is not None:
                    self._genesis.draw_track_frame(track)
                elapsed = time.perf_counter() - t0
                if synced_count % 30 == 0:
                    logger.info(
                        "track frame=%s trans_m=%s infer_s=%.3f",
                        track.frame_index,
                        [round(float(v), 3) for v in track.translation_m.tolist()],
                        elapsed,
                    )
                if min_dt > 0.0:
                    sleep_s = min_dt - (time.perf_counter() - t0)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                if self._genesis is not None and self._genesis.runtime is not None:
                    viewer = getattr(self._genesis.runtime.scene, "visualizer", None)
                    if viewer is not None and hasattr(viewer, "viewer"):
                        if not viewer.viewer.is_alive():
                            break
        finally:
            self.close()

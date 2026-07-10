from __future__ import annotations

import json
import logging

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.inference.multiview_tracker import MultiviewTrackFrame
from projects.genesis_ue_sync.multiview_realtime.track_stream import (
    DEFAULT_TRACK_PUB_BIND,
    TOPIC_MULTIVIEW_TRACK_V1,
    track_clear_to_dict,
    track_keypoints3d_to_dict,
    track_pose_to_dict,
)

logger = logging.getLogger(__name__)


class TrackPosePublisher:
    """PUB socket for latest multiview track poses."""

    def __init__(
        self,
        *,
        bind: str = DEFAULT_TRACK_PUB_BIND,
        topic: str = TOPIC_MULTIVIEW_TRACK_V1,
        publish_keypoints_fallback: bool = True,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq required for track pose publisher.") from exc
        self._topic = str(topic).encode("utf-8")
        self._publish_keypoints_fallback = bool(publish_keypoints_fallback)
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.bind(str(bind))
        logger.info("TrackPosePublisher bind=%s topic=%s", bind, topic)

    def publish(self, track: MultiviewTrackFrame) -> None:
        recon = dict(getattr(track, "reconstruction", {}) or {})
        if bool(recon.get("smpl_fit_ok")) and float(np.max(np.abs(track.pose_aa))) > 0.0:
            payload = track_pose_to_dict(
                frame_index=track.frame_index,
                timestamp_ns=track.timestamp_ns,
                pose_aa=track.pose_aa,
                betas=track.betas,
                translation_m=track.translation_m,
            )
        elif self._publish_keypoints_fallback:
            payload = track_keypoints3d_to_dict(
                frame_index=track.frame_index,
                timestamp_ns=track.timestamp_ns,
                keypoints3d=track.keypoints3d,
                schema=track.keypoints3d_schema,
                translation_m=track.translation_m,
            )
        else:
            payload = track_clear_to_dict(
                frame_index=track.frame_index,
                timestamp_ns=track.timestamp_ns,
                reason=str(recon.get("reason") or "smpl_fit_unavailable"),
            )
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self._sock.send_multipart([self._topic, body])

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass

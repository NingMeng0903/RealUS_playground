"""Undistort a short hardware-sync burst so the live publisher can stay raw."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.ingress.camera_stream import SyncedMultiviewFrame

logger = logging.getLogger(__name__)


def _as_k(meta: dict[str, Any]) -> np.ndarray:
    geometry = dict(meta.get("image_geometry") or {})
    intrinsics = dict(meta.get("intrinsics") or {})
    raw = geometry.get("effective_K") or intrinsics.get("K")
    if raw is None:
        raise ValueError("frame metadata missing K (image_geometry.effective_K / intrinsics.K)")
    k = np.asarray(raw, dtype=np.float64).reshape(3, 3)
    return k


def _as_dist(meta: dict[str, Any]) -> np.ndarray:
    dist = np.asarray((meta.get("intrinsics") or {}).get("distortion") or [0.0] * 5, dtype=np.float64).reshape(-1)
    if dist.size < 5:
        dist = np.pad(dist, (0, 5 - dist.size))
    return dist[:5]


def _already_undistorted(meta: dict[str, Any]) -> bool:
    geometry = dict(meta.get("image_geometry") or {})
    return bool(geometry.get("undistorted")) and str(geometry.get("projection_distortion_model") or "") == "zero"


def _mark_undistorted(meta: dict[str, Any], k: np.ndarray) -> dict[str, Any]:
    out = dict(meta)
    geometry = dict(out.get("image_geometry") or {})
    geometry["undistorted"] = True
    geometry["projection_distortion_model"] = "zero"
    geometry["effective_K"] = k.tolist()
    out["image_geometry"] = geometry
    return out


def ensure_undistorted_burst(
    frames: list[SyncedMultiviewFrame],
    camera_ids: list[str],
) -> list[SyncedMultiviewFrame]:
    """Return burst frames that satisfy undistorted/zero-distortion DLT contract.

    Live Cam can publish raw JPEG. This remaps only the 8–12 synced groups.
    Already-undistorted frames are left as-is.
    """
    if not frames:
        return frames
    import cv2

    maps: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    did_remap = False
    out: list[SyncedMultiviewFrame] = []
    for frame in frames:
        views = dict(frame.views_rgb)
        metas = {cid: dict(frame.metadata_by_camera.get(cid) or {}) for cid in camera_ids}
        for extra, meta in frame.metadata_by_camera.items():
            if extra not in metas:
                metas[extra] = dict(meta)
        for cid in camera_ids:
            meta = metas.get(cid) or {}
            if not meta:
                raise ValueError(f"{cid} frame {frame.frame_index} missing metadata")
            rgb = views.get(cid)
            if rgb is None:
                raise ValueError(f"{cid} frame {frame.frame_index} missing RGB")
            if _already_undistorted(meta):
                continue
            k = _as_k(meta)
            dist = _as_dist(meta)
            h, w = np.asarray(rgb).shape[:2]
            cached = maps.get(cid)
            if cached is None:
                map_x, map_y = cv2.initUndistortRectifyMap(k, dist, None, k, (int(w), int(h)), cv2.CV_32FC1)
                maps[cid] = (map_x, map_y, k)
            else:
                map_x, map_y, k = cached
            views[cid] = cv2.remap(np.asarray(rgb), map_x, map_y, interpolation=cv2.INTER_LINEAR)
            metas[cid] = _mark_undistorted(meta, k)
            did_remap = True
        out.append(
            SyncedMultiviewFrame(
                frame_index=int(frame.frame_index),
                views_rgb=views,
                metadata_by_camera=metas,
                timestamp_ns=int(frame.timestamp_ns),
            )
        )
    if did_remap:
        logger.info("undistorted %d burst groups in capture (publisher raw)", len(out))
    return out

"""Reject multiview frames where the subject is leaving the image or poorly observed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptureFrameQualityConfig:
    min_valid_body25_per_cam: int = 18
    min_cameras_passing: int = 5
    edge_margin_frac: float = 0.08
    min_bbox_area_frac: float = 0.012
    max_bbox_area_frac: float = 0.92
    confidence_threshold: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "CaptureFrameQualityConfig":
        payload = dict(payload or {})
        return cls(
            min_valid_body25_per_cam=max(1, int(payload.get("min_valid_body25_per_cam", cls.min_valid_body25_per_cam))),
            min_cameras_passing=max(1, int(payload.get("min_cameras_passing", cls.min_cameras_passing))),
            edge_margin_frac=float(payload.get("edge_margin_frac", cls.edge_margin_frac)),
            min_bbox_area_frac=float(payload.get("min_bbox_area_frac", cls.min_bbox_area_frac)),
            max_bbox_area_frac=float(payload.get("max_bbox_area_frac", cls.max_bbox_area_frac)),
            confidence_threshold=float(payload.get("confidence_threshold", cls.confidence_threshold)),
        )


def _body_bbox_norm(
    keypoints: np.ndarray,
    *,
    conf_th: float,
    img_wh: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    if kp.shape[0] < 8:
        return None
    valid = np.isfinite(kp[:, 0]) & np.isfinite(kp[:, 1]) & (kp[:, 2] >= float(conf_th))
    if int(np.sum(valid)) < 6:
        return None
    pts = kp[valid, :2]
    w = max(1, int(img_wh[0]))
    h = max(1, int(img_wh[1]))
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    cx = 0.5 * (xmin + xmax) / float(w)
    cy = 0.5 * (ymin + ymax) / float(h)
    bw = (xmax - xmin) / float(w)
    bh = (ymax - ymin) / float(h)
    return float(cx), float(cy), float(bw), float(bh)


def evaluate_multiview_capture_quality(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    views_rgb: dict[str, np.ndarray],
    *,
    config: CaptureFrameQualityConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = config or CaptureFrameQualityConfig()
    per_cam: dict[str, Any] = {}
    passing = 0
    for camera_id, annot in annots_by_cam.items():
        rgb = np.asarray(views_rgb[camera_id])
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        body = np.asarray(annot.get("keypoints"), dtype=np.float32).reshape(-1, 3)
        conf = body[:, 2] if body.size else np.zeros(0, dtype=np.float32)
        valid_n = int(np.sum(conf >= float(cfg.confidence_threshold)))
        bbox = _body_bbox_norm(body, conf_th=float(cfg.confidence_threshold), img_wh=(w, h))
        reasons: list[str] = []
        if valid_n < int(cfg.min_valid_body25_per_cam):
            reasons.append("min_valid_body25")
        if bbox is None:
            reasons.append("bbox_missing")
        else:
            cx, cy, bw, bh = bbox
            m = float(cfg.edge_margin_frac)
            if cx < m or cy < m or cx > (1.0 - m) or cy > (1.0 - m):
                reasons.append("subject_near_edge")
            area = float(bw * bh)
            if area < float(cfg.min_bbox_area_frac):
                reasons.append("subject_too_small")
            if area > float(cfg.max_bbox_area_frac):
                reasons.append("subject_too_large")
        ok_cam = not reasons
        if ok_cam:
            passing += 1
        per_cam[camera_id] = {
            "ok": ok_cam,
            "valid_body25": valid_n,
            "bbox_norm": list(bbox) if bbox is not None else None,
            "reasons": reasons,
        }
    ok = passing >= int(cfg.min_cameras_passing)
    report = {
        "ok": ok,
        "cameras_passing": int(passing),
        "min_cameras_passing": int(cfg.min_cameras_passing),
        "per_camera": per_cam,
    }
    if not ok:
        report["reject_reasons"] = ["multiview_capture_quality"]
    return ok, report

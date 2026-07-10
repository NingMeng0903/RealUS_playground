"""Frame quality checks for multiview beta calibration and SMPL fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from projects.genesis_ue_sync.multiview_realtime.fitting.capture_frame_quality import (
    CaptureFrameQualityConfig,
    evaluate_multiview_capture_quality,
)
from projects.genesis_ue_sync.multiview_realtime.fitting.smpl_shape_calibration import (
    ShapeFrameQualityConfig,
    evaluate_shape_frame_quality,
)

OPENPOSE_BODY_JOINTS = 25
OPENPOSE_HAND_JOINTS = 21
OPENPOSE_FACE_JOINTS = 70


@dataclass(frozen=True)
class Pose2dFrameQualityConfig:
    capture: CaptureFrameQualityConfig = CaptureFrameQualityConfig()
    shape3d: ShapeFrameQualityConfig = ShapeFrameQualityConfig()
    min_valid_hand_per_cam: int = 10
    min_valid_face_per_cam: int = 12
    min_cameras_with_hands: int = 3
    require_shape3d: bool = True
    min_valid_hand3d: int = 6
    reject_on_hand3d: bool = False
    reject_on_2d_capture: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "Pose2dFrameQualityConfig":
        payload = dict(payload or {})
        return cls(
            capture=CaptureFrameQualityConfig.from_dict(payload.get("capture")),
            shape3d=ShapeFrameQualityConfig.from_dict(payload.get("shape3d")),
            min_valid_hand_per_cam=max(0, int(payload.get("min_valid_hand_per_cam", cls.min_valid_hand_per_cam))),
            min_valid_face_per_cam=max(0, int(payload.get("min_valid_face_per_cam", cls.min_valid_face_per_cam))),
            min_cameras_with_hands=max(0, int(payload.get("min_cameras_with_hands", cls.min_cameras_with_hands))),
            require_shape3d=bool(payload.get("require_shape3d", cls.require_shape3d)),
            min_valid_hand3d=max(0, int(payload.get("min_valid_hand3d", cls.min_valid_hand3d))),
            reject_on_hand3d=bool(payload.get("reject_on_hand3d", cls.reject_on_hand3d)),
            reject_on_2d_capture=bool(payload.get("reject_on_2d_capture", cls.reject_on_2d_capture)),
        )


def _count_valid(kp: np.ndarray, *, conf_th: float) -> int:
    arr = np.asarray(kp, dtype=np.float32).reshape(-1, 3)
    if arr.size == 0:
        return 0
    ok = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 2] >= float(conf_th))
    return int(np.sum(ok))


def evaluate_body25_views_quality(
    keypoints2d_by_cam: dict[str, np.ndarray],
    views_rgb: dict[str, np.ndarray],
    *,
    config: Pose2dFrameQualityConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = config or Pose2dFrameQualityConfig()
    annots_stub = {
        cid: {"keypoints": np.asarray(keypoints2d_by_cam[cid], dtype=np.float32)}
        for cid in keypoints2d_by_cam
    }
    ok_cap, cap_report = evaluate_multiview_capture_quality(
        annots_stub,
        views_rgb,
        config=cfg.capture,
    )
    report: dict[str, Any] = {"capture": cap_report, "reasons": []}
    if not ok_cap:
        report["reasons"].extend(cap_report.get("reject_reasons") or ["capture_quality"])
        report["ok"] = False
        return False, report
    report["ok"] = True
    return True, report


def evaluate_easymocap_annot_quality(
    annots_by_cam: dict[str, dict[str, np.ndarray]],
    views_rgb: dict[str, np.ndarray],
    *,
    keypoints3d_world: np.ndarray | None = None,
    config: Pose2dFrameQualityConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = config or Pose2dFrameQualityConfig()
    conf_th = float(cfg.capture.confidence_threshold)
    ok_cap, cap_report = evaluate_multiview_capture_quality(annots_by_cam, views_rgb, config=cfg.capture)
    report: dict[str, Any] = {"capture": cap_report, "per_camera": {}, "reasons": []}
    hands_ok = 0
    for camera_id, annot in annots_by_cam.items():
        hand_l = _count_valid(annot.get("handl2d"), conf_th=conf_th)
        hand_r = _count_valid(annot.get("handr2d"), conf_th=conf_th)
        face_n = _count_valid(annot.get("face2d"), conf_th=conf_th)
        cam_reasons: list[str] = []
        if hand_l < int(cfg.min_valid_hand_per_cam):
            cam_reasons.append("handl_sparse")
        if hand_r < int(cfg.min_valid_hand_per_cam):
            cam_reasons.append("handr_sparse")
        if face_n < int(cfg.min_valid_face_per_cam):
            cam_reasons.append("face_sparse")
        if not cam_reasons:
            hands_ok += 1
        report["per_camera"][camera_id] = {
            "valid_handl": hand_l,
            "valid_handr": hand_r,
            "valid_face": face_n,
            "reasons": cam_reasons,
        }
    if cfg.reject_on_2d_capture and not ok_cap:
        report["reasons"].extend(cap_report.get("reject_reasons") or ["capture_quality"])
    if cfg.reject_on_hand3d and hands_ok < int(cfg.min_cameras_with_hands):
        report["reasons"].append("min_cameras_with_hands")
    if cfg.require_shape3d and keypoints3d_world is not None:
        ok3d, shape_report = evaluate_shape_frame_quality(keypoints3d_world, cfg.shape3d)
        report["shape3d"] = shape_report
        if not ok3d:
            report["reasons"].extend(shape_report.get("reasons") or ["shape3d_quality"])
    report["ok"] = not report["reasons"]
    return bool(report["ok"]), report


def _valid3d_count(kp3d: np.ndarray, *, conf_th: float) -> int:
    arr = np.asarray(kp3d, dtype=np.float32).reshape(-1, 4)
    ok = np.all(np.isfinite(arr[:, :3]), axis=1) & (arr[:, 3] >= float(conf_th))
    return int(np.sum(ok))


def evaluate_bodyhand3d_quality(
    parts3d: dict[str, np.ndarray],
    *,
    config: Pose2dFrameQualityConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    cfg = config or Pose2dFrameQualityConfig()
    body = np.asarray(parts3d.get("keypoints3d"), dtype=np.float32).reshape(-1, 4)
    ok_body, body_report = evaluate_shape_frame_quality(body, cfg.shape3d)
    hand_l = _valid3d_count(parts3d.get("handl3d", np.zeros((0, 4), dtype=np.float32)), conf_th=cfg.shape3d.confidence_threshold)
    hand_r = _valid3d_count(parts3d.get("handr3d", np.zeros((0, 4), dtype=np.float32)), conf_th=cfg.shape3d.confidence_threshold)
    reasons: list[str] = []
    if cfg.require_shape3d and not ok_body:
        reasons.extend(body_report.get("reasons") or ["body3d_quality"])
    if cfg.reject_on_hand3d and (hand_l < int(cfg.min_valid_hand3d) or hand_r < int(cfg.min_valid_hand3d)):
        reasons.append("hand3d_sparse")
    return not reasons, {
        "ok": not reasons,
        "reasons": reasons,
        "body3d": body_report,
        "valid_handl3d": int(hand_l),
        "valid_handr3d": int(hand_r),
        "reject_on_hand3d": bool(cfg.reject_on_hand3d),
    }

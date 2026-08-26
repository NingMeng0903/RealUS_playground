"""AprilTag detection wrapping pupil_apriltags.

Returns per-tag pixel corners in the **raw pupil_apriltags order**. Consumers
that pair these corners with the board-frame model (``BoardGeometry.corners_by_tag``)
must first call ``corners_to_board_frame`` — this is done in
``BoardGeometry.gather_correspondences`` for PnP and in the Stage 1 BA
observation builder. Keeping raw order on disk avoids double-remapping when
the same JSON is consumed by different code paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pupil_apriltags import Detector

from multicam_calib.io.config import BoardConfig, DetectorConfig

# Default = large bed board. pupil_apriltags corner index order differs from
# board-frame [BL, BR, TR, TL]. Empirically: bed board [3,0,1,2], EE board
# [1,2,3,0] (tags printed 90° relative to the bed sheet). Prefer
# ``BoardConfig.pupil_to_board_corner_perm`` via ``gather_correspondences``.
_PUPIL_TO_BOARD_CORNER_PERM = np.array([3, 0, 1, 2], dtype=np.int64)


def corners_to_board_frame(
    corners: np.ndarray,
    perm: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Map pupil_apriltags corners to board-model corner order [BL, BR, TR, TL]."""
    c = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    p = _PUPIL_TO_BOARD_CORNER_PERM if perm is None else np.asarray(perm, dtype=np.int64).reshape(4)
    return c[p]


@dataclass
class TagDetection:
    tag_id: int
    corners: np.ndarray  # (4, 2) float64, pixel coords in [BL, BR, TR, TL] order
    center: np.ndarray   # (2,) float64
    decision_margin: float
    hamming: int


def scale_detections(
    dets: list[TagDetection],
    *,
    from_wh: tuple[int, int],
    to_wh: tuple[int, int],
) -> list[TagDetection]:
    """Map detections from a resized detect image back to the display/capture frame."""
    fw, fh = int(from_wh[0]), int(from_wh[1])
    tw, th = int(to_wh[0]), int(to_wh[1])
    if fw <= 0 or fh <= 0 or (fw == tw and fh == th):
        return dets
    sx = tw / float(fw)
    sy = th / float(fh)
    out: list[TagDetection] = []
    for d in dets:
        out.append(
            TagDetection(
                tag_id=d.tag_id,
                corners=np.asarray(d.corners, dtype=np.float64).reshape(4, 2) * np.array([sx, sy]),
                center=np.asarray(d.center, dtype=np.float64).reshape(2) * np.array([sx, sy]),
                decision_margin=d.decision_margin,
                hamming=d.hamming,
            )
        )
    return out


class AprilTagDetector:
    """Reusable tag detector bound to one AprilTag family."""

    def __init__(self, board: BoardConfig, det_cfg: DetectorConfig | None = None) -> None:
        det_cfg = det_cfg or DetectorConfig()
        self._detector = Detector(
            families=board.family,
            nthreads=int(det_cfg.nthreads),
            quad_decimate=float(det_cfg.quad_decimate),
            quad_sigma=float(det_cfg.quad_sigma),
            refine_edges=int(det_cfg.refine_edges),
            decode_sharpening=float(det_cfg.decode_sharpening),
        )
        self._valid_ids = board.all_expected_ids()
        # Grazing / tiny quads make OpenCV invert a singular homography and
        # spam "WRN: Matrix is singular". Harmless for live preview.
        # OpenCV 4.11 wheels expose setLogLevel but not LOG_LEVEL_ERROR (2 = ERROR).
        cv2.setLogLevel(2)

    def detect(self, image_bgr: np.ndarray) -> list[TagDetection]:
        """Detect tags on this board in a BGR image."""
        if image_bgr.ndim == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr
        # pupil_apriltags requires contiguous uint8.
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        raw = self._detector.detect(gray)
        out: list[TagDetection] = []
        for d in raw:
            tag_id = int(d.tag_id)
            if tag_id not in self._valid_ids:
                continue
            out.append(
                TagDetection(
                    tag_id=tag_id,
                    corners=np.asarray(d.corners, dtype=np.float64).reshape(4, 2),
                    center=np.asarray(d.center, dtype=np.float64).reshape(2),
                    decision_margin=float(d.decision_margin),
                    hamming=int(d.hamming),
                )
            )
        return out

    @staticmethod
    def detections_to_dict(dets: list[TagDetection]) -> dict[int, np.ndarray]:
        """Collapse a detection list to ``{tag_id: (4,2) corners}``."""
        return {d.tag_id: d.corners for d in dets}


def draw_detections(image_bgr: np.ndarray, dets: list[TagDetection]) -> np.ndarray:
    """Draw tag outlines + IDs on a copy of the image (for UI overlay)."""
    canvas = image_bgr.copy()
    for d in dets:
        pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cx, cy = int(round(d.center[0])), int(round(d.center[1]))
        cv2.putText(
            canvas,
            str(d.tag_id),
            (cx - 10, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas

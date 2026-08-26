"""Geometric model of the AprilTag calibration board.

Board coordinate frame:
- Origin at the *geometric center* of the board (center of the whole grid).
- X axis: along columns, pointing from column 0 to column (cols-1) — "right".
- Y axis: in the board plane, perpendicular to X, pointing from row (rows-1)
  to row 0 — "up" (i.e. opposite to row-index growth).
- Z axis: out of the front of the board (right-handed).

Tag corner order in the board frame (after ``corners_to_board_frame`` remap):
    [0] bottom-left, [1] bottom-right, [2] top-right, [3] top-left
where +X is right along columns and +Y is up along rows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multicam_calib.io.config import BoardConfig
from multicam_calib.board.detector import corners_to_board_frame


@dataclass
class BoardGeometry:
    """Cached 3D corner geometry for one board.

    ``corners_by_tag[tag_id]`` is a (4, 3) array of the tag's 4 corners in the
    board frame, ordered [BL, BR, TR, TL] to match pupil_apriltags. All units
    are metres.
    """

    config: BoardConfig
    corners_by_tag: dict[int, np.ndarray]

    @property
    def all_tag_ids(self) -> set[int]:
        return set(self.corners_by_tag.keys())

    def width_m(self) -> float:
        """Board width along X (from leftmost tag left edge to rightmost tag right edge)."""
        c = self.config
        return (c.cols - 1) * c.pitch_m + c.tag_size_m

    def height_m(self) -> float:
        c = self.config
        return (c.rows - 1) * c.pitch_m + c.tag_size_m

    def gather_correspondences(
        self, detections: dict[int, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Return matched 3D-2D correspondences for use with cv2.solvePnP.

        Parameters
        ----------
        detections
            ``{tag_id: (4, 2) float array of pixel corners in [BL, BR, TR, TL] order}``.
            Extra tag ids not on this board are silently ignored.

        Returns
        -------
        object_points : (N, 3) float64
        image_points : (N, 2) float64
        used_tag_ids : list of tag ids that contributed
        """
        obj: list[np.ndarray] = []
        img: list[np.ndarray] = []
        used: list[int] = []
        for tag_id, px in detections.items():
            model = self.corners_by_tag.get(int(tag_id))
            if model is None:
                continue
            px_arr = corners_to_board_frame(px, self.config.pupil_to_board_corner_perm)
            obj.append(model)
            img.append(px_arr)
            used.append(int(tag_id))
        if not obj:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                [],
            )
        return (
            np.concatenate(obj, axis=0).astype(np.float64),
            np.concatenate(img, axis=0).astype(np.float64),
            used,
        )


def build_board_geometry(cfg: BoardConfig) -> BoardGeometry:
    """Precompute the 4-corner 3D coordinate of every tag on the board.

    The board's geometric center is (0, 0, 0). Tag (row=0, col=0) is at the
    top-left of the printed board. Tag centres sit at:

        cx =   -X_span/2 + col * pitch
        cy =   +Y_span/2 - row * pitch

    where X_span = (cols-1) * pitch and Y_span = (rows-1) * pitch. Each tag has
    its own 4 corners at ±(tag_size/2) around (cx, cy) in the board plane.
    """
    half = 0.5 * cfg.tag_size_m
    x_span = (cfg.cols - 1) * cfg.pitch_m
    y_span = (cfg.rows - 1) * cfg.pitch_m

    corners: dict[int, np.ndarray] = {}
    for row in range(cfg.rows):
        for col in range(cfg.cols):
            tag_id = cfg.tag_id(row, col)
            cx = -0.5 * x_span + col * cfg.pitch_m
            cy = +0.5 * y_span - row * cfg.pitch_m
            # Board-frame corner order (after corners_to_board_frame): BL, BR, TR, TL.
            corners[tag_id] = np.array(
                [
                    [cx - half, cy - half, 0.0],
                    [cx + half, cy - half, 0.0],
                    [cx + half, cy + half, 0.0],
                    [cx - half, cy + half, 0.0],
                ],
                dtype=np.float64,
            )
    return BoardGeometry(config=cfg, corners_by_tag=corners)

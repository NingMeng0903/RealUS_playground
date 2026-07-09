"""Per-camera pinhole intrinsics: chessboard calibration + fallback loader.

`load_intrinsics_for_aliases` is the canonical loader used by every stage.
Priority per alias:
  1. `calibration_results/intrinsics.yaml` if the alias has an entry.
  2. Otherwise the driver's factory intrinsics (RealSense EEPROM).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from multicam_calib.devices.base import CameraDevice
from multicam_calib.io.results import Intrinsics, load_intrinsics_map, upsert_intrinsics


def load_intrinsics_for_aliases(
    aliases: list[str],
    *,
    devices: dict[str, CameraDevice] | None = None,
    file_path: Path | None = None,
) -> dict[str, Intrinsics]:
    """Return intrinsics for every alias, with file-overrides-factory semantics.

    - Reads `calibration_results/intrinsics.yaml` first; any alias with a stored
      entry uses that directly.
    - For missing aliases, asks the corresponding CameraDevice for its factory
      intrinsics (requires the device to be already open()'d).
    """
    file_map = load_intrinsics_map(file_path)
    out: dict[str, Intrinsics] = {}
    missing: list[str] = []
    for a in aliases:
        if a in file_map:
            out[a] = file_map[a]
        else:
            missing.append(a)
    if missing:
        if devices is None:
            raise RuntimeError(
                f"Aliases without stored intrinsics: {missing}. "
                "Pass `devices=` so we can query factory intrinsics."
            )
        for a in missing:
            dev = devices.get(a)
            if dev is None:
                raise RuntimeError(f"No device provided for alias {a!r} and no stored intrinsics.")
            out[a] = dev.factory_intrinsics()
    return out


# --- chessboard capture / solve ---

@dataclass
class ChessboardConfig:
    """Inner-corner-based chessboard geometry."""

    cols: int              # number of INNER corners horizontally (squares - 1)
    rows: int              # number of INNER corners vertically
    square_size_m: float   # physical size of one square edge


@dataclass
class ChessboardCaptures:
    """Accumulated 2D-3D correspondences for one camera's chessboard set."""

    cfg: ChessboardConfig
    image_size: tuple[int, int] | None = None  # (width, height)
    object_points: list[np.ndarray] = field(default_factory=list)   # each (N, 3)
    image_points: list[np.ndarray] = field(default_factory=list)    # each (N, 2)

    def object_grid(self) -> np.ndarray:
        c, r = self.cfg.cols, self.cfg.rows
        pts = np.zeros((c * r, 3), dtype=np.float32)
        grid = np.mgrid[0:c, 0:r].T.reshape(-1, 2).astype(np.float32)
        pts[:, :2] = grid * self.cfg.square_size_m
        return pts

    def try_add(self, image_bgr: np.ndarray) -> bool:
        """Detect a chessboard in the image; if found, append to captures."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if self.image_size is None:
            self.image_size = (w, h)
        elif self.image_size != (w, h):
            raise ValueError(f"Image size mismatch (had {self.image_size}, got {(w, h)})")

        found, corners = cv2.findChessboardCornersSB(
            gray, (self.cfg.cols, self.cfg.rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
        if not found:
            return False
        # findChessboardCornersSB already returns sub-pixel corners; still refine
        # to catch light corner shift caused by motion blur.
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)

        self.object_points.append(self.object_grid())
        self.image_points.append(corners.reshape(-1, 2))
        return True

    def num_captures(self) -> int:
        return len(self.image_points)

    def solve(self) -> tuple[Intrinsics, float]:
        """Run cv2.calibrateCamera and return (Intrinsics, reprojection RMSE)."""
        if self.image_size is None or not self.image_points:
            raise RuntimeError("No captures yet.")
        obj = [p.astype(np.float32) for p in self.object_points]
        img = [p.astype(np.float32) for p in self.image_points]
        rms, K, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            obj, img, self.image_size, None, None
        )
        return (
            Intrinsics(
                K=np.asarray(K, dtype=np.float64),
                dist=np.asarray(dist, dtype=np.float64).reshape(-1),
                image_size=self.image_size,
                source="chessboard",
            ),
            float(rms),
        )


def persist_intrinsics(alias: str, intr: Intrinsics, path: Path | None = None) -> None:
    upsert_intrinsics({alias: intr}, path)

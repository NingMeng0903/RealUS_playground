"""Calibration-world vs Genesis viewer coordinates (bridge alignment)."""

from __future__ import annotations

import numpy as np

from projects.genesis_ue_sync.tracking.calibration import CalibrationBundle


def genesis_m_from_calibration_world(
    points_m: np.ndarray,
    calibration: CalibrationBundle,
) -> np.ndarray:
    """Map tracking/triangulation world (cameras.yaml) into Genesis scene meters."""
    mat = np.asarray(calibration.convention.world_from_genesis, dtype=np.float64).reshape(4, 4)
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim == 1:
        homo = np.concatenate([pts.reshape(3), [1.0]])
        return (mat @ homo)[:3].astype(np.float32)
    homo = np.concatenate([pts.reshape(-1, 3), np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (homo @ mat.T)[:, :3].astype(np.float32)

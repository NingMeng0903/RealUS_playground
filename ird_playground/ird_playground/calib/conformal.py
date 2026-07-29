"""Split-conformal calibration for conservative clearance."""

from __future__ import annotations

import numpy as np


def fit_conformal(
    scores: np.ndarray,
    labels_reachable: np.ndarray,
    alpha: float = 0.01,
) -> float:
    """Return a one-sided conformal threshold for target miscoverage ``alpha``.

    Nonconformity is ``-score`` when the pose is reachable and ``+score`` when
    it is not.  The returned threshold is subtracted from predictions to obtain
    conservative clearance.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels_reachable, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels_reachable must have the same shape")
    if scores.size == 0:
        return 0.0
    nc = np.where(labels, -scores, scores)
    level = min(1.0, max(0.0, 1.0 - float(alpha)))
    rank = int(np.ceil((scores.size + 1) * level))
    rank = min(max(rank, 1), scores.size)
    return float(np.sort(nc)[rank - 1])


def calibrated_clearance(pred: np.ndarray, threshold: float) -> np.ndarray:
    """Conservative clearance: ``pred - threshold``."""
    return np.asarray(pred, dtype=np.float64) - float(threshold)


__all__ = ["calibrated_clearance", "fit_conformal"]

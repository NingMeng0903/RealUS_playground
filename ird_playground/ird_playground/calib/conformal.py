"""Split-conformal calibration for conservative clearance thresholds.

Given calibration scores ``s`` (positive ⇒ reachable) and boolean reachable
labels, nonconformity is ``-s`` on reachable poses and ``+s`` on unreachable
ones.  The returned threshold ``q`` yields conservative clearance
``s - q`` with finite-sample coverage guarantees under exchangeability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalResult:
    threshold: float
    alpha: float
    n_calib: int
    quantile_level: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_conformal(
    scores: np.ndarray,
    labels_reachable: np.ndarray,
    alpha: float = 0.01,
) -> float:
    """Return a one-sided split-conformal threshold for miscoverage ``alpha``."""
    return fit_split_conformal(scores, labels_reachable, alpha=alpha).threshold


def fit_split_conformal(
    scores: np.ndarray,
    labels_reachable: np.ndarray,
    alpha: float = 0.01,
) -> ConformalResult:
    """Fit a split-conformal threshold from calibration scores and labels.

    Nonconformity is ``-score`` when the pose is reachable and ``+score`` when
    it is not.  Predictions are made conservative by subtracting the threshold
    from raw scores (see :func:`calibrated_clearance`).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels_reachable, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels_reachable must have the same shape")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if scores.size == 0:
        return ConformalResult(threshold=0.0, alpha=float(alpha), n_calib=0, quantile_level=1.0 - float(alpha))
    nc = np.where(labels, -scores, scores)
    level = min(1.0, max(0.0, 1.0 - float(alpha)))
    # Standard split-conformal finite-sample correction: ceil((n+1)·(1-α))/n.
    rank = int(np.ceil((scores.size + 1) * level))
    rank = min(max(rank, 1), scores.size)
    threshold = float(np.sort(nc)[rank - 1])
    return ConformalResult(
        threshold=threshold,
        alpha=float(alpha),
        n_calib=int(scores.size),
        quantile_level=level,
    )


def calibrated_clearance(pred: np.ndarray, threshold: float) -> np.ndarray:
    """Conservative clearance: ``pred - threshold``."""
    return np.asarray(pred, dtype=np.float64) - float(threshold)


def predict_reachable(
    pred: np.ndarray,
    threshold: float,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Reachable iff calibrated clearance is at least ``margin``."""
    return calibrated_clearance(pred, threshold) >= float(margin)


def empirical_coverage(
    scores: np.ndarray,
    labels_reachable: np.ndarray,
    threshold: float,
    *,
    margin: float = 0.0,
) -> dict[str, float]:
    """Report coverage of the conservative predictor on a held-out set.

    For reachable labels we require calibrated score ≥ ``margin`` (no false
    reject under the conformal set).  For unreachable labels we require
    calibrated score < ``margin`` (no false accept).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels_reachable, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels_reachable must have the same shape")
    if scores.size == 0:
        return {
            "coverage_reachable": float("nan"),
            "coverage_unreachable": float("nan"),
            "coverage_overall": float("nan"),
            "n": 0.0,
        }
    cal = calibrated_clearance(scores, threshold)
    pred_pos = cal >= float(margin)
    cov_pos = float(pred_pos[labels].mean()) if labels.any() else float("nan")
    cov_neg = float((~pred_pos[~labels]).mean()) if (~labels).any() else float("nan")
    correct = pred_pos == labels
    return {
        "coverage_reachable": cov_pos,
        "coverage_unreachable": cov_neg,
        "coverage_overall": float(correct.mean()),
        "n": float(scores.size),
        "threshold": float(threshold),
        "margin": float(margin),
    }


__all__ = [
    "ConformalResult",
    "calibrated_clearance",
    "empirical_coverage",
    "fit_conformal",
    "fit_split_conformal",
    "predict_reachable",
]

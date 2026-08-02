"""Independent geometric-zero and one-sided false-accept calibration."""

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


@dataclass(frozen=True)
class ZeroBiasResult:
    zero_bias: float
    n_calib: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SafetyThresholdResult:
    safety_threshold: float
    alpha: float
    n_unreachable: int
    quantile_level: float

    def to_dict(self) -> dict:
        return asdict(self)


def fit_zero_bias(
    on_manifold_zero_scores: np.ndarray,
    *,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> ZeroBiasResult:
    """Fit a robust additive bias from independently held-out boundary poses."""
    scores = np.asarray(on_manifold_zero_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("zero-bias calibration requires finite boundary scores")
    bias = float(np.median(scores))
    rng = np.random.default_rng(seed)
    centered_medians = np.empty(max(1, int(bootstrap_samples)), dtype=np.float64)
    for i in range(len(centered_medians)):
        sample = rng.choice(scores, size=len(scores), replace=True)
        centered_medians[i] = np.median(sample) - bias
    lo, hi = np.quantile(centered_medians, [0.025, 0.975])
    return ZeroBiasResult(bias, int(scores.size), float(lo), float(hi))


def fit_unreachable_safety_threshold(
    geometric_scores_unreachable: np.ndarray,
    *,
    alpha: float = 0.05,
) -> SafetyThresholdResult:
    """Control false accepts using an order statistic of unreachable scores."""
    scores = np.asarray(geometric_scores_unreachable, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if scores.size == 0:
        raise ValueError("safety calibration requires unreachable samples")
    level = 1.0 - float(alpha)
    rank = min(scores.size, max(1, int(np.ceil((scores.size + 1) * level))))
    threshold = float(np.sort(scores)[rank - 1])
    return SafetyThresholdResult(threshold, float(alpha), int(scores.size), level)


def false_acceptance_report(
    geometric_scores: np.ndarray,
    labels_reachable: np.ndarray,
    safety_threshold: float,
    *,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Report false accepts and an exact beta-binomial upper confidence bound."""
    from scipy.stats import beta

    scores = np.asarray(geometric_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels_reachable, dtype=bool).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels_reachable must have the same shape")
    accepted = scores >= float(safety_threshold)
    unreachable = ~labels
    n = int(unreachable.sum())
    false_accepts = int((accepted & unreachable).sum())
    upper = (
        1.0
        if n == 0
        else float(beta.ppf(confidence, false_accepts + 1, n - false_accepts))
    )
    return {
        "false_accepts": float(false_accepts),
        "n_unreachable": float(n),
        "false_accept_rate": float(false_accepts / n) if n else float("nan"),
        "false_accept_rate_upper": upper,
        "confidence": float(confidence),
        "reachable_recall": float(accepted[labels].mean()) if labels.any() else float("nan"),
    }


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


def geometric_clearance(pred: np.ndarray, zero_bias: float) -> np.ndarray:
    """Return the learned geometric score with its boundary zero restored."""
    return np.asarray(pred, dtype=np.float64) - float(zero_bias)


def accepted_reachability(
    pred: np.ndarray,
    *,
    zero_bias: float,
    safety_threshold: float,
) -> np.ndarray:
    return geometric_clearance(pred, zero_bias) >= float(safety_threshold)


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
    "SafetyThresholdResult",
    "ZeroBiasResult",
    "calibrated_clearance",
    "accepted_reachability",
    "empirical_coverage",
    "false_acceptance_report",
    "fit_conformal",
    "fit_split_conformal",
    "fit_unreachable_safety_threshold",
    "fit_zero_bias",
    "geometric_clearance",
    "predict_reachable",
]

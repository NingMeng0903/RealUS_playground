"""Post-hoc calibration for conservative clearance queries."""

from __future__ import annotations

import json
from pathlib import Path

from ird_playground.calib.conformal import (
    ConformalResult,
    SafetyThresholdResult,
    ZeroBiasResult,
    accepted_reachability,
    calibrated_clearance,
    empirical_coverage,
    false_acceptance_report,
    fit_conformal,
    fit_split_conformal,
    fit_unreachable_safety_threshold,
    fit_zero_bias,
    geometric_clearance,
    predict_reachable,
)


def load_conformal_json(path: str | Path, *, allow_legacy: bool = False) -> dict:
    """Load the independent zero/safety calibration artifact."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "ird_clearance_calibration_v2":
        if not allow_legacy:
            raise ValueError(f"legacy or incompatible calibration schema: {path}")
    if "safety_threshold" in data:
        data.setdefault("threshold", data["safety_threshold"])
        data.setdefault("m_safe", data["safety_threshold"] + data.get("zero_bias", 0.0))
    elif "threshold" not in data:
        raise KeyError(f"calibration file missing safety threshold: {path}")
    return data


__all__ = [
    "ConformalResult",
    "SafetyThresholdResult",
    "ZeroBiasResult",
    "accepted_reachability",
    "calibrated_clearance",
    "empirical_coverage",
    "false_acceptance_report",
    "fit_conformal",
    "fit_split_conformal",
    "fit_unreachable_safety_threshold",
    "fit_zero_bias",
    "geometric_clearance",
    "load_conformal_json",
    "predict_reachable",
]

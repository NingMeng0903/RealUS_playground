"""Post-hoc calibration for conservative clearance queries."""

from __future__ import annotations

import json
from pathlib import Path

from ird_playground.calib.conformal import (
    ConformalResult,
    calibrated_clearance,
    empirical_coverage,
    fit_conformal,
    fit_split_conformal,
    predict_reachable,
)


def load_conformal_json(path: str | Path) -> dict:
    """Load a conformal calib JSON (threshold / m_safe)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "threshold" not in data:
        raise KeyError(f"conformal file missing threshold: {path}")
    return data


__all__ = [
    "ConformalResult",
    "calibrated_clearance",
    "empirical_coverage",
    "fit_conformal",
    "fit_split_conformal",
    "load_conformal_json",
    "predict_reachable",
]

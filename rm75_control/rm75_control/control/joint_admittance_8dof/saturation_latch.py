"""Continuous nullspace fade from task slack.  Never a boolean latch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.control.joint_admittance_8dof.filters import smoothstep01


@dataclass
class SaturationConfig:
    slack_enter: float = 0.15
    slack_exit: float = 0.03
    secondary_scale: float = 0.15
    secondary_scale_tau_s: float = 0.10


def secondary_scale_from_slack(slack_norm: float, cfg: SaturationConfig) -> float:
    """Continuous nullspace fade from slack; never a boolean latch.

    Mid-ranging ``u_mid`` is computed outside this scale: recovering
    task feasibility must not be suppressed when slack is already large.
    """

    enter = float(cfg.slack_enter)
    exit_ = float(cfg.slack_exit)
    lo = float(np.clip(cfg.secondary_scale, 0.0, 1.0))
    slack = float(slack_norm) if np.isfinite(slack_norm) else 0.0
    span = max(enter - exit_, 1.0e-9)
    s = float(np.clip((slack - exit_) / span, 0.0, 1.0))
    return 1.0 + smoothstep01(s) * (lo - 1.0)


def predict_rail_position_m(
    sample_m: float,
    velocity_m_s: float,
    age_s: float,
    *,
    max_age_s: float | None = None,
    lo_m: float | None = None,
    hi_m: float | None = None,
) -> float:
    """Hold-aware rail FK: last encoder + bounded coast, not a low-pass."""

    if not np.isfinite(sample_m):
        return float("nan")
    age = float(age_s) if np.isfinite(age_s) else 0.0
    if max_age_s is not None and np.isfinite(max_age_s):
        age = min(max(age, 0.0), max(float(max_age_s), 0.0))
    else:
        age = max(age, 0.0)
    vel = float(velocity_m_s) if np.isfinite(velocity_m_s) else 0.0
    pred = float(sample_m) + vel * age
    if lo_m is not None and np.isfinite(lo_m):
        pred = max(pred, float(lo_m))
    if hi_m is not None and np.isfinite(hi_m):
        pred = min(pred, float(hi_m))
    return pred


__all__ = [
    "SaturationConfig",
    "predict_rail_position_m",
    "secondary_scale_from_slack",
    "smoothstep01",
]

"""AND-latch that opens the Cartesian reference governor.

Healthy tracking must never slow the clock.  The latch therefore requires
both a geometric pin (joint, rail, or J4 branch wall) and a sustained QP
task slack.  Replay of 022330 / 022459 / 022056 never fired at 150 ms dwell;
022415 fired at t≈12.15 s while TCP error was still <5 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SaturationConfig:
    slack_enter: float = 0.03
    slack_exit: float = 0.015
    dwell_s: float = 0.15
    rail_margin_m: float = 0.010
    branch_margin_rad: float = 0.035  # ~2° neighbourhood of the 0.35 rad wall
    secondary_scale: float = 0.15
    secondary_scale_tau_s: float = 0.10
    crawl_floor: float = 0.05
    freeze_timeout_s: float = 9.0
    stall_improve_mm: float = 1.0


@dataclass
class SaturationFlags:
    near_arm: bool = False
    near_rail: bool = False
    near_branch: bool = False
    slack_over: bool = False
    pinned: bool = False
    cannot_follow: bool = False


def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def secondary_scale_from_slack(slack_norm: float, cfg: SaturationConfig) -> float:
    """Continuous nullspace fade from slack; never a boolean latch."""

    enter = float(cfg.slack_enter)
    exit_ = float(cfg.slack_exit)
    lo = float(np.clip(cfg.secondary_scale, 0.0, 1.0))
    slack = float(slack_norm) if np.isfinite(slack_norm) else 0.0
    span = max(enter - exit_, 1.0e-9)
    s = float(np.clip((slack - exit_) / span, 0.0, 1.0))
    return 1.0 + smoothstep01(s) * (lo - 1.0)


class SaturationLatch:
    """Dwell + hysteresis on ``pinned AND slack_over``."""

    def __init__(self, cfg: SaturationConfig | None = None) -> None:
        self.cfg = cfg or SaturationConfig()
        self.latched: bool = False
        self._streak_s: float = 0.0
        self.freeze_s: float = 0.0

    def reset(self) -> None:
        self.latched = False
        self._streak_s = 0.0
        self.freeze_s = 0.0

    def update(
        self,
        *,
        q_cmd: np.ndarray,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        rail_soft_min_m: float,
        rail_soft_max_m: float,
        near_arm_margin_rad: float,
        branch_eps_rad: float,
        slack_norm: float,
        dt_s: float,
        j4_index: int = 4,
    ) -> SaturationFlags:
        q = np.asarray(q_cmd, dtype=float).reshape(-1)
        lo = np.asarray(q_lower, dtype=float).reshape(-1)
        hi = np.asarray(q_upper, dtype=float).reshape(-1)
        nv = min(q.size, lo.size, hi.size)
        near_arm = False
        if nv > 1:
            margin = float(near_arm_margin_rad)
            near_arm = bool(
                np.any(q[1:nv] < lo[1:nv] + margin)
                or np.any(q[1:nv] > hi[1:nv] - margin)
            )
        near_rail = False
        if nv > 0 and np.isfinite(q[0]):
            d_lo = float(q[0]) - float(rail_soft_min_m)
            d_hi = float(rail_soft_max_m) - float(q[0])
            near_rail = bool(min(d_lo, d_hi) < float(self.cfg.rail_margin_m))
        near_branch = False
        if q.size > int(j4_index) and np.isfinite(q[int(j4_index)]):
            near_branch = bool(
                abs(abs(float(q[int(j4_index)])) - float(branch_eps_rad))
                <= float(self.cfg.branch_margin_rad)
            )
        slack = float(slack_norm) if np.isfinite(slack_norm) else 0.0
        slack_over = slack > float(self.cfg.slack_enter)
        pinned = bool(near_arm or near_rail or near_branch)
        enter = bool(pinned and slack_over)
        exit_now = (not pinned) or slack < float(self.cfg.slack_exit)
        dt = max(float(dt_s), 0.0) if np.isfinite(dt_s) else 0.0
        if self.latched:
            if exit_now:
                self.latched = False
                self._streak_s = 0.0
        else:
            if enter:
                self._streak_s += dt
                if self._streak_s >= float(self.cfg.dwell_s):
                    self.latched = True
            else:
                self._streak_s = 0.0
        return SaturationFlags(
            near_arm=near_arm,
            near_rail=near_rail,
            near_branch=near_branch,
            slack_over=slack_over,
            pinned=pinned,
            cannot_follow=bool(self.latched),
        )


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
    "SaturationFlags",
    "SaturationLatch",
    "predict_rail_position_m",
    "secondary_scale_from_slack",
    "smoothstep01",
]

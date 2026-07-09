"""Task 1: find y_b that allows a full scan of all waypoints."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.inversion.interval_set import IntervalSet
from rm75_control.tools.reachability.inversion.quality import (
    QualityWeights,
    pick_rail_y_in_interval,
    score_placement,
)
from rm75_control.tools.reachability.inversion.reach_set import rail_feasible_y_shift
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory


@dataclass
class FullScanResult:
    feasible: bool
    y_b_best: float | None
    score: float
    rail_y_series: list[float] = field(default_factory=list)
    feasible_y_b_intervals: list[tuple[float, float]] = field(default_factory=list)
    per_waypoint_intervals: list[list[tuple[float, float]]] = field(default_factory=list)


def _yb_candidates(yb_range: tuple[float, float], yb_step: float) -> np.ndarray:
    lo, hi = float(yb_range[0]), float(yb_range[1])
    return np.arange(lo, hi + 0.5 * yb_step, yb_step, dtype=np.float64)


def full_scan_best_yb(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    *,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    rail_travel_half: float = 0.18,
    yb_range: tuple[float, float] = (-1.0, 1.0),
    yb_step: float = 0.01,
    quality: QualityWeights | None = None,
    y_shift_step: float | None = None,
) -> FullScanResult:
    """Return the best rail-base Y position for scanning the whole trajectory.

    Each waypoint must be individually reachable (possibly with a different
    ``rail_y`` per waypoint); we do **not** require a single shared rail_y across
    the full path — only that every ``A_i`` is non-empty for the chosen ``y_b``.
    """
    qw = quality or QualityWeights()
    best: FullScanResult | None = None
    feasible_intervals: list[tuple[float, float]] = []

    for y_b in _yb_candidates(yb_range, yb_step):
        per_wp_sets: list[IntervalSet] = []
        ok = True
        for wp in traj.waypoints:
            s = rail_feasible_y_shift(
                cm, wp, float(y_b), xz_base_world,
                rail_travel_half=rail_travel_half, y_shift_step=y_shift_step,
            )
            per_wp_sets.append(s)
            if s.empty:
                ok = False
                break
        if not ok:
            continue

        feasible_intervals.append((float(y_b), float(y_b)))  # point intervals for feasible y_b

        rail_y_series: list[float] = []
        per_wp_pairs: list[list[tuple[float, float]]] = []
        for wp, s in zip(traj.waypoints, per_wp_sets):
            per_wp_pairs.append(s.to_pairs())
            # pick best rail_y inside the first (widest) interval for scoring
            iv = s.intervals[0]
            rail_y_series.append(
                pick_rail_y_in_interval(cm, wp, float(y_b), iv.lo, iv.hi, xz_base_world)
            )

        score = score_placement(
            cm, traj.waypoints, float(y_b), np.asarray(rail_y_series),
            xz_base_world, qw, rail_travel_half=rail_travel_half,
        )
        cand = FullScanResult(
            feasible=True,
            y_b_best=float(y_b),
            score=float(score),
            rail_y_series=rail_y_series,
            feasible_y_b_intervals=[(float(y_b), float(y_b))],
            per_waypoint_intervals=per_wp_pairs,
        )
        if best is None or cand.score > best.score:
            best = cand

    if best is None:
        return FullScanResult(feasible=False, y_b_best=None, score=0.0)
    best.feasible_y_b_intervals = feasible_intervals
    return best

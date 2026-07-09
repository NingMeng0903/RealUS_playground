"""Task 2: longest scan prefix from the trajectory start (lazy-ORM sliding AND)."""

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
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory, Waypoint


@dataclass
class PrefixResult:
    feasible: bool
    y_b_best: float | None
    last_wp_index: int
    arc_len_m: float
    score: float
    rail_y: float
    rail_y_series: list[float] = field(default_factory=list)
    relaxed: bool = False
    strict_last_wp_index: int = -1


def _prefix_for_yb(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    y_b: float,
    xz_base_world: tuple[float, float],
    *,
    rail_travel_half: float,
    y_shift_step: float | None,
    waypoints: list[Waypoint],
) -> tuple[int, float, IntervalSet]:
    """Forward sweep: return (last_index, arc_len, final_nonempty_intersection)."""
    inter = IntervalSet.full(-np.inf, np.inf)
    last_ok = -1
    last_nonempty = IntervalSet.none()
    for j, wp in enumerate(waypoints):
        a_j = rail_feasible_y_shift(
            cm, wp, float(y_b), xz_base_world,
            rail_travel_half=rail_travel_half, y_shift_step=y_shift_step,
        )
        inter = inter.intersect(a_j)
        if inter.empty:
            break
        last_ok = j
        last_nonempty = inter
    arc = traj.arc_length_m(last_ok) if last_ok >= 0 else 0.0
    return last_ok, arc, last_nonempty


def longest_prefix(
    cm: CapabilityMap,
    traj: ScanTrajectory,
    *,
    xz_base_world: tuple[float, float] = (0.0, 0.0),
    rail_travel_half: float = 0.18,
    yb_range: tuple[float, float] = (-1.0, 1.0),
    yb_step: float = 0.01,
    quality: QualityWeights | None = None,
    y_shift_step: float | None = None,
    try_relaxed: bool = True,
) -> PrefixResult:
    """Find ``y_b`` maximising the longest prefix scannable with one fixed ``rail_y``.

    Prefix feasibility uses cumulative intersection ``⋂_{j≤i} A_j ≠ ∅`` — i.e. a
    single ``y_shift`` (hence single ``rail_y``) must work for every waypoint in
    the prefix simultaneously.
    """
    qw = quality or QualityWeights()
    lo, hi = float(yb_range[0]), float(yb_range[1])
    yb_grid = np.arange(lo, hi + 0.5 * yb_step, yb_step, dtype=np.float64)

    best: PrefixResult | None = None

    def _eval(y_b: float, wps: list[Waypoint], relaxed: bool) -> PrefixResult | None:
        last_idx, arc_len, inter = _prefix_for_yb(
            cm, traj, float(y_b), xz_base_world,
            rail_travel_half=rail_travel_half, y_shift_step=y_shift_step,
            waypoints=wps,
        )
        if last_idx < 0 or inter.empty:
            return None
        iv = inter.intervals[0]
        # single rail_y for entire prefix — pick best y_shift in intersection
        rail_y = pick_rail_y_in_interval(
            cm, wps[last_idx], float(y_b), iv.lo, iv.hi, xz_base_world,
        )
        rail_series = [rail_y] * (last_idx + 1)
        score = score_placement(
            cm, wps[: last_idx + 1], float(y_b), np.asarray(rail_series),
            xz_base_world, qw, rail_travel_half=rail_travel_half,
        )
        return PrefixResult(
            feasible=True,
            y_b_best=float(y_b),
            last_wp_index=int(last_idx),
            arc_len_m=float(arc_len),
            score=float(score),
            rail_y=float(rail_y),
            rail_y_series=rail_series,
            relaxed=relaxed,
            strict_last_wp_index=int(last_idx),
        )

    for y_b in yb_grid:
        strict = _eval(float(y_b), traj.waypoints, relaxed=False)
        if strict is not None:
            if best is None or (
                strict.last_wp_index > best.last_wp_index
                or (strict.last_wp_index == best.last_wp_index and strict.arc_len_m > best.arc_len_m)
                or (
                    strict.last_wp_index == best.last_wp_index
                    and abs(strict.arc_len_m - best.arc_len_m) < 1e-9
                    and strict.score > best.score
                )
            ):
                best = strict
            continue

        if not try_relaxed:
            continue
        relaxed_wps = [wp.with_relaxed_tolerances(2.0) for wp in traj.waypoints]
        relaxed = _eval(float(y_b), relaxed_wps, relaxed=True)
        if relaxed is None:
            continue
        if best is None or (
            relaxed.last_wp_index > best.last_wp_index
            or (relaxed.last_wp_index == best.last_wp_index and relaxed.arc_len_m > best.arc_len_m)
            or (
                relaxed.last_wp_index == best.last_wp_index
                and abs(relaxed.arc_len_m - best.arc_len_m) < 1e-9
                and relaxed.score > best.score
            )
        ):
            best = relaxed

    if best is None:
        return PrefixResult(
            feasible=False, y_b_best=None, last_wp_index=-1, arc_len_m=0.0,
            score=0.0, rail_y=0.0,
        )
    return best

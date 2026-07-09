"""Scoring weights for base-placement optimisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.data_model.frames import apply_yshift_world_to_arm_base
from rm75_control.tools.reachability.inversion.reach_set import _neighbor_voxels, _orient_indices_for_wp
from rm75_control.tools.reachability.inversion.trajectory import Waypoint


@dataclass
class QualityWeights:
    manipulability: float = 1.0
    center_rail: float = 0.2
    smooth_rail: float = 0.5
    d_neighbor: float = 0.1


def _mu_at(cm: CapabilityMap, wp: Waypoint, xz_base_world: tuple[float, float], y_shift: float) -> float:
    if cm.mu_mean is None:
        return 0.5
    p_ab = apply_yshift_world_to_arm_base(wp.p_world, xz_base_world, y_shift)
    ori_neighbors = _orient_indices_for_wp(cm, wp)
    best = 0.0
    for ijk in _neighbor_voxels(cm.grid, p_ab, wp.pos_tol_m):
        row = cm.row_of(ijk)
        if row is None:
            continue
        if cm.any_orient_reachable(ijk, ori_neighbors):
            best = max(best, float(cm.mu_mean[row]))
    return best


def _d_at(cm: CapabilityMap, wp: Waypoint, xz_base_world: tuple[float, float], y_shift: float) -> float:
    p_ab = apply_yshift_world_to_arm_base(wp.p_world, xz_base_world, y_shift)
    best = 0.0
    for ijk in _neighbor_voxels(cm.grid, p_ab, wp.pos_tol_m):
        row = cm.row_of(ijk)
        if row is None:
            continue
        best = max(best, float(cm.d_value[row]))
    return best


def score_placement(
    cm: CapabilityMap,
    waypoints: list[Waypoint],
    y_b: float,
    rail_y_series: np.ndarray,
    xz_base_world: tuple[float, float],
    weights: QualityWeights,
    *,
    rail_travel_half: float = 0.18,
) -> float:
    """Higher is better."""
    if len(waypoints) == 0:
        return 0.0
    rail_y_series = np.asarray(rail_y_series, dtype=np.float64)
    score = 0.0
    for i, wp in enumerate(waypoints):
        y_shift = float(y_b + rail_y_series[i])
        w = float(wp.weight)
        score += weights.manipulability * w * _mu_at(cm, wp, xz_base_world, y_shift)
        score += weights.d_neighbor * w * _d_at(cm, wp, xz_base_world, y_shift)
    # rail centre preference
    score += weights.center_rail * float(-np.mean(rail_y_series ** 2) / (rail_travel_half ** 2 + 1e-12))
    # smoothness
    if rail_y_series.size >= 2:
        dif = np.diff(rail_y_series)
        score += weights.smooth_rail * float(-np.mean(dif ** 2) / (rail_travel_half ** 2 + 1e-12))
    return float(score)


def pick_rail_y_in_interval(
    cm: CapabilityMap,
    wp: Waypoint,
    y_b: float,
    interval_lo: float,
    interval_hi: float,
    xz_base_world: tuple[float, float],
    *,
    n_samples: int = 9,
) -> float:
    """Pick rail_y = y_shift - y_b inside [interval_lo, interval_hi] maximising μ then D."""
    if interval_hi < interval_lo:
        return 0.0
    ys = np.linspace(interval_lo, interval_hi, max(2, n_samples))
    best_y = float(ys[0] - y_b)
    best_val = -1e18
    for y_shift in ys:
        mu = _mu_at(cm, wp, xz_base_world, float(y_shift))
        d = _d_at(cm, wp, xz_base_world, float(y_shift))
        val = mu + 0.1 * d
        if val > best_val:
            best_val = val
            best_y = float(y_shift - y_b)
    return float(np.clip(best_y, -0.18, 0.18))

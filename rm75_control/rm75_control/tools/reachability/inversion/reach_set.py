"""Reachability inversion: waypoint → allowed y_shift intervals."""

from __future__ import annotations

import numpy as np

from rm75_control.tools.reachability.data_model.capability_map import CapabilityMap
from rm75_control.tools.reachability.data_model.frames import apply_yshift_world_to_arm_base
from rm75_control.tools.reachability.inversion.interval_set import IntervalSet, run_length_true_mask
from rm75_control.tools.reachability.inversion.trajectory import Waypoint


def _neighbor_voxels(grid, p_ab: np.ndarray, pos_tol_m: float) -> list[tuple[int, int, int]]:
    """Return unique in-bounds voxel indices whose centres lie within ``pos_tol_m`` of ``p_ab``."""
    step = float(grid.step_m)
    r_cells = max(0, int(np.ceil(pos_tol_m / step)))
    ijk0 = grid.idx_of(p_ab)
    if not grid.in_bounds(ijk0):
        return []
    i0, j0, k0 = int(ijk0[0]), int(ijk0[1]), int(ijk0[2])
    out: set[tuple[int, int, int]] = set()
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            for dk in range(-r_cells, r_cells + 1):
                ijk = (i0 + di, j0 + dj, k0 + dk)
                if not grid.in_bounds(np.array(ijk, dtype=np.int32)):
                    continue
                c = grid.center_of(np.array(ijk, dtype=np.int32))
                if float(np.linalg.norm(c - p_ab)) <= pos_tol_m + 1e-9:
                    out.add(ijk)
    return list(out)


def _orient_indices_for_wp(cm: CapabilityMap, wp: Waypoint) -> np.ndarray:
    """Orientation indices to test (neighbors within tolerance, else nearest)."""
    nb = cm.orientations.neighbors_of_dir(wp.tool_axis_world, wp.axis_tol_deg)
    if nb.size > 0:
        return nb
    return np.array([cm.orientations.nearest(wp.tool_axis_world)], dtype=np.int64)


def waypoint_reachable_at_yshift(
    cm: CapabilityMap,
    wp: Waypoint,
    xz_base_world: tuple[float, float],
    y_shift: float,
) -> bool:
    """True if ``wp`` is reachable when the arm base is at world Y = ``y_shift``."""
    p_ab = apply_yshift_world_to_arm_base(wp.p_world, xz_base_world, float(y_shift))
    ori_neighbors = _orient_indices_for_wp(cm, wp)
    for ijk in _neighbor_voxels(cm.grid, p_ab, wp.pos_tol_m):
        if cm.any_orient_reachable(ijk, ori_neighbors):
            return True
    return False


def allowed_y_shift(
    cm: CapabilityMap,
    wp: Waypoint,
    xz_base_world: tuple[float, float],
    *,
    y_shift_range: tuple[float, float],
    y_shift_step: float | None = None,
) -> IntervalSet:
    """Intervals of ``y_shift`` (= world Y of arm base) where ``wp`` is reachable."""
    lo, hi = float(y_shift_range[0]), float(y_shift_range[1])
    step = float(y_shift_step if y_shift_step is not None else cm.grid.step_m * 0.5)
    if step <= 0:
        raise ValueError("y_shift_step must be > 0")
    xs = np.arange(lo, hi + 0.5 * step, step, dtype=np.float64)
    mask = np.array(
        [waypoint_reachable_at_yshift(cm, wp, xz_base_world, float(x)) for x in xs],
        dtype=bool,
    )
    return run_length_true_mask(mask, xs)


def rail_feasible_y_shift(
    cm: CapabilityMap,
    wp: Waypoint,
    y_b: float,
    xz_base_world: tuple[float, float],
    *,
    rail_travel_half: float = 0.18,
    y_shift_step: float | None = None,
) -> IntervalSet:
    """``allowed_y_shift`` clipped to ``[y_b - rail_travel_half, y_b + rail_travel_half]``."""
    full = allowed_y_shift(
        cm, wp, xz_base_world,
        y_shift_range=(y_b - rail_travel_half, y_b + rail_travel_half),
        y_shift_step=y_shift_step,
    )
    clip = IntervalSet.from_pairs([(y_b - rail_travel_half, y_b + rail_travel_half)])
    return full.intersect(clip)

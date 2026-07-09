"""IntervalSet algebra + reach_set intersection tests."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.tools.reachability.data_model import (
    BitmaskLayout,
    CapabilityMap,
    IcosphereToolAxisGrid,
    MapMeta,
    VoxelGrid,
)
from rm75_control.tools.reachability.data_model.capability_map import pack_bits_5dof
from rm75_control.tools.reachability.inversion.interval_set import Interval, IntervalSet, run_length_true_mask
from rm75_control.tools.reachability.inversion.reach_set import allowed_y_shift, rail_feasible_y_shift
from rm75_control.tools.reachability.inversion.trajectory import Waypoint


def test_interval_intersect_and_union():
    a = IntervalSet.from_pairs([(0.0, 0.5), (0.8, 1.0)])
    b = IntervalSet.from_pairs([(0.3, 0.9)])
    inter = a.intersect(b)
    assert inter.to_pairs() == [(0.3, 0.5), (0.8, 0.9)]
    union = a.union(b)
    assert union.to_pairs() == [(0.0, 1.0)]


def test_run_length_mask():
    xs = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    mask = np.array([True, True, False, True, True])
    s = run_length_true_mask(mask, xs)
    assert s.to_pairs() == [(0.0, 0.1), (0.3, 0.4)]


def _synthetic_map_for_yshift() -> CapabilityMap:
    """Map where reachability depends only on arm-base Y (via voxel j index)."""
    grid = VoxelGrid(origin_m=np.array([-0.5, -0.5, 0.0]), step_m=0.10, shape=(10, 10, 5))
    orient = IcosphereToolAxisGrid.build(subdiv=0)
    # voxel at j=5 centre y=0.0; mark orient 0 reachable
    ijk = np.array([[5, 5, 2]], dtype=np.int32)
    bool_mat = np.ones((1, orient.n), dtype=bool)
    packed = pack_bits_5dof(bool_mat)
    layout = BitmaskLayout(n_orient=orient.n, n_roll=0)
    return CapabilityMap(
        grid=grid, orientations=orient, roll=None, layout=layout,
        voxel_ids=ijk, bitmask=packed,
        d_value=np.ones(1, dtype=np.float32), meta=MapMeta(),
    )


def test_allowed_y_shift_finds_band():
    cm = _synthetic_map_for_yshift()
    centre = cm.grid.center_of(np.array([5, 5, 2], dtype=np.int32))
    wp = Waypoint(
        p_world=centre,
        tool_axis_world=cm.orientations.vectors[0],
        axis_tol_deg=30.0,
        pos_tol_m=0.05,
    )
    s = allowed_y_shift(cm, wp, xz_base_world=(0.0, 0.0), y_shift_range=(-0.5, 0.5), y_shift_step=0.05)
    assert not s.empty
    assert s.contains_value(0.0)


def test_rail_feasible_clips_to_travel():
    cm = _synthetic_map_for_yshift()
    centre = cm.grid.center_of(np.array([5, 5, 2], dtype=np.int32))
    wp = Waypoint(
        p_world=centre,
        tool_axis_world=cm.orientations.vectors[0],
        axis_tol_deg=30.0,
        pos_tol_m=0.05,
    )
    s = rail_feasible_y_shift(cm, wp, y_b=0.0, xz_base_world=(0.0, 0.0), rail_travel_half=0.18, y_shift_step=0.05)
    for lo, hi in s.to_pairs():
        assert lo >= -0.18 - 1e-9
        assert hi <= 0.18 + 1e-9


def test_cumulative_intersection_shrinks():
    """Simulate prefix AND: later waypoint with narrower band shrinks intersection."""
    a = IntervalSet.from_pairs([(-0.1, 0.1)])
    b = IntervalSet.from_pairs([(-0.05, 0.05)])
    inter = a.intersect(b)
    assert inter.to_pairs() == [(-0.05, 0.05)]
    c = IntervalSet.from_pairs([(0.2, 0.3)])
    assert a.intersect(c).empty

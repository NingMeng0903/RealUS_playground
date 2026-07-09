"""Prefix solver tests with a handcrafted tiny map."""

from __future__ import annotations

import numpy as np

from rm75_control.tools.reachability.data_model import (
    BitmaskLayout,
    CapabilityMap,
    IcosphereToolAxisGrid,
    MapMeta,
    VoxelGrid,
)
from rm75_control.tools.reachability.data_model.capability_map import pack_bits_5dof
from rm75_control.tools.reachability.inversion.base_optimizer import full_scan_best_yb
from rm75_control.tools.reachability.inversion.prefix_solver import longest_prefix
from rm75_control.tools.reachability.inversion.trajectory import ScanTrajectory, Waypoint


def _map_with_voxels(ijk_list, orient_idx=0):
    grid = VoxelGrid(origin_m=np.array([-0.5, -0.5, 0.0]), step_m=0.10, shape=(12, 12, 5))
    orient = IcosphereToolAxisGrid.build(subdiv=0)
    ijk = np.asarray(ijk_list, dtype=np.int32)
    n = ijk.shape[0]
    bool_mat = np.ones((n, orient.n), dtype=bool)  # all tool axes at this voxel
    packed = pack_bits_5dof(bool_mat)
    layout = BitmaskLayout(n_orient=orient.n, n_roll=0)
    return CapabilityMap(
        grid=grid, orientations=orient, roll=None, layout=layout,
        voxel_ids=ijk, bitmask=packed,
        d_value=np.ones(n, dtype=np.float32), meta=MapMeta(),
    )


def _wp_at(p, axis=None):
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    return Waypoint(p_world=np.asarray(p, float), tool_axis_world=axis, axis_tol_deg=25.0, pos_tol_m=0.08)


def _centre(cm, ijk):
    return cm.grid.center_of(np.asarray(ijk, dtype=np.int32))


def test_prefix_stops_at_first_infeasible():
    cm = _map_with_voxels([[5, 5, 2]])
    traj = ScanTrajectory(waypoints=[
        _wp_at(_centre(cm, [5, 5, 2])),
        _wp_at([0.50, 0.50, 0.25]),  # no voxel in map here
    ])
    res = longest_prefix(
        cm, traj, xz_base_world=(0.0, 0.0), yb_range=(-0.1, 0.1), yb_step=0.05,
        try_relaxed=False,
    )
    assert res.feasible
    assert res.last_wp_index == 0


def test_full_scan_all_waypoints_individually():
    # nearby voxels j=5,6 — each reachable with its own rail_y when y_b=0.1
    cm = _map_with_voxels([[5, 5, 2], [5, 6, 2]])
    traj = ScanTrajectory(waypoints=[
        _wp_at(_centre(cm, [5, 5, 2])),
        _wp_at(_centre(cm, [5, 6, 2])),
    ])
    full = full_scan_best_yb(
        cm, traj, xz_base_world=(0.0, 0.0), yb_range=(0.05, 0.15), yb_step=0.05,
    )
    assert full.feasible
    assert len(full.rail_y_series) == 2


def test_prefix_longer_when_shared_yshift_exists():
    cm = _map_with_voxels([[5, 5, 2]])
    c0 = _centre(cm, [5, 5, 2])
    traj = ScanTrajectory(waypoints=[
        _wp_at(c0),
        _wp_at(c0 + np.array([0.02, 0.0, 0.0])),
        _wp_at(c0 + np.array([0.04, 0.0, 0.0])),
    ])
    res = longest_prefix(
        cm, traj, xz_base_world=(0.0, 0.0), yb_range=(-0.05, 0.05), yb_step=0.05,
        try_relaxed=False,
    )
    assert res.feasible
    assert res.last_wp_index >= 2

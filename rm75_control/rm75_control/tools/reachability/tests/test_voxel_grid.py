"""Tests for VoxelGrid (idx/center round-trip, bounds, flat/unflat, centers ball)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.tools.reachability.data_model import VoxelGrid, VoxelGridConfig


@pytest.fixture
def small_grid() -> VoxelGrid:
    return VoxelGrid(origin_m=np.array([-0.15, -0.15, -0.15]), step_m=0.05, shape=(6, 6, 6))


def test_from_config_matches_manual():
    cfg = VoxelGridConfig(origin_m=(-1.0, -1.0, -0.3), step_m=0.03, shape=(20, 20, 15))
    g = VoxelGrid.from_config(cfg)
    assert g.step_m == 0.03
    assert g.shape == (20, 20, 15)
    assert g.n_voxels == 20 * 20 * 15


def test_idx_center_roundtrip(small_grid):
    ijk = np.array([(0, 0, 0), (5, 5, 5), (2, 3, 1)], dtype=np.int32)
    centers = small_grid.center_of(ijk)
    ijk_back = small_grid.idx_of(centers)
    np.testing.assert_array_equal(ijk, ijk_back)


def test_center_is_at_half_step_offset(small_grid):
    c = small_grid.center_of(np.array([0, 0, 0], dtype=np.int32))
    expected = small_grid.origin_m + 0.5 * small_grid.step_m
    np.testing.assert_allclose(c, expected)


def test_in_bounds(small_grid):
    inside = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.int32)
    outside = np.array([[-1, 0, 0], [6, 5, 5], [0, 0, -1]], dtype=np.int32)
    np.testing.assert_array_equal(small_grid.in_bounds(inside), [True, True])
    np.testing.assert_array_equal(small_grid.in_bounds(outside), [False, False, False])


def test_flat_unflat(small_grid):
    ijk = np.array([[0, 0, 0], [5, 5, 5], [2, 3, 1]], dtype=np.int32)
    flat = small_grid.flat(ijk)
    back = small_grid.unflat(flat)
    np.testing.assert_array_equal(ijk, back)
    # scalar path
    assert small_grid.flat(np.array([1, 2, 3], dtype=np.int32)) == 1 * 36 + 2 * 6 + 3


def test_all_centers_matches_center_of(small_grid):
    centers = small_grid.all_centers()
    assert centers.shape == (216, 3)
    # first voxel
    np.testing.assert_allclose(centers[0], small_grid.center_of(np.array([0, 0, 0])))
    # last voxel
    np.testing.assert_allclose(centers[-1], small_grid.center_of(np.array([5, 5, 5])))


def test_centers_inside_ball(small_grid):
    ijk = small_grid.centers_inside_ball(radius_m=0.15)
    assert ijk.shape[1] == 3
    centers = small_grid.center_of(ijk)
    assert np.all(np.linalg.norm(centers, axis=1) <= 0.15 + 1e-9)


def test_bad_shapes():
    with pytest.raises(ValueError):
        VoxelGrid(origin_m=np.zeros(2), step_m=0.05, shape=(2, 2, 2))
    with pytest.raises(ValueError):
        VoxelGrid(origin_m=np.zeros(3), step_m=0.0, shape=(2, 2, 2))
    with pytest.raises(ValueError):
        VoxelGrid(origin_m=np.zeros(3), step_m=0.05, shape=(2, 0, 2))

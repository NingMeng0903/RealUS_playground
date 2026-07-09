"""Small-scale Monte-Carlo build tests (single-process, deterministic seed)."""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.tools.reachability.build.mc_sampler import (
    SparseBitmap,
    sample_and_bin,
)
from rm75_control.tools.reachability.data_model import (
    IcosphereToolAxisGrid,
    VoxelGrid,
    VoxelGridConfig,
)
from rm75_control.tools.reachability.kinematics import build_locked_rail_model
from rm75_control.tools.reachability.kinematics.model_locked_rail import DEFAULT_URDF


@pytest.fixture(scope="module")
def lm():
    if not DEFAULT_URDF.exists():
        pytest.skip(f"URDF missing at {DEFAULT_URDF}")
    return build_locked_rail_model(DEFAULT_URDF)


def _small_grid() -> VoxelGrid:
    return VoxelGrid.from_config(
        VoxelGridConfig(origin_m=(-0.9, -0.9, -0.2), step_m=0.06, shape=(30, 30, 25))
    )


def test_sparse_bitmap_add_and_merge():
    sb = SparseBitmap.empty(n_orient=20)
    flat = np.array([1, 1, 2, 3, 3], dtype=np.int64)
    ori = np.array([0, 5, 9, 2, 2], dtype=np.int64)
    sb.add_hits(flat, ori)
    # voxel 1 should have bits 0 and 5 set (byte 0 = 0b00100001 = 33)
    assert sb.data[1][0] == 0b00100001
    # voxel 2 has bit 9 → byte 1, bit 1 = 2
    assert sb.data[2][1] == 0b00000010
    # voxel 3 has bit 2 (added twice, idempotent)
    assert sb.data[3][0] == 0b00000100

    other = SparseBitmap.empty(n_orient=20)
    other.add_hits(np.array([1, 4], dtype=np.int64), np.array([1, 7], dtype=np.int64))
    sb.merge_from(other)
    assert sb.data[1][0] == 0b00100011
    assert sb.data[4][0] == 0b10000000


def test_sample_and_bin_produces_nonzero_and_deterministic(lm):
    grid = _small_grid()
    orient = IcosphereToolAxisGrid.build(subdiv=1)  # 42 dirs (fast)
    rng1 = np.random.default_rng(123)
    sb1 = sample_and_bin(lm, grid, orient, n_samples=50_000, batch=25_000, rng=rng1, reachable_radius_m=1.2)
    rng2 = np.random.default_rng(123)
    sb2 = sample_and_bin(lm, grid, orient, n_samples=50_000, batch=25_000, rng=rng2, reachable_radius_m=1.2)

    assert len(sb1.data) > 0
    # Determinism: same seed → identical dict keys and identical uint8 rows.
    assert set(sb1.data.keys()) == set(sb2.data.keys())
    for k in sb1.data:
        np.testing.assert_array_equal(sb1.data[k], sb2.data[k])


def test_sample_and_bin_touches_reasonable_region(lm):
    """FK'd samples should occupy a shell roughly inside the arm reach."""
    grid = _small_grid()
    orient = IcosphereToolAxisGrid.build(subdiv=1)
    sb = sample_and_bin(
        lm, grid, orient, n_samples=30_000, batch=10_000,
        rng=np.random.default_rng(0), reachable_radius_m=1.2,
    )
    ijk, _bm = sb.to_dense(grid)
    assert ijk.shape[0] > 100
    centers = grid.center_of(ijk)
    r = np.linalg.norm(centers, axis=1)
    # No sample should fall outside the reachable ball
    assert r.max() <= 1.2 + 1e-6
    # Some samples should be in the middle region (not all near the base)
    assert (r > 0.3).sum() > 20


def test_to_dense_layout(lm):
    grid = _small_grid()
    orient = IcosphereToolAxisGrid.build(subdiv=1)
    sb = sample_and_bin(lm, grid, orient, n_samples=10_000, batch=5_000, rng=np.random.default_rng(1))
    ijk, bm = sb.to_dense(grid)
    assert ijk.dtype == np.int32
    assert bm.dtype == np.uint8
    assert bm.shape == (ijk.shape[0], (orient.n + 7) // 8)
    # sorted by flat index
    flat = grid.flat(ijk)
    assert np.all(np.diff(flat) > 0)

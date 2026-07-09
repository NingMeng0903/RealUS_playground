"""Unit tests for the Phase-2 IK refiner.

* ``_rotation_from_axis_and_roll`` orients the tool's +Z axis correctly.
* ``_unreached_orient_indices`` returns exactly the zero bits.
* ``refine`` on a synthetic MC output only flips reachable orientations, never
  creates rows that were absent, and preserves already-set bits.
"""

from __future__ import annotations

import numpy as np
import pytest

from rm75_control.tools.reachability.build.config import IkRefineConfig
from rm75_control.tools.reachability.build.ik_refiner import (
    _rotation_from_axis_and_roll,
    _unreached_orient_indices,
    refine,
)
from rm75_control.tools.reachability.build.mc_sampler import SparseBitmap, sample_and_bin
from rm75_control.tools.reachability.data_model import (
    IcosphereToolAxisGrid,
    VoxelGrid,
    VoxelGridConfig,
)
from rm75_control.tools.reachability.data_model.capability_map import d_value_from_bitmask
from rm75_control.tools.reachability.kinematics import (
    SeedPoolConfig,
    build_locked_rail_model,
)
from rm75_control.tools.reachability.kinematics.model_locked_rail import DEFAULT_URDF


@pytest.fixture(scope="module")
def lm():
    if not DEFAULT_URDF.exists():
        pytest.skip("URDF missing")
    return build_locked_rail_model(DEFAULT_URDF)


def test_rotation_z_aligns_with_axis():
    for axis in np.eye(3):
        R = _rotation_from_axis_and_roll(axis, 0.0)
        z = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(z, axis, atol=1e-9)
    # oblique
    ax = np.array([0.5, 0.5, np.sqrt(0.5)])
    ax /= np.linalg.norm(ax)
    R = _rotation_from_axis_and_roll(ax, 0.7)
    np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), ax, atol=1e-9)
    # orthonormal
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_unreached_indices_matches_ground_truth():
    n_orient = 20
    row = np.zeros(3, dtype=np.uint8)
    for idx in (0, 5, 7, 15):
        b, k = divmod(idx, 8)
        row[b] |= np.uint8(1 << k)
    unreached = _unreached_orient_indices(row, n_orient)
    reached = np.setdiff1d(np.arange(n_orient), unreached)
    np.testing.assert_array_equal(sorted(reached.tolist()), [0, 5, 7, 15])


def _mini_sparse(lm) -> tuple[VoxelGrid, IcosphereToolAxisGrid, "SparseBitmap"]:
    """A very small MC pass so refinement finishes in ~1 second per test."""
    grid = VoxelGrid.from_config(
        VoxelGridConfig(origin_m=(-0.4, -0.4, 0.2), step_m=0.20, shape=(4, 4, 4))
    )
    orient = IcosphereToolAxisGrid.build(subdiv=0)  # 12 dirs — bare icosahedron
    sparse = sample_and_bin(
        lm, grid, orient,
        n_samples=20_000, batch=10_000,
        rng=np.random.default_rng(0), reachable_radius_m=0.85,
    )
    return grid, orient, sparse


def _fast_cfg(**over) -> IkRefineConfig:
    base = dict(
        enabled=True,
        boundary_d_min=0.01,
        boundary_d_max=0.99,
        per_voxel_budget_s=0.02,      # short budget → test bounded wall time
        max_iter=8,
        lam=0.10,
        keep_manipulability=False,
        seeds=SeedPoolConfig(n_random=0, include_nominal=True, include_elbow_flip=False, include_zeros=False),
    )
    base.update(over)
    return IkRefineConfig(**base)


def test_refine_only_flips_reachable_bits(lm):
    grid, orient, sparse = _mini_sparse(lm)
    n0 = sum(int(np.unpackbits(row).sum()) for row in sparse.data.values())
    sparse_before_keys = set(sparse.data.keys())

    sparse2, mu, stats = refine(lm, grid, orient, sparse, _fast_cfg(), progress=False, log_every_n=10_000)
    # never invent new voxels
    assert set(sparse2.data.keys()) == sparse_before_keys
    n1 = sum(int(np.unpackbits(row).sum()) for row in sparse2.data.values())
    # refinement should only ever ADD bits
    assert n1 >= n0
    assert stats.bits_flipped == (n1 - n0)
    assert mu is None


def test_refine_records_manipulability(lm):
    grid, orient, sparse = _mini_sparse(lm)
    _sparse, mu, _stats = refine(
        lm, grid, orient, sparse, _fast_cfg(keep_manipulability=True),
        progress=False, log_every_n=10_000,
    )
    assert mu is not None
    assert mu.ndim == 1
    assert mu.dtype == np.float32
    assert (mu >= 0.0).all()


def test_d_value_monotonic_across_refine(lm):
    """D(x) after refine ≥ D(x) after MC for every retained voxel row."""
    grid, orient, sparse = _mini_sparse(lm)
    ijk_before, bm_before = sparse.to_dense(grid)
    d_before = d_value_from_bitmask(bm_before, orient.n)

    sparse2, _mu, _stats = refine(lm, grid, orient, sparse, _fast_cfg(), progress=False, log_every_n=10_000)
    ijk_after, bm_after = sparse2.to_dense(grid)
    d_after = d_value_from_bitmask(bm_after, orient.n)

    np.testing.assert_array_equal(ijk_before, ijk_after)
    assert np.all(d_after + 1e-12 >= d_before)
    assert float(d_after.mean()) >= float(d_before.mean())

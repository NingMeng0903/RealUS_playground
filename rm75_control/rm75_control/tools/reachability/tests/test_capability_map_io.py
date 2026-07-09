"""Round-trip save/load, bitmask pack/unpack, D(x) counting."""

from __future__ import annotations

import numpy as np

from rm75_control.tools.reachability.data_model import (
    BitmaskLayout,
    CapabilityMap,
    IcosphereToolAxisGrid,
    MapMeta,
    VoxelGrid,
)
from rm75_control.tools.reachability.data_model.capability_map import (
    d_value_from_bitmask,
    pack_bits_5dof,
    unpack_bits_5dof,
)


def _tiny_map(rng: np.random.Generator, n_voxels: int = 25) -> CapabilityMap:
    grid = VoxelGrid(origin_m=np.array([-0.15, -0.15, -0.15]), step_m=0.05, shape=(6, 6, 6))
    orient = IcosphereToolAxisGrid.build(subdiv=1)  # 42 dirs → 6 bytes/voxel
    ijk_all = np.stack(
        np.meshgrid(np.arange(6), np.arange(6), np.arange(6), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    rng.shuffle(ijk_all)
    ijk = ijk_all[:n_voxels].astype(np.int32)
    bool_mat = rng.integers(0, 2, size=(n_voxels, orient.n)).astype(bool)
    packed = pack_bits_5dof(bool_mat)
    d = d_value_from_bitmask(packed, orient.n)
    layout = BitmaskLayout(n_orient=orient.n, n_roll=0)
    return CapabilityMap(
        grid=grid,
        orientations=orient,
        roll=None,
        layout=layout,
        voxel_ids=ijk,
        bitmask=packed,
        d_value=d,
        meta=MapMeta(mc_samples=42, urdf_path="dummy.urdf"),
    )


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.integers(0, 2, size=(30, 197)).astype(bool)
    packed = pack_bits_5dof(m)
    assert packed.shape == (30, (197 + 7) // 8)
    back = unpack_bits_5dof(packed, 197)
    np.testing.assert_array_equal(m, back)


def test_d_value_counts_match_direct_sum():
    rng = np.random.default_rng(1)
    m = rng.integers(0, 2, size=(50, 300)).astype(bool)
    packed = pack_bits_5dof(m)
    d = d_value_from_bitmask(packed, 300)
    d_ref = m.sum(axis=1).astype(np.float32) / 300.0
    np.testing.assert_allclose(d, d_ref)


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    cm = _tiny_map(rng)
    out_dir = cm.save(tmp_path / "map")
    assert (out_dir / "manifest.yaml").exists()
    assert (out_dir / "voxels.npz").exists()
    assert (out_dir / "bitmask.npy").exists()

    cm2 = CapabilityMap.load(out_dir, mmap=False)
    assert cm2.grid.shape == cm.grid.shape
    assert cm2.orientations.n == cm.orientations.n
    assert isinstance(cm2.orientations, IcosphereToolAxisGrid)
    np.testing.assert_array_equal(cm2.voxel_ids, cm.voxel_ids)
    np.testing.assert_array_equal(cm2.bitmask, cm.bitmask)
    np.testing.assert_allclose(cm2.d_value, cm.d_value)
    assert cm2.meta.mc_samples == 42


def test_is_reachable_and_row_lookup(tmp_path):
    rng = np.random.default_rng(3)
    cm = _tiny_map(rng, n_voxels=10)
    for row, ijk in enumerate(cm.voxel_ids):
        assert cm.row_of(tuple(ijk)) == row
    assert cm.row_of((99, 99, 99)) is None
    # spot-check bit truth
    b = unpack_bits_5dof(cm.bitmask, cm.orientations.n)
    for row in (0, cm.n_reachable_voxels - 1):
        ijk = tuple(int(x) for x in cm.voxel_ids[row])
        for oi in (0, 5, cm.orientations.n - 1):
            assert cm.is_reachable(ijk, oi) == bool(b[row, oi])


def test_d_grid_has_nan_outside():
    rng = np.random.default_rng(4)
    cm = _tiny_map(rng, n_voxels=5)
    dg = cm.d_grid()
    assert dg.shape == cm.grid.shape
    filled = np.sum(np.isfinite(dg))
    assert filled == cm.n_reachable_voxels

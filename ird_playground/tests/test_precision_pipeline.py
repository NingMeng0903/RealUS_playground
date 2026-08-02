from __future__ import annotations

import numpy as np

from ird_playground.ird.canonical import canonical_flange_from_se3_features
from ird_playground.ird.canonical_gt import interpolate_boundary_features
from ird_playground.ird.splits import (
    FiveWaySplitConfig,
    assert_split_disjoint,
    five_way_split_indices,
)
from ird_playground.neural.train_signed import paired_epoch_batches


def test_five_way_split_is_complete_disjoint_and_source_owned():
    source = np.repeat(np.arange(100), 3)
    boundary = np.repeat(np.arange(100), 3)
    splits = five_way_split_indices(
        boundary,
        source,
        seed=7,
        config=FiveWaySplitConfig(0.15, 0.10, 0.10, 0.10),
    )
    assert_split_disjoint(splits, source)
    assert sum(len(rows) for rows in splits.values()) == len(source)
    assert all(len(rows) > 0 for rows in splits.values())


def test_paired_sampler_keeps_complete_variable_length_groups():
    sizes = [8, 9, 8, 9]
    boundary = np.concatenate(
        [np.full(size, group, dtype=np.int64) for group, size in enumerate(sizes)]
        + [np.full(32, -1, dtype=np.int64)]
    )
    reachable = np.concatenate(
        [np.r_[np.ones(size // 2), np.zeros(size - size // 2)] for size in sizes]
        + [np.tile([0.0, 1.0], 16)]
    )
    origin = np.r_[np.zeros(sum(sizes), dtype=np.int8), np.full(32, 2, dtype=np.int8)]
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    a = paired_epoch_batches(
        np.arange(len(boundary)), boundary, reachable, rng_a,
        batch_size=32, groups_per_batch=2, sample_origin=origin,
    )
    b = paired_epoch_batches(
        np.arange(len(boundary)), boundary, reachable, rng_b,
        batch_size=32, groups_per_batch=2, sample_origin=origin,
    )
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    seen = set()
    for batch in a:
        for group in np.unique(boundary[batch][boundary[batch] >= 0]):
            rows = set(np.flatnonzero(boundary == group).tolist())
            assert rows.issubset(set(batch.tolist()))
            seen.add(int(group))
    assert seen == set(range(4))


def test_zero_pose_is_interpolated_on_se3_before_canonicalization():
    neg = np.array([[0.2, 0.0, 0.3, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
    pos = np.array([[0.0, 0.2, 0.3, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
    zero = interpolate_boundary_features(neg, pos, np.array([-1.0]), np.array([1.0]))
    axis = np.eye(4, dtype=np.float32)
    tool = np.eye(4, dtype=np.float32)
    encoded = canonical_flange_from_se3_features(zero, tool, T_root_axis=axis)
    arithmetic = 0.5 * (
        canonical_flange_from_se3_features(neg, tool, T_root_axis=axis)
        + canonical_flange_from_se3_features(pos, tool, T_root_axis=axis)
    )
    assert np.isclose(encoded[0, 1], np.sqrt(0.1**2 + 0.1**2 + 1.0e-6))
    assert not np.allclose(encoded, arithmetic)

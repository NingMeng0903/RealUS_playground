"""Exact-output regression against frozen pre-optimization implementations.

The oracle functions below are copied from the existing repository before the
bounded performance change. Tests compare arrays AND subsequent RNG state.
"""
from __future__ import annotations
import numpy as np
import pytest
from ird_playground.ird.splits import _leakage_group_ids
from ird_playground.neural.train_signed import paired_epoch_batches

def oracle_leakage_group_ids(
    boundary_id: np.ndarray,
    source_pose_id: np.ndarray,
) -> np.ndarray:
    """Assign every row a group while prioritising source-pose ownership."""
    boundary = np.asarray(boundary_id, dtype=np.int64).reshape(-1)
    source = np.asarray(source_pose_id, dtype=np.int64).reshape(-1)
    if boundary.shape != source.shape:
        raise ValueError("boundary_id and source_pose_id must have the same shape")

    groups = np.empty(len(source), dtype=np.int64)
    next_group = 0
    source_values = np.unique(source[source >= 0])
    source_map = {int(value): i for i, value in enumerate(source_values.tolist())}
    next_group = len(source_map)
    for value, group in source_map.items():
        groups[source == value] = group

    unresolved_boundary = (source < 0) & (boundary >= 0)
    boundary_values = np.unique(boundary[unresolved_boundary])
    for value in boundary_values.tolist():
        groups[unresolved_boundary & (boundary == value)] = next_group
        next_group += 1

    unresolved_rows = np.flatnonzero((source < 0) & (boundary < 0))
    groups[unresolved_rows] = np.arange(
        next_group, next_group + len(unresolved_rows), dtype=np.int64
    )
    return groups

def oracle_paired_epoch_batches(
    tr_idx: np.ndarray,
    boundary_id: np.ndarray,
    reachable: np.ndarray,
    rng: np.random.Generator,
    *,
    batch_size: int,
    groups_per_batch: int = 256,
    sample_origin: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Build batches containing complete boundary groups plus stratified globals."""
    tr = np.asarray(tr_idx, dtype=np.int64)
    bid = np.asarray(boundary_id, dtype=np.int64)
    y = np.asarray(reachable, dtype=np.float32)
    if batch_size < 1 or groups_per_batch < 1:
        raise ValueError("batch_size and groups_per_batch must be positive")
    boundary_rows = tr[bid[tr] >= 0]
    groups = np.unique(bid[boundary_rows])
    rng.shuffle(groups)
    rows_by_group = {int(group): boundary_rows[bid[boundary_rows] == group] for group in groups}
    for group, rows in rows_by_group.items():
        if not ((y[rows] > 0.5).any() and (y[rows] < 0.5).any()):
            raise ValueError(f"boundary group {group} lacks a positive/negative pair")

    global_rows = tr[bid[tr] < 0]
    origin = (
        np.zeros(len(bid), dtype=np.int8)
        if sample_origin is None
        else np.asarray(sample_origin, dtype=np.int8)
    )
    strata: list[np.ndarray] = []
    for source_kind in np.unique(origin[global_rows]).tolist():
        source_rows = global_rows[origin[global_rows] == source_kind]
        for label in (False, True):
            rows = source_rows[(y[source_rows] > 0.5) == label]
            if len(rows):
                strata.append(rows)
    stratum_weights = np.asarray([len(rows) for rows in strata], dtype=np.float64)
    if stratum_weights.size:
        stratum_weights /= stratum_weights.sum()

    if not len(groups):
        order = rng.permutation(tr)
        return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]

    batches = []
    for start in range(0, len(groups), groups_per_batch):
        selected_groups = groups[start : start + groups_per_batch]
        paired = np.concatenate([rows_by_group[int(group)] for group in selected_groups])
        if len(paired) > batch_size:
            raise ValueError(
                f"{len(selected_groups)} complete groups require {len(paired)} rows, "
                f"exceeding batch_size={batch_size}"
            )
        n_fill = batch_size - len(paired)
        fill_parts = []
        if n_fill and strata:
            raw_counts = stratum_weights * n_fill
            counts = np.floor(raw_counts).astype(int)
            for index in np.argsort(raw_counts - counts)[::-1][: n_fill - counts.sum()]:
                counts[index] += 1
            for rows, count in zip(strata, counts, strict=True):
                if count:
                    fill_parts.append(rng.choice(rows, size=int(count), replace=count > len(rows)))
        ids = np.concatenate((paired, *fill_parts)) if fill_parts else paired
        rng.shuffle(ids)
        batches.append(ids)
    return batches

@pytest.mark.parametrize("seed", range(8))
def test_group_ids_identical_to_original(seed):
    rng = np.random.default_rng(seed)
    source = rng.integers(-3, 100, size=2000)
    boundary = rng.integers(-1, 60, size=2000)
    np.testing.assert_array_equal(_leakage_group_ids(boundary, source), oracle_leakage_group_ids(boundary, source))


@pytest.mark.parametrize("boundary,source", [([], []), ([-1,-1,-1],[-1,-1,-1]), ([8,8,2],[-1,-1,-1]), ([5,5,5],[7,-1,2]), ([-1,4],[10**12,10**11])])
def test_group_ids_edge_cases(boundary, source):
    np.testing.assert_array_equal(_leakage_group_ids(boundary, source), oracle_leakage_group_ids(boundary, source))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("with_origins", [False, True])
def test_paired_batches_arrays_and_rng_identical(seed, with_origins):
    rng = np.random.default_rng(73)
    group = np.repeat(np.arange(31) * 7 + 2, 9)
    bid = np.r_[group, np.full(317, -1)]
    reachable = np.r_[np.tile([0,0,0,0,.5,1,1,1,1],31), rng.integers(0,2,317)]
    # Unsorted input is deliberate: stable sorting must preserve original row order.
    tr_idx = rng.permutation(len(bid))
    origin = rng.integers(0,4,len(bid)) if with_origins else None
    a_rng, b_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    a = paired_epoch_batches(tr_idx,bid,reachable,a_rng,batch_size=61,groups_per_batch=5,sample_origin=origin)
    b = oracle_paired_epoch_batches(tr_idx,bid,reachable,b_rng,batch_size=61,groups_per_batch=5,sample_origin=origin)
    assert len(a) == len(b)
    for aa, bb in zip(a,b): np.testing.assert_array_equal(aa,bb)
    np.testing.assert_array_equal(a_rng.integers(0,2**31,100), b_rng.integers(0,2**31,100))


@pytest.mark.parametrize("n", [0,1,43])
def test_paired_batches_no_boundary_groups(n):
    tr = np.arange(n)[::-1]; bid=np.full(n,-1); y=np.zeros(n)
    a_rng,b_rng=np.random.default_rng(20),np.random.default_rng(20)
    a=paired_epoch_batches(tr,bid,y,a_rng,batch_size=17)
    b=oracle_paired_epoch_batches(tr,bid,y,b_rng,batch_size=17)
    assert len(a)==len(b)
    for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
    assert a_rng.bit_generator.state == b_rng.bit_generator.state


@pytest.mark.parametrize("kwargs,match", [({"batch_size":0},"positive"), ({"batch_size":1},"exceeding"), ({"bad_pair":True},"lacks")])
def test_paired_batches_errors_unchanged(kwargs,match):
    kw=dict(kwargs); bad=kw.pop("bad_pair",False)
    for func in [paired_epoch_batches,oracle_paired_epoch_batches]:
        with pytest.raises(ValueError,match=match):
            func(np.arange(2),np.array([8,8]),np.array([1,1] if bad else [0,1]),np.random.default_rng(4),**({"batch_size":4}|kw))

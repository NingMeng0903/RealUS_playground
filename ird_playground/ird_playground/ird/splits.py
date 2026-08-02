"""Leak-free grouped dataset splits for signed reachability training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SPLIT_NAMES = ("train", "selection", "zero_calibration", "safety_calibration", "test")


@dataclass(frozen=True)
class FiveWaySplitConfig:
    selection_fraction: float = 0.15
    zero_calibration_fraction: float = 0.05
    safety_calibration_fraction: float = 0.05
    test_fraction: float = 0.10

    def held_out_fractions(self) -> tuple[float, float, float, float]:
        values = (
            self.selection_fraction,
            self.zero_calibration_fraction,
            self.safety_calibration_fraction,
            self.test_fraction,
        )
        if any(not 0.0 <= float(value) < 1.0 for value in values):
            raise ValueError("split fractions must lie in [0, 1)")
        if sum(values) >= 1.0:
            raise ValueError("held-out split fractions must sum to less than one")
        return values


def _leakage_group_ids(
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


def five_way_split_indices(
    boundary_id: np.ndarray,
    source_pose_id: np.ndarray,
    *,
    seed: int,
    config: FiveWaySplitConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return five mutually exclusive row sets grouped by source pose."""
    cfg = config or FiveWaySplitConfig()
    fractions = cfg.held_out_fractions()
    row_group = _leakage_group_ids(boundary_id, source_pose_id)
    unique_groups = np.unique(row_group)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    counts = [int(np.floor(len(unique_groups) * fraction)) for fraction in fractions]
    for i, fraction in enumerate(fractions):
        if fraction > 0.0 and counts[i] == 0 and len(unique_groups) > sum(counts):
            counts[i] = 1
    while sum(counts) >= len(unique_groups) and any(counts):
        counts[int(np.argmax(counts))] -= 1

    out: dict[str, np.ndarray] = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES[1:], counts, strict=True):
        selected = unique_groups[offset : offset + count]
        out[name] = np.flatnonzero(np.isin(row_group, selected))
        offset += count
    train_groups = unique_groups[offset:]
    out["train"] = np.flatnonzero(np.isin(row_group, train_groups))
    return {name: out[name] for name in SPLIT_NAMES}


def assert_split_disjoint(
    splits: dict[str, np.ndarray],
    source_pose_id: np.ndarray,
) -> None:
    """Fail if a non-negative source pose occurs in more than one split."""
    source = np.asarray(source_pose_id, dtype=np.int64)
    owners: dict[int, str] = {}
    rows: set[int] = set()
    for name in SPLIT_NAMES:
        idx = np.asarray(splits[name], dtype=np.int64)
        overlap = rows.intersection(idx.tolist())
        if overlap:
            raise ValueError(f"row leakage into {name}: {len(overlap)} rows")
        rows.update(idx.tolist())
        for value in np.unique(source[idx][source[idx] >= 0]).tolist():
            previous = owners.setdefault(int(value), name)
            if previous != name:
                raise ValueError(f"source_pose_id {value} leaks across {previous}/{name}")
    if len(rows) != len(source):
        raise ValueError("five-way split does not cover every row")


__all__ = [
    "FiveWaySplitConfig",
    "SPLIT_NAMES",
    "assert_split_disjoint",
    "five_way_split_indices",
]

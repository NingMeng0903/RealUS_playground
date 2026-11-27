"""Focused offline-Q contract checks.

These tests intentionally stay independent of the simulator and only exercise
the transition audit, terminal Bellman target, deterministic artifact, and
FrozenQ inference contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fit_q import FrozenQ, _bellman_targets, fit_offline_q, load_transition_shards


def _write_shard(path: Path, offset: float = 0.0) -> None:
    features = np.asarray(
        [
            [[0.0 + offset, 0.0], [1.0 + offset, 0.0], [99.0, 99.0]],
            [[1.0 + offset, 0.0], [2.0 + offset, 0.0], [99.0, 99.0]],
            [[2.0 + offset, 0.0], [3.0 + offset, 0.0], [99.0, 99.0]],
        ],
        dtype=np.float64,
    )
    mask = np.asarray([[True, True, False], [True, True, False], [True, False, False]])
    actions = np.asarray([0, 1, 0], dtype=np.int64)
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    next_index = np.asarray([1, -1, -1], dtype=np.int64)
    episode_seed = np.asarray([42, 42, 42], dtype=np.int64)
    np.savez(
        path,
        features=features,
        mask=mask,
        actions=actions,
        rewards=rewards,
        next_index=next_index,
        episode_seed=episode_seed,
    )


def test_terminal_bellman_target_excludes_padding() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0])
    next_index = np.asarray([1, -1, -1])
    mask = np.asarray([[True, True, False], [True, True, False], [True, False, False]])
    # The padded slot has a very large score and must not enter the max.
    scores = np.asarray([[0.0, 0.0, 0.0], [4.0, 5.0, 1000.0], [8.0, 0.0, 0.0]])
    target, audit = _bellman_targets(rewards, next_index, mask, scores, 0.99)
    assert np.isclose(target[0], 1.0 + 0.99 * 5.0)
    assert np.isclose(target[1], 2.0)
    assert np.isclose(target[2], 3.0)
    assert audit["terminal_zero_bootstrap_rows"] == 2


def test_fit_checkpoint_and_frozen_q(tmp_path: Path) -> None:
    shard = tmp_path / "TRANSITIONS.npz"
    _write_shard(shard)
    data = load_transition_shards([shard])
    training = fit_offline_q(data, tmp_path / "out")
    policy = tmp_path / "out" / "POLICY.npz"
    assert policy.is_file()
    assert (tmp_path / "out" / "MC_POLICY.npz").is_file()
    report = json.loads((tmp_path / "out" / "TRAINING.json").read_text())
    assert report["target_audit"]["terminal_zero_bootstrap_required"] is True
    assert training["policy_sha256"] == report["policy_sha256"]

    frozen = FrozenQ(policy)
    scores = frozen.predict(data.features[0], data.mask[0])
    assert scores.shape == (3,)
    assert np.isneginf(scores[2])
    assert np.all(np.isfinite(scores[:2]))


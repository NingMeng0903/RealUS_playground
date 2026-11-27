#!/usr/bin/env python3
"""Deterministic offline fitted-Q iteration for candidate transition shards.

The trainer deliberately has no simulator dependency.  A transition shard is
an ``npz`` file containing arrays named ``features``, ``mask``, ``actions``,
``rewards``, ``next_index`` and ``episode_seed``.  ``next_index`` is local to
its shard; it is offset while shards are concatenated.  The resulting
checkpoint is an ordinary, non-pickle ``npz`` file and can be read by
``FrozenQ`` without importing torch.

This module keeps the fitting implementation in NumPy so that it remains
usable in the small data-preparation environment.  The network is a shared
per-candidate MLP: every candidate in a state is scored by the same network,
and the state mask is used only for action support and Bellman maximisation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import zipfile
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GAMMA = 0.99
N_FQI_ITERATIONS = 5
HIDDEN_DIM = 32
INITIAL_EPOCHS = 60
FQI_EPOCHS = 35
BATCH_SIZE = 2048
LEARNING_RATE = 2.0e-3
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1.0e-8
CONSERVATIVE_PENALTY = 2.0e-2
WEIGHT_DECAY = 1.0e-5
SEED = 1729
MASK_VALUE = -np.inf


class DatasetAuditError(ValueError):
    """Raised when a transition shard violates the offline data contract."""


@dataclass(frozen=True)
class TransitionData:
    """Concatenated transition arrays and their audit metadata."""

    features: np.ndarray
    mask: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_index: np.ndarray
    episode_seed: np.ndarray
    source_paths: tuple[str, ...]
    source_sha256: tuple[str, ...]
    source_rows: tuple[int, ...]
    audit: Mapping[str, Any]


def _as_path_list(paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> list[Path]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    result = [Path(p).expanduser().resolve() for p in paths]
    if not result:
        raise DatasetAuditError("at least one transition shard is required")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_array(data: Mapping[str, Any], name: str, path: Path) -> np.ndarray:
    if name not in data:
        raise DatasetAuditError(f"{path}: missing required array {name!r}")
    return np.asarray(data[name])


def _check_integer_array(array: np.ndarray, name: str, path: Path) -> np.ndarray:
    if not np.issubdtype(array.dtype, np.integer):
        raise DatasetAuditError(f"{path}: {name} must have an integer dtype")
    return np.asarray(array, dtype=np.int64)


def _check_shard(path: Path, expected_k: int | None, expected_f: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.is_file():
        raise DatasetAuditError(f"transition shard does not exist: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - error text is environment-specific
        raise DatasetAuditError(f"could not read {path}: {exc}") from exc
    with loaded as data:
        features = _require_array(data, "features", path)
        mask_raw = _require_array(data, "mask", path)
        actions = _require_array(data, "actions", path)
        rewards = _require_array(data, "rewards", path)
        next_index = _require_array(data, "next_index", path)
        episode_seed = _require_array(data, "episode_seed", path)

    if features.ndim != 3:
        raise DatasetAuditError(f"{path}: features must have shape (N,K,F), got {features.shape}")
    n, k, f = features.shape
    if n == 0 or k == 0 or f == 0:
        raise DatasetAuditError(f"{path}: features must have non-zero N, K and F")
    if expected_k is not None and k != expected_k:
        raise DatasetAuditError(f"{path}: K={k} disagrees with previous shards K={expected_k}")
    if expected_f is not None and f != expected_f:
        raise DatasetAuditError(f"{path}: F={f} disagrees with previous shards F={expected_f}")

    if mask_raw.shape != (n, k):
        raise DatasetAuditError(f"{path}: mask must have shape {(n, k)}, got {mask_raw.shape}")
    if mask_raw.dtype == np.bool_:
        mask = mask_raw.astype(bool, copy=False)
    else:
        # Accept an unambiguous 0/1 integer mask while rejecting accidental
        # floating-point probabilities or arbitrary integer labels.
        try:
            finite_mask = np.isfinite(mask_raw)
            binary_mask = (mask_raw == 0) | (mask_raw == 1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DatasetAuditError(f"{path}: mask must be boolean or contain only 0/1 values") from exc
        if not np.all(finite_mask) or not np.all(binary_mask):
            raise DatasetAuditError(f"{path}: mask must be boolean or contain only 0/1 values")
        mask = mask_raw.astype(bool)
    if not np.all(mask[:, 0]):
        raise DatasetAuditError(f"{path}: anchor candidate slot 0 must be valid in every row")
    # A candidate mask is a padded prefix.  This catches an off-by-one or a
    # malformed shard before it can silently alter the Bellman max.
    if k > 1 and np.any(mask[:, 1:] & ~np.logical_and.accumulate(mask[:, :-1], axis=1)):
        raise DatasetAuditError(f"{path}: mask must be a contiguous valid prefix (anchor slot 0 included)")

    if actions.shape != (n,):
        raise DatasetAuditError(f"{path}: actions must have shape {(n,)}, got {actions.shape}")
    actions = _check_integer_array(actions, "actions", path)
    if np.any(actions < 0) or np.any(actions >= k):
        raise DatasetAuditError(f"{path}: observed action is outside candidate slots")
    row_ids = np.arange(n)
    if np.any(~mask[row_ids, actions]):
        bad = int(np.flatnonzero(~mask[row_ids, actions])[0])
        raise DatasetAuditError(f"{path}: observed action {int(actions[bad])} is padded/invalid at row {bad}")

    if rewards.shape != (n,):
        raise DatasetAuditError(f"{path}: rewards must have shape {(n,)}, got {rewards.shape}")
    try:
        rewards = np.asarray(rewards, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetAuditError(f"{path}: rewards must be numeric and finite") from exc
    if not np.all(np.isfinite(rewards)):
        raise DatasetAuditError(f"{path}: rewards must be finite")

    if next_index.shape != (n,):
        raise DatasetAuditError(f"{path}: next_index must have shape {(n,)}, got {next_index.shape}")
    next_index = _check_integer_array(next_index, "next_index", path)
    if np.any(next_index < -1) or np.any(next_index >= n):
        raise DatasetAuditError(f"{path}: next_index must be -1 or a local row index")
    nonterminal = next_index >= 0
    if np.any(nonterminal & (next_index <= row_ids)):
        bad = int(np.flatnonzero(nonterminal & (next_index <= row_ids))[0])
        raise DatasetAuditError(
            f"{path}: nonterminal next_index must be strictly forward; row {bad} -> {int(next_index[bad])}"
        )

    if episode_seed.shape != (n,):
        raise DatasetAuditError(f"{path}: episode_seed must have shape {(n,)}, got {episode_seed.shape}")
    # Comparing object arrays is deliberately not supported: they would make
    # the audit and the deterministic checkpoint dependent on pickle-like
    # Python objects.  Strings and numeric seeds are both fine.
    if episode_seed.dtype.kind not in "biufSU":
        raise DatasetAuditError(f"{path}: episode_seed must be numeric or string, not {episode_seed.dtype}")
    for row in np.flatnonzero(nonterminal):
        nxt = int(next_index[row])
        if episode_seed[row] != episode_seed[nxt]:
            raise DatasetAuditError(
                f"{path}: nonterminal transition changes episode_seed at row {row} -> {nxt}"
            )

    # Invalid padded feature values are not observations and may be arbitrary;
    # valid candidate features must remain finite for normalization and fitting.
    try:
        features = np.asarray(features, dtype=np.float64)
        finite_valid = np.isfinite(features[mask])
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetAuditError(f"{path}: features must be numeric and finite on valid candidates") from exc
    if np.any(~finite_valid):
        raise DatasetAuditError(f"{path}: valid candidate features must be finite")
    valid = mask[..., None]
    # Keep invalid slots finite so downstream code can vectorize safely.  They
    # are always excluded by the mask and never affect training or prediction.
    features = np.where(valid, features, 0.0)

    audit = {
        "rows": int(n),
        "candidate_slots": int(k),
        "feature_dim": int(f),
        "valid_candidate_count": int(mask.sum()),
        "padded_candidate_count": int(mask.size - mask.sum()),
        "observed_action_count": int(n),
        "terminal_count": int((~nonterminal).sum()),
        "nonterminal_count": int(nonterminal.sum()),
        "same_episode_forward_count": int(nonterminal.sum()),
        "unique_episode_seed_count": int(np.unique(episode_seed).size),
        "invalid_observed_action_count": 0,
        "anchor_invalid_count": 0,
        "noncontiguous_mask_count": 0,
    }
    return {
        "features": features,
        "mask": mask,
        "actions": actions,
        "rewards": rewards,
        "next_index": next_index,
        "episode_seed": episode_seed.copy(),
    }, audit


def load_transition_shards(paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]]) -> TransitionData:
    """Load, audit and concatenate transition shards.

    Each ``next_index`` is interpreted in its shard's local row coordinate and
    offset after validation.  The returned arrays are suitable for training;
    all counts in ``audit`` are Python scalars so the audit can be serialized
    directly to JSON.
    """

    path_list = _as_path_list(paths)
    chunks: list[dict[str, np.ndarray]] = []
    audits: list[dict[str, Any]] = []
    k: int | None = None
    f: int | None = None
    offsets: list[int] = []
    source_rows: list[int] = []
    source_hashes: list[str] = []
    total = 0
    for path in path_list:
        offsets.append(total)
        chunk, audit = _check_shard(path, k, f)
        if k is None:
            k = int(chunk["features"].shape[1])
            f = int(chunk["features"].shape[2])
        chunks.append(chunk)
        audits.append(audit)
        rows = int(chunk["features"].shape[0])
        source_rows.append(rows)
        source_hashes.append(_sha256(path))
        total += rows

    assert k is not None and f is not None
    next_parts = []
    for chunk, offset in zip(chunks, offsets):
        local = chunk["next_index"]
        next_parts.append(np.where(local >= 0, local + offset, -1))
    features = np.concatenate([c["features"] for c in chunks], axis=0)
    mask = np.concatenate([c["mask"] for c in chunks], axis=0)
    actions = np.concatenate([c["actions"] for c in chunks], axis=0)
    rewards = np.concatenate([c["rewards"] for c in chunks], axis=0)
    next_index = np.concatenate(next_parts, axis=0).astype(np.int64, copy=False)
    episode_seed = np.concatenate([c["episode_seed"] for c in chunks], axis=0)

    nonterminal = next_index >= 0
    if np.any(nonterminal):
        rows = np.arange(total)
        if np.any(next_index[nonterminal] <= rows[nonterminal]):
            raise DatasetAuditError("concatenated next_index is not strictly forward")
        if np.any(episode_seed[nonterminal] != episode_seed[next_index[nonterminal]]):
            raise DatasetAuditError("concatenated next_index changes episode_seed")

    audit: dict[str, Any] = {
        "rows": int(total),
        "shards": int(len(chunks)),
        "candidate_slots": int(k),
        "feature_dim": int(f),
        "valid_candidate_count": int(mask.sum()),
        "padded_candidate_count": int(mask.size - mask.sum()),
        "observed_action_count": int(total),
        "terminal_count": int((~nonterminal).sum()),
        "nonterminal_count": int(nonterminal.sum()),
        "same_episode_forward_count": int(nonterminal.sum()),
        "unique_episode_seed_count": int(np.unique(episode_seed).size),
        "invalid_observed_action_count": 0,
        "anchor_invalid_count": 0,
        "noncontiguous_mask_count": 0,
        "per_shard": audits,
        "shard_offsets": [int(x) for x in offsets],
    }
    return TransitionData(
        features=features,
        mask=mask,
        actions=actions,
        rewards=rewards,
        next_index=next_index,
        episode_seed=episode_seed,
        source_paths=tuple(str(p) for p in path_list),
        source_sha256=tuple(source_hashes),
        source_rows=tuple(source_rows),
        audit=audit,
    )


def train_normalizer(features: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute feature moments from valid training candidates only."""

    valid = np.asarray(features, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    if valid.ndim != 2 or valid.shape[0] == 0:
        raise DatasetAuditError("cannot normalize an empty valid training feature set")
    mean = valid.mean(axis=0)
    variance = np.mean((valid - mean) ** 2, axis=0)
    # The floor is fixed rather than data-tuned.  Constant features are still
    # retained, while avoiding a divide-by-zero in both trainer and simulator.
    std = np.sqrt(np.maximum(variance, 1.0e-12))
    audit = {
        "valid_feature_rows": int(valid.shape[0]),
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "constant_feature_count": int(np.count_nonzero(std <= 1.0e-6)),
    }
    return mean, std, audit


def _discounted_empirical_returns(rewards: np.ndarray, next_index: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns along strictly-forward logged links."""

    n = len(rewards)
    returns = np.zeros(n, dtype=np.float64)
    for row in range(n - 1, -1, -1):
        nxt = int(next_index[row])
        returns[row] = float(rewards[row]) if nxt < 0 else float(rewards[row]) + gamma * returns[nxt]
    return returns


class _NumpyMLP:
    """Tiny tanh MLP with deterministic Adam updates."""

    def __init__(self, feature_dim: int, hidden_dim: int, seed: int = SEED) -> None:
        rng = np.random.default_rng(seed)
        # Xavier initialization is fixed by the explicit seed.  Float64 keeps
        # CPU training stable; checkpoints are converted to float32 on write.
        scale = math.sqrt(6.0 / (feature_dim + hidden_dim))
        self.w1 = rng.uniform(-scale, scale, size=(feature_dim, hidden_dim)).astype(np.float64)
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        scale2 = math.sqrt(6.0 / (hidden_dim + 1))
        self.w2 = rng.uniform(-scale2, scale2, size=(hidden_dim, 1)).astype(np.float64)
        self.b2 = np.zeros(1, dtype=np.float64)
        self._adam_m = [np.zeros_like(p) for p in self.parameters]
        self._adam_v = [np.zeros_like(p) for p in self.parameters]
        self._step = 0

    @property
    def parameters(self) -> list[np.ndarray]:
        return [self.w1, self.b1, self.w2, self.b2]

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.tanh(x @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2).reshape(-1)

    def _gradient(self, x: np.ndarray, labels: np.ndarray, logged: np.ndarray, penalty: float) -> tuple[float, list[np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        hidden = np.tanh(z1)
        q = (hidden @ self.w2 + self.b2).reshape(-1)
        logged = np.asarray(logged, dtype=bool)
        unlogged = ~logged
        n_logged = max(int(logged.sum()), 1)
        n_unlogged = max(int(unlogged.sum()), 1)
        dq = np.zeros_like(q)
        if np.any(logged):
            residual = q[logged] - labels[logged]
            dq[logged] = residual / n_logged
            mse = float(np.mean(residual * residual))
        else:
            mse = 0.0
        if np.any(unlogged):
            # Only optimistic (positive) unsupported values are shrunk.  This
            # is a small one-sided conservative penalty: it does not force an
            # unlogged option with genuinely negative value toward zero.
            positive = np.maximum(q[unlogged], 0.0)
            dq[unlogged] += penalty * positive / n_unlogged
            conservative = float(np.mean(positive * positive))
        else:
            conservative = 0.0
        d_out = dq[:, None]
        dw2 = hidden.T @ d_out + WEIGHT_DECAY * self.w2
        db2 = d_out.sum(axis=0)
        dhidden = d_out @ self.w2.T
        dz1 = dhidden * (1.0 - hidden * hidden)
        dw1 = x.T @ dz1 + WEIGHT_DECAY * self.w1
        db1 = dz1.sum(axis=0)
        loss = 0.5 * mse + 0.5 * penalty * conservative + 0.5 * WEIGHT_DECAY * (
            float(np.sum(self.w1 * self.w1)) + float(np.sum(self.w2 * self.w2))
        )
        return loss, [dw1, db1, dw2, db2]

    def update(self, grads: Sequence[np.ndarray], learning_rate: float = LEARNING_RATE) -> None:
        self._step += 1
        b1 = ADAM_BETA1
        b2 = ADAM_BETA2
        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            self._adam_m[i] = b1 * self._adam_m[i] + (1.0 - b1) * grad
            self._adam_v[i] = b2 * self._adam_v[i] + (1.0 - b2) * (grad * grad)
            m_hat = self._adam_m[i] / (1.0 - b1**self._step)
            v_hat = self._adam_v[i] / (1.0 - b2**self._step)
            param -= learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        logged: np.ndarray,
        *,
        epochs: int,
        batch_size: int = BATCH_SIZE,
        penalty: float = CONSERVATIVE_PENALTY,
    ) -> list[float]:
        if x.ndim != 2 or labels.shape != (len(x),) or logged.shape != (len(x),):
            raise ValueError("MLP training arrays have incompatible shapes")
        losses: list[float] = []
        # No random minibatch order: deterministic order is part of the offline
        # artifact.  A fixed batch size still bounds memory on large shards.
        for _ in range(int(epochs)):
            total_loss = 0.0
            total_count = 0
            for start in range(0, len(x), int(batch_size)):
                stop = min(start + int(batch_size), len(x))
                loss, grads = self._gradient(x[start:stop], labels[start:stop], logged[start:stop], penalty)
                self.update(grads)
                count = stop - start
                total_loss += loss * count
                total_count += count
            losses.append(total_loss / max(total_count, 1))
        return losses


def _normalize_features(features: np.ndarray, mask: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    finite = np.isfinite(features)
    safe = np.where(finite, features, 0.0)
    normalized = (safe - mean) / std
    # Keep padded rows finite and neutral; the mask will exclude them from all
    # losses and Bellman maxima.
    return np.where(mask[..., None], normalized, 0.0)


def _flatten_valid(
    normalized: np.ndarray,
    mask: np.ndarray,
    actions: np.ndarray,
    row_selection: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, k, f = normalized.shape
    if row_selection is None:
        # Keep the original default path and operation order unchanged.
        rows, candidates = np.nonzero(mask)
        flat_x = normalized[rows, candidates]
    else:
        selected = np.flatnonzero(np.asarray(row_selection, dtype=bool))
        local_rows, candidates = np.nonzero(mask[selected])
        rows = selected[local_rows]
        flat_x = normalized[rows, candidates]
    logged = candidates == actions[rows]
    logged_rows = rows[logged]
    logged_candidates = candidates[logged]
    expected_rows = n if row_selection is None else int(np.count_nonzero(row_selection))
    assert len(logged_rows) == expected_rows
    return flat_x, rows, candidates, logged


def _score_all(net: _NumpyMLP, normalized: np.ndarray) -> np.ndarray:
    return net.forward(normalized.reshape(-1, normalized.shape[-1])).reshape(normalized.shape[:2])


def _bellman_targets(
    rewards: np.ndarray,
    next_index: np.ndarray,
    mask: np.ndarray,
    next_scores: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(rewards)
    target = np.asarray(rewards, dtype=np.float64).copy()
    nonterminal = next_index >= 0
    bootstrap = np.zeros(n, dtype=np.float64)
    for row in np.flatnonzero(nonterminal):
        nxt = int(next_index[row])
        valid_scores = next_scores[nxt][mask[nxt]]
        if valid_scores.size == 0:  # should be prevented by the anchor audit
            raise DatasetAuditError(f"next state {nxt} has no valid candidates")
        bootstrap[row] = float(np.max(valid_scores))
        target[row] += gamma * bootstrap[row]
    audit = {
        "rows": int(n),
        "terminal_rows": int((~nonterminal).sum()),
        "nonterminal_rows": int(nonterminal.sum()),
        "terminal_zero_bootstrap_rows": int((~nonterminal & (bootstrap == 0.0)).sum()),
        "nonterminal_bootstrap_min": float(bootstrap[nonterminal].min()) if np.any(nonterminal) else 0.0,
        "nonterminal_bootstrap_max": float(bootstrap[nonterminal].max()) if np.any(nonterminal) else 0.0,
        "target_min": float(target.min()) if target.size else 0.0,
        "target_max": float(target.max()) if target.size else 0.0,
        "target_mean": float(target.mean()) if target.size else 0.0,
    }
    return target, audit


def _save_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with stable member order and timestamps (no pickle)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            if not name.endswith(".npy"):
                member = f"{name}.npy"
            else:
                member = name
            payload = io.BytesIO()
            np.lib.format.write_array(payload, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(member)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    os.replace(temporary, path)


def fit_offline_q(
    data: TransitionData,
    out_dir: str | os.PathLike[str],
    *,
    gamma: float = GAMMA,
    device: str = "auto",
    fit_row_mask: np.ndarray | None = None,
    fit_row_mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit the fixed offline Q model and write ``POLICY.npz``/``TRAINING.json``."""

    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")
    # ``device`` is accepted for CLI/API compatibility.  This NumPy trainer
    # is intentionally CPU-deterministic; no hidden test-set or device tuning
    # is performed.
    requested_device = str(device)
    selected_rows: np.ndarray | None = None
    if fit_row_mask is not None:
        selected_rows = np.asarray(fit_row_mask)
        if selected_rows.ndim != 1 or len(selected_rows) != len(data.rewards):
            raise DatasetAuditError(
                "fit_row_mask must be a one-dimensional vector with one entry per transition row"
            )
        if selected_rows.dtype != np.bool_:
            raise DatasetAuditError("fit_row_mask must have boolean dtype")
        selected_rows = selected_rows.astype(bool, copy=False)
        if not np.any(selected_rows):
            raise DatasetAuditError("fit_row_mask must select at least one transition row")
        mean, std, norm_audit = train_normalizer(
            data.features[selected_rows], data.mask[selected_rows]
        )
    else:
        # Preserve the original default call exactly.
        mean, std, norm_audit = train_normalizer(data.features, data.mask)
    normalized = _normalize_features(data.features, data.mask, mean, std)
    flat_x, flat_rows, flat_candidates, logged = _flatten_valid(
        normalized,
        data.mask,
        data.actions,
        row_selection=selected_rows,
    )
    empirical_returns = _discounted_empirical_returns(data.rewards, data.next_index, gamma)
    logged_labels = empirical_returns[flat_rows]
    labels = np.zeros(len(flat_x), dtype=np.float64)
    labels[logged] = logged_labels[logged]

    net = _NumpyMLP(normalized.shape[-1], HIDDEN_DIM, seed=SEED)
    logs: list[dict[str, Any]] = []
    initial_losses = net.fit(flat_x, labels, logged, epochs=INITIAL_EPOCHS)
    logs.append(
        {
            "iteration": 0,
            "kind": "empirical_return_initialization",
            "epochs": int(INITIAL_EPOCHS),
            "loss_first": float(initial_losses[0]),
            "loss_last": float(initial_losses[-1]),
            "logged_target_min": float(empirical_returns.min()),
            "logged_target_max": float(empirical_returns.max()),
            "logged_target_mean": float(empirical_returns.mean()),
        }
    )

    # Keep the empirical-return initialization as a deterministic, same-data
    # non-bootstrapping control.  It uses the exact same architecture and
    # normalizer as POLICY.npz; no second fit or hyperparameter selection is
    # performed.
    mc_arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "feature_dim": np.asarray(normalized.shape[-1], dtype=np.int64),
        "hidden_dim": np.asarray(HIDDEN_DIM, dtype=np.int64),
        "action_count": np.asarray(normalized.shape[1], dtype=np.int64),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "mask_value": np.asarray(MASK_VALUE, dtype=np.float64),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "w1": net.w1.astype(np.float64),
        "b1": net.b1.astype(np.float64),
        "w2": net.w2.astype(np.float64),
        "b2": net.b2.astype(np.float64),
    }

    for iteration in range(1, N_FQI_ITERATIONS + 1):
        previous_scores = _score_all(net, normalized)
        bellman, target_audit = _bellman_targets(data.rewards, data.next_index, data.mask, previous_scores, gamma)
        labels = np.zeros(len(flat_x), dtype=np.float64)
        labels[logged] = bellman[flat_rows[logged]]
        losses = net.fit(flat_x, labels, logged, epochs=FQI_EPOCHS)
        logs.append(
            {
                "iteration": int(iteration),
                "kind": "fitted_bellman",
                "epochs": int(FQI_EPOCHS),
                "loss_first": float(losses[0]),
                "loss_last": float(losses[-1]),
                "target_audit": target_audit,
                "logged_target_min": float(bellman.min()),
                "logged_target_max": float(bellman.max()),
                "logged_target_mean": float(bellman.mean()),
            }
        )

    out = Path(out_dir).expanduser().resolve()
    checkpoint = out / "POLICY.npz"
    arrays = {
        "format_version": np.asarray(1, dtype=np.int64),
        "feature_dim": np.asarray(normalized.shape[-1], dtype=np.int64),
        "hidden_dim": np.asarray(HIDDEN_DIM, dtype=np.int64),
        "action_count": np.asarray(normalized.shape[1], dtype=np.int64),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "mask_value": np.asarray(MASK_VALUE, dtype=np.float64),
        "mean": mean.astype(np.float64),
        "std": std.astype(np.float64),
        "w1": net.w1.astype(np.float64),
        "b1": net.b1.astype(np.float64),
        "w2": net.w2.astype(np.float64),
        "b2": net.b2.astype(np.float64),
    }
    _save_deterministic_npz(checkpoint, arrays)
    mc_checkpoint = out / "MC_POLICY.npz"
    _save_deterministic_npz(mc_checkpoint, mc_arrays)
    policy_sha256 = _sha256(checkpoint)
    mc_policy_sha256 = _sha256(mc_checkpoint)
    training = {
        "format_version": 1,
        "method": "offline_fitted_q_iteration",
        "policy_file": "POLICY.npz",
        "policy_sha256": policy_sha256,
        "mc_policy_file": "MC_POLICY.npz",
        "mc_policy_sha256": mc_policy_sha256,
        "requested_device": requested_device,
        "actual_device": "cpu_numpy",
        "hyperparameters": {
            "gamma": float(gamma),
            "fitted_iterations": int(N_FQI_ITERATIONS),
            "hidden_dim": int(HIDDEN_DIM),
            "initial_epochs": int(INITIAL_EPOCHS),
            "fqi_epochs": int(FQI_EPOCHS),
            "batch_size": int(BATCH_SIZE),
            "learning_rate": float(LEARNING_RATE),
            "conservative_penalty": float(CONSERVATIVE_PENALTY),
            "weight_decay": float(WEIGHT_DECAY),
            "seed": int(SEED),
            "activation": "tanh",
        },
        "data": {
            "paths": list(data.source_paths),
            "sha256": list(data.source_sha256),
            "rows_per_shard": [int(x) for x in data.source_rows],
            "rows": int(len(data.rewards)),
        },
        "audit": dict(data.audit),
        "normalizer": norm_audit,
        "target_audit": {
            "empirical_return_min": float(empirical_returns.min()),
            "empirical_return_max": float(empirical_returns.max()),
            "empirical_return_mean": float(empirical_returns.mean()),
            "terminal_zero_bootstrap_required": True,
            "iterations": [entry["target_audit"] for entry in logs if "target_audit" in entry],
        },
        "logs": logs,
    }
    if selected_rows is not None:
        mask_sha = fit_row_mask_sha256
        if mask_sha is None:
            mask_sha = hashlib.sha256(
                np.ascontiguousarray(selected_rows, dtype=np.bool_).tobytes()
            ).hexdigest()
        training["fit_row_mask"] = {
            "selected_rows": int(selected_rows.sum()),
            "total_rows": int(len(selected_rows)),
            "sha256": mask_sha,
        }
    out.mkdir(parents=True, exist_ok=True)
    training_path = out / "TRAINING.json"
    temporary = training_path.with_name(training_path.name + ".tmp")
    temporary.write_text(json.dumps(training, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, training_path)
    return training


class FrozenQ:
    """Read-only NumPy inference for an offline ``POLICY.npz`` checkpoint."""

    def __init__(self, checkpoint: str | os.PathLike[str]) -> None:
        path = Path(checkpoint).expanduser()
        if path.is_dir():
            path = path / "POLICY.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as data:
            required = ("feature_dim", "action_count", "mean", "std", "w1", "b1", "w2", "b2")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(f"{self.path}: checkpoint missing {missing}")
            self.feature_dim = int(np.asarray(data["feature_dim"]).item())
            self.action_count = int(np.asarray(data["action_count"]).item())
            self.mean = np.asarray(data["mean"], dtype=np.float64).copy()
            self.std = np.asarray(data["std"], dtype=np.float64).copy()
            self.w1 = np.asarray(data["w1"], dtype=np.float64).copy()
            self.b1 = np.asarray(data["b1"], dtype=np.float64).copy()
            self.w2 = np.asarray(data["w2"], dtype=np.float64).copy()
            self.b2 = np.asarray(data["b2"], dtype=np.float64).copy()
        if self.mean.shape != (self.feature_dim,) or self.std.shape != (self.feature_dim,):
            raise ValueError("checkpoint normalizer shape does not match feature_dim")
        hidden = self.w1.shape[1] if self.w1.ndim == 2 else -1
        if self.w1.shape != (self.feature_dim, hidden) or self.b1.shape != (hidden,):
            raise ValueError("checkpoint first layer shape is invalid")
        if self.w2.shape not in {(hidden, 1), (hidden,)}:
            raise ValueError("checkpoint output layer shape is invalid")
        if self.b2.size != 1:
            raise ValueError("checkpoint output bias must be scalar")
        if np.any(~np.isfinite(self.mean)) or np.any(~np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("checkpoint normalizer is not finite/positive")

    def predict(self, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return one score per candidate, with padded candidates set to ``-inf``.

        ``features``/``mask`` may be ``(K,F)``/``(K,)`` for one state or
        ``(B,K,F)``/``(B,K)`` for a batch.  The unbatched form is the simulator
        contract and returns a one-dimensional length-``K`` array.
        """

        x = np.asarray(features)
        m = np.asarray(mask, dtype=bool)
        batched = x.ndim == 3
        if x.ndim == 2:
            x_work = x[None, ...]
            m_work = m[None, ...]
        elif x.ndim == 3:
            x_work = x
            m_work = m
        else:
            raise ValueError(f"features must have shape (K,F) or (B,K,F), got {x.shape}")
        if x_work.shape[-1] != self.feature_dim:
            raise ValueError(f"features last dimension must be {self.feature_dim}, got {x_work.shape[-1]}")
        if m_work.shape != x_work.shape[:2]:
            raise ValueError(f"mask shape must be {x_work.shape[:2]}, got {m_work.shape}")
        if np.any(~np.isfinite(x_work[m_work])):
            raise ValueError("valid candidate features must be finite")
        safe = np.where(np.isfinite(x_work), x_work, 0.0).astype(np.float64, copy=False)
        normalized = (safe - self.mean) / self.std
        flat = normalized.reshape(-1, self.feature_dim)
        hidden = np.tanh(flat @ self.w1 + self.b1)
        q = (hidden @ self.w2.reshape(-1, 1) + float(self.b2.reshape(-1)[0])).reshape(x_work.shape[:2])
        q = np.where(m_work, q, MASK_VALUE)
        return q if batched else q[0]


def _load_fit_row_mask(path: str | os.PathLike[str], total_rows: int) -> tuple[np.ndarray, str]:
    """Load and validate an explicit boolean row mask in concatenated order."""

    mask_path = Path(path).expanduser().resolve()
    if not mask_path.is_file():
        raise DatasetAuditError(f"fit row mask does not exist: {mask_path}")
    try:
        loaded = np.load(mask_path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - numpy error text varies
        raise DatasetAuditError(f"could not read fit row mask {mask_path}: {exc}") from exc
    if not isinstance(loaded, np.ndarray):
        raise DatasetAuditError("fit row mask must be a .npy boolean vector")
    mask = np.asarray(loaded)
    if mask.ndim != 1 or len(mask) != int(total_rows):
        raise DatasetAuditError(
            f"fit row mask must have shape ({int(total_rows)},), got {mask.shape}"
        )
    if mask.dtype != np.bool_:
        raise DatasetAuditError("fit row mask must have boolean dtype")
    if not np.any(mask):
        raise DatasetAuditError("fit row mask must select at least one transition row")
    return mask.astype(bool, copy=False), _sha256(mask_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        action="append",
        nargs="+",
        required=True,
        help="one or more TRANSITIONS.npz shard paths; option may be repeated",
    )
    parser.add_argument("--out", required=True, help="output directory for POLICY.npz and TRAINING.json")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="recorded device hint")
    parser.add_argument("--gamma", type=float, default=GAMMA, help=f"Bellman discount (default: {GAMMA:g})")
    parser.add_argument(
        "--fit-row-mask",
        default=None,
        help="optional .npy boolean row mask in concatenated loaded-shard order",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = [item for group in args.data for item in group]
    data = load_transition_shards(paths)
    fit_row_mask = None
    fit_row_mask_sha256 = None
    if args.fit_row_mask is not None:
        fit_row_mask, fit_row_mask_sha256 = _load_fit_row_mask(
            args.fit_row_mask, len(data.rewards)
        )
    training = fit_offline_q(
        data,
        args.out,
        device=args.device,
        gamma=args.gamma,
        fit_row_mask=fit_row_mask,
        fit_row_mask_sha256=fit_row_mask_sha256,
    )
    print(
        json.dumps(
            {
                "policy": str(Path(args.out).expanduser().resolve() / "POLICY.npz"),
                "rows": training["data"]["rows"],
                "shards": training["data"]["rows_per_shard"],
                "iterations": training["hyperparameters"]["fitted_iterations"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())

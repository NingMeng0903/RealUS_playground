"""Offline candidate policies and a continuous task-information proxy.

This module is deliberately a small adapter around the frozen learner and
candidate-feature implementations from ``ab_knee_latent_rl_20261112``.  The
map recheck driver owns transition collection and feasibility; this file only
adds the public resource summary to the frozen proposal features, fits the
three requested learners, and supplies a geometry-only Fisher-style bonus.

No simulator case, truth parameter, or identity label is accepted by any of
the public functions below.  In particular, :func:`fisher_bonus` uses the
current posterior grid and the public projection/design geometry only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


# The frozen modules use imports such as ``from belief import ...`` rather
# than package-relative imports.  Put their directory first while importing
# them, without modifying any source in the frozen experiment.
_HERE = Path(__file__).resolve().parent
_FROZEN = _HERE.parent / "ab_knee_latent_rl_20261112"
if str(_FROZEN) not in sys.path:
    sys.path.insert(0, str(_FROZEN))

import learners as _learners  # noqa: E402
from policy import proposal_features as _proposal_features  # noqa: E402


MAX_FRAMES = 3
ACTION_SLOTS = 6
RESOURCE_DIM = 3
_MODEL_SPECS = {
    "cql_7": ("cql", 7),
    "cql_29": ("cql", 29),
    "fqi_7": ("fqi", 7),
}


def _resource_vector(resource_fractions: Sequence[float] | Mapping[str, float]) -> np.ndarray:
    """Return the public ``(distance, time, rotation)`` resource summary."""

    if isinstance(resource_fractions, Mapping):
        try:
            values = [resource_fractions[key] for key in ("distance", "time", "rotation")]
        except KeyError as exc:  # pragma: no cover - defensive API message
            raise ValueError("resource_fractions mapping needs distance, time, rotation") from exc
    else:
        values = np.asarray(resource_fractions, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != RESOURCE_DIM:
        raise ValueError("resource_fractions must contain distance, time, rotation")
    if not np.isfinite(values).all():
        raise ValueError("resource_fractions must be finite")
    return values


def features(world: Any, belief: Any, actions: Sequence[Any], remaining: int,
             resource_fractions: Sequence[float] | Mapping[str, float]):
    """Build public, masked candidate features for one six-slot decision.

    ``proposal_features`` already includes the public belief summary and the
    action hit/separation descriptors.  The three resource fractions are
    appended as a broadcast row suffix, including on masked rows; the mask is
    what makes ``None`` candidates unavailable to a learner.
    """

    actions = tuple(actions)
    if len(actions) != ACTION_SLOTS:
        raise ValueError(f"actions must contain exactly {ACTION_SLOTS} slots")
    resources = _resource_vector(resource_fractions)
    base, base_mask = _proposal_features(
        world, belief, actions, int(remaining), MAX_FRAMES,
    )
    base = np.asarray(base, dtype=np.float32)
    base_mask = np.asarray(base_mask, dtype=bool).reshape(-1)
    if base.shape != (ACTION_SLOTS, base.shape[1]):
        raise ValueError("frozen proposal_features returned an unexpected slot shape")
    if base_mask.shape != (ACTION_SLOTS,):
        raise ValueError("frozen proposal_features returned an unexpected mask shape")
    # ``broadcast_to`` is read-only, but concatenate creates the owned output
    # expected by the learner and keeps the feature dtype explicitly float32.
    suffix = np.broadcast_to(resources[None, :], (ACTION_SLOTS, RESOURCE_DIM))
    result = np.concatenate((base, suffix), axis=1).astype(np.float32, copy=False)
    mask = base_mask & np.asarray([action is not None for action in actions], dtype=bool)
    return result, mask.astype(bool, copy=False)


def _transition_table(dataset: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Coerce the driver's shared transition fields to learner arrays."""

    required = ("features", "mask", "action", "reward", "next_features", "next_mask", "done")
    if isinstance(dataset, Mapping):
        missing = [name for name in required if name not in dataset]
        if missing:
            raise ValueError(f"dataset is missing transition fields: {missing}")
        rows = {name: np.asarray(dataset[name]) for name in required}
    else:
        records = list(dataset)
        if not records:
            raise ValueError("dataset cannot be empty")
        missing = [name for name in required if name not in records[0]]
        if missing:
            raise ValueError(f"dataset is missing transition fields: {missing}")
        rows = {name: np.asarray([record[name] for record in records]) for name in required}
    # Keep one common transition table for all three fits.  learners.train
    # performs the final shape/mask validation and makes contiguous copies.
    return {
        "features": np.asarray(rows["features"], dtype=np.float32),
        "mask": np.asarray(rows["mask"], dtype=bool),
        "action": np.asarray(rows["action"], dtype=np.int64).reshape(-1),
        "reward": np.asarray(rows["reward"], dtype=np.float32).reshape(-1),
        "next_features": np.asarray(rows["next_features"], dtype=np.float32),
        "next_mask": np.asarray(rows["next_mask"], dtype=bool),
        "done": np.asarray(rows["done"], dtype=bool).reshape(-1),
    }


def fit_models(dataset: Mapping[str, Any] | Sequence[Mapping[str, Any]], out: str | Path,
               *, device: str = "cpu", steps: int = 600):
    """Fit CQL/FQI learners on one shared transition table.

    The suffixes in the returned names are fixed seeds, not fold numbers.
    This makes the three comparisons reproducible while sharing exactly the
    same transitions and update count.
    """

    if int(steps) < 1:
        raise ValueError("steps must be positive")
    # The driver evaluates checkpoints in forked workers.  Keep the tiny
    # learner deterministic and avoid each worker inheriting a large BLAS
    # thread pool.
    if str(device).lower() == "cpu" and hasattr(_learners, "torch"):
        _learners.torch.set_num_threads(1)
    table = _transition_table(dataset)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    models = {}
    manifest_models = {}
    for name, (method, seed) in _MODEL_SPECS.items():
        model = _learners.train(
            table, method, seed=seed, steps=int(steps), device=device,
        )
        path = output / f"{name}.pt"
        _learners.save(model, path)
        models[name] = model
        manifest_models[name] = {
            "method": method,
            "seed": seed,
            "checkpoint": path.name,
        }
    manifest = {
        "version": 1,
        "rows": int(table["reward"].shape[0]),
        "feature_shape": list(table["features"].shape),
        "actions": int(table["features"].shape[1]),
        "steps": int(steps),
        "device": str(device),
        "shared_transition_table": True,
        "models": manifest_models,
    }
    (output / "TRAIN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return models


def values(model: Any, feature_array: np.ndarray, mask: np.ndarray):
    """Return learner scores with illegal candidates set to ``-inf``."""

    x = np.asarray(feature_array, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    single = x.ndim == 2
    if single:
        x = x[None, ...]
        m = m[None, ...]
    if x.ndim != 3 or m.shape != x.shape[:2]:
        raise ValueError("features/mask must have shapes (N,A,D)/(N,A)")
    for module_name in ("q1", "q2", "actor"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.eval()
    scored = np.asarray(_learners.scores(model, x, m), dtype=np.float32)
    scored = np.where(m, scored, -np.inf).astype(np.float32, copy=False)
    return scored[0] if single else scored


def fisher_bonus(world: Any, belief: Any, action: Any) -> float:
    """Return a normalized, continuous task-direction covariance reduction.

    The state is ``[w0, b, z, w_local]``.  Candidate Fisher information is
    weighted by public projected hit rates.  The score compares the posterior
    trace of continuous target-position covariance before/after the candidate
    and is clipped to ``[0, 1]``.  No discrete identity/label information is
    used.
    """

    if action is None:
        return 0.

    # MapFilter's prior is a joint (n_map, n_z, 4) distribution.  Keep the
    # covariance on that actual four-dimensional state; do not reconstruct a
    # phantom local parameter from the frozen three-column knee model.
    reading = belief.predicted_reading(action.t)
    parameters = np.asarray(belief.parameters, dtype=float)
    prior, _ = belief.prior_at(reading)
    prior = np.asarray(prior, dtype=float)
    if parameters.ndim != 3 or parameters.shape[-1] != 4:
        raise ValueError("MapFilter parameters must have shape (n_map,n_z,4)")
    if prior.shape != parameters.shape[:2]:
        raise ValueError("MapFilter prior must match the joint parameter grid")
    if not np.isfinite(parameters).all() or not np.isfinite(prior).all():
        raise ValueError("MapFilter state must be finite")
    prior = np.maximum(prior, 0.)
    prior /= max(float(prior.sum()), np.finfo(float).tiny)
    flat = parameters.reshape(-1, 4)
    probability = prior.reshape(-1)
    mean = np.sum(flat * probability[:, None], axis=0)
    delta = flat - mean
    covariance = (delta * probability[:, None]).T @ delta
    covariance = (covariance + covariance.T) * .5

    _, rows = world.projection(reading.q, action.site, action.view)
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 3 or rows.shape[1:] != (2, 4):
        raise ValueError("MapWorld projection design must have shape (n_lumen,2,4)")
    _, hit = belief.imaging(world, reading, action.site, action.view)
    hit = np.asarray(hit, dtype=float)
    if hit.shape != (parameters.shape[0], parameters.shape[1], rows.shape[0]):
        raise ValueError("projected hit rates must match the joint grid and lumens")
    hit_rate = np.sum(prior[..., None] * hit, axis=(0, 1))
    hit_rate = np.clip(hit_rate, 0., 1.)

    # Whiten each projected centroid coordinate.  The identity term in the
    # update below accounts for unit covariance after whitening.
    sigma = np.array([1.1, .85], dtype=float)
    H = (rows / sigma[None, :, None]) * np.sqrt(hit_rate)[:, None, None]
    H = H.reshape(-1, 4)
    if not H.size:
        return 0.
    innovation_covariance = np.eye(len(H)) + H @ covariance @ H.T
    updated = covariance - covariance @ H.T @ np.linalg.inv(innovation_covariance) @ H @ covariance
    updated = (updated + updated.T) * .5

    _, design = world.world_design(reading.q, world.query_sites)
    task_jacobian = np.asarray(design, dtype=float)
    if task_jacobian.ndim != 3 or task_jacobian.shape[1:] != (3, 4):
        raise ValueError("MapWorld world_design must have shape (n,3,4)")
    before = float(np.sum(np.trace(
        task_jacobian @ covariance @ task_jacobian.transpose(0, 2, 1), axis1=1, axis2=2,
    )))
    after = float(np.sum(np.trace(
        task_jacobian @ updated @ task_jacobian.transpose(0, 2, 1), axis1=1, axis2=2,
    )))
    if not np.isfinite(before) or not np.isfinite(after) or before <= 1.e-12:
        return 0.
    return float(np.clip((before - after) / before, 0., 1.))


__all__ = ["features", "fit_models", "values", "fisher_bonus"]

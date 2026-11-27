"""Bounded paired teacher and a small public-state value student for v7.

The teacher and the student use the same deployment gate.  A non-reference
action may replace the frozen reference only when its strict-delivery delta has
a positive lower posterior bound and its wrong-branch delta has an upper
posterior bound no larger than ``wrong_budget``.  The reference is therefore a
real fallback, rather than an action which is silently removed by a soft value
normalisation step.

This module intentionally has no simulator or rollout dependency.  The public
API consumes paired tape outcomes of the form ``[soft, strict_delta,
wrong_delta]``.  ``teacher_choice`` uses a bounded Jeffreys/Dirichlet sign
posterior.  ``fit`` trains a very small three-channel value head on CUDA using
all positive, negative, and neutral paired samples, then restores the best
layout-disjoint validation epoch.  Calibration layouts are never used for
optimisation and provide residual quantiles for ``decide``.

``--smoke`` is a short reproducible API and CUDA check.  It is deliberately
not a simulator run and writes only to a temporary directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REFERENCE_DEFAULT = "continue"
TRIPLE_NAMES = ("soft", "strict", "wrong")
MIN_PAIRED_TAPES = 3
MIN_CALIBRATION_TAPES = 8
POSTERIOR_DRAWS = 4096
EPS = 1.0e-9


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _triples(values: Any) -> list[tuple[float, float, float]]:
    """Parse finite paired tape rows, ignoring an interrupted tail."""

    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return []
    parsed: list[tuple[float, float, float]] = []
    for row in iterator:
        try:
            if len(row) < 3:  # type: ignore[arg-type]
                continue
            values3 = tuple(_finite(row[index]) for index in range(3))
        except (TypeError, IndexError):
            continue
        if all(value is not None for value in values3):
            parsed.append(tuple(float(value) for value in values3))  # type: ignore[arg-type]
    return parsed


def _bounded_values(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return np.clip(array, -1.0, 1.0)


def _stable_seed(action: str, channel: str) -> int:
    digest = hashlib.sha256(f"v7-dirichlet:{action}:{channel}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _sign_posterior(values: Sequence[float], *, action: str, channel: str) -> dict[str, Any]:
    """Summarise bounded outcomes with a non-degenerate Dirichlet interval.

    The posterior is over the signs ``{-1, 0, +1}``, with Jeffreys
    ``alpha=0.5`` in every category.  This is deliberately a small-count
    Bayesian uncertainty summary, not a frequentist confidence guarantee.
    ``mean`` and ``variance`` still describe the observed bounded deltas, so
    no per-tape outcome is hidden from the report.
    """

    array = _bounded_values(values)
    count = int(array.size)
    positive = int(np.sum(array > 0.0))
    negative = int(np.sum(array < 0.0))
    neutral = int(count - positive - negative)
    if count:
        observed_mean = float(array.mean())
        observed_variance = float(array.var(ddof=1)) if count > 1 else 0.0
        observed_std = float(math.sqrt(max(0.0, observed_variance)))
    else:
        observed_mean = 0.0
        observed_variance = 0.0
        observed_std = 0.0

    # Jeffreys prior, preserving a non-zero probability for every sign even
    # when the observed tape sample is all positive or all neutral.
    alpha = np.asarray([negative + 0.5, neutral + 0.5, positive + 0.5], dtype=np.float64)
    rng = np.random.default_rng(_stable_seed(action, channel))
    draws = rng.dirichlet(alpha, size=POSTERIOR_DRAWS)
    posterior_mean_draws = draws[:, 2] - draws[:, 0]
    interval = np.quantile(posterior_mean_draws, [0.025, 0.975]).astype(np.float64)
    posterior_mean = float(posterior_mean_draws.mean())
    posterior_variance = float(posterior_mean_draws.var())
    return {
        "n": count,
        "mean": observed_mean,
        "variance": observed_variance,
        "std": observed_std,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "posterior_mean": posterior_mean,
        "posterior_variance": posterior_variance,
        "interval": [float(interval[0]), float(interval[1])],
        "ci95": [float(interval[0]), float(interval[1])],
        "interval_type": "Jeffreys-Dirichlet sign posterior; not a frequentist guarantee",
        "bounded_range": [-1.0, 1.0],
        "alpha": [float(value) for value in alpha],
    }


def _estimate(action: str, values: Any) -> dict[str, Any]:
    samples = _triples(values)
    channels = {
        name: _sign_posterior([row[index] for row in samples], action=str(action), channel=name)
        for index, name in enumerate(TRIPLE_NAMES)
    }
    return {
        "action": str(action),
        "n": len(samples),
        "soft": channels["soft"],
        "strict": channels["strict"],
        "wrong": channels["wrong"],
        # Keep the old spelling available to consumers of previous reports.
        "wrong_branch": channels["wrong"],
    }


def _estimates(samples: Mapping[str, Any] | None, reference: str) -> dict[str, dict[str, Any]]:
    values: Mapping[str, Any] = samples if isinstance(samples, Mapping) else {}
    result: dict[str, dict[str, Any]] = {}
    for action, action_values in values.items():
        name = str(action)
        if name not in result:
            result[name] = _estimate(name, action_values)
    if reference not in result:
        result[reference] = _estimate(reference, [])
    return result


def _gate_from_estimate(
    estimate: Mapping[str, Any],
    *,
    wrong_budget: float,
    min_samples: int = MIN_PAIRED_TAPES,
) -> dict[str, Any]:
    strict = estimate["strict"]
    wrong = estimate["wrong"]
    strict_lower = float(strict["interval"][0])
    wrong_upper = float(wrong["interval"][1])
    n = int(estimate["n"])
    enough_samples = n >= int(min_samples)
    strict_ok = strict_lower > 0.0
    wrong_ok = wrong_upper <= float(wrong_budget)
    uncertain = not enough_samples or not (strict_ok and wrong_ok)
    eligible = bool(enough_samples and strict_ok and wrong_ok)
    reasons: list[str] = []
    if not enough_samples:
        reasons.append("insufficient_paired_tapes")
    if not strict_ok:
        reasons.append("strict_lower_not_positive")
    if not wrong_ok:
        reasons.append("wrong_upper_exceeds_budget")
    return {
        "n": n,
        "strict_lower": strict_lower,
        "wrong_upper": wrong_upper,
        "wrong_budget": float(wrong_budget),
        "min_samples": int(min_samples),
        "enough_samples": bool(enough_samples),
        "strict_lower_gt_zero": bool(strict_ok),
        "wrong_upper_leq_budget": bool(wrong_ok),
        "uncertain": bool(uncertain),
        "eligible": eligible,
        "reasons": reasons,
    }


def teacher_choice(
    samples: Mapping[str, Any] | None,
    reference: str,
    *,
    wrong_budget: float = 0.10,
) -> tuple[str, dict[str, Any]]:
    """Choose an action conservatively from paired tapes.

    All deltas are against the same frozen reference tape.  A candidate is
    eligible only under the shared bounded rule.  If every candidate is
    uncertain, the reference is returned.  The diagnostic contains every
    parsed sample count and the posterior interval; it does not describe the
    interval as a frequentist guarantee.
    """

    reference = str(reference)
    wrong_budget = float(wrong_budget)
    if not math.isfinite(wrong_budget) or not 0.0 <= wrong_budget <= 1.0:
        raise ValueError("wrong_budget must be finite and in [0, 1]")
    estimates = _estimates(samples, reference)
    best_name = reference
    best_score = 0.0
    entries: dict[str, Any] = {}
    for action in sorted(estimates):
        estimate = estimates[action]
        if action == reference:
            gate = {
                "n": int(estimate["n"]),
                "strict_lower": 0.0,
                "wrong_upper": 0.0,
                "wrong_budget": wrong_budget,
                "min_samples": MIN_PAIRED_TAPES,
                "enough_samples": True,
                "strict_lower_gt_zero": False,
                "wrong_upper_leq_budget": True,
                "uncertain": False,
                "eligible": True,
                "reasons": ["frozen_reference"],
            }
            entries[action] = {**estimate, "score": 0.0, "gate": gate}
            continue
        gate = _gate_from_estimate(estimate, wrong_budget=wrong_budget)
        strict_lower = float(gate["strict_lower"])
        wrong_upper = float(gate["wrong_upper"])
        soft_mean = float(estimate["soft"]["mean"])
        # Soft utility is a small tie-breaker.  It cannot make an ineligible
        # action pass the strict/wrong gate.
        score = strict_lower - 0.75 * max(0.0, wrong_upper) + 0.05 * float(np.clip(soft_mean, -1.0, 1.0))
        entries[action] = {**estimate, "score": float(score), "gate": gate}
        if gate["eligible"] and score > best_score + EPS:
            best_name = action
            best_score = float(score)
    diagnostics = {
        "reference": reference,
        "selected": best_name,
        "estimates": entries,
        "all_candidates_passed_gate": bool(best_name != reference),
        "gate": {
            "strict_condition": "posterior lower bound > 0",
            "wrong_condition": "posterior upper bound <= wrong_budget",
            "wrong_budget": wrong_budget,
            "minimum_paired_tapes": MIN_PAIRED_TAPES,
            "uncertain_falls_back_to_reference": True,
            "interval_interpretation": "bounded Jeffreys-Dirichlet posterior, not a frequentist guarantee",
        },
        "paired_against_reference": True,
    }
    return best_name, diagnostics


def uncertain_actions(samples: Mapping[str, Any] | None, reference: str) -> list[str]:
    """Return reference plus at most two actions worth extra paired tapes."""

    reference = str(reference)
    estimates = _estimates(samples, reference)
    candidates: list[tuple[float, str]] = []
    for action, estimate in estimates.items():
        if action == reference:
            continue
        gate = _gate_from_estimate(estimate, wrong_budget=0.10)
        strict = estimate["strict"]
        wrong = estimate["wrong"]
        strict_interval = strict["interval"]
        wrong_interval = wrong["interval"]
        # Do not spend new tapes on an action already decisively harmful under
        # both channels.  Everything else is retained as a possible teaching
        # comparison, including actions with no samples yet.
        clear_bad = (
            int(estimate["n"]) >= MIN_PAIRED_TAPES
            and float(strict_interval[1]) <= 0.0
            and float(wrong_interval[0]) > 0.10
        )
        if clear_bad:
            continue
        width = float(strict_interval[1] - strict_interval[0])
        width += float(wrong_interval[1] - wrong_interval[0])
        plausible = max(0.0, float(strict_interval[1]))
        if gate["uncertain"]:
            # Unknown actions receive a deterministic high priority, but the
            # action name remains the final tie-breaker for reproducibility.
            priority = 2.0 + width + plausible
        else:
            priority = width + plausible
        candidates.append((priority, action))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [reference] + [action for _, action in candidates[:2]]


def _row_samples(row: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("samples", "paired_samples"):
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _layout(row: Mapping[str, Any], index: int) -> str:
    value = row.get("layout", f"__missing_layout_{index}")
    return str(value)


def _feature_matrix(rows: Sequence[Mapping[str, Any]], fit_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    for index, row in enumerate(rows):
        value = row.get("features")
        if value is None:
            raise ValueError(f"row {index} is missing public features")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(f"row {index} has empty or non-finite public features")
        vectors.append(vector)
    dimension = int(vectors[0].size)
    if any(vector.size != dimension for vector in vectors):
        raise ValueError("all rows must have the same feature dimension")
    fit_values = np.asarray([vectors[index] for index in fit_indices], dtype=np.float64)
    mean = fit_values.mean(axis=0).astype(np.float32)
    scale = fit_values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1.0e-6, scale, 1.0).astype(np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    return (matrix - mean) / scale, mean, scale


def _split_layouts(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[set[str], set[str], set[str]]:
    names = sorted({_layout(row, index) for index, row in enumerate(rows)})
    rng = np.random.default_rng(int(seed))
    rng.shuffle(names)
    count = len(names)
    if count < 2:
        return set(names), set(), set()
    if count == 2:
        return {names[0]}, {names[1]}, set()
    validation_count = max(1, int(round(0.20 * count)))
    calibration_count = max(1, int(round(0.20 * count)))
    while count - validation_count - calibration_count < 1:
        if validation_count > 1:
            validation_count -= 1
        elif calibration_count > 1:
            calibration_count -= 1
        else:
            break
    train_end = count - validation_count - calibration_count
    train = set(names[:train_end])
    validation = set(names[train_end : train_end + validation_count])
    calibration = set(names[train_end + validation_count :])
    return train, validation, calibration


def _make_examples(
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    action_names: Sequence[str],
    reference: str,
    feature_matrix: np.ndarray,
) -> dict[str, Any]:
    features: list[np.ndarray] = []
    action_ids: list[int] = []
    targets: list[tuple[float, float, float]] = []
    row_ids: list[int] = []
    counts: dict[str, dict[str, int]] = {
        action: {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
        for action in action_names
    }
    action_to_id = {action: index for index, action in enumerate(action_names)}
    for row_index in row_indices:
        sample_map = _row_samples(rows[row_index])
        for action in action_names:
            if action == reference:
                raw = _triples(sample_map.get(action, []))
                # The reference delta is exactly zero.  Keep one example per
                # reference tape where present and one fallback example when
                # no explicit reference tape was persisted.
                tape_count = max(1, len(raw))
                values = [(0.0, 0.0, 0.0)] * tape_count
            else:
                values = _triples(sample_map.get(action, []))
            if not values:
                continue
            for triple in values:
                bounded = tuple(float(np.clip(value, -1.0, 1.0)) for value in triple)
                features.append(feature_matrix[row_index])
                action_ids.append(action_to_id[action])
                targets.append(bounded)  # preserve every sign, including zero
                row_ids.append(row_index)
                summary = counts[action]
                summary["total"] += 1
                for value in bounded:
                    if value > 0.0:
                        summary["positive"] += 1
                    elif value < 0.0:
                        summary["negative"] += 1
                    else:
                        summary["neutral"] += 1
    if features:
        feature_array = np.asarray(features, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
    else:
        feature_array = np.empty((0, feature_matrix.shape[1]), dtype=np.float32)
        target_array = np.empty((0, 3), dtype=np.float32)
    return {
        "features": feature_array,
        "action_ids": np.asarray(action_ids, dtype=np.int64),
        "targets": target_array,
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "counts": counts,
    }


def _merge_counts(*maps: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for mapping in maps:
        for action, values in mapping.items():
            output = result.setdefault(action, {"total": 0, "positive": 0, "negative": 0, "neutral": 0})
            for key in output:
                output[key] += int(values.get(key, 0))
    return result


def _choose_hidden(input_dim: int, output_dim: int) -> tuple[int, int]:
    for hidden in (32, 24, 16, 12, 8, 4):
        parameters = (input_dim + 1) * hidden + (hidden + 1) * output_dim
        if parameters <= 10000:
            return hidden, int(parameters)
    raise ValueError(f"public feature dimension {input_dim} cannot fit a <=10,000 parameter student")


def _cpu_state_dict(torch: Any, state: Mapping[str, Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for key, value in state.items():
        if hasattr(value, "detach"):
            result[str(key)] = value.detach().cpu().numpy().tolist()
        else:
            result[str(key)] = np.asarray(value).tolist()
    return result


def _cpu_optimizer_state(torch: Any, value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_optimizer_state(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_optimizer_state(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_optimizer_state(torch, item) for item in value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_torch_save(torch: Any, value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _output_paths(out: str | os.PathLike[str]) -> tuple[Path, Path]:
    requested = Path(out)
    if requested.suffix.lower() == ".pt":
        return requested, requested.with_suffix(".json")
    if requested.suffix.lower() == ".json":
        return requested.with_suffix(".pt"), requested
    requested.mkdir(parents=True, exist_ok=True)
    return requested / "student_value_v7.pt", requested / "student_value_v7.json"


def _calibration_residuals(
    rows: Sequence[Mapping[str, Any]],
    row_indices: Sequence[int],
    action_names: Sequence[str],
    reference: str,
    predictions: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    action_to_id = {action: index for index, action in enumerate(action_names)}
    for action in action_names:
        if action == reference:
            continue
        residuals: list[list[float]] = [[], [], []]
        for row_index in row_indices:
            values = _triples(_row_samples(rows[row_index]).get(action, []))
            if not values:
                continue
            prediction = predictions[row_index, action_to_id[action], :]
            for triple in values:
                for channel, value in enumerate(triple):
                    residuals[channel].append(float(np.clip(value, -1.0, 1.0) - prediction[channel]))
        channel_result: dict[str, Any] = {}
        for channel, name in enumerate(TRIPLE_NAMES):
            array = np.asarray(residuals[channel], dtype=np.float64)
            array = array[np.isfinite(array)]
            count = int(array.size)
            if count >= MIN_CALIBRATION_TAPES:
                q05, q50, q95 = np.quantile(array, [0.05, 0.50, 0.95]).tolist()
                sufficient = True
            else:
                q05, q50, q95 = -1.0, 0.0, 1.0
                sufficient = False
            channel_result[name] = {
                "n": count,
                "q05": float(q05),
                "q50": float(q50),
                "q95": float(q95),
                "std": float(array.std(ddof=1)) if count > 1 else None,
                "sufficient": bool(sufficient),
                "fallback": "bounded_worst_case" if not sufficient else None,
            }
        result[action] = channel_result
    return result


def fit(
    rows: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    reference: str,
    out: str | os.PathLike[str],
    *,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the tiny public-state value model on actual CUDA.

    Layouts, not tape rows, define the fit/validation/calibration split.  Every
    valid paired tape becomes a masked training example, so positive, negative,
    and neutral outcomes are all retained.  The returned bundle contains only
    Python lists and scalars and is therefore JSON-friendly for CPU workers.
    """

    if not rows:
        raise ValueError("fit requires at least one row")
    action_names: list[str] = []
    for action in actions:
        name = str(action)
        if name not in action_names:
            action_names.append(name)
    reference = str(reference)
    if reference not in action_names:
        action_names.insert(0, reference)
    train_layouts, validation_layouts, calibration_layouts = _split_layouts(rows, int(seed))
    train_indices = [index for index, row in enumerate(rows) if _layout(row, index) in train_layouts]
    validation_indices = [index for index, row in enumerate(rows) if _layout(row, index) in validation_layouts]
    calibration_indices = [index for index, row in enumerate(rows) if _layout(row, index) in calibration_layouts]
    if not train_indices:
        raise ValueError("layout split produced no fit rows")
    feature_matrix, feature_mean, feature_scale = _feature_matrix(rows, train_indices)
    train_examples = _make_examples(rows, train_indices, action_names, reference, feature_matrix)
    validation_examples = _make_examples(rows, validation_indices, action_names, reference, feature_matrix)
    if train_examples["features"].shape[0] == 0:
        raise ValueError("rows contain no valid paired samples")

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("fit requires PyTorch with CUDA support") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("fit requires an available CUDA device")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    batch_rng = np.random.default_rng(int(seed) + 9176)
    input_dim = int(feature_matrix.shape[1])
    output_dim = int(3 * len(action_names))
    hidden_dim, parameter_count = _choose_hidden(input_dim, output_dim)
    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, output_dim),
    ).to(device)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)

    train_targets = train_examples["targets"]
    target_center = train_targets.mean(axis=0).astype(np.float32)
    target_scale = train_targets.std(axis=0).astype(np.float32)
    target_scale = np.where(target_scale > 1.0e-3, target_scale, 1.0).astype(np.float32)
    train_target_z = (train_targets - target_center.reshape(1, 3)) / target_scale.reshape(1, 3)
    validation_target_z = (
        (validation_examples["targets"] - target_center.reshape(1, 3)) / target_scale.reshape(1, 3)
        if validation_examples["targets"].size
        else validation_examples["targets"]
    )
    train_features = torch.as_tensor(train_examples["features"], dtype=torch.float32, device=device)
    train_actions = torch.as_tensor(train_examples["action_ids"], dtype=torch.long, device=device)
    train_targets_tensor = torch.as_tensor(train_target_z, dtype=torch.float32, device=device)
    val_features = torch.as_tensor(validation_examples["features"], dtype=torch.float32, device=device)
    val_actions = torch.as_tensor(validation_examples["action_ids"], dtype=torch.long, device=device)
    val_targets_tensor = torch.as_tensor(validation_target_z, dtype=torch.float32, device=device)
    channel_weight = torch.tensor([0.25, 1.0, 1.0], dtype=torch.float32, device=device)

    epochs = 60 if train_features.shape[0] < 2048 else 32
    batch_size = min(64, max(8, int(train_features.shape[0])))
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_state: dict[str, Any] | None = None
    best_optimizer_state: Any = None
    optimizer_steps = 0

    def evaluate(features: Any, action_ids: Any, targets: Any) -> float:
        if int(features.shape[0]) == 0:
            return float("nan")
        model.eval()
        with torch.no_grad():
            all_prediction = model(features).reshape(-1, len(action_names), 3)
            selected = all_prediction[torch.arange(features.shape[0], device=device), action_ids, :]
            values = F.smooth_l1_loss(selected, targets, reduction="none") * channel_weight.reshape(1, 3)
            return float(values.sum().item() / max(float(features.shape[0]), 1.0))

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = batch_rng.permutation(int(train_features.shape[0]))
        running = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
            prediction = model(train_features.index_select(0, index_tensor)).reshape(-1, len(action_names), 3)
            selected = prediction[
                torch.arange(index_tensor.shape[0], device=device),
                train_actions.index_select(0, index_tensor),
                :,
            ]
            target_batch = train_targets_tensor.index_select(0, index_tensor)
            values = F.smooth_l1_loss(selected, target_batch, reduction="none") * channel_weight.reshape(1, 3)
            loss = values.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            running += float(loss.detach().item()) * int(index_tensor.shape[0])
            seen += int(index_tensor.shape[0])
        train_losses.append(float(running / max(seen, 1)))
        val_loss = evaluate(val_features, val_actions, val_targets_tensor)
        if not math.isfinite(val_loss):
            val_loss = train_losses[-1]
        validation_losses.append(float(val_loss))
        if val_loss < best_validation_loss - 1.0e-12:
            best_validation_loss = float(val_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer_state = _cpu_optimizer_state(torch, copy.deepcopy(optimizer.state_dict()))

    if best_state is None:
        raise RuntimeError("student fit did not produce a validation checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    torch.cuda.synchronize()
    changed = any(not torch.equal(parameter.detach(), original) for parameter, original in zip(model.parameters(), initial_parameters))
    cuda_optimizer_step_verified = bool(changed and optimizer_steps > 0 and all(parameter.is_cuda for parameter in model.parameters()))

    # Compute predictions for all rows with the restored best epoch.  These
    # feed calibration residuals, never optimisation.
    all_features = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        all_predictions = model(all_features).reshape(-1, len(action_names), 3).detach().cpu().numpy()
    calibration = _calibration_residuals(rows, calibration_indices, action_names, reference, all_predictions)

    checkpoint, manifest_path = _output_paths(out)
    checkpoint_payload = {
        "format": "v7_bounded_paired_value",
        "version": 1,
        "architecture": {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim},
        "actions": action_names,
        "reference": reference,
        "state_dict": best_state,
        "optimizer_state": best_optimizer_state,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_center": target_center.tolist(),
        "target_scale": target_scale.tolist(),
        "calibration": calibration,
        "seed": int(seed),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }

    split = {
        "fit_layouts": sorted(train_layouts),
        "validation_layouts": sorted(validation_layouts),
        "calibration_layouts": sorted(calibration_layouts),
        "layout_disjoint": not (
            train_layouts & validation_layouts
            or train_layouts & calibration_layouts
            or validation_layouts & calibration_layouts
        ),
    }
    sample_counts = {
        "fit": train_examples["counts"],
        "validation": validation_examples["counts"],
        "calibration": _merge_counts(
            _make_examples(rows, calibration_indices, action_names, reference, feature_matrix)["counts"]
        ),
    }
    manifest: dict[str, Any] = {
        "format": "v7_bounded_paired_value_manifest",
        "version": 1,
        "actions": action_names,
        "reference": reference,
        "wrong_budget": 0.10,
        "architecture": {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim},
        "parameter_count": parameter_count,
        "rows": len(rows),
        "fit_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "calibration_rows": len(calibration_indices),
        "split": split,
        "sample_counts": sample_counts,
        "all_positive_negative_neutral_samples_preserved": True,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_checkpoint_restored": True,
        "best_validation_loss": float(best_validation_loss),
        "final_validation_loss": float(validation_losses[-1]),
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_device": str(torch.cuda.get_device_name(device)),
        "cuda_optimizer_step_verified": cuda_optimizer_step_verified,
        "optimizer": "AdamW",
        "optimizer_steps": int(optimizer_steps),
        "target_fields": list(TRIPLE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_center": target_center.tolist(),
        "target_scale": target_scale.tolist(),
        "calibration": calibration,
        "calibration_min_samples": MIN_CALIBRATION_TAPES,
        "calibration_fallback": "bounded_worst_case_and_reference",
        "checkpoint": str(checkpoint),
    }
    _atomic_torch_save(torch, checkpoint_payload, checkpoint)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    _atomic_json(manifest_path, manifest)

    bundle: dict[str, Any] = {
        "format": "v7_bounded_paired_value_bundle",
        "version": 1,
        "actions": action_names,
        "reference": reference,
        "wrong_budget": 0.10,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "parameter_count": parameter_count,
        "state_dict": _cpu_state_dict(torch, model.state_dict()),
        "feature_mean": feature_mean.tolist(),
        "feature_scale": feature_scale.tolist(),
        "target_center": target_center.tolist(),
        "target_scale": target_scale.tolist(),
        "calibration": calibration,
        "calibration_min_samples": MIN_CALIBRATION_TAPES,
        "best_epoch": best_epoch,
        "best_validation_checkpoint_restored": True,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
    }
    return bundle, manifest


def _bundle_array(bundle: Mapping[str, Any], key: str) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        raise ValueError(f"bundle is missing {key}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _student_scores(features: Sequence[float], bundle: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    actions = [str(action) for action in bundle.get("actions", [])]
    if not actions:
        raise ValueError("bundle has no actions")
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    input_dim = int(bundle.get("input_dim", vector.size))
    if vector.size != input_dim:
        raise ValueError(f"expected {input_dim} public features, got {vector.size}")
    mean = _bundle_array(bundle, "feature_mean")
    scale = _bundle_array(bundle, "feature_scale")
    vector = (vector - mean) / np.where(scale > 1.0e-6, scale, 1.0)
    state = bundle.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("bundle has invalid state_dict")
    first_weight = np.asarray(state["0.weight"], dtype=np.float32)
    first_bias = np.asarray(state["0.bias"], dtype=np.float32)
    second_weight = np.asarray(state["2.weight"], dtype=np.float32)
    second_bias = np.asarray(state["2.bias"], dtype=np.float32)
    hidden = np.tanh(first_weight @ vector + first_bias)
    normalized = second_weight @ hidden + second_bias
    center = _bundle_array(bundle, "target_center")
    target_scale = _bundle_array(bundle, "target_scale")
    values = normalized.reshape(len(actions), 3) * target_scale.reshape(1, 3) + center.reshape(1, 3)
    result: dict[str, dict[str, float]] = {}
    reference = str(bundle.get("reference", actions[0]))
    for index, action in enumerate(actions):
        if action == reference:
            row = np.zeros(3, dtype=np.float32)
        else:
            row = np.clip(values[index], -1.0, 1.0)
        result[action] = {name: float(row[channel]) for channel, name in enumerate(TRIPLE_NAMES)}
    return result


def decide(
    features: Sequence[float],
    bundle: Mapping[str, Any],
    reference: str,
) -> tuple[str, dict[str, Any]]:
    """Apply the same conservative gate as ``teacher_choice`` to the student."""

    reference = str(reference)
    scores = _student_scores(features, bundle)
    if reference not in scores:
        raise ValueError(f"reference action {reference!r} is not in bundle actions")
    wrong_budget = float(bundle.get("wrong_budget", 0.10))
    calibration = bundle.get("calibration", {})
    selected = reference
    best_score = 0.0
    entries: dict[str, Any] = {}
    for action in sorted(scores):
        values = scores[action]
        if action == reference:
            entries[action] = {
                "predicted": values,
                "strict_lower": 0.0,
                "wrong_upper": 0.0,
                "gate": {"eligible": True, "uncertain": False, "reason": "frozen_reference"},
                "score": 0.0,
            }
            continue
        action_calibration = calibration.get(action, {}) if isinstance(calibration, Mapping) else {}
        strict_cal = action_calibration.get("strict", {}) if isinstance(action_calibration, Mapping) else {}
        wrong_cal = action_calibration.get("wrong", {}) if isinstance(action_calibration, Mapping) else {}
        strict_sufficient = bool(strict_cal.get("sufficient", False))
        wrong_sufficient = bool(wrong_cal.get("sufficient", False))
        if strict_sufficient:
            strict_lower = float(np.clip(values["strict"] + float(strict_cal.get("q05", -1.0)), -1.0, 1.0))
        else:
            strict_lower = -1.0
        if wrong_sufficient:
            wrong_upper = float(np.clip(values["wrong"] + float(wrong_cal.get("q95", 1.0)), -1.0, 1.0))
        else:
            wrong_upper = 1.0
        enough = bool(strict_sufficient and wrong_sufficient)
        strict_ok = bool(strict_lower > 0.0)
        wrong_ok = bool(wrong_upper <= wrong_budget)
        eligible = bool(enough and strict_ok and wrong_ok)
        reasons: list[str] = []
        if not enough:
            reasons.append("insufficient_calibration")
        if not strict_ok:
            reasons.append("strict_lower_not_positive")
        if not wrong_ok:
            reasons.append("wrong_upper_exceeds_budget")
        score = strict_lower - 0.75 * max(0.0, wrong_upper) + 0.05 * float(np.clip(values["soft"], -1.0, 1.0))
        entries[action] = {
            "predicted": values,
            "strict_lower": strict_lower,
            "wrong_upper": wrong_upper,
            "calibration": {"strict": strict_cal, "wrong": wrong_cal},
            "gate": {
                "eligible": eligible,
                "uncertain": not enough,
                "enough_calibration": enough,
                "strict_lower_gt_zero": strict_ok,
                "wrong_upper_leq_budget": wrong_ok,
                "wrong_budget": wrong_budget,
                "reasons": reasons,
            },
            "score": float(score),
        }
        if eligible and score > best_score + EPS:
            selected = action
            best_score = float(score)
    diagnostics = {
        "reference": reference,
        "selected": selected,
        "estimates": entries,
        "gate": {
            "strict_condition": "calibrated residual lower bound > 0",
            "wrong_condition": "calibrated residual upper bound <= wrong_budget",
            "wrong_budget": wrong_budget,
            "calibration_min_samples": int(bundle.get("calibration_min_samples", MIN_CALIBRATION_TAPES)),
            "insufficient_calibration_falls_back_to_reference": True,
        },
        "public_features_only": True,
    }
    return selected, diagnostics


def student_scores(features: Sequence[float], bundle: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Public CPU-only score readout for diagnostics and rollout workers."""

    return _student_scores(features, bundle)


def student_action(
    features: Sequence[float],
    bundle: Mapping[str, Any],
    reference: str,
) -> str:
    """Compatibility wrapper returning only the gated action name."""

    return decide(features, bundle, reference)[0]


# Compatibility names make migration from v6 explicit without changing the v7
# gate.  They are deliberately thin aliases, not alternate decision logic.
select_teacher = teacher_choice
recheck = uncertain_actions


def _smoke() -> dict[str, Any]:
    """Run a short deterministic CPU estimator and real CUDA toy fit."""

    # The bounded posterior is intentionally conservative at small n.  Thirty
    # two identical paired tapes are enough for this synthetic positive action
    # to clear both the strict and wrong-risk gates, while the window action
    # remains ineligible.
    samples = {
        "continue": [[0.0, 0.0, 0.0]] * 32,
        "anchor": [[0.02, 0.10, 0.00]] * 32,
        "window": [[-0.01, -0.10, 0.10]] * 32,
    }
    teacher_name, teacher_diag = teacher_choice(samples, "continue")
    shortlist = uncertain_actions(samples, "continue")
    if teacher_name != "anchor":
        raise AssertionError(f"bounded teacher smoke selected {teacher_name!r}, expected anchor")
    if shortlist[0] != "continue" or len(shortlist) > 3:
        raise AssertionError("uncertain_actions contract failed")

    rows: list[dict[str, Any]] = []
    for index in range(15):
        rows.append(
            {
                "features": [float(index), float(index % 3), float((index * 7) % 5)],
                "layout": f"layout-{index}",
                "samples": {
                    "continue": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "anchor": [
                        [0.02 + 0.001 * index, 0.04, 0.00],
                        [-0.01, 0.02, 0.01],
                        [0.00, 0.03, 0.00],
                    ],
                    "window": [[-0.01, -0.02, 0.05], [0.00, 0.01, 0.03]],
                },
            }
        )
    with tempfile.TemporaryDirectory(prefix="v7-value-smoke-") as directory:
        bundle, manifest = fit(rows, ["continue", "anchor", "window"], "continue", directory, seed=19)
        selected, student_diag = decide(rows[0]["features"], bundle, "continue")
        if not manifest["cuda_optimizer_step_verified"]:
            raise AssertionError("CUDA optimizer update was not verified")
        if int(manifest["best_epoch"]) > len(manifest["validation_losses"]):
            raise AssertionError("best epoch is outside recorded validation history")
        if not manifest["split"]["layout_disjoint"]:
            raise AssertionError("layout split is not disjoint")
        if not manifest["all_positive_negative_neutral_samples_preserved"]:
            raise AssertionError("sample preservation manifest missing")
    return {
        "teacher": teacher_name,
        "teacher_interval_type": teacher_diag["gate"]["interval_interpretation"],
        "shortlist": shortlist,
        "student_action": selected,
        "student_entries": sorted(student_diag["estimates"]),
        "best_epoch": int(manifest["best_epoch"]),
        "parameter_count": int(manifest["parameter_count"]),
        "cuda_optimizer_step_verified": bool(manifest["cuda_optimizer_step_verified"]),
        "layout_disjoint": bool(manifest["split"]["layout_disjoint"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v7 bounded paired teacher and CUDA value smoke")
    parser.add_argument("--smoke", action="store_true", help="run a short CPU estimator plus CUDA toy fit")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; this module is an importable API, not a rollout entry point")
    print(json.dumps(_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()

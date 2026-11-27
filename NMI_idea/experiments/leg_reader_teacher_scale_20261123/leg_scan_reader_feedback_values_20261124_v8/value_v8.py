"""Expected action-gain value model and bounded teacher allocator for v8.

The public target for each layout/action is the mean of its bounded paired
outcomes [soft_delta, strict_delta, wrong_delta]. Individual tape rows are
never used as calibration groups: calibration and acceptance uncertainty is
computed over held-out layout means with a deterministic layout bootstrap.
This keeps a layout with many noisy tapes from dominating a cross-layout
claim.

The teacher API is intentionally separate from the learned value model. It
uses local paired tapes to decide which actions merit more replication, while
the fitted model is split by layout and calibrated on disjoint layouts for
deployment. A missing or under-supported action falls back to the frozen
reference rather than being treated as a measured negative.
"""

from __future__ import annotations

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
MIN_INITIAL_REPLICATES = 4
MIN_INTERMEDIATE_REPLICATES = 8
MAX_PAIRED_REPLICATES = 32
MIN_CALIBRATION_LAYOUTS = 8
BOOTSTRAP_DRAWS = 4096
DEFAULT_WRONG_BUDGET = 0.10
EPS = 1.0e-9


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _triples(values: Any) -> list[tuple[float, float, float]]:
    if values is None:
        return []
    try:
        iterator = iter(values)
    except TypeError:
        return []
    result: list[tuple[float, float, float]] = []
    for row in iterator:
        try:
            if len(row) < 3:
                continue
            parsed = tuple(_finite(row[index]) for index in range(3))
        except (TypeError, IndexError):
            continue
        if all(value is not None for value in parsed):
            result.append(tuple(float(value) for value in parsed))
    return result


def _bounded(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    return np.clip(array, -1.0, 1.0)


def _stable_seed(*parts: Any) -> int:
    text = "v8-expected-value:" + ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _layout(row: Mapping[str, Any], index: int) -> str:
    value = row.get("layout", f"__missing_layout_{index}")
    return str(value)


def _samples(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("samples", row.get("paired_samples", {}))
    return value if isinstance(value, Mapping) else {}


def _action_names(actions: Sequence[str], reference: str) -> list[str]:
    result: list[str] = []
    for action in [reference, *actions]:
        name = str(action)
        if name not in result:
            result.append(name)
    return result


def _bootstrap_mean(values: Sequence[float], *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict[str, Any]:
    """Bounded finite-count interval for one local mean.

    The local teacher uses a Jeffreys Dirichlet posterior over negative,
    neutral, and positive signs.  It therefore does not call four or six
    all-zero tapes a zero-risk action.  A small bounded amplitude floor keeps
    that finite-count uncertainty visible while allowing a 32-tape zero-harm
    action to clear a 0.10 wrong-branch budget.
    """
    array = _bounded(values)
    n = int(array.size)
    if not n:
        return {
            "n": 0, "mean": 0.0, "q05": -1.0, "q50": 0.0, "q95": 1.0,
            "std": None, "sufficient": False, "fallback": "bounded_worst_case",
            "unit": "paired_replicate",
        }
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    negative = int(np.count_nonzero(array < 0.))
    neutral = int(np.count_nonzero(array == 0.))
    positive = int(np.count_nonzero(array > 0.))
    alpha = np.asarray([negative + .5, neutral + .5, positive + .5], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws_sign = rng.dirichlet(alpha, size=int(draws))
    sign_mean = draws_sign[:, 2] - draws_sign[:, 0]
    nonzero = np.abs(array[array != 0.])
    amplitude = float(nonzero.mean()) if nonzero.size else 0.0
    # At small n, zero observations still carry a bounded non-zero risk.  The
    # 1.25/sqrt(n) floor makes n=4/6 conservative, while n=32 remains below a
    # 0.10 wrong-branch budget when all wrong deltas are zero.
    amplitude = max(amplitude, 1.25 / math.sqrt(float(n)))
    posterior = np.clip(sign_mean * amplitude, -1., 1.)
    q05, q50, q95 = np.quantile(posterior, [0.05, 0.50, 0.95]).tolist()
    return {
        "n": n, "mean": mean, "q05": float(q05), "q50": float(q50), "q95": float(q95),
        "std": std, "sufficient": bool(n >= MIN_INITIAL_REPLICATES),
        "fallback": None if n >= MIN_INITIAL_REPLICATES else "minimum_paired_replicates",
        "unit": "paired_replicate", "interval_method": "Jeffreys-Dirichlet sign posterior",
        "positive": positive, "negative": negative, "neutral": neutral,
        "alpha": alpha.tolist(), "amplitude": amplitude,
    }


def _action_estimate(action: str, values: Any, *, seed: int) -> dict[str, Any]:
    triples = _triples(values)
    channels: dict[str, dict[str, Any]] = {}
    for channel, name in enumerate(TRIPLE_NAMES):
        channels[name] = _bootstrap_mean(
            [row[channel] for row in triples],
            seed=_stable_seed(seed, action, name),
        )
    return {
        "action": str(action), "n": len(triples),
        "soft": channels["soft"], "strict": channels["strict"], "wrong": channels["wrong"],
        "wrong_branch": channels["wrong"],
    }


def _gate(estimate: Mapping[str, Any], *, wrong_budget: float = DEFAULT_WRONG_BUDGET) -> dict[str, Any]:
    n = int(estimate["n"])
    strict = estimate["strict"]
    wrong = estimate["wrong"]
    enough = n >= MIN_INITIAL_REPLICATES
    strict_lower = float(strict["q05"]) if enough else -1.0
    wrong_upper = float(wrong["q95"]) if enough else 1.0
    strict_ok = strict_lower > 0.0
    wrong_ok = wrong_upper <= float(wrong_budget)
    reasons: list[str] = []
    if not enough:
        reasons.append("insufficient_initial_paired_replicates")
    if not strict_ok:
        reasons.append("expected_strict_lower_not_positive")
    if not wrong_ok:
        reasons.append("expected_wrong_upper_exceeds_budget")
    return {
        "n": n, "min_replicates": MIN_INITIAL_REPLICATES,
        "enough_replicates": enough, "strict_lower": strict_lower,
        "wrong_upper": wrong_upper, "wrong_budget": float(wrong_budget),
        "strict_lower_gt_zero": strict_ok, "wrong_upper_leq_budget": wrong_ok,
        "eligible": bool(enough and strict_ok and wrong_ok),
        "uncertain": bool(not enough or not (strict_ok and wrong_ok)),
        "structurally_impossible": False,
        "reasons": reasons,
    }


def _all_estimates(samples: Mapping[str, Any] | None, reference: str) -> dict[str, dict[str, Any]]:
    values = samples if isinstance(samples, Mapping) else {}
    names = _action_names([str(key) for key in values], reference)
    return {
        action: _action_estimate(
            action,
            values.get(action, []),
            seed=_stable_seed("local", action),
        )
        for action in names
    }


def teacher_summary(
    samples: Mapping[str, Any] | None,
    reference: str,
) -> tuple[str, dict[str, Any]]:
    """Select a locally promising action from bounded paired mean outcomes."""
    reference = str(reference)
    estimates = _all_estimates(samples, reference)
    selected = reference
    best_score = 0.0
    entries: dict[str, Any] = {}
    for action, estimate in sorted(estimates.items()):
        if action == reference:
            entries[action] = {
                **estimate, "score": 0.0,
                "gate": {
                    "eligible": True, "uncertain": False,
                    "reasons": ["frozen_reference"], "n": int(estimate["n"]),
                },
            }
            continue
        gate = _gate(estimate)
        strict_lower = float(gate["strict_lower"])
        wrong_upper = float(gate["wrong_upper"])
        score = strict_lower - .75 * max(0.0, wrong_upper)
        score += .05 * float(np.clip(estimate["soft"]["mean"], -1.0, 1.0))
        entries[action] = {**estimate, "score": float(score), "gate": gate}
        if gate["eligible"] and score > best_score + EPS:
            selected = action
            best_score = score
    diagnostics = {
        "reference": reference, "selected": selected, "estimates": entries,
        "paired_against_reference": True,
        "comparison_scope": "local_adaptive_paired_comparison",
        "final_cross_layout_acceptance": False,
        "bounded_mean_comparison": True,
        "uncertain_falls_back_to_reference": True,
        "support_policy": {
            "initial_replicates": MIN_INITIAL_REPLICATES,
            "intermediate_replicates": MIN_INTERMEDIATE_REPLICATES,
            "maximum_replicates": MAX_PAIRED_REPLICATES,
            "interval": "bootstrap over paired replicate means; bounded in [-1,1]",
            "wrong_budget": DEFAULT_WRONG_BUDGET,
        },
    }
    return selected, diagnostics


def _remaining_for(remaining: Any, action: str) -> int | None:
    if remaining is None:
        return None
    if isinstance(remaining, Mapping):
        value = remaining.get(action, remaining.get(str(action), 0))
    else:
        value = remaining
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def promising(
    samples: Mapping[str, Any] | None,
    reference: str,
    remaining: Any,
) -> list[str]:
    """Return reference first, then actions worth paired replication.

    Unknown actions are not declared impossible. They receive an initial
    four-tape allocation, then eight, and finally up to 32 when the bounded
    mean gate remains uncertain. A clearly harmful action can be deprioritized
    only after it has enough evidence; no structural impossibility is inferred
    from missing tapes.
    """
    reference = str(reference)
    values = samples if isinstance(samples, Mapping) else {}
    extra_actions = list(remaining.keys()) if isinstance(remaining, Mapping) else []
    names = _action_names([str(key) for key in [*values, *extra_actions]], reference)
    ranked: list[tuple[float, str]] = []
    for action in names:
        if action == reference:
            continue
        estimate = _action_estimate(action, values.get(action, []), seed=_stable_seed("local", action))
        n = int(estimate["n"])
        available = _remaining_for(remaining, action)
        if available is not None and available <= 0:
            continue
        strict = estimate["strict"]
        wrong = estimate["wrong"]
        clear_bad = (
            n >= MIN_INTERMEDIATE_REPLICATES
            and float(strict["q95"]) <= 0.0
            and float(wrong["q05"]) > DEFAULT_WRONG_BUDGET
        )
        if clear_bad and n >= MAX_PAIRED_REPLICATES:
            continue
        if n < MIN_INITIAL_REPLICATES:
            priority = 4.0 + (MIN_INITIAL_REPLICATES - n) / MIN_INITIAL_REPLICATES
        elif n < MIN_INTERMEDIATE_REPLICATES:
            priority = 3.0 + (MIN_INTERMEDIATE_REPLICATES - n) / MIN_INTERMEDIATE_REPLICATES
        elif n < MAX_PAIRED_REPLICATES:
            width = float(strict["q95"] - strict["q05"] + wrong["q95"] - wrong["q05"])
            plausible = max(0.0, float(strict["q95"]))
            priority = 1.0 + width + plausible
            if clear_bad:
                priority *= .25
        else:
            priority = 0.0
        if priority > 0.0:
            ranked.append((priority, action))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [reference] + [action for _, action in ranked]


def _features(rows: Sequence[Mapping[str, Any]], fit_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    for index, row in enumerate(rows):
        value = row.get("features")
        if value is None:
            raise ValueError(f"row {index} is missing features")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError(f"row {index} has empty or non-finite features")
        vectors.append(vector)
    dimension = vectors[0].size
    if any(vector.size != dimension for vector in vectors):
        raise ValueError("all rows must use one feature dimension")
    if not fit_indices:
        fit_indices = list(range(len(rows)))
    fit_values = np.asarray([vectors[index] for index in fit_indices], dtype=np.float64)
    mean = fit_values.mean(axis=0).astype(np.float32)
    scale = np.maximum(fit_values.std(axis=0), 1.e-6).astype(np.float32)
    scale = np.where(scale > 1.e-6, scale, 1.0).astype(np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    return (matrix - mean) / scale, mean, scale


def _split_layouts(rows: Sequence[Mapping[str, Any]], seed: int) -> tuple[set[str], set[str], set[str], dict[str, Any]]:
    names = sorted({_layout(row, index) for index, row in enumerate(rows)})
    rng = np.random.default_rng(int(seed))
    rng.shuffle(names)
    count = len(names)
    if count < 3:
        fit = set(names[: max(1, count)])
        return fit, set(), set(), {
            "layout_disjoint": True, "valid": False,
            "fallback": "fewer_than_three_layouts",
            "fit_layouts": sorted(fit), "validation_layouts": [], "calibration_layouts": [],
        }
    validation_count = max(1, int(round(.20 * count)))
    calibration_count = max(1, int(round(.20 * count)))
    while count - validation_count - calibration_count < 1:
        if validation_count > 1:
            validation_count -= 1
        elif calibration_count > 1:
            calibration_count -= 1
        else:
            break
    fit_end = count - validation_count - calibration_count
    fit = set(names[:fit_end])
    validation = set(names[fit_end:fit_end + validation_count])
    calibration = set(names[fit_end + validation_count:])
    return fit, validation, calibration, {
        "layout_disjoint": not (fit & validation or fit & calibration or validation & calibration),
        "valid": bool(fit and validation and calibration),
        "fallback": None,
        "fit_layouts": sorted(fit), "validation_layouts": sorted(validation),
        "calibration_layouts": sorted(calibration),
    }


def _layout_action_mean(values: Any, *, action: str, reference: str) -> np.ndarray | None:
    if action == reference:
        return np.zeros(3, dtype=np.float32)
    triples = _triples(values)
    if not triples:
        return None
    return np.clip(np.asarray(triples, dtype=np.float32).mean(axis=0), -1., 1.)


def _examples(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    actions: Sequence[str],
    reference: str,
    feature_matrix: np.ndarray,
) -> dict[str, Any]:
    x: list[np.ndarray] = []
    action_ids: list[int] = []
    targets: list[np.ndarray] = []
    row_ids: list[int] = []
    counts: dict[str, dict[str, int]] = {
        action: {"layouts": 0, "replicates": 0, "positive": 0, "negative": 0, "neutral": 0}
        for action in actions
    }
    ids = {action: index for index, action in enumerate(actions)}
    for row_index in indices:
        samples = _samples(rows[row_index])
        for action in actions:
            target = _layout_action_mean(samples.get(action, []), action=action, reference=reference)
            if target is None:
                continue
            x.append(feature_matrix[row_index])
            action_ids.append(ids[action])
            targets.append(target)
            row_ids.append(row_index)
            count = counts[action]
            count["layouts"] += 1
            raw = _triples(samples.get(action, []))
            count["replicates"] += len(raw)
            for value in target:
                if value > 0:
                    count["positive"] += 1
                elif value < 0:
                    count["negative"] += 1
                else:
                    count["neutral"] += 1
    dimension = feature_matrix.shape[1]
    return {
        "features": np.asarray(x, dtype=np.float32).reshape(-1, dimension),
        "action_ids": np.asarray(action_ids, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float32).reshape(-1, 3),
        "row_ids": np.asarray(row_ids, dtype=np.int64),
        "counts": counts,
    }


def _merge_counts(*maps: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for mapping in maps:
        for action, values in mapping.items():
            target = result.setdefault(action, {
                "layouts": 0, "replicates": 0, "positive": 0, "negative": 0, "neutral": 0,
            })
            for key in target:
                target[key] += int(values.get(key, 0))
    return result


def _choose_hidden(input_dim: int, output_dim: int) -> tuple[int, int]:
    for hidden in (64, 48, 32, 24, 16, 12, 8, 4):
        parameters = (input_dim + 1) * hidden + (hidden + 1) * output_dim
        if parameters <= 10000:
            return hidden, int(parameters)
    raise ValueError(f"feature dimension {input_dim} cannot fit a <=10,000 parameter value model")


def _cpu_state_dict(state: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {
        str(key): value.detach().cpu().numpy().tolist() if hasattr(value, "detach")
        else np.asarray(value).tolist()
        for key, value in state.items()
    }


def _cpu_optimizer_state(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_optimizer_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_optimizer_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_optimizer_state(item) for item in value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _paths(out: str | os.PathLike[str]) -> tuple[Path, Path, Path]:
    requested = Path(out)
    if requested.suffix.lower() == ".json":
        return requested, requested.with_suffix(".pt"), requested.with_name(requested.stem + "_manifest.json")
    if requested.suffix.lower() == ".pt":
        return requested.with_suffix(".json"), requested, requested.with_name(requested.stem + "_manifest.json")
    requested.mkdir(parents=True, exist_ok=True)
    return requested / "value_v8_model.json", requested / "value_v8_checkpoint.pt", requested / "value_v8_manifest.json"


def _layout_bootstrap(
    residuals: np.ndarray,
    *,
    seed: int,
    min_layouts: int = MIN_CALIBRATION_LAYOUTS,
) -> dict[str, Any]:
    """Calibrate residual means over layouts, never over individual tapes."""
    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n < min_layouts:
        return {
            "n_layouts": n, "mean": float(values.mean()) if n else 0.0,
            "q05": -1.0, "q50": 0.0, "q95": 1.0,
            "std": float(values.std(ddof=1)) if n > 1 else None,
            "sufficient": False, "fallback": "bounded_worst_case",
            "unit": "layout_mean", "no_per_tape_quantile": True,
        }
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    bootstrap = values[indices].mean(axis=1)
    q05, q50, q95 = np.quantile(bootstrap, [0.05, 0.50, 0.95]).tolist()
    return {
        "n_layouts": n, "mean": float(values.mean()),
        "q05": float(q05), "q50": float(q50), "q95": float(q95),
        "std": float(values.std(ddof=1)) if n > 1 else 0.0,
        "sufficient": True, "fallback": None,
        "unit": "layout_mean", "no_per_tape_quantile": True,
    }


def _calibration(
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    actions: Sequence[str],
    reference: str,
    predictions: np.ndarray,
) -> dict[str, Any]:
    action_ids = {action: index for index, action in enumerate(actions)}
    output: dict[str, Any] = {}
    for action in actions:
        if action == reference:
            continue
        # A calibration unit is one layout. If a layout contributes multiple
        # rows, average those row residuals before bootstrapping so it cannot
        # receive weight proportional to its tape count.
        grouped: dict[str, list[list[float]]] = {}
        for row_index in indices:
            actual = _layout_action_mean(
                _samples(rows[row_index]).get(action, []),
                action=action,
                reference=reference,
            )
            if actual is None:
                continue
            residual = np.asarray(actual, dtype=np.float64) - predictions[row_index, action_ids[action]]
            grouped.setdefault(_layout(rows[row_index], row_index), []).append(
                np.clip(residual, -1., 1.).tolist())
        residuals: list[list[float]] = [[], [], []]
        for layout_values in grouped.values():
            layout_mean = np.asarray(layout_values, dtype=np.float64).mean(axis=0)
            for channel in range(3):
                residuals[channel].append(float(np.clip(layout_mean[channel], -1., 1.)))
        output[action] = {
            name: _layout_bootstrap(
                residuals[channel],
                seed=_stable_seed("calibration", action, name),
            )
            for channel, name in enumerate(TRIPLE_NAMES)
        }
    return output


def fit_phase(
    rows: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    reference: str,
    out: str | os.PathLike[str],
    *,
    phase: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit expected layout/action gains on CUDA and return CPU JSON artifacts."""
    if not rows:
        raise ValueError("fit_phase requires rows")
    reference = str(reference)
    action_names = _action_names(actions, reference)
    fit_layouts, validation_layouts, calibration_layouts, split = _split_layouts(rows, int(seed))
    fit_indices = [i for i, row in enumerate(rows) if _layout(row, i) in fit_layouts]
    validation_indices = [i for i, row in enumerate(rows) if _layout(row, i) in validation_layouts]
    calibration_indices = [i for i, row in enumerate(rows) if _layout(row, i) in calibration_layouts]
    if not fit_indices:
        raise ValueError("layout split produced no fit rows")
    feature_matrix, feature_mean, feature_scale = _features(rows, fit_indices)
    fit_examples = _examples(rows, fit_indices, action_names, reference, feature_matrix)
    validation_examples = _examples(rows, validation_indices, action_names, reference, feature_matrix)
    if not len(fit_examples["features"]):
        raise ValueError("fit rows contain no action means")
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("fit_phase requires PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("fit_phase requires CUDA")

    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-3, weight_decay=1.e-4)
    target_center = fit_examples["targets"].mean(axis=0).astype(np.float32)
    target_scale = np.maximum(fit_examples["targets"].std(axis=0), 1.e-3).astype(np.float32)
    target_scale = np.where(target_scale > 1.e-3, target_scale, 1.0)
    train_targets = (fit_examples["targets"] - target_center) / target_scale
    val_targets = (
        (validation_examples["targets"] - target_center) / target_scale
        if len(validation_examples["targets"]) else validation_examples["targets"]
    )
    train_x = torch.as_tensor(fit_examples["features"], dtype=torch.float32, device=device)
    train_a = torch.as_tensor(fit_examples["action_ids"], dtype=torch.long, device=device)
    train_y = torch.as_tensor(train_targets, dtype=torch.float32, device=device)
    val_x = torch.as_tensor(validation_examples["features"], dtype=torch.float32, device=device)
    val_a = torch.as_tensor(validation_examples["action_ids"], dtype=torch.long, device=device)
    val_y = torch.as_tensor(val_targets, dtype=torch.float32, device=device)
    channel_weight = torch.tensor([.25, 1., 1.], dtype=torch.float32, device=device)
    batch_rng = np.random.default_rng(int(seed) + 4107)
    epochs = 60 if len(train_x) < 2048 else 32
    batch_size = min(64, max(8, len(train_x)))
    best_state: dict[str, Any] | None = None
    best_optimizer: Any = None
    best_torch_rng: Any = None
    best_cuda_rng: Any = None
    best_val = float("inf")
    best_epoch = 0
    optimizer_steps = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    def evaluate(x: Any, action_ids: Any, targets: Any) -> float:
        if not len(x):
            return float("nan")
        model.eval()
        with torch.no_grad():
            outputs = model(x).reshape(-1, len(action_names), 3)
            selected = outputs[torch.arange(len(x), device=device), action_ids]
            value = F.smooth_l1_loss(selected, targets, reduction="none") * channel_weight
            return float(value.sum().item() / max(len(x), 1))

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = batch_rng.permutation(len(train_x))
        total = 0.0
        seen = 0
        for start in range(0, len(permutation), batch_size):
            indexes = torch.as_tensor(permutation[start:start + batch_size], dtype=torch.long, device=device)
            outputs = model(train_x.index_select(0, indexes)).reshape(-1, len(action_names), 3)
            selected = outputs[
                torch.arange(len(indexes), device=device),
                train_a.index_select(0, indexes),
            ]
            loss = (F.smooth_l1_loss(selected, train_y.index_select(0, indexes), reduction="none")
                    * channel_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total += float(loss.detach().item()) * len(indexes)
            seen += len(indexes)
        train_losses.append(total / max(seen, 1))
        val_loss = evaluate(val_x, val_a, val_y)
        if not math.isfinite(val_loss):
            val_loss = train_losses[-1]
        val_losses.append(float(val_loss))
        if val_loss < best_val - 1.e-12:
            best_val = float(val_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_optimizer = _cpu_optimizer_state(copy.deepcopy(optimizer.state_dict()))
            best_torch_rng = torch.get_rng_state().clone()
            best_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
    if best_state is None:
        raise RuntimeError("fit_phase did not produce a best checkpoint")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    if best_optimizer is not None:
        optimizer.load_state_dict(copy.deepcopy(best_optimizer))
    if best_torch_rng is not None:
        torch.set_rng_state(best_torch_rng)
    if best_cuda_rng is not None:
        torch.cuda.set_rng_state_all(best_cuda_rng)
    torch.cuda.synchronize()
    changed = any(
        not torch.equal(parameter.detach(), initial)
        for parameter, initial in zip(model.parameters(), initial_parameters)
    )
    cuda_optimizer_step_verified = bool(
        changed and optimizer_steps > 0 and all(parameter.is_cuda for parameter in model.parameters())
    )

    all_x = torch.as_tensor(feature_matrix, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        normalized_predictions = model(all_x).reshape(-1, len(action_names), 3).cpu().numpy()
    predictions = normalized_predictions * target_scale.reshape(1, 1, 3) + target_center.reshape(1, 1, 3)
    predictions = np.clip(predictions, -1., 1.)
    calibration = _calibration(rows, calibration_indices, action_names, reference, predictions)
    calibration_layout_count = len({_layout(rows[i], i) for i in calibration_indices})
    model_path, checkpoint_path, manifest_path = _paths(out)
    checkpoint_payload = {
        "format": "v8_expected_group_value_checkpoint", "version": 1, "phase": str(phase),
        "actions": action_names, "reference": reference,
        "architecture": {"input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim},
        "state_dict": best_state, "optimizer_state": best_optimizer,
        "best_epoch": best_epoch, "train_losses": train_losses, "validation_losses": val_losses,
        "feature_mean": feature_mean.tolist(), "feature_scale": feature_scale.tolist(),
        "target_center": target_center.tolist(), "target_scale": target_scale.tolist(),
        "calibration": calibration, "seed": int(seed),
        "torch_rng": None if best_torch_rng is None else best_torch_rng,
        "cuda_rng": None if best_cuda_rng is None else best_cuda_rng,
    }
    _atomic_torch_save(torch, checkpoint_payload, checkpoint_path)
    bundle: dict[str, Any] = {
        "format": "v8_expected_group_value_bundle", "version": 1, "phase": str(phase),
        "actions": action_names, "reference": reference, "wrong_budget": DEFAULT_WRONG_BUDGET,
        "input_dim": input_dim, "hidden_dim": hidden_dim, "output_dim": output_dim,
        "parameter_count": parameter_count, "state_dict": _cpu_state_dict(model.state_dict()),
        "feature_mean": feature_mean.tolist(), "feature_scale": feature_scale.tolist(),
        "target_center": target_center.tolist(), "target_scale": target_scale.tolist(),
        "calibration": calibration, "calibration_min_layouts": MIN_CALIBRATION_LAYOUTS,
        "best_epoch": best_epoch, "best_validation_checkpoint_restored": True,
        "checkpoint": str(checkpoint_path), "manifest": str(manifest_path),
    }
    manifest: dict[str, Any] = {
        "format": "v8_expected_group_value_manifest", "version": 1, "phase": str(phase),
        "actions": action_names, "reference": reference, "rows": len(rows),
        "fit_rows": len(fit_indices), "validation_rows": len(validation_indices),
        "calibration_rows": len(calibration_indices), "split": split,
        "parameter_count": parameter_count, "layout_mean_target": True,
        "sample_signs_preserved": True, "per_tape_q05_forbidden": True,
        "calibration": {
            "unit": "layout_mean", "bootstrap": "layout bootstrap of residual means",
            "minimum_layouts": MIN_CALIBRATION_LAYOUTS,
            "fallback": "bounded_worst_case_and_reference_when calibration layouts are insufficient",
            "group_validity": bool(split["valid"] and calibration_layout_count >= MIN_CALIBRATION_LAYOUTS),
            "calibration_layout_count": int(calibration_layout_count),
            "actions": calibration,
        },
        "sample_counts": {
            "fit": fit_examples["counts"], "validation": validation_examples["counts"],
            "calibration": _examples(rows, calibration_indices, action_names, reference, feature_matrix)["counts"],
        },
        "seed": int(seed), "best_epoch": int(best_epoch),
        "best_validation_checkpoint_restored": True, "best_validation_loss": float(best_val),
        "final_validation_loss": float(val_losses[-1]), "train_losses": train_losses,
        "validation_losses": val_losses, "device": str(device),
        "torch_version": str(torch.__version__), "cuda_device": str(torch.cuda.get_device_name(device)),
        "cuda_optimizer_step_verified": cuda_optimizer_step_verified,
        "optimizer": "AdamW", "optimizer_steps": int(optimizer_steps),
        "optimizer_rng_checkpointed": True, "checkpoint": str(checkpoint_path),
    }
    _atomic_json(model_path, bundle)
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    _atomic_json(manifest_path, manifest)
    return bundle, manifest


def _bundle_array(bundle: Mapping[str, Any], key: str) -> np.ndarray:
    value = bundle.get(key)
    if value is None:
        raise ValueError(f"bundle is missing {key}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def student_scores(features: Sequence[float], bundle: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    input_dim = int(bundle.get("input_dim", vector.size))
    if vector.size != input_dim:
        raise ValueError(f"expected {input_dim} features, got {vector.size}")
    mean = _bundle_array(bundle, "feature_mean")
    scale = _bundle_array(bundle, "feature_scale")
    normalized = (vector - mean) / np.where(scale > 1.e-6, scale, 1.)
    state = bundle.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("bundle has invalid state_dict")
    first_weight = np.asarray(state["0.weight"], dtype=np.float32)
    first_bias = np.asarray(state["0.bias"], dtype=np.float32)
    second_weight = np.asarray(state["2.weight"], dtype=np.float32)
    second_bias = np.asarray(state["2.bias"], dtype=np.float32)
    hidden = np.tanh(first_weight @ normalized + first_bias)
    output = second_weight @ hidden + second_bias
    actions = [str(action) for action in bundle.get("actions", [])]
    center = _bundle_array(bundle, "target_center")
    scale_target = _bundle_array(bundle, "target_scale")
    values = output.reshape(len(actions), 3) * scale_target.reshape(1, 3) + center.reshape(1, 3)
    reference = str(bundle.get("reference", actions[0] if actions else REFERENCE_DEFAULT))
    result: dict[str, dict[str, float]] = {}
    for index, action in enumerate(actions):
        row = np.zeros(3, dtype=np.float32) if action == reference else np.clip(values[index], -1., 1.)
        result[action] = {name: float(row[channel]) for channel, name in enumerate(TRIPLE_NAMES)}
    return result


def decide(
    features: Sequence[float],
    bundle: Mapping[str, Any],
    reference: str,
) -> tuple[str, dict[str, Any]]:
    """Choose a public action using layout-calibrated expected-gain bounds."""
    reference = str(reference)
    scores = student_scores(features, bundle)
    if reference not in scores:
        raise ValueError(f"reference action {reference!r} is not in bundle actions")
    wrong_budget = float(bundle.get("wrong_budget", DEFAULT_WRONG_BUDGET))
    calibration = bundle.get("calibration", {})
    selected = reference
    best_score = 0.0
    entries: dict[str, Any] = {}
    for action in sorted(scores):
        predicted = scores[action]
        if action == reference:
            entries[action] = {
                "predicted": predicted, "strict_lower": 0., "wrong_upper": 0.,
                "score": 0., "gate": {"eligible": True, "uncertain": False, "reasons": ["frozen_reference"]},
            }
            continue
        action_cal = calibration.get(action, {}) if isinstance(calibration, Mapping) else {}
        strict_cal = action_cal.get("strict", {}) if isinstance(action_cal, Mapping) else {}
        wrong_cal = action_cal.get("wrong", {}) if isinstance(action_cal, Mapping) else {}
        sufficient = bool(
            strict_cal.get("sufficient", False)
            and wrong_cal.get("sufficient", False)
            and int(strict_cal.get("n_layouts", 0)) >= MIN_CALIBRATION_LAYOUTS
            and int(wrong_cal.get("n_layouts", 0)) >= MIN_CALIBRATION_LAYOUTS
        )
        if sufficient:
            strict_lower = float(np.clip(predicted["strict"] + float(strict_cal.get("q05", -1.)), -1., 1.))
            wrong_upper = float(np.clip(predicted["wrong"] + float(wrong_cal.get("q95", 1.)), -1., 1.))
        else:
            strict_lower, wrong_upper = -1., 1.
        strict_ok = strict_lower > 0.
        wrong_ok = wrong_upper <= wrong_budget
        eligible = bool(sufficient and strict_ok and wrong_ok)
        reasons: list[str] = []
        if not sufficient:
            reasons.append("insufficient_layout_calibration")
        if not strict_ok:
            reasons.append("expected_strict_lower_not_positive")
        if not wrong_ok:
            reasons.append("expected_wrong_upper_exceeds_budget")
        score = strict_lower - .75 * max(0., wrong_upper) + .05 * float(np.clip(predicted["soft"], -1., 1.))
        entries[action] = {
            "predicted": predicted, "strict_lower": strict_lower, "wrong_upper": wrong_upper,
            "calibration": {"strict": strict_cal, "wrong": wrong_cal},
            "score": float(score),
            "gate": {
                "eligible": eligible, "uncertain": not sufficient,
                "enough_layout_calibration": sufficient,
                "strict_lower_gt_zero": strict_ok,
                "wrong_upper_leq_budget": wrong_ok,
                "wrong_budget": wrong_budget, "reasons": reasons,
            },
        }
        if eligible and score > best_score + EPS:
            selected, best_score = action, float(score)
    diagnostics = {
        "reference": reference, "selected": selected, "estimates": entries,
        "public_features_only": True, "comparison_scope": "deployment_from_layout_calibrated_model",
        "local_adaptive_teacher_is_separate": True,
        "gate": {
            "strict_condition": "predicted expected strict gain plus layout residual lower bound > 0",
            "wrong_condition": "predicted expected wrong gain plus layout residual upper bound <= wrong_budget",
            "wrong_budget": wrong_budget, "minimum_calibration_layouts": MIN_CALIBRATION_LAYOUTS,
            "insufficient_calibration_falls_back_to_reference": True,
        },
    }
    return selected, diagnostics


def _smoke() -> dict[str, Any]:
    """Run teacher mechanisms and a real CUDA optimizer smoke."""
    rescue = {"continue": [[0., 0., 0.]] * 32, "rescue": [[.02, .20, 0.]] * 32}
    selected, teacher_diag = teacher_summary(rescue, "continue")
    if selected != "rescue":
        raise AssertionError(f"expected rescue action, got {selected}")
    rescue_gate = teacher_diag["estimates"]["rescue"]["gate"]
    if not rescue_gate["strict_lower_gt_zero"] or not rescue_gate["wrong_upper_leq_budget"]:
        raise AssertionError("20 percent rescue / zero harm did not clear local bounded gate")
    shortlist = promising({"continue": [], "window": [], "rescue": [[.01, .01, 0.]] * 4},
                          "continue", 64)
    if shortlist[0] != "continue" or "window" not in shortlist:
        raise AssertionError("promising allocation did not prioritize unknown action")
    rows: list[dict[str, Any]] = []
    for index in range(24):
        rows.append({
            "layout": f"layout-{index}",
            "features": [float(index), float(index % 4), float((index * 5) % 7)],
            "samples": {
                "continue": [[0., 0., 0.]] * 4,
                "rescue": [[.01, .18 + .001 * index, 0.]] * (4 + index % 3),
                "window": [[-.01, -.05, .08]] * (4 + index % 2),
            },
        })
    with tempfile.TemporaryDirectory(prefix="v8-value-smoke-") as directory:
        bundle, manifest = fit_phase(
            rows, ["continue", "rescue", "window"], "continue",
            directory, phase="smoke", seed=17,
        )
        chosen, decision_diag = decide(rows[0]["features"], bundle, "continue")
        if not manifest["cuda_optimizer_step_verified"]:
            raise AssertionError("CUDA optimizer update was not verified")
        if not manifest["split"]["layout_disjoint"]:
            raise AssertionError("layout split is not disjoint")
        if not manifest["layout_mean_target"] or not manifest["per_tape_q05_forbidden"]:
            raise AssertionError("layout calibration contract missing")
        if int(manifest["parameter_count"]) > 10000:
            raise AssertionError("value network exceeds 10,000 parameters")
        if decision_diag["gate"]["insufficient_calibration_falls_back_to_reference"] is not True:
            raise AssertionError("deployment gate contract missing")
    return {
        "teacher_selected": selected,
        "rescue_strict_lower": float(rescue_gate["strict_lower"]),
        "rescue_wrong_upper": float(rescue_gate["wrong_upper"]),
        "promising": shortlist,
        "student_action": chosen,
        "cuda_optimizer_step_verified": bool(manifest["cuda_optimizer_step_verified"]),
        "layout_disjoint": bool(manifest["split"]["layout_disjoint"]),
        "parameter_count": int(manifest["parameter_count"]),
    }


__all__ = [
    "fit_phase", "teacher_summary", "promising", "decide", "student_scores",
    "TRIPLE_NAMES", "MIN_INITIAL_REPLICATES", "MIN_INTERMEDIATE_REPLICATES",
    "MAX_PAIRED_REPLICATES", "MIN_CALIBRATION_LAYOUTS",
]


if __name__ == "__main__":
    print(json.dumps(_smoke(), sort_keys=True))

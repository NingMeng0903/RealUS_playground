"""Small public-history predictors for relational continuation choices.

The predictor is deliberately a narrow offline component.  It learns paired
terminal-completion differences against the incumbent MPC from public context
and checked candidate descriptors.  A missing candidate or a non-positive
prediction leaves the incumbent selected, with exact-zero tie handling.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import numbers
from typing import Any, Iterable

import numpy as np


RIDGE = 4.0
VARIANTS = ("nearest", "relations")
METRIC_COMPLETION = 0


def _flatten(value: Any, prefix: str) -> dict[str, float]:
    """Flatten numeric public/action fields with stable names."""
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            result.update(_flatten(value[key], f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.ndim == 0:
            return _flatten(array.item(), prefix)
        for index, item in np.ndenumerate(array):
            suffix = ".".join(str(index_part) for index_part in index)
            result.update(_flatten(item.item() if hasattr(item, "item") else item,
                                  f"{prefix}.{suffix}" if prefix else suffix))
        return result
    if isinstance(value, numbers.Real) and not isinstance(value, (str, bytes)):
        scalar = float(value)
        if np.isfinite(scalar):
            result[prefix] = scalar
    return result


def _public_features(public: dict[str, Any], variant: str) -> dict[str, float]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown evidence variant {variant!r}")
    fields = ["common"]
    if variant == "nearest":
        fields.append("nearest")
    else:
        fields.extend(("nearest", "relations"))
    result: dict[str, float] = {}
    for field in fields:
        result.update({f"public.{key}": value for key, value in
                       _flatten(public.get(field, {}), field).items()})
    return result


def _features(task: str, action: str, public: dict[str, Any], action_info: dict[str, Any],
              variant: str) -> dict[str, float]:
    result = _public_features(public, variant)
    # Evidence can change the ordering of two alternatives as well as the
    # choice between an alternative and incumbent MPC.
    result.update({f"route_evidence.{action}.{key}": value for key, value in tuple(result.items())})
    for key, value in _flatten(action_info.get("features", {}), "action").items():
        result[key] = value
    result[f"task.{task}"] = 1.0
    result[f"action.{action}"] = 1.0
    result[f"task_action.{task}.{action}"] = 1.0
    return result


def _metric_value(metrics: Iterable[Any], index: int = METRIC_COMPLETION) -> float:
    values = [float(row[index]) for row in metrics]
    if not values:
        raise ValueError("an action must contain at least one terminal metric")
    if not np.all(np.isfinite(values)):
        raise ValueError("terminal metrics contain non-finite completion values")
    return float(np.mean(values))


def _fit_design(examples: list[tuple[dict[str, float], float, float]]) -> dict[str, Any]:
    if not examples:
        return {"n_examples": 0, "features": [], "center": [], "scale": [],
                "coef": [], "intercept": 0.0, "ridge": RIDGE}
    names = sorted({name for fields, _, _ in examples for name in fields})
    index = {name: column for column, name in enumerate(names)}
    x = np.zeros((len(examples), len(names)), dtype=float)
    y = np.empty(len(examples), dtype=float)
    weights = np.empty(len(examples), dtype=float)
    for row, (fields, label, weight) in enumerate(examples):
        for name, value in fields.items():
            x[row, index[name]] = value
        y[row] = label
        weights[row] = weight
    weights = np.maximum(weights, 0.)
    if not np.any(weights > 0.):
        weights[:] = 1.0
    total_weight = float(weights.sum())
    center = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - center) ** 2, axis=0, weights=weights))
    scale = np.maximum(scale, .05)
    z = (x - center) / scale
    sqrt_w = np.sqrt(weights / total_weight)
    zw = z * sqrt_w[:, None]
    yw = y * sqrt_w
    regularizer = RIDGE * np.eye(z.shape[1], dtype=float)
    coef_standard = np.linalg.solve(zw.T @ zw + regularizer, zw.T @ yw)
    # Keep an explicit intercept in the original feature coordinates.  The
    # ridge penalty acts only on standardized slopes.
    coef = coef_standard / scale
    intercept = float(np.average(y, weights=weights) - center @ coef)
    return {"n_examples": len(examples), "features": names,
            "center": center.tolist(), "scale": scale.tolist(),
            "coef": coef.tolist(), "intercept": intercept, "ridge": RIDGE}


def _predict(model: dict[str, Any], fields: dict[str, float]) -> float:
    names = model.get("features", [])
    if not names:
        return 0.0
    vector = np.asarray([fields.get(name, 0.0) for name in names], dtype=float)
    coef = np.asarray(model["coef"], dtype=float)
    return float(model.get("intercept", 0.0) + vector @ coef)


def _context_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    result: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        contexts = row.get("contexts", {}) or {}
        for context_name in sorted(contexts):
            context = contexts[context_name] or {}
            result[context_name].append((row, context))
    return result


def _fit_fixed(context_rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, str]:
    """Select an action type by full-denominator training gain."""
    by_task_action: dict[tuple[str, str], float] = defaultdict(float)
    case_counts: Counter[str] = Counter()
    tasks = sorted({str(row.get("task", "")) for row, _ in context_rows})
    for row, context in context_rows:
        task = str(row.get("task", ""))
        actions = context.get("actions", {}) or {}
        repeats = context.get("repeat", {}) or {}
        baseline = repeats.get("MPC")
        if baseline is None:
            continue
        case_counts[task] += 1
        base = _metric_value(baseline)
        # Missing candidate rows remain zero gain in the full denominator.
        for action in sorted(actions):
            if action != "MPC" and repeats.get(action):
                by_task_action[task, str(action)] += _metric_value(repeats[action]) - base
    fixed: dict[str, str] = {}
    for task in tasks:
        candidates = [(gain / case_counts[task], action)
                      for (candidate_task, action), gain in by_task_action.items()
                      if candidate_task == task]
        # MPC remains the incumbent on an empty or non-positive full-denominator
        # table.  Lexicographic action order makes ties deterministic.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        if candidates and candidates[0][0] > 0.0:
            fixed[task] = candidates[0][1]
        else:
            fixed[task] = "MPC"
    return fixed


def fit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Fit one public predictor per context and evidence representation.

    The training target for a candidate is its mean completion rate over the
    supplied repeats minus the matched MPC mean for the same public state.
    Each public state receives equal total weight across its available
    candidates.  Relations are absent from every nearest model by construction.
    """
    rows = list(rows)
    grouped = _context_rows(rows)
    contexts: dict[str, Any] = {}
    for context_name, entries in sorted(grouped.items()):
        variants: dict[str, Any] = {}
        for variant in VARIANTS:
            examples: list[tuple[dict[str, float], float, float]] = []
            for row, context in entries:
                actions = context.get("actions", {}) or {}
                repeats = context.get("repeat", {}) or {}
                baseline = repeats.get("MPC")
                if not baseline:
                    continue
                base = _metric_value(baseline)
                available = [action for action in actions if action != "MPC"
                             and repeats.get(action)]
                if not available:
                    continue
                weight = 1.0 / len(available)
                for action in sorted(available):
                    label = _metric_value(repeats[action]) - base
                    examples.append((_features(str(row.get("task", "")), str(action),
                                                context.get("public", {}) or {}, actions[action], variant),
                                     label, weight))
            variants[variant] = _fit_design(examples)
        contexts[context_name] = {
            "variants": variants,
            "fixed": _fit_fixed(entries),
            "n_cases": len(entries),
        }
    return {"version": 1, "ridge": RIDGE, "variants": list(VARIANTS),
            "contexts": contexts}


def _fixed_choice(fit_context: dict[str, Any], task: str,
                  actions: dict[str, Any]) -> str:
    action = fit_context.get("fixed", {}).get(task, "MPC")
    if action != "MPC" and action in actions:
        return action
    return "MPC"


def choose(fit: dict[str, Any], task: str, context_name: str,
           public: dict[str, Any], actions: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Choose fixed, nearest-feature and relation-feature routes.

    The score for MPC is exactly zero.  A candidate is selected only when its
    predicted difference is strictly positive, so exact ties and all-negative
    predictions preserve the incumbent without a tunable threshold.
    """
    context = fit.get("contexts", {}).get(context_name, {})
    choices: dict[str, str] = {"MPC": "MPC"}
    forecasts: dict[str, dict[str, float]] = {variant: {"MPC": 0.0} for variant in VARIANTS}
    for variant in VARIANTS:
        model = context.get("variants", {}).get(variant,
                                                {"features": [], "coef": [], "intercept": 0.0})
        for action in sorted(actions):
            if action == "MPC":
                continue
            fields = _features(str(task), str(action), public, actions[action] or {}, variant)
            forecasts[variant][action] = _predict(model, fields)
        chosen = "MPC"
        best = 0.0
        for action in sorted(forecasts[variant]):
            if action == "MPC":
                continue
            score = forecasts[variant][action]
            if score > best + 1.e-12:
                best, chosen = score, action
        choices[variant] = chosen
    choices["fixed"] = _fixed_choice(context, str(task), actions)
    return choices, forecasts


__all__ = ["fit_rows", "choose"]

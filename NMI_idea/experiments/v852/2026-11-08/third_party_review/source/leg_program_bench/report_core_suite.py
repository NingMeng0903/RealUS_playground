"""Summarise the prospective V8.5.2 core-suite run.

The report intentionally treats a complete anatomy configuration (``case_id``)
as the statistical unit.  Trees, arms, frames and calls are execution records,
not additional people.  This module is a report reader only: it does not
change the runner or re-fit any model.

The plotting code uses English labels so the figures can be reused in a paper;
``RESULTS.txt`` is a short Chinese audit trail for the run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BOOTSTRAP_REPEATS = 200
TARGET_COMPLETION = 0.8


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        return rows
    return rows


def _evaluation_rows(path: Path) -> list[dict[str, Any]]:
    value = _load_json(path, [])
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict):
        for key in ("rows", "evaluation", "records"):
            if isinstance(value.get(key), list):
                return [x for x in value[key] if isinstance(x, dict)]
    return _load_jsonl(path)


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _key(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker not in seen:
            result.append(value)
            seen.add(marker)
    return result


def _case_id(row: Mapping[str, Any], fallback: int = 0) -> str:
    for field in ("case_id", "object_id", "case", "id"):
        if row.get(field) is not None:
            return str(row[field])
    return f"row-{fallback}"


def _weight(row: Mapping[str, Any]) -> float:
    value = _finite_float(row.get("population_weight"), 1.0)
    if value is None or value <= 0.0:
        return 1.0
    return value


def _vector(value: Any, n: int | None = None) -> np.ndarray | None:
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if n is not None and len(result) != n:
        return None
    result[~np.isfinite(result)] = np.nan
    return result


def _scalar(row: Mapping[str, Any], field: str) -> float | None:
    value = _finite_float(row.get(field))
    return value


def _case_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_case_id(row, index)].append(row)
    return dict(groups)


def _combine_case_vectors(
    rows: Sequence[Mapping[str, Any]], field: str, n: int
) -> tuple[np.ndarray | None, float]:
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for row in rows:
        vector = _vector(row.get(field), n)
        if vector is not None and np.isfinite(vector).any():
            vectors.append(vector)
            weights.append(_weight(row))
    if not vectors:
        return None, 1.0
    matrix = np.vstack(vectors)
    weight_array = np.asarray(weights, dtype=float)
    result = np.full(n, np.nan, dtype=float)
    for column in range(n):
        valid = np.isfinite(matrix[:, column])
        if valid.any():
            result[column] = float(
                np.average(matrix[valid, column], weights=weight_array[valid])
            )
    return result, float(np.average(weight_array))


def _aggregate_vector(
    rows: Sequence[Mapping[str, Any]], field: str, n: int
) -> tuple[np.ndarray | None, int]:
    """Average a curve over case clusters using design weights."""

    clusters: list[np.ndarray] = []
    cluster_weights: list[float] = []
    for cluster in _case_groups(rows).values():
        vector, cluster_weight = _combine_case_vectors(cluster, field, n)
        if vector is not None and np.isfinite(vector).any():
            clusters.append(vector)
            cluster_weights.append(cluster_weight)
    if not clusters:
        return None, 0
    matrix = np.vstack(clusters)
    weights = np.asarray(cluster_weights, dtype=float)
    result = np.full(n, np.nan, dtype=float)
    for column in range(n):
        valid = np.isfinite(matrix[:, column])
        if valid.any():
            result[column] = float(
                np.average(matrix[valid, column], weights=weights[valid])
            )
    return result, len(clusters)


def _aggregate_scalar(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[float | None, int]:
    values: list[float] = []
    weights: list[float] = []
    for cluster in _case_groups(rows).values():
        cluster_values = [v for v in (_scalar(row, field) for row in cluster) if v is not None]
        if cluster_values:
            values.append(float(np.mean(cluster_values)))
            weights.append(float(np.mean([_weight(row) for row in cluster])))
    if not values:
        return None, 0
    return float(np.average(np.asarray(values), weights=np.asarray(weights))), len(values)


def _curve_metrics(
    rows: Sequence[Mapping[str, Any]], deadlines: Sequence[float]
) -> dict[str, Any]:
    n = len(deadlines)
    predicted, n_pred = _aggregate_vector(rows, "predicted", n)
    measured, n_measured = _aggregate_vector(rows, "measured", n)
    lower, _ = _aggregate_vector(rows, "lower", n)
    upper, _ = _aggregate_vector(rows, "upper", n)
    retrospective, _ = _aggregate_vector(rows, "retrospective_best_library", n)
    row_mean_error, _ = _aggregate_scalar(rows, "mean_abs_error")
    row_max_error, _ = _aggregate_scalar(rows, "max_abs_error")
    if predicted is not None and measured is not None:
        error = np.abs(predicted - measured)
        mean_error_curve = float(np.nanmean(error))
        max_error_curve = float(np.nanmax(error))
    else:
        mean_error_curve = None
        max_error_curve = None
    containment: float | None = None
    if measured is not None and lower is not None and upper is not None:
        valid = np.isfinite(measured) & np.isfinite(lower) & np.isfinite(upper)
        if valid.any():
            containment = float(
                np.mean((measured[valid] >= lower[valid]) & (measured[valid] <= upper[valid]))
            )
    return {
        "predicted": _json_array(predicted),
        "measured": _json_array(measured),
        "lower": _json_array(lower),
        "upper": _json_array(upper),
        "retrospective_best_library": _json_array(retrospective),
        # Population-curve errors compare the aggregated conditional
        # probability curves.  Row errors are retained separately because a
        # single case's p-y discrepancy also contains irreducible realization
        # noise and must not be called model misspecification.
        "population_curve_mean_abs_error": mean_error_curve,
        "population_curve_max_abs_error": max_error_curve,
        "case_realization_discrepancy_mean_abs_error": row_mean_error,
        "case_realization_discrepancy_max_abs_error": row_max_error,
        # Compatibility aliases for older consumers; the explicit names above
        # are the labels used in the report and plots.
        "mean_abs_error_curve": mean_error_curve,
        "max_abs_error_curve": max_error_curve,
        "row_mean_abs_error": row_mean_error,
        "row_max_abs_error": row_max_error,
        "descriptive_envelope_containment": containment,
        "n_case_clusters": max(n_pred, n_measured),
    }


def _json_array(value: np.ndarray | None) -> list[float | None] | None:
    if value is None:
        return None
    result: list[float | None] = []
    for item in value.tolist():
        result.append(float(item) if math.isfinite(float(item)) else None)
    return result


def _group_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    return dict(grouped)


def _nested_put(root: dict[str, Any], keys: Sequence[Any], value: Any) -> None:
    current = root
    for key in keys[:-1]:
        current = current.setdefault(_key(key), {})
    current[_key(keys[-1])] = value


def _curve_from_summary(summary: Mapping[str, Any], field: str) -> np.ndarray | None:
    return _vector(summary.get(field))


def _build_a_summary(
    rows: Sequence[Mapping[str, Any]],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    arms: Sequence[Any],
    methods: Sequence[Any],
    deadlines: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in levels:
        for rho in rhos:
            for arm in arms:
                for method in methods:
                    selected = [
                        row
                        for row in rows
                        if row.get("level") == level
                        and row.get("rho") == rho
                        and row.get("arm") == arm
                        and row.get("method") == method
                    ]
                    _nested_put(result, [level, rho, arm, method], _curve_metrics(selected, deadlines))
    return result


def _lookup_summary(
    summary: Mapping[str, Any], level: Any, rho: Any, arm: Any, method: Any
) -> Mapping[str, Any] | None:
    level_map = summary.get(_key(level), {})
    rho_map = level_map.get(_key(rho), {}) if isinstance(level_map, Mapping) else {}
    arm_map = rho_map.get(_key(arm), {}) if isinstance(rho_map, Mapping) else {}
    value = arm_map.get(_key(method), {}) if isinstance(arm_map, Mapping) else {}
    return value if isinstance(value, Mapping) else None


def _collapse_case_curve_map(
    rows: Sequence[Mapping[str, Any]], field: str, n: int
) -> dict[str, tuple[np.ndarray, float]]:
    result: dict[str, tuple[np.ndarray, float]] = {}
    for case, cluster in _case_groups(rows).items():
        vector, weight = _combine_case_vectors(cluster, field, n)
        if vector is not None:
            result[case] = (vector, weight)
    return result


def _bootstrap_mean(
    values: Mapping[str, tuple[np.ndarray, float]],
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = 20261108,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not values:
        return None, None
    cases = list(values)
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for _ in range(repeats):
        picked = rng.choice(cases, size=len(cases), replace=True)
        vectors = np.vstack([values[case][0] for case in picked])
        weights = np.asarray([values[case][1] for case in picked], dtype=float)
        out = np.full(vectors.shape[1], np.nan, dtype=float)
        for col in range(vectors.shape[1]):
            valid = np.isfinite(vectors[:, col])
            if valid.any():
                out[col] = float(np.average(vectors[valid, col], weights=weights[valid]))
        samples.append(out)
    matrix = np.vstack(samples)
    return np.nanpercentile(matrix, 2.5, axis=0), np.nanpercentile(matrix, 97.5, axis=0)


def _build_b_effects(
    rows: Sequence[Mapping[str, Any]],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    methods: Sequence[Any],
    deadlines: Sequence[float],
    baseline_arm: str,
    release_arms: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    n = len(deadlines)
    for level in levels:
        for rho in rhos:
            for method in methods:
                base_rows = [
                    row
                    for row in rows
                    if row.get("level") == level
                    and row.get("rho") == rho
                    and row.get("arm") == baseline_arm
                    and row.get("method") == method
                ]
                base_pred = _collapse_case_curve_map(base_rows, "predicted", n)
                base_meas = _collapse_case_curve_map(base_rows, "measured", n)
                for release_arm in release_arms:
                    late_rows = [
                        row
                        for row in rows
                        if row.get("level") == level
                        and row.get("rho") == rho
                        and row.get("arm") == release_arm
                        and row.get("method") == method
                    ]
                    late_pred = _collapse_case_curve_map(late_rows, "predicted", n)
                    late_meas = _collapse_case_curve_map(late_rows, "measured", n)
                    paired_pred: dict[str, tuple[np.ndarray, float]] = {}
                    paired_meas: dict[str, tuple[np.ndarray, float]] = {}
                    for case in sorted(set(base_pred) & set(late_pred)):
                        delta = late_pred[case][0] - base_pred[case][0]
                        paired_pred[case] = (delta, float(np.mean([base_pred[case][1], late_pred[case][1]])))
                    for case in sorted(set(base_meas) & set(late_meas)):
                        delta = late_meas[case][0] - base_meas[case][0]
                        paired_meas[case] = (delta, float(np.mean([base_meas[case][1], late_meas[case][1]])))
                    pred_delta, n_pred = _aggregate_paired(paired_pred, n)
                    meas_delta, n_meas = _aggregate_paired(paired_meas, n)
                    pred_lo, pred_hi = _bootstrap_mean(paired_pred, seed=20261108 + n_pred)
                    meas_lo, meas_hi = _bootstrap_mean(paired_meas, seed=20261109 + n_meas)
                    value = {
                        "baseline_arm": baseline_arm,
                        "comparison_arm": release_arm,
                        "predicted_delta": _json_array(pred_delta),
                        "measured_delta": _json_array(meas_delta),
                        "predicted_bootstrap_low": _json_array(pred_lo),
                        "predicted_bootstrap_high": _json_array(pred_hi),
                        "measured_bootstrap_low": _json_array(meas_lo),
                        "measured_bootstrap_high": _json_array(meas_hi),
                        "bootstrap_repeats": BOOTSTRAP_REPEATS,
                        "n_case_pairs": min(n_pred, n_meas),
                        "bootstrap_is_formal_confidence": False,
                    }
                    _nested_put(result, [level, rho, method, release_arm], value)
    return result


def _aggregate_paired(
    values: Mapping[str, tuple[np.ndarray, float]], n: int
) -> tuple[np.ndarray | None, int]:
    if not values:
        return None, 0
    matrix = np.vstack([item[0] for item in values.values()])
    weights = np.asarray([item[1] for item in values.values()], dtype=float)
    result = np.full(n, np.nan, dtype=float)
    for column in range(n):
        valid = np.isfinite(matrix[:, column])
        if valid.any():
            result[column] = float(np.average(matrix[valid, column], weights=weights[valid]))
    return result, len(values)


def _target_entry(
    row: Mapping[str, Any], deadlines: Sequence[float], target: float
) -> dict[str, Any]:
    measured = _vector(row.get("measured"), len(deadlines))
    if measured is None:
        return {"attained": False, "deadline": None, "cost": None}
    valid = np.isfinite(measured) & (measured >= target)
    if not valid.any():
        return {"attained": False, "deadline": None, "cost": None}
    index = int(np.flatnonzero(valid)[0])
    cost: list[float] | None = None
    curves = row.get("cost_curves")
    if isinstance(curves, list) and index < len(curves) and isinstance(curves[index], (list, tuple)):
        try:
            values = [float(x) for x in curves[index]]
            if all(math.isfinite(x) for x in values):
                cost = values
        except (TypeError, ValueError):
            cost = None
    return {
        "attained": True,
        "deadline": float(deadlines[index]),
        "cost": cost,
        "cost_available": cost is not None,
    }


def _combine_case_costs(
    rows: Sequence[Mapping[str, Any]], n: int
) -> tuple[np.ndarray | None, float]:
    """Combine one case's [deadline, 4-cost] curve without changing its gate."""

    matrices: list[np.ndarray] = []
    weights: list[float] = []
    for row in rows:
        raw = row.get("cost_curves")
        if not isinstance(raw, list) or len(raw) != n:
            continue
        try:
            matrix = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            continue
        if matrix.shape != (n, 4) or not np.isfinite(matrix).any():
            continue
        matrices.append(matrix)
        weights.append(_weight(row))
    if not matrices:
        return None, 1.0
    stack = np.stack(matrices, axis=0)
    weight_array = np.asarray(weights, dtype=float)
    result = np.full((n, 4), np.nan, dtype=float)
    for deadline_index in range(n):
        for cost_index in range(4):
            valid = np.isfinite(stack[:, deadline_index, cost_index])
            if valid.any():
                result[deadline_index, cost_index] = float(
                    np.average(stack[valid, deadline_index, cost_index], weights=weight_array[valid])
                )
    return result, float(np.average(weight_array))


def _aggregate_cost_curve(
    rows: Sequence[Mapping[str, Any]], n: int
) -> tuple[np.ndarray | None, int]:
    matrices: list[np.ndarray] = []
    weights: list[float] = []
    for cluster in _case_groups(rows).values():
        matrix, weight = _combine_case_costs(cluster, n)
        if matrix is not None:
            matrices.append(matrix)
            weights.append(weight)
    if not matrices:
        return None, 0
    stack = np.stack(matrices, axis=0)
    weight_array = np.asarray(weights, dtype=float)
    result = np.full((n, 4), np.nan, dtype=float)
    for deadline_index in range(n):
        for cost_index in range(4):
            valid = np.isfinite(stack[:, deadline_index, cost_index])
            if valid.any():
                result[deadline_index, cost_index] = float(
                    np.average(stack[valid, deadline_index, cost_index], weights=weight_array[valid])
                )
    return result, len(matrices)


def _population_target_stats(
    rows: Sequence[Mapping[str, Any]], deadlines: Sequence[float], target: float
) -> dict[str, Any]:
    """Find the first budget where the weighted population curve reaches target."""

    n = len(deadlines)
    measured, n_cases = _aggregate_vector(rows, "measured", n)
    costs, n_cost_cases = _aggregate_cost_curve(rows, n)
    result: dict[str, Any] = {
        "target_completion": target,
        "aggregation": "population_weighted_case_cluster_measured_curve",
        "n_case_clusters": n_cases,
        "n_cost_case_clusters": n_cost_cases,
        "deadline": "unattainable",
        "attained_completion": None,
        "cost_at_population_target": None,
        "cost_available": costs is not None,
    }
    if measured is None:
        return result
    valid = np.isfinite(measured) & (measured >= target)
    if not valid.any():
        return result
    index = int(np.flatnonzero(valid)[0])
    result["deadline"] = float(deadlines[index])
    result["attained_completion"] = float(measured[index])
    if costs is not None and index < len(costs):
        values = costs[index]
        if np.isfinite(values).all():
            result["cost_at_population_target"] = [float(value) for value in values]
    return result


def _target_stats(
    rows: Sequence[Mapping[str, Any]], deadlines: Sequence[float], target: float
) -> dict[str, Any]:
    entries_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        entry = _target_entry(row, deadlines, target)
        entry["weight"] = _weight(row)
        entries_by_case[_case_id(row, index)].append(entry)
    attained_deadlines: list[float] = []
    costs: list[np.ndarray] = []
    case_weights: list[float] = []
    attained_cases = 0
    cost_cases = 0
    for entries in entries_by_case.values():
        attained = [entry for entry in entries if entry.get("attained")]
        if not attained:
            continue
        attained_cases += 1
        weight = float(np.mean([entry["weight"] for entry in attained]))
        case_weights.append(weight)
        attained_deadlines.append(float(np.average([entry["deadline"] for entry in attained])))
        available = [entry["cost"] for entry in attained if entry.get("cost") is not None]
        if available:
            cost_cases += 1
            costs.append(np.mean(np.asarray(available, dtype=float), axis=0))
    total_cases = len(entries_by_case)
    output: dict[str, Any] = {
        "target_completion": target,
        "n_case_clusters": total_cases,
        "n_attained": attained_cases,
        "n_unattained": max(0, total_cases - attained_cases),
        "n_cost_available": cost_cases,
        "deadline_median": float(np.median(attained_deadlines)) if attained_deadlines else "unattainable",
        "deadline_weighted_mean": (
            float(np.average(np.asarray(attained_deadlines), weights=np.asarray(case_weights)))
            if attained_deadlines
            else "unattainable"
        ),
        "cost_median": None,
        "cost_weighted_mean": None,
        "cost_available": bool(costs),
    }
    if costs:
        matrix = np.vstack(costs)
        output["cost_median"] = [float(x) for x in np.median(matrix, axis=0)]
        # The cost vectors are [seconds, calls, unique frames, recoveries].
        output["cost_weighted_mean"] = [float(x) for x in np.mean(matrix, axis=0)]
    return output


def _build_c_summary(
    rows: Sequence[Mapping[str, Any]],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    arms: Sequence[Any],
    methods: Sequence[Any],
    deadlines: Sequence[float],
    target: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in levels:
        for rho in rhos:
            for arm in arms:
                for method in methods:
                    selected = [
                        row
                        for row in rows
                        if row.get("level") == level
                        and row.get("rho") == rho
                        and row.get("arm") == arm
                        and row.get("method") == method
                    ]
                    _nested_put(
                        result,
                        [level, rho, arm, method],
                        {
                            "population_target": _population_target_stats(selected, deadlines, target),
                            "per_case_first_attainment": _target_stats(selected, deadlines, target),
                        },
                    )
    return result


def _learning_method(methods: Sequence[Any], label: str) -> str | None:
    names = [str(method) for method in methods]
    exact = {
        "n8": ("joint_n8", "joint_8"),
        "n16": ("joint_n16", "joint_16"),
        "nfull": ("joint_nfull", "joint_full", "joint_robust"),
    }[label]
    for candidate in exact:
        if candidate in names:
            return candidate
    if label == "n8":
        for method in names:
            if re.search(r"(?:^|[_-])8(?:$|[_-])", method.lower()):
                return method
    if label == "n16":
        for method in names:
            if re.search(r"(?:^|[_-])16(?:$|[_-])", method.lower()):
                return method
    return None


def _build_learning_summary(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[Any],
    level: Any,
    arm: Any,
    rho: Any,
    deadlines: Sequence[float],
    completion_deadline: float,
) -> dict[str, Any]:
    index = int(np.argmin(np.abs(np.asarray(deadlines, dtype=float) - completion_deadline)))
    result: dict[str, Any] = {
        "completion_deadline_requested": completion_deadline,
        "completion_deadline_used": float(deadlines[index]) if deadlines else None,
        "levels": {},
    }
    for label in ("n8", "n16", "nfull"):
        method = _learning_method(methods, label)
        selected = [
            row
            for row in rows
            if row.get("level") == level
            and row.get("rho") == rho
            and row.get("arm") == arm
            and (method is not None and row.get("method") == method)
        ]
        metrics = _curve_metrics(selected, deadlines) if method is not None else {}
        completion = None
        predicted_completion = None
        if metrics.get("measured") is not None:
            completion = metrics["measured"][index]
        if metrics.get("predicted") is not None:
            predicted_completion = metrics["predicted"][index]
        result["levels"][label] = {
            "method": method,
            "available": method is not None and bool(selected),
            "population_curve_mean_abs_error": metrics.get("population_curve_mean_abs_error"),
            "mean_abs_error_curve": metrics.get("population_curve_mean_abs_error"),
            "completion_measured": completion,
            "completion_predicted": predicted_completion,
            "n_case_clusters": metrics.get("n_case_clusters", 0),
        }
    return result


def _build_window_summary(
    rows: Sequence[Mapping[str, Any]],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    arms: Sequence[Any],
    methods: Sequence[Any],
    deadlines: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for level in levels:
        for rho in rhos:
            for arm in arms:
                for method in methods:
                    selected = [
                        row
                        for row in rows
                        if row.get("level") == level
                        and row.get("rho") == rho
                        and row.get("arm") == arm
                        and row.get("method") == method
                    ]
                    metrics = _curve_metrics(selected, deadlines)
                    population_value = metrics.get("population_curve_max_abs_error")
                    realization_value = metrics.get("case_realization_discrepancy_max_abs_error")
                    _nested_put(
                        result,
                        [level, rho, arm, method],
                        {
                            "population_curve_max_error": population_value,
                            "case_realization_discrepancy_max_error": realization_value,
                            "weighted_mean": population_value,
                            "case_realization_weighted_mean": realization_value,
                            "n_case_clusters": metrics.get("n_case_clusters", 0),
                        },
                    )
    return result


def _failure_summary(input_dir: Path, run_status: Mapping[str, Any]) -> dict[str, Any]:
    executions = _load_jsonl(input_dir / "executions.jsonl")
    top_failures = sum(not bool(row.get("success", False)) for row in executions)
    status_counts: Counter[str] = Counter()
    stress_counts: Counter[str] = Counter()
    retry_count = 0
    recoveries = 0
    for row in executions:
        stress_counts[str(row.get("level", "unknown"))] += 1
        recoveries += int(_finite_float(row.get("recoveries"), 0.0) or 0.0)
        for call in row.get("calls", []) if isinstance(row.get("calls"), list) else []:
            if isinstance(call, Mapping):
                status_counts[str(call.get("status", "unknown"))] += 1
                if bool(call.get("is_retry", False)):
                    retry_count += 1
    return {
        "n_execution_records": len(executions),
        "n_failed_execution_records": int(top_failures),
        "call_status_counts": dict(status_counts),
        "stress_level_execution_counts": dict(stress_counts),
        "total_recoveries": recoveries,
        "retry_calls": retry_count,
        "run_status": dict(run_status),
    }


def _manifest_geometry_map(input_dir: Path) -> dict[str, Any]:
    manifest = _load_json(input_dir / "test_manifest.json", [])
    if not isinstance(manifest, list):
        return {}
    result: dict[str, Any] = {}
    for item in manifest:
        if not isinstance(item, Mapping) or item.get("case_id") is None:
            continue
        spec = item.get("spec", {})
        if isinstance(spec, Mapping) and spec.get("geometry_seed") is not None:
            result[str(item["case_id"])] = spec.get("geometry_seed")
        elif item.get("geometry_seed") is not None:
            result[str(item["case_id"])] = item.get("geometry_seed")
    return result


def _annotate_geometry(
    rows: Sequence[Mapping[str, Any]], input_dir: Path, run_id: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    geometry_map = _manifest_geometry_map(input_dir)
    annotated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        original_case = _case_id(row, index)
        if run_id is not None:
            item["run_id"] = run_id
            item["case_id"] = f"{run_id}::{original_case}"
        item["geometry_seed"] = geometry_map.get(original_case)
        annotated.append(item)
    metadata = _geometry_metadata(annotated)
    metadata["manifest_available"] = bool(geometry_map)
    metadata["run_id"] = run_id
    return annotated, metadata


def _geometry_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seeds = {
        str(row.get("geometry_seed"))
        for row in rows
        if row.get("geometry_seed") is not None
    }
    case_clusters = {_case_id(row, index) for index, row in enumerate(rows)}
    levels: dict[str, set[str]] = defaultdict(set)
    run_level_cases: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        level = str(row.get("level", "unknown"))
        if row.get("geometry_seed") is not None:
            levels[level].add(str(row["geometry_seed"]))
        run_level_cases[level].add(_case_id(row, index))
    overlap: dict[str, int] = {}
    level_names = sorted(levels)
    for left_index, left in enumerate(level_names):
        for right in level_names[left_index + 1 :]:
            overlap[f"{left}|{right}"] = len(levels[left] & levels[right])
    return {
        "unique_geometry_seed_count": len(seeds),
        "per_level_independent_counts": {key: len(value) for key, value in sorted(levels.items())},
        "cross_level_shared_geometry_seed_counts": overlap,
        "per_level_run_case_cluster_counts": {
            key: len(value) for key, value in sorted(run_level_cases.items())
        },
        "n_run_case_clusters": len(case_clusters),
    }


def _audit_artifact_summary(input_dir: Path) -> dict[str, Any]:
    """Read audit and local-bundle artifacts without treating them as cases."""

    audit_files = sorted(input_dir.glob("model_audit_*.json"))
    model_audits: list[dict[str, Any]] = []
    strict_flags: list[bool] = []
    for path in audit_files:
        value = _load_json(path, {})
        models: dict[str, Any] = {}
        if isinstance(value, Mapping):
            for name, item in value.items():
                if not isinstance(item, Mapping):
                    continue
                strict = bool(item.get("bootstrap_is_strict_confidence_set", False))
                strict_flags.append(strict)
                models[str(name)] = {
                    "mode": item.get("mode"),
                    "n_objects": item.get("n_objects"),
                    "n_strata": item.get("n_strata"),
                    "bootstrap_is_strict_confidence_set": strict,
                }
        model_audits.append({"file": path.name, "models": models})
    bundles = _load_jsonl(input_dir / "local_bundles.jsonl")
    bundle_levels = Counter(str(row.get("level", "unknown")) for row in bundles)
    bundle_objects = {str(row.get("object_id")) for row in bundles if row.get("object_id") is not None}
    method_label_audit = _load_json(input_dir / "method_label_audit.json", {})
    if not isinstance(method_label_audit, Mapping):
        method_label_audit = {}
    manifest = _load_json(input_dir / "test_manifest.json", [])
    design_by_level: dict[str, dict[str, Any]] = {}
    if isinstance(manifest, list):
        for item in manifest:
            if not isinstance(item, Mapping):
                continue
            level = str(item.get("level", "unknown"))
            spec = item.get("spec", {})
            if not isinstance(spec, Mapping):
                spec = {}
            uv = f"u={spec.get('u')},v={spec.get('v')}"
            chain_value = item.get("chain", [])
            chain = ",".join(map(str, chain_value)) if isinstance(chain_value, (list, tuple)) else str(chain_value)
            level_record = design_by_level.setdefault(
                level, {"n_cases": 0, "uv_counts": Counter(), "chain_counts": Counter(), "uv_chain_counts": Counter()}
            )
            level_record["n_cases"] += 1
            level_record["uv_counts"][uv] += 1
            level_record["chain_counts"][chain] += 1
            level_record["uv_chain_counts"][f"{uv}|chain={chain}"] += 1
        for level_record in design_by_level.values():
            for field in ("uv_counts", "chain_counts", "uv_chain_counts"):
                level_record[field] = dict(level_record[field])
    return {
        "model_audit_files": model_audits,
        "n_model_audit_files": len(audit_files),
        "n_local_bundle_rows": len(bundles),
        "n_local_bundle_objects": len(bundle_objects),
        "local_bundle_level_rows": dict(bundle_levels),
        "any_strict_confidence_set_flag": any(strict_flags),
        "strict_confidence_set_flags_are_descriptive": True,
        "method_label_audit": dict(method_label_audit),
        "test_manifest_design": {
            "n_rows": len(manifest) if isinstance(manifest, list) else 0,
            "by_level": design_by_level,
            "purpose": "request x UV allocation audit; not an additional independent-sample count",
        },
    }


def _method_labels_from_audit(audit: Mapping[str, Any]) -> dict[str, str]:
    """Return display-only labels; raw method keys remain unchanged."""

    labels: dict[str, str] = {}
    method_audit = audit.get("method_label_audit", {})
    if isinstance(method_audit, Mapping):
        alias_text = str(method_audit.get("joint_nominal_actual_objective", ""))
        if (
            "same robust objective" in alias_text
            or "consistency alias" in alias_text
            or bool(method_audit.get("joint_nominal_is_robust_alias", False))
        ):
            labels["joint_nominal"] = "joint_nominal (robust alias; duplicate)"
    return labels


def _four_cell_gamma_summary(input_dir: Path, deadlines: Sequence[float]) -> dict[str, Any]:
    """Read the fixed-mode four-cell contrast written by the suite runner.

    ``four_cells.json`` is deliberately separate from ``evaluation.json``:
    its cells keep the observed-sign/mode intervention, while the main
    evaluation curve may re-select a mixture at each budget.  We therefore do
    not fold Gamma into the CDF-like budget curves.
    """

    raw = _load_json(input_dir / "four_cells.json", [])
    if not isinstance(raw, list):
        raw = []
    n = len(deadlines)
    records: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        measured = _vector(item.get("gamma_measured"), n)
        predicted = _vector(item.get("gamma_predicted"), n)
        if measured is not None and predicted is not None:
            error = np.abs(predicted - measured)
            mean_error = float(np.nanmean(error))
            max_error = float(np.nanmax(error))
        else:
            mean_error = None
            max_error = None
        records.append(
            {
                "level": item.get("level"),
                "rho": item.get("rho"),
                "n_cases": item.get("n_cases", {}),
                "measured_cells": item.get("measured"),
                "predicted_cells": item.get("predicted"),
                "gamma_measured": _json_array(measured),
                "gamma_predicted": _json_array(predicted),
                "mean_abs_error": mean_error,
                "max_abs_error": max_error,
                "all_four_cells_present": measured is not None and predicted is not None,
            }
        )
    return {
        "available": bool(records),
        "n_records": len(records),
        "records": records,
        "definition": "Gamma(T)=cell(+1,+1)+cell(-1,-1)-cell(+1,-1)-cell(-1,+1)",
        "fixed_mode_intervention": True,
    }


def _method_label(method: Any, labels: Mapping[str, str] | None = None) -> str:
    raw = str(method)
    return str(labels.get(raw, raw)) if labels else raw


def _methods_text(methods: Sequence[Any], labels: Mapping[str, str] | None = None) -> str:
    return ", ".join(_method_label(x, labels) for x in methods) if methods else "(none)"


def _plot_a(
    out: Path,
    a_summary: Mapping[str, Any],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    methods: Sequence[Any],
    arm: str,
    deadlines: Sequence[float],
    method_labels: Mapping[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(
        max(1, len(levels)), max(1, len(rhos)), squeeze=False,
        figsize=(5.0 * max(1, len(rhos)) + 2.2, 3.8 * max(1, len(levels))),
        sharex=True, sharey=True,
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(methods))))
    x = np.asarray(deadlines, dtype=float)
    for i, level in enumerate(levels or ["none"]):
        for j, rho in enumerate(rhos or ["none"]):
            ax = axes[i][j]
            if level == "none" or rho == "none":
                ax.text(0.5, 0.5, "no evaluation rows", ha="center", va="center")
                continue
            plotted = False
            for color, method in zip(colors, methods):
                record = _lookup_summary(a_summary, level, rho, arm, method)
                if not record:
                    continue
                measured = _vector(record.get("measured"), len(x))
                predicted = _vector(record.get("predicted"), len(x))
                if measured is not None:
                    ax.plot(x, measured, color=color, label=f"{_method_label(method, method_labels)} measured")
                    plotted = True
                if predicted is not None:
                    ax.plot(x, predicted, color=color, linestyle="--", alpha=0.75, label=f"{_method_label(method, method_labels)} predicted")
            ax.set_title(f"level={level}, rho={rho:g}" if isinstance(rho, float) else f"level={level}, rho={rho}")
            ax.set_ylim(-0.03, 1.05)
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("deadline (s)")
    for row in axes:
        row[0].set_ylabel("completion probability")
    handles: list[Any] = []
    labels: list[str] = []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    fig.suptitle(f"A. Budgeted completion curves by method ({arm})")
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.83, 0.5), fontsize=8)
    fig.subplots_adjust(right=0.81, bottom=0.10, top=0.92, left=0.06, wspace=0.12, hspace=0.22)
    fig.savefig(out / "A_completion_CDF.png", dpi=170)
    plt.close(fig)


def _plot_b(
    out: Path,
    b_summary: Mapping[str, Any],
    levels: Sequence[Any],
    rho: Any,
    methods: Sequence[Any],
    release_arms: Sequence[str],
    deadlines: Sequence[float],
    method_labels: Mapping[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(
        max(1, len(levels)), 1, squeeze=False,
        figsize=(11.0, 3.9 * max(1, len(levels))), sharex=True,
    )
    x = np.asarray(deadlines, dtype=float)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(methods))))
    for i, level in enumerate(levels or ["none"]):
        ax = axes[i][0]
        if level == "none":
            ax.text(0.5, 0.5, "no evaluation rows", ha="center", va="center")
            continue
        for color, method in zip(colors, methods):
            for release in release_arms:
                level_map = b_summary.get(_key(level), {})
                rho_map = level_map.get(_key(rho), {}) if isinstance(level_map, Mapping) else {}
                method_map = rho_map.get(_key(method), {}) if isinstance(rho_map, Mapping) else {}
                record = method_map.get(_key(release), {}) if isinstance(method_map, Mapping) else {}
                measured = _vector(record.get("measured_delta"), len(x)) if record else None
                predicted = _vector(record.get("predicted_delta"), len(x)) if record else None
                if measured is not None:
                    ax.plot(x, measured, color=color, label=f"{_method_label(method, method_labels)} {release} measured")
                if predicted is not None:
                    ax.plot(x, predicted, color=color, linestyle="--", alpha=0.75, label=f"{_method_label(method, method_labels)} {release} predicted")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"level={level}, rho={rho:g}" if isinstance(rho, float) else f"level={level}, rho={rho}")
        ax.set_ylabel("release − retain completion")
        ax.grid(alpha=0.25)
    axes[-1][0].set_xlabel("deadline (s)")
    fig.suptitle("B. Effect of release timing (cluster bootstrap bands are descriptive)")
    handles: list[Any] = []
    labels: list[str] = []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.80, 0.5), fontsize=8)
    fig.subplots_adjust(right=0.78, bottom=0.11, top=0.92, left=0.08, hspace=0.24)
    fig.savefig(out / "B_release_effect.png", dpi=170)
    plt.close(fig)


def _c_lookup(c_summary: Mapping[str, Any], level: Any, rho: Any, arm: Any, method: Any) -> Mapping[str, Any] | None:
    level_map = c_summary.get(_key(level), {})
    rho_map = level_map.get(_key(rho), {}) if isinstance(level_map, Mapping) else {}
    arm_map = rho_map.get(_key(arm), {}) if isinstance(rho_map, Mapping) else {}
    result = arm_map.get(_key(method), {}) if isinstance(arm_map, Mapping) else {}
    return result if isinstance(result, Mapping) else None


def _population_c_target(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not record:
        return None
    target = record.get("population_target")
    if isinstance(target, Mapping):
        return target
    # Backward-compatible handling for reports generated before the explicit
    # population-target contract.
    return record


def _plot_c(
    out: Path,
    c_summary: Mapping[str, Any],
    levels: Sequence[Any],
    rho: Any,
    methods: Sequence[Any],
    arms: Sequence[str],
    method_labels: Mapping[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    x = np.arange(len(methods), dtype=float)
    width = 0.8 / max(1, len(arms))
    colors = {"retain_early": "#4472c4", "hide_old": "#c0504d"}
    any_cost = False
    for level_index, level in enumerate(levels or ["none"]):
        # One line per level is enough for the compact figure; all levels remain
        # in summary.json.
        for arm_index, arm in enumerate(arms):
            deadlines: list[float] = []
            costs: list[float] = []
            for method in methods:
                record = _population_c_target(_c_lookup(c_summary, level, rho, arm, method))
                deadline = record.get("deadline") if record else "unattainable"
                deadlines.append(float(deadline) if isinstance(deadline, (int, float)) else np.nan)
                cost = record.get("cost_at_population_target") if record else None
                costs.append(float(cost[0]) if isinstance(cost, list) and cost else np.nan)
                if isinstance(cost, list) and cost:
                    any_cost = True
            offset = (arm_index - (len(arms) - 1) / 2.0) * width
            axes[0].bar(x + offset, deadlines, width, label=f"{level} {arm}", color=colors.get(arm))
            if any(np.isfinite(costs)):
                axes[1].bar(x + offset, costs, width, label=f"{level} {arm}", color=colors.get(arm))
    axes[0].set_title("Target completion 0.8: deadline")
    axes[0].set_ylabel("first deadline reaching 0.8 (s)")
    axes[0].set_xticks(x, [_method_label(method, method_labels) for method in methods], rotation=35, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    if any_cost:
        axes[1].set_title("Same target: interaction cost")
        axes[1].set_ylabel("seconds at that deadline")
        axes[1].set_xticks(x, [_method_label(method, method_labels) for method in methods], rotation=35, ha="right")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "cost_curves unavailable in this run", ha="center", va="center")
    fig.suptitle(f"C. Equal completion target, rho={rho:g}" if isinstance(rho, float) else "C. Equal completion target")
    fig.tight_layout()
    fig.savefig(out / "C_equal_target_0p8.png", dpi=170)
    plt.close(fig)


def _plot_learning(out: Path, learning: Mapping[str, Any]) -> None:
    labels = ["n8", "n16", "nfull"]
    entries = learning.get("levels", {})
    error = [entries.get(label, {}).get("mean_abs_error_curve", np.nan) for label in labels]
    completion = [entries.get(label, {}).get("completion_measured", np.nan) for label in labels]
    error = [float(value) if isinstance(value, (int, float)) else np.nan for value in error]
    completion = [float(value) if isinstance(value, (int, float)) else np.nan for value in completion]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    x = np.arange(len(labels))
    axes[0].plot(x, error, marker="o")
    axes[0].set_title("Learning curve: prediction error")
    axes[0].set_ylabel("mean absolute curve error")
    axes[1].plot(x, completion, marker="o")
    axes[1].set_title("Learning curve: measured completion")
    axes[1].set_ylabel("completion probability")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.grid(alpha=0.25)
    fig.suptitle("Learning curve (missing n8/n16 points are intentionally blank)")
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=170)
    plt.close(fig)


def _plot_window(
    out: Path,
    window: Mapping[str, Any],
    level: Any,
    rho: Any,
    arm: Any,
    methods: Sequence[Any],
    method_labels: Mapping[str, str] | None = None,
) -> None:
    population_values: list[float] = []
    realization_values: list[float] = []
    labels: list[str] = []
    for method in methods:
        level_map = window.get(_key(level), {})
        rho_map = level_map.get(_key(rho), {}) if isinstance(level_map, Mapping) else {}
        arm_map = rho_map.get(_key(arm), {}) if isinstance(rho_map, Mapping) else {}
        record = arm_map.get(_key(method), {}) if isinstance(arm_map, Mapping) else {}
        population = record.get("population_curve_max_error") if isinstance(record, Mapping) else None
        realization = record.get("case_realization_discrepancy_max_error") if isinstance(record, Mapping) else None
        population_values.append(float(population) if isinstance(population, (int, float)) else np.nan)
        realization_values.append(float(realization) if isinstance(realization, (int, float)) else np.nan)
        labels.append(_method_label(method, method_labels))
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    ax.bar(x - width / 2, population_values, width, color="#8064a2", label="population curve max error")
    ax.bar(x + width / 2, realization_values, width, color="#c0504d", label="case realization discrepancy max")
    ax.set_title(f"Local window maximum error: level={level}, rho={rho}, arm={arm}")
    ax.set_ylabel("error (explicit population vs realization labels)")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.34, top=0.86)
    fig.savefig(out / "window_max_error.png", dpi=170)
    plt.close(fig)


def _plot_gamma(out: Path, gamma: Mapping[str, Any], levels: Sequence[Any], deadlines: Sequence[float]) -> None:
    records = [item for item in gamma.get("records", []) if isinstance(item, Mapping)]
    nrows = max(1, len(levels))
    fig, axes = plt.subplots(nrows, 1, squeeze=False, figsize=(9.0, 3.8 * nrows), sharex=True)
    x = np.asarray(deadlines, dtype=float)
    colors = plt.get_cmap("tab10")
    for level_index, level in enumerate(levels or ["none"]):
        ax = axes[level_index][0]
        if level == "none":
            ax.text(0.5, 0.5, "four_cells.json unavailable", ha="center", va="center")
            continue
        selected = [item for item in records if item.get("level") == level]
        if not selected:
            ax.text(0.5, 0.5, "four_cells.json unavailable for this level", ha="center", va="center")
        for color_index, item in enumerate(selected):
            rho = item.get("rho")
            run_id = item.get("run_id")
            suffix = f" {run_id}" if run_id else ""
            measured = _vector(item.get("gamma_measured"), len(x))
            predicted = _vector(item.get("gamma_predicted"), len(x))
            color = colors(color_index % 10)
            if measured is not None:
                ax.plot(x, measured, color=color, label=f"rho={rho:g}{suffix} measured" if isinstance(rho, float) else f"rho={rho}{suffix} measured")
            if predicted is not None:
                ax.plot(x, predicted, color=color, linestyle="--", alpha=0.75, label=f"rho={rho:g}{suffix} predicted" if isinstance(rho, float) else f"rho={rho}{suffix} predicted")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(f"level={level}")
        ax.set_ylabel("Gamma(T)")
        ax.grid(alpha=0.25)
        if selected:
            ax.legend(fontsize=8, ncol=2)
    axes[-1][0].set_xlabel("deadline (s)")
    fig.suptitle("D. Fixed-mode four-cell Gamma(T)")
    fig.tight_layout()
    fig.savefig(out / "D_Gamma_four_cell.png", dpi=170)
    plt.close(fig)


def _format_value(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _write_results(
    path: Path,
    input_dir: Path,
    protocol: Mapping[str, Any],
    run_status: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[Any],
    levels: Sequence[Any],
    rhos: Sequence[Any],
    arms: Sequence[Any],
    a_summary: Mapping[str, Any],
    b_summary: Mapping[str, Any],
    c_summary: Mapping[str, Any],
    learning: Mapping[str, Any],
    learning_by_level: Mapping[str, Any],
    window: Mapping[str, Any],
    failures: Mapping[str, Any],
    gamma: Mapping[str, Any],
    geometry_meta: Mapping[str, Any],
    audit: Mapping[str, Any],
    method_labels: Mapping[str, str],
    cost_curves_available: bool,
) -> None:
    rho_main = 0.8 if 0.8 in rhos else (rhos[0] if rhos else 0.0)
    arm_main = "retain_early" if "retain_early" in arms else (arms[0] if arms else "")
    lines = [
        "V8.5.2 core-suite 结果审计",
        "",
        f"输入：{input_dir}",
        f"schema：{protocol.get('schema', 'unknown')}",
        f"训练 seed：{protocol.get('combined_training_seeds', protocol.get('seed', 'unknown'))}",
        f"评测行数：{len(rows)}；方法：{_methods_text(methods, method_labels)}",
        f"level：{', '.join(map(str, levels)) or '(none)'}；rho：{', '.join(map(str, rhos)) or '(none)'}；arm：{', '.join(map(str, arms)) or '(none)'}",
        f"执行后端：{protocol.get('execution_backend', 'unknown')}；deadline/cost 的秒是执行得到的 Cartesian 模拟时钟秒，不是实机墙钟。",
        "",
        "统计口径",
        "- 独立统计单位是完整对象配置的 case_id；同一 case 的树、arm、重置和帧均先聚合，不当作额外样本。",
        "- 聚合使用 evaluation 中的 population_weight；它来自分层受控设计，而不是把 frame/call 数当作权重。",
        "- A 的曲线是每个预算 T 重新评价预算相关策略得到的完成率曲线；不能自动解释成一条固定策略的时间次 CDF。",
        "- 只有固定树/固定模式的介入才可按完成时间次分布解释；A 同时保留 predicted、measured、lower、upper 和事后树库参考的完整数字。",
        "- cost_curves 是这套 servo 模拟执行后统计的交互成本；不能当作物理机器人时间，也不与其他后端的时钟混合。",
        "- lower/upper 只作为描述性 envelope containment 报告，不能解释为 95% 置信覆盖。",
        f"- geometry_seed 去重数={geometry_meta.get('unique_geometry_seed_count')}；各 level 独立数={geometry_meta.get('per_level_independent_counts')}；run×case cluster 数={geometry_meta.get('n_run_case_clusters')}；跨 level 重合={geometry_meta.get('cross_level_shared_geometry_seed_counts')}。",
        "- 跨 level 合并时按 geometry_seed 聚类；同一 geometry_seed 在不同 pressure level 是配对敏感性条件，不应冒充新的独立解剖样本。",
        f"- 已读取 model_audit 文件={audit.get('n_model_audit_files')}、local_bundles 行={audit.get('n_local_bundle_rows')}（这些是模型/训练审计，不作为额外独立病例）。",
        f"- model audit 中 strict confidence-set 标记={audit.get('any_strict_confidence_set_flag')}；本报告仍不把 envelope 称为正式置信集合。",
        f"- 方法标签审计：{method_labels or '无 alias 标记'}；joint_nominal 若标为 robust alias 只作一致性重复，不作为 Bayes 点规划基线。",
        "",
        "A：完成窗口与预测误差",
    ]
    main_methods = list(methods)
    for level in levels:
        lines.append(f"level={level}：")
        for method in main_methods:
            record = _lookup_summary(a_summary, level, rho_main, arm_main, method)
            if not record:
                continue
            lines.append(
                f"- {_method_label(method, method_labels)}（rho={rho_main}, {arm_main}）："
                f"population_curve_mean_abs_error={_format_value(record.get('population_curve_mean_abs_error'))}，"
                f"population_curve_max_abs_error={_format_value(record.get('population_curve_max_abs_error'))}，"
                f"case_realization_discrepancy_mean={_format_value(record.get('case_realization_discrepancy_mean_abs_error'))}，"
                f"case_realization_discrepancy_max={_format_value(record.get('case_realization_discrepancy_max_abs_error'))}，"
                f"envelope containment={_format_value(record.get('descriptive_envelope_containment'))}，"
                f"case clusters={record.get('n_case_clusters', 0)}。"
            )
    lines.extend(["", "B：晚到信息的影响（release − retain）"])
    b_effect_count = 0
    b_not_distinguishable = 0
    for level in levels:
        lines.append(f"level={level}：")
        level_map = b_summary.get(_key(level), {})
        rho_map = level_map.get(_key(rho_main), {}) if isinstance(level_map, Mapping) else {}
        for method in methods:
            method_map = rho_map.get(_key(method), {}) if isinstance(rho_map, Mapping) else {}
            for release, value in method_map.items() if isinstance(method_map, Mapping) else []:
                if not isinstance(value, Mapping):
                    continue
                measured = _vector(value.get("measured_delta"))
                low = _vector(value.get("measured_bootstrap_low"))
                high = _vector(value.get("measured_bootstrap_high"))
                if measured is None:
                    continue
                b_effect_count += 1
                if low is None or high is None or np.any((low <= 0.0) & (high >= 0.0)):
                    b_not_distinguishable += 1
                final = measured[-1] if len(measured) else np.nan
                lines.append(
                    f"- {_method_label(method, method_labels)}, {release}：末端 measured delta={_format_value(final)}，"
                    f"case pairs={value.get('n_case_pairs', 0)}；200 次 cluster bootstrap 仅作描述性范围。"
                )
    if b_effect_count == 0:
        lines.append("- 没有可配对的 release/retain case，无法估计该效应。")
    elif b_not_distinguishable:
        lines.append(f"- {b_not_distinguishable}/{b_effect_count} 个 effect 的描述性范围跨过 0；本轮不能声称差异显著。")
    lines.extend(["", "C：同一完成目标 0.8 的首次预算与交互代价（探索性）"])
    def c_line(label: str, item: Mapping[str, Any] | None) -> str:
        if not item:
            return f"{label}=missing"
        population = _population_c_target(item)
        per_case = item.get("per_case_first_attainment", {}) if isinstance(item, Mapping) else {}
        cost = population.get("cost_at_population_target") if population else None
        return (
            f"{label} population_deadline={population.get('deadline') if population else 'unattainable'}，"
            f"attained_completion={population.get('attained_completion') if population else None}，"
            f"n_case_clusters={population.get('n_case_clusters') if population else 0}，"
            f"cost={cost if cost is not None else 'unavailable'}，"
            f"per_case_first_attainment_median={per_case.get('deadline_median', 'unavailable')}"
        )
    for level in levels:
        lines.append(f"level={level}：")
        for method in methods:
            retain = _c_lookup(c_summary, level, rho_main, "retain_early", method)
            hidden = _c_lookup(c_summary, level, rho_main, "hide_old", method)
            if not retain and not hidden:
                continue
            lines.append(f"- {_method_label(method, method_labels)}：{c_line('retain', retain)}；{c_line('hide_old', hidden)}。")
    if not cost_curves_available:
        lines.append("- 当前输入没有 cost_curves；旧 smoke 输出的 primary_costs 不能替代达到 0.8 时刻的曲线成本，因此成本写为 unavailable。")
    lines.extend(["", "学习曲线与局部误差"])
    for level in levels:
        level_learning = learning_by_level.get(_key(level), {})
        lines.append(f"level={level}（completion deadline={level_learning.get('completion_deadline_used')}）：")
        levels_learning = level_learning.get("levels", {})
        for label in ("n8", "n16", "nfull"):
            item = levels_learning.get(label, {})
            lines.append(
                f"- {label}：method={_method_label(item.get('method'), method_labels)}，available={item.get('available')}，"
                f"population_curve_error={item.get('population_curve_mean_abs_error', item.get('mean_abs_error_curve'))}，"
                f"completion={item.get('completion_measured')}。"
            )
    lines.append("- window_max_error 同时显示 population_curve_max_error 与 case_realization_discrepancy_max_error；二者都不是置信区间。")
    lines.extend(["", "失败、压力档与限制"])
    lines.append(
        f"- execution records={failures.get('n_execution_records')}，"
        f"failed records={failures.get('n_failed_execution_records')}，"
        f"call statuses={failures.get('call_status_counts')}，"
        f"recoveries={failures.get('total_recoveries')}，retries={failures.get('retry_calls')}。"
    )
    lines.append(f"- 压力档 execution counts={failures.get('stress_level_execution_counts')}。")
    lines.append(f"- predictions_unchanged={run_status.get('predictions_unchanged')}；real_hardware={run_status.get('real_hardware')}。")
    lines.append("- 本报告不把 retrospective_best_library 当作全球能力上界；它只是同一树库内的事后乐观参考。")
    lines.append("- full_history 是同 public interface/tree library 的强 baseline；当前 joint_nominal 按 method_label_audit 标为 robust alias/一致性重复，不是独立 Bayes 点规划基线。")
    lines.append("- C 主结果先按 population_weight 聚合 measured 完成率曲线，再取总体首次达到 0.8 的预算点、attained_completion 和同点成本；per_case_first_attainment 仅作为另存的描述性字段，不能替代总体目标。")
    lines.append("- 若总体目标未达到 0.8，population_target.deadline 为 unattainable；不做等完成率外推，也不强行单调化曲线。")
    lines.extend(["", "D：固定模式四格 Gamma(T)"])
    if gamma.get("available"):
        for item in gamma.get("records", []):
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- level={item.get('level')}, rho={item.get('rho')}："
                f"Gamma measured/predicted mean abs error={item.get('mean_abs_error')}，"
                f"max abs error={item.get('max_abs_error')}，"
                f"all four cells={item.get('all_four_cells_present')}。"
            )
    else:
        lines.append("- 当前输入没有 four_cells.json，Gamma 不能计算。")
    lines.append("- 这些结果是当前 finite-kernel/cartesian-lag 实验的执行审计，不等于真实 B-mode、接触或硬件保证；需在主运行和实测数据上复核。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report(
    input_dir: Path,
    output_dir: Path,
    *,
    rows_override: Sequence[Mapping[str, Any]] | None = None,
    protocol_override: Mapping[str, Any] | None = None,
    run_status_override: Mapping[str, Any] | None = None,
    gamma_override: Mapping[str, Any] | None = None,
    failures_override: Mapping[str, Any] | None = None,
    geometry_meta_override: Mapping[str, Any] | None = None,
    audit_override: Mapping[str, Any] | None = None,
    input_label: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = dict(protocol_override) if protocol_override is not None else _load_json(input_dir / "protocol.json", {})
    run_status = dict(run_status_override) if run_status_override is not None else _load_json(input_dir / "run_status.json", {})
    if rows_override is None:
        loaded_rows = _evaluation_rows(input_dir / "evaluation.json")
        if not loaded_rows:
            loaded_rows = _load_jsonl(input_dir / "evaluation.jsonl")
        rows, geometry_meta = _annotate_geometry(loaded_rows, input_dir)
    else:
        rows = [dict(row) for row in rows_override]
        geometry_meta = dict(geometry_meta_override or _geometry_metadata(rows))
    observed_levels = _ordered_unique(row.get("level") for row in rows if row.get("level") is not None)
    observed_rhos = _ordered_unique(row.get("rho") for row in rows if row.get("rho") is not None)
    observed_arms = _ordered_unique(row.get("arm") for row in rows if row.get("arm") is not None)
    observed_methods = _ordered_unique(row.get("method") for row in rows if row.get("method") is not None)
    levels = list(protocol.get("levels", {}).keys()) if isinstance(protocol.get("levels"), Mapping) else observed_levels
    levels = [x for x in levels if x in observed_levels] or observed_levels
    rhos = list(protocol.get("population_rhos", observed_rhos))
    rhos = [x for x in rhos if x in observed_rhos] or observed_rhos
    arms_from_protocol = [item.get("name") for item in protocol.get("arms", []) if isinstance(item, Mapping)]
    arms = [x for x in arms_from_protocol if x in observed_arms] or observed_arms
    methods = sorted(observed_methods, key=str)
    n = max((len(row.get("measured", [])) for row in rows if isinstance(row.get("measured"), list)), default=0)
    raw_deadlines = protocol.get("deadlines", [])
    deadlines = [float(x) for x in raw_deadlines] if isinstance(raw_deadlines, list) and len(raw_deadlines) == n else [float(i) for i in range(n)]
    a_summary = _build_a_summary(rows, levels, rhos, arms, methods, deadlines)
    baseline_arm = "retain_early" if "retain_early" in arms else (str(arms[0]) if arms else "")
    release_arms = [arm for arm in ("release_3s", "release_8s") if arm in arms]
    b_summary = _build_b_effects(rows, levels, rhos, methods, deadlines, baseline_arm, release_arms)
    c_summary = _build_c_summary(rows, levels, rhos, arms, methods, deadlines, TARGET_COMPLETION)
    rho_main = 0.8 if 0.8 in rhos else (rhos[0] if rhos else 0.0)
    level_main = levels[0] if levels else ""
    learning_by_level = {
        _key(level): _build_learning_summary(
            rows, methods, level, baseline_arm, rho_main, deadlines, completion_deadline=30.0
        )
        for level in levels
    }
    learning = learning_by_level.get(_key(level_main), _build_learning_summary(
        rows, methods, level_main, baseline_arm, rho_main, deadlines, completion_deadline=30.0
    ))
    window = _build_window_summary(rows, levels, rhos, arms, methods, deadlines)
    failures = dict(failures_override) if failures_override is not None else _failure_summary(input_dir, run_status)
    gamma = dict(gamma_override) if gamma_override is not None else _four_cell_gamma_summary(input_dir, deadlines)
    audit = dict(audit_override) if audit_override is not None else _audit_artifact_summary(input_dir)
    method_labels = _method_labels_from_audit(audit)
    cost_curves_available = any(isinstance(row.get("cost_curves"), list) for row in rows)

    summary: dict[str, Any] = {
        "schema": "v852_core_suite_report_v1",
        "input": str(input_dir),
        "protocol_schema": protocol.get("schema"),
        "training_seed": protocol.get("seed"),
        "training_seeds": protocol.get("combined_training_seeds", [protocol.get("seed")] if protocol.get("seed") is not None else []),
        "n_evaluation_rows": len(rows),
        "levels": levels,
        "rhos": rhos,
        "arms": arms,
        "methods": methods,
        "method_display_labels": method_labels,
        "deadlines": deadlines,
        "execution_backend": protocol.get("execution_backend"),
        "clock_semantics": "Cartesian simulation clock seconds from executed servo model; not physical robot wall-clock seconds",
        "independent_unit": protocol.get("independent_unit", "case_id cluster"),
        "geometry": geometry_meta,
        "population_weight_source": protocol.get("population_design"),
        "cost_curves_available": cost_curves_available,
        "target_completion": TARGET_COMPLETION,
        "a_cdf": a_summary,
        "b_release_effect": b_summary,
        "c_equal_target": c_summary,
        "learning_curve": learning,
        "learning_curve_by_level": learning_by_level,
        "window_max_error": window,
        "four_cell_gamma": gamma,
        "audit_artifacts": audit,
        "failures": failures,
        "interpretation_limits": {
            "envelope_is_formal_confidence": False,
            "retrospective_best_is_global_upper_bound": False,
            "full_history_is_strong_baseline": True,
            "joint_nominal_is_finite_posterior_reference": False,
            "joint_nominal_is_robust_alias_in_this_run": bool(method_labels.get("joint_nominal")),
            "retrospective_best_is_same_tree_library_oracle": True,
            "finite_kernel_is_real_hardware": False,
            "cluster_bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_results(
        output_dir / "RESULTS.txt",
        input_label or input_dir,
        protocol,
        run_status,
        rows,
        methods,
        levels,
        rhos,
        arms,
        a_summary,
        b_summary,
        c_summary,
        learning,
        learning_by_level,
        window,
        failures,
        gamma,
        geometry_meta,
        audit,
        method_labels,
        cost_curves_available,
    )
    main_plot_methods = [
        method
        for method in ("joint_robust", "full_history", "shuffled_robust", "identify_execute")
        if method in methods
    ] or list(methods)
    b_plot_methods = [method for method in ("joint_robust", "full_history") if method in methods] or main_plot_methods
    b_plot_arms = ["release_8s"] if "release_8s" in release_arms else release_arms
    _plot_a(output_dir, a_summary, levels, rhos, main_plot_methods, baseline_arm, deadlines, method_labels)
    _plot_b(output_dir, b_summary, levels, rho_main, b_plot_methods, b_plot_arms, deadlines, method_labels)
    _plot_c(
        output_dir,
        c_summary,
        levels,
        rho_main,
        methods,
        [arm for arm in ("retain_early", "hide_old") if arm in arms] or arms,
        method_labels,
    )
    _plot_learning(output_dir, learning)
    _plot_window(output_dir, window, level_main, rho_main, baseline_arm, methods, method_labels)
    _plot_gamma(output_dir, gamma, levels, deadlines)
    return summary


def _combine_failure_summaries(
    input_dirs: Sequence[Path], run_statuses: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    parts = [_failure_summary(path, status) for path, status in zip(input_dirs, run_statuses)]
    status_counts: Counter[str] = Counter()
    stress_counts: Counter[str] = Counter()
    for part in parts:
        status_counts.update(part.get("call_status_counts", {}))
        stress_counts.update(part.get("stress_level_execution_counts", {}))
    return {
        "n_execution_records": sum(int(part.get("n_execution_records", 0)) for part in parts),
        "n_failed_execution_records": sum(int(part.get("n_failed_execution_records", 0)) for part in parts),
        "call_status_counts": dict(status_counts),
        "stress_level_execution_counts": dict(stress_counts),
        "total_recoveries": sum(int(part.get("total_recoveries", 0)) for part in parts),
        "retry_calls": sum(int(part.get("retry_calls", 0)) for part in parts),
        "per_run": parts,
        "run_status": {"runs": list(run_statuses)},
    }


def _combine_gamma(input_dirs: Sequence[Path], deadlines: Sequence[float]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for run_index, path in enumerate(input_dirs, start=1):
        item = _four_cell_gamma_summary(path, deadlines)
        per_run.append({"run_id": f"run{run_index}", **item})
        for record in item.get("records", []):
            if isinstance(record, Mapping):
                records.append({"run_id": f"run{run_index}", **dict(record)})
    return {
        "available": bool(records),
        "n_records": len(records),
        "records": records,
        "per_run": per_run,
        "definition": "Gamma(T)=cell(+1,+1)+cell(-1,-1)-cell(+1,-1)-cell(-1,+1)",
        "fixed_mode_intervention": True,
        "combined_gamma_is_descriptive_run_average": True,
    }


def _combine_audit_artifacts(input_dirs: Sequence[Path]) -> dict[str, Any]:
    parts = [_audit_artifact_summary(path) for path in input_dirs]
    files: list[dict[str, Any]] = []
    level_rows: Counter[str] = Counter()
    design_levels: dict[str, dict[str, Any]] = {}
    strict = False
    method_label_audits: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        for item in part.get("model_audit_files", []):
            files.append({"run_id": f"run{index}", **dict(item)})
        level_rows.update(part.get("local_bundle_level_rows", {}))
        strict = strict or bool(part.get("any_strict_confidence_set_flag", False))
        method_label_audits.append({"run_id": f"run{index}", **dict(part.get("method_label_audit", {}))})
        design = part.get("test_manifest_design", {})
        for level, level_item in (design.get("by_level", {}) if isinstance(design, Mapping) else {}).items():
            target = design_levels.setdefault(
                str(level), {"n_cases": 0, "uv_counts": Counter(), "chain_counts": Counter(), "uv_chain_counts": Counter()}
            )
            target["n_cases"] += int(level_item.get("n_cases", 0))
            for field in ("uv_counts", "chain_counts", "uv_chain_counts"):
                target[field].update(level_item.get(field, {}))
    for level_item in design_levels.values():
        for field in ("uv_counts", "chain_counts", "uv_chain_counts"):
            level_item[field] = dict(level_item[field])
    return {
        "model_audit_files": files,
        "n_model_audit_files": len(files),
        "n_local_bundle_rows": sum(int(part.get("n_local_bundle_rows", 0)) for part in parts),
        "n_local_bundle_objects": sum(int(part.get("n_local_bundle_objects", 0)) for part in parts),
        "local_bundle_level_rows": dict(level_rows),
        "any_strict_confidence_set_flag": strict,
        "strict_confidence_set_flags_are_descriptive": True,
        "method_label_audit": {
            "per_run": method_label_audits,
            "joint_nominal_is_robust_alias": all(
                "same robust objective" in str(item.get("joint_nominal_actual_objective", ""))
                for item in method_label_audits
            ) if method_label_audits else False,
        },
        "test_manifest_design": {
            "n_rows": sum(int(part.get("test_manifest_design", {}).get("n_rows", 0)) for part in parts),
            "by_level": design_levels,
            "purpose": "request x UV allocation audit; not an additional independent-sample count",
        },
        "per_run": parts,
    }


def run_combined(input_dirs: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    """Render a combined report for independent training seeds.

    Evaluation clusters are prefixed by ``runN::case_id``.  Thus a case from
    two independent training seeds cannot silently collapse during the normal
    case-cluster aggregation.  Geometry metadata separately reports the
    de-duplicated ``geometry_seed`` count needed when pressure levels are
    pooled for a sensitivity analysis.
    """

    if len(input_dirs) < 2:
        raise ValueError("combined report needs at least two input run directories")
    loaded: list[tuple[Path, Mapping[str, Any], Mapping[str, Any], list[dict[str, Any]]]] = []
    all_rows: list[dict[str, Any]] = []
    run_geometry: list[dict[str, Any]] = []
    for index, path in enumerate(input_dirs, start=1):
        protocol = _load_json(path / "protocol.json", {})
        status = _load_json(path / "run_status.json", {})
        raw_rows = _evaluation_rows(path / "evaluation.json")
        if not raw_rows:
            raw_rows = _load_jsonl(path / "evaluation.jsonl")
        rows, geometry = _annotate_geometry(raw_rows, path, run_id=f"run{index}")
        loaded.append((path, protocol, status, rows))
        all_rows.extend(rows)
        run_geometry.append({"run_id": f"run{index}", **geometry})
    if not all_rows:
        raise ValueError("combined report has no evaluation rows")
    backends = {str(protocol.get("execution_backend", "unknown")) for _, protocol, _, _ in loaded}
    if len(backends) > 1:
        raise ValueError(
            "cannot combine runs with different execution backends/clocks: "
            + ", ".join(sorted(backends))
        )
    first_protocol = dict(loaded[0][1])
    first_protocol["independent_unit"] = (
        "run_id x case_id cluster; two independent training seeds; "
        "cross-level pooled sensitivity uses geometry_seed clusters"
    )
    first_protocol["combined_training_seeds"] = [status.get("seed", protocol.get("seed")) for _, protocol, status, _ in loaded]
    # run_status.json normally does not repeat the seed; use protocol seed.
    first_protocol["combined_run_count"] = len(loaded)
    statuses = [status for _, _, status, _ in loaded]
    combined_status: dict[str, Any] = {
        "runs": [dict(status) for status in statuses],
        "predictions_unchanged": all(bool(status.get("predictions_unchanged", False)) for status in statuses),
        "real_hardware": any(bool(status.get("real_hardware", False)) for status in statuses),
        "training_seeds": [protocol.get("seed") for _, protocol, _, _ in loaded],
    }
    geometry_meta = _geometry_metadata(all_rows)
    geometry_meta["per_run"] = run_geometry
    geometry_meta["unique_training_seed_count"] = len(
        {str(protocol.get("seed")) for _, protocol, _, _ in loaded if protocol.get("seed") is not None}
    )
    geometry_meta["training_seeds"] = [protocol.get("seed") for _, protocol, _, _ in loaded]
    geometry_meta["note"] = (
        "Within-level run x case clusters are independent across seeds; the same geometry_seed "
        "reappearing in another pressure level is paired sensitivity, not a new anatomy."
    )
    deadlines = first_protocol.get("deadlines", [])
    deadlines = [float(x) for x in deadlines] if isinstance(deadlines, list) else []
    gamma = _combine_gamma(input_dirs, deadlines)
    failures = _combine_failure_summaries(input_dirs, statuses)
    audit = _combine_audit_artifacts(input_dirs)
    label = Path("combined(" + ", ".join(f"run{i+1}:{path}" for i, path in enumerate(input_dirs)) + ")")
    return run_report(
        input_dirs[0],
        output_dir,
        rows_override=all_rows,
        protocol_override=first_protocol,
        run_status_override=combined_status,
        gamma_override=gamma,
        failures_override=failures,
        geometry_meta_override=geometry_meta,
        audit_override=audit,
        input_label=label,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path, help="one core-suite output directory")
    inputs.add_argument("--combine-input", nargs="+", type=Path, help="two or more independent run directories")
    parser.add_argument("--output", required=True, type=Path, help="report/figure directory")
    args = parser.parse_args(argv)
    if args.combine_input:
        run_combined(args.combine_input, args.output)
    else:
        run_report(args.input, args.output)
    print(f"wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

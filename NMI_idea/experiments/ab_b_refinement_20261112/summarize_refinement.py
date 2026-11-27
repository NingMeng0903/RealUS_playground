"""Summarize paired refinement shards and render a portable trajectory view.

The refinement runner writes one ``RUNS.jsonl`` per shard.  This module keeps
aggregation deliberately independent of the runner: it discovers arm names
from the supplied rows, pairs only complete ``(seed, arm)`` observations, and
never uses evaluation-only labels for a planning or divergence claim.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
GEOMETRY = HERE.parent / "ab_leg_dynamic_20261112" / "DYNAMIC_GEOMETRY_FROZEN.npz"
CAPTURE_BUDGETS = (40, 50, 80)
PATH_BUDGETS_M = (2.5, 3.0)
BOOTSTRAP_SAMPLES = 8000


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite values for JSON output."""

    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (float, np.floating)):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _same_number(left: Any, right: Any, tolerance: float = 1.0e-9) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    except (TypeError, ValueError):
        return left == right


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float, np.number)) and isinstance(right, (int, float, np.number)):
        return _same_number(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


def _read_shard(path: Path) -> tuple[list[dict], dict]:
    """Read one shard directory, accepting a direct RUNS.jsonl path too."""

    path = Path(path)
    runs_path = path if path.name == "RUNS.jsonl" else path / "RUNS.jsonl"
    if not runs_path.is_file():
        raise FileNotFoundError(f"missing RUNS.jsonl in shard {path}")
    rows = []
    with runs_path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {runs_path}:{line_number}") from exc
            if not isinstance(row, dict) or "seed" not in row or "arm" not in row:
                raise ValueError(f"row at {runs_path}:{line_number} lacks seed/arm")
            rows.append(row)
    config_path = runs_path.with_name("CONFIG.json")
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    return rows, config


def collect(runs: Sequence[Path]) -> tuple[list[dict], dict, dict[str, dict[str, dict]]]:
    """Collect rows and index them by seed and arm.

    Identical duplicate rows are harmless when a caller repeats a shard.  A
    conflicting duplicate is rejected rather than silently changing a paired
    result.
    """

    rows: list[dict] = []
    configs: dict[str, dict] = {}
    cases: dict[str, dict[str, dict]] = defaultdict(dict)
    for shard in runs:
        shard_rows, config = _read_shard(Path(shard))
        configs[str(Path(shard))] = config
        for row in shard_rows:
            seed = str(row["seed"])
            arm = str(row["arm"])
            existing = cases[seed].get(arm)
            if existing is not None:
                if existing != row:
                    raise ValueError(f"conflicting duplicate for seed={seed!r}, arm={arm!r}")
                continue
            cases[seed][arm] = row
            rows.append(row)
    return rows, configs, cases


def _steps(row: Mapping[str, Any]) -> list[dict]:
    return [step for step in row.get("steps", []) if isinstance(step, dict)]


def _supplemental_captures(row: Mapping[str, Any]) -> int:
    return len(_steps(row))


def _total_captures(row: Mapping[str, Any]) -> int:
    return int(row.get("prefix_acquisitions", 0)) + _supplemental_captures(row)


def _centroid_count(row: Mapping[str, Any]) -> int:
    return sum(len(step.get("centroids_mm", []) or []) for step in _steps(row))


def _label_count(row: Mapping[str, Any], *, clutter: bool) -> int:
    count = 0
    for step in _steps(row):
        labels = step.get("eval_only_labels")
        if not isinstance(labels, (list, tuple)):
            continue
        count += sum((label < 0) if clutter else (label >= 0) for label in labels)
    return count


def _path_m(row: Mapping[str, Any]) -> float:
    if "recheck_path_mm" in row:
        return float(row["recheck_path_mm"]) / 1000.0
    steps = _steps(row)
    if not steps:
        return 0.0
    if "cumulative_m" in steps[-1]:
        return float(steps[-1]["cumulative_m"])
    return float(sum(float(step.get("length_m", 0.0)) for step in steps))


def _same_site_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    sites = []
    for step in _steps(row):
        action = step.get("action", [])
        sites.append(action[0] if action else None)
    same_site_steps = sum(a is not None and b == a for a, b in zip(sites, sites[1:]))
    prior_sites: set[Any] = set()
    returns = 0
    for site in sites:
        if site in prior_sites:
            returns += 1
        if site is not None:
            prior_sites.add(site)
    denominator = max(0, len(sites) - 1)
    return {
        "steps": len(sites),
        "same_site_adjacent_steps": int(same_site_steps),
        "same_site_return_steps": int(returns),
        "same_site_return_rate": float(returns / denominator) if denominator else 0.0,
    }


def _change_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    steps = _steps(row)
    changes = [step for step in steps if bool(step.get("changed_suffix", False))]
    tails = [step for step in steps if bool(step.get("tail", False))]
    gains = []
    predicted = []
    for step in steps:
        choice = step.get("tail_choice")
        if isinstance(choice, dict):
            if isinstance(choice.get("gain"), (int, float)):
                gains.append(float(choice["gain"]))
            if isinstance(choice.get("predicted_gain"), (int, float)):
                predicted.append(float(choice["predicted_gain"]))
    return {
        "actual_suffix_changes": len(changes),
        "changed_suffix_steps": len(changes),
        "tail_steps": len(tails),
        "rows_with_actual_suffix_change": int(bool(changes)),
        "change_rate_over_tails": float(len(changes) / len(tails)) if tails else 0.0,
        "tail_gain_mean": float(np.mean(gains)) if gains else None,
        "tail_predicted_gain_mean": float(np.mean(predicted)) if predicted else None,
    }


def _arm_summary(rows: Sequence[dict]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    complete = np.asarray([bool(row.get("complete", False)) for row in rows], dtype=bool)
    online = np.asarray([bool(row.get("online_stop", False)) for row in rows], dtype=bool)
    false_stops = online & ~complete
    wrong_counts = np.asarray([len(row.get("incorrect_witnesses", []) or []) for row in rows], dtype=float)
    max_errors = np.asarray([float(row.get("map_max_error_mm", np.nan)) for row in rows])
    rmses = np.asarray([float(row.get("map_rmse_mm", np.nan)) for row in rows])
    bands = np.asarray([float(row.get("max_band_mm", np.nan)) for row in rows])
    captures = np.asarray([_supplemental_captures(row) for row in rows], dtype=float)
    total_captures = np.asarray([_total_captures(row) for row in rows], dtype=float)
    centroid_captures = np.asarray([_centroid_count(row) for row in rows], dtype=float)
    clutter = np.asarray([_label_count(row, clutter=True) for row in rows], dtype=float)
    genuine = np.asarray([_label_count(row, clutter=False) for row in rows], dtype=float)
    paths = np.asarray([_path_m(row) for row in rows], dtype=float)
    changes = [_change_metrics(row) for row in rows]
    sites = [_same_site_metrics(row) for row in rows]

    def finite_mean(values: np.ndarray) -> float | None:
        values = values[np.isfinite(values)]
        return float(values.mean()) if len(values) else None

    completion_curve = {
        str(limit): int(sum(bool(row.get("complete", False)) and _supplemental_captures(row) <= limit for row in rows))
        for limit in CAPTURE_BUDGETS
    }
    path_curve = {
        str(limit): int(sum(bool(row.get("complete", False)) and _path_m(row) <= limit + 1.e-12 for row in rows))
        for limit in PATH_BUDGETS_M
    }
    same_site = {
        "adjacent_steps": int(sum(item["same_site_adjacent_steps"] for item in sites)),
        "return_steps": int(sum(item["same_site_return_steps"] for item in sites)),
        "rows_with_return": int(sum(item["same_site_return_steps"] > 0 for item in sites)),
        "return_rate": float(sum(item["same_site_return_steps"] for item in sites) / max(1, sum(item["steps"] - 1 for item in sites))),
    }
    return {
        "count": len(rows),
        "arm_count": len(rows),
        "strict_successes": int(complete.sum()),
        "complete": int(complete.sum()),
        "online_stops": int(online.sum()),
        "online_stop": int(online.sum()),
        "false_stops": int(false_stops.sum()),
        "false_stop": int(false_stops.sum()),
        "not_stopped_by_cap": int((~online).sum()),
        "witness_error_breakdown": {
            "rows_with_wrong_witness": int((wrong_counts > 0).sum()),
            "wrong_witness_rows": int((wrong_counts > 0).sum()),
            "wrong_witness_targets": int(wrong_counts.sum()),
            "rows_over_band": int(np.sum(max_errors > 3.0)),
            "rows_over_2band": int(np.sum(max_errors > 2.0 * bands)),
            "mean_wrong_witness_targets": finite_mean(wrong_counts),
            "mean_max_error_mm": finite_mean(max_errors),
            "mean_rmse_mm": finite_mean(rmses),
            "mean_max_band_mm": finite_mean(bands),
        },
        "error_breakdown": {
            "wrong_witness_rows": int((wrong_counts > 0).sum()),
            "wrong_witness_targets": int(wrong_counts.sum()),
            "max_error_mm_mean": finite_mean(max_errors),
            "max_error_mm_p95": float(np.nanpercentile(max_errors, 95)) if np.isfinite(max_errors).any() else None,
            "rmse_mm_mean": finite_mean(rmses),
        },
        "captures": {
            "supplemental_mean": finite_mean(captures),
            "supplemental_total": int(captures.sum()),
            "total_mean": finite_mean(total_captures),
            "centroid_mean": finite_mean(centroid_captures),
            "centroid_total": int(centroid_captures.sum()),
            "genuine_centroid_total": int(genuine.sum()),
            "clutter_centroid_total": int(clutter.sum()),
            "completion_at_supplemental_budget": completion_curve,
            "completion_at_capture_budget": completion_curve,
            "completion_at_total_capture_budget": {
                str(limit): int(sum(bool(row.get("complete", False)) and _total_captures(row) <= limit for row in rows))
                for limit in CAPTURE_BUDGETS
            },
        },
        "travel": {
            "mean_m": finite_mean(paths),
            "total_m": float(paths[np.isfinite(paths)].sum()) if np.isfinite(paths).any() else 0.0,
            "max_m": float(np.nanmax(paths)) if np.isfinite(paths).any() else None,
            "completion_at_path_budget_m": path_curve,
        },
        "same_site_return": same_site,
        "same_site": same_site,
        "actual_suffix_changes": {
            "total": int(sum(item["actual_suffix_changes"] for item in changes)),
            "rows": int(sum(item["rows_with_actual_suffix_change"] for item in changes)),
            "tail_steps": int(sum(item["tail_steps"] for item in changes)),
            "rate_over_tails": float(sum(item["actual_suffix_changes"] for item in changes) / max(1, sum(item["tail_steps"] for item in changes))),
            "mean_gain": float(np.mean([item["tail_gain_mean"] for item in changes if item["tail_gain_mean"] is not None])) if any(item["tail_gain_mean"] is not None for item in changes) else None,
            "mean_predicted_gain": float(np.mean([item["tail_predicted_gain_mean"] for item in changes if item["tail_predicted_gain_mean"] is not None])) if any(item["tail_predicted_gain_mean"] is not None for item in changes) else None,
        },
        "completion_curve": completion_curve,
        "path_completion_curve_m": path_curve,
        "completion_at_joint_budgets": {
            str(captures): {
                str(distance): sum(
                    bool(row["complete"]) and _supplemental_captures(row) <= captures
                    and _path_m(row) <= distance + 1.e-12 for row in rows
                )
                for distance in PATH_BUDGETS_M
            }
            for captures in CAPTURE_BUDGETS
        },
    }


def _paired_rows(cases: Mapping[str, Mapping[str, dict]], left: str, right: str) -> list[tuple[str, dict, dict]]:
    pairs = []
    for seed, arms in sorted(cases.items()):
        if left not in arms or right not in arms:
            continue
        first, second = arms[left], arms[right]
        # A split-arm run is pairable only when both arms consumed the same
        # shared public prefix.  Older rows may omit the field, so absence on
        # one side does not make an otherwise usable pair incomplete.
        p_first, p_second = first.get("prefix_acquisitions"), second.get("prefix_acquisitions")
        if p_first is not None and p_second is not None and p_first != p_second:
            continue
        pairs.append((seed, first, second))
    return pairs


def _paired_difference(cases: Mapping[str, Mapping[str, dict]], left: str, right: str) -> dict[str, Any]:
    pairs = _paired_rows(cases, left, right)
    if not pairs:
        return {"left": left, "right": right, "paired_cases": 0, "status": "unavailable"}
    differences = np.asarray([
        int(bool(a.get("complete", False))) - int(bool(b.get("complete", False)))
        for _, a, b in pairs
    ], dtype=float)
    wins = int(np.sum(differences > 0))
    losses = int(np.sum(differences < 0))
    ties = int(len(differences) - wins - losses)
    generator = np.random.default_rng(7191)
    # Keep memory bounded for pooled shard sets while retaining the old
    # deterministic 8,000-resample convention.
    boot = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for start in range(0, BOOTSTRAP_SAMPLES, 512):
        stop = min(start + 512, BOOTSTRAP_SAMPLES)
        boot[start:stop] = generator.choice(
            differences, size=(stop - start, len(differences))
        ).mean(axis=1)
    non_ties = wins + losses
    # Exact two-sided sign test without depending on a particular SciPy API.
    if non_ties:
        k = min(wins, losses)
        exact = min(1.0, float(2 * sum(math.comb(non_ties, j) for j in range(k + 1))) / float(1 << non_ties))
    else:
        exact = 1.0
    left_steps = [tuple((step.get("pose"), tuple(step.get("action", []))) for step in a.get("steps", [])) for _, a, _ in pairs]
    right_steps = [tuple((step.get("pose"), tuple(step.get("action", []))) for step in b.get("steps", [])) for _, _, b in pairs]
    identical = sum(x == y for x, y in zip(left_steps, right_steps))
    left_caps = np.asarray([_supplemental_captures(a) for _, a, _ in pairs], dtype=float)
    right_caps = np.asarray([_supplemental_captures(b) for _, _, b in pairs], dtype=float)
    left_paths = np.asarray([_path_m(a) for _, a, _ in pairs], dtype=float)
    right_paths = np.asarray([_path_m(b) for _, _, b in pairs], dtype=float)
    return {
        "left": left,
        "right": right,
        "paired_cases": len(pairs),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "difference": float(differences.mean()),
        "bootstrap_95": np.percentile(boot, [2.5, 97.5]).tolist(),
        "exact_two_sided_p": float(exact),
        "identical_action_sequences": int(identical),
        "mean_supplemental_capture_difference": float((left_caps - right_caps).mean()),
        "mean_path_difference_m": float((left_paths - right_paths).mean()),
    }


def _public_step(step: Mapping[str, Any]) -> dict[str, Any]:
    """Keep divergence evidence strictly public (no eval-only labels)."""

    choice = step.get("tail_choice")
    score = step.get("score")
    if score is None and isinstance(choice, dict):
        score = choice.get("score")
    return {
        "pose": step.get("pose"),
        "action": step.get("action"),
        "tail": bool(step.get("tail", False)),
        "centroids_mm": step.get("centroids_mm", []),
        "cumulative_m": step.get("cumulative_m"),
        "evidence_gains": step.get("evidence_gains", []),
        "score": score,
        "tail_choice": {
            key: choice.get(key)
            for key in ("locked", "usable", "gain", "predicted_gain")
            if isinstance(choice, dict) and key in choice
        },
    }


def _first_divergence(cases: Mapping[str, Mapping[str, dict]], live: str, locked: str) -> dict[str, Any]:
    pairs = _paired_rows(cases, live, locked)
    records = []
    for seed, live_row, locked_row in pairs:
        live_steps, locked_steps = _steps(live_row), _steps(locked_row)
        prefix = 0
        while prefix < min(len(live_steps), len(locked_steps)):
            left, right = live_steps[prefix], locked_steps[prefix]
            same = all(_same_value(left.get(key), right.get(key)) for key in ("pose", "action", "centroids_mm", "cumulative_m"))
            if not same:
                break
            prefix += 1
        diverged = prefix < min(len(live_steps), len(locked_steps)) or len(live_steps) != len(locked_steps)
        record: dict[str, Any] = {
            "seed": seed,
            "paired": True,
            "diverged": bool(diverged),
            "identical_prefix_steps": prefix,
            "live_steps": len(live_steps),
            "locked_steps": len(locked_steps),
            "locked_trace_matches_actual": None,
        }
        if diverged:
            live_step = live_steps[prefix] if prefix < len(live_steps) else None
            locked_step = locked_steps[prefix] if prefix < len(locked_steps) else None
            record["first_divergence_index"] = prefix
            record["live"] = _public_step(live_step) if live_step is not None else None
            record["locked"] = _public_step(locked_step) if locked_step is not None else None
            record["distinct_executed_tail"] = {
                "live_actions": [step.get("action") for step in live_steps[prefix:]],
                "locked_actions": [step.get("action") for step in locked_steps[prefix:]],
                "action_sequences_distinct": [
                    step.get("action") for step in live_steps[prefix:]
                ] != [step.get("action") for step in locked_steps[prefix:]],
            }
            live_public = [_public_step(step) for step in live_steps[prefix:]]
            locked_public = [_public_step(step) for step in locked_steps[prefix:]]
            record["public_evidence"] = {
                "live_evidence_gains": [step.get("evidence_gains", []) for step in live_public],
                "locked_evidence_gains": [step.get("evidence_gains", []) for step in locked_public],
                "live_score_deltas": [step.get("tail_choice", {}).get("gain") for step in live_public],
                "locked_score_deltas": [step.get("tail_choice", {}).get("gain") for step in locked_public],
                "live_scores": [step.get("score") for step in live_public],
                "locked_scores": [step.get("score") for step in locked_public],
                "live_predicted_gains": [step.get("tail_choice", {}).get("predicted_gain") for step in live_public],
                "locked_predicted_gains": [step.get("tail_choice", {}).get("predicted_gain") for step in locked_public],
                "public_trace_only": True,
                "no_lookahead": True,
            }
            trace = live_step.get("tail_choice") if isinstance(live_step, dict) else None
            if isinstance(trace, dict) and locked_step is not None:
                record["locked_trace_matches_actual"] = _same_value(trace.get("locked"), locked_step.get("action"))
            elif trace is not None:
                record["locked_trace_matches_actual"] = False
        records.append(record)
    diverged_records = [r for r in records if r["diverged"]]
    match_values = [r["locked_trace_matches_actual"] for r in diverged_records if r["locked_trace_matches_actual"] is not None]
    return {
        "live": live,
        "locked": locked,
        "paired_cases": len(pairs),
        "diverged_cases": len(diverged_records),
        "identical_prefix_cases": len(pairs) - len(diverged_records),
        "locked_trace_matches": int(sum(match_values)),
        "locked_trace_mismatches": int(sum(not value for value in match_values)),
        "distinct_executed_tail_cases": int(sum(
            bool(record.get("distinct_executed_tail", {}).get("action_sequences_distinct", False))
            for record in diverged_records
        )),
        "records": records,
        "public_only": True,
        "no_lookahead_claim": "divergence report uses executed public trace fields only; no hidden suffix is inferred",
    }


def _trajectory(row: Mapping[str, Any]) -> list[list[float]]:
    points = []
    for step in _steps(row):
        for point in step.get("path_m", []) or []:
            try:
                if len(point) == 3 and all(math.isfinite(float(v)) for v in point):
                    points.append([1000.0 * float(v) for v in point])
            except (TypeError, ValueError):
                continue
    return points


def _geometry_payload() -> dict[str, Any]:
    """Load trunk/branch templates from the previous dynamic geometry."""

    try:
        sys.path.insert(0, str(GEOMETRY.parent))
        from dynamic_scan import DynamicScan  # type: ignore

        with np.load(GEOMETRY) as data:
            scan = DynamicScan(data)
            pose = int(scan.bent)
            return {
                "pose": pose,
                "trunk": (1000.0 * np.asarray(scan.trunk[pose])).tolist(),
                "branch": (1000.0 * np.asarray(scan.dense_branch[pose])).tolist(),
            }
    except Exception as exc:  # pragma: no cover - fallback is for portable report use
        try:
            with np.load(GEOMETRY) as data:
                trunk = 1000.0 * np.asarray(data["posed_vessel_m"][0])
            return {"pose": 0, "trunk": trunk.tolist(), "branch": [], "geometry_warning": str(exc)}
        except Exception:
            return {"pose": None, "trunk": [], "branch": [], "geometry_warning": str(exc)}


def _choose_sample(cases: Mapping[str, Mapping[str, dict]], preferred: tuple[str, str] = ("B", "locked")) -> dict[str, Any]:
    pair_choices = [preferred, ("B_once", "locked"), ("B", "old_fixed")]
    # Prefer the first actual strict winner that also changed an executed
    # suffix, then any actual trajectory change.  A deterministic fallback is
    # explicit in the HTML when no such case exists.
    for left, right in pair_choices:
        for seed, arms in sorted(cases.items()):
            if left in arms and right in arms:
                a, b = arms[left], arms[right]
                if (
                    bool(a.get("complete", False))
                    and not bool(b.get("complete", False))
                    and _change_metrics(a)["actual_suffix_changes"] > 0
                ):
                    return {"seed": seed, "winner": left, "reason": "first strict winner with actual suffix change", "arms": {left: a, right: b}}
    for left, right in pair_choices:
        for seed, arms in sorted(cases.items()):
            if left in arms and right in arms and _trajectory(arms[left]) != _trajectory(arms[right]):
                return {"seed": seed, "winner": "trajectory-change", "reason": "first actual trajectory change", "arms": {left: arms[left], right: arms[right]}}
    left, right = preferred
    for seed, arms in sorted(cases.items()):
        if left in arms or right in arms:
            names = [name for name in (left, right) if name in arms]
            if len(names) == 1 and "B_once" in arms:
                names = ["B_once", names[0]]
            return {"seed": seed, "winner": "deterministic sample", "reason": "no changed winner available", "arms": {name: arms[name] for name in names}}
    return {"seed": None, "winner": None, "reason": "no paired rows", "arms": {}}


def _html_report(path: Path, summary: Mapping[str, Any], cases: Mapping[str, Mapping[str, dict]]) -> None:
    sample = _choose_sample(cases)
    arms = sample.get("arms", {})
    sample_payload = {
        "seed": sample.get("seed"),
        "winner": sample.get("winner"),
        "reason": sample.get("reason"),
        "arms": {
            name: {
                "complete": bool(row.get("complete", False)),
                "online_stop": bool(row.get("online_stop", False)),
                "steps": _supplemental_captures(row),
                "path_m": _path_m(row),
                "rmse_mm": row.get("map_rmse_mm"),
                "max_error_mm": row.get("map_max_error_mm"),
                "changes": _change_metrics(row)["actual_suffix_changes"],
                "trajectory": _trajectory(row),
            }
            for name, row in arms.items()
        },
    }
    payload = {"geometry": _geometry_payload(), "sample": sample_payload,
               "arms": sorted(summary.get("arms", {}).keys()),
               "paired_differences": summary.get("paired_differences", {})}
    data = json.dumps(_jsonable(payload), separators=(",", ":"))
    escaped_title = "Refinement rechecks · paired trajectories"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped_title}</title>
<style>
body{{margin:0;font:14px system-ui,sans-serif;color:#16202a;background:#f7f9fb}}
main{{max-width:1280px;margin:auto;padding:18px}} h1{{font-size:21px;margin:0 0 4px}}
.note{{color:#53616d;margin-bottom:12px}} .grid{{display:grid;grid-template-columns:minmax(560px,2fr) minmax(320px,1fr);gap:14px}}
section{{background:#fff;border:1px solid #d9e0e6;border-radius:9px;padding:12px;box-shadow:0 2px 8px #0b22330d}}
canvas{{width:100%;height:560px;background:#101923;border-radius:6px;display:block;cursor:grab}} canvas:active{{cursor:grabbing}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:6px 5px;border-bottom:1px solid #e8edf0;text-align:right}} th:first-child,td:first-child{{text-align:left}}
.legend{{display:flex;gap:15px;flex-wrap:wrap;margin:8px 0;color:#40515e}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
pre{{white-space:pre-wrap;max-height:180px;overflow:auto;background:#f3f6f8;padding:8px;border-radius:5px;font-size:11px}}
label{{display:inline-flex;gap:6px;align-items:center;margin:6px 12px 4px 0}}
@media(max-width:900px){{.grid{{grid-template-columns:1fr}}canvas{{height:420px}}}}
</style></head><body><main><h1>{escaped_title}</h1>
<div class="note">Sample: <span id="sample"></span>. Drag to orbit; wheel changes scale. Geometry is the previous dynamic trunk/branch template. The report is self-contained.</div>
<div class="grid"><section><canvas id="view" width="900" height="560"></canvas><div class="legend">
<span><i class="dot" style="background:#5bc0be"></i>trunk</span><span><i class="dot" style="background:#f0a35b"></i>branch</span>
<span><i class="dot" style="background:#e05d6f"></i>B</span><span><i class="dot" style="background:#5b8def"></i>locked</span></div>
<label>Yaw <input id="yaw" type="range" min="-3.14" max="3.14" step=".01" value=".35"></label>
<label>Pitch <input id="pitch" type="range" min="-1.4" max="1.4" step=".01" value=".35"></label>
<label>Scale <input id="scale" type="range" min=".4" max="3" step=".01" value="1"></label></section>
<section><h2>Paired resource comparison</h2><div id="resource"></div><h2>Report provenance</h2><pre id="meta"></pre></section></div></main>
<script>
const DATA={data};
const canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
const yaw=document.getElementById('yaw'),pitch=document.getElementById('pitch'),scale=document.getElementById('scale');
const sample=DATA.sample; document.getElementById('sample').textContent=`${{sample.seed ?? 'none'}} · ${{sample.reason}}`;
const palette={{B:'#e05d6f',locked:'#5b8def',B_once:'#d979bc',old_fixed:'#8996a3',trunk:'#5bc0be',branch:'#f0a35b'}};
function project(p){{let y=+yaw.value,z=+pitch.value,s=+scale.value;let x=p[0],yy=p[1],zz=p[2];let a=Math.cos(y)*x-Math.sin(y)*yy,b=Math.sin(y)*x+Math.cos(y)*yy;let c=Math.cos(z)*b-Math.sin(z)*zz,d=Math.sin(z)*b+Math.cos(z)*zz;return [a*s,d*s,c];}}
function allPoints(){{let out=[];for(const key of ['trunk','branch'])for(const p of (DATA.geometry[key]||[]))out.push(project(p));for(const row of Object.values(sample.arms||{{}}))for(const p of row.trajectory||[])out.push(project(p));return out;}}
function drawLine(points,color,width,factor){{if(!points.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>{{let x=canvas.width/2+p[0]*factor,y=canvas.height/2-p[1]*factor+(p[2]||0)*factor*.18;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);}});ctx.stroke();}}
function draw(){{ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);let pts=allPoints();let mx=Math.max(1,...pts.map(p=>Math.abs(p[0])),...pts.map(p=>Math.abs(p[1])));let factor=220/mx;drawLine((DATA.geometry.trunk||[]).map(project),'#5bc0be',2.5,factor);drawLine((DATA.geometry.branch||[]).map(project),'#f0a35b',2.5,factor);for(const [name,row] of Object.entries(sample.arms||{{}})drawLine((row.trajectory||[]).map(project),palette[name]||'#d5dce2',2.2,factor);}}
for(const input of [yaw,pitch,scale])input.addEventListener('input',draw);draw();
let dragging=false,last=[0,0];canvas.addEventListener('pointerdown',e=>{{dragging=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)}});canvas.addEventListener('pointermove',e=>{{if(!dragging)return;yaw.value=+yaw.value+(e.clientX-last[0])*.008;pitch.value=Math.max(-1.4,Math.min(1.4,+pitch.value+(e.clientY-last[1])*.008));last=[e.clientX,e.clientY];draw()}});canvas.addEventListener('pointerup',()=>dragging=false);
const resource=document.getElementById('resource');let rows='<table><tr><th>arm</th><th>strict</th><th>online</th><th>captures</th><th>travel m</th><th>changes</th><th>RMSE mm</th></tr>';for(const [name,row] of Object.entries(sample.arms||{{}}))rows+=`<tr><td>${{name}}</td><td>${{row.complete}}</td><td>${{row.online_stop}}</td><td>${{row.steps}}</td><td>${{Number(row.path_m).toFixed(3)}}</td><td>${{row.changes}}</td><td>${{Number(row.rmse_mm).toFixed(3)}}</td></tr>`;resource.innerHTML=rows+'</table>';
document.getElementById('meta').textContent=JSON.stringify({{seed:sample.seed,reason:sample.reason,geometry_pose:DATA.geometry.pose,paired_differences:DATA.paired_differences}},null,2);
</script></body></html>"""
    path.write_text(html)


def summarize(rows: Sequence[dict], cases: Mapping[str, Mapping[str, dict]], configs: Mapping[str, dict]) -> dict[str, Any]:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_arm[str(row["arm"])].append(row)
    result: dict[str, Any] = {
        "cases": len(cases),
        "rows": len(rows),
        "arm_names": sorted(by_arm),
        "incomplete_case_count": int(sum(len(arms) < len(by_arm) for arms in cases.values())),
        "arms": {arm: _arm_summary(sorted(arm_rows, key=lambda row: str(row.get("seed")))) for arm, arm_rows in sorted(by_arm.items())},
        "paired_cases_by_comparison": {},
        "paired_differences": {},
        "first_divergence": {},
        "capture_budgets": list(CAPTURE_BUDGETS),
        "path_budgets_m": list(PATH_BUDGETS_M),
        "shards": sorted(configs),
    }
    comparisons = [
        ("B", "locked", "B_vs_locked"),
        ("B", "old_fixed", "B_vs_old_fixed"),
        ("B", "B_no_fisher", "B_vs_B_no_fisher"),
        ("B_once", "locked", "B_once_vs_locked"),
        ("B", "A", "B_vs_A"),
        ("B_seen", "locked_seen", "B_seen_vs_locked_seen"),
        ("locked_seen", "locked", "locked_seen_vs_locked"),
        ("B_guarded", "B", "B_guarded_vs_B"),
    ]
    for left, right, key in comparisons:
        paired = _paired_rows(cases, left, right)
        result["paired_cases_by_comparison"][key] = len(paired)
        result["paired_differences"][key] = _paired_difference(cases, left, right)
    result["paired_B"] = {
        "locked": result["paired_differences"]["B_vs_locked"],
        "old_fixed": result["paired_differences"]["B_vs_old_fixed"],
        "B_no_fisher": result["paired_differences"]["B_vs_B_no_fisher"],
    }
    result["paired_B_once"] = {"locked": result["paired_differences"]["B_once_vs_locked"]}
    result["first_divergence"]["B_vs_locked"] = _first_divergence(cases, "B", "locked")
    result["first_divergence"]["B_once_vs_locked"] = _first_divergence(cases, "B_once", "locked")
    if "B_seen" in by_arm and "locked_seen" in by_arm:
        result["first_divergence"]["B_seen_vs_locked_seen"] = _first_divergence(cases, "B_seen", "locked_seen")
    # Keep explicit aliases convenient for audit scripts that consume the
    # requested pair names directly.
    result["first_divergence_B_locked"] = result["first_divergence"]["B_vs_locked"]
    result["first_divergence_B_once_locked"] = result["first_divergence"]["B_once_vs_locked"]
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="one or more shard directories")
    parser.add_argument("--out", type=Path, required=True, help="summary/output directory")
    args = parser.parse_args(argv)
    rows, configs, cases = collect(args.runs)
    if not rows:
        raise ValueError("no run rows found")
    summary = summarize(rows, cases, configs)
    sample = _choose_sample(cases)
    summary["visualization_sample"] = {
        "seed": sample.get("seed"),
        "winner": sample.get("winner"),
        "reason": sample.get("reason"),
        "arms": sorted(sample.get("arms", {}).keys()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "SUMMARY.json").write_text(json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n")
    _html_report(args.out / "REFINEMENT_3D.html", summary, cases)
    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Summarize paired closed-loop result files without frame-level pseudoreplication.

The runner writes one ``RESULT.json`` per fixture.  This module deliberately
keeps raw packet counts separate from subject-level inference.  Each case is
first averaged over repeated layouts or repeats, then the case IDs are the
resampling units for the paired bootstrap.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import re
import tempfile
from typing import Any, Iterable

import numpy as np


BOOTSTRAP_SEED = 721421
BOOTSTRAP_SAMPLES = 10_000
ARMS = (
    "I0_F0", "I0_F1", "I1_F0", "B_I1_F1", "A_greedy", "pair_fixed",
    "same_side_I0", "same_side_I1", "same_pair_I0", "same_pair_I1",
)
NUMERIC_FIELDS = (
    "success", "success_with_planning_charged", "correct_stop", "first_success_s",
    "first_success_with_planning_s", "restricted_time_s", "sim_seconds", "wall_planning_s",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _case_id(row: dict[str, Any], path: Path, index: int) -> str | None:
    for key in ("case_id", "subject_id", "scenario_id", "object_id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return None


def _layout(row: dict[str, Any], fixture: dict[str, Any], path: Path) -> str:
    for source in (row, fixture):
        for key in ("layout", "fixture", "layout_name", "name"):
            value = source.get(key) if isinstance(source, dict) else None
            if value is not None and str(value) != "":
                return str(value)
    return path.parent.name


def _variant(row: dict[str, Any]) -> str:
    return str(row.get("variant", "primary"))


def _arm(row: dict[str, Any]) -> str:
    return str(row.get("arm", row.get("method", "unknown")))


def _events(row: dict[str, Any]) -> list[dict[str, Any]]:
    events = row.get("events", [])
    return [event for event in events if isinstance(event, dict)] if isinstance(events, list) else []


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "pass", "passed", "correct"}:
            return True
        if normalized in {"false", "no", "0", "fail", "failed", "incorrect"}:
            return False
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
        return bool(value)
    return None


def _planning_charged_success(record: dict[str, Any]) -> bool | None:
    explicit = _as_bool_or_none(record.get("success_with_planning_charged"))
    if explicit is not None:
        return explicit
    if not bool(record.get("success", False)):
        return False
    deadline = _finite(record.get("deadline_s"))
    planning = _finite(record.get("wall_planning_s"))
    elapsed = _finite(record.get("first_success_with_planning_s"))
    if elapsed is not None:
        return bool(elapsed <= deadline + 1e-12)
    elapsed = _finite(record.get("first_success_s"))
    if elapsed is None:
        elapsed = _finite(record.get("restricted_time_s"))
    if elapsed is None or deadline is None or planning is None:
        return None
    return bool(elapsed + planning <= deadline + 1e-12)


def _correct_stop(record: dict[str, Any]) -> bool | None:
    for key in ("correct_stop", "correct_public_stop", "submitted_correct", "correct_submission"):
        parsed = _as_bool_or_none(record.get(key))
        if parsed is not None:
            return parsed
    return None


def _candidate_path(value: Any, source: str) -> Path | None:
    if not isinstance(value, str) or not value or not value.lower().endswith(".npz"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(source).parent / path
    path = path.resolve()
    return path if path.is_file() else None


def _motion_manifest(source: str) -> Path | None:
    result = Path(source).resolve()
    candidates = (
        result.parent / "execution" / "MOTION.json",
        result.parent / "MOTION.json",
        result.parent.parent / "execution" / "MOTION.json",
    )
    return next((path for path in candidates if path.is_file()), None)


def _route_references(event: dict[str, Any], source: str) -> list[dict[str, Any]]:
    route_id = event.get("route_id")
    if not isinstance(route_id, str) or not route_id:
        return []
    manifest = _motion_manifest(source)
    if manifest is None:
        return [{"source": source, "route_id": route_id, "matched": False}]
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        return [{"source": source, "route_id": route_id, "motion_manifest": str(manifest), "matched": False,
                 "manifest_error": str(exc)}]
    routes = payload.get("routes", []) if isinstance(payload, dict) else []
    matched = [route for route in routes if isinstance(route, dict) and str(route.get("route", "")) == route_id]
    if not matched:
        return [{"source": source, "route_id": route_id, "motion_manifest": str(manifest), "matched": False}]
    references = []
    for route in matched:
        paths = []
        for step in route.get("steps", []) if isinstance(route.get("steps", []), list) else []:
            if not isinstance(step, dict):
                continue
            record = step.get("record")
            if not isinstance(record, str):
                continue
            candidate = (manifest.parent / f"{record}.npz").resolve()
            if candidate.is_file():
                paths.append(str(candidate))
        references.append({"source": source, "route_id": route_id, "motion_manifest": str(manifest),
                           "matched": True, "npz_paths": paths})
    return references


def _reference_values(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """Return references already present in the input record.

    The summarizer never constructs a trace path.  It records only explicit
    fields or existing event references supplied by the runner.
    """

    keys = ("trace_file", "trace_path", "record", "result_file", "result_path", "path")
    values: list[dict[str, Any]] = []
    for key in keys:
        candidate = _candidate_path(row.get(key), source)
        if candidate is not None:
            values.append({"source": source, "reference": str(candidate)})
    for event in _events(row):
        for key in keys:
            candidate = _candidate_path(event.get(key), source)
            if candidate is not None:
                values.append({"source": source, "reference": str(candidate)})
        values.extend(_route_references(event, source))
    return values


def _unique_references(references: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for reference in references:
        try:
            key = json.dumps(reference, sort_keys=True, ensure_ascii=False)
        except TypeError:
            key = str(reference)
        if key not in seen:
            seen.add(key)
            output.append(reference)
    return output


def _top_level_metrics(payload: dict[str, Any], source: str) -> dict[str, Any]:
    """Collect execution counters without assuming one runner spelling."""

    aliases = {
        "native_executions": ("native_executions", "native_execution_count", "native_runs",
                               "unique_native_execution_routes", "distinct_native_executions"),
        "cache_hits": ("cache_hits", "execution_cache_hits", "cached_executions",
                        "identical_state_cache_hits"),
        "distinct_capture_states": ("distinct_capture_states", "distinct_executed_capture_states"),
        "native_cache_misses": ("native_cache_misses", "cache_misses"),
        "dird_queries": ("dird_queries", "dird_query_count"),
    }
    result: dict[str, Any] = {"source": source}
    nested = payload.get("execution", {}) if isinstance(payload.get("execution"), dict) else {}
    nested_counts = payload.get("execution_counts", {}) if isinstance(payload.get("execution_counts"), dict) else {}
    for name, keys in aliases.items():
        value = None
        for key in keys:
            if key in payload:
                value = _finite(payload[key])
                if value is not None:
                    break
            if key in nested:
                value = _finite(nested[key])
                if value is not None:
                    break
            if key in nested_counts:
                value = _finite(nested_counts[key])
                if value is not None:
                    break
        if value is not None:
            result[name] = value
    return result


def load_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = [root] if root.is_file() else sorted(root.rglob("RESULT.json"))
    rows: list[dict[str, Any]] = []
    diagnostics = {"result_files": len(files), "skipped_files": [], "skipped_records": [], "references": [],
                   "top_level_execution_metrics": []}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics["skipped_files"].append({"path": str(path), "reason": f"json:{exc}"})
            continue
        fixture = payload.get("fixture", {}) if isinstance(payload, dict) else {}
        if not isinstance(fixture, dict):
            fixture = {}
        diagnostics["top_level_execution_metrics"].append(_top_level_metrics(payload, str(path)))
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            diagnostics["skipped_files"].append({"path": str(path), "reason": "records_not_list"})
            continue
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                diagnostics["skipped_records"].append({"path": str(path), "index": index, "reason": "record_not_object"})
                continue
            case_id = _case_id(record, path, index)
            if case_id is None:
                diagnostics["skipped_records"].append({"path": str(path), "index": index, "reason": "missing_case_id"})
                continue
            source = str(path)
            normalized = {
                "case_id": case_id,
                "layout": _layout(record, fixture, path),
                "variant": _variant(record),
                "arm": _arm(record),
                "success": bool(record.get("success", False)),
                "success_with_planning_charged": _planning_charged_success(record),
                "correct_stop": _correct_stop(record),
                "first_success_s": _finite(record.get("first_success_s")),
                "first_success_with_planning_s": _finite(record.get("first_success_with_planning_s")),
                "restricted_time_s": _finite(record.get("restricted_time_s")),
                "sim_seconds": _finite(record.get("sim_seconds")),
                "wall_planning_s": _finite(record.get("wall_planning_s")),
                "first_choice": str(record.get("first_choice", "unknown")),
                "terminal_capture_count": int(record.get("terminal_capture_count", 0) or 0),
                "misses": int(record.get("misses", 0) or 0),
                "clutter": int(record.get("clutter", 0) or 0),
                "false_stop": bool(record.get("false_stop", False)),
                "reason": str(record.get("reason", "")),
                "events": _events(record),
                "source": source,
                "references": _reference_values(record, source),
            }
            rows.append(normalized)
            diagnostics["references"].extend(normalized["references"])
    diagnostics["references"] = _unique_references(diagnostics["references"])
    return rows, diagnostics


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(numbers)) if numbers else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated layouts within a case before subject-level pairing."""

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["layout"], row["variant"], row["arm"])].append(row)
    layout_rows: list[dict[str, Any]] = []
    for (case_id, layout, variant, arm), group in sorted(groups.items()):
        item: dict[str, Any] = {"case_id": case_id, "layout": layout, "variant": variant, "arm": arm,
                                "replicate_count": len(group), "success": _mean([float(x["success"]) for x in group]),
                                "success_with_planning_charged": _mean(x["success_with_planning_charged"] for x in group),
                                "correct_stop": _mean(x["correct_stop"] for x in group),
                                "first_choice": Counter(x["first_choice"] for x in group).most_common(1)[0][0],
                                "terminal_capture_count": _mean(x["terminal_capture_count"] for x in group),
                                "misses": _mean(x["misses"] for x in group), "clutter": _mean(x["clutter"] for x in group),
                                "false_stop": _mean([float(x["false_stop"]) for x in group]),
                                "references": [ref for x in group for ref in x["references"]]}
        for field in ("first_success_s", "first_success_with_planning_s", "restricted_time_s", "sim_seconds", "wall_planning_s"):
            item[field] = _mean(x[field] for x in group)
        layout_rows.append(item)
    return layout_rows


def aggregate_cases(layout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average layout-level rows so each case contributes one paired unit."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in layout_rows:
        groups[(row["case_id"], row["variant"], row["arm"])].append(row)
    case_rows: list[dict[str, Any]] = []
    for (case_id, variant, arm), group in sorted(groups.items()):
        item: dict[str, Any] = {"case_id": case_id, "variant": variant, "arm": arm,
                                "layout_count": len(group), "replicate_count": sum(x["replicate_count"] for x in group),
                                "success": _mean(x["success"] for x in group),
                                "success_with_planning_charged": _mean(x["success_with_planning_charged"] for x in group),
                                "correct_stop": _mean(x["correct_stop"] for x in group),
                                "first_choice": Counter(x["first_choice"] for x in group).most_common(1)[0][0],
                                "terminal_capture_count": _mean(x["terminal_capture_count"] for x in group),
                                "misses": _mean(x["misses"] for x in group), "clutter": _mean(x["clutter"] for x in group),
                                "false_stop": _mean(x["false_stop"] for x in group),
                                "references": [ref for x in group for ref in x["references"]]}
        for field in ("first_success_s", "first_success_with_planning_s", "restricted_time_s", "sim_seconds", "wall_planning_s"):
            item[field] = _mean(x[field] for x in group)
        case_rows.append(item)
    return case_rows


def _index_cases(case_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {(row["case_id"], row["variant"], row["arm"]): row for row in case_rows}


def _paired_values(index: dict[tuple[str, str, str], dict[str, Any]], left: str, right: str, variant: str,
                   field: str = "success") -> tuple[list[str], np.ndarray, np.ndarray]:
    cases = sorted({key[0] for key in index if key[1] == variant and key[2] == left}
                   & {key[0] for key in index if key[1] == variant and key[2] == right})
    a = [index[(case, variant, left)].get(field) for case in cases]
    b = [index[(case, variant, right)].get(field) for case in cases]
    keep = [i for i, (x, y) in enumerate(zip(a, b)) if _finite(x) is not None and _finite(y) is not None]
    return [cases[i] for i in keep], np.asarray([float(a[i]) for i in keep]), np.asarray([float(b[i]) for i in keep])


def _bootstrap_difference(index: dict[tuple[str, str, str], dict[str, Any]], left: str, right: str,
                          variant: str, field: str, seed: int, samples: int) -> dict[str, Any]:
    cases, a, b = _paired_values(index, left, right, variant, field)
    diff = a - b
    result: dict[str, Any] = {"left": left, "right": right, "variant": variant, "field": field,
                              "estimate": float(np.mean(diff)) if len(diff) else None, "n_pairs": len(diff),
                              "wins": int(np.sum(diff > 0)), "losses": int(np.sum(diff < 0)), "ties": int(np.sum(diff == 0)),
                              "case_ids": cases, "bootstrap_seed": seed, "bootstrap_samples": samples}
    if not len(diff):
        result["reason"] = "no_paired_cases"
        return result
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(samples, len(diff)))
    boot = diff[draws].mean(axis=1)
    result["ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    return result


def _bootstrap_cross_variant(index: dict[tuple[str, str, str], dict[str, Any]], arm: str,
                             left_variant: str, right_variant: str, field: str,
                             seed: int, samples: int) -> dict[str, Any]:
    cases = sorted({key[0] for key in index if key[1] == left_variant and key[2] == arm}
                   & {key[0] for key in index if key[1] == right_variant and key[2] == arm})
    left = np.asarray([index[(case, left_variant, arm)].get(field) for case in cases], dtype=object)
    right = np.asarray([index[(case, right_variant, arm)].get(field) for case in cases], dtype=object)
    keep = [i for i, (a, b) in enumerate(zip(left, right)) if _finite(a) is not None and _finite(b) is not None]
    diff = np.asarray([float(left[i]) - float(right[i]) for i in keep], dtype=float)
    result: dict[str, Any] = {"arm": arm, "left_variant": left_variant, "right_variant": right_variant,
                              "field": field, "estimate": float(np.mean(diff)) if len(diff) else None,
                              "n_pairs": len(diff), "wins": int(np.sum(diff > 0)),
                              "losses": int(np.sum(diff < 0)), "ties": int(np.sum(diff == 0)),
                              "case_ids": [cases[i] for i in keep], "bootstrap_seed": seed,
                              "bootstrap_samples": samples}
    if len(diff):
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(diff), size=(samples, len(diff)))
        boot = diff[draws].mean(axis=1)
        result["ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    else:
        result["reason"] = "no_paired_cases"
    return result


def _interaction(index: dict[tuple[str, str, str], dict[str, Any]], variant: str = "primary") -> dict[str, Any]:
    required = ("B_I1_F1", "I0_F1", "I1_F0", "I0_F0")
    cases = sorted(set.intersection(*[{key[0] for key in index if key[1] == variant and key[2] == arm} for arm in required])) if all(any(key[1] == variant and key[2] == arm for key in index) for arm in required) else []
    values = []
    for case in cases:
        values.append(sum((1 if arm in ("B_I1_F1", "I0_F0") else -1) * float(index[(case, variant, arm)]["success"]) for arm in required))
    arr = np.asarray(values, dtype=float)
    result: dict[str, Any] = {"formula": "(I1F1-I0F1)-(I1F0-I0F0)", "variant": variant,
                              "estimate": float(np.mean(arr)) if len(arr) else None, "n_pairs": len(arr),
                              "case_ids": cases, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES}
    if len(arr):
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = rng.integers(0, len(arr), size=(BOOTSTRAP_SAMPLES, len(arr)))
        boot = arr[draws].mean(axis=1)
        result["ci95"] = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
    else:
        result["reason"] = "missing_four_cells"
    return result


def _same_first_contrasts(index: dict[tuple[str, str, str], dict[str, Any]], variant: str,
                          seed: int, samples: int) -> dict[str, Any]:
    """Find every ``same_<program>_I1``/``same_<program>_I0`` pair.

    Program names are deliberately discovered from the input.  This keeps
    side-left, side-right, fork-minus, fork-plus, and pair-order records
    separate when the runner adds them, while retaining the older aggregate
    same-side and same-pair labels below.
    """

    programs: set[str] = set()
    pattern = re.compile(r"^same_(.+)_I([01])$")
    for case_id, row_variant, arm in index:
        if row_variant != variant:
            continue
        match = pattern.match(arm)
        if match:
            programs.add(match.group(1))
    return {
        program: _bootstrap_difference(index, f"same_{program}_I1", f"same_{program}_I0",
                                       variant, "success", seed, samples)
        for program in sorted(programs)
        if any(key[1] == variant and key[2] == f"same_{program}_I1" for key in index)
        and any(key[1] == variant and key[2] == f"same_{program}_I0" for key in index)
    }


def _execution_totals(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    names = ("native_executions", "cache_hits", "native_cache_misses", "distinct_capture_states", "dird_queries")
    totals: dict[str, Any] = {"files_with_metrics": 0, "files": len(metrics), "totals": {}}
    for name in names:
        values = [entry[name] for entry in metrics if _finite(entry.get(name)) is not None]
        if values:
            totals["totals"][name] = float(np.sum(values))
    totals["files_with_metrics"] = sum(1 for entry in metrics if any(name in entry for name in names))
    totals["distinct_native_execution_definition"] = "sum of explicit per-RESULT native execution counters"
    totals["cache_hit_definition"] = "sum of explicit per-RESULT cache-hit counters"
    if not totals["totals"]:
        totals["reason"] = "top_level_execution_counters_absent"
    return totals


def _elapsed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['variant']}::{row['arm']}"].append(row)
    output: dict[str, Any] = {}
    for label, group in sorted(grouped.items()):
        fields = {}
        for field in ("first_success_s", "first_success_with_planning_s", "restricted_time_s", "sim_seconds", "wall_planning_s"):
            values = [float(x[field]) for x in group if x.get(field) is not None and math.isfinite(float(x[field]))]
            if values:
                fields[field] = {"n": len(values), "mean": float(np.mean(values)),
                                 "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95))}
        output[label] = fields
    return output


def _summary_counts(rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rates(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"records": len(items), "case_units": len({x["case_id"] for x in items}),
                "success_mean": _mean(x["success"] for x in items),
                "success_with_planning_charged_mean": _mean(x["success_with_planning_charged"] for x in items),
                "correct_stop_mean": _mean(x["correct_stop"] for x in items),
                "false_stop_mean": _mean(x["false_stop"] for x in items),
                "miss_mean": _mean(x["misses"] for x in items), "clutter_mean": _mean(x["clutter"] for x in items)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['variant']}::{row['arm']}"] .append(row)
    return {key: rates(value) for key, value in sorted(grouped.items())}


def _first_choices(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = defaultdict(dict)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[f"{row['variant']}::{row['arm']}"][row["first_choice"]] += 1
    return {key: dict(sorted(counter.items())) for key, counter in sorted(counts.items())}


def _event_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = Counter()
    hidden = Counter()
    accepted = Counter()
    for row in rows:
        for event in row["events"]:
            count["events"] += 1
            if event.get("capture") is not None:
                count[f"capture:{event.get('capture')}"] += 1
            if event.get("accepted") is True:
                accepted["accepted"] += 1
            hidden_kind = event.get("hidden_kind")
            if hidden_kind is not None:
                hidden[str(hidden_kind)] += 1
    return {"raw_event_counts": dict(sorted(count.items())), "accepted_count": int(accepted["accepted"]),
            "private_audit_hidden_kind_counts": dict(sorted(hidden.items()))}


def _examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["references"]:
            by_case[row["case_id"]].append(row)
    positive = []
    reversal = []
    def pack(case: str, group: list[dict[str, Any]]) -> dict[str, Any]:
        # Keep a bounded arm sample plus deduplicated trace sources.  The raw
        # RESULT files remain the per-record source of truth; copying every
        # event reference into SUMMARY.json makes it unnecessarily large.
        records = [{"arm": x["arm"], "variant": x["variant"], "success": x["success"]}
                   for x in group[:24]]
        references = _unique_references(ref for x in group for ref in x["references"])
        return {"case_id": case, "records": records, "references": references}
    for case, group in sorted(by_case.items()):
        if any(x["success"] for x in group) and not any(not x["success"] for x in group):
            positive.append(pack(case, group))
        if any(x["success"] for x in group) and any(not x["success"] for x in group):
            reversal.append(pack(case, group))
    return {"positive": positive[:3], "reversal": reversal[:3], "available": bool(positive or reversal)}


def summarize(root: Path, *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES) -> dict[str, Any]:
    rows, diagnostics = load_rows(root)
    layout_rows = aggregate_rows(rows)
    case_rows = aggregate_cases(layout_rows)
    index = _index_cases(case_rows)
    contrasts: dict[str, Any] = {}
    companion_pairs = (
        ("B_minus_I1F0", "B_I1_F1", "I1_F0", "primary"),
        ("B_minus_I0F1", "B_I1_F1", "I0_F1", "primary"),
        ("B_minus_A_greedy", "B_I1_F1", "A_greedy", "primary"),
        ("B_minus_A_geometry", "B_I1_F1", "A_geometry", "primary"),
        ("B_minus_pair_fixed", "B_I1_F1", "pair_fixed", "primary"),
        ("B_minus_response_reset", "B_I1_F1", "response_reset", "primary"),
        ("same_side_information", "same_side_I1", "same_side_I0", "primary"),
        ("same_pair_information", "same_pair_I1", "same_pair_I0", "primary"),
    )
    for label, left, right, variant in companion_pairs:
        contrasts[label] = _bootstrap_difference(index, left, right, variant, "success", seed, samples)
    contrasts["I_by_F_interaction"] = _interaction(index, "primary")
    variants = sorted({row["variant"] for row in case_rows})
    same_first = {variant: _same_first_contrasts(index, variant, seed, samples) for variant in variants}
    response_reset = {}
    if "primary" in variants and "response_reset" in variants:
        shared_arms = sorted({key[2] for key in index if key[1] == "primary"}
                             & {key[2] for key in index if key[1] == "response_reset"})
        response_reset = {
            arm: _bootstrap_cross_variant(index, arm, "primary", "response_reset", "success", seed, samples)
            for arm in shared_arms
        }
    planning_contrasts = {
        label: _bootstrap_difference(index, left, right, variant, "success_with_planning_charged", seed, samples)
        for label, left, right, variant in companion_pairs
        if label.startswith("B_minus_") or label in {"same_side_information", "same_pair_information"}
    }
    planning_same_first = {}
    for variant in variants:
        stems = _same_first_contrasts(index, variant, seed, samples)
        planning_same_first[variant] = {}
        for stem in stems:
            planning_same_first[variant][stem] = _bootstrap_difference(
                index, f"same_{stem}_I1", f"same_{stem}_I0", variant,
                "success_with_planning_charged", seed, samples,
            )
    controls = {}
    for variant in variants:
        controls[variant] = {
            label: _bootstrap_difference(index, left, right, variant, "success", seed, samples)
            for label, left, right in (
                ("B_minus_I1F0", "B_I1_F1", "I1_F0"),
                ("B_minus_I0F1", "B_I1_F1", "I0_F1"),
                ("B_minus_A_greedy", "B_I1_F1", "A_greedy"),
                ("B_minus_A_geometry", "B_I1_F1", "A_geometry"),
                ("B_minus_pair_fixed", "B_I1_F1", "pair_fixed"),
                ("B_minus_response_reset", "B_I1_F1", "response_reset"),
            )
        }
    # Case rows are the paired inference units.  Keep them aggregate-only;
    # trace references are retained once in diagnostics and in bounded native
    # examples, while raw RESULT.json remains available for full replay.
    compact_case_rows = [{key: value for key, value in row.items() if key != "references"}
                         for row in case_rows]
    return {
        "schema": "ab_closed_loop_summary_v1", "bootstrap": {"seed": seed, "samples": samples, "unit": "case_id"},
        "input": {"root": str(root), "diagnostics": diagnostics, "raw_records": len(rows),
                   "layout_units": len(layout_rows), "case_units": len(case_rows),
                   "unique_case_ids": len({x["case_id"] for x in case_rows}),
                   "unique_layouts": len({x["layout"] for x in layout_rows}),
                   "variants": sorted({x["variant"] for x in rows}), "arms": sorted({x["arm"] for x in rows})},
        "primary_contrasts": contrasts, "control_contrasts": controls,
        "same_first_contrasts": same_first,
        "response_reset_contrasts": response_reset,
        "planning_charged_sensitivity": planning_contrasts,
        "planning_charged_same_first": planning_same_first,
        "stratified": _summary_counts(rows, case_rows), "first_action_distribution": _first_choices(rows),
        "elapsed_summary": _elapsed_summary(rows),
        "execution_audit": _execution_totals(diagnostics["top_level_execution_metrics"]),
        "event_audit": _event_audit(rows), "examples_from_input": _examples(rows),
        "case_units": compact_case_rows,
    }


def _self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="ab-summary-") as temp:
        root = Path(temp)
        execution = root / "execution"
        execution.mkdir()
        (execution / "MOTION.json").write_text(json.dumps({"routes": [{"route": "r1", "steps": [{"record": "step1"}]}]}), encoding="utf-8")
        (execution / "step1.npz").write_bytes(b"npz")
        records = [
            {"case_id": 1, "variant": "primary", "arm": "B_I1_F1", "success": True, "success_with_planning_charged": True, "correct_stop": True, "first_choice": "side", "events": [{"capture": "terminal", "accepted": True, "route_id": "r1"}]},
            {"case_id": 1, "variant": "primary", "arm": "I1_F0", "success": False, "first_choice": "side", "events": [{"route_id": "r1"}]},
            {"case_id": 1, "variant": "primary", "arm": "A_geometry", "success": False, "first_choice": "direct"},
            {"case_id": 1, "variant": "primary", "arm": "A_greedy", "success": False, "first_choice": "direct"},
            {"case_id": 1, "variant": "primary", "arm": "pair_fixed", "success": False, "first_choice": "pair"},
            {"case_id": 1, "variant": "primary", "arm": "response_reset", "success": False, "first_choice": "reset"},
            {"case_id": 1, "variant": "primary", "arm": "same_side_left_I1", "success": True, "first_choice": "side"},
            {"case_id": 1, "variant": "primary", "arm": "same_side_left_I0", "success": False, "first_choice": "side"},
            {"case_id": 1, "variant": "primary", "arm": "same_fork_minus_I1", "success": True, "first_choice": "fork"},
            {"case_id": 1, "variant": "primary", "arm": "same_fork_minus_I0", "success": False, "first_choice": "fork"},
        ]
        payload = {"fixture": {"name": "left", "layout": "left"}, "native_execution_count": 4, "cache_hits": 2, "records": records}
        (root / "RESULT.json").write_text(json.dumps(payload), encoding="utf-8")
        result = summarize(root, samples=200)
        contrast = result["primary_contrasts"]["B_minus_I1F0"]
        assert contrast["estimate"] == 1.0 and contrast["wins"] == 1 and contrast["losses"] == 0
        assert result["event_audit"]["raw_event_counts"]["events"] == 2
        assert result["execution_audit"]["totals"]["native_executions"] == 4.0
        assert "side_left" in result["same_first_contrasts"]["primary"]
        assert "fork_minus" in result["same_first_contrasts"]["primary"]
        assert result["examples_from_input"]["available"]
    print("summary self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="run root containing RESULT.json files")
    parser.add_argument("--output", type=Path, default=None, help="summary path, default ROOT/SUMMARY.json")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    root = args.root.resolve()
    output = (args.output or (root / "SUMMARY.json")).resolve()
    result = summarize(root, seed=args.seed, samples=args.bootstrap_samples)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "raw_records": result["input"]["raw_records"], "case_units": result["input"]["case_units"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

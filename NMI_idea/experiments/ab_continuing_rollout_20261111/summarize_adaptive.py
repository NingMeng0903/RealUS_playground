#!/usr/bin/env python3
"""Aggregate adaptive-stopping runner output without mirror pseudoreplication.

The runner writes one result for each ``(case_id, fixture_index/mode, arm)``.
This reader is intentionally tolerant of the two formats used during the
implementation: the current ``RESULT.json`` files written by
``run_adaptive.py`` and a future JSONL stream with one record per line.  It
never reruns a planner, estimates a new model, or fills a missing result.

The primary unit is an object/subject cluster.  The two mirrored runs are
kept as separate execution records and averaged *within* an object before a
paired subject bootstrap.  Thus two mirrors contribute information, but the
bootstrap has at most one draw unit per object.

Examples
--------
``python summarize_adaptive.py RUN_ROOT``

Writes ``RUN_ROOT/summary/SUMMARY.json`` and, when enough data are present,
``RUN_ROOT/summary/adaptive_outcomes.png``.  Use ``--plot`` to request the
plot explicitly; the default also attempts it when matplotlib is available.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import re
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


BOOTSTRAP_SEED = 721505
BOOTSTRAP_SAMPLES = 10_000
CI_LEVEL = 0.975
CI_ALPHA = (1.0 - CI_LEVEL) / 2.0
DEFAULT_EXPECTED_MIRRORS = 2
DEFAULT_RISK_GATE = 0.90
DEFAULT_TEST_SUBJECTS = 128
REQUIRED_CAPS = (2, 4, 6, 8)
REQUIRED_GUARDS = (8, 12)

_COMMON_NESTED_KEYS = (
    "result", "outcome", "metrics", "summary", "run", "record", "payload",
    "terminal", "execution", "statistics", "adaptive", "planner_result",
    "sourceprimaryconfig", "source_primary_config", "primary_config",
    "frozen_config", "protocol", "config",
)
_ROW_LIST_KEYS = (
    "records", "results", "runs", "rows", "subjects", "outcomes", "entries",
)
_ID_KEYS = ("object_id", "subject_id", "case_id", "case", "object", "subject", "id")
_ARM_KEYS = ("arm", "method", "policy", "strategy", "program")
_MIRROR_KEYS = (
    "mirror", "mirror_id", "fixture", "fixture_index", "layout", "layout_id",
    "layout_name", "view", "orientation", "mode",
)
_STAGE_KEYS = ("stage", "split", "dataset")


def _finite(value: Any) -> float | None:
    """Return a finite float, treating JSON null/non-numbers as missing."""
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bool(value: Any) -> bool | None:
    """Parse booleans conservatively; do not treat ``"false"`` as truthy."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = _finite(value)
        return None if number is None else bool(number)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "pass", "passed", "correct", "success", "accepted"}:
        return True
    if text in {"false", "no", "n", "0", "fail", "failed", "incorrect", "wrong", "rejected"}:
        return False
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normal_key(value: Any) -> str:
    """Stable printable key for IDs, including integer IDs from the runner."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _maps(value: Mapping[str, Any], *, depth: int = 3) -> Iterator[Mapping[str, Any]]:
    """Yield a record and its small result containers in precedence order."""
    yield value
    file_metadata = value.get("_file_metadata")
    if isinstance(file_metadata, Mapping):
        yield file_metadata
    if depth <= 0:
        return
    for key in _COMMON_NESTED_KEYS:
        child = value.get(key)
        if isinstance(child, Mapping):
            yield from _maps(child, depth=depth - 1)


def _lookup(value: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Look up a scalar in direct fields, then known result containers."""
    wanted = tuple(keys)
    for mapping in _maps(value):
        for key in wanted:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
    return default


def _mapping_list(value: Mapping[str, Any], keys: Sequence[str]) -> list[Mapping[str, Any]]:
    """Return the first list-of-mappings under any known event key."""
    for mapping in _maps(value, depth=2):
        for key in keys:
            candidate = mapping.get(key)
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, Mapping)]
    return []


def _first_bool(value: Mapping[str, Any], keys: Sequence[str]) -> bool | None:
    for mapping in _maps(value):
        for key in keys:
            if key in mapping:
                parsed = _bool(mapping[key])
                if parsed is not None:
                    return parsed
    return None


def _merge_metadata(base: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Merge manifest metadata while retaining all list-valued declarations."""
    for key, value in incoming.items():
        if key in _ROW_LIST_KEYS:
            continue
        if key == "metadata" and isinstance(value, Mapping):
            _merge_metadata(base, value)
            continue
        if key not in base:
            base[key] = value
            continue
        old = base[key]
        if old == value:
            continue
        # Manifest runs can contribute different case_start/index values.  Keep
        # every value so expected IDs can be reconstructed without guessing.
        if isinstance(old, list):
            values = old
        else:
            values = [old]
        additions = value if isinstance(value, list) else [value]
        for item in additions:
            if item not in values:
                values.append(item)
        base[key] = values
    return base


def _looks_like_row(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in (*_ID_KEYS, *_ARM_KEYS)) and (
        any(key in payload for key in ("correct_complete", "false_complete", "success", "reason", "events", "decisions"))
        or any(key in payload for key in _ARM_KEYS)
    )


def _payload_records(payload: Any, *, source: Path, line: int | None = None,
                     inherited: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flatten a JSON/JSONL payload into row dictionaries and metadata."""
    inherited = dict(inherited or {})
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    if isinstance(payload, list):
        for item in payload:
            child_rows, child_meta = _payload_records(item, source=source, line=line, inherited=inherited)
            rows.extend(child_rows)
            _merge_metadata(metadata, child_meta)
        return rows, metadata
    if not isinstance(payload, Mapping):
        return rows, metadata

    local_meta = dict(inherited)
    _merge_metadata(local_meta, payload)
    nested_metadata = payload.get("metadata")
    if isinstance(nested_metadata, Mapping):
        _merge_metadata(metadata, nested_metadata)

    list_found = False
    for key in _ROW_LIST_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            list_found = True
            for index, item in enumerate(candidate):
                child_rows, child_meta = _payload_records(
                    item, source=source, line=index if line is None else line,
                    inherited=local_meta,
                )
                rows.extend(child_rows)
                _merge_metadata(metadata, child_meta)
    if _looks_like_row(payload) and not list_found:
        row = dict(payload)
        row["_source"] = str(source)
        if line is not None:
            row["_line"] = line
        row["_file_metadata"] = dict(inherited)
        rows.append(row)
    _merge_metadata(metadata, payload)
    return rows, metadata


def _candidate_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Do not ingest our own generated summary or old unrelated source JSON.
        if "summary" in {part.lower() for part in path.relative_to(root).parts}:
            continue
        if path.name.upper().startswith("SUMMARY"):
            continue
        if path.suffix.lower() in {".jsonl", ".ndjson"} or path.name == "RESULT.json" or path.name.startswith("MANIFEST"):
            files.append(path)
    # A caller may pass a directory containing a single non-standard JSON file.
    if not files:
        files = [path for path in sorted(root.glob("*.json")) if path.is_file()]
    return files


def load_records(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Read runner files.

    Returns ``(normalized_source_rows, merged_metadata, diagnostics)``.  The
    rows are intentionally still raw; :func:`normalize_rows` performs the
    outcome and metric interpretation.
    """
    root_path = Path(root).resolve()
    files = _candidate_files(root_path)
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"input_root": str(root_path), "source_files": []}
    diagnostics: dict[str, Any] = {
        "files_seen": len(files), "files_parsed": 0, "parse_errors": [],
        "non_record_files": [], "raw_rows": 0,
    }
    for path in files:
        try:
            if path.suffix.lower() in {".jsonl", ".ndjson"}:
                payloads = []
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        payloads.append((line_no, json.loads(line)))
                    except Exception as exc:  # keep other records usable
                        diagnostics["parse_errors"].append({"path": str(path), "line": line_no, "error": str(exc)})
                file_rows: list[dict[str, Any]] = []
                file_meta: dict[str, Any] = {}
                for line_no, payload in payloads:
                    child_rows, child_meta = _payload_records(payload, source=path, line=line_no)
                    file_rows.extend(child_rows)
                    _merge_metadata(file_meta, child_meta)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                file_rows, file_meta = _payload_records(payload, source=path)
        except Exception as exc:
            diagnostics["parse_errors"].append({"path": str(path), "error": str(exc)})
            continue
        diagnostics["files_parsed"] += 1
        diagnostics["raw_rows"] += len(file_rows)
        if not file_rows:
            diagnostics["non_record_files"].append(str(path))
        rows.extend(file_rows)
        metadata["source_files"].append(str(path))
        _merge_metadata(metadata, file_meta)
    return rows, metadata, diagnostics


def _extract_identifier(row: Mapping[str, Any]) -> str | None:
    value = _lookup(row, _ID_KEYS)
    return None if value is None else _normal_key(value)


def _extract_mirror(row: Mapping[str, Any]) -> str | None:
    value = _lookup(row, _MIRROR_KEYS)
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _normal_key(value)


def _event_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(row, ("events", "history", "event_log", "acquisitions", "packets"))


def _decision_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(row, ("decisions", "decision_log", "planning_decisions"))


def _private_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return _mapping_list(row, ("privateeval", "private_eval", "private", "truth_events"))


def _row_reason(row: Mapping[str, Any]) -> str:
    value = _lookup(row, ("reason", "status", "stop_reason", "termination_reason", "outcome"), "")
    return str(value or "").strip().lower()


def _outcome(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]], private: Sequence[Mapping[str, Any]]) -> tuple[str, bool, bool, bool]:
    """Classify final outcome; false public stops always override qualification."""
    correct = _first_bool(row, ("correct_complete", "correct_completion", "final_correct", "private_correct"))
    false = _first_bool(row, ("false_complete", "false_completion", "final_false", "private_false", "false_stop", "wrong_complete"))
    # ``success`` is accepted only as a fallback; explicit false_complete wins.
    if correct is None:
        correct = _first_bool(row, ("success", "completed", "complete"))
    for event in reversed(list(events)):
        public = _bool(event.get("public_accepted", event.get("accepted")))
        private_correct = _bool(event.get("private_correct", event.get("correct")))
        private_false = _bool(event.get("private_false", event.get("false")))
        status = str(event.get("status", "")).lower()
        if private_false is True or (public is True and private_correct is False):
            false = True
            correct = False
            break
        if private_correct is True and ("terminal" in status or event.get("kind") == "terminal"):
            correct = True
            break
    for item in reversed(list(private)):
        if _bool(item.get("false")) is True:
            false = True
            correct = False
            break
    reason = _row_reason(row)
    watchdog = bool(_first_bool(row, ("watchdog", "watchdog_triggered", "guard_triggered")) is True or
                    any(token in reason for token in ("watchdog", "guard", "step_guard")))
    withdraw = bool(_first_bool(row, ("withdraw", "withdrawn", "abstain", "withdrew")) is True or
                    any(token in reason for token in ("withdraw", "abstain", "no_checked", "no_lawful", "exit")))
    if false is True:
        return "false_complete", False, True, False
    if correct is True:
        return "correct_complete", True, False, False
    if watchdog:
        return "watchdog", False, False, True
    if withdraw:
        return "withdraw", False, False, False
    return "no_complete", False, False, False


def _has_commit(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]], decisions: Sequence[Mapping[str, Any]]) -> bool:
    direct = _first_bool(row, ("committed", "commit", "commit_attempted", "terminal_option_started"))
    if direct is not None:
        return direct
    terminal_count = _lookup(row, ("terminal_count", "terminal_captures", "terminal_capture_count"))
    if (_finite(terminal_count) or 0.0) > 0.0:
        return True
    for decision in decisions:
        kind = str(decision.get("decision", decision.get("kind", ""))).lower()
        if kind in {"commit", "committed", "terminal", "complete_option"}:
            return True
    for event in events:
        status = str(event.get("status", "")).lower()
        kind = str(event.get("kind", "")).lower()
        action = event.get("action") if isinstance(event.get("action"), Mapping) else {}
        if "terminal" in status or kind == "terminal" or str(action.get("kind", "")).lower() == "terminal":
            return True
    return False


def _ever_qualified(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]], private: Sequence[Mapping[str, Any]], correct: bool) -> bool:
    explicit = _first_bool(row, ("everqualified", "ever_qualified", "ever_qualifying", "best_private_qualified"))
    if explicit is True:
        return True
    if correct:
        return True
    for item in (*events, *private):
        if _bool(item.get("ever_qualified", item.get("qualified", item.get("private_qualified")))) is True:
            return True
        if _bool(item.get("quality_pass")) is True and _bool(item.get("public_accepted")) is not False:
            return True
    return False


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    value = _lookup(row, keys)
    return _finite(value)


def _dwell_values(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    direct = _lookup(row, ("dwell_s", "dwells", "dwell", "dwell_histogram"))
    if isinstance(direct, Mapping):
        for key, count in direct.items():
            dwell = _finite(key)
            n = _finite(count)
            if dwell is not None and n is not None:
                values.extend([dwell] * max(0, int(round(n))))
    elif isinstance(direct, list):
        values.extend(float(v) for v in (_finite(x) for x in direct) if v is not None)
    else:
        number = _finite(direct)
        if number is not None:
            values.append(number)
    for event in events:
        action = event.get("action") if isinstance(event.get("action"), Mapping) else event
        for key in ("dwell_s", "dwell", "duration_s"):
            value = _finite(action.get(key)) if isinstance(action, Mapping) else None
            if value is not None and key != "duration_s":
                values.append(value)
                break
    return values


def _event_count(events: Sequence[Mapping[str, Any]], tokens: Sequence[str]) -> int:
    count = 0
    for event in events:
        text = " ".join(str(event.get(key, "")).lower() for key in ("status", "kind", "event", "reason", "action_id"))
        if any(token in text for token in tokens):
            count += 1
    return count


def _metric_number(row: Mapping[str, Any], keys: Sequence[str], nested: Sequence[str] = ()) -> float | None:
    value = _lookup(row, keys)
    number = _finite(value)
    if number is not None:
        return number
    for container in _maps(row, depth=2):
        for name in nested:
            child = container.get(name)
            number = _finite(child)
            if number is not None:
                return number
    return None


def _prediction_from_mapping(mapping: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    raw = _finite(mapping.get("raw_p", mapping.get("raw_complete_p", mapping.get("raw_complete_probability"))))
    cal = _finite(mapping.get("cal_p", mapping.get("calibrated_p", mapping.get("p_cal", mapping.get("cal_score")))))
    risk = _finite(mapping.get("risk_score", mapping.get("gate_score", mapping.get("common_score"))))
    if risk is None and raw is not None and cal is not None:
        risk = min(raw, cal)
    return raw, cal, risk


def _commit_prediction(row: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]], committed: bool) -> tuple[float | None, float | None, float | None, str | None]:
    """Extract only a prediction attached to a commit, never a probe score."""
    for key in ("commit_prediction", "committed_prediction", "terminal_prediction", "commit_score"):
        candidate = _lookup(row, (key,))
        if isinstance(candidate, Mapping):
            raw, cal, risk = _prediction_from_mapping(candidate)
            if any(v is not None for v in (raw, cal, risk)):
                return raw, cal, risk, key
    candidates: list[tuple[float | None, float | None, float | None, str]] = []
    for decision in decisions:
        kind = str(decision.get("decision", decision.get("kind", ""))).lower()
        if kind not in {"commit", "committed", "terminal", "complete_option"}:
            continue
        mappings: list[Mapping[str, Any]] = [decision]
        for key in ("option", "prediction", "terminal_option", "selected_option"):
            child = decision.get(key)
            if isinstance(child, Mapping):
                mappings.append(child)
        diagnostics = decision.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            mappings.append(diagnostics)
            options = diagnostics.get("terminal_options")
            if isinstance(options, list):
                mappings.extend(item for item in options if isinstance(item, Mapping))
        for mapping in mappings:
            raw, cal, risk = _prediction_from_mapping(mapping)
            if any(v is not None for v in (raw, cal, risk)):
                candidates.append((raw, cal, risk, "decision_commit"))
    # A top-level prediction is safe only when the row actually committed;
    # this rejects high-confidence exploratory probe records.
    if committed:
        for key in ("prediction", "raw_complete_p", "raw_p"):
            candidate = _lookup(row, (key,))
            if isinstance(candidate, Mapping):
                raw, cal, risk = _prediction_from_mapping(candidate)
                if any(v is not None for v in (raw, cal, risk)):
                    candidates.append((raw, cal, risk, f"{key}_committed"))
            elif key in {"raw_complete_p", "raw_p"}:
                raw = _finite(candidate)
                cal = _finite(_lookup(row, ("calibrated_p", "cal_score", "p_cal")))
                risk = _finite(_lookup(row, ("risk_score", "gate_score")))
                if risk is None and raw is not None and cal is not None:
                    risk = min(raw, cal)
                if raw is not None or cal is not None or risk is not None:
                    candidates.append((raw, cal, risk, f"{key}_committed"))
    return candidates[-1] if candidates else (None, None, None, None)


def normalize_rows(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize current runner records into the summary's stable row schema."""
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row = dict(raw)
        object_id = _extract_identifier(row)
        mirror = _extract_mirror(row)
        arm_value = _lookup(row, _ARM_KEYS, "unknown")
        arm = str(arm_value if arm_value is not None else "unknown")
        stage = _lookup(row, _STAGE_KEYS)
        stage = None if stage is None else str(stage)
        if stage is None and row.get("_source"):
            # RESULT.json from the current runner has no stage field; its path
            # is .../<stage>/fixture_.../arm/case_.../RESULT.json.
            path_parts = {part.lower() for part in Path(str(row["_source"])).parts}
            for candidate in ("test", "dev", "smoke"):
                if candidate in path_parts:
                    stage = candidate
                    break
        variant = _lookup(row, ("variant", "condition", "control", "run_variant"))
        variant = None if variant is None else str(variant)
        events = _event_rows(row)
        decisions = _decision_rows(row)
        private = _private_rows(row)
        outcome, correct, false, watchdog = _outcome(row, events, private)
        withdraw = outcome == "withdraw"
        committed = _has_commit(row, events, decisions)
        everqualified = _ever_qualified(row, events, private, correct)
        raw_p, cal_p, risk_p, prediction_source = _commit_prediction(row, decisions, committed)
        probes = _metric_number(row, ("probes", "probe_count", "diagnostic_count", "n_probes"))
        if probes is None:
            probes = float(_event_count(events, ("probe_executed", "pair_probe", "diagnostic", "probe")))
        motiontime = _metric_number(row, ("motiontime", "motion_time_s", "motion_seconds", "simtime", "native_elapsed_s", "elapsed_s"))
        if motiontime is None and events:
            cumulative = [_finite(event.get("native_elapsed_s", event.get("elapsed_s"))) for event in events]
            cumulative = [x for x in cumulative if x is not None]
            motiontime = max(cumulative) if cumulative else None
        planningwall = _metric_number(row, ("planningwall", "planning_wall_s", "wall_planning_s", "planning_seconds"))
        if planningwall is None:
            planningwall = 0.0 if "planningwall" in row else None
        loss_events = _metric_number(row, ("loss_reacquire_events", "reacquisition_events", "reacquire_events", "loss_events", "recoveryloss"))
        if loss_events is None:
            loss_events = float(_event_count(events, ("reacquir", "loss", "visibility_lost", "recovery")))
            if loss_events == 0 and _first_bool(row, ("reacquired", "reacquisition")) is True:
                loss_events = 1.0
        loss_time = _metric_number(row, ("loss_reacquire_time_s", "reacquire_time_s", "time_to_reacquire_s", "loss_to_reacquire_s", "recovery_time_s"))
        native_failures = _metric_number(row, ("native_failures", "native_failure_count", "physical_failures", "motion_failures", "failed_moves"))
        if native_failures is None:
            native_failures = float(_event_count(events, ("native_invalid", "native_fail", "motion_fail", "failed_move", "execution_failed")))
        cache_hits = _metric_number(row, ("cache_hits", "native_cache_hits", "execution_cache_hits", "cached_executions"))
        cache_misses = _metric_number(row, ("cache_misses", "native_cache_misses", "execution_cache_misses"))
        dwell = _dwell_values(row, events)
        terminal_count = _metric_number(row, ("terminal_count", "terminal_captures", "terminal_capture_count"))
        normalized.append({
            "_index": index, "_source": row.get("_source"), "_line": row.get("_line"), "raw": row,
            "object_id": object_id, "mirror": mirror, "arm": arm, "stage": stage, "variant": variant,
            "outcome": outcome, "correct_complete": correct, "false_complete": false,
            "no_complete": outcome == "no_complete", "withdraw": withdraw, "watchdog": watchdog,
            "committed": committed, "committed_failure": bool(committed and not correct),
            "everqualified": everqualified, "probes": probes, "dwell_s": dwell,
            "motiontime_s": motiontime, "planningwall_s": planningwall,
            "loss_reacquire_events": loss_events, "loss_reacquire_time_s": loss_time,
            "native_failures": native_failures, "cache_hits": cache_hits,
            "cache_misses": cache_misses, "terminal_count": terminal_count,
            "raw_p": raw_p, "calibrated_p": cal_p, "risk_score": risk_p,
            "prediction_source": prediction_source, "events": events, "decisions": decisions,
            "privateeval": private,
        })
    return normalized


def _mean(values: Iterable[float | None]) -> float | None:
    numbers = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.mean(numbers)) if numbers else None


def _quantile(values: Iterable[float | None], q: float) -> float | None:
    numbers = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return float(np.quantile(numbers, q)) if numbers else None


def _metric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [row.get(key) for row in rows]
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {"n": len(finite), "missing": len(values) - len(finite),
            "mean": float(np.mean(finite)) if finite else None,
            "median": float(np.median(finite)) if finite else None,
            "p95": float(np.quantile(finite, 0.95)) if finite else None,
            "total": float(np.sum(finite)) if finite else None}


def _object_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("object_id") is not None:
            groups[str(row["object_id"])].append(row)
    return dict(groups)


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one arm with both mirror-level counts and object rates."""
    groups = _object_groups(rows)
    object_ids = sorted(groups)
    outcomes = ("correct_complete", "false_complete", "no_complete", "withdraw", "watchdog")
    mirror_counts = Counter(str(row["outcome"]) for row in rows)
    object_counts = {
        name: float(sum(np.mean([1.0 if item.get("outcome") == name else 0.0 for item in groups[obj]]) for obj in object_ids))
        for name in outcomes
    }
    n_objects = len(object_ids)
    n_mirrors = len(rows)
    metrics = {key: _metric_summary(rows, key) for key in (
        "probes", "motiontime_s", "planningwall_s", "loss_reacquire_events",
        "loss_reacquire_time_s", "native_failures", "cache_hits", "cache_misses", "terminal_count",
    )}
    dwell_counter: Counter[str] = Counter()
    dwell_values: list[float] = []
    for row in rows:
        for dwell in row.get("dwell_s", []):
            if _finite(dwell) is None:
                continue
            dwell_values.append(float(dwell))
            dwell_counter[f"{float(dwell):g}"] += 1
    committed = sum(1 for row in rows if row.get("committed"))
    committed_fail = sum(1 for row in rows if row.get("committed_failure"))
    return {
        "mirror_records": n_mirrors, "object_clusters": n_objects,
        "object_ids": object_ids,
        "mirror_counts": {name: int(mirror_counts.get(name, 0)) for name in outcomes},
        "object_cluster_counts": object_counts,
        "all_subject_denominator": n_objects,
        "rates_all_subjects": {name: (object_counts[name] / n_objects if n_objects else None) for name in outcomes},
        "undecided_count": object_counts["no_complete"] + object_counts["withdraw"] + object_counts["watchdog"],
        "undecided_rate": ((object_counts["no_complete"] + object_counts["withdraw"] + object_counts["watchdog"]) / n_objects
                           if n_objects else None),
        "commitcoverage": committed / n_mirrors if n_mirrors else None,
        "committed_fail_rate": committed_fail / committed if committed else None,
        "commit_counts": {"committed": committed, "committed_failure": committed_fail,
                          "not_committed": n_mirrors - committed},
        "everqualified": sum(1 for row in rows if row.get("everqualified")) / n_mirrors if n_mirrors else None,
        "metrics": metrics,
        "dwell_histogram_s": dict(sorted(dwell_counter.items(), key=lambda item: float(item[0]))),
        "dwell_summary_s": _metric_summary([{"dwell": value} for value in dwell_values], "dwell"),
    }


def _expected_ids(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], stage: str | None,
                  expected_subjects: int | None) -> tuple[list[str], bool]:
    candidates: list[Any] = []
    for key in ("test_subject_ids", "subject_ids", "object_ids", "objects", "subjects", "case_ids"):
        value = metadata.get(key)
        if isinstance(value, list):
            candidates.extend(value)
        elif value is not None:
            candidates.append(value)
    # Current run_adaptive manifests declare case_start/count, possibly once
    # per fixture mirror.  Build the union rather than multiplying it.
    starts = metadata.get("case_start", 0)
    counts = metadata.get("count", metadata.get("n_subjects", metadata.get("expected_subject_count")))
    starts_list = starts if isinstance(starts, list) else [starts]
    counts_list = counts if isinstance(counts, list) else [counts]
    for start in starts_list:
        s = _finite(start)
        if s is None:
            continue
        for count in counts_list:
            n = _finite(count)
            if n is not None and n > 0:
                candidates.extend(range(int(s), int(s) + int(n)))
    if expected_subjects is not None and expected_subjects > 0:
        candidates.extend(range(expected_subjects))
    unique = sorted({_normal_key(x) for x in candidates if x is not None}, key=_natural_key)
    if unique:
        return unique, True
    observed = sorted({str(row["object_id"]) for row in rows if row.get("object_id") is not None}, key=_natural_key)
    # For a declared test stage, the frozen contract is 128 even if a runner
    # forgot to copy its manifest.  We can still identify missing integer IDs.
    if (stage or "").lower() == "test" and len(observed) == DEFAULT_TEST_SUBJECTS:
        return observed, False
    if (stage or "").lower() == "test" and len(observed) < DEFAULT_TEST_SUBJECTS:
        return [str(i) for i in range(DEFAULT_TEST_SUBJECTS)], False
    return observed, False


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(value)))


def _expected_mirrors(metadata: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], expected_mirrors: int) -> list[str]:
    for key in ("mirror_ids", "mirrors", "layouts", "layout_ids"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return sorted({_normal_key(x) for x in value}, key=_natural_key)
    observed = sorted({str(row["mirror"]) for row in rows if row.get("mirror") is not None}, key=_natural_key)
    if observed:
        return observed
    return [str(i) for i in range(expected_mirrors)]


def _resolve_arm(observed: Sequence[str], requested: str | None, aliases: Sequence[str]) -> str | None:
    if requested and requested in observed:
        return requested
    lower = {str(arm).lower(): str(arm) for arm in observed}
    for alias in ((requested,) if requested else ()):
        if alias and str(alias).lower() in lower:
            return lower[str(alias).lower()]
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]
    return None


def _declared_arm(metadata: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    """Read a frozen method name from sourceprimaryconfig/protocol metadata."""
    for mapping in _maps(metadata, depth=4):
        for key in keys:
            value = mapping.get(key)
            if isinstance(value, list):
                value = next((item for item in value if isinstance(item, str) and item.strip()), None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _completeness(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, stage: str | None,
                  expected_subjects: int | None, expected_mirrors: int, b_arm: str | None,
                  a_arm: str | None, pair_arm: str | None) -> dict[str, Any]:
    observed_arms = sorted({str(row["arm"]) for row in rows})
    required = {name: value for name, value in (("B", b_arm), ("strongA", a_arm), ("complete_pair", pair_arm)) if value}
    object_ids, metadata_ids = _expected_ids(metadata, rows, stage, expected_subjects)
    mirror_ids = _expected_mirrors(metadata, rows, expected_mirrors)
    missing: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for label, arm in required.items():
        if arm not in observed_arms:
            missing.append({"arm": arm, "reason": "primary_method_absent"})
            continue
        for object_id in object_ids:
            candidates = [row for row in rows if str(row.get("object_id")) == str(object_id) and row.get("arm") == arm]
            seen = Counter(str(row.get("mirror")) for row in candidates)
            for mirror in mirror_ids:
                if seen.get(str(mirror), 0) == 0:
                    missing.append({"arm": arm, "object_id": str(object_id), "mirror": str(mirror),
                                    "reason": "missing_primary_record"})
                elif seen.get(str(mirror), 0) > 1:
                    duplicates.append({"arm": arm, "object_id": str(object_id), "mirror": str(mirror),
                                       "records": int(seen[str(mirror)])})
    # Any row without an object or mirror cannot be paired, even if the main
    # outcomes look plausible.  Keep it in diagnostics rather than dropping it.
    missing_identity = [
        {"source": row.get("_source"), "index": row.get("_index"), "reason": "missing_object_or_mirror"}
        for row in rows if row.get("object_id") is None or row.get("mirror") is None
    ]
    reasons: list[str] = []
    if not metadata_ids:
        reasons.append("expected_subject_ids_inferred_from_observed_rows")
    if missing:
        reasons.append("missing_primary_method_object_or_layout_records")
    if duplicates:
        reasons.append("duplicate_primary_method_object_layout_records")
    if missing_identity:
        reasons.append("rows_missing_object_or_mirror_identity")
    complete = not missing and not duplicates and not missing_identity and bool(required)
    return {
        "complete": complete, "reasons": reasons,
        "required_arms": required, "observed_arms": observed_arms,
        "expected_object_clusters": len(object_ids), "expected_object_ids": object_ids,
        "expected_object_ids_from_metadata": metadata_ids,
        "expected_mirror_count": len(mirror_ids), "expected_mirror_ids": mirror_ids,
        "expected_subject_contract": {"subjects": expected_subjects or (DEFAULT_TEST_SUBJECTS if (stage or "").lower() == "test" else None),
                                      "mirrors_per_subject": expected_mirrors},
        "missing_records": missing, "duplicate_records": duplicates,
        "rows_missing_identity": missing_identity,
    }


def _cluster_values(rows: Sequence[Mapping[str, Any]], arm: str, field: str = "correct_complete") -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("arm") == arm and row.get("object_id") is not None:
            grouped[str(row["object_id"])].append(row)
    return {obj: float(np.mean([1.0 if row.get(field) else 0.0 for row in members]))
            for obj, members in grouped.items()}


def paired_subject_bootstrap(rows: Sequence[Mapping[str, Any]], b_arm: str, comparator_arm: str,
                             *, seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES,
                             field: str = "correct_complete") -> dict[str, Any]:
    """Bootstrap B-minus-comparator over object clusters, preserving mirrors."""
    b = _cluster_values(rows, b_arm, field)
    comparator = _cluster_values(rows, comparator_arm, field)
    object_ids = sorted(set(b) & set(comparator), key=_natural_key)
    result: dict[str, Any] = {
        "b_arm": b_arm, "comparator_arm": comparator_arm, "field": field,
        "bootstrap_seed": int(seed), "bootstrap_samples": int(samples),
        "cluster_unit": "object_id/subject_id; all mirrors retained within each cluster",
        "mirror_units_not_bootstrap_units": True, "object_ids": object_ids,
        "n_clusters": len(object_ids), "n_mirror_records_B": int(sum(1 for row in rows if row.get("arm") == b_arm)),
        "n_mirror_records_comparator": int(sum(1 for row in rows if row.get("arm") == comparator_arm)),
    }
    if not object_ids:
        result.update({"estimate": None, "estimate_pp": None, "ci_97_5_two_sided": None,
                       "ci_lower": None, "ci_upper": None, "reason": "no_paired_object_clusters"})
        return result
    differences = np.asarray([b[obj] - comparator[obj] for obj in object_ids], dtype=float)
    estimate = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(int(samples), len(differences)))
    boot = differences[draws].mean(axis=1)
    lower, upper = (float(x) for x in np.quantile(boot, [CI_ALPHA, 1.0 - CI_ALPHA]))
    result.update({
        "estimate": estimate, "estimate_pp": 100.0 * estimate,
        "ci_97_5_two_sided": [lower, upper], "ci_lower": lower, "ci_upper": upper,
        "ci_lower_pp": 100.0 * lower, "ci_upper_pp": 100.0 * upper,
        "wins": int(np.sum(differences > 0)), "losses": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)), "cluster_differences": differences.tolist(),
    })
    return result


def _calibration_summary(rows: Sequence[Mapping[str, Any]], bins: int = 10) -> dict[str, Any]:
    aligned = [row for row in rows if row.get("committed") and
               (row.get("raw_p") is not None or row.get("calibrated_p") is not None or row.get("risk_score") is not None)]
    missing = sum(1 for row in rows if row.get("committed") and row.get("raw_p") is None and row.get("calibrated_p") is None and row.get("risk_score") is None)
    output: dict[str, Any] = {"aligned_commit_predictions": len(aligned), "committed_without_prediction": missing,
                              "excluded_noncommit_predictions": sum(1 for row in rows if not row.get("committed")),
                              "prediction_alignment": "commit prediction paired with final correct_complete; probe scores excluded"}
    for name in ("raw_p", "calibrated_p", "risk_score"):
        pairs = [(float(row[name]), 1.0 if row.get("correct_complete") else 0.0)
                 for row in aligned if row.get(name) is not None and math.isfinite(float(row[name]))]
        if not pairs:
            output[name] = {"n": 0, "brier": None, "reliability": []}
            continue
        prediction = np.asarray([p for p, _ in pairs], dtype=float)
        observed = np.asarray([y for _, y in pairs], dtype=float)
        pred = np.clip(prediction, 0.0, 1.0)
        reliability = []
        for index in range(bins):
            lo = index / bins
            hi = (index + 1) / bins
            mask = (pred >= lo) & (pred <= hi if index == bins - 1 else pred < hi)
            reliability.append({"bin": index, "lower": lo, "upper": hi, "n": int(mask.sum()),
                                "mean_predicted": float(np.mean(pred[mask])) if mask.any() else None,
                                "observed_correct": float(np.mean(observed[mask])) if mask.any() else None,
                                "brier": float(np.mean((pred[mask] - observed[mask]) ** 2)) if mask.any() else None})
        output[name] = {"n": len(pairs), "brier": float(np.mean((pred - observed) ** 2)),
                        "reliability": reliability}
    return output


def _condition_label(row: Mapping[str, Any]) -> str:
    value = row.get("variant")
    if value:
        return str(value)
    raw = row.get("raw", {})
    value = _lookup(raw, ("condition", "variant", "control", "run_variant"))
    return str(value) if value is not None else "primary"


def _cap_value(row: Mapping[str, Any]) -> int | None:
    value = _lookup(row.get("raw", {}), ("observation_cap", "matched_count", "max_observations", "prefix_length", "diagnostic_cap"))
    if value is None:
        value = row.get("raw", {}).get("shadow_count") if isinstance(row.get("raw"), Mapping) else None
    number = _finite(value)
    return int(number) if number is not None and number in REQUIRED_CAPS else None


def _guard_value(row: Mapping[str, Any]) -> int | None:
    value = _lookup(row.get("raw", {}), ("guard", "watchdog_guard", "max_probe_frames", "watchdog_max_probe_frames"))
    number = _finite(value)
    if number is not None and int(number) in REQUIRED_GUARDS:
        return int(number)
    text = " ".join(str(row.get(key, "")) for key in ("variant", "stage"))
    match = re.search(r"(?:guard|cap)[_-]?(8|12)", text, flags=re.I)
    return int(match.group(1)) if match else None


def _is_shadow(row: Mapping[str, Any]) -> bool:
    direct = _first_bool(row.get("raw", {}), ("shadow", "is_shadow", "shadow_run", "matched_shadow"))
    if direct is not None:
        return direct
    text = " ".join(str(row.get(key, "")) for key in ("variant", "stage"))
    return "shadow" in text.lower()


def _controls_summary(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, required_arms: Sequence[str]) -> dict[str, Any]:
    guards: dict[str, Any] = {}
    for guard in REQUIRED_GUARDS:
        subset = [row for row in rows if _guard_value(row) == guard]
        guards[f"guard{guard}"] = {"guard": guard, "rows": len(subset),
                                    "arms": sorted({str(row["arm"]) for row in subset}),
                                    "status": "complete" if subset else "incomplete",
                                    "reason": None if subset else "guard_variant_not_run"}
    caps: dict[str, Any] = {}
    for cap in REQUIRED_CAPS:
        subset = [row for row in rows if _cap_value(row) == cap and not _is_shadow(row)]
        caps[str(cap)] = {"observation_cap": cap, "rows": len(subset),
                          "arms": sorted({str(row["arm"]) for row in subset}),
                          "status": "complete" if subset else "incomplete",
                          "reason": None if subset else "common_observation_cap_not_run"}
    declared_first32: list[str] = []
    for key in ("first32_test_subject_ids", "test_subject_ids", "subject_ids"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            declared_first32 = [_normal_key(x) for x in value[:32]]
            break
    if not declared_first32:
        declared_first32 = sorted({str(row["object_id"]) for row in rows if row.get("object_id") is not None}, key=_natural_key)[:32]
    shadows: dict[str, Any] = {}
    for cap in (2, 4):
        subset = [row for row in rows if _cap_value(row) == cap and _is_shadow(row) and str(row.get("object_id")) in declared_first32]
        observed_ids = sorted({str(row["object_id"]) for row in subset}, key=_natural_key)
        missing_ids = [obj for obj in declared_first32 if obj not in set(observed_ids)]
        missing_arms = [arm for arm in required_arms if not any(row.get("arm") == arm for row in subset)]
        shadows[str(cap)] = {"observation_cap": cap, "expected_subjects": len(declared_first32),
                             "rows": len(subset), "observed_subject_ids": observed_ids,
                             "missing_subject_ids": missing_ids, "missing_required_arms": missing_arms,
                             "status": "complete" if not missing_ids and not missing_arms and declared_first32 else "incomplete",
                             "reason": None if not missing_ids and not missing_arms and declared_first32 else "first32_shadow_not_run_or_incomplete"}
    return {"guard8_vs_guard12": guards, "common_observation_caps_2_4_6_8": caps,
            "first32_shadow_2_4": shadows,
            "attribution_complete": all(item["status"] == "complete" for item in shadows.values()),
            "shadow_subject_ids": declared_first32,
            "note": "Missing guard/cap/shadow runs remain explicitly incomplete; no rows are imputed."}


def _risk_gate_audit(rows: Sequence[Mapping[str, Any]], risk_gate: float) -> dict[str, Any]:
    committed = [row for row in rows if row.get("committed")]
    missing = [row for row in committed if row.get("risk_score") is None]
    violations = [row for row in committed if row.get("risk_score") is not None and float(row["risk_score"]) < risk_gate]
    return {"threshold": risk_gate, "committed_records": len(committed),
            "missing_commit_prediction": len(missing), "gate_violations": len(violations),
            "status": "complete" if committed and not missing and not violations else "incomplete",
            "reason": None if committed and not missing and not violations else "risk_gate_evidence_missing_or_violated"}


def _plot(summary: Mapping[str, Any], path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary.setdefault("plot", {})["error"] = f"matplotlib unavailable: {exc}"
        return None
    arms = list(summary.get("arms", []))
    if not arms:
        return None
    categories = ("correct_complete", "false_complete", "undecided")
    colors = ("#26734d", "#c53d3d", "#777777")
    labels = {"correct_complete": "correct", "false_complete": "false", "undecided": "undecided"}
    counts = {category: [] for category in categories}
    motion = []
    planning = []
    for arm in arms:
        item = summary["arms_summary"].get(arm, {})
        mirror_counts = item.get("mirror_counts", {})
        counts["correct_complete"].append(mirror_counts.get("correct_complete", 0))
        counts["false_complete"].append(mirror_counts.get("false_complete", 0))
        counts["undecided"].append(sum(mirror_counts.get(x, 0) for x in ("no_complete", "withdraw", "watchdog")))
        metrics = item.get("metrics", {})
        motion.append(metrics.get("motiontime_s", {}).get("mean"))
        planning.append(metrics.get("planningwall_s", {}).get("mean"))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(len(arms))
    bottom = np.zeros(len(arms), dtype=float)
    for category, color in zip(categories, colors):
        values = np.asarray(counts[category], dtype=float)
        axes[0].bar(x, values, bottom=bottom, color=color, label=labels[category])
        bottom += values
    axes[0].set_xticks(x, arms, rotation=30, ha="right")
    axes[0].set_ylabel("mirror execution count (clusters are paired)")
    axes[0].set_title("Outcome counts")
    axes[0].legend(frameon=False)
    width = 0.36
    axes[1].bar(x - width / 2, [v if v is not None else 0 for v in motion], width, label="motion/sim time (s)", color="#356ca8")
    axes[1].bar(x + width / 2, [v if v is not None else 0 for v in planning], width, label="planning wall (s)", color="#d38b28")
    axes[1].set_xticks(x, arms, rotation=30, ha="right")
    axes[1].set_ylabel("mean seconds")
    axes[1].set_title("Execution and planning time")
    axes[1].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return str(path)


def summarize(root: str | Path, *, stage: str | None = None, b_arm: str | None = None,
              a_arm: str | None = None, pair_arm: str | None = None,
              expected_subjects: int | None = None, expected_mirrors: int = DEFAULT_EXPECTED_MIRRORS,
              seed: int = BOOTSTRAP_SEED, samples: int = BOOTSTRAP_SAMPLES,
              make_plot: bool = True, output_dir: str | Path | None = None) -> dict[str, Any]:
    raw_rows, metadata, diagnostics = load_records(root)
    normalized = normalize_rows(raw_rows)
    if stage is None:
        declared_stage = metadata.get("stage")
        if isinstance(declared_stage, list):
            stage = "test" if "test" in {str(x).lower() for x in declared_stage} else None
        elif declared_stage is not None:
            stage = str(declared_stage)
        if stage is None:
            root_text = str(root).lower()
            stage = "test" if "test" in root_text else None
    if stage:
        selected = [row for row in normalized if row.get("stage") in (None, stage)]
    else:
        selected = normalized
    observed_arms = sorted({str(row["arm"]) for row in selected})
    declared_b = _declared_arm(metadata, (
        "selected_b_arm", "selected_b", "frozen_b_arm", "frozen_B", "frozen_b",
        "primary_b_arm", "b_arm", "primary_method",
    ))
    declared_a = _declared_arm(metadata, (
        "strong_a_arm", "strong_a", "frozen_strong_a_arm", "frozen_strongA",
        "frozen_a", "strongA_arm", "strongA", "selected_a_arm", "selected_a", "a_arm",
    ))
    declared_pair = _declared_arm(metadata, (
        "complete_pair_arm", "selected_pair", "frozen_pair_arm", "frozen_pair",
        "pair_arm", "complete_pair",
    ))
    # The frozen config is authoritative.  In its absence B has a harmless
    # unique alias fallback, while strong-A intentionally does not default to
    # the old ``A``: A_task/A_info must be selected or named explicitly.
    resolved_b = _resolve_arm(observed_arms, b_arm or declared_b, ("B_H1", "B_H2", "B", "adaptive_B"))
    resolved_a = _resolve_arm(observed_arms, a_arm or declared_a,
                              ("A_full", "A_task", "A_info", "strongA"))
    if not (a_arm or declared_a) and resolved_a is None:
        candidates_a = [arm for arm in observed_arms if arm in {"A_full", "A_task", "A_info", "strongA"}]
        if len(candidates_a) == 1:
            resolved_a = candidates_a[0]
    resolved_pair = _resolve_arm(observed_arms, pair_arm or declared_pair,
                                 ("complete_pair", "pair_fixed", "Complete_pair", "pair"))
    required_arms = [arm for arm in (resolved_b, resolved_a, resolved_pair) if arm]
    arms_summary = {arm: _arm_summary([row for row in selected if row.get("arm") == arm]) for arm in observed_arms}
    completeness = _completeness(selected, metadata, stage=stage, expected_subjects=expected_subjects,
                                  expected_mirrors=expected_mirrors, b_arm=resolved_b,
                                  a_arm=resolved_a, pair_arm=resolved_pair)
    comparisons: dict[str, Any] = {}
    if resolved_b and resolved_a:
        comparisons["B_minus_strongA"] = paired_subject_bootstrap(selected, resolved_b, resolved_a, seed=seed, samples=samples)
    else:
        comparisons["B_minus_strongA"] = {"reason": "B_or_strongA_arm_missing", "estimate": None}
    if resolved_b and resolved_pair:
        comparisons["B_minus_complete_pair"] = paired_subject_bootstrap(selected, resolved_b, resolved_pair, seed=seed, samples=samples)
    else:
        comparisons["B_minus_complete_pair"] = {"reason": "B_or_complete_pair_arm_missing", "estimate": None}
    risk_gate = _finite(_lookup(metadata, ("risk_gate", "common_risk_gate"))) or DEFAULT_RISK_GATE
    risk_audit = _risk_gate_audit([row for row in selected if row.get("arm") == resolved_b], risk_gate) if resolved_b else {
        "status": "incomplete", "reason": "B_arm_missing", "threshold": risk_gate}
    controls = _controls_summary(selected, metadata, required_arms=required_arms)
    pass_effect = all(
        comparison.get("estimate") is not None and float(comparison.get("estimate")) >= 0.10 and
        comparison.get("ci_lower") is not None and float(comparison.get("ci_lower")) > 0.0
        for comparison in comparisons.values()
    ) and len(comparisons) == 2
    researchgate = bool(completeness.get("complete") and pass_effect and
                         risk_audit.get("status") == "complete" and controls.get("attribution_complete"))
    if not completeness.get("complete") or not controls.get("attribution_complete") or risk_audit.get("status") != "complete":
        gate_status = "incomplete"
    elif researchgate:
        gate_status = "pass"
    else:
        gate_status = "negative"
    result: dict[str, Any] = {
        "schema": "ab_adaptive_stopping_summary_v1",
        "protocol": {
            "bootstrap_seed": seed, "bootstrap_samples": samples,
            "confidence_level": CI_LEVEL, "two_sided_alpha": 2 * CI_ALPHA,
            "bootstrap_unit": "object/subject cluster; 2 mirrors retained within cluster",
            "primary_effect_requirement": ">= +10 percentage points and 97.5% two-sided CI lower bound > 0 for both comparisons",
            "all_subject_denominator": True, "false_stop_is_not_success": True,
        },
        "input": {"root": str(Path(root).resolve()), "stage": stage, "raw_rows": len(raw_rows),
                   "selected_rows": len(selected), "metadata": metadata, "diagnostics": diagnostics},
        "primary_methods": {"B": resolved_b, "strongA": resolved_a, "complete_pair": resolved_pair},
        "arms": observed_arms, "arms_summary": arms_summary,
        "completeness": completeness,
        "primary_comparisons": comparisons,
        "risk_gate_audit": risk_audit,
        "calibration": {arm: _calibration_summary([row for row in selected if row.get("arm") == arm]) for arm in observed_arms},
        "controls": controls,
        "research_gate": {"status": gate_status, "researchgate_pass": researchgate,
                           "effect_requirement_met": pass_effect,
                           "requires_complete_primary_records": True,
                           "requires_complete_guard_cap_shadow_controls": True},
    }
    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir)
    else:
        root_path = Path(root)
        output_path = root_path.parent / "summary" if root_path.is_file() else root_path / "summary"
    if make_plot:
        plot_path = _plot(result, output_path / "adaptive_outcomes.png")
        result["plot"] = {"path": plot_path, "status": "written" if plot_path else "not_written"}
    return result


def _self_test() -> None:
    """Small paired fixture tests for cluster accounting and gate direction."""
    rows: list[dict[str, Any]] = []
    for object_id in ("0", "1"):
        for mirror in ("constrained", "open"):
            # B succeeds on one object, strong A succeeds on the other; each
            # object's two mirrors remain separate rows.
            b_correct = object_id == "0"
            a_correct = object_id == "1"
            rows.append({"object_id": object_id, "mirror": mirror, "arm": "B_H1",
                         "correct_complete": b_correct, "false_complete": False,
                         "committed": True, "raw_p": .95, "calibrated_p": .95})
            rows.append({"object_id": object_id, "mirror": mirror, "arm": "A",
                         "correct_complete": a_correct, "false_complete": False,
                         "committed": True, "raw_p": .95, "calibrated_p": .95})
            rows.append({"object_id": object_id, "mirror": mirror, "arm": "pair_fixed",
                         "correct_complete": False, "false_complete": False,
                         "committed": False})
    normalized = normalize_rows(rows)
    # Two mirrors are two execution units but only two object bootstrap units.
    contrast = paired_subject_bootstrap(normalized, "B_H1", "A", seed=721505, samples=2000)
    assert contrast["n_clusters"] == 2
    assert contrast["n_mirror_records_B"] == 4
    assert contrast["estimate"] == 0.0
    summary = _arm_summary([r for r in normalized if r["arm"] == "B_H1"])
    assert summary["mirror_records"] == 4 and summary["object_clusters"] == 2

    # An intermediate qualification followed by a false final stop is false,
    # never a correct completion.
    false_row = normalize_rows([{
        "object_id": "x", "mirror": "m0", "arm": "B_H1", "correct_complete": False,
        "false_complete": True, "everqualified": True, "committed": True,
    }])[0]
    assert false_row["outcome"] == "false_complete" and not false_row["correct_complete"]

    # Missing one mirror/object is explicit incompleteness.
    missing = [row for row in normalized if not (row["object_id"] == "1" and row["arm"] == "pair_fixed" and row["mirror"] == "open")]
    completeness = _completeness(missing, {"stage": "test", "subject_ids": ["0", "1"], "mirror_ids": ["constrained", "open"]},
                                  stage="test", expected_subjects=2, expected_mirrors=2,
                                  b_arm="B_H1", a_arm="A", pair_arm="pair_fixed")
    assert not completeness["complete"]
    assert any(item.get("object_id") == "1" and item.get("arm") == "pair_fixed" for item in completeness["missing_records"])

    # CI direction is B-minus-A: all-B success must have a positive estimate;
    # reversing the labels must produce a negative estimate and lower bound.
    positive = []
    for object_id in range(4):
        positive.append({"object_id": str(object_id), "mirror": "m0", "arm": "B", "correct_complete": True})
        positive.append({"object_id": str(object_id), "mirror": "m0", "arm": "A", "correct_complete": False})
    positive_rows = normalize_rows(positive)
    effect = paired_subject_bootstrap(positive_rows, "B", "A", seed=721505, samples=2000)
    assert effect["estimate"] == 1.0 and effect["ci_lower"] <= effect["estimate"]
    reverse = paired_subject_bootstrap(positive_rows, "A", "B", seed=721505, samples=2000)
    assert reverse["estimate"] == -1.0 and reverse["ci_upper"] >= reverse["estimate"]
    assert reverse["ci_upper"] <= 0.0
    print("adaptive summary self-test passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("."), help="runner output root or JSON/JSONL file")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON path; default ROOT/summary/SUMMARY.json")
    parser.add_argument("--stage", choices=("auto", "smoke", "dev", "test"), default="auto")
    parser.add_argument("--b-arm", default=None, help="frozen B arm; default B_H1/B_H2/B alias")
    parser.add_argument("--a-arm", default=None, help="frozen strong-A arm; default A_full/A alias")
    parser.add_argument("--pair-arm", default=None, help="completed-pair arm; default complete_pair/pair_fixed")
    parser.add_argument("--expected-subjects", type=int, default=None)
    parser.add_argument("--expected-mirrors", type=int, default=DEFAULT_EXPECTED_MIRRORS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if args.bootstrap_samples <= 0 or args.expected_mirrors <= 0:
        parser.error("bootstrap-samples and expected-mirrors must be positive")
    stage = None if args.stage == "auto" else args.stage
    root = args.root.resolve()
    result = summarize(root, stage=stage, b_arm=args.b_arm, a_arm=args.a_arm, pair_arm=args.pair_arm,
                       expected_subjects=args.expected_subjects, expected_mirrors=args.expected_mirrors,
                       seed=args.seed, samples=args.bootstrap_samples, make_plot=not args.no_plot)
    if args.output is None:
        output_dir = root.parent / "summary" if root.is_file() else root / "summary"
        output = output_dir / "SUMMARY.json"
    else:
        output = args.output.resolve()
        output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["research_gate"]["status"],
                      "selected_rows": result["input"]["selected_rows"],
                      "complete": result["completeness"]["complete"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


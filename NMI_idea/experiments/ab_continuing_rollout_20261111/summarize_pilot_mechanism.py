"""Summarize completed pilot mechanisms from a PILOT_SUMMARY manifest.

The summary is descriptive only.  It reads the final per-case RESULT files,
pairs rows by ``(case_id, fixture_index)``, and never reconstructs hidden
truth or infers a causal/effect conclusion.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


_CASE_RE = re.compile(r"case_(\d+)")
_FIXTURE_RE = re.compile(r"fixture_(\d+)_")
_B_ARMS = ("B_H1", "B_H2", "B_rollout")
_A_ARMS = ("A_task", "A_info", "A_full", "A")


def _number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _bool(value):
    return bool(value)


def _resolve_result(path_value, manifest_path):
    path = Path(str(path_value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    # ``paths`` is an explicit selection made by the pilot driver.  Preserve
    # that selection even when a later continuation file is beside it.  Only
    # use a continuation/guard file as a fallback when the manifest path is
    # absent (or when a directory, rather than an exact file, was supplied).
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("RESULT_GUARD_*.json"))
        if not candidates:
            candidates = sorted(path.glob("RESULT.json"))
        if not candidates:
            candidates = sorted(path.glob("RESULT_CONTINUED.json"))
        path = candidates[-1] if candidates else path
    elif path.name == "RESULT.json":
        continued = path.with_name("RESULT_CONTINUED.json")
        if continued.is_file():
            path = continued
    return path


def _manifest_paths(row):
    """Return result path values for both the v4 and legacy row schemas."""
    values = row.get("paths")
    if values is None:
        values = row.get("result_paths", ())
    if isinstance(values, (str, bytes, Path)):
        values = [values]
    elif isinstance(values, Mapping):
        # Be tolerant of a keyed path map while preserving JSON insertion
        # order.  The normal driver schema is a list of exact file paths.
        values = list(values.values())
    if values is None:
        return []
    output = []
    for value in values:
        if isinstance(value, Mapping):
            value = (value.get("path") or value.get("result_path") or
                     value.get("result") or value.get("file"))
        if value is not None:
            output.append(value)
    return output


def _row_configuration(row):
    """Decode v4 configuration tags into the arm and optional dwell."""
    configuration = str(row.get("configuration", row.get("arm", "unknown")))
    match = re.fullmatch(r"pair_fixed_dwell_(\d+)", configuration)
    if match:
        return "pair_fixed", int(match.group(1)) / 100.0
    return configuration, None


def _case_fixture(result, path, row):
    case_id = result.get("case_id")
    fixture_index = result.get("fixture_index")
    if case_id is None:
        match = _CASE_RE.search(str(path))
        case_id = int(match.group(1)) if match else None
    if fixture_index is None:
        match = _FIXTURE_RE.search(str(path))
        fixture_index = int(match.group(1)) if match else None
    if (case_id is None or fixture_index is None) and row.get("case_keys"):
        # Paths in the manifest follow the same order as case_keys.
        keys = row["case_keys"]
        try:
            manifest_paths = _manifest_paths(row)
            index = next(index for index, raw in enumerate(manifest_paths)
                         if str(raw) == str(path) or Path(str(raw)).name == path.name)
            case_id, fixture_index = keys[index]
        except (StopIteration, IndexError, TypeError):
            pass
    return (None if case_id is None else int(case_id),
            None if fixture_index is None else int(fixture_index))


def _status(result):
    if _bool(result.get("correct_complete")):
        return "success"
    if _bool(result.get("false_complete")):
        return "false_stop"
    return "unresolved"


def _physical_seconds(result):
    value = result.get("simtime")
    if value is not None:
        return _number(value)
    return float(sum(_number(event.get("native_elapsed_s"))
                     for event in result.get("events", ())
                     if isinstance(event, Mapping)))


def _decision_index(decision, fallback):
    """Use the index written into a decision, with list position as fallback."""
    if isinstance(decision, Mapping):
        value = decision.get("step", fallback)
    else:
        value = fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _event_decision_indices(result):
    """Decision indices which have an actual collected event in the result."""
    indices = set()
    for event in result.get("events", ()):
        if not isinstance(event, Mapping) or event.get("decision_index") is None:
            continue
        try:
            indices.add(int(event["decision_index"]))
        except (TypeError, ValueError):
            continue
    return indices


def _rollout_metrics(result):
    """Count rollout decisions and diagnostic/executed switch decisions.

    A B decision is a rollout attempt only when its diagnostics actually carry
    the ``rollout`` record.  Switches are counted at the decision layer and
    separately intersected with event ``decision_index`` values, so a logged
    proposal which was never collected is not reported as executed.
    """
    rollout_indices = set()
    switch_indices = set()
    decisions = result.get("decisions", ())
    if not isinstance(decisions, list):
        decisions = ()
    for position, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            continue
        diagnostics = decision.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping) or "rollout" not in diagnostics:
            continue
        index = _decision_index(decision, position)
        rollout_indices.add(index)
        reason = str(diagnostics.get("rollout_switch_reason", ""))
        switched = _bool(diagnostics.get("rollout_switched"))
        # Keep compatibility with older result records that emitted a switch
        # reason but omitted the boolean flag.
        switched = switched or ("switch" in reason.lower())
        if switched:
            switch_indices.add(index)
    event_indices = _event_decision_indices(result)
    return {
        "attempts": len(rollout_indices),
        "switches": len(switch_indices),
        "switches_executed": len(switch_indices & event_indices),
        "event_decision_indices": event_indices,
    }


def _switch_count(result):
    """Legacy helper retained for callers outside ``_record``."""
    return _rollout_metrics(result)["switches"]


def _event_missing_count(result):
    """Recompute public missing observations from the recorded packets."""
    count = 0
    for event in result.get("events", ()):
        if not isinstance(event, Mapping):
            continue
        public = event.get("public", {})
        detected = public.get("detected") if isinstance(public, Mapping) else None
        count += int(not _bool(detected))
    return count


def _prune_count(result):
    explicit = result.get("local_option_rejections")
    blacklist = result.get("blacklist", ())
    return int(_number(explicit)) + (len(blacklist) if isinstance(blacklist, list) else 0)


def _record(result, path, manifest_row):
    case_id, fixture_index = _case_fixture(result, path, manifest_row)
    configured_arm, configured_dwell = _row_configuration(manifest_row)
    arm = str(result.get("arm", configured_arm))
    dwell = result.get("pair_dwell", manifest_row.get("pair_dwell_configuration"))
    if dwell is None:
        dwell = configured_dwell
    dwell_value = None if dwell is None else _number(dwell)
    events = result.get("events", ())
    decisions = result.get("decisions", ())
    native = result.get("native_counts", {})
    rollout_metrics = _rollout_metrics(result)
    event_indices = rollout_metrics["event_decision_indices"]
    # N4 events carry decision_index; the native counter is retained as a
    # fallback for legacy manifests whose historical events predate that key.
    event_execution_count = sum(
        1 for event in events
        if isinstance(event, Mapping) and event.get("decision_index") is not None
    ) if isinstance(events, list) else 0
    actual_execution_count = (event_execution_count if event_execution_count
                              else int(_number(native.get("execution_count"))))
    actual_verification_count = int(_number(native.get("verification_count")))
    missing_count = _event_missing_count(result)
    status = _status(result)
    return {
        "case_id": case_id,
        "fixture_index": fixture_index,
        "arm": arm,
        "pair_dwell": dwell_value,
        "status": status,
        "correct_complete": _bool(result.get("correct_complete")),
        "false_complete": _bool(result.get("false_complete")),
        "abstain": _bool(result.get("abstain")),
        "watchdog": _bool(result.get("watchdog")),
        "reason": result.get("reason"),
        "physical_time_s": _physical_seconds(result),
        "probe_count": int(_number(result.get("probe_count"))),
        "fresh_count": int(_number(result.get("fresh_count"))),
        "terminal_count": int(_number(result.get("terminal_count"))),
        "decision_count": len(decisions) if isinstance(decisions, list) else 0,
        "event_count": len(events) if isinstance(events, list) else 0,
        "actual_execution_count": actual_execution_count,
        "actual_verification_count": actual_verification_count,
        "actual_execution_decision_count": len(event_indices),
        "missing_count": missing_count,
        "public_missing_count": missing_count,
        "B_attempts": rollout_metrics["attempts"] if arm in _B_ARMS else 0,
        "B_switches": rollout_metrics["switches"] if arm in _B_ARMS else 0,
        "B_switches_executed": (rollout_metrics["switches_executed"]
                                 if arm in _B_ARMS else 0),
        "B_prunes": _prune_count(result) if arm in _B_ARMS else 0,
        "result_path": str(path),
        "_result": result,
    }


def _event_signature(event):
    if not isinstance(event, Mapping):
        return None
    public = event.get("public", {})
    if not isinstance(public, Mapping):
        public = {}
    # Action and public packet are both part of the observed shared prefix.
    return (
        event.get("action_id"),
        event.get("status"),
        public.get("window"),
        public.get("detected"),
        public.get("quality"),
        public.get("center_offset_m"),
        public.get("axis_angle_rad"),
        public.get("candidate_type"),
    )


def _first_divergence(left, right):
    left_events = left.get("_result", {}).get("events", ())
    right_events = right.get("_result", {}).get("events", ())
    length = min(len(left_events), len(right_events))
    for index in range(length):
        if _event_signature(left_events[index]) != _event_signature(right_events[index]):
            return {"shared_prefix_events": index, "first_disagreement_index": index,
                    "reason": "event_action_or_public_packet_differs"}
    if len(left_events) != len(right_events):
        return {"shared_prefix_events": length, "first_disagreement_index": length,
                "reason": "one_result_ended"}
    return {"shared_prefix_events": length, "first_disagreement_index": None,
            "reason": "identical_recorded_event_prefix"}


def _public_record(record):
    return {key: value for key, value in record.items() if key != "_result"}


def _arm_label(record):
    if record["arm"] == "pair_fixed":
        dwell = record.get("pair_dwell")
        return "pair_fixed" if dwell is None else f"pair_fixed@{dwell:g}s"
    return record["arm"]


def _aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[_arm_label(record)].append(record)
    output = {}
    for label, items in sorted(groups.items()):
        n = len(items)
        counts = {status: sum(item["status"] == status for item in items)
                  for status in ("success", "false_stop", "unresolved")}
        output[label] = {
            "n": n,
            "success_count": counts["success"],
            "false_stop_count": counts["false_stop"],
            "unresolved_count": counts["unresolved"],
            "success_rate": counts["success"] / n if n else None,
            "false_stop_rate": counts["false_stop"] / n if n else None,
            "unresolved_rate": counts["unresolved"] / n if n else None,
            "mean_physical_time_s": sum(item["physical_time_s"] for item in items) / n if n else None,
            "mean_physical_frames": sum(item["fresh_count"] for item in items) / n if n else None,
            "mean_probe_count": sum(item["probe_count"] for item in items) / n if n else None,
            "mean_fresh_count": sum(item["fresh_count"] for item in items) / n if n else None,
            "mean_actual_execution_count": sum(item["actual_execution_count"] for item in items) / n if n else None,
            "mean_actual_verification_count": sum(item["actual_verification_count"] for item in items) / n if n else None,
            "mean_actual_execution_decision_count": sum(item["actual_execution_decision_count"] for item in items) / n if n else None,
            "mean_missing_count": sum(item["missing_count"] for item in items) / n if n else None,
            "mean_B_attempts": sum(item["B_attempts"] for item in items) / n if n else None,
            "mean_B_switches": sum(item["B_switches"] for item in items) / n if n else None,
            "mean_B_switches_executed": sum(item["B_switches_executed"] for item in items) / n if n else None,
            "mean_B_prunes": sum(item["B_prunes"] for item in items) / n if n else None,
        }
    return output


def _paired_comparison(left, right):
    if left is None or right is None:
        return None
    score = lambda item: 1 if item["status"] == "success" else 0
    left_score, right_score = score(left), score(right)
    divergence = _first_divergence(left, right)
    return {
        "left_arm": _arm_label(left),
        "right_arm": _arm_label(right),
        "left_status": left["status"],
        "right_status": right["status"],
        "success_pp_left_minus_right": 100. * (left_score - right_score),
        "left_success_win": left_score > right_score,
        "right_success_win": right_score > left_score,
        "tie_on_success_indicator": left_score == right_score,
        "first_disagreement": divergence,
        # ``same_actual_prefix`` means at least one recorded public event is
        # identical before the first arm-specific divergence.  A completely
        # identical trace is reported separately to avoid conflating the two.
        "same_actual_prefix": divergence["shared_prefix_events"] > 0,
        "identical_actual_trace": divergence["first_disagreement_index"] is None,
    }


def _actual_prefix_counts(by_pair):
    """Count paired B/A_task episodes sharing an observed event prefix."""
    output = {}
    for b_arm in _B_ARMS:
        label = f"{b_arm}_vs_A_task"
        paired = 0
        same_prefix = 0
        identical_trace = 0
        lengths = defaultdict(int)
        for arms in by_pair.values():
            left, right = arms.get(b_arm), arms.get("A_task")
            if left is None or right is None:
                continue
            paired += 1
            divergence = _first_divergence(left, right)
            shared = int(divergence["shared_prefix_events"])
            lengths[str(shared)] += 1
            same_prefix += int(shared > 0)
            identical_trace += int(divergence["first_disagreement_index"] is None)
        output[label] = {
            "paired_episodes": paired,
            "same_actual_prefix_episodes": same_prefix,
            "identical_actual_trace_episodes": identical_trace,
            "shared_prefix_event_counts": dict(sorted(lengths.items(),
                                                       key=lambda item: int(item[0]))),
        }
    return output


def summarize(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    seen = set()
    missing = []
    for row in payload.get("rows", ()):
        if not isinstance(row, Mapping):
            continue
        for raw_path in _manifest_paths(row):
            path = _resolve_result(raw_path, manifest_path)
            if not path.is_file():
                missing.append(str(path))
                continue
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                missing.append(str(path))
                continue
            record = _record(result, path, row)
            key = (record["case_id"], record["fixture_index"], _arm_label(record))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    by_pair = defaultdict(dict)
    for record in records:
        by_pair[(record["case_id"], record["fixture_index"])][_arm_label(record)] = record
    pair_rows = []
    comparison_counts = defaultdict(lambda: {"comparable": 0, "left_success_wins": 0,
                                             "right_success_wins": 0, "ties": 0,
                                             "mean_success_pp_left_minus_right": 0.0})
    for key in sorted(by_pair):
        arms = by_pair[key]
        comparisons = {}
        for b_arm in _B_ARMS:
            for a_arm in ("A_task", "A_info"):
                comparison = _paired_comparison(arms.get(b_arm), arms.get(a_arm))
                if comparison is None:
                    continue
                label = f"{b_arm}_vs_{a_arm}"
                comparisons[label] = comparison
                stats = comparison_counts[label]
                stats["comparable"] += 1
                stats["left_success_wins"] += int(comparison["left_success_win"])
                stats["right_success_wins"] += int(comparison["right_success_win"])
                stats["ties"] += int(comparison["tie_on_success_indicator"])
                stats["mean_success_pp_left_minus_right"] += comparison["success_pp_left_minus_right"]
        pair_rows.append({"case_id": key[0], "fixture_index": key[1],
                          "arms": {label: _public_record(record) for label, record in sorted(arms.items())},
                          "comparisons": comparisons})
    for stats in comparison_counts.values():
        if stats["comparable"]:
            stats["mean_success_pp_left_minus_right"] /= stats["comparable"]
    aggregate = _aggregate(records)
    actual_prefix_counts = _actual_prefix_counts(by_pair)
    a_candidates = [label for label in aggregate if label in _A_ARMS]
    pair_candidates = [label for label in aggregate if label.startswith("pair_fixed")]
    best_key = lambda label: (-aggregate[label]["success_rate"],
                              aggregate[label]["unresolved_rate"],
                              aggregate[label]["mean_physical_time_s"], label)
    strong_a = min(a_candidates, key=best_key) if a_candidates else None
    best_pair = min(pair_candidates, key=best_key) if pair_candidates else None
    return {
        "schema": "pilot_mechanism_summary_v2",
        "input": str(manifest_path),
        "descriptive_only": True,
        "selection_note": "strong_A/best_pair are descriptive observed-rate labels only; all arms/configurations remain listed",
        "source_scope": {key: payload.get(key) for key in
                         ("scope", "created_utc", "target_objects", "mirrors",
                          "planned_episodes", "completed_episodes", "computed_episodes",
                          "test_used")},
        "loaded_results": len(records),
        "missing_or_invalid_paths": sorted(set(missing)),
        "strong_A": strong_a,
        "strong_A_summary": None if strong_a is None else aggregate[strong_a],
        "best_pair": best_pair,
        "best_pair_summary": None if best_pair is None else aggregate[best_pair],
        "aggregate": aggregate,
        "paired_outcomes": dict(sorted(comparison_counts.items())),
        "actual_prefix_counts": actual_prefix_counts,
        "pairs": pair_rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_summary", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = summarize(args.pilot_summary)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

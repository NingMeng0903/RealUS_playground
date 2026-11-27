#!/usr/bin/env python3
"""Cheap, read-only inventory of saved native checkpoints.

This script intentionally does *not* import the robot/controller stack, replay
NPZ traces, or run a native check.  It only parses saved JSON checkpoints,
NATIVE_WITNESS.json, and text logs.  The default roots are the two prior
studies used by INVENTORY.json; pass alternate roots for a follow-up audit.

Usage::

    python scan_inventory.py > /tmp/native-exit-inventory.json

The output is a compact summary rather than a dump of all 2k+ checkpoints.
Representative records retain their absolute checkpoint paths so a reviewer
can reproduce each classification directly from the saved artifact.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONTINUING = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_continuing_rollout_20261111"
)
DEFAULT_ADAPTIVE = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_adaptive_stopping_20261111"
)
DEFAULT_WITNESS = Path(
    "/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/"
    "ab_executable_skills_20261111/NATIVE_WITNESS.json"
)

RECONSTRUCTION_RE = re.compile(
    r"full controller-history reconstruction mismatch:\s*(\{.*\})"
)


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _fixture(path: Path, checkpoint: dict[str, Any]) -> str:
    """Use the saved rig name, with path names as a fallback."""

    fp = checkpoint.get("adapter_fingerprint") or {}
    rig = fp.get("rig") if isinstance(fp, dict) else None
    name = rig.get("name") if isinstance(rig, dict) else None
    if isinstance(name, str) and name:
        return name
    for part in reversed(path.parts):
        if part.startswith("fixture_"):
            return part
    return "fixture_unknown"


def _native_keys(state: Any) -> list[str]:
    return sorted(state) if isinstance(state, dict) else []


def _snapshot_schema(state: Any) -> str | None:
    native = state.get("native") if isinstance(state, dict) else None
    snapshot = native.get("snapshot") if isinstance(native, dict) else None
    if isinstance(snapshot, dict):
        value = snapshot.get("schema")
        if isinstance(value, str):
            return value
    return None


def _reconstruction_max(route: dict[str, Any]) -> dict[str, float | None]:
    maxima: dict[str, float | None] = {"q": None, "qdot": None, "acceleration": None}
    for step in route.get("steps", []) if isinstance(route, dict) else []:
        errors = step.get("reconstruction_error") if isinstance(step, dict) else None
        if not isinstance(errors, dict):
            continue
        for key in maxima:
            value = errors.get(key)
            if isinstance(value, (int, float)):
                maxima[key] = value if maxima[key] is None else max(maxima[key], value)
    return maxima


def _record(path: Path, study: str) -> dict[str, Any] | None:
    checkpoint = _json(path)
    if checkpoint is None:
        return None
    event = checkpoint.get("event") or {}
    result = event.get("result") or {}
    route = result.get("route") or {}
    if not isinstance(route, dict):
        route = {}
    reason = result.get("reason")
    steps = route.get("steps") if isinstance(route.get("steps"), list) else []
    tracking_pass = [bool(s.get("tracking_pass")) for s in steps if isinstance(s, dict)]
    native_gate = [bool(s.get("native_tracking_gate")) for s in steps if isinstance(s, dict)]
    valid = [bool(s.get("valid")) for s in steps if isinstance(s, dict)]
    native_failures = [
        s.get("native_failure_count", 0)
        for s in steps
        if isinstance(s, dict) and isinstance(s.get("native_failure_count", 0), (int, float))
    ]
    is_pass = (
        reason == "executed"
        and route.get("valid") is True
        and bool(steps)
        and all(tracking_pass)
        and all(native_gate)
        and all(valid)
        and all(value == 0 for value in native_failures)
    )
    is_no_witness = reason == "no_checked_route" and not route
    outcome = "pass" if is_pass else "no_witness" if is_no_witness else "other"
    input_state = checkpoint.get("input_state") or {}
    output_state = checkpoint.get("output_state") or {}
    state_key = tuple(input_state.get("key") or [])
    return {
        "path": str(path),
        "study": study,
        "fixture": _fixture(path, checkpoint),
        "mode": checkpoint.get("mode"),
        "action_id": checkpoint.get("action_id"),
        "input_fingerprint": checkpoint.get("input_fingerprint"),
        "input_key": list(state_key),
        "output_key": list(output_state.get("key") or []),
        "outcome": outcome,
        "reason": reason,
        "feasible": event.get("feasible"),
        "route_id": route.get("route"),
        "candidate": route.get("candidate"),
        "route_start": list(route.get("start") or []),
        "route_valid": route.get("valid"),
        "input_state_keys": sorted(input_state) if isinstance(input_state, dict) else [],
        "output_state_keys": sorted(output_state) if isinstance(output_state, dict) else [],
        "input_native_keys": _native_keys(input_state.get("native")),
        "output_native_keys": _native_keys(output_state.get("native")),
        "input_snapshot_schema": _snapshot_schema(input_state),
        "output_snapshot_schema": _snapshot_schema(output_state),
        "input_valid": input_state.get("valid"),
        "output_valid": output_state.get("valid"),
        "step_count": len(steps),
        "tracking_pass_count": sum(tracking_pass),
        "native_gate_pass_count": sum(native_gate),
        "valid_step_count": sum(valid),
        "native_failure_count_total": sum(native_failures) if native_failures else 0,
        "reconstruction_error_max": _reconstruction_max(route),
    }


def _checkpoint_records(roots: Iterable[tuple[str, Path]]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    unreadable = 0
    for study, root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("planning/checkpoints/*.json")):
            item = _record(path, study)
            if item is None:
                unreadable += 1
            else:
                records.append(item)
    return records, unreadable


def _group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    # input_fingerprint is the saved exact-state identity.  Keep input_key in
    # the fallback so older checkpoints without that field remain groupable.
    fingerprint = record.get("input_fingerprint")
    identity = fingerprint if fingerprint is not None else tuple(record["input_key"])
    return record["fixture"], identity, record["action_id"]


def _sample(records: list[dict[str, Any]], *, outcome: str, candidate: str | None = None) -> dict[str, Any] | None:
    candidates = [r for r in records if r["outcome"] == outcome]
    if candidate is not None:
        candidates = [r for r in candidates if r.get("candidate") == candidate]
    if not candidates:
        return None
    item = min(candidates, key=lambda r: r["path"])
    # The complete state-key/route fields are the useful part of a sample;
    # do not copy full q vectors or controller snapshots into the inventory.
    keep = (
        "path", "study", "fixture", "mode", "action_id", "input_fingerprint",
        "input_key", "output_key", "outcome", "reason", "feasible", "route_id",
        "candidate", "route_start", "route_valid", "input_state_keys",
        "output_state_keys", "input_native_keys", "output_native_keys",
        "input_snapshot_schema", "output_snapshot_schema", "input_valid",
        "output_valid", "step_count", "tracking_pass_count",
        "native_gate_pass_count", "valid_step_count", "native_failure_count_total",
        "reconstruction_error_max",
    )
    return {key: item[key] for key in keep}


def _route_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = Counter(
        r["candidate"] if r["outcome"] == "pass" else "no_candidate"
        for r in records
    )
    pair_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in records:
        if r["outcome"] == "pass":
            candidate = "direct" if r.get("candidate") == "direct" else r.get("candidate")
            pair_counts[(r["fixture"], r["action_id"])][candidate] += 1
    both: list[dict[str, Any]] = []
    direct_only = 0
    return_only = 0
    for (fixture, action_id), counts in sorted(pair_counts.items()):
        direct = counts.get("direct", 0)
        returned = sum(counts.get(name, 0) for name in ("retreat_fork", "round_tip"))
        if direct and returned:
            both.append({
                "fixture": fixture,
                "action_id": action_id,
                "direct_pass": direct,
                "retreat_fork_pass": counts.get("retreat_fork", 0),
                "round_tip_pass": counts.get("round_tip", 0),
            })
        elif direct:
            direct_only += 1
        elif returned:
            return_only += 1
    terminal_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for (fixture, action_id), counts in pair_counts.items():
        if action_id not in {"left", "right"}:
            continue
        terminal_counts[(fixture, action_id)] = counts
    terminal_both = terminal_direct_only = terminal_return_only = 0
    terminal_direct_pass = terminal_returned_pass = 0
    for counts in terminal_counts.values():
        direct = counts.get("direct", 0)
        returned = counts.get("retreat_fork", 0) + counts.get("round_tip", 0)
        terminal_direct_pass += direct
        terminal_returned_pass += returned
        if direct and returned:
            terminal_both += 1
        elif direct:
            terminal_direct_only += 1
        elif returned:
            terminal_return_only += 1
    return {
        "pass_candidate_counts": dict(sorted(candidate_counts.items())),
        "direct_pass": candidate_counts.get("direct", 0),
        "returned_pass": candidate_counts.get("retreat_fork", 0) + candidate_counts.get("round_tip", 0),
        "return_by_candidate": {
            "retreat_fork": candidate_counts.get("retreat_fork", 0),
            "round_tip": candidate_counts.get("round_tip", 0),
        },
        "no_witness_no_candidate": candidate_counts.get("no_candidate", 0),
        "fixture_action_pairs_with_both_direct_and_return": both,
        "fixture_action_pair_counts": {
            "both_direct_and_return": len(both),
            "direct_only": direct_only,
            "return_only": return_only,
        },
        "prescribed_terminal_scan_left_right": {
            "both_direct_and_return": terminal_both,
            "direct_only": terminal_direct_only,
            "return_only": terminal_return_only,
            "direct_pass": terminal_direct_pass,
            "returned_pass": terminal_returned_pass,
        },
    }


def _native_witness(path: Path) -> dict[str, Any]:
    value = _json(path) if path.exists() else None
    if value is None:
        return {"path": str(path), "exists": False}
    transitions = []
    for transition in value.get("transitions", []):
        if not isinstance(transition, dict):
            continue
        input_state = transition.get("input_state") or {}
        exit_state = transition.get("exit_state") or {}
        transitions.append({
            "index": transition.get("index"),
            "action_id": transition.get("action_id"),
            "route_id": transition.get("route_id"),
            "source_checkpoint": transition.get("source_checkpoint"),
            "source_trace": transition.get("source_trace"),
            "trace_samples": transition.get("trace_samples"),
            "elapsed_s": transition.get("elapsed_s"),
            "input_state_keys": sorted(input_state),
            "exit_state_keys": sorted(exit_state),
            "input_snapshot_schema": input_state.get("snapshot_schema"),
            "exit_snapshot_schema": exit_state.get("snapshot_schema"),
            "linkage": transition.get("linkage"),
        })
    downstream = []
    for item in value.get("downstream_route_feasible_sets", []):
        if not isinstance(item, dict):
            continue
        actions = item.get("actions") or {}
        statuses = Counter(
            action.get("status")
            for action in actions.values()
            if isinstance(action, dict)
        )
        downstream.append({
            "exit_index": item.get("exit_index"),
            "route_set_kind": item.get("route_set_kind"),
            "native_qp_checked": item.get("native_qp_checked"),
            "prediction_status_contract": item.get("prediction_status_contract"),
            "action_count": item.get("action_count"),
            "available_action_count": item.get("available_action_count"),
            "status_counts": dict(statuses),
            "saved_proposal_match_count": item.get("saved_proposal_match_count"),
        })
    replay = value.get("fresh_native_replay") or {}
    segments = replay.get("segments") if isinstance(replay, dict) else []
    replay_summary = {
        "requested": replay.get("requested") if isinstance(replay, dict) else None,
        "performed": replay.get("performed") if isinstance(replay, dict) else None,
        "backend": replay.get("backend") if isinstance(replay, dict) else None,
        "segment_count": len(segments) if isinstance(segments, list) else 0,
        "whole_segment_pass_count": sum(
            bool(segment.get("whole_segment_pass"))
            for segment in segments
            if isinstance(segment, dict)
        ),
        "exit_state_valid_count": sum(
            bool(segment.get("exit_state_valid"))
            for segment in segments
            if isinstance(segment, dict)
        ),
        "total_wall_seconds": replay.get("total_wall_seconds") if isinstance(replay, dict) else None,
    }
    checks = value.get("checks") or {}
    provenance = value.get("provenance") or {}
    return {
        "path": str(path),
        "exists": True,
        "schema": value.get("schema"),
        "status": value.get("status"),
        "scope": value.get("scope"),
        "source_result": value.get("source_result"),
        "transitions": transitions,
        "continuity": value.get("continuity"),
        "fresh_native_replay": replay_summary,
        "downstream_route_feasible_sets": downstream,
        "checks": {
            "count": checks.get("count"),
            "passed": checks.get("passed"),
            "failed": checks.get("failed"),
        },
        "provenance": {
            "complete_exit_contract": provenance.get("complete_exit_contract"),
            "route_set_contract": provenance.get("route_set_contract"),
            "live_robot_connection": provenance.get("live_robot_connection"),
            "external_network": provenance.get("external_network"),
        },
    }


def _log_exceptions(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.log")):
            if path in seen:
                continue
            seen.add(path)
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if "full controller-history reconstruction mismatch" in text:
                match = RECONSTRUCTION_RE.search(text)
                rows.append({
                    "path": str(path),
                    "kind": "replay_exception",
                    "exception": "full controller-history reconstruction mismatch",
                    "details": match.group(1) if match else None,
                })
            elif "KeyboardInterrupt" in text:
                rows.append({
                    "path": str(path),
                    "kind": "computation_interrupt",
                    "exception": "KeyboardInterrupt",
                    "details": "log interruption; not a checkpoint pass/no-witness outcome",
                })
    return rows


def build_inventory(
    continuing: Path = DEFAULT_CONTINUING,
    adaptive: Path = DEFAULT_ADAPTIVE,
    witness: Path = DEFAULT_WITNESS,
) -> dict[str, Any]:
    roots = [("ab_continuing_rollout_20261111", continuing), ("ab_adaptive_stopping_20261111", adaptive)]
    records, unreadable = _checkpoint_records(roots)
    outcomes = Counter(r["outcome"] for r in records)
    by_study: dict[str, Counter[str]] = defaultdict(Counter)
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_fixture_action: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_study[record["study"]][record["outcome"]] += 1
        by_action[record["action_id"]][record["outcome"]] += 1
        by_fixture_action[f"{record['fixture']}::{record['action_id']}"][record["outcome"]] += 1

    groups: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        groups[_group_key(record)].add(record["outcome"])
    group_outcomes = Counter(
        "mixed" if len(values) > 1 else next(iter(values))
        for values in groups.values()
    )

    route = _route_summary(records)
    exceptions = _log_exceptions([continuing, adaptive])
    return {
        "schema": "native_exit_inventory_v1",
        "scope": {
            "description": "Saved native checkpoint/result inventory only; no replay or execution performed.",
            "roots": {"continuing": str(continuing), "adaptive": str(adaptive)},
            "witness": str(witness),
        },
        "checkpoint_counts": {
            "total_read": len(records),
            "unreadable_json": unreadable,
            "outcomes": dict(sorted(outcomes.items())),
            "by_study": {
                study: dict(sorted(counts.items()))
                for study, counts in sorted(by_study.items())
            },
            "by_action": {
                action: dict(sorted(counts.items()))
                for action, counts in sorted(by_action.items())
            },
            "by_fixture_action": {
                key: dict(sorted(counts.items()))
                for key, counts in sorted(by_fixture_action.items())
            },
        },
        "exact_state_action_groups": {
            "group_key": "fixture + input_fingerprint (exact complete state identity) + action_id",
            "total_groups": len(groups),
            "outcome_groups": dict(sorted(group_outcomes.items())),
            "mixed_group_count": group_outcomes.get("mixed", 0),
        },
        "routes": route,
        "representative_checkpoints": {
            "direct_pass": _sample(records, outcome="pass", candidate="direct"),
            "returned_retreat_fork_pass": _sample(records, outcome="pass", candidate="retreat_fork"),
            "returned_round_tip_pass": _sample(records, outcome="pass", candidate="round_tip"),
            "no_witness": _sample(records, outcome="no_witness"),
        },
        "state_contract": {
            "checkpoint_top_level_keys": [
                "schema", "index", "mode", "action_id", "adapter_fingerprint",
                "input_fingerprint", "input_state", "output_state", "event",
            ],
            "state_keys": ["key", "elapsed_s", "pose", "valid", "native"],
            "native_keys": ["q", "qdot", "acceleration", "snapshot"],
            "snapshot_schema": "ultrapoc_qp_complete_history_v2",
            "no_witness_contract": "output_state.valid=false, route={}, result.reason=no_checked_route; output_state.native retains full q/qdot/acceleration/snapshot",
        },
        "native_witness": _native_witness(witness),
        "failed_checks_and_throws": {
            "checkpoint_no_witness_is_mechanical_route_aggregation": True,
            "log_exception_count": len(exceptions),
            "replay_exception_count": sum(e["kind"] == "replay_exception" for e in exceptions),
            "computation_interrupt_count": sum(e["kind"] == "computation_interrupt" for e in exceptions),
            "exceptions": exceptions,
            "note": "These log throws are not added to checkpoint outcome counts; they are replay/computation diagnostics, not native pass/no-witness rows.",
        },
        "mechanical_pilot_audit": {
            "path": str(adaptive / "NATIVE_PILOT_AUDIT.txt"),
            "classification": "mechanical_native_tracking_gate_failure",
            "checkpoint": str(adaptive / "study_v3/A_task_dwell_30/fixture_00_constrained/dev/fixture_00_constrained/native_shared/planning/checkpoints/acbc475b5405d0156af7c8f0.json"),
            "summary": {
                "source": "four side_right__single_dwell_0.6 probes; fifth side_left__single_dwell_0.6",
                "result_reason": "native_probe_invalid then no_checked_route checkpoint",
                "candidate_routes": {
                    "round_tip": {"same_tick_position_error_m": 0.003993224, "rotation_error_rad": 0.006494228, "geometry_gap_m": 0.000553052, "native_failure_count": 0, "exit_state_valid": True},
                    "retreat_fork": {"same_tick_position_error_m": 0.002210762, "rotation_error_rad": 0.002572224, "geometry_gap_m": 0.000654213, "native_failure_count": 0, "exit_state_valid": True},
                },
                "reason": "both candidates fail position tracking gate >2 mm; geometry/window/wall/collision/pause/native-failure/state-continuation checks do not explain rejection",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuing-root", type=Path, default=DEFAULT_CONTINUING)
    parser.add_argument("--adaptive-root", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--witness", type=Path, default=DEFAULT_WITNESS)
    args = parser.parse_args()
    print(json.dumps(build_inventory(args.continuing_root, args.adaptive_root, args.witness), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

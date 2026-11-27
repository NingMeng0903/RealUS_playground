#!/usr/bin/env python3
"""Summarize public exploration/recovery proxies from the v4 study.

This is a read-only audit of result artifacts.  It deliberately does not read
``privateeval`` (or any hidden renderer field).  The unit of analysis is a
public acquisition: native no-witness rows that have no ``public`` mapping are
not part of the public sequence.

The summary has two views of the same 96 primary episodes:

* ``initial`` is the original ``RESULT.json`` for every primary key.
* ``latest`` chooses the artifact with the greatest ``fresh_count`` and then
  the continuation rank (continued > guard32 > guard16 > primary).  This
  makes guard/continuation artifacts reproducibly replace, rather than add to,
  their primary episode.

The same-pose metric is explicitly only a transition proxy.  A repeated pose
can still carry information; this report makes no claim that it is useless.
"Recovery" means a contiguous run of public missing acquisitions followed by
the next public detected acquisition.  Window-change metrics use adjacent
public acquisitions, so their denominators are reported explicitly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RESULT_NAMES = (
    "RESULT.json",
    "RESULT_CONTINUED.json",
    "RESULT_GUARD_16.json",
    "RESULT_GUARD_32.json",
)
ARM_ORDER = ("A_task", "A_info", "B_rollout", "pair_fixed")
FAMILY_ORDER = ("A", "B", "pair")
FAMILY_ARMS = {
    "A": ("A_task", "A_info"),
    "B": ("B_rollout",),
    "pair": ("pair_fixed",),
}
_KIND_RANK = {
    "RESULT.json": 0,
    "RESULT_GUARD_16.json": 16,
    "RESULT_GUARD_32.json": 32,
    "RESULT_CONTINUED.json": 100,
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"result is not an object: {path}")
    return value


def _kind(path: Path) -> str:
    return path.name


def _episode_key(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    """Public episode identity; pair dwell is part of the identity."""

    arm = payload.get("arm")
    mode = payload.get("mode")
    fixture = payload.get("fixture_index")
    case = payload.get("case_id")
    dwell = payload.get("pair_dwell")
    if arm is None or mode is None or fixture is None or case is None:
        raise ValueError("result lacks arm/mode/fixture_index/case_id")
    return (str(arm), str(mode), int(fixture), int(case), None if dwell is None else round(float(dwell), 9))


def _record_sort_key(path: Path, payload: Mapping[str, Any]) -> tuple[int, int, str]:
    fresh = int(payload.get("fresh_count", 0) or 0)
    return (fresh, _KIND_RANK.get(_kind(path), -1), str(path))


def _records(root: Path) -> tuple[dict[tuple[Any, ...], tuple[Path, dict[str, Any]]], dict[tuple[Any, ...], tuple[Path, dict[str, Any]]]]:
    """Load primary records and reproducibly select one latest artifact/key."""

    grouped: dict[tuple[Any, ...], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        if path.name not in RESULT_NAMES:
            continue
        payload = _load(path)
        grouped[_episode_key(payload)].append((path, payload))

    if not grouped:
        raise FileNotFoundError(f"no result artifacts under {root}")

    primary: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    latest: dict[tuple[Any, ...], tuple[Path, dict[str, Any]]] = {}
    for key, entries in grouped.items():
        originals = [(path, payload) for path, payload in entries if path.name == "RESULT.json"]
        if len(originals) != 1:
            raise AssertionError(f"expected one primary RESULT.json for {key}, got {len(originals)}")
        primary[key] = originals[0]
        latest[key] = max(entries, key=lambda item: _record_sort_key(*item))

    if set(primary) != set(latest):
        raise AssertionError("initial/latest episode denominators differ")
    return primary, latest


def _round_or_none(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _pose(action: Mapping[str, Any], public: Mapping[str, Any]) -> tuple[Any, ...]:
    """Canonical public action pose used by all proxy comparisons."""

    # Public action is expected to carry the selected action.  The public
    # packet's window is used as a fallback because it is the exposed packet
    # boundary, not a private/native field.
    window = action.get("window", public.get("window"))
    xy = action.get("xy_m")
    if isinstance(xy, Sequence) and not isinstance(xy, (str, bytes)) and len(xy) >= 2:
        x, y = xy[0], xy[1]
    else:
        x = public.get("center_offset_m")
        y = None
    return (
        None if window is None else str(window),
        _round_or_none(x),
        _round_or_none(y),
        _round_or_none(action.get("yaw_rad")),
    )


def _public_events(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract only public acquisition packets, never private evaluation."""

    result: list[dict[str, Any]] = []
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, list):
        return result
    for index, event in enumerate(raw_events):
        if not isinstance(event, Mapping):
            continue
        public = event.get("public")
        action = event.get("action")
        if not isinstance(public, Mapping) or not isinstance(action, Mapping):
            # Native no-witness/recovery bookkeeping rows are intentionally
            # excluded from the public acquisition sequence.
            continue
        detected = bool(public.get("detected", False))
        window = public.get("window", action.get("window"))
        result.append(
            {
                "event_index": index,
                "event_id": public.get("event_id", event.get("event_id")),
                "detected": detected,
                "candidate_type": public.get("candidate_type"),
                "window": None if window is None else str(window),
                "pose": _pose(action, public),
                "quality": _round_or_none(public.get("quality")),
                "center_offset_m": _round_or_none(public.get("center_offset_m")),
                "axis_angle_rad": _round_or_none(public.get("axis_angle_rad")),
            }
        )
    return result


def _sequence_metrics(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute deterministic adjacent-public-event exploration proxies."""

    n = len(events)
    missing_indices = [i for i, event in enumerate(events) if not event["detected"]]
    missing_count = len(missing_indices)
    missing_runs: list[int] = []
    current = 0
    for event in events:
        if not event["detected"]:
            current += 1
        elif current:
            missing_runs.append(current)
            current = 0
    if current:
        missing_runs.append(current)

    # A repeat proxy is an adjacent transition whose previous public packet
    # was missing and whose canonical action pose is unchanged.  It is not a
    # claim that the repeated acquisition had no information.
    repeat_transition_indices = [
        i
        for i in range(1, n)
        if not events[i - 1]["detected"] and events[i]["pose"] == events[i - 1]["pose"]
    ]
    repeat_runs: list[int] = []
    if repeat_transition_indices:
        run_len = 1
        for previous, current_index in zip(repeat_transition_indices, repeat_transition_indices[1:]):
            if current_index == previous + 1:
                run_len += 1
            else:
                repeat_runs.append(run_len)
                run_len = 1
        repeat_runs.append(run_len)

    missing_to_switch = [
        i
        for i in range(1, n)
        if not events[i - 1]["detected"]
        and events[i]["window"] != events[i - 1]["window"]
    ]
    window_switch_indices = [
        i for i in range(1, n) if events[i]["window"] != events[i - 1]["window"]
    ]
    # A->B->A is a public window-return/retraction proxy.  It counts the
    # middle-to-return transition, so each such occurrence is one event.
    window_return_indices = [
        i
        for i in range(2, n)
        if events[i - 2]["window"] != events[i - 1]["window"]
        and events[i]["window"] == events[i - 2]["window"]
    ]

    recovery_gaps: list[int] = []
    index = 0
    while index < n:
        if events[index]["detected"]:
            index += 1
            continue
        start = index
        while index < n and not events[index]["detected"]:
            index += 1
        if index < n:
            recovery_gaps.append(index - start)

    return {
        "public_event_count": n,
        "public_detected_count": n - missing_count,
        "public_missing_count": missing_count,
        "public_missing_run_count": len(missing_runs),
        "public_missing_run_lengths": missing_runs,
        "public_recovery_event_count": len(recovery_gaps),
        "public_recovery_missing_frames": sum(recovery_gaps),
        "public_recovery_gap_lengths": recovery_gaps,
        "unresolved_public_missing_run_count": len(missing_runs) - len(recovery_gaps),
        "same_pose_repeat_events_proxy": len(repeat_transition_indices),
        "same_pose_repeat_run_count_proxy": len(repeat_runs),
        "same_pose_repeat_run_lengths_proxy": repeat_runs,
        "missing_to_window_switch_events": len(missing_to_switch),
        "actual_window_switch_events": len(window_switch_indices),
        "actual_window_return_events_proxy": len(window_return_indices),
    }


def _episode_rows(records: Mapping[tuple[Any, ...], tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(records, key=str):
        path, payload = records[key]
        events = _public_events(payload)
        row: dict[str, Any] = {
            "episode_key": list(key),
            "arm": payload.get("arm"),
            "mode": payload.get("mode"),
            "fixture_index": payload.get("fixture_index"),
            "case_id": payload.get("case_id"),
            "pair_dwell": payload.get("pair_dwell"),
            "source_kind": path.name,
            "source": str(path),
            "fresh_count": int(payload.get("fresh_count", 0) or 0),
            "watchdog": bool(payload.get("watchdog", False)),
            "events": events,
        }
        row.update(_sequence_metrics(events))
        rows.append(row)
    return rows


def _sum_metric(rows: Sequence[Mapping[str, Any]], name: str) -> int:
    return sum(int(row.get(name, 0) or 0) for row in rows)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    aggregate: dict[str, Any] = {
        "episode_count": n,
        "public_event_count": _sum_metric(rows, "public_event_count"),
        "public_detected_count": _sum_metric(rows, "public_detected_count"),
        "public_missing_count": _sum_metric(rows, "public_missing_count"),
        "public_missing_run_count": _sum_metric(rows, "public_missing_run_count"),
        "public_recovery_event_count": _sum_metric(rows, "public_recovery_event_count"),
        "public_recovery_missing_frames": _sum_metric(rows, "public_recovery_missing_frames"),
        "unresolved_public_missing_run_count": _sum_metric(rows, "unresolved_public_missing_run_count"),
        "same_pose_repeat_events_proxy": _sum_metric(rows, "same_pose_repeat_events_proxy"),
        "same_pose_repeat_run_count_proxy": _sum_metric(rows, "same_pose_repeat_run_count_proxy"),
        "missing_to_window_switch_events": _sum_metric(rows, "missing_to_window_switch_events"),
        "actual_window_switch_events": _sum_metric(rows, "actual_window_switch_events"),
        "actual_window_return_events_proxy": _sum_metric(rows, "actual_window_return_events_proxy"),
    }
    for name in (
        "public_missing_run_count",
        "public_recovery_event_count",
        "same_pose_repeat_events_proxy",
        "missing_to_window_switch_events",
        "actual_window_switch_events",
        "actual_window_return_events_proxy",
    ):
        aggregate[f"episodes_with_{name}"] = sum(bool(row.get(name, 0)) for row in rows)
    aggregate["episodes_with_any_public_missing"] = sum(bool(row.get("public_missing_count", 0)) for row in rows)
    aggregate["episodes_with_any_public_recovery"] = sum(bool(row.get("public_recovery_event_count", 0)) for row in rows)
    aggregate["episodes_with_any_same_pose_repeat_proxy"] = sum(bool(row.get("same_pose_repeat_events_proxy", 0)) for row in rows)
    aggregate["episodes_with_any_window_switch"] = sum(bool(row.get("actual_window_switch_events", 0)) for row in rows)
    aggregate["episodes_with_any_window_return_proxy"] = sum(bool(row.get("actual_window_return_events_proxy", 0)) for row in rows)
    return aggregate


def _same_packet(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = ("detected", "candidate_type", "window", "pose", "quality", "center_offset_m", "axis_angle_rad")
    return all(left.get(key) == right.get(key) for key in keys)


def _consistency(rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare A_task and B event streams at matching fixture/case IDs."""

    def episode_id(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (row.get("mode"), row.get("fixture_index"), row.get("case_id"))

    by_a = {episode_id(row): row for row in rows_a}
    by_b = {episode_id(row): row for row in rows_b}
    common = sorted(set(by_a) & set(by_b), key=str)
    length_equal = 0
    common_prefix = 0
    aligned = 0
    same_detection = 0
    same_window = 0
    same_detection_window = 0
    same_packet = 0
    for key in common:
        a_events = by_a[key]["events"]
        b_events = by_b[key]["events"]
        if len(a_events) == len(b_events):
            length_equal += 1
        if a_events and b_events:
            prefix = 0
            for left, right in zip(a_events, b_events):
                if left["event_index"] != right["event_index"]:
                    # event_index is local to the stored event list, not an
                    # identity constraint; compare the public sequence itself.
                    pass
                if left["detected"] == right["detected"] and left["window"] == right["window"]:
                    prefix += 1
                else:
                    break
            if prefix == min(len(a_events), len(b_events)):
                common_prefix += 1
        for left, right in zip(a_events, b_events):
            aligned += 1
            same_detection += left["detected"] == right["detected"]
            same_window += left["window"] == right["window"]
            same_detection_window += left["detected"] == right["detected"] and left["window"] == right["window"]
            same_packet += _same_packet(left, right)
    def rate(value: int, denominator: int) -> float | None:
        return None if denominator == 0 else round(value / denominator, 6)

    return {
        "episode_pairs": len(common),
        "aligned_event_pairs": aligned,
        "equal_public_event_length_episodes": length_equal,
        "common_detection_window_prefix_episodes": common_prefix,
        "same_detection_events": same_detection,
        "same_window_events": same_window,
        "same_detection_and_window_events": same_detection_window,
        "same_public_packet_events": same_packet,
        "same_detection_rate": rate(same_detection, aligned),
        "same_window_rate": rate(same_window, aligned),
        "same_detection_and_window_rate": rate(same_detection_window, aligned),
        "same_public_packet_rate": rate(same_packet, aligned),
        "event_denominator_definition": "sum over matched fixture/case episodes of min(A_task public event count, B public event count)",
    }


def _strip_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key != "events"} for row in rows]


def summarize(root: Path) -> dict[str, Any]:
    primary, latest = _records(root)
    initial_rows = _episode_rows(primary)
    latest_rows = _episode_rows(latest)

    def by_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for arm in ARM_ORDER:
            arm_rows = [row for row in rows if row["arm"] == arm]
            result[arm] = _aggregate(arm_rows)
        return result

    def by_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            family: _aggregate([row for row in rows if row["arm"] in arms])
            for family, arms in FAMILY_ARMS.items()
        }

    def pair_dwell(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for row in rows:
            if row["arm"] != "pair_fixed":
                continue
            dwell = str(row["pair_dwell"])
            result.setdefault(dwell, []).append(row)
        return {dwell: _aggregate(group) for dwell, group in sorted(result.items())}

    initial_a = [row for row in initial_rows if row["arm"] == "A_task"]
    initial_b = [row for row in initial_rows if row["arm"] == "B_rollout"]
    latest_a = [row for row in latest_rows if row["arm"] == "A_task"]
    latest_b = [row for row in latest_rows if row["arm"] == "B_rollout"]
    output = {
        "schema": "v4_exploration_proxy_summary_v1",
        "source_root": str(root),
        "privateeval_used": False,
        "primary_episode_count": len(primary),
        "initial_latest_same_denominator": len(primary) == len(latest),
        "initial": {
            "episode_count": len(initial_rows),
            "by_arm": by_arm(initial_rows),
            "by_family": by_family(initial_rows),
            "pair_by_dwell": pair_dwell(initial_rows),
            "episodes": _strip_events(initial_rows),
            "b_vs_a_task": _consistency(initial_a, initial_b),
        },
        "latest": {
            "episode_count": len(latest_rows),
            "by_arm": by_arm(latest_rows),
            "by_family": by_family(latest_rows),
            "pair_by_dwell": pair_dwell(latest_rows),
            "episodes": _strip_events(latest_rows),
            "b_vs_a_task": _consistency(latest_a, latest_b),
        },
        "definitions": {
            "public_event": "event with mapping-valued public and action; no-witness native rows excluded",
            "missing": "public.detected is false",
            "same_pose": "same window, xy_m and yaw_rad rounded to 6 decimal places",
            "same_pose_repeat_proxy": "adjacent public transition whose previous event is missing and whose canonical pose is unchanged",
            "missing_to_window_switch": "adjacent public transition whose previous event is missing and whose window changes",
            "actual_window_switch": "adjacent public events with different windows",
            "window_return_proxy": "public A-to-B-to-A triplet, counted on the B-to-A transition",
            "public_recovery": "each contiguous missing run followed by the next detected public event; unresolved terminal runs are not recoveries",
            "latest_selection": "max(fresh_count, kind_rank, path), with continued=100, guard32=32, guard16=16, primary=0",
            "same_pose_caveat": "repeat counts are observational proxies; repeated acquisition may carry information and is not labeled useless",
            "consistency": "A_task and B_rollout matched by mode, fixture_index and case_id; rates use aligned event pairs",
        },
    }
    return output


def _print_short(output: Mapping[str, Any], output_path: Path) -> None:
    print(f"schema={output['schema']} primary={output['primary_episode_count']} same_denominator={output['initial_latest_same_denominator']}")
    for phase in ("initial", "latest"):
        print(f"[{phase}]")
        for arm in ARM_ORDER:
            row = output[phase]["by_arm"][arm]
            print(
                f"  {arm}: n={row['episode_count']} public={row['public_event_count']} "
                f"missing={row['public_missing_count']} repeat_proxy={row['same_pose_repeat_events_proxy']} "
                f"missing_to_switch={row['missing_to_window_switch_events']} "
                f"switches={row['actual_window_switch_events']} "
                f"returns={row['actual_window_return_events_proxy']} "
                f"recovery={row['public_recovery_event_count']}"
            )
        consistency = output[phase]["b_vs_a_task"]
        print(
            f"  B-vs-A_task: episodes={consistency['episode_pairs']} "
            f"aligned={consistency['aligned_event_pairs']} "
            f"same_detection_window={consistency['same_detection_and_window_rate']} "
            f"same_packet={consistency['same_public_packet_rate']}"
        )
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="study root containing RESULT artifacts")
    parser.add_argument("--output", type=Path, required=True, help="JSON summary path")
    args = parser.parse_args()
    root = args.root.resolve()
    output_path = args.output.resolve()
    summary = summarize(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _print_short(summary, output_path)


if __name__ == "__main__":
    main()

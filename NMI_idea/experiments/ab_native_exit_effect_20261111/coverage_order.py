#!/usr/bin/env python3
"""Native paired test of a prescribed three-footprint acquisition order.

Both selectors use the same measured surface, ProDMP, route checks and native
QP. The recorded quantities describe motion and requested capture dwell.
Acoustic frames and correctness of image identity are outside this test.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "ab_continuing_rollout_20261111"
sys.path.insert(0, str(SOURCE))

from fixture import SETTLE_S, local_pose, segment_duration, single_dwell_action
from native_adapter import NativeAdapter, _state_fingerprint

FOOTPRINTS = tuple(single_dwell_action(name, .30) for name in ("fork_minus", "fork_plus"))
TARGETS = ("left", "right")


def geometric_forecast(adapter, order):
    """Full-tour strong geometric comparator using the real route library."""
    start = local_pose(adapter.measured_frame, adapter.start.pose)
    total = 0.0
    for name in order:
        capture = adapter.rig.captures[name]
        paths = adapter.rig.route_options(start, capture.goal)
        if not paths:
            return None
        times = []
        for waypoints in paths.values():
            duration = 0.0
            for index, (entry, goal) in enumerate(zip((start, *waypoints[:-1]), waypoints)):
                planned = segment_duration(entry, goal)
                if capture.kind == "fork":
                    planned = max(planned, 1.10 if abs(goal[2] - entry[2]) > .12 else .80)
                dwell = capture.dwell_s if index == len(waypoints) - 1 else 0.0
                duration += (math.ceil((planned + SETTLE_S + dwell) / .005) + 1) * .005
            times.append(duration)
        total += min(times)
        start = capture.goal
    return total


def native_link(adapter, entry, action, execution=False):
    """Evaluate one link with unchanged native and trajectory guards."""
    call = adapter.execute if execution else adapter.checked
    try:
        event = call(entry, action)
    except RuntimeError as error:
        return None, {"action": action, "status": "exception", "detail": str(error)}
    route = event.result["route"]
    output = event.output if event.feasible else None
    traces = adapter.execution_output if execution else adapter.planning_output
    path = 0.0
    for record in event.trajectory_records:
        with np.load(traces / f"{record}.npz") as samples:
            xyz = samples["actual"][:, :3, 3]
            path += float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
    row = {
        "action": action,
        "status": "passed" if event.feasible else event.result["reason"],
        "route": event.route_id,
        "elapsed_s": event.duration_s,
        "path_m": path,
        "waypoints": len(route.get("steps", ())),
        "state_in": _state_fingerprint(entry),
        "state_out": _state_fingerprint(event.output),
        "traces": list(event.trajectory_records),
    }
    return output, row


def choose_greedy(orders, footprints):
    """Select by the shortest currently verified move at every real exit."""
    remaining = tuple(footprints)
    prefix = ()
    while remaining:
        choices = {}
        for trial in orders:
            order = tuple(trial["order"])
            if order[:len(prefix)] != prefix:
                continue
            if len(trial["links"]) <= len(prefix):
                continue
            link = trial["links"][len(prefix)]
            if link["status"] == "passed":
                choices[link["action"]] = link["elapsed_s"]
        if not choices:
            return None, list(prefix)
        action = min(choices, key=lambda name: (round(choices[name], 9), name))
        prefix = (*prefix, action)
        remaining = tuple(name for name in remaining if name != action)
    return next(trial for trial in orders if tuple(trial["order"]) == prefix), list(prefix)


def evaluate_orders(adapter, target, memo):
    footprints = (*FOOTPRINTS, target)
    rows = []
    for order in sorted(permutations(footprints)):
        state = adapter.start
        links = []
        for action in order:
            key = (_state_fingerprint(state), action)
            if key not in memo:
                memo[key] = native_link(adapter, state, action)
            state, row = memo[key]
            links.append(row)
            if state is None:
                break
        passes = state is not None
        rows.append({
            "order": list(order), "complete": passes, "links": links,
            "nominal_s": geometric_forecast(adapter, order),
            "motion_s": sum(row["elapsed_s"] for row in links) if passes else None,
            "path_m": sum(row["path_m"] for row in links) if passes else None,
        })
    feasible = [row for row in rows if row["complete"]]
    geometric = [row for row in rows if row["nominal_s"] is not None]
    native = min(feasible, key=lambda row: (round(row["motion_s"], 9), row["order"])) if feasible else None
    nominal = min(geometric, key=lambda row: (round(row["nominal_s"], 9), row["order"])) if geometric else None
    greedy, greedy_prefix = choose_greedy(rows, footprints)
    return {
        "target": target, "orders": rows,
        "A_greedy": greedy, "A_greedy_prefix": greedy_prefix,
        "A_geometric_tour": nominal, "B_verified_tour": native,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=int, choices=(0, 1), required=True)
    parser.add_argument("--mode", choices=("constrained", "open"), required=True)
    args = parser.parse_args()
    output = HERE / f"coverage_{args.fixture:02d}_{args.mode}"
    output.mkdir(exist_ok=True)
    with NativeAdapter(args.fixture, args.mode, output, reuse_checked=False) as adapter:
        memo = {}
        for target in TARGETS:
            result = evaluate_orders(adapter, target, memo)
            (output / f"{target}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(args.fixture, args.mode, target, {
                arm: None if result[arm] is None else (result[arm]["order"], result[arm]["motion_s"])
                for arm in ("A_greedy", "A_geometric_tour", "B_verified_tour")
            }, flush=True)
        adapter.finish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Native retry validation for optimistic H1 winners.

Each competitor's score includes every potentially admitted two-frame
terminal option. Restricting feasible retries only removes options from
that max. A winner with all its own admitted links verified and a strict
lead over every other optimistic score remains a winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_exit_witness import NativeAdapter, native_event, read_matrix, TERMINALS, _state_fingerprint


def validate(matrix, forecasts, output):
    metadata, records = read_matrix(matrix)
    report = json.loads(forecasts.read_text())
    groups = {}
    for replay in report["replays"]:
        winner = replay["choice"]["B_actual"]
        scores = {
            row["action_id"]: row["continuation"]["actual"]
            for row in replay["candidate_scores"]
        }
        best = scores[winner]
        others = max((score["value"] for name, score in scores.items() if name != winner), default=0.0)
        groups.setdefault(winner, []).append({
            "branches": replay["branches"],
            "A_task": replay["choice"]["A_task"], "A_info": replay["choice"]["A_info"],
            "B_nominal": replay["choice"]["B_nominal"],
            "upper_bound_winner": best["value"], "upper_bound_runner_up": others,
            "upper_bound_margin": best["value"] - others,
            "required": best["potentially_admitted_terminals"],
        })

    output.mkdir(parents=True, exist_ok=True)
    audits = []
    with NativeAdapter(metadata["fixture_index"], metadata["mode"], output, reuse_checked=False) as adapter:
        for probe, groups_by_n in groups.items():
            probe_exit = native_event(records[(None, probe)], adapter, adapter.start)
            results = {}
            required = {name for group in groups_by_n for name in group["required"]}
            for name in TERMINALS:
                if name not in required:
                    continue
                first_exit = native_event(records[(probe, name)], adapter, probe_exit)
                try:
                    retry = adapter.execute(first_exit, name)
                except RuntimeError as error:
                    results[name] = {"verified": False, "exception": str(error)}
                    continue
                results[name] = {
                    "verified": retry.feasible,
                    "reason": retry.result["reason"],
                    "exact_input": _state_fingerprint(first_exit),
                    "actual_output": _state_fingerprint(retry.output),
                    "route": retry.route_id,
                    "records": list(retry.trajectory_records),
                    "duration_s": retry.duration_s,
                }
            for group in groups_by_n:
                group["strict_winner_certified"] = group["upper_bound_margin"] > 1e-9 and all(
                    results[name]["verified"] for name in group["required"]
                )
            audits.append({"probe": probe, "scores": groups_by_n, "retry_replays": results})
        adapter.finish()
    result = {"matrix": str(matrix), "optimistic_forecasts": str(forecasts), "candidates": audits}
    (output / "RETRY_VALIDATION.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("forecasts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.matrix, args.forecasts, args.output)
    for winner in result["candidates"]:
        print(winner["probe"], winner["scores"], flush=True)


if __name__ == "__main__":
    main()

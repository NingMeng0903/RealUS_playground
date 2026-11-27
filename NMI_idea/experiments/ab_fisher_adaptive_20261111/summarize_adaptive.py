"""Normalize frozen experiment logs, audit the episode contract and compare paired arms."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_pairs import analyze


def normalize(path):
    result = json.loads(path.read_text())
    assert result["no_deadline"]
    count = result["count"]
    assert len(result["records"]) == count * len(result["methods"])
    label = f"fixture_{result['fixture_index']:02d}_{result['mode']}"
    rows = []
    for row in result["records"]:
        events = row["events"]
        diagnostics = [e for e in events if e["kind"] == "diagnostic"]
        terminals = [e for e in events if e["kind"] == "terminal"]
        if any(e["kind"] not in ("diagnostic", "terminal") for e in events):
            raise ValueError(f"unadmitted event: {label} {row['case_id']} {row['arm']}")
        assert events == diagnostics + terminals, "a probe followed formal acquisition"
        assert len(diagnostics) == row["diagnostic_capture_count"] <= result["diagnostic_cap"]
        assert len(terminals) == row["terminal_capture_count"] <= result["terminal_cap"]
        assert any(e["accepted"] for e in terminals) == row["success"]
        assert sum(bool(e["public_stop"]) for e in terminals) <= 1
        if any(e["public_stop"] for e in terminals):
            assert terminals[-1]["public_stop"]
        elapsed = sum(e["duration_s"] for e in events)
        assert math.isclose(elapsed, row["elapsed_s"], abs_tol=1e-8)
        wrong = row["wrong_window"][0] if terminals else {}
        if not terminals:
            category = "no_scan"
        elif wrong.get("status") != "compared":
            category = wrong.get("status", "not_compared")
        elif wrong["two_good"]:
            category = "both_good"
        elif wrong["two_bad"]:
            category = "both_bad"
        else:
            category = "selected_bad" if wrong["wrong_window"] else "selected_good"
        # Counts here are actual top-level candidate scores, not total nested expansions.
        calls = sum(d.get("diagnostics", {}).get("scored_candidate_count", 0)
                    for d in row["decisions"] if not d.get("reason", "").startswith("fixed_pair"))
        planning = row.get("wall_planning_s")
        if planning is None:
            raise ValueError("planner wall time missing")
        rows.append({
            "method": row["arm"], "case_id": row["case_id"], "layout": label,
            "success": int(row["success"]), "first_success": int(bool(terminals and terminals[0]["accepted"])),
            "wrong_window": int(wrong.get("wrong_window", False)), "probes": len(diagnostics),
            "terminals": len(terminals), "seconds": elapsed, "planning_seconds": planning,
            "false_stop": int(row["false_stop"]), "no_scan": int(not terminals),
            "utility": float(row["success"]) - result["soft_time_cost_per_s"] * elapsed,
            "used_shift": int(any(e["action_id"] == "fork_plus_shift3" for e in diagnostics)),
            "value_calls": calls, "first_action": row["first_choice"] or "none",
            "first_window": row["first_scan_action"] or "none", "wrong_window_class": category,
            "sequence": [e["action_id"] for e in events],
            "routes": [e["route_id"] for e in events],
        })
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contrast", action="append", default=[])
    args = parser.parse_args()
    results = [normalize(path) for path in args.results]
    keys = ("seed", "noise_seed", "count", "soft_time_cost_per_s", "top_k", "methods")
    for key in keys:
        if any(result[key] != results[0][0][key] for result, _ in results[1:]):
            raise ValueError(f"contradictory paired manifest field: {key}")
    rows = [row for _, group in results for row in group]
    summary = analyze(rows, [tuple(spec.split(":")) for spec in args.contrast])
    summary["inputs"] = [str(path.resolve()) for path in args.results]
    summary["protocol"] = {key: results[0][0][key] for key in keys}
    with args.output.open("x") as handle:
        json.dump(summary, handle, allow_nan=False, indent=2)
        handle.write("\n")
    for method, data in summary["methods"].items():
        print(method, data["counts"], {k: round(data["averages"][k], 4) for k in
              ("probes", "seconds", "utility", "planning_seconds", "value_calls")})


if __name__ == "__main__":
    main()

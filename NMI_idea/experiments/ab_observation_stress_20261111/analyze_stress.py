"""Diagnostics from frozen trajectories, with counterfactuals on shared public histories."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from planner import Planner, initial_belief


def source_and_stop(rows):
    summaries = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["regime"], row["method"]].append(row)
    for (regime, method), group in grouped.items():
        records = [block for row in group for block in row["blocks"]]
        stopped = [row["blocks"][-1] for row in group if row["stop"]]
        qualified = sum(block["audit"]["correct_if_stopped"] for block in stopped)
        errors = defaultdict(int)
        losses = defaultdict(lambda: {"events": 0, "recovered": 0, "longest_blocks_without_return": 0})
        for row in group:
            event = None
            length = 0
            for block in row["blocks"]:
                audit = block["audit"]
                if audit["loss_started"]:
                    event = "in_fov" if audit["target_in_fov"] else "out_of_fov"
                    length = 0
                    losses[event]["events"] += 1
                if event and audit["loss_open"]:
                    length += 1
                    losses[event]["longest_blocks_without_return"] = max(
                        losses[event]["longest_blocks_without_return"], length
                    )
                if event and audit["reacquired"]:
                    losses[event]["recovered"] += 1
                    event = None
        for block in stopped:
            audit = block["audit"]
            if not audit["correct_if_stopped"]:
                label = audit["source"] if audit["source"] != "target" else "target_unqualified"
                errors[label] += 1
        summaries.append({
            "regime": regime,
            "method": method,
            "trials": len(group),
            "acquisitions": len(records),
            "in_fov_without_target_return": sum(
                b["audit"]["target_in_fov"] and b["audit"]["source"] != "target" for b in records
            ),
            "outside_fov": sum(not b["audit"]["target_in_fov"] for b in records),
            "loss_by_onset": dict(losses),
            "stop_calibration": {
                "stops": len(stopped),
                "correct": qualified,
                "mean_model_correct_probability": float(np.mean([
                    b["public"]["posterior_correct_probability"] for b in stopped
                ])) if stopped else None,
                "empirical_correct_fraction": qualified / len(stopped) if stopped else None,
                "false_stop_source": dict(errors),
            },
        })
    return summaries


def shared_history_comparison(rows, model, stop_threshold):
    regimes = model["mode_model_ids"].tolist()
    planners = {
        regime: Planner(
            model["likelihood"], model["correct_joint"], model["joint"],
            model["public_gate"], model["mode_transitions"][index], stop_threshold,
        )
        for index, regime in enumerate(regimes)
    }
    counts = defaultdict(lambda: {
        "public_histories": 0, "adaptive_first_differs_from_fixed": 0,
        "future_capability_changes_first": 0, "future_capability_changes_any_score": 0,
        "adaptive_expected_gain_at_same_first_positive": 0,
        "sum_adaptive_expected_gain_at_same_first_s": 0.0,
        "max_adaptive_expected_gain_at_same_first_s": 0.0,
        "adaptive_outcome_uses_different_continuation": 0,
        "nonstop_with_both_continuations": 0,
        "capability_blind_predicts_unavailable_for_observed_outcome_at_same_first": 0,
        "model_top_posterior_world_probability": [],
        "expected_gain_s_by_independent_object": defaultdict(list),
    })
    free = model["motion_and_observation_s"]
    for row in rows:
        if row["method"] != "B_two":
            continue
        regime = row["model_regime"]
        planner = planners[regime]
        belief = initial_belief(planner.world_count, model["stationary_modes"][regimes.index(regime)])
        context = row["context_index"]
        actual, available = model["action_cost_s"][context], model["available"][context]
        group = counts[row["regime"], row["context"]]
        for block in row["blocks"]:
            p = block["public"]
            a, o = p["action_index"], p["outcome_index"]
            if p["ordinal"] % 2 == 0 and p["remaining_actions"] > 1:
                args = (
                    belief, p["from_index"], p["remaining_before_planning_s"],
                    p["remaining_actions"], available, actual, free,
                )
                adaptive = planner.plan("B_two", *args)
                fixed = planner.plan("B_fixed", *args)
                blind = planner.plan("B_no_future_cap", *args)
                if adaptive.action != a:
                    raise ValueError("could not reproduce adaptive first-step choice")
                legal = np.isfinite(adaptive.scores)
                gap = float(adaptive.scores[a] - fixed.scores[a])
                if gap < -1e-9:
                    raise ValueError("common-first adaptive planning lost model value")
                group["public_histories"] += 1
                group["adaptive_first_differs_from_fixed"] += int(a != fixed.action)
                group["future_capability_changes_first"] += int(a != blind.action)
                group["future_capability_changes_any_score"] += int(
                    np.max(np.abs(adaptive.scores[legal] - blind.scores[legal])) > 1e-9
                )
                group["adaptive_expected_gain_at_same_first_positive"] += int(gap > 1e-9)
                group["sum_adaptive_expected_gain_at_same_first_s"] += gap
                group["max_adaptive_expected_gain_at_same_first_s"] = max(
                    gap, group["max_adaptive_expected_gain_at_same_first_s"]
                )
                group["expected_gain_s_by_independent_object"][row["object_id"]].append(gap)
                group["model_top_posterior_world_probability"].append(float(belief.sum(axis=1).max()))
                if not p["accepted_by_posterior_gate"] and fixed.action == a:
                    b = adaptive.continuations[o]
                    if b >= 0 and fixed.fixed_action is not None:
                        group["nonstop_with_both_continuations"] += 1
                        group["adaptive_outcome_uses_different_continuation"] += int(b != fixed.fixed_action)
                if blind.action == a:
                    b = blind.continuations[o]
                    if b >= 0 and not available[b]:
                        group["capability_blind_predicts_unavailable_for_observed_outcome_at_same_first"] += 1
            if not p["accepted_by_posterior_gate"] and p["observation_probability"] > 0:
                belief = planner.advance(belief, a, o)
    result = []
    for (regime, context), group in counts.items():
        worlds = group.pop("expected_gain_s_by_independent_object")
        world_means = [np.mean(values) for values in worlds.values()]
        belief_tops = group.pop("model_top_posterior_world_probability")
        group["mean_adaptive_model_gain_at_same_first_s_by_object"] = (
            float(np.mean(world_means)) if world_means else None
        )
        group["median_top_world_weight_across_histories"] = (
            float(np.median(belief_tops)) if belief_tops else None
        )
        result.append({"regime": regime, "context": context, **group})
    return result


def paired_paths(rows):
    methods = ("A_geometry", "A_iid", "A_hard", "B_two", "B_fixed", "B_no_future_cap")
    pairs = [
        ("B_two", "A_task"), ("B_two", "B_fixed"), ("B_two", "B_no_future_cap"),
        ("A_task", "A_geometry"), ("A_task", "A_iid"), ("A_task", "A_hard"),
    ]
    groups = defaultdict(dict)
    for row in rows:
        groups[row["regime"], row["context"]][row["object_id"], row["method"]] = row
    results = []
    for (regime, context), group in groups.items():
        subjects = sorted({obj for obj, _ in group})
        for left, right in pairs:
            if any((obj, method) not in group for obj in subjects for method in (left, right)):
                continue
            entries = [(group[obj, left], group[obj, right]) for obj in subjects]
            pair = {
                "regime": regime, "context": context, "left": left, "right": right,
                "both_correct": sum(a["success"] and b["success"] for a, b in entries),
                "left_only_correct": sum(a["success"] and not b["success"] for a, b in entries),
                "right_only_correct": sum(b["success"] and not a["success"] for a, b in entries),
                "same_public_action_sequence": sum(
                    [p["public"]["action_index"] for p in a["blocks"]]
                    == [p["public"]["action_index"] for p in b["blocks"]] for a, b in entries
                ),
            }
            results.append(pair)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.results / "rows.jsonl").read_text().splitlines()]
    manifest = json.loads((args.results / "manifest.json").read_text())
    with threadpool_limits(limits=1, user_api="blas"), np.load(args.results / "model_inputs.npz") as data:
        model = {name: data[name] for name in data.files}
        comparison = shared_history_comparison(rows, model, manifest["protocol"]["task"]["stop_probability"])
    report = {
        "scope": "Post-run descriptions. Loss/recovery is target-return interrupted after an observed target; onset can be inside or outside geometric FOV. Counterfactual scores use identical public history and clock; model saved time is not measured task benefit.",
        "source_and_stop": source_and_stop(rows),
        "same_public_history": comparison,
        "paired_paths": paired_paths(rows),
    }
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()

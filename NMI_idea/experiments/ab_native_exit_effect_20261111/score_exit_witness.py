#!/usr/bin/env python3
"""Score the prespecified root-state decision with a matched public belief.

Native continuation evidence and geometric forecasts are separate inputs.
One new frame per checked terminal is supported by the motion matrix. There
is no modeled retry at an exit for which a retry trajectory was not checked.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "ab_continuing_rollout_20261111"
MODEL = SOURCE / "models" / "v3" / "MODEL.json"
sys.path.insert(0, str(SOURCE))

from adaptive_planning import (
    AdaptivePlanner,
    PlannerConfig,
    _belief_task_classes,
    _entropy,
    _relabel_packet,
    load_model_params,
    public_acceptance,
)
from belief_model import CompleteOption, predict_complete_option
from native_adapter import NativeAdapter, _state_fingerprint, _state_from_payload
from rollout_policy import task_action_scores

TERMINALS = ("left", "right")


def read_matrix(path):
    """Load the complete-state matrix with the newest result for each cell."""
    with path.open() as stream:
        document = [json.loads(line) for line in stream if line.strip()]
    manifest = next(row for row in document if row["row_type"] == "manifest")
    rows = {}
    for row in document:
        if row["row_type"] == "cell":
            probe = row["root_action"] if row["stage"] == "continuation" else None
            rows[(probe, row["action_id"])] = row
    return manifest, rows


def native_event(row, adapter, entry):
    if row is None or row["status"] in {"unknown", "exception", "cache_miss"}:
        raise ValueError("native matrix cell missing or unresolved")
    if not row["feasible"]:
        return None
    output = _state_from_payload(row["output_state"])
    if (
        row["adapter_fingerprint"]["sha256"] != adapter._adapter_fingerprint()["sha256"]
        or row["input_state"]["fingerprint"] != _state_fingerprint(entry)
        or row["output_state"]["fingerprint"] != _state_fingerprint(output)
        or output.native is None
        or not output.valid
    ):
        raise ValueError(f"invalid full-state link {row['cell_key']}")
    return output


def acceptance(packet, belief):
    params = belief.params
    return public_acceptance(
        packet,
        quality_threshold=params.public_quality_threshold,
        center_tolerance_m=params.center_tolerance_m,
        axis_tolerance_rad=params.axis_tolerance_rad,
        confidence_threshold=params.public_confidence_threshold,
    )


def option_scores(belief, terminal_actions, attempts):
    return {
        name: predict_complete_option(
            belief, CompleteOption((action,) * attempts, max_retries=attempts, gh_order=33)
        )
        for name, action in terminal_actions.items()
    }


def branch_option(predictions, candidates):
    if not candidates:
        return 0.0, 0.0, 0.0
    predictions = [predictions[name] for name in candidates]
    raw = max(p.raw_complete_p for p in predictions)
    admitted = [p for p in predictions if p.admitted]
    if not admitted:
        return 0.0, float(raw), 0.0
    selected = max(
        admitted, key=lambda p: (p.calibrated_p, p.raw_complete_p, -p.p_false_early_stop)
    )
    return float(selected.calibrated_p), float(raw), float(selected.p_false_early_stop)


def score_candidate(planner, candidate, terminal_actions, modes, n, attempts,
                    immediate_correct, immediate_false):
    belief = planner.belief
    prior_entropy = _entropy(_belief_task_classes(belief, *terminal_actions.values()))
    probabilities = {mode: np.zeros(3) for mode in modes}
    supports = dict.fromkeys(modes, 0)
    retry_links = {mode: set() for mode in modes}
    largest_raw = 0.0
    reject_weight = 0.0
    expected_entropy = 0.0
    for index, (weight, packet) in enumerate(
        planner._prediction_branches(candidate, n, depth=0, tag="A")
    ):
        if acceptance(packet, belief):
            continue
        event_id = f"plan/A/0/{candidate.action_id}/{index}"
        posterior = belief.clone()
        posterior.update(event_id, candidate.action, _relabel_packet(packet, event_id))
        reject_weight += weight
        expected_entropy += weight * _entropy(
            _belief_task_classes(posterior, *terminal_actions.values())
        )
        predictions = option_scores(posterior, terminal_actions, attempts)
        largest_raw = max(largest_raw, *(p.raw_complete_p for p in predictions.values()))
        for mode, candidates in modes.items():
            result = branch_option(predictions, candidates)
            probabilities[mode] += weight * np.asarray(result)
            supports[mode] += int(result[0] > 0.0)
            retry_links[mode].update(name for name in candidates if predictions[name].admitted)

    actual_reject = max(0.0, 1.0 - immediate_correct - immediate_false)
    if reject_weight == 0.0 and actual_reject > 1e-12:
        raise ValueError(f"no sampled rejection for {candidate.action_id} at {n} branches")
    results = {}
    for mode, summed in probabilities.items():
        continued, raw, false = summed * actual_reject / max(reject_weight, 1e-12)
        results[mode] = {
            "value": float(immediate_correct + continued),
            "expected_raw": float(immediate_correct + raw),
            "expected_false": float(immediate_false + false),
            "admitted_rejecting_branches": supports[mode],
            "supported_terminals": list(modes[mode]),
            "potentially_admitted_terminals": sorted(retry_links[mode]),
        }
    return prior_entropy - expected_entropy, reject_weight, largest_raw, results


def select_b(rows, mode):
    def key(row):
        score = row["continuation"][mode]
        return (
            -score["value"],
            -score["expected_raw"],
            score["expected_false"],
            row["canonical_index"],
        )
    return min(rows, key=key)["action_id"]


def run(matrix, n, attempts):
    metadata, records = read_matrix(matrix)
    with NativeAdapter(metadata["fixture_index"], metadata["mode"], None, backend="native") as adapter:
        planner = AdaptivePlanner(
            adapter, adapter.measured_frame, config=PlannerConfig(arm="A_task"),
            model_params=load_model_params(MODEL),
        )
        start = planner.state
        candidates = planner._probe_candidates()
        verified_probes = {}
        for candidate in candidates:
            record = records.get((None, candidate.action_id))
            state = native_event(record, adapter, start)
            if state is not None:
                verified_probes[candidate.action_id] = state
        candidates = tuple(c for c in candidates if c.action_id in verified_probes)
        if not candidates:
            raise ValueError("no verified first probe")
        terminals = {name: planner._predicted(name) for name in TERMINALS}
        terminal_actions = {name: terminal.action for name, terminal in terminals.items()}
        admitted_at_root = planner._best_terminal(planner._terminal_options())
        if admitted_at_root is not None:
            verified = native_event(records.get((None, admitted_at_root.first_id)), adapter, start)
            if verified is not None:
                return {"common_direct": admitted_at_root.first_id,
                        "root_state": _state_fingerprint(start), "branches": n}
        correct, false, _ = task_action_scores(planner.belief, [c.action for c in candidates])
        by_action = []
        for index, (candidate, c0, w0) in enumerate(zip(candidates, correct, false)):
            actual = verified_probes[candidate.action_id]
            modes = {"actual": [], "nominal": []}
            for name in TERMINALS:
                record = records.get((candidate.action_id, name))
                if native_event(record, adapter, actual) is not None:
                    modes["actual"].append(name)
                if planner._predicted(name, candidate.predicted_output or start).feasible:
                    modes["nominal"].append(name)
            ig, rejecting, largest_raw, values = score_candidate(
                planner, candidate, terminal_actions, modes, n, attempts, float(c0), float(w0)
            )
            by_action.append({
                "action_id": candidate.action_id,
                "canonical_index": index,
                "motion": _state_fingerprint(actual),
                "immediate_correct": float(c0),
                "immediate_false": float(w0),
                "sample_rejecting_mass": rejecting,
                "highest_branch_terminal_raw": largest_raw,
                "probe_motion_s": records[(None, candidate.action_id)]["transfer_time_s"],
                "information_gain": float(ig),
                "continuation": values,
            })
        a_task = planner._task_probe_v4(candidates).candidate.action_id
        a_info = planner._info_gain_probe(candidates, depth=0).candidate.action_id
        if n == planner.config.qmc_initial:
            independent_info = max(by_action, key=lambda r: (r["information_gain"], r["action_id"]))
            if independent_info["action_id"] != a_info:
                raise AssertionError("independent A_info replay disagreed")
        filtered = [r for r in by_action if r["continuation"]["actual"]["admitted_rejecting_branches"]]
        a_filter = max(filtered, key=lambda r: (r["information_gain"], r["action_id"]))["action_id"] if filtered else a_info
        return {
            "common_direct": None,
            "root_state": _state_fingerprint(start),
            "model": str(MODEL),
            "model_sha256": sha256(MODEL.read_bytes()).hexdigest(),
            "root_admission": [
                {"target": option.first_id, "admitted": option.allowed,
                 "raw": option.prediction.raw_complete_p}
                for option in planner._terminal_options()
            ],
            "branches": n,
            "terminal_attempts": attempts,
            "retry_certification": "not_applicable" if attempts == 1 else "optimistic_unverified",
            "single_terminal_raw_ceiling": (planner.belief.params.target_detect_probability
                * max(1.0 - planner.belief.params.present_to_lost, planner.belief.params.lost_to_present)),
            "choice": {
                "A_task": a_task,
                "A_info": a_info,
                "A_info_native_filter": a_filter,
                "B_actual": select_b(by_action, "actual"),
                "B_nominal": select_b(by_action, "nominal"),
            },
            "candidate_scores": by_action,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branches", nargs="+", type=int, default=(16, 64))
    parser.add_argument("--attempts", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    report = {"matrix": str(args.matrix), "replays": [run(args.matrix, n, args.attempts) for n in args.branches]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for replay in report["replays"]:
        print(replay["branches"], replay.get("choice", replay["common_direct"]))


if __name__ == "__main__":
    main()

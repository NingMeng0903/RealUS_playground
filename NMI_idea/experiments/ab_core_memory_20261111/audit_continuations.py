"""Audit two-step predictions on shared histories from the frozen dense study."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
STRESS = HERE.parent / "ab_observation_stress_20261111"
ARCHIVE = STRESS / "dense_followup/dense/test_721129"
sys.path.insert(0, str(STRESS))
from planner import Planner, _accepted, _choose, initial_belief
from sensor import channel, env


@dataclass(frozen=True)
class BranchPolicy:
    first: int
    stop_first: np.ndarray
    continuation: np.ndarray
    stop_second: np.ndarray


@dataclass(frozen=True)
class Expected:
    correct_saved_s: float
    success: float
    false_stop: float


def stop_and_continue(planner, belief, first, now, remaining, available, costs, fixed=False):
    first_assessment = planner.assess(belief, first)
    first_cost = costs[now, first]
    after_first = remaining - first_cost
    second_cost = costs[first]
    allowed = available & np.isfinite(second_cost) & (second_cost <= after_first)

    branch_weights = planner.emissions[first] * belief.ravel()
    mass, correct = planner._gate_masses(branch_weights @ planner.future_operator)
    accepted = _accepted(mass, correct, planner.threshold)
    continuation_mass = np.where(accepted, correct, 0.0).sum(axis=-1)
    time_saved = np.where(allowed, after_first - second_cost, 0.0)
    scores = continuation_mass * time_saved[None, :]
    branch_eligible = (first_assessment.probability > 0) & ~first_assessment.accepted
    scores[~branch_eligible] = 0.0

    moves = np.full(planner.outcome_count, -1, dtype=int)
    if fixed:
        action = _choose(scores.sum(axis=0), second_cost, allowed)
        if action is not None:
            moves[branch_eligible] = action
    elif allowed.any():
        choices = np.flatnonzero(allowed)
        ordered = choices[np.lexsort((choices, second_cost[choices]))]
        moves[branch_eligible] = ordered[scores[branch_eligible][:, ordered].argmax(axis=-1)]
    active = np.flatnonzero(moves >= 0)
    stops = np.zeros((planner.outcome_count, len(planner.public_columns)), dtype=bool)
    stops[active] = accepted[active, moves[active]]
    return BranchPolicy(first, first_assessment.accepted, moves, stops)


def evaluate_policy(policy, belief, likelihood, correct, transition, now, remaining, costs, gate):
    first = policy.first
    branch_weights = belief[:, :, None] * likelihood[:, :, first, :]
    branch_good = np.einsum("wh,who->o", belief, correct[:, :, first, :])
    stopped = policy.stop_first
    completed = float(branch_good[stopped].sum())
    false = float(branch_weights[:, :, stopped].sum() - completed)
    saved = (remaining - costs[now, first]) * completed

    active = np.flatnonzero(policy.continuation >= 0)
    if active.size:
        second = policy.continuation[active]
        predicted = np.einsum("who,hk->owk", branch_weights[:, :, active], transition)
        prob = np.take(likelihood[..., gate], second, axis=2).transpose(2, 0, 1, 3)
        success = np.take(correct[..., gate], second, axis=2).transpose(2, 0, 1, 3)
        stop = policy.stop_second[active]
        second_mass = np.einsum("nwh,nwhg,ng->n", predicted, prob, stop)
        second_good = np.einsum("nwh,nwhg,ng->n", predicted, success, stop)
        completed += float(second_good.sum())
        false += float((second_mass - second_good).sum())
        saved += float(second_good @ (remaining - costs[now, first] - costs[first, second]))
    return Expected(float(saved), float(completed), max(0.0, float(false)))


def model_value(policy, planner, belief, now, remaining, costs):
    return evaluate_policy(
        policy, belief, planner.likelihood, planner.correct_joint, planner.transition,
        now, remaining, costs, planner.public_columns,
    )


def oracle_second(policy, planner, belief, actual_belief, likelihood, correct,
                  now, remaining, available, costs, *, public_stop_only=False):
    """Choose a second pose for audit using true marginals and the shared time."""
    first = policy.first
    after_first = remaining - costs[now, first]
    cost = costs[first]
    allowed = available & np.isfinite(cost) & (cost <= after_first)
    first_mass = planner.emissions[first] * belief.ravel()
    modeled, good = planner._gate_masses(first_mass @ planner.future_operator)
    accepted = _accepted(modeled, good, planner.threshold)
    if public_stop_only:
        accepted = np.ones_like(accepted)
    prior_branches = actual_belief[:, :, None] * likelihood[:, :, first, :]
    predicted = np.einsum("who,hk->owk", prior_branches, planner.transition)
    good_private = np.einsum(
        "owh,whag->oag", predicted, correct[..., planner.public_columns],
    )
    scores = (accepted * good_private).sum(axis=-1) * np.where(allowed, after_first - cost, 0.0)
    branch_eligible = (prior_branches.sum(axis=(0, 1)) > 0) & ~policy.stop_first
    moves = np.full(planner.outcome_count, -1, dtype=int)
    stops = np.zeros_like(policy.stop_second)
    choices = np.flatnonzero(allowed)
    if choices.size:
        order = choices[np.lexsort((choices, cost[choices]))]
        active = np.flatnonzero(branch_eligible)
        moves[active] = order[scores[active][:, order].argmax(axis=-1)]
        stops[active] = accepted[active, moves[active]]
    return BranchPolicy(first, policy.stop_first, moves, stops)


def public_policies(planner, alternate, belief, belief_alt, now, remaining, available, costs):
    args = (belief, now, remaining, 2, available, costs, costs)
    a = planner.plan("A_task", *args)
    b = planner.plan("B_two", *args)
    fixed = planner.plan("B_fixed", *args)
    if a.action is None or b.action is None or fixed.action is None:
        raise ValueError("no affordable current action")
    common = (now, remaining, available, costs)
    policies = {
        "greedy_first_then_adapt": stop_and_continue(planner, belief, a.action, *common),
        "two_adaptive": stop_and_continue(planner, belief, b.action, *common),
        "two_fixed": stop_and_continue(planner, belief, fixed.action, *common, fixed=True),
        "fixed_at_adaptive_first": stop_and_continue(planner, belief, b.action, *common, fixed=True),
    }
    values = {
        name: model_value(policy, planner, belief, now, remaining, costs)
        for name, policy in policies.items()
    }
    if not np.isclose(values["two_adaptive"].correct_saved_s, b.scores[b.action], atol=1e-10):
        raise AssertionError("adaptive primary forecast does not match original planner")
    if not np.isclose(values["two_fixed"].correct_saved_s, fixed.scores[fixed.action], atol=1e-10):
        raise AssertionError("fixed primary forecast does not match original planner")
    if not np.array_equal(policies["two_adaptive"].continuation, b.continuations):
        raise AssertionError("branch actions differ from original planner")

    forecast_gap = values["two_adaptive"].correct_saved_s - values["greedy_first_then_adapt"].correct_saved_s
    alt_values = {
        name: model_value(policies[name], alternate, belief_alt, now, remaining, costs)
        for name in ("two_adaptive", "greedy_first_then_adapt")
    }
    alt_gap = alt_values["two_adaptive"].correct_saved_s - alt_values["greedy_first_then_adapt"].correct_saved_s
    agree = forecast_gap > 1e-10 and alt_gap > 1e-10
    chosen = "two_adaptive" if agree else "greedy_first_then_adapt"
    policies["both_catalogues_positive"] = policies[chosen]
    values["both_catalogues_positive"] = values[chosen]
    return policies, values, float(alt_gap), agree


def alt_planner(poses, public_gate, transition, threshold):
    worlds = env.sample_worlds(721212, 96)
    parts = [channel(worlds, poses[start:start + 20]) for start in range(0, len(poses), 20)]
    likelihood = np.concatenate([c.likelihood for c in parts], axis=2)
    correct = np.concatenate([c.correct_joint for c in parts], axis=2)
    joint = np.concatenate([c.joint for c in parts], axis=2)
    return Planner(likelihood, correct, joint, public_gate, transition, threshold)


def load_audit(archive):
    with np.load(archive / "model_inputs.npz") as src:
        names = (
            "poses", "likelihood", "correct_joint", "joint", "public_gate", "mode_model_ids",
            "mode_transitions", "stationary_modes", "available", "action_cost_s",
        )
        model = {name: src[name] for name in names}
    with np.load(archive / "renderer_audit.npz") as src:
        truth_joint = src["joint"]
        qualified = src["target_qualified"]
    truth_likelihood = truth_joint.sum(axis=-2)
    truth_correct = truth_joint[..., 0, :] * qualified[:, None, :, None] * model["public_gate"]
    return model, truth_likelihood, truth_correct


def run(archive):
    model, truth_likelihood, truth_correct = load_audit(archive)
    manifest = json.loads((archive / "manifest.json").read_text())
    mode = model["mode_model_ids"].tolist().index("bursty")
    transition = model["mode_transitions"][mode]
    stationary = model["stationary_modes"][mode]
    threshold = manifest["protocol"]["task"]["stop_probability"]
    planner = Planner(
        model["likelihood"], model["correct_joint"], model["joint"],
        model["public_gate"], transition, threshold,
    )
    alternate = alt_planner(model["poses"], model["public_gate"], transition, threshold)
    choices_cache = {}
    with (archive / "rows.jsonl").open() as file:
        for line in file:
            row = json.loads(line)
            if row["method"] != "A_task":
                continue
            obj, ctx = row["object_id"], row["context_index"]
            belief = initial_belief(planner.world_count, stationary)
            belief_alt = initial_belief(alternate.world_count, stationary)
            belief_true = stationary[None, :].copy()
            likelihood = truth_likelihood[obj:obj + 1]
            correct = truth_correct[obj:obj + 1]
            available, costs = model["available"][ctx], model["action_cost_s"][ctx]
            for ordinal, block in enumerate(row["blocks"][:3]):
                public = block["public"]
                now, remaining = public["from_index"], public["remaining_before_planning_s"]
                if public["remaining_actions"] < 2 or not np.any(available & (costs[now] <= remaining)):
                    break
                if ordinal == 0:
                    key = (ctx, now, remaining)
                    if key not in choices_cache:
                        choices_cache[key] = public_policies(
                            planner, alternate, belief, belief_alt, now, remaining, available, costs,
                        )
                    policies, forecast, alt_gap, switched = choices_cache[key]
                else:
                    policies, forecast, alt_gap, switched = public_policies(
                        planner, alternate, belief, belief_alt, now, remaining, available, costs,
                    )
                true_values = {
                    name: evaluate_policy(
                        policy, belief_true, likelihood, correct, transition,
                        now, remaining, costs, planner.public_columns,
                    )
                    for name, policy in policies.items()
                }
                for name, public_only in [
                    ("oracle_second_shared_gate", False),
                    ("oracle_second_public_gate_only", True),
                ]:
                    oracle = oracle_second(
                        policies["two_adaptive"], planner, belief, belief_true, likelihood, correct,
                        now, remaining, available, costs, public_stop_only=public_only,
                    )
                    true_values[name] = evaluate_policy(
                        oracle, belief_true, likelihood, correct, transition,
                        now, remaining, costs, planner.public_columns,
                    )
                yield {
                    "object": obj, "context": row["context"], "prefix": ordinal,
                    "forecast": {name: vars(v) for name, v in forecast.items()},
                    "true": {name: vars(v) for name, v in true_values.items()},
                    "first": {name: p.first for name, p in policies.items()},
                    "alternate_forecast_gap_s": alt_gap, "both_catalogues_positive": switched,
                }
                action, outcome = public["action_index"], public["outcome_index"]
                belief = planner.advance(belief, action, outcome)
                belief_alt = alternate.advance(belief_alt, action, outcome)
                q = belief_true * likelihood[:, :, action, outcome]
                belief_true = (q / q.sum()) @ transition


def mean_and_interval(rows, method, baseline, key):
    by_object = defaultdict(list)
    for row in rows:
        by_object[row["object"]].append(row["true"][method][key] - row["true"][baseline][key])
    paired = np.array([np.mean(deltas) for deltas in by_object.values()])
    indices = np.random.default_rng(721213).integers(len(paired), size=(4000, len(paired)))
    return {
        "mean_by_object": float(paired.mean()),
        "object_bootstrap_95": np.quantile(paired[indices].mean(axis=1), [.025, .975]).tolist(),
    }


def summarize(rows):
    comparisons = [
        ("two_adaptive", "greedy_first_then_adapt"),
        ("two_adaptive", "two_fixed"),
        ("two_adaptive", "fixed_at_adaptive_first"),
        ("both_catalogues_positive", "greedy_first_then_adapt"),
    ]
    results = {}
    for method, baseline in comparisons:
        forecast = np.array([
            r["forecast"][method]["correct_saved_s"] - r["forecast"][baseline]["correct_saved_s"]
            for r in rows
        ])
        truth = np.array([
            r["true"][method]["correct_saved_s"] - r["true"][baseline]["correct_saved_s"]
            for r in rows
        ])
        different_first = sum(r["first"][method] != r["first"][baseline] for r in rows)
        predicted_positive = forecast > 1e-10
        results[f"{method}_vs_{baseline}"] = {
            "different_first": different_first,
            "mean_forecast_gap_s": float(forecast.mean()),
            "mean_true_gap_s": float(truth.mean()),
            "forecast_positive": int(predicted_positive.sum()),
            "true_positive": int((truth > 1e-10).sum()),
            "forecast_positive_true_not_positive": int((predicted_positive & (truth <= 1e-10)).sum()),
            "forecast_positive_true_negative": int((predicted_positive & (truth < -1e-10)).sum()),
            "paired_true_deltas": {
                metric: mean_and_interval(rows, method, baseline, metric)
                for metric in ("correct_saved_s", "success", "false_stop")
            },
        }
    return {
        "prefixes": len(rows), "independent_objects": len({r["object"] for r in rows}),
        "comparisons": results,
        "audit_only_true_oracle_deltas_from_two_adaptive": {
            method: {
                metric: mean_and_interval(rows, method, "two_adaptive", metric)
                for metric in ("correct_saved_s", "success", "false_stop")
            }
            for method in ("oracle_second_shared_gate", "oracle_second_public_gate_only")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        rows = list(run(args.archive))
    with args.output.open("x") as file:
        json.dump({
            "scope": "Retrospective model and exact synthetic conditional return at archived shared prefixes",
            "all": summarize(rows), "initial": summarize([r for r in rows if r["prefix"] == 0]),
            "rows": rows,
        }, file, indent=2, allow_nan=False)
        file.write("\n")


if __name__ == "__main__":
    main()

"""Recompute paired outcomes, real option activation, and execution margins."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


COSTS = ("path_mm", "turn_deg", "motion_s")
MARGINS = np.array([2., 2., .10])
GAP = .05
HARM_MARGIN = .02
HANDOFF, COMPLEMENT, RECOVER = 0, 1, 2


def verify(row, edges, targets):
    progress = 0
    false = False
    route = row["trace"]
    assert len(route) == row["steps"]
    expected_cost = np.zeros(3)
    clock = 0.
    for i, step in enumerate(route):
        if i:
            assert step["start"] == route[i - 1]["goal"]
        assert step["slot"] == i
        edge = edges[step["start"], step["goal"]]
        clock += edge["duration_s"]
        assert abs(step["t_s"] - clock) < 1.e-8
        costs = [edge[k] for k in ("travel_mm", "turn_deg", "rail_mm")]
        logged = [step[k] for k in ("travel_mm", "turn_deg", "rail_mm")]
        np.testing.assert_allclose(costs, logged, atol=1.e-8, rtol=0.)
        expected_cost += costs
        expected_claim = (step["observation"] > 0 and step["confidence"] >= .87
                          and step["site"] == targets[progress])
        assert step["claim"] == expected_claim
        if step["claim"]:
            progress += 1
            false |= step["eval_only_kind"] != 0
    np.testing.assert_allclose(expected_cost, [row[k] for k in ("path_mm", "turn_deg", "rail_mm")],
                               atol=1.e-7, rtol=0.)
    if route:
        assert abs(row["motion_s"] - route[-1]["t_s"]) < 1.e-8
    assert row["progress"] == progress
    assert row["success"] == (progress == len(targets) and not false)
    assert row["wrong"] == false
    assert row["option_steps"] == sum(step["phase"] != HANDOFF for step in route)
    if row["handoff_at"] is not None:
        assert row["handoff_state"] is not None
        assert all(s["phase"] == HANDOFF for s in route[row["handoff_at"]:])


def binary_difference(a, b):
    return np.array([int(i) - int(j) for i, j in zip(a, b)], dtype=float)


def compare(arm, base):
    n = len(base)
    wrong = binary_difference([r["wrong"] for r in arm], [r["wrong"] for r in base])
    success = binary_difference([r["success"] for r in arm], [r["success"] for r in base])
    cost = np.array([[a[k] - b[k] for k in (*COSTS, "measured_combined_s")] for a, b in zip(arm, base)])
    data = np.column_stack((success, wrong, cost))
    rng = np.random.default_rng(57216)
    boot = data[rng.integers(n, size=(20000, n))].mean(axis=1)
    effect = np.quantile(boot[:, 0], [.025, .975])
    # The harmful-disagreement probability upper-bounds the paired risk
    # increase and is valid even when no wrong-claim discordance was seen.
    harm = int((wrong > 0).sum())
    risk_upper = binomtest(harm, n, alternative="less").proportion_ci(.975).high
    alpha = .05 / (2 * len(COSTS))
    cost_ci = np.quantile(boot[:, 2:5], [alpha, 1. - alpha], axis=0).T
    equivalent = bool(np.all(cost_ci[:, 0] >= -MARGINS) and np.all(cost_ci[:, 1] <= MARGINS))
    order = lambda r: tuple(i["goal"] for i in r["trace"])
    return dict(n=n, success_delta=float(success.mean()), success_ci95=effect.tolist(),
                wins_losses=[int((success > 0).sum()), int((success < 0).sum())],
                wrong_delta=float(wrong.mean()),
                wrong_discordances=[harm, int((wrong < 0).sum())],
                wrong_harm_975_upper=float(risk_upper), cost_delta=cost.mean(axis=0).tolist(),
                cost_bonferroni_95=cost_ci.tolist(), equivalent=equivalent,
                combined_time_ci95=np.quantile(boot[:, 5], [.025, .975]).tolist(),
                first_changed=sum(order(a)[:1] != order(b)[:1] for a, b in zip(arm, base)),
                route_changed=sum(order(a) != order(b) for a, b in zip(arm, base)),
                strict_gate=bool(success.mean() >= GAP and effect[0] > 0.
                                 and risk_upper <= HARM_MARGIN and equivalent))


def arm_summary(cohort):
    return dict(correct=sum(r["success"] for r in cohort), wrong=sum(r["wrong"] for r in cohort),
                n=len(cohort), option_activations=sum(r["option_steps"] > 0 for r in cohort),
                executed_option_lengths=dict(Counter(r["option_steps"] for r in cohort)),
                early_handoff=sum(r["handoff_at"] is not None and 0 < r["option_steps"] < r["option"][1]
                                  for r in cohort),
                with_complement=sum(any(e["phase"] == COMPLEMENT for e in r["trace"]) for r in cohort),
                with_recovery=sum(any(e["phase"] == RECOVER for e in r["trace"]) for r in cohort),
                mean={k: float(np.mean([r[k] for r in cohort]))
                      for k in (*COSTS, "rail_mm", "option_planning_cpu_s", "online_loop_cpu_s",
                                "measured_combined_s", "min_ess")})


def headroom_report(rows):
    stats = {}
    for task in sorted({r["task"] for r in rows}):
        entries = [r for r in rows if r["task"] == task]
        base, chosen, hindsight = [], [], []
        for row in entries:
            results = defaultdict(list)
            for outcome in row["results"]:
                results[tuple(outcome["option"])].append(outcome)
            baseline = next(key for key in results if key[0] < 0)
            rates = lambda key: np.mean([r["success"] - r["wrong"] for r in results[key][::2]])
            selected = max(results, key=lambda key: (rates(key), key == baseline))
            base.append(np.mean([r["success"] for r in results[baseline][1::2]]))
            chosen.append(np.mean([r["success"] for r in results[selected][1::2]]))
            hindsight.append(np.mean(np.max(np.array([[r["success"] for r in o]
                                                   for o in results.values()]), axis=0)))
        delta = np.array(chosen) - base
        stats[task] = dict(n=len(entries), baseline=float(np.mean(base)), split_selection=float(np.mean(chosen)),
                           per_case_paired_delta=delta.tolist(),
                           same_noise_hindsight=float(np.mean(hindsight)))
    return stats


def run(cases, bank, settings):
    nodes = sorted(bank["nodes"], key=lambda row: row["id"])
    ids = {int(n["id"]): i for i, n in enumerate(nodes)}
    edges = {(ids[e["start"]], ids[e["goal"]]): e for e in bank["edges"]}
    groups = defaultdict(dict)
    headroom = []
    for row in cases:
        if row["arm"] == "evaluator_headroom":
            headroom.append(row)
            continue
        verify(row, edges, settings["sites_and_labels"][row["task"]][0])
        key = row["task"], row["initial"]
        entry = (row["index"], row["arm"])
        assert entry not in groups[key]
        groups[key][entry] = row
    result = {}
    for key, entries in sorted(groups.items()):
        indices = sorted({idx for idx, _ in entries})
        assert len(indices) == settings["n"]
        arms = sorted({arm for _, arm in entries})
        cohorts = {arm: [entries[idx, arm] for idx in indices] for arm in arms}
        base = cohorts["MPC"]
        for arm in arms:
            for a, b in zip(cohorts[arm], base):
                assert all(a[k] == b[k] for k in ("case_seed", "start_state", "perturbation_hash",
                                                  "start_observation", "task", "initial"))
                if not a["option_steps"]:
                    # Planning wall time can differ; all actual MPC decisions
                    # and observations on a zero-length option must coincide.
                    assert a["trace"] == b["trace"]
        result["/".join(key)] = dict(
            arms={arm: arm_summary(rows) for arm, rows in cohorts.items()},
            vs_mpc={arm: compare(rows, base) for arm, rows in cohorts.items() if arm != "MPC"},
            b_vs_other={arm: compare(cohorts["B"], cohorts[arm]) for arm in arms
                        if arm not in ("B", "MPC") and "B" in cohorts},
        )
    return dict(split=settings["split"], cap_steps=settings["captures"], motion_metrics=COSTS,
                paired=result, headroom=headroom_report(headroom))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--motion", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads((args.folder / "settings.json").read_text())
    bank = json.loads(args.motion.read_text())
    rows = [json.loads(line) for line in (args.folder / "cases.jsonl").read_text().splitlines()]
    report = run(rows, bank, settings)
    (args.folder / "paired.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

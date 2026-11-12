"""Pair per-case benchmarks and summarize endpoints and sampled motion costs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


COST_KEYS = ("travel_mm", "turn_deg", "elapsed_s", "stroke_mm")
COST_MARGINS = np.array([2., 2., .10])


def route(row):
    return tuple(tuple(event.get("action", event.get("goal"))) for event in row["steps"])


def paired_stats(arm, base):
    p = np.array([row["success"] for row in arm])
    q = np.array([row["success"] for row in base])
    wrong_a = np.array([row["wrong"] for row in arm])
    wrong_b = np.array([row["wrong"] for row in base])
    diff = np.column_stack((
        p.astype(float) - q, wrong_a.astype(float) - wrong_b,
        np.array([row["costs"] for row in arm]) - np.array([row["costs"] for row in base]),
    ))
    n = len(diff)
    rng = np.random.default_rng(20261113)
    boot = diff[rng.integers(n, size=(10000, n))].mean(axis=1)
    alpha = .05 / (2 * len(COST_MARGINS))
    cost_bounds = np.quantile(boot[:, 2:5], [alpha, 1. - alpha], axis=0).T
    wins, losses = int((p & ~q).sum()), int((q & ~p).sum())
    harm, avoided = int((wrong_a & ~wrong_b).sum()), int((wrong_b & ~wrong_a).sum())
    # Bound the probability of a harmful discordance alone. This is a
    # conservative population upper bound for the paired risk difference.
    harm_upper = (1. if harm == n else
                  float(binomtest(harm, n, alternative="less").proportion_ci(.975).high))
    routes = [(route(a), route(b)) for a, b in zip(arm, base)]
    return {
        "n_independent_cases": n, "success_difference": float(diff[:, 0].mean()),
        "success_wins_losses": [wins, losses],
        "success_difference_bootstrap_95": np.quantile(boot[:, 0], [.025, .975]).tolist(),
        "success_exact_p": float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.,
        "wrong_difference": float(diff[:, 1].mean()), "wrong_increased_prevented": [harm, avoided],
        "wrong_harm_conservative_975_upper": harm_upper,
        "cost_differences": dict(zip(COST_KEYS, diff[:, 2:].mean(axis=0).tolist())),
        "cost_bonferroni_95": dict(zip(COST_KEYS[:3], cost_bounds.tolist())),
        "cost_equivalence": bool(np.all(cost_bounds[:, 0] >= -COST_MARGINS)
                                 and np.all(cost_bounds[:, 1] <= COST_MARGINS)),
        "route_changed": sum(a != b for a, b in routes),
        "first_changed": sum(a[:1] != b[:1] for a, b in routes),
    }


def run(path):
    groups = defaultdict(dict)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            group = tuple(row[key] for key in ("task", "reserve", "estimator") if key in row)
            index = row["index"], row["arm"]
            if index in groups[group]:
                raise ValueError(f"duplicate case {group} {index}")
            groups[group][index] = row
    result = {}
    for group, data in sorted(groups.items()):
        idx = sorted({case for case, _ in data})
        arms = sorted({arm for _, arm in data})
        rows = {arm: [data[i, arm] for i in idx] for arm in arms}
        if any(len({r["seed"] for r in group_rows}) != len(idx) for group_rows in rows.values()):
            raise ValueError("the cell has repeated independent seeds")
        if any([r["seed"] for r in rows[arm]] != [r["seed"] for r in rows["MPC"]]
               for arm in arms):
            raise ValueError("unpaired seed")
        table = {}
        for arm, cohort in rows.items():
            node_counts = [sum(e.get("search_nodes", 0) for e in row["steps"]) for row in cohort]
            table[arm] = {
                "n": len(cohort), "success": sum(row["success"] for row in cohort),
                "wrong": sum(row["wrong"] for row in cohort),
                "mean_costs": dict(zip(COST_KEYS, np.mean([row["costs"] for row in cohort], axis=0).tolist())),
                "reasons": dict(Counter(row["reason"] for row in cohort)),
                "search_nodes": sum(node_counts),
            }
        contrasts = {}
        for arm in arms:
            if arm != "MPC":
                contrasts[arm + " vs MPC"] = paired_stats(rows[arm], rows["MPC"])
            for comparator in ("MPC_rec1", "B_feedback", "Ref_fixed", "Near_fixed"):
                if arm in ("B_status", "B_bound", "B_challenge") and comparator in arms:
                    contrasts[arm + " vs " + comparator] = paired_stats(rows[arm], rows[comparator])
        result["/".join(group)] = {"table": table, "paired": contrasts}
        audits = [r for r in rows["MPC"] if r.get("evaluator_root_audit")]
        if audits:
            result["/".join(group)]["root_hindsight"] = {
                "baseline_success": sum(row["success"] for row in audits),
                "any_first_success": sum(any(a["success"] for a in row["evaluator_root_audit"])
                                         for row in audits),
                "n": len(audits),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.cases), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

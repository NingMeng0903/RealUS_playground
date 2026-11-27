"""Separate same-tape hindsight from repeat-validated continuation opportunity."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from paired_metrics import FIELD_NAMES, summarize


BASE = "MPC"
MARGINS = np.array([2., 2., .10])


def read_rows(folder):
    with (Path(folder) / "cases.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def cohorts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"], row["depth"]].append(row)
    return {key: sorted(value, key=lambda row: row["index"]) for key, value in sorted(groups.items())}


def array_outcomes(row):
    return {key: np.asarray(values) for key, values in row["outcomes"].items()}


def rank_repeat(outcomes, train, cost_screen):
    means = {key: a[train].mean(axis=0) for key, a in outcomes.items()}
    base = means[BASE]
    ranked = [BASE]
    ranked.extend(key for key in sorted(means) if key != BASE and (
        not cost_screen or np.all(np.abs(means[key][2:5] - base[2:5]) <= MARGINS + 1.e-10)))
    return max(ranked, key=lambda key: means[key][0]), len(ranked) - 1


def split_selector(rows, cost_screen):
    choices = Counter()
    admitted = Counter()
    executed, baselines = [], []
    for row in rows:
        vals = array_outcomes(row)
        baseline = vals[BASE]
        mid = len(baseline) // 2
        if mid == 0 or mid * 2 != len(baseline):
            raise ValueError("opportunity requires two equally sized independent repeat halves")
        halves = [slice(0, mid), slice(mid, None)]
        selected = []
        for rank_slice, test_slice in zip(halves, reversed(halves)):
            best, n = rank_repeat(vals, rank_slice, cost_screen)
            choices[best] += 1
            admitted[n] += 1
            selected.append(vals[best][test_slice])
        executed.append(np.concatenate(selected[::-1]))
        baselines.append(baseline)
    return dict(summarize(executed, baselines), ranking_choices=dict(choices),
                number_cost_admissible_other_than_mpc=dict(admitted))


def cohort_choices(outcomes, rank_slice):
    cases, keys, deltas = [], [], []
    for i, vals in enumerate(outcomes):
        base = vals[BASE][rank_slice].mean(axis=0)
        for key in (BASE, *sorted(k for k in vals if k != BASE)):
            cases.append(i)
            keys.append(key)
            deltas.append(vals[key][rank_slice].mean(axis=0) - base)
    n, width = len(outcomes), len(keys)
    deltas = np.asarray(deltas)
    costs = deltas[:, 2:5] / MARGINS
    cols = np.arange(width)
    counts = coo_matrix((np.ones(width), (cases, cols)), shape=(n, width)).tocsc()
    constraints = [LinearConstraint(counts, np.ones(n), np.ones(n)), LinearConstraint(costs.T, -n, n)]
    # This small preference for MPC cannot change the optimum correct count.
    tie = 1.e-6 * np.asarray([key != BASE for key in keys])
    result = milp(-deltas[:, 0] + tie, integrality=np.ones(width),
                  bounds=Bounds(0., 1.), constraints=constraints,
                  options={"mip_rel_gap": .005, "time_limit": 5.})
    if result.x is None:
        raise ValueError(f"cohort optimizer failed without an incumbent: {result.message}")
    pick = np.flatnonzero(result.x > .5)
    assert len(pick) == n and [cases[i] for i in pick] == list(range(n))
    assert np.all(np.abs(costs[pick].mean(axis=0)) <= 1. + 1.e-7)
    metadata = dict(status=int(result.status), message=result.message,
                    mip_gap=float(result.mip_gap), predicted_deltas=deltas[pick].mean(axis=0).tolist())
    return [keys[i] for i in pick], metadata


def cohort_selector(rows):
    outcomes = [array_outcomes(row) for row in rows]
    base = np.asarray([vals[BASE] for vals in outcomes])
    mid = base.shape[1] // 2
    halves = [slice(0, mid), slice(mid, None)]
    selected = base.copy()
    choices, solver_runs = Counter(), []
    for rank_slice, test_slice in zip(halves, reversed(halves)):
        keys, stats = cohort_choices(outcomes, rank_slice)
        solver_runs.append(dict(stats, chosen_by_case_index=keys))
        choices.update(keys)
        for i, key in enumerate(keys):
            selected[i, test_slice] = outcomes[i][key][test_slice]
    return dict(summarize(selected, base), ranking_choices=dict(choices), solver_runs=solver_runs,
                interval_scope="Descriptive across cases, conditional on the two fitted allocations.")


def hindsight(rows):
    max_success, baseline = [], []
    for row in rows:
        vals = array_outcomes(row)
        base = vals[BASE]
        max_success.append(np.max([out[:, 0] for out in vals.values()], axis=0))
        baseline.append(base[:, 0])
    difference = np.array(max_success) - np.array(baseline)
    by_case = difference.mean(axis=1)
    rng = np.random.default_rng(57217)
    boot = by_case[rng.integers(len(rows), size=(20000, len(rows)))].mean(axis=1)
    return dict(correct_gain_ceiling_sample=float(by_case.mean()),
                case_cluster_ci95=np.quantile(boot, [.025, .975]).tolist(),
                cases_with_positive_tape=int(np.count_nonzero(by_case)),
                fraction_incumbent_failures_rescued=float(difference.sum() / max(1, np.sum(1 - np.array(baseline)))))


def route_census(rows):
    routes = defaultdict(list)
    deltas = defaultdict(list)
    for row in rows:
        outcomes = array_outcomes(row)
        base = outcomes[BASE]
        for key, val in outcomes.items():
            if key == BASE:
                continue
            routes[key].append(row["actions"][key]["prefix"])
            deltas[key].append((val - base).mean(axis=0))
    result = {}
    for key, route in routes.items():
        available = len(route)
        effect = np.asarray(deltas[key]).mean(axis=0)
        result[key] = dict(prefix=route[0], available_cases=available,
                           conditional_mean_delta=dict(zip(FIELD_NAMES, effect.tolist())),
                           fallback_all_case_delta=dict(zip(FIELD_NAMES, (effect * available / len(rows)).tolist())))
    return result


def analyze(rows):
    result = {}
    for (task, depth), group in cohorts(rows).items():
        result[f"{task}/{depth}"] = dict(n=len(group),
            n_repeats=len(group[0]["outcomes"][BASE]),
            baseline_first=dict(Counter(row["public"]["baseline"] for row in group)),
            library_size=dict(Counter(len(row["actions"]) for row in group)),
            label_assisted_hard_cap=split_selector(group, cost_screen=False),
            label_assisted_cost_screen=split_selector(group, cost_screen=True),
            label_assisted_cohort_cost=cohort_selector(group),
            same_tape_hard_cap_ceiling=hindsight(group),
            route_census=route_census(group))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    report = analyze(read_rows(args.folder))
    (args.folder / "opportunity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label, r in report.items():
        msg = []
        for key in ("label_assisted_hard_cap", "label_assisted_cost_screen", "label_assisted_cohort_cost"):
            v = r[key]
            delta = v["delta"]["correct_completion"]
            interval = v["delta_ci95_case_clustered"]["correct_completion"]
            msg.append(f"{key} {delta:+.2%} {interval}, motion-equivalent {v['all_costs_equivalent']}")
        print(label, *msg, sep=" | ")


if __name__ == "__main__":
    main()

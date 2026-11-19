"""Recompute confirmation counts, deployed choices, and paired inference from raw rows."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import lzma
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, binomtest


MARGINS = np.array([2., 2., .1])
BASE = "MPC"


def case_stream(folder):
    raw = folder / "cases.jsonl"
    return raw.open("rb") if raw.exists() else lzma.open(raw.with_suffix(".jsonl.xz"), "rb")


def sha256(stream):
    value = hashlib.sha256()
    with stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_freeze(study):
    path = study / "FREEZE.json"
    freeze = json.loads(path.read_text())
    for name, expected in freeze["files"].items():
        assert sha256((study / name).open("rb")) == expected, name
    for split, expected in freeze["input_cohort_decompressed_sha256"].items():
        assert sha256(case_stream(study / "data" / split)) == expected, split
    return freeze, sha256(path.open("rb"))


def case_seed(namespace, index):
    return int(np.random.SeedSequence([namespace, index]).generate_state(1)[0])


def checked_metrics(row, arm, nodes, edges, start):
    target = {"daughter": (4, 3), "trunk": (6, 7)}[row["task"]]
    data = row["arms"][arm]
    prefix, tail = row["prefix_trace"], data["tail"]
    forced = row["actions"][data["chosen"]]["prefix"]
    assert data["continuation"] == forced
    if data["chosen"] == BASE:
        assert tail == row["arms"][BASE]["tail"]
    if len(prefix) != row["depth"]:
        assert not forced
    node, tick, progress, stage = start, 0, 0, 0
    used = np.zeros(3)
    bad = set()
    for i, step in enumerate(prefix + tail):
        at, goal, packet, claim, kind, slot, owns = (int(step[j]) for j in (0, 1, 2, 4, 5, 10, 11))
        assert at == node and slot == i and progress < 2
        if i == len(prefix):
            stage = progress
            assert row["public"]["progress"] == progress and row["public"]["at"] == node
        relative = i - len(prefix)
        expected_owns = 0 <= relative < len(forced) and (relative == 0 or progress == stage)
        assert bool(owns) == expected_owns
        if owns:
            assert goal == forced[relative]
        edge = edges[nodes[at]["id"], nodes[goal]["id"]]
        tick += int(round(edge["duration_s"] / 1.5))
        cost = np.array([edge[k] for k in ("travel_mm", "turn_deg", "rail_mm")])
        assert int(step[9]) == tick and np.allclose(cost, step[6:9], atol=1e-10, rtol=0)
        assert bool(claim) == (packet > 0 and step[3] >= .87 and nodes[goal]["site"] == target[progress])
        assert kind in ((0, 1, 2) if packet > 0 else (-1,))
        if claim and kind != 0:
            bad.add({1: "other_lumen", 2: "persistent_or_local_false_structure"}[kind])
        used += cost
        node, progress = goal, progress + claim
    assert np.all(used <= [65 + 1e-8, 90 + 1e-8, 250 + 1e-8]) and tick <= 6
    expected = [int(progress == 2 and not bad), int(bool(bad)), used[0], used[1], 1.5 * tick, used[2], len(prefix + tail)]
    np.testing.assert_allclose(data["measured"][:7], expected, atol=1e-10, rtol=0)
    assert data["measured"][7] >= expected[4] + data["prediction_cpu_s"] - 1e-12
    return bad


def best_score(scores):
    choice = BASE
    for key, score in scores.items():
        if score > scores[choice] + 1e-12:
            choice = key
    return choice


def check_forecast(row, feature_vector, fit, names, coefficient, intercept):
    public, actions = row["public"], row["actions"]
    group = f"{row['task']}/{row['depth']}"
    forecasts = {BASE: np.zeros(7)}
    for key in sorted(actions):
        if key != BASE:
            fields = feature_vector(row["task"], public, key, actions[key])
            j = [names[name] for name in fields if name in names]
            x = [value for name, value in fields.items() if name in names]
            forecasts[key] = np.asarray(x) @ coefficient[j] + intercept
    assert set(forecasts) == set(row["forecasts"])
    for key, forecast in forecasts.items():
        np.testing.assert_allclose(forecast, row["forecasts"][key], atol=1e-10, rtol=0)
    price = np.array(fit["prices"][group]["normalized_price"])
    fixed = fit["fixed"][group]["choice"]
    expected = {
        BASE: BASE,
        "B_cohort": best_score({k: q[0] - price @ (q[2:5] / MARGINS) for k, q in forecasts.items()}),
        "B_local": best_score({k: q[0] for k, q in forecasts.items() if np.all(np.abs(q[2:5]) <= MARGINS)}),
        "B_hard_cap": best_score({k: q[0] for k, q in forecasts.items()}),
        "fixed": fixed if fixed in actions else BASE,
    }
    assert {arm: value["chosen"] for arm, value in row["arms"].items()} == expected


def compare(delta, report, fields):
    n = len(delta)
    rng = np.random.default_rng(57216)
    boot = np.concatenate([delta[rng.integers(n, size=(min(500, 20000 - i), n))].mean(axis=1)
                           for i in range(0, 20000, 500)])
    mean = delta.mean(axis=0)
    np.testing.assert_allclose(mean, [report["delta"][k] for k in fields], atol=1e-10, rtol=0)
    ci = np.quantile(boot, [.025, .975], axis=0).T
    np.testing.assert_allclose(ci, [report["delta_ci95_case_clustered"][k] for k in fields], atol=1e-10, rtol=0)
    cost_ci = np.quantile(boot[:, 2:5], [.05 / 6, 1 - .05 / 6], axis=0).T
    np.testing.assert_allclose(cost_ci, [report["cost_bonferroni_ci95"][k] for k in fields[2:5]], atol=1e-10, rtol=0)
    win, loss = (int(np.count_nonzero(delta[:, 0] == k)) for k in (1, -1))
    harm, benefit = (int(np.count_nonzero(delta[:, 1] == k)) for k in (1, -1))
    assert report["success_wins_losses_ties"] == [win, loss, n - win - loss]
    assert report["wrong_harm_benefit_ties_cases"] == [harm, benefit, n - harm - benefit]
    upper = beta.ppf(.975, harm + 1, n - harm) if harm < n else 1.
    pvalue = binomtest(win, win + loss, .5, alternative="greater").pvalue if win + loss else 1.
    assert np.isclose(upper, report["wrong_harm_975_upper"], atol=1e-12, rtol=0)
    assert np.isclose(pvalue, report["success_sign_p_greater"], atol=1e-12, rtol=0)
    equivalent = bool(np.all(cost_ci[:, 0] >= -MARGINS) and np.all(cost_ci[:, 1] <= MARGINS))
    noninferior = bool(np.all(cost_ci[:, 1] <= MARGINS))
    assert equivalent == report["all_costs_equivalent"]
    assert noninferior == report["all_costs_noninferior_with_margin"]
    assert bool(np.all(cost_ci[:, 1] <= 0.)) == report["all_costs_nonincrease"]
    assert (mean[0] >= .05 and ci[0, 0] > 0 and upper <= .02 and equivalent) == report["strong_gate_single_repeat"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path("work/continuation_b"))
    p.add_argument("--motion", type=Path, default=Path("work/full_b/motion/BANK_MAIN.json"))
    args = p.parse_args()
    freeze, freeze_hash = verify_freeze(args.study)
    assert sha256(args.motion.open("rb")) == freeze["shared_bank_sha256"]
    bank = json.loads(args.motion.read_text())
    nodes = sorted(bank["nodes"], key=lambda n: n["id"])
    node_indices = {n["id"]: i for i, n in enumerate(nodes)}
    edges = {(e["start"], e["goal"]): e for e in bank["edges"]}
    sys.path.insert(0, str(args.study))
    from predictor import features

    fit = json.loads((args.study / "model/predictor.json").read_text())
    names = {key: i for i, key in enumerate(fit["names"])}
    coefficient, intercept = np.asarray(fit["coefficients"]), np.asarray(fit["intercept"])
    folder = args.study / "data/confirmation"
    settings = json.loads((folder / "settings.json").read_text())
    for key, value in freeze["confirmation"].items():
        assert settings[key] == value
    assert settings["namespace"] == freeze["confirmation_namespace"]
    report = json.loads((folder / "paired.json").read_text())
    result, wrong, seen, tapes = defaultdict(lambda: defaultdict(list)), defaultdict(Counter), defaultdict(set), {}
    arm_paths = edge_count = 0
    with case_stream(folder) as stream:
        for line in stream:
            row = json.loads(line)
            task, index = row["task"], row["index"]
            group = f"{task}/{row['depth']}"
            assert index not in seen[group]
            seen[group].add(index)
            seed = case_seed(settings["namespace"], index)
            assert seed == row["case_seed"]
            assert tapes.setdefault(index, row["tape_sha256"]) == row["tape_sha256"]
            assert row["public"]["history"] == [[int(step[1]), int(step[2])] for step in row["prefix_trace"]]
            check_forecast(row, features, fit, names, coefficient, intercept)
            for arm in settings["arms"]:
                bad = checked_metrics(row, arm, nodes, edges, node_indices[bank["task_starts"][task]])
                wrong[group, arm].update(bad)
                result[group][arm].append(row["arms"][arm]["measured"])
                arm_paths += 1
                edge_count += len(row["prefix_trace"]) + len(row["arms"][arm]["tail"])
    assert len(set(tapes.values())) == settings["n"]
    for group, data in result.items():
        assert seen[group] == set(range(settings["n"])) and report[group]["n_cases"] == settings["n"]
        measured = {arm: np.asarray(values) for arm, values in data.items()}
        rr = report[group]
        for arm, values in measured.items():
            assert rr["endpoint_counts"][arm] == dict(zip(("correct", "wrong"), values[:, :2].sum(axis=0)))
            assert rr["mechanism"][arm]["wrong_cases_by_return_kind"] == dict(wrong[group, arm])
            if arm != BASE:
                compare(values - measured[BASE], rr["vs_mpc"][arm], settings["metrics"])
        compare(measured["B_cohort"] - measured["fixed"], rr["cohort_vs_fixed"], settings["metrics"])
        print(group, rr["endpoint_counts"])
    print(json.dumps({"freeze_sha256": freeze_hash, "independent_cases": len(tapes), "public_states": sum(map(len, seen.values())),
                      "all_arm_paths_verified": arm_paths, "logged_edges_verified": edge_count,
                      "paired_contrasts_recomputed": len(result) * 5}, indent=2))


if __name__ == "__main__":
    main()

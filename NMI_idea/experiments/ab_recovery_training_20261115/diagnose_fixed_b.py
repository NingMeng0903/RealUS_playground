"""Audit completion ranking and action headroom in the earlier fixed-angle benchmark."""

import argparse
from collections import Counter
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import run


METRICS = ("success", "wrong", "fresh_local", "tcp_mm", "turn_deg", "seconds", "frames")
SEARCH = tuple(range(91000, 91008))
VERIFY = tuple(range(92000, 92008))
WORLD = None
CONTROLLER = None


def read_rows(path):
    with gzip.open(path, "rt") as stream:
        return list(map(json.loads, stream))


def replay_checkpoint(row, split):
    scene = run.case(WORLD, split, row["seed"] - run.SPLIT_BASE[split], "fixed")
    state, prefix = run.context(CONTROLLER, scene)
    assert state.frame == row["checkpoint"]["frame"]
    np.testing.assert_allclose(state.belief, row["checkpoint"]["belief"], atol=1e-12, rtol=0.)
    return scene, state, prefix, CONTROLLER.options(state, False)


def outcome_vector(outcome):
    return [float(outcome.get(field, 0.)) for field in METRICS]


def score_diagnostic(row):
    best = None
    new_outcome = row["mpc"]
    if row["trigger"]:
        probabilities = row["B_predictions"]
        best = max(probabilities, key=lambda a: (probabilities[a][0], a == row["actions"]["mpc"]))
        stored = next((row[arm] for arm in ("mpc", "learned", "fixed") if row["actions"][arm] == best), None)
        if stored is not None:
            new_outcome = stored
        else:
            scene, state, prefix, menu = replay_checkpoint(row, "test")
            new_outcome, _ = run.run_action(CONTROLLER, scene, state, prefix, menu[best], alias=False)
    return {"seed": row["seed"], "original_action": row["actions"].get("learned"),
            "completion_only_action": best,
            "verification": {"mpc": [outcome_vector(row["mpc"])],
                             "learned_original": [outcome_vector(row["learned"])],
                             "learned_completion_only": [outcome_vector(new_outcome)]}}


def headroom_diagnostic(row):
    scene, state, prefix, menu = replay_checkpoint(row, "dev")
    base = row["baseline_action"]
    assert CONTROLLER.mpc(state, menu, False).name == base

    def evaluate(action, tapes):
        results = []
        for tape in tapes:
            future = WORLD.resample_tail(scene, state.frame, tape)
            outcome, _ = run.run_action(CONTROLLER, future, state, prefix, menu[action], alias=False)
            results.append(outcome_vector(outcome))
        return results

    forecasts = {a: np.asarray(evaluate(a, SEARCH)).mean(axis=0) for a in menu}
    incumbent = forecasts[base]
    limits = {METRICS.index(key): margin for key, margin in run.MARGINS.items()}
    admissible = [a for a, m in forecasts.items() if m[1] <= incumbent[1] + 1e-9
                  and all(m[j] <= incumbent[j] + margin + 1e-9 for j, margin in limits.items())]

    def rank(action):
        success, wrong, local, distance, *_ = forecasts[action]
        return (success, -wrong, local, -distance, action == base)

    choices = {"mpc": base, "one_tape": row["replicate_witness"]["first_tape_best_action"],
               "eight_tapes": max(menu, key=rank), "eight_tapes_resource_limit": max(admissible, key=rank)}
    measured = {action: evaluate(action, VERIFY) for action in set(choices.values())}
    return {"seed": row["seed"], "actions": choices, "search_means": {a: m.tolist() for a, m in forecasts.items()},
            "verification": {arm: measured[a] for arm, a in choices.items()}}


def compare(rows, reference, seed):
    subset = np.random.default_rng(seed).integers(len(rows), size=(10_000, len(rows)))
    base = np.asarray([r["verification"][reference] for r in rows]).mean(axis=1)
    report = {}
    for arm in rows[0]["verification"]:
        values = np.asarray([r["verification"][arm] for r in rows])
        difference = values.mean(axis=1) - base
        intervals = np.quantile(difference[subset].mean(axis=1), [.025, .975], axis=0)
        report[arm] = {"completed": int(values[..., 0].sum()), "trials": int(np.prod(values.shape[:2])),
                       "mean": dict(zip(METRICS, values.mean(axis=(0, 1)).tolist())),
                       "delta": dict(zip(METRICS, difference.mean(axis=0).tolist())),
                       "paired_ci95": dict(zip(METRICS, intervals.T.tolist()))}
    return report


def summarize_fixed(rows):
    changed = [row for row in rows if row["trigger"] and row["actions"]["mpc"] != row["actions"]["learned"]]
    losses = [r["learned"] for r in changed if r["mpc"]["success"] and not r["learned"]["success"]]
    difference = [r["B_predictions"][r["actions"]["learned"]][0] - r["B_predictions"][r["actions"]["mpc"]][0]
                  for r in changed]
    return {"changed": len(changed), "b_only_wins": sum(r["learned"]["success"] and not r["mpc"]["success"] for r in changed),
            "b_only_losses": len(losses), "b_predicted_higher_completion": sum(v > 1e-10 for v in difference),
            "changed_predicted": {arm: float(np.mean([r["B_predictions"][r["actions"][arm]][0] for r in changed]))
                                  for arm in ("mpc", "learned")},
            "changed_observed": {arm: float(np.mean([r[arm]["success"] for r in changed])) for arm in ("mpc", "learned")},
            "loss_first_failure": dict(Counter("wrong" if r["wrong"] else "did_not_arrive" if not r["arrived"]
                                               else "missing_coverage" for r in losses))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    global WORLD, CONTROLLER
    WORLD = run.World()
    with np.load(args.archive / "prodmp_weights.npz") as weights:
        CONTROLLER = run.Controller(WORLD, run.load_sections(), dict(weights))
    test = read_rows(args.archive / "test_fixed.jsonl.gz")
    dev = [r for r in read_rows(args.archive / "dev_branches.jsonl.gz") if r["trigger"]]
    with get_context("fork").Pool(args.jobs) as pool:
        scores = pool.map(score_diagnostic, test)
        headroom = pool.map(headroom_diagnostic, dev)
    report = {"fields": METRICS, "headroom_search_tapes": SEARCH, "headroom_verification_tapes": VERIFY,
              "fixed_test": summarize_fixed(test), "score_comparison": compare(scores, "mpc", 662),
              "headroom_comparison": compare(headroom, "mpc", 337)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, rows in (("fixed_score_replay", scores), ("development_headroom_replay", headroom)):
        with gzip.open(args.out / f"{name}.jsonl.gz", "wt", compresslevel=3) as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

"""Audit the first-interval regression and test adding independent startup cases."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np

import benchmark_arbitrary_start as scans
from benchmark_window_encoding import action_choice, choice_from_predictions
from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import fit_targets, predict_deltas
from run_adaptive_recovery import paired_statistics, real_trial
from run_learned_recovery import CONTROLLERS, FAMILIES, fingerprint, read_rows, write_json
from train_startup_b import actual_values, load_controllers
from window_embedding import ViewCalibration


EARLY_TRAIN_SEED = 63_000_000
TEST_SEED = 64_000_000
EARLY_TRAIN_PER_CELL = 6
EARLY_TEST_PER_CELL = 8
OTHER_TEST_PER_CELL = 4
DIAGNOSTIC_REPEATS = 32
DIAGNOSTIC_PURPOSE = 19230
ARMS = ("mpc", "frozen_b", "retrained_b", "augmented_b")
MODELS = {}


def is_first(row):
    return row["kind"] == "cold" and scans.station_bin(row["start_s"]) == 0


def early_cell(job):
    family, regime, downstream, positive_x, start, quota = job
    midpoint = float(np.mean(scans.EDGES[:2]))
    cases = []
    for seed in range(start, start + 500):
        row = scans.make_row(family, regime, "cold", 0, seed)
        nominal_x = CONTROLLERS[family].world.nominal(row["start_s"])[0]
        if (row["start_s"] >= midpoint) != downstream or (row["start_x"] >= nominal_x) != positive_x:
            continue
        cases.append(row)
        if len(cases) == quota:
            return {"seed": start, "cases": cases}
    raise ValueError("insufficient first-interval starts in a geometry and disturbance cell")


def prepare(out, jobs):
    groups = list(product(FAMILIES, scans.REGIMES, (False, True), (False, True)))
    splits = {}
    for split, seed, quota in (("train", EARLY_TRAIN_SEED, EARLY_TRAIN_PER_CELL),
                               ("test", TEST_SEED, EARLY_TEST_PER_CELL)):
        cohorts = run_parallel(early_cell, [(*g, seed + i * 1000, quota) for i, g in enumerate(groups)],
                               jobs, f"{split} first-interval starts")
        if split == "test":
            other = [(f, r, "cold", j) for f, r, j in product(FAMILIES, scans.REGIMES, range(1, scans.BINS))]
            other.extend((f, r, "rolling", n) for f, r, n in product(FAMILIES, scans.REGIMES, (1, 2, 3)))
            cohorts.extend(run_parallel(scans.sample_cell,
                                        [(*g, seed + 100_000 + i * 1000, OTHER_TEST_PER_CELL)
                                         for i, g in enumerate(other)], jobs, "later and in-progress starts"))
        rows = [r for c in cohorts for r in c["cases"]]
        path = out / f"{split}_starts.jsonl.gz"
        save_archive(path, rows)
        splits[split] = {"scenes": len(rows), "sha256": fingerprint(path)}
    write_json(out / "splits.json", splits)


def audit_trace(controller, root, prefix, future, action):
    metrics, record = real_trial(controller, root, prefix, future, [action], detailed=True)
    if metrics["max_handoff_error"] >= 1e-7:
        raise ValueError("first-interval replay became discontinuous")
    bins = {scans.station_bin(p["s"]) for p in record.packets if 0 in p["roles_truth"]}
    ledger = record.packets[-1]["visible_ledger"]
    packets = {p["frame"]: p for p in record.packets}
    unconfirmed = [j for j, entry in enumerate(ledger) if entry is None]
    wrong = [j for j, entry in enumerate(ledger) if entry is not None and packets[entry[0]]["roles_truth"][entry[1]] != 0]
    executed = [d for d in record.decisions if d["frame"] >= root.frame]
    return {
        "completed": metrics["corrected_complete"], "travel_mm": metrics["tcp_mm"],
        "endpoint": bool(record.motion[-1][1] >= scans.END - .28),
        "physically_missing_intervals": sorted(set(range(scans.BINS)) - bins),
        "unconfirmed_intervals": unconfirmed, "wrong_intervals": wrong,
        "first_interval_target_packets": sum(0 in p["roles_truth"] and scans.station_bin(p["s"]) == 0
                                              for p in record.packets),
        "first_segment_goals": executed[0]["executed_goal"],
        "after_scout_actions": [d["action"] for d in executed],
    }


def audit_case(row):
    controller, scene, root, prefix, menu, inputs, _, _ = scans.decision_row(row, row["task"])
    if list(menu) != row["names"] or inputs["incumbent"] != row["selection"]["mpc"]:
        raise ValueError("first-interval archive menu changed")
    arms = ("frozen_b", "retrained_b")
    trials = []
    for tape, future in enumerate(scans.paired_futures(scene, row, scans.TEST_REPEATS, scans.TEST_PURPOSE)):
        cache = {n: audit_trace(controller, root, prefix, future, menu[n])
                 for n in dict.fromkeys(row["selection"][a] for a in arms)}
        trial = {a: cache[row["selection"][a]] for a in arms}
        for a in arms:
            prev = row["verification"][tape][a]
            check = trial[a]
            if (check["completed"] != prev["corrected_complete"] or check["travel_mm"] != prev["tcp_mm"]
                    or (not check["physically_missing_intervals"]) != prev["all_task_intervals_imaged"]
                    or bool(check["wrong_intervals"]) != bool(prev["final_wrong"])):
                raise ValueError(f"an archived first-interval result changed for {row['seed']} tape {tape}")
        trials.append(trial)
    return {k: row[k] for k in ("seed", "family", "regime", "start_s", "start_x", "selection")} | {"verification": trials}


def repeat_discordant_case(row):
    controller, scene, root, prefix, menu, _, _, _ = scans.decision_row(row, row["task"])
    arms = ("frozen_b", "retrained_b")
    tapes = [
        {a: audit_trace(controller, root, prefix, future, menu[row["selection"][a]]) for a in arms}
        for future in scans.paired_futures(scene, row, DIAGNOSTIC_REPEATS, DIAGNOSTIC_PURPOSE)
    ]
    return {"seed": row["seed"], "selection": row["selection"], "verification": tapes}


def audit(out, source, jobs):
    archive = source / "test.jsonl.gz"
    if fingerprint(archive) != json.loads((source / "summary.json").read_text())["test_sha256"]:
        raise ValueError("arbitrary-start results changed")
    rows = read_rows(archive)
    selected = [r for r in rows if r["task"] == scans.TASKS[0] and is_first(r)]
    details = run_parallel(audit_case, selected, jobs, "first-interval trajectory and evidence audit")
    save_archive(out / "first_interval_audit.jsonl.gz", details)
    # Use two previously discordant states as diagnostics on independent tapes.
    critical = sorted(selected, key=lambda r: sum(t["retrained_b"]["corrected_complete"]
                                                  - t["frozen_b"]["corrected_complete"] for t in r["verification"]))[:2]
    repeated = run_parallel(repeat_discordant_case, critical, jobs, "earlier discordant cases on new disturbances")
    save_archive(out / "discordant_repeats.jsonl.gz", repeated)
    retreats = {}
    for task in scans.TASKS[:2]:
        later = [r for r in rows if r["task"] == task and r["handoff_s"] >= 22.]
        retreats[task] = {arm: float(np.mean([r["handoff_s"] - t[arm]["minimum_station_mm"]
                                            for r in later for t in r["verification"]]))
                          for arm in ("frozen_b", "retrained_b")}
    lost_or_gained = {}
    for arm, other in (("retrained_b", "frozen_b"), ("frozen_b", "retrained_b")):
        failures = [t[arm] for r in details for t in r["verification"] if t[other]["completed"] and not t[arm]["completed"]]
        lost_or_gained[arm] = {"failed_while_other_completed": len(failures),
                              "missing_physical_image": sum(bool(f["physically_missing_intervals"]) for f in failures),
                              "all_physically_imaged_unconfirmed": sum(not f["physically_missing_intervals"]
                                                                        and bool(f["unconfirmed_intervals"]) for f in failures),
                              "any_retained_wrong_identity": sum(bool(f["wrong_intervals"]) for f in failures)}
    write_json(out / "first_interval_audit.json", {
        "source_sha256": fingerprint(archive), "scenes": len(selected), "paired": paired_statistics(
            selected, ("frozen_b", "retrained_b"), scans.FIELDS, lambda r: r["verification"]),
        "discordant_failures": lost_or_gained, "late_mean_backward_displacement_mm": retreats,
        "discordant_repeats": {r["seed"]: {a: int(sum(t[a]["completed"] for t in r["verification"]))
                                          for a in ("frozen_b", "retrained_b")} for r in repeated},
    })


def train(out, source, jobs):
    archive = source / "training.jsonl.gz"
    frozen = json.loads((source / "frozen.json").read_text())
    if fingerprint(archive) != frozen["training_sha256"]:
        raise ValueError("original arbitrary-start training data changed")
    original = read_rows(archive)
    rows = run_parallel(scans.collect_case, scans.checked_split(out, "train"), jobs, "additional first-interval labels")
    added = [t for r in rows for t in r["tasks"]]
    if {r["seed"] for r in original} & {r["seed"] for r in added}:
        raise ValueError("the additional training scenes overlap")
    save_archive(out / "additional_training.jsonl.gz", added)
    fit = original + added
    x = np.concatenate([scans.retrained_inputs(r) for r in fit])
    y = np.concatenate([actual_values(r) for r in fit])
    joblib.dump(fit_targets(x, y), out / "models.joblib", compress=3)
    write_json(out / "frozen.json", {
        "original_training_sha256": fingerprint(archive), "additional_tasks": len(added),
        "additional_candidates": sum(len(r["names"]) for r in added), "pooled_tasks": len(fit),
        "pooled_first_interval_tasks": sum(is_first(r) for r in fit),
        "frozen_sha256": {f: fingerprint(out / f) for f in ("models.joblib", "additional_training.jsonl.gz",
                                                           "test_starts.jsonl.gz")},
    })


def test_case(row):
    tasks, previous = [], {}
    for task in scans.task_names(row):
        controller, scene, root, prefix, menu, inputs, _, _ = scans.decision_row(row, task)
        if root.first_required_bin in previous:
            tasks.append(previous[root.first_required_bin] | {"task": task})
            continue
        choices = {"mpc": inputs["incumbent"], "frozen_b": action_choice(inputs, "calibrated", MODELS["frozen_b"])[0]}
        for arm in ("retrained_b", "augmented_b"):
            choices[arm] = choice_from_predictions(inputs, predict_deltas(MODELS[arm], scans.retrained_inputs(inputs)))[0]
        verification = []
        for future in scans.paired_futures(scene, row, scans.TEST_REPEATS, scans.TEST_PURPOSE):
            cache = {n: scans.measured_trial(controller, root, prefix, future, menu[n], row["drop_frames"])
                     for n in dict.fromkeys(choices.values())}
            verification.append({a: cache[n] for a, n in choices.items()})
        result = {k: v for k, v in inputs.items() if k not in ("features", "margins", "calibrated_detection")}
        result |= {"selection": choices, "verification": verification}
        previous[root.first_required_bin] = result
        tasks.append(result)
    return {"seed": row["seed"], "tasks": tasks}


def summarize(rows):
    first = [r for r in rows if r["task"] == scans.TASKS[0] and is_first(r)]
    groups = {"first_interval": first} | {
        name: [r for r in rows if r["task"] == name and not is_first(r)] for name in scans.TASKS}
    groups.update({f"first_{regime}": [r for r in first if r["regime"] == regime] for regime in scans.REGIMES})
    info = {}
    for name, cohort in groups.items():
        tapes = [t for r in cohort for t in r["verification"]]
        info[name] = {
            "runs_per_arm": len(tapes), "counts": {a: {k: int(sum(t[a][k] for t in tapes))
                                                      for k in scans.VISIBLE_TARGETS[:3]} for a in ARMS},
            "vs_retrained": paired_statistics(cohort, ("retrained_b", "augmented_b"), scans.FIELDS,
                                               lambda r: r["verification"]),
            "vs_frozen": paired_statistics(cohort, ("frozen_b", "augmented_b"), scans.FIELDS,
                                            lambda r: r["verification"]),
            "first_choices": {a: dict(Counter(r["selection"][a] for r in cohort)) for a in ARMS},
        }
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "prepare", "train", "test"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("results/arbitrary_start_v1"))
    parser.add_argument("--window", type=Path, default=Path("results/window_encoding_v1"))
    parser.add_argument("--library", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    load_controllers(args.library)
    frozen = json.loads((args.source / "frozen.json").read_text())
    upstream = frozen["source_sha256"]
    for directory, files in ((args.library, ("library.json", "prodmp_fit.npy")),
                             (args.window, ("labels.json", "models.joblib", "calibration.json"))):
        if any(fingerprint(directory / f) != upstream[f] for f in files):
            raise ValueError("a shared arbitrary-start input changed")
    scans.CALIBRATION = ViewCalibration(**json.loads((args.window / "calibration.json").read_text())["parameters"])
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        audit(args.out, args.source, args.jobs)
        return
    if args.stage == "prepare":
        prepare(args.out, args.jobs)
        return
    if args.stage == "train":
        train(args.out, args.source, args.jobs)
        return
    if fingerprint(args.source / "models.joblib") != frozen["models_sha256"]:
        raise ValueError("retrained B reference weights changed")
    augmented = json.loads((args.out / "frozen.json").read_text())
    if any(fingerprint(args.out / f) != value for f, value in augmented["frozen_sha256"].items()):
        raise ValueError("frozen augmented B input changed")
    MODELS.update(frozen_b=joblib.load(args.window / "models.joblib")["calibrated"],
                  retrained_b=joblib.load(args.source / "models.joblib"),
                  augmented_b=joblib.load(args.out / "models.joblib"))
    starts = scans.checked_split(args.out, "test")
    rows = run_parallel(test_case, starts, args.jobs, "fresh paired startup-data test")
    tasks = [t for r in rows for t in r["tasks"]]
    path = args.out / "test.jsonl.gz"
    save_archive(path, tasks)
    write_json(args.out / "summary.json", {"test_sha256": fingerprint(path), "scenes": len(rows), "groups": summarize(tasks)})


if __name__ == "__main__":
    main()

"""Evaluate the frozen recovery selector without relative forecast gates."""

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np

from diagnose_b_selection import run_parallel, save_archive
from learned_recovery import OBJECTIVE, VISIBLE_TARGETS, choose_from_deltas, predict_deltas
from run_adaptive_recovery import paired_statistics
from run_learned_recovery import FAMILIES, candidate_set, fingerprint, read_rows, write_json
from train_startup_b import (MODELS, SPLITS, TEST_PURPOSE, TEST_TAPES, choose, futures, load_controllers,
                             menu_metadata, start_case)
from verify_startup_ood import OUTCOMES, execute_trial, screen


ARMS = ("mpc", "learned_b", "no_gates", "completion_first", "fixed")
FRESH_START = 48_610_000
FRESH_PURPOSE = 16132
FRESH_PER_CELL = 32
SCREEN_PER_FAMILY = 768
COUNT_FIELDS = ("corrected_complete", "final_wrong", "new_identity_mistake")


def load_study(source):
    load_controllers(source)
    config = json.loads((source / "frozen_models.json").read_text())
    for label, name in (("models", "models.joblib"), ("library", "library.json"), ("labels", "labels.json")):
        if fingerprint(source / name) != config[f"{label}_sha256"]:
            raise ValueError(f"frozen {label} digest mismatch")
    MODELS[:] = joblib.load(source / "models.joblib")
    check = json.loads((source / "fixed_check_summary.json").read_text())
    if fingerprint(source / "dev.jsonl.gz") != check["selection_archive_sha256"]:
        raise ValueError("the development records changed after fixing the reference action")
    return check


def recorded_menu(row):
    metadata = {k: row[k] for k in ("names", "renderers")}
    metadata["requests"] = [tuple(r) if r is not None else None for r in row["requests"]]
    return metadata


def select_all(names, incumbent, prediction, fixed):
    learned, _, _ = choose({"names": names, "incumbent": incumbent}, prediction)
    no_gates, clean, _ = choose_from_deltas(names, incumbent, prediction, targets=VISIBLE_TARGETS, caps={})
    utilities = clean @ OBJECTIVE
    first = max(range(len(names)), key=lambda i: (clean[i, 0], utilities[i], names[i] == incumbent))
    choices = {"mpc": incumbent, "learned_b": learned, "no_gates": no_gates, "completion_first": names[first],
               "fixed": fixed if fixed in names else incumbent}
    return choices, clean


def evaluate_case(job):
    row, fixed, purpose, previous = job
    controller, scene, root, prefix = start_case(row)
    menu, incumbent, features = candidate_set(controller, root, prefix)
    selected, prediction = select_all(list(menu), incumbent, predict_deltas(MODELS, features), fixed)
    if previous is not None:
        if menu_metadata(menu) != recorded_menu(row) or any(selected[a] != row["selection"][a]
                                                            for a in ("mpc", "learned_b")):
            raise ValueError("the original candidates or gated policy could not be reproduced")
        np.testing.assert_allclose(prediction, row["predicted_delta"], atol=1e-12, rtol=0.)
        if selected["fixed"] != previous["fixed"]:
            raise ValueError("the original fixed action could not be reproduced")
    verification = []
    executions = 0
    for i, future in enumerate(futures(scene, row["seed"], TEST_TAPES, purpose)):
        cache = {}
        if previous is not None:
            cache.update({selected[a]: previous["verification"][i][a] for a in ("mpc", "learned_b", "fixed")})
        for request in dict.fromkeys(selected.values()):
            if request not in cache:
                cache[request] = execute_trial(controller, root, prefix, future, menu[request])
                executions += 1
        verification.append({arm: cache[request] for arm, request in selected.items()})
    start = {k: row[k] for k in ("seed", "family", "group", "observed_vessels", "blank_packets", "scout_frame")}
    return start | menu_metadata(menu) | {"selection": selected, "predicted_delta": prediction,
                                         "verification": verification, "new_closed_loops": executions}


def ceiling_case(job):
    row, purpose = job
    controller, scene, root, prefix = start_case(row)
    menu = controller.options_shared(root, renderer="fit")
    if menu_metadata(menu) != recorded_menu(row):
        raise ValueError("the ceiling diagnostic has a different admissible menu")
    by_request = {a: [] for a in menu}
    executions = 0
    for i, future in enumerate(futures(scene, row["seed"], TEST_TAPES, purpose)):
        existing = {name: row["verification"][i][arm] for arm, name in row["selection"].items()}
        for name, action in menu.items():
            if name not in existing:
                existing[name] = execute_trial(controller, root, prefix, future, action)
                executions += 1
            by_request[name].append(existing[name])
    fixed = row["selection"]["fixed"]
    completed = {a: int(sum(t["corrected_complete"] for t in tape)) for a, tape in by_request.items()}
    wrong = {a: int(sum(t["final_wrong"] for t in tape)) for a, tape in by_request.items()}
    best = max(menu, key=lambda a: (completed[a], -wrong[a], a == fixed))
    return {k: row[k] for k in ("seed", "family", "group")} | {
        "best_action": best, "fixed_action": fixed, "completed_per_request": completed,
        "final_wrong_per_request": wrong, "new_closed_loops": executions, "verification": by_request,
    }


def fresh_starts(source, jobs):
    candidates = [(family, FRESH_START + 10_000 * i + j, "holdout")
                  for i, family in enumerate(FAMILIES) for j in range(SCREEN_PER_FAMILY)]
    rows = run_parallel(screen, candidates, jobs, "fresh startup screen")
    starts = []
    for family, group in product(FAMILIES, ("zero", "one")):
        roles = [] if group == "zero" else [1]
        group_rows = [r for r in rows if r["family"] == family and r["observed_vessels"] == roles]
        if len(group_rows) < FRESH_PER_CELL:
            raise ValueError(f"not enough fresh {family}/{group} scenes")
        starts.extend(group_rows[:FRESH_PER_CELL])
    used = {r["seed"] for split in SPLITS for r in read_rows(source / f"{split}_starts.jsonl.gz")}
    if used & {r["seed"] for r in starts}:
        raise ValueError("fresh holdout reused a previous scene")
    return starts


def outcome_counts(rows):
    return {arm: {k: int(sum(t[arm][k] for r in rows for t in r["verification"]))
                  for k in COUNT_FIELDS} for arm in ARMS}


def report(source, out, label, rows, purpose, jobs):
    path = out / f"{label}.jsonl.gz"
    save_archive(path, rows)
    hard = [r for r in rows if any(not t["fixed"]["corrected_complete"] for t in r["verification"])]
    ceilings = run_parallel(ceiling_case, [(r, purpose) for r in hard], jobs, f"{label} ceiling") if hard else []
    save_archive(out / f"{label}_ceiling.jsonl.gz", ceilings)
    optimal = ((len(rows) - len(ceilings)) * TEST_TAPES
               + sum(r["completed_per_request"][r["best_action"]] for r in ceilings))

    def compare(reference):
        return paired_statistics(rows, (reference, *(a for a in ARMS if a != reference)), OUTCOMES,
                                 lambda r: r["verification"])

    write_json(out / f"{label}_summary.json", {
        "archive_sha256": fingerprint(path), "models_sha256": fingerprint(source / "models.joblib"),
        "counts": outcome_counts(rows),
        "by_family": {f: outcome_counts([r for r in rows if r["family"] == f]) for f in FAMILIES},
        "by_opening": {g: outcome_counts([r for r in rows if r["group"] == g]) for g in ("zero", "one")},
        "contrasts": {r: compare(r) for r in ("mpc", "learned_b", "fixed")},
        "fixed_choice_cases": {a: sum(r["selection"][a] == r["selection"]["fixed"] for r in rows) for a in ARMS},
        "selection_counts": {a: dict(Counter(r["selection"][a] for r in rows)) for a in ARMS},
        "new_paired_closed_loops": sum(r["new_closed_loops"] for r in rows),
        "ceiling": {"best_per_scout_completed": optimal, "scenes_with_fixed_failure": len(ceilings),
                    "new_closed_loops": sum(r["new_closed_loops"] for r in ceilings)},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("replay", "fresh"))
    parser.add_argument("--source", type=Path, default=Path("results/startup_b_new_library_v1"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=24)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source
    check = load_study(source)
    fixed = check["fixed_by_family"]
    if args.phase == "replay":
        archive = source / "test.jsonl.gz"
        summary = json.loads((source / "test_summary.json").read_text())
        if fingerprint(archive) != check["test_archive_sha256"] or fingerprint(archive) != summary["archive_sha256"]:
            raise ValueError("the prior paired test changed")
        by_seed = {r["seed"]: r for r in read_rows(source / "fixed_check.jsonl.gz")}
        jobs = [(r, fixed[r["family"]], TEST_PURPOSE, by_seed[r["seed"]]) for r in read_rows(archive)]
        purpose = TEST_PURPOSE
    else:
        starts = fresh_starts(source, args.jobs)
        save_archive(args.out / "fresh_starts.jsonl.gz", starts)
        jobs = [(r, fixed[r["family"]], FRESH_PURPOSE, None) for r in starts]
        purpose = FRESH_PURPOSE
    rows = run_parallel(evaluate_case, jobs, args.jobs, f"{args.phase} without relative gates")
    report(source, args.out, args.phase, rows, purpose, args.jobs)


if __name__ == "__main__":
    main()

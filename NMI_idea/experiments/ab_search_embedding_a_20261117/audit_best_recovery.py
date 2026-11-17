"""Audit the best verified finalists from the two equal-budget searches."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from adaptive_teacher import utility
from benchmark_search_efficiency import METHODS
from diagnose_b_selection import run_parallel, save_archive
from run_adaptive_recovery import tape_for
from run_learned_recovery import CONTROLLERS, configure, fingerprint, read_rows, write_json
from verify_recovery_components import bands
from verify_startup_ood import OUTCOMES, execute_trial, scout, startup_controller


AUDIT_TAPE = 15771
REPEATS = 16
ARMS = ("mpc", "shared_first", "random_best", "cem_best")


def best_verified(record):
    choices = []
    for round_record in record["rounds"]:
        for finalist in round_record["finalists"]:
            if not finalist["qualified"]:
                continue
            request = round_record["scored"][finalist["rank"]]["request"]
            score = float(np.mean([utility(trial) for trial in finalist["trials"]]))
            choices.append((score, finalist["after_rollouts"], request))
    score, at, request = max(choices, key=lambda entry: (entry[0], -entry[1]))
    return {"request": request, "after_rollouts": at, "confirmation_utility": score}


def audit_case(row):
    c = CONTROLLERS[row["family"]]
    seed = row["seed"]
    scene = c.world.startup_draw(seed, "holdout")
    root, prefix = scout(c, scene)
    first = row["search"]["random"]["first"]
    assert first["request"] == row["search"]["cem"]["first"]["request"]
    choices = {f"{method}_best": best_verified(row["search"][method]) for method in METHODS}
    requests = {"shared_first": first["request"]} | {name: spec["request"] for name, spec in choices.items()}
    actions = {"mpc": c.choose_mpc(root, expanded=True, open_decision=True)}
    for name, request in requests.items():
        candidate = c.free_candidate(root, "offline_probe", *request)
        if candidate is None:
            raise ValueError("archived motion is no longer admissible")
        actions[name] = candidate
    verification = []
    for rep in range(REPEATS):
        tape = tape_for(scene.latent, seed, rep, AUDIT_TAPE).tape
        future = replace(scene, tape=tape)
        results, by_request = {}, {}
        for name, action in actions.items():
            key = tuple(requests[name]) if name in requests else None
            if key not in by_request:
                by_request[key] = execute_trial(c, root, prefix, future, action)
            results[name] = by_request[key]
        verification.append(results)
    return {k: row[k] for k in ("seed", "family", "group")} | {
        "choices": choices, "first_request": first["request"], "verification": verification,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/rollout_matched_search_v1"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=20)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    configure(Path("results/visible_atlas_v1"), controller_type=startup_controller)
    archive = args.source / "discovery.jsonl.gz"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = run_parallel(audit_case, read_rows(archive), args.jobs, "best finalist audit")
    save_archive(args.out / "audit.jsonl.gz", rows)
    write_json(args.out / "summary.json", {
        "source_sha256": fingerprint(archive),
        "versus_direct_mpc": bands(rows, ARMS, OUTCOMES),
        "versus_random": bands(rows, ("random_best", "cem_best", "shared_first"), OUTCOMES),
        "different_best_requests": sum(r["choices"]["random_best"]["request"] != r["choices"]["cem_best"]["request"]
                                       for r in rows),
        "best_after_shared_first_round": {arm: sum(r["choices"][arm]["after_rollouts"] > 48 for r in rows)
                                          for arm in ("random_best", "cem_best")},
    })


if __name__ == "__main__":
    main()

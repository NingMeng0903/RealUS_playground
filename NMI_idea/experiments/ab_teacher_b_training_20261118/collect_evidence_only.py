"""Generate paired B training data using the evidence teacher alone."""

import argparse
from pathlib import Path

import numpy as np

import experiment as study
from adaptive_teacher import evaluate, posterior_worlds
from diagnose_b_selection import run_parallel, save_archive
from run_learned_recovery import fingerprint, write_json


FIRST_SEED = 74_400_000
ROOTS_PER_CELL = 32


def collect(row):
    controller, _, state, prefix, menu, aliases, incumbent, _ = study.at_handoff(row)
    inputs = study.policy_inputs(controller, state, prefix, menu, incumbent)
    new_actions = sorted(set(aliases.values()) - {incumbent})
    pool = new_actions if len(new_actions) >= study.PROPOSALS else sorted(set(menu) - {incumbent})
    if len(pool) < study.PROPOSALS:
        raise ValueError(f"insufficient feasible actions at seed {row['seed']}")

    forecast = posterior_worlds(state, study.generator.PROBE_SAMPLES, row["seed"] + 18304)
    scores = {name: study.generator.evidence_score(controller, state, menu[name], forecast)
              for name in pool}
    nominated = sorted(pool, key=lambda name: (-round(scores[name]["score"], 12), menu[name].distance, name))
    nominated = nominated[:study.PROPOSALS]

    names = sorted([incumbent, *nominated])
    bank = list(study.label_worlds(state, prefix.packets, row["seed"]))
    outcomes = {}
    for name in names:
        replays, _ = evaluate(controller, state, menu[name], bank, open_blocks=1)
        outcomes[name] = [{field: replay[field] for field in study.VISIBLE_TARGETS} for replay in replays]
    means = np.mean([[[trial[field] for field in study.VISIBLE_TARGETS] for trial in outcomes[name]]
                     for name in names], axis=1)
    return row | {
        "names": names, "incumbent": incumbent, "nominated": nominated,
        "features": inputs["features"][[inputs["names"].index(name) for name in names]],
        "targets_vs_mpc": means - means[names.index(incumbent)],
        "outcomes": outcomes, "scores": scores, "aliases": aliases,
        "sampled_modes": [scene.latent for scene in bank],
    }


def audit(rows):
    states_with_gain = 0
    clearly_failed = 0
    high_wrong = 0
    total_replays = 0
    scenes = set()
    for row in rows:
        scenes.add(row["seed"])
        names = row["names"]
        if len(set(names)) != study.PROPOSALS + 1 or row["features"].shape != (len(names), 457):
            raise ValueError(f"incomplete features or candidates for seed {row['seed']}")
        targets = row["targets_vs_mpc"]
        if not np.isfinite(targets).all() or not np.isfinite(row["features"]).all():
            raise ValueError(f"non-finite training data for seed {row['seed']}")
        if np.any(targets[names.index(row["incumbent"])] != 0.):
            raise ValueError(f"inconsistent MPC reference for seed {row['seed']}")
        states_with_gain += int(any(t[0] >= 1. / study.LABEL_TAPES - 1e-9 and t[1] <= 1e-9 for t in targets))
        for name, trials in row["outcomes"].items():
            if len(trials) != study.LABEL_TAPES:
                raise ValueError(f"incomplete paired replays for seed {row['seed']}")
            total_replays += len(trials)
            if name != row["incumbent"]:
                clearly_failed += sum(t["corrected_complete"] < .5 for t in trials)
                high_wrong += sum(t["final_wrong"] > .5 for t in trials)
    if len(scenes) != len(rows):
        raise ValueError("duplicate scene seeds")
    return {
        "states": len(rows), "nominated_rows": study.PROPOSALS * len(rows), "mpc_anchor_rows": len(rows),
        "total_paired_replays": total_replays, "nominated_completion_below_half": clearly_failed,
        "nominated_final_wrong_above_half": high_wrong,
        "states_with_at_least_one_eighth_completion_gain_without_added_final_wrong": states_with_gain,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = study.configure()
    cases = [study.scenario(cell, FIRST_SEED + cell_id * 10_000 + index)
             for cell_id, cell in enumerate(study.CELLS) for index in range(ROOTS_PER_CELL)]
    rows = run_parallel(collect, cases, args.workers, "evidence-only paired replays")
    stats = audit(rows)
    archive = args.out / "train_cases.jsonl.gz"
    save_archive(archive, rows)
    write_json(args.out / "manifest.json", {
        "generator": "evidence_teacher",
        "features_per_action": 457,
        "targets_vs_mpc_order": study.VISIBLE_TARGETS,
        "first_seed": FIRST_SEED,
        "roots_per_family_and_condition": ROOTS_PER_CELL,
        "cells": study.CELLS,
        "paired_replays_per_action": study.LABEL_TAPES,
        "source_hashes": sources,
        "experiment_sha256": fingerprint(Path(study.__file__)),
        "collector_sha256": fingerprint(Path(__file__)),
        "archive_sha256": fingerprint(archive),
        "counts": stats,
    })
    print(stats, flush=True)


if __name__ == "__main__":
    main()

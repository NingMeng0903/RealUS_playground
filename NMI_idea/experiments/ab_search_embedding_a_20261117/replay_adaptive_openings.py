"""Restrict the public teacher's first action to the legacy option types."""

import argparse
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from run_adaptive_recovery import (OUTCOMES, WORKERS, init_worker, jsonable, paired_statistics,
                                   real_trial, shared_prefix, tape_for)


ARMS = ("mpc_shared_menu", "public_original_first_menu", "public_first_only")


def replay(row):
    family, seed = row["family"], row["seed"]
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    np.testing.assert_allclose(root.belief, row["public_root"]["belief"], atol=1e-12, rtol=0.)
    menu = controller.options_shared(root, expanded=False)
    original = controller.mpc(root, menu, False, active=True, informative=True)
    legal = [score for score in row["first_decision"]["candidates"] if score["action"] in menu]
    picked = max(legal, key=lambda r: (r["utility"], r["action"] == original.name))["action"]
    outcomes = []
    for trial in row["verification"]:
        future = tape_for(scene.latent, seed, trial["replica"], 1101)
        m, _ = real_trial(controller, root, prefix, future, [menu[picked]])
        outcomes.append({arm: m if arm == "public_original_first_menu" else trial["outcomes"][arm] for arm in ARMS})
    return {"family": family, "seed": seed, "picked": picked, "verification": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.archive, "rt") as source:
        rows = [row for row in map(json.loads, source) if row["family"] in ("connection", "parallel")]
    init_worker(None)
    with get_context("fork").Pool(8) as pool:
        outcomes = pool.map(replay, rows)
    args.out.mkdir(parents=True, exist_ok=True)
    cohorts = {family: [r for r in outcomes if r["family"] == family] for family in ("connection", "parallel")}
    cohorts["pooled"] = outcomes
    summary = {key: paired_statistics(group, ARMS, OUTCOMES, lambda c: c["verification"])
               for key, group in cohorts.items()}
    (args.out / "summary.json").write_text(json.dumps(summary, default=jsonable, indent=2) + "\n")
    with gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream:
        for row in outcomes:
            stream.write(json.dumps(row, default=jsonable, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

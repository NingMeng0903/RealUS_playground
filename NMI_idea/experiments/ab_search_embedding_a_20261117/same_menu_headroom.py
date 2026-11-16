"""Measure privileged headroom inside the shared action menu on development scenes."""

import argparse
import gzip
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from adaptive_teacher import utility
from run_adaptive_recovery import (OUTCOMES, WORKERS, init_worker, jsonable, paired_statistics,
                                   real_trial, shared_prefix, tape_for)


def screen(job):
    family, seed = job
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    menu = controller.options_shared(root)
    incumbent = controller.mpc(root, menu, False, active=True, informative=True)
    fields = ("corrected_complete", "final_wrong", "new_centered_mistake", "tcp_mm", "turn_degs", "seconds")

    def trials(action, count, purpose):
        return [real_trial(controller, root, prefix, tape_for(scene.latent, seed, i, purpose), [action])[0]
                for i in range(count)]

    forecast = {}
    for name, action in menu.items():
        rows = trials(action, 5, 2561)
        forecast[name] = {key: float(np.mean([row[key] for row in rows])) for key in fields}
    reference = forecast[incumbent.name]
    allowed = [name for name, row in forecast.items() if row["tcp_mm"] <= reference["tcp_mm"] + 2.
               and row["turn_degs"] <= reference["turn_degs"] + 2.
               and row["seconds"] <= reference["seconds"] + .10000001
               and row["new_centered_mistake"] <= reference["new_centered_mistake"]
               and row["final_wrong"] <= reference["final_wrong"]]
    rank = lambda name: (utility(forecast[name]), name == incumbent.name)
    picked = {"mpc": incumbent.name, "privileged_shared": max(forecast, key=rank),
              "privileged_resource_screen": max(allowed, key=rank)}
    measured = {name: trials(menu[name], 6, 927) for name in set(picked.values())}
    return {"family": family, "seed": seed, "picked": picked, "forecast": forecast,
            "verification": [{arm: measured[name][i] for arm, name in picked.items()} for i in range(6)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery_archive", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    with gzip.open(args.discovery_archive, "rt") as file:
        keys = [(row["family"], row["seed"]) for row in map(json.loads, file)]
    init_worker(None)
    with get_context("fork").Pool(args.jobs) as pool:
        rows = list(pool.imap(screen, keys))
    args.out.mkdir(parents=True, exist_ok=True)
    arms = ("mpc", "privileged_shared", "privileged_resource_screen")
    report = {"all": paired_statistics(rows, arms, OUTCOMES, lambda row: row["verification"]),
              "natural": paired_statistics([r for r in rows if r["family"] in ("connection", "parallel")],
                                           arms, OUTCOMES, lambda row: row["verification"])}
    (args.out / "summary.json").write_text(json.dumps(report, default=jsonable, indent=2) + "\n")
    with gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as file:
        for row in rows:
            file.write(json.dumps(row, default=jsonable, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()

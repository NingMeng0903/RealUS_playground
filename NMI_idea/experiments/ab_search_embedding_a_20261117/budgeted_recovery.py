"""Compare public first-action ranking with and without forecast resource constraints."""

import argparse
import gzip
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

from adaptive_teacher import public_scores
from run_adaptive_recovery import (MARGINS, OUTCOMES, WORKERS, init_worker, jsonable, paired_statistics,
                                   public_root, real_trial, root_report, shared_prefix, tape_for, trace)


FAMILIES = ("connection", "parallel")
ARMS = ("mpc_shared_menu", "public_unconstrained", "public_budgeted")
CAPS = {"tcp_mm": 2., "turn_degs": 2., "seconds": .10, "final_wrong": .02, "new_centered_mistake": .02}


def run_case(job):
    family, seed, samples, replicas = job
    controller = WORKERS[family]
    scene = controller.world.draw(seed)
    root, prefix = shared_prefix(controller, scene)
    start = perf_counter()
    menu, incumbent, scores = public_scores(controller, root, samples=samples, remaining_open=1)
    reference = next(row["expected"] for row in scores if row["action"] == incumbent)
    allowed = [row for row in scores if all(row["expected"][cost] <= reference[cost] + cap + 1e-8
                                           for cost, cap in CAPS.items())]

    def best(rows):
        return max(rows, key=lambda row: (row["utility"], row["action"] == incumbent))["action"]

    selected = dict(zip(ARMS, (incumbent, best(scores), best(allowed))))
    calculation_time = perf_counter() - start
    trials = []
    for replica in range(replicas):
        future = tape_for(scene.latent, seed, replica, 1117)
        outcomes, traces = {}, {}
        # An identical selected action yields precisely an identical execution.
        executed = {}
        for arm, name in selected.items():
            if name not in executed:
                outcome, record = real_trial(controller, root, prefix, future, [menu[name]], detailed=replica == 0)
                executed[name] = outcome, trace(record, after_frame=root.frame) if replica == 0 else None
            outcomes[arm], log = executed[name]
            if replica == 0:
                traces[arm] = log
        trials.append({"replica": replica, "outcomes": outcomes, "traces": traces})
    return {"family": family, "seed": seed, "checkpoint": root_report(root, prefix),
            "prefix": trace(prefix), "public_root": public_root(root), "hidden_index_evaluator": scene.latent,
            "selected": selected, "scores": scores, "allowed": [row["action"] for row in allowed],
            "cpu_wall_s": calculation_time, "verification": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cases", default=28, type=int)
    parser.add_argument("--samples", default=12, type=int)
    parser.add_argument("--replicas", default=3, type=int)
    parser.add_argument("--jobs", default=12, type=int)
    args = parser.parse_args()
    if min(args.cases, args.samples, args.replicas, args.jobs) <= 0 or args.cases > 90_000:
        parser.error("positive counts required, with at most 90,000 distinct seeds per family")
    init_worker(None)
    aligned = [(family, 8_210_000 + i * 100_000 + j, args.samples, args.replicas)
               for i, family in enumerate(FAMILIES) for j in range(args.cases)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    started = perf_counter()
    with (get_context("fork").Pool(args.jobs) as pool,
          gzip.open(args.out / "cases.jsonl.gz", "wt", compresslevel=3) as stream):
        for row in pool.imap(run_case, aligned):
            stream.write(json.dumps(row, default=jsonable, allow_nan=False, separators=(",", ":")) + "\n")
            rows.append(row)
            if len(rows) % 8 == 0:
                print(f"budgeted {len(rows)}/{len(aligned)}", flush=True)

    groups = {family: [r for r in rows if r["family"] == family] for family in FAMILIES}
    groups["pooled"] = rows
    report = {}
    for group, cases in groups.items():
        def extract(row):
            return [r["outcomes"] for r in row["verification"]]
        report[group] = paired_statistics(cases, ARMS, OUTCOMES, extract)
        for arm, values in report[group]["vs_reference"].items():
            report[group].setdefault("gates", {})[arm] = {
                "completion": values["corrected_complete"]["delta"] >= .05
                and values["corrected_complete"]["ci95"][0] > 0.,
                "at_most_margin": {k: values[k]["ci95"][1] <= v for k, v in MARGINS.items()},
                "within_margin": {k: values[k]["ci95"][0] >= -v and values[k]["ci95"][1] <= v
                                  for k, v in MARGINS.items()},
                "risk": {k: values[k]["ci95"][1] <= .02 for k in ("final_wrong", "new_centered_mistake")},
            }
    report.update(config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  elapsed_s=perf_counter() - started, prediction_caps=CAPS)
    (args.out / "summary.json").write_text(json.dumps(report, default=jsonable, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

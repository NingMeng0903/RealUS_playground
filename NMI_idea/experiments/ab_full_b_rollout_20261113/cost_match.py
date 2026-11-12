"""One independent cost-only activation diagnostic after the primary study.

Use only the previously frozen model predictions for option selection. Actual
case outcomes are not used to select candidates or cases. The purpose is to
exercise some nonzero interventions at predeclared matched expected costs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

import acoustics
import audit
import control
import run


NAMESPACE = 716906
FORECASTS = {}
MARGINS = np.array([2., 2., .10])


def choose(task, obs, prior, args):
    key = f"{task}/{obs}"
    started = time.process_time()
    if key in FORECASTS:
        rows = FORECASTS[key]["arms"]["B"]["screen"]
        cached = True
    else:
        selections, _, _ = run.pair_plan(task, prior, obs, args)
        rows = selections["B"][1]["screen"]
        cached = False
    unit = run.DATA[task]["scene"].clock_unit
    passing = []
    for row in rows:
        d = row["delta"]
        if np.all(np.abs([d[3], d[4], d[2] * unit]) <= MARGINS + 1.e-12):
            passing.append(row)
    selected = max(passing, key=lambda row: (row["delta"][0], -row["delta"][1]), default=None)
    elapsed = time.process_time() - started
    if selected is None:
        return control.empty_option(), None, elapsed, cached
    return np.array(selected["option"], dtype=np.int64), selected, elapsed, cached


def one(item):
    index, args = item
    seed = acoustics.seed_for(NAMESPACE, index)
    rows = []
    for task, data in run.DATA.items():
        scene, geometry = data["scene"], data["geometry"]
        truth = acoustics.make_scenario(geometry.world, seed, "joint")
        _, _, detect, cdf = acoustics.probabilities(geometry, truth, data["clocks"], "joint")
        detect = np.ascontiguousarray(detect[:, :, None])
        cdf = np.ascontiguousarray(cdf[:, :, None])
        tape = acoustics.tape(seed, len(data["clocks"]))
        prior, obs, kind = run.initial_prior(data, detect, cdf, tape, "anchor")
        option, prediction, cpu, cached = choose(task, obs, prior, args)
        shared = dict(index=index, case_seed=seed, task=task, initial="anchor",
                      start_observation=obs, start_eval_only_kind=kind,
                      start_state=run.encode_state(prior, prior, scene.start, 0, 0, 0, scene.reserve),
                      perturbation_hash=hashlib.sha256(tape.tobytes()).hexdigest())
        for arm, program, planning in (("MPC", control.empty_option(), 0.),
                                       ("cost_matched_option", option, cpu)):
            row = run.actual(scene, data, prior, program, detect, cdf, tape, planning)
            rows.append(dict(shared, arm=arm, **row, predicted_match=prediction if arm != "MPC" else None,
                             model_forecast_cached=cached if arm != "MPC" else None))
    return rows


def main():
    global FORECASTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=run.HERE.parent / "RealUS_playground")
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--forecast", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.screen, args.validate, args.arms = 24, 48, ["MPC", "B"]
    FORECASTS = json.loads(args.forecast.read_text())
    args.out.mkdir(exist_ok=False, parents=True)
    run.initialize(args.repo, args.motion, 256, 6, "joint")
    run.warmup()
    settings = {"split": "independent_cost_diagnostic", "captures": 6, "n": args.n,
                "seed_namespace": NAMESPACE, "model_size": 256, "initial": ["anchor"],
                "sites_and_labels": run.TASKS, "forecast_sha256": hashlib.sha256(args.forecast.read_bytes()).hexdigest(),
                "motion_bank_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                "admission_margins": MARGINS.tolist(),
                "source_sha256": {name: hashlib.sha256((run.HERE / name).read_bytes()).hexdigest()
                                  for name in ("cost_match.py", "control.py", "run.py", "acoustics.py")}}
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    rows = []
    with (args.out / "cases.jsonl").open("w") as stream:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
            for i, batch in enumerate(pool.map(one, [(j, args) for j in range(args.n)]), 1):
                rows.extend(batch)
                for row in batch:
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                if i == 1 or i % 24 == 0 or i == args.n:
                    print(f"{i}/{args.n} paired cost diagnostic cases", flush=True)
    report = audit.run(rows, run.BANK, settings)
    active = {}
    for task in run.TASKS:
        cohort = [r for r in rows if r["task"] == task]
        treated = [r for r in cohort if r["arm"] == "cost_matched_option" and r["option_steps"]]
        base = {r["index"]: r for r in cohort if r["arm"] == "MPC"}
        active[task] = ({"n": len(treated), "options": audit.arm_summary(treated),
                         "mpc": audit.arm_summary([base[r["index"]] for r in treated]),
                         "contrast": audit.compare(treated, [base[r["index"]] for r in treated])}
                        if treated else {"n": 0})
    report["publicly_triggered_subset"] = active
    (args.out / "paired.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

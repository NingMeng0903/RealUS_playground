"""Evaluate repeatable first-action headroom in the checked reference graph.

Keeps anatomy, dropout times, persistent clutter, prefix, and sampled motion
fixed. Only post-prefix acoustic innovations change. The old replanning MPC
selects the baseline root from public history, and all forced roots use its
public tail replanner. Select an oracle root on even replicates and evaluate
that selection on odd replicates. This is a development diagnostic.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np

import challenge as study

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noise_headroom import NoiseBank, suffix_observe


SEED_BASE = 6173093


def capture(case, path, clock, q, cache, noise, jitter):
    key = path.id, clock, np.asarray(q).tobytes()
    if key not in cache:
        trace, exit_q = study.MODEL.execute(path, clock, lambda t: case.true_angle_rate(t)[0], q)
        cost = np.array([trace["travel_mm"], trace["turn_deg"], trace["stroke_mm"]])
        cache[key] = trace["legal_completed"], trace["t_end"], exit_q, cost
    valid, time, exit_q, cost = cache[key]
    if not valid:
        return None, time, exit_q, cost
    world = study.WORLD
    frame = suffix_observe(world, case, time, *path.goal, noise)
    angle, _ = case.true_angle_rate(time)
    base, _ = world.projection(angle, *path.goal)
    persistent = world._persistent_clutter(case, angle, *path.goal, base)
    if persistent is not None:
        values = frame.centroids.copy()
        values[frame.labels.index(-1)] = np.clip(persistent + jitter, study.LOWER, study.UPPER)
        order = np.argsort(values[:, 0], kind="stable")
        frame = replace(frame, centroids=values[order], labels=tuple(frame.labels[i] for i in order))
    return frame, time, exit_q, cost


def force_root(case, graph, initial, first, cache, replicates):
    ensemble, clock, q = initial
    old = ensemble.filters
    ensemble.filters = tuple(f.copy() for f in old)
    first_noise, first_jitter, tail_noise, tail_jitter = replicates
    try:
        start, time, exit_q, first_cost = capture(case, first, clock, q, cache, first_noise, first_jitter)
        if start is None:
            return 0, 0, True
        ensemble.assimilate(study.packet(start), 64)
        planned_tail, _ = study.tail(ensemble, graph.tails[first.id])
        final, _, _, terminal_cost = capture(case, planned_tail, time, exit_q, cache,
                                            tail_noise, tail_jitter)
        caps = (study.MAX_TRAVEL_MM, study.MAX_TURN_DEG, study.MAX_STROKE_MM)
        if final is None or np.any(first_cost + terminal_cost > caps):
            return 0, 0, True
        ordinal, _, _ = ensemble.claims(study.packet(final))
        if ordinal is None:
            return 0, 0, False
        success = int(final.labels[ordinal] == ensemble.label)
        return success, 1 - success, False
    finally:
        ensemble.filters = old


def one(job):
    index, repeats = job
    seed = int(np.random.SeedSequence([SEED_BASE, index]).generate_state(1)[0])
    case = study.WORLD.case(seed)
    initial_cache, cache = {}, {}
    noise = []
    for rep in range(repeats):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep, 89172]))
        jitter = rng.normal(size=(2, 2)) * [1., .68]
        noise.append((NoiseBank.make(seed, rep, 0), jitter[0],
                      NoiseBank.make(seed, rep, 1), jitter[1]))
    rows = []
    for task, graph in study.GRAPHS.items():
        initial = study.start_task(case, task, "fitted", initial_cache)
        if initial is None:
            rows.append(dict(index=index, seed=seed, task=task, prefix_failed=True))
            continue
        predicted, actions = study.plan(initial[0], graph)
        result = np.array([[force_root(case, graph, initial, first, cache, draw)
                            for first in graph.first] for draw in noise])
        baseline = actions["MPC"]
        train = result[::2, :, 0] - result[::2, :, 1]
        order = sorted(range(len(graph.first)), key=lambda j: (-float(train[:, j].mean()), j != baseline, j))
        best = order[0]
        rows.append(dict(index=index, seed=seed, task=task, prefix_failed=False,
                         roots=[p.goal for p in graph.first], scores=predicted,
                         baseline=baseline, paired=actions["B_challenge"], selected=best,
                         replicates=result.tolist()))
    return rows


def summarize(rows):
    data = {}
    for task in study.TASKS:
        group = [row for row in rows if row["task"] == task]
        valid = [row for row in group if not row["prefix_failed"]]
        invalid = len(group) - len(valid)
        arr = np.array([row["replicates"] for row in valid])
        indices = np.arange(len(valid))
        best = np.array([row["selected"] for row in valid])
        base = np.array([row["baseline"] for row in valid])
        paired = np.array([row["paired"] for row in valid])
        n = len(group)
        test = arr[:, 1::2]
        success_baseline = test[indices, :, base, 0].mean(axis=1)
        success_oracle = test[indices, :, best, 0].mean(axis=1)
        success_b = test[indices, :, paired, 0].mean(axis=1)
        advantage = success_oracle - success_baseline
        rng = np.random.default_rng(871)
        sample = np.r_[advantage, np.zeros(invalid)]
        boot = sample[rng.integers(n, size=(10000, n))].mean(axis=1)
        data[task] = dict(n_cases=n, prefix_failures=invalid,
                          root_outcomes=np.sum(arr, axis=(0, 1)).tolist(),
                          mpc_held_half=float(success_baseline.sum() / n),
                          paired_b_held_half=float(success_b.sum() / n),
                          noise_split_oracle_held_half=float(success_oracle.sum() / n),
                          noise_split_gain_ci95=np.quantile(boot, [.025, .975]).tolist(),
                          switched_cases=int(np.count_nonzero(best != base)),
                          noise_split_better_worse=[int(np.count_nonzero(advantage > 0)),
                                                    int(np.count_nonzero(advantage < 0))])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=study.DEFAULT_REPO)
    parser.add_argument("--fit", type=Path, default=Path(__file__).with_name("clutter_fit.json"))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 4 or args.repeats % 2:
        raise ValueError("repeat count must be even and at least four")
    study.initialize(args.repo, 16, args.fit)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as pool:
        for i, cohort in enumerate(pool.map(one, [(k, args.repeats) for k in range(args.n)]), 1):
            rows.extend(cohort)
            print(f"{i}/{args.n} paired noise-repeat cases", flush=True)
    output = dict(seed_base=SEED_BASE, replicates=args.repeats, summary=summarize(rows), rows=rows)
    args.out.write_text(json.dumps(output, allow_nan=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

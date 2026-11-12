"""Test public route ranking at identical all-return MPC checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import bench
import route_predictor


BASE = "MPC"
CONTEXTS = ("checkpoint", "probe")
POLICIES = (BASE, "fixed", "nearest", "relations")
SET = bench.SET
REPEATS = 16
CAPTURES = 6
FIT = None


def cheapest(scene, at, nodes):
    return min(nodes, key=lambda node: (*scene.edge_costs[at, node], node), default=-1)


def routes(data, state, eligible):
    result = {BASE: dict(goal=-1, features={}, witness=None)}
    belief, good, reserve, at, slot, clock, progress, _ = state
    scene = data.scene
    if not eligible or progress == len(scene.targets) or slot >= CAPTURES:
        return result
    baseline = bench.set_mpc.action(data.all_returns, belief, good, at, slot, clock, progress, reserve)
    target = scene.targets[progress]
    witnesses = {}
    for goal in np.flatnonzero(scene.site == target):
        witness = bench.continuations.witness_for(scene, at, CAPTURES - slot, progress, reserve, (int(goal),))
        if witness is not None:
            witnesses[int(goal)] = witness
    centered = [j for j in witnesses if scene.tilt[j] == 0]
    tilted = [j for j in witnesses if scene.tilt[j] != 0]
    opposing = [j for j in tilted if scene.tilt[j] == -scene.tilt[at]]
    for key, group in (("advance", centered), ("view", opposing or tilted)):
        goal = cheapest(scene, at, group)
        if goal < 0 or goal == baseline:
            continue
        tcp, turn, rail, ticks = scene.edge_costs[at, goal]
        result[key] = dict(goal=goal, witness=witnesses[goal], features=dict(
            travel=float(tcp), turn=float(turn), rail=float(rail), duration=float(ticks * scene.clock_unit),
            tilt=int(scene.tilt[goal]), rho_change=int(scene.rho[goal] != scene.rho[at]),
            same_site=int(scene.site[goal] == scene.site[at])))
    return result


def probe_goal(data, state):
    _, _, reserve, at, slot, _, progress, _ = state
    scene = data.scene
    if slot != 2 or progress == len(scene.targets):
        return -1
    candidates = []
    for goal in np.flatnonzero(scene.site == scene.site[at]):
        if goal != at and scene.tilt[goal] == scene.tilt[at]:
            continue
        expense = scene.edge_costs[at, goal]
        if not np.all(expense <= reserve + 1.e-8):
            continue
        if bench.continuations.full_witness(scene, int(goal), CAPTURES - slot - 1,
                                            progress, reserve - expense) is not None:
            candidates.append(int(goal))
    changes = [j for j in candidates if j != at]
    return cheapest(scene, at, changes) if changes else (int(at) if at in candidates else -1)


def belief_features(prior, material):
    return dict(displacement=float(prior @ material), log_ess=float(-np.log(prior @ prior)),
                displacement_positive=float(prior[material >= 0.].sum()))


def public_packet(data, before, after, row):
    scene, tables = data.scene, data.tables
    at, progress, slot, clock = after[3], after[6], after[4], after[5]
    common = dict(progress=int(progress), slot=int(slot), clock=int(clock), site=int(scene.site[at]),
                  tilt=int(scene.tilt[at]), rho=int(scene.rho[at]), before_good=float(before[1].sum()),
                  target_offset=int(scene.targets[progress] - scene.site[at]) if progress < len(scene.targets) else 0)
    common.update({f"resource{i}": float(v) for i, v in enumerate(after[2])})
    common.update({f"before_{k}": v for k, v in belief_features(before[0], scene.material_coordinate).items()})
    if row is None:
        return dict(common=common, nearest={}, relations={})
    packet = int(row[5])
    coarse = int(tables.nearest_packet[packet])
    n_bins = len(tables.coordinates)
    near = tables.coordinates[(coarse - 1) % n_bins] if coarse else np.zeros(2)
    near_prior, confidence = bench.control.posterior(before[0], data.compressed[0], data.compressed[1],
                                                      clock, scene.to_address[at], coarse)
    nearest = dict(present=int(coarse > 0), multiple=int(coarse > n_bins),
                   nearest_x=float(near[0]), nearest_z=float(near[1]), nearest_confidence=float(confidence))
    nearest.update({f"nearest_{k}": v for k, v in belief_features(near_prior, scene.material_coordinate).items()})
    visible = tables.bins[packet][tables.bins[packet] > 0]
    counts = np.bincount(visible, minlength=n_bins + 1)
    coords = tables.coordinates[visible - 1]
    spread = np.ptp(coords, axis=0) if len(coords) else np.zeros(2)
    pair_max = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1).max() if len(coords) else 0.
    relations = dict(return_count=len(visible), span_x=float(spread[0]), span_z=float(spread[1]),
                     max_pair_distance=float(pair_max), set_confidence=float(row[11]))
    relations.update({f"bin{i}": int(counts[i]) for i in range(1, n_bins + 1)})
    relations.update({f"set_{k}": v for k, v in belief_features(after[0], scene.material_coordinate).items()})
    return dict(common=common, nearest=nearest, relations=relations)


def tape_at(data, seed, index, through_clock, preserved):
    subseed = bench.acoustics.seed_for(seed, index)
    tape = bench.acoustics.tape(subseed, len(data.clocks))
    tape[:through_clock + 1] = preserved[:through_clock + 1]
    return tape, subseed


def finish(data, detect, cdf, tape, state, forced, upstream_s):
    started = time.process_time()
    final, trace = bench.advance(data, detect, cdf, tape, state, SET, forced=forced)
    return dict(measured=bench.metric(data, final, upstream_s + time.process_time() - started),
                tail=trace.tolist(), final_state_sha256=bench.state_digest(final))


def collect_context(task, seed, context, detect, cdf, data, before, state, step, prefix_tape,
                    root, has_probe, eligible, prefix_cpu, probe_cpu):
    start = time.process_time()
    public = public_packet(data, before, state, step)
    actions = routes(data, state, eligible)
    chosen, forecasts = route_predictor.choose(FIT, task, context, public, actions) if FIT is not None else (None, None)
    planning_s = time.process_time() - start
    repeats, incumbent = {k: [] for k in actions}, []
    base_index = 210000 if context == "checkpoint" else 310000
    tape_seeds = []
    for rep in range(REPEATS):
        tape, tape_seed = tape_at(data, seed, base_index + rep, state[5], prefix_tape)
        tape_seeds.append(tape_seed)
        for key, action in actions.items():
            result = finish(data, detect, cdf, tape, state, action["goal"], prefix_cpu + probe_cpu)
            repeats[key].append(result["measured"])
        if has_probe:
            direct = finish(data, detect, cdf, tape, root, -1, prefix_cpu)
            incumbent.append(direct["measured"])
        else:
            incumbent.append(repeats[BASE][-1])
    record = dict(public=public, actions=actions, repeat=repeats, paired_no_probe=incumbent,
                  repeat_seeds=tape_seeds, state_sha256=bench.state_digest(state))
    if chosen is not None:
        tape, tape_seed = tape_at(data, seed, base_index + 1000000, state[5], prefix_tape)
        measurements = {}
        for policy in POLICIES:
            key = chosen[policy]
            overhead = planning_s if policy != BASE else 0.
            result = finish(data, detect, cdf, tape, state, actions[key]["goal"], prefix_cpu + probe_cpu + overhead)
            measurements[policy] = dict(result, route=key, goal=actions[key]["goal"])
        without = finish(data, detect, cdf, tape, root, -1, prefix_cpu) if has_probe else measurements[BASE]
        record.update(fresh=measurements, fresh_no_probe=without, fresh_seed=tape_seed,
                      fresh_tape_sha256=hashlib.sha256(tape.tobytes()).hexdigest(), forecasts=forecasts,
                      feature_and_both_rankers_cpu_s=planning_s)
    return record


def trial(job):
    index, split = job
    seed = bench.acoustics.seed_for(bench.NAMESPACES[split], index)
    rows = []
    for task, data in bench.DATA.items():
        detect, cdf = bench.case(task, seed)
        original = bench.acoustics.tape(seed, len(data.clocks))
        start = time.process_time()
        initial, anchor = bench.begin(data, detect, cdf, original, SET)
        before, first = bench.advance(data, detect, cdf, original, initial, SET, stop=1)
        state, second = bench.advance(data, detect, cdf, original, before, SET, stop=2)
        prefix_cpu = time.process_time() - start
        eligible = state[4] == 2 and len(second) == 1 and state[6] < len(data.scene.targets)
        common = np.concatenate((first, second))
        checkpoint = collect_context(task, seed, "checkpoint", detect, cdf, data, before, state,
            second[-1] if len(second) else None, original, state, False, eligible, prefix_cpu, 0.)
        start = time.process_time()
        goal = probe_goal(data, state) if eligible else -1
        if goal >= 0:
            probe_tape, prefix_seed = tape_at(data, seed, 300000, state[5], original)
            after_probe, probe_trace = bench.advance(data, detect, cdf, probe_tape, state, SET,
                                                     forced=goal, stop=state[4] + 1)
            assert len(probe_trace) == 1
        else:
            after_probe, probe_trace, probe_tape, prefix_seed = state, np.empty((0, 16)), original, seed
        probe_cpu = time.process_time() - start
        probe = collect_context(task, seed, "probe", detect, cdf, data, state, after_probe,
            probe_trace[-1] if len(probe_trace) else None, probe_tape, state, goal >= 0,
            goal >= 0 and after_probe[6] == state[6], prefix_cpu, probe_cpu)
        if FIT is not None:
            paid_tape, _ = tape_at(data, seed, 1310000, after_probe[5], probe_tape)
            free = checkpoint["fresh"]["relations"]
            probe["fresh_free_relations"] = finish(data, detect, cdf, paid_tape, state, free["goal"],
                                                     prefix_cpu + checkpoint["feature_and_both_rankers_cpu_s"])
        rows.append(dict(index=index, case_seed=seed, task=task, anchor=anchor,
                         prefix=common.tolist(), probe_goal=goal, probe_trace=probe_trace.tolist(),
                         probe_prefix_seed=prefix_seed, contexts=dict(checkpoint=checkpoint, probe=probe)))
    return rows


def report(rows):
    result = {}
    for task in sorted({r["task"] for r in rows}):
        group = [row for row in rows if row["task"] == task]
        result[task] = {}
        for context in CONTEXTS:
            states = [row["contexts"][context] for row in group]
            base = np.asarray([s["repeat"][BASE] for s in states])
            no_probe = [s["paired_no_probe"] for s in states]
            oracle = base.copy()
            picks, availability, signs = Counter(), Counter(), Counter()
            for i, state in enumerate(states):
                outcomes = {key: np.asarray(value) for key, value in state["repeat"].items()}
                availability.update(outcomes.keys())
                mid = len(base[i]) // 2
                halves = (slice(0, mid), slice(mid, None))
                for key, y in outcomes.items():
                    if key != BASE:
                        d = y[:, 0] - base[i, :, 0]
                        sign = tuple(int(np.sign(d[s].mean())) for s in halves)
                        signs[f"{key}/{sign[0]:+d},{sign[1]:+d}"] += 1
                for training, testing in zip(halves, reversed(halves)):
                    best = max(outcomes, key=lambda key: outcomes[key][training, 0].mean())
                    picks[best] += 1
                    oracle[i, testing] = outcomes[best][testing]
            summary = dict(opportunity=bench.summarize(oracle, base), case_options=dict(availability),
                           ranking_half_choices=dict(picks), half_signs=dict(signs))
            if "fresh" in states[0]:
                chosen = {policy: [s["fresh"][policy]["route"] for s in states] for policy in POLICIES}
                repeated = {policy: [s["repeat"][key] for s, key in zip(states, keys)] for policy, keys in chosen.items()}
                fresh = {policy: [s["fresh"][policy]["measured"] for s in states] for policy in POLICIES}
                fresh_net = [s["fresh_no_probe"]["measured"] for s in states]
                summary.update(choice_counts={policy: dict(Counter(keys)) for policy, keys in chosen.items()},
                    relation_choice_disagreement_with_nearest=sum(a != b for a, b in zip(chosen["relations"], chosen["nearest"])),
                    fresh_counts={policy: dict(correct=sum(a[0] for a in values), wrong=sum(a[1] for a in values))
                                  for policy, values in {**fresh, "no_probe": fresh_net}.items()},
                    repeated_vs_exit_mpc={k: bench.summarize(v, base) for k, v in repeated.items() if k != BASE},
                    repeated_vs_no_probe={k: bench.summarize(v, no_probe) for k, v in repeated.items()},
                    fresh_vs_exit_mpc={k: bench.summarize(v, fresh[BASE]) for k, v in fresh.items() if k != BASE},
                    fresh_vs_no_probe={k: bench.summarize(v, fresh_net) for k, v in fresh.items()},
                    relations_vs_same_packet_nearest=bench.summarize(repeated["relations"], repeated["nearest"]),
                    relations_vs_train_fixed=bench.summarize(repeated["relations"], repeated["fixed"]))
                if context == "probe":
                    free = [s["fresh_free_relations"]["measured"] for s in states]
                    summary["paid_vs_free_relations_same_tape"] = bench.summarize(fresh["relations"], free)
                    summary["probe_actions"] = dict(Counter(row["probe_goal"] for row in group))
            result[task][context] = summary
    return result


def read_rows(folder):
    with (Path(folder) / "cases.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--motion", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("smoke", "train", "heldout"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    global FIT, REPEATS
    if args.split == "heldout" and args.freeze is None:
        p.error("--freeze is required for held-out execution")
    if args.split == "smoke":
        REPEATS = 4
    if args.freeze is not None:
        obj = json.loads(args.freeze.read_text())
        for name, sha in obj["sha256"].items():
            assert hashlib.sha256((args.freeze.parent / name).read_bytes()).hexdigest() == sha, name
        FIT = json.loads((args.freeze.parent / "route_fit.json").read_text())
        if args.split == "heldout":
            assert args.n == obj["n_heldout"] and bench.NAMESPACES["heldout"] == obj["namespace"]
    args.out.mkdir(parents=True, exist_ok=True)
    bank, sources = bench.initialize(args.repo, args.motion, args.out)
    bench.sensor_case((10000, "smoke"))
    settings = dict(split=args.split, n=args.n, n_repeats=REPEATS, bank_sha256=bank, sources=sources,
                    model_size=256, grid=3, trace=bench.TRACE_FIELDS, metrics=bench.FIELDS,
                    namespace=bench.NAMESPACES[args.split], heldout_frozen=args.freeze is not None)
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    results = report(bench.collect(args.out, args.n, args.workers, args.split, trial, "route"))
    (args.out / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    for task, group in results.items():
        for context, value in group.items():
            print(task, context, round(value["opportunity"]["delta"]["correct_completion"], 4),
                  value.get("fresh_counts", {}))


if __name__ == "__main__":
    main()

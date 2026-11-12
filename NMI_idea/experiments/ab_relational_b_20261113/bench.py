"""Paired full episodes under three views of the same acoustic returns."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import lzma
from multiprocessing import get_context
from pathlib import Path
import sys
import time

from numba import njit
import numpy as np


HERE = Path(__file__).resolve().parent
FULL = "ab_full_b_rollout_20261113"
CONT = "ab_continuation_b_20261113"
NAMESPACES = dict(smoke=719500, sensor=719501, train=719502, heldout=719503, sensitivity=719504)
OLD, NEAREST, SET = 0, 1, 2
ARMS = ("legacy_nearest_7x7", "matched_nearest", "all_returns")
FIELDS = ("correct_completion", "any_wrong", "tcp_mm", "turn_deg", "motion_s", "rail_mm",
          "captures", "combined_s")
TRACE_FIELDS = ("start", "goal", "slot", "clock", "observation", "set_packet", "selected_slot",
                "eval_only_kind", "eval_only_nearest_kind", "eval_only_target_present", "claim", "confidence",
                "tcp_mm", "turn_deg", "rail_mm", "forced")
EMPTY = np.empty(0, dtype=np.int64)
DATA = {}


@dataclass(frozen=True)
class Task:
    scene: object
    geometry: object
    clocks: np.ndarray
    tables: object
    compressed: tuple
    all_returns: tuple


def initialize(repo, motion, out, model_size=256, grid=3):
    global acoustics, control, shared, set_mpc, return_sets, continuations, summarize
    base = Path(repo).resolve() / "NMI_idea" / "experiments"
    full = base / FULL
    sys.path.insert(0, str(full))
    sys.path.insert(0, str(base / CONT))
    import acoustics
    import control
    import run as shared
    import continuations
    from paired_metrics import summarize
    import return_sets
    import set_mpc

    hashes = json.loads((full / "FREEZE.json").read_text())["sha256"]
    for name in ("acoustics.py", "control.py", "run.py"):
        assert hashlib.sha256((full / name).read_bytes()).hexdigest() == hashes[name]
    if motion is None:
        motion = out / "BANK_MAIN.json"
        motion.write_bytes(lzma.decompress((full / "motion/BANK_MAIN.json.xz").read_bytes()))
    bank_hash = hashlib.sha256(Path(motion).read_bytes()).hexdigest()
    assert bank_hash == hashes["motion/BANK_MAIN.json"]
    shared.initialize(repo, motion, model_size, shared.CAPTURES, "joint")
    for task, info in shared.DATA.items():
        scene = info["scene"]
        tables = return_sets.build_tables(scene.detect, scene.cdf, grid)
        compressed = (tables.nearest_total, tables.nearest_correct, *scene.public()[2:])
        rich = (tables.total, tables.joint, tables.claimable, tables.bins, scene.to_address, scene.site,
                scene.edge_costs, scene.targets, scene.potential, scene.needs)
        DATA[task] = Task(scene, info["geometry"], info["clocks"], tables, compressed, rich)
    return bank_hash, {key: hashes[key] for key in ("acoustics.py", "control.py", "run.py")}


@njit(cache=True)
def update(mode, compressed, rich, bins, clock, address, obs, prior, good, target_site):
    if mode == SET:
        like, joint = rich[0], rich[1]
        post, confidence, selected = set_mpc.posterior(prior, like, joint, bins, clock, address, obs)
        claim = target_site and selected >= 0 and confidence >= control.CONFIDENCE
        new_good, _ = set_mpc.advance(prior, good, like, joint, clock, address, obs, selected, claim)
    else:
        like, correct = compressed[0], compressed[1]
        post, confidence = control.posterior(prior, like, correct, clock, address, obs)
        selected = -1
        claim = target_site and obs > 0 and confidence >= control.CONFIDENCE
        new_good, _ = control.advance_joint(prior, good, like, correct, clock, address, obs, claim)
    return post, new_good, confidence, selected, claim


@njit(cache=True)
def run(mode, compressed, rich, mapping, lookup, nearest_slot, nearest_packet,
        true_detect, true_cdf, tape, state, forced, stop_slot):
    belief, good, reserve, at, slot, clock, progress, wrong = state
    _, _, _, bins, addresses, sites, costs, targets, potential, _ = rich
    trace = np.empty((len(potential) - 1 - slot, 16), dtype=np.float64)
    length = 0
    while slot < stop_slot and slot < len(potential) - 1 and progress < len(targets):
        is_forced = length == 0 and forced >= 0
        if is_forced:
            goal = forced
        elif mode == SET:
            goal = set_mpc.action(rich, belief, good, at, slot, clock, progress, reserve)
        else:
            goal = control.mpc_action(compressed, belief, good, at, slot, clock, progress, reserve)
        if goal < 0:
            break
        if not control.allowed(costs, at, goal, reserve):
            raise ValueError("selected motion violates shared caps")
        cost = costs[at, goal]
        clock += int(cost[3])
        address = addresses[goal]
        packet, kinds = return_sets.draw(true_detect, true_cdf, clock, address, 0, tape, mapping, lookup)
        nearest = nearest_slot[packet]
        nearest_kind = kinds[nearest] if nearest >= 0 else -1
        if mode == SET:
            obs, kind = packet, -1
        elif mode == NEAREST:
            obs, kind = nearest_packet[packet], nearest_kind
        else:
            obs, kind = control.capture_draw(true_detect, true_cdf, clock, address, 0, tape)
        target_site = sites[goal] == targets[progress]
        belief, good, confidence, selected, claim = update(mode, compressed, rich, bins, clock, address,
                                                          obs, belief, good, target_site)
        if mode == SET and selected >= 0:
            kind = kinds[selected]
        if mode == NEAREST:
            selected = nearest
        if claim:
            progress += 1
            wrong = wrong or kind != 0
        trace[length, 0], trace[length, 1], trace[length, 2], trace[length, 3] = at, goal, slot, clock
        trace[length, 4], trace[length, 5], trace[length, 6], trace[length, 7] = obs, packet, selected, kind
        trace[length, 8], trace[length, 9] = nearest_kind, int(np.any(kinds == 0))
        trace[length, 10], trace[length, 11] = int(claim), confidence
        trace[length, 12], trace[length, 13], trace[length, 14] = cost[0], cost[1], cost[2]
        trace[length, 15] = int(is_forced)
        reserve = reserve - cost
        at, slot, length = goal, slot + 1, length + 1
    return (belief, good, reserve, at, slot, clock, progress, wrong), trace[:length]


def case(task, seed):
    data = DATA[task]
    scenario = acoustics.make_scenario(data.geometry.world, seed)
    _, _, detect, cdf = acoustics.probabilities(data.geometry, scenario, data.clocks, "joint")
    return np.ascontiguousarray(detect[:, :, None]), np.ascontiguousarray(cdf[:, :, None])


def compressed_model(data, mode):
    return data.scene.public() if mode == OLD else data.compressed


def begin(data, detect, cdf, tape, mode):
    scene, tables = data.scene, data.tables
    prior = np.full(scene.likelihood.shape[-1], 1. / scene.likelihood.shape[-1])
    addr = scene.to_address[scene.start]
    packet, _ = return_sets.draw(detect, cdf, 0, addr, 0, tape, tables.mapping, tables.lookup)
    obs = packet if mode == SET else int(tables.nearest_packet[packet])
    if mode == OLD:
        obs, _ = control.capture_draw(detect, cdf, 0, addr, 0, tape)
    posterior, _, _, _, _ = update(mode, compressed_model(data, mode), data.all_returns, tables.bins,
                                    0, addr, obs, prior, prior, False)
    return (posterior, posterior.copy(), scene.reserve.copy(), scene.start, 0, 0, 0, False), int(packet)


def advance(data, detect, cdf, tape, state, mode, forced=-1, stop=6):
    tables = data.tables
    return run(mode, compressed_model(data, mode), data.all_returns, tables.mapping, tables.lookup,
               tables.nearest_slot, tables.nearest_packet, detect, cdf, tape, state, forced, stop)


def state_digest(state):
    b, g, r, at, slot, clock, progress, _ = state
    return shared.encode_state(b, g, at, slot, clock, progress, r)


def metric(data, final, elapsed):
    _, _, reserve, _, slot, clock, progress, wrong = final
    expense = data.scene.reserve - reserve
    t = float(data.clocks[clock])
    return [int(progress == len(data.scene.targets) and not wrong), int(wrong), *map(float, expense[:2]),
            t, float(expense[2]), int(slot), t + elapsed]


def legacy_parity(data, detect, cdf, tape, state, final, trace):
    prior = data.scene.public()
    s, w, n, _, _, _, _, costs, old = control.simulate(prior, detect, cdf, 0, tape, state[0],
        data.scene.start, data.scene.reserve, control.empty_option())
    assert [s, w] == metric(data, final, 0.)[:2] and n == len(trace)
    np.testing.assert_allclose(trace[:, [0, 1, 4, 11, 10, 7, 12, 13, 14, 3]],
                               old[:n, [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]], atol=1.e-12, rtol=0.)


def sensor_case(job):
    index, split = job
    seed = acoustics.seed_for(NAMESPACES[split], index)
    rows = []
    for task, data in DATA.items():
        detect, cdf = case(task, seed)
        tape = acoustics.tape(seed, len(data.clocks))
        arms = {}
        for mode, arm in enumerate(ARMS):
            start = time.process_time()
            state, anchor = begin(data, detect, cdf, tape, mode)
            final, trace = advance(data, detect, cdf, tape, state, mode)
            arms[arm] = dict(measured=metric(data, final, time.process_time() - start), anchor=anchor,
                             start=state_digest(state), end=state_digest(final), trace=trace.tolist())
            if index < 4 and mode == OLD:
                legacy_parity(data, detect, cdf, tape, state, final, trace)
        rows.append(dict(index=index, case_seed=seed, task=task, tape_sha256=hashlib.sha256(tape.tobytes()).hexdigest(), arms=arms))
    return rows


def sensor_report(rows):
    result = {}
    for task in DATA:
        group = [row for row in rows if row["task"] == task]
        vals = {arm: [row["arms"][arm]["measured"] for row in group] for arm in ARMS}
        identity = {}
        for arm in ARMS:
            c = Counter()
            for row in group:
                progress = 0
                for edge in row["arms"][arm]["trace"]:
                    if DATA[task].scene.site[int(edge[1])] == DATA[task].scene.targets[progress]:
                        c["current_target_frames"] += 1
                        c["target_present_frames"] += int(edge[9])
                        c["correct_claims"] += int(edge[10] and edge[7] == 0.)
                        c["wrong_claims"] += int(edge[10] and edge[7] != 0.)
                        c["correct_claim_where_coarse_nearest_wrong"] += int(edge[10] and edge[7] == 0. and edge[8] != 0.)
                    progress += int(edge[10])
            identity[arm] = c
        result[task] = dict(counts={arm: dict(correct=sum(m[0] for m in values), wrong=sum(m[1] for m in values))
                                     for arm, values in vals.items()}, identity=identity,
                            all_vs_matched_nearest=summarize(vals[ARMS[SET]], vals[ARMS[NEAREST]]),
                            all_vs_legacy=summarize(vals[ARMS[SET]], vals[ARMS[OLD]]),
                            matched_nearest_vs_legacy=summarize(vals[ARMS[NEAREST]], vals[ARMS[OLD]]))
    return result


def collect(folder, n, workers, split, worker, label):
    rows = []
    with ((folder / "cases.jsonl").open("w") as out,
          ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as pool):
        for i, batch in enumerate(pool.map(worker, ((j, split) for j in range(n)))):
            rows.extend(batch)
            for row in batch:
                out.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            if i % 12 == 0 or i + 1 == n:
                out.flush()
                print(f"{label}/{split}: {i + 1}/{n}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--motion", type=Path)
    p.add_argument("--split", choices=("smoke", "sensor", "sensitivity"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--model-size", type=int, default=256)
    p.add_argument("--grid", choices=(3, 5), type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bank, sources = initialize(args.repo, args.motion, args.out, args.model_size, args.grid)
    sensor_case((10000, "smoke"))
    settings = dict(split=args.split, n=args.n, model_size=args.model_size, grid=args.grid, bank_sha256=bank,
                    sources=sources, trace=TRACE_FIELDS, metrics=FIELDS, arms=ARMS, namespace=NAMESPACES[args.split])
    (args.out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    report = sensor_report(collect(args.out, args.n, args.workers, args.split, sensor_case, "sensor"))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for task, row in report.items():
        print(task, row["counts"], row["identity"]["all_returns"])


if __name__ == "__main__":
    main()

"""Privileged six-decision teacher, full ordinary MPC continuations and strict public readout."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from corridors import OFFSETS, nominal_fields, path, scaffold, scan, view_fields, observe_hold
from environment import spec_for

SEED = 535047191
POLICIES = ("strict", "composite")


def menu(env, spec, node, limits, views, region, width=9, remaining_options=None):
    s, end = int(node.station), spec[2]
    width = min(width, end - s)
    route = env.route
    spine = scaffold(env, s, end, int(node.sector))
    plain = path(env, s, spine[:width+1])
    # Physical reserve is computed at the common end pose. Earlier regions share the slack.
    tip = int(plain.sectors[-1])
    pos = s + width - spec[1]
    max_ticks = limits.ticks - node.tick - int(limits.reserve_ticks[pos, tip])
    max_mm = limits.mm - node.distance - limits.reserve_mm[pos, tip]
    if max_ticks < plain.ticks or max_mm < plain.distance:
        return {"continue": None}
    remaining = 6-region if remaining_options is None else remaining_options
    ticks = plain.ticks + min(20, (max_ticks - plain.ticks) // remaining)
    distance = plain.distance + min(85., (max_mm - plain.distance) / remaining)
    options = []
    for offset in OFFSETS:
        bump = np.rint(offset * np.sin(np.linspace(0., np.pi, width+1))).astype(int)
        candidate = path(env, s, (plain.sectors + bump) % 24, offset)
        if candidate and candidate.ticks <= ticks and candidate.distance <= distance + 1.e-8:
            options.append(candidate)
    candidates = {"continue": None, "reference": (plain, width//2, ticks-plain.ticks)}
    used = {(tuple(plain.slots), width//2, ticks-plain.ticks)}
    for key, field, weights in (("nominal", views[0], [.13,.87,0.]),
                                ("individual", views[1], [.13,.87,0.]),
                                ("pair", views[1], [.18,.32,.50])):
        best_value, best, hold_at = -np.inf, None, None
        for candidate in options:
            merit = field[np.arange(s,s+width+1), candidate.sectors] @ weights
            j = int(1 + np.argmax(merit[1:-1]))
            value = .55 * np.mean(merit[1:]) + .45 * merit[j]
            if value > best_value or (value == best_value and abs(candidate.offset) < abs(best.offset)):
                best_value, best, hold_at = value, candidate, j
        signature = (tuple(best.slots), hold_at, ticks-best.ticks)
        if signature not in used:
            candidates[key] = (best, hold_at, ticks-best.ticks)
            used.add(signature)
    return candidates


def apply(env, spec, node, trial, limits, choice, length=9):
    if choice is None:
        end = min(node.station+length, spec[2])
        while node.station < end:
            env.route.advance(node, spec, trial, limits)
        return
    plan, hold_at, extra = choice
    for i, slot in enumerate(plan.slots, 1):
        scan.ref.reference.apply_route_slot(env, spec, node, trial, limits, int(slot))
        if i == hold_at:
            observe_hold(env, node, spec, trial, extra)


def grade(frames, history):
    r = scan.get_outcome(frames, history.fork())
    return dict(strict=r["pass"], wrong=r["wrong_regions"], false_scan=r["wrong_delivery"],
                correct=r["correct_regions"], captured=r["captured_regions"], labels=r["chosen_labels"],
                composite=r["score"])


def rollout(env, spec, prefix, trial, limits, choice, history, masked=False):
    node = prefix.fork()
    apply(env, spec, node, trial, limits, choice)
    last = len(node.reader.frames)
    scan.ref.finish(env, node, spec, trial, limits)
    result = grade(node.reader.frames, history)
    if masked and choice is not None:
        no_images = prefix.reader.frames + node.reader.frames[last:]
        result["masked"] = grade(no_images, history)
    return result


def best_branch(values, rule):
    def key(name):
        outcomes = values[name]
        if rule == "composite":
            return (sum(o["composite"] for o in outcomes),)
        return (sum(o["strict"] for o in outcomes), sum(o["correct"] for o in outcomes),
                -sum(o["wrong"] for o in outcomes), -sum(o["false_scan"] for o in outcomes))
    return max(values, key=key)


def episode(case, env, spec, original, trial, limits, views, checks, rule, n_rollouts, cache):
    node = original.fork()
    histories = scan.public_context(env)[0]
    log, spans, previous = [], [], []
    for r, s in enumerate(checks):
        while node.station < s:
            env.route.advance(node, spec, trial, limits)
        histories.append(node.reader.frames)
        options = menu(env, spec, node, limits, views, r)
        key = (r, *previous)
        if key in cache:
            values, matches = cache[key]
        else:
            values = {k: [] for k in options}
            for t in range(n_rollouts):
                independent = trial.branch(node.tick, scan.rng_seed(case.world_seed, r, t, 41377))
                for k, c in options.items():
                    values[k].append(rollout(env, spec, node, independent, limits, c, histories))
            # Choice for both policies is locked before computing any actual-tape comparisons.
            matches = {}
            cache[key] = values, matches
        decisions = {mode: best_branch(values, mode) for mode in POLICIES}
        choice = decisions[rule]
        comparisons = [*decisions.values(), "continue"]
        if "reference" in options:
            comparisons.append("reference")
        for k in dict.fromkeys(comparisons):
            if k not in matches:
                matches[k] = rollout(env, spec, node, trial, limits, options[k], histories, masked=True)
        estimates = {k:[sum(o[f] for o in v) for f in ("strict", "correct", "wrong","composite")]
                     for k,v in values.items()}
        log.append(dict(region=r, station=int(node.station), tick=int(node.tick), word=choice, also=decisions,
                        estimates=estimates, actual_if_then_mpc=matches.copy()))
        start = len(node.reader.frames)
        apply(env, spec, node, trial, limits, options[choice])
        if choice != "continue":
            spans.append((start,len(node.reader.frames)))
        previous.append(choice)
    scan.ref.finish(env, node, spec, trial, limits)
    last = node.reader.frames[-1]
    assert abs(last[0]-node.station) < 1.e-7 and abs((last[1]-node.sector+12.)%24.-12.) < 1.e-7
    full = grade(node.reader.frames, histories)
    mask = np.ones(len(node.reader.frames), dtype=bool)
    for i,j in spans:
        mask[i:j] = False
    full["masked"] = grade([f for f,take in zip(node.reader.frames,mask) if take], scan.public_context(env)[0])
    full.update(ticks=int(node.tick), distance=float(node.distance))
    return dict(terminal=full, events=log)


def work(item):
    case, draws, destination = item
    env = scan.ref.reference.load_environment(str(scan.ATLAS), case.shape)
    spec = spec_for(env.scanner, case)
    initial, trial, limits = env.start(case, spec)
    views = nominal_fields(env), view_fields(env, trial.layout.centers)
    first, end = spec[1:3]
    areas = [scan.MOTION.region_of(spec, s) for s in range(first,end+1)]
    checks = []
    for region in range(6):
        indices = np.flatnonzero(np.array(areas) == region)
        checks.append(int(first+indices[(len(indices)-10)//2]))
    baseline = scan.ref.finish(env, initial.fork(), spec, trial, limits)
    base = grade(baseline.reader.frames, scan.public_context(env)[0])
    base.update(ticks=int(baseline.tick), distance=float(baseline.distance))
    row = dict(case=case.identifier, world=int(case.world_seed), profile=case.profile, shape=int(case.shape),
               layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes() + bytes([case.shape])).hexdigest(),
               mpc=base)
    cache = {}
    for rule in POLICIES:
        row[rule] = episode(case, env, spec, initial, trial, limits, views, checks, rule, draws, cache)
    (destination/f"{case.world_seed}.json").write_text(json.dumps(row,separators=(",",":")))
    return case.identifier


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--worlds",type=int,default=32)
    p.add_argument("--draws",type=int,default=8)
    p.add_argument("--workers",type=int,default=12)
    args=p.parse_args()
    args.directory.mkdir(exist_ok=True,parents=True)
    jobs=[(c,args.draws,args.directory) for c in scan.cases("test",args.worlds,SEED)
          if not (args.directory/f"{c.world_seed}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for i,case in enumerate(pool.map(work,jobs)):
            print(i+1, "/", len(jobs),case,flush=True)

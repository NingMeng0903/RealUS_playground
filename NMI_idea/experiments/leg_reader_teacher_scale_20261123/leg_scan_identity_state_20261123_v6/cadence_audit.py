"""Measure the v5 intake-cadence effect on one fixed physical trajectory."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from v6_data import DEFAULT_PACK, ROOT, ROUTES, atomic_json, cases, init_worlds, read_json
from anchor_policy import Session
from checks_v6 import rebuild
from forward_anchor import find_anchor
from readout_state import confidence, evaluate
from reader_study import frozen_reader, warm_bundle
from route_metrics import paired, totals
from study_data import motion, start
from study_worlds import truth_at
import numpy as np


_READERS = None


def prepare(atlas, readers):
    global _READERS
    init_worlds(atlas)
    _READERS = readers


def one(case):
    env, spec, node, tape, limits = start(case)
    anchor = find_anchor(env, spec, route=case.task, prefix_stations=3 + case.band % 4)
    route_boundaries, intervention = [], []
    while node.station < spec[2]:
        region = motion.region_of(spec, node.station)
        env.route.advance(node, spec, tape, limits)
        if motion.region_of(spec, node.station) != region:
            route_boundaries.append(len(node.reader.frames))
        if anchor.prefix_station <= node.station <= anchor.station:
            intervention.append(len(node.reader.frames))
    end = len(node.reader.frames)
    schedules = {"complete_only": [end], "region_boundaries": sorted(set([*route_boundaries, end])),
                 "anchor_stations_then_regions": sorted(set([*intervention, *(i for i in route_boundaries if i > intervention[-1]), end]))}
    f = _READERS[case.task]["v5"]
    posterior, available = {}, {}
    for key, schedule in schedules.items():
        session = Session(env, spec, f)
        for cut in schedule:
            public, state = session.update(rebuild(node, cut))
        values = confidence(public, state, f["risk"])
        labels = truth_at(node, public, tape)["labels"]
        available[key] = evaluate(public, labels, values, f["threshold"], complete=True)
        posterior[key] = state["unary"][:, 0]
    baseline = posterior["complete_only"]
    affected = {key: {"candidates_changed_gt_0_01": int((abs(p - baseline) > .01).sum()),
                     "largest_target_probability_change": float(abs(p - baseline).max(initial=0.))} for key,p in posterior.items()}
    return {"task": case.task, "case": case.identifier, "layout": tape.layout.identifier, "raw_candidate_count": len(baseline),
            "scores": available, "schedule_points": schedules, "changed": affected}


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", default=ROOT / "run/confirmed", type=Path)
    p.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    p.add_argument("--workers", default=4, type=int)
    args = p.parse_args()
    readers = frozen_reader(warm_bundle(args.out, args.atlas_pack), read_json(args.out / "reader_freeze.json"))
    quota = {r: 16 for r in ROUTES}
    cohort = []
    for case in cases("reader_fresh"):
        if quota[case.task]:
            quota[case.task] -= 1
            cohort.append(case)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=prepare,
                             initargs=(args.atlas_pack, readers)) as pool:
        rows = list(pool.map(one, cohort, chunksize=1))
    result = {}
    for route in ROUTES:
        selected = [r for r in rows if r["task"] == route]
        ids = [r["layout"] for r in selected]
        reference = [r["scores"]["complete_only"] for r in selected]
        scopes = {}
        for scope in selected[0]["scores"]:
            arms = [r["scores"][scope] for r in selected]
            scopes[scope] = totals(arms) | {"vs_once": paired(arms, reference, ids, 630_711_554),
                "changed_candidates": sum(r["changed"][scope]["candidates_changed_gt_0_01"] for r in selected),
                "worst_probability_delta": max(r["changed"][scope]["largest_target_probability_change"] for r in selected)}
        result[route] = {"layouts": len(selected), "candidates": sum(r["raw_candidate_count"] for r in selected), "scores": scopes}
    atomic_json(args.out / "cadence_audit.json", {"program": "ordinary MPC fixed path, v5 probability intake schedules",
        "layout_scope": "first 16 per route from the v6 fixed-track reader confirmation", "routes": result, "per_layout": rows})
    for name, data in result.items():
        print(name, {k: (v["strict"], v["wrong_branch"], v["changed_candidates"]) for k,v in data["scores"].items()}, flush=True)


if __name__ == "__main__":
    main()

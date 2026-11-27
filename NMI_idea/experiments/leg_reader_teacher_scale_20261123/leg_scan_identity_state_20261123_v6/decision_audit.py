"""Compare successive public evidence states while holding forecast geometry fixed."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

from v6_data import DEFAULT_PACK, ROOT, ROUTES, atomic_json, cases, read_json
from anchor_experiment import alternatives
from anchor_logic import beliefs, rank
from anchor_policy import view_predictions
from causal_reader import PrefixReader
from forward_anchor import step_slot
from live_programs import init_live, prefix_setup
from reader_study import frozen_reader, warm_bundle
import numpy as np


def previous_mass(old_input, old_state, public, state, track_ids):
    keys = {(int(f), int(s)): i for i, (f, s) in enumerate(zip(old_input["frames"], old_input["slots"], strict=True))}
    roles, old_tracks = [], []
    for track in track_ids:
        current = np.flatnonzero(state["track_ids"] == track)
        lookup = [keys[(int(public["frames"][i]), int(public["slots"][i]))] for i in current
                  if (int(public["frames"][i]), int(public["slots"][i])) in keys]
        if not lookup:
            return None
        candidate = max(lookup, key=lambda i: (old_input["frames"][i], old_state["usable"][i]))
        roles.append(float(old_state["object_role"][candidate, 0]))
        old_tracks.append(old_state["track_ids"][candidate])
    if len(set(old_tracks)) < 2 or sum(roles) <= 0.:
        return None
    return np.asarray(roles) / sum(roles)


def audit(case):
    ctx = prefix_setup(case)
    env, node, spec = (ctx[k] for k in ("env", "node", "spec"))
    reader = PrefixReader(env, spec, ctx["reader"])
    trace, previous = [], None
    if ctx["trigger"]:
        while node.station < ctx["anchor"].station:
            public, states = reader.update(node); state = states["objects"]
            b = beliefs(public, state, node.station)
            sectors, legal, base = alternatives(env, spec, node, ctx["limits"], ctx["anchor"], ctx["clock"], True)
            choice = int(np.argmax(base))
            if b is not None:
                d,s = view_predictions(env.scanner.surface, node.station + 1, sectors, b["points"])
                values, _ = rank(base, legal, d, s, b["weights"])
                choice = int(np.argmax(values))
                last = previous_mass(*previous, public, state, b["tracks"]) if previous is not None else None
                if last is not None:
                    previous_values, _ = rank(base, legal, d, s, last)
                    prior_choice = int(np.argmax(previous_values))
                    new = public["frames"] > previous[0]["frames"].max(initial=-1)
                    attached = int((new & np.isin(state["track_ids"], b["tracks"])).sum())
                    trace.append({"station": int(node.station), "tick": int(node.tick), "same_forecast_changed_slot": choice != prior_choice,
                                  "previous_slot": prior_choice, "actual_slot": choice, "existing_object_weights": last.tolist(),
                                  "new_object_weights": b["weights"].tolist(), "new_attached_candidates": attached})
            previous = (public, state)
            step_slot(env, spec, node, ctx["tape"], ctx["limits"], choice)
    return {"route": case.task, "case": case.identifier, "layout_id": ctx["tape"].layout.identifier,
            "triggered": ctx["trigger"], "decisions": trace}


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", default=ROOT / "run/confirmed", type=Path)
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", default=4, type=int)
    args = p.parse_args()
    freeze = read_json(args.out / "reader_freeze.json")
    reader = frozen_reader(warm_bundle(args.out, args.atlas_pack), freeze)
    cohort, quota = [], {route: 16 for route in ROUTES}
    for case in cases("closed_test"):
        if quota[case.task]:
            cohort.append(case)
            quota[case.task] -= 1
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                             initializer=init_live, initargs=(args.atlas_pack, reader)) as pool:
        rows = list(pool.map(audit, cohort, chunksize=1))
    by_route = {}
    for task in ROUTES:
        selection = [v for v in rows if v["route"] == task]
        decisions = [d for v in selection for d in v["decisions"]]
        by_route[task] = {"layouts": len(selection), "triggered": sum(v["triggered"] for v in selection),
                          "comparable_same_geometry": len(decisions), "new_evidence_changes_slot":
                          sum(d["same_forecast_changed_slot"] for d in decisions),
                          "changed_with_attached_new_candidates":
                          sum(d["same_forecast_changed_slot"] and d["new_attached_candidates"] > 0 for d in decisions)}
    atomic_json(args.out / "decision_audit.json", {"scope": "fixed first 16 held-out layouts per route, no policy selection",
        "temper_same_geometry": "same current forecasts, swap only previous object weights on overlapping frame-slot histories",
        "routes": by_route, "observations": rows})
    print(by_route, flush=True)


if __name__ == "__main__":
    main()

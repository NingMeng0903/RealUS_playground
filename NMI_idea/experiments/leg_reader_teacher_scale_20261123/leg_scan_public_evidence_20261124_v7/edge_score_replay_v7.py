"""Audit endpoint scores with and without new cross-window links on identical frames."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import multiprocessing as mp

from study_v7 import DEFAULT_PACK, ROOT, ROUTES, atomic_json, digest, pinned, read_json, roster
import motion_v7 as sim
from live_programs import prefix_setup
import numpy as np


def measure(ctx, node, reader):
    candidates = {}
    goal_builder = sim.make_goal

    def collect(context, current, scores, ids=()):
        candidates.update(scores)
        return goal_builder(context, current, scores, ids)

    with patch.object(sim, "make_goal", collect):
        goal, forecast = sim.propose(ctx, node, reader)
    return candidates, goal.as_dict() if goal else None, forecast


def replay(case):
    ctx = prefix_setup(case)
    if not ctx["trigger"]:
        return None
    node = ctx["node"].fork()
    base = sim.make_reader(ctx, False)
    with_links = sim.make_reader(ctx, True)
    goal, expectation = sim.propose(ctx, node, base)
    if goal is None:
        return None
    visit = sim.drive_goal(ctx, node, ctx["tape"], base, replan=False, goal=goal, estimate=expectation)
    x, states_off = base.update(node)
    y, states_on = with_links.update(node)
    for key in x:
        np.testing.assert_array_equal(x[key], y[key])
    off, plan_off, forecast_off = measure(ctx, node, base)
    on, plan_on, forecast_on = measure(ctx, node, with_links)
    shared = off.keys() & on.keys()
    ranks_off = sorted(shared, key=lambda key: (-off[key], key))
    ranks_on = sorted(shared, key=lambda key: (-on[key], key))
    role_change = np.max(np.abs(states_off["objects"]["object_role"] - states_on["objects"]["object_role"]), initial=0.)
    return {"route": case.task, "layout_id": ctx["tape"].layout.identifier, "pose": visit["trace"][-1]["pose"],
            "same_features": True, "role_change_max": float(role_change),
            "new_edges": len(states_on["objects"]["edges"]) - len(states_off["objects"]["edges"]),
            "max_common_endpoint_score_change": float(max((abs(off[key] - on[key]) for key in shared), default=0.)),
            "common_rank_changed": ranks_off != ranks_on,
            "endpoint_scores": [[*key, off.get(key), on.get(key)] for key in sorted(off.keys() | on.keys())],
            "plan_off": plan_off, "plan_on": plan_on, "forecast_off": forecast_off, "forecast_on": forecast_on}


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    protocol_path = a.out / "mechanism_protocol.json"
    protocol = read_json(protocol_path)
    models = {name: a.out / f"{name}_model.json" for name in ("observation", "bridge")}
    if (any(digest(path) != protocol["models"][name] for name, path in models.items())
            or any(digest(ROOT / name) != sha for name, sha in protocol["source_sha256"].items())):
        raise RuntimeError("Replay inputs differ from the frozen same-frame experiment")
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), initializer=sim.prepare,
            initargs=(a.atlas_pack, pinned(a.atlas_pack), read_json(models["observation"]), read_json(models["bridge"]))) as pool:
        rows = [r for r in pool.map(replay, roster("mechanism_test")) if r is not None]
    routes = {}
    for route in ROUTES:
        group = [r for r in rows if r["route"] == route]
        routes[route] = {"fixed_prefixes": len(group), "cases_with_added_edges": sum(r["new_edges"] > 0 for r in group),
                         "role_changed_cases": sum(r["role_change_max"] > 1.e-12 for r in group),
                         "score_changed_cases": sum(r["max_common_endpoint_score_change"] > 1.e-12 for r in group),
                         "rank_changed_cases": sum(r["common_rank_changed"] for r in group)}
    result = {"parent_protocol_sha256": digest(protocol_path), "source_sha256": digest(Path(__file__)),
              "summary": routes, "per_case": rows}
    atomic_json(a.out / "edge_score_replay.json", result)
    print(routes, flush=True)


if __name__ == "__main__":
    main()

"""A fresh same-track check of reliability-weighted association messages."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import multiprocessing as mp
from pathlib import Path

from study_worlds import (DEFAULT_PACK, READER_SCANS, ROOT, ROUTES, SHARD, atomic_json, cases_for, commit_shard,
                          complete, completed, digest, init_worlds, load_shard, pack_acquired, read_json, stash_public,
                          start)
import numpy as np
from fit_public import load_public
from guarded_association import LINK_THRESHOLD, edge_features, fit_edge_confidence, guard
from reader_eval import predictions
from readout_state import METHODS, candidates, confidence, evaluate, fit_risk
from route_metrics import paired, totals


FRESH_COUNT = 512
FRESH_BASE = 1_989_777_000_000_000


def acquire_terminal(case):
    env, spec, node, trial, limits = start(case)
    complete(env, spec, node, trial, limits)
    snapshot = stash_public(env, spec, node, trial, "terminal", len(node.reader.frames), include_legacy=True)
    return [snapshot], {"case": asdict(case), "layout_id": trial.layout.identifier, "complete": True,
                        "histories": [snapshot[1]]}


def read_edge_examples(paths, models, links):
    x, y = [], []
    for _, hist, public, states, _, physical in predictions(paths, models, links):
        if hist["scope"] not in ("after_probe", "terminal"):
            continue
        graph = states["fixed_graph"]
        edges = graph["edges"]
        left, right = edges[:, :2].astype(int).T
        x.append(edge_features(public, graph["unary"], graph["embedding"], edges))
        y.append((physical[left] >= 0) & (physical[left] == physical[right]))
    return np.concatenate(x), np.concatenate(y)


def fit_delivery(paths, models, links, edges):
    fit = defaultdict(list)
    for case, hist, public, states, labels, _ in predictions(paths, models, links):
        if hist["scope"] not in ("before", "terminal"):
            continue
        target, x = candidates(public, guard(public, states["fixed_graph"], links, edges))
        fit[case["case"]["task"]].append((x, (labels[target] == 0).astype(np.float32)))
    return {route: fit_risk(fit[route]) for route in ROUTES}


def fresh_roster():
    return [replace(case, identifier=f"v4_link_confirmation_{i:05d}", split="link_confirmation",
                    world_seed=FRESH_BASE + i * 7, tape_seed=FRESH_BASE + i * 7 + 1, action_seed=FRESH_BASE + i * 7 + 2)
            for i, case in enumerate(cases_for("reader_test", FRESH_COUNT))]


def confirm(paths, models, links, frozen, edges, fit):
    data = defaultdict(list)
    for case, _, public, states, labels, physical in predictions(paths, models, links):
        route = case["case"]["task"]
        states["guarded_graph"] = guard(public, states["fixed_graph"], links, edges)
        risk = frozen["risk"][route] | {"guarded_graph": fit[route]}
        scores = {method: confidence(public, state, risk.get(method)) for method, state in states.items()}
        prior = frozen["thresholds_and_reader"][route]["variants"]["v3_scaled"]["threshold"]
        count = evaluate(public, labels, scores["v3_scaled"], prior, complete=True)["submitted"]
        paired_scores = {name: evaluate(public, labels, values, 0., complete=True, submit_count=count)
                         for name, values in scores.items()}
        if any(row["submitted"] != count for row in paired_scores.values()):
            raise RuntimeError("The fresh graph ablation must match submissions")
        trace = {}
        for name in ("fixed_graph", "guarded_graph"):
            a, b = states[name]["edges"][:, :2].astype(int).T
            positive = (physical[a] >= 0) & (physical[a] == physical[b])
            trace[name] = (int(positive.sum()), len(a))
        data[route].append((case["layout_id"], paired_scores, trace))
    output = {}
    for route, rows in data.items():
        keys = [k for k, _, _ in rows]
        arms = {name: [score[name] for _, score, _ in rows] for name in (*METHODS, "guarded_graph")}
        output[route] = {name: totals(arm) | {"paired_vs_v3": paired(arm, arms["v3_scaled"], keys, 9127),
                                            "paired_vs_unary": paired(arm, arms["fixed_unary"], keys, 9127),
                                            "role4_region3": int(sum(x["selected_ids"][2] == 4 for x in arm))}
                         for name, arm in arms.items()}
        output[route]["links"] = {name: np.asarray([t[name] for _, _, t in rows]).sum(axis=0).tolist()
                                   for name in ("fixed_graph", "guarded_graph")}
    return output


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    out = args.out
    reader = out / "reader_freeze.json"
    f = read_json(reader)
    models = {"v3_scaled": load_public(f["prior_reader"], "cuda:0"),
              "updated": load_public(f["updated"]["weights"], "cuda:0")}
    with np.load(args.atlas_pack, allow_pickle=False) as pack:
        links = pack["links"]
    cal, _ = completed(out, "calibrate", READER_SCANS["calibrate"], SHARD)
    dev, _ = completed(out, "select_reader", READER_SCANS["select_reader"], SHARD)
    edge_model = fit_edge_confidence(*read_edge_examples(cal, models, links))
    risk = fit_delivery(dev, models, links, edge_model)
    lock = out / "link_guard_freeze.json"
    atomic_json(lock, {"reader_sha256": digest(reader), "edge_model": edge_model, "risk": risk, "threshold": LINK_THRESHOLD,
                       "new_natural_evaluation_cases": FRESH_COUNT, "evaluation_seed_base": FRESH_BASE,
                       "source_sha256": {p.name: digest(p) for p in (ROOT / "confirm_link_guard.py", ROOT / "guarded_association.py")}})
    roster = fresh_roster()
    paths = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=init_worlds,
                             initargs=(args.atlas_pack,)) as pool:
        for begin in range(0, len(roster), SHARD):
            batch = roster[begin:begin + SHARD]
            folder = out / "link_confirmation" / f"{begin // SHARD:05d}"
            if not (folder / "commit.json").exists():
                entries = list(pool.map(acquire_terminal, batch, chunksize=1))
                commit_shard(folder, pack_acquired(entries), [m for _, m in entries])
            file, cases = load_shard(folder)
            if [m["case"] for m in cases] != [asdict(c) for c in batch]:
                raise RuntimeError("Fresh guard cases do not match the frozen schedule")
            paths.append(file)
            print(f"Link guard new paired scans {min(begin + SHARD, len(roster))}/{len(roster)}", flush=True)
    result = confirm(paths, models, links, f, edge_model, risk)
    atomic_json(out / "link_guard_results.json", {"frozen_sha256": digest(lock), "routes": result,
                                                  "scope": "fresh same-track confirmation, independent of v4 graph error audit"})
    for route, report in result.items():
        print(route, {k: (a["strict"], a["wrong_region_count"], a["wrong_branch"]) for k, a in report.items() if k != "links"}, flush=True)


if __name__ == "__main__":
    main()

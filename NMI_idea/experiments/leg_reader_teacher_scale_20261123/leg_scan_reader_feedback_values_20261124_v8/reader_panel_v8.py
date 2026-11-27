"""Fit supervised public readout and freeze it before scoring new fixed tracks."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
import multiprocessing as mp
from pathlib import Path

from study_v8 import (BASE, DEFAULT_PACK, ROOT, ROUTES, V4_RUN, V6_RUN, atomic_json, check_hashes, committed,
                      digest, freeze, hashes, pinned, read_json, verify_shard)
from causal_reader import PUBLIC_FIELDS, inputs
from reader_study import warm_bundle
from reader_v8 import MODES, decoder, fit_delivery, public_states
from readout_state import evaluate
from route_metrics import paired, totals
import numpy as np


_BASE = _EDGE = None


def init(base, edge):
    global _BASE, _EDGE
    _BASE, _EDGE = base, edge


def snapshots(item):
    path, development = item
    _, entries = verify_shard(path.parent)
    with np.load(path, allow_pickle=False) as archive:
        z = {key: archive[key] for key in (*PUBLIC_FIELDS, "labels", "instance", "offsets")}
    descriptors = [(r, history) for r in entries for history in r["histories"]]
    result = []
    for (a, b), (r, history) in zip(pairwise(z["offsets"]), descriptors, strict=True):
        fold = (int(r["case"]["identifier"].rsplit("_", 1)[1]) // 2) % 4
        if history["scope"] != "terminal" and (not development or fold == 3 or history["scope"] not in ("before", "after_probe")):
            continue
        pub = inputs({k: z[k][a:b] for k in PUBLIC_FIELDS})
        states = public_states(pub, _BASE, _BASE["links"], _EDGE)
        labels, instance = z["labels"][a:b], z["instance"][a:b]
        edges = {}
        for name in ("old", "new"):
            s = states[name]
            first, second = s["edges"][:, :2].astype(int).T
            edges[name] = [int(np.sum((instance[first] >= 0) & (instance[first] == instance[second]))), len(first)]
            for key in ("scored_edges", "edges", "stats", "track_sources"):
                s.pop(key, None)
        result.append({"public": {k: value.copy() for k, value in pub.items() if k != "features"},
                       "states": states, "labels": labels.copy(), "layout": r["layout_id"], "fold": fold,
                       "route": r["case"]["task"], "scope": history["scope"], "edge_counts": edges})
    return result


def grade(row, mode, config, *, threshold=None, count=None):
    x = decoder(row["public"], row["states"], mode, config)
    cut = config["threshold"] if threshold is None else threshold
    return evaluate(row["public"], row["labels"], x, cut, complete=True, submit_count=count)


def threshold_for(rows, mode, config, reference):
    pairs = [(decoder(r["public"], r["states"], mode, config), r["labels"], r["public"]["regions"]) for r in rows]
    peak = np.full((len(rows), 6), -1.)
    kinds = np.full((len(rows), 6), -99)
    for i, (score, label, region) in enumerate(pairs):
        ix = np.flatnonzero(score >= 0.)
        peak[i, region[ix]], kinds[i, region[ix]] = score[ix], label[ix]
    coverage = sum(x["submitted"] for x in reference)
    wrong = sum(x["wrong_branch"] for x in reference)
    trials = []
    for cut in np.unique(np.r_[0., peak[peak >= 0], 1.]):
        accepted = peak >= cut
        submitted = int(accepted.sum())
        errors = int((accepted & (kinds != 0)).sum())
        wrong_branch = int(np.any(accepted & (kinds > 0), axis=1).sum())
        strict = int(np.all(accepted & (kinds == 0), axis=1).sum())
        allowed = abs(coverage-submitted) <= max(4, .035*coverage) and wrong_branch <= wrong
        trials.append((allowed, submitted, errors, wrong_branch, strict, float(cut)))
    valid = [x for x in trials if x[0]]
    chosen = max(valid, key=lambda x: (x[4], -x[2], -abs(x[1]-coverage))) if valid else min(trials, key=lambda x: (abs(x[1]-coverage), x[3], x[2]))
    return dict(zip(("admissible", "submitted", "wrong_regions", "wrong_branch", "strict", "threshold"), chosen, strict=True))


def fit(rows, old):
    result = {}
    for route in ROUTES:
        deck = [r for r in rows if r["route"] == route]
        threshold = [r for r in deck if r["fold"] == 3]
        protocol = {"unary": old[route]["unary"]}
        unary = [grade(r, "unary", protocol["unary"]) for r in threshold]
        ref = totals(unary)
        metrics = {"unary": {"fixed": ref, "matched": ref, "admissible": True}}
        for mode in MODES[1:]:
            cfg = fit_delivery(deck, mode)
            setting = threshold_for(threshold, mode, cfg, unary)
            cfg["threshold"] = setting["threshold"]
            protocol[mode] = cfg
            fixed = totals([grade(r, mode, cfg) for r in threshold])
            matched = totals([grade(r, mode, cfg, threshold=0., count=b["submitted"]) for r,b in zip(threshold, unary, strict=True)])
            metrics[mode] = {"fixed": fixed, "matched": matched, "admissible": setting["admissible"]}
        approved = [mode for mode in MODES[1:] if metrics[mode]["admissible"] and all(
            m["wrong_region_count"] <= ref["wrong_region_count"] and m["wrong_branch"] <= ref["wrong_branch"] and m["strict"] >= ref["strict"]
            for m in (metrics[mode]["fixed"], metrics[mode]["matched"])) and any(
            metrics[mode]["matched"][k] != ref[k] for k in ("strict", "wrong_region_count", "wrong_branch"))]
        mode = max(approved, default="unary", key=lambda a:(metrics[a]["fixed"]["strict"], -metrics[a]["fixed"]["wrong_region_count"],
                                                              metrics[a]["matched"]["strict"]))
        result[route] = {"delivery": mode, "config": protocol, "development": metrics,
                         "layout_counts": {i: len({r["layout"] for r in deck if r["fold"] == i}) for i in range(4)}}
    return result


def regions(rows):
    capture = np.array([r["capture"] for r in rows]); label = np.array([r["selected_ids"] for r in rows]); result = []
    for i in range(6):
        seen, y = capture[:, i], label[:, i]
        result.append({"region":i+1, "captured": int(seen.sum()), "correct": int((y == 0).sum()),
                       "wrong_branch": int((y > 0).sum()), "artifact": int(((y < 0) & (y != -99)).sum()),
                       "seen_but_abstained": int((seen & (y == -99)).sum()), "never_captured": int((~seen).sum())})
    return result


def evaluate_fixed(rows, bundle):
    result = {}
    for route in ROUTES:
        deck = [r for r in rows if r["route"] == route and r["scope"] == "terminal"]
        ids = [r["layout"] for r in deck]
        configs = bundle[route]["config"]
        base = [grade(r, "unary", configs["unary"]) for r in deck]
        scores = {k: {} for k in ("fixed", "matched")}
        records = {r["layout"]: {} for r in deck}
        for mode, cfg in configs.items():
            for protocol, subset in (("fixed", [grade(r,mode,cfg) for r in deck]), ("matched", [
                    grade(r,mode,cfg,threshold=0.,count=base[i]["submitted"]) for i,r in enumerate(deck)])):
                if protocol == "matched" and any(a["submitted"] != b["submitted"] for a,b in zip(subset,base,strict=True)):
                    raise RuntimeError("Fixed-trajectory submission counts mismatch")
                scores[protocol][mode] = totals(subset) | {"vs_unary": paired(subset,base,ids,866_044_330),"by_region": regions(subset)}
                for tag, v in zip(ids,subset,strict=True):
                    records[tag][f"{protocol}_{mode}"] = v
        edges = {name: np.sum([r["edge_counts"][name] for r in deck],axis=0).tolist() for name in ("old","new")}
        result[route] = {"fixed": scores["fixed"], "matched": scores["matched"], "cases": len(deck),
                         "decoder_frozen_on_development": bundle[route]["delivery"], "kept_true_total_edges": edges, "pairs": records}
    return result


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--edge-bundle",type=Path, default=None)
    args = p.parse_args()
    edge_path = args.edge_bundle or args.out/"edge_model_v8.json"
    edge_sha = digest(edge_path)
    edge = read_json(edge_path)
    pinned(args.atlas_pack)
    base = warm_bundle(V6_RUN,args.atlas_pack)
    old = read_json(V6_RUN/"reader_freeze.json")["calibration"]
    dev = sorted((V4_RUN/"select_reader").glob("*/data.npz"))
    source = hashes(("study_v8.py","reader_v8.py","reader_panel_v8.py","edge_v8.py"))
    rule = {"base_git": BASE, "sources": source,"atlas_sha256":digest(args.atlas_pack),"edge_bundle_sha256":edge_sha,
            "development_shards": [digest(i) for i in dev], "capture_protocol_sha256":digest(args.out/"capture_protocol.json"),
            "development_folds": ["object identity", "candidate delivery", "winner reliability", "threshold and route selection"],
            "oracle_used_in_deployed_fit":False}
    lock = freeze(args.out,"reader_protocol.json",rule)
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context("spawn"),initializer=init,initargs=(base,edge)) as pool:
        saved = args.out/"reader_bundle.json"
        if saved.exists():
            settings = read_json(saved)
            if settings["protocol_sha256"] != lock:
                raise RuntimeError("Frozen reader belongs to another experiment")
        else:
            rows = [r for group in pool.map(snapshots,((i,True) for i in dev),chunksize=1) for r in group]
            chosen = fit(rows,old)
            settings = {"protocol_sha256":lock,"edge_bundle_sha256":edge_sha,"routes":chosen}
            check_hashes(source)
            if digest(edge_path)!=edge_sha:
                raise RuntimeError("Edge bundle changed during readout fitting")
            atomic_json(saved,settings)
            del rows
            print("Development froze",{r:chosen[r]["delivery"] for r in ROUTES},flush=True)
        committed(args.out,"reader_test")
        heldout = sorted((args.out/"reader_test").glob("*/data.npz"))
        rows = [r for group in pool.map(snapshots,((i,False) for i in heldout),chunksize=1) for r in group]
    check_hashes(source)
    if digest(edge_path)!=edge_sha:
        raise RuntimeError("Edge bundle changed during held-out comparison")
    result = evaluate_fixed(rows,settings["routes"])
    atomic_json(args.out/"reader_results.json",{"frozen_reader_sha256":digest(saved),"rules":lock,"routes":result})
    for route, b in result.items():
        print(route,b["decoder_frozen_on_development"], {name:{k:d[k] for k in ("strict","wrong_branch","wrong_region_count")} for name,d in b["matched"].items()},flush=True)


if __name__ == "__main__":
    main()

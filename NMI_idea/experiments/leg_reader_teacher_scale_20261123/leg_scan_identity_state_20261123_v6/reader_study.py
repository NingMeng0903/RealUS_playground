"""Fit delivery calibration on archived development layouts and compare fixed tracks."""

from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
import multiprocessing as mp
from pathlib import Path

from v6_data import (BASE, DEFAULT_PACK, ROOT, ROUTES, STAGES, V4_RUN, V5_RUN, atomic_json, collect, current_scan,
                     digest, init_worlds, load_shard, pack_acquired, read_json)
from causal_reader import JOINT_FIELDS, METHODS, PUBLIC_FIELDS, choose, likelihood, model_state, observation_features, scores
from fit_public import load_public
from readout_state import evaluate, export_cpu, fit_risk
from route_metrics import paired, totals
import numpy as np


LIVE = METHODS[:4]
_BASE = None


def warm_bundle(out, atlas):
    prior = read_json(V5_RUN / "reader_freeze.json")
    checkpoint = Path(prior["fit"]["weights"])
    if digest(checkpoint) != prior["fit"]["sha256"]:
        raise RuntimeError("The pinned v5 association model failed verification")
    with np.load(atlas, allow_pickle=False) as pack:
        links = pack["links"]
    usability_path = out / "usability_bundle.json"
    return {"model": export_cpu(load_public(checkpoint)), "links": links,
            "edge_model": prior["edge_models"]["prematch"], "usability": read_json(usability_path), "v5": prior}


def init_panel(bundle):
    global _BASE
    _BASE = bundle


def scope_for(stage):
    return {"calibrate": ("before", "terminal"),
            "evaluate": ("before", "after_probe", "after_hold", "terminal")}[stage]


def load_panel(job):
    path, stage = job
    _, cases = load_shard(path.parent)
    result = []
    with np.load(path, allow_pickle=False) as z:
        columns = {k: z[k] for k in (*PUBLIC_FIELDS, "labels", "instance", "offsets")}
        descriptions = [(row, h) for row in cases for h in row["histories"]]
        for (lo, hi), (row, desc) in zip(pairwise(columns["offsets"]), descriptions, strict=True):
            if desc["scope"] not in scope_for(stage):
                continue
            ordinal = int(row["case"]["identifier"].rsplit("_", 1)[1]); fold = (ordinal // 2) % 3
            if stage == "calibrate" and fold == 2 and desc["scope"] != "terminal":
                continue
            public = {k: columns[k][lo:hi] for k in PUBLIC_FIELDS}
            truth, physical = (columns[k][lo:hi] for k in ("labels", "instance"))
            states = model_state(public, _BASE["model"], _BASE["links"], _BASE["edge_model"], _BASE["usability"],
                                 physical=physical if stage == "evaluate" or fold != 0 else None)
            compact = {}
            for method, state in states.items():
                keys = ("probabilities", "track_ids", "support") if "probabilities" in state else (
                    "role", "object_role", "local_role", "usable", "view_counts", "unary", "track_ids", "support")
                a, b = state["edges"][:, :2].astype(int).T
                edges = [int(np.sum((physical[a] >= 0) & (physical[a] == physical[b]))), len(a)]
                compact[method] = {key: state[key] for key in keys} | {"edge_counts": edges}
            result.append({"public": {k: v.copy() for k, v in public.items() if k != "features"},
                           "states": compact, "labels": truth.copy(), "desc": desc, "fold": fold,
                           "layout": row["layout_id"], "case": row["case"]["identifier"],
                           "route": row["case"]["task"], "region": row["probe_in_region"]})
    return result


def sample_panels(pool, paths, stage):
    for rows in pool.map(load_panel, ((path, stage) for path in paths), chunksize=1):
        yield from rows


def fit_joint(data, mode):
    sampler = np.random.default_rng(22_611_227)
    obs = []
    for v in data:
        if v["fold"] != 0:
            continue
        x = observation_features(v["public"], v["states"][mode], object_role=mode == "objects")
        y = v["labels"] == 0
        pick = np.arange(len(y)) if len(y) <= 800 else sampler.choice(len(y), 800, replace=False)
        if len(pick):
            obs.append((x[pick], y[pick]))
    return fit_risk(obs) | {"field_order": JOINT_FIELDS}


def calibrate(rows, reference):
    rules, decisions = {}, {}
    for route in ROUTES:
        deck = [r for r in rows if r["route"] == route]
        joined = {key: fit_joint(deck, key) for key in ("role_only", "objects")}
        appended = {"role_oracle": "role_only", "object_oracle": "objects"}
        config = {}
        for mode in METHODS:
            joint = joined.get(appended.get(mode, mode))
            points = []
            for row in deck:
                if row["fold"] == 1:
                    picks, _, feature = choose(row["public"], row["states"][mode], mode, joint)
                    if len(picks):
                        points.append((feature, row["labels"][picks] == 0))
            config[mode] = {"joint": joint, "winner": fit_risk(points)}
        test = [r for r in deck if r["fold"] == 2]
        ref_cutoff = reference["thresholds"][route]["methods"]["unary"]["threshold"]
        reference_rows = [evaluate(r["public"], r["labels"], scores(r["public"], r["states"], "unary",
                          {"winner": reference["risk"][route]["unary"]}), ref_cutoff, complete=True) for r in test]
        accepted = sum(r["submitted"] for r in reference_rows)
        branches = sum(r["wrong_branch"] for r in reference_rows)
        comparisons = {}
        for mode in METHODS:
            joined_scores, counts = [], []
            for entry, ref in zip(test, reference_rows, strict=True):
                public, labs, states = entry["public"], entry["labels"], entry["states"]
                values = scores(public, states, mode, config[mode])
                matched = evaluate(public, labs, values, 0., complete=True, submit_count=ref["submitted"])
                counts.append(matched)
                best = np.full(6, -1.)
                label = np.full(6, -99, dtype=int)
                ix = np.flatnonzero(values >= 0.)
                best[public["regions"][ix]], label[public["regions"][ix]] = values[ix], labs[ix]
                joined_scores.append((best, label))
            best, label = (np.stack(v) for v in zip(*joined_scores, strict=True))
            thresholds = np.unique(np.r_[0., np.quantile(best[best >= 0], np.linspace(0, 1, 151)), 1.])
            options = []
            for threshold in thresholds:
                submitted = best >= threshold
                count = int(submitted.sum())
                mistakes = int((submitted & (label != 0)).sum())
                wrong = int((submitted & (label > 0)).any(axis=1).sum())
                strict = int((submitted & (label == 0)).all(axis=1).sum())
                admissible = abs(count - accepted) <= max(4, .035 * accepted) and wrong <= branches
                options.append((bool(admissible), count, mistakes, wrong, strict, float(threshold)))
            eligible = [item for item in options if item[0]]
            choice = (max(eligible, key=lambda a: (a[4], -a[2], -abs(a[1] - accepted))) if eligible else
                      min(options, key=lambda a: (abs(a[1] - accepted), a[3], a[2])))
            config[mode]["threshold"] = choice[-1]
            comparisons[mode] = dict(zip(("admissible", "submitted", "wrong_regions", "wrong_branch", "strict", "threshold"), choice))
            comparisons[mode]["matched"] = totals(counts)
        # Choose the terminal decoder entirely on development results. The
        # closed-loop actor is a separate ablation of state and evidence.
        baseline = comparisons["unary"]["matched"]
        challengers = [mode for mode in LIVE[1:] if comparisons[mode]["admissible"] and
                       comparisons[mode]["matched"]["wrong_branch"] <= baseline["wrong_branch"] and
                       comparisons[mode]["matched"]["wrong_region_count"] <= baseline["wrong_region_count"] and
                       comparisons[mode]["matched"]["strict"] >= baseline["strict"] and
                       (comparisons[mode]["matched"]["strict"] > baseline["strict"] or
                        comparisons[mode]["matched"]["wrong_region_count"] < baseline["wrong_region_count"])]
        chosen = max(challengers, default="unary", key=lambda k: (comparisons[k]["matched"]["strict"],
                     -comparisons[k]["matched"]["wrong_region_count"]))
        rules[route] = config
        decisions[route] = {"decoder": chosen, "development": comparisons, "reference_submissions": accepted,
                            "reference_wrong_branches": branches, "threshold_layouts": len(test),
                            "joint_layouts": len({r["layout"] for r in deck if r["fold"] == 0}),
                            "risk_layouts": len({r["layout"] for r in deck if r["fold"] == 1})}
    return rules, decisions


def frozen_reader(base, calibration):
    return {task: {"model": base["model"], "edge_model": base["edge_model"], "usability": base["usability"],
                   "v5": {"model": base["model"], "edge_model": base["edge_model"],
                          "risk": base["v5"]["risk"][task]["prematch_track"],
                          "threshold": base["v5"]["thresholds"][task]["methods"]["prematch_track"]["threshold"]},
                   "delivery": calibration["decision"][task]["decoder"], "calibration": calibration["calibration"][task]}
            for task in ROUTES}


def compare(rows, freeze):
    groups = defaultdict(dict)
    for r in rows:
        route, case = r["route"], r["case"]
        config = freeze["calibration"][route]
        public, labs, states = r["public"], r["labels"], r["states"]
        history = groups[route].setdefault(case, {"layout": r["layout"], "case": case, "picks": {}})
        scope = r["desc"]["scope"]
        old_pool = public["frames"] < r["desc"]["first_frame_count"]
        visible = old_pool & (public["regions"] == r["region"])
        if scope != "terminal":
            stage = {}
            for name in METHODS:
                v = scores(public, states, name, config[name], mask=old_pool)
                winner = np.flatnonzero(visible & (v >= 0.))
                if len(winner):
                    i = int(winner[np.argmax(v[winner])]); stage[name] = (int(labs[i]) if v[i] >= config[name]["threshold"] else -99)
                else:
                    stage[name] = -99
            history["picks"][scope] = stage
            continue
        q = {m: scores(public, states, m, config[m]) for m in METHODS}
        fixed = {m: evaluate(public, labs, q[m], config[m]["threshold"], complete=True) for m in METHODS}
        coverage = fixed["unary"]["submitted"]
        matched = {m: evaluate(public, labs, q[m], 0., complete=True, submit_count=coverage) for m in METHODS}
        all_winners = {m: evaluate(public, labs, q[m], 0., complete=True) for m in METHODS}
        if any(v["submitted"] != coverage for v in matched.values()):
            raise RuntimeError("Same-coverage reader comparison failed")
        history.update(fixed=fixed, matched=matched, all_winners=all_winners,
                       edge_counts={m: states[m]["edge_counts"] for m in METHODS})
    panels = {}
    for route, cases in groups.items():
        rows = list(cases.values()); ids = [r["layout"] for r in rows]
        modes = {}
        for scope in ("matched", "fixed", "all_winners"):
            reference = [r[scope]["unary"] for r in rows]
            modes[scope] = {m: totals([r[scope][m] for r in rows]) |
                {"vs_unary": paired([r[scope][m] for r in rows], reference, ids, 203_262_901)} for m in METHODS}
        old = {}
        usable = [r["picks"] for r in rows if {"after_hold", "after_probe"} <= r["picks"].keys()]
        for m in METHODS:
            a, b = (np.asarray([r[k][m] == 0 for r in usable]) for k in ("after_hold", "after_probe"))
            old[m] = {"pairs": len(a), "rescues": int((~a & b).sum()), "losses": int((a & ~b).sum())}
        panels[route] = {"layouts": len(rows), "panels": modes, "old_pool_after_probe_vs_hold": old,
                         "edges_correct_total": {m: np.asarray([r["edge_counts"][m] for r in rows]).sum(axis=0).tolist()
                                                 for m in METHODS}, "rows": rows}
    return panels


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "run")
    p.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    out, atlas = args.out.resolve(), args.atlas_pack.resolve()
    base = warm_bundle(out, atlas)
    manifest = out / "reader_freeze.json"
    protocol = {"experiment": ROOT.name, "base_commit": BASE, "atlas_sha256": digest(atlas),
                "v5_reader_sha256": digest(V5_RUN / "reader_freeze.json"), "usability_sha256": digest(out / "usability_bundle.json"),
                "reader_source_sha256": {name: digest(ROOT / name) for name in
                    ("v6_data.py", "role_fusion.py", "usability_fit.py", "causal_reader.py", "reader_study.py")},
                "calibration_splits": "archive development ordinal//2 modulo 3, joint/risk/submit",
                "history_semantics": "recompute from a complete causal prefix, archive snapshots stored fp16",
                "fresh_roster": STAGES["reader_fresh"], "dird": False}
    ctx = mp.get_context("spawn")
    if manifest.exists():
        f = read_json(manifest)
        if any(f[key] != value for key,value in protocol.items()):
            raise RuntimeError("v6 frozen reader protocol changed")
    else:
        sources = sorted((V4_RUN / "select_reader").glob("*/data.npz"))
        with ProcessPoolExecutor(min(args.workers, len(sources)), mp_context=ctx, initializer=init_panel, initargs=(base,)) as pool:
            fit = list(sample_panels(pool, sources, "calibrate"))
        calibration, decisions = calibrate(fit, base["v5"])
        f = protocol | {"calibration": calibration, "decision": decisions,
                        "development_input_sha256": {path.parent.name: digest(path) for path in sources}}
        atomic_json(manifest, f)
        del fit
    print("Reader frozen", {r: v["decoder"] for r,v in f["decision"].items()}, flush=True)
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_worlds, initargs=(atlas,)) as pool:
        fresh, _ = collect(pool, out, "reader_fresh", current_scan, pack_acquired, size=32)
    archive = sorted((V5_RUN / "reader_test").glob("*/data.npz"))
    with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=init_panel, initargs=(base,)) as pool:
        outcomes = {key: compare(sample_panels(pool, paths, "evaluate"), f) for key, paths in
                    (("archived_v5", archive), ("fresh_v6", fresh))}
    atomic_json(out / "reader_results.json", {"reader_freeze_sha256": digest(manifest), "comparison": outcomes})
    for key, panel in outcomes.items():
        print(key, {route: {kind: [v["strict"], v["wrong_region_count"], v["wrong_branch"]] for kind,v in
                           info["panels"]["matched"].items()} for route,info in panel.items()}, flush=True)


if __name__ == "__main__":
    main()

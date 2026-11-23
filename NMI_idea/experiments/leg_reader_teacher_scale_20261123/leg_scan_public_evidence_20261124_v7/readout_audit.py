"""Compare object-first and direct joint delivery with matched observation counts."""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path

from study_v7 import DEFAULT_PACK, ROOT, ROUTES, V4_RUN, V6_RUN, atomic_json, digest, read_json
from causal_reader import choose, likelihood, observation_features, scores
from reader_study import init_panel, sample_panels, warm_bundle
from readout_state import evaluate, fit_risk
from route_metrics import paired, totals
from reader_v7 import candidate_delivery, direct_features
import numpy as np


VARIANTS = ("handwritten_objects", "direct_public", "learned_object_public", "direct_oracle_frozen", "direct_oracle_refit")
OBJECT_FIELDS = ("role_target", "peak_usable", "mean_usable", "peak_joint", "mean_physical_mass", "peak_quality",
                 "view_count", "mean_target_unary")


def groups(public, state, joint):
    result = []
    for region in range(6):
        indices = np.flatnonzero(public["regions"] == region)
        tracks = state["track_ids"]
        for key in np.unique(tracks[indices]):
            ii = indices[tracks[indices] == key]
            best = int(ii[np.argmax(joint[ii])])
            x = [np.max(state["object_role"][ii,0]), np.max(state["usable"][ii]), np.mean(state["usable"][ii]), joint[best],
                 np.mean(state["unary"][ii,:6].sum(axis=1)), np.max(public["quality"][ii]),
                 np.minimum(np.max(state["view_counts"][ii]), 6.)/6., np.mean(state["unary"][ii,0])]
            result.append((region, ii, best, x))
    return result


def train(tape, old):
    # Layout-level folds are inherited from the v6 development panel. The
    # oracle experiment can only change its diagnostic scoring model.
    fit = [r for r in tape if r["fold"] == 0]
    cv = [r for r in tape if r["fold"] == 1]
    oracle_joint, object_rows = [], []
    public_joint = old["objects"]["joint"]
    for r in fit:
        pub, g, lab = r["public"], r["states"], r["labels"]
        # For the oracle-specific calibration, preserve all negative artefacts.
        if "object_oracle" in g:
            oracle_joint.append((direct_features(pub,g["object_oracle"]), lab==0))
        cand = candidate_delivery(pub,g["objects"],public_joint)
        entries = groups(pub,g["objects"],cand)
        if entries:
            x = [row[-1] for row in entries]
            target = [np.sum(lab[row[1]] == 0) > .5 * len(row[1]) for row in entries]
            object_rows.append((np.asarray(x), np.asarray(target)))
    new_joint = fit_risk(oracle_joint) | {"field_order": list(old["objects"]["joint"]["field_order"])}
    object_fit = fit_risk(object_rows) | {"field_order": OBJECT_FIELDS, "target": "majority actual target in regional object support"}
    config = {"handwritten_objects": old["objects"], "direct_public": {"joint": public_joint},
              "learned_object_public": {"joint": public_joint, "object_fit": object_fit},
              "direct_oracle_frozen": {"joint": public_joint}, "direct_oracle_refit": {"joint": new_joint}}
    for mode in VARIANTS[1:]:
        values = []
        for r in cv:
            ids, obs = pick(r, mode, config[mode])
            if len(ids):
                values.append((obs, r["labels"][ids]==0))
        config[mode]["winner"] = fit_risk(values)
    return config


def pick(entry, mode, cfg):
    pub = entry["public"]; state = entry["states"]["object_oracle" if "oracle" in mode else "objects"]
    if mode == "handwritten_objects":
        idx, _, x = choose(pub, state, "objects", cfg["joint"])
        return idx, x
    joint = candidate_delivery(pub,state,cfg["joint"])
    out = []
    if "object_fit" in cfg:
        options = groups(pub,state,joint)
        if options:
            object_scores = likelihood(np.array([x[-1] for x in options]),cfg["object_fit"])
            by_region = np.asarray([x[0] for x in options])
            for region in np.unique(by_region):
                select = np.flatnonzero(by_region==region)
                out.append(options[int(select[np.argmax(object_scores[select])])][2])
    else:
        for region in range(6):
            ii = np.flatnonzero(pub["regions"] == region)
            if len(ii):
                out.append(int(ii[np.argmax(joint[ii])]))
    rows = []
    for i in out:
        r = int(pub["regions"][i]); ix = np.flatnonzero(pub["regions"]==r); ids = state["track_ids"][ix]
        competitor = joint[ix][ids!=state["track_ids"][i]].max(initial=0.)
        q = state["object_role"][i]
        rows.append([joint[i], joint[i]-competitor, q[0]-q[1:].max(), pub["quality"][i],
                     min(1., float(state["view_counts"][i])/5.), 1.-state["usable"][i], min(1.,len(ix)/120.),
                     min(1.,len(np.unique(ids))/40.), *np.eye(6)[r]])
    return np.asarray(out,dtype=int), np.asarray(rows,dtype=float).reshape(-1,14)


def report(test, old, settings):
    results = {}
    for route in ROUTES:
        rows = [r for r in test if r["route"]==route]
        domains = defaultdict(list); led = []
        for r in rows:
            public, labs = (r[k] for k in ("public","labels")); calibration = old[route]["unary"]
            u = scores(public,r["states"],"unary",calibration)
            ref = evaluate(public,labs,u,calibration["threshold"],complete=True)
            domains["unary"].append(ref); rec = {"layout": r["layout"], "unary": ref}
            for mode,cfg in settings[route].items():
                i,x = pick(r,mode,cfg)
                y = np.full(len(labs),-1.); y[i] = likelihood(x,cfg["winner"])
                out = evaluate(public,labs,y,0.,complete=True,submit_count=ref["submitted"])
                assert out["submitted"] == ref["submitted"]
                domains[mode].append(out); rec[mode] = out
            led.append(rec)
        result = {mode: totals(a) | {"vs_unary": paired(a,domains["unary"],[r["layout"] for r in rows],205_341_470)}
                  for mode,a in domains.items()}
        results[route] = {"layouts":len(rows),"matched":result,"rows":led}
    return results


def loader(job):
    # Evaluation scope in the inherited panel extracts oracle and public states
    # for all layouts. This study consumes only the terminal fixed-track panel.
    from reader_study import load_panel
    return [r for r in load_panel((job,"evaluate")) if r["desc"]["scope"]=="terminal"]


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", default=ROOT/"run", type=Path)
    p.add_argument("--atlas-pack", default=DEFAULT_PACK, type=Path)
    p.add_argument("--workers", default=4, type=int)
    args=p.parse_args(); gold = V6_RUN/"reader_freeze.json"; old=read_json(gold)["calibration"]
    base=warm_bundle(V6_RUN,args.atlas_pack)
    dev=sorted((V4_RUN/"select_reader").glob("*/data.npz"))[:3]
    test=sorted((V6_RUN/"reader_fresh").glob("*/data.npz"))
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context("spawn"),initializer=init_panel,initargs=(base,)) as pool:
        training=[r for batch in pool.map(loader,dev) for r in batch]
        settings={route:train([r for r in training if r["route"]==route], old[route]) for route in ROUTES}
        atomic_json(args.out/"readout_diagnostic_fit.json", {"development_only":True,"v6_readout":digest(gold),
                    "dev_shards_sha256":[digest(f) for f in dev],"settings":settings})
        del training
        out=report([r for batch in pool.map(loader,test) for r in batch], old,settings)
    atomic_json(args.out/"readout_diagnostic.json",out)
    for route, block in out.items():
        print(route,{m:[v[k] for k in ("strict","wrong_region_count","wrong_branch","artifact_regions")] for m,v in block["matched"].items()},flush=True)


if __name__=="__main__":
    main()

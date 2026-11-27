"""Verify the four held-out shards and render paired whole-leg rechecks."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binomtest

from certification import ARMS, DYNAMIC, HERE, STRESS, DynamicScan, digest

LABELS = {"A":"Strong A","A_fisher":"Fisher A","fixed":"Fixed continuation","B":"B"}
COLORS = {"A":"#ba712a","A_fisher":"#337c9e","fixed":"#777f8b","B":"#763c9d"}
COUNTS = np.arange(0,81,5)


def collect(prefix, starts, shard_size, seed_base):
    rows, audit = [], []
    for start in starts:
        folder = HERE/f"{prefix}_{start}"
        d = json.loads((folder/"SUMMARY.json").read_text())
        assert d["start_index"]==start and d["n"]==shard_size and d["cap"]==80 and d["seed_base"]==seed_base
        assert d["source_sha256"]==digest(HERE/"certification.py")
        with (folder/"RUNS.jsonl").open() as f:
            block = list(map(json.loads,f))
        assert len(block)==len(ARMS)*d["n"]
        rows.extend(block)
        audit.append(d)
    cases = defaultdict(dict)
    for row in rows:
        cases[row["seed"]][row["arm"]] = row
    assert len(cases)==len(starts)*shard_size and all(set(v)==set(ARMS) for v in cases.values())
    for per in cases.values():
        assert len({tuple(r["eval_only_truth"]) for r in per.values()})==1
        assert len({r["prefix_acquisitions"] for r in per.values()})==1
    return cases,audit


def summarize(cases, audits):
    rows = list(cases.values())
    result = {"cases":len(rows),"prefix_complete":sum(r["A"]["prefix_complete"] for r in rows),
              "shared_geometry_sha256":audits[0]["geometry_sha256"],
              "sampled_joint_edges_checked_per_shard":[a["n_checked_edge_pairs"] for a in audits],"arms":{}}
    for arm in ARMS:
        d = [r[arm] for r in rows]
        result["arms"][arm] = {
            "complete":sum(r["complete"] for r in d),
            "online_stop":sum(r["online_stop"] for r in d),
            "wrong_witnesses":sum(len(r["incorrect_witnesses"]) for r in d),
            "mean_rechecks":float(np.mean([len(r["steps"]) for r in d])),
            "mean_rmse_mm":float(np.mean([r["map_rmse_mm"] for r in d])),
            "mean_max_band_mm":float(np.mean([r["max_band_mm"] for r in d])),
            "mean_recheck_path_mm":float(np.mean([r["recheck_path_mm"] for r in d])),
            "changes":sum(s["changed_suffix"] for r in d for s in r["steps"]),
            "completion_curve":{str(m):int(sum(r["complete"] and len(r["steps"])<=m for r in d)) for m in COUNTS},
        }
    result["paired_B"] = {}
    for arm in ("A","A_fisher","fixed"):
        differences = np.array([int(r["B"]["complete"])-int(r[arm]["complete"]) for r in rows])
        pos,neg = int((differences>0).sum()),int((differences<0).sum())
        generator = np.random.default_rng(7191)
        boot = generator.choice(differences,(8000,len(differences))).mean(axis=1)
        identical = sum([(s["pose"],s["action"]) for s in r["B"]["steps"]]==
                        [(s["pose"],s["action"]) for s in r[arm]["steps"]] for r in rows)
        result["paired_B"][arm] = {
            "wins":pos,"losses":neg,"ties":int(len(differences)-pos-neg),
            "difference":float(differences.mean()),
            "bootstrap_95":np.percentile(boot,[2.5,97.5]).tolist(),
            "exact_two_sided_p":float(binomtest(pos,pos+neg,.5).pvalue) if pos+neg else 1.,
            "identical_action_sequences":identical,
        }
    return result


def draw(cases, stats):
    with np.load(DYNAMIC/"DYNAMIC_GEOMETRY_FROZEN.npz") as data:
        scan = DynamicScan(data)
        transforms = data["joint_transforms"][scan.bent]
    surface = HERE.parent/"ab_validation_20261110/repair_20261110/harmonic_full_domain_followup/contact_multichart_right.npz"
    with np.load(surface) as data:
        xyz,tri,weights = data["vertices"],data["faces"],data["weights"]
    tf = np.einsum("vj,jkl->vkl",weights,transforms)
    posed = 1000.*np.einsum("vkl,vl->vk",tf,np.column_stack([xyz,np.ones(len(xyz))]))[:,:3]
    seed,rows = next(iter(cases.items()))
    p = scan.bent
    line = 1000.*scan.trunk[p]
    branch = 1000.*scan.dense_branch[p]
    fig = make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"xy"}]],
                        subplot_titles=("Simulated rechecks · 15° atlas reference","Correct completion by recheck count"),
                        column_widths=[.62,.38],horizontal_spacing=.08)
    fig.add_trace(go.Mesh3d(x=posed[:,0],y=posed[:,1],z=posed[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],
                            color="#aabbc8",opacity=.13,name="Posed skin",showlegend=False),1,1)
    for positions,label in ((line,"Trunk template"),(branch,"Daughter template")):
        fig.add_trace(go.Scatter3d(x=positions[:,0],y=positions[:,1],z=positions[:,2],mode="lines",
                                     name=label,line=dict(width=6,color="#0d8f82")),1,1)
    for arm in ARMS:
        r=rows[arm]
        points=[]
        for step in r["steps"]:
            points.extend(step["path_m"])
            points.append([np.nan]*3)
        if points:
            x=np.asarray(points)*1000.
            fig.add_trace(go.Scatter3d(x=x[:,0],y=x[:,1],z=x[:,2],mode="lines",
                                         legendgroup=arm,name=LABELS[arm],
                                         line=dict(width=4,color=COLORS[arm])),1,1)
        values=stats["arms"][arm]["completion_curve"]
        fig.add_trace(go.Scatter(x=COUNTS,y=[values[str(m)]/stats["cases"] for m in COUNTS],
                                 name=LABELS[arm],mode="lines+markers",showlegend=False,legendgroup=arm,
                                 line=dict(color=COLORS[arm],shape="hv")),1,2)
    fig.update_scenes(aspectmode="data",camera=dict(eye=dict(x=2.5,y=.8,z=2.1),up=dict(x=.16,y=1,z=0)),
                      xaxis_title="X (mm)",yaxis_title="Y (mm)",zaxis_title="Z (mm)")
    fig.update_xaxes(title="Supplemental acquisitions",row=1,col=2)
    fig.update_yaxes(title="Strict completion across matched cases",range=[0,1.04],tickformat=".0%",row=1,col=2)
    fig.update_layout(title=f"Full-leg rechecks · {stats['cases']} matched cases · first example {seed}",
                       height=760,template="plotly_white",legend=dict(orientation="h",y=-.05))
    fig.write_html(HERE/"RECHECK_3D.html",include_plotlyjs=True)


if __name__=="__main__":
    cases,audits=collect("recheck_holdout",(0,6,12,18),6,91261127)
    report=summarize(cases,audits)
    (HERE/"RECHECK_HOLDOUT.json").write_text(json.dumps(report,indent=2)+"\n")
    if all((HERE/f"recheck_replication_{i}/SUMMARY.json").is_file() for i in (0,8,16,24,32,40)):
        other,other_audit=collect("recheck_replication",(0,8,16,24,32,40),8,61024891)
        assert not cases.keys()&other.keys()
        replication=summarize(other,other_audit)
        (HERE/"RECHECK_REPLICATION.json").write_text(json.dumps(replication,indent=2)+"\n")
        cases.update(other)
        report=summarize(cases,audits+other_audit)
        (HERE/"RECHECK_POOLED.json").write_text(json.dumps(report,indent=2)+"\n")
    draw(cases,report)
    print(json.dumps(report,indent=2))

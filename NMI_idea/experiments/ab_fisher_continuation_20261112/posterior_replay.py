"""Replay public observations from the frozen long scan and show map uncertainty."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
DYNAMIC = HERE.parent/"ab_leg_dynamic_20261112"
sys.path.insert(0,str(DYNAMIC))
from dynamic_scan import DynamicScan, update_set


def replay():
    with np.load(DYNAMIC/"DYNAMIC_GEOMETRY_FROZEN.npz") as data:
        scan = DynamicScan(data)
    with (DYNAMIC/"matched_paired/RUNS.jsonl").open() as handle:
        row = next(r for r in map(json.loads,handle) if r["scenario"]=="bent50" and r["method"]=="B")
    mean,C = np.zeros(scan.dim),scan.prior.copy()
    C[-1,-1] = 1.e-6
    visited = set()
    frames = []

    def snapshot(t):
        sigma = np.sqrt(np.diag(C)[:scan.n])
        frames.append({"observations":t,"mean_mm":mean[:scan.n].tolist(),"sigma_mm":sigma.tolist(),
                       "sites_with_centroid":sorted(visited)})

    snapshot(0)
    for k,s in enumerate(row["steps"],1):
        p = s["pose"]
        i,v = s["action"]
        values = s["observation_mm"]
        mean,C,*_ = update_set(mean,C,scan.models(p,(i,v),False),values,25.)
        if values:
            visited.add(i)
        snapshot(k)
    if not np.allclose(mean,row["estimate"],atol=1.e-11) or not np.allclose(C,row["cov"],atol=1.e-11):
        raise RuntimeError("posterior replay differs from the frozen run")
    return scan,row,frames


def main():
    scan,row,frames = replay()
    phase_ids = sorted(set([0,1,5,10,20,len(frames)-1]))
    pose = scan.bent
    default_frame = 20
    trunk = scan.trunk[pose]*1000.
    tangent = scan.frames[pose,:,:,1]
    truth = np.asarray(row["eval_only_truth"])[:scan.n]
    target = trunk+tangent*truth[:,None]
    figure = make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"xy"}]],
                           subplot_titles=("15° leg atlas and internal vessel","Lateral uncertainty by material address"))
    for xy,name,color,dash in ((trunk,"Template prior","#8090a2","dash"),(target,"Simulated truth","#b26934","dot")):
        figure.add_trace(go.Scatter3d(x=xy[:,0],y=xy[:,1],z=xy[:,2],mode="lines",name=name,
                                      line=dict(color=color,width=5,dash=dash)),1,1)
    center=scan.views[pose,:,scan.central]*1000.
    figure.add_trace(go.Scatter3d(x=center[:,0],y=center[:,1],z=center[:,2],mode="lines",name="Surface route",
                                  line=dict(color="#c6cdd5",width=3)),1,1)
    fixed=len(figure.data)
    for frame in (frames[t] for t in phase_ids):
        sigma=np.asarray(frame["sigma_mm"])
        line=trunk+tangent*np.asarray(frame["mean_mm"])[:,None]
        vis=frame["observations"]==default_frame
        figure.add_trace(go.Scatter3d(x=line[:,0],y=line[:,1],z=line[:,2],mode="lines",name="Posterior mean",
                                     line=dict(color="#218579",width=7),visible=vis),1,1)
        seen=np.array(frame["sites_with_centroid"],dtype=int)
        a=center[seen]
        figure.add_trace(go.Scatter3d(x=a[:,0],y=a[:,1],z=a[:,2],mode="markers",name="Sites with a centroid",
                                     marker=dict(color="#70399f",size=5),visible=vis),1,1)
        figure.add_trace(go.Scatter(x=scan.s,y=sigma,mode="lines+markers",name="Posterior SD",
                                    line=dict(color="#218579"),visible=vis),1,2)
        figure.add_trace(go.Scatter(x=scan.s[seen],y=sigma[seen],mode="markers",name="Direct observations",
                                    marker=dict(color="#70399f",size=9),visible=vis),1,2)
    sliders=[]
    for i,t in enumerate(phase_ids):
        visible=[True]*fixed+[j//4==i for j in range(4*len(phase_ids))]
        sliders.append(dict(label=str(t),method="update",args=[{"visible":visible}]))
    figure.update_layout(title=f"Posterior replay · paired seed {row['seed']} · 50 mm",
                         template="plotly_white",height=690,margin=dict(b=110,t=100),
                         sliders=[dict(steps=sliders,active=phase_ids.index(default_frame),
                                       currentvalue=dict(prefix="Acquisitions: "))])
    figure.update_scenes(aspectmode="data",camera=dict(eye=dict(x=2.2,y=.8,z=1.8),up=dict(x=0,y=1,z=0)))
    figure.update_xaxes(title="Material arc length (mm)",row=1,col=2)
    figure.update_yaxes(title="Marginal SD (mm)",range=[0,8],row=1,col=2)
    figure.write_html(HERE/"POSTERIOR_UPDATES_3D.html",include_plotlyjs=True)
    first=row["steps"][0]
    i,j=first["action"]
    independent=np.diag(np.diag(scan.prior))
    independent[-1,-1]=1.e-6
    im,ic,*_=update_set(np.zeros(scan.dim),independent,scan.models(first["pose"],(i,j),False),
                        first["observation_mm"],25.)
    remote=np.arange(scan.n)!=i
    report={"seed":row["seed"],"scope":"public centroid replay of one 15° case; posterior SD per trunk node",
            "arc_mm":scan.s.tolist(),"stages":frames,
            "zero_cross_covariance_control":{
                "max_unseen_mean_change_mm":float(abs(im[:scan.n][remote]).max()),
                "max_unseen_sd_change_mm":float(abs(np.sqrt(np.diag(ic)[:scan.n][remote])-
                                                 np.sqrt(np.diag(independent)[:scan.n][remote])).max())}}
    (HERE/"POSTERIOR_REPLAY.json").write_text(json.dumps(report,indent=2)+"\n")
    for t in phase_ids:
        f=frames[t]; v=np.asarray(f["sigma_mm"])
        seen=np.zeros(len(v),dtype=bool);seen[f["sites_with_centroid"]]=True
        print(t,int(seen.sum()),round(float(v[seen].mean()),3) if seen.any() else None,
              round(float(v[~seen].mean()),3) if (~seen).any() else None)


if __name__=="__main__":
    main()

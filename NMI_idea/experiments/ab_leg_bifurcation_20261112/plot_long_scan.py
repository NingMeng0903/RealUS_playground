"""Standalone 3-D audit view of the actual atlas, paths and lateral posterior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import ndtri


HERE = Path(__file__).resolve().parent
FULL_SURFACE = (HERE.parent / "ab_validation_20261110/repair_20261110/harmonic_full_domain_followup"
                / "contact_multichart_right.npz")


def line(points, name, color, *, width=4, dash=None, group=None, legend=True, text=None):
    x,y,z = np.asarray(points).T*1000.
    return go.Scatter3d(x=x,y=y,z=z,mode="lines",name=name,text=text,
                        line=dict(color=color,width=width,dash=dash),
                        legendgroup=group or name,showlegend=legend)


def marker(points,name,color,*,symbol="circle",text=None):
    x,y,z=np.asarray(points).T*1000.
    return go.Scatter3d(x=x,y=y,z=z,mode="markers",name=name,text=text,
                        marker=dict(color=color,size=4,symbol=symbol))


def mesh(vertices,faces,name,color,opacity):
    x,y,z=np.asarray(vertices).T*1000.
    a,b,c=np.asarray(faces).T
    return go.Mesh3d(x=x,y=y,z=z,i=a,j=b,k=c,name=name,color=color,opacity=opacity,
                     hoverinfo="skip",showlegend=True,legendgroup=name)


def ribbon(mean,lateral,band,name,color):
    edge=band[:,None]*lateral/1000.
    pts=np.vstack((mean-edge,mean+edge))
    n=len(mean)
    idx=np.arange(n-1)
    triangles=np.vstack((np.c_[idx,idx+1,idx+n],np.c_[idx+1,idx+n,idx+n+1]))
    return mesh(pts,triangles,name,color,.22)


def program_path(episode):
    points=[]
    for segment in episode["program"]:
        points.extend(segment["path_m"])
        points.append([np.nan,np.nan,np.nan])
    return np.asarray(points, dtype=float)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("--dropout",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    run=args.run
    summary=json.loads((run/"SUMMARY.json").read_text())
    plan=json.loads((run/"BRANCH_AND_GPS.json").read_text())
    with (run/"RUNS.jsonl").open() as handle:
        episodes = [r for r in map(json.loads, handle) if r["dropout"] == args.dropout]
    r=next(r for r in episodes if r["method"]=="B_joint")
    comparisons={a["method"]:a for a in episodes if a["seed"]==r["seed"]}
    source=Path(summary["view_source"])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=summary["view_sha256"]:
        raise ValueError("view table does not match the recorded run")
    with np.load(source,allow_pickle=False) as data:
        s,base,frames,skin,views,rows=[data[k] for k in ("s_mm","vessel_m","frames","skin_m","views_m","row_ids")]
    with np.load(HERE/"LONG_LEG_ASSETS.npz",allow_pickle=False) as data:
        full,full_skin,valid=[data[k] for k in ("vessel_m","skin_m","internal_valid")]
    with np.load(FULL_SURFACE,allow_pickle=False) as data:
        verts,faces=data["vertices"],data["faces"]
    mean,truth,C=np.asarray(r["mean"]),np.asarray(r["eval_only_truth"]),np.asarray(r["cov"])
    side=frames[:,:,1]
    posterior=base+side*mean[:-1,None]/1000.
    truth_main=base+side*truth[:-1,None]/1000.
    multiplier=ndtri(1.-.10/(2.*(len(s)+len(plan["branch"]["target_stations"]))))
    band=multiplier*np.sqrt(np.maximum(0.,np.diag(C)[:-1]))
    branch=plan["branch"]
    u=np.asarray(branch["dense_s_mm"])
    branch_base=np.asarray(branch["dense_centerline_m"])
    f=(u-u[0])/(u[-1]-u[0])
    direction=np.column_stack([np.interp(u,s,side[:,i]) for i in range(3)])
    direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    branch_mean=branch_base+direction*(np.interp(u,s,mean[:-1])+f*mean[-1])[:,None]/1000.
    branch_truth=branch_base+direction*(np.interp(u,s,truth[:-1])+f*truth[-1])[:,None]/1000.
    bq=np.column_stack([np.interp(u,s,col) for col in np.eye(len(s))]+[f])
    branch_band=multiplier*np.sqrt(np.maximum(0.,np.einsum("ij,jk,ik->i",bq,C,bq)))
    figure=make_subplots(rows=1,cols=2,specs=[[{"type":"scene"},{"type":"scene"}]],
                        subplot_titles=("Full baked leg and source vein","Simulated scan and internal posterior"),
                        horizontal_spacing=.035)
    figure.add_trace(mesh(verts,faces,"Full skin atlas","#b9cbd6",.14),1,1)
    figure.add_trace(line(full,"Original 795 mm vessel template","#5d6a83",width=5),1,1)
    figure.add_trace(line(full_skin,"Original surface footprint","#bcb8a6",width=3,dash="dot"),1,1)
    figure.add_trace(line(full[rows[0]:rows[-1]+1],"Tested trunk interval","#007d6f",width=7),1,1)
    figure.add_trace(line(full[29:104],"Other strictly rebound segment","#557aaf",width=6),1,1)
    figure.add_trace(marker(full[104:106],"Knee gap in original atlas","#c14641",symbol="x",
                            text=["Original row 104","Original row 105"]),1,1)
    outside=(~valid).copy()
    outside[104:106]=False
    figure.add_trace(marker(full[outside],"Source points outside verified volume","#7e8593"),1,1)
    figure.add_trace(line(branch_base,"Synthetic branch","#8452a9",width=6),1,1)

    # Show the actual skin in a crop, using the same vertices and face IDs.
    lower=np.min(np.vstack((base,skin,branch_base)),axis=0)-.028
    upper=np.max(np.vstack((base,skin,branch_base)),axis=0)+.028
    mask=((verts>=lower)&(verts<=upper)).all(axis=1)
    patch=faces[mask[faces].all(axis=1)]
    used=np.unique(patch)
    figure.add_trace(mesh(verts[used],np.searchsorted(used,patch),"Scan skin patch","#b9cbd6",.13),1,2)
    figure.add_trace(line(base,"Initial internal prior mean","#a3aabc",width=5,dash="dash"),1,2)
    figure.add_trace(line(branch_base,"Initial internal prior mean","#a3aabc",width=5,dash="dash",legend=False),1,2)
    figure.add_trace(line(truth_main,"Simulated truth","#b54d29",width=4,dash="dot"),1,2)
    figure.add_trace(line(branch_truth,"Simulated truth","#b54d29",width=4,dash="dot",legend=False),1,2)
    figure.add_trace(line(posterior,"Posterior vessel mean","#006f65",width=7),1,2)
    figure.add_trace(line(branch_mean,"Posterior vessel mean","#006f65",width=7,legend=False),1,2)
    band_label="Approx. 90% lateral band"
    figure.add_trace(ribbon(posterior,side,band,band_label,"#0c988c"),1,2)
    branch_ribbon=ribbon(branch_mean,direction,branch_band,band_label,"#0c988c")
    branch_ribbon.showlegend=False
    figure.add_trace(branch_ribbon,1,2)
    initial=np.asarray([views[x["station"],x["view"]] for x in plan["gps"] if x["view"] is not None])
    if len(initial):
        figure.add_trace(marker(initial,"Prior GPS contacts","#997632"),1,2)
    for method,label,color in (("B_joint","B sampled ProDMP","#654faa"),
                               ("A_joint","Strong A sampled ProDMP","#ae7023"),
                               ("B_open_loop","Open-loop forecast sampled ProDMP","#456a8a")):
        path=program_path(comparisons[method])
        if len(path):
            trace=line(path,label,color,width=3)
            trace.visible=True if method=="B_joint" else "legendonly"
            figure.add_trace(trace,1,2)
    required=np.asarray(plan["targets"])
    unresolved=sorted((set(range(len(required)))-set(r["covered"]))|set(r["wrong"]))
    if unresolved:
        p0=skin[required[unresolved,0]]
        figure.add_trace(marker(p0,"Unconfirmed or wrong target","#b53234",symbol="x",
                                text=[f"Target {j} · {s[required[j,0]]:.1f} mm · type {required[j,1]}" for j in unresolved]),1,2)
    camera=dict(eye=dict(x=2.5,y=.78,z=2.05),up=dict(x=.16,y=1,z=0))
    figure.update_scenes(aspectmode="data",xaxis_title="Atlas X (mm)",yaxis_title="Atlas Y (mm)",
                         zaxis_title="Atlas Z (mm)",camera=camera)
    figure.update_layout(template="plotly_white",font=dict(family="Arial",size=12,color="#243444"),height=760,
                         title=(f"Leg vessel scan · {summary['span_mm']:.0f} mm trunk + {summary['branch_arc_mm']:.0f} mm branch"
                                f"<br><sup>Seed {r['seed']} · {'hidden dropout' if r['dropout'] else 'regular'} · "
                                f"{len(r['covered'])-len(r['wrong'])}/{r['targets']} correct targets · "
                                f"RMSE {r['rmse_before_mm']:.2f} → {r['rmse_after_mm']:.2f} mm</sup>"),
                         margin=dict(t=112,b=82,l=12,r=12),legend=dict(orientation="h",y=-.035))
    figure.write_html(args.output,include_plotlyjs=True,full_html=True,auto_open=False)
    print(args.output)


if __name__=="__main__":
    main()

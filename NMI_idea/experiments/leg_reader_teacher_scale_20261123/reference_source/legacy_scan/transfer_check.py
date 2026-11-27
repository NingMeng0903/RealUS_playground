"""Locked-route atlas priors and one measured identity anchor on the same Blender tape."""
from concurrent.futures import ProcessPoolExecutor
from copy import copy
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import scan as experiment
from geometry import (YS, TASKS, TEST, SOURCES, BP, OPEN, NEW_OPEN, neutral_lines,
                      meshes, read_mesh_pack, profiles, training_meshes)

# A junction proxy uses the closest areas of the two named lumen surfaces in
# a predeclared proximal span. It records location only; the graph edge is fixed.
JUNCTION_WINDOW = (-585., -470.)
ANCHOR_SCALE_MM = 165.


def proximal_node_prior():
    pack = meshes()
    other = read_mesh_pack(NEW_OPEN)
    pairs=[]
    for side in ("right","left"):
        main, branch = [np.asarray(pack[f"BodyParts3D|{side} {name}"].vertices).copy()
                        for name in ("femoral vein","great saphenous vein")]
        pairs.append((main,branch))
    pairs.append(tuple(other[k][0] for k in ("Femoral_vein.r","Great_saphenous_vein.r")))
    ys=[]
    gaps=[]
    for main,branch in pairs:
        a,b=(v[(v[:,1]>=JUNCTION_WINDOW[0])&(v[:,1]<=JUNCTION_WINDOW[1])] for v in (main,branch))
        distance,idx=cKDTree(a).query(b)
        high = distance<=distance.min()+.65
        y=.5*(b[high,1]+a[idx[high],1])
        ys.append(float(np.median(y)))
        gaps.append(float(distance.min()))
    lines=neutral_lines()
    end=lines["R_SAPH_V"][0]
    assert min(np.linalg.norm(lines["R_COM_FEM_V"][[-1,0]]-end,axis=1))<1.e-6
    return np.asarray(ys), np.asarray(gaps), float(end[1])


def atlas_prior_coverage():
    delta=np.linalg.norm(experiment.STORE.centers[:3,:,0]-experiment.STORE.centers[TEST,:,0],axis=-1)
    nearest=np.where(np.isfinite(delta),delta,np.inf).min(axis=0)
    results={}
    for name,(hi,lo,_) in TASKS.items():
        selector=(YS<=hi)&(YS>=lo)
        d=nearest[selector]
        d=d[np.isfinite(d)]
        results[name]={"stations":int(selector.sum()),
                       "stations_with_matched_reference":len(d),
                       "delta_mm_p50_p90":np.quantile(d,[.50,.90]).tolist(),
                       "within_90pct_radius_of_any_reference":{
                           str(std):int((d<=np.sqrt(-2.*np.log(.1))*std).sum())
                           for std in (4,8,12,20)}}
    return results


class AdjustableJunctionModel(experiment.Model):
    """Expand axial geometry support while retaining the same named graph."""
    def __init__(self, sigma, axial_sigma, source_ys, source_templates):
        super().__init__(sigma,True,True)
        self.axial_offsets_mm=np.rint(self.samples[:,2]*axial_sigma/5.).astype(int)*5.
        for s,y in enumerate(YS):
            q=np.rint((source_ys[0]-(y-self.axial_offsets_mm))/5.).astype(int)
            assert ((q>=0)&(q<len(source_ys))).all()
            center=source_templates[:,q,:5].copy()
            shape=.35*sigma*self.samples[:,3]*np.sin(s*.10)
            center += (sigma*self.samples[:,:2]+np.stack((shape,-.5*shape),axis=-1))[None,:,None]
            self.h[s]=center.reshape(len(self.prior),5,2)


def weighted_nodes(model, posterior):
    axial = getattr(model,"axial_offsets_mm",np.rint(model.samples[:,2]*2).astype(int)*5.)
    heights = (NODE_PRIOR[:,None]+axial[None,:]).ravel()
    order=np.argsort(heights)
    cumul=np.cumsum(posterior[order])
    idx=np.minimum(np.searchsorted(cumul,[.05,.5,.95]),len(order)-1)
    lo,mid,hi=heights[order[idx]]
    return [float(lo),float(mid),float(hi)]


def first_anchor(frames):
    for i,row in enumerate(frames):
        if row[5]!=0:
            break
        hits=np.flatnonzero(row[3]==0)
        if len(hits)==1:
            return i,int(hits[0])
    return None


def centered_model(model, row, k):
    s,a,z=row[:3]
    point=experiment.Reader.absolute(s,a,z[k])
    updated=copy(model)
    # The only test-time added datum is the labelled measured centroid.
    # No export coordinates or labels from later frames enter this update.
    delta=(point-model.h[s,:,0])*.86
    delta=np.nan_to_num(np.clip(delta,-65.,65.),nan=0.)
    decay=np.exp(-.5*((YS-YS[s])/ANCHOR_SCALE_MM)**2)
    updated.h=model.h+decay[:,None,None,None]*delta[None,:,None,:]
    return updated


def audit_junction(frames,model,roles,anchor=None):
    reader=experiment.Reader(model,roles,True)
    if anchor is not None:
        first,k=anchor
    else:
        first=None
    history=[]
    for krow,row in enumerate(frames):
        s,a,z,truth,q,region=row[:6]
        if krow==first:
            reader.model=centered_model(model,row,k)
            reader.w=reader.model.prior.copy()
        reader.push(s,a,z,truth,q,region)
        history.append(reader.w.copy())
    joints=weighted_nodes(model,history[-1])
    return {"y05_med_95_mm":joints,"absolute_median_error_mm":abs(joints[1]-NODE_TEST),
            "true_node_in_90pct_interval":int(joints[0]<=NODE_TEST<=joints[2])}


def score_causal_anchor(frames,model,roles,anchor):
    if anchor is None:
        return experiment.score(frames,model,roles,True)
    i,known=anchor
    anchored=centered_model(model,frames[i],known)
    raw=experiment.Reader(model,roles,True)
    for k,row in enumerate(frames):
        s,a,z,y,q,region=row[:6]
        if k==i:
            raw.model=anchored
            raw.w=anchored.prior.copy()
        raw.push(s,a,z,y,q,region)
    # The anchor can count as the independently confirmed first witness.
    ledger=[raw.retained_result(i) for i in range(6)]
    ledger[0]=1
    correct=int(sum(x==1 for x in ledger))
    wrong=int(sum(x==0 for x in ledger))
    return {"correct_regions":correct,"wrong_regions":wrong,"pass":int(correct==6),
            "wrong_delivery":int(wrong>0),"captured_regions":len({int(r[5]) for r in frames if 0 in r[3]}),
            "no_result_regions":6-correct-wrong}


def evaluate(job):
    task,profile,tape=job
    seed=experiment.TEST_SEED+1000*list(TASKS).index(task)+tape
    trial=experiment.Trial(TEST,seed,profile)
    frames,t,length,reviews,slows,_=experiment.scan(task,trial,"mpc",BASE)
    roles=TASKS[task][2]
    anchor=first_anchor(frames)
    arms={name:experiment.score(frames,model,roles,True) for name,model in MODELS.items()}
    arms["sigma20_plus_first_label"]=score_causal_anchor(frames,MODELS["sigma20"],roles,anchor)
    for a in arms.values():
        a.update(time_s=t,travel_mm=length,rechecks=reviews,slow_segments=slows)
    junction=None
    if task=="saphenofemoral":
        junction={"no_label":audit_junction(frames,MODELS["sigma20"],roles),
                  "first_label":audit_junction(frames,MODELS["sigma20"],roles,anchor),
                  "wider_axial":audit_junction(frames,MODELS["sigma20_axial35"],roles),
                  "wider_axial_with_first_label":audit_junction(frames,MODELS["sigma20_axial35"],roles,anchor)}
    return {"task":task,"profile":profile,"tape":tape,"arms":arms,"anchor_available":anchor is not None,
            "junction":junction}


def main(out,jobs):
    global BASE,MODELS,NODE_PRIOR,NODE_TEST
    first=json.loads((out/"summary.json").read_text())
    reps=first["test"]["tapes_per_task_profile"]
    experiment.STORE=experiment.Store(out)
    coverage=atlas_prior_coverage()
    MODELS={f"sigma{s}":experiment.Model(s,True,True) for s in (8,12,20,30)}
    yext=np.arange(-420.,-1381.,-5.)
    shapes=np.stack([profiles(mesh,terms,"mesh",yext)[0] for mesh,terms in zip(training_meshes(),(BP,BP,OPEN),strict=True)])
    MODELS["sigma20_axial35"]=AdjustableJunctionModel(20,35,yext,shapes)
    BASE=MODELS["sigma8"]
    NODE_PRIOR,gaps,NODE_TEST=proximal_node_prior()
    test=[(task,profile,tape) for task in TASKS for profile in experiment.PROFILES for tape in range(reps)]
    with ProcessPoolExecutor(max_workers=jobs) as p:
        rows=list(p.map(evaluate,test,chunksize=1))
    by_task={k:experiment.totals([r for r in rows if r["task"]==k]) for k in TASKS}
    assert experiment.totals(rows)["sigma8"] == first["totals"]["graph_bone_history"]
    jrows=[r["junction"] for r in rows if r["junction"]]
    jsummary={
        "neutral_blender_node_y_mm":NODE_TEST,"reference_proximity_proxy_y_mm":dict(zip(SOURCES[:3],NODE_PRIOR.tolist())),
        "proxy_surface_gap_mm":dict(zip(SOURCES[:3],gaps.tolist())),
        "node_prior_y05_med_95_mm":weighted_nodes(MODELS["sigma20"],MODELS["sigma20"].prior),
        "wider_node_prior_y05_med_95_mm":weighted_nodes(MODELS["sigma20_axial35"],MODELS["sigma20_axial35"].prior),
        "per_policy":{k:{"median_abs_error_mm":float(np.median([r[k]["absolute_median_error_mm"] for r in jrows])),
                         "true_node_in_90pct_interval":int(sum(r[k]["true_node_in_90pct_interval"] for r in jrows)),
                         "n":len(jrows)} for k in ("no_label","first_label","wider_axial","wider_axial_with_first_label")}}
    score={"atlas_prior_coverage":coverage,"one_time_first_label_available":sum(r["anchor_available"] for r in rows),
           "totals":experiment.totals(rows),"by_task":by_task,"saphenofemoral_node_diagnostic":jsummary,
           "paired_against_sigma8":{k:{m:experiment.bootstrap(rows,k,"sigma8",m)
                           for m in ("pass","correct_regions","wrong_delivery")}
                         for k in ("sigma12","sigma20","sigma30","sigma20_axial35","sigma20_plus_first_label")}}
    (out/"transfer_checks.json").write_text(json.dumps(score,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(score["totals"],indent=2),flush=True)
    print(json.dumps(jsummary),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--jobs",type=int,default=8)
    a=cli.parse_args()
    main(a.out,a.jobs)

"""Full three-station lookahead with a budgeted option to reacquire the original request."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import numpy as np

from core import (Landmark,advance,band,fixed_slot,get_world,legal_slots,lite_scores,observe_hold,path,public_belief,
                  req,scan,terminal,trace_plan,visit)
from anchor import edge_poses,reserve
from experiments import OLD,conditional
from observers import observer_outcomes

SEED=607184903
T=16
DRAWS=16


def entry_trip(env,spec,node,h,lim,entry,clock):
    s,a=int(node.station),int(node.sector);distance=s-entry
    if distance<1:return []
    fields=req.nominal_fields(env);tt,mm=reserve(node,spec,lim)
    out=[]
    for offset in (0,-1,1,-2,2):
        angles=(a+np.rint(np.linspace(0,offset,distance+1)).astype(int))%24
        route=path(env,entry,angles[::-1])
        if route is None or 2*route.ticks+7>min(70,tt-3*clock-3) or 2*route.distance>min(180.,mm-28.):
            continue
        poses=edge_poses(env,spec,entry,angles[::-1])[::-1]
        cycle=np.vstack((poses,np.repeat(poses[-1:],7,axis=0),poses[-2::-1]))
        lengths=env.scanner._path_metrics(cycle,spec)
        if lengths is None:continue
        dest=int(angles[-1]);merit=float(fields[entry,dest]@[.20,.60,.20])
        out.append((offset,merit,Landmark(cycle,lengths,"entry_window",0,merit,entry)))
    if not out:return []
    choices=[]
    zero=next((v for v in out if v[0]==0),None)
    if zero:choices.append(zero)
    for direction in (-1,1):
        pool=[v for v in out if v[0]*direction>0]
        if pool:choices.append(max(pool,key=lambda x:x[1]))
    return [p for _,score,p in choices]


def apply_option(env,spec,node,trial,plan,duration):
    if plan is not None:visit(env,spec,node,trial,plan,True)
    observe_hold(env,node,spec,trial,duration-(0 if plan is None else len(plan.lengths)))


def continue_score(env,spec,n,h,trial,lim,goal,stage,clock,legal):
    fields=req.nominal_fields(env)
    return int(legal[np.argmax(lite_scores(env,spec,n,h,goal,stage,legal,fields))])


def value(env,spec,n,h,t,lim,goal,entry,cut,bands,stage,clock,first=None):
    b=list(bands)
    for j in range(stage,3):
        before=len(n.reader.frames)
        legal=legal_slots(env,spec,n,lim,clock,3-j)
        slot=first if j==stage and first is not None else continue_score(env,spec,n,h,t,lim,goal,j,clock,legal)
        advance(env,spec,n,t,lim,slot,clock);h.append(n.reader.frames);b.append(band(n.reader.frames,h,before))
    r=terminal(n,h,entry,cut,b)
    def score(p):return p["correct"]-p["wrong"]
    return (.25*score(r["entry"])+.27*score(r["fresh"])+.18*r["both"]+.12*r["chain"]
            +.18*np.mean([score(a) for a in b]))


def select_pre(env,spec,n,h,trial,lim,goal,entry,clock,options,duration,seed):
    totals=np.zeros(len(options))
    for k in range(DRAWS):
        t=conditional(trial,n.tick,scan.rng_seed(seed,entry,331,k))
        for j,plan in enumerate(options):
            nn=n.fork();hp=h.fork();apply_option(env,spec,nn,t,plan,duration);hp.append(nn.reader.frames)
            totals[j]+=value(env,spec,nn,hp,t,lim,goal,entry,len(nn.reader.frames),[],0,clock)
    return int(totals.argmax())


def select_step(env,spec,n,h,trial,lim,goal,entry,clock,bands,cut,stage,legal,seed):
    values=np.zeros(len(legal))
    for k in range(DRAWS):
        t=conditional(trial,n.tick,scan.rng_seed(seed,n.station,3299,k))
        for j,slot in enumerate(legal):
            values[j]+=value(env,spec,n.fork(),h.fork(),t,lim,goal,entry,cut,bands,stage,clock,int(slot))
    return int(legal[np.argmax(values)])


def case_job(job):
    case,split,outdir=job
    old=json.loads((OLD/f"{case.world_seed}.json").read_text()) if split=="pilot" else None
    env,spec,trial,lim,views,items=get_world(case,old)
    record=dict(world=case.world_seed,layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest(),
                profile=case.profile,requests={})
    for item in items:
        goal,entry,original,h0,_,_,_,_=item
        # The costly probe is studied on the difficult target only; its implementation has no name branch.
        if goal!=req.VESSELS[2]:continue
        available,_=reserve(original,spec,lim);clock=max(3,min(T,int(available//3)+1))
        bridge=trace_plan(env,spec,original,h0,lim,clock,prefer_named=True)
        opts=[None,*entry_trip(env,spec,original,h0,lim,entry,clock)]
        if bridge is not None:opts.append(bridge)
        duration=max([len(p.lengths) for p in opts if p is not None],default=0)
        selected=select_pre(env,spec,original,h0,trial,lim,goal,entry,clock,opts,duration,case.world_seed)
        out={}
        for mode in ("equal","full_teacher"):
            n=original.fork();h=h0.fork()
            choice=opts[selected] if mode=="full_teacher" else None
            apply_option(env,spec,n,trial,choice,duration);h.append(n.reader.frames);cut=len(n.reader.frames)
            bands=[];spans=[];slots=[]
            for j in range(3):
                legal=legal_slots(env,spec,n,lim,clock,3-j)
                slot=(select_step(env,spec,n,h,trial,lim,goal,entry,clock,bands,cut,j,legal,case.world_seed)
                      if mode=="full_teacher" else fixed_slot(env,spec,n,lim,legal))
                start=len(n.reader.frames);advance(env,spec,n,trial,lim,slot,clock);h.append(n.reader.frames)
                bands.append(band(n.reader.frames,h,start));spans.append((start,len(n.reader.frames)));slots.append(slot)
            result=terminal(n,h,entry,cut,bands)
            result.update(trace=None if choice is None else choice.kind,
                          options=[p.kind if p is not None else "hold" for p in opts],
                          ticks=int(n.tick-original.tick),mm=float(n.distance-original.distance),
                          slots=[int(i) for i in slots])
            result["additional_readers"]=observer_outcomes(env,trial,item,(n,h,cut,spans));out[mode]=result
        assert out["equal"]["ticks"]==out["full_teacher"]["ticks"]
        record["requests"][goal]=dict(arms=out)
    (outdir/f"{case.world_seed}.json").write_text(json.dumps(record,separators=(",",":")))
    return case.world_seed


if __name__=="__main__":
    p=ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--split",default="pilot",choices=("pilot","new"))
    p.add_argument("--worlds",type=int,default=64);p.add_argument("--workers",type=int,default=7);args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    cases=list(scan.cases("test",args.worlds,req.SEED if args.split=="pilot" else SEED))
    with ProcessPoolExecutor(args.workers) as pool:
        for i,w in enumerate(pool.map(case_job,[(c,args.split,args.out) for c in cases])):
            if i%12==0:print(i+1,len(cases),w,flush=True)

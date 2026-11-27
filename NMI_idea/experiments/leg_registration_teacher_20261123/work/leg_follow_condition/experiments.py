from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
from copy import copy
import json
import os
from pathlib import Path
import numpy as np

from core import (HERE,GOALS,HORIZON,advance,band,fixed_slot,get_world,grade,legal_slots,lite_scores,
                  pick,public_belief,req,scan,smooth_suffix,state_features,terminal,visit)

SEEDS={"pilot":req.SEED,"train":628495193,"valid":209517619,"test":138426523}
OLD=HERE.parent/"leg_teacher_rebuild/request_pairs"
MODEL=HERE/"models"
DATA=HERE/"train_batches"
SENSOR_DRAWS=10


@lru_cache(maxsize=20)
def actor(file):
    from learning_q import Actor,load
    return Actor(load(file))


def conditional(trial,tick,seed):
    if trial.branch_tick<0 or tick<=trial.branch_tick:
        return trial.branch(tick,seed)
    # A donor execution can itself be a conditional branch. Its observed last noise
    # belongs to its own future, independent of the main evaluation tape.
    from scan_kernel import Tape
    result=copy(trial)
    offset=tick-trial.branch_tick-1
    result.branch_tick=tick
    result.future=Tape.make(seed,trial.profile,len(trial.tape.normal)-tick-1,
                            (trial.future.normal[offset],trial.future.dropout[offset]))
    return result


def short_value(frames,h,begin):
    q=smooth_suffix(h,begin)
    ids=range(begin,len(frames))
    ans=grade(frames,pick(frames,q,ids,begin),ids)
    signal=(ans["correct"]-ans["wrong"])
    true=0.;other=0.
    for f,p in zip(frames[begin:],q):
        pos=f[3]==0
        true=max(true,float(p[0,pos].max(initial=0.)))
        other=max(other,float(p[0,~pos].max(initial=0.)))
    return float(signal+.24*(true-other))


def forecast(env,spec,node,h,trial,limits,goal,stage,clock,legal,fields,draws,seed):
    backup=lite_scores(env,spec,node,h,goal,stage,legal,fields)
    tally=np.zeros(len(legal))
    for k in range(draws):
        branch=conditional(trial,node.tick,scan.rng_seed(seed,int(node.station),k,77))
        for j,slot in enumerate(legal):
            n=node.fork();p=h.fork();before=len(n.reader.frames)
            advance(env,spec,n,branch,limits,slot,clock)
            p.append(n.reader.frames)
            value=short_value(n.reader.frames,p,before)
            if stage<2:
                # Public, posterior-dependent one-step continuation on the same conditional future.
                nxt=legal_slots(env,spec,n,limits,clock,2-stage)
                b=int(nxt[np.argmax(lite_scores(env,spec,n,p,goal,stage+1,nxt,fields))])
                before=len(n.reader.frames)
                advance(env,spec,n,branch,limits,b,clock)
                p.append(n.reader.frames)
                value+=.50*short_value(n.reader.frames,p,before)
            tally[j]+=value
    return tally/draws+.04*backup


def trace_decision(env,spec,node,h,trial,limits,goal,entry,clock,plan,fields,seed,draws):
    if plan is None:return False
    values=np.zeros(2)
    for k in range(draws):
        branch=conditional(trial,node.tick,scan.rng_seed(seed,int(node.station),k,893))
        for move in (0,1):
            n=node.fork();p=h.fork()
            visit(env,spec,n,branch,plan,move);p.append(n.reader.frames)
            # Posterior re-read of the same old album is paired to the following newly acquired band.
            local=[i for i,f in enumerate(n.reader.frames) if entry-req.LOOKBACK-1.e-8<=f[0]<=entry+2.+1.e-8]
            begin=min(local,default=len(n.reader.frames)-1)
            oldp=smooth_suffix(p,begin)
            old=grade(n.reader.frames,pick(n.reader.frames,oldp,local,begin),local)
            legal=legal_slots(env,spec,n,limits,clock,3)
            s=legal[np.argmax(lite_scores(env,spec,n,p,goal,0,legal,fields))]
            start=len(n.reader.frames)
            advance(env,spec,n,branch,limits,s,clock);p.append(n.reader.frames)
            values[move]+=.45*(old["correct"]-old["wrong"])+short_value(n.reader.frames,p,start)
    return bool(values[1]>values[0]+1.e-8)


def base_trace(h,plan):
    if plan is None:return False
    mixture,c,_=public_belief(h)
    return bool(c[0]<.60 or mixture[6]+mixture[7]>.28)


def execute(world,item,mode,seed,return_training=False,random=None,round_idx=0,model_file=None,audit=None):
    env,spec,trial,limits,views=world
    goal,entry,original,h0,kind,first,clock,plan=item
    node=original.fork();h=h0.fork()
    donor=None
    if mode=="yoked":
        d=conditional(trial,original.tick,scan.rng_seed(seed,entry,59033))
        donor=execute((env,spec,d,limits,views),item,"teacher",seed)[0]
        use=donor["trace"]
    elif mode=="legacy":
        use=False
    elif mode in ("teacher","registered_teacher"):
        use=trace_decision(env,spec,node,h,trial,limits,goal,entry,clock,plan,views[1],seed,SENSOR_DRAWS)
    elif mode in ("geometry","registered_public","student","training"):
        use=base_trace(h,plan)
    else:use=False
    if mode!="legacy":
        visit(env,spec,node,trial,plan,use);h.append(node.reader.frames)
    cut=len(node.reader.frames)
    stats=[];choices=[];transitions=[];times=[]
    distance=node.distance
    for stage in range(HORIZON):
        legal=legal_slots(env,spec,node,limits,clock,HORIZON-stage)
        x=state_features(env,spec,node,h,goal,stage,legal)
        if mode in ("legacy","equal"):
            slot=fixed_slot(env,spec,node,limits,legal)
        elif mode=="yoked":
            slot=donor["slots"][stage]
            if slot not in legal:raise RuntimeError("Donor path unexecutable on identical geometry")
        elif mode in ("registered_teacher","registered_public"):
            from observers import reader,geometry_history
            est=reader(env).estimate(node.reader.frames,anatomy=True)
            hp=geometry_history(env,est.centers,node.reader.frames)
            posterior_fields=req.view_fields(env,est.centers)
            if mode=="registered_teacher":
                q=forecast(env,spec,node,hp,trial,limits,goal,stage,clock,legal,posterior_fields,SENSOR_DRAWS,seed)
            else:
                q=lite_scores(env,spec,node,hp,goal,stage,legal,posterior_fields)
            slot=legal[np.argmax(q)]
        elif mode in ("teacher","teacher_hold"):
            q=forecast(env,spec,node,h,trial,limits,goal,stage,clock,legal,views[1],SENSOR_DRAWS,seed)
            slot=legal[np.argmax(q)]
        elif mode=="geometry":
            q=lite_scores(env,spec,node,h,goal,stage,legal,views[0])
            slot=legal[np.argmax(q)]
        elif mode in ("training","student"):
            network=actor(model_file) if model_file else None
            q=(network(x) if network is not None else lite_scores(env,spec,node,h,goal,stage,legal,views[0]))
            if mode=="training" and random.random()<(.60,.40,.25,.17)[min(round_idx,3)]:
                slot=random.choice(legal)
            else:slot=legal[np.argmax(q)]
        else:raise ValueError(mode)
        if return_training:
            rows=[]
            # Full-action collection is training-only. The move driving the next state was locked above.
            for t in legal:
                n=node.fork();hp=h.fork();old=len(n.reader.frames)
                advance(env,spec,n,trial,limits,t,clock);hp.append(n.reader.frames)
                scored=band(n.reader.frames,hp,old)
                reward=scored["correct"]-scored["wrong"]
                nx=np.zeros((5,x.shape[1]),np.float32);mask=np.zeros(5,bool)
                if stage<2:
                    nxt=legal_slots(env,spec,n,limits,clock,2-stage)
                    nx[:len(nxt)]=state_features(env,spec,n,hp,goal,stage+1,nxt)
                    mask[:len(nxt)]=True
                else:
                    out=terminal(n,hp,entry,cut,[*stats,scored])
                    reward+=.55*(out["fresh"]["correct"]-out["fresh"]["wrong"])+.40*out["chain"]
                    reward+=.25*(out["entry"]["correct"]-out["entry"]["wrong"])
                rows.append((float(reward),nx,mask))
            transitions.extend((x[j],r,nx,m,stage) for j,(r,nx,m) in enumerate(rows))
        start=len(node.reader.frames)
        advance(env,spec,node,trial,limits,slot,None if mode=="legacy" else clock)
        h.append(node.reader.frames)
        stats.append(band(node.reader.frames,h,start))
        choices.append(int(slot));times.append([int(start),len(node.reader.frames)])
    result=terminal(node,h,entry,cut,stats)
    if mode=="legacy" and audit is not None:
        original_out=audit["actual"][first]
        for key,r in zip(("entry","fresh"),original_out["answers"]):
            if not np.isclose(result[key]["conf"],r["p"]) or result[key]["role"]!=r["role"]:
                raise RuntimeError("Common old prefix or terminal reader disagreed with archived result")
    last=node.reader.frames[-1]
    assert node.station==original.station+HORIZON and node.tick<=limits.ticks and node.distance<=limits.mm+1.e-6
    assert abs(last[0]-node.station)<1.e-8 and abs((last[1]-node.sector+12.)%24-12.)<1.e-8
    result.update(slots=choices,ticks=int(node.tick-original.tick),mm=float(node.distance-original.distance),
                  sectors=[int(h.samples[j-1].sector)%24 for _,j in times],trace=bool(use),
                  proposed=None if plan is None else dict(kind=plan.kind,station=plan.station,score=plan.score,
                                                          rival=plan.rival,ticks=len(plan.lengths)))
    return result,transitions,(node,h,cut,times)


def train_world(args):
    case,directory,round_idx,model=args
    env,spec,trial,limits,views,items=get_world(case)
    rng=np.random.default_rng(scan.rng_seed(case.world_seed,round_idx,379))
    rows=[]
    for item in items:
        _,data,_=execute((env,spec,trial,limits,views),item,"training",case.world_seed,True,rng,round_idx,model)
        rows.extend(data)
    x,r,nx,m,st=zip(*rows)
    layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    np.savez_compressed(directory/f"{case.world_seed}.npz",x=x,reward=np.array(r,np.float32),next=nx,mask=m,
                        stage=np.array(st,np.int8),layout=layout,world=np.uint64(case.world_seed))
    return case.world_seed


def evaluate(args):
    case,split,output,modes,reader=args
    old=json.loads((OLD/f"{case.world_seed}.json").read_text()) if split=="pilot" else None
    env,spec,trial,limits,views,items=get_world(case,old)
    layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest()
    w=dict(world=case.world_seed,shape=int(case.shape),profile=case.profile,layout=layout,requests={})
    for i,item in enumerate(items):
        name=item[0];history_kind=item[4];first=item[5];plan=item[-1]
        ws=dict(kind=history_kind,first=first,station=item[1],clock=item[6],arms={})
        for mode in modes:
            actual="student" if mode.startswith("rl_") else mode
            file=str(MODEL/f"{mode[3:]}.pt") if actual=="student" else None
            r,_,extra=execute((env,spec,trial,limits,views),item,actual,case.world_seed,
                              model_file=file,audit=old["requests"][name] if old else None)
            if reader:
                from observers import observer_outcomes
                r["additional_readers"]=observer_outcomes(env,trial,item,extra)
            ws["arms"][mode]=r
        w["requests"][name]=ws
    (output/f"{case.world_seed}.json").write_text(json.dumps(w,separators=(",",":")))
    return case.world_seed


def main():
    p=ArgumentParser();p.add_argument("--split",choices=SEEDS,required=True);p.add_argument("--size",type=int,default=64)
    p.add_argument("--workers",type=int,default=10);p.add_argument("--round",type=int,default=0)
    p.add_argument("--model");p.add_argument("--modes",nargs="+",default=["legacy","equal","geometry","teacher"])
    p.add_argument("--reader",action="store_true");p.add_argument("--output",type=Path,required=True)
    opts=p.parse_args();opts.output.mkdir(exist_ok=True,parents=True)
    seed=SEEDS[opts.split]+opts.round*75741
    cases=list(scan.cases("train" if opts.split=="train" else "test",opts.size,seed))
    if opts.split=="train":
        jobs=[(c,opts.output,opts.round,opts.model) for c in cases];fn=train_world
    else:
        jobs=[(c,opts.split,opts.output,opts.modes,opts.reader) for c in cases];fn=evaluate
    with ProcessPoolExecutor(opts.workers) as pool:
        for i,w in enumerate(pool.map(fn,jobs)):
            if i%12==0:print(i+1,len(jobs),w,flush=True)
if __name__=="__main__":main()

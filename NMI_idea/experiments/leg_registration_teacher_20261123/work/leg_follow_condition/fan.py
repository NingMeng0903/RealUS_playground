"""Subsector acquisition at each material station, with exactly matched acoustic time."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np

from core import (HERE,advance,band,fixed_slot,get_world,legal_slots,public_belief,req,scan,
                  terminal,trace_plan,visit)
from experiments import OLD,conditional,short_value,trace_decision
from observers import geometry_history,observer_outcomes,reader
from anchor import reserve

FAN_SEED=482_576_031
FINE_PHASES=(-1.4,-.85,-.45,0.,.45,.85,1.4)
TICKS=22


@dataclass
class View:
    slot:int
    phase:float
    poses:np.ndarray
    distances:np.ndarray
    dwell:int
    target:float
    separation:float
    difference:float


@lru_cache(maxsize=32000)
def visit_path(env,first,last,s,a,phase):
    if phase==0.:return np.array([[s,a]],float),np.empty(0)
    spec=(None,first,last)
    # Approximate arc-length first, then validate speed and contact on the same interpolated surface.
    points=np.column_stack((np.full(17,s),np.linspace(a,a+phase,17)))
    coordinates=env.scanner.surface.path_world(points[:,0],points[:,1])
    if coordinates is None:return None
    length=np.linalg.norm(np.diff(coordinates,axis=0),axis=1).sum()
    t=max(1,int(np.ceil(1.08*length/(scan.MOTION.MAX_SPEED_MM_S*scan.MOTION.CLOCK_S))))
    for ticks in (t,t+1,t+2):
        out=np.column_stack((np.full(ticks+1,s),np.linspace(a,a+phase,ticks+1)))
        dist=env.scanner._path_metrics(out,spec)
        if dist is not None:return out,dist
    return None


def merit(env,s,angle,predicted,model,rival_weight):
    surf=env.scanner.surface;xy=surf.project(float(s),float(angle),predicted)
    p=surf.project(float(s),float(angle),model)
    xy=np.nan_to_num(xy,nan=1000.);p=np.nan_to_num(p,nan=1000.)
    inside=(np.abs(xy[:,0])<=25.)&(xy[:,1]>=3.)&(xy[:,1]<=80.)
    shade=surf.shadow(float(s),float(angle),xy)
    v=inside*(1.-.96*shade)
    gap=(xy[1:]-xy[:1])/[3.5,2.8]
    merge=np.exp(-.5*np.square(gap).sum(axis=1))
    observed=1.-np.max(v[1:]*merge)
    # Estimate whether a target could itself look more compatible with a competing public role.
    e=np.square((xy[:,None,:]-p[None,:,:])/[2.5,1.9]).sum(axis=-1)
    margin=np.clip((e[:,1:].min(axis=1)-e[:,0])/2.,-30.,30.)
    recognized=1./(1.+np.exp(-margin))
    risk=np.max(v[1:]*recognized[1:])
    return v[0],v[0]*(.35*observed+.65*recognized[0])-.45*risk, float(rival_weight@np.abs(v[0]-v[1:]))


def proposals(env,spec,n,h,lim,clock,true_geometry,registered):
    s=int(n.station);a=int(n.sector)
    legal=legal_slots(env,spec,n,lim,clock,3)
    public=public_belief(h)[0][1:6]+.02;public/=public.sum()
    opts=[]
    for slot in legal:
        dst=int(env.route.sectors[a,slot])
        for phase in FINE_PHASES:
            action=visit_path(env,spec[1],spec[2],s+1,dst,phase)
            if action is None:continue
            out,dist=action
            tick=int(clock-env.route.clock[s,a,slot]-2*len(dist))
            if tick<0:continue
            projected=(n.distance+env.route.distance[s,a,slot]+2*dist.sum()
                       +lim.reserve_mm[s+1-spec[1],dst])
            if projected>lim.mm+1.e-6:continue
            q=merit(env,s+1,(dst+phase)%24,true_geometry[s+1],registered[s+1],public)
            opts.append(View(int(slot),phase,out,dist,tick,*q))
    if not opts:raise RuntimeError("No legal subwindow")
    default=fixed_slot(env,spec,n,lim,legal)
    straight=next(o for o in opts if o.slot==default and o.phase==0.)
    # A short menu covers an isolated target, differential visibility, and two motion directions.
    potential=[straight]
    for direction in (0,-1,1):
        candidates=opts if not direction else [o for o in opts if direction*(env.route.deltas[o.slot]+o.phase)>0.]
        if candidates:
            for metric in ((lambda o:.20*o.target+.80*o.separation+.07*o.difference),
                           (lambda o:.70*o.target+.15*o.separation+.35*o.difference)):
                best=max(candidates,key=lambda o:metric(o)+.002*o.dwell)
                if all((best.phase,best.slot)!=(o.phase,o.slot) for o in potential):potential.append(best)
    return straight,potential


def perform(env,spec,n,t,lim,o):
    start=n.tick
    advance(env,spec,n,t,lim,o.slot,None)
    def pulse(x,a,d):
        n.tick+=1;n.distance+=float(d)
        env.scanner._observe(n,spec,t,float(x),float(a),n.tick)
    for (x,a),d in zip(o.poses[1:],o.distances):pulse(x,a,d)
    for _ in range(o.dwell):pulse(*o.poses[-1],0.)
    for (x,a),d in zip(o.poses[-2::-1],o.distances[::-1]):pulse(x,a,d)
    frame=n.reader.frames[-1]
    assert abs(frame[0]-n.station)<1.e-7 and abs((frame[1]-n.sector+12)%24-12)<1.e-7


def choose(env,spec,n,h,trial,lim,choices,seed,draws):
    estimates=np.zeros(len(choices))
    for i in range(draws):
        t=conditional(trial,n.tick,scan.rng_seed(seed,int(n.station),i,5312))
        for j,o in enumerate(choices):
            copy=n.fork();hp=h.fork();first=len(copy.reader.frames)
            perform(env,spec,copy,t,lim,o);hp.append(copy.reader.frames)
            estimates[j]+=short_value(copy.reader.frames,hp,first)
    return choices[int(np.argmax(estimates/draws+.025*np.array([o.separation for o in choices])))]


def experiment(args):
    case,directory,split=args
    previous=json.loads((OLD/f"{case.world_seed}.json").read_text()) if split=="pilot" else None
    env,spec,trial,lim,views,items=get_world(case,previous)
    row=dict(world=case.world_seed,layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest(),
             profile=case.profile,requests={})
    for item in items:
        goal,entry,original,h0,_,_,_,_=item
        slack,_=reserve(original,spec,lim)
        clock=max(3,min(TICKS,int(slack//3)+1))
        plan=trace_plan(env,spec,original,h0,lim,clock,prefer_named=True)
        actions={}
        for mode in ("equal","fine_teacher","public_fine"):
            n=original.fork();h=h0.fork()
            if mode=="fine_teacher":
                move=trace_decision(env,spec,n,h,trial,lim,goal,entry,clock,plan,views[1],case.world_seed,8)
            else:
                move=(mode=="public_fine" and plan is not None and public_belief(h)[1][0]<.6)
            visit(env,spec,n,trial,plan,move);h.append(n.reader.frames)
            cut=len(n.reader.frames);bands=[];spans=[];selected=[]
            for step in range(3):
                registered=reader(env).estimate(n.reader.frames,anatomy=True).centers
                gh=geometry_history(env,registered,n.reader.frames)
                actual=trial.layout.centers if mode=="fine_teacher" else registered
                ref,options=proposals(env,spec,n,gh,lim,clock,actual,registered)
                if mode=="fine_teacher":o=choose(env,spec,n,gh,trial,lim,options,case.world_seed,10)
                elif mode=="public_fine":
                    o=max(options,key=lambda a:.18*a.target+.75*a.separation+.15*a.difference+.002*a.dwell)
                else:o=ref
                begin=len(n.reader.frames);perform(env,spec,n,trial,lim,o);h.append(n.reader.frames)
                if len(n.reader.frames)-begin!=clock:raise RuntimeError("Clock mismatch")
                bands.append(band(n.reader.frames,h,begin));spans.append((begin,len(n.reader.frames)))
                selected.append([o.slot,o.phase])
            out=terminal(n,h,entry,cut,bands)
            out.update(ticks=n.tick-original.tick,mm=n.distance-original.distance,phase_actions=selected,trace=bool(move),
                       proposed=None if plan is None else dict(kind=plan.kind,station=plan.station,ticks=len(plan.lengths)))
            out["additional_readers"]=observer_outcomes(env,trial,item,(n,h,cut,spans))
            actions[mode]=out
        if len({v["ticks"] for v in actions.values()})!=1:raise RuntimeError("Matched comparison changed clock")
        row["requests"][goal]=dict(arms=actions)
    (directory/f"{case.world_seed}.json").write_text(json.dumps(row,separators=(",",":")))
    return case.world_seed


if __name__=="__main__":
    p=ArgumentParser();p.add_argument("out",type=Path);p.add_argument("--split",choices=("pilot","new"),default="new")
    p.add_argument("--worlds",type=int,default=64);p.add_argument("--workers",type=int,default=6);args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    cases=list(scan.cases("test",args.worlds,req.SEED if args.split=="pilot" else FAN_SEED))
    with ProcessPoolExecutor(args.workers) as pool:
        for i,num in enumerate(pool.map(experiment,[(c,args.out,args.split) for c in cases])):
            if i%12==0:print(i+1,len(cases),num,flush=True)

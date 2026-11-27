"""Prospective bounded reacquisition with or without privileged verification of public nominees.

Every submitted echo must first win a three-pulse public identity comparison. The verifier
receives only a pass/fail bit about that specific nominee. Public modes never receive this bit.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib,json
import numpy as np
from core import advance,get_world,grade,legal_slots,observe_hold,pick,public_belief,req,scan,visit
from observers import reader
from recovery import entry_trip
from anchor import reserve
from fan import proposals,visit_path
from experiments import OLD

SEED=296338463


def nominate(frames,h,indices):
    # Consecutive available pulses are grouped without crossing gaps in the eligible view band.
    indices=list(indices)
    groups=[]
    for i in indices:
        if not groups or len(groups[-1])>=3 or i!=groups[-1][-1]+1:groups.append([])
        groups[-1].append(i)
    out=[]
    for group in groups:
        winner=pick(frames,h.online,group,0)
        if winner is not None:out.append(winner)
    return out


def answer(frames,nominees,mode):
    if mode=="privileged":
        # A real already-acquired public nominee may be certified. A wrong one can never be submitted.
        for p in nominees:
            if p[2]>=.5 and int(frames[p[0]][3][p[1]])==0:return p
        return None
    return max(nominees,key=lambda p:p[2],default=None)


def scoring(frames,choice,indices):
    return grade(frames,choice,indices)


def confident(nominees):
    # Sensor-only stopping: require corroboration by at least two non-overlapping pulse groups.
    return len([p for p in nominees if p[2]>=.80])>=2


def pulse_path(env,spec,n,t,lim,phase):
    s,a=int(n.station),int(n.sector)
    x=visit_path(env,spec[1],spec[2],s,a,phase)
    if x is None:return False
    poses,ds=x
    remain_t,remain_mm=reserve(n,spec,lim)
    if 2*len(ds)+5+7>remain_t or 2*ds.sum()+25>remain_mm:return False
    def pulse(p,length):
        n.tick+=1;n.distance+=float(length)
        env.scanner._observe(n,spec,t,float(p[0]),float(p[1]),n.tick)
    for p,length in zip(poses[1:],ds):pulse(p,length)
    for _ in range(5):pulse(poses[-1],0.)
    for p,length in zip(poses[-2::-1],ds[::-1]):pulse(p,length)
    return True


def public_regrade(env,frames,indices,nominees):
    # Keep the same shortlist and history, change only the proposed identity probabilities.
    probs=reader(env).read(frames)
    p=max(((i,k,float(probs[i][0,k])) for i,k,_ in nominees),key=lambda z:z[2],default=None)
    return grade(frames,p,indices)


def execute_one(world,item,mode):
    env,spec,trial,lim=world
    goal,entry,original,h0,*_=item
    n=original.fork();h=h0.fork()
    duration=[];phases=[]
    local_indices=[i for i,f in enumerate(n.reader.frames) if entry-3.-1.e-8<=f[0]<=entry+2.+1.e-8]
    nominees=nominate(n.reader.frames,h,local_indices)
    def accept(f,p):
        return answer(f,p,"privileged") is not None if mode=="privileged" else confident(p)
    # Reserve time for downstream progress. If needed, revisit the original named material request.
    options=entry_trip(env,spec,n,h,lim,entry,10)
    visit_count=0
    if options and not accept(n.reader.frames,nominees):
        plan=max(options,key=lambda p:p.score)
        start=len(n.reader.frames);visit(env,spec,n,trial,plan,True);h.append(n.reader.frames)
        ix=[i for i in range(start,len(n.reader.frames)) if entry-3.-1.e-8<=n.reader.frames[i][0]<=entry+2.+1.e-8]
        local_indices+=ix;nominees+=nominate(n.reader.frames,h,ix);visit_count=1
    locked=answer(n.reader.frames,nominees,mode)
    entry_ans=scoring(n.reader.frames,locked,local_indices)
    cut=len(n.reader.frames);all_noms=[];bands=[];public_bands=[];num=[]
    for j in range(3):
        start=len(n.reader.frames)
        slack,_=reserve(n,spec,lim)
        clock=max(3,min(9,int(slack/(3-j))+1))
        field=reader(env).estimate(n.reader.frames,anatomy=True).centers
        ref,opts=proposals(env,spec,n,h,lim,clock,field,field)
        q=max(opts,key=lambda o:.18*o.target+.72*o.separation+.16*o.difference)
        # Reach the chosen next window first; retry decisions use only observations already acquired.
        advance(env,spec,n,trial,lim,q.slot,None);h.append(n.reader.frames)
        pending=nominate(n.reader.frames,h,range(start,len(n.reader.frames)))
        used=[]
        for phase in (0.,-.70,.70,-1.15,1.15):
            if accept(n.reader.frames,pending):break
            b=len(n.reader.frames)
            if pulse_path(env,spec,n,trial,lim,phase):
                h.append(n.reader.frames);pending+=nominate(n.reader.frames,h,range(b,len(n.reader.frames)));used.append(phase)
        ids=range(start,len(n.reader.frames))
        c=answer(n.reader.frames,pending,mode)
        bands.append(scoring(n.reader.frames,c,ids))
        public_bands.append(public_regrade(env,n.reader.frames,ids,pending))
        num.append(len(pending));phases.append(used);duration.append(len(ids))
        all_noms+=pending
    fresh=grade(n.reader.frames,answer(n.reader.frames,all_noms,mode),range(cut,len(n.reader.frames)))
    public=[public_regrade(env,n.reader.frames,local_indices,nominees),
            public_regrade(env,n.reader.frames,range(cut,len(n.reader.frames)),all_noms)]
    def out(e,f,b):
        return dict(entry=e,fresh=f,both=int(e["correct"] and f["correct"]),chain=int(all(a["correct"] for a in b)),
            prefix_mm=5*next((i for i,a in enumerate(b) if not a["correct"]),3),
            capture_both=int(e["captured"] and f["captured"]),capture_chain=int(all(a["captured"] for a in b)))
    result=out(entry_ans,fresh,bands)
    result.update(ticks=n.tick-original.tick,mm=n.distance-original.distance,entry_visit=visit_count,
        public_on_identical_path=out(*public,public_bands),bands=bands,counts=num,phases=phases,ticks_per_band=duration)
    assert n.station==original.station+3 and n.distance<=lim.mm+1.e-6 and n.tick<=lim.ticks
    return result


def one(args):
    case,outdir,old=args
    baseline=json.loads((OLD/f"{case.world_seed}.json").read_text()) if old else None
    env,spec,t,lim,_,items=get_world(case,baseline)
    rec=dict(world=case.world_seed,profile=case.profile,layout=hashlib.sha256(np.nan_to_num(t.layout.centers).tobytes()+
              bytes([case.shape])).hexdigest(),requests={})
    for it in items:
        rec["requests"][it[0]]=dict(arms={k:execute_one((env,spec,t,lim),it,k) for k in ("privileged","public")})
    (outdir/f"{case.world_seed}.json").write_text(json.dumps(rec,separators=(",",":")))
    return case.world_seed


if __name__=="__main__":
    p=ArgumentParser();p.add_argument("directory",type=Path);p.add_argument("--size",type=int,default=64)
    p.add_argument("--split",choices=("pilot","new"),default="pilot");p.add_argument("--workers",type=int,default=6)
    a=p.parse_args();a.directory.mkdir(exist_ok=True,parents=True)
    cases=scan.cases("test",a.size,req.SEED if a.split=="pilot" else SEED)
    with ProcessPoolExecutor(a.workers) as pool:
        for i,id in enumerate(pool.map(one,[(c,a.directory,a.split=="pilot") for c in cases])):
            if i%12==0:print(i+1,a.size,id,flush=True)

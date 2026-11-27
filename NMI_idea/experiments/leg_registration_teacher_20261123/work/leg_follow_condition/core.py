"""Closed-loop steering of the common contact scanner after a named-vessel request."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"leg_teacher_rebuild"))
import request_benchmark as req
from corridors import nominal_fields, observe_hold, path, scaffold, scan, view_fields
from environment import spec_for
from history import History
from anchor import edge_poses, reserve

GOALS=req.VESSELS
DELTAS=(0,-1,1,-2,2)
CLOCK=12
HORIZON=3


def smooth_suffix(history,begin):
    """Exactly the original backward association on just the portion to be re-read."""
    mean,var=history.reset()
    previous=None
    out=[None]*(len(history.samples)-begin)
    for j in range(len(history.samples)-1,begin-1,-1):
        f=history.samples[j]
        mb,vb=history.predict(f,previous,mean,var)
        mf,vf=history.predictions[j]
        iv=np.maximum(1./vf+1./vb-1./history.base_var,1./(2.*history.base_var))
        va=1./iv
        out[j-begin]=history.probability(f,va*(mf/vf+mb/vb),va)
        mean,var,_=history.update(f,mb,vb)
        previous=f
    return out


def pick(frames,probs,indices,begin,role=0):
    winner=None
    for i in indices:
        q=probs[i-begin]
        if q.shape[1]:
            k=int(np.argmax(q[role]))
            if winner is None or q[role,k]>winner[2]:
                winner=(i,k,float(q[role,k]))
    return winner


def grade(frames,choice,indices):
    truth=None if choice is None else int(frames[choice[0]][3][choice[1]])
    p=0. if choice is None else float(choice[2])
    return dict(conf=p,role=truth,correct=int(truth==0 and p>=.5),
                wrong=int(truth is not None and truth!=0 and p>=.5),
                captured=int(any(np.any(frames[i][3]==0) for i in indices)))


def band(frames,h,start):
    ids=range(start,len(frames))
    return grade(frames,pick(frames,smooth_suffix(h,start),ids,start),ids)


def terminal(node,h,entry,cut,bands,other=None):
    f=node.reader.frames
    local=[i for i in range(cut) if entry-req.LOOKBACK-1.e-8<=f[i][0]<=entry+2.+1.e-8]
    distant=list(range(cut,len(f)))
    lo=min(local,default=cut)
    p=smooth_suffix(h,lo) if other is None else other[lo:]
    ans=[grade(f,pick(f,p,ids,lo),ids) for ids in (local,distant)]
    return dict(entry=ans[0],fresh=ans[1],both=int(all(a["correct"] for a in ans)),
                chain=int(all(a["correct"] for a in bands)),
                prefix_mm=5*next((i for i,a in enumerate(bands) if not a["correct"]),len(bands)),
                capture_chain=int(all(a["captured"] for a in bands)),
                capture_both=int(all(a["captured"] for a in ans)),bands=bands)


def public_belief(hist):
    # The role marginals are computed from anonymous echoes and the public atlas only.
    mixture=np.zeros(8);target=[];margins=[];offset=np.zeros(2);weight=0.
    for sample,q in zip(hist.samples[-12:],hist.online[-12:]):
        if not q.shape[1]:
            target.append(0.)
            continue
        k=int(q[0].argmax());v=q[:,k]
        mixture+=v;target.append(v[0]);margins.append(v[0]-np.max(v[1:]))
        if np.isfinite(sample.points[0]).all():
            err=sample.echoes[k]-sample.points[0]
            _,normal,tangent=hist.env.scanner.surface.pose(sample.station,sample.sector%24)
            offset+=v[0]*(err[0]*tangent-err[1]*normal)
            weight+=v[0]
    mixture/=max(mixture.sum(),1.e-6)
    return mixture,np.array([np.mean(target) if target else 0.,max(target,default=0.),
                           np.mean(margins) if margins else 0.,np.mean(np.array(target)<.35) if target else 1.]),offset/max(weight,1.)


@dataclass
class Landmark:
    poses: np.ndarray
    lengths: np.ndarray
    kind: str
    rival: int
    score: float
    station: int


@lru_cache(maxsize=8)
def named_nodes(env):
    atlas=json.loads((scan.ATLAS.parent/"atlas_routes_manifest.json").read_text())
    bridges=atlas["sources"]["Open3DModel right"]["bridge_edges"]
    return {int(np.argmin(np.abs(env.scanner.stations+e["right_station_mm"]))) for e in bridges}


def trace_plan(env,spec,node,h,limits,clock,prefer_named=False):
    """Look for an observed/predicted change in the relationship of two public hypotheses."""
    s,a=int(node.station),int(node.sector)
    fields=nominal_fields(env);positions=env.scanner._anatomy.centers[2]
    prob,summary,_=public_belief(h)
    weights=prob[1:6].copy()
    dist=np.linalg.norm(positions[s,1:]-positions[s,0],axis=-1)
    proximity=np.exp(-.5*(np.nan_to_num(dist,nan=100.)/6.)**2)
    weights=(weights+.06)*proximity
    if weights.sum()<.00001:return None
    rival=int(np.argmax(weights))+1
    current=positions[s,0]-positions[s,rival]
    links=env.scanner._anatomy.links[2]
    tick_slack,mm_slack=reserve(node,spec,limits)
    max_ticks=min(100 if prefer_named else 72,int(tick_slack-3*clock-3))
    max_mm=min(250.,mm_slack-28.) if prefer_named else min(180.,mm_slack-110.)
    if max_ticks<7 or max_mm<30.:return None
    proposals=[]
    reach=25 if prefer_named else 16
    for anchor in range(max(spec[1],s-reach),min(spec[2],s+reach)+1):
        distance=abs(anchor-s)
        if distance<3 or 10.*distance>max_mm+5.:continue
        lo,hi=sorted((anchor,s))
        if not np.all(links[lo:hi,0]):continue
        connected_pair=bool(np.all(links[lo:hi,rival]) and np.isfinite(positions[anchor,rival]).all())
        is_named=anchor in named_nodes(env)
        if not is_named and not connected_pair:continue
        change=(np.linalg.norm((positions[anchor,0]-positions[anchor,rival])-current)
                if connected_pair else 0.)
        # Only an atlas-supported bridge is considered a topological landmark.
        # Mere lack of a competitor annotation is never positive identity evidence.
        if not is_named and change<1.:continue
        for turn in (0,-1,1):
            goal=(a+turn)%24
            angles=(a+np.rint(np.linspace(0.,turn,distance+1)).astype(int))%24
            sectors=angles if anchor>s else angles[::-1]
            route=path(env,lo,sectors)
            if route is None:continue
            ticks=2*route.ticks+2
            if ticks>max_ticks or 2*route.distance>max_mm:continue
            merit=(float(is_named)*1.25+min(3.,change/2.))*fields[anchor,goal,0]-.006*ticks
            if merit>0:proposals.append((merit,anchor,float(change),is_named,sectors))
    if not proposals:return None
    ranked=[p for p in proposals if p[3]] if prefer_named else []
    score,anchor,_,is_named,sectors=max(ranked or proposals,key=lambda x:x[0])
    lo=min(s,anchor)
    positions=edge_poses(env,spec,lo,sectors)
    outward=positions if anchor>s else positions[::-1]
    reverse=outward[::-1].copy()
    poses=np.vstack((outward,np.repeat(outward[-1:],2,axis=0),reverse[1:]))
    lengths=env.scanner._path_metrics(poses,spec)
    if lengths is None or sum(lengths)>max_mm+1.e-5 or len(lengths)>max_ticks:return None
    return Landmark(poses,lengths,"named_bridge" if is_named else "geometry",rival,float(score),anchor)


def visit(env,spec,node,trial,plan,move):
    if plan is None:return
    if not move:
        observe_hold(env,node,spec,trial,len(plan.lengths));return
    s,a=node.station,node.sector
    for (x,y),length in zip(plan.poses[1:],plan.lengths,strict=True):
        node.tick+=1;node.distance+=float(length)
        env.scanner._observe(node,spec,trial,float(x),float(y),node.tick)
    last=node.reader.frames[-1]
    assert abs(last[0]-s)<1.e-7 and abs((last[1]-a+12.)%24.-12.)<1.e-7


def initial_action(env,spec,prefix,trial,limits,choice):
    typ,value=choice
    node=prefix.fork()
    if typ=="route":
        req.apply(env,spec,node,trial,limits,value,length=req.PROGRESS)
    else:
        motion,ref,slack=value
        node=scan.run_action(env,spec,prefix,trial,limits,motion)
        spent=node.tick-prefix.tick
        if spent>slack:raise RuntimeError("Unbudgeted prefix")
        observe_hold(env,node,spec,trial,slack-spent)
        for slot in ref.slots:scan.ref.reference.apply_route_slot(env,spec,node,trial,limits,int(slot))
    return node


def get_world(case,record=None):
    env=scan.ref.reference.load_environment(str(scan.ATLAS),case.shape)
    spec=spec_for(env.scanner,case)
    initial,trial,limits=env.start(case,spec)
    found=req.public_requests(env,case,spec,initial,trial,limits,req.boundaries(env))
    views=(nominal_fields(env),view_fields(env,trial.layout.centers))
    result=[]
    for goal,(prefix,h,kind,p) in found.items():
        bank=req.actions(env,spec,prefix,trial,limits,views)
        action=(record["requests"][goal]["selected"]["1.0"] if record is not None else
                "backtrack" if kind!="confirmed" and "backtrack" in bank else "continue")
        node=initial_action(env,spec,prefix,trial,limits,bank[action])
        h.append(node.reader.frames)
        remaining,_=reserve(node,spec,limits)
        clock=max(3,min(CLOCK,int(remaining//HORIZON)+1))
        plan=trace_plan(env,spec,node,h,limits,clock)
        result.append((goal,int(prefix.station),node,h,kind,action,clock,plan))
    return env,spec,trial,limits,views,result


def legal_slots(env,spec,node,limits,clock,left):
    r=env.route;s,a=int(node.station),int(node.sector)
    mask=r.bounded_candidates(node,spec,limits)
    # Reserve contact distance to finish the original whole route as well as time for the local clock.
    reserve_ticks=limits.reserve_ticks[s+1-spec[1],r.sectors[a]]
    mask &= (r.clock[s,a]<=clock) & (node.tick+clock+reserve_ticks<=limits.ticks)
    slots=np.flatnonzero(mask)
    if not len(slots):raise RuntimeError(f"No contact-legal slot at {(s,a,clock,left)}")
    return slots


def advance(env,spec,node,trial,limits,slot,clock):
    start=node.tick
    scan.ref.reference.apply_route_slot(env,spec,node,trial,limits,int(slot))
    if clock is not None:
        extra=clock-(node.tick-start)
        if extra<0:raise RuntimeError("Move exceeds equal acquisition period")
        observe_hold(env,node,spec,trial,extra)


def fixed_slot(env,spec,node,limits,legal):
    s,a=int(node.station),int(node.sector)
    spine=scaffold(env,s,spec[2],a)
    slot=int(path(env,s,spine[:2]).slots[0])
    if slot not in legal:
        return int(min(legal,key=lambda k:abs((int(env.route.sectors[a,k])-spine[1]+12)%24-12)))
    return slot


def state_features(env,spec,node,h,goal,stage,legal):
    s,a=int(node.station),int(node.sector)
    mixture,confidence,residual=public_belief(h)
    angle=a*np.pi/12.
    xy=s/160.
    pos=np.r_[xy,np.sin(angle),np.cos(angle),np.sin(2*np.pi*xy),np.cos(2*np.pi*xy),
              np.sin(4*np.pi*xy),np.cos(4*np.pi*xy),stage/2.]
    name=np.asarray([g==goal for g in GOALS],float)
    history=np.r_[mixture,confidence,np.clip(residual/6.,-2.,2.)]
    obs=[]
    for slot in legal:
        nxt=int(env.route.sectors[a,slot]);d=DELTAS[int(slot)]
        base=np.r_[d/2.,abs(d)/2.,np.sin(nxt*np.pi/12.),np.cos(nxt*np.pi/12.),
                   env.route.clock[s,a,slot]/15.,env.route.distance[s,a,slot]/40.]
        projected,shade=env.scanner._anatomy.truth(env.scanner.surface,2,float(s+1),float(nxt))
        safe=np.nan_to_num(projected,nan=120.)
        visible=(np.abs(safe[:,0])<=25.)&(safe[:,1]>=3.)&(safe[:,1]<=80.)
        detect=visible*(1.-.96*shade)
        diff=np.clip((safe[1:]-safe[:1])/[6.,6.],-3.,3.)
        # The filter's material residual predicts where the requested and competing contours may appear.
        _,n,t=env.scanner.surface.pose(float(s+1),float(nxt))
        shifted=safe[0]+np.array([residual@t,-residual@n])
        rival=mixture[1:6]+.01
        rival=rival/rival.sum()
        kernel=np.exp(-.5*np.square(diff).sum(axis=1))
        conflict=rival@(detect[1:]*kernel)
        paired=np.r_[conflict,rival@np.abs(detect[0]-detect[1:]),rival@diff,shifted/[25.,60.]]
        atlas=np.r_[np.clip(safe[0]/[25.,60.],-2.,2.),detect,diff.ravel()]
        obs.append(np.r_[pos,name,history,base,atlas,paired])
    return np.nan_to_num(np.array(obs),nan=0.).astype(np.float32)


def lite_scores(env,spec,node,h,goal,stage,legal,fields):
    x=state_features(env,spec,node,h,goal,stage,legal)
    a=int(node.sector);s=int(node.station)
    destinations=env.route.sectors[a,legal]
    v=fields[s+1,destinations]
    return .42*v[:,0]+.50*v[:,1]+.12*v[:,2]-.20*x[:,-6]+.08*x[:,-5]-.004*np.abs(env.route.deltas[legal])

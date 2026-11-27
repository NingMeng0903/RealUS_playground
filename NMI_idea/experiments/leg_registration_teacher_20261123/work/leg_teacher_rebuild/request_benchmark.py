"""Named-vessel requests with evidence revision, a second fresh view and prospective action values."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from closed_teacher import apply, menu
from corridors import nominal_fields, observe_hold, path, scaffold, scan, view_fields
from environment import spec_for
from request_objective import two_view_score, utility

SEED = 375_928_537
VESSELS = ("Femoral_vein.r","Popliteal_vein.r","Posterior_tibial_vein.r")
LOOKBACK, PROGRESS, FRESH = 3, 6, 3
LAMBDAS = (.5,1.,2.)


def boundaries(env):
    document=json.loads((scan.ATLAS.parent/"atlas_routes_manifest.json").read_text())
    records=document["routes"]["groin_to_distal_calf"]["per_source"]["Open3DModel right"]["station_records"]
    allowed={}
    for r in records:
        # Requests may be issued on individually named, supported stations only.
        if r["supported"] and not r["synthetic"] and r["target_name"] in VESSELS:
            s=int(np.argmin(np.abs(env.scanner.stations+r["station_mm"])))
            allowed[s]=r["target_name"]
    return allowed


def prefix_album(frames,s):
    return [i for i,f in enumerate(frames) if s-LOOKBACK-1.e-8<=f[0]<=s+1.e-8]


def best(frames,posterior,ids):
    # Return a public decision first. Access the evaluator's origin in a separate function.
    candidate=None
    for i in ids:
        p=posterior[i]
        if not p.shape[1]: continue
        k=int(np.argmax(p[0]));value=float(p[0,k])
        if candidate is None or candidate[2]<value:
            candidate=(i,k,value)
    return candidate


def truth_of(frames,choice):
    return None if choice is None else int(frames[choice[0]][3][choice[1]])


def state_summary(history,frames,s):
    ids=prefix_album(frames,s)
    winner=best(frames,history.online,ids)
    p=0. if winner is None else winner[2]
    if p<.35: kind="lost"
    elif p<.82: kind="ambiguous"
    else: kind="confirmed"
    return kind,p


def public_requests(env,case,spec,node,trial,limits,allowed):
    history=scan.public_context(env)[0]
    # The invitation time is drawn independently of the subject's anatomy and echoes.
    rng=np.random.default_rng(scan.rng_seed(case.action_seed,18_391))
    windows={}
    for name in VESSELS:
        legal=[s for s in sorted(allowed) if all(allowed.get(j)==name for j in range(s-LOOKBACK,s+PROGRESS+FRESH+1))]
        if not legal: raise RuntimeError("Missing named request support")
        # A randomized six-site invitation avoids always presenting problems at the same end of the vessel.
        start=int(rng.integers(len(legal)))
        group=legal[start:start+6]
        windows[name]=set(group)
    found={}
    while node.station<=spec[2]:
        s=int(node.station)
        name=allowed.get(s)
        if name in windows and s in windows[name] and name not in found:
            history.append(node.reader.frames)
            kind,conf=state_summary(history,node.reader.frames,s)
            # First uncertain public prefix after invitation; confirmed controls remain represented.
            if kind!="confirmed" or s==max(windows[name]):
                found[name]=(node.fork(),history.fork(),kind,conf)
        if len(found)==len(VESSELS) or s==spec[2]:break
        env.route.advance(node,spec,trial,limits)
    if len(found)!=3: raise RuntimeError("Failed prospective request collection")
    return found


def actions(env,spec,prefix,trial,limits,views):
    offered=menu(env,spec,prefix,limits,views,0,width=PROGRESS,remaining_options=1)
    actions={k:("route",v) for k,v in offered.items()}
    if "reference" not in offered:return actions
    reference,_,slack=offered["reference"]
    caps=limits.probe(prefix,spec)
    lower,s=spec[1],int(prefix.station)
    tip=int(reference.sectors[-1])
    available_mm=limits.mm-prefix.distance-limits.reserve_mm[s+PROGRESS-lower,tip]-reference.distance
    for name,skill,controls in (("backtrack",2,(0.,1.,0.)),("cross_window",1,(.72,0.,-.25))):
        plan=env.plan(prefix,spec,skill,np.asarray(controls),caps,scan.ARM)
        if plan is not None and plan.ticks<=slack and plan.distance<=available_mm:
            actions[name]=("probe",(scan.Action(name,"disambiguate",plan.ticks,plan.distance,plan,
                                tuple(plan.goal),controls),reference,slack))
    return actions


def run(env,spec,prefix,trial,limits,option):
    kind,value=option
    node=prefix.fork()
    if kind=="route":
        apply(env,spec,node,trial,limits,value,length=PROGRESS)
    else:
        motion,ref,slack=value
        node=scan.run_action(env,spec,prefix,trial,limits,motion)
        consumed=node.tick-prefix.tick
        if consumed>slack: raise RuntimeError("Probe exceeded paired time")
        observe_hold(env,node,spec,trial,slack-consumed)
        for slot in ref.slots:
            scan.ref.reference.apply_route_slot(env,spec,node,trial,limits,int(slot))
    cut=len(node.reader.frames)
    anchor=[int(node.station),int(node.sector),int(node.tick)]
    # All matched options are evaluated with the exact same public downstream scan.
    # Plain MPC may have arrived at a different sector or time; it gets the same continuation rule.
    s=int(node.station)
    spine=scaffold(env,s,spec[2],int(node.sector))
    tail=path(env,s,spine[:FRESH+1])
    for slot in tail.slots:scan.ref.reference.apply_route_slot(env,spec,node,trial,limits,int(slot))
    if node.station!=prefix.station+PROGRESS+FRESH or node.tick>limits.ticks or node.distance>limits.mm+1.e-6:
        raise RuntimeError("Named request failed physical completion")
    last=node.reader.frames[-1]
    if abs(last[0]-node.station)>1.e-7 or abs((last[1]-node.sector+12.)%24.-12.)>1.e-7:
        raise RuntimeError("Final physical frame is inconsistent")
    return node.reader.frames,cut,anchor,int(node.tick-prefix.tick),float(node.distance-prefix.distance)


def identity_brier(frames,post,locked):
    # The same entry echoes are scored for every action. -2 is merged, -1 clutter.
    if not locked:return None
    scores=[]
    for i,k in locked:
        q=post[i][:,k]
        label=int(frames[i][3][k]); label=6 if label==-2 else 7 if label==-1 else label
        target=np.zeros(8);target[label]=1.
        scores.append(.5*float(np.square(q-target).sum()))
    return float(np.mean(scores))


def decisions(frames,posterior,sets):
    output=[]
    for ids in sets:
        winner=best(frames,posterior,ids)
        capture=any(np.any(frames[i][3]==0) for i in ids)
        output.append(dict(p=0. if winner is None else winner[2],role=truth_of(frames,winner),
                           capture=int(capture),count=sum(len(frames[i][2]) for i in ids)))
    return output


def assess(frames,cut,s,history,locked,baseline_brier):
    h=history.fork();h.append(frames);post=h.smoothed()
    # A proximal requested image may be recovered in the entry archive or in newly obtained views.
    local=[i for i in range(cut) if s-LOOKBACK-1.e-8<=frames[i][0]<=s+2.+1.e-8]
    fresh=list(range(cut,len(frames)))
    answers=decisions(frames,post,[local,fresh])
    brier=identity_brier(frames,post,locked)
    return dict(answers=answers,brier=brier,delta_brier=None if brier is None else baseline_brier-brier)


def select_with_feedback(values,penalty,controls):
    def score(arm):
        rows=values[arm]
        u=np.mean([two_view_score(r["answers"],penalty) for r in rows])
        improvements=[r["delta_brier"] for r in rows if r["delta_brier"] is not None]
        return (round(float(u),10),round(float(np.mean(improvements)),6) if improvements else 0.,
                -float(np.mean([r["ticks"] for r in rows])))
    return max(controls,key=score)


def digest_suffix(frames,cut):
    h=hashlib.sha256()
    for f in frames[cut:]:
        for value in (f[0],f[1],f[2],f[3],f[4],f[8]):h.update(np.asarray(value).tobytes())
    return h.hexdigest()


def experiment(job):
    case,draws,directory=job
    env=scan.ref.reference.load_environment(str(scan.ATLAS),case.shape)
    spec=spec_for(env.scanner,case)
    initial,trial,limits=env.start(case,spec)
    found=public_requests(env,case,spec,initial.fork(),trial,limits,boundaries(env))
    views=(nominal_fields(env),view_fields(env,trial.layout.centers))
    record=dict(world=case.world_seed,shape=int(case.shape),profile=case.profile,requests={},
        layout=hashlib.sha256(np.nan_to_num(trial.layout.centers).tobytes()+bytes([case.shape])).hexdigest())
    for name,(prefix,history,kind,confidence) in found.items():
        s=int(prefix.station);start=len(prefix.reader.frames)
        locked=[(i,int(history.online[i][0].argmax())) for i in prefix_album(prefix.reader.frames,s)
                if history.online[i].shape[1]]
        base_post=history.smoothed()
        brier=identity_brier(prefix.reader.frames,base_post,locked)
        choices=actions(env,spec,prefix,trial,limits,views)
        values={k:[] for k in choices};actual={}
        for i in range(1,draws+1):
            common=trial.branch(prefix.tick,scan.rng_seed(case.world_seed,s,i,64102))
            for k,option in choices.items():
                f,cut,pos,ticks,mm=run(env,spec,prefix,common,limits,option)
                r=assess(f,cut,s,history,locked,brier)
                r.update(ticks=ticks);values[k].append(r)
        public_actions=[k for k in choices if k not in ("individual","pair")]
        # All choices are locked before the actual outcomes are read.
        selected={str(p):select_with_feedback(values,p,list(choices)) for p in LAMBDAS}
        common_only=select_with_feedback(values,1.,public_actions)
        checks={}
        for k,option in choices.items():
            f,cut,pos,ticks,mm=run(env,spec,prefix,trial,limits,option)
            r=assess(f,cut,s,history,locked,brier)
            r.update(ticks=ticks,mm=mm,anchor=pos)
            if k!="continue":
                checks[k]=(pos,digest_suffix(f,cut))
                masked=prefix.reader.frames+f[cut:]
                r["masked"]=assess(masked,start,s,history,locked,brier)
            actual[k]=r
        if checks and any(value!=next(iter(checks.values())) for value in checks.values()):
            raise RuntimeError("Paired subprograms changed new downstream images or timing")
        record["requests"][name]=dict(kind=kind,station=s,mm=-float(env.scanner.stations[s]),
            prefix_peak=confidence,locked_count=len(locked),prefix_brier=brier,
            selected=selected,public_menu_teacher=common_only,actual=actual,model=values)
    (directory/f"{case.world_seed}.json").write_text(json.dumps(record,separators=(",",":")))
    return case.identifier


if __name__=="__main__":
    p=ArgumentParser();p.add_argument("directory",type=Path);p.add_argument("--worlds",type=int,default=48)
    p.add_argument("--draws",type=int,default=8);p.add_argument("--workers",type=int,default=12);args=p.parse_args()
    args.directory.mkdir(exist_ok=True,parents=True)
    jobs=[(c,args.draws,args.directory) for c in scan.cases("test",args.worlds,SEED)
          if not (args.directory/f"{c.world_seed}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        for i,name in enumerate(pool.map(experiment,jobs)):
            if i%6==0: print(i+1,"/",len(jobs),name,flush=True)

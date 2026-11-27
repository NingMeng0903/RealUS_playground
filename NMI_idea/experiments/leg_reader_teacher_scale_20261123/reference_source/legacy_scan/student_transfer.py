"""Search, student failure aggregation and history training on CT-derived geometry."""
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse
import gzip
import json

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import scan as env
from geometry import ANGLES, NORMALS, TANGENTS, YS, TASKS, digest, bone_prior
from shape_training import BETAS, CT, SOURCE_NAMES, move_section

WIDTHS = (8, 30)
TARGETS = (0, 1, 2, 3)
REVIEW_LIMIT = 4
RISK_THRESHOLD = .81
GUARD_CONFIDENCE = .88
MIN_GAIN = .025
HISTORY_NEIGHBORS = False
CT_ROUTES = {target: (int(np.where(YS == -530)[0][0]),
                      int(np.where(YS == -555)[0][0])) for target in TARGETS}
TARGET_NAMES = ("deep vein", "femoral artery", "great saphenous", "deep femoral vein")


class TrainingStore(env.Store):
    """Uses measured CT labels and references without opening Blender geometry."""
    def __init__(self, pack, shape):
        self.y = pack["body_stations_mm"][shape]
        self.centers = pack["ct_derived_centers"][shape].copy()
        self.contact = pack["skin_contacts"][shape]
        self.valid = self.contact[:, :, 0] < -1
        self.bone = bone_prior()
        radius, origin = pack["skin_radii"][shape], pack["skin_origins"][shape]
        radius0, origin0 = pack["reference_radii"], pack["reference_origins"]
        offsets = np.arange(-25., 25.1, 2.5)
        depths = np.arange(4., 80.1, 2.)
        self.depth_count = len(depths)
        self.slew = np.linalg.norm(self.contact-np.roll(self.contact,-1,axis=1),axis=-1)
        self.ray = np.empty((len(YS),len(ANGLES),len(offsets),len(depths)),dtype=np.float32)
        for s,y in enumerate(YS):
            aperture = self.contact[s,:,None,None]+TANGENTS[:,None,None]*offsets[None,:,None,None]
            rays = aperture-NORMALS[:,None,None]*depths[None,None,:,None]
            ref = move_section(rays,origin,radius,origin0,radius0,s)
            occ = self.bone(np.stack((ref[...,0],np.full(ref.shape[:-1],y),ref[...,1]),axis=-1))
            occ = np.where(np.isfinite(occ),occ,.15)
            self.ray[s] = np.maximum.accumulate(occ,axis=-1)

    @lru_cache(maxsize=8000)
    def truth(self,source,s,a):
        rel = self.centers[source,s]-self.contact[s,a]
        means = rel@np.stack((TANGENTS[a],-NORMALS[a])).T
        return means,self.shadow(s,a,np.nan_to_num(means,nan=150.))

    def travel(self,s,a,s1,a1):
        neutral = super().travel(s,a,s1,a1)
        return float(np.sqrt(max(0.,neutral**2-(YS[s1]-YS[s])**2+(self.y[s1]-self.y[s])**2)))


def definition(target,begin,end,roles=(0,1,2,3,4)):
    assert end>begin
    n=end-begin+1
    return target,begin,end,tuple(roles),np.minimum(np.arange(n)*6//n,5)


def test_definition(task):
    high,low,roles=TASKS[task]
    begin,end=np.flatnonzero((YS<=high)&(YS>=low))[[0,-1]]
    return definition(0,int(begin),int(end),roles)


def activate(shape,target,width):
    env.STORE=STORES[shape,target]
    return MODELS[shape,target,width]


def setup_train(path):
    global STORES,MODELS
    STORES,MODELS={},{}
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k] for k in z.files}
    for shape in range(len(BETAS)):
        base=TrainingStore(arrays,shape)
        for target in TARGETS:
            a=copy(base)
            order=[target]+[i for i in range(6) if i!=target]
            a.centers=base.centers[:,:,order]
            STORES[shape,target]=a
            for width in WIDTHS:
                env.STORE=a
                MODELS[shape,target,width]=env.Model(width,True,True)


def setup_test(path):
    global STORES,MODELS
    s=env.Store(path)
    STORES={(0,0):s}
    env.STORE=s
    MODELS={(0,0,w):env.Model(w,True,True) for w in WIDTHS}


@dataclass(frozen=True)
class Job:
    shape: int
    source: int
    target: int
    bounds: tuple
    profile: str
    seed: int
    width: int


# Geometry cohorts and seed splits are specified before opening the holdout.
def training_jobs(cohort,stage,tapes):
    out=[]
    offsets={"history":0,"teacher":90_000,"student":180_000,"validation":270_000}
    for shape in range(len(BETAS)):
        for target in TARGETS:
            for k in range(tapes):
                profile=env.PROFILES[(k+shape+target)%2]
                for width in WIDTHS:
                    seed=5_300_000+offsets[stage]+1000*shape+100*target+3*k
                    out.append(Job(shape,CT,target,CT_ROUTES[target],profile,seed,width))
        if cohort=="ct_plus_atlases":
            for geom in range(3):
                # Archive-supported long reaches. CT is kept as a separate
                # source; these models provide additional anatomy coverage.
                task = ("femoral","popliteal","calf","saphenofemoral")[(shape+geom)%4]
                _,begin,end,_,_=test_definition(task)
                for k in range(tapes):
                    width=WIDTHS[k%len(WIDTHS)]
                    seed=5_600_000+offsets[stage]+5000*shape+500*geom+3*k
                    out.append(Job(shape,geom,0,(begin,end),env.PROFILES[k%2],seed,width))
    return out


def region_of(spec,s):
    return int(spec[4][s-spec[1]])


def candidate_features(reader,region):
    candidates=[i for i,f in enumerate(reader.frames) if f[5]==region and len(f[2])]
    if not candidates:
        return [],np.empty((0,18))
    context=candidates
    if HISTORY_NEIGHBORS:
        stations=[reader.frames[i][0] for i in candidates]
        low,high=min(stations)-3,max(stations)+3
        neighbors=[i for i,f in enumerate(reader.frames) if f[5]!=region and low<=f[0]<=high and len(f[2])]
        context=candidates+neighbors

    rows=[]
    for frame in context:
        f=reader.frames[frame]
        for slot,xy in enumerate(f[2]):
            rows.append((frame,slot,f[0],f[1],reader.absolute(f[0],f[1],xy),
                         float(f[6][slot]),float(f[7][slot])))
    count=sum(len(reader.frames[i][2]) for i in candidates)
    # Neighboring stations may support identity. The selected witness
    # still has to be an acquired frame from the requested region.
    world=np.stack([r[4] for r in rows])
    station=np.asarray([r[2] for r in rows])
    view=np.asarray([r[3] for r in rows])
    prob=np.asarray([r[5] for r in rows])
    dd=abs(station[:count,None]-station[None,:])
    da=abs(view[:count,None]-view[None,:])
    da=np.minimum(da,len(ANGLES)-da)
    gaze=((dd>0)&(dd<=3))|((dd==0)&(da>=2))
    dist=np.linalg.norm(world[:count,None]-world[None,:],axis=-1)
    affinity=np.where(gaze,np.exp(-.5*(dist/5.5)**2)*prob[None,:],0.)
    near=affinity.max(axis=1)
    support=(affinity>.22).sum(axis=1)
    corpus=[]
    for i,(ti,slot,s,a,point,p,unassigned) in enumerate(rows[:count]):
        f=reader.frames[ti]
        q=f[4]
        xy=f[2][slot]
        expected=reader.model.h[s,:,0,:]
        finite=np.isfinite(expected).all(axis=1)
        d=np.linalg.norm(expected[finite]-point,axis=-1)
        rival=float(np.max(np.delete(f[6],slot),initial=0.))
        alt=np.max(np.where(gaze[i],prob,0.),initial=0.)
        corpus.append([p,unassigned,q,rival,near[i],min(1.,support[i]/4.),alt,
                       len(f[2])/6.,xy[0]/25.,xy[1]/80.,
                       np.clip(d.min(initial=100.)/45.,0.,2.2),
                       np.clip(np.median(d)/60.,0.,2.) if len(d) else 2.,
                       abs(xy[0])/25.,s/len(YS),np.sin(ANGLES[a]),np.cos(ANGLES[a]),
                       reader.model.sigma/30.,float(gaze[i].any())])
    return rows[:count],np.asarray(corpus)


def verify(reader,region,history=True,guard=False,threshold=None):
    if not history:
        base=env.Reader(reader.model,reader.roles,False)
        base.frames=reader.frames
        return base.select(region)
    ids,x=candidate_features(reader,region)
    if not ids:
        return None
    p=HISTORY.predict_proba(x)[:,1]
    # Unknown/clutter remains a reportable outcome. Candidate identity comes
    # from the fixed graph and requires support from the observed archive.
    okay=(x[:,1]<.94)&((x[:,0]>=.12)|(x[:,4]>=.22))
    if guard:
        okay &= ((x[:,4]>.21)|(x[:,0]>.89))
    indices=np.flatnonzero(okay&(p>=(RISK_THRESHOLD if threshold is None else threshold)))
    if not len(indices):
        return None
    i=int(indices[np.argmax(p[indices])])
    return ids[i][0],ids[i][1],float(p[i])


def report(reader,history=True,guard=False,threshold=None):
    results=[]
    for r in range(6):
        selected=verify(reader,r,history,guard,threshold)
        label=-1 if selected is None else int(reader.frames[selected[0]][3][selected[1]]==0)
        results.append(label)
    covered={int(f[5]) for f in reader.frames if 0 in f[3]}
    correct=results.count(1)
    wrong=results.count(0)
    return {"captured_regions":len(covered),"correct_regions":correct,"wrong_regions":wrong,
            "no_result_regions":results.count(-1),"pass":int(correct==6),
            "wrong_delivery":int(wrong>0),"region_outcomes":results}


@dataclass
class Node:
    reader: object
    station: int
    sector: int
    tick: int
    distance: float
    visits: int = 0
    slow: int = 0

    def fork(self):
        r=env.Reader(self.reader.model,self.reader.roles,True)
        r.frames=self.reader.frames.copy()
        r.state=self.reader.state
        r.w=self.reader.w.copy()
        return Node(r,self.station,self.sector,self.tick,self.distance,self.visits,self.slow)


def move(node,s,a,trial,spec,slow=False):
    if not env.STORE.valid[s,a]:
        raise ValueError("Unsupported skin contact")
    tick,ds=env.keep_station(node.station,node.sector,s,a,slow,node.tick,trial,
                             node.reader,region_of(spec,s))
    node.tick = tick
    node.station,node.sector=s,a
    node.distance += ds


def path_ok(s,a,b):
    shift=(b-a+env.A//2)%env.A-env.A//2
    return all(env.STORE.valid[s,(a+i*np.sign(shift))%env.A] for i in range(abs(shift)+1))


def menu(node,spec,permit_slow=True):
    s,a=node.station,node.sector
    around=[(a+i)%env.A for i in (-3,-2,-1,1,2,3)]
    shifted=[(s,b,False) for b in around if path_ok(s,a,b)]
    prev=s-1
    if prev>=spec[1] and env.STORE.valid[prev,a]:
        shifted.extend([(prev,b,False) for b in env.alternatives(prev,a)
                        if b!=a and path_ok(prev,a,b) and env.STORE.valid[s,b]])
        if permit_slow:
            shifted.append((prev,a,True))
    return shifted


def review(node,action,trial,spec):
    s,a,slow=action
    origin=node.station
    move(node,s,a,trial,spec,slow)
    if s<origin:
        move(node,origin,a,trial,spec,slow)
    node.visits+=1
    node.slow+=int(slow)


def candidate_vector(node,action,spec):
    s,a,slow=action
    r=region_of(spec,node.station)
    reader=node.reader
    d=reader.model.feature(s,a,reader.w)
    b=reader.model.feature(node.station,node.sector,reader.w)
    reading=verify(reader,r)
    online=verify(reader,r,False)
    peak=float(max(reader.frames[-1][6],default=0.))
    cost=env.STORE.travel(node.station,node.sector,s,a)
    return np.r_[b,d,d-b,np.sin(ANGLES[a]),np.cos(ANGLES[a]),
                 (s-node.station),((a-node.sector+12)%24-12)/2.,slow,
                 r/6.,node.station/len(YS),node.visits/REVIEW_LIMIT,cost/24.,
                 peak,reading[-1] if reading else 0.,online[-1] if online else 0.,
                 float(online is not None and reading is None),
                 float(reading is not None and online is not None and reading[:2]!=online[:2]),
                 reader.model.sigma/30.]


def decision_point(spec,s):
    return s==spec[2] or region_of(spec,s+1)!=region_of(spec,s)


def should_guard(node,spec):
    r=region_of(spec,node.station)
    current=verify(node.reader,r)
    old=verify(node.reader,r,False)
    if current is None or old is None or current[:2]!=old[:2] or current[-1]<GUARD_CONFIDENCE:
        return False
    # Check whether this archive contains a distinct supporting view.
    ids,x=candidate_features(node.reader,r)
    return any((item[0],item[1])==current[:2] and row[4]>.30 for item,row in zip(ids,x))


def start(spec,trial,model):
    _,s,_,roles,_=spec
    reader=env.Reader(model,roles,True)
    sector=max(np.flatnonzero(env.STORE.valid[s]),
               key=lambda a:model.feature(s,a,reader.w)[0])
    n=Node(reader,s,sector,0,0.)
    move(n,s,sector,trial,spec)
    return n


def execute(spec,trial,model,actor=None,guard=False,permit_slow=True,watch=False,
            exploration=None,initial=None,exit_strategy="native"):
    node=start(spec,trial,model) if initial is None else initial
    states=[]
    trace=[]
    continuation={}
    initial_station = None if initial is None else initial.station
    while True:
        if initial_station is None or node.station>initial_station:
            if decision_point(spec,node.station) and node.visits<REVIEW_LIMIT:
                opts=menu(node,spec,permit_slow)
                if exit_strategy!="native":
                    opts=[a for a in opts if path_ok(node.station,a[1],node.sector)]
                if opts and watch:
                    states.append(node.fork())
                if opts and actor is not None and not (guard and should_guard(node,spec)):
                    x=np.stack([candidate_vector(node,a,spec) for a in opts])
                    q=actor.predict(x)
                    if exit_strategy in ("return","protected","return_cost_only"):
                        # Account for the return trip in the same motion
                        # units used by the CT search teacher.
                        for k,(_,sector,_) in enumerate(opts):
                            tick,mm=env.move(node.station,sector,0,node.station,node.sector)
                            q[k]-=.002*tick+.0018*mm
                    j=int(np.argmax(q))
                    exploratory = exploration is not None and exploration.random()<.22
                    if q[j]>=MIN_GAIN or exploratory:
                        if exploratory and q[j]<MIN_GAIN:
                            j=int(exploration.integers(len(opts)))
                        trace.append((node.station,node.tick,exploratory,opts[j]))
                        original_sector=node.sector
                        if exit_strategy=="protected":
                            # Compute the short continuation before seeing
                            # the recheck; only the public belief is used.
                            sector=original_sector
                            continuation={}
                            for s in range(node.station+1,min(spec[2],node.station+5)+1):
                                sector=env.pick_angle(model,node.reader.w,s-1,sector,s)
                                continuation[s]=sector
                        review(node,opts[j],trial,spec)
                        if exit_strategy in ("return","protected","return_unpriced") and node.sector!=original_sector:
                            move(node,node.station,original_sector,trial,spec)
        if node.station==spec[2]:
            return node,states,trace
        target=node.station+1
        angle=continuation.pop(target,None)
        if angle is None:
            angle=env.pick_angle(model,node.reader.w,node.station,node.sector,target)
        move(node,target,angle,trial,spec)


def structured_job(job):
    spec=definition(job.target,*job.bounds)
    model=activate(job.shape,job.target,job.width)
    return spec,model,env.Trial(job.source,job.seed,job.profile)


def history_rows(job):
    spec,model,trial=structured_job(job)
    n,_,_=execute(spec,trial,model)
    xs,ys=[],[]
    for r in range(6):
        ids,x=candidate_features(n.reader,r)
        if len(ids):
            xs.extend(x)
            ys.extend(int(n.reader.frames[i[0]][3][i[1]]==0) for i in ids)
    return xs,ys


def utility(node):
    metric=report(node.reader)
    return (2.5*metric["pass"]+.31*metric["correct_regions"]-.53*metric["wrong_regions"]
            +.12*metric["captured_regions"]-.020*node.tick/10.-.0018*node.distance)


def collect_teacher(job,student=False):
    spec,model,trial=structured_job(job)
    rng=np.random.default_rng(job.seed+8331)
    endpoint,history,trace=execute(spec,trial,model,INITIAL if student else None,
                                   watch=True,exploration=rng if student else None)
    end=report(endpoint.reader)
    # Ground truth ranks failure states only during training. Every state is
    # a snapshot that was physically reached by the current student.
    if not history:
        return [],{"scans":1,"failed_scans":int(not end["pass"]),"student_actions":len(trace)}
    priority=[]
    for n in history:
        r=region_of(spec,n.station)
        after=bool(trace) and n.tick>trace[0][1]
        priority.append(1.+2.4*(end["region_outcomes"][r]!=1)+1.1*(student and after))
    use=rng.choice(len(history),size=min(2,len(history)),replace=False,p=np.asarray(priority)/sum(priority))
    rows=[]
    after_count=failed_nodes=0
    for idx in sorted(use):
        node=history[idx]
        r=region_of(spec,node.station)
        # The no-intervention arm and all teacher actions share this prefix
        # and run the same MPC tail with the same disturbance mechanism.
        null,_,_=execute(spec,trial,model,initial=node.fork())
        fallback=utility(null)
        candidates=menu(node,spec)
        for a in candidates:
            child=node.fork()
            review(child,a,trial,spec)
            last,_,_=execute(spec,trial,model,initial=child)
            rows.append((candidate_vector(node,a,spec).tolist(),float(utility(last)-fallback)))
        failed_nodes+=int(end["region_outcomes"][r]!=1)
        after_count+=int(bool(trace) and node.tick>trace[0][1])
    return rows,{"scans":1,"failed_scans":int(not end["pass"]),"student_actions":len(trace),
                 "failure_nodes":failed_nodes,"reached_nodes":len(use),"post_student_nodes":after_count,
                 "student_model_actions":sum(int(not e[2]) for e in trace),
                 "student_exploration_actions":sum(int(e[2]) for e in trace),
                 "comparisons":len(rows),"positive_comparisons":sum(y>MIN_GAIN for _,y in rows)}


def collect_initial(job):
    return collect_teacher(job,False)


def collect_student(job):
    return collect_teacher(job,True)


def grouped(work,jobs,workers):
    with ProcessPoolExecutor(max_workers=workers) as p:
        return list(p.map(work,jobs,chunksize=2))


def fit_ranker(rows):
    x,y=map(np.asarray,zip(*rows))
    return HistGradientBoostingRegressor(max_iter=115,max_leaf_nodes=15,min_samples_leaf=20,
                                         l2_regularization=2.5,random_state=30301).fit(x,y)


THRESHOLDS = (.37,.45,.53,.61,.69,.77,.85)


def calibrate(job):
    spec,model,trial=structured_job(job)
    node,_,_=execute(spec,trial,model)
    return [report(node.reader,threshold=t) for t in THRESHOLDS]


def train(out,cohort,workers,tapes,history_neighbors=False):
    global HISTORY,INITIAL,RISK_THRESHOLD,HISTORY_NEIGHBORS
    HISTORY_NEIGHBORS=history_neighbors
    setup_train(out/"ct_smplx_shapes.npz")
    learned={}
    history=grouped(history_rows,training_jobs(cohort,"history",tapes),workers)
    x=np.concatenate([xs for xs,_ in history if len(xs)])
    y=np.concatenate([ys for _,ys in history if len(ys)])
    HISTORY=HistGradientBoostingClassifier(max_iter=125,max_leaf_nodes=15,min_samples_leaf=30,
                                           l2_regularization=3.,random_state=921).fit(x,y)
    print(cohort,"history",len(y),"candidates",flush=True)
    validation=grouped(calibrate,training_jobs(cohort,"validation",2),workers)
    validation_scores=[]
    for t,rows in zip(THRESHOLDS,zip(*validation),strict=True):
        correct=sum(r["correct_regions"] for r in rows)
        wrong=sum(r["wrong_regions"] for r in rows)
        passed=sum(r["pass"] for r in rows)
        validation_scores.append({"threshold":t,"correct":correct,"wrong":wrong,"pass":passed,
                                  "utility":correct-.95*wrong+2.5*passed})
    RISK_THRESHOLD=max(validation_scores,key=lambda x:x["utility"])["threshold"]
    first=grouped(collect_initial,training_jobs(cohort,"teacher",tapes),workers)
    initial=[row for batch,_ in first for row in batch]
    INITIAL=fit_ranker(initial)
    print(cohort,"initial teacher",len(initial),"branch comparisons",flush=True)
    failure=grouped(collect_student,training_jobs(cohort,"student",tapes),workers)
    additional=[row for batch,_ in failure for row in batch]
    final=fit_ranker(initial+additional)
    frozen={"history":HISTORY,"initial":INITIAL,"aggregated":final,"history_neighbors":HISTORY_NEIGHBORS,"thresholds":{
            "witness":RISK_THRESHOLD,"intervene":MIN_GAIN,"guard":GUARD_CONFIDENCE}}
    def counts(rows):
        groups=[r for _,r in rows]
        return {k:sum(g.get(k,0) for g in groups) for k in set().union(*(r.keys() for r in groups))}
    model=out/f"{cohort}.joblib"
    joblib.dump(frozen,model,compress=3)
    for name,rows in (("teacher",first),("student_failures",failure)):
        with gzip.open(out/f"{cohort}_{name}.jsonl.gz","wt",compresslevel=5) as f:
            for (pairs,counts_i),job in zip(rows,training_jobs(cohort,"teacher" if name=="teacher" else "student",tapes)):
                f.write(json.dumps({"source":SOURCE_NAMES[job.source],"shape":job.shape,"target":TARGET_NAMES[job.target],
                                    "seed":job.seed,"width":job.width,"profile":job.profile,
                                    "counts":counts_i,"counterfactuals":pairs},allow_nan=False,separators=(",",":"))+"\n")
    learned={"cohort":cohort,"no_blender_geometry_loaded":True,
             "history_neighbors":HISTORY_NEIGHBORS,
             "clinical_CT_subjects":1,"smplx_shape_count":len(BETAS),"CT_targets":TARGET_NAMES,
             "history_candidates":len(y),"positive_history_candidates":int(y.sum()),
             "held_CT_threshold_selection":validation_scores,"selected_witness_threshold":RISK_THRESHOLD,
             "initial_teacher":counts(first),"student_failure_collection":counts(failure),
             "model_sha256":digest(model),"geometry_sha256":digest(out/"ct_smplx_shapes.npz")}
    (out/f"{cohort}_training.json").write_text(json.dumps(learned,indent=2)+"\n")
    print(json.dumps(learned),flush=True)


def evaluate(job):
    task,profile,tape=job
    spec=test_definition(task)
    trial=env.Trial(3,env.TEST_SEED+1000*list(TASKS).index(task)+tape,profile)
    arms={}
    for k,width,actor,guard,slow in (
        ("mpc",8,None,False,False),("search_teacher",8,INITIAL,False,False),
        ("search_student_history",8,AGGREGATED,False,False),
        ("plus_history_first",8,AGGREGATED,True,False),
        ("plus_wider_prior",30,AGGREGATED,True,False),
        ("plus_speed",30,AGGREGATED,True,True)):
        model=activate(0,0,width)
        end,_,_=execute(spec,trial,model,actor,guard,slow)
        metric=report(end.reader,True,False)
        metrics={"time_s":end.tick/10.,"travel_mm":end.distance,
                 "rechecks":end.visits,"slow_segments":end.slow}
        arms[k]=metric|metrics
        if k in ("mpc","search_student_history"):
            arms[f"{k}_online_reader"]=report(end.reader,False)|metrics
        if k=="plus_history_first":
            # Compare widths on exactly the same measured path as well.
            wide=activate(0,0,30)
            replay=env.Reader(wide,end.reader.roles,True)
            for s,a,z,y,q,r,_,_,tick in end.reader.frames:
                replay.push(s,a,z,y,q,r,tick)
            arms["same_path_wider_prior"]=report(replay,True,False)|metrics
            arms["same_path_evidence_gate"]=report(end.reader,True,True)|metrics
    return {"task":task,"profile":profile,"tape":tape,"arms":arms}


def test(out,cohort,parent,workers,reps):
    global HISTORY,INITIAL,AGGREGATED,RISK_THRESHOLD,HISTORY_NEIGHBORS
    meta=json.loads((out/f"{cohort}_training.json").read_text())
    file=out/f"{cohort}.joblib"
    if digest(file)!=meta["model_sha256"] or not meta["no_blender_geometry_loaded"]:
        raise ValueError("Training freeze is missing or changed")
    learned=joblib.load(file)
    HISTORY,INITIAL,AGGREGATED=(learned[k] for k in ("history","initial","aggregated"))
    HISTORY_NEIGHBORS=learned.get("history_neighbors",False)
    RISK_THRESHOLD=learned["thresholds"]["witness"]
    setup_test(parent)
    tests=[(task,profile,tape) for task in TASKS for profile in env.PROFILES for tape in range(reps)]
    rows=grouped(evaluate,tests,workers)
    summary={"model_freeze":meta["model_sha256"],"test_geometry_sha256":digest(parent/"geometry.npz"),
             "cases":len(rows),"Blender_anatomy_subjects":1,"full_route_length_mm":730.,
             "total":env.totals(rows),"per_task":{t:env.totals([r for r in rows if r["task"]==t]) for t in TASKS},
             "paired":{a:{f:env.bootstrap(rows,a,b,f)
                          for f in ("pass","wrong_delivery","correct_regions","time_s")}
                       for a,b in (("search_teacher","mpc"),("search_student_history","search_teacher"),
                                   ("plus_history_first","search_student_history"),
                                   ("same_path_wider_prior","plus_history_first"),
                                   ("same_path_evidence_gate","plus_history_first"),
                                   ("plus_wider_prior","plus_history_first"),("plus_speed","plus_wider_prior"))}}
    (out/f"{cohort}_blender_test.json").write_text(json.dumps(summary,indent=2)+"\n")
    with gzip.open(out/f"{cohort}_blender_trials.jsonl.gz","wt",compresslevel=5) as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
    print(json.dumps(summary["total"],indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("train","test"))
    p.add_argument("--cohort",choices=("ct_only","ct_plus_atlases"),required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=10)
    p.add_argument("--tapes",type=int,default=4)
    p.add_argument("--reps",type=int,default=16)
    p.add_argument("--history-neighbors",action="store_true",
                   help="use acquired frames at nearby material stations as identity context during training")
    a=p.parse_args()
    with threadpool_limits(limits=1):
        if a.stage=="train":
            train(a.out,a.cohort,a.workers,a.tapes,a.history_neighbors)
        else:
            test(a.out,a.cohort,a.out.parent,a.workers,a.reps)

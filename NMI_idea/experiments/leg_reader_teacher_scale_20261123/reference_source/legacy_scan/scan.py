"""Paired synthetic ultrasound scans on a held-out registered Blender leg."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor
from joblib import dump

from geometry import ANGLES, NORMALS, TANGENTS, YS, TASKS, TEST, SOURCES, bone_prior, digest
from leg_route_belief import associate

A = len(ANGLES)
SECTIONS = len(YS)
LATERAL, DEPTH_MIN, DEPTH_MAX = 25., 3., 80.
WINDOW_AREA = 2 * LATERAL * (DEPTH_MAX-DEPTH_MIN)
TRAIN_TASKS = tuple(TASKS)[:-1]
TRAIN_COUNT = 9
PROFILES = ("ordinary", "stress")
TEST_SEED = 884_000
TRAIN_SEED = 671_000
RISK_THRESHOLD = .84
FORECAST_MARGIN = .075
MAX_REVIEWS = 6
BASE_SPEED = 43.


@dataclass
class Tape:
    normal: np.ndarray
    segmentation: np.ndarray
    dropout: np.ndarray
    false: np.ndarray

    @classmethod
    def make(cls, seed, profile, n=3800):
        rng = np.random.default_rng(seed)
        normal = rng.standard_normal((n, 6, 2))
        u = rng.random(n)
        fail = False
        dropout = np.zeros(n,dtype=bool)
        for i,v in enumerate(u):
            enter, stay = (.05,.65) if profile=="stress" else (.022,.38)
            fail = v < (stay if fail else enter)
            dropout[i]=fail
        q = .78 if profile=="stress" else .40
        from scipy.signal import lfilter
        normal=lfilter([np.sqrt(1.-q*q)],[1.,-q],normal,axis=0)
        return cls(normal,rng.random((n,6)),dropout,rng.random((n,12)))


class Trial:
    def __init__(self, source, seed, profile):
        self.source, self.tape, self.profile = source,Tape.make(seed,profile),profile

    def frame(self, s, a, t):
        truth, gate = STORE.truth(self.source,s,a)
        return self.observe(truth, gate, t)

    def observe(self, truth, gate, t):
        tape = self.tape
        q = tape.false[t]
        strength = .90 if self.profile=="ordinary" else .68
        soft = .94 if self.profile=="ordinary" else .80
        quality = np.clip(soft - .28*np.max(gate,initial=0.) + .26*(q[10]-.5),.05,1.)
        if tape.dropout[t]:
            return np.empty((0,2)), np.empty(0,dtype=int), float(.1+.3*q[11])
        visible = np.isfinite(truth).all(axis=1)&(abs(truth[:,0])<=LATERAL)&(truth[:,1]>=DEPTH_MIN)&(truth[:,1]<=DEPTH_MAX)
        detect = visible * strength * (1. - .96*gate)
        seen = np.flatnonzero(tape.segmentation[t]<detect)
        noise = np.array([1.4,1.1]) if self.profile=="ordinary" else np.array([2.3,1.8])
        echoes = truth[seen]+noise*tape.normal[t,seen]
        roles=seen.tolist()
        pts=echoes.tolist()
        if len(seen)>1:
            norm=np.linalg.norm(truth[seen,None]-truth[None,seen],axis=-1)
            norm[np.tril_indices(len(seen))]=np.inf
            i,j=np.unravel_index(np.argmin(norm),norm.shape)
            merge=(.44 if self.profile=="ordinary" else .79)*np.exp(-.5*(norm[i,j]/4.7)**2)
            if q[0]<merge:
                f=.35+.30*q[1]
                pts[i]=((1.-f)*echoes[i]+f*echoes[j]).tolist()
                roles[i]=-2
                del pts[j],roles[j]
        if q[2]<(.31 if self.profile=="ordinary" else .66):
            plausible=np.flatnonzero(visible)
            if q[3]<.72 and len(plausible):
                # False contour near any observed structure or the edge of a bone shadow.
                idx=plausible[int(q[4]*len(plausible))]
                d=truth[idx]+[2.7+5.*(q[5]-.5), 2.+5.*(q[6]-.5)]
            else:
                d=[(2.*q[7]-1.)*LATERAL,DEPTH_MIN+q[8]*(DEPTH_MAX-DEPTH_MIN)]
            pts.append(d)
            roles.append(-1)
        p=np.asarray(pts).reshape(-1,2)
        labels=np.asarray(roles,dtype=int)
        good=np.flatnonzero((abs(p[:,0])<=LATERAL)&(p[:,1]>=DEPTH_MIN)&(p[:,1]<=DEPTH_MAX))
        # Order reveals lateral position only.
        order=good[np.argsort(p[good,0],kind="stable")]
        return p[order],labels[order],float(quality)


class Store:
    def __init__(self,output):
        pack=np.load(output/"geometry.npz")
        self.centers=pack["centers"]
        self.contact=pack["contacts"]
        self.exact=pack["bone_distance"]
        self.inside=pack["inside_bone"]
        self.valid=pack["valid_contact"]
        self.bone=bone_prior()
        self._make_fields()

    def _make_fields(self):
        # Approximate occupancy of each ray by its highest local population
        # occupancy. Along-ray voxels are correlated, so do not multiply them.
        offsets=np.arange(-25.,25.1,2.5)
        depths=np.arange(4.,80.1,2.)
        origins=self.contact[:,:,None,None]+TANGENTS[None,:,None,None]*offsets[None,None,:,None,None]
        rays=origins-NORMALS[None,:,None,None]*depths[None,None,None,:,None]
        stations=np.broadcast_to(YS[:,None,None,None],rays.shape[:-1])
        xyz=np.stack((rays[...,0],stations,rays[...,1]),axis=-1)
        occ=self.bone(xyz)
        self.bone_support=float(np.isfinite(occ).mean())
        uncertain=np.where(np.isfinite(occ),occ,.15)
        self.ray=np.maximum.accumulate(uncertain,axis=-1).astype(np.float32)
        next_a=np.roll(self.contact,-1,axis=1)
        self.slew=np.linalg.norm(self.contact-next_a,axis=-1)
        self.depth_count=len(depths)

    def shadow(self,s,a,p):
        u=np.clip(np.rint((p[...,0]+25.)/2.5),0,20).astype(int)
        z=np.clip(np.rint((p[...,1]-4.)/2.),0,self.depth_count-1).astype(int)
        return self.ray[s,a,u,z]

    def travel(self,s,a,s1,a1):
        da=(a1-a+A//2)%A-A//2
        current=a
        turn=0.
        for _ in range(abs(da)):
            b=(current+int(np.sign(da)))%A
            turn+=.5*(self.slew[s,current if da>0 else b]+self.slew[s1,current if da>0 else b])
            current=b
        straight=np.linalg.norm(np.r_[self.contact[s1,a]-self.contact[s,a],YS[s1]-YS[s]])
        return float(np.hypot(turn,straight))

    @lru_cache(maxsize=SECTIONS*A*4)
    def truth(self,source,s,a):
        p=self.centers[source,s]-self.contact[s,a]
        xy=p@np.stack((TANGENTS[a],-NORMALS[a])).T
        if source==TEST:
            block=(self.exact[s,a]<xy[:,1]-1.)|self.inside[s]
            occ=block.astype(float)
        else:
            occ=self.shadow(s,a,np.nan_to_num(xy,nan=150.))
        return xy,occ

    def preflight(self):
        value={}
        for task,(hi,lo,_) in TASKS.items():
            ids=np.flatnonzero((YS<=hi)&(YS>=lo))
            present=[]; in_window=[]; visible=[]
            for s in ids:
                stage=[self.truth(TEST,s,a) for a in np.flatnonzero(self.valid[s])]
                k=[(np.isfinite(p[0]).all(),abs(p[0,0])<=25 and 3<=p[0,1]<=80,occ[0]<.5) for p,occ in stage]
                present.append(any(x[0] for x in k))
                in_window.append(any(x[1] for x in k))
                visible.append(any(x[1] and x[2] for x in k))
            regions=np.array_split(np.asarray(visible),6)
            value[task]={"stations":len(ids),"target_in_blender":sum(present),
                         "physically_in_50mm_fov_for_some_pose":sum(in_window),
                         "has_bone_clear_pose":sum(visible),
                         "regions_with_bone_clear_pose":sum(bool(g.any()) for g in regions)}
        return value


class Model:
    def __init__(self, sigma, graph, bone):
        self.sigma,self.graph,self.bone=sigma,graph,bone
        # Keep each registered reference as a separate component. A branch
        # moves longitudinally with its parent branch model and never spawns.
        sobol=qmc.Sobol(4,scramble=True,seed=62191).random_base2(5)
        from scipy.special import ndtri
        draws=ndtri(np.clip(sobol,.04,.96))
        self.samples=np.vstack(([0.,0.,0.,0.],draws))
        self.prior=np.repeat(1./(3*len(self.samples)),3*len(self.samples))
        self.h=np.empty((SECTIONS,len(self.prior),5,2))
        for s in range(SECTIONS):
            shift=np.rint(self.samples[:,2]*2.).astype(int)
            locations=np.clip(s+shift,0,SECTIONS-1)
            p=STORE.centers[:3,locations,:5,:].copy()
            curve = .35*sigma*self.samples[:,3]*np.sin(s*.10)
            p += (sigma*self.samples[:,:2] + np.stack((curve,-.5*curve),axis=-1))[None,:,None]
            self.h[s]=p.reshape(len(self.prior),5,2)

    @lru_cache(maxsize=SECTIONS*A)
    def view(self,s,a):
        p=(self.h[s]-STORE.contact[s,a])@np.stack((TANGENTS[a],-NORMALS[a])).T
        available=np.isfinite(p).all(axis=-1)
        p=np.nan_to_num(p,nan=999.)
        occupancy=STORE.shadow(s,a,p) if self.bone else np.zeros(p.shape[:2])
        detect=.74*(1.-.92*occupancy)*available
        sigma=np.array([2.35,1.85])
        near=(np.maximum(DEPTH_MIN,p[...,1])<DEPTH_MAX+9.) & (np.abs(p[...,0])<LATERAL+9.)
        detect *= near
        integration = (ndtr(([LATERAL,DEPTH_MAX]-p)/sigma)-ndtr(([-LATERAL,DEPTH_MIN]-p)/sigma)).prod(axis=-1)
        distances=np.linalg.norm(p[:,1:]-p[:,:1],axis=-1)
        distances=np.where(available[:,1:],distances,50.)
        sep=np.min(distances,axis=1)
        return p,detect,integration,available,occupancy,sep

    def feature(self,s,a,w):
        p,d,f,avail,oc,sep=self.view(s,a)
        expected=w@(d[:,0]*f[:,0])
        proximity=w@(np.minimum(25.,sep)/25.)
        variability=np.sqrt(max(0., w@((d[:,0]*f[:,0]-expected)**2)))
        return np.array([expected,proximity,variability,w@oc[:,0],w@(f[:,0]),
                         w@(np.exp(-.5*(sep/5.)**2))])

    def analyze(self,s,a,w,z,quality,roles):
        centers,base,integral,_,occ,sep=self.view(s,a)
        k=roles if self.graph else (0,)
        centers,base,integral=centers[:,k],base[:,k],integral[:,k]
        detection=base*(.45+.64*quality)
        missed=1.-detection*integral
        if len(z)==0:
            # Known graph stays intact. Empty/low quality frames have a
            # shared dropout explanation alongside a geometric miss.
            likelihood=.34+.66*missed.prod(axis=-1)
            return likelihood,np.empty((len(w),0)),np.empty((len(w),0))
        sig=np.array([2.35,1.85])
        delta=(z[None,:,None]-centers[:,None])/sig
        density=np.exp(-.5*np.sum(delta**2,axis=-1))/(2.*np.pi*np.prod(sig))
        ghost=np.full((len(w),len(z)),.62/WINDOW_AREA)
        if self.graph and len(k)>1:
            i,j=np.triu_indices(len(k),1)
            dist=np.linalg.norm(centers[:,i]-centers[:,j],axis=-1)
            pair_p=np.exp(-.5*(dist/6.5)**2)*detection[:,i]*detection[:,j]
            mid=.5*(centers[:,i]+centers[:,j])
            residual=(z[None,:,None]-mid[:,None])/np.array([3.3,2.8])
            mixed=np.exp(-.5*np.sum(residual**2,axis=-1))/(2.*np.pi*3.3*2.8)
            ghost=ghost+.61*np.sum(mixed*pair_p[:,None],axis=-1)
        ratio=density*detection[:,None]/np.maximum(1.e-10,ghost[...,None])
        # Each contour can be assigned to a known single lumen, or remain
        # unexplained. The ambiguous/merged term never creates a graph edge.
        mass,conditional=associate(missed,ratio)
        return mass*np.prod(ghost,axis=1),conditional[...,0],1.-conditional.sum(axis=-1)


def wreset(w,prior,delta,quality):
    # Keep hypothesis support across long routes with changing vessel shape,
    # preserving enough uncertainty to recover from an early false match.
    mix=.014*min(2.,abs(delta))+.007+.010*(1.-quality)
    return (1.-mix)*w+mix*prior


class Reader:
    def __init__(self,model,roles,history):
        self.model,self.roles,self.history=model,roles,history
        self.w=model.prior.copy()
        self.frames=[]
        self.state=None

    def eval(self,s,a,z,q,w,last):
        strength=.57 if last is None or last!=(s,a) else .32
        r=wreset(w,self.model.prior,1 if last is None else s-last[0],q)
        ll,prob,unknown=self.model.analyze(s,a,r,z,q,self.roles)
        if len(z)==0:
            strength=.68*q
        posterior=r*np.maximum(ll,1.e-90)**strength
        posterior/=posterior.sum()
        return posterior,posterior@prob,posterior@unknown

    def push(self,s,a,z,labels,q,region,tick=None):
        self.w,p,unknown=self.eval(s,a,z,q,self.w,self.state)
        self.state=(s,a)
        self.frames.append((s,a,z,labels,q,region,p,unknown,tick))

    def options(self,region):
        return [(idx,slot,row[0],row[1],self.absolute(row[0],row[1],row[2][slot]),prob,row[4])
                for idx,row in enumerate(self.frames) if row[5]==region
                for slot,prob in enumerate(row[6]) if prob>=.12]

    @staticmethod
    def absolute(s,a,xy):
        return STORE.contact[s,a]+xy[0]*TANGENTS[a]-xy[1]*NORMALS[a]

    def select(self,region):
        items=self.options(region)
        if not items:
            return None
        if not self.history:
            v=max(items,key=lambda row:row[5])
            return (v[0],v[1],v[5]) if v[5]>=RISK_THRESHOLD else None
        best=None
        for item in items:
            idx,slot,s,a,point,prob,q=item
            support=0.
            for other in items:
                t,b=other[2:4]
                if 1<=abs(s-t)<=3 or (s==t and min((a-b)%A,(b-a)%A)>=2):
                    residual=np.linalg.norm(point-other[4])
                    support=max(support,other[5]*np.exp(-.5*(residual/5.5)**2))
            score=.75*prob+.22*support+.03*q
            if prob>=.55 and score>=.66 and (best is None or score>best[-1]):
                best=(idx,slot,score)
        return best

    def retained_result(self,r):
        choice=self.select(r)
        if choice is None:
            return -1
        return 1 if self.frames[choice[0]][3][choice[1]]==0 else 0


def alternatives(s,a):
    return [(a+i)%A for i in (0,-1,1,-2,2) if STORE.valid[s,(a+i)%A]]


def pick_angle(model,w,s,a,next_s,lookahead=True):
    options=alternatives(next_s,a)
    if not options:
        options=list(np.flatnonzero(STORE.valid[next_s]))
    def value(b):
        q=model.feature(next_s,b,w)
        val=q[0]+.19*q[1]-.012*STORE.travel(s,a,next_s,b)
        if lookahead and next_s < SECTIONS-1:
            after=alternatives(next_s+1,b)
            val+=.47*max((model.feature(next_s+1,c,w)[0]-.012*STORE.travel(next_s,b,next_s+1,c) for c in after),default=0.)
        return val
    return max(options,key=value)


def move(s,a,t,s1,a1,slow=False):
    mm=STORE.travel(s,a,s1,a1)
    units=max(1,int(np.ceil(10.*mm/(BASE_SPEED*(.5 if slow else 1.)))))
    return t+units,mm


def keep_station(s,a,s1,a1,slow,t,trial,reader,bin_index):
    t1,mm=move(s,a,t,s1,a1,slow)
    # Pose sampled at each reached grid station. The optional half-speed
    # model accrues extra measured frames at the same terminal station.
    captures=1 if not slow else min(4,t1-t)
    for tt in range(t1-captures+1,t1+1):
        z,y,q=trial.frame(s1,a1,tt)
        reader.push(s1,a1,z,y,q,bin_index,tt)
    return t1,mm


def features(s,a,b,t,reader,region,action):
    m=reader.model
    p=reader.frames[-1][6]
    choices=reader.options(region)
    evidence=reader.select(region)
    return np.r_[m.feature(s,a,reader.w),m.feature(s,b,reader.w),
                 max(p,default=0.),len(p),float(reader.frames[-1][4]),min(len(choices),12)/12.,
                 evidence[-1] if evidence else 0.,STORE.travel(s,a,s,b)/25.,
                 np.sin(ANGLES[a]),np.cos(ANGLES[a]),s/SECTIONS,np.log1p(t)/6.,
                 action=="slow"]


def review(s,a,b,t,action,trial,reader,region):
    if action != "slow":
        return keep_station(s,a,s,b,False,t,trial,reader,region)
    mid,length0=keep_station(s,a,s-1,a,True,t,trial,reader,region)
    end,length1=keep_station(s-1,a,s,a,True,mid,trial,reader,region)
    return end,length0+length1


def hypothetical(trial,reader,s,a,b,t,region,action):
    # The acquisition action is evaluated from the same measured prefix.
    # Previously gathered evidence stays available to all candidate actions.
    shadow=Reader(reader.model,reader.roles,reader.history)
    shadow.w=reader.w.copy()
    shadow.state=reader.state
    shadow.frames=reader.frames.copy()
    end,d=review(s,a,b,t,action,trial,shadow,region)
    value=lambda x: 1. if x==1 else (-1.15 if x==0 else 0.)
    advantage=value(shadow.retained_result(region))-value(reader.retained_result(region))
    return advantage-.025*(end-t)/10.-.004*d


def scan(task,trial,movement,model,learner=None,training=False):
    hi,lo,roles=TASKS[task]
    ids=np.flatnonzero((YS<=hi)&(YS>=lo))
    groups=np.minimum(np.arange(len(ids))*6//len(ids),5)
    r=Reader(model,roles,True)
    a=max(np.flatnonzero(STORE.valid[ids[0]]),
          key=lambda angle:model.feature(ids[0],angle,r.w)[0]) 
    t=0
    s=ids[0]
    d=0.
    reviews=slow_count=0
    used=set()
    events=[]
    for i,s1 in enumerate(ids):
        region=groups[i]
        b=a
        if i and movement!="straight":
            b=pick_angle(model,r.w,s,a,s1)
        elif i and not STORE.valid[s1,b]:
            b=pick_angle(model,r.w,s,a,s1,False)
        t,mm=keep_station(s,a,s1,b,False,t,trial,r,region)
        d+=mm
        s,a=s1,b
        span=region not in used and i>=1 and (i==len(ids)-1 or groups[min(i+1,len(ids)-1)]!=region)
        # Evaluate one candidate decision per scan region before leaving it.
        if not span:
            continue
        opts=[(b,"recheck") for b in alternatives(s,a) if b!=a]
        if i and groups[i-1]==region and STORE.valid[s-1,a]:
            opts.append((a,"slow"))
        if training:
            for b,act in opts:
                x=features(s,a,b,t,r,region,act)
                y=hypothetical(trial,r,s,a,b,t,region,act)
                events.append((x.tolist(),y))
        chosen=None
        if movement=="rule" and r.select(region) is None:
            cand=[b for b,act in opts if act!="slow"]
            if cand:
                chosen=(max(cand,key=lambda b:model.feature(s,b,r.w)[0]-.01*STORE.travel(s,a,s,b)),"recheck")
        if movement in ("value","value_speed"):
            options=[o for o in opts if o[1]!="slow" or movement=="value_speed"]
            if options:
                x=np.stack([features(s,a,b,t,r,region,act) for b,act in options])
                gains=learner.predict(x)
                j=int(np.argmax(gains))
                if gains[j]>FORECAST_MARGIN:
                    chosen=options[j]
        if chosen and reviews<MAX_REVIEWS:
            b,action=chosen
            t1,mm=review(s,a,b,t,action,trial,r,region)
            t=t1
            a=b
            d+=mm
            reviews+=1
            slow_count+=int(action=="slow")
            used.add(region)
    return r.frames,t/10.,d,reviews,slow_count,events


def score(frames,model,roles,history):
    r=Reader(model,roles,history)
    for s,a,z,y,q,region,_,_,tick in frames:
        r.push(s,a,z,y,q,region,tick)
    accepted=[r.retained_result(b) for b in range(6)]
    truth_seen=sorted({int(row[5]) for row in r.frames if 0 in row[3]})
    identified=sum(x==1 for x in accepted)
    wrong=sum(x==0 for x in accepted)
    # Verify targets within the task remain the same named graph; any
    # unmatched structures can be retained as unattributed local echoes.
    return {"captured_regions":len(truth_seen),"correct_regions":identified,
            "wrong_regions":wrong,"no_result_regions":6-identified-wrong,
            "pass":int(identified==6),"wrong_delivery":int(wrong>0)}


def training(job):
    src,task,profile,tape=job
    trial=Trial(src,TRAIN_SEED+10000*src+100*TRAIN_TASKS.index(task)+tape,profile)
    return scan(task,trial,"mpc",MODELS["8"],training=True)[-1]


def evaluation(job):
    task,profile,tape=job
    seed=TEST_SEED+1000*list(TASKS).index(task)+tape
    main=Trial(TEST,seed,profile)
    output={}
    for movement in ("mpc","straight","rule","value","value_speed"):
        frames,t,length,reviews,slows,_=scan(task,main,movement,MODELS["8"],LEARNER)
        conditions=(("target_only","single",False),("graph","blind",False),
                    ("graph_bone","8",False),("graph_bone_history","8",True),
                    ("sigma4_history","4",True),("sigma12_history","12",True)) if movement=="mpc" else (
                    (movement,"8",True),)
        for title,prior,history in conditions:
            detail=score(frames,MODELS[prior],TASKS[task][2],history)
            output[title]={**detail,"time_s":t,"travel_mm":length,"rechecks":reviews,"slow_segments":slows}
    return dict(task=task,profile=profile,tape=tape,seed=seed,arms=output)


def totals(rows):
    names=rows[0]["arms"]
    out={}
    for arm in names:
        values=[r["arms"][arm] for r in rows]
        sums={k:int(sum(v[k] for v in values)) for k in ("pass","wrong_delivery","wrong_regions","correct_regions","captured_regions","no_result_regions","rechecks","slow_segments")}
        means={k:float(np.mean([v[k] for v in values])) for k in ("time_s","travel_mm")}
        out[arm]={"n":len(values),**sums,**means}
    return out


def bootstrap(rows,alt,ref,metric):
    ds=np.array([r["arms"][alt][metric]-r["arms"][ref][metric] for r in rows])
    # Resample tapes within each task/profile, and keep the two conditions
    # on a common disturbance seed paired inside resamples.
    cells={}
    for i,r in enumerate(rows):
        cells.setdefault((r["task"],r["tape"]),[]).append(i)
    task_groups={}
    for (task,_),ix in cells.items():
        task_groups.setdefault(task,[]).append(np.asarray(ix))
    rng=np.random.default_rng(1057)
    draws=[]
    for _ in range(1000):
        keep=np.concatenate([np.concatenate([indices[i] for i in rng.integers(0,len(indices),len(indices))])
                             for indices in task_groups.values()])
        draws.append(ds[keep].mean())
    return dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())


def main(out,reps,jobs):
    global STORE,MODELS,LEARNER
    STORE=Store(out)
    MODELS={str(s):Model(s,True,True) for s in (4,8,12)} | {"single":Model(8,False,False),"blind":Model(8,True,False)}
    preflight=STORE.preflight()
    print("Geometry preflight",json.dumps(preflight),flush=True)
    train_jobs=[(src,task,prof,tape) for src in range(3) for task in TRAIN_TASKS for prof in PROFILES
                for tape in range(TRAIN_COUNT)]
    with ProcessPoolExecutor(max_workers=jobs) as p:
        samples=[pair for mini in p.map(training,train_jobs,chunksize=2) for pair in mini]
    x,y=map(np.array,zip(*samples))
    LEARNER=ExtraTreesRegressor(n_estimators=85,min_samples_leaf=11,max_features=.85,
                                random_state=19631,n_jobs=1).fit(x,y)
    dump(LEARNER,out/"paired_action_value.joblib")
    print("Trained",len(samples),"simulated paired action comparisons; starting Blender holdout",flush=True)
    tests=[(task,prof,tape) for task in TASKS for prof in PROFILES for tape in range(reps)]
    rows=[]
    with ProcessPoolExecutor(max_workers=jobs) as p:
        for row in p.map(evaluation,tests,chunksize=1):
            rows.append(row)
    with gzip.open(out/"paired_trials.jsonl.gz","wt",compresslevel=5) as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
    summary={"geometry":preflight,"bone_ray_samples_inside_population_grid_fraction":STORE.bone_support,
             "training":{"geometries":SOURCES[:3],"episodes":len(train_jobs),"candidate_actions":len(samples),
                         "train_action_advantage_positive_fraction":float((y>0).mean())},
             "test":{"geometry":SOURCES[TEST],"tapes_per_task_profile":reps,"trials":len(rows),
                     "conditions":PROFILES,"geometry_source_digest":digest(out/"geometry.npz")},
             "totals":totals(rows),"by_task":{task:totals([r for r in rows if r["task"]==task]) for task in TASKS},
             "by_profile":{kind:totals([r for r in rows if r["profile"]==kind]) for kind in PROFILES},
             "paired_vs_mpc_history":{arm:{field:bootstrap(rows,arm,"graph_bone_history",field)
                       for field in ("pass","wrong_delivery","correct_regions","time_s")}
                         for arm in ("rule","value","value_speed")},
             "paired_same_path":{f"{a}_vs_{b}":{field:bootstrap(rows,a,b,field)
                         for field in ("pass","wrong_delivery","correct_regions")}
                         for a,b in (("graph","target_only"),("graph_bone","graph"),("graph_bone_history","graph_bone"),
                                     ("sigma4_history","graph_bone_history"),("sigma12_history","graph_bone_history"))}}
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    print(json.dumps(summary["totals"],indent=2),flush=True)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("--out",type=Path,required=True)
    arg.add_argument("--reps",type=int,default=16)
    arg.add_argument("--jobs",type=int,default=12)
    a=arg.parse_args()
    main(a.out,a.reps,a.jobs)

"""Public-history model-based A/B planner; patient truth is an evaluator argument only."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import json
import sys
import time
import numpy as np
from scipy.special import ndtr,logsumexp
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/media/camp/EXT_DRIVE/Among_US')
from observation import environment as env
from observation.servo import candidate_actions,template_candidates
sys.path.insert(0,str(ROOT/'capability'))
from capability import CapabilityEvaluator
from DMP_playground.dmp_honest.dmp import fit_discrete_dmp

CFG=json.loads((Path(__file__).parent/'PROTOCOL.json').read_text())
FAIL=60.


class Motion:
    """Actual linear DMP basis; calibrated endpoint, no hidden anatomical target."""
    def __init__(self):
        self.phase=np.linspace(0.,1.,41)
        primitive=fit_discrete_dmp(self.phase,self.phase[:,None])
        values,_=primitive.rollout(self.phase)
        self.raw_endpoint=float(values[-1,0])
        self.shape=values[:,0]/self.raw_endpoint
    def path(self,start,target):
        return start[None]+self.shape[:,None]*(target-start)[None]
    def duration(self,start,target):
        path=self.path(np.asarray(start),np.asarray(target))
        steps=np.diff(path,axis=0)
        return float(max(.1,np.linalg.norm(steps[:,:2],axis=1).max()*40/.02,
                         np.abs(steps[:,2]).max()*40/.5))


MOTION=Motion()


def body_transform(pose,anchor):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,0,float(pose[2])]).as_matrix()
    T[:2,3]=pose[:2]
    return anchor@T


class Response:
    """Memoized legal-particle geometric response, isolated per policy episode."""
    def __init__(self,particles):
        self.particles=particles;self.cache={};self.geometry_queries=0
    @staticmethod
    def key(p):return tuple(np.round(np.asarray(p),7))
    def predict(self,poses):
        keys=[self.key(p) for p in poses]
        missing=list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            batch=env.predict_batch(self.particles,np.array(missing));self.geometry_queries+=len(missing)
            for j,k in enumerate(missing):self.cache[k]={a:b[:,j].copy() for a,b in batch.items()}
        return {a:np.stack([self.cache[k][a] for k in keys],axis=1) for a in self.cache[keys[0]]}


def quality_arrays(pred):
    """Per-hypothesis correct public five-frame stop, versus public stop event."""
    m=pred['means'];sd=env.SDS
    pu=ndtr((.002-m[...,0])/sd[0])-ndtr((-.002-m[...,0])/sd[0])
    pl=ndtr((m[...,2]-.015)/sd[2]);pi=ndtr((m[...,4]-.75)/sd[4])
    source_gate=pred['p_source']*pu*pl*pi
    public=(source_gate.sum(-1)*pred['contact'])**5
    geom=(pred['geometric_length'][...,0]>=.015)&(abs(m[...,0,0])<=.002)&(pred['angle_audit'][...,0]<=np.deg2rad(10))&pred['contact']
    correct=geom*source_gate[...,0]**5
    return correct,public,geom


def block_posterior(weights,frames,pred,a):
    ll=sum((env.log_likelihood_from_prediction(f,pred,a) for f in frames),np.zeros(len(weights)))
    lw=np.where(weights>0,np.log(np.maximum(weights,1e-300)),-np.inf)+ll
    norm=logsumexp(lw)
    if not np.isfinite(norm):raise ValueError('incompatible public model branch')
    return np.exp(lw-norm),float(norm),ll


def correct_given_public(weights,frames,pred,a):
    if not env.public_stop(frames):return 0.
    _,_,geom=quality_arrays(pred)
    full=sum((env.log_likelihood_from_prediction(f,pred,a) for f in frames),np.zeros(len(weights)))
    target=np.zeros(len(weights))
    mu=pred['means'][:,a,0]
    for f in frames:
        obs=np.array([f[x] for x in env.FIELDS])
        target+=env._feature_logpdf(obs,mu)
    target+=5*np.where(pred['p_source'][:,a,0]>0,np.log(np.maximum(pred['p_source'][:,a,0],1e-300)),-np.inf)
    target=np.where(geom[:,a],target,-np.inf)
    logw=np.where(weights>0,np.log(np.maximum(weights,1e-300)),-np.inf)
    return float(np.clip(np.exp(logsumexp(logw+target)-logsumexp(logw+full)),0,1))


class Robot:
    def __init__(self,evaluator,anchor,q_seed):
        self.evaluator=evaluator;self.anchor=anchor;self.q_seed=q_seed.copy();self.cache={};self.queries=0
        self.vmax=np.asarray(evaluator.kin.q_velocity_limits if hasattr(evaluator.kin,'q_velocity_limits') else [.4]+[2.512]*7)
    def endpoint(self,pose,q):
        key=Response.key(pose)
        if key not in self.cache:
            T=body_transform(pose,self.anchor)
            st=self.evaluator._static_pose(T,float(self.q_seed[0]),self.q_seed)
            proxy=self.evaluator.score(T,self.q_seed[0]);found=bool(st['found'])
            self.cache[key]=dict(found=found,q_kin=np.array(st['q_solution']) if found else None,
                proxy=float(np.asarray(proxy['raw_score']).reshape(-1)[0]),
                endpoint_scope='bounded static search; q_kin is NOT native q_out')
            self.queries+=1
        return self.cache[key]
    def costs(self,source,poses,q,*,use_cap=True):
        costs=[];rows=[]
        for p in poses:
            if use_cap:
                row=self.endpoint(p,q)
                if row['found']:
                    joint=float(np.max(np.abs(row['q_kin']-q)/self.vmax))
                    cost=max(MOTION.duration(source,p),joint)+.5
                else:cost=FAIL+.5
            else:row=None;cost=MOTION.duration(source,p)+.5
            costs.append(cost);rows.append(row)
        return np.asarray(costs),rows


@dataclass
class Decision:
    pose:np.ndarray
    value:float
    diagnostics:dict
    second:np.ndarray|None=None


class Planner:
    """Only public prior, history, measured/model robot state enter this class."""
    def __init__(self,particles,method,robot,*,seed=720503,samples=None,future_mode='compact'):
        self.particles=particles;self.method=method;self.robot=robot
        self.response=Response(particles);self.initial=np.ones(len(particles))/len(particles)
        self.seed=seed;self.calls=0;self.pending=None
        self.samples=CFG['planner']['model_observation_samples_per_root'] if samples is None else int(samples)
        self.future_mode=future_mode
    def candidates(self,pose,history,weights):
        return np.unique(np.round(candidate_actions(pose,history,self.particles,weights,
            max_candidates=CFG['shared_permissions']['max_candidates']),7),axis=0)
    def forecast_candidates(self,pose,history,weights):
        if self.future_mode=='same_full':return self.candidates(pose,history,weights)
        # Predictive terminal proposals depend on posterior geometry, not on
        # separately memorizing every noisy synthetic centroid sample.
        base=[pose.copy()]+template_candidates(self.particles,weights,pose)
        p=self.response.predict(base);extra=[]
        for j,target in enumerate(base[1:],1):
            u=float(weights@p['means'][:,j,0,0])
            centered=target.copy();centered[:2]+=u*np.array([np.cos(target[2]),np.sin(target[2])])
            extra.append(env.clip_pose(centered))
        return np.unique(np.round(base+extra,7),axis=0)
    def terminal(self,pose,history,weights,q,*,cap=True,remaining_time=60.,future=False):
        poses=self.forecast_candidates(pose,history,weights) if future else self.candidates(pose,history,weights)
        pred=self.response.predict(poses)
        correct,public,_=quality_arrays(pred)
        costs,rows=self.robot.costs(pose,poses,q,use_cap=cap)
        values=remaining_time-(remaining_time-costs)*(weights@correct)
        values=np.where(costs<=remaining_time,values,remaining_time+.5)
        if cap:
            values=np.where([r['found'] for r in rows],values,remaining_time+.5)
        i=int(np.argmin(values))
        if cap:
            ties=np.flatnonzero(np.abs(values-values[i])<=1e-9)
            i=int(max(ties,key=lambda j:rows[j]['proxy']))
        return float(min(remaining_time,values[i])),poses[i],dict(poses=poses,pred=pred,correct=correct,
            public=public,costs=costs,rows=rows,values=values)
    def decide(self,pose,history,weights,q,*,remaining_actions=6,remaining_time=60.):
        self.calls+=1
        if remaining_actions<=0 or remaining_time<=0:
            return Decision(None,max(0.,remaining_time),dict(kind='no_remaining_budget'))
        if self.method=='fixed_two' and self.pending is not None:
            target=self.pending;self.pending=None
            pending_cost=self.robot.costs(pose,[target],q,use_cap=True)[0][0]
            if pending_cost<=remaining_time:
                return Decision(target,0.,dict(kind='committed_fixed_second'))
            return Decision(None,remaining_time,dict(kind='fixed_second_no_longer_affordable'))
        usew=self.initial if self.method=='B_frozen_belief' else weights
        direct,selected,data=self.terminal(pose,history,usew,q,remaining_time=remaining_time)
        poses,pred,correct,public=data['poses'],data['pred'],data['correct'],data['public']
        if not any(r['found'] and c<=remaining_time for r,c in zip(data['rows'],data['costs'])):
            return Decision(None,remaining_time,dict(kind='no_affordable_action'))
        if self.method=='A_continuous_servo':
            utility=usew@public+.03*(usew@pred['p_visible'])
            utility=np.where([r['found'] and c<=remaining_time for r,c in zip(data['rows'],data['costs'])],utility,-np.inf)
            if not np.isfinite(utility).any():
                return Decision(None,remaining_time,dict(kind='no_affordable_action'))
            j=int(np.argmax(utility))
            return Decision(poses[j],float(data['values'][j]),dict(kind='continuous_image_servo',candidate_count=len(poses)))
        if self.method=='A_myopic' or remaining_actions==1:
            return Decision(selected,direct,dict(kind='myopic',candidate_count=len(poses)))
        h0=float(-np.sum(usew[usew>0]*np.log(usew[usew>0])))
        best=(direct,selected,dict(kind='direct',candidate_count=len(poses)),None)
        # Common stratified model scenarios; the sampled index generates Y only.
        # Every follow-up posterior below is recomputed against ALL particles.
        n=self.samples
        marks=(np.arange(n)+.5)/n
        hs=np.searchsorted(np.cumsum(usew),marks).clip(0,len(usew)-1)
        candidates=[]
        for ai,target in enumerate(poses):
            row=data['rows'][ai]
            if not row['found'] or data['costs'][ai]>remaining_time:continue
            qnext=row['q_kin'];cost=data['costs'][ai]
            if self.method=='fixed_two':
                future=self.forecast_candidates(target,history,usew)
                future_pred=self.response.predict(future)
                future_correct,_,_=quality_arrays(future_pred)
                c2,r2=self.robot.costs(target,future,qnext,use_cap=True)
                residual=usew*(1-public[:,ai])
                gain=np.maximum(0.,remaining_time-cost-c2)
                v=remaining_time-(remaining_time-cost)*float(usew@correct[:,ai])-residual@(gain[None,:]*future_correct)
                v=np.where([r['found'] for r in r2],v,remaining_time+.5)
                j=int(np.argmin(v));value=float(v[j])
                if value<best[0]-1e-9:best=(value,target,dict(kind='fixed_two',candidate_count=len(poses)),future[j])
                continue
            values=[];entropies=[];future_actions=[]
            for k,h in enumerate(hs):
                frames=env.sample_block_from_prediction(pred,int(h),ai,
                    (self.seed,self.calls,ai,k),start_time=cost-.5)
                post,_,_=block_posterior(usew,frames,pred,ai)
                entropy=-float(np.sum(post[post>0]*np.log(post[post>0])))
                entropies.append(entropy)
                if self.method=='entropy':
                    values.append(0.);future_actions.append(None)
                    continue
                if env.public_stop(frames):
                    values.append((remaining_time-cost)*(1-correct_given_public(usew,frames,pred,ai)))
                    future_actions.append(None)
                else:
                    bh=history+[(f,target.copy()) for f in frames]
                    value,action,_=self.terminal(target,bh,post,qnext,
                        cap=self.method!='B_no_future_cap',remaining_time=remaining_time-cost,future=True)
                    values.append(value);future_actions.append(action.tolist())
            value=float(cost+np.mean(values))
            candidates.append(dict(action=target.tolist(),value=value,entropy_after=float(np.mean(entropies))))
            if self.method=='entropy':
                metric=float((np.mean(entropies)-h0)/max(cost,.1))
            else:metric=value
            if self.method!='entropy' and metric<best[0]-1e-9:
                best=(metric,target,dict(kind='branch',candidate_count=len(poses),
                    model_branches=n,future_actions=future_actions),None)
        if self.method=='entropy' and candidates:
            candidates.sort(key=lambda d:((d['entropy_after']-h0)/max(MOTION.duration(pose,np.array(d['action']))+.5,.1)))
            target=np.array(candidates[0]['action'])
            best=(candidates[0]['value'],target,dict(kind='entropy',candidate_count=len(poses)),None)
        if self.method=='fixed_two':self.pending=best[3]
        diagnostic=dict(best[2]);diagnostic['root_candidates']=candidates
        return Decision(np.asarray(best[1]),float(best[0]),diagnostic,best[3])


def run_episode(private_world,particles,method,robot,world_key,*,max_actions=6):
    """Private world is used only in rendering/audit, never planner construction."""
    belief=env.Belief(particles,support='finite');weights=belief.weights.copy()
    planner=Planner(particles,method,robot)
    pose=np.zeros(3);q=robot.q_seed.copy();history=[];clock=0.;events=[]
    correct=False;false=False;ever=False;public_stopped=False;invalid=0
    for ordinal in range(max_actions):
        tick=time.perf_counter();decision=planner.decide(pose,history,weights,q,
            remaining_actions=max_actions-ordinal,remaining_time=max(0.,60.-clock))
        plan_s=time.perf_counter()-tick;clock+=plan_s
        if clock>=60. or decision.pose is None:break
        target=env.clip_pose(decision.pose)
        tick=time.perf_counter();endpoint=robot.endpoint(target,q)
        endpoint_s=time.perf_counter()-tick;clock+=endpoint_s
        if not endpoint['found']:
            invalid+=1;clock+=.1
            events.append(dict(ordinal=ordinal,reason='bounded endpoint search not found',
                target=target.tolist(),planning_s=plan_s))
            break
        duration=max(MOTION.duration(pose,target),float(np.max(np.abs(endpoint['q_kin']-q)/robot.vmax)))
        path=MOTION.path(pose,target)
        # Native execution is validated separately. This actor is kinematic L0.
        pose=path[-1].copy();q=endpoint['q_kin'].copy()
        if clock+duration+.5>60.:clock=60.;break
        clock+=duration
        frames,audits=env.render_block(private_world,pose,(world_key,ordinal,tuple(np.round(pose,7))),start_time=clock)
        # Acquisition IDs must not expose the private selected catalogue index.
        for i,(f,audit_frame) in enumerate(zip(frames,audits)):
            f['frame_id']=audit_frame['frame_id']=f'acq-{ordinal}-frame-{i}'
        before=weights.copy()
        tick=time.perf_counter()
        for f in frames:belief.update(f,pose);history.append((f,pose.copy()))
        belief_s=time.perf_counter()-tick
        weights=belief.weights.copy();clock+=.5+belief_s
        stopped=env.public_stop(frames);audit=env.audit_stop(audits);ever|=audit
        timely=clock<=60.+1e-12
        events.append(dict(ordinal=ordinal,target=target.tolist(),actual_pose=pose.tolist(),
            q_kin=q.tolist(),q_state_kind='static endpoint realization, not native q_out',
            planning_s=plan_s,endpoint_s=endpoint_s,belief_update_s=belief_s,motion_s=duration,clock_s=clock,public=frames,
            audit_for_evaluation_only=audits,posterior=weights.tolist(),prior_before=before.tolist(),
            public_stop=stopped,timely_public_decision=timely,audit_correct=audit,decision=decision.diagnostics))
        if stopped and timely:
            public_stopped=True;correct=audit;false=not audit;break
        if not timely:break
    return dict(method=method,correct_public_stop=correct,false_public_stop=false,
        public_stop=public_stopped,ever_audit_quality=ever,
        rmt_s=float(clock if correct else 60.),elapsed_s=float(clock),
        invalid_endpoint_attempts=invalid,actions=len(events),events=events,
        endpoint_queries=robot.queries,model_geometry_queries=planner.response.geometry_queries,
        posterior=weights.tolist(),scope='L0 geometric sensing plus real static robot endpoint search')

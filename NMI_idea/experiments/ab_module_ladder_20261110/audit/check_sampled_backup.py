"""Independent unnormalized-mass backup on one fixed DEV/public robot state.

Planner.decide is invoked once as the implementation UNDER TEST. The reference
does not call Planner.decide, terminal, block_posterior or correct_given_public.
It independently enumerates finite sample branches and all latent/source masses,
using the same public candidate specification and real static robot witnesses.
This checks sampled-kernel arithmetic, not continuous integral convergence or
hardware performance. No locked catalogue and no native controller are used.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.special import logsumexp, ndtr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'planning'))
import system as impl
env=impl.env
from observation.servo import candidate_actions,template_candidates

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'planning/system.py',
    ROOT/'planning/PROTOCOL.json',ROOT/'observation/environment.py',
    ROOT/'observation/servo.py',ROOT/'capability/capability.py']}

particles=env.sample_prior(720501,16)
weights=np.ones(16)/16
pose=np.zeros(3)
history=[]
remaining=60.
n=32
evaluator=impl.CapabilityEvaluator()
candidates=json.loads((ROOT/'capability/frozen_public/candidates.json').read_text())
checks=json.loads((ROOT/'capability/endpoint_checks.json').read_text())
j=next(j for j,r in enumerate(candidates) if r['id']=='a0_identity')
assert checks[j]['found']
q=np.asarray(checks[j]['q_solution'])
anchor=evaluator.fk(q)
robot=impl.Robot(evaluator,anchor,q)
sut=impl.Planner(particles,'B_two_step',robot,samples=n,future_mode='compact')
started=time.perf_counter()
got=sut.decide(pose,history,weights,q,remaining_actions=6,remaining_time=remaining)
print('implementation completed',time.perf_counter()-started,flush=True)

def predict(poses):
    # Response's declared key quantization is part of the model being compared.
    return env.predict_batch(particles,np.round(np.asarray(poses),7))

def geometry_and_probabilities(pred):
    mu=pred['means']
    pu=ndtr((.002-mu[...,0])/env.SDS[0])-ndtr((-.002-mu[...,0])/env.SDS[0])
    length=ndtr((mu[...,2]-.015)/env.SDS[2])
    identity=ndtr((mu[...,4]-.75)/env.SDS[4])
    one_source=pred['p_source']*pu*length*identity
    geom=(pred['geometric_length'][...,0]>=.015)&(abs(mu[...,0,0])<=.002)&(
        pred['angle_audit'][...,0]<=np.deg2rad(10))&pred['contact']
    correct=geom*one_source[...,0]**5
    public=(np.sum(one_source,axis=-1)*pred['contact'])**5
    return geom,correct,public

def full_candidates(at,hist,w):
    return np.unique(np.round(candidate_actions(at,hist,particles,w,
        max_candidates=impl.CFG['shared_permissions']['max_candidates']),7),axis=0)

def compact_candidates(at,w):
    base=[at.copy()]+template_candidates(particles,w,at)
    pr=predict(base)
    extra=[]
    for k,target in enumerate(base[1:],1):
        u=float(w@pr['means'][:,k,0,0])
        centered=target.copy()
        centered[:2]+=u*np.array([np.cos(target[2]),np.sin(target[2])])
        extra.append(env.clip_pose(centered))
    return np.unique(np.round(base+extra,7),axis=0)

def transition_cost(at,targets,qin):
    cs=[];rs=[]
    for target in targets:
        row=robot.endpoint(target,qin)
        if row['found']:
            path=at[None]+impl.MOTION.shape[:,None]*(target-at)[None]
            diff=np.diff(path,axis=0)
            motion=max(.1,np.max(np.linalg.norm(diff[:,:2],axis=1))*40/.02,
                np.max(abs(diff[:,2]))*40/.5)
            joint=np.max(abs(row['q_kin']-qin)/robot.vmax)
            cost=max(motion,joint)+.5
        else:cost=60.5
        cs.append(cost);rs.append(row)
    return np.asarray(cs),rs

def feature_logdens(f,means):
    observed=np.asarray([f[k] for k in env.FIELDS])
    d=observed-means
    d[...,3]=(d[...,3]+np.pi/2)%np.pi-np.pi/2
    terms=-.5*(d/env.SDS)**2-np.log(env.SDS*np.sqrt(2*np.pi))
    for k,bound,sign in [(2,0.,-1),(4,0.,-1),(4,1.,1)]:
        if observed[k]==bound:
            terms[...,k]=np.log(np.maximum(ndtr(sign*(means[...,k]-bound)/env.SDS[k]),1e-300))
    return terms.sum(-1)

def branch_masses(frames,pred,ai):
    ps=pred['p_source'][:,ai]
    with np.errstate(divide='ignore'):
        lps=np.log(ps)
        log_prior=np.log(weights)
    all_ll=np.zeros(len(weights));target_ll=np.zeros(len(weights))
    for f in frames:
        if not f['visible']:
            with np.errstate(divide='ignore'):
                all_ll+=np.log(1-ps.sum(-1))
            target_ll[:]=-np.inf
        else:
            source=lps+feature_logdens(f,pred['means'][:,ai])
            all_ll+=logsumexp(source,axis=-1)
            target_ll+=source[:,0]
        all_ll=np.where(pred['contact'][:,ai]==bool(f['contact']),all_ll,-np.inf)
    log_mass=log_prior+all_ll
    scale=np.max(log_mass)
    mass=np.exp(log_mass-scale)
    target_mass=np.exp(log_prior+target_ll-scale)
    return mass,target_mass

actions=full_candidates(pose,history,weights)
pred=predict(actions)
geom,correct,public=geometry_and_probabilities(pred)
costs,rows=transition_cost(pose,actions,q)
direct=remaining-(remaining-costs)*(weights@correct)
direct=np.where([r['found'] and c<=remaining for r,c in zip(rows,costs)],direct,remaining+.5)
direct_min=float(min(remaining,np.min(direct)))
ties=np.flatnonzero(abs(direct-np.min(direct))<=1e-9)
direct_index=int(max(ties,key=lambda i:rows[i]['proxy']))
best_value=direct_min;best_action=actions[direct_index];best_kind='direct'
latent_indices=np.searchsorted(np.cumsum(weights),(np.arange(n)+.5)/n).clip(0,len(weights)-1)
reference=[]
branch_count=0;stop_count=0
min_mass=1.;max_posterior_error=0.
for ai,target in enumerate(actions):
    if not rows[ai]['found'] or costs[ai]>remaining:continue
    c=float(costs[ai]);rnext=remaining-c;values=[]
    for k,h in enumerate(latent_indices):
        frames=env.sample_block_from_prediction(pred,int(h),ai,
            (impl.CFG['planner']['branch_seed'],1,ai,k),start_time=c-.5)
        mass,target_mass=branch_masses(frames,pred,ai)
        z=float(mass.sum());post=mass/z
        min_mass=min(min_mass,z)
        branch_count+=1
        if env.public_stop(frames):
            # Sum target-source-only legal masses over ALL h before dividing.
            true_mass=float(np.sum(target_mass*geom[:,ai]))
            values.append(rnext*(1-true_mass/z));stop_count+=1
        else:
            follow=compact_candidates(target,post)
            follow_pred=predict(follow)
            _,future_correct,_=geometry_and_probabilities(follow_pred)
            c2,r2=transition_cost(target,follow,rows[ai]['q_kin'])
            # The latent masses remain unnormalized inside the terminal backup.
            losses=rnext-(rnext-c2[None,:])*future_correct
            value_by_action=(mass@losses)/z
            valid=np.array([r['found'] for r in r2])&(c2<=rnext)
            value_by_action=np.where(valid,value_by_action,rnext+.5)
            values.append(min(rnext,float(value_by_action.min())))
    v=c+float(np.mean(values))
    reference.append(dict(action=target.tolist(),value=v))
    if v<best_value-1e-9:
        best_value=v;best_action=target;best_kind='branch'

actual=got.diagnostics['root_candidates']
assert len(reference)==len(actual)
errors=[]
for a,b in zip(actual,reference):
    np.testing.assert_array_equal(a['action'],b['action'])
    errors.append(abs(a['value']-b['value']))
np.testing.assert_allclose(got.pose,best_action,rtol=0,atol=0)
np.testing.assert_allclose(got.value,best_value,rtol=0,atol=1e-9)
assert max(errors,default=0)<1e-9
assert sources=={name:sha(ROOT/name) for name in sources},'source changed during review'
report=dict(scope='DEV finite sampled-kernel arithmetic; real static witnesses; no native/hardware/test catalogue',
    dev_seed=720501,particles=16,model_samples=32,context='a0_identity',remaining_s=60.,
    actual_decision=got.pose.tolist(),reference_decision=best_action.tolist(),
    actual_value=got.value,reference_value=best_value,actual_kind=got.diagnostics['kind'],
    reference_kind=best_kind,root_candidates=len(actions),branches=branch_count,
    sampled_public_stops=stop_count,max_root_value_abs_error=max(errors,default=0.),
    root_values=reference,static_endpoint_query_count=robot.queries,
    native_calls=0,private_world_instantiated=False,source_sha256=sources,
    robot_manifest=evaluator.manifest(),passed=True,
    warning='Same 32 sampled observations; not a convergence certificate for the continuous predictive expectation.')
(HERE/'sampled_backup_check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['passed','branches','sampled_public_stops','max_root_value_abs_error',
    'actual_value','reference_value','actual_kind','static_endpoint_query_count']},indent=2),flush=True)

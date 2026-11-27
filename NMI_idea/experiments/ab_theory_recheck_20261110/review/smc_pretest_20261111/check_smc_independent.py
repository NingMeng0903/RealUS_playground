"""Independent bounded SMC audit. Only writes its own audit directory.

No locked test cohort is instantiated. Numerical references do not call the
production beta scheduler, CESS function, or MH kernel to compute expected
answers. Pass the sealed source hash to avoid auditing a changing file.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import math
import sys
import traceback
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


ROOT = Path('/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_theory_recheck_20261110')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT/'inference/continuous_smc.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reference_cess(w, ll, delta):
    w=np.asarray(w, float); ll=np.asarray(ll, float)
    if delta == 0:
        return float(len(w))
    v=delta*ll
    finite=np.isfinite(v) & (w>0)
    if not finite.any():
        return 0.
    # Direct scaled finite sums, independent from logsumexp implementation.
    v0=max(v[finite]); u=np.zeros(len(w)); u[finite]=np.exp(v[finite]-v0)
    return float(len(w)*(w@u)**2/(w@(u*u)))


def check_cess(smc):
    ws=[np.array([.91,.03,.02,.02,.01,.01]),np.array([.35,.3,.2,.1,.05,0.])]
    lls=[np.array([-30.,-2.,0.,3.,8.,10.]), np.full(6,-10000.)]
    errs=[]
    for w in ws:
        assert 1/(w@w) < .7*len(w)
        for ll in lls:
            for delta in (0.,1e-9,.01,.2,1.):
                expected=reference_cess(w,ll,delta)
                actual=smc._cess(w,ll,delta)
                errs.append(abs(expected-actual))
                assert abs(expected-actual)<2e-9,(w,ll,delta,expected,actual)
            assert abs(smc._cess(w,ll,0.)-len(w))<1e-12
    w=ws[0]; ll=lls[0]
    for d in (.01,.2,.7):
        assert abs(smc._cess(w,ll,d)-smc._cess(w,ll+123.4,d))<1e-10
    return {'max_direct_formula_error':max(errs),'nonuniform_delta0_equals_N':True,
            'constant_likelihood_equals_N':True}


def check_scheduler(smc):
    w=np.array([.91,.03,.02,.02,.01,.01]); ll=np.array([-30.,-2.,0.,3.,8.,10.])
    target=.7*len(w); lo,hi=0.,1.
    for _ in range(100):
        mid=(lo+hi)/2
        if reference_cess(w,ll,mid)>=target: lo=mid
        else: hi=mid
    beta,cess=smc.adaptive_next_beta(0.,w,ll,target_cess=target)
    assert abs(beta-lo)<1e-9,(beta,lo)
    assert abs(cess-target)<1e-7
    assert smc.adaptive_next_beta(0.,w,np.full(6,-3.5),target_cess=target)[0]==1.
    # Hard zero support is a real discontinuity, not a plateau in raw ESS.
    hz=np.array([0.,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf])
    hz_w=np.full(6,1/6)
    hb,hc=smc.adaptive_next_beta(0.,hz_w,hz,target_cess=target)
    assert hb>0. and hc<target
    return {'independent_beta':lo,'implementation_beta':beta,'cess':cess,
            'hard_zero_positive_step':hb,'hard_zero_cess':hc}


def no_move(x,rng,scale):
    return x


def check_exact_weight_and_logz(smc):
    w=np.array([.47,.19,.11,.09,.07,.05,.02]); xs=list(range(7))
    l1=np.array([-8.,2.,-1.,4.,.2,-3.,5.])
    l2=np.array([3.,-4.,2.,-3.,.4,1.,-6.])
    rng=np.random.default_rng(720991)
    def run(weights,ll,hist):
        return smc.adaptive_tempered_resample_move(xs,weights,
            lambda ps:ll[np.asarray(ps,int)], lambda ps:hist[np.asarray(ps,int)],
            lambda x:0.,no_move,rng,resample_fraction=.001,mh_sweeps=0,
            max_temper_steps=100)
    u1=run(w,l1,np.zeros(7))
    assert u1.committed and u1.diagnostics['resample_count']==0
    assert not u1.diagnostics['initial_preemptive_resample']
    assert all(abs(stage['cess_target']-.7*len(xs))<1e-12 for stage in u1.diagnostics['stages'])
    z1=float(logsumexp(np.log(w)+l1))
    expected1=np.exp(np.log(w)+l1-z1)
    assert np.max(np.abs(u1.weights-expected1))<1e-12
    assert abs(u1.diagnostics['attempted_logZ_increment']-z1)<1e-12
    u2=run(u1.weights,l2,l1)
    z12=float(logsumexp(np.log(w)+l1+l2))
    assert u2.committed
    assert all(abs(stage['cess_target']-.7*len(xs))<1e-12 for stage in u2.diagnostics['stages'])
    assert np.max(np.abs(u2.weights-np.exp(np.log(w)+l1+l2-z12)))<1e-12
    assert abs(u1.diagnostics['attempted_logZ_increment']+u2.diagnostics['attempted_logZ_increment']-z12)<1e-12
    uconstant=run(w,np.full(7,-7.3),np.zeros(7))
    assert uconstant.committed
    assert np.max(np.abs(uconstant.weights-w))<1e-12
    assert uconstant.diagnostics['temperature_path']==[0.,1.]
    assert abs(uconstant.diagnostics['attempted_logZ_increment']+7.3)<1e-12
    return {'first_logZ':z1,'joint_logZ':z12,
            'first_stages':u1.diagnostics['stage_count'],'second_stages':u2.diagnostics['stage_count'],
            'exact_telescoping_and_weights':True}


class Uniforms:
    def __init__(self, values): self.values=np.asarray(values)
    def random(self,n):
        assert n==len(self.values)
        return self.values.copy()


def check_full_history_mh(smc):
    # Target log pi_beta(x)=.3x-1.5x+.4*(2x)=-.4x, x=0/1.
    # Symmetric proposal flips x. Exact acceptance 0->1=e^(-.4), 1->0=1.
    start=[0,0,1,1]
    moved,rate,count=smc._mcmc_move(start,.4,
        lambda ps:2*np.asarray(ps),lambda ps:-1.5*np.asarray(ps),
        lambda x:.3*x,lambda x,rng,scale:1-x,
        Uniforms([.6,.7,.999,.01]),sweeps=1,scales=(1.,))
    assert moved==[1,0,0,0],moved
    assert count==3 and rate==.75
    ratio=math.exp(-.4); pi=np.array([1.,ratio])/(1+ratio)
    K=np.array([[1-ratio,ratio],[1.,0.]])
    assert np.max(np.abs(pi@K-pi))<1e-12
    # Constant likelihood + non-resampled weighted population: MH does not
    # silently reset carried importance weights to uniform.
    w=np.array([.4,.3,.2,.1])
    upd=smc.adaptive_tempered_resample_move(start,w,
        lambda ps:np.full(len(ps),-1.7),lambda ps:-1.5*np.asarray(ps),
        lambda x:.3*x,lambda x,rng,scale:1-x,
        np.random.default_rng(720992),resample_fraction=.001,mh_sweeps=1)
    assert upd.committed and upd.diagnostics['resample_count']==0
    assert np.max(np.abs(upd.weights-w))<1e-12
    return {'full_prior_plus_history_plus_beta_block_acceptance':rate,
            'expected_states':moved,'weighted_move_preserves_weights':True,
            'two_state_invariance_error':float(np.max(np.abs(pi@K-pi)))}


def check_endpoint_commit(smc):
    xs=[-1.,0.,1.]; w=np.array([.2,.3,.5])
    with patch.object(smc,'adaptive_next_beta',return_value=(1.-5e-10,3.)):
        u=smc.adaptive_tempered_resample_move(xs,w,
            lambda ps:np.zeros(len(ps)),lambda ps:np.zeros(len(ps)),
            lambda x:0.,no_move,np.random.default_rng(720993),
            resample_fraction=.001,mh_sweeps=0,max_temper_steps=1)
    assert not u.committed,'A beta<1 target was relabeled complete'
    assert u.diagnostics['temperature_path'][-1]<1.
    assert np.array_equal(u.weights,w)
    return {'near_one_beta_still_uncommitted':u.diagnostics['temperature_path'][-1]}


class ProposalRng:
    def __init__(self, block, source, sign, component=0):
        self.values=iter([block,source,component]); self.sign=sign
    def integers(self,*args,**kwargs): return next(self.values)
    def normal(self,loc=0.,scale=1.,size=None):
        z=np.full(size,self.sign*.37) if size is not None else self.sign*.37
        return loc+scale*z


def world_bytes(w):
    fields=('registration','modes','y_controls','z_controls','depths','identity_z','detect_z','points')
    return b''.join(np.asarray(getattr(w,k)).tobytes() for k in fields)


def check_proposal(smc):
    env=smc.env; w=env.sample_prior(720901,1)[0]; before=world_bytes(w)
    maxerr=0.
    for block in range(7):
        for source in (0,1):
            p=env._propose(w,ProposalRng(block,source,+1),.8)
            back=env._propose(p,ProposalRng(block,source,-1),.8)
            maxerr=max(maxerr,float(np.max(np.abs(back.points-w.points))))
            assert np.allclose(back.points,w.points,atol=2e-16,rtol=0)
            assert np.allclose(back.registration,w.registration,atol=2e-16,rtol=0)
            assert np.array_equal(back.modes,w.modes)
    assert world_bytes(w)==before
    d=w.depths.copy(); d[0]=.08
    outside=env._world(w.registration.copy(),w.modes.copy(),w.y_controls.copy(),
        w.z_controls.copy(),d,w.identity_z.copy(),w.detect_z.copy())
    assert env._logprior(outside)==-np.inf
    return {'all_7_blocks_roundtrip_max_point_error_m':maxerr,
            'input_world_immutable':True,'unclipped_depth_proposal_rejected_by_prior':True,
            'source_review':'Gaussian zero-mean increments and symmetric sign flips, state-independent block/source selection; no truncation/resampling of invalid proposals.'}


def check_geometry_likelihood(smc):
    env=smc.env; particles=env.sample_prior(720901,6)
    # Only the public protocol DEV seed; never instantiate locked 720913.
    world=env.sample_worlds(720902,1)[0]
    poses=[np.array([0.,-.01,0.]),np.array([0.,.01,.2])]
    pairs=[]
    for j,p in enumerate(poses):
        frames,_=env.render_block(world,p,('independent-dev-likelihood',j))
        pairs += [(f,p.copy()) for f in frames]
    actual=smc.ContinuousBelief._loglike_frames(pairs,particles)
    expected=np.array([sum(env.log_likelihood(f,p,q) for f,q in pairs) for p in particles])
    assert np.array_equal(np.isneginf(actual),np.isneginf(expected))
    ok=np.isfinite(actual)&np.isfinite(expected)
    assert np.any(ok)
    err=float(np.max(np.abs(actual[ok]-expected[ok])))
    assert err<1e-8
    h_ll,b_ll=smc.ContinuousBelief._joint_loglikes(pairs[:5],pairs[5:],particles)
    for sub,actual_ll in ((pairs[:5],h_ll),(pairs[5:],b_ll)):
        expected_ll=np.array([sum(env.log_likelihood(f,p,q) for f,q in sub) for p in particles])
        assert np.array_equal(np.isneginf(actual_ll),np.isneginf(expected_ll))
        finite=np.isfinite(actual_ll)&np.isfinite(expected_ll)
        assert np.max(np.abs(actual_ll[finite]-expected_ll[finite]))<1e-8
    original=smc.ContinuousBelief._prediction_for_poses
    calls=[]
    def counted(ps,poses,*,backend='numpy'):
        calls.append((tuple(map(tuple,poses)),sha_bytes(b''.join(world_bytes(p) for p in ps))))
        return original(ps,poses,backend=backend)
    with patch.object(smc.ContinuousBelief,'_prediction_for_poses',staticmethod(counted)):
        smc.ContinuousBelief._loglike_frames(pairs,particles)
        altered=[env._propose(p,np.random.default_rng(720994+i),1.) for i,p in enumerate(particles)]
        smc.ContinuousBelief._loglike_frames(pairs,altered)
    assert len(calls)==2 and calls[0][1]!=calls[1][1]
    return {'ten_public_frames_two_poses_max_loglike_error':err,
            'fresh_particle_geometry_cache_checked':True}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def check_wrapper_history_transaction(smc):
    worlds=smc.env.sample_prior(720901,4)
    b=smc.ContinuousBelief(worlds,mh_sweeps=0)
    groups=[[{'frame_id':f'a{i}','v':v} for i,v in enumerate((1.,2.,3.))],
            [{'frame_id':f'b{i}','v':v} for i,v in enumerate((4.,5.))]]
    pose=np.array([0.,0.,0.]); calls=[]
    def synthetic_ll(pairs,particles,**kwargs):
        return sum(f['v'] for f,_ in pairs)*(1+np.arange(len(particles)))
    def synthetic_joint(history,block,particles,**kwargs):
        return synthetic_ll(history,particles),synthetic_ll(block,particles)
    def engine(particles,weights,block_ll,history_ll,*args,**kwargs):
        calls.append({'block':block_ll(particles).copy(),'history':history_ll(particles).copy()})
        if kwargs.get('joint_loglike') is not None:
            hist,block=kwargs['joint_loglike'](particles)
            assert np.array_equal(hist,calls[-1]['history'])
            assert np.array_equal(block,calls[-1]['block'])
        return smc.TemperedUpdate(list(particles),np.asarray(weights).copy(),
            {'posterior_valid':True,'unfinished':False,'temperature_path':[0.,1.],
             'attempted_logZ_increment':.25},True)
    with patch.object(smc.ContinuousBelief,'_loglike_frames',staticmethod(synthetic_ll)),patch.object(smc.ContinuousBelief,'_joint_loglikes',staticmethod(synthetic_joint)),patch.object(smc,'adaptive_tempered_resample_move',side_effect=engine):
        b.update_block(groups[0],pose)
        b.update_block(groups[1]+[groups[0][0],groups[1][0]],pose)
        repeat=b.update_block(groups[0]+groups[1],pose)
    assert len(calls)==2
    assert np.array_equal(calls[0]['history'],np.zeros(4))
    assert np.array_equal(calls[0]['block'],6*(1+np.arange(4)))
    assert np.array_equal(calls[1]['history'],6*(1+np.arange(4)))
    assert np.array_equal(calls[1]['block'],9*(1+np.arange(4)))
    assert len(b.history)==5 and len(b.seen)==5 and b.logZ==.5
    assert repeat['skipped']
    before=(world_bytes(b.particles[0]),b.weights.copy(),len(b.history),b.seen.copy(),b.logZ)
    def failed(particles,weights,*args,**kwargs):
        return smc.TemperedUpdate(list(particles),np.asarray(weights).copy(),
            {'posterior_valid':False,'unfinished':True,'temperature_path':[0.,.4],
             'attempted_logZ_increment':88.},False)
    with patch.object(smc,'adaptive_tempered_resample_move',side_effect=failed):
        failure=b.update_block([{'frame_id':'uncommitted','v':9.}],pose)
    assert failure['history_committed'] is False
    assert world_bytes(b.particles[0])==before[0] and np.array_equal(b.weights,before[1])
    assert len(b.history)==before[2] and b.seen==before[3] and b.logZ==before[4]
    assert 'uncommitted' not in b.seen
    with patch.object(smc.ContinuousBelief,'_loglike_frames',staticmethod(lambda pairs,ps,**kw:np.full(len(ps),-np.inf))):
        incompatible=b.update_block([{'frame_id':'zero-support','v':1.}],pose)
    assert incompatible['failure']=='all_particles_incompatible'
    assert not incompatible['history_committed'] and incompatible['temperature_path']==[0.]
    assert 'zero-support' not in b.seen and b.logZ==before[4]
    return {'unique_frame_count':len(b.history),'duplicate_calls_skipped':True,
            'full_old_history_and_new_block_separate':True,'uncommitted_state_rolled_back':True,
            'hard_zero_support_returns_failed_uncommitted_diagnostic':True}


def check_cuda_predictor(smc):
    gpu=importlib.import_module('inference.gpu_predictor')
    assert gpu.available(),'CUDA test requested but unavailable'
    env=smc.env; ps=env.sample_prior(720901,8)
    poses=np.array([[0.,-.02,0.],[0.,.02,.4],[.04,0.,-.3],[.041,0.,0.],[0.,0.,0.]])
    cpu=env.predict_batch(ps,poses); cuda=gpu.predict_batch(ps,poses)
    errors={}
    for key in cpu:
        if np.asarray(cpu[key]).dtype==bool:
            assert np.array_equal(cpu[key],cuda[key]);errors[key]=0.
        else:
            diff=np.asarray(cpu[key])-np.asarray(cuda[key])
            if key=='means':diff[...,3]=(diff[...,3]+np.pi/2)%np.pi-np.pi/2
            err=float(np.max(np.abs(diff)));errors[key]=err
            assert err<1e-8,(key,err)
    world=env.sample_worlds(720902,1)[0]; llerrs=[]
    for a,pose in enumerate(poses):
        frames,_=env.render_block(world,pose,('audit-cuda-dev',a))
        for f in frames:
            x=env.log_likelihood_from_prediction(f,cpu,a)
            y=env.log_likelihood_from_prediction(f,cuda,a)
            assert np.array_equal(np.isneginf(x),np.isneginf(y))
            good=np.isfinite(x)&np.isfinite(y)
            if np.any(good):llerrs.append(float(np.max(np.abs(x[good]-y[good]))))
    assert max(llerrs)<1e-7
    return {'predictor_field_max_errors':errors,'max_public_frame_loglike_error':max(llerrs),
            'gpu_source_sha':sha(ROOT/'inference/gpu_predictor.py')}


def check_sequential_gaussian(smc):
    ys=np.array([1.2,-.2]); sigmas=np.array([.4,.6])
    precision=1+np.sum(1/sigmas**2)
    expected_var=1/precision; expected_mean=expected_var*np.sum(ys/sigmas**2)
    evidence=float(multivariate_normal.logpdf(ys,mean=np.zeros(2),cov=np.ones((2,2))+np.diag(sigmas**2)))
    records=[]
    def ll(xs,y,s):
        return -.5*((np.asarray(xs)-y)/s)**2-math.log(s*math.sqrt(2*math.pi))
    for seed in range(720995,720999):
        rng=np.random.default_rng(seed); particles=list(rng.normal(size=4096)); weights=np.full(4096,1/4096)
        total_z=0.; history=[]
        for y,s in zip(ys,sigmas):
            def hist(xs):
                return sum((ll(xs,hy,hs) for hy,hs in history),np.zeros(len(xs)))
            up=smc.adaptive_tempered_resample_move(particles,weights,
                lambda xs:ll(xs,y,s),hist,lambda x:-.5*x*x,
                lambda x,rng,scale:float(x+rng.normal(0,.3*scale)),rng,
                max_temper_steps=30,mh_sweeps=8)
            assert up.committed,up.diagnostics
            particles,weights=up.particles,up.weights
            total_z+=up.diagnostics['attempted_logZ_increment']; history.append((y,s))
        x=np.asarray(particles); mean=float(weights@x); var=float(weights@((x-mean)**2))
        records.append({'seed':seed,'mean':mean,'variance':var,'logZ':total_z})
        assert abs(mean-expected_mean)<.06 and abs(var-expected_var)<.025
        assert abs(total_z-evidence)<.15
    assert abs(np.mean([r['mean'] for r in records])-expected_mean)<.035
    assert abs(np.mean([r['logZ'] for r in records])-evidence)<.08
    return {'analytic_mean':expected_mean,'analytic_variance':expected_var,
            'analytic_logZ':evidence,'four_seeds':records,
            'interpretation':'Monte Carlo agreement, not exact finite-N evidence or coverage proof.'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args(); before=sha(SOURCE)
    if before!=args.expected_sha: raise SystemExit('Source differs from sealed expected SHA')
    sys.path.insert(0,str(ROOT)); smc=importlib.import_module('inference.continuous_smc')
    tests=[check_cess,check_scheduler,check_exact_weight_and_logz,check_full_history_mh,
           check_endpoint_commit,check_proposal,check_geometry_likelihood,
           check_wrapper_history_transaction,check_sequential_gaussian,check_cuda_predictor]
    results={}
    for fn in tests:
        try:
            result=fn(smc); results[fn.__name__]={'passed':True,'result':result}
        except Exception as e:
            results[fn.__name__]={'passed':False,'error':repr(e),'traceback':traceback.format_exc()}
    after=sha(SOURCE)
    report={'source':str(SOURCE),'sha_before':before,'sha_after':after,
            'source_unchanged':before==after,'tests':results,
            'all_passed':before==after and all(v['passed'] for v in results.values()),
            'scope':'No locked 720913 cohort, no native execution; fixed DEV only for likelihood API.'}
    (OUT/'INDEPENDENT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'all_passed':report['all_passed'],
                      'tests':{k:v['passed'] for k,v in results.items()},'source_sha':before},indent=2))
    if not report['all_passed']: raise SystemExit(1)


if __name__=='__main__': main()

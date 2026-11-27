"""Bounded independent density, target and trained-model checks; no training."""
from pathlib import Path
import hashlib, json, sys, importlib.util
from unittest.mock import patch
import numpy as np
from scipy.linalg import block_diag
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, uniform

ROOT=Path('/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_theory_recheck_20261110')
sys.path.insert(0,str(ROOT))
from inference import informed_proposal as ip

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def proposal_checks():
    world=ip.env.sample_worlds(721111,1)[0]
    # Known-world placement is only a density unit-test fixture, never a
    # controller/policy success experiment. Ensure nontrivial visible LLs.
    points=world.points[0];mid=len(points)//2;direction=points[mid+1]-points[mid-1]
    pose=np.r_[points[mid,:2],np.arctan2(direction[1],direction[0])]
    frames,_=ip.env.render_block(world,pose,('independent-density-visible',0),start_time=0.)
    assert frames[0]['visible'] and frames[1]['visible']
    history=[(f,pose.copy()) for f in frames[:2]]
    p=ip.InformedProposal(history)
    rng=np.random.default_rng(721112)
    errs=[]
    for c in [p.components[0],p.components[-1]]:
        xs=np.stack([c.mean,c.mean+c.chol@rng.normal(size=28),c.mean+c.chol@rng.normal(size=28)])
        expected=multivariate_normal.logpdf(xs,mean=c.mean,cov=c.covariance)
        actual=ip._log_gaussian(xs,c.mean,c.chol,c.logdet)
        errs.extend(abs(expected-actual))
    xs,ms=ip.sample_prior_vectors(3,seed=721113)
    indices=np.r_[np.arange(22),np.arange(24,28)]
    cov=block_diag(.004**2*np.eye(2),.003**2*ip.CHOL@ip.CHOL.T,.003**2*ip.CHOL@ip.CHOL.T,
                   .0015**2*ip.CHOL@ip.CHOL.T,.0015**2*ip.CHOL@ip.CHOL.T,np.eye(4))
    lprior=multivariate_normal.logpdf(xs[:,indices],mean=np.zeros(26),cov=cov)
    lprior+=uniform.logpdf(xs[:,22:24],loc=.025,scale=.020).sum(axis=1)-np.log(16)
    prior_error=float(np.max(abs(lprior-p.log_prior(xs,ms))))
    qrefs=[]
    for x,m,lp in zip(xs,ms,lprior):
        terms=[np.log(c.weight)+multivariate_normal.logpdf(x,mean=c.mean,cov=c.covariance)
               for c in p.components if np.array_equal(np.array(c.mode).reshape(2,2),m)]
        qrefs.append(np.logaddexp(np.log(.1)+lp,np.log(.9)+logsumexp(terms)))
    q_error=float(np.max(abs(np.array(qrefs)-p.log_q(xs,ms))))
    assert max(errs)<1e-5 and prior_error<1e-5 and q_error<1e-5
    assert abs(sum(c.weight for c in p.components)-1)<1e-12
    assert np.all(p.log_q(xs,ms)>=np.log(.1)+lprior-1e-12)
    outside=xs.copy();outside[:,22]=.05
    assert np.all(np.isneginf(p.log_prior(outside,ms)))
    assert np.all(np.isfinite(p.log_q(outside,ms)))
    worlds=[world]+ip.env.sample_worlds(721114,5)
    xv=np.array([ip.vector_from_world(w) for w in worlds]);mv=np.array([w.modes for w in worlds])
    expected_ll=np.array([sum(ip.env.log_likelihood(f,w,at) for f,at in history) for w in worlds])
    got_ll=p.log_likelihood_full(worlds)
    assert np.array_equal(np.isfinite(expected_ll),np.isfinite(got_ll))
    finite=np.isfinite(expected_ll)
    assert np.max(abs(expected_ll[finite]-got_ll[finite]))<1e-10
    with patch.object(p,'sample',return_value=(xv,mv,worlds)):
        imp=p.importance_sample(len(worlds))
    expected_w=p.log_prior(xv,mv)+expected_ll-p.log_q(xv,mv)
    ff=np.isfinite(expected_w)
    assert np.allclose(imp.weights[ff],np.exp(expected_w[ff]-logsumexp(expected_w[ff])))
    assert np.array_equal(imp.weights[~ff],np.zeros((~ff).sum()))
    with patch.object(p,'sample',return_value=(xv,mv,worlds)),patch.object(p,'log_likelihood_full',return_value=np.full(len(worlds),-np.inf)):
        zero=p.importance_sample(len(worlds))
    assert not zero.diagnostics['posterior_valid'] and not zero.weights.any()
    # This is an API/evidence-ledger fixture, not a strategy experiment.
    try:
        dup=ip.InformedProposal(history+history)
        duplicate_ll=dup.log_likelihood_full([world])[0]
        single_ll=p.log_likelihood_full([world])[0]
        duplicate=dict(rejected=False,history_len=len(dup.history),single_ll=float(single_ll),duplicate_ll=float(duplicate_ll),
                       counted_once=bool(np.isclose(single_ll,duplicate_ll,rtol=0,atol=1e-12)))
    except ValueError as e:
        duplicate=dict(rejected=True,error=str(e),counted_once=True)
    invalid=np.zeros_like(ms)
    try:
        invalid_q=p.log_q(xs,invalid)
        invalid_mode=dict(rejected=False,all_negative_infinity=bool(np.isneginf(invalid_q).all()))
    except ValueError as e:invalid_mode=dict(rejected=True,error=str(e),all_negative_infinity=True)
    return dict(component_scipy_max_error=float(max(errs)),component_count=len(p.components),component_points_checked=6,
                normalized_prior_scipy_max_error=prior_error,mixture_scipy_max_error=q_error,
                prior_mixture_support_preserved=True,outside_prior_depth_zero_without_proposal_clipping=True,
                fullhistory_and_importance_passed=True,allzero_returns_invalid=True,
                duplicate_frame_check=duplicate,invalid_mode_check=invalid_mode)

def response_checks():
    import torch
    path=ROOT/'response_learning_geometry/geometry_addendum/run_corrected.py'
    spec=importlib.util.spec_from_file_location('response_independent_audit',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    data=module.base.make_samples(3,721115,'ood',None,False)
    out={}
    for name in ['probe_local_dyadic_mlp','e3nn_typed_basis']:
        model=module.MODELS[name]().cpu().double()
        weight=path.parent/'weights'/f'{name}_positive_n2048_seed101.pt'
        saved=torch.load(weight,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict']);model.eval()
        errors=[]
        for i in range(3):
            tensors={k:torch.tensor(data[k][i:i+1],dtype=torch.float64) for k in ['s','R','n_world','t_world','u_local']}
            def fn(u):
                R=tensors['R'];uw=torch.einsum('nij,nj->ni',R,u)
                qw=model(tensors['s'],uw,tensors['n_world'],tensors['t_world'],R)
                return torch.einsum('nij,nj->ni',R.transpose(-1,-2),qw)[0]
            u=tensors['u_local'];j=torch.autograd.functional.jacobian(fn,u)[:,0,:].detach().numpy()
            h=1e-5;fd=[]
            for k in range(3):
                du=torch.zeros_like(u);du[0,k]=h
                fd.append(((fn(u+du)-fn(u-du))/(2*h)).detach().numpy())
            errors.append(float(np.linalg.norm(j-np.array(fd).T)))
        assert max(errors)<1e-7
        out[name]=dict(trained_weight_sha256=sha(weight),double_fd_fro_errors=errors)
    result=json.loads((path.parent/'RESULTS.json').read_text())
    # Recompute chosen-action success/regret from unchanged generated task data.
    task=module.base.make_samples(512,module.TASK_OOD_SEED,'ood',module.base.CANDIDATES,False)
    target=np.array([.4,-.3]);cost=np.linalg.norm(task['total_local'][...,[0,2]]-target,axis=-1)
    task_checks={}
    for name,row in result['task_scale_rows'].items():
        if name=='oracle':continue
        chosen=np.array(row['chosen']);actual=cost[np.arange(512),chosen];regret=actual-cost.min(axis=1)
        success=float((actual<=.2).mean());mean=float(regret.mean())
        assert success==row['success_rate'] and abs(mean-row['mean_regret_normalized'])<2e-7
        task_checks[name]=dict(correct_count=int((actual<=.2).sum()),mean_regret_normalized=mean,
                               mean_regret_mm=10*mean)
    negative=module.base.make_samples(8,module.NEG_TEST_SEED,'ood',None,True)
    assert not np.any(negative['q_world'])
    out['task_recount']=task_checks
    out['negative_q_exactly_zero']=True
    out['vendor_e3nn_version']=__import__('e3nn').__version__
    out['real_tp_classes']=[type(model.dot).__module__,type(model.scalar_vector).__module__]
    return out

if __name__=='__main__':
    before=sha(ip.__file__)
    results={'proposal_source_sha_before':before,'proposal':proposal_checks(),'response':response_checks(),
             'scope':'Independent bounded fixtures and trained-weight checks; no training/native/locked720913.'}
    results['proposal_source_sha_after']=sha(ip.__file__)
    results['source_unchanged']=before==results['proposal_source_sha_after']
    filename=ROOT/'review'/('FINAL_MODULE_AUDIT_checks_visible_'+before[:12]+'.json')
    assert not filename.exists()
    filename.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print(json.dumps(results,indent=2))

"""Deferred independent M8/compression subset, no native/optimization rollout.

Run only AFTER the timed formal policy experiment ends, so these checks cannot
contaminate its charged host-compute time. All outputs remain in audit/.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--timed-policy-run-finished',action='store_true')
args=parser.parse_args()
if not args.timed_policy_run_finished:
    raise SystemExit('Deferred: require root confirmation that formal policy timing has ended.')

import numpy as np
import torch
from scipy.special import logsumexp
from scipy.spatial.transform import Rotation

torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OPT=ROOT/'optimization'
COMP=ROOT/'theory/operator_experiment/compression'
sys.path.insert(0,str(OPT))
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'capability'))
from capability import CapabilityEvaluator
from observation import environment as env
from run_optimization import load_cases
from torch_geometry import geometry,objective,SCALE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    return v

manifest=json.loads((OPT/'MANIFEST.json').read_text())
for name,h in manifest['sha256'].items():
    assert sha(OPT/name)==h,('optimization hash',name)
source=ROOT/'theory/operator_experiment/operator_arrays.npz'
assert sha(source)=='69dee7a49016a0b66ff020a4bd6b0f858aecfe7c48ced65867102525661859d4'
assert sha(COMP/'compression_arrays.npz')=='600632172ff7f569dd992a2c0686928fba2d664d8817c4c76e933bde7485d503'
report=dict(scope='Independent bounded subset after timed run; no native, new optimization or heldout world',
    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [OPT/'MANIFEST.json',OPT/'RESULTS.json',
        OPT/'FROZEN_CASES.json',OPT/'torch_geometry.py',COMP/'RESULTS.json',
        COMP/'compression_arrays.npz',source]})

# Recompile from L and R, not from the saved full M or factorized result.
with np.load(source,allow_pickle=False) as z:
    L=z['likelihood'].copy();R=z['risk'].copy()
    aliases=np.concatenate([z['b_plus'],z['b_minus']])
with np.load(COMP/'compression_arrays.npz',allow_pickle=False) as z:
    saved={k:z[k].copy() for k in ['source_matrix','test_beliefs','exact_xi','exact_future',
        'compressed_xi','requested_ranks','exact_global_kind','exact_global_action']}
M=(L.transpose(1,2,0)[:,:,None,:]*R.T[None,None,:,:]).reshape(4275,64)
np.testing.assert_array_equal(M,saved['source_matrix'])
B=np.concatenate([aliases,np.random.default_rng(720512).dirichlet(np.ones(64),size=32)])
np.testing.assert_array_equal(B,saved['test_beliefs'])
X=(B@M.T).reshape(96,15,19,15)
F=X.min(-1).sum(-1)
D=B@R
np.testing.assert_allclose(X.reshape(96,-1),saved['exact_xi'],rtol=0,atol=1e-12)
np.testing.assert_allclose(F,saved['exact_future'],rtol=0,atol=1e-12)
direct=D.min(-1);probe=F+.5
kind=direct<=probe.min(-1)
action=np.where(kind,D.argmin(-1),probe.argmin(-1))
assert int(kind.sum())==95
assert bool(kind[:64].all())
U,S,Vt=np.linalg.svd(M,full_matrices=False)
numerical_rank=int(np.sum(S>max(M.shape)*np.finfo(float).eps*S[0]))
assert numerical_rank==64
gap=np.sort(F,axis=1)[:,1]-F.min(1)
rank_rows=[]
for rank in [32,64]:
    A=U[:,:rank]*S[:rank]
    C=Vt[:rank]
    Mr=A@C
    Xr=((B@C.T)@A.T).reshape(96,15,19,15)
    Fr=Xr.min(-1).sum(-1)
    pr=Fr+.5
    kr=direct<=pr.min(-1)
    ar=np.where(kr,D.argmin(-1),pr.argmin(-1))
    changed=(kr!=kind)|(ar!=action)
    error=abs(Mr-M).reshape(15,19,15,64)
    bound=float(error.max(-1).max(-1).sum(-1).max())
    future_error=float(np.max(abs(Fr-F)))
    assert future_error<=bound+1e-10
    rindex=list(saved['requested_ranks']).index(rank)
    np.testing.assert_allclose(Xr.reshape(96,-1),saved['compressed_xi'][rindex],rtol=0,atol=1e-10)
    applied_true=np.where(kr,direct,probe[np.arange(96),ar])
    regret=applied_true-np.minimum(direct,probe.min(-1))
    row=dict(rank=rank,global_policy_changes=int(changed.sum()),
        global_kind_changes=int(np.sum(kr!=kind)),future_value_sup_error=future_error,
        simplex_entry_bound=float(error.max()),simplex_backup_bound=bound,
        negative_xi_count=int(np.sum(Xr<0)),minimum_xi=float(Xr.min()),
        negative_M_count=int(np.sum(Mr<0)),minimum_M=float(Mr.min()),
        conditional_probe_argmin_changes=int(np.sum(Fr.argmin(-1)!=F.argmin(-1))),
        conditional_non_tie_changes=int(np.sum((Fr.argmin(-1)!=F.argmin(-1))&(gap>1e-10))),
        selected_policy_true_regret_max=float(regret.max()),
        factor_bytes=int(A.nbytes+C.nbytes),dense_M_bytes=int(M.nbytes))
    if rank==32:assert row['global_policy_changes']==34
    if rank==64:assert row['global_policy_changes']==0 and future_error<1e-10
    rank_rows.append(row)
report['compression']=dict(matrix_shape=list(M.shape),fresh_numerical_rank=numerical_rank,
    belief_count=96,exact_direct=95,exact_probe=1,all_aliases_direct=True,
    exact_conditional_tie_count=int(np.sum(gap<=1e-10)),ranks=rank_rows)
print('compression subset passed',flush=True)

# M8: authoritative NumPy image geometry, independently composed scipy poses.
data=load_cases()
results=json.loads((OPT/'RESULTS.json').read_text())
cap=CapabilityEvaluator(device='cpu')
def transform(p):
    t=np.eye(4);t[:3,:3]=Rotation.from_rotvec([0,0,float(p[2])]).as_matrix();t[:2,3]=p[:2]
    return t

def numpy_energy(particles,w,pose):
    p=env.predict_batch(particles,[pose]);mu=p['means'][:,0]
    cost=(mu[...,0]/.004)**2+2*np.maximum((.018-mu[...,2])/.018,0)**2
    cost+=.02*((mu[...,1]-.035)/.010)**2+(1-mu[...,4])**2
    return float(np.asarray(w)@((1-p['p_visible'][:,0])*3+np.sum(p['p_source'][:,0]*cost,-1)))

def numpy_total(case,delta,joint):
    initial=np.asarray(case['initial']);target=initial+delta*SCALE
    image=numpy_energy(case['particles'],case['weights'],target)
    if not joint:return image
    mapping=np.asarray(case['body_to_world'])
    path=np.stack([mapping@transform(initial+t*(target-initial)) for t in data['dmp_shape']])
    margin=np.asarray(cap.score(path,case['q_seed'][0])['raw_calibrated_margin'])
    softmin=-.05*logsumexp(-margin/.05)
    return float(image+.25*np.logaddexp(0.,-softmin/.10))

image_errors=[];margin_errors=[];endpoint_rows=[];counts={m:0 for m in ['initial','image_only','joint_proxy']}
for c,r in zip(data['cases'],results['cases']):
    assert c['id']==r['id']
    initial=np.asarray(c['initial']);mapping=np.asarray(c['body_to_world'])
    for method,rec in r['methods'].items():
        target=np.asarray(rec['target'])
        image_errors.append(abs(numpy_energy(c['particles'],c['weights'],target)-rec['expected_image_energy']))
        path=np.stack([mapping@transform(initial+t*(target-initial)) for t in data['dmp_shape']])
        margin=np.asarray(cap.score(path,c['q_seed'][0])['raw_calibrated_margin'])
        margin_errors.append(abs(float(margin.min())-rec['min_path_raw_calibrated_margin']))
        checked=cap.check_q(rec['exact_details']['q_solution'],mapping@transform(target))
        assert checked['valid']
        counts[method]+=int(checked['valid'])
        endpoint_rows.append(dict(case=c['id'],method=method,check=checked))
assert max(image_errors)<1e-10 and max(margin_errors)<1e-10

gradient_rows=[]
for i in [0,5]:
    c=data['cases'][i]
    for joint in [False,True]:
        delta=torch.zeros(3,dtype=torch.float64,requires_grad=True)
        loss,_=objective(c['particles'],c['weights'],torch.tensor(c['initial'],dtype=torch.float64),delta,
            torch.tensor(c['body_to_world'],dtype=torch.float64),data['dmp_shape'],cap,c['q_seed'][0],joint)
        grad=torch.autograd.grad(loss,delta)[0].detach().numpy()
        fd=[]
        for step in [1e-4,1e-5]:
            val=[]
            for k in range(3):
                d=np.zeros(3);d[k]=step
                val.append((numpy_total(c,d,joint)-numpy_total(c,-d,joint))/(2*step))
            fd.append(val)
        err=np.max(abs(np.asarray(fd)-grad),axis=1)
        assert err[1]<1e-5
        if i==5:assert err[0]>1.
        gradient_rows.append(dict(case=i,joint=joint,AD=grad,independent_numpy_FD=fd,max_abs_error=err))

fixtures={'equal_disconnected_components':np.array([[-.05,0,-.03],[-.02,0,-.03],[-.019,.01,-.03],
    [.019,.01,-.03],[.02,0,-.03],[.05,0,-.03]]),
    'exact_slab_boundary':np.array([[-.05,.002,-.03],[.05,.002,-.03]])}
negatives=[]
for name,curve in fixtures.items():
    points=curve[None,None]
    p=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    actual=geometry(torch.tensor(points),p)
    for feature in ['u','depth','length']:
        grad=torch.autograd.grad(actual[feature].sum(),p,retain_graph=True)[0].detach().numpy()[0]*SCALE
        fd=[]
        for k in range(3):
            d=np.zeros((1,3));d[0,k]=1e-5*SCALE[k]
            fd.append((env._geometry_batch(points,d)[feature].item()-env._geometry_batch(points,-d)[feature].item())/(2e-5))
        negatives.append(dict(fixture=name,feature=feature,AD=grad,independent_numpy_FD=fd,
            max_abs_error=float(np.max(abs(grad-fd)))))
report['optimization']=dict(all_manifest_hashes_match=True,image_value_max_abs_error=max(image_errors),
    path_proxy_max_abs_error=max(margin_errors),endpoint_counts=counts,
    endpoint_rows=endpoint_rows,gradient_subset=gradient_rows,boundary_negative_controls=negatives,
    native_calls=0,new_optimizer_runs=0,actual_incremental_endpoint_gain=0,
    robot_manifest=cap.manifest())
report['passed']=True
(HERE/'module_subset_checks.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n')
print(json.dumps(clean(dict(passed=True,compression=report['compression'],
    m8_image_error=max(image_errors),m8_path_margin_error=max(margin_errors),endpoint_counts=counts,
    gradient_subset=gradient_rows,boundary_negative_controls=negatives)),indent=2),flush=True)

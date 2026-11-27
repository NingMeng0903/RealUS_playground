"""Frozen 16-development-case optional M8 diagnostic. No production writes."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'capability'));sys.path.insert(0,'/media/camp/EXT_DRIVE/Among_US')
from capability import CapabilityEvaluator
from observation.environment import World,sample_prior,predict_batch,_geometry_batch
from DMP_playground.dmp_honest.dmp import fit_discrete_dmp
from torch_geometry import geometry,image_energy,body_transform,objective,SCALE

torch.set_num_threads(1)

def clean(x):
    if isinstance(x,torch.Tensor):return clean(x.detach().cpu().numpy())
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def dump(name,value):
    (HERE/name).write_text(json.dumps(clean(value),indent=2)+'\n')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def Tnp(pose):
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec([0,0,pose[2]]).as_matrix();T[:2,3]=pose[:2];return T

def dmp_shape():
    s=np.linspace(0,1,41);d=fit_discrete_dmp(s,s[:,None]);raw,_=d.rollout(s)
    corrected=raw[:,0]/raw[-1,0]
    # Independent actual source fit of the compensated demo, not synthetic line.
    d2=fit_discrete_dmp(s,(s/raw[-1,0])[:,None]);actual,_=d2.rollout(s)
    return corrected,dict(raw_endpoint=float(raw[-1,0]),goal_compensation=float(1/raw[-1,0]),
        compensated_source_error=float(np.max(abs(actual[:,0]-corrected))),raw_shape=raw[:,0])

def freeze():
    candidates=json.loads((ROOT/'capability/frozen_public/candidates.json').read_text())
    endpoints=json.loads((ROOT/'capability/endpoint_checks.json').read_text());byid={p['id']:p for p in candidates}
    nominal=next(p for p in endpoints if p['id']=='a0_identity')
    near=min([p for p in endpoints if p['found']],key=lambda p:p['raw_score'])
    rng=np.random.default_rng(720520);cases=[]
    shape,shape_meta=dmp_shape()
    for i in range(16):
        mode=rng.choice([-1.,1.],2);seed=int(rng.integers(1,2**31-1))
        pool=sample_prior(seed,256);particles=[p for p in pool if np.array_equal(p.modes[0],mode)][:32]
        if len(particles)!=32:raise ValueError('predeclared legal rejection sample insufficient')
        initial=np.array([rng.uniform(-.002,.002),mode[0]*.014+rng.uniform(-.003,.003),np.arctan(mode[1]*.25)+rng.uniform(-.08,.08)])
        anchor=nominal if i%2==0 else near;T=np.asarray(byid[anchor['id']]['T_world_tcp'])
        mapping=T@np.linalg.inv(Tnp(initial))
        cases.append(dict(id=i,public_mode=mode,particle_seed=seed,
            particles=[{name:getattr(p,name) for name in p.__dataclass_fields__} for p in particles],
            weights=np.ones(32)/32,initial=initial,body_to_world=mapping,
            anchor_id=anchor['id'],q_seed=anchor['q_solution']))
    manifest=dict(seed=720520,cases=cases,dmp_shape=shape,dmp_metadata=shape_meta,
        anchor_selection={'nominal':nominal['id'],'near_bound':near['id']},
        source_sha256={str(p):sha(p) for p in [HERE/'PROTOCOL.json',HERE/'torch_geometry.py',HERE/'run_optimization.py',ROOT/'observation/environment.py',ROOT/'capability/capability.py',Path('/media/camp/EXT_DRIVE/Among_US/DMP_playground/dmp_honest/dmp.py')]})
    dump('FROZEN_CASES.json',manifest)
    return clean(manifest)

def load_cases():
    data=json.loads((HERE/'FROZEN_CASES.json').read_text())
    for p,h in data['source_sha256'].items():
        if sha(p)!=h:raise ValueError(f'frozen source changed: {p}')
    for c in data['cases']:c['particles']=[World(**{k:np.asarray(v) for k,v in p.items()}) for p in c['particles']]
    return data

def numpy_image(particles,weights,pose):
    m=predict_batch(particles,[pose]);mu=m['means'][:,0]
    cost=(mu[...,0]/.004)**2+2*np.maximum((.018-mu[...,2])/.018,0)**2+.02*((mu[...,1]-.035)/.010)**2+(1-mu[...,4])**2
    return float(np.asarray(weights)@((1-m['p_visible'][:,0])*3+(m['p_source'][:,0]*cost).sum(-1)))

def numpy_total(particles,weights,initial,delta,mapping,shape,cap,qseed,joint):
    target=initial+delta*SCALE;image=numpy_image(particles,weights,target)
    path=np.asarray([mapping@Tnp(initial+t*(target-initial)) for t in shape])
    score=np.asarray(cap.score(path,qseed[0])['raw_calibrated_margin'])
    softmin=-.05*logsumexp(-score/.05)
    return image+(.25*np.logaddexp(0,-softmin/.10) if joint else 0.)

def branch_signature(particles,pose):
    g=_geometry_batch(np.asarray([p.points for p in particles]),[pose])
    # Independent exact NumPy renderer does not export topology. Slab/FOV/depth
    # point support plus chosen geometric length captures discrete changes.
    return g

def contracts(cap,data):
    rng=np.random.default_rng(720521);particles=sample_prior(720521,32)
    poses=np.c_[rng.uniform(-.02,.02,8),rng.uniform(-.025,.025,8),rng.uniform(-.5,.5,8)]
    n=_geometry_batch(np.array([p.points for p in particles]),poses)
    t=geometry(torch.tensor(np.array([p.points for p in particles])),torch.tensor(poses))
    errs={name:float(np.max(abs(n[name]-t[name].detach().numpy()))) for name in ['u','depth','length','geometric_length']}
    rows=[]
    # Fresh contract fixture; finite differences evaluate independent NumPy
    # geometry and scipy Rz/SE3, not the Torch computational graph.
    for i in range(8):
        c=data['cases'][i];particles=c['particles'];initial=np.array(c['initial']);mapping=np.array(c['body_to_world']);q=np.array(c['q_seed'])
        for joint in [False,True]:
            delta=torch.zeros(3,dtype=torch.float64,requires_grad=True)
            loss,diag=objective(particles,c['weights'],torch.tensor(initial),delta,torch.tensor(mapping),data['dmp_shape'],cap,q[0],joint)
            grad=torch.autograd.grad(loss,delta)[0].detach().numpy();fds=[];switches=[]
            base=diag['response']
            for h in [1e-4,1e-5]:
                fd=[];switched=False
                for j in range(3):
                    d=np.zeros(3);d[j]=h
                    a=numpy_total(particles,c['weights'],initial,d,mapping,data['dmp_shape'],cap,q,joint)
                    b=numpy_total(particles,c['weights'],initial,-d,mapping,data['dmp_shape'],cap,q,joint)
                    fd.append((a-b)/(2*h))
                    gp=geometry(delta.new_tensor(np.array([p.points for p in particles])),delta.new_tensor((initial+d*SCALE)[None]))
                    gm=geometry(delta.new_tensor(np.array([p.points for p in particles])),delta.new_tensor((initial-d*SCALE)[None]))
                    switched|=bool(torch.any(gp['keep']!=gm['keep']) or torch.any(gp['valid']!=gm['valid']))
                fds.append(fd);switches.append(switched)
            err=np.max(abs(grad-np.array(fds)),axis=1)
            rows.append(dict(case=i,joint=joint,autodiff=grad,finite_difference=fds,max_abs_error=err,
                topology_switch=switches,component_ties=int(base['component_tie'].sum()),
                gradient_pass=bool(np.all(err<1e-3+1e-3*np.max(abs(grad))))))
    c=data['cases'][0];p=np.array(c['initial']);mapping=np.array(c['body_to_world']);q=np.array(c['q_seed'])
    T=mapping@Tnp(p);G=np.eye(4);G[:3,:3]=Rotation.from_rotvec([.3,-.2,.1]).as_matrix();G[:3,3]=[.2,-.1,.3]
    s0=cap.score(T,q[0])['raw_score'];s1=cap.score(G@T,q[0],G)['raw_score']
    return dict(numpy_torch_max_abs_errors=errs,gradient_rows=rows,
        gradient_failure_count=sum(not r['gradient_pass'] for r in rows),
        component_tie_count=sum(r['component_ties'] for r in rows),
        fd_topology_switch_rows=sum(any(r['topology_switch']) for r in rows),
        common_SE3_score_error=float(np.max(abs(np.asarray(s0)-np.asarray(s1)))),
        all_samples_retained=True)

def evaluate(cap,data):
    out=[];shape=data['dmp_shape'];started=time.perf_counter()
    for c in data['cases']:
        case_started=time.perf_counter();particles=c['particles'];initial=np.array(c['initial']);q=np.array(c['q_seed']);mapping=np.array(c['body_to_world'])
        row=dict(id=c['id'],anchor_id=c['anchor_id'],public_mode=c['public_mode'],methods={})
        for method in ['initial','image_only','joint_proxy']:
            d=torch.zeros(3,dtype=torch.float64,requires_grad=True);trace=[];opt_started=time.perf_counter()
            if method!='initial':
                for step in range(20):
                    loss,info=objective(particles,c['weights'],torch.tensor(initial),d,torch.tensor(mapping),shape,cap,q[0],method=='joint_proxy')
                    grad=torch.autograd.grad(loss,d)[0]
                    trace.append(dict(step=step,loss=loss,image=info['image'],min_margin=info['min_margin'],
                        gradient=grad,component_ties=int(info['response']['component_tie'].sum())))
                    with torch.no_grad():d-=.15*grad/torch.linalg.vector_norm(grad).clamp(min=1e-12);d.clamp_(-2,2)
            _,info=objective(particles,c['weights'],torch.tensor(initial),d,torch.tensor(mapping),shape,cap,q[0],method=='joint_proxy')
            opt_seconds=time.perf_counter()-opt_started;target=info['target'].detach().numpy();T=mapping@Tnp(target)
            actual=cap.evaluate_pose(T,q,exact=True,native=False);static=actual['static'];found=static['endpoint_found']
            check=static['waypoint_checks'][-1].get('check') if found else None
            row['methods'][method]=dict(target=target,scaled_delta=d,expected_image_energy=info['image'],
                min_path_raw_calibrated_margin=info['min_margin'],softmin_path_raw_calibrated_margin=info['softmin_margin'],
                cap_penalty=info['cap_penalty'],optimizer_seconds=opt_seconds,
                endpoint_found=found,endpoint_check=check,actual_query_seconds=actual['seconds'],
                sampled_joint_connection_pass=static['sampled_joint_connection_pass'],
                exact_details=static,trace=trace)
        row['seconds']=time.perf_counter()-case_started;out.append(clean(row))
        print(f"case {c['id']:02d} {c['anchor_id']} image {[round(row['methods'][m]['expected_image_energy'].item(),4) for m in row['methods']]} found {[row['methods'][m]['endpoint_found'] for m in row['methods']]}",flush=True)
    summary={}
    for method in ['initial','image_only','joint_proxy']:
        rows=[r['methods'][method] for r in out]
        summary[method]=dict(endpoint_found=sum(r['endpoint_found'] for r in rows),
            mean_image_energy=float(np.mean([r['expected_image_energy'] for r in rows])),
            mean_min_path_margin=float(np.mean([r['min_path_raw_calibrated_margin'] for r in rows])),
            mean_optimization_seconds=float(np.mean([r['optimizer_seconds'] for r in rows])),
            mean_actual_query_seconds=float(np.mean([r['actual_query_seconds'] for r in rows])),
            self_collision_pass_among_found=sum(bool(r['endpoint_check'] and r['endpoint_check']['self_collision_pass']) for r in rows))
    summary['paired_joint_vs_image']=dict(endpoint_found_gain=sum(r['methods']['joint_proxy']['endpoint_found']-r['methods']['image_only']['endpoint_found'] for r in out),
        image_energy_better=sum(r['methods']['joint_proxy']['expected_image_energy']<r['methods']['image_only']['expected_image_energy'] for r in out),
        minimum_proxy_margin_better=sum(r['methods']['joint_proxy']['min_path_raw_calibrated_margin']>r['methods']['image_only']['min_path_raw_calibrated_margin'] for r in out))
    return dict(summary=summary,cases=out,seconds=time.perf_counter()-started,
                no_native_or_task_success_claim=True,all_failures_retained=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--evaluate',action='store_true');args=parser.parse_args()
    if args.freeze:freeze();print('frozen cases/protocol/source hashes before evaluation')
    if args.evaluate:
        data=load_cases();cap=CapabilityEvaluator(device='cpu')
        check=contracts(cap,data);dump('GRADIENT_CHECKS.json',check)
        result=evaluate(cap,data);result['frozen_manifest_sha256']=sha(HERE/'FROZEN_CASES.json');result['gradient_summary']={k:v for k,v in check.items() if k!='gradient_rows'}
        dump('RESULTS.json',result);print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()

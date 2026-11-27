"""Independent-world synthetic A response proof; actual DMP action endpoints."""
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
ROOT=Path(__file__).resolve().parent.parent

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def features(c,u):
    c,u=np.asarray(c),np.asarray(u)
    return np.concatenate([u]+[c[...,j,None]*u for j in range(3)],axis=-1)
def worlds(seed,n):
    r=np.random.default_rng(seed)
    return dict(c=r.uniform(-1,1,(n,3)),e=r.uniform(-.9,.9,(n,2)),E=r.normal(0,.04,(n,2,2)),u=r.uniform(-1,1,(n,12,2)),noise=r.normal(0,.015,(n,12,2)),probe_noise=r.normal(0,.015,(n,4,2)))
def matrices(w,positive):
    c=w['c']; J=np.repeat(np.eye(2)[None],len(c),axis=0)
    if positive:
        J+=.45*np.stack([np.stack([c[:,0],.5*c[:,1]],axis=-1),np.stack([-.35*c[:,1],c[:,2]],axis=-1)],axis=1)+w['E']
    return J

def bootstrap(x,seed=41104):
    r=np.random.default_rng(seed); x=np.asarray(x); b=x[r.integers(0,len(x),(2000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())
def main():
    motion=module('core_motion',ROOT/'motion/adapter.py')
    rows=[]
    for x in [-1,-.5,0,.5,1]:
      for y in [-1,-.5,0,.5,1]:
        rows+=motion.generate_candidates([0,0,0],[.04*x,.04*y,0],n=21,normal_xyz=[0,0,1],rotation=np.eye(3),lateral_offsets_m=(0.,),methods=('dmp',))
    actions=np.array([r['xyz_m'][-1,:2]/.04 for r in rows]); lengths=np.array([r['metrics']['path_length_m']/.04 for r in rows]); move=.002*lengths
    splits={k:worlds(s,n) for k,s,n in [('train',41101,256),('validation',41102,64),('test',41103,128)]}
    result={'evidence':'SYNTHETIC MECHANISM with real DMP candidate endpoints; no servo/physical ultrasound', 'protocol_sha256':hashlib.sha256((ROOT/'CORE_PROTOCOL.json').read_bytes()).hexdigest(),'seed':{'train':41101,'validation':41102,'test':41103,'bootstrap':41104},'split_hashes':{k:{key:sha(v) for key,v in w.items()} for k,w in splits.items()},'action_grid':actions.tolist(),'action_path_length_m':(lengths*.04).tolist(),'conditions':{}}
    arrays={}
    for positive,name in [(False,'analytic_exact_negative_control'),(True,'context_dependent_response_shift')]:
      train=splits['train']; J=matrices(train,positive); u=train['u']; c=np.repeat(train['c'][:,None],12,axis=1)
      F=features(c,u).reshape(-1,8); residual=np.einsum('nij,nkj->nki',J-np.eye(2),u)+train['noise']; Y=residual.reshape(-1,2)
      t=time.perf_counter(); W=np.linalg.solve(F.T@F+1e-3*np.eye(8),F.T@Y); fit_s=time.perf_counter()-t
      condition={'fit_seconds':fit_s,'ridge':1e-3,'coefficient':W.tolist(),'splits':{}}
      for split in ['validation','test']:
        w=splits[split]; J=matrices(w,positive); n=len(w['c']); true=w['e'][:,None,:]+np.einsum('nij,aj->nai',J,actions)
        learned=w['e'][:,None,:]+actions[None]+features(w['c'][:,None,:],np.broadcast_to(actions,(n,len(actions),2)))@W
        analytic=w['e'][:,None,:]+actions[None]
        # Four measured responses to actual local central-difference probe actions.
        probe=np.array([[.25,0],[-.25,0],[0,.25],[0,-.25]])
        measured=w['e'][:,None,:]+np.einsum('nij,pj->npi',J,probe)+w['probe_noise']
        Jhat=np.stack([(measured[:,0]-measured[:,1])/.5,(measured[:,2]-measured[:,3])/.5],axis=2)
        online=w['e'][:,None,:]+np.einsum('nij,aj->nai',Jhat,actions)
        preds={'analytic_geometry':analytic,'canonical_learned_residual':learned,'online_local_jacobian_calibration':online,'oracle_response_reference':true}
        out={}; selected={}
        for method,pred in preds.items():
          ix=np.argmin(np.sum(pred**2,axis=2)+move[None],axis=1); err=np.sum(true[np.arange(n),ix]**2,axis=1)
          probes=4 if method.startswith('online') else 0
          probe_dist=2. if probes else 0. # four .25-norm probes, each out+return
          charged=err+move[ix]+.01*probes+.002*probe_dist
          out[method]={'true_selected_error_mean':float(err.mean()),'prediction_error_mse_all_actions':float(np.mean((pred-true)**2)),'selected_prediction_mse':float(np.mean((pred[np.arange(n),ix]-true[np.arange(n),ix])**2)), 'selected_action_path_length_m_mean':float(np.mean(lengths[ix]*.04)),'extra_probe_count':probes,'probe_out_and_return_distance_m':.04*probe_dist,'total_response_observations':1+probes,'charged_objective_mean':float(charged.mean())}
          selected[method]=ix; arrays[f'{name}_{split}_{method}_error']=err; arrays[f'{name}_{split}_{method}_charged']=charged; arrays[f'{name}_{split}_{method}_index']=ix
        condition['splits'][split]={'methods':out,'learned_minus_analytic_error':bootstrap(arrays[f'{name}_{split}_canonical_learned_residual_error']-arrays[f'{name}_{split}_analytic_geometry_error']),'online_minus_learned_charged':bootstrap(arrays[f'{name}_{split}_online_local_jacobian_calibration_charged']-arrays[f'{name}_{split}_canonical_learned_residual_charged'])}
      result['conditions'][name]=condition
    # Passive common SE3: local canonical coordinates computed from transformed axes.
    from scipy.spatial.transform import Rotation
    rng=np.random.default_rng(41105); errs=[]
    for _ in range(50):
      R=Rotation.random(random_state=rng).as_matrix(); t=rng.normal(size=3); p=rng.normal(size=3); v=rng.normal(size=3)
      errs.append(np.max(np.abs((R@np.eye(3)).T@((R@(p+v)+t)-(R@p+t))-v)))
    result['checks']={'common_passive_se3_local_coordinates_max_error':float(max(errs)),'not_checked':'body-only movement changes robot capability; e3nn omitted, no necessity claim','probe_models':'actual synthetic response central differences; no oracle Jacobian access','negative_control_note':'Noisy training permits small adverse residual effects; no learned advantage is imposed.'}
    np.savez(ROOT/'core/a_raw.npz',**arrays); (ROOT/'core/a_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps({k:v['splits']['test'] for k,v in result['conditions'].items()},indent=2))
if __name__=='__main__': main()

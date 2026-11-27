"""Post-evaluation branch audit; never edits frozen cases, source or results."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run_optimization import load_cases,dump
from torch_geometry import geometry,SCALE

def signature(points,pose):
    c,s=np.cos(pose[2]),np.sin(pose[2]);d=points-np.array([pose[0],pose[1],0.])
    local=np.stack([c*d[...,0]+s*d[...,1],-s*d[...,0]+c*d[...,1],-d[...,2]],-1)
    a,b=local[...,:-1,:],local[...,1:,:];delta=b-a;parallel=abs(delta)<1e-15
    safe=np.where(parallel,1,delta);t0=(np.array([-.025,-.002,.015])-a)/safe;t1=(np.array([.025,.002,.070])-a)/safe
    lower=np.where(parallel,-np.inf,np.minimum(t0,t1));upper=np.where(parallel,np.inf,np.maximum(t0,t1))
    entry=np.concatenate([np.zeros((*lower.shape[:-1],1)),lower],-1).argmax(-1)
    leave=np.concatenate([np.ones((*upper.shape[:-1],1)),upper],-1).argmin(-1)
    g=geometry(torch.tensor(points),torch.tensor(pose[None]))
    return dict(entry=entry,leave=leave,valid=g['valid'][:,0].numpy(),keep=g['keep'][:,0].numpy(),
                ties=g['component_tie'][:,0].numpy())

def main():
    data=load_cases();results=json.loads((HERE/'RESULTS.json').read_text());rows=[]
    for case,result in zip(data['cases'],results['cases']):
        initial=np.array(case['initial']);points=np.array([p.points for p in case['particles']])
        for method,rec in result['methods'].items():
            delta=np.zeros(3);previous=None
            for j,step in enumerate(rec['trace']+[None]):
                pose=initial+delta*SCALE;sig=signature(points,pose)
                changed=[]
                if previous is not None:
                    ch=np.zeros((32,2),bool)
                    ch|=np.any(sig['keep']!=previous['keep'],axis=-1)
                    relevant=sig['keep']|previous['keep']
                    for field in ['entry','leave']:ch|=np.any((sig[field]!=previous[field])&relevant,axis=-1)
                    changed=np.argwhere(ch).tolist()
                rows.append(dict(case=case['id'],method=method,step=j,
                    scaled_delta=delta.copy(),topology_changed_particle_sources=changed,
                    tied_particle_sources=np.argwhere(sig['ties']).tolist()))
                previous=sig
                if step is not None:
                    grad=np.array(step['gradient']);delta=np.clip(delta-.15*grad/max(np.linalg.norm(grad),1e-12),-2,2)
            if np.max(abs(delta-np.array(rec['scaled_delta'])))>1e-12:raise AssertionError('trace reconstruction mismatch')
    # Explicit non-differentiable negative controls, independent of fixed cases.
    fixtures={
      'equal_disconnected_components':np.array([[-.05,0,-.03],[-.02,0,-.03],[-.019,.01,-.03],[.019,.01,-.03],[.02,0,-.03],[.05,0,-.03]]),
      'exact_slab_boundary':np.array([[-.05,.002,-.03],[.05,.002,-.03]])}
    negatives=[]
    for name,curve in fixtures.items():
        pts=torch.tensor(curve[None,None]);pose=torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
        base=geometry(pts,pose)
        for feature in ['u','depth','length']:
            grad=torch.autograd.grad(base[feature].sum(),pose,retain_graph=True)[0].detach().numpy()[0]*SCALE
            fd=[]
            for j in range(3):
                d=np.zeros((1,3));d[0,j]=1e-5*SCALE[j]
                plus=geometry(pts,torch.tensor(d))[feature].item();minus=geometry(pts,torch.tensor(-d))[feature].item()
                fd.append((plus-minus)/(2e-5))
            negatives.append(dict(fixture=name,feature=feature,component_tie=bool(base['component_tie'].any()),
                autodiff_scaled=grad,finite_difference_scaled=fd,max_abs_error=float(np.max(abs(grad-fd))),
                expected_nondifferentiable=True))
    report=dict(all_optimization_steps_retained=True,rows=rows,
        steps_with_branch_change=sum(bool(r['topology_changed_particle_sources']) for r in rows),
        particle_source_branch_changes=sum(len(r['topology_changed_particle_sources']) for r in rows),
        tied_optimization_samples=sum(len(r['tied_particle_sources']) for r in rows),
        explicit_boundary_and_tie_negative_controls=negatives,
        scope='Post-evaluation audit only; no optimizer retuning or case replacement',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump('BRANCH_DIAGNOSTICS.json',report)
    print({k:v for k,v in report.items() if k not in ['rows','explicit_boundary_and_tie_negative_controls']})
    print(negatives)
if __name__=='__main__':main()

"""Float64 differentiable exact piecewise counterpart of observation geometry.

No smoothed visibility or hidden actual-patient tangent. Selected topology is
piecewise constant; diagnostics expose ties and finite-difference switches.
"""
import numpy as np
import torch
import torch.nn.functional as F

SCALE=np.array([.004,.004,.1])


def geometry(points,poses):
    poses=torch.atleast_2d(poses)
    dx=points[:,None,:,:,0]-poses[None,:,None,None,0]
    dy=points[:,None,:,:,1]-poses[None,:,None,None,1]
    c=torch.cos(poses[:,2])[None,:,None,None];s=torch.sin(poses[:,2])[None,:,None,None]
    local=torch.stack([c*dx+s*dy,-s*dx+c*dy,
        (-points[:,None,:,:,2]).expand_as(dx)],-1)
    a,b=local[...,:-1,:],local[...,1:,:];delta=b-a
    low=poses.new_tensor([-.025,-.002,.015]);high=poses.new_tensor([.025,.002,.070])
    parallel=delta.abs()<1e-15;safe=torch.where(parallel,torch.ones_like(delta),delta)
    t0,t1=(low-a)/safe,(high-a)/safe
    lower=torch.where(parallel,torch.full_like(delta,-torch.inf),torch.minimum(t0,t1))
    upper=torch.where(parallel,torch.full_like(delta,torch.inf),torch.maximum(t0,t1))
    lo=torch.maximum(poses.new_tensor(0.),lower.amax(-1));hi=torch.minimum(poses.new_tensor(1.),upper.amin(-1))
    outside=(parallel&((a<low)|(a>high))).any(-1)
    valid=(hi>lo+1e-12)&~outside
    lo,hi=lo.clamp(0,1),hi.clamp(0,1)
    ca=a+lo[...,None]*delta;cb=a+hi[...,None]*delta
    l3=torch.linalg.vector_norm(cb-ca,dim=-1)*valid
    connected=torch.zeros_like(valid)
    connected[...,1:]=(valid[...,1:]&valid[...,:-1]&(lo[...,1:]<1e-9)&(hi[...,:-1]>1-1e-9))
    breaks=valid&~connected;labels=breaks.cumsum(-1)
    total=l3.cumsum(-1);before=total-l3
    reset=torch.cummax(torch.where(breaks,before,torch.zeros_like(before)),dim=-1).values
    run=torch.where(valid,total-reset,torch.zeros_like(total))
    idx=run.argmax(-1);best_label=labels.gather(-1,idx[...,None])[...,0]
    keep=valid&(labels==best_label[...,None])
    weight=l3*keep;length=weight.sum(-1)
    pa,pb=ca[...,[0,2]],cb[...,[0,2]]
    lp=torch.linalg.vector_norm(pb-pa,dim=-1)*keep
    plen=lp.sum(-1);centroid=(lp[...,None]*(pa+pb)*.5).sum(-2)/plen.clamp(min=1e-15)[...,None]
    # Components end before a break/miss or at last segment. Compare complete
    # component sums only; repeated interior cumulative values are not ties.
    ends=torch.ones_like(valid);ends[...,:-1]=~connected[...,1:]
    complete=torch.where(valid&ends,run,torch.full_like(run,-1.))
    top=torch.topk(complete,k=min(2,complete.shape[-1]),dim=-1).values
    tie=((top[...,0]>0)&((top[...,0]-top[...,-1]).abs()<1e-8)
         if complete.shape[-1]>1 else torch.zeros_like(top[...,0],dtype=torch.bool))
    boundary=((lo>0)&(lo<1)&valid)|((hi>0)&(hi<1)&valid)
    return dict(u=centroid[...,0],depth=centroid[...,1],length=plen,
                geometric_length=length,exists=length>.0005,
                valid=valid,component=best_label,keep=keep,
                component_tie=tie,clipped_segment_count=boundary.sum(-1))


def response(particles,poses):
    points=poses.new_tensor(np.asarray([p.points for p in particles]))
    g=geometry(points,poses)
    detect=torch.sigmoid(poses.new_tensor(2.8+np.asarray([p.detect_z for p in particles])))[:,None,:]
    source_q=detect*g['exists']*(g['length']/.004).clamp(0,1)
    pany=1-(1-source_q).prod(-1)
    strength=source_q*g['length'].clamp(min=.001)
    probs=pany[...,None]*strength/strength.sum(-1,keepdim=True).clamp(min=1e-30)
    ident=torch.sigmoid(poses.new_tensor(np.array([1.75,.75])[None,:]+.35*np.asarray([p.identity_z for p in particles])))[:,None,:]
    return dict(**g,p_source=probs,p_visible=pany,identity=ident)


def image_energy(particles,pose,weights):
    r=response(particles,pose[None])
    source=(r['u']/.004)**2+2*F.relu((.018-r['length'])/.018)**2+.02*((r['depth']-.035)/.010)**2+(1-r['identity'])**2
    per=(1-r['p_visible'])*3+(r['p_source']*source).sum(-1)
    w=pose.new_tensor(np.asarray(weights)).detach()
    return (w[:,None]*per).sum(),r


def body_transform(pose):
    """Proper SE3: body translation and Rz, with no Euler matrix entries mixed."""
    pose=torch.atleast_2d(pose);x,y,yaw=pose.unbind(-1);c,s=torch.cos(yaw),torch.sin(yaw)
    z=torch.zeros_like(x);o=torch.ones_like(x)
    return torch.stack([torch.stack([c,-s,z,x],-1),torch.stack([s,c,z,y],-1),
                        torch.stack([z,z,o,z],-1),torch.stack([z,z,z,o],-1)],-2)


def path_poses(initial,target,dmp_shape,body_to_world):
    shape=target.new_tensor(dmp_shape)
    poses=initial[None]+shape[:,None]*(target-initial)[None]
    return body_to_world[None]@body_transform(poses)


def objective(particles,weights,initial,scaled_delta,body_to_world,dmp_shape,capability,rail,joint):
    target=initial+scaled_delta*scaled_delta.new_tensor(SCALE)
    image,r=image_energy(particles,target,weights)
    worldpath=path_poses(initial,target,dmp_shape,body_to_world)
    raw=capability.score_tensor(worldpath,torch.full((len(worldpath),),float(rail),dtype=target.dtype,device=target.device))
    margins=raw-capability.bridge.calibration['m_safe']
    softmin=-.05*torch.logsumexp(-margins/.05,dim=0)
    penalty=F.softplus(-softmin/.10)
    loss=image+(.25*penalty if joint else 0.)
    return loss,dict(image=image,cap_penalty=penalty,softmin_margin=softmin,
                     min_margin=margins.min(),mean_margin=margins.mean(),target=target,response=r)

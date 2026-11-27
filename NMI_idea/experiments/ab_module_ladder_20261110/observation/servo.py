"""Common continuous A servo/candidate permissions. Inputs are public only."""
from __future__ import annotations
import numpy as np
from .environment import clip_pose, public_quality, predicted_public_utility


def _entries(history):
    """Accept [(public_frame, pose),...] or dicts with public/pose fields."""
    return [(item['public'],np.asarray(item['pose'],float)) if isinstance(item,dict)
            else (item[0],np.asarray(item[1],float)) for item in history]


def _latest_visible(history):
    for frame,pose in reversed(_entries(history)):
        if frame['visible']:
            return frame,pose
    return None


def reconstruct_line(history):
    """Causal visible centroid fit; uncertainty includes unknown slab coordinate.

    It does not turn a 2D image-axis into a 3D tangent. Requires actual public
    translation span, coherent depth and a line-like public centroid cloud.
    """
    entries=_entries(history)[-40:]
    points=[]; depths=[]; seen=set()
    for f,p in entries:
        if (not f['visible'] or f['identity_score']<.70 or f['frame_id'] in seen): continue
        seen.add(f['frame_id']); c,s=np.cos(p[2]),np.sin(p[2])
        points.append([p[0]+c*f['image_u_m'],p[1]+s*f['image_u_m']])
        depths.append(f['image_depth_m'])
    if len(points)<6: return None
    points=np.asarray(points); depths=np.asarray(depths)
    close=abs(depths-np.median(depths))<.004
    points=points[close]
    if len(points)<6: return None
    center=points.mean(0); cov=np.cov(points.T); vals,vecs=np.linalg.eigh(cov)
    axis=vecs[:,-1]; axis*=1 if axis[0]>=0 else -1
    along=(points-center)@axis
    span=np.ptp(along); rms=np.sqrt(max(vals[0],0))
    if span<.012 or vals[-1]/max(vals[0],1e-12)<6 or rms>.0025: return None
    return dict(center=center,yaw=float(np.arctan2(axis[1],axis[0])),
                span_m=float(span),orthogonal_rms_m=float(rms),
                estimated_only=True,slab_ambiguity_m=.002)


def _quality_gradient(history, pose):
    """Measured finite-difference local response, no hidden tangent/lookup table."""
    entries=_entries(history); unique={}
    for f,p in entries[-40:]:
        if not f['visible']: continue
        key=tuple(np.round(p,7)); unique.setdefault(key,[]).append(public_quality(f))
    if len(unique)<3: return None
    rows=[]; ys=[]
    for key,q in unique.items():
        p=np.asarray(key); dp=p-pose
        if np.linalg.norm(dp[:2])>.015 or abs(dp[2])>.25: continue
        n=np.array([-np.sin(pose[2]),np.cos(pose[2])])
        rows.append([1.,np.dot(dp[:2],n)/.004,dp[2]/.10]);ys.append(np.mean(q))
    if len(rows)<3:return None
    rows=np.asarray(rows);ys=np.asarray(ys)
    # Require two independently excited directions before joint yaw/normal update.
    if np.linalg.matrix_rank(rows)<3:return None
    coef=np.linalg.solve(rows.T@rows+np.diag([1e-6,.02,.02]),rows.T@ys)
    return np.clip(coef[1:],-.3,.3)


def template_candidates(particles, weights, pose):
    """Four public lane/slope posterior summaries; precision is continuous."""
    weights=np.asarray(weights); pts=np.asarray([p.points[0] for p in particles])
    x=float(np.clip(pose[0],-.025,.025))
    paths=[]
    for curve in pts:
        y=np.interp(x,curve[:,0],curve[:,1])
        idx=np.clip(np.searchsorted(curve[:,0],x),1,len(curve)-2)
        d=curve[idx+1]-curve[idx-1]
        paths.append([x,y,np.arctan2(d[1],d[0])])
    paths=np.asarray(paths); modes=np.asarray([p.modes[0] for p in particles])
    out=[]
    for lane in [-1,1]:
        for slope in [-1,1]:
            mask=(modes[:,0]==lane)&(modes[:,1]==slope)
            mass=weights[mask].sum()
            if mass>1e-8:
                out.append((mass,np.average(paths[mask],axis=0,weights=weights[mask])))
    return [clip_pose(p) for _,p in sorted(out,key=lambda z:-z[0])]


def candidate_actions(pose, history, particles, weights, *, max_candidates=12):
    """Common continuous target poses for A/B, history/prior only; no robot cost.

    Centers from observed u, fine normal/yaw probes, causal reconstruction,
    response-gradient step and multimodal prior recovery share one permission set.
    """
    pose=np.asarray(pose,float); entries=_entries(history); latest=entries[-1] if entries else None
    out=[pose.copy()]
    if latest is not None and latest[0]['visible']:
        f,at=latest; c,s=np.cos(at[2]),np.sin(at[2])
        centered=at.copy();centered[:2]+=f['image_u_m']*np.array([c,s])
        out.append(centered)
        line=reconstruct_line(history)
        if line is not None:
            aligned=centered.copy();aligned[2]=line['yaw'];out.append(aligned)
        grad=_quality_gradient(history,at)
        normal=np.array([-s,c])
        if grad is not None:
            improved=centered.copy(); improved[:2]+=.004*grad[0]*normal
            improved[2]+=.10*grad[1];out.append(improved)
        # Centering is applied jointly to all local probes.
        for delta in [-.06,.06]:
            p=centered.copy();p[2]+=delta;out.append(p)
        for delta in [-.0015,.0015]:
            p=centered.copy();p[:2]+=delta*normal;out.append(p)
    # Crossing surveys make a broad prior observable before fine yaw alignment:
    # yaw=0 cuts the +/-0.25 slope modes, so transverse offset appears as public u.
    # Purely parallel long-axis probes can miss an entire shifted curve.
    templates=template_candidates(particles,weights,pose)
    if latest is None or not latest[0]['visible']:
        for sign in [-1,1]:
            lane=[p for p in templates if np.sign(p[1])==sign]
            if lane:
                survey=lane[0].copy();survey[2]=0.;out.append(survey)
    # Four public mode summaries, continuously located, available in every state.
    out.extend(templates)
    # Missing images still permit continuous normal search around latest public hit.
    if latest is not None and not latest[0]['visible']:
        visible=_latest_visible(history)
        if visible:
            f,at=visible;c,s=np.cos(at[2]),np.sin(at[2]);normal=np.array([-s,c])
            p=at.copy();p[:2]+=f['image_u_m']*np.array([c,s]);out.append(p)
            for dy in [-.003,.003]:
                q=p.copy();q[:2]+=normal*dy;out.append(q)
    unique=[]
    for p in out:
        p=clip_pose(p)
        if not any(np.linalg.norm((p-q)*[1,1,.03])<1e-6 for q in unique):unique.append(p)
    return np.asarray(unique[:max_candidates])


def continuous_servo(pose, history, particles, weights):
    """Strong shared analytic A baseline: continuous candidates + exact response."""
    candidates=candidate_actions(pose,history,particles,weights)
    response=predicted_public_utility(particles,candidates,weights)
    score=response['block_gate']+.03*response['visible']
    chosen=int(np.argmax(score))
    return candidates[chosen],dict(candidates=candidates,scores=score,response=response)

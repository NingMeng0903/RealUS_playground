"""Continuous geometric L0 observation contract. Truth never enters public control.

World/particle are persistent prior draws. Only render() and audit functions may
inspect the patient World. Predictions inspect independently sampled particles.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import sys
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp, ndtr, expit

sys.path.insert(0, '/media/camp/EXT_DRIVE/Among_US')
from DMP_playground.kmp_honest.imaging import probe_plane_hit
from DMP_playground.kmp_honest.servo import filter_obs_for_control

FRAME_OFFSETS = np.array([0., .125, .25, .375, .5])
X = np.linspace(-.05, .05, 101)
CONTROL_X = np.linspace(-.05, .05, 5)
CORR = .65 ** np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
CHOL = np.linalg.cholesky(CORR)
SDS = np.array([.00035, .0006, .00055, .03, .10])
FIELDS = ('image_u_m', 'image_depth_m', 'visible_length_m', 'image_axis_rad', 'identity_score')
BOUNDS = np.array([[-.04, .04], [-.04, .04], [-.8, .8]])

@dataclass(frozen=True)
class World:
    """Private latent world OR independent model particle; never pass truth to servo."""
    registration: np.ndarray  # two common x/y registration variables
    modes: np.ndarray        # source x [lane sign, slope sign]
    y_controls: np.ndarray   # two sources x five correlated controls
    z_controls: np.ndarray
    depths: np.ndarray
    identity_z: np.ndarray   # persistent per-source appearance nuisance
    detect_z: np.ndarray
    points: np.ndarray       # two continuous 101-knot polylines


def _points(registration, modes, yc, zc, depths):
    yres = CubicSpline(CONTROL_X, yc, axis=-1, bc_type='natural')(X)
    zres = CubicSpline(CONTROL_X, zc, axis=-1, bc_type='natural')(X)
    pts = np.empty((2, len(X), 3))
    pts[..., 0] = X + registration[0]
    pts[..., 1] = modes[:, 0, None] * .014 + modes[:, 1, None] * .25 * X + yres + registration[1]
    pts[..., 2] = -(depths[:, None] + zres)
    return pts


def _world(reg, modes, yc, zc, depths, identity_z, detect_z):
    return World(reg, modes, yc, zc, depths, identity_z, detect_z,
                 _points(reg, modes, yc, zc, depths))


def sample_worlds(seed, n, *, split=None):
    """Use split='test' only following root release; this API itself does no test I/O."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        reg = rng.normal(0, .004, 2)
        modes = rng.choice([-1., 1.], (2, 2))
        yc = rng.normal(size=(2, 5)) @ CHOL.T * .003
        zc = rng.normal(size=(2, 5)) @ CHOL.T * .0015
        depths = rng.uniform(.025, .045, 2)
        iz, dz = rng.normal(size=(2, 2))
        out.append(_world(reg, modes, yc, zc, depths, iz, dz))
    return out


def sample_prior(seed, n, **kwargs):
    """Independent legal model draws, with the same broad support as worlds."""
    return sample_worlds(seed, n, **kwargs)


def clip_pose(pose):
    return np.clip(np.asarray(pose, float), BOUNDS[:, 0], BOUNDS[:, 1])


def _geometry_batch(points, poses):
    """Vectorized exact segment clipping: [particles, actions, sources, segments]."""
    pts, poses = np.asarray(points), np.atleast_2d(poses)
    dx = pts[:, None, :, :, 0] - poses[None, :, None, None, 0]
    dy = pts[:, None, :, :, 1] - poses[None, :, None, None, 1]
    c = np.cos(poses[:, 2])[None, :, None, None]
    s = np.sin(poses[:, 2])[None, :, None, None]
    local = np.stack([c*dx+s*dy, -s*dx+c*dy,
                      np.broadcast_to(-pts[:, None, :, :, 2], dx.shape)], -1)
    a, b = local[..., :-1, :], local[..., 1:, :]
    delta = b-a
    low = np.array([-.025, -.002, .015])
    high = np.array([.025, .002, .070])
    safe = np.where(abs(delta) < 1e-15, 1., delta)
    t0, t1 = (low-a)/safe, (high-a)/safe
    parallel = abs(delta)<1e-15
    outside = np.any(parallel & ((a < low) | (a > high)), axis=-1)
    # A constant coordinate inside its interval imposes no parametric bound.
    lower = np.where(parallel, -np.inf, np.minimum(t0, t1))
    upper = np.where(parallel, np.inf, np.maximum(t0, t1))
    lo = np.maximum(0., lower.max(axis=-1))
    hi = np.minimum(1., upper.min(axis=-1))
    valid = (hi > lo + 1e-12) & ~outside
    lo = np.clip(lo, 0., 1.)
    hi = np.clip(hi, 0., 1.)
    ca = a + lo[..., None] * delta
    cb = a + hi[..., None] * delta
    l3 = np.linalg.norm(cb-ca, axis=-1) * valid
    connected_prev = np.zeros_like(valid)
    connected_prev[..., 1:] = (valid[..., 1:] & valid[..., :-1] &
        (lo[..., 1:] < 1e-9) & (hi[..., :-1] > 1.-1e-9))
    breaks = valid & ~connected_prev
    labels = np.cumsum(breaks, axis=-1)
    total = np.cumsum(l3, axis=-1)
    before = total-l3
    reset = np.maximum.accumulate(np.where(breaks, before, 0.), axis=-1)
    run = total-reset
    run = np.where(valid, run, 0.)
    best_idx = run.argmax(axis=-1)
    best_label = np.take_along_axis(labels, best_idx[..., None], -1)[..., 0]
    keep = valid & (labels == best_label[..., None])
    weight = l3 * keep
    length = weight.sum(axis=-1)
    denom = np.maximum(length, 1e-15)
    mid = (ca+cb)*.5
    centroid = (weight[..., None]*mid).sum(axis=-2)/denom[..., None]
    # Public image geometry uses only projected clipped segments.
    pa, pb = ca[..., [0, 2]], cb[..., [0, 2]]
    lp = np.linalg.norm(pb-pa, axis=-1)*keep
    plen = lp.sum(axis=-1)
    pden = np.maximum(plen, 1e-15)
    pm = (pa+pb)*.5
    pc = (lp[..., None]*pm).sum(axis=-2)/pden[..., None]
    e2 = ((pa[..., :, None]*pa[..., None, :]+pb[..., :, None]*pb[..., None, :])/3 +
          (pa[..., :, None]*pb[..., None, :]+pb[..., :, None]*pa[..., None, :])/6)
    cov = (lp[..., None, None]*e2).sum(axis=-3)/pden[..., None, None]-pc[..., :, None]*pc[..., None, :]
    axis = .5*np.arctan2(2*cov[..., 0, 1], cov[..., 0, 0]-cov[..., 1, 1])
    # Audit-only RMS actual tangent angle to the image plane.
    segnorm = np.maximum(np.linalg.norm(delta, axis=-1), 1e-15)
    sine = np.clip(abs(delta[..., 1])/segnorm, 0, 1)
    angle = np.sqrt((weight*np.arcsin(sine)**2).sum(axis=-1)/denom)
    exists = length > .0005
    return dict(u=pc[..., 0], depth=pc[..., 1], length=plen,
                geometric_length=length, axis=axis, angle_audit=angle, exists=exists)


def predict_batch(particles, poses):
    """Feature mixture, shape [P,A,2,5]; fast reusable response for all planners.

    p_source events are mutually exclusive; 1-sum(p_source) is no detection.
    No cost, robot ability, hidden-world information or posterior reward here.
    """
    if not particles:
        raise ValueError('particles must be nonempty')
    poses = np.atleast_2d(np.asarray(poses, float))
    g = _geometry_batch(np.asarray([p.points for p in particles]), poses)
    detect = expit(2.8 + np.asarray([p.detect_z for p in particles]))[:, None, :]
    # Length threshold is detector resolution, not fabricated off-plane visibility.
    source_q = detect * g['exists'] * np.clip(g['length']/.004, 0, 1)
    # Independent return events followed by strongest projected segment association.
    # Winner competition is length weighted; sum <= 1 exactly.
    pany = 1-np.prod(1-source_q, axis=-1)
    strength = source_q * np.maximum(g['length'], .001)
    probs = pany[..., None]*strength/np.maximum(strength.sum(-1, keepdims=True), 1e-30)
    identity = expit(np.array([1.75, .75])[None, :]+.35*np.asarray([p.identity_z for p in particles]))[:, None, :]
    identity = np.broadcast_to(identity, g['u'].shape)
    means = np.stack([g['u'], g['depth'], g['length'], g['axis'], identity], -1)
    contact = (np.max(abs(poses[:, :2]), axis=-1) <= .04)
    return dict(means=means, p_source=probs, p_visible=pany,
                contact=np.broadcast_to(contact[None, :], pany.shape),
                geometric_length=g['geometric_length'], angle_audit=g['angle_audit'])


def render_model(particle, pose):
    r = predict_batch([particle], [pose])
    return {k: v[0, 0] for k, v in r.items()}

predict = render_model


def _rng(key):
    digest = hashlib.sha256(repr(key).encode()).digest()
    return np.random.default_rng(np.frombuffer(digest[:16], dtype=np.uint32))


def _sample_model(model, key, *, frame_id=None, time_s=0.):
    rng = _rng(key)
    ps = np.asarray(model['p_source'])
    source = int(rng.choice(3, p=np.r_[ps, max(0., 1-ps.sum())]/np.r_[ps, max(0., 1-ps.sum())].sum()))
    contact = bool(model['contact'])
    public = dict(frame_id=str(frame_id if frame_id is not None else key), time_s=float(time_s),
                  visible=source<2, contact=contact)
    if source == 2:
        public.update({f: None for f in FIELDS})
        return public, None
    mu = model['means'][source]
    obs = mu + rng.normal(size=5)*SDS
    obs[2] = max(0., obs[2]); obs[4] = np.clip(obs[4], 0, 1)
    obs[3] = (obs[3]+np.pi/2)%np.pi-np.pi/2
    public.update(zip(FIELDS, map(float, obs)))
    # Reuse established miss/measurement filter, without its GT tangent channels.
    legal = filter_obs_for_control({'visible': 1., 'image_u_m': public['image_u_m'],
                                     'image_depth_m': public['image_depth_m']})
    public['image_u_m'] = legal['image_u_m']; public['image_depth_m'] = legal['image_depth_m']
    return public, source


def sample_model_observation(particle, pose, key, **kwargs):
    """Legal lookahead branch observation sampled from one persistent model world."""
    return _sample_model(render_model(particle, pose), key, **kwargs)[0]


def render(world, pose, key, *, frame_id=None, time_s=0.):
    """Actual sensor and separate audit. Only return public to policy/belief."""
    pose = np.asarray(pose, float)
    model = render_model(world, pose)
    public, source = _sample_model(model, key, frame_id=frame_id, time_s=time_s)
    audit = dict(frame_id=public['frame_id'], time_s=public['time_s'],
                 visible=public['visible'], source_is_target=source==0,
                 contact=bool(model['contact']), true_u_m=None,
                 geometric_length_m=None, out_of_plane_angle_rad=None, passed=False)
    if source is not None:
        mu = model['means'][source]
        angle = float(model['angle_audit'][source])
        length = float(model['geometric_length'][source])
        audit.update(true_u_m=float(mu[0]), geometric_length_m=length,
                     out_of_plane_angle_rad=angle,
                     passed=bool(source==0 and model['contact'] and length>=.015 and
                                 abs(mu[0])<=.002 and angle<=np.deg2rad(10)))
        # Established plane-intersection reference is audit-only. Its single point
        # is never substituted for clipped centroids or used for visibility.
        yaw=pose[2]; c,s=np.cos(yaw),np.sin(yaw)
        rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        _, seg, hit, dp = probe_plane_hit(world.points[source], np.r_[pose[:2], 0.], rot)
        audit['reference_plane_hit'] = bool(hit)
        audit['reference_segment'] = int(seg)
    return public, audit


def render_block(world, pose, key, *, start_time=0.):
    pairs=[render(world, pose, (key, i), frame_id=f'{key}/{i}', time_s=start_time+t)
           for i,t in enumerate(FRAME_OFFSETS)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _valid_five(frames):
    if len(frames)!=5 or len({f['frame_id'] for f in frames})!=5:
        return False
    t=np.array([f['time_s'] for f in frames])
    return bool(np.allclose(t-t[0], FRAME_OFFSETS, atol=1e-7))


def public_stop(frames):
    """Observable LA proxy; 3D angle is explicitly audit-only, never observed."""
    return bool(_valid_five(frames) and all(f['visible'] and f['contact'] and
        f['visible_length_m']>=.015 and abs(f['image_u_m'])<=.002 and
        f['identity_score']>=.75 for f in frames))


def audit_stop(frames):
    return bool(_valid_five(frames) and all(f['passed'] for f in frames))


def _safe_log(value):
    value=np.asarray(value)
    out=np.full(value.shape,-np.inf,dtype=float)
    np.log(value,out=out,where=value>0)
    return out


def _feature_logpdf(obs, means):
    diff=np.asarray(obs)-means
    diff[..., 3]=(diff[..., 3]+np.pi/2)%np.pi-np.pi/2
    term=-.5*(diff/SDS)**2-np.log(SDS*np.sqrt(2*np.pi))
    # Clipping measurement distribution has point masses at identity 0/1,
    # and length 0. Use proper tail probability at those atoms.
    for dim, bound, direction in [(2,0.,-1),(4,0.,-1),(4,1.,1)]:
        if obs[dim]==bound:
            prob=ndtr(direction*(means[..., dim]-bound)/SDS[dim])
            term[..., dim]=np.log(np.maximum(prob,1e-300))
    return term.sum(-1)


def log_likelihood_batch(public, particles, pose):
    model=predict_batch(particles,[pose]); probs=model['p_source'][:,0]
    if bool(public['contact']) != bool(model['contact'][0,0]):
        return np.full(len(particles), -np.inf)
    if not public['visible']:
        return _safe_log(1-probs.sum(-1))
    obs=np.array([public[f] for f in FIELDS],float)
    ll=_feature_logpdf(obs,model['means'][:,0])
    return logsumexp(_safe_log(probs)+ll,axis=-1)


def log_likelihood(public, particle, pose):
    return float(log_likelihood_batch(public,[particle],pose)[0])


def public_quality(public):
    if not public['visible']:
        return 0.
    return float(np.clip(public['visible_length_m']/.018,0,1)*
                 np.exp(-.5*(public['image_u_m']/.004)**2)*public['identity_score'])


def predicted_public_utility(particles, poses, weights=None):
    """Shared analytic A task response, no robot cost. Public single-frame gate P.

    Five-frame blocks can be estimated by the joint persistent-world probability
    q**5 per particle, then posterior average. It is not (mean q)**5.
    """
    model=predict_batch(particles,poses); m=model['means']
    pu=ndtr((.002-m[...,0])/SDS[0])-ndtr((-.002-m[...,0])/SDS[0])
    pl=ndtr((m[...,2]-.015)/SDS[2]); pi=ndtr((m[...,4]-.75)/SDS[4])
    q=(model['p_source']*pu*pl*pi).sum(-1)*model['contact']
    weights=np.ones(len(particles))/len(particles) if weights is None else np.asarray(weights)
    return dict(single_gate=weights@q, block_gate=weights@(q**5),
                visible=weights@model['p_visible'], per_particle_gate=q)


def _logprior(w):
    val=-.5*np.sum((w.registration/.004)**2)
    for controls,sd in [(w.y_controls,.003),(w.z_controls,.0015)]:
        white=np.linalg.solve(CHOL,controls.T).T/sd
        val-=.5*np.sum(white**2)
    val-=.5*(np.sum(w.identity_z**2)+np.sum(w.detect_z**2))
    if np.any(w.depths<.025) or np.any(w.depths>.045): return -np.inf
    return float(val)


def _propose(w,rng,scale):
    # Symmetric blocked random walk: each move changes one identifiable block,
    # avoiding simultaneous destruction of all five tightly measured features.
    modes=w.modes.copy();reg=w.registration.copy();yc=w.y_controls.copy()
    zc=w.z_controls.copy();depths=w.depths.copy();iz=w.identity_z.copy();dz=w.detect_z.copy()
    block=int(rng.integers(7));source=int(rng.integers(2))
    if block==0: reg+=rng.normal(0,.00065*scale,2)
    elif block==1: yc[source]+=rng.normal(size=5)@CHOL.T*.00065*scale
    elif block==2: zc[source]+=rng.normal(size=5)@CHOL.T*.00035*scale
    elif block==3: depths[source]+=rng.normal(0,.00045*scale)
    elif block==4: iz[source]+=rng.normal(0,.30*scale)
    elif block==5: dz[source]+=rng.normal(0,.30*scale)
    else: modes[source,int(rng.integers(2))]*=-1
    return _world(reg,modes,yc,zc,depths,iz,dz)


class Belief:
    """Persistent particle filter with full-history MH, unique evidence and diagnostics."""
    def __init__(self, particles, weights=None, *, seed=720282, ess_fraction=.55,
                 mh_steps=4, max_history=None, support='continuous'):
        self.particles=list(particles); n=len(particles)
        if not n: raise ValueError('nonempty particles required')
        self.weights=np.ones(n)/n if weights is None else np.asarray(weights,float)/np.sum(weights)
        self.log_weights=_safe_log(self.weights)
        self.rng=np.random.default_rng(seed); self.ess_fraction=ess_fraction
        self.mh_steps=mh_steps; self.history=[]; self.seen=set(); self.diagnostics=[]
        if support not in ('continuous','finite'): raise ValueError('support must be continuous or finite')
        self.support=support
        self.max_history=max_history
    @property
    def ess(self): return float(1/np.sum(self.weights**2))
    def copy(self):
        obj=Belief(self.particles,self.weights,seed=0,ess_fraction=self.ess_fraction,mh_steps=self.mh_steps,support=self.support)
        obj.rng.bit_generator.state=self.rng.bit_generator.state
        obj.log_weights=self.log_weights.copy()
        obj.history=[(dict(o),p.copy()) for o,p in self.history]
        obj.seen=set(self.seen); obj.diagnostics=[dict(d) for d in self.diagnostics]
        return obj
    def _history_loglike(self,particles):
        return sum((log_likelihood_batch(o,particles,p) for o,p in self.history),np.zeros(len(particles)))
    def update(self, public, pose, *, rejuvenate=True):
        fid=public['frame_id']
        if fid in self.seen:
            return dict(duplicate=True,frame_id=fid,ess=self.ess)
        # Validate support before mutating evidence state.
        ll=log_likelihood_batch(public,self.particles,pose)
        lw=self.log_weights+ll
        z=logsumexp(lw); mismatch=not np.isfinite(z)
        if mismatch: raise ValueError('all particles incompatible with contact/public evidence')
        # Store exactly one public evidence event; never audit or policy reward.
        self.seen.add(fid); self.history.append((dict(public),np.asarray(pose,float).copy()))
        self.log_weights=lw-z
        self.weights=np.exp(self.log_weights)
        raw_ess=self.ess; maxw=float(self.weights.max())
        d=dict(frame_id=fid,duplicate=False,raw_ess=raw_ess,max_weight=maxw,
               severe_degeneracy=raw_ess<2.,resampled=False,mh_acceptance=0.,
               finite_particle_warning=raw_ess<max(4.,len(self.particles)*.08),
               log_evidence=float(z),history_frames=len(self.history))
        if rejuvenate and self.support=='continuous' and self.ess<len(self.particles)*self.ess_fraction:
            n=len(self.particles); marks=(self.rng.random()+np.arange(n))/n
            indices=np.searchsorted(np.cumsum(self.weights),marks).clip(0,n-1)
            self.particles=[self.particles[i] for i in indices]; self.weights=np.ones(n)/n
            self.log_weights=np.full(n,-np.log(n))
            current=self._history_loglike(self.particles)+np.array([_logprior(p) for p in self.particles])
            accepted=0
            # Three scales permit local repair while retaining target-correct MH.
            for step in range(self.mh_steps):
                proposals=[_propose(p,self.rng,[1.,.5,1.8][step%3]) for p in self.particles]
                prop=self._history_loglike(proposals)+np.array([_logprior(p) for p in proposals])
                accept=np.log(self.rng.random(n))<prop-current
                self.particles=[proposals[i] if accept[i] else self.particles[i] for i in range(n)]
                current=np.where(accept,prop,current); accepted+=int(accept.sum())
            d.update(resampled=True,mh_acceptance=accepted/max(1,n*self.mh_steps))
        d['ess_after']=self.ess
        d['support']=self.support
        d['unique_geometries']=len({p.points.tobytes() for p in self.particles})
        d['post_resample_ess_is_not_coverage']=bool(d['resampled'])
        self.diagnostics.append(d)
        return d
    def update_block(self,frames,pose,**kwargs):
        return [self.update(f,pose,**kwargs) for f in frames]


def particle_axis_plane_angle(particles, pose):
    """[N,2] RMS angle for each legal hypothesized source, never actual truth.

    Same clipped component definition as audit. Invisible source returns NaN.
    Callers may combine these angles with public evidence association weights.
    """
    m=predict_batch(particles,[pose])
    return np.where(m['geometric_length'][:,0]>.0005,m['angle_audit'][:,0],np.nan)


def posterior_angle_probability(particles, weights, pose, public=None, *, degrees=10.):
    """P(associated visible segment within angle limit | legal particles, public).

    If public is supplied, condition each source mixture on that exact public
    frame; particle weights must ALREADY contain that frame (no double counting).
    Returns probability and source-association mass for diagnostic transparency.
    """
    m=predict_batch(particles,[pose]); prob=m['p_source'][:,0]
    if public is not None and public['visible']:
        obs=np.array([public[f] for f in FIELDS])
        lp=_safe_log(prob)+_feature_logpdf(obs,m['means'][:,0])
        norm=logsumexp(lp,axis=-1,keepdims=True)
        assoc=np.exp(lp-np.where(np.isfinite(norm),norm,0.))
    else:
        assoc=prob/np.maximum(prob.sum(-1,keepdims=True),1e-300)
    mass=np.asarray(weights)[:,None]*assoc
    good=(m['angle_audit'][:,0]<=np.deg2rad(degrees))&(m['geometric_length'][:,0]>.0005)
    return dict(probability=float(np.sum(mass*good)),
                source_mass=mass.sum(0),observable_estimate=True,
                criterion='posterior hypothesis geometry; not measured out-of-plane angle')


def log_likelihood_from_prediction(public, prediction, action_index=0):
    """Reuse predict_batch for many B branches: all P particles, no geometry rerun."""
    prob=prediction['p_source'][:,action_index]
    contact=prediction['contact'][:,action_index]
    if not public['visible']:
        ll=_safe_log(1-prob.sum(-1))
    else:
        obs=np.array([public[f] for f in FIELDS],float)
        feature_ll=_feature_logpdf(obs,prediction['means'][:,action_index])
        ll=logsumexp(_safe_log(prob)+feature_ll,axis=-1)
    return np.where(contact==bool(public['contact']),ll,-np.inf)


def sample_public_from_prediction(prediction, particle_index, action_index, key, **kwargs):
    """Sample one model branch from a precomputed P×A geometric response."""
    model={k:v[particle_index,action_index] for k,v in prediction.items()}
    return _sample_model(model,key,**kwargs)[0]


def sample_block_from_prediction(prediction, particle_index, action_index, key, *, start_time=0.):
    """Five new model frames; persistent hypothesis and nuisances across block."""
    return [sample_public_from_prediction(prediction,particle_index,action_index,(key,i),
            frame_id=f'{key}/{i}',time_s=start_time+offset)
            for i,offset in enumerate(FRAME_OFFSETS)]

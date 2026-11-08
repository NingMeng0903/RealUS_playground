"""Persistent anatomical world, extracted-feature sensor, and shared image servo.

The execution plant is explicitly a velocity/rate-limited Cartesian surrogate.
It is not the original SMPL/Genesis benchmark or a robot/contact simulator.
Outcomes are computed by executing the servo against fixed vessel geometry;
there is no class-to-success table or assigned action-duration table.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib
import numpy as np
from scipy.spatial.transform import Rotation


def unit(v):
    a = np.asarray(v, dtype=float)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def window(s, lo, hi):
    t = np.clip((np.asarray(s) - lo) / (hi - lo), 0., 1.)
    return np.where((np.asarray(s) > lo) & (np.asarray(s) < hi), 16*t*t*(1-t)*(1-t), 0.)


def interp_rows(s, values, query):
    q = np.atleast_1d(query)
    a = np.column_stack([np.interp(q, s, values[:, j]) for j in range(values.shape[1])])
    return a[0] if np.ndim(query) == 0 else a


@dataclass(frozen=True)
class Spec:
    object_id: str
    geometry_seed: int
    u: int
    v: int
    knee_deg: float = 0.
    width_scale: float = 1.
    length_scale: float = 1.
    amplitude_scale: float = 1.


@dataclass(frozen=True)
class FeaturePacket:
    frame_id: int
    object_id: str
    structure_id: str
    capture_time: float
    release_time: float
    capture_position: tuple
    capture_rotation: tuple
    detected: bool
    lateral_m: float | None
    depth_m: float | None
    confidence: float


@dataclass
class ServoState:
    position: np.ndarray
    rotation: np.ndarray
    time: float = 0.
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_id: int = 0
    queue: list = field(default_factory=list)
    recon_history: list = field(default_factory=list)
    used_frames: set = field(default_factory=set)


@dataclass
class Result:
    program_id: str
    success: bool
    status: str
    duration: float
    end_error_m: float
    nominal_duration: float
    accepted_frames: int
    unique_frames: list
    coverage: float
    packets: list
    audit: dict

    def record(self):
        d = asdict(self)
        d.pop('packets')
        return d


class LegWorld:
    """Private world. Policies receive packets/context, never this instance."""
    def __init__(self, atlas, spec: Spec, n=260):
        self.atlas, self.spec = atlas, spec
        self.s = np.linspace(0., 1., n)
        base_s = atlas.centerline_progress
        xi0 = np.asarray(atlas.centerline_xi, float).copy()
        xi0[:, 0] = np.unwrap(xi0[:, 0])
        self.base_xi = interp_rows(base_s, xi0, self.s)
        rng = np.random.default_rng(spec.geometry_seed)
        self.u_amp = spec.amplitude_scale * rng.uniform(.085, .145)
        self.v_amp = spec.amplitude_scale * rng.uniform(.16, .27)
        self.v_lo = rng.uniform(.49, .54)
        self.v_hi = rng.uniform(.92, .97)
        self.true_xi = self.base_xi.copy()
        self.true_xi[:, 0] += spec.u*self.u_amp*window(self.s, .075, .30)
        self.true_xi[:, 0] += spec.v*self.v_amp*window(self.s, self.v_lo, self.v_hi)
        # Persistent depth variation; independent of class, pose, and body shape.
        self.true_xi[:, 2] = np.clip(self.true_xi[:, 2] + rng.uniform(-.015, .015)*window(self.s,.34,.94), .01,.98)
        self.canonical_true = np.asarray(atlas.point_at_material(self.true_xi), float)
        self.true_points = self.pose(self.canonical_true)
        self.references = {}
        self.audit_frames = {}
        self.digest = hashlib.sha256(np.ascontiguousarray(self.true_xi).tobytes()).hexdigest()
        self._next_capture = 0.

    def pose(self, points):
        return np.asarray(self.atlas.deform_points(points, self.spec.knee_deg,
                           (self.spec.width_scale, self.spec.length_scale)), float)

    def unpose(self, points):
        return np.asarray(self.atlas.undeform_points(points, self.spec.knee_deg,
                           (self.spec.width_scale, self.spec.length_scale)), float)

    def reference(self, mode=0):
        if mode in self.references:
            return self.references[mode]
        xi = self.base_xi.copy()
        # These are declared, fixed skill parameters, not the hidden V.
        xi[:, 0] += mode*.20*window(self.s, .51, .95)
        inside = np.asarray(self.atlas.point_at_material(xi),float)
        skin_xi=xi.copy(); skin_xi[:,2]=0.
        skin=np.asarray(self.atlas.point_at_material(skin_xi),float)
        skin_w, inside_w = self.pose(skin), self.pose(inside)
        z=unit(skin_w-inside_w)
        t=unit(np.gradient(inside_w,axis=0))
        y=unit(t-z*np.sum(t*z,axis=1,keepdims=True))
        x=unit(np.cross(y,z)); y=unit(np.cross(z,x))
        rotations=np.stack((x,y,z),axis=-1)
        self.references[mode]=(skin_w,rotations,inside_w)
        return self.references[mode]

    def reference_at(self,s,mode=0):
        p,r,_=self.reference(mode)
        i=int(np.clip(np.searchsorted(self.s,s),0,len(self.s)-1))
        return interp_rows(self.s,p,s),r[i]

    def new_state(self,s,mode=0):
        p,r=self.reference_at(s,mode)
        return ServoState(p.copy(),r.copy())

    def observe(self,state,rng,delay=.04,noise=.00035):
        """Finite-thickness plane intersection followed by feature extraction.

        A miss never returns a nearest-truth displacement. Frame contents are
        generated once at capture and stay immutable while queued.
        """
        a=self.true_points[:-1]; b=self.true_points[1:]
        pa=(a-state.position)@state.rotation
        pb=(b-state.position)@state.rotation
        dy=pb[:,1]-pa[:,1]
        tt=np.clip(np.divide(-pa[:,1],dy,out=np.full_like(dy,.5),where=np.abs(dy)>1e-12),0,1)
        pp=pa+tt[:,None]*(pb-pa)
        in_slice=(pa[:,1]*pb[:,1]<=0)|(np.minimum(np.abs(pa[:,1]),np.abs(pb[:,1]))<.0015)
        valid=in_slice&(np.abs(pp[:,0])<.025)&(-pp[:,2]>.001)&(-pp[:,2]<.100)
        # Sensor produces a recognized target feature; no true segment ID escapes.
        ids=np.flatnonzero(valid)
        detected=bool(len(ids))
        if detected:
            j=ids[np.argmin(np.abs(pp[ids,1])+.05*np.abs(pp[ids,0]))]
            u=float(pp[j,0]+rng.normal(0,noise)); depth=float(-pp[j,2]+rng.normal(0,noise))
            q=float(np.clip(1-abs(u)/.025,.05,1))
            true_s=float(self.s[j]+tt[j]*(self.s[j+1]-self.s[j]))
        else:
            u=depth=None; q=0.
        packet=FeaturePacket(state.frame_id,self.spec.object_id,'right_leg_vessel_chain',state.time,state.time+delay,
                             tuple(state.position),tuple(state.rotation.reshape(-1)),
                             detected,u,depth,q)
        self.audit_frames[state.frame_id]={'true_s':true_s if detected else None,
                                          'true_u':float(pp[j,0]) if detected else None}
        state.frame_id+=1
        return packet

    def reconstructed_canonical(self,packet):
        if not packet.detected:
            return None
        p=np.asarray(packet.capture_position)
        r=np.asarray(packet.capture_rotation).reshape(3,3)
        observed=p+r@np.array([packet.lateral_m,0.,-packet.depth_m])
        return self.unpose(observed[None,:])[0]

    def evidence(self,packets,representation='material'):
        """Representation of legal reconstructed features, not simulator labels."""
        values=[]
        material=[]
        base=np.asarray(self.atlas.centerline_points,float)
        for packet in packets:
            can=self.reconstructed_canonical(packet)
            if can is None: continue
            j=int(np.argmin(np.sum((base-can)**2,axis=1)))
            # Normalized anatomical chart feature; estimated from reconstructed point.
            xi=np.asarray(self.atlas.material_at_point(can[None,:]))[0]
            ref=self.atlas.centerline_xi[j]
            angle=float(np.arctan2(np.sin(xi[0]-ref[0]),np.cos(xi[0]-ref[0])))
            material.append(angle)
            rw=np.asarray(packet.capture_rotation).reshape(3,3)
            obs=np.asarray(packet.capture_position)+rw@np.array([packet.lateral_m,0,-packet.depth_m])
            refcan=base[j]
            if representation=='world': values.append(obs-self.pose(refcan[None,:])[0])
            elif representation=='anchor': values.append(can-refcan)
            else: values.append([angle,0.,0.])
        if not values: return np.zeros(3),0
        arr=np.asarray(values)
        feature=np.median(arr,axis=0)
        signed=float(np.median(material))
        return feature,int(np.sign(signed))

    def public_context(self,start,end,mode):
        p,r,_=self.reference(mode)
        sel=(self.s>=start)&(self.s<=end)
        p=p[sel]; r=r[sel]
        length=float(np.sum(np.linalg.norm(np.diff(p,axis=0),axis=1)))
        turn=float(np.sum(np.linalg.norm(Rotation.from_matrix(np.transpose(r[:-1],(0,2,1))@r[1:]).as_rotvec(),axis=1))) if len(r)>1 else 0.
        return {'path_length':length,'path_turn':turn,'start':float(start),'end':float(end)}


def _rotate_toward(r,target,max_angle):
    delta=Rotation.from_matrix(r.T@target).as_rotvec()
    norm=float(np.linalg.norm(delta))
    if norm>max_angle:delta*=max_angle/norm
    return r@Rotation.from_rotvec(delta).as_matrix()


def move_to(world,state,target_s,rng,mode=0,dt=.04,speed=.08,max_time=20.):
    target,r=world.reference_at(target_s,mode)
    initial=state.time
    while state.time-initial<max_time:
        error=target-state.position
        if np.linalg.norm(error)<.0005 and np.linalg.norm(Rotation.from_matrix(state.rotation.T@r).as_rotvec())<.02:
            return True
        v=error/.12
        norm=np.linalg.norm(v)
        if norm>speed:v*=speed/norm
        # First-order actuator lag, shared across all methods.
        state.velocity += min(1.,dt/.09)*(v-state.velocity)
        state.position += dt*state.velocity
        state.rotation=_rotate_toward(state.rotation,r,.9*dt)
        state.time+=dt
    return False


def run_scan(world,state,start,end,mode,rng,*,speed=.028,program_id='scan',max_time=20.,fresh_after=0.,delay=.04):
    """Run the same image servo; only declared reference parameters differ."""
    began=state.time
    moved=move_to(world,state,start,rng,mode)
    context=world.public_context(start,end,mode)
    nominal=max(context['path_length']/speed,.04)
    if not moved:
        return Result(program_id,False,'entry_timeout',state.time-began,float('inf'),nominal,0,[],0.,[],{})
    dt=.04; cursor=float(start); last_capture=-1e9
    last_valid=None; lost=0.; packets=[]; accepted=[]; bins=set()
    target_bins=12
    entry_time=state.time
    entry_error=float(np.linalg.norm(state.position-world.reference_at(start,mode)[0]))
    path_length=max(context['path_length'],.001)
    side_correction=0.
    while state.time-began<max_time:
        if state.time-last_capture>=.099:
            state.queue.append(world.observe(state,rng,delay=delay))
            last_capture=state.time
        ready=[p for p in state.queue if p.release_time<=state.time+1e-9]
        state.queue=[p for p in state.queue if p.release_time>state.time+1e-9]
        for packet in ready:
            if packet.frame_id in state.used_frames:continue
            state.used_frames.add(packet.frame_id);packets.append(packet)
            if state.time-packet.capture_time>.25:
                # Retain late observations for the ledger/belief, but an old
                # image error cannot be applied as a current-pose correction.
                continue
            if packet.detected:
                last_valid=packet;lost=0.
                can=world.reconstructed_canonical(packet)
                # Estimation uses the known nominal anatomical trajectory, not GT phase.
                canon0=np.asarray(world.atlas.centerline_points)
                jj=int(np.argmin(np.sum((canon0-can)**2,axis=1)))
                sobs=float(world.atlas.centerline_progress[jj])
                # Public acceptance is derived solely from the extracted feature.
                if abs(packet.lateral_m)<.003 and packet.capture_time>=fresh_after and start-.015<=sobs<=end+.015:
                    bi=int(np.clip((sobs-start)/max(end-start,1e-9)*target_bins,0,target_bins-1))
                    bins.add(bi); accepted.append(packet.frame_id)
                # Only released, detected features can alter the servo.
                side_correction += float(np.clip(.65*packet.lateral_m,-.003,.003))
                side_correction=float(np.clip(side_correction,-.022,.022))
            else:
                last_valid=None
        if last_valid is not None and state.time-last_valid.capture_time>.30:
            last_valid=None
        if last_valid is None:lost+=dt
        if lost>1.0:
            status='lost_lock';break
        # On a miss, halt forward command; do not invent the missing error.
        if last_valid is not None:
            # Shared quality throttle: progress depends on a released image
            # feature, never on the hidden vessel class or program identifier.
            alpha=1./(1.+(float(last_valid.lateral_m)/.002)**2)
            cursor=min(end,cursor+alpha*speed*dt/path_length*(end-start))
        pref,rref=world.reference_at(cursor,mode)
        target=pref+rref[:,0]*side_correction
        desired=(target-state.position)/.12
        norm=np.linalg.norm(desired)
        if norm>.075:desired*=.075/norm
        state.velocity+=min(1.,dt/.09)*(desired-state.velocity)
        state.position+=dt*state.velocity
        state.rotation=_rotate_toward(state.rotation,rref,.9*dt)
        state.time+=dt
        if cursor>=end-1e-9 and np.linalg.norm(target-state.position)<.001:
            # Capture at the terminal pose rather than silently losing the last frame.
            if state.time-entry_time>nominal+.6:
                status='completed';break
    else:
        status='timeout'
    coverage=len(bins)/target_bins
    # Current proxy task is geometric feature coverage, explicitly not B-mode quality.
    success=coverage>=.75 and status=='completed'
    if status=='completed' and not success:status='coverage_gap'
    err=abs(last_valid.lateral_m) if last_valid is not None else .025
    audit_bins=set()
    for packet in packets:
        truth=world.audit_frames.get(packet.frame_id,{})
        ss=truth.get('true_s')
        if ss is not None and start-.015<=ss<=end+.015 and abs(truth['true_u'])<.003 and packet.capture_time>=fresh_after:
            audit_bins.add(int(np.clip((ss-start)/max(end-start,1e-9)*target_bins,0,target_bins-1)))
    return Result(program_id,success,status,state.time-began,float(err),nominal,
                  len(set(accepted)),sorted(set(accepted)),coverage,packets,
                  {'world_hash':world.digest,'plant':'cartesian_rate_limited_surrogate',
                   'mode_parameter':mode,'start':start,'end':end,'speed':speed,
                   'entry_duration':entry_time-began,'entry_error_m':entry_error,
                   'exit_position_local':(world.reference_at(end,mode)[1].T@(state.position-world.reference_at(end,mode)[0])).tolist(),
                   'exit_rotation_local':Rotation.from_matrix(world.reference_at(end,mode)[1].T@state.rotation).as_rotvec().tolist(),
                   'exit_velocity_local':(world.reference_at(end,mode)[1].T@state.velocity).tolist(),
                   'independent_coverage':len(audit_bins)/target_bins,
                   'independent_success':len(audit_bins)/target_bins>=.75 and status=='completed'})


def capture_probe(world,state,start,end,rng,*,delay=.04,program_id='probe'):
    return run_scan(world,state,start,end,0,rng,speed=.010,program_id=program_id,
                    max_time=15.,delay=delay)

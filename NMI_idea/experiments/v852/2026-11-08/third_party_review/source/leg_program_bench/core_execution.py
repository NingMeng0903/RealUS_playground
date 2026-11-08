"""Causal program executor shared by observed and learned-bundle experiments.

The default plant is the explicitly labelled Cartesian lag plant. An optional
JointServoBackend runs the same sensor/servo from measured 8-DOF FK instead.
Only initialization resets state; calls and recovery preserve actual exits.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import copy
import hashlib
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .simulation import LegWorld, Spec, ServoState, Result, _rotate_toward
from .evidence_ledger import EvidenceLedger, EvidenceRecord

SPANS = {'near':(.10,.25),'probe':(.56,.60),'a':(.62,.76),
         'b':(.78,.91),'f':(.14,.21)}
EXIT_KEYS = ('exit_position_local','exit_rotation_local','exit_velocity_local')
LEVELS = {
    'basic':dict(amplitude_scale=1.,noise=.00035,dropout=0.,lag=.09),
    'variable':dict(amplitude_scale=1.65,noise=.00065,dropout=.03,lag=.14),
    'stress':dict(amplitude_scale=2.4,noise=.001,dropout=.08,lag=.21),
}

class CoreWorld(LegWorld):
    def __init__(self, atlas, spec, level='basic'):
        self.profile=LEVELS[level];self.level=level
        super().__init__(atlas,replace(spec,amplitude_scale=self.profile['amplitude_scale']))
    def observe(self,state,rng,delay=.04,noise=None):
        packet=super().observe(state,rng,delay=delay,noise=self.profile['noise'])
        if packet.detected and rng.random()<self.profile['dropout']:
            packet=replace(packet,detected=False,lateral_m=None,depth_m=None,confidence=0.)
        return packet

class ProgramExecutor:
    """Program wrapper: causal image feedback, independent acceptance, clock."""
    def __init__(self,world,state=None,*,backend=None):
        self.world=world;self.state=state or world.new_state(.60)
        self.backend=backend;self.calls=[];self.trace=[]
        self.ledger=EvidenceLedger();self.wall_start=time.perf_counter()
        self.dt=.04 if backend is None else backend.config.dt
        self.safe=True
        if backend is not None:self._sync_joint()
    def _sync_joint(self):
        s=self.backend.state();T=np.asarray(s['T_image_world'])
        self.state.position=T[:3,3].copy();self.state.rotation=T[:3,:3].copy()
        self.state.time=s['sim_time_s']
    def advance(self,target,rotation):
        s=self.state;dt=self.dt
        if self.backend is None:
            desired=(target-s.position)/.12
            norm=np.linalg.norm(desired)
            if norm>.075:desired*=.075/norm
            s.velocity+=min(1.,dt/self.world.profile['lag'])*(desired-s.velocity)
            s.position+=dt*s.velocity
            s.rotation=_rotate_toward(s.rotation,rotation,.9*dt)
            s.time+=dt
        else:
            T=np.eye(4);T[:3,3]=target;T[:3,:3]=rotation
            old=s.position.copy()
            row=self.backend.step_pose(T)
            self._sync_joint();s.velocity=(s.position-old)/dt
            self.safe=bool(row['safe'])
            self.trace.append(row)
    def wait(self,duration):
        end=self.state.time+max(0.,float(duration))
        target=self.state.position.copy();r=self.state.rotation.copy()
        while self.state.time+1e-8<end and self.safe:self.advance(target,r)
    def move(self,start,mode,max_time=20.):
        target,r=self.world.reference_at(start,mode);began=self.state.time
        while self.state.time-began<max_time and self.safe:
            if np.linalg.norm(target-self.state.position)<.0008 and np.linalg.norm(
                Rotation.from_matrix(self.state.rotation.T@r).as_rotvec())<.025:return True
            self.advance(target,r)
        return False
    def call(self,name,mode=0,*,fresh_after=0.):
        start,end=SPANS[name];w=self.world;s=self.state;began=s.time
        moved=self.move(start,mode)
        entry=s.time;ctx=w.public_context(start,end,mode)
        speed=.010 if name in ('near','probe') else .028
        nominal=max(ctx['path_length']/speed,.04)
        cursor=start;last_capture=-1e9;last_valid=None;lost=0.;side=0.
        packets=[];accepted=set();bins=set();truth_bins=set();status='entry_timeout' if self.safe else 'joint_safety_stop'
        if moved:
            while s.time-began<30. and self.safe:
                if s.time-last_capture>=.099:
                    s.queue.append(w.observe(s,self.rng));last_capture=s.time
                ready=[p for p in s.queue if p.release_time<=s.time+1e-8]
                s.queue=[p for p in s.queue if p.release_time>s.time+1e-8]
                for p in ready:
                    if p.frame_id in s.used_frames:continue
                    s.used_frames.add(p.frame_id)
                    # Late previous-program packets remain auditable, but do
                    # not enter this program's evidence or acceptance region.
                    if p.capture_time<entry-1e-8:continue
                    packets.append(p)
                    self.ledger.add(EvidenceRecord(
                        f'frame:{p.frame_id}',w.spec.object_id,name,p.capture_time,
                        p.release_time,p.release_time,asdict(p),kind='current'))
                    if s.time-p.capture_time>.25:continue
                    if p.detected:
                        last_valid=p;lost=0.
                        can=w.reconstructed_canonical(p)
                        j=int(np.argmin(np.sum((w.atlas.centerline_points-can)**2,axis=1)))
                        sobs=float(w.atlas.centerline_progress[j])
                        if abs(p.lateral_m)<.003 and p.capture_time>=fresh_after and start-.015<=sobs<=end+.015:
                            bins.add(int(np.clip((sobs-start)/(end-start)*12,0,11)))
                            accepted.add(p.frame_id)
                        side=float(np.clip(side+np.clip(.65*p.lateral_m,-.003,.003),-.022,.022))
                    else:last_valid=None
                if last_valid is not None and s.time-last_valid.capture_time>.30:last_valid=None
                if last_valid is None:lost+=self.dt
                if lost>1.0:status='lost_lock';break
                if last_valid is not None:
                    alpha=1/(1+(last_valid.lateral_m/.002)**2)
                    cursor=min(end,cursor+alpha*speed*self.dt/max(ctx['path_length'],.001)*(end-start))
                p,r=w.reference_at(cursor,mode);target=p+r[:,0]*side
                self.advance(target,r)
                if cursor>=end-1e-9 and np.linalg.norm(target-s.position)<.001 and s.time-entry>nominal+.6:
                    status='completed';break
            else:status='timeout' if self.safe else 'joint_safety_stop'
        for p in packets:
            a=w.audit_frames.get(p.frame_id,{})
            ss=a.get('true_s')
            if ss is not None and p.detected and start-.015<=ss<=end+.015 and abs(a['true_u'])<.003 and p.capture_time>=fresh_after:
                truth_bins.add(int(np.clip((ss-start)/(end-start)*12,0,11)))
        independent_success=status=='completed' and len(truth_bins)>=9
        if status=='completed' and len(bins)<9:status='coverage_gap'
        public_success=status=='completed' and len(bins)>=9
        success=bool(public_success and independent_success)
        endp,endr=w.reference_at(end,mode)
        record={'program':name,'mode':mode,'success':bool(success),'public_success':bool(public_success),
            'independent_success':bool(independent_success),
            'status':status,'duration':s.time-began,'service_duration':s.time-entry,
            'entry_duration':entry-began,'nominal_duration':nominal+.6,
            'scan_speed_m_s':speed,
            'exit_position_local':(endr.T@(s.position-endp)).tolist(),
            'exit_rotation_local':Rotation.from_matrix(endr.T@s.rotation).as_rotvec().tolist(),
            'exit_velocity_local':(endr.T@s.velocity).tolist(),'frame_count':len(packets),
            'frame_capture_times':[p.capture_time for p in packets],
            'fresh_frames':len(accepted),'coverage':len(truth_bins)/12,
            'capture_min':min((p.capture_time for p in packets),default=None),
            'capture_max':max((p.capture_time for p in packets),default=None),
            'start_time':began,'end_time':s.time,
            'backend':'cartesian_lag' if self.backend is None else self.backend.config.backend}
        self.calls.append(record)
        return record,packets
    def set_rng(self,seed):self.rng=np.random.default_rng(seed);return self

def local_bundle(atlas,spec,level,seed):
    world=CoreWorld(atlas,spec,level)
    near_engine=ProgramExecutor(world,world.new_state(.10)).set_rng(seed)
    near,packets=near_engine.call('near')
    nf,_=world.evidence(packets)
    programs={};probe_value=0.
    for index,(name,mode) in enumerate((('a',-1),('a',1),('b',-1),('b',1),('f',0),('probe',0))):
        state=world.new_state(SPANS[name][0],0)
        engine=ProgramExecutor(world,state).set_rng(seed+100+index)
        result,packets=engine.call(name,mode)
        programs[f'{name}:{mode}']=result
        if name=='probe':probe_value=float(world.evidence(packets)[0][0])
    row={'object_id':spec.object_id,'stratum':f'{spec.knee_deg:g}:{spec.width_scale:g}:{spec.length_scale:g}',
         'near':float(nf[0]),'probe':probe_value,'programs':programs,
         'public_pose':spec.knee_deg,'public_shape':[spec.width_scale,spec.length_scale],
         'acquisition_seconds':near['duration']+sum(p['duration'] for p in programs.values())}
    return row,{'spec':asdict(world.spec),'world_hash':world.digest,'near_result':near}

@dataclass(frozen=True)
class Tree:
    name:str
    kind:str='react'
    default_mode:int=-1
    near_mapping:int=1
    probe_mapping:int=1
    recover:bool=True

TREES=tuple([Tree('fixed_minus','fixed',-1),Tree('fixed_plus','fixed',1)] +
    [Tree(f'react_{d:+d}_{m:+d}','react',d,m) for d in (-1,1) for m in (-1,1)] +
    [Tree(f'probe_{m:+d}','probe',-1,1,m) for m in (-1,1)] +
    [Tree(f'wait_{m:+d}','wait',-1,m) for m in (-1,1)])

def sign(value):return 1 if value>=0 else -1

def old_ledger(object_id,value,release,hidden=False):
    ledger=EvidenceLedger()
    packet=EvidenceRecord('old:near',object_id,'near',-1.,-.1,0.,{'near':float(value)},'historical')
    ledger.add(packet);ledger.reschedule_release(packet.packet_id,float(release))
    ledger.set_hidden(packet.packet_id,bool(hidden))
    content_hash=hashlib.sha256(json.dumps(packet.payload,sort_keys=True).encode()).hexdigest()
    return ledger,content_hash

def visible_near(ledger,now,object_id):
    rows=ledger.visible(now,object_id=object_id,region_id='near',include_historical=True)
    for p in rows:
        if p.kind=='historical':return float(p.payload['near'])
    return None

def execute_tree(world,near,tree,*,release=0.,hidden=False,chain=('a','b','f'),seed=0,backend=None):
    engine=ProgramExecutor(world,backend=backend).set_rng(seed)
    old,packet_hash=old_ledger(world.spec.object_id,near,release,hidden)
    engine.ledger.add(EvidenceRecord('old:near',world.spec.object_id,'near',-1.,-.1,0.,{'near':near},'historical'))
    engine.ledger.reschedule_release('old:near',release);engine.ledger.set_hidden('old:near',hidden)
    decisions=[];mode=tree.default_mode;calls=0;recoveries=0;probe=None
    if tree.kind=='wait' and not hidden:engine.wait(release)
    if tree.kind=='probe':
        _,packets=engine.call('probe');probe=float(world.evidence(packets)[0][0]);calls+=1
        mode=tree.probe_mapping*sign(probe)
    success=True
    for name in chain:
        y=visible_near(old,engine.state.time,world.spec.object_id)
        if tree.kind in ('react','wait') and y is not None:mode=tree.near_mapping*sign(y)
        selected=0 if name=='f' else mode
        decisions.append({'time':engine.state.time,'stage':name,'mode':selected,'near_visible':y is not None,
                          'probe_visible':probe is not None})
        request=engine.state.time if name=='f' else 0.
        result,_=engine.call(name,selected,fresh_after=request);calls+=1
        if not result['public_success'] and tree.recover and name!='f' and engine.safe:
            # Natural executed backtracking and retry; no class-to-cost table.
            recoveries+=1
            mode=-selected
            result,_=engine.call(name,mode,fresh_after=request);calls+=1
            result['is_retry']=True
        success=bool(success and result['public_success'] and result.get('independent_success',result['success']))
        if not result['public_success']:break
    fresh=[r for r in engine.calls if r['program']=='f']
    return {'tree':tree.name,'success':bool(success and len(fresh)==1),
            'duration':engine.state.time,'program_calls':calls,'recoveries':recoveries,
            'new_frames':sum(r['frame_count'] for r in engine.calls),
            'fresh_frames':sum(r['fresh_frames'] for r in fresh),'calls':engine.calls,'decisions':decisions,
            'packet_hash':packet_hash,'release':release,'hidden':hidden,
            'world_hash':world.digest,'ledger':engine.ledger.summary(),
            'backend':'cartesian_lag' if backend is None else backend.config.backend,
            'wall_seconds':time.perf_counter()-engine.wall_start}

class BundleExecutor:
    """One persistent observed training object composed with public transitions.

    No test anatomy or hidden class is read. Service and exit are sampled as
    one joint row, and the object remains fixed across every program call.
    """
    def __init__(self,world,bundle,*,independent_rows=None,time_rows=None,rng=None):
        self.world=world;self.bundle=bundle;self.rows=independent_rows;self.time_rows=time_rows;self.rng=rng
        self.time=0.;self.source_s=.60;self.source_mode=0;self.exit=None;self.calls=[]
    def call(self,name,mode):
        from .run_experiment import _exit_bucket
        start,end=SPANS[name]
        bucket=_exit_bucket(self.exit)
        if not hasattr(self.world,'core_transition_cache'):self.world.core_transition_cache={}
        key=(self.source_s,self.source_mode,start,mode,bucket)
        if key not in self.world.core_transition_cache:
            state=self.world.new_state(self.source_s,self.source_mode)
            r=state.rotation.copy();b=np.asarray(bucket)
            state.position+=r@b[:3];state.rotation=r@Rotation.from_rotvec(b[3:6]).as_matrix();state.velocity=r@b[6:9]
            engine=ProgramExecutor(self.world,state).set_rng(0)
            ok=engine.move(start,mode)
            self.world.core_transition_cache[key]=(state.time,ok)
        transition,ok=self.world.core_transition_cache[key]
        source=self.bundle if self.rows is None else self.rows[int(self.rng.integers(len(self.rows)))]
        r=source['programs'][f'{name}:{mode}']
        time_record=r if self.time_rows is None else self.time_rows[int(self.rng.integers(len(self.time_rows)))]['programs'][f'{name}:{mode}']
        speed=r.get('scan_speed_m_s',.010 if name=='probe' else .028)
        nominal=self.world.public_context(start,end,mode)['path_length']/speed+.6
        scale=nominal/max(r['nominal_duration'],.001)
        service=time_record['service_duration']*scale
        self.time+=transition+service
        self.source_s=end;self.source_mode=mode;self.exit=r
        public_success=bool(ok and r['public_success'])
        result={**r,'success':bool(public_success and r['success']),
                'public_success':public_success,'duration':transition+service}
        self.calls.append(result)
        return result

def predict_tree(world,bundle,tree,*,release=0.,hidden=False,chain=('a','b','f'),independent_rows=None,time_rows=None,rng=None):
    e=BundleExecutor(world,bundle,independent_rows=independent_rows,time_rows=time_rows,rng=rng)
    ledger,_=old_ledger('prediction_object',bundle['near'],release,hidden)
    mode=tree.default_mode
    if tree.kind=='wait' and not hidden:e.time=float(release)
    if tree.kind=='probe':
        e.call('probe',0);mode=tree.probe_mapping*sign(bundle['probe'])
    success=True;recoveries=0
    for name in chain:
        near=visible_near(ledger,e.time,'prediction_object')
        if tree.kind in ('react','wait') and near is not None:
            mode=tree.near_mapping*sign(near)
        selected=0 if name=='f' else mode
        r=e.call(name,selected)
        if not r['public_success'] and tree.recover and name!='f':
            recoveries+=1;mode=-selected;r=e.call(name,mode)
        success=bool(success and r['success'])
        if not r['public_success']:break
    return {'success':bool(success and any(r['program']=='f' for r in e.calls)),
            'duration':e.time,'program_calls':len(e.calls),'recoveries':recoveries,
            'new_frames':sum(r['frame_count'] for r in e.calls),'fresh_frames':sum(r['fresh_frames'] for r in e.calls if r['program']=='f')}

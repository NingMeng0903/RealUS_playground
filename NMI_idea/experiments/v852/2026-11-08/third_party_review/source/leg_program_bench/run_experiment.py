"""Run a prospective, configuration-held-out leg feature experiment.

Usage: PYTHONPATH=work python -m leg_program_bench.run_experiment --output outputs/leg_pilot
Training contains local scans/paired diagnostic probes only. Complete test
chains and their outcomes are generated after predictions have been written.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation

from .smplx_scene import SmplxAtlasLeg as AtlasLeg
from .atlas_scene import AtlasLeg as SurrogateAtlasLeg
from .simulation import Spec,LegWorld,run_scan,capture_probe,move_to
from .learning import fit_joint,fit_evidence_pairs,common_policy_mixture

SPAN_A=(.62,.76)
SPAN_B=(.78,.91)
SPAN_F=(.14,.21)
SPAN_Q=(.56,.60)
MODES=(-1,1)
POLICIES=('minus','plus','diagnose')
MATERIAL_KEYS=['q','start','end','path_length','path_turn','entry_error_m']
ANCHOR_KEYS=['q','can_x','can_y','can_z','path_length','path_turn','entry_error_m']
WORLD_KEYS=['q','world_x','world_y','world_z','knee_deg','width_scale','length_scale','path_length','path_turn','entry_error_m']
EXIT_KEYS=('exit_position_local','exit_rotation_local','exit_velocity_local')


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def json_line(handle,obj):
    handle.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()


def context(world,start,end,mode,q,entry_position=None,entry_error_m=0.,probe=False):
    d=world.public_context(start,end,mode)
    public=world.spec
    p,_=world.reference_at(start,mode)
    can=world.unpose(p[None,:])[0]
    entry_distance=float(np.linalg.norm(p-entry_position)) if entry_position is not None else 0.
    speed=.010 if probe else .028
    d.update(q=float(q),entry_error_m=float(entry_error_m),
             can_x=float(can[0]),can_y=float(can[1]),can_z=float(can[2]),
             world_x=float(p[0]),world_y=float(p[1]),world_z=float(p[2]),
             knee_deg=public.knee_deg,width_scale=public.width_scale,length_scale=public.length_scale,
             nominal_duration=d['path_length']/speed+.6,
             program_id='probe' if probe else f'scan_{mode:+d}')
    return d


def get_evidence(world,rng):
    # Local diagnostic calibration uses separate standardized starts. The two
    # feature records retain the same object identity and hidden configuration.
    near=capture_probe(world,world.new_state(.10),.10,.25,rng,program_id='near_sa')
    far=capture_probe(world,world.new_state(SPAN_Q[0]),*SPAN_Q,rng,program_id='diagnostic')
    near_f,y=world.evidence(near.packets)
    far_f,q=world.evidence(far.packets)
    return int(y),int(q),near,far,near_f,far_f


def generate_training(atlas,n,seed,rho,output):
    rng=np.random.default_rng(seed)
    records=[];pairs=[];manifest=[]
    with (output/'local_records.jsonl').open('w') as handle:
        for i in range(n):
            u=int(rng.choice([-1,1]));v=u if rng.random()<(1+rho)/2 else -u
            spec=Spec(f'train_{i:04}',seed*1000+i,u,v,
                      float(rng.choice([0,30])),float(rng.choice([.9,1.,1.1])),
                      float(rng.choice([.95,1.,1.05])))
            world=LegWorld(atlas,spec)
            y,q,near,diagnostic,_,_=get_evidence(world,rng)
            pairs.append({'object_id':spec.object_id,'y':y,'q':q})
            manifest.append({**asdict(spec),'split':'train','world_hash':world.digest})
            for start,end in (SPAN_A,SPAN_B,SPAN_F):
                modes=(0,) if (start,end)==SPAN_F else MODES
                for mode in modes:
                    state=world.new_state(max(0,start-.015))
                    entry_error=float(rng.choice([-.0015,0.,.0015]))
                    state.position+=state.rotation[:,0]*entry_error
                    ctx=context(world,start,end,mode,q,state.position,abs(entry_error))
                    out=run_scan(world,state,start,end,mode,rng,program_id=ctx['program_id'])
                    rec={**ctx,**out.record(),'nominal_duration':ctx['nominal_duration'],
                         'duration':out.duration-out.audit['entry_duration'],
                         'entry_error_m':out.audit['entry_error_m'],
                         **{key:out.audit[key] for key in EXIT_KEYS},
                         'success':bool(out.audit.get('independent_success',False)),
                         'object_id':spec.object_id,'observed_y':y}
                    records.append(rec);json_line(handle,rec)
            # Local Q probe cost is also measured, never assigned as a penalty.
            qc=context(world,*SPAN_Q,0,q,world.reference_at(SPAN_Q[0],0)[0],probe=True)
            qr={**qc,**diagnostic.record(),'program_id':'probe','nominal_duration':qc['nominal_duration'],
                'duration':diagnostic.duration-diagnostic.audit['entry_duration'],
                'entry_error_m':diagnostic.audit['entry_error_m'],
                **{key:diagnostic.audit[key] for key in EXIT_KEYS},
                'success':bool(diagnostic.audit.get('independent_success',False)),
                'object_id':spec.object_id,'observed_y':y}
            records.append(qr);json_line(handle,qr)
            if (i+1)%8==0: print(f'local objects {i+1}/{n}',flush=True)
    write_json(output/'training_manifest.json',manifest)
    write_json(output/'evidence_pairs.json',pairs)
    return records,pairs


def nominal_start(world,s):
    return world.reference_at(s,0)[0]


def _exit_bucket(record):
    if record is None:return (0.,)*9
    # Declared finite exit abstraction; no per-batch averaging of exits.
    scales=np.array([.0001]*3+[.002]*3+[.0005]*3)
    x=np.concatenate([np.asarray(record[k],float) for k in EXIT_KEYS])
    return tuple((np.round(x/scales)*scales).tolist())


def entry_transition(world,source_s,source_mode,target_s,target_mode,bucket):
    if not hasattr(world,'transition_cache'):world.transition_cache={}
    key=(source_s,source_mode,target_s,target_mode,bucket)
    if key in world.transition_cache:return world.transition_cache[key]
    state=world.new_state(source_s,source_mode)
    r=state.rotation.copy();b=np.asarray(bucket)
    state.position+=r@b[:3]
    state.rotation=r@Rotation.from_rotvec(b[3:6]).as_matrix()
    state.velocity=r@b[6:9]
    ok=move_to(world,state,target_s,np.random.default_rng(0),target_mode)
    error=float(np.linalg.norm(state.position-world.reference_at(target_s,target_mode)[0]))
    result=(state.time,error,ok)
    world.transition_cache[key]=result
    return result


def predict_fixed(model,pairs,world,y,policy,deadlines,rng,*,fresh=True,nmc=256,hide=False,delay=0.):
    """Propagate one persistent diagnostic class through all local kernels.

    Delay compares the declared wait-for-old-evidence policy, not an assertion
    that no alternative delayed-information policy can do better.
    """
    distribution=(getattr(pairs,'marginal_distribution',pairs.distribution(999))
                  if hide else pairs.distribution(y))
    categories=list(distribution)
    prob=np.asarray([distribution[k] for k in categories]);prob/=prob.sum()
    qq=rng.choice(categories,size=nmc,p=prob)
    duration=np.full(nmc,float(delay));good=np.ones(nmc,bool)
    end_error=np.zeros(nmc)
    exits=[None]*nmc
    for q in categories:
        idx=np.flatnonzero(qq==q)
        if not len(idx):continue
        mode=(1 if q>=0 else -1) if policy=='diagnose' else (-1 if policy=='minus' else 1)
        stages=[]
        if policy=='diagnose':stages.append((*SPAN_Q,0,True))
        stages.extend([(*SPAN_A,mode,False),(*SPAN_B,mode,False)])
        if fresh:stages.append((*SPAN_F,0,False))
        source_s,source_mode=.60,0
        for start,end,stage_mode,is_probe in stages:
            groups={}
            for particle in idx:
                groups.setdefault(_exit_bucket(exits[particle]),[]).append(int(particle))
            for bucket,particles in groups.items():
                entry_seconds,error,ok=entry_transition(world,source_s,source_mode,start,stage_mode,bucket)
                ctx=context(world,start,end,stage_mode,q,entry_error_m=error,probe=is_probe)
                samples=model.sample(ctx,rng,n=len(particles))
                duration[particles]+=entry_seconds+np.asarray([s['duration'] for s in samples])
                good[particles]&=ok & np.asarray([s['success'] for s in samples])
                for particle,sample in zip(particles,samples):exits[particle]=sample
            source_s,source_mode=end,stage_mode
    return np.asarray([np.mean(good&(duration<=t)) for t in deadlines]),duration,good


def execute_fixed(world,y,policy,rng,*,fresh=True,delay=0.,initial_state=None):
    state=copy.deepcopy(initial_state) if initial_state is not None else world.new_state(.60)
    began=state.time
    state.time+=float(delay)
    calls=[]
    if policy=='diagnose':
        out=capture_probe(world,state,*SPAN_Q,rng,program_id='diagnostic')
        _,q=world.evidence(out.packets)
        calls.append(out)
        mode=1 if q>=0 else -1
    else:mode=-1 if policy=='minus' else 1
    for j,(start,end) in enumerate((SPAN_A,SPAN_B)):
        calls.append(run_scan(world,state,start,end,mode,rng,program_id=f'distal_{j}'))
        if calls[-1].status in ('lost_lock','timeout','entry_timeout'):break
    if fresh and len([o for o in calls if o.program_id.startswith('distal')])==2:
        request_time=state.time
        # run_scan's entry controller executes the entire return. No pose reset.
        calls.append(run_scan(world,state,*SPAN_F,0,rng,program_id='fresh_recheck',fresh_after=request_time))
    expected=3+(policy=='diagnose') if fresh else 2+(policy=='diagnose')
    success=len(calls)==expected and all(o.audit.get('independent_success',False) for o in calls)
    fresh_calls=[o for o in calls if o.program_id=='fresh_recheck']
    return {'policy':policy,'success':bool(success),'duration':state.time-began,
            'program_calls':len(calls),'new_frames':sum(len(o.packets) for o in calls),
            'fresh_frames':sum(o.accepted_frames for o in fresh_calls),
            'statuses':[o.status for o in calls],
            'world_hash':world.digest,'calls':[o.record() for o in calls]}


def build_bundle(records,pairs,keys,k=12):
    pt=fit_evidence_pairs(pairs)
    categories=pt.q_categories
    counts={q:1+sum(int(row['q']==q) for row in pairs) for q in categories}
    pt.marginal_distribution={q:n/sum(counts.values()) for q,n in counts.items()}
    return fit_joint(records,keys,k=k,extra_outcome_keys=EXIT_KEYS),pt


def make_test_specs(n,seed,rho):
    rng=np.random.default_rng(seed)
    specs=[]
    for i in range(n):
        u=int(rng.choice([-1,1]));v=u if rng.random()<(1+rho)/2 else -u
        gid=seed*1000+i
        # Same held-out internal object is paired across test pose/shape strata.
        for group,pose,width,length in (
            ('new_object_known_pose',0.,1.,1.),
            ('heldout_pose',60.,1.,1.),
            ('heldout_shape',30.,1.18,.88),
            ('heldout_pose_and_shape',60.,1.18,.88)):
            specs.append((group,Spec(f'test_{i:04}',gid,u,v,pose,width,length)))
    return specs


def run(args):
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new empty experiment output directory; existing outcomes must not precede prospective predictions')
    output.mkdir(parents=True,exist_ok=True)
    started=time.time()
    backend=getattr(args,'geometry_backend','native_smplx')
    cls=SurrogateAtlasLeg if backend=='atlas_surrogate' else AtlasLeg
    atlas=cls.load(args.asset_root)
    write_json(output/'atlas_audit.json',atlas.audit())
    cfg={'seed':args.seed,'train_objects':args.train_objects,'test_objects':args.test_objects,
         'rho':args.rho,'deadlines':np.arange(4.,36.01,.2).round(6).tolist(),
         'geometry_backend':backend,
         'shape_control_semantics':'native SMPL-X beta0 += (control1-1)*5, beta1 += (control2-1)*5; controls are not literal width/length ratios' if backend=='native_smplx' else 'geometric affine controls',
         'plant':'cartesian_velocity_and_orientation_rate_limited_surrogate',
         'pose_training':[0,30],'pose_holdout':[60],
         'shape_training':{'width':[.9,1.,1.1],'length':[.95,1.,1.05]},
         'shape_holdout':{'width':1.18,'length':.88},
         'task':'two distal geometric SA intervals followed by fresh proximal recheck',
         'acceptance':'independent geometry proxy; not B-mode quality',
         'formal_finite_sample_guarantee':False,'robot_execution':False}
    write_json(output/'frozen_protocol.json',cfg)
    records,pairs=generate_training(atlas,args.train_objects,args.seed,args.rho,output)
    models={name:build_bundle(records,pairs,keys) for name,keys in
            [('joint_material',MATERIAL_KEYS),('joint_anchor',ANCHOR_KEYS),('joint_world',WORLD_KEYS)]}
    models['marginal_ablation']=models['joint_material']
    for n in (8,16):
        if n<args.train_objects:
            ids={p['object_id'] for p in pairs[:n]}
            models[f'joint_material_n{n}']=build_bundle([r for r in records if r['object_id'] in ids],pairs[:n],MATERIAL_KEYS)
    specs=make_test_specs(args.test_objects,args.seed+991,args.rho)
    deadlines=np.asarray(cfg['deadlines'])
    # Freeze all test predictions on disk before any full test chain is executed.
    predictions=[];test_inputs=[]
    pred_path=output/'prospective_predictions.jsonl'
    with pred_path.open('w') as handle:
        for case,(group,spec) in enumerate(specs):
            world=LegWorld(atlas,spec)
            prep_state=world.new_state(.10)
            prep_rng=np.random.default_rng(args.seed+case+80000)
            near=capture_probe(world,prep_state,.10,.25,prep_rng,program_id='near_sa')
            _,y=world.evidence(near.packets)
            moved=move_to(world,prep_state,.60,prep_rng,mode=0)
            if not moved:raise RuntimeError('Common preparation could not establish the declared entry')
            # A shared settling controller establishes a repeatable entry. The
            # entire executed preparation is counted separately from request time.
            target,r=world.reference_at(.60,0)
            for _ in range(25):
                prep_state.position += .04*(target-prep_state.position)/.12
                prep_state.velocity*=.6
                prep_state.time+=.04
            test_inputs.append((group,spec,y,prep_state.time,near.accepted_frames,prep_state))
            row={'case':case,'object_id':spec.object_id,'stratum':group,'world_hash':world.digest,
                 'legal_y':y,'predictions':{}}
            for name,(model,pt) in models.items():
                curves=[]
                for pi,policy in enumerate(POLICIES):
                    curve,_,_=predict_fixed(model,pt,world,y,policy,deadlines,
                        np.random.default_rng(args.seed+case*100+pi),nmc=args.mc,
                        hide=name=='marginal_ablation')
                    curves.append(curve)
                curves=np.asarray(curves)
                # A finite declared strategy family. Nominal selection and
                # empirical mixture are not a certified robust confidence set.
                weights=np.eye(len(POLICIES))[np.argmax(curves,axis=0)]
                value=np.sum(weights*curves.T,axis=1)
                row['predictions'][name]={'probability':value.tolist(),'weights':weights.tolist(),
                                          'policy_curves':curves.tolist()}
            # Strong engineering reference: shortest public nominal path, no U/V.
            geometric=[sum(world.public_context(*span,m)['path_length'] for span in (SPAN_A,SPAN_B)) for m in MODES]
            default=int(np.argmin(geometric))
            row['predictions']['geometry_servo']={'weights':np.tile(np.eye(3)[default],(len(deadlines),1)).tolist()}
            row['predictions']['identify_then_execute']={'weights':np.tile(np.eye(3)[2],(len(deadlines),1)).tolist()}
            predictions.append(row);json_line(handle,row)
    prediction_sha=hashlib.sha256(pred_path.read_bytes()).hexdigest()
    write_json(output/'prediction_freeze.json',{'sha256':prediction_sha,'frozen_at_unix':time.time(),
                                             'before_any_complete_test_chain':True,
                                             'test_outcomes_used_for_fit':False})
    traces=[];evaluations=[]
    with (output/'complete_test_traces.jsonl').open('w') as handle:
        for case,(group,spec,y,prep_time,prep_frames,initial_state) in enumerate(test_inputs):
            outcomes=[]
            for pi,policy in enumerate(POLICIES):
                world=LegWorld(atlas,spec)
                trace=execute_fixed(world,y,policy,np.random.default_rng(args.seed+900000+case*10+pi),initial_state=initial_state)
                trace.update(case=case,stratum=group,object_id=spec.object_id,
                             preparation_seconds=prep_time,preparation_frames=prep_frames)
                traces.append(trace);outcomes.append(trace);json_line(handle,trace)
            observed=np.asarray([[float(t['success'] and t['duration']<=d) for t in outcomes] for d in deadlines])
            for method,pr in predictions[case]['predictions'].items():
                weights=np.asarray(pr['weights'])
                actual=np.sum(weights*observed,axis=1)
                prob=np.asarray(pr.get('probability',actual))
                evaluations.append({'case':case,'object_id':spec.object_id,'stratum':group,'method':method,
                    'observed_finish':actual.tolist(),'predicted_finish':prob.tolist(),
                    'interaction_seconds':np.dot(weights,np.asarray([t['duration'] for t in outcomes])).tolist(),
                    'new_frames':np.dot(weights,np.asarray([t['new_frames'] for t in outcomes])).tolist(),
                    'has_prediction':'probability' in pr})
            if (case+1)%8==0:print(f'complete held-out cases {case+1}/{len(test_inputs)}',flush=True)
    write_json(output/'evaluation.json',evaluations)
    # Phase-level association/time check uses actual measured executions, not a
    # hidden-class success rule. Its sample size is intentionally a pilot.
    four=[]
    for group in sorted({g for g,_ in specs}):
        for y in (-1,1):
            ids=[i for i,t in enumerate(test_inputs) if t[0]==group and t[2]==y]
            for mi in (0,1):
                tt=[traces[i*3+mi] for i in ids]
                four.append({'stratum':group,'y':y,'mode':MODES[mi],'n_objects':len(tt),
                             'finish':[float(np.mean([t['success'] and t['duration']<=d for t in tt])) if tt else None for d in deadlines]})
    write_json(output/'four_cell_measured.json',four)
    methods=sorted({e['method'] for e in evaluations})
    summaries=[]
    for group in sorted({e['stratum'] for e in evaluations}):
        for method in methods:
            ee=[e for e in evaluations if e['stratum']==group and e['method']==method]
            observed=np.mean([e['observed_finish'] for e in ee],axis=0)
            predicted=np.mean([e['predicted_finish'] for e in ee],axis=0)
            err=float(np.max(np.abs(observed-predicted))) if ee[0]['has_prediction'] else None
            summaries.append({'stratum':group,'method':method,'n_independent_objects':len(ee),
                              'mean_finish_curve':observed.tolist(),'mean_predicted_curve':predicted.tolist(),
                              'max_completion_curve_error':err})
    write_json(output/'summary.json',{'protocol':cfg,'models':{k:v[0].metadata for k,v in models.items()},
                                    'rows':summaries,'elapsed_wall_seconds':time.time()-started,
                                    'prediction_sha256':prediction_sha,
                                    'counterfactual_evaluation':'Every fixed policy is executed in a cloned persistent world. Frozen weights give an unbiased mixture-value estimate; these are not oracle-selected outcomes.'})
    print(json.dumps({'output':str(output),'wall_seconds':round(time.time()-started,2),
                      'local_records':len(records),'complete_chains':len(traces)},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--asset-root',default='/media/camp/EXT_DRIVE/Among_US')
    p.add_argument('--train-objects',type=int,default=32)
    p.add_argument('--test-objects',type=int,default=12)
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--rho',type=float,default=.8)
    p.add_argument('--geometry-backend',choices=['native_smplx','atlas_surrogate'],default='native_smplx')
    p.add_argument('--mc',type=int,default=256)
    run(p.parse_args())


if __name__=='__main__':main()

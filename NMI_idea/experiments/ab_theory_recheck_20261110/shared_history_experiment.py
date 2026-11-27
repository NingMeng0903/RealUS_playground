"""Matched-public-history continuous-anatomy module experiment.

Prediction/terminal evaluation frames are independent diagnostic forks. They
never enter the shared sensing history or the online posterior. Truth is used
only by rendering and retrospective scoring. No hardware or native replay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import traceback
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parent
LADDER=ROOT.parent/'ab_module_ladder_20261110'
sys.path.insert(0,str(LADDER/'planning'))
from system import env,MOTION,quality_arrays
from observation.servo import template_candidates
from inference.tempered_belief import TemperedBelief

METHODS=('fixed32','fixed128','tempered128')
PARAMS=('registration','modes','y_controls','z_controls','depths','identity_z','detect_z')
PREDICTION_BACKEND='numpy'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release(path,approved_sha):
    """Verify both the approval record and every byte covered by it."""
    if not path.exists() or approved_sha!=sha(path):
        raise ValueError('Locked test requires independently approved manifest')
    manifest=json.loads(path.read_text())
    required={str(Path(__file__).resolve()),str(ROOT/'STUDY_PROTOCOL.txt'),
              str(ROOT/'inference/continuous_smc.py'),str(ROOT/'inference/tempered_belief.py'),
              str(ROOT/'inference/gpu_predictor.py'),str(Path(env.__file__).resolve()),
              str(LADDER/'planning/system.py'),str(LADDER/'observation/servo.py')}
    covered={str((Path(filename) if Path(filename).is_absolute() else ROOT/filename).resolve())
             for filename in manifest['files']}
    if not required<=covered:
        raise ValueError(f'Frozen manifest omits required sources: {sorted(required-covered)}')
    for filename,expected in manifest['files'].items():
        item=Path(filename)
        if not item.is_absolute():item=ROOT/item
        if not item.is_file() or sha(item)!=expected:
            raise ValueError(f'Frozen source or dependency changed: {filename}')
    return manifest


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(np.floating,np.integer,np.bool_)):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value


def geometry_digest(world):
    return hashlib.sha256(np.ascontiguousarray(world.points).tobytes()).hexdigest()


def shifted(world):
    reg=world.registration.copy();reg[1]+=.008
    controls=world.y_controls.copy()+.10*env.CONTROL_X[None,:]
    return env._world(reg,world.modes.copy(),controls,world.z_controls.copy(),
                      world.depths.copy(),world.identity_z.copy(),world.detect_z.copy())


def public_anchor(blocks):
    """No particle state or audit/GT input is accepted by this function."""
    options=[]
    for block in blocks:
        visible=[f for f in block['public'] if f['visible'] and f['contact']]
        if not visible:continue
        score=float(np.median([f['identity_score'] for f in visible]))
        at=np.array(block['pose']);u=float(np.median([f['image_u_m'] for f in visible]))
        at[:2]+=u*np.array([np.cos(at[2]),np.sin(at[2])]);at[2]=0.
        options.append((score,len(visible),at))
    return env.clip_pose(max(options,key=lambda x:(x[0],x[1]))[2]) if options else np.zeros(3)


def generate_history(private_world,key):
    blocks=[];previous=np.zeros(3);clock=0.;policy_compute=0.
    poses=[np.array([0.,y,0.]) for y in [-.02,-.01,0.,.01,.02]]
    for i in range(8):
        if i==5:
            tick=time.perf_counter()
            anchor=public_anchor(blocks)
            poses.extend([anchor,env.clip_pose(anchor+[0.,.002,0.]),
                          env.clip_pose(anchor+[0.,-.002,0.])])
            policy_compute+=time.perf_counter()-tick
        pose=poses[i];duration=MOTION.duration(previous,pose);clock+=duration
        frames,audits=env.render_block(private_world,pose,(key,'sensing',i),start_time=clock)
        for j,(frame,audit) in enumerate(zip(frames,audits)):
            frame['frame_id']=audit['frame_id']=f'acq-{i}-frame-{j}'
        clock+=.5
        blocks.append(dict(pose=pose.copy(),public=frames,audit_for_evaluation_only=audits,
                           motion_s=duration,elapsed_s=clock,shared_policy_compute_s=policy_compute))
        previous=pose
    return blocks


def public_lines(history):
    """Strong public-only geometric baseline; weighted robust line estimates."""
    points=[];weights=[];poses=[]
    for frame,at in history:
        if not frame['visible'] or not frame['contact'] or frame['identity_score']<.70:continue
        c,s=np.cos(at[2]),np.sin(at[2]);u=frame['image_u_m']
        points.append([at[0]+c*u,at[1]+s*u,-frame['image_depth_m']])
        weights.append(max(.01,(frame['identity_score']-.5)**2))
        poses.append(tuple(np.round(at,7)))
    if len(points)<3:return []
    points=np.array(points);weights=np.array(weights)
    rng=np.random.default_rng(720905);proposals=[]
    for _ in range(128):
        i,j=rng.choice(len(points),2,replace=False)
        if poses[i]==poses[j]:continue
        d=points[j]-points[i];length=np.linalg.norm(d)
        if length<.006:continue
        d/=length
        dist=np.linalg.norm(np.cross(points-points[i],d),axis=1)
        mask=dist<=.003
        if mask.sum()<3:continue
        w=weights[mask];p=points[mask]
        center=np.average(p,axis=0,weights=w)
        cov=((p-center)*w[:,None]).T@(p-center)/w.sum()
        vals,vecs=np.linalg.eigh(cov);axis=vecs[:,-1]
        if axis[0]<0:axis=-axis
        span=float(np.ptp((p-center)@axis))
        score=float(w.sum()*min(span/.012,1.))
        target=env.clip_pose(np.array([center[0],center[1],np.arctan2(axis[1],axis[0])]))
        proposals.append((score,target))
    unique=[]
    for score,target in sorted(proposals,key=lambda z:-z[0]):
        if not any(np.linalg.norm((target-p)*[1,1,.03])<.002 for p in unique):unique.append(target)
        if len(unique)==2:break
    return unique


def choose_terminal(particles,weights,history):
    current=history[-1][1]
    poses=[current.copy()]+public_lines(history)+template_candidates(particles,weights,np.zeros(3))
    for frame,at in reversed(history):
        if frame['visible']:
            centered=at.copy();centered[:2]+=frame['image_u_m']*np.array([np.cos(at[2]),np.sin(at[2])])
            poses.append(env.clip_pose(centered));break
    poses=np.unique(np.round(np.array(poses),7),axis=0)
    pred=env.predict_batch(particles,poses);correct,public,_=quality_arrays(pred)
    utility=np.asarray(weights)@correct
    costs=np.array([MOTION.duration(current,p)+.5 for p in poses])
    values=60.-(60.-costs)*utility
    j=int(np.argmin(values))
    return poses[j],dict(candidates=poses,correct_public_stop_probability=utility,
                         public_stop_probability=np.asarray(weights)@public,
                         terminal_values=values)


def predictive_metrics(particles,weights,private_world,key):
    poses=np.array([[0.,y,yaw] for y in [-.02,0.,.02] for yaw in [-.4,0.,.4]])
    pred=env.predict_batch(particles,poses);logw=np.full(len(weights),-np.inf)
    positive=np.asarray(weights)>0;logw[positive]=np.log(np.asarray(weights)[positive])
    log_scores=[];zero=0;oracle=[]
    truth=env.predict_batch([private_world],poses)
    for a,pose in enumerate(poses):
        frames,_=env.render_block(private_world,pose,(key,'heldout_prediction',a),start_time=0.)
        for f in frames:
            score=float(logsumexp(logw+env.log_likelihood_from_prediction(f,pred,a)))
            if not np.isfinite(score):zero+=1
            log_scores.append(score if np.isfinite(score) else None)
            oracle.append(float(env.log_likelihood_from_prediction(f,truth,a)[0]))
    capped=[max(-1000.,s) if s is not None else -1000. for s in log_scores]
    return dict(query_count=len(poses),frame_count=len(log_scores),zero_predictive_support_frames=zero,
                mean_capped_NLL=float(-np.mean(capped)),log_density_floor_for_scoring_only=-1000.,
                oracle_mean_NLL=float(-np.mean(oracle)),log_scores=log_scores,
                heldout_frames_never_update_belief=True)


def terminal_evaluation(private_world,pose,key,prefix,clock):
    frames,audits=env.render_block(private_world,pose,(key,'terminal',prefix,tuple(pose)),start_time=clock)
    for i,(f,a) in enumerate(zip(frames,audits)):f['frame_id']=a['frame_id']=f'terminal-{prefix}-frame-{i}'
    public=env.public_stop(frames);correct=env.audit_stop(audits)
    return dict(correct_public_stop=bool(public and correct),wrong_public_stop=bool(public and not correct),
                public_stop=bool(public),audit_quality=bool(correct),public=frames,
                audit_for_evaluation_only=audits,diagnostic_fork_only=True)


def run_case(private_world,prior,key,outdir):
    shared=generate_history(private_world,key);history=[];arrays={};rows=[]
    for method in METHODS:
        n=32 if method=='fixed32' else 128
        belief=TemperedBelief(prior,seed=720906,prediction_backend=PREDICTION_BACKEND) if method=='tempered128' else env.Belief(prior[:n],support='finite')
        history=[];compute=0.;valid=True;steps=[]
        for i,block in enumerate(shared):
            tick=time.perf_counter()
            try:
                d=belief.update_block(block['public'],block['pose'])
                if method=='tempered128' and (d.get('unfinished') or not d.get('posterior_valid',False)):
                    steps.append(dict(index=i,status='rejected',diagnostics=d))
                    raise RuntimeError('temperature did not reach1; failed block cannot be omitted')
            except Exception as exc:
                compute+=time.perf_counter()-tick;valid=False
                steps.append(dict(index=i,status='failed',error=str(exc),traceback=traceback.format_exc()))
                break
            compute+=time.perf_counter()-tick
            history.extend((f,block['pose'].copy()) for f in block['public'])
            steps.append(dict(index=i,status='committed',diagnostics=d))
            if i+1 not in (5,8):continue
            prefix=i+1
            tick=time.perf_counter();pose,decision=choose_terminal(belief.particles,belief.weights,history)
            selection=time.perf_counter()-tick
            duration=MOTION.duration(block['pose'],pose)
            terminal=terminal_evaluation(private_world,pose,key,prefix,block['elapsed_s']+duration)
            terminal.update(method=method,prefix=prefix,valid_posterior=True,selected_pose=pose,
                decision=decision,inference_s=compute,selection_s=selection,
                shared_motion_observation_s=block['elapsed_s'],
                total_module_time_s=block['elapsed_s']+duration+.5+compute+selection+block['shared_policy_compute_s'],
                predictive=predictive_metrics(belief.particles,belief.weights,private_world,(key,prefix)),
                rawESS=float(1/np.square(belief.weights).sum()),max_weight=float(np.max(belief.weights)),
                posterior_history_frames=len(belief.history),steps=steps.copy())
            terminal['geometric_correct_public_stop']=terminal['correct_public_stop']
            terminal['decision_within_60s']=terminal['total_module_time_s']<=60.
            terminal['correct_public_stop']=bool(terminal['correct_public_stop'] and terminal['decision_within_60s'])
            rows.append(terminal)
            for field in PARAMS:arrays[f'{method}_{prefix}_{field}']=np.array([getattr(p,field) for p in belief.particles])
            arrays[f'{method}_{prefix}_weights']=belief.weights.copy()
        for prefix in (5,8):
            if not any(r['method']==method and r['prefix']==prefix for r in rows):
                rows.append(dict(method=method,prefix=prefix,valid_posterior=False,
                    correct_public_stop=False,wrong_public_stop=False,inference_s=compute,steps=steps,
                    failure='No completed full-history posterior for this prefix'))
    # Public geometry-only line fit gets the same history and terminal verifier.
    for prefix in (5,8):
        history=[(f,b['pose']) for b in shared[:prefix] for f in b['public']]
        tick=time.perf_counter();lines=public_lines(history);selection=time.perf_counter()-tick
        pose=lines[0] if lines else public_anchor(shared[:prefix])
        movement=MOTION.duration(shared[prefix-1]['pose'],pose)
        terminal=terminal_evaluation(private_world,pose,key,prefix,shared[prefix-1]['elapsed_s']+movement)
        terminal.update(method='public_line',prefix=prefix,valid_posterior=True,selected_pose=pose,
            inference_s=0.,selection_s=selection,total_module_time_s=shared[prefix-1]['elapsed_s']+movement+.5+selection+shared[prefix-1]['shared_policy_compute_s'],
            explicit_no_bayesian_model=True)
        terminal['geometric_correct_public_stop']=terminal['correct_public_stop']
        terminal['decision_within_60s']=terminal['total_module_time_s']<=60.
        terminal['correct_public_stop']=bool(terminal['correct_public_stop'] and terminal['decision_within_60s'])
        rows.append(terminal)
    filename=f"posterior_{key[0]}_{key[1]}.npz";np.savez_compressed(outdir/filename,**arrays)
    return dict(rows=rows,shared=shared,private_geometry_hash_for_audit_only=geometry_digest(private_world),
                posterior_parameters_file=filename)


def summaries(cases):
    output={}
    for case in cases:
        for r in case['rows']:
            group=f"{case['condition']}/{r['method']}/prefix{r['prefix']}"
            output.setdefault(group,[]).append(r)
    return {k:dict(n=len(rs),correct=sum(r['correct_public_stop'] for r in rs),
        wrong=sum(r['wrong_public_stop'] for r in rs),valid=sum(r['valid_posterior'] for r in rs),
        mean_inference_s=float(np.mean([r['inference_s'] for r in rs])),
        predictive_scored_n=sum('predictive' in r for r in rs),
        posterior_invalid_n=sum(not r['valid_posterior'] for r in rs),
        mean_predictive_NLL_valid_only=(float(np.mean([r['predictive']['mean_capped_NLL'] for r in rs if 'predictive' in r]))
                             if any('predictive' in r for r in rs) else None)) for k,rs in output.items()}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['dev','test'],default='dev')
    parser.add_argument('--objects',type=int,default=2);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approved-manifest-sha')
    parser.add_argument('--prediction-backend',choices=['numpy','cuda'],default='numpy');args=parser.parse_args()
    PREDICTION_BACKEND=args.prediction_backend
    release=ROOT/'FROZEN_STUDY_MANIFEST.json'
    if args.cohort=='test':
        try:verify_release(release,args.approved_manifest_sha)
        except (ValueError,KeyError) as exc:raise SystemExit(str(exc))
        args.objects=16
    args.output.mkdir(exist_ok=False,parents=True)
    prior=env.sample_prior(720901,128)
    worlds=env.sample_worlds(720902 if args.cohort=='dev' else 720913,args.objects)
    result=dict(scope='Matched-public-history L0 module; independent prediction/terminal forks; no complete online A/B or real ultrasound claim',
        cohort=args.cohort,code_sha256=sha(__file__),prediction_backend=PREDICTION_BACKEND,
        inference_source_sha256=sha(ROOT/'inference/continuous_smc.py'),
        initial_prior128_digest=hashlib.sha256(b''.join(p.points.tobytes() for p in prior)).hexdigest(),
        fixed32_is_first32_of_same128=True,cases=[],summary={})
    for h,world in enumerate(worlds):
        for condition,private in [('iid',world),('shifted',shifted(world))]:
            tick=time.perf_counter();case=run_case(private,prior,(h,condition,args.cohort),args.output)
            case.update(object_index_for_audit_only=h,condition=condition,wall_s=time.perf_counter()-tick)
            result['cases'].append(case);result['summary']=summaries(result['cases'])
            (args.output/'RESULTS.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
            print(args.cohort,h,condition,'wall',case['wall_s'],
                  [(r['method'],r['prefix'],r['valid_posterior'],r['correct_public_stop']) for r in case['rows']],flush=True)

"""Prospective A/B/C suite: local-only fitting, true release gates and recovery.

All requested policy mixtures and their primary-deadline random draws are saved
before any target full chain is run. Complete policy evaluations are paired
counterfactual simulator interventions, not clinical or hardware trials.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np

from .smplx_scene import SmplxAtlasLeg
from .simulation import Spec
from .core_execution import (CoreWorld,ProgramExecutor,local_bundle,TREES,LEVELS,
                             execute_tree,predict_tree,SPANS)
from .core_models import ObjectBundleModel,common_plan

def dump(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def line(handle,obj):
    handle.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');handle.flush()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()

def specs(n,seed,split):
    """Balanced controlled factorial, independent geometry seed per object."""
    r=np.random.default_rng(seed);out=[]
    for i in range(n):
        u,v=((-1,-1),(-1,1),(1,-1),(1,1))[i%4]
        if split=='train':
            pose=(0.,30.)[(i//4)%2];shape=((1.,1.),(.94,1.04),(1.08,.96))[(i//8)%3]
            group='train'
        else:
            group,pose,shape=(('known_pose',0.,(1.,1.)),('new_pose',60.,(1.,1.)),
                              ('new_shape',30.,(1.18,.88)),('new_pose_shape',60.,(1.18,.88)))[(i//4)%4]
        out.append((group,Spec(f'{split}_{seed}_{i:04}',seed*10000+i,u,v,pose,*shape)))
    return out

def population_rows(rows,manifest,rho):
    """Training sampling design may use audited labels; model inputs may not."""
    groups={pair:[] for pair in ((-1,-1),(-1,1),(1,-1),(1,1))}
    lookup={x['spec']['object_id']:x['spec'] for x in manifest}
    for row in rows:
        s=lookup[row['object_id']];groups[(s['u'],s['v'])].append(row)
    minimum=min(map(len,groups.values()))
    # Equal training count across correlation populations. Small factorial
    # pools round the requested rho; realized counts are recoverable in audit.
    per_sign=max(2,minimum)
    minor=max(1,round(per_sign*(1-abs(rho))/2))
    major=per_sign-minor
    selected=[]
    for (u,v),rr in groups.items():
        favoured=(u*v>0) if rho>=0 else (u*v<0)
        count=major if favoured else minor
        selected.extend(rr[:count])
    np.random.default_rng(411).shuffle(selected)
    return selected

def prepare(atlas,spec,level,seed):
    # Test-time old SA is legal preparation on this held-out object. For the
    # pose-change group it is acquired at 0deg, without refitting any model.
    from dataclasses import replace
    oldspec=replace(spec,knee_deg=0.)
    w=CoreWorld(atlas,oldspec,level)
    e=ProgramExecutor(w,w.new_state(.10)).set_rng(seed)
    result,packets=e.call('near')
    near=float(w.evidence(packets)[0][0])
    return near,{'object_id':spec.object_id,'near':near,'old_pose':0.,'current_pose':spec.knee_deg,
                 'old_world_hash':w.digest,'old_acquisition_seconds':result['duration'],
                 'old_acquisition_frames':result['frame_count'],'old_status':result['status'],
                 'packet_payload_sha256':hashlib.sha256(json.dumps({'near':near},sort_keys=True).encode()).hexdigest(),
                 'pose_change_cost_measured':False}

def model_definitions(rows,seed):
    models={
        'joint_robust':ObjectBundleModel(rows,mode='paired',seed=seed),
        'joint_nominal':ObjectBundleModel(rows,mode='paired',seed=seed),
        'full_history':ObjectBundleModel(rows,mode='full_history',seed=seed,bandwidth=.45),
        'shuffled_robust':ObjectBundleModel(rows,mode='shuffled',seed=seed),
        'independent':ObjectBundleModel(rows,mode='independent',seed=seed),
        'time_factorized':ObjectBundleModel(rows,mode='paired',seed=seed),
        'identify_execute':ObjectBundleModel(rows,mode='paired',seed=seed),
        'fixed_reference':ObjectBundleModel(rows,mode='paired',seed=seed),
    }
    for count in (8,16):
        if count<len(rows):models[f'joint_n{count}']=ObjectBundleModel(rows[:count],mode='paired',seed=seed)
    return models

def policy_predictions(world,model,method,near,arm,chain,deadlines,seed,n_models):
    hidden=arm['hidden'];release=arm['release']
    evidence={'near':near} if release==0 and not hidden else {}
    rows=model.rows
    # One object remains fixed through the whole predicted sequence. Ablation
    # explicitly breaks that dependence, and is labelled as such.
    values=np.zeros((len(rows),len(TREES),len(deadlines)))
    costs=np.zeros((len(rows),len(TREES),3))
    if not hasattr(world,'core_rollout_cache'):world.core_rollout_cache={}
    for i,bundle in enumerate(rows):
        for j,tree in enumerate(TREES):
            key=(bundle['object_id'],int(np.sign(bundle['near'])),tree.name,release,hidden,chain,
                 (seed,i,len(rows),method) if method in ('independent','time_factorized') else None)
            if key not in world.core_rollout_cache:
                conditional=[r for r in rows if np.sign(r['probe'])==np.sign(bundle['probe'])]
                o=predict_tree(world,bundle,tree,release=release,hidden=hidden,chain=chain,
                               independent_rows=conditional if method=='independent' else None,
                               time_rows=conditional if method=='time_factorized' else None,
                               rng=np.random.default_rng(seed+100*i+j))
                world.core_rollout_cache[key]=o
            o=world.core_rollout_cache[key]
            values[i,j]=o['success']&(o['duration']<=deadlines)
            costs[i,j]=[o['duration'],o['program_calls'],o['new_frames']]
    posterior=model.weights(evidence)
    nominal=np.einsum('n,nkt->kt',posterior,values)
    scenarios=model.candidate_weights(evidence,n_models=n_models,seed=seed).scenarios
    matrices=np.einsum('mn,nkt->mkt',scenarios,values)
    if method=='shuffled_robust':
        # Missing cross-program association is not resolved by bootstrapping
        # an arbitrary permutation. Retain two matched-marginal couplings,
        # including their different future delayed-evidence branches.
        for label,scenario in model.scenario_rows(evidence).items():
            if label=='observed':continue
            alternate=np.zeros_like(values)
            for i,bundle in enumerate(scenario['rows']):
                for j,tree in enumerate(TREES):
                    key=(bundle['object_id'],int(np.sign(bundle['near'])),tree.name,release,hidden,chain,None)
                    if key not in world.core_rollout_cache:
                        world.core_rollout_cache[key]=predict_tree(world,bundle,tree,release=release,hidden=hidden,chain=chain)
                    o=world.core_rollout_cache[key]
                    alternate[i,j]=o['success']&(o['duration']<=deadlines)
            matrices=np.concatenate((matrices,np.einsum('n,nkt->kt',scenario['weights'],alternate)[None]),axis=0)
    weights=[];lower=[];upper=[];pred=[]
    plan_cache={}
    for ti in range(len(deadlines)):
        expected_duration=posterior@np.minimum(costs[:,:,0],deadlines[ti])
        if method in ('joint_robust','shuffled_robust') or (method.startswith('joint_n') and method[7:].isdigit()):
            matrix=matrices[:,:,ti]
            key=(matrix.tobytes(),expected_duration.round(6).tobytes())
            if key not in plan_cache:plan_cache[key]=common_plan(matrix,costs=expected_duration)
            mix=np.asarray(plan_cache[key]['weights'])
        else:
            allowed=list(range(len(TREES)))
            if method=='identify_execute':allowed=[i for i,t in enumerate(TREES) if t.kind=='probe']
            elif method=='fixed_reference':allowed=[0,1]
            best=max(allowed,key=lambda k:(nominal[k,ti],-float(posterior@costs[:,k,0])))
            mix=np.eye(len(TREES))[best]
        weights.append(mix.tolist());pred.append(float(nominal[:,ti]@mix))
        lower.append(float(np.min(matrices[:,:,ti]@mix)));upper.append(float(np.max(matrices[:,:,ti])))
    return {'weights':weights,'predicted':pred,'lower':lower,'upper':upper,
            'envelope_kind':'empirical_bootstrap_and_coupling_scenarios_not_formal_confidence' if method=='shuffled_robust' else 'empirical_object_bootstrap_not_formal_confidence',
            'visible_evidence':evidence,'n_training_configurations':len(rows),
            'fixed_mode_curves':nominal[:2].tolist(),
            'predicted_costs':(np.einsum('n,nkc->kc',posterior,costs)).tolist()}

def run(args):
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use new empty output directory')
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    atlas=SmplxAtlasLeg.load(args.asset_root)
    deadlines=np.arange(6.,46.01,1.)
    # Delays are declared before target chains, shared across all objects.
    arms=[{'name':'retain_early','release':0.,'hidden':False},
          {'name':'release_3s','release':3.,'hidden':False},
          {'name':'release_8s','release':8.,'hidden':False},
          {'name':'hide_old','release':0.,'hidden':True}]
    levels=args.levels.split(',');rhos=(-.8,0.,.8)
    protocol={'schema':'v852_prospective_core_suite_v1','created':utc(),'seed':args.seed,
        'levels':{k:LEVELS[k] for k in levels},'training_pool':args.train_pool,'test_objects':args.test_objects,
        'population_rhos':list(rhos),'population_design':'balanced factorial test weighted by (1+rho*U*V); training stratified subsamples',
        'independent_unit':'unique complete anatomy configuration; reset/counterfactual runs are clustered',
        'deadlines':deadlines.tolist(),'arms':arms,'trees':[asdict(t) for t in TREES],
        'chains':[['a','f'],['a','b','f'],['b','a','f']],
        'complete_target_chains_in_training':0,'native_smplx':True,
        'feature_sensor':'finite-thickness geometric vessel slices, not B-mode',
        'execution_backend':'cartesian_lag; separate Genesis and finite-kernel reports',
        'real_contact_or_hardware':False,'formal_confidence_set':False,
        'notes':['post-request state starts at standardized public entry independent of old SA',
                 'test old SA at0deg, request pose may be60deg; no model refitting',
                 'body reposition cost before request is not measured',
                 'all mode/retry/return costs after request executed and counted',
                 'early/late/hide same historical content and same fresh obligations']}
    protocol['source_sha256']={name:sha(Path(__file__).parent/name) for name in
                               ('run_core_suite.py','core_execution.py','core_models.py','evidence_ledger.py')}
    dump(output/'protocol.json',protocol);dump(output/'atlas_audit.json',atlas.audit())
    all_models={};preparations={};testcases=[]
    # All local data and test preparation precede the forecast freeze. No full
    # target chain has been run at this point.
    reuse=Path(args.reuse_local) if args.reuse_local else None
    reused_rows=[];reused_manifest=[]
    if reuse:
        reused_rows=[json.loads(s) for s in (reuse/'local_bundles.jsonl').read_text().splitlines()]
        reused_manifest=[json.loads(s) for s in (reuse/'training_manifest.jsonl').read_text().splitlines()]
        protocol['local_reuse']={'source':str(reuse.resolve()),'raw_sha256':sha(reuse/'local_bundles.jsonl'),
            'reason':'single-program physical execution used only visible feedback; separate audit labels from public program feedback',
            'complete_chain_outcomes_reused':False,'new_test_configuration_seed':args.seed+100}
        dump(output/'protocol.json',protocol)
    with (output/'local_bundles.jsonl').open('w') as f,(output/'training_manifest.jsonl').open('w') as mf:
        for li,level in enumerate(levels):
            rows=[];manifest=[]
            if reuse:
                for raw in reused_rows:
                    if raw['level']!=level:continue
                    row=dict(raw)
                    for r in row['programs'].values():
                        r['independent_success']=bool(r.get('independent_success',r['success']))
                        r['success']=bool(r['public_success'] and r['independent_success'])
                    rows.append(row);line(f,row)
                manifest=[r for r in reused_manifest if r['level']==level]
                for audit in manifest:line(mf,audit)
                assert len(rows)==args.train_pool
                print(f'{level} reused audited local configurations {len(rows)}',flush=True)
            else:
                for i,(_,spec) in enumerate(specs(args.train_pool,args.seed+li*10,'train')):
                    row,audit=local_bundle(atlas,spec,level,args.seed+1000*li+i*20)
                    rows.append(row);manifest.append(audit);line(f,{'level':level,**row});line(mf,{'level':level,**audit})
                    if (i+1)%8==0:print(f'{level} local configurations {i+1}/{args.train_pool}',flush=True)
            for ri,rho in enumerate(rhos):
                selected=population_rows(rows,manifest,rho)
                models=model_definitions(selected,args.seed+ri)
                all_models[(level,rho)]=models
                dump(output/f'model_audit_{level}_{rho:+.1f}.json',{k:m.audit() for k,m in models.items()})
            for i,(group,spec) in enumerate(specs(args.test_objects,args.seed+100+li*10,'test')):
                near,prep=prepare(atlas,spec,level,args.seed+70000+li*100+i)
                case_id=f'{level}:{spec.object_id}'
                preparations[case_id]=prep
                chain=protocol['chains'][i%len(protocol['chains'])]
                testcases.append({'case_id':case_id,'level':level,'group':group,'spec':asdict(spec),'near':near,'chain':chain})
    dump(output/'test_manifest.json',testcases);dump(output/'preparations.json',preparations)
    predictions=[]
    with (output/'predictions.jsonl').open('w') as f:
        for ci,case in enumerate(testcases):
            spec=Spec(**case['spec']);level=case['level']
            # This world supplies public reference geometry only. Remove the
            # true curve so an accidental sensor call fails during prediction.
            public=CoreWorld(atlas,Spec('public_reference',0,1,1,spec.knee_deg,spec.width_scale,spec.length_scale),level)
            public.true_points=None
            for ri,rho in enumerate(rhos):
                for ai,arm in enumerate(arms):
                    for mi,(name,model) in enumerate(all_models[(level,rho)].items()):
                        p=policy_predictions(public,model,name,case['near'],arm,tuple(case['chain']),deadlines,
                                             args.seed+ci*100+ri,args.models)
                        # Primary deadline is declared, never selected using
                        # outcomes; complete CDFs assess the entire window.
                        ti=int(np.argmin(abs(deadlines-20.)))
                        draw=int(np.random.default_rng(args.seed+ci*10000+ri*1000+ai*100+mi).choice(len(TREES),p=p['weights'][ti]))
                        row={'case_id':case['case_id'],'level':level,'group':case['group'],'rho':rho,
                             'arm':arm['name'],'method':name,'chain':case['chain'],
                             'primary_deadline':float(deadlines[ti]),'primary_tree':draw,**p}
                        predictions.append(row);line(f,row)
            print(f'forecasts saved {ci+1}/{len(testcases)} {case["case_id"]}',flush=True)
    freeze={'created':utc(),'outcomes_exist':False,'sha256':{p.name:sha(p) for p in
             (output/'protocol.json',output/'local_bundles.jsonl',output/'predictions.jsonl',output/'test_manifest.json')},
             'n_predictions':len(predictions),'selection_precommitted':True}
    dump(output/'prediction_freeze.json',freeze)
    del all_models
    actual={}
    with (output/'executions.jsonl').open('w') as f:
        for ci,case in enumerate(testcases):
            w=CoreWorld(atlas,Spec(**case['spec']),case['level'])
            for ai,arm in enumerate(arms):
                runs=[]
                for j,tree in enumerate(TREES):
                    o=execute_tree(w,case['near'],tree,release=arm['release'],hidden=arm['hidden'],
                                   chain=tuple(case['chain']),seed=args.seed+90000+ci*100)
                    record={'case_id':case['case_id'],'level':case['level'],'group':case['group'],
                            'arm':arm['name'],'tree_index':j,**o}
                    line(f,record);runs.append(o)
                assert len({r['packet_hash'] for r in runs})==1
                actual[(case['case_id'],arm['name'])]=runs
            print(f'full-chain interventions {ci+1}/{len(testcases)}',flush=True)
    evaluations=[];deployments=[]
    cases={c['case_id']:c for c in testcases}
    for p in predictions:
        runs=actual[(p['case_id'],p['arm'])];mix=np.asarray(p['weights'])
        measured=np.asarray([[float(r['success'] and r['duration']<=T) for r in runs] for T in deadlines])
        curve=np.sum(mix*measured,axis=1)
        err=np.asarray(p['predicted'])-curve
        case=cases[p['case_id']];s=case['spec'];popweight=1+p['rho']*s['u']*s['v']
        ti=int(np.argmin(abs(deadlines-p['primary_deadline'])))
        cost_curve=[]
        for tt,T in enumerate(deadlines):
            bounded=[]
            for r in runs:
                calls=[c for c in r['calls'] if c['start_time']<T]
                bounded.append([min(r['duration'],T),len(calls),
                    sum(sum(t<=T for t in c['frame_capture_times']) for c in calls),
                    sum(bool(c.get('is_retry',False)) for c in calls)])
            cost_curve.append((mix[tt]@np.asarray(bounded)).tolist())
        cost=cost_curve[ti]
        row={k:p[k] for k in ('case_id','level','group','rho','arm','method','chain')}
        row.update(population_weight=popweight,predicted=p['predicted'],measured=curve.tolist(),
                   lower=p['lower'],upper=p['upper'],mean_abs_error=float(np.mean(abs(err))),
                   max_abs_error=float(np.max(abs(err))),primary_costs=cost,cost_curves=cost_curve,
                   retrospective_best_library=[float(max(r['success'] and r['duration']<=T for r in runs)) for T in deadlines])
        evaluations.append(row)
        picked=runs[p['primary_tree']]
        deployments.append({**{k:p[k] for k in ('case_id','rho','arm','method','primary_deadline','primary_tree')},
                            'selected_before_execution':True,'population_weight':popweight,
                            'success_by_deadline':picked['success'] and picked['duration']<=p['primary_deadline'],
                            'duration':picked['duration'],'new_frames':picked['new_frames'],
                            'interaction_seconds_by_deadline':min(picked['duration'],p['primary_deadline']),
                            'scope':'precommitted draw evaluated via paired simulator policy intervention'})
    dump(output/'evaluation.json',evaluations);dump(output/'precommitted_deployments.json',deployments)
    four=[]
    point_predictions={(p['case_id'],p['rho']):p for p in predictions
                       if p['method']=='joint_nominal' and p['arm']=='retain_early'}
    for level in levels:
        for rho in rhos:
            group_cases=[c for c in testcases if c['level']==level]
            cells={};predcells={};counts={}
            for y in (-1,1):
                cc=[c for c in group_cases if (1 if c['near']>=0 else -1)==y]
                for mi,mode in enumerate((-1,1)):
                    key=f'{y:+d},{mode:+d}';counts[key]=len(cc)
                    if not cc:cells[key]=predcells[key]=None;continue
                    ww=np.asarray([1+rho*c['spec']['u']*c['spec']['v'] for c in cc]);ww/=ww.sum()
                    observed=np.asarray([[float(actual[(c['case_id'],'retain_early')][mi]['success'] and
                                       actual[(c['case_id'],'retain_early')][mi]['duration']<=t)
                                       for t in deadlines] for c in cc])
                    predicted=np.asarray([point_predictions[(c['case_id'],rho)]['fixed_mode_curves'][mi] for c in cc])
                    cells[key]=(ww@observed).tolist();predcells[key]=(ww@predicted).tolist()
            valid=all(v is not None for v in cells.values())
            def gamma(d):return (np.asarray(d['+1,+1'])+np.asarray(d['-1,-1'])-np.asarray(d['+1,-1'])-np.asarray(d['-1,+1'])).tolist()
            four.append({'level':level,'rho':rho,'n_cases':counts,'measured':cells,'predicted':predcells,
                         'gamma_measured':gamma(cells) if valid else None,
                         'gamma_predicted':gamma(predcells) if valid else None,
                         'intervention':'fixed initial mode plus same allowed recovery; all failed entries included'})
    dump(output/'four_cells.json',four)
    assert freeze['sha256']['predictions.jsonl']==sha(output/'predictions.jsonl')
    for name,digest in protocol['source_sha256'].items():assert sha(Path(__file__).parent/name)==digest,'Source modified during frozen run'
    status={'finished':utc(),'wall_seconds':time.perf_counter()-started,'n_local_configurations':len(levels)*args.train_pool,
            'n_local_programs':len(levels)*args.train_pool*7,'n_test_configurations':len(testcases),
            'n_complete_chain_interventions':sum(len(v) for v in actual.values()),'n_precommitted_deployments':len(deployments),
            'predictions_unchanged':True,'real_hardware':False,'finite_kernel_report':'../v852_finite_limits_20261108',
            'joint_execution_report':'../v852_joint_execution_20261108'}
    dump(output/'run_status.json',status);print(json.dumps(status),flush=True)
    return status

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--asset-root',default='/media/camp/EXT_DRIVE/Among_US')
    p.add_argument('--train-pool',type=int,default=48);p.add_argument('--test-objects',type=int,default=16)
    p.add_argument('--levels',default='basic,variable,stress');p.add_argument('--models',type=int,default=9)
    p.add_argument('--seed',type=int,default=20261128)
    p.add_argument('--reuse-local',default=None,help='audited local-only source; never reuse complete target outcomes')
    args=p.parse_args()
    if args.train_pool%4 or args.test_objects%4:raise ValueError('Factorial object counts must be divisible by4')
    run(args)
if __name__=='__main__':main()

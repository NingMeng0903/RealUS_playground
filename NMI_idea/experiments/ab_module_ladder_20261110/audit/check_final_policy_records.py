"""Independent record/statistical audit; no policy, sensor, IK or native rerun."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FINITE=ROOT/'planning/test_frozen/results.json'
CONT=ROOT/'continuous_mismatch/formal_run_20261110/RESULTS.json'
ANALYSIS=ROOT/'planning/test_frozen/ANALYSIS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(FINITE)=='913be5af55c0d85b969e203635bdbb0979413ccfbd4e1192668bec9efd69ab85'
assert sha(CONT)=='6da6319338367651d235a9c82b245dedea987888484ee02e87bc87b976f4adbe'
finite=json.loads(FINITE.read_text());cont=json.loads(CONT.read_text())
analysis=json.loads(ANALYSIS.read_text())
protocol=json.loads((ROOT/'planning/PROTOCOL.json').read_text())
assert finite['protocol_sha256']==sha(ROOT/'planning/PROTOCOL.json')==analysis['protocol_sha256']
assert finite['system_sha256']==sha(ROOT/'planning/system.py')
assert analysis['source']==sha(FINITE)
fields={'frame_id','time_s','visible','contact','image_u_m','image_depth_m',
    'visible_length_m','image_axis_rad','identity_score'}
offsets=[0,.125,.25,.375,.5]

def true_frame(a):
    return bool(a['source_is_target'] and a['contact'] and a['geometric_length_m'] is not None
        and a['geometric_length_m']>=.015 and abs(a['true_u_m'])<=.002
        and a['out_of_plane_angle_rad']<=math.radians(10))

def check_rows(rows):
    blocks=frames=0;maxclockerr=0.;maxdecomp=0.;maxpriorerr=0.
    wrong=[];long_decoy_frames=0;qualified_decoy_singleframes=0
    tails=[];max_elapsed=0.;all_final_weights=[]
    for row in rows:
        assert row['actions']==len(row['events'])<=6
        prevclock=0.;prevweights=[1/32]*32
        for ordinal,e in enumerate(row['events']):
            assert e['ordinal']==ordinal
            assert e['q_state_kind']=='static endpoint realization, not native q_out'
            assert len(e['q_kin'])==8 and all(math.isfinite(x) for x in e['q_kin'])
            np.testing.assert_allclose(e['target'],e['actual_pose'],rtol=0,atol=1e-14)
            assert len(e['public'])==len(e['audit_for_evaluation_only'])==5
            assert all(abs(a-b)<1e-10 for a,b in zip(e['prior_before'],prevweights))
            maxpriorerr=max(maxpriorerr,max(abs(a-b) for a,b in zip(e['prior_before'],prevweights)))
            assert abs(sum(e['posterior'])-1)<1e-10 and all(0<=x<=1+1e-10 for x in e['posterior'])
            publicpass=True;auditpass=True
            acquisition=prevclock+e['planning_s']+e['endpoint_s']+e['motion_s']
            for i,(f,a) in enumerate(zip(e['public'],e['audit_for_evaluation_only'])):
                assert set(f)==fields
                assert f['frame_id']==a['frame_id']==f'acq-{ordinal}-frame-{i}'
                assert abs(f['time_s']-a['time_s'])<1e-10
                assert abs(f['time_s']-acquisition-offsets[i])<1e-8
                if not f['visible']:
                    assert all(f[k] is None for k in fields-{'frame_id','time_s','visible','contact'})
                public_good=bool(f['visible'] and f['contact'] and f['visible_length_m']>=.015
                    and abs(f['image_u_m'])<=.002 and f['identity_score']>=.75)
                real_good=true_frame(a)
                assert a['passed']==real_good
                publicpass&=public_good;auditpass&=real_good
                if f['visible'] and not a['source_is_target'] and f['visible_length_m']>=.015:
                    long_decoy_frames+=1
                    qualified_decoy_singleframes+=int(public_good)
                frames+=1
            assert e['public_stop']==publicpass and e['audit_correct']==auditpass
            assert e['timely_public_decision']==(e['clock_s']<=60+1e-12)
            expectedclock=acquisition+.5+e['belief_update_s']
            maxclockerr=max(maxclockerr,abs(expectedclock-e['clock_s']))
            assert abs(expectedclock-e['clock_s'])<1e-8
            if publicpass and e['timely_public_decision']:
                assert ordinal==len(row['events'])-1,'audit-enabled retry'
                assert row['public_stop']
                assert row['correct_public_stop']==auditpass
                assert row['false_public_stop']==(not auditpass)
                if not auditpass:
                    aas=e['audit_for_evaluation_only']
                    wrong.append(dict(object=row['object_index_for_audit_only'],context=row['context'],
                        method=row['method'],all_sources_target=all(a['source_is_target'] for a in aas),
                        max_true_axis_plane_deg=max(math.degrees(a['out_of_plane_angle_rad']) for a in aas),
                        max_abs_true_u_m=max(abs(a['true_u_m']) for a in aas)))
            prevclock=e['clock_s'];prevweights=e['posterior'];blocks+=1
        assert row['public_stop']==(row['correct_public_stop'] or row['false_public_stop'])
        assert not(row['correct_public_stop'] and row['false_public_stop'])
        assert row['rmt_s']==(row['elapsed_s'] if row['correct_public_stop'] else 60.)
        assert 0<=row['rmt_s']<=60.+1e-12
        if row['correct_public_stop']:assert row['elapsed_s']<=60.+1e-12
        logged=sum(e['planning_s']+e['endpoint_s']+e['motion_s']+.5+e['belief_update_s'] for e in row['events'])
        residual=row['elapsed_s']-logged
        tails.append(residual);maxdecomp=max(maxdecomp,abs(residual))
        max_elapsed=max(max_elapsed,row['elapsed_s'])
        all_final_weights.append(row['posterior'])
        if not row['public_stop']:assert len(row['events'])==6
    return dict(episodes=len(rows),blocks=blocks,frames=frames,complete_failure_partition=True,
        no_private_public_fields=True,no_audit_enabled_retry=True,
        max_event_clock_residual_s=maxclockerr,max_cost_decomposition_residual_s=maxdecomp,
        max_prior_chain_error=maxpriorerr,max_elapsed_s=max_elapsed,
        all_incomplete_episodes_reached_action_cap=True,wrong_stops=wrong,
        long_decoy_public_frames=long_decoy_frames,individually_qualified_decoy_frames=qualified_decoy_singleframes)

rows=finite['rows'];methods=protocol['methods'];ids=protocol['catalogue']['test_hidden_indices'];contexts=protocol['contexts']
by={(r['object_index_for_audit_only'],r['context'],r['method']):r for r in rows}
assert len(rows)==len(by)==len(ids)*len(contexts)*len(methods)==224
assert set(by)=={(i,c,m) for i in ids for c in contexts for m in methods}
fcheck=check_rows(rows)
summaries={};paired={};vectors={};maxstat=0.;maxci=0.
bootstrap=np.random.default_rng(720504).integers(0,len(ids),size=(2000,len(ids)))
metrics={'success':'correct_public_stop','wrong_stop':'false_public_stop','RMT_s':'rmt_s'}
for method in methods:
    selected=[by[h,c,method] for h in ids for c in contexts]
    summary=dict(n=len(selected),correct=sum(r['correct_public_stop'] for r in selected),
        wrong=sum(r['false_public_stop'] for r in selected),
        incomplete=sum(not r['public_stop'] for r in selected),
        mean_rmt_s=float(np.mean([r['rmt_s'] for r in selected])),
        mean_charged_planning_s=float(np.mean([sum(e['planning_s'] for e in r['events']) for r in selected])),
        endpoint_queries=sum(r['endpoint_queries'] for r in selected),
        invalid_endpoint_attempts=sum(r['invalid_endpoint_attempts'] for r in selected),
        positive_true_weight_logprob=[],zero_true_weight_count=0)
    for r in selected:
        p=r['posterior'][r['object_index_for_audit_only']]
        if p>0:summary['positive_true_weight_logprob'].append(math.log(p))
        else:summary['zero_true_weight_count']+=1
    summary['mean_true_h_logprob']=float(np.mean(summary.pop('positive_true_weight_logprob'))) if not summary['zero_true_weight_count'] else None
    summaries[method]=summary
    for metric,key in metrics.items():
        # Average both contexts inside an object before every bootstrap draw.
        x=np.array([sum(float(by[h,c,method][key]) for c in contexts)/len(contexts) for h in ids])
        vectors[method,metric]=x
        draws=x[bootstrap].mean(1)
        ci=np.quantile(draws,[.025,.975])
        claimed=analysis['methods'][method][metric]
        maxstat=max(maxstat,abs(x.mean()-claimed['mean']))
        maxci=max(maxci,float(np.max(abs(ci-claimed['object_clustered_95pct']))))
        assert abs(x.mean()-claimed['mean'])<1e-12
        assert np.max(abs(ci-claimed['object_clustered_95pct']))<1e-10
for method in methods:
    if method=='B_two_step':continue
    paired[method]={}
    for metric in metrics:
        d=vectors['B_two_step',metric]-vectors[method,metric]
        ci=np.quantile(d[bootstrap].mean(1),[.025,.975])
        claimed=analysis['paired'][method][metric]
        assert abs(d.mean()-claimed['B_minus_baseline'])<1e-12
        assert np.max(abs(ci-claimed['object_clustered_95pct']))<1e-10
        maxci=max(maxci,float(np.max(abs(ci-claimed['object_clustered_95pct']))))
        paired[method][metric]=dict(mean_difference=float(d.mean()),paired_cluster_ci=ci.tolist())

crows=cont['rows'];cb={(r['object_index_for_audit_only'],r['condition'],r['method']):r for r in crows}
assert len(crows)==len(cb)==32
assert set(cb)=={(i,c,m) for i in range(8) for c in ['iid','shifted'] for m in ['A_continuous_servo','B_two_step']}
assert all(r['status']=='completed' for r in crows)
ccheck=check_rows(crows)
assert all(not r['correct_public_stop'] and not r['false_public_stop'] and r['rmt_s']==60 for r in crows)
for r in crows:
    for e in r['events']:
        w=e['posterior'];d=e['belief_diagnostic']
        assert abs(1/sum(x*x for x in w)-d['raw_ess'])<1e-10
        assert max(w)==d['posterior_max_weight']
        assert sum(x>1e-12 for x in w)==d['unique_effective_support']
continuous_summary={}
for c in ['iid','shifted']:
    for m in ['A_continuous_servo','B_two_step']:
        selected=[cb[i,c,m] for i in range(8)]
        final_ess=[1/sum(x*x for x in r['posterior']) for r in selected]
        continuous_summary[f'{c}/{m}']=dict(n=8,correct=0,wrong=0,incomplete=8,
            final_ess=final_ess,ever_quality=sum(r['ever_audit_quality'] for r in selected),
            max_elapsed_s=max(r['elapsed_s'] for r in selected))

out=dict(passed=True,scope='Independent full-record and paired-statistical recalculation; no policy/native rerun',
    input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [FINITE,CONT,ANALYSIS,ROOT/'planning/PROTOCOL.json',ROOT/'planning/analyze_frozen.py']},
    finite_records=fcheck,finite_summary=summaries,paired=paired,
    independent_bootstrap=dict(seed=720504,resamples=2000,clusters=16,
        contexts_kept_within_object=True,max_summary_mean_error=maxstat,max_CI_endpoint_error=maxci),
    continuous_records=ccheck,continuous_summary=continuous_summary)
(HERE/'final_policy_record_checks.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,finite_records=fcheck,finite_summary=summaries,
    bootstrap=out['independent_bootstrap'],continuous_records=ccheck,continuous_summary=continuous_summary),indent=2))

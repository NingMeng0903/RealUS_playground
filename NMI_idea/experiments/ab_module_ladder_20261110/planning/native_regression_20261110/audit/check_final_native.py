"""Final saved-trace/layout audit; no native trajectory or IK-search rerun."""
from pathlib import Path
import json
import math
import sys
import numpy as np
import yaml
from check_native_regression import (HERE,REG,ROOT,REPAIR,sha,load,cfg,cfg_diff,
                                    CapabilityEvaluator,trace_check)

sys.path.insert(0,str(ROOT/'planning'))
from system import MOTION,body_transform,env
from observation.servo import template_candidates
import hashlib

BIN=Path('/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/native/wbc_rt/build_position_admission_20261110/wbc_rt')
CONFIG=Path('/media/camp/EXT_DRIVE/RealUS_playground/peirastic/configs/controller.yaml')
PROFILE=REPAIR/'qp_cross_track_fix/scan_tracking_profile.yaml'
INNER=Path('/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/native/wbc_rt/src/inner.cpp')

def main():
    source=ROOT/'planning/test_frozen/results.json'
    assert sha(source)=='913be5af55c0d85b969e203635bdbb0979413ccfbd4e1192668bec9efd69ab85'
    assert sha(CONFIG)=='71134e9e16a952e688085a996ab192b41886b4e12094949082b748f0f186748f'
    assert sha(INNER)=='28f8f7c302e70accba0c61b580cf0804e73f04c5e6d9b37e9a84acb0625b1ac9'
    raw=yaml.safe_load(CONFIG.read_text());profile=yaml.safe_load(PROFILE.read_text())
    assert profile['controller_overrides']['posture_policy']=='configured'
    assert profile['controller_overrides']['extra_posture_limits'] is False
    assert profile['controller_overrides']['native_bin']==str(BIN)
    before_default_binary=''.join(line for line in PROFILE.read_text().splitlines(True)
                                 if not line.startswith('  native_bin:'))
    replay_profile_hash=hashlib.sha256(before_default_binary.encode()).hexdigest()
    assert replay_profile_hash=='3123e193411c4b44e8dda5c798a42c968f96b402b32e8ab5c4eba4a09f66ff23'
    assert sha(Path(raw['inner']['native_bin']))=='cc34bf96b772ef463aa2a9468153e649da550192b8486d35ed52145ea07e7d07'
    new_binary_hash=sha(BIN)
    assert new_binary_hash!=sha(Path(raw['inner']['native_bin']))
    evaluator=CapabilityEvaluator()
    cohort=load(source)
    candidates=load(REG/'public_layout/CANDIDATES.json');selected=load(REG/'public_layout/SELECTED.json')
    assert candidates['cohort_sha256']==sha(source) and candidates['config_sha256']==sha(CONFIG)
    assert len(candidates['candidates'])==9
    nominal=np.deg2rad(raw['inner']['nullspace']['q_nominal_deg']);nominal[0]=.4
    np.testing.assert_array_equal(nominal,candidates['nominal_q'])
    paths=[];targets=[]
    for row in cohort['rows']:
        if row['method']!='B_two_step' or row['object_index_for_audit_only'] not in [0,1]:continue
        previous=np.zeros(3)
        for event in row['events']:
            target=np.array(event['target']);paths.extend(MOTION.path(previous,target));targets.append(target);previous=target
    targets=np.unique(np.round(targets,7),axis=0)
    assert len(targets)==8 and len(paths)==820
    rescored=[]
    for c in candidates['candidates']:
        q=np.array(c['q_seed']);anchor=evaluator.fk(q)
        np.testing.assert_allclose(anchor,c['T_world_body'],atol=1e-14,rtol=0)
        ck=evaluator.check_q(q);assert ck['valid']
        matrices=np.array([body_transform(p,anchor) for p in paths])
        margins=np.array(evaluator.score(matrices,q[0])['raw_calibrated_margin'])
        delta=(q-nominal)/np.r_[.1,np.repeat(.2,7)]
        score=float(margins.min()-.05*np.dot(delta,delta))
        assert abs(margins.min()-c['min_dird_margin'])<1e-10
        assert abs(score-c['score'])<1e-10
        assert c['found_count']==c['checked_endpoint_count']==8
        rescored.append(dict(id=c['id'],score=score,min_raw_calibrated_margin=float(margins.min())))
    assert max(rescored,key=lambda x:x['score'])['id']==selected['id']=='q2_+1'
    assert selected==next(c for c in candidates['candidates'] if c['id']==selected['id'])
    # Endpoint counts are read from the sealed static-search record. This audit
    # independently scores all 9x820 poses and checks all 9 initial q; it does
    # not claim a new 72-endpoint finite-SRS search or a global layout optimum.
    prior=load(REG/'public_prior_layout/CANDIDATES.json')
    prior_selected=load(REG/'public_prior_layout/SELECTED.json')
    assert prior['basis']=='prior' and prior['cohort_sha256'] is None and prior['prior_seed']==720502
    assert prior['config_sha256']==sha(CONFIG)
    particles=env.sample_prior(720502,32);weights=np.ones(32)/32
    prior_paths=[];prior_targets=[]
    for x in [-.025,0.,.025]:
        for base in template_candidates(particles,weights,np.array([x,0.,0.])):
            for yaw in [-.06,0.,.06]:
                target=base.copy();target[2]+=yaw;target=env.clip_pose(target)
                prior_paths.extend(MOTION.path(np.zeros(3),target));prior_targets.append(target)
    assert len(np.unique(np.round(prior_targets,7),axis=0))==36
    prior_rescored=[]
    for c in prior['candidates']:
        q=np.array(c['q_seed']);anchor=evaluator.fk(q)
        matrices=np.array([body_transform(p,anchor) for p in prior_paths])
        margins=np.array(evaluator.score(matrices,q[0])['raw_calibrated_margin'])
        delta=(q-nominal)/np.r_[.1,np.repeat(.2,7)]
        score=float(margins.min()-.05*np.dot(delta,delta))
        assert abs(margins.min()-c['min_dird_margin'])<1e-10 and abs(score-c['score'])<1e-10
        assert c['found_count']==c['checked_endpoint_count']==36
        prior_rescored.append(dict(id=c['id'],score=score,min_raw_calibrated_margin=float(margins.min())))
    assert max(prior_rescored,key=lambda x:x['score'])['id']==prior_selected['id']==selected['id']
    np.testing.assert_array_equal(prior_selected['q_seed'],selected['q_seed'])
    np.testing.assert_array_equal(prior_selected['T_world_body'],selected['T_world_body'])

    outdir=REG/'layout20_preserved_preferences'
    data=load(outdir/'RESULTS.json')
    assert 'in_progress' not in data and len(data['episodes'])==4
    assert data['execution_layout_override']==selected
    assert data['policy_results_sha256']==sha(source)
    assert data['runner_sha256']==sha(ROOT/'planning/run_native_subset.py')
    results=[];total=0;allticks=0;sequences=[]
    for ep in data['episodes']:
        expected=next(r for r in cohort['rows'] if r['method']=='B_two_step' and r['context']==ep['context']
            and r['object_index_for_audit_only']==ep['object_index_for_audit_only'])
        assert ep['all_frozen_commands_replayed'] and len(ep['segments'])==len(expected['events'])
        assert ep['execution_context']==selected['id']
        q=np.array(selected['q_seed']);v=np.zeros(8);a=np.zeros(8);events=[];lasttime=None
        previous=np.zeros(3);segment_summaries=[];command_sequence=[]
        for s,e in zip(ep['segments'],expected['events']):
            assert s['status']=='completed' and s['ordinal']==e['ordinal'] and s['target']==e['target']
            command_sequence.append(s['target'])
            np.testing.assert_array_equal(s['q_actual_entry'],q)
            desired=np.array([body_transform(p,np.array(selected['T_world_body'])) for p in MOTION.path(previous,np.array(e['target']))])
            np.testing.assert_allclose(s['desired_DMP_poses'],desired,rtol=0,atol=1e-14)
            n=s['result']['native'];t=s['same_time_measured_tracking']
            assert not n['failures'] and n['exit_state_valid']
            np.testing.assert_array_equal(n['qdot_entry'],v)
            np.testing.assert_allclose(n['acceleration_entry'],a,rtol=0,atol=1e-12)
            checked,z=trace_check(evaluator,outdir/s['trace_file'],n,(t['max_position_m'],t['max_rotation_rad']))
            assert checked['pass_2mm_2deg']==t['pass_2mm_2deg']
            np.testing.assert_array_equal(z['q_meas_before'][0],q)
            if lasttime is not None:assert abs(z['native_model_t_mono'][0]-lasttime-.005)<2e-12
            lasttime=z['native_model_t_mono'][-1]
            q=z['q'][-1];v=z['qdot'][-1];a=z['command_acceleration'][-1]
            np.testing.assert_array_equal(s['result']['q_out'],q)
            state=n['exit_state'];assert state['valid'] and state['schema']=='ultrapoc_qp_complete_history_v2'
            assert state['events'][:-1]==events
            events=state['events'];assert len(events)==s['ordinal']+1
            np.testing.assert_array_equal(state['q_exit'],q)
            np.testing.assert_array_equal(state['qdot_exit'],v)
            np.testing.assert_allclose(state['acceleration_exit'],a,rtol=0,atol=1e-12)
            assert max(n['prefix_reconstruction_max_abs_error'].values())<=1e-7
            fp=state['fingerprint'];assert fp['native_binary_sha256']==new_binary_hash
            assert fp['config_sha256']==sha(CONFIG) and fp['scan_profile_sha256']==replay_profile_hash
            previous=np.array(e['target']);total+=1;allticks+=checked['samples']
            segment_summaries.append(dict(ordinal=s['ordinal'],**checked))
        sequences.append(command_sequence)
        results.append(dict(original_object_label=ep['object_index_for_audit_only'],original_context_label=ep['context'],
            actual_execution_context=ep['execution_context'],segments=segment_summaries,full_state_prefix_verified=True))
    assert total==20
    segments=[x for ep in results for x in ep['segments']]
    # Original context labels index command sequences; all four now share one
    # execution placement/start. Count unique command sequences explicitly.
    unique_sequences=len({json.dumps(s) for s in sequences})
    old=load(REG/'fixed_binary_old_paths/RESULTS.json')
    old_checks=[]
    for row in old['cases']:
        assert row['name'] in ['old_path0','old_path1']
        checked,_=trace_check(evaluator,REG/'fixed_binary_old_paths'/(row['name']+'_trace.npz'),row['result'],
            (row['same_time_max_position_m'],row['same_time_max_rotation_rad']))
        assert row['profile']['native_binary_sha256']==new_binary_hash
        before=cfg(REG/'baseline_v2'/(row['name']+'_native_config.txt'))
        after=cfg(REG/'fixed_binary_old_paths'/(row['name']+'_native_config.txt'))
        difference=cfg_diff(before,after)
        assert difference=={'qp.branch_barrier.enabled':['1','0']}
        for k in ['qp.sigma_setbased.enabled','qp.joint_comfort.enabled','qp.j4_design_comfort.enabled','arm_angle.enabled','psi_retarget.enabled']:
            assert after[k]==before[k]=='1'
        old_checks.append(dict(name=row['name'],effective_config_changes=difference,**checked))
    assert len(old_checks)==2
    out=dict(passed=True,scope='Read-only independent full FK, command/state, v/a, collision-subset and public-layout rescore; no native trajectory or IK-search rerun',
        source_hashes={str(p):sha(p) for p in [source,CONFIG,PROFILE,INNER,BIN,REG/'public_layout/CANDIDATES.json',REG/'public_layout/SELECTED.json',REG/'public_prior_layout/CANDIDATES.json',REG/'public_prior_layout/SELECTED.json',REG/'select_public_layout.py',outdir/'RESULTS.json',REG/'fixed_binary_old_paths/RESULTS.json']},
        layout=dict(selected_id=selected['id'],candidate_count=9,rescored_path_poses_per_candidate=len(paths),
            unique_endpoints=8,score_reproduction=rescored,scope='Posthoc public recorded-command set; not prospective unknown-anatomy placement or global optimum'),
        prior_only_layout=dict(selected_id=prior_selected['id'],candidate_count=9,
            independent_public_prior_seed=720502,unique_endpoints=36,
            rescored_path_poses_per_candidate=len(prior_paths),scores=prior_rescored,
            selected_q_and_placement_bit_identical=True,
            scope='Separate public-prior-only selector has no completed-command input; retrospective verification, finite candidate comparison, no global optimum claim'),
        final_replay=dict(planned_segments=20,completed_segments=total,ticks=allticks,
            complete_tracking_pass=sum(x['complete_execution_tracking_pass'] for x in segments),
            max_position_m=max(x['max_position_m'] for x in segments),
            max_rotation_rad=max(x['max_rotation_rad'] for x in segments),
            min_velocity_margin=min(x['velocity_margin_min'] for x in segments),
            min_acceleration_margin=min(x['acceleration_margin_min'] for x in segments),
            min_recorded_collision_m=min(x['minimum_saved_collision_m'] for x in segments),
            min_proposed_urdf_margin=min(x['minimum_proposed_urdf_margin'] for x in segments),
            min_jerk_margin_record_only=min(x['jerk_margin_min_record_only'] for x in segments),
            unique_execution_placements=1,unique_recorded_command_sequences=unique_sequences,
            original_four_start_conditions_restored=False),
        old_path_regression=old_checks,episodes=results)
    (HERE/'FINAL_NATIVE_REPRODUCTION.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['episodes','old_path_regression','source_hashes']},indent=2))

if __name__=='__main__':main()

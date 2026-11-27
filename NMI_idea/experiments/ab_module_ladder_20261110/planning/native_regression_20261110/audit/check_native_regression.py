"""Independent saved-trace audit. Never creates a native controller or actor."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
REG=HERE.parent
ROOT=REG.parents[1]
REPAIR=ROOT.parent/'ab_validation_20261110/repair_20261110'
sys.path.insert(0,str(ROOT/'capability'))
from capability import CapabilityEvaluator

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text())
def cfg(p):
    out={}
    for line in Path(p).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key,*value=line.split(); assert key not in out
            out[key]=' '.join(value)
    return out
def cfg_diff(a,b):
    return {k:[a.get(k),b.get(k)] for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}

def trace_check(evaluator,path,result,expected_tracking=None,allow_rejected=False):
    z=np.load(path,allow_pickle=False)
    q=z['q_meas_before'];send=z['q_send_command'];v=z['qdot'];n=len(q);dt=.005
    accepted=z['q']
    rejected=np.any(accepted!=send,axis=1)
    if allow_rejected and result['failures']:
        assert np.flatnonzero(rejected).tolist()==[n-1]
        assert np.array_equal(accepted[-1],q[-1])
        assert result['failures'][0]['rejected_before_ideal_plant_integration']
        assert not result['exit_state_valid']
    else:
        assert np.array_equal(accepted,send)
        assert not result['failures']
    assert np.array_equal(q[1:],accepted[:-1])
    np.testing.assert_allclose(np.diff(z['t_s']),dt,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.diff(z['native_model_t_mono']),dt,rtol=0,atol=2e-12)
    actual=np.array([evaluator.fk(x) for x in q])
    ref=z['pose_desired']
    pe=np.linalg.norm(actual[:,:3,3]-ref[:,:3,3],axis=1)
    re=Rotation.from_matrix(ref[:,:3,:3].transpose(0,2,1)@actual[:,:3,:3]).magnitude()
    if expected_tracking:
        assert abs(pe.max()-expected_tracking[0])<1e-12
        assert abs(re.max()-expected_tracking[1])<1e-12
    assert len(q)==result['n_steps']
    assert np.all(z['qp1_status']=='solved')
    assert not np.any(z['task_paused']) and not np.any(z['solver_fault_latched'])
    ve=np.array(result['qdot_entry']);ae=np.array(result['acceleration_entry'])
    acceleration=np.diff(np.vstack([ve,v]),axis=0)/dt
    jerk=np.diff(np.vstack([ae,acceleration]),axis=0)/dt
    np.testing.assert_allclose(acceleration,z['command_acceleration'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(jerk,z['command_jerk'],atol=1e-10,rtol=0)
    vl=np.array(result['velocity_limit_by_coordinate']);al=np.array(result['acceleration_limit_by_coordinate'])
    jl=np.array(result['jerk_limit_by_coordinate'])
    vm=float(np.min(vl-abs(v)));am=float(np.min(al-abs(acceleration)))
    jm=float(np.min(jl-abs(jerk)))
    assert abs(vm-result['velocity_margin_min'])<1e-12
    assert abs(am-result['acceleration_margin_min'])<1e-10
    assert abs(jm-result['jerk_margin_min'])<1e-8
    # The controller applies the unchanged declared rail hard range [.005,.78]
    # to its model limits. The static capability model retains the URDF [0,.8].
    physical_cfg=cfg(REG/'baseline_v2/new_first_nominal_native_config.txt')
    lower=evaluator.kin.q_lower.copy();upper=evaluator.kin.q_upper.copy()
    lower[0]=max(lower[0],float(physical_cfg['rail.hard_min_m']))
    upper[0]=min(upper[0],float(physical_cfg['rail.hard_max_m']))
    position_margin=np.minimum(accepted-lower,upper-accepted)
    np.testing.assert_allclose(position_margin.min(0),result['joint_limit_min_margin_by_coordinate'],atol=1e-12,rtol=0)
    assert position_margin.min()>=-1e-8
    assert vm>=-1e-5 and am>=-1e-4
    distances=z['min_self_distance_m']
    assert abs(distances.min()-result['minimum_self_collision_distance_m'])<1e-12
    assert distances.min()>=.01
    inds=sorted(set([0,n//2,n-1,int(pe.argmax()),int(distances.argmin()),
        int(np.unravel_index(abs(acceleration).argmax(),acceleration.shape)[0])]))
    collision=[]
    for i in inds:
        ck=evaluator.check_q(accepted[i])
        assert ck['pair_count']==35 and ck['valid']
        err=abs(ck['min_self_collision_distance_m']-distances[i])
        assert err<1e-10
        collision.append(dict(tick=i,minimum_m=ck['min_self_collision_distance_m'],saved_error_m=err))
    qv=(accepted-q)/dt
    out=dict(trace_sha256=sha(path),samples=n,max_position_m=float(pe.max()),
        max_rotation_rad=float(re.max()),position_peak_tick=int(pe.argmax()),
        rotation_peak_tick=int(re.argmax()),pass_2mm_2deg=bool(pe.max()<=.002 and re.max()<=math.radians(2)),
        prefix_tracking_only=bool(result['failures']),
        complete_execution_tracking_pass=bool(not result['failures'] and pe.max()<=.002 and re.max()<=math.radians(2)),
        rejected_command_count=int(np.sum(rejected)),
        minimum_proposed_urdf_margin=float(np.minimum(send-evaluator.kin.q_lower,evaluator.kin.q_upper-send).min()),
        max_post_vs_pre_time_misalignment_m=float(np.max(np.linalg.norm(z['pose_actual'][:,:3,3]-actual[:,:3,3],axis=1))),
        velocity_margin_min=vm,acceleration_margin_min=am,jerk_margin_min_record_only=jm,
        joint_position_margin_min=float(position_margin.min()),
        minimum_saved_collision_m=float(distances.min()),collision_fresh_subset=collision,
        qsend_measured_chain_exact=True,all_qp_solved=True,pause_count=0,
        model_dt_min=float(np.diff(z['native_model_t_mono']).min()),
        model_dt_max=float(np.diff(z['native_model_t_mono']).max()),
        qsend_fd_velocity_margin_min=float(np.min(vl-abs(qv))),
        qsend_fd_vs_published_qdot_max_abs=float(np.max(abs(qv-v))),
        q1_range_rad=[float(q[:,1].min()),float(q[:,1].max())],
        q1_velocity_min=float(v[:,1].min()),q1_box_lower_min=float(z['box_lo'][:,1].min()),
        q1_box_lower_max=float(z['box_lo'][:,1].max()))
    return out,z

def causal(evaluator):
    seals={};reports={}
    baseline=load(REG/'baseline_v2/RESULTS.json')
    guard=load(REG/'j1_guard_only/RESULTS.json')
    two=load(REG/'causal_2x2/RESULTS.json')
    for group,data in [('baseline_v2',baseline),('j1_guard_only',guard),('causal_2x2',two)]:
        seals[group+'/RESULTS.json']=sha(REG/group/'RESULTS.json')
        reports[group]=[]
        for row in data['cases']:
            name=row.get('name',row.get('case'))
            if 'variant' in row:name=row['variant']+'_'+name
            path=REG/group/(name+'_trace.npz')
            assert sha(path)==row['trace_sha256']
            expected=(row.get('same_time_max_position_m',row.get('max_position_m')),
                      row.get('same_time_max_rotation_rad',row.get('max_rotation_rad')))
            checked,_=trace_check(evaluator,path,row['result'],expected)
            reports[group].append(dict(name=name,**checked))
    old_cfg=cfg(REG/'baseline_v2/new_first_nominal_native_config.txt')
    new_cfg=cfg(REG/'j1_guard_only/motion_native_config.txt')
    differences=cfg_diff(old_cfg,new_cfg)
    assert set(differences)=={'qp.branch_barrier.j1_overfold_abs_rad'}
    assert cfg_diff(cfg(REG/'baseline_v2/hold_nominal_10s_native_config.txt'),
                    cfg(REG/'j1_guard_only/hold10s_native_config.txt'))==differences
    assert float(new_cfg['qp.branch_barrier.j1_overfold_abs_rad'])==min(abs(evaluator.kin.q_lower[1]),abs(evaluator.kin.q_upper[1]))
    for oldname,newname in [('new_first_nominal','motion'),('hold_nominal_10s','hold10s')]:
        a=np.load(REG/'baseline_v2'/f'{oldname}_trace.npz')
        b=np.load(REG/'j1_guard_only'/f'{newname}_trace.npz')
        assert np.array_equal(a['q_meas_before'][0],b['q_meas_before'][0])
        for key in ['pose_desired','t_s','native_model_t_mono']:
            assert np.array_equal(a[key],b[key]),key
    q0=np.load(REG/'baseline_v2/new_first_nominal_trace.npz')['q_meas_before'][0]
    cap=float(old_cfg['qp.branch_barrier.j1_overfold_abs_rad'])
    activation=float(old_cfg['qp.branch_barrier.j1_overfold_activate_rad'])
    eps=float(old_cfg['qp.branch_barrier.j1_overfold_eps_rad'])
    gain=float(np.clip((cap-abs(q0[1])-eps)/(activation-eps),0,1))
    assert q0[1]<-cap and gain==0 and q0[1]>evaluator.kin.q_lower[1]
    snapshot=load(REG/'production_before_fix/MANIFEST.json')
    for original,digest in snapshot.items():
        assert sha(REG/'production_before_fix'/Path(original).name)==digest
    return dict(passed=True,scope='Offline immutable diagnostic traces; no native rerun',
        source_results_sha256=seals,effective_one_factor_config_diff=differences,
        same_initial_q_reference_and_model_time=True,
        j1_wall=dict(q0_rad=float(q0[1]),q0_deg=float(np.rad2deg(q0[1])),
            old_wall_rad=cap,physical_lower_rad=float(evaluator.kin.q_lower[1]),
            physical_upper_rad=float(evaluator.kin.q_upper[1]),old_overfold_d= gain),
        archived_source_hashes_verified=True,reproductions=reports)

def replay(evaluator,dirname):
    directory=REG/dirname;data=load(directory/'RESULTS.json')
    assert 'in_progress' not in data and len(data['episodes'])==4
    assert data['policy_results_sha256']==sha(ROOT/'planning/test_frozen/results.json')
    policy=load(ROOT/'planning/test_frozen/results.json')
    episodes=[];pass_count=ticks=0;complete_count=attempted=0
    for episode in data['episodes']:
        h,c=episode['object_index_for_audit_only'],episode['context']
        wanted=next(r for r in policy['rows'] if r['object_index_for_audit_only']==h and r['context']==c and r['method']=='B_two_step')
        scene=next(s for s in policy['contexts'] if s['id']==c)
        assert episode['all_frozen_commands_replayed']==(len(episode['segments'])==len(wanted['events']))
        assert len(episode['segments'])<=len(wanted['events'])
        q=np.array(scene['q_seed']);v=np.zeros(8);a=np.zeros(8);lasttime=None;events=[]
        details=[]
        for segment,w in zip(episode['segments'],wanted['events']):
            assert segment['ordinal']==w['ordinal'] and segment['target']==w['target']
            assert segment['status']=='completed'
            np.testing.assert_array_equal(segment['q_actual_entry'],q)
            result=segment['result']['native']
            np.testing.assert_array_equal(result['qdot_entry'],v)
            np.testing.assert_allclose(result['acceleration_entry'],a,rtol=0,atol=1e-12)
            tracked=segment['same_time_measured_tracking']
            checked,z=trace_check(evaluator,directory/segment['trace_file'],result,
                (tracked['max_position_m'],tracked['max_rotation_rad']),allow_rejected=True)
            assert checked['pass_2mm_2deg']==tracked['pass_2mm_2deg']
            np.testing.assert_array_equal(z['q_meas_before'][0],q)
            if lasttime is not None:assert abs(z['native_model_t_mono'][0]-lasttime-.005)<2e-12
            lasttime=z['native_model_t_mono'][-1]
            q=z['q'][-1]
            last_valid=-2 if result['failures'] else -1
            v=z['qdot'][last_valid];a=z['command_acceleration'][last_valid]
            np.testing.assert_array_equal(segment['result']['q_out'],q)
            state=result['exit_state']
            assert state['schema']=='ultrapoc_qp_complete_history_v2'
            assert state['valid']==(not result['failures'])
            assert state['events'][:-1]==events
            events=state['events']
            assert len(events)==segment['ordinal']+1
            assert max(result['prefix_reconstruction_max_abs_error'].values())<=1e-7
            np.testing.assert_array_equal(state['q_exit'],q)
            np.testing.assert_array_equal(state['qdot_exit'],v)
            np.testing.assert_allclose(state['acceleration_exit'],a,rtol=0,atol=1e-12)
            assert len(state['fingerprint']['native_binary_sha256'])==64
            details.append(dict(ordinal=segment['ordinal'],**checked))
            pass_count+=checked['complete_execution_tracking_pass'];ticks+=checked['samples']
            complete_count+=not result['failures'];attempted+=1
            if result['failures']:assert segment is episode['segments'][-1]
        episodes.append(dict(object=h,context=c,segments=details,prefix_chain_verified=True,
            planned_commands=len(wanted['events']),attempted_commands=len(episode['segments'])))
    allsegments=[s for e in episodes for s in e['segments']]
    assert sum(e['planned_commands'] for e in episodes)==20
    return dict(passed=True,scope='Complete frozen command and derivative/state audit, all same-time FK, selected fresh collision checks; no native rerun',
        directory=dirname,result_sha256=sha(directory/'RESULTS.json'),planned_segments=20,
        attempted_segments=attempted,completed_without_rejection=complete_count,
        rejected_segments=attempted-complete_count,unattempted_segments=20-attempted,ticks=ticks,
        tracking_pass_segments=pass_count,max_position_m=max(s['max_position_m'] for s in allsegments),
        max_rotation_rad=max(s['max_rotation_rad'] for s in allsegments),
        minimum_velocity_margin=min(s['velocity_margin_min'] for s in allsegments),
        minimum_acceleration_margin=min(s['acceleration_margin_min'] for s in allsegments),
        minimum_recorded_collision_m=min(s['minimum_saved_collision_m'] for s in allsegments),
        minimum_jerk_margin_record_only=min(s['jerk_margin_min_record_only'] for s in allsegments),
        episodes=episodes)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay-dir');args=parser.parse_args()
    evaluator=CapabilityEvaluator()
    out=replay(evaluator,args.replay_dir) if args.replay_dir else causal(evaluator)
    name=(args.replay_dir+'_audit.json') if args.replay_dir else 'causal_trace_audit.json'
    (HERE/name).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['reproductions','episodes']},indent=2))

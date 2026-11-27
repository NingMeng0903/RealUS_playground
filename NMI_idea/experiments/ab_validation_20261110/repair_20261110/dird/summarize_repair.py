"""Assemble immutable measured results and current hashes; no tuning/training."""
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from common import HERE, BASELINE, SOURCE, ROBOT, save, sha256


def load(p):return json.loads(Path(p).read_text())


def qp_summary(folder):
    rows=load(HERE/folder/'results.json')
    legal=[r for r in rows if not r['failures'] and min(r['joint_limit_min_margin_by_coordinate'])>=-1e-8 and r['velocity_margin_min']>=-1e-5 and r['minimum_self_collision_distance_m']>=.01]
    return {'n':len(rows),'raw_rows':rows,'command_position_velocity_and_self_collision_legal_n':len(legal),
        'controller_fault_n':sum(bool(r['failures']) for r in rows),
        'main_2mm_2deg_gate_n':sum(r['motion_tracking_pass_2mm_2deg'] for r in rows),
        'p95_position_per_path_mm':[1000*r['position_error_p95_m'] for r in rows],
        'p95_rotation_per_path_deg':np.degrees([r['rotation_error_p95_rad'] for r in rows]),
        'legal_position_p95_range_mm':None if not legal else [min(r['position_error_p95_m'] for r in legal)*1000,max(r['position_error_p95_m'] for r in legal)*1000],
        'legal_rotation_p95_range_deg':None if not legal else np.degrees([min(r['rotation_error_p95_rad'] for r in legal),max(r['rotation_error_p95_rad'] for r in legal)]),
        'acceleration_gate_n':sum(r['acceleration_margin_min']>=-1e-4 for r in rows),
        'commanded_jerk_within_configured_box_n':sum(r['jerk_margin_min']>=-1e-3 for r in rows),
        'invalid_case_ids':[f"{r['method']}/{r['proposal']}" for r in rows if r not in legal]}


def main():
    old=load(HERE/'old_current_eval.json');new=load(HERE/'candidate/eval_report.json');tr=load(HERE/'candidate/train_report.json');audit=load(HERE/'independent_model_audit.json')
    rng=np.random.default_rng(20261132);intervals={}
    for scope in ['global','local']:
        a=audit['models']['old'][scope];b=audit['models']['new'][scope]
        z=np.load(HERE/('fresh_global_audit.npz' if scope=='global' else 'fresh_local_audit.npz'))
        y=z['reachable']>.5;p=np.flatnonzero(y);n=np.flatnonzero(~y)
        ad=np.array(a['classified'])==y;bd=np.array(b['classified'])==y
        delta=[];far=[]
        for _ in range(2000):
            ip=rng.choice(p,len(p),replace=True);inn=rng.choice(n,len(n),replace=True)
            delta.append(.5*((bd[ip].mean()-ad[ip].mean())+(bd[inn].mean()-ad[inn].mean())))
            far.append((np.array(b['accepted'])[inn].mean()-np.array(a['accepted'])[inn].mean()))
        intervals[scope]={'paired_stratified_pose_bootstrap_n':2000,'seed':20261132,
            'BA_new_minus_old_95percentile_interval':np.percentile(delta,[2.5,97.5]),
            'FAR_new_minus_old_95percentile_interval':np.percentile(far,[2.5,97.5]),
            'scope':'independent synthetic pose draws, not patients/worlds or physical US samples'}
    files=list(HERE.glob('*.py'))+[HERE/'run_checks.sh',HERE/'candidate/train.yaml',HERE/'candidate/selected.pt',HERE/'candidate/conformal.json',
        SOURCE/'ird_playground/neural/train_signed.py',BASELINE/'dird/adapter.py',SOURCE/'data/checkpoints/rm4d_signed_current_robot_20261108/selected.pt',
        SOURCE/'data/ird/current_robot_20261108/canonical.npz',ROBOT/'native/wbc_rt/build/wbc_rt']
    result={'architecture_correction':'SRS finite-lattice lift is an offline solver ablation, NOT actual UltraPOC trajectory execution and NOT a dIRD safety oracle',
        'training':{'seconds':tr['elapsed_seconds'],'epochs':len(tr['history']),'split_fingerprint':tr['split_fingerprint'],'split_counts':tr['split_counts'],
            'old_weight_unchanged':sha256(SOURCE/'data/checkpoints/rm4d_signed_current_robot_20261108/selected.pt')=='08698c273bd1f0f4f2a86d956df95448ba502ee9d7fc17445f06258a13286e51',
            'old_eval':old,'new_eval':new,'new_not_promoted':True},
        'fresh_audit':audit,'paired_audit_uncertainty':intervals,
        'native_urdf_frozen_branch0':qp_summary('qp_native_urdf'),
        'native_urdf_branch4_sensitivity_with_invalids_quarantined':qp_summary('qp_native_urdf_branch4_sensitivity'),
        'native_original_first_bad_tick':load(HERE/'isolated_anomaly_native.json')['results'][0]['failures'][0],
        'same_pose_python_anomalies':[r['failures'] for r in load(HERE/'isolated_anomaly_python.json')['results']],
        'instrumented_same_source_result_failures':load(HERE/'instrumented_native_result.json')['result']['failures'],
        'native_rail_root_assignment_unresolved':True,
        'state_replay_native':load(HERE/'qp_state_validation_native.json'),
        'state_replay_python':load(HERE/'qp_state_validation_python.json'),
        'calibration_fix':load(HERE/'calibration_fix_impact.json'),
        'hashes':{str(p):sha256(p) for p in files},
        'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name()},
        'limitations':['ideal q/rail feedback only','contact/image/force outer loop not replayed','fixed nominal reference clock, not full live phase governor/publication stack','no physical robot or human experiment','native rail anomaly blocks executable-certificate claim','fresh audits are finite synthetic pose tests']}
    save('SUMMARY.json',result)
    print('written SUMMARY.json',flush=True)


if __name__=='__main__':main()

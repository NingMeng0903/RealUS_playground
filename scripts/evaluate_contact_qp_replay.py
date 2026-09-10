"""Static real-image → hypothetical outer-QP diagnostics; no closed-loop replay.

Inputs contain same-frame v1/v2/v3 confidence observations. Since the archived
H5 lacks the new controller's executed commands, all nominal velocities and
mechanical states below are declared diagnostic fixtures, not reconstructed
counterfactuals. No future frame is used to claim repair success.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from peirastic.contact_qp.features import FeatureConfig, welleweerd_config
from peirastic.contact_qp.geometry import window_rows, aperture_rows
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.types import ContactObservation, ProbeGeometry

PROFILES = {
    'v1': ('randomwalk_thesis_v1', .5),
    'v2': ('randomwalk_camp_bmode_v2', .5),
    'v3': ('randomwalk_welleweerd2020_v3', .8),
}
FORCES = (3.7, 4., 4.3)


def json_value(value):
    if isinstance(value, dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):return json_value(value.tolist())
    if isinstance(value, np.generic):return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):return None
    return value


def profile_config(name, record):
    default=(welleweerd_config() if name=='v3' else FeatureConfig(algorithm_version=PROFILES[name][0]))
    raw=record.get('feature_config')
    config=FeatureConfig(**raw) if raw is not None else default
    if config.algorithm_version!=PROFILES[name][0]:raise ValueError('profile algorithm mismatch')
    if name=='v3' and config.low_confidence_threshold!=PROFILES[name][1]:
        raise ValueError('declared v3 replay requires threshold 0.8')
    if record.get('window_version',config.window_version)!=config.window_version:
        raise ValueError('profile window version does not match FeatureConfig')
    return config


def observation_for(record, config, frame_seq, *, now_s=1., age_s=.16):
    """Place a historical image in a declared diagnostic freshness fixture."""
    quality=record.get('quality',record.get('quality_lcr'))
    valid=record.get('valid')
    if quality is None or valid is None:raise ValueError('profile quality and valid are required')
    return ContactObservation(frame_seq,'offline-image-diagnostic',now_s-age_s,now_s,
        quality,valid,record.get('registration_version','offline-registration-fixture'),
        config.window_version,calibration_version=config.calibration_version)


def evaluate_case(frame, profile, force_n, nominal_mode, *, intervals=False, age_s=.16):
    """One independent solve; never integrate its command into later frames."""
    geometry=ProbeGeometry.synthetic(.025)
    path=np.array([0.,.02,0.,0.,0.,0.])
    nominal=path.copy()
    if nominal_mode=='force_recovery':nominal[2]=.002*(4.-force_n)
    elif nominal_mode!='visual_isolation':raise ValueError('unknown nominal fixture')
    record={} if profile=='nominal_no_image' else frame['profiles'][profile]
    config=FeatureConfig() if profile=='nominal_no_image' else profile_config(profile,record)
    threshold=.5 if profile=='nominal_no_image' else PROFILES[profile][1]
    cfg=QpConfig(c_min=threshold,lateral_windows=config.lateral_windows,
        quality_policy_version='offline_same_image_diagnostic_'+profile,
        enable_visual=profile!='nominal_no_image',enable_progress_loss=profile!='nominal_no_image',
        compute_interval_diagnostics=intervals)
    observation=None if profile=='nominal_no_image' else observation_for(record,config,
        int(frame.get('frame_index',0)),age_s=age_s)
    result=ContactQp(cfg).solve(QpInput(geometry,nominal,path,force_n,.005,1.,
        observation=observation,previous_twist=nominal,measured_angle=0.))
    diagnostics=dict(result.diagnostics)
    quality=None if observation is None else observation.quality
    row=dict(frame_id=frame.get('frame_id',frame.get('frame_index')),
        source_file=frame.get('source_file',frame.get('source_h5')),frame_index=frame.get('frame_index'),
        historical_force_n_audit_only=frame.get('force_n'),
        dark_fraction_lcr_analysis_only=frame.get('dark_fraction_lcr'),
        profile=profile,force_n=force_n,force_error_n=force_n-4.,nominal_mode=nominal_mode,
        nominal_twist_tool=nominal,quality=quality,valid=None if observation is None else observation.valid,
        c_min=threshold,window_version=config.window_version,
        image_valid=diagnostics.get('image_valid'),visual_rows_active=diagnostics.get('visual_rows_active'),
        quality_left_deficit=None if quality is None else max(0.,threshold-quality[0]),
        quality_right_deficit=None if quality is None else max(0.,threshold-quality[2]),
        alpha=result.alpha,alpha_preferred=diagnostics.get('alpha_preferred'),
        success=result.success,status=result.status.value,reason=diagnostics.get('reason'),
        slack_m_s=result.slack,visual_requests_m_s=diagnostics.get('visual_requests_m_s'),
        force_sign_reliable=diagnostics.get('force_sign_reliable'),
        active_hard_rows=diagnostics.get('active_hard_rows',()),
        omega_interval_mechanical=diagnostics.get('omega_interval_mechanical'),
        omega_interval_force_priority=diagnostics.get('omega_interval_force_priority'),
        omega_interval_complete=diagnostics.get('omega_interval_complete'),
        geometry_verified=False,physical_motion_observed=False,repair_success_measured=None,
        energy_constraint_enabled=False,solver_time_s=diagnostics.get('solver_time_s'),
        solve_total_time_s=diagnostics.get('total_time_s'))
    if result.qp_twist is not None:
        command=result.qp_twist
        local=window_rows(geometry,config.lateral_windows) @ command
        added=window_rows(geometry,config.lateral_windows) @ (command-nominal)
        endpoint=aperture_rows(geometry) @ (command-nominal)
        hard=result.hard_constraints
        values=hard.A @ command
        margins=np.minimum(values-hard.lower,hard.upper-values)
        row.update(qp_twist_tool=command,normal_m_s=float(command[2]),rocking_rad_s=float(command[4]),
            local_velocity_lcr_m_s=local,added_local_velocity_lcr_m_s=added,
            added_aperture_endpoint_m_s=endpoint,max_hard_violation=hard.violation(command),
            hard_row_count=len(hard.labels),minimum_hard_margin=float(np.min(margins)),
            binding_hard_rows=[dict(label=label,lower=lo,value=v,upper=hi,margin=m)
                for label,lo,v,hi,m in zip(hard.labels,hard.lower,values,hard.upper,margins) if m<=1e-8],
            force_priority_added_endpoint_residual=(np.sign(force_n-4.)*endpoint
                if diagnostics.get('force_sign_reliable') else None),
            visual_satisfied_with_slack=(bool(np.all(local[[0,2]]+result.slack>=
                np.asarray(diagnostics.get('visual_requests_m_s',[0.,0.]))-1e-8))
                if diagnostics.get('visual_rows_active') else None))
        reference=frame['profiles']['v3']
        refcfg=profile_config('v3',reference)
        refq=np.asarray(reference.get('quality',reference.get('quality_lcr')),dtype=float)[[0,2]]
        refvalid=bool(np.asarray(reference['valid'],dtype=bool)[[0,2]].all())
        reftarget=.002*np.maximum(.8-refq,0.)/.8-.004*np.maximum(refq-.8,0.)/.2
        reflocal=(window_rows(geometry,refcfg.lateral_windows) @ command)[[0,2]]
        refadded=(window_rows(geometry,refcfg.lateral_windows) @ (command-nominal))[[0,2]]
        row.update(v3_task_reference_valid=refvalid,v3_task_reference_quality_lr=refq,
            v3_task_reference_requests_m_s=reftarget if refvalid else None,
            v3_task_reference_local_m_s=reflocal,v3_task_reference_added_local_m_s=refadded,
            v3_task_reference_required_slack_m_s=np.maximum(reftarget-reflocal,0.) if refvalid else None,
            v3_task_reference_low_window_repair_intent=((refq<.8)&(refadded>1e-8)).tolist() if refvalid else None,
            v3_task_reference_is_ground_truth=False)
    return json_value(row)


class SummaryAccumulator:
    def __init__(self):self.groups={}
    def add(self,row):
        key=(row['profile'],row['force_n'],row['nominal_mode'])
        group=self.groups.setdefault(key,dict(count=0,solve_success_count=0,image_valid_count=0,
            v3_reference_valid_count=0,v3_low_window_count_lr=np.zeros(2),
            v3_repair_intent_count_lr=np.zeros(2),sum={},maxabs={}))
        group['count']+=1;group['image_valid_count']+=int(bool(row['image_valid']))
        if not row['success']:return
        group['solve_success_count']+=1
        fields=('alpha','normal_m_s','rocking_rad_s','max_hard_violation','slack_m_s',
                'local_velocity_lcr_m_s','added_local_velocity_lcr_m_s','solve_total_time_s')
        for field in fields:
            if row.get(field) is None:continue
            value=np.asarray(row[field],dtype=float)
            group['sum'][field]=group['sum'].get(field,0.)+value
            group['maxabs'][field]=np.maximum(group['maxabs'].get(field,0.),np.abs(value))
        if row.get('v3_task_reference_valid'):
            group['v3_reference_valid_count']+=1
            group['v3_low_window_count_lr']+=np.asarray(row['v3_task_reference_quality_lr'])<.8
            group['v3_repair_intent_count_lr']+=np.asarray(row['v3_task_reference_low_window_repair_intent'])
            field='v3_task_reference_required_slack_m_s'
            group['sum'][field]=group['sum'].get(field,0.)+np.asarray(row[field])
    def result(self):
        rows=[]
        for (profile,force,mode),group in sorted(self.groups.items()):
            item=dict(profile=profile,force_n=force,nominal_mode=mode,
                      **{k:v for k,v in group.items() if k not in ('sum','maxabs')})
            for field,total in group['sum'].items():
                n=group['v3_reference_valid_count'] if field.startswith('v3_') else group['solve_success_count']
                item[field+'_mean']=total/max(n,1)
                if field in group['maxabs']:item[field+'_max_abs']=group['maxabs'][field]
            rows.append(json_value(item))
        return rows


def summarize(rows):
    accumulator=SummaryAccumulator()
    for row in rows:accumulator.add(row)
    return accumulator.result()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--feature-configs',type=Path,help='default: feature_configs.json beside input')
    parser.add_argument('--intervals',action='store_true',help='also compute LP rocking intervals (slower)')
    parser.add_argument('--image-age-s',type=float,default=.16)
    parser.add_argument('--forces',type=float,nargs='+',default=list(FORCES))
    parser.add_argument('--nominal-modes',nargs='+',choices=('isolated','recovery','visual_isolation','force_recovery'),default=['isolated','recovery'])
    args=parser.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    configurations_path=args.feature_configs or args.input.with_name('feature_configs.json')
    configurations=json.loads(configurations_path.read_text()) if configurations_path.exists() else {}
    accumulator=SummaryAccumulator();count=0;case_count=0;started=time.perf_counter()
    scene_accumulators={}

    with args.input.open() as stream, (args.output/'qp_diagnostics.jsonl').open('w') as output:
        for line in stream:
            if not line.strip():continue
            frame=json.loads(line)
            for profile,values in configurations.items():
                if profile in frame['profiles']:frame['profiles'][profile]['feature_config']=values
            scene=str(frame.get('source_file',frame.get('source_h5','unknown')))
            scene_accumulator=scene_accumulators.setdefault(scene,SummaryAccumulator())
            for force in args.forces:
                for mode in args.nominal_modes:
                    nominal={'isolated':'visual_isolation','recovery':'force_recovery'}.get(mode,mode)
                    for profile in ('nominal_no_image',*PROFILES):
                        row=evaluate_case(frame,profile,force,nominal,intervals=args.intervals,age_s=args.image_age_s)
                        output.write(json.dumps(row,allow_nan=False)+'\n');accumulator.add(row);scene_accumulator.add(row);case_count+=1
            count+=1
            if args.max_frames and count>=args.max_frames:break
    report=dict(schema=1,input=str(args.input),input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        frame_count=count,case_count=case_count,elapsed_s=time.perf_counter()-started,groups=accumulator.result(),
        groups_by_source={key:value.result() for key,value in scene_accumulators.items()},
        feature_configs=configurations,forces=args.forces,nominal_modes=args.nominal_modes,
        assumptions=dict(geometry='synthetic identity TCP-face, half-aperture 25 mm, image_x_sign +1',
            reference_scan_m_s=.02,force_target_n=4.,dt_s=.005,assumed_image_age_s=args.image_age_s,
            previous_twist='equal to declared nominal, no reconstructed execution history',
            nominal_recovery_rule='vn = 0.002 * (4-F), omega=0; illustrative, not fitted original controller',
            energy='off; no physical work history supplied'),
        scope='static command plausibility only; not original-controller counterfactual, closed-loop repair, or measured coverage/force-performance validation')
    (args.output/'summary.json').write_text(json.dumps(json_value(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(frame_count=count,case_count=case_count,output=str(args.output))))
    return 0


if __name__=='__main__':raise SystemExit(main())

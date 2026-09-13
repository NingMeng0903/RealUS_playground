#!/usr/bin/env python3
"""One-step weighted-QP audit on logged state, never an end-to-end replay.

Uses old recorded 4 N mechanical loading, recomputes production TorqueTilt from
full logged wrench/orientation history, and evaluates each new allocation against
causally preceding actual old-controller publications. No robot or IPC is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.delayed_kf import DelayConfidenceKF, KfConfig
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.geometry import affine_contact_motion, contact_cop_from_wrench, twist_tcp_to_face, wrench_tcp_to_face
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput, _linear_feasible
from peirastic.contact_qp.numeric_record import encode as numeric_encode, SCHEMA as NUMERIC_SCHEMA
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.region_visual import region_visual_request
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from peirastic.contact_qp.runtime_config import calibrated_geometry, load_study_config, validate_study_config
from peirastic.contact_qp.types import ContactObservation, TwistConstraints, REGION_FEATURE_VERSION, WEAK_SIDE_FEATURE_VERSION
from peirastic.realman8dof.force.contact_nominal import build_contact_nominal
from peirastic.scan_path import ForearmReference

SCHEMA = 'frozen_logged_state_regional_one_step_qp_audit_v2'
ASSUMPTIONS = [
    'Each row is an independent new-QP allocation at an old-controller logged reference phase and decision time, not an end-to-end execution success rate.',
    'Normal mechanical vFM is the old log fusion.loading_requested_tool.z; the original 4 N normal observer/filter/integrator state is not reconstructed or invented.',
    'New mechanical omegaM is recomputed with production TorqueTilt on all logged compensated wrench/orientation/dt samples including pre-crop warmup, with zero angular state at the first logged sample because pre-log state is absent.',
    'TorqueTilt nominal transactions commit their mechanical proposal only when the corresponding old actual dual-device publication succeeded; subsequent QP audit outcomes do not alter any logged inputs or old publication history.',
    'Pre-scan mechanical contact is inferred using the configured 0.8 N threshold; original contact hysteresis and QPIK slack-freeze observations are not fully logged. No unlogged slack freeze is asserted.',
    'Path feedforward is reconstructed from production FiniteIntervalReference using the logged reference_s and reference_interval_s and the original path specification.',
    'Effective fixed path feedback is old fusion.scan_requested_tool minus reconstructed unit-progress feedforward. It includes old feedback filtering and any old total-speed saturation residual, not a separately logged feedback signal.',
    'Full preceding old successful published model velocity and angular acceleration are reconstructed from dual-publication events, their actual publication times and sample rotations; previous_outer_command_tool is never used as final publication history.',
    'Rows without two causally prior actual publications, valid old lease, valid force timestamp, or required decomposition fields are excluded with reasons; nothing is cold restarted after a refusal.',
    'All configured equal-width confidence regions drive visual requests and progress; legacy left/center/right statistics are diagnostic only. Symmetric bad-region masks remain visible without inventing a rocking direction.',
    'New cached q25 features use the corresponding old worker logged receipt time when available. Frames without a matched worker receipt are omitted; replacement-worker execution latency is not measured.',
    'KF is initialized from available cached new features. Missing pre-crop q25 frames are not manufactured; initial unavailable-image rows withdraw visual tasks.',
    'Physical velocity/acceleration/jerk/angle limits are unchanged. Offline solver calls have no wall-clock execution deadline; measured compute time is diagnostic and excludes no hard constraint.',
    'The candidate is not sent or integrated into a robot or reference clock. Recorded final velocity is an old-command comparator only. No image improvement or new-controller stability is inferred.'
]


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def log_path_for(record, data_root, session):
    name=record['name'].split('_failed_')[0]
    trial=next(t for t in session['trials'] if Path(t['name']).stem==name)
    if record['failed_attempt']:
        suffix=record['name'].split('_failed_')[1]
        attempt=next(a for a in trial['attempts'] if Path(a['directory']).name==suffix)
    else:
        attempt=next(a for a in reversed(trial['attempts']) if a['status']=='completed')
    return data_root/attempt['directory']/'contact_qp.jsonl',trial['path']


def read_log(path):
    samples=[];by_id={};publication_by_id={};actual_times={};receipts={};stops=[]
    for line in Path(path).open():
        row=json.loads(line);event=row['event'];identifier=row.get('control_id')
        if event=='control_sample':
            sample={k:row.get(k) for k in ('control_id','source','proposal_time_s','control_actual_dt_s',
                'reference_s','reference_interval_s','h_ref_s','rotation_base_tcp','control_wrench_tool',
                'physical_wrench_candidate_tool','fusion_components')}
            samples.append(sample);by_id[identifier]=sample
            feature=row.get('feature')
            if feature is not None:
                key=(int(feature['frame_seq']),round(float(feature['effective_time_s']),7))
                stamp=float(feature['received_time_s'])
                receipts[key]=min(stamp,receipts.get(key,stamp))
        elif event=='publication_transport_result':
            actual_times[identifier]=float(row['publication_time_s'])
        elif event=='publication' and row.get('success') and row.get('facts',{}).get('arm')=='sent' and row.get('facts',{}).get('rail') in ('sent','disabled'):
            publication_by_id[identifier]=row
        elif event in ('force_supervisor_stop','stop'):
            stops.append(row)
    publications=[]
    for identifier,row in publication_by_id.items():
        if identifier not in by_id or identifier not in actual_times:continue
        sample=by_id[identifier]
        publications.append(dict(control_id=identifier,time_s=actual_times[identifier],
            created_s=float(sample['proposal_time_s']),rotation=np.asarray(sample['rotation_base_tcp']),
            final=np.asarray(row['final_command_model_tool'],dtype=float)))
    publications.sort(key=lambda p:p['time_s'])
    for i,p in enumerate(publications):
        p['omega_base']=p['rotation'] @ p['final'][3:]
        p['acceleration_base']=None
        if i and p['time_s']>publications[i-1]['time_s']:
            old=publications[i-1]
            p['acceleration_base']=(p['omega_base']-old['omega_base'])/(p['time_s']-old['time_s'])
    return samples,publications,receipts,stops,set(publication_by_id)


def replay_settings(config):
    law,_=build_contact_nominal(.005)
    controller=law.controller.cfg;tilt=law.tilt
    vmax=np.asarray(controller.max_velocity).copy()
    vmax[2]=min(vmax[2],controller.max_vz_tool_m_s)
    vmax[4]=min(vmax[4],tilt.cfg.vmax_rad_s)
    acceleration=np.asarray(controller.max_acceleration).copy()
    acceleration[4]=min(acceleration[4],tilt.cfg.a_max)
    fc=FeatureConfig(**config['feature']['config'])
    settings=dict(config['qp'])
    settings.update(c_min=config['feature']['c_min'],quality_policy_version=config['feature']['quality_policy_version'],
        lateral_windows=fc.lateral_windows,max_velocity=vmax,max_acceleration=acceleration,
        angle_limit_rad=tilt.cfg.theta_max_rad)
    geometry,_=calibrated_geometry(config)
    if config['geometry']['face_normal_convention']=='outward':
        geometry=replace(geometry,T_tcp_face=geometry.T_tcp_face @ np.diag([1.,-1.,-1.,1.]))
    if not np.allclose(geometry.T_tcp_face,np.eye(4),atol=1e-12,rtol=0):
        raise ValueError('logged vFM/scan decomposition is only identified for the recorded centered aligned face')
    return ContactQp(QpConfig(**settings)),tilt,geometry,fc


def old_history_at(publications, times, now_s):
    index=int(np.searchsorted(times,now_s,side='left'))-1
    if index<1 or publications[index]['acceleration_base'] is None:return None
    return publications[index]


def split_path(sample, reference):
    interval=float(sample['reference_interval_s'])
    ref,_=reference.candidate(float(sample['reference_s']),interval if interval>0 else .005)
    rotation=np.asarray(sample['rotation_base_tcp'])
    ff=np.r_[rotation.T @ ref.vel_ff[:3],rotation.T @ ref.vel_ff[3:]]
    if float(sample['h_ref_s'])<=0 or interval<=0:ff[:]=0.
    ff[[2,4]]=0.
    scan=np.asarray(sample['fusion_components']['scan_requested_tool'],dtype=float)
    feedback=scan-ff
    feedback[[2,4]]=0.
    return feedback,ff


def observations(data, receipts, config, fc):
    if not np.array_equal(data['region_edges'],fc.region_edges):
        raise ValueError('cached region layout does not match configured geometry')
    items=[];missing=0
    for j in range(len(data['frame_seq'])):
        key=(int(data['frame_seq'][j]),round(float(data['effective_s'][j]),7))
        receipt=receipts.get(key)
        if receipt is None:
            missing+=1;continue
        if receipt<float(data['effective_s'][j]):
            missing+=1;continue
        obs=ContactObservation(frame_seq=int(data['frame_seq'][j]),source_id=str(data['source_id'][j]),
            effective_time_s=float(data['effective_s'][j]),received_time_s=receipt,
            quality=data['quality_lcr'][j],valid=[bool(data['valid_lr'][j,0]),True,bool(data['valid_lr'][j,1])],
            registration_version=config['feature']['registration_version'],window_version=fc.window_version,
            calibration_version=fc.calibration_version,confidence_lr=data['confidence_lr'][j],
            confidence_lr_valid=data['valid_lr'][j],weakside_feature_version=WEAK_SIDE_FEATURE_VERSION,
            region_confidence=data['region_confidence'][j],region_valid=data['region_valid'][j],
            region_edges=data['region_edges'],region_feature_version=REGION_FEATURE_VERSION,
            region_layout_version=fc.region_layout_version,
            timestamp_semantics='effective_image_time')
        items.append(obs)
    items.sort(key=lambda obs:(obs.received_time_s,obs.frame_seq))
    return items,missing


def normal_summary(values):
    array=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if not len(array):return dict(count=0)
    return dict(count=len(array),mean=float(np.mean(array)),p50=float(np.median(array)),
        p95=float(np.quantile(array,.95)),max=float(np.max(array)))


def audit(record, spec, config, log_path, output):
    with np.load(record['cache']) as saved:data={key:saved[key] for key in saved.files}
    samples,publications,receipts,stops,success_ids=read_log(log_path)
    publication_by_id={p['control_id']:p for p in publications}
    pub_times=np.asarray([p['time_s'] for p in publications])
    solver,tilt,geometry,fc=replay_settings(config)
    reference=FiniteIntervalReference(ForearmReference(spec),.005)
    kf=DelayConfidenceKF(KfConfig(**config['image_kf']))
    images,unmatched_images=observations(data,receipts,config,fc)
    image_index=0
    smoothing=RockingSmoothing(solver.config.max_velocity[4],solver.config.max_acceleration[4],tilt.cfg.mass,tilt.cfg.damping)
    start_s=float(data['force_s'][0]);end_s=float(data['force_s'][-1])
    counters=Counter();excluded=Counter();refusals=Counter();refusal_classes=Counter();rows=[];angle_origin=None;was_contact=False
    tf=twist_tcp_to_face(geometry)
    first_refusal=None;first_feasible_refusal=None
    latest_image=None
    for sample in samples:
        identifier=sample['control_id'];source=sample['source'] or {}
        source_s=float(source.get('source_t_s',np.nan));now=float(sample['proposal_time_s'])
        dt=float(sample['control_actual_dt_s']);rotation=np.asarray(sample['rotation_base_tcp'])
        wrench=np.asarray(sample['control_wrench_tool'],dtype=float)
        raw=np.asarray(sample['physical_wrench_candidate_tool'] if sample['physical_wrench_candidate_tool'] is not None else wrench,dtype=float)
        contact_wrench=wrench_tcp_to_face(wrench,geometry)
        force=float(contact_wrench[2]);contact=force>=tilt.cfg.contact_n
        if contact and not was_contact:angle_origin=rotation.copy()
        was_contact=contact
        pose=np.r_[np.zeros(3),Rotation.from_matrix(rotation).as_euler('xyz')]
        omega=tilt.prepare(contact_wrench,np.array([0.,0.,4.,0.,0.,0.]),measurement_id=identifier,
            dt_s=dt,pose=pose,contact=contact,measurement_fresh=bool(source.get('fresh',True)))
        counters['torque_recomputed_full_log_samples']+=1
        if identifier in success_ids:tilt.commit_applied(omega)
        else:tilt.abort()
        while image_index<len(images) and images[image_index].received_time_s<=now:
            arriving=images[image_index]
            kf.ingest(arriving,now)
            if latest_image is None or arriving.effective_time_s>=latest_image.effective_time_s:latest_image=arriving
            image_index+=1
        if not start_s-1e-6<=source_s<=end_s+1e-6:continue
        counters['logged_samples_in_cache_interval']+=1
        prediction=kf.predict(now)
        row=dict(control_id=identifier,source_time_s=source_s,decision_time_s=now,
            reference_s=sample['reference_s'],source_age_ms=(now-source_s)*1000,
            force_contact_n=force,omega_nominal_rad_s=omega,audit_status='excluded',reason='',
            original_publication_succeeded=identifier in success_ids,
            new_image_valid=prediction.valid)
        rows.append(row)
        def exclude(reason):
            row['reason']=reason;excluded[reason]+=1
        previous=old_history_at(publications,pub_times,now)
        if previous is None:
            exclude('missing_two_causal_successful_publications');continue
        if previous['control_id']>=identifier:
            exclude('noncausal_publication_join');continue
        age=now-source_s
        if not 0<=age<float(config['source']['max_age_s']):
            exclude('recorded_force_source_not_fresh_at_decision');continue
        if now>=previous['created_s']+float(config['command']['max_interval_s']):
            exclude('recorded_previous_command_lease_expired');continue
        fusion=sample['fusion_components'] or {}
        if 'loading_requested_tool' not in fusion or 'scan_requested_tool' not in fusion:
            exclude('missing_logged_loading_scan_decomposition');continue
        if max(force,float(raw[2]),float(wrench[2]))>=solver.config.differential_repair.hard_stop_n:
            exclude('force_supervisor_stop_required');continue
        if tilt.needs_normal_retract:
            exclude('recomputed_torque_requests_normal_retract');continue
        feedback,ff=split_path(sample,reference)
        normal=float(fusion['loading_requested_tool'][2])
        offset,basis=affine_contact_motion(geometry,feedback,ff)
        previous_rotation=rotation.T @ previous['rotation']
        previous_tool=np.r_[previous_rotation @ previous['final'][:3],previous_rotation @ previous['final'][3:]]
        smoothing.time_s=previous['time_s'];smoothing.omega_base=previous['omega_base'].copy()
        smoothing.acceleration_base=previous['acceleration_base'].copy()
        _,bounds,history=smoothing.preview(rotation,now,dt)
        low=float(np.max(bounds[::2]));high=float(np.min(bounds[1::2]))
        row.update(previous_published_control_id=previous['control_id'],previous_publication_s=previous['time_s'],
            previous_created_s=previous['created_s'],history_dt_s=history['elapsed_s'],
            previous_omega_contact_rad_s=history['previous_rad_s'],
            previous_acceleration_contact_rad_s2=history['previous_acceleration_rad_s2'],
            normal_nominal_m_s=normal,feedback_x_m_s=feedback[0],feedback_y_m_s=feedback[1],
            ff_x_m_s=ff[0],ff_y_m_s=ff[1],rocking_lower_rad_s=low,rocking_upper_rad_s=high)
        if low>high:
            exclude('recorded_published_speed_acceleration_jerk_intersection_empty');continue
        mechanical=TwistConstraints(tf[[4]],[low],[high],valid_until_s=now+float(config['command']['max_interval_s']),
            labels=('published_contact_rocking_speed_acc_jerk',))
        q=prediction.quality_raw
        region_task=region_visual_request(q,None if latest_image is None else latest_image.region_valid,
            fc.region_edges,geometry,c_min=solver.config.c_min,repair_speed_m_s=solver.config.repair_speed_m_s,
            evidence_valid=prediction.valid)
        request=region_task.request_rad_s
        gamma=solver.config.differential_repair.force_gate(force)
        alpha_des=region_task.alpha_preferred
        cop,cop_reason=contact_cop_from_wrench(-contact_wrench,min_force_n=solver.config.cop_min_force_n,
                                            max_abs_m=min(geometry.half_length_m,solver.config.cop_max_m))
        # Logged waiting_for_contact also covers the old 4 N path-start gate.
        visual_execution=contact and fusion.get('visual_reason')!='waiting_for_contact'
        measured_angle=0. if angle_origin is None else float(Rotation.from_matrix(angle_origin.T @ rotation).as_rotvec()[1])
        nominal=offset+basis @ [normal,omega,1.]
        inputs=QpInput(geometry,nominal,ff,force,min(dt,.005),now,previous_twist=previous_tool,
            mechanical=mechanical,measured_angle=measured_angle,acceleration_dt_s=history['elapsed_s'],
            mechanical_normal_m_s=normal,mechanical_omega_rad_s=omega,path_feedback_contact=feedback,
            path_feedforward_contact=ff,visual_request_rad_s=request,visual_task_valid=region_task.valid,
            visual_gamma=gamma,cop_m=cop,alpha_preferred=alpha_des,repair_execution_enabled=visual_execution)
        started=time.perf_counter()
        result=solver.solve(inputs,online=True)
        compute=(time.perf_counter()-started)*1000
        row.update(audit_status='admitted' if result.success else 'refused',reason=result.diagnostics.get('reason',''),
            compute_ms=compute,region_quality=json.dumps(None if q is None else q.tolist()),
            region_deficits=json.dumps(None if region_task.region_deficits is None else region_task.region_deficits.tolist()),
            bad_region_indices=json.dumps(region_task.bad_region_indices),
            bad_region_positions_m=json.dumps(region_task.bad_region_positions_m),
            deficit_first_moment_m=region_task.deficit_first_moment_m,region_visual_reason=region_task.reason,
            request_rad_s=request,visual_gamma=gamma,alpha_preferred=alpha_des,cop_m=cop,cop_reason=cop_reason,
            measured_contact_angle_rad=measured_angle)
        counters['qp_evaluated']+=1
        if result.success:
            counters['qp_admitted']+=1
            diag=result.diagnostics;sol=np.asarray(diag['solution'])
            row.update(qp_U_m_s=sol[0],qp_Omega_rad_s=sol[1],qp_alpha=sol[2],sigma_I_rad_s=sol[3],
                cop_increment_residual_m_s=diag['cop_increment_residual_m_s'],max_hard_violation=diag['max_hard_violation'],
                active_hard_rows='|'.join(diag['active_hard_rows']),visual_rows_active=diag['visual_rows_active'])
            if diag['visual_rows_active']:counters['visual_active']+=1
            if sol[3]>1e-8:counters['visual_shortfall']+=1
            if sol[2]<alpha_des-1e-8:counters['progress_reduced_by_constraints']+=1
            # Compare to the old actual current publication only as a residual.
            actual=publication_by_id.get(identifier)
            if actual is not None:
                old_final=actual['final'];row['old_final_Omega_rad_s']=old_final[4]
                row['old_final_cop_increment_residual_m_s']=None if cop is None else (old_final[2]-normal)-cop*(old_final[4]-omega)
        else:
            counters['qp_refused']+=1
            reason=str(result.diagnostics.get('reason','unknown'));refusals[reason]+=1
            problem=result.diagnostics.get('numeric_problem')
            feasible=None if problem is None else _linear_feasible(np.asarray(problem['C']),np.asarray(problem['l']),np.asarray(problem['u']))
            classification=('mathematically_infeasible' if feasible is False else
                            'feasible_but_numeric_certificate_refused' if feasible is True else 'missing_numeric_problem')
            row['independent_linear_feasible']=feasible;row['refusal_class']=classification
            refusal_classes[classification]+=1
            case=None
            if first_refusal is None or (feasible is True and first_feasible_refusal is None):
                diagnostic_path=output/(record['name']+f'_refusal_{identifier}_numeric.json')
                payload=dict(schema=NUMERIC_SCHEMA,audit_schema=SCHEMA,
                    classification=classification,independent_linear_feasible=feasible,
                    control_id=identifier,source_time_s=source_s,decision_time_s=now,
                    payload=numeric_encode(dict(qp_input=asdict(inputs),qp_config=asdict(solver.config),
                        diagnostics=dict(result.diagnostics))))
                diagnostic_path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
                case=dict(control_id=identifier,source_time_s=source_s,reference_s=sample['reference_s'],
                    reason=reason,independent_linear_feasible=feasible,diagnostics_json=str(diagnostic_path),
                    solver_backend_version=result.diagnostics.get('solver_backend_version'),
                    numeric_attempts=numeric_encode(result.diagnostics.get('numeric_attempts',())))
            if first_refusal is None:
                first_refusal=dict(case,affine_feedback_tcp=feedback.tolist(),
                    feedforward_tcp=ff.tolist(),previous_published_tcp=previous_tool.tolist(),
                    normal_nominal_m_s=normal,omega_nominal_rad_s=omega,history_dt_s=history['elapsed_s'])
            if feasible is True and first_feasible_refusal is None:first_feasible_refusal=case
        if counters['qp_evaluated']%1000==0:print(record['name'],dict(counters),flush=True)
    fields=sorted({key for row in rows for key in row})
    csv_path=output/(record['name']+'.csv')
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    admitted=[row for row in rows if row['audit_status']=='admitted']
    evaluated=[row for row in rows if row['audit_status'] in ('admitted','refused')]
    return dict(name=record['name'],failed_attempt=record['failed_attempt'],source_sha256=record['source_sha256'],
        cache=str(record['cache']),cache_sha256=file_hash(record['cache']),log=str(log_path),log_sha256=file_hash(log_path),
        region_count=fc.region_count,region_edges=fc.region_edges.tolist(),
        region_layout_version=fc.region_layout_version,region_feature_version=REGION_FEATURE_VERSION,
        counts=dict(counters),excluded_reasons=dict(excluded),refusal_reasons=dict(refusals),
        refusal_classes=dict(refusal_classes),first_feasible_refusal_diagnostic=first_feasible_refusal,
        admitted_fraction_of_evaluated=None if not evaluated else len(admitted)/len(evaluated),
        logged_publications=len(publications),cached_frames=len(data['frame_seq']),
        frames_with_logged_worker_receipt=len(images),frames_omitted_without_worker_receipt=unmatched_images,
        first_refusal_diagnostic=first_refusal,commands_csv=str(csv_path),
        nominal_state_provenance=dict(normal='old_logged_loading_requested_tool.z',
            torque='production_TorqueTilt_recomputed_full_log_with_old_publication_commit_mask',
            pre_log_angular_state='unknown_zero_model_initialization',normal_filter_state='not_reconstructed'),
        compute_ms=normal_summary([r.get('compute_ms') for r in evaluated]),
        abs_cop_increment_residual_m_s=normal_summary([None if r.get('cop_increment_residual_m_s') is None else abs(r['cop_increment_residual_m_s']) for r in admitted]),
        visual_shortfall_rad_s=normal_summary([r.get('sigma_I_rad_s') for r in admitted]),
        original_stops=[dict(event=r['event'],reason=r.get('reason'),control_id=r.get('control_id'),
            filtered_original_force_n=r.get('filtered_original_force_n'),raw_original_force_n=r.get('raw_original_force_n'),
            new_force_supervisor_would_stop=(max(r.get('filtered_original_force_n',0.) or 0.,r.get('raw_original_force_n',0.) or 0.)>=solver.config.differential_repair.hard_stop_n) if r['event']=='force_supervisor_stop' else None,
            time_s=r.get('record_monotonic_s')) for r in stops],
        scan_window_s=[start_s,end_s],original_stop_in_window=any(start_s<=r.get('record_monotonic_s',-np.inf)<=end_s+.1 for r in stops))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.resolve()==args.data_root.resolve() or args.data_root.resolve() in args.output.resolve().parents:
        parser.error('outputs must remain outside original acquisition data')
    config=load_study_config(args.config);validate_study_config(config)
    session=json.loads((args.data_root/'session.json').read_text())
    records=json.loads((args.calibration/'manifest.json').read_text())['records']
    args.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for record in records:
        log_path,spec=log_path_for(record,args.data_root,session)
        try:results.append(audit(record,spec,config,log_path,args.output))
        except (ValueError,KeyError,OSError) as exc:
            results.append(dict(name=record['name'],failed_attempt=record['failed_attempt'],excluded_entire_record=True,
                                reason=type(exc).__name__+': '+str(exc)))
        totals=Counter()
        for result in results:totals.update(result.get('counts',{}))
        report=dict(schema=SCHEMA,assumptions=ASSUMPTIONS,actual_image_improvement_measured=False,
            end_to_end_runtime_success_rate_measured=False,totals=dict(totals),results=results,
            provenance=dict(script_sha256=file_hash(__file__),config_sha256=file_hash(args.config),
                production_qp_sha256=file_hash(Path(__file__).parents[1]/'peirastic/contact_qp/weighted_outer.py'),
                production_torque_sha256=file_hash(Path(__file__).parents[1]/'peirastic/realman8dof/force/torque_tilt.py')))
        (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('completed',record['name'],results[-1].get('counts',results[-1]),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())

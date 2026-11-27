"""Frozen command replay with actual complete native continuation state.

This does not replay or replan the L0 image experiment, and does not supply
GT geometry to any controller. No physical robot interface is used.
"""
from pathlib import Path
import hashlib
import json
import time
import traceback
import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from system import ROOT, MOTION, CapabilityEvaluator, body_transform


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__=='__main__':
    source=Path(__file__).parent/'test_frozen/results.json'
    cohort=json.loads(source.read_text())
    assert len(cohort['rows'])==224, 'wait for the complete frozen policy run'
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,
        default=Path(__file__).parent/'native_subset')
    parser.add_argument('--execution-layout',type=Path)
    parser.add_argument('--native-bin',type=Path)
    args=parser.parse_args();outdir=args.output
    outdir.mkdir(exist_ok=False)
    evaluator=CapabilityEvaluator()
    layout=None if args.execution_layout is None else json.loads(args.execution_layout.read_text())
    if args.native_bin is not None:
        sys.path.insert(0,str(ROOT.parent/'ab_validation_20261110/repair_20261110/qp_cross_track_fix'))
        from scan_replay import ScanReplayBridge
        evaluator._native=ScanReplayBridge()
        evaluator._native.raw['inner']['native_bin']=str(args.native_bin.resolve())
    output=dict(scope='Fixed B commands only; actual native continuation; no image/contact loop',
        policy_results_sha256=sha(source),
        protocol_sha256=sha(Path(__file__).parent/'NATIVE_SUBSET_PROTOCOL.txt'),
        runner_sha256=sha(Path(__file__)),robot=evaluator.manifest(),episodes=[],
        execution_layout_override=layout,
        comparison_scope='Original frozen layout' if layout is None else
            'NEW setup with changed body placement and initial q; not the original20 conditions')
    for h in [0,1]:
        for context in cohort['contexts']:
            row=next(r for r in cohort['rows'] if r['method']=='B_two_step' and
                     r['object_index_for_audit_only']==h and r['context']==context['id'])
            execution=context if layout is None else layout
            q=np.array(execution['q_seed']);anchor=np.array(execution['T_world_body'])
            previous=np.zeros(3);state=None
            episode=dict(object_index_for_audit_only=h,context=context['id'],segments=[],
                execution_context=execution.get('id','public_layout_override'),
                L0_correct_public_stop=row['correct_public_stop'],
                note='The L0 success flag is not reassessed by a motion-only replay.')
            for event in row['events']:
                if 'motion_s' not in event:continue
                target=np.array(event['target'])
                decoded=MOTION.path(previous,target)
                poses=np.array([body_transform(p,anchor) for p in decoded])
                desired=poses.copy();chord=np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)
                distance=float(chord.sum());angle=abs(float(target[2]-previous[2]))
                speed=min(.02,.5*distance/max(angle,1e-12)) if distance>1e-10 else .02
                # Rotation and translation are proportional in this planar DMP
                # family. This shared speed limit preserves both registered caps.
                moving=distance>1e-10
                if moving:
                    poses[0]=evaluator.fk(q)
                    keep=np.r_[True,np.linalg.norm(np.diff(poses[:,:3,3],axis=0),axis=1)>1e-10]
                    poses=poses[keep]
                record=dict(ordinal=event['ordinal'],target=target.tolist(),
                    source_body_pose=previous.tolist(),q_actual_entry=q.tolist(),
                    L0_motion_estimate_s=event['motion_s'],requested_speed_m_s=speed,
                    reference_uses_measured_start=moving,desired_DMP_poses=desired.tolist())
                tick=time.perf_counter()
                try:
                    ans=evaluator.evaluate_path(poses,q,exact=False,native=True,
                        native_options=dict(speed_m_s=speed,settle_s=.5,return_trace=True,
                                            return_state=True,initial_state=state))
                    trace=ans['native'].pop('trace',[])
                    if trace:
                        # q_meas_before and pose_desired are from the SAME tick.
                        # The bridge's legacy q_next-vs-current-reference metric
                        # is retained in its result, but is not our replay grade.
                        measured=np.array([evaluator.fk(np.array(r['q_meas_before'])) for r in trace])
                        reference=np.array([r['pose_desired'] for r in trace])
                        pe=np.linalg.norm(measured[:,:3,3]-reference[:,:3,3],axis=1)
                        re=Rotation.from_matrix(np.transpose(reference[:,:3,:3],(0,2,1))@
                                                measured[:,:3,:3]).magnitude()
                        record['same_time_measured_tracking']=dict(
                            sample_count=len(trace),max_position_m=float(pe.max()),
                            p95_position_m=float(np.quantile(pe,.95)),
                            max_rotation_rad=float(re.max()),
                            p95_rotation_rad=float(np.quantile(re,.95)),
                            pass_2mm_2deg=bool(pe.max()<=.002 and re.max()<=np.deg2rad(2)),
                            contract='FK(q_meas_before[t]) versus pose_desired[t]')
                        name=f"h{h}_{context['id']}_{event['ordinal']}_trace.npz"
                        np.savez_compressed(outdir/name,**{
                            k:np.asarray([r[k] for r in trace]) for k in trace[0]})
                        record['trace_file']=name
                    q=np.array(ans['q_out']);state=ans['native']['exit_state']
                    actual=evaluator.fk(q);goal=desired[-1]
                    record.update(status='completed',result=ans,
                        final_position_error_m=float(np.linalg.norm(actual[:3,3]-goal[:3,3])),
                        final_rotation_error_rad=float(Rotation.from_matrix(goal[:3,:3].T@actual[:3,:3]).magnitude()),
                        wall_s=time.perf_counter()-tick)
                    continuation=bool(ans['native']['exit_state_valid'])
                except Exception as exc:
                    record.update(status='error',error_type=type(exc).__name__,error=str(exc),
                                  traceback=traceback.format_exc(),wall_s=time.perf_counter()-tick)
                    continuation=False
                episode['segments'].append(record)
                (outdir/'RESULTS.json').write_text(json.dumps(output|{'in_progress':episode},indent=2,allow_nan=False))
                print(h,context['id'],event['ordinal'],record['status'],
                      record.get('final_position_error_m'),flush=True)
                if not continuation:break
                previous=target
            episode['all_frozen_commands_replayed']=len(episode['segments'])==sum('motion_s' in e for e in row['events'])
            output['episodes'].append(episode)
            (outdir/'RESULTS.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    print('Completed four fixed episodes',flush=True)

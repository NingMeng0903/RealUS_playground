#!/usr/bin/env python3
"""Independent read-only audit of the fixed native continuation matrix."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'ab_continuing_rollout_20261111'
sys.path.insert(0,str(SOURCE))
from native_adapter import NativeAdapter,_state_fingerprint,_state_from_payload
from fixture import segment_hits_box


def same(a,b):
    return json.dumps(a,sort_keys=True,separators=(',',':'))==json.dumps(b,sort_keys=True,separators=(',',':'))


def run():
    failures=[]; metric_warnings=[]; blocks=[];checked_segments=0;max_path_difference=0.;max_tracking_difference=0.
    source_hashes={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('fixture.py','native_adapter.py','robot_motion.py')}
    for fixture,mode in ((0,'constrained'),(1,'constrained'),(0,'open'),(1,'open')):
        path=HERE/f'fixture_{fixture:02d}_{mode}'/'native_exit_matrix.jsonl'
        if not path.exists():
            blocks.append({'fixture':fixture,'mode':mode,'status':'not_run'});continue
        allrows=[]
        for line in path.read_text().splitlines():
            try:allrows.append(json.loads(line))
            except json.JSONDecodeError:failures.append(f'{path}:incomplete_json_line')
        rows={r['cell_key']:r for r in allrows if r.get('row_type')=='cell'}
        summaries=[r for r in allrows if r.get('row_type')=='summary']
        failures_at_start=len(failures)
        with NativeAdapter(fixture,mode,None,backend='native') as adapter:
            root_fp=_state_fingerprint(adapter.start)
            selected_routes={};status_counts={}
            for key,row in rows.items():
                prefix=f'{fixture}/{mode}/{key}'
                status=row['status'];status_counts[status]=status_counts.get(status,0)+1
                state=_state_from_payload(row['input_state'])
                if row['input_state']['fingerprint']!=_state_fingerprint(state):failures.append(prefix+':input_hash')
                if row['stage'] in ('direct_control','root_probe') and _state_fingerprint(state)!=root_fp:failures.append(prefix+':not_root')
                if row['stage']=='continuation':
                    parent=rows.get(f"root_probe|{row['root_action']}|")
                    if parent is None or not same(row['input_state'],parent['output_state']):failures.append(prefix+':full_parent_exit_mismatch')
                out=row.get('output_state')
                if out is not None and out.get('fingerprint')!=_state_fingerprint(_state_from_payload(out)):failures.append(prefix+':output_hash')
                if not row.get('feasible'):continue
                if not out or not out['valid'] or not out.get('native'):failures.append(prefix+':valid_event_without_native_exit');continue
                route=row.get('primary_route')
                if not route or not route['valid']:failures.append(prefix+':missing_valid_primary_route');continue
                selected_routes[route['candidate']]=selected_routes.get(route['candidate'],0)+1
                current_pose=np.asarray(state.pose);current_snapshot=None if state.native is None else state.native.snapshot
                total_path=0.;total_duration=0.
                for step in route['steps']:
                    checked_segments+=1;record=f'{prefix}/{step["record"]}'
                    tracepath=Path(step['trace']['trace_path'])
                    if not tracepath.exists() and row.get('source_checkpoint'):
                        candidate=Path(row['source_checkpoint']).parent.parent/f"{step['record']}.npz"
                        if candidate.exists():
                            metric_warnings.append(record+':stored_trace_path_corrected_to_source_planning_directory');tracepath=candidate
                    if not tracepath.exists():failures.append(record+':trace_missing');continue
                    if not same(step['entry_snapshot'],current_snapshot):failures.append(record+':entry_snapshot_mismatch')
                    if not np.array_equal(np.asarray(step['entry_pose']),current_pose):failures.append(record+':entry_pose_mismatch')
                    with np.load(tracepath,allow_pickle=False) as t:
                        actual=np.asarray(t['actual']);desired=np.asarray(t['desired'])
                        before=np.concatenate((current_pose[None],actual[:-1]),axis=0)
                        pe=float(np.linalg.norm(before[:,:3,3]-desired[:,:3,3],axis=1).max())
                        rr=np.swapaxes(before[:,:3,:3],1,2)@desired[:,:3,:3]
                        re=float(Rotation.from_matrix(rr).magnitude().max())
                        max_tracking_difference=max(max_tracking_difference,abs(pe-step['same_tick_position_error_max_m']),abs(re-step['same_tick_rotation_error_max_rad']))
                        if pe>.002+1e-12 or re>math.radians(2)+1e-12:failures.append(record+':aligned_gate')
                        if not step['native_tracking_gate'] or step['paused_count'] or not step['exit_state_valid']:failures.append(record+':native_gate_or_pause')
                        public=(actual[:,:3,3]-adapter.rig.measured[:3,3])@adapter.rig.measured[:3,:3]
                        if not all(adapter.rig.geometry.check_local(p[:2])[0] for p in public):failures.append(record+':workspace')
                        if adapter.rig.wall is not None and any(segment_hits_box(a,b,adapter.rig.wall) for a,b in zip(public[:-1],public[1:])):failures.append(record+':swept_wall')
                        true=(actual[:,:3,3]-adapter.rig.actual_surface[:3,3])@adapter.rig.actual_surface[:3,:3]
                        if np.max(np.abs(true[:,2]))>.003+1e-12:failures.append(record+':contact_gap')
                        xyz=np.concatenate((current_pose[None,:3,3],actual[:,:3,3]),axis=0)
                        length=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum());total_path+=length
                        reported=step['trace']['actual_tcp_path_length_m']
                        if reported is None:metric_warnings.append(record+':stored_path_missing_independently_recomputed')
                        else:max_path_difference=max(max_path_difference,abs(length-reported))
                        current_pose=actual[-1]
                        if step is route['steps'][-1]:
                            for field in ('q','qdot'):
                                if not np.array_equal(t[field][-1],np.asarray(out['native'][field])):failures.append(record+':exit_'+field)
                # Output controller snapshot is stored at the last selected segment.
                    current_snapshot=step['exit_snapshot'];total_duration+=step['duration_s']
                if not np.array_equal(current_pose,np.asarray(out['pose'])):failures.append(prefix+':final_pose')
                if not same(current_snapshot,out['native']['snapshot']):failures.append(prefix+':final_snapshot')
                if abs(total_duration-row['transfer_time_s'])>1e-10:failures.append(prefix+':duration_sum')
                if row['actual_tcp_path_length_m'] is None:metric_warnings.append(prefix+':stored_path_missing_independently_recomputed')
                elif abs(total_path-row['actual_tcp_path_length_m'])>1e-8:metric_warnings.append(prefix+':stored_path_excludes_entry_to_first_sample')
            complete=bool(summaries) and len(rows)==38
            blocks.append({'fixture':fixture,'mode':mode,'status':'complete' if complete else 'running_or_partial','cells':len(rows),'status_counts':status_counts,'selected_routes':selected_routes,'failures':len(failures)-failures_at_start,'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    result={'source_hashes':source_hashes,'blocks':blocks,'selected_segments_checked':checked_segments,'max_path_report_difference_m':max_path_difference,'max_tracking_report_difference':max_tracking_difference,'metric_warnings':metric_warnings,'failures':failures,'all_observed_checks_pass':not failures,'scope':'independent recomputation of selected-route traces and complete-state continuity; no claim of physical robot contact or ultrasound image quality'}
    (HERE/'ASTRA_NATIVE_MATRIX_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()

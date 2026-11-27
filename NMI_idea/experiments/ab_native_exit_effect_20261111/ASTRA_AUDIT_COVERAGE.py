#!/usr/bin/env python3
"""Independent enumeration/state/trace audit of prescribed-pose coverage."""
from pathlib import Path
from itertools import permutations
import hashlib,json,math,sys
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'ab_continuing_rollout_20261111';sys.path.insert(0,str(SOURCE))
from native_adapter import NativeAdapter,_state_fingerprint,_state_from_payload
from fixture import segment_hits_box,segment_duration,SETTLE_S,local_pose
FORKS=('fork_minus__single_dwell_0.3','fork_plus__single_dwell_0.3')

def geom(adapter,order):
    start=local_pose(adapter.measured_frame,adapter.start.pose);total=0.
    for a in order:
        cap=adapter.rig.captures[a];cost=[]
        for points in adapter.rig.route_options(start,cap.goal).values():
            t=0.;old=start
            for i,p in enumerate(points):
                d=segment_duration(old,p)
                if cap.kind=='fork':d=max(d,1.1 if abs(p[2]-old[2])>.12 else .8)
                t+=(math.ceil((d+SETTLE_S+(cap.dwell_s if i==len(points)-1 else 0.))/.005)+1)*.005;old=p
            cost.append(t)
        if not cost:return None
        total+=min(cost);start=cap.goal
    return total

def main():
    failures=[];tasks=[];links_audited=set();segments_audited=set();path_entry_omission_max=0.
    for f,m in ((0,'constrained'),(1,'constrained'),(0,'open'),(1,'open')):
        directory=HERE/f'coverage_{f:02d}_{m}';checkpoints={}
        for p in sorted((directory/'planning/checkpoints').glob('*.json')):
            d=json.loads(p.read_text());out=_state_from_payload(d['output_state'])
            checkpoints[(d['input_fingerprint'],d['action_id'],_state_fingerprint(out))]=(p,d)
        with NativeAdapter(f,m,None,backend='native') as adapter:
            for target in ('left','right'):
                p=directory/f'{target}.json'
                if not p.exists():failures.append(f'{f}/{m}/{target}:not_run');continue
                d=json.loads(p.read_text());orders=d['orders'];required=(*FORKS,target);label=f'{f}/{m}/{target}'
                expected=set(permutations(required));got={tuple(x['order']) for x in orders}
                if got!=expected or len(orders)!=6:failures.append(label+':order_cohort')
                for order in orders:
                    prefix=label+'/'+','.join(order['order']);state_fp=_state_fingerprint(adapter.start);elapsed=0.;pathsum=0.
                    for link in order['links']:
                        if link.get('state_in')!=state_fp:failures.append(prefix+':state_chain')
                        if link['status']!='passed':break
                        key=(link['state_in'],link['action'],link['state_out']);source=checkpoints.get(key)
                        if source is None:failures.append(prefix+':no_complete_checkpoint');continue
                        cp,entry=source;state=_state_from_payload(entry['input_state']);out=_state_from_payload(entry['output_state'])
                        if entry['adapter_fingerprint']['sha256']!=adapter._adapter_fingerprint()['sha256']:failures.append(prefix+':source_version')
                        if _state_fingerprint(state)!=link['state_in'] or not out.valid:failures.append(prefix+':checkpoint_state')
                        if abs(out.elapsed-state.elapsed-link['elapsed_s'])>1e-10:failures.append(prefix+':duration')
                        route=entry['event']['result']['route'];actualpath=0.;beforepose=state.pose
                        if not route['valid'] or route['route']!=link['route']:failures.append(prefix+':selected_route')
                        for step in route['steps']:
                            tracepath=directory/'planning'/f"{step['record']}.npz"
                            with np.load(tracepath,allow_pickle=False) as t:
                                actual=t['actual'];desired=t['desired'];before=np.concatenate((beforepose[None],actual[:-1]),axis=0)
                                pos=np.linalg.norm(before[:,:3,3]-desired[:,:3,3],axis=1).max();rot=Rotation.from_matrix(np.swapaxes(before[:,:3,:3],1,2)@desired[:,:3,:3]).magnitude().max()
                                if pos>.002+1e-12 or rot>math.radians(2)+1e-12 or not step['native_tracking_gate'] or step['paused']:failures.append(prefix+':tracking_gate')
                                xy=(actual[:,:3,3]-adapter.measured_frame[:3,3])@adapter.measured_frame[:3,:3]
                                true=(actual[:,:3,3]-adapter.rig.actual_surface[:3,3])@adapter.rig.actual_surface[:3,:3]
                                if np.max(np.abs(true[:,2]))>.003+1e-12 or not all(adapter.rig.geometry.check_local(q[:2])[0] for q in xy):failures.append(prefix+':contact_workspace')
                                if adapter.rig.wall and any(segment_hits_box(a,b,adapter.rig.wall) for a,b in zip(xy[:-1],xy[1:])):failures.append(prefix+':wall')
                                entrygap=float(np.linalg.norm(actual[0,:3,3]-beforepose[:3,3]));path_entry_omission_max=max(path_entry_omission_max,entrygap)
                                actualpath+=float(np.linalg.norm(np.diff(actual[:,:3,3],axis=0),axis=1).sum());beforepose=actual[-1]
                                if step is route['steps'][-1]:
                                    if not np.array_equal(t['q'][-1],out.native.q) or not np.array_equal(t['qdot'][-1],out.native.qdot) or not np.array_equal(actual[-1],out.pose):failures.append(prefix+':exit_state')
                            segments_audited.add(str(tracepath))
                        if abs(actualpath-link['path_m'])>1e-10:failures.append(prefix+':path_total')
                        elapsed+=link['elapsed_s'];pathsum+=link['path_m'];state_fp=link['state_out'];links_audited.add(str(cp))
                    if order['complete'] and (len(order['links'])!=3 or any(x['status']!='passed' for x in order['links'])):failures.append(prefix+':completion')
                    if order['complete'] and (abs(elapsed-order['motion_s'])>1e-10 or abs(pathsum-order['path_m'])>1e-10):failures.append(prefix+':tour_total')
                    forecast=geom(adapter,order['order'])
                    if forecast is None or abs(forecast-order['nominal_s'])>1e-10:failures.append(prefix+':geometric_forecast')
                feasible=[o for o in orders if o['complete']]
                best=min(feasible,key=lambda o:(round(o['motion_s'],9),o['order'])) if feasible else None
                geometric=min(orders,key=lambda o:(round(o['nominal_s'],9),o['order']))
                prefix=[]
                for stage in range(3):
                    immediate={o['links'][stage]['action']:o['links'][stage]['elapsed_s'] for o in orders if o['order'][:stage]==prefix and len(o['links'])>stage and o['links'][stage]['status']=='passed'}
                    if not immediate:break
                    prefix.append(min(immediate,key=lambda a:(round(immediate[a],9),a)))
                greedy=next((o for o in orders if o['order']==prefix),None)
                for arm,expected_arm in (('A_greedy',greedy),('A_geometric_tour',geometric),('B_verified_tour',best)):
                    if d[arm]!=expected_arm:failures.append(label+':'+arm+'_selection')
                tasks.append({'fixture':f,'mode':m,'target':target,'all_six_complete':len(feasible)==6,'greedy_order':None if greedy is None else greedy['order'],'geometric_order':geometric['order'],'native_order':None if best is None else best['order'],'greedy_s':None if greedy is None else greedy['motion_s'],'geometric_s':geometric['motion_s'],'native_s':None if best is None else best['motion_s'],'native_minus_greedy_s':None if greedy is None or best is None else best['motion_s']-greedy['motion_s'],'native_equals_geometric':best is not None and best['order']==geometric['order'],'result_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    result={'tasks':tasks,'native_links_checked':len(links_audited),'native_segments_checked':len(segments_audited),'max_omitted_entry_to_first_sample_m':path_entry_omission_max,'failures':failures,'passed':not failures,'source_sha256':hashlib.sha256((HERE/'coverage_order.py').read_bytes()).hexdigest(),'scope':'deterministic prescribed-pose motion and dwell; no image quality, patients or learned prediction'}
    (HERE/'ASTRA_COVERAGE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

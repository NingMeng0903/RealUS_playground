"""Replay a pose array using the checked scan profile, without robot I/O."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scan_replay import ScanReplayBridge,HERE
from common import clean_json,sha256


def middle_summary(position,angle,t,duration,edge):
    mask=(t>=edge)&(t<=duration-edge)
    if not mask.any():
        return dict(samples=0,position_max_mm=None,angle_max_deg=None)
    return dict(samples=int(mask.sum()),position_max_mm=float(position[mask].max()*1000),
        angle_max_deg=float(np.degrees(angle[mask].max())))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE.parent/'audit'/'phase2_native_subset_inputs.npz')
    parser.add_argument('--poses-key',default='branch4_path0_T')
    parser.add_argument('--q-key',default='branch4_path0_q_seed')
    parser.add_argument('--profile',type=Path,default=HERE/'scan_tracking_profile.yaml')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    data=np.load(args.input,allow_pickle=False)
    bridge=ScanReplayBridge(args.profile)
    result=bridge.query_pose_path(data[args.poses_key],data[args.q_key],
        speed_m_s=bridge.reference_options['speed_m_s'],settle_s=bridge.reference_options['settle_s'],
        return_trace=True,return_state=False,provenance='explicit reusable offline scan profile')
    trace=result.pop('trace')
    if not trace:
        result['time_aligned_middle']=dict(samples=0,position_max_mm=None,angle_max_deg=None)
        (args.output/'results.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
        raise SystemExit(2)
    target=args.output/'trace.npz'
    np.savez_compressed(target,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
    # Current measured pose, not the next-tick command, is paired with T(t).
    kin,controller,cfg=bridge.make_controller()
    controller.close()
    actual=np.array([kin.fk_placement(row['q_meas_before']).homogeneous.copy() for row in trace])
    desired=np.array([row['pose_desired'] for row in trace])
    position=np.linalg.norm(desired[:,:3,3]-actual[:,:3,3],axis=1)
    angle=Rotation.from_matrix(np.transpose(desired[:,:3,:3],(0,2,1))@actual[:,:3,:3]).magnitude()
    t=np.array([row['t_s'] for row in trace]);edge=bridge.profile['evaluation']['middle_edge_s']
    result['time_aligned_middle']=middle_summary(position,angle,t,result['nominal_path_duration_s'],edge)
    result.update(input_sha256=sha256(args.input),trace_sha256=sha256(target))
    (args.output/'results.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
    print(result['time_aligned_middle'])

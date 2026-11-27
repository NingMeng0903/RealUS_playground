"""Compare measured FK(q_k) with commanded T(t_k), preserving old metrics."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
import run_lead_ablation as lead

HERE=Path(__file__).resolve().parent


if __name__=='__main__':
    bridge=lead.old.replay.QPReplayBridge('native','urdf',native_clock='deterministic')
    kin,inner,cfg=bridge.make_controller()
    inner.close()
    runs=json.loads((HERE/'smooth_reference_results.json').read_text())['runs']
    out=dict(evaluation='FK(q_meas_before[k]) versus desired_pose[k], both at t_k',
        old_evaluation='FK(q_send[k]) belongs to t_k+dt but was compared with desired_pose[k]',
        actual_ramp_s=.8,legacy_bridge_ramp_metadata_incorrect=.4,
        protocol_sha256=lead.sha(HERE/'MIDSCAN_EVALUATION_PROTOCOL.txt'),runs={})
    for run in runs:
        name=run['case']
        z=np.load(HERE/f'{name}_trace.npz',allow_pickle=False)
        q=z['q_meas_before']
        np.testing.assert_array_equal(q[1:],z['q'][:-1])
        np.testing.assert_allclose(z['native_model_t_mono'],1000.+z['t_s'],atol=1e-10,rtol=0.)
        actual=np.array([kin.fk_placement(x).homogeneous.copy() for x in q])
        desired=z['pose_desired']
        error=desired[:,:3,3]-actual[:,:3,3]
        position=np.linalg.norm(error,axis=1)
        angle=Rotation.from_matrix(np.transpose(desired[:,:3,:3],(0,2,1))@actual[:,:3,:3]).magnitude()
        speed=np.linalg.norm(z['vel_ff'][:,:3],axis=1)
        tangent=np.divide(z['vel_ff'][:,:3],speed[:,None],out=np.zeros_like(error),where=speed[:,None]>1e-12)
        along=(error*tangent).sum(axis=1)
        normal=np.sqrt(np.maximum(0,position**2-along**2))
        t=z['t_s'];mask=(t>=.8)&(t<=run['nominal_path_duration_s']-.8)
        out['runs'][name]=dict(samples=int(mask.sum()),
            position_middle_max_mm=float(position[mask].max()*1000),
            position_middle_p95_mm=float(np.percentile(position[mask],95)*1000),
            angle_middle_max_deg=float(np.degrees(angle[mask].max())),
            angle_middle_p95_deg=float(np.degrees(np.percentile(angle[mask],95))),
            normal_middle_max_mm=float(normal[mask].max()*1000),
            position_full_max_mm=float(position.max()*1000),angle_full_max_deg=float(np.degrees(angle.max())),
            q_before_equals_previous_qsend_max=float(np.max(np.abs(q[1:]-z['q'][:-1]))),
            original_jerk_margin_min=run['jerk_margin_min'],
            original_strict_mechanical_gate=run['strict_mechanical_gate'])
        target=HERE/f'{name}_aligned_errors.npz'
        np.savez_compressed(target,t_s=t,measured_pose=actual,desired_pose=desired,
            position_error_m=position,angle_error_rad=angle,normal_error_m=normal,
            along_error_m=along,middle_mask=mask)
        out['runs'][name]['aligned_errors_sha256']=lead.sha(target)
    (HERE/'time_aligned_tracking.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

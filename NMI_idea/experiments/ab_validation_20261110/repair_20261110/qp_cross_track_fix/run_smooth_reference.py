"""Analytic smooth6D scan reference on the same input waypoints."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
import run_lead_ablation as lead
import summarize_midscan as mid

from rm75_control.control.joint_admittance_8dof.scan_reference import WorldPoseSplineReference
from rm75_control.control.joint_admittance_8dof import scan_reference as production

HERE=Path(__file__).resolve().parent
W=50*(np.deg2rad(2.)/.002)**2


class Scan08(WorldPoseSplineReference):
    def __init__(self,*a,**kw):
        kw['ramp_s']=.8
        super().__init__(*a,**kw)


class ScanBridge(lead.LeadBridge):
    def make_controller(self):
        kin,inner,cfg=super().make_controller()
        self.native_config=Path(inner._native._cfg_path).read_text()
        return kin,inner,cfg


def geometry_deviation(ref):
    u=np.linspace(0.,ref.knots[-1],4097)
    x=ref.position_spline(u)
    p=ref.points;d=np.diff(p,axis=0);d2=(d*d).sum(axis=1)
    alpha=np.clip(((x[:,None]-p[:-1])*d).sum(axis=2)/d2,0,1)
    foot=p[:-1][None]+alpha[:,:,None]*d[None]
    distance=np.linalg.norm(x[:,None]-foot,axis=2).min(axis=1)
    return dict(sample_count=len(x),max_distance_to_input_chords_m=float(distance.max()),
        peak_translation_m_s=ref.speed_m_s,parameter_speed=ref.parameter_rate,
        maximum_position_derivative=ref.maximum_parameter_speed)


if __name__=='__main__':
    old=lead.old
    data=np.load(old.INPUT,allow_pickle=False)
    old.replay.WorldPolylineReference=Scan08
    out=dict(protocol_sha256=lead.sha(HERE/'SMOOTH_REFERENCE_PROTOCOL.txt'),
        code_sha256=lead.sha(__file__),production_reference_sha256=lead.sha(production.__file__),
        input_sha256=lead.sha(old.INPUT),runs=[])
    for path in (0,1):
        for metric,linear_w in [('baseline',100.),('unit',W)]:
            name=f'path{path}_smooth_{metric}'
            bridge=ScanBridge('native','urdf',native_clock='deterministic')
            bridge.raw['qpik']['hard_limits']['command_lead_arm_deg']=20.
            bridge.raw['inner']['qp']['task_weight']=[linear_w]*3+[50.]*3
            T,q=data[f'branch4_path{path}_T'],data[f'branch4_path{path}_q_seed']
            raw=bridge.query_pose_path(T,q,speed_m_s=.02,settle_s=2.,return_trace=True,
                return_state=False,provenance=f'frozen differentiable6D reference {name}')
            trace=raw.pop('trace')
            old.summarize(raw,trace)
            raw['case']=name
            ref=Scan08(T[:,:3,3],rpy=Rotation.from_matrix(T[:,:3,:3]).as_euler('xyz'),speed_m_s=.02)
            reference=[ref.sample(row['t_s']) for row in trace]
            for a,b in zip(reference,trace):
                np.testing.assert_allclose(a.pose_d[:3],b['pose_desired'][:3,3],atol=1e-13)
                np.testing.assert_allclose(Rotation.from_euler('xyz',a.pose_d[3:]).as_matrix(),b['pose_desired'][:3,:3],atol=1e-13)
            p=HERE/f'{name}_trace.npz'
            np.savez_compressed(p,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]},
                vel_ff=np.array([x.vel_ff for x in reference]))
            native=HERE/f'{name}_native_config.txt';native.write_text(bridge.native_config)
            raw.update(trace_sha256=lead.sha(p),native_config_sha256=lead.sha(native),
                geometry=geometry_deviation(ref))
            with np.load(p,allow_pickle=False) as z:
                raw['regional']=mid.summarize(T,z,raw['nominal_path_duration_s'])
            out['runs'].append(raw)
            (HERE/'smooth_reference_results.json').write_text(json.dumps(old.clean_json(out),indent=2,allow_nan=False))
            print(name,'fullmm',1000*raw['full_position_max_m'],'fulldeg',raw['full_rotation_max_deg'],
                'middle',raw['regional']['middle_phase'],'jerk',raw['jerk_margin_min'],flush=True)

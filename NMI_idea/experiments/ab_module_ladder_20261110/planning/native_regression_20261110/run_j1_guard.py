"""One-factor causal test of artificial J1 overfold versus actual URDF limit."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_controlled_comparison import ROOT,DiagnosticBridge,CapabilityEvaluator,MOTION,body_transform,clean_json,sha


if __name__=='__main__':
    directory=HERE/'j1_guard_only';directory.mkdir(exist_ok=False)
    cohort=json.loads((ROOT/'planning/test_frozen/results.json').read_text());scene=cohort['contexts'][0]
    q=np.array(scene['q_seed']);anchor=np.array(scene['T_world_body'])
    event=next(r for r in cohort['rows'] if r['object_index_for_audit_only']==0 and
               r['method']=='B_two_step' and r['context']==scene['id'])['events'][0]
    evaluator=CapabilityEvaluator()
    moving=np.array([body_transform(p,anchor) for p in MOTION.path(np.zeros(3),np.array(event['target']))])
    hold=np.repeat(evaluator.fk(q)[None],2,axis=0)
    cases=[('motion',moving,.5),('hold10s',hold,10.)]
    output=dict(scope='Only J1 artificial overfold changed; every other original profile setting retained',
                code_sha256=sha(__file__),cases=[])
    for name,poses,settle in cases:
        bridge=DiagnosticBridge()
        # Remove the EXTRA140degree preference wall; physical joint bounds,
        # velocity/acceleration/jerk and collision constraints stay unchanged.
        cap=min(abs(float(evaluator.kin.q_lower[1])),abs(float(evaluator.kin.q_upper[1])))
        bridge.raw['inner']['qp'].setdefault('branch_barrier',{})['j1_overfold_abs_rad']=cap
        start=time.perf_counter()
        ans=bridge.query_pose_path(poses,q,speed_m_s=.02,settle_s=settle,
                                  return_trace=True,return_state=True,provenance=f'j1guard {name}')
        trace=ans.pop('trace')
        actual=np.array([evaluator.fk(np.array(r['q_meas_before'])) for r in trace])
        desired=np.array([r['pose_desired'] for r in trace])
        pe=np.linalg.norm(actual[:,:3,3]-desired[:,:3,3],axis=1)
        re=Rotation.from_matrix(np.transpose(desired[:,:3,:3],(0,2,1))@actual[:,:3,:3]).magnitude()
        path=directory/f'{name}_trace.npz'
        np.savez_compressed(path,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
        (directory/f'{name}_native_config.txt').write_text(bridge.effective_native)
        row=dict(case=name,j1_cap_rad=cap,max_position_m=float(pe.max()),max_rotation_rad=float(re.max()),
                 end_position_m=float(pe[-1]),trace_sha256=sha(path),wall_s=time.perf_counter()-start,result=ans)
        output['cases'].append(row)
        (directory/'RESULTS.json').write_text(json.dumps(clean_json(output),indent=2,allow_nan=False))
        print(name,'maxmm',pe.max()*1000,'maxdeg',np.rad2deg(re.max()),'endmm',pe[-1]*1000,flush=True)

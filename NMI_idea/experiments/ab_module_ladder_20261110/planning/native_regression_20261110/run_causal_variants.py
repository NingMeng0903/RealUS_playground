"""Offline causal 2x2 diagnostic; no production settings or hardware changes."""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_controlled_comparison import ROOT,DiagnosticBridge,CapabilityEvaluator,MOTION,body_transform,clean_json,sha


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    directory=parser.parse_args().output;directory.mkdir(exist_ok=False)
    cohort=json.loads((ROOT/'planning/test_frozen/results.json').read_text())
    scene=cohort['contexts'][0];q=np.array(scene['q_seed']);anchor=np.array(scene['T_world_body'])
    event=next(r for r in cohort['rows'] if r['object_index_for_audit_only']==0 and
               r['method']=='B_two_step' and r['context']==scene['id'])['events'][0]
    motion=np.array([body_transform(p,anchor) for p in MOTION.path(np.zeros(3),np.array(event['target']))])
    evaluator=CapabilityEvaluator();hold=np.repeat(evaluator.fk(q)[None],2,axis=0)
    result=dict(scope='Offline only; causal task scaling x jerk-box removal; actual velocity/acceleration/position/collision unchanged',
        code_sha256=sha(__file__),cases=[])
    for normalized,relaxed in [(False,False),(True,False),(False,True),(True,True)]:
        tag=f"{'unit' if normalized else 'raw'}_{'jerk_loose' if relaxed else 'jerk_original'}"
        for name,poses,settle in [('motion',motion,.5),('hold6s',hold,6.)]:
            bridge=DiagnosticBridge()
            if normalized:
                wp=50*(np.deg2rad(2.)/.002)**2
                bridge.raw['inner']['qp']['task_weight']=[wp]*3+[50.]*3
            if relaxed:
                bridge.raw['inner']['qp']['j_max_arm_rad_s3']=1e9
                bridge.raw['inner']['qp']['j_max_rail_m_s3']=1e9
            tick=time.perf_counter()
            ans=bridge.query_pose_path(poses,q,speed_m_s=.02,settle_s=settle,return_trace=True,
                return_state=False,provenance=f'causal diagnostic {tag} {name}')
            trace=ans.pop('trace')
            actual=np.array([evaluator.fk(np.array(r['q_meas_before'])) for r in trace])
            desired=np.array([r['pose_desired'] for r in trace])
            pe=np.linalg.norm(actual[:,:3,3]-desired[:,:3,3],axis=1)
            re=Rotation.from_matrix(np.transpose(desired[:,:3,:3],(0,2,1))@actual[:,:3,:3]).magnitude()
            target=directory/f'{tag}_{name}_trace.npz'
            np.savez_compressed(target,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
            (directory/f'{tag}_{name}_native_config.txt').write_text(bridge.effective_native)
            row=dict(variant=tag,case=name,max_position_m=float(pe.max()),
                max_rotation_rad=float(re.max()),last_position_m=float(pe[-1]),
                trace_sha256=sha(target),result=ans,wall_s=time.perf_counter()-tick)
            result['cases'].append(row)
            (directory/'RESULTS.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
            print(tag,name,'maxmm',pe.max()*1000,'maxdeg',np.rad2deg(re.max()),'endmm',pe[-1]*1000,flush=True)

"""Old-success/new-failure controlled offline replay, with sources left intact."""
from pathlib import Path
import hashlib
import json
import sys
import time
import argparse
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPAIR=ROOT.parent/'ab_validation_20261110/repair_20261110'
sys.path.insert(0,str(ROOT/'planning'))
from system import CapabilityEvaluator, MOTION, body_transform
from capability import clean_json
sys.path.insert(0,str(REPAIR/'qp_cross_track_fix'))
from scan_replay import ScanReplayBridge


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class DiagnosticBridge(ScanReplayBridge):
    def make_controller(self):
        kin,inner,cfg=super().make_controller()
        self.effective_native=Path(inner._native._cfg_path).read_text()
        self.cfg=cfg
        return kin,inner,cfg


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--native-bin',type=Path)
    args=parser.parse_args();outdir=args.output
    outdir.mkdir(exist_ok=False)
    old=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz')
    cohort=json.loads((ROOT/'planning/test_frozen/results.json').read_text())
    nominal=cohort['contexts'][0];stress=cohort['contexts'][1]
    first=next(r for r in cohort['rows'] if r['object_index_for_audit_only']==0 and
               r['method']=='B_two_step' and r['context']==nominal['id'])['events'][0]
    q=np.array(nominal['q_seed']);anchor=np.array(nominal['T_world_body'])
    T=np.array([body_transform(p,anchor) for p in MOTION.path(np.zeros(3),np.array(first['target']))])
    cases=[('old_path0',old['branch4_path0_T'],old['branch4_path0_q_seed'],2.),
           ('old_path1',old['branch4_path1_T'],old['branch4_path1_q_seed'],2.),
           ('new_first_nominal',T,q,.5)]
    evaluator=CapabilityEvaluator()
    for name,scene in [('nominal',nominal),('stress',stress)]:
        seed=np.array(scene['q_seed'])
        cases.append((f'hold_{name}_10s',np.repeat(evaluator.fk(seed)[None],2,axis=0),seed,10.))
    result=dict(scope='Offline ideal plant only; source/config unchanged',cases=[],
        code_sha256=sha(__file__),old_input_sha256=sha(REPAIR/'audit/phase2_native_subset_inputs.npz'))
    for name,poses,seed,settle in cases:
        if args.cases is not None and name not in args.cases:continue
        bridge=DiagnosticBridge()
        if args.native_bin is not None:
            bridge.raw['inner']['native_bin']=str(args.native_bin.resolve())
        tick=time.perf_counter()
        ans=bridge.query_pose_path(poses,seed,speed_m_s=.02,settle_s=settle,
                                  return_trace=True,return_state=True,provenance=name)
        trace=ans.pop('trace')
        actual=np.array([evaluator.fk(np.array(r['q_meas_before'])) for r in trace])
        desired=np.array([r['pose_desired'] for r in trace])
        p=np.linalg.norm(actual[:,:3,3]-desired[:,:3,3],axis=1)
        a=Rotation.from_matrix(np.transpose(desired[:,:3,:3],(0,2,1))@actual[:,:3,:3]).magnitude()
        trace_path=outdir/f'{name}_trace.npz'
        np.savez_compressed(trace_path,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
        (outdir/f'{name}_native_config.txt').write_text(bridge.effective_native)
        row=dict(name=name,q_seed=np.asarray(seed).tolist(),same_time_max_position_m=float(p.max()),
            same_time_max_rotation_rad=float(a.max()),last_same_time_position_m=float(p[-1]),
            profile=bridge._state_fingerprint(),result=ans,
            trace_sha256=sha(trace_path),wall_s=time.perf_counter()-tick)
        result['cases'].append(row)
        (outdir/'RESULTS.json').write_text(json.dumps(clean_json(result),indent=2,allow_nan=False))
        print(name,'max_mm',p.max()*1000,'max_deg',np.rad2deg(a.max()),
              'end_mm',p[-1]*1000,'wall_s',row['wall_s'],flush=True)

"""Causal offline removal of one software guard, with physical limits intact."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qp_reference_followup'))
import run_reference_protocol as old


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class LeadBridge(old.replay.QPReplayBridge):
    def make_controller(self):
        kin,inner,cfg=super().make_controller()
        self.last_cfg=cfg
        update=inner.update
        self.native_rows=[]
        def record_update(*a,**kw):
            step=update(*a,**kw)
            o=inner._native._out[0]
            self.native_rows.append({key:o[key].copy().tolist() for key in
                ['qp1_status','qp2_status','qp1_iter','qp2_iter','n_cbf_active',
                 'qdot_qp_vs_sent_max','task_progress_alpha','hard_residual_max','equality_residual_max']})
            return step
        inner.update=record_update
        return kin,inner,cfg


def decompose(T,trace):
    a=np.asarray([r['pose_actual'][:3,3] for r in trace])
    d=np.asarray([r['pose_desired'][:3,3] for r in trace])
    p=T[:,:3,3]; v=np.diff(p,axis=0);l=np.linalg.norm(v,axis=1)
    u=np.clip(((d[:,None,:]-p[:-1])*v).sum(axis=2)/l**2,0,1)
    projected=p[:-1][None]+u[:,:,None]*v[None]
    idx=np.linalg.norm(d[:,None]-projected,axis=2).argmin(axis=1)
    tangent=v[idx]/l[idx,None]
    e=d-a; along=(e*tangent).sum(axis=1)
    normal=np.sqrt(np.maximum(0,(e*e).sum(axis=1)-along**2))
    return dict(normal_max_m=float(normal.max()),normal_p95_m=float(np.percentile(normal,95)),
        along_abs_max_m=float(np.abs(along).max()),along_abs_p95_m=float(np.percentile(np.abs(along),95)))


if __name__=='__main__':
    data=np.load(old.INPUT,allow_pickle=False)
    old.replay.WorldPolylineReference=old.Analytic6DReference
    out=dict(protocol_sha256=sha(HERE/'LEAD_ABLATION_PROTOCOL.txt'),code_sha256=sha(__file__),
        input_sha256=sha(old.INPUT),replay_sha256=sha(old.replay.__file__),runs=[])
    for path in (0,1):
        for degrees in (6.,0.):
            name=f'path{path}_lead{int(degrees)}'
            bridge=LeadBridge('native','urdf',native_clock='deterministic')
            bridge.raw['qpik']['hard_limits']['command_lead_arm_deg']=degrees
            T,q=data[f'branch4_path{path}_T'],data[f'branch4_path{path}_q_seed']
            raw=bridge.query_pose_path(T,q,speed_m_s=.02,settle_s=2.,return_trace=True,
                return_state=False,provenance=f'fixed lead guard causal diagnostic {name}')
            trace=raw.pop('trace')
            old.summarize(raw,trace)
            raw.update(decompose(T,trace))
            raw.update(case=name,effective_resync_err_rad=bridge.last_cfg.resync_err_rad)
            assert abs(bridge.last_cfg.resync_err_rad-np.deg2rad(degrees))<1e-12
            target=HERE/f'{name}_trace.npz'
            np.savez_compressed(target,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
            native=HERE/f'{name}_native.json'
            native.write_text(json.dumps(bridge.native_rows,indent=2))
            raw['trace_sha256']=sha(target);raw['native_sha256']=sha(native)
            out['runs'].append(raw)
            (HERE/'lead_ablation_results.json').write_text(json.dumps(old.clean_json(out),indent=2,allow_nan=False))
            print(name,'maxmm',1000*raw['full_position_max_m'],'normalmm',1000*raw['normal_max_m'],
                'maxdeg',raw['full_rotation_max_deg'],'pose',raw['strict_pose_max_gate'],
                'mechanical',raw['strict_mechanical_gate'],flush=True)

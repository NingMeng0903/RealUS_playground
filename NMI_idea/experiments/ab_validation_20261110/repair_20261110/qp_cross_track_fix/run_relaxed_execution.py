"""Relax software lead cap while retaining physical limits and consistent6D timing."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qp_terminal_retime'))
import run_terminal_retime as terminal
import run_lead_ablation as lead


class Symmetric08(terminal.SymmetricReference):
    def __init__(self,*a,**kw):
        kw['ramp_s']=.8
        super().__init__(*a,**kw)


if __name__=='__main__':
    old=lead.old
    data=np.load(old.INPUT,allow_pickle=False)
    out=dict(protocol_sha256=lead.sha(HERE/'RELAXED_EXECUTION_PROTOCOL.txt'),
        code_sha256=lead.sha(__file__),input_sha256=lead.sha(old.INPUT),runs=[])
    for path in (0,1):
        for label,degrees,ref in [('lead20_start04',20.,old.Analytic6DReference),
                                 ('lead20_both08',20.,Symmetric08),
                                 ('lead0_both08',0.,Symmetric08)]:
            old.replay.WorldPolylineReference=ref
            name=f'path{path}_{label}'
            bridge=lead.LeadBridge('native','urdf',native_clock='deterministic')
            bridge.raw['qpik']['hard_limits']['command_lead_arm_deg']=degrees
            T,q=data[f'branch4_path{path}_T'],data[f'branch4_path{path}_q_seed']
            raw=bridge.query_pose_path(T,q,speed_m_s=.02,settle_s=2.,return_trace=True,
                return_state=False,provenance=f'predeclared relaxed execution {name}')
            trace=raw.pop('trace')
            old.summarize(raw,trace);raw.update(lead.decompose(T,trace))
            raw.update(case=name,effective_lead_rad=bridge.last_cfg.resync_err_rad)
            p=HERE/f'{name}_trace.npz'
            np.savez_compressed(p,**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
            raw['trace_sha256']=lead.sha(p)
            native=HERE/f'{name}_native.json';native.write_text(json.dumps(bridge.native_rows,indent=2))
            raw['native_sha256']=lead.sha(native)
            out['runs'].append(raw)
            (HERE/'relaxed_execution_results.json').write_text(json.dumps(old.clean_json(out),indent=2,allow_nan=False))
            print(name,'maxmm',1000*raw['full_position_max_m'],'normalmm',1000*raw['normal_max_m'],
                'maxdeg',raw['full_rotation_max_deg'],'pose',raw['strict_pose_max_gate'],
                'mechanical',raw['strict_mechanical_gate'],flush=True)

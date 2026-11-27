"""Fixed composition of independently tested, unchanged timing components."""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'qp_governor_followup'))
sys.path.insert(0,str(HERE.parent/'qp_terminal_retime'))
import run_governor as gov
import run_terminal_retime as end

if __name__=='__main__':
    inputs=np.load(gov.old.INPUT,allow_pickle=False)
    gov.old.Analytic6DReference=end.SymmetricReference
    result=dict(protocol_sha256=gov.sha(HERE/'PROTOCOL.txt'),code_sha256=gov.sha(__file__),
        governor_sha256=gov.sha(gov.__file__),terminal_reference_sha256=gov.sha(end.__file__),
        input_sha256=gov.sha(gov.old.INPUT),runs=[])
    for path in (0,1):
        metrics,trace=gov.run_case(inputs[f'branch4_path{path}_T'],inputs[f'branch4_path{path}_q_seed'],True)
        name=f'path{path}_combined'
        metrics['case']=name
        target=HERE/f'{name}_trace.npz'
        np.savez_compressed(target,**trace)
        metrics['trace_sha256']=gov.sha(target)
        result['runs'].append(metrics)
        (HERE/'RESULTS.json').write_text(json.dumps(gov.old.clean_json(result),indent=2,allow_nan=False))
        print(name,'seconds',metrics['duration_s'],'maxmm',1000*metrics['position_max_m'],
            'maxdeg',metrics['rotation_max_deg'],'pose',metrics['pose_gate'],
            'mechanical',metrics['strict_mechanical_gate'],flush=True)

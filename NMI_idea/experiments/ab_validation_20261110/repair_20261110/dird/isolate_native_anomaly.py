"""First-invalid-tick capture on the unchanged offending frozen inputs."""
import argparse
import json
import numpy as np
from common import BASELINE, save
from qp_replay import QPReplayBridge


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',default='native')
    a=p.parse_args()
    bridge=QPReplayBridge(a.backend,'urdf')
    seed=np.asarray(json.load(open('initialization_sensitivity_seed.json'))['q_seed'])
    paths=np.load(BASELINE/'dird/trajectory_paths.npz')
    results=[]
    for method,index in [('no_dird',1),('joint_guided',3),('joint_guided',5)]:
        out=bridge.query_pose_path(paths[method+'_T_tcp_railbase'][index],seed,
            provenance=f'isolated unchanged offending {method}/{index}',return_trace=True,return_state=False)
        out['method'],out['proposal']=method,index
        results.append(out)
        save(f'isolated_anomaly_{a.backend}.json',{'backend':a.backend,'results':results})
        print(method,index,'steps',out.get('n_steps'),'first failures',
            [(r.get('tick'),r.get('violations')) for r in out['failures']],flush=True)


if __name__=='__main__':main()

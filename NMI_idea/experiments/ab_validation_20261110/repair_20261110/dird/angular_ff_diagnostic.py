"""Separate reference-interface experiment; no production module is edited.

Same frozen geometric poses, same timing/initial seeds. Only the offline
reference wrapper supplies SO(3) angular feedforward from production Slerp.
This is not the production outer-loop result and never supplies B graph state.
"""
from dataclasses import replace
import json
import numpy as np
from scipy.spatial.transform import Rotation
from common import BASELINE, save
import qp_replay as replay

ProductionReference = replay.WorldPolylineReference


class ReferenceWithAngularFF(ProductionReference):
    def sample(self, t_s):
        mid = super().sample(t_s)
        h = 1e-5
        lo, hi = super().sample(t_s - h), super().sample(t_s + h)
        Rlo = Rotation.from_euler(self.euler_order, lo.pose_d[3:]).as_matrix()
        Rhi = Rotation.from_euler(self.euler_order, hi.pose_d[3:]).as_matrix()
        angular = Rotation.from_matrix(Rhi @ Rlo.T).as_rotvec() / (2 * h)
        vel = mid.vel_ff.copy()
        vel[3:] = angular
        return replace(mid, vel_ff=vel)


def main():
    replay.WorldPolylineReference = ReferenceWithAngularFF  # owned harness only
    bridge = replay.QPReplayBridge('native', 'urdf')
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    sensitivity = json.load(open('initialization_sensitivity_seed.json'))
    path = np.load(BASELINE / 'dird/trajectory_paths.npz')['no_dird_T_tcp_railbase'][0]
    result = {'scope': 'separate diagnostic wrapper around actual production Slerp; production outer and QP sources untouched; not main12 or B graph',
        'feedforward': 'base/world SO(3) central difference h=1e-5 s of actual WorldPolylineReference.sample',
        'same_frozen_pose_path': 'no_dird proposal0', 'speed_m_s': .02, 'settle_s': 2.,
        'cases': {}}
    for name, seed in [('frozen_branch0', nominal['q_seed']), ('production_branch4', sensitivity['q_seed'])]:
        out = bridge.query_pose_path(path, seed, return_trace=True, return_state=False,
            provenance='independent angular-feedforward diagnostic, unchanged frozen path0')
        trace = out.pop('trace')
        result['cases'][name] = out
        result['cases'][name]['trace'] = trace
        save('angular_ff_diagnostic.json', result)
        print(name, 'posp95', out['position_error_p95_m'], 'rotp95', out['rotation_error_p95_rad'],
            'tracking_gate', out['motion_tracking_pass_2mm_2deg'], flush=True)


if __name__ == '__main__':
    main()

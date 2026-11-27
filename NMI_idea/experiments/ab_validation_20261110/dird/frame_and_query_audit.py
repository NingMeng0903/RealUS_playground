"""Real Pinocchio frame references, fixed branch contexts and query timing."""
import json
import time
import numpy as np
import pinocchio as pin
import torch
from adapter import CapabilityBridge
from benchmark import HERE, save
from ird_playground.region.operator import base_from_rail_torch


def main():
    b = CapabilityBridge()
    nominal = json.loads((HERE / 'nominal_context.json').read_text())
    seed = np.asarray(nominal['q_seed'])
    T = np.asarray(nominal['T_tcp_railbase'])
    refs = []
    for rail in [0., seed[0], .8]:
        q = seed.copy()
        q[0], q[1] = rail, 0.
        pin.forwardKinematics(b.kin.model, b.kin.data, q)
        jointid = b.kin.model.getJointId('joint_1')
        actual = b.kin.data.oMi[jointid].homogeneous.copy()
        predicted = base_from_rail_torch(torch.tensor(rail, dtype=b.dtype),
            torch.eye(4, dtype=b.dtype), b.axis0).numpy()
        refs.append({'rail_y_raw_m': rail, 'predicted_axis': predicted,
                     'pinocchio_joint1_placement_at_q1zero': actual,
                     'max_abs_error': float(np.abs(predicted - actual).max())})
    # Public placement perturbations, no observation outcomes and no selection.
    queries = []
    for dy in [0., -.04, .04]:
        tcp = np.repeat(T[None], 9, axis=0)
        tcp[:, 1, 3] += np.linspace(0, dy, 9)
        rr = np.full(9, seed[0])
        query = b.query_path(tcp, rr, seed,
            provenance='frame audit only: public fixed-orientation Y interpolation, not a claimed DMP path')
        queries.append({'translation_dy_m': dy, 'query': query})
    timings = []
    for _ in range(20):
        query = b.query_path(np.repeat(T[None], 3, axis=0), np.full(3, seed[0]), seed)
        timings.append(query['seconds'])
    save('frame_query_audit.json', {'axis_references': refs,
        'common_se3': 'see gradients.json: arbitrary rotation and translation applied jointly',
        'public_context_queries': queries, 'nominal_3point_warm_query_timings': timings,
        'nominal_3point_median_total_seconds': float(np.median([x['total'] for x in timings])),
        'nominal_3point_p95_total_seconds': float(np.percentile([x['total'] for x in timings], 95))})
    print('axis errors', [x['max_abs_error'] for x in refs])
    print('placement sampled-valid', [(x['translation_dy_m'], x['query']['exact']['validated']) for x in queries])
    print('warm total p50/p95', np.percentile([x['total'] for x in timings], [50, 95]).tolist())


if __name__ == '__main__':
    main()

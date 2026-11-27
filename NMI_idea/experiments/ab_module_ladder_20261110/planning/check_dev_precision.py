"""Prespecified DEV sensitivity of sampled backup and compact future proposals.

No locked test world is instantiated. Parameters remain frozen regardless of
these results. This measures sensitivity, not convergence or exact MPC quality.
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from system import ROOT, Planner, Robot, CapabilityEvaluator, env
from run_experiment import contexts


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    output = Path(__file__).parent / 'dev_precision'
    output.mkdir(exist_ok=False)
    particles = env.sample_prior(720501, 16)
    evaluator = CapabilityEvaluator()
    scene = contexts(evaluator)[0]
    q0 = scene['q_seed']
    initial = dict(name='initial', pose=np.zeros(3), history=[],
                   weights=np.ones(16) / 16, q=q0.copy(), remaining=60.)
    saved = json.loads((Path(__file__).parent / 'dev_resource_compact/results.json').read_text())
    row = next(r for r in saved['rows'] if r['object_index_for_audit_only'] == 0
               and r['method'] == 'A_continuous_servo')
    event = row['events'][0]
    after = dict(name='after_first_public_block', pose=np.array(event['actual_pose']),
                 history=[(f, np.array(event['actual_pose'])) for f in event['public']],
                 weights=np.array(event['posterior']), q=np.array(event['q_kin']),
                 remaining=60. - event['clock_s'])
    out = dict(scope='DEV sensitivity only; no locked test; no parameter selection',
               dev_seed=720501, particles=16, variants=[], comparisons=[],
               source_sha256={str(p.relative_to(ROOT)): fingerprint(p) for p in
                              [Path(__file__), Path(__file__).parent/'system.py',
                               Path(__file__).parent/'PROTOCOL.json',
                               ROOT/'observation/environment.py', ROOT/'observation/servo.py']})
    for state in [initial, after]:
        local = []
        for samples, future in [(8, 'compact'), (32, 'compact'),
                                (64, 'compact'), (32, 'same_full')]:
            robot = Robot(evaluator, scene['T_world_body'], q0)
            planner = Planner(particles, 'B_two_step', robot,
                              samples=samples, future_mode=future)
            started = time.perf_counter()
            decision = planner.decide(state['pose'], state['history'], state['weights'],
                                      state['q'], remaining_time=state['remaining'],
                                      remaining_actions=6 if state['name']=='initial' else 5)
            result = dict(state=state['name'], samples=samples, future= future,
                          selected=decision.pose.tolist(), value=decision.value,
                          kind=decision.diagnostics['kind'],
                          root_values=decision.diagnostics.get('root_candidates', []),
                          wall_s=time.perf_counter()-started,
                          endpoint_queries=robot.queries,
                          geometry_queries=planner.response.geometry_queries)
            local.append(result); out['variants'].append(result)
            (output/'RESULTS.json').write_text(json.dumps(out, indent=2, allow_nan=False))
            print(state['name'], samples, future, 'action', result['selected'],
                  'value', result['value'], 'wall', result['wall_s'], flush=True)
        reference = local[1]
        for result in local:
            ra={tuple(r['action']): r['value'] for r in reference['root_values']}
            rb={tuple(r['action']): r['value'] for r in result['root_values']}
            assert ra.keys() == rb.keys()
            out['comparisons'].append(dict(state=state['name'],
                reference='32_compact', variant=f"{result['samples']}_{result['future']}",
                selected_action_changed=reference['selected'] != result['selected'],
                max_root_value_abs_difference=max(abs(ra[a]-rb[a]) for a in ra),
                selected_value_difference=result['value']-reference['value']))
        (output/'RESULTS.json').write_text(json.dumps(out, indent=2, allow_nan=False))
    print(json.dumps(out['comparisons'], indent=2), flush=True)

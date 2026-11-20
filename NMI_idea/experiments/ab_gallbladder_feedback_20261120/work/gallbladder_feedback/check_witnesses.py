"""Verify reachable strict geometry on the frozen end-to-end evaluation roots."""

from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np

from engine import Scanner, SensorExit, endpoint, prior_run, station_cine
from acquire import CONDITIONS, HOLD_FRAMES, MIN_HOLD_FRAMES, execute, fit_anatomy
from witnesses import find_witness
from train import OUT, save_json


def witness_one(seed):
    wide = Scanner(seed, CONDITIONS[0], 120)
    plan, target, metric, tested, count = find_witness(wide.scene, wide.prefix)
    witness = {'seed': seed, 'reachable': plan is not None, 'strict_pose_count': count, 'tested': tested}
    if plan is None:
        return witness
    witness.update({'target': target.tolist(), 'center_mm': metric.center, 'axis_deg': metric.angle,
                    'on_evaluation_tapes': {}})
    for i, profile in enumerate(CONDITIONS):
        scanner = wide if i == 0 else Scanner(seed, profile, 120+i)
        scene = scanner.scene
        history = list(scanner.initial.history)
        arrived = execute(scene, scanner.ctrl, scanner.tape, scanner.prefix, plan, None, history, 'offline_witness')
        assert np.max(np.abs(arrived.q-target)) < 1.e-7 and np.max(np.abs(arrived.v)) < 1.e-7
        fitted, _, _ = fit_anatomy(scene, arrived, history, None)
        anatomy = scene.body.anatomy(fitted)
        point = SensorExit(arrived, tuple(history))
        first = None
        for k in range(3):
            if scanner.deadline - point.state.tick < MIN_HOLD_FRAMES or point.accepted is not None:
                break
            point = station_cine(scene, scanner.ctrl, scanner.tape, point, anatomy, scanner.deadline)
            if k == 0:
                first = endpoint(scene, point, scanner.tape, scanner.prefix, scanner.deadline)
        result = endpoint(scene, point, scanner.tape, scanner.prefix, scanner.deadline)
        witness['on_evaluation_tapes'][profile] = {'first_cine': first, 'up_to_three_cines': result}
    return witness


if __name__ == '__main__':
    freeze = json.loads((OUT/'freeze.json').read_text())
    assert freeze['test_noise_streams'] == [120, 121]
    with ProcessPoolExecutor(max_workers=8, initializer=prior_run.initialize) as pool:
        roots = list(pool.map(witness_one, range(*freeze['test_anatomies'])))
    tapes = [out for root in roots for out in root.get('on_evaluation_tapes', {}).values()]
    summary = {'anatomies': len(roots), 'reachable': sum(r['reachable'] for r in roots),
               'first_cine_correct': sum(bool(r['first_cine'] and r['first_cine']['correct']) for r in tapes),
               'correct_max_three_cines': sum(r['up_to_three_cines']['correct'] for r in tapes),
               'wrong_max_three_cines': sum(r['up_to_three_cines']['wrong'] for r in tapes)}
    save_json(OUT/'heldout_witnesses.json', {'summary': summary, 'anatomies': roots})
    print(json.dumps(summary), flush=True)

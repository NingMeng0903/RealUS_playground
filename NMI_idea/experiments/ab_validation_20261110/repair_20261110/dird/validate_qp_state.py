"""Actual native/Python persistent-vs-restored prefix and observation hold."""
import argparse
import json
import numpy as np
from common import BASELINE, save
from qp_replay import QPReplayBridge, QPReplaySession


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', default='native')
    a = p.parse_args()
    bridge = QPReplayBridge(backend=a.backend, tcp_mode='urdf')
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    q = np.asarray(nominal['q_seed'])
    T = np.asarray(nominal['T_tcp_railbase'])
    path1 = np.repeat(T[None], 5, axis=0)
    path1[:, 0, 3] += np.linspace(0, .002, 5)
    path2 = np.repeat(path1[-1][None], 5, axis=0)
    path2[:, 1, 3] += np.linspace(0, .002, 5)
    with QPReplaySession(bridge, q) as persistent:
        first = persistent.step_path(path1, settle_s=0., return_trace=True)
        held = persistent.hold(.4, return_trace=True)
        state = persistent.snapshot()
        direct = persistent.step_path(path2, settle_s=0., return_trace=True)
    save(f'qp_state_prefix_diagnostic_{a.backend}.json', {
        'first_fallback_counts': {str(x): sum(r['fallback_reason'] == x for r in first['trace']) for x in set(r['fallback_reason'] for r in first['trace'])},
        'hold_fallback_counts': {str(x): sum(r['fallback_reason'] == x for r in held['trace']) for x in set(r['fallback_reason'] for r in held['trace'])},
        'first_trace': first['trace'], 'held_trace': held['trace'], 'saved_state': state})
    restored = bridge.query_pose_path(path2, state['q_exit'], settle_s=0.,
        initial_state=json.loads(json.dumps(state)), return_trace=True)
    err_q = float(np.max(np.abs(np.asarray(direct['q_exit']) - np.asarray(restored['q_exit']))))
    err_v = float(np.max(np.abs(np.asarray(direct['qdot_exit']) - np.asarray(restored['qdot_exit']))))
    trace_q_error = float(np.max(np.abs(np.asarray([x['q'] for x in direct['trace']]) - np.asarray([x['q'] for x in restored['trace']]))))
    fingerprint_reject = False
    bad = json.loads(json.dumps(state))
    bad['fingerprint']['config_sha256'] = 'deliberately different'
    try:
        bridge.query_pose_path(path2, bad['q_exit'], initial_state=bad, return_trace=False)
    except ValueError:
        fingerprint_reject = True
    result = {'backend': a.backend, 'tcp_mode': 'urdf',
        'scope': 'real actual-update replay restores every internal state through complete prefix, not private subset; no hardware',
        'prefix_motion_steps': first['n_steps'], 'observation_hold_steps': held['n_steps'],
        'observation_hold_integrated_s': held['integrated_duration_s'],
        'state_roundtrip_json': True, 'saved_state': state,
        'prefix_reconstruction_max_abs_error': restored['prefix_reconstruction_max_abs_error'],
        'same_next_path_exit_q_max_abs_error': err_q,
        'same_next_path_exit_qdot_max_abs_error': err_v,
        'same_next_path_entire_q_trace_max_abs_error': trace_q_error,
        'source_config_fingerprint_mismatch_rejected': fingerprint_reject,
        'passed': err_q < 1e-7 and err_v < 1e-7 and trace_q_error < 1e-7 and fingerprint_reject and held['n_steps'] == 80}
    save(f'qp_state_validation_{a.backend}.json', result)
    print(result['passed'], err_q, err_v, trace_q_error, 'hold steps', held['n_steps'], flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()

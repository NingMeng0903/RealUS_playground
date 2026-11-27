"""Independent full-history oracles for path/hold acceleration and jerk."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import BASELINE, clean_json
from qp_replay import QPReplayBridge, QPReplaySession, command_derivatives


def write(name, result):
    (HERE / name).write_text(json.dumps(clean_json(result), indent=2, allow_nan=False))


def independent_global_difference(segments, dt):
    """Directly concatenate all actual command rows before any differences."""
    velocities = np.concatenate(segments)
    acceleration = np.diff(np.vstack((np.zeros(8), velocities)), axis=0) / dt
    jerk = np.diff(np.vstack((np.zeros(8), acceleration)), axis=0) / dt
    return acceleration, jerk


def test_nonzero_velocity_and_acceleration_with_known_constant_jerk():
    dt = .005
    v0 = np.linspace(-.07, .09, 8)
    a0 = np.linspace(.2, -.3, 8)
    j0 = np.linspace(-2., 3., 8)
    expected_a, velocities = [], []
    a, v = a0.copy(), v0.copy()
    for _ in range(11):
        a = a + dt * j0
        v = v + dt * a
        expected_a.append(a.copy())
        velocities.append(v.copy())
    aa, jj = command_derivatives(velocities, dt, v0, a0)
    np.testing.assert_allclose(aa, expected_a, rtol=0, atol=2e-15)
    np.testing.assert_allclose(jj, np.repeat(j0[None], len(velocities), axis=0), rtol=0, atol=1e-12)


def test_stationary_root_statistics_are_bitwise_unchanged():
    velocities = np.random.default_rng(20261133).normal(size=(37, 8))
    old_a = np.diff(np.vstack((np.zeros(8), velocities)), axis=0) / .005
    old_j = np.diff(np.vstack((np.zeros(8), old_a)), axis=0) / .005
    aa, jj = command_derivatives(velocities, .005)
    assert np.array_equal(aa, old_a)
    assert np.array_equal(jj, old_j)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_frozen_actual_motion_hold_trace_matches_independent_audit(backend):
    frozen = json.loads((HERE.parent / f'qp_state_prefix_diagnostic_{backend}.json').read_text())
    first = np.asarray([r['qdot'] for r in frozen['first_trace']])
    hold = np.asarray([r['qdot'] for r in frozen['held_trace']])
    a_full, j_full = independent_global_difference([first, hold], .005)
    aa, jj = command_derivatives(hold, .005, first[-1], (first[-1] - first[-2]) / .005)
    assert np.array_equal(aa, a_full[len(first):])
    assert np.array_equal(jj, j_full[len(first):])
    audit = json.loads((HERE.parent.parent / 'audit' / f'continuation_derivatives_{backend}.json').read_text())
    np.testing.assert_allclose(aa[0], audit['first_command_acceleration_corrected'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(jj[0], audit['first_command_jerk_corrected'], rtol=0, atol=1e-12)
    # The defect is visible in old data; do not hide the actual residual jerk.
    old_a, old_j = command_derivatives(hold, .005)
    assert np.max(np.abs(old_a - aa)) > 1.
    assert np.max(np.abs(old_j - jj)) > 100.
    write(f'frozen_boundary_{backend}.json', {
        'prefix_rows': len(first), 'hold_rows': len(hold),
        'velocity_entry': first[-1], 'acceleration_entry': (first[-1] - first[-2]) / .005,
        'first_corrected_acceleration': aa[0], 'first_corrected_jerk': jj[0],
        'max_old_acceleration_statistic_error': np.max(np.abs(old_a - aa)),
        'max_old_jerk_statistic_error': np.max(np.abs(old_j - jj)),
        'global_concat_acceleration_max_abs_error': np.max(np.abs(aa - a_full[len(first):])),
        'global_concat_jerk_max_abs_error': np.max(np.abs(jj - j_full[len(first):])),
        'corrected_acceleration': aa, 'corrected_jerk': jj,
        'independent_audit_first_derivatives_match': True})


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_real_path_hold_path_and_json_restore_propagate_both_histories(backend):
    bridge = QPReplayBridge(backend=backend, tcp_mode='urdf')
    nominal = json.loads((BASELINE / 'dird/nominal_context.json').read_text())
    root_q = np.asarray(nominal['q_seed'])
    T = np.asarray(nominal['T_tcp_railbase'])
    path1 = np.repeat(T[None], 5, axis=0)
    path1[:, 0, 3] += np.linspace(0, .002, 5)
    path2 = np.repeat(path1[-1][None], 5, axis=0)
    path2[:, 1, 3] += np.linspace(0, .002, 5)
    with QPReplaySession(bridge, root_q) as session:
        first = session.step_path(path1, settle_s=0., return_trace=True)
        held = session.hold(.4, return_trace=True)
        state = session.snapshot()
        direct = session.step_path(path2, settle_s=0., return_trace=True)
    assert all(not r['failures'] for r in [first, held, direct])
    assert held['n_steps'] == 80
    velocity_segments = [np.asarray([r['qdot'] for r in x['trace']]) for x in [first, held, direct]]
    a_full, j_full = independent_global_difference(velocity_segments, .005)
    offset = 0
    prev_v, prev_a = np.zeros(8), np.zeros(8)
    for result, vel in zip([first, held, direct], velocity_segments):
        aa, jj = a_full[offset:offset+len(vel)], j_full[offset:offset+len(vel)]
        np.testing.assert_array_equal(result['qdot_entry'], prev_v)
        np.testing.assert_array_equal(result['acceleration_entry'], prev_a)
        np.testing.assert_array_equal([r['command_acceleration'] for r in result['trace']], aa)
        np.testing.assert_array_equal([r['command_jerk'] for r in result['trace']], jj)
        np.testing.assert_array_equal(result['acceleration_max_abs_by_coordinate'], np.abs(aa).max(axis=0))
        np.testing.assert_array_equal(result['jerk_max_abs_by_coordinate'], np.abs(jj).max(axis=0))
        assert result['acceleration_margin_min'] == np.min(result['acceleration_limit_by_coordinate'] - np.abs(aa))
        assert result['jerk_margin_min'] == np.min(result['jerk_limit_by_coordinate'] - np.abs(jj))
        np.testing.assert_array_equal(result['acceleration_exit'], aa[-1])
        prev_v, prev_a = vel[-1], aa[-1]
        offset += len(vel)
    assert state['schema'] == 'ultrapoc_qp_complete_history_v2'
    np.testing.assert_array_equal(state['acceleration_exit'], held['acceleration_exit'])
    restored = bridge.query_pose_path(path2, state['q_exit'], settle_s=0.,
        initial_state=json.loads(json.dumps(state)), return_trace=True)
    assert restored['prefix_reconstruction_max_abs_error'] == {'q': 0., 'qdot': 0., 'acceleration': 0.}
    for key in ['q', 'qdot', 'command_acceleration', 'command_jerk']:
        np.testing.assert_array_equal([r[key] for r in direct['trace']], [r[key] for r in restored['trace']])
    tampered = json.loads(json.dumps(state))
    tampered['acceleration_exit'][0] += .01
    with pytest.raises(ValueError, match='reconstruction mismatch'):
        bridge.query_pose_path(path2, tampered['q_exit'], initial_state=tampered, return_trace=False)
    write(f'actual_continuation_{backend}.json', {
        'scope': 'fresh short actual-controller trace; no claim about native long-path stability',
        'entry_state_schema': state['schema'], 'root': first, 'hold': held, 'next_direct': direct,
        'next_restored': restored,
        'global_concatenation_derivatives_match_exactly': True,
        'restored_q_velocity_acceleration_jerk_match_exactly': True,
        'tampered_acceleration_state_rejected': True})

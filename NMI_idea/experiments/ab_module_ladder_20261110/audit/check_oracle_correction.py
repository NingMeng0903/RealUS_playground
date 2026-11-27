"""Independent posthoc oracle audit. No actor, IK, or native execution."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from observation import environment as env
from observation.servo import template_candidates

OD = ROOT / 'continuous_mismatch/oracle_diagnostic'
CORRECTED = OD / 'run_corrected_parameterization_20261110/RESULTS.json'
ORIGINAL = OD / 'run_20261110/RESULTS.json'
LEGAL = ROOT / 'continuous_mismatch/formal_run_20261110/RESULTS.json'
FINITE = ROOT / 'planning/test_frozen/results.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

expected_hashes = {
    CORRECTED: '70b423baaf48f5e15a0a984f55a908366e8edcc57f369ebe60fbc6ba4a464165',
    ORIGINAL: '7f3b71a8629c2dd599b7851eb85c3ff5c288fb584a9bc7ff493fd9d585e3b9af',
    LEGAL: '6da6319338367651d235a9c82b245dedea987888484ee02e87bc87b976f4adbe',
    FINITE: '913be5af55c0d85b969e203635bdbb0979413ccfbd4e1192668bec9efd69ab85',
}
for p, expected in expected_hashes.items():
    assert sha(p) == expected, p
data = json.loads(CORRECTED.read_text())
old = json.loads(ORIGINAL.read_text())
legal = json.loads(LEGAL.read_text())
assert data['status'] == 'complete' and data['ORACLE_LEAKED_TRUTH']
assert data['original_oracle_results_sha256'] == sha(ORIGINAL)
assert data['legal_results_sha256'] == sha(LEGAL)
assert data['legal_public_prior_sha256'] == legal['public_prior_sha256']

source_paths = {
    'correction_protocol_sha256': OD / 'CORRECTION.txt',
    'correction_runner_sha256': OD / 'run_corrected_parameterization_20261110.py',
    'original_oracle_runner_sha256': OD / 'run_oracle_diagnostic.py',
    'original_oracle_protocol_sha256': OD / 'PROTOCOL.txt',
    'environment_sha256': ROOT / 'observation/environment.py',
    'planning_system_sha256': ROOT / 'planning/system.py',
    'planning_protocol_sha256': ROOT / 'planning/PROTOCOL.json',
}
for k, p in source_paths.items():
    assert sha(p) == data['source_hashes'][k], k

FIELDS = ('registration', 'modes', 'y_controls', 'z_controls', 'depths',
          'identity_z', 'detect_z', 'points')

def world_hash(worlds):
    # Reproduce the declared byte-hash serialization, not the worker helper.
    h = hashlib.sha256()
    for i, w in enumerate(worlds):
        h.update(f'world:{i}\n'.encode())
        for key in FIELDS:
            a = np.ascontiguousarray(getattr(w, key))
            h.update(key.encode())
            h.update(str(a.dtype).encode())
            h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
            h.update(a.tobytes())
    return h.hexdigest()

def make_shift(w):
    reg = w.registration.copy()
    reg[1] += .008
    modes = w.modes.copy()
    modes[:, 1] += .4
    return env._world(reg, modes, w.y_controls.copy(), w.z_controls.copy(),
                      w.depths.copy(), w.identity_z.copy(), w.detect_z.copy())

def canonical(w):
    modes = w.modes.copy()
    modes[:, 1] = np.sign(w.modes[:, 1])
    delta = w.modes[:, 1] - modes[:, 1]
    controls = w.y_controls + .25 * delta[:, None] * env.CONTROL_X[None, :]
    return env._world(w.registration.copy(), modes, controls, w.z_controls.copy(),
                      w.depths.copy(), w.identity_z.copy(), w.detect_z.copy())

# These are audit-only reconstitutions of already-sealed private worlds. They
# never enter any legal actor. No worker correction/verification helper is used.
iid = env.sample_worlds(seed=720380, n=8, split='test')
worlds = {'iid': iid, 'shifted': [make_shift(w) for w in iid]}
gt_hashes = {c: world_hash(ws) for c, ws in worlds.items()}
assert gt_hashes == data['AUDIT_ONLY']['private_gt_sha256']
assert gt_hashes == old['AUDIT_ONLY']['private_gt_sha256']
assert gt_hashes == legal['audit_private']['private_gt_sha256']
grid = np.array([[x, y, a] for x in np.linspace(-.04, .04, 9)
                 for y in np.linspace(-.04, .04, 9)
                 for a in np.linspace(-.8, .8, 7)])
reported = {(r['condition'], r['object_index_for_audit_only']): r
            for r in data['AUDIT_ONLY']['parameterization_audit']}
assert len(reported) == 16
equivalence = []
for c, ws in worlds.items():
    for i, w in enumerate(ws):
        h = w if c == 'iid' else canonical(w)
        assert world_hash([h]) == reported[c, i]['hypothesis_world_sha256']
        assert np.all(np.isin(h.modes[:, 1], [-1., 1.]))
        pe = float(np.max(np.abs(w.points - h.points)))
        assert pe <= 1e-14
        # Independently check the continuous spline identity between stored knots.
        x = np.linspace(-.05, .05, 1001)
        original_y = w.modes[:, 1, None] * .25 * x + CubicSpline(
            env.CONTROL_X, w.y_controls, axis=-1, bc_type='natural')(x)
        canonical_y = h.modes[:, 1, None] * .25 * x + CubicSpline(
            env.CONTROL_X, h.y_controls, axis=-1, bc_type='natural')(x)
        de = float(np.max(np.abs(original_y - canonical_y)))
        assert de <= 1e-14
        p, q = env.predict_batch([w], grid), env.predict_batch([h], grid)
        metrics = {}
        for key in ('means', 'p_source', 'p_visible', 'geometric_length', 'angle_audit'):
            error = float(np.max(np.abs(p[key] - q[key])))
            metrics[key] = error
            np.testing.assert_allclose(p[key], q[key], atol=1e-12 if key == 'p_source' else 1e-8,
                                       rtol=0 if key == 'p_source' else 1e-10)
        assert np.array_equal(p['contact'], q['contact'])
        candidates = template_candidates([h], np.ones(1), np.zeros(3))
        assert len(candidates) > 0
        old_candidates = template_candidates([w], np.ones(1), np.zeros(3))
        if c == 'shifted':
            assert len(old_candidates) == 0
        equivalence.append(dict(condition=c, object=i, points_max_abs_m=pe,
            dense_1001_x_spline_max_abs_m=de, prediction_error=metrics,
            initial_template_candidates=len(candidates),
            initial_uncorrected_template_candidates=len(old_candidates)))

rows = data['rows']
by = {(r['condition'], r['object_index_for_audit_only']): r for r in rows}
assert len(rows) == len(by) == 16
assert set(by) == {(c, i) for c in ('iid', 'shifted') for i in range(8)}
assert {r['case_index'] for r in rows} == set(range(16))
public_fields = {'frame_id', 'time_s', 'visible', 'contact', 'image_u_m',
    'image_depth_m', 'visible_length_m', 'image_axis_rad', 'identity_score'}
blocks = frames = 0
max_clock_error = max_cost_error = 0.
false_stops = []
for row in rows:
    assert row['status'] == 'completed'
    assert not any(k in row for k in ('failure_type', 'failure_traceback', 'numeric_anomaly_paths'))
    assert row['ORACLE_LEAKED_TRUTH'] and row['particles_are_private_gt']
    assert row['private_gt_rendering_unchanged']
    assert row['method'] == 'A_continuous_servo' and row['context'] == 'a0_identity'
    assert row['renderer_key'] == ['continuous_mismatch', row['condition'],
                                   row['object_index_for_audit_only'], 'a0_identity']
    assert row['actions'] == len(row['events']) <= 6
    assert row['posterior'] == [1.0]
    before = 0.
    any_quality = False
    for ordinal, e in enumerate(row['events']):
        assert e['ordinal'] == ordinal
        assert e['prior_before'] == e['posterior'] == [1.0]
        assert len(e['public']) == len(e['audit_for_evaluation_only']) == 5
        assert e['q_state_kind'] == 'static endpoint realization, not native q_out'
        np.testing.assert_allclose(e['target'], e['actual_pose'], rtol=0, atol=1e-14)
        acquisition = before + e['planning_s'] + e['endpoint_s'] + e['motion_s']
        pub_ok = audit_ok = True
        for frame_i, (f, a) in enumerate(zip(e['public'], e['audit_for_evaluation_only'])):
            assert set(f) == public_fields
            assert f['frame_id'] == a['frame_id'] == f'acq-{ordinal}-frame-{frame_i}'
            assert abs(f['time_s'] - a['time_s']) < 1e-10
            assert abs(f['time_s'] - acquisition - .125 * frame_i) < 1e-8
            if not f['visible']:
                assert all(f[k] is None for k in public_fields - {'frame_id', 'time_s', 'visible', 'contact'})
            pub_ok &= bool(f['visible'] and f['contact'] and f['visible_length_m'] >= .015
                           and abs(f['image_u_m']) <= .002 and f['identity_score'] >= .75)
            true_ok = bool(a['source_is_target'] and a['contact']
                and a['geometric_length_m'] is not None and a['geometric_length_m'] >= .015
                and abs(a['true_u_m']) <= .002 and a['out_of_plane_angle_rad'] <= math.radians(10))
            assert a['passed'] == true_ok
            audit_ok &= true_ok
            frames += 1
        assert e['public_stop'] == pub_ok and e['audit_correct'] == audit_ok
        any_quality |= audit_ok
        expected = acquisition + .5 + e['belief_update_s']
        max_clock_error = max(max_clock_error, abs(expected - e['clock_s']))
        assert abs(expected - e['clock_s']) < 1e-8
        assert e['timely_public_decision'] == (e['clock_s'] <= 60 + 1e-12)
        if pub_ok and e['timely_public_decision']:
            assert ordinal == row['actions'] - 1
            assert row['correct_public_stop'] == audit_ok
            assert row['false_public_stop'] == (not audit_ok)
            if not audit_ok:
                false_stops.append(dict(condition=row['condition'],
                    object=row['object_index_for_audit_only'],
                    all_sources_target=all(a['source_is_target'] for a in e['audit_for_evaluation_only']),
                    max_angle_deg=max(math.degrees(a['out_of_plane_angle_rad']) for a in e['audit_for_evaluation_only']),
                    max_abs_u_m=max(abs(a['true_u_m']) for a in e['audit_for_evaluation_only'])))
        before = e['clock_s']
        blocks += 1
    assert row['ever_audit_quality'] == any_quality
    assert row['public_stop'] == (row['correct_public_stop'] or row['false_public_stop'])
    assert row['rmt_s'] == (row['elapsed_s'] if row['correct_public_stop'] else 60.)
    assert 0 <= row['elapsed_s'] <= 60.
    if not row['public_stop']:
        assert row['actions'] == 6
    charged = sum(e['planning_s'] + e['endpoint_s'] + e['motion_s'] + .5 + e['belief_update_s']
                  for e in row['events'])
    max_cost_error = max(max_cost_error, abs(charged - row['elapsed_s']))
    assert abs(charged - row['elapsed_s']) < 1e-8

summary = {}
for condition in ('iid', 'shifted'):
    rs = [by[condition, i] for i in range(8)]
    summary[condition] = dict(n=8, completed=8, failures=0,
        correct_public_stops=sum(r['correct_public_stop'] for r in rs),
        false_public_stops=sum(r['false_public_stop'] for r in rs),
        public_stops=sum(r['public_stop'] for r in rs),
        ever_audit_quality=sum(r['ever_audit_quality'] for r in rs),
        mean_RMT_s=float(np.mean([r['rmt_s'] for r in rs])),
        actions=sum(r['actions'] for r in rs))
    for k, v in summary[condition].items():
        assert abs(v - data['summary'][condition][k]) < 1e-12
    assert summary[condition]['correct_public_stops'] == 5
    assert summary[condition]['false_public_stops'] == 1

old_by = {(r['condition'], r['object_index_for_audit_only']): r for r in old['rows']}
# IID correction is the identity. Verify the original outcomes/actions/public
# observations survive exactly; timestamps are expected to differ with compute.
for i in range(8):
    a, b = old_by['iid', i], by['iid', i]
    for key in ('actions', 'correct_public_stop', 'false_public_stop', 'ever_audit_quality'):
        assert a[key] == b[key]
    for x, y in zip(a['events'], b['events']):
        assert x['target'] == y['target'] and x['q_kin'] == y['q_kin']
        for xs, ys in zip(x['public'], y['public']):
            assert {k: v for k, v in xs.items() if k != 'time_s'} == {k: v for k, v in ys.items() if k != 'time_s'}

logpath = OD / 'run_corrected_parameterization_20261110.log'
log = logpath.read_text()
assert 'unexpected shifted mode labels' in log and '"episodes": 16' in log and '"status": "complete"' in log
out = dict(passed=True,
    scope='Posthoc privileged diagnostic integrity, parameterization and record audit only; no actor/IK/native rerun',
    input_sha256={str(p.relative_to(ROOT)): sha(p) for p in list(expected_hashes) + list(source_paths.values()) + [logpath]},
    private_gt_sha256=gt_hashes,
    equivalence=equivalence,
    max_source_points_residual_m=max(x['points_max_abs_m'] for x in equivalence),
    max_dense_spline_residual_m=max(x['dense_1001_x_spline_max_abs_m'] for x in equivalence),
    max_prediction_mean_residual=max(x['prediction_error']['means'] for x in equivalence),
    max_prediction_p_source_residual=max(x['prediction_error']['p_source'] for x in equivalence),
    records=dict(episodes=16, blocks=blocks, frames=frames, exceptions=0,
        all_unique_cases_completed=True, max_clock_error_s=max_clock_error,
        max_cost_error_s=max_cost_error, false_stops=false_stops,
        iid_original_outcomes_actions_and_nontime_public_fields_identical=True),
    summary=summary,
    exclusion='ORACLE_LEAKED_TRUTH: never include these rows in legal A/B or treat this heuristic as an optimal upper bound',
    historical_preflight='Retained log traceback occurs in all-particle validation before first episode/write; current source refuses existing output. Original failed guard source is not independently archived; see provenance report.')
(HERE / 'oracle_correction_checks.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in out.items() if k not in ('equivalence', 'input_sha256')}, indent=2))

"""Independently replay saved atlas outcomes from robot poses and disturbance tapes."""

import collections
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

base = Path('..')
root = base / sys.argv[1]
cohorts = sys.argv[2:] or ['reference', 'minus-plus', 'plus-minus', 'beta2', 'pose25', 'transient']
if not root.exists():
    sys.exit(f'Waiting for {root}')
if (root / 'SUMMARY.json').is_file():
    cohorts = ['.']

ARMS = ('mpc_stream', 'mpc_iid', 'mpc_ou')
LOWER = np.array([-25., 3.])
UPPER = np.array([25., 52.])
ALL = {}
COMPARISONS = {}
SUMMARIES = {}
shared_cases = {}
shared_motion = {}
total_paths = total_frames = total_ticks = 0

def plane(vessels, pose, shift):
    # Work in probe millimetres. Compute independent intersections for each vessel.
    coordinates = (vessels + shift - pose[:3, 3]) @ pose[:3, :3] * 1000.
    results = np.full((2, 2), np.nan)
    for role in range(2):
        curve = coordinates[:, role]
        abscissa = curve[:,0]
        first, second = abscissa[:-1], abscissa[1:]
        index = np.where(((first <= 0.) & (second > 0.)) | ((first >= 0.) & (second < 0.)))[0]
        u = -first[index] / (second[index] - first[index])
        points = curve[index, 1:] * (1.-u[:,None]) + curve[index+1,1:] * u[:,None]
        points = points[points[:, 1] > 0.]
        if len(points):
            results[role] = points[np.argmin(points[:,1])]
    return results

def verify_packet(vessels, axes, pose, offset, bias, z, clear, noise):
    effective_shift = (axes @ (offset + [z, 0.])) * .001
    physical_shift = (axes @ (offset + [z, 0.] - bias)) * .001
    observed = plane(vessels, pose, effective_shift) + noise
    actual = plane(vessels, pose, physical_shift)[0, 0]
    roles = sorted((i for i in range(2)
                    if clear[i] and np.all(LOWER <= observed[i]) and np.all(observed[i] <= UPPER)),
                   key=lambda i: observed[i,0])
    return observed[roles], roles, actual

def recover(clear, outcomes):
    starts = np.flatnonzero(~clear & np.r_[True, clear[:-1]])
    for start in starts:
        if start < 5:
            continue
        rest = np.flatnonzero(clear[start:])
        if len(rest):
            first_good = start+rest[0]
            if first_good-start >= 3 and first_good+5 <= len(clear):
                return any(item['correct'] for item in outcomes[first_good:first_good+5]), start, first_good
    return None, None, None

def missing_reasons(bins, items):
    why = collections.Counter()
    for index in set(range(5)) - bins:
        group = [r for r in items if r['bin'] == index]
        physics = [r for r in group if r['physical']]
        visible = [r for r in physics if r['artery_visible']]
        associated = [r for r in visible if r['selected_role'] == 0]
        if not group:
            why['no_exposure'] += 1
        elif not physics:
            if any(r['plane_intersection'] for r in group):
                why['never_physically_centred'] += 1
            else:
                why['no_plane_intersection'] += 1
        elif not visible:
            why['artery_missing_when_physically_centred'] += 1
        elif not associated:
            why['artery_not_selected_on_available_centred_frame'] += 1
        else:
            why['noisy_centroid_outside_acceptance'] += 1
            assert all(not r['accepted'] for r in associated)
    return why

for cohort in cohorts:
    output = root / cohort
    if not (output / 'SUMMARY.json').is_file() or not (output / 'TRACES.npz').is_file():
        continue
    report = json.loads((output / 'SUMMARY.json').read_text())
    parsed = collections.defaultdict(list)
    with np.load(output / 'TRACES.npz', allow_pickle=False) as traces:
        path = traces['station_ref_mm']
        nominal_vessels = traces['nominal_vessels_m']
        axes = traces['offset_axes']
        assert np.all(np.isfinite(axes)) and np.all(np.isfinite(nominal_vessels))
        assert nominal_vessels.shape == (65,2,3)
        assert len(path) == 201 and np.all(np.diff(path) > 0)
        assert report['sample_dt_s'] == .02 and report['frame_dt_s'] == .1
        assert report['frames_per_complete_run'] == 41
        for case_id, case in enumerate(report['cases']):
            assert case['case'] == case_id
            z, noise, clear = (traces[f'{case_id}_{name}'] for name in ['z_mm','noise_mm','clear'])
            assert z.shape == (41,) and noise.shape == (41,2,2) and clear.shape == (41,2)
            offset = np.asarray(case['hidden_static_offset_mm'])
            bias = np.asarray(case['hidden_calibration_bias_mm'])
            if case_id not in shared_cases:
                shared_cases[case_id] = (noise, clear, offset, bias)
            else:
                for a, b in zip(shared_cases[case_id], (noise, clear, offset, bias), strict=True):
                    np.testing.assert_array_equal(a, b)
            if report['motion']['kind'] == 'ou':
                if case_id not in shared_motion:
                    shared_motion[case_id] = z
                else:
                    np.testing.assert_array_equal(shared_motion[case_id], z)
            elif report['motion']['kind'] == 'static':
                np.testing.assert_array_equal(z, np.zeros(41))
            first_q = first_pose = first_packet = None
            assert tuple(case['arms']) == ARMS
            for arm, row in case['arms'].items():
                q, fk, xv = (traces[f'{case_id}_{arm}_{key}'] for key in ('q','fk_m','x_v'))
                assert q.shape == (len(fk), 8) and fk.shape == (len(q), 4, 4) and xv.shape == (len(q), 2)
                assert np.all(np.isfinite(q)) and np.all(np.isfinite(fk)) and np.all(np.isfinite(xv))
                assert np.all(q[:,0] == report['fixed_rail_m'])
                assert np.allclose(fk[:,3], [0.,0.,0.,1.], rtol=0., atol=1e-12)
                assert -30. <= np.min(xv[:,0]) and np.max(xv[:,0]) <= 0.
                assert np.max(np.abs(xv[:,1])) <= 18. + 1e-10
                if len(xv)>1:
                    assert np.max(np.abs(np.diff(xv[:,1]))) <= 2. + 1e-10
                if first_q is None:
                    first_q, first_pose, first_packet = q[0], fk[0], row['frames'][0]['packet_mm']
                np.testing.assert_array_equal(q[0], first_q)
                np.testing.assert_array_equal(fk[0], first_pose)
                assert row['frames'][0]['packet_mm'] == first_packet
                if row['failure'] is None:
                    assert len(q) == 201 and len(row['frames']) == 41
                    assert [d['step'] for d in row['decisions']] == [0,50,100,150]
                else:
                    assert len(row['frames']) in ((len(q)-1)//5, (len(q)-1)//5+1)
                assert row['max_chart_handoff_x_ref_mm'] < 1e-9
                assert row['max_chart_handoff_v_ref_mm_s'] < 1e-9
                assert np.allclose(row['duration_s'], .02*(len(q)-1), atol=1e-10, rtol=0.)
                distance = 1000. * np.linalg.norm(np.diff(fk[:, :3, 3], axis=0), axis=1).sum()
                r = fk[:, :3, :3]
                relative = np.swapaxes(r[:-1],1,2) @ r[1:]
                skew = np.stack([relative[:,2,1]-relative[:,1,2],
                                 relative[:,0,2]-relative[:,2,0],
                                 relative[:,1,0]-relative[:,0,1]], axis=1) / 2.
                rotation = np.rad2deg(np.arctan2(np.linalg.norm(skew,axis=1),
                                                 (np.trace(relative, axis1=1,axis2=2)-1.) / 2.)).sum()
                np.testing.assert_allclose([row['tcp_mm'],row['turn_deg']], [distance,rotation], atol=1e-9, rtol=0.)
                public_bins, true_bins = set(), set()
                wrong_count = feedback_wrong = revisions = 0
                errors, items = [], []
                for frame, saved in enumerate(row['frames']):
                    assert abs(saved['t_s'] - .1*frame) < 1e-12
                    assert saved['station_ref_mm'] == path[5*frame]
                    packet, roles, physical = verify_packet(nominal_vessels,axes,fk[5*frame],offset,bias,z[frame],clear[frame],noise[frame])
                    np.testing.assert_allclose(packet, np.asarray(saved['packet_mm']).reshape(-1,2), atol=1e-10, rtol=0.)
                    assert saved['role_truth'] == roles
                    if np.isfinite(physical):
                        assert abs(saved['physical_lateral_mm'] - physical) < 1e-10
                        errors.append(physical)
                    else:
                        assert saved['physical_lateral_mm'] is None
                    p = np.array(saved['artery_probability'])
                    assert len(p) == len(roles)
                    assert np.all((p >= -1e-12) & (p <= 1.+1e-12)) and p.sum() <= 1.+1e-12
                    if len(roles) == 2:
                        assert abs(p.sum() - 1.) < 1e-12
                    slot = int(np.argmax(p)) if len(p) and max(p) >= .95 else -1
                    accepted = slot >= 0 and abs(packet[slot,0]) <= 3.
                    wrong = accepted and roles[slot] != 0
                    correct = accepted and roles[slot] == 0 and abs(physical) <= 3.
                    for key, truth in [('selected_slot',slot),('accepted',accepted),('wrong',wrong),('correct',correct)]:
                        assert saved[key] == truth, (cohort,case_id,arm,frame,key,saved[key],truth)
                    index = min(4, int(5. * (path[5*frame] - path[0]) / (path[-1]-path[0])))
                    if accepted:
                        public_bins.add(index)
                    if correct:
                        true_bins.add(index)
                    wrong_count += int(wrong)
                    feedback_wrong += int(frame % 10 != 0 and slot >= 0 and roles[slot] != 0)
                    if 'next_frame_shift_from_revision_mm' in saved:
                        assert frame % 10 != 0 and slot >= 0
                        revisions += int(saved['next_frame_shift_from_revision_mm'] >= .001)
                    items.append({'bin': index, 'physical': abs(physical)<=3., 'artery_visible': 0 in roles,
                                  'plane_intersection': np.isfinite(physical),
                                  'selected_role': roles[slot] if slot >= 0 else -1, 'accepted': accepted,
                                  'correct': correct})
                success = row['failure'] is None and len(true_bins)==5 and not wrong_count
                false_complete = len(public_bins)==5 and not success
                for key, truth in [('success',success),('wrong',bool(wrong_count)),('false_complete',false_complete),
                                   ('coverage',len(true_bins)),('revisions',revisions)]:
                    assert row[key] == truth
                reacquired, start, end = recover(clear[:,0], items)
                missed = missing_reasons(true_bins, items)
                parsed[arm].append({'case_id':case_id,'success':success,'false_complete':false_complete,'wrong_count':wrong_count,
                                    'feedback_wrong':feedback_wrong,'failure':row['failure'],
                                    'travel':distance,'turn':rotation,'duration':row['duration_s'],
                                    'mean_abs': np.mean(np.abs(errors)), 'rmse':np.sqrt(np.mean(np.square(errors))),
                                    'centred_fraction':float(np.mean(np.abs(errors)<=3.)),
                                    'no_physical_frames':len(items)-len(errors),'revisions':revisions,
                                    'recovery':reacquired, 'recovery_start':start, 'recovery_end':end,'missing':missed})
                total_paths += 1
                total_frames += len(items)
                total_ticks += len(q)
    SUMMARIES[cohort] = {
        'source_sha256': {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ('SUMMARY.json', 'TRACES.npz')},
        'arms': {},
    }
    for arm, rows in parsed.items():
        for k, our_value in [('success',np.mean([r['success'] for r in rows])),
                             ('wrong',np.mean([bool(r['wrong_count']) for r in rows])),
                             ('false_complete',np.mean([r['false_complete'] for r in rows])),
                             ('tcp_mm',np.mean([r['travel'] for r in rows])),
                             ('turn_deg',np.mean([r['turn'] for r in rows])),
                             ('duration_s',np.mean([r['duration'] for r in rows])),
                             ('revisions',np.mean([r['revisions'] for r in rows])),
                             ('run_failures',sum(r['failure'] is not None for r in rows))]:
            assert abs(our_value-report['means'][arm][k]) < 1e-9
        counts = {k: int(sum(bool(r[k]) for r in rows)) for k in ['success','false_complete','wrong_count','feedback_wrong','failure','recovery']}
        mean = {k: round(float(np.mean([r[k] for r in rows])),5) for k in ['mean_abs','rmse','centred_fraction','travel','turn','duration','revisions']}
        missing = sum((r['missing'] for r in rows), collections.Counter())
        SUMMARIES[cohort]['arms'][arm] = {
            'cases': len(rows),
            'counts': counts,
            'means': mean,
            'loss_eligible': sum(r['recovery_start'] is not None for r in rows),
            'missing_physical_exposures': sum(r['no_physical_frames'] for r in rows),
            'missed_coverage_bins': dict(missing),
            'false_completion_cases': [r['case_id'] for r in rows if r['false_complete']],
            'wrong_identity_cases': [r['case_id'] for r in rows if r['wrong_count']],
            'wrong_feedback_cases': [r['case_id'] for r in rows if r['feedback_wrong']],
            'false_completion_missed_bins': dict(sum((r['missing'] for r in rows if r['false_complete']),
                                                      collections.Counter())),
        }
        print(cohort, arm, len(rows), counts, mean, 'recovery eligible',sum(r['recovery_start'] is not None for r in rows),
              'no physical frames',sum(r['no_physical_frames'] for r in rows),'missing bins',dict(missing), flush=True)
    ALL[cohort] = parsed

primary = ['reference','minus-plus','plus-minus']
rng = np.random.default_rng(978122)
if all(name in ALL for name in primary):
    n = len(ALL['reference'][ARMS[0]])
    resample = rng.integers(0, n, size=(50000,n))
    for left,right in [(ARMS[2],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[1],ARMS[0])]:
        print('PRIMARY CLUSTER BOOTSTRAP',left,'-',right)
        comparison = {}
        COMPARISONS[f'{left}_vs_{right}'] = comparison
        for k in ['success','false_complete','mean_abs','centred_fraction','travel','turn','duration','revisions']:
            x = np.array([[float(r[k]) for r in ALL[c][left]] for c in primary])
            y = np.array([[float(r[k]) for r in ALL[c][right]] for c in primary])
            paired = x-y
            cluster = paired.mean(axis=0)
            ci = np.quantile(cluster[resample].mean(axis=1), [.025,.975])
            extra = (int(np.sum(paired==1.)),int(np.sum(paired==-1.))) if k == 'success' else ''
            comparison[k] = {'delta': float(cluster.mean()), 'clustered_ci95': ci.tolist()}
            if extra:
                comparison[k].update(left_only=extra[0], right_only=extra[1])
            print(k, round(float(cluster.mean()),6), ci, extra, flush=True)
print('Independent replay verified',total_paths,'paths,',total_ticks,'ticks and',total_frames,'exposures',flush=True)
if result_path := os.environ.get('ATLAS_SOL_REPLAY_JSON'):
    record = {'cohorts': SUMMARIES, 'primary_clustered_comparisons': COMPARISONS,
              'paths': total_paths, 'ticks': total_ticks, 'exposures': total_frames}
    Path(result_path).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')

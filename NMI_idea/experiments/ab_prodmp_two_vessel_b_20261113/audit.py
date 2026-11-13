"""Independently replay the learned ProDMP verification artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import import_module
import json
import lzma
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


HERE = Path(__file__).resolve().parent
INHERITED = Path('NMI_idea/experiments/ab_static_two_vessel_b_20261113')
EXPERIMENTS = INHERITED.parent


def read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    with lzma.open(path.with_suffix(path.suffix + '.xz'), 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle(first, second):
    relative = first.transpose(0, 2, 1) @ second
    return Rotation.from_matrix(relative).magnitude()


def cost_from_samples(q, t, fk):
    rotation = fk[:, :3, :3]
    return np.array([
        1000. * np.linalg.norm(np.diff(fk[:, :3, 3], axis=0), axis=1).sum(),
        np.rad2deg(angle(rotation[:-1], rotation[1:])).sum(),
        1000. * np.abs(np.diff(q[:, 0])).sum(),
        t[-1] - t[0],
    ])


def audit_motion(repo, robot_repo, directory, geometry):
    sys.path[:0] = [str(repo), str(robot_repo), str(robot_repo / 'rm75_control')]
    pkg = 'NMI_idea.experiments.'
    capability = import_module(pkg + 'ab_module_ladder_20261110.capability.capability')
    learned = import_module(pkg + 'ab_leg_bifurcation_20261112.learned_prodmp')
    cap = capability.CapabilityEvaluator(device='cpu')
    primitive = learned.LearnedProDMP.fit_default(seed=20261112)
    motion = read_json(directory / 'MOTION.json')
    bank_path = repo / EXPERIMENTS / 'ab_full_b_rollout_20261113/motion/BANK_MAIN.json.xz'
    alternate_path = repo / EXPERIMENTS / 'ab_inverse_acquisition_b_20261113/motion/BANK.json.xz'
    atlas_path = repo / EXPERIMENTS / 'ab_leg_dynamic_20261112/DYNAMIC_GEOMETRY_FROZEN.npz'
    actual_hashes = {path.name: sha(path) for path in (bank_path, atlas_path, Path(capability.__file__), Path(learned.__file__))}
    if learned.BASIS_SOURCE == 'reference':
        actual_hashes['reference_basis'] = sha(learned.REFERENCE_MOTION)
    assert actual_hashes == motion['source_sha256']
    current_robot_hashes = {Path(path).name: value for path, value in cap.manifest()['source_sha256'].items()}
    assert {Path(path).name: value for path, value in motion['robot']['source_sha256'].items()} == current_robot_hashes
    assert hashlib.sha256(np.asarray(primitive.weight_mean['advance'], dtype='<f8').tobytes()).hexdigest() == motion['advance_weight_sha256']
    assert motion['basis_source'] == learned.BASIS_SOURCE
    assert motion['fitted_seed'] == 20261112 and motion['mode'] == 'learned'
    assert motion['slot_s'] == 1.50 and motion['capture_dwell_s'] == .21 and motion['max_sample_dt_s'] == .020
    assert motion['poses'] == geometry['poses']
    with lzma.open(bank_path, 'rt', encoding='utf-8') as stream:
        bank = json.load(stream)
    with lzma.open(alternate_path, 'rt', encoding='utf-8') as stream:
        alternate = json.load(stream)
    ids = [p['id'] for p in geometry['poses']]
    nodes = {n['id']: n['q'] for n in bank['nodes']}
    endpoints = np.array([nodes[i] for i in ids])
    np.testing.assert_array_equal(motion['endpoint_q'], endpoints)
    by_source = {'ab_full_b_rollout_20261113': bank, 'ab_inverse_acquisition_b_20261113': alternate}
    old_edges = {label: {(e['start'], e['goal']): e for e in data['edges']} for label, data in by_source.items()}
    with np.load(atlas_path, allow_pickle=False) as atlas:
        pose_index = int(np.flatnonzero(atlas['pose_knee_deg'] == 15.).item())
        stations, offsets = atlas['s_mm'], atlas['view_offsets_mm']
        atlas_views = atlas['posed_views_m'][pose_index]
        atlas_frames = atlas['posed_frames'][pose_index]
        robot_atlas = atlas['robot_from_atlas']
    unique_offsets, surface_views = np.unique(offsets, return_index=True)

    costs = np.full((len(ids), len(ids), 4), np.inf)
    keys = set()
    edges = {}
    samples = 0
    metrics = []
    differences = []
    shape_checks = []
    for row in motion['edges']:
        a, b = row['start'], row['goal']
        assert (a, b) not in keys
        keys.add((a, b))
        assert row['legal'], (a, b, row['reason'])
        q, t, target, saved_fk = (np.array(row[k]) for k in ('q', 't_s', 'target', 'fk'))
        assert q.shape == (len(t), 8) and target.shape == saved_fk.shape == (len(t), 4, 4)
        assert t[0] == 0. and t[-1] == 1.50
        dt = np.diff(t)
        assert (dt > 0.).all() and dt.max() <= .020 + 1.e-12
        transit = np.flatnonzero(t <= 1.29 + 1.e-12)
        assert abs(t[transit[-1]] - 1.29) < 1.e-12
        np.testing.assert_array_equal(q[0], endpoints[a])
        np.testing.assert_array_equal(q[transit[-1]:], np.broadcast_to(endpoints[b], q[transit[-1]:].shape))
        np.testing.assert_array_equal(target[transit[-1]:], np.broadcast_to(target[transit[-1]], target[transit[-1]:].shape))

        u, v = geometry['poses'][a], geometry['poses'][b]
        c0 = np.array([stations[u['site']], offsets[u['view']]])
        c1 = np.array([stations[v['site']], offsets[v['view']]])
        chart = primitive.path(c0, c1, samples=len(transit))
        np.testing.assert_allclose(chart, row['chart_mm'], rtol=0., atol=1.e-12)
        if (a, b) in {(0, 3), (3, 5), (3, 6), (4, 4), (5, 1)}:
            control = primitive.path(c0, c1, samples=len(transit), learned=False)
            phase_control = primitive.path([0., 0.], [1., 0.], samples=len(transit), learned=False)
            phase_learned = primitive.path([0., 0.], [1., 0.], samples=len(transit))
            shape_checks.append({'edge': [a, b], 'max_chart_difference_mm': float(np.linalg.norm(chart - control, axis=1).max()),
                                 'max_unit_phase_difference': float(np.abs(phase_control[:, 0] - phase_learned[:, 0]).max()) if a != b else 0.})

        # Derive target translation and rail independently from the fitted chart.
        if u['site'] == v['site']:
            progress = np.clip(primitive.path([0., 0.], [1., 0.], samples=len(transit))[:, 0], 0., 1.)
        else:
            progress = np.clip((chart[:, 0] - c0[0]) / (c1[0] - c0[0]), 0., 1.)
        np.testing.assert_allclose(q[transit, 0], endpoints[a, 0] + progress * (endpoints[b, 0] - endpoints[a, 0]),
                                   atol=1.e-12, rtol=0.)
        if a != b:
            local = np.array([[np.interp(s, stations, [np.interp(x, unique_offsets, station[surface_views, axis])
                              for station in atlas_views]) for axis in range(3)] for s, x in chart])
            xyz = local @ robot_atlas[:3, :3].T + robot_atlas[:3, 3]
            xyz += (1. - progress)[:, None] * (saved_fk[0, :3, 3] - xyz[0])
            np.testing.assert_allclose(target[transit, :3, 3], xyz, atol=1.e-12, rtol=0.)
        endpoint_position = robot_atlas[:3, :3] @ atlas_views[v['site'], v['view']] + robot_atlas[:3, 3]
        endpoint_rotation = robot_atlas[:3, :3] @ atlas_frames[v['site'], v['view']]
        np.testing.assert_allclose(target[-1, :3, 3], endpoint_position, atol=1.e-12, rtol=0.)
        np.testing.assert_allclose(target[-1, :3, :3], endpoint_rotation, atol=1.e-12, rtol=0.)
        if a != b:
            rotation = Rotation.from_matrix(np.stack((saved_fk[0, :3, :3], endpoint_rotation)))
            # The endpoints retain the recorded FK/atlas matrices. Slerp
            # orthonormalizes its inputs, so compare only interior samples.
            np.testing.assert_allclose(target[transit[1:-1], :3, :3], Slerp([0., 1.], rotation)(progress[1:-1]).as_matrix(),
                                       atol=1.e-12, rtol=0.)

        fk = np.stack([cap.fk(sample) for sample in q])
        np.testing.assert_allclose(fk, saved_fk, atol=1.e-12, rtol=0.)
        checks = [cap.check_q(sample, desired) for sample, desired in zip(q, target, strict=True)]
        assert all(c['valid'] for c in checks)
        clearance = min(c['min_self_collision_distance_m'] for c in checks)
        margin = min(min(c['joint_limit_margin_by_coordinate']) for c in checks)
        pos_error = 1000. * np.linalg.norm(fk[:, :3, 3] - target[:, :3, 3], axis=1).max()
        rot_error = np.rad2deg(angle(fk[:, :3, :3], target[:, :3, :3])).max()
        rate = (np.abs(np.diff(q, axis=0)) / dt[:, None] / cap.kin.v_max).max()
        assert rate <= 1. and clearance >= .01
        actual = (clearance, margin, pos_error, rot_error, rate)
        fields = ('sampled_clearance_m', 'min_joint_margin', 'max_position_error_mm', 'max_rotation_error_deg', 'max_rate_fraction')
        np.testing.assert_allclose(actual, [row[k] for k in fields], atol=1.e-10, rtol=0.)
        costs[a, b] = cost_from_samples(q, t, fk)
        np.testing.assert_allclose(costs[a, b], row['cost'], atol=1.e-10, rtol=0.)
        if a != b:
            old = old_edges[geometry['edge_sources'][a][b]][ids[a], ids[b]]
            old_tip = np.array(old['samples']['tip_xyz_mm'])
            old_t = np.array(old['samples']['t_s'])
            old_on_grid = np.stack([np.interp(t, old_t, old_tip[:, axis]) for axis in range(3)], axis=1)
            max_curve_delta = float(np.linalg.norm(1000. * fk[:, :3, 3] - old_on_grid, axis=1).max())
            np.testing.assert_allclose([old['travel_mm'], old['turn_deg'], old['rail_mm'], old['duration_s']], geometry['costs'][a][b], atol=1.e-9, rtol=0.)
            differences.append({'edge': [a, b], 'tcp_delta_mm': float(costs[a, b, 0] - old['travel_mm']),
                                'max_time_aligned_tip_displacement_mm': max_curve_delta})
        metrics.append(actual)
        samples += len(t)
        edges[a, b] = {'q': q, 't_s': t, 'target': target, 'fk': fk, 'cost': costs[a, b]}

    declared = {(a, b) for a, left in enumerate(geometry['poses']) for b, right in enumerate(geometry['poses'])
                if abs(left['site'] - right['site']) <= 1}
    assert keys == declared and len(keys) == motion['attempted'] == motion['accepted'] == 63
    assert sum(row['kind'] == 'hold' for row in motion['edges']) == 9
    np.testing.assert_array_equal(np.isfinite(costs).all(axis=2), np.array([[edge is not None for edge in row] for row in geometry['costs']]))
    metrics = np.asarray(metrics)
    return costs, edges, {
        'checked_edges': len(keys), 'sampled_fk_limit_collision_rechecks': samples, 'holds': 9,
        'minimum_clearance_m': float(metrics[:, 0].min()), 'minimum_joint_margin': float(metrics[:, 1].min()),
        'maximum_tracking_position_error_mm': float(metrics[:, 2].max()),
        'maximum_tracking_rotation_error_deg': float(metrics[:, 3].max()), 'maximum_rate_fraction': float(metrics[:, 4].max()),
        'archived_path_comparison': differences, 'learned_endpoint_only_chart_checks': shape_checks,
    }


def audit_replay(repo, directory, geometry, costs, compiled):
    prior = repo / INHERITED
    sys.path.insert(0, str(prior))
    audit = import_module('audit_results')
    summary = read_json(directory / 'SUMMARY.json')
    for name, expected in summary['source_sha256'].items():
        path = (HERE if name in {'run.py', 'motion.py', 'PROTOCOL.md'} else prior) / name
        assert sha(path) == expected, path
    checked_geometry = dict(geometry, costs=[[edge.tolist() if np.isfinite(edge).all() else None
                                             for edge in row] for row in costs])
    model = audit.Reconstruction(checked_geometry)
    rows = audit.read_jsonl(directory / 'prescan_states.jsonl')
    active = [row for row in rows if 'actions' in row]
    root_map = {tuple(row['prescan']): row for row in active}
    assert len(root_map) == len(active)
    high = sorted(active, key=lambda r: r['mass'], reverse=True)[:12]
    contributions = sorted(active, key=lambda r: r['mass'] * (r['b'] - r['mpc_executed']), reverse=True)[:8]
    selected_roots = {tuple(row['prescan']): row for row in high + contributions}
    selected_masses = {}
    for prefix, row in selected_roots.items():
        w = model.root_weight(prefix)
        f, g, two = model.score_roots(w / w.sum(), prefix[2])
        for array, key in ((f, 'mpc_root_scores'), (g, 'b_root_scores')):
            np.testing.assert_allclose(np.where(np.isfinite(array), array, np.nan),
                                       [np.nan if x is None else x for x in row[key]], atol=1.e-12, rtol=0.)
        assert [model.choose(scores)[0] for scores in (f, g, two)] == row['actions'][:3]
        selected_masses[prefix] = w.sum()

    total_mass = 0.
    prior_paid = np.zeros(4)
    exact = summary['exact']
    for row in rows:
        prefix = row['prescan']
        mass = model.root_weight(prefix).sum()
        np.testing.assert_allclose(mass, row['mass'], atol=1.e-14, rtol=0.)
        prior_paid += mass * (costs[3, prefix[0]] + costs[prefix[0], prefix[2]])
        total_mass += mass
    assert len(rows) == exact['prescan_states'] and abs(total_mass - 1.) < 1.e-12
    aggregates = {name: sum(r['mass'] * r[key] for r in active) for name, key in (
        ('adaptive_public_ceiling', 'b'), ('mpc_executed', 'mpc_executed'), ('mpc_forecast', 'mpc_forecast'))}
    aggregates['adaptive_over_mpc'] = aggregates['adaptive_public_ceiling'] - aggregates['mpc_executed']
    aggregates['root_disagreement_mass'] = sum(r['mass'] for r in active if r['actions'][0] != r['actions'][1])
    aggregates['omitted_prescan_probability'] = sum(r['mass'] for r in rows if 'actions' not in r)
    for key, value in aggregates.items():
        np.testing.assert_allclose(exact[key], value, atol=1.e-12, rtol=0.)
    np.testing.assert_allclose(prior_paid, [exact['prescan_mean_resources'][k] for k in audit.METRICS[2:6]], atol=1.e-12, rtol=0.)

    records = audit.read_jsonl(directory / 'paired.jsonl')
    rng = np.random.default_rng(summary['replay']['seed'])
    assert len(records) == summary['replay']['n']
    hiddens = rng.choice(len(model.prior), size=len(records), p=model.prior)
    tapes = rng.random((len(records), 5, 3, 3))
    used = set()
    all_metrics = []
    counts = Counter()
    for k, (row, h, tape) in enumerate(zip(records, hiddens, tapes, strict=True)):
        assert row['case'] == k and row['shift_id'] == h
        uncached = k < summary['replay']['uncached_cases']
        assert row['uncached'] == uncached
        u, _ = model.draw(h, model.initial, tape[0, 1])
        at = model.anchor(u)
        v, _ = model.draw(h, at, tape[1, 0])
        prefix = (model.initial, u, at, v)
        assert prefix == tuple(row['prescan'])
        used.update(((3, model.initial), (model.initial, at)))
        counts[uncached] += 2
        commands = root_map[prefix]['actions']
        measured = []
        for policy, command in zip(audit.POLICIES, commands, strict=True):
            horizon = 1 if policy == 'two_images' else 2
            metrics, observations = model.run_trace(prefix, command, h, tape[2:], horizon)
            result = row['arms'][policy]
            assert observations == result['trace'], (k, policy, observations, result['trace'])
            np.testing.assert_allclose(metrics, [result[m] for m in audit.METRICS], atol=1.e-10, rtol=0.)
            previous = at
            w = model.root_weight(prefix)
            chosen = None
            for i, (action, packet, _) in enumerate(observations):
                used.add((previous, action))
                if action in audit.FAR:
                    near_packet = observations[i - 1][1]
                    joint = (w[:, None] * model.ratio[previous, near_packet]).T @ model.target[action, packet]
                    pair = np.unravel_index(joint.argmax(), joint.shape)
                    if joint[pair] > 0. and joint[pair] >= audit.THRESHOLD * (w @ model.like[action, packet]):
                        chosen = list(pair)
                w *= model.like[action, packet]
                previous = action
            assert result['selected'] == chosen, (k, policy, result['selected'], chosen)
            counts[uncached] += len(observations)
            measured.append(metrics)
        all_metrics.append(measured)
    replay = summary['replay']
    assert replay['uncached_replay']['generated_and_checked'] == counts[True]
    assert replay['uncached_replay']['identical_state_action_reuses'] == 0
    assert replay['cached_replay']['generated_and_checked'] + replay['cached_replay']['identical_state_action_reuses'] == counts[False]
    for mode in ('cached_replay', 'uncached_replay'):
        np.testing.assert_array_equal(replay[mode]['maximum_forecast_execution_cost_difference'], np.zeros(4))
    with np.load(directory / 'EXECUTED_MOTION.npz', allow_pickle=False) as archive:
        fields = {}
        for (a, b) in used:
            for name, values in compiled[a, b].items():
                key = f'edge_{a}_{b}_{name}'
                fields[key] = values
                np.testing.assert_allclose(archive[key], values, atol=1.e-12, rtol=0.)
        assert set(fields) == set(archive.files)
    all_metrics = np.asarray(all_metrics)
    for i, policy in enumerate(audit.POLICIES):
        np.testing.assert_allclose(all_metrics[:, i].mean(axis=0), [replay['total'][policy][k] for k in audit.METRICS],
                                   atol=1.e-10, rtol=0.)
    differences = {}
    for arm in ('mpc', 'fixed_center'):
        difference = all_metrics[:, 1] - all_metrics[:, audit.POLICIES.index(arm)]
        actual = {'wins': int((difference[:, 0] > 0.).sum()), 'losses': int((difference[:, 0] < 0.).sum())}
        assert all(replay['contrast_b'][arm][key] == value for key, value in actual.items())
        actual['mean'] = dict(zip(audit.METRICS, difference.mean(axis=0).tolist(), strict=True))
        np.testing.assert_allclose(list(actual['mean'].values()), [replay['contrast_b'][arm]['mean'][k] for k in audit.METRICS],
                                   atol=1.e-10, rtol=0.)
        differences[arm] = actual
    return {'source_hashes_match': True, 'prescan_states_recomputed': len(rows), 'prescan_aggregates': aggregates,
            'exact_roots_recomputed': len(selected_roots), 'exact_roots_probability': sum(selected_masses.values()),
            'paired_cases_reconstructed': len(records), 'paired_arm_traces_reconstructed': len(records) * len(audit.POLICIES),
            'execution_motion_arrays_checked': len(fields), 'unique_executed_edges_checked': len(used),
            'uncached_acquisitions_reconstructed': counts[True], 'memoized_replay_acquisitions_reconstructed': counts[False],
            'replay_differences_b': differences}


def shape_check(repo, robot_repo, directory, geometry):
    from motion import ProDMPSynthesizer

    learned = read_json(directory / 'MOTION.json')
    costs, control = ProDMPSynthesizer(repo, robot_repo, geometry, primitive_mode='endpoint_only').compile()
    np.testing.assert_array_equal(learned['endpoint_q'], control['endpoint_q'])
    matched = {(edge['start'], edge['goal']): edge for edge in control['edges']}
    output = []
    for edge in learned['edges']:
        a, b = edge['start'], edge['goal']
        other = matched[a, b]
        info = {'edge': [a, b], 'learned_legal': edge['legal'], 'endpoint_only_legal': other['legal'],
                'learned_reason': edge['reason'], 'endpoint_only_reason': other['reason']}
        if 'target' in edge and 'target' in other:
            t = np.asarray(edge['t_s'])
            np.testing.assert_array_equal(t, other['t_s'])
            q, q_other = np.asarray(edge['q']), np.asarray(other['q'])
            np.testing.assert_array_equal(q[[0, -1]], q_other[[0, -1]])
            xyz = np.asarray(edge['fk'])[:, :3, 3] * 1000.
            xyz_other = np.asarray(other['fk'])[:, :3, 3] * 1000.
            targets = np.asarray(edge['target'])[:, :3, 3] * 1000.
            targets_other = np.asarray(other['target'])[:, :3, 3] * 1000.
            diff = xyz - xyz_other
            chord = targets[-1] - targets[0]
            if geometry['poses'][a]['site'] != geometry['poses'][b]['site']:
                unit = chord / np.linalg.norm(chord)
                cross = lambda displacement: np.linalg.norm(displacement - (displacement @ unit) * unit)
                middle = int(np.argmin(np.abs(t - (learned['slot_s'] - learned['capture_dwell_s']) / 2.)))
                midpoint_cross_difference = float(cross(diff[middle]))
            else:
                midpoint_cross_difference = None
            info.update(max_same_time_tcp_difference_mm=float(np.linalg.norm(diff, axis=1).max()),
                        max_same_time_reference_difference_mm=float(np.linalg.norm(targets - targets_other, axis=1).max()),
                        midpoint_cross_track_tcp_difference_mm=midpoint_cross_difference,
                        learned_cost=edge['cost'], endpoint_only_cost=other['cost'],
                        learned_peak_rate_fraction=edge['max_rate_fraction'],
                        endpoint_only_peak_rate_fraction=other['max_rate_fraction'])
        output.append(info)
    result = {'scope': 'matched-endpoint sampled motion only', 'learned_accepted': learned['accepted'],
              'endpoint_only_accepted': control['accepted'], 'attempted': len(output),
              'endpoint_only_costs': [[cell.tolist() if np.isfinite(cell).all() else None for cell in row] for row in costs],
              'edges': output}
    (directory / 'PRODMP_SHAPE_CHECK.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--robot-repo', required=True, type=Path)
    parser.add_argument('--results', type=Path, default=HERE / 'verification')
    parser.add_argument('--motion-only', action='store_true')
    parser.add_argument('--shape-only', action='store_true')
    args = parser.parse_args()
    repo, robot_repo = args.repo.resolve(), args.robot_repo.resolve()
    geometry = read_json(repo / INHERITED / 'GEOMETRY.json')
    if args.shape_only:
        result = shape_check(repo, robot_repo, args.results, geometry)
        print(json.dumps({key: result[key] for key in ('attempted', 'learned_accepted', 'endpoint_only_accepted')}))
        return
    costs, edges, result = audit_motion(repo, robot_repo, args.results, geometry)
    if args.motion_only:
        print(json.dumps({k: v for k, v in result.items() if not isinstance(v, list)}, indent=2))
        return
    result = {'motion': result, **audit_replay(repo, args.results, geometry, costs, edges)}
    (args.results / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

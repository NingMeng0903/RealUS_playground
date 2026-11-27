"""Independent calculations from saved experiment artifacts; no worker imports.

Recomputes A regression and selected errors, B optimal finite recursion and
all-policy outcome accounting, and motion counterfactual metrics. Writes only
inside audit/. Numerical equality is correctness, not physical validation.
"""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def independent_A():
    saved = read('core/a_results.json')
    raw = np.load(ROOT / 'core/a_raw.npz')
    act = np.asarray(saved['action_grid'])
    move = .002 * np.asarray(saved['action_path_length_m']) / .04
    split = {}
    for name, seed, n in [('train', 41101, 256), ('validation', 41102, 64), ('test', 41103, 128)]:
        rng = np.random.default_rng(seed)
        c = rng.uniform(-1, 1, (n, 3)); e = rng.uniform(-.9, .9, (n, 2))
        E = rng.normal(0, .04, (n, 2, 2)); u = rng.uniform(-1, 1, (n, 12, 2))
        noise = rng.normal(0, .015, (n, 12, 2)); pn = rng.normal(0, .015, (n, 4, 2))
        split[name] = dict(c=c, e=e, E=E, u=u, noise=noise, probe_noise=pn)
        for key, value in split[name].items():
            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == saved['split_hashes'][name][key]
    records = {}
    for shifted, name in [(False, 'analytic_exact_negative_control'), (True, 'context_dependent_response_shift')]:
        def matrix(w):
            J = np.tile(np.eye(2), (len(w['c']), 1, 1))
            if shifted:
                c = w['c']
                J[:, 0, 0] += .45 * c[:, 0]; J[:, 0, 1] += .225 * c[:, 1]
                J[:, 1, 0] -= .1575 * c[:, 1]; J[:, 1, 1] += .45 * c[:, 2]
                J += w['E']
            return J
        train = split['train']; columns = []
        for world in range(len(train['c'])):
            for u in train['u'][world]:
                columns.append(np.outer(np.r_[1., train['c'][world]], u).ravel())
        design = np.array(columns)
        target = (np.einsum('nij,nkj->nki', matrix(train) - np.eye(2), train['u']) + train['noise']).reshape(-1, 2)
        coeff = np.linalg.solve(design.T @ design + .001 * np.eye(8), design.T @ target)
        assert np.max(np.abs(coeff - np.asarray(saved['conditions'][name]['coefficient']))) < 1e-12
        test = split['test']; n = len(test['c']); J = matrix(test)
        true = test['e'][:, None] + np.einsum('nij,aj->nai', J, act)
        fp = np.einsum('ni,aj->naij', np.c_[np.ones(n), test['c']], act).reshape(n, len(act), 8)
        analytic = test['e'][:, None] + act[None]
        learned = analytic + fp @ coeff
        probe = np.array([[.25, 0], [-.25, 0], [0, .25], [0, -.25]])
        measurements = test['e'][:, None] + np.einsum('nij,pj->npi', J, probe) + test['probe_noise']
        Jfit = np.stack([2 * (measurements[:, 0] - measurements[:, 1]), 2 * (measurements[:, 2] - measurements[:, 3])], -1)
        preds = {'analytic_geometry': analytic, 'canonical_learned_residual': learned,
                 'online_local_jacobian_calibration': test['e'][:, None] + np.einsum('nij,aj->nai', Jfit, act),
                 'oracle_response_reference': true}
        errors = {}; means = {}
        for method, pred in preds.items():
            index = np.argmin(np.sum(pred ** 2, -1) + move, axis=1)
            err = np.sum(true[np.arange(n), index] ** 2, -1)
            charged = err + move[index] + (.044 if method.startswith('online') else 0.)
            assert np.array_equal(index, raw[f'{name}_test_{method}_index'])
            assert np.max(np.abs(err - raw[f'{name}_test_{method}_error'])) < 1e-12
            assert np.max(np.abs(charged - raw[f'{name}_test_{method}_charged'])) < 1e-12
            errors[method] = err
            means[method] = {'error': float(err.mean()), 'charged': float(charged.mean())}
        delta = errors['canonical_learned_residual'] - errors['analytic_geometry']
        rng = np.random.default_rng(41104)
        ci = np.quantile(delta[rng.integers(0, n, (2000, n))].mean(1), [.025, .975])
        records[name] = {'methods': means, 'learned_minus_analytic': float(delta.mean()), 'ci95': ci.tolist()}
    return records


def edge_cost(e):
    return e['path_length_m'] / .02 + .4 + e['proxy_cost']


def independent_optimum(g, node, weights, remaining, L):
    """Unnormalised exhaustive minimisation after grouping observed outcomes."""
    mass = float(np.sum(weights)); choices = [20. * mass]
    for target, eid in g['nodes'][node]['terminal'].items():
        edge = g['edges'][eid]
        if edge['exact_validated']:
            choices.append(edge_cost(edge) * mass + 20. * (mass - weights[int(target)]))
    if remaining:
        for action, eid in g['nodes'][node]['inspect'].items():
            edge = g['edges'][eid]
            if edge['exact_validated']:
                value = edge_cost(edge) * mass
                K = L[node] if isinstance(L, dict) else L
                for outcome in range(K.shape[-1]):
                    postmass = weights * K[int(action), :, outcome]
                    if np.sum(postmass):
                        value += independent_optimum(g, edge['next'], postmass, remaining - 1, L)
                choices.append(value)
    return min(choices)


def evaluate_policy(g, policy, belief, L):
    # Explicit realized terminal paths; endpoint masses sum to one.
    totals = {key: 0. for key in ['total_loss', 'success', 'wrong_identity', 'explicit_failure', 'invalid_failure',
                                 'movement_m', 'observations', 'movement_time', 'observation_cost', 'proxy_cost', 'outcome_mass']}
    stack = [(h, float(belief[h]), policy, 0., 0., 0.) for h in range(4)]
    while stack:
        h, prob, pol, distance, observations, proxy = stack.pop()
        if not prob:
            continue
        final = None
        if pol['kind'] == 'fail':
            final = 'explicit_failure'
        else:
            edge = g['edges'][pol['edge']]
            distance += edge['path_length_m']; observations += 1.; proxy += edge['proxy_cost']
            if not edge['exact_validated']:
                final = 'invalid_failure'
            elif pol['kind'] == 'terminal':
                final = 'success' if pol['target'] == h else 'wrong_identity'
            else:
                K = L[pol['edge'].split('/')[0]] if isinstance(L, dict) else L
                for y in range(K.shape[-1]):
                    likelihood = K[pol['action'], h, y]
                    if likelihood:
                        stack.append((h, prob * likelihood, pol['branches'][str(y)], distance, observations, proxy))
        if final is not None:
            totals[final] += prob; totals['outcome_mass'] += prob
            totals['movement_m'] += prob * distance; totals['observations'] += prob * observations
            totals['movement_time'] += prob * distance / .02
            totals['observation_cost'] += prob * observations * .4; totals['proxy_cost'] += prob * proxy
            totals['total_loss'] += prob * (distance / .02 + observations * .4 + proxy + (0. if final == 'success' else 20.))
    assert abs(totals['outcome_mass'] - 1) < 1e-12
    return totals


def independent_B():
    graph = read('core/graph.json'); saved = read('core/b_results.json'); policies = read('core/b_policies.json')
    L = np.asarray(saved['sensor_likelihood']); per_method = {}; maxerr = 0.; departing = []
    for ci, g in graph['contexts'].items():
        for eid, e in g['edges'].items():
            if e['exact_validated']:
                q = np.asarray(e['capability']['exact']['q_ref']); seed = np.asarray(e['q_seed'])
                departure = float(np.rad2deg(np.max(np.abs(q[0, 1:] - seed[1:]))))
                departing.append({'context': ci, 'edge': eid, 'reachable': e['source_node_reachable'], 'departure_deg': departure})
    for row in saved['rows']:
        ci = str(row['context']); g = graph['contexts'][ci]; b = np.asarray(row['belief'])
        expected = independent_optimum(g, 'root', b, 2, L)
        maxerr = max(maxerr, abs(expected - row['methods']['B_observation_contingent']['planning_objective']))
        for method, pol in policies[f'{ci}:{row["belief_id"]}'].items():
            rec = evaluate_policy(g, pol, b, L)
            assert abs(rec['total_loss'] - row['methods'][method]['expected_taskloss_plus_cost']) < 1e-12
            assert abs(rec['wrong_identity'] - row['methods'][method]['false_acceptance_probability']) < 1e-12
            assert abs(rec['observations'] - row['methods'][method]['acquired_observations_including_terminal']) < 1e-12
            per_method.setdefault(method, []).append(rec)
    assert maxerr < 1e-12
    summaries = {method: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} for method, rows in per_method.items()}
    no_proxy_changed = sum(p['B_observation_contingent'] != p['B_without_dird_proxy'] for p in policies.values())
    return {'n_scenarios': len(saved['rows']), 'independent_optimal_value_max_error': maxerr,
            'independent_outcome_accounting': summaries, 'full_vs_no_proxy_changed_policy_count': no_proxy_changed,
            'departure_above_12deg': [d for d in departing if d['departure_deg'] > 12],
            'reachable_departure_max_deg': max(d['departure_deg'] for d in departing if d['reachable']),
            'important_scope': 'Total loss is accumulated motion/sensing/proxy costs plus20 for failure, not deadline-truncated first acquisition time.'}


def independent_motion():
    directory = ROOT / 'motion/controlled_servo'; result = {}
    for method in ['kmp', 'dmp']:
        pair = {}
        for gain in [0, 1]:
            prefix = f'{method}_not_detected_depth_plus120mm_lateral_'
            plus = np.load(directory / f'{prefix}plus10mm_gain{gain}.npz')
            minus = np.load(directory / f'{prefix}minus10mm_gain{gain}.npz')
            anyvisible = (plus['visible'] != 0) | (minus['visible'] != 0)
            end = int(np.flatnonzero(anyvisible)[0]) if anyvisible.any() else len(anyvisible)
            distance = np.linalg.norm(plus['p_cmd'][:end] - minus['p_cmd'][:end], axis=1)
            pair[f'gain{gain}'] = {'common_missing_prefix': end, 'world_detection_totals': [int(plus['visible'].sum()), int(minus['visible'].sum())],
                                    'path_max_difference_mm': float(1000 * distance.max())}
        for scenario in ['nominal_gain0', 'nominal_gain1', 'nominal_ruckig']:
            d = np.load(directory / f'{method}_{scenario}.npz')
            pair[scenario] = {'mean_abs_u_mm': float(np.abs(d['image_u_m']).mean() * 1000), 'visible_fraction': float(d['visible'].mean())}
        result[method] = pair
    depth = read('motion/depth_contract.json')
    stock = np.asarray(depth['runtime_xi']); corrected = np.asarray(depth['corrected_adapter_xi'])
    result['depth_saved_sample_recompute'] = {'stock_skin_d_max': float(np.abs(stock[:, 2]).max()),
        'stock_skin_d_mean': float(np.abs(stock[:, 2]).mean()), 'corrected_skin_d_max': float(np.abs(corrected[:, 2]).max()),
        'h_theta_change_max': float(np.max(np.abs(stock[:, :2] - corrected[:, :2])))}
    return result


if __name__ == '__main__':
    result = {'A': independent_A(), 'B': independent_B(), 'motion': independent_motion()}
    result['inputs_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
        ['core/a_response.py', 'core/a_results.json', 'core/a_raw.npz', 'core/b_planner.py', 'core/b_results.json', 'core/b_policies.json', 'core/graph.json', 'motion/metrics.json', 'motion/depth_contract.json']}
    (OUT / 'independent_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

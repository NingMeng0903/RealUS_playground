"""Independent policy construction/evaluation of prior x response factorial."""
from pathlib import Path
import hashlib
import json
import numpy as np
from independent_extension import make_kernels
from independent_results import edge_cost, evaluate_policy

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads((ROOT / p).read_text())


def choose_terminal(g, node, weights):
    mass = float(sum(weights)); best = (20.*mass, {'kind': 'fail'})
    for target, eid in g['nodes'][node]['terminal'].items():
        e = g['edges'][eid]
        if e['exact_validated']:
            value = edge_cost(e)*mass + 20.*(mass-weights[int(target)])
            if value < best[0] - 1e-12*mass:
                best = (value, {'kind': 'terminal', 'edge': eid, 'target': int(target)})
    return best


def make_policy(g, node, weights, horizon, kernel, greedy=False):
    current, policy = choose_terminal(g, node, weights)
    if not horizon:
        return current, policy
    mass = float(sum(weights)); possibilities = []
    for action, eid in g['nodes'][node]['inspect'].items():
        e = g['edges'][eid]
        if not e['exact_validated']:
            continue
        q = edge_cost(e)*mass; children = {}; a = int(action)
        for y in range(kernel[node].shape[-1]):
            joint = weights * kernel[node][a, :, y]
            if greedy:
                v, child = choose_terminal(g, e['next'], joint)
            else:
                v, child = make_policy(g, e['next'], joint, horizon-1, kernel)
            q += v; children[str(y)] = child
        possibilities.append((q, eid, a, children))
    if not possibilities:
        return current, policy
    selected = possibilities[0]
    for candidate in possibilities[1:]:
        if candidate[0] < selected[0] - (0. if greedy else 1e-12*mass):
            selected = candidate
    q, eid, a, children = selected
    if q >= current - 1e-12*mass:
        return current, policy
    e = g['edges'][eid]
    if greedy:
        q = edge_cost(e)*mass; children = {}
        for y in range(kernel[node].shape[-1]):
            joint = weights * kernel[node][a, :, y]
            v, child = make_policy(g, e['next'], joint, horizon-1, kernel, True)
            q += v; children[str(y)] = child
    return q, {'kind': 'inspect', 'edge': eid, 'action': a, 'branches': children}


def main():
    graph = read('core/graph.json'); saved = read('core/prior_factorial_results.json')
    hidden = read('core/extension_hidden_manifest.json')
    W = np.asarray(read('core/a_results.json')['conditions']['context_dependent_response_shift']['coefficient'])
    source = np.array([.4, .3, .2, .1]); counts = np.random.default_rng(41108).multinomial(256, source)
    prior = (counts+1)/260.
    assert np.array_equal(counts, saved['source_counts']) and np.array_equal(prior, saved['fitted_prior'])
    priors = {'broad_uniform': np.ones(4)/4, 'source_fitted': prior}
    populations = {'source': source, 'shifted': np.roll(source, 1)}
    rows = {(r['context'], r['scenario_id'], r['controller'], r['test_population']): r for r in saved['rows']}
    worst = 0.; summary = {}; mismatches = []
    for item in hidden:
        ci = item['context']; bi = item['belief_id']; g = graph['contexts'][str(ci)]
        KT, KL, KA, _ = make_kernels(g, np.array(item['public_context']), np.array(item['persistent_true_E_by_hypothesis']), W,
                                   np.asarray(graph['config']['target_offsets_m']))
        for controller in ['greedy', 'B']:
            for priorname, p in priors.items():
                for response, K in [('analytic', KA), ('learned', KL)]:
                    model_value, policy = make_policy(g, 'root', p, 2, K, controller == 'greedy')
                    name = priorname+':'+response
                    for population, truth in populations.items():
                        record = rows[(ci, bi, controller, population)]['methods'][name]
                        result = evaluate_policy(g, policy, truth, KT)
                        error = abs(result['total_loss'] - record['true_expected_taskloss_plus_cost'])
                        if error > 1e-10:
                            mismatches.append([ci, bi, controller, population, name, error, policy.get('edge', 'fail')])
                        worst = max(worst, error, abs(model_value-record['planning_objective']))
                        summary.setdefault(population+':'+controller, {}).setdefault(name, []).append(result['total_loss'])
    assert worst < 1e-11, {'worst': worst, 'first_mismatches': mismatches[:10]}
    means = {cell: {method: float(np.mean(values)) for method, values in methods.items()} for cell, methods in summary.items()}
    result = {'source_counts': counts.tolist(), 'fitted_prior': prior.tolist(),
              'independent_policy_and_loss_max_difference': worst, 'loss_cell_means': means,
              'source_prior_minus_broad_learned_B': {pop: means[pop+':B']['source_fitted:learned'] - means[pop+':B']['broad_uniform:learned'] for pop in populations}}
    result['inputs_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
        ['core/graph.json', 'core/prior_response_factorial.py', 'core/prior_factorial_results.json', 'core/extension_hidden_manifest.json', 'core/a_results.json']}
    (OUT / 'independent_factorial.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

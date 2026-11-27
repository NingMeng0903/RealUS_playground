"""Independent reconstruction of learned A -> finite B synthetic extension."""
from pathlib import Path
import hashlib
import json
import numpy as np
from independent_results import independent_optimum, evaluate_policy

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def read(p):
    return json.loads((ROOT / p).read_text())


def categorical(z):
    pd = .1 + .85 * np.exp(-np.sum(z*z) / 1.28)
    logits = np.array([z[0], -z[0], z[1], -z[1]]) / .4
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    return np.r_[1. - pd, pd * probabilities]


def make_kernels(graph, c, E, W, target_offsets):
    origin = np.asarray(graph['nodes']['root']['pose_exit'])[:3, 3]
    targets = origin + np.asarray(graph['placement_translation_m']) + target_offsets
    true = {}; learned = {}; analytic = {}; feature_errors = []
    Jmean = np.eye(2) + np.array([[.45*c[0], .225*c[1]], [-.1575*c[1], .45*c[2]]])
    for node, n in graph['nodes'].items():
        if not n['inspect']:
            continue
        KT = np.empty((3, 4, 5)); KL = KT.copy(); KA = KT.copy()
        for action, eid in n['inspect'].items():
            edge = graph['edges'][eid]; a = int(action)
            end = np.asarray(edge['T_tcp'])[-1, :3, 3]
            assert np.array_equal(end, np.asarray(edge['pose_exit'])[:3, 3])
            u = (end[:2] - origin[:2]) / .04
            learned_delta = u + np.outer(np.r_[1., c], u).ravel() @ W
            for h in range(4):
                initial = (origin[:2] - targets[h, :2]) / .04
                zt = initial + (Jmean + E[h]) @ u
                za = initial + u; zl = initial + learned_delta
                KT[a, h] = categorical(zt); KL[a, h] = categorical(zl); KA[a, h] = categorical(za)
                feature_errors.append([np.sum((za - zt)**2), np.sum((zl - zt)**2)])
        true[node] = KT; learned[node] = KL; analytic[node] = KA
    return true, learned, analytic, np.mean(feature_errors, 0)


def main():
    graph = read('core/graph.json'); records = read('core/extension_results.json')
    policies = read('core/extension_policies.json'); manifest = read('core/extension_hidden_manifest.json')
    W = np.asarray(read('core/a_results.json')['conditions']['context_dependent_response_shift']['coefficient'])
    offsets = np.asarray(graph['config']['target_offsets_m']); rng = np.random.default_rng(41107)
    totals = {}; optimal_maxerr = 0.; featureerr = 0.; kernelerr = 0.; changed_truth = []
    for row, truth in zip(records['rows'], manifest):
        ci = str(row['context']); bi = row['belief_id']; g = graph['contexts'][ci]
        c = rng.uniform(-1, 1, 3); E = rng.normal(0, .04, (4, 2, 2))
        assert np.array_equal(c, truth['public_context']) and np.array_equal(E, truth['persistent_true_E_by_hypothesis'])
        assert np.array_equal(c, row['public_context'])
        KT, KL, KA, fe = make_kernels(g, c, E, W, offsets)
        for j, name in enumerate(['analytic', 'learned']):
            featureerr = max(featureerr, abs(fe[j] - row['feature_prediction_squared_error'][name]))
        for name, K in [('analytic', KA), ('learned', KL)]:
            mse = float(np.mean([(K[node] - KT[node])**2 for node in KT]))
            kernelerr = max(kernelerr, abs(mse - row['kernel_probability_mse'][name]))
        for model in [KT, KL, KA]:
            for K in model.values():
                assert np.min(K) > 0 and np.max(np.abs(K.sum(-1) - 1)) < 1e-12
        b = np.asarray(row['belief'])
        optimum = independent_optimum(g, 'root', b, 2, KL)
        optimal_maxerr = max(optimal_maxerr, abs(optimum - row['methods']['learned_B']['planning_objective']))
        for method, pol in policies[f'{ci}:{bi}'].items():
            result = evaluate_policy(g, pol, b, KT)
            assert abs(result['total_loss'] - row['methods'][method]['true_expected_taskloss_plus_cost']) < 1e-11
            totals.setdefault(method, []).append(result)
        # Concrete hidden-response mutation: Khat must stay exactly fixed.
        KT2, KL2, KA2, _ = make_kernels(g, c, E + .11, W, offsets)
        assert all(np.array_equal(KL[node], KL2[node]) and np.array_equal(KA[node], KA2[node]) for node in KL)
        changed_truth.append(max(float(np.max(np.abs(KT2[node] - KT[node]))) for node in KT))
    assert optimal_maxerr < 1e-11 and featureerr < 1e-12 and kernelerr < 1e-12
    summary = {m: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} for m, rows in totals.items()}
    result = {'scenarios': len(records['rows']), 'learned_model_independent_optimum_max_error': optimal_maxerr,
              'feature_error_reconstruction_max_difference': float(featureerr),
              'kernel_MSE_reconstruction_max_difference': float(kernelerr),
              'hidden_response_mutation_changes_Ktrue_but_not_Khat': True,
              'mutated_Ktrue_max_changes_range': [min(changed_truth), max(changed_truth)],
              'independent_true_outcome_accounting': summary,
              'terminal_success_scope': 'Success is correct discrete target identity at a validated graph edge, not ultrasound quality.'}
    result['inputs_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
        ['core/graph.json', 'core/ab_integrated_extension.py', 'core/extension_results.json', 'core/extension_policies.json', 'core/extension_hidden_manifest.json', 'core/a_results.json']}
    (OUT / 'independent_extension.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

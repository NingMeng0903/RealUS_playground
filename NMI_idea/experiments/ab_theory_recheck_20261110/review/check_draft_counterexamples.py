"""Small independent checks of the V9.3/V6.1 claim boundaries.

No atlas, learned model, simulator, controller, or locked cohort is run here.
"""
from pathlib import Path
import json
import numpy as np


def unit(x):
    a = np.asarray(x, dtype=float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def moment(points):
    return np.einsum('ni,nj->ij', points, points) / len(points)


F = np.diag([2., 1., 1.])
t = unit([1., 1., 0.])
n = unit([1., -1., 0.])
t_new = unit(F @ t)
n_new = unit(np.linalg.solve(F.T, n))
n_wrong = unit(F @ n)
assert abs(t_new @ n_new) < 1e-14
assert abs(t_new @ n_wrong - .6) < 1e-14

P = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
Q = unit([[1., 1., 0.], [-1., -1., 0.], [1., -1., 0.], [-1., 1., 0.]])
assert np.allclose(moment(P), moment(Q), atol=1e-14)
P_new, Q_new = unit(P @ F.T), unit(Q @ F.T)
assert np.allclose(moment(P_new), np.diag([.5, .5, 0.]))
assert np.allclose(moment(Q_new), np.diag([.8, .2, 0.]))

# r=theta and B=theta: cross terms are necessary even with zero means.
theta = np.array([-1., 1.])
rows = np.column_stack((theta, theta))
Gamma = rows.T @ rows / 2
delta = -Gamma[0, 1] / Gamma[1, 1]
actual_optimal_risk = float(np.mean((theta + theta * delta)**2))
assert delta == -1. and actual_optimal_risk == 0.
assert np.allclose(Gamma, np.ones((2, 2)))

# A best-vs-rest gap condition preserves the winner, not every pairwise rank.
true_cost = np.array([0., .50, .51])
model_cost = np.array([0., .54, .47])
epsilon = .05
assert np.max(np.abs(true_cost - model_cost)) <= epsilon
assert np.min(true_cost[1:] - true_cost[0]) > 2 * epsilon
assert np.argmin(true_cost) == np.argmin(model_cost)
assert np.argsort(true_cost).tolist() != np.argsort(model_cost).tolist()

out = {
    'scope': 'Exact finite mathematical examples, not an ultrasound or atlas performance result.',
    'material_not_shape_invariance': {
        'F': F.tolist(),
        'direction_change_degrees': float(np.rad2deg(np.arccos(np.clip(t @ t_new, -1, 1)))),
        'proper_normal_dot_tangent': float(n_new @ t_new),
        'wrong_normal_dot_tangent': float(n_wrong @ t_new),
    },
    'same_current_moment_different_nonrigid_transport': {
        'old_M': moment(P).tolist(), 'new_M_P': moment(P_new).tolist(),
        'new_M_Q': moment(Q_new).tolist(),
    },
    'full_residual_cross_terms': {
        'Gamma': Gamma.tolist(), 'optimal_delta': float(delta),
        'actual_risk_with_full_Gamma': actual_optimal_risk,
        'actual_risk_ignoring_cross_terms': float(np.mean(theta**2)),
    },
    'transfer_best_action_vs_full_ranking': {
        'true_cost': true_cost.tolist(), 'model_cost': model_cost.tolist(),
        'epsilon': epsilon, 'winner_preserved': True, 'full_ranking_preserved': False,
    },
    'all_passed': True,
}
target = Path(__file__).with_name('DRAFT_COUNTEREXAMPLES.json')
if target.exists():
    raise SystemExit(f'Preserve existing audit output: {target}')
target.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
print(json.dumps(out, indent=2))

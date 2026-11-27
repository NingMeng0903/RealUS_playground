"""Independent exact-arithmetic check of the proposed moment/VOI example.

X = coefficient * sqrt(square_scale); all coefficients and probabilities are
rational. This checks a static estimation decision with action a in R and loss
(X-a)^2. It is not an ultrasound or robot experiment.
"""
from fractions import Fraction as F
import json


def summarize(points, square_scale, ternary=False):
    assert sum(p for c, p in points) == 1
    m1 = sum(c * p for c, p in points)
    m2 = sum(c * c * p for c, p in points)
    prior_risk = square_scale * (m2 - m1 * m1)
    branches = {}
    for c, p in points:
        outcome = (1 if c > 0 else -1 if c < 0 else 0) if ternary else int(c > 0)
        branches.setdefault(outcome, []).append((c, p))
    post_risk = F(0)
    branch_rows = []
    for outcome, group in sorted(branches.items()):
        mass = sum(p for c, p in group)
        cm1 = sum(c * p for c, p in group) / mass
        cm2 = sum(c * c * p for c, p in group) / mass
        variance = square_scale * (cm2 - cm1 * cm1)
        post_risk += mass * variance
        branch_rows.append({"outcome": outcome, "probability": str(mass),
                            "mean_coefficient": str(cm1),
                            "conditional_variance": str(variance)})
    cost = F(4, 5)
    return {"square_scale": square_scale,
            "E_X": 0 if m1 == 0 else str(m1),
            "E_X_squared": str(square_scale * m2),
            "Gamma_for_r_X_B_minus_1": [[str(square_scale * m2), "0"], ["0", "1"]],
            "prior_min_MSE": str(prior_risk),
            "expected_posterior_min_MSE": str(post_risk),
            "observation_cost": str(cost),
            "observe_total_risk": str(cost + post_risk),
            "optimal_choice": "observe" if cost + post_risk < prior_risk else "skip",
            "branches": branch_rows}


P = [(F(-1), F(1, 2)), (F(1), F(1, 2))]
Q = [(F(-1), F(1, 6)), (F(0), F(2, 3)), (F(1), F(1, 6))]
result = {"binary_Y_1_if_X_gt_0": {"P": summarize(P, 1), "Q": summarize(Q, 3)},
          "ternary_sign_countercheck": {"P": summarize(P, 1, True), "Q": summarize(Q, 3, True)}}
assert result["binary_Y_1_if_X_gt_0"]["P"]["expected_posterior_min_MSE"] == "0"
assert result["binary_Y_1_if_X_gt_0"]["Q"]["expected_posterior_min_MSE"] == "2/5"
assert result["binary_Y_1_if_X_gt_0"]["P"]["optimal_choice"] == "observe"
assert result["binary_Y_1_if_X_gt_0"]["Q"]["optimal_choice"] == "skip"
assert result["ternary_sign_countercheck"]["Q"]["expected_posterior_min_MSE"] == "0"
print(json.dumps(result, indent=2))

"""Smoke tests for the public-outcome Fisher/design helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_information import (  # noqa: E402
    binned_signed_centroid_fisher,
    binned_signed_centroid_probabilities,
    categorical_fisher,
    categorical_mutual_information,
    chain_fisher,
    finite_difference_fisher,
    greedy_common_candidates,
    information_to_covariance,
    schur_complement_information,
    task_design_score,
)


EDGES = (-np.inf, -0.25, 0.0, 0.25, np.inf)


def test_binned_signed_centroid_normalizes_with_state_dependent_detection() -> None:
    result = binned_signed_centroid_fisher(
        centroid=0.08,
        scale=0.14,
        p_detect=0.73,
        edges=EDGES,
        parameter_jacobian=np.asarray([
            [0.8, 0.0],   # d centroid / d (task, nuisance)
            [0.0, 0.25],  # d log scale / d (...)
            [0.3, -0.1],  # state-dependent detection
        ]),
    )
    assert result.labels[0] == "miss"
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert np.all(result.probabilities >= 0.0)
    assert np.allclose(result.probability_jacobian.sum(axis=0), 0.0, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(result.fisher) >= -1e-12)
    assert result.parameter_fisher is not None
    assert result.parameter_fisher.shape == (2, 2)
    # A p_detect derivative must alter the miss/bin probabilities, hence the
    # parameter Fisher cannot be reproduced by dropping the miss category.
    assert result.probability_jacobian[0, 2] != 0.0
    assert result.diagnostics["mixture_mode_identification"] == "not_claimed"


def test_analytic_jacobian_matches_central_difference() -> None:
    theta = np.asarray([0.08, math_log(0.14), 0.73])

    def law(values: np.ndarray) -> np.ndarray:
        return binned_signed_centroid_probabilities(
            values[0], np.exp(values[1]), values[2], EDGES,
        )

    analytic = binned_signed_centroid_fisher(
        theta[0], np.exp(theta[1]), theta[2], EDGES,
    )
    numeric = finite_difference_fisher(law, theta, relative_step=2e-6, return_details=True)
    assert np.allclose(analytic.probabilities, numeric.probabilities, atol=1e-12)
    assert np.allclose(analytic.probability_jacobian, numeric.probability_jacobian, rtol=2e-5, atol=3e-7)
    assert np.allclose(analytic.fisher, numeric.fisher, rtol=3e-5, atol=3e-7)


def math_log(value: float) -> float:
    """Local spelling keeps this test dependency-free and explicit."""

    return float(np.log(value))


def test_explicit_censoring_and_optional_clutter_are_normalized() -> None:
    result = binned_signed_centroid_fisher(
        centroid=0.10,
        scale=0.08,
        p_detect=0.60,
        edges=(-0.2, 0.2),
        censoring="tail",
        clutter_centroid=-0.35,
        clutter_scale=0.20,
        p_clutter=0.12,
    )
    assert result.labels[:3] == ("miss", "left_censored", "y_bin_0")
    assert result.labels[-1] == "right_censored"
    assert np.isclose(result.probabilities.sum(), 1.0)
    assert np.all(result.probabilities >= 0.0)
    assert result.fisher.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(result.fisher) >= -1e-12)

    miss = binned_signed_centroid_probabilities(
        0.10, 0.08, 0.60, (-0.2, 0.2), censoring="miss",
        clutter_centroid=-0.35, clutter_scale=0.20, p_clutter=0.12,
    )
    assert np.isclose(miss.sum(), 1.0)
    # Under censor-to-miss, the miss probability is larger because both tails
    # are censored rather than exposed as public categories.
    assert miss[0] > result.probabilities[0]


def test_categorical_fisher_and_mutual_information_exact_cases() -> None:
    probabilities = np.asarray([0.25, 0.75])
    derivative = np.asarray([[-1.0], [1.0]])
    fisher = categorical_fisher(probabilities, derivative)
    assert np.allclose(fisher, [[1.0 / 0.25 + 1.0 / 0.75]])

    perfect = categorical_mutual_information(np.eye(2), [0.5, 0.5])
    assert np.isclose(perfect, np.log(2.0))
    assert np.isclose(categorical_mutual_information(np.ones((2, 2)) / 2.0, [0.5, 0.5]), 0.0)


def test_chain_and_task_score_include_nuisance_uncertainty() -> None:
    local = np.diag([2.0, 0.5])
    jacobian = np.asarray([[1.0, 1.0], [0.0, 1.0]])
    chained = chain_fisher(local, jacobian)
    assert np.allclose(chained, jacobian.T @ local @ jacobian)

    prior = np.diag([1.0, 4.0])
    candidate = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    direction = np.asarray([[1.0], [0.0]])
    score = task_design_score(candidate, prior, decision_directions=direction)
    assert score > 0.0
    schur = task_design_score(
        candidate, prior, method="schur", task_indices=[0], nuisance_indices=[1],
        decision_directions=np.ones((1, 1)),
    )
    assert schur >= 0.0
    before = information_to_covariance(prior, np.zeros_like(prior))
    after = information_to_covariance(prior, candidate)
    assert after[0, 0] < before[0, 0]
    total_info = np.linalg.pinv(prior) + candidate
    assert np.all(schur_complement_information(total_info, [0], [1]) >= -1e-12)


def test_greedy_common_selection_honors_safe_mask_and_tie_order() -> None:
    prior = np.eye(2)
    candidates = {
        "unsafe_best": np.diag([10.0, 0.0]),
        "safe_task": np.diag([2.0, 0.0]),
        "safe_nuisance": np.diag([0.0, 1.0]),
    }
    chosen = greedy_common_candidates(
        candidates,
        prior,
        budget=2,
        safe_mask=[False, True, True],
        decision_directions=np.asarray([[1.0], [0.0]]),
    )
    assert chosen.selected_ids == ("safe_task",)
    assert chosen.safe_mask[0] is False or not bool(chosen.safe_mask[0])
    assert chosen.marginal_scores[0] > 0.0
    assert np.isclose(chosen.combined_fisher[0, 0], 2.0)


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("probe_information smoke tests passed")


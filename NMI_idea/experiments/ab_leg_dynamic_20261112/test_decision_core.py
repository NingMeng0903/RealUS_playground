"""Small algebraic smoke tests for ``decision_core``.

These tests intentionally avoid geometry and motion.  They pin down the
strict feedback witness, the null case, posterior/miss normalization, and the
fact that A/B see exactly the same checked mask and transition costs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decision_core import (  # noqa: E402
    BayesBelief,
    CheckedCandidateSet,
    bayes_update,
    centroid_outcome_kernel,
    compare_methods,
    plan_scan_pair,
    posterior_table,
)


def _strict_feedback_case() -> tuple[BayesBelief, CheckedCandidateSet, np.ndarray]:
    """Two hidden worlds; a probe reveals the world and L/R finish it."""

    # Outcome 0/1 are H/L here; explicit labels are supplied below.  The
    # centroid helper's first outcome is always miss, so this witness uses a
    # generic finite public kernel directly to isolate max-vs-expectation.
    kernels = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # probe: perfect H/L observation
            [[1.0, 0.0], [1.0, 0.0]],  # L action: no informative outcome
            [[1.0, 0.0], [1.0, 0.0]],  # R action: no informative outcome
        ]
    )
    candidates = CheckedCandidateSet(
        ("probe", "L", "R"),
        kernels,
        first_cost=np.zeros(3),
        after_action_feasible=np.ones((3, 2, 3), dtype=bool),
        # Every branch pays exactly the same transition cost.  This keeps the
        # strict gap attributable to feedback, not to policy-specific costs.
        transition_cost=np.full((3, 2, 3), 0.1),
        outcome_labels=("H", "L"),
    )
    # A successful L/R continuation is latent-world dependent, not a private
    # branch label.  The two rows are therefore the same for both first
    # outcomes and are integrated against the posterior after the probe.
    continuation_value = np.zeros((3, 2, 3, 2, 2), dtype=float)
    continuation_value[0, :, 1, :, 0] = (1.0, 0.0)
    continuation_value[0, :, 2, :, 0] = (0.0, 1.0)
    return BayesBelief(np.asarray([[0.0], [1.0]]), np.asarray([0.5, 0.5])), candidates, continuation_value


def test_bayes_update_and_predictive_table() -> None:
    posterior, evidence = bayes_update(np.asarray([0.25, 0.75]), np.asarray([0.8, 0.2]))
    assert np.isclose(evidence, 0.35)
    assert np.allclose(posterior, [4.0 / 7.0, 3.0 / 7.0])

    predictive, table = posterior_table(
        np.asarray([0.25, 0.75]),
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )
    assert np.allclose(predictive, [0.35, 0.65])
    assert np.allclose(table[0], [4.0 / 7.0, 3.0 / 7.0])
    assert np.allclose(table[1], [1.0 / 13.0, 12.0 / 13.0])


def test_centroid_kernel_has_valid_detection_and_miss() -> None:
    kernel, labels = centroid_outcome_kernel(
        centroid=np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        scale=0.1,
        detect_probability=np.asarray([[0.8, 0.7], [1.0, 1.0]]),
        y_edges=(-np.inf, 0.5, np.inf),
        valid_mask=np.asarray([[True, False], [True, True]]),
        return_labels=True,
    )
    assert labels == ("miss", "y_bin_0", "y_bin_1")
    assert np.allclose(kernel.sum(axis=-1), 1.0)
    assert np.isclose(kernel[0, 1, 0], 1.0)  # checked-invalid candidate is miss
    assert kernel[0, 0, 0] > 0.0  # valid detection retains miss probability
    assert kernel[1, 0, 1] > 0.0 and kernel[1, 1, 2] > 0.0


def test_strict_feedback_gap_and_shared_costs() -> None:
    belief, candidates, continuation_value = _strict_feedback_case()
    result = compare_methods(belief, candidates, continuation_value=continuation_value)
    a, b = result["A"], result["B"]
    assert a.first_action_id == b.first_action_id == "probe"
    # The first probe costs the same for both arms; A pays one fixed
    # continuation expectation while B can choose L/R after H/L.
    assert np.isclose(a.value, 0.4)  # .5 success - .1 common transition cost
    assert np.isclose(b.value, 0.9)  # 1.0 success - .1 common transition cost
    assert a.continuation_action_ids == ("L", "L")
    assert b.continuation_action_ids == ("L", "R")
    assert np.array_equal(a.feasible_after, b.feasible_after)
    assert np.allclose(a.transition_costs, b.transition_costs)


def test_null_case_has_no_feedback_gain() -> None:
    kernels = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    candidates = CheckedCandidateSet(
        ("p", "q"),
        kernels,
        after_action_feasible=np.ones((2, 2, 2), dtype=bool),
        transition_cost=np.full((2, 2, 2), 0.3),
    )
    continuation = np.ones((2, 2, 2, 2, 2), dtype=float)
    # Every policy has the same objective, but finite equal costs remain part
    # of both arms' values.
    result = compare_methods(
        BayesBelief(np.asarray([[0.0], [1.0]]), np.asarray([0.5, 0.5])),
        candidates,
        continuation_value=continuation,
    )
    assert np.isclose(result["A"].value, result["B"].value)
    assert result["A"].first_action_id == result["B"].first_action_id == "p"
    assert result["A"].continuation_action_ids == ("p", "p")
    # B leaves the impossible second outcome explicitly unassigned rather than
    # inventing a branch; on the only live outcome it agrees with A.
    assert result["B"].continuation_action_ids == ("p", None)


def test_compact_scan_pair_strict_gap_and_stop_branch() -> None:
    # Same strict witness as above, in the compact uniform-particle API.
    likelihood = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # probe
            [[1.0, 0.0], [1.0, 0.0]],  # no-probe control
        ]
    )
    captures = np.asarray([[1.0, 0.0], [0.0, 1.0]])  # L/R tails by world
    tail_cost = np.full((2, 2), 0.1)
    stop = np.asarray([[False, True], [False, True]])
    first_value = np.asarray([0.0, -0.2])
    a_first, a_suffix, a_details = plan_scan_pair(
        first_value, likelihood, captures, tail_cost, stop, method="A", discount=1.0
    )
    b_first, b_suffix, b_details = plan_scan_pair(
        first_value, likelihood, captures, tail_cost, stop, method="B", discount=1.0
    )
    assert a_first == b_first == 0
    # A must use one fixed tail; B can respond to the live H branch.  The L
    # branch is an explicit first-stage stop and therefore has suffix -1.
    assert np.array_equal(a_suffix, [0, -1])
    assert np.array_equal(b_suffix, [0, -1])
    # This stop placement makes the strict witness one-sided; the other strict
    # smoke above leaves both outcomes live.  The returned details still expose
    # the exact same-first gap and JSON-safe branch table.
    assert np.isclose(a_details["same_first_gap"], 0.0)
    assert np.isclose(b_details["same_first_gap"], 0.0)
    assert np.isclose(a_details["support_weighted_alternate_branch_mass"], 0.0)
    assert isinstance(b_details["joint"], list)

    live_stop = np.zeros_like(stop)
    _, live_suffix, live_details = plan_scan_pair(
        first_value, likelihood, captures, tail_cost, live_stop, method="B", discount=1.0
    )
    assert np.array_equal(live_suffix, [0, 1])
    assert np.isclose(live_details["same_first_gap"], 0.5)
    assert np.isclose(live_details["support_weighted_alternate_branch_mass"], 0.5)


def test_compact_scan_pair_null_case() -> None:
    likelihood = np.asarray([[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
    captures = np.asarray([[0.4, 0.4], [0.4, 0.4]])
    tail_cost = np.full((2, 2), 0.05)
    stop = np.zeros((2, 2), dtype=bool)
    a_first, a_suffix, a_details = plan_scan_pair(
        np.zeros(2), likelihood, captures, tail_cost, stop, method="A", discount=.9
    )
    b_first, b_suffix, b_details = plan_scan_pair(
        np.zeros(2), likelihood, captures, tail_cost, stop, method="B", discount=.9
    )
    assert a_first == b_first == 0
    assert np.array_equal(a_suffix, b_suffix)
    assert np.isclose(a_details["same_first_gap"], 0.0)
    assert np.isclose(b_details["same_first_gap"], 0.0)
    assert np.isclose(b_details["support_weighted_alternate_branch_mass"], 0.0)


if __name__ == "__main__":
    for name in sorted(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("decision_core smoke tests passed")


"""Independent small-case enumeration and stopping/filtering regression checks."""
import numpy as np
import pytest

from planner import Planner, initial_belief


def _planner(joint, gate, transition=None, threshold=0.65, qualified=None):
    correct = joint[..., 0, :] * gate
    if qualified is not None:
        correct *= qualified[:, None, :, None]
    return Planner(
        joint.sum(axis=-2), correct, joint, gate,
        np.eye(2) if transition is None else transition, threshold,
    )


def _discrimination_toy():
    # Action zero discriminates the worlds without a stop. Actions one and
    # two emit the same gate outcome in both worlds but qualify only one.
    # Action three provides an expensive, 0.7-correct fallback.
    joint = np.zeros((2, 2, 4, 4, 4))
    joint[0, :, 0, 2, 1] = 1.0
    joint[1, :, 0, 2, 2] = 1.0
    for world in range(2):
        joint[world, :, world + 1, 0, 3] = 1.0
        joint[world, :, 2 - world, 1, 3] = 1.0
    joint[:, :, 3, 0, 3] = 0.7
    joint[:, :, 3, 1, 3] = 0.3
    planner = _planner(joint, np.array([False, False, False, True]))
    belief = initial_belief(2, np.array([0.7, 0.3]))
    costs = np.tile([0.5, 1.0, 1.0, 3.0], (4, 1))
    return planner, belief, costs


def test_exact_adaptive_and_fixed_values_and_commitment():
    planner, belief, costs = _discrimination_toy()
    available = np.ones(4, dtype=bool)
    args = (belief, 0, 10.0, 2, available, costs, costs)
    one = planner.plan("A_task", *args)
    adaptive = planner.plan("B_two", *args)
    fixed = planner.plan("B_fixed", *args)
    assert one.action == fixed.action == 3
    np.testing.assert_allclose(one.scores, [0.0, 0.0, 0.0, 4.9])
    np.testing.assert_allclose(fixed.scores, [4.55, 4.2, 4.2, 4.9])
    np.testing.assert_allclose(adaptive.scores, [8.5, 4.2, 4.2, 4.9])
    assert adaptive.action == 0
    np.testing.assert_array_equal(adaptive.continuations, [-1, 1, 2, -1])
    np.testing.assert_allclose(adaptive.continuation_scores, [0.0, 4.25, 4.25, 0.0])

    limited = np.array([True, True, True, False])
    fixed = planner.plan("B_fixed", belief, 0, 10.0, 2, limited, costs, costs)
    assert fixed.action == 0
    assert fixed.fixed_action == 1
    np.testing.assert_array_equal(fixed.continuations, [-1, 1, 1, -1])
    second_belief = planner.advance(belief, 0, 2)
    committed = planner.plan(
        "B_fixed", second_belief, 0, 9.5, 1, limited, costs, costs, pending=1,
    )
    uncommitted = planner.plan("B_fixed", second_belief, 0, 9.5, 1, limited, costs, costs)
    assert committed.action == 1
    assert uncommitted.action == 2
    unaffordable = planner.plan(
        "B_fixed", second_belief, 0, 0.8, 1, limited, costs, costs, pending=1,
    )
    assert unaffordable.action is None


def test_capability_removed_only_for_second_action():
    planner, belief, costs = _discrimination_toy()
    available = np.array([True, True, False, False])
    constrained = planner.plan("B_two", belief, 0, 10.0, 2, available, costs, costs)
    unconstrained = planner.plan("B_no_future_cap", belief, 0, 10.0, 2, available, costs, costs)
    assert constrained.action == unconstrained.action == 0
    assert constrained.scores[0] == pytest.approx(4.25)
    assert unconstrained.scores[0] == pytest.approx(8.5)
    assert unconstrained.continuations[2] == 2
    assert np.isnan(unconstrained.scores[2:]).all()

    joint_limited = costs.copy()
    joint_limited[0, 2] = 4.0
    available[:] = True
    constrained = planner.plan("B_two", belief, 0, 10.0, 2, available, joint_limited, costs)
    unconstrained = planner.plan("B_no_future_cap", belief, 0, 10.0, 2, available, joint_limited, costs)
    assert constrained.scores[0] == pytest.approx(7.0)
    assert unconstrained.scores[0] == pytest.approx(8.5)

    for method in ("B_two", "B_fixed", "B_no_future_cap"):
        final = planner.plan(method, belief, 0, 10.0, 1, available, joint_limited, costs)
        assert final.action == 3
        np.testing.assert_allclose(final.scores, [0.0, 0.0, 0.0, 4.9])


def test_threshold_is_inclusive_and_wrong_source_does_not_count():
    joint = np.zeros((1, 2, 4, 4, 2))
    for action, mass in enumerate((0.6499, 0.65, 0.0, 1.0)):
        joint[:, :, action, 0, 1] = mass
        joint[:, :, action, 1, 1] = 1 - mass
    # An impossible gate outcome cannot trigger a stop.
    joint[:, :, 3, 0, 1] = 0.0
    joint[:, :, 3, 3, 0] = 1.0
    planner = _planner(joint, np.array([False, True]))
    belief = initial_belief(1, np.array([1.0, 0.0]))
    assert not planner.assess(belief, 0).accepted[1]
    assert planner.assess(belief, 1).accepted[1]
    assert not planner.assess(belief, 2).accepted[1]
    assert not planner.assess(belief, 3).accepted[1]
    costs = np.ones((4, 4))
    plan = planner.plan("A_task", belief, 0, 5.0, 1, np.ones(4, dtype=bool), costs, costs)
    assert plan.action == 1
    np.testing.assert_allclose(plan.scores, [0.0, 2.6, 0.0, 0.0])


def test_joint_world_mode_filter_uses_likelihood_and_advance_in_order():
    joint = np.zeros((2, 2, 1, 4, 3))
    p0 = np.array([[0.70, 0.05], [0.15, 0.10]])
    p1 = np.array([[0.05, 0.50], [0.30, 0.60]])
    joint[:, :, 0, 0, 1] = p0
    joint[:, :, 0, 1, 1] = p1
    joint[:, :, 0, 3, 0] = 1 - p0 - p1
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    planner = _planner(joint, np.array([False, True, False]), transition)
    prior = np.array([[0.45, 0.05], [0.10, 0.40]])
    weighted = prior * (p0 + p1)
    expected = (weighted / weighted.sum()) @ transition
    np.testing.assert_allclose(planner.advance(prior, 0, 1), expected)
    source_zero = prior * p0
    hard_expected = (source_zero / source_zero.sum()) @ transition
    np.testing.assert_allclose(planner.advance(prior, 0, 1, hard=True), hard_expected)

    iid = _planner(joint, np.array([False, True, False]), np.tile([0.7, 0.3], (2, 1)))
    np.testing.assert_allclose(iid.advance(prior, 0, 1).sum(axis=0), [0.7, 0.3])
    with pytest.raises(ValueError, match="zero model support"):
        planner.advance(prior, 0, 2)
    with pytest.raises(ValueError, match="zero model support"):
        planner.advance(prior, 0, 2, hard=True)


def _exact_value(planner, belief, cost, free_cost, available, current, first, horizon, method):
    expected = 0.0
    remaining = horizon - cost[current, first]
    contributions = np.zeros((planner.outcome_count, planner.action_count))
    second_cost = free_cost if method == "B_no_future_cap" else cost
    for outcome in range(planner.outcome_count):
        likelihood = planner.likelihood[:, :, first, outcome]
        probability = float(np.sum(belief * likelihood))
        correct = float(np.sum(belief * planner.correct_joint[:, :, first, outcome]))
        if probability == 0.0:
            continue
        if planner.gate[outcome] and correct / probability >= planner.threshold:
            expected += remaining * correct
            continue
        after = ((belief * likelihood) / probability) @ planner.transition
        for second in range(planner.action_count):
            if second_cost[first, second] > remaining:
                continue
            if method != "B_no_future_cap" and not available[second]:
                continue
            for last in np.flatnonzero(planner.gate):
                mass = float(np.sum(after * planner.likelihood[:, :, second, last]))
                success = float(np.sum(after * planner.correct_joint[:, :, second, last]))
                if mass > 0.0 and success / mass >= planner.threshold:
                    contributions[outcome, second] += probability * success * (
                        remaining - second_cost[first, second]
                    )
    continuation = (
        contributions.sum(axis=0).max() if method == "B_fixed"
        else contributions.max(axis=1).sum()
    )
    return expected + continuation


@pytest.mark.parametrize("method", ["B_two", "B_fixed", "B_no_future_cap"])
def test_full_branch_enumeration_with_nontrivial_mode_transition(method):
    rng = np.random.default_rng(84617)
    joint = rng.dirichlet(np.full(4 * 5, 0.6), size=(3, 2, 4)).reshape(3, 2, 4, 4, 5)
    transition = np.array([[0.83, 0.17], [0.31, 0.69]])
    gate = np.array([False, True, False, True, True])
    planner = _planner(joint, gate, transition, threshold=0.37)
    prior = rng.dirichlet(np.ones(6)).reshape(3, 2)
    free_cost = rng.uniform(0.5, 3.5, size=(4, 4))
    cost = free_cost + rng.uniform(0.0, 1.3, size=(4, 4))
    available = np.array([True, True, False, True])
    horizon = 5.5
    current = 1
    result = planner.plan(method, prior, current, horizon, 2, available, cost, free_cost)
    expected = [
        _exact_value(planner, prior, cost, free_cost, available, current, first, horizon, method)
        if permitted else np.nan
        for first, permitted in enumerate(available)
    ]
    np.testing.assert_allclose(result.scores, expected, atol=1e-14)
    assert result.action == np.nanargmax(expected)

"""Entry-state and information-flow tests for the v3 joint program kernel."""
from copy import deepcopy
import numpy as np

from leg_program_bench.core_execution import Tree
from leg_program_bench.run_joint_core_entry import entry_distance, predict_one, evaluate_tree, posterior_weights


def kernel(action, entry, exit_q, duration, *, audit=True, public=True, name="row"):
    return {"record_id": name, "action": action, "entry_q": list(entry), "entry_qdot": [0.] * 8,
            "exit_q": list(exit_q), "exit_qdot": [0.] * 8, "duration": duration,
            "public_success": public, "independent_success": audit,
            "success": bool(public and audit), "safe_exit": True, "frame_count": 10, "feature": .1}


def test_entry_support_checks_position_and_velocity():
    record = kernel("a:-1", [0.] * 8, [0.] * 8, 1.)
    assert entry_distance(np.zeros(8), np.zeros(8), record) == 0.
    q = np.zeros(8); q[0] = .061
    assert entry_distance(q, np.zeros(8), record) > 1.
    qd = np.zeros(8); qd[2] = .121
    assert entry_distance(np.zeros(8), qd, record) > 1.


def test_sampled_joint_exit_is_next_entry_not_reset():
    initial = np.zeros(8); after_a = initial.copy(); after_a[2] = .3
    bundle = {"near": .1, "kernels": [
        kernel("a:-1", initial, after_a, 2., name="a"),
        kernel("probe:0", initial, initial, 1., name="wrong_entry"),
        kernel("probe:0", after_a, initial, 8., name="correct_entry")]}
    result = predict_one(bundle, initial, Tree("fixed", "fixed", -1))
    assert result["success"] and result["supported"]
    assert result["duration"] == 10.
    assert result["calls"][-1]["record_id"] == "correct_entry"


def test_independent_audit_changes_score_without_changing_policy():
    q = np.zeros(8)
    bundle = {"near": .1, "kernels": [kernel("a:-1", q, q, 2.), kernel("probe:0", q, q, 3.)]}
    tree = Tree("fixed", "fixed", -1)
    good = predict_one(bundle, q, tree)
    failed_audit = deepcopy(bundle)
    failed_audit["kernels"][0]["independent_success"] = False
    failed_audit["kernels"][0]["success"] = False
    bad = predict_one(failed_audit, q, tree)
    assert good["success"] and not bad["success"]
    assert [r["action"] for r in good["calls"]] == [r["action"] for r in bad["calls"]]
    assert good["duration"] == bad["duration"]


def test_outside_entry_support_retains_zero_one_uncertainty():
    q = np.zeros(8); after_a = q.copy(); after_a[0] = .2
    bundle = {"near": .1, "kernels": [kernel("a:-1", q, after_a, 2.), kernel("probe:0", q, q, 3.)]}
    result = evaluate_tree([bundle], [1.], q, Tree("fixed", "fixed", -1), [20., 40.])
    assert result["unsupported_mass"] == 1.
    assert result["completion_lower"] == [0., 0.]
    assert result["completion_upper"] == [1., 1.]


def test_unreleased_old_content_does_not_condition_object_posterior():
    bundles = [{"near": -.1}, {"near": .1}]
    np.testing.assert_equal(posterior_weights(bundles, -.1, visible=False),
                            posterior_weights(bundles, .1, visible=False))
    assert posterior_weights(bundles, -.1, visible=True)[0] > .99

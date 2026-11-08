from __future__ import annotations

import numpy as np
import pytest

from leg_program_bench.core_identifiability import (
    FiniteOptionTree,
    binary_joint,
    binary_pairing_audit,
    common_policy_audit,
    exact_completion_cdf,
    monte_carlo_completion_cdf,
    triple_probe_audit,
)


def test_binary_association_keeps_margins_and_changes_task_value():
    report = binary_pairing_audit()
    assert report["all_margins_equal"]
    assert report["pairing_changes_task_value"]
    assert report["task_value_range"] == pytest.approx([0.1, 0.9])
    assert report["unpaired_margin_only_interval"] == [0.0, 1.0]
    assert all(sum(binary_joint(0.8).values()) == pytest.approx(1.0) for _ in [0])


def test_common_policy_calls_shared_maxmin_solver():
    report = common_policy_audit()
    assert np.allclose(report["mixture"], [0.5, 0.5], atol=1e-9)
    assert report["lower"] == pytest.approx(0.5)
    assert report["upper"] == pytest.approx(1.0)
    assert report["shared_policy_not_model_oracle"]


def test_pairwise_parity_probe_is_identical_but_triple_probe_resolves_world():
    report = triple_probe_audit()
    assert report["all_pairwise_probes_identical"]
    assert report["full_triple_total_variation"] == pytest.approx(1.0)
    assert report["additional_pair_only_data_changes_model_set"] is False
    assert report["pair_only_common_policy_lower"] == pytest.approx(0.5)
    assert report["triple_probe_policy_value"] == pytest.approx(1.0)


def test_release_gate_option_prediction_matches_independent_monte_carlo():
    tree = FiniteOptionTree()
    deadlines = (2.0, 2.4, 2.8, 3.0)
    predicted = exact_completion_cdf(tree, "adaptive_old", deadlines, release_time=0.8, retain_old=True)
    measured = monte_carlo_completion_cdf(
        tree,
        "adaptive_old",
        deadlines,
        release_time=0.8,
        retain_old=True,
        n=12_000,
        seed=17,
    )
    assert np.max(np.abs(predicted - measured)) < 0.025
    assert predicted[0] == pytest.approx(0.0)
    assert predicted[-1] == pytest.approx(1.0)


def test_late_release_wait_and_hidden_old_packet_requires_fresh_probe():
    tree = FiniteOptionTree()
    late = tree.run(1, "adaptive_old", release_time=0.8, retain_old=True, run_id=1)
    hidden = tree.run(1, "adaptive_old", release_time=0.0, retain_old=False, run_id=2)
    assert "WaitForRelease" in late.trace
    assert "Probe" in hidden.trace
    assert late.success and hidden.success
    # Waiting for a release at 0.8 is itself an allowed option and is cheaper
    # here than acquiring a replacement fresh probe (0.9); it is not modeled as
    # an automatic delay penalty.
    assert late.duration < hidden.duration
    assert "FreshRecheck" in hidden.trace


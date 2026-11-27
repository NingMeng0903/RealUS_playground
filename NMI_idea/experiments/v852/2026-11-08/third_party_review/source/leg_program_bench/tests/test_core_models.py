from copy import deepcopy

import numpy as np
import pytest

from work.leg_program_bench.core_models import ObjectBundleModel, common_plan


def _rows(correlation=1, n=8):
    rows = []
    for i in range(n):
        near = -1.0 if i % 2 == 0 else 1.0
        probe = near if correlation > 0 else -near
        rows.append(
            {
                "object_id": f"obj-{i}",
                "stratum": "pose0" if i < n // 2 else "pose30",
                "near": near,
                "probe": probe,
                "programs": {
                    "direct": {
                        "success": bool(i % 2),
                        "duration": float(i + 1),
                        "status": "ok" if i % 2 else "timeout",
                        "exit": [float(i), 0.0],
                    },
                    "probe": {
                        "success": True,
                        "duration": 2.0,
                        "status": "ok",
                        "exit": [0.0, float(i)],
                    },
                },
            }
        )
    return rows


def _legacy_candidate_scenarios(model, evidence, n_models, seed):
    """Frozen pre-optimization oracle for the bootstrap feature posterior.

    This intentionally retains the old full-row deep copy.  It is test-only:
    the production implementation is allowed to omit ``programs`` because
    ``weights`` never reads program outcomes.
    """
    rng = np.random.default_rng(model.seed if seed is None else int(seed))
    scenarios = []
    for _ in range(int(n_models)):
        sample_indices = rng.choice(len(model._rows), size=len(model._rows), replace=True)
        resampled_rows = []
        for position, original_index in enumerate(sample_indices):
            row = deepcopy(model._rows[int(original_index)])
            row["object_id"] = f"{row['object_id']}__bootstrap_{position}"
            resampled_rows.append(row)
        bootstrap_mode = "full_history" if model.mode == "full_history" else "paired"
        bootstrap_model = ObjectBundleModel(
            resampled_rows,
            mode=bootstrap_mode,
            seed=model.seed,
            bandwidth=model.bandwidth,
        )
        local = bootstrap_model.weights(evidence)
        mapped = np.zeros(len(model._rows), dtype=float)
        for picked, original_index in enumerate(sample_indices):
            mapped[int(original_index)] += float(local[picked])
        scenarios.append(mapped / mapped.sum())
    return np.asarray(scenarios, dtype=float)


def test_paired_draw_keeps_program_outcomes_from_one_object():
    model = ObjectBundleModel(_rows(), mode="paired", seed=3)
    samples = model.draw({"near": 1.0, "probe": 1.0, "stratum": "pose0"}, np.random.default_rng(4), n=30)
    by_id = {row["object_id"]: row for row in model.rows}
    for sample in samples:
        source = by_id[sample["object_id"]]
        assert sample["programs"] == source["programs"]
        assert "u" not in sample and "v" not in sample


def test_shuffle_preserves_stratum_marginals_and_program_rows():
    model = ObjectBundleModel(_rows(), mode="shuffled", seed=9)
    audit = model.audit()["shuffle_audit"]
    assert audit["applied"] is True
    assert audit["within_stratum"] is True
    assert audit["near_multiset_by_stratum_preserved"] is True
    assert audit["program_rows_unchanged"] is True
    assert audit["probe_rows_unchanged"] is True
    for original, shuffled in zip(_rows(), model.rows):
        assert original["programs"] == shuffled["programs"]


def test_public_evidence_rejects_hidden_oracle_keys():
    model = ObjectBundleModel(_rows())
    with pytest.raises(ValueError, match="hidden or oracle"):
        model.weights({"near": 0.0, "hidden_u": 1.0})
    with pytest.raises(ValueError, match="hidden or oracle"):
        model.weights({"near": 0.0, "true_geometry": 1.0})
    with pytest.raises(ValueError, match="unknown evidence"):
        model.weights({"near": 0.0, "unregistered_feature": 1.0})


def test_positive_and_negative_association_change_posterior():
    positive = ObjectBundleModel(_rows(correlation=1), bandwidth=0.2)
    negative = ObjectBundleModel(_rows(correlation=-1), bandwidth=0.2)
    p_pos = positive.weights({"near": 1.0, "probe": 1.0})
    p_neg = negative.weights({"near": 1.0, "probe": -1.0})
    # Positive table has +/+ objects at odd indices.  For the negative table,
    # the matching +/− pair is also at odd indices: the learned association is
    # visible in the joint evidence query, without exposing the generator sign.
    assert p_pos[1] > p_pos[0]
    assert p_neg[1] > p_neg[0]


def test_paired_structure_and_full_history_are_distinct_models():
    rows = _rows()
    # Keep sign coupling fixed but make amplitudes different; paired mode only
    # uses the sign structure when probe is absent, while full history keeps the
    # continuous near amplitude.
    rows[0]["near"] = -0.1
    rows[2]["near"] = -3.0
    rows[4]["near"] = -0.2
    rows[6]["near"] = -2.5
    paired = ObjectBundleModel(rows, mode="paired", bandwidth=0.2)
    full = ObjectBundleModel(rows, mode="full_history", bandwidth=0.2)
    paired_weights = paired.weights({"near": -0.15})
    full_weights = full.weights({"near": -0.15})
    assert not np.allclose(paired_weights, full_weights)
    assert paired.summary()["posterior_structure"].startswith("laplace_")
    assert full.summary()["posterior_structure"] == "continuous_joint_rbf_knn"


def test_common_plan_differs_from_model_specific_oracle():
    result = common_plan(np.asarray([[1.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(result["weights"], [0.5, 0.5])
    assert result["lower"] == pytest.approx(0.5)
    assert result["upper"] == pytest.approx(1.0)
    assert result["policy_is_common"] is True
    assert result["model_specific_oracle_not_used"] is True
    assert result["tie_break"]["enabled"] is False


def test_common_plan_cost_tie_break_keeps_robust_value():
    # Both policies are equally safe in every model, so the second LP should
    # select the cheaper policy without changing L or U.
    result = common_plan(np.ones((3, 2)), costs=[5.0, 1.0])
    assert np.allclose(result["weights"], [0.0, 1.0], atol=1e-8)
    assert result["lower"] == pytest.approx(1.0)
    assert result["upper"] == pytest.approx(1.0)
    assert result["tie_break"]["expected_cost"] == pytest.approx(1.0)
    assert result["tie_break"]["value_after_tie_break"] >= result["lower"] - 1e-9


def test_common_plan_cost_tie_break_cannot_lower_opposing_model_floor():
    result = common_plan(np.asarray([[1.0, 0.0], [0.0, 1.0]]), costs=[0.0, 10.0])
    assert result["lower"] == pytest.approx(0.5)
    assert result["tie_break"]["value_after_tie_break"] >= 0.5 - 1e-9
    assert result["weights"][0] == pytest.approx(0.5, abs=1e-8)
    assert result["weights"][1] == pytest.approx(0.5, abs=1e-8)


def test_bootstrap_envelope_is_explicitly_empirical_and_missing_evidence_is_wider():
    model = ObjectBundleModel(_rows(), seed=12)
    complete = model.candidate_weights({"near": 1.0, "probe": 1.0}, n_models=25, seed=7)
    missing = model.candidate_weights({"near": 1.0}, n_models=25, seed=7)
    assert complete.metadata["strict_confidence_set"] is False
    assert complete.scenarios.shape == (25, len(_rows()))
    assert np.allclose(complete.scenarios.sum(axis=1), 1.0)
    # A missing probe should not yield a more concentrated empirical posterior
    # than the same query with both public features available.
    entropy = lambda matrix: np.mean(-np.sum(matrix * np.log(np.maximum(matrix, 1e-12)), axis=1))
    assert entropy(missing.scenarios) >= entropy(complete.scenarios) - 1e-6
    assert set(complete.coupling_weights) == {"positive", "negative"}
    for label, audit in complete.coupling_audit.items():
        assert audit["program_rows_unchanged"] is True
        assert audit["probe_multiset_by_stratum_preserved"] is True
        assert audit["generated_from_public_near_probe"] is True
    assert complete.metadata["coupling_scenarios"] == ["positive", "negative"]
    scenarios = model.scenario_rows({"near": 1.0})
    assert set(scenarios) == {"observed", "positive", "negative"}
    for scenario in scenarios.values():
        assert np.isclose(np.asarray(scenario["weights"]).sum(), 1.0)
        assert scenario["audit"]["strict_confidence_set"] is False


@pytest.mark.parametrize("mode", ["paired", "full_history"])
def test_bootstrap_matches_frozen_full_row_oracle_and_replays_rng(mode):
    model = ObjectBundleModel(_rows(), mode=mode, seed=12, bandwidth=0.7)
    evidence = {"near": 1.0, "probe": 1.0, "stratum": "pose0"}
    optimized = model.candidate_weights(evidence, n_models=7, seed=31)
    frozen = _legacy_candidate_scenarios(model, evidence, n_models=7, seed=31)
    assert np.allclose(optimized.scenarios, frozen)
    replay = model.candidate_weights(evidence, n_models=7, seed=31)
    assert np.allclose(replay.scenarios, optimized.scenarios)


def test_feature_only_coupling_models_match_full_rows_and_rollout_rows_remain_complete():
    model = ObjectBundleModel(_rows(), seed=5, bandwidth=0.5)
    for label in ("positive", "negative"):
        feature_model = model._coupling_feature_model(label)
        full_model = model.coupling_model(label)
        for evidence in ({"near": 1.0}, {"near": 1.0, "probe": 1.0}, {}):
            assert np.allclose(feature_model.weights(evidence), full_model.weights(evidence))
    scenarios = model.scenario_rows({"near": 1.0})
    assert "direct" in scenarios["positive"]["rows"][0]["programs"]
    assert "__weights_only__" not in scenarios["positive"]["rows"][0]["programs"]


def test_posterior_update_keeps_object_identity_and_independent_mode_marks_provenance():
    model = ObjectBundleModel(_rows(), mode="paired")
    weights = model.weights({"near": 1.0})
    updated = model.posterior_update(
        weights,
        {"program_id": "direct", "success": True, "status": "ok", "duration": 7.0},
    )
    assert np.isclose(updated.sum(), 1.0)
    assert updated[1] > updated[0]
    with pytest.raises(ValueError, match="hidden or oracle"):
        model.posterior_update(weights, {"program_id": "direct", "hidden_v": 1})

    independent = ObjectBundleModel(_rows(), mode="independent")
    sampled = independent.draw({"near": 0.0}, np.random.default_rng(2), n=1)[0]
    assert "independent_source_object_ids" in sampled
    assert set(sampled["independent_source_object_ids"].values()) <= set(independent.object_ids)
    assert independent.summary()["within_object_program_outcomes_preserved"] is False


def test_posterior_update_prefers_public_success_over_audit_success():
    rows = _rows()
    for index, row in enumerate(rows):
        row["programs"]["direct"]["success"] = False
        row["programs"]["direct"]["public_success"] = index == 0
    model = ObjectBundleModel(rows, mode="paired")
    updated = model.posterior_update(
        np.full(len(rows), 1.0 / len(rows)),
        {"program_id": "direct", "success": True},
    )
    assert updated[0] > updated[1]

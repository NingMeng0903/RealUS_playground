"""Run-level acceptance tests; these fixtures are not held-out experiments."""
from copy import deepcopy

import numpy as np
import pytest

from peirastic.contact_qp.evaluation import FORCE_METRICS, force_metrics, paired_report


@pytest.fixture
def acceptance():
    return dict(scenarios=["healthy", "left_gap"], acceptance_seeds=[1000, 1001, 1002],
                design_seeds=[0, 1, 2], ablations=["baseline", "full", "matched_scan_speed", "matched_rocking"],
                bootstrap_seed=29, bootstrap_replicates=2000, repairable_scenarios=["left_gap"],
                quality_comparators=["baseline", "matched_scan_speed", "matched_rocking"])


def records_for(cfg):
    return [dict(scenario=s, seed=k, variant=v, status="complete", valid=True,
                 rmse=1.+k*.01, absolute_mean_bias=.2+k*.001, peak_absolute_error=2.+k*.01,
                 bad_path_m=.01 if v == "full" else .02,
                 valid_coverage_rate_m_s=.01, valid_coverage_fraction=.8)
            for s in cfg["scenarios"] for k in cfg["acceptance_seeds"] for v in cfg["ablations"]]


def test_force_metrics_use_current_signed_error_and_absolute_mean_bias():
    result = force_metrics([2., 4., 6., 8.])
    assert result == dict(rmse=np.sqrt(6.), absolute_mean_bias=1., peak_absolute_error=4.)
    assert force_metrics([3., 5.])["absolute_mean_bias"] == 0.


@pytest.mark.parametrize("force", [[], [[4.]], [np.nan], [np.inf]])
def test_force_metrics_reject_missing_nonfinite_or_nonvector_samples(force):
    with pytest.raises(ValueError):
        force_metrics(force)


def test_zero_force_margin_allows_exact_equality_and_requires_quality_gain(acceptance):
    records = records_for(acceptance)
    report = paired_report(records, acceptance)
    assert report["accepted"] and report["margin_n"] == 0.
    for metric in FORCE_METRICS:
        assert report["force"][metric]["pooled"]["upper95"] == 0.
    for row in records:
        row["bad_path_m"] = .02
    report = paired_report(records, acceptance)
    assert not report["accepted"]
    assert all(not row["passed"] for row in report["quality"].values())


@pytest.mark.parametrize("metric", FORCE_METRICS)
def test_even_tiny_positive_force_difference_fails_zero_margin(acceptance, metric):
    records = records_for(acceptance)
    for row in records:
        if row["variant"] == "full":
            row[metric] += 1e-10
    report = paired_report(records, acceptance)
    assert not report["accepted"]
    assert report["force"][metric]["pooled"]["upper95"] > 0.


def test_pooled_improvement_cannot_hide_one_scenario_force_regression(acceptance):
    records = records_for(acceptance)
    for row in records:
        if row["variant"] == "full":
            row["rmse"] += .01 if row["scenario"] == "healthy" else -.1
    result = paired_report(records, acceptance)["force"]["rmse"]
    assert result["pooled"]["upper95"] < 0.
    assert result["scenarios"]["healthy"]["upper95"] > 0.
    assert not result["passed"]


def test_bootstrap_resamples_paired_run_differences_and_equal_scenario_means(acceptance):
    records = records_for(acceptance)
    deltas = np.array([[-.3, -.1, .1], [-.1, -.2, -.4]])
    for row in records:
        if row["variant"] == "full":
            row["rmse"] += deltas[acceptance["scenarios"].index(row["scenario"]), row["seed"]-1000]
    rng = np.random.default_rng(acceptance["bootstrap_seed"])
    ids = rng.integers(0, 3, (acceptance["bootstrap_replicates"], 2, 3))
    means = np.column_stack([deltas[i][ids[:, i]].mean(axis=1) for i in range(2)]).mean(axis=1)
    result = paired_report(records, acceptance)["force"]["rmse"]["pooled"]
    assert result == pytest.approx(dict(mean=deltas.mean(), lower95=np.quantile(means, .05),
                                        upper95=np.quantile(means, .95)))
    assert paired_report(list(reversed(records)), acceptance)["force"]["rmse"]["pooled"] == result


@pytest.mark.parametrize("problem", ["missing", "duplicate", "aborted", "invalid", "nan"])
def test_missing_duplicate_failed_and_nonfinite_runs_never_dropped(acceptance, problem):
    records = records_for(acceptance)
    if problem == "missing":
        records.pop()
    elif problem == "duplicate":
        records.append(deepcopy(records[0]))
    elif problem == "aborted":
        records[0].update(status="aborted", reason="solver_failed")
    elif problem == "invalid":
        records[0]["valid"] = False
    else:
        records[0]["rmse"] = np.nan
    report = paired_report(records, acceptance)
    assert not report["accepted"]
    if problem == "missing":
        assert report["missing"]
    elif problem == "duplicate":
        assert report["duplicate_runs"]
    elif problem in ("aborted", "invalid"):
        assert report["failed_runs"]


@pytest.mark.parametrize("metric", ["valid_coverage_rate_m_s", "valid_coverage_fraction"])
def test_quality_bad_path_gain_cannot_hide_slower_coverage_or_lost_fraction(acceptance, metric):
    records = records_for(acceptance)
    for row in records:
        if row["variant"] == "full":
            row[metric] -= 1e-4
    assert not paired_report(records, acceptance)["accepted"]


def test_acceptance_rejects_wholly_missing_seed(acceptance):
    records = [r for r in records_for(acceptance) if r["seed"] != 1002]
    report = paired_report(records, acceptance)
    assert not report["accepted"]
    assert {missing[1] for missing in report["missing"]} == {1002}


def test_design_report_also_requires_every_prescribed_design_seed(acceptance):
    records = records_for(acceptance)
    for record in records:
        record["seed"] -= 1000
    assert paired_report(records, acceptance, require_acceptance_seeds=False)["accepted"]
    records = [record for record in records if record["seed"] != 2]
    report = paired_report(records, acceptance, require_acceptance_seeds=False)
    assert not report["accepted"]
    assert {missing[1] for missing in report["missing"]} == {2}

"""Frozen run-unit paired statistics. Correlated control frames are not replicates."""
from __future__ import annotations

import numpy as np

FORCE_METRICS = ("rmse", "absolute_mean_bias", "peak_absolute_error")


def force_metrics(current_force, target=4.):
    force = np.asarray(current_force, dtype=float)
    if force.ndim != 1 or not force.size or not np.isfinite(force).all():
        raise ValueError("finite current force samples required")
    error = force-target
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                absolute_mean_bias=float(abs(np.mean(error))),
                peak_absolute_error=float(np.max(np.abs(error))))


def paired_report(records, acceptance, *, require_acceptance_seeds=True):
    scenarios = acceptance["scenarios"]
    seeds = acceptance["acceptance_seeds"] if require_acceptance_seeds else acceptance["design_seeds"]
    expected = {(s, k, v) for s in scenarios for k in seeds for v in acceptance["ablations"]}
    indexed = {(r["scenario"], r["seed"], r["variant"]): r for r in records}
    missing = sorted(expected-set(indexed))
    duplicates = len(indexed) != len(records)
    failures = [dict(scenario=r["scenario"], seed=r["seed"], variant=r["variant"],
                     status=r["status"], reason=r.get("reason", "")) for r in records
                if r["status"] == "aborted" or not r.get("valid", False)]
    report = {"accepted": False, "missing": missing, "duplicate_runs": duplicates,
              "failed_runs": failures, "force": {}, "quality": {}, "margin_n": 0.0,
              "unit": "paired independent scenario/seed run"}
    if missing or duplicates:
        return report
    rng = np.random.default_rng(acceptance["bootstrap_seed"])
    n, ns = len(seeds), len(scenarios)
    indices = rng.integers(0, n, (acceptance["bootstrap_replicates"], ns, n))
    all_checks = not failures

    def differences(metric, comparator):
        full = np.array([[indexed[s, k, "full"].get(metric, np.nan) for k in seeds] for s in scenarios], dtype=float)
        baseline = np.array([[indexed[s, k, comparator].get(metric, np.nan) for k in seeds] for s in scenarios], dtype=float)
        return full-baseline

    def interval(deltas, chosen=None):
        selected = np.arange(ns) if chosen is None else np.array(chosen, dtype=int)
        sampled = np.stack([deltas[i][indices[:, i]].mean(axis=1) for i in selected], axis=1)
        means = sampled.mean(axis=1)
        return dict(mean=float(deltas[selected].mean()), lower95=float(np.quantile(means, .05)),
                    upper95=float(np.quantile(means, .95)))

    for metric in FORCE_METRICS:
        delta = differences(metric, "baseline")
        if not np.isfinite(delta).all():
            report["force"][metric] = {"passed": False, "reason": "invalid run metric"}
            all_checks = False
            continue
        pooled = interval(delta)
        families = {s: interval(delta, [i]) for i, s in enumerate(scenarios)}
        passed = pooled["upper95"] <= 0 and all(v["upper95"] <= 0 for v in families.values())
        report["force"][metric] = dict(pooled=pooled, scenarios=families, passed=passed)
        all_checks &= passed
    repairable = [scenarios.index(s) for s in acceptance["repairable_scenarios"]]
    for comparator in acceptance["quality_comparators"]:
        bad = differences("bad_path_m", comparator)
        rate = differences("valid_coverage_rate_m_s", comparator)
        fraction = differences("valid_coverage_fraction", comparator)
        if not all(np.isfinite(a).all() for a in (bad, rate, fraction)):
            report["quality"][comparator] = {"passed": False, "reason": "invalid run metric"}
            all_checks = False
            continue
        bad_ci, rate_ci = interval(bad, repairable), interval(rate, repairable)
        means = {scenarios[i]: float(fraction[i].mean()) for i in repairable}
        passed = bad_ci["upper95"] < 0 and rate_ci["lower95"] >= 0 and min(means.values()) >= 0
        report["quality"][comparator] = dict(bad_path=bad_ci, valid_rate=rate_ci,
                                                scenario_fraction_deltas=means, passed=passed)
        all_checks &= passed
    report["accepted"] = bool(all_checks)
    return report

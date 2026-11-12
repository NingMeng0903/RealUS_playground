"""Clustered paired statistics for continuation-policy experiments.

The first axis is the independent latent case.  A second axis is optional and
contains repeated acoustic tapes for the same case.  Repeats are averaged
within case before bootstrap resampling, so they cannot inflate the apparent
number of independent experiments.
"""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np


FIELD_NAMES = (
    "correct_completion",
    "any_wrong",
    "tcp_mm",
    "turn_deg",
    "motion_s",
    "rail_mm",
    "captures",
    "combined_s",
)

_COST_FIELDS = ("tcp_mm", "turn_deg", "motion_s")
_COST_MARGINS = {"tcp_mm": 2.0, "turn_deg": 2.0, "motion_s": 0.10}
_BOOTSTRAP_SAMPLES = 20_000


def _as_repeated(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2 and array.shape[1] == len(FIELD_NAMES):
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[2] != len(FIELD_NAMES):
        raise ValueError(
            f"{name} must have shape (cases, 8) or (cases, repeats, 8), got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one case and one repeat")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.all((array[..., :2] == 0) | (array[..., :2] == 1)):
        raise ValueError("correct_completion and any_wrong must be binary")
    return array


def _quantile_interval(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    return [float(x) for x in np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])]


def _bootstrap_case_means(case_values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return bootstrap means with case as the resampling unit."""
    n_cases, n_metrics = case_values.shape
    output = np.empty((_BOOTSTRAP_SAMPLES, n_metrics), dtype=float)
    batch = 512
    for start in range(0, _BOOTSTRAP_SAMPLES, batch):
        stop = min(start + batch, _BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, n_cases, size=(stop - start, n_cases))
        output[start:stop] = case_values[indices].mean(axis=1)
    return output


def _binomial_cdf(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0
    return sum(comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k + 1))


def _binomial_upper_975(k: int, n: int) -> float:
    """Exact one-sided 97.5% upper confidence bound for a binomial rate."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    # Invert the lower-tail test with a monotone binary search.  For x in
    # [0, 1], P[X <= k | p=x] = .025 at the upper confidence endpoint.
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2.0
        cdf = sum(comb(n, i) * mid**i * (1.0 - mid) ** (n - i) for i in range(k + 1))
        if cdf > 0.025:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2.0)


def _summary_rate(array: np.ndarray) -> dict[str, float]:
    return {
        "correct_completion": float(array[..., 0].mean()),
        "any_wrong": float(array[..., 1].mean()),
    }


def summarize(candidate: Any, baseline: Any, *, seed: int = 57216) -> dict[str, Any]:
    """Summarize a matched candidate-versus-baseline experiment.

    ``candidate`` and ``baseline`` contain, in order,
    ``correct_completion, any_wrong, tcp_mm, turn_deg, motion_s, rail_mm,
    captures, combined_s``.  The arrays must share the same case and repeat
    axes.  Repeated tapes are clustered within their latent case for all
    intervals and point contrasts.
    """
    candidate_array = _as_repeated(candidate, "candidate")
    baseline_array = _as_repeated(baseline, "baseline")
    if candidate_array.shape != baseline_array.shape:
        raise ValueError(
            f"candidate and baseline shapes differ: {candidate_array.shape} vs {baseline_array.shape}"
        )

    n_cases, n_repeats, _ = candidate_array.shape
    difference = candidate_array - baseline_array
    case_difference = difference.mean(axis=1)
    point_delta = difference.mean(axis=(0, 1))
    rng = np.random.default_rng(seed)
    bootstrap = _bootstrap_case_means(case_difference, rng)

    delta_ci = {
        field: _quantile_interval(bootstrap[:, index])
        for index, field in enumerate(FIELD_NAMES)
    }
    candidate_rates = _summary_rate(candidate_array)
    baseline_rates = _summary_rate(baseline_array)

    cost_indices = [FIELD_NAMES.index(field) for field in _COST_FIELDS]
    bonferroni_alpha = 0.05 / len(_COST_FIELDS)
    cost_ci = {
        field: _quantile_interval(bootstrap[:, index], alpha=bonferroni_alpha)
        for field, index in zip(_COST_FIELDS, cost_indices)
    }
    cost_point = {field: float(point_delta[index]) for field, index in zip(_COST_FIELDS, cost_indices)}
    cost_equivalent = {
        field: bool(
            cost_ci[field][0] >= -_COST_MARGINS[field]
            and cost_ci[field][1] <= _COST_MARGINS[field]
        )
        for field in _COST_FIELDS
    }
    cost_nonincrease = {field: bool(cost_ci[field][1] <= 0.) for field in _COST_FIELDS}
    cost_noninferior = {field: bool(cost_ci[field][1] <= _COST_MARGINS[field]) for field in _COST_FIELDS}

    success_case = case_difference[:, 0]
    wrong_case = case_difference[:, 1]
    success_wins = int(np.sum(success_case > 0.0))
    success_losses = int(np.sum(success_case < 0.0))
    wrong_harm_cases = int(np.sum(wrong_case > 0.0))
    wrong_benefit_cases = int(np.sum(wrong_case < 0.0))
    repeated = n_repeats > 1

    if repeated:
        inference = (
            "Repeated tapes are clustered by latent case. Case-level bootstrap intervals are "
            "descriptive and no binomial discordance inference is reported."
        )
        success_sign_p = None
        wrong_sign_p = None
        wrong_harm_upper = None
    else:
        informative_success = success_wins + success_losses
        success_sign_p = float(
            _binomial_cdf(success_losses, informative_success)
            if informative_success
            else 1.0
        )
        informative_wrong = wrong_harm_cases + wrong_benefit_cases
        wrong_sign_p = float(
            _binomial_cdf(wrong_benefit_cases, informative_wrong)
            if informative_wrong
            else 1.0
        )
        wrong_harm_upper = _binomial_upper_975(wrong_harm_cases, n_cases)
        inference = "One repeat per latent case. Exact paired sign tests and the one-sided harmful-discordance bound are reported."

    # This gate is intentionally only evaluated for the single-repeat setting,
    # where the requested exact discordance bound has its stated meaning.
    success_lower = delta_ci["correct_completion"][0]
    gate = bool(
        not repeated
        and point_delta[0] >= 0.05
        and success_lower > 0.0
        and wrong_harm_upper is not None
        and wrong_harm_upper <= 0.02
        and all(cost_equivalent.values())
    )

    return {
        "n_cases": int(n_cases),
        "n_repeats": int(n_repeats),
        "repeated_case_clustered": bool(repeated),
        "candidate_rate": candidate_rates,
        "baseline_rate": baseline_rates,
        "delta": {field: float(point_delta[index]) for index, field in enumerate(FIELD_NAMES)},
        "delta_ci95_case_clustered": delta_ci,
        "success_wins_losses_ties": [
            success_wins,
            success_losses,
            int(n_cases - success_wins - success_losses),
        ],
        "wrong_harm_benefit_ties_cases": [
            wrong_harm_cases,
            wrong_benefit_cases,
            int(n_cases - wrong_harm_cases - wrong_benefit_cases),
        ],
        "wrong_harm_975_upper": wrong_harm_upper,
        "success_sign_p_greater": success_sign_p,
        "wrong_harm_sign_p_greater": wrong_sign_p,
        "cost_delta": cost_point,
        "cost_bonferroni_ci95": cost_ci,
        "cost_margins": dict(_COST_MARGINS),
        "cost_equivalent": cost_equivalent,
        "cost_nonincrease": cost_nonincrease,
        "cost_noninferior_with_margin": cost_noninferior,
        "all_costs_equivalent": bool(all(cost_equivalent.values())),
        "all_costs_nonincrease": bool(all(cost_nonincrease.values())),
        "all_costs_noninferior_with_margin": bool(all(cost_noninferior.values())),
        "strong_gate_single_repeat": gate,
        "inference_note": inference,
        "bootstrap_samples": int(_BOOTSTRAP_SAMPLES),
        "bootstrap_seed": int(seed),
    }


__all__ = ["FIELD_NAMES", "summarize"]

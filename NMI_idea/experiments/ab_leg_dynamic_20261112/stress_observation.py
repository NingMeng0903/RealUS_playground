"""Stress observation law for the dynamic-leg scan.

This module is intentionally independent of the geometry and runner.  It
provides a small public observation kernel and a Gaussian assumed-density
filter (ADF) for the existing local linear model

    y = base + h @ theta + e.

The target noise is a declared Gaussian mixture.  Detection is gated to the
centroid aperture ``[-hw, hw]`` with probability ``p_detect``.  Independently,
at most one false centroid is emitted with probability ``false_rate`` and is
uniform over that aperture.  ``update_set`` marginalizes one unassigned
observation as that false centroid; no hidden truth or simulator state enters
the update.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Iterable, Sequence

import numpy as np


DEFAULT_NOISE_MIXTURE: tuple[tuple[float, float], ...] = ((0.9, 0.85), (0.1, 4.0))
DEFAULT_P_DETECT = 0.91
DEFAULT_FALSE_RATE = 0.06

__all__ = [
    "DEFAULT_FALSE_RATE",
    "DEFAULT_NOISE_MIXTURE",
    "DEFAULT_P_DETECT",
    "bin_kernel",
    "hit_probability",
    "update_set",
]

_TINY = np.finfo(float).tiny
_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _normal_cdf(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    flat = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / _SQRT_TWO)) for value in array.ravel()),
        dtype=float,
        count=array.size,
    )
    return flat.reshape(array.shape)


def _normal_pdf(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.exp(-0.5 * array * array) / _SQRT_TWO_PI


def _validate_noise_mixture(noise_mixture: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = np.asarray(noise_mixture, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_mixture must contain (weight, standard_deviation) pairs") from exc
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0:
        raise ValueError("noise_mixture must have shape (components, 2)")
    weights, scales = raw[:, 0], raw[:, 1]
    if not np.isfinite(raw).all() or (weights < 0.0).any() or (scales <= 0.0).any():
        raise ValueError("noise-mixture weights must be nonnegative and scales positive")
    total = float(weights.sum())
    if not total > 0.0:
        raise ValueError("noise-mixture weights must have positive mass")
    return weights / total, scales


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def _signal_table(signal: np.ndarray | float) -> np.ndarray:
    table = np.asarray(signal, dtype=float)
    if table.ndim == 0:
        table = table.reshape(1, 1)
    elif table.ndim == 1:
        table = table.reshape(1, -1)
    elif table.ndim != 2:
        raise ValueError("signal must be scalar, (particles,), or (actions, particles)")
    if not np.isfinite(table).all():
        raise ValueError("signal must be finite")
    return table


def hit_probability(
    signal: np.ndarray | float,
    hw: float,
    *,
    noise_mixture: Sequence[Sequence[float]] = DEFAULT_NOISE_MIXTURE,
    p_detect: float = DEFAULT_P_DETECT,
) -> np.ndarray:
    """Return gated genuine-detection probability for each predicted signal."""

    half_width = float(hw)
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("hw must be finite and positive")
    detect = _probability(p_detect, "p_detect")
    weights, scales = _validate_noise_mixture(noise_mixture)
    original = np.asarray(signal, dtype=float)
    table = _signal_table(original)
    mass = np.zeros_like(table, dtype=float)
    for weight, scale in zip(weights, scales):
        upper = (half_width - table) / scale
        lower = (-half_width - table) / scale
        mass += float(weight) * (_normal_cdf(upper) - _normal_cdf(lower))
    result = np.clip(float(detect) * mass, 0.0, 1.0)
    return result.reshape(original.shape) if original.ndim else result.reshape(())


def _validate_edges(edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise ValueError("edges must be a finite strictly increasing one-dimensional sequence")
    return values


def bin_kernel(
    signal: np.ndarray | float,
    edges: Sequence[float],
    *,
    noise_mixture: Sequence[Sequence[float]] = DEFAULT_NOISE_MIXTURE,
    false_rate: float = DEFAULT_FALSE_RATE,
    p_detect: float = DEFAULT_P_DETECT,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return binned public outcomes and their genuine-single contribution.

    The output arrays have shape ``(actions, particles, 1 + n_bins + 1)``.
    Their outcome order is ``miss, bin_0, ..., bin_{n-1}, multiple``.  A
    genuine target in a bin and no false centroid contributes to ``genuine``;
    a false-only centroid contributes to the corresponding public bin; and a
    target plus false centroid contributes to ``multiple``.  The first output
    is the full public kernel ``L`` and the second has the same shape but only
    genuine-single-bin mass.
    """

    bounds = _validate_edges(edges)
    table = _signal_table(signal)
    weights, scales = _validate_noise_mixture(noise_mixture)
    detect = _probability(p_detect, "p_detect")
    false = _probability(false_rate, "false_rate")
    n_actions, n_particles = table.shape
    n_bins = bounds.size - 1
    target_bins = np.zeros((n_actions, n_particles, n_bins), dtype=float)
    for weight, scale in zip(weights, scales):
        upper = (bounds[1:][None, None, :] - table[..., None]) / scale
        lower = (bounds[:-1][None, None, :] - table[..., None]) / scale
        target_bins += float(weight) * (_normal_cdf(upper) - _normal_cdf(lower))
    target_bins = np.maximum(target_bins, 0.0)
    target_gate = target_bins.sum(axis=-1)
    # A false centroid is uniform over the declared aperture represented by
    # edges.  Its bin masses sum exactly to one even with unequal bins.
    aperture = float(bounds[-1] - bounds[0])
    false_bins = np.diff(bounds) / aperture
    target_hit_bins = float(detect) * target_bins
    target_hit = target_hit_bins.sum(axis=-1)
    target_not_observed = np.maximum(0.0, 1.0 - target_hit)

    genuine = np.zeros((n_actions, n_particles, n_bins + 2), dtype=float)
    # Index 0 is miss; indexes 1:-1 are single-centroid bins; final is
    # multiple.  ``genuine`` means a public *single* target centroid, hence
    # the no-false factor belongs here.
    genuine[..., 1:-1] = (1.0 - false) * target_hit_bins
    public = genuine.copy()
    # A false-only centroid is visible whenever the target produced no public
    # centroid, including a target outside the aperture or a detector miss.
    public[..., 1:-1] += float(false) * target_not_observed[..., None] * false_bins[None, None, :]
    public[..., -1] = float(false) * target_hit
    public[..., 0] = (1.0 - false) * target_not_observed
    # Construction is normalized analytically; this correction removes only
    # floating-point tail error and keeps every row a legal probability row.
    normalizer = np.maximum(public.sum(axis=-1, keepdims=True), _TINY)
    public /= normalizer
    genuine /= normalizer
    labels = ("miss",) + tuple(f"bin_{i}" for i in range(n_bins)) + ("multiple",)
    return public, genuine, labels


def _symmetrize_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) * 0.5
    if not np.isfinite(symmetric).all():
        raise ValueError("posterior covariance became non-finite")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("posterior covariance eigendecomposition failed") from exc
    if float(eigenvalues.min()) < -1.0e-7 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise ValueError("posterior covariance is materially non-PSD")
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _truncated_component_moments(
    mean: float,
    variance: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    """Return ``(mass, E[Y|gate], Var[Y|gate])`` for one normal component."""

    scale = math.sqrt(max(float(variance), _TINY))
    alpha = (float(lower) - mean) / scale
    beta = (float(upper) - mean) / scale
    pa, pb = float(_normal_pdf(alpha)), float(_normal_pdf(beta))
    mass = float(_normal_cdf(beta) - _normal_cdf(alpha))
    if mass <= 1.0e-15:
        return 0.0, mean, variance
    delta = (pa - pb) / mass
    conditional_mean = mean + scale * delta
    edge_term_a = 0.0 if not math.isfinite(alpha) else alpha * pa
    edge_term_b = 0.0 if not math.isfinite(beta) else beta * pb
    conditional_variance = variance * max(
        0.0,
        1.0 + (edge_term_a - edge_term_b) / mass - delta * delta,
    )
    return mass, conditional_mean, conditional_variance


def _posterior_step(
    mean: np.ndarray,
    covariance: np.ndarray,
    h: np.ndarray,
    base: float,
    value: float | None,
    half_width: float,
    *,
    noise_mixture: Sequence[Sequence[float]],
    false_rate: float,
    p_detect: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One target-model ADF step, returning posterior and event likelihood."""

    weights, scales = _validate_noise_mixture(noise_mixture)
    detect = _probability(p_detect, "p_detect")
    _probability(false_rate, "false_rate")  # validate at the public boundary
    mean = np.asarray(mean, dtype=float)
    covariance = _symmetrize_psd(np.asarray(covariance, dtype=float))
    h = np.asarray(h, dtype=float)
    if mean.ndim != 1 or h.shape != mean.shape or covariance.shape != (mean.size, mean.size):
        raise ValueError("mean, covariance, and model row dimensions do not match")
    if not np.isfinite(mean).all() or not np.isfinite(h).all() or not math.isfinite(float(base)):
        raise ValueError("model and belief values must be finite")
    ch = covariance @ h
    latent_variance = float(h @ ch)
    y0 = float(base + h @ mean)
    prior_second = covariance + np.outer(mean, mean)

    if value is not None:
        observation = float(value)
        if not math.isfinite(observation):
            raise ValueError("observed centroids must be finite")
        if abs(observation) > half_width + 1.0e-12:
            # Public target detections are aperture-gated.  A clutter
            # observation outside the aperture is also impossible; callers
            # receive a zero evidence hypothesis rather than a truth label.
            return mean.copy(), covariance.copy(), 0.0
        component_mass = []
        component_means = []
        component_covariances = []
        for weight, scale in zip(weights, scales):
            variance = latent_variance + float(scale) ** 2
            sd = math.sqrt(max(variance, _TINY))
            density = float(weight) * float(_normal_pdf((observation - y0) / sd)) / sd
            component_mass.append(density)
            gain = ch / variance
            component_means.append(mean + gain * (observation - y0))
            component_covariances.append(covariance - np.outer(ch, ch) / variance)
        total_density = float(np.sum(component_mass))
        likelihood = float(detect) * total_density
        if likelihood <= _TINY or not math.isfinite(likelihood):
            return mean.copy(), covariance.copy(), 0.0
        component_weights = np.asarray(component_mass, dtype=float) / max(total_density, _TINY)
        posterior_mean = sum(weight * value_ for weight, value_ in zip(component_weights, component_means))
        posterior_second = sum(
            weight * (component_cov + np.outer(value_, value_))
            for weight, component_cov, value_ in zip(component_weights, component_covariances, component_means)
        )
        posterior_covariance = _symmetrize_psd(posterior_second - np.outer(posterior_mean, posterior_mean))
        return posterior_mean, posterior_covariance, likelihood

    # No target centroid was observed.  The miss event includes detector
    # failure and target values outside the aperture.  We compute its first
    # two moments by subtracting the gated target event from the prior event.
    gated_first = np.zeros_like(mean)
    gated_second = np.zeros_like(covariance)
    gated_probability = 0.0
    for weight, scale in zip(weights, scales):
        variance = latent_variance + float(scale) ** 2
        mass, conditional_y_mean, conditional_y_variance = _truncated_component_moments(
            y0, variance, -half_width, half_width
        )
        event_weight = float(detect) * float(weight) * mass
        if event_weight <= 0.0:
            continue
        theta_mean = mean + ch * (conditional_y_mean - y0) / variance
        theta_covariance = covariance + np.outer(ch, ch) * (conditional_y_variance - variance) / (variance * variance)
        gated_probability += event_weight
        gated_first += event_weight * theta_mean
        gated_second += event_weight * (theta_covariance + np.outer(theta_mean, theta_mean))
    likelihood = float(1.0 - gated_probability)
    if likelihood <= _TINY or not math.isfinite(likelihood):
        return mean.copy(), covariance.copy(), 0.0
    posterior_mean = (mean - gated_first) / likelihood
    posterior_second = (prior_second - gated_second) / likelihood
    posterior_covariance = _symmetrize_psd(posterior_second - np.outer(posterior_mean, posterior_mean))
    return posterior_mean, posterior_covariance, likelihood


def _clutter_density(value: float, half_width: float) -> float:
    return 1.0 / (2.0 * half_width) if abs(float(value)) <= half_width + 1.0e-12 else 0.0


def update_set(
    mean: np.ndarray,
    cov: np.ndarray,
    models: Iterable[tuple[np.ndarray, float]],
    values: Sequence[float],
    hw: float,
    *,
    noise_mixture: Sequence[Sequence[float]] = DEFAULT_NOISE_MIXTURE,
    false_rate: float = DEFAULT_FALSE_RATE,
    p_detect: float = DEFAULT_P_DETECT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], float]:
    """Update an unlabeled target set with at most one false centroid.

    The return contract matches ``dynamic_scan.update_set``:

    ``(posterior_mean, posterior_covariance, association_marginals,
    best_assignment, best_assignment_probability)``.

    ``best_assignment`` contains one observed-value index per target model or
    ``-1`` for a target miss.  A false centroid is the one observed value not
    assigned by that tuple; its index is intentionally not exposed so the
    public return shape remains compatible with the current runner.
    """

    half_width = float(hw)
    if not math.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("hw must be finite and positive")
    false = _probability(false_rate, "false_rate")
    detect = _probability(p_detect, "p_detect")
    _validate_noise_mixture(noise_mixture)
    current_mean = np.asarray(mean, dtype=float)
    current_cov = _symmetrize_psd(np.asarray(cov, dtype=float))
    if current_mean.ndim != 1 or current_cov.shape != (current_mean.size, current_mean.size):
        raise ValueError("mean/cov dimensions are inconsistent")
    observations = [float(value) for value in values]
    if observations and not np.isfinite(np.asarray(observations, dtype=float)).all():
        raise ValueError("observed centroids must be finite")
    parsed_models: list[tuple[np.ndarray, float]] = []
    for model in models:
        if len(model) != 2:
            raise ValueError("each model must be (row, base)")
        row, base = np.asarray(model[0], dtype=float), float(model[1])
        if row.shape != current_mean.shape or not np.isfinite(row).all() or not math.isfinite(base):
            raise ValueError("model rows and bases must match the belief and be finite")
        parsed_models.append((row, base))
    n_models = len(parsed_models)
    n_values = len(observations)
    if n_values > n_models + 1:
        raise ValueError("at most one false centroid is supported")

    hypotheses: list[tuple[tuple[int, ...], np.ndarray, np.ndarray, float]] = []
    # -1 means target miss; nonnegative values index the observed values.
    # Every observed value must be assigned to a target or to at most one
    # uniform clutter hypothesis.
    for assignment in product(range(-1, n_values), repeat=n_models):
        used = [index for index in assignment if index >= 0]
        if len(set(used)) != len(used):
            continue
        unassigned = [index for index in range(n_values) if index not in used]
        if len(unassigned) > 1:
            continue
        if unassigned:
            clutter_probability = float(false) * _clutter_density(observations[unassigned[0]], half_width)
        else:
            clutter_probability = float(1.0 - false)
        if clutter_probability <= 0.0:
            continue
        posterior_mean = current_mean.copy()
        posterior_covariance = current_cov.copy()
        probability = clutter_probability
        supported = True
        for (row, base), index in zip(parsed_models, assignment):
            observation = None if index < 0 else observations[index]
            posterior_mean, posterior_covariance, likelihood = _posterior_step(
                posterior_mean,
                posterior_covariance,
                row,
                base,
                observation,
                half_width,
                noise_mixture=noise_mixture,
                false_rate=false,
                p_detect=detect,
            )
            if likelihood <= 0.0 or not math.isfinite(likelihood):
                supported = False
                break
            probability *= likelihood
        if supported and probability > 0.0 and math.isfinite(probability):
            hypotheses.append((tuple(int(index) for index in assignment), posterior_mean, posterior_covariance, probability))

    evidence = float(sum(item[3] for item in hypotheses))
    if not hypotheses or evidence <= 0.0 or not math.isfinite(evidence):
        raise ValueError("observation outside all modeled target/clutter associations")
    normalized = np.asarray([item[3] / evidence for item in hypotheses], dtype=float)
    posterior_mean = sum(weight * item[1] for weight, item in zip(normalized, hypotheses))
    posterior_second = sum(
        weight * (item[2] + np.outer(item[1], item[1]))
        for weight, item in zip(normalized, hypotheses)
    )
    posterior_covariance = _symmetrize_psd(posterior_second - np.outer(posterior_mean, posterior_mean))
    marginal = np.zeros((n_models, n_values), dtype=float)
    for weight, (assignment, _, _, _) in zip(normalized, hypotheses):
        for model_index, observed_index in enumerate(assignment):
            if observed_index >= 0:
                marginal[model_index, observed_index] += float(weight)
    best_index = int(np.argmax(normalized))
    best_assignment, _, _, _ = hypotheses[best_index]
    return posterior_mean, posterior_covariance, marginal, best_assignment, float(normalized[best_index])


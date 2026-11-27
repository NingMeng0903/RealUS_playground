"""Public-outcome Fisher information and task-directed probe design.

This file is deliberately small and model-only.  It does not import a robot,
an anatomy model, or a policy runner.  A checked probe supplies a signed
centroid model ``Y ~ Normal(mu, scale**2)`` and a detection probability.  The
public observation is a finite categorical outcome:

``miss, y_bin_0, ..., y_bin_(K-1)``

or, with ``censoring="tail"``, ``miss, left_censored, bins, right_censored``.
Finite edges with ``censoring="miss"`` put detected samples outside the
reported interval into ``miss``.  An optional clutter component is an
*unconditional* detected mass ``p_clutter`` with its own Gaussian law; the
remaining mass is miss.  This convention is explicit so target/clutter mass
is never silently counted twice.

The local design coordinates are
``(centroid, log_scale, p_detect[, clutter_centroid, clutter_log_scale,
p_clutter])``.  ``parameter_jacobian`` maps task/nuisance parameters to those
coordinates, with shape ``(n_design, n_parameter)``.  The parameter Fisher is
then exactly ``J.T @ F_design @ J``.  Fisher information here is a local
curvature/forecast quantity; this module makes no claim that it identifies a
mixture mode or resolves association labels.

Only NumPy and SciPy are required.  The public functions are intentionally
usable without dataclasses from sibling experiments, so a planner can pass a
plain Fisher matrix or a :class:`FisherResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import log_ndtr, ndtr


__all__ = [
    "DESIGN_COORDINATES",
    "FisherResult",
    "GreedySelection",
    "categorical_fisher",
    "categorical_fisher_information",
    "categorical_mutual_information",
    "finite_state_mutual_information",
    "mutual_information_finite_state",
    "finite_difference_fisher",
    "finite_difference_probability_jacobian",
    "binned_signed_centroid_probabilities",
    "binned_signed_centroid_fisher",
    "binned_fisher",
    "signed_centroid_probabilities",
    "signed_centroid_fisher",
    "chain_fisher",
    "information_to_covariance",
    "schur_complement_information",
    "task_design_score",
    "task_score",
    "predicted_covariance_reduction",
    "greedy_common_candidates",
    "greedy_candidate_selection",
    "select_greedy_common_candidates",
]


_SQRT_2PI = math.sqrt(2.0 * math.pi)
_TINY = np.finfo(float).tiny
_EPS = np.finfo(float).eps
DESIGN_COORDINATES = (
    "centroid",
    "log_scale",
    "p_detect",
    "clutter_centroid",
    "clutter_log_scale",
    "p_clutter",
)


def _as_float_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}; got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _normal_pdf(z: np.ndarray | float) -> np.ndarray:
    values = np.asarray(z, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.exp(-0.5 * values * values) / _SQRT_2PI
    return np.where(np.isfinite(values), result, 0.0)


def _zphi(z: float) -> float:
    """Return z*phi(z), treating the two infinite tails as zero."""

    if not math.isfinite(float(z)):
        return 0.0
    return float(z * math.exp(-0.5 * float(z) * float(z)) / _SQRT_2PI)


def _logdiffexp(log_a: float, log_b: float) -> float:
    """Stable ``log(exp(log_a)-exp(log_b))`` for ``log_a >= log_b``."""

    if not math.isfinite(log_a):
        return -math.inf
    if not math.isfinite(log_b):
        return log_a
    if log_b >= log_a:
        if log_b - log_a <= 4.0 * _EPS:
            return -math.inf
        raise FloatingPointError("logdiffexp requires the first argument to dominate")
    gap = log_b - log_a
    return float(log_a + math.log1p(-math.exp(gap)))


def _normal_interval_mass(z_lo: float, z_hi: float) -> float:
    """Stable standard-normal mass on ``[z_lo, z_hi]``."""

    if z_lo == -math.inf and z_hi == math.inf:
        return 1.0
    if not z_lo < z_hi:
        return 0.0
    # Use the smaller tail representation.  Direct subtraction is accurate
    # around the centre; log-CDF/log-survival avoids cancellation in a tail.
    if z_hi <= 0.0:
        log_hi = float(log_ndtr(z_hi))
        log_lo = float(log_ndtr(z_lo)) if z_lo > -math.inf else -math.inf
        if log_lo == -math.inf:
            return float(math.exp(log_hi))
        value = _logdiffexp(log_hi, log_lo)
        return float(math.exp(value)) if math.isfinite(value) else 0.0
    if z_lo >= 0.0:
        # Phi(hi)-Phi(lo) = Phi(-lo)-Phi(-hi).
        log_lo = float(log_ndtr(-z_lo)) if z_lo < math.inf else -math.inf
        log_hi = float(log_ndtr(-z_hi)) if z_hi < math.inf else -math.inf
        if log_lo == -math.inf:
            return 0.0
        value = _logdiffexp(log_lo, log_hi)
        return float(math.exp(value)) if math.isfinite(value) else 0.0
    value = float(ndtr(z_hi) - ndtr(z_lo))
    return max(0.0, value)


def _normalize_probability_and_derivative(
    probabilities: np.ndarray,
    derivative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a categorical vector while applying the quotient rule."""

    p = np.asarray(probabilities, dtype=float)
    dp = np.asarray(derivative, dtype=float)
    if p.ndim != 1 or dp.ndim != 2 or dp.shape[0] != p.size:
        raise ValueError("probabilities/derivative have incompatible shapes")
    if not np.isfinite(p).all() or not np.isfinite(dp).all():
        raise ValueError("probabilities and derivatives must be finite")
    if (p < -1.0e-11).any():
        raise ValueError(f"negative categorical mass: {p.min()!r}")
    p = np.maximum(p, 0.0)
    total = float(p.sum())
    if not total > 0.0 or not math.isfinite(total):
        raise ValueError("categorical mass must have positive finite total")
    # The formulas below are normally already normalized.  Applying this even
    # for tiny roundoff makes both p and dp satisfy the categorical contract.
    dtotal = np.sum(dp, axis=0)
    normalized_dp = (dp * total - p[:, None] * dtotal[None, :]) / (total * total)
    return p / total, normalized_dp


def _validate_edges(edges: Sequence[float]) -> np.ndarray:
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("edges must contain at least two ordered values")
    if np.isnan(values).any() or np.any(np.diff(values) <= 0.0) or np.isnan(np.diff(values)).any():
        raise ValueError("edges must be strictly increasing and may only use +/-inf as non-finite values")
    # Infinite endpoints are useful for a complete partition.  An interior
    # infinity is not an ordered finite bin boundary and often indicates a
    # malformed caller array (it also makes adjacent differences NaN).
    if np.isinf(values[1:-1]).any():
        raise ValueError("only the first/last edge may be infinite")
    if np.isposinf(values[0]) or np.isneginf(values[-1]):
        raise ValueError("the first edge cannot be +inf and the last cannot be -inf")
    return values


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < -1.0e-12 or value > 1.0 + 1.0e-12:
        raise ValueError(f"{name} must lie in [0, 1]; got {value!r}")
    return float(np.clip(value, 0.0, 1.0))


def _validate_design_jacobian(jacobian: Any, rows: int, *, name: str) -> np.ndarray | None:
    if jacobian is None:
        return None
    values = np.asarray(jacobian, dtype=float)
    if values.ndim != 2 or values.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows}, n_parameters); got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    return values


def _component_bins(
    centroid: float,
    scale: float,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin mass and derivatives wrt ``(centroid, log_scale)``."""

    if not math.isfinite(float(centroid)) or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("centroid must be finite and scale must be positive")
    masses = []
    dm = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        z_lo = float((lo - centroid) / scale) if math.isfinite(float(lo)) else -math.inf
        z_hi = float((hi - centroid) / scale) if math.isfinite(float(hi)) else math.inf
        mass = _normal_interval_mass(z_lo, z_hi)
        phi_lo = float(_normal_pdf(z_lo))
        phi_hi = float(_normal_pdf(z_hi))
        d_centroid = (phi_lo - phi_hi) / scale
        d_log_scale = _zphi(z_lo) - _zphi(z_hi)
        masses.append(mass)
        dm.append((d_centroid, d_log_scale))
    return np.asarray(masses, dtype=float), np.asarray(dm, dtype=float)


def _component_tail_parts(
    centroid: float,
    scale: float,
    edges: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Return left tail, right tail and derivatives wrt centroid/log-scale."""

    lo = float(edges[0])
    hi = float(edges[-1])
    z_lo = (lo - centroid) / scale if math.isfinite(lo) else -math.inf
    z_hi = (hi - centroid) / scale if math.isfinite(hi) else math.inf
    left = float(ndtr(z_lo)) if math.isfinite(z_lo) else 0.0
    right = float(ndtr(-z_hi)) if math.isfinite(z_hi) else 0.0
    phi_lo = float(_normal_pdf(z_lo))
    phi_hi = float(_normal_pdf(z_hi))
    d_left = np.asarray((-phi_lo / scale, -_zphi(z_lo)), dtype=float)
    d_right = np.asarray((phi_hi / scale, _zphi(z_hi)), dtype=float)
    return left, right, d_left, d_right


def _component_observations(
    centroid: float,
    scale: float,
    edges: np.ndarray,
    *,
    censoring: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Component masses/derivatives wrt (centroid, log-scale)."""

    masses, dm_bins = _component_bins(centroid, scale, edges)
    mode = str(censoring).lower()
    if mode in {"explicit", "tail", "tails"}:
        left, right, d_left, d_right = _component_tail_parts(centroid, scale, edges)
        masses = np.concatenate(([left], masses, [right]))
        dm = np.vstack((d_left, dm_bins, d_right))
        labels = ("left_censored",) + tuple(f"y_bin_{i}" for i in range(edges.size - 1)) + ("right_censored",)
        return masses, dm, labels
    if mode not in {"miss", "missing", "censor_to_miss"}:
        raise ValueError("censoring must be 'miss' or 'tail'")
    return masses, dm_bins, tuple(f"y_bin_{i}" for i in range(edges.size - 1))


def binned_signed_centroid_probabilities(
    centroid: float,
    scale: float,
    p_detect: float,
    edges: Sequence[float],
    *,
    censoring: str = "miss",
    clutter_centroid: float | None = None,
    clutter_scale: float | None = None,
    p_clutter: float = 0.0,
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[str, ...]]:
    """Return normalized public outcome probabilities for one probe.

    ``p_detect`` and ``p_clutter`` are unconditional masses.  With
    ``censoring='miss'``, finite edge tails join the miss category.  With
    ``censoring='tail'``, tails receive explicit categories and the miss mass
    is ``1-p_detect-p_clutter``.  The optional clutter Gaussian uses the same
    bin edges and is independent of the target centroid law.
    """

    result = binned_signed_centroid_fisher(
        centroid,
        scale,
        p_detect,
        edges,
        censoring=censoring,
        clutter_centroid=clutter_centroid,
        clutter_scale=clutter_scale,
        p_clutter=p_clutter,
        parameter_jacobian=None,
        clutter_parameter_jacobian=None,
    )
    if return_labels:
        return result.probabilities, result.labels
    return result.probabilities


@dataclass(frozen=True)
class FisherResult:
    """Analytic categorical Fisher result and its public-outcome contract."""

    probabilities: np.ndarray
    probability_jacobian: np.ndarray
    fisher: np.ndarray
    labels: tuple[str, ...]
    design_names: tuple[str, ...]
    parameter_jacobian: np.ndarray | None = None
    parameter_fisher: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def derivative(self) -> np.ndarray:
        """Alias for ``probability_jacobian`` (outcome x design)."""

        return self.probability_jacobian

    @property
    def jacobian(self) -> np.ndarray:
        return self.probability_jacobian

    @property
    def parameter_probability_jacobian(self) -> np.ndarray | None:
        """Return ``dP/dparameter`` when a chain Jacobian was supplied."""

        if self.parameter_jacobian is None:
            return None
        return self.probability_jacobian @ self.parameter_jacobian

    @property
    def parameter_derivative(self) -> np.ndarray | None:
        return self.parameter_probability_jacobian

    @property
    def probabilities_sum(self) -> float:
        return float(np.sum(self.probabilities))

    def __getitem__(self, key: str) -> Any:
        if key in {"p", "probabilities"}:
            return self.probabilities
        if key in {"dp", "derivative", "probability_jacobian"}:
            return self.probability_jacobian
        if key in {"F", "fisher", "fisher_design"}:
            return self.fisher
        if key in {"F_parameter", "parameter_fisher"}:
            return self.parameter_fisher
        if key in {"dP_parameter", "parameter_probability_jacobian", "parameter_derivative"}:
            return self.parameter_probability_jacobian
        if key == "labels":
            return self.labels
        if key == "design_names":
            return self.design_names
        if key == "diagnostics":
            return self.diagnostics
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "probabilities": self.probabilities.tolist(),
            "probability_jacobian": self.probability_jacobian.tolist(),
            "fisher": self.fisher.tolist(),
            "labels": list(self.labels),
            "design_names": list(self.design_names),
            "parameter_jacobian": None if self.parameter_jacobian is None else self.parameter_jacobian.tolist(),
            "parameter_fisher": None if self.parameter_fisher is None else self.parameter_fisher.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


def binned_signed_centroid_fisher(
    centroid: float,
    scale: float,
    p_detect: float,
    edges: Sequence[float],
    *,
    censoring: str = "miss",
    clutter_centroid: float | None = None,
    clutter_scale: float | None = None,
    p_clutter: float = 0.0,
    parameter_jacobian: Any = None,
    clutter_parameter_jacobian: Any = None,
    design_jacobian: Any = None,
    jacobian: Any = None,
    return_labels: bool = True,
) -> FisherResult:
    """Compute analytic Fisher for a binned signed-centroid observation.

    The returned local derivative has shape ``(n_outcomes, n_design)`` and
    design order ``(centroid, log_scale, p_detect[, clutter_centroid,
    clutter_log_scale, p_clutter])``.  ``parameter_jacobian`` is the matrix
    ``d design / d parameter`` and therefore has shape
    ``(n_design, n_parameter)``.  ``design_jacobian`` is accepted as a clear
    alias for ``parameter_jacobian``.
    """

    supplied_jacobians = [value is not None for value in (parameter_jacobian, design_jacobian, jacobian)]
    if sum(supplied_jacobians) > 1:
        raise ValueError("supply only one of parameter_jacobian, design_jacobian, or jacobian")
    if design_jacobian is not None:
        parameter_jacobian = design_jacobian
    if jacobian is not None:
        parameter_jacobian = jacobian
    edges_array = _validate_edges(edges)
    p_target = _validate_probability(p_detect, name="p_detect")
    p_false = _validate_probability(p_clutter, name="p_clutter")
    if p_target + p_false > 1.0 + 1.0e-11:
        raise ValueError("p_detect + p_clutter must be <= 1")
    if (clutter_centroid is None) != (clutter_scale is None):
        raise ValueError("clutter_centroid and clutter_scale must be supplied together")
    use_clutter = p_false > 0.0 or clutter_centroid is not None or clutter_parameter_jacobian is not None
    if use_clutter:
        if clutter_centroid is None:
            clutter_centroid = 0.0
        if clutter_scale is None:
            clutter_scale = scale
        _ = _component_bins(float(clutter_centroid), float(clutter_scale), edges_array)

    target_m, target_dm, bin_labels = _component_observations(
        float(centroid), float(scale), edges_array, censoring=censoring)
    clutter_m = np.zeros_like(target_m)
    clutter_dm = np.zeros((target_m.size, 2), dtype=float)
    if use_clutter:
        clutter_m, clutter_dm, clutter_labels = _component_observations(
            float(clutter_centroid), float(clutter_scale), edges_array, censoring=censoring)
        if clutter_labels != bin_labels:
            raise AssertionError("target and clutter outcome contracts diverged")

    mode = str(censoring).lower()
    explicit_tails = mode in {"explicit", "tail", "tails"}
    # In the explicit-tail contract, target/clutter component arrays have a
    # miss slot only after we prepend it below.  In the miss contract, finite
    # tails become part of that same miss slot.
    if explicit_tails:
        target_observed = target_m
        clutter_observed = clutter_m
        target_mass = float(target_observed.sum())
        clutter_mass = float(clutter_observed.sum())
        target_dmass = target_dm.sum(axis=0)
        clutter_dmass = clutter_dm.sum(axis=0)
        # A complete Gaussian with +/-inf edges has mass one; finite edges
        # with explicit tails do too up to roundoff.
        probabilities = np.concatenate(([1.0 - p_target - p_false],
                                        p_target * target_observed + p_false * clutter_observed))
        derivative_rows = np.zeros((probabilities.size, 6), dtype=float)
        derivative_rows[0, 2] = -1.0
        derivative_rows[0, 5] = -1.0
        derivative_rows[1:, 0:2] = p_target * target_dm
        derivative_rows[1:, 2] = target_observed
        derivative_rows[1:, 3:5] = p_false * clutter_dm
        derivative_rows[1:, 5] = clutter_observed
        labels = ("miss",) + tuple(bin_labels)
        _ = target_mass, clutter_mass, target_dmass, clutter_dmass
    else:
        target_observed = target_m
        clutter_observed = clutter_m
        probabilities = np.concatenate(([1.0 - p_target * float(target_observed.sum())
                                          - p_false * float(clutter_observed.sum())],
                                        p_target * target_observed + p_false * clutter_observed))
        derivative_rows = np.zeros((probabilities.size, 6), dtype=float)
        derivative_rows[0, 0:2] = -p_target * target_dm.sum(axis=0)
        derivative_rows[0, 2] = -float(target_observed.sum())
        derivative_rows[0, 3:5] = -p_false * clutter_dm.sum(axis=0)
        derivative_rows[0, 5] = -float(clutter_observed.sum())
        derivative_rows[1:, 0:2] = p_target * target_dm
        derivative_rows[1:, 2] = target_observed
        derivative_rows[1:, 3:5] = p_false * clutter_dm
        derivative_rows[1:, 5] = clutter_observed
        labels = ("miss",) + tuple(bin_labels)

    # Without a clutter component, omit unused columns and names.  Retaining
    # three columns for p_clutter would create an artificial zero-information
    # nuisance coordinate and makes Jacobian shape mistakes harder to detect.
    if use_clutter:
        design_names = DESIGN_COORDINATES
        local_dp = derivative_rows
    else:
        design_names = DESIGN_COORDINATES[:3]
        local_dp = derivative_rows[:, :3]
    probabilities, local_dp = _normalize_probability_and_derivative(probabilities, local_dp)
    local_fisher = categorical_fisher(probabilities, local_dp, normalize=False)

    # A combined 6-row Jacobian is convenient for callers that already have
    # one design table.  For readability, callers may instead provide the
    # target 3-row Jacobian in ``parameter_jacobian`` and an independent
    # clutter 3-row Jacobian in ``clutter_parameter_jacobian``; combine them
    # here before applying the chain rule.
    parameter_jacobian_array: np.ndarray | None
    if clutter_parameter_jacobian is not None and not use_clutter:
        raise ValueError("clutter_parameter_jacobian requires a clutter component")
    if clutter_parameter_jacobian is not None and use_clutter:
        clutter_jacobian_array = _validate_design_jacobian(
            clutter_parameter_jacobian, 3, name="clutter_parameter_jacobian")
        if parameter_jacobian is None:
            target_jacobian_array = np.zeros_like(clutter_jacobian_array)
        else:
            raw_target = np.asarray(parameter_jacobian, dtype=float)
            if raw_target.ndim != 2 or raw_target.shape[0] not in (3, 6):
                raise ValueError("parameter_jacobian must have 3 rows (target) or 6 rows (target+clutter)")
            if raw_target.shape[0] == 6:
                raise ValueError("a 6-row parameter_jacobian cannot be combined with clutter_parameter_jacobian")
            target_jacobian_array = _validate_design_jacobian(
                raw_target, 3, name="parameter_jacobian")
            assert target_jacobian_array is not None
        if target_jacobian_array.shape[1] != clutter_jacobian_array.shape[1]:
            raise ValueError("target and clutter Jacobians must share parameter count")
        parameter_jacobian_array = np.vstack((target_jacobian_array, clutter_jacobian_array))
    else:
        parameter_jacobian_array = _validate_design_jacobian(
            parameter_jacobian,
            len(design_names),
            name="parameter_jacobian",
        )
    parameter_fisher = None
    parameter_dp = None
    if parameter_jacobian_array is not None:
        parameter_dp = local_dp @ parameter_jacobian_array
        parameter_fisher = categorical_fisher(probabilities, parameter_dp, normalize=False)
    diagnostics = {
        "model": "binned_signed_centroid",
        "censoring": "tail" if explicit_tails else "miss",
        "has_clutter": bool(use_clutter),
        "probability_sum": float(probabilities.sum()),
        "fisher_scope": "public categorical outcome",
        "mixture_mode_identification": "not_claimed",
        "jacobian_orientation": "d_design_d_parameter (design x parameter)",
    }
    return FisherResult(
        probabilities=probabilities,
        probability_jacobian=local_dp if parameter_dp is None else local_dp,
        fisher=local_fisher,
        labels=tuple(labels) if return_labels else tuple(),
        design_names=tuple(design_names),
        parameter_jacobian=parameter_jacobian_array,
        parameter_fisher=parameter_fisher,
        diagnostics=diagnostics,
    )


# Short aliases used by small planners and notebooks.
signed_centroid_probabilities = binned_signed_centroid_probabilities
signed_centroid_fisher = binned_signed_centroid_fisher
binned_fisher = binned_signed_centroid_fisher


def categorical_fisher(
    probabilities: Sequence[float],
    derivatives: np.ndarray,
    *,
    min_probability: float = 1.0e-300,
    normalize: bool = True,
) -> np.ndarray:
    """Return ``sum_k (d p_k)(d p_k)^T / p_k`` for a categorical law.

    ``derivatives`` is outcome-by-parameter.  Zero-probability outcomes are
    skipped; a valid probability model has zero derivative there.  If
    ``normalize`` is true, a raw mass vector and its raw derivative are
    normalized with the quotient rule, which is useful for finite-difference
    models that accumulate tiny roundoff.
    """

    p = np.asarray(probabilities, dtype=float)
    dp = np.asarray(derivatives, dtype=float)
    if p.ndim != 1 or dp.ndim != 2:
        raise ValueError("probabilities must be (K,), derivatives must be (K, P)")
    # Accept the common parameter-by-outcome convention when it is
    # unambiguous; the returned Fisher is identical after transposition.
    if dp.shape[0] != p.size and dp.shape[1] == p.size:
        dp = dp.T
    if dp.shape[0] != p.size:
        raise ValueError("probabilities must be (K,), derivatives must be (K, P)")
    if not np.isfinite(p).all() or not np.isfinite(dp).all():
        raise ValueError("probabilities and derivatives must be finite")
    if (p < -1.0e-11).any():
        raise ValueError("probabilities must be nonnegative")
    if normalize:
        p, dp = _normalize_probability_and_derivative(p, dp)
    else:
        if abs(float(p.sum()) - 1.0) > 5.0e-9:
            raise ValueError(f"categorical probabilities must sum to one; got {p.sum()!r}")
        p = np.maximum(p, 0.0)
    fisher = np.zeros((dp.shape[1], dp.shape[1]), dtype=float)
    for mass, gradient in zip(p, dp):
        if mass <= float(min_probability):
            continue
        fisher += np.outer(gradient, gradient) / float(mass)
    fisher = (fisher + fisher.T) * 0.5
    # The construction is PSD.  Symmetric rounding below zero is not useful
    # to a planner and can destabilize a subsequent pseudo-inverse.
    eigenvalues, eigenvectors = np.linalg.eigh(fisher)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _coerce_state_weights(weights: Sequence[float] | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=float)
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size != n or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("state_weights must be a finite nonnegative vector matching states")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("state_weights must have positive mass")
    return values / total


def categorical_mutual_information(
    conditional_probabilities: Sequence[Sequence[float]],
    state_weights: Sequence[float] | None = None,
    *,
    base: float = math.e,
) -> float:
    """Exact finite-state ``I(S;Y)`` in the requested log base.

    Rows are ``P(Y | S=s)``.  The function normalizes each row for benign
    roundoff and skips zero terms, so no ``0*log(0)`` NaNs are emitted.
    """

    table = np.asarray(conditional_probabilities, dtype=float)
    if table.ndim != 2 or table.shape[0] == 0 or table.shape[1] == 0:
        raise ValueError("conditional_probabilities must be a nonempty (n_states, n_outcomes) array")
    if not np.isfinite(table).all() or (table < -1.0e-11).any():
        raise ValueError("conditional probabilities must be finite and nonnegative")
    table = np.maximum(table, 0.0)
    totals = table.sum(axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("each conditional row must have positive mass")
    table = table / totals[:, None]
    weights = _coerce_state_weights(state_weights, table.shape[0])
    predictive = weights @ table
    if base <= 0.0 or math.isclose(base, 1.0):
        raise ValueError("base must be positive and different from one")
    info = 0.0
    log_base = math.log(float(base))
    for w, row in zip(weights, table):
        positive = (row > 0.0) & (predictive > 0.0)
        info += float(w) * float(np.sum(row[positive] * np.log(row[positive] / predictive[positive])))
    return float(max(0.0, info / log_base))


finite_state_mutual_information = categorical_mutual_information
mutual_information = categorical_mutual_information
mutual_information_finite_state = categorical_mutual_information
categorical_fisher_information = categorical_fisher


def finite_difference_probability_jacobian(
    probability_function: Callable[[np.ndarray], Sequence[float]],
    parameters: Sequence[float],
    *,
    step: float | Sequence[float] | None = None,
    relative_step: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Central finite-difference Jacobian for a categorical probability map."""

    theta = np.asarray(parameters, dtype=float)
    if theta.ndim != 1 or not np.isfinite(theta).all():
        raise ValueError("parameters must be a finite one-dimensional vector")
    if step is None:
        steps = relative_step * np.maximum(1.0, np.abs(theta))
    else:
        steps = np.asarray(step, dtype=float)
        if steps.ndim == 0:
            steps = np.full(theta.size, float(steps))
        if steps.shape != theta.shape:
            raise ValueError("step must be scalar or match parameters")
    if not np.isfinite(steps).all() or (steps <= 0.0).any():
        raise ValueError("finite-difference steps must be positive and finite")
    centre = np.asarray(probability_function(theta), dtype=float)
    if centre.ndim != 1:
        raise ValueError("probability_function must return a one-dimensional vector")
    jacobian = np.empty((centre.size, theta.size), dtype=float)
    for j, delta in enumerate(steps):
        plus = theta.copy(); plus[j] += delta
        minus = theta.copy(); minus[j] -= delta
        p_plus = np.asarray(probability_function(plus), dtype=float)
        p_minus = np.asarray(probability_function(minus), dtype=float)
        if p_plus.shape != centre.shape or p_minus.shape != centre.shape:
            raise ValueError("probability_function changed output shape")
        jacobian[:, j] = (p_plus - p_minus) / (2.0 * float(delta))
    centre, jacobian = _normalize_probability_and_derivative(centre, jacobian)
    return centre, jacobian


def finite_difference_fisher(
    probability_function: Callable[[np.ndarray], Sequence[float]],
    parameters: Sequence[float],
    *,
    step: float | Sequence[float] | None = None,
    relative_step: float = 1.0e-5,
    return_details: bool = False,
) -> np.ndarray | FisherResult:
    """Numerically estimate categorical Fisher using central differences."""

    probabilities, jacobian = finite_difference_probability_jacobian(
        probability_function, parameters, step=step, relative_step=relative_step)
    fisher = categorical_fisher(probabilities, jacobian, normalize=False)
    if not return_details:
        return fisher
    return FisherResult(
        probabilities=probabilities,
        probability_jacobian=jacobian,
        fisher=fisher,
        labels=tuple(f"outcome_{i}" for i in range(probabilities.size)),
        design_names=tuple(f"parameter_{i}" for i in range(jacobian.shape[1])),
        parameter_jacobian=np.eye(jacobian.shape[1]),
        parameter_fisher=fisher,
        diagnostics={"method": "central_finite_difference", "mixture_mode_identification": "not_claimed"},
    )


def chain_fisher(local_fisher: np.ndarray, parameter_jacobian: np.ndarray) -> np.ndarray:
    """Chain ``F_design`` through ``J=d design/d parameter``."""

    fisher = np.asarray(local_fisher, dtype=float)
    jacobian = np.asarray(parameter_jacobian, dtype=float)
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("local_fisher must be square")
    if jacobian.ndim != 2 or jacobian.shape[0] != fisher.shape[0]:
        raise ValueError("parameter_jacobian must have one row per design coordinate")
    if not np.isfinite(fisher).all() or not np.isfinite(jacobian).all():
        raise ValueError("Fisher and Jacobian must be finite")
    result = jacobian.T @ ((fisher + fisher.T) * 0.5) @ jacobian
    return (result + result.T) * 0.5


def information_to_covariance(
    prior_covariance: np.ndarray,
    fisher: np.ndarray,
    *,
    rcond: float = 1.0e-12,
    ridge: float = 0.0,
) -> np.ndarray:
    """Return ``pinv(pinv(C_prior)+F)`` with symmetric PSD output."""

    covariance = np.asarray(prior_covariance, dtype=float)
    information = np.asarray(fisher, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("prior_covariance must be square")
    if information.shape != covariance.shape:
        raise ValueError("fisher and prior_covariance must have the same shape")
    if not np.isfinite(covariance).all() or not np.isfinite(information).all():
        raise ValueError("covariance and Fisher must be finite")
    covariance = (covariance + covariance.T) * 0.5
    information = (information + information.T) * 0.5
    if ridge < 0.0 or not math.isfinite(float(ridge)):
        raise ValueError("ridge must be finite and nonnegative")
    precision = np.linalg.pinv(covariance, rcond=rcond)
    posterior = np.linalg.pinv(precision + information + float(ridge) * np.eye(covariance.shape[0]), rcond=rcond)
    posterior = (posterior + posterior.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(posterior)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def schur_complement_information(
    information: np.ndarray,
    task_indices: Sequence[int],
    nuisance_indices: Sequence[int],
    *,
    rcond: float = 1.0e-12,
) -> np.ndarray:
    """Marginal task information ``I_tt-I_tn pinv(I_nn) I_nt``."""

    matrix = np.asarray(information, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("information must be square")
    task = np.asarray(tuple(int(i) for i in task_indices), dtype=int)
    nuisance = np.asarray(tuple(int(i) for i in nuisance_indices), dtype=int)
    n = matrix.shape[0]
    if task.size == 0:
        raise ValueError("task_indices must be nonempty")
    if np.any(task < 0) or np.any(task >= n) or np.any(nuisance < 0) or np.any(nuisance >= n):
        raise ValueError("task/nuisance index out of range")
    itt = matrix[np.ix_(task, task)]
    if nuisance.size == 0:
        return (itt + itt.T) * 0.5
    itn = matrix[np.ix_(task, nuisance)]
    inn = matrix[np.ix_(nuisance, nuisance)]
    result = itt - itn @ np.linalg.pinv(inn, rcond=rcond) @ itn.T
    result = (result + result.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(result)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _risk_from_covariance(
    covariance: np.ndarray,
    *,
    decision_directions: np.ndarray | None,
    weights: np.ndarray | None,
) -> float:
    cov = np.asarray(covariance, dtype=float)
    if decision_directions is None:
        if weights is None:
            return float(np.trace(cov))
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.ndim == 1:
            if weight_array.size != cov.shape[0]:
                raise ValueError("full-space weights must match covariance dimension")
            return float(np.sum(weight_array * np.diag(cov)))
        if weight_array.shape != cov.shape:
            raise ValueError("full-space weight matrix must match covariance shape")
        return float(np.trace(weight_array @ cov))
    directions = np.asarray(decision_directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] != cov.shape[0]:
        raise ValueError("decision_directions must have shape (n_parameters, n_directions)")
    projected = directions.T @ cov @ directions
    if weights is None:
        return float(np.trace(projected))
    weight_array = np.asarray(weights, dtype=float)
    if weight_array.ndim == 1:
        if weight_array.size != directions.shape[1]:
            raise ValueError("direction weights must match number of directions")
        return float(np.sum(weight_array * np.diag(projected)))
    if weight_array.shape != projected.shape:
        raise ValueError("direction weight matrix has incompatible shape")
    return float(np.trace(weight_array @ projected))


def task_design_score(
    fisher: np.ndarray,
    prior_covariance: np.ndarray,
    *,
    decision_directions: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    nuisance_indices: Sequence[int] | None = None,
    task_indices: Sequence[int] | None = None,
    method: str = "directional_trace",
    ridge: float = 0.0,
    return_details: bool = False,
) -> float | dict[str, Any]:
    """Score a candidate by predicted posterior covariance reduction.

    The default is weighted trace reduction in supplied decision directions.
    ``method='schur'`` additionally reports the marginal task information
    Schur complement after adding the candidate Fisher.  Nuisance coordinates
    are never silently discarded: the posterior covariance is formed in the
    complete task+nuisance space before projection.
    """

    candidate_fisher = _coerce_fisher_matrix(fisher)
    prior = np.asarray(prior_covariance, dtype=float)
    if candidate_fisher.ndim != 2 or candidate_fisher.shape[0] != candidate_fisher.shape[1]:
        raise ValueError("fisher must be square")
    if prior.shape != candidate_fisher.shape:
        raise ValueError("fisher and prior_covariance must have the same shape")
    if not np.isfinite(candidate_fisher).all() or not np.isfinite(prior).all():
        raise ValueError("fisher and prior_covariance must be finite")
    prior = (prior + prior.T) * 0.5
    candidate_fisher = (candidate_fisher + candidate_fisher.T) * 0.5
    baseline = information_to_covariance(prior, np.zeros_like(candidate_fisher), ridge=ridge)
    posterior = information_to_covariance(prior, candidate_fisher, ridge=ridge)
    mode = str(method).lower()
    if mode not in {"directional_trace", "weighted_trace", "schur"}:
        raise ValueError("method must be directional_trace, weighted_trace, or schur")
    if mode == "weighted_trace" and decision_directions is not None:
        raise ValueError("weighted_trace uses full-space weights; pass no directions")
    if mode == "schur":
        if task_indices is None:
            if nuisance_indices is None:
                raise ValueError("schur scoring requires nuisance_indices or task_indices")
            nuisance = tuple(int(i) for i in nuisance_indices)
            task = tuple(i for i in range(prior.shape[0]) if i not in nuisance)
        else:
            task = tuple(int(i) for i in task_indices)
            nuisance = tuple(int(i) for i in (nuisance_indices or ()))
        total_information = np.linalg.pinv(prior, rcond=1.0e-12) + candidate_fisher
        prior_information = np.linalg.pinv(prior, rcond=1.0e-12)
        task_info_after = schur_complement_information(total_information, task, nuisance)
        task_info_before = schur_complement_information(prior_information, task, nuisance)
        task_cov_after = np.linalg.pinv(task_info_after, rcond=1.0e-12)
        task_cov_before = np.linalg.pinv(task_info_before, rcond=1.0e-12)
        task_cov_after = (task_cov_after + task_cov_after.T) * 0.5
        task_cov_before = (task_cov_before + task_cov_before.T) * 0.5
        directions = None if decision_directions is None else np.asarray(decision_directions, dtype=float)
        if directions is not None and directions.shape[0] != len(task):
            raise ValueError("schur decision_directions must be expressed in task coordinates")
        baseline_risk = _risk_from_covariance(task_cov_before, decision_directions=directions, weights=weights)
        posterior_risk = _risk_from_covariance(task_cov_after, decision_directions=directions, weights=weights)
        score = max(0.0, float(baseline_risk - posterior_risk))
        if not return_details:
            return score
        return {
            "score": score,
            "baseline_risk": baseline_risk,
            "posterior_risk": posterior_risk,
            "method": "schur",
            "task_indices": list(task),
            "nuisance_indices": list(nuisance),
            "task_information_before": task_info_before,
            "task_information_after": task_info_after,
            "covariance_before": task_cov_before,
            "covariance_after": task_cov_after,
        }
    baseline_risk = _risk_from_covariance(
        baseline,
        decision_directions=decision_directions,
        weights=weights,
    )
    posterior_risk = _risk_from_covariance(
        posterior,
        decision_directions=decision_directions,
        weights=weights,
    )
    score = max(0.0, float(baseline_risk - posterior_risk))
    if not return_details:
        return score
    return {
        "score": score,
        "baseline_risk": baseline_risk,
        "posterior_risk": posterior_risk,
        "method": mode,
        "covariance_before": baseline,
        "covariance_after": posterior,
        "nuisance_indices": None if nuisance_indices is None else list(nuisance_indices),
    }


predicted_covariance_reduction = task_design_score
task_score = task_design_score


def _coerce_fisher_matrix(value: Any) -> np.ndarray:
    if isinstance(value, FisherResult):
        if value.parameter_fisher is not None:
            return np.asarray(value.parameter_fisher, dtype=float)
        return np.asarray(value.fisher, dtype=float)
    if hasattr(value, "parameter_fisher") and getattr(value, "parameter_fisher") is not None:
        return np.asarray(getattr(value, "parameter_fisher"), dtype=float)
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class GreedySelection:
    """Auditable output from :func:`greedy_common_candidates`."""

    selected_indices: tuple[int, ...]
    selected_ids: tuple[Any, ...]
    marginal_scores: tuple[float, ...]
    cumulative_scores: tuple[float, ...]
    combined_fisher: np.ndarray
    safe_mask: np.ndarray
    candidate_scores: tuple[tuple[float, ...], ...]

    @property
    def selected(self) -> tuple[Any, ...]:
        return self.selected_ids

    def __getitem__(self, key: str) -> Any:
        if key in {"selected", "selected_ids"}:
            return self.selected_ids
        if key == "selected_indices":
            return self.selected_indices
        if key == "marginal_scores":
            return self.marginal_scores
        if key == "cumulative_scores":
            return self.cumulative_scores
        if key in {"fisher", "combined_fisher"}:
            return self.combined_fisher
        if key == "safe_mask":
            return self.safe_mask
        if key == "candidate_scores":
            return self.candidate_scores
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_indices": list(self.selected_indices),
            "selected_ids": list(self.selected_ids),
            "marginal_scores": list(self.marginal_scores),
            "cumulative_scores": list(self.cumulative_scores),
            "combined_fisher": self.combined_fisher.tolist(),
            "safe_mask": self.safe_mask.tolist(),
            "candidate_scores": [list(row) for row in self.candidate_scores],
        }


def _coerce_candidate_bank(
    candidates: Any,
    candidate_ids: Sequence[Any] | None,
) -> tuple[list[Any], list[np.ndarray]]:
    if isinstance(candidates, Mapping):
        ids = list(candidates.keys())
        matrices = [_coerce_fisher_matrix(value) for value in candidates.values()]
    else:
        values = list(candidates) if not isinstance(candidates, np.ndarray) else None
        if isinstance(candidates, np.ndarray):
            array = np.asarray(candidates, dtype=float)
            if array.ndim != 3:
                raise ValueError("candidate array must have shape (n_candidates, n, n)")
            matrices = [array[i] for i in range(array.shape[0])]
            ids = list(range(array.shape[0]))
        else:
            parsed: list[tuple[Any, np.ndarray]] = []
            for i, item in enumerate(values or []):
                if isinstance(item, Mapping) and "fisher" in item:
                    ident = item.get("id", i)
                    parsed.append((ident, _coerce_fisher_matrix(item["fisher"])))
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    # The second item may be a FisherResult, which is not
                    # coercible by np.asarray but is still a valid candidate.
                    second_is_matrix = isinstance(item[1], FisherResult) or hasattr(item[1], "parameter_fisher")
                    if not second_is_matrix:
                        try:
                            second_is_matrix = np.asarray(item[1]).ndim == 2
                        except (TypeError, ValueError):
                            second_is_matrix = False
                    if second_is_matrix:
                        parsed.append((item[0], _coerce_fisher_matrix(item[1])))
                    else:
                        parsed.append((i, _coerce_fisher_matrix(item)))
                else:
                    parsed.append((i, _coerce_fisher_matrix(item)))
            ids = [item[0] for item in parsed]
            matrices = [item[1] for item in parsed]
    if candidate_ids is not None:
        if len(candidate_ids) != len(matrices):
            raise ValueError("candidate_ids must match candidate count")
        ids = list(candidate_ids)
    if not matrices:
        raise ValueError("candidate bank must be nonempty")
    return ids, matrices


def greedy_common_candidates(
    candidates: Any,
    prior_covariance: np.ndarray,
    budget: int,
    *,
    safe_mask: Sequence[bool] | None = None,
    candidate_ids: Sequence[Any] | None = None,
    base_fisher: np.ndarray | None = None,
    decision_directions: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    nuisance_indices: Sequence[int] | None = None,
    task_indices: Sequence[int] | None = None,
    method: str = "directional_trace",
    ridge: float = 0.0,
    min_marginal_score: float = 0.0,
) -> GreedySelection:
    """Greedily choose a common safe candidate bank by marginal score.

    ``safe_mask`` is applied before scoring and is part of the returned audit
    record.  Non-finite or shape-incompatible Fisher matrices are also marked
    unsafe; they are never silently repaired or selected.  Candidate order is
    the supplied order, so equal-score ties are deterministic.
    """

    ids, matrices = _coerce_candidate_bank(candidates, candidate_ids)
    prior = np.asarray(prior_covariance, dtype=float)
    if prior.ndim != 2 or prior.shape[0] != prior.shape[1]:
        raise ValueError("prior_covariance must be square")
    n = prior.shape[0]
    if int(budget) < 0:
        raise ValueError("budget must be nonnegative")
    if safe_mask is None:
        mask = np.ones(len(matrices), dtype=bool)
    else:
        mask = np.asarray(safe_mask, dtype=bool)
        if mask.ndim != 1 or mask.size != len(matrices):
            raise ValueError("safe_mask must match candidate count")
        mask = mask.copy()
    valid_matrices: list[np.ndarray] = []
    for i, matrix in enumerate(matrices):
        value = np.asarray(matrix, dtype=float)
        if value.shape != (n, n) or not np.isfinite(value).all():
            mask[i] = False
            valid_matrices.append(np.zeros((n, n), dtype=float))
            continue
        value = (value + value.T) * 0.5
        # Candidate Fisher should be PSD.  Reject materially indefinite rows;
        # tiny negative eigenvalues are just symmetrization roundoff.
        minimum = float(np.min(np.linalg.eigvalsh(value)))
        if minimum < -1.0e-9 * max(1.0, float(np.max(np.abs(value)))):
            mask[i] = False
        valid_matrices.append(value)
    effective_safe_mask = mask.copy()
    if base_fisher is None:
        combined = np.zeros((n, n), dtype=float)
    else:
        combined = np.asarray(base_fisher, dtype=float)
        if combined.shape != (n, n) or not np.isfinite(combined).all():
            raise ValueError("base_fisher must be finite and match prior dimension")
        combined = (combined + combined.T) * 0.5
    selected: list[int] = []
    marginal: list[float] = []
    cumulative: list[float] = []
    audit_rows: list[tuple[float, ...]] = []
    current_score = float(task_design_score(
        combined,
        prior,
        decision_directions=decision_directions,
        weights=weights,
        nuisance_indices=nuisance_indices,
        task_indices=task_indices,
        method=method,
        ridge=ridge,
    ))
    for _ in range(int(budget)):
        available = [i for i in range(len(valid_matrices)) if mask[i] and i not in selected]
        if not available:
            break
        scores = []
        for i in available:
            trial = combined + valid_matrices[i]
            absolute = float(task_design_score(
                trial,
                prior,
                decision_directions=decision_directions,
                weights=weights,
                nuisance_indices=nuisance_indices,
                task_indices=task_indices,
                method=method,
                ridge=ridge,
            ))
            scores.append((absolute - current_score, i, absolute))
        audit_rows.append(tuple(float(scores[j][0]) for j in range(len(scores))))
        best_gain, best_index, best_absolute = max(scores, key=lambda row: (row[0], -row[1]))
        if best_gain <= float(min_marginal_score):
            break
        selected.append(best_index)
        combined = combined + valid_matrices[best_index]
        current_score = best_absolute
        marginal.append(float(best_gain))
        cumulative.append(float(current_score))
    return GreedySelection(
        selected_indices=tuple(selected),
        selected_ids=tuple(ids[i] for i in selected),
        marginal_scores=tuple(marginal),
        cumulative_scores=tuple(cumulative),
        combined_fisher=combined,
        safe_mask=effective_safe_mask,
        candidate_scores=tuple(audit_rows),
    )


greedy_candidate_selection = greedy_common_candidates
select_greedy_common_candidates = greedy_common_candidates


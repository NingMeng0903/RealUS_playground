"""Finite Bayes decision core for the dynamic-leg paired comparison.

This module deliberately contains no geometry, motion solver, or simulator.  A
geometry worker supplies a *common* finite bank of checked probe/scan programs
and their public outcome kernels.  The two policy arms then use exactly that
bank, prior, objective, and transition-cost table:

``A``
    The first program is chosen with a fixed second program.  The second
    program is selected by ``max_b E_o`` (the maximum is outside the first
    observation expectation).  A fixed continuation is admitted only if it is
    feasible after every first-stage outcome with positive prior predictive
    probability.

``B``
    The first program is chosen with a feedback continuation.  For each
    observed first-stage outcome, ``max_b`` is taken before averaging over
    outcomes (``E_o max_b``).  The returned ``continuation_by_outcome`` is the
    complete public-feedback policy map.

The algebra is finite and exact up to the supplied particle and outcome
quadrature approximation.  For latent particles ``z`` with prior ``w`` and a
first-stage outcome kernel ``K_a(z,o)``,

    q_a(o) = sum_z w(z) K_a(z,o),
    w_a(z | o) = w(z) K_a(z,o) / q_a(o).

The continuation value is evaluated with this posterior.  A and B differ only
in the order of ``max`` and ``sum_o q(o)``; costs and objective arrays are
shared.  In particular, this file does not silently give B a different cost,
an extra candidate, or a hidden truth label.

The public API is NumPy-only.  ``math.erf`` is used through a small vectorized
normal-CDF helper so SciPy is not required by the decision core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


__all__ = [
    "BayesBelief",
    "Belief",
    "CandidateLibrary",
    "CheckedCandidateSet",
    "DecisionResult",
    "FiniteBelief",
    "OutcomeKernel",
    "bayes_update",
    "centroid_outcome_kernel",
    "compare_methods",
    "make_centroid_kernel",
    "normal_bin_probabilities",
    "plan",
    "plan_scan_pair",
    "plan_two_stage",
    "posterior_table",
    "posterior_update",
    "predictive_distribution",
    "select_action",
    "update_belief",
]


_TINY = np.finfo(float).tiny
_DEFAULT_TOL = 1.0e-12


def _normal_cdf(x: np.ndarray | float) -> np.ndarray:
    """Standard-normal CDF with no SciPy dependency."""

    values = np.asarray(x, dtype=float)
    # np.erf is not part of the NumPy public API on all supported versions.
    flat = np.fromiter(
        (0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in values.ravel()),
        dtype=float,
        count=values.size,
    )
    return flat.reshape(values.shape)


def _as_probability_vector(values: Any, n: int | None = None, *, name: str = "weights") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if n is not None and array.size != int(n):
        raise ValueError(f"{name} has length {array.size}, expected {int(n)}")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(array.sum())
    if not total > 0.0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def normalize_weights(weights: Any) -> np.ndarray:
    """Return nonnegative one-dimensional weights normalized to unit mass."""

    return _as_probability_vector(weights)


def _validate_probability_kernel(kernel: Any, *, name: str = "kernel") -> np.ndarray:
    array = np.asarray(kernel, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_particles, n_outcomes)")
    if not np.isfinite(array).all() or (array < -1.0e-12).any():
        raise ValueError(f"{name} must be finite and nonnegative")
    array = np.maximum(array, 0.0)
    row_sums = array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=2.0e-10, atol=2.0e-12):
        raise ValueError(f"{name} rows must sum to one; got range {row_sums.min()}..{row_sums.max()}")
    return array


@dataclass(frozen=True)
class BayesBelief:
    """A shared finite-particle public belief.

    ``particles`` may be a numeric latent table (``n_particles x n_features``)
    or an object/index table.  The planner only uses its row count; keeping the
    table here lets a simulator attach the same particle identities to all
    paired arms without changing the decision algebra.
    """

    particles: np.ndarray
    weights: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        particles = np.asarray(self.particles)
        weights = _as_probability_vector(self.weights, len(particles), name="belief weights")
        if particles.ndim == 0:
            particles = particles.reshape(1)
        if len(particles) != weights.size:
            raise ValueError("particles and weights must have the same first dimension")
        object.__setattr__(self, "particles", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_particles(self) -> int:
        return int(self.weights.size)

    def predictive(self, kernel: np.ndarray) -> np.ndarray:
        """Return ``P(outcome)`` for a ``(particle, outcome)`` kernel."""

        return predictive_distribution(self.weights, kernel)

    def update(self, likelihood: np.ndarray, *, min_evidence: float = _TINY) -> tuple["BayesBelief", float]:
        """Condition on one observed outcome likelihood.

        A zero-evidence event raises rather than inventing a posterior.  A
        planner can skip zero-predictive quadrature bins before calling this
        method.
        """

        posterior, evidence = bayes_update(self.weights, likelihood, min_evidence=min_evidence)
        return BayesBelief(self.particles, posterior, self.metadata), evidence


# Names used by small runners in sibling studies.
Belief = BayesBelief
FiniteBelief = BayesBelief


def predictive_distribution(weights: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute a predictive outcome distribution from a finite kernel."""

    prior = _as_probability_vector(weights, name="weights")
    table = _validate_probability_kernel(kernel)
    if table.shape[0] != prior.size:
        raise ValueError("kernel particle count does not match weights")
    result = prior @ table
    # Keep tiny round-off from violating the probability contract.
    result = np.maximum(result, 0.0)
    return result / max(float(result.sum()), _TINY)


def bayes_update(
    weights: np.ndarray,
    likelihood: np.ndarray,
    *,
    min_evidence: float = _TINY,
) -> tuple[np.ndarray, float]:
    """Condition finite prior weights on an outcome likelihood.

    Returns ``(posterior, evidence)`` where ``evidence = sum(w*l)``.  The
    likelihood need not itself be a complete outcome row; it is the
    particle-wise likelihood of the particular observed outcome.
    """

    prior = _as_probability_vector(weights, name="weights")
    like = np.asarray(likelihood, dtype=float)
    if like.ndim != 1 or like.size != prior.size:
        raise ValueError("likelihood must be one-dimensional and match weights")
    if not np.isfinite(like).all() or (like < -1.0e-12).any():
        raise ValueError("likelihood must be finite and nonnegative")
    like = np.maximum(like, 0.0)
    evidence = float(prior @ like)
    if evidence <= float(min_evidence) or not math.isfinite(evidence):
        raise ValueError(f"observed event has zero evidence ({evidence!r})")
    posterior = prior * like / evidence
    posterior /= max(float(posterior.sum()), _TINY)
    return posterior, evidence


def posterior_update(
    belief_or_weights: BayesBelief | np.ndarray,
    likelihood: np.ndarray,
    *,
    particles: np.ndarray | None = None,
    min_evidence: float = _TINY,
) -> tuple[BayesBelief | np.ndarray, float]:
    """Compatibility wrapper for a one-event Bayes update."""

    if isinstance(belief_or_weights, BayesBelief):
        return belief_or_weights.update(likelihood, min_evidence=min_evidence)
    posterior, evidence = bayes_update(belief_or_weights, likelihood, min_evidence=min_evidence)
    return posterior, evidence


def update_belief(
    belief_or_weights: BayesBelief | np.ndarray,
    likelihood: np.ndarray,
    **kwargs: Any,
) -> tuple[BayesBelief | np.ndarray, float]:
    return posterior_update(belief_or_weights, likelihood, **kwargs)


def posterior_table(
    weights: np.ndarray,
    kernel: np.ndarray,
    *,
    zero_evidence_prior: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictive, posterior_by_outcome)``.

    ``posterior_by_outcome`` has shape ``(outcome, particle)``.  For a
    zero-predictive outcome, the prior is returned by default so callers can
    safely inspect a complete finite table; those branches still carry zero
    probability and are excluded by the planner's support mask.
    """

    prior = _as_probability_vector(weights, name="weights")
    table = _validate_probability_kernel(kernel)
    if table.shape[0] != prior.size:
        raise ValueError("kernel particle count does not match weights")
    predictive = prior @ table
    unnormalized = table.T * prior[None, :]
    posterior = np.empty_like(unnormalized)
    positive = predictive > _TINY
    posterior[positive] = unnormalized[positive] / predictive[positive, None]
    if np.any(~positive):
        if zero_evidence_prior:
            posterior[~positive] = prior
        else:
            posterior[~positive] = 0.0
    return predictive, posterior


def normal_bin_probabilities(
    mean: np.ndarray | float,
    scale: np.ndarray | float,
    *,
    edges: Sequence[float] | None = None,
    centers: Sequence[float] | None = None,
    quadrature_weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Approximate ``P(y-bin | mean, scale)`` for centroid observations.

    Give ``edges`` for exact normal mass per bin (``-inf``/``inf`` are
    accepted), or give ``centers`` and positive quadrature weights for a
    midpoint/quadrature table.  The returned last dimension is the number of
    bins.  Mass outside finite ``edges`` or omitted quadrature support is left
    out intentionally; :func:`centroid_outcome_kernel` places that mass in
    ``miss`` instead of silently renormalizing it.
    """

    mu = np.asarray(mean, dtype=float)
    sd = np.asarray(scale, dtype=float)
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or (sd <= 0.0).any():
        raise ValueError("mean must be finite and scale must be finite and positive")
    if (edges is None) == (centers is None):
        raise ValueError("provide exactly one of edges or centers")
    if edges is not None:
        bounds = np.asarray(edges, dtype=float)
        if bounds.ndim != 1 or bounds.size < 2 or np.isnan(bounds).any() or np.any(np.diff(bounds) <= 0.0):
            raise ValueError("edges must be strictly increasing and contain at least two values")
        hi = (bounds[1:] - mu[..., None]) / sd[..., None]
        lo = (bounds[:-1] - mu[..., None]) / sd[..., None]
        result = _normal_cdf(hi) - _normal_cdf(lo)
        return np.maximum(result, 0.0)
    values = np.asarray(centers, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("centers must be a nonempty finite one-dimensional sequence")
    if quadrature_weights is None:
        raise ValueError("quadrature_weights are required when centers are supplied")
    quad = np.asarray(quadrature_weights, dtype=float)
    if quad.shape != values.shape or not np.isfinite(quad).all() or (quad <= 0.0).any():
        raise ValueError("quadrature_weights must be finite and positive and match centers")
    z = (values - mu[..., None]) / sd[..., None]
    density = np.exp(-0.5 * z * z) / (math.sqrt(2.0 * math.pi) * sd[..., None])
    return np.maximum(density * quad, 0.0)


def centroid_outcome_kernel(
    centroid: np.ndarray | float,
    scale: np.ndarray | float,
    detect_probability: np.ndarray | float,
    *,
    y_edges: Sequence[float] | None = None,
    y_centers: Sequence[float] | None = None,
    quadrature_weights: Sequence[float] | None = None,
    valid_mask: np.ndarray | Sequence[bool] | None = None,
    labels: Sequence[str] | None = None,
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[str, ...]]:
    """Build ``P(miss or valid-centroid bin | particle, action)``.

    ``centroid``, ``scale`` and ``detect_probability`` broadcast to
    ``(n_actions, n_particles)``.  A one-dimensional input is interpreted as
    one action over all particles; a scalar is one action/one particle.  The
    first outcome is always ``miss``.  With no y bins, the second outcome is a
    generic ``detected`` bucket.  With bins, normal mass in each bin is
    multiplied by the detection probability; mass outside the supplied
    quadrature support remains a miss.  ``valid_mask`` gates detection before
    outcomes are formed, so checked-invalid poses cannot create a valid hit.
    """

    means = np.asarray(centroid, dtype=float)
    scales = np.asarray(scale, dtype=float)
    detect = np.asarray(detect_probability, dtype=float)
    try:
        means, scales, detect = np.broadcast_arrays(means, scales, detect)
    except ValueError as exc:
        raise ValueError("centroid, scale, and detect_probability are not broadcastable") from exc
    if means.ndim == 0:
        means = means.reshape(1, 1)
        scales = scales.reshape(1, 1)
        detect = detect.reshape(1, 1)
    elif means.ndim == 1:
        means = means.reshape(1, -1)
        scales = scales.reshape(1, -1)
        detect = detect.reshape(1, -1)
    elif means.ndim != 2:
        raise ValueError("centroid inputs must be scalar, (particles,), or (actions, particles)")
    if not np.isfinite(means).all() or not np.isfinite(scales).all() or not np.isfinite(detect).all():
        raise ValueError("centroid kernel inputs must be finite")
    if (scales <= 0.0).any() or (detect < 0.0).any() or (detect > 1.0).any():
        raise ValueError("scale must be positive and detect_probability must lie in [0, 1]")
    n_actions, n_particles = means.shape
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool)
        try:
            valid = np.broadcast_to(valid, (n_actions, n_particles))
        except ValueError as exc:
            raise ValueError("valid_mask is not broadcastable to (actions, particles)") from exc
        detect = np.where(valid, detect, 0.0)
    if y_edges is None and y_centers is None:
        bins = np.ones((n_actions, n_particles, 1), dtype=float)
        default_labels = ("miss", "detected")
    else:
        bins = normal_bin_probabilities(
            means,
            scales,
            edges=y_edges,
            centers=y_centers,
            quadrature_weights=quadrature_weights,
        )
        n_bins = int(bins.shape[-1])
        default_labels = ("miss",) + tuple(f"y_bin_{i}" for i in range(n_bins))
    detected = detect[..., None] * bins
    miss = np.maximum(0.0, 1.0 - detected.sum(axis=-1))
    kernel = np.concatenate([miss[..., None], detected], axis=-1)
    # The residual construction is already normalized; this correction only
    # removes a possible last-bit sum error from very wide tails.
    kernel /= np.maximum(kernel.sum(axis=-1, keepdims=True), _TINY)
    if labels is None:
        outcome_labels = default_labels
    else:
        outcome_labels = tuple(str(label) for label in labels)
        if len(outcome_labels) != kernel.shape[-1]:
            raise ValueError("labels must match the miss-plus-bin outcome count")
        if outcome_labels[0] != "miss":
            raise ValueError("the first outcome label must be 'miss'")
    if return_labels:
        return kernel, outcome_labels
    return kernel


make_centroid_kernel = centroid_outcome_kernel


@dataclass(frozen=True)
class OutcomeKernel:
    """Named wrapper around a common ``(action, particle, outcome)`` table."""

    probabilities: np.ndarray
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        table = np.asarray(self.probabilities, dtype=float)
        if table.ndim != 3:
            raise ValueError("OutcomeKernel probabilities must have shape (actions, particles, outcomes)")
        for index in range(table.shape[0]):
            _validate_probability_kernel(table[index], name="OutcomeKernel probabilities")
        labels = tuple(str(label) for label in self.labels)
        if labels and len(labels) != table.shape[-1]:
            raise ValueError("OutcomeKernel labels must match the outcome dimension")
        object.__setattr__(self, "probabilities", table)
        object.__setattr__(self, "labels", labels or tuple(f"outcome_{i}" for i in range(table.shape[-1])))


@dataclass(frozen=True)
class CheckedCandidateSet:
    """A finite candidate bank shared by both policy arms.

    Parameters
    ----------
    action_ids:
        Canonical program identifiers.  Their order is the deterministic tie
        break shared by A and B.
    outcome_probabilities:
        ``(n_actions, n_particles, n_outcomes)`` first-stage public kernels.
        Every row must sum to one, and outcome zero is conventionally ``miss``.
    initial_feasible:
        Checked feasibility at the current state, one bool per action.
    first_cost:
        Cost paid for executing the first program.  It is common to A and B.
    after_action_feasible:
        ``(first_action, first_outcome, second_action)`` checked mask.  A
        fixed second action must be true for all positive-probability outcomes;
        B can choose a different true action per outcome.
    transition_cost:
        Same shape as ``after_action_feasible``.  This is where the geometry
        worker places outward change, recovery, dwell, and any other declared
        post-outcome motion/transition costs.  It is never altered by policy
        arm.  A 2-D ``(first, second)`` table is broadcast over outcomes.
    continuation_probabilities:
        Optional branch-conditioned kernels with shape
        ``(first_action, first_outcome, second_action, particle, second_outcome)``.
        If omitted, the second action's first-stage kernel is reused.  The
        branch-conditioned form lets a simulator represent a checked move to
        a new window without putting physics in this module.
    """

    action_ids: tuple[str, ...] | Sequence[str]
    outcome_probabilities: np.ndarray
    initial_feasible: np.ndarray | Sequence[bool] | None = None
    first_cost: np.ndarray | Sequence[float] | None = None
    after_action_feasible: np.ndarray | None = None
    transition_cost: np.ndarray | None = None
    continuation_probabilities: np.ndarray | None = None
    outcome_labels: tuple[str, ...] | Sequence[str] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(str(value) for value in self.action_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("action_ids must be nonempty and unique")
        kernels = np.asarray(self.outcome_probabilities, dtype=float)
        if kernels.ndim != 3:
            raise ValueError("outcome_probabilities must have shape (actions, particles, outcomes)")
        n_actions, n_particles, n_outcomes = kernels.shape
        if n_actions != len(ids):
            raise ValueError("action_ids count does not match outcome_probabilities")
        for action in range(n_actions):
            _validate_probability_kernel(kernels[action], name="outcome_probabilities")
        if self.initial_feasible is None:
            initial = np.ones(n_actions, dtype=bool)
        else:
            initial = np.asarray(self.initial_feasible, dtype=bool)
            if initial.shape != (n_actions,):
                raise ValueError("initial_feasible must have shape (actions,)")
        if self.first_cost is None:
            first_cost = np.zeros(n_actions, dtype=float)
        else:
            first_cost = np.asarray(self.first_cost, dtype=float)
            if first_cost.shape != (n_actions,) or not np.isfinite(first_cost).all() or (first_cost < 0.0).any():
                raise ValueError("first_cost must be finite, nonnegative, and shaped (actions,)")
        if self.after_action_feasible is None:
            after_feasible = np.broadcast_to(initial[None, None, :], (n_actions, n_outcomes, n_actions)).copy()
        else:
            raw = np.asarray(self.after_action_feasible, dtype=bool)
            if raw.shape == (n_actions, n_actions):
                raw = np.broadcast_to(raw[:, None, :], (n_actions, n_outcomes, n_actions))
            if raw.shape != (n_actions, n_outcomes, n_actions):
                raise ValueError("after_action_feasible must have shape (first, outcome, second)")
            after_feasible = raw.copy()
        if self.transition_cost is None:
            costs = np.zeros((n_actions, n_outcomes, n_actions), dtype=float)
        else:
            raw_cost = np.asarray(self.transition_cost, dtype=float)
            if raw_cost.shape == (n_actions, n_actions):
                raw_cost = np.broadcast_to(raw_cost[:, None, :], (n_actions, n_outcomes, n_actions))
            if raw_cost.shape != (n_actions, n_outcomes, n_actions):
                raise ValueError("transition_cost must have shape (first, outcome, second)")
            if not np.isfinite(raw_cost).all() or (raw_cost < 0.0).any():
                raise ValueError("transition_cost must be finite and nonnegative")
            costs = raw_cost.copy()
        # An unchecked branch is never accidentally admitted because both the
        # mask and the cost are retained; infeasible cost entries are still
        # finite for easy JSON serialization and diagnostics.
        labels = tuple("miss" if i == 0 else f"outcome_{i}" for i in range(n_outcomes))
        if self.outcome_labels is not None:
            labels = tuple(str(value) for value in self.outcome_labels)
            if len(labels) != n_outcomes:
                raise ValueError("outcome_labels must match the outcome dimension")
        continuation = None
        if self.continuation_probabilities is not None:
            continuation = np.asarray(self.continuation_probabilities, dtype=float)
            expected = (n_actions, n_outcomes, n_actions, n_particles, n_outcomes)
            if continuation.shape != expected:
                raise ValueError(f"continuation_probabilities must have shape {expected}")
            for first in range(n_actions):
                for outcome in range(n_outcomes):
                    for second in range(n_actions):
                        _validate_probability_kernel(
                            continuation[first, outcome, second],
                            name="continuation_probabilities",
                        )
        object.__setattr__(self, "action_ids", ids)
        object.__setattr__(self, "outcome_probabilities", kernels)
        object.__setattr__(self, "initial_feasible", initial)
        object.__setattr__(self, "first_cost", first_cost)
        object.__setattr__(self, "after_action_feasible", after_feasible)
        object.__setattr__(self, "transition_cost", costs)
        object.__setattr__(self, "continuation_probabilities", continuation)
        object.__setattr__(self, "outcome_labels", labels)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_actions(self) -> int:
        return len(self.action_ids)

    @property
    def n_particles(self) -> int:
        return int(self.outcome_probabilities.shape[1])

    @property
    def n_outcomes(self) -> int:
        return int(self.outcome_probabilities.shape[2])

    # Explicit aliases make the after-action boundary discoverable to a
    # simulator and avoid hiding feasibility in an objective callback.
    @property
    def feasible_mask(self) -> np.ndarray:
        return self.initial_feasible

    @property
    def after_feasible(self) -> np.ndarray:
        return self.after_action_feasible

    @property
    def transition_costs(self) -> np.ndarray:
        return self.transition_cost

    @property
    def kernels(self) -> np.ndarray:
        return self.outcome_probabilities

    def continuation_kernels(self) -> np.ndarray:
        """Return a full branch-conditioned kernel tensor."""

        if self.continuation_probabilities is not None:
            return self.continuation_probabilities
        # [first, outcome, second, particle, second_outcome]
        return np.broadcast_to(
            self.outcome_probabilities[None, None, :, :, :],
            (self.n_actions, self.n_outcomes, self.n_actions,
             self.n_particles, self.n_outcomes),
        ).copy()


CandidateLibrary = CheckedCandidateSet


def _broadcast_first_values(value: Any, shape: tuple[int, int, int], *, name: str) -> np.ndarray:
    """Broadcast an objective table to ``(action, particle, outcome)``."""

    n_actions, n_particles, n_outcomes = shape
    if value is None:
        return np.zeros(shape, dtype=float)
    raw = np.asarray(value, dtype=float)
    candidates: list[np.ndarray] = []
    if raw.ndim == 0:
        candidates.append(np.broadcast_to(raw, shape))
    if raw.shape == (n_actions,):
        candidates.append(np.broadcast_to(raw[:, None, None], shape))
    if raw.shape == (n_particles,):
        candidates.append(np.broadcast_to(raw[None, :, None], shape))
    if raw.shape == (n_outcomes,):
        candidates.append(np.broadcast_to(raw[None, None, :], shape))
    if raw.shape == (n_actions, n_particles):
        candidates.append(np.broadcast_to(raw[:, :, None], shape))
    if raw.shape == (n_actions, n_outcomes):
        candidates.append(np.broadcast_to(raw[:, None, :], shape))
    if raw.shape == (n_particles, n_outcomes):
        candidates.append(np.broadcast_to(raw[None, :, :], shape))
    if raw.shape == shape:
        candidates.append(raw)
    if not candidates:
        raise ValueError(f"{name} cannot broadcast from shape {raw.shape} to {shape}")
    result = np.asarray(candidates[0], dtype=float).copy()
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _broadcast_continuation_values(
    value: Any,
    shape: tuple[int, int, int, int, int],
    *,
    name: str,
) -> np.ndarray:
    """Broadcast continuation rewards to ``(first, outcome, second, particle, outcome2)``."""

    n_first, n_first_outcome, n_second, n_particles, n_second_outcome = shape
    if value is None:
        return np.zeros(shape, dtype=float)
    raw = np.asarray(value, dtype=float)
    if raw.ndim == 0:
        return np.full(shape, float(raw), dtype=float)
    if raw.shape == (n_second, n_particles, n_second_outcome):
        return np.broadcast_to(raw[None, None, :, :, :], shape).copy()
    if raw.shape == (n_first, n_first_outcome, n_second, n_particles, n_second_outcome):
        return raw.copy()
    if raw.shape == (n_first, n_first_outcome, n_second, n_particles):
        return np.broadcast_to(raw[..., None], shape).copy()
    if raw.shape == (n_second,):
        return np.broadcast_to(raw[None, None, :, None, None], shape).copy()
    if raw.shape == (n_first, n_first_outcome, n_second):
        return np.broadcast_to(raw[..., None, None], shape).copy()
    if raw.shape == (n_first, n_first_outcome, n_second, n_second_outcome):
        return np.broadcast_to(raw[..., None, :], shape).copy()
    raise ValueError(f"{name} cannot broadcast from shape {raw.shape} to {shape}")


@dataclass(frozen=True)
class DecisionResult:
    """Complete finite-horizon decision and diagnostics."""

    method: str
    first_index: int
    first_action_id: str
    value: float
    first_scores: np.ndarray
    first_expected_scores: np.ndarray
    predictive: np.ndarray
    posteriors: np.ndarray
    continuation_indices: np.ndarray
    continuation_action_ids: tuple[str | None, ...]
    continuation_values: np.ndarray
    common_continuation_scores: np.ndarray
    branch_values: np.ndarray
    feasible_after: np.ndarray
    transition_costs: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def second_index(self) -> int | None:
        """The fixed A continuation, or ``None`` for B's branch policy."""

        if self.method.upper() == "A":
            index = int(self.continuation_indices[0])
            return None if index < 0 else index
        return None

    @property
    def second_action_id(self) -> str | None:
        index = self.second_index
        return None if index is None else str(self.continuation_action_ids[0])

    def policy_dict(self) -> dict[str, Any]:
        """JSON-friendly public policy map for runner logs."""

        return {
            "method": self.method,
            "first_action": self.first_action_id,
            "first_index": int(self.first_index),
            "value": float(self.value),
            "continuation_by_outcome": {
                str(outcome): (None if action is None else str(action))
                for outcome, action in enumerate(self.continuation_action_ids)
            },
            "predictive": [float(value) for value in self.predictive],
        }


def _argmax_stable(values: np.ndarray, *, feasible: np.ndarray | None = None) -> int:
    array = np.asarray(values, dtype=float)
    if feasible is None:
        feasible = np.isfinite(array)
    else:
        feasible = np.asarray(feasible, dtype=bool) & np.isfinite(array)
    valid = np.flatnonzero(feasible)
    if valid.size == 0:
        return -1
    # np.argmax over the stable canonical bank order is an intentional common
    # tie break.  NaN never enters because feasible includes isfinite.
    return int(valid[int(np.argmax(array[valid]))])


def _method_name(method: str) -> str:
    value = str(method).strip().upper()
    if value in {"A", "STRONG_A", "FIXED", "OPEN_LOOP"}:
        return "A"
    if value in {"B", "FEEDBACK", "ADAPTIVE", "CLOSED_LOOP"}:
        return "B"
    raise ValueError("method must be 'A' or 'B'")


def plan_two_stage(
    belief: BayesBelief | np.ndarray,
    candidates: CheckedCandidateSet,
    *,
    immediate_value: np.ndarray | None = None,
    continuation_value: np.ndarray | None = None,
    common_recovery_value: np.ndarray | None = None,
    method: str = "B",
    cost_weight: float = 1.0,
    support_tolerance: float = _DEFAULT_TOL,
) -> DecisionResult:
    """Select a two-stage program under the supplied common objective.

    ``immediate_value`` and ``common_recovery_value`` are arbitrary caller-
    supplied realized objective tables; each may be scalar/action/particle/
    outcome broadcastable to ``(A, P, O)``.  Their expected sum is the current
    acquisition score.  ``continuation_value`` is similarly arbitrary and may
    be supplied as ``(second_action, particle, second_outcome)`` or as a full
    branch table ``(first, first_outcome, second, particle, second_outcome)``.
    It is *not* a hidden truth score: the caller controls the public objective,
    while latent particles only provide the common predictive integration.

    ``cost_weight`` applies to both arms identically.  ``first_cost`` and
    ``transition_cost`` are subtracted in the same units as the objective.
    ``common_recovery_value`` is a convenience for an immediate-observability
    score plus a shared recovery term; any branch-specific recovery can be
    placed directly in ``continuation_value`` or the transition-cost table.
    """

    arm = _method_name(method)
    if not isinstance(candidates, CheckedCandidateSet):
        raise TypeError("candidates must be a CheckedCandidateSet")
    if not math.isfinite(float(cost_weight)) or float(cost_weight) < 0.0:
        raise ValueError("cost_weight must be finite and nonnegative")
    if isinstance(belief, BayesBelief):
        weights = belief.weights
        n_particles = belief.n_particles
    else:
        weights = _as_probability_vector(belief, name="weights")
        n_particles = weights.size
    if n_particles != candidates.n_particles:
        raise ValueError("belief particle count does not match candidate kernels")
    kernels = candidates.outcome_probabilities
    n_actions, _, n_outcomes = kernels.shape
    first_shape = (n_actions, n_particles, n_outcomes)
    first_reward = _broadcast_first_values(immediate_value, first_shape, name="immediate_value")
    first_reward += _broadcast_first_values(common_recovery_value, first_shape, name="common_recovery_value")
    first_expected = np.einsum("p,apo,apo->a", weights, kernels, first_reward)
    first_expected -= float(cost_weight) * candidates.first_cost

    predictive = np.zeros((n_actions, n_outcomes), dtype=float)
    posteriors = np.zeros((n_actions, n_outcomes, n_particles), dtype=float)
    for action in range(n_actions):
        predictive[action], posteriors[action] = posterior_table(weights, kernels[action])
    # Branch-conditioned continuation kernel, [first, outcome, second, p, o2].
    continuation_kernels = candidates.continuation_kernels()
    continuation_shape = (n_actions, n_outcomes, n_actions, n_particles, n_outcomes)
    continuation_reward = _broadcast_continuation_values(
        continuation_value,
        continuation_shape,
        name="continuation_value",
    )
    branch_values = np.einsum(
        "aop,aobpq,aobpq->aob",
        posteriors,
        continuation_kernels,
        continuation_reward,
    )
    branch_values -= float(cost_weight) * candidates.transition_cost

    # Initial infeasibility is common and never hidden in objective scores.
    first_expected = np.where(candidates.initial_feasible, first_expected, -np.inf)
    support = predictive > float(support_tolerance)
    after_feasible = candidates.after_action_feasible.copy()
    # A's fixed continuation must be feasible on every positive-probability
    # first outcome.  B's branch map only needs the outcome-specific mask.
    common_feasible = np.ones((n_actions, n_actions), dtype=bool)
    for first in range(n_actions):
        if np.any(support[first]):
            common_feasible[first] = np.all(after_feasible[first, support[first], :], axis=0)
        else:
            common_feasible[first] = False
    # For impossible branches, zeroing before expectation avoids 0 * -inf and
    # leaves a finite diagnostic table.  Positive branches without any valid
    # continuation remain -inf and make the parent first action inadmissible.
    branch_finite = np.isfinite(branch_values)
    branch_with_support = np.where(support[:, :, None], branch_finite, True)
    branch_valid = after_feasible & branch_with_support

    fixed_scores = np.full((n_actions, n_actions), -np.inf, dtype=float)
    for first in range(n_actions):
        for second in range(n_actions):
            if not common_feasible[first, second]:
                continue
            fixed_scores[first, second] = float(
                np.sum(np.where(support[first], predictive[first] * branch_values[first, :, second], 0.0))
            )
    fixed_best = np.asarray([_argmax_stable(row) for row in fixed_scores], dtype=int)
    fixed_future = np.asarray([
        row[index] if index >= 0 else -np.inf
        for row, index in zip(fixed_scores, fixed_best)
    ], dtype=float)

    feedback_best = np.full((n_actions, n_outcomes), -1, dtype=int)
    feedback_values = np.full((n_actions, n_outcomes), -np.inf, dtype=float)
    for first in range(n_actions):
        for outcome in range(n_outcomes):
            if not support[first, outcome]:
                # This branch never executes; retaining -1 makes the public
                # policy map explicit without inventing a choice.
                continue
            index = _argmax_stable(branch_values[first, outcome], feasible=branch_valid[first, outcome])
            feedback_best[first, outcome] = index
            if index >= 0:
                feedback_values[first, outcome] = branch_values[first, outcome, index]
    feedback_future = np.full(n_actions, -np.inf, dtype=float)
    for first in range(n_actions):
        live = support[first]
        if not np.any(live):
            continue
        if np.any(feedback_best[first, live] < 0):
            continue
        feedback_future[first] = float(np.sum(predictive[first, live] * feedback_values[first, live]))

    continuation_scores = fixed_scores if arm == "A" else feedback_values
    total_scores = first_expected + (fixed_future if arm == "A" else feedback_future)
    first_index = _argmax_stable(total_scores)
    if first_index < 0:
        raise ValueError("no feasible complete two-stage policy under the supplied masks")
    if arm == "A":
        second = int(fixed_best[first_index])
        continuation_indices = np.full(n_outcomes, second, dtype=int)
        continuation_ids: tuple[str | None, ...] = tuple(
            candidates.action_ids[second] if second >= 0 else None for _ in range(n_outcomes)
        )
        value = float(total_scores[first_index])
    else:
        continuation_indices = feedback_best[first_index].copy()
        continuation_ids = tuple(
            candidates.action_ids[index] if index >= 0 else None
            for index in continuation_indices
        )
        value = float(total_scores[first_index])
    if arm == "A":
        # Per-first-outcome branch values for the selected fixed continuation;
        # ``common_continuation_scores`` below retains the full second-action
        # ranking for this first action.
        selected_continuation_values = branch_values[first_index, :, int(continuation_indices[0])].copy()
    else:
        selected_continuation_values = feedback_values[first_index].copy()
    diagnostics = {
        "first_expected_scores": first_expected.copy(),
        "fixed_future_scores": fixed_future.copy(),
        "feedback_future_scores": feedback_future.copy(),
        "fixed_common_feasible": common_feasible.copy(),
        "feedback_best_indices": feedback_best.copy(),
        "support": support.copy(),
        "cost_weight": float(cost_weight),
        "objective_shared": True,
    }
    return DecisionResult(
        method=arm,
        first_index=int(first_index),
        first_action_id=candidates.action_ids[first_index],
        value=value,
        first_scores=total_scores.copy(),
        first_expected_scores=first_expected.copy(),
        predictive=predictive[first_index].copy(),
        posteriors=posteriors[first_index].copy(),
        continuation_indices=continuation_indices,
        continuation_action_ids=continuation_ids,
        continuation_values=selected_continuation_values,
        common_continuation_scores=fixed_scores[first_index].copy(),
        branch_values=branch_values[first_index].copy(),
        feasible_after=after_feasible[first_index].copy(),
        transition_costs=candidates.transition_cost[first_index].copy(),
        diagnostics=diagnostics,
    )


def select_action(
    belief: BayesBelief | np.ndarray,
    candidates: CheckedCandidateSet,
    **kwargs: Any,
) -> DecisionResult:
    """Short alias for :func:`plan_two_stage`."""

    return plan_two_stage(belief, candidates, **kwargs)


plan = plan_two_stage


def compare_methods(
    belief: BayesBelief | np.ndarray,
    candidates: CheckedCandidateSet,
    *,
    immediate_value: np.ndarray | None = None,
    continuation_value: np.ndarray | None = None,
    common_recovery_value: np.ndarray | None = None,
    cost_weight: float = 1.0,
    support_tolerance: float = _DEFAULT_TOL,
) -> dict[str, DecisionResult]:
    """Evaluate A and B with one immutable bank, objective, prior, and costs."""

    common = dict(
        immediate_value=immediate_value,
        continuation_value=continuation_value,
        common_recovery_value=common_recovery_value,
        cost_weight=cost_weight,
        support_tolerance=support_tolerance,
    )
    return {
        "A": plan_two_stage(belief, candidates, method="A", **common),
        "B": plan_two_stage(belief, candidates, method="B", **common),
    }


def plan_scan_pair(
    first_value: np.ndarray,
    likelihood: np.ndarray,
    captures: np.ndarray,
    tail_cost: np.ndarray,
    stop: np.ndarray,
    *,
    method: str = "B",
    discount: float = 1.0,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Compact two-stage scan planner for a uniform finite particle bank.

    This is the runner-facing specialization of :func:`plan_two_stage`.  It
    avoids materializing a five-dimensional branch reward tensor when the
    continuation objective is simply a latent-particle capture probability.

    Parameters use the following shapes:

    ``first_value``
        Current acquisition score, one value per first program ``(A,)``.
        The caller owns this score (for example immediate centroid capture
        plus its common recovery term).
    ``likelihood``
        Public first-program kernel ``L[a, p, o]`` with rows summing to one.
    ``captures``
        Tail capture probability ``C[p, b]`` for each shared latent particle
        and common tail candidate.
    ``tail_cost``
        Checked outward/recovery/transition cost ``T[a, b]``.  It is common
        to A and B and is weighted by the first outcome probability.
    ``stop``
        Boolean ``stop[a, o]``.  A true branch has completed at the first
        stage and receives zero continuation value; its returned suffix index
        is ``-1``.

    The shared branch table is exactly

    ``J[a,o,b] = mean_p(L[a,p,o] * C[p,b])
                 - mean_p(L[a,p,o]) * T[a,b]``.

    A chooses one suffix using ``max_b sum_o J`` (``max`` outside the
    expectation).  B chooses a suffix after every non-stop public outcome via
    ``sum_o max_b J`` (``max`` inside the expectation).  ``discount`` applies
    to the same continuation term for both arms.  No policy-specific cost or
    candidate list is introduced here.

    Returns ``(first_index, suffix_indices, details)``.  ``suffix_indices``
    is indexed by first-stage outcome for the selected first program; all stop
    branches are ``-1``.  ``details`` contains only Python scalars/lists so it
    can be written directly to JSON logs.
    """

    values = np.asarray(first_value, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("first_value must be a finite one-dimensional array")
    L = np.asarray(likelihood, dtype=float)
    if L.ndim != 3 or not np.isfinite(L).all() or (L < -1.0e-12).any():
        raise ValueError("likelihood must be finite, nonnegative, and shaped (first, particle, outcome)")
    n_first, n_particles, n_outcomes = L.shape
    if n_first != values.size:
        raise ValueError("first_value length must match likelihood first-program count")
    L = np.maximum(L, 0.0)
    row_mass = L.sum(axis=2)
    if not np.allclose(row_mass, 1.0, rtol=2.0e-9, atol=2.0e-11):
        raise ValueError("likelihood rows must sum to one")
    C = np.asarray(captures, dtype=float)
    if C.ndim != 2 or C.shape[0] != n_particles or not np.isfinite(C).all():
        raise ValueError("captures must have shape (particles, tails) and be finite")
    if (C < -1.0e-12).any() or (C > 1.0 + 1.0e-12).any():
        raise ValueError("captures must lie in [0, 1]")
    C = np.clip(C, 0.0, 1.0)
    n_tails = C.shape[1]
    T = np.asarray(tail_cost, dtype=float)
    if T.shape != (n_first, n_tails) or not np.isfinite(T).all() or (T < 0.0).any():
        raise ValueError("tail_cost must be finite, nonnegative, and shaped (first, tails)")
    S = np.asarray(stop, dtype=bool)
    if S.shape != (n_first, n_outcomes):
        raise ValueError("stop must have shape (first, outcome)")
    discount_value = float(discount)
    if not math.isfinite(discount_value) or discount_value < 0.0:
        raise ValueError("discount must be finite and nonnegative")

    # This is intentionally written in the exact algebraic form used by the
    # dynamic runner.  L.mean(axis=1) is the predictive public outcome mass
    # because the particle prior is uniform in this compact interface.
    predictive = L.mean(axis=1)
    joint = np.einsum("apo,pb->aob", L, C) / float(n_particles)
    joint -= predictive[..., None] * T[:, None, :]
    # A valid first-stage completion has no second-stage tail.  Zeroing both
    # reward and cost on those branches prevents an accidental continuation
    # from changing the result while preserving the branch probability in the
    # first-stage score supplied by the caller.
    joint = np.where(S[..., None], 0.0, joint)

    fixed_by_tail = joint.sum(axis=1)
    fixed_index = np.argmax(fixed_by_tail, axis=1).astype(int)
    fixed_future = fixed_by_tail[np.arange(n_first), fixed_index]

    branch_index = np.argmax(joint, axis=2).astype(int)
    branch_best = np.max(joint, axis=2)
    feedback_future = branch_best.sum(axis=1)

    fixed_score = values + discount_value * fixed_future
    feedback_score = values + discount_value * feedback_future
    arm = _method_name(method)
    scores = fixed_score if arm == "A" else feedback_score
    first = int(np.argmax(scores))

    if arm == "A":
        suffixes = np.where(S[first], -1, fixed_index[first]).astype(int)
    else:
        suffixes = np.where(S[first], -1, branch_index[first]).astype(int)

    # Branch mass where B's feedback suffix differs from A's fixed suffix on
    # the same first action.  The predictive mass is the correct weighting;
    # raw counts would exaggerate tiny/zero quadrature bins.
    alternate = (~S[first]) & (branch_index[first] != fixed_index[first])
    alternate_mass = float(np.sum(predictive[first] * alternate))
    same_first_gap = float(feedback_score[first] - fixed_score[first])
    details: dict[str, Any] = {
        "method": arm,
        "requested_method": str(method),
        "first_index": first,
        "first_value": values.tolist(),
        "first_scores_A_fixed": fixed_score.tolist(),
        "first_scores_B_feedback": feedback_score.tolist(),
        "selected_score": float(scores[first]),
        "predictive_outcome": predictive[first].tolist(),
        "support": ((predictive[first] > _DEFAULT_TOL) & (~S[first])).tolist(),
        "stop": S[first].tolist(),
        "fixed_index": fixed_index.tolist(),
        "selected_fixed_index": int(fixed_index[first]),
        "feedback_index_by_first": branch_index.tolist(),
        "selected_feedback_index": branch_index[first].tolist(),
        "suffix_indices": suffixes.tolist(),
        "fixed_future_by_first": fixed_future.tolist(),
        "feedback_future_by_first": feedback_future.tolist(),
        "same_first_gap": same_first_gap,
        "support_weighted_alternate_branch_mass": alternate_mass,
        # Short aliases are useful in compact JSON reports and preserve the
        # mathematical names used in the study notes.
        "alternate_branch_mass": alternate_mass,
        "fixed_index_selected": int(fixed_index[first]),
        "joint": joint.tolist(),
        "discount": discount_value,
        "shared_tail_cost": True,
        "common_candidate_bank": True,
    }
    return first, suffixes, details


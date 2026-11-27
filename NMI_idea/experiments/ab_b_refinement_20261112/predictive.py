"""Small deterministic predictive-set approximation for the public sensor.

The refinement runner only needs a *finite* public outcome support when it
scores a candidate first acquisition.  :func:`predictive_sets` supplies that
support from a Gaussian belief and one or two correlated linear projections::

    z_i = base_i + h_i @ theta,       theta ~ N(mean, covariance)

The returned values are public centroids, not latent target labels.  A target
centroid is emitted with probability ``p_detect`` only when the noisy value is
inside ``[-half_width, half_width]``.  ``clutter_rate`` denotes an
independent, at-most-one uniform false centroid in that same aperture.  A
deterministic Sobol rule integrates the correlated projected Gaussian and the
noise-mixture components.  The finite support is then reduced by weighted
quantile pooling; this keeps the number of public sets bounded by ``count``
without enumerating all histories.

This is intentionally a *predictive quadrature*, not a posterior update.  A
caller can pass each returned public tuple to its usual ``update_set``
routine; at ``count=8`` this helper exposes at most eight such calls (and no
truth labels or random seed are involved here).

The approximation has the expected sensor-only checks: with ``p_detect=0``
the empty and clutter-only masses are ``1-clutter_rate`` and
``clutter_rate``; with zero clutter the output contains no clutter-only
branch; and all emitted centroids stay inside the declared gate.  The main
limitation is the finite deterministic quadrature/quantile reduction: very
narrow, highly multimodal, or nearly gate-boundary predictive laws need a
larger ``count`` (or an exact continuous likelihood) for accurate tail masses.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # SciPy is available in the experiment environment.
    from scipy.special import ndtri as _ndtri
    from scipy.stats import qmc as _qmc
except Exception:  # pragma: no cover - a small stdlib fallback keeps the API portable.
    _ndtri = None
    _qmc = None


__all__ = ["predictive_sets"]

_TINY = 1.0e-14
_DEFAULT_NOISE_MIXTURE = ((0.85, 0.85), (0.15, 4.0))


def _finite_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty one-dimensional array")
    return array


def _belief_moments(belief: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(mean, covariance)`` from the small belief variants in use."""

    mean = covariance = None
    if isinstance(belief, Mapping):
        for key in ("mean", "mu", "location"):
            if key in belief:
                mean = belief[key]
                break
        for key in ("covariance", "cov", "C", "variance"):
            if key in belief:
                covariance = belief[key]
                break
    elif isinstance(belief, (tuple, list)) and len(belief) == 2:
        mean, covariance = belief
    else:
        for key in ("mean", "mu", "location"):
            if hasattr(belief, key):
                mean = getattr(belief, key)
                break
        for key in ("covariance", "cov", "C", "variance"):
            if hasattr(belief, key):
                covariance = getattr(belief, key)
                break
    if mean is None or covariance is None:
        raise TypeError("belief must provide mean and covariance")
    mean_array = _finite_vector(mean, "belief mean")
    covariance_array = np.asarray(covariance, dtype=float)
    if covariance_array.shape != (mean_array.size, mean_array.size):
        raise ValueError("belief covariance shape does not match its mean")
    if not np.isfinite(covariance_array).all():
        raise ValueError("belief covariance must be finite")
    covariance_array = (covariance_array + covariance_array.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_array)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(np.min(eigenvalues)) < -1.0e-10 * scale:
        raise ValueError("belief covariance must be positive semidefinite")
    # A clipped eigendecomposition also gives a stable square root for a
    # perfectly correlated pair of projections.
    covariance_array = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return mean_array, (covariance_array + covariance_array.T) * 0.5


def _model_rows(models: Any, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize rows from ``(h, base)`` pairs or compact array/dict forms."""

    if isinstance(models, Mapping) and ("h" in models or "H" in models):
        rows = models.get("h", models.get("H"))
        bases = models.get("base", models.get("bases", models.get("g")))
        if bases is None:
            raise ValueError("models mapping must provide base/bases with h/H")
        h_array = np.asarray(rows, dtype=float)
        base_array = np.asarray(bases, dtype=float)
        if h_array.ndim == 1:
            h_array = h_array.reshape(1, -1)
        if base_array.ndim == 0:
            base_array = base_array.reshape(1)
        if h_array.ndim != 2 or base_array.ndim != 1 or h_array.shape[0] != base_array.size:
            raise ValueError("models h/H and base/bases have incompatible shapes")
        parsed = [(h_array[index], base_array[index]) for index in range(h_array.shape[0])]
    else:
        # A single pair is a convenient spelling for the one-projection case
        # (``models=(h, base)``).  Distinguish it from the normal iterable of
        # pairs before trying the compact matrix representation below.
        if isinstance(models, (tuple, list)) and len(models) == 2:
            try:
                first = np.asarray(models[0], dtype=float)
                second = np.asarray(models[1], dtype=float)
            except (TypeError, ValueError):
                first = second = np.asarray([], dtype=float)
            if first.ndim == 1 and first.size == dimension and second.ndim == 0:
                models = [models]
        array = None
        try:
            candidate = np.asarray(models, dtype=float)
            if candidate.ndim == 2 and candidate.shape[1] == dimension + 1:
                array = candidate
        except (TypeError, ValueError):
            array = None
        if array is not None:
            parsed = [(row[:-1], row[-1]) for row in array]
        else:
            try:
                iterable = list(models)
            except TypeError as exc:
                raise TypeError("models must be one or two (h, base) rows") from exc
            parsed = []
            for model in iterable:
                if isinstance(model, Mapping):
                    row = model.get("h", model.get("row", model.get("H")))
                    base = model.get("base", model.get("g"))
                else:
                    row = getattr(model, "h", getattr(model, "row", None))
                    base = getattr(model, "base", getattr(model, "g", None))
                    if row is None or base is None:
                        try:
                            row, base = model
                        except (TypeError, ValueError) as exc:
                            raise ValueError("each model must be (h, base)") from exc
                if row is None or base is None:
                    raise ValueError("each model must provide h/row and base")
                parsed.append((row, base))
    if len(parsed) not in (1, 2):
        raise ValueError("predictive_sets supports exactly one or two projections")
    h_rows = np.asarray([_finite_vector(row, "model row") for row, _ in parsed], dtype=float)
    bases = np.asarray([float(base) for _, base in parsed], dtype=float)
    if h_rows.shape != (len(parsed), dimension) or not np.isfinite(bases).all():
        raise ValueError("model rows/bases do not match the belief dimension")
    return h_rows, bases


def _probabilities(value: Any, n: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.repeat(float(values), n)
    if values.shape != (n,) or not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} must be a scalar or a length-{n} sequence in [0, 1]")
    return values


def _mixture(value: Any) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        value = _DEFAULT_NOISE_MIXTURE
    try:
        raw = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_mixture must contain (weight, standard_deviation) pairs") from exc
    if raw.ndim == 1 and raw.size == 2:
        raw = raw.reshape(1, 2)
    if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] == 0 or not np.isfinite(raw).all():
        raise ValueError("noise_mixture must have shape (components, 2)")
    weights, scales = raw[:, 0], raw[:, 1]
    if np.any(weights < 0.0) or np.any(scales <= 0.0) or not float(np.sum(weights)) > 0.0:
        raise ValueError("noise-mixture weights must be nonnegative and scales positive")
    keep = weights > 0.0
    return weights[keep] / float(np.sum(weights)), scales[keep]


def _qmc_uniform(n: int, dimension: int) -> np.ndarray:
    """Return a deterministic, endpoint-free low-discrepancy rectangle rule."""

    if _qmc is not None:
        exponent = int(round(math.log2(n)))
        raw = np.asarray(_qmc.Sobol(d=dimension, scramble=False).random_base2(exponent), dtype=float)
    else:  # pragma: no cover - exercised only on installations without SciPy.
        # Deterministic Halton fallback.  It is less uniform than Sobol but
        # still avoids a random seed and is adequate for this tiny support.
        primes = (2, 3, 5, 7, 11, 13, 17, 19)

        def radical(index: int, base: int) -> float:
            result = 0.0
            scale = 1.0 / base
            while index:
                index, digit = divmod(index, base)
                result += digit * scale
                scale /= base
            return result

        raw = np.array(
            [[radical(index + 1, primes[j % len(primes)]) for j in range(dimension)] for index in range(n)],
            dtype=float,
        )
    # Sobol's first point is the all-zero vertex.  A fixed midpoint shift is
    # deterministic and prevents an artificial extreme-normal miss branch.
    return (raw + 0.5 / float(n)) % 1.0


def _standard_normal(uniform: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(uniform, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    if _ndtri is not None:
        return np.asarray(_ndtri(clipped), dtype=float)
    inverse = np.vectorize(NormalDist().inv_cdf, otypes=[float])
    return inverse(clipped)


def _quantile_pool(records: list[tuple[tuple[float, ...], float]], slots: int, half_width: float) -> list[tuple[tuple[float, ...], float]]:
    """Pool a same-branch weighted sample into at most ``slots`` tuples."""

    if slots <= 0 or not records:
        return []
    merged: dict[tuple[float, ...], float] = defaultdict(float)
    for values, weight in records:
        if weight > 0.0 and math.isfinite(weight):
            merged[values] += float(weight)
    items = sorted(merged.items(), key=lambda item: item[0])
    if not items:
        return []
    if slots >= len(items):
        return [(values, float(weight)) for values, weight in items]
    total = float(sum(weight for _, weight in items))
    if not total > 0.0:
        return []
    slots = min(int(slots), len(items))
    means = [np.zeros(len(items[0][0]), dtype=float) for _ in range(slots)]
    masses = np.zeros(slots, dtype=float)
    cumulative = 0.0
    bucket = 0
    for values, weight in items:
        remaining = float(weight)
        vector = np.asarray(values, dtype=float)
        while remaining > _TINY and bucket < slots:
            boundary = total * float(bucket + 1) / float(slots)
            take = min(remaining, max(boundary - cumulative, 0.0))
            if take <= _TINY:
                bucket += 1
                continue
            means[bucket] += take * vector
            masses[bucket] += take
            cumulative += take
            remaining -= take
            if cumulative >= boundary - _TINY:
                bucket += 1
    pooled: list[tuple[tuple[float, ...], float]] = []
    for mean, mass in zip(means, masses):
        if mass <= _TINY:
            continue
        values = np.clip(mean / mass, -float(half_width), float(half_width))
        pooled.append((tuple(sorted(float(value) for value in values)), float(mass)))
    return pooled


def predictive_sets(
    belief: Any,
    models: Iterable[Any],
    *,
    half_width: float,
    p_detect: float | Sequence[float],
    noise_mixture: Sequence[Sequence[float]] = _DEFAULT_NOISE_MIXTURE,
    clutter_rate: float,
    count: int = 8,
) -> list[tuple[tuple[float, ...], float]]:
    """Return a bounded weighted support of public centroid sets.

    Parameters are deliberately sensor-only.  ``belief`` accepts a
    ``(mean, covariance)`` pair, a mapping with ``mean``/``covariance`` (or
    their common short names), or an object exposing those attributes.
    ``models`` is one or two ``(h, base)`` pairs; a compact ``(n, d+1)`` array
    with the base in its final column is also accepted.  A noise component is
    ``(mixture_weight, standard_deviation_mm)``.

    The result is deterministic and has at most ``count`` entries.  Each
    entry is ``(tuple(sorted_centroid_values), probability_weight)``.  Empty
    public sets are pooled globally, while non-empty branches receive
    weighted quantile pooling so target/miss and clutter support survives the
    small ``count=4``/``count=8`` budgets.
    """

    width = float(half_width)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("half_width must be finite and positive")
    false = float(clutter_rate)
    if not math.isfinite(false) or false < 0.0 or false > 1.0:
        raise ValueError("clutter_rate must lie in [0, 1]")
    try:
        count_value = int(count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("count must be a positive integer") from exc
    if isinstance(count, bool) or count_value != count or count_value < 1:
        raise ValueError("count must be a positive integer")
    budget = count_value
    mean, covariance = _belief_moments(belief)
    h_rows, bases = _model_rows(models, mean.size)
    n_targets = h_rows.shape[0]
    detects = _probabilities(p_detect, n_targets, "p_detect")
    mixture_weights, mixture_scales = _mixture(noise_mixture)

    # Project the belief before adding sensor noise.  In particular, the
    # off-diagonal term is retained for two correlated projections.
    projected_mean = bases + h_rows @ mean
    projected_covariance = h_rows @ covariance @ h_rows.T
    projected_covariance = (projected_covariance + projected_covariance.T) * 0.5
    eigvals, eigvecs = np.linalg.eigh(projected_covariance)
    scale = max(1.0, float(np.max(np.abs(eigvals))))
    if float(np.min(eigvals)) < -1.0e-10 * scale:
        raise ValueError("projected model covariance is not positive semidefinite")
    projected_root = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0)))

    # The rule grows only with the requested support budget.  For n<=2 this
    # is inexpensive even after the at-most-one-clutter and detection-mask
    # branches are expanded.
    qmc_count = 1 << int(math.ceil(math.log2(max(16, 4 * budget))))
    qmc_points = _qmc_uniform(qmc_count, 2 * n_targets + 1)
    latent_normals = _standard_normal(qmc_points[:, :n_targets])
    noise_normals = _standard_normal(qmc_points[:, n_targets : 2 * n_targets])
    latent = projected_mean[None, :] + latent_normals @ projected_root.T
    clutter_values = -width + 2.0 * width * qmc_points[:, -1]

    # Each record carries a branch signature so the reducer can keep a
    # clutter-only and a two-target branch even when their public cardinality
    # happens to match.  The signature is not returned and is not a truth
    # label; it is only a deterministic quadrature partition.
    branch_records: dict[tuple[int, bool], list[tuple[tuple[float, ...], float]]] = defaultdict(list)
    for component_indices in product(range(len(mixture_weights)), repeat=n_targets):
        component_weight = float(np.prod([mixture_weights[index] for index in component_indices]))
        if component_weight <= 0.0:
            continue
        scales_by_target = np.asarray([mixture_scales[index] for index in component_indices], dtype=float)
        observations = latent + noise_normals * scales_by_target[None, :]
        for row_index, observation in enumerate(observations):
            hit = detects * (np.abs(observation) <= width + 1.0e-12)
            for mask in range(1 << n_targets):
                branch_probability = 1.0
                values: list[float] = []
                for target_index in range(n_targets):
                    if mask & (1 << target_index):
                        branch_probability *= float(hit[target_index])
                        values.append(float(observation[target_index]))
                    else:
                        branch_probability *= float(1.0 - hit[target_index])
                if branch_probability <= 0.0:
                    continue
                sample_weight = component_weight * branch_probability / float(qmc_count)
                # No clutter branch.
                if 1.0 - false > 0.0:
                    public = tuple(sorted(values))
                    branch_records[(mask, False)].append((public, sample_weight * (1.0 - false)))
                # At most one uniform clutter centroid.
                if false > 0.0:
                    public = tuple(sorted(values + [float(clutter_values[row_index])]))
                    branch_records[(mask, True)].append((public, sample_weight * false))

    # Empty is special: all miss/out-of-gate branches are the same public
    # observation and must be grouped before support reduction.
    empty_mass = 0.0
    for signature in list(branch_records):
        retained: list[tuple[tuple[float, ...], float]] = []
        for values, weight in branch_records[signature]:
            if not values:
                empty_mass += float(weight)
            else:
                retained.append((values, float(weight)))
        if retained:
            branch_records[signature] = retained
        else:
            del branch_records[signature]

    group_mass = {
        signature: float(sum(weight for _, weight in records))
        for signature, records in branch_records.items()
        if records and float(sum(weight for _, weight in records)) > _TINY
    }
    # Reserve the empty outcome whenever it has positive mass.  A clutter-only
    # branch is also reserved when possible, so clutter cannot disappear from
    # an eight-candidate support merely because its rate is small.
    selected: list[tuple[int, bool]] = []
    total_slots = budget - (1 if empty_mass > _TINY else 0)
    available = total_slots
    clutter_only = (0, True)
    if available > 0 and clutter_only in group_mass:
        selected.append(clutter_only)
        available -= 1
    remaining = [signature for signature in group_mass if signature not in selected]
    if available > 0:
        remaining.sort(key=lambda signature: (-group_mass[signature], signature))
        selected.extend(remaining[:available])

    # If an all-detection law has no empty mass and budget is one, choose its
    # highest-mass non-empty branch.  This also handles a pathological
    # floating-point all-zero empty branch gracefully.
    if not selected and empty_mass <= _TINY and group_mass:
        selected = [max(group_mass, key=lambda signature: (group_mass[signature], tuple(-x for x in signature)))]

    # Split extra slots among selected branches by residual mass.  Every
    # selected branch receives one slot; extras follow the largest current
    # mass/slot ratio and therefore preserve broad high-mass branches.
    slots_by_signature = {signature: 1 for signature in selected}
    extra = max(0, total_slots - len(selected)) if selected else 0
    while extra:
        target = max(
            selected,
            key=lambda signature: (group_mass[signature] / slots_by_signature[signature], -signature[0], -int(signature[1])),
        )
        slots_by_signature[target] += 1
        extra -= 1

    reduced: list[tuple[tuple[float, ...], float]] = []
    if empty_mass > _TINY:
        reduced.append((tuple(), empty_mass))
    for signature in selected:
        reduced.extend(_quantile_pool(branch_records[signature], slots_by_signature[signature], width))

    # Quantile representatives from different signatures can coincide (for
    # example, identical h rows).  Public sets are unlabeled, so merge those
    # exact tuples after reduction and renormalize once at the boundary.
    merged: dict[tuple[float, ...], float] = defaultdict(float)
    for values, weight in reduced:
        if weight > 0.0 and math.isfinite(weight):
            merged[values] += float(weight)
    if not merged:
        # The branch sum is analytically one; this fallback is only protection
        # against an unusual underflow in a caller-provided mixture.
        return [(tuple(), 1.0)]
    total = float(sum(merged.values()))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("predictive quadrature produced non-finite probability mass")
    result = [(values, float(weight) / total) for values, weight in merged.items()]
    result.sort(key=lambda item: (0 if not item[0] else 1, -item[1], item[0]))
    return result[:budget]


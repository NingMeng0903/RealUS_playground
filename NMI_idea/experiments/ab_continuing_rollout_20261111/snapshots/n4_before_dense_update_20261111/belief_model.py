"""Bounded SENSOR_V2 public belief model.

The planner-facing state contains only public packets.  It is a bounded
Rao--Blackwellized approximation: Sobol geometry particles, a present/lost
HMM, and a capped Gaussian association mixture for local response ``w``.
Gaussian natural statistics are exact only conditionally on one geometry and
association component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Iterable

import numpy as np
from scipy.special import ndtr, owens_t
from scipy.stats import qmc

try:  # Optional performance backend; the NumPy reference remains canonical.
    from numba import njit as _numba_njit
    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on minimal deployments
    _numba_njit = None
    _NUMBA_AVAILABLE = False

try:
    from acoustics import (AXIS_TOLERANCE_RAD, CENTER_TOLERANCE_M, FORK_SUPPORT_M,
                           P_CLUTTER, PRIOR_SD, QUALITY_THRESHOLD, ProbeAction,
                           PublicObservation, YAW_SCALE_RAD)
except ImportError:  # pragma: no cover
    from .acoustics import (AXIS_TOLERANCE_RAD, CENTER_TOLERANCE_M, FORK_SUPPORT_M,
                            P_CLUTTER, PRIOR_SD, QUALITY_THRESHOLD, ProbeAction,
                            PublicObservation, YAW_SCALE_RAD)

N_GEOMETRY = 512
MAX_COMPONENTS = 4
GEOMETRY_BOUNDS = np.array(((-.003, .003), (-.08, .08), (-.35, .35),
                            (.018, .026)), dtype=float)


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -60., 60.)))


def _angle_difference(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2. * np.pi) - np.pi


def _log_normal(x, mean, variance):
    variance = max(float(variance), 1.e-12)
    d = float(x) - float(mean)
    return -.5 * (d * d / variance + math.log(2. * math.pi * variance))


def _log_uniform(x, low, high):
    return -math.log(high - low) if np.isfinite(x) and low <= x <= high else -math.inf


def _log(value):
    return math.log(max(float(value), 1.e-300))


def _normalize_logs(logs):
    logs = np.asarray(logs, dtype=float)
    finite = np.isfinite(logs)
    if not np.any(finite):
        raise FloatingPointError("all public likelihood masses are zero")
    mx = float(np.max(logs[finite]))
    scaled = np.zeros_like(logs)
    scaled[finite] = np.exp(logs[finite] - mx)
    total = float(np.sum(scaled))
    if not np.isfinite(total) or total <= 0.:
        raise FloatingPointError("non-finite public likelihood mass")
    return scaled / total, mx + math.log(total)


def _fingerprint(event_id, action, packet):
    def clean(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, float) and not np.isfinite(x):
            return "nan" if np.isnan(x) else ("inf" if x > 0 else "-inf")
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in sorted(x.items(), key=lambda z: str(z[0]))}
        if isinstance(x, (tuple, list)):
            return [clean(v) for v in x]
        if hasattr(x, "__dict__"):
            return clean(vars(x))
        return x
    raw = json.dumps(clean({"event": event_id, "action": action, "packet": packet}), sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class ModelParams:
    """Train-fitted/frozen parameters; defaults are SENSOR_V2 declared values."""
    geometry_bounds: np.ndarray = field(default_factory=lambda: GEOMETRY_BOUNDS.copy())
    response_mean: np.ndarray = field(default_factory=lambda: np.zeros(2))
    response_covariance: np.ndarray = field(default_factory=lambda: np.diag(PRIOR_SD ** 2))
    sensor_sd_at_015: float = .035
    target_detect_probability: float = .80
    clutter_probability: float = .10
    present_to_lost: float = .03
    lost_to_present: float = .70
    center_noise_sd_at_015: float = .0005
    axis_noise_sd_at_015: float = .03
    clutter_quality_mean: float = .60
    clutter_quality_sd: float = .12
    clutter_center_halfwidth: float = .022
    clutter_axis_halfwidth: float = .35
    quality_threshold: float = QUALITY_THRESHOLD
    center_tolerance_m: float = CENTER_TOLERANCE_M
    axis_tolerance_rad: float = AXIS_TOLERANCE_RAD
    public_confidence_threshold: float = .65
    public_quality_threshold: float = QUALITY_THRESHOLD
    particle_count: int = N_GEOMETRY
    particle_seed: int = 20261111
    max_components: int = MAX_COMPONENTS
    # SENSOR_V2 keeps the common Sobol support fixed.  ESS is reported as a
    # diagnostic; resampling without a protocol-approved move step would
    # duplicate geometry particles and change the declared quadrature.
    resample_threshold: float = 0.
    platt_a: float = 1.
    platt_b: float = 0.
    risk_gate: float = .90

    def __post_init__(self):
        bounds = np.asarray(self.geometry_bounds, float)
        mean = np.asarray(self.response_mean, float).reshape(2)
        covariance = np.asarray(self.response_covariance, float).reshape(2, 2)
        if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("invalid geometry_bounds")
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
            raise ValueError("invalid response prior")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.:
            raise ValueError("response_covariance must be positive definite")
        if int(self.particle_count) != self.particle_count or self.particle_count < 8:
            raise ValueError("particle_count must be an integer >=8")
        if int(self.max_components) != self.max_components or not 1 <= self.max_components <= 4:
            raise ValueError("max_components must be in [1,4]")
        if any(not 0. <= float(p) <= 1. for p in (self.target_detect_probability,
                                                   self.clutter_probability,
                                                   self.present_to_lost,
                                                   self.lost_to_present)):
            raise ValueError("detector/HMM probabilities must lie in [0,1]")
        if min(self.sensor_sd_at_015, self.center_noise_sd_at_015,
               self.axis_noise_sd_at_015, self.clutter_quality_sd) <= 0.:
            raise ValueError("noise scales must be positive")
        if self.platt_a < 0. or not 0. < self.risk_gate < 1.:
            raise ValueError("invalid calibration/risk parameters")
        object.__setattr__(self, "geometry_bounds", bounds.copy())
        object.__setattr__(self, "response_mean", mean.copy())
        object.__setattr__(self, "response_covariance", covariance.copy())

    def sensor_variance(self, action):
        return float(self.sensor_sd_at_015 ** 2 * (.15 / float(action.duration_s)))

    def center_variance(self, action):
        return float(self.center_noise_sd_at_015 ** 2 * (.15 / float(action.duration_s)))

    def axis_variance(self, action):
        return float(self.axis_noise_sd_at_015 ** 2 * (.15 / float(action.duration_s)))

    def to_dict(self):
        result = dict(self.__dict__)
        for key in ("geometry_bounds", "response_mean", "response_covariance"):
            result[key] = getattr(self, key).tolist()
        return result

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class PublicMixtureComponent:
    weight: float
    quality_mean: float
    quality_variance: float
    center_mean: float
    center_variance: float
    axis_mean: float
    axis_variance: float
    # Public law marker only; it does not identify target/clutter association.
    # The planner's sampler uses this to preserve SENSOR_V2's uniform center
    # and axis law for the corresponding mixture component.
    feature_distribution: str = "gaussian"


@dataclass(frozen=True)
class PacketPrediction:
    p_candidate: float
    p_missing: float
    components: tuple[PublicMixtureComponent, ...]
    particle_ess: float
    support_fraction: float
    p_public_accept: float = 0.0
    p_correct_accept: float = 0.0
    p_false_accept: float = 0.0


@dataclass(frozen=True)
class UpdateResult:
    applied: bool
    duplicate: bool
    log_evidence: float
    particle_ess: float
    component_count: int
    resampled: bool = False


@dataclass(frozen=True)
class CompleteOption:
    """One-window fresh option: first terminal plus at most one retry."""
    actions: tuple[ProbeAction, ...]
    max_retries: int = 2
    gh_order: int = 33
    qmc_seed: int = 918273  # retained for manifest compatibility
    require_detected_for_success: bool = False

    def __post_init__(self):
        if not self.actions or not 1 <= int(self.max_retries) <= 2:
            raise ValueError("complete option needs one or two attempts")
        if len(self.actions) < self.max_retries:
            raise ValueError("actions must include all allowed retries")
        if len({action.window for action in self.actions[:self.max_retries]}) != 1:
            raise ValueError("complete option retries must stay in one window")
        if int(self.gh_order) != self.gh_order or self.gh_order < 5:
            raise ValueError("gh_order must be >=5")


@dataclass(frozen=True)
class CompletePrediction:
    raw_complete_p: float
    calibrated_p: float
    risk_score: float
    admitted: bool
    p_target_identity: float
    p_geometry_gate: float
    p_quality_gate: float
    p_joint_first_attempt: float
    p_false_early_stop: float
    p_public_success_stop: float
    quadrature_order: int
    particle_ess: float
    calibration_interval: tuple[float, float] | None = None
    p_ever_qualified: float = 0.0


def _sobol_geometry(params):
    count = int(params.particle_count)
    sampler = qmc.Sobol(d=4, scramble=True, seed=int(params.particle_seed))
    if count & (count - 1) == 0:
        unit = sampler.random_base2(int(math.log2(count)))
    else:
        unit = sampler.random(count)
    return params.geometry_bounds[:, 0] + unit * (params.geometry_bounds[:, 1] - params.geometry_bounds[:, 0])


def _geometry_terms(geometry, action):
    x, y = map(float, action.xy_m)
    offset, slope, curvature, depth = np.asarray(geometry).T
    target_y = .018 + offset + slope * x + .5 * curvature * x * x
    beam_y = y - depth * math.tan(float(action.tilt_rad))
    center = beam_y - target_y
    axis = _angle_difference(float(action.tilt_rad), np.arctan(slope + curvature * x))
    support = np.abs(center) <= FORK_SUPPORT_M
    quality = .64 - 11. * np.minimum(np.abs(center), FORK_SUPPORT_M) - .38 * np.abs(axis)
    gate = (np.abs(center) <= CENTER_TOLERANCE_M) & (np.abs(axis) <= AXIS_TOLERANCE_RAD)
    return center, axis, quality, support, gate


def _inverse_2x2(matrix):
    """Analytic inverse for the tiny SPD matrices used by response mixtures."""
    matrix = np.asarray(matrix)
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    determinant = a * d - b * c
    bad = (~np.isfinite(determinant)) | (determinant <= 1.e-14)
    inverse = np.empty_like(matrix)
    inverse[..., 0, 0] = d / determinant
    inverse[..., 0, 1] = -b / determinant
    inverse[..., 1, 0] = -c / determinant
    inverse[..., 1, 1] = a / determinant
    if np.any(bad):
        # This path is only for numerical edge cases; normal 2x2 updates use
        # the analytic inverse and avoid thousands of tiny LAPACK calls.
        if matrix.ndim == 2:
            inverse = np.linalg.inv(matrix + np.eye(2) * 1.e-12)
        else:
            for index in np.argwhere(np.asarray(bad)):
                index = tuple(int(value) for value in index)
                inverse[index] = np.linalg.inv(matrix[index] + np.eye(2) * 1.e-12)
    return inverse


def _response_stats(precision, information):
    precision = np.asarray(precision)
    covariance = _inverse_2x2(precision)
    mean = np.einsum("...ij,...j->...i", covariance, information)
    return mean, covariance


def _regularize_covariance(covariance):
    """Keep moment-merged 2-D covariances numerically PSD."""
    covariance = .5 * (np.asarray(covariance) + np.asarray(covariance).swapaxes(-1, -2))
    if covariance.ndim == 2:
        determinant = float(covariance[0, 0] * covariance[1, 1] - covariance[0, 1] * covariance[1, 0])
        scale = float(np.max(np.abs(covariance)))
        if np.isfinite(determinant) and determinant > max(1.e-24, scale * scale * 1.e-12):
            return covariance
        values, vectors = np.linalg.eigh(covariance)
        floor = max(1.e-12, float(np.max(np.abs(values))) * 1.e-12)
        return (vectors * np.maximum(values, floor)) @ vectors.T
    result = np.empty_like(covariance)
    for index in range(len(covariance)):
        matrix = covariance[index]
        determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
        scale = float(np.max(np.abs(matrix)))
        if np.isfinite(determinant) and determinant > max(1.e-24, scale * scale * 1.e-12):
            result[index] = matrix
        else:
            result[index] = _regularize_covariance(matrix)
    return result


def _sym_kl(pa, ia, pb, ib):
    ma, ca = _response_stats(pa[None], ia[None]); mb, cb = _response_stats(pb[None], ib[None])
    ma, ca, mb, cb = ma[0], ca[0], mb[0], cb[0]
    def kl(m1, c1, m2, c2):
        inv = np.linalg.inv(c2); d = m2 - m1
        return .5 * (np.trace(inv @ c1) + d @ inv @ d - 2. +
                     math.log(max(np.linalg.det(c2), 1e-300) / max(np.linalg.det(c1), 1e-300)))
    return float(.5 * (kl(ma, ca, mb, cb) + kl(mb, cb, ma, ca)))


def _merge_reference(records, limit):
    records = [(float(w), p.copy(), i.copy()) for w, p, i in records if w > 0.]
    while len(records) > limit:
        # Batched two-dimensional Gaussian KL avoids an inverse for every
        # candidate pair (transitioning 512 particles otherwise dominates
        # every packet update).  The selected pair is still the exact minimum
        # symmetric KL under the same formula as the scalar reference above.
        precisions = np.asarray([row[1] for row in records])
        informations = np.asarray([row[2] for row in records])
        means, covariances = _response_stats(precisions, informations)
        # Moment covariances from valid Gaussian natural parameters are
        # positive definite; only invoke the eigendecomposition fallback for
        # an actual bad determinant.
        covariances = .5 * (covariances + covariances.swapaxes(-1, -2))
        determinants = covariances[..., 0, 0] * covariances[..., 1, 1] - covariances[..., 0, 1] * covariances[..., 1, 0]
        if (not np.isfinite(determinants).all()) or np.any(determinants <= 1.e-24):
            covariances = _regularize_covariance(covariances)
        inverses = _inverse_2x2(covariances)
        determinants = covariances[..., 0, 0] * covariances[..., 1, 1] - covariances[..., 0, 1] * covariances[..., 1, 0]
        logdet = np.log(np.maximum(determinants, 1e-300))
        diff = means[None, :, :] - means[:, None, :]
        trace = np.einsum("jab,iba->ij", inverses, covariances)
        quad = np.einsum("ija,jab,ijb->ij", diff, inverses, diff)
        kl = .5 * (trace + quad - 2. + logdet[None, :] - logdet[:, None])
        distance = .5 * (kl + kl.T)
        distance[np.diag_indices_from(distance)] = np.inf
        pair = np.unravel_index(int(np.argmin(distance)), distance.shape)
        i, j = pair
        wa, pa, ia = records[i]; wb, pb, ib = records[j]
        ma, ca = _response_stats(pa[None], ia[None]); mb, cb = _response_stats(pb[None], ib[None])
        ma, ca, mb, cb = ma[0], ca[0], mb[0], cb[0]
        total = wa + wb; mean = (wa * ma + wb * mb) / total
        da, db = ma - mean, mb - mean
        cov = (wa * (ca + np.outer(da, da)) + wb * (cb + np.outer(db, db))) / total
        cov = .5 * (cov + cov.T)
        determinant = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
        if (not np.isfinite(determinant)) or determinant <= 1.e-24:
            cov = _regularize_covariance(cov)
        precision = _inverse_2x2(cov)
        records[i] = (total, precision, precision @ mean)
        del records[j]
    return records


if _NUMBA_AVAILABLE:
    @_numba_njit(cache=True)
    def _merge_numba_kernel(input_weights, input_precisions, input_informations, limit):
        """Fixed-size float64 merge kernel; tie scan is row-major like argmin."""
        n = input_weights.shape[0]
        weights = input_weights.copy()
        precisions = input_precisions.copy()
        informations = input_informations.copy()
        means = np.empty((n, 2), dtype=np.float64)
        covariances = np.empty((n, 2, 2), dtype=np.float64)
        inverses = np.empty((n, 2, 2), dtype=np.float64)
        logdet = np.empty(n, dtype=np.float64)
        while n > limit:
            for r in range(n):
                p00 = precisions[r, 0, 0]
                p01 = precisions[r, 0, 1]
                p10 = precisions[r, 1, 0]
                p11 = precisions[r, 1, 1]
                determinant = p00 * p11 - p01 * p10
                if determinant > 1.e-14 and np.isfinite(determinant):
                    c00 = p11 / determinant
                    c01 = -p01 / determinant
                    c10 = -p10 / determinant
                    c11 = p00 / determinant
                else:
                    matrix = np.empty((2, 2), dtype=np.float64)
                    matrix[0, 0] = p00 + 1.e-12
                    matrix[0, 1] = p01
                    matrix[1, 0] = p10
                    matrix[1, 1] = p11 + 1.e-12
                    inverse = np.linalg.inv(matrix)
                    c00 = inverse[0, 0]
                    c01 = inverse[0, 1]
                    c10 = inverse[1, 0]
                    c11 = inverse[1, 1]
                covariances[r, 0, 0] = c00
                covariances[r, 0, 1] = c01
                covariances[r, 1, 0] = c10
                covariances[r, 1, 1] = c11
                # ``inverses`` is the inverse of the response covariance,
                # i.e. the original natural precision, not the covariance
                # just computed above.
                inverses[r, 0, 0] = p00
                inverses[r, 0, 1] = p01
                inverses[r, 1, 0] = p10
                inverses[r, 1, 1] = p11
                means[r, 0] = c00 * informations[r, 0] + c01 * informations[r, 1]
                means[r, 1] = c10 * informations[r, 0] + c11 * informations[r, 1]
                covariance_determinant = c00 * c11 - c01 * c10
                logdet[r] = math.log(max(covariance_determinant, 1.e-300))
            best_distance = math.inf
            best_i = 0
            best_j = 1
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    trace = (inverses[j, 0, 0] * covariances[i, 0, 0]
                             + inverses[j, 0, 1] * covariances[i, 1, 0]
                             + inverses[j, 1, 0] * covariances[i, 0, 1]
                             + inverses[j, 1, 1] * covariances[i, 1, 1])
                    d0 = means[j, 0] - means[i, 0]
                    d1 = means[j, 1] - means[i, 1]
                    quadratic = (d0 * (inverses[j, 0, 0] * d0 + inverses[j, 0, 1] * d1)
                                 + d1 * (inverses[j, 1, 0] * d0 + inverses[j, 1, 1] * d1))
                    kl_ij = .5 * (trace + quadratic - 2.
                                   + logdet[j] - logdet[i])
                    trace_reverse = (inverses[i, 0, 0] * covariances[j, 0, 0]
                                     + inverses[i, 0, 1] * covariances[j, 1, 0]
                                     + inverses[i, 1, 0] * covariances[j, 0, 1]
                                     + inverses[i, 1, 1] * covariances[j, 1, 1])
                    reverse_quadratic = (d0 * (inverses[i, 0, 0] * d0 + inverses[i, 0, 1] * d1)
                                         + d1 * (inverses[i, 1, 0] * d0 + inverses[i, 1, 1] * d1))
                    kl_ji = .5 * (trace_reverse + reverse_quadratic - 2.
                                   + logdet[i] - logdet[j])
                    distance = .5 * (kl_ij + kl_ji)
                    if distance < best_distance:
                        best_distance = distance
                        best_i = i
                        best_j = j
            wa = weights[best_i]
            wb = weights[best_j]
            total = wa + wb
            mean0 = (wa * means[best_i, 0] + wb * means[best_j, 0]) / total
            mean1 = (wa * means[best_i, 1] + wb * means[best_j, 1]) / total
            da0 = means[best_i, 0] - mean0
            da1 = means[best_i, 1] - mean1
            db0 = means[best_j, 0] - mean0
            db1 = means[best_j, 1] - mean1
            cov00 = (wa * (covariances[best_i, 0, 0] + da0 * da0)
                     + wb * (covariances[best_j, 0, 0] + db0 * db0)) / total
            cov01 = (wa * (covariances[best_i, 0, 1] + da0 * da1)
                     + wb * (covariances[best_j, 0, 1] + db0 * db1)) / total
            cov10 = (wa * (covariances[best_i, 1, 0] + da1 * da0)
                     + wb * (covariances[best_j, 1, 0] + db1 * db0)) / total
            cov11 = (wa * (covariances[best_i, 1, 1] + da1 * da1)
                     + wb * (covariances[best_j, 1, 1] + db1 * db1)) / total
            cov01 = .5 * (cov01 + cov10)
            cov10 = cov01
            determinant = cov00 * cov11 - cov01 * cov10
            scale = max(abs(cov00), abs(cov01), abs(cov10), abs(cov11))
            if (not np.isfinite(determinant)) or determinant <= max(1.e-24, scale * scale * 1.e-12):
                matrix = np.empty((2, 2), dtype=np.float64)
                matrix[0, 0] = cov00
                matrix[0, 1] = cov01
                matrix[1, 0] = cov10
                matrix[1, 1] = cov11
                values, vectors = np.linalg.eigh(matrix)
                floor = max(1.e-12, max(abs(values[0]), abs(values[1])) * 1.e-12)
                value0 = max(values[0], floor)
                value1 = max(values[1], floor)
                cov00 = vectors[0, 0] * vectors[0, 0] * value0 + vectors[0, 1] * vectors[0, 1] * value1
                cov01 = vectors[0, 0] * vectors[1, 0] * value0 + vectors[0, 1] * vectors[1, 1] * value1
                cov10 = cov01
                cov11 = vectors[1, 0] * vectors[1, 0] * value0 + vectors[1, 1] * vectors[1, 1] * value1
                determinant = cov00 * cov11 - cov01 * cov10
            precision00 = cov11 / determinant
            precision01 = -cov01 / determinant
            precision10 = -cov10 / determinant
            precision11 = cov00 / determinant
            if best_i < best_j:
                weights[best_i] = total
                precisions[best_i, 0, 0] = precision00
                precisions[best_i, 0, 1] = precision01
                precisions[best_i, 1, 0] = precision10
                precisions[best_i, 1, 1] = precision11
                informations[best_i, 0] = precision00 * mean0 + precision01 * mean1
                informations[best_i, 1] = precision10 * mean0 + precision11 * mean1
                for r in range(best_j, n - 1):
                    weights[r] = weights[r + 1]
                    for a in range(2):
                        for b in range(2):
                            precisions[r, a, b] = precisions[r + 1, a, b]
                    for a in range(2):
                        informations[r, a] = informations[r + 1, a]
            else:
                for r in range(best_j, n - 1):
                    weights[r] = weights[r + 1]
                    for a in range(2):
                        for b in range(2):
                            precisions[r, a, b] = precisions[r + 1, a, b]
                    for a in range(2):
                        informations[r, a] = informations[r + 1, a]
                best_i -= 1
                weights[best_i] = total
                precisions[best_i, 0, 0] = precision00
                precisions[best_i, 0, 1] = precision01
                precisions[best_i, 1, 0] = precision10
                precisions[best_i, 1, 1] = precision11
                informations[best_i, 0] = precision00 * mean0 + precision01 * mean1
                informations[best_i, 1] = precision10 * mean0 + precision11 * mean1
            n -= 1
        return weights, precisions, informations, n


    @_numba_njit(cache=True)
    def _transition_numba_kernel(mode_weights, component_weights, precisions,
                                 informations, transition, prior_precision,
                                 prior_information, limit):
        particles = mode_weights.shape[0]
        components = component_weights.shape[2]
        modes = np.zeros((particles, 2), dtype=np.float64)
        weights = np.zeros((particles, 2, components), dtype=np.float64)
        output_precisions = np.empty_like(precisions)
        output_informations = np.empty_like(informations)
        output_precisions[:, :, :, :, :] = prior_precision
        output_informations[:, :, :, :] = prior_information
        for i in range(particles):
            for new_mode in range(2):
                record_weights = np.empty(2 * components, dtype=np.float64)
                record_precisions = np.empty((2 * components, 2, 2), dtype=np.float64)
                record_informations = np.empty((2 * components, 2), dtype=np.float64)
                count = 0
                for old_mode in range(2):
                    transfer = mode_weights[i, old_mode] * transition[old_mode, new_mode]
                    for j in range(components):
                        mass = transfer * component_weights[i, old_mode, j]
                        if mass > 0.:
                            record_weights[count] = mass
                            for a in range(2):
                                for b in range(2):
                                    record_precisions[count, a, b] = precisions[i, old_mode, j, a, b]
                                record_informations[count, a] = informations[i, old_mode, j, a]
                            count += 1
                if count > 0:
                    merged_weights, merged_precisions, merged_informations, merged_count = _merge_numba_kernel(
                        record_weights[:count], record_precisions[:count], record_informations[:count], limit)
                    total = 0.
                    for j in range(merged_count):
                        total += merged_weights[j]
                    modes[i, new_mode] = total
                    if total > 0.:
                        for j in range(merged_count):
                            weights[i, new_mode, j] = merged_weights[j] / total
                            for a in range(2):
                                for b in range(2):
                                    output_precisions[i, new_mode, j, a, b] = merged_precisions[j, a, b]
                                output_informations[i, new_mode, j, a] = merged_informations[j, a]
            total_modes = modes[i, 0] + modes[i, 1]
            if total_modes > 0.:
                modes[i, 0] /= total_modes
                modes[i, 1] /= total_modes
            else:
                modes[i, 0] = 1.
                weights[i, 0, 0] = 1.
        return modes, weights, output_precisions, output_informations
else:  # pragma: no cover - exercised only when numba is absent
    _merge_numba_kernel = None
    _transition_numba_kernel = None


def _numba_merge_safe(records):
    if not _NUMBA_AVAILABLE or len(records) <= 4:
        return False
    # Preserve the NumPy reference path for subnormal mixture masses.  In
    # that regime the weighted moment calculation is intentionally allowed to
    # underflow before the reference eigendecomposition regularizer runs.  A
    # compiled reduction can associate those operations differently and
    # produce a different (but otherwise finite) precision for a component
    # whose mass is effectively zero.  Falling back here keeps the JIT path
    # bit-for-bit compatible on the declared tiny-weight boundary.
    weights = np.asarray([row[0] for row in records], dtype=np.float64)
    if np.any((weights > 0.) & (weights < np.finfo(np.float64).tiny)):
        return False
    precisions = np.asarray([row[1] for row in records], dtype=np.float64)
    informations = np.asarray([row[2] for row in records], dtype=np.float64)
    if not np.isfinite(precisions).all() or not np.isfinite(informations).all():
        return False
    determinant = precisions[..., 0, 0] * precisions[..., 1, 1] - precisions[..., 0, 1] * precisions[..., 1, 0]
    if not np.isfinite(determinant).all() or np.any(determinant <= 1.e-14):
        return False
    covariance00 = precisions[..., 1, 1] / determinant
    covariance01 = -precisions[..., 0, 1] / determinant
    covariance10 = -precisions[..., 1, 0] / determinant
    covariance11 = precisions[..., 0, 0] / determinant
    covariance_det = covariance00 * covariance11 - covariance01 * covariance10
    scale = np.maximum.reduce((np.abs(covariance00), np.abs(covariance01),
                               np.abs(covariance10), np.abs(covariance11)))
    return bool(np.isfinite(covariance_det).all() and
                np.all(covariance_det > np.maximum(1.e-24, scale * scale * 1.e-12)))


def _numba_transition_safe(mode_weights, component_weights, precisions, informations,
                           transition):
    if not _NUMBA_AVAILABLE or not np.isfinite(precisions).all() or not np.isfinite(informations).all():
        return False
    # The transition kernel forms mode/component masses internally.  Match
    # _numba_merge_safe's subnormal guard before entering compiled reductions.
    masses = (mode_weights[:, :, None] * component_weights)
    masses = masses[..., None] * transition[None, :, None, :]
    if np.any((masses > 0.) & (masses < np.finfo(np.float64).tiny)):
        return False
    determinant = precisions[..., 0, 0] * precisions[..., 1, 1] - precisions[..., 0, 1] * precisions[..., 1, 0]
    if not np.isfinite(determinant).all() or np.any(determinant <= 1.e-14):
        return False
    covariance00 = precisions[..., 1, 1] / determinant
    covariance01 = -precisions[..., 0, 1] / determinant
    covariance10 = -precisions[..., 1, 0] / determinant
    covariance11 = precisions[..., 0, 0] / determinant
    covariance_det = covariance00 * covariance11 - covariance01 * covariance10
    scale = np.maximum.reduce((np.abs(covariance00), np.abs(covariance01),
                               np.abs(covariance10), np.abs(covariance11)))
    return bool(np.isfinite(covariance_det).all() and
                np.all(covariance_det > np.maximum(1.e-24, scale * scale * 1.e-12)))


def _merge(records, limit):
    cleaned = [(float(w), p.copy(), i.copy()) for w, p, i in records if w > 0.]
    if len(cleaned) <= limit:
        return cleaned
    if _numba_merge_safe(cleaned):
        weights = np.asarray([row[0] for row in cleaned], dtype=np.float64)
        precisions = np.asarray([row[1] for row in cleaned], dtype=np.float64)
        informations = np.asarray([row[2] for row in cleaned], dtype=np.float64)
        merged_weights, merged_precisions, merged_informations, count = _merge_numba_kernel(
            weights, precisions, informations, int(limit))
        return [(float(merged_weights[j]), merged_precisions[j].copy(), merged_informations[j].copy())
                for j in range(int(count))]
    return _merge_reference(cleaned, limit)


class Belief:
    """Mutable posterior.  Call :func:`fork` before hypothetical updates."""

    def __init__(self, params=None):
        self.params = params or ModelParams()
        self.geometry = _sobol_geometry(self.params)
        p, k = len(self.geometry), self.params.max_components
        self.particle_weights = np.full(p, 1. / p)
        self.mode_weights = np.zeros((p, 2)); self.mode_weights[:, 0] = 1.
        self.component_weights = np.zeros((p, 2, k)); self.component_weights[:, :, 0] = 1.
        precision = np.linalg.inv(self.params.response_covariance)
        information = precision @ self.params.response_mean
        self.precisions = np.empty((p, 2, k, 2, 2)); self.precisions[...] = precision
        self.informations = np.empty((p, 2, k, 2)); self.informations[...] = information
        self.event_ids: set[str] = set()
        self.event_fingerprints: dict[str, str] = {}
        self.resample_count = 0
        self.last_diagnostics: dict[str, object] = {}
        self._transition_cache = None
        self._prediction_cache = {}
        self._complete_cache = {}

    def clone(self):
        other = object.__new__(Belief)
        other.params = self.params
        for name in ("geometry", "particle_weights", "mode_weights", "component_weights",
                     "precisions", "informations"):
            setattr(other, name, getattr(self, name).copy())
        other.event_ids = set(self.event_ids)
        other.event_fingerprints = dict(self.event_fingerprints)
        other.resample_count = self.resample_count
        other.last_diagnostics = dict(self.last_diagnostics)
        other._transition_cache = None if self._transition_cache is None else tuple(value.copy() for value in self._transition_cache)
        other._prediction_cache = dict(self._prediction_cache)
        other._complete_cache = dict(self._complete_cache)
        return other

    def particle_ess(self):
        return float(1. / max(np.sum(self.particle_weights ** 2), 1e-300))

    def component_count(self):
        return int(np.count_nonzero(self.component_weights > 0.))

    def posterior_moments(self):
        means, covariances = _response_stats(self.precisions.reshape(-1, 2, 2), self.informations.reshape(-1, 2))
        masses = (self.particle_weights[:, None, None] * self.mode_weights[:, :, None] * self.component_weights).reshape(-1)
        masses /= max(float(np.sum(masses)), 1e-300)
        mean = np.sum(masses[:, None] * means, axis=0)
        covariance = np.zeros((2, 2))
        for weight, local_mean, local_cov in zip(masses, means, covariances):
            d = local_mean - mean
            covariance += weight * (local_cov + np.outer(d, d))
        return mean, covariance

    def snapshot(self):
        mean, covariance = self.posterior_moments()
        return {"geometry_particles": len(self.geometry), "mean": mean.tolist(),
                "covariance": covariance.tolist(), "events": sorted(self.event_ids),
                "particle_ess": self.particle_ess(), "component_count": self.component_count(),
                "resample_count": self.resample_count,
                "approximation": "bounded_RB_geometry_association_HMM"}

    def _transition_state(self):
        """Transition the HMM and transfer each response mixture before sensing."""
        if self._transition_cache is not None:
            return self._transition_cache
        p, k = len(self.geometry), self.params.max_components
        t = np.array(((1. - self.params.present_to_lost, self.params.present_to_lost),
                      (self.params.lost_to_present, 1. - self.params.lost_to_present)))
        modes = np.zeros((p, 2)); weights = np.zeros((p, 2, k))
        precisions = np.zeros_like(self.precisions); informations = np.zeros_like(self.informations)
        prior_precision = np.linalg.inv(self.params.response_covariance)
        prior_information = prior_precision @ self.params.response_mean
        precisions[...] = prior_precision; informations[...] = prior_information
        if _numba_transition_safe(self.mode_weights, self.component_weights,
                                  self.precisions, self.informations, t):
            self._transition_cache = _transition_numba_kernel(
                self.mode_weights, self.component_weights, self.precisions,
                self.informations, t, prior_precision, prior_information, k)
            return self._transition_cache
        for i in range(p):
            for new_mode in range(2):
                records = []
                for old_mode in range(2):
                    transfer = self.mode_weights[i, old_mode] * t[old_mode, new_mode]
                    for j in range(k):
                        mass = transfer * self.component_weights[i, old_mode, j]
                        if mass > 0.:
                            records.append((mass, self.precisions[i, old_mode, j], self.informations[i, old_mode, j]))
                merged = _merge(records, k)
                modes[i, new_mode] = sum(row[0] for row in merged)
                if modes[i, new_mode] > 0.:
                    for j, (mass, precision, information) in enumerate(merged):
                        weights[i, new_mode, j] = mass / modes[i, new_mode]
                        precisions[i, new_mode, j] = precision
                        informations[i, new_mode, j] = information
            total = modes[i].sum()
            if total > 0.:
                modes[i] /= total
            else:
                modes[i, 0] = 1.; weights[i, 0, 0] = 1.
                precisions[i, 0, 0] = np.linalg.inv(self.params.response_covariance)
                informations[i, 0, 0] = precisions[i, 0, 0] @ self.params.response_mean
        self._transition_cache = (modes, weights, precisions, informations)
        return self._transition_cache

    def _likelihood_rows(self, action, packet, transitioned):
        modes, weights, precisions, informations = transitioned
        center, axis, quality, support, _ = _geometry_terms(self.geometry, action)
        h = np.array((1., float(action.tilt_rad) / YAW_SCALE_RAD))
        rv, cv, av = self.params.sensor_variance(action), self.params.center_variance(action), self.params.axis_variance(action)
        candidate = bool(packet.detected or packet.candidate_type == "candidate")
        rows = []
        for i in range(len(self.geometry)):
            for mode in range(2):
                target_p = self.params.target_detect_probability * float(support[i]) if mode == 0 else 0.
                clutter_p = (1. - target_p) * self.params.clutter_probability
                for j in range(self.params.max_components):
                    mass = self.particle_weights[i] * modes[i, mode] * weights[i, mode, j]
                    if mass <= 0.: continue
                    base = _log(mass)
                    if candidate:
                        mean, covariance = _response_stats(precisions[i, mode, j][None], informations[i, mode, j][None])
                        predicted_mean = quality[i] + h @ mean[0]
                        predicted_variance = h @ covariance[0] @ h + rv
                        target = _log(target_p)
                        clutter = _log(clutter_p)
                        if np.isfinite(packet.quality):
                            target += _log_normal(packet.quality, predicted_mean, predicted_variance)
                            clutter += _log_normal(packet.quality, self.params.clutter_quality_mean, self.params.clutter_quality_sd ** 2)
                        if np.isfinite(packet.center_offset_m):
                            target += _log_normal(packet.center_offset_m, center[i], cv)
                            clutter += _log_uniform(packet.center_offset_m, -self.params.clutter_center_halfwidth, self.params.clutter_center_halfwidth)
                        if np.isfinite(packet.axis_angle_rad):
                            target += _log_normal(packet.axis_angle_rad, axis[i], av)
                            clutter += _log_uniform(packet.axis_angle_rad, -self.params.clutter_axis_halfwidth, self.params.clutter_axis_halfwidth)
                        if np.isfinite(packet.quality):
                            updated_precision = precisions[i, mode, j] + np.outer(h, h) / rv
                            updated_information = informations[i, mode, j] + h * (packet.quality - quality[i]) / rv
                        else:
                            updated_precision, updated_information = precisions[i, mode, j].copy(), informations[i, mode, j].copy()
                        rows.append((i, mode, base + target, updated_precision, updated_information))
                        rows.append((i, mode, base + clutter, precisions[i, mode, j].copy(), informations[i, mode, j].copy()))
                    else:
                        rows.append((i, mode, base + _log((1. - target_p) * (1. - self.params.clutter_probability)),
                                     precisions[i, mode, j].copy(), informations[i, mode, j].copy()))
        return rows

    def update(self, event_id, action, packet):
        event_id = str(event_id)
        if not event_id: raise ValueError("event_id must be non-empty")
        if getattr(packet, "event_id", event_id) != event_id:
            raise ValueError("event_id does not match packet.event_id")
        fp = _fingerprint(event_id, action, packet)
        if event_id in self.event_ids:
            if self.event_fingerprints[event_id] != fp:
                raise ValueError("event_id reused with a different action or packet")
            return UpdateResult(False, True, 0., self.particle_ess(), self.component_count())
        transitioned = self._transition_state()
        rows = self._likelihood_rows(action, packet, transitioned)
        normalized, log_evidence = _normalize_logs([row[2] for row in rows])
        p, k = len(self.geometry), self.params.max_components
        grouped = [[[] for _ in range(2)] for _ in range(p)]
        particle_weights = np.zeros(p)
        for mass, (i, mode, _, precision, information) in zip(normalized, rows):
            grouped[i][mode].append((float(mass), precision, information)); particle_weights[i] += mass
        mode_weights = np.zeros((p, 2)); component_weights = np.zeros((p, 2, k))
        precisions = np.zeros_like(self.precisions); informations = np.zeros_like(self.informations)
        prior_precision = np.linalg.inv(self.params.response_covariance)
        prior_information = prior_precision @ self.params.response_mean
        precisions[...] = prior_precision; informations[...] = prior_information
        for i in range(p):
            for mode in range(2):
                merged = _merge(grouped[i][mode], k)
                mode_weights[i, mode] = sum(row[0] for row in merged)
                if mode_weights[i, mode] > 0.:
                    for j, (mass, precision, information) in enumerate(merged):
                        component_weights[i, mode, j] = mass / mode_weights[i, mode]
                        precisions[i, mode, j] = precision; informations[i, mode, j] = information
            if mode_weights[i].sum() > 0.: mode_weights[i] /= mode_weights[i].sum()
            else: mode_weights[i, 0] = 1.; component_weights[i, 0, 0] = 1.
        total = particle_weights.sum()
        if not np.isfinite(total) or total <= 0.: raise FloatingPointError("posterior mass vanished")
        self.particle_weights = particle_weights / total
        self.mode_weights, self.component_weights = mode_weights, component_weights
        self.precisions, self.informations = precisions, informations
        self.event_ids.add(event_id); self.event_fingerprints[event_id] = fp
        self._transition_cache = None
        self._prediction_cache.clear(); self._complete_cache.clear()
        resampled = self._maybe_resample()
        self.last_diagnostics = {"event_id": event_id, "log_evidence": float(log_evidence),
                                 "particle_ess": self.particle_ess(), "resampled": resampled,
                                 "component_count": self.component_count(),
                                 "approximation": "bounded_RB_geometry_association_HMM"}
        return UpdateResult(True, False, float(log_evidence), self.particle_ess(), self.component_count(), resampled)

    condition = update

    def _maybe_resample(self):
        if self.particle_ess() >= self.params.resample_threshold * len(self.geometry): return False
        n = len(self.geometry)
        rng = np.random.default_rng(self.params.particle_seed + len(self.event_ids) * 104729 + self.resample_count)
        positions = (rng.random() + np.arange(n)) / n
        indices = np.searchsorted(np.cumsum(self.particle_weights), positions, side="right").clip(0, n - 1)
        for name in ("geometry", "mode_weights", "component_weights", "precisions", "informations"):
            setattr(self, name, getattr(self, name)[indices].copy())
        self.particle_weights = np.full(n, 1. / n); self.resample_count += 1
        self._transition_cache = None
        self._prediction_cache.clear(); self._complete_cache.clear()
        return True

    def public_predict(self, action):
        key = (action.window, tuple(action.xy_m), float(action.yaw_rad), float(action.duration_s), action.kind)
        if key in self._prediction_cache:
            return self._prediction_cache[key]
        modes, weights, precisions, informations = self._transition_state()
        center, axis, quality, support, _ = _geometry_terms(self.geometry, action)
        h = np.array((1., float(action.tilt_rad) / YAW_SCALE_RAD))
        rv, cv, av = self.params.sensor_variance(action), self.params.center_variance(action), self.params.axis_variance(action)
        components = []; target_mass = clutter_mass = 0.
        public_accept = correct_accept = false_accept = 0.
        clutter_accept = _clutter_accept_probability(self.params)
        for i in range(len(self.geometry)):
            for mode in range(2):
                target_p = self.params.target_detect_probability * float(support[i]) if mode == 0 else 0.
                clutter_p = (1. - target_p) * self.params.clutter_probability
                for j in range(self.params.max_components):
                    mass = self.particle_weights[i] * modes[i, mode] * weights[i, mode, j]
                    if mass <= 0.: continue
                    mean, covariance = _response_stats(precisions[i, mode, j][None], informations[i, mode, j][None])
                    predicted_mean = float(quality[i] + h @ mean[0])
                    predicted_variance = float(h @ covariance[0] @ h + rv)
                    real_mass, fake_mass = mass * target_p, mass * clutter_p
                    if real_mass > 0.:
                        components.append(PublicMixtureComponent(real_mass, predicted_mean, predicted_variance,
                            float(center[i]), cv, float(axis[i]), av)); target_mass += real_mass
                        quality_tail = _normal_tail(self.params.public_quality_threshold, predicted_mean, predicted_variance)
                        center_pass = _interval_prob(center[i], cv, self.params.center_tolerance_m)
                        axis_pass = _interval_prob(axis[i], av, self.params.axis_tolerance_rad)
                        target_accept = real_mass * quality_tail * center_pass * axis_pass
                        correct_accept += target_accept if abs(center[i]) <= self.params.center_tolerance_m and abs(axis[i]) <= self.params.axis_tolerance_rad else 0.
                        false_accept += target_accept if not (abs(center[i]) <= self.params.center_tolerance_m and abs(axis[i]) <= self.params.axis_tolerance_rad) else 0.
                    if fake_mass > 0.:
                        components.append(PublicMixtureComponent(fake_mass, self.params.clutter_quality_mean,
                            self.params.clutter_quality_sd ** 2, 0., self.params.clutter_center_halfwidth ** 2 / 3.,
                            0., self.params.clutter_axis_halfwidth ** 2 / 3.,
                            "uniform_center_axis")); clutter_mass += fake_mass
                        false_accept += fake_mass * clutter_accept
        candidate = target_mass + clutter_mass; missing = max(0., 1. - candidate)
        total = candidate + missing
        if not np.isfinite(total) or total <= 0.: raise FloatingPointError("non-finite public prediction mass")
        scale = 1. / total
        public_accept = correct_accept + false_accept
        public_accept *= scale; correct_accept *= scale; false_accept *= scale
        prediction = PacketPrediction(float(candidate * scale), float(missing * scale),
            tuple(PublicMixtureComponent(c.weight * scale, c.quality_mean, c.quality_variance,
                                         c.center_mean, c.center_variance, c.axis_mean, c.axis_variance,
                                         c.feature_distribution)
                  for c in components), self.particle_ess(), float(np.mean(support)),
            float(public_accept), float(correct_accept), float(false_accept))
        self._prediction_cache[key] = prediction
        return prediction

    predict = public_predict

    def batched_outcomes(self, actions: Iterable[ProbeAction]):
        return tuple(self.public_predict(action) for action in actions)

    def predict_complete_option(self, option):
        return _predict_complete_option(self, option)


def new_belief(model=None, *, params=None):
    if model is not None and params is not None: raise ValueError("pass model or params, not both")
    selected = params if params is not None else model
    if isinstance(selected, dict): selected = ModelParams.from_dict(selected)
    return Belief(selected or ModelParams())


def fork(belief):
    return belief.clone()


clone = fork


def update(belief, event_id, action, packet):
    return belief.update(event_id, action, packet)


def condition(belief, event_id, action, packet):
    return belief.update(event_id, action, packet)


def public_predict(belief, action):
    return belief.public_predict(action)


def predict(belief, action):
    return belief.public_predict(action)


def batched_outcomes(belief, actions):
    return belief.batched_outcomes(actions)


def condition_batch(belief, events):
    return tuple(belief.update(event_id, action, packet) for event_id, action, packet in events)


def _normal_tail(threshold, mean, variance):
    sd = math.sqrt(max(float(variance), 1.e-12))
    return float(ndtr((float(mean) - float(threshold)) / sd))


def _interval_prob(mean, variance, tolerance):
    sd = math.sqrt(max(float(variance), 1.e-12))
    return float(ndtr((tolerance - float(mean)) / sd) - ndtr((-tolerance - float(mean)) / sd))


def _clutter_accept_probability(params):
    quality = _normal_tail(params.public_quality_threshold, params.clutter_quality_mean,
                           params.clutter_quality_sd ** 2)
    center = min(1., params.center_tolerance_m / params.clutter_center_halfwidth)
    axis = min(1., params.axis_tolerance_rad / params.clutter_axis_halfwidth)
    return quality * center * axis


def _predict_complete_option(belief, option):
    """Shared-latent GH recursion for two same-window fresh attempts.

    For every geometry/response component and GH node, ``s`` is the correct
    public stop probability, ``f`` is an early false stop probability, and
    ``c=1-s-f`` continues to the retry.  This keeps the response latent shared
    across attempts; it is not an independent-marginal calculation.
    """
    modes, weights, precisions, informations = belief._transition_state()
    order = int(option.gh_order)
    nodes, gh_weights = np.polynomial.hermite.hermgauss(order)
    gh_weights = gh_weights / math.sqrt(math.pi)
    transition = np.array(((1. - belief.params.present_to_lost, belief.params.present_to_lost),
                           (belief.params.lost_to_present, 1. - belief.params.lost_to_present)))
    first = option.actions[0]
    second = option.actions[min(1, len(option.actions) - 1)]
    center1, axis1, quality1, support1, gate1 = _geometry_terms(belief.geometry, first)
    center2, axis2, quality2, support2, gate2 = _geometry_terms(belief.geometry, second)
    h1 = np.array((1., float(first.tilt_rad) / YAW_SCALE_RAD))
    h2 = np.array((1., float(second.tilt_rad) / YAW_SCALE_RAD))
    clutter_accept = _clutter_accept_probability(belief.params)
    raw = false = public_success = 0.; identity = geom_gate = quality_gate = joint = ever = 0.
    particle_mass = belief.particle_weights
    for i in range(len(belief.geometry)):
        for initial_mode in range(2):
            for j in range(belief.params.max_components):
                base_mass = particle_mass[i] * modes[i, initial_mode] * weights[i, initial_mode, j]
                if base_mass <= 0.: continue
                mean, covariance = _response_stats(precisions[i, initial_mode, j][None], informations[i, initial_mode, j][None])
                mean, covariance = mean[0], covariance[0]
                projected_mean = float(h1 @ mean)
                projected_variance = float(h1 @ covariance @ h1)
                projected_sd = math.sqrt(max(projected_variance, 1.e-12))
                for node, node_weight in zip(nodes, gh_weights):
                    response_projection = projected_mean + math.sqrt(2.) * projected_sd * float(node)
                    mass = float(base_mass * node_weight)
                    # ``modes`` already contains the one HMM transition before
                    # the first terminal capture (``_transition_state``).
                    # Only the retry below gets another transition.  Keeping
                    # this reference implementation in lock-step with the
                    # vectorized path is useful for the GH audit.
                    p_mode1 = np.array((1., 0.)) if initial_mode == 0 else np.array((0., 1.))
                    s1 = np.zeros(2); f1 = np.zeros(2); q1 = np.zeros(2)
                    s2 = np.zeros(2); f2 = np.zeros(2); q2 = np.zeros(2)
                    for mode in range(2):
                        target_p1 = belief.params.target_detect_probability * float(support1[i]) if mode == 0 else 0.
                        target_tail1 = _normal_tail(belief.params.quality_threshold,
                            quality1[i] + response_projection, belief.params.sensor_variance(first))
                        public_tail1 = _normal_tail(belief.params.public_quality_threshold,
                            quality1[i] + response_projection, belief.params.sensor_variance(first))
                        noisy_center1 = _interval_prob(center1[i], belief.params.center_variance(first), belief.params.center_tolerance_m)
                        noisy_axis1 = _interval_prob(axis1[i], belief.params.axis_variance(first), belief.params.axis_tolerance_rad)
                        target_public1 = target_p1 * public_tail1 * noisy_center1 * noisy_axis1
                        correct1 = target_public1 if gate1[i] else 0.
                        clutter1 = (1. - target_p1) * belief.params.clutter_probability * clutter_accept
                        s1[mode], f1[mode], q1[mode] = correct1, max(0., target_public1 - correct1) + clutter1, target_tail1
                        target_p2 = belief.params.target_detect_probability * float(support2[i]) if mode == 0 else 0.
                        if np.allclose(h1, h2):
                            response2 = response_projection
                        else:
                            response2 = float(h2 @ mean + math.sqrt(max(h2 @ covariance @ h2, 1e-12)) * node * math.sqrt(2.))
                        target_tail2 = _normal_tail(belief.params.quality_threshold,
                            quality2[i] + response2, belief.params.sensor_variance(second))
                        public_tail2 = _normal_tail(belief.params.public_quality_threshold,
                            quality2[i] + response2, belief.params.sensor_variance(second))
                        noisy_center2 = _interval_prob(center2[i], belief.params.center_variance(second), belief.params.center_tolerance_m)
                        noisy_axis2 = _interval_prob(axis2[i], belief.params.axis_variance(second), belief.params.axis_tolerance_rad)
                        target_public2 = target_p2 * public_tail2 * noisy_center2 * noisy_axis2
                        correct2 = target_public2 if gate2[i] else 0.
                        clutter2 = (1. - target_p2) * belief.params.clutter_probability * clutter_accept
                        s2[mode], f2[mode], q2[mode] = correct2, max(0., target_public2 - correct2) + clutter2, target_tail2
                    c1 = np.clip(1. - s1 - f1, 0., 1.)
                    c2 = np.clip(1. - s2 - f2, 0., 1.)
                    first_success = float(np.dot(p_mode1, s1))
                    first_false = float(np.dot(p_mode1, f1))
                    retry_success = 0.; retry_false = 0.
                    if int(option.max_retries) >= 2:
                        retry_success = float(c1[initial_mode] * np.dot(transition[initial_mode], s2))
                        retry_false = float(c1[initial_mode] * np.dot(transition[initial_mode], f2))
                    raw += mass * (first_success + retry_success)
                    false += mass * (first_false + retry_false)
                    public_success += mass * (first_success + retry_success)
                    target_first = float(np.dot(p_mode1, [float(support1[i]) if m == 0 else 0. for m in range(2)]))
                    geom_first = float(np.dot(p_mode1, [float(support1[i] and gate1[i]) if m == 0 else 0. for m in range(2)]))
                    qual_first = float(np.dot(p_mode1, [float(support1[i]) * q1[m] if m == 0 else 0. for m in range(2)]))
                    identity += mass * target_first; geom_gate += mass * geom_first; quality_gate += mass * qual_first
                    joint += mass * first_success
                    # True qualifying image irrespective of public detector/stop.
                    first_ever = float(np.dot(p_mode1, [float(support1[i] and gate1[i]) * q1[m] if m == 0 else 0. for m in range(2)]))
                    second_ever = 0.
                    if int(option.max_retries) >= 2:
                        second_ever = float(c1[initial_mode] * np.dot(
                            transition[initial_mode],
                            [float(support2[i] and gate2[i]) * q2[m] if m == 0 else 0. for m in range(2)]))
                    ever += mass * (first_ever + second_ever)
    raw = float(np.clip(raw, 0., 1.)); false = float(np.clip(false, 0., 1.))
    # Public stopping includes both a correct and an erroneous acceptance;
    # ``raw`` is the correct-completion probability only.
    public_success = float(np.clip(raw + false, 0., 1.))
    clipped = np.clip(raw, 1e-5, 1. - 1e-5); z = math.log(float(clipped / (1. - clipped)))
    calibrated = float(_sigmoid(belief.params.platt_a * z + belief.params.platt_b))
    risk = min(raw, calibrated)
    return CompletePrediction(raw, calibrated, risk, risk >= belief.params.risk_gate,
        float(identity), float(geom_gate), float(quality_gate), float(joint), false,
        public_success, order, belief.particle_ess())


def _same_pose(first, second):
    return (first.window == second.window
            and np.allclose(first.xy_m, second.xy_m, rtol=0., atol=1.e-14)
            and math.isclose(float(first.yaw_rad), float(second.yaw_rad), rel_tol=0., abs_tol=1.e-14))


def _shared_quality_moments(mean, latent_variance, noise_variance1,
                            noise_variance2, threshold, order):
    """Return E[Q1], E[Q2], E[Q1 Q2] for one shared Gaussian response.

    Q_i is the probability that the independent quality measurement for
    attempt i exceeds ``threshold`` conditional on the shared response.  For
    equal dwell the second moment uses the exact Owen-T identity.  Unequal
    dwell keeps the same one-dimensional latent integral but evaluates it with
    a higher-order deterministic GH rule (the actual terminal protocol uses
    equal dwell).
    """
    mean = np.asarray(mean, dtype=float)
    latent_variance = np.maximum(np.asarray(latent_variance, dtype=float), 0.)
    noise_variance1 = float(max(noise_variance1, 1.e-16))
    noise_variance2 = float(max(noise_variance2, 1.e-16))
    total1 = np.maximum(latent_variance + noise_variance1, 1.e-16)
    total2 = np.maximum(latent_variance + noise_variance2, 1.e-16)
    z1 = (mean - threshold) / np.sqrt(total1)
    z2 = (mean - threshold) / np.sqrt(total2)
    mu1 = ndtr(z1)
    mu2 = ndtr(z2)
    equal_noise = math.isclose(noise_variance1, noise_variance2, rel_tol=1.e-12, abs_tol=1.e-16)
    if equal_noise:
        rho_term = np.sqrt(noise_variance1 / np.maximum(2. * latent_variance + noise_variance1, 1.e-16))
        mu12 = mu1 - 2. * owens_t(z1, rho_term)
        # At zero latent variance the two measurements are independent.
        zero_latent = latent_variance <= 1.e-14 * total1
        if np.any(zero_latent):
            mu12 = np.where(zero_latent, mu1 * mu2, mu12)
    else:
        nodes, weights = np.polynomial.hermite.hermgauss(max(65, int(order)))
        weights = weights / math.sqrt(math.pi)
        mu12 = np.zeros_like(mu1)
        latent_sd = np.sqrt(2. * latent_variance)
        for node, weight in zip(nodes, weights):
            response = mean + latent_sd * float(node)
            mu12 += float(weight) * ndtr((response - threshold) / math.sqrt(noise_variance1)) * ndtr((response - threshold) / math.sqrt(noise_variance2))
        zero_latent = latent_variance <= 1.e-14 * total1
        if np.any(zero_latent):
            mu12 = np.where(zero_latent, mu1 * mu2, mu12)
    return np.clip(mu1, 0., 1.), np.clip(mu2, 0., 1.), np.clip(mu12, 0., 1.)


def _predict_complete_option_same_pose(belief, option):
    """Exact shared-latent first/second-moment recursion for one pose."""
    modes, weights, precisions, informations = belief._transition_state()
    first = option.actions[0]
    second = option.actions[min(1, len(option.actions) - 1)]
    center1, axis1, quality1, support1, gate1 = _geometry_terms(belief.geometry, first)
    center2, axis2, quality2, support2, gate2 = _geometry_terms(belief.geometry, second)
    h = np.array((1., float(first.tilt_rad) / YAW_SCALE_RAD))
    particle_index, mode_index, component_index = np.indices(weights.shape)
    base = belief.particle_weights[:, None, None] * modes[:, :, None] * weights
    active = base > 0.
    particle_index = particle_index[active]
    mode_index = mode_index[active]
    base = base[active]
    means, covariances = _response_stats(precisions[active], informations[active])
    projected_mean = means @ h
    projected_variance = np.einsum("ni,nij,nj->n", np.broadcast_to(h, means.shape), covariances,
                                   np.broadcast_to(h, means.shape))
    mu1, mu2, mu12 = _shared_quality_moments(
        projected_mean, projected_variance, belief.params.sensor_variance(first),
        belief.params.sensor_variance(second),
        belief.params.public_quality_threshold - quality1[particle_index],
        int(option.gh_order))
    center1_n, axis1_n = center1[particle_index], axis1[particle_index]
    center2_n, axis2_n = center2[particle_index], axis2[particle_index]
    support1_n, support2_n = support1[particle_index], support2[particle_index]
    gate1_n, gate2_n = gate1[particle_index], gate2[particle_index]
    center_sd1 = math.sqrt(belief.params.center_variance(first))
    axis_sd1 = math.sqrt(belief.params.axis_variance(first))
    center_sd2 = math.sqrt(belief.params.center_variance(second))
    axis_sd2 = math.sqrt(belief.params.axis_variance(second))
    center_pass1 = ndtr((belief.params.center_tolerance_m - center1_n) / center_sd1) - ndtr((-belief.params.center_tolerance_m - center1_n) / center_sd1)
    axis_pass1 = ndtr((belief.params.axis_tolerance_rad - axis1_n) / axis_sd1) - ndtr((-belief.params.axis_tolerance_rad - axis1_n) / axis_sd1)
    center_pass2 = ndtr((belief.params.center_tolerance_m - center2_n) / center_sd2) - ndtr((-belief.params.center_tolerance_m - center2_n) / center_sd2)
    axis_pass2 = ndtr((belief.params.axis_tolerance_rad - axis2_n) / axis_sd2) - ndtr((-belief.params.axis_tolerance_rad - axis2_n) / axis_sd2)
    detect = belief.params.target_detect_probability
    clutter_accept = _clutter_accept_probability(belief.params)
    target1 = detect * support1_n.astype(float)
    target2 = detect * support2_n.astype(float)
    u1_present = target1 * center_pass1 * axis_pass1
    u2_present = target2 * center_pass2 * axis_pass2
    a1_present = u1_present * gate1_n.astype(float)
    a2_present = u2_present * gate2_n.astype(float)
    b1_present = (1. - target1) * belief.params.clutter_probability * clutter_accept
    b2_present = (1. - target2) * belief.params.clutter_probability * clutter_accept
    b1_lost = belief.params.clutter_probability * clutter_accept
    b2_lost = belief.params.clutter_probability * clutter_accept
    present = mode_index == 0
    u1 = np.where(present, u1_present, 0.)
    a1 = np.where(present, a1_present, 0.)
    b1 = np.where(present, b1_present, b1_lost)
    v1 = u1 - a1
    first_correct = a1 * mu1
    first_false = b1 + v1 * mu1
    f2_present = b2_present + (u2_present - a2_present) * mu2
    f2_lost = b2_lost
    if int(option.max_retries) >= 2:
        retry_correct = (1. - b1) * a2_present * mu2 - u1 * a2_present * mu12
        retry_correct *= np.where(present, 1. - belief.params.present_to_lost, belief.params.lost_to_present)
        retry_false_present = ((1. - b1) * b2_present - u1 * b2_present * mu1
                               - u1 * (u2_present - a2_present) * mu12
                               + (1. - b1) * (u2_present - a2_present) * mu2)
        retry_false_lost = ((1. - b1) * f2_lost - u1 * b2_lost * mu1)
        retry_false = np.where(present,
            (1. - belief.params.present_to_lost) * retry_false_present + belief.params.present_to_lost * retry_false_lost,
            belief.params.lost_to_present * retry_false_present + (1. - belief.params.lost_to_present) * retry_false_lost)
    else:
        retry_correct = np.zeros_like(first_correct)
        retry_false = np.zeros_like(first_false)
    raw = float(np.clip(np.sum(base * (first_correct + retry_correct)), 0., 1.))
    false = float(np.clip(np.sum(base * (first_false + retry_false)), 0., 1.))
    public_success = float(np.clip(raw + false, 0., 1.))
    identity = float(np.clip(np.sum(base * present * support1_n), 0., 1.))
    geom_gate = float(np.clip(np.sum(base * present * support1_n * gate1_n), 0., 1.))
    quality_gate = float(np.clip(np.sum(base * present * support1_n * mu1), 0., 1.))
    joint = float(np.clip(np.sum(base * first_correct), 0., 1.))
    first_true = present * support1_n * gate1_n.astype(float) * mu1
    if int(option.max_retries) >= 2:
        second_true = np.where(present,
            (1. - belief.params.present_to_lost) * ((1. - b1) * support2_n * gate2_n * mu2 - u1 * support2_n * gate2_n * mu12)
            + belief.params.present_to_lost * ((1. - b1) * 0.),
            belief.params.lost_to_present * ((1. - b1) * support2_n * gate2_n * mu2)
            + (1. - belief.params.lost_to_present) * ((1. - b1) * 0.))
    else:
        second_true = np.zeros_like(first_true)
    ever = float(np.clip(np.sum(base * (first_true + second_true)), 0., 1.))
    clipped = np.clip(raw, 1.e-5, 1. - 1.e-5)
    z = math.log(float(clipped / (1. - clipped)))
    calibrated = float(_sigmoid(belief.params.platt_a * z + belief.params.platt_b))
    risk = min(raw, calibrated)
    return CompletePrediction(raw, calibrated, risk, risk >= belief.params.risk_gate,
        identity, geom_gate, quality_gate, joint, false, public_success,
        int(option.gh_order), belief.particle_ess(), None, ever)


def _predict_complete_option_fast(belief, option):
    """Vectorized form of :func:`_predict_complete_option`.

    The public contract remains the same.  Flattening active response
    components removes the expensive Python loop over 512 particles and keeps
    the GH33 calculation deterministic and shared by all arms.
    """
    if _same_pose(option.actions[0], option.actions[min(1, len(option.actions) - 1)]):
        return _predict_complete_option_same_pose(belief, option)
    modes, weights, precisions, informations = belief._transition_state()
    order = int(option.gh_order)
    nodes, gh_weights = np.polynomial.hermite.hermgauss(order)
    gh_weights = gh_weights / math.sqrt(math.pi)
    transition = np.array(((1. - belief.params.present_to_lost, belief.params.present_to_lost),
                           (belief.params.lost_to_present, 1. - belief.params.lost_to_present)))
    first = option.actions[0]; second = option.actions[min(1, len(option.actions) - 1)]
    center1, axis1, quality1, support1, gate1 = _geometry_terms(belief.geometry, first)
    center2, axis2, quality2, support2, gate2 = _geometry_terms(belief.geometry, second)
    h1 = np.array((1., float(first.tilt_rad) / YAW_SCALE_RAD)); h2 = np.array((1., float(second.tilt_rad) / YAW_SCALE_RAD))
    particle_index, mode_index, component_index = np.indices(weights.shape)
    base = belief.particle_weights[:, None, None] * modes[:, :, None] * weights
    active = base > 0.
    particle_index = particle_index[active]; mode_index = mode_index[active]; component_index = component_index[active]
    base = base[active]
    mean, covariance = _response_stats(precisions[active], informations[active])
    h1_mean = mean @ h1; h1_variance = np.einsum("ni,nij,nj->n", np.broadcast_to(h1, mean.shape), covariance,
                                                   np.broadcast_to(h1, mean.shape))
    h2_mean = mean @ h2; h2_variance = np.einsum("ni,nij,nj->n", np.broadcast_to(h2, mean.shape), covariance,
                                                   np.broadcast_to(h2, mean.shape))
    # For different terminal poses the two projected response latents are a
    # correlated bivariate Gaussian.  A shared scalar GH node would force
    # correlation one (and can even give the wrong sign).  Conditional GH in
    # the second projection is algebraically equivalent to tensor-product
    # Gaussian quadrature while retaining the same order in each dimension.
    cross_variance = np.einsum("i,nij,j->n", h1, covariance, h2)
    same_projection = bool(np.allclose(h1, h2, rtol=0., atol=1.e-14))
    if not same_projection:
        safe_h1_variance = np.maximum(h1_variance, 1.e-12)
        conditional_slope = cross_variance / safe_h1_variance
        conditional_variance = np.maximum(h2_variance - cross_variance * conditional_slope, 0.)
    support1_n = support1[particle_index]; support2_n = support2[particle_index]
    gate1_n = gate1[particle_index]; gate2_n = gate2[particle_index]
    c1_n, a1_n, g1_n = center1[particle_index], axis1[particle_index], quality1[particle_index]
    c2_n, a2_n, g2_n = center2[particle_index], axis2[particle_index], quality2[particle_index]
    clutter_accept = _clutter_accept_probability(belief.params)
    raw = false = public_success = identity = geom_gate = quality_gate = joint = ever = 0.
    row_index = np.arange(len(base))
    for node1, node1_weight in zip(nodes, gh_weights):
        response1 = h1_mean + math.sqrt(2.) * np.sqrt(np.maximum(h1_variance, 1.e-12)) * node1
        if same_projection:
            latent_pairs = ((response1, response1, float(node1_weight)),)
        else:
            latent_pairs = tuple(
                (response1,
                 h2_mean + conditional_slope * (response1 - h1_mean)
                 + math.sqrt(2.) * np.sqrt(conditional_variance) * node2,
                 float(node1_weight * node2_weight))
                for node2, node2_weight in zip(nodes, gh_weights))
        for response1, response2, node_weight in latent_pairs:
            s1 = np.zeros((len(base), 2)); f1 = np.zeros_like(s1); q1 = np.zeros_like(s1)
            s2 = np.zeros_like(s1); f2 = np.zeros_like(s1); q2 = np.zeros_like(s1)
            for mode in range(2):
                target_p1 = belief.params.target_detect_probability * support1_n.astype(float) if mode == 0 else 0.
                target_p2 = belief.params.target_detect_probability * support2_n.astype(float) if mode == 0 else 0.
                target_tail1 = ndtr((g1_n + response1 - belief.params.quality_threshold) /
                                    math.sqrt(belief.params.sensor_variance(first)))
                public_tail1 = ndtr((g1_n + response1 - belief.params.public_quality_threshold) /
                                    math.sqrt(belief.params.sensor_variance(first)))
                center_pass1 = ndtr((belief.params.center_tolerance_m - c1_n) / math.sqrt(belief.params.center_variance(first))) - ndtr((-belief.params.center_tolerance_m - c1_n) / math.sqrt(belief.params.center_variance(first)))
                axis_pass1 = ndtr((belief.params.axis_tolerance_rad - a1_n) / math.sqrt(belief.params.axis_variance(first))) - ndtr((-belief.params.axis_tolerance_rad - a1_n) / math.sqrt(belief.params.axis_variance(first)))
                target_public1 = target_p1 * public_tail1 * center_pass1 * axis_pass1
                correct1 = target_public1 * gate1_n.astype(float)
                clutter1 = (1. - target_p1) * belief.params.clutter_probability * clutter_accept
                s1[:, mode] = correct1; f1[:, mode] = np.maximum(0., target_public1 - correct1) + clutter1; q1[:, mode] = target_tail1
                target_tail2 = ndtr((g2_n + response2 - belief.params.quality_threshold) /
                                    math.sqrt(belief.params.sensor_variance(second)))
                public_tail2 = ndtr((g2_n + response2 - belief.params.public_quality_threshold) /
                                    math.sqrt(belief.params.sensor_variance(second)))
                center_pass2 = ndtr((belief.params.center_tolerance_m - c2_n) / math.sqrt(belief.params.center_variance(second))) - ndtr((-belief.params.center_tolerance_m - c2_n) / math.sqrt(belief.params.center_variance(second)))
                axis_pass2 = ndtr((belief.params.axis_tolerance_rad - a2_n) / math.sqrt(belief.params.axis_variance(second))) - ndtr((-belief.params.axis_tolerance_rad - a2_n) / math.sqrt(belief.params.axis_variance(second)))
                target_public2 = target_p2 * public_tail2 * center_pass2 * axis_pass2
                correct2 = target_public2 * gate2_n.astype(float)
                clutter2 = (1. - target_p2) * belief.params.clutter_probability * clutter_accept
                s2[:, mode] = correct2; f2[:, mode] = np.maximum(0., target_public2 - correct2) + clutter2; q2[:, mode] = target_tail2
            c1 = np.clip(1. - s1 - f1, 0., 1.)
            first_s = s1[row_index, mode_index]; first_f = f1[row_index, mode_index]
            if int(option.max_retries) >= 2:
                retry_s = c1[row_index, mode_index] * (transition[mode_index, 0] * s2[:, 0] +
                                                        transition[mode_index, 1] * s2[:, 1])
                retry_f = c1[row_index, mode_index] * (transition[mode_index, 0] * f2[:, 0] +
                                                        transition[mode_index, 1] * f2[:, 1])
            else:
                retry_s = np.zeros_like(first_s); retry_f = np.zeros_like(first_f)
            mass = base * float(node_weight)
            raw += float(np.sum(mass * (first_s + retry_s))); false += float(np.sum(mass * (first_f + retry_f)))
            public_success += float(np.sum(mass * (first_s + retry_s)))
            present_now = (mode_index == 0).astype(float)
            identity += float(np.sum(mass * present_now * support1_n)); geom_gate += float(np.sum(mass * present_now * support1_n * gate1_n))
            quality_gate += float(np.sum(mass * present_now * support1_n * q1[:, 0])); joint += float(np.sum(mass * first_s))
            first_true = present_now * support1_n * gate1_n * q1[:, 0]
            if int(option.max_retries) >= 2:
                second_true = (c1[row_index, mode_index] * transition[mode_index, 0] *
                               support2_n * gate2_n * q2[:, 0])
            else:
                second_true = np.zeros_like(first_true)
            ever += float(np.sum(mass * (first_true + second_true)))
    raw = float(np.clip(raw, 0., 1.)); false = float(np.clip(false, 0., 1.))
    # Public stopping includes both a correct and an erroneous acceptance;
    # ``raw`` is the correct-completion probability only.
    public_success = float(np.clip(raw + false, 0., 1.))
    clipped = np.clip(raw, 1.e-5, 1. - 1.e-5); z = math.log(float(clipped / (1. - clipped)))
    calibrated = float(_sigmoid(belief.params.platt_a * z + belief.params.platt_b)); risk = min(raw, calibrated)
    return CompletePrediction(raw, calibrated, risk, risk >= belief.params.risk_gate,
        identity, geom_gate, quality_gate, joint, false, public_success, order, belief.particle_ess(),
        None, float(np.clip(ever, 0., 1.)))


def _option_key(option):
    return (tuple((a.window, tuple(a.xy_m), float(a.yaw_rad), float(a.duration_s), a.kind)
                  for a in option.actions), int(option.max_retries), int(option.gh_order),
            bool(option.require_detected_for_success))


def _cached_complete(self, option):
    key = _option_key(option)
    if key not in self._complete_cache:
        self._complete_cache[key] = _predict_complete_option_fast(self, option)
    return self._complete_cache[key]


# The vectorized implementation is the production path; the preceding scalar
# implementation remains as a transparent reference for debugging.
Belief.predict_complete_option = _cached_complete


def predict_complete_option(belief, option):
    return _predict_complete_option_fast(belief, option)


__all__ = ["Belief", "CompleteOption", "CompletePrediction", "GEOMETRY_BOUNDS",
           "ModelParams", "PacketPrediction", "PublicMixtureComponent", "UpdateResult",
           "batched_outcomes", "clone", "condition", "condition_batch", "fork",
           "new_belief", "predict", "predict_complete_option", "public_predict", "update"]

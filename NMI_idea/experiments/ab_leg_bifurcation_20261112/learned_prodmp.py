#!/usr/bin/env python3
"""A small learned 2-D ProDMP motion primitive for the leg bifurcation study.

The module is deliberately independent from the controller.  A path is
represented in an endpoint-normalised *physical tangential chart*: the first
coordinate is distance along the start-to-goal tangent divided by the step
length, and the second is signed cross-track distance divided by that same
length.  The chart makes the learned weights reusable for arbitrary millimetre
start/goal pairs while retaining a real, signed sweep rather than reducing the
primitive to endpoint interpolation.

The trajectory primitive is the integral-basis form of a critically damped
ProDMP.  ``_IntegralBasis`` and ``_homogeneous`` are loaded from the proven
observation-stress implementation when available.  A small local fallback is
kept so that this audit artifact remains runnable in isolation.

``LearnedProDMP.fit_default(seed=...)`` performs actual constrained weight
fitting on analytic demonstrations.  Train and test trajectories are sampled
from independent analytic shape parameters; the fitted basis is not used to
generate either split.  The default public interface is::

    model = LearnedProDMP.fit_default(seed=20261112)
    xy_mm = model.path(start_xy_mm, goal_xy_mm, skill="advance", samples=21)

``learned=False`` gives the endpoint-only ProDMP baseline used by the audit.
The module's CLI writes compact ``prodmp_*.npz/json`` artifacts beside this
file; ``prodmp_audit.py`` independently checks those artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REFERENCE_MOTION = (
    Path("/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments")
    / "ab_observation_stress_20261111"
    / "motion.py"
)

LAMBDA = 5.0
RBF_CENTERS = np.asarray([0.12, 0.36, 0.64, 0.88], dtype=float)
RBF_WIDTHS = np.full(4, 55.0, dtype=float)
PHASE_GATE_DECAY = 3.0
SKILLS = ("advance", "sweep_left", "sweep_right")
DEFAULT_SEED = 20261112
DEFAULT_TRAIN_PER_SKILL = 12
DEFAULT_TEST_PER_SKILL = 8
DEFAULT_TRAIN_PHASES = 81
DEFAULT_TEST_PHASES = 97
REFERENCE_START_MM = np.asarray([25.0, -12.0], dtype=float)
REFERENCE_GOAL_MM = np.asarray([105.0, 18.0], dtype=float)


def _load_reference_basis():
    """Load the integrated basis without making the neighbouring experiment a package."""
    try:
        if not REFERENCE_MOTION.is_file():
            raise FileNotFoundError(REFERENCE_MOTION)
        module_name = "_observation_stress_motion_for_learned_prodmp"
        spec = importlib.util.spec_from_file_location(module_name, REFERENCE_MOTION)
        if spec is None or spec.loader is None:
            raise ImportError("could not create a module spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module._IntegralBasis, module._homogeneous, "reference"
    except Exception:
        return None, None, "fallback"


_ReferenceIntegralBasis, _reference_homogeneous, BASIS_SOURCE = _load_reference_basis()


def _phi(s: np.ndarray | float) -> np.ndarray:
    ss = np.asarray(s, dtype=float).reshape(-1)
    ss = np.clip(ss, 0.0, 1.0)
    psi = np.exp(-RBF_WIDTHS[None, :] * (ss[:, None] - RBF_CENTERS[None, :]) ** 2)
    psi /= np.maximum(psi.sum(axis=1, keepdims=True), 1.0e-14)
    return psi * np.exp(-PHASE_GATE_DECAY * ss[:, None])


class _FallbackIntegralBasis:
    """Numerically identical interface to the integrated observation basis."""

    def __init__(self, *, lam: float = LAMBDA, quadrature_order: int = 64):
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lambda must be finite and positive")
        if int(quadrature_order) < 16:
            raise ValueError("quadrature_order must be at least 16")
        self.lam = float(lam)
        self.order = int(quadrature_order)
        nodes, weights = np.polynomial.legendre.leggauss(self.order)
        self.nodes = (nodes + 1.0) / 2.0
        self.weights = weights / 2.0

    def evaluate(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ss = np.asarray(s, dtype=float).reshape(-1)
        if not np.isfinite(ss).all() or np.any((ss < -1.0e-12) | (ss > 1.0 + 1.0e-12)):
            raise ValueError("phase must lie in [0, 1]")
        ss = np.clip(ss, 0.0, 1.0)
        out_b = np.zeros((ss.size, 4), dtype=float)
        out_bp = np.zeros((ss.size, 4), dtype=float)
        nz = ss > 0.0
        if np.any(nz):
            s_nz = ss[nz]
            u = s_nz[:, None] * self.nodes[None, :]
            delta = s_nz[:, None] - u
            kernel = np.exp(-self.lam * delta)
            phi = _phi(u.reshape(-1)).reshape(s_nz.size, self.order, 4)
            quadrature = self.weights[None, :, None] * phi
            out_b[nz] = s_nz[:, None] * np.sum(
                quadrature * delta[:, :, None] * kernel[:, :, None], axis=1
            )
            out_bp[nz] = s_nz[:, None] * np.sum(
                quadrature
                * (1.0 - self.lam * delta)[:, :, None]
                * kernel[:, :, None],
                axis=1,
            )
        return out_b, out_bp


if _ReferenceIntegralBasis is None:
    _IntegralBasis = _FallbackIntegralBasis
else:
    _IntegralBasis = _ReferenceIntegralBasis


def _homogeneous(
    s: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    start_velocity_normalized: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback critically damped homogeneous solution and phase derivative."""
    if _reference_homogeneous is not None:
        return _reference_homogeneous(s, start, goal, start_velocity_normalized, lam)
    ss = np.asarray(s, dtype=float).reshape(-1)
    delta = np.asarray(start, dtype=float) - np.asarray(goal, dtype=float)
    e = np.exp(-lam * ss)[:, None]
    eta = np.asarray(goal, dtype=float)[None, :] + e * (
        (1.0 + lam * ss[:, None]) * delta[None, :]
        + ss[:, None] * np.asarray(start_velocity_normalized, dtype=float)[None, :]
    )
    etap = e * (
        np.asarray(start_velocity_normalized, dtype=float)[None, :]
        - lam * lam * ss[:, None] * delta[None, :]
        - lam * ss[:, None] * np.asarray(start_velocity_normalized, dtype=float)[None, :]
    )
    return eta, etap


def _finite_vector(value: Iterable[float], name: str, size: int = 2) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != size or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite length-{size} vector")
    return arr


def _validate_skill(skill: str) -> str:
    value = str(skill)
    if value not in SKILLS:
        raise ValueError(f"skill must be one of {SKILLS}, got {skill!r}")
    return value


def _validate_phases(phases: Iterable[float]) -> np.ndarray:
    arr = np.asarray(phases, dtype=float).reshape(-1)
    if not arr.size or not np.isfinite(arr).all() or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("phases must be finite values in [0, 1]")
    return arr


def _endpoint_matrix(basis: _IntegralBasis) -> np.ndarray:
    b1, bp1 = basis.evaluate(1.0)
    return np.vstack((b1[0], bp1[0]))


def _endpoint_only_weights(basis: _IntegralBasis, lam: float) -> np.ndarray:
    """Return the minimum-norm endpoint position/derivative weights."""
    endpoint_b, endpoint_bp = basis.evaluate(1.0)
    eta1, etap1 = _homogeneous(
        np.asarray([1.0]),
        np.zeros(2, dtype=float),
        np.asarray([1.0, 0.0]),
        np.zeros(2, dtype=float),
        lam,
    )
    constraints = np.vstack((endpoint_b[0], endpoint_bp[0]))
    rhs = np.vstack((np.asarray([1.0, 0.0]) - eta1[0], -etap1[0]))
    weights, *_ = np.linalg.lstsq(constraints, rhs, rcond=None)
    return weights


def _constrained_fit_weights(
    phases: np.ndarray,
    chart: np.ndarray,
    chart_derivative: np.ndarray,
    basis: _IntegralBasis,
    lam: float,
    derivative_weight: float = 0.35,
) -> np.ndarray:
    """Fit weights with exact terminal position and derivative constraints.

    The equality-constrained least-squares solution is obtained in a nullspace
    of the two endpoint equations.  Thus fitting cannot trade endpoint error
    for a better interior match.
    """
    s = _validate_phases(phases)
    z = np.asarray(chart, dtype=float)
    zp = np.asarray(chart_derivative, dtype=float)
    if z.shape != (s.size, 2) or zp.shape != z.shape or not np.isfinite(z).all() or not np.isfinite(zp).all():
        raise ValueError("chart and chart_derivative must be finite (N,2) arrays")
    b, bp = basis.evaluate(s)
    eta, etap = _homogeneous(
        s,
        np.zeros(2, dtype=float),
        np.asarray([1.0, 0.0]),
        np.zeros(2, dtype=float),
        lam,
    )
    # Stack position and derivative residuals.  Derivative samples add useful
    # shape information while the nullspace still enforces the terminal rows.
    design = np.vstack((b, float(derivative_weight) * bp))
    target = np.vstack((z - eta, float(derivative_weight) * (zp - etap)))
    constraints = _endpoint_matrix(basis)
    eta1, etap1 = _homogeneous(
        np.asarray([1.0]),
        np.zeros(2, dtype=float),
        np.asarray([1.0, 0.0]),
        np.zeros(2, dtype=float),
        lam,
    )
    endpoint_target = np.vstack((np.asarray([1.0, 0.0]) - eta1[0], -etap1[0]))
    particular, *_ = np.linalg.lstsq(constraints, endpoint_target, rcond=None)
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular > max(constraints.shape) * np.finfo(float).eps * singular[0]))
    nullspace = vh[rank:].T
    if nullspace.shape[1] == 0:
        return particular
    result = np.empty((4, 2), dtype=float)
    for channel in range(2):
        coeff, *_ = np.linalg.lstsq(
            design @ nullspace,
            target[:, channel] - design @ particular[:, channel],
            rcond=None,
        )
        result[:, channel] = particular[:, channel] + nullspace @ coeff
    return result


def _smoothstep5(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quintic analytic easing used only for independent demonstrations."""
    q = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    qp = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    return q, qp


def _analytic_chart(
    skill: str,
    phases: Iterable[float],
    parameters: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Independent analytic path shape used as demonstration source.

    Parameters are ``(cross_amplitude, longitudinal_bump, phase, frequency,
    modulation)``.  The envelope ``sin(pi*s)^2`` makes both position and
    derivative satisfy the endpoint contract; the varying sinusoidal factor
    keeps train and test curves disjoint and independent from the RBF basis.
    """
    skill = _validate_skill(skill)
    s = _validate_phases(phases)
    p = np.asarray(parameters, dtype=float).reshape(-1)
    if p.size != 5 or not np.isfinite(p).all():
        raise ValueError("parameters must contain five finite analytic-shape values")
    amplitude, longitudinal_bump, phase, frequency, modulation = p
    q, qp = _smoothstep5(s)
    envelope = np.sin(np.pi * s) ** 2
    envelope_p = np.pi * np.sin(2.0 * np.pi * s)
    argument = 2.0 * np.pi * frequency * s + phase
    oscillation = np.sin(argument)
    oscillation_p = 2.0 * np.pi * frequency * np.cos(argument)
    longitudinal = q + longitudinal_bump * envelope * oscillation
    longitudinal_p = qp + longitudinal_bump * (envelope_p * oscillation + envelope * oscillation_p)
    mod_argument = 2.0 * np.pi * (0.7 + 0.25 * frequency) * s + 0.7 * phase
    modulation_shape = 1.0 + modulation * np.cos(mod_argument)
    modulation_p = -modulation * 2.0 * np.pi * (0.7 + 0.25 * frequency) * np.sin(mod_argument)
    # ``advance`` uses a gentle positive bow (rather than a straight line),
    # while the two sweep skills use the larger signed amplitudes.
    signed = 1.0 if skill in ("advance", "sweep_left") else -1.0
    cross = signed * amplitude * envelope * modulation_shape
    cross_p = signed * amplitude * (envelope_p * modulation_shape + envelope * modulation_p)
    return np.column_stack((longitudinal, cross)), np.column_stack((longitudinal_p, cross_p))


def _draw_parameters(rng: np.random.Generator, skill: str) -> np.ndarray:
    skill = _validate_skill(skill)
    # Sweep amplitudes deliberately span 15--35% of the endpoint length.  The
    # advance has a small but nonzero cross-track bow so it is not a straight
    # endpoint interpolation.
    amplitude = float(rng.uniform(0.15, 0.35) if skill != "advance" else rng.uniform(0.045, 0.085))
    longitudinal_bump = float(rng.uniform(-0.035, 0.035))
    phase = float(rng.uniform(-np.pi, np.pi))
    frequency = float(rng.integers(1, 4))
    modulation = float(rng.uniform(0.035, 0.12))
    return np.asarray([amplitude, longitudinal_bump, phase, frequency, modulation], dtype=float)


def _path_from_chart(
    phases: np.ndarray,
    weights: np.ndarray,
    basis: _IntegralBasis,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    eta, etap = _homogeneous(
        phases,
        np.zeros(2, dtype=float),
        np.asarray([1.0, 0.0]),
        np.zeros(2, dtype=float),
        lam,
    )
    b, bp = basis.evaluate(phases)
    chart = eta + b @ weights
    chart_p = etap + bp @ weights
    # Keep the declared contract exact even on a platform with tiny quadrature
    # differences.  The derivative is already analytic and remains unchanged.
    at_start = np.isclose(phases, 0.0, atol=1.0e-14)
    at_end = np.isclose(phases, 1.0, atol=1.0e-14)
    chart[at_start] = [0.0, 0.0]
    chart[at_end] = [1.0, 0.0]
    chart_p[at_start] = [0.0, 0.0]
    chart_p[at_end] = [0.0, 0.0]
    return chart, chart_p


def _map_chart_to_xy(
    chart: np.ndarray,
    chart_derivative: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    delta = goal - start
    length = float(np.linalg.norm(delta))
    if length <= 1.0e-12:
        return np.repeat(start[None, :], chart.shape[0], axis=0), np.zeros_like(chart_derivative)
    tangent = delta / length
    normal = np.asarray([-tangent[1], tangent[0]])
    frame = np.column_stack((tangent, normal))
    xy = start[None, :] + length * (chart @ frame.T)
    dxy_dphase = length * (chart_derivative @ frame.T)
    return xy, dxy_dphase


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=-1))))


@dataclass(frozen=True)
class SkillDistribution:
    """A learned skill's fitted weight distribution."""

    mean: np.ndarray
    covariance: np.ndarray
    samples: np.ndarray


class LearnedProDMP:
    """Learned 2-D ProDMP with endpoint-constrained skill distributions."""

    def __init__(
        self,
        weight_mean: Mapping[str, np.ndarray],
        weight_covariance: Mapping[str, np.ndarray],
        *,
        basis: _IntegralBasis | None = None,
        lam: float = LAMBDA,
        train_demos: Mapping[str, np.ndarray] | None = None,
        train_derivatives: Mapping[str, np.ndarray] | None = None,
        test_demos: Mapping[str, np.ndarray] | None = None,
        test_derivatives: Mapping[str, np.ndarray] | None = None,
        train_weights: Mapping[str, np.ndarray] | None = None,
        test_weights: Mapping[str, np.ndarray] | None = None,
        train_parameters: Mapping[str, np.ndarray] | None = None,
        test_parameters: Mapping[str, np.ndarray] | None = None,
        protocol: Mapping[str, object] | None = None,
        diagnostics: Mapping[str, object] | None = None,
    ):
        self.lam = float(lam)
        if not np.isfinite(self.lam) or self.lam <= 0.0:
            raise ValueError("lam must be finite and positive")
        self.basis = basis if basis is not None else _IntegralBasis(lam=self.lam, quadrature_order=64)
        self.skills = tuple(SKILLS)
        self.weight_mean = {}
        self.weight_covariance = {}
        for skill in self.skills:
            mean = np.asarray(weight_mean[skill], dtype=float)
            covariance = np.asarray(weight_covariance[skill], dtype=float)
            if mean.shape != (4, 2) or not np.isfinite(mean).all():
                raise ValueError(f"weight_mean[{skill!r}] must be finite shape (4,2)")
            if covariance.shape != (8, 8) or not np.isfinite(covariance).all():
                raise ValueError(f"weight_covariance[{skill!r}] must be finite shape (8,8)")
            self.weight_mean[skill] = mean.copy()
            self.weight_covariance[skill] = covariance.copy()
        # ``weight_cov`` is a concise alias used by audit/consumer code.
        self.weight_cov = self.weight_covariance
        self.train_demos = {k: np.asarray(v, dtype=float).copy() for k, v in (train_demos or {}).items()}
        self.train_derivatives = {k: np.asarray(v, dtype=float).copy() for k, v in (train_derivatives or {}).items()}
        self.test_demos = {k: np.asarray(v, dtype=float).copy() for k, v in (test_demos or {}).items()}
        self.test_derivatives = {k: np.asarray(v, dtype=float).copy() for k, v in (test_derivatives or {}).items()}
        self.train_weights = {k: np.asarray(v, dtype=float).copy() for k, v in (train_weights or {}).items()}
        self.test_weights = {k: np.asarray(v, dtype=float).copy() for k, v in (test_weights or {}).items()}
        self.train_parameters = {k: np.asarray(v, dtype=float).copy() for k, v in (train_parameters or {}).items()}
        self.test_parameters = {k: np.asarray(v, dtype=float).copy() for k, v in (test_parameters or {}).items()}
        self.protocol = dict(protocol or {})
        self.diagnostics = dict(diagnostics or {})
        self.endpoint_only_weights = _endpoint_only_weights(self.basis, self.lam)

    @classmethod
    def fit_default(
        cls,
        seed: int = DEFAULT_SEED,
        *,
        train_per_skill: int = DEFAULT_TRAIN_PER_SKILL,
        test_per_skill: int = DEFAULT_TEST_PER_SKILL,
        train_phases: int = DEFAULT_TRAIN_PHASES,
        test_phases: int = DEFAULT_TEST_PHASES,
        quadrature_order: int = 64,
    ) -> "LearnedProDMP":
        """Fit deterministic default skills from disjoint analytic trajectories."""
        train_per_skill = int(train_per_skill)
        test_per_skill = int(test_per_skill)
        train_phases = int(train_phases)
        test_phases = int(test_phases)
        if min(train_per_skill, test_per_skill) < 2:
            raise ValueError("at least two demonstrations are required per split")
        if min(train_phases, test_phases) < 9:
            raise ValueError("at least nine phase samples are required")
        rng = np.random.default_rng(int(seed))
        basis = _IntegralBasis(lam=LAMBDA, quadrature_order=int(quadrature_order))
        train_demos: dict[str, np.ndarray] = {}
        train_derivatives: dict[str, np.ndarray] = {}
        test_demos: dict[str, np.ndarray] = {}
        test_derivatives: dict[str, np.ndarray] = {}
        train_weights: dict[str, np.ndarray] = {}
        test_weights: dict[str, np.ndarray] = {}
        train_parameters: dict[str, np.ndarray] = {}
        test_parameters: dict[str, np.ndarray] = {}
        train_phase_grid = np.linspace(0.0, 1.0, train_phases)
        test_phase_grid = np.linspace(0.0, 1.0, test_phases)
        for skill in SKILLS:
            # Draw streams sequentially but use distinct counts/grids and keep
            # the actual parameter arrays, making the split auditable.
            tr_params = np.vstack([_draw_parameters(rng, skill) for _ in range(train_per_skill)])
            te_params = np.vstack([_draw_parameters(rng, skill) for _ in range(test_per_skill)])
            tr_chart = []
            tr_derivative = []
            tr_fit = []
            for params in tr_params:
                chart, derivative = _analytic_chart(skill, train_phase_grid, params)
                tr_chart.append(chart)
                tr_derivative.append(derivative)
                tr_fit.append(
                    _constrained_fit_weights(
                        train_phase_grid, chart, derivative, basis, LAMBDA
                    )
                )
            te_chart = []
            te_derivative = []
            te_fit = []
            for params in te_params:
                chart, derivative = _analytic_chart(skill, test_phase_grid, params)
                te_chart.append(chart)
                te_derivative.append(derivative)
                te_fit.append(
                    _constrained_fit_weights(
                        test_phase_grid, chart, derivative, basis, LAMBDA
                    )
                )
            train_parameters[skill] = tr_params
            test_parameters[skill] = te_params
            train_demos[skill] = np.asarray(tr_chart)
            train_derivatives[skill] = np.asarray(tr_derivative)
            test_demos[skill] = np.asarray(te_chart)
            test_derivatives[skill] = np.asarray(te_derivative)
            train_weights[skill] = np.asarray(tr_fit)
            test_weights[skill] = np.asarray(te_fit)
        weight_mean = {skill: np.mean(train_weights[skill], axis=0) for skill in SKILLS}
        weight_covariance = {
            skill: np.cov(train_weights[skill].reshape(train_weights[skill].shape[0], -1), rowvar=False, ddof=1)
            for skill in SKILLS
        }
        protocol = {
            "schema": "learned-prodmp-2d-v1",
            "seed": int(seed),
            "basis_source": BASIS_SOURCE,
            "lambda": LAMBDA,
            "quadrature_order": int(quadrature_order),
            "skills": list(SKILLS),
            "train_per_skill": train_per_skill,
            "test_per_skill": test_per_skill,
            "train_phases": train_phases,
            "test_phases": test_phases,
            "chart": "endpoint-normalized physical tangent/left-normal; coordinates divided by step length",
            "reference_start_mm": REFERENCE_START_MM.tolist(),
            "reference_goal_mm": REFERENCE_GOAL_MM.tolist(),
            "train_generator": "quintic smoothstep plus sin(pi*s)^2 analytic sinusoidal bow; independent RNG stream",
            "test_generator": "same analytic family with independent held-out parameters and phase grid",
            "sweep_amplitude_fraction": [0.15, 0.35],
        }
        model = cls(
            weight_mean,
            weight_covariance,
            basis=basis,
            train_demos=train_demos,
            train_derivatives=train_derivatives,
            test_demos=test_demos,
            test_derivatives=test_derivatives,
            train_weights=train_weights,
            test_weights=test_weights,
            train_parameters=train_parameters,
            test_parameters=test_parameters,
            protocol=protocol,
        )
        model.diagnostics = model._compute_diagnostics()
        return model

    @property
    def weight_distributions(self) -> dict[str, SkillDistribution]:
        return {
            skill: SkillDistribution(
                self.weight_mean[skill].copy(),
                self.weight_covariance[skill].copy(),
                self.train_weights.get(skill, np.empty((0, 4, 2))).copy(),
            )
            for skill in self.skills
        }

    def skill_distribution(self, skill: str) -> SkillDistribution:
        skill = _validate_skill(skill)
        return self.weight_distributions[skill]

    def _chart(self, phases: Iterable[float], skill: str, learned: bool) -> tuple[np.ndarray, np.ndarray]:
        skill = _validate_skill(skill)
        s = _validate_phases(phases)
        weights = self.weight_mean[skill] if learned else self.endpoint_only_weights
        return _path_from_chart(s, weights, self.basis, self.lam)

    def path(
        self,
        start_xy_mm: Iterable[float],
        goal_xy_mm: Iterable[float],
        skill: str = "advance",
        samples: int = 21,
        learned: bool = True,
    ) -> np.ndarray:
        """Return ``(samples,2)`` millimetre positions for one learned skill."""
        start = _finite_vector(start_xy_mm, "start_xy_mm")
        goal = _finite_vector(goal_xy_mm, "goal_xy_mm")
        count = int(samples)
        if count < 2:
            raise ValueError("samples must be at least 2")
        phases = np.linspace(0.0, 1.0, count)
        chart, chart_derivative = self._chart(phases, skill, bool(learned))
        xy, _ = _map_chart_to_xy(chart, chart_derivative, start, goal)
        xy[0] = start
        xy[-1] = goal
        return xy

    def path_with_velocity(
        self,
        start_xy_mm: Iterable[float],
        goal_xy_mm: Iterable[float],
        skill: str = "advance",
        samples: int = 21,
        learned: bool = True,
        duration_s: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and analytic physical velocity in mm/s."""
        start = _finite_vector(start_xy_mm, "start_xy_mm")
        goal = _finite_vector(goal_xy_mm, "goal_xy_mm")
        count = int(samples)
        tau = float(duration_s)
        if count < 2 or not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("samples must be at least 2 and duration_s positive")
        phases = np.linspace(0.0, 1.0, count)
        chart, chart_derivative = self._chart(phases, skill, bool(learned))
        xy, dxy_dphase = _map_chart_to_xy(chart, chart_derivative, start, goal)
        xy[0] = start
        xy[-1] = goal
        return xy, dxy_dphase / tau

    # Intuitive aliases for controller/audit callers.
    path_and_velocity = path_with_velocity
    velocity_path = path_with_velocity

    def velocity(
        self,
        start_xy_mm: Iterable[float],
        goal_xy_mm: Iterable[float],
        skill: str = "advance",
        samples: int = 21,
        learned: bool = True,
        duration_s: float = 1.0,
    ) -> np.ndarray:
        return self.path_with_velocity(
            start_xy_mm,
            goal_xy_mm,
            skill=skill,
            samples=samples,
            learned=learned,
            duration_s=duration_s,
        )[1]

    def sample_covariance(
        self,
        start_xy_mm: Iterable[float],
        goal_xy_mm: Iterable[float],
        skill: str = "advance",
        samples: int = 21,
    ) -> np.ndarray:
        """Return optional ``(samples,2,2)`` position covariance in mm²."""
        start = _finite_vector(start_xy_mm, "start_xy_mm")
        goal = _finite_vector(goal_xy_mm, "goal_xy_mm")
        skill = _validate_skill(skill)
        count = int(samples)
        if count < 2:
            raise ValueError("samples must be at least 2")
        phases = np.linspace(0.0, 1.0, count)
        b, _ = self.basis.evaluate(phases)
        covariance = self.weight_covariance[skill]
        delta = goal - start
        length = float(np.linalg.norm(delta))
        if length <= 1.0e-12:
            return np.zeros((count, 2, 2), dtype=float)
        tangent = delta / length
        normal = np.asarray([-tangent[1], tangent[0]])
        frame = np.column_stack((tangent, normal))
        transform = length * frame
        output = np.zeros((count, 2, 2), dtype=float)
        for i, row in enumerate(b):
            selector = np.zeros((2, 8), dtype=float)
            selector[0, 0::2] = row
            selector[1, 1::2] = row
            chart_cov = selector @ covariance @ selector.T
            output[i] = transform @ chart_cov @ transform.T
        return output

    def endpoint_only_path(
        self,
        start_xy_mm: Iterable[float],
        goal_xy_mm: Iterable[float],
        samples: int = 21,
    ) -> np.ndarray:
        return self.path(start_xy_mm, goal_xy_mm, skill="advance", samples=samples, learned=False)

    def _compute_diagnostics(self) -> dict[str, object]:
        """Compute numerical diagnostics used by the independent audit."""
        endpoint_errors = {}
        mean_errors = {}
        covariance_errors = {}
        for skill in self.skills:
            ws = self.train_weights[skill]
            b1, bp1 = self.basis.evaluate(1.0)
            constraints = np.vstack((b1[0], bp1[0]))
            eta1, etap1 = _homogeneous(
                np.asarray([1.0]), np.zeros(2), np.asarray([1.0, 0.0]), np.zeros(2), self.lam
            )
            target = np.vstack((np.asarray([1.0, 0.0]) - eta1[0], -etap1[0]))
            residuals = np.asarray([constraints @ w - target for w in ws])
            endpoint_errors[skill] = float(np.max(np.abs(residuals)))
            mean_errors[skill] = float(
                max(
                    np.max(np.abs(np.mean(ws, axis=0) - self.weight_mean[skill])),
                    np.max(np.abs(np.mean(ws.reshape(ws.shape[0], -1), axis=0) - self.weight_mean[skill].reshape(-1))),
                )
            )
            empirical_cov = np.cov(ws.reshape(ws.shape[0], -1), rowvar=False, ddof=1)
            covariance_errors[skill] = float(np.max(np.abs(empirical_cov - self.weight_covariance[skill])))

        learned_rmse = {}
        baseline_rmse = {}
        for skill in self.skills:
            phases = np.linspace(0.0, 1.0, self.test_demos[skill].shape[1])
            learned_chart, _ = self._chart(phases, skill, learned=True)
            baseline_chart, _ = self._chart(phases, skill, learned=False)
            learned_rmse[skill] = _rmse(self.test_demos[skill] - learned_chart)
            baseline_rmse[skill] = _rmse(self.test_demos[skill] - baseline_chart)
        aggregate_learned = float(np.mean(list(learned_rmse.values())))
        aggregate_baseline = float(np.mean(list(baseline_rmse.values())))

        # Analytic derivative checks in chart coordinates, then a simple
        # retiming invariance check in physical coordinates.
        derivative_errors = {}
        derivative_grid = np.linspace(0.0, 1.0, 1001)
        h = derivative_grid[1] - derivative_grid[0]
        for skill in self.skills:
            chart, derivative = self._chart(derivative_grid, skill, learned=True)
            finite = np.gradient(chart, h, axis=0, edge_order=2)
            derivative_errors[skill] = float(np.max(np.abs(finite[2:-2] - derivative[2:-2])))
        start, goal = REFERENCE_START_MM, REFERENCE_GOAL_MM
        phases = np.linspace(0.0, 1.0, 31)
        p_fast, v_fast = self.path_with_velocity(start, goal, "sweep_left", 31, True, 0.7)
        p_slow, v_slow = self.path_with_velocity(start, goal, "sweep_left", 31, True, 1.4)
        retime_position_error = float(np.max(np.abs(p_fast - p_slow)))
        # Endpoint velocities are exactly zero, so exclude those components
        # from the ratio check instead of treating 0/0 as a large error.
        nonzero_velocity = np.abs(v_slow) > 1.0e-8
        retime_velocity_ratio_error = float(
            np.max(np.abs(v_fast[nonzero_velocity] / v_slow[nonzero_velocity] - 2.0))
            if np.any(nonzero_velocity)
            else 0.0
        )
        del phases  # phases is retained above in comments as the check's grid.
        train_hashes = {skill: _array_hash(self.train_parameters[skill]) for skill in self.skills}
        test_hashes = {skill: _array_hash(self.test_parameters[skill]) for skill in self.skills}
        return {
            "endpoint_constraint_max_abs": endpoint_errors,
            "weight_mean_validation_max_abs": mean_errors,
            "weight_covariance_validation_max_abs": covariance_errors,
            "heldout_position_rmse_normalized": learned_rmse,
            "endpoint_only_position_rmse_normalized": baseline_rmse,
            "heldout_position_rmse_mm": {
                skill: float(value * np.linalg.norm(REFERENCE_GOAL_MM - REFERENCE_START_MM))
                for skill, value in learned_rmse.items()
            },
            "endpoint_only_position_rmse_mm": {
                skill: float(value * np.linalg.norm(REFERENCE_GOAL_MM - REFERENCE_START_MM))
                for skill, value in baseline_rmse.items()
            },
            "aggregate_heldout_position_rmse_normalized": aggregate_learned,
            "aggregate_endpoint_only_position_rmse_normalized": aggregate_baseline,
            "aggregate_heldout_position_rmse_mm": float(
                aggregate_learned * np.linalg.norm(REFERENCE_GOAL_MM - REFERENCE_START_MM)
            ),
            "aggregate_endpoint_only_position_rmse_mm": float(
                aggregate_baseline * np.linalg.norm(REFERENCE_GOAL_MM - REFERENCE_START_MM)
            ),
            "analytic_derivative_max_abs": derivative_errors,
            "retimed_position_max_abs_mm": retime_position_error,
            "retimed_velocity_ratio_max_abs_error": retime_velocity_ratio_error,
            "train_parameter_sha256": train_hashes,
            "test_parameter_sha256": test_hashes,
            "train_test_parameter_hashes_disjoint": all(train_hashes[k] != test_hashes[k] for k in self.skills),
            "basis_source": BASIS_SOURCE,
        }

    def save_artifacts(self, out_dir: str | Path | None = None) -> Path:
        """Save demonstrations, fitted weights, protocol and diagnostics."""
        destination = Path(out_dir) if out_dir is not None else HERE
        destination.mkdir(parents=True, exist_ok=True)
        demo_payload: dict[str, np.ndarray] = {}
        weight_payload: dict[str, np.ndarray] = {}
        for skill in self.skills:
            demo_payload[f"train_{skill}_chart"] = self.train_demos[skill]
            demo_payload[f"train_{skill}_derivative"] = self.train_derivatives[skill]
            demo_payload[f"test_{skill}_chart"] = self.test_demos[skill]
            demo_payload[f"test_{skill}_derivative"] = self.test_derivatives[skill]
            demo_payload[f"train_{skill}_parameters"] = self.train_parameters[skill]
            demo_payload[f"test_{skill}_parameters"] = self.test_parameters[skill]
            weight_payload[f"train_{skill}"] = self.train_weights[skill]
            weight_payload[f"test_{skill}"] = self.test_weights[skill]
            weight_payload[f"mean_{skill}"] = self.weight_mean[skill]
            weight_payload[f"covariance_{skill}"] = self.weight_covariance[skill]
        np.savez_compressed(destination / "prodmp_demos.npz", **demo_payload)
        np.savez_compressed(destination / "prodmp_weights.npz", **weight_payload)
        protocol = dict(self.protocol)
        protocol["artifact_files"] = ["prodmp_demos.npz", "prodmp_weights.npz", "prodmp_protocol.json", "prodmp_diagnostics.json"]
        (destination / "prodmp_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        (destination / "prodmp_diagnostics.json").write_text(json.dumps(_jsonable(self.diagnostics), indent=2, sort_keys=True) + "\n")
        return destination


def _array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float64).tobytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def load_default(seed: int = DEFAULT_SEED, **kwargs) -> LearnedProDMP:
    """Convenience loader retaining a deterministic fit API."""
    return LearnedProDMP.fit_default(seed=seed, **kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--train-per-skill", type=int, default=DEFAULT_TRAIN_PER_SKILL)
    parser.add_argument("--test-per-skill", type=int, default=DEFAULT_TEST_PER_SKILL)
    args = parser.parse_args(argv)
    model = LearnedProDMP.fit_default(
        seed=args.seed,
        train_per_skill=args.train_per_skill,
        test_per_skill=args.test_per_skill,
    )
    destination = model.save_artifacts(args.out_dir)
    print(json.dumps({"out_dir": str(destination), "diagnostics": _jsonable(model.diagnostics)}, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BASIS_SOURCE",
    "DEFAULT_SEED",
    "LearnedProDMP",
    "SKILLS",
    "SkillDistribution",
    "load_default",
]


if __name__ == "__main__":
    raise SystemExit(main())

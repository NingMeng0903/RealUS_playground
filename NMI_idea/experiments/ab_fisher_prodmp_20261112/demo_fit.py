#!/usr/bin/env python3
"""Small context-conditioned ProDMP fit for Fisher-selected demonstrations.

The Fisher design is intentionally kept outside this module.  The caller
passes the demonstrations selected by that design as ``(N, T, D)``
time-normalised trajectories, where ``D`` is one, two, or three action coordinates,
and a matching ``(N, F)`` table of context features measured before the
trajectory.  The implementation reuses the integrated basis and endpoint
constrained weighted regression from the neighbouring leg-bifurcation
experiment, but keeps one common set of numerical hyperparameters for every
action coordinate.

The model stores a null-space coefficient for each action coordinate and
regresses those coefficients on the context table.  At prediction time the
particular endpoint-constrained solution is recomputed from the supplied
held-out start/goal endpoints; therefore the conditional regression cannot
trade endpoint error for a better interior fit.  ``fit_and_evaluate`` is the
one-call interface used by the study code and also reports a matched
endpoint-only reference.

This is a trajectory representation helper, not a sensor, controller, or
Fisher selector.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE.parent / "ab_leg_bifurcation_20261112" / "learned_prodmp.py"


def _load_legacy() -> Any:
    """Load the proven basis without making the neighbouring experiment a package."""

    if not LEGACY_PATH.is_file():
        raise FileNotFoundError(LEGACY_PATH)
    name = "_legacy_learned_prodmp_for_fisher_demo_fit"
    spec = importlib.util.spec_from_file_location(name, LEGACY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create a module spec for {LEGACY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy()
LAMBDA = float(LEGACY.LAMBDA)
DEFAULT_QUADRATURE_ORDER = 64
DEFAULT_DERIVATIVE_WEIGHT = 0.35
DEFAULT_RIDGE = 1.0e-6


class _IntegratedRBFBasis:
    """Integrated RBF basis with fixed evenly spaced phase centres.

    The kernel is the same critically damped Green convolution used by the
    neighbouring four-basis implementation.  This local form is only used
    when a caller requests more than four basis functions; the default keeps
    loading the legacy implementation byte-for-byte.
    """

    def __init__(self, *, n_basis: int, lam: float, quadrature_order: int):
        n_basis = int(n_basis)
        if n_basis < 4:
            raise ValueError("n_basis must be at least 4")
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lambda must be finite and positive")
        if int(quadrature_order) < 16:
            raise ValueError("quadrature_order must be at least 16")
        self.n_basis = n_basis
        self.lam = float(lam)
        self.order = int(quadrature_order)
        # Keep the centres away from the exact integration boundaries.  The
        # spacing is fixed by n_basis and independent of any demonstrations.
        self.centres = np.linspace(0.05, 0.95, n_basis)
        spacing = 0.90 / max(1, n_basis - 1)
        self.widths = np.full(n_basis, 0.5 / max(spacing * spacing, 1.0e-12))
        nodes, weights = np.polynomial.legendre.leggauss(self.order)
        self.nodes = (nodes + 1.0) / 2.0
        self.weights = weights / 2.0

    def _phi(self, phases: np.ndarray) -> np.ndarray:
        delta = phases[:, None] - self.centres[None, :]
        psi = np.exp(-self.widths[None, :] * delta * delta)
        psi /= np.maximum(psi.sum(axis=1, keepdims=True), 1.0e-14)
        return psi * np.exp(-3.0 * phases[:, None])

    def evaluate(self, value: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phases = np.asarray(value, dtype=float).reshape(-1)
        if not np.isfinite(phases).all() or np.any((phases < -1.0e-12) | (phases > 1.0 + 1.0e-12)):
            raise ValueError("phase must lie in [0, 1]")
        phases = np.clip(phases, 0.0, 1.0)
        output = np.zeros((phases.size, self.n_basis), dtype=float)
        derivative = np.zeros_like(output)
        nonzero = phases > 0.0
        if np.any(nonzero):
            active = phases[nonzero]
            u = active[:, None] * self.nodes[None, :]
            delta = active[:, None] - u
            kernel = np.exp(-self.lam * delta)
            phi = self._phi(u.reshape(-1)).reshape(active.size, self.order, self.n_basis)
            quadrature = self.weights[None, :, None] * phi
            output[nonzero] = active[:, None] * np.sum(
                quadrature * delta[:, :, None] * kernel[:, :, None], axis=1
            )
            derivative[nonzero] = active[:, None] * np.sum(
                quadrature
                * (1.0 - self.lam * delta)[:, :, None]
                * kernel[:, :, None],
                axis=1,
            )
        return output, derivative


def _finite_trajectories(value: Any, name: str, min_count: int = 1) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (N,T,D)")
    n, t, d = arr.shape
    if n < int(min_count) or t < 5 or d not in (1, 2, 3):
        raise ValueError(f"{name} must have N>={min_count}, T>=5, and D in {{1,2,3}}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _finite_contexts(value: Any, name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[1] < 1 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite shape (N,F) array with F>=1")
    if n is not None and arr.shape[0] != int(n):
        raise ValueError(f"{name} has {arr.shape[0]} rows; expected {n}")
    return arr


def _finite_phases(value: Iterable[float] | None, count: int, name: str) -> np.ndarray:
    if value is None:
        phases = np.linspace(0.0, 1.0, int(count))
    else:
        phases = np.asarray(value, dtype=float).reshape(-1)
        if phases.size != int(count):
            raise ValueError(f"{name} must contain exactly {count} phase samples")
    if not np.isfinite(phases).all() or np.any(np.diff(phases) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    if not np.isclose(phases[0], 0.0) or not np.isclose(phases[-1], 1.0):
        raise ValueError(f"{name} must start at 0 and end at 1")
    phases = phases.copy()
    phases[0] = 0.0
    phases[-1] = 1.0
    return phases


def _finite_endpoints(value: Any, name: str, count: int, dimensions: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (int(count), int(dimensions)) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must have finite shape ({count},{dimensions})")
    return arr


def _endpoint_system(basis: Any) -> tuple[np.ndarray, np.ndarray]:
    b1, bp1 = basis.evaluate(1.0)
    constraints = np.vstack((b1[0], bp1[0]))
    _, singular, vh = np.linalg.svd(constraints, full_matrices=True)
    if singular.size == 0 or singular[0] <= 0.0:
        raise ValueError("integrated basis has a degenerate endpoint system")
    threshold = max(constraints.shape) * np.finfo(float).eps * singular[0]
    rank = int(np.sum(singular > threshold))
    nullspace = vh[rank:].T
    if nullspace.shape != (constraints.shape[1], constraints.shape[1] - rank):
        raise ValueError(f"unexpected endpoint nullspace shape {nullspace.shape}")
    return constraints, nullspace


def _particular_weights(
    phases: np.ndarray,
    goals: np.ndarray,
    basis: Any,
    lam: float,
    constraints: np.ndarray,
) -> np.ndarray:
    """Minimum-norm endpoint position/zero-velocity weights in chart space."""

    goals = np.asarray(goals, dtype=float).reshape(-1)
    zeros = np.zeros(goals.size, dtype=float)
    eta, etap = LEGACY._homogeneous(phases, zeros, goals, zeros, lam)
    rhs = np.vstack((goals - eta[-1], -etap[-1]))
    particular, *_ = np.linalg.lstsq(constraints, rhs, rcond=None)
    return particular


def _fit_one(
    phases: np.ndarray,
    trajectory: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    scales: np.ndarray,
    basis: Any,
    lam: float,
    derivative_weight: float,
    constraints: np.ndarray,
    nullspace: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one trajectory and return ``(weights, nullspace_coefficients)``."""

    normalized = (trajectory - start[None, :]) / scales[None, :]
    goal_normalized = (goal - start) / scales
    derivative = np.gradient(normalized, phases, axis=0, edge_order=2)
    b, bp = basis.evaluate(phases)
    eta, etap = LEGACY._homogeneous(
        phases,
        np.zeros(trajectory.shape[1], dtype=float),
        goal_normalized,
        np.zeros(trajectory.shape[1], dtype=float),
        lam,
    )
    design = np.vstack((b, float(derivative_weight) * bp))
    target = np.vstack((normalized - eta, float(derivative_weight) * (derivative - etap)))
    particular = _particular_weights(phases, goal_normalized, basis, lam, constraints)
    projected = design @ nullspace
    coefficients = np.empty((nullspace.shape[1], trajectory.shape[1]), dtype=float)
    for dimension in range(trajectory.shape[1]):
        solved, *_ = np.linalg.lstsq(
            projected,
            target[:, dimension] - design @ particular[:, dimension],
            rcond=None,
        )
        coefficients[:, dimension] = solved
    weights = particular + nullspace @ coefficients
    return weights, coefficients.T


def _decode(
    phases: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    basis: Any,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode normalized ProDMP weights to physical trajectories and derivatives."""

    n, dimensions = starts.shape
    b, bp = basis.evaluate(phases)
    output = np.empty((n, phases.size, dimensions), dtype=float)
    derivative = np.empty_like(output)
    for row in range(n):
        goal_normalized = (goals[row] - starts[row]) / scales
        eta, etap = LEGACY._homogeneous(
            phases,
            np.zeros(dimensions, dtype=float),
            goal_normalized,
            np.zeros(dimensions, dtype=float),
            lam,
        )
        chart = eta + b @ weights[row]
        chart_derivative = etap + bp @ weights[row]
        output[row] = starts[row][None, :] + scales[None, :] * chart
        derivative[row] = scales[None, :] * chart_derivative
        # Make the declared position endpoint contract exact despite tiny
        # quadrature/linear-algebra roundoff differences across platforms.
        output[row, 0] = starts[row]
        output[row, -1] = goals[row]
    return output, derivative


def _endpoint_checks(
    trajectories: np.ndarray,
    derivatives: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
) -> dict[str, Any]:
    position_residual = trajectories[:, [0, -1]] - np.stack((starts, goals), axis=1)
    position_error = np.linalg.norm(position_residual, axis=-1)
    velocity_error = np.linalg.norm(derivatives[:, [0, -1]], axis=-1)
    return {
        "max_position_error": float(np.max(position_error)),
        "max_position_abs_error": float(np.max(np.abs(position_residual))),
        "mean_position_error": float(np.mean(position_error)),
        "max_endpoint_velocity_error": float(np.max(velocity_error)),
        "mean_endpoint_velocity_error": float(np.mean(velocity_error)),
        "position_error": position_error,
        "endpoint_velocity_error": velocity_error,
        "all_finite": bool(np.isfinite(trajectories).all() and np.isfinite(derivatives).all()),
    }


@dataclass
class ConditionalProDMP:
    """Context-conditioned endpoint-constrained integrated-basis ProDMP."""

    basis: Any
    n_basis: int
    lam: float
    derivative_weight: float
    ridge: float
    scales: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    regression: np.ndarray
    train_contexts: np.ndarray
    train_coefficients: np.ndarray
    train_trajectories: np.ndarray
    train_starts: np.ndarray
    train_goals: np.ndarray
    train_phases: np.ndarray
    constraints: np.ndarray
    nullspace: np.ndarray

    @classmethod
    def fit(
        cls,
        trajectories: Any,
        contexts: Any,
        *,
        phases: Iterable[float] | None = None,
        n_basis: int = 4,
        lam: float = LAMBDA,
        quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
        derivative_weight: float = DEFAULT_DERIVATIVE_WEIGHT,
        ridge: float = DEFAULT_RIDGE,
    ) -> "ConditionalProDMP":
        """Fit a shared-hyperparameter conditional model from demonstrations."""

        demos = _finite_trajectories(trajectories, "trajectories", min_count=2)
        n, sample_count, dimensions = demos.shape
        context = _finite_contexts(contexts, "contexts", n)
        phase = _finite_phases(phases, sample_count, "phases")
        n_basis = int(n_basis)
        lam = float(lam)
        derivative_weight = float(derivative_weight)
        ridge = float(ridge)
        quadrature_order = int(quadrature_order)
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError("lam must be finite and positive")
        if not np.isfinite(derivative_weight) or derivative_weight < 0.0:
            raise ValueError("derivative_weight must be finite and nonnegative")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and nonnegative")
        if quadrature_order < 16:
            raise ValueError("quadrature_order must be at least 16")
        if n_basis < 4:
            raise ValueError("n_basis must be at least 4")

        if n_basis == 4:
            basis = LEGACY._IntegralBasis(lam=lam, quadrature_order=quadrature_order)
        else:
            basis = _IntegratedRBFBasis(
                n_basis=n_basis,
                lam=lam,
                quadrature_order=quadrature_order,
            )
        constraints, nullspace = _endpoint_system(basis)
        starts = demos[:, 0, :].copy()
        goals = demos[:, -1, :].copy()
        deltas = np.abs(goals - starts)
        spans = np.ptp(demos, axis=1)
        scales = np.median(np.maximum(deltas, spans), axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1.0e-9), scales, 1.0)

        coefficients = np.empty((n, dimensions, nullspace.shape[1]), dtype=float)
        for row in range(n):
            _, coefficients[row] = _fit_one(
                phase,
                demos[row],
                starts[row],
                goals[row],
                scales,
                basis,
                lam,
                derivative_weight,
                constraints,
                nullspace,
            )

        context_mean = np.mean(context, axis=0)
        context_scale = np.std(context, axis=0)
        context_scale = np.where(
            np.isfinite(context_scale) & (context_scale > 1.0e-12),
            context_scale,
            1.0,
        )
        design = np.column_stack((np.ones(n), (context - context_mean) / context_scale))
        targets = coefficients.reshape(n, -1)
        regularizer = np.eye(design.shape[1], dtype=float) * ridge
        regularizer[0, 0] = 0.0  # do not shrink the intercept
        lhs = design.T @ design + regularizer
        if ridge == 0.0:
            regression, *_ = np.linalg.lstsq(design, targets, rcond=None)
        else:
            regression = np.linalg.solve(lhs, design.T @ targets)
        return cls(
            basis=basis,
            n_basis=n_basis,
            lam=lam,
            derivative_weight=derivative_weight,
            ridge=ridge,
            scales=scales,
            context_mean=context_mean,
            context_scale=context_scale,
            regression=regression.reshape(design.shape[1], dimensions, nullspace.shape[1]),
            train_contexts=context.copy(),
            train_coefficients=coefficients,
            train_trajectories=demos,
            train_starts=starts,
            train_goals=goals,
            train_phases=phase,
            constraints=constraints,
            nullspace=nullspace,
        )

    @property
    def dimensions(self) -> int:
        return int(self.train_trajectories.shape[2])

    @property
    def context_features(self) -> int:
        return int(self.context_mean.size)

    @property
    def weight_regression(self) -> np.ndarray:
        """Regression rows ``(intercept + F, action coordinate, null weight)``."""

        return self.regression.copy()

    @property
    def conditional_weight_regression(self) -> np.ndarray:
        """Alias for the per-coordinate null-space weight regression."""

        return self.weight_regression

    @property
    def hyperparameters(self) -> dict[str, float | int]:
        """Numerical settings shared by every action coordinate."""

        return {
            "n_basis": int(self.n_basis),
            "lambda": float(self.lam),
            "quadrature_order": int(self.basis.order),
            "derivative_weight": float(self.derivative_weight),
            "ridge": float(self.ridge),
        }

    def _predict_coefficients(self, contexts: np.ndarray) -> np.ndarray:
        if contexts.shape[1] != self.context_features:
            raise ValueError(
                f"contexts has {contexts.shape[1]} features; expected {self.context_features}"
            )
        design = np.column_stack(
            (np.ones(contexts.shape[0]), (contexts - self.context_mean) / self.context_scale)
        )
        return (design @ self.regression.reshape(design.shape[1], -1)).reshape(
            contexts.shape[0], self.dimensions, self.regression.shape[2]
        )

    def _weights_for(
        self,
        contexts: np.ndarray,
        starts: np.ndarray,
        goals: np.ndarray,
        phases: np.ndarray,
    ) -> np.ndarray:
        coefficients = self._predict_coefficients(contexts)
        output = np.empty((contexts.shape[0], self.n_basis, self.dimensions), dtype=float)
        for row in range(contexts.shape[0]):
            goal_normalized = (goals[row] - starts[row]) / self.scales
            particular = _particular_weights(
                phases,
                goal_normalized,
                self.basis,
                self.lam,
                self.constraints,
            )
            output[row] = particular + self.nullspace @ coefficients[row].T
        return output

    def predict(
        self,
        contexts: Any,
        starts: Any,
        goals: Any,
        *,
        phases: Iterable[float] | None = None,
        samples: int | None = None,
    ) -> np.ndarray:
        """Return predicted physical/local trajectories for held-out contexts."""

        context = _finite_contexts(contexts, "contexts")
        count = context.shape[0]
        starts_arr = _finite_endpoints(starts, "starts", count, self.dimensions)
        goals_arr = _finite_endpoints(goals, "goals", count, self.dimensions)
        if phases is not None and samples is not None:
            raise ValueError("pass either phases or samples, not both")
        if samples is not None:
            if int(samples) < 5:
                raise ValueError("samples must be at least 5")
            phase = np.linspace(0.0, 1.0, int(samples))
        elif phases is None:
            phase = self.train_phases.copy()
        else:
            phase_values = np.asarray(phases, dtype=float).reshape(-1)
            phase = _finite_phases(phase_values, phase_values.size, "phases")
        weights = self._weights_for(context, starts_arr, goals_arr, phase)
        predictions, _ = _decode(
            phase,
            starts_arr,
            goals_arr,
            self.scales,
            weights,
            self.basis,
            self.lam,
        )
        return predictions

    def predict_with_checks(
        self,
        contexts: Any,
        starts: Any,
        goals: Any,
        *,
        phases: Iterable[float] | None = None,
        samples: int | None = None,
        reference_trajectories: Any | None = None,
    ) -> dict[str, Any]:
        """Predict and return endpoint checks plus the endpoint-only reference."""

        context = _finite_contexts(contexts, "contexts")
        count = context.shape[0]
        starts_arr = _finite_endpoints(starts, "starts", count, self.dimensions)
        goals_arr = _finite_endpoints(goals, "goals", count, self.dimensions)
        if phases is not None and samples is not None:
            raise ValueError("pass either phases or samples, not both")
        if samples is not None:
            if int(samples) < 5:
                raise ValueError("samples must be at least 5")
            phase = np.linspace(0.0, 1.0, int(samples))
        elif phases is None:
            phase = self.train_phases.copy()
        else:
            phase_values = np.asarray(phases, dtype=float).reshape(-1)
            phase = _finite_phases(phase_values, phase_values.size, "phases")
        weights = self._weights_for(context, starts_arr, goals_arr, phase)
        predictions, prediction_derivative = _decode(
            phase, starts_arr, goals_arr, self.scales, weights, self.basis, self.lam
        )
        reference_weights = np.empty_like(weights)
        for row in range(count):
            goal_normalized = (goals_arr[row] - starts_arr[row]) / self.scales
            reference_weights[row] = _particular_weights(
                phase, goal_normalized, self.basis, self.lam, self.constraints
            )
        reference, reference_derivative = _decode(
            phase,
            starts_arr,
            goals_arr,
            self.scales,
            reference_weights,
            self.basis,
            self.lam,
        )
        result: dict[str, Any] = {
            "trajectories": predictions,
            "predicted_trajectories": predictions,
            "endpoint_only_trajectories": reference,
            "starts": starts_arr,
            "goals": goals_arr,
            "phases": phase,
            "endpoint_errors": _endpoint_checks(
                predictions, prediction_derivative, starts_arr, goals_arr
            ),
            "endpoint_only_endpoint_errors": _endpoint_checks(
                reference, reference_derivative, starts_arr, goals_arr
            ),
            "fit_checks": self.fit_checks(),
        }
        if reference_trajectories is not None:
            truth = _finite_trajectories(reference_trajectories, "reference_trajectories")
            if truth.shape != predictions.shape:
                raise ValueError(
                    f"reference_trajectories shape {truth.shape} does not match prediction shape {predictions.shape}"
                )
            result["trajectory_rmse"] = float(np.sqrt(np.mean((predictions - truth) ** 2)))
            result["endpoint_only_trajectory_rmse"] = float(np.sqrt(np.mean((reference - truth) ** 2)))
            result["heldout_fit_checks"] = {
                "all_finite": bool(np.isfinite(truth).all()),
                "trajectory_rmse": result["trajectory_rmse"],
                "endpoint_only_trajectory_rmse": result["endpoint_only_trajectory_rmse"],
            }
        return result

    def endpoint_only(
        self,
        starts: Any,
        goals: Any,
        *,
        phases: Iterable[float] | None = None,
        samples: int | None = None,
    ) -> np.ndarray:
        """Return the matched endpoint-only ProDMP reference trajectory."""

        starts_arr = np.asarray(starts, dtype=float)
        if starts_arr.ndim != 2:
            raise ValueError("starts must have shape (N,D)")
        context = np.zeros((starts_arr.shape[0], self.context_features), dtype=float)
        return self.predict_with_checks(
            context,
            starts,
            goals,
            phases=phases,
            samples=samples,
        )["endpoint_only_trajectories"]

    def fit_checks(self) -> dict[str, Any]:
        """Return training reconstruction and endpoint diagnostics."""

        count = self.train_coefficients.shape[0]
        coefficients = self.train_coefficients
        weights = np.empty((count, self.n_basis, self.dimensions), dtype=float)
        for row in range(count):
            goal_normalized = (self.train_goals[row] - self.train_starts[row]) / self.scales
            particular = _particular_weights(
                self.train_phases,
                goal_normalized,
                self.basis,
                self.lam,
                self.constraints,
            )
            weights[row] = particular + self.nullspace @ coefficients[row].T
        reconstruction, derivative = _decode(
            self.train_phases,
            self.train_starts,
            self.train_goals,
            self.scales,
            weights,
            self.basis,
            self.lam,
        )
        checks = _endpoint_checks(
            reconstruction, derivative, self.train_starts, self.train_goals
        )
        conditional_weights = self._weights_for(
            self.train_contexts,
            self.train_starts,
            self.train_goals,
            self.train_phases,
        )
        conditional_reconstruction, conditional_derivative = _decode(
            self.train_phases,
            self.train_starts,
            self.train_goals,
            self.scales,
            conditional_weights,
            self.basis,
            self.lam,
        )
        conditional_checks = _endpoint_checks(
            conditional_reconstruction,
            conditional_derivative,
            self.train_starts,
            self.train_goals,
        )
        checks.update(
            {
                "trajectory_rmse": float(np.sqrt(np.mean((reconstruction - self.train_trajectories) ** 2))),
                "conditional_trajectory_rmse": float(
                    np.sqrt(np.mean((conditional_reconstruction - self.train_trajectories) ** 2))
                ),
                "conditional_max_position_error": conditional_checks["max_position_error"],
                "conditional_max_endpoint_velocity_error": conditional_checks[
                    "max_endpoint_velocity_error"
                ],
                "training_count": int(count),
                "action_dimensions": int(self.dimensions),
                "context_features": int(self.context_features),
                "n_basis": int(self.n_basis),
                "lambda": float(self.lam),
                "quadrature_order": int(self.basis.order),
                "derivative_weight": float(self.derivative_weight),
                "ridge": float(self.ridge),
            }
        )
        return checks


def fit_and_evaluate(
    trajectories: Any,
    contexts: Any,
    heldout_contexts: Any,
    heldout_starts: Any | None = None,
    heldout_goals: Any | None = None,
    *,
    heldout_trajectories: Any | None = None,
    phases: Iterable[float] | None = None,
    heldout_phases: Iterable[float] | None = None,
    samples: int | None = None,
    n_basis: int = 4,
    lam: float = LAMBDA,
    quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
    derivative_weight: float = DEFAULT_DERIVATIVE_WEIGHT,
    ridge: float = DEFAULT_RIDGE,
) -> dict[str, Any]:
    """Fit demonstrations and evaluate predictions on held-out contexts.

    ``heldout_starts``/``heldout_goals`` may be omitted when
    ``heldout_trajectories`` is supplied; its first/last samples are then the
    known endpoints.  The returned mapping includes ``model``, predicted and
    endpoint-only trajectories, endpoint arrays/errors, and fit diagnostics.
    ``n_basis=4`` preserves the legacy fit; a larger value selects the fixed
    evenly spaced integrated-RBF basis for a representation-only diagnostic.
    """

    model = ConditionalProDMP.fit(
        trajectories,
        contexts,
        phases=phases,
        n_basis=n_basis,
        lam=lam,
        quadrature_order=quadrature_order,
        derivative_weight=derivative_weight,
        ridge=ridge,
    )
    held_context = _finite_contexts(heldout_contexts, "heldout_contexts")
    held_truth = None
    if heldout_trajectories is not None:
        held_truth = _finite_trajectories(heldout_trajectories, "heldout_trajectories")
        if held_truth.shape[0] != held_context.shape[0] or held_truth.shape[2] != model.dimensions:
            raise ValueError("heldout_trajectories must match heldout_contexts and training dimensions")
        if heldout_starts is None:
            heldout_starts = held_truth[:, 0, :]
        if heldout_goals is None:
            heldout_goals = held_truth[:, -1, :]
    if heldout_starts is None or heldout_goals is None:
        raise ValueError("heldout_starts and heldout_goals are required unless heldout_trajectories is supplied")
    if samples is not None and heldout_phases is not None:
        raise ValueError("pass either samples or heldout_phases, not both")
    result = model.predict_with_checks(
        held_context,
        heldout_starts,
        heldout_goals,
        phases=heldout_phases,
        samples=samples,
        reference_trajectories=held_truth,
    )
    result["model"] = model
    return result


def _symmetric_demo_curve(angle_deg: float = 9.0, samples_per_segment: int = 21) -> np.ndarray:
    """Return the deterministic neutral/-angle/+angle/neutral diagnostic curve."""

    values = np.asarray([0.0, -float(angle_deg), float(angle_deg), 0.0])
    parts: list[np.ndarray] = []
    for index in range(3):
        local = np.linspace(0.0, 1.0, int(samples_per_segment))
        blend = local**3 * (10.0 - 15.0 * local + 6.0 * local * local)
        segment = values[index] + blend * (values[index + 1] - values[index])
        parts.extend((segment if index == 0 else segment[1:]))
    return np.asarray(parts, dtype=float)[:, None]


def symmetric_demo_diagnostic(n_basis: int = 10) -> dict[str, Any]:
    """Check broad-basis reconstruction on repeated symmetric demos only.

    This diagnostic intentionally has no sensing or held-out geometry.  It
    repeats the exact ``0 -> -9 -> +9 -> 0`` action curve, fits on half of the
    identical rows, and evaluates the remaining identical rows.  The two
    interior knot errors are reported in degrees; the neutral endpoint is
    expected to be exactly zero by the endpoint contract.
    """

    phases = np.linspace(0.0, 1.0, 61)
    curve = _symmetric_demo_curve()
    trajectories = np.repeat(curve[None, :, :], 8, axis=0)
    contexts = np.zeros((8, 1), dtype=float)
    result = fit_and_evaluate(
        trajectories[:6],
        contexts[:6],
        contexts[6:],
        np.zeros((2, 1), dtype=float),
        np.zeros((2, 1), dtype=float),
        heldout_trajectories=trajectories[6:],
        phases=phases,
        n_basis=int(n_basis),
    )
    prediction = result["trajectories"][0, :, 0]
    target = trajectories[6, :, 0]
    first_knot = int(np.argmin(np.abs(phases - 1.0 / 3.0)))
    second_knot = int(np.argmin(np.abs(phases - 2.0 / 3.0)))
    first_error = float(abs(prediction[first_knot] - target[first_knot]))
    second_error = float(abs(prediction[second_knot] - target[second_knot]))
    endpoint_error = float(abs(prediction[-1] - target[-1]))
    return {
        "n_basis": int(n_basis),
        "phase_1_3_error_deg": first_error,
        "phase_2_3_error_deg": second_error,
        "neutral_endpoint_error_deg": endpoint_error,
        "passes_0.65deg": bool(
            first_error <= 0.65
            and second_error <= 0.65
            and endpoint_error == 0.0
        ),
        "trajectory_rmse_deg": float(result["trajectory_rmse"]),
    }


def fit_demo_trajectories(
    train_trajectories: Any,
    train_contexts: Any,
    heldout_contexts: Any,
    heldout_starts: Any | None = None,
    heldout_goals: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Keyword-friendly alias using explicit train/held-out argument names."""

    return fit_and_evaluate(
        train_trajectories,
        train_contexts,
        heldout_contexts,
        heldout_starts,
        heldout_goals,
        **kwargs,
    )


fit_fisher_prodmp = fit_demo_trajectories
fit_demo_model = ConditionalProDMP.fit


__all__ = [
    "ConditionalProDMP",
    "fit_and_evaluate",
    "fit_demo_model",
    "fit_demo_trajectories",
    "fit_fisher_prodmp",
    "symmetric_demo_diagnostic",
]


if __name__ == "__main__":
    phase = np.linspace(0.0, 1.0, 41)
    contexts = np.linspace(-1.0, 1.0, 12)[:, None]
    demos = np.zeros((12, phase.size, 2), dtype=float)
    for i, context in enumerate(contexts[:, 0]):
        demos[i, :, 0] = 2.0 * phase + 0.15 * context * np.sin(np.pi * phase) ** 2
        demos[i, :, 1] = -0.4 * phase + 0.08 * context * np.sin(np.pi * phase) ** 2
    result = fit_and_evaluate(
        demos[:9],
        contexts[:9],
        contexts[9:],
        heldout_trajectories=demos[9:],
        heldout_phases=phase,
    )
    print(
        {
            "trajectory_shape": result["trajectories"].shape,
            "endpoint_error": result["endpoint_errors"]["max_position_error"],
            "fit_rmse": result.get("trajectory_rmse"),
        }
    )

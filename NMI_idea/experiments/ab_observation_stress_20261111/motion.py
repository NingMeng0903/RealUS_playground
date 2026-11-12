"""Local ProDMP motion layer for the B observation stress experiment.

This module is deliberately independent from the production controller.  The
primitive is the *integral basis* form of a critically damped ProDMP.  The
phase variable is ``s=t/tau`` and the local chart is three dimensional:
``(x_m, y_m, yaw_rad)`` in a material/body frame.  A fixed body anchor maps
that chart to a world TCP pose.  The only velocities returned to a controller
are the analytic geometric six-dimensional world twist.

For each normalized RBF basis ``phi_j`` we integrate the Green function of
``(D + lambda)^2``:

    b_j(s) = integral_0^s (s-u) exp(-lambda*(s-u)) phi_j(u) du.

The weight matrix is the minimum-norm solution that enforces the declared
terminal position and normalized velocity.  This makes the endpoint contract
explicit and avoids fitting a waypoint spline and then differentiating it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


# The reference class is optional.  Keeping a local fallback makes this file
# runnable as a standalone audit artifact, while a controller run receives the
# exact same field names as rm75_control's MotionReference.
try:  # pragma: no cover - import availability depends on the caller's path
    _RM75_ROOT = Path("/media/camp/EXT_DRIVE/RealUS_playground/rm75_control")
    if _RM75_ROOT.is_dir() and str(_RM75_ROOT) not in sys.path:
        sys.path.insert(0, str(_RM75_ROOT))
    from rm75_control.control.admittance_common.reference import MotionReference  # type: ignore
except Exception:  # pragma: no cover - exercised by isolated imports
    @dataclass(frozen=True)
    class MotionReference:  # noqa: D101 - compatibility fallback
        pose_d: np.ndarray
        vel_ff: np.ndarray
        t_ref: float = 0.0
        valid: bool = True


LAMBDA = 5.0
RBF_CENTERS = np.asarray([0.12, 0.36, 0.64, 0.88], dtype=float)
RBF_WIDTHS = np.full(4, 55.0, dtype=float)
PHASE_GATE_DECAY = 3.0
DEFAULT_TRANSLATION_SPEED_M_S = 0.02
DEFAULT_YAW_SPEED_RAD_S = 0.5
MIN_DURATION_S = 0.1


def _validate_chart(value: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite length-3 local chart (x, y, yaw)")
    return arr


def _validate_anchor(anchor: np.ndarray) -> np.ndarray:
    arr = np.asarray(anchor, dtype=float)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError("anchor must be a finite 4x4 SE(3) transform")
    if not np.allclose(arr[3], [0., 0., 0., 1.], atol=1e-8):
        raise ValueError("anchor must have homogeneous bottom row [0, 0, 0, 1]")
    R = arr[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-7) or np.linalg.det(R) < 1. - 2e-7:
        raise ValueError("anchor rotation must be a proper orthonormal matrix")
    return arr.copy()


def _phi(s: np.ndarray | float) -> np.ndarray:
    """Normalized RBFs multiplied by a phase gate.

    The output shape is ``(n, 4)`` for a vector input and ``(1, 4)`` for a
    scalar input.  The normalization is across the four RBFs before applying
    the terminal phase gate.
    """
    ss = np.asarray(s, dtype=float).reshape(-1)
    ss = np.clip(ss, 0., 1.)
    psi = np.exp(-RBF_WIDTHS[None, :] * (ss[:, None] - RBF_CENTERS[None, :]) ** 2)
    psi /= np.maximum(psi.sum(axis=1, keepdims=True), 1.e-14)
    return psi * np.exp(-PHASE_GATE_DECAY * ss[:, None])


class _IntegralBasis:
    """Evaluate Green-convolution bases with deterministic Gauss quadrature."""

    def __init__(self, *, lam: float = LAMBDA, quadrature_order: int = 64):
        if not np.isfinite(lam) or lam <= 0.:
            raise ValueError("lambda must be finite and positive")
        if int(quadrature_order) < 16:
            raise ValueError("quadrature_order must be at least 16")
        self.lam = float(lam)
        self.order = int(quadrature_order)
        nodes, weights = np.polynomial.legendre.leggauss(self.order)
        self.nodes = (nodes + 1.) / 2.
        self.weights = weights / 2.

    def evaluate(self, s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``b(s), b'(s)`` with shapes ``(n,4)``.

        At ``s=0`` the integral is exactly zero.  For nonzero ``s`` a
        Gauss-Legendre change of variables integrates on ``u in [0,s]``.
        """
        ss = np.asarray(s, dtype=float).reshape(-1)
        if not np.isfinite(ss).all() or np.any((ss < -1.e-12) | (ss > 1. + 1.e-12)):
            raise ValueError("phase must lie in [0, 1]")
        ss = np.clip(ss, 0., 1.)
        out_b = np.zeros((ss.size, 4), dtype=float)
        out_bp = np.zeros((ss.size, 4), dtype=float)
        nz = ss > 0.
        if np.any(nz):
            s_nz = ss[nz]
            u = s_nz[:, None] * self.nodes[None, :]
            delta = s_nz[:, None] - u
            kernel = np.exp(-self.lam * delta)
            phi = _phi(u.reshape(-1)).reshape(s_nz.size, self.order, 4)
            quadrature = self.weights[None, :, None] * phi
            out_b[nz] = s_nz[:, None] * np.sum(quadrature * delta[:, :, None] * kernel[:, :, None], axis=1)
            out_bp[nz] = s_nz[:, None] * np.sum(quadrature * (1. - self.lam * delta)[:, :, None] * kernel[:, :, None], axis=1)
        return out_b, out_bp


_DEFAULT_BASIS = _IntegralBasis()


def _homogeneous(s: np.ndarray, start: np.ndarray, goal: np.ndarray,
                 start_velocity_normalized: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Critically damped homogeneous solution and phase derivative."""
    ss = np.asarray(s, dtype=float).reshape(-1)
    delta = start - goal
    e = np.exp(-lam * ss)[:, None]
    eta = goal[None, :] + e * ((1. + lam * ss[:, None]) * delta[None, :] + ss[:, None] * start_velocity_normalized[None, :])
    # d/ds of ((1+lambda*s) delta + s nu) exp(-lambda*s)
    etap = e * (start_velocity_normalized[None, :] - lam * lam * ss[:, None] * delta[None, :] - lam * ss[:, None] * start_velocity_normalized[None, :])
    return eta, etap


@dataclass(frozen=True)
class LocalState:
    """A phase sample in the material chart, with derivatives in phase units."""

    phase: float
    position: np.ndarray
    derivative: np.ndarray


class ProDMPReference:
    """Original integral-basis ProDMP decoded to a world pose.

    ``start_velocity`` and ``end_velocity`` are local physical velocities in
    ``(m/s, m/s, rad/s)``.  The trajectory is evaluated with phase ``s=t/tau``
    and therefore the derivative returned to the controller is divided by
    ``tau`` before SE(3) decoding.
    """

    def __init__(self, start: Iterable[float], goal: Iterable[float], anchor: np.ndarray,
                 *, duration_s: float, start_velocity: Iterable[float] | None = None,
                 end_velocity: Iterable[float] | None = None, lam: float = LAMBDA,
                 quadrature_order: int = 64, euler_order: str = "xyz"):
        self.start = _validate_chart(start, "start")
        self.goal = _validate_chart(goal, "goal")
        self.anchor = _validate_anchor(anchor)
        self.tau = float(duration_s)
        if not np.isfinite(self.tau) or self.tau <= 0.:
            raise ValueError("duration_s must be finite and positive")
        self.start_velocity = np.zeros(3, dtype=float) if start_velocity is None else _validate_chart(start_velocity, "start_velocity")
        self.end_velocity = np.zeros(3, dtype=float) if end_velocity is None else _validate_chart(end_velocity, "end_velocity")
        self.lam = float(lam)
        self.euler_order = str(euler_order)
        self._basis = _IntegralBasis(lam=self.lam, quadrature_order=quadrature_order)
        self._nu0 = self.tau * self.start_velocity
        self._nu1 = self.tau * self.end_velocity
        # A W matrix with rows corresponding to the four RBF bases and columns
        # to (x, y, yaw).  lstsq is the Moore-Penrose minimum Frobenius norm
        # solution because the endpoint contract has two rows and four bases.
        endpoint_b, endpoint_bp = self._basis.evaluate(1.)
        eta1, etap1 = _homogeneous(np.asarray([1.]), self.start, self.goal, self._nu0, self.lam)
        constraints = np.vstack([endpoint_b[0], endpoint_bp[0]])
        rhs = np.vstack([self.goal - eta1[0], self._nu1 - etap1[0]])
        self.weights, _, self.constraint_rank, self.constraint_singular_values = np.linalg.lstsq(constraints, rhs, rcond=None)
        self.constraint_residual = constraints @ self.weights - rhs

    def duration_s(self) -> float:
        return self.tau

    @property
    def rank(self) -> int:
        return int(self.constraint_rank)

    def local_state(self, t_s: float) -> LocalState:
        """Return local position and phase derivative at physical time ``t_s``."""
        t = float(t_s)
        if not np.isfinite(t):
            raise ValueError("t_s must be finite")
        # Hold exactly at both ends.  This makes a replan at the endpoint have
        # a declared zero velocity instead of leaking a numerical phase clamp.
        s = float(np.clip(t / self.tau, 0., 1.))
        eta, etap = _homogeneous(np.asarray([s]), self.start, self.goal, self._nu0, self.lam)
        b, bp = self._basis.evaluate(s)
        z = eta[0] + b[0] @ self.weights
        zp = etap[0] + bp[0] @ self.weights
        if t <= 0.:
            z, zp = self.start.copy(), self._nu0.copy()
        elif t >= self.tau:
            z, zp = self.goal.copy(), self._nu1.copy()
        return LocalState(s, z, zp)

    def local_position_velocity(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        state = self.local_state(t_s)
        return state.position.copy(), state.derivative.copy() / self.tau

    def _world_pose_twist(self, position: np.ndarray, velocity_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, y, yaw = position
        T_local = np.eye(4)
        T_local[:3, :3] = Rotation.from_rotvec([0., 0., yaw]).as_matrix()
        T_local[:3, 3] = [x, y, 0.]
        T_world = self.anchor @ T_local
        pose = np.r_[T_world[:3, 3], Rotation.from_matrix(T_world[:3, :3]).as_euler(self.euler_order)]
        R_anchor = self.anchor[:3, :3]
        linear_world = R_anchor @ np.r_[velocity_local[:2], 0.]
        angular_world = R_anchor @ np.array([0., 0., velocity_local[2]])
        return pose, np.r_[linear_world, angular_world]

    def sample(self, t_s: float) -> MotionReference:
        """Return a controller-compatible world pose and analytic 6D twist."""
        state = self.local_state(t_s)
        pose, twist = self._world_pose_twist(state.position, state.derivative / self.tau)
        return MotionReference(pose_d=pose, vel_ff=twist, t_ref=float(t_s))

    def sample_world(self, t_s: float) -> tuple[np.ndarray, np.ndarray]:
        state = self.local_state(t_s)
        return self._world_pose_twist(state.position, state.derivative / self.tau)


def _unit_shape(*, lam: float = LAMBDA, quadrature_order: int = 64, n: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    """Return the scalar zero-velocity 0->1 normalized ProDMP shape."""
    if n < 5:
        raise ValueError("n must be at least 5")
    ref = ProDMPReference([0., 0., 0.], [1., 1., 1.], np.eye(4), duration_s=1., lam=lam, quadrature_order=quadrature_order)
    s = np.linspace(0., 1., int(n))
    values = np.asarray([ref.local_state(float(x)).position for x in s])
    return s, values[:, 0]


_UNIT_SHAPE_CACHE: dict[tuple[float, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _shape_derivative_peak(method: str) -> float:
    method = str(method).lower()
    if method == "prodmp":
        key = (LAMBDA, 64, 1001)
        if key not in _UNIT_SHAPE_CACHE:
            _UNIT_SHAPE_CACHE[key] = _unit_shape()
        s, shape = _UNIT_SHAPE_CACHE[key]
        derivative = np.gradient(shape, s, edge_order=2)
        return float(np.max(np.abs(derivative)))
    if method == "dmp":
        source = Path("/media/camp/EXT_DRIVE/Among_US")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from DMP_playground.dmp_honest.dmp import fit_discrete_dmp
        phase = np.linspace(0., 1., 41)
        primitive = fit_discrete_dmp(phase, phase[:, None])
        values, _ = primitive.rollout(phase)
        shape = values[:, 0] / values[-1, 0]
        return float(np.max(np.abs(np.diff(shape)) * 40.))
    raise ValueError("method must be 'prodmp' or 'dmp'")


def _duration_pair(start: np.ndarray, goal: np.ndarray, *, method: str) -> float:
    delta = goal - start
    peak = _shape_derivative_peak(method)
    translation = peak * float(np.linalg.norm(delta[:2])) / DEFAULT_TRANSLATION_SPEED_M_S
    yaw = peak * abs(float(delta[2])) / DEFAULT_YAW_SPEED_RAD_S
    return float(max(MIN_DURATION_S, translation, yaw))


def duration_matrix(poses: np.ndarray, method: str = "prodmp") -> np.ndarray:
    """Estimate pairwise durations using the declared peak speed limits.

    ``poses`` is ``(N,3)`` in local ``(x_m,y_m,yaw_rad)`` coordinates.  The
    diagonal is the explicit same-pose hold time ``0.1 s``.  A caller that
    wants a dwell after arrival must add it separately; this keeps hold and
    travel duration distinguishable in B's cost accounting.
    """
    arr = np.asarray(poses, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all():
        raise ValueError("poses must be a finite (N,3) local chart array")
    result = np.empty((arr.shape[0], arr.shape[0]), dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            result[i, j] = MIN_DURATION_S if i == j else _duration_pair(arr[i], arr[j], method=method)
    return result


def reference(start: Iterable[float], goal: Iterable[float], anchor: np.ndarray, *,
              duration: float | None = None, duration_s: float | None = None,
              start_velocity: Iterable[float] | None = None,
              end_velocity: Iterable[float] | None = None,
              method: str = "prodmp", euler_order: str = "xyz",
              lam: float = LAMBDA, quadrature_order: int = 64) -> ProDMPReference:
    """Build a material-chart reference and decode it with ``anchor``.

    ``duration`` and ``duration_s`` are aliases.  The ProDMP implementation is
    selected by default.  ``method='dmp'`` remains available only for the
    duration/speed baseline and still returns the integral-basis reference
    shape, so callers cannot silently label a DMP rollout as ProDMP.
    """
    if duration is not None and duration_s is not None and abs(float(duration) - float(duration_s)) > 1.e-12:
        raise ValueError("duration and duration_s disagree")
    start_arr, goal_arr = _validate_chart(start, "start"), _validate_chart(goal, "goal")
    duration_value = duration if duration is not None else duration_s
    if duration_value is None:
        duration_value = _duration_pair(start_arr, goal_arr, method=method)
    return ProDMPReference(start_arr, goal_arr, anchor, duration_s=float(duration_value),
                           start_velocity=start_velocity, end_velocity=end_velocity,
                           lam=lam, quadrature_order=quadrature_order,
                           euler_order=euler_order)


__all__ = [
    "LAMBDA", "RBF_CENTERS", "RBF_WIDTHS", "PHASE_GATE_DECAY",
    "MotionReference", "LocalState", "ProDMPReference", "duration_matrix",
    "reference",
]

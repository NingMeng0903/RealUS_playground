"""Causal C2 composition of a material ProDMP and a moving pose atlas.

Coordinates are longitudinal metres, lateral metres, and normalized probe
tilt. Each reference accepts one selected target and an incoming material
state. The fixed atlas and body clock can be shared across successive actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BPoly, RegularGridInterpolator
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

MOTION_ROOT = Path(__file__).resolve().parent.parent / "ab_observation_stress_20261111"
sys.path.insert(0, str(MOTION_ROOT))
from motion import MotionReference, _homogeneous, _phi  # noqa: E402


@dataclass(frozen=True)
class BodyClock:
    """Synthetic knee motion on one absolute clock through the capture poses."""

    slot_s: float
    start_index: int
    center_rad: float
    half_range_rad: float

    def sample(self, time_s):
        rate = np.pi / (2. * self.slot_s)
        phase = self.start_index * np.pi / 2. + rate * np.asarray(time_s)
        return (
            self.center_rad - self.half_range_rad * np.cos(phase),
            self.half_range_rad * rate * np.sin(phase),
            self.half_range_rad * rate**2 * np.cos(phase),
        )


def _along(spline, coordinates, rate, acceleration):
    """Value and derivatives along a curve through a tensor-product spline."""
    point = spline(coordinates)
    first = np.zeros_like(point)
    second = np.zeros_like(point) if acceleration is not None else None
    for i in range(coordinates.shape[-1]):
        derivative = np.zeros(coordinates.shape[-1], dtype=int)
        derivative[i] = 1
        partial = spline(coordinates, nu=tuple(derivative))
        first += partial * rate[:, i, None]
        if second is None:
            continue
        second += partial * acceleration[:, i, None]
        for j in range(i, coordinates.shape[-1]):
            mixed = derivative.copy()
            mixed[j] += 1
            factor = 1. if i == j else 2.
            second += spline(coordinates, nu=tuple(mixed)) * (factor * rate[:, i] * rate[:, j])[:, None]
    return point, first, second


class SmoothAtlas:
    """C2 surrogate of the baked contact/attitude table in robot coordinates."""

    def __init__(self, geometry):
        angles = np.deg2rad(geometry["pose_knee_deg"])
        self.angles = angles
        self.stations = geometry["s_mm"] * 1.e-3
        self.offsets = geometry["view_offsets_mm"] * 1.e-3
        tilts = geometry["view_tilts_deg"]
        self.tilts = tilts / np.max(np.abs(tilts))
        grid_offsets, views = np.unique(self.offsets, return_index=True)
        center = int(np.flatnonzero((self.offsets == 0.) & (self.tilts == 0.))[0])
        if views[grid_offsets == 0.][0] != center:
            raise ValueError("expected the neutral view in the contact grid")
        minus = int(np.flatnonzero((self.offsets == 0.) & (self.tilts == -1.))[0])
        plus = int(np.flatnonzero((self.offsets == 0.) & (self.tilts == 1.))[0])
        self.targets = geometry["target_robot"]
        self.lower = np.array([self.stations[0], grid_offsets[0], -1.])
        self.upper = np.array([self.stations[-1], grid_offsets[-1], 1.])
        self.orientation_anchor = Rotation.from_matrix(self.targets[0, 0, center, :3, :3])
        rotations = self.targets[..., :3, :3]
        logs = (self.orientation_anchor.inv() * Rotation.from_matrix(rotations.reshape(-1, 3, 3)))
        logs = logs.as_rotvec().reshape(rotations.shape[:-2] + (3,))
        if np.max(np.linalg.norm(logs, axis=-1)) >= np.pi - .15:
            raise ValueError("baked probe orientations cross the selected SO(3) log chart")
        base = np.concatenate((self.targets[..., :3, 3][:, :, views], logs[:, :, views]), axis=-1)
        linear = (logs[:, :, plus] - logs[:, :, minus]) / 2.
        quadratic = (logs[:, :, plus] + logs[:, :, minus]) / 2. - logs[:, :, center]
        self.base = RegularGridInterpolator(
            (angles, self.stations, grid_offsets), base, method="cubic", solver=spsolve,
        )
        self.tilt = RegularGridInterpolator(
            (angles, self.stations), np.concatenate((linear, quadratic), axis=-1),
            method="cubic", solver=spsolve,
        )

    def material(self, state):
        _, station, view = state
        return np.array([self.stations[station], self.offsets[view], self.tilts[view]])

    def sample(self, body, position, velocity, acceleration=None):
        """Compose the body and material jets before decoding a physical twist."""
        theta, theta_dot, theta_ddot = body
        if np.any(theta < self.angles[0] - 1.e-12) or np.any(theta > self.angles[-1] + 1.e-12):
            raise ValueError("body pose left the calibrated atlas interval")
        theta = np.clip(theta, self.angles[0], self.angles[-1])
        points = np.column_stack((theta, position[:, :2]))
        speed = np.column_stack((theta_dot, velocity[:, :2]))
        acc = None if acceleration is None else np.column_stack((theta_ddot, acceleration[:, :2]))
        value, rate, derivative2 = _along(self.base, points, speed, acc)
        tilt, tilt_rate, tilt_acc = _along(
            self.tilt, points[:, :2], speed[:, :2], None if acc is None else acc[:, :2],
        )
        beta, beta_rate = position[:, 2, None], velocity[:, 2, None]
        powers = np.hstack((beta, beta**2))
        powers_rate = np.hstack((beta_rate, 2. * beta * beta_rate))
        tilt = tilt.reshape(-1, 2, 3)
        tilt_rate = tilt_rate.reshape(-1, 2, 3)
        value[:, 3:] += np.einsum("ni,nij->nj", powers, tilt)
        rate[:, 3:] += (np.einsum("ni,nij->nj", powers_rate, tilt)
                        + np.einsum("ni,nij->nj", powers, tilt_rate))
        if derivative2 is not None:
            beta_acc = acceleration[:, 2, None]
            powers_acc = np.hstack((beta_acc, 2. * (beta_rate**2 + beta * beta_acc)))
            derivative2[:, 3:] += (
                np.einsum("ni,nij->nj", powers_acc, tilt)
                + 2. * np.einsum("ni,nij->nj", powers_rate, tilt_rate)
                + np.einsum("ni,nij->nj", powers, tilt_acc.reshape(-1, 2, 3))
            )
        return value, rate, derivative2

    def pose_and_twist(self, value, rate):
        """Map rotation-vector rates to world angular velocity, not Euler rates."""
        vector, vector_rate = value[:, 3:], rate[:, 3:]
        rotation = self.orientation_anchor * Rotation.from_rotvec(vector)
        angle = np.linalg.norm(vector, axis=1)
        cross = np.cross(vector, vector_rate)
        a = .5 * np.sinc(angle / (2. * np.pi))**2
        b = np.full_like(angle, 1. / 6.)
        usable = angle > 1.e-4
        b[usable] = (angle[usable] - np.sin(angle[usable])) / angle[usable]**3
        omega = vector_rate + a[:, None] * cross + b[:, None] * np.cross(vector, cross)
        return value[:, :3], rotation, np.column_stack((rate[:, :3], self.orientation_anchor.apply(omega)))


def _phase(s):
    """A short C2 ramp of the learned phase with no change to its spatial shape."""
    width = .12
    u = np.clip(np.minimum(s, 1. - s) / width, 0., 1.)
    ramp = width * (u**3 - .5 * u**4)
    progress = np.where(s < width, ramp, np.where(s > 1. - width, 1. - width - ramp, s - width / 2.))
    speed = u**2 * (3. - 2. * u)
    acceleration = 6. * u * (1. - u) * np.where(s < .5, 1., -1.) / width
    return progress / (1. - width), speed / (1. - width), acceleration / (1. - width)


def _shape_spline(primitive, skill):
    """Tabulate the integrated ProDMP with analytic first and second jets."""
    phases = np.linspace(0., 1., 129)
    weights = primitive.endpoint_only_weights if skill == "endpoint_only" else primitive.weight_mean[skill]
    origin, goal = np.zeros(2), np.array([1., 0.])
    base, base_rate = _homogeneous(phases, origin, goal, origin, primitive.lam)
    basis, basis_rate = primitive.basis.evaluate(phases)
    value = base + basis @ weights
    velocity = base_rate + basis_rate @ weights
    np.testing.assert_allclose(value[[0, -1]], [origin, goal], atol=1.e-11, rtol=0.)
    np.testing.assert_allclose(velocity[[0, -1]], 0., atol=1.e-11, rtol=0.)
    value[[0, -1]], velocity[[0, -1]] = [origin, goal], 0.
    acceleration = (_phi(phases) @ weights - 2. * primitive.lam * velocity
                    - primitive.lam**2 * (value - goal))
    return BPoly.from_derivatives(phases, np.stack((value, velocity, acceleration), axis=1))


class TimedProbe:
    """One observed-action segment with endpoint material position, speed and acceleration."""

    def __init__(self, atlas, clock, shape, time_s, duration_s, start_jet, goal):
        self.atlas = atlas
        self.clock = clock
        self.shape = shape
        self.time_s = float(time_s)
        self.duration = float(duration_s)
        self.start_jet = np.asarray(start_jet, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        if not np.isfinite(self.duration) or self.duration <= 0. or not np.isfinite(self.time_s):
            raise ValueError("need finite start time and positive duration")
        if self.start_jet.shape != (3, 3) or self.goal.shape != (3,):
            raise ValueError("need positive duration and finite material endpoint jets")
        if not np.isfinite(self.start_jet).all() or not np.isfinite(self.goal).all():
            raise ValueError("nonfinite material endpoints")
        delta = self.goal - self.start_jet[0]
        self.basis_xy = np.column_stack((delta[:2], [-delta[1], delta[0]]))
        self.linear = BPoly.from_derivatives([0., 1.], [[0., 0., 0.], [1., 0., 0.]])
        self.correction = BPoly.from_derivatives([0., 1.], [
            [np.zeros(3), self.duration * self.start_jet[1], self.duration**2 * self.start_jet[2]],
            [np.zeros(3), np.zeros(3), np.zeros(3)],
        ])
        for use_shape in (True, False):
            self.use_shape = use_shape and self.shape is not None
            positions = self.material_jet(np.linspace(0., self.duration, 513))[0]
            if np.all(positions >= atlas.lower - 1.e-12) and np.all(positions <= atlas.upper + 1.e-12):
                break
        else:
            raise ValueError("no endpoint-conditioned reference in the material corridor")

    def duration_s(self):
        return self.duration

    def material_jet(self, elapsed_s):
        s = np.asarray(elapsed_s, dtype=float).reshape(-1) / self.duration
        if np.any(s < -1.e-11) or np.any(s > 1. + 1.e-11):
            raise ValueError("sample is outside this segment")
        s = np.clip(s, 0., 1.)
        u, u1, u2 = _phase(s)
        xy = np.zeros((3, len(s), 2))
        if self.use_shape:
            xy[0] = self.shape(u) @ self.basis_xy.T
            xy[1] = self.shape(u, 1) @ self.basis_xy.T * u1[:, None]
            xy[2] = (self.shape(u, 2) @ self.basis_xy.T * u1[:, None]**2
                     + self.shape(u, 1) @ self.basis_xy.T * u2[:, None])
        else:
            xy = np.array([self.linear(s, i)[:, None] * (self.goal - self.start_jet[0])[:2] for i in range(3)])
        beta = (self.goal[2] - self.start_jet[0, 2]) * np.array([self.linear(s, i) for i in range(3)])
        jet = np.concatenate((xy, beta[..., None]), axis=2)
        for order in range(3):
            jet[order] += self.correction(s, order)
            jet[order] /= self.duration**order
        jet[0] += self.start_jet[0]
        # Preserve the exact capture sites while only snapping roundoff.
        jet[0, s == 0.] = self.start_jet[0]
        jet[0, s == 1.] = self.goal
        return jet

    def world_jet(self, elapsed_s, *, acceleration=True):
        elapsed = np.asarray(elapsed_s, dtype=float).reshape(-1)
        position, velocity, derivative2 = self.material_jet(elapsed)
        # Bounds are checked during construction and at every evaluator call.
        if np.any(position < self.atlas.lower - 1.e-12) or np.any(position > self.atlas.upper + 1.e-12):
            raise ValueError("reference left the sampled material corridor")
        position = np.clip(position, self.atlas.lower, self.atlas.upper)
        return self.atlas.sample(
            self.clock.sample(self.time_s + elapsed), position, velocity,
            derivative2 if acceleration else None,
        )

    def sample(self, elapsed_s, euler_order="xyz"):
        value, rate, _ = self.world_jet(elapsed_s, acceleration=False)
        position, rotation, twist = self.atlas.pose_and_twist(value, rate)
        return MotionReference(np.r_[position[0], rotation.as_euler(euler_order)[0]], twist[0], float(elapsed_s))


def rest_jet(position):
    return np.vstack((np.asarray(position, dtype=float), np.zeros((2, 3))))


def learned_shapes(primitive):
    return {skill: _shape_spline(primitive, skill) for skill in (*primitive.skills, "endpoint_only")}

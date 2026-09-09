"""Compact, reproducible two-point forearm paths. No hardware or file I/O.

Planning and arc-length integration run in the acquisition process. The
controller receives a small immutable specification and only evaluates it.
All geometry is in the controller FK frame, metres and xyz Euler radians.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from rm75_control.control.admittance_common.reference import MotionReference
from rm75_control.control.joint_admittance_8dof.reference import _soft_start_time_warp

SCAN_ORDER = tuple((shape, direction) for direction in ("DtP", "PtD") for shape in "LCS")
SCAN_FORCE_AXES = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
TILT_PROFILE = dict(mass=0.051, damping=0.22, coulomb_nm=0.025,
                    vmax_rad_s=0.28, a_max=3.0)
PEAK_RANGE_M = (0.010, 0.015)
# Stored amplitude is pre-normalization. New C/S peaks use PEAK_RANGE_M;
# old 20 mm specs and the legacy L placeholder (0.02) still load.
AMPLITUDE_LOAD_MAX_M = 0.020


def force_profile(name="icra"):
    if name not in ("baseline", "icra"):
        raise ValueError("force profile must be baseline or icra")
    return {"hybrid_motion": {"torque_tilt": dict(TILT_PROFILE)}} if name == "icra" else {}


def outward(pose):
    return -Rotation.from_euler("xyz", np.asarray(pose)[3:6]).as_matrix()[:, 2]


def lift_pose(pose, distance=0.03):
    result = np.asarray(pose, dtype=float).reshape(6).copy()
    result[:3] += float(distance) * outward(result)
    return result


def _offset(u, shape, coefficients, amplitude):
    """Analytic lateral displacement and derivative in canonical D→P order."""
    u = np.asarray(u, dtype=float)
    if shape == "L":
        return np.zeros_like(u), np.zeros_like(u)
    x = np.pi * u
    base = np.sin(x) ** 2
    slope = np.pi * np.sin(2 * x)
    if shape == "S":
        norm = 3 * math.sqrt(3) / 8
        slope = (slope * np.sin(2*x) + base * 2*np.pi * np.cos(2*x)) / norm
        base = base * np.sin(2*x) / norm
    coefficients = np.asarray(coefficients)
    noise, noise_d = np.zeros_like(u) + coefficients[0], np.zeros_like(u)
    # A small random amplitude component plus smooth spatial variations.
    for index, coefficient in enumerate(coefficients[1:]):
        k = index // 2 + 1
        phase = k * np.pi * u
        if index % 2 == 0:
            noise += coefficient * np.sin(phase)
            noise_d += coefficient * k*np.pi * np.cos(phase)
        else:
            noise += coefficient * np.cos(phase)
            noise_d -= coefficient * k*np.pi * np.sin(phase)
    factor = 1 + 0.075 * noise
    return amplitude * base * factor, amplitude * (slope * factor + base * 0.075 * noise_d)


def make_spec(distal, proximal, shape, direction, seed, *, speed=0.02, side=1):
    poses = np.asarray([distal, proximal], dtype=float).reshape(2, 6)
    if not np.isfinite(poses).all():
        raise ValueError("Invalid taught pose.")
    delta = poses[1, :3] - poses[0, :3]
    distance = float(np.linalg.norm(delta))
    if not 0.05 <= distance <= 0.50:
        raise ValueError("Point distance must be 5-50 cm. Teach again.")
    if shape not in "LCS" or len(shape) != 1 or direction not in ("DtP", "PtD"):
        raise ValueError("invalid scan shape/direction")
    if side not in (-1, 1) or not math.isfinite(speed) or not 0 < speed <= 0.02:
        raise ValueError("side must be ±1; scan speed must be in (0, 0.02] m/s")
    cross = np.cross(outward(poses[0]), delta / distance)
    if np.linalg.norm(cross) < 0.3:
        raise ValueError("Points align with probe normal. Teach again.")
    lateral = side * cross / np.linalg.norm(cross)
    rng = np.random.default_rng(int(seed))
    coefficients = np.zeros(6)
    if shape != "L":
        coefficients[0] = rng.choice([-0.85, 0.85])
        spatial = rng.uniform(-1, 1, 5)
        coefficients[1:] = 0.15 * spatial / max(float(np.abs(spatial).sum()), 1e-9)
    # Integrate externally, then transmit only 129 monotonically spaced knots.
    grid = np.linspace(0, 1, 4097)
    amplitude = 0.02  # Preserve the legacy straight-line specification.
    peaks = np.zeros(2)
    if shape != "L":
        # Find the actual noisy extrema, not just the nominal base amplitude.
        # This planning work stays outside the controller's sampling loop.
        from scipy.optimize import brentq

        _, slope = _offset(grid, shape, coefficients, 1.0)
        brackets = np.flatnonzero(slope[:-1] * slope[1:] < 0)
        extrema = [0.0, 0.5, 1.0]
        extrema += [brentq(lambda u: float(_offset(u, shape, coefficients, 1.0)[1]),
                          grid[i], grid[i+1], xtol=1e-14) for i in brackets]
        offsets, _ = _offset(extrema, shape, coefficients, 1.0)
        peaks = np.array([max(offsets), -min(offsets)])
        if shape == "C":
            peaks[1] = 0.0
        lobes = peaks[:1] if shape == "C" else peaks
        # Both S lobes must remain in range. One common scale preserves the
        # smooth centre crossing; independent half-curve scaling would kink it.
        amplitude = float(rng.uniform((PEAK_RANGE_M[0] + 1e-10) / min(lobes),
                                      (PEAK_RANGE_M[1] - 1e-10) / max(lobes)))
        peaks *= amplitude
    _, dy = _offset(grid, shape, coefficients, amplitude)
    tangent = delta[None, :] + dy[:, None] * lateral
    rate = np.linalg.norm(tangent, axis=1)
    arc = np.r_[0.0, np.cumsum((rate[:-1] + rate[1:]) * (0.5 / 4096))]
    return dict(schema="icra_path_v1", poses=poses.tolist(), shape=shape, direction=direction,
                seed=int(seed), side=int(side), lateral=lateral.tolist(),
                noise_coefficients=coefficients.tolist(), amplitude_m=amplitude,
                peak_range_m=list(PEAK_RANGE_M), peak_offsets_m=peaks.tolist(),
                speed_m_s=float(speed), ramp_s=0.4, arc_m=arc[::32].tolist())


class ForearmReference:
    """Absolute smooth path with an arc-length clock and shortest orientation."""
    def __init__(self, spec):
        self.spec = copy.deepcopy(spec)
        if spec.get("schema") != "icra_path_v1":
            raise ValueError("unsupported ICRA path schema")
        self.poses = np.asarray(spec["poses"], dtype=float).reshape(2, 6)
        self.lateral = np.asarray(spec["lateral"], dtype=float).reshape(3)
        self.coefficients = np.asarray(spec["noise_coefficients"], dtype=float).reshape(6)
        self.arc = np.asarray(spec["arc_m"], dtype=float)
        self.speed = float(spec["speed_m_s"])
        self.ramp = float(spec["ramp_s"])
        self.amplitude = float(spec["amplitude_m"])
        if (self.arc.shape != (129,) or self.arc[0] != 0 or np.any(np.diff(self.arc) <= 0)
                or not all(np.isfinite(a).all() for a in
                           (self.poses, self.lateral, self.coefficients, self.arc))
                or not 0 < self.speed <= 0.02 or self.ramp != 0.4
                # The coefficient factor is in [0.925, 1.075]. The stored
                # amplitude is pre-normalization; old 20 mm specs still load.
                or not PEAK_RANGE_M[0]/1.075 <= self.amplitude <= AMPLITUDE_LOAD_MAX_M/0.925
                or abs(np.linalg.norm(self.lateral) - 1) > 1e-6
                or np.abs(self.coefficients).sum() > 1.000001
                or spec["shape"] not in ("L", "C", "S")
                or spec["direction"] not in ("DtP", "PtD")):
            raise ValueError("invalid ICRA geometry/timing specification")
        self.delta = self.poses[1, :3] - self.poses[0, :3]
        if abs(float(self.delta @ self.lateral)) > 1e-6:
            raise ValueError("lateral must be perpendicular to D→P")
        self.length_m = float(self.arc[-1])
        if not 0.05 <= np.linalg.norm(self.delta) <= self.length_m + 1e-6 or self.length_m > 1:
            raise ValueError("invalid path length")
        self.duration_s = self.length_m / self.speed + self.ramp
        self._u = PchipInterpolator(self.arc, np.linspace(0, 1, len(self.arc)))
        self._du = self._u.derivative()
        rots = Rotation.from_euler("xyz", self.poses[:, 3:])
        self._rot = Slerp([0, 1], rots)
        self._rotation_vector_world = rots[0].apply((rots[0].inv() * rots[1]).as_rotvec())
        self._origin = 0.0

    def set_origin(self, pose0, *, t_s=0.0):
        self._origin = 0.0 if t_s is None else float(t_s)

    def at(self, u):
        u = float(np.clip(u, 0, 1))
        offset, slope = _offset(u, self.spec["shape"], self.coefficients, self.amplitude)
        xyz = self.poses[0, :3] + u * self.delta + offset * self.lateral
        pose = np.r_[xyz, self._rot(u).as_euler("xyz")]
        return pose, self.delta + slope * self.lateral

    def sample(self, t_s):
        t = max(0.0, float(t_s) - self._origin)
        tau, speed_scale = _soft_start_time_warp(t, self.ramp, self.duration_s, self.ramp)
        travelled = float(np.clip(self.speed * tau, 0, self.length_m))
        reverse = self.spec["direction"] == "PtD"
        s = self.length_m - travelled if reverse else travelled
        u = float(np.clip(self._u(s), 0, 1))
        pose, tangent = self.at(u)
        du_dt = float(self._du(s)) * self.speed * speed_scale * (-1 if reverse else 1)
        if t >= self.duration_s:
            du_dt = 0.0
            pose = self.poses[0 if reverse else 1].copy()
        return MotionReference(pose_d=pose,
                               vel_ff=np.r_[tangent * du_dt, self._rotation_vector_world * du_dt],
                               t_ref=float(t_s))

"""Differentiable world-pose scan reference with analytic geometric velocity."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline

from rm75_control.control.admittance_common.reference import MotionReference


def _maximum_spline_speed(spline: CubicSpline) -> float:
    """Maximum |dp/du| from all real stationary points of its quartic square."""
    maximum = 0.0
    for i, span in enumerate(np.diff(spline.x)):
        # Each derivative component is quadratic in local segment parameter.
        coeff = np.stack((spline.c[2, i], 2*spline.c[1, i], 3*spline.c[0, i]))
        squared = sum((np.convolve(coeff[:, j], coeff[:, j]) for j in range(3)),
                      start=np.zeros(5))
        derivative = np.arange(1, len(squared))*squared[1:]
        roots = np.polynomial.polynomial.polyroots(np.trim_zeros(derivative, 'b')) if np.any(derivative) else []
        candidates = [0., float(span)]
        candidates.extend(float(r.real) for r in roots
                          if abs(r.imag) < 1e-9 and 0. < r.real < span)
        for u in candidates:
            maximum = max(maximum, float(np.linalg.norm(spline(spline.x[i]+u, 1))))
    return maximum


class WorldPoseSplineReference:
    """C2 pose curve through supplied poses, with a shared start/stop time law.

    Linear velocity is dp/dt. Angular velocity is vee(dR/dt R.T), in the
    world/base frame, rather than Euler-angle derivatives. The original
    poses are interpolated exactly; the curve between them is a spline and
    must be checked as the actual candidate path.

    Consecutive positions must differ: this scan reference is parameterized
    by input chord length. Pure rotation uses a separately timed reference.
    """
    def __init__(self, points, *, rpy, speed_m_s=.02, ramp_s=.8,
                 euler_order="xyz", soft_start=True, **kwargs):
        # Explicitly accept the old start flag but always use a symmetric
        # ramp in this scan-specific reference; it is not a legacy mode.
        if not soft_start:
            raise ValueError("scan pose reference requires soft_start=True")
        if kwargs:
            raise TypeError(f"unsupported scan reference options: {sorted(kwargs)}")
        points = np.asarray(points, dtype=float)
        angles = np.asarray(rpy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("points must have shape (N,3), N>=2")
        if angles.shape != points.shape or not np.isfinite(points).all() or not np.isfinite(angles).all():
            raise ValueError("rpy must match finite points")
        self.points, self.rpy = points.copy(), angles.copy()
        self.speed_m_s, self.ramp_s = float(speed_m_s), float(ramp_s)
        if not np.isfinite(self.speed_m_s) or self.speed_m_s <= 0.:
            raise ValueError("speed_m_s must be finite and positive")
        if not np.isfinite(self.ramp_s) or self.ramp_s <= 0.:
            raise ValueError("ramp_s must be finite and positive")
        self.euler_order = str(euler_order)
        chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(chord <= 1e-10):
            raise ValueError("consecutive scan positions must be distinct")
        self.knots = np.r_[0., np.cumsum(chord)]
        self.position_spline = CubicSpline(self.knots, points, bc_type='natural')
        self.rotation_spline = RotationSpline(self.knots, Rotation.from_euler(self.euler_order, angles))
        self.maximum_parameter_speed = _maximum_spline_speed(self.position_spline)
        self.parameter_rate = self.speed_m_s/(self.maximum_parameter_speed*(1.+1e-12))
        nominal_duration = self.knots[-1]/self.parameter_rate
        self.effective_ramp_s = min(self.ramp_s, nominal_duration)
        self.cruise_s = max(0., nominal_duration-self.effective_ramp_s)
        self._t_anchor = 0.

    def set_origin(self, pose0, *, t_s=None):
        del pose0
        if t_s is not None:
            self._t_anchor = float(t_s)

    def duration_s(self):
        return 2*self.effective_ramp_s+self.cruise_s

    def parameter_state(self, t_s):
        t = float(t_s)-self._t_anchor
        r, c, v = self.effective_ramp_s, self.cruise_s, self.parameter_rate
        if t <= 0.:
            return 0., 0.
        if t >= self.duration_s():
            return float(self.knots[-1]), 0.
        if t < r:
            return v*(t/2-r/(2*np.pi)*np.sin(np.pi*t/r)), v/2*(1-np.cos(np.pi*t/r))
        if t < r+c:
            return v*r/2+v*(t-r), v
        u=t-r-c
        return v*r/2+v*c+v*(u/2+r/(2*np.pi)*np.sin(np.pi*u/r)), v/2*(1+np.cos(np.pi*u/r))

    def sample(self, t_s):
        u, u_dot = self.parameter_state(t_s)
        rot = self.rotation_spline(u)
        pose = np.r_[self.position_spline(u), rot.as_euler(self.euler_order)]
        linear = self.position_spline(u, 1)*u_dot
        # scipy's angular rate is expressed in the instantaneous local frame.
        angular = rot.apply(self.rotation_spline(u, 1))*u_dot
        return MotionReference(pose, np.r_[linear, angular], float(t_s))

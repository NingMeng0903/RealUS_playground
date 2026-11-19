"""Smooth material coordinates on the fixed right-leg lattice.

The geometric parameters are (h, theta, r). The public material coordinates
are (theta, h, d), with d defined once from the reference graph depths.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

TAU = 2. * np.pi
INVERSE_TOL_M = 2.e-9


def wrap_angle(theta):
    angle = np.remainder(theta, TAU)
    # Subtracting an angle smaller than an ulp can round up to the seam.
    return np.where(angle == TAU, 0., angle)


class PeriodicCubic:
    """Tensor cubic interpolant with a periodic angular axis.

    Coefficients use ascending powers of normalized local h and theta.
    The same knot grids and interpolation operators are used for every shape.
    """

    def __init__(self, levels, angles, samples):
        self.h = np.asarray(levels, dtype=float)
        self.theta = np.r_[angles, TAU]
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 3 or samples.shape[:2] != (len(self.h), len(angles)):
            raise ValueError("expected a vector at each material lattice point")
        if self.theta[0] != 0. or not np.all(np.diff(self.theta) > 0.):
            raise ValueError("angular knots must increase from zero to the periodic seam")
        self.dh, self.dt = np.diff(self.h), np.diff(self.theta)
        order = np.arange(4)
        h_coeff = CubicSpline(self.h, samples, axis=0).c[::-1]
        h_coeff *= self.dh[None, :, None, None] ** order[:, None, None, None]
        periodic = np.concatenate((h_coeff, h_coeff[:, :, :1]), axis=2)
        coeff = CubicSpline(self.theta, periodic, axis=2, bc_type="periodic").c[::-1]
        coeff *= self.dt[None, :, None, None, None] ** order[:, None, None, None, None]
        self.coeff = np.ascontiguousarray(coeff.transpose(3, 1, 2, 0, 4))

    @staticmethod
    def basis(x):
        return np.column_stack((np.ones_like(x), x, x * x, x ** 3))

    @staticmethod
    def derivative(x, widths):
        return np.column_stack((np.zeros_like(x), np.ones_like(x), 2. * x, 3. * x * x)) / widths[:, None]

    def evaluate(self, h, theta, *, derivative=False):
        h, theta = np.broadcast_arrays(h, theta)
        shape = h.shape
        h, theta = h.ravel(), wrap_angle(theta.ravel())
        if np.any((h < self.h[0]) | (h > self.h[-1])) or not np.isfinite(h + theta).all():
            raise ValueError("query outside the spline h/theta domain")
        i = np.clip(np.searchsorted(self.h, h, side="right") - 1, 0, len(self.dh) - 1)
        j = np.searchsorted(self.theta, theta, side="right") - 1
        u, v = (h - self.h[i]) / self.dh[i], (theta - self.theta[j]) / self.dt[j]
        block = self.coeff[i, j]
        bh, bt = self.basis(u), self.basis(v)

        def evaluate(left, right):
            return np.einsum("ni,nijc,nj->nc", left, block, right).reshape(shape + (block.shape[-1],))

        value = evaluate(bh, bt)
        if not derivative:
            return value
        return value, evaluate(self.derivative(u, self.dh[i]), bt), evaluate(bh, self.derivative(v, self.dt[j]))


class RadialDepth:
    """Interpolate the reference graph depths with a strictly inward-increasing d.

    Interpolate the log of the positive layer gaps in (h, theta), then
    normalize the gaps to sum to one. Along r, monotone Hermite cubics share
    slopes on both sides of a knot. The slopes are smooth functions of the
    positive gaps, so the full depth field is C1, not just each radial ray.
    """

    def __init__(self, levels, angles, radii, depth):
        self.r = np.asarray(radii, dtype=float)
        self.dr = np.diff(self.r)
        depth = np.asarray(depth, dtype=float)
        gaps = -np.diff(depth, axis=1)
        if (np.any(gaps <= 0.) or np.any(self.dr <= 0.) or not np.all(depth[:, 0] == 1.)
                or not np.all(depth[:, -1] == 0.)):
            raise ValueError("reference graph depths must strictly decrease outwards from 1 to 0")
        self.log_gaps = PeriodicCubic(levels, angles, np.log(gaps.transpose(0, 2, 1)))

    def profile(self, h, theta):
        log_gaps = self.log_gaps.evaluate(h, theta)
        gaps = np.exp(log_gaps - log_gaps.max(axis=-1, keepdims=True))
        gaps /= gaps.sum(axis=-1, keepdims=True)
        secants = gaps / self.dr
        # The smooth limiter reproduces linear profiles and keeps each knot
        # slope below 1.5 times either secant. All three radial Bezier
        # derivative controls therefore have a strictly positive magnitude.
        left, right = secants[..., :-1], secants[..., 1:]
        interior = 3. * left * right / (left + right + np.sqrt((left - right) ** 2 + left * right))
        slopes = np.concatenate((secants[..., :1], interior, secants[..., -1:]), axis=-1)
        bounds = np.concatenate((np.ones_like(gaps[..., :1]), 1. - np.cumsum(gaps, axis=-1)), axis=-1)
        bounds[..., -1] = 0.
        return bounds, slopes

    def _evaluate(self, radius, bounds, slopes):
        interval = np.clip(np.searchsorted(self.r, radius, side="right") - 1, 0, len(self.dr) - 1)
        width = self.dr[interval]
        t = (radius - self.r[interval]) / width

        def at(values, i):
            return np.take_along_axis(values, i[..., None], axis=-1)[..., 0]

        first = at(bounds, interval)
        last = at(bounds, interval + 1)
        left = first - width * at(slopes, interval) / 3.
        right = last + width * at(slopes, interval + 1) / 3.
        one = 1. - t
        return first * one ** 3 + 3. * left * one ** 2 * t + 3. * right * one * t ** 2 + last * t ** 3

    def evaluate(self, h, theta, radius):
        h, theta, radius = np.broadcast_arrays(h, theta, radius)
        if np.any((radius < self.r[0]) | (radius > self.r[-1])):
            raise ValueError("outside radial depth domain")
        return self._evaluate(radius, *self.profile(h, theta))

    def inverse(self, h, theta, depth):
        h, theta, depth = np.broadcast_arrays(h, theta, depth)
        if np.any((depth < 0.) | (depth > 1.)):
            raise ValueError("d must lie in [0, 1]")
        bounds, slopes = self.profile(h, theta)
        low, high = np.full_like(depth, self.r[0]), np.full_like(depth, self.r[-1])
        for _ in range(42):
            mid = (low + high) / 2.
            go_out = self._evaluate(mid, bounds, slopes) > depth
            low, high = np.where(go_out, mid, low), np.where(go_out, high, mid)
        middle = (low + high) / 2.
        return np.where(depth == 0., self.r[-1], np.where(depth == 1., self.r[0], middle))


class SplineTube:
    """Periodic cubic skin/core with an exactly radial interior.

    The radial form interpolates all original volume lattice positions up to
    the precision of their stored coordinates. Between lattice vertices the
    new geometry is curved. The material parameters remain fixed.
    """

    def __init__(self, levels, angles, radii, volume):
        self.h, self.angles, self.r = (np.asarray(x, dtype=float) for x in (levels, angles, radii))
        self.grid = np.asarray(volume, dtype=float).reshape(len(levels), len(radii), len(angles), 3)
        skin = self.grid[:, -1]
        cores = (self.grid[:, 0] - self.r[0] * skin) / (1. - self.r[0])
        core = cores.mean(axis=1)
        polar = core[:, None, None] + self.r[None, :, None, None] * (skin - core[:, None])[:, None]
        self.radial_binding_error_m = float(np.linalg.norm(polar - self.grid, axis=-1).max())
        if self.radial_binding_error_m > 2.e-7:
            raise ValueError("volume lattice does not come from a single radial core")
        self.skin = PeriodicCubic(self.h, self.angles, skin)
        self.core = CubicSpline(self.h, core, axis=0, extrapolate=False)
        degree = np.arange(4)
        self.core_coeff = self.core.c[::-1].transpose(1, 0, 2) * np.diff(self.h)[:, None, None] ** degree[None, :, None]
        self._seeds = np.stack(np.meshgrid(self.h, self.r, self.angles, indexing="ij"), axis=-1).reshape(-1, 3)
        self._seeds = self._seeds[:, [0, 2, 1]]
        self._tree = cKDTree(self.grid.reshape(-1, 3))

    def forward(self, params, *, derivative=False):
        params = np.asarray(params, dtype=float)
        h, theta, r = np.moveaxis(params, -1, 0)
        core = self.core(h)
        if derivative:
            skin, sh, st = self.skin.evaluate(h, theta, derivative=True)
        else:
            skin = self.skin.evaluate(h, theta)
        radial = skin - core
        value = core + r[..., None] * radial
        if not derivative:
            return value
        first = (1. - r[..., None]) * self.core(h, 1) + r[..., None] * sh
        return value, np.stack((first, r[..., None] * st, radial), axis=-1)

    def _clip(self, params):
        params = params.copy()
        params[:, 0] = np.clip(params[:, 0], self.h[0], self.h[-1])
        params[:, 1] = wrap_angle(params[:, 1])
        params[:, 2] = np.clip(params[:, 2], self.r[0], self.r[-1])
        return params

    def _newton(self, target, params):
        for _ in range(26):
            position, jac = self.forward(params, derivative=True)
            error = np.linalg.norm(target - position, axis=1)
            live = np.flatnonzero(error > INVERSE_TOL_M)
            if not len(live):
                break
            change = np.linalg.solve(jac[live], (target[live] - position[live])[..., None])[..., 0]
            cap = np.maximum(1., np.max(np.abs(change) / [.08, .55, .33], axis=1))
            change /= cap[:, None]
            moved = False
            for _ in range(9):
                candidate = self._clip(params[live] + change)
                closer = np.linalg.norm(target[live] - self.forward(candidate), axis=1) < error[live]
                params[live[closer]] = candidate[closer]
                moved |= bool(np.any(closer))
                live, change = live[~closer], change[~closer] * .5
                if not len(live):
                    break
            if not moved:
                break
        return params

    def inverse(self, points):
        points = np.asarray(points, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("physical points must end in three finite coordinates")
        target = points.reshape(-1, 3)
        if not len(target):
            return np.empty_like(points)
        _, indices = self._tree.query(target, k=6)
        result = self._newton(target, self._seeds[indices[:, 0]].copy())
        for j in range(1, indices.shape[1]):
            pending = np.flatnonzero(np.linalg.norm(self.forward(result) - target, axis=1) > INVERSE_TOL_M)
            if not len(pending):
                break
            result[pending] = self._newton(target[pending], self._seeds[indices[pending, j]].copy())
        if np.any(np.linalg.norm(self.forward(result) - target, axis=1) > INVERSE_TOL_M):
            raise ValueError("point is outside the certified spline chart or inverse did not converge")
        return result.reshape(points.shape)


class SplineRefiner:
    """Public xyz <-> (theta, h, d) adapter with the common reference depth."""

    def __init__(self, tube, depth):
        self.tube, self.depth = tube, depth

    def p_to_xi(self, points):
        h, theta, r = np.moveaxis(self.tube.inverse(points), -1, 0)
        return np.stack((theta, h, self.depth.evaluate(h, theta, r)), axis=-1)

    def xi_to_p(self, addresses):
        addresses = np.asarray(addresses, dtype=float)
        if addresses.shape[-1:] != (3,) or not np.isfinite(addresses).all():
            raise ValueError("material addresses must end in three finite coordinates")
        theta, h, d = np.moveaxis(addresses, -1, 0)
        r = self.depth.inverse(h, theta, d)
        return self.tube.forward(np.stack((h, theta, r), axis=-1))

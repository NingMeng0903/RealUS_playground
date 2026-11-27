"""Neutral right-leg annulus in (azimuth, world Y in mm, inward depth)."""
from functools import cache
from io import BytesIO

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.special import expit, logit

from .anatomy_assets import BONE_OCCUPANCY_PATH, NEUTRAL_SMPLX_SURFACE_PATH, verify_neutral_smplx_surface_sha256
from .smplx_dynamic_atlas import DEFAULT_MODEL_PATH


_STEP_H = 5.
_N_THETA = 128
# Includes proximal/distal drift margins beyond the graph's nominal stations.
_H = np.arange(-1320., -475. + _STEP_H, _STEP_H)
_THETA = np.linspace(-np.pi, np.pi, _N_THETA + 1)
_CLEARANCE = 6.


@njit
def _skin_rays(h, theta, origins, d_origins, triangles, starts, ids):
    """First positive intersection with a horizontal mesh slice, and derivatives."""
    cut = np.empty((2, 2))
    slope = np.empty((2, 2))
    out = np.empty((len(h), 3))
    for i, y in enumerate(h):
        cell = min(int((y - _H[0]) / _STEP_H), len(starts) - 2)
        ux, uz = np.cos(theta[i]), np.sin(theta[i])
        radius, dtheta, dh = np.inf, 0., 0.
        for index in range(starts[cell], starts[cell + 1]):
            tri = triangles[ids[index]]
            n = 0
            for j in range(3):
                a, b = tri[j], tri[(j + 1) % 3]
                if (a[1] > y) != (b[1] > y):
                    sy = 1. / (b[1] - a[1])
                    slope[n, 0] = (b[0] - a[0]) * sy
                    slope[n, 1] = (b[2] - a[2]) * sy
                    cut[n, 0] = a[0] + (y - a[1]) * slope[n, 0]
                    cut[n, 1] = a[2] + (y - a[1]) * slope[n, 1]
                    n += 1
            if n != 2:
                continue
            qx, qz = cut[0, 0] - origins[i, 0], cut[0, 1] - origins[i, 1]
            ex, ez = cut[1, 0] - cut[0, 0], cut[1, 1] - cut[0, 1]
            den = ux * ez - uz * ex
            if abs(den) < 1e-12:
                continue
            t = (qx * uz - qz * ux) / den
            if -1e-12 <= t <= 1. + 1e-12:
                r = (qx * ez - qz * ex) / den
                if 0. < r < radius:
                    dqx, dqz = slope[0, 0] - d_origins[i, 0], slope[0, 1] - d_origins[i, 1]
                    dex, dez = slope[1, 0] - slope[0, 0], slope[1, 1] - slope[0, 1]
                    radius = r
                    dtheta = r * (uz * ez + ux * ex) / den
                    dh = (dqx * ez - dqz * ex + qx * dez - qz * dex - r * (ux * dez - uz * dex)) / den
        if ux > 0.:
            plane = (-2. - origins[i, 0]) / ux
            if plane < radius:
                radius, dtheta, dh = plane, plane * uz / ux, -d_origins[i, 0] / ux
        out[i, 0], out[i, 1], out[i, 2] = radius, dtheta, dh
    return out


@njit
def _box_ray_interval(lo_x, hi_x, lo_z, hi_z, ux, uz):
    """Positive-ray interval inside a 2D box; empty when hi < lo."""
    lo, hi = 0., np.inf
    for a, b, u in ((lo_x, hi_x, ux), (lo_z, hi_z, uz)):
        if abs(u) < 1e-14:
            if a > 0. or b < 0.:
                return 0., -1.
        else:
            u0, u1 = a / u, b / u
            lo, hi = max(lo, min(u0, u1)), min(hi, max(u0, u1))
    return lo, hi


@njit
def _sample_bone_radius(origin, theta, skin, voxels):
    """Sample voxel exits for the nominal spline, not the safety floor."""
    out = np.zeros(len(theta))
    for i, angle in enumerate(theta):
        ux, uz = np.cos(angle), np.sin(angle)
        for x, z in voxels:
            x, z = x - origin[0], z - origin[1]
            lo, hi = _box_ray_interval(x - 1.5, x + 1.5, z - 1.5, z + 1.5, ux, uz)
            if hi >= lo and lo < skin[i]:
                out[i] = max(out[i], hi)
    return out


@njit
def _bone_cell_bounds(axis_min, axis_max, slab_ids, starts, voxels, angular_bins):
    """Conservative max radius of every occupied box in each angular wedge."""
    delta = 2. * np.pi / angular_bins
    angles = np.arange(angular_bins) * delta - np.pi
    ux, uz = np.cos(angles), np.sin(angles)
    out = np.zeros((len(slab_ids), angular_bins))
    for i, slab in enumerate(slab_ids):
        for k in range(starts[slab], starts[slab + 1]):
            x, z = voxels[k]
            # Expand the voxel by the extrema of axis movement in this slab
            # cell. Clip only to the chart's exact right-hemibody cap.
            a, b = x - 1.5 - axis_max[i, 0], min(x + 1.5, -2.) - axis_min[i, 0]
            c, d = z - 1.5 - axis_max[i, 1], z + 1.5 - axis_min[i, 1]
            if x - 1.5 > -2.:
                continue
            center = np.arctan2(c + d, a + b)
            first, last = 0, angular_bins - 1
            min_angle, max_angle = np.inf, -np.inf
            contains_origin = a <= 0. <= b and c <= 0. <= d
            for vx, vz in ((a, c), (a, d), (b, c), (b, d)):
                angle = np.arctan2(vz, vx)
                rel = (angle - center + np.pi) % (2. * np.pi) - np.pi
                min_angle, max_angle = min(min_angle, rel), max(max_angle, rel)
                j = int(np.floor((angle + np.pi) / delta)) % angular_bins
                out[i, j] = max(out[i, j], np.hypot(vx, vz))
            if not contains_origin:
                first = int(np.floor((center + min_angle + np.pi) / delta - 1e-10))
                last = int(np.floor((center + max_angle + np.pi) / delta + 1e-10))
            # A wedge-clipped rectangle has only box corners and ray/box
            # intersections as vertices. A convex norm attains its maximum
            # on those vertices, so interior rays need no point sampling.
            for j in range(first, last + 2):
                m = j % angular_bins
                lo, hi = _box_ray_interval(a, b, c, d, ux[m], uz[m])
                exit_r = hi if hi >= lo else 0.
                out[i, m] = max(out[i, m], exit_r)
                prev = (m - 1) % angular_bins
                out[i, prev] = max(out[i, prev], exit_r)
    return out


class _ContinuousBoneEnvelope:
    """Periodic C2 upper envelope of the raw thresholded voxel geometry."""

    def __init__(self, axis, raster):
        edges, starts, voxels = raster
        # Partition at every slab edge and spline breakpoint. On each
        # subcell the axis is one cubic, bounded by its Bézier control hull.
        heights = np.unique(np.r_[_H[0], _H[-1], axis.x, np.arange(edges[0], edges[-1], .5)])
        self.heights = heights = heights[(_H[0] <= heights) & (heights <= _H[-1])]
        width = np.diff(heights)
        center, slope = axis(heights), axis(heights, 1)
        hull = np.stack((center[:-1], center[:-1] + width[:, None] * slope[:-1] / 3.,
                         center[1:] - width[:, None] * slope[1:] / 3., center[1:]))
        self.angular_bins = 1024
        slabs = np.searchsorted(edges, heights[:-1] + width / 2., side='right') - 1
        cells = _bone_cell_bounds(hull.min(axis=0), hull.max(axis=0), slabs, starts, voxels, self.angular_bins)
        corner = np.maximum(cells, np.roll(cells, 1, axis=1))
        self.vertices = np.vstack((corner[:1], np.maximum(corner[:-1], corner[1:]), corner[-1:]))

    @staticmethod
    def _weight(t):
        return t**3 * (10. - 15. * t + 6. * t**2), 30. * t**2 * (1. - t)**2

    def evaluate(self, h, theta):
        hi = np.clip(np.searchsorted(self.heights, h, side='right') - 1, 0, len(self.heights) - 2)
        dh = self.heights[hi + 1] - self.heights[hi]
        u, du = self._weight((h - self.heights[hi]) / dh)
        angular_scale = self.angular_bins / (2. * np.pi)
        position = (theta + np.pi) * angular_scale
        j = np.floor(position).astype(np.intp)
        v, dv = self._weight(position - j)
        j %= self.angular_bins
        nxt = (j + 1) % self.angular_bins
        a, b = self.vertices[hi, j], self.vertices[hi, nxt]
        c, d = self.vertices[hi + 1, j], self.vertices[hi + 1, nxt]
        left, right = a + u * (c - a), b + u * (d - b)
        # Tensor-product smoothstep is still a convex interpolation of
        # corners. Every corner bounds all cells incident to that corner.
        return np.column_stack((left + v * (right - left),
                                angular_scale * dv * (right - left),
                                du / dh * ((1. - v) * (c - a) + v * (d - b))))


class _PeriodicSurface:
    """Tensor product of a C2 periodic angular spline and a C2 height spline."""

    def __init__(self, values):
        angular = CubicSpline(_THETA, np.c_[values, values[:, 0]], axis=1, bc_type='periodic')
        self.coefficients = CubicSpline(_H, angular.c, axis=2).c

    def evaluate(self, h, theta):
        i = np.clip(np.searchsorted(_H, h, side='right') - 1, 0, len(_H) - 2)
        j = np.clip(np.searchsorted(_THETA, theta, side='right') - 1, 0, _N_THETA - 1)
        u, v = (h - _H[i])[None, :], theta - _THETA[j]
        # Coefficient axes are (height degree, height cell, angle degree, angle cell).
        a, b, c, d = self.coefficients[:, i, :, j].transpose(1, 2, 0)
        p = ((a * u + b) * u + c) * u + d
        dp = (3. * a * u + 2. * b) * u + c
        value = ((p[0] * v + p[1]) * v + p[2]) * v + p[3]
        dtheta = (3. * p[0] * v + 2. * p[1]) * v + p[2]
        dh = ((dp[0] * v + dp[1]) * v + dp[2]) * v + dp[3]
        return np.column_stack((value, dtheta, dh))


@cache
def _canonical_geometry(skin_bytes):
    with np.load(BytesIO(skin_bytes), allow_pickle=False) as asset:
        triangles = asset['vertices'][asset['triangles']].astype(np.float64)
    tri_y = triangles[:, :, 1]
    bottom, top = tri_y.min(axis=1), tri_y.max(axis=1)
    buckets = [np.flatnonzero((bottom <= h + _STEP_H) & (top >= h)) for h in _H[:-1]]
    mesh = triangles, np.r_[0, np.cumsum([len(b) for b in buckets])], np.concatenate(buckets)

    with np.load(DEFAULT_MODEL_PATH, allow_pickle=False) as model:
        joints = (model['J_regressor'][[2, 5, 8]].astype(float) @ model['v_template']) * 1000.
    if not np.allclose(joints, [[-56.70, -472.01, -3.89], [-107.43, -851.83, -18.34],
                               [-90.27, -1286.74, -58.28]], atol=.02, rtol=0):
        raise ValueError('neutral SMPL-X joints do not register to the right-leg training assets')

    with np.load(BONE_OCCUPANCY_PATH, allow_pickle=False) as asset:
        occupied = asset['count'] >= .55 * int(asset['cohort_size'])
        x, y, z = (asset[f'{axis}_mm'] for axis in 'xyz')
    if not all(np.all(np.diff(v) == 3.) for v in (x, y, z)):
        raise ValueError('expected 3 mm bone proxy voxels')
    voxels = []
    for i in range(len(y)):
        rows, cols = np.nonzero(occupied[:, i, :])
        voxels.append(np.column_stack((x[rows], z[cols])))
    raster = np.r_[y - 1.5, y[-1] + 1.5], np.r_[0, np.cumsum([len(v) for v in voxels])], np.concatenate(voxels)
    anchors = [*joints]
    for h in (-1320., -1250., -1200., -1150., -1100., -1050., -1000., -950., -900.,
              -810., -760., -710., -660., -610., -560., -525., -500.):
        rows, cols = np.nonzero(occupied[:, np.argmin(abs(y - h)), :])
        anchors.append([x[rows].mean(), h, z[cols].mean()])
    anchors = np.asarray(sorted(anchors, key=lambda p: p[1]))
    axis = CubicSpline(anchors[:, 1], anchors[:, [0, 2]])
    hh = np.repeat(_H, _N_THETA)
    angles = np.tile(_THETA[:-1], len(_H))
    raw_skin = _skin_rays(hh, angles, axis(hh), axis(hh, 1), *mesh)[:, 0].reshape(len(_H), -1)
    if not np.isfinite(raw_skin).all() or raw_skin.min() <= _CLEARANCE + 5.:
        raise ValueError('centerline does not admit a right-leg annulus on the chart grid')

    raw_bone = []
    outer_bone = []
    for h, origin, skin in zip(_H, axis(_H), raw_skin, strict=True):
        slab = np.searchsorted(raster[0], h, side='right') - 1
        exact = _sample_bone_radius(origin, _THETA[:-1], skin, voxels[slab])
        raw_bone.append(np.maximum(4., exact + 2.))
        # Guard sub-grid features in both dimensions by including voxel cells
        # whose slabs overlap this sample's half-height neighborhood.
        rows, cols = np.nonzero(occupied[:, abs(y - h) <= 4., :].any(axis=1))
        nearby = np.column_stack((x[rows], z[cols]))
        rays = _sample_bone_radius(origin, _THETA[:-1], skin, nearby)
        outer_bone.append(np.maximum(4., np.maximum.reduce((rays, np.roll(rays, 1), np.roll(rays, -1))) + 3.5))
    raw_bone = np.asarray(raw_bone)
    reference = CubicSpline(_H, ((raw_skin + np.minimum(raw_bone, raw_skin - _CLEARANCE)) * .5).mean(axis=1))
    return (axis, mesh, _ContinuousBoneEnvelope(axis, raster), _PeriodicSurface(raw_skin - 1.),
            _PeriodicSurface(raw_bone), _PeriodicSurface(np.asarray(outer_bone)), reference)


class RightLegMaterialChart:
    """Deterministic annular chart; h is physical Y and theta is periodic.

    The bone floor is a conservative C2 envelope. Switches between bounds
    and mesh facets are piecewise differentiable; Jacobians are those of
    the selected branch away from switches.
    """

    h_bounds_mm = (float(_H[0]), float(_H[-1]))
    d_min = .04
    d_max = .94

    def __init__(self):
        skin_bytes = NEUTRAL_SMPLX_SURFACE_PATH.read_bytes()
        verify_neutral_smplx_surface_sha256(skin_bytes)
        (self._axis, self._mesh, self._bone_envelope, self._skin, self._bone,
         self._outer_bone, self._reference) = _canonical_geometry(skin_bytes)

    @staticmethod
    def proximal_limit_mm(theta):
        """Upper height-tail onset and derivative on the traversable right thigh.

        Medial rays close to the hemibody cut encounter pelvic occupancy
        patches, not a single skin-to-bone compartment. Stop the generated
        vessel chart below the patches in those directions while keeping
        the anterior groin and all observed proximal stations in the chart.
        """
        degrees = np.rad2deg((theta + np.pi) % (2. * np.pi) - np.pi)
        limit = np.full_like(theta, -480.)
        derivative = np.zeros_like(theta)
        for lo, hi, shoulder, reduction in ((-3., 13., 8., 46.), (22., 43., 8., 15.)):
            u = np.minimum((degrees - lo) / shoulder + 1., (hi - degrees) / shoulder + 1.)
            t = np.clip(u, 0., 1.)
            limit -= reduction * t**3 * (10. - 15. * t + 6. * t**2)
            dt = np.copysign(np.rad2deg(1.) / shoulder, (lo + hi) / 2. - degrees)
            derivative -= 30. * reduction * t**2 * (1. - t)**2 * dt
        return limit, derivative

    @staticmethod
    def _rows(values):
        array = np.asarray(values, dtype=float)
        if not array.ndim or array.shape[-1] != 3 or not np.isfinite(array).all():
            raise ValueError('chart coordinates must be finite with shape (...,3)')
        return array, array.reshape(-1, 3)

    def _boundaries(self, theta, h, origins, slopes):
        lo, hi = self.h_bounds_mm
        if np.any((h < lo) | (h > hi)):
            raise ValueError(f'material h must be in [{lo}, {hi}] mm')
        theta = (theta + np.pi) % (2. * np.pi) - np.pi
        skin = self._skin.evaluate(h, theta)
        raw = _skin_rays(h, theta, origins, slopes, *self._mesh)
        skin_guard = raw - [1., 0., 0.]
        skin = np.where((skin_guard[:, 0] < skin[:, 0])[:, None], skin_guard, skin)
        base = self._bone.evaluate(h, theta)
        outer = self._outer_bone.evaluate(h, theta)
        bone = np.where((base[:, 0] > outer[:, 0])[:, None], base, outer)
        core = np.array([4., 0., 0.])
        bone = np.where((bone[:, 0] >= core[0])[:, None], bone, core)
        safe = self._bone_envelope.evaluate(h, theta) + [.05, 0., 0.]
        ceiling = raw - [.05, 0., 0.]
        room = np.where((bone[:, 0] >= safe[:, 0])[:, None], bone, safe) + [_CLEARANCE, 0., 0.]
        room = np.where((room[:, 0] < ceiling[:, 0])[:, None], room, ceiling)
        skin = np.where((skin[:, 0] < room[:, 0])[:, None], room, skin)
        # Relax surplus spline clearance, but never by moving through raw
        # bone. Nonpositive signed width is exposed as unsupported.
        width = skin - base + [1.5, 0., 0.]
        width = np.where((width[:, 0] < _CLEARANCE)[:, None], width, [_CLEARANCE, 0., 0.])
        width = np.where((width[:, 0] > 2.)[:, None], width, [2., 0., 0.])
        limited = skin - width
        bone = np.where((bone[:, 0] <= limited[:, 0])[:, None], bone, limited)
        bone = np.where((safe[:, 0] > bone[:, 0])[:, None], safe, bone)
        return skin, bone

    def corridor_supported(self, theta, h):
        """Whether a ray is in-domain and has positive guarded corridor width."""
        theta, h = np.broadcast_arrays(theta, h)
        lo, hi = self.h_bounds_mm
        supported = np.asarray(np.isfinite(theta) & (lo <= h) & (h <= hi))
        if supported.any():
            heights = h[supported]
            origins, slopes = self._axis(heights), self._axis(heights, 1)
            angles = (theta[supported] + np.pi) % (2. * np.pi) - np.pi
            raw = _skin_rays(heights, angles, origins, slopes, *self._mesh)[:, 0]
            bone = self._bone_envelope.evaluate(heights, angles)[:, 0]
            supported[supported] = raw - .05 > bone + .05
        return supported

    def count_unsupported_world_rays(self, xyz_mm):
        """Count audit points on unsupported annular rays, independently of depth."""
        h = xyz_mm[:, 1]
        x, z = (xyz_mm[:, [0, 2]] - self._axis(h)).T
        return int(np.count_nonzero(~self.corridor_supported(np.arctan2(z, x), h)))

    def _geometry(self, rows):
        theta, h, d = rows.T
        origins, slopes = self._axis(h), self._axis(h, 1)
        skin, bone = self._boundaries(theta, h, origins, slopes)
        return origins, slopes, skin, bone, skin + d[:, None] * (bone - skin)

    def xi_to_world(self, xi):
        """Decode addresses; unsupported rays are only an audit continuation."""
        array, rows = self._rows(xi)
        theta, h, _ = rows.T
        origins, _, _, _, r = self._geometry(rows)
        return np.column_stack((origins[:, 0] + r[:, 0] * np.cos(theta), h,
                                origins[:, 1] + r[:, 0] * np.sin(theta))).reshape(array.shape)

    def world_to_xi(self, xyz_mm):
        array, rows = self._rows(xyz_mm)
        h = rows[:, 1]
        origins = self._axis(h)
        x, z = (rows[:, [0, 2]] - origins).T
        theta = np.arctan2(z, x)
        skin, bone = self._boundaries(theta, h, origins, self._axis(h, 1))
        d = (skin[:, 0] - np.hypot(x, z)) / (skin[:, 0] - bone[:, 0])
        return np.column_stack((theta, h, d)).reshape(array.shape)

    def d_from_eta(self, eta):
        return self.d_min + (self.d_max - self.d_min) * expit(eta)

    def eta_from_d(self, d):
        clipped = np.clip(d, self.d_min + 1e-4, self.d_max - 1e-4)
        return logit((clipped - self.d_min) / (self.d_max - self.d_min))

    def xi_from_tilde_xi(self, tilde_xi):
        array, _ = self._rows(tilde_xi)
        xi = array.copy()
        xi[..., 2] = self.d_from_eta(xi[..., 2])
        return xi

    def tilde_xi_to_world(self, tilde_xi):
        return self.xi_to_world(self.xi_from_tilde_xi(tilde_xi))

    def encode_with_audit(self, xyz_mm):
        """Return latent address, raw inverse address and explicit projection distances."""
        xi = self.world_to_xi(xyz_mm)
        tilde = xi.copy()
        tilde[..., 2] = self.eta_from_d(xi[..., 2])
        lo, hi = self.d_min + 1e-4, self.d_max - 1e-4
        ids = np.flatnonzero(((xi[..., 2] < lo) | (xi[..., 2] > hi)).ravel())
        residual = np.linalg.norm(self.tilde_xi_to_world(tilde) - xyz_mm, axis=-1).ravel()[ids]
        return tilde, xi, {
            'n_knots': xi.size // 3,
            'n_projected_knots': int(ids.size),
            'median_projection_residual_mm': float(np.median(residual)) if ids.size else 0.,
            'max_projection_residual_mm': float(residual.max(initial=0.)),
            'projected_knot_indices': ids.tolist(),
            'projection_residual_mm': residual.tolist(),
        }

    def dworld_dxi(self, xi):
        array, rows = self._rows(xi)
        theta = rows[:, 0]
        _, slopes, skin, bone, r = self._geometry(rows)
        u = np.column_stack((np.cos(theta), np.sin(theta)))
        tangent = np.column_stack((-u[:, 1], u[:, 0]))
        jac = np.zeros((len(rows), 3, 3))
        jac[:, [0, 2], 0] = r[:, :1] * tangent + r[:, 1:2] * u
        jac[:, [0, 2], 1] = slopes + r[:, 2:3] * u
        jac[:, 1, 1] = 1.
        jac[:, [0, 2], 2] = (bone[:, :1] - skin[:, :1]) * u
        return jac.reshape(array.shape[:-1] + (3, 3))

    def dworld_dtilde_xi(self, tilde_xi):
        xi = self.xi_from_tilde_xi(tilde_xi)
        result = self.dworld_dxi(xi)
        s = (xi[..., 2] - self.d_min) / (self.d_max - self.d_min)
        result[..., :, 2] *= ((self.d_max - self.d_min) * s * (1. - s))[..., None]
        return result

    def nominal_reference_radius(self, h):
        return self._reference(h)

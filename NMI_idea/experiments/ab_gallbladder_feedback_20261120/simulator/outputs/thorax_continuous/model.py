"""SMPL-X material transport and phase-conditioned cardiac and rib belief."""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "thorax_a4c_pilot"))
from geometry import Anatomy, BEAM_ANGLE, DEPTH_MM, PREVIEW, SOURCE
from generate import Bones, PeriodicCubic


from gallbladder_runtime import ATLAS_ARCHIVE, SMPLX_ARCHIVE

SMPLX = SMPLX_ARCHIVE
READOUT = np.linspace(-BEAM_ANGLE, BEAM_ANGLE, 61)
ECHO = np.arange(1, len(READOUT), 4)
TARGET_PHASE = np.array([.2083333333, .5])
PHASE_SD = .012
RIB_MODE = np.array(list(product([-6., -3., 0., 3., 6.], [-6., -3., 0., 3., 6.])))
INITIAL_SD = np.array([8., 6., 5., .075, .20, .22])
MEASURE_SD = np.array([.115, 1.6, 1.65, 1.75])
HEART_BOUNDS = np.array([[-22., -20., -17., -.24, -.65, -.65], [22., 20., 17., .24, .65, .65]])
TIMED_BOUNDS = np.column_stack((HEART_BOUNDS, [[-.11, -.11], [.11, .11]]))
HEART_PROCESS = np.array([.017, .017, .017, .000018, .000060, .000060, .000002, .000002])
CONTACT = np.array([[-182., -57., -17., -13.], [-124., -24., 17., 13.]])


@dataclass(frozen=True)
class View:
    chart: np.ndarray
    address: np.ndarray
    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    normal: np.ndarray
    bone_distance: np.ndarray

    def beam_blocked(self, x, y):
        bearing = np.arctan2(x, y)
        q = np.rint((bearing / BEAM_ANGLE + 1.) * (len(self.bone_distance) - 1) / 2.).astype(int)
        return np.hypot(x, y) >= self.bone_distance[q.clip(0, len(self.bone_distance) - 1)]


class Reference:
    def __init__(self):
        self.anatomy = Anatomy()
        self.atlas = self.anatomy.atlas
        with np.load(SMPLX, allow_pickle=False) as model, np.load(SOURCE, allow_pickle=False) as source:
            assert np.array_equal(model["f"], source["skin_faces"])
            shape = model["shapedirs"][..., :10].astype(float)
            delta = source["skin_vertices"] - model["v_template"]
            self.beta = np.linalg.lstsq(shape.reshape(-1, 10), delta.ravel(), rcond=None)[0]
            error = delta - np.einsum("vci,i->vc", shape, self.beta)
            self.smplx_fit_mm = float(1000. * np.sqrt(np.mean(error ** 2)))
            self.shape = shape[..., :2] * 1000.
            face = source["skin_faces"][np.maximum(self.atlas.face, 0)]
        weights = np.nan_to_num(self.atlas.bary)
        self.rim = np.einsum("hmv,hmvcd->hmcd", weights, self.shape[face])
        for i, ring in enumerate(self.rim):
            actual = self.atlas.face[i] >= 0
            for coord, component in product(range(3), range(2)):
                ring[~actual, coord, component] = np.interp(self.atlas.theta[~actual], self.atlas.theta[actual],
                                                            ring[actual, coord, component], period=2. * np.pi)
        with np.load(ATLAS_ARCHIVE, allow_pickle=False) as data:
            self.bone_points = data["bone_vertices_mm"]
            self.bone_faces, self.bone_ids = data["bone_faces"], data["bone_face_id"]


class Body(Anatomy):
    """Known case SMPL-X beta with reference chart addresses carried to its skin."""

    def __init__(self, reference, beta_delta):
        self.atlas = reference.atlas
        self.beta_delta = np.asarray(beta_delta)
        rim = reference.rim @ self.beta_delta
        self.rim = PeriodicCubic(self.atlas.h, self.atlas.theta, rim)
        self.core = CubicSpline(self.atlas.h, rim.mean(axis=1), axis=0)
        self.skin_change_max_mm = float(np.linalg.norm(rim, axis=-1).max())
        self.skin_change_rms_mm = float(np.sqrt(np.mean(np.sum(rim ** 2, axis=-1))))
        anatomy = reference.anatomy
        self.phase, self.volume, self.motion = anatomy.phase, anatomy.volume, anatomy.motion
        self.shortening = anatomy.shortening
        self.rest = self.warp(anatomy.rest)
        # Push the three chamber axes through the material deformation and
        # retain their transported lengths and a proper orthogonal frame.
        unit = anatomy.rotation.transpose(0, 2, 1)
        chord = self.warp(anatomy.rest[:, None] + 2. * unit) - self.warp(anatomy.rest[:, None] - 2. * unit)
        self.rest_axes = anatomy.rest_axes * np.linalg.norm(chord, axis=-1) / 4.
        u, _, vt = np.linalg.svd(chord.transpose(0, 2, 1))
        self.rotation = u @ vt
        assert np.min(np.linalg.det(self.rotation)) > .99
        self.midpoint = self.rest.mean(axis=0)
        self.plane_normal = np.linalg.svd(self.rest - self.midpoint, full_matrices=False)[2][-1]
        self.bones = Bones(self.warp(reference.bone_points), reference.bone_faces, reference.bone_ids)
        self.echo_cache = {}
        self.fan_cache = {}

    def warp(self, point):
        h, theta, rho = self.atlas.to_disk(np.asarray(point))
        core = self.core(h)
        return point + core + rho[..., None] ** 3 * (self.rim.evaluate(h, theta) - core)

    def surface(self, chart):
        q = np.asarray(chart)
        h = (q[..., 0] - self.atlas.y0) / (self.atlas.y1 - self.atlas.y0)
        theta = np.deg2rad(q[..., 1])
        return self.atlas.to_world(h, theta, np.ones_like(h)) + self.rim.evaluate(h, theta)

    def view(self, chart, rib=None):
        q = np.asarray(chart)
        h = (q[0] - self.atlas.y0) / (self.atlas.y1 - self.atlas.y0)
        theta, roll, rock = np.deg2rad(q[1:])
        origin = self.surface(q)
        forward = self.midpoint - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(self.plane_normal, forward)
        right /= np.linalg.norm(right)
        if right[0] < 0.:
            right = -right
        right = np.cos(roll) * right + np.sin(roll) * np.cross(forward, right)
        forward = np.cos(rock) * forward + np.sin(rock) * np.cross(forward, right)
        normal = np.cross(forward, right)
        address = np.array([h, np.sin(theta), np.cos(theta)])
        view = View(q, address, origin, forward, right, normal, np.full(len(READOUT), np.inf))
        return view if rib is None else self.with_ribs(view, rib)

    def rays(self, view, offset, bearings=READOUT):
        y, z = offset
        rays = np.cos(bearings[:, None]) * view.forward + np.sin(bearings[:, None]) * view.right
        return self.bones.hits(view.origin - [0., y, z], rays)[0]

    def with_ribs(self, view, offset):
        key = (*view.chart, *offset)
        if key not in self.fan_cache:
            if len(self.fan_cache) >= 4096:
                self.fan_cache.clear()
            self.fan_cache[key] = self.rays(view, offset)
        return replace(view, bone_distance=self.fan_cache[key])

    def trajectories(self, parameters, phase, *, smooth_periodic=False):
        p = np.atleast_2d(parameters)
        phases = np.atleast_1d(phase)
        if p.shape[1] > 6:
            chamber_phase = (phases[None, :, None] - p[:, None, [6, 6, 7, 7]]) % 1.
            motion_phase = (phases[None, :] - p[:, 7, None]) % 1.
        else:
            chamber_phase = np.broadcast_to(phases[None, :, None], (1, len(phases), 4))
            motion_phase = phases[None]
        if smooth_periodic:
            if not hasattr(self, "_periodic_cardiac"):
                # Log volume retains positive axes. Periodic endpoint
                # derivatives make the mechanical velocity continuous at wrap.
                volumes = [CubicSpline(self.phase, np.log(v), bc_type="periodic") for v in self.volume.T]
                motion = CubicSpline(self.phase, self.motion, axis=0, bc_type="periodic")
                self._periodic_cardiac = volumes, motion
            volumes, motion = self._periodic_cardiac
            log_volume = np.stack([curve(chamber_phase[..., i]) for i, curve in enumerate(volumes)], axis=-1)
            displacement = motion(motion_phase)
        else:
            log_volume = np.log(np.stack([np.interp(chamber_phase[..., i], self.phase, self.volume[:, i])
                                          for i in range(4)], axis=-1))
            displacement = np.stack([np.interp(motion_phase, self.phase, self.motion[:, i])
                                     for i in range(3)], axis=-1)
        gain = np.exp(p[:, [4, 4, 5, 5]])
        volume = np.exp(gain[:, None] * log_volume)
        axes = self.rest_axes * np.exp(p[:, None, 3, None, None]) * volume[..., None] ** self.shortening
        centers = self.rest + p[:, None, None, :3] + np.exp(p[:, 5])[:, None, None, None] * displacement[:, :, None]
        return centers, axes, volume

    def moments(self, view, parameters, phases):
        c, s, _ = self.trajectories(parameters, phases)
        return self.predict(view, c, s)

    def goal_metrics(self, view, parameters):
        _, central, length, apex, apical, outline = self.moments(view, parameters, TARGET_PHASE)
        minimum = lambda x: np.min(x, axis=tuple(range(1, x.ndim)))
        return np.column_stack((minimum(central), minimum(length[..., 2:]), np.max(apex, axis=1),
                                np.all(apical, axis=1), minimum(outline)))

    def raster(self, view, centers, axes, size=196):
        return super().raster(view, centers, axes, size)


def passes(metrics):
    coverage, length, apex, approach, outline = np.asarray(metrics).T
    return (coverage >= .40) & (length >= .84) & (apex <= 9.) & (approach > .5) & (outline >= .80)


def likelihood_margin(metrics):
    c, length, apex, approach, outline = np.asarray(metrics).T
    # Smooth penalties retain a direction even when a current slice misses the target.
    values = np.column_stack(((c - .40) / .18, (length - .84) / .10, (9. - apex) / 7., (outline - .80) / .15))
    return -logsumexp(-3. * values, axis=1) / 3. - 1.1 * (1. - approach)


@dataclass(frozen=True)
class Belief:
    mean: np.ndarray
    covariance: np.ndarray
    rib: np.ndarray

    @classmethod
    def prior(cls, *, timed=False):
        sd = np.r_[INITIAL_SD, .036, .034] if timed else INITIAL_SD
        return cls(np.zeros(len(sd)), np.diag(sd ** 2), np.full(len(RIB_MODE), 1. / len(RIB_MODE)))

    def bounds(self):
        return TIMED_BOUNDS[:, :len(self.mean)]

    def points(self):
        # Equal-weight cubature approximates the continuous Gaussian posterior.
        chol = np.linalg.cholesky(self.covariance)
        offsets = np.sqrt(len(self.mean)) * chol.T
        return np.clip(np.vstack((self.mean + offsets, self.mean - offsets)), *self.bounds())

    def bone_mean(self):
        return self.rib @ RIB_MODE

    def bone_std(self):
        return np.sqrt(np.maximum(self.rib @ (RIB_MODE - self.bone_mean()) ** 2, 0.))


@dataclass(frozen=True)
class Case:
    id: int
    beta: np.ndarray
    cardiac: np.ndarray
    rib: np.ndarray
    start: np.ndarray
    bpm: float
    phase_origin: float


@dataclass(frozen=True)
class Frame:
    phase: float
    y: np.ndarray
    echo: np.ndarray
    seen: np.ndarray


def make_cases(n, seed, start_id):
    rng = np.random.default_rng(seed)
    beta = rng.uniform([-.60, -.32], [.60, .32], (n, 2))
    cardiac = rng.uniform([-9., -6., -7., -.057, -.20, -.26], [9., 6., 7., .057, .20, .26], (n, 6))
    rib = rng.uniform(-5.6, 5.6, (n, 2))
    start = np.clip(np.array([-148., -39., 0., 0.]) + rng.normal(size=(n, 4)) * [12., 5.5, 6.5, 5.], *CONTACT)
    bpm, origin = rng.uniform(61., 86., n), rng.uniform(0., 1., n)
    return [Case(start_id + i, beta[i], cardiac[i], rib[i], start[i], bpm[i], origin[i]) for i in range(n)]


def phase_at(case, tick):
    return (case.phase_origin + np.asarray(tick) * (case.bpm / 600.)) % 1.


def det_prob(area):
    return .012 + .942 * (1. - np.exp(-area / 115.))


class Sensors:
    def __init__(self, body, case, replica, *, frames=240):
        self.body, self.case = body, case
        rng = np.random.default_rng([351747, case.id, replica])
        self.phase_error = rng.normal(0., PHASE_SD, frames)
        self.feature_noise = rng.standard_normal((frames, 4, 4)) * MEASURE_SD
        self.detect = rng.random((frames, 4))
        self.echo_noise = rng.normal(0., 1.25, (frames, len(ECHO)))
        self.echo_lost = rng.random((frames, len(ECHO))) < .09
        outage = rng.random(frames) < .024
        self.outage = outage | np.roll(outage, 1)

    def sample(self, view, tick):
        true = self.body.with_ribs(view, self.case.rib)
        phi = phase_at(self.case, tick)
        features = self.body.moments(true, self.case.cardiac, phi)[0][0, 0]
        seen = (self.detect[tick] < det_prob(features[:, 0])) & (not self.outage[tick])
        features[:, 0] = np.log1p(features[:, 0] / 75.)
        y = np.where(seen[:, None], features + self.feature_noise[tick], np.nan)
        depths = true.bone_distance[ECHO]
        has_echo = (depths < 82.) & ~self.echo_lost[tick] & (not self.outage[tick])
        echo = np.where(has_echo, depths + self.echo_noise[tick], np.nan)
        return Frame(float((phi + self.phase_error[tick]) % 1.), y, echo, seen)


def rib_update(body, view, b, echo):
    seen = np.isfinite(echo)
    if not np.any(seen):
        return b
    key = tuple(view.chart)
    if key not in body.echo_cache:
        if len(body.echo_cache) >= 2048:
            body.echo_cache.clear()
        body.echo_cache[key] = np.array([body.rays(view, x, READOUT[ECHO]) for x in RIB_MODE])
    predictions = body.echo_cache[key][:, seen]
    delta = (np.minimum(predictions, 200.) - echo[seen]) / 2.75
    log_l = np.maximum(-.5 * delta ** 2, -3.).sum(axis=-1)
    # A conservative likelihood limits the influence of correlated echoes
    # at adjacent scan lines and retains support between coarse rib offsets.
    prior = np.log(.996 * b + .004 / len(b))
    weight = prior + .42 * log_l
    return np.exp(weight - logsumexp(weight))


def heart_update(body, view, b, frame):
    points = b.points()
    raw, _, _, _, _, intact = body.moments(view, points, frame.phase)
    members = raw[:, 0]
    active = frame.seen & (np.mean(intact[:, 0], axis=0) >= .75) & (np.mean(members[:, :, 0], axis=0) > 90.)
    if not np.any(active):
        return b.mean, b.covariance
    m = members[:, active].copy()
    m[:, :, 0] = np.log1p(m[:, :, 0] / 75.)
    predict = m.reshape(len(points), -1)
    observed = frame.y[active].ravel()
    dz = predict - predict.mean(axis=0)
    dx = points - points.mean(axis=0)
    cs = dz.T @ dz / len(points)
    cross = dx.T @ dz / len(points)
    sigma = np.tile([.18, 2.2, 2.2, 2.5], active.sum())
    residual = observed - predict.mean(axis=0)
    total_sd = np.sqrt(np.diag(cs) + sigma ** 2)
    # Keep noisy detections near the edge of a window from collapsing the posterior.
    inflation = np.clip(np.abs(residual) / (2.2 * total_sd), 1., 8.)
    innovation = cs + np.diag((sigma * inflation) ** 2)
    gain = np.linalg.solve(innovation, cross.T).T
    mu = np.clip(b.mean + gain @ residual, *b.bounds())
    cov = b.covariance - gain @ cross.T
    cov = .5 * (cov + cov.T) + np.diag(HEART_PROCESS[:len(b.mean)])
    value, vec = np.linalg.eigh(cov)
    cov = (vec * np.maximum(value, 2.e-5)) @ vec.T
    return mu, cov


def observe(body, view, b, packet, *, learn_rib=True):
    rib = rib_update(body, view, b.rib, packet.echo) if learn_rib else b.rib
    offset = rib @ RIB_MODE
    mean, cov = heart_update(body, body.with_ribs(view, offset), b, packet)
    return Belief(mean, cov, rib)


def predictive(body, chart, phase, b):
    view = body.view(chart)
    all_area, all_detect, all_weight = [], [], []
    points = b.points()
    for offset, probability in zip(RIB_MODE, b.rib):
        if probability < 1.e-3:
            continue
        m = body.moments(body.with_ribs(view, offset), points, phase)[0][:, 0, :, 0]
        all_area.append(m)
        all_detect.append(det_prob(m))
        all_weight.extend([probability / len(points)] * len(points))
    values = np.vstack(all_area)
    w = np.asarray(all_weight)
    w /= w.sum()
    means = w @ values
    band = np.empty((2, 4))
    for j in range(4):
        order = np.argsort(values[:, j])
        band[:, j] = np.interp([.05, .95], np.cumsum(w[order]), values[order, j])
    return means, w @ np.vstack(all_detect), band

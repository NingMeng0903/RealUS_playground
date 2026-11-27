"""Matched MPC and frame-reconditioned ProDMP on the registered leg simulator."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc


import scan_kernel as legacy_scan
import scan_kernel as legacy_geometry


LEGACY = Path(os.environ.get("LEGACY_LEG_STUDY", "/home/camp/Documents/Codex/2026-11-13/ieee-t-ro-b-mpc-b"))
LEGACY_HOLDOUT = LEGACY / "work/leg_reference_transfer/blender_holdout"
EXTRA_DEPS = Path(os.environ.get("LEG_LEGACY_DEPS", "/tmp/right-leg-render-deps"))


def _archived_prodmp():
    path = Path(__file__).parent / "prodmp/ab_leg_bifurcation_20261112/learned_prodmp.py"
    spec = spec_from_file_location("_leg_scan_archived_prodmp", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ARCHIVED_PRODMP = _archived_prodmp()
BASIS_SOURCE = ARCHIVED_PRODMP.BASIS_SOURCE
Trial = legacy_scan.Trial
CLOCK_S = .1
MAX_SPEED_MM_S = legacy_scan.BASE_SPEED
ANGLE_STEP = float(legacy_geometry.ANGLES[1])
SECTORS = len(legacy_geometry.ANGLES)
ANGLE_RANGE = 2.5
AXIAL_RANGE = 2.
EPS = 1.e-8
DEFAULT_TICKS = 24
DEFAULT_DISTANCE = 80.
FEATURE_NAMES = (
    "expected_detection", "separation", "detection_sd", "shadow", "in_window",
    "merge_risk", "belief_peak", "belief_entropy", "latest_quality",
    "latest_confidence", "latest_ambiguity", "latest_echo_count",
    "history_confidence", "history_support", "online_confidence", "disagreement",
    "route_fraction", "region_fraction", "sin_angle", "cos_angle", "prior_width",
)


def definition(target, begin, end, roles=(0, 1, 2, 3, 4)):
    if end <= begin:
        raise ValueError("A route needs at least two stations")
    size = end - begin + 1
    return target, begin, end, tuple(roles), np.minimum(np.arange(size) * 6 // size, 5)


def holdout_specs(scanner):
    result = {}
    for name, (high, low, roles) in legacy_geometry.TASKS.items():
        indices = np.flatnonzero((scanner.stations <= high) & (scanner.stations >= low))
        result[name] = definition(0, int(indices[0]), int(indices[-1]), roles)
    return result


def region_of(spec, station):
    nearest = int(np.floor(station + .5))
    return int(spec[4][np.clip(nearest, spec[1], spec[2]) - spec[1]])


def _between(array, s):
    lo = int(s)
    fraction = s - lo
    if fraction < EPS or lo == len(array) - 1:
        return array[lo]
    return (1. - fraction) * array[lo] + fraction * array[lo + 1]


def _sector_delta(a, other):
    return (other - a + SECTORS / 2) % SECTORS - SECTORS / 2


def wrap_sector(sector):
    wrapped = sector % SECTORS
    # Negative ulps near zero may produce an exact upper endpoint after modulo.
    return 0. if wrapped >= SECTORS - EPS else wrapped


def _angular_mix(array, sector):
    index = np.floor(sector).astype(int) % SECTORS
    weight = sector % 1.
    return (1. - weight) * array[index] + weight * array[(index + 1) % SECTORS]


def _chart_indices(station, sector, last_station):
    station = np.clip(station, 0, last_station)
    sector = np.mod(sector, SECTORS)
    sector = np.where(sector >= SECTORS - EPS, 0., sector)
    s0, a0 = np.floor(station).astype(int), np.floor(sector).astype(int)
    s1, a1 = np.minimum(s0 + 1, last_station), (a0 + 1) % SECTORS
    return s0, s1, a0, a1, station - s0, sector - a0


class Surface:
    """Public contact chart, prior hypothesis geometry and population bone field."""

    def __init__(self, y, contacts, priors, valid, shape=None):
        self.y, self.contact, self.priors, self.valid = y, contacts, priors, valid
        self.shape = shape
        self.cross_origin = contacts.mean(axis=1) if shape is None else shape[0]
        self.bone = legacy_geometry.bone_prior()
        self.slew = np.linalg.norm(contacts - np.roll(contacts, -1, axis=1), axis=-1)
        self.offsets = np.arange(-25., 25.1, 2.5)
        self.depths = np.arange(4., 80.1, 2.)
        self.ray = np.empty((len(y), SECTORS, len(self.offsets), len(self.depths)), dtype=np.float32)
        for s in range(len(y)):
            ray = self._rays(contacts[s], legacy_geometry.NORMALS, legacy_geometry.TANGENTS,
                             self.offsets[None, :, None], self.depths[None, None, :])
            self.ray[s] = np.maximum.accumulate(self.occupancy(s, ray), axis=-1)

    @staticmethod
    def _rays(contact, normal, tangent, x, depth):
        return (contact[..., None, None, :] + x[..., None] * tangent[..., None, None, :]
                - depth[..., None] * normal[..., None, None, :])

    def occupancy(self, s, points):
        y = _between(self.y, s)
        if self.shape is not None:
            origin, radii, base_origin, base_radii, base_y = self.shape
            local = points - _between(origin, s)
            sector = np.arctan2(local[..., 1], local[..., 0]) / ANGLE_STEP
            scale = _angular_mix(_between(base_radii, s), sector) / _angular_mix(_between(radii, s), sector)
            points = _between(base_origin, s) + local * scale[..., None]
            y = _between(base_y, s)
        xyz = np.stack((points[..., 0], np.full(points.shape[:-1], y), points[..., 1]), axis=-1)
        occ = self.bone(xyz)
        return np.where(np.isfinite(occ), occ, .15)

    @lru_cache(maxsize=8192)
    def pose(self, s, a):
        a = wrap_sector(a)
        if float(s).is_integer() and float(a).is_integer():
            return (self.contact[int(s), int(a)], legacy_geometry.NORMALS[int(a)],
                    legacy_geometry.TANGENTS[int(a)])
        theta = a * ANGLE_STEP
        normal = np.array([np.cos(theta), np.sin(theta)])
        tangent = np.array([-normal[1], normal[0]])
        if self.shape is None:
            contact = np.array([_angular_mix(row, np.asarray(a)) for row in _between(self.contact, s).T])
        else:
            origin, radii = self.shape[:2]
            contact = _between(origin, s) + _angular_mix(_between(radii, s), np.asarray(a)) * normal
        return contact, normal, tangent

    def legal(self, s, a):
        if not 0 <= s <= len(self.y) - 1:
            return False
        a = wrap_sector(a)
        lo, hi = int(s + EPS), int(np.ceil(s - EPS))
        left, right = int(a + EPS) % SECTORS, int(np.ceil(a - EPS)) % SECTORS
        return bool(self.valid[lo, left] and self.valid[lo, right] and self.valid[hi, left]
                    and self.valid[hi, right] and self.pose(s, a)[0][0] < -1.)

    def world(self, s, a):
        contact, _, _ = self.pose(s, a % SECTORS)
        return np.array([contact[0], _between(self.y, s), contact[1]])

    def path_coordinates(self, station, sector):
        """World coordinates and contact legality on a vectorized chart."""
        s0, s1, a0, a1, ds, da = _chart_indices(station, sector, len(self.y) - 1)
        sa = np.where(ds < EPS, s0, s1)
        aa = np.where(da < EPS, a0, a1)
        valid = self.valid[s0, a0] & self.valid[s0, aa] & self.valid[sa, a0] & self.valid[sa, aa]
        y = self.y[s0] * (1. - ds) + self.y[s1] * ds
        if self.shape is None:
            s0, s1, ds, da = s0, s1, ds[..., None], da[..., None]
            first = self.contact[s0, a0] * (1. - da) + self.contact[s0, a1] * da
            second = self.contact[s1, a0] * (1. - da) + self.contact[s1, a1] * da
            contact = first * (1. - ds) + second * ds
        else:
            origin, radii = self.shape[:2]
            radius0 = radii[s0, a0] * (1. - da) + radii[s0, a1] * da
            radius1 = radii[s1, a0] * (1. - da) + radii[s1, a1] * da
            radius = radius0 * (1. - ds) + radius1 * ds
            angle = sector * ANGLE_STEP
            contact = origin[s0] * (1. - ds[..., None]) + origin[s1] * ds[..., None]
            contact += radius[..., None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)
        valid &= contact[..., 0] < -1.
        return np.stack((contact[..., 0], y, contact[..., 1]), axis=-1), valid

    def path_world(self, station, sector):
        coordinates, valid = self.path_coordinates(station, sector)
        return coordinates if valid.all() else None

    def absolute(self, s, a, xy):
        contact, normal, tangent = self.pose(s, a % SECTORS)
        return contact + xy[0] * tangent - xy[1] * normal

    def project(self, s, a, centers):
        contact, normal, tangent = self.pose(s, a % SECTORS)
        return (centers - contact) @ np.stack((tangent, -normal)).T

    def shadow(self, s, a, coords):
        a = wrap_sector(a)
        u = np.clip(np.rint((coords[..., 0] + 25.) / 2.5), 0, len(self.offsets) - 1).astype(int)
        z = np.clip(np.rint((coords[..., 1] - 4.) / 2.), 0, len(self.depths) - 1).astype(int)
        lo, angle = int(s), int(a)
        f_s, f_a = s - lo, a - angle
        first = self.ray[lo, angle, u, z]
        if f_a >= EPS:
            first = (1. - f_a) * first + f_a * self.ray[lo, (angle + 1) % SECTORS, u, z]
        if f_s < EPS:
            return first
        second = self.ray[lo + 1, angle, u, z]
        if f_a >= EPS:
            second = (1. - f_a) * second + f_a * self.ray[lo + 1, (angle + 1) % SECTORS, u, z]
        return (1. - f_s) * first + f_s * second

    def exact_population_shadow(self, s, a, means):
        if float(s).is_integer() and float(a).is_integer():
            return self.shadow(s, a, np.nan_to_num(means, nan=150.))
        safe = np.nan_to_num(means, nan=150.)
        contact, normal, tangent = self.pose(s, a % SECTORS)
        count = len(self.depths)
        origins = contact + np.clip(safe[:, 0], -25., 25.)[:, None] * tangent
        end = np.clip(safe[:, 1], self.depths[0], self.depths[-1])
        ray = origins[:, None] - np.linspace(self.depths[0], end, count).T[..., None] * normal
        return self.occupancy(s, ray).max(axis=-1)

    def legacy_travel(self, s, a, s1, a1):
        delta = int(_sector_delta(a, a1))
        direction = int(np.sign(delta))
        sectors = (a + np.arange(abs(delta)) * direction + min(0, direction)) % SECTORS
        turn = .5 * (self.slew[s, sectors] + self.slew[s1, sectors]).sum()
        straight = np.linalg.norm(self.world(s1, a) - self.world(s, a))
        return float(np.hypot(turn, straight))


class Anatomy:
    """Simulator-only truth and ultrasound gate."""

    def __init__(self, centers, *, bone_triangles=None, exact=None, inside=None, links=None):
        self.centers, self.exact, self.inside, self.links = centers, exact, inside, links
        self.triangles = bone_triangles
        if self.triangles is not None:
            self.y_min = self.triangles[:, :, 1].min(axis=1)
            self.y_max = self.triangles[:, :, 1].max(axis=1)

    @classmethod
    def blender(cls, centers, exact, inside, links=None):
        # Archive retarget imports read the SMPL-X registration. Keep them in evaluation.
        if str(LEGACY_HOLDOUT) not in sys.path:
            sys.path.append(str(LEGACY_HOLDOUT))
        if EXTRA_DEPS.is_dir() and str(EXTRA_DEPS) not in sys.path:
            sys.path.append(str(EXTRA_DEPS))
        from geometry import BLENDER_BONES, MORPH, hits

        names = ("Femur_R", "Tibia_R", "Fibula_R", "Patella_R")
        with np.load(BLENDER_BONES, allow_pickle=False) as pack:
            triangles = np.concatenate([
                MORPH.transform(pack[f"{name}_retarget_mm"].copy())[pack[f"{name}_faces"]]
                for name in names])
        anatomy = cls(centers, bone_triangles=triangles, exact=exact, inside=inside, links=links)
        anatomy.hits = hits
        return anatomy

    @lru_cache(maxsize=2048)
    def _slice_bones(self, y):
        tris = self.triangles[(self.y_min <= y) & (self.y_max > y)]
        second = np.roll(tris, -1, axis=1)
        first_y, second_y = tris[..., 1], second[..., 1]
        intersects = (first_y <= y) != (second_y <= y)
        first, second = tris[intersects], second[intersects]
        weight = (y - first[:, 1]) / (second[:, 1] - first[:, 1])
        return (first + weight[:, None] * (second - first))[:, [0, 2]].reshape(-1, 2, 2)

    @lru_cache(maxsize=8192)
    def truth(self, surface, source, s, a):
        key = 0 if self.triangles is not None else source
        centers = _between(self.centers[key], s)
        if self.links is not None and not float(s).is_integer():
            centers = np.where(self.links[key, int(s), :, None], centers, np.nan)
        means = surface.project(s, a, centers)
        if self.triangles is None:
            return means, surface.exact_population_shadow(s, a, means)
        if float(s).is_integer() and float(a).is_integer():
            blocked = (self.exact[int(s), int(a)] < means[:, 1] - 1.) | self.inside[int(s)]
        else:
            edges = self._slice_bones(float(_between(surface.y, s)))
            if not len(edges):
                return means, np.zeros(len(means))
            contact, normal, tangent = surface.pose(s, a)
            present = np.isfinite(centers).all(axis=-1)
            blocked = np.zeros(len(centers), dtype=bool)
            origins = contact + means[present, 0, None] * tangent
            first = self.hits(edges, origins, np.broadcast_to(-normal, origins.shape)).min(axis=-1)
            points = centers[present]
            crosses = self.hits(edges, points, np.broadcast_to([.994, .109], points.shape))
            blocked[present] = (first < means[present, 1] - 1.) | (np.isfinite(crosses).sum(axis=-1) % 2 == 1)
        return means, blocked.astype(float)

    def training_truth(self, surface, trial, station, sector):
        layout = trial.layout
        if layout is not None and getattr(layout, "shared_vertices", ()):
            centers = layout.at(station)
        else:
            centers = _between(self.centers[trial.source] if layout is None else layout.centers, station)
            if self.links is not None and not float(station).is_integer():
                centers = np.where(self.links[trial.source, int(station), :, None], centers, np.nan)
        if layout is None:
            centers = trial.deform(surface, station, centers)
        points = surface.project(station, sector, centers)
        return points, surface.shadow(station, sector, np.nan_to_num(points, nan=150.))


class Model:
    """Original Bayesian reader with its view calculation evaluated at real probe poses."""

    def __init__(self, surface, width):
        self.surface, self.sigma, self.graph, self.bone = surface, width, True, True
        sobol = qmc.Sobol(4, scramble=True, seed=62191).random_base2(3)
        from scipy.special import ndtri
        self.samples = np.vstack(([0., 0., 0., 0.], ndtri(np.clip(sobol, .04, .96))))
        self.prior = np.full(3 * len(self.samples), 1. / (3 * len(self.samples)))
        self.h = np.empty((len(surface.y), len(self.prior), 5, 2))
        prior_links = getattr(surface, "reference_links", None)
        self.reference_support = None
        shift = np.rint(self.samples[:, 2] * 2.).astype(int)
        if prior_links is not None:
            indices = np.clip(np.arange(prior_links.shape[1])[:, None] + shift, 0,
                              prior_links.shape[1] - 1)
            self.reference_support = prior_links[:, indices, :5].transpose(1, 0, 2, 3)
            self.reference_support = self.reference_support.reshape(len(indices), len(self.prior), 5)
        for s in range(len(surface.y)):
            locations = np.clip(s + shift, 0, len(surface.y) - 1)
            points = surface.priors[:, locations, :5, :].copy()
            curve = .35 * width * self.samples[:, 3] * np.sin(s * .10)
            points += (width * self.samples[:, :2] + np.stack((curve, -.5 * curve), axis=-1))[None, :, None]
            self.h[s] = points.reshape(len(self.prior), 5, 2)

    @lru_cache(maxsize=6144)
    def view(self, s, a):
        means = _between(self.h, s)
        if self.reference_support is not None and not float(s).is_integer():
            means = np.where(self.reference_support[int(s), :, :, None], means, np.nan)
        projected = self.surface.project(s, a, means)
        present = np.isfinite(projected).all(axis=-1)
        p = np.nan_to_num(projected, nan=999.)
        occ = self.surface.shadow(s, a, p)
        detect = .74 * (1. - .92 * occ) * present
        detect *= (p[..., 1] < legacy_scan.DEPTH_MAX + 9.) & (np.abs(p[..., 0]) < legacy_scan.LATERAL + 9.)
        sigma = np.array([2.35, 1.85])
        low = [-legacy_scan.LATERAL, legacy_scan.DEPTH_MIN]
        high = [legacy_scan.LATERAL, legacy_scan.DEPTH_MAX]
        integration = (ndtr((high - p) / sigma) - ndtr((low - p) / sigma)).prod(axis=-1)
        separation = np.linalg.norm(p[:, 1:] - p[:, :1], axis=-1)
        separation = np.min(np.where(present[:, 1:], separation, 50.), axis=1)
        return p, detect, integration, present, occ, separation

    @lru_cache(maxsize=8192)
    def predictive(self, s, a):
        _, detect, integration, _, shadow, separation = self.view(s, a)
        visibility = detect[:, 0] * integration[:, 0]
        return np.vstack((visibility, np.minimum(25., separation) / 25., visibility ** 2,
                          shadow[:, 0], integration[:, 0], np.exp(-.5 * (separation / 5.) ** 2)))

    def feature(self, s, a, w):
        value = self.predictive(s, a) @ w
        value[2] = np.sqrt(max(0., value[2] - value[0] ** 2))
        return value

    def analyze_explanations(self, s, a, w, z, quality, roles):
        points, detect, integration, _, _, _ = self.view(s, a)
        return legacy_scan.association(points, detect, integration, z, quality,
                                       tuple(roles))

    def analyze(self, s, a, w, z, quality, roles):
        return self.analyze_explanations(s, a, w, z, quality, roles)[:3]


class Reader:
    """Frame association and unthresholded public Bayesian observation state."""

    def __init__(self, model, roles, history):
        self.model, self.roles, self.history = model, roles, history
        self.w = model.prior.copy()
        self.frames = []
        self.mixed_posterior = []
        self.state = None

    def absolute(self, s, a, xy):
        return self.model.surface.absolute(s, a, xy)

    def eval(self, s, a, z, quality, w, previous):
        same = previous is not None
        ds = abs(s - previous[0]) if same else 1.
        da = abs(_sector_delta(a, previous[1])) if same else 1.
        separation = min(1., .72 * ds + .60 * da)
        mix = .006 + .019 * separation + .011 * (1. - quality)
        offset = (1. - mix) * w + mix * self.model.prior
        ll, p, unknown, mixed = self.model.analyze_explanations(s, a, offset, z, quality, self.roles)
        strength = (.11 + .49 * separation) * (.4 + .6 * quality) if len(z) else .50 * quality
        posterior = offset * np.maximum(ll, 1.e-90) ** strength
        posterior /= posterior.sum()
        return posterior, posterior @ p, posterior @ unknown, posterior @ mixed

    def push(self, s, a, z, labels, q, region, tick=None):
        self.w, p, unknown, mixed = self.eval(s, a, z, q, self.w, self.state)
        self.state = (s, a)
        self.frames.append((s, a, z, labels, q, region, p, unknown, tick))
        self.mixed_posterior.append(mixed)

    def fork(self):
        reader = type(self)(self.model, self.roles, self.history)
        reader.frames = self.frames.copy()
        reader.mixed_posterior = self.mixed_posterior.copy()
        reader.w, reader.state = self.w.copy(), self.state
        return reader

    def select(self, region, threshold=.52):
        best = max(((i, j, float(p)) for i, row in enumerate(self.frames)
                    if row[5] == region for j, p in enumerate(row[6])),
                   key=lambda value: value[2], default=None)
        return best if best is None or best[2] >= threshold else None


@dataclass
class Node:
    reader: Reader
    station: int
    sector: int
    tick: int
    distance: float
    visits: int = 0
    slow: int = 0

    def fork(self):
        return Node(self.reader.fork(), self.station, self.sector, self.tick, self.distance, self.visits, self.slow)


@dataclass(frozen=True)
class ContinuousAction:
    angle: float
    axial: float
    shape: float

    def __post_init__(self):
        if not all(np.isfinite(v) and -1. <= v <= 1. for v in (self.angle, self.axial, self.shape)):
            raise ValueError("Motion components must lie in [-1, 1]")


@dataclass(frozen=True)
class MotionResult:
    node: Node
    ticks: int
    distance: float
    legal: bool
    reason: str
    trajectory: np.ndarray
    feedback_updates: int = 0
    applied_fraction: float = 0.


def public_features(node, spec):
    reader, s, a = node.reader, node.station, node.sector
    row = reader.frames[-1]
    region = region_of(spec, s)
    selected = reader.select(region)
    # The single-frame association anchor ignores history in its selection.
    online = max(((i, j, float(value)) for i, frame in enumerate(reader.frames) if frame[5] == region
                  for j, value in enumerate(frame[6])), key=lambda item: item[2], default=None)
    confidence, support = (0., 0.)
    if selected is not None:
        frame, slot, confidence = selected
        choice = reader.frames[frame]
        absolute = reader.absolute(choice[0], choice[1], choice[2][slot])
        for row2 in reader.frames:
            delta_s = abs(row2[0] - choice[0])
            different_view = delta_s < .25 and abs(_sector_delta(row2[1], choice[1])) >= .75
            if (.25 <= delta_s <= 3. or different_view) and len(row2[2]):
                points = np.array([reader.absolute(row2[0], row2[1], xy) for xy in row2[2]])
                affinity = np.exp(-.5 * (np.linalg.norm(points - absolute, axis=1) / 5.5) ** 2)
                support = max(support, float(row2[4] * np.max(affinity)))
    peak = np.max(row[6], initial=0.)
    entropy = -reader.w @ np.log(np.maximum(reader.w, 1.e-100)) / np.log(len(reader.w))
    return np.r_[reader.model.feature(s, a, reader.w), np.max(reader.w), entropy, row[4], peak,
                 np.max(row[7], initial=0.), len(row[2]) / 6., confidence, support,
                 0. if online is None else online[2],
                 float((selected is None) != (online is None) or
                       (selected is not None and online is not None and selected[:2] != online[:2])),
                 (s - spec[1]) / (spec[2] - spec[1]), region / 5.,
                 np.sin(a * ANGLE_STEP), np.cos(a * ANGLE_STEP), reader.model.sigma / 30.]


class ProDMP:
    """Endpoint-conditioned integrated forcing with a one-dimensional shape action."""

    def __init__(self):
        basis = ARCHIVED_PRODMP._IntegralBasis()
        self.basis = basis
        values, slopes = basis.evaluate(np.array([1.]))
        endpoint = np.vstack((values[0], slopes[0]))
        self.inverse = np.linalg.pinv(endpoint)
        null = np.linalg.svd(endpoint, full_matrices=True)[2][2:].T
        middle, _ = basis.evaluate(np.array([.5]))
        mode = null @ (middle[0] @ null)
        self.mode = mode / float(middle[0] @ mode)

    @lru_cache(maxsize=72)
    def grid(self, ticks):
        phase = np.linspace(0., 1., ticks + 1)
        values, slopes = self.basis.evaluate(phase)
        return phase, values, slopes

    def segment(self, start, velocity, goal, ticks, shape):
        phase, values, slopes = self.grid(ticks)
        pos, derivative = ARCHIVED_PRODMP._homogeneous(phase, start, goal, ticks * velocity, self.basis.lam)
        weights = self.inverse @ np.vstack((goal - pos[-1], -derivative[-1]))
        weights += self.mode[:, None] * shape[None, :]
        positions, velocities = pos + values @ weights, (derivative + slopes @ weights) / ticks
        positions[[0, -1]] = [start, goal]
        velocities[[0, -1]] = [velocity, np.zeros_like(velocity)]
        return positions, velocities


PRIMITIVE = ProDMP()


@lru_cache(maxsize=8)
def _load_pack(path):
    with np.load(path, allow_pickle=False) as pack:
        return {name: pack[name] for name in pack.files}


class Scanner:
    def __init__(self, stations, surface, anatomy, ct_bounds=None, edge_status=None):
        self.stations, self.surface, self._anatomy = stations, surface, anatomy
        self.ct_bounds = ct_bounds
        self.valid = surface.valid
        self.edge_status = edge_status

    @classmethod
    def training(cls, pack, shape=0, target=0):
        data = _load_pack(Path(pack))
        order = [target] + [i for i in range(6) if i != target]
        centers = data["centers"][shape][..., order, :]
        origins, radii, y = (data[key][shape] for key in ("origins", "radii", "ys"))
        contacts = origins[:, None, :] + radii[..., None] * legacy_geometry.NORMALS[None]
        chart = (origins, radii, data["origins"][0], data["radii"][0], data["ys"][0])
        surface = Surface(y, contacts, centers[:3].copy(), contacts[..., 0] < -1., shape=chart)
        topology = data.get("links")
        links = None if topology is None else topology[..., order]
        surface.reference_links = None if links is None else links[:3]
        bounds = tuple(map(int, data["ct_bounds"]))
        return cls(data["neutral_grid"], surface, Anatomy(centers, links=links), bounds,
                   data["edge_status"][..., order])

    @classmethod
    def holdout(cls, pack, legacy_geometry_pack, *, topology=None):
        path = Path(legacy_geometry_pack)
        data = _load_pack(path / "geometry.npz" if path.is_dir() else path)
        surface = Surface(data["ys"], data["contacts"], data["centers"][:3].copy(), data["valid_contact"])
        topology = Path(topology) if topology is not None else Path(pack).with_name("blender_holdout_geometry_links.npz")
        links = None
        edge_status = None
        if topology.is_file():
            topology_data = _load_pack(topology)
            if not np.array_equal(topology_data["neutral_grid"], data["ys"]):
                raise ValueError("Blender topology and registered acquisition stations differ")
            links = topology_data["links"]
            if links.shape != (len(data["ys"]) - 1, data["centers"].shape[2]):
                raise ValueError("Blender topology must identify each of the adjacent station edges")
            links = links[None]
            edge_status = topology_data["edge_status"][None]
        prior_pack = _load_pack(Path(pack))
        first = np.searchsorted(-prior_pack["neutral_grid"], -data["ys"][0])
        projection = prior_pack["links"][:3, first:first + len(data["ys"]) - 1]
        if projection.shape[1] == len(data["ys"]) - 1:
            surface.reference_links = projection
        anatomy = Anatomy.blender(data["centers"][3:].copy(), data["bone_distance"], data["inside_bone"], links)
        return cls(data["ys"], surface, anatomy, edge_status=edge_status)

    @lru_cache(maxsize=4)
    def model(self, width=8):
        return Model(self.surface, width)

    def _observe(self, node, spec, trial, station, sector, tick):
        sector = wrap_sector(sector)
        if self._anatomy.triangles is None:
            points, gates = self._anatomy.training_truth(self.surface, trial, station, sector)
        else:
            points, gates = self._anatomy.truth(self.surface, trial.source, station, sector)
        coupling = 1. - .12 * abs(np.sin(sector * ANGLE_STEP))
        z, labels, q = trial.observe(points, gates, tick, station=station, sector=sector,
                                     contact_quality=coupling)
        nearest = min(spec[2], max(spec[1], int(station + .5)))
        node.reader.push(station, sector, z, labels, q, int(spec[4][nearest - spec[1]]), tick)

    def start(self, spec, trial, model):
        reader = getattr(self, "reader_type", Reader)(model, spec[3], True)
        s = spec[1]
        a = int(max(np.flatnonzero(self.valid[s]), key=lambda b: model.feature(s, int(b), reader.w)[0]))
        node = Node(reader, s, a, 1, 0.)
        self._observe(node, spec, trial, s, a, 1)
        return node

    def _alternatives(self, s, a):
        return [(a + i) % SECTORS for i in (0, -1, 1, -2, 2) if self.valid[s, (a + i) % SECTORS]]

    def _pick_mpc(self, s, a, following, reader):
        possible = self._alternatives(following, a) or np.flatnonzero(self.valid[following]).tolist()

        def score(b):
            feature = reader.model.feature(following, b, reader.w)
            distance = self.surface.legacy_travel(s, a, following, b)
            value = feature[0] + .19 * feature[1] - .012 * distance
            if following + 1 < len(self.stations):
                after = self._alternatives(following + 1, b)
                value += .47 * max((reader.model.feature(following + 1, c, reader.w)[0]
                                    - .012 * self.surface.legacy_travel(following, b, following + 1, c)
                                    for c in after), default=0.)
            return value

        return max(possible, key=score)

    def advance(self, node, spec, trial):
        if node.station >= spec[2]:
            return node
        s, a = node.station, node.sector
        target = s + 1
        sector = self._pick_mpc(s, a, target, node.reader)
        dist = self.surface.legacy_travel(s, a, target, sector)
        node.tick += max(1, int(np.ceil(dist / (CLOCK_S * MAX_SPEED_MM_S))))
        node.distance += dist
        node.station, node.sector = target, sector
        self._observe(node, spec, trial, target, sector, node.tick)
        return node

    @lru_cache(maxsize=32)
    def _route_reserve(self, begin, end):
        ticks = np.zeros(end - begin + 1, dtype=int)
        mm = np.zeros(end - begin + 1)
        for s in range(end - 1, begin - 1, -1):
            dist = max(self.surface.legacy_travel(s, int(a), s + 1, b)
                       for a in np.flatnonzero(self.valid[s])
                       for b in (self._alternatives(s + 1, int(a)) or np.flatnonzero(self.valid[s + 1])))
            i = s - begin
            ticks[i] = ticks[i + 1] + max(1, int(np.ceil(dist / (CLOCK_S * MAX_SPEED_MM_S))))
            mm[i] = mm[i + 1] + dist
        return ticks, mm

    def route_reserve(self, spec):
        return self._route_reserve(spec[1], spec[2])

    def mpc_actions(self, node, spec):
        s, a = node.station, node.sector
        shift = [(s, (a + da) % SECTORS, False) for da in (-3, -2, -1, 1, 2, 3)]
        if s > spec[1] and self.valid[s - 1, a]:
            shift.extend((s - 1, b, False) for b in self._alternatives(s - 1, a) if b != a)
            shift.append((s - 1, a, True))
        return tuple(action for action in shift if self._mpc_goal(node, spec, action) is not None)

    def _mpc_goal(self, node, spec, action):
        if isinstance(action, ContinuousAction):
            sector = node.sector + int(np.rint(ANGLE_RANGE * action.angle))
            station = int(np.clip(node.station + np.rint(AXIAL_RANGE * action.axial), spec[1], spec[2]))
            slow = False
        else:
            station, sector, slow = action
            sector = node.sector + _sector_delta(node.sector, sector)
        if (not spec[1] <= station <= spec[2] or
                not self.surface.legal(station, sector % SECTORS)):
            return None
        return np.array([station, sector], dtype=float), slow

    def _path_metrics(self, positions, spec, speed=MAX_SPEED_MM_S):
        # A millimetre-scale subdivision bounds both contact legality and the
        # length of each curved pose-to-pose segment on the skin chart.
        starts, ends = positions[:-1], positions[1:]
        chart = starts[:, None, :] + (ends - starts)[:, None, :] * np.linspace(0., 1., 5)[None, :, None]
        flat = chart.reshape(-1, 2)
        if np.any((flat[:, 0] < spec[1] - EPS) | (flat[:, 0] > spec[2] + EPS)):
            return None
        chart[..., 0] = np.clip(chart[..., 0], spec[1], spec[2])
        world = self.surface.path_world(chart[..., 0], chart[..., 1])
        if world is None:
            return None
        lengths = np.linalg.norm(np.diff(world, axis=1), axis=-1).sum(axis=1)
        if lengths.max(initial=0.) > speed * CLOCK_S + EPS:
            return None
        return lengths

    @staticmethod
    def _rejected(node, reason):
        return MotionResult(node, 0, 0., False, reason, np.empty((0, 5)))

    def _execute(self, node, spec, trial, plan, max_mm, *, feedback=False, fraction=1.):
        output = node.fork()
        path = []
        total = 0.
        updates = 0
        speed = plan[0][4]
        base_cost = [self._path_metrics(item[0], spec, speed) for item in plan]
        for segment, (poses, velocities, goal, shape, _) in enumerate(plan):
            ticks = len(poses) - 1
            lengths = base_cost[segment].copy()
            tail = sum(cost.sum() for cost in base_cost[segment + 1:])
            for k in range(ticks):
                remain = ticks - k
                if feedback and remain >= 2:
                    correction = self._correction(output.reader.frames[-1])
                    if abs(correction) > 1.e-6:
                        # Preserve the current pose and derivative. Recondition
                        # the remaining basis weights using the latest echo only.
                        strength = min(1., remain / 4.) ** 2
                        updated_shape = shape * (remain / ticks) ** 2 + np.array([0., strength * correction])
                        suffix, velocity = PRIMITIVE.segment(poses[k], velocities[k], goal, remain, updated_shape)
                        cost = self._path_metrics(suffix, spec, speed)
                        if cost is not None:
                            changed = np.linalg.norm(suffix[1] - poses[k + 1]) > 1.e-5
                            if changed and total + cost.sum() + tail <= max_mm + EPS:
                                poses[k:], velocities[k:], lengths[k:] = suffix, velocity, cost
                                updates += 1
                s, a = poses[k + 1]
                ds = float(lengths[k])
                total += ds
                output.tick += 1
                self._observe(output, spec, trial, float(s), float(a), output.tick)
                path.append((output.tick, s, a % SECTORS, ds, total))
        output.distance += total
        output.visits += 1
        output.slow += int(speed < MAX_SPEED_MM_S)
        return MotionResult(output, output.tick - node.tick, total, True, "completed",
                            np.asarray(path, dtype=float), updates, fraction)

    @staticmethod
    def _correction(frame):
        z, quality, probabilities = frame[2], frame[4], frame[6]
        if not len(z):
            return 0.
        weight = probabilities ** 2
        lateral = weight @ z[:, 0] / (.10 + weight.sum())
        return float(np.clip(quality * lateral / 8., -.65, .65))

    def _fit_loop(self, node, spec, goal, shape, max_ticks, max_mm, *, prodmp, speed):
        start = np.array([node.station, node.sector], dtype=float)
        velocity = np.zeros(2)
        world_mm = np.linalg.norm(self.surface.world(*start) - self.surface.world(*goal))
        min_ticks = max(2 if prodmp else 1, int(np.ceil(world_mm / (speed * CLOCK_S))))
        for ticks in range(min_ticks, max_ticks // 2 + 1):
            if prodmp:
                outward = PRIMITIVE.segment(start, velocity, goal, ticks, shape)
                inward = PRIMITIVE.segment(goal, velocity, start, ticks, -.35 * shape)
            else:
                first, second = np.linspace(start, goal, ticks + 1), np.linspace(goal, start, ticks + 1)
                outward, inward = (first, np.zeros_like(first)), (second, np.zeros_like(second))
            lengths = [self._path_metrics(pos, spec, speed) for pos, _ in (outward, inward)]
            if all(value is not None for value in lengths) and sum(value.sum() for value in lengths) <= max_mm + EPS:
                return (outward + (goal, shape, speed), inward + (start, -.35 * shape, speed))
        return None

    def prodmp(self, node, spec, trial, action, *, feedback=True,
               max_ticks=DEFAULT_TICKS, max_mm=DEFAULT_DISTANCE):
        max_ticks = min(int(max_ticks), len(trial.tape.false) - 1 - node.tick)
        start = np.array([node.station, node.sector], dtype=float)
        goal = start + [AXIAL_RANGE * action.axial, ANGLE_RANGE * action.angle]
        goal[0] = np.clip(goal[0], spec[1], spec[2])
        # The forcing lies in the endpoint nullspace. Near a route end the
        # angular chart can still express shape without leaving the route.
        shape = action.shape * np.array([0., .60])
        if np.linalg.norm(goal - start) < EPS and np.linalg.norm(shape) < EPS:
            return self._rejected(node, "zero_displacement")
        for fraction in (1., .75, .5, .25):
            applied = start + fraction * (goal - start)
            if self.surface.legal(*applied):
                plan = self._fit_loop(node, spec, applied, fraction * shape, max_ticks, max_mm,
                                      prodmp=True, speed=MAX_SPEED_MM_S)
                if plan is not None:
                    return self._execute(node, spec, trial, plan, max_mm, feedback=feedback, fraction=fraction)
        return self._rejected(node, "contact_or_budget")

    def mpc(self, node, spec, trial, action, *, max_ticks=DEFAULT_TICKS, max_mm=DEFAULT_DISTANCE):
        max_ticks = min(int(max_ticks), len(trial.tape.false) - 1 - node.tick)
        target = self._mpc_goal(node, spec, action)
        if target is None:
            return self._rejected(node, "unsupported_goal")
        goal, slow = target
        if np.array_equal(goal, [node.station, node.sector]):
            return self._rejected(node, "zero_displacement")
        plan = self._fit_loop(node, spec, goal, np.zeros(2), max_ticks, max_mm,
                              prodmp=False, speed=MAX_SPEED_MM_S * (.5 if slow else 1.))
        if plan is None:
            return self._rejected(node, "contact_or_budget")
        return self._execute(node, spec, trial, plan, max_mm)

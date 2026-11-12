"""Online identity smoothing and lag-state filtering at calibrated knee poses."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.special import ndtr

# Physical units throughout: s, degrees, mm. z's scale is fixed by the atlas mode.
Z_AXIS = np.linspace(-7., 7., 17)
W_AXIS = np.linspace(-6., 6., 7)
B_AXIS = np.linspace(-6., 6., 9)
LAG_TAU = .72
VELOCITY_COUPLING = .83
PROCESS_SD = .8
IMAGE_SD = np.array([1.1, .85])
OUTLIER_SD = 2.8 * IMAGE_SD
OUTLIER_RATE = .06
NEAR_CLUTTER_SD = np.array([7., 3.5])
NEAR_CLUTTER_RATE = .38
HALF_WIDTH = 25.
DEPTH = (3., 52.)
CONFIDENCE = .87


@lru_cache(maxsize=None)
def assignments(n_lumens, n_obs):
    """Maps each true lumen to a unique detection or a miss."""
    result = []
    for entry in product(range(-1, n_obs), repeat=n_lumens):
        matched = [i for i in entry if i >= 0]
        if len(set(matched)) != len(matched):
            continue
        unused = tuple(i for i in range(n_obs) if i not in matched)
        result.append((entry, unused))
    return result


def norm_pdf(x, sigma):
    sigma = np.asarray(sigma)
    return np.exp(-.5 * (x / sigma) ** 2) / (np.sqrt(2. * np.pi) * sigma)


def grid_prior(axis, sd):
    values = norm_pdf(axis, sd)
    return values / values.sum()


def in_view_probability(center, sd):
    lower = np.array([-HALF_WIDTH, DEPTH[0]])
    upper = np.array([HALF_WIDTH, DEPTH[1]])
    return np.prod(ndtr((upper - center) / sd) - ndtr((lower - center) / sd), axis=-1)


@dataclass(frozen=True)
class Reading:
    t: float
    q: float
    velocity: float


@dataclass(frozen=True)
class Witness:
    """Joint posterior for the identity of one earlier unlabeled centroid."""
    event: int
    site: int
    ordinal: int
    coordinate: tuple[float, float]
    key: tuple[int, int]
    joint: np.ndarray


class Filter:
    def __init__(self, *, latent=True):
        self.z = Z_AXIS if latent else np.zeros(1)
        self.w = W_AXIS
        self.b = B_AXIS
        self.latent = latent
        self.parameters = np.stack(np.meshgrid(self.w, self.b, self.z, indexing="ij"), axis=-1)
        self.parameters = self.parameters.reshape(-1, len(self.z), 3)
        self.weights = (
            np.outer(grid_prior(self.w, 2.8), grid_prior(self.b, 2.7)).reshape(-1, 1)
            * grid_prior(self.z, 1.5)[None, :]
        )
        self.reading = None
        self.witnesses: list[Witness] = []

    def copy(self):
        """Weights and witness tensors are replaced, never edited in place."""
        other = self.__class__.__new__(self.__class__)
        other.__dict__ = self.__dict__.copy()
        return other

    def transition(self, dt, velocity):
        if dt < -1.e-10:
            raise ValueError("filter time must increase")
        if not self.latent or dt <= 1.e-10:
            return np.eye(len(self.z))
        decay = np.exp(-dt / LAG_TAU)
        mean = decay * self.z + VELOCITY_COUPLING * LAG_TAU * (1. - decay) * velocity
        sd = max(.23, PROCESS_SD * np.sqrt(1. - decay**2))
        midpoints = (self.z[1:] + self.z[:-1]) * .5
        cdf = ndtr((midpoints[None, :] - mean[:, None]) / sd)
        edges = np.pad(cdf, ((0, 0), (1, 1)), constant_values=(0., 1.))
        return np.diff(edges, axis=1)

    def predicted_reading(self, t):
        if self.reading is None:
            raise ValueError("filter has no current reading")
        dt = t - self.reading.t
        # Constant velocity over short horizons, saturated at validated angle knots.
        q = float(np.clip(self.reading.q + self.reading.velocity * dt, 11., 19.))
        velocity = self.reading.velocity if 11. < q < 19. else 0.
        return Reading(float(t), q, velocity)

    def prior_at(self, reading):
        if self.reading is None:
            return self.weights, np.eye(len(self.z))
        dt = reading.t - self.reading.t
        velocity = .5 * (reading.velocity + self.reading.velocity)
        trans = self.transition(dt, velocity)
        return self.weights @ trans, trans

    def means(self, base, rows, reading):
        pars = self.parameters
        centers = base + np.einsum("wzk,ldk->wzld", pars, rows)
        if not self.latent:
            centers = centers + (.25 * reading.velocity) * rows[:, :, 2]
        return centers

    def imaging(self, world, reading, site, view):
        base, rows = world.projection(reading.q, site, view)
        centers = self.means(base, rows, reading)
        visible = ((1. - OUTLIER_RATE) * in_view_probability(centers, IMAGE_SD)
                   + OUTLIER_RATE * in_view_probability(centers, OUTLIER_SD))
        detection = world.detection_probability(reading.q, site, view) * visible
        return centers, detection

    def emission(self, world, reading, site, view, obs):
        centers, hit = self.imaging(world, reading, site, view)
        count = len(obs)
        n_lumens = centers.shape[-2]
        errors = obs[None, None, None, :, :] - centers[..., None, :]
        gaussian = np.prod(norm_pdf(errors, IMAGE_SD), axis=-1)
        broad = np.prod(norm_pdf(errors, OUTLIER_SD), axis=-1)
        pd = world.detection_probability(reading.q, site, view)
        observed = pd * ((1. - OUTLIER_RATE) * gaussian + OUTLIER_RATE * broad)
        # A broad component admits image artifacts near the nominal lumens.
        nominal, _ = world.projection(reading.q, site, view)
        clutter_center = np.mean(nominal, axis=0)
        clutter = (1. - NEAR_CLUTTER_RATE) / (2. * HALF_WIDTH * (DEPTH[1] - DEPTH[0]))
        clutter = clutter + (
            NEAR_CLUTTER_RATE * np.prod(norm_pdf(obs - clutter_center, NEAR_CLUTTER_SD), axis=1)
            / max(1.e-9, in_view_probability(clutter_center, NEAR_CLUTTER_SD))
        )
        chance = world.clutter_probability(site, view)
        likelihood = np.zeros_like(self.weights)
        responsibilities = np.zeros((count, 3, *self.weights.shape))
        for assigned, unused in assignments(n_lumens, count):
            if len(unused) > 1:
                continue
            term = np.ones_like(self.weights)
            for label, j in enumerate(assigned):
                term *= observed[..., label, j] if j >= 0 else 1. - hit[..., label]
            if unused:
                term *= chance * clutter[unused[0]]
            else:
                term *= 1. - chance
            likelihood += term
            for label, j in enumerate(assigned):
                if j >= 0:
                    responsibilities[j, label] += term
            for j in unused:
                responsibilities[j, 2] += term
        return likelihood, responsibilities

    def observe(self, world, reading, site, view, obs, *, event, retain, track=True):
        obs = np.asarray(obs, dtype=float).reshape(-1, 2)
        prior, transition = self.prior_at(reading)
        likelihood, responsibilities = self.emission(world, reading, site, view, obs)
        normalizer = float(np.sum(prior * likelihood))
        if not np.isfinite(normalizer) or normalizer <= 0.:
            raise FloatingPointError("observation outside identity model")
        self.weights = prior * likelihood / normalizer
        self.reading = reading
        if track:
            multiplier = likelihood / normalizer
            self.witnesses = [
                Witness(w.event, w.site, w.ordinal, w.coordinate, w.key, (w.joint @ transition) * multiplier)
                for w in self.witnesses
            ]
        if retain:
            self.witnesses = self.witnesses + [
                Witness(event, site, j, tuple(value), (event, j), prior[None] * responsibilities[j] / normalizer)
                for j, value in enumerate(obs)
            ]
        return normalizer

    def advance(self, reading, *, track=True):
        self.weights, transition = self.prior_at(reading)
        self.reading = reading
        if track:
            self.witnesses = [
                Witness(w.event, w.site, w.ordinal, w.coordinate, w.key, w.joint @ transition)
                for w in self.witnesses
            ]

    def stats(self):
        weights = self.weights[..., None]
        mean = np.sum(weights * self.parameters, axis=(0, 1))
        var = np.sum(weights * (self.parameters - mean) ** 2, axis=(0, 1))
        return mean, np.sqrt(var)

    def probability(self, witness):
        return witness.joint.sum(axis=(1, 2))

    def decisions(self):
        """Highest confidence witnesses; each centroid can certify only one label."""
        if not self.witnesses:
            return (None, None), np.zeros(2)
        probs = np.stack([self.probability(w) for w in self.witnesses])
        # At most one centroid can be above CONFIDENCE for both labels. Resolve
        # low-confidence conflicts with a maximum product one-to-one assignment.
        products = probs[:, 0, None] * probs[None, :, 1]
        np.fill_diagonal(products, -1.)
        if len(probs) == 1 or products.max() <= 0.:
            k, label = np.unravel_index(np.argmax(probs[:, :2]), (len(probs), 2))
            indices = [None, None]
            indices[label] = int(k)
        else:
            indices = list(np.unravel_index(np.argmax(products), products.shape))
        keys = tuple(self.witnesses[i].key if i is not None else None for i in indices)
        confidence = np.array([probs[i, label] if i is not None else 0.
                               for label, i in enumerate(indices)])
        return keys, confidence

    def forecast(self, world, time, sites):
        reading = self.predicted_reading(time)
        prior, _ = self.prior_at(reading)
        xyz, design = world.world_design(reading.q, sites)
        samples = xyz + np.einsum("wzk,nck->wznc", self.parameters, design)
        if not self.latent:
            samples += .25 * reading.velocity * design[:, :, 2]
        mean = np.einsum("wz,wznc->nc", prior, samples)
        variance = np.einsum("wz,wznc->nc", prior, (samples - mean) ** 2)
        # Coordinate-wise model uncertainty plus first-order pose uncertainty.
        q_hi = min(19., reading.q + .15)
        q_lo = max(11., reading.q - .15)
        if q_hi > q_lo:
            x_hi, _ = world.world_design(q_hi, sites)
            x_lo, _ = world.world_design(q_lo, sites)
            pose_sd = .26 + .5 * max(0., time - self.reading.t)
            variance += (pose_sd * (x_hi - x_lo) / (q_hi - q_lo)) ** 2
        return mean, np.sqrt(variance)

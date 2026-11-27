"""Public gallbladder belief and a noisy synthetic segmentation observation model."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import qmc

from gallbladder_geometry import BEAM_ANGLE, DEPTH_MM


ORGAN_LIMIT = np.array([12., 14., 9., 19., 14., .21, .25])
ANGLE = np.arange(24) * (np.pi * (3. - np.sqrt(5.)))
RADIUS = np.sqrt((np.arange(len(ANGLE)) + .5) / len(ANGLE))
DISK = np.column_stack((RADIUS * np.cos(ANGLE), RADIUS * np.sin(ANGLE)))
SENSOR = {
    "wide": (.04, .009, .025, np.array([.24, 3.0, 3.8, 4.5, .12])),
    "stress": (.10, .025, .063, np.array([.41, 5.3, 6.4, 7.5, .20])),
}


def _ellipse(verts):
    """Fit a principal-axis ellipsoid enclosing the central mesh extent."""
    center = np.mean(verts, axis=0)
    _, vectors = np.linalg.eigh(np.cov((verts - center).T))
    frame = vectors[:, ::-1]
    projected = (verts - center) @ frame
    low, high = np.quantile(projected, [.015, .985], axis=0)
    return center + frame @ (.5 * (low + high)), frame, .5 * (high - low)


def _angle_difference(left, right):
    return (left - right + np.pi / 2.) % np.pi - np.pi / 2.


def _project(view, center, precision, axis, length):
    """Get a sliced ellipse and quadrature moments when a ray remains visible."""
    basis = np.column_stack((view.right, view.forward))
    delta = view.origin - center
    projected = precision @ basis
    matrix = basis.T @ projected
    linear = np.einsum("...i,...ij->...j", delta, projected)
    inverse = np.linalg.inv(matrix)
    midpoint = -np.einsum("...ij,...j->...i", inverse, linear)
    factor = 1. - np.einsum("...i,...ij,...j->...", delta, precision, delta) - np.sum(midpoint * linear, axis=-1)
    area = np.maximum(factor, 0.) * np.pi * np.sqrt(np.linalg.det(inverse))
    radius = np.sqrt(np.maximum(factor, 0.))
    circle = DISK @ np.swapaxes(np.linalg.cholesky(inverse), -1, -2)
    points = midpoint[..., None, :] + radius[..., None, None] * circle
    x, depth = np.moveaxis(points, -1, 0)
    available = ((depth > 2.) & (np.hypot(x, depth) < DEPTH_MM)
                 & (np.abs(x) < depth * np.tan(BEAM_ANGLE)))
    visible = available & ~view.beam_blocked(x, depth) & (factor[..., None] > 0.)
    count = visible.sum(axis=-1)
    weight = visible / np.maximum(count, 1)[..., None]
    centroid = (points * weight[..., None]).sum(axis=-2)
    aligned = axis @ basis
    aligned_norm = np.linalg.norm(aligned, axis=-1)
    direction = aligned / np.maximum(aligned_norm, 1.e-8)[..., None]
    coordinate = (points * direction[..., None, :]).sum(axis=-1)
    lo = np.min(np.where(visible, coordinate, np.inf), axis=-1)
    hi = np.max(np.where(visible, coordinate, -np.inf), axis=-1)
    span = np.maximum(hi - lo, 0.)
    heading = np.arctan2(direction[..., 0], direction[..., 1])
    features = np.stack((area * visible.mean(axis=-1), centroid[..., 0], centroid[..., 1], span, heading), axis=-1)
    coverage = np.clip(span / length, 0., 1.25)
    intact = visible.mean(axis=-1)
    return features, coverage, intact, aligned_norm


@dataclass(frozen=True)
class Forecast:
    features: np.ndarray
    detect: np.ndarray
    quality: np.ndarray


class Prior:
    """Fixed, public hypothesis bank from the transported anatomy template."""

    def __init__(self, body, *, size=64):
        sampler = qmc.Sobol(len(ORGAN_LIMIT), scramble=True, rng=np.random.default_rng(2107))
        n = int(np.ceil(np.log2(size)))
        offsets = np.vstack((np.zeros(len(ORGAN_LIMIT)), 2. * sampler.random_base2(n)[:size - 1] - 1.))
        self.parameters = offsets * ORGAN_LIMIT
        ellipses = []
        for p in self.parameters:
            anatomy = body.anatomy(p)
            ellipses.append([_ellipse(mesh) for mesh in (anatomy.gallbladder, anatomy.liver)])
        self.center = np.asarray([[entry[0] for entry in sample] for sample in ellipses])
        vectors = np.asarray([[entry[1] for entry in sample] for sample in ellipses])
        axes = np.asarray([[entry[2] for entry in sample] for sample in ellipses])
        self.axis = vectors[..., 0]
        self.axis *= np.where((self.axis * self.axis[0]).sum(axis=-1) < 0., -1., 1.)[..., None]
        self.length = 2. * axes[..., 0]
        self.precision = (vectors / axes[..., None, :] ** 2) @ np.swapaxes(vectors, -1, -2)
        self.initial = np.full(len(self.center), 1. / len(self.center))

    def information(self, weights):
        center = weights @ self.center[:, 0]
        axis = weights @ self.axis[:, 0]
        axis /= np.linalg.norm(axis)
        deviation = self.center[:, 0] - center
        radius = np.sqrt(np.sum(weights * (deviation ** 2).sum(axis=-1)))
        return center, axis, float(radius), float(weights @ self.length[:, 0])

    def predict(self, view, condition):
        features, coverage, intact, aligned = _project(view, self.center, self.precision, self.axis, self.length)
        detect = detection(features[..., 0], condition)
        # Smooth margins make small pose changes useful to the planner.
        shape = np.clip((coverage[:, 0] - .49) / .39, 0., 1.)
        clear = np.clip((intact[:, 0] - .42) / .40, 0., 1.)
        central = np.clip((53. - np.abs(features[:, 0, 1])) / 22., 0., 1.)
        quality = shape * clear * central * np.clip((aligned[:, 0] - .85) / .11, 0., 1.)
        return Forecast(features, detect, quality)

    def update(self, weight, forecast, value, condition, *, power=.42):
        _, fp, onset, _ = SENSOR[condition]
        outage = 1. - (1. - onset) ** 3
        p = forecast.detect
        observed = np.isfinite(value[:, 0])
        log_p = np.zeros(len(weight))
        if not np.any(observed):
            likelihood = outage + (1. - outage) * (1. - p).prod(axis=1)
            log_p = np.log(np.maximum(likelihood, 1.e-40))
        else:
            model_sd = np.array([[.54, 6.0, 6.5, 9.5, .29], [.78, 11., 13., 22., .49]])
            sd = np.hypot(model_sd, SENSOR[condition][-1])
            for index in range(2):
                if not observed[index]:
                    log_p += np.log1p(-p[:, index])
                    continue
                predicted = forecast.features[:, index]
                delta = predicted - value[index]
                scale = 100. if index == 0 else 500.
                delta[:, 0] = np.log1p(predicted[:, 0] / scale) - np.log1p(value[index, 0] / scale)
                delta[:, 4] = _angle_difference(predicted[:, 4], value[index, 4])
                normal = -.5 * np.sum((delta / sd[index]) ** 2, axis=1)
                normal -= np.log(sd[index]).sum() + 2.5 * np.log(2. * np.pi)
                clutter = .075 if index == 0 else .12
                diffuse = -np.log(5. * 145. * 180. * 105. * np.pi)
                true = np.maximum(p[:, index] - fp, 1.e-15)
                log_p += np.logaddexp(np.log((1. - clutter) * true) + normal,
                                      np.log(fp + clutter * true) + diffuse)
        score = np.log(np.maximum(weight, 1.e-60)) + power * log_p
        posterior = np.exp(score - logsumexp(score))
        return .998 * posterior + .002 * self.initial


def detection(area, condition):
    miss, fp, _, _ = SENSOR[condition]
    return fp + (1. - miss - fp) * (-np.expm1(-area / np.array([72., 140.])))


class Noise:
    """A time-indexed tape shared by all policies reaching different poses."""

    def __init__(self, seed, copy, condition, frames):
        self.condition = condition
        rng = np.random.default_rng([917_312, seed, copy])
        _, _, rate, sd = SENSOR[condition]
        self.detect = rng.random((frames, 2))
        independent = rng.normal(size=(frames, 2, 5))
        self.errors = independent.copy()
        for i in range(1, frames):
            self.errors[i] = .64 * self.errors[i - 1] + np.sqrt(1. - .64 ** 2) * independent[i]
        self.errors *= sd
        self.errors[:, 1] *= [1., 1.45, 1.45, 1.45, 1.]
        onset = rng.random(frames) < rate
        self.outage = onset.copy()
        self.outage[1:] |= onset[:-1]
        self.outage[2:] |= onset[:-2]
        self.fake = rng.uniform([25., -50., 40., 7., -np.pi / 2.],
                                [330., 50., 157., 44., np.pi / 2.], (frames, 2, 5))

    def sample(self, section, tick):
        _, fp, _, _ = SENSOR[self.condition]
        values = np.stack((section.gall, section.liver))
        detected = detection(values[:, 0], self.condition)
        genuine = self.detect[tick] < detected - fp
        false = (~genuine) & (self.detect[tick] < detected)
        if self.outage[tick]:
            genuine[:] = False
            false[:] = False
        error = self.errors[tick]
        result = values + error
        result[:, 0] = values[:, 0] * np.exp(error[:, 0])
        result[:, 3] = np.maximum(result[:, 3], 1.)
        result[:, 4] = _angle_difference(result[:, 4], 0.)
        result[false] = self.fake[tick, false]
        result[~(genuine | false)] = np.nan
        return result, genuine

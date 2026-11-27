"""Synthetic Y lumen observations in the repaired leg chart.

The renderer owns the true coefficients and detection labels. The estimator
receives only unlabelled geometric observations and the acquired view.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp, ndtr


DEPTH_MM = 18.0
HALF_FOV_MM = 8.0
DEPTH_BOUNDS_MM = (4.0, 36.0)
SIGMA_MM = np.array([0.85, 0.75])
CLUTTER_PROBABILITY = 0.075
MAX_DETECTION = 0.92
CELL_HALF_WIDTHS = np.array([1.7, 1.25, 1.25, 1.25, 1.3, 1.15])
TRUNK, LOWER, UPPER, FORK, JUNCTION = 0, 1, 2, 3, 4
LABELS = ("trunk", "lower", "upper", "fork", "junction")


def prior(shared: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Columns are shared lateral shift, three residuals, fork and depth.

    Without the shared shift, the three residuals have the same marginal
    endpoint distribution as the correlated model. Both retain fork and
    depth uncertainty and the same interpolated junction geometry.
    """
    beta = np.arange(-8.0, 8.1, 4.0)
    p_beta = np.array([0.11, 0.23, 0.32, 0.23, 0.11])
    residual = np.array([-4.0, 0.0, 4.0])
    p_residual = np.array([0.24, 0.52, 0.24])
    fork = np.array([-4.0, 0.0, 4.0])
    depth = np.array([-3.0, 0.0, 3.0])
    p_fork_depth = np.array([0.25, 0.5, 0.25])
    if not shared:
        sums = (beta[:, None] + residual).ravel()
        pair_weights = (p_beta[:, None] * p_residual).ravel()
        residual, inv = np.unique(sums, return_inverse=True)
        p_residual = np.bincount(inv, weights=pair_weights)
        beta, p_beta = np.array([0.0]), np.array([1.0])
    coeff = np.array(list(product(beta, residual, residual, residual, fork, depth)))
    weight = np.array(list(product(p_beta, p_residual, p_residual, p_residual,
                                   p_fork_depth, p_fork_depth))).prod(axis=1)
    return coeff, weight / weight.sum()


def draw_truth(seed: int, continuous: bool) -> np.ndarray:
    """Independent episode draws, kept fixed for every view of that episode."""
    rng = np.random.default_rng(seed)
    theta, weight = prior()
    out = theta[rng.choice(len(theta), p=weight)].copy()
    if continuous:
        # Between-grid anatomy, including fixed per-episode perturbations.
        # Training demonstrations and inference do not see these offsets.
        out += rng.uniform(-1.0, 1.0, 6) * CELL_HALF_WIDTHS
    return out


def graph_point(theta: np.ndarray, x: np.ndarray | float, label: int) -> np.ndarray:
    """Known smooth graph family in chart units; daughters join exactly."""
    theta = np.asarray(theta)
    x = np.broadcast_to(x, theta.shape[:-1])
    beta, wt, wl, wu, fork, dz = np.moveaxis(theta, -1, 0)
    s = x - fork
    if label == TRUNK:
        y = beta + wt + 1.1 * np.sin(np.minimum(s, 0.0) / 16.0)
        depth = DEPTH_MM + dz + 0.026 * np.minimum(s, 0.0)
    else:
        sign = -1.0 if label == LOWER else 1.0
        t = np.maximum(s, 0.0)
        share = np.minimum(t / 36.0, 1.0)
        w = wl if label == LOWER else wu
        y = beta + (1.0 - share) * wt + share * w + sign * (
            0.42 * t + 0.65 * np.sin(t / 9.0)
        )
        depth = DEPTH_MM + dz + sign * 0.029 * t
    return np.stack((x, y, depth), axis=-1)


def lumen_section(theta: np.ndarray, x: float) -> tuple[np.ndarray, np.ndarray]:
    """At most two resolvable centres in one material short-axis section."""
    trunk = graph_point(theta, x, TRUNK)
    lower = graph_point(theta, x, LOWER)
    upper = graph_point(theta, x, UPPER)
    s = x - theta[:, 4]
    split = (s > 0.0) & ((upper[:, 1] - lower[:, 1]) >= 3.2)
    common = np.where((s > 0.0)[:, None], (lower + upper) / 2.0, trunk)
    points = np.stack((np.where(split[:, None], lower, common), upper), axis=1)
    labels = np.column_stack((np.where(split, LOWER, TRUNK), np.where(split, UPPER, -1)))
    return points, labels


def target_points(theta: np.ndarray, x: float, label: int) -> np.ndarray:
    """Samples for a GPS projection. Fork views should contain both daughters."""
    if label in (FORK, JUNCTION):
        midpoint = (graph_point(theta, x, LOWER) + graph_point(theta, x, UPPER)) / 2.0
        return np.where((x < theta[:, 4])[:, None], graph_point(theta, x, TRUNK), midpoint)
    return graph_point(theta, x, label)


class SectionGeometry:
    """Project real atlas-decoded tissue into a synthetic section feature view.

    The calibrated longitudinal material section is given. Its transverse
    projection uses the current posed atlas frame. This is a geometric
    feature renderer, with ideal contact and known external registration.
    """

    def __init__(self, chart):
        self.chart = chart
        x = np.arange(-42.0, 42.1, 2.0)
        y = np.arange(-36.0, 36.1, 2.0)
        z = np.array([12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0])
        xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)
        xyz = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        xyz_skin = np.concatenate((xy, np.zeros(xy.shape[:-1] + (1,))), axis=-1)
        origin_xy = np.column_stack((x, np.zeros(len(x))))
        origin = chart.world(np.column_stack((origin_xy, np.zeros(len(x)))))
        frame = chart.frame(origin_xy)
        self.skin = chart.world(xyz_skin)
        self.body = chart.world(xyz)
        transverse = frame[:, :, 1:]
        skin_uv = np.einsum("xyd,xdk->xyk", self.skin - origin[:, None, :], transverse)
        body_uv = np.einsum("xyzd,xdk->xyzk", self.body - origin[:, None, None, :], transverse)
        self.skin_to_uv = RegularGridInterpolator((x, y), skin_uv, bounds_error=True)
        self.body_to_uv = RegularGridInterpolator((x, y, z), body_uv, bounds_error=True)
        self.skin_world = RegularGridInterpolator((x, y), self.skin, bounds_error=True)
        self.body_world = RegularGridInterpolator((x, y, z), self.body, bounds_error=True)

    def predict(self, theta: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points, labels = lumen_section(theta, float(xy[0]))
        mu = self.body_to_uv(points) - self.skin_to_uv(xy)[0]
        p = detection_probability(mu) * (labels >= 0)
        return mu, p, labels


def detection_probability(uv: np.ndarray) -> np.ndarray:
    """Soft calibrated aperture with known falloff, independent frame dropout."""
    y, z = np.moveaxis(uv, -1, 0)
    near, far = DEPTH_BOUNDS_MM
    return (MAX_DETECTION * ndtr((HALF_FOV_MM - np.abs(y)) / 1.0)
            * ndtr((z - near) / 1.25) * ndtr((far - z) / 1.25))


def render(geometry, truth, xy, rng, *, suppressed_label: int | None = None, dropout=False):
    """Return the public centres and a separate evaluation-only identity list."""
    mu, probability, labels = geometry.predict(truth[None], xy)
    mu, probability, labels = mu[0], probability[0].copy(), labels[0]
    if dropout:
        probability[:] = 0.0
    elif suppressed_label is not None:
        probability[labels == suppressed_label] = 0.0
    detected = rng.random(2) < probability
    measured = mu[detected] + rng.normal(size=(detected.sum(), 2)) * SIGMA_MM
    identity = labels[detected]
    if rng.random() < CLUTTER_PROBABILITY:
        clutter = [rng.uniform(-HALF_FOV_MM, HALF_FOV_MM), rng.uniform(*DEPTH_BOUNDS_MM)]
        measured = np.vstack((measured, clutter))
        identity = np.r_[identity, -1]
    order = np.argsort(measured[:, 0], kind="stable")
    return measured[order], identity[order]


def assignment_terms(mu, probability, observations):
    """Log densities of all valid two-slot signal/clutter assignments."""
    count = len(observations)
    slots = range(-1, count)
    near, far = DEPTH_BOUNDS_MM
    if count:
        err = (observations[None, None, :, :] - mu[:, :, None, :]) / SIGMA_MM
        log_noise = -0.5 * (err**2).sum(axis=-1) - np.log(2.0 * np.pi * SIGMA_MM.prod())
    with np.errstate(divide="ignore"):
        log_hit = np.log(probability)
        log_miss = np.log1p(-probability)
    for a, b in product(slots, repeat=2):
        if a == b and a >= 0:
            continue
        free = [i for i in range(count) if i != a and i != b]
        if len(free) > 1:
            continue
        if free:
            u, v = observations[free[0]]
            if abs(u) > HALF_FOV_MM or not near <= v <= far:
                continue
            clutter = np.log(CLUTTER_PROBABILITY / (2.0 * HALF_FOV_MM * (far - near)))
        else:
            clutter = np.log1p(-CLUTTER_PROBABILITY)
        first = log_miss[:, 0] if a < 0 else log_hit[:, 0] + log_noise[:, 0, a]
        second = log_miss[:, 1] if b < 0 else log_hit[:, 1] + log_noise[:, 1, b]
        yield (a, b), first + second + clutter


def observe(log_weights, mu, probability, labels, observations):
    """Exact finite Bayes step and marginal observed-centre identities."""
    terms = list(assignment_terms(mu, probability, observations))
    if not terms:
        raise ValueError("frame is outside the observation model support")
    scores = np.array([terms_i + log_weights for _, terms_i in terms])
    normalizer = logsumexp(scores)
    if not np.isfinite(normalizer):
        raise ValueError("all posterior hypotheses rejected this frame")
    updated = logsumexp(scores, axis=0) - normalizer
    identity = np.zeros((3, len(observations)))
    for (a, b), joint in zip((assignment for assignment, _ in terms), np.exp(scores - normalizer)):
        for slot, obs_idx in enumerate((a, b)):
            if obs_idx >= 0:
                for label in (TRUNK, LOWER, UPPER):
                    identity[label, obs_idx] += joint[labels[:, slot] == label].sum()
    return updated, identity


def acquisition_probability(probability, labels, target_label):
    if target_label == FORK:
        return probability.prod(axis=1) * (labels[:, 0] == LOWER) * (labels[:, 1] == UPPER)
    if target_label == JUNCTION:
        return 1.0 - np.prod(1.0 - probability, axis=1)
    return np.sum(probability * (labels == target_label), axis=1)


def predictive_bins(mu, probability):
    """Distribution of the leftmost reported centre's bin, including empty.

    Used only in one-step lookahead. Online estimation uses the complete
    unlabelled (lateral,depth) set and the exact assignment likelihood.
    """
    edges = np.array([-np.inf, -4.0, 0.0, 4.0, np.inf])
    cdf = ndtr((edges - mu[:, :, 0, None]) / SIGMA_MM[0])
    uniform_cdf = np.clip((edges + HALF_FOV_MM) / (2.0 * HALF_FOV_MM), 0.0, 1.0)
    survives = np.prod(1.0 - probability[..., None] * cdf, axis=1) * (
        1.0 - CLUTTER_PROBABILITY * uniform_cdf
    )
    return np.column_stack((survives[:, :-1] - survives[:, 1:], survives[:, -1]))


@dataclass
class Belief:
    theta: np.ndarray
    log_weights: np.ndarray
    events: set

    @classmethod
    def initialize(cls, shared=True):
        theta, weights = prior(shared)
        return cls(theta, np.log(weights), set())

    @property
    def weights(self):
        return np.exp(self.log_weights)

    def update(self, event_id, geometry, xy, observations):
        if event_id in self.events:
            return None
        updated, identities = observe(self.log_weights, *geometry.predict(self.theta, xy), observations)
        self.log_weights = updated
        self.events.add(event_id)
        return identities


def graph_nodes(theta: np.ndarray) -> np.ndarray:
    """Read out the entire Y in coordinates relative to its uncertain fork."""
    nodes = []
    for label, offsets in ((TRUNK, np.linspace(-32.0, 0.0, 9)),
                           (LOWER, np.linspace(4.0, 32.0, 8)),
                           (UPPER, np.linspace(4.0, 32.0, 8))):
        nodes.extend(graph_point(theta, theta[:, 4] + off, label) for off in offsets)
    return np.stack(nodes, axis=1)

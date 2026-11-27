"""Continuous graph posterior from one replay of the retained observations.

The coarse grid proposes hypotheses and supports online planning. Continuous
importance samples use the declared prior and complete unordered-set likelihood.
Fitting a proposal to the history does not add an observation factor.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from bifurcation_model import (
    CELL_HALF_WIDTHS, FORK, JUNCTION, LOWER, SIGMA_MM, TRUNK, UPPER,
    assignment_terms, graph_nodes, graph_point, prior,
)


SAMPLE_COUNT = 2048
MIN_ESS = 64.0
PRIOR_FRACTION = .125
JOINT_IDENTITY_GATE = .90


@dataclass(frozen=True)
class Event:
    xy: np.ndarray
    uv: np.ndarray
    association: np.ndarray


def physical_coefficients(theta):
    return np.column_stack((theta[:, :1] + theta[:, 1:4], theta[:, 4:]))


def graph_coefficients(z):
    return np.column_stack((np.zeros(len(z)), z))


def one_dimensional_density(values, centers, weights, width):
    return (np.abs(values[:, None] - centers) <= width) @ (weights / (2. * width))


def graph_prior_density(z, shared):
    """Marginalize the unidentifiable split of shared and local displacement.

    z is (beta+w_trunk, beta+w_lower, beta+w_upper, fork, depth). Given
    mixture centres, the common beta integration is an interval intersection.
    """
    c, w = prior()
    mask = (c[:, 4] == 0.) & (c[:, 5] == 0.)
    c, w = c[mask, :4], w[mask]
    w = w / w.sum()
    h_beta, h_local = CELL_HALF_WIDTHS[:2]
    if shared:
        implied_beta = z[:, None, :3] - c[None, :, 1:4]
        lo = np.maximum(c[None, :, 0] - h_beta, (implied_beta - h_local).max(axis=-1))
        hi = np.minimum(c[None, :, 0] + h_beta, (implied_beta + h_local).min(axis=-1))
        density = np.maximum(hi - lo, 0.) @ (w / ((2. * h_beta) * (2. * h_local)**3))
    else:
        density = np.ones(len(z))
        for i in range(3):
            implied_beta = z[:, i, None] - c[None, :, i+1]
            lo = np.maximum(c[None, :, 0] - h_beta, implied_beta - h_local)
            hi = np.minimum(c[None, :, 0] + h_beta, implied_beta + h_local)
            density *= np.maximum(hi - lo, 0.) @ (w / (4. * h_beta * h_local))
    for i, centers in ((3, [-4., 0., 4.]), (4, [-3., 0., 3.])):
        density *= one_dimensional_density(z[:, i], centers, np.array([.25, .5, .25]), CELL_HALF_WIDTHS[i+1])
    return density


def sample_prior(rng, n, shared):
    theta, weights = prior()
    base = theta[rng.choice(len(theta), size=n, p=weights)]
    draw = physical_coefficients(base + rng.uniform(-1., 1., (n, 6)) * CELL_HALF_WIDTHS)
    if not shared:
        for i in (1, 2):
            c = theta[rng.choice(len(theta), size=n, p=weights)]
            draw[:, i] = c[:, 0] + c[:, i+1] + (
                rng.uniform(-1., 1., n) * CELL_HALF_WIDTHS[0]
                + rng.uniform(-1., 1., n) * CELL_HALF_WIDTHS[i+1]
            )
    return draw


def proposal_models(geometry, coarse, events):
    # The public association probabilities are used solely to place efficient
    # proposals. All frames, including ambiguities and misses, enter the exact
    # importance weights below.
    xs, views, observations, identities, scores = [], [], [], [], []
    for event in events:
        for i, obs in enumerate(event.uv):
            label = int(np.argmax(event.association[:, i]))
            confidence = event.association[label, i]
            if confidence >= .90:
                xs.append(event.xy[0])
                views.append(event.xy)
                observations.append(obs)
                identities.append(label)
                scores.append(confidence)
    xs, identities = np.asarray(xs), np.asarray(identities)
    observed = np.asarray(observations).reshape(-1, 2)
    skin = geometry.skin_to_uv(np.asarray(views).reshape(-1, 2)) if xs.size else observed
    weight = np.sqrt(np.asarray(scores))[:, None] / SIGMA_MM
    z = physical_coefficients(coarse.theta)
    probs = coarse.weights

    def residual(t, anchor):
        coeff = np.broadcast_to(graph_coefficients(t[None]), (len(xs), 6))
        if xs.size:
            trunk, lower, upper = (graph_point(coeff, xs, label) for label in (TRUNK, LOWER, UPPER))
            common = np.where((xs > t[3])[:, None], (lower + upper) / 2., trunk)
            points = np.where((identities == LOWER)[:, None], lower,
                              np.where((identities == UPPER)[:, None], upper, common))
            errors = ((geometry.body_to_uv(points) - skin - observed) * weight).ravel()
        else:
            errors = np.empty(0)
        return np.r_[errors, (t-anchor) / [7., 7., 7., 3., 3.]]

    models = []
    # Retain all three fork modes. The continuous support has gaps between them.
    for f in (-4., 0., 4.):
        mask = z[:, 3] == f
        inds = np.flatnonzero(mask)
        best = inds[int(np.argmax(probs[inds]))]
        anchor = z[best].copy()
        lower = np.array([-14.95, -14.95, -14.95, f-1.2999, -4.1499])
        upper = np.array([14.95, 14.95, 14.95, f+1.2999, 4.1499])
        fit = least_squares(residual, np.clip(anchor, lower + 1.e-5, upper - 1.e-5), args=(anchor,),
                            bounds=(lower, upper), loss="soft_l1", max_nfev=36)
        precision = fit.jac.T @ fit.jac + np.diag([.035, .035, .035, .23, .16])
        covariance = 1.65 * np.linalg.inv(precision) + .035**2 * np.eye(5)
        models.append(multivariate_normal(fit.x, covariance))
    return models


def refine(geometry, coarse, events, seed, shared):
    models = proposal_models(geometry, coarse, events)
    rng = np.random.default_rng(seed)
    prior_count = int(SAMPLE_COUNT * PRIOR_FRACTION)
    counts = np.full(len(models), (SAMPLE_COUNT-prior_count) // len(models))
    counts[0] += SAMPLE_COUNT-prior_count - counts.sum()
    samples = np.concatenate([sample_prior(rng, prior_count, shared),
                              *[m.rvs(size=int(n), random_state=rng) for m, n in zip(models, counts)]])
    density = graph_prior_density(samples, shared)
    proposals = [np.log(n / SAMPLE_COUNT) + m.logpdf(samples) for m, n in zip(models, counts)]
    with np.errstate(divide="ignore"):
        log_prior = np.log(density)
    log_proposal = logsumexp(np.vstack((np.log(prior_count / SAMPLE_COUNT) + log_prior, *proposals)), axis=0)
    valid = np.isfinite(log_prior)
    theta = graph_coefficients(samples[valid])
    log_weights = log_prior[valid] - log_proposal[valid]
    for event in events:
        means, detect, _ = geometry.predict(theta, event.xy)
        terms = list(assignment_terms(means, detect, event.uv))
        if not terms:
            return None
        log_weights += logsumexp([score for _, score in terms], axis=0)
    normalizer = logsumexp(log_weights)
    if not np.isfinite(normalizer):
        return None
    log_weights -= normalizer
    weights = np.exp(log_weights)
    world = geometry.body_world(graph_nodes(theta))
    mean = weights @ world.reshape(len(world), -1)
    mean = mean.reshape(-1, 3)
    variance = np.einsum("i,ijk->jk", weights, (world-mean)**2)
    deviations = np.linalg.norm(world-mean, axis=-1).max(axis=1)
    order = np.argsort(deviations)
    quantile = min(len(order)-1, np.searchsorted(np.cumsum(weights[order]), .9))
    return {"theta": theta, "log_weights": log_weights, "readout": (mean, variance, float(deviations[order[quantile]])),
            "ess": float(1. / (weights @ weights)), "support_samples": int(len(theta))}


def association_probability(geometry, theta, event, indices, section_label):
    """Per-hypothesis conditional probability of the retained assignment."""
    means, detect, labels = geometry.predict(theta, event.xy)
    terms = list(assignment_terms(means, detect, event.uv))
    total = logsumexp([score for _, score in terms], axis=0)
    accepted = np.full(len(theta), -np.inf)
    for assignment, score in terms:
        if section_label == FORK:
            valid = np.zeros(len(theta), dtype=bool)
            if assignment == tuple(indices):
                valid = (labels[:, 0] == LOWER) & (labels[:, 1] == UPPER)
        else:
            valid = np.zeros(len(theta), dtype=bool)
            for slot, idx in enumerate(assignment):
                if idx == indices[0]:
                    valid |= labels[:, slot] >= 0 if section_label == JUNCTION else labels[:, slot] == section_label
        accepted = np.logaddexp(accepted, np.where(valid, score, -np.inf))
    return np.exp(accepted - np.where(np.isfinite(total), total, 0.))


def review_coverage(geometry, theta, log_weights, evidence):
    """Review all section identities under the final map without recounting frames."""
    weights = np.exp(log_weights)
    per_section = {}
    joint = np.ones(len(theta))
    for section, records in evidence.items():
        best = None
        for event_id, event, indices in records:
            conditional = association_probability(geometry, theta, event, indices, section.label)
            p = float(weights @ conditional)
            if best is None or p > best[0]:
                best = p, conditional, event_id
        per_section[section] = (best[0], best[2])
        joint *= best[1]
    return float(weights @ joint), per_section

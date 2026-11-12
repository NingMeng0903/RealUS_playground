"""Shared stochastic observation program for the two moving-Y planners."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from belief import (CONFIDENCE, DEPTH, HALF_WIDTH, IMAGE_SD, NEAR_CLUTTER_RATE,
                    NEAR_CLUTTER_SD, OUTLIER_RATE, OUTLIER_SD)


OUTER_DRAWS = 32
INNER_DRAWS = 128
OUTER_SEED = 842991
INNER_SEED = 842997
WRONG_ASSOCIATION_COST = 1.
TRAVEL_COST_PER_MM = .00045
TIME_COST_PER_S = .055
LOWER = np.array([-HALF_WIDTH, DEPTH[0]])
UPPER = np.array([HALF_WIDTH, DEPTH[1]])


def fixed_nodes(size, seed):
    return qmc.Sobol(d=16, scramble=True, seed=seed).random_base2(int(np.log2(size)))


OUTER = fixed_nodes(OUTER_DRAWS, OUTER_SEED)
INNER = fixed_nodes(INNER_DRAWS, INNER_SEED)


def motion_cost(path):
    return TIME_COST_PER_S * path.duration + TRAVEL_COST_PER_MM * path.cost_mm


def simulated_masks(world, belief, reading, goal, uniforms):
    """Unlabeled aperture-gated masks from the frozen MapFilter public model."""
    prior, _ = belief.prior_at(reading)
    site, view = goal
    base, rows = world.projection(reading.q, site, view)
    centers = belief.means(base, rows, reading).reshape(-1, len(base), 2)
    cumulative = np.cumsum(prior.ravel())
    cumulative[-1] = 1.
    sampled = centers[np.searchsorted(cumulative, uniforms[:, 0], side='right')]
    sigma = np.where(uniforms[:, 3:5, None] < OUTLIER_RATE, OUTLIER_SD, IMAGE_SD)
    errors = ndtri(uniforms[:, 5:9]).reshape(-1, 2, 2) * sigma
    values = sampled + errors
    p_detect = world.detection_probability(reading.q, site, view)
    visible = ((uniforms[:, 1:3] < p_detect)
               & np.all((values >= LOWER) & (values <= UPPER), axis=2))
    # Inverse-CDF samples for the exactly aperture-truncated near component.
    from scipy.special import ndtr
    center = base.mean(axis=0)
    low, high = ndtr((LOWER - center) / NEAR_CLUTTER_SD), ndtr((UPPER - center) / NEAR_CLUTTER_SD)
    near = center + NEAR_CLUTTER_SD * ndtri(low + uniforms[:, 11:13] * (high - low))
    uniform = LOWER + uniforms[:, 13:15] * (UPPER - LOWER)
    clutter = np.where((uniforms[:, 10] < NEAR_CLUTTER_RATE)[:, None], near, uniform)
    false_present = uniforms[:, 9] < world.clutter_probability(site, view)
    result = []
    for value, keep, false, add in zip(values, visible, clutter, false_present):
        obs = value[keep]
        if add:
            obs = np.vstack((obs, false))
        result.append(obs[np.argsort(obs[:, 0], kind='stable')])
    return tuple(result)


def primary_utility(probability):
    return np.where(probability >= CONFIDENCE,
                    probability - WRONG_ASSOCIATION_COST * (1. - probability), 0.)


@dataclass
class Kernel:
    density: np.ndarray
    daughter: np.ndarray
    proposal_density: np.ndarray


def kernel(world, belief, reading, goal, uniforms):
    masks = simulated_masks(world, belief, reading, goal, uniforms)
    density = np.empty((len(masks), *belief.weights.shape), dtype=float)
    daughter = np.zeros((3, *density.shape), dtype=float)
    for row, obs in enumerate(masks):
        likelihood, assignment = belief.emission(world, reading, *goal, obs)
        density[row] = likelihood
        daughter[:len(obs), row] = assignment[:, 1]
    prior, _ = belief.prior_at(reading)
    return Kernel(density, daughter, density.reshape(len(masks), -1) @ prior.ravel())


def posterior_rows(prior, likelihoods, frozen_marginal):
    unnormalized = prior[None] * likelihoods
    if frozen_marginal is None:
        return unnormalized / unnormalized.sum(axis=(1, 2))[:, None, None]
    normalizer = unnormalized.sum(axis=2)
    scale = np.divide(frozen_marginal[None], normalizer, out=np.zeros_like(normalizer),
                      where=normalizer > 0.)
    return unnormalized * scale[:, :, None]


def tail_values(priors, k, frozen_marginal):
    """Self-normalized predictive importance quadrature on shared full masks."""
    n_observations = len(k.density)
    if frozen_marginal is None:
        weights = priors.reshape(len(priors), -1)
        density = weights @ k.density.reshape(n_observations, -1).T
        joint = weights @ k.daughter.reshape(3 * n_observations, -1).T
        joint = joint.reshape(len(priors), 3, n_observations).max(axis=1)
        confidence = np.divide(joint, density, out=np.zeros_like(density), where=density > 0.)
    else:
        static_priors = priors.transpose(1, 0, 2)
        by_row = static_priors @ k.density.transpose(1, 2, 0)
        density = by_row.sum(axis=0)
        scale = np.divide(frozen_marginal[:, None, None], by_row, out=np.zeros_like(by_row),
                          where=by_row > 0.)
        confidence = np.maximum.reduce([
            np.sum((static_priors @ d.transpose(1, 2, 0)) * scale, axis=0)
            for d in k.daughter
        ])
    importance = density / k.proposal_density[None]
    importance /= importance.sum(axis=1)[:, None]
    scores = np.sum(importance * primary_utility(confidence), axis=1)
    ess = 1. / np.sum(importance**2, axis=1)
    return scores, ess


@dataclass
class Programs:
    first: tuple
    tails: tuple
    values: np.ndarray
    ess: np.ndarray


def forecast(world, belief, first, tails):
    """Evaluate the same Q(y, a1, a2) for all first/tail decisions."""
    if len({p.duration for p in first}) != 1 or len({p.duration for p in tails}) != 1:
        raise ValueError('predictive sharing requires a common clock at each stage')
    first_reading = belief.predicted_reading(belief.reading.t + first[0].duration)
    tail_reading = belief.predicted_reading(first_reading.t + tails[0].duration)
    first_prior, _ = belief.prior_at(first_reading)
    transition = belief.transition(tail_reading.t - first_reading.t,
                                  .5 * (first_reading.velocity + tail_reading.velocity))
    branch_priors = []
    for p in first:
        first_masks = simulated_masks(world, belief, first_reading, p.goal, OUTER)
        likelihoods = np.stack([belief.emission(world, first_reading, *p.goal, obs)[0]
                                for obs in first_masks])
        branch_priors.append(posterior_rows(first_prior, likelihoods, belief.frozen_marginal) @ transition)
    priors = np.concatenate(branch_priors, axis=0)
    values, ess = [], []
    for p in tails:
        k = kernel(world, belief, tail_reading, p.goal, INNER)
        score, effective = tail_values(priors, k, belief.frozen_marginal)
        values.append(score.reshape(len(first), OUTER_DRAWS))
        ess.append(effective.reshape(len(first), OUTER_DRAWS))
    return Programs(tuple(p.goal[1] for p in first), tuple(p.goal[1] for p in tails),
                    np.stack(values, axis=2), np.stack(ess, axis=2))


def choose_program(program, first, tails_by_exit, *, feedback):
    first_values, candidates = [], []
    for move in first:
        idx = program.first.index(move.goal[1])
        tails = tails_by_exit[move.goal[1]]
        if not tails:
            continue
        columns = [program.tails.index(path.goal[1]) for path in tails]
        adjusted = program.values[idx][:, columns] - np.array([motion_cost(path) for path in tails])[None]
        open_mean = adjusted.mean(axis=0)
        open_idx = int(np.argmax(open_mean))
        adaptive_idx = np.argmax(adjusted, axis=1)
        adaptive_value = float(np.mean(adjusted[np.arange(OUTER_DRAWS), adaptive_idx]))
        base = float(open_mean[open_idx])
        score = (adaptive_value if feedback else base) - motion_cost(move)
        first_values.append(score)
        candidates.append(dict(first_path=move.id, first_view=move.goal[1],
                               legal_tail_paths=[p.id for p in tails],
                               legal_tail_views=[p.goal[1] for p in tails],
                               open_loop_tail_view=tails[open_idx].goal[1],
                               open_loop_tail_scores=open_mean.tolist(),
                               adaptive_tail_histogram={str(p.goal[1]): int(np.count_nonzero(adaptive_idx == k))
                                                        for k, p in enumerate(tails)},
                               conditional_tail_value=adaptive_value,
                               open_loop_tail_value=base, recourse_gap=adaptive_value - base,
                               inner_ess_min=float(program.ess[idx][:, columns].min()),
                               inner_ess_mean=float(program.ess[idx][:, columns].mean()),
                               score=score))
    if not first_values:
        raise ValueError('no first path has a legal continuation')
    selected = candidates[int(np.argmax(first_values))]
    return selected, candidates


def choose_real_tail(world, belief, tails):
    reading = belief.predicted_reading(belief.reading.t + tails[0].duration)
    prior, _ = belief.prior_at(reading)
    values, ess = [], []
    for p in tails:
        k = kernel(world, belief, reading, p.goal, INNER)
        score, effective = tail_values(prior[None], k, belief.frozen_marginal)
        values.append(float(score[0]) - motion_cost(p))
        ess.append(float(effective[0]))
    return tails[int(np.argmax(values))], dict(values=values, ess=ess,
                                               paths=[path.id for path in tails],
                                               views=[path.goal[1] for path in tails])

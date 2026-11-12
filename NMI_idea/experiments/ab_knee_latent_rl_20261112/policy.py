"""Shared features and observation-conditioned finite-horizon rule."""
from __future__ import annotations

import numpy as np

from belief import (CONFIDENCE, DEPTH, HALF_WIDTH, IMAGE_SD, NEAR_CLUTTER_RATE,
                    NEAR_CLUTTER_SD, OUTLIER_RATE, OUTLIER_SD)


RULE_OUTCOMES = 5
RULE_DEPTH = 2


def travel_cost(action):
    return .055 * action.duration + .00045 * action.distance_mm


def terminal_score(both, strict, correct, wrong, forecast_rmse):
    return 1.45 * both + .45 * strict + .20 * correct - 2.5 * wrong - .04 * min(forecast_rmse, 15.)


def leaf_value(world, belief):
    _, confidence = belief.decisions()
    confirmed = confidence >= CONFIDENCE
    p_correct = np.sum(confirmed * confidence)
    p_wrong = np.sum(confirmed * (1. - confidence))
    p_both = float(np.prod(confirmed * confidence))
    mean, _ = belief.stats()
    _, design = world.world_design(belief.reading.q, world.query_sites)
    err = np.einsum("wzk,nck->wznc", belief.parameters - mean, design)
    in_band = np.max(np.linalg.norm(err, axis=-1), axis=-1) <= 3.
    p_strict = p_both * np.sum(belief.weights * in_band)
    _, sd = belief.forecast(world, belief.reading.t + .5, world.query_sites)
    rmse = float(np.sqrt(np.mean(np.sum(sd**2, axis=-1))))
    return terminal_score(p_both, p_strict, p_correct, p_wrong, rmse)


def proposal_features(world, belief, actions, remaining, maximum):
    """Candidates share a public belief summary; rows contain action descriptors."""
    mean, sd = belief.stats()
    _, confidence = belief.decisions()
    state = np.r_[remaining / maximum, (belief.reading.q - 15.) / 4.,
                  belief.reading.velocity / 8., mean / 5., sd / 5., confidence,
                  len(belief.witnesses) / 30.]
    n_slots = len(actions)
    features = np.zeros((n_slots, len(state) + 11 + n_slots), np.float32)
    mask = np.array([a is not None for a in actions])
    for slot in np.flatnonzero(mask):
        action = actions[slot]
        reading = belief.predicted_reading(action.t)
        weights, _ = belief.prior_at(reading)
        centers, hit = belief.imaging(world, reading, action.site, action.view)
        mu = np.einsum("wz,wzld->ld", weights, centers)
        pd = np.einsum("wz,wzl->l", weights, hit)
        sep = float(np.linalg.norm(mu[1] - mu[0])) if len(mu) == 2 else 0.
        details = [
            action.site / max(1, world.n - 1), action.view / max(1, world.n_views - 1),
            action.duration / 2., action.distance_mm / 100.,
            (reading.q - 15.) / 4., reading.velocity / 8.,
            float(pd[0]), float(pd[1]) if len(pd) == 2 else 0.,
            sep / 12., world.clutter_probability(action.site, action.view), len(mu) - 1.,
        ]
        features[slot] = np.r_[state, details, np.eye(n_slots)[slot]]
    return features, mask


def imagined_frames(world, belief, action, draws, *, depth):
    """Stratified draws from the policy's predictive distribution only."""
    reading = belief.predicted_reading(action.t)
    weights, _ = belief.prior_at(reading)
    base, rows = world.projection(reading.q, action.site, action.view)
    centers = belief.means(base, rows, reading).reshape(-1, len(base), 2)
    pd = world.detection_probability(reading.q, action.site, action.view)
    cumulative = np.cumsum(weights.ravel())
    random = np.random.default_rng(35039 + 41 * depth)
    offset = random.uniform(.15, .85)
    ids = np.searchsorted(cumulative, (np.arange(draws) + offset) / draws)
    n_lumens = len(base)
    u = random.random((draws, n_lumens))
    sd = np.where(random.random((draws, n_lumens, 1)) < OUTLIER_RATE, OUTLIER_SD, IMAGE_SD)
    noise = random.normal(size=(draws, n_lumens, 2)) * sd
    clutter_draw = random.random(draws)
    near_draw = random.random(draws)
    lower, upper = np.array([-HALF_WIDTH, DEPTH[0]]), np.array([HALF_WIDTH, DEPTH[1]])
    for j, index in enumerate(ids):
        values = centers[index] + noise[j]
        visible = (u[j] < pd) & np.all((values >= lower) & (values <= upper), axis=-1)
        obs = values[visible]
        if clutter_draw[j] < world.clutter_probability(action.site, action.view):
            if near_draw[j] < NEAR_CLUTTER_RATE:
                while True:
                    clutter = random.normal(base.mean(axis=0), NEAR_CLUTTER_SD)
                    if np.all((clutter >= lower) & (clutter <= upper)):
                        break
            else:
                clutter = random.uniform(lower, upper)
            obs = np.vstack((obs, clutter))
        yield reading, obs


def rule_scores(world, belief, previous, actions, remaining, *, depth=RULE_DEPTH):
    scores = np.full(len(actions), -np.inf)
    for j, action in enumerate(actions):
        if action is None:
            continue
        outcomes = []
        draws = RULE_OUTCOMES if depth == RULE_DEPTH else 3
        for reading, obs in imagined_frames(world, belief, action, draws, depth=depth):
            updated = belief.copy()
            updated.observe(world, reading, action.site, action.view, obs,
                            event=-remaining, retain=world.identity_site(action.site))
            if depth > 1 and remaining > 1:
                tail = world.candidates(reading, action)
                values = rule_scores(world, updated, action, tail, remaining - 1, depth=depth - 1)
                outcomes.append(float(np.max(values)) if np.any(np.isfinite(values)) else leaf_value(world, updated))
            else:
                outcomes.append(leaf_value(world, updated))
        scores[j] = float(np.mean(outcomes)) - travel_cost(action)
    return scores


def fast_behavior_scores(features, mask):
    """Low-cost logging policy, mixed with uniform legal exploration."""
    n_slots = len(mask)
    details = features[:, -11-n_slots:-n_slots]
    score = (.7 * details[:, 6] + .9 * details[:, 7] + .14 * details[:, 8]
             - .38 * details[:, 2] - .2 * details[:, 3] - .7 * details[:, 9])
    return np.where(mask, score, -np.inf)

"""Two-capture MPC with posterior association of an unordered return packet."""
from __future__ import annotations

import math

from numba import njit
import numpy as np

from control import (CONFIDENCE, MPC_SCENARIOS, RISK_PENALTY, allowed, dot_model,
                     motion_price, remaining_value)


@njit(cache=True)
def best_candidate(prior, joint, bins):
    """Choose a single public slot using the belief marginal, before truth."""
    best, chosen = 0., -1
    for j in range(len(bins)):
        if bins[j] > 0:
            mass = dot_model(prior, joint[j])
            if mass > best:
                best, chosen = mass, j
    return chosen, best


@njit(cache=True)
def posterior(prior, like, joint, bins, clock, address, packet):
    ordinary = like[clock, address, packet]
    products = prior * ordinary
    den = products.sum()
    if den <= 1.e-260:
        return prior.copy(), 0., -1
    chosen, mass = best_candidate(prior, joint[clock, address, packet], bins[packet])
    return products / den, mass / den, chosen


@njit(cache=True)
def advance(prior, good, like, joint, clock, address, packet, chosen, claim):
    ordinary = like[clock, address, packet]
    probability = dot_model(prior, ordinary)
    if probability <= 1.e-260:
        return good.copy(), 0.
    inherited = dot_model(good, ordinary) / probability
    evidence = joint[clock, address, packet, chosen] if claim else ordinary
    revised = good * evidence / probability
    return revised, max(0., inherited - revised.sum())


@njit(cache=True)
def expected_claim(prior, good, like, joint, claimable, bins, clock, address):
    """Exact second-observation expectation on the finite packet alphabet."""
    yes, wrong = 0., 0.
    for packet in range(1, like.shape[2]):
        if not claimable[clock, address, packet]:
            continue
        ordinary = like[clock, address, packet]
        probability = dot_model(prior, ordinary)
        if probability <= 1.e-260:
            continue
        candidate_joint = joint[clock, address, packet]
        chosen, mass = best_candidate(prior, candidate_joint, bins[packet])
        if mass >= CONFIDENCE * probability:
            accepted = dot_model(good, candidate_joint[chosen])
            yes += accepted
            wrong += max(0., dot_model(good, ordinary) - accepted)
    return yes, wrong


@njit(cache=True)
def action(model, prior, good, at, slot, clock, progress, reserve):
    """Keep the incumbent horizon, value, costs, risk and receding update."""
    like, joint, claimable, bins, addresses, sites, costs, targets, potential, needs = model
    cap = len(potential) - 1
    if slot >= cap or progress >= len(targets):
        return -1
    best, selected = -math.inf, -1
    for first in range(len(sites)):
        if not allowed(costs, at, first, reserve):
            continue
        first_address = addresses[first]
        first_price = motion_price(costs, at, first)
        first_clock = clock + int(costs[at, first, 3])
        if slot + 1 == cap:
            if sites[first] != targets[progress] or progress + 1 != len(targets):
                continue
            c, w = expected_claim(prior, good, like, joint, claimable, bins, first_clock, first_address)
            value = c - RISK_PENALTY * w - first_price
            if value > best + 1.e-12:
                best, selected = value, first
            continue
        remaining = reserve - costs[at, first]
        masses = np.empty(like.shape[2])
        for packet in range(len(masses)):
            masses[packet] = dot_model(prior, like[first_clock, first_address, packet])
        cum = np.cumsum(masses)
        cum[-1] = 1.
        next_prior = np.empty((MPC_SCENARIOS, len(prior)))
        next_good = np.empty_like(next_prior)
        next_progress = np.empty(MPC_SCENARIOS, dtype=np.int64)
        good_mass = np.empty(MPC_SCENARIOS)
        loss = np.empty(MPC_SCENARIOS)
        first_value = 0.
        for i in range(MPC_SCENARIOS):
            packet = min(np.searchsorted(cum, (i + .5) / MPC_SCENARIOS), len(cum) - 1)
            post, conf, candidate = posterior(prior, like, joint, bins, first_clock, first_address, packet)
            claim = sites[first] == targets[progress] and candidate >= 0 and conf >= CONFIDENCE
            next_prior[i] = post
            next_progress[i] = progress + int(claim)
            g, new_loss = advance(prior, good, like, joint, first_clock, first_address, packet, candidate, claim)
            next_good[i] = g
            good_mass[i] = g.sum()
            loss[i] = new_loss
            future = remaining_value(potential, needs, cap - slot - 1, first, next_progress[i], remaining)
            first_value += good_mass[i] * future - RISK_PENALTY * loss[i]
        value = first_value / MPC_SCENARIOS - first_price
        if value > best + 1.e-12:
            best, selected = value, first
        for second in range(len(sites)):
            if not allowed(costs, first, second, remaining):
                continue
            second_price = motion_price(costs, first, second)
            after_second = remaining - costs[first, second]
            second_clock = first_clock + int(costs[first, second, 3])
            value = 0.
            for i in range(MPC_SCENARIOS):
                after = next_progress[i]
                if after == len(targets):
                    value += good_mass[i] - RISK_PENALTY * loss[i]
                    continue
                old = remaining_value(potential, needs, cap - slot - 2, second, after, after_second)
                increment, wrong = 0., 0.
                if sites[second] == targets[after]:
                    c, wrong = expected_claim(next_prior[i], next_good[i], like, joint, claimable, bins,
                                              second_clock, addresses[second])
                    advanced = remaining_value(potential, needs, cap - slot - 2, second, after + 1, after_second)
                    increment = c * advanced - (c + wrong) * old
                value += (good_mass[i] * old + increment - RISK_PENALTY * (wrong + loss[i]) - second_price)
            value = value / MPC_SCENARIOS - first_price
            if value > best + 1.e-12:
                best, selected = value, first
    return selected if best > 0. else -1

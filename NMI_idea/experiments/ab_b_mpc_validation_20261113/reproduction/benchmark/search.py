"""Receding-horizon planners for fresh-image scan tasks.

The planning observation is the public verifier's accept/reject event.
Incorrect acceptance is absorbing for joint success and the one-time
wrong-claim penalty. Motion is a hard constraint and measured separately.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _can_finish(pending, remaining):
    if pending == 3:
        return remaining >= 2
    return pending == 0 or remaining >= 1


@njit(cache=True)
def _fixed(scores, edges, bits, step, at, budget, mass, wrong_cost):
    """Best fixed sequence, even though real execution replans every frame."""
    if step == len(scores):
        return mass[0], -1, 1
    best = mass[0]
    selected = -1
    nodes = 1
    for nxt in range(len(bits)):
        expense = edges[at, nxt]
        if np.any(expense > budget):
            continue
        bit = bits[nxt]
        correct, wrong = scores[step, nxt]
        updated = mass.copy()
        moving_mass = 0.
        for need in range(1, len(mass)):
            amount = mass[need]
            if not _can_finish(need, len(scores) - step):
                updated[need] -= amount
                continue
            moving_mass += amount
            if need & bit:
                updated[need] -= amount * (correct + wrong)
                updated[need ^ bit] += amount * correct
        if moving_mass <= 1.e-14:
            continue
        value, _, count = _fixed(
            scores, edges, bits, step + 1, nxt, budget - expense,
            updated, wrong_cost)
        risk = 0.
        for need in range(1, len(mass)):
            if need & bit and _can_finish(need, len(scores) - step):
                risk += mass[need] * wrong
        value -= wrong_cost * risk
        nodes += count
        if value > best + 1.e-12:
            best = value
            selected = nxt
    return best, selected, nodes


@njit(cache=True)
def _tail_upper(upper, step, pending):
    return upper[step, pending]


@njit(cache=True)
def _feedback(scores, edges, bits, upper, step, at, budget, pending,
              wrong_cost, depth, prune):
    if pending == 0:
        return 1., -1, 1
    if not _can_finish(pending, len(scores) - step):
        return 0., -1, 1
    if depth == 0:
        mass = np.zeros(4)
        mass[pending] = 1.
        return _fixed(scores, edges, bits, step, at, budget, mass, wrong_cost)
    best = 0.
    selected = -1
    nodes = 1
    for nxt in range(len(bits)):
        expense = edges[at, nxt]
        if np.any(expense > budget):
            continue
        bit = bits[nxt]
        correct, wrong = scores[step, nxt]
        if not pending & bit:
            correct, wrong = 0., 0.
        reject = 1. - correct - wrong
        resolved = pending & ~bit
        immediate_cost = wrong_cost * wrong
        if prune:
            ceiling = (correct * _tail_upper(upper, step + 1, resolved)
                       + reject * _tail_upper(upper, step + 1, pending)
                       - immediate_cost)
            if ceiling < best - 1.e-12:
                continue
        accept_value = 0.
        if correct > 1.e-14:
            accept_value, _, count = _feedback(
                scores, edges, bits, upper, step + 1, nxt, budget - expense, resolved, wrong_cost, depth - 1, prune)
            nodes += count
        reject_value = 0.
        if reject > 1.e-14:
            reject_value, _, count = _feedback(
                scores, edges, bits, upper, step + 1, nxt, budget - expense, pending, wrong_cost, depth - 1, prune)
            nodes += count
        value = correct * accept_value + reject * reject_value - immediate_cost
        if value > best + 1.e-12:
            best = value
            selected = nxt
    return best, selected, nodes


@njit(cache=True)
def _relaxed_completion(scores, bits):
    """Optimistic success bound with travel and wrong claims removed."""
    upper = np.zeros((len(scores) + 1, 4))
    upper[:, 0] = 1.
    for step in range(len(scores) - 1, -1, -1):
        for pending in range(1, 4):
            best = upper[step + 1, pending]
            for nxt in range(len(bits)):
                bit = bits[nxt]
                if pending & bit:
                    correct = scores[step, nxt, 0]
                    value = (correct * upper[step + 1, pending ^ bit]
                             + (1. - correct) * upper[step + 1, pending])
                    best = max(best, value)
            upper[step, pending] = best
    return upper


def choose(scores, edges, bits, at, budget, pending, method, wrong_cost=1.):
    """Return a forecast utility, first action and counted search nodes."""
    if method == "MPC":
        mass = np.zeros(4)
        mass[pending] = 1.
        return _fixed(scores, edges, bits, 0, at, budget, mass, wrong_cost)
    depth = 1 if method == "MPC_rec1" else len(scores)
    upper = _relaxed_completion(scores, bits)
    return _feedback(scores, edges, bits, upper, 0, at, budget, pending,
                     wrong_cost, depth, method == "B_bound")

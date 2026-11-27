"""Bounded association search with a shared frame-pose Gaussian."""
from math import log, pi, sqrt

import numpy as np
from numba import njit


_LOG_2PI = log(2. * pi)
_MIN_WIDTH = 12


@njit(inline="always")
def _offer(order, scores, size, limit, first_slot, score):
    """Insert into a bounded stable ranking without moving the stored payloads."""
    if size == limit:
        pos = size - 1
        slot = order[pos]
        if score <= scores[slot]:
            return -1
    else:
        pos = size
        slot = first_slot + size
    while pos and score > scores[order[pos - 1]]:
        order[pos] = order[pos - 1]
        pos -= 1
    order[pos] = slot
    scores[slot] = score
    return slot


@njit
def _extension(joint, code, x, y, chain, count, codes, rows, blocks, cross):
    """Append two Cholesky rows and score the standardized innovation."""
    i = 2 * code
    s00 = joint[i, i]
    s10 = joint[i + 1, i]
    s11 = joint[i + 1, i + 1]
    for p in range(count):
        node = chain[count - 1 - p]
        j = 2 * codes[node]
        c00 = joint[i, j]
        c01 = joint[i, j + 1]
        c10 = joint[i + 1, j]
        c11 = joint[i + 1, j + 1]
        a = 2 * p
        for k in range(a):
            l0 = rows[node, 0, k]
            l1 = rows[node, 1, k]
            u = cross[0, k]
            v = cross[1, k]
            c00 -= u * l0
            c01 -= u * l1
            c10 -= v * l0
            c11 -= v * l1
        inv0, l10, inv1 = blocks[node, 0], blocks[node, 1], blocks[node, 2]
        z0, z1 = blocks[node, 3], blocks[node, 4]
        u0 = c00 * inv0
        v0 = c10 * inv0
        u1 = (c01 - u0 * l10) * inv1
        v1 = (c11 - v0 * l10) * inv1
        cross[0, a] = u0
        cross[1, a] = v0
        cross[0, a + 1] = u1
        cross[1, a + 1] = v1
        s00 -= u0 * u0 + u1 * u1
        s10 -= v0 * u0 + v1 * u1
        s11 -= v0 * v0 + v1 * v1
        x -= u0 * z0 + u1 * z1
        y -= v0 * z0 + v1 * z1

    if not s00 > 0.:
        return -np.inf, 0., 0., 0., 0., 0.
    l00 = sqrt(s00)
    inv0 = 1. / l00
    l10 = s10 * inv0
    s11 -= l10 * l10
    if not s11 > 0.:
        return -np.inf, 0., 0., 0., 0., 0.
    l11 = sqrt(s11)
    inv1 = 1. / l11
    z0 = x * inv0
    z1 = (y - l10 * z0) * inv1
    score = -.5 * (z0 * z0 + z1 * z1) - log(l00) - log(l11) - _LOG_2PI
    return score, inv0, l10, inv1, z0, z1


@njit
def joint_beam(base, residual, joint, role_bits, topk):
    """Return up to ``topk`` paths and their scores, highest first.

    ``base[M,G]`` contains log acoustic/detection scores; code zero is NULL.
    ``residual[M,G-1,2]`` omits NULL, while ``joint[2*G,2*G]`` includes
    its leading zero rows and columns. Non-NULL ``role_bits[G]`` entries are
    nonempty vessel masks, so intersecting masks cannot be assigned together.

    Every viable first-echo anchor (including NULL) gets an independent beam
    of width at most ``max(12, topk)``. Each extension adds the exact conditional log
    density given its preceding non-NULL assignments, using block Cholesky of
    the full selected covariance. Final ranking across anchors thus uses exact
    joint densities plus base terms, without the global all-missed baseline.
    The beam bounds the search, not the accuracy of scores of retained paths.
    Ties retain traversal order (ascending anchor code, then ranked parents
    and ascending child code).
    """
    m, g = base.shape
    if topk <= 0:
        return np.empty((0, m), dtype=np.intp), np.empty(0)
    if m == 0:
        return np.empty((1, 0), dtype=np.intp), np.zeros(1)

    width = 1 if m <= 2 else max(_MIN_WIDTH, topk)
    capacity = 1 + (m - 1) * width
    parents = np.empty(capacity, dtype=np.intp)
    codes = np.empty(capacity, dtype=np.intp)
    heads = np.empty(capacity, dtype=np.intp)
    used = np.empty(capacity, dtype=np.uint64)
    scores = np.empty(capacity)
    # Surviving nodes share all preceding Cholesky rows. Matching nodes own
    # just their two appended rows, and NULL nodes own no numerical state.
    max_prior = min(m - 1, g - 1)
    rows = np.empty((capacity, 2, 2 * min(max(m - 2, 0), max_prior)))
    blocks = np.empty((capacity, 5))  # reciprocal diagonal, subdiagonal, whitened residual
    chain = np.empty(max_prior, dtype=np.intp)
    cross = np.empty((2, 2 * max_prior))
    current = np.empty(width, dtype=np.intp)
    proposed = np.empty(width, dtype=np.intp)
    heads[0] = 0
    used[0] = 0
    scores[0] = 0.

    best = np.empty(topk, dtype=np.intp)
    paths = np.empty((topk, m), dtype=np.intp)
    final_scores = np.empty(topk)
    n_best = 0
    for anchor in range(g):
        current[0] = 0
        n_current = 1
        for echo in range(m):
            first = anchor if echo == 0 else 0
            stop = anchor + 1 if echo == 0 else g
            final = echo == m - 1
            n_proposed = 0
            start_slot = 1 + echo * width
            for rank in range(n_current):
                node = current[rank]
                head = heads[node]
                count = 0
                while head:
                    chain[count] = head
                    count += 1
                    head = heads[parents[head]]
                mask = used[node]
                for code in range(first, stop):
                    add = base[echo, code]
                    if not np.isfinite(add):
                        continue
                    bits = np.uint64(role_bits[code]) if code else np.uint64(0)
                    if mask & bits:
                        continue
                    inv0 = l10 = inv1 = z0 = z1 = 0.
                    if code:
                        ll, inv0, l10, inv1, z0, z1 = _extension(
                            joint, code, residual[echo, code - 1, 0], residual[echo, code - 1, 1],
                            chain, count, codes, rows, blocks, cross,
                        )
                        add += ll
                    score = scores[node] + add
                    if not np.isfinite(score):
                        continue
                    if final:
                        slot = _offer(best, final_scores, n_best, topk, 0, score)
                        if slot >= 0:
                            n_best = min(n_best + 1, topk)
                            paths[slot, echo] = code
                            prev = node
                            for k in range(echo - 1, -1, -1):
                                paths[slot, k] = codes[prev]
                                prev = parents[prev]
                    else:
                        slot = _offer(proposed, scores, n_proposed, width, start_slot, score)
                        if slot >= 0:
                            n_proposed = min(n_proposed + 1, width)
                            parents[slot] = node
                            codes[slot] = code
                            used[slot] = mask | bits
                            heads[slot] = slot if code else heads[node]
                            if code:
                                for k in range(2 * count):
                                    rows[slot, 0, k] = cross[0, k]
                                    rows[slot, 1, k] = cross[1, k]
                                blocks[slot, 0] = inv0
                                blocks[slot, 1] = l10
                                blocks[slot, 2] = inv1
                                blocks[slot, 3] = z0
                                blocks[slot, 4] = z1
            if final or not n_proposed:
                break
            current, proposed = proposed, current
            n_current = n_proposed

    order = best[:n_best]
    return paths[order], final_scores[order]

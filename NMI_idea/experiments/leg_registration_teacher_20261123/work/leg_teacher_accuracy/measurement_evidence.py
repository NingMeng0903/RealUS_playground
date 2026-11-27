"""Likelihood of anonymous echoes under the leg benchmark's observation model."""
import numpy as np
from numba import njit


@njit(cache=True)
def _gaussian_1d(value, mean, sigma):
    residual = (value - mean) / sigma
    return np.exp(-.5 * residual * residual) / (np.sqrt(2. * np.pi) * sigma)


@njit(cache=True)
def _gaussian_2d(echo, x, depth, sx, sz):
    return _gaussian_1d(echo[0], x, sx) * _gaussian_1d(echo[1], depth, sz)


@njit(cache=True)
def _is_arterial(role):
    # In this two-route atlas, role 1 is the named accompanying artery.
    return role == 1


@njit(cache=True)
def _type_mean(role):
    return 1. if _is_arterial(role) else -1.


@njit(cache=True)
def target_marginal(points, detection, visible, echoes, stress, blur, type_signal):
    """Marginalize detected-role subsets, at most one merge, and at most one ghost."""
    n = len(echoes)
    answer = np.zeros(n)
    if n == 0:
        return answer
    typed = len(type_signal) == n
    available = np.flatnonzero(visible)
    count = len(available)
    sx, sz = (2.3, 1.8) if stress else (1.4, 1.1)
    ghost_rate = .66 if stress else .31
    merge_rate = (.79 if stress else .44) * (1. - blur) + .99 * blur
    pure = np.zeros((len(points), n))
    ghost = np.empty(n)
    types_sd = .80
    for k in range(n):
        feature = type_signal[k] if typed else 0.
        background = 1. / (50. * 77.)
        density = (.28 if count else 1.) * background
        if typed:
            density *= .5 * (_gaussian_1d(feature, -1., types_sd) + _gaussian_1d(feature, 1., types_sd))
        for role in available:
            x, z = points[role]
            branch = _gaussian_2d(echoes[k], x + 2.7, z + 2., 1., 1.)
            role_like = _gaussian_1d(feature, _type_mean(role), types_sd) if typed else 1.
            density += .72 * branch * role_like / max(1, count)
            pure[role, k] = _gaussian_2d(echoes[k], x, z, sx, sz) * role_like
        ghost[k] = density

    sizes = np.zeros(1 << n, dtype=np.int64)
    for i in range(1, len(sizes)):
        sizes[i] = sizes[i >> 1] + (i & 1)

    full = (1 << n) - 1
    total = 0.
    observed = np.empty(count, dtype=np.int64)
    rows = np.empty((n, n))
    weights = np.zeros(1 << n)
    for mask in range(1 << count):
        p = 1.
        number = 0
        for j in range(count):
            role = available[j]
            if mask & (1 << j):
                observed[number] = role
                number += 1
                p *= detection[role]
            else:
                p *= 1. - detection[role]
        if number < n - 1 or number > n + 1 or p <= 1e-12:
            continue
        pair_a, pair_b = -1, -1
        shortest = np.inf
        for i in range(number - 1):
            for j in range(i + 1, number):
                a, b = observed[i], observed[j]
                dx = (points[a, 0] - points[b, 0]) / (4.7 + 10. * blur)
                dz = (points[a, 1] - points[b, 1]) / (4.7 + 7. * blur)
                distance = dx * dx + dz * dz
                if distance < shortest:
                    shortest = distance
                    pair_a, pair_b = a, b
        merge = merge_rate * np.exp(-.5 * shortest)
        for blend in range(2):
            ghost_count = n - (number - blend)
            if ghost_count < 0 or ghost_count > 1 or (blend and pair_a < 0):
                continue
            chance = p * (merge if blend else 1. - merge) * (ghost_rate if ghost_count else 1. - ghost_rate)
            if chance <= 1e-12:
                continue
            num_rows = 0
            has_target = False
            for j in range(number):
                role = observed[j]
                if blend and (role == pair_a or role == pair_b):
                    continue
                if role == 0:
                    has_target = True
                    continue
                rows[num_rows, :] = pure[role]
                num_rows += 1
            if blend:
                for k in range(n):
                    density = 0.
                    for fraction in (.37, .50, .63):
                        mean = (1. - fraction) * points[pair_a] + fraction * points[pair_b]
                        reduced = np.sqrt((1. - fraction)**2 + fraction**2)
                        density += _gaussian_2d(echoes[k], mean[0], mean[1], sx * reduced, sz * reduced) / 3.
                    if typed:
                        mean = .5 * (_type_mean(pair_a) + _type_mean(pair_b))
                        density *= _gaussian_1d(type_signal[k], mean, types_sd)
                    rows[num_rows, k] = density
                num_rows += 1
            if ghost_count:
                rows[num_rows, :] = ghost
                num_rows += 1
            weights[:] = 0.
            weights[0] = 1.
            for used in range(1, full + 1):
                row = sizes[used] - 1
                if row >= num_rows:
                    continue
                for k in range(n):
                    bit = 1 << k
                    if used & bit:
                        weights[used] += rows[row, k] * weights[used ^ bit]
            if has_target:
                for k in range(n):
                    mass = chance * pure[0, k] * weights[full ^ (1 << k)]
                    total += mass
                    answer[k] += mass
            else:
                total += chance * weights[full]
    if total > 1e-290:
        answer /= total
    return answer


def synthetic_type_signal(labels, rng):
    """Simulate a noisy artery/vein cue, without transmitting vessel identity."""
    labels = np.asarray(labels)
    mean = np.where(labels == 1, 1., -1.)
    ghost = labels == -1
    merged = labels == -2
    mean[ghost] = rng.choice([-1., 1.], int(ghost.sum()))
    mean[merged] = 0.
    return mean + .8 * rng.standard_normal(len(labels))

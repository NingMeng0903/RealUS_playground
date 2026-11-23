"""Shared public scanner constants and the atlas/CT observation model."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import lfilter

LATERAL, DEPTH_MIN, DEPTH_MAX = 25., 3., 80.
WINDOW_AREA = 2. * LATERAL * (DEPTH_MAX - DEPTH_MIN)
BASE_SPEED = 43.
ANGLES = np.arange(24) * 2. * np.pi / 24.
NORMALS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
TANGENTS = NORMALS[:, ::-1] * [-1., 1.]
TASKS = {
    "saphenofemoral": (-525, -590, (0, 1, 2, 3)),
    "femoral": (-595, -790, (0, 1, 2, 3)),
    "popliteal": (-835, -875, (0, 1, 2, 3)),
    "calf": (-980, -1255, (0, 1, 2, 3)),
    "groin_to_distal_calf": (-525, -1255, (0, 1, 2, 3, 4)),
}
PROFILES = ("ordinary", "stress")
BONE_PRIOR = Path("/media/camp/PEI_T7/SMPLX_Vascular_Probability_Cloud/right_leg_joint/bonehub_right_leg_occupancy.npz")


@lru_cache(maxsize=1)
def bone_prior():
    with np.load(BONE_PRIOR, allow_pickle=False) as pack:
        axes = tuple(pack[f"{axis}_mm"] for axis in "xyz")
        return RegularGridInterpolator(axes, pack["count"] / int(pack["cohort_size"]),
                                       bounds_error=False, fill_value=np.nan)


@njit(cache=True, nogil=True)
def _dropout(uniform, enter, stay, previous):
    result = np.empty(len(uniform), dtype=np.bool_)
    for index, value in enumerate(uniform):
        previous = value < (stay if previous else enter)
        result[index] = previous
    return result


@dataclass(frozen=True)
class Tape:
    normal: np.ndarray
    segmentation: np.ndarray
    dropout: np.ndarray
    false: np.ndarray

    @classmethod
    def make(cls, seed, profile, n, previous=None):
        rng = np.random.default_rng(seed)
        correlation, enter, stay = (.78, .05, .65) if profile == "stress" else (.40, .022, .38)
        state = np.zeros((6, 2)) if previous is None else previous[0] * correlation
        normal, _ = lfilter([np.sqrt(1. - correlation ** 2)], [1., -correlation],
                            rng.standard_normal((n, 6, 2)), axis=0, zi=state[None])
        dropout = _dropout(rng.random(n), enter, stay, False if previous is None else previous[1])
        return cls(normal, rng.random((n, 6)), dropout, rng.random((n, 12)))


@njit(cache=True, nogil=True)
def _render(truth, gate, normal, segmentation, dropout, q, stress, contact_quality, blur):
    # Coupling/quality comes from the probe and pulse tape alone.
    quality = min(1., max(.05, (.81 if stress else .92) + .23 * (q[10] - .5)
                          - .15 * (1. - contact_quality)))
    pts = np.empty((7, 2), dtype=np.float64)
    labels = np.empty(7, dtype=np.int64)
    if dropout:
        return pts[:0], labels[:0], .10 + .22 * q[11]
    plausible = np.empty(6, dtype=np.int64)
    candidates, seen = 0, 0
    for role in range(len(truth)):
        x, depth = truth[role]
        if not (-LATERAL <= x <= LATERAL and DEPTH_MIN <= depth <= DEPTH_MAX):
            continue
        plausible[candidates] = role
        candidates += 1
        rate = (.69 if stress else .91) * (.82 + .18 * contact_quality) * (1. - .96 * gate[role])
        if segmentation[role] < rate:
            pts[seen, 0] = x + (2.3 if stress else 1.4) * normal[role, 0]
            pts[seen, 1] = depth + (1.8 if stress else 1.1) * normal[role, 1]
            labels[seen] = role
            seen += 1

    if seen > 1:
        best, a, b = 1e8, 0, 1
        for i in range(seen - 1):
            for j in range(i + 1, seen):
                dx = truth[labels[i], 0] - truth[labels[j], 0]
                dz = truth[labels[i], 1] - truth[labels[j], 1]
                square = (dx / (4.7 + 10. * blur)) ** 2 + (dz / (4.7 + 7. * blur)) ** 2
                if square < best:
                    best, a, b = square, i, j
        merge_rate = (.79 if stress else .44) * (1. - blur) + .99 * blur
        if q[0] < merge_rate * np.exp(-.5 * best):
            f = .35 + .30 * q[1]
            pts[a] = (1. - f) * pts[a] + f * pts[b]
            labels[a] = -2
            for j in range(b + 1, seen):
                pts[j - 1] = pts[j]
                labels[j - 1] = labels[j]
            seen -= 1

    if q[2] < (.66 if stress else .31):
        if q[3] < .72 and candidates:
            role = plausible[min(candidates - 1, int(q[4] * candidates))]
            pts[seen, 0] = truth[role, 0] + 2.7 + 5. * (.65 * normal[role, 0] / 4. + .35 * (q[5] - .5))
            pts[seen, 1] = truth[role, 1] + 2. + 5. * (.65 * normal[role, 1] / 4. + .35 * (q[6] - .5))
        else:
            pts[seen, 0] = (2. * q[7] - 1.) * LATERAL
            pts[seen, 1] = DEPTH_MIN + q[8] * (DEPTH_MAX - DEPTH_MIN)
        labels[seen] = -1
        seen += 1

    # Insertion sort by lateral position. No hidden identity enters the ordering.
    count = 0
    for i in range(seen):
        x, depth = pts[i]
        if not (-LATERAL <= x <= LATERAL and DEPTH_MIN <= depth <= DEPTH_MAX):
            continue
        role = labels[i]
        j = count
        while j > 0 and pts[j - 1, 0] > x:
            pts[j] = pts[j - 1]
            labels[j] = labels[j - 1]
            j -= 1
        pts[j, 0], pts[j, 1], labels[j] = x, depth, role
        count += 1
    return pts[:count], labels[:count], quality


class Trial:
    """A scan, with distinct static geometry and time-indexed measurement seeds."""

    def __init__(self, source, seed, profile, *, world_seed=None, split="training", n=1800):
        self.source, self.profile, self.split = source, profile, split
        self.stress = profile == "stress"
        self.tape_seed = int(seed)
        self.tape = Tape.make(seed, profile, n)
        self.branch_tick = -1
        self.future = self.tape
        self.world_seed = world_seed
        rng = np.random.default_rng(world_seed if world_seed is not None else seed)
        self.bias = rng.uniform(-1., 1., (5, 2))
        self.phase = rng.uniform(0., 2. * np.pi, 6)
        self.level = rng.uniform(.60, 1.25)
        self.shadow_strength = rng.uniform(.26, .76) if self.stress else rng.uniform(.09, .38)
        self.jitter = 0. if world_seed is None or split == "blender" else 1.
        regime = .85 if split == "training" else 1.13
        self.magnitude = (9.0 if self.stress else 4.2) * self.jitter * regime * self.level
        self.station_warp_cache = {}
        self.layout = None
        self.window = None

    def branch(self, tick, seed):
        """Keep the acquired prefix and draw independent conditional innovations."""
        from copy import copy

        result = copy(self)
        result.branch_tick = int(tick)
        n = len(self.tape.normal) - tick - 1
        result.future = Tape.make(seed, self.profile, n,
                                  (self.tape.normal[tick], self.tape.dropout[tick]))
        return result

    def deform(self, surface, station, centers):
        if not self.jitter:
            return centers
        if station in self.station_warp_cache:
            origin, rotation, shift = self.station_warp_cache[station]
        else:
            q = station / 34.
            p = self.phase
            bias = self.bias
            shift = self.magnitude * (1.25 * bias[0] + .66 * bias[1] * np.sin(q + p[0])
                                      + .45 * bias[2] * np.cos(.47 * q + p[1]))
            origin = surface.cross_origin[int(station)]
            angle = .09 * self.jitter * self.level * np.sin(.52 * q + p[2])
            c, s = np.cos(angle), np.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            if len(self.station_warp_cache) < 1024:
                self.station_warp_cache[station] = origin, rotation, shift
        return (centers - origin) @ rotation + origin + shift

    def observe(self, truth, gate, tick, *, station=0., sector=0., contact_quality=1.):
        tape = self.tape if tick <= self.branch_tick else self.future
        index = tick if self.branch_tick < 0 or tick <= self.branch_tick else tick - self.branch_tick - 1
        if self.world_seed is not None:
            # A fixed synthetic anatomy uses time-dependent sensor artifacts
            # and correlated noise with no time-dependent geometry shift.
            jitter = self.jitter if self.layout is None else 0.
            truth, gate = _time_perturb(truth, gate, tick, station, sector, jitter,
                                        self.shadow_strength, self.phase, tape.normal[index, 0, 0])
        blur = 0.
        if self.window is not None:
            ds = (station - self.window.station) / 1.15
            da = ((sector - self.window.sector + 12.) % 24. - 12.) / .80
            blur = np.exp(-.5 * (ds * ds + da * da))
        return _render(truth, gate, tape.normal[index], tape.segmentation[index], tape.dropout[index],
                       tape.false[index], self.stress, contact_quality, blur)


@njit(cache=True, nogil=True)
def _time_perturb(truth, gate, tick, station, sector, jitter, strength, phase, innovation):
    x = jitter * (1.3 * np.sin(.018 * tick + phase[3]) + .55 * innovation)
    z = jitter * (1.05 * np.sin(.013 * tick + phase[4]))
    # A transient acoustic window affects structures by their beam coordinate.
    center = 23. * np.sin(station * .11 + phase[0])
    pulse = max(0., (.75 * np.sin(.044 * tick + phase[5]) + .23 * innovation - .11))
    aperture = .65 + .35 * np.cos(sector * .262 - phase[1]) ** 2
    next_truth = truth.copy()
    next_gate = gate.copy()
    for role in range(len(truth)):
        next_truth[role, 0] += x
        next_truth[role, 1] += z
        offset = (next_truth[role, 0] - center) / 8.
        shade = min(.88, strength * pulse * aperture * np.exp(-.5 * offset * offset))
        next_gate[role] += (1. - next_gate[role]) * shade
    return next_truth, next_gate


@njit(cache=True, nogil=True)
def association(centers, base, integral, echoes, quality, roles):
    """Exact one-to-one identity marginals including clutter and mixed lumens."""
    hypotheses, count, channels = len(centers), len(echoes), len(roles)
    likelihood = np.empty(hypotheses)
    target = np.empty((hypotheses, count))
    unknown = np.empty_like(target)
    mixed = np.empty_like(target)
    masks = 1 << count
    tail = np.empty((channels + 1, masks))
    prefix = np.empty((2, masks))
    ratios = np.empty((channels, count))
    missed, detection = np.empty(channels), np.empty(channels)
    blend = np.empty(count)
    all_sum = np.empty(count)
    for h in range(hypotheses):
        for c in range(channels):
            role = roles[c]
            detection[c] = base[h, role] * (.45 + .64 * quality)
            missed[c] = 1. - detection[c] * integral[h, role]
        if not count:
            product = 1.
            for c in range(channels):
                product *= missed[c]
            likelihood[h] = .34 + .66 * product
            continue
        factor = 1.
        for i in range(count):
            x, z = echoes[i]
            ghost = .62 / WINDOW_AREA
            mixture = 0.
            for a in range(channels - 1):
                u, v = centers[h, roles[a]]
                for b in range(a + 1, channels):
                    uu, vv = centers[h, roles[b]]
                    sep = ((u - uu) ** 2 + (v - vv) ** 2) / (6.5 * 6.5)
                    radius = ((x - .5 * (u + uu)) / 3.3) ** 2 + ((z - .5 * (v + vv)) / 2.8) ** 2
                    mixture += .61 * np.exp(-.5 * (sep + radius)) * detection[a] * detection[b] / (2. * np.pi * 3.3 * 2.8)
            ghost += mixture
            blend[i] = mixture / ghost
            factor *= ghost
            for c in range(channels):
                u, v = centers[h, roles[c]]
                square = ((x - u) / 2.35) ** 2 + ((z - v) / 1.85) ** 2
                ratios[c, i] = np.exp(-.5 * square) * detection[c] / (2. * np.pi * 2.35 * 1.85 * ghost)

        tail[channels, :] = 1.
        for c in range(channels - 1, -1, -1):
            for mask in range(masks):
                probability = missed[c] * tail[c + 1, mask]
                for i in range(count):
                    bit = 1 << i
                    if (mask & bit) == 0:
                        probability += ratios[c, i] * tail[c + 1, mask | bit]
                tail[c, mask] = probability

        mass = tail[0, 0]
        likelihood[h] = mass * factor
        norm = max(1.e-100, mass)
        prefix[0, :] = 0.
        prefix[0, 0] = 1.
        all_sum[:] = 0.
        for c in range(channels):
            current, following = c % 2, (c + 1) % 2
            for mask in range(masks):
                prefix[following, mask] = missed[c] * prefix[current, mask]
            for i in range(count):
                paired_mass = 0.
                bit = 1 << i
                for mask in range(masks):
                    if (mask & bit) == 0:
                        claimed = mask | bit
                        weight = ratios[c, i] * prefix[current, mask]
                        prefix[following, claimed] += weight
                        paired_mass += weight * tail[c + 1, claimed]
                value = paired_mass / norm
                all_sum[i] += value
                if c == 0:
                    target[h, i] = value
        for i in range(count):
            u = min(1., max(0., 1. - all_sum[i]))
            unknown[h, i] = u
            mixed[h, i] = u * blend[i]
    return likelihood, target, unknown, mixed

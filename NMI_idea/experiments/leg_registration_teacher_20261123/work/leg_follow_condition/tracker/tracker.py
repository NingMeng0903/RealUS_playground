"""Read named anatomical roles from public echoes and a connected atlas."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

ROLES = 6
MERGED = 6
CLUTTER = 7


@njit(cache=True)
def density(x, z, dx, dz, sx, sz):
    return np.exp(-.5 * ((x - dx)**2 / sx**2 + (z - dz)**2 / sz**2)) / (2. * np.pi * sx * sz)


@njit(cache=True)
def association(centers, gate, mean, learned, sigma, deviation, fraction, sector, echoes):
    """Marginalize one-to-one assignments, missed detections, a merge and clutter."""
    n = len(echoes)
    output = np.zeros((8, n))
    if not n:
        return output
    available = np.empty(ROLES, dtype=np.int64)
    detection = np.zeros(ROLES)
    pure = np.zeros((ROLES, n))
    ghosts = np.zeros((ROLES, n))
    count = 0
    base = (.91 - .22 * fraction) * (1. - .0216 * abs(np.sin(sector * np.pi / 12.)))
    for r in range(ROLES):
        x, z = centers[r]
        if not np.isfinite(x) or abs(x) > 30. or z < -2. or z > 85.:
            continue
        available[count] = r
        count += 1
        sx, sz = deviation[r]
        retention = (.5 + .5 * np.tanh((25. - abs(x)) / max(.6, learned[r, 0])))
        retention *= (.5 + .5 * np.tanh(min(z - 3., 80. - z) / max(.6, learned[r, 1])))
        # The center needs to be visible, and the measured echo must also lie in the aperture.
        mx, mz = x + mean[r, 0], z + mean[r, 1]
        retention *= (.5 + .5 * np.tanh((25. - abs(mx)) / (1.15 * sx)))
        retention *= (.5 + .5 * np.tanh(min(mz - 3., 80. - mz) / (1.15 * sz)))
        detection[r] = base * (1. - .96 * gate[r]) * retention
        for k in range(n):
            ex, ez = echoes[k]
            pure[r, k] = density(ex, ez, mx, mz, sx, sz)
            gx = x + 2.7 + .8125 * mean[r, 0] / sigma[0]
            gz = z + 2. + .8125 * mean[r, 1] / sigma[1]
            # Range noise of the acoustic replica is smaller than that of a lumen.
            gx_dev = np.sqrt(.82 + learned[r, 0]**2)
            gz_dev = np.sqrt(.82 + learned[r, 1]**2)
            ghosts[r, k] = density(ex, ez, gx, gz, gx_dev, gz_dev)

    ghost_rate = .31 + .35 * fraction
    fusion_rate = .44 + .35 * fraction
    full = (1 << n) - 1
    sizes = np.zeros(full + 1, dtype=np.int64)
    for i in range(1, full + 1):
        sizes[i] = sizes[i >> 1] + (i & 1)
    rows = np.empty((n, n))
    kinds = np.empty(n, dtype=np.int64)
    observed = np.empty(count, dtype=np.int64)
    forward = np.empty(full + 1)
    backward = np.empty(full + 1)
    ghost = np.full(n, (.28 if count else 1.) / 3850.)
    for j in range(count):
        ghost += .72 / count * ghosts[available[j]]
    blends = np.zeros((ROLES, ROLES, n))
    gaps = np.full((ROLES, ROLES), np.inf)
    for i in range(count - 1):
        a = available[i]
        for j in range(i + 1, count):
            b = available[j]
            gaps[a, b] = np.sum((centers[a] - centers[b])**2)
            # A blend is a mixture over admissible fractions, not a point midpoint.
            for k in range(n):
                for u in (.37, .50, .63):
                    v = 1. - u
                    x, z = (centers[a] + mean[a]) * v + (centers[b] + mean[b]) * u
                    dev = np.sqrt(v**2 * deviation[a]**2 + u**2 * deviation[b]**2)
                    blends[a, b, k] += density(echoes[k, 0], echoes[k, 1], x, z, dev[0], dev[1]) / 3.

    total = 0.
    for mask in range(1 << count):
        detections = 0
        prior = 1.
        for j in range(count):
            r = available[j]
            if mask & (1 << j):
                observed[detections] = r
                detections += 1
                prior *= detection[r]
            else:
                prior *= 1. - detection[r]
        if detections < n - 1 or detections > n + 1 or prior < 1.e-12:
            continue
        pair_a, pair_b, nearest = -1, -1, np.inf
        for i in range(detections - 1):
            a = observed[i]
            for j in range(i + 1, detections):
                b = observed[j]
                if gaps[a, b] < nearest:
                    nearest, pair_a, pair_b = gaps[a, b], a, b
        # A small heavy tail accommodates moderate unobserved changes in resolution.
        merge = fusion_rate * (.95 * np.exp(-.5 * nearest / 4.7**2) +
                               .05 * np.exp(-.5 * nearest / 9.**2))
        for blend in range(2):
            extra = n - detections + blend
            if not 0 <= extra <= 1 or (blend and pair_a < 0):
                continue
            chance = prior * (merge if blend else 1. - merge) * (ghost_rate if extra else 1. - ghost_rate)
            if chance <= 1.e-12:
                continue
            row = 0
            for j in range(detections):
                r = observed[j]
                if blend and (r == pair_a or r == pair_b):
                    continue
                rows[row] = pure[r]
                kinds[row] = r
                row += 1
            if blend:
                rows[row] = blends[pair_a, pair_b]
                kinds[row] = MERGED
                row += 1
            if extra:
                rows[row] = ghost
                kinds[row] = CLUTTER

            forward[:] = 0.
            forward[0] = 1.
            for used in range(1, full + 1):
                i = sizes[used] - 1
                for k in range(n):
                    bit = 1 << k
                    if used & bit:
                        forward[used] += rows[i, k] * forward[used ^ bit]
            mass = forward[full]
            if mass <= 1.e-290:
                continue
            backward[:] = 0.
            backward[full] = 1.
            for used in range(full - 1, -1, -1):
                i = sizes[used]
                for k in range(n):
                    bit = 1 << k
                    if not used & bit:
                        tail = rows[i, k] * backward[used | bit]
                        backward[used] += tail
                        output[kinds[i], k] += chance * forward[used] * tail
            total += chance * mass
    if total <= 1.e-290:
        output[CLUTTER] = 1.
        return output
    return output / total


@njit(cache=True)
def regress(stations, queries, ticks, query_ticks, components, query_components, scales,
            residual, weight, prior, regularization):
    """Fit separate slowly varying residuals along connected atlas edges."""
    shifts = np.zeros((len(queries), ROLES, 2))
    deviation = np.full((len(queries), ROLES, 2), prior)
    for i, station in enumerate(queries):
        for r in range(ROLES):
            component = query_components[i, r]
            if component < 0:
                continue
            bandwidth = scales[component]
            denom = regularization
            sx, sz = 0., 0.
            for j, other in enumerate(stations):
                distance = abs(station - other)
                if distance > 2.6 * bandwidth or components[j, r] != component:
                    continue
                # Hold out a short burst around the query. This prevents correlated
                # replicate noise from being reinterpreted as persistent geometry.
                if query_ticks[i] >= 0 and abs(query_ticks[i] - ticks[j]) <= 4:
                    continue
                factor = weight[j, r] * np.exp(-.5 * (distance / bandwidth)**2)
                denom += factor
                sx += factor * residual[j, r, 0]
                sz += factor * residual[j, r, 1]
            shifts[i, r, 0] = sx / denom
            shifts[i, r, 1] = sz / denom
            deviation[i, r] = np.sqrt(.45**2 + prior**2 / (1. + .9 * (denom - regularization)))
    return shifts, deviation


@njit(cache=True)
def evidence(posterior, echoes, counts, nominal, transform, ticks, stations, fraction):
    """Extract location evidence with soft assignments and limits on replicate weight."""
    residual = np.zeros((len(echoes), ROLES, 2))
    weight = np.zeros((len(echoes), ROLES))
    for i in range(len(echoes)):
        independence = 1. / (1.2 + 1.1 * fraction)
        for r in range(ROLES):
            if not np.isfinite(nominal[i, r, 0]):
                continue
            for k in range(counts[i]):
                p = posterior[i, r, k]
                w = p**1.65
                delta = echoes[i, k] - nominal[i, r]
                if np.sum(delta**2) > 12.**2:
                    continue
                residual[i, r] += w * (delta @ transform[i])
                weight[i, r] += w
            if weight[i, r] > 1.e-8:
                residual[i, r] /= weight[i, r]
                weight[i, r] *= independence
    return residual, weight


@njit(cache=True)
def deformation_prior(residual, weights, nominal, stations, ticks, components, fallback):
    """Estimate common deformation amplitude from reproducible offsets of isolated tracks."""
    cross = 0.
    evidence = 0.
    for i in range(len(stations)):
        for r in range(ROLES):
            if weights[i, r] < .14:
                continue
            nearby = False
            for alt in range(ROLES):
                if alt != r and np.sum((nominal[i, r] - nominal[i, alt])**2) < 7.5**2:
                    nearby = True
                    break
            if nearby:
                continue
            for j in range(max(0, i - 30), i):
                distance = abs(stations[i] - stations[j])
                if (weights[j, r] < .14 or ticks[i] - ticks[j] < 7 or distance > 4. or
                        components[i, r] != components[j, r]):
                    continue
                w = weights[i, r] * weights[j, r] * np.exp(-.1 * distance**2)
                cross += w * .5 * np.sum(residual[i, r] * residual[j, r])
                evidence += w
    # The empirical cross-covariance removes transient measurement variance.
    variance = (7. * (fallback * .75)**2 + 1.25 * cross) / (7. + evidence)
    return np.sqrt(min(4.**2, max(.55**2, variance)))


@njit(cache=True)
def temporal(centers, echoes, counts, posterior, ticks, components, fraction):
    """Use nearby pulses for transient correlated measurement errors."""
    n = len(centers)
    offset = np.zeros((n, ROLES, 2))
    reduction = np.zeros((n, ROLES))
    rho = .40 + .38 * fraction
    for i in range(n):
        for r in range(ROLES):
            if components[i, r] < 0:
                continue
            for direction in (-1, 1):
                j = i + direction
                while 0 <= j < n and abs(ticks[j] - ticks[i]) <= 10:
                    if components[j, r] != components[i, r]:
                        break
                    dt = abs(ticks[j] - ticks[i])
                    weight = 0.
                    for k in range(counts[j]):
                        p = posterior[j, r, k]
                        w = p**2
                        offset[i, r] += rho**dt * w * (echoes[j, k] - centers[j, r])
                        weight += w
                    reduction[i, r] += rho**(2 * dt) * weight
                    if weight > .4:
                        break
                    j += direction
    scale = np.maximum(1., 1. + reduction)[:, :, None]
    return np.clip(offset / scale, -2.8, 2.8), np.maximum(.55, 1. - .7 * reduction)


@njit(cache=True)
def infer(nominal, gate, shift, spatial_dev, transform, echoes, counts, sectors, fraction, offset, reduction):
    result = np.zeros((len(echoes), 8, echoes.shape[1]))
    sigma = np.array([1.4 + .9 * fraction, 1.1 + .7 * fraction])
    for i in range(len(echoes)):
        centers = nominal[i] + shift[i] @ transform[i].T
        dev = np.sqrt(sigma**2 * reduction[i, :, None] + spatial_dev[i]**2)
        result[i, :, :counts[i]] = association(centers, gate[i], offset[i], spatial_dev[i],
                                              sigma, dev, fraction, sectors[i], echoes[i, :counts[i]])
    return result


@dataclass
class Prepared:
    stations: np.ndarray
    sectors: np.ndarray
    ticks: np.ndarray
    echoes: np.ndarray
    counts: np.ndarray
    quality: np.ndarray
    nominal: np.ndarray
    transform: np.ndarray
    components: np.ndarray


@dataclass
class Readout:
    probabilities: list
    centers: np.ndarray | None
    uncertainty: np.ndarray | None


def noise_fraction(quality, sectors, counts):
    """Infer measurement spread from coupling and the public B-mode quality field."""
    nominal_loss = .018 * np.abs(np.sin(sectors * np.pi / 12.))
    good = (quality > .40) | (counts > 0)
    q = quality[good] + nominal_loss[good]
    if not len(q):
        return .5
    below = np.maximum(0., .805 - q)
    above = np.maximum(0., q - .925)
    log_odds = np.clip(120. * (np.sum(below) - np.sum(above)), -14., 14.)
    return float(1. / (1. + np.exp(-log_odds)))


class Reader:
    """Subject-specific registration through anonymous echo history and a public chart."""

    def __init__(self, env, source=2, *, adaptive=True):
        self.surface = env.scanner.surface
        self.atlas = self.surface.priors[source]
        self.links = self.surface.reference_links[source]
        self.adaptive = adaptive
        self.components = np.full(self.atlas.shape[:2], -1, dtype=int)
        scales = []
        for role in range(ROLES):
            k = 0
            while k < len(self.atlas):
                if not np.isfinite(self.atlas[k, role]).all():
                    k += 1
                    continue
                lo = k
                while k + 1 < len(self.atlas) and self.links[k, role] and np.isfinite(self.atlas[k + 1, role]).all():
                    k += 1
                self.components[lo:k + 1, role] = len(scales)
                scales.append(np.clip((k - lo + 1) / 16., 2., 6.))
                k += 1
        self.scales = np.array(scales)
        self.geometry = lru_cache(maxsize=8192)(self._geometry)

    def _geometry(self, s, a):
        lo = int(s)
        frac = s - lo
        if frac > 1.e-10:
            center = np.where(self.links[lo, :, None],
                              self.atlas[lo] * (1. - frac) + self.atlas[lo + 1] * frac, np.nan)
        else:
            center = self.atlas[lo]
        _, normal, tangent = self.surface.pose(s, a)
        transform = np.stack((tangent, -normal))
        points = self.surface.project(s, a, center)
        components = np.where(np.isfinite(points[:, 0]), self.components[lo], -1)
        return points, transform, components

    def prepare(self, frames):
        n = len(frames)
        station = np.empty(n)
        sector = np.empty(n)
        quality = np.empty(n)
        nominal = np.empty((n, ROLES, 2))
        transform = np.empty((n, 2, 2))
        components = np.empty((n, ROLES), dtype=int)
        counts = np.empty(n, dtype=int)
        ticks = np.empty(n, dtype=int)
        echoes = np.zeros((n, 7, 2))
        for i, frame in enumerate(frames):
            # Legacy frame slots for labels and hidden hypotheses are ignored.
            station[i], sector[i], points, quality[i], ticks[i] = frame[0], frame[1], frame[2], frame[4], frame[8]
            counts[i] = len(points)
            echoes[i, :counts[i]] = points
            nominal[i], transform[i], components[i] = self.geometry(station[i], sector[i])
        return Prepared(station, sector, ticks, echoes, counts, quality, nominal, transform, components)

    def gates(self, data, shifts):
        local = data.nominal + np.einsum("nrc,nkc->nrk", shifts, data.transform)
        safe = np.nan_to_num(local, nan=150.)
        u = np.clip(np.rint((safe[..., 0] + 25.) / 2.5), 0, 20).astype(int)
        z = np.clip(np.rint((safe[..., 1] - 4.) / 2.), 0, 38).astype(int)
        low = data.stations.astype(int)
        high = np.minimum(len(self.atlas) - 1, low + 1)
        left = (data.sectors % 24.).astype(int)
        right = (left + 1) % 24
        ds = (data.stations - low)[:, None]
        da = (data.sectors % 24. - left)[:, None]
        ray = self.surface.ray
        first = (1. - da) * ray[low[:, None], left[:, None], u, z] + da * ray[low[:, None], right[:, None], u, z]
        second = (1. - da) * ray[high[:, None], left[:, None], u, z] + da * ray[high[:, None], right[:, None], u, z]
        return (1. - ds) * first + ds * second

    def estimate(self, frames, *, anatomy=False):
        if not len(frames):
            return Readout([], self.atlas.copy() if anatomy else None, None)
        data = self.prepare(frames)
        fraction = noise_fraction(data.quality, data.sectors, data.counts)
        prior = 2.1 + .65 * fraction
        shift = np.zeros_like(data.nominal)
        spatial_dev = np.full_like(shift, prior)
        offset = np.zeros_like(shift)
        reduction = np.ones_like(shift[:, :, 0])

        def assign():
            return infer(data.nominal, self.gates(data, shift), shift, spatial_dev, data.transform,
                         data.echoes, data.counts, data.sectors, fraction, offset, reduction)

        def samples(posterior):
            return evidence(posterior, data.echoes, data.counts, data.nominal, data.transform,
                            data.ticks, data.stations, fraction)

        posterior = assign()
        regularization = .8
        for iteration in range(2):
            residual, weight = samples(posterior)
            if iteration == 0 and self.adaptive:
                prior = deformation_prior(residual, weight, data.nominal, data.stations,
                                          data.ticks, data.components, prior)
                regularization = .8 + (.75 + .3 * fraction)**2 / prior**2
            shift, spatial_dev = regress(data.stations, data.stations, data.ticks, data.ticks,
                                          data.components, data.components, self.scales,
                                          residual, weight, prior, regularization)
            posterior = assign()
        centers = data.nominal + np.einsum("nrc,nkc->nrk", shift, data.transform)
        offset, reduction = temporal(centers, data.echoes, data.counts, posterior,
                                     data.ticks, data.components, fraction)
        posterior = assign()
        predictions = [posterior[i, :, :count].copy() for i, count in enumerate(data.counts)]
        if not anatomy:
            return Readout(predictions, None, None)
        grid = np.arange(len(self.atlas), dtype=np.float64)
        residual, weight = samples(posterior)
        field, uncertainty = regress(data.stations, grid, data.ticks, np.full(len(grid), -1),
                                     data.components, self.components, self.scales,
                                     residual, weight, prior, regularization)
        return Readout(predictions, self.atlas + field, uncertainty)

    def read(self, frames):
        return self.estimate(frames).probabilities

    def registration(self, frames):
        return self.estimate(frames, anatomy=True).centers


@lru_cache(maxsize=8)
def _reader(env):
    return Reader(env)


def read(env, frames):
    """Return one normalized eight-class probability matrix per acquired frame."""
    return _reader(env).read(frames)

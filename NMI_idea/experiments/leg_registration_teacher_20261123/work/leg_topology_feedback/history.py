"""Anonymous echo association along the named atlas tracks."""
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import ndtr

from measurement_evidence import _gaussian_2d


N_ROLES = 6
MERGED = 6
GHOST = 7
N_ORIGINS = 8


@njit(cache=True)
def origin_probabilities(points, visible, detection, echoes, sigma, offset, deviation, stress, blur):
    """Marginalize one-to-one vessel, merged and ghost assignments in one frame."""
    n = len(echoes)
    result = np.zeros((N_ORIGINS, n))
    if n == 0:
        return result
    available = np.flatnonzero(visible)
    count = len(available)
    ghost_rate = .66 if stress else .31
    merge_rate = (.79 if stress else .44) * (1. - blur) + .99 * blur
    pure = np.zeros((N_ROLES, n))
    ghost = np.zeros(n)
    for k in range(n):
        ghost[k] = (.28 if count else 1.) / (50. * 77.)
        for role in available:
            x, z = points[role]
            dx, dz = offset[role]
            sx, sz = deviation[role]
            pure[role, k] = _gaussian_2d(echoes[k], x + dx, z + dz, sx, sz)
            ghost[k] += .72 * _gaussian_2d(
                echoes[k], x + 2.7 + .8125 * dx / sigma[0],
                z + 2. + .8125 * dz / sigma[1], 1., 1.,
            ) / count

    full = (1 << n) - 1
    sizes = np.zeros(full + 1, dtype=np.int64)
    for i in range(1, len(sizes)):
        sizes[i] = sizes[i >> 1] + (i & 1)
    observed = np.empty(count, dtype=np.int64)
    rows = np.empty((n, n))
    origins = np.empty(n, dtype=np.int64)
    forward = np.empty(full + 1)
    backward = np.empty(full + 1)
    joint = np.zeros((N_ORIGINS, n))
    total = 0.
    for mask in range(1 << count):
        prior = 1.
        detections = 0
        for j in range(count):
            role = available[j]
            if mask & (1 << j):
                observed[detections] = role
                detections += 1
                prior *= detection[role]
            else:
                prior *= 1. - detection[role]
        if detections < n - 1 or detections > n + 1 or prior <= 1e-12:
            continue
        pair_a, pair_b, nearest = -1, -1, np.inf
        for i in range(detections - 1):
            for j in range(i + 1, detections):
                a, b = observed[i], observed[j]
                dx = (points[a, 0] - points[b, 0]) / (4.7 + 10. * blur)
                dz = (points[a, 1] - points[b, 1]) / (4.7 + 7. * blur)
                distance = dx * dx + dz * dz
                if distance < nearest:
                    nearest, pair_a, pair_b = distance, a, b
        merge = merge_rate * np.exp(-.5 * nearest)
        for blend in range(2):
            extra = n - (detections - blend)
            if not 0 <= extra <= 1 or (blend and pair_a < 0):
                continue
            chance = prior * (merge if blend else 1. - merge) * (ghost_rate if extra else 1. - ghost_rate)
            if chance <= 1e-12:
                continue
            row = 0
            for j in range(detections):
                role = observed[j]
                if blend and (role == pair_a or role == pair_b):
                    continue
                origins[row] = role
                rows[row] = pure[role]
                row += 1
            if blend:
                origins[row] = MERGED
                for k in range(n):
                    density = 0.
                    for u in (.37, .50, .63):
                        v = 1. - u
                        center = v * (points[pair_a] + offset[pair_a]) + u * (points[pair_b] + offset[pair_b])
                        variance = v * v * deviation[pair_a] ** 2 + u * u * deviation[pair_b] ** 2
                        density += _gaussian_2d(echoes[k], center[0], center[1],
                                                np.sqrt(variance[0]), np.sqrt(variance[1])) / 3.
                    rows[row, k] = density
                row += 1
            if extra:
                origins[row] = GHOST
                rows[row] = ghost

            forward[:] = 0.
            forward[0] = 1.
            for used in range(1, full + 1):
                i = sizes[used] - 1
                for k in range(n):
                    bit = 1 << k
                    if used & bit:
                        forward[used] += rows[i, k] * forward[used ^ bit]
            if forward[full] <= 1e-290:
                continue
            backward[:] = 0.
            backward[full] = 1.
            for used in range(full - 1, -1, -1):
                i = sizes[used]
                for k in range(n):
                    bit = 1 << k
                    if not used & bit:
                        suffix = rows[i, k] * backward[used | bit]
                        backward[used] += suffix
                        joint[origins[i], k] += chance * forward[used] * suffix
            total += chance * forward[full]
    if total > 1e-290:
        result = joint / total
    else:
        result[GHOST] = 1.
    return result


@dataclass(frozen=True)
class Sample:
    tick: int
    station: float
    sector: float
    region: int
    echoes: np.ndarray
    points: np.ndarray
    gate: np.ndarray
    blur: float


class History:
    """Anatomical role alternatives with uncertain, correlated residuals."""

    def __init__(self, env, trial, forecast, *, correlation_scale=1., oracle_geometry=True):
        self.env = env
        self.trial = trial
        self.forecast = forecast
        self.oracle_geometry = oracle_geometry
        self.profile = trial.stress
        self.rho = (.78 if self.profile else .40) * correlation_scale
        self.sigma = np.array((2.3, 1.8) if self.profile else (1.4, 1.1))
        self.base_var = self.sigma**2
        self.links = (trial.layout.links if oracle_geometry
                      else env.scanner._anatomy.links[trial.source])
        self.samples = []
        self.predictions = []
        self.online = []
        self.mean, self.variance = self.reset()

    def reset(self):
        return np.zeros((N_ROLES, 2)), np.broadcast_to(self.base_var, (N_ROLES, 2)).copy()

    def extract(self, frame):
        # Identity in the frame archive is consumed only by the evaluator.
        s, a, echoes, _, _, r, _, _, tick = frame
        sc = self.env.scanner
        if self.oracle_geometry:
            points, gate = sc._anatomy.training_truth(sc.surface, self.trial, s, a)
        else:
            points, gate = sc._anatomy.truth(sc.surface, self.trial.source, s, a)
        return Sample(tick, s, a, r, echoes, points, gate, self.forecast.blur(s, a))

    def predict(self, current, previous, mean, variance):
        if previous is None:
            return self.reset()
        rho = self.rho**abs(current.tick - previous.tick)
        predicted_mean = rho * mean
        predicted_variance = rho**2 * variance + (1. - rho**2) * self.base_var
        lo, hi = sorted((current.station, previous.station))
        i, j = int(np.floor(lo + 1e-8)), int(np.ceil(hi - 1e-8))
        if j > i:
            connected = np.all(self.links[i:j], axis=0)
            predicted_mean[~connected] = 0.
            predicted_variance[~connected] = self.base_var
        return predicted_mean, predicted_variance

    def probability(self, frame, predicted_mean, predicted_variance):
        x, z = frame.points.T
        visible = (np.abs(x) <= 25.) & (z >= 3.) & (z <= 80.)
        points = np.nan_to_num(frame.points, nan=999.)
        deviation = np.sqrt(np.maximum(predicted_variance, .12 * self.base_var))
        coupling = 1. - .12 * abs(np.sin(frame.sector * np.pi / 12.))
        detection = (.69 if self.profile else .91) * (.82 + .18 * coupling) * (1. - .96 * frame.gate)
        filtered = points + predicted_mean
        high = ndtr((np.array([25., 80.]) - filtered) / deviation)
        low = ndtr((np.array([-25., 3.]) - filtered) / deviation)
        detection *= (high - low).prod(axis=-1)
        probabilities = origin_probabilities(points, visible, detection, frame.echoes, self.sigma,
                                              predicted_mean, deviation, self.profile, frame.blur)
        return probabilities

    def update(self, frame, mean, variance):
        prob = self.probability(frame, mean, variance)
        if not len(frame.echoes):
            return mean, variance, prob
        # Moment-matching across plausible identities includes the missed-detection component.
        missed = 1. - prob[:N_ROLES].sum(axis=1)
        residual = frame.echoes[None] - np.nan_to_num(frame.points, nan=999.)[:, None]
        innovation = .20 * self.base_var
        gain = variance / (variance + innovation)
        conditioned = mean[:, None] + gain[:, None] * (residual - mean[:, None])
        updated_mean = missed[:, None] * mean + np.sum(prob[:N_ROLES, :, None] * conditioned, axis=1)
        spread = (conditioned - updated_mean[:, None])**2 + ((1. - gain) * variance)[:, None]
        updated_var = missed[:, None] * (variance + (mean - updated_mean)**2)
        updated_var += np.sum(prob[:N_ROLES, :, None] * spread, axis=1)
        return updated_mean, np.maximum(updated_var, .12 * self.base_var), prob

    def append(self, frames):
        for frame in frames[len(self.samples):]:
            sample = self.extract(frame)
            prior = self.predict(sample, self.samples[-1] if self.samples else None, self.mean, self.variance)
            self.mean, self.variance, posterior = self.update(sample, *prior)
            self.samples.append(sample)
            self.predictions.append(prior)
            self.online.append(posterior)

    def smoothed(self):
        """Combine predictions from earlier and later acquired frames before reading each echo."""
        mean, variance = self.reset()
        out = [None] * len(self.samples)
        later = None
        for i in range(len(self.samples) - 1, -1, -1):
            sample = self.samples[i]
            mb, vb = self.predict(sample, later, mean, variance)
            mf, vf = self.predictions[i]
            iv = np.maximum(1. / vf + 1. / vb - 1. / self.base_var, 1. / (2. * self.base_var))
            va = 1. / iv
            ma = va * (mf / vf + mb / vb)
            out[i] = self.probability(sample, ma, va)
            mean, variance, _ = self.update(sample, mb, vb)
            later = sample
        return out

    def fork(self):
        """Copy only the mutable prefix state for an equal-information ablation."""
        from copy import copy
        clone = copy(self)
        clone.samples = list(self.samples)
        clone.predictions = list(self.predictions)
        clone.online = list(self.online)
        return clone


def select(samples, probabilities, threshold):
    proposals = [None] * 6
    for i, (sample, prob) in enumerate(zip(samples, probabilities)):
        if not prob.shape[1]:
            continue
        slot = int(prob[0].argmax())
        target = float(prob[0, slot])
        old = proposals[sample.region]
        if target >= threshold and (old is None or target > old[2]):
            proposals[sample.region] = (i, slot, target)
    return proposals

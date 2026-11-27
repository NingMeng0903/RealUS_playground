"""Public centerline registration with a displacement prior fitted from training layouts."""

from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FOLLOW = WORK / 'leg_follow_condition'
OLDER = WORK / 'leg_reader_teacher_20261123'
sys.path[:0] = [str(FOLLOW / 'learned_reader'), str(FOLLOW), str(OLDER / 'track')]
from tracker import Reader
from tracker.tracker import evidence, infer, noise_fraction, temporal
from pair_tracker import PairReader
from public_identity import FrozenReader

# Calibrate the gain of the registration update and preserve the existing readout as a candidate.
PROGRAMS = {
    'atlas': (0., 1.),
    'weak_update': (.50, 1.),
    'updated': (1., 1.),
    'broader_atlas': (0., 1.55),
    'broader_update': (.65, 1.55),
}


@njit(cache=True, nogil=True)
def shrinkage(stations, ticks, components, scales, offsets, weights, sigma, prior, strength):
    """Local conjugate update with a centered, anisotropic material-space prior."""
    new = np.zeros_like(offsets)
    deviation = np.zeros_like(offsets)
    for i in range(len(stations)):
        for role in range(offsets.shape[1]):
            center = components[i, role]
            prior_var = prior[i, role] ** 2
            if center < 0 or strength == 0.:
                deviation[i, role] = np.sqrt(prior_var)
                continue
            bandwidth = min(5., scales[center])
            mass = 0.
            numerator = np.zeros(2)
            for j in range(len(stations)):
                distance = abs(stations[i] - stations[j])
                # Leave a local burst out of the persistent geometry estimate.
                if distance > bandwidth * 2.4 or components[j, role] != center or abs(ticks[i] - ticks[j]) <= 5:
                    continue
                alpha = weights[j, role] * np.exp(-.5 * (distance / bandwidth) ** 2)
                mass += alpha
                numerator += alpha * offsets[j, role]
            for k in range(2):
                err = sigma[k] ** 2
                factor = strength * prior_var[k] / err
                new[i, role, k] = factor * numerator[k] / (1. + factor * mass)
                deviation[i, role, k] = np.sqrt(.25 ** 2 + prior_var[k] / (1. + factor * mass))
    return new, deviation


def public(frames):
    return FrozenReader.sanitize(frames)


def require_anonymous(frames):
    if any(frame[index] is not None for frame in frames for index in (3, 6, 7)):
        raise ValueError('Only anonymous acquired echoes and observed probe poses are accepted')


def phase_stations(stations):
    return np.clip((np.asarray(stations) // 12).astype(int), 0, 13)


@lru_cache(maxsize=10)
def stock(surface):
    return PairReader(surface)


@lru_cache(maxsize=10)
def chart(surface):
    return Reader(SimpleNamespace(scanner=SimpleNamespace(surface=surface)))


def posterior(surface, prepared, table, specification, *, anatomy=False):
    reg = chart(surface)
    fraction = noise_fraction(prepared.quality, prepared.sectors, prepared.counts)
    k = phase_stations(prepared.stations)
    base = (1. - fraction) * table[0, k] + fraction * table[1, k]
    gain, factor = specification
    prior = base * factor + .12
    shift = np.zeros_like(prepared.nominal)
    dev = prior
    local = np.zeros_like(shift)
    variance = np.ones_like(shift[:, :, 0])
    sigma = np.array([1.4 + .9 * fraction, 1.1 + .7 * fraction])
    gates = reg.gates(prepared, shift)

    def associate():
        return infer(prepared.nominal, gates, shift, dev, prepared.transform,
                     prepared.echoes, prepared.counts, prepared.sectors, fraction, local, variance)

    raw = associate()
    for _ in range(2 if gain else 0):
        offsets, strength = evidence(raw, prepared.echoes, prepared.counts, prepared.nominal,
                                     prepared.transform, prepared.ticks, prepared.stations, fraction)
        shift, dev = shrinkage(prepared.stations, prepared.ticks, prepared.components, reg.scales,
                               offsets, strength, sigma, prior, gain)
        gates = reg.gates(prepared, shift)
        raw = associate()
    estimated = prepared.nominal + np.einsum('nrc,nkc->nrk', shift, prepared.transform)
    local, variance = temporal(estimated, prepared.echoes, prepared.counts, raw, prepared.ticks,
                               prepared.components, fraction)
    weighted = associate()
    probabilities = np.moveaxis(weighted, 1, -1)
    return (probabilities, shift, dev) if anatomy else probabilities


def key_groups(prepared, groups):
    return [np.asarray([(i, j) for i in group for j in range(prepared.counts[i])],
                       dtype=int).reshape(-1, 2) for group in groups]


def table_from_file(path=None):
    obj = json.loads(Path(path or HERE/'reader_config.json').read_text())
    return obj, np.asarray(obj['prior_sd_mm'], dtype=float)


class RegisteredReader:
    """Drop-in anonymous reader, with a separately accessible geometric posterior."""

    threshold = .45

    def __init__(self, surface, config=None):
        self.surface = surface
        self.geometry = chart(surface)
        self.config, self.prior = table_from_file(config)
        choice = self.config['selected']
        self.spec = PROGRAMS[choice['program']]
        self.geometry_weight = float(choice['geometry_weight'])
        self.stress_weight = float(choice.get('stress_weight', self.geometry_weight))

    @staticmethod
    def sanitize(frames):
        return public(frames)

    def field(self, frames):
        require_anonymous(frames)
        prepared = self.geometry.prepare(frames)
        p, displacement, deviation = posterior(self.surface, prepared, self.prior, self.spec, anatomy=True)
        return prepared, p, displacement, deviation

    def all_probabilities(self, frames, groups):
        require_anonymous(frames)
        groups = [list(g) for g in groups]
        if not frames or not any(groups):
            return [(np.empty((0,2),dtype=int), np.empty((0,8))) for _ in groups]
        if any(i < 0 or i >= len(frames) for g in groups for i in g):
            raise ValueError('Candidate groups must lie in the available history')
        prepared = self.geometry.prepare(frames)
        ids = key_groups(prepared, groups)
        pp = posterior(self.surface, prepared, self.prior, self.spec)
        fraction = noise_fraction(prepared.quality, prepared.sectors, prepared.counts)
        fusion = self.geometry_weight * (1. - fraction) + self.stress_weight * fraction
        if fusion>=1.-1.e-15:
            return [(ix, pp[ix[:,0],ix[:,1]]) for ix in ids]
        blended = []
        for ix, (other, legacy) in zip(ids, stock(self.surface).paired_scores(frames, groups), strict=True):
            np.testing.assert_array_equal(ix, other)
            scores = pp[ix[:,0],ix[:,1]].copy()
            # PairReader exposes target, co-running artery, fused and all other.
            other_roles = [2,3,4,5,7]
            factor = legacy[:,3] / np.maximum(1.e-10, scores[:,other_roles].sum(axis=1))
            shares = scores[:,other_roles] * factor[:,None]
            shares[shares.sum(axis=1) < 1.e-10,-1] = legacy[shares.sum(axis=1) < 1.e-10,3]
            expanded = np.empty_like(scores)
            expanded[:,[0,1,6]] = legacy[:,:3]
            expanded[:,other_roles] = shares
            blended.append((ix,fusion*scores + (1.-fusion)*expanded))
        return blended

    def scores(self, frames, groups):
        return [(i,p[:,0]) for i,p in self.all_probabilities(frames,groups)]

    def best(self, frames, groups):
        answers = []
        for indices, probability in self.scores(frames, groups):
            if not len(probability):
                answers.append(None)
                continue
            winner = int(probability.argmax())
            i, j = indices[winner]
            answers.append((int(i), int(j), float(probability[winner])))
        return answers

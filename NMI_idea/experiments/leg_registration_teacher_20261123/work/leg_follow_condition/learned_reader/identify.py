"""Candidate-level identity features from anonymous B-mode history."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import History, scan
from tracker import Reader


@njit(cache=True)
def temporal_support(ids, stations, sectors, ticks, residuals, probabilities):
    """Repeated hits close in position and material residual provide independent support."""
    out = np.zeros((len(ids), 24))
    for row, i in enumerate(ids):
        accum = np.zeros((4, 5))
        for j in range(max(0, i - 220), min(len(stations), i + 221)):
            dt = abs(ticks[i] - ticks[j])
            if not 0 < dt <= 85:
                continue
            ds = abs(stations[i] - stations[j])
            if ds > 14.:
                continue
            da = abs((sectors[i] - sectors[j] + 12.) % 24. - 12.)
            group = 0 if ds < .6 and da < .7 else (1 if ds < 2.4 else (2 if ds < 6. else 3))
            dx = residuals[j, 0] - residuals[i, 0]
            dz = residuals[j, 1] - residuals[i, 1]
            agree = np.exp(-.5 * (dx*dx + dz*dz) / 2.5**2)
            factor = np.exp(-.065 * ds)
            mass = factor * probabilities[j, 0]
            accum[group, 0] += mass
            accum[group, 1] += mass * agree
            accum[group, 2] += mass * dx
            accum[group, 3] += mass * dz
            accum[group, 4] += factor * (1. - probabilities[j, 0]) * agree
            out[row, 6*group + 5] = max(out[row, 6*group + 5], mass * agree)
        for group in range(4):
            mass = max(.01, accum[group, 0])
            out[row, 6*group:6*group+5] = np.array([
                np.log1p(mass), accum[group, 1] / mass, accum[group, 2] / mass,
                accum[group, 3] / mass, np.log1p(accum[group, 4]),
            ])
    return out


class Atlas:
    """Expose only the nominal public chart to the baseline extractor."""
    def __init__(self, surface, registration):
        self.surface = surface
        self.registration = registration
        self.links = surface.reference_links

    def truth(self, surface, source, station, sector):
        points = self.registration.geometry(station, sector)[0]
        return points, surface.exact_population_shadow(station, sector, points)


class FeatureReader:
    def __init__(self, sensor):
        self.surface = sensor.surface
        self.reader = Reader(SimpleNamespace(scanner=SimpleNamespace(surface=self.surface)))
        atlas = Atlas(self.surface, self.reader)
        self.atlas_env = SimpleNamespace(scanner=SimpleNamespace(surface=self.surface, _anatomy=atlas))
        self.nominal = scan.Trial(2, 0, 'stress', n=1)
        self.forecast = SimpleNamespace(blur=lambda *_: 0.)

    @staticmethod
    def public(frames):
        """Only five physical fields and the public region annotation pass this boundary."""
        return [(f[0], f[1], f[2], None, f[4], f[5], None, None, f[8]) for f in frames]

    def feature_sets(self, frames, groups):
        """Provide one feature matrix per group, all from the supplied time prefix."""
        native = History(self.atlas_env, self.nominal, self.forecast, oracle_geometry=False)
        native.append(frames)
        old = native.smoothed()
        reg = self.reader.estimate(frames, anatomy=True)
        prepared = self.reader.prepare(frames)
        count = prepared.counts
        ends = np.r_[0, np.cumsum(count)]
        owner = np.repeat(np.arange(len(frames)), count)
        echo_id = np.arange(len(owner)) - ends[owner]
        if not len(owner):
            return [(np.empty((0, 0), dtype=np.float32), np.empty((0, 2), dtype=int)) for _ in groups]
        ids = np.column_stack((owner, echo_id))
        requested = [np.concatenate([np.arange(ends[i], ends[i + 1]) for i in group])
                     if len(group) else np.empty(0, dtype=int) for group in groups]
        used = np.unique(np.concatenate(requested))
        if not len(used):
            return [(np.empty((0, 0), dtype=np.float32), ids[:0]) for _ in groups]
        station = prepared.stations[owner]
        sectors = prepared.sectors[owner]
        ticks = prepared.ticks[owner]
        reg_p = np.concatenate(reg.probabilities, axis=1).T
        native_p = np.concatenate(old, axis=1).T[used]
        online_p = np.concatenate(native.online, axis=1).T[used]
        points = prepared.echoes[owner, echo_id]
        nominal = prepared.nominal[owner]
        deltas = np.nan_to_num(points[:, None, :] - nominal, nan=60.)
        world = np.einsum('ni,nik->nk', deltas[:, 0], prepared.transform[owner])
        support = temporal_support(used, station, sectors, ticks, world, reg_p)

        frame = owner[used]
        key = echo_id[used]
        fpos = prepared.stations[frame]
        lo = np.floor(fpos).astype(int)
        hi = np.ceil(fpos).astype(int)
        u = (fpos - lo)[:, None, None]
        registered = (1. - u) * reg.centers[lo] + u * reg.centers[hi]
        registered -= u * self.reader.atlas[hi] + (1. - u) * self.reader.atlas[lo]
        camera_offset = np.einsum('nrc,nkc->nrk', registered, prepared.transform[frame])
        shifted = np.nan_to_num(deltas[used] - camera_offset, nan=60.)
        registration_sd = (1. - u) * reg.uncertainty[lo] + u * reg.uncertainty[hi]

        gaps = np.nan_to_num(nominal[used, 1:] - nominal[used, :1], nan=60.)
        reg_gaps = shifted[:, 1:] - shifted[:, :1]
        dnorm = np.linalg.norm(deltas[used], axis=-1)
        snorm = np.linalg.norm(shifted, axis=-1)
        nearest = np.argsort(snorm[:, 1:], axis=1)[:, :2] + 1
        idx = np.arange(len(frame))
        pose_angle = prepared.sectors[frame] * np.pi / 12.
        coupling = prepared.quality[frame]
        shadow = np.array([native.samples[j].gate for j in frame])
        sibling = prepared.echoes[frame] - points[used, None, :]
        sibling_dist = np.linalg.norm(sibling, axis=-1)
        sibling_dist[idx, key] = np.inf
        sibling_dist[np.arange(7) >= count[frame, None]] = np.inf
        closest = sibling_dist.argmin(axis=1)
        sibling_delta = np.where(count[frame,None] > 1, sibling[idx,closest], 0.)
        close_sibling_dist = np.minimum(sibling_dist.min(axis=1), 20.)
        view = nominal[used, 0]
        radial = np.array([np.linalg.norm(self.surface.pose(s, a)[0] -
                                         self.surface.cross_origin[int(s)])
                           for s,a in zip(fpos, prepared.sectors[frame])])
        margin = np.minimum.reduce((25. - np.abs(view[:,0]), view[:,1]-3., 80.-view[:,1]))
        candidates = reg_p[used]
        probabilities = np.column_stack((candidates, native_p, online_p,
                                          candidates[:, 0] - candidates[:, 1:].max(axis=1),
                                          native_p[:, 0] - native_p[:, 1:].max(axis=1)))
        topology = np.column_stack((gaps.reshape(len(frame), -1), reg_gaps.reshape(len(frame), -1),
                                     snorm[idx[:, None], nearest],
                                     shifted[idx[:, None], nearest].reshape(len(frame), -1),
                                     registration_sd[:, 0], camera_offset[:, 0]))
        imaging = np.column_stack((coupling, fpos / 160., np.sin(pose_angle), np.cos(pose_angle),
                                   points[used], count[frame], key, view, radial, margin, shadow,
                                   sibling_delta, close_sibling_dist, points[used] - [2.7, 2.]))
        locations = np.column_stack((deltas[used].reshape(len(frame), -1), shifted.reshape(len(frame), -1),
                                    np.minimum(dnorm, 40.), np.minimum(snorm, 40.)))
        features = np.nan_to_num(np.column_stack((probabilities, support, topology, imaging, locations)),
                                 nan=0., posinf=60., neginf=-60.).astype(np.float32)
        return [(features[np.searchsorted(used, q)], ids[q]) for q in requested]


@lru_cache(maxsize=7)
def public_reader(env):
    return FeatureReader(env.scanner)


class CandidateModel:
    """Predict a target probability for every actual candidate in a public feature matrix."""
    def __init__(self, estimator, registered_weight=0.):
        self.estimator = estimator
        self.registered_weight = registered_weight

    def predict(self, matrix):
        learned = self.estimator.predict_proba(matrix)[:, 1]
        return (1. - self.registered_weight) * learned + self.registered_weight * matrix[:, 0]

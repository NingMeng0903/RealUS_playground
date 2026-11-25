"""Public candidate graph for the frozen v13 distal readout.

Rows of the archived embedding are used as they were exported by v13. Echo
identity and simulated instance IDs belong to the evaluator, not this module.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree


REGION = 5
STATION_FRACTION = 29
CROSS_XZ = slice(133, 135)
PROBE_SECTOR = slice(30, 32)
ROLE_PROBABILITY = slice(167, 175)
QUALITY = 176
WIDTH = 177
MAX_PAIRS = 4
POSITION_LIMIT = 1.15


@dataclass(frozen=True)
class RelationPrior:
    grid: np.ndarray
    origins: np.ndarray       # shape, station, xz
    centers: np.ndarray       # shape, source, station, role, xz
    links: np.ndarray         # source, station boundary, role
    junctions: tuple = ()     # explicitly sourced, typed junctions


@dataclass(frozen=True)
class Junction:
    station: int
    stem_role: int
    side_role: int
    kind: str                 # bifurcation or confluence along scan direction
    weight: float


def load_prior(pack, junctions=()):
    with np.load(pack, allow_pickle=False) as a:
        return RelationPrior(a['neutral_grid'], a['origins'], a['centers'],
                             a['links'], tuple(junctions))


def scan_bounds(prior):
    start = int(np.argmin(abs(prior.grid + 525.)))
    end = int(np.argmin(abs(prior.grid + 1255.)))
    if not np.array_equal(prior.grid[[start, end]], [-525., -1255.]):
        raise ValueError('Unexpected frozen full-route coordinates')
    return start, end


def addresses(x, shape, prior):
    """Undo the archived public chart normalization at the recorded station."""
    if x.ndim != 2 or x.shape[1] != WIDTH:
        raise ValueError('Expected the frozen v13 177-field schema')
    first, last = scan_bounds(prior)
    station = np.clip(np.rint(first + x[:, STATION_FRACTION] * (last - first)),
                      first, last).astype(int)
    point = x[:, CROSS_XZ] * 50. + prior.origins[shape, station]
    sector = np.rint(np.arctan2(x[:, PROBE_SECTOR.start], x[:, PROBE_SECTOR.stop-1])
                    * (12. / np.pi)).astype(int) % 24
    return station, point, sector


def _ordered_pairs(frame):
    """All distinct candidates in a measured frame, in both directions."""
    if np.any(np.diff(frame) < 0):
        raise ValueError('Expected candidates in observed frame order')
    _, counts = np.unique(frame, return_counts=True)
    index = np.arange(len(frame))
    first, second = [], []
    for lag in range(1, int(counts.max(initial=1))):
        left = index[:-lag]
        right = index[lag:]
        use = frame[left] == frame[right]
        first.extend((left[use], right[use]))
        second.extend((right[use], left[use]))
    if not first:
        return np.empty(0, int), np.empty(0, int)
    return np.concatenate(first), np.concatenate(second)


def cross_section_features(point, station, frame, role, quality, templates):
    """Compare relative echo constellations with plausible atlas role pairs.

    The wide radial term tolerates deformations. A smaller signed vector term
    retains orientation information. Neighbor roles are probability mixtures.
    A no-neighbor observation has an explicit support value of zero.
    """
    first, second = _ordered_pairs(frame)
    output = np.zeros((len(point), 6), dtype=np.float32)
    if not len(first):
        return output
    observed = point[second] - point[first]
    observed_length = np.linalg.norm(observed, axis=-1)
    nroles = min(templates.shape[-2], role.shape[1] - 2)
    center = templates[:, station[first], :nroles].transpose(1, 0, 2, 3)
    votes = np.zeros((len(first), 4))
    role_mass = role[second, :nroles]
    for hypothesis in range(4):
        expected = center - center[:, :, hypothesis:hypothesis+1, :]
        supported = np.isfinite(expected).all(axis=-1)
        supported[:, :, hypothesis] = False  # one object per simultaneous echo
        expected = np.nan_to_num(expected)
        length = np.linalg.norm(expected, axis=-1)
        scale = 4.5 + .33 * length
        radial = (observed_length[:, None, None] - length) / scale
        vector = np.linalg.norm(observed[:, None, None, :] - expected, axis=-1)
        # Cauchy rather than a hard distance gate. Rotation-sensitive evidence
        # weakens smoothly in highly bent sections.
        fit = supported / (1. + radial**2) * (.77 + .23/(1. + (vector/(scale+4.))**2))
        best_source = fit.max(axis=1)
        votes[:, hypothesis] = (best_source * role_mass).sum(axis=-1)
    weight = np.sqrt(quality[first] * quality[second])
    den = np.bincount(first, weights=weight, minlength=len(point))
    for j in range(4):
        numer = np.bincount(first, weights=weight*votes[:, j], minlength=len(point))
        output[:, j] = numer / np.maximum(den, 1.e-8)
    output[:, 4] = np.clip(den, 0., 2.) / 2.
    output[:, 5] = output[:, 0] - output[:, 1]
    return output


def _pool_tracks(track, station, sector, frame, point, quality, role, relative):
    """Best measured row per station and view. A burst of repeats votes once."""
    _, group = np.unique(track, return_inverse=True)
    order = np.lexsort((frame, -quality, sector, station, group))
    same = ((group[order[1:]] == group[order[:-1]]) &
            (station[order[1:]] == station[order[:-1]]) &
            (sector[order[1:]] == sector[order[:-1]]))
    keep = order[np.r_[True, ~same]]
    track_rows = np.split(keep, np.flatnonzero(np.diff(group[keep]))+1)
    # Representative by station for continuity, overlap and bifurcation tests.
    representative = []
    for rows in track_rows:
        ordered = rows[np.lexsort((-quality[rows], station[rows]))]
        representative.append(ordered[np.r_[True, np.diff(station[ordered]) != 0]])
    start = np.array([r[0] for r in representative])
    end = np.array([r[-1] for r in representative])
    ss, es = station[start], station[end]
    pooled = []
    for rows, samples in zip(track_rows, representative, strict=True):
        weight = np.maximum(quality[rows], .03)
        pooled.append(np.r_[np.average(relative[rows], axis=0, weights=weight),
                           np.log1p(len(samples))/4., np.log1p(len(rows))/5.,
                           min(20, np.ptp(station[samples]))/20.,
                           np.average(quality[rows]),
                           np.average(role[rows, 6:].sum(axis=-1), weights=weight)])
    return group, representative, start, end, ss, es, np.array(pooled, dtype=np.float32)


def _continuation(graph, station, point, region, role, quality, reference):
    """Directed continuation on track fragments, keeping a consistent source.

    A gap with no supported path contributes zero evidence. To avoid treating
    two simultaneously observed structures as one, parent and child occupy
    strictly ordered station intervals. Geometry is a compatibility weight.
    """
    _, rows, first, last, lo, hi, _ = graph
    nodes = len(rows)
    widths = np.maximum(1, hi-lo)
    drift = (point[last]-point[first]) / widths[:, None]
    scale = 2.8
    roots = np.column_stack((hi*scale, point[last]))
    tips = np.column_stack((lo*scale, point[first]))
    nearby = cKDTree(roots).query_ball_point(tips, 14.)
    gaps = np.cumsum(~reference[:, :, :2], axis=1, dtype=int)
    gaps = np.pad(gaps, ((0, 0), (1, 0), (0, 0)))
    spans = gaps[:, hi] == gaps[:, lo]
    path = np.zeros((len(reference), nodes, 2))
    reach = np.zeros(nodes)
    for node in np.argsort(lo, kind='stable'):
        data = rows[node]
        early = data[region[data] < REGION]
        if len(early):
            anchor = np.average(role[early, :2], axis=0, weights=np.maximum(quality[early], .03))
            anchor *= min(1., .35 + .23*len(early))
            path[:, node] = anchor
        prev = np.asarray(nearby[node], dtype=int)
        gap = lo[node] - hi[prev]
        allowed = (gap >= 1) & (gap <= 4)
        prev, gap = prev[allowed], gap[allowed]
        if len(prev):
            delta = point[first[node]] - point[last[prev]]
            length = np.linalg.norm(delta, axis=-1)
            after = np.linalg.norm(delta - drift[prev]*gap[:, None], axis=-1)
            # Use a bounded compatibility even if the apparent tangent twists.
            link = .80 / (1. + (length/(5.+gap))**2) + .20/(1.+(after/(5.+gap))**2)
            same_source = gaps[:, lo[node], None] == gaps[:, hi[prev]]
            arrival = link[None, :, None] * path[:, prev] * same_source
            path[:, node] = np.maximum(path[:, node], arrival.max(axis=1))
            reach[node] = link.max()
        path[:, node] *= spans[:, node]
    return np.column_stack((path.max(axis=0), reach)).astype(np.float32)


def junction_compatibility(measured_kind, expected_kind, strength):
    """A missing observation/edge is unknown. Only an observed typed mismatch
    can oppose the proposed relation. This function is shared with smoke tests.
    """
    if measured_kind is None or expected_kind is None:
        return 0.
    if measured_kind not in ('bifurcation', 'confluence') or expected_kind not in ('bifurcation', 'confluence'):
        raise ValueError('Unknown junction type')
    return float(np.clip(strength, 0., 1.)) * (1. if measured_kind == expected_kind else -1.)


def _pair_features(a, b, group, samples, station, point, frame, role, junctions):
    """A soft fork/join hypothesis from separated co-observations.

    These signals describe the trajectories of observed candidates. They are
    not proof that two vessel lumens share a node. A topology bonus uses only
    explicit named reference events, kept separate from the descriptive cues.
    """
    if b < 0:
        return np.zeros((2, 9), dtype=np.float32), np.zeros(2, dtype=np.float32)
    ar, br = samples[group[a]], samples[group[b]]
    shared, ia, ib = np.intersect1d(station[ar], station[br], return_indices=True)
    ra, rb = ar[ia], br[ib]
    distance = np.linalg.norm(point[ra]-point[rb], axis=-1)
    coobserved = len(np.intersect1d(frame[group == group[a]], frame[group == group[b]]))
    delta = station[b] - station[a]
    vals = np.zeros(5)
    rise = fall = 0.
    if len(distance):
        vals = [min(len(distance), 10)/10., np.median(distance)/20.,
                np.min(distance)/15., min(coobserved, 5)/5., min(np.ptp(shared), 12)/12.]
    if len(distance) >= 3 and coobserved:
        span = min(1., float(np.ptp(shared))/4.)
        end = min(2, len(distance)//2)
        near, far = np.mean(distance[:end]), np.mean(distance[-end:])
        rise = np.clip((far-near)/8., 0., 1.) * np.exp(-.5*(near/6.)**2) * span
        fall = np.clip((near-far)/8., 0., 1.) * np.exp(-.5*(far/6.)**2) * span
    side = min(abs(delta), 8)/8.
    rel = np.asarray([np.r_[vals, rise, fall, side, np.sign(delta)],
                      np.r_[vals, rise, fall, side, -np.sign(delta)]], dtype=np.float32)
    effect = np.zeros(2, dtype=np.float32)
    if not len(shared) or max(rise, fall) <= .2:
        return rel, effect
    observed = 'bifurcation' if rise >= fall else 'confluence'
    s = shared[0] if observed == 'bifurcation' else shared[-1]
    for row, (stem, other) in enumerate(((ar, br), (br, ar))):
        # Require a measured stem on the opposite side of the inferred split
        # and both candidates on the side that supports two distinct returns.
        stem_extends = (station[stem].min() < s < station[other].max()
                        if observed == 'bifurcation' else
                        station[stem].max() > s > station[other].min())
        if not stem_extends:
            continue
        for event in junctions:
            if abs(event.station-s) <= 2 and event.stem_role == 0:
                if event.side_role >= role.shape[1]-2:
                    continue
                val = max(rise, fall)*float(np.mean(role[other, event.side_role]))*event.weight
                comp = junction_compatibility(observed, event.kind, val)
                if abs(comp) > abs(effect[row]):
                    effect[row] = comp
    return rel, effect


def panel(snapshot, frozen_scores, shape, prior, *, points=None):
    """A and up to four B proposals, each with an A/B/unknown interpretation.

    Uses no scoring labels. The first candidate reproduces the frozen v13
    ranking. Alternatives include high unary and relation-supported tracks.
    """
    x = np.asarray(snapshot['x'], dtype=np.float32)
    region, frame, track = (snapshot[k] for k in ('region', 'frame', 'track'))
    candidate = np.flatnonzero(region == REGION)
    if not len(candidate):
        return None
    station, point, sector = addresses(x, shape, prior)
    if points is not None:
        point = points
    role, quality = x[:, ROLE_PROBABILITY], np.clip(x[:, QUALITY], .01, 1.)
    template = prior.centers[shape]
    relative = cross_section_features(point, station, frame, role, quality, template)
    graph = _pool_tracks(track, station, sector, frame, point, quality, role, relative)
    group, samples, *_, pooled = graph
    path = _continuation(graph, station, point, region, role, quality, prior.links)
    features = np.column_stack((relative, pooled[group], path[group]))
    a = int(candidate[np.argmax(frozen_scores[candidate])])
    # Retrieve competing observed tracks without using the evaluator labels.
    order = candidate[np.argsort(-frozen_scores[candidate], kind='stable')]
    first = np.unique(group[order], return_index=True)[1]
    best = order[np.sort(first)]
    other = best[group[best] != group[a]]
    bs = list(other[:MAX_PAIRS-1])
    left = other[~np.isin(other, bs)]
    if len(left):
        strength = features[left, 5] + .5*(path[group[left], 0] - path[group[left], 1])
        retrieve = .65*np.tanh(frozen_scores[left]/4.) + strength
        bs.append(int(left[np.argmax(retrieve)]))
    if not bs:
        bs = [-1]
    pair_rows = np.array([(a, b) for b in bs], dtype=int)
    states, effects = [], []
    for a, b in pair_rows:
        rel, effect = _pair_features(a, b, group, samples, station, point, frame, role, prior.junctions)
        state = np.stack((features[a], features[b] if b >= 0 else np.zeros(features.shape[1])))
        states.append(np.column_stack((state, rel)))
        effects.append(effect)
    valid = pair_rows >= 0
    # Only this small signed term uses the completely frozen embedding score.
    pos = POSITION_LIMIT*np.tanh(frozen_scores[pair_rows.clip(min=0)]/3.) * valid
    return {'pairs': pair_rows, 'state': np.asarray(states, np.float32),
            'position': pos.astype(np.float32), 'motif': np.asarray(effects, np.float32), 'valid': valid,
            'candidate_count': len(candidate)}


def target_set(panel, labels):
    """Supervision accepts both proposals when tracks fragment the same vein."""
    hit = (labels[panel['pairs'].clip(min=0)] == 0) & panel['valid']
    return np.column_stack((hit, ~hit.any(axis=1)))

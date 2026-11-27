"""Causal acoustic tracks and distinct instance hypotheses in the material chart."""

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import expit, softmax
from scipy.spatial import cKDTree


REGIONS = (3, 4, 5)
EDGE_DIM = 25


def edges(p, prediction):
    """Retrieve plausible returns of one object. Identity scores stay outside matching."""
    n = len(p["frames"])
    if n < 2:
        return np.empty((0, 2), dtype=int), np.empty((0, EDGE_DIM), dtype=np.float32)
    stations, sectors, points, frames = (p[k] for k in ("stations", "sectors", "points", "frames"))
    chart = np.column_stack((2.0 * stations, points))
    _, near = cKDTree(chart).query(chart, k=min(16, n), distance_upper_bound=13.)
    first, second = np.where(near < n)
    second = near[first, second]
    use = first != second
    first, second = first[use], second[use]
    pairs = np.unique(np.sort(np.column_stack((first, second)), axis=1), axis=0)
    first, second = pairs.T
    ds = np.abs(stations[first] - stations[second])
    xy = np.abs(points[first] - points[second])
    dist = np.linalg.norm(xy, axis=1)
    use = (ds <= 3.1) & (dist <= 10.5 + ds)
    first, second, ds, xy, dist = first[use], second[use], ds[use], xy[use], dist[use]
    angles = np.abs((sectors[first] - sectors[second] + 12.) % 24. - 12.)
    e = prediction["embedding"][:, :8]
    q = p["quality"]
    f = p["features"].astype(np.float32)
    signals = np.column_stack((ds/3., xy/12., dist/12., angles/12., np.minimum(q[first], q[second]),
                              frames[first] == frames[second],
                              f[first, 24] - f[second, 24], f[first, 28] - f[second, 28],
                              e[first] * e[second], np.abs(e[first] - e[second])))
    return np.column_stack((first, second)), signals.astype(np.float32)


def linear(model, x):
    z = ((x - model["mean"]) / model["scale"]).clip(-6., 6.)
    for w, b in zip(model["weight"][:-1], model["bias"][:-1], strict=True):
        z = z @ w.T + b
        z = z * expit(z)
    return z @ model["weight"][-1].T + model["bias"][-1]


@njit(cache=True, nogil=True)
def _track_union(pairs, order, frames):
    # A track may explain at most one echo in a frame. Mark both components
    # before merging so two old candidate objects cannot silently collapse.
    n = len(frames)
    parent = np.arange(n)
    sizes = np.ones(n, dtype=np.int64)
    tail = np.arange(n)
    following = np.full(n, -1, dtype=np.int64)
    seen = np.full(frames.max() + 1, -1, dtype=np.int64)
    for edge in order:
        a, b = pairs[edge]
        i, j = a, b
        while parent[i] != i:
            i = parent[i]
        while parent[j] != j:
            j = parent[j]
        if i == j:
            continue
        if sizes[j] > sizes[i]:
            i, j = j, i
        row = i
        while row >= 0:
            seen[frames[row]] = edge
            row = following[row]
        row = j
        while row >= 0 and seen[frames[row]] != edge:
            row = following[row]
        if row >= 0:
            continue
        parent[j] = i
        sizes[i] += sizes[j]
        following[tail[i]] = j
        tail[i] = tail[j]
    for i in range(n):
        j = i
        while parent[j] != j:
            j = parent[j]
        parent[i] = j
    return parent


def tracks(public, prediction, edge_model):
    frames = public["frames"]
    pairs, x = edges(public, prediction)
    if not len(x):
        return np.arange(len(frames))
    p = expit(linear(edge_model, x)[:, 0]) if edge_model else np.exp(-x[:, 3]**2/.12)
    threshold = edge_model["threshold"] if edge_model else .69
    keep = np.flatnonzero((x[:, 6] == 0) & (p >= threshold))
    order = keep[np.argsort(-p[keep], kind="stable")]
    return _track_union(pairs, order, frames)


def features(public, base, surface, edge_model, *, material=True):
    ids = tracks(public, base, edge_model)
    n = len(ids)
    if not n:
        raise ValueError("A feature snapshot requires measured echoes")
    station = np.clip(np.rint(public["stations"]).astype(int), 0, len(surface.y) - 1)
    angle = np.rint(public["sectors"]).astype(int) % 24
    point, frame_quality = public["points"], public["quality"]
    xyz = surface.priors[:, station, :5]
    residual = np.linalg.norm(xyz - point[None, :, None], axis=-1)
    residual = np.where(np.isfinite(residual), residual, 65.).transpose(1, 0, 2) / 32.
    offset = (point - surface.cross_origin[station]) / 50.
    raw = public["features"].astype(np.float32)
    roles = base["role_posterior"]
    local = np.column_stack((roles, base["target"], base["null"], base["embedding"], raw,
                             residual.reshape(n, -1), offset, public["regions"]/5.))
    if not material:
        raw_first = local.shape[1] - 114
        local[:, raw_first+24:raw_first+54] = 0.
        local[:, -18:-1] = 0.
    view_id = station * 24 + angle
    quality = frame_quality * np.clip(1. - roles[:, 6:].sum(axis=1), .06, 1.)
    pooled = np.zeros((n, 31), dtype=np.float32)
    for group in np.unique(ids):
        rows = np.flatnonzero(ids == group)
        ranked = rows[np.argsort(-quality[rows], kind="stable")]
        _, first = np.unique(view_id[ranked], return_index=True)
        take = ranked[first]
        weight = np.maximum(.03, quality[take])
        totals = np.average(roles[take], axis=0, weights=weight)
        central = np.average(residual[take].reshape(len(take), -1), axis=0, weights=weight)
        extent = np.ptp(station[take])
        pooled[rows] = np.r_[totals, central if material else np.zeros_like(central), np.log1p(len(take))/4.,
                             min(extent, 25)/25., len(np.unique(angle[take]))/24.,
                             np.max(roles[take, :3], axis=0), np.median(frame_quality[take]),
                             np.average(roles[take, 6:].sum(axis=1), weights=weight)]
    return np.column_stack((local, pooled)).astype(np.float32), ids


def predict(public, base, surface, edge_model, readout, blend):
    x, ids = features(public, base, surface, edge_model, material=not getattr(surface, "disable_material", False))
    logits = linear(readout, x)
    post = softmax(logits[:, :8] / readout["temperature"], axis=1)
    reliable = expit(logits[:, 8] / readout["rank_temperature"])
    return from_logits(public, base, ids, post, reliable, blend)


def from_logits(public, base, ids, post, reliable, blend):
    role = blend * post + (1. - blend) * base["role_posterior"]
    score, winner = base["score"].copy(), base["winner_score"].copy()
    # The local readout learns on two late regions and the distal section.
    # Preserve the already reliable anatomic lead-in from the frozen reader.
    late = np.isin(public["regions"], REGIONS)
    target = base["target"].copy()
    target[late] = (1. - blend) * target[late] + blend * (role[late, 0] * .65 + reliable[late] * .35)
    for region in REGIONS:
        rows = np.flatnonzero(public["regions"] == region)
        if not len(rows) or blend == 0.:
            continue
        quality = public["quality"][rows]
        basepoint = .6 * base["target"][rows] + .4 * base["role_posterior"][rows, 0]
        basepoint += .10 * np.maximum(0., base["winner_score"][rows])
        value = (1. - blend) * basepoint + blend * (.66 * reliable[rows] + .27 * post[rows, 0] + .07 * quality)
        candidate = rows[np.argmax(value)]
        score[rows], winner[rows] = -1., -1.
        confidence = float(value.max())
        winner[candidate] = confidence
        score[candidate] = confidence if confidence >= .30 else -1.
    return base | {"target": target, "score": score, "winner_score": winner, "role_posterior": role,
                   "track_id": ids, "observed_identity": post, "candidate_reliability": reliable}


@dataclass(frozen=True)
class Request:
    region: int
    indices: tuple
    keys: tuple
    support: tuple
    station: int
    sector: int
    risk: float
    state: np.ndarray


def request(public, p, region):
    rows = np.flatnonzero(public["regions"] == region)
    if not len(rows):
        return None
    w = p["winner_score"][rows]
    a = rows[np.argmax(w if w.max() >= 0 else p["target"][rows])]
    s, sectors = public["stations"], public["sectors"]
    ids = p["track_id"]
    x = public["points"]
    # Co-observed returns establish distinctness. Alternatively require
    # separated locations at an overlapping longitudinal section.
    same_frame = public["frames"][rows] == public["frames"][a]
    distance = np.linalg.norm(x[rows] - x[a], axis=1)
    rival = rows[(ids[rows] != ids[a]) & (np.abs(s[rows] - s[a]) <= 2.6) &
                 (same_frame | ((distance >= 6.) & (distance <= 22.)))]
    b = rival[np.argmax(.76*p["target"][rival] + .12*public["quality"][rival]
                        + .12*(public["frames"][rival] == public["frames"][a]))] if len(rival) else -1
    keys = tuple((int(public["frames"][k]), int(public["slots"][k])) if k >= 0 else None for k in (a, b))
    support = tuple(np.flatnonzero(ids == ids[k]) if k >= 0 else np.empty(0, dtype=int) for k in (a, b))
    features_b = p["role_posterior"][b] if b >= 0 else np.zeros(8)
    risk = float(1. - p["target"][a] + .40 * p["role_posterior"][a, 1] + .22*(p["score"][a] < 0.))
    state = np.r_[p["role_posterior"][a], features_b, p["target"][a], p["target"][b] if b >= 0 else 0.,
                  np.log1p(len(support[0]))/4., np.log1p(len(support[1]))/4., p["score"][a],
                  float(b >= 0), public["quality"][a], risk]
    return Request(region, (int(a), int(b)), keys, support, int(np.rint(s[a])), int(np.rint(sectors[a])) % 24,
                   risk, np.asarray(state, dtype=np.float32))

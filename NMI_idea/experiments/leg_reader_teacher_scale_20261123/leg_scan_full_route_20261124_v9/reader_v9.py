"""End-to-end candidate ranking with learned soft association over a public prefix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

PUBLIC_FIELDS = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")
FEATURES = 96
REGIONS = 6
EMBEDDING = 12
NEIGHBORS = 12
EDGE_FIELDS = 5
CANDIDATE_FIELDS = FEATURES + REGIONS + 2
ROUTES = ("femoral", "groin_to_distal_calf")
FORMAT = "end_to_end_public_candidate_reader_v9"


def prepare(public):
    """Build the same bounded, label-free spatial neighborhood for fit and inference."""
    # Acquired feature archives use fp16. Quantize live features the same way.
    raw = np.asarray(public["features"], dtype=np.float16).astype(np.float32)
    frames = np.asarray(public["frames"], dtype=np.int32)
    regions = np.asarray(public["regions"], dtype=np.int32)
    stations = np.asarray(public["stations"], dtype=np.float32)
    sectors = np.asarray(public["sectors"], dtype=np.float32)
    quality = np.asarray(public["quality"], dtype=np.float32)
    points = np.asarray(public["points"], dtype=np.float32)
    n = len(frames)
    if raw.shape != (n, FEATURES) or points.shape != (n, 2):
        raise ValueError("Expected candidate-aligned public features and 2-D echo points")
    if np.any((regions < 0) | (regions >= REGIONS)):
        raise ValueError("Candidate region outside the six route regions")
    x = np.column_stack((raw, np.eye(REGIONS, dtype=np.float32)[regions], quality, stations / 160.))
    neighbors = np.zeros((n, NEIGHBORS), dtype=np.int32)
    valid = np.zeros((n, NEIGHBORS), dtype=bool)
    geometry = np.zeros((n, NEIGHBORS, EDGE_FIELDS), dtype=np.float32)
    if n < 2:
        return x, neighbors, valid, geometry, regions
    coordinates = np.column_stack((stations / 1.8, points / 7.0))
    distance, proposed = cKDTree(coordinates).query(coordinates, k=min(n, 32), distance_upper_bound=3.3)
    available = np.isfinite(distance)
    proposed = np.minimum(proposed, n - 1)
    station_gap = np.abs(stations[:, None] - stations[proposed])
    azimuth_gap = np.abs((sectors[:, None] - sectors[proposed] + 12.) % 24. - 12.)
    point_gap = np.linalg.norm(points[:, None] - points[proposed], axis=-1)
    allowed = (available & (frames[:, None] != frames[proposed])
               & (station_gap <= 3.2) & (point_gap <= 16.)
               & ((azimuth_gap <= 1.5) | (station_gap <= 2.05)))
    # Prefer a mixture of views when many samples share almost the same pose.
    preference = np.where(allowed, distance + .18 * (azimuth_gap < .30), np.inf)
    order = np.argsort(preference, axis=1, kind="stable")[:, :NEIGHBORS]
    row = np.arange(n)[:, None]
    count = order.shape[1]
    neighbor = proposed[row, order]
    neighbors[:, :count] = neighbor
    valid[:, :count] = allowed[row, order]
    geometry[:, :count] = np.stack((
        station_gap[row, order] / 3.2,
        azimuth_gap[row, order] / 12.,
        point_gap[row, order] / 16.,
        np.abs(quality[:, None] - quality[neighbor]),
        quality[neighbor],
    ), axis=-1)
    return x, neighbors, valid, geometry, regions


def load_bundle(path):
    source = Path(path)
    if source.is_dir():
        source = source / "reader_bundle.json"
    bundle = json.loads(source.read_text())
    if bundle["format"] != FORMAT:
        raise ValueError("Unsupported candidate reader bundle")
    bundle["weights"] = {key: np.asarray(value, dtype=np.float32) for key, value in bundle["weights"].items()}
    for key in ("center", "scale"):
        bundle[key] = np.asarray(bundle[key], dtype=np.float32)
    return bundle


def _linear(x, weight, name):
    return x @ weight[name + ".weight"].T + weight[name + ".bias"]


def _silu(x):
    return x * expit(x)


def _probability_inputs(x, neighbors, valid, geometry, regions, bundle, use_history):
    weights = bundle["weights"]
    x = np.clip((x - bundle["center"]) / bundle["scale"], -9., 9.)
    hidden = _silu(_linear(_silu(_linear(x, weights, "encode1")), weights, "encode2"))
    embedding = _linear(hidden, weights, "embedding")
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
    other = embedding[neighbors]
    similarity = np.einsum("nkf,nf->nk", other, embedding)
    edge_input = np.concatenate((geometry, similarity[:, :, None], np.abs(embedding[:, None] - other)), axis=2)
    edge = _linear(_silu(_linear(edge_input, weights, "edge1")), weights, "edge2")[:, :, 0]
    attention = expit(edge) * valid * (.25 + .75 * geometry[:, :, -1])
    if not use_history:
        attention.fill(0.)
    support = attention.sum(axis=1, keepdims=True)
    value = _linear(hidden, weights, "value")
    summary = np.einsum("nk,nkf->nf", attention, value[neighbors]) / (1. + support)
    fused = hidden + _silu(_linear(np.concatenate((hidden, summary, np.log1p(support)), axis=1), weights, "fuse"))
    target = _linear(fused, weights, "target")[:, 0] + _linear(x, weights, "raw_skip")[:, 0]
    rank = target + _linear(fused, weights, "rank")[:, 0]
    role = softmax(_linear(fused, weights, "role"), axis=1).astype(np.float32)
    count = np.bincount(regions, minlength=REGIONS).astype(np.float32)[:, None]
    pool = np.zeros((REGIONS, fused.shape[1]), dtype=np.float32)
    np.add.at(pool, regions, fused)
    pool /= np.maximum(count, 1.)
    pooled = np.concatenate((pool, np.log1p(count) / 5.), axis=1)
    null = _linear(_silu(_linear(pooled, weights, "null1")), weights, "null2")[:, 0]
    return target, rank, null, embedding, role, edge, support[:, 0]


def infer_prepared(prepared, bundle, use_history=True):
    """Bare network forward, also used by calibration and numerical parity checks."""
    with threadpool_limits(limits=1, user_api="blas"):
        return _probability_inputs(*prepared, bundle, use_history)


def predict(public, bundle, route=None, *, threshold=None, use_history=True):
    """Return dense role/candidate outputs and at most one submitted view per region."""
    n = len(public["frames"])
    if not n:
        empty = np.empty(0, dtype=np.float32)
        return {"target": empty, "score": empty, "winner_score": empty, "null": empty,
                "embedding": np.empty((0, EMBEDDING), dtype=np.float32),
                "role": np.empty((0, 3), dtype=np.float32), "support": empty}
    target, rank, null, embedding, role, _, support = infer_prepared(prepare(public), bundle, use_history)
    regions = np.asarray(public["regions"], dtype=np.int32)
    # The per-region set rank chooses the view. Its margin over the learned
    # null option is the reliability of that view for threshold calibration.
    reliability = expit(rank - null[regions])
    winner_score = np.full(n, -1., dtype=np.float32)
    for region in range(REGIONS):
        indices = np.flatnonzero(regions == region)
        if len(indices):
            best = indices[np.argmax(rank[indices])]
            winner_score[best] = reliability[best]
    if threshold is None:
        settings = bundle["thresholds"]
        threshold = settings.get(route, bundle["fallback_threshold"])
    return {"target": expit(target), "score": np.where(winner_score >= threshold, winner_score, -1.),
            "winner_score": winner_score, "embedding": embedding, "role": role,
            "null": 1. - reliability, "support": support}

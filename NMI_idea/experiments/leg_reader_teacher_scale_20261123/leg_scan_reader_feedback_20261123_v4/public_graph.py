"""Public-only causal association graph for the v4 reader.

The graph consumes anonymous candidate geometry and a reader's soft class
probabilities.  It never reads evaluator labels, private anatomy, or a latent
layout.  Candidates are processed in frame order.  An edge may only connect a
new frame to an earlier frame, and a greedy matching pass prevents two
candidates from one frame entering the same track.  This keeps the returned
state causal and avoids treating a repeated static view as independent
evidence.

``associate`` returns revised probabilities for every input candidate.  No
candidate is removed when an edge is rejected.  The graph is intentionally a
small public-state module so it can be used from CPU workers without importing
torch.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np


CLASS_COUNT = 8
EMBEDDING_DIM = 8
DEFAULT_STATION_RADIUS = 3.2
DEFAULT_DISTANCE_MM = 16.0
DEFAULT_EDGE_THRESHOLD = 0.16
DEFAULT_STATIC_STATION = 0.55
DEFAULT_STATIC_ANGLE = 0.75
DEFAULT_TARGET_GATE = 0.35


def _array(public: Mapping[str, Any], name: str, *, dtype, ndim: int | None = None) -> np.ndarray:
    if name not in public:
        raise KeyError(f"public candidate mapping is missing {name!r}")
    value = np.asarray(public[name], dtype=dtype)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"public[{name!r}] must have {ndim} dimensions, got {value.shape}")
    return value


def _validate_inputs(public: Mapping[str, Any], probabilities: Any, embedding: Any):
    frames = _array(public, "frames", dtype=np.int64, ndim=1)
    slots = _array(public, "slots", dtype=np.int64, ndim=1)
    regions = _array(public, "regions", dtype=np.int64, ndim=1)
    stations = _array(public, "stations", dtype=np.float64, ndim=1)
    sectors = _array(public, "sectors", dtype=np.float64, ndim=1)
    points = _array(public, "points", dtype=np.float64, ndim=2)
    quality = _array(public, "quality", dtype=np.float64, ndim=1)
    p = np.asarray(probabilities, dtype=np.float64)
    z = np.asarray(embedding, dtype=np.float64)
    count = len(frames)
    if points.shape != (count, 2):
        raise ValueError(f"public['points'] must have shape ({count}, 2), got {points.shape}")
    for name, value in (("slots", slots), ("regions", regions), ("stations", stations),
                        ("sectors", sectors), ("quality", quality)):
        if len(value) != count:
            raise ValueError(f"public[{name!r}] has length {len(value)}, expected {count}")
    if p.shape != (count, CLASS_COUNT):
        raise ValueError(f"probabilities must have shape ({count}, {CLASS_COUNT}), got {p.shape}")
    if z.shape != (count, EMBEDDING_DIM):
        raise ValueError(f"embedding must have shape ({count}, {EMBEDDING_DIM}), got {z.shape}")
    if not all(np.isfinite(value).all() for value in (stations, sectors, points, quality, p, z)):
        raise ValueError("public graph inputs must be finite")
    if np.any(frames < 0) or np.any(slots < 0):
        raise ValueError("frame and slot indices must be nonnegative")
    if np.any(quality < 0.) or np.any(quality > 1.):
        quality = np.clip(quality, 0., 1.)
    # Accept logits or probabilities while keeping the public contract a soft
    # probability state.  A row that already sums to one is left unchanged.
    row_sum = p.sum(axis=1)
    if np.any(p < -1.e-8) or np.any(row_sum <= 1.e-12):
        p = _softmax(p)
    else:
        p = p / row_sum[:, None]
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norm, 1.e-12)
    order = np.lexsort((np.arange(count), slots, frames))
    return (frames, slots, regions, stations, sectors, points, quality, p, z, order)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80., 80.))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.e-12)


def _topology_links(reference_links: Any) -> np.ndarray | None:
    if reference_links is None:
        return None
    links = np.asarray(reference_links, dtype=bool)
    if links.ndim == 2:
        links = links[None]
    if links.ndim != 3 or links.shape[2] < 1 or links.shape[1] < 1:
        raise ValueError("reference_links must have shape [source, station_edge, role]")
    return links


def _target_topology_supported(links: np.ndarray | None, first: float, last: float) -> bool:
    """Return whether every crossed target edge is publicly connected.

    The route scanner uses station indices.  If a caller provides physical
    coordinates outside the link index range, the graph cannot assert a gap
    and leaves the edge admissible.  Within the declared range, each crossed
    edge needs support from at least one public source for role zero.
    """
    if links is None:
        return True
    lo = int(np.floor(min(first, last)))
    hi = int(np.ceil(max(first, last))) - 1
    if lo < 0 or hi >= links.shape[1] or hi < lo:
        return True
    supported_by_source = np.any(links[:, lo:hi + 1, 0], axis=0)
    return bool(np.all(supported_by_source))


def _view_key(station: float, sector: float) -> tuple[int, int]:
    return int(np.rint(station)), int(np.rint(sector)) % 24


def _angle_delta(first: float, second: float) -> float:
    return float(abs((first - second + 12.) % 24. - 12.))


def _candidate_edge(
    left: int,
    right: int,
    stations: np.ndarray,
    sectors: np.ndarray,
    points: np.ndarray,
    quality: np.ndarray,
    probabilities: np.ndarray,
    embedding: np.ndarray,
    links: np.ndarray | None,
    *,
    station_radius: float,
    distance_radius: float,
    edge_threshold: float,
    target_gate: float,
) -> tuple[float, bool, bool]:
    """Return ``(weight, admissible, topology_rejected)`` for one pair."""
    ds = abs(float(stations[right] - stations[left]))
    da = _angle_delta(float(sectors[right]), float(sectors[left]))
    # A neighboring axial view or a nearby cross-window view is admissible.
    if not ((.60 <= ds <= station_radius) or (ds <= 2.05 and da >= DEFAULT_STATIC_ANGLE)):
        return 0., False, False
    distance = float(np.linalg.norm(points[right] - points[left]))
    if not np.isfinite(distance) or distance > distance_radius:
        return 0., False, False
    left_view = _view_key(stations[left], sectors[left])
    right_view = _view_key(stations[right], sectors[right])
    if left_view == right_view:
        return 0., False, False
    target_mass = min(float(probabilities[left, 0]), float(probabilities[right, 0]))
    if target_mass >= target_gate and not _target_topology_supported(
            links, float(stations[left]), float(stations[right])):
        return 0., False, True
    cosine = float(np.clip(np.dot(embedding[left], embedding[right]), -1., 1.))
    embedding_affinity = .15 + .85 * (.5 + .5 * cosine)
    spatial = np.exp(-.5 * (distance / 6.5) ** 2
                     - .5 * (ds / 1.9) ** 2
                     - .5 * (da / 3.2) ** 2)
    confidence = (.60 + .40 * min(float(quality[left]), float(quality[right])))
    confidence *= .55 + .45 * np.sqrt(max(target_mass, 0.))
    probability_overlap = float(np.dot(probabilities[left], probabilities[right]))
    probability_overlap = .50 + .50 * np.clip(probability_overlap, 0., 1.)
    weight = float(np.clip(spatial * embedding_affinity * confidence * probability_overlap, 0., 1.))
    return weight, bool(weight >= edge_threshold), False


def _vector_candidate_edges(
    candidates: np.ndarray,
    right: int,
    frames: np.ndarray,
    stations: np.ndarray,
    sectors: np.ndarray,
    points: np.ndarray,
    quality: np.ndarray,
    probabilities: np.ndarray,
    embedding: np.ndarray,
    topology_support: np.ndarray | None,
    topology_bad_prefix: np.ndarray | None,
    *,
    station_radius: float,
    distance_radius: float,
    edge_threshold: float,
    target_gate: float,
    max_edges_per_candidate: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Score one right-hand candidate against its local bin in one NumPy pass.

    The scalar ``_candidate_edge`` remains the reference implementation and is
    useful for source-level checks.  Long routes contain thousands of the same
    small pair calculations, so the deployed path batches those calculations
    while retaining the scalar path's predicates and deterministic
    ``(-weight, left_index)`` tie order.
    """
    if len(candidates) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0)
    candidates = np.asarray(candidates, dtype=np.int64)
    # The bin only contains candidates from previous input positions.  Frames
    # can still be tied because a frame contributes several anonymous slots.
    candidates = candidates[frames[candidates] < frames[right]]
    if len(candidates) == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0)

    ds = np.abs(stations[candidates] - stations[right])
    da = np.abs((sectors[candidates] - sectors[right] + 12.) % 24. - 12.)
    geometry = ((ds >= .60) & (ds <= station_radius)) | (
        (ds <= 2.05) & (da >= DEFAULT_STATIC_ANGLE))
    if not np.any(geometry):
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0)

    delta = points[candidates] - points[right]
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    geometry &= np.isfinite(distance) & (distance <= distance_radius)
    left_station_key = np.rint(stations[candidates]).astype(np.int64)
    right_station_key = int(np.rint(stations[right]))
    left_sector_key = np.rint(sectors[candidates]).astype(np.int64) % 24
    right_sector_key = int(np.rint(sectors[right])) % 24
    geometry &= ~((left_station_key == right_station_key)
                  & (left_sector_key == right_sector_key))
    if not np.any(geometry):
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0)

    target_mass = np.minimum(probabilities[candidates, 0], probabilities[right, 0])
    topology_bad = np.zeros(len(candidates), dtype=bool)
    if topology_support is not None and topology_bad_prefix is not None:
        lo = np.floor(np.minimum(stations[candidates], stations[right])).astype(np.int64)
        hi = np.ceil(np.maximum(stations[candidates], stations[right])).astype(np.int64) - 1
        in_range = ((lo >= 0) & (hi < len(topology_support)) & (hi >= lo))
        supported = np.ones(len(candidates), dtype=bool)
        selected = np.flatnonzero(in_range)
        if len(selected):
            unsupported = (topology_bad_prefix[hi[selected] + 1]
                           - topology_bad_prefix[lo[selected]]) != 0
            supported[selected] = ~unsupported
        topology_bad = geometry & (target_mass >= target_gate) & ~supported
        geometry &= ~topology_bad
    topology_rejected = int(np.count_nonzero(topology_bad))
    if not np.any(geometry):
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), topology_rejected)

    cosine = np.clip(np.einsum("ij,j->i", embedding[candidates], embedding[right]), -1., 1.)
    embedding_affinity = .15 + .85 * (.5 + .5 * cosine)
    spatial = np.exp(-.5 * (distance / 6.5) ** 2
                     - .5 * (ds / 1.9) ** 2
                     - .5 * (da / 3.2) ** 2)
    confidence = (.60 + .40 * np.minimum(quality[candidates], quality[right]))
    confidence *= .55 + .45 * np.sqrt(np.maximum(target_mass, 0.))
    probability_overlap = .50 + .50 * np.clip(
        np.einsum("ij,j->i", probabilities[candidates], probabilities[right]), 0., 1.)
    weights = np.clip(spatial * embedding_affinity * confidence * probability_overlap, 0., 1.)
    geometry &= weights >= edge_threshold
    selected = np.flatnonzero(geometry)
    if not len(selected):
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), topology_rejected)
    # np.lexsort's final key is primary, matching local.sort(key=(-weight,left)).
    left = candidates[selected]
    weight = weights[selected]
    local_order = np.lexsort((left, -weight))[:max_edges_per_candidate]
    return left[local_order], weight[local_order], topology_rejected


def _vector_edge_batch(
    left: np.ndarray,
    right: np.ndarray,
    stations: np.ndarray,
    sectors: np.ndarray,
    points: np.ndarray,
    quality: np.ndarray,
    probabilities: np.ndarray,
    embedding: np.ndarray,
    topology_support: np.ndarray | None,
    topology_bad_prefix: np.ndarray | None,
    *,
    station_radius: float,
    distance_radius: float,
    edge_threshold: float,
    target_gate: float,
    max_edges_per_candidate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Score and locally cap all proposed pairs in one contiguous batch."""
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if not len(left):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), 0
    ds = np.abs(stations[right] - stations[left])
    da = np.abs((sectors[right] - sectors[left] + 12.) % 24. - 12.)
    geometry = ((ds >= .60) & (ds <= station_radius)) | (
        (ds <= 2.05) & (da >= DEFAULT_STATIC_ANGLE))
    delta = points[right] - points[left]
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    geometry &= np.isfinite(distance) & (distance <= distance_radius)
    left_station_key = np.rint(stations[left]).astype(np.int64)
    right_station_key = np.rint(stations[right]).astype(np.int64)
    left_sector_key = np.rint(sectors[left]).astype(np.int64) % 24
    right_sector_key = np.rint(sectors[right]).astype(np.int64) % 24
    geometry &= ~((left_station_key == right_station_key)
                  & (left_sector_key == right_sector_key))

    target_mass = np.minimum(probabilities[left, 0], probabilities[right, 0])
    topology_bad = np.zeros(len(left), dtype=bool)
    if topology_support is not None and topology_bad_prefix is not None:
        lo = np.floor(np.minimum(stations[left], stations[right])).astype(np.int64)
        hi = np.ceil(np.maximum(stations[left], stations[right])).astype(np.int64) - 1
        in_range = ((lo >= 0) & (hi < len(topology_support)) & (hi >= lo))
        supported = np.ones(len(left), dtype=bool)
        selected = np.flatnonzero(in_range)
        if len(selected):
            unsupported = (topology_bad_prefix[hi[selected] + 1]
                           - topology_bad_prefix[lo[selected]]) != 0
            supported[selected] = ~unsupported
        topology_bad = geometry & (target_mass >= target_gate) & ~supported
        geometry &= ~topology_bad
    topology_rejected = int(np.count_nonzero(topology_bad))
    if not np.any(geometry):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), topology_rejected

    cosine = np.clip(np.einsum("ij,ij->i", embedding[left], embedding[right]), -1., 1.)
    embedding_affinity = .15 + .85 * (.5 + .5 * cosine)
    spatial = np.exp(-.5 * (distance / 6.5) ** 2
                     - .5 * (ds / 1.9) ** 2
                     - .5 * (da / 3.2) ** 2)
    confidence = (.60 + .40 * np.minimum(quality[left], quality[right]))
    confidence *= .55 + .45 * np.sqrt(np.maximum(target_mass, 0.))
    probability_overlap = .50 + .50 * np.clip(
        np.einsum("ij,ij->i", probabilities[left], probabilities[right]), 0., 1.)
    weights = np.clip(spatial * embedding_affinity * confidence * probability_overlap, 0., 1.)
    geometry &= weights >= edge_threshold
    selected = np.flatnonzero(geometry)
    if not len(selected):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), topology_rejected

    # First key is the right candidate, then descending weight, then the left
    # index.  Capping each right group here is exactly local.sort(...)[ :K ]
    # from the scalar implementation, while avoiding N small NumPy calls.
    ordered = np.lexsort((left[selected], -weights[selected], right[selected]))
    left = left[selected][ordered]
    right = right[selected][ordered]
    weights = weights[selected][ordered]
    begins = np.r_[True, right[1:] != right[:-1]]
    begin_indices = np.flatnonzero(begins)
    ends = np.r_[begin_indices[1:], len(right)]
    rank = np.arange(len(right), dtype=np.int64) - np.repeat(begin_indices, ends - begin_indices)
    keep = rank < max_edges_per_candidate
    return left[keep], right[keep], weights[keep], topology_rejected


class _UnionFind:
    def __init__(self, count: int, frames: np.ndarray):
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)
        self.frame_sets = [{int(frames[i])} for i in range(count)]

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union_if_disjoint_frames(self, first: int, second: int) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right or self.frame_sets[left].intersection(self.frame_sets[right]):
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        self.frame_sets[left].update(self.frame_sets[right])
        self.frame_sets[right] = set()
        return True


def associate(
    public: Mapping[str, Any],
    probabilities: Any,
    embedding: Any,
    reference_links: Any = None,
    *,
    enabled: bool = True,
    strength: float = .65,
    station_radius: float = DEFAULT_STATION_RADIUS,
    distance_radius: float = DEFAULT_DISTANCE_MM,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
    target_gate: float = DEFAULT_TARGET_GATE,
    max_edges_per_candidate: int = 8,
) -> dict[str, Any]:
    """Associate anonymous candidates and revise their soft class state.

    ``public`` must contain ``frames``, ``slots``, ``regions``, ``stations``,
    ``sectors``, ``points`` and ``quality``.  ``reference_links`` is an
    optional public topology array with shape ``[source, edge, role]``.  The
    returned arrays preserve input order.  The ``edges`` array has columns
    ``(left_index, right_index, weight)`` and always points from an earlier
    frame to a later frame.
    """
    if not 0. <= float(strength) <= 1.:
        raise ValueError("strength must be in [0, 1]")
    if station_radius < .60 or distance_radius <= 0. or edge_threshold < 0. or max_edges_per_candidate < 1:
        raise ValueError("invalid public graph limits")
    frames, slots, regions, stations, sectors, points, quality, base, z, order = _validate_inputs(
        public, probabilities, embedding)
    count = len(frames)
    links = _topology_links(reference_links)
    if not enabled or count == 0 or strength == 0.:
        tracks = np.arange(count, dtype=np.int32)
        return {
            "probabilities": base.astype(np.float32), "track_ids": tracks,
            "support": np.zeros(count, dtype=np.float32), "edges": np.empty((0, 3), dtype=np.float32),
            "stats": {"enabled": bool(enabled), "candidate_count": int(count), "proposed_edges": 0,
                      "kept_edges": 0, "topology_rejected": 0, "matching_conflicts": 0,
                      "track_count": int(count), "mean_edge_weight": 0., "max_support": 0.},
        }

    # Work in chronological order and index previous candidates by station.
    # The local bins keep this O(N * local_candidates) on long scans instead
    # of constructing an all-pairs matrix.  Pair scoring itself is batched in
    # NumPy below; the scalar helper remains available as a reference.
    bins: dict[int, list[int]] = defaultdict(list)
    proposed: list[tuple[float, int, int]] = []
    topology_rejected = 0
    if links is None:
        topology_support = topology_bad_prefix = None
    else:
        topology_support = np.any(links[:, :, 0], axis=0)
        # Prefix count makes the "all crossed public edges are supported"
        # check constant-time for every local pair.
        topology_bad_prefix = np.r_[0, np.cumsum(~topology_support, dtype=np.int64)]
    pair_left: list[int] = []
    pair_right: list[int] = []
    for right in order:
        station_bin = int(np.floor(stations[right]))
        candidates: list[int] = []
        for key in range(station_bin - int(np.ceil(station_radius)) - 1,
                         station_bin + int(np.ceil(station_radius)) + 2):
            candidates.extend(bins.get(key, ()))
        if candidates:
            previous = np.asarray(candidates, dtype=np.int64)
            previous = previous[frames[previous] < frames[right]]
            pair_left.extend(previous.tolist())
            pair_right.extend([int(right)] * len(previous))
        bins[station_bin].append(int(right))
    left_ids, right_ids, weights, topology_rejected = _vector_edge_batch(
        np.asarray(pair_left, dtype=np.int64), np.asarray(pair_right, dtype=np.int64),
        stations, sectors, points, quality, base, z, topology_support, topology_bad_prefix,
        station_radius=station_radius, distance_radius=distance_radius,
        edge_threshold=edge_threshold, target_gate=target_gate,
        max_edges_per_candidate=max_edges_per_candidate)
    proposed.extend((float(weight), int(left), int(right))
                    for left, right, weight in zip(left_ids, right_ids, weights, strict=True))

    proposed.sort(key=lambda value: (-value[0], value[1], value[2]))
    union = _UnionFind(count, frames)
    kept: list[tuple[float, int, int]] = []
    matching_conflicts = 0
    for weight, left, right in proposed:
        if union.union_if_disjoint_frames(left, right):
            kept.append((weight, left, right))
        else:
            matching_conflicts += 1

    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(count)]
    for weight, left, right in kept:
        neighbors[left].append((right, weight))
        neighbors[right].append((left, weight))
    revised = base.copy()
    support = np.zeros(count, dtype=np.float64)
    for index, row in enumerate(neighbors):
        if not row:
            continue
        # Keep one vote from each distinct public view.  The graph matching
        # already prevents duplicate frames within a track, while this guard
        # also handles a caller that repeats a frame under a different slot.
        unique: dict[tuple[int, int], tuple[int, float]] = {}
        for other, weight in sorted(row, key=lambda value: (-value[1], value[0])):
            view = _view_key(stations[other], sectors[other])
            if view not in unique:
                unique[view] = (other, weight)
        values = list(unique.values())
        total = sum(weight for _, weight in values)
        if total <= 1.e-12:
            continue
        vote = sum((weight * base[other] for other, weight in values), np.zeros(CLASS_COUNT)) / total
        # The candidate's own public posterior remains in the state, so a
        # single ambiguous neighbor cannot replace it outright.
        revised[index] = (1. - float(strength)) * base[index] + float(strength) * vote
        revised[index] /= max(float(revised[index].sum()), 1.e-12)
        support[index] = float(sum(min(1., weight) for _, weight in values))

    roots = [union.find(i) for i in range(count)]
    root_order = {root: rank for rank, root in enumerate(sorted(set(roots), key=lambda item: (item,))) }
    track_ids = np.asarray([root_order[root] for root in roots], dtype=np.int32)
    # Public edge contract is ``(left_index, right_index, weight)``.  The
    # internal list is sorted as ``(weight, left, right)`` for deterministic
    # matching, so reorder it at the boundary.
    edges = (np.asarray([(left, right, weight) for weight, left, right in kept], dtype=np.float32)
             .reshape(-1, 3) if kept else np.empty((0, 3), dtype=np.float32))
    stats = {
        "enabled": True, "candidate_count": int(count), "proposed_edges": int(len(proposed)),
        "kept_edges": int(len(kept)), "topology_rejected": int(topology_rejected),
        "matching_conflicts": int(matching_conflicts), "track_count": int(len(root_order)),
        "mean_edge_weight": float(edges[:, 2].mean()) if len(edges) else 0.,
        "max_support": float(support.max(initial=0.)),
        "embedding_dimension": EMBEDDING_DIM, "class_count": CLASS_COUNT,
        "causal_order": "edge left frame < right frame; input order restored in outputs",
        "topology_role": "role zero only gates target propagation across unsupported public edges",
    }
    return {"probabilities": revised.astype(np.float32), "track_ids": track_ids,
            "support": support.astype(np.float32), "edges": edges, "stats": stats}


def _smoke() -> None:
    """Numerical contract check used by the v4 launcher and source review."""
    public = {
        "frames": np.array([0, 0, 1, 1, 2], dtype=np.int64),
        "slots": np.array([0, 1, 0, 1, 0], dtype=np.int64),
        "regions": np.array([0, 0, 0, 0, 0], dtype=np.int64),
        "stations": np.array([3., 3., 4., 4., 5.]),
        "sectors": np.array([1., 5., 1., 5., 1.]),
        "points": np.array([[0., 20.], [10., 20.], [0.3, 20.1], [10.2, 19.9], [0.6, 20.2]]),
        "quality": np.ones(5),
        "features": np.zeros((5, 96), dtype=np.float32),
    }
    probabilities = np.full((5, CLASS_COUNT), .02)
    probabilities[:, 0] = np.array([.78, .12, .74, .14, .76])
    probabilities[:, 1] = np.array([.08, .70, .10, .68, .09])
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    embedding = np.zeros((5, EMBEDDING_DIM), dtype=float)
    embedding[[0, 2, 4], 0] = 1.
    embedding[[1, 3], 1] = 1.
    links = np.ones((1, 8, 6), dtype=bool)
    first = associate(public, probabilities, embedding, links)
    assert first["stats"]["kept_edges"] >= 2
    assert np.allclose(first["probabilities"].sum(axis=1), 1.)
    assert len(set(first["track_ids"])) >= 2
    # Swapping two learned embeddings changes the causal posterior without
    # changing geometry or the number of candidates.
    swapped = embedding.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    second = associate(public, probabilities, swapped, links)
    assert not np.allclose(first["probabilities"], second["probabilities"])
    assert np.array_equal(first["probabilities"].shape, second["probabilities"].shape)
    # An unsupported target edge is rejected while the candidates remain.
    unsupported = np.zeros_like(links)
    blocked = associate(public, probabilities, embedding, unsupported)
    assert blocked["stats"]["topology_rejected"] > 0
    assert len(blocked["probabilities"]) == len(public["frames"])
    # Causality and one-to-one frame constraints hold for every retained edge.
    for left, right, _ in first["edges"]:
        assert public["frames"][int(left)] < public["frames"][int(right)]
    for track in np.unique(first["track_ids"]):
        assert len(set(public["frames"][first["track_ids"] == track])) == int(np.sum(first["track_ids"] == track))
    print({"kept_edges": first["stats"]["kept_edges"], "tracks": first["stats"]["track_count"],
           "topology_rejected": blocked["stats"]["topology_rejected"]})


if __name__ == "__main__":
    _smoke()


__all__ = ["associate", "CLASS_COUNT", "EMBEDDING_DIM"]

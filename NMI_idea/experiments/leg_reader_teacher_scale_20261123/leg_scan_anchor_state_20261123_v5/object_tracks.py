"""Public candidate association and persistent object tracks for v5.

This module deliberately keeps the deployed reader public-only. It accepts
anonymous candidate geometry, unary reader probabilities, and an association
embedding. A calibrated edge model is applied before one-to-one matching;
rejected edges are allowed to abstain and lower-confidence alternatives may
then be selected. Accepted edges form a forest whose components are public
object tracks. Evidence is deduplicated by rounded station/sector view and is
propagated with bounded path reliability, so a single weak edge cannot replace
an old posterior and a later anchor can reach an early node through a trusted
multi-hop path.

infer never reads an instance label or private geometry. A separate, explicit
infer_oracle helper exists only for mechanism diagnostics. The ObjectState
ledger admits each (frame, slot) once and retains the original unary/embedding
evidence; repeated updates therefore do not vote old output probabilities back
into the input.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


CLASS_COUNT = 8
PHYSICAL_COUNT = 6
EMBEDDING_DIM = 8
DEFAULT_STATION_RADIUS = 3.2
DEFAULT_CROSS_STATION_RADIUS = 2.05
DEFAULT_CROSS_ANGLE = 0.75
DEFAULT_DISTANCE_MM = 16.0
DEFAULT_EDGE_THRESHOLD = 0.90
MAX_ASSOCIATION_HOPS = 32

EDGE_FIELDS = (
    "base_weight", "embedding_cosine", "separation_mm", "axial_delta",
    "angle_delta", "min_quality", "min_physical_probability",
    "max_artifact_probability", "role_overlap", "target_probability_disagreement",
)


def _array(public: Mapping[str, Any], name: str, *, dtype: Any, ndim: int | None = None) -> np.ndarray:
    if name not in public:
        raise KeyError(f"public candidate mapping is missing {name!r}")
    value = np.asarray(public[name], dtype=dtype)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"public[{name!r}] must have {ndim} dimensions, got {value.shape}")
    return value


def _public_arrays(public: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and normalize public candidate arrays."""
    out = {
        "frames": _array(public, "frames", dtype=np.int64, ndim=1),
        "slots": _array(public, "slots", dtype=np.int64, ndim=1),
        "regions": _array(public, "regions", dtype=np.int64, ndim=1),
        "stations": _array(public, "stations", dtype=np.float64, ndim=1),
        "sectors": _array(public, "sectors", dtype=np.float64, ndim=1),
        "points": _array(public, "points", dtype=np.float64, ndim=2),
        "quality": _array(public, "quality", dtype=np.float64, ndim=1),
    }
    n = len(out["frames"])
    if out["points"].shape != (n, 2):
        raise ValueError(f"public['points'] must have shape ({n}, 2), got {out['points'].shape}")
    for name in ("slots", "regions", "stations", "sectors", "quality"):
        if len(out[name]) != n:
            raise ValueError(f"public[{name!r}] has length {len(out[name])}, expected {n}")
    if np.any(out["frames"] < 0) or np.any(out["slots"] < 0):
        raise ValueError("frame and slot indices must be nonnegative")
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError("public candidate arrays must be finite")
    out["quality"] = np.clip(out["quality"], 0., 1.)
    if "features" in public:
        features = np.asarray(public["features"])
        if features.ndim == 0 or len(features) != n:
            raise ValueError("public['features'] must have one row per candidate")
        out["features"] = features
    return out


def _probabilities(values: Any, count: int) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.shape != (count, CLASS_COUNT):
        raise ValueError(f"probabilities must have shape ({count}, {CLASS_COUNT}), got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    sums = p.sum(axis=1)
    if np.any(p < -1.e-8) or np.any(sums <= 1.e-12):
        shifted = p - np.max(p, axis=1, keepdims=True)
        p = np.exp(np.clip(shifted, -80., 80.))
        sums = p.sum(axis=1)
    return p / np.maximum(sums[:, None], 1.e-12)


def _embedding(values: Any, count: int) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    if z.shape != (count, EMBEDDING_DIM):
        raise ValueError(f"embedding must have shape ({count}, {EMBEDDING_DIM}), got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("embedding must be finite")
    norm = np.linalg.norm(z, axis=1, keepdims=True)
    return z / np.maximum(norm, 1.e-12)


def _links(reference_links: Any) -> np.ndarray | None:
    if reference_links is None:
        return None
    value = np.asarray(reference_links, dtype=bool)
    if value.ndim == 2:
        value = value[None]
    if value.ndim != 3 or value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < PHYSICAL_COUNT:
        raise ValueError("links must have shape [source, station_edge, role>=6]")
    return value


def _view_key(station: float, sector: float) -> tuple[int, int]:
    return int(np.rint(station)), int(np.rint(sector)) % 24


def _angle_delta(first: np.ndarray | float, second: np.ndarray | float) -> np.ndarray | float:
    return np.abs((np.asarray(first) - np.asarray(second) + 12.) % 24. - 12.)


def retrieve(public: Mapping[str, Any]) -> np.ndarray:
    """Retrieve public geometry-neighbour pairs without a frame-gap cut.

    A pair is eligible when it is in an axial neighbourhood (station distance
    at most 3.2) or a nearby cross-window neighbourhood (station distance at
    most 2.05 and circular sector difference at least 0.75), is within 16 mm in
    public world coordinates, and comes from distinct rounded views and
    distinct frames. Returned rows are [earlier_frame_index, later_index] and
    are deterministic. The same function is used for training pair extraction
    and inference; no hidden instance or source label is consulted.
    """
    a = _public_arrays(public)
    n = len(a["frames"])
    if n < 2:
        return np.empty((0, 2), dtype=np.int64)
    frames, stations, sectors, points = a["frames"], a["stations"], a["sectors"], a["points"]
    order = np.lexsort((np.arange(n), a["slots"], frames))
    bins: dict[int, list[int]] = defaultdict(list)
    pairs: list[tuple[int, int]] = []
    radius_bins = int(np.ceil(DEFAULT_STATION_RADIUS)) + 1
    view_keys = [_view_key(float(stations[i]), float(sectors[i])) for i in range(n)]
    for right in order:
        right = int(right)
        station_bin = int(np.floor(stations[right]))
        candidates: list[int] = []
        for key in range(station_bin - radius_bins, station_bin + radius_bins + 1):
            candidates.extend(bins.get(key, ()))
        if candidates:
            left = np.asarray(candidates, dtype=np.int64)
            distinct_frame = frames[left] != frames[right]
            if np.any(distinct_frame):
                left = left[distinct_frame]
                ds = np.abs(stations[left] - stations[right])
                da = np.asarray(_angle_delta(sectors[left], sectors[right]), dtype=np.float64)
                geometry = ((ds > 0.) & (ds <= DEFAULT_STATION_RADIUS)) | (
                    (ds <= DEFAULT_CROSS_STATION_RADIUS) & (da >= DEFAULT_CROSS_ANGLE))
                if np.any(geometry):
                    left = left[geometry]
                    delta = points[left] - points[right]
                    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                    keep = np.isfinite(distance) & (distance <= DEFAULT_DISTANCE_MM)
                    if np.any(keep):
                        left = left[keep]
                        distinct_view = np.asarray([view_keys[int(i)] != view_keys[right] for i in left])
                        for value in left[distinct_view]:
                            i = int(value)
                            if (frames[i], a["slots"][i]) <= (frames[right], a["slots"][right]):
                                pairs.append((i, right))
                            else:
                                pairs.append((right, i))
        bins[station_bin].append(right)
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    result = np.asarray(sorted(set(pairs), key=lambda pair: (pair[0], pair[1])), dtype=np.int64)
    return result.reshape(-1, 2)


def edge_inputs(
    public: Mapping[str, Any],
    unary: Any,
    embedding: Any,
    pairs: Any,
) -> np.ndarray:
    """Construct public edge features for externally supplied labels/calibration."""
    a = _public_arrays(public)
    n = len(a["frames"])
    p = _probabilities(unary, n)
    z = _embedding(embedding, n)
    edges = np.asarray(pairs, dtype=np.int64)
    if edges.size == 0:
        return np.empty((0, len(EDGE_FIELDS)), dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] < 2:
        raise ValueError("pairs must have shape [n, 2] or [n, >=2]")
    left, right = edges[:, 0], edges[:, 1]
    if np.any(left < 0) or np.any(right < 0) or np.any(left >= n) or np.any(right >= n):
        raise IndexError("edge indices are outside the candidate table")
    ds = np.abs(a["stations"][right] - a["stations"][left])
    da = np.asarray(_angle_delta(a["sectors"][left], a["sectors"][right]), dtype=np.float64)
    world = np.linalg.norm(a["points"][right] - a["points"][left], axis=1)
    cosine = np.sum(z[left] * z[right], axis=1)
    base = np.exp(-.5 * (world / 6.5) ** 2 - .5 * (ds / 1.9) ** 2 - .5 * (da / 3.2) ** 2)
    physical_left = p[left, :PHYSICAL_COUNT].sum(axis=1)
    physical_right = p[right, :PHYSICAL_COUNT].sum(axis=1)
    physical = np.minimum(physical_left, physical_right)
    artifact = np.maximum(p[left, PHYSICAL_COUNT:].sum(axis=1), p[right, PHYSICAL_COUNT:].sum(axis=1))
    overlap = np.sum(p[left, :PHYSICAL_COUNT] * p[right, :PHYSICAL_COUNT], axis=1)
    disagreement = np.abs(p[left, 0] - p[right, 0])
    return np.column_stack((base, cosine, world / DEFAULT_DISTANCE_MM, ds / DEFAULT_STATION_RADIUS,
                            da / 2., np.minimum(a["quality"][left], a["quality"][right]), physical,
                            artifact, overlap, disagreement)).astype(np.float64, copy=False)


def _fit_logistic(features: np.ndarray, labels: np.ndarray, regularization: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = features.mean(axis=0)
    scale = np.maximum(features.std(axis=0), .035)
    x = np.c_[(features - center) / scale, np.ones(len(features))]
    target = np.clip(labels.astype(np.float64), 0., 1.)
    penalty = np.r_[np.full(features.shape[1], regularization), 0.]

    def objective(weight: np.ndarray):
        logits = x @ weight
        loss = np.mean(np.logaddexp(0., logits) - target * logits) + .5 * np.sum(penalty * weight * weight)
        gradient = x.T @ (expit(logits) - target) / max(len(target), 1) + penalty * weight
        return float(loss), gradient

    fit = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"edge confidence fit failed: {fit.message}")
    return np.asarray(fit.x, dtype=np.float64), center, scale


def fit_edge_model(
    train_inputs: Any,
    train_labels: Any,
    cal_inputs: Any | None = None,
    cal_labels: Any | None = None,
    *,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
    regularization: float = 2.e-3,
) -> dict[str, Any]:
    """Fit edge confidence from labelled train/cal panels supplied by caller."""
    x = np.asarray(train_inputs, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(EDGE_FIELDS) or len(x) != len(y) or len(x) < 2:
        raise ValueError("train_inputs must be [n, 10] and match train_labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("edge training inputs must be finite")
    if np.any((y < 0.) | (y > 1.)):
        raise ValueError("edge labels must be in [0, 1]")
    weight, center, scale = _fit_logistic(x, y, regularization)
    model: dict[str, Any] = {
        "mean": center.tolist(), "scale": scale.tolist(), "weight": weight.tolist(),
        "features": list(EDGE_FIELDS), "threshold": float(threshold),
        "train_rows": int(len(y)), "train_positive": int(np.count_nonzero(y >= .5)),
        "calibration": [1., 0.], "cal_rows": 0,
    }
    if (cal_inputs is None) != (cal_labels is None):
        raise ValueError("cal_inputs and cal_labels must be supplied together")
    if cal_inputs is not None:
        c = np.asarray(cal_inputs, dtype=np.float64)
        cy = np.asarray(cal_labels, dtype=np.float64).reshape(-1)
        if c.ndim != 2 or c.shape[1] != len(EDGE_FIELDS) or len(c) != len(cy) or not len(c):
            raise ValueError("cal_inputs must be a nonempty [n, 10] panel matching cal_labels")
        if not np.isfinite(c).all() or np.any((cy < 0.) | (cy > 1.)):
            raise ValueError("calibration inputs or labels are invalid")
        raw = (c - center) / scale @ weight[:-1] + weight[-1]
        design = np.c_[raw, np.ones(len(raw))]

        def objective(ab: np.ndarray):
            logits = design @ ab
            reg = np.array([1.e-3, 1.e-3])
            loss = np.mean(np.logaddexp(0., logits) - cy * logits) + .5 * np.sum(reg * ab * ab)
            grad = design.T @ (expit(logits) - cy) / len(cy) + reg * ab
            return float(loss), grad

        calibrated = minimize(objective, np.array([1., 0.]), jac=True, method="L-BFGS-B")
        if not calibrated.success:
            raise RuntimeError(f"edge calibration fit failed: {calibrated.message}")
        model["calibration"] = np.asarray(calibrated.x, dtype=np.float64).tolist()
        model["cal_rows"] = int(len(cy))
        model["cal_positive"] = int(np.count_nonzero(cy >= .5))
    return model


def edge_probabilities(inputs: Any, edge_model: Mapping[str, Any] | None = None) -> np.ndarray:
    """Return calibrated public edge confidence for each feature row."""
    x = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(EDGE_FIELDS):
        raise ValueError("edge inputs must have shape [n, 10]")
    if not len(x):
        return np.empty(0, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("edge inputs must be finite")
    if edge_model is None:
        base, cosine, separation, axial, angle, quality, physical, artifact, overlap, disagreement = x.T
        logits = (-5.0 + 4.0 * base + 4.0 * cosine + 1.5 * quality + 2.0 * physical
                  + 3.0 * overlap - 3.0 * artifact - .5 * separation - .5 * axial
                  - .3 * angle - disagreement)
        return expit(logits)
    mean = np.asarray(edge_model["mean"], dtype=np.float64)
    scale = np.asarray(edge_model["scale"], dtype=np.float64)
    weight = np.asarray(edge_model["weight"], dtype=np.float64)
    if mean.shape != (len(EDGE_FIELDS),) or scale.shape != mean.shape or weight.shape != (len(EDGE_FIELDS) + 1,):
        raise ValueError("edge_model has incompatible feature dimensions")
    raw = ((x - mean) / np.maximum(scale, .035)) @ weight[:-1] + weight[-1]
    calibration = np.asarray(edge_model.get("calibration", [1., 0.]), dtype=np.float64)
    if calibration.shape != (2,):
        raise ValueError("edge_model calibration must contain [scale, bias]")
    return expit(calibration[0] * raw + calibration[1])


def edge_threshold(edge_model: Mapping[str, Any] | None) -> float:
    if edge_model is None:
        return DEFAULT_EDGE_THRESHOLD
    return float(edge_model.get("threshold", DEFAULT_EDGE_THRESHOLD))


def _span_mask(links: np.ndarray | None, first: float, last: float) -> np.ndarray | None:
    if links is None:
        return None
    sources, edge_count, roles = links.shape
    mask = np.ones((sources, PHYSICAL_COUNT), dtype=bool)
    # A same-station cross-window pair has no axial span to validate.  In
    # particular, station 0.5 should not accidentally consume edge 0 merely
    # because ceil(0.5) is one.
    if np.isclose(float(first), float(last)):
        return mask
    low = int(np.floor(min(first, last)))
    high = int(np.ceil(max(first, last))) - 1
    begin, end = max(0, low), min(edge_count - 1, high)
    if begin <= end:
        mask &= np.all(links[:, begin:end + 1, :PHYSICAL_COUNT], axis=1)
    return mask


class _Forest:
    """Maximum-confidence forest with frame and full-span topology guards."""

    def __init__(self, frames: np.ndarray, stations: np.ndarray, links: np.ndarray | None):
        n = len(frames)
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)
        self.frames = [{int(v)} for v in frames]
        self.indices = [{int(i)} for i in range(n)]
        self.min_station = stations.astype(float).copy()
        self.max_station = stations.astype(float).copy()
        self.mask = [(_span_mask(links, stations[i], stations[i]) if links is not None else None) for i in range(n)]
        self.links = links

    def find(self, item: int) -> int:
        root = int(item)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            previous = int(self.parent[item])
            self.parent[item] = root
            item = previous
        return root

    def try_union(self, first: int, second: int) -> bool:
        left, right = self.find(first), self.find(second)
        if left == right or self.frames[left].intersection(self.frames[right]):
            return False
        low = min(self.min_station[left], self.min_station[right])
        high = max(self.max_station[left], self.max_station[right])
        combined = _span_mask(self.links, low, high) if self.links is not None else None
        if combined is not None and not np.any(combined):
            return False
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]
        self.frames[left].update(self.frames[right])
        self.indices[left].update(self.indices[right])
        self.frames[right] = set()
        self.indices[right] = set()
        self.min_station[left] = low
        self.max_station[left] = high
        self.mask[left] = combined
        return True


def _path_reliabilities(
    sources: list[int],
    adjacency: list[list[tuple[int, float]]],
    *,
    max_hops: int = MAX_ASSOCIATION_HOPS,
) -> dict[int, dict[int, float]]:
    result: dict[int, dict[int, float]] = {}
    for source in sources:
        values = {source: 1.0}
        queue: deque[tuple[int, int]] = deque([(source, 0)])
        while queue:
            node, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for other, probability in adjacency[node]:
                if other in values:
                    continue
                values[other] = max(0., min(1., values[node] * float(probability)))
                queue.append((other, hops + 1))
        result[source] = values
    return result


def _track_outputs(
    arrays: dict[str, np.ndarray],
    unary: np.ndarray,
    edges: np.ndarray,
    links: np.ndarray | None,
    *,
    strength_cap: float = .42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    n = len(unary)
    frames = arrays["frames"]
    stations = arrays["stations"]
    sectors = arrays["sectors"]
    quality = arrays["quality"]
    view_keys = [_view_key(float(stations[i]), float(sectors[i])) for i in range(n)]
    forest = _Forest(frames, stations, links)
    for left, right, probability in edges:
        forest.try_union(int(left), int(right))
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(n):
        components[forest.find(index)].append(index)
    anchors = {root: min(indices) for root, indices in components.items()}
    track_ids = np.empty(n, dtype=np.int64)
    revised = unary.copy()
    support = np.zeros(n, dtype=np.float64)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for left, right, probability in edges:
        i, j, p = int(left), int(right), float(probability)
        if forest.find(i) != forest.find(j):
            continue
        adjacency[i].append((j, p))
        adjacency[j].append((i, p))

    tracks_public: list[dict[str, Any]] = []
    hypotheses: list[dict[str, Any]] = []
    for root, indices in sorted(components.items(), key=lambda item: anchors[item[0]]):
        indices = sorted(indices)
        anchor = anchors[root]
        track_ids[indices] = anchor
        views: dict[tuple[int, int], int] = {}
        for index in sorted(indices, key=lambda i: (-quality[i], i)):
            views.setdefault(_view_key(float(stations[index]), float(sectors[index])), index)
        representatives = sorted(views.values())
        paths = _path_reliabilities(representatives, adjacency)
        mask = forest.mask[forest.find(anchor)]
        if mask is None:
            role_mask = np.ones(PHYSICAL_COUNT, dtype=bool)
            source_role_mask = None
        else:
            source_role_mask = np.asarray(mask, dtype=bool)
            role_mask = np.any(source_role_mask, axis=0)
        for target in indices:
            contributions: list[tuple[float, int]] = []
            for representative in representatives:
                if representative == target:
                    continue
                if view_keys[representative] == view_keys[target]:
                    continue
                weight = paths.get(representative, {}).get(target, 0.)
                if weight > 1.e-6:
                    contributions.append((weight, representative))
            if not contributions:
                continue
            total = float(sum(weight for weight, _ in contributions))
            vote = np.zeros(CLASS_COUNT, dtype=np.float64)
            for weight, representative in contributions:
                vote += weight * unary[representative]
            vote /= max(total, 1.e-12)
            allowed = np.r_[role_mask, True, True]
            own_allowed_mass = float(unary[target, allowed].sum())
            vote_allowed = vote[allowed]
            if vote_allowed.sum() > 1.e-12:
                vote[allowed] = vote_allowed * own_allowed_mass / vote_allowed.sum()
                # Roles unsupported by the same-source full-span track are
                # never imported from a neighbour. Preserve the candidate's
                # own mass for those roles rather than blending a rival role.
                vote[~allowed] = unary[target, ~allowed]
            else:
                vote = unary[target].copy()
            independent_views = len({
                view_keys[representative]
                for _, representative in contributions
            })
            mixing = min(strength_cap, .34 * min(1., total) * min(1., independent_views / 2.))
            revised[target] = (1. - mixing) * unary[target] + mixing * vote
            revised[target] /= max(revised[target].sum(), 1.e-12)
            support[target] = min(4., total)
        aggregate = np.average(revised[indices], axis=0, weights=np.maximum(quality[indices], .05))
        track_record = {
            "track_id": int(anchor),
            "indices": [int(i) for i in indices],
            "evidence_views": [[int(v[0]), int(v[1])] for v in sorted(views)],
            "station_min": float(np.min(stations[indices])),
            "station_max": float(np.max(stations[indices])),
            "support": float(np.max(support[indices], initial=0.)),
            "source_role_mask": None if source_role_mask is None else source_role_mask.astype(bool).tolist(),
            "supported_roles": np.flatnonzero(role_mask).astype(int).tolist(),
        }
        tracks_public.append(track_record)
        hypotheses.append({
            "track_id": int(anchor), "indices": [int(i) for i in indices],
            "posterior": aggregate.astype(float).tolist(),
            "source_role_mask": track_record["source_role_mask"],
        })
    return revised, track_ids, support, tracks_public, hypotheses


def _infer_impl(
    public: Mapping[str, Any],
    unary: Any,
    embedding: Any,
    links: Any,
    *,
    edge_model: Mapping[str, Any] | None,
    oracle_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    arrays = _public_arrays(public)
    n = len(arrays["frames"])
    base = _probabilities(unary, n)
    z = _embedding(embedding, n)
    pairs = retrieve(public)
    features = edge_inputs(public, base, z, pairs)
    probabilities = edge_probabilities(features, edge_model)
    if oracle_edges is not None:
        truth_pairs = {tuple(map(int, row[:2])) for row in np.asarray(oracle_edges, dtype=np.int64).reshape(-1, 2)}
        probabilities = np.asarray([
            1. if tuple(map(int, pair)) in truth_pairs else 0.
            for pair in pairs
        ], dtype=np.float64)
    threshold = 1.e-12 if oracle_edges is not None else edge_threshold(edge_model)
    accepted = probabilities >= threshold
    scored = np.column_stack((pairs, probabilities)).astype(np.float64, copy=False)
    if len(pairs):
        order = np.lexsort((pairs[:, 1], pairs[:, 0], -probabilities))
    else:
        order = np.empty(0, dtype=np.int64)
    retained: list[tuple[int, int, float]] = []
    forest = _Forest(arrays["frames"], arrays["stations"], _links(links))
    topology_rejected = 0
    matching_rejected = 0
    for position in order:
        if not accepted[position]:
            continue
        left, right, probability = int(pairs[position, 0]), int(pairs[position, 1]), float(probabilities[position])
        root_left, root_right = forest.find(left), forest.find(right)
        if root_left != root_right and forest.frames[root_left].intersection(forest.frames[root_right]):
            matching_rejected += 1
            continue
        if root_left != root_right:
            low = min(forest.min_station[root_left], forest.min_station[root_right])
            high = max(forest.max_station[root_left], forest.max_station[root_right])
            mask = _span_mask(forest.links, low, high) if forest.links is not None else None
            if mask is not None and not np.any(mask):
                topology_rejected += 1
                continue
        if forest.try_union(left, right):
            retained.append((left, right, probability))
        else:
            matching_rejected += 1
    edges = np.asarray(retained, dtype=np.float64).reshape(-1, 3) if retained else np.empty((0, 3), dtype=np.float64)
    revised, track_ids, support, tracks, hypotheses = _track_outputs(
        arrays, base, edges, forest.links)
    stats = {
        "candidate_count": int(n),
        "retrieved_edges": int(len(pairs)),
        "confidence_accepted": int(np.count_nonzero(accepted)),
        "abstained_edges": int(len(pairs) - np.count_nonzero(accepted)),
        "kept_edges": int(len(edges)),
        "topology_rejected": int(topology_rejected),
        "matching_rejected": int(matching_rejected),
        "track_count": int(len(tracks)),
        "edge_threshold": float(threshold),
        "mean_edge_confidence": float(edges[:, 2].mean()) if len(edges) else 0.,
        "causal_pair_order": "left/right are ordered by (frame, slot); no frame-gap cutoff",
        "association_truth_used": bool(oracle_edges is not None),
        "evidence_deduplication": "one representative per rounded station/sector view per track",
        "topology_rule": "one source and role support across the complete station span",
    }
    return {
        "probabilities": revised.astype(np.float32),
        "unary": base.astype(np.float32),
        "embedding": z.astype(np.float32),
        "track_ids": track_ids.astype(np.int32),
        "support": support.astype(np.float32),
        "edges": edges.astype(np.float32),
        "scored_edges": scored.astype(np.float32),
        "tracks": tracks,
        "hypotheses": hypotheses,
        "stats": stats,
    }


def infer(
    public: Mapping[str, Any],
    unary: Any,
    embedding: Any,
    links: Any,
    *,
    edge_model: Mapping[str, Any] | None = None,
    oracle_instance: Any | None = None,
) -> dict[str, Any]:
    """Infer public tracks; truth labels are refused on the deployed path."""
    if oracle_instance is not None:
        raise ValueError("oracle_instance is analysis-only; call infer_oracle explicitly")
    return _infer_impl(public, unary, embedding, links, edge_model=edge_model)


def infer_oracle(
    public: Mapping[str, Any],
    unary: Any,
    embedding: Any,
    links: Any,
    instance: Any,
    *,
    edge_model: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analysis-only identity-edge diagnostic; never call from deployment."""
    arrays = _public_arrays(public)
    labels = np.asarray(instance)
    if labels.ndim != 1 or len(labels) != len(arrays["frames"]):
        raise ValueError("oracle instance labels must have one value per public candidate")
    pairs = retrieve(public)
    oracle = pairs[labels[pairs[:, 0]] >= 0] if len(pairs) else pairs
    if len(oracle):
        oracle = oracle[labels[oracle[:, 0]] == labels[oracle[:, 1]]]
    return _infer_impl(public, unary, embedding, links, edge_model=edge_model, oracle_edges=oracle)


class ObjectState:
    """Incremental public object-state ledger.

    A candidate key is (frame, slot). Existing rows retain their original unary
    probabilities and embedding; only new rows are admitted. The whole ledger
    is re-inferred when a new frame arrives so later anchors can revise earlier
    objects, but repeated updates with no new keys return the cached state
    without re-voting old output probabilities.
    """

    def __init__(self, links: Any = None, edge_model: Mapping[str, Any] | None = None):
        self.links = links
        self.edge_model = edge_model
        self._public: dict[str, np.ndarray] | None = None
        self._unary: np.ndarray | None = None
        self._embedding: np.ndarray | None = None
        self._keys: set[tuple[int, int]] = set()
        self._state: dict[str, Any] | None = None
        self.admitted_count = 0

    @property
    def public(self) -> dict[str, np.ndarray] | None:
        return self._public

    def update(self, public: Mapping[str, Any], unary: Any, embedding: Any) -> dict[str, Any]:
        arrays = _public_arrays(public)
        n = len(arrays["frames"])
        p = _probabilities(unary, n)
        z = _embedding(embedding, n)
        keys = [(int(frame), int(slot)) for frame, slot in zip(
            arrays["frames"], arrays["slots"], strict=True)]
        new_indices = [i for i, key in enumerate(keys) if key not in self._keys]
        if not new_indices:
            if self._state is None:
                self._public = {name: value.copy() for name, value in arrays.items()}
                self._unary, self._embedding = p.copy(), z.copy()
                self._state = infer(self._public, self._unary, self._embedding, self.links,
                                    edge_model=self.edge_model)
                self.admitted_count = 0
            return self._state
        additions = np.asarray(new_indices, dtype=np.int64)
        if self._public is None:
            self._public = {name: value[additions].copy() for name, value in arrays.items()}
            self._unary, self._embedding = p[additions].copy(), z[additions].copy()
        else:
            for name in self._public:
                self._public[name] = np.concatenate(
                    (self._public[name], arrays[name][additions]), axis=0)
            self._unary = np.concatenate((self._unary, p[additions]), axis=0)
            self._embedding = np.concatenate((self._embedding, z[additions]), axis=0)
        self._keys.update(keys[i] for i in new_indices)
        order = np.lexsort((
            np.arange(len(self._keys)), self._public["slots"], self._public["frames"]))
        for name in self._public:
            self._public[name] = self._public[name][order]
        self._unary = self._unary[order]
        self._embedding = self._embedding[order]
        self.admitted_count = len(self._keys)
        self._state = infer(self._public, self._unary, self._embedding, self.links,
                            edge_model=self.edge_model)
        self._state["ledger"] = {
            "admitted_count": int(self.admitted_count),
            "unique_keys": int(len(self._keys)),
            "repeated_update_revoted": False,
            "key": "(frame, slot)",
        }
        return self._state

    def query(self) -> dict[str, Any] | None:
        """Return the current whole-history state without admitting evidence."""
        return self._state


def _smoke() -> None:
    """Mechanism checks for multi-hop anchors, topology, gating, and ledger."""
    public = {
        "frames": np.array([0, 1, 2], dtype=np.int64),
        "slots": np.array([0, 0, 0], dtype=np.int64),
        "regions": np.zeros(3, dtype=np.int64),
        "stations": np.array([0., 1., 2.]),
        "sectors": np.array([1., 1., 1.]),
        "points": np.array([[0., 0.], [8.5, 0.], [17., 0.]]),
        "quality": np.ones(3),
        "features": np.zeros((3, 96), dtype=np.float32),
    }
    unary = np.full((3, CLASS_COUNT), .01, dtype=np.float64)
    unary[:, 0] = [.50, .50, .90]
    unary[:, 1] = [.40, .40, .02]
    unary /= unary.sum(axis=1, keepdims=True)
    embedding = np.zeros((3, EMBEDDING_DIM), dtype=np.float64)
    embedding[:, 0] = 1.
    links = np.ones((1, 8, PHYSICAL_COUNT), dtype=bool)
    pairs = retrieve(public)
    assert np.array_equal(pairs, np.array([[0, 1], [1, 2]], dtype=np.int64)), pairs
    state = infer(public, unary, embedding, links)
    assert state["stats"]["kept_edges"] == 2, state["stats"]
    assert state["probabilities"][0, 0] > unary[0, 0], state["probabilities"]
    alternating = np.zeros((2, 8, PHYSICAL_COUNT), dtype=bool)
    alternating[0, 0, :] = True
    alternating[1, 1, :] = True
    blocked = infer(public, unary, embedding, alternating)
    assert blocked["stats"]["topology_rejected"] >= 1, blocked["stats"]
    assert blocked["stats"]["kept_edges"] == 1, blocked["stats"]
    low = np.array([[0, 1], [0, 2]], dtype=np.int64)
    values = edge_inputs(public, unary, embedding, low)
    model = fit_edge_model(np.vstack((values, values)), np.array([0., 1., 0., 1.]), threshold=.90)
    probs = edge_probabilities(values, model)
    assert probs.shape == (2,)
    ledger = ObjectState(links=links)
    first = ledger.update({k: v[:2] for k, v in public.items()}, unary[:2], embedding[:2])
    before = first["probabilities"].copy()
    repeated = ledger.update({k: v[:2] for k, v in public.items()}, unary[:2], embedding[:2])
    assert ledger.admitted_count == 2 and np.array_equal(before, repeated["probabilities"])
    after = ledger.update(public, unary, embedding)
    assert ledger.admitted_count == 3 and after["probabilities"][0, 0] > before[0, 0]
    try:
        infer(public, unary, embedding, links, oracle_instance=np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("deployed infer accepted oracle_instance")
    print({"API": ["retrieve", "edge_inputs", "fit_edge_model", "edge_probabilities",
                   "infer", "infer_oracle", "ObjectState"],
           "multi_hop": True, "source_alternation_rejected": True,
           "ledger_deduplicated": True, "kept_edges": int(state["stats"]["kept_edges"]),
           "tracks": int(state["stats"]["track_count"])})


__all__ = [
    "CLASS_COUNT", "EMBEDDING_DIM", "EDGE_FIELDS", "ObjectState", "edge_inputs",
    "edge_probabilities", "edge_threshold", "fit_edge_model", "infer", "infer_oracle", "retrieve",
]


if __name__ == "__main__":
    _smoke()

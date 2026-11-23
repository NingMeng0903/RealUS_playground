"""Object identity and candidate usability fusion for the v6 reader.

The v5 graph mixes an observation's physical role with whether that particular
observation is a good image to submit.  This module keeps those quantities
separate.  Association edges are still inferred from the public v5 matcher,
while each accepted track gets a robust conditional role posterior.  The
posterior is repeated for its members and the observation-level ``usable``
value is retained for delivery ranking.

No joint probability is formed by multiplying role and usability.  A caller
that needs a delivery score should evaluate that event with its own calibrated
model.  ``oracle_instance`` is accepted only for an explicitly labelled
diagnostic and is recorded in the returned statistics.
"""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


CLASS_COUNT = 8
PHYSICAL_COUNT = 6
MAX_HOPS = 32
EPS = 1.0e-6


def _v5():
    """Load the frozen public matcher without making v6 import order global."""
    sibling = Path(__file__).resolve().parent.parent / "leg_scan_anchor_state_20261123_v5"
    if str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
    import object_tracks  # type: ignore

    return object_tracks


def _public(public: Mapping[str, Any]) -> dict[str, np.ndarray]:
    required = ("frames", "slots", "stations", "sectors", "points", "quality")
    arrays = {name: np.asarray(public[name]) for name in required}
    n = len(arrays["frames"])
    if arrays["points"].shape != (n, 2):
        raise ValueError(f"points must have shape ({n}, 2)")
    if any(value.ndim != 1 or len(value) != n for value in arrays.values() if value is not arrays["points"]):
        raise ValueError("public candidate fields must be one-dimensional and aligned")
    arrays["frames"] = arrays["frames"].astype(np.int64, copy=False)
    arrays["slots"] = arrays["slots"].astype(np.int64, copy=False)
    arrays["stations"] = arrays["stations"].astype(np.float64, copy=False)
    arrays["sectors"] = arrays["sectors"].astype(np.float64, copy=False)
    arrays["points"] = arrays["points"].astype(np.float64, copy=False)
    arrays["quality"] = np.clip(arrays["quality"].astype(np.float64, copy=False), 0., 1.)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("public candidate fields must be finite")
    return arrays


def _probabilities(unary: Any, n: int) -> np.ndarray:
    values = np.asarray(unary, dtype=np.float64)
    if values.shape != (n, CLASS_COUNT):
        raise ValueError(f"unary probabilities must have shape ({n}, 8)")
    if not np.isfinite(values).all() or np.any(values < -EPS):
        raise ValueError("unary probabilities must be finite and nonnegative")
    values = np.maximum(values, 0.)
    return values / np.maximum(values.sum(axis=1, keepdims=True), EPS)


def split_roles(unary: Any) -> dict[str, np.ndarray]:
    """Split eight reader classes into conditional roles and local usability.

    ``role`` is conditional on the six physical roles.  The mass assigned to
    artifact/unknown classes is returned separately so it can calibrate the
    observation-level usability without being mixed into object identity.
    """
    values = np.asarray(unary, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != CLASS_COUNT:
        raise ValueError("unary must have shape [candidate, 8]")
    values = np.maximum(values, 0.)
    physical_mass = values[:, :PHYSICAL_COUNT].sum(axis=1)
    role = values[:, :PHYSICAL_COUNT] / np.maximum(physical_mass[:, None], EPS)
    return {
        "role": role.astype(np.float32),
        "physical_mass": physical_mass.astype(np.float32),
        "artifact_probability": values[:, PHYSICAL_COUNT:].sum(axis=1).astype(np.float32),
    }


def _edges_from_graph(graph: Mapping[str, Any], n: int) -> np.ndarray:
    edges = np.asarray(graph.get("edges", np.empty((0, 3))), dtype=np.float64)
    if edges.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] < 3:
        raise ValueError("graph edges must have shape [edge, 3]")
    edges = edges[:, :3].copy()
    if not np.isfinite(edges).all() or np.any(edges[:, :2] < 0) or np.any(edges[:, :2] >= n):
        raise ValueError("graph edges contain invalid indices")
    edges[:, :2] = np.rint(edges[:, :2])
    edges[:, 2] = np.clip(edges[:, 2], 0., 1.)
    return edges


def _components(n: int, edges: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    parent = np.arange(n, dtype=np.int64)

    def find(item: int) -> int:
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            before = int(parent[item])
            parent[item] = root
            item = before
        return root

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left, right, probability in edges:
        i, j, p = int(left), int(right), float(probability)
        adjacency[i].append((j, p))
        adjacency[j].append((i, p))
        union(i, j)
    track_ids = np.asarray([find(i) for i in range(n)], dtype=np.int64)
    # Stable public IDs use the smallest candidate index in a component.
    remap = {}
    for index, root in enumerate(track_ids):
        remap.setdefault(int(root), index)
        remap[int(root)] = min(remap[int(root)], index)
    track_ids = np.asarray([remap[int(root)] for root in track_ids], dtype=np.int64)
    return track_ids, adjacency


def _path_weights(indices: list[int], adjacency: list[list[tuple[int, float]]]) -> dict[int, float]:
    """Find maximum product edge reliability from the component anchor."""
    anchor = min(indices)
    best = {anchor: 1.0}
    queue: list[tuple[float, int, int]] = [(-1., anchor, 0)]
    while queue:
        neg_score, node, hops = heappop(queue)
        score = -neg_score
        if score + EPS < best.get(node, 0.) or hops >= MAX_HOPS:
            continue
        for other, edge_probability in adjacency[node]:
            candidate = score * float(np.clip(edge_probability, 0., 1.))
            if candidate > best.get(other, 0.) + EPS:
                best[other] = candidate
                heappush(queue, (-candidate, other, hops + 1))
    return best


def _representatives(indices: list[int], arrays: Mapping[str, np.ndarray]) -> list[int]:
    chosen: dict[tuple[int, int], int] = {}
    for index in sorted(indices, key=lambda i: (-float(arrays["quality"][i]), i)):
        key = (int(np.rint(arrays["stations"][index])), int(np.rint(arrays["sectors"][index])) % 24)
        chosen.setdefault(key, index)
    return sorted(chosen.values())


def _fuse_track(indices: list[int], local_role: np.ndarray, usable: np.ndarray,
                arrays: Mapping[str, np.ndarray], adjacency: list[list[tuple[int, float]]],
                allowed_roles: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reps = _representatives(indices, arrays)
    paths = _path_weights(indices, adjacency)
    # Both measured quality and explicit usability gate evidence strength. The
    # root gets unit path weight and isolated tracks therefore preserve their
    # local role posterior exactly. Quality is deliberately not only a view
    # selector because two views with the same usability can have very
    # different evidential value.
    weights = np.asarray([
        max(0., paths.get(i, 0.)) * max(float(usable[i]), .0) * max(float(arrays["quality"][i]), .0)
        for i in reps
    ])
    if not np.any(weights > EPS):
        weights = np.ones(len(reps), dtype=np.float64)
    # Weighted geometric pooling is robust to a weak view. A high-quality,
    # independent contradiction still moves the object posterior in the
    # opposite direction. It avoids the v5 arithmetic vote that lowered a
    # clear candidate merely because a low-quality frame existed.
    evidence = np.asarray(local_role[reps], dtype=np.float64).copy()
    if allowed_roles is not None and np.any(allowed_roles):
        evidence *= np.asarray(allowed_roles, dtype=np.float64)[None, :]
        evidence /= np.maximum(evidence.sum(axis=1, keepdims=True), EPS)
    logs = np.log(np.clip(evidence, EPS, 1.))
    object_role = np.exp(np.average(logs, axis=0, weights=weights))
    if allowed_roles is not None and np.any(allowed_roles):
        object_role[~allowed_roles] = 0.
    object_role /= max(float(object_role.sum()), EPS)
    view_count = np.full(len(indices), len(reps), dtype=np.float32)
    # The object posterior remains a separate state. Candidate-facing role
    # probabilities retain local strong evidence and move toward the object
    # consensus only when an independent, sufficiently credible view exists.
    rep_weight = {int(index): float(weight) for index, weight in zip(reps, weights, strict=True)}
    total_weight = float(weights.sum())
    credibility = []
    for index in indices:
        own = rep_weight.get(int(index), 0.)
        independent = max(0., total_weight - own)
        # A low-quality or low-usability neighbour contributes little to this
        # candidate's displayed role. A strong contradictory view can still
        # move it substantially, so contradictions are never frozen out.
        alpha = min(.55, .42 * independent / (independent + .35)) if independent > .05 else 0.
        credibility.append(alpha)
    return (object_role, np.asarray(reps, dtype=np.int64), view_count,
            np.asarray(weights, dtype=np.float64), np.asarray(credibility, dtype=np.float64))


def infer_roles(
    public: Mapping[str, Any],
    unary: Any,
    embedding: Any,
    links: Any,
    edge_model: Mapping[str, Any] | None,
    usable: Any,
    *,
    oracle_instance: Any | None = None,
    graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer public tracks and fuse only conditional physical role evidence.

    ``usable`` is an observation-level value in [0, 1]. It is preserved in the
    output and only gates the evidence weight used to form ``object_role``.
    The returned ``role`` is conditional on physical roles and is never
    multiplied by ``usable``.
    """
    arrays = _public(public)
    n = len(arrays["frames"])
    probabilities = _probabilities(unary, n)
    embedding_values = np.asarray(embedding, dtype=np.float64)
    if embedding_values.shape != (n, 8) or not np.isfinite(embedding_values).all():
        raise ValueError("embedding must have shape [candidate, 8] and be finite")
    usable_values = np.asarray(usable, dtype=np.float64).reshape(-1)
    if len(usable_values) != n or not np.isfinite(usable_values).all():
        raise ValueError("usable must have one finite value per candidate")
    usable_values = np.clip(usable_values, 0., 1.)
    if graph is None:
        matcher = _v5()
        if oracle_instance is None:
            graph_state = matcher.infer(public, probabilities, embedding_values, links, edge_model=edge_model)
        else:
            graph_state = matcher.infer_oracle(public, probabilities, embedding_values, links,
                                               np.asarray(oracle_instance), edge_model=edge_model)
    else:
        if oracle_instance is not None:
            raise ValueError("oracle_instance cannot be combined with a supplied graph")
        graph_state = graph
    edges = _edges_from_graph(graph_state, n)
    if oracle_instance is not None:
        # Keep oracle diagnostics comparable to calibrated inference. Truth is
        # only used to choose which edge exists, then all retained edges use
        # the same capped strength in path weighting.
        edges[:, 2] = np.minimum(edges[:, 2], .99)
    track_ids, adjacency = _components(n, edges)
    split = split_roles(probabilities)
    local_role = np.asarray(split["role"], dtype=np.float64)
    role = local_role.copy()
    object_role = local_role.copy()
    view_counts = np.ones(n, dtype=np.float32)
    support = np.zeros(n, dtype=np.float32)
    tracks = defaultdict(list)
    for index, track in enumerate(track_ids):
        tracks[int(track)].append(index)
    representative_total = 0
    track_masks: dict[int, np.ndarray] = {}
    for record in graph_state.get("tracks", []):
        track_id = int(record.get("track_id", -1))
        raw_mask = record.get("source_role_mask")
        if raw_mask is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.ndim == 2 and mask.shape[1] >= PHYSICAL_COUNT:
            allowed = np.any(mask[:, :PHYSICAL_COUNT], axis=0)
            if np.any(allowed):
                track_masks[track_id] = allowed
    for track, indices in tracks.items():
        fused, reps, counts, weights, alphas = _fuse_track(
            indices, local_role, usable_values, arrays, adjacency, track_masks.get(track))
        representative_total += len(reps)
        object_role[indices] = fused
        for index, alpha in zip(indices, alphas, strict=True):
            candidate_role = (1. - alpha) * local_role[index] + alpha * fused
            if track in track_masks:
                candidate_role[track_masks[track] == 0] = 0.
            candidate_role /= max(float(candidate_role.sum()), EPS)
            role[index] = candidate_role
        view_counts[indices] = counts
        support[indices] = float(np.sum(weights))
    scored = np.asarray(graph_state.get("scored_edges", edges), dtype=np.float64)
    alternatives = []
    if scored.ndim == 2 and scored.shape[1] >= 3 and len(scored):
        threshold = float(graph_state.get("stats", {}).get("edge_threshold", .90))
        near = scored[(scored[:, 2] >= threshold - .08) & (scored[:, 2] < threshold + .08)]
        order = np.argsort(-near[:, 2], kind="stable")[:256]
        alternatives = [[int(row[0]), int(row[1]), float(row[2])] for row in near[order, :3]]
    stats = dict(graph_state.get("stats", {}))
    stats.update({"fusion": "physical_role_only_weighted_geometric_pool",
                  "candidate_count": n, "track_count": len(tracks),
                  "representative_view_count": int(representative_total),
                  "oracle_instance_used": oracle_instance is not None,
                  "oracle_edge_strength_cap": .99 if oracle_instance is not None else None,
                  "usable_is_separate_from_role": True,
                  "joint_probability_not_formed": True})
    return {"role": role.astype(np.float32), "local_role": local_role.astype(np.float32),
            "unary": probabilities.astype(np.float32),
            "physical_mass": split["physical_mass"], "artifact_probability": split["artifact_probability"],
            "usable": usable_values.astype(np.float32), "track_ids": track_ids.astype(np.int32),
            "support": support, "edges": edges.astype(np.float32),
            "object_role": object_role.astype(np.float32), "view_counts": view_counts,
            "alternative_edges": alternatives, "stats": stats}


def _smoke() -> None:
    public = {"frames": np.array([0, 1, 2, 3]), "slots": np.zeros(4),
              "stations": np.arange(4.), "sectors": np.ones(4),
              "points": np.column_stack((np.arange(4.) * 8., np.zeros(4))),
              "quality": np.ones(4)}
    unary = np.full((4, 8), .01)
    unary[0, :2] = [.95, .04]
    unary[1, :2] = [.20, .79]
    unary[2, :2] = [.20, .79]
    unary[3, :2] = [.80, .19]
    unary /= unary.sum(axis=1, keepdims=True)
    embedding = np.zeros((4, 8)); embedding[:, 0] = 1.
    links = np.ones((1, 8, 6), dtype=bool)
    direct = np.array([[0, 1, .99], [1, 2, .99]], dtype=float)
    # Weak views have the same usable value as the clear view in this fixture;
    # measured quality must still suppress their identity vote.
    weak_public = dict(public)
    weak_public["quality"] = np.array([1., .01, .01, 1.])
    low = infer_roles(weak_public, unary, embedding, links, None, np.ones(4), graph={"edges": direct})
    assert low["role"][0, 0] > .90, low["role"]
    assert low["artifact_probability"][3] > 0.
    contradiction = infer_roles(public, unary, embedding, links, None, np.ones(4),
                                graph={"edges": np.array([[0, 1, .99], [1, 2, .99], [2, 3, .99]])})
    assert contradiction["role"][0, 0] < low["role"][0, 0], (low["role"], contradiction["role"])
    assert contradiction["stats"]["joint_probability_not_formed"]
    low_quality_public = dict(public)
    low_quality_public["quality"] = np.array([1., .01, .01, 1.])
    high_quality_public = dict(public)
    high_quality_public["quality"] = np.ones(4)
    low_quality = infer_roles(low_quality_public, unary, embedding, links, None, np.ones(4), graph={"edges": direct})
    high_quality = infer_roles(high_quality_public, unary, embedding, links, None, np.ones(4), graph={"edges": direct})
    low_target = float(low_quality["object_role"][0, 0])
    high_target = float(high_quality["object_role"][0, 0])
    assert low_target > high_target, (low_target, high_target)
    print(f"quality_fixture target_role_probability low_q={low_target:.6f} high_q={high_target:.6f}", flush=True)
    print("role_fusion smoke passed", flush=True)


if __name__ == "__main__":
    _smoke()

"""Targeted public tracklet reacquisition for the v7 association audit.

The base v5 matcher remains authoritative for retrieval, embedding scores,
the .90 acceptance threshold, and all of its one-to-one/topology guards.  This
module only proposes a small number of *grouped* old/new tracklet bridges from
already-retrieved lower-confidence edges.  A model trained on disjoint frozen
v5 layouts may accept a proposal at a high-precision calibrated threshold;
otherwise the bridge is deliberately inactive.

Public API
----------
``recover(public, p, e, links, u, graph, cutoff=..., settings=..., route=...)``
returns a graph-shaped mapping.  ``graph['edges']`` contains the original
accepted edges plus accepted bridge edges; ``graph['scored_edges']`` remains
the original v5 scored retrieval table; ``graph['bridge_edges']`` records the
extra [left, right, bridge_probability] rows.  Refreshed ``tracks`` and
``source_role_mask`` are generated with the same v5 track output code.

Truth labels are used only by the offline fit/diagnostic entry points.  The
runtime ``recover`` path reads public arrays and the frozen model settings
only.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent
V5 = HANDOFF / "leg_scan_anchor_state_20261123_v5"
V6 = HANDOFF / "leg_scan_identity_state_20261123_v6"
V5_RUN = V5 / "run"
V6_RUN = V6 / "run" / "confirmed"
sys.path.insert(0, str(V6))
sys.path.insert(0, str(V5))

from causal_reader import inputs  # noqa: E402
from fit_public import load_public  # noqa: E402
from object_tracks import (  # noqa: E402
    _Forest,
    _public_arrays,
    _span_mask,
    _track_outputs,
    edge_inputs,
    edge_probabilities,
    edge_threshold,
    infer,
    retrieve,
)
from readout_state import heads  # noqa: E402
from reader_study import warm_bundle  # noqa: E402
from usability_fit import predict_usability  # noqa: E402
from v6_data import DEFAULT_PACK, atomic_json, digest, load_shard, read_json  # noqa: E402


BRIDGE_FIELDS = (
    "max_edge_score", "mean_top2_edge_score", "score_gap_to_base_threshold",
    "witness_count", "old_witness_count", "new_witness_count",
    "old_frame_count", "new_frame_count", "old_view_count", "new_view_count",
    "min_world_distance", "mean_world_distance", "min_station_delta", "mean_station_delta",
    "min_angle_delta", "max_embedding_cosine", "mean_embedding_cosine",
    "min_pair_quality", "mean_pair_quality", "max_pair_quality",
    "mean_physical_mass", "max_role_overlap", "mean_target_disagreement",
    "old_track_span", "new_track_span", "availability_min",
)

MAX_ASSOCIATION_HOPS = 32
_BUNDLE: dict[str, Any] | None = None
_FIT_CFG: dict[str, Any] | None = None


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _cfg_for_route(settings: Mapping[str, Any] | None, route: Any) -> dict[str, Any]:
    cfg = dict(settings or {})
    # Accept either the emitted ``bridge_model.json`` wrapper or its nested
    # ``settings`` object so callers do not have to duplicate unpacking logic.
    if "settings" in cfg and "model" not in cfg and isinstance(cfg["settings"], Mapping):
        cfg = dict(cfg["settings"])
    if "routes" in cfg and isinstance(cfg["routes"], Mapping):
        name = route if isinstance(route, str) else getattr(route, "task", getattr(route, "name", ""))
        selected = cfg["routes"].get(str(name))
        if isinstance(selected, Mapping):
            merged = dict(cfg)
            merged.update(selected)
            return merged
    return cfg


def _finite_array(values: Any, length: int, default: float = 0.) -> np.ndarray:
    if values is None:
        return np.full(length, default, dtype=np.float64)
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(out) != length:
        raise ValueError(f"availability has {len(out)} values, expected {length}")
    return np.nan_to_num(out, nan=default, posinf=default, neginf=default)


def _views(stations: np.ndarray, sectors: np.ndarray, indices: np.ndarray) -> set[tuple[int, int]]:
    return {(int(np.rint(stations[i])), int(np.rint(sectors[i])) % 24) for i in np.asarray(indices, dtype=int)}


def _group_candidates(
    public: Mapping[str, Any], p: np.ndarray, e: np.ndarray, graph: Mapping[str, Any], u: np.ndarray,
    *, cutoff: int, settings: Mapping[str, Any], truth: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    """Construct public tracklet proposals from the existing scored retrieval."""
    arrays = _public_arrays(public)
    n = len(arrays["frames"])
    track_ids = np.asarray(graph.get("track_ids", np.arange(n)), dtype=np.int64)
    scored = np.asarray(graph.get("scored_edges", np.empty((0, 3))), dtype=np.float64)
    if scored.size == 0:
        return np.empty((0, len(BRIDGE_FIELDS))), [], {"scored": 0}
    if scored.ndim != 2 or scored.shape[1] < 3:
        raise ValueError("graph.scored_edges must have shape [n, >=3]")
    pairs = np.rint(scored[:, :2]).astype(np.int64)
    scores = np.clip(scored[:, 2], 0., 1.)
    base_threshold = float(graph.get("stats", {}).get("edge_threshold", edge_threshold(settings.get("edge_model"))))
    min_score = float(settings.get("min_candidate_score", .35))
    max_score = float(settings.get("max_candidate_score", base_threshold - 1.e-7))
    min_quality = float(settings.get("min_quality", .10))
    min_usable = float(settings.get("min_usable", .05))
    min_witnesses = int(settings.get("min_witnesses", 2))
    frame = arrays["frames"]
    left, right = pairs[:, 0], pairs[:, 1]
    old_side = frame[left] < int(cutoff)
    new_side = frame[right] >= int(cutoff)
    reverse = (frame[right] < int(cutoff)) & (frame[left] >= int(cutoff))
    cross = old_side & new_side | reverse
    distinct_tracks = track_ids[left] != track_ids[right]
    distinct_views = np.asarray([
        (int(np.rint(arrays["stations"][i])), int(np.rint(arrays["sectors"][i])) % 24) !=
        (int(np.rint(arrays["stations"][j])), int(np.rint(arrays["sectors"][j])) % 24)
        for i, j in zip(left, right, strict=True)
    ], dtype=bool)
    availability = np.clip(np.asarray(u, dtype=np.float64), 0., 1.) * arrays["quality"]
    available = (availability[left] >= min_usable * min_quality) & (availability[right] >= min_usable * min_quality)
    keep = cross & distinct_tracks & distinct_views & (scores >= min_score) & (scores <= max_score) & available
    stats = Counter({"scored": int(len(pairs)), "cross_cutoff": int(np.count_nonzero(cross)),
                     "lower_confidence": int(np.count_nonzero(cross & (scores < base_threshold))),
                     "track_separate": int(np.count_nonzero(cross & distinct_tracks)),
                     "availability": int(np.count_nonzero(cross & available)),
                     "candidate_edges": int(np.count_nonzero(keep))})
    if not np.any(keep):
        return np.empty((0, len(BRIDGE_FIELDS))), [], dict(stats)
    edge_x = edge_inputs(public, p, e, pairs)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pos in np.flatnonzero(keep):
        i, j = int(left[pos]), int(right[pos])
        if frame[i] >= cutoff:
            i, j = j, i
        groups[(int(track_ids[i]), int(track_ids[j]))].append(int(pos))
    features, proposals = [], []
    for (old_track, new_track), positions in sorted(groups.items()):
        if len(positions) < min_witnesses:
            stats["insufficient_witness_groups"] += 1
            continue
        positions = np.asarray(positions, dtype=np.int64)
        pair_rows = pairs[positions]
        # Orient witness endpoints old->new for aggregate measurement features.
        old_end, new_end = [], []
        for i, j in pair_rows:
            if frame[i] < cutoff:
                old_end.append(int(i)); new_end.append(int(j))
            else:
                old_end.append(int(j)); new_end.append(int(i))
        old_end, new_end = np.asarray(old_end, dtype=np.int64), np.asarray(new_end, dtype=np.int64)
        x = edge_x[positions]
        old_views = _views(arrays["stations"], arrays["sectors"], old_end)
        new_views = _views(arrays["stations"], arrays["sectors"], new_end)
        old_frames = set(map(int, frame[old_end].tolist()))
        new_frames = set(map(int, frame[new_end].tolist()))
        world = x[:, 2] * 16.0
        station = x[:, 3] * 3.2
        angle = x[:, 4] * 2.0
        order = np.argsort(-scores[positions], kind="stable")
        top = scores[positions][order[:2]]
        old_member = np.flatnonzero(track_ids == old_track)
        new_member = np.flatnonzero(track_ids == new_track)
        vector = np.asarray([
            float(np.max(scores[positions])), float(np.mean(top)), float(base_threshold - np.max(scores[positions])),
            float(len(positions)), float(len(set(old_end.tolist()))), float(len(set(new_end.tolist()))),
            float(len(old_frames)), float(len(new_frames)), float(len(old_views)), float(len(new_views)),
            float(np.min(world)), float(np.mean(world)), float(np.min(station)), float(np.mean(station)),
            float(np.min(angle)), float(np.max(x[:, 1])), float(np.mean(x[:, 1])),
            float(np.min(x[:, 5])), float(np.mean(x[:, 5])), float(np.max(x[:, 5])),
            float(np.mean(x[:, 6])), float(np.max(x[:, 8])), float(np.mean(x[:, 9])),
            float(np.ptp(arrays["stations"][old_member])) if len(old_member) else 0.,
            float(np.ptp(arrays["stations"][new_member])) if len(new_member) else 0.,
            float(np.min(availability[np.r_[old_end, new_end]])),
        ], dtype=np.float64)
        truth_label = None
        if truth is not None:
            truth_label = bool(np.any((truth[old_end] >= 0) & (truth[old_end] == truth[new_end])))
        features.append(vector)
        proposals.append({"old_track": old_track, "new_track": new_track,
                          "positions": positions.tolist(), "old_endpoints": old_end.tolist(),
                          "new_endpoints": new_end.tolist(), "label": truth_label,
                          "max_base_score": float(np.max(scores[positions])),
                          "witness_count": len(positions)})
    stats["proposals"] = len(proposals)
    return np.asarray(features, dtype=np.float64).reshape(-1, len(BRIDGE_FIELDS)), proposals, dict(stats)


def _fit_logistic(x: np.ndarray, y: np.ndarray, regularization: float = .01) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(BRIDGE_FIELDS) or len(x) != len(y) or len(x) < 2:
        raise ValueError("invalid bridge training arrays")
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .035)
    z = np.c_[(x - center) / scale, np.ones(len(x))]
    penalty = np.r_[np.full(len(BRIDGE_FIELDS), regularization), 0.]

    def objective(w: np.ndarray):
        logits = z @ w
        loss = np.mean(np.logaddexp(0., logits) - y * logits) + .5 * np.sum(penalty * w * w)
        grad = z.T @ (expit(logits) - y) / len(y) + penalty * w
        return float(loss), grad

    result = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"bridge logistic fit failed: {result.message}")
    return {"center": center.tolist(), "scale": scale.tolist(), "weight": result.x.tolist(),
            "features": list(BRIDGE_FIELDS), "train_rows": int(len(y)),
            "train_positive": int(np.count_nonzero(y >= .5)), "regularization": regularization}


def _predict(x: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    center, scale = np.asarray(model["center"]), np.asarray(model["scale"])
    weight = np.asarray(model["weight"])
    return expit(((x - center) / np.maximum(scale, .035)) @ weight[:-1] + weight[-1])


def _select_threshold(pred: np.ndarray, y: np.ndarray, *, precision: float = .94) -> dict[str, Any]:
    pred, y = np.asarray(pred), np.asarray(y).astype(bool)
    positives = int(y.sum())
    options = []
    for threshold in np.unique(np.r_[1., pred, 0.]):
        selected = pred >= threshold
        count = int(selected.sum()); correct = int((selected & y).sum())
        ppv = correct / count if count else 1.
        recall = correct / positives if positives else 0.
        if count and ppv >= precision:
            options.append((recall, -threshold, threshold, count, correct, ppv))
    if not options:
        return {"active": False, "threshold": float("inf"), "precision_target": precision,
                "selected": 0, "correct": 0, "precision": 1., "recall": 0., "positives": positives}
    _, _, threshold, count, correct, ppv = max(options)
    return {"active": True, "threshold": float(threshold), "precision_target": precision,
            "selected": int(count), "correct": int(correct), "precision": float(ppv),
            "recall": float(correct / positives) if positives else 0., "positives": positives}


def _candidate_edge_record(public: Mapping[str, Any], graph: Mapping[str, Any], bridge_rows: list[tuple[int, int, float]],
                           links: Any, p: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply one-to-one and complete-span guards to bridge rows."""
    arrays = _public_arrays(public)
    base = np.asarray(graph.get("edges", np.empty((0, 3))), dtype=np.float64)
    forest = _Forest(arrays["frames"], arrays["stations"], links)
    for row in base:
        if not forest.try_union(int(row[0]), int(row[1])):
            # Base graph was already checked by v5.  Keep a diagnostic error
            # rather than silently modifying its matching.
            raise RuntimeError("base graph is not reproducible under _Forest")
    kept, rejected = [], Counter()
    for left, right, score in sorted(bridge_rows, key=lambda row: (-row[2], row[0], row[1])):
        root_left, root_right = forest.find(left), forest.find(right)
        if root_left == root_right:
            rejected["connected_redundant"] += 1; continue
        if forest.frames[root_left].intersection(forest.frames[root_right]):
            rejected["same_frame"] += 1; continue
        low = min(forest.min_station[root_left], forest.min_station[root_right])
        high = max(forest.max_station[root_left], forest.max_station[root_right])
        mask = _span_mask(forest.links, low, high) if forest.links is not None else None
        if mask is not None and not np.any(mask):
            rejected["topology"] += 1; continue
        if forest.try_union(left, right):
            kept.append((left, right, score))
        else:
            rejected["matching_other"] += 1
    return (np.asarray(kept, dtype=np.float64).reshape(-1, 3) if kept else np.empty((0, 3), dtype=np.float64),
            [{"reason": key, "count": int(value)} for key, value in sorted(rejected.items())])


def recover(public, p, e, links, u, graph, *, cutoff, settings=None, route=None):
    """Add calibrated, public-only old/new tracklet bridges to a v5 graph.

    ``settings`` is the JSON object emitted by :func:`fit_and_diagnose` or its
    ``model`` field.  A missing/inactive model is an intentional no-op.  The
    base ``graph['scored_edges']`` table is never recomputed, and all base
    accepted edges are preserved verbatim.
    """
    cfg = _cfg_for_route(settings, route)
    base_edges = np.asarray(graph.get("edges", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    base_scored = np.asarray(graph.get("scored_edges", np.empty((0, 3))), dtype=np.float64).reshape(-1, 3)
    out = dict(graph)
    out["edges"] = base_edges.copy()
    out["scored_edges"] = base_scored.copy()
    out["bridge_edges"] = np.empty((0, 3), dtype=np.float32)
    stats = dict(graph.get("stats", {}))
    stats.update({"bridge_active": False, "bridge_candidates": 0, "bridge_proposals": 0,
                  "bridge_selected": 0, "bridge_rejected": {}, "bridge_reason": "inactive"})
    out["stats"] = stats
    if not cfg.get("active", False) or not isinstance(cfg.get("model"), Mapping):
        return out
    arrays = _public_arrays(public); p = np.asarray(p, dtype=np.float64); e = np.asarray(e, dtype=np.float64)
    usability = _finite_array(u, len(arrays["frames"]))
    features, proposals, candidate_stats = _group_candidates(public, p, e, graph, usability,
                                                              cutoff=int(cutoff), settings=cfg)
    stats.update({"bridge_active": True, "bridge_reason": "calibrated_model", "bridge_candidates": candidate_stats.get("candidate_edges", 0),
                  "bridge_proposals": len(proposals), "bridge_candidate_stats": candidate_stats})
    if not len(proposals):
        out["stats"] = stats
        return out
    scores = _predict(features, cfg["model"])
    threshold = float(cfg.get("threshold", float("inf")))
    selected = []
    for score, proposal in zip(scores, proposals, strict=True):
        if float(score) < threshold:
            continue
        positions = np.asarray(proposal["positions"], dtype=np.int64)
        # The bridge does not invent a new edge; it chooses the strongest
        # already-retrieved witness in this public tracklet proposal.
        scored = np.asarray(graph["scored_edges"])[positions]
        winner = int(positions[int(np.argmax(scored[:, 2]))])
        selected.append((int(graph["scored_edges"][winner, 0]), int(graph["scored_edges"][winner, 1]), float(score)))
    bridge, rejected = _candidate_edge_record(public, graph, selected, links, p)
    stats.update({"bridge_selected": int(len(bridge)), "bridge_rejected": rejected,
                  "bridge_threshold": threshold, "bridge_proposals_scored": int(len(selected))})
    out["bridge_edges"] = bridge.astype(np.float32)
    if len(bridge):
        combined = np.vstack((base_edges, bridge))
        revised, track_ids, support, tracks, hypotheses = _track_outputs(arrays, p, combined, links)
        out.update({"edges": combined.astype(np.float32), "probabilities": revised.astype(np.float32),
                    "track_ids": track_ids.astype(np.int32), "support": support.astype(np.float32),
                    "tracks": tracks, "hypotheses": hypotheses})
    out["stats"] = stats
    return out


def _snapshot_rows(path: Path, bundle: Mapping[str, Any], *, collect_labels: bool) -> tuple[np.ndarray, np.ndarray, Counter]:
    """Extract proposal features from one frozen v5 shard."""
    _, roster = load_shard(path.parent)
    rows, labels, counts = [], [], Counter()
    fields = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality", "labels", "instance", "offsets")
    with np.load(path, allow_pickle=False) as z:
        columns = {k: z[k] for k in fields}
        descriptions = [(row, h) for row in roster for h in row["histories"]]
        for (lo, hi), (row, desc) in zip(pairwise(columns["offsets"]), descriptions, strict=True):
            if desc["scope"] != "after_probe":
                continue
            public = inputs({k: columns[k][lo:hi] for k in ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality")})
            p, e = heads(public["features"], bundle["model"])
            graph = infer(public, p, e, bundle["links"], edge_model=bundle["edge_model"])
            u = predict_usability(public, bundle["usability"])
            # Fit and calibration use exactly the runtime proposal eligibility
            # rules.  In particular, a proposal is not labelled positive just
            # because one mixed witness is true: recover() executes the single
            # highest-base-score witness, so that exact witness supplies the
            # supervised label below.
            x, proposals, stat = _group_candidates(
                public, p, e, graph, u, cutoff=int(desc["first_frame_count"]),
                settings={"edge_model": bundle["edge_model"], "min_witnesses": 2,
                          "min_candidate_score": .35, "max_candidate_score": .8999999,
                          "min_usable": .05, "min_quality": .10})
            counts.update(stat); counts["layouts"] += 1
            if len(x):
                rows.append(x)
                if collect_labels:
                    instance = columns["instance"][lo:hi]
                    scored_edges = np.asarray(graph["scored_edges"], dtype=np.float64)
                    selected_labels = []
                    for proposal in proposals:
                        positions = np.asarray(proposal["positions"], dtype=np.int64)
                        winner = int(positions[int(np.argmax(scored_edges[positions, 2]))])
                        left, right = (int(v) for v in scored_edges[winner, :2])
                        selected_labels.append(float(instance[left] >= 0 and instance[left] == instance[right]))
                    labels.append(np.asarray(selected_labels, dtype=np.float64))
    if not rows:
        return np.empty((0, len(BRIDGE_FIELDS))), np.empty(0), counts
    return np.vstack(rows), np.concatenate(labels) if labels else np.empty(0), counts


def _init_worker(bundle: Mapping[str, Any]) -> None:
    global _BUNDLE
    _BUNDLE = dict(bundle)


def _worker_extract(job: tuple[Path, bool]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    if _BUNDLE is None:
        raise RuntimeError("bridge worker has no frozen bundle")
    x, y, counts = _snapshot_rows(job[0], _BUNDLE, collect_labels=job[1])
    return x, y, dict(counts)


def _extract(paths: list[Path], bundle: Mapping[str, Any], *, labels: bool, workers: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(paths)), mp_context=ctx,
                             initializer=_init_worker, initargs=(bundle,)) as pool:
        outputs = list(pool.map(_worker_extract, [(p, labels) for p in paths], chunksize=1))
    xs = [x for x, _, _ in outputs if len(x)]
    ys = [y for _, y, _ in outputs if len(y)]
    total = Counter()
    for _, _, count in outputs:
        total.update(count)
    return (np.vstack(xs) if xs else np.empty((0, len(BRIDGE_FIELDS))),
            np.concatenate(ys) if ys else np.empty(0), dict(total))


def _diagnostic_case(path: Path, bundle: Mapping[str, Any], settings: Mapping[str, Any], old_rows: Mapping[str, Any]) -> list[dict[str, Any]]:
    _, roster = load_shard(path.parent)
    output = []
    fields = ("features", "frames", "slots", "regions", "stations", "sectors", "points", "quality", "labels", "instance", "offsets")
    with np.load(path, allow_pickle=False) as z:
        columns = {k: z[k] for k in fields}
        descriptions = [(row, h) for row in roster for h in row["histories"]]
        per: dict[str, dict[str, np.ndarray]] = {}; current = None; current_row = None
        for (lo, hi), (row, desc) in zip(pairwise(columns["offsets"]), descriptions, strict=True):
            case_id = row["case"]["identifier"]
            if current is not None and case_id != current:
                if current_row and current_row["case"]["task"] == "groin_to_distal_calf" and "before" in per and "after_probe" in per:
                    output.append(_diagnostic_one(current_row, per, bundle, settings, old_rows.get(current)))
                per = {}
            current, current_row = case_id, row
            if desc["scope"] in ("before", "after_probe"):
                per[desc["scope"]] = {k: columns[k][lo:hi] for k in fields[:-1]}
        if current_row and current_row["case"]["task"] == "groin_to_distal_calf" and "before" in per and "after_probe" in per:
            output.append(_diagnostic_one(current_row, per, bundle, settings, old_rows.get(current)))
    return output


def _diagnostic_one(row: Mapping[str, Any], snapshots: Mapping[str, Any], bundle: Mapping[str, Any], settings: Mapping[str, Any], old_row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row["case"]["task"] != "groin_to_distal_calf":
        return {"case": row["case"]["identifier"], "route": row["case"]["task"], "excluded": True}
    after = inputs(snapshots["after_probe"])
    p, e = heads(after["features"], bundle["model"])
    graph = infer(after, p, e, bundle["links"], edge_model=bundle["edge_model"])
    u = predict_usability(after, bundle["usability"])
    cutoff = int(next(h["first_frame_count"] for h in row["histories"] if h["scope"] == "after_probe"))
    bridged = recover(after, p, e, bundle["links"], u, graph, cutoff=cutoff, settings=settings, route=row["case"]["task"])
    instance = snapshots["after_probe"]["instance"]
    old_pair = None if old_row is None else old_row.get("old_candidate")
    old_idx = -1
    if old_pair:
        ix = np.flatnonzero((snapshots["before"]["frames"] == old_pair[0]) & (snapshots["before"]["slots"] == old_pair[1]))
        if len(ix) == 1:
            old_idx = int(np.flatnonzero((after["frames"] == old_pair[0]) & (after["slots"] == old_pair[1]))[0])
    if old_idx < 0:
        return {"case": row["case"]["identifier"], "route": row["case"]["task"], "category": "no_witness",
                "bridge_edges": int(len(bridged["bridge_edges"]))}
    old_instance = int(instance[old_idx]); fresh = np.flatnonzero((after["frames"] >= cutoff) & (instance == old_instance) & (old_instance >= 0))
    base_tracks = np.asarray(graph.get("track_ids", np.arange(len(after["frames"]))), dtype=int)
    new_tracks = np.asarray(bridged.get("track_ids", base_tracks), dtype=int)
    base_conn = bool(len(fresh) and np.any(base_tracks[fresh] == base_tracks[old_idx]))
    bridge_conn = bool(len(fresh) and np.any(new_tracks[fresh] == new_tracks[old_idx]))
    bridge = np.asarray(bridged.get("bridge_edges", np.empty((0, 3))), dtype=float)
    true_bridge = int(sum(instance[int(i)] >= 0 and instance[int(i)] == instance[int(j)] for i, j in bridge[:, :2].astype(int))) if len(bridge) else 0
    return {"case": row["case"]["identifier"], "layout_id": row["layout_id"], "route": row["case"]["task"],
            "old_truth_role": int(snapshots["before"]["labels"][old_idx]) if old_idx < len(snapshots["before"]["labels"]) else None,
            "same_instance_seen": bool(len(fresh)), "base_connected": base_conn, "bridge_connected": bridge_conn,
            "newly_connected": bool(bridge_conn and not base_conn), "bridge_edges": int(len(bridge)),
            "bridge_true_edges": true_bridge, "bridge_false_edges": int(len(bridge) - true_bridge),
            "stats": bridged["stats"]}


def fit_and_diagnose(out: Path, *, atlas_pack: Path = DEFAULT_PACK, workers: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    bundle = warm_bundle(V6_RUN, atlas_pack.resolve())
    fit_paths = sorted((V5_RUN / "fit_revisit").glob("*/data.npz"))
    cal_paths = sorted((V5_RUN / "reader_test").glob("*/data.npz"))
    if not fit_paths or not cal_paths:
        raise FileNotFoundError("frozen v5 fit_revisit and reader_test shards are required")
    x_fit, y_fit, fit_stats = _extract(fit_paths, bundle, labels=True, workers=workers)
    x_cal, y_cal, cal_stats = _extract(cal_paths, bundle, labels=True, workers=workers)
    # Keep the fit target honest: all labels are proposal-level positives, not
    # hidden simulator inputs at runtime.
    model = _fit_logistic(x_fit, y_fit)
    threshold = _select_threshold(_predict(x_cal, model), y_cal, precision=.94)
    settings = {"version": 1, "active": bool(threshold["active"]), "model": model,
                "threshold": threshold["threshold"], "calibration": threshold,
                "label_semantics": "physical same-instance label of the highest-base-score witness executed by recover",
                "min_candidate_score": .35, "max_candidate_score": .8999999,
                "min_quality": .10, "min_usable": .05, "min_witnesses": 2,
                "fit_shards": [str(p) for p in fit_paths], "calibration_shards": [str(p) for p in cal_paths],
                "fit_stats": fit_stats, "calibration_stats": cal_stats}
    old_audit = read_json(V6_RUN / "link_evidence_audit.json")
    old_rows = {r["case"]: r for r in old_audit["per_layout"] if r["route"] == "groin_to_distal_calf"}
    v6_paths = sorted((V6_RUN / "reader_fresh").glob("*/data.npz"))
    ctx = mp.get_context("spawn")
    # Diagnostic workers receive only the immutable frozen bundle/settings;
    # v6 layouts are never passed to fit/calibration extraction.
    with ProcessPoolExecutor(max_workers=min(workers, len(v6_paths)), mp_context=ctx,
                             initializer=_init_worker, initargs=(bundle,)) as pool:
        # _diagnostic_case has explicit args; submit avoids changing the
        # process-wide fit globals used by the extractor.
        futures = [pool.submit(_diagnostic_case, p, bundle, settings, old_rows) for p in v6_paths]
        diagnostic_rows = [row for future in futures for row in future.result()]
    diagnostic_rows.sort(key=lambda r: r["case"])
    target = [r for r in diagnostic_rows if r.get("old_truth_role") == 0]
    bridge_summary = {
        "cases": len(diagnostic_rows), "target_cases": len(target),
        "same_instance_seen": int(sum(r.get("same_instance_seen", False) for r in target)),
        "base_connected": int(sum(r.get("base_connected", False) for r in target)),
        "bridge_connected": int(sum(r.get("bridge_connected", False) for r in target)),
        "newly_connected": int(sum(r.get("newly_connected", False) for r in target)),
        "bridge_edges": int(sum(r.get("bridge_edges", 0) for r in target)),
        "bridge_true_edges": int(sum(r.get("bridge_true_edges", 0) for r in target)),
        "bridge_false_edges": int(sum(r.get("bridge_false_edges", 0) for r in target)),
    }
    model_output = {"experiment": ROOT.name, "scope": "targeted lower-confidence old/new tracklet reacquisition",
                    "base_v5_reader_sha256": digest(V6_RUN / "reader_freeze.json"),
                    "fit_input_sha256": [digest(p) for p in fit_paths],
                    "calibration_input_sha256": [digest(p) for p in cal_paths], "settings": settings}
    diagnostic_output = {"experiment": ROOT.name, "scope": "v6 reader_fresh whole-route after_probe held-out diagnostic",
                         "fit_excluded_v6": True, "summary": bridge_summary, "per_case": diagnostic_rows}
    atomic_json(out / "bridge_model.json", model_output)
    atomic_json(out / "bridge_diagnostic.json", diagnostic_output)
    return model_output, diagnostic_output


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    model, diagnostic = fit_and_diagnose(args.out.resolve(), atlas_pack=args.atlas_pack.resolve(), workers=args.workers)
    print({"active": model["settings"]["active"], "calibration": model["settings"]["calibration"],
           "diagnostic": diagnostic["summary"]}, flush=True)


if __name__ == "__main__":
    main()


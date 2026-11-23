"""Stage the v6 public-association failure modes on the frozen 96-case route.

This is an offline diagnostic only.  It replays the public v5 matcher in the
same score order used by deployment, while truth is used only to label which
retrieved pairs really belong to the fixed old candidate's physical instance.
No truth label is passed to the deployed reader.  The staged graph counts are
intended to separate retrieval misses, edge-score misses, matching/topology
conflicts, and failures of the public graph to connect an observed revisit.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
import heapq
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent
V6 = HANDOFF / "leg_scan_identity_state_20261123_v6"
V5 = HANDOFF / "leg_scan_anchor_state_20261123_v5"
V6_RUN = V6 / "run" / "confirmed"
sys.path.insert(0, str(V6))
sys.path.insert(0, str(V5))

from causal_reader import PUBLIC_FIELDS, inputs, model_state  # noqa: E402
from object_tracks import (  # noqa: E402
    _Forest,
    _public_arrays,
    _span_mask,
    edge_inputs,
    edge_probabilities,
    edge_threshold,
    retrieve,
)
from readout_state import heads  # noqa: E402
from reader_study import warm_bundle  # noqa: E402
from v6_data import DEFAULT_PACK, atomic_json, digest, load_shard, read_json  # noqa: E402


_BUNDLE: dict[str, Any] | None = None
_OLD_ROWS: dict[str, dict[str, Any]] = {}


def _init(bundle: dict[str, Any], old_rows: dict[str, dict[str, Any]]) -> None:
    global _BUNDLE, _OLD_ROWS
    _BUNDLE = bundle
    _OLD_ROWS = old_rows


def _uf_components(n: int, edges: np.ndarray) -> np.ndarray:
    """Return simple graph components without frame/topology constraints."""
    parent = np.arange(n, dtype=np.int64)

    def find(item: int) -> int:
        root = int(item)
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            old = int(parent[item])
            parent[item] = root
            item = old
        return root

    for row in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left, right = (int(row[0]), int(row[1]))
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a
    return np.asarray([find(i) for i in range(n)], dtype=np.int64)


def _reachable(old: int, fresh: np.ndarray, edges: np.ndarray) -> tuple[bool, list[int]]:
    """Check old-to-fresh graph connectivity and return reachable witnesses."""
    fresh = np.asarray(fresh, dtype=np.int64)
    if old < 0 or not len(fresh):
        return False, []
    adjacency: dict[int, list[int]] = defaultdict(list)
    for row in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left, right = int(row[0]), int(row[1])
        adjacency[left].append(right)
        adjacency[right].append(left)
    target = set(map(int, fresh.tolist()))
    found: list[int] = []
    seen = {int(old)}
    queue = deque([int(old)])
    while queue:
        node = queue.popleft()
        if node in target:
            found.append(node)
        for other in adjacency.get(node, ()):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return bool(found), sorted(found)


def _make_adjacency(n: int, edges: np.ndarray) -> list[list[tuple[int, float]]]:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for row in np.asarray(edges, dtype=np.float64).reshape(-1, 3):
        left, right, p = int(row[0]), int(row[1]), float(row[2])
        adjacency[left].append((right, p))
        adjacency[right].append((left, p))
    return adjacency


def _greedy_trace(
    public: Mapping[str, Any],
    pairs: np.ndarray,
    probabilities: np.ndarray,
    links: Any,
    threshold: float,
    *,
    truth_instance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay object_tracks._infer_impl and retain a reason per edge.

    If ``truth_instance`` is supplied, only pairs with the same nonnegative
    instance are eligible.  Learned edge probabilities and their deployment
    ordering remain unchanged; this is the truth-only diagnostic stage, not a
    deployment path.
    """
    arrays = _public_arrays(public)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(pairs) != len(probabilities):
        raise ValueError("pair/probability length mismatch")
    labels = None if truth_instance is None else np.asarray(truth_instance, dtype=np.int64)
    order = np.lexsort((pairs[:, 1], pairs[:, 0], -probabilities)) if len(pairs) else np.empty(0, dtype=np.int64)
    forest = _Forest(arrays["frames"], arrays["stations"], links)
    retained: list[tuple[int, int, float]] = []
    reasons: dict[tuple[int, int], str] = {}
    counts = Counter()
    for position in order:
        position = int(position)
        left, right = (int(v) for v in pairs[position])
        key = (left, right)
        if labels is not None:
            same = labels[left] >= 0 and labels[left] == labels[right]
            if not same:
                reasons[key] = "not_truth_positive"
                counts["not_truth_positive"] += 1
                continue
        if probabilities[position] < threshold:
            reasons[key] = "below_threshold"
            counts["below_threshold"] += 1
            continue
        root_left, root_right = forest.find(left), forest.find(right)
        if root_left == root_right:
            reasons[key] = "connected_redundant"
            counts["connected_redundant"] += 1
            continue
        if forest.frames[root_left].intersection(forest.frames[root_right]):
            reasons[key] = "same_frame"
            counts["same_frame"] += 1
            continue
        low = min(forest.min_station[root_left], forest.min_station[root_right])
        high = max(forest.max_station[root_left], forest.max_station[root_right])
        mask = _span_mask(forest.links, low, high) if forest.links is not None else None
        if mask is not None and not np.any(mask):
            reasons[key] = "topology"
            counts["topology"] += 1
            continue
        if forest.try_union(left, right):
            reasons[key] = "kept"
            retained.append((left, right, float(probabilities[position])))
            counts["kept"] += 1
        else:
            reasons[key] = "matching_other"
            counts["matching_other"] += 1
    kept = np.asarray(retained, dtype=np.int64).reshape(-1, 3) if retained else np.empty((0, 3), dtype=np.float64)
    return {"kept": kept, "reasons": reasons, "counts": dict(counts), "order": order.tolist()}


def _path_weights_from_source(source: int, n: int, edges: np.ndarray, max_hops: int = 32) -> dict[int, float]:
    """Maximum product path reliability from one node, bounded as deployed."""
    adjacency = _make_adjacency(n, edges)
    # v6 role_fusion uses a max-product heap with a 32-hop bound.  Keep the
    # source explicit here so the audit can report old-to-fresh as well as the
    # deployed anchor-to-fresh reliability.
    best = {int(source): 1.0}
    queue: list[tuple[float, int, int]] = [(-1.0, int(source), 0)]
    while queue:
        neg_score, node, hops = heapq.heappop(queue)
        score = -neg_score
        if score + 1.0e-6 < best.get(node, 0.) or hops >= max_hops:
            continue
        for other, probability in adjacency[node]:
            candidate = score * float(np.clip(probability, 0., 1.))
            if candidate > best.get(other, 0.) + 1.0e-6:
                best[other] = candidate
                heapq.heappush(queue, (-candidate, other, hops + 1))
    return best


def _v6_effective_weights(public: Mapping[str, Any], state: Mapping[str, Any], usability: np.ndarray,
                          old_idx: int, fresh_indices: np.ndarray) -> dict[str, Any]:
    """Recover the v6 anchor/path and representative evidence weights."""
    n = len(public["frames"])
    edges = np.asarray(state.get("edges", np.empty((0, 3))), dtype=np.float64)
    track_ids = np.asarray(state.get("track_ids", np.arange(n)), dtype=np.int64)
    if old_idx < 0 or old_idx >= n:
        return {}
    track = int(track_ids[old_idx])
    members = np.flatnonzero(track_ids == track).astype(np.int64)
    if not len(members):
        return {}
    adjacency = _make_adjacency(n, edges)
    anchor = int(np.min(members))
    anchor_paths = _path_weights_from_source(anchor, n, edges)
    old_paths = _path_weights_from_source(old_idx, n, edges)
    views: dict[tuple[int, int], int] = {}
    for index in sorted(members.tolist(), key=lambda i: (-float(public["quality"][i]), i)):
        key = (int(np.rint(public["stations"][index])), int(np.rint(public["sectors"][index])) % 24)
        views.setdefault(key, int(index))
    rep_weights = {
        int(index): float(anchor_paths.get(int(index), 0.)
                          * max(float(usability[int(index)]), 0.)
                          * max(float(public["quality"][int(index)]), 0.))
        for index in views.values()
    }
    fresh_info = []
    for index in np.asarray(fresh_indices, dtype=np.int64):
        fresh_info.append({
            "index": int(index),
            "anchor_path_weight": float(anchor_paths.get(int(index), 0.)),
            "old_path_weight": float(old_paths.get(int(index), 0.)),
            "is_representative": bool(int(index) in rep_weights),
            "representative_effective_weight": float(rep_weights.get(int(index), 0.)),
            "quality": float(public["quality"][int(index)]),
            "usable": float(usability[int(index)]),
        })
    return {"track_id": track, "anchor_index": anchor, "member_count": int(len(members)),
            "representative_count": int(len(rep_weights)), "fresh": fresh_info}


def _pair_record(
    old_idx: int,
    fresh_idx: int,
    instance: np.ndarray,
    pairs: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    true_trace: Mapping[str, Any],
    public_trace: Mapping[str, Any],
    old_role_p: float,
    old_unary_p: float,
    public_state: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> dict[str, Any]:
    key = (min(int(old_idx), int(fresh_idx)), max(int(old_idx), int(fresh_idx)))
    positions = np.flatnonzero(np.all(pairs == np.asarray(key, dtype=np.int64), axis=1)) if len(pairs) else np.empty(0, dtype=int)
    retrieved = bool(len(positions))
    score = None if not retrieved else float(probabilities[int(positions[0])])
    fresh_effective = next((row for row in effective.get("fresh", []) if row["index"] == int(fresh_idx)), {})
    return {
        "old_index": int(old_idx), "fresh_index": int(fresh_idx),
        "instance": int(instance[fresh_idx]), "retrieved": retrieved,
        "edge_score": score, "above_threshold": bool(retrieved and score >= threshold),
        "threshold": float(threshold),
        "true_only_reason": true_trace.get("reasons", {}).get(key, "not_retrieved"),
        "public_reason": public_trace.get("reasons", {}).get(key, "not_retrieved"),
        "public_same_track": bool(public_state.get("track_ids", np.array([-1]))[old_idx] ==
                                   public_state.get("track_ids", np.array([-2]))[fresh_idx]),
        "old_unary_target_p": float(old_unary_p), "old_prior_role_target_p": float(old_role_p),
        "effective": fresh_effective,
    }


def _case_snapshot(row: Mapping[str, Any], arrays: Mapping[str, np.ndarray], bundle: Mapping[str, Any],
                   old_row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Analyze one whole-route case from before and after_probe snapshots."""
    case = row["case"]
    if case["task"] != "groin_to_distal_calf":
        return None
    descs = {h["scope"]: h for h in row["histories"]}
    if "before" not in descs or "after_probe" not in descs:
        return None
    before_desc, after_desc = descs["before"], descs["after_probe"]
    # The caller has already sliced each history; this function receives a
    # dictionary keyed by scope to keep the case logic explicit.
    before = arrays["before"]
    after = arrays["after_probe"]
    assert _BUNDLE is not None
    before_state = model_state(before, _BUNDLE["model"], _BUNDLE["links"], _BUNDLE["edge_model"],
                               _BUNDLE["usability"])
    old_pair = None if old_row is None else tuple(old_row.get("old_candidate", ()))
    old_idx = -1
    if len(old_pair) == 2:
        matches = np.flatnonzero((before["frames"] == int(old_pair[0])) & (before["slots"] == int(old_pair[1])))
        if len(matches) == 1:
            old_idx = int(matches[0])
    old_truth_role = int(before["labels"][old_idx]) if old_idx >= 0 else None
    old_instance = int(before["instance"][old_idx]) if old_idx >= 0 else -1
    old_unary_p = float(before_state["unary"]["probabilities"][old_idx, 0]) if old_idx >= 0 else None
    old_role_p = float(before_state["objects"]["object_role"][old_idx, 0]) if old_idx >= 0 else None
    category = "no_witness" if old_idx < 0 else (
        "old_target" if old_truth_role == 0 else
        "old_unknown" if old_truth_role is None or old_truth_role < 0 else "old_wrong")
    if old_idx < 0 or old_instance < 0:
        return {"case": case["identifier"], "layout_id": row["layout_id"], "route": case["task"],
                "category": category, "old_candidate": list(old_pair) if old_pair else None,
                "old_truth_role": old_truth_role, "old_instance": old_instance,
                "same_instance_seen": False, "stages": {}, "positive_pairs": [],
                "old_unary_target_p": old_unary_p, "old_prior_role_target_p": old_role_p}

    # Rebuild the same causal prefix on the archived after_probe snapshot.
    physical = after["instance"]
    after_state = model_state(after, _BUNDLE["model"], _BUNDLE["links"], _BUNDLE["edge_model"],
                              _BUNDLE["usability"], physical=physical)
    p, embedding = heads(after["features"].astype(np.float16, copy=False), _BUNDLE["model"])
    pairs = retrieve(after)
    edge_x = edge_inputs(after, p, embedding, pairs)
    probs = edge_probabilities(edge_x, _BUNDLE["edge_model"])
    threshold = edge_threshold(_BUNDLE["edge_model"])
    true_positive = (physical[pairs[:, 0]] >= 0) & (physical[pairs[:, 0]] == physical[pairs[:, 1]]) if len(pairs) else np.empty(0, dtype=bool)
    true_pairs = pairs[true_positive]
    retrieved_true_components = _uf_components(len(after["frames"]), true_pairs)
    threshold_pairs = pairs[true_positive & (probs >= threshold)] if len(pairs) else np.empty((0, 2), dtype=np.int64)
    threshold_components = _uf_components(len(after["frames"]), threshold_pairs)
    true_trace = _greedy_trace(after, pairs, probs, _BUNDLE["links"], threshold,
                               truth_instance=physical)
    public_trace = _greedy_trace(after, pairs, probs, _BUNDLE["links"], threshold)
    true_greedy = true_trace["kept"][:, :2].astype(np.int64) if len(true_trace["kept"]) else np.empty((0, 2), dtype=np.int64)
    public_greedy = public_trace["kept"][:, :2].astype(np.int64) if len(public_trace["kept"]) else np.empty((0, 2), dtype=np.int64)
    new_mask = (after["frames"] >= int(after_desc["first_frame_count"]))
    fresh = np.flatnonzero(new_mask & (physical == old_instance)).astype(np.int64)
    same_seen = bool(len(fresh))
    old_component = int(retrieved_true_components[old_idx])
    old_threshold_component = int(threshold_components[old_idx])
    stages = {
        "retrieved_true_path": bool(same_seen and any(retrieved_true_components[int(i)] == old_component for i in fresh)),
        "thresholded_true_path": bool(same_seen and any(threshold_components[int(i)] == old_threshold_component for i in fresh)),
    }
    stages["true_only_greedy_path"], true_witnesses = _reachable(old_idx, fresh, true_greedy)
    stages["public_greedy_path"], public_witnesses = _reachable(old_idx, fresh, public_greedy)
    # The state produced by model_state is the exact public deployment graph.
    public_track_ids = np.asarray(after_state["objects"]["track_ids"], dtype=np.int64)
    stages["public_state_track_path"] = bool(same_seen and np.any(public_track_ids[fresh] == public_track_ids[old_idx]))
    oracle_track_ids = np.asarray(after_state["object_oracle"]["track_ids"], dtype=np.int64)
    stages["oracle_track_path"] = bool(same_seen and np.any(oracle_track_ids[fresh] == oracle_track_ids[old_idx]))
    usability = np.asarray(after_state["objects"]["usable"], dtype=np.float64)
    effective = _v6_effective_weights(after, after_state["objects"], usability, old_idx, fresh)
    positive_pairs = [_pair_record(old_idx, int(i), physical, pairs, probs, threshold, true_trace, public_trace,
                                   old_role_p, old_unary_p, after_state["objects"], effective) for i in fresh]
    # Direct retrieved true edges, and stage rejection counts, are useful for
    # diagnosing whether a failed path is a retrieval or matching problem.
    direct_retrieved = int(sum(item["retrieved"] for item in positive_pairs))
    direct_above = int(sum(item["above_threshold"] for item in positive_pairs))
    reason_counts = Counter(item["true_only_reason"] for item in positive_pairs)
    public_reason_counts = Counter(item["public_reason"] for item in positive_pairs)
    return {
        "case": case["identifier"], "layout_id": row["layout_id"], "route": case["task"],
        "category": category, "old_candidate": list(old_pair) if old_pair else None,
        "old_index": int(old_idx), "old_truth_role": old_truth_role, "old_instance": old_instance,
        "old_unary_target_p": old_unary_p, "old_prior_role_target_p": old_role_p,
        "fresh_same_instance_count": int(len(fresh)), "same_instance_seen": same_seen,
        "stages": stages, "retrieved_true_direct_pairs": direct_retrieved,
        "thresholded_true_direct_pairs": direct_above,
        "true_only_reason_counts": dict(reason_counts), "public_reason_counts": dict(public_reason_counts),
        "true_only_greedy_counts": true_trace["counts"], "public_greedy_counts": public_trace["counts"],
        "public_stats": after_state["objects"].get("stats", {}),
        "oracle_stats": after_state["object_oracle"].get("stats", {}),
        "public_track_path_witnesses": public_witnesses,
        "oracle_track_path_witnesses": [int(i) for i in fresh if oracle_track_ids[int(i)] == oracle_track_ids[old_idx]],
        "effective_path": effective,
        "positive_pairs": positive_pairs,
    }


def _one_shard(path: Path) -> list[dict[str, Any]]:
    assert _BUNDLE is not None
    _, roster = load_shard(path.parent)
    with np.load(path, allow_pickle=False) as z:
        columns = {k: z[k] for k in (*PUBLIC_FIELDS, "labels", "instance", "offsets")}
        outputs: list[dict[str, Any]] = []
        descriptions = [(row, h) for row in roster for h in row["histories"]]
        slices: dict[str, dict[str, np.ndarray]] = {}
        current_row: Mapping[str, Any] | None = None
        current_case = None
        for (lo, hi), (row, desc) in zip(pairwise(columns["offsets"]), descriptions, strict=True):
            case_id = row["case"]["identifier"]
            if case_id != current_case:
                if current_row is not None:
                    old_row = _OLD_ROWS.get(current_row["case"]["identifier"])
                    analyzed = _case_snapshot(current_row, {"before": slices["before"], "after_probe": slices["after_probe"]}, _BUNDLE, old_row)
                    if analyzed is not None:
                        outputs.append(analyzed)
                current_case, current_row, slices = case_id, row, {}
            if desc["scope"] in ("before", "after_probe"):
                slices[desc["scope"]] = {k: columns[k][lo:hi] for k in (*PUBLIC_FIELDS, "labels", "instance")}
        if current_row is not None and "before" in slices and "after_probe" in slices:
            old_row = _OLD_ROWS.get(current_row["case"]["identifier"])
            analyzed = _case_snapshot(current_row, slices, _BUNDLE, old_row)
            if analyzed is not None:
                outputs.append(analyzed)
    return outputs


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: r["case"])
    summary: dict[str, Any] = {"cases": len(rows), "categories": dict(Counter(r["category"] for r in rows))}
    target = [r for r in rows if r["category"] == "old_target"]
    summary["old_target_cases"] = len(target)
    summary["same_instance_seen"] = int(sum(r["same_instance_seen"] for r in target))
    for key in ("retrieved_true_path", "thresholded_true_path", "true_only_greedy_path",
                "public_greedy_path", "public_state_track_path", "oracle_track_path"):
        summary[key] = int(sum(bool(r["stages"].get(key, False)) for r in target))
    summary["oracle_only_public_missed"] = int(sum(
        r["stages"].get("oracle_track_path", False) and not r["stages"].get("public_state_track_path", False)
        for r in target))
    summary["oracle_not_connected"] = int(sum(
        r["same_instance_seen"] and not r["stages"].get("oracle_track_path", False) for r in target))
    summary["direct_positive_pair_counts"] = {
        key: int(sum(r["retrieved_true_direct_pairs"] if key == "retrieved" else r["thresholded_true_direct_pairs"]
                     for r in target))
        for key in ("retrieved", "thresholded")
    }
    reason = Counter()
    public_reason = Counter()
    true_greedy = Counter()
    public_greedy = Counter()
    for r in target:
        reason.update(r["true_only_reason_counts"])
        public_reason.update(r["public_reason_counts"])
        true_greedy.update(r["true_only_greedy_counts"])
        public_greedy.update(r["public_greedy_counts"])
    summary["true_positive_pair_reasons"] = dict(reason)
    summary["public_positive_pair_reasons"] = dict(public_reason)
    summary["greedy_edge_reasons_target_cases"] = {
        "true_only": dict(true_greedy), "public": dict(public_greedy),
    }
    # Explicitly report the expected v6 split, but derive it from rows rather
    # than copying the prior audit's headline numbers.
    summary["reproduced_split"] = {
        "same_seen": summary["same_instance_seen"],
        "public_connected": summary["public_state_track_path"],
        "oracle_connected": summary["oracle_track_path"],
        "oracle_only": summary["oracle_only_public_missed"],
        "oracle_not_connected": summary["oracle_not_connected"],
    }
    return summary


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run" / "association_audit.json")
    parser.add_argument("--v6-run", type=Path, default=V6_RUN)
    parser.add_argument("--atlas-pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    v6_run = args.v6_run.resolve()
    old_audit = read_json(v6_run / "link_evidence_audit.json")
    old_rows = {row["case"]: row for row in old_audit["per_layout"] if row["route"] == "groin_to_distal_calf"}
    bundle = warm_bundle(v6_run, args.atlas_pack.resolve())
    paths = sorted((v6_run / "reader_fresh").glob("*/data.npz"))
    if not paths:
        raise FileNotFoundError(f"no reader_fresh shards under {v6_run}")
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(paths)), mp_context=ctx,
                             initializer=_init, initargs=(bundle, old_rows)) as pool:
        rows = [row for batch in pool.map(_one_shard, paths, chunksize=1) for row in batch]
    rows.sort(key=lambda r: r["case"])
    full = _summarize(rows)
    output = {
        "experiment": ROOT.name,
        "scope": "v6 confirmed reader_fresh whole-route before→after_probe, fixed old IDs from v6 link_evidence_audit",
        "inputs": {
            "v6_run": str(v6_run), "link_evidence_audit_sha256": digest(v6_run / "link_evidence_audit.json"),
            "atlas_pack": str(args.atlas_pack.resolve()), "atlas_sha256": digest(args.atlas_pack.resolve()),
            "reader_freeze_sha256": digest(v6_run / "reader_freeze.json"), "workers": args.workers,
            "python_start_method": "spawn", "truth_use": "offline labels only for pair/path staging",
        },
        "stage_definitions": {
            "retrieved_true_path": "simple connectivity using all retrieved pairs with equal nonnegative instance",
            "thresholded_true_path": "simple connectivity using equal-instance retrieved pairs above learned threshold",
            "true_only_greedy_path": "exact deployment score ordering and _Forest guards, with only true pairs eligible",
            "public_greedy_path": "exact deployment score ordering and _Forest guards, all public pairs eligible",
            "public_state_track_path": "v6 objects track IDs after role fusion",
            "oracle_track_path": "v6 object_oracle track IDs, truth edges only offline",
        },
        "summary": full,
        "per_case": rows,
    }
    atomic_json(args.out.resolve(), output)
    print({"summary": full, "out": str(args.out.resolve())}, flush=True)


if __name__ == "__main__":
    main()


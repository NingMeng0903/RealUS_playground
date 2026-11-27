"""CUDA fine-tuning and CPU inference for the v10 public recovery reader."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp, softmax
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
V9 = ROOT.parent / "leg_scan_full_route_20261124_v9"
if str(V9) not in sys.path:
    sys.path.insert(0, str(V9))

import reader_v9 as v9


FORMAT = "recovery_public_candidate_reader_v10"
ROLE_NAMES = ("deep_target", "arterial_competitor", "great_saphenous", "deep_femoral_or_fibular",
              "small_saphenous_or_other", "unmodeled_blender_vessel", "artifact", "mixed")
ROLE_LABELS = (0, 1, 2, 3, 4, 5, -1, -2)
ROLE_COUNT = len(ROLE_NAMES)
ABSTAIN = -99
HISTORY_LOGIT_CAP = .75
SPLITS = ("train", "cal", "dev")
SPLIT_ALIASES = {"fit": "train", "calibration": "cal", "development": "dev"}
REQUIRED_UPDATES = ("encode1.weight", "encode2.weight", "embedding.weight", "fuse.weight",
                    "rank.weight", "null1.weight", "null2.weight", "role.weight")


def load_bundle(path):
    """Load an exact v10 model or the frozen v9 comparison model."""
    path = Path(path)
    if path.is_dir():
        path /= "reader_bundle.json"
    bundle = json.loads(path.read_text())
    if bundle["format"] == v9.FORMAT:
        return v9.load_bundle(path)
    if (bundle["format"] != FORMAT or tuple(bundle["role_names"]) != ROLE_NAMES
            or tuple(bundle["role_labels"]) != ROLE_LABELS or bundle["history_logit_cap"] != HISTORY_LOGIT_CAP):
        raise ValueError(f"Unsupported recovery reader bundle at {path}")
    for key in ("scale", "center"):
        bundle[key] = np.asarray(bundle[key], dtype=np.float32)
    bundle["weights"] = {key: np.asarray(value, dtype=np.float32) for key, value in bundle["weights"].items()}
    return bundle


def _numpy_forward(prepared, bundle, use_history):
    x, neighbors, valid, geometry, groups = prepared
    weight = bundle["weights"]
    linear, silu = v9._linear, v9._silu
    x = np.clip((x - bundle["center"]) / bundle["scale"], -9., 9.)
    hidden = silu(linear(silu(linear(x, weight, "encode1")), weight, "encode2"))
    embedding = linear(hidden, weight, "embedding")
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1.e-8)
    other = embedding[neighbors]
    similarity = np.einsum("nkf,nf->nk", other, embedding)
    edge_input = np.concatenate((geometry, similarity[:, :, None], np.abs(embedding[:, None] - other)), axis=2)
    edge = linear(silu(linear(edge_input, weight, "edge1")), weight, "edge2")[:, :, 0]
    attention = expit(edge) * valid * (.25 + .75 * geometry[:, :, -1])
    if not use_history:
        attention.fill(0.)
    support = attention.sum(axis=1, keepdims=True)
    values = linear(hidden, weight, "value")
    summary = np.einsum("nk,nkf->nf", attention, values[neighbors]) / (1. + support)
    direct = hidden + silu(linear(np.column_stack((hidden, np.zeros_like(summary), np.zeros_like(support))),
                                  weight, "fuse"))
    fused = hidden + silu(linear(np.column_stack((hidden, summary, np.log1p(support))), weight, "fuse"))
    delta = fused - direct
    cap = HISTORY_LOGIT_CAP

    def residual(name):
        return cap * np.tanh((delta @ weight[name + ".weight"].T) / cap)

    target_direct = linear(direct, weight, "target") + linear(x, weight, "raw_skip")
    target = (target_direct + residual("target"))[:, 0]
    rank_delta = delta @ (weight["target.weight"] + weight["rank.weight"]).T
    rank = (target_direct + linear(direct, weight, "rank") + cap * np.tanh(rank_delta / cap))[:, 0]
    role = linear(direct, weight, "role") + residual("role")
    count = np.bincount(groups, minlength=v9.REGIONS).astype(np.float32)[:, None]

    def null_logits(states):
        pool = np.zeros((v9.REGIONS, states.shape[1]), dtype=np.float32)
        np.add.at(pool, groups, states)
        pooled = np.column_stack((pool / np.maximum(count, 1.), np.log1p(count) / 5.))
        return linear(silu(linear(pooled, weight, "null1")), weight, "null2")[:, 0]

    direct_null = null_logits(direct)
    null = direct_null + cap * np.tanh((null_logits(fused) - direct_null) / cap)
    return target, rank, null, role, embedding, support[:, 0], edge


def _winner_indices(regions, rank):
    winners = np.full(v9.REGIONS, -1, dtype=int)
    for region in range(v9.REGIONS):
        indices = np.flatnonzero(regions == region)
        if len(indices):
            winners[region] = indices[np.argmax(rank[indices])]
    return winners


def _calibration(bundle, route, use_history):
    mode = "history" if use_history else "history_off"
    settings = bundle["calibration"][mode]
    return settings["routes"].get(route, settings["fallback"])


def _score(margin, settings):
    return expit(margin * settings["winner_scale"] + settings["winner_bias"])


def predict(public, bundle, route=None, *, threshold=None, use_history=True):
    """Return v9-compatible outputs plus calibrated probabilities in ROLE_NAMES order."""
    if bundle["format"] == v9.FORMAT:
        return v9.predict(public, bundle, route, threshold=threshold, use_history=use_history) | {
            "role_posterior": None, "role_names": None}
    if bundle["format"] != FORMAT:
        raise ValueError("Unknown recovery reader format")
    n = len(public["frames"])
    if not n:
        empty = np.empty(0, dtype=np.float32)
        return {"target": empty, "score": empty, "winner_score": empty, "null": empty, "support": empty,
                "embedding": np.empty((0, v9.EMBEDDING), dtype=np.float32),
                "role": np.empty((0, 3), dtype=np.float32),
                "role_posterior": np.empty((0, ROLE_COUNT), dtype=np.float32), "role_names": ROLE_NAMES}
    prepared = v9.prepare(public)
    with threadpool_limits(limits=1, user_api="blas"):
        target, rank, null, logits, embedding, support, _ = _numpy_forward(prepared, bundle, use_history)
    regions = prepared[-1]
    settings = _calibration(bundle, route, use_history)
    reliability = _score(rank - null[regions], settings)
    role = softmax(logits / settings["role_temperature"], axis=1).astype(np.float32)
    coarse = np.column_stack((role[:, 0], role[:, 1:6].sum(axis=1), role[:, 6:].sum(axis=1)))
    winners = _winner_indices(regions, rank)
    winners = winners[winners >= 0]
    winner_score = np.full(n, -1., dtype=np.float32)
    winner_score[winners] = reliability[winners]
    cut = settings["threshold"] if threshold is None else threshold
    return {"target": expit(target), "score": np.where(winner_score >= cut, winner_score, -1.),
            "winner_score": winner_score, "null": 1. - reliability, "embedding": embedding,
            "role": coarse, "role_posterior": role, "role_names": ROLE_NAMES, "support": support}


def read_shard(path):
    """Yield snapshots from one v10 NPZ and its aligned JSONL metadata."""
    path = Path(path)
    names = ("snapshots.jsonl", "snapshots.jsonl.gz", "metadata.jsonl", "metadata.jsonl.gz")
    metadata = next((path.with_name(name) for name in names if path.with_name(name).is_file()), None)
    if metadata is None:
        raise FileNotFoundError(f"Missing snapshot JSONL metadata beside {path}")
    opener = gzip.open if metadata.suffix == ".gz" else open
    with opener(metadata, "rt") as stream:
        descriptors = [json.loads(line) for line in stream if line.strip()]
    with np.load(path, allow_pickle=False) as archive:
        offsets = archive["offsets"]
        arrays = {key: archive[key] for key in (*v9.PUBLIC_FIELDS, "labels", "instance")}
    total = len(arrays["labels"])
    if (offsets.shape != (len(descriptors) + 1,) or offsets[0] != 0 or offsets[-1] != total
            or not np.issubdtype(offsets.dtype, np.integer) or np.any(offsets[1:] < offsets[:-1])
            or any(len(value) != total for value in arrays.values())):
        raise ValueError(f"Invalid snapshot offsets or candidate columns at {path}")
    for i, descr in enumerate(descriptors):
        first, end = offsets[i:i + 2]
        public = {key: arrays[key][first:end] for key in v9.PUBLIC_FIELDS}
        yield descr | {"public": public, "labels": arrays["labels"][first:end],
                       "instance": arrays["instance"][first:end]}


@dataclass
class Snapshot:
    inputs: tuple
    labels: np.ndarray
    instance: np.ndarray
    split: str
    layout: str
    route: str
    metadata: dict
    correct_visible: np.ndarray
    base_winners: np.ndarray
    base_scores: np.ndarray
    base_ids: np.ndarray


def _visible_winners(regions, labels, rank, margin):
    winners = _winner_indices(regions, rank)
    present = winners >= 0
    scores = np.full(v9.REGIONS, -1., dtype=np.float32)
    ids = np.full(v9.REGIONS, ABSTAIN, dtype=int)
    scores[present] = margin[winners[present]]
    ids[present] = labels[winners[present]]
    return winners, scores, ids


def _prepare_rows(rows, base):
    if isinstance(rows, (str, Path)):
        rows = (rows,)
    if isinstance(rows, Mapping):
        if "public" in rows:
            rows = (rows,)
        else:
            rows = (row | {"split": split} for split, items in rows.items() for row in items)
    deck = {split: [] for split in SPLITS}
    layout_split = {}
    fingerprint = hashlib.sha256()
    with threadpool_limits(limits=1, user_api="blas"):
        for source in rows:
            if isinstance(source, Mapping):
                items = (source,)
            else:
                path = Path(source)
                paths = sorted(path.rglob("snapshots.npz")) if path.is_dir() else (path,)
                items = (row for shard in paths for row in read_shard(shard))
            for row in items:
                layout = str(row["layout_id"])
                route = str(row["route"])
                split = SPLIT_ALIASES.get(row["split"], row["split"])
                if split not in deck or route not in v9.ROUTES or not layout:
                    raise ValueError(f"Invalid split, layout or route in {row.get('trajectory_id', layout)}")
                if layout_split.setdefault(layout, split) != split:
                    raise ValueError(f"Layout {layout} occurs in conflicting reader partitions")
                public = row["public"]
                labels = np.asarray(row["labels"])
                instance = np.asarray(row["instance"])
                n = len(public["frames"])
                if labels.shape != (n,) or instance.shape != (n,) or not np.isin(labels, ROLE_LABELS).all():
                    raise ValueError(f"Invalid labels or instances for {layout}")
                x, neighbors, valid, geometry, regions = prepared = v9.prepare(public)
                if not np.isfinite(x).all() or not np.isfinite(geometry).all():
                    raise ValueError(f"Non-finite public input for {layout}")
                correct_visible = np.bincount(regions[labels == 0], minlength=v9.REGIONS) > 0
                rank, null = v9._probability_inputs(*prepared, base, True)[1:3]
                winners, confidence, ids = _visible_winners(regions, labels, rank, expit(rank - null[regions]))
                metadata = {key: row[key] for key in ("trajectory_id", "scope", "tick", "station", "complete",
                                                       "behavior", "snapshot_id", "target_region") if key in row}
                packed = (x.astype(np.float32), neighbors.astype(np.int32), valid, geometry.astype(np.float32),
                          regions.astype(np.int8))
                labels, instance = labels.astype(np.int8), instance.astype(np.int32)
                for value in (*packed, labels, instance):
                    fingerprint.update(np.ascontiguousarray(value).view(np.uint8))
                fingerprint.update(json.dumps([split, layout, route, n, metadata], sort_keys=True).encode())
                deck[split].append(Snapshot(packed, labels, instance, split, layout, route, metadata,
                                            correct_visible, winners, confidence, ids))
    if not all(any(len(s.labels) for s in items) for items in deck.values()):
        raise ValueError("Train, calibration and development must each contain candidate-bearing snapshots")
    return deck, fingerprint.hexdigest()


def _make_model(base):
    """Import torch only when fitting. NumPy inference works without it."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from fit_reader_v9 import Reader

    class RecoveryReader(Reader):
        def __init__(self):
            super().__init__(base["center"], base["scale"])
            state = {key: torch.as_tensor(value) for key, value in base["weights"].items()}
            self.load_state_dict(state | {"center": self.center, "scale": self.scale}, strict=True)
            original = self.role
            self.role = nn.Linear(original.in_features, ROLE_COUNT)
            role_ids = torch.tensor([0, 1, 1, 1, 1, 1, 2, 2])
            multiplicity = torch.tensor([1., 5., 5., 5., 5., 5., 2., 2.]).log()
            with torch.no_grad():
                self.role.weight.copy_(original.weight[role_ids])
                self.role.bias.copy_(original.bias[role_ids] - multiplicity)

        def forward(self, x, neighbors, valid, geometry, groups, group_count, history=True):
            x = ((x - self.center) / self.scale).clamp(-9., 9.)
            hidden = F.silu(self.encode2(F.silu(self.encode1(x))))
            embedding = F.normalize(self.embedding(hidden), dim=1, eps=1.e-8)
            other = embedding[neighbors]
            similarity = (embedding[:, None] * other).sum(dim=2)
            edge_input = torch.cat((geometry, similarity[:, :, None], (embedding[:, None] - other).abs()), dim=2)
            edge = self.edge2(F.silu(self.edge1(edge_input)))[:, :, 0]
            attention = edge.sigmoid() * valid * (.25 + .75 * geometry[:, :, -1]) * history
            support = attention.sum(dim=1, keepdim=True)
            values = self.value(hidden)[neighbors]
            summary = (attention[:, :, None] * values).sum(dim=1) / (1. + support)
            direct = hidden + F.silu(self.fuse(torch.cat((hidden, torch.zeros_like(summary),
                                                        torch.zeros_like(support)), dim=1)))
            fused = hidden + F.silu(self.fuse(torch.cat((hidden, summary, support.log1p()), dim=1)))
            delta = fused - direct
            cap = HISTORY_LOGIT_CAP

            def residual(layer):
                return cap * (F.linear(delta, layer.weight) / cap).tanh()

            raw = self.raw_skip(x)
            direct_target = self.target(direct)
            target = (direct_target + raw + residual(self.target))[:, 0]
            rank_delta = F.linear(delta, self.target.weight + self.rank.weight)
            rank = (direct_target + raw + self.rank(direct) + cap * (rank_delta / cap).tanh())[:, 0]
            role = self.role(direct) + residual(self.role)
            counts = hidden.new_zeros(group_count).index_add_(0, groups, torch.ones_like(target))[:, None]

            def null_logits(states):
                pool = states.new_zeros((group_count, states.shape[1])).index_add_(0, groups, states)
                pooled = torch.cat((pool / counts.clamp_min(1.), counts.log1p() / 5.), dim=1)
                return self.null2(F.silu(self.null1(pooled)))[:, 0]

            direct_null = null_logits(direct)
            null = direct_null + cap * ((null_logits(fused) - direct_null) / cap).tanh()
            return target, rank, null, role, embedding, support[:, 0], edge, similarity

    return RecoveryReader()


def _batch(items, role_lookup, base, device):
    import torch
    from fit_reader_v9 import to_batch

    inputs, labels, instances = to_batch(items, device)
    accepted = np.concatenate([(row.base_ids == 0) & (row.base_scores >= _base_cut(base, row.route))
                               for row in items])
    accepted = torch.as_tensor(accepted, dtype=torch.bool, device=device)
    return inputs, labels, role_lookup[labels + 2], instances, accepted


def _loss(model, packed, role_weight, use_history):
    import torch
    from torch.nn import functional as F

    inputs, labels, roles, instance, previous_correct = packed
    _, neighbors, valid, _, groups, group_count = inputs
    target, rank, null, role, _, _, edge, similarity = model(*inputs, history=use_history)
    positive = labels == 0
    diff = rank - null[groups]
    maxima = rank.new_zeros(group_count).scatter_reduce_(0, groups, diff.detach(), reduce="amax")
    masses = (diff - maxima[groups]).exp()
    denominator = (-maxima).exp() + masses.new_zeros(group_count).index_add_(0, groups, masses)
    target_masses = masses.new_zeros(group_count).index_add_(0, groups, masses * positive)
    visible_targets = rank.new_zeros(group_count).index_add_(0, groups, positive.float()) > 0
    target_masses = torch.where(visible_targets, target_masses, (-maxima).exp())
    present = rank.new_zeros(group_count).index_add_(0, groups, torch.ones_like(rank)) > 0
    weights = present * (1. + .5 * previous_correct)
    list_loss = ((denominator.log() - target_masses.clamp_min(1.e-20).log()) * weights).sum() / weights.sum()
    group_rank = rank.new_full((group_count,), -torch.inf).scatter_reduce_(0, groups, rank, reduce="amax")
    ids = torch.arange(len(rank), device=rank.device)
    winners = torch.full((group_count,), len(rank), dtype=torch.long, device=rank.device)
    winners.scatter_reduce_(0, groups, torch.where(rank == group_rank[groups], ids, len(rank)), reduce="amin")
    correct = torch.cat((positive, present.new_zeros(1)))[winners]
    margin = torch.where(present, group_rank - null, 0.)
    decision = (F.binary_cross_entropy_with_logits(margin, correct.float(), reduction="none") * weights).sum() / weights.sum()
    same = (instance[:, None] >= 0) & (instance[:, None] == instance[neighbors])
    n_pos = (valid & same).sum().clamp_min(1.)
    n_neg = (valid & ~same).sum().clamp_min(1.)
    balanced = valid * torch.where(same, .5 / n_pos, .5 / n_neg)
    association = (F.binary_cross_entropy_with_logits(edge, same.float(), reduction="none") * balanced).sum()
    metric = (F.binary_cross_entropy_with_logits((similarity - .25) * 5., same.float(), reduction="none") * balanced).sum()
    unary = F.binary_cross_entropy_with_logits(target, positive.float())
    kinds = F.cross_entropy(role, roles, weight=role_weight)
    total = list_loss + .60 * decision + .65 * unary + .42 * kinds + .45 * association + .18 * metric
    return total, torch.stack((total, list_loss, decision, unary, kinds, association, metric))


def _role_index(labels):
    return np.asarray([7, 6, 0, 1, 2, 3, 4, 5], dtype=np.int64)[labels.astype(int) + 2]


def _base_cut(base, route):
    return base["thresholds"].get(route, base["fallback_threshold"])


def _panel(snapshots, bundle, use_history, collect_roles=True):
    data = []
    with threadpool_limits(limits=1, user_api="blas"):
        for row in snapshots:
            target, rank, null, logits, _, _, _ = _numpy_forward(row.inputs, bundle, use_history)
            regions = row.inputs[-1]
            winners, margins, ids = _visible_winners(regions, row.labels, rank, rank - null[regions])
            data.append((row, winners, margins, ids, logits if collect_roles else None))
    return data


def _cluster_weights(snapshots, counts):
    by_layout = Counter()
    for row, count in zip(snapshots, counts, strict=True):
        by_layout[row.layout] += count
    return np.concatenate([np.full(count, 1. / by_layout[row.layout], dtype=np.float64)
                           for row, count in zip(snapshots, counts, strict=True) if count])


def _calibration_fit(data):
    snapshots = [row[0] for row in data]
    roles = np.concatenate([_role_index(row.labels) for row in snapshots])
    logits = np.concatenate([entry[4] for entry in data])
    representative_weight = _cluster_weights(snapshots, [len(row.labels) for row in snapshots])
    present = np.stack([entry[1] >= 0 for entry in data])
    margins = np.stack([entry[2] for entry in data])[present].astype(np.float64)
    ids = np.stack([entry[3] for entry in data])[present]
    winners_weight = _cluster_weights(snapshots, present.sum(axis=1))
    weights = winners_weight / winners_weight.sum()
    correct = ids == 0

    def role_nll(temperature):
        z = logits / temperature
        nll = logsumexp(z, axis=1) - z[np.arange(len(roles)), roles]
        return float(np.average(nll, weights=representative_weight))

    fit_role = minimize_scalar(role_nll, bounds=(.5, 4.), method="bounded")
    role_temperature = float(fit_role.x) if fit_role.success else 1.

    def objective(params):
        slope, bias = params
        z = slope * margins + bias
        value = np.sum(weights * (np.logaddexp(0., z) - correct * z))
        value += .002 * ((slope - 1.) ** 2 + bias ** 2)
        error = weights * (expit(z) - correct)
        grad = np.array([error @ margins, error.sum()]) + .004 * np.array([slope - 1., bias])
        return value, grad

    # Winner calibration needs both outcomes. Keep the warm-start confidence transform otherwise.
    winner_fit = minimize(objective, (1., 0.), bounds=((.2, 5.), (-4., 4.)), method="L-BFGS-B", jac=True) if (
        correct.any() and not correct.all()) else None
    confidence_fit = winner_fit is not None and winner_fit.success
    slope, bias = winner_fit.x.tolist() if confidence_fit else (1., 0.)
    role_layouts = {label: set() for label in ROLE_LABELS}
    for row in snapshots:
        for label in np.unique(row.labels):
            role_layouts[label].add(row.layout)
    return {"winner_scale": slope, "winner_bias": bias, "role_temperature": role_temperature,
            "candidate_count": int(len(roles)), "winner_count": int(len(ids)),
            "winner_correct": int(correct.sum()), "role_support": dict(zip(
                ROLE_NAMES, np.bincount(roles, minlength=ROLE_COUNT).tolist(), strict=True)),
            "role_layout_support": {name: len(role_layouts[label])
                                    for name, label in zip(ROLE_NAMES, ROLE_LABELS, strict=True)},
            "role_nll_unscaled": role_nll(1.), "role_nll_calibrated": role_nll(role_temperature),
            "role_temperature_fitted": bool(fit_role.success), "winner_fit": confidence_fit,
            "layout_count": len({row.layout for row in snapshots if len(row.labels)})}


def _threshold_fit(data, settings, base):
    snapshots = [entry[0] for entry in data]
    ids = np.stack([entry[3] for entry in data])
    scores = _score(np.stack([entry[2] for entry in data]), settings)
    scores[ids == ABSTAIN] = -1.
    old_ids = np.stack([row.base_ids for row in snapshots])
    old_taken = np.stack([row.base_scores >= _base_cut(base, row.route) for row in snapshots])
    old_count = int(old_taken.sum())
    old_branch = int(((old_ids > 0) & old_taken).sum())
    old_branch_cases = int(((old_ids > 0) & old_taken).any(axis=1).sum())
    old_correct = (old_ids == 0) & old_taken
    old_artifact = int(((old_ids < 0) & old_taken).sum())
    complete = np.array([_complete(row) for row in snapshots])
    # Keep v9 exposure to wrong branches at both route and region levels. Among
    # nearby coverages prefer strict completions, then retention and rescues.
    tolerance = max(2, int(np.ceil(.035 * old_count)))
    cuts = np.unique(np.r_[0., scores[scores >= 0.], np.nextafter(np.float32(1.), np.float32(2.))])
    best_key, best = None, None
    for cut in cuts:
        taken = scores >= cut
        count = int(taken.sum())
        branch = int(((ids > 0) & taken).sum())
        branch_cases = int(((ids > 0) & taken).any(axis=1).sum())
        errors = int(((ids != 0) & taken).sum())
        artifact = int(((ids < 0) & taken).sum())
        correct = (ids == 0) & taken
        kept = int((correct & old_correct).sum())
        strict = int((correct.all(axis=1) & complete).sum())
        distance = abs(count - old_count)
        safe = branch <= old_branch and branch_cases <= old_branch_cases
        close = distance <= tolerance
        key = (safe, close and safe, strict if close and safe else -distance, kept,
               int(correct.sum()), -errors, -distance, float(cut))
        if best_key is None or key > best_key:
            best_key = key
            best = {"threshold": float(cut), "coverage_near_v9": close, "within_v9_branch_risk": safe,
                    "submitted": count, "wrong_branch_regions": branch, "wrong_branch_cases": branch_cases,
                    "artifact_or_mixed_regions": artifact, "correct_regions": int(correct.sum()),
                    "retained_v9_correct": kept, "strict": strict}
    return best | {"coverage_tolerance": tolerance, "baseline_submitted": old_count,
                   "baseline_wrong_branch_regions": old_branch, "baseline_wrong_branch_cases": old_branch_cases,
                   "baseline_artifact_or_mixed_regions": old_artifact, "baseline_correct_regions": int(old_correct.sum()),
                   "snapshots": len(data)}


def _complete(row):
    return bool(row.metadata.get("complete", "terminal" in str(row.metadata.get("scope", "")).split("_")))


def _fit_calibration(panel, base):
    fallback = _calibration_fit(panel)
    fallback |= _threshold_fit(panel, fallback, base)
    per_route = {}
    for route in v9.ROUTES:
        data = [entry for entry in panel if entry[0].route == route]
        if data and any(len(entry[0].labels) for entry in data):
            settings = _calibration_fit(data)
            per_route[route] = settings | _threshold_fit(data, settings, base)
    return {"routes": per_route, "fallback": fallback}


def _role_name(label):
    return "abstain" if label == ABSTAIN else ROLE_NAMES[ROLE_LABELS.index(int(label))]


def _risk_summary(ids, accepted, correct_visible, complete):
    correct = (ids == 0) & accepted
    wrong = (ids != 0) & accepted
    branch = (ids > 0) & accepted
    wrong_names = {name: int((accepted & (ids == role)).sum()) for role, name in zip(
        ROLE_LABELS[1:], ROLE_NAMES[1:], strict=True)}
    total = int(accepted.sum())
    return {"submitted": total, "wrong_regions": int(wrong.sum()), "wrong_branch_regions": int(branch.sum()),
            "wrong_branch_cases": int(branch.any(axis=1).sum()), "wrong_delivery_cases": int(wrong.any(axis=1).sum()),
            "correct_regions": int(correct.sum()), "strict": int((correct.all(axis=1) & complete).sum()),
            "risk": float(wrong.sum() / total) if total else None,
            "wrong_branch_risk": float(branch.sum() / total) if total else None, "wrong_role_counts": wrong_names,
            "captured_regions": int(correct_visible.sum()), "readout_misses": int((correct_visible & ~correct).sum()),
            "captured_and_abstained": int((correct_visible & ~accepted).sum()),
            "captured_and_wrong": int((correct_visible & wrong).sum())}


def _retention(old_ids, old, ids, new):
    old_good = (old_ids == 0) & old
    new_good = (ids == 0) & new
    lost = old_good & ~new_good
    counts = {"previous_correct": int(old_good.sum()), "retained_correct": int((old_good & new_good).sum()),
              "rescued_correct": int((new_good & ~old_good).sum()), "lost_to_abstention": int((lost & ~new).sum()),
              "lost_to_wrong_selection": int((lost & new).sum()), "lost_by_named_role": {
                  name: int((lost & new & (ids == role)).sum())
                  for name, role in zip(ROLE_NAMES[1:], ROLE_LABELS[1:], strict=True)}}
    counts["per_region"] = [{"region": r, "previous_correct": int(old_good[:, r].sum()),
                             "retained_correct": int((old_good & new_good)[:, r].sum()),
                             "rescued_correct": int((new_good & ~old_good)[:, r].sum()),
                             "lost_correct": int(lost[:, r].sum())} for r in range(v9.REGIONS)]
    return counts


def _match_count(scores, counts):
    ranking = np.argsort(-scores, axis=1, kind="stable")
    mask_in_order = np.arange(v9.REGIONS) < counts[:, None]
    accepted = np.zeros_like(scores, dtype=bool)
    np.put_along_axis(accepted, ranking, mask_in_order, axis=1)
    return accepted & (scores >= 0.)


def _coverage_curve(ids, scores, visible, complete):
    values = np.r_[np.linspace(.05, .95, 19), np.nextafter(np.float32(1.), np.float32(2.))]
    opportunities = int((scores >= 0.).sum())
    curve = []
    for cut in values:
        accepted = scores >= cut
        stats = _risk_summary(ids, accepted, visible, complete)
        curve.append({"threshold": float(cut), "coverage_of_observed_regions": stats["submitted"] / max(1, opportunities),
                      **{key: stats[key] for key in ("submitted", "correct_regions", "wrong_regions", "risk",
                                                     "wrong_branch_regions", "wrong_branch_risk")}})
    return curve


def _paired_intervals(snapshots, old_ids, old_taken, ids, new_taken, complete, seed):
    layout_keys, layout_inverse = np.unique([row.layout for row in snapshots], return_inverse=True)
    old_correct = (old_ids == 0) & old_taken
    new_correct = (ids == 0) & new_taken
    old_branch = (old_ids > 0) & old_taken
    new_branch = (ids > 0) & new_taken
    contrasts = {"correct_regions_per_snapshot": new_correct.sum(axis=1) - old_correct.sum(axis=1),
                 "strict_per_snapshot": (new_correct.all(axis=1) & complete).astype(int)
                 - (old_correct.all(axis=1) & complete).astype(int),
                 "wrong_branch_regions_per_snapshot": new_branch.sum(axis=1) - old_branch.sum(axis=1)}
    rng = np.random.default_rng(seed)
    by_layout_n = np.bincount(layout_inverse)
    draws = rng.integers(0, len(layout_keys), (1500, len(layout_keys)))
    n_drawn = by_layout_n[draws].sum(axis=1)
    results = {}
    for key, diff in contrasts.items():
        aggregates = np.bincount(layout_inverse, weights=diff)
        interval = np.quantile(aggregates[draws].sum(axis=1) / n_drawn, [.025, .975]).tolist()
        results[key] = {"difference": float(diff.mean()), "clustered_95": interval if len(layout_keys) >= 2 else None}
    return results


def _role_metrics(data, bundle):
    confusion = np.zeros((ROLE_COUNT, ROLE_COUNT), dtype=np.int64)
    nll, squared, counts = 0., 0., 0
    bin_counts = np.zeros(10)
    bin_correct = np.zeros(10)
    bin_prob = np.zeros(10)
    for row, _, _, _, logits in data:
        if not len(row.labels):
            continue
        settings = _calibration(bundle, row.route, True)
        probs = softmax(logits / settings["role_temperature"], axis=1)
        truth = _role_index(row.labels)
        pred = probs.argmax(axis=1)
        np.add.at(confusion, (truth, pred), 1)
        p_true = probs[np.arange(len(truth)), truth]
        nll -= np.log(p_true.clip(1.e-12)).sum(dtype=np.float64)
        squared += ((probs ** 2).sum(axis=1) - 2. * p_true + 1.).sum(dtype=np.float64)
        counts += len(truth)
        confidence = probs[np.arange(len(pred)), pred]
        bins = np.minimum(9, (10 * confidence).astype(int))
        bin_counts += np.bincount(bins, minlength=10)
        bin_correct += np.bincount(bins, weights=pred == truth, minlength=10)
        bin_prob += np.bincount(bins, weights=confidence, minlength=10)
    return {"role_names": ROLE_NAMES, "confusion_truth_rows": confusion.tolist(), "candidates": counts,
            "negative_log_likelihood": float(nll / counts) if counts else None,
            "brier_multiclass": float(squared / counts) if counts else None,
            "top_label_ece": float(np.abs(bin_correct - bin_prob).sum() / counts) if counts else None}


def _summary(data, history_off, bundle, base, seed):
    snapshots = [entry[0] for entry in data]
    old_ids = np.stack([row.base_ids for row in snapshots])
    old_scores = np.stack([row.base_scores for row in snapshots])
    old_taken = old_scores >= np.array([_base_cut(base, row.route) for row in snapshots])[:, None]
    visible = np.stack([row.correct_visible for row in snapshots])
    complete = np.array([_complete(row) for row in snapshots])
    baseline = _risk_summary(old_ids, old_taken, visible, complete)

    def score_mode(panel, history):
        settings = [_calibration(bundle, row.route, history) for row in snapshots]
        margins = [entry[2] for entry in panel]
        scores = np.stack([_score(m, s) for m, s in zip(margins, settings, strict=True)])
        ids = np.stack([entry[3] for entry in panel])
        scores[ids == ABSTAIN] = -1.
        taken = scores >= np.array([s["threshold"] for s in settings])[:, None]
        matched = _match_count(scores, old_taken.sum(axis=1))
        report = {"autonomous": _risk_summary(ids, taken, visible, complete),
                  "matched_v9_count": _risk_summary(ids, matched, visible, complete),
                  "matched_counts_exact": bool(np.array_equal(old_taken.sum(axis=1), matched.sum(axis=1))),
                  "retention": _retention(old_ids, old_taken, ids, taken),
                  "retention_at_matched_count": _retention(old_ids, old_taken, ids, matched),
                  "risk_coverage": _coverage_curve(ids, scores, visible, complete),
                  "paired_layout_bootstrap": _paired_intervals(snapshots, old_ids, old_taken, ids, taken, complete, seed)}
        return report, (scores, ids, taken)

    history_report, (scores, ids, taken) = score_mode(data, True)
    no_history_report, _ = score_mode(history_off, False)
    scopes = defaultdict(list)
    for i, row in enumerate(snapshots):
        scopes[str(row.metadata.get("scope", "unspecified"))].append(i)
    scope_reports = {}
    for scope, indices in scopes.items():
        ix = np.asarray(indices)
        scope_reports[scope] = {"snapshots": len(ix), "layouts": len({snapshots[i].layout for i in indices}),
                                "completed": int(complete[ix].sum()),
                                "v9": _risk_summary(old_ids[ix], old_taken[ix], visible[ix], complete[ix]),
                                "v10": _risk_summary(ids[ix], taken[ix], visible[ix], complete[ix]),
                                "retention": _retention(old_ids[ix], old_taken[ix], ids[ix], taken[ix])}
    reports = {"snapshots": len(snapshots), "layouts": len({row.layout for row in snapshots}),
               "completed": int(complete.sum()), "v9": baseline,
               "v9_risk_coverage": _coverage_curve(old_ids, old_scores, visible, complete),
               "v10": history_report, "v10_history_off": no_history_report, "roles": _role_metrics(data, bundle),
               "by_acquisition_scope": scope_reports}
    paired = []
    for i, row in enumerate(snapshots):
        paired.append({"layout_id": row.layout, "route": row.route, **row.metadata,
                       "regions_with_target": visible[i].tolist(),
                       "v9_selected_roles": [_role_name(v) for v in np.where(old_taken[i], old_ids[i], ABSTAIN)],
                       "v10_selected_roles": [_role_name(v) for v in np.where(taken[i], ids[i], ABSTAIN)],
                       "v9_winner_scores": old_scores[i].tolist(), "v10_winner_scores": scores[i].tolist()})
    return reports, paired


def _write_json(path, value):
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def _write_checkpoint(path, state, torch):
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(state, pending)
    pending.replace(path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(bundle):
    return bundle | {"center": bundle["center"].tolist(), "scale": bundle["scale"].tolist(),
                     "weights": {key: value.tolist() for key, value in bundle["weights"].items()}}


def _cpu_bundle(model):
    return {"format": FORMAT, "role_names": ROLE_NAMES, "role_labels": ROLE_LABELS,
            "center": model.center.detach().cpu().numpy(), "scale": model.scale.detach().cpu().numpy(),
            "weights": {key: t.detach().cpu().numpy().copy() for key, t in model.named_parameters()}}


def _parity(model, bundle, snapshots, base):
    import torch
    from fit_reader_v9 import to_batch

    errors, warm_errors = {}, []
    mode_names = ("target", "rank", "null", "role_logits", "embedding", "support", "edge_logits")
    model.eval()
    with torch.no_grad(), threadpool_limits(limits=1, user_api="blas"):
        for snapshot in snapshots:
            inputs = to_batch([snapshot], "cuda")[0]
            for history in (True, False):
                cpu_outputs = _numpy_forward(snapshot.inputs, bundle, history)
                gpu_outputs = model(*inputs, history=history)
                for name, cpu, gpu in zip(mode_names, cpu_outputs, gpu_outputs[:7], strict=True):
                    key = ("history_" if history else "direct_") + name
                    errors[key] = max(errors.get(key, 0.), float(np.abs(cpu - gpu.cpu().numpy()).max(initial=0.)))
            direct = _numpy_forward(snapshot.inputs, bundle, False)
            prior = v9._probability_inputs(*snapshot.inputs, base, False)
            for i in range(3):
                warm_errors.append(float(np.abs(direct[i] - prior[i]).max(initial=0.)))
            named = softmax(direct[3], axis=1)
            coarse = np.c_[named[:, 0], named[:, 1:6].sum(1), named[:, 6:].sum(1)]
            warm_errors.append(float(np.abs(coarse - prior[4]).max(initial=0.)))
    return dict(errors, absolute_max=max(errors.values())), max(warm_errors)


def _pick_parity_rows(cal):
    rows = []
    for route in v9.ROUTES:
        match = next((row for row in cal if len(row.labels) and row.route == route), None)
        if match is not None:
            rows.append(match)
    return rows


def fit_reader(rows, base_bundle_path, output_dir, seed=26112410, *, epochs=8, batch_size=16,
               learning_rate=.00065, resume=False):
    """Fit the public reader on train layouts, calibrate on cal, and score dev once."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("v10 reader fine-tuning requires CUDA")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Expected positive epochs, batch size and learning rate")
    started = monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_bundle_path)
    if base_path.is_dir():
        base_path /= "reader_bundle.json"
    base = load_bundle(base_path)
    if base["format"] != v9.FORMAT:
        raise ValueError("The initialization bundle must be the frozen v9 reader")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    deck, data_fingerprint = _prepare_rows(rows, base)
    role_counts = np.bincount(np.concatenate([_role_index(row.labels) for row in deck["train"]]),
                              minlength=ROLE_COUNT)
    layout_ids = {name: sorted({row.layout for row in items}) for name, items in deck.items()}
    protocol = {"format": FORMAT, "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate,
                "v9_sha256": _sha256(base_path), "data_sha256": data_fingerprint, "layout_ids": layout_ids,
                "source_sha256": _sha256(__file__), "v9_reader_source_sha256": _sha256(V9 / "reader_v9.py"),
                "v9_gpu_source_sha256": _sha256(V9 / "fit_reader_v9.py"),
                "history_logit_cap": HISTORY_LOGIT_CAP, "role_names": ROLE_NAMES,
                "snapshot_counts": {name: len(items) for name, items in deck.items()},
                "candidate_counts": {name: sum(len(row.labels) for row in items) for name, items in deck.items()},
                "train_role_counts": dict(zip(ROLE_NAMES, role_counts.tolist(), strict=True))}
    track_file = out / "reader_protocol.json"
    if resume:
        if json.dumps(json.loads(track_file.read_text()), sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError("Resume requires the same reader source, input deck, and optimizer settings")
    else:
        _write_json(track_file, protocol)
    model = _make_model(base).cuda()
    start_weights = {key: t.detach().cpu().clone() for key, t in model.named_parameters()}
    parity_rows = _pick_parity_rows(deck["cal"])
    warm_parity, warm_error = _parity(model, _cpu_bundle(model), parity_rows, base)
    if warm_error > 1.e-3 or warm_parity["absolute_max"] > 1.e-3:
        raise RuntimeError(f"Warm-start CPU or CUDA check failed: {warm_error}, {warm_parity}")
    train = [row for row in deck["train"] if len(row.labels)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.004, fused=True)
    role_lookup = torch.tensor([7, 6, 0, 1, 2, 3, 4, 5], device="cuda")
    nonzero = role_counts[role_counts > 0]
    role_weight = np.clip((float(np.median(nonzero)) / np.maximum(role_counts, 1)) ** .35, .55, 2.7)
    role_weight = torch.tensor(role_weight, dtype=torch.float32, device="cuda")
    trace, first, steps = [], 1, 0
    checkpoint = out / "reader_latest.pt"
    protocol_hash = _sha256(track_file)
    if resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_hash:
            raise ValueError("Checkpoint hash does not match the reader protocol")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        first, trace, steps = saved["epoch"] + 1, saved["trace"], saved["steps"]
        if first > epochs + 1:
            raise ValueError("Requested epochs are earlier than the resume checkpoint")
    print("CUDA v10 reader", protocol["snapshot_counts"], "layouts", {k: len(v) for k, v in layout_ids.items()},
          flush=True)
    loss_names = ("total", "listwise", "decision", "unary", "named_role", "association", "embedding")
    for epoch in range(first, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        history_disabled = 0
        sums = torch.zeros(len(loss_names), device="cuda")
        n = 0
        for begin in range(0, len(order), batch_size):
            items = [train[i] for i in order[begin:begin + batch_size]]
            packed = _batch(items, role_lookup, base, "cuda")
            # Whole-prefix dropout gives the direct path its own supervision.
            use_history = n == 0 or rng.random() >= .28
            history_disabled += int(not use_history)
            loss, terms = _loss(model, packed, role_weight, use_history)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4., error_if_nonfinite=True)
            optimizer.step()
            sums += terms.detach()
            n += 1
        steps += n
        averages = dict(zip(loss_names, (sums / n).cpu().tolist(), strict=True))
        trace.append({"epoch": epoch, "optimizer_steps": steps, "history_disabled_batches": history_disabled,
                      "train_loss": averages})
        payload = {"epoch": epoch, "steps": steps, "trace": trace, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                   "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                   "protocol_sha256": protocol_hash}
        _write_checkpoint(checkpoint, payload, torch)
        print(f"CUDA v10 reader epoch {epoch} steps {steps} loss {averages['total']:.5f} "
              f"seconds {monotonic() - started:.1f}", flush=True)
    deltas = {key: float((t.detach().cpu() - start_weights[key]).abs().max()) for key, t in model.named_parameters()}
    active_optimizer = max((int(item["step"].item()) for item in optimizer.state.values()), default=0)
    if active_optimizer != steps or steps <= 0 or any(deltas[key] <= 0. for key in REQUIRED_UPDATES):
        raise RuntimeError("Required shared encoder, association or readout parameters did not update on CUDA")
    bundle = _cpu_bundle(model)
    parity, _ = _parity(model, bundle, parity_rows, base)
    if parity["absolute_max"] > 1.e-3:
        raise RuntimeError(f"CPU/CUDA recovery reader parity failed: {parity}")

    cal = _panel(deck["cal"], bundle, True)
    cal_off = _panel(deck["cal"], bundle, False)
    calibration = {"history": _fit_calibration(cal, base), "history_off": _fit_calibration(cal_off, base)}
    old_thresholds = dict(base["thresholds"], fallback=base["fallback_threshold"])
    history_cal = calibration["history"]
    thresholds = {route: values["threshold"] for route, values in history_cal["routes"].items()}
    bundle |= {"feature_order": base["feature_order"], "calibration": calibration, "thresholds": thresholds,
               "fallback_threshold": history_cal["fallback"]["threshold"], "v9_thresholds": old_thresholds,
               "train_role_counts": protocol["train_role_counts"], "history_logit_cap": HISTORY_LOGIT_CAP,
               "v9_bundle_sha256": protocol["v9_sha256"], "protocol_sha256": protocol_hash,
               "checkpoint_sha256": _sha256(checkpoint)}
    bundle_path = out / "reader_bundle.json"
    _write_json(bundle_path, _plain(bundle))
    frozen = load_bundle(bundle_path)
    fixed, paired = {}, {}
    for name in SPLITS:
        if name == "cal":
            panel, panel_off = cal, cal_off
        else:
            panel = _panel(deck[name], frozen, True)
            panel_off = _panel(deck[name], frozen, False, collect_roles=False)
        fixed[name], paired[name] = _summary(panel, panel_off, frozen, base, seed)
        fixed[name]["routes"] = {}
        for route in v9.ROUTES:
            ids = [i for i, entry in enumerate(panel) if entry[0].route == route]
            if ids:
                report, _ = _summary([panel[i] for i in ids], [panel_off[i] for i in ids], frozen, base, seed)
                fixed[name]["routes"][route] = report
    results_path = out / "reader_results.json"
    _write_json(results_path, {"bundle_sha256": _sha256(bundle_path), "fixed_trajectory": True,
                               "partitions": fixed, "paired_prefixes": paired})
    manifest = protocol | {"gpu": torch.cuda.get_device_name(), "device": str(next(model.parameters()).device),
                          "torch_version": str(torch.__version__), "optimizer_steps": steps, "trace": trace,
                          "max_absolute_parameter_updates": deltas, "warm_start_direct_v9_error": warm_error,
                          "warm_start_cuda_cpu": warm_parity, "final_cuda_cpu": parity,
                          "bundle_sha256": _sha256(bundle_path), "checkpoint_sha256": bundle["checkpoint_sha256"],
                          "results_sha256": _sha256(results_path), "canonical_thresholds": {"v9": old_thresholds,
                              "v10": dict(thresholds, fallback=bundle["fallback_threshold"])},
                          "elapsed_seconds": monotonic() - started}
    _write_json(out / "reader_manifest.json", manifest)
    return frozen


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, nargs="+")
    parser.add_argument("--base", type=Path, default=V9 / "run" / "reader_bundle.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=26112410)
    parser.add_argument("--learning-rate", type=float, default=.00065)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = args.rows[0] if len(args.rows) == 1 else args.rows
    fit_reader(rows, args.base, args.out, args.seed, epochs=args.epochs, batch_size=args.batch_size,
               learning_rate=args.learning_rate, resume=args.resume)


if __name__ == "__main__":
    main()

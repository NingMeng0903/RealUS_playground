"""Train and freeze the end-to-end reader on layout-disjoint acquired scans."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.special import expit, softmax
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from reader_v9 import (CANDIDATE_FIELDS, EDGE_FIELDS, EMBEDDING, FORMAT, PUBLIC_FIELDS, REGIONS, ROUTES,
                       infer_prepared, load_bundle, predict, prepare)

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "leg_scan_reader_feedback_20261123_v4" / "run"
V5 = ROOT.parent / "leg_scan_anchor_state_20261123_v5" / "run"
V8 = ROOT.parent / "leg_scan_reader_feedback_values_20261124_v8" / "run"
sys.path.insert(0, str(ROOT.parent / "leg_scan_target_routes_20261123_v2"))
from route_metrics import outcomes, paired, totals


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, payload):
    temp = path.with_suffix(path.suffix + ".pending")
    torch.save(payload, temp)
    temp.replace(path)


@dataclass
class Snapshot:
    inputs: tuple
    labels: np.ndarray
    instance: np.ndarray
    layout: str
    route: str
    scope: str


def open_shard(path, terminal=False):
    ledger = path.with_name("cases.jsonl.gz")
    commit = json.loads(path.with_name("commit.json").read_text())
    hashes = {"data": digest(path), "metadata": digest(ledger)}
    if hashes != {"data": commit["data_sha256"], "metadata": commit["metadata_sha256"]}:
        raise RuntimeError(f"Archived shard hashes changed at {path}")
    with gzip.open(ledger, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != commit["cases"]:
        raise RuntimeError(f"Missing cases at {path}")
    descriptors = [(row, record["scope"]) for row in rows for record in row["histories"]]
    with np.load(path, allow_pickle=False) as arch:
        offsets = arch["offsets"]
        columns = {key: arch[key] for key in (*PUBLIC_FIELDS, "labels", "instance")}
        schema = arch["feature_order"].tolist()
    snaps = []
    for (start, end), (row, scope) in zip(pairwise(offsets), descriptors, strict=True):
        if terminal and scope != "terminal":
            continue
        public = {key: columns[key][start:end] for key in PUBLIC_FIELDS}
        snaps.append((public, columns["labels"][start:end], columns["instance"][start:end],
                      row["layout_id"], row["case"]["task"], scope))
    return snaps, hashes | {"cases": len(rows), "snapshots": len(snaps)}, schema


def train_shard(path):
    raw, provenance, schema = open_shard(path)
    snaps = []
    for public, labels, instance, layout, route, scope in raw:
        features, neighbors, valid, geometry, regions = prepare(public)
        packed = (features.astype(np.float16), neighbors.astype(np.int16), valid,
                  geometry.astype(np.float16), regions.astype(np.int8))
        snaps.append(Snapshot(packed, labels, instance, layout, route, scope))
    return snaps, provenance, schema


class Reader(nn.Module):
    def __init__(self, center, scale):
        super().__init__()
        width, values = 48, 24
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.encode1 = nn.Linear(CANDIDATE_FIELDS, 64)
        self.encode2 = nn.Linear(64, width)
        self.embedding = nn.Linear(width, EMBEDDING)
        self.edge1 = nn.Linear(EDGE_FIELDS + EMBEDDING + 1, 16)
        self.edge2 = nn.Linear(16, 1)
        self.value = nn.Linear(width, values)
        self.fuse = nn.Linear(width + values + 1, width)
        self.target = nn.Linear(width, 1)
        self.raw_skip = nn.Linear(CANDIDATE_FIELDS, 1)
        self.rank = nn.Linear(width, 1)
        self.role = nn.Linear(width, 3)
        self.null1 = nn.Linear(width + 1, 16)
        self.null2 = nn.Linear(16, 1)

    def forward(self, x, neighbors, valid, geometry, groups, group_count):
        x = ((x - self.center) / self.scale).clamp(-9., 9.)
        hidden = F.silu(self.encode2(F.silu(self.encode1(x))))
        embedding = F.normalize(self.embedding(hidden), dim=1, eps=1.e-8)
        other = embedding[neighbors]
        similarity = (embedding[:, None] * other).sum(dim=2)
        edge_input = torch.cat((geometry, similarity[:, :, None], (embedding[:, None] - other).abs()), dim=2)
        edge = self.edge2(F.silu(self.edge1(edge_input)))[:, :, 0]
        attention = edge.sigmoid() * valid * (.25 + .75 * geometry[:, :, -1])
        support = attention.sum(dim=1, keepdim=True)
        values = self.value(hidden)[neighbors]
        summary = (attention[:, :, None] * values).sum(dim=1) / (1. + support)
        fused = hidden + F.silu(self.fuse(torch.cat((hidden, summary, support.log1p()), dim=1)))
        target = (self.target(fused) + self.raw_skip(x))[:, 0]
        rank = target + self.rank(fused)[:, 0]
        role = self.role(fused)
        pool = fused.new_zeros((group_count, fused.shape[1])).index_add_(0, groups, fused)
        counts = fused.new_zeros(group_count).index_add_(0, groups, torch.ones_like(target))[:, None]
        pooled = torch.cat((pool / counts.clamp_min(1.), counts.log1p() / 5.), dim=1)
        null = self.null2(F.silu(self.null1(pooled)))[:, 0]
        return target, rank, null, role, similarity, edge, embedding, support[:, 0]


def to_batch(snapshots, device):
    features, neighbor, valid, geometry, groups, labels, instances = ([] for _ in range(7))
    begin = 0
    for i, item in enumerate(snapshots):
        x, n, v, g, region = item.inputs
        features.append(x)
        neighbor.append(n.astype(np.int32) + begin)
        valid.append(v)
        geometry.append(g)
        groups.append(region.astype(np.int32) + i * REGIONS)
        labels.append(item.labels)
        instances.append(item.instance)
        begin += len(x)
    def tensor(parts, dtype):
        return torch.from_numpy(np.concatenate(parts)).to(device=device, dtype=dtype)
    inputs = (tensor(features, torch.float32), tensor(neighbor, torch.long), tensor(valid, torch.bool),
              tensor(geometry, torch.float32), tensor(groups, torch.long), len(snapshots) * REGIONS)
    return inputs, tensor(labels, torch.long), tensor(instances, torch.long)


def supervised(model, batch):
    inputs, labels, instance = batch
    _, neighbors, valid, _, groups, group_count = inputs
    target, rank, null, role, similarity, edge, _, _ = model(*inputs)
    positive = labels == 0
    role_id = torch.where(positive, 0, torch.where(labels > 0, 1, 2))
    difference = rank - null[groups]
    # Multi-positive listwise probability on each region's visible set plus
    # the null option; the target set is every correct candidate, or null.
    maxima = rank.new_zeros(group_count).scatter_reduce_(0, groups, difference.detach(), reduce="amax")
    mass = (difference - maxima[groups]).exp()
    denominator = (-maxima).exp() + mass.new_zeros(group_count).index_add_(0, groups, mass)
    pos_mass = mass.new_zeros(group_count).index_add_(0, groups, mass * positive)
    has_positive = pos_mass > 0
    pos_mass = torch.where(has_positive, pos_mass, (-maxima).exp())
    present = rank.new_zeros(group_count).index_add_(0, groups, torch.ones_like(rank)) > 0
    list_loss = ((denominator.log() - pos_mass.clamp_min(1.e-20).log()) * present).sum() / present.sum().clamp_min(1)
    group_rank = rank.new_full((group_count,), -40.).scatter_reduce_(0, groups, rank, reduce="amax")
    # Pick a unique view for reliability diagnostics even when ranks tie.
    ids = torch.arange(len(rank), device=rank.device)
    winners = torch.full((group_count,), len(rank), dtype=torch.long, device=rank.device).scatter_reduce_(
        0, groups, torch.where(rank == group_rank[groups], ids, len(rank)), reduce="amin")
    correct = torch.cat((positive, present.new_zeros(1)))[winners]
    decision = (F.binary_cross_entropy_with_logits(group_rank - null, correct.float(), reduction="none")
                * present).sum() / present.sum().clamp_min(1)

    same = (instance[:, None] >= 0) & (instance[:, None] == instance[neighbors])
    n_pos = (valid & same).sum().clamp_min(1)
    n_neg = (valid & ~same).sum().clamp_min(1)
    balanced = valid * torch.where(same, .5 / n_pos, .5 / n_neg)
    association = (F.binary_cross_entropy_with_logits(edge, same.float(), reduction="none") * balanced).sum()
    metric = (F.binary_cross_entropy_with_logits((similarity - .25) * 5., same.float(), reduction="none") * balanced).sum()
    unary = F.binary_cross_entropy_with_logits(target, positive.float())
    kinds = F.cross_entropy(role, role_id)
    loss = list_loss + .60 * decision + .70 * unary + .22 * kinds + .55 * association + .15 * metric
    edge_accuracy = ((edge > 0).eq(same) * balanced).sum()
    return loss, torch.stack((loss, list_loss, decision, unary, kinds, association, metric,
                              (correct & present).sum(), has_positive.sum(), present.sum(), edge_accuracy,
                              (valid & same).sum(), (valid & ~same).sum(), ((group_rank >= null) & correct).sum()))


STAT_FIELDS = ("loss", "listwise", "decision", "candidate", "role", "association", "embedding",
               "correct_picks", "seen_regions", "observed_regions", "balanced_edge_accuracy",
               "true_pairs", "false_pairs", "correct_non_null")


def pass_batches(model, deck, indices, batch_size, optimizer=None):
    model.train(optimizer is not None)
    sums = torch.zeros(len(STAT_FIELDS), device="cuda")
    steps = 0
    for begin in range(0, len(indices), batch_size):
        items = [deck[int(i)] for i in indices[begin:begin + batch_size] if len(deck[int(i)].labels)]
        if not items:
            continue
        packed = to_batch(items, "cuda")
        with torch.set_grad_enabled(optimizer is not None):
            loss, row = supervised(model, packed)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.)
            optimizer.step()
        sums += row.detach()
        steps += 1
    if not steps:
        raise RuntimeError("Reader pass has no candidate-bearing snapshots")
    stats = dict(zip(STAT_FIELDS, sums.cpu().tolist(), strict=True))
    for key in (*STAT_FIELDS[:7], "balanced_edge_accuracy"):
        stats[key] /= steps
    for key in ("correct_picks", "seen_regions", "observed_regions", "true_pairs", "false_pairs", "correct_non_null"):
        stats[key] = int(stats[key])
    return stats | {"steps": steps}


def fit_scaling(deck):
    total, square, count = (np.zeros(CANDIDATE_FIELDS, dtype=np.float64) for _ in range(2)), None, 0
    total, square = total
    for snapshot in deck:
        x = snapshot.inputs[0].astype(np.float32)
        total += x.sum(axis=0, dtype=np.float64)
        square += np.square(x).sum(axis=0, dtype=np.float64)
        count += len(x)
    center = total / count
    scale = np.sqrt(np.maximum(square / count - center ** 2, .025 ** 2))
    return center.astype(np.float32), scale.astype(np.float32)


def changed_parameters(model, initial):
    return [key for key, value in model.named_parameters() if not torch.equal(value.detach().cpu(), initial[key])]


def tiny_smoke(out, paths, seed):
    # Four deterministic terminal layouts straddling both route families.
    source, _, _ = train_shard(paths[0])
    deck = [v for v in source if v.scope == "terminal"][:4]
    center, scale = fit_scaling(deck)
    model = Reader(center, scale).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.004)
    initial = {k: t.detach().cpu().clone() for k, t in model.named_parameters()}
    ix = np.arange(len(deck))
    before = pass_batches(model, deck, ix, len(deck))
    for _ in range(100):
        pass_batches(model, deck, ix, len(deck), optimizer)
    after = pass_batches(model, deck, ix, len(deck))
    updated = changed_parameters(model, initial)
    verified = (after["loss"] < .35 * before["loss"] and after["correct_picks"] >= .90 * after["seen_regions"]
                and after["true_pairs"] > 0 and after["false_pairs"] > 0
                and len(updated) == len(initial))
    result = {"seed": seed, "layouts": [v.layout for v in deck], "routes": [v.route for v in deck],
              "before": before, "after": after, "changed_parameters": updated, "cuda_optimizer_steps": 100,
              "gpu": torch.cuda.get_device_name(), "verified": verified}
    atomic_json(out / "reader_smoke.json", result)
    print("Reader tiny overfit", before["loss"], after["loss"], after["correct_picks"], after["seen_regions"], flush=True)
    if not verified:
        raise RuntimeError("The reader failed the fixed-layout CUDA overfit check")


def reference_bundle():
    index = json.loads((V5 / "reader_freeze.json").read_text())
    model_path = V5 / "reader_weights" / Path(index["fit"]["weights"]).name
    actual = digest(model_path)
    if actual != index["fit"]["sha256"]:
        raise RuntimeError("Pinned v8 unary baseline checkpoint hash changed")
    old = torch.load(model_path, map_location="cpu", weights_only=False)
    configs = V8 / "reader_bundle.json"
    return old["center"], old["scale"], {key: value.numpy() for key, value in old["model"].items()}, json.loads(
        configs.read_text())["routes"], {"weights_sha256": actual, "calibration_sha256": digest(configs)}


def reference_predict(public, route, bundle):
    center, scale, weights, configs, _ = bundle
    x = np.clip((public["features"].astype(np.float32) - center) / scale, -9., 9.)
    with threadpool_limits(limits=1, user_api="blas"):
        for layer in ("features.0", "features.2"):
            x = x @ weights[layer + ".weight"].T + weights[layer + ".bias"]
            x *= expit(x)
        probability = softmax(x @ weights["identity.weight"].T + weights["identity.bias"], axis=1)
    config = configs[route]["config"]["unary"]
    risk = config["winner"]
    weight = np.asarray(risk["weight"])
    scores = np.full(len(x), -1.)
    for region in range(REGIONS):
        ix = np.flatnonzero(public["regions"] == region)
        if len(ix):
            best = ix[np.argmax(probability[ix, 0])]
            p = probability[best]
            others = ix[ix != best]
            value = np.r_[p[0], p[0] - probability[others, 0].max(initial=0.),
                          p[0] - p[1:].max(), public["quality"][best], 0.,
                          p[6] + p[7], min(1., len(ix) / 120.), min(1., len(ix) / 40.), np.eye(REGIONS)[region]]
            scores[best] = expit(((value - risk["center"]) / risk["scale"]) @ weight[:-1] + weight[-1])
    return scores, config["threshold"]


def load_panel(paths, terminal=True):
    rows = []
    provenance = {}
    for path in paths:
        array, hashes, schema = open_shard(path, terminal)
        rows.extend(array)
        provenance[str(path.relative_to(ROOT.parent))] = hashes
    return rows, provenance, schema


def score_panel(rows, bundle, reference):
    panel = {route: [] for route in ROUTES}
    for public, labels, _, layout, route, scope in rows:
        regions = public["regions"]
        truth = {"label": labels, "region": regions, "complete": scope == "terminal"}
        base_scores, base_cut = reference_predict(public, route, reference)
        winner = predict(public, bundle, route, threshold=0.)["winner_score"]
        no_history = predict(public, bundle, route, threshold=0., use_history=False)["winner_score"]
        panel[route].append((layout, truth, winner, no_history, outcomes(truth, base_scores, base_cut)))
    return panel


def select_threshold(rows, output_index):
    ref = totals(row[4] for row in rows)
    peak = np.full((len(rows), REGIONS), -1.)
    labels = np.full((len(rows), REGIONS), -99)
    for i, row in enumerate(rows):
        ix = np.flatnonzero(row[output_index] >= 0)
        regions = row[1]["region"][ix]
        peak[i, regions] = row[output_index][ix]
        labels[i, regions] = row[1]["label"][ix]
    options = []
    for cut in np.unique(np.r_[0., peak[peak >= 0.], 1.]):
        accepted = peak >= cut
        n = int(accepted.sum())
        strict = int(np.all(accepted & (labels == 0), axis=1).sum())
        wrong = int((accepted & (labels > 0)).any(axis=1).sum())
        errors = int((accepted & (labels != 0)).sum())
        delta = abs(n - ref["submitted"])
        allowed = delta <= max(4, .035 * ref["submitted"]) and wrong <= ref["wrong_branch"]
        options.append((allowed, n, strict, wrong, errors, delta, float(cut)))
    valid = [row for row in options if row[0]]
    best = max(valid, key=lambda r: (r[2], -r[4], -r[5], r[6])) if valid else min(
        options, key=lambda r: (r[5], r[3], r[4]))
    return dict(zip(("admissible", "submitted", "strict", "wrong_branch", "wrong_regions", "coverage_delta", "threshold"),
                    best, strict=True)) | {"cases": len(rows), "reference": ref}


def evaluate_panel(panel, settings, seed):
    results = {}
    for route, rows in panel.items():
        cases, truth, history, no_history, base = zip(*rows, strict=True)
        fixed = [outcomes(t, s, settings[route]["history"]["threshold"]) for t, s in zip(truth, history, strict=True)]
        fixed_no = [outcomes(t, s, settings[route]["history_off"]["threshold"]) for t, s in zip(truth, no_history, strict=True)]
        matched = [outcomes(t, s, 0., b["submitted"]) for t, s, b in zip(truth, history, base, strict=True)]
        matched_no = [outcomes(t, s, 0., b["submitted"]) for t, s, b in zip(truth, no_history, base, strict=True)]
        if any(a["submitted"] != b["submitted"] for group in (matched, matched_no) for a, b in zip(group, base, strict=True)):
            raise RuntimeError("Submission counts differ in matched-coverage reader comparison")
        summaries = {}
        for name, alternative in (("autonomous", fixed), ("matched", matched), ("autonomous_history_off", fixed_no),
                                  ("matched_history_off", matched_no)):
            summaries[name] = totals(alternative) | {"vs_unary": paired(alternative, base, cases, seed)}
        summaries["matched"]["vs_history_off"] = paired(matched, matched_no, cases, seed + 1)
        results[route] = summaries | {"v8_unary": totals(base), "pairs": {
            case: {"v8_unary": b, "autonomous": a, "matched": m, "autonomous_history_off": f, "matched_history_off": n}
            for case, b, a, m, f, n in zip(cases, base, fixed, matched, fixed_no, matched_no, strict=True)}}
    return results


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=26112409)
    parser.add_argument("--heldout", type=Path, default=V8 / "reader_test")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the end-to-end reader fit")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    files = sorted((V4 / "fit").glob("*/data.npz")) + sorted((V5 / "fit_revisit").glob("*/data.npz"))
    calibration_files = sorted((V4 / "select_reader").glob("*/data.npz"))
    heldout_files = sorted(args.heldout.glob("*/data.npz"))
    if not (files and calibration_files and heldout_files):
        raise RuntimeError("The fit, calibration and held-out scan archives must be complete")
    if not args.resume:
        tiny_smoke(args.out, files, args.seed)
    if args.smoke_only:
        return

    start = monotonic()
    source_hashes = {path.name: digest(path) for path in (Path(__file__), ROOT / "reader_v9.py")}
    fit, validation, shards = [], [], {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for path, (rows, metadata, schema) in zip(files, executor.map(train_shard, files), strict=True):
            shards[str(path.relative_to(ROOT.parent))] = metadata
            if "feature_order" not in locals():
                feature_order = schema
            if schema != feature_order:
                raise RuntimeError("Feature order changed across training shards")
            for row in rows:
                # Deterministic layout hash holds out complete acquisition histories.
                is_val = int(hashlib.sha256(row.layout.encode()).hexdigest()[:8], 16) % 20 == 0
                (validation if is_val else fit).append(row)
            if len(shards) % 10 == 0:
                print("Reader loaded shards", len(shards), "/", len(files), "seconds", round(monotonic() - start), flush=True)

    fit_ids = {row.layout for row in fit}
    val_ids = {row.layout for row in validation}
    cal_rows, cal_shards, cal_schema = load_panel(calibration_files)
    held_rows, held_shards, held_schema = load_panel(heldout_files)
    cal_ids = {row[3] for row in cal_rows}
    held_ids = {row[3] for row in held_rows}
    partitions = (fit_ids, val_ids, cal_ids, held_ids)
    if any(a & b for i, a in enumerate(partitions) for b in partitions[i + 1:]) or not all(partitions):
        raise RuntimeError("Reader stage layout sets overlap or are empty")
    if cal_schema != feature_order or held_schema != feature_order:
        raise RuntimeError("Feature schema mismatch between fit and evaluation")
    ref = reference_bundle()
    provenance = {"seed": args.seed, "sources": source_hashes, "fit_sources": shards,
                  "calibration_sources": cal_shards, "heldout_sources": held_shards, "v8_unary": ref[-1],
                  "layouts": {name: sorted(ids) for name, ids in zip(("fit", "validation", "calibration", "heldout"),
                                                                   partitions, strict=True)},
                  "feature_order": feature_order}
    manifest = args.out / "reader_protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise RuntimeError("Existing reader protocol differs from this run")
    atomic_json(manifest, provenance)
    protocol_sha = digest(manifest)
    center, scale = fit_scaling(fit)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = Reader(center, scale).cuda()
    initial = {key: value.detach().cpu().clone() for key, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, fused=True)
    best_loss, steps, trace, begin = float("inf"), 0, [], 1
    recent_path, best_path = (args.out / f"reader_{name}.pt" for name in ("latest", "best"))
    if args.resume:
        saved = torch.load(recent_path, map_location="cpu", weights_only=False)
        if saved["protocol_sha256"] != protocol_sha or saved["batch_size"] != args.batch:
            raise RuntimeError("Cannot resume with a different protocol or batch size")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        best_loss, steps, trace, begin = saved["best_loss"], saved["steps"], saved["trace"], saved["epoch"] + 1
    print("CUDA full reader fit", {key: len(v) for key, v in zip(("fit", "val", "cal", "heldout"), partitions)},
          len(fit), len(validation), "snapshots", flush=True)
    for epoch in range(begin, args.epochs + 1):
        fit_stats = pass_batches(model, fit, rng.permutation(len(fit)), args.batch, optimizer)
        valid_stats = pass_batches(model, validation, np.arange(len(validation)), args.batch)
        steps += fit_stats["steps"]
        trace.append({"epoch": epoch, "fit": fit_stats, "validation": valid_stats})
        is_best = valid_stats["loss"] < best_loss
        best_loss = min(best_loss, valid_stats["loss"])
        payload = {"epoch": epoch, "steps": steps, "batch_size": args.batch, "best_loss": best_loss, "trace": trace,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state,
                   "protocol_sha256": protocol_sha}
        save_checkpoint(recent_path, payload)
        if is_best:
            save_checkpoint(best_path, payload)
        print("Reader CUDA epoch", epoch, "steps", steps, "fit", round(fit_stats["loss"], 4),
              "val", round(valid_stats["loss"], 4), "best", is_best, "seconds", round(monotonic() - start), flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    updated = changed_parameters(model, initial)
    if len(updated) != len(initial) or not optimizer.state:
        raise RuntimeError("Not all deployed reader parameters were updated on CUDA")
    weights = {key: value.detach().cpu().numpy() for key, value in model.named_parameters()}
    bundle = {"format": FORMAT, "weights": weights, "center": center, "scale": scale,
              "thresholds": {}, "fallback_threshold": .5}
    sample = next(row for row in validation if len(row.labels))
    gpu = model(*to_batch([sample], "cuda")[0])
    reference_cpu = infer_prepared(sample.inputs, bundle)
    pairs = zip(reference_cpu[:6], (gpu[0], gpu[1], gpu[2], gpu[6], gpu[3].softmax(1), gpu[5]), strict=True)
    parity = max(float(np.max(np.abs(a - b.detach().cpu().numpy()))) for a, b in pairs)
    if parity > 1.e-3:
        raise RuntimeError(f"CPU inference disagreed with the trained CUDA reader: {parity}")
    thresholds = {}
    calibration = score_panel(cal_rows, bundle, ref)
    for route, rows in calibration.items():
        thresholds[route] = {"history": select_threshold(rows, 2), "history_off": select_threshold(rows, 3)}
    bundle["thresholds"] = {route: values["history"]["threshold"] for route, values in thresholds.items()}
    bundle["fallback_threshold"] = float(np.mean(list(bundle["thresholds"].values())))
    bundle_path = args.out / "reader_bundle.json"
    payload = bundle | {"weights": {key: val.tolist() for key, val in weights.items()},
                        "center": center.tolist(), "scale": scale.tolist(), "feature_order": feature_order,
                        "protocol_sha256": protocol_sha, "checkpoint_sha256": digest(best_path)}
    atomic_json(bundle_path, payload)
    atomic_json(args.out / "reader_fit.json", {"device": "cuda:0", "gpu": torch.cuda.get_device_name(),
                "gpu_parameter_updates": updated, "parameters": int(sum(p.numel() for p in model.parameters())),
                "optimizer_steps": steps, "best_checkpoint_step": best["steps"], "best_epoch": best["epoch"],
                "train_snapshots": len(fit), "train_candidate_rows": int(sum(len(s.labels) for s in fit)),
                "fit_layouts": len(fit_ids), "validation_layouts": len(val_ids), "calibration_layouts": len(cal_ids),
                "heldout_layouts": len(held_ids), "epochs": trace, "calibration": thresholds, "parity_abs_max": parity,
                "protocol_sha256": protocol_sha, "bundle_sha256": digest(bundle_path)})
    print("Reader frozen", bundle["thresholds"], "parity", parity, "SHA256", digest(bundle_path), flush=True)
    frozen = load_bundle(bundle_path)
    held_panel = score_panel(held_rows, frozen, ref)
    result = evaluate_panel(held_panel, thresholds, args.seed)
    atomic_json(args.out / "reader_results.json", {"protocol_sha256": protocol_sha, "bundle_sha256": digest(bundle_path),
                "fixed_trajectory": "held-out terminal observed prefixes",
                "autonomous": "calibration-frozen region submission thresholds",
                "matched": "exactly the v8 unary submission count per observed trajectory",
                "routes": result})
    for route, summary in result.items():
        print("Held-out reader", route, {name: {k: row[k] for k in ("strict", "wrong_branch", "submitted")}
              for name, row in summary.items() if name in ("v8_unary", "autonomous", "matched", "matched_history_off")}, flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()

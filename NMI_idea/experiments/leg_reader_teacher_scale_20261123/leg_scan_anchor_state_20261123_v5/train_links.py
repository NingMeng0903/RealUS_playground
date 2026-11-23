"""CUDA-update the association head with the deployed spatial retrieval."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from study_data import ROOT, V4_RUN, atomic_json, digest
import numpy as np
import torch
from torch.nn import functional as F

from fit_public import _atomic_torch_save, _cpu_optimizer, load_public
from object_tracks import retrieve


PAIR_FIELDS = ("frames", "slots", "stations", "sectors", "points", "quality", "regions")
GROUPS = ("same_instance", "near_artery", "clutter", "mixture", "same_named_or_role_competitor", "other_instance")


def pair_types(rows, pairs, lo, hi):
    x, y = pairs.T
    labels = rows["labels"][lo:hi].astype(int)
    instance = rows["instance"][lo:hi]
    a, b = labels[x], labels[y]
    same = (instance[x] >= 0) & (instance[x] == instance[y])
    arterial = ((a == 0) & (b == 1)) | ((b == 0) & (a == 1))
    alternative_same_name = ((a == 0) & (b == 5)) | ((b == 0) & (a == 5)) | ((a == b) & (a >= 0) & ~same)
    group = np.select((same, (a == -2) | (b == -2), (a == -1) | (b == -1), arterial, alternative_same_name),
                      (0, 3, 2, 1, 4), default=5)
    interval = rows["frames"][lo:hi][y] - rows["frames"][lo:hi][x]
    stationed = abs(rows["stations"][lo:hi][y] - rows["stations"][lo:hi][x]) <= 1.5
    far_view = (interval >= 20) & stationed
    return group.astype(np.int8), far_view


def retrieve_training_pairs(rows, rng):
    selected, tags = [], []
    counts = Counter()
    for lo, hi in pairwise(rows["offsets"]):
        local = {name: rows[name][lo:hi] for name in PAIR_FIELDS}
        pairs = retrieve(local)
        if not len(pairs):
            continue
        group, long_gap = pair_types(rows, pairs, lo, hi)
        local_indices = []
        for kind, cap in enumerate((104, 40, 45, 45, 35, 45)):
            prior = np.flatnonzero(group == kind)
            if len(prior):
                local_indices.append(rng.choice(prior, min(cap, len(prior)), replace=False))
        more = np.flatnonzero(long_gap)
        if len(more):
            local_indices.append(rng.choice(more, min(72, len(more)), replace=False))
        IDs = np.unique(np.concatenate(local_indices))
        population = np.bincount(pairs[group == 0, 1], minlength=hi - lo)
        counts["competing_pairs_at_shared_destination"] += int(((population[pairs[IDs, 1]] > 0) & (group[IDs] != 0)).sum())
        counts["long_gap_pairs"] += int(long_gap[IDs].sum())
        counts["long_gap_positive_pairs"] += int((long_gap[IDs] & (group[IDs] == 0)).sum())
        for category, name in enumerate(GROUPS):
            counts[name] += int((group[IDs] == category).sum())
        counts["retrieved_before_sampling"] += len(pairs)
        selected.append(pairs[IDs] + lo)
        tags.append(group[IDs] == 0)
    return np.concatenate(selected), np.concatenate(tags), counts


def train(paths, warm, out, *, epochs=3, seed=603_087_231):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for association updating")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cuda = torch.device("cuda:0")
    checkpoint = load_public(warm, cuda)
    net = checkpoint["model"]
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.association.parameters():
        parameter.requires_grad_(True)
    starting = {k: t.detach().clone() for k, t in net.state_dict().items()}
    optim = torch.optim.AdamW(net.association.parameters(), lr=.0025, weight_decay=.004, fused=True)
    ledger, losses = Counter(), []
    steps = 0
    cached = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in (*PAIR_FIELDS, "features", "offsets", "labels", "instance")}
        pairs, labels, stats = retrieve_training_pairs(arrays, rng)
        normalized = np.clip((arrays["features"].astype(np.float32) - checkpoint["center"]) / checkpoint["scale"], -9., 9.)
        with torch.no_grad():
            encoded = net.features(torch.as_tensor(normalized, device=cuda)).half()
        cached.append((encoded, torch.as_tensor(pairs, device=cuda), torch.as_tensor(labels, device=cuda, dtype=torch.float32)))
        ledger.update(stats)
        ledger["candidate_rows"] += len(normalized)
        ledger["observed_histories"] += len(arrays["offsets"]) - 1
    for epoch in range(epochs):
        track = []
        for source in rng.permutation(len(cached)):
            encoded, edges, same = cached[source]
            positive = torch.clamp((1. - same.mean()) / same.mean(), min=1., max=3.)
            for ids in torch.randperm(len(edges), device=cuda).split(1536):
                samples = edges[ids]
                latent = F.normalize(net.association(encoded[samples].float()), dim=-1)
                similarity = torch.sum(latent[:, 0] * latent[:, 1], dim=1)
                objective = F.binary_cross_entropy_with_logits(7. * (similarity - .35), same[ids], pos_weight=positive)
                optim.zero_grad(set_to_none=True)
                objective.backward()
                optim.step()
                track.append(float(objective.detach()))
                steps += 1
        losses.append(float(np.mean(track)))
        print(f"association CUDA {epoch+1}/{epochs}, {steps} steps, loss {losses[-1]:.4f}", flush=True)
        payload = {k: checkpoint[k] for k in ("feature_order", "center", "scale", "classes")}
        payload.update({"model": {k: t.detach().cpu() for k, t in net.state_dict().items()},
                        "optimizer": _cpu_optimizer(optim), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                        "epoch": epoch + 1, "variant": "v5_retrieval_association"})
        _atomic_torch_save(out / f"association_epoch_{epoch+1:02d}.pt", payload)
    torch.cuda.synchronize()
    changed = [k for k, value in net.state_dict().items() if not torch.equal(starting[k], value)]
    if not changed or any(not k.startswith("association.") for k in changed):
        raise RuntimeError("The update should change only the association head")
    final = out / "association.pt"
    _atomic_torch_save(final, payload)
    result = {"weights": str(final), "sha256": digest(final), "warm_start": str(warm), "warm_start_sha256": digest(warm),
              "device": str(cuda), "optimizer_steps": steps, "epochs": epochs, "loss": losses,
              "identity_unchanged": True, "changed_parameters": changed, "supervision": dict(ledger),
              "shards": [str(p) for p in paths]}
    atomic_json(out / "fit.json", result)
    return result

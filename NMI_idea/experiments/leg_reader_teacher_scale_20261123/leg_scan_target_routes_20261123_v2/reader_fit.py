"""Fit matched candidate readers on acquired public vectors in a fixed atlas panel."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CLASSES = ("connected_target", "arterial_competitor", "great_saphenous", "other_vein",
           "other_atlas_role", "alternate_same_name_component", "clutter", "mixed")
TRAIN_SEED = 371_700_513


class Readout(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(size, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU())
        self.identity = nn.Linear(24, len(CLASSES))
        self.association = nn.Linear(24, 8)

    def forward(self, x):
        encoded = self.features(x)
        return self.identity(encoded), F.normalize(self.association(encoded), dim=-1)


def transform_labels(raw):
    raw = np.asarray(raw, dtype=int)
    if not np.all(np.isin(raw, np.arange(-2, 6))):
        raise ValueError("Unrecognized evaluator-only candidate label")
    return np.where(raw == -1, 6, np.where(raw == -2, 7, raw))


def association_pairs(items, rng):
    """A supervised contrast uses genuinely different acquired poses."""
    offsets = []
    labels = []
    cursor = 0
    for item in items:
        f, s, a, y = (np.asarray(item[key]) for key in ("frame", "station", "sector", "label"))
        if len(f) < 2:
            cursor += len(f)
            continue
        first = rng.integers(len(f), size=min(8 * len(f), 2400))
        second = np.clip(first + rng.integers(-18, 19, size=len(first)), 0, len(f) - 1)
        ds = abs(s[first] - s[second])
        da = abs((a[first] - a[second] + 12.) % 24. - 12.)
        ok = ((f[first] != f[second]) & (ds <= 3.1)
              & ((ds >= .63) | (da >= .74)) & (y[first] >= 0) & (y[second] >= 0))
        first, second = first[ok], second[ok]
        same = y[first] == y[second]
        # Grouped nuisance roles need not be the same anatomical object.
        ok = ~same | (y[first] <= 2) | (y[first] == 5)
        offsets.extend(zip(first[ok] + cursor, second[ok] + cursor))
        labels.extend(same[ok])
        cursor += len(f)
    return np.asarray(offsets, dtype=np.int64).reshape(-1, 2), np.asarray(labels, dtype=np.float32)


def fit(items, local_features, names, output, *, epochs=9):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed reader experiment requires the approved CUDA environment")
    device = torch.device("cuda:0")
    data = np.concatenate([np.asarray(x["features"], dtype=np.float32) for x in items], axis=0)
    labels = transform_labels(np.concatenate([x["label"] for x in items]))
    case_counts = np.concatenate([np.full(len(x["label"]), 1. / max(1, len(x["label"]))) for x in items])
    case_counts /= np.mean(case_counts)
    rng = np.random.default_rng(TRAIN_SEED)
    pairs, same = association_pairs(items, rng)
    if data.shape[1] != len(names) or not np.isfinite(data).all() or len(pairs) == 0:
        raise ValueError("Invalid public feature panel or missing association examples")
    center = data.mean(axis=0)
    scale = np.maximum(data.std(axis=0), .035)
    z = np.clip((data - center) / scale, -9., 9.)
    split = torch.as_tensor(labels, device=device)
    weights = torch.as_tensor(case_counts, device=device, dtype=torch.float32)
    pair_idx = torch.as_tensor(pairs, device=device)
    pair_y = torch.as_tensor(same, device=device)
    counts = np.bincount(labels, minlength=len(CLASSES)).tolist()
    summaries, trained = {}, {}
    for variant in ("local_embedding", "full_public_embedding"):
        torch.manual_seed(TRAIN_SEED)
        torch.cuda.manual_seed_all(TRAIN_SEED)
        net = Readout(len(names)).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.005, fused=True)
        x = torch.as_tensor(z.copy(), device=device)
        if variant == "local_embedding":
            x[:, local_features:] = 0.
        before = net.identity.weight.detach().clone()
        trace = []
        for epoch in range(epochs):
            order = torch.randperm(len(split), device=device)
            roll = torch.randperm(len(pair_idx), device=device)
            for index, batch in enumerate(order.split(768)):
                pair_at = roll[(torch.arange(192, device=device) + index * 192) % len(roll)]
                sample = pair_idx[pair_at]
                logits, _ = net(x[batch])
                _, left = net(x[sample[:, 0]])
                _, right = net(x[sample[:, 1]])
                match = 4. * (torch.sum(left * right, dim=1) - .40)
                identity = (F.cross_entropy(logits, split[batch], reduction="none") * weights[batch]).mean()
                loss = identity + .18 * F.binary_cross_entropy_with_logits(match, pair_y[pair_at])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            trace.append(float(loss.detach().item()))
        torch.cuda.synchronize()
        if torch.equal(before, net.identity.weight):
            raise RuntimeError("Identity head did not update on CUDA")
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        destination = output / f"{variant}.pt"
        tmp = destination.with_suffix(".pending")
        torch.save({"model": state, "feature_order": tuple(names), "center": center,
                    "scale": scale, "local_feature_count": local_features, "classes": CLASSES,
                    "variant": variant}, tmp)
        tmp.replace(destination)
        trained[variant] = {"model": net.cpu().eval(), "center": center, "scale": scale,
                            "variant": variant, "local_feature_count": local_features}
        summaries[variant] = {"device": str(device), "gpu": torch.cuda.get_device_name(device),
                              "candidate_count": len(labels), "label_counts": counts,
                              "association_pairs": len(pairs), "positive_associations": int(same.sum()),
                              "loss_by_epoch": trace, "weights": str(destination),
                              "parameters": sum(v.numel() for v in net.parameters())}
    return trained, summaries


def score(data, reader):
    values = np.clip((data - reader["center"]) / reader["scale"], -9., 9.)
    if reader["variant"] == "local_embedding":
        values[:, reader["local_feature_count"]:] = 0.
    with torch.inference_mode():
        p, _ = reader["model"](torch.as_tensor(values, dtype=torch.float32))
        return p.softmax(dim=-1)[:, 0].numpy()

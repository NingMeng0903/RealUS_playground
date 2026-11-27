"""Train the cross-window association head and the terminal reader on CUDA."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import os

import numpy as np
from scipy.special import expit, softmax, logsumexp
from scipy.optimize import minimize_scalar

from replay_v11 import ROOT
from recovery_reader import ROLE_LABELS, load_bundle, predict, read_shard
from recovery_task import DEFAULT_PACK
from run_recovery import _snapshots_verified
from readout_state import evaluate
from run_v11 import READER
from study_v9 import atomic_json, digest, read_json
from study_worlds import StudyWorlds
from track_state import REGIONS, edges, features, from_logits, linear


def net(width, widths, device="cuda"):
    import torch
    from torch import nn

    dims = [width, *widths]
    seq = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        seq.append(nn.Linear(a, b))
        if i < len(widths) - 1:
            seq.append(nn.SiLU())
    return nn.Sequential(*seq).to(device)


def bundle(net_, center, scale):
    tensors = [l for l in net_ if hasattr(l, "weight")]
    return {"mean": center, "scale": scale, "weight": [m.weight.detach().cpu().numpy().copy() for m in tensors],
            "bias": [m.bias.detach().cpu().numpy().copy() for m in tensors]}


def hydrate(obj):
    return obj | {k: np.asarray(obj[k], dtype=np.float32) for k in ("mean", "scale")} | {
        k: [np.asarray(a, dtype=np.float32) for a in obj[k]] for k in ("weight", "bias")}


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(x) for x in value]
    return value.tolist() if hasattr(value, "tolist") else value


def _checkpoint(path, model, optimizer, rng, **extra):
    import torch

    tmp = path.with_suffix(".part")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state, **extra}, tmp)
    os.replace(tmp, path)
    return digest(path)


def _norm(x):
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), .075).astype(np.float32)


def _torch_setup(seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch


def load_deck(root, atlas):
    scanner = StudyWorlds(atlas)
    frozen = load_bundle(READER)
    deck, manifests = {}, {}
    for stage in ("fit", "cal", "reader_dev"):
        rows = []
        for path in sorted((root / stage).glob("*/observer/snapshots.npz")):
            if not _snapshots_verified(path.parent):
                raise RuntimeError("A raw observation shard changed")
            manifests[str(path.relative_to(root))] = digest(path.parent / "snapshot_commit.json")
            for doc in read_shard(path):
                p = doc["public"]
                if len(p["frames"]) < 2:
                    continue
                doc["base"] = predict(p, frozen, doc["route"])
                doc["surface"] = scanner.environment(doc["shape"]).scanner.surface
                rows.append(doc)
        if not rows:
            raise RuntimeError(f"No reader data for {stage}")
        deck[stage] = rows
        print("prepared reader", stage, len(rows), flush=True)
    layout_sets = [{r["layout_id"] for r in rows} for rows in deck.values()]
    if len(set.union(*layout_sets)) != sum(map(len, layout_sets)):
        raise RuntimeError("Reader cohorts overlap")
    return deck, manifests


def edge_items(rows, rng, max_per_scan):
    data, targets = [], []
    for row in rows:
        pair, x = edges(row["public"], row["base"])
        labels, inst = row["labels"], row["instance"]
        a, b = pair.T
        matched = (inst[a] >= 0) & (inst[a] == inst[b]) & (x[:, 6] == 0)
        mask = np.isin(row["public"]["regions"][a], REGIONS) | (rng.random(len(a)) < .22)
        keep = np.flatnonzero(mask)
        if len(keep) > max_per_scan:
            keep = rng.choice(keep, max_per_scan, replace=False)
        if len(keep):
            data.append(x[keep])
            targets.append(matched[keep])
    return np.concatenate(data), np.concatenate(targets)


def fit_edges(deck, out, rng):
    torch = _torch_setup(202612)
    from torch.nn import functional as F

    fit_x, fit_y = edge_items(deck["fit"], rng, 800)
    cal_x, cal_y = edge_items(deck["cal"], rng, 1300)
    center, scale = _norm(fit_x)
    model = net(fit_x.shape[1], (40, 24, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.002, fused=True)
    xb = torch.as_tensor(((fit_x-center)/scale).clip(-6., 6.), device="cuda")
    yb = torch.as_tensor(fit_y, dtype=torch.float32, device="cuda")
    for i in range(320):
        row = torch.as_tensor(rng.integers(len(xb), size=1536), device="cuda")
        loss = F.binary_cross_entropy_with_logits(model(xb[row]).squeeze(-1), yb[row])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    exported = bundle(model, center, scale)
    p = expit(linear(exported, cal_x)[:, 0])
    eligible = cal_x[:, 6] == 0
    positive = int((eligible & cal_y).sum())
    options = []
    for t in (.55, .65, .72, .78, .82, .86, .90, .93, .96):
        chosen = (p >= t) & eligible
        correct = int(np.sum(chosen & cal_y))
        precision = correct / max(1, int(chosen.sum()))
        recall = correct/max(1, positive)
        options.append((t, precision, recall, int(chosen.sum())))
    acceptable = [r for r in options if r[1] >= .88]
    selected = max(acceptable, key=lambda a:a[2]) if acceptable else max(options, key=lambda a:a[1])
    exported["threshold"] = selected[0]
    stats = {"cuda_steps": 320, "training_pairs": len(fit_x), "calibration_pairs": len(cal_x),
             "calibration_options": options, "selected": selected,
             "checkpoint_sha256": _checkpoint(out/"association.pt", model, opt, rng, step=320)}
    atomic_json(out/"association.json", plain(exported))
    atomic_json(out/"association_training.json", stats)
    print("CUDA association", stats, flush=True)
    return exported


def role_labels(labels):
    lookup = {r: i for i, r in enumerate(ROLE_LABELS)}
    return np.asarray([lookup[int(r)] for r in labels], dtype=np.int64)


def group_rows(fit, rng):
    """Preserve rare artery and ambiguity rows in per-region ranking batches."""
    for row in fit:
        p, y = row["public"], row["labels"]
        for r in REGIONS:
            at = np.flatnonzero(p["regions"] == r)
            if not len(at):
                continue
            peak = at[np.argsort(-row["base"]["target"][at], kind="stable")[:20]]
            remaining = np.setdiff1d(at, peak)
            remaining = rng.choice(remaining, size=min(55, len(remaining)), replace=False)
            at = np.r_[peak, remaining]
            yield row["features"][at], role_labels(y[at]), r


def prepare_groups(rows, rng):
    all_groups = list(group_rows(rows, rng))
    n, width = len(all_groups), all_groups[0][0].shape[1]
    x = np.zeros((n, 75, width), dtype=np.float32)
    y = np.full((n, 75), -100, dtype=np.int64)
    masks = np.zeros((n, 75), dtype=bool)
    weights = np.empty(n, dtype=np.float32)
    for i, (f, truth, region) in enumerate(all_groups):
        x[i, :len(f)], y[i, :len(f)], masks[i, :len(f)] = f, truth, True
        weights[i] = 1.45 if region == 4 else 1.
    return x, y, masks, weights


def _test(rows, model, blend, *, material=True):
    actual, types, cnt = [], Counter(), Counter()
    for row in rows:
        if not row["complete"]:
            continue
        p, base = row["public"], row["base"]
        x = row["features"]
        if not material:
            x, _ = features(p, base, row["surface"], row["edge"], material=False)
        z = linear(model, x)
        res = from_logits(p, base, row["tracks"], softmax(z[:, :8]/model["temperature"], axis=1),
                          expit(z[:, 8]/model["rank_temperature"]), blend)
        score = evaluate(p, row["labels"], res["score"], 0., complete=True)
        ids = score["selected_ids"]
        cnt.update(strict=score["strict"], wrong=score["wrong_branch"], r5_correct=ids[4]==0,
                   r5_artery=ids[4]==1, layouts=1)
        types[row["scope"]] += 1
        actual.append(score["strict"])
    return dict(cnt) | {"trajectory_types": dict(types)}


def _temperatures(model, rows):
    role, rank, labels = [], [], []
    for row in rows:
        x, y = row["features"], row["labels"]
        take = np.isin(row["public"]["regions"], REGIONS)
        z = linear(model, x[take])
        labels.append(role_labels(y[take]))
        role.append(z[:, :8])
        rank.append(z[:, 8])
    r, q, y = np.concatenate(role), np.concatenate(rank), np.concatenate(labels)
    i = np.arange(len(y))
    t = minimize_scalar(lambda s: np.mean(logsumexp(r/s, axis=1) - r[i, y]/s), bounds=(.35, 3.), method="bounded").x
    b = (y == 0).astype(float)
    t_rank = minimize_scalar(lambda s: np.mean(np.logaddexp(0., q/s)-b*q/s),
                             bounds=(.35, 3.), method="bounded").x
    return float(t), float(t_rank)


def fit_reader(deck, association, out, rng):
    torch = _torch_setup(415002)
    from torch.nn import functional as F

    for rows in deck.values():
        for row in rows:
            row["features"], row["tracks"] = features(row["public"], row["base"], row["surface"], association)
            row["edge"] = association
    x, y, mask, w = prepare_groups(deck["fit"], rng)
    center, scale = _norm(x[mask])
    model = net(x.shape[-1], (72, 48, 9))
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=.0018, weight_decay=.006, fused=True)
    tx = torch.as_tensor(((x-center)/scale).clip(-6., 6.), device="cuda")
    ty = torch.as_tensor(y, device="cuda")
    tm = torch.as_tensor(mask, device="cuda")
    tw = torch.as_tensor(w, device="cuda")
    hist = np.bincount(y[mask], minlength=8)
    weight = torch.as_tensor(np.clip((np.median(hist[hist > 0])/np.maximum(1, hist))**.18, .55, 2.),
                             dtype=torch.float32, device="cuda")
    incumbent, trace, best = None, [], None
    for epoch in range(1, 9):
        for step in range(55):
            index = rng.integers(len(tx), size=25)
            ix = torch.as_tensor(index, device="cuda")
            z = model(tx[ix])
            valid, labels = tm[ix], ty[ix]
            g = labels == 0
            rank = z[..., 8]
            ids = F.cross_entropy(z[..., :8][valid], labels[valid], weight=weight)
            weight_row = tw[ix].unsqueeze(-1).expand_as(valid)[valid]
            binary = (F.binary_cross_entropy_with_logits(rank[valid], g[valid].float(), reduction="none")*weight_row).mean()
            term = rank.masked_fill(~valid, -40.)
            log_any = torch.logsumexp(torch.cat((term, torch.zeros_like(term[:, :1])), dim=1), dim=1)
            true = torch.logsumexp(rank.masked_fill(~g, -40.), dim=1)
            gain = torch.where(g.any(dim=1), true, 0.)
            ranking = ((log_any - gain)*tw[ix]).mean()
            loss = ids + .55 * binary + .50 * ranking
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        model_np = bundle(model, center, scale)
        model_np["temperature"], model_np["rank_temperature"] = _temperatures(model_np, deck["cal"])
        if incumbent is None:
            incumbent = _test(deck["reader_dev"], model_np, 0.)
        variants = {str(b): _test(deck["reader_dev"], model_np, b) for b in (.30, .60, .85, 1.)}
        blend, result = max(variants.items(), key=lambda a:(a[1]["strict"], -a[1]["wrong"], a[1]["r5_correct"]))
        key = (result["strict"], -result["wrong"], result["r5_correct"])
        trace.append({"epoch": epoch, "loss": float(loss.detach()), "variants": variants})
        if best is None or key > best[0]:
            parity = np.max(np.abs(linear(model_np, x[0, :20]) - model(tx[0, :20]).detach().cpu().numpy()))
            if parity >= .002 or not all(torch.any(initial[k] != w).item() for k, w in model.named_parameters()):
                raise RuntimeError("Reader CUDA/CPU parity or real parameter update failed")
            checkpoint = _checkpoint(out/"reader.pt", model, opt, rng, epoch=epoch)
            best = (key, float(blend), model_np, epoch, float(parity), checkpoint)
        print("CUDA reader", epoch, "old", incumbent["strict"], "best", blend, result, flush=True)
    _, blend, selected, epoch, parity, checkpoint = best
    settings = {"epoch": epoch, "blend": blend, "baseline": incumbent, "dev_history": trace, "optimizer_steps": 440,
                "groups": len(x), "train_role_counts": hist.tolist(), "cpu_cuda_error": parity,
                "reader_checkpoint_sha256": checkpoint, "selected_without_material": _test(
                    deck["reader_dev"], selected, blend, material=False)}
    atomic_json(out/"reader.json", plain(selected))
    atomic_json(out/"reader_training.json", settings)
    return settings


def train(args):
    out = args.out
    if not read_json(out/"gpu_preflight.json")["cuda_optimizer_step_verified"]:
        raise RuntimeError("GPU preflight failed")
    saved = out/"models"
    saved.mkdir(parents=True, exist_ok=True)
    deck, manifests = load_deck(out/"reader_data", args.atlas)
    rng = np.random.default_rng(155028)
    association = fit_edges(deck, saved, rng)
    fit_reader(deck, association, saved, rng)
    atomic_json(saved/"reader_manifest.json", {"source_sha256": digest(Path(__file__)),
                "data_freeze_sha256": digest(out/"reader_data_freeze.json"), "snapshot_commits": manifests,
                "layouts": {k: sorted({r["layout_id"] for r in v}) for k, v in deck.items()},
                "association_sha256": digest(saved/"association.json"), "reader_sha256": digest(saved/"reader.json")})


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"run")
    p.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    train(p.parse_args())

"""Equal-count pilot on ordinary, relative-anatomy, and targeted replay samples."""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax

from fit_v12 import (_checkpoint, _norm, _temperatures, _torch_setup, bundle, hydrate, load_deck,
                     net, plain, prepare_groups)
from readout_state import evaluate
from recovery_task import DEFAULT_PACK
from replay_v11 import ROOT
from study_v9 import atomic_json, digest, freeze, read_json, verify_shard
from track_state import features, from_logits, linear


COMMON, ADDED, UPDATES = 64, 16, 200
SEED = 802_578


def _roster(folder):
    meta = [v for f in sorted(folder.glob("*/commit.json")) for v in verify_shard(f.parent)[1]]
    rng = np.random.default_rng(SEED)
    plain_cases, anatomy_cases, failed_cases = [], [], []
    for r in meta:
        correct = r["arms"]["plain"]["selected_ids"][4] == 0
        if r["relative"]:
            anatomy_cases.append(r["layout_id"])
        elif not correct and "5_trace_b" in r["arms"]:
            failed_cases.append(r["layout_id"])
        elif correct:
            plain_cases.append(r["layout_id"])
    allocation = rng.permutation(plain_cases)
    return allocation[:COMMON].tolist(), {
        "ordinary": rng.choice(allocation[COMMON:], ADDED, replace=False).tolist(),
        "relative_anatomy": rng.choice(anatomy_cases, ADDED, replace=False).tolist(),
        "targeted_recovery": rng.choice(failed_cases, ADDED, replace=False).tolist()}


def _evaluate(rows, model):
    table = defaultdict(list)
    for r in rows:
        p, b = r["public"], r["base"]
        z = linear(model, r["features"])
        pred = from_logits(p, b, r["tracks"], softmax(z[:, :8]/model["temperature"], axis=1),
                           expit(z[:, 8]/model["rank_temperature"]), 1.)
        g = evaluate(p, r["labels"], pred["score"], 0., complete=r["complete"])
        table[r["layout_id"]].append([g["strict"], g["wrong_branch"], g["selected_ids"][4] == 0,
                                      g["selected_ids"][4] == 1])
    return np.asarray([table[k] for k in sorted(table)], dtype=np.int32)


def _fit(group, cal, dev, result_dir, label):
    import torch
    from torch.nn import functional as F

    _torch_setup(SEED)
    rng = np.random.default_rng(SEED)
    x, y, mask, _ = prepare_groups(group, rng)
    center, scale = _norm(x[mask])
    model = net(x.shape[-1], (72, 48, 9))
    opt = torch.optim.AdamW(model.parameters(), lr=.0018, weight_decay=.006, fused=True)
    x = torch.as_tensor(((x-center)/scale).clip(-6., 6.), device="cuda")
    y, mask = torch.as_tensor(y, device="cuda"), torch.as_tensor(mask, device="cuda")
    for _ in range(UPDATES):
        batch = torch.as_tensor(rng.integers(len(x), size=24), device="cuda")
        z = model(x[batch])
        labels, m = y[batch], mask[batch]
        rank = z[..., 8]
        target = labels == 0
        obj = F.cross_entropy(z[..., :8][m], labels[m])
        points = F.binary_cross_entropy_with_logits(rank[m], target[m].float())
        bg = torch.zeros_like(rank[:, :1])
        all_rank = torch.logsumexp(torch.cat((rank.masked_fill(~m, -40.), bg), dim=-1), dim=-1)
        best = torch.where(target.any(dim=1), torch.logsumexp(rank.masked_fill(~target, -40.), dim=1), 0.)
        loss = obj + .55*points + .5*(all_rank-best).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    exported = bundle(model, center, scale)
    exported["temperature"], exported["rank_temperature"] = _temperatures(exported, cal)
    ck = _checkpoint(result_dir/f"{label}.pt", model, opt, rng, steps=UPDATES)
    return _evaluate(dev, exported), ck


def run(args):
    out = args.out
    sub = out/"data_mix"
    association = out/"models"/"association.json"
    shared = hydrate(read_json(association))
    common, specs = _roster(out/"reader_data"/"fit")
    lock = freeze(sub, "freeze.json", {"source": digest(Path(__file__)), "association": digest(association),
        "fixed_updates": UPDATES, "common_layouts": common, "matched_added_layouts": specs, "fit_seed": SEED,
        "assay": "downstream reader only; fixed held-out development trajectories; shared fitted association"})
    deck, _ = load_deck(out/"reader_data", args.atlas)
    relevant = set(common).union(*specs.values())
    train = {}
    for row in deck["fit"]:
        if row["layout_id"] not in relevant or row["scope"] not in ("plain", "5_trace_b"):
            continue
        train[row["layout_id"], row["scope"]] = row
    for r in [*train.values(), *deck["cal"], *deck["reader_dev"]]:
        r["features"], r["tracks"] = features(r["public"], r["base"], r["surface"], shared)
    common_rows = [train[k, "plain"] for k in common]
    result, scored = {}, {}
    for label, ids in specs.items():
        key = "5_trace_b" if label == "targeted_recovery" else "plain"
        rows = common_rows + [train[k, key] for k in ids]
        values, check = _fit(rows, deck["cal"], deck["reader_dev"], sub, label)
        scored[label] = values
        result[label] = {"strict": int(values[..., 0].sum()), "wrong": int(values[..., 1].sum()),
                         "r5_correct": int(values[..., 2].sum()), "r5_artery": int(values[..., 3].sum()),
                         "layout_count": int(values.shape[0]), "trajectories": int(np.prod(values.shape[:2])),
                         "checkpoint_sha256": check}
        print("equal-count data mix", label, result[label], flush=True)
    for label in ("relative_anatomy", "targeted_recovery"):
        x = (scored[label] - scored["ordinary"]).mean(axis=1)
        rng = np.random.default_rng(SEED)
        q = np.quantile(x[rng.integers(len(x), size=(2500, len(x)))].mean(axis=1), [.025, .975], axis=0)
        result[label]["paired_strict_wrong_r5_artery"] = np.vstack((x.mean(axis=0), q)).tolist()
    atomic_json(sub/"results.json", plain({"freeze_sha256": lock, "results": result}))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"run")
    p.add_argument("--atlas", type=Path, default=DEFAULT_PACK)
    run(p.parse_args())

"""Fit and verify the shared v15 identity graph on a local CUDA device."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

import identity_v15 as ident
from study_v9 import atomic_json, digest


ROLES = {v: i for i, v in enumerate(ident.ROLE_LABELS)}
WIDTHS = {"local": 24, "shared": 20, "fused": 40}


@dataclass(frozen=True)
class Supervised:
    graph: ident.Graph
    region: np.ndarray
    label: np.ndarray
    instance: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class Pair:
    before: Supervised
    after: Supervised
    first: np.ndarray
    last: np.ndarray
    evidence: np.ndarray
    proposed: np.ndarray
    keys: np.ndarray


def _supervised(snapshot, graph):
    if graph is None:
        return None
    rows = graph.rows
    region, semantic, instance = (np.asarray(snapshot[k])[rows] for k in ("region", "label", "instance"))
    labels = np.fromiter((ROLES[int(x)] for x in semantic), dtype=np.int64, count=len(rows))
    # Each simulator instance contributes comparably despite repeated views.
    # Artifact and mixed echoes have no instance; count them by role/region.
    values = np.column_stack((region, semantic, instance))
    _, inverse, count = np.unique(values, return_inverse=True, return_counts=True, axis=0)
    weight = np.sqrt(1. / count[inverse]).astype(np.float32)
    weight /= weight.mean()
    return Supervised(graph, region.astype(int) - ident.REGIONS[0], labels, instance, weight)


def _addresses(snap):
    return np.column_stack((snap["frame"], snap["slot"])).astype(np.int64)


def _match(pair, before, after):
    if before is None or after is None or int(pair["region"]) not in ident.REGIONS:
        return None
    b, a = pair["before"], pair["after"]
    bg, ag = before.graph, after.graph
    region = int(pair["region"])
    bk, ak = _addresses(b)[bg.rows], _addresses(a)[ag.rows]
    ax = {tuple(key): i for i, key in enumerate(ak)}
    named = {tuple(keys) for keys in pair.get("keys", ()) if keys is not None}
    old = np.flatnonzero(before.region == region - ident.REGIONS[0])
    merit = bg.prior[old] + .30 * b["quality"][bg.rows[old]]
    keep = np.lexsort((-merit, bg.object_of_row[old]))
    _, start = np.unique(bg.object_of_row[old][keep], return_index=True)
    first = list(dict.fromkeys([i for i in old if tuple(bk[i]) in named] + old[keep[start]].tolist()))
    if not first:
        return None
    first = np.asarray(first, dtype=int)
    last = np.asarray([ax.get(tuple(bk[i]), -1) for i in first], dtype=int)
    exists = last >= 0
    if not exists.all():
        raise ValueError("An old echo address was absent after evidence acquisition")
    if np.any(before.label[first] != after.label[last]) or np.any(before.instance[first] != after.instance[last]):
        raise ValueError("A matched echo changed its offline identity")
    old_frame = int(np.max(b["frame"], initial=-1))
    start_tick = max(int(pair.get("start_tick", old_frame + 1)), old_frame + 1)
    before_view = np.column_stack((np.rint(b["station"]), np.rint(b["sector"]) % 24)).astype(int)
    after_view = np.column_stack((np.rint(a["station"]), np.rint(a["sector"]) % 24)).astype(int)
    b_track, a_track = np.asarray(b["track"]), np.asarray(a["track"])
    new = a["frame"] >= start_tick
    a_label, a_inst = np.asarray(a["label"]), np.asarray(a["instance"])
    evidence = []
    for u, v in zip(first, last, strict=True):
        i, j = bg.rows[u], ag.rows[v]
        past = {tuple(w) for w in before_view[b_track == b_track[i]]}
        arrived = new & (a_track == a_track[j])
        if before.instance[u] >= 0:
            arrived &= a_inst == before.instance[u]
        else:
            arrived &= a_label == a_label[j]
        evidence.append(any(tuple(w) not in past for w in after_view[arrived]))
    proposed = np.asarray([tuple(bk[i]) in named for i in first], dtype=bool)
    return Pair(before, after, first, last, np.asarray(evidence, dtype=bool), proposed, bk[first])


def _prepare(examples, include_pairs):
    prepared = []
    for ex in examples:
        terminal = ex["terminal"]
        term = _supervised(terminal, ident.build_graph(terminal))
        if term is None:
            raise ValueError("A complete identity training episode has no late echoes")
        pairs = []
        if include_pairs:
            for p in ex.get("pairs", ()):
                before, after = p["before"], p["after"]
                pair = _match(p, _supervised(before, ident.build_graph(before)),
                              _supervised(after, ident.build_graph(after)))
                if pair is not None:
                    pairs.append(pair)
        prepared.append((term, pairs))
    return prepared


def _pack(views, torch, device):
    place = {k: [] for k in ("x", "context", "prior", "roles", "object", "view", "view_weight",
                              "group", "group_region", "group_rep", "group_weight", "scene",
                              "row_region", "label", "weight")}
    row_base, group_base, object_base = 0, 0, 0
    starts = []
    for i, example in enumerate(views):
        g = example.graph
        starts.append(row_base)
        part = {"x": g.x, "context": g.context, "prior": g.prior, "roles": g.roles,
                "object": g.object_of_row + object_base, "view": g.view_rows + row_base,
                "view_weight": g.view_weight, "group": g.group_of_row + group_base,
                "group_region": g.group_region + 3*i, "group_rep": g.group_rep + row_base,
                "group_weight": g.group_weight, "scene": g.scene,
                "label": example.label, "weight": example.weight, "row_region": example.region + 3*i}
        for name, array in part.items():
            place[name].append(array)
        row_base += len(g.rows)
        group_base += len(g.group_region)
        object_base += g.object_count
    result = {k: torch.as_tensor(np.concatenate(v), device=device) for k, v in place.items()}
    result["n_object"] = object_base
    result["n_case"] = len(views)
    return result, np.asarray(starts, dtype=int)


def _norm(rows):
    z = np.concatenate(rows).astype(np.float32)
    return z.mean(axis=0), z.std(axis=0).clip(.055, None)


def _make_net(torch, norm, device):
    from torch import nn
    from torch.nn import functional as F

    local, shared, fused = (WIDTHS[k] for k in ("local", "shared", "fused"))

    class Reader(nn.Module):
        def __init__(self):
            super().__init__()
            self.encode1 = nn.Linear(ident.WIDTH, 48)
            self.encode2 = nn.Linear(48, local)
            self.scene = nn.Linear(3 * (local + ident.CASE_WIDTH), shared)
            self.fuse = nn.Linear(2 * local + shared + ident.CONTEXT_WIDTH, fused)
            self.rank = nn.Linear(fused, 1)
            self.role = nn.Linear(fused, 8)
            self.unknown1 = nn.Linear(shared + 6, 20)
            self.unknown2 = nn.Linear(20, 1)
            for name, data in norm.items():
                self.register_buffer(name, torch.as_tensor(data))
            self.register_buffer("region_eye", torch.eye(3))
            nn.init.normal_(self.rank.weight, std=.012)
            nn.init.zeros_(self.rank.bias)
            nn.init.normal_(self.role.weight, std=.012)
            nn.init.zeros_(self.role.bias)

        @staticmethod
        def weighted(values, index, weight, n):
            total = values.new_zeros((n, values.shape[-1])).index_add_(0, index, values * weight[:, None])
            denom = values.new_zeros(n).index_add_(0, index, weight)
            return total / denom[:, None].clamp(min=1.e-6)

        def forward(self, d):
            x = ((d["x"].float() - self.center) / self.scale).clamp(-6., 6.)
            u = ((d["context"] - self.context_center) / self.context_scale).clamp(-6., 6.)
            local = F.silu(self.encode2(F.silu(self.encode1(x))))
            view = d["view"]
            pooled = self.weighted(local[view], d["object"][view], d["view_weight"], d["n_object"])
            n = d["n_case"] * 3
            region = d["group_region"]
            regional = self.weighted(local[d["group_rep"]], region, d["group_weight"], n)
            shared = F.silu(self.scene(torch.cat((regional, d["scene"]), dim=1).reshape(d["n_case"], -1)))
            feature = torch.cat((local, pooled[d["object"]], u, shared[d["row_region"] // 3]), dim=1)
            h = F.silu(self.fuse(feature))
            logits = d["prior"] + ident.RESIDUAL * torch.tanh(self.rank(h)[:, 0] / 2.)
            roles = d["roles"].clamp(.005, 1.).log() + self.role(h)
            g = logits.new_full(region.shape, -1.e4).scatter_reduce_(0, d["group"], logits, reduce="amax")
            top = logits.new_full((n,), -1.e4).scatter_reduce_(0, region, g, reduce="amax")
            count = logits.new_zeros(n).index_add_(0, region, torch.ones_like(g))
            mean = logits.new_zeros(n).index_add_(0, region, g) / count.clamp(min=1.)
            summary = torch.cat((shared.repeat_interleave(3, dim=0), self.region_eye.repeat(d["n_case"], 1),
                                 top[:, None].clamp(-10., 10.) / 5., mean[:, None] / 5.,
                                 count[:, None].log1p() / 5.), dim=1)
            unknown = self.unknown2(F.silu(self.unknown1(summary)))[:, 0]
            return logits, roles, g, unknown

    return Reader().to(device)


def _losses(data, outputs, torch):
    from torch.nn import functional as F

    logits, roles, group_score, unknown = outputs
    region = data["row_region"]
    positive = data["label"] == 0
    top = unknown.new_full(unknown.shape, -1.e4).scatter_reduce_(
        0, data["group_region"], group_score, reduce="amax").maximum(unknown).detach()
    all_mass = (unknown - top).exp().index_add(0, data["group_region"],
                                              (group_score - top[data["group_region"]]).exp())
    target_best = unknown.new_full(unknown.shape, -1.e4).scatter_reduce_(
        0, region, logits.masked_fill(~positive, -1.e4), reduce="amax")
    rival_best = unknown.new_full(unknown.shape, -1.e4).scatter_reduce_(
        0, region, logits.masked_fill(positive, -1.e4), reduce="amax")
    has_target = target_best > -1.e3
    target = torch.where(has_target, target_best, unknown)
    region_weights = (1. + .40 * (torch.arange(len(unknown), device=logits.device) % 3 == 2))
    rank = ((all_mass.log() + top - target) * region_weights).mean()
    hard = F.softplus(.40 + rival_best - target_best.masked_fill(~has_target, 1.e4))
    hard = (hard * region_weights).mean()
    weight = data["weight"] * (1. + .55 * positive)
    identity = F.binary_cross_entropy_with_logits(logits, positive.float(), reduction="none")
    role = F.cross_entropy(roles, data["label"], reduction="none")
    return rank + .24 * hard + .15 * (identity * weight).mean() + .24 * (role * weight).mean()


def _paired_loss(pairs, torch, device, net):
    from torch.nn import functional as F

    views = [x for pair in pairs for x in (pair.before, pair.after)]
    data, starts = _pack(views, torch, device)
    logits, roles, _, _ = net(data)
    before = np.concatenate([pair.first + offset for pair, offset in zip(pairs, starts[::2], strict=True)])
    after = np.concatenate([pair.last + offset for pair, offset in zip(pairs, starts[1::2], strict=True)])
    keys = torch.as_tensor(np.column_stack((before, after)), device=device)
    evidence = torch.as_tensor(np.concatenate([p.evidence for p in pairs]), device=device)
    proposed = torch.as_tensor(np.concatenate([p.proposed for p in pairs]), device=device)
    truth = data["label"][keys[:, 1]]
    b, a = logits[keys].unbind(dim=1)
    difference = a - b.detach()
    target = (truth == 0).float()
    weight = 1. + 1.5 * proposed
    change = F.softplus(.22 - difference * (2. * target - 1.))
    stable = F.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none", beta=.25)
    sem = F.cross_entropy(roles[keys[:, 1]], truth, reduction="none")
    consistency = torch.where(evidence, .60 * change, .42 * stable)
    classification = F.binary_cross_entropy_with_logits(a, target, reduction="none")
    return ((.22 * classification + .16 * sem + consistency) * weight).sum() / weight.sum()


def _bundle(net, norm, epoch):
    return dict(format=ident.FORMAT, width=ident.WIDTH, widths=WIDTHS, epoch=epoch, **norm,
                weights={k: v.detach().cpu().numpy().copy() for k, v in net.named_parameters()})


def _distribution(model, prepared):
    result = []
    for terminal, _ in prepared:
        g = terminal.graph
        z, _, shared = ident.numpy_encode(model, g)
        scores, null = ident.numpy_pool(model, g, z, shared)
        best = np.full(len(scores), len(z), dtype=int)
        tied = np.flatnonzero(z == scores[g.group_of_row])
        np.minimum.at(best, g.group_of_row[tied], tied)
        for region in range(3):
            ix = np.flatnonzero(g.group_region == region)
            candidate = scores[ix]
            values = np.r_[candidate, null[region]]
            y = np.r_[terminal.label[best[ix]] == 0, not np.any(terminal.label[best[ix]] == 0)]
            result.append((values, y, terminal.label[best[ix]]))
    return result


def _case_metrics(labels):
    labels = np.asarray(labels)
    result = {
        "layouts": len(labels), "strict": int(np.all(labels == 0, axis=1).sum()),
        "wrong": int(np.any(labels > 0, axis=1).sum()), "correct_regions": int((labels == 0).sum()),
    }
    for r, y in enumerate(labels.T, 1):
        result |= {f"r{r}_correct": int((y == 0).sum()), f"r{r}_artery": int((y == 1).sum()),
                   f"r{r}_wrong_branch": int((y > 0).sum()), f"r{r}_reject": int((y == -99).sum()),
                   f"r{r}_artifact": int(np.isin(y, (-1, -2)).sum())}
    return result


def _baselines(examples):
    return np.asarray([e["terminal"]["baseline_ids"] for e in examples], dtype=int)


def _decisions(distributions, baselines, temperature, threshold):
    y = baselines.copy()
    for i, (logits, _, role) in enumerate(distributions):
        case, r = divmod(i, 3)
        label = -99
        if len(role):
            p = softmax(logits / temperature)
            top = int(np.argmax(p[:-1]))
            if p[top] >= max(p[-1], threshold):
                label = ident.ROLE_LABELS[int(role[top])]
        y[case, r+ident.REGIONS[0]] = label
    return y


def _criterion(metrics):
    return (metrics["strict"], -metrics["wrong"], metrics["r6_correct"],
            metrics["correct_regions"], -metrics["r6_artery"])


def _calibrate(model, edges, examples):
    predictions = _distribution(model, edges)

    def objective(t):
        terms = [logsumexp(z / t) - logsumexp((z / t)[y]) for z, y, _ in predictions]
        return float(np.mean(terms))

    t = float(minimize_scalar(objective, bounds=(.35, 4.), method="bounded").x)
    base = _baselines(examples)
    trials = [(cut, _case_metrics(_decisions(predictions, base, t, cut)))
              for cut in (.00, .035, .06, .09, .12, .16, .21, .28, .36, .46, .58, .70, .82)]
    cut, measures = max(trials, key=lambda v: (*_criterion(v[1]), v[0]))
    model |= {"temperature": t, "threshold": cut}
    return {"threshold": cut, "temperature": t, "calibration_nll": objective(t), "outcome": measures,
            "threshold_sweep": [{ "threshold": k, "strict": v["strict"], "wrong": v["wrong"]}
                                for k, v in trials]}


def evaluate_reader(model, examples, *, include_pairs=True):
    """Grade v13 and v15 on the identical frozen public snapshots."""
    cases, old_ids, new_ids, evidence = [], [], [], Counter()
    paired_rows = []
    for ex, base in zip(examples, _baselines(examples), strict=True):
        term = ex["terminal"]
        result = ident.score_snapshot(model, term)
        selection = base.copy()
        for region in ident.REGIONS:
            row = result["selected_rows"][region]
            selection[region] = int(term["label"][row]) if row >= 0 else -99
        case = ex["case"]["identifier"] if isinstance(ex["case"], dict) else ex["case"]
        old_ids.append(base)
        new_ids.append(selection)
        cases.append({"case": case, "v13": base.tolist(), "v15": selection.tolist(),
                      "candidate_rows": result["selected_rows"].tolist(),
                      "unknown_probability": result["unknown_probability"].tolist()})
        if not include_pairs:
            continue
        for pair in ex.get("pairs", ()):
            b, a = pair["before"], pair["after"]
            bg, ag = ident.build_graph(b), ident.build_graph(a)
            m = _match(pair, _supervised(b, bg), _supervised(a, ag))
            if m is None:
                continue
            v0 = ident.score_snapshot(model, b, graph=bg)
            v1 = ident.score_snapshot(model, a, graph=ag)
            first, last = bg.rows[m.first], ag.rows[m.last]
            old_delta = a["prior"][last] - b["prior"][first]
            new_delta = v1["identity_score"][last] - v0["identity_score"][first]
            direction = np.where(m.before.label[m.first] == 0, 1., -1.)
            evidence.update(programs=1, matched=len(first), measured=int(m.evidence.sum()),
                            proposed=int(m.proposed.sum()), v15_direction=int((m.evidence & (direction*new_delta>0)).sum()),
                            v13_direction=int((m.evidence & (direction*old_delta>0)).sum()))
            for i in np.flatnonzero(m.proposed):
                paired_rows.append({"case": case, "region": int(pair["region"])+1, "key": m.keys[i].tolist(),
                                    "label": int(b["label"][first[i]]), "new_same_instance_view": bool(m.evidence[i]),
                                    "v13_delta": float(old_delta[i]), "v15_delta": float(new_delta[i])})
    return {"v13": _case_metrics(old_ids), "v15": _case_metrics(new_ids), "cases": cases,
            "paired": dict(evidence), "paired_focus": paired_rows}


def _parity(torch, device, model, reader, examples):
    data, starts = _pack(examples, torch, device)
    with torch.no_grad():
        gpu_z, gpu_role, gpu_pool, gpu_unknown = [x.detach().cpu().numpy() for x in reader(data)]
    row, role, scores, unknown = [], [], [], []
    np_start = 0
    for offset, x in zip(starts, examples, strict=True):
        z, semantic, shared = ident.numpy_encode(model, x.graph)
        p, u = ident.numpy_pool(model, x.graph, z, shared)
        row.append(np.max(np.abs(z - gpu_z[offset:offset+len(z)])))
        role.append(np.max(np.abs(semantic - gpu_role[offset:offset+len(z)])))
        scores.append(np.max(np.abs(p - gpu_pool[np_start:np_start+len(p)])))
        np_start += len(p)
        unknown.append(u)
    errors = {"identity": float(max(row)), "semantic": float(max(role)),
              "object": float(max(scores)), "unknown": float(np.max(np.abs(np.concatenate(unknown)-gpu_unknown)))}
    if max(errors.values()) > 2.e-4:
        raise RuntimeError(f"Serialized NumPy inference disagrees with CUDA: {errors}")
    return errors


def fit_reader(splits, out, seed, *, epochs=18, batch_size=12):
    """Train with terminal and paired histories and commit a verified reader."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the shared identity reader")
    if min(len(splits[k]) for k in ("fit", "cal", "dev")) == 0:
        raise ValueError("The three reader cohorts must all have observations")
    if epochs < 1 or batch_size < 1:
        raise ValueError("Reader epochs and batch size must be positive")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    rng = np.random.default_rng(seed)
    started = monotonic()
    prepared = {k: _prepare(splits[k], include_pairs=k == "fit") for k in ("fit", "cal", "dev")}
    train, cal, dev = (prepared[k] for k in ("fit", "cal", "dev"))
    center, scale = _norm([term.graph.x[::11] for term, _ in train])
    context_center, context_scale = _norm([term.graph.context[::11] for term, _ in train])
    norm = {"center": center, "scale": scale, "context_center": context_center, "context_scale": context_scale}
    reader = _make_net(torch, norm, device)
    initial = {k: b.detach().clone() for k, b in reader.named_parameters()}
    opt = torch.optim.AdamW(reader.parameters(), lr=.0017, weight_decay=.006, fused=True)
    n_batches = (len(train) + batch_size - 1) // batch_size
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*n_batches, eta_min=.00033)
    masses = Counter()
    history, best = [], None
    checkpoints = {1, min(3, epochs), min(6, epochs), min(10, epochs), min(14, epochs), epochs}
    dev_base = _baselines(splits["dev"])
    print("v15 shared reader prepared", {k: len(v) for k, v in prepared.items()},
          "on", torch.cuda.get_device_name(device), flush=True)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        reader.train()
        loss_sums = torch.zeros(2, device=device)
        for ix in np.array_split(rng.permutation(len(train)), n_batches):
            terms = [train[i][0] for i in ix]
            pairs = [train[i][1][(epoch-1) % len(train[i][1])] for i in ix if train[i][1]]
            data, _ = _pack(terms, torch, device)
            terminal_loss = _losses(data, reader(data), torch)
            paired_loss = _paired_loss(pairs, torch, device, reader) if pairs else terminal_loss.new_zeros(())
            loss = terminal_loss + .55 * paired_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 3.)
            opt.step()
            schedule.step()
            loss_sums += torch.stack((terminal_loss.detach(), paired_loss.detach()))
            masses.update(steps=1, terminal_histories=len(terms), paired_histories=len(pairs),
                          terminal_echoes=len(data["label"]), fixed_address_echoes=sum(len(p.first) for p in pairs))
        if epoch not in checkpoints:
            continue
        reader.eval()
        model = _bundle(reader, norm, epoch)
        calibration = _calibrate(model, cal, splits["cal"])
        distribution = _distribution(model, dev)
        measured = _case_metrics(_decisions(distribution, dev_base, model["temperature"], model["threshold"]))
        terminal_loss, paired_loss = (loss_sums / n_batches).tolist()
        record = {"epoch": epoch, "terminal_loss": terminal_loss, "paired_loss": paired_loss,
                  "dev": measured, "calibration": calibration, "elapsed_s": monotonic()-started}
        history.append(record)
        key = _criterion(measured)
        print("v15 reader", epoch, "strict", measured["strict"], "wrong", measured["wrong"],
              "R6", measured["r6_correct"], "temperature", round(model["temperature"], 3),
              "cut", model["threshold"], flush=True)
        if best is None or key > best[0]:
            errors = _parity(torch, device, model, reader, [d[0] for d in dev[:2]])
            best = key, model, record, errors
    _, model, selected, parity = best
    changes = {k: float((b - initial[k]).abs().max().item()) for k, b in reader.named_parameters()}
    if min(changes.values()) <= 0. or not masses["paired_histories"]:
        raise RuntimeError("The shared reader needs parameter updates and matched evidence histories")
    payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in model.items() if k != "weights"}
    payload["weights"] = {k: v.tolist() for k, v in model["weights"].items()}
    atomic_json(out/"reader.json", payload)
    disk = ident.load_reader(out/"reader.json")
    verified = evaluate_reader(disk, splits["dev"])
    if verified["v15"] != selected["dev"]:
        raise RuntimeError("Re-loaded JSON reader changed same-history checkpoint results")
    pair_availability = Counter(len(pair.first) for _, pairs in train for pair in pairs)
    report = {
        "device": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "seed": seed, "epochs": epochs, "selected_epoch": selected["epoch"],
        "elapsed_s": monotonic()-started, "reader_sha256": digest(out/"reader.json"),
        "max_cuda_bytes": torch.cuda.max_memory_allocated(device), "weights_max_change": changes,
        "cuda_numpy_max_abs": parity, "seen": dict(masses),
        "unique_fit": {"complete_histories": len(train), "paired_histories": sum(pair_availability.values()),
                       "matched_old_echoes": sum(k*v for k, v in pair_availability.items())},
        "feature_schema": {"v13_observation_width": ident.WIDTH, "public_graph_context": ident.CONTEXT_WIDTH,
                           "case_state_per_region": ident.CASE_WIDTH, "late_regions": [4, 5, 6]},
        "history": history, "same_snapshot_development": verified,
    }
    atomic_json(out/"reader_training.json", report)
    return report

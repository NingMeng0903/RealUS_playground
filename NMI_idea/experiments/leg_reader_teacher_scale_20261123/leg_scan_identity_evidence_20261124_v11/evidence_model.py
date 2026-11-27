"""Learn public priors, action-conditioned evidence and continuation outcomes."""

from pathlib import Path

import numpy as np
from scipy.special import softmax

from evidence import update_belief
from study_v9 import atomic_json, digest, read_json, verify_shard


def _normalize(x, center, scale):
    return np.clip((x - center) / scale, -6., 6.)


def _forward(x, params, name):
    hidden = np.tanh(x @ params[name+".0.weight"].T + params[name+".0.bias"])
    return hidden @ params[name+".2.weight"].T + params[name+".2.bias"]


def prior_for(model, state):
    logits = _forward(_normalize(state.features, *model["history"]), model["parameters"], "prior")
    if state.anchors[1] is None:
        logits[1] = -np.inf
    return softmax(logits / model["prior_temperature"])


def infer(model, actions):
    inputs = _normalize(np.stack([a.features for a in actions]), *model["action"])
    logits = _forward(inputs, model["parameters"], "outcomes")
    return softmax(logits[:, :12].reshape(-1, 3, 4) / model["event_temperature"], axis=2), logits[:, 12:]


def load_model(path):
    result = read_json(Path(path))
    result["parameters"] = {k: np.asarray(v, np.float32) for k, v in result["parameters"].items()}
    for key in ("history", "action"):
        result[key] = tuple(np.asarray(a, np.float32) for a in result[key])
    return result


def load_shards(shards):
    output = {}
    for path in shards:
        _, info = verify_shard(path.parent)
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["features"].size == 0:
                continue
            x = np.stack([arrays["layout_index"], arrays["row"]], axis=1)
            fields = {k: arrays[k] for k in ("features", "context", "event", "identity", "deltas")}
            fields["layout"] = np.array([info[i]["layout_id"] for i in x[:, 0]])
            fields["key"] = np.array([f"{info[i]['layout_id']}/{j}" for i, j in x])
            for k, data in fields.items():
                output.setdefault(k, []).append(data)
    return {k: np.concatenate(v) for k, v in output.items()}


def _temperature(logits, labels):
    cuts = (.70, .90, 1., 1.2, 1.5, 2.)
    return min(cuts, key=lambda t: -np.log(softmax(logits/t, axis=1)[np.arange(len(labels)), labels].clip(1.e-9)).mean())


def fit(shards, output, seed=26112411, epochs=15):
    import torch
    from torch import nn
    from torch.nn import functional as F

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    raw = load_shards(shards)
    by_layout = np.unique(raw["layout"])
    if len(by_layout) < 24:
        raise RuntimeError("At least 24 independent simulation layouts are needed")
    dev_layouts = rng.permutation(by_layout)[:max(8, len(by_layout)//5)]
    retained = np.isin(raw["layout"], dev_layouts)
    train = np.flatnonzero(~retained)
    dev = np.flatnonzero(retained)
    _, unique = np.unique(raw["key"], return_index=True)
    distinct_train = unique[~retained[unique]]
    distinct_dev = unique[retained[unique]]
    centers = raw["context"][distinct_train].mean(0), raw["features"][train].mean(0)
    scales = raw["context"][distinct_train].std(0).clip(.05), raw["features"][train].std(0).clip(.05)
    X = torch.as_tensor(_normalize(raw["context"], centers[0], scales[0]), dtype=torch.float32, device="cuda")
    A = torch.as_tensor(_normalize(raw["features"], centers[1], scales[1]), dtype=torch.float32, device="cuda")
    Y = torch.as_tensor(raw["identity"], dtype=torch.long, device="cuda")
    E = torch.as_tensor(raw["event"], dtype=torch.long, device="cuda")
    D = torch.as_tensor(raw["deltas"], dtype=torch.float32, device="cuda")
    models = nn.ModuleDict({"prior": nn.Sequential(nn.Linear(X.shape[1], 32), nn.Tanh(), nn.Linear(32, 3)),
                            "outcomes": nn.Sequential(nn.Linear(A.shape[1], 48), nn.Tanh(), nn.Linear(48, 16))}).cuda()
    opt = torch.optim.AdamW(models.parameters(), lr=.0015, weight_decay=.001, fused=True)
    start = {k: v.detach().clone() for k, v in models.named_parameters()}
    best, train_trace, steps = np.inf, [], 0
    for epoch in range(1, epochs+1):
        order = rng.permutation(train)
        for a in np.array_split(order, max(1, int(np.ceil(len(order)/96)))):
            # Prior supervision is sampled per distinct public prefix.
            h = rng.choice(distinct_train, min(len(distinct_train), len(a)), replace=False)
            logits = models["prior"](X[h])
            other = models["outcomes"](A[a])
            conditional = other[:, :12].reshape(-1, 3, 4)[torch.arange(len(a), device="cuda"), Y[a]]
            delta_loss = (F.smooth_l1_loss(other[:, 12:], D[a], reduction="none") *
                          (1. + .4 * D[a].abs())).mean()
            loss = F.cross_entropy(logits, Y[h]) + .85 * F.cross_entropy(conditional, E[a]) + 2. * delta_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
        with torch.no_grad():
            belief = models["prior"](X[distinct_dev])
            other = models["outcomes"](A[dev])
            conditional = other[:, :12].reshape(-1, 3, 4)[torch.arange(len(dev), device="cuda"), Y[dev]]
            loss0 = float(F.cross_entropy(belief, Y[distinct_dev]))
            loss1 = float(F.cross_entropy(conditional, E[dev]))
            loss2 = float(F.mse_loss(other[:, 12:], D[dev]))
            combined = loss0 + .85 * loss1 + loss2
            train_trace.append({"epoch": epoch, "prior_nll": loss0, "event_nll": loss1, "delta_mse": loss2})
        if combined < best:
            best = combined
            selected = epoch
            states = {k: t.detach().cpu().clone() for k, t in models.state_dict().items()}
            temporary = output / "checkpoint.pending"
            torch.save({"model": states, "optimizer": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "numpy_rng": rng.bit_generator.state,
                        "selection_validation_epoch": epoch}, temporary)
            temporary.replace(output / "checkpoint.pt")
    models.load_state_dict(states)
    with torch.no_grad():
        x = models["prior"](X[distinct_dev]).cpu().numpy()
        pred = models["outcomes"](A[dev]).cpu().numpy()
    likelihood = pred[:, :12].reshape(-1, 3, 4)
    conditional = likelihood[np.arange(len(dev)), raw["identity"][dev]]
    p_temp = _temperature(x, raw["identity"][distinct_dev])
    e_temp = _temperature(conditional, raw["event"][dev])
    bundle = {"history": [a.tolist() for a in (centers[0], scales[0])],
              "action": [a.tolist() for a in (centers[1], scales[1])],
              "parameters": {k: t.tolist() for k, t in states.items()},
              "prior_temperature": p_temp, "event_temperature": e_temp}
    atomic_json(output / "model.json", bundle)
    live = load_model(output / "model.json")
    np.testing.assert_allclose(_forward(_normalize(raw["context"][distinct_dev], *live["history"]), live["parameters"], "prior"),
                               x, atol=2.e-5)
    np.testing.assert_allclose(_forward(_normalize(raw["features"][dev], *live["action"]), live["parameters"], "outcomes"),
                               pred, atol=2.e-5)
    prior_hist = np.bincount(raw["identity"][distinct_train], minlength=3) + 1.
    event_hist = np.bincount(raw["event"][train], minlength=4) + 1.
    p = softmax(x/p_temp, axis=1)
    posterior = []
    lookup = {raw["key"][j]: j0 for j0, j in enumerate(distinct_dev)}
    for i, row in enumerate(dev):
        b = p[lookup[raw["key"][row]]]
        posterior.append(update_belief(b, softmax(likelihood[i]/e_temp, axis=1), raw["event"][row]))
    posterior = np.asarray(posterior)
    prior_branch = p[[lookup[raw["key"][row]] for row in dev]]
    manifest = {"device": str(A.device), "gpu": torch.cuda.get_device_name(), "cuda_optimizer_steps": steps,
                "fit_layout_ids": sorted(set(by_layout) - set(dev_layouts)), "validation_layout_ids": sorted(dev_layouts),
                "labels_fit": len(train), "labels_validation": len(dev), "prefixes_fit": len(distinct_train),
                "prefixes_validation": len(distinct_dev), "all_identity_rows": np.bincount(raw["identity"], minlength=3).tolist(),
                "all_event_rows": np.bincount(raw["event"], minlength=4).tolist(),
                "prior_nll": float(-np.log(p[np.arange(len(x)), raw["identity"][distinct_dev]].clip(1.e-9)).mean()),
                "base_prior_nll": float(-np.log((prior_hist/prior_hist.sum())[raw["identity"][distinct_dev]]).mean()),
                "event_nll": float(-np.log(softmax(conditional/e_temp, axis=1)[np.arange(len(dev)), raw["event"][dev]].clip(1.e-9)).mean()),
                "base_event_nll": float(-np.log((event_hist/event_hist.sum())[raw["event"][dev]]).mean()),
                "paired_prior_accuracy": float((prior_branch.argmax(axis=1) == raw["identity"][dev]).mean()),
                "paired_updated_accuracy": float((posterior.argmax(axis=1) == raw["identity"][dev]).mean()),
                "delta_mse": np.square(pred[:, 12:] - raw["deltas"][dev]).mean(axis=0).tolist(),
                "zero_delta_mse": np.square(raw["deltas"][dev]).mean(axis=0).tolist(),
                "selected_epoch": selected, "trace": train_trace,
                "actor_weights_changed": all(not torch.equal(v, start[k]) for k, v in models.named_parameters()),
                "model_sha256": digest(output / "model.json"), "checkpoint_sha256": digest(output / "checkpoint.pt")}
    if not manifest["actor_weights_changed"]:
        raise RuntimeError("CUDA optimizer updates did not reach all evidence model parameters")
    atomic_json(output / "training.json", manifest)
    return output / "model.json"

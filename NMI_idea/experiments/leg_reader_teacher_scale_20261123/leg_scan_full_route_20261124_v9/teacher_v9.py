"""Fit public, reference-relative action gains from measured paired continuations."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn


ACTIONS = ("continue", "shift_left", "shift_right", "hold", "revisit_left", "revisit_right", "finish")
FORMAT = "public_paired_action_gain_v9"
WIDTH = 32
ATTEMPTED_EPOCHS = 64
TINY_EPOCHS = 24
LEARNING_RATE = .002
BATCH_SIZE = 64


def _public(row, n_actions):
    observation = np.asarray(row["observations"], dtype=np.float32)
    history = np.asarray(row["history"], dtype=np.float32)
    mask = np.asarray(row["mask"], dtype=bool)
    if not (observation.ndim == history.ndim == 1 and observation.size and history.size):
        raise ValueError("The teacher expects a single public observation and its previous actor state")
    if not (np.isfinite(observation).all() and np.isfinite(history).all()):
        raise ValueError("Teacher features must be finite")
    reference = int(row["reference"])
    if mask.shape != (n_actions,) or not 0 <= reference < n_actions or not mask[reference]:
        raise ValueError("The paired reference must be legal at the observed state")
    return np.concatenate((observation, history), dtype=np.float32), mask, reference


def _measured(row, legal, reference, n_actions):
    """Aggregate each measured state/action over paired tapes, keeping every sign."""
    values = row["label_differences"]
    if not isinstance(values, Mapping):
        raise ValueError("Expected actual return differences indexed by action")
    nominated = {int(a) for a in row["actions"]}
    means = np.zeros(n_actions, dtype=np.float32)
    measured = np.zeros(n_actions, dtype=bool)
    counts = np.zeros(n_actions, dtype=np.int32)
    for index, tapes in values.items():
        action = int(index)
        if not 0 <= action < n_actions or not legal[action]:
            raise ValueError("A paired return difference refers to an illegal action")
        if action != reference and action not in nominated:
            raise ValueError("A paired return difference refers to an unmeasured action")
        paired = np.asarray(tapes, dtype=np.float64)
        if paired.ndim != 1 or not np.isfinite(paired).all() or not paired.size:
            raise ValueError("Expected at least one finite paired return difference")
        if action == reference:
            if np.any(paired != 0.):
                raise ValueError("The paired reference difference must be zero")
            continue
        means[action] = paired.mean()
        measured[action] = True
        counts[action] = len(paired)
    return means, measured, counts


def _deck(rows, n_actions):
    observed = []
    feature_widths = set()
    for row in rows:
        layout = row["layout"]
        if not isinstance(layout, str) or not layout:
            raise ValueError("Teacher measurements require a layout identifier")
        features, legal, reference = _public(row, n_actions)
        target, measured, counts = _measured(row, legal, reference, n_actions)
        if measured.any():
            feature_widths.add((len(row["observations"]), len(row["history"])))
            observed.append((str(layout), features, legal, reference, target, measured, counts))
    if len(feature_widths) > 1:
        raise ValueError("Teacher observation and actor history widths must be consistent")
    if not observed:
        raise ValueError("Teacher received no measured paired comparisons")
    columns = tuple(np.asarray(col) for col in zip(*observed, strict=True))
    return columns, feature_widths.pop()


def _split(layout, seed):
    distinct = set(layout.tolist())
    ordered = sorted(distinct, key=lambda item: hashlib.sha256(f"{seed}/{item}".encode()).digest())
    n_val = max(1, len(ordered) // 5) if len(ordered) > 1 else 0
    val_layouts, fit_layouts = ordered[:n_val], ordered[n_val:]
    val = np.flatnonzero(np.isin(layout, val_layouts))
    fit = np.flatnonzero(np.isin(layout, fit_layouts))
    assert not (set(fit_layouts) & set(val_layouts))
    return fit, val, fit_layouts, val_layouts


class _Teacher(nn.Module):
    """Learn public action potentials. Only measured differences supervise them."""

    def __init__(self, n_features, n_actions):
        super().__init__()
        self.first = nn.Linear(n_features + n_actions, WIDTH)
        self.second = nn.Linear(WIDTH, WIDTH)
        self.potential = nn.Linear(WIDTH, n_actions)
        nn.init.normal_(self.potential.weight, std=.06)
        nn.init.zeros_(self.potential.bias)

    def forward(self, features, legal, reference):
        state = torch.cat((features, legal.float()), dim=1)
        hidden = self.first(state).tanh()
        potential = self.potential(self.second(hidden).tanh())
        relative = potential - potential.gather(1, reference[:, None])
        return torch.where(legal, relative, 0.)


def _cpu_parameters(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _quality(prediction, target, measured):
    y = target[measured]
    p = prediction[measured]
    counts = {"measured": len(y), "positive": int((y > 0.).sum()),
              "negative": int((y < 0.).sum()), "zero": int((y == 0.).sum())}
    delta = np.square(p - y)
    null = np.square(y)
    if not len(y):
        return counts | {"mse": None, "zero_mse": None, "signed_accuracy": None, "by_sign": {}}
    groups = {}
    for sign, take in (("positive", y > 0.), ("negative", y < 0.), ("zero", y == 0.)):
        groups[sign] = {"n": int(take.sum()), "mse": float(delta[take].mean()) if take.any() else None,
                        "zero_mse": float(null[take].mean()) if take.any() else None,
                        "mean_actual": float(y[take].mean()) if take.any() else None,
                        "mean_predicted": float(p[take].mean()) if take.any() else None}
    signed = y != 0.
    return counts | {"mse": float(delta.mean()), "zero_mse": float(null.mean()),
                     "signed_accuracy": float(np.mean(np.sign(p[signed]) == np.sign(y[signed]))) if signed.any() else None,
                     "by_sign": groups}


def _atomic(path, content):
    with NamedTemporaryFile(dir=path.parent, prefix="." + path.name, suffix=".pending", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _save(output, bundle, metrics, checkpoint):
    stream = BytesIO()
    torch.save(checkpoint, stream)
    data = stream.getvalue()
    metrics["checkpoint_sha256"] = hashlib.sha256(data).hexdigest()
    weights = _json_bytes(bundle)
    metrics["bundle_sha256"] = hashlib.sha256(weights).hexdigest()
    _atomic(output / "teacher_checkpoint.pt", data)
    _atomic(output / "teacher_bundle.json", weights)
    _atomic(output / "teacher_fit.json", _json_bytes(metrics))


def teacher_q(row, bundle):
    """Predict signed paired gain on CPU. Consumers apply the measured mask."""
    if bundle["format"] != FORMAT:
        raise ValueError("Unsupported v9 teacher bundle")
    features, legal, reference = _public(row, bundle["n_actions"])
    if (len(row["observations"]), len(row["history"])) != (bundle["obs_dim"], bundle["hidden_dim"]):
        raise ValueError("Teacher features differ from the frozen public observation and history widths")
    center = np.asarray(bundle["center"], dtype=np.float32)
    scale = np.asarray(bundle["scale"], dtype=np.float32)
    state = np.concatenate((np.clip((features - center) / scale, -9., 9.), legal), dtype=np.float32)
    weights = bundle["weights"]

    def linear(x, name):
        return (np.asarray(weights[name + ".weight"], dtype=np.float32) @ x
                + np.asarray(weights[name + ".bias"], dtype=np.float32))

    hidden = np.tanh(linear(state, "first"))
    potential = linear(np.tanh(linear(hidden, "second")), "potential")
    return np.where(legal, potential - potential[reference], 0.)


def fit_teacher(rows, n_actions, seed, output):
    """Train and save the supervised CUDA teacher and its layout-held-out audit."""
    if n_actions != len(ACTIONS):
        raise ValueError("v9 teacher action order must match the seven shared skills")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v9 teacher optimization")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    columns, (obs_dim, hidden_dim) = _deck(rows, n_actions)
    layout, features, legal, reference, values, measured, counts = columns
    fit, val, fit_layouts, val_layouts = _split(layout, seed)
    center = features[fit].mean(axis=0)
    scale = np.maximum(features[fit].std(axis=0), .05)
    normalized = np.clip((features - center) / scale, -9., 9.)
    tensors = [torch.as_tensor(column, device="cuda") for column in (normalized, legal, reference, values, measured)]
    x, mask, ref, actual, supervised = tensors
    rng = np.random.default_rng(seed)
    epochs = ATTEMPTED_EPOCHS if len(fit_layouts) >= 4 else TINY_EPOCHS
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256()
    for column in columns:
        fingerprint.update(str((column.shape, column.dtype)).encode())
        fingerprint.update(np.ascontiguousarray(column).tobytes())
    data_sha = fingerprint.hexdigest()
    best, trace, steps = None, [], 0
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = _Teacher(features.shape[1], n_actions).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.02)
        initial = _cpu_parameters(model)

        def loss_for(take):
            ix = torch.as_tensor(take, device="cuda")
            p = model(x[ix], mask[ix], ref[ix])
            return (p[supervised[ix]] - actual[ix][supervised[ix]]).square().mean()

        for epoch in range(1, epochs + 1):
            order = rng.permutation(fit)
            for begin in range(0, len(order), BATCH_SIZE):
                loss = loss_for(order[begin:begin + BATCH_SIZE])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.)
                optimizer.step()
                steps += 1
            with torch.no_grad():
                fit_loss = float(loss_for(fit))
                val_loss = float(loss_for(val)) if len(val) else None
            trace.append({"epoch": epoch, "fit_mse": fit_loss, "validation_mse": val_loss})
            if (len(val) and (best is None or val_loss < best["validation_mse"])) or (not len(val) and epoch == epochs):
                # Serialized bytes retain the exact optimizer moments at this checkpoint.
                optimizer_buffer = BytesIO()
                torch.save(optimizer.state_dict(), optimizer_buffer)
                best = {"model": _cpu_parameters(model),
                        "optimizer_serialized": optimizer_buffer.getvalue(),
                        "torch_rng": torch.get_rng_state().clone(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "numpy_rng": rng.bit_generator.state, "epoch": epoch, "steps": steps,
                        "validation_mse": val_loss}
        model.load_state_dict(best["model"])
        updated = [key for key in initial if not torch.equal(best["model"][key], initial[key])]
        if len(updated) != len(initial) or not optimizer.state or not steps:
            raise RuntimeError("Every deployed teacher layer must receive a CUDA update")
        with torch.no_grad():
            prediction = model(x, mask, ref).cpu().numpy()
    bundle = {"format": FORMAT, "actions": list(ACTIONS), "n_actions": n_actions,
              "obs_dim": obs_dim, "hidden_dim": hidden_dim,
              "center": center.tolist(), "scale": scale.tolist(),
              "weights": {key: val.tolist() for key, val in best["model"].items()},
              "source_sha256": source, "data_sha256": data_sha}
    parity = 0.
    for i in range(min(32, len(layout))):
        parity_row = {"observations": features[i, :obs_dim], "history": features[i, obs_dim:],
                      "mask": legal[i], "reference": reference[i]}
        parity = max(parity, float(np.max(np.abs(teacher_q(parity_row, bundle) - prediction[i]))))
    if not np.isfinite(prediction).all() or parity > 1.e-4:
        raise RuntimeError("The frozen CPU teacher disagrees with the trained CUDA model")
    best["optimizer"] = torch.load(BytesIO(best.pop("optimizer_serialized")), map_location="cpu", weights_only=False)
    checkpoint = best | {"format": FORMAT, "seed": seed, "center": center, "scale": scale,
                         "layout_ids": {"fit": fit_layouts, "validation": val_layouts},
                         "source_sha256": source, "data_sha256": data_sha}
    metrics = {"format": FORMAT, "seed": seed, "device": "cuda:0", "gpu": torch.cuda.get_device_name(),
               "parameters": sum(value.numel() for value in best["model"].values()),
               "optimizer_steps": steps, "best_optimizer_step": best["steps"], "best_epoch": best["epoch"],
               "fit": _quality(prediction[fit], values[fit], measured[fit]),
               "validation": _quality(prediction[val], values[val], measured[val]),
               "validation_available": bool(len(val)), "fit_rows": len(fit), "validation_rows": len(val),
               "fit_paired_tapes": int(counts[fit].sum()), "validation_paired_tapes": int(counts[val].sum()),
               "layout_ids": checkpoint["layout_ids"], "parameters_updated": updated, "trace": trace,
               "cpu_cuda_max_abs_error": parity,
               "source_sha256": source, "data_sha256": data_sha}
    _save(output, bundle, metrics, checkpoint)
    return bundle, metrics

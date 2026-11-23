"""CUDA action-value fitting with paired continue and equal-time references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scaled_value import _atomic_torch_save, _cpu_copy, _sha256


REFERENCES = ("continue", "equal_time_hold")
OUTPUTS = ("utility_delta", "strict_delta", "branch_delta", "p_rescue", "p_loss")


class DualValue(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, 48), nn.SiLU(), nn.Linear(48, 40), nn.SiLU(), nn.Linear(40, 10))

    def forward(self, x):
        return self.net(x).reshape(-1, len(REFERENCES), len(OUTPUTS))


def bounded(raw):
    """Use one identical parameterization for fitted expectations and deployment."""
    return torch.cat((raw[..., :1], raw[..., 1:3].tanh(), raw[..., 3:].sigmoid()), dim=-1)


def measured_targets(outcomes):
    a = outcomes[:, :, None]
    reference = outcomes[:, None, :2]
    delta = a - reference
    rescue = (a[..., 1] > .5) & (reference[..., 1] <= .5)
    loss = (a[..., 1] <= .5) & (reference[..., 1] > .5)
    return np.stack((delta[..., 0], delta[..., 1], delta[..., 2], rescue, loss), axis=-1).mean(axis=-2)


def fit_dual(features, valid, outcomes, destination, *, epochs=25, seed=917_521_633):
    if not torch.cuda.is_available():
        raise RuntimeError("Action-value fit requires CUDA")
    if outcomes.shape[2] < 2 or not valid[:, :2].all():
        raise ValueError("Independent common outcomes and both reference actions are required")
    x = np.asarray(features[valid], dtype=np.float32)
    y = np.asarray(measured_targets(outcomes)[valid], dtype=np.float32)
    center = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), .035)
    utility_scale = np.maximum(y[..., 0].std(axis=0), .08)
    y[..., 0] /= utility_scale
    torch.manual_seed(seed)
    model = DualValue(x.shape[1]).to("cuda:0")
    before = {k: t.detach().clone() for k, t in model.state_dict().items()}
    optim = torch.optim.AdamW(model.parameters(), lr=.002, fused=True)
    xx = torch.as_tensor(np.clip((x - center) / scale, -9., 9.), device="cuda:0")
    yy = torch.as_tensor(y, device="cuda:0")
    trace = []
    for epoch in range(epochs):
        batch_losses = []
        for ids in torch.randperm(len(xx), device="cuda:0").split(512):
            raw = model(xx[ids])
            predictions = bounded(raw)
            goal = yy[ids]
            regression = F.smooth_l1_loss(predictions[..., :3], goal[..., :3])
            probabilities = F.binary_cross_entropy_with_logits(raw[..., 3:], goal[..., 3:])
            loss = regression + .7 * probabilities
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            batch_losses.append(float(loss.detach()))
        trace.append(float(np.mean(batch_losses)))
    torch.cuda.synchronize()
    changed = any(not torch.equal(before[k], t) for k, t in model.state_dict().items())
    if not changed:
        raise RuntimeError("Dual-reference value parameters did not update")
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "center": center, "scale": scale,
               "features": x.shape[1], "utility_scale": utility_scale, "optimizer": _cpu_copy(optim.state_dict()),
               "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "seed": seed,
               "epochs": epochs, "trace": trace, "references": REFERENCES, "outputs": OUTPUTS}
    destination = Path(destination)
    _atomic_torch_save(destination, payload)
    result = {"weights": str(destination), "sha256": _sha256(destination), "device": "cuda:0",
              "changed_on_cuda": True, "fit_actions": len(xx), "fit_prefixes": len(valid),
              "fit_tapes_per_action": outcomes.shape[2], "epochs": epochs, "loss": trace,
              "identical_train_inference_transform": True}
    return result


def load_dual(path, device="cuda:0"):
    data = torch.load(path, map_location="cpu", weights_only=False)
    model = DualValue(data["features"])
    model.load_state_dict(data["model"])
    return data | {"network": model.to(device).eval(), "device": device}


def predict_dual(bundle, features, valid):
    x = np.clip((features[valid] - bundle["center"]) / bundle["scale"], -9., 9.)
    with torch.inference_mode():
        y = bounded(bundle["network"](torch.as_tensor(x, dtype=torch.float32, device=bundle["device"]))).cpu().numpy()
    y[..., 0] *= bundle["utility_scale"]
    out = np.zeros((*valid.shape, len(REFERENCES), len(OUTPUTS)), dtype=np.float32)
    out[valid] = y
    return out


def calibrated_actions(prediction, valid, rule):
    """A probe needs positive strict evidence against both comparators."""
    margin, risk = rule["strict_margin"], rule["loss_limit"]
    if margin is None:
        return np.zeros(len(valid), dtype=int)
    strict, branch, loss = prediction[..., 1], prediction[..., 2], prediction[..., 4]
    eligible = ((strict > margin).all(axis=-1) & (branch <= 0).all(axis=-1)
                & (loss < risk).all(axis=-1) & valid)
    eligible[:, :2] = False
    rank = np.where(eligible, strict.min(axis=-1), -np.inf)
    actions = np.argmax(rank, axis=1)
    return np.where(eligible.any(axis=1), actions, 0)


def calibrate_dual(prediction, valid, outcomes):
    rules = [{"strict_margin": None, "loss_limit": 0.}]
    rules.extend({"strict_margin": margin, "loss_limit": loss} for margin in (.01, .04, .08, .16)
                 for loss in (.12, .25, .4))
    indices = np.arange(len(valid))
    observed = outcomes[:, :, :, 1:3].mean(axis=2)
    scored = []
    for rule in rules:
        actions = calibrated_actions(prediction, valid, rule)
        reviewed = actions >= 2
        reach = observed[indices, actions]
        delta_mpc = (reach - observed[:, 0]).sum(axis=0)
        delta_hold = (reach - observed[:, 1])[reviewed].sum(axis=0)
        if (reviewed.any() and (delta_mpc[0] <= 0 or delta_hold[0] <= 0 or delta_mpc[1] > 0 or delta_hold[1] > 0)):
            continue
        scored.append(((float(delta_mpc[0]), -float(delta_mpc[1]), -int(reviewed.sum())), rule))
    return max(scored, key=lambda row: row[0])[1]

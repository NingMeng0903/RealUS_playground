"""Fit a public-history action-value model for the continuation teacher.

The input is deliberately a counterfactual table rather than a teacher action
label.  Each row contains the public prefix and the public description of
every legal action, while ``outcomes`` contains several *paired* future tapes
for those actions.  Targets are computed against the equal-time hold action on
the same tape.  This lets the model learn an expected rescue/harm trade-off
when the privileged teacher would choose different actions in different
layouts.

The evaluator-only outcome table is used by :func:`fit_value` for supervision.
``predict_value`` accepts only the public feature tensor and a legal-action
mask, so deployment cannot accidentally read anatomy labels or future tapes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
import os
import tempfile
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OUTCOME_NAMES = (
    "utility", "strict", "wrong_branch", "wrong_delivery",
    "correct_regions", "correct_mm", "old_correct", "old_wrong",
)
PREDICTION_NAMES = (
    "utility_delta", "strict_delta", "wrong_branch_delta",
    "rescue_prob", "loss_prob",
)
HOLD_ACTION = 1
MODEL_VERSION = "paired_public_action_value_v1"


class ActionValueNet(nn.Module):
    """Small multi-head network used for public action proposals."""

    def __init__(self, feature_count: int, hidden: int = 48):
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        self.feature_count = int(feature_count)
        self.hidden = int(hidden)
        self.body = nn.Sequential(
            nn.Linear(feature_count, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        # utility, strict, wrong-branch, rescue logit, loss logit
        self.head = nn.Linear(hidden, len(PREDICTION_NAMES))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(features))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".pending")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".pending")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _cpu_copy(value):
    """Move optimizer checkpoint tensors to CPU without mutating the live state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _read_shard(path: Path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"features", "valid", "outcomes"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing action-value arrays {sorted(missing)}")
        features = np.asarray(data["features"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool)
        outcomes = np.asarray(data["outcomes"], dtype=np.float32)
        route = np.asarray(data["route"], dtype=np.str_) if "route" in data.files else None
    if features.ndim != 3:
        raise ValueError(f"{path}: features must have shape [N,A,D], got {features.shape}")
    if valid.shape != features.shape[:2]:
        raise ValueError(f"{path}: valid shape {valid.shape} does not match features {features.shape[:2]}")
    if outcomes.ndim != 4 or outcomes.shape[:2] != features.shape[:2] or outcomes.shape[3] != len(OUTCOME_NAMES):
        raise ValueError(f"{path}: outcomes must have shape [N,A,T,8], got {outcomes.shape}")
    if outcomes.shape[2] < 2:
        raise ValueError(f"{path}: paired training needs at least two future tapes")
    if not np.isfinite(features).all() or not np.isfinite(outcomes).all():
        raise ValueError(f"{path}: non-finite feature or outcome value")
    if not valid[:, HOLD_ACTION].all():
        raise ValueError(f"{path}: equal-time hold action {HOLD_ACTION} must be legal for every prefix")
    return features, valid, outcomes, route


def _paired_targets(outcomes: np.ndarray, valid: np.ndarray):
    """Return action-level regression/classification targets from same-tape pairs."""
    hold = outcomes[:, HOLD_ACTION, :, :]
    # Every candidate is compared to hold on the exact same future tape.  The
    # mean is an expectation target, not a hard best-action label.
    utility = outcomes[:, :, :, 0] - hold[:, None, :, 0]
    strict_delta = outcomes[:, :, :, 1] - hold[:, None, :, 1]
    wrong_delta = outcomes[:, :, :, 2] - hold[:, None, :, 2]
    rescue = ((outcomes[:, :, :, 1] > .5) & (hold[:, None, :, 1] <= .5)).astype(np.float32)
    harm = ((outcomes[:, :, :, 1] <= .5) & (hold[:, None, :, 1] > .5)).astype(np.float32)
    target = np.stack((utility.mean(axis=2), strict_delta.mean(axis=2),
                       wrong_delta.mean(axis=2), rescue.mean(axis=2), harm.mean(axis=2)), axis=-1)
    # Invalid candidates have no teacher target. Hold is retained as a legal
    # zero-delta reference and helps calibrate the action heads.
    rows = valid.reshape(-1)
    return target.reshape(-1, target.shape[-1])[rows], rows


def _flatten_shards(shards: Iterable[Path]):
    arrays = []
    route_rows = []
    feature_count = None
    tape_count = None
    for raw_path in shards:
        path = Path(raw_path)
        features, valid, outcomes, route = _read_shard(path)
        if feature_count is None:
            feature_count = features.shape[-1]
            tape_count = outcomes.shape[2]
        if features.shape[-1] != feature_count:
            raise ValueError("all action-value shards must have the same feature dimension")
        arrays.append((features, valid, outcomes))
        if route is not None:
            route_rows.extend(route.tolist())
        else:
            route_rows.extend(["unknown"] * len(features))
    if not arrays:
        raise ValueError("no action-value shards were supplied")
    features = np.concatenate([x[0] for x in arrays], axis=0)
    valid = np.concatenate([x[1] for x in arrays], axis=0)
    outcomes = np.concatenate([x[2] for x in arrays], axis=0)
    target, legal = _paired_targets(outcomes, valid)
    x = features.reshape(-1, feature_count)[legal]
    route_rows = np.repeat(np.asarray(route_rows, dtype="U64"), valid.shape[1])[legal]
    return x, target, route_rows, features.shape, int(outcomes.shape[2])


def _positive_weight(values: np.ndarray) -> float:
    positives = float(np.sum(values > .5))
    negatives = float(len(values) - positives)
    if positives < 1. or negatives < 1.:
        return 1.0
    return float(np.clip(negatives / positives, 1., 25.))


def _model_output(raw: torch.Tensor, utility_mean: float, utility_scale: float):
    utility = raw[:, 0] * utility_scale + utility_mean
    strict = torch.tanh(raw[:, 1])
    wrong = torch.tanh(raw[:, 2])
    rescue = torch.sigmoid(raw[:, 3])
    harm = torch.sigmoid(raw[:, 4])
    return torch.stack((utility, strict, wrong, rescue, harm), dim=-1)


def fit_value(shards: Iterable[Path], out: Path, *, epochs: int = 24,
              seed: int = 581_420_117, batch_size: int = 1024,
              hidden: int = 48) -> dict:
    """Fit the paired public action-value model on CUDA.

    ``out`` may be a checkpoint path or an output directory.  The function
    writes an atomic ``action_value.pt`` checkpoint and a JSON manifest beside
    it, and returns the manifest as a plain dictionary.
    """
    shard_paths = [Path(path) for path in shards]
    x, target, route_rows, source_shape, tape_count = _flatten_shards(shard_paths)
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("paired action-value training requires the approved CUDA environment")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    device = torch.device("cuda:0")
    model = ActionValueNet(x.shape[-1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.e-3, weight_decay=5.e-3, fused=True)

    center = x.mean(axis=0).astype(np.float32)
    scale = np.maximum(x.std(axis=0), .035).astype(np.float32)
    normalized = np.clip((x - center) / scale, -9., 9.).astype(np.float32)
    utility_mean = float(target[:, 0].mean())
    utility_scale = float(max(target[:, 0].std(), .05))
    normalized_targets = target.copy().astype(np.float32)
    normalized_targets[:, 0] = (normalized_targets[:, 0] - utility_mean) / utility_scale
    tensor_x = torch.as_tensor(normalized, dtype=torch.float32, device=device)
    tensor_y = torch.as_tensor(normalized_targets, dtype=torch.float32, device=device)
    pos_rescue = _positive_weight(target[:, 3])
    pos_harm = _positive_weight(target[:, 4])
    rescue_weight = torch.tensor(pos_rescue, dtype=torch.float32, device=device)
    harm_weight = torch.tensor(pos_harm, dtype=torch.float32, device=device)

    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    traces = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for epoch in range(int(epochs)):
        order = torch.randperm(len(tensor_x), generator=generator, device=device)
        losses = []
        for batch_indices in order.split(int(batch_size)):
            raw = model(tensor_x[batch_indices])
            y = tensor_y[batch_indices]
            identity = F.smooth_l1_loss(raw[:, 0], y[:, 0])
            strict = F.smooth_l1_loss(raw[:, 1], y[:, 1])
            wrong = F.smooth_l1_loss(raw[:, 2], y[:, 2])
            rescue = F.binary_cross_entropy_with_logits(raw[:, 3], y[:, 3], pos_weight=rescue_weight)
            harm = F.binary_cross_entropy_with_logits(raw[:, 4], y[:, 4], pos_weight=harm_weight)
            loss = identity + .75 * strict + .35 * wrong + .25 * rescue + .25 * harm
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        traces.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                       "identity": float(identity.detach().item()),
                       "strict": float(strict.detach().item()),
                       "wrong_branch": float(wrong.detach().item()),
                       "rescue": float(rescue.detach().item()),
                       "harm": float(harm.detach().item())})
    torch.cuda.synchronize(device)
    changed = any(not torch.equal(before[key], value.detach()) for key, value in model.state_dict().items())
    if not changed:
        raise RuntimeError("action-value model parameters did not change on CUDA")

    destination = Path(out)
    if destination.suffix != ".pt":
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination / "action_value.pt"
    model_cpu = model.detach().cpu() if hasattr(model, "detach") else model.cpu()
    checkpoint = {
        "model_version": MODEL_VERSION,
        "model": model_cpu.state_dict(),
        "feature_count": int(x.shape[-1]), "hidden": int(hidden),
        "center": center, "scale": scale,
        "utility_mean": utility_mean, "utility_scale": utility_scale,
        "prediction_names": PREDICTION_NAMES,
        "outcome_names": OUTCOME_NAMES,
        "hold_action": HOLD_ACTION,
        "optimizer": _cpu_copy(optimizer.state_dict()),
        "rng": {
            "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        },
        "training": {
            "seed": int(seed), "epochs": int(epochs), "batch_size": int(batch_size),
            "candidate_rows": int(len(x)), "prefix_count": int(source_shape[0]),
            "source_shape": list(map(int, source_shape)), "tape_count": int(tape_count),
            "rescue_positive_weight": pos_rescue, "harm_positive_weight": pos_harm,
            "changed_on_cuda": bool(changed), "loss_trace": traces,
            "route_counts": {str(route): int(np.sum(route_rows == route)) for route in np.unique(route_rows)},
            "target_definition": "per-action mean paired future delta against equal-time hold on the same tape",
        },
    }
    _atomic_torch_save(destination, checkpoint)
    digest = _sha256(destination)
    manifest = {
        "model_version": MODEL_VERSION, "weights": str(destination), "sha256": digest,
        "device": str(device), "gpu": torch.cuda.get_device_name(device),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "feature_count": int(x.shape[-1]), "hidden": int(hidden),
        "candidate_rows": int(len(x)), "prefix_count": int(source_shape[0]),
        "source_shape": list(map(int, source_shape)), "tape_count": int(tape_count),
        "hold_action": HOLD_ACTION, "prediction_names": list(PREDICTION_NAMES),
        "outcome_names": list(OUTCOME_NAMES), "changed_on_cuda": bool(changed),
        "loss_trace": traces,
        "route_counts": {str(route): int(np.sum(route_rows == route)) for route in np.unique(route_rows)},
        "paired_target": "same-tape action outcome minus equal-time hold, with expected rescue/harm probabilities",
        "public_deployment_inputs": "features and legal mask only",
    }
    _atomic_json(destination.with_suffix(".json"), manifest)
    return manifest


def load_value(path: Path, device: str | torch.device = "cpu") -> dict:
    """Load a frozen action-value checkpoint for public-only inference."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("unsupported action-value checkpoint version")
    model = ActionValueNet(int(checkpoint["feature_count"]), int(checkpoint["hidden"]))
    model.load_state_dict(checkpoint["model"])
    model.to(torch.device(device)).eval()
    return {
        "model": model, "device": torch.device(device),
        "center": np.asarray(checkpoint["center"], dtype=np.float32),
        "scale": np.asarray(checkpoint["scale"], dtype=np.float32),
        "utility_mean": float(checkpoint["utility_mean"]),
        "utility_scale": float(checkpoint["utility_scale"]),
        "feature_count": int(checkpoint["feature_count"]),
        "prediction_names": tuple(checkpoint["prediction_names"]),
    }


def predict_value(features: np.ndarray, valid: np.ndarray, bundle: Mapping) -> dict[str, np.ndarray]:
    """Predict paired action values from public features and a legal mask.

    Outputs are finite ``[N,A]`` arrays.  Illegal entries are set to zero and
    should be masked by the caller before choosing an action.
    """
    values = np.asarray(features, dtype=np.float32)
    legal = np.asarray(valid, dtype=bool)
    if values.ndim != 3:
        raise ValueError(f"features must have shape [N,A,D], got {values.shape}")
    if legal.shape != values.shape[:2]:
        raise ValueError("valid mask does not match feature prefix/action dimensions")
    if values.shape[-1] != int(bundle["feature_count"]):
        raise ValueError("feature dimension does not match the frozen action-value model")
    if not np.isfinite(values).all():
        raise ValueError("public action features contain non-finite values")
    flat = values.reshape(-1, values.shape[-1])
    normalized = np.clip((flat - bundle["center"]) / bundle["scale"], -9., 9.)
    device = torch.device(bundle.get("device", "cpu"))
    model = bundle["model"]
    with torch.inference_mode():
        raw = model(torch.as_tensor(normalized, dtype=torch.float32, device=device))
        outputs = _model_output(raw, bundle["utility_mean"], bundle["utility_scale"]).cpu().numpy()
    outputs = outputs.reshape(*values.shape[:2], len(PREDICTION_NAMES))
    outputs[~legal] = 0.
    return {name: np.asarray(outputs[..., index], dtype=np.float32)
            for index, name in enumerate(PREDICTION_NAMES)} | {"valid": legal.copy()}


def _main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", default=24, type=int)
    parser.add_argument("--seed", default=581420117, type=int)
    args = parser.parse_args()
    print(json.dumps(fit_value(args.shard, args.out, epochs=args.epochs, seed=args.seed),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()


__all__ = ["ActionValueNet", "fit_value", "load_value", "predict_value",
           "OUTCOME_NAMES", "PREDICTION_NAMES"]

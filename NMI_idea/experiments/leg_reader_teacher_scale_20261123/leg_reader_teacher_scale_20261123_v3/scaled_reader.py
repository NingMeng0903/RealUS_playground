"""Streaming CUDA updates for the public leg-scan reader.

This module is deliberately independent of the rollout driver.  It consumes
anonymous candidate shards and keeps the evaluator labels in the training
process only.  Two readers are started from the frozen v2 public reader:

``scaled_identity``
    The v2 identity and distinct-view association objective on the new scan
    population.

``updated_reader``
    The same objective plus hard positive/rival ranking at the end of each
    route region.  Positive target candidates receive a recall term and a
    high-scoring non-target candidate receives an explicit wrong-branch term.

The data loader processes complete scans in bounded blocks.  ``np.load`` must
decompress one NPZ shard at a time, but no concatenated training matrix is
created.  Checkpoints contain the optimizer and RNG state so a caller can
resume from an epoch boundary without changing the public feature contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F

# Keep the model contract identical to the frozen v2 reader.  The parent
# launcher puts the v2 study directory on ``sys.path``.  The explicit fallback
# also makes this module importable when called directly from this directory.
try:
    from reader_fit import CLASSES, Readout
except ImportError:
    import importlib.util

    _V2_READER = Path(__file__).resolve().parents[1] / "leg_scan_target_routes_20261123_v2" / "reader_fit.py"
    _SPEC = importlib.util.spec_from_file_location("leg_v2_reader_fit", _V2_READER)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"cannot load the v2 reader implementation from {_V2_READER}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    CLASSES, Readout = _MODULE.CLASSES, _MODULE.Readout

DEFAULT_BATCH_ROWS = 16384
DEFAULT_ROW_BATCH = 768
DEFAULT_PAIR_BATCH = 192
DEFAULT_MAX_PAIRS_PER_SCAN = 1024
DEFAULT_HARD_PAIRS_PER_SCAN = 6


def transform_labels(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int64)
    if not np.all(np.isin(values, np.arange(-2, 6))):
        raise ValueError("Unrecognized evaluator-only candidate label")
    return np.where(values == -1, 6, np.where(values == -2, 7, values)).astype(np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".pending")
    try:
        with open(fd, "wb", closefd=True) as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            stream.fileno()
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".pending")
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _as_path_list(shards: Sequence[Any] | Any) -> list[Any]:
    if isinstance(shards, (str, Path, Mapping)):
        return [shards]
    result = list(shards)
    if not result:
        raise ValueError("at least one training shard is required")
    return result


def _open_shard(source: Any):
    if isinstance(source, Mapping):
        return source, None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False), path


def _array(shard: Mapping[str, Any], *names: str) -> np.ndarray:
    for name in names:
        if name in shard:
            return np.asarray(shard[name])
    raise KeyError(f"shard is missing one of {names}")


def _validate_shard(shard: Mapping[str, Any], source: Any) -> dict[str, Any]:
    features = _array(shard, "features")
    offsets = _array(shard, "offsets").astype(np.int64, copy=False).reshape(-1)
    labels = _array(shard, "labels", "evaluator_labels").astype(np.int8, copy=False).reshape(-1)
    regions = _array(shard, "regions", "region").astype(np.int8, copy=False).reshape(-1)
    frames = _array(shard, "frames", "frame").astype(np.float32, copy=False).reshape(-1)
    stations = _array(shard, "stations", "station").astype(np.float32, copy=False).reshape(-1)
    sectors = _array(shard, "sectors", "sector").astype(np.float32, copy=False).reshape(-1)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"invalid feature matrix in {source}")
    n, dimensions = features.shape
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"invalid scan offsets in {source}")
    for name, array in (("labels", labels), ("regions", regions), ("frames", frames),
                        ("stations", stations), ("sectors", sectors)):
        if len(array) != n:
            raise ValueError(f"{name} length {len(array)} does not match features {n} in {source}")
    if not np.isfinite(features.astype(np.float32, copy=False)).all():
        raise ValueError(f"non-finite features in {source}")
    transform_labels(labels)
    if np.any((regions < 0) | (regions >= 6)):
        raise ValueError(f"regions must be in [0, 5] in {source}")
    return {"features": features, "offsets": offsets, "labels": labels,
            "regions": regions, "frames": frames, "stations": stations, "sectors": sectors,
            "feature_order": np.asarray(shard["feature_order"]).tolist()
            if "feature_order" in shard else None}


def _blocks(offsets: np.ndarray, max_rows: int = DEFAULT_BATCH_ROWS) -> Iterator[tuple[int, int, int, int]]:
    scan_count = len(offsets) - 1
    first = 0
    while first < scan_count:
        last = first + 1
        while last < scan_count and offsets[last + 1] - offsets[first] <= max_rows:
            last += 1
        yield first, last, int(offsets[first]), int(offsets[last])
        first = last


def _association_pairs(
    frames: np.ndarray,
    stations: np.ndarray,
    sectors: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    rng: np.random.Generator,
    max_pairs_per_scan: int = DEFAULT_MAX_PAIRS_PER_SCAN,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample v2-compatible pairs from complete scans in one block."""
    pairs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for scan in range(len(offsets) - 1):
        lo, hi = int(offsets[scan]), int(offsets[scan + 1])
        count = hi - lo
        if count < 2:
            continue
        amount = min(max_pairs_per_scan, max(8, 8 * count))
        first = rng.integers(count, size=amount)
        second = np.clip(first + rng.integers(-18, 19, size=amount), 0, count - 1)
        first_global, second_global = first + lo, second + lo
        ds = np.abs(stations[first_global] - stations[second_global])
        da = np.abs((sectors[first_global] - sectors[second_global] + 12.) % 24. - 12.)
        raw_first, raw_second = labels[first_global], labels[second_global]
        valid = ((frames[first_global] != frames[second_global]) & (ds <= 3.1)
                 & ((ds >= .63) | (da >= .74)) & (raw_first >= 0) & (raw_second >= 0))
        if not np.any(valid):
            continue
        first_global, second_global = first_global[valid], second_global[valid]
        raw_first, raw_second = raw_first[valid], raw_second[valid]
        same = raw_first == raw_second
        keep = (~same) | (raw_first <= 2) | (raw_first == 5)
        if np.any(keep):
            pairs.append(np.column_stack((first_global[keep], second_global[keep])))
            targets.append(same[keep].astype(np.float32))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.float32)
    return np.concatenate(pairs, axis=0).astype(np.int64), np.concatenate(targets).astype(np.float32)


def _hard_region_pairs(
    net: Readout,
    x: torch.Tensor,
    labels: np.ndarray,
    regions: np.ndarray,
    frames: np.ndarray,
    offsets: np.ndarray,
    max_per_scan: int = DEFAULT_HARD_PAIRS_PER_SCAN,
) -> np.ndarray:
    """Find hard pairs over every candidate acquired in each delivery region.

    Delivery selects the maximum score over the complete acquired region.  The
    hard examples therefore use all candidates in that region, rather than a
    late-frame slice.  Candidate ranking is measured with the target log odds
    used by the deployed softmax reader.  The top three (or the configured
    limit) true candidates are paired with the strongest three rivals.
    """
    with torch.no_grad():
        logits, _ = net(x)
        target_score = (logits[:, 0] - torch.logsumexp(logits[:, 1:], dim=-1)).detach().cpu().numpy()
    pairs: list[tuple[int, int]] = []
    for scan in range(len(offsets) - 1):
        lo, hi = int(offsets[scan]), int(offsets[scan + 1])
        for region in range(6):
            candidates = np.flatnonzero(regions[lo:hi] == region) + lo
            if not len(candidates):
                continue
            positive = candidates[labels[candidates] == 0]
            rival = candidates[labels[candidates] != 0]
            if not len(positive) or not len(rival):
                continue
            # Keep the highest-scoring true and rival candidates because these
            # are exactly the candidates that compete for the deployed argmax.
            positive = positive[np.argsort(-target_score[positive])[:max_per_scan]]
            rival = rival[np.argsort(-target_score[rival])[:max_per_scan]]
            for left in positive:
                for right in rival:
                    pairs.append((int(left), int(right)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _load_prior(path: Path | None, dimensions: int, device: torch.device) -> tuple[dict[str, Any], Readout]:
    if path is None:
        model = Readout(dimensions).to(device)
        return {"feature_order": tuple(f"feature_{i}" for i in range(dimensions)),
                "center": np.zeros(dimensions, dtype=np.float32),
                "scale": np.ones(dimensions, dtype=np.float32), "variant": "random"}, model
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_order = tuple(payload.get("feature_order", ()))
    if len(feature_order) != dimensions:
        raise ValueError("prior reader feature contract does not match shard features")
    center = np.asarray(payload.get("center"), dtype=np.float32).reshape(-1)
    scale = np.asarray(payload.get("scale"), dtype=np.float32).reshape(-1)
    if center.shape != (dimensions,) or scale.shape != (dimensions,) or not np.isfinite(scale).all():
        raise ValueError("prior reader has invalid normalization statistics")
    model = Readout(dimensions)
    model.load_state_dict(payload["model"])
    model.to(device)
    return {"feature_order": feature_order, "center": center, "scale": np.maximum(scale, .035),
            "variant": payload.get("variant", "full_public_embedding"),
            "local_feature_count": payload.get("local_feature_count")}, model


def _state_cpu(model: Readout) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _optimizer_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    tensors = {index: {key: item.detach().cpu() if torch.is_tensor(item) else item
                       for key, item in value.items()} for index, value in state["state"].items()}
    return state | {"state": tensors}


def _train_variant(
    sources: list[Any],
    prior: dict[str, Any],
    initial_state: Mapping[str, torch.Tensor],
    output: Path,
    variant: str,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
    batch_rows: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = Readout(len(prior["feature_order"])).to(device)
    model.load_state_dict(initial_state)
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.005, fused=True)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.005)
    before = {k: value.detach().clone() for k, value in model.state_dict().items()}
    trace: list[dict[str, float]] = []
    steps = 0
    candidates = scans = associations = hard_pairs_total = 0
    for epoch in range(epochs):
        epoch_losses = []
        epoch_identity = []
        epoch_association = []
        epoch_rank = []
        epoch_wrong = []
        epoch_recall = []
        for source in sources:
            shard_file = _open_shard(source)
            raw, handle = shard_file
            data = _validate_shard(raw, source)
            features = data["features"]
            offsets = data["offsets"]
            center = prior["center"]
            scale = prior["scale"]
            for scan_lo, scan_hi, lo, hi in _blocks(offsets, batch_rows):
                local = {key: data[key][lo:hi] for key in ("labels", "regions", "frames", "stations", "sectors")}
                values = np.asarray(features[lo:hi], dtype=np.float32)
                values = np.clip((values - center) / scale, -9., 9.)
                x = torch.as_tensor(values, device=device)
                y = torch.as_tensor(transform_labels(local["labels"]), device=device)
                row_offsets = offsets[scan_lo:scan_hi + 1] - lo
                case_weights = np.empty(hi - lo, dtype=np.float32)
                for scan in range(len(row_offsets) - 1):
                    first, last = int(row_offsets[scan]), int(row_offsets[scan + 1])
                    case_weights[first:last] = 1. / max(1, last - first)
                case_weights /= max(float(case_weights.mean()), 1.e-12)
                weights = torch.as_tensor(case_weights, device=device)
                pairs, same = _association_pairs(local["frames"], local["stations"], local["sectors"],
                                                  local["labels"], row_offsets, rng)
                if len(pairs):
                    pairs = pairs.astype(np.int64, copy=False)
                    pair_y = torch.as_tensor(same, device=device)
                    pair_indices = torch.as_tensor(pairs, device=device)
                    pair_order = torch.randperm(len(pairs), device=device)
                else:
                    pair_y = pair_indices = pair_order = None
                hard = np.empty((0, 2), dtype=np.int64)
                if variant == "updated_reader":
                    hard = _hard_region_pairs(model, x, local["labels"], local["regions"],
                                              local["frames"], row_offsets)
                hard_tensor = torch.as_tensor(hard, device=device) if len(hard) else None
                row_order = torch.randperm(len(y), device=device)
                for batch_number, row_batch in enumerate(row_order.split(DEFAULT_ROW_BATCH)):
                    logits, embedding = model(x[row_batch])
                    identity = (F.cross_entropy(logits, y[row_batch], reduction="none")
                                * weights[row_batch]).mean()
                    association = torch.zeros((), device=device)
                    if pair_indices is not None:
                        take = (torch.arange(DEFAULT_PAIR_BATCH, device=device)
                                + batch_number * DEFAULT_PAIR_BATCH) % len(pair_indices)
                        selected = pair_indices[pair_order[take]]
                        _, left = model(x[selected[:, 0]])
                        _, right = model(x[selected[:, 1]])
                        match = 4. * (torch.sum(left * right, dim=1) - .40)
                        association = F.binary_cross_entropy_with_logits(match, pair_y[pair_order[take]])
                    ranking = torch.zeros((), device=device)
                    wrong = torch.zeros((), device=device)
                    recall = torch.zeros((), device=device)
                    if hard_tensor is not None:
                        take = (torch.arange(min(DEFAULT_PAIR_BATCH, len(hard_tensor)), device=device)
                                + batch_number * DEFAULT_PAIR_BATCH) % len(hard_tensor)
                        selected = hard_tensor[take]
                        hard_logits, _ = model(x[selected.reshape(-1)] )
                        hard_logits = hard_logits.reshape(-1, 2, len(CLASSES))
                        pos, rival = hard_logits[:, 0], hard_logits[:, 1]
                        pos_odds = pos[:, 0] - torch.logsumexp(pos[:, 1:], dim=-1)
                        rival_odds = rival[:, 0] - torch.logsumexp(rival[:, 1:], dim=-1)
                        ranking = F.relu(.65 - (pos_odds - rival_odds)).mean()
                        wrong = F.softplus(rival_odds).mean()
                        recall = F.softplus(-pos_odds).mean()
                    loss = identity + .18 * association
                    if variant == "updated_reader":
                        loss = loss + .45 * ranking + .12 * wrong + .08 * recall
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    epoch_losses.append(float(loss.detach().cpu()))
                    epoch_identity.append(float(identity.detach().cpu()))
                    epoch_association.append(float(association.detach().cpu()))
                    epoch_rank.append(float(ranking.detach().cpu()))
                    epoch_wrong.append(float(wrong.detach().cpu()))
                    epoch_recall.append(float(recall.detach().cpu()))
                candidates += hi - lo
                scans += scan_hi - scan_lo
                associations += len(pairs)
                hard_pairs_total += len(hard)
            if handle is not None and hasattr(handle, "close"):
                handle.close()
        trace.append({"epoch": epoch + 1, "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.,
                      "identity": float(np.mean(epoch_identity)) if epoch_identity else 0.,
                      "association": float(np.mean(epoch_association)) if epoch_association else 0.,
                      "ranking": float(np.mean(epoch_rank)) if epoch_rank else 0.,
                      "wrong_branch": float(np.mean(epoch_wrong)) if epoch_wrong else 0.,
                      "recall": float(np.mean(epoch_recall)) if epoch_recall else 0.,
                      "steps": len(epoch_losses)})
        checkpoint = {
            "model": _state_cpu(model), "optimizer": _optimizer_cpu(optimizer),
            "feature_order": tuple(prior["feature_order"]), "center": prior["center"],
            "scale": prior["scale"], "classes": CLASSES, "variant": variant,
            "local_feature_count": prior.get("local_feature_count"), "epoch": epoch + 1,
            "steps": steps, "seed": seed, "rng_numpy": rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "shard_cursor": "epoch_boundary", "training_trace": trace,
        }
        _atomic_torch_save(output / f"{variant}_epoch_{epoch + 1:04d}.pt", checkpoint)
        print(f"{variant}: CUDA epoch {epoch + 1}/{epochs}, {steps} optimizer steps, "
              f"loss {trace[-1]['loss']:.4f}", flush=True)
    changed = any(not torch.equal(before[key], value.detach()) for key, value in model.state_dict().items())
    if not changed:
        raise RuntimeError(f"{variant} identity parameters did not update")
    final = output / f"{variant}.pt"
    _atomic_torch_save(final, checkpoint)
    return {
        "variant": variant, "weights": str(final), "sha256": _sha256(final),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(value.numel() for value in model.parameters()), "epochs": epochs,
        "optimizer_steps": steps, "candidate_count": candidates // epochs,
        "scan_count": scans // epochs, "sampled_association_pairs_all_epochs": associations,
        "mined_region_pairs_all_epochs": hard_pairs_total, "parameter_update_verified": changed,
        "training_trace": trace,
    }


def fit_readers(
    shards: Sequence[Any] | Any,
    prior_path: str | Path | None,
    out: str | Path,
    *,
    epochs: int = 4,
    seed: int = 371700513,
    batch_rows: int = DEFAULT_BATCH_ROWS,
) -> dict[str, Any]:
    """Update both reader arms on new anonymous scan shards using CUDA."""
    if epochs <= 0 or batch_rows <= 0:
        raise ValueError("epochs and batch_rows must be positive")
    sources = _as_path_list(shards)
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("scaled reader training requires the approved CUDA environment")
    device = torch.device("cuda:0")
    first_raw, first_handle = _open_shard(sources[0])
    first = _validate_shard(first_raw, sources[0])
    dimensions = int(first["features"].shape[1])
    if first_handle is not None and hasattr(first_handle, "close"):
        first_handle.close()
    prior_file = Path(prior_path) if prior_path is not None else None
    prior, base_model = _load_prior(prior_file, dimensions, device)
    initial_state = _state_cpu(base_model)
    for source in sources[1:]:
        raw, handle = _open_shard(source)
        check = _validate_shard(raw, source)
        if check["features"].shape[1] != dimensions:
            raise ValueError("all training shards must have the same feature dimension")
        if prior["feature_order"] and check.get("feature_order") is not None:
            if tuple(check["feature_order"]) != tuple(prior["feature_order"]):
                raise ValueError("training shard feature order differs from the frozen reader")
        if handle is not None and hasattr(handle, "close"):
            handle.close()
    summaries = {}
    for variant in ("scaled_identity", "updated_reader"):
        summaries[variant] = _train_variant(
            sources, prior, initial_state, output, variant,
            epochs=epochs, seed=seed, device=device, batch_rows=batch_rows,
        )
    manifest = {
        "format": "leg_public_reader_update_v1", "feature_count": dimensions,
        "feature_order": list(prior["feature_order"]), "prior_reader": str(prior_file) if prior_file else None,
        "prior_sha256": _sha256(prior_file) if prior_file else None, "epochs": epochs,
        "seed": seed, "batch_rows": batch_rows, "device": str(device),
        "cuda_device": torch.cuda.get_device_name(device), "cuda_verified": True,
        "shards": [str(source) for source in sources], "readers": summaries,
    }
    _atomic_json(output / "reader_update_manifest.json", manifest)
    return manifest


def load_reader(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a v2 or v3 reader checkpoint for CPU or CUDA inference."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    order = tuple(payload.get("feature_order", ()))
    if not order:
        raise ValueError("reader checkpoint has no feature contract")
    model = Readout(len(order))
    model.load_state_dict(payload["model"])
    target = torch.device(device)
    model.to(target).eval()
    return {"model": model, "feature_order": order,
            "center": np.asarray(payload["center"], dtype=np.float32),
            "scale": np.maximum(np.asarray(payload["scale"], dtype=np.float32), .035),
            "variant": payload.get("variant", "full_public_embedding"),
            "local_feature_count": payload.get("local_feature_count"),
            "classes": tuple(payload.get("classes", CLASSES)), "device": str(target),
            "weights": str(path), "checkpoint": payload}


def predict(features: np.ndarray, bundle: Mapping[str, Any]) -> np.ndarray:
    """Return connected-target probabilities for one feature matrix."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(bundle["feature_order"]):
        raise ValueError("features do not match the frozen public reader contract")
    values = np.clip((values - bundle["center"]) / bundle["scale"], -9., 9.)
    local_count = bundle.get("local_feature_count")
    if bundle.get("variant") == "local_embedding" and local_count is not None:
        values[:, int(local_count):] = 0.
    model = bundle["model"]
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits, _ = model(torch.as_tensor(values, dtype=torch.float32, device=device))
        return logits.softmax(dim=-1)[:, 0].detach().cpu().numpy()


__all__ = ["CLASSES", "Readout", "fit_readers", "load_reader", "predict", "transform_labels"]

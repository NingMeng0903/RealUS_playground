"""CUDA training for the corrected public reader and instance association.

The model keeps the compact v2 ``Readout`` contract.  Identity supervision is
still evaluated per anonymous candidate, while the normalized association head
is trained against a local physical-component id supplied by the evaluator
side of a synthetic shard.  That id is never part of the model input and is
not accepted by :func:`predict_heads`.
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
from torch import nn
from torch.nn import functional as F


try:
    from reader_fit import CLASSES, Readout
except ImportError:
    import importlib.util

    _V2 = Path(__file__).resolve().parents[1] / "leg_scan_target_routes_20261123_v2" / "reader_fit.py"
    _SPEC = importlib.util.spec_from_file_location("leg_v2_reader_fit", _V2.resolve())
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"cannot load the compact reader from {_V2}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    CLASSES, Readout = _MODULE.CLASSES, _MODULE.Readout


DEFAULT_BATCH_ROWS = 16384
DEFAULT_ROW_BATCH = 768
DEFAULT_PAIR_BATCH = 192
DEFAULT_MAX_PAIRS_PER_SCAN = 1024
MODEL_VERSION = "leg_public_reader_instance_association_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
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
            os_fileno = stream.fileno()
            if os_fileno >= 0:
                import os
                os.fsync(os_fileno)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".pending")
    try:
        with open(fd, "w", encoding="utf-8", closefd=True) as stream:
            json.dump(dict(payload), stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
            stream.write("\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _as_paths(shards: Sequence[Any] | Any) -> list[Any]:
    if isinstance(shards, (str, Path, Mapping)):
        return [shards]
    result = list(shards)
    if not result:
        raise ValueError("at least one training shard is required")
    return result


def _open(source: Any):
    if isinstance(source, Mapping):
        return source, None
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False), path


def _array(shard: Mapping[str, Any], *names: str, required: bool = True):
    for name in names:
        if name in shard:
            return np.asarray(shard[name])
    if required:
        raise KeyError(f"shard is missing one of {names}")
    return None


def _transform_labels(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int64)
    if not np.all(np.isin(values, np.arange(-2, 6))):
        raise ValueError("unrecognized evaluator candidate label")
    return np.where(values == -1, 6, np.where(values == -2, 7, values)).astype(np.int64)


def _validate(shard: Mapping[str, Any], source: Any) -> dict[str, Any]:
    features = _array(shard, "features")
    offsets = _array(shard, "offsets").astype(np.int64, copy=False).reshape(-1)
    labels = _array(shard, "labels", "evaluator_labels").astype(np.int8, copy=False).reshape(-1)
    regions = _array(shard, "regions", "region").astype(np.int8, copy=False).reshape(-1)
    frames = _array(shard, "frames", "frame").astype(np.float32, copy=False).reshape(-1)
    stations = _array(shard, "stations", "station").astype(np.float32, copy=False).reshape(-1)
    sectors = _array(shard, "sectors", "sector").astype(np.float32, copy=False).reshape(-1)
    points = _array(shard, "points", required=False)
    instance = _array(shard, "instance", "instances", required=False)
    if instance is None:
        raise ValueError(f"{source}: v4 instance ids are required for association supervision")
    instance = instance.astype(np.int64, copy=False).reshape(-1)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"{source}: features must be a numeric [N,D] matrix")
    n, dimensions = features.shape
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) <= 0):
        raise ValueError(f"{source}: offsets do not delimit complete non-empty scans")
    for name, value in (("labels", labels), ("regions", regions), ("frames", frames),
                        ("stations", stations), ("sectors", sectors), ("instance", instance)):
        if len(value) != n:
            raise ValueError(f"{source}: {name} length {len(value)} does not match features {n}")
    if points is not None:
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (n, 2):
            raise ValueError(f"{source}: points must have shape [N,2]")
        if not np.isfinite(points).all():
            raise ValueError(f"{source}: points contain non-finite values")
    if not np.isfinite(features.astype(np.float32, copy=False)).all():
        raise ValueError(f"{source}: features contain non-finite values")
    _transform_labels(labels)
    if np.any((regions < 0) | (regions >= 6)):
        raise ValueError(f"{source}: regions must be in [0,5]")
    if np.any(instance < -1):
        raise ValueError(f"{source}: instance ids must be -1 or nonnegative")
    order = tuple(np.asarray(shard["feature_order"]).tolist()) if "feature_order" in shard else None
    return {"features": features, "offsets": offsets, "labels": labels, "regions": regions,
            "frames": frames, "stations": stations, "sectors": sectors, "points": points,
            "instance": instance, "feature_order": order, "dimensions": dimensions}


def _blocks(offsets: np.ndarray, max_rows: int) -> Iterator[tuple[int, int, int, int]]:
    first = 0
    scans = len(offsets) - 1
    while first < scans:
        last = first + 1
        while last < scans and offsets[last + 1] - offsets[first] <= max_rows:
            last += 1
        yield first, last, int(offsets[first]), int(offsets[last])
        first = last


def _circular_delta(a, b):
    return np.abs((a - b + 12.) % 24. - 12.)


def _association_pairs(frames, stations, sectors, instance, offsets, rng,
                       points=None, regions=None, max_pairs_per_scan=DEFAULT_MAX_PAIRS_PER_SCAN):
    """Return balanced local pairs with physical-component targets.

    Static repeats are removed by requiring either a station displacement or a
    meaningful view change.  Negative candidates are preferentially local in
    measured point space when points are present.  Region 2 receives more
    proposals so the known competing-vessel failure mode is represented.
    """
    positive, negative = [], []
    positive_region2 = []
    for scan in range(len(offsets) - 1):
        lo, hi = int(offsets[scan]), int(offsets[scan + 1])
        count = hi - lo
        if count < 2:
            continue
        amount = min(max_pairs_per_scan * 2, max(32, 12 * count))
        left = rng.integers(count, size=amount) + lo
        right = np.clip(left - lo + rng.integers(-24, 25, size=amount), 0, count - 1) + lo
        ds = np.abs(stations[left] - stations[right])
        da = _circular_delta(sectors[left], sectors[right])
        keep = ((frames[left] != frames[right]) & (left != right)
                & (ds <= 3.2) & ((ds >= .63) | (da >= .74))
                & (instance[left] >= 0) & (instance[right] >= 0))
        if not np.any(keep):
            continue
        left, right = left[keep], right[keep]
        ds, da = ds[keep], da[keep]
        same = instance[left] == instance[right]
        # A static view can yield several equivalent candidates. Keep only one
        # proposal per coarse view pair to avoid counting repeated frames as
        # independent association evidence.
        view_key = np.stack((np.rint(stations[left] * 2), np.rint(sectors[left]),
                             np.rint(stations[right] * 2), np.rint(sectors[right])), axis=1)
        _, unique = np.unique(view_key, axis=0, return_index=True)
        left, right, same, ds, da = left[unique], right[unique], same[unique], ds[unique], da[unique]
        if points is not None:
            distance = np.linalg.norm(points[left] - points[right], axis=1)
            same &= distance <= 15.0
            local_negative = (~same) & (distance <= 14.0)
        else:
            local_negative = ~same
        for i, j, is_same, region, is_local_negative in zip(
                left, right, same, regions[left], local_negative, strict=True):
            pair = (int(i), int(j))
            if is_same:
                (positive_region2 if int(region) == 2 else positive).append(pair)
            elif is_local_negative:
                negative.append(pair)
    if not positive and not positive_region2:
        return np.empty((0, 2), np.int64), np.empty(0, np.float32)
    positives = positive_region2 + positive
    rng.shuffle(positives)
    rng.shuffle(negative)
    target = min(len(positives), len(negative))
    if target == 0:
        selected = positives if positives else negative
        return np.asarray(selected, np.int64), np.asarray([1.] * len(positives) + [0.] * len(negative), np.float32)[:len(selected)]
    # Maintain a material region-2 emphasis without allowing its duplicates to
    # dominate all scans.
    selected_pos = positives[:target]
    selected_neg = negative[:target]
    pairs = np.asarray(selected_pos + selected_neg, np.int64)
    y = np.asarray([1.] * target + [0.] * target, np.float32)
    order = rng.permutation(len(pairs))
    return pairs[order], y[order]


def _load_prior(path: Path, dimensions: int, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    order = tuple(payload.get("feature_order", ()))
    if len(order) != dimensions:
        raise ValueError("prior reader feature contract does not match v4 features")
    center = np.asarray(payload["center"], dtype=np.float32).reshape(-1)
    scale = np.asarray(payload["scale"], dtype=np.float32).reshape(-1)
    if center.shape != (dimensions,) or scale.shape != (dimensions,):
        raise ValueError("prior reader normalization shape is invalid")
    model = Readout(dimensions).to(device)
    model.load_state_dict(payload["model"])
    return {"feature_order": order, "center": center, "scale": np.maximum(scale, .035),
            "classes": tuple(payload.get("classes", CLASSES)), "prior_variant": payload.get("variant")}, model


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _cpu_optimizer(optimizer):
    state = optimizer.state_dict()
    copied = {index: {key: value.detach().cpu() if torch.is_tensor(value) else value
                      for key, value in slot.items()} for index, slot in state["state"].items()}
    return state | {"state": copied}


def _train(shards, prior, initial_state, destination, *, epochs, seed, batch_rows, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    model = Readout(len(prior["feature_order"])).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.005, fused=True)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    traces, steps = [], 0
    total_candidates = total_scans = total_pairs = 0
    identities = np.zeros(len(CLASSES), dtype=np.int64)
    tracks = 0
    region2_pairs = 0
    checkpoint = None
    for epoch in range(epochs):
        losses = []
        identity_losses, association_losses = [], []
        for source in shards:
            raw, handle = _open(source)
            data = _validate(raw, source)
            center, scale = prior["center"], prior["scale"]
            for scan_lo, scan_hi, lo, hi in _blocks(data["offsets"], batch_rows):
                values = np.clip((data["features"][lo:hi].astype(np.float32) - center) / scale, -9., 9.)
                x = torch.as_tensor(values, device=device)
                labels = _transform_labels(data["labels"][lo:hi])
                y = torch.as_tensor(labels, device=device)
                row_offsets = data["offsets"][scan_lo:scan_hi + 1] - lo
                scan_weights = np.empty(hi - lo, dtype=np.float32)
                for scan in range(len(row_offsets) - 1):
                    first, last = int(row_offsets[scan]), int(row_offsets[scan + 1])
                    scan_weights[first:last] = 1. / max(1, last - first)
                scan_weights /= max(float(scan_weights.mean()), 1.e-12)
                # Region 2 is the known competing-vein region. It receives a
                # modest identity emphasis, but no ranking head is trained.
                scan_weights[data["regions"][lo:hi] == 2] *= 1.35
                weights = torch.as_tensor(scan_weights, device=device)
                pairs, pair_y = _association_pairs(
                    data["frames"][lo:hi], data["stations"][lo:hi], data["sectors"][lo:hi],
                    data["instance"][lo:hi], row_offsets, rng,
                    None if data["points"] is None else data["points"][lo:hi],
                    data["regions"][lo:hi],
                )
                if len(pairs):
                    pair_indices = torch.as_tensor(pairs, dtype=torch.long, device=device)
                    pair_target = torch.as_tensor(pair_y, dtype=torch.float32, device=device)
                    pair_order = torch.randperm(len(pairs), device=device)
                    tracks += int(np.unique(np.r_[data["instance"][lo:hi][pairs[:, 0]],
                                                  data["instance"][lo:hi][pairs[:, 1]]]).size)
                    total_pairs += len(pairs)
                    region2_pairs += int(np.sum(data["regions"][lo:hi][pairs[:, 0]] == 2))
                else:
                    pair_indices = pair_target = pair_order = None
                row_order = torch.randperm(len(y), device=device)
                for batch_no, row_batch in enumerate(row_order.split(DEFAULT_ROW_BATCH)):
                    logits, _ = model(x[row_batch])
                    identity_loss = (F.cross_entropy(logits, y[row_batch], reduction="none")
                                     * weights[row_batch]).mean()
                    association_loss = torch.zeros((), device=device)
                    if pair_indices is not None:
                        take = (torch.arange(min(DEFAULT_PAIR_BATCH, len(pair_indices)), device=device)
                                + batch_no * DEFAULT_PAIR_BATCH) % len(pair_indices)
                        chosen = pair_indices[pair_order[take]]
                        _, left = model(x[chosen[:, 0]])
                        _, right = model(x[chosen[:, 1]])
                        logit = 4. * (torch.sum(left * right, dim=1) - .40)
                        # Pair sampling is balanced, so the BCE has no hidden
                        # prevalence correction from the full route.
                        association_loss = F.binary_cross_entropy_with_logits(logit, pair_target[pair_order[take]])
                    loss = identity_loss + .18 * association_loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    steps += 1
                    losses.append(float(loss.detach().cpu()))
                    identity_losses.append(float(identity_loss.detach().cpu()))
                    association_losses.append(float(association_loss.detach().cpu()))
                total_candidates += hi - lo
                total_scans += scan_hi - scan_lo
                identities += np.bincount(labels, minlength=len(CLASSES))
            if handle is not None and hasattr(handle, "close"):
                handle.close()
        traces.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                       "identity": float(np.mean(identity_losses)), "association": float(np.mean(association_losses)),
                       "steps": len(losses)})
        checkpoint = {"model": _cpu_state(model), "optimizer": _cpu_optimizer(optimizer),
                      "feature_order": prior["feature_order"], "center": prior["center"], "scale": prior["scale"],
                      "classes": CLASSES, "variant": "v4_public_association", "local_feature_count": None,
                      "model_version": MODEL_VERSION, "epoch": epoch + 1, "steps": steps,
                      "seed": seed, "rng_numpy": rng.bit_generator.state, "rng_torch": torch.get_rng_state(),
                      "rng_cuda": torch.cuda.get_rng_state_all(), "shard_cursor": "epoch_boundary",
                      "training_trace": traces}
        _atomic_torch_save(destination / f"public_association_epoch_{epoch + 1:04d}.pt", checkpoint)
        print(f"v4 public association CUDA epoch {epoch + 1}/{epochs}, {steps} optimizer steps, "
              f"loss {traces[-1]['loss']:.4f}", flush=True)
    changed = any(not torch.equal(before[key], value.detach()) for key, value in model.state_dict().items())
    if not changed:
        raise RuntimeError("v4 reader parameters did not update on CUDA")
    final = destination / "public_association.pt"
    _atomic_torch_save(final, checkpoint)
    return {"weights": str(final), "sha256": _sha256(final), "device": str(device),
            "gpu": torch.cuda.get_device_name(device), "parameters": int(sum(x.numel() for x in model.parameters())),
            "epochs": epochs, "optimizer_steps": steps, "candidate_count": total_candidates // epochs,
            "scan_count": total_scans // epochs, "unique_track_id_count": int(tracks),
            "association_pairs_all_epochs": total_pairs, "region2_association_pairs_all_epochs": region2_pairs,
            "identity_counts_all_epochs": identities.tolist(), "parameter_update_verified": changed,
            "training_trace": traces}


def fit_public(shards: Sequence[Any] | Any, prior_path: str | Path, out: str | Path,
               *, epochs: int = 3, seed: int = 371700513, batch_rows: int = DEFAULT_BATCH_ROWS) -> dict:
    """Warm-start the compact public reader and train identity plus association on CUDA."""
    if epochs <= 0 or batch_rows <= 0:
        raise ValueError("epochs and batch_rows must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("v4 reader training requires CUDA")
    sources = _as_paths(shards)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    first_raw, first_handle = _open(sources[0])
    first = _validate(first_raw, sources[0])
    dimensions = first["dimensions"]
    if first_handle is not None and hasattr(first_handle, "close"):
        first_handle.close()
    device = torch.device("cuda:0")
    prior, base = _load_prior(Path(prior_path), dimensions, device)
    for source in sources[1:]:
        raw, handle = _open(source)
        check = _validate(raw, source)
        if check["dimensions"] != dimensions or check["feature_order"] != prior["feature_order"]:
            raise ValueError(f"{source}: feature contract differs from the warm-start reader")
        if handle is not None and hasattr(handle, "close"):
            handle.close()
    summary = _train(sources, prior, {k: v.detach().cpu() for k, v in base.state_dict().items()}, destination,
                     epochs=epochs, seed=seed, batch_rows=batch_rows, device=device)
    manifest = {"format": MODEL_VERSION, "feature_count": dimensions, "feature_order": list(prior["feature_order"]),
                "prior_reader": str(prior_path), "prior_sha256": _sha256(Path(prior_path)), "epochs": epochs,
                "seed": seed, "batch_rows": batch_rows, "device": str(device),
                "cuda_device": torch.cuda.get_device_name(device), "cuda_verified": True,
                "shards": [str(source) for source in sources], "weights": summary}
    _atomic_json(destination / "public_fit_manifest.json", manifest)
    return summary | {"manifest": str(destination / "public_fit_manifest.json"), "feature_count": dimensions,
                      "feature_order": list(prior["feature_order"])}


def load_public(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    order = tuple(payload.get("feature_order", ()))
    if not order:
        raise ValueError("v4 checkpoint has no public feature contract")
    model = Readout(len(order))
    model.load_state_dict(payload["model"])
    target = torch.device(device)
    model.to(target).eval()
    return {"model": model, "feature_order": order,
            "center": np.asarray(payload["center"], dtype=np.float32),
            "scale": np.maximum(np.asarray(payload["scale"], dtype=np.float32), .035),
            "variant": payload.get("variant", "v4_public_association"),
            "classes": tuple(payload.get("classes", CLASSES)), "device": str(target),
            "weights": str(path), "checkpoint": payload}


def predict_heads(features: np.ndarray, bundle: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(bundle["feature_order"]):
        raise ValueError("features do not match the v4 public feature contract")
    if not np.isfinite(values).all():
        raise ValueError("public features contain non-finite values")
    values = np.clip((values - bundle["center"]) / bundle["scale"], -9., 9.)
    model = bundle["model"]
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits, embedding = model(torch.as_tensor(values, dtype=torch.float32, device=device))
        probabilities = logits.softmax(dim=-1).cpu().numpy()
        embeddings = embedding.cpu().numpy()
    return probabilities, embeddings


__all__ = ["CLASSES", "Readout", "fit_public", "load_public", "predict_heads",
           "transform_labels"]

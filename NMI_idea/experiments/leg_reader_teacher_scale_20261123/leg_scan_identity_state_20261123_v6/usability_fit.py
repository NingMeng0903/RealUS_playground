"""Fit a local public-observation usability head for the v6 reader state.

The head is deliberately separate from identity and association.  Its target is
the observable acquisition proxy ``label >= 0`` used only while fitting on the
atlas simulation.  At deployment the prediction is a per-candidate local
confidence that an observed candidate is physically interpretable.  It is not
multiplied into an identity posterior by this module.
"""

from __future__ import annotations

from argparse import ArgumentParser
import gzip
from pathlib import Path
import hashlib
import json
import os
import random
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


EXPERIMENT = "leg_scan_identity_state_20261123_v6"
DEFAULT_V5_RUN = Path(__file__).resolve().parents[1] / "leg_scan_anchor_state_20261123_v5" / "run"
DEFAULT_OUT = Path(__file__).resolve().parent / "run"
DEFAULT_SHARDS = 8
TRAIN_SHARDS = 6
VAL_SHARDS = 2
SEED = 6_021_114
HIDDEN = 24

# These names are present in the v5 corrected public feature contract.  They
# describe one candidate and its local support only.  In particular, no label,
# instance id, source name, or future outcome is read into this vector.
LOCAL_FIELD_NAMES = (
    "public_evidence", "target_posterior", "unknown_posterior", "merge_posterior",
    "same_frame_rival_separation", "echo_lateral", "echo_depth", "probe_station_fraction",
    "probe_sector_sin", "probe_sector_cos", "probe_tick_fraction", "probe_quality",
    "material_axial_fraction", "material_cross_section_u", "material_cross_section_v",
    "material_sector_sin", "material_sector_cos", "material_supported",
    "harmonic_theta_sin", "harmonic_theta_cos", "harmonic_h", "harmonic_d",
    "harmonic_supported", "harmonic_registration_error", "atlas_role_0_proximity",
    "atlas_role_0_distance", "atlas_target_hypothesis_count", "atlas_target_hypothesis_entropy",
    "atlas_proximal_link_support", "atlas_distal_link_support", "atlas_same_component_support",
    "atlas_supported_hypothesis_mass", "history_track_support", "history_bidirectional_support",
    "history_correlation_discount", "history_distinct_view_count", "history_neighbor_frame_count",
    "history_cross_window_fraction",
)

_BASELINE_NAMES = (
    "public_evidence", "target_posterior", "unknown_posterior", "merge_posterior",
    "same_frame_rival_separation", "echo_lateral", "echo_depth", "probe_station_fraction",
    "probe_sector_sin", "probe_sector_cos", "probe_tick_fraction", "probe_quality",
)
_TAIL_NAMES = (
    "material_axial_fraction", "material_cross_section_u", "material_cross_section_v",
    "material_sector_sin", "material_sector_cos", "material_supported", "harmonic_theta_sin",
    "harmonic_theta_cos", "harmonic_h", "harmonic_d", "harmonic_supported",
    "harmonic_registration_error", "atlas_role_0_proximity", "atlas_role_0_distance",
    "atlas_role_1_proximity", "atlas_role_1_distance", "atlas_role_2_proximity",
    "atlas_role_2_distance", "atlas_role_3_proximity", "atlas_role_3_distance",
    "atlas_role_4_proximity", "atlas_role_4_distance", "atlas_role_5_proximity",
    "atlas_role_5_distance", "atlas_target_hypothesis_count", "atlas_target_hypothesis_entropy",
    "atlas_proximal_link_support", "atlas_distal_link_support", "atlas_same_component_support",
    "atlas_supported_hypothesis_mass", "history_track_support", "history_bidirectional_support",
    "history_correlation_discount", "history_distinct_view_count", "history_neighbor_frame_count",
    "history_cross_window_fraction",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _default_order() -> tuple[str, ...]:
    """Return the v4/v5 feature order when a runtime public object omits it."""
    # The acquisition contract is baseline, baseline availability, extended
    # tail, then tail availability.  This order is intentionally explicit so a
    # runtime public object without ``feature_order`` remains compatible.
    return tuple(_BASELINE_NAMES + tuple("available_" + name for name in _BASELINE_NAMES)
                 + _TAIL_NAMES + tuple("available_" + name for name in _TAIL_NAMES))


def _field_indices(feature_order: Any) -> tuple[list[int], list[str]]:
    order = tuple(str(x) for x in (feature_order if feature_order is not None else _default_order()))
    missing = [name for name in LOCAL_FIELD_NAMES if name not in order]
    if missing:
        raise ValueError(f"Public feature contract lacks local usability fields: {missing}")
    indices = [order.index(name) for name in LOCAL_FIELD_NAMES]
    return indices, list(LOCAL_FIELD_NAMES)


def _array_feature_matrix(features: np.ndarray, quality: np.ndarray, stations: np.ndarray,
                          sectors: np.ndarray, points: np.ndarray, indices: list[int]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] <= max(indices, default=-1):
        raise ValueError(f"public features must have [candidate, >=96], got {features.shape}")
    quality = np.asarray(quality, dtype=np.float32).reshape(-1)
    stations = np.asarray(stations, dtype=np.float32).reshape(-1)
    sectors = np.asarray(sectors, dtype=np.float32).reshape(-1)
    points = np.asarray(points, dtype=np.float32)
    n = len(features)
    if not (len(quality) == len(stations) == len(sectors) == len(points) == n and points.shape == (n, 2)):
        raise ValueError("public fields have inconsistent candidate lengths")
    selected = features[:, indices]
    # The feature contract already contains probe quality and angle encoding.
    # These explicit public fields make the runtime contract robust to callers
    # that construct an embedding without all optional availability columns.
    angle = sectors * (2. * np.pi / 24.)
    local = np.c_[selected, quality, np.sin(angle), np.cos(angle), points]
    return np.nan_to_num(local, nan=0., posinf=8., neginf=-8.).astype(np.float32, copy=False)


def _load_shard(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = ("features", "quality", "stations", "sectors", "points", "labels", "feature_order")
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(f"{path} misses required public fields {missing}")
        indices, names = _field_indices(data["feature_order"])
        values = _array_feature_matrix(data["features"], data["quality"], data["stations"],
                                       data["sectors"], data["points"], indices)
        labels = (np.asarray(data["labels"], dtype=np.int8) >= 0).astype(np.float32)
        artifacts = np.asarray(data["labels"], dtype=np.int8)
    if len(values) != len(labels):
        raise ValueError(f"{path} public fields and labels differ")
    return values, labels, {"field_indices": indices, "field_names": names,
                            "artifact_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(artifacts, return_counts=True))}}


class _UsabilityHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1.
    negative = labels == 0.
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    ties = {}
    for value in np.unique(scores):
        ids = np.flatnonzero(scores == value)
        ranks[ids] = ranks[ids].mean()
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1.) / 2.)
                 / (positive.sum() * negative.sum()))


def _metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    brier = float(np.mean((scores - labels) ** 2))
    threshold = .5
    predicted = scores >= threshold
    return {"rows": int(len(labels)), "positive": int(labels.sum()), "positive_rate": float(labels.mean()),
            "roc_auc": _rank_auc(scores, labels), "brier": brier,
            "threshold": threshold, "predicted_positive": int(predicted.sum()),
            "true_positive": int(np.sum(predicted & (labels == 1.))),
            "false_positive": int(np.sum(predicted & (labels == 0.)))}


def _split_files(v5_run: Path, max_shards: int) -> tuple[list[Path], list[Path]]:
    files = sorted((v5_run / "fit_revisit").glob("*/data.npz"))[:max_shards]
    if len(files) < 4:
        raise FileNotFoundError(f"Need at least four v5 fit_revisit shards under {v5_run}")
    split = max(2, len(files) - max(2, len(files) // 4))
    if split >= len(files):
        split = len(files) - 2
    return files[:split], files[split:]


def _layout_ids(data_path: Path) -> tuple[str, ...]:
    """Read the archived layout identifiers for one fit shard.

    ``snapshot_case`` in the NPZ is a shard-local index and is intentionally
    not used for the split: it restarts at zero in every shard.  The archived
    JSONL is the authoritative case/layout manifest.
    """
    manifest = data_path.parent / "cases.jsonl.gz"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing fit shard layout manifest: {manifest}")
    layouts = []
    with gzip.open(manifest, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            layout = record.get("layout_id")
            if not layout:
                raise ValueError(f"{manifest}:{line_number} has no layout_id")
            layouts.append(str(layout))
    if not layouts or len(layouts) != len(set(layouts)):
        raise ValueError(f"{manifest} must contain one unique layout_id per case")
    return tuple(layouts)


def _normalization(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train_x.std(axis=0, dtype=np.float64), .05).astype(np.float32)
    return center, scale


def _export_bundle(model: _UsabilityHead, center: np.ndarray, scale: np.ndarray,
                   field_indices: list[int], field_names: list[str], *, threshold: float = .5) -> dict[str, Any]:
    state = {name: value.detach().cpu().numpy().astype(np.float32).tolist() for name, value in model.state_dict().items()}
    return {"version": 1, "experiment": EXPERIMENT, "target": "physical_candidate_label_ge_0",
            "feature_contract": "public local candidate features, explicit quality, sector angle and point",
            "field_indices": [int(x) for x in field_indices], "field_names": field_names,
            "center": center.tolist(), "scale": scale.tolist(), "hidden": HIDDEN,
            "state_dict": state, "threshold": float(threshold), "device_training": "cuda:0"}


def predict_usability(public: Mapping[str, Any], bundle: Mapping[str, Any]) -> np.ndarray:
    """Return per-candidate physical/usability probability on CPU.

    ``public`` must contain label-free ``features``, ``quality``, ``stations``,
    ``sectors`` and ``points``.  The function never accepts or inspects labels.
    """
    indices = [int(x) for x in bundle["field_indices"]]
    x = _array_feature_matrix(public["features"], public["quality"], public["stations"],
                              public["sectors"], public["points"], indices)
    center = np.asarray(bundle["center"], dtype=np.float32)
    scale = np.asarray(bundle["scale"], dtype=np.float32)
    if x.shape[1] != len(center) or len(center) != len(scale):
        raise ValueError("Usability bundle and public feature width differ")
    x = np.clip((x - center) / scale, -9., 9.)
    state = {key: np.asarray(value, dtype=np.float32) for key, value in bundle["state_dict"].items()}
    x = x @ state["net.0.weight"].T + state["net.0.bias"]
    x = x * (1. / (1. + np.exp(-np.clip(x, -40., 40.))))
    x = x @ state["net.2.weight"].T + state["net.2.bias"]
    return (1. / (1. + np.exp(-np.clip(x[:, 0], -40., 40.)))).astype(np.float32)


def fit_usability(v5_run: str | os.PathLike[str] | Path = DEFAULT_V5_RUN,
                  outdir: str | os.PathLike[str] | Path = DEFAULT_OUT,
                  *, max_shards: int = DEFAULT_SHARDS, epochs: int = 4,
                  batch_size: int = 8192, seed: int = SEED) -> dict[str, Any]:
    """Fit on disjoint v5 fit shards with actual CUDA optimizer updates."""
    v5_run, outdir = Path(v5_run).resolve(), Path(outdir).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v6 usability fit")
    if not (4 <= max_shards <= 8):
        raise ValueError("max_shards must be between 4 and 8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    train_files, val_files = _split_files(v5_run, max_shards)
    train_layouts = tuple(layout for path in train_files for layout in _layout_ids(path))
    val_layouts = tuple(layout for path in val_files for layout in _layout_ids(path))
    if len(train_layouts) != len(set(train_layouts)) or len(val_layouts) != len(set(val_layouts)):
        raise ValueError("Selected v5 fit shards contain duplicate layouts within a split")
    layout_overlap = sorted(set(train_layouts).intersection(val_layouts))
    if layout_overlap:
        raise ValueError(f"v5 fit layout split is not disjoint: {layout_overlap[:5]}")
    train_x, train_y, train_meta, val_x, val_y, val_meta = [], [], [], [], [], []
    for path in train_files:
        x, y, meta = _load_shard(path); train_x.append(x); train_y.append(y); train_meta.append(meta)
    for path in val_files:
        x, y, meta = _load_shard(path); val_x.append(x); val_y.append(y); val_meta.append(meta)
    train_x, train_y = np.concatenate(train_x), np.concatenate(train_y)
    val_x, val_y = np.concatenate(val_x), np.concatenate(val_y)
    center, scale = _normalization(train_x)
    train_z = np.clip((train_x - center) / scale, -9., 9.)
    val_z = np.clip((val_x - center) / scale, -9., 9.)
    model = _UsabilityHead(train_z.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.002, fused=True)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x_cuda = torch.as_tensor(train_z, device="cuda", dtype=torch.float32)
    y_cuda = torch.as_tensor(train_y, device="cuda", dtype=torch.float32)
    trace = []
    for epoch in range(epochs):
        permutation = torch.randperm(len(x_cuda), generator=generator, device="cuda")
        losses = []
        model.train()
        for indices in permutation.split(batch_size):
            logits = model(x_cuda[indices])
            loss = criterion(logits, y_cuda[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            train_score = torch.sigmoid(model(x_cuda)).cpu().numpy()
            val_score = torch.sigmoid(model(torch.as_tensor(val_z, device="cuda"))).cpu().numpy()
        trace.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                      "train": _metrics(train_score, train_y), "validation": _metrics(val_score, val_y)})
        print(f"usability CUDA {epoch + 1}/{epochs}, loss {trace[-1]['loss']:.5f}, "
              f"val_auc {trace[-1]['validation']['roc_auc']:.4f}", flush=True)
    torch.cuda.synchronize()
    # Preserve the exact indices found in the NPZ contract.  The default order
    # is only a fallback for runtime public objects that omit feature_order.
    indices, names = train_meta[0]["field_indices"], train_meta[0]["field_names"]
    bundle = _export_bundle(model, center, scale, indices, names)
    outdir.mkdir(parents=True, exist_ok=True)
    bundle_path = outdir / "usability_bundle.json"
    checkpoint_path = outdir / "usability_head.pt"
    _atomic_json(bundle_path, bundle)
    _atomic_torch(checkpoint_path, {"bundle": bundle, "optimizer": optimizer.state_dict(), "seed": seed})
    summary = {"experiment": EXPERIMENT, "v5_run": str(v5_run), "train_shards": [str(p) for p in train_files],
               "validation_shards": [str(p) for p in val_files], "train_layout_shards": len(train_files),
               "validation_layout_shards": len(val_files), "train_layout_count": len(train_layouts),
               "validation_layout_count": len(val_layouts), "layout_overlap_count": len(layout_overlap),
               "layout_split_verified": True, "train_rows": int(len(train_y)),
               "validation_rows": int(len(val_y)), "train_artifact_counts": train_meta,
               "validation_artifact_counts": val_meta, "epochs": epochs, "batch_size": batch_size,
               "seed": seed, "device": "cuda:0", "parameter_count": int(sum(x.numel() for x in model.parameters())),
               "cuda_optimizer_step_verified": True, "trace": trace,
               "bundle": str(bundle_path), "bundle_sha256": _sha256(bundle_path),
               "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
               "target_definition": "label >= 0 physical candidate; labels -1 and -2 are artifacts/unknowns",
               "runtime_contract": "features, quality, stations, sectors, points only; no labels or instances"}
    _atomic_json(outdir / "usability_fit.json", summary)
    return summary


def _main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--v5-run", type=Path, default=DEFAULT_V5_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()
    fit_usability(args.v5_run, args.out, max_shards=args.max_shards, epochs=args.epochs)


if __name__ == "__main__":
    _main()

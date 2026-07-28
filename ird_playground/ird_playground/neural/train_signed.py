"""Train and evaluate the RM4D signed reachability field."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import json
import numpy as np
import time
import yaml

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from ird_playground.neural.signed_field import ReachabilitySDF, SignedReachabilityField


@dataclass(frozen=True)
class SignedTrainConfig:
    gt_npz: str
    checkpoint: str
    report: str
    width: int = 192
    depth: int = 5
    fourier_bands: int = 3
    softplus_beta: float = 20.0
    seed: int = 47
    device: str = "cuda"
    epochs: int = 30
    batch_size: int = 4096
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    val_fraction: float = 0.15
    early_stop_patience: int = 6
    logit_scale: float = 3.0
    normal_batch_max: int = 512
    lambda_classification: float = 1.0
    lambda_signed_value: float = 1.0
    lambda_normal_direction: float = 0.25
    lambda_eikonal: float = 0.05
    max_global_rows: int | None = None
    max_boundary_groups: int | None = None


def load_signed_train_config(path: str | Path, *, root: Path | None = None) -> SignedTrainConfig:
    path = Path(path)
    root = root or path.resolve().parents[1]
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    build, model = dict(raw.get("build") or {}), dict(raw.get("model") or {})
    train, loss, io = dict(raw.get("training") or {}), dict(raw.get("loss") or {}), dict(raw.get("io") or {})

    def resolve(value: str) -> str:
        p = Path(value)
        return str(p if p.is_absolute() else root / p)

    gt = resolve(str(build.get("output_npz", "data/ird/rm4d_signed_production.npz")))
    if not Path(gt).is_file():
        raise FileNotFoundError(gt)
    return SignedTrainConfig(
        gt_npz=gt,
        checkpoint=resolve(str(io.get("checkpoint", "data/checkpoints/rm4d_signed/selected.pt"))),
        report=resolve(str(io.get("report", "data/reports/train_rm4d_signed.json"))),
        width=int(model.get("width", 192)),
        depth=int(model.get("depth", 5)),
        fourier_bands=int(model.get("fourier_bands", 3)),
        softplus_beta=float(model.get("softplus_beta", 20.0)),
        seed=int(train.get("seed", 47)),
        device=str(train.get("device", "cuda")),
        epochs=int(train.get("epochs", 30)),
        batch_size=int(train.get("batch_size", 4096)),
        learning_rate=float(train.get("learning_rate", 3.0e-4)),
        weight_decay=float(train.get("weight_decay", 1.0e-4)),
        val_fraction=float(train.get("val_fraction", 0.15)),
        early_stop_patience=int(train.get("early_stop_patience", 6)),
        logit_scale=float(train.get("logit_scale", 3.0)),
        normal_batch_max=int(train.get("normal_batch_max", 512)),
        lambda_classification=float(loss.get("classification", 1.0)),
        lambda_signed_value=float(loss.get("signed_value", 1.0)),
        lambda_normal_direction=float(loss.get("normal_direction", 0.25)),
        lambda_eikonal=float(loss.get("eikonal", 0.05)),
        max_global_rows=(None if train.get("max_global_rows") in (None, "null") else int(train["max_global_rows"])),
        max_boundary_groups=(None if train.get("max_boundary_groups") in (None, "null") else int(train["max_boundary_groups"])),
    )


def _subset_training_arrays(
    arrays: dict[str, np.ndarray],
    *,
    max_global_rows: int | None,
    max_boundary_groups: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    if max_global_rows is None and max_boundary_groups is None:
        return arrays
    bid = arrays["boundary_id"]
    rng = np.random.default_rng(seed)
    global_idx = np.flatnonzero(bid < 0)
    if max_global_rows is not None and len(global_idx) > max_global_rows:
        global_idx = rng.choice(global_idx, size=max_global_rows, replace=False)
    groups = np.unique(bid[bid >= 0])
    if max_boundary_groups is not None and len(groups) > max_boundary_groups:
        groups = rng.choice(groups, size=max_boundary_groups, replace=False)
    boundary_idx = np.flatnonzero(np.isin(bid, groups))
    keep = np.sort(np.concatenate((global_idx, boundary_idx)))
    n = len(bid)
    return {
        key: (value[keep] if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n else value)
        for key, value in arrays.items()
    }


def _split_indices(
    boundary_id: np.ndarray,
    fraction: float,
    seed: int,
    source_pose_id: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Group validation rows by source pose, then by boundary group."""
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(boundary_id), dtype=bool)
    source = (
        np.full(len(boundary_id), -1, dtype=np.int64)
        if source_pose_id is None
        else np.asarray(source_pose_id, dtype=np.int64)
    )
    grouped_sources = np.unique(source[source >= 0])
    rng.shuffle(grouped_sources)
    if len(grouped_sources):
        val_sources = grouped_sources[: max(1, int(len(grouped_sources) * fraction))]
        is_val[source >= 0] = np.isin(source[source >= 0], val_sources)

    base = np.flatnonzero((boundary_id < 0) & (source < 0))
    if len(base):
        count = max(1, int(len(base) * fraction))
        is_val[rng.choice(base, size=min(count, len(base)), replace=False)] = True

    unresolved_boundary = (boundary_id >= 0) & (source < 0)
    groups = np.unique(boundary_id[unresolved_boundary])
    rng.shuffle(groups)
    if len(groups):
        val_groups = groups[: max(1, int(len(groups) * fraction))]
        is_val[unresolved_boundary] = np.isin(
            boundary_id[unresolved_boundary], val_groups
        )
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def _side_indices(boundary_id: np.ndarray, signed: np.ndarray, positive: bool, nearest: bool) -> tuple[np.ndarray, np.ndarray]:
    side = signed > 0 if positive else signed < 0
    idx = np.flatnonzero((boundary_id >= 0) & side & np.isfinite(signed))
    secondary = np.abs(signed[idx]) if nearest else -np.abs(signed[idx])
    order = np.lexsort((secondary, boundary_id[idx]))
    idx = idx[order]
    groups, first = np.unique(boundary_id[idx], return_index=True)
    return groups, idx[first]


def evaluate_signed_field(field: ReachabilitySDF, arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    pred = field.score_np(arrays["canonical"][idx])
    supervised = arrays.get("classification_weight", np.ones(len(arrays["reachable"]), dtype=np.float32))[idx] > 0
    y = arrays["reachable"][idx][supervised] > 0.5
    positive = pred[supervised] >= 0.0
    tpr = float(positive[y].mean()) if y.any() else 0.0
    tnr = float((~positive[~y]).mean()) if (~y).any() else 0.0
    metrics = {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "reachable_recall": tpr,
        "unreachable_specificity": tnr,
        "accuracy": float((positive == y).mean()),
        "n": int(supervised.sum()),
        "n_with_zero_boundary": int(len(idx)),
    }
    local_bid = arrays["boundary_id"][idx]
    for kind, suffix, signed_key in ((0, "m", "boundary_signed_m"), (1, "deg", "boundary_signed_rot_deg")):
        signed = arrays[signed_key][idx]
        valid_kind = arrays["clearance_kind"][idx] == kind
        b = np.where(valid_kind, local_bid, -1)
        gp, ip = _side_indices(b, signed, True, True)
        gn, inn = _side_indices(b, signed, False, True)
        common, pa, na = np.intersect1d(gp, gn, assume_unique=True, return_indices=True)
        ip, inn = ip[pa], inn[na]
        fp, fn = pred[ip], pred[inn]
        direction = fp > fn
        straddle = (fp >= 0.0) & (fn < 0.0)
        denom = fp - fn
        crossing = np.full(len(common), np.nan, dtype=np.float64)
        usable = straddle & (np.abs(denom) > 1.0e-9)
        crossing[usable] = signed[inn[usable]] + (-fn[usable]) * (
            signed[ip[usable]] - signed[inn[usable]]
        ) / denom[usable]
        gpw, ipw = _side_indices(b, signed, True, False)
        gnw, inw = _side_indices(b, signed, False, False)
        commonw, paw, naw = np.intersect1d(gpw, gnw, assume_unique=True, return_indices=True)
        fpw, fnw = pred[ipw[paw]], pred[inw[naw]]
        wide = (fpw >= 0.0) & (fnw < 0.0)
        finite = np.abs(crossing[np.isfinite(crossing)])
        metrics.update({
            f"direction_agreement_{suffix}": float(direction.mean()) if len(direction) else 0.0,
            f"strict_straddle_rate_{suffix}": float(straddle.mean()) if len(straddle) else 0.0,
            f"wide_straddle_rate_{suffix}": float(wide.mean()) if len(wide) else 0.0,
            f"crossing_mae_{suffix}": float(finite.mean()) if len(finite) else float("inf"),
            f"crossing_p95_{suffix}": float(np.percentile(finite, 95)) if len(finite) else float("inf"),
            f"boundary_groups_{suffix}": int(len(common)),
        })
    return metrics


def _selection(metrics: dict[str, float]) -> float:
    return (
        metrics["balanced_accuracy"]
        + 0.25 * metrics.get("direction_agreement_m", 0.0)
        + 0.25 * metrics.get("direction_agreement_deg", 0.0)
        + 0.15 * metrics.get("strict_straddle_rate_m", 0.0)
        + 0.15 * metrics.get("strict_straddle_rate_deg", 0.0)
        + 0.05 * metrics.get("wide_straddle_rate_m", 0.0)
        + 0.05 * metrics.get("wide_straddle_rate_deg", 0.0)
    )


def train_signed_field(cfg: SignedTrainConfig) -> dict:
    if torch is None:
        raise ImportError("torch required")
    started = time.perf_counter()
    raw = np.load(cfg.gt_npz, allow_pickle=False)
    dataset_meta = json.loads(str(raw["meta_json"].item())) if "meta_json" in raw.files else {}
    arrays = {k: raw[k] for k in raw.files if k != "meta_json"}
    arrays = _subset_training_arrays(
        arrays,
        max_global_rows=cfg.max_global_rows,
        max_boundary_groups=cfg.max_boundary_groups,
        seed=cfg.seed + 1000,
    )
    tr_idx, va_idx = _split_indices(
        arrays["boundary_id"],
        cfg.val_fraction,
        cfg.seed,
        arrays.get("source_pose_id"),
    )
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = SignedReachabilityField(
        width=cfg.width,
        depth=cfg.depth,
        fourier_bands=cfg.fourier_bands,
        softplus_beta=cfg.softplus_beta,
        input_center=arrays["input_center"],
        input_scale=arrays["input_scale"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    supervised_train = arrays.get("classification_weight", np.ones(len(arrays["reachable"]), dtype=np.float32))[tr_idx] > 0
    pos = arrays["reachable"][tr_idx][supervised_train] > 0.5
    pos_weight = float((~pos).sum() / max(pos.sum(), 1))
    best_score = -float("inf")
    best_state = None
    stale = 0
    history = []
    checkpoint = Path(cfg.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        order = rng.permutation(tr_idx)
        totals = np.zeros(5, dtype=np.float64)
        batches = 0
        model.train()
        for start in range(0, len(order), cfg.batch_size):
            ids = order[start : start + cfg.batch_size]
            canonical = torch.as_tensor(arrays["canonical"][ids], device=device)
            xn = model.normalize(canonical).detach().requires_grad_(True)
            target_y = torch.as_tensor(arrays["reachable"][ids], device=device)
            classification_weight = torch.as_tensor(arrays.get("classification_weight", np.ones(len(arrays["reachable"]), dtype=np.float32))[ids], device=device)
            target_sdf = torch.as_tensor(arrays["sdf_target"][ids], device=device)
            sdf_weight = torch.as_tensor(arrays["sdf_weight"][ids], device=device)
            normal = torch.as_tensor(arrays["normal"][ids], device=device)
            normal_weight = torch.as_tensor(arrays["normal_weight"][ids], device=device)
            slope = torch.as_tensor(arrays["normal_slope"][ids], device=device)
            pred = model.forward_normalized(xn)
            per_cls = F.binary_cross_entropy_with_logits(
                cfg.logit_scale * pred,
                target_y,
                pos_weight=torch.tensor(pos_weight, device=device),
                reduction="none",
            )
            cls = (per_cls * classification_weight).sum() / classification_weight.sum().clamp_min(1.0)
            smask = sdf_weight > 0
            sdf = F.smooth_l1_loss(pred[smask], target_sdf[smask], beta=0.1) if smask.any() else pred.new_zeros(())
            nids = torch.nonzero(normal_weight > 0, as_tuple=False).flatten()
            if len(nids) > cfg.normal_batch_max:
                nids = nids[torch.randperm(len(nids), device=device)[: cfg.normal_batch_max]]
            if len(nids):
                grad_all = torch.autograd.grad(pred[nids].sum(), xn, create_graph=True)[0]
                grad = grad_all[nids]
                normal_loss = (1.0 - F.cosine_similarity(grad, normal[nids], dim=-1, eps=1.0e-8)).mean()
                eikonal = F.smooth_l1_loss(torch.linalg.vector_norm(grad, dim=-1), slope[nids], beta=0.2)
            else:
                normal_loss = pred.new_zeros(())
                eikonal = pred.new_zeros(())
            loss = (
                cfg.lambda_classification * cls
                + cfg.lambda_signed_value * sdf
                + cfg.lambda_normal_direction * normal_loss
                + cfg.lambda_eikonal * eikonal
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            totals += [float(loss.detach()), float(cls.detach()), float(sdf.detach()), float(normal_loss.detach()), float(eikonal.detach())]
            batches += 1
        field = ReachabilitySDF(model, device=str(device))
        metrics = evaluate_signed_field(field, arrays, va_idx)
        score = _selection(metrics)
        row = {"epoch": epoch, "loss": float(totals[0] / batches), "classification_loss": float(totals[1] / batches), "signed_loss": float(totals[2] / batches), "normal_loss": float(totals[3] / batches), "eikonal_loss": float(totals[4] / batches), "selection_score": score, **metrics}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.4f} bal={metrics['balanced_accuracy']:.3f} "
            f"dir_m={metrics.get('direction_agreement_m',0):.3f} dir_deg={metrics.get('direction_agreement_deg',0):.3f} "
            f"wide_m={metrics.get('wide_straddle_rate_m',0):.3f} wide_deg={metrics.get('wide_straddle_rate_deg',0):.3f} sel={score:.3f}",
            flush=True,
        )
        if score > best_score + 1.0e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.early_stop_patience:
            print(f"early stop at epoch={epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    field = ReachabilitySDF(model, device=str(device))
    final_metrics = evaluate_signed_field(field, arrays, va_idx)
    field.save(
        checkpoint,
        meta={
            "training_config": asdict(cfg),
            "dataset_meta": dataset_meta,
            "robot_contract": dataset_meta.get("robot_contract"),
            "metrics": final_metrics,
            "selection_score": best_score,
        },
    )
    result = {
        "checkpoint": str(checkpoint),
        "metrics": final_metrics,
        "history": history,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_total": int(len(arrays["boundary_id"])),
        "n_global": int((arrays["boundary_id"] < 0).sum()),
        "n_boundary_groups": int(np.unique(arrays["boundary_id"][arrays["boundary_id"] >= 0]).size),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report = Path(cfg.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["SignedTrainConfig", "evaluate_signed_field", "load_signed_train_config", "train_signed_field"]

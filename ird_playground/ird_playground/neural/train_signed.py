"""Train and evaluate the flange-chart signed reachability field."""

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

from ird_playground.ird.metric import unit_speed_eikonal_loss
from ird_playground.neural.signed_field import (
    NEAR_AXIS_R_M,
    ReachabilitySDF,
    SignedReachabilityField,
    assert_fitted_normalization,
    compute_input_stats,
)

# External holdout modes.  ``orbit`` prefers an explicit ``orbit_id`` column;
# if absent, quantized ``q_best`` is accepted as a stand-in joint-orbit key.
EXTERNAL_SPLIT_MODES = ("block", "orbit")
EXTERNAL_SPLIT_REQUIRED_KEYS = {
    "block": ("block_id",),
    "orbit": ("orbit_id", "q_best"),  # at least one of these
}


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
    report_near_axis: bool = True
    near_axis_r_m: float = NEAR_AXIS_R_M
    external_test_mode: str | None = None
    external_test_fraction: float = 0.1


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
        report_near_axis=bool(train.get("report_near_axis", True)),
        near_axis_r_m=float(train.get("near_axis_r_m", NEAR_AXIS_R_M)),
        external_test_mode=(
            None
            if train.get("external_test_mode") in (None, "null", "")
            else str(train["external_test_mode"])
        ),
        external_test_fraction=float(train.get("external_test_fraction", 0.1)),
    )


def require_source_pose_id(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Return ``source_pose_id`` or raise — no silent fallback."""
    if "source_pose_id" not in arrays:
        raise KeyError(
            "source_pose_id is required for grouped train/val splits; "
            "silent fallback is forbidden"
        )
    source = np.asarray(arrays["source_pose_id"], dtype=np.int64).reshape(-1)
    n = len(arrays["boundary_id"]) if "boundary_id" in arrays else len(arrays["canonical"])
    if source.shape[0] != n:
        raise ValueError(f"source_pose_id length {source.shape[0]} != n={n}")
    return source


def _orbit_ids_from_arrays(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "orbit_id" in arrays:
        return np.asarray(arrays["orbit_id"], dtype=np.int64).reshape(-1)
    if "q_best" not in arrays:
        raise KeyError(
            "external_test_mode='orbit' requires ``orbit_id`` or ``q_best``; "
            f"available keys: {sorted(arrays)}"
        )
    q = np.asarray(arrays["q_best"], dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"q_best must be (N, dof), got {q.shape}")
    # Coarse joint bins → stable orbit proxy without an explicit orbit_id column.
    bins = np.floor(q / (np.pi / 12.0)).astype(np.int64)
    # Hash joints into a single int64 id.
    orbit = np.zeros(len(bins), dtype=np.int64)
    for col in range(bins.shape[1]):
        orbit = orbit * 97 + (bins[:, col] + 64)
    return orbit


def external_holdout_indices(
    arrays: dict[str, np.ndarray],
    *,
    mode: str,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-holdout by workspace ``block_id`` or joint ``orbit_id``.

    Returns ``(keep_idx, holdout_idx)``.  Holdout groups are disjoint from keep.
    """
    if mode not in EXTERNAL_SPLIT_MODES:
        raise ValueError(
            f"unknown external_test_mode={mode!r}; expected one of {EXTERNAL_SPLIT_MODES}. "
            f"Required keys: {EXTERNAL_SPLIT_REQUIRED_KEYS}"
        )
    if mode == "block":
        if "block_id" not in arrays:
            raise KeyError(
                "external_test_mode='block' requires ``block_id``. "
                f"Required keys by mode: {EXTERNAL_SPLIT_REQUIRED_KEYS}"
            )
        group = np.asarray(arrays["block_id"], dtype=np.int64).reshape(-1)
    else:
        group = _orbit_ids_from_arrays(arrays)
    n = len(group)
    rng = np.random.default_rng(seed)
    uniq = np.unique(group)
    rng.shuffle(uniq)
    n_hold = max(1, int(round(len(uniq) * float(fraction)))) if len(uniq) else 0
    hold_groups = set(uniq[:n_hold].tolist())
    is_hold = np.isin(group, list(hold_groups))
    return np.flatnonzero(~is_hold), np.flatnonzero(is_hold)


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
    source_pose_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Group validation rows by ``source_pose_id`` (required), then boundary."""
    if source_pose_id is None:
        raise KeyError(
            "source_pose_id is required for grouped train/val splits; "
            "silent fallback is forbidden"
        )
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(boundary_id), dtype=bool)
    source = np.asarray(source_pose_id, dtype=np.int64)
    if source.shape[0] != len(boundary_id):
        raise ValueError("source_pose_id length must match boundary_id")

    grouped_sources = np.unique(source[source >= 0])
    rng.shuffle(grouped_sources)
    if len(grouped_sources):
        val_sources = grouped_sources[: max(1, int(len(grouped_sources) * fraction))]
        is_val[source >= 0] = np.isin(source[source >= 0], val_sources)

    # Rows without a source id (should be rare after Phase-3 GT); still group by boundary.
    unresolved = source < 0
    base = np.flatnonzero((boundary_id < 0) & unresolved)
    if len(base):
        count = max(1, int(len(base) * fraction))
        is_val[rng.choice(base, size=min(count, len(base)), replace=False)] = True

    unresolved_boundary = (boundary_id >= 0) & unresolved
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


def evaluate_signed_field(
    field: ReachabilitySDF,
    arrays: dict[str, np.ndarray],
    idx: np.ndarray,
    *,
    report_near_axis: bool = True,
    near_axis_r_m: float = NEAR_AXIS_R_M,
) -> dict[str, float]:
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
        "accuracy": float((positive == y).mean()) if y.size else 0.0,
        "n": int(supervised.sum()),
        "n_with_zero_boundary": int(len(idx)),
    }
    if report_near_axis:
        r = np.asarray(arrays["canonical"][idx, 1], dtype=np.float64)
        near = supervised & (r < float(near_axis_r_m))
        if near.any():
            y_near = arrays["reachable"][idx][near] > 0.5
            p_near = pred[near] >= 0.0
            tpr_n = float(p_near[y_near].mean()) if y_near.any() else 0.0
            tnr_n = float((~p_near[~y_near]).mean()) if (~y_near).any() else 0.0
            metrics.update(
                {
                    "near_axis_r_lt_5cm_n": int(near.sum()),
                    "near_axis_r_lt_5cm_accuracy": float((p_near == y_near).mean()),
                    "near_axis_r_lt_5cm_balanced_accuracy": 0.5 * (tpr_n + tnr_n),
                    "near_axis_r_m": float(near_axis_r_m),
                }
            )
        else:
            metrics.update(
                {
                    "near_axis_r_lt_5cm_n": 0,
                    "near_axis_r_lt_5cm_accuracy": float("nan"),
                    "near_axis_r_lt_5cm_balanced_accuracy": float("nan"),
                    "near_axis_r_m": float(near_axis_r_m),
                }
            )
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


def _resolve_input_normalization(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if "input_center" in arrays and "input_scale" in arrays:
        center = np.asarray(arrays["input_center"], dtype=np.float32).reshape(-1)
        scale = np.asarray(arrays["input_scale"], dtype=np.float32).reshape(-1)
        try:
            assert_fitted_normalization(center, scale)
            return center, scale
        except ValueError:
            pass
    return compute_input_stats(arrays["canonical"][train_idx])


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
    source_pose_id = require_source_pose_id(arrays)

    external_idx = None
    if cfg.external_test_mode is not None:
        keep_idx, external_idx = external_holdout_indices(
            arrays,
            mode=cfg.external_test_mode,
            fraction=cfg.external_test_fraction,
            seed=cfg.seed + 17,
        )
        arrays = {
            key: (
                value[keep_idx]
                if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(source_pose_id)
                else value
            )
            for key, value in arrays.items()
        }
        source_pose_id = require_source_pose_id(arrays)

    tr_idx, va_idx = _split_indices(
        arrays["boundary_id"],
        cfg.val_fraction,
        cfg.seed,
        source_pose_id,
    )
    input_center, input_scale = _resolve_input_normalization(arrays, tr_idx)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = SignedReachabilityField(
        width=cfg.width,
        depth=cfg.depth,
        fourier_bands=cfg.fourier_bands,
        softplus_beta=cfg.softplus_beta,
        input_center=input_center,
        input_scale=input_scale,
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
            eikonal_terms = []
            if len(nids):
                grad_all = torch.autograd.grad(pred[nids].sum(), xn, create_graph=True)[0]
                grad = grad_all[nids]
                normal_loss = (1.0 - F.cosine_similarity(grad, normal[nids], dim=-1, eps=1.0e-8)).mean()
                eikonal_terms.append(
                    unit_speed_eikonal_loss(grad, target_slope=slope[nids], beta=0.2)
                )
            else:
                normal_loss = pred.new_zeros(())
            # Declared-metric unit-speed prior on far/near SDF-supervised rows.
            sids = torch.nonzero(smask, as_tuple=False).flatten()
            if cfg.lambda_eikonal > 0.0 and len(sids):
                if len(sids) > cfg.normal_batch_max:
                    sids = sids[torch.randperm(len(sids), device=device)[: cfg.normal_batch_max]]
                grad_sdf = torch.autograd.grad(pred[sids].sum(), xn, create_graph=True)[0][sids]
                eikonal_terms.append(unit_speed_eikonal_loss(grad_sdf, target_slope=None, beta=0.2))
            eikonal = (
                torch.stack(eikonal_terms).mean()
                if eikonal_terms
                else pred.new_zeros(())
            )
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
        metrics = evaluate_signed_field(
            field,
            arrays,
            va_idx,
            report_near_axis=cfg.report_near_axis,
            near_axis_r_m=cfg.near_axis_r_m,
        )
        score = _selection(metrics)
        row = {"epoch": epoch, "loss": float(totals[0] / batches), "classification_loss": float(totals[1] / batches), "signed_loss": float(totals[2] / batches), "normal_loss": float(totals[3] / batches), "eikonal_loss": float(totals[4] / batches), "selection_score": score, **metrics}
        history.append(row)
        near_msg = ""
        if cfg.report_near_axis and "near_axis_r_lt_5cm_accuracy" in metrics:
            near_msg = f" near5cm={metrics['near_axis_r_lt_5cm_accuracy']:.3f}"
        print(
            f"epoch={epoch} loss={row['loss']:.4f} bal={metrics['balanced_accuracy']:.3f} "
            f"dir_m={metrics.get('direction_agreement_m',0):.3f} dir_deg={metrics.get('direction_agreement_deg',0):.3f} "
            f"wide_m={metrics.get('wide_straddle_rate_m',0):.3f} wide_deg={metrics.get('wide_straddle_rate_deg',0):.3f} "
            f"eik={row['eikonal_loss']:.4f} sel={score:.3f}{near_msg}",
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
    final_metrics = evaluate_signed_field(
        field,
        arrays,
        va_idx,
        report_near_axis=cfg.report_near_axis,
        near_axis_r_m=cfg.near_axis_r_m,
    )
    field.save(
        checkpoint,
        meta={
            "training_config": asdict(cfg),
            "dataset_meta": dataset_meta,
            "robot_contract": dataset_meta.get("robot_contract"),
            "metrics": final_metrics,
            "selection_score": best_score,
            "q1_aux_head": False,
            "input_center": input_center.tolist(),
            "input_scale": input_scale.tolist(),
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
        "n_external_holdout": int(len(external_idx)) if external_idx is not None else 0,
        "external_test_mode": cfg.external_test_mode,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    report = Path(cfg.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = [
    "EXTERNAL_SPLIT_MODES",
    "EXTERNAL_SPLIT_REQUIRED_KEYS",
    "SignedTrainConfig",
    "evaluate_signed_field",
    "external_holdout_indices",
    "load_signed_train_config",
    "require_source_pose_id",
    "train_signed_field",
]

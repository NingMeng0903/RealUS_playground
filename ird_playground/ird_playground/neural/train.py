"""Train Neural IRD v6: BCE(hard y) + masked SmoothL1(margin) + SmoothL1(q|pos).

Cycling (no-replace) difficulty batches, block-split val with fixed calib/test,
report IoU@0.5 and IoU@calibrated separately.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ird_playground.ird.export_gt import (
    LAYER_BND_NEG,
    LAYER_BND_POS,
    LAYER_EXTERIOR,
    LAYER_INTERIOR,
    LAYER_JITTER_NEG,
    LAYER_JITTER_POS,
    assert_gt_contract,
    load_ird_gt,
    make_synthetic_ird_gt,
)
from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class TrainConfig:
    gt_npz: str | None = None
    synthetic_n: int = 8192
    epochs: int = 100
    batch_size: int = 1024
    num_workers: int = 4
    torch_compile: bool = False
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.01
    grad_clip_norm: float = 10.0
    log_every_steps: int = 10
    print_every_steps: int = 50
    save_freq: int = 25
    val_frac: float = 0.15
    num_freqs: int = 6
    num_freqs_u: int = 5
    hidden: int = 256
    depth: int = 5
    tau_m: float = 1.0
    lambda_q_score: float = 0.5
    use_physical_pe: bool = True
    p_scale_m: float = 1.0
    seed: int = 42
    checkpoint: str = "data/checkpoints/latest.pt"
    checkpoint_dir: str = "data/checkpoints"
    report: str = "data/reports/train_point.json"
    device: str | None = None
    lambda_cls: float = 1.0
    lambda_margin: float = 0.0
    lambda_q: float = 0.0
    lambda_local: float = 0.0
    sigma_local_m: float = 0.06
    hardneg_every: int = 0
    hardneg_frac: float = 0.0
    # batch mix: interior / bnd+ / bnd- / jitter / exterior
    mix_interior: float = 0.15
    mix_bnd_pos: float = 0.25
    mix_bnd_neg: float = 0.25
    mix_jitter_pos: float = 0.10
    mix_jitter_neg: float = 0.10
    mix_exterior: float = 0.15
    # alias for old yaml key "jitter"
    mix_jitter: float = 0.0
    val_eval_n: int = 65536
    val_calib_frac: float = 0.5
    train_hard_y: bool = True  # Phase A: BCE on reachable, not y_soft
    mae_max: float = 0.35
    spearman_min: float = 0.70
    boundary_iou_min: float = 0.70
    grad_cosine_min: float = 0.30
    ascent_improve_min: float = 0.40
    rail_ad_fd_rel_max: float = 0.25
    rail_sign_agree_min: float = 0.80
    region_improve_min: float = 0.40
    wandb_enable: bool = False
    wandb_project: str = "neural-ird-rm75"
    wandb_entity: str = "lpei82060-technical-university-of-munich"
    wandb_mode: str = "online"
    wandb_run_name: str | None = None
    wandb_tags: list | None = None
    # Phase B warm-start
    init_checkpoint: str | None = None
    freeze_cls_epochs: int = 0  # freeze stem+blocks+head_cls for first N epochs


def _as_path(root: Path, p: str | None) -> str | None:
    if p is None or p == "null":
        return None
    path = Path(str(p))
    if not path.is_absolute():
        path = root / path
    return str(path)


def _normalize_device(raw) -> str | None:
    if raw in (None, "null", ""):
        return None
    s = str(raw).strip()
    return "cuda" if s.upper() == "CUDA" else s


def load_train_config(path: str | Path, *, root: Path | None = None) -> TrainConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = root or cfg_path.resolve().parents[1]
    data, model = dict(raw.get("data") or {}), dict(raw.get("model") or {})
    train = dict(raw.get("training") or raw.get("train") or {})
    loss, io = dict(raw.get("loss") or {}), dict(raw.get("io") or {})
    pas, wb = dict(raw.get("pass") or {}), dict(raw.get("wandb") or {})
    mix = dict(train.get("batch_mix") or raw.get("batch_mix") or {})

    gt = data.get("gt_npz")
    gt_path = None if gt in (None, "null", "") else _as_path(root, str(gt))
    if gt_path and not Path(gt_path).exists():
        gt_path = None

    tags = wb.get("tags")
    if tags is not None and not isinstance(tags, list):
        tags = [str(tags)]
    lr = train.get("learning_rate", train.get("lr", 3e-4))

    return TrainConfig(
        gt_npz=gt_path,
        synthetic_n=int(data.get("synthetic_n", 8192)),
        val_frac=float(data.get("val_frac", 0.15)),
        num_freqs=int(model.get("num_freqs", 6)),
        num_freqs_u=int(model.get("num_freqs_u", 5)),
        hidden=int(model.get("hidden", 256)),
        depth=int(model.get("depth", 5)),
        tau_m=float(model.get("tau_m", 1.0)),
        lambda_q_score=float(model.get("lambda_q", 0.5)),
        use_physical_pe=bool(model.get("use_physical_pe", True)),
        p_scale_m=float(model.get("p_scale_m", 1.0)),
        epochs=int(train.get("epochs", 100)),
        batch_size=int(train.get("batch_size", 1024)),
        num_workers=int(train.get("num_workers", 4)),
        torch_compile=bool(train.get("torch_compile", False)),
        lr=float(lr),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        warmup_steps=int(train.get("warmup_steps", 500)),
        min_lr_ratio=float(train.get("min_lr_ratio", 0.01)),
        grad_clip_norm=float(train.get("grad_clip_norm", 10.0)),
        log_every_steps=int(train.get("log_every_steps", 10)),
        print_every_steps=int(train.get("print_every_steps", 50)),
        save_freq=int(train.get("save_freq", 25)),
        hardneg_every=int(train.get("hardneg_every", 0)),
        hardneg_frac=float(train.get("hardneg_frac", 0.0)),
        seed=int(train.get("seed", 42)),
        device=_normalize_device(train.get("device")),
        mix_interior=float(mix.get("interior", 0.15)),
        mix_bnd_pos=float(mix.get("bnd_pos", 0.25)),
        mix_bnd_neg=float(mix.get("bnd_neg", 0.25)),
        mix_jitter_pos=float(mix.get("jitter_pos", mix.get("jitter", 0.20) / 2)),
        mix_jitter_neg=float(mix.get("jitter_neg", mix.get("jitter", 0.20) / 2)),
        mix_exterior=float(mix.get("exterior", 0.15)),
        val_eval_n=int(train.get("val_eval_n", 65536)),
        val_calib_frac=float(train.get("val_calib_frac", 0.5)),
        train_hard_y=bool(train.get("train_hard_y", True)),
        lambda_cls=float(loss.get("lambda_cls", 1.0)),
        lambda_margin=float(loss.get("lambda_margin", 0.0)),
        lambda_q=float(loss.get("lambda_q", 0.0)),
        lambda_local=float(loss.get("lambda_local", 0.0)),
        sigma_local_m=float(loss.get("sigma_local_m", 0.06)),
        checkpoint=str(_as_path(root, io.get("checkpoint", "data/checkpoints/latest.pt"))),
        checkpoint_dir=str(_as_path(root, io.get("checkpoint_dir", "data/checkpoints"))),
        report=str(_as_path(root, io.get("report", "data/reports/train_point.json"))),
        mae_max=float(pas.get("mae_max", 0.35)),
        spearman_min=float(pas.get("spearman_min", 0.70)),
        boundary_iou_min=float(pas.get("boundary_iou_min", 0.70)),
        grad_cosine_min=float(pas.get("grad_cosine_min", 0.30)),
        ascent_improve_min=float(pas.get("ascent_improve_min", 0.40)),
        rail_ad_fd_rel_max=float(pas.get("rail_ad_fd_rel_max", 0.25)),
        rail_sign_agree_min=float(pas.get("rail_sign_agree_min", 0.80)),
        region_improve_min=float(pas.get("region_improve_min", 0.40)),
        wandb_enable=bool(wb.get("enable", False)),
        wandb_project=str(wb.get("project", "neural-ird-rm75")),
        wandb_entity=str(wb.get("entity", "lpei82060-technical-university-of-munich")),
        wandb_mode=str(wb.get("mode", "online")),
        wandb_run_name=(None if wb.get("run_name") in (None, "null", "") else str(wb.get("run_name"))),
        wandb_tags=tags,
        init_checkpoint=_as_path(root, train.get("init_checkpoint")),
        freeze_cls_epochs=int(
            train.get("freeze_cls_epochs", train.get("freeze_trunk_epochs", 0))
        ),
    )


def _y_key(a):
    return "reachable" if "reachable" in a else "p_reach"


def _q_key(a):
    return "q" if "q" in a else "q_comfort"


def _m_key(a):
    return "m_gt" if "m_gt" in a else "d"


def _block_split(arrays, val_frac, seed):
    """Split by block_id so duplicate (spatial,orient) cannot leak train→val."""
    n = arrays["features"].shape[0]
    if "block_id" in arrays:
        blocks = arrays["block_id"]
        uniq = np.unique(blocks)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_val_b = max(1, int(len(uniq) * val_frac))
        val_blocks = set(uniq[:n_val_b].tolist())
        is_val = np.array([int(b) in val_blocks for b in blocks], dtype=bool)
        val_idx = np.flatnonzero(is_val)
        tr_idx = np.flatnonzero(~is_val)
        if tr_idx.size == 0 or val_idx.size == 0:
            # fallback random
            idx = rng.permutation(n)
            n_val = max(1, int(n * val_frac))
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
    else:
        idx = np.random.default_rng(seed).permutation(n)
        n_val = max(1, int(n * val_frac))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def take(ix):
        out = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out[k] = v[ix]
            else:
                out[k] = v
        return out

    return take(tr_idx), take(val_idx)


# keep alias used by older callers
_split = _block_split


class IRDTensorDataset(Dataset if torch is not None else object):  # type: ignore[misc]
    def __init__(self, arrays: dict, yk: str, mk: str, qk: str, *, hard_y: bool = True):
        self.x = torch.as_tensor(arrays["features"], dtype=torch.float32)
        # Phase A: hard classification on reachable; y_soft reserved for density head
        if hard_y:
            y_raw = arrays[yk]
        else:
            y_raw = arrays.get("y_soft", arrays[yk])
        self.y = torch.as_tensor(y_raw, dtype=torch.float32)
        self.m = torch.as_tensor(arrays[mk], dtype=torch.float32)
        self.q = torch.as_tensor(arrays[qk], dtype=torch.float32)
        mw = arrays.get("margin_weight")
        self.mw = torch.as_tensor(
            mw if mw is not None else np.ones(len(self.y), dtype=np.float32),
            dtype=torch.float32,
        )
        cw = arrays.get("cls_weight")
        self.cw = torch.as_tensor(
            cw if cw is not None else np.ones(len(self.y), dtype=np.float32),
            dtype=torch.float32,
        )
        layer = arrays.get("layer_id")
        self.layer = (
            torch.as_tensor(layer, dtype=torch.int64)
            if layer is not None
            else torch.zeros(len(self.y), dtype=torch.int64)
        )

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i], self.q[i], self.mw[i], self.cw[i], self.layer[i]


class CyclingLayerPool:
    """Without-replacement cycling within a layer (shuffle on wrap)."""

    def __init__(self, indices: np.ndarray, seed: int):
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.rng = np.random.default_rng(seed)
        self.pos = len(self.indices)  # force shuffle on first take

    def take(self, n: int) -> np.ndarray:
        if len(self.indices) == 0:
            return np.zeros(n, dtype=np.int64)
        result = []
        remain = int(n)
        while remain > 0:
            if self.pos >= len(self.indices):
                self.rng.shuffle(self.indices)
                self.pos = 0
            k = min(remain, len(self.indices) - self.pos)
            result.append(self.indices[self.pos : self.pos + k])
            self.pos += k
            remain -= k
        return np.concatenate(result)


class DifficultyBatchSampler(Sampler if torch is not None else object):  # type: ignore[misc]
    """Fixed mix: interior / bnd+ / bnd- / jitter_pos / jitter_neg / exterior."""

    def __init__(self, layer: np.ndarray, batch_size: int, mix: dict[int, float], *, seed: int = 0, steps: int | None = None):
        self.batch_size = int(batch_size)
        self.pools = {}
        for lid in (
            LAYER_INTERIOR,
            LAYER_BND_POS,
            LAYER_BND_NEG,
            LAYER_JITTER_POS,
            LAYER_JITTER_NEG,
            LAYER_EXTERIOR,
        ):
            idx = np.flatnonzero(layer == lid)
            self.pools[lid] = idx if idx.size else np.array([], dtype=np.int64)
        fallback = np.arange(len(layer), dtype=np.int64)
        for lid, idx in list(self.pools.items()):
            if idx.size == 0:
                self.pools[lid] = fallback
        weights = {
            LAYER_INTERIOR: mix.get(LAYER_INTERIOR, 0.15),
            LAYER_BND_POS: mix.get(LAYER_BND_POS, 0.25),
            LAYER_BND_NEG: mix.get(LAYER_BND_NEG, 0.25),
            LAYER_JITTER_POS: mix.get(LAYER_JITTER_POS, 0.10),
            LAYER_JITTER_NEG: mix.get(LAYER_JITTER_NEG, 0.10),
            LAYER_EXTERIOR: mix.get(LAYER_EXTERIOR, 0.15),
        }
        wsum = sum(weights.values()) or 1.0
        counts = {k: max(1, int(round(self.batch_size * v / wsum))) for k, v in weights.items()}
        while sum(counts.values()) > self.batch_size:
            k = max(counts, key=counts.get)
            counts[k] -= 1
        while sum(counts.values()) < self.batch_size:
            k = max(weights, key=weights.get)
            counts[k] += 1
        self.counts = counts
        self.layer_pools = {
            lid: CyclingLayerPool(idx, seed=seed + int(lid) * 97)
            for lid, idx in self.pools.items()
        }
        n = len(layer)
        self.steps = int(steps) if steps is not None else max(1, n // self.batch_size)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            for lid, c in self.counts.items():
                batch.append(self.layer_pools[lid].take(c))
            yield np.concatenate(batch).tolist()


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb_enable:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=cfg.wandb_run_name or "neural_ird_v6",
        tags=cfg.wandb_tags or ["neural_ird", "v6", "stable_support"],
        config={k: v for k, v in asdict(cfg).items() if not k.startswith("wandb_")},
    )


def _build_scheduler(opt, cfg: TrainConfig, steps_per_epoch: int):
    total_steps = max(1, int(cfg.epochs) * max(1, steps_per_epoch))
    warmup = max(0, int(cfg.warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        remain = max(1, total_steps - warmup)
        t = min(max(step - warmup, 0), remain) / remain
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), total_steps


def _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, mw, cw, cfg: TrainConfig):
    reach_logit = reach_logit.squeeze(-1)
    margin = margin.squeeze(-1)
    q = q.squeeze(-1)
    # unknown samples: cls_weight=0 → excluded from BCE
    if cw is not None and (cw > 0).any():
        L_cls = torch.nn.functional.binary_cross_entropy_with_logits(
            reach_logit, y, weight=cw, reduction="sum"
        ) / cw.sum().clamp_min(1.0)
    else:
        L_cls = torch.nn.functional.binary_cross_entropy_with_logits(reach_logit, y)
    mask = mw > 0
    if mask.any() and cfg.lambda_margin > 0:
        L_m = torch.nn.functional.smooth_l1_loss(margin[mask], m_gt[mask], beta=0.1)
    else:
        L_m = margin.new_zeros(())
    pos = y >= 0.5
    if pos.any() and cfg.lambda_q > 0:
        L_q = torch.nn.functional.smooth_l1_loss(q[pos], q_gt[pos], beta=0.1)
    else:
        L_q = margin.new_zeros(())
    loss = cfg.lambda_cls * L_cls + cfg.lambda_margin * L_m + cfg.lambda_q * L_q
    return loss, {
        "L_cls": float(L_cls.detach()),
        "L_m": float(L_m.detach()),
        "L_q": float(L_q.detach()),
        "L_local": 0.0,
    }


def _layer_metrics(y: np.ndarray, p: np.ndarray, layer: np.ndarray, *, threshold: float = 0.5) -> dict[str, float]:
    out = {}
    names = {
        LAYER_INTERIOR: "interior",
        LAYER_BND_POS: "bnd_pos",
        LAYER_BND_NEG: "bnd_neg",
        LAYER_JITTER_POS: "jitter_pos",
        LAYER_JITTER_NEG: "jitter_neg",
        LAYER_EXTERIOR: "exterior",
    }
    pred = p >= threshold
    gt = y >= 0.5
    for lid, name in names.items():
        m = layer == lid
        if not m.any():
            continue
        if lid in (LAYER_INTERIOR, LAYER_BND_POS, LAYER_JITTER_POS):
            pos = m & gt
            out[f"{name}_recall"] = float(pred[pos].mean()) if pos.any() else 0.0
        elif lid in (LAYER_BND_NEG, LAYER_EXTERIOR, LAYER_JITTER_NEG):
            neg = m & (~gt)
            out[f"{name}_spec"] = float((~pred[neg]).mean()) if neg.any() else 0.0
        else:
            out[f"{name}_acc"] = float((pred[m] == gt[m]).mean())
    inter = float(np.logical_and(gt, pred).sum())
    union = float(np.logical_or(gt, pred).sum()) + 1e-9
    out["iou"] = inter / union
    out["accuracy"] = float((pred == gt).mean())
    return out


def _pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    gt = y >= 0.5
    order = np.argsort(-p)
    y_s = gt[order].astype(np.float64)
    if y_s.sum() <= 0 or (~gt).sum() <= 0:
        return 0.0
    tp = np.cumsum(y_s)
    fp = np.cumsum(1.0 - y_s)
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / y_s.sum()
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


def _best_iou_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    gt = y >= 0.5
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_iou = 0.5, -1.0
    for t in thresholds:
        yp = p >= t
        inter = float(np.logical_and(gt, yp).sum())
        union = float(np.logical_or(gt, yp).sum()) + 1e-9
        iou = inter / union
        if iou > best_iou:
            best_iou, best_t = iou, float(t)
    return best_t, best_iou


def _make_fixed_eval_indices(arrays: dict, n_eval: int, seed: int) -> np.ndarray:
    n = arrays["features"].shape[0]
    rng = np.random.default_rng(seed)
    cw_all = arrays.get("cls_weight")
    supervised = np.flatnonzero(cw_all > 0) if cw_all is not None else np.arange(n)
    if supervised.size == 0:
        supervised = np.arange(n)
    if "layer_id" in arrays and supervised.size > n_eval:
        layer_all = arrays["layer_id"]
        picks = []
        per = max(1, n_eval // 6)
        for lid in (
            LAYER_INTERIOR,
            LAYER_BND_POS,
            LAYER_BND_NEG,
            LAYER_JITTER_POS,
            LAYER_JITTER_NEG,
            LAYER_EXTERIOR,
        ):
            idx = supervised[layer_all[supervised] == lid]
            if idx.size == 0:
                continue
            picks.append(rng.choice(idx, size=min(per, idx.size), replace=False))
        return np.concatenate(picks) if picks else rng.choice(supervised, size=min(n_eval, supervised.size), replace=False)
    return rng.choice(supervised, size=min(n_eval, supervised.size), replace=False)


def _split_val_blocks(val: dict, frac: float, seed: int) -> tuple[dict, dict]:
    """Split validation arrays into fixed calibration / test by block_id."""
    n = val["features"].shape[0]

    def take(ix):
        out = {}
        for k, v in val.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out[k] = v[ix]
            else:
                out[k] = v
        return out

    if "block_id" in val:
        blocks = val["block_id"]
        uniq = np.unique(blocks)
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        n_cal = max(1, int(len(uniq) * frac))
        cal_blocks = set(uniq[:n_cal].tolist())
        is_cal = np.array([int(b) in cal_blocks for b in blocks], dtype=bool)
        cal_idx, test_idx = np.flatnonzero(is_cal), np.flatnonzero(~is_cal)
        if cal_idx.size == 0 or test_idx.size == 0:
            idx = rng.permutation(n)
            n_cal_s = max(1, int(n * frac))
            cal_idx, test_idx = idx[:n_cal_s], idx[n_cal_s:]
    else:
        idx = np.random.default_rng(seed).permutation(n)
        n_cal_s = max(1, int(n * frac))
        cal_idx, test_idx = idx[:n_cal_s], idx[n_cal_s:]
    return take(cal_idx), take(test_idx)


def _eval_fixed(
    net,
    arrays: dict,
    idx: np.ndarray,
    *,
    threshold: float = 0.5,
    mk: str | None = None,
) -> dict[str, float]:
    yk = _y_key(arrays)
    mk = mk or _m_key(arrays)
    feats = arrays["features"][idx]
    pred = net.score_features_np(feats)
    y = arrays[yk][idx]
    layer = arrays["layer_id"][idx] if "layer_id" in arrays else np.zeros(len(idx), dtype=np.int32)
    metrics = _layer_metrics(y, pred["p_reach"], layer, threshold=threshold)
    metrics["pr_auc"] = _pr_auc(y, pred["p_reach"])
    metrics["threshold"] = float(threshold)
    mw = arrays["margin_weight"][idx] if "margin_weight" in arrays else np.ones(len(idx))
    mask = mw > 0
    if mask.any():
        metrics["boundary_margin_mae"] = float(
            np.mean(np.abs(pred["m"][mask] - arrays[mk][idx][mask]))
        )
    else:
        metrics["boundary_margin_mae"] = 0.0
    return metrics, y, pred["p_reach"]


def _eval_subset(net, arrays, cfg: TrainConfig, seed: int = 0) -> dict[str, float]:
    """Legacy single-split eval (kept for callers); prefer _eval_calib_test."""
    idx = _make_fixed_eval_indices(arrays, cfg.val_eval_n, seed)
    pred = net.score_features_np(arrays["features"][idx])
    y = arrays[_y_key(arrays)][idx]
    layer = arrays["layer_id"][idx] if "layer_id" in arrays else np.zeros(len(idx), dtype=np.int32)
    metrics = _layer_metrics(y, pred["p_reach"], layer, threshold=0.5)
    metrics["pr_auc"] = _pr_auc(y, pred["p_reach"])
    t_star, best_iou = _best_iou_threshold(y, pred["p_reach"])
    metrics["best_iou"] = best_iou
    metrics["best_threshold"] = t_star
    metrics["iou_t05"] = metrics["iou"]
    mk = _m_key(arrays)
    mw = arrays["margin_weight"][idx] if "margin_weight" in arrays else np.ones(len(idx))
    mask = mw > 0
    metrics["boundary_margin_mae"] = (
        float(np.mean(np.abs(pred["m"][mask] - arrays[mk][idx][mask]))) if mask.any() else 0.0
    )
    return metrics


def _eval_calib_test(
    net,
    val_calib: dict,
    val_test: dict,
    calib_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, float]:
    """Fixed calibration → threshold; fixed test → report IoU@0.5 and IoU@t*."""
    # Calib: choose threshold
    calib_pred = net.score_features_np(val_calib["features"][calib_idx])
    y_cal = val_calib[_y_key(val_calib)][calib_idx]
    t_star, calib_best = _best_iou_threshold(y_cal, calib_pred["p_reach"])

    # Test @ 0.5
    m05, y_te, p_te = _eval_fixed(net, val_test, test_idx, threshold=0.5)
    # Test @ calibrated
    mcal, _, _ = _eval_fixed(net, val_test, test_idx, threshold=t_star)

    out = {
        "iou_t05": float(m05["iou"]),
        "iou_calibrated": float(mcal["iou"]),
        "val_threshold": float(t_star),
        "calib_best_iou": float(calib_best),
        "pr_auc": float(m05["pr_auc"]),
        "accuracy": float(m05["accuracy"]),
        "boundary_margin_mae": float(m05["boundary_margin_mae"]),
        # layer metrics at calibrated threshold (more informative for boundary)
        **{k: v for k, v in mcal.items() if k.endswith("_recall") or k.endswith("_spec")},
        # keep aliases for checkpoint selection
        "best_iou": float(mcal["iou"]),
        "iou": float(m05["iou"]),
        "best_threshold": float(t_star),
    }
    return out


def train_point_field(cfg: TrainConfig) -> dict:
    if torch is None:
        raise ImportError("torch required")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    arrays = load_ird_gt(cfg.gt_npz) if cfg.gt_npz else make_synthetic_ird_gt(cfg.synthetic_n, seed=cfg.seed)
    if arrays["features"].shape[1] != 6:
        raise ValueError(f"expected 6-D features, got {arrays['features'].shape[1]} — regenerate GT")
    assert_gt_contract(arrays)

    yk, qk, mk = _y_key(arrays), _q_key(arrays), _m_key(arrays)
    train, val = _block_split(arrays, cfg.val_frac, cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wb_run = _maybe_init_wandb(cfg)

    aabb_lo = np.asarray(arrays["aabb_lo"], dtype=np.float32).reshape(3)
    aabb_hi = np.asarray(arrays["aabb_hi"], dtype=np.float32).reshape(3)

    tr_ds = IRDTensorDataset(train, yk, mk, qk, hard_y=cfg.train_hard_y)
    va_ds = IRDTensorDataset(val, yk, mk, qk, hard_y=cfg.train_hard_y)
    mix = {
        LAYER_INTERIOR: cfg.mix_interior,
        LAYER_BND_POS: cfg.mix_bnd_pos,
        LAYER_BND_NEG: cfg.mix_bnd_neg,
        LAYER_JITTER_POS: cfg.mix_jitter_pos,
        LAYER_JITTER_NEG: cfg.mix_jitter_neg,
        LAYER_EXTERIOR: cfg.mix_exterior,
    }
    layer_np = train["layer_id"] if "layer_id" in train else np.zeros(len(tr_ds), dtype=np.int32)
    steps_per_epoch = max(1, len(tr_ds) // cfg.batch_size)
    tr_sampler = DifficultyBatchSampler(layer_np, cfg.batch_size, mix, seed=cfg.seed, steps=steps_per_epoch)
    tr_loader = DataLoader(
        tr_ds,
        batch_sampler=tr_sampler,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    # Fixed calib / test indices for comparable epoch curves
    val_calib, val_test = _split_val_blocks(val, cfg.val_calib_frac, cfg.seed + 7)
    per_half = max(1, cfg.val_eval_n // 2)
    calib_idx = _make_fixed_eval_indices(val_calib, per_half, cfg.seed)
    test_idx = _make_fixed_eval_indices(val_test, per_half, cfg.seed + 1)
    print(
        f"[train] fixed val: calib_n={len(calib_idx)} test_n={len(test_idx)} "
        f"physical_pe={cfg.use_physical_pe} freqs_u={cfg.num_freqs_u}",
        flush=True,
    )

    model = NeuralIRDPoint(
        in_dim=6,
        num_freqs=cfg.num_freqs,
        num_freqs_u=cfg.num_freqs_u,
        hidden=cfg.hidden,
        depth=cfg.depth,
        tau_m=cfg.tau_m,
        lambda_q=cfg.lambda_q_score,
        use_physical_pe=cfg.use_physical_pe,
        p_scale_m=cfg.p_scale_m,
    ).to(device)
    model.set_aabb(aabb_lo, aabb_hi)
    if cfg.init_checkpoint:
        ck0 = Path(cfg.init_checkpoint)
        if not ck0.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {ck0}")
        blob = torch.load(ck0, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
        print(
            f"[train] warm-start from {ck0}  missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    if cfg.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    def _set_cls_trainable(trainable: bool) -> None:
        src = model._orig_mod if hasattr(model, "_orig_mod") else model
        for p in src.stem.parameters():
            p.requires_grad = trainable
        for blk in src.blocks:
            for p in blk.parameters():
                p.requires_grad = trainable
        for p in src.head_cls.parameters():
            p.requires_grad = trainable

    if cfg.freeze_cls_epochs > 0:
        _set_cls_trainable(False)
        print(f"[train] freeze stem+blocks+cls for {cfg.freeze_cls_epochs} epochs", flush=True)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler, total_steps = _build_scheduler(opt, cfg, steps_per_epoch)

    history = []
    best_iou, best_margin_mae = -1.0, float("inf")
    best_iou_state, best_margin_state = None, None
    global_step = 0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def model_cfg():
        src = model._orig_mod if hasattr(model, "_orig_mod") else model
        waves = src.p_wavelengths_m.detach().cpu().numpy().tolist()
        return {
            "in_dim": 6,
            "num_freqs": cfg.num_freqs,
            "num_freqs_u": cfg.num_freqs_u,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "tau_m": cfg.tau_m,
            "lambda_q": cfg.lambda_q_score,
            "use_physical_pe": cfg.use_physical_pe,
            "p_wavelengths_m": waves,
            "p_scale_m": cfg.p_scale_m,
            "aabb": {"lo": aabb_lo.tolist(), "hi": aabb_hi.tolist()},
            "feature_kind": "natural_pu",
        }

    def clone_state(m):
        src = m._orig_mod if hasattr(m, "_orig_mod") else m
        return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}

    def save(path: Path, state) -> None:
        clean = NeuralIRDPoint(
            in_dim=6,
            num_freqs=cfg.num_freqs,
            num_freqs_u=cfg.num_freqs_u,
            hidden=cfg.hidden,
            depth=cfg.depth,
            tau_m=cfg.tau_m,
            lambda_q=cfg.lambda_q_score,
            use_physical_pe=cfg.use_physical_pe,
            p_scale_m=cfg.p_scale_m,
        )
        clean.load_state_dict(state)
        clean.set_aabb(aabb_lo, aabb_hi)
        NeuralIRD(clean, device=str(device)).save(
            path,
            model_cfg=model_cfg(),
            meta={
                "best_iou": best_iou,
                "best_margin_mae": best_margin_mae,
                "global_step": global_step,
                "aabb_lo": aabb_lo,
                "aabb_hi": aabb_hi,
            },
        )

    try:
        for epoch in range(int(cfg.epochs)):
            if cfg.freeze_cls_epochs > 0 and epoch == int(cfg.freeze_cls_epochs):
                _set_cls_trainable(True)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                scheduler, _ = _build_scheduler(opt, cfg, steps_per_epoch)
                print(f"[train] unfreeze all @ epoch {epoch}", flush=True)
            model.train()
            tr_loss = n_tr = 0.0
            for x, y, m_gt, q_gt, mw, cw, _layer in tr_loader:
                x = x.to(device)
                y, m_gt, q_gt, mw, cw = (
                    y.to(device),
                    m_gt.to(device),
                    q_gt.to(device),
                    mw.to(device),
                    cw.to(device),
                )
                reach_logit, margin, q, _ = model(x)
                loss, parts = _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, mw, cw, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip_norm and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                opt.step()
                scheduler.step()
                global_step += 1
                tr_loss += float(loss.item()) * x.shape[0]
                n_tr += x.shape[0]
                if wb_run is not None and global_step % max(1, cfg.log_every_steps) == 0:
                    import wandb

                    wandb.log(
                        {
                            "train/loss_step": float(loss.item()),
                            "train/L_cls": parts["L_cls"],
                            "train/L_m": parts["L_m"],
                            "train/L_q": parts["L_q"],
                            "train/lr": float(opt.param_groups[0]["lr"]),
                            "step": global_step,
                        },
                        step=global_step,
                    )
                if global_step % max(1, cfg.print_every_steps) == 0:
                    print(
                        f"step={global_step}/{total_steps} epoch={epoch} "
                        f"loss={float(loss.item()):.4f} "
                        f"cls={parts['L_cls']:.3f} m={parts['L_m']:.3f} "
                        f"q={parts['L_q']:.3f} lr={opt.param_groups[0]['lr']:.2e}"
                    )

            model.eval()
            va_loss = n_va = 0.0
            with torch.no_grad():
                for x, y, m_gt, q_gt, mw, cw, _layer in va_loader:
                    x = x.to(device)
                    y, m_gt, q_gt, mw, cw = (
                        y.to(device),
                        m_gt.to(device),
                        q_gt.to(device),
                        mw.to(device),
                        cw.to(device),
                    )
                    reach_logit, margin, q, _ = model(x)
                    loss, _ = _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, mw, cw, cfg)
                    va_loss += float(loss.item()) * x.shape[0]
                    n_va += x.shape[0]

            wrapper = NeuralIRD(
                model._orig_mod if hasattr(model, "_orig_mod") else model, device=str(device)
            )
            val_m = _eval_calib_test(wrapper, val_calib, val_test, calib_idx, test_idx)
            # also fixed train subset for train/val gap diagnosis
            train_idx_fixed = _make_fixed_eval_indices(train, min(8192, cfg.val_eval_n // 4), cfg.seed + 99)
            train_m, _, _ = _eval_fixed(wrapper, train, train_idx_fixed, threshold=0.5)
            train_m["pr_auc"] = _pr_auc(
                train[_y_key(train)][train_idx_fixed],
                wrapper.score_features_np(train["features"][train_idx_fixed])["p_reach"],
            )
            val_iou = float(val_m.get("iou_calibrated", val_m.get("best_iou", val_m["iou"])))
            bmae = float(val_m.get("boundary_margin_mae", 0.0))

            row = {
                "epoch": epoch,
                "train_loss": tr_loss / max(n_tr, 1),
                "val_loss": va_loss / max(n_va, 1),
                "val_iou": val_iou,
                "boundary_margin_mae": bmae,
                "lr": float(opt.param_groups[0]["lr"]),
                "train_iou_t05": float(train_m["iou"]),
                "train_pr_auc": float(train_m["pr_auc"]),
                **{f"val_{k}": v for k, v in val_m.items()},
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} "
                f"iou@0.5={float(val_m.get('iou_t05', 0)):.3f} "
                f"iou@cal={val_iou:.3f}@t={float(val_m.get('val_threshold', 0.5)):.2f} "
                f"pr_auc={float(val_m.get('pr_auc', 0)):.3f} "
                f"train_iou={float(train_m['iou']):.3f} "
                f"bnd_pos_r={float(val_m.get('bnd_pos_recall', 0)):.3f} "
                f"bnd_neg_s={float(val_m.get('bnd_neg_spec', 0)):.3f} "
                f"lr={row['lr']:.2e}"
            )
            if wb_run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": row["train_loss"],
                        "val/loss": row["val_loss"],
                        "val/iou_t05": float(val_m.get("iou_t05", 0)),
                        "val/iou_calibrated": val_iou,
                        "val/threshold": float(val_m.get("val_threshold", 0.5)),
                        "val/pr_auc": float(val_m.get("pr_auc", 0)),
                        "train/iou_t05": float(train_m["iou"]),
                        "train/pr_auc": float(train_m["pr_auc"]),
                        "val/boundary_margin_mae": bmae,
                        **{f"val/{k}": v for k, v in val_m.items()},
                        "train/lr_epoch": row["lr"],
                    },
                    step=global_step,
                )

            current = clone_state(model)
            save(Path(cfg.checkpoint), current)
            save(ckpt_dir / "latest.pt", current)
            if val_iou > best_iou:
                best_iou = val_iou
                best_iou_state = current
                save(ckpt_dir / "best_iou.pt", current)
                save(ckpt_dir / "best.pt", current)
            if bmae < best_margin_mae and cfg.lambda_margin > 0:
                best_margin_mae = bmae
                best_margin_state = current
                save(ckpt_dir / "best_margin.pt", current)
            if cfg.save_freq > 0 and (epoch + 1) % cfg.save_freq == 0:
                save(ckpt_dir / f"epoch_{epoch+1:04d}.pt", current)

        final_state = best_iou_state or clone_state(model)
        clean = NeuralIRDPoint(
            in_dim=6,
            num_freqs=cfg.num_freqs,
            num_freqs_u=cfg.num_freqs_u,
            hidden=cfg.hidden,
            depth=cfg.depth,
            tau_m=cfg.tau_m,
            lambda_q=cfg.lambda_q_score,
            use_physical_pe=cfg.use_physical_pe,
            p_scale_m=cfg.p_scale_m,
        )
        clean.load_state_dict(final_state)
        clean.set_aabb(aabb_lo, aabb_hi)
        wrapper = NeuralIRD(clean, device=str(device))
        wrapper.save(
            cfg.checkpoint,
            model_cfg=model_cfg(),
            meta={
                "history_tail": history[-5:],
                "best_iou": best_iou,
                "best_margin_mae": best_margin_mae,
                "n_train": int(train["features"].shape[0]),
                "global_step": global_step,
                "aabb_lo": aabb_lo,
                "aabb_hi": aabb_hi,
            },
        )
        metrics = evaluate_point_field(wrapper, val)
        metrics.update(_eval_calib_test(wrapper, val_calib, val_test, calib_idx, test_idx))
        if wb_run is not None:
            import wandb

            wandb.log({f"val/{k}": v for k, v in metrics.items() if np.isscalar(v)}, step=global_step)
            wandb.save(cfg.checkpoint)
        return {"checkpoint": str(cfg.checkpoint), "history": history, "val_metrics": metrics}
    finally:
        if wb_run is not None:
            import wandb

            wandb.finish()


def evaluate_point_field(net: NeuralIRD, arrays: dict[str, np.ndarray]) -> dict[str, float]:
    pred = net.score_features_np(arrays["features"])
    yk, qk, mk = _y_key(arrays), _q_key(arrays), _m_key(arrays)
    m_gt = arrays[mk].astype(np.float64)
    m_pr = pred["m"].astype(np.float64)
    q_gt = arrays[qk].astype(np.float64)
    q_pr = pred["q"].astype(np.float64)
    y_gt = arrays[yk].astype(np.float64)
    p_pr = pred["p_reach"].astype(np.float64)

    mask = y_gt >= 0.5
    mw = arrays["margin_weight"].astype(np.float64) if "margin_weight" in arrays else np.ones_like(y_gt)
    mw_mask = mw > 0
    mae_m = float(np.mean(np.abs(m_pr[mw_mask] - m_gt[mw_mask]))) if mw_mask.any() else 0.0
    mae_q = float(np.mean(np.abs(q_pr[mask] - q_gt[mask]))) if mask.any() else 0.0
    from scipy.stats import spearmanr

    sp = spearmanr(q_gt[mask], q_pr[mask]) if mask.sum() > 5 else None
    gt_b, pr_b = y_gt >= 0.5, p_pr >= 0.5
    inter = float(np.logical_and(gt_b, pr_b).sum())
    union = float(np.logical_or(gt_b, pr_b).sum()) + 1e-9
    score_gt = arrays["d"].astype(np.float64) if "d" in arrays else y_gt * q_gt
    out = {
        "mae": float(np.mean(np.abs(pred["score"].astype(np.float64) - score_gt))),
        "mae_m": mae_m,
        "mae_q": mae_q,
        "spearman": float(sp.correlation) if sp is not None and sp.correlation is not None else 0.0,
        "boundary_iou": inter / union,
        "reach_accuracy": float((gt_b == pr_b).mean()),
        "n": int(y_gt.shape[0]),
    }
    if "layer_id" in arrays:
        out.update(_layer_metrics(y_gt, p_pr, arrays["layer_id"]))
    return out


def differentiability_smoke(net: NeuralIRD) -> float:
    if torch is None:
        raise ImportError("torch required")
    x = torch.zeros(1, 6, dtype=torch.float32, device=net.device)
    with torch.no_grad():
        x[0, 5] = 1.0  # tool axis +Z
        x[0, 0] = 0.2
    x = x.detach().requires_grad_(True)
    _, _, _, score = net.model(x)
    score.sum().backward()
    assert x.grad is not None
    return float(x.grad.norm().item())

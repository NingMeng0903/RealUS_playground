"""Train / eval the generic Neural IRD point field f_θ → (m, q).

Loss: λ_cls BCE(σ(m), y) + λ_m margin + λ_q y·L_q + λ_local local consistency.
No default Eikonal. Hard-neg mining every N epochs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ird_playground.ird.export_gt import load_ird_gt, make_synthetic_ird_gt
from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class TrainConfig:
    gt_npz: str | None = None
    synthetic_n: int = 8192
    epochs: int = 200
    batch_size: int = 1024
    num_workers: int = 4
    torch_compile: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.01
    grad_clip_norm: float = 10.0
    log_every_steps: int = 10
    print_every_steps: int = 50
    save_freq: int = 25
    val_frac: float = 0.15
    num_freqs: int = 6
    hidden: int = 256
    depth: int = 5
    tau_m: float = 1.0
    lambda_q_score: float = 0.5
    seed: int = 42
    checkpoint: str = "data/checkpoints/latest.pt"
    checkpoint_dir: str = "data/checkpoints"
    report: str = "data/reports/train_point.json"
    device: str | None = None
    # loss weights
    lambda_cls: float = 1.0
    lambda_margin: float = 0.5
    lambda_q: float = 1.0
    lambda_local: float = 0.1
    hardneg_every: int = 10
    hardneg_frac: float = 0.05
    # pass thresholds
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
    if s.upper() == "CUDA":
        return "cuda"
    return s


def load_train_config(path: str | Path, *, root: Path | None = None) -> TrainConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = root or cfg_path.resolve().parents[1]

    data = dict(raw.get("data") or {})
    model = dict(raw.get("model") or {})
    train = dict(raw.get("training") or raw.get("train") or {})
    loss = dict(raw.get("loss") or {})
    io = dict(raw.get("io") or {})
    pas = dict(raw.get("pass") or {})
    wb = dict(raw.get("wandb") or {})

    gt = data.get("gt_npz")
    if gt in (None, "null", ""):
        gt_path = None
    else:
        gt_path = _as_path(root, str(gt))
        if gt_path and not Path(gt_path).exists():
            gt_path = None

    tags = wb.get("tags")
    if tags is not None and not isinstance(tags, list):
        tags = [str(tags)]

    lr = train.get("learning_rate", train.get("lr", 1e-4))

    return TrainConfig(
        gt_npz=gt_path,
        synthetic_n=int(data.get("synthetic_n", 8192)),
        val_frac=float(data.get("val_frac", 0.15)),
        num_freqs=int(model.get("num_freqs", 6)),
        hidden=int(model.get("hidden", 256)),
        depth=int(model.get("depth", 5)),
        tau_m=float(model.get("tau_m", 1.0)),
        lambda_q_score=float(model.get("lambda_q", 0.5)),
        epochs=int(train.get("epochs", 200)),
        batch_size=int(train.get("batch_size", 1024)),
        num_workers=int(train.get("num_workers", 4)),
        torch_compile=bool(train.get("torch_compile", False)),
        lr=float(lr),
        weight_decay=float(train.get("weight_decay", 1e-4)),
        warmup_steps=int(train.get("warmup_steps", 1000)),
        min_lr_ratio=float(train.get("min_lr_ratio", 0.01)),
        grad_clip_norm=float(train.get("grad_clip_norm", 10.0)),
        log_every_steps=int(train.get("log_every_steps", 10)),
        print_every_steps=int(train.get("print_every_steps", 50)),
        save_freq=int(train.get("save_freq", 25)),
        hardneg_every=int(train.get("hardneg_every", 10)),
        hardneg_frac=float(train.get("hardneg_frac", 0.05)),
        seed=int(train.get("seed", 42)),
        device=_normalize_device(train.get("device")),
        lambda_cls=float(loss.get("lambda_cls", 1.0)),
        lambda_margin=float(loss.get("lambda_margin", 0.5)),
        lambda_q=float(loss.get("lambda_q", 1.0)),
        lambda_local=float(loss.get("lambda_local", 0.1)),
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
    )


def _y_key(arrays: dict[str, np.ndarray]) -> str:
    return "reachable" if "reachable" in arrays else "p_reach"


def _q_key(arrays: dict[str, np.ndarray]) -> str:
    return "q" if "q" in arrays else "q_comfort"


def _m_key(arrays: dict[str, np.ndarray]) -> str:
    return "m_gt" if "m_gt" in arrays else "d"


def _split(arrays: dict[str, np.ndarray], val_frac: float, seed: int):
    n = arrays["features"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _take(ix):
        out = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n:
                out[k] = v[ix]
            else:
                out[k] = v
        return out

    return _take(tr_idx), _take(val_idx)


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb_enable:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=cfg.wandb_run_name or "neural_ird_mq",
        tags=cfg.wandb_tags or ["neural_ird", "m_q"],
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


def _compute_loss(m, q, y, m_gt, q_gt, cfg: TrainConfig, *, local_pair=None):
    m = m.squeeze(-1)
    q = q.squeeze(-1)
    p = torch.sigmoid(m)
    L_cls = torch.nn.functional.binary_cross_entropy(p, y)
    L_m = torch.nn.functional.mse_loss(m, m_gt)
    # comfort only on reachable
    w = y
    L_q = ((q - q_gt) ** 2 * w).sum() / (w.sum() + 1e-6)
    L_local = m.new_tensor(0.0)
    if local_pair is not None:
        m2, m_gt2 = local_pair
        L_local = torch.nn.functional.mse_loss(m - m2.squeeze(-1), m_gt - m_gt2)
    loss = (
        cfg.lambda_cls * L_cls
        + cfg.lambda_margin * L_m
        + cfg.lambda_q * L_q
        + cfg.lambda_local * L_local
    )
    return loss, {
        "L_cls": float(L_cls.detach()),
        "L_m": float(L_m.detach()),
        "L_q": float(L_q.detach()),
        "L_local": float(L_local.detach()),
    }


def _mine_hard_negatives(model, features, y, device, frac: float) -> np.ndarray:
    """Return indices of high-confidence misclassifications."""
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        conf = []
        for i in range(0, x.shape[0], 4096):
            m, _, _ = model(x[i : i + 4096])
            conf.append(torch.sigmoid(m.squeeze(-1)).cpu().numpy())
        p = np.concatenate(conf, axis=0)
    y_np = y.astype(np.float64)
    err = np.abs(p - y_np)
    # high confidence wrong: large |p-y| and p near 0/1
    score = err * np.maximum(p, 1.0 - p)
    n = max(1, int(frac * features.shape[0]))
    return np.argsort(-score)[:n]


def train_point_field(cfg: TrainConfig) -> dict:
    if torch is None:
        raise ImportError("torch required for training")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.gt_npz:
        arrays = load_ird_gt(cfg.gt_npz)
    else:
        arrays = make_synthetic_ird_gt(cfg.synthetic_n, seed=cfg.seed)

    yk, qk, mk = _y_key(arrays), _q_key(arrays), _m_key(arrays)
    train, val = _split(arrays, cfg.val_frac, cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wb_run = _maybe_init_wandb(cfg)

    aabb_lo = np.asarray(arrays.get("aabb_lo", [-1, -1, -1]), dtype=np.float32).reshape(3)
    aabb_hi = np.asarray(arrays.get("aabb_hi", [1, 1, 1]), dtype=np.float32).reshape(3)

    def _loader(a, shuffle: bool, extra_idx: np.ndarray | None = None):
        feat = a["features"]
        y = a[yk]
        q = a[qk]
        m = a[mk]
        if extra_idx is not None and extra_idx.size:
            feat = np.concatenate([feat, a["features"][extra_idx]], axis=0)
            y = np.concatenate([y, a[yk][extra_idx]], axis=0)
            q = np.concatenate([q, a[qk][extra_idx]], axis=0)
            m = np.concatenate([m, a[mk][extra_idx]], axis=0)
        ds = TensorDataset(
            torch.as_tensor(feat, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
            torch.as_tensor(m, dtype=torch.float32),
            torch.as_tensor(q, dtype=torch.float32),
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=int(cfg.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    hard_idx: np.ndarray | None = None
    tr_loader = _loader(train, True)
    va_loader = _loader(val, False)
    steps_per_epoch = max(1, len(tr_loader))

    model = NeuralIRDPoint(
        in_dim=9,
        num_freqs=cfg.num_freqs,
        hidden=cfg.hidden,
        depth=cfg.depth,
        tau_m=cfg.tau_m,
        lambda_q=cfg.lambda_q_score,
    ).to(device)
    model.set_aabb(aabb_lo, aabb_hi)
    if cfg.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler, total_steps = _build_scheduler(opt, cfg, steps_per_epoch)

    history = []
    best_val = float("inf")
    best_state = None
    global_step = 0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _model_cfg() -> dict:
        return {
            "in_dim": 9,
            "num_freqs": cfg.num_freqs,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "tau_m": cfg.tau_m,
            "lambda_q": cfg.lambda_q_score,
            "aabb": {"lo": aabb_lo.tolist(), "hi": aabb_hi.tolist()},
        }

    def _save(path: Path, state) -> None:
        clean = NeuralIRDPoint(
            in_dim=9,
            num_freqs=cfg.num_freqs,
            hidden=cfg.hidden,
            depth=cfg.depth,
            tau_m=cfg.tau_m,
            lambda_q=cfg.lambda_q_score,
        )
        clean.load_state_dict(state)
        clean.set_aabb(aabb_lo, aabb_hi)
        NeuralIRD(clean, device=str(device)).save(
            path,
            model_cfg=_model_cfg(),
            meta={"best_val_loss": best_val, "global_step": global_step, "aabb_lo": aabb_lo, "aabb_hi": aabb_hi},
        )

    try:
        for epoch in range(int(cfg.epochs)):
            if (
                cfg.hardneg_every > 0
                and epoch > 0
                and epoch % cfg.hardneg_every == 0
            ):
                src = model._orig_mod if hasattr(model, "_orig_mod") else model
                hard_idx = _mine_hard_negatives(
                    src, train["features"], train[yk], device, cfg.hardneg_frac
                )
                tr_loader = _loader(train, True, extra_idx=hard_idx)
                steps_per_epoch = max(1, len(tr_loader))

            model.train()
            tr_loss = 0.0
            n_tr = 0
            for x, y, m_gt, q_gt in tr_loader:
                x = x.to(device)
                y = y.to(device)
                m_gt = m_gt.to(device)
                q_gt = q_gt.to(device)
                m, q, _ = model(x)
                # local consistency: pair with a random shuffle in-batch
                perm = torch.randperm(x.shape[0], device=device)
                m2, _, _ = model(x[perm])
                loss, parts = _compute_loss(
                    m, q, y, m_gt, q_gt, cfg, local_pair=(m2, m_gt[perm])
                )
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
                        f"loss={float(loss.item()):.4f} lr={opt.param_groups[0]['lr']:.2e}"
                    )

            model.eval()
            va_loss = 0.0
            n_va = 0
            with torch.no_grad():
                for x, y, m_gt, q_gt in va_loader:
                    x = x.to(device)
                    y = y.to(device)
                    m_gt = m_gt.to(device)
                    q_gt = q_gt.to(device)
                    m, q, _ = model(x)
                    loss, _ = _compute_loss(m, q, y, m_gt, q_gt, cfg)
                    va_loss += float(loss.item()) * x.shape[0]
                    n_va += x.shape[0]

            row = {
                "epoch": epoch,
                "train_loss": tr_loss / max(n_tr, 1),
                "val_loss": va_loss / max(n_va, 1),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} lr={row['lr']:.2e}"
            )
            if wb_run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": row["train_loss"],
                        "val/loss": row["val_loss"],
                        "train/lr_epoch": row["lr"],
                    },
                    step=global_step,
                )

            state_src = model._orig_mod if hasattr(model, "_orig_mod") else model
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}
                _save(Path(cfg.checkpoint), best_state)
                _save(ckpt_dir / "best.pt", best_state)

            if cfg.save_freq > 0 and (epoch + 1) % cfg.save_freq == 0 and best_state is not None:
                _save(ckpt_dir / f"epoch_{epoch+1:04d}.pt", best_state)
                _save(Path(cfg.checkpoint), best_state)

        if best_state is None:
            state_src = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}

        clean = NeuralIRDPoint(
            in_dim=9,
            num_freqs=cfg.num_freqs,
            hidden=cfg.hidden,
            depth=cfg.depth,
            tau_m=cfg.tau_m,
            lambda_q=cfg.lambda_q_score,
        )
        clean.load_state_dict(best_state)
        clean.set_aabb(aabb_lo, aabb_hi)
        wrapper = NeuralIRD(clean, device=str(device))
        wrapper.save(
            cfg.checkpoint,
            model_cfg=_model_cfg(),
            meta={
                "history_tail": history[-5:],
                "best_val_loss": best_val,
                "n_train": int(train["features"].shape[0]),
                "global_step": global_step,
                "aabb_lo": aabb_lo,
                "aabb_hi": aabb_hi,
            },
        )
        metrics = evaluate_point_field(wrapper, val)
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

    mae_m = float(np.mean(np.abs(m_pr - m_gt)))
    # q MAE on reachable only
    mask = y_gt >= 0.5
    mae_q = float(np.mean(np.abs(q_pr[mask] - q_gt[mask]))) if mask.any() else 0.0

    from scipy.stats import spearmanr

    sp_q = spearmanr(q_gt[mask], q_pr[mask]) if mask.sum() > 5 else None
    gt_b = y_gt >= 0.5
    pr_b = p_pr >= 0.5
    inter = float(np.logical_and(gt_b, pr_b).sum())
    union = float(np.logical_or(gt_b, pr_b).sum()) + 1e-9

    # legacy aliases for older dashboards
    score_gt = arrays["d"].astype(np.float64) if "d" in arrays else y_gt * q_gt
    mae = float(np.mean(np.abs(pred["score"].astype(np.float64) - score_gt)))

    return {
        "mae": mae,
        "mae_m": mae_m,
        "mae_q": mae_q,
        "spearman": float(sp_q.correlation) if sp_q is not None and sp_q.correlation is not None else 0.0,
        "boundary_iou": inter / union,
        "reach_accuracy": float((gt_b == pr_b).mean()),
        "n": int(y_gt.shape[0]),
    }


def differentiability_smoke(net: NeuralIRD) -> float:
    if torch is None:
        raise ImportError("torch required")
    x = torch.zeros(1, 9, dtype=torch.float32, device=net.device, requires_grad=True)
    with torch.no_grad():
        x[0, 3] = 1.0
        x[0, 7] = 1.0
    x = x.detach().requires_grad_(True)
    m, q, score = net.model(x)
    score.sum().backward()
    assert x.grad is not None
    return float(x.grad.norm().item())

"""Train Neural IRD: BCEWithLogits(reach) + SmoothL1(margin) + SmoothL1(q|pos).

No local loss / hard-neg in v1 (GT contract first).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ird_playground.ird.export_gt import assert_gt_contract, load_ird_gt, make_synthetic_ird_gt
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
    hidden: int = 256
    depth: int = 5
    tau_m: float = 1.0
    lambda_q_score: float = 0.5
    seed: int = 42
    checkpoint: str = "data/checkpoints/latest.pt"
    checkpoint_dir: str = "data/checkpoints"
    report: str = "data/reports/train_point.json"
    device: str | None = None
    lambda_cls: float = 1.0
    lambda_margin: float = 0.5
    lambda_q: float = 0.25
    lambda_local: float = 0.0
    sigma_local_m: float = 0.06
    hardneg_every: int = 0
    hardneg_frac: float = 0.0
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
    return "cuda" if s.upper() == "CUDA" else s


def load_train_config(path: str | Path, *, root: Path | None = None) -> TrainConfig:
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = root or cfg_path.resolve().parents[1]
    data, model = dict(raw.get("data") or {}), dict(raw.get("model") or {})
    train = dict(raw.get("training") or raw.get("train") or {})
    loss, io = dict(raw.get("loss") or {}), dict(raw.get("io") or {})
    pas, wb = dict(raw.get("pass") or {}), dict(raw.get("wandb") or {})

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
        hidden=int(model.get("hidden", 256)),
        depth=int(model.get("depth", 5)),
        tau_m=float(model.get("tau_m", 1.0)),
        lambda_q_score=float(model.get("lambda_q", 0.5)),
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
        lambda_cls=float(loss.get("lambda_cls", 1.0)),
        lambda_margin=float(loss.get("lambda_margin", 0.5)),
        lambda_q=float(loss.get("lambda_q", 0.25)),
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
    )


def _y_key(a):
    return "reachable" if "reachable" in a else "p_reach"


def _q_key(a):
    return "q" if "q" in a else "q_comfort"


def _m_key(a):
    return "m_gt" if "m_gt" in a else "d"


def _split(arrays, val_frac, seed):
    n = arrays["features"].shape[0]
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


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb_enable:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=cfg.wandb_run_name or "neural_ird_v3",
        tags=cfg.wandb_tags or ["neural_ird", "3head", "5dof"],
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


def _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, cfg: TrainConfig):
    reach_logit = reach_logit.squeeze(-1)
    margin = margin.squeeze(-1)
    q = q.squeeze(-1)
    L_cls = torch.nn.functional.binary_cross_entropy_with_logits(reach_logit, y)
    L_m = torch.nn.functional.smooth_l1_loss(margin, m_gt, beta=0.25)
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
    train, val = _split(arrays, cfg.val_frac, cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    wb_run = _maybe_init_wandb(cfg)

    aabb_lo = np.asarray(arrays["aabb_lo"], dtype=np.float32).reshape(3)
    aabb_hi = np.asarray(arrays["aabb_hi"], dtype=np.float32).reshape(3)

    def loader(a, shuffle: bool):
        ds = TensorDataset(
            torch.as_tensor(a["features"], dtype=torch.float32),
            torch.as_tensor(a[yk], dtype=torch.float32),
            torch.as_tensor(a[mk], dtype=torch.float32),
            torch.as_tensor(a[qk], dtype=torch.float32),
        )
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=int(cfg.num_workers),
            pin_memory=(device.type == "cuda"),
        )

    tr_loader, va_loader = loader(train, True), loader(val, False)
    steps_per_epoch = max(1, len(tr_loader))

    model = NeuralIRDPoint(
        in_dim=6,
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

    history, best_val, best_state, global_step = [], float("inf"), None, 0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def model_cfg():
        return {
            "in_dim": 6,
            "num_freqs": cfg.num_freqs,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "tau_m": cfg.tau_m,
            "lambda_q": cfg.lambda_q_score,
            "aabb": {"lo": aabb_lo.tolist(), "hi": aabb_hi.tolist()},
        }

    def save(path: Path, state) -> None:
        clean = NeuralIRDPoint(
            in_dim=6,
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
            model_cfg=model_cfg(),
            meta={"best_val_loss": best_val, "global_step": global_step, "aabb_lo": aabb_lo, "aabb_hi": aabb_hi},
        )

    try:
        for epoch in range(int(cfg.epochs)):
            model.train()
            tr_loss = n_tr = 0.0
            for x, y, m_gt, q_gt in tr_loader:
                x, y, m_gt, q_gt = x.to(device), y.to(device), m_gt.to(device), q_gt.to(device)
                reach_logit, margin, q, _ = model(x)
                loss, parts = _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, cfg)
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
                for x, y, m_gt, q_gt in va_loader:
                    x, y, m_gt, q_gt = x.to(device), y.to(device), m_gt.to(device), q_gt.to(device)
                    reach_logit, margin, q, _ = model(x)
                    loss, _ = _compute_loss(reach_logit, margin, q, y, m_gt, q_gt, cfg)
                    va_loss += float(loss.item()) * x.shape[0]
                    n_va += x.shape[0]

            # quick val IoU each epoch
            with torch.no_grad():
                pred = NeuralIRD(
                    model._orig_mod if hasattr(model, "_orig_mod") else model, device=str(device)
                ).score_features_np(val["features"][:8192])
                yv = val[yk][:8192]
                iou_inter = float(np.logical_and(yv >= 0.5, pred["p_reach"] >= 0.5).sum())
                iou_union = float(np.logical_or(yv >= 0.5, pred["p_reach"] >= 0.5).sum()) + 1e-9
                val_iou = iou_inter / iou_union

            row = {
                "epoch": epoch,
                "train_loss": tr_loss / max(n_tr, 1),
                "val_loss": va_loss / max(n_va, 1),
                "val_iou": val_iou,
                "lr": float(opt.param_groups[0]["lr"]),
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_iou={val_iou:.3f} lr={row['lr']:.2e}"
            )
            if wb_run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": row["train_loss"],
                        "val/loss": row["val_loss"],
                        "val/iou": val_iou,
                        "train/lr_epoch": row["lr"],
                    },
                    step=global_step,
                )

            state_src = model._orig_mod if hasattr(model, "_orig_mod") else model
            if row["val_loss"] < best_val:
                best_val = row["val_loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}
                save(Path(cfg.checkpoint), best_state)
                save(ckpt_dir / "best.pt", best_state)
            if cfg.save_freq > 0 and (epoch + 1) % cfg.save_freq == 0 and best_state is not None:
                save(ckpt_dir / f"epoch_{epoch+1:04d}.pt", best_state)

        if best_state is None:
            state_src = model._orig_mod if hasattr(model, "_orig_mod") else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_src.state_dict().items()}

        clean = NeuralIRDPoint(
            in_dim=6,
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
            model_cfg=model_cfg(),
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

    mask = y_gt >= 0.5
    mae_m = float(np.mean(np.abs(m_pr - m_gt)))
    mae_q = float(np.mean(np.abs(q_pr[mask] - q_gt[mask]))) if mask.any() else 0.0
    from scipy.stats import spearmanr

    sp = spearmanr(q_gt[mask], q_pr[mask]) if mask.sum() > 5 else None
    gt_b, pr_b = y_gt >= 0.5, p_pr >= 0.5
    inter = float(np.logical_and(gt_b, pr_b).sum())
    union = float(np.logical_or(gt_b, pr_b).sum()) + 1e-9
    score_gt = arrays["d"].astype(np.float64) if "d" in arrays else y_gt * q_gt
    return {
        "mae": float(np.mean(np.abs(pred["score"].astype(np.float64) - score_gt))),
        "mae_m": mae_m,
        "mae_q": mae_q,
        "spearman": float(sp.correlation) if sp is not None and sp.correlation is not None else 0.0,
        "boundary_iou": inter / union,
        "reach_accuracy": float((gt_b == pr_b).mean()),
        "n": int(y_gt.shape[0]),
    }


def differentiability_smoke(net: NeuralIRD) -> float:
    if torch is None:
        raise ImportError("torch required")
    x = torch.zeros(1, 6, dtype=torch.float32, device=net.device, requires_grad=True)
    with torch.no_grad():
        x[0, 3] = 0.0
        x[0, 4] = 0.0
        x[0, 5] = 1.0
    x = x.detach().requires_grad_(True)
    _, _, _, score = net.model(x)
    score.sum().backward()
    assert x.grad is not None
    return float(x.grad.norm().item())

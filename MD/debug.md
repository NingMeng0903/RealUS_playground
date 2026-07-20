# Neural IRD RM75 — Sprint 0 审查归档

Generated: 2026-07-20 20:15:52

Repository: `/media/camp/EXT_DRIVE/RealUS_playground`  
Package: `ird_playground/`

---

## 1. Executive summary

### 1.1 Architecture (定稿)

| 层 | 变量 | 说明 |
|---|---|---|
| **外层 NLP (P1)** | `(λ, r)` → Bernstein `c_λ, c_r` | **唯一**优化自由度 |
| **几何映射** | `T_tcp = G(λ)` | GT 血管/体表确定中心位姿；不是独立姿态优化 |
| **内层 Region A** | 局部椭球 + 工具轴圆锥 | 固定联合 Sobol K=32；**不是**优化变量 |
| **Point IRD** | `f(p,u)→(ℓ,m,q)` | 冻结的 5-DoF 点场；`cost` 用 ℓ+m+q |

计算图:

```
(λ, r)
  → T0 = G(λ)
  → 局部椭球 δp ∈ R_local · diag(a_t,a_b,a_n) · ε
  → 方向圆锥 u = Exp([δω]_×) u0   (area-uniform ρ,φ)
  → 共享 T_base(r)   # Sprint 0A 不扰动 rail
  → flatten [N·K, 6] → Point IRD
  → C = 0.3·mean + 0.7·soft-worst + w_cov·coverage
```

默认区域: `a≈(3,4,2) mm`, `β_max≈3°`, `P_min=0.9`.

### 1.2 v6 Point IRD 基线（已训）

| Phase | IoU@cal | PR-AUC | 备注 |
|---|---|---|---|
| A cls-only | ~0.844 | ~0.93 | `best_iou.pt` |
| B +margin+q | ~0.844 | ~0.926 | 现应 warm-start `best_iou.pt` |

重要: 0.8 mm margin MAE = 拟合体素中点合成边界，不等于物理 0.8 mm 精度。  
Region ±3 mm / ±3° 查询 = 网络插值梯度检查，不等于已 IK 验证的安全保证。

### 1.3 Sprint 0 已落地

- `optimization_cost(reach_logit, margin, q)` — 废弃把 margin 当地球可达势
- 全 tensor AD: `T_tcp(λ)`, `T_base(r)` → `cost`
- `local_region_cost`: 椭球+圆锥+联合 Sobol
- P1: `optimize_p1_lambda_rail`
- Phase B: `init_checkpoint` + `freeze_cls_epochs`
- 已删: 复杂 η 感知传播、`robust_region.py`、`manifold_region.py`

### 1.4 后续（未做）

- Sprint 0B: 共享全局配准偏置、亚毫米 rail 零点
- 连续 FK/IK GT + SE(3) 二分
- 任务区 2°–3° 角度加密；可选 Region Student

---

## 2. File index

| Path | Role |
|---|---|
| `neural/cost.py` | P1 用的 C(ℓ,m,q) |
| `neural/model.py` | Point IRD + PE |
| `neural/train.py` | Phase A/B + warm-start |
| `ird/query_base.py` | rail/λ AD 查询 |
| `ird/export_gt.py` | v6 GT |
| `region/local_region.py` | **Sprint 0A Region A** |
| `traj/manifold.py` | G(λ) 合成流形 |
| `traj/p1_optimize.py` | 二维 B-spline P1 |
| `configs/train_*.yaml` | 训练配置 |

---

## 3. Complete source


### `neural/cost.py`

```python
"""Trajectory-optimization cost from Neural IRD heads.

Do **not** use margin alone as a global reachability potential.
``reach_logit`` provides the global field; ``margin`` is a local safety
cushion near the supervised boundary; ``q`` is an interior comfort bonus.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


def optimization_cost(
    reach_logit: "torch.Tensor",
    margin: "torch.Tensor",
    quality: "torch.Tensor",
    *,
    logit_safe: float = 1.0,
    margin_safe: float = 0.20,
    tau_logit: float = 0.5,
    tau_margin: float = 0.1,
    w_cls: float = 1.0,
    w_margin: float = 0.5,
    w_q: float = 0.2,
) -> "torch.Tensor":
    """Scalar / batched IRD cost (lower is better).

    C = w_cls * softplus((ℓ_safe − ℓ)/τ_ℓ)
      + w_margin * softplus((m_safe − m)/τ_m)
      − w_q * q
    """
    if torch is None:
        raise ImportError("torch required")
    c_cls = F.softplus((float(logit_safe) - reach_logit) / max(float(tau_logit), 1e-6))
    c_margin = F.softplus((float(margin_safe) - margin) / max(float(tau_margin), 1e-6))
    return float(w_cls) * c_cls + float(w_margin) * c_margin - float(w_q) * quality


def legacy_margin_score(
    margin: "torch.Tensor",
    quality: "torch.Tensor",
    *,
    tau_m: float = 1.0,
    lambda_q: float = 0.5,
) -> "torch.Tensor":
    """Deprecated v6 score: −softplus(−m/τ) + λ_q q. Kept for checkpoint compat."""
    if torch is None:
        raise ImportError("torch required")
    return -F.softplus(-margin / max(float(tau_m), 1e-6)) + float(lambda_q) * quality
```

### `neural/model.py`

```python
"""Neural IRD v6: f_θ(p,u) → (reach_logit, margin, q).

Physical-wavelength Fourier PE on position (independent of AABB span);
Fourier PE on tool axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


# Physical wavelengths (meters): coarse workspace → single-voxel boundary
DEFAULT_P_WAVELENGTHS_M = (0.48, 0.24, 0.12, 0.06, 0.03, 0.015)


def positional_encoding(x: "torch.Tensor", num_freqs: int) -> "torch.Tensor":
    """Normalized-space Fourier (used for direction u)."""
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * np.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


def physical_position_encoding(
    p_m: "torch.Tensor",
    wavelengths_m: "torch.Tensor",
    *,
    p_scale_m: float = 1.0,
) -> "torch.Tensor":
    """Fourier features with fixed physical wavelengths (meters).

    Returns [p/p_scale, sin(2π p/λ), cos(2π p/λ)] for each λ.
    """
    p_raw = p_m / max(float(p_scale_m), 1e-6)
    phase = 2.0 * np.pi * p_m.unsqueeze(-1) / wavelengths_m
    return torch.cat(
        [p_raw, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)],
        dim=-1,
    )


# backward-compat alias
positional_encoding_xyz = positional_encoding


class ResidualSiLUBlock(nn.Module if nn is not None else object):  # type: ignore[misc]
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = F.silu(self.fc1(x))
        h = self.fc2(h)
        return F.silu(x + h)


class NeuralIRDPoint(nn.Module if nn is not None else object):  # type: ignore[misc]
    """6-D natural [p(3), u(3)] → reach_logit, margin, q.

    Position: physical-wavelength Fourier (default 48…1.5 cm).
    Direction: raw u + num_freqs_u Fourier bands.
    """

    def __init__(
        self,
        *,
        in_dim: int = 6,
        num_freqs: int = 6,
        num_freqs_u: int = 5,
        hidden: int = 256,
        depth: int = 5,
        tau_m: float = 1.0,
        lambda_q: float = 0.5,
        p_wavelengths_m: tuple[float, ...] | list[float] | None = None,
        p_scale_m: float = 1.0,
        use_physical_pe: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for NeuralIRDPoint")
        super().__init__()
        if in_dim != 6:
            raise ValueError("expected 6-D features (p + tool axis)")
        self.in_dim = 6
        self.num_freqs = int(num_freqs)
        self.num_freqs_u = int(num_freqs_u)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.tau_m = float(tau_m)
        self.lambda_q = float(lambda_q)
        self.p_scale_m = float(p_scale_m)
        self.use_physical_pe = bool(use_physical_pe)
        waves = tuple(p_wavelengths_m) if p_wavelengths_m is not None else DEFAULT_P_WAVELENGTHS_M
        self.register_buffer(
            "p_wavelengths_m",
            torch.tensor(waves, dtype=torch.float32),
        )
        n_wave = int(self.p_wavelengths_m.numel())
        if self.use_physical_pe:
            pe_p = 3 + 3 * 2 * n_wave  # p_raw + sin/cos per λ per axis
        else:
            pe_p = 3 + 3 * 2 * self.num_freqs
        pe_u = 3 + 3 * 2 * self.num_freqs_u
        self.stem = nn.Linear(pe_p + pe_u, hidden)
        self.blocks = nn.ModuleList([ResidualSiLUBlock(hidden) for _ in range(max(1, depth - 1))])
        self.head_cls = nn.Linear(hidden, 1)
        self.head_margin = nn.Linear(hidden, 1)
        self.head_q = nn.Linear(hidden, 1)
        self.register_buffer("aabb_lo", torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32))
        self.register_buffer("aabb_hi", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    def set_aabb(self, lo: np.ndarray | "torch.Tensor", hi: np.ndarray | "torch.Tensor") -> None:
        self.aabb_lo.copy_(torch.as_tensor(lo, dtype=torch.float32).reshape(3))
        self.aabb_hi.copy_(torch.as_tensor(hi, dtype=torch.float32).reshape(3))

    def encode(self, features: "torch.Tensor") -> "torch.Tensor":
        p = features[..., :3]
        u = features[..., 3:6]
        u = u / (u.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        if self.use_physical_pe:
            p_enc = physical_position_encoding(
                p, self.p_wavelengths_m, p_scale_m=self.p_scale_m
            )
        else:
            span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
            p_n = 2.0 * (p - self.aabb_lo) / span - 1.0
            p_enc = positional_encoding(p_n, self.num_freqs)
        u_enc = positional_encoding(u, self.num_freqs_u)
        return torch.cat([p_enc, u_enc], dim=-1)

    def forward(
        self, features: "torch.Tensor"
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        h = F.silu(self.stem(self.encode(features)))
        for block in self.blocks:
            h = block(h)
        reach_logit = self.head_cls(h)
        margin = self.head_margin(h)
        q = torch.sigmoid(self.head_q(h))
        # Legacy score kept for checkpoint / wandb compat — prefer optimization_cost.
        from ird_playground.neural.cost import legacy_margin_score

        score = legacy_margin_score(margin, q, tau_m=self.tau_m, lambda_q=self.lambda_q)
        return reach_logit, margin, q, score

    def score_features(self, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        from ird_playground.neural.cost import optimization_cost

        reach_logit, margin, q, score = self.forward(features)
        p_reach = torch.sigmoid(reach_logit)
        cost = optimization_cost(reach_logit, margin, q)
        return {
            "reach_logit": reach_logit,
            "m": margin,
            "margin": margin,
            "q": q,
            "q_comfort": q,
            "score": score,  # deprecated: margin-only
            "cost": cost,  # preferred for P1 trajectory optimization
            "p_reach": p_reach,
            "d": score,
        }


@dataclass
class PointScore:
    m: float
    q: float
    score: float
    p_reach: float = 0.0
    q_comfort: float = 0.0
    d: float = 0.0
    reach_logit: float = 0.0


class NeuralIRD:
    def __init__(self, model: NeuralIRDPoint, device: str | None = None) -> None:
        if torch is None:
            raise ImportError("torch is required")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def load(cls, checkpoint: str | Path, device: str | None = None) -> "NeuralIRD":
        ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = dict(ckpt.get("model_cfg", {}))
        model = NeuralIRDPoint(
            in_dim=int(cfg.get("in_dim", 6)),
            num_freqs=int(cfg.get("num_freqs", 6)),
            num_freqs_u=int(cfg.get("num_freqs_u", 5)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 5)),
            tau_m=float(cfg.get("tau_m", 1.0)),
            lambda_q=float(cfg.get("lambda_q", 0.5)),
            p_wavelengths_m=cfg.get("p_wavelengths_m"),
            p_scale_m=float(cfg.get("p_scale_m", 1.0)),
            use_physical_pe=bool(cfg.get("use_physical_pe", True)),
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        aabb = cfg.get("aabb")
        if aabb is not None:
            model.set_aabb(np.asarray(aabb["lo"]), np.asarray(aabb["hi"]))
        meta = ckpt.get("meta") or {}
        if "aabb_lo" in meta and "aabb_hi" in meta:
            model.set_aabb(meta["aabb_lo"], meta["aabb_hi"])
        return cls(model, device=device)

    def save(self, path: str | Path, *, model_cfg: dict | None = None, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        waves = self.model.p_wavelengths_m.detach().cpu().numpy().tolist()
        cfg = model_cfg or {
            "in_dim": 6,
            "num_freqs": self.model.num_freqs,
            "num_freqs_u": self.model.num_freqs_u,
            "hidden": self.model.hidden,
            "depth": self.model.depth,
            "tau_m": self.model.tau_m,
            "lambda_q": self.model.lambda_q,
            "use_physical_pe": self.model.use_physical_pe,
            "p_wavelengths_m": waves,
            "p_scale_m": self.model.p_scale_m,
            "aabb": {
                "lo": self.model.aabb_lo.detach().cpu().numpy().tolist(),
                "hi": self.model.aabb_hi.detach().cpu().numpy().tolist(),
            },
        }
        torch.save({"state_dict": self.model.state_dict(), "model_cfg": cfg, "meta": meta or {}}, path)

    @torch.no_grad()
    def score_features_np(self, features: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x[None, :]
        out = self.model.score_features(x)
        return {k: v.detach().cpu().numpy().reshape(-1) for k, v in out.items()}

    def score(self, delta_T: np.ndarray) -> PointScore:
        from ird_playground.probe.se3 import features_from_delta_T

        feat = features_from_delta_T(delta_T)
        out = self.score_features_np(feat)
        return PointScore(
            m=float(out["m"][0]),
            q=float(out["q"][0]),
            score=float(out["score"][0]),
            p_reach=float(out["p_reach"][0]),
            q_comfort=float(out["q"][0]),
            d=float(out["score"][0]),
            reach_logit=float(out["reach_logit"][0]),
        )

    def score_batch_delta_T(self, delta_Ts: np.ndarray) -> dict[str, np.ndarray]:
        from ird_playground.probe.se3 import batch_features_from_delta_T

        return self.score_features_np(batch_features_from_delta_T(delta_Ts))

    def region_score(self, **kwargs):
        from ird_playground.region.aggregate import region_score_a

        return region_score_a(self, **kwargs)
```

### `neural/train.py`

```python
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
```

### `ird/query_base.py`

```python
"""Query-time base pose from rail_y + full SE(3) composition (torch AD).

Optimization variables are (λ, r). T_tcp(λ) must stay a Tensor — never
np.asarray — so ∂C/∂λ and ∂C/∂r both survive.
"""

from __future__ import annotations

import numpy as np

from ird_playground.neural.cost import legacy_margin_score, optimization_cost
from ird_playground.probe.se3 import features_from_delta_T, invert_T, se3_mul

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def trans_y(r: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[1, 3] = float(r)
    return T


def T_base_from_rail_y(
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    Twr = np.eye(4, dtype=np.float64) if T_world_rail is None else np.asarray(T_world_rail, dtype=np.float64)
    Trb = np.eye(4, dtype=np.float64) if T_rail_base0 is None else np.asarray(T_rail_base0, dtype=np.float64)
    return se3_mul(se3_mul(Twr, trans_y(rail_y)), Trb)


def delta_T_from_tcp_and_rail(
    T_tcp: np.ndarray,
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    T_base = T_base_from_rail_y(
        rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    return invert_T(np.asarray(T_tcp, dtype=np.float64)) @ T_base


def score_vs_rail_y(
    neural_ird,
    T_tcp: np.ndarray,
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> dict[str, float]:
    dT = delta_T_from_tcp_and_rail(
        T_tcp, rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    ps = neural_ird.score(dT)
    return {
        "m": ps.m,
        "q": ps.q,
        "score": ps.score,
        "reach_logit": ps.reach_logit,
        "p_reach": ps.p_reach,
    }


def invert_T_torch(T: "torch.Tensor") -> "torch.Tensor":
    """Batch-capable inverse: (...,4,4) → (...,4,4)."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    Ti = torch.zeros_like(T)
    # identity on last  row/col then fill
    eye = torch.eye(4, dtype=T.dtype, device=T.device)
    if T.ndim == 2:
        Ti = eye.clone()
        Ti[:3, :3] = Rt
        Ti[:3, 3] = -(Rt @ t)
        return Ti
    Ti = eye.expand(T.shape).clone()
    Ti[..., :3, :3] = Rt
    Ti[..., :3, 3] = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return Ti


def features_from_delta_T_torch(dT: "torch.Tensor") -> "torch.Tensor":
    """(…,4,4) → (…,6) natural [p,u]."""
    R_delta = dT[..., :3, :3]
    t_delta = dT[..., :3, 3]
    R_base_tcp = R_delta.transpose(-1, -2)
    p = -(R_base_tcp @ t_delta.unsqueeze(-1)).squeeze(-1)
    u = R_base_tcp[..., :, 2]
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.cat([p, u], dim=-1)


def T_base_from_rail_y_torch(
    rail_y: "torch.Tensor",
    *,
    T_world_rail: "torch.Tensor",
    T_rail_base0: "torch.Tensor",
) -> "torch.Tensor":
    """rail_y: () or (N,) → T_base (4,4) or (N,4,4)."""
    device, dtype = rail_y.device, rail_y.dtype
    eye = torch.eye(4, dtype=dtype, device=device)
    if rail_y.ndim == 0:
        Ty = eye.clone()
        Ty[1, 3] = rail_y
        return T_world_rail @ Ty @ T_rail_base0
    n = int(rail_y.shape[0])
    Ty = eye.expand(n, 4, 4).clone()
    Ty[:, 1, 3] = rail_y
    return T_world_rail @ Ty @ T_rail_base0


def cost_from_tcp_and_rail_torch(
    neural_ird,
    T_tcp: "torch.Tensor",
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | "torch.Tensor" | None = None,
    T_rail_base0: np.ndarray | "torch.Tensor" | None = None,
    use_optimization_cost: bool = True,
    cost_kwargs: dict | None = None,
) -> dict[str, "torch.Tensor"]:
    """Full AD w.r.t. T_tcp and rail_y.

    Returns reach_logit, m, q, cost (and legacy score).
    """
    if torch is None:
        raise ImportError("torch required")
    device = neural_ird.device
    dtype = torch.float32
    T_tcp = T_tcp.to(device=device, dtype=dtype)
    rail_y = rail_y.to(device=device, dtype=dtype)

    if T_world_rail is None:
        Twr_t = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_world_rail):
        Twr_t = T_world_rail.to(device=device, dtype=dtype)
    else:
        Twr_t = torch.as_tensor(np.asarray(T_world_rail), dtype=dtype, device=device)

    if T_rail_base0 is None:
        Trb_t = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_rail_base0):
        Trb_t = T_rail_base0.to(device=device, dtype=dtype)
    else:
        Trb_t = torch.as_tensor(np.asarray(T_rail_base0), dtype=dtype, device=device)

    T_base = T_base_from_rail_y_torch(rail_y, T_world_rail=Twr_t, T_rail_base0=Trb_t)
    # broadcast T_tcp if scalar rail / batch mismatch
    if T_tcp.ndim == 2 and T_base.ndim == 3:
        T_tcp = T_tcp.expand(T_base.shape[0], 4, 4)
    elif T_tcp.ndim == 3 and T_base.ndim == 2:
        T_base = T_base.expand(T_tcp.shape[0], 4, 4)

    dT = invert_T_torch(T_tcp) @ T_base
    feat = features_from_delta_T_torch(dT)
    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    out = neural_ird.model.score_features(feat)
    logit = out["reach_logit"].squeeze(-1)
    m = out["m"].squeeze(-1)
    q = out["q"].squeeze(-1)
    legacy = legacy_margin_score(m, q, tau_m=neural_ird.model.tau_m, lambda_q=neural_ird.model.lambda_q)
    if use_optimization_cost:
        cost = optimization_cost(logit, m, q, **(cost_kwargs or {}))
    else:
        cost = -legacy

    def _sq(x):
        return x.squeeze(0) if squeeze and x.ndim > 0 and x.shape[0] == 1 else x

    return {
        "reach_logit": _sq(logit),
        "m": _sq(m),
        "q": _sq(q),
        "cost": _sq(cost),
        "score": _sq(legacy),  # deprecated margin-only score
        "p_reach": _sq(torch.sigmoid(logit)),
        "features": feat if not squeeze else feat[0],
        "delta_T": dT if not (squeeze and dT.ndim == 3) else dT[0],
    }


def score_vs_rail_y_torch(
    neural_ird,
    T_tcp,
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> "torch.Tensor":
    """Backward-compat: returns legacy score. Prefer cost_from_tcp_and_rail_torch.

    If ``T_tcp`` is a numpy array it is converted once (∂/∂T_tcp disabled);
    pass a Tensor to keep the full graph.
    """
    if torch is None:
        raise ImportError("torch required")
    if not torch.is_tensor(T_tcp):
        T_tcp = torch.as_tensor(np.asarray(T_tcp, dtype=np.float64), dtype=torch.float32, device=neural_ird.device)
    out = cost_from_tcp_and_rail_torch(
        neural_ird,
        T_tcp,
        rail_y,
        T_world_rail=T_world_rail,
        T_rail_base0=T_rail_base0,
        use_optimization_cost=False,
    )
    return out["score"]


def cost_vs_lambda_rail_torch(
    neural_ird,
    manifold,
    lam: "torch.Tensor",
    rail: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
    cost_kwargs: dict | None = None,
) -> dict[str, "torch.Tensor"]:
    """C(λ, r) with T_tcp = G(λ) from the vessel/skin manifold (full AD)."""
    if torch is None:
        raise ImportError("torch required")
    if hasattr(manifold, "sample_torch"):
        T_tcp = manifold.sample_torch(lam, dtype=torch.float32, device=neural_ird.device)
    else:
        raise TypeError("manifold must implement sample_torch for AD w.r.t. λ")
    return cost_from_tcp_and_rail_torch(
        neural_ird,
        T_tcp,
        rail,
        T_world_rail=T_world_rail,
        T_rail_base0=T_rail_base0,
        use_optimization_cost=True,
        cost_kwargs=cost_kwargs,
    )


def rail_y_grad_ad_fd(
    neural_ird,
    *,
    n: int = 32,
    rail_y: float = 0.0,
    eps: float = 1e-3,
    seed: int = 0,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> dict[str, float]:
    """Compare AD ∂cost/∂rail_y to central finite differences."""
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.probe.se3 import complete_frame_from_tool_axis, mat4_from_Rt

    rng = np.random.default_rng(seed)
    rels = []
    signs = []
    neural_ird.model.eval()
    for _ in range(n):
        p = rng.uniform(-0.5, 0.5, size=3)
        u = rng.normal(size=3)
        u = u / (np.linalg.norm(u) + 1e-12)
        T_tcp_np = mat4_from_Rt(complete_frame_from_tool_axis(u), p)
        T_tcp = torch.as_tensor(T_tcp_np, dtype=torch.float32, device=neural_ird.device)

        r = torch.tensor(float(rail_y), dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        out = cost_from_tcp_and_rail_torch(
            neural_ird, T_tcp, r, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )
        out["cost"].backward()
        g_ad = float(r.grad.item())

        with torch.no_grad():
            sp = cost_from_tcp_and_rail_torch(
                neural_ird,
                T_tcp,
                torch.tensor(rail_y + eps, dtype=torch.float32, device=neural_ird.device),
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["cost"].item()
            sm = cost_from_tcp_and_rail_torch(
                neural_ird,
                T_tcp,
                torch.tensor(rail_y - eps, dtype=torch.float32, device=neural_ird.device),
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["cost"].item()
        g_fd = (sp - sm) / (2.0 * eps)
        denom = max(abs(g_fd), abs(g_ad), 1e-6)
        rels.append(abs(g_ad - g_fd) / denom)
        signs.append(1.0 if np.sign(g_ad) == np.sign(g_fd) or abs(g_fd) < 1e-8 else 0.0)

    return {
        "rail_ad_fd_rel": float(np.median(rels)),
        "rail_sign_agree": float(np.mean(signs)),
        "rail_n": float(n),
    }


def lambda_rail_grad_ad_fd(
    neural_ird,
    manifold,
    *,
    n: int = 16,
    lam0: float = 0.15,
    rail0: float = 0.0,
    eps_lam: float = 1e-3,
    eps_rail: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """AD vs FD for ∂C/∂λ and ∂C/∂r on the 2D decision manifold."""
    if torch is None:
        raise ImportError("torch required")
    rng = np.random.default_rng(seed)
    rel_l, rel_r, sign_l, sign_r = [], [], [], []
    neural_ird.model.eval()
    for _ in range(n):
        lam_v = float(lam0 + rng.uniform(-0.05, 0.05))
        rail_v = float(rail0 + rng.uniform(-0.05, 0.05))
        lam = torch.tensor(lam_v, dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        rail = torch.tensor(rail_v, dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        c = cost_vs_lambda_rail_torch(neural_ird, manifold, lam, rail)["cost"]
        c.backward()
        g_lam_ad = float(lam.grad.item())
        g_rail_ad = float(rail.grad.item())

        with torch.no_grad():
            def _c(lv, rv):
                return float(
                    cost_vs_lambda_rail_torch(
                        neural_ird,
                        manifold,
                        torch.tensor(lv, dtype=torch.float32, device=neural_ird.device),
                        torch.tensor(rv, dtype=torch.float32, device=neural_ird.device),
                    )["cost"].item()
                )

            g_lam_fd = (_c(lam_v + eps_lam, rail_v) - _c(lam_v - eps_lam, rail_v)) / (2 * eps_lam)
            g_rail_fd = (_c(lam_v, rail_v + eps_rail) - _c(lam_v, rail_v - eps_rail)) / (2 * eps_rail)

        for g_ad, g_fd, rels, signs in (
            (g_lam_ad, g_lam_fd, rel_l, sign_l),
            (g_rail_ad, g_rail_fd, rel_r, sign_r),
        ):
            denom = max(abs(g_fd), abs(g_ad), 1e-6)
            rels.append(abs(g_ad - g_fd) / denom)
            signs.append(1.0 if np.sign(g_ad) == np.sign(g_fd) or abs(g_fd) < 1e-8 else 0.0)

    return {
        "lambda_ad_fd_rel": float(np.median(rel_l)),
        "rail_ad_fd_rel": float(np.median(rel_r)),
        "lambda_sign_agree": float(np.mean(sign_l)),
        "rail_sign_agree": float(np.mean(sign_r)),
        "n": float(n),
    }
```

### `ird/export_gt.py`

```python
"""Build IRD GT v6 — stable-support boundary; MC-hit ≠ unreachable.

Contract:
  features = [p_base,tcp(3), u_base(3)]  natural 5-DoF
  exact MC hit → positive
  trusted face pair: C+ >= min_positive_support AND C- == 0
    (non-overlapping half-neighborhoods; never soft_neg <= tau)
  near-miss / unstable faces → not exported (unknown)
  margin: continuous face-pair interpolation only on trusted faces
  jitter: face-normal ±delta from same trusted faces (pos/neg half)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

LAYER_INTERIOR = 0
LAYER_BND_POS = 1
LAYER_BND_NEG = 2
LAYER_JITTER_POS = 3
LAYER_JITTER_NEG = 4
LAYER_EXTERIOR = 5

# backward-compat aliases
LAYER_JITTER = LAYER_JITTER_POS


@dataclass
class IrdGtConfig:
    n_interior: int = 300_000
    n_boundary: int = 800_000
    n_exterior: int = 400_000
    n_positive: int = 700_000
    n_negative: int = 500_000
    seed: int = 0
    comfort_from: str = "auto"
    bbox_margin_m: float = 0.20
    max_orients_per_voxel: int = 28
    hard_negative_frac: float = 0.50
    hard_negative_radius_m: float = 0.06
    sigma_p_m: float = 0.03
    sigma_r_deg: float = 10.0
    m_clip: float = 3.0
    m_eps: float = 0.05
    w_manip: float = 0.5
    w_d: float = 0.5
    k_candidates: int = 4
    n_dof: int = 7
    aabb_pad_frac: float = 0.05
    aabb_pad_min_m: float = 0.02
    n_jitter: int = 400_000
    # soft / exterior thresholds (NOT used for boundary trust)
    orient_knn: int = 7
    soft_tau: float = 0.05
    unknown_soft_max: float = 0.25
    trusted_neg_soft_max: float = 1e-6
    # v6: stable-support boundary (C+ / C-)
    min_positive_support: int = 3
    min_trusted_face_pairs: int = 5000


def features_from_p_u(p: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(N,6): natural 5-DoF — TCP position in base + tool axis in base."""
    p = np.asarray(p, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
        u = u[None, :]
    u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    out = np.concatenate([p, u], axis=1).astype(np.float32)
    return out[0] if single else out


def _enforce_sign(m: np.ndarray, y: np.ndarray, eps: float, m_clip: float) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64)
    m = np.clip(m, -m_clip, m_clip)
    m = np.where((y >= 0.5) & (m <= 0.0), eps, m)
    m = np.where((y < 0.5) & (m >= 0.0), -eps, m)
    return np.clip(m, -m_clip, m_clip).astype(np.float32)


def _orient_knn(orients: np.ndarray, k: int) -> np.ndarray:
    dots = orients @ orients.T
    return np.argsort(-dots, axis=1)[:, :k].astype(np.int32)


def _tangents_for_dlt(dlt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit tangent axes orthogonal to face normal ``dlt`` (int voxel step)."""
    n = np.asarray(dlt, dtype=np.int32).reshape(3)
    if abs(int(n[0])) == 1:
        t1, t2 = np.array([0, 1, 0], np.int32), np.array([0, 0, 1], np.int32)
    elif abs(int(n[1])) == 1:
        t1, t2 = np.array([1, 0, 0], np.int32), np.array([0, 0, 1], np.int32)
    else:
        t1, t2 = np.array([1, 0, 0], np.int32), np.array([0, 1, 0], np.int32)
    return t1, t2


def _half_neighborhood(dlt: np.ndarray, *, positive_side: bool) -> np.ndarray:
    """Non-overlapping half-neighborhood for support counting.

    Positive side: current + interior (-dlt) + 4 tangents.
    Negative side: current + exterior (+dlt) + 4 tangents.
    """
    t1, t2 = _tangents_for_dlt(dlt)
    interior = -dlt if positive_side else dlt
    return np.stack(
        [
            np.array([0, 0, 0], np.int32),
            interior.astype(np.int32),
            t1,
            -t1,
            t2,
            -t2,
        ],
        axis=0,
    )


def export_ird_gt_from_capability_map(
    cm,
    cfg: IrdGtConfig | None = None,
    *,
    batch_size: int = 65536,
) -> dict[str, np.ndarray]:
    cfg = cfg or IrdGtConfig()
    rng = np.random.default_rng(cfg.seed)
    from ird_playground.ird.capability_io import unpack_bits_5dof

    n_int, n_bnd, n_ext = int(cfg.n_interior), int(cfg.n_boundary), int(cfg.n_exterior)
    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    n_orient = int(orients.shape[0])
    voxel_xyz = cm.grid.center_of(cm.voxel_ids)
    d_vals = np.asarray(cm.d_value, dtype=np.float64)
    M = int(cm.voxel_ids.shape[0])
    shape = tuple(int(s) for s in cm.grid.shape)
    nx, ny, nz = shape
    step = float(cm.grid.step_m)
    origin = np.asarray(cm.grid.origin_m, dtype=np.float64)

    print(f"[gt] unpack bitmask M={M:,} n_orient={n_orient}", flush=True)
    bits = (
        unpack_bits_5dof(np.asarray(cm.bitmask), n_orient)
        if cm.roll is None
        else np.any(cm.bitmask, axis=-1)
    )
    pos_rows, pos_oids = np.nonzero(bits)
    print(
        f"[gt] MC hits={pos_rows.size:,} ({100.0 * pos_rows.size / max(bits.size, 1):.3f}% of "
        f"{bits.size:,} sparse bins) — bit=0 is NOT verified unreachable",
        flush=True,
    )

    lin = (
        cm.voxel_ids[:, 0].astype(np.int64) * (ny * nz)
        + cm.voxel_ids[:, 1].astype(np.int64) * nz
        + cm.voxel_ids[:, 2].astype(np.int64)
    )
    row_of = -np.ones(nx * ny * nz, dtype=np.int32)
    row_of[lin] = np.arange(M, dtype=np.int32)

    knn = _orient_knn(orients, int(cfg.orient_knn))
    spat = np.array(
        [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.int32,
    )

    def _lookup_rows(ijk: np.ndarray) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=np.int32)
        n = ijk.shape[0]
        inb = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
        )
        rows = np.full(n, -1, dtype=np.int32)
        if inb.any():
            keys = (
                ijk[inb, 0].astype(np.int64) * (ny * nz)
                + ijk[inb, 1].astype(np.int64) * nz
                + ijk[inb, 2].astype(np.int64)
            )
            rows[inb] = row_of[keys]
        return rows

    def local_orient_hit_count(
        ijk: np.ndarray,
        oids: np.ndarray,
        spatial_offsets: np.ndarray,
    ) -> np.ndarray:
        """Count local MC hits over spatial_offsets × orient-KNN (integer)."""
        ijk = np.asarray(ijk, dtype=np.int32)
        oids = np.asarray(oids, dtype=np.int32)
        n = oids.shape[0]
        o_nb = knn[oids]
        count = np.zeros(n, dtype=np.int32)
        for dlt in spatial_offsets:
            rows = _lookup_rows(ijk + dlt)
            ok = rows >= 0
            if ok.any():
                count[ok] += bits[rows[ok][:, None], o_nb[ok]].sum(axis=1).astype(np.int32)
        return count

    def soft_at(ijk: np.ndarray, oids: np.ndarray) -> np.ndarray:
        """Local MC-hit fraction (7-spatial × K-orient) — exterior diagnostic only."""
        ijk = np.asarray(ijk, dtype=np.int32)
        oids = np.asarray(oids, dtype=np.int32)
        n = oids.shape[0]
        o_nb = knn[oids]
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        for dlt in spat:
            rows = _lookup_rows(ijk + dlt)
            ok = rows >= 0
            if not ok.any():
                continue
            acc[ok] += bits[rows[ok][:, None], o_nb[ok]].mean(axis=1)
            cnt[ok] += 1.0
        return (acc / np.maximum(cnt, 1.0)).astype(np.float32)

    def soft_at_batched(ijk: np.ndarray, oids: np.ndarray) -> np.ndarray:
        return soft_at(ijk, oids)

    d_max = float(max(d_vals.max(), 1e-6))
    d_n = np.clip(d_vals / d_max, 0.0, 1.0).astype(np.float32)
    if cm.mu_mean is not None:
        mu = np.asarray(cm.mu_mean, dtype=np.float64)
        q_manip = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0).astype(np.float32)
        q_manip[~np.isfinite(mu)] = d_n[~np.isfinite(mu)]
    else:
        q_manip = d_n.copy()
    q_cap = np.clip(cfg.w_manip * q_manip + cfg.w_d * d_n, 0.0, 1.0).astype(np.float32)

    d_hi = float(np.percentile(d_vals, 70))
    d_lo = float(np.percentile(d_vals, 35))
    is_int = d_vals >= d_hi
    is_bnd = (d_vals >= d_lo) & (d_vals < d_hi)
    if not is_int.any():
        is_int[:] = True
    if not is_bnd.any():
        is_bnd = is_int.copy()

    pool_int = np.flatnonzero(is_int[pos_rows])
    pool_bnd = np.flatnonzero(is_bnd[pos_rows])
    if pool_int.size == 0:
        pool_int = np.arange(pos_rows.size)
    if pool_bnd.size == 0:
        pool_bnd = pool_int

    # Trusted negatives: voxels with D≈0 (no MC hit at all) — still sparse-map cells
    # or off-map. On-map zeros with soft≈0 for a random orient.
    zero_d_rows = np.flatnonzero(d_vals <= 1e-8)
    print(f"[gt] zero-D voxels (trusted exterior candidates)={zero_d_rows.size:,}", flush=True)

    chunks: dict[str, list] = {k: [] for k in ("f", "y", "ys", "cw", "m", "q", "qm", "mw", "layer", "vid", "oid")}

    def flush(ps, us, y, y_soft, cw, m, mw, layer, rows_or_none, oids):
        feat = features_from_p_u(ps, us)
        n = len(y)
        q = np.zeros(n, dtype=np.float32)
        qm = np.zeros(n, dtype=np.float32)
        vid = np.full(n, -1, dtype=np.int32)
        if rows_or_none is not None:
            pos = np.asarray(y) >= 0.5
            rows_or_none = np.asarray(rows_or_none, dtype=np.int32)
            ok = pos & (rows_or_none >= 0)
            if ok.any():
                q[ok] = q_cap[rows_or_none[ok]]
                qm[ok] = q_manip[rows_or_none[ok]]
            vid[:] = rows_or_none
        layer_arr = np.asarray(layer, dtype=np.int32)
        if layer_arr.ndim == 0:
            layer_arr = np.full(n, int(layer_arr), dtype=np.int32)
        chunks["f"].append(feat)
        chunks["y"].append(np.asarray(y, dtype=np.float32))
        chunks["ys"].append(np.asarray(y_soft, dtype=np.float32))
        chunks["cw"].append(np.asarray(cw, dtype=np.float32))
        chunks["m"].append(np.asarray(m, dtype=np.float32))
        chunks["q"].append(q)
        chunks["qm"].append(qm)
        chunks["mw"].append(np.asarray(mw, dtype=np.float32))
        chunks["layer"].append(layer_arr)
        chunks["vid"].append(vid)
        chunks["oid"].append(np.asarray(oids, dtype=np.int32))

    # --- Interior: exact MC hits only (trusted positives) ---
    print(f"[gt] interior exact-hits {n_int:,}", flush=True)
    for s in range(0, n_int, batch_size):
        n = min(batch_size, n_int - s)
        ti = pool_int[rng.integers(0, pool_int.size, size=n)]
        rows, oids = pos_rows[ti], pos_oids[ti]
        y = np.ones(n, dtype=np.float32)
        ys = np.ones(n, dtype=np.float32)
        cw = np.ones(n, dtype=np.float32)
        m = np.full(n, cfg.m_eps, dtype=np.float32)
        mw = np.zeros(n, dtype=np.float32)
        flush(voxel_xyz[rows], orients[oids], y, ys, cw, m, mw, LAYER_INTERIOR, rows, oids)

    # --- Boundary face pairs with stable-support filter (v6) ---
    print("[gt] boundary face pairs…", flush=True)
    neigh = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.int32)
    n_cand = min(400_000, pool_bnd.size)
    ti = pool_bnd[rng.choice(pool_bnd.size, size=n_cand, replace=False)]
    rows_c, oids_c = pos_rows[ti], pos_oids[ti]
    ijk0 = cm.voxel_ids[rows_c].astype(np.int32)
    assigned = np.zeros(n_cand, dtype=bool)
    bnd_r = np.empty(n_cand, dtype=np.int32)
    bnd_o = np.empty(n_cand, dtype=np.int32)
    bnd_ijk_neg = np.empty((n_cand, 3), dtype=np.int32)
    bnd_dlt = np.empty((n_cand, 3), dtype=np.int32)
    for dlt in neigh:
        j = ijk0 + dlt
        out_of = (
            (j[:, 0] < 0) | (j[:, 0] >= nx) | (j[:, 1] < 0) | (j[:, 1] >= ny) | (j[:, 2] < 0) | (j[:, 2] >= nz)
        )
        keys = (
            np.clip(j[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
            + np.clip(j[:, 1], 0, ny - 1).astype(np.int64) * nz
            + np.clip(j[:, 2], 0, nz - 1).astype(np.int64)
        )
        r2 = row_of[keys]
        reachable_nb = np.zeros(n_cand, dtype=bool)
        ok = (~out_of) & (r2 >= 0)
        if ok.any():
            reachable_nb[ok] = bits[r2[ok], oids_c[ok]]
        fail = (~assigned) & (out_of | (~reachable_nb))
        if fail.any():
            bnd_r[fail] = rows_c[fail]
            bnd_o[fail] = oids_c[fail]
            bnd_ijk_neg[fail] = j[fail]
            bnd_dlt[fail] = dlt
            assigned[fail] = True
        if assigned.all():
            break
    keep = assigned
    bnd_r, bnd_o, bnd_ijk_neg, bnd_dlt = (
        bnd_r[keep],
        bnd_o[keep],
        bnd_ijk_neg[keep],
        bnd_dlt[keep],
    )
    print(f"[gt] face pairs kept={bnd_r.size:,}", flush=True)
    if bnd_r.size == 0:
        raise RuntimeError("no boundary face pairs found")

    # v6: C+ / C- support on non-overlapping half-neighborhoods
    print("[gt] stable-support filter (C+, C-)…", flush=True)
    ijk_pos = cm.voxel_ids[bnd_r].astype(np.int32)
    support_pos = np.zeros(bnd_r.size, dtype=np.int32)
    support_neg = np.zeros(bnd_r.size, dtype=np.int32)
    for dlt in neigh:
        mask = np.all(bnd_dlt == dlt, axis=1)
        if not mask.any():
            continue
        pos_off = _half_neighborhood(dlt, positive_side=True)
        neg_off = _half_neighborhood(dlt, positive_side=False)
        support_pos[mask] = local_orient_hit_count(ijk_pos[mask], bnd_o[mask], pos_off)
        support_neg[mask] = local_orient_hit_count(bnd_ijk_neg[mask], bnd_o[mask], neg_off)

    cmin = int(cfg.min_positive_support)
    trusted = (support_pos >= cmin) & (support_neg == 0)
    if trusted.sum() < int(cfg.min_trusted_face_pairs) and cmin > 2:
        print(
            f"[gt] C+>={cmin} & C-=0 → {trusted.sum():,} pairs; "
            f"relaxing min_positive_support to 2",
            flush=True,
        )
        cmin = 2
        trusted = (support_pos >= cmin) & (support_neg == 0)

    trusted_idx = np.flatnonzero(trusted)
    qs = [0.0, 0.1, 0.5, 0.9, 1.0]
    print(
        f"[gt] support_pos quantiles={np.quantile(support_pos, qs).astype(int).tolist()} "
        f"support_neg quantiles={np.quantile(support_neg, qs).astype(int).tolist()}",
        flush=True,
    )
    print(
        f"[gt] trusted faces={trusted_idx.size:,}/{bnd_r.size:,} "
        f"(C+>={cmin} & C-=0); rejected={bnd_r.size - trusted_idx.size:,}",
        flush=True,
    )
    if trusted_idx.size < int(cfg.min_trusted_face_pairs):
        raise RuntimeError(
            f"Not enough trusted boundary pairs ({trusted_idx.size} < {cfg.min_trusted_face_pairs}). "
            "Increase MC coverage or lower min_positive_support explicitly — "
            "do NOT fall back to all face pairs."
        )

    # Cap n_bnd / n_jitter to available trusted diversity (no fake continuity)
    n_trusted = int(trusted_idx.size)
    n_bnd_eff = min(n_bnd, max(n_trusted * 2, n_trusted))
    if n_bnd_eff < n_bnd:
        print(f"[gt] capping boundary samples {n_bnd:,} → {n_bnd_eff:,}", flush=True)
    n_bnd = n_bnd_eff

    print(f"[gt] boundary interpolate {n_bnd:,} (trusted face pairs only)", flush=True)
    for s in range(0, n_bnd, batch_size):
        n = min(batch_size, n_bnd - s)
        pick = trusted_idx[rng.integers(0, trusted_idx.size, size=n)]
        rows = bnd_r[pick]
        oids = bnd_o[pick]
        ijk_neg = bnd_ijk_neg[pick]
        p_pos = voxel_xyz[rows]
        p_neg = origin + step * (ijk_neg.astype(np.float64) + 0.5)
        alpha = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
        ps = (1.0 - alpha[:, None]) * p_pos + alpha[:, None] * p_neg
        m = ((0.5 - alpha) * step / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
        m = np.clip(m, -cfg.m_clip, cfg.m_clip)
        y = (alpha < 0.5).astype(np.float32)
        m = _enforce_sign(m, y, cfg.m_eps, cfg.m_clip)
        ys = y.copy()
        cw = np.ones(n, dtype=np.float32)
        mw = np.ones(n, dtype=np.float32)
        layer = np.where(y >= 0.5, LAYER_BND_POS, LAYER_BND_NEG).astype(np.int32)
        rows_q = np.where(y >= 0.5, rows, -1)
        flush(ps, orients[oids], y, ys, cw, m, mw, layer, rows_q, oids)

    # --- Jitter from face normal (pos/neg half-half), NOT isotropic MC-noise ---
    n_jit = int(cfg.n_jitter)
    n_jit = min(n_jit, max(n_trusted * 2, n_trusted))
    n_jp = n_jit // 2
    n_jn = n_jit - n_jp
    print(f"[gt] face-normal jitter pos={n_jp:,} neg={n_jn:,}", flush=True)

    def face_jitter(n_samples: int, positive: bool):
        pick = trusted_idx[rng.integers(0, trusted_idx.size, size=n_samples)]
        rows = bnd_r[pick]
        oids = bnd_o[pick]
        ijk_neg = bnd_ijk_neg[pick]
        p_plus = voxel_xyz[rows]
        p_minus = origin + step * (ijk_neg.astype(np.float64) + 0.5)
        p_face = 0.5 * (p_plus + p_minus)
        nrm = p_minus - p_plus
        nn = np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        n_hat = nrm / nn
        delta = rng.uniform(0.05 * step, 0.45 * step, size=n_samples)
        # tangent noise
        a = np.where(np.abs(n_hat[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        t1 = np.cross(a, n_hat)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
        t2 = np.cross(n_hat, t1)
        rad = rng.uniform(0.0, 0.35 * step, size=n_samples)
        ang = rng.uniform(0.0, 2 * np.pi, size=n_samples)
        tang = (rad * np.cos(ang))[:, None] * t1 + (rad * np.sin(ang))[:, None] * t2
        if positive:
            ps = p_face - delta[:, None] * n_hat + tang
            y = np.ones(n_samples, dtype=np.float32)
            layer = LAYER_JITTER_POS
            m = (delta / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
            m = np.clip(m, cfg.m_eps, cfg.m_clip)
            rows_out = rows
        else:
            ps = p_face + delta[:, None] * n_hat + tang
            y = np.zeros(n_samples, dtype=np.float32)
            layer = LAYER_JITTER_NEG
            m = (-delta / max(cfg.sigma_p_m, 1e-6)).astype(np.float32)
            m = np.clip(m, -cfg.m_clip, -cfg.m_eps)
            rows_out = np.full(n_samples, -1, dtype=np.int32)
        ys = y.copy()
        cw = np.ones(n_samples, dtype=np.float32)
        mw = np.ones(n_samples, dtype=np.float32)
        flush(ps, orients[oids], y, ys, cw, m, mw, layer, rows_out, oids)

    for s in range(0, n_jp, batch_size):
        face_jitter(min(batch_size, n_jp - s), True)
    for s in range(0, n_jn, batch_size):
        face_jitter(min(batch_size, n_jn - s), False)

    # --- Exterior trusted negatives: soft≈0 on-map bit=0 + off-map ---
    # Saved voxels almost never have D=0 (they exist because some orient hit).
    n_hard = int(round(n_ext * cfg.hard_negative_frac))
    n_unif = max(0, n_ext - n_hard)
    print(f"[gt] trusted exterior soft0={n_hard:,} offmap={n_unif:,}", flush=True)

    if n_hard:
        # Rejection-sample (row, oid) with exact bit=0 and local soft≈0 (far from MC hits)
        got = 0
        attempts = 0
        max_attempts = max(40, (n_hard // batch_size) * 80)
        thr = float(cfg.trusted_neg_soft_max)
        while got < n_hard and attempts < max_attempts:
            attempts += 1
            n = min(batch_size * 4, max(batch_size, (n_hard - got) * 4))
            rows = rng.integers(0, M, size=n).astype(np.int32)
            oids = rng.integers(0, n_orient, size=n).astype(np.int32)
            hit = bits[rows, oids]
            soft = soft_at_batched(cm.voxel_ids[rows], oids)
            keep = (~hit) & (soft <= thr)
            if not keep.any() and attempts > max_attempts // 4:
                thr = float(cfg.soft_tau)  # relax once if too strict
                keep = (~hit) & (soft <= thr)
            if not keep.any():
                continue
            take = min(int(keep.sum()), n_hard - got)
            sel = np.flatnonzero(keep)[:take]
            rows, oids, soft = rows[sel], oids[sel], soft[sel]
            n = len(rows)
            y = np.zeros(n, dtype=np.float32)
            ys = soft
            cw = np.ones(n, dtype=np.float32)
            m = np.full(n, -cfg.m_eps, dtype=np.float32)
            mw = np.zeros(n, dtype=np.float32)
            flush(voxel_xyz[rows], orients[oids], y, ys, cw, m, mw, LAYER_EXTERIOR, None, oids)
            got += n
        print(f"[gt] soft0 exterior accepted={got:,} attempts={attempts} thr={thr}", flush=True)

    if n_unif:
        mins = voxel_xyz.min(0) - float(cfg.bbox_margin_m)
        maxs = voxel_xyz.max(0) + float(cfg.bbox_margin_m)
        got = 0
        attempts = 0
        max_attempts = max(20, (n_unif // batch_size) * 40)
        while got < n_unif and attempts < max_attempts:
            attempts += 1
            n = min(batch_size, n_unif - got)
            ps = rng.uniform(mins, maxs, size=(n, 3))
            oids = rng.integers(0, n_orient, size=n).astype(np.int32)
            ijk = np.floor((ps - origin) / step).astype(np.int32)
            inb = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < nx)
                & (ijk[:, 1] >= 0) & (ijk[:, 1] < ny)
                & (ijk[:, 2] >= 0) & (ijk[:, 2] < nz)
            )
            soft = np.zeros(n, dtype=np.float32)
            if inb.any():
                soft[inb] = soft_at_batched(ijk[inb], oids[inb])
            # keep clearly off-map OR soft≈0; never exact hits
            hit = np.zeros(n, dtype=bool)
            keys = (
                np.clip(ijk[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
                + np.clip(ijk[:, 1], 0, ny - 1).astype(np.int64) * nz
                + np.clip(ijk[:, 2], 0, nz - 1).astype(np.int64)
            )
            r = np.full(n, -1, dtype=np.int32)
            r[inb] = row_of[keys[inb]]
            ok = r >= 0
            if ok.any():
                hit[ok] = bits[r[ok], oids[ok]]
            keep = ((~inb) | (soft <= cfg.soft_tau)) & (~hit)
            if not keep.any():
                continue
            n = int(keep.sum())
            y = np.zeros(n, dtype=np.float32)
            ys = soft[keep]
            cw = np.ones(n, dtype=np.float32)
            m = np.full(n, -cfg.m_eps, dtype=np.float32)
            mw = np.zeros(n, dtype=np.float32)
            flush(ps[keep], orients[oids[keep]], y, ys, cw, m, mw, LAYER_EXTERIOR, None, oids[keep])
            got += n
        print(f"[gt] offmap/soft0 exterior accepted={got:,} attempts={attempts}", flush=True)

    features = np.concatenate(chunks["f"], axis=0)
    y = np.concatenate(chunks["y"], axis=0)
    y_soft = np.concatenate(chunks["ys"], axis=0)
    cw = np.concatenate(chunks["cw"], axis=0)
    m_arr = np.concatenate(chunks["m"], axis=0)
    q_arr = np.concatenate(chunks["q"], axis=0)
    qm_arr = np.concatenate(chunks["qm"], axis=0)
    mw_arr = np.concatenate(chunks["mw"], axis=0)
    layer = np.concatenate(chunks["layer"], axis=0)
    vid = np.concatenate(chunks["vid"], axis=0)
    oid = np.concatenate(chunks["oid"], axis=0)

    q_arr = np.where(y >= 0.5, q_arr, 0.0).astype(np.float32)
    qm_arr = np.where(y >= 0.5, qm_arr, 0.0).astype(np.float32)
    mw_pos = mw_arr > 0
    if mw_pos.any():
        m_arr[mw_pos] = _enforce_sign(m_arr[mw_pos], y[mw_pos], cfg.m_eps, cfg.m_clip)

    max_abs = np.max(np.abs(features[:, :3]), axis=0)
    scale = np.maximum(max_abs * 1.05, 0.1).astype(np.float32)
    aabb_lo, aabb_hi = -scale, scale.copy()

    # sign only where margin supervised
    bad = mw_pos & (((y >= 0.5) & (m_arr <= 0.0)) | ((y < 0.5) & (m_arr >= 0.0)))
    if bad.any():
        raise RuntimeError(f"sign conflict on margin_weight>0: {bad.mean():.4%} n={bad.sum()}")
    outside = np.any((features[:, :3] < aabb_lo) | (features[:, :3] > aabb_hi), axis=1)
    if outside.mean() > 1e-4:
        raise RuntimeError(f"outside AABB {outside.mean():.4%}")

    ijk_feat = np.floor((features[:, :3] - origin) / step).astype(np.int32)
    block = (
        (np.clip(ijk_feat[:, 0], 0, nx - 1) // 8).astype(np.int64) * 1_000_000
        + (np.clip(ijk_feat[:, 1], 0, ny - 1) // 8).astype(np.int64) * 1_000
        + (np.clip(ijk_feat[:, 2], 0, nz - 1) // 8).astype(np.int64)
        + oid.astype(np.int64) * 10_000_000_000
    )

    perm = rng.permutation(features.shape[0])
    n = int(features.shape[0])
    supervised = cw > 0
    print(
        f"[gt] N={n:,} reach={float(y.mean()):.3f} supervised={float(supervised.mean()):.3f} "
        f"sup_pos={float(y[supervised].mean()) if supervised.any() else 0:.3f} "
        f"layers={dict(zip(*np.unique(layer, return_counts=True)))}",
        flush=True,
    )
    return {
        "features": features[perm],
        "reachable": y[perm],
        "p_reach": y[perm],
        "y_soft": y_soft[perm],
        "cls_weight": cw[perm],
        "m_gt": m_arr[perm],
        "margin_weight": mw_arr[perm],
        "layer_id": layer[perm],
        "voxel_id": vid[perm],
        "orient_id": oid[perm],
        "block_id": block[perm],
        "q": q_arr[perm],
        "q_comfort": q_arr[perm],
        "q_capability": q_arr[perm],
        "q_manip": qm_arr[perm],
        "q_joint": q_arr[perm],
        "q_selfcol": q_arr[perm],
        "q_nullspace": q_arr[perm],
        "q_best": np.zeros((n, cfg.n_dof), dtype=np.float32),
        "q_candidates": np.zeros((n, cfg.k_candidates, cfg.n_dof), dtype=np.float32),
        "d": (y * q_arr)[perm],
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([cfg.sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.array([cfg.sigma_r_deg], dtype=np.float32),
        "feature_dim": np.array([6], dtype=np.int32),
        "feature_kind": np.array([1], dtype=np.int32),
        "label_kind": np.array([3], dtype=np.int32),  # 3 = stable-support v6
    }


def save_ird_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if meta is not None:
        path.with_suffix(".yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return path


def load_ird_gt(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def assert_gt_contract(arrays: dict[str, np.ndarray]) -> None:
    x, y, m = arrays["features"], arrays["reachable"], arrays["m_gt"]
    q = arrays["q"]
    lo, hi = arrays["aabb_lo"], arrays["aabb_hi"]
    assert x.shape[1] == 6
    assert np.isfinite(x).all() and np.isfinite(m).all() and np.isfinite(q).all()
    cw = arrays.get("cls_weight")
    mw = arrays.get("margin_weight")
    if mw is not None:
        mask = mw > 0
        if mask.any():
            bad = ((y[mask] > 0.5) & (m[mask] <= 0.0)) | ((y[mask] < 0.5) & (m[mask] >= 0.0))
            assert float(bad.mean()) < 1e-5, f"sign conflict {bad.mean()}"
    if cw is not None:
        assert float((cw >= 0).mean()) == 1.0
    outside = np.any((x[:, :3] < lo) | (x[:, :3] > hi), axis=1)
    assert float(outside.mean()) < 1e-4, f"outside AABB {outside.mean()}"


def make_synthetic_ird_gt(n: int = 4096, *, seed: int = 0, reach_radius: float = 0.6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(-1.0, 1.0, size=(n, 3))
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    features = features_from_p_u(p, u)
    dist = np.linalg.norm(p, axis=1)
    y = (dist < reach_radius).astype(np.float32)
    m = np.clip((reach_radius - dist) / reach_radius * 3.0, -3.0, 3.0)
    m = _enforce_sign(m, y, 0.05, 3.0)
    mw = (np.abs(dist - reach_radius) < 0.15).astype(np.float32)
    # unknown band
    unknown = (np.abs(dist - reach_radius) >= 0.15) & (np.abs(dist - reach_radius) < 0.25)
    cw = (~unknown).astype(np.float32)
    q = (np.clip(1.0 - dist / (reach_radius + 1e-6), 0, 1) * y).astype(np.float32)
    layer = np.full(n, LAYER_INTERIOR, dtype=np.int32)
    layer = np.where(mw > 0, np.where(y >= 0.5, LAYER_BND_POS, LAYER_BND_NEG), layer)
    scale = np.maximum(np.max(np.abs(features[:, :3]), axis=0) * 1.05, 0.1).astype(np.float32)
    return {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "y_soft": y,
        "cls_weight": cw,
        "m_gt": m,
        "margin_weight": mw,
        "layer_id": layer,
        "voxel_id": np.arange(n, dtype=np.int32),
        "orient_id": np.zeros(n, dtype=np.int32),
        "block_id": (np.floor(p[:, 0] * 4).astype(np.int64) * 1000 + np.floor(p[:, 1] * 4).astype(np.int64)),
        "q": q,
        "q_comfort": q,
        "q_capability": q,
        "q_manip": q,
        "q_joint": q,
        "q_selfcol": q,
        "q_nullspace": q,
        "q_best": np.zeros((n, 7), dtype=np.float32),
        "q_candidates": np.zeros((n, 4, 7), dtype=np.float32),
        "d": y * q,
        "aabb_lo": -scale,
        "aabb_hi": scale,
        "sigma_p_m": np.array([0.03], dtype=np.float32),
        "sigma_r_deg": np.array([10.0], dtype=np.float32),
        "feature_dim": np.array([6], dtype=np.int32),
        "feature_kind": np.array([1], dtype=np.int32),
        "label_kind": np.array([2], dtype=np.int32),
    }
```

### `ird/capability_io.py`

```python
"""File-format CapabilityMap loader (no rm75_control package import).

Reads the same on-disk layout as ``rm75_control.tools.reachability`` CapabilityMap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class SimpleVoxelGrid:
    origin_m: np.ndarray
    step_m: float
    shape: tuple[int, int, int]

    def center_of(self, ijk: np.ndarray) -> np.ndarray:
        arr = np.asarray(ijk, dtype=np.float64)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        c = self.origin_m[None, :] + self.step_m * (arr + 0.5)
        return c[0] if single else c


@dataclass
class SimpleOrientations:
    vectors: np.ndarray

    @property
    def n(self) -> int:
        return int(self.vectors.shape[0])


@dataclass
class LoadedCapabilityMap:
    grid: SimpleVoxelGrid
    orientations: SimpleOrientations
    roll: object | None
    voxel_ids: np.ndarray
    bitmask: np.ndarray
    d_value: np.ndarray
    mu_mean: np.ndarray | None
    n_orient: int
    manifest: dict


def unpack_bits_5dof(packed: np.ndarray, n_orient: int) -> np.ndarray:
    m, n_bytes = packed.shape
    out = np.zeros((m, n_bytes * 8), dtype=bool)
    for k in range(8):
        out[:, k::8] = ((packed >> k) & 1).astype(bool)
    return out[:, :n_orient]


def load_capability_map_dir(map_dir: str | Path, *, mmap: bool = True) -> LoadedCapabilityMap:
    p = Path(map_dir)
    manifest = yaml.safe_load((p / "manifest.yaml").read_text(encoding="utf-8"))
    g = manifest["grid"]
    grid = SimpleVoxelGrid(
        origin_m=np.asarray(g["origin_m"], dtype=np.float64),
        step_m=float(g["step_m"]),
        shape=tuple(int(s) for s in g["shape"]),
    )
    vectors = np.load(p / "orientations.npy").astype(np.float64)
    voxels = np.load(p / "voxels.npz")
    bitmask = np.load(p / "bitmask.npy", mmap_mode=("r" if mmap else None))
    mu = voxels["mu_mean"] if "mu_mean" in voxels.files else None
    n_orient = int(manifest["layout"]["n_orient"])
    roll = manifest.get("roll")
    return LoadedCapabilityMap(
        grid=grid,
        orientations=SimpleOrientations(vectors=vectors),
        roll=roll,
        voxel_ids=voxels["ijk"].astype(np.int32),
        bitmask=bitmask,
        d_value=voxels["d_value"].astype(np.float32),
        mu_mean=(mu.astype(np.float32) if mu is not None else None),
        n_orient=n_orient,
        manifest=manifest,
    )
```

### `region/local_region.py`

```python
"""Sprint 0A Region A: local ellipsoid + tool-axis cone around G(λ).

Outer NLP: only (λ, r).
Inner fixed joint Sobol (K=32): [ε_t, ε_b, ε_n, z, v] →
  p = p0 + R_local @ (a ⊙ ε)
  u = Exp([δω]_×) u0   with area-uniform cone (ρ, φ)

No G(y+δy), no perception stack, no rail/plan neighborhood product.
Optional Sprint 0B: shared global SE(3) bias + tiny shared δr (off by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

from ird_playground.neural.cost import optimization_cost


@dataclass
class LocalExtent:
    """Local ellipsoid half-axes (m) + cone half-angle (deg)."""

    a_t_m: float = 0.003
    a_b_m: float = 0.004
    a_n_m: float = 0.002
    beta_max_deg: float = 3.0
    # Sprint 0B (default off)
    global_trans_m: float = 0.0
    global_rot_deg: float = 0.0
    rail_bias_m: float = 0.0


@dataclass
class RegionAggConfig:
    logit_safe: float = 1.0
    margin_safe: float = 0.20
    tau_logit: float = 0.5
    tau_margin: float = 0.1
    w_cls: float = 1.0
    w_margin: float = 0.5
    w_q: float = 0.2
    w_mean: float = 0.3
    w_worst: float = 0.7
    w_cov: float = 1.0
    p_min: float = 0.90
    tau_p: float = 0.1
    tau_worst: float = 0.5


def make_joint_sobol_ellipsoid_cone(
    num_samples: int = 32,
    *,
    extent: LocalExtent | None = None,
    seed: int = 0,
    antithetic: bool = True,
    device=None,
    dtype=None,
) -> "torch.Tensor":
    """Fixed joint Sobol → physical (δp_t, δp_b, δp_n, ρ, φ).

    Unit cube u∈[0,1]^5 mapped by:
      ε_* = 2u-1 ∈ [-1,1] for ellipsoid axes
      cosρ = 1 - z (1 - cos β_max)   (area-uniform spherical cap)
      φ = 2π v
    Returns (K,5) tensor. Always include an explicit center row at index 0.
    """
    if torch is None:
        raise ImportError("torch required")
    from scipy.stats import qmc

    ext = extent or LocalExtent()
    beta = float(np.deg2rad(ext.beta_max_deg))
    cos_b = float(np.cos(beta))

    n_draw = max(num_samples - 1, 1)
    if antithetic:
        n_pair = (n_draw + 1) // 2
        eng = qmc.Sobol(d=5, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_pair, 2))))
        u = eng.random_base2(m)[:n_pair]
        # antithetic in unit cube about 0.5
        paired = np.empty((u.shape[0] * 2, 5), dtype=np.float64)
        paired[0::2] = u
        paired[1::2] = 1.0 - u
        u_all = paired[:n_draw]
    else:
        eng = qmc.Sobol(d=5, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n_draw, 2))))
        u_all = eng.random_base2(m)[:n_draw]

    eps = 2.0 * u_all[:, :3] - 1.0
    dpt = eps[:, 0] * ext.a_t_m
    dpb = eps[:, 1] * ext.a_b_m
    dpn = eps[:, 2] * ext.a_n_m
    z = u_all[:, 3]
    v = u_all[:, 4]
    cos_rho = 1.0 - z * (1.0 - cos_b)
    rho = np.arccos(np.clip(cos_rho, -1.0, 1.0))
    phi = 2.0 * np.pi * v
    samples = np.stack([dpt, dpb, dpn, rho, phi], axis=1)
    center = np.zeros((1, 5), dtype=np.float64)
    arr = np.concatenate([center, samples], axis=0)[:num_samples]
    return torch.as_tensor(arr, device=device, dtype=dtype or torch.float32)


def _rodrigues(u: "torch.Tensor", dw: "torch.Tensor") -> "torch.Tensor":
    """Rotate unit vectors u by small/finite rotvecs dw (same broadcast shape)."""
    theta = dw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = dw / theta
    # Rodrigues: u' = u cosθ + (k×u) sinθ + k (k·u) (1-cosθ)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    kxu = torch.linalg.cross(k, u)
    kdu = (k * u).sum(dim=-1, keepdim=True)
    return u * cos_t + kxu * sin_t + k * kdu * (1.0 - cos_t)


def _frames_from_u(u: "torch.Tensor", t_ref: "torch.Tensor") -> "torch.Tensor":
    u = F.normalize(u, dim=-1, eps=1e-8)
    t = t_ref - u * (t_ref * u).sum(dim=-1, keepdim=True)
    t_n = t.norm(dim=-1, keepdim=True)
    fb = torch.zeros_like(t)
    fb[..., 0] = 1.0
    t = torch.where(t_n > 1e-5, t / t_n.clamp_min(1e-6), fb)
    t = t - u * (t * u).sum(dim=-1, keepdim=True)
    t = F.normalize(t, dim=-1, eps=1e-8)
    b = F.normalize(torch.linalg.cross(t, u), dim=-1, eps=1e-8)
    t = torch.linalg.cross(u, b)
    return torch.stack([b, t, u], dim=-1)


def _se3_exp_batch(xi: "torch.Tensor") -> "torch.Tensor":
    """ξ (...,6)=[δp,δω] → (...,4,4)."""
    dp, dw = xi[..., :3], xi[..., 3:]
    theta = dw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = dw / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    z = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([z, -kz, ky], dim=-1),
            torch.stack([kz, z, -kx], dim=-1),
            torch.stack([-ky, kx, z], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device).expand_as(K)
    th = theta.unsqueeze(-1)
    R = eye + torch.sin(th) * K + (1.0 - torch.cos(th)) * (K @ K)
    small = (dw.norm(dim=-1) < 1e-6)[..., None, None]
    R = torch.where(small, eye + K * th, R)
    T = torch.eye(4, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 4, 4).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = dp
    return T


def local_region_cost(
    point_ird,
    lambda_center: "torch.Tensor",
    rail_center: "torch.Tensor",
    surface_model,
    *,
    local_eps: "torch.Tensor" | None = None,
    extent: LocalExtent | None = None,
    agg: RegionAggConfig | None = None,
    num_samples: int = 32,
    sobol_seed: int = 0,
    T_world_rail: np.ndarray | "torch.Tensor" | None = None,
    T_rail_base0: np.ndarray | "torch.Tensor" | None = None,
) -> dict[str, "torch.Tensor"]:
    """Ellipsoid+cone Region A; returns cost / coverage / per-knot tensors."""
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.ird.query_base import (
        features_from_delta_T_torch,
        invert_T_torch,
        T_base_from_rail_y_torch,
    )

    ext = extent or LocalExtent()
    agg = agg or RegionAggConfig()
    device = point_ird.device
    dtype = torch.float32

    lam = lambda_center.to(device=device, dtype=dtype).reshape(-1)
    rail = rail_center.to(device=device, dtype=dtype).reshape(-1)
    if lam.shape != rail.shape:
        raise ValueError("lambda_center and rail_center must match")
    n = int(lam.shape[0])

    if local_eps is None:
        local_eps = make_joint_sobol_ellipsoid_cone(
            num_samples, extent=ext, seed=sobol_seed, device=device, dtype=dtype
        )
    else:
        local_eps = local_eps.to(device=device, dtype=dtype)
    k = int(local_eps.shape[0])

    if not hasattr(surface_model, "sample_torch"):
        raise TypeError("surface_model must implement sample_torch")
    T0 = surface_model.sample_torch(lam, dtype=dtype, device=device)
    if T0.ndim == 2:
        T0 = T0.unsqueeze(0)
    p0 = T0[..., :3, 3]
    # R_local columns in SyntheticVesselSkinManifold: [b, t, n]
    b = T0[..., :3, 0]
    t = T0[..., :3, 1]
    nn = T0[..., :3, 2]
    u0 = nn

    dpt, dpb, dpn = local_eps[:, 0], local_eps[:, 1], local_eps[:, 2]
    rho, phi = local_eps[:, 3], local_eps[:, 4]

    p = (
        p0[:, None, :]
        + dpt[None, :, None] * t[:, None, :]
        + dpb[None, :, None] * b[:, None, :]
        + dpn[None, :, None] * nn[:, None, :]
    )
    # cone: δω = ρ (cosφ t + sinφ b)
    dw = (
        rho[None, :, None]
        * (
            torch.cos(phi)[None, :, None] * t[:, None, :]
            + torch.sin(phi)[None, :, None] * b[:, None, :]
        )
    )
    u = F.normalize(_rodrigues(u0[:, None, :].expand(-1, k, -1), dw), dim=-1, eps=1e-8)

    R = _frames_from_u(u, t_ref=t[:, None, :].expand(-1, k, -1))
    T_local = torch.eye(4, dtype=dtype, device=device).expand(n, k, 4, 4).clone()
    T_local[..., :3, :3] = R
    T_local[..., :3, 3] = p

    # Optional Sprint 0B: one global SE(3) bias per scenario, shared across knots
    if ext.global_trans_m > 0.0 or ext.global_rot_deg > 0.0:
        # derive global twists from first 3 pos + angles of each sample (shared over N)
        g_scale_t = float(ext.global_trans_m)
        g_scale_r = float(np.deg2rad(ext.global_rot_deg))
        # use sample index hash from (dpt,phi) so scenarios differ but are fixed
        g_dp = torch.stack(
            [
                torch.tanh(dpt / (ext.a_t_m + 1e-9)) * g_scale_t,
                torch.tanh(dpb / (ext.a_b_m + 1e-9)) * g_scale_t,
                torch.tanh(dpn / (ext.a_n_m + 1e-9)) * g_scale_t,
            ],
            dim=-1,
        )  # (K,3)
        g_dw = torch.stack(
            [
                torch.cos(phi) * (rho / (np.deg2rad(ext.beta_max_deg) + 1e-9)) * g_scale_r,
                torch.sin(phi) * (rho / (np.deg2rad(ext.beta_max_deg) + 1e-9)) * g_scale_r,
                torch.zeros_like(phi),
            ],
            dim=-1,
        )
        xi_g = torch.cat([g_dp, g_dw], dim=-1)  # (K,6)
        T_glob = _se3_exp_batch(xi_g)  # (K,4,4)
        T_tcp = T_glob[None, :, :, :] @ T_local
    else:
        T_tcp = T_local

    if T_world_rail is None:
        Twr = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_world_rail):
        Twr = T_world_rail.to(device=device, dtype=dtype)
    else:
        Twr = torch.as_tensor(np.asarray(T_world_rail), dtype=dtype, device=device)
    if T_rail_base0 is None:
        Trb = torch.eye(4, dtype=dtype, device=device)
    elif torch.is_tensor(T_rail_base0):
        Trb = T_rail_base0.to(device=device, dtype=dtype)
    else:
        Trb = torch.as_tensor(np.asarray(T_rail_base0), dtype=dtype, device=device)

    # Shared rail per knot; optional tiny shared δr per scenario (0B)
    if ext.rail_bias_m > 0.0:
        dr = torch.tanh(dpt / (ext.a_t_m + 1e-9)) * float(ext.rail_bias_m)  # (K,)
        rail_s = rail[:, None] + dr[None, :]  # (N,K)
        rail_flat = rail_s.reshape(-1)
        T_base = T_base_from_rail_y_torch(rail_flat, T_world_rail=Twr, T_rail_base0=Trb)
        T_base = T_base.reshape(n, k, 4, 4)
    else:
        T_base = T_base_from_rail_y_torch(rail, T_world_rail=Twr, T_rail_base0=Trb)
        T_base = T_base[:, None, :, :].expand(-1, k, -1, -1)

    dT = invert_T_torch(T_tcp) @ T_base
    feat = features_from_delta_T_torch(dT).reshape(n * k, 6)

    point_ird.model.eval()
    out = point_ird.model.score_features(feat)
    logit = out["reach_logit"].reshape(n, k)
    margin = out["m"].reshape(n, k)
    quality = out["q"].reshape(n, k)

    c_k = optimization_cost(
        logit,
        margin,
        quality,
        logit_safe=agg.logit_safe,
        margin_safe=agg.margin_safe,
        tau_logit=agg.tau_logit,
        tau_margin=agg.tau_margin,
        w_cls=agg.w_cls,
        w_margin=agg.w_margin,
        w_q=agg.w_q,
    )

    p_cov = torch.sigmoid(logit).mean(dim=1)
    cov_pen = F.softplus((agg.p_min - p_cov) / max(agg.tau_p, 1e-6))

    c_mean = c_k.mean(dim=1)
    tau = max(float(agg.tau_worst), 1e-6)
    c_shift = c_k - c_k.max(dim=1, keepdim=True).values
    c_worst = tau * torch.log(torch.exp(c_shift / tau).mean(dim=1) + 1e-12) + c_k.max(dim=1).values

    cost_per_knot = (
        agg.w_mean * c_mean + agg.w_worst * c_worst + agg.w_cov * cov_pen
    )
    return {
        "cost": cost_per_knot.mean(),
        "cost_per_knot": cost_per_knot,
        "c_mean": c_mean,
        "c_worst": c_worst,
        "p_cov": p_cov,
        "point_cost": c_k,
        "reach_logit": logit,
        "margin": margin,
        "quality": quality,
        "local_eps": local_eps,
        "num_samples": torch.tensor(float(k), device=device),
    }


# Back-compat name
robust_region_cost = local_region_cost
```

### `region/aggregate.py`

```python
"""Query-side Region A: anisotropic Exp perturbations + softmin(m) + mean(q)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from ird_playground.probe.se3 import se3_exp, se3_mul


@dataclass(frozen=True)
class PositionExtent:
    tangent_1_m: float = 0.020
    tangent_2_m: float = 0.010
    normal_m: float = 0.002


@dataclass(frozen=True)
class OrientationExtent:
    tilt_tangent_1_deg: float = 8.0
    tilt_tangent_2_deg: float = 5.0
    axial_roll_deg: float = 3.0


@dataclass
class RegionScore:
    score: float
    m_robust: float
    q_region: float
    mean_score: float
    softmin_score: float
    coverage: float
    min_score: float
    num_samples: int


def sobol_unit_cube(n: int, dim: int, *, seed: int = 0) -> np.ndarray:
    eng = qmc.Sobol(d=dim, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(max(n, 2))))
    u = eng.random_base2(m)
    return u[:n]


def sample_anisotropic_xi(
    position: PositionExtent,
    orientation: OrientationExtent,
    num_samples: int,
    *,
    seed: int = 0,
    antithetic: bool = True,
) -> np.ndarray:
    """Return (K,6) ξ; if antithetic, pair ξ_{2j}=-ξ_{2j-1}."""
    if antithetic:
        n_pair = (num_samples + 1) // 2
        u = sobol_unit_cube(n_pair, 6, seed=seed)
        s = 2.0 * u - 1.0
        # build extents
        pos = PositionExtent(
            position.tangent_1_m, position.tangent_2_m, position.normal_m
        )
        ori = orientation
        b1 = np.deg2rad(ori.tilt_tangent_1_deg)
        b2 = np.deg2rad(ori.tilt_tangent_2_deg)
        psi = np.deg2rad(ori.axial_roll_deg)
        dp = np.stack(
            [s[:, 0] * pos.tangent_1_m, s[:, 1] * pos.tangent_2_m, s[:, 2] * pos.normal_m],
            axis=1,
        )
        dw = np.stack([s[:, 3] * b1, s[:, 4] * b2, s[:, 5] * psi], axis=1)
        half = np.concatenate([dp, dw], axis=1)
        paired = np.empty((half.shape[0] * 2, 6), dtype=np.float64)
        paired[0::2] = half
        paired[1::2] = -half
        return paired[:num_samples]

    u = sobol_unit_cube(num_samples, 6, seed=seed)
    s = 2.0 * u - 1.0
    dp = np.stack(
        [
            s[:, 0] * position.tangent_1_m,
            s[:, 1] * position.tangent_2_m,
            s[:, 2] * position.normal_m,
        ],
        axis=1,
    )
    b1 = np.deg2rad(orientation.tilt_tangent_1_deg)
    b2 = np.deg2rad(orientation.tilt_tangent_2_deg)
    psi = np.deg2rad(orientation.axial_roll_deg)
    dw = np.stack([s[:, 3] * b1, s[:, 4] * b2, s[:, 5] * psi], axis=1)
    return np.concatenate([dp, dw], axis=1).astype(np.float64)


def softmin(values: np.ndarray, tau: float, weights: np.ndarray | None = None) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    w = np.ones_like(v) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)
    tau = max(float(tau), 1e-8)
    m = v.min()
    return float(-tau * np.log(np.sum(w * np.exp(-(v - m) / tau)) + 1e-12) + m)


def coverage_from_m(m: np.ndarray, m_min: float = 0.0, tau_c: float = 0.5) -> float:
    z = (np.asarray(m, dtype=np.float64) - m_min) / max(tau_c, 1e-8)
    return float(np.mean(1.0 / (1.0 + np.exp(-z))))


def aggregate_mq(
    m: np.ndarray,
    q: np.ndarray,
    *,
    tau: float = 0.5,
    lambda_q: float = 0.5,
    tau_m_cost: float = 1.0,
    weights: np.ndarray | None = None,
) -> RegionScore:
    """m_robust = softmin(m); q_region = mean(q); score = -softplus(-m_r/τ)+λq."""
    mv = np.asarray(m, dtype=np.float64).reshape(-1)
    qv = np.asarray(q, dtype=np.float64).reshape(-1)
    w = np.ones_like(mv) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / (w.sum() + 1e-12)
    m_rob = softmin(mv, tau, w)
    q_reg = float(np.sum(w * qv))
    # softplus(-m/τ) = log(1+exp(-m/τ))
    cost_m = float(np.logaddexp(0.0, -m_rob / max(tau_m_cost, 1e-6)))
    score = -cost_m + float(lambda_q) * q_reg
    return RegionScore(
        score=score,
        m_robust=m_rob,
        q_region=q_reg,
        mean_score=float(np.sum(w * mv)),
        softmin_score=m_rob,
        coverage=coverage_from_m(mv),
        min_score=float(mv.min()),
        num_samples=int(mv.size),
    )


def aggregate_mean_softmin(
    values: np.ndarray,
    *,
    lam: float = 0.6,
    tau: float = 0.1,
    d_min: float = 0.3,
    tau_c: float = 0.05,
    weights: np.ndarray | None = None,
) -> RegionScore:
    """Legacy scalar aggregator (treat values as m)."""
    return aggregate_mq(values, np.zeros_like(values), tau=tau, lambda_q=0.0, tau_m_cost=1.0, weights=weights)


def perturb_center_poses(T_mu: np.ndarray, xi: np.ndarray) -> np.ndarray:
    T_mu = np.asarray(T_mu, dtype=np.float64).reshape(4, 4)
    out = np.empty((xi.shape[0], 4, 4), dtype=np.float64)
    for i, x in enumerate(xi):
        out[i] = se3_mul(T_mu, se3_exp(x))
    return out


def region_score_a(
    neural_ird,
    *,
    delta_T_center: np.ndarray | None = None,
    T_mu: np.ndarray | None = None,
    T_base: np.ndarray | None = None,
    position_extent: tuple[float, float, float] | PositionExtent = (0.02, 0.01, 0.002),
    orientation_extent: tuple[float, float, float] | OrientationExtent = (8.0, 5.0, 3.0),
    aggregation: str = "softmin_m_mean_q",
    num_samples: int = 32,
    lam: float = 0.6,
    tau: float = 0.5,
    d_min: float = 0.3,
    lambda_q: float = 0.5,
    seed: int = 0,
) -> RegionScore:
    if isinstance(position_extent, tuple):
        position_extent = PositionExtent(*position_extent)
    if isinstance(orientation_extent, tuple):
        orientation_extent = OrientationExtent(*orientation_extent)

    xi = sample_anisotropic_xi(position_extent, orientation_extent, num_samples, seed=seed)

    if T_mu is not None:
        T_base = np.eye(4) if T_base is None else np.asarray(T_base, dtype=np.float64)
        Ts = perturb_center_poses(T_mu, xi)
        from ird_playground.probe.se3 import invert_T

        dTs = np.stack([invert_T(Tk) @ T_base for Tk in Ts], axis=0)
    elif delta_T_center is not None:
        dT0 = np.asarray(delta_T_center, dtype=np.float64).reshape(4, 4)
        dTs = np.stack([se3_mul(dT0, se3_exp(x)) for x in xi], axis=0)
    else:
        raise ValueError("provide delta_T_center or T_mu")

    out = neural_ird.score_batch_delta_T(dTs)
    m = out.get("m", out.get("d"))
    q = out.get("q", out.get("q_comfort", np.zeros_like(m)))
    if aggregation in ("softmin_m_mean_q", "mean_softmin"):
        return aggregate_mq(m, q, tau=tau, lambda_q=lambda_q)
    raise ValueError(f"unsupported aggregation {aggregation!r}")
```

### `region/__init__.py`

```python
"""Region package: legacy SE(3) aggregate + Sprint-0A local ellipsoid/cone."""

from ird_playground.region.aggregate import (
    OrientationExtent,
    PositionExtent,
    RegionScore,
    aggregate_mean_softmin,
    aggregate_mq,
    region_score_a,
    sample_anisotropic_xi,
)
from ird_playground.region.local_region import (
    LocalExtent,
    RegionAggConfig,
    local_region_cost,
    make_joint_sobol_ellipsoid_cone,
    robust_region_cost,
)

__all__ = [
    "LocalExtent",
    "OrientationExtent",
    "PositionExtent",
    "RegionAggConfig",
    "RegionScore",
    "aggregate_mean_softmin",
    "aggregate_mq",
    "local_region_cost",
    "make_joint_sobol_ellipsoid_cone",
    "region_score_a",
    "robust_region_cost",
    "sample_anisotropic_xi",
]
```

### `traj/manifold.py`

```python
"""GT vessel → skin manifold: λ ↦ T_tcp (deterministic; not NLP free vars).

Optimization variables are only (λ, r). Probe pose is produced by this map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _complete_frame_n_t(n: np.ndarray, t: np.ndarray) -> np.ndarray:
    """R = [b, t_hat, n_hat] with n ≈ skin normal (probe +Z), t ≈ vessel tangent."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    t = t - n * float(np.dot(t, n))
    tn = np.linalg.norm(t)
    if tn < 1e-9:
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t = np.cross(a, n)
        t = t / (np.linalg.norm(t) + 1e-12)
    else:
        t = t / tn
    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1e-12)
    t = np.cross(n, b)
    return np.stack([b, t, n], axis=1)


@dataclass
class VesselSkinSample:
    p: np.ndarray  # (3,) skin contact
    n: np.ndarray  # (3,) skin outward normal → probe +Z
    t: np.ndarray  # (3,) vessel tangent (in-plane)
    R: np.ndarray  # (3,3)
    T: np.ndarray  # (4,4)


class VesselSkinManifold:
    """Abstract λ ∈ ℝ → T_tcp. Concrete maps may use LBS / fiber / lookup tables."""

    def sample(self, lam: float) -> VesselSkinSample:
        raise NotImplementedError

    def sample_batch(self, lams: np.ndarray) -> list[VesselSkinSample]:
        return [self.sample(float(x)) for x in np.asarray(lams, dtype=np.float64).reshape(-1)]

    def T_tcp(self, lam: float) -> np.ndarray:
        return self.sample(lam).T

    def T_tcp_batch(self, lams: np.ndarray) -> np.ndarray:
        return np.stack([s.T for s in self.sample_batch(lams)], axis=0)


class SyntheticVesselSkinManifold(VesselSkinManifold):
    """Smooth synthetic skin curve for AD / P1 tests (no patient mesh required).

    λ is arc-length-like in meters along a planar arc in the xz plane, with a
    mild y undulation. Normal points roughly +Y (patient facing robot).
    """

    def __init__(
        self,
        *,
        center: tuple[float, float, float] = (0.35, 0.0, 0.25),
        radius_m: float = 0.12,
        length_m: float = 0.40,
        y_amp_m: float = 0.02,
    ) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.radius_m = float(radius_m)
        self.length_m = float(length_m)
        self.y_amp_m = float(y_amp_m)

    def sample(self, lam: float) -> VesselSkinSample:
        # Map λ∈ℝ → angle along arc; clamp softly via tanh for stability
        s = float(lam)
        # centerline in xz, bulge in −y (skin facing robot at +y? use +y normal)
        ang = (s / max(self.length_m, 1e-6)) * np.pi  # ~0…π over length
        cx, cy, cz = self.center
        p = np.array(
            [
                cx + self.radius_m * np.sin(ang),
                cy + self.y_amp_m * np.sin(2.0 * ang),
                cz + self.radius_m * (1.0 - np.cos(ang)),
            ],
            dtype=np.float64,
        )
        # analytic tangent ds
        dang = np.pi / max(self.length_m, 1e-6)
        t = np.array(
            [
                self.radius_m * np.cos(ang) * dang,
                self.y_amp_m * 2.0 * np.cos(2.0 * ang) * dang,
                self.radius_m * np.sin(ang) * dang,
            ],
            dtype=np.float64,
        )
        # outward normal approx: from arc center toward point in xz, + small y
        n = np.array(
            [np.sin(ang), 0.85, 1.0 - np.cos(ang)],
            dtype=np.float64,
        )
        n = n / (np.linalg.norm(n) + 1e-12)
        R = _complete_frame_n_t(n, t)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = p
        return VesselSkinSample(p=p, n=n, t=t / (np.linalg.norm(t) + 1e-12), R=R, T=T)

    def sample_torch(
        self,
        lam: "torch.Tensor",
        *,
        dtype=None,
        device=None,
    ) -> "torch.Tensor":
        """Differentiable λ → T_tcp (4,4) or (N,4,4) for the synthetic map."""
        if torch is None:
            raise ImportError("torch required")
        dtype = dtype or torch.float32
        device = device or (lam.device if hasattr(lam, "device") else "cpu")
        lam = lam.to(device=device, dtype=dtype)
        single = lam.ndim == 0
        if single:
            lam = lam.reshape(1)
        L = max(self.length_m, 1e-6)
        ang = (lam / L) * np.pi
        cx, cy, cz = [float(x) for x in self.center]
        r = self.radius_m
        ya = self.y_amp_m
        px = cx + r * torch.sin(ang)
        py = cy + ya * torch.sin(2.0 * ang)
        pz = cz + r * (1.0 - torch.cos(ang))
        p = torch.stack([px, py, pz], dim=-1)  # (N,3)

        dang = np.pi / L
        tx = r * torch.cos(ang) * dang
        ty = ya * 2.0 * torch.cos(2.0 * ang) * dang
        tz = r * torch.sin(ang) * dang
        t = torch.stack([tx, ty, tz], dim=-1)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        nx = torch.sin(ang)
        ny = torch.full_like(ang, 0.85)
        nz = 1.0 - torch.cos(ang)
        n = torch.stack([nx, ny, nz], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # Gram–Schmidt: R = [b, t_hat, n]
        t = t - n * (t * n).sum(dim=-1, keepdim=True)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        b = torch.cross(t, n, dim=-1)
        b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        t = torch.cross(n, b, dim=-1)
        R = torch.stack([b, t, n], dim=-1)  # (N,3,3)

        N = lam.shape[0]
        T = torch.eye(4, dtype=dtype, device=device).expand(N, 4, 4).clone()
        T[:, :3, :3] = R
        T[:, :3, 3] = p
        return T[0] if single else T
```

### `traj/p1_optimize.py`

```python
"""P1 offline optimizer: Bernstein (c_λ, c_r) + local ellipsoid/cone Region A."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def bernstein_basis(s: "torch.Tensor", n_ctrl: int) -> "torch.Tensor":
    if torch is None:
        raise ImportError("torch required")
    s = s.reshape(-1).clamp(0.0, 1.0)
    deg = int(n_ctrl) - 1
    return torch.stack(
        [float(comb(deg, i)) * (s**i) * ((1.0 - s) ** (deg - i)) for i in range(deg + 1)],
        dim=1,
    )


@dataclass
class P1Config:
    n_ctrl: int = 8
    n_knots_eval: int = 48
    region_k: int = 32
    sobol_seed: int = 0
    w_ird: float = 1.0
    w_track: float = 0.5
    w_smooth_lam: float = 0.1
    w_smooth_rail: float = 0.1
    w_d2_lam: float = 0.05
    w_d2_rail: float = 0.05
    steps: int = 60
    lr: float = 5e-3


def optimize_p1_lambda_rail(
    neural_ird,
    manifold,
    *,
    lambda_ref: np.ndarray | None = None,
    rail_ref: np.ndarray | None = None,
    cfg: P1Config | None = None,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
    extent=None,
    agg=None,
) -> dict:
    if torch is None:
        raise ImportError("torch required")
    from ird_playground.region.local_region import (
        local_region_cost,
        make_joint_sobol_ellipsoid_cone,
    )

    cfg = cfg or P1Config()
    device = neural_ird.device
    n_ctrl = int(cfg.n_ctrl)
    m = int(cfg.n_knots_eval)
    length = float(getattr(manifold, "length_m", 0.40))

    if lambda_ref is None:
        lambda_ref = np.linspace(0.05 * length, 0.95 * length, m)
    else:
        lambda_ref = np.asarray(lambda_ref, dtype=np.float64).reshape(-1)
        m = int(lambda_ref.size)
    if rail_ref is None:
        rail_ref = np.zeros(m, dtype=np.float64)
    else:
        rail_ref = np.asarray(rail_ref, dtype=np.float64).reshape(-1)

    s = torch.linspace(0.0, 1.0, m, device=device, dtype=torch.float32)
    B = bernstein_basis(s, n_ctrl)
    Br = B.detach().cpu().numpy()
    c_lam = torch.tensor(
        np.linalg.lstsq(Br, lambda_ref, rcond=None)[0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    c_rail = torch.tensor(
        np.linalg.lstsq(Br, rail_ref, rcond=None)[0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    lam_ref_t = torch.as_tensor(lambda_ref, dtype=torch.float32, device=device)

    # Fixed joint Sobol — never resample inside the loop
    local_eps = make_joint_sobol_ellipsoid_cone(
        cfg.region_k, extent=extent, seed=cfg.sobol_seed, device=device
    )

    opt = torch.optim.Adam([c_lam, c_rail], lr=cfg.lr)
    history: list[float] = []
    neural_ird.model.eval()
    for _ in range(int(cfg.steps)):
        opt.zero_grad()
        lam = B @ c_lam
        rail = B @ c_rail
        reg = local_region_cost(
            neural_ird,
            lam,
            rail,
            manifold,
            local_eps=local_eps,
            extent=extent,
            agg=agg,
            T_world_rail=T_world_rail,
            T_rail_base0=T_rail_base0,
        )
        d1_l, d1_r = lam[1:] - lam[:-1], rail[1:] - rail[:-1]
        d2_l, d2_r = d1_l[1:] - d1_l[:-1], d1_r[1:] - d1_r[:-1]
        loss = (
            cfg.w_ird * reg["cost"]
            + cfg.w_track * ((lam - lam_ref_t) ** 2).mean()
            + cfg.w_smooth_lam * (d1_l**2).mean()
            + cfg.w_smooth_rail * (d1_r**2).mean()
            + cfg.w_d2_lam * (d2_l**2).mean()
            + cfg.w_d2_rail * (d2_r**2).mean()
        )
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        lam_f = (B @ c_lam).detach().cpu().numpy()
        rail_f = (B @ c_rail).detach().cpu().numpy()
        cov = float(
            local_region_cost(
                neural_ird,
                torch.as_tensor(lam_f, device=device),
                torch.as_tensor(rail_f, device=device),
                manifold,
                local_eps=local_eps,
                extent=extent,
                agg=agg,
                T_world_rail=T_world_rail,
                T_rail_base0=T_rail_base0,
            )["p_cov"]
            .mean()
            .cpu()
        )
    return {
        "lambda": lam_f,
        "rail": rail_f,
        "c_lambda": c_lam.detach().cpu().numpy(),
        "c_rail": c_rail.detach().cpu().numpy(),
        "history": history,
        "final_loss": history[-1] if history else float("nan"),
        "final_coverage": cov,
    }
```

### `probe/se3.py`

```python
"""SE(3) helpers: ΔT → natural (p,u) 5-DoF features, Exp map."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def mat4_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def delta_T_tcp_inv_base(T_base_tcp: np.ndarray) -> np.ndarray:
    """ΔT = T_tcp^{-1} T_base = (T_base_tcp)^{-1} when T_base = I in arm-base frame."""
    return invert_T(T_base_tcp)


def rot6d_from_R(R: np.ndarray) -> np.ndarray:
    """Zhou et al. continuous 6D rotation: first two columns of R."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0)


def features_from_delta_T(delta_T: np.ndarray) -> np.ndarray:
    """(6,) = natural 5-DoF [p_base,tcp, u_base] recovered from ΔT.

    ΔT = T_tcp^{-1} T_base. With T_base=I:
      R_base,tcp = R_Δᵀ
      p_base,tcp = −R_Δᵀ t_Δ
      u_base = R_base,tcp @ e_z = R_Δᵀ[:,2] = R_Δ[2,:]ᵀ wait: (R_Δᵀ)[:,2] = R_Δ[2,:].T
    """
    T = np.asarray(delta_T, dtype=np.float64).reshape(4, 4)
    R_delta = T[:3, :3]
    t_delta = T[:3, 3]
    R_base_tcp = R_delta.T
    p = -(R_base_tcp @ t_delta)
    u = R_base_tcp[:, 2].copy()
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.concatenate([p, u], axis=0).astype(np.float64)


def batch_features_from_delta_T(delta_Ts: np.ndarray) -> np.ndarray:
    """(N,6) from (N,4,4)."""
    Ts = np.asarray(delta_Ts, dtype=np.float64)
    if Ts.ndim == 2:
        return features_from_delta_T(Ts)[None, :]
    out = np.empty((Ts.shape[0], 6), dtype=np.float64)
    for i, T in enumerate(Ts):
        out[i] = features_from_delta_T(T)
    return out


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """ξ = [δp(3), δω(3)] → SE(3) via scipy Rotation (axis-angle)."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    dp, dw = xi[:3], xi[3:]
    R = Rotation.from_rotvec(dw).as_matrix()
    return mat4_from_Rt(R, dp)


def se3_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float64) @ np.asarray(B, dtype=np.float64)


def complete_frame_from_tool_axis(tool_axis: np.ndarray) -> np.ndarray:
    """Build a rotation whose +Z is ``tool_axis`` (Zacharias tool axis = TCP +Z)."""
    z = np.asarray(tool_axis, dtype=np.float64).reshape(3)
    z = z / (np.linalg.norm(z) + 1e-12)
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
```

### `cli/train.py`

```python
"""Train generic Neural IRD point field (hyperparams from YAML)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.train import load_train_config, train_point_field


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Training YAML (configs/train_config.yaml)",
    )
    ap.add_argument(
        "--gt-npz",
        type=Path,
        default=None,
        help="Optional override of data.gt_npz",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional override of io.checkpoint")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing train config: {cfg_path}")

    cfg = load_train_config(cfg_path, root=root)
    if args.gt_npz is not None:
        cfg.gt_npz = str(args.gt_npz if args.gt_npz.is_absolute() else root / args.gt_npz)
    if args.checkpoint is not None:
        cfg.checkpoint = str(args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint)

    result = train_point_field(cfg)
    report = Path(cfg.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["val_metrics"], indent=2))
    print(f"checkpoint → {result['checkpoint']}")
    print(f"report → {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `cli/build_ird_gt.py`

```python
"""Export IRD GT NPZ from a capability map (sampling from YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import (
    IrdGtConfig,
    assert_gt_contract,
    export_ird_gt_from_capability_map,
    save_ird_gt,
)
from ird_playground.ird.capability_io import load_capability_map_dir
from ird_playground.ird.map_loader import resolve_map_dir


def load_ird_gt_config(path: Path, *, root: Path) -> tuple[Path, Path, IrdGtConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    samp = dict(raw.get("sampling") or {})
    map_dir = Path(raw.get("map_dir", ""))
    out = Path(raw.get("out", "data/ird/gt_samples.npz"))
    if not map_dir.is_absolute():
        map_dir = (root / map_dir).resolve()
    if not out.is_absolute():
        out = root / out

    n_int = int(samp.get("n_interior", 700_000))
    n_bnd = int(samp.get("n_boundary", 800_000))
    n_ext = int(samp.get("n_exterior", 500_000))

    cfg = IrdGtConfig(
        n_interior=n_int,
        n_boundary=n_bnd,
        n_exterior=n_ext,
        n_jitter=int(samp.get("n_jitter", 400_000)),
        max_orients_per_voxel=int(samp.get("max_orients_per_voxel", 28)),
        hard_negative_frac=float(samp.get("hard_negative_frac", 0.45)),
        hard_negative_radius_m=float(samp.get("hard_negative_radius_m", 0.06)),
        sigma_p_m=float(samp.get("sigma_p_m", 0.03)),
        sigma_r_deg=float(samp.get("sigma_r_deg", 10.0)),
        m_clip=float(samp.get("m_clip", 3.0)),
        m_eps=float(samp.get("m_eps", 0.05)),
        bbox_margin_m=float(samp.get("bbox_margin_m", 0.20)),
        comfort_from=str(samp.get("comfort_from", "auto")),
        k_candidates=int(samp.get("k_candidates", 4)),
        seed=int(samp.get("seed", 0)),
        orient_knn=int(samp.get("orient_knn", 7)),
        soft_tau=float(samp.get("soft_tau", 0.05)),
        unknown_soft_max=float(samp.get("unknown_soft_max", 0.25)),
        trusted_neg_soft_max=float(samp.get("trusted_neg_soft_max", 0.0)),
        min_positive_support=int(samp.get("min_positive_support", 3)),
        min_trusted_face_pairs=int(samp.get("min_trusted_face_pairs", 5000)),
    )
    return map_dir, out, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/ird_gt_config.yaml"))
    ap.add_argument("--map", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    map_dir, out, cfg = load_ird_gt_config(cfg_path, root=root)
    if args.map is not None:
        map_dir = resolve_map_dir(args.map if args.map.is_absolute() else root / args.map)
    else:
        map_dir = resolve_map_dir(map_dir)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else root / args.out

    cm = load_capability_map_dir(map_dir, mmap=True)
    arrays = export_ird_gt_from_capability_map(cm, cfg)
    assert_gt_contract(arrays)
    save_ird_gt(
        out,
        arrays,
        meta={
            "map_dir": str(map_dir),
            "config": str(cfg_path),
            "n_interior": cfg.n_interior,
            "n_boundary": cfg.n_boundary,
            "n_exterior": cfg.n_exterior,
            "n_jitter": cfg.n_jitter,
            "sigma_p_m": cfg.sigma_p_m,
            "m_clip": cfg.m_clip,
            "feature_dim": 6,
            "seed": cfg.seed,
            "n_total": int(arrays["features"].shape[0]),
            "contract": "MC-hit=pos; C+>=min & C-==0 trusted faces; no soft_tau fallback; natural(p,u)",
            "feature_kind": "natural_pu",
            "label_kind": "stable_support_v6",
        },
    )
    print(f"wrote {out}  N={arrays['features'].shape[0]} dim={arrays['features'].shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `cli/viz_ird.py`

```python
"""Visualize **global IRD** (Vahrenkamp): base poses in TCP frame.

Default is Inverse Reachability Distribution — NOT Zacharias forward capability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.viz.viz_style import PROBE_COMPARE_BAR_MAX, PROBE_COMPARE_CLIM, PROBE_COMPARE_N_LEVELS


def _resolve(root: Path, p: Path) -> Path:
    return p if p.is_absolute() else root / p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["ird", "ird_gt", "capability", "scatter"],
        default="ird",
        help="ird=invert capability→global IRD (default); ird_gt=from GT npz; "
        "capability=Zacharias forward (legacy); scatter=debug cloud",
    )
    ap.add_argument("--map-dir", type=Path, default=None)
    ap.add_argument("--gt-npz", type=Path, default=Path("data/ird/gt_samples.npz"))
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional neural IRD overlay grid")
    ap.add_argument("--out", type=Path, default=Path("data/reports/global_ird.png"))
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--d-min", type=float, default=0.02)
    ap.add_argument(
        "--clim",
        type=float,
        nargs=2,
        default=list(PROBE_COMPARE_CLIM),
        metavar=("LO", "HI"),
        help="Colour limits in fraction units (default shared probe compare 0 0.18)",
    )
    ap.add_argument(
        "--clim-auto",
        action="store_true",
        help="Stretch colour bar to [d_min, max(values)] for this figure",
    )
    ap.add_argument(
        "--clim-abs",
        action="store_true",
        help="Absolute 0..1 scale (0–100%% bar, cross-map compare)",
    )
    ap.add_argument(
        "--step-m",
        type=float,
        default=None,
        help="IRD lattice step (default: map grid step_m)",
    )
    ap.add_argument(
        "--max-orients",
        type=int,
        default=None,
        help="Cap orientations per voxel when inverting (default: all reachable)",
    )
    ap.add_argument(
        "--legacy-cloud",
        action="store_true",
        help="Use old sparse point cloud (subsample + non-lattice voxelize)",
    )
    ap.add_argument("--max-voxels", type=int, default=12_000, help="Cap map voxels (legacy-cloud only)")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    out = _resolve(root, args.out)
    if args.clim_abs:
        clim = (0.0, 1.0)
    else:
        clim = (float(args.clim[0]), float(args.clim[1]))
    clim_auto = bool(args.clim_auto)
    if args.mode == "scatter":
        from ird_playground.ird.export_gt import load_ird_gt
        from ird_playground.viz.ird_compare import features_to_xyz, render_ird_comparison

        arrays = load_ird_gt(_resolve(root, args.gt_npz))
        path = render_ird_comparison(
            xyz=features_to_xyz(arrays["features"]),
            gt=arrays["d"],
            pred=None,
            out_path=out if out.suffix else out.with_suffix(".png"),
            value_name="IRD d (base in TCP)",
        )
        print(f"wrote {path}")
        return 0

    if args.mode == "capability":
        from ird_playground.viz.sphere_ird import render_ird_spheres

        map_dir = _map_dir(root, args)
        path = render_ird_spheres(
            map_dir, out if out.suffix else out.with_suffix(".png"),
            channel="reach_D", d_min=args.d_min, clim=clim,
        )
        print(f"wrote [capability-forward] {path}")
        return 0

    from ird_playground.viz.global_ird import (
        build_ird_lattice_from_capability,
        build_ird_points_from_gt_npz,
        render_global_ird,
        render_global_ird_from_capability,
        voxelize_max,
    )

    xyz = None
    q = None
    if args.mode == "ird_gt":
        xyz, q = build_ird_points_from_gt_npz(_resolve(root, args.gt_npz))
        title = "Global IRD"
        step_m = float(args.step_m) if args.step_m is not None else 0.03
        xyz, q = voxelize_max(xyz, q, step_m=step_m, lattice_centers=True)
        path = render_global_ird(
            xyz,
            q,
            out if out.suffix else out.with_suffix(".png"),
            d_min=args.d_min,
            clim=clim,
            clim_auto=clim_auto,
            title=title,
            step_m=step_m,
            n_color_levels=PROBE_COMPARE_N_LEVELS,
            bar_max=PROBE_COMPARE_BAR_MAX,
        )
    elif args.legacy_cloud:
        from ird_playground.ird.capability_io import load_capability_map_dir

        map_dir = _map_dir(root, args)
        cm = load_capability_map_dir(map_dir)
        import numpy as np

        order = np.argsort(-cm.d_value)[: int(args.max_voxels)]
        class _Sub:
            pass

        sub = _Sub()
        sub.orientations = cm.orientations
        sub.roll = cm.roll
        sub.bitmask = cm.bitmask[order]
        sub.d_value = cm.d_value[order]
        sub.voxel_ids = cm.voxel_ids[order]
        sub.grid = cm.grid
        from ird_playground.viz.global_ird import build_ird_points_from_capability

        max_orients = 6 if args.max_orients is None else int(args.max_orients)
        xyz, q = build_ird_points_from_capability(sub, max_orients_per_voxel=max_orients)
        step_m = float(args.step_m) if args.step_m is not None else 0.05
        xyz, q = voxelize_max(xyz, q, step_m=step_m)
        title = "Global IRD"
        path = render_global_ird(
            xyz,
            q,
            out if out.suffix else out.with_suffix(".png"),
            d_min=args.d_min,
            clim=clim,
            clim_auto=clim_auto,
            title=title,
            sphere_radius_m=step_m * 0.55,
            n_color_levels=PROBE_COMPARE_N_LEVELS,
            bar_max=PROBE_COMPARE_BAR_MAX,
        )
    else:
        from ird_playground.ird.capability_io import load_capability_map_dir

        map_dir = _map_dir(root, args)
        cm = load_capability_map_dir(map_dir)
        title = "Global IRD"
        xyz, q = build_ird_lattice_from_capability(
            cm,
            step_m=args.step_m,
            max_orients_per_voxel=args.max_orients,
        )
        path = render_global_ird_from_capability(
            cm,
            out if out.suffix else out.with_suffix(".png"),
            d_min=args.d_min,
            clim=clim,
            clim_auto=clim_auto,
            title=title,
            step_m=args.step_m,
            max_orients_per_voxel=args.max_orients,
            n_color_levels=PROBE_COMPARE_N_LEVELS,
            bar_max=PROBE_COMPARE_BAR_MAX,
        )
    if xyz is not None and q is not None:
        print(
            f"wrote {path}  n_cells={xyz.shape[0]}  mean={float(q.mean()):.4f}  "
            f"max={float(q.max()):.4f}  clim={clim if not clim_auto else 'auto'}"
        )
    else:
        print(f"wrote {path}  clim={clim if not clim_auto else 'auto'}")

    if args.checkpoint is not None and xyz is not None:
        from ird_playground.neural.model import NeuralIRD
        from ird_playground.viz.global_ird import predict_ird_grid, render_global_ird as _r

        net = NeuralIRD.load(_resolve(root, args.checkpoint), device=args.device)
        lo = xyz.min(axis=0) - 0.05
        hi = xyz.max(axis=0) + 0.05
        step_m = float(args.step_m) if args.step_m is not None else 0.05
        gxyz, gd = predict_ird_grid(
            net,
            bbox=((float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1])), (float(lo[2]), float(hi[2]))),
            step_m=step_m,
            n_orients=4,
        )
        pred_out = out.with_name(out.stem + "_pred" + out.suffix) if out.suffix else Path(str(out) + "_pred.png")
        gxyz, gd = voxelize_max(gxyz, gd, step_m=step_m, lattice_centers=True)
        _r(
            gxyz,
            gd,
            pred_out,
            d_min=args.d_min,
            clim=clim,
            title="Neural IRD",
            step_m=step_m,
            n_color_levels=PROBE_COMPARE_N_LEVELS,
            bar_max=PROBE_COMPARE_BAR_MAX,
        )
        print(f"wrote {pred_out}")
    return 0


def _map_dir(root: Path, args) -> Path:
    if args.map_dir is not None:
        p = Path(args.map_dir)
        return p if p.is_absolute() else root / p
    meta = _resolve(root, args.gt_npz).with_suffix(".yaml")
    if meta.is_file():
        return Path(yaml.safe_load(meta.read_text(encoding="utf-8"))["map_dir"])
    raise SystemExit("Provide --map-dir or gt npz sibling .yaml with map_dir")


if __name__ == "__main__":
    raise SystemExit(main())
```

### `viz/viz_style.py`

```python
"""Shared figure style for capability / global-IRD cross-probe compare."""

from __future__ import annotations

# Zacharias D fraction → colour bar ticks 0 … BAR_MAX (same units as capability figures).
PROBE_COMPARE_CLIM: tuple[float, float] = (0.0, 0.18)
PROBE_COMPARE_BAR_MAX: float = 18.0
PROBE_COMPARE_N_LEVELS: int = 8
PROBE_COMPARE_D_MIN: float = 0.02
SPHERE_RADIUS_FACTOR: float = 0.48
```

### `tests/test_core.py`

```python
"""Unit tests for probe, region A, and neural point field (synthetic GT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ird_playground.probe.se3 import (
    complete_frame_from_tool_axis,
    delta_T_tcp_inv_base,
    features_from_delta_T,
    mat4_from_Rt,
    se3_exp,
)
from ird_playground.probe.transform import default_ultrasound_probe, load_probe_yaml
from ird_playground.region.aggregate import (
    OrientationExtent,
    PositionExtent,
    aggregate_mean_softmin,
    sample_anisotropic_xi,
    softmin,
)


def test_default_probe_composition():
    p = default_ultrasound_probe()
    assert np.allclose(p.translation_m, [0.05, 0.0, 0.07], atol=1e-9)
    R = p.rotation_matrix()
    assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)


def test_probe_yaml_roundtrip(tmp_path):
    src = Path(__file__).resolve().parents[1] / "configs" / "probe_default.yaml"
    if not src.exists():
        pytest.skip("configs/probe_default.yaml missing")
    p = load_probe_yaml(src)
    assert p.name.startswith("ultrasound")


def test_delta_T_features_dim():
    R = complete_frame_from_tool_axis([0, 0, 1])
    T = mat4_from_Rt(R, [0.3, 0.1, 0.2])
    dT = delta_T_tcp_inv_base(T)
    f = features_from_delta_T(dT)
    assert f.shape == (6,)
    assert np.allclose(f[:3], [0.3, 0.1, 0.2], atol=1e-6)
    assert np.allclose(f[3:], R[:, 2], atol=1e-6)


def test_softmin_approaches_min():
    v = np.array([0.9, 0.2, 0.8])
    s = softmin(v, tau=1e-4)
    assert abs(s - 0.2) < 1e-3


def test_region_aggregate_not_mean_only():
    from ird_playground.region.aggregate import aggregate_mq

    v = np.array([1.0, 1.0, 0.0])
    q = np.array([0.8, 0.7, 0.1])
    rs = aggregate_mq(v, q, tau=0.05, lambda_q=0.5)
    assert rs.mean_score > rs.softmin_score
    assert rs.m_robust == rs.softmin_score
    assert abs(rs.q_region - float(q.mean())) < 1e-6
    assert rs.min_score == 0.0
    rs2 = aggregate_mean_softmin(v, lam=0.6, tau=0.05)
    assert rs2.softmin_score < rs2.mean_score


def test_anisotropic_extents_respect_bounds():
    xi = sample_anisotropic_xi(
        PositionExtent(0.02, 0.01, 0.002),
        OrientationExtent(8.0, 5.0, 3.0),
        64,
        seed=0,
    )
    assert xi.shape == (64, 6)
    assert np.max(np.abs(xi[:, 0])) <= 0.02 + 1e-9
    assert np.max(np.abs(xi[:, 2])) <= 0.002 + 1e-9


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_optimization_cost_uses_reach_logit():
    import torch
    from ird_playground.neural.cost import optimization_cost

    logit = torch.tensor([2.0, -2.0])
    margin = torch.tensor([0.5, 0.5])
    q = torch.tensor([0.5, 0.5])
    c = optimization_cost(logit, margin, q)
    assert c[0] < c[1]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_lambda_rail_ad_and_robust_region():
    import torch
    from ird_playground.ird.query_base import cost_vs_lambda_rail_torch, lambda_rail_grad_ad_fd
    from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint
    from ird_playground.region.local_region import local_region_cost, make_joint_sobol_ellipsoid_cone
    from ird_playground.traj.manifold import SyntheticVesselSkinManifold

    model = NeuralIRDPoint(hidden=64, depth=2, num_freqs_u=2, use_physical_pe=False)
    net = NeuralIRD(model, device="cpu")
    man = SyntheticVesselSkinManifold()

    g = lambda_rail_grad_ad_fd(net, man, n=6, seed=0)
    assert g["lambda_ad_fd_rel"] < 0.35
    assert g["rail_ad_fd_rel"] < 0.35

    lam = torch.tensor(0.15, requires_grad=True)
    rail = torch.tensor(0.0, requires_grad=True)
    cost_vs_lambda_rail_torch(net, man, lam, rail)["cost"].backward()
    assert lam.grad is not None and abs(float(lam.grad)) + abs(float(rail.grad)) > 0

    eps = make_joint_sobol_ellipsoid_cone(32, seed=0, device="cpu")
    assert eps.shape == (32, 5)
    assert torch.allclose(eps[0], torch.zeros(5))  # center included
    N = 8
    lam_c = torch.linspace(0.05, 0.35, N, requires_grad=True)
    rail_c = torch.zeros(N, requires_grad=True)
    out = local_region_cost(net, lam_c, rail_c, man, local_eps=eps)
    assert out["point_cost"].shape == (N, 32)
    out["cost"].backward()
    assert float(lam_c.grad.abs().sum()) > 0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_p1_smoke():
    from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint
    from ird_playground.traj.manifold import SyntheticVesselSkinManifold
    from ird_playground.traj.p1_optimize import P1Config, optimize_p1_lambda_rail

    net = NeuralIRD(
        NeuralIRDPoint(hidden=64, depth=2, num_freqs_u=2, use_physical_pe=False),
        device="cpu",
    )
    man = SyntheticVesselSkinManifold()
    res = optimize_p1_lambda_rail(
        net,
        man,
        cfg=P1Config(n_ctrl=5, n_knots_eval=12, region_k=16, steps=5, lr=1e-2),
    )
    assert len(res["history"]) == 5
    assert np.isfinite(res["final_loss"])
    assert res["lambda"].shape == (12,)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_synthetic_and_region_a(tmp_path):
    from ird_playground.neural.model import NeuralIRD
    from ird_playground.neural.train import TrainConfig, differentiability_smoke, train_point_field
    from ird_playground.region.aggregate import region_score_a

    ckpt = tmp_path / "m.pt"
    cfg = TrainConfig(
        gt_npz=None,
        synthetic_n=4096,
        epochs=25,
        batch_size=256,
        hidden=128,
        depth=3,
        num_freqs=4,
        num_freqs_u=2,
        use_physical_pe=False,
        warmup_steps=0,
        lr=3e-3,
        hardneg_every=0,
        checkpoint=str(ckpt),
        seed=0,
        num_workers=0,
        lambda_cls=1.0,
        lambda_margin=0.0,
        lambda_q=0.0,
        lambda_local=0.0,
        wandb_enable=False,
    )
    result = train_point_field(cfg)
    assert ckpt.exists()
    assert result["val_metrics"]["boundary_iou"] > 0.5
    assert result["val_metrics"]["mae_m"] < 1.5

    net = NeuralIRD.load(ckpt)
    assert differentiability_smoke(net) >= 0.0

    T_mu = mat4_from_Rt(np.eye(3), np.array([0.2, 0.0, 0.1]))
    rs = region_score_a(net, T_mu=T_mu, num_samples=16, seed=0)
    assert np.isfinite(rs.score)
    assert rs.num_samples == 16

    from ird_playground.ird.query_base import rail_y_grad_ad_fd

    rail = rail_y_grad_ad_fd(net, n=8, seed=0)
    assert rail["rail_ad_fd_rel"] < 0.5
    assert rail["rail_sign_agree"] >= 0.5


def test_load_neural_point_yaml():
    from ird_playground.neural.train import load_train_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_train_config(root / "configs/train_config.yaml", root=root)
    assert cfg.epochs >= 1
    cfg_b = load_train_config(root / "configs/train_phase_b.yaml", root=root)
    assert cfg_b.init_checkpoint is not None
    assert cfg_b.freeze_cls_epochs >= 1


def test_ird_viz_gt_only(tmp_path):
    from ird_playground.ird.export_gt import make_synthetic_ird_gt
    from ird_playground.viz.ird_compare import features_to_xyz, render_ird_comparison

    arrays = make_synthetic_ird_gt(2000, seed=0)
    out = tmp_path / "ird.png"
    render_ird_comparison(
        xyz=features_to_xyz(arrays["features"]),
        gt=arrays["d"],
        pred=None,
        out_path=out,
        max_points=1500,
    )
    assert out.exists() and out.stat().st_size > 1000


def test_se3_exp_identity():
    T = se3_exp(np.zeros(6))
    assert np.allclose(T, np.eye(4), atol=1e-9)
```

### `configs/train_config.yaml`

```yaml
# train_config.yaml — Neural IRD v6 phase A: cls-only on stable-support labels
# Env: cd ird_playground && source env.sh

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6
  num_freqs_u: 5
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5
  use_physical_pe: true
  p_scale_m: 1.0

training:
  seed: 42
  batch_size: 1024
  num_workers: 4
  torch_compile: false
  epochs: 40
  save_freq: 10
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 500
  min_lr_ratio: 0.01
  grad_clip_norm: 10.0
  log_every_steps: 10
  print_every_steps: 50
  hardneg_every: 0
  hardneg_frac: 0.0
  val_eval_n: 65536
  val_calib_frac: 0.5
  train_hard_y: true
  device: cuda
  batch_mix:
    interior: 0.15
    bnd_pos: 0.25
    bnd_neg: 0.25
    jitter_pos: 0.10
    jitter_neg: 0.10
    exterior: 0.15

loss:
  lambda_cls: 1.0
  lambda_margin: 0.0
  lambda_q: 0.0
  lambda_local: 0.0

io:
  checkpoint: data/checkpoints/latest.pt
  checkpoint_dir: data/checkpoints
  report: data/reports/train_point.json

pass:
  mae_max: 9.0
  spearman_min: 0.0
  boundary_iou_min: 0.65
  grad_cosine_min: 0.0
  ascent_improve_min: 0.0
  rail_ad_fd_rel_max: 1.0
  rail_sign_agree_min: 0.0
  region_improve_min: 0.0

wandb:
  enable: true
  project: neural-ird-rm75
  entity: lpei82060-technical-university-of-munich
  mode: online
  run_name: neural_ird_v6_stable_support
  tags: [neural_ird, v6, stable_support, physical_pe, cls_only]
```

### `configs/train_phase_b.yaml`

```yaml
# Phase B: cls + boundary margin + q (after v6 Phase A gate)

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6
  num_freqs_u: 5
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5
  use_physical_pe: true
  p_scale_m: 1.0

training:
  seed: 42
  batch_size: 1024
  num_workers: 4
  torch_compile: false
  epochs: 60
  save_freq: 10
  learning_rate: 1.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 300
  min_lr_ratio: 0.01
  grad_clip_norm: 10.0
  log_every_steps: 10
  print_every_steps: 50
  hardneg_every: 0
  hardneg_frac: 0.0
  val_eval_n: 65536
  val_calib_frac: 0.5
  train_hard_y: true
  device: cuda
  # Phase B1: warm-start Phase A; freeze trunk+cls, train margin/q heads
  init_checkpoint: data/checkpoints/best_iou.pt
  freeze_cls_epochs: 5
  batch_mix:
    interior: 0.15
    bnd_pos: 0.25
    bnd_neg: 0.25
    jitter_pos: 0.10
    jitter_neg: 0.10
    exterior: 0.15

loss:
  lambda_cls: 1.0
  lambda_margin: 0.25
  lambda_q: 0.1
  lambda_local: 0.0

io:
  checkpoint: data/checkpoints/phase_b_latest.pt
  checkpoint_dir: data/checkpoints
  report: data/reports/train_phase_b.json

pass:
  mae_max: 0.35
  spearman_min: 0.70
  boundary_iou_min: 0.65
  grad_cosine_min: 0.30
  ascent_improve_min: 0.40
  rail_ad_fd_rel_max: 0.25
  rail_sign_agree_min: 0.80
  region_improve_min: 0.40

wandb:
  enable: true
  project: neural-ird-rm75
  entity: lpei82060-technical-university-of-munich
  mode: online
  run_name: neural_ird_v6_margin_q
  tags: [neural_ird, v6, stable_support, physical_pe, margin_q]
```

### `configs/ird_gt_config.yaml`

```yaml
# IRD GT v6 — stable-support boundary (C+>=3 & C-==0); no soft_tau fallback

map_dir: ../rm75_control/data/reachability/rm75_6f_1p5cm_15deg_coll_probe
out: data/ird/gt_samples_1p5cm_probe.npz

sampling:
  n_interior: 300000
  n_boundary: 800000
  n_exterior: 400000
  n_jitter: 400000
  max_orients_per_voxel: 28
  hard_negative_frac: 0.50
  hard_negative_radius_m: 0.06
  sigma_p_m: 0.03
  sigma_r_deg: 10.0
  m_clip: 3.0
  m_eps: 0.05
  bbox_margin_m: 0.20
  comfort_from: auto
  k_candidates: 4
  seed: 42
  orient_knn: 7
  soft_tau: 0.05
  unknown_soft_max: 0.25
  trusted_neg_soft_max: 1.0e-6
  min_positive_support: 3
  min_trusted_face_pairs: 5000
```

---

## 4. Training report JSON

### `ird_playground/data/reports/train_point.json`

```json
{
  "checkpoint": "/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/data/checkpoints/latest.pt",
  "history": [
    {
      "epoch": 0,
      "train_loss": 0.41850236470486313,
      "val_loss": 0.48885138706614595,
      "val_iou": 0.7301475204017004,
      "boundary_margin_mae": 0.195896714925766,
      "lr": 0.00029996288667459293,
      "train_iou_t05": 0.7758186397983258,
      "train_pr_auc": 0.9012428950634288,
      "val_iou_t05": 0.7301475204017004,
      "val_iou_calibrated": 0.735074345233547,
      "val_val_threshold": 0.35,
      "val_calib_best_iou": 0.7394791053560373,
      "val_pr_auc": 0.8553474777808483,
      "val_accuracy": 0.8351706288343558,
      "val_boundary_margin_mae": 0.195896714925766,
      "val_interior_recall": 0.958798754806812,
      "val_bnd_pos_recall": 0.9287144573488186,
      "val_bnd_neg_spec": 0.8900398406374502,
      "val_jitter_pos_recall": 0.8932515337423312,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.5566746017212965,
      "val_best_iou": 0.735074345233547,
      "val_best_threshold": 0.35
    },
    {
      "epoch": 1,
      "train_loss": 0.32617552801751953,
      "val_loss": 0.45626780018560964,
      "val_iou": 0.7560382685068415,
      "boundary_margin_mae": 0.2571864724159241,
      "lr": 0.00029922305305056155,
      "train_iou_t05": 0.8057720665681891,
      "train_pr_auc": 0.9175439186964478,
      "val_iou_t05": 0.7560382685068415,
      "val_iou_calibrated": 0.7560382685068415,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.7647372836116247,
      "val_pr_auc": 0.8786024958372202,
      "val_accuracy": 0.8508914877300614,
      "val_boundary_margin_mae": 0.2571864724159241,
      "val_interior_recall": 0.9419520234389306,
      "val_bnd_pos_recall": 0.9263115738886664,
      "val_bnd_neg_spec": 0.9115537848605577,
      "val_jitter_pos_recall": 0.8932515337423312,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.6377952755905512,
      "val_best_iou": 0.7560382685068415,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 2,
      "train_loss": 0.3008196822203889,
      "val_loss": 0.39465764398831427,
      "val_iou": 0.7881709010701096,
      "boundary_margin_mae": 0.2593247890472412,
      "lr": 0.00029753875906486215,
      "train_iou_t05": 0.8269896193769837,
      "train_pr_auc": 0.9276964971962574,
      "val_iou_t05": 0.7881709010701096,
      "val_iou_calibrated": 0.7886957934058154,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.793385183439328,
      "val_pr_auc": 0.8941139026857696,
      "val_accuracy": 0.8757189417177914,
      "val_boundary_margin_mae": 0.2593247890472412,
      "val_interior_recall": 0.9489104559604468,
      "val_bnd_pos_recall": 0.9231077292751302,
      "val_bnd_neg_spec": 0.9199203187250996,
      "val_jitter_pos_recall": 0.9100204498977505,
      "val_jitter_neg_spec": 0.9385542168674699,
      "val_exterior_spec": 0.7148873832631386,
      "val_best_iou": 0.7886957934058154,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 3,
      "train_loss": 0.2777564586797777,
      "val_loss": 0.3442924921487963,
      "val_iou": 0.8076094164455564,
      "boundary_margin_mae": 0.26140859723091125,
      "lr": 0.00029492077317153125,
      "train_iou_t05": 0.8465747428319442,
      "train_pr_auc": 0.9343727330481035,
      "val_iou_t05": 0.8076094164455564,
      "val_iou_calibrated": 0.8076094164455564,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8143775100400952,
      "val_pr_auc": 0.9073724246360314,
      "val_accuracy": 0.8887557515337423,
      "val_boundary_margin_mae": 0.26140859723091125,
      "val_interior_recall": 0.9600805713239333,
      "val_bnd_pos_recall": 0.9235082098518221,
      "val_bnd_neg_spec": 0.9203187250996016,
      "val_jitter_pos_recall": 0.8973415132924335,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.7597509613623878,
      "val_best_iou": 0.8076094164455564,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 4,
      "train_loss": 0.2604499385619713,
      "val_loss": 0.3081015926584958,
      "val_iou": 0.8237974683543609,
      "boundary_margin_mae": 0.24353300034999847,
      "lr": 0.00029138583333971176,
      "train_iou_t05": 0.8557180851061933,
      "train_pr_auc": 0.9412940126812853,
      "val_iou_t05": 0.8237974683543609,
      "val_iou_calibrated": 0.8237974683543609,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8290549378678197,
      "val_pr_auc": 0.917621328326375,
      "val_accuracy": 0.8999233128834356,
      "val_boundary_margin_mae": 0.24353300034999847,
      "val_interior_recall": 0.9604468046145395,
      "val_bnd_pos_recall": 0.9223067681217461,
      "val_bnd_neg_spec": 0.9250996015936255,
      "val_jitter_pos_recall": 0.905521472392638,
      "val_jitter_neg_spec": 0.940562248995984,
      "val_exterior_spec": 0.7965574070683025,
      "val_best_iou": 0.8237974683543609,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 5,
      "train_loss": 0.24769440900953085,
      "val_loss": 0.2927216419494084,
      "val_iou": 0.8312494684017324,
      "boundary_margin_mae": 0.26994502544403076,
      "lr": 0.000286956540040238,
      "train_iou_t05": 0.8624833110812499,
      "train_pr_auc": 0.9457477060541848,
      "val_iou_t05": 0.8312494684017324,
      "val_iou_calibrated": 0.8312494684017324,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8343770599867473,
      "val_pr_auc": 0.921890993559339,
      "val_accuracy": 0.9049079754601227,
      "val_boundary_margin_mae": 0.26994502544403076,
      "val_interior_recall": 0.9514740889946897,
      "val_bnd_pos_recall": 0.9231077292751302,
      "val_bnd_neg_spec": 0.9258964143426295,
      "val_jitter_pos_recall": 0.9292433537832311,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.8143197216626992,
      "val_best_iou": 0.8312494684017324,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 6,
      "train_loss": 0.24006291143918243,
      "val_loss": 0.30294901083054887,
      "val_iou": 0.8340077558589754,
      "boundary_margin_mae": 0.4012444019317627,
      "lr": 0.00028166121175059313,
      "train_iou_t05": 0.8657079646015783,
      "train_pr_auc": 0.9504207521898732,
      "val_iou_t05": 0.8340077558589754,
      "val_iou_calibrated": 0.8352769679299575,
      "val_val_threshold": 0.65,
      "val_calib_best_iou": 0.8435028718887169,
      "val_pr_auc": 0.9258680566129865,
      "val_accuracy": 0.9056269171779141,
      "val_boundary_margin_mae": 0.4012444019317627,
      "val_interior_recall": 0.9474455227980223,
      "val_bnd_pos_recall": 0.921505806968362,
      "val_bnd_neg_spec": 0.9286852589641434,
      "val_jitter_pos_recall": 0.9267893660531697,
      "val_jitter_neg_spec": 0.9393574297188755,
      "val_exterior_spec": 0.8298846365134591,
      "val_best_iou": 0.8352769679299575,
      "val_best_threshold": 0.65
    },
    {
      "epoch": 7,
      "train_loss": 0.23192946531243558,
      "val_loss": 0.2649643812473848,
      "val_iou": 0.8374209203569773,
      "boundary_margin_mae": 0.39210522174835205,
      "lr": 0.00027553370390206105,
      "train_iou_t05": 0.8699186991867953,
      "train_pr_auc": 0.9562572718168052,
      "val_iou_t05": 0.8374209203569773,
      "val_iou_calibrated": 0.8398283261801854,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8473333333332627,
      "val_pr_auc": 0.9291557545985822,
      "val_accuracy": 0.9100843558282209,
      "val_boundary_margin_mae": 0.39210522174835205,
      "val_interior_recall": 0.9582494048709027,
      "val_bnd_pos_recall": 0.921505806968362,
      "val_bnd_neg_spec": 0.9247011952191235,
      "val_jitter_pos_recall": 0.9202453987730062,
      "val_jitter_neg_spec": 0.9389558232931727,
      "val_exterior_spec": 0.8340963193554294,
      "val_best_iou": 0.8398283261801854,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 8,
      "train_loss": 0.22503704425237364,
      "val_loss": 0.27336324215711033,
      "val_iou": 0.8429432013768637,
      "boundary_margin_mae": 0.4704406261444092,
      "lr": 0.0002686131924266254,
      "train_iou_t05": 0.8715699505171228,
      "train_pr_auc": 0.9595545185771817,
      "val_iou_t05": 0.8429432013768637,
      "val_iou_calibrated": 0.8429432013768637,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.845749562025457,
      "val_pr_auc": 0.9283475848007654,
      "val_accuracy": 0.9125287576687117,
      "val_boundary_margin_mae": 0.4704406261444092,
      "val_interior_recall": 0.9564182384178722,
      "val_bnd_pos_recall": 0.9255106127352823,
      "val_bnd_neg_spec": 0.9282868525896414,
      "val_jitter_pos_recall": 0.9247443762781186,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.8383080021973998,
      "val_best_iou": 0.8429432013768637,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 9,
      "train_loss": 0.21957305327100782,
      "val_loss": 0.26556964434437225,
      "val_iou": 0.8393678909437493,
      "boundary_margin_mae": 0.43538597226142883,
      "lr": 0.00026094392328749954,
      "train_iou_t05": 0.8771571298817262,
      "train_pr_auc": 0.9625145366709653,
      "val_iou_t05": 0.8393678909437493,
      "val_iou_calibrated": 0.842281590928542,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8507350484463038,
      "val_pr_auc": 0.9288271387217303,
      "val_accuracy": 0.9113305214723927,
      "val_boundary_margin_mae": 0.43538597226142883,
      "val_interior_recall": 0.955868888481963,
      "val_bnd_pos_recall": 0.9235082098518221,
      "val_bnd_neg_spec": 0.9278884462151394,
      "val_jitter_pos_recall": 0.9321063394683027,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8348287859366417,
      "val_best_iou": 0.842281590928542,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 10,
      "train_loss": 0.21349913502375056,
      "val_loss": 0.27231731999386066,
      "val_iou": 0.8438469493277453,
      "boundary_margin_mae": 0.46200379729270935,
      "lr": 0.0002525749295946502,
      "train_iou_t05": 0.8774774774772798,
      "train_pr_auc": 0.9640835497841461,
      "val_iou_t05": 0.8438469493277453,
      "val_iou_calibrated": 0.8438469493277453,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8489383046312615,
      "val_pr_auc": 0.9300139504648135,
      "val_accuracy": 0.9131518404907976,
      "val_boundary_margin_mae": 0.46200379729270935,
      "val_interior_recall": 0.958798754806812,
      "val_bnd_pos_recall": 0.9199038846615939,
      "val_bnd_neg_spec": 0.9286852589641434,
      "val_jitter_pos_recall": 0.9239263803680982,
      "val_jitter_neg_spec": 0.9385542168674699,
      "val_exterior_spec": 0.8408716352316425,
      "val_best_iou": 0.8438469493277453,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 11,
      "train_loss": 0.2099476332537379,
      "val_loss": 0.2681516713170253,
      "val_iou": 0.8427902134304318,
      "boundary_margin_mae": 0.4221237599849701,
      "lr": 0.00024355971811392555,
      "train_iou_t05": 0.8777173913041489,
      "train_pr_auc": 0.9665019474541227,
      "val_iou_t05": 0.8427902134304318,
      "val_iou_calibrated": 0.8436394997843171,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8486115757777811,
      "val_pr_auc": 0.9309659767482299,
      "val_accuracy": 0.9131518404907976,
      "val_boundary_margin_mae": 0.4221237599849701,
      "val_interior_recall": 0.9531221388024171,
      "val_bnd_pos_recall": 0.920704845814978,
      "val_bnd_neg_spec": 0.9274900398406375,
      "val_jitter_pos_recall": 0.9316973415132924,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8434352682658853,
      "val_best_iou": 0.8436394997843171,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 12,
      "train_loss": 0.20471227050944088,
      "val_loss": 0.26883386146771465,
      "val_iou": 0.840285542272225,
      "boundary_margin_mae": 0.3681005537509918,
      "lr": 0.00023395592717407503,
      "train_iou_t05": 0.8778781038372735,
      "train_pr_auc": 0.9664109997843222,
      "val_iou_t05": 0.840285542272225,
      "val_iou_calibrated": 0.840285542272225,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8436612660764286,
      "val_pr_auc": 0.9299771907999207,
      "val_accuracy": 0.9109950153374233,
      "val_boundary_margin_mae": 0.3681005537509918,
      "val_interior_recall": 0.9628273210034792,
      "val_bnd_pos_recall": 0.9162995594713657,
      "val_bnd_neg_spec": 0.9223107569721115,
      "val_jitter_pos_recall": 0.9096114519427403,
      "val_jitter_neg_spec": 0.940562248995984,
      "val_exterior_spec": 0.8386742354880059,
      "val_best_iou": 0.840285542272225,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 13,
      "train_loss": 0.20018888052327488,
      "val_loss": 0.2675953709920815,
      "val_iou": 0.8441120525677983,
      "boundary_margin_mae": 0.4378807246685028,
      "lr": 0.00022382495815881717,
      "train_iou_t05": 0.8832993428504682,
      "train_pr_auc": 0.9694131967628098,
      "val_iou_t05": 0.8441120525677983,
      "val_iou_calibrated": 0.84270815524014,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.850224663005421,
      "val_pr_auc": 0.9294324406936052,
      "val_accuracy": 0.9135832055214724,
      "val_boundary_margin_mae": 0.4378807246685028,
      "val_interior_recall": 0.9677714704266618,
      "val_bnd_pos_recall": 0.9227072486984381,
      "val_bnd_neg_spec": 0.9135458167330678,
      "val_jitter_pos_recall": 0.9280163599182004,
      "val_jitter_neg_spec": 0.934136546184739,
      "val_exterior_spec": 0.8326313861930049,
      "val_best_iou": 0.84270815524014,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 14,
      "train_loss": 0.1953404669029912,
      "val_loss": 0.2679686478786935,
      "val_iou": 0.8388129434157433,
      "boundary_margin_mae": 0.40320348739624023,
      "lr": 0.0002132315829399933,
      "train_iou_t05": 0.8879526303801211,
      "train_pr_auc": 0.9711618676044179,
      "val_iou_t05": 0.8388129434157433,
      "val_iou_calibrated": 0.8388129434157433,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.845028994033041,
      "val_pr_auc": 0.9303643866366744,
      "val_accuracy": 0.9107074386503068,
      "val_boundary_margin_mae": 0.40320348739624023,
      "val_interior_recall": 0.9518403222852957,
      "val_bnd_pos_recall": 0.9094913896676011,
      "val_bnd_neg_spec": 0.9211155378486056,
      "val_jitter_pos_recall": 0.9104294478527607,
      "val_jitter_neg_spec": 0.9357429718875502,
      "val_exterior_spec": 0.8540560336934627,
      "val_best_iou": 0.8388129434157433,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 15,
      "train_loss": 0.18972616155312452,
      "val_loss": 0.2721048505115658,
      "val_iou": 0.8385452660676539,
      "boundary_margin_mae": 0.40575361251831055,
      "lr": 0.00020224352976166442,
      "train_iou_t05": 0.8892930211409662,
      "train_pr_auc": 0.9725295632963787,
      "val_iou_t05": 0.8385452660676539,
      "val_iou_calibrated": 0.8385452660676539,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8422198041348588,
      "val_pr_auc": 0.9296051617411568,
      "val_accuracy": 0.9104198619631901,
      "val_boundary_margin_mae": 0.40575361251831055,
      "val_interior_recall": 0.9505585057681744,
      "val_bnd_pos_recall": 0.9090909090909091,
      "val_bnd_neg_spec": 0.9183266932270916,
      "val_jitter_pos_recall": 0.9186094069529652,
      "val_jitter_neg_spec": 0.9325301204819277,
      "val_exterior_spec": 0.8535066837575536,
      "val_best_iou": 0.8385452660676539,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 16,
      "train_loss": 0.18511594110504007,
      "val_loss": 0.2706585433887567,
      "val_iou": 0.8375713544368761,
      "boundary_margin_mae": 0.4191840887069702,
      "lr": 0.00019093105022278232,
      "train_iou_t05": 0.8895481515287791,
      "train_pr_auc": 0.9729320563736255,
      "val_iou_t05": 0.8375713544368761,
      "val_iou_calibrated": 0.8375713544368761,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.840154310633945,
      "val_pr_auc": 0.9304518107340402,
      "val_accuracy": 0.9099884969325154,
      "val_boundary_margin_mae": 0.4191840887069702,
      "val_interior_recall": 0.9518403222852957,
      "val_bnd_pos_recall": 0.9054865839006808,
      "val_bnd_neg_spec": 0.9227091633466136,
      "val_jitter_pos_recall": 0.9100204498977505,
      "val_jitter_neg_spec": 0.9373493975903614,
      "val_exterior_spec": 0.8518586339498261,
      "val_best_iou": 0.8375713544368761,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 17,
      "train_loss": 0.1801638848628709,
      "val_loss": 0.2713570190995338,
      "val_iou": 0.8381767860231069,
      "boundary_margin_mae": 0.4416922330856323,
      "lr": 0.00017936647012690993,
      "train_iou_t05": 0.8905608755127927,
      "train_pr_auc": 0.9742839188977517,
      "val_iou_t05": 0.8381767860231069,
      "val_iou_calibrated": 0.8391259463178906,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8430290872617149,
      "val_pr_auc": 0.929821427419754,
      "val_accuracy": 0.9103240030674846,
      "val_boundary_margin_mae": 0.4416922330856323,
      "val_interior_recall": 0.9598974546786303,
      "val_bnd_pos_recall": 0.9130957148578294,
      "val_bnd_neg_spec": 0.9231075697211155,
      "val_jitter_pos_recall": 0.912883435582822,
      "val_jitter_neg_spec": 0.9349397590361446,
      "val_exterior_spec": 0.8414209851675517,
      "val_best_iou": 0.8391259463178906,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 18,
      "train_loss": 0.174000618973967,
      "val_loss": 0.26850442882055614,
      "val_iou": 0.8344659348317796,
      "boundary_margin_mae": 0.40707385540008545,
      "lr": 0.00016762372707061277,
      "train_iou_t05": 0.8943014705880297,
      "train_pr_auc": 0.9751062589650152,
      "val_iou_t05": 0.8344659348317796,
      "val_iou_calibrated": 0.8384859124774481,
      "val_val_threshold": 0.35,
      "val_calib_best_iou": 0.8387873754152126,
      "val_pr_auc": 0.9289947144867179,
      "val_accuracy": 0.9089340490797546,
      "val_boundary_margin_mae": 0.40707385540008545,
      "val_interior_recall": 0.965390954037722,
      "val_bnd_pos_recall": 0.9146976371645975,
      "val_bnd_neg_spec": 0.9183266932270916,
      "val_jitter_pos_recall": 0.9141104294478528,
      "val_jitter_neg_spec": 0.9293172690763052,
      "val_exterior_spec": 0.8364768357443692,
      "val_best_iou": 0.8384859124774481,
      "val_best_threshold": 0.35
    },
    {
      "epoch": 19,
      "train_loss": 0.16896025165486062,
      "val_loss": 0.27339208773886825,
      "val_iou": 0.8336515098923261,
      "boundary_margin_mae": 0.43572598695755005,
      "lr": 0.00015577789772692807,
      "train_iou_t05": 0.897912365221175,
      "train_pr_auc": 0.9779545654392623,
      "val_iou_t05": 0.8336515098923261,
      "val_iou_calibrated": 0.8352618496613666,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8314820954906472,
      "val_pr_auc": 0.9256925054097181,
      "val_accuracy": 0.9081192484662577,
      "val_boundary_margin_mae": 0.43572598695755005,
      "val_interior_recall": 0.9562351217725691,
      "val_bnd_pos_recall": 0.9134961954345214,
      "val_bnd_neg_spec": 0.9087649402390439,
      "val_jitter_pos_recall": 0.9169734151329243,
      "val_jitter_neg_spec": 0.9204819277108434,
      "val_exterior_spec": 0.8467313678813404,
      "val_best_iou": 0.8352618496613666,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 20,
      "train_loss": 0.16361886806120446,
      "val_loss": 0.2781357268650822,
      "val_iou": 0.8381009822504878,
      "boundary_margin_mae": 0.45061758160591125,
      "lr": 0.00014390471784620173,
      "train_iou_t05": 0.903771428571222,
      "train_pr_auc": 0.9788411508272411,
      "val_iou_t05": 0.8381009822504878,
      "val_iou_calibrated": 0.8381009822504878,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8331529431353898,
      "val_pr_auc": 0.9280813230644325,
      "val_accuracy": 0.9099405674846626,
      "val_boundary_margin_mae": 0.45061758160591125,
      "val_interior_recall": 0.9555026551913569,
      "val_bnd_pos_recall": 0.9134961954345214,
      "val_bnd_neg_spec": 0.9135458167330678,
      "val_jitter_pos_recall": 0.9112474437627812,
      "val_jitter_neg_spec": 0.9281124497991968,
      "val_exterior_spec": 0.8522248672404321,
      "val_best_iou": 0.8381009822504878,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 21,
      "train_loss": 0.1570993201640738,
      "val_loss": 0.2746293237947821,
      "val_iou": 0.834342560553561,
      "boundary_margin_mae": 0.4413347542285919,
      "lr": 0.00013208009804314651,
      "train_iou_t05": 0.9031443653888218,
      "train_pr_auc": 0.9795764884709481,
      "val_iou_t05": 0.834342560553561,
      "val_iou_calibrated": 0.8355544165384559,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8355164980931159,
      "val_pr_auc": 0.9258259717655953,
      "val_accuracy": 0.9082151073619632,
      "val_boundary_margin_mae": 0.4413347542285919,
      "val_interior_recall": 0.9679545870719648,
      "val_bnd_pos_recall": 0.907889467360833,
      "val_bnd_neg_spec": 0.9091633466135458,
      "val_jitter_pos_recall": 0.9112474437627812,
      "val_jitter_neg_spec": 0.9265060240963855,
      "val_exterior_spec": 0.8366599523896722,
      "val_best_iou": 0.8355544165384559,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 22,
      "train_loss": 0.15311087265353038,
      "val_loss": 0.281019922871763,
      "val_iou": 0.8305735745987682,
      "boundary_margin_mae": 0.45839619636535645,
      "lr": 0.00012037963846591231,
      "train_iou_t05": 0.9056302712557771,
      "train_pr_auc": 0.9804766269983293,
      "val_iou_t05": 0.8305735745987682,
      "val_iou_calibrated": 0.8305735745987682,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8295407529995176,
      "val_pr_auc": 0.9213317996039816,
      "val_accuracy": 0.90486004601227,
      "val_boundary_margin_mae": 0.45839619636535645,
      "val_interior_recall": 0.9564182384178722,
      "val_bnd_pos_recall": 0.9126952342811374,
      "val_bnd_neg_spec": 0.899601593625498,
      "val_jitter_pos_recall": 0.9116564417177914,
      "val_jitter_neg_spec": 0.9112449799196787,
      "val_exterior_spec": 0.8461820179454312,
      "val_best_iou": 0.8305735745987682,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 23,
      "train_loss": 0.14755773245858872,
      "val_loss": 0.28743589189733665,
      "val_iou": 0.8322602798522764,
      "boundary_margin_mae": 0.4674295485019684,
      "lr": 0.00010887814545011011,
      "train_iou_t05": 0.9082125603862644,
      "train_pr_auc": 0.9808393780586795,
      "val_iou_t05": 0.8322602798522764,
      "val_iou_calibrated": 0.8323516858727416,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8307501036054015,
      "val_pr_auc": 0.9238083139685108,
      "val_accuracy": 0.9063458588957055,
      "val_boundary_margin_mae": 0.4674295485019684,
      "val_interior_recall": 0.9701519868156016,
      "val_bnd_pos_recall": 0.9018822587104526,
      "val_bnd_neg_spec": 0.9139442231075697,
      "val_jitter_pos_recall": 0.9002044989775051,
      "val_jitter_neg_spec": 0.927710843373494,
      "val_exterior_spec": 0.8322651529023988,
      "val_best_iou": 0.8323516858727416,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 24,
      "train_loss": 0.14322808126164788,
      "val_loss": 0.2878010178689768,
      "val_iou": 0.8289745354438378,
      "boundary_margin_mae": 0.4683523178100586,
      "lr": 9.764915324803924e-05,
      "train_iou_t05": 0.9111213446923989,
      "train_pr_auc": 0.9824560476520735,
      "val_iou_t05": 0.8289745354438378,
      "val_iou_calibrated": 0.8295833689793113,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8260258107213082,
      "val_pr_auc": 0.9177640704535162,
      "val_accuracy": 0.9047162576687117,
      "val_boundary_margin_mae": 0.4683523178100586,
      "val_interior_recall": 0.958798754806812,
      "val_bnd_pos_recall": 0.9034841810172206,
      "val_bnd_neg_spec": 0.9043824701195219,
      "val_jitter_pos_recall": 0.901840490797546,
      "val_jitter_neg_spec": 0.9180722891566265,
      "val_exterior_spec": 0.8458157846548251,
      "val_best_iou": 0.8295833689793113,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 25,
      "train_loss": 0.13945309181163565,
      "val_loss": 0.2821642189918993,
      "val_iou": 0.8330313228060373,
      "boundary_margin_mae": 0.47823548316955566,
      "lr": 8.676445389091839e-05,
      "train_iou_t05": 0.9153597785975748,
      "train_pr_auc": 0.9821214512861555,
      "val_iou_t05": 0.8330313228060373,
      "val_iou_calibrated": 0.834664605342194,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8300789364353277,
      "val_pr_auc": 0.9180081566564965,
      "val_accuracy": 0.907256518404908,
      "val_boundary_margin_mae": 0.47823548316955566,
      "val_interior_recall": 0.9562351217725691,
      "val_bnd_pos_recall": 0.9106928313976772,
      "val_bnd_neg_spec": 0.9087649402390439,
      "val_jitter_pos_recall": 0.9087934560327199,
      "val_jitter_neg_spec": 0.9180722891566265,
      "val_exterior_spec": 0.8522248672404321,
      "val_best_iou": 0.834664605342194,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 26,
      "train_loss": 0.13543461804259407,
      "val_loss": 0.28752858255243724,
      "val_iou": 0.8312392426849543,
      "boundary_margin_mae": 0.49189165234565735,
      "lr": 7.629363818992554e-05,
      "train_iou_t05": 0.9192059095104064,
      "train_pr_auc": 0.9839478939361008,
      "val_iou_t05": 0.8312392426849543,
      "val_iou_calibrated": 0.8312392426849543,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8291260518203092,
      "val_pr_auc": 0.9157921113352897,
      "val_accuracy": 0.9060103527607362,
      "val_boundary_margin_mae": 0.49189165234565735,
      "val_interior_recall": 0.9511078557040835,
      "val_bnd_pos_recall": 0.9014817781337605,
      "val_bnd_neg_spec": 0.9063745019920318,
      "val_jitter_pos_recall": 0.905521472392638,
      "val_jitter_neg_spec": 0.921285140562249,
      "val_exterior_spec": 0.8560703167917963,
      "val_best_iou": 0.8312392426849543,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 27,
      "train_loss": 0.13208875109465734,
      "val_loss": 0.2893802533694592,
      "val_iou": 0.8358003442340072,
      "boundary_margin_mae": 0.5149006843566895,
      "lr": 6.63036508106338e-05,
      "train_iou_t05": 0.9210344031399397,
      "train_pr_auc": 0.984048934759558,
      "val_iou_t05": 0.8358003442340072,
      "val_iou_calibrated": 0.8358003442340072,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8316543168761276,
      "val_pr_auc": 0.9166939684085275,
      "val_accuracy": 0.9085506134969326,
      "val_boundary_margin_mae": 0.5149006843566895,
      "val_interior_recall": 0.9555026551913569,
      "val_bnd_pos_recall": 0.9086904285142171,
      "val_bnd_neg_spec": 0.9087649402390439,
      "val_jitter_pos_recall": 0.9100204498977505,
      "val_jitter_neg_spec": 0.9228915662650602,
      "val_exterior_spec": 0.8542391503387657,
      "val_best_iou": 0.8358003442340072,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 28,
      "train_loss": 0.1290918326687057,
      "val_loss": 0.29088322118151183,
      "val_iou": 0.8314163090128042,
      "boundary_margin_mae": 0.5087964534759521,
      "lr": 5.6858362265454576e-05,
      "train_iou_t05": 0.9189937687512302,
      "train_pr_auc": 0.9841364978213021,
      "val_iou_t05": 0.8314163090128042,
      "val_iou_calibrated": 0.8302863056225569,
      "val_val_threshold": 0.5499999999999999,
      "val_calib_best_iou": 0.8270356127737145,
      "val_pr_auc": 0.9125379632114471,
      "val_accuracy": 0.9058665644171779,
      "val_boundary_margin_mae": 0.5087964534759521,
      "val_interior_recall": 0.9472624061527193,
      "val_bnd_pos_recall": 0.9026832198638366,
      "val_bnd_neg_spec": 0.9023904382470119,
      "val_jitter_pos_recall": 0.9002044989775051,
      "val_jitter_neg_spec": 0.923293172690763,
      "val_exterior_spec": 0.861380699505585,
      "val_best_iou": 0.8302863056225569,
      "val_best_threshold": 0.5499999999999999
    },
    {
      "epoch": 29,
      "train_loss": 0.12646190899421572,
      "val_loss": 0.2958447068659213,
      "val_iou": 0.8312661498707294,
      "boundary_margin_mae": 0.5291464924812317,
      "lr": 4.801816056053316e-05,
      "train_iou_t05": 0.9213691026824881,
      "train_pr_auc": 0.9846419492266404,
      "val_iou_t05": 0.8312661498707294,
      "val_iou_calibrated": 0.8320329246333849,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8289200630861767,
      "val_pr_auc": 0.9130402340806761,
      "val_accuracy": 0.9061062116564417,
      "val_boundary_margin_mae": 0.5291464924812317,
      "val_interior_recall": 0.9573338216443875,
      "val_bnd_pos_recall": 0.9038846615939127,
      "val_bnd_neg_spec": 0.9031872509960159,
      "val_jitter_pos_recall": 0.9075664621676892,
      "val_jitter_neg_spec": 0.9172690763052209,
      "val_exterior_spec": 0.8514924006592199,
      "val_best_iou": 0.8320329246333849,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 30,
      "train_loss": 0.12393183913887063,
      "val_loss": 0.30129186658931956,
      "val_iou": 0.8290899707551325,
      "boundary_margin_mae": 0.5458419322967529,
      "lr": 3.9839565107884453e-05,
      "train_iou_t05": 0.9196304849882402,
      "train_pr_auc": 0.9850723820918336,
      "val_iou_t05": 0.8290899707551325,
      "val_iou_calibrated": 0.8303092256579921,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.822750452376968,
      "val_pr_auc": 0.9080221045420739,
      "val_accuracy": 0.9047641871165644,
      "val_boundary_margin_mae": 0.5458419322967529,
      "val_interior_recall": 0.965207837392419,
      "val_bnd_pos_recall": 0.9054865839006808,
      "val_bnd_neg_spec": 0.895617529880478,
      "val_jitter_pos_recall": 0.9059304703476483,
      "val_jitter_neg_spec": 0.9124497991967871,
      "val_exterior_spec": 0.8432521516205823,
      "val_best_iou": 0.8303092256579921,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 31,
      "train_loss": 0.12206302154862915,
      "val_loss": 0.3024251819998285,
      "val_iou": 0.8296060553930131,
      "boundary_margin_mae": 0.5503641366958618,
      "lr": 3.237486537120169e-05,
      "train_iou_t05": 0.9226672840933264,
      "train_pr_auc": 0.9847933760646623,
      "val_iou_t05": 0.8296060553930131,
      "val_iou_calibrated": 0.8299991487187511,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8243777814405122,
      "val_pr_auc": 0.9088427146689573,
      "val_accuracy": 0.905051763803681,
      "val_boundary_margin_mae": 0.5503641366958618,
      "val_interior_recall": 0.9705182201062077,
      "val_bnd_pos_recall": 0.9014817781337605,
      "val_bnd_neg_spec": 0.8968127490039841,
      "val_jitter_pos_recall": 0.8993865030674847,
      "val_jitter_neg_spec": 0.9144578313253012,
      "val_exterior_spec": 0.8403222852957334,
      "val_best_iou": 0.8299991487187511,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 32,
      "train_loss": 0.1199199934235052,
      "val_loss": 0.30306634079988104,
      "val_iou": 0.8304093567250748,
      "boundary_margin_mae": 0.5590139627456665,
      "lr": 2.567178655564044e-05,
      "train_iou_t05": 0.9221169401430731,
      "train_pr_auc": 0.9854753728062398,
      "val_iou_t05": 0.8304093567250748,
      "val_iou_calibrated": 0.8304093567250748,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8245322245321559,
      "val_pr_auc": 0.9085488069726347,
      "val_accuracy": 0.9054831288343558,
      "val_boundary_margin_mae": 0.5590139627456665,
      "val_interior_recall": 0.9544039553195386,
      "val_bnd_pos_recall": 0.8994793752503003,
      "val_bnd_neg_spec": 0.9015936254980079,
      "val_jitter_pos_recall": 0.8989775051124744,
      "val_jitter_neg_spec": 0.9208835341365462,
      "val_exterior_spec": 0.8569859000183117,
      "val_best_iou": 0.8304093567250748,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 33,
      "train_loss": 0.11904757756535876,
      "val_loss": 0.30454234943319314,
      "val_iou": 0.8288288288287576,
      "boundary_margin_mae": 0.5596349835395813,
      "lr": 1.9773184478973925e-05,
      "train_iou_t05": 0.92295233688086,
      "train_pr_auc": 0.9857278995812323,
      "val_iou_t05": 0.8288288288287576,
      "val_iou_calibrated": 0.8288288288287576,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8236514522820892,
      "val_pr_auc": 0.9065150309876957,
      "val_accuracy": 0.9043807515337423,
      "val_boundary_margin_mae": 0.5596349835395813,
      "val_interior_recall": 0.9567844717084782,
      "val_bnd_pos_recall": 0.8978774529435323,
      "val_bnd_neg_spec": 0.8968127490039841,
      "val_jitter_pos_recall": 0.8969325153374234,
      "val_jitter_neg_spec": 0.9160642570281124,
      "val_exterior_spec": 0.8564365500824025,
      "val_best_iou": 0.8288288288287576,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 34,
      "train_loss": 0.11750743283704997,
      "val_loss": 0.30506036444583706,
      "val_iou": 0.828438948995292,
      "boundary_margin_mae": 0.5677558183670044,
      "lr": 1.4716771574946322e-05,
      "train_iou_t05": 0.9229701596111674,
      "train_pr_auc": 0.9857150885825672,
      "val_iou_t05": 0.828438948995292,
      "val_iou_calibrated": 0.828438948995292,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.823197544179804,
      "val_pr_auc": 0.9071009695994803,
      "val_accuracy": 0.9042369631901841,
      "val_boundary_margin_mae": 0.5677558183670044,
      "val_interior_recall": 0.9564182384178722,
      "val_bnd_pos_recall": 0.8966760112134562,
      "val_bnd_neg_spec": 0.8988047808764941,
      "val_jitter_pos_recall": 0.8940695296523518,
      "val_jitter_neg_spec": 0.9204819277108434,
      "val_exterior_spec": 0.855154733565281,
      "val_best_iou": 0.828438948995292,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 35,
      "train_loss": 0.11643428194419794,
      "val_loss": 0.30518335904481336,
      "val_iou": 0.8294153104278972,
      "boundary_margin_mae": 0.5692489147186279,
      "lr": 1.0534875780609204e-05,
      "train_iou_t05": 0.9235403151063661,
      "train_pr_auc": 0.9858113523963622,
      "val_iou_t05": 0.8294153104278972,
      "val_iou_calibrated": 0.8295230756057171,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8236170917521676,
      "val_pr_auc": 0.9077747708219236,
      "val_accuracy": 0.905051763803681,
      "val_boundary_margin_mae": 0.5692489147186279,
      "val_interior_recall": 0.9593481047427211,
      "val_bnd_pos_recall": 0.9018822587104526,
      "val_bnd_neg_spec": 0.899601593625498,
      "val_jitter_pos_recall": 0.8985685071574642,
      "val_jitter_neg_spec": 0.9160642570281124,
      "val_exterior_spec": 0.8507599340780077,
      "val_best_iou": 0.8295230756057171,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 36,
      "train_loss": 0.11572435256684205,
      "val_loss": 0.30799682393194044,
      "val_iou": 0.8285173718609539,
      "boundary_margin_mae": 0.5745746493339539,
      "lr": 7.254233849182988e-06,
      "train_iou_t05": 0.9228809634087718,
      "train_pr_auc": 0.9857925636670787,
      "val_iou_t05": 0.8285173718609539,
      "val_iou_calibrated": 0.8305517711170954,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8230838890260291,
      "val_pr_auc": 0.9070164653445646,
      "val_accuracy": 0.9044286809815951,
      "val_boundary_margin_mae": 0.5745746493339539,
      "val_interior_recall": 0.9705182201062077,
      "val_bnd_pos_recall": 0.9022827392871445,
      "val_bnd_neg_spec": 0.895617529880478,
      "val_jitter_pos_recall": 0.9002044989775051,
      "val_jitter_neg_spec": 0.9136546184738956,
      "val_exterior_spec": 0.8417872184581578,
      "val_best_iou": 0.8305517711170954,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 37,
      "train_loss": 0.1158654165886321,
      "val_loss": 0.3071477001561402,
      "val_iou": 0.8291738382099114,
      "boundary_margin_mae": 0.5733893513679504,
      "lr": 4.89582040988815e-06,
      "train_iou_t05": 0.9238073182026577,
      "train_pr_auc": 0.9858774191838093,
      "val_iou_t05": 0.8291738382099114,
      "val_iou_calibrated": 0.8304779852663328,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8237240466539147,
      "val_pr_auc": 0.9068775957175322,
      "val_accuracy": 0.90486004601227,
      "val_boundary_margin_mae": 0.5733893513679504,
      "val_interior_recall": 0.9633766709393884,
      "val_bnd_pos_recall": 0.8994793752503003,
      "val_bnd_neg_spec": 0.900796812749004,
      "val_jitter_pos_recall": 0.8948875255623722,
      "val_jitter_neg_spec": 0.9184738955823293,
      "val_exterior_spec": 0.8500274674967955,
      "val_best_iou": 0.8304779852663328,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 38,
      "train_loss": 0.11443867376584141,
      "val_loss": 0.3083003169336293,
      "val_iou": 0.8280364198590596,
      "boundary_margin_mae": 0.5757826566696167,
      "lr": 3.474713867644215e-06,
      "train_iou_t05": 0.9247685185183043,
      "train_pr_auc": 0.9857753661943619,
      "val_iou_t05": 0.8280364198590596,
      "val_iou_calibrated": 0.8292724630246363,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8239960337133677,
      "val_pr_auc": 0.9064920736286751,
      "val_accuracy": 0.904045245398773,
      "val_boundary_margin_mae": 0.5757826566696167,
      "val_interior_recall": 0.9641091375206006,
      "val_bnd_pos_recall": 0.8998798558269924,
      "val_bnd_neg_spec": 0.8988047808764941,
      "val_jitter_pos_recall": 0.8948875255623722,
      "val_jitter_neg_spec": 0.9156626506024096,
      "val_exterior_spec": 0.8480131843984619,
      "val_best_iou": 0.8292724630246363,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 39,
      "train_loss": 0.11476776861620568,
      "val_loss": 0.3086963927061923,
      "val_iou": 0.828249849643369,
      "boundary_margin_mae": 0.5772822499275208,
      "lr": 2.9999999999999997e-06,
      "train_iou_t05": 0.9240740740738601,
      "train_pr_auc": 0.9858584014824332,
      "val_iou_t05": 0.828249849643369,
      "val_iou_calibrated": 0.8293830752117027,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8231354390606231,
      "val_pr_auc": 0.9063438133580639,
      "val_accuracy": 0.9041890337423313,
      "val_boundary_margin_mae": 0.5772822499275208,
      "val_interior_recall": 0.9642922541659037,
      "val_bnd_pos_recall": 0.8990788946736084,
      "val_bnd_neg_spec": 0.897609561752988,
      "val_jitter_pos_recall": 0.8924335378323108,
      "val_jitter_neg_spec": 0.9164658634538153,
      "val_exterior_spec": 0.8500274674967955,
      "val_best_iou": 0.8293830752117027,
      "val_best_threshold": 0.44999999999999996
    }
  ],
  "val_metrics": {
    "mae": 0.5820985139737609,
    "mae_m": 0.46034235578077587,
    "mae_q": 0.11176038213863265,
    "spearman": 0.32535996940325584,
    "boundary_iou": 0.8009656816306452,
    "reach_accuracy": 0.896122745179349,
    "n": 125398,
    "interior_recall": 0.958798754806812,
    "bnd_pos_recall": 0.9199038846615939,
    "bnd_neg_spec": 0.9286852589641434,
    "jitter_pos_recall": 0.9239263803680982,
    "jitter_neg_spec": 0.9385542168674699,
    "exterior_spec": 0.8408716352316425,
    "iou": 0.8438469493277453,
    "accuracy": 0.9131518404907976,
    "iou_t05": 0.8438469493277453,
    "iou_calibrated": 0.8438469493277453,
    "val_threshold": 0.49999999999999994,
    "calib_best_iou": 0.8489383046312615,
    "pr_auc": 0.9300139504648135,
    "boundary_margin_mae": 0.46200379729270935,
    "best_iou": 0.8438469493277453,
    "best_threshold": 0.49999999999999994
  }
}
```

### `ird_playground/data/reports/train_phase_b.json`

```json
{
  "checkpoint": "/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/data/checkpoints/phase_b_latest.pt",
  "history": [
    {
      "epoch": 0,
      "train_loss": 0.4327040866919141,
      "val_loss": 0.49850877776617686,
      "val_iou": 0.7262459521364247,
      "boundary_margin_mae": 0.0428873710334301,
      "lr": 0.00019995562646809908,
      "train_iou_t05": 0.7766990291260496,
      "train_pr_auc": 0.9036435460548728,
      "val_iou_t05": 0.7262459521364247,
      "val_iou_calibrated": 0.7338528398021329,
      "val_val_threshold": 0.3,
      "val_calib_best_iou": 0.7389302937307658,
      "val_pr_auc": 0.8541618057764874,
      "val_accuracy": 0.8338765337423313,
      "val_boundary_margin_mae": 0.0428873710334301,
      "val_interior_recall": 0.965574070683025,
      "val_bnd_pos_recall": 0.9303163796555867,
      "val_bnd_neg_spec": 0.8872509960159363,
      "val_jitter_pos_recall": 0.8989775051124744,
      "val_jitter_neg_spec": 0.9397590361445783,
      "val_exterior_spec": 0.5403772202893243,
      "val_best_iou": 0.7338528398021329,
      "val_best_threshold": 0.3
    },
    {
      "epoch": 1,
      "train_loss": 0.3378329308386838,
      "val_loss": 0.47548096809366824,
      "val_iou": 0.7484413965086698,
      "boundary_margin_mae": 0.03465825691819191,
      "lr": 0.00019966179940681097,
      "train_iou_t05": 0.7932189200500641,
      "train_pr_auc": 0.9129472064532873,
      "val_iou_t05": 0.7484413965086698,
      "val_iou_calibrated": 0.7484413965086698,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.7527987897124998,
      "val_pr_auc": 0.8731777292316512,
      "val_accuracy": 0.8452837423312883,
      "val_boundary_margin_mae": 0.03465825691819191,
      "val_interior_recall": 0.9304156747848379,
      "val_bnd_pos_recall": 0.9247096515818983,
      "val_bnd_neg_spec": 0.9131474103585657,
      "val_jitter_pos_recall": 0.905521472392638,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.6224134773850943,
      "val_best_iou": 0.7484413965086698,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 2,
      "train_loss": 0.3166648041274431,
      "val_loss": 0.4415128702897695,
      "val_iou": 0.7664970583557985,
      "boundary_margin_mae": 0.03250065818428993,
      "lr": 0.00019909360898998047,
      "train_iou_t05": 0.8106785790255667,
      "train_pr_auc": 0.9213729967932067,
      "val_iou_t05": 0.7664970583557985,
      "val_iou_calibrated": 0.7662940806044737,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.7719418960244058,
      "val_pr_auc": 0.8822801637902904,
      "val_accuracy": 0.8592312116564417,
      "val_boundary_margin_mae": 0.03250065818428993,
      "val_interior_recall": 0.9512909723493865,
      "val_bnd_pos_recall": 0.9267120544653584,
      "val_bnd_neg_spec": 0.9203187250996016,
      "val_jitter_pos_recall": 0.9104294478527607,
      "val_jitter_neg_spec": 0.9389558232931727,
      "val_exterior_spec": 0.6431056583043399,
      "val_best_iou": 0.7662940806044737,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 3,
      "train_loss": 0.30307041918080546,
      "val_loss": 0.41185797993341194,
      "val_iou": 0.7855409995147351,
      "boundary_margin_mae": 0.030681397765874863,
      "lr": 0.00019825263526814681,
      "train_iou_t05": 0.8278459221001016,
      "train_pr_auc": 0.9276758442885696,
      "val_iou_t05": 0.7855409995147351,
      "val_iou_calibrated": 0.7851067244422416,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.7878670187975153,
      "val_pr_auc": 0.890847047841593,
      "val_accuracy": 0.8728911042944786,
      "val_boundary_margin_mae": 0.030681397765874863,
      "val_interior_recall": 0.9566013550631752,
      "val_bnd_pos_recall": 0.9275130156187424,
      "val_bnd_neg_spec": 0.9195219123505977,
      "val_jitter_pos_recall": 0.9177914110429448,
      "val_jitter_neg_spec": 0.9385542168674699,
      "val_exterior_spec": 0.6879692364035891,
      "val_best_iou": 0.7851067244422416,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 4,
      "train_loss": 0.287139357381664,
      "val_loss": 0.363378710032943,
      "val_iou": 0.8030516626585286,
      "boundary_margin_mae": 0.02944490686058998,
      "lr": 0.00019714121686044758,
      "train_iou_t05": 0.8436403508770078,
      "train_pr_auc": 0.9343218656447313,
      "val_iou_t05": 0.8030516626585286,
      "val_iou_calibrated": 0.8039878058827713,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8080727504785571,
      "val_pr_auc": 0.9032460514492533,
      "val_accuracy": 0.8861675613496932,
      "val_boundary_margin_mae": 0.02944490686058998,
      "val_interior_recall": 0.959164988097418,
      "val_bnd_pos_recall": 0.9223067681217461,
      "val_bnd_neg_spec": 0.9243027888446215,
      "val_jitter_pos_recall": 0.9067484662576687,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.7454678630287493,
      "val_best_iou": 0.8039878058827713,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 5,
      "train_loss": 0.27489268605836187,
      "val_loss": 0.34762363004078695,
      "val_iou": 0.8126673360106415,
      "boundary_margin_mae": 0.03191624581813812,
      "lr": 0.00019576244445127717,
      "train_iou_t05": 0.8517210944393531,
      "train_pr_auc": 0.9365935445971894,
      "val_iou_t05": 0.8126673360106415,
      "val_iou_calibrated": 0.8126673360106415,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8127618433773112,
      "val_pr_auc": 0.9074031158594136,
      "val_accuracy": 0.8926859662576687,
      "val_boundary_margin_mae": 0.03191624581813812,
      "val_interior_recall": 0.946346822926204,
      "val_bnd_pos_recall": 0.9191029235082099,
      "val_bnd_neg_spec": 0.9294820717131475,
      "val_jitter_pos_recall": 0.9202453987730062,
      "val_jitter_neg_spec": 0.9385542168674699,
      "val_exterior_spec": 0.7767808093755723,
      "val_best_iou": 0.8126673360106415,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 6,
      "train_loss": 0.2677979265320885,
      "val_loss": 0.35129269873315055,
      "val_iou": 0.8143836181982648,
      "boundary_margin_mae": 0.029423099011182785,
      "lr": 0.00019412015219556671,
      "train_iou_t05": 0.8550470768555166,
      "train_pr_auc": 0.9393529989079327,
      "val_iou_t05": 0.8143836181982648,
      "val_iou_calibrated": 0.8168788466401806,
      "val_val_threshold": 0.65,
      "val_calib_best_iou": 0.8231886422975829,
      "val_pr_auc": 0.90718057989109,
      "val_accuracy": 0.8922546012269938,
      "val_boundary_margin_mae": 0.029423099011182785,
      "val_interior_recall": 0.9472624061527193,
      "val_bnd_pos_recall": 0.9199038846615939,
      "val_bnd_neg_spec": 0.9314741035856574,
      "val_jitter_pos_recall": 0.9075664621676892,
      "val_jitter_neg_spec": 0.9393574297188755,
      "val_exterior_spec": 0.7921626075810291,
      "val_best_iou": 0.8168788466401806,
      "val_best_threshold": 0.65
    },
    {
      "epoch": 7,
      "train_loss": 0.2576469692914218,
      "val_loss": 0.3030247588727396,
      "val_iou": 0.8304073632179282,
      "boundary_margin_mae": 0.02952745370566845,
      "lr": 0.00019221890705658633,
      "train_iou_t05": 0.8617995088187403,
      "train_pr_auc": 0.9449077249974622,
      "val_iou_t05": 0.8304073632179282,
      "val_iou_calibrated": 0.8302126577987943,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8354264138553036,
      "val_pr_auc": 0.9176907734332281,
      "val_accuracy": 0.9046203987730062,
      "val_boundary_margin_mae": 0.02952745370566845,
      "val_interior_recall": 0.9635597875846914,
      "val_bnd_pos_recall": 0.9279134961954345,
      "val_bnd_neg_spec": 0.9254980079681275,
      "val_jitter_pos_recall": 0.9079754601226994,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.8051638893975462,
      "val_best_iou": 0.8302126577987943,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 8,
      "train_loss": 0.248013401044377,
      "val_loss": 0.2928083187844678,
      "val_iou": 0.8341463414633432,
      "boundary_margin_mae": 0.0292693879455328,
      "lr": 0.00019006399610591978,
      "train_iou_t05": 0.8640928364202489,
      "train_pr_auc": 0.9500477587072977,
      "val_iou_t05": 0.8341463414633432,
      "val_iou_calibrated": 0.8329895134948571,
      "val_val_threshold": 0.5499999999999999,
      "val_calib_best_iou": 0.8384705098299867,
      "val_pr_auc": 0.9225248453432887,
      "val_accuracy": 0.9071127300613497,
      "val_boundary_margin_mae": 0.0292693879455328,
      "val_interior_recall": 0.943416956601355,
      "val_bnd_pos_recall": 0.9235082098518221,
      "val_bnd_neg_spec": 0.9290836653386454,
      "val_jitter_pos_recall": 0.9132924335378323,
      "val_jitter_neg_spec": 0.9393574297188755,
      "val_exterior_spec": 0.8348287859366417,
      "val_best_iou": 0.8329895134948571,
      "val_best_threshold": 0.5499999999999999
    },
    {
      "epoch": 9,
      "train_loss": 0.2411260458208642,
      "val_loss": 0.2866451540609914,
      "val_iou": 0.833347702388065,
      "boundary_margin_mae": 0.028055427595973015,
      "lr": 0.00018766141182092775,
      "train_iou_t05": 0.867971210076278,
      "train_pr_auc": 0.9568332670421267,
      "val_iou_t05": 0.833347702388065,
      "val_iou_calibrated": 0.8361935816028113,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8442185938019956,
      "val_pr_auc": 0.92511924893832,
      "val_accuracy": 0.9073523773006135,
      "val_boundary_margin_mae": 0.028055427595973015,
      "val_interior_recall": 0.9467130562168101,
      "val_bnd_pos_recall": 0.9231077292751302,
      "val_bnd_neg_spec": 0.9294820717131475,
      "val_jitter_pos_recall": 0.9284253578732107,
      "val_jitter_neg_spec": 0.9373493975903614,
      "val_exterior_spec": 0.8318989196117927,
      "val_best_iou": 0.8361935816028113,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 10,
      "train_loss": 0.2341669580821345,
      "val_loss": 0.2889236136761143,
      "val_iou": 0.8397364593136956,
      "boundary_margin_mae": 0.029504766687750816,
      "lr": 0.00018501783542058633,
      "train_iou_t05": 0.8691860465114335,
      "train_pr_auc": 0.9566057529130974,
      "val_iou_t05": 0.8397364593136956,
      "val_iou_calibrated": 0.8387343710129421,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8449202281556712,
      "val_pr_auc": 0.9272077985177801,
      "val_accuracy": 0.9102281441717791,
      "val_boundary_margin_mae": 0.029504766687750816,
      "val_interior_recall": 0.9679545870719648,
      "val_bnd_pos_recall": 0.9263115738886664,
      "val_bnd_neg_spec": 0.9294820717131475,
      "val_jitter_pos_recall": 0.9251533742331288,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.8126716718549716,
      "val_best_iou": 0.8387343710129421,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 11,
      "train_loss": 0.23079079657964816,
      "val_loss": 0.2811796256760231,
      "val_iou": 0.8374622356494745,
      "boundary_margin_mae": 0.028453681617975235,
      "lr": 0.00018214061828603987,
      "train_iou_t05": 0.8693693693691735,
      "train_pr_auc": 0.9594590276925175,
      "val_iou_t05": 0.8374622356494745,
      "val_iou_calibrated": 0.8374622356494745,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8459217034117825,
      "val_pr_auc": 0.9268402851146957,
      "val_accuracy": 0.9097488496932515,
      "val_boundary_margin_mae": 0.028453681617975235,
      "val_interior_recall": 0.9401208569859,
      "val_bnd_pos_recall": 0.9255106127352823,
      "val_bnd_neg_spec": 0.9310756972111554,
      "val_jitter_pos_recall": 0.9231083844580777,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.8434352682658853,
      "val_best_iou": 0.8374622356494745,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 12,
      "train_loss": 0.2259546705556534,
      "val_loss": 0.27951801942076193,
      "val_iou": 0.8391356542616327,
      "boundary_margin_mae": 0.028475221246480942,
      "lr": 0.0001790377615175359,
      "train_iou_t05": 0.8687275985661136,
      "train_pr_auc": 0.9591834397921385,
      "val_iou_t05": 0.8391356542616327,
      "val_iou_calibrated": 0.8391356542616327,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8476937192425683,
      "val_pr_auc": 0.9287645821463303,
      "val_accuracy": 0.9100843558282209,
      "val_boundary_margin_mae": 0.028475221246480942,
      "val_interior_recall": 0.9613623878410548,
      "val_bnd_pos_recall": 0.9219062875450541,
      "val_bnd_neg_spec": 0.9282868525896414,
      "val_jitter_pos_recall": 0.9137014314928426,
      "val_jitter_neg_spec": 0.9401606425702811,
      "val_exterior_spec": 0.829701519868156,
      "val_best_iou": 0.8391356542616327,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 13,
      "train_loss": 0.22221848132459163,
      "val_loss": 0.28477377924261815,
      "val_iou": 0.84187924074544,
      "boundary_margin_mae": 0.02745364047586918,
      "lr": 0.0001757178936845907,
      "train_iou_t05": 0.8721686476786561,
      "train_pr_auc": 0.9631027238420093,
      "val_iou_t05": 0.84187924074544,
      "val_iou_calibrated": 0.8401105067771009,
      "val_val_threshold": 0.5499999999999999,
      "val_calib_best_iou": 0.8486936066537134,
      "val_pr_auc": 0.926336810409301,
      "val_accuracy": 0.9117618865030674,
      "val_boundary_margin_mae": 0.02745364047586918,
      "val_interior_recall": 0.943416956601355,
      "val_bnd_pos_recall": 0.9243091710052063,
      "val_bnd_neg_spec": 0.9290836653386454,
      "val_jitter_pos_recall": 0.9288343558282208,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8449002014283098,
      "val_best_iou": 0.8401105067771009,
      "val_best_threshold": 0.5499999999999999
    },
    {
      "epoch": 14,
      "train_loss": 0.21839161575785287,
      "val_loss": 0.2741065655537554,
      "val_iou": 0.8373645426960695,
      "boundary_margin_mae": 0.029503893107175827,
      "lr": 0.0001721902468312585,
      "train_iou_t05": 0.8732426303852894,
      "train_pr_auc": 0.9640420293987959,
      "val_iou_t05": 0.8373645426960695,
      "val_iou_calibrated": 0.8414309054948111,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8484899048889664,
      "val_pr_auc": 0.9285925814980208,
      "val_accuracy": 0.9100843558282209,
      "val_boundary_margin_mae": 0.029503893107175827,
      "val_interior_recall": 0.9562351217725691,
      "val_bnd_pos_recall": 0.9247096515818983,
      "val_bnd_neg_spec": 0.9286852589641434,
      "val_jitter_pos_recall": 0.9218813905930471,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8366599523896722,
      "val_best_iou": 0.8414309054948111,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 15,
      "train_loss": 0.21504481911916898,
      "val_loss": 0.28616084412219667,
      "val_iou": 0.8412739290925537,
      "boundary_margin_mae": 0.02902509644627571,
      "lr": 0.00016846463080323108,
      "train_iou_t05": 0.8761519442569394,
      "train_pr_auc": 0.9639130782984315,
      "val_iou_t05": 0.8412739290925537,
      "val_iou_calibrated": 0.8412739290925537,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8492328218811833,
      "val_pr_auc": 0.927169093033165,
      "val_accuracy": 0.9113784509202454,
      "val_boundary_margin_mae": 0.02902509644627571,
      "val_interior_recall": 0.9555026551913569,
      "val_bnd_pos_recall": 0.9247096515818983,
      "val_bnd_neg_spec": 0.9314741035856574,
      "val_jitter_pos_recall": 0.9296523517382413,
      "val_jitter_neg_spec": 0.9373493975903614,
      "val_exterior_spec": 0.8318989196117927,
      "val_best_iou": 0.8412739290925537,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 16,
      "train_loss": 0.21260045505308625,
      "val_loss": 0.2768591938744105,
      "val_iou": 0.8427228021740278,
      "boundary_margin_mae": 0.027957044541835785,
      "lr": 0.0001645514059681589,
      "train_iou_t05": 0.8779882724400364,
      "train_pr_auc": 0.9655838362731162,
      "val_iou_t05": 0.8427228021740278,
      "val_iou_calibrated": 0.8435864435863711,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.851196930519572,
      "val_pr_auc": 0.9285860182210147,
      "val_accuracy": 0.9126246165644172,
      "val_boundary_margin_mae": 0.027957044541835785,
      "val_interior_recall": 0.9598974546786303,
      "val_bnd_pos_recall": 0.9291149379255106,
      "val_bnd_neg_spec": 0.9298804780876494,
      "val_jitter_pos_recall": 0.9284253578732107,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8313495696758836,
      "val_best_iou": 0.8435864435863711,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 17,
      "train_loss": 0.21016921483044665,
      "val_loss": 0.27698924156689325,
      "val_iou": 0.8408286577470141,
      "boundary_margin_mae": 0.02836100198328495,
      "lr": 0.00016046145440505528,
      "train_iou_t05": 0.8819538670282944,
      "train_pr_auc": 0.9671780242047372,
      "val_iou_t05": 0.8408286577470141,
      "val_iou_calibrated": 0.8408286577470141,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8497148607849002,
      "val_pr_auc": 0.9275275596791609,
      "val_accuracy": 0.9116180981595092,
      "val_boundary_margin_mae": 0.02836100198328495,
      "val_interior_recall": 0.9503753891228712,
      "val_bnd_pos_recall": 0.9231077292751302,
      "val_bnd_neg_spec": 0.9294820717131475,
      "val_jitter_pos_recall": 0.9186094069529652,
      "val_jitter_neg_spec": 0.9381526104417671,
      "val_exterior_spec": 0.8441677348470976,
      "val_best_iou": 0.8408286577470141,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 18,
      "train_loss": 0.20653370465084867,
      "val_loss": 0.27160405253713393,
      "val_iou": 0.8392252236601373,
      "boundary_margin_mae": 0.029288148507475853,
      "lr": 0.00015620614964290114,
      "train_iou_t05": 0.8808278371615007,
      "train_pr_auc": 0.9672052364661232,
      "val_iou_t05": 0.8392252236601373,
      "val_iou_calibrated": 0.8411704135918275,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8497124281069559,
      "val_pr_auc": 0.9274010936218018,
      "val_accuracy": 0.9112825920245399,
      "val_boundary_margin_mae": 0.029288148507475853,
      "val_interior_recall": 0.9631935542940854,
      "val_bnd_pos_recall": 0.9231077292751302,
      "val_bnd_neg_spec": 0.9250996015936255,
      "val_jitter_pos_recall": 0.9153374233128835,
      "val_jitter_neg_spec": 0.9373493975903614,
      "val_exterior_spec": 0.8339132027101264,
      "val_best_iou": 0.8411704135918275,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 19,
      "train_loss": 0.20478635912583953,
      "val_loss": 0.27943174478847976,
      "val_iou": 0.844275731170663,
      "boundary_margin_mae": 0.028103552758693695,
      "lr": 0.00015179732503260243,
      "train_iou_t05": 0.8811992786291971,
      "train_pr_auc": 0.9680971409335551,
      "val_iou_t05": 0.844275731170663,
      "val_iou_calibrated": 0.8443206172309606,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8511364582465364,
      "val_pr_auc": 0.9264274027105357,
      "val_accuracy": 0.9134873466257669,
      "val_boundary_margin_mae": 0.028103552758693695,
      "val_interior_recall": 0.9641091375206006,
      "val_bnd_pos_recall": 0.9275130156187424,
      "val_bnd_neg_spec": 0.9250996015936255,
      "val_jitter_pos_recall": 0.9276073619631902,
      "val_jitter_neg_spec": 0.9373493975903614,
      "val_exterior_spec": 0.8318989196117927,
      "val_best_iou": 0.8443206172309606,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 20,
      "train_loss": 0.2021904184512515,
      "val_loss": 0.27567216876498113,
      "val_iou": 0.8417590027700103,
      "boundary_margin_mae": 0.029568320140242577,
      "lr": 0.00014724724084025365,
      "train_iou_t05": 0.8836050724635679,
      "train_pr_auc": 0.9696469851857676,
      "val_iou_t05": 0.8417590027700103,
      "val_iou_calibrated": 0.8417884780738742,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8459551778721282,
      "val_pr_auc": 0.9275949535517191,
      "val_accuracy": 0.9123849693251533,
      "val_boundary_margin_mae": 0.029568320140242577,
      "val_interior_recall": 0.959164988097418,
      "val_bnd_pos_recall": 0.9223067681217461,
      "val_bnd_neg_spec": 0.9211155378486056,
      "val_jitter_pos_recall": 0.9198364008179959,
      "val_jitter_neg_spec": 0.934136546184739,
      "val_exterior_spec": 0.8416041018128548,
      "val_best_iou": 0.8417884780738742,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 21,
      "train_loss": 0.19793493100992196,
      "val_loss": 0.2769015661779265,
      "val_iou": 0.840961986035614,
      "boundary_margin_mae": 0.030707500874996185,
      "lr": 0.00014256855015321493,
      "train_iou_t05": 0.8812485862924946,
      "train_pr_auc": 0.9702736400869238,
      "val_iou_t05": 0.840961986035614,
      "val_iou_calibrated": 0.840961986035614,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8459027370887381,
      "val_pr_auc": 0.9269910387072336,
      "val_accuracy": 0.9115701687116564,
      "val_boundary_margin_mae": 0.030707500874996185,
      "val_interior_recall": 0.9608130379051456,
      "val_bnd_pos_recall": 0.9154985983179815,
      "val_bnd_neg_spec": 0.9258964143426295,
      "val_jitter_pos_recall": 0.90920245398773,
      "val_jitter_neg_spec": 0.9385542168674699,
      "val_exterior_spec": 0.8427028016846732,
      "val_best_iou": 0.840961986035614,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 22,
      "train_loss": 0.19632512041144826,
      "val_loss": 0.2748819728585266,
      "val_iou": 0.8435662400552711,
      "boundary_margin_mae": 0.029828427359461784,
      "lr": 0.0001377742636938141,
      "train_iou_t05": 0.8852421910364677,
      "train_pr_auc": 0.9711159225426063,
      "val_iou_t05": 0.8435662400552711,
      "val_iou_calibrated": 0.8435662400552711,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.847390903760713,
      "val_pr_auc": 0.9291145256672589,
      "val_accuracy": 0.9132956288343558,
      "val_boundary_margin_mae": 0.029828427359461784,
      "val_interior_recall": 0.9597143380333272,
      "val_bnd_pos_recall": 0.9154985983179815,
      "val_bnd_neg_spec": 0.9306772908366534,
      "val_jitter_pos_recall": 0.9112474437627812,
      "val_jitter_neg_spec": 0.9397590361445783,
      "val_exterior_spec": 0.8467313678813404,
      "val_best_iou": 0.8435662400552711,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 23,
      "train_loss": 0.19285958696167477,
      "val_loss": 0.2716760208264335,
      "val_iou": 0.8397746967070329,
      "boundary_margin_mae": 0.028948737308382988,
      "lr": 0.00013287771363852007,
      "train_iou_t05": 0.8883561643833587,
      "train_pr_auc": 0.9706781375076257,
      "val_iou_t05": 0.8397746967070329,
      "val_iou_calibrated": 0.8435327439704022,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8480453446694299,
      "val_pr_auc": 0.9276776298028587,
      "val_accuracy": 0.9113784509202454,
      "val_boundary_margin_mae": 0.028948737308382988,
      "val_interior_recall": 0.9694195202343893,
      "val_bnd_pos_recall": 0.9162995594713657,
      "val_bnd_neg_spec": 0.9290836653386454,
      "val_jitter_pos_recall": 0.9186094069529652,
      "val_jitter_neg_spec": 0.9377510040160643,
      "val_exterior_spec": 0.8324482695477019,
      "val_best_iou": 0.8435327439704022,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 24,
      "train_loss": 0.18987400290859535,
      "val_loss": 0.27667114249211044,
      "val_iou": 0.8416450216449487,
      "boundary_margin_mae": 0.029469676315784454,
      "lr": 0.00012789251654320177,
      "train_iou_t05": 0.8877737226275346,
      "train_pr_auc": 0.9720569529103292,
      "val_iou_t05": 0.8416450216449487,
      "val_iou_calibrated": 0.8407321871524386,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8447617466378179,
      "val_pr_auc": 0.9269243483981786,
      "val_accuracy": 0.9123370398773006,
      "val_boundary_margin_mae": 0.029469676315784454,
      "val_interior_recall": 0.9701519868156016,
      "val_bnd_pos_recall": 0.9167000400480577,
      "val_bnd_neg_spec": 0.9270916334661354,
      "val_jitter_pos_recall": 0.9169734151329243,
      "val_jitter_neg_spec": 0.9365461847389558,
      "val_exterior_spec": 0.826588536898004,
      "val_best_iou": 0.8407321871524386,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 25,
      "train_loss": 0.18694552392712244,
      "val_loss": 0.27133198594394475,
      "val_iou": 0.839429312580991,
      "boundary_margin_mae": 0.03063729777932167,
      "lr": 0.00012283253547757143,
      "train_iou_t05": 0.8896975210368683,
      "train_pr_auc": 0.9727201070049699,
      "val_iou_t05": 0.839429312580991,
      "val_iou_calibrated": 0.8393332760545718,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8416097317113113,
      "val_pr_auc": 0.9278642870842463,
      "val_accuracy": 0.9109950153374233,
      "val_boundary_margin_mae": 0.03063729777932167,
      "val_interior_recall": 0.9661234206189343,
      "val_bnd_pos_recall": 0.9134961954345214,
      "val_bnd_neg_spec": 0.9179282868525896,
      "val_jitter_pos_recall": 0.9047034764826176,
      "val_jitter_neg_spec": 0.9321285140562249,
      "val_exterior_spec": 0.842336568394067,
      "val_best_iou": 0.8393332760545718,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 26,
      "train_loss": 0.1834183771603389,
      "val_loss": 0.27668626666901797,
      "val_iou": 0.8408678364594311,
      "boundary_margin_mae": 0.030143465846776962,
      "lr": 0.00011771184147411064,
      "train_iou_t05": 0.8941015713958337,
      "train_pr_auc": 0.974463480314945,
      "val_iou_t05": 0.8408678364594311,
      "val_iou_calibrated": 0.8413899879579527,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8480314302431791,
      "val_pr_auc": 0.9261310512858995,
      "val_accuracy": 0.9117618865030674,
      "val_boundary_margin_mae": 0.030143465846776962,
      "val_interior_recall": 0.9593481047427211,
      "val_bnd_pos_recall": 0.9183019623548258,
      "val_bnd_neg_spec": 0.9203187250996016,
      "val_jitter_pos_recall": 0.9202453987730062,
      "val_jitter_neg_spec": 0.9337349397590361,
      "val_exterior_spec": 0.8428859183299762,
      "val_best_iou": 0.8413899879579527,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 27,
      "train_loss": 0.17995288841404902,
      "val_loss": 0.2718750529516336,
      "val_iou": 0.8354474070238298,
      "boundary_margin_mae": 0.03261638060212135,
      "lr": 0.00011254467439868386,
      "train_iou_t05": 0.8973774230328625,
      "train_pr_auc": 0.9740686471175972,
      "val_iou_t05": 0.8354474070238298,
      "val_iou_calibrated": 0.8354474070238298,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8412312337498246,
      "val_pr_auc": 0.9265497577229334,
      "val_accuracy": 0.9085985429447853,
      "val_boundary_margin_mae": 0.03261638060212135,
      "val_interior_recall": 0.9531221388024171,
      "val_bnd_pos_recall": 0.9046856227472968,
      "val_bnd_neg_spec": 0.9155378486055777,
      "val_jitter_pos_recall": 0.907157464212679,
      "val_jitter_neg_spec": 0.9257028112449799,
      "val_exterior_spec": 0.8555209668558872,
      "val_best_iou": 0.8354474070238298,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 28,
      "train_loss": 0.17708311704462473,
      "val_loss": 0.2736314129822942,
      "val_iou": 0.8360245229254092,
      "boundary_margin_mae": 0.031436797231435776,
      "lr": 0.00010734540335165195,
      "train_iou_t05": 0.8963233614978542,
      "train_pr_auc": 0.9748712190606427,
      "val_iou_t05": 0.8360245229254092,
      "val_iou_calibrated": 0.8355189851717806,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.842404006677726,
      "val_pr_auc": 0.9263835823358265,
      "val_accuracy": 0.9089819785276073,
      "val_boundary_margin_mae": 0.031436797231435776,
      "val_interior_recall": 0.965574070683025,
      "val_bnd_pos_recall": 0.9058870644773729,
      "val_bnd_neg_spec": 0.9111553784860558,
      "val_jitter_pos_recall": 0.9051124744376278,
      "val_jitter_neg_spec": 0.929718875502008,
      "val_exterior_spec": 0.8414209851675517,
      "val_best_iou": 0.8355189851717806,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 29,
      "train_loss": 0.1729497396164394,
      "val_loss": 0.2756427009369316,
      "val_iou": 0.8366762177649703,
      "boundary_margin_mae": 0.03147752955555916,
      "lr": 0.0001021284867096046,
      "train_iou_t05": 0.8987370838115042,
      "train_pr_auc": 0.9762246114280985,
      "val_iou_t05": 0.8366762177649703,
      "val_iou_calibrated": 0.837796086508682,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8419077795379937,
      "val_pr_auc": 0.9262462436872132,
      "val_accuracy": 0.909844708588957,
      "val_boundary_margin_mae": 0.03147752955555916,
      "val_interior_recall": 0.9650247207471159,
      "val_bnd_pos_recall": 0.9054865839006808,
      "val_bnd_neg_spec": 0.9203187250996016,
      "val_jitter_pos_recall": 0.9124744376278119,
      "val_jitter_neg_spec": 0.9333333333333333,
      "val_exterior_spec": 0.8383080021973998,
      "val_best_iou": 0.837796086508682,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 30,
      "train_loss": 0.16907013448153863,
      "val_loss": 0.2848104165509914,
      "val_iou": 0.8310197086545988,
      "boundary_margin_mae": 0.03327419236302376,
      "lr": 9.6908431918829e-05,
      "train_iou_t05": 0.9012289485660214,
      "train_pr_auc": 0.9771676113689298,
      "val_iou_t05": 0.8310197086545988,
      "val_iou_calibrated": 0.8296149539149944,
      "val_val_threshold": 0.5499999999999999,
      "val_calib_best_iou": 0.8342107462186985,
      "val_pr_auc": 0.9234117021879628,
      "val_accuracy": 0.9054831288343558,
      "val_boundary_margin_mae": 0.03327419236302376,
      "val_interior_recall": 0.9478117560886284,
      "val_bnd_pos_recall": 0.9002803364036844,
      "val_bnd_neg_spec": 0.9063745019920318,
      "val_jitter_pos_recall": 0.9026584867075664,
      "val_jitter_neg_spec": 0.9220883534136546,
      "val_exterior_spec": 0.8577183665995239,
      "val_best_iou": 0.8296149539149944,
      "val_best_threshold": 0.5499999999999999
    },
    {
      "epoch": 31,
      "train_loss": 0.16661515243630587,
      "val_loss": 0.2769289729029511,
      "val_iou": 0.8348194961474474,
      "boundary_margin_mae": 0.03206133097410202,
      "lr": 9.169975515232349e-05,
      "train_iou_t05": 0.9026122823096006,
      "train_pr_auc": 0.9781060539528112,
      "val_iou_t05": 0.8348194961474474,
      "val_iou_calibrated": 0.8358877217150993,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8395989974936642,
      "val_pr_auc": 0.9260111896804978,
      "val_accuracy": 0.9085506134969326,
      "val_boundary_margin_mae": 0.03206133097410202,
      "val_interior_recall": 0.9571507049990844,
      "val_bnd_pos_recall": 0.9046856227472968,
      "val_bnd_neg_spec": 0.9135458167330678,
      "val_jitter_pos_recall": 0.9087934560327199,
      "val_jitter_neg_spec": 0.9281124497991968,
      "val_exterior_spec": 0.8507599340780077,
      "val_best_iou": 0.8358877217150993,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 32,
      "train_loss": 0.16235031911120978,
      "val_loss": 0.28317075883086007,
      "val_iou": 0.8312338222604977,
      "boundary_margin_mae": 0.03333672136068344,
      "lr": 8.651694094254327e-05,
      "train_iou_t05": 0.9042039972430727,
      "train_pr_auc": 0.9779162195347735,
      "val_iou_t05": 0.8312338222604977,
      "val_iou_calibrated": 0.8320171673819028,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8329172907305015,
      "val_pr_auc": 0.9238077922222083,
      "val_accuracy": 0.90625,
      "val_boundary_margin_mae": 0.03333672136068344,
      "val_interior_recall": 0.9615455044863578,
      "val_bnd_pos_recall": 0.89547456948338,
      "val_bnd_neg_spec": 0.9123505976095617,
      "val_jitter_pos_recall": 0.9022494887525563,
      "val_jitter_neg_spec": 0.9248995983935743,
      "val_exterior_spec": 0.8461820179454312,
      "val_best_iou": 0.8320171673819028,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 33,
      "train_loss": 0.15947994249483693,
      "val_loss": 0.28517002660514545,
      "val_iou": 0.8262399176247789,
      "boundary_margin_mae": 0.03601361811161041,
      "lr": 8.137440190213364e-05,
      "train_iou_t05": 0.9081539166282848,
      "train_pr_auc": 0.9788790013295936,
      "val_iou_t05": 0.8262399176247789,
      "val_iou_calibrated": 0.8262399176247789,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8295407185128923,
      "val_pr_auc": 0.9200937705263066,
      "val_accuracy": 0.9029428680981595,
      "val_boundary_margin_mae": 0.03601361811161041,
      "val_interior_recall": 0.9522065555759018,
      "val_bnd_pos_recall": 0.895074088906688,
      "val_bnd_neg_spec": 0.895617529880478,
      "val_jitter_pos_recall": 0.8973415132924335,
      "val_jitter_neg_spec": 0.9144578313253012,
      "val_exterior_spec": 0.857901483244827,
      "val_best_iou": 0.8262399176247789,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 34,
      "train_loss": 0.15573225102738963,
      "val_loss": 0.27999427271506444,
      "val_iou": 0.8297486395438516,
      "boundary_margin_mae": 0.034199949353933334,
      "lr": 7.62864386446616e-05,
      "train_iou_t05": 0.9091536338544366,
      "train_pr_auc": 0.9800384847665881,
      "val_iou_t05": 0.8297486395438516,
      "val_iou_calibrated": 0.8297486395438516,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8326349312771679,
      "val_pr_auc": 0.9237159610825353,
      "val_accuracy": 0.9055310582822086,
      "val_boundary_margin_mae": 0.034199949353933334,
      "val_interior_recall": 0.9514740889946897,
      "val_bnd_pos_recall": 0.8934721665999199,
      "val_bnd_neg_spec": 0.9083665338645418,
      "val_jitter_pos_recall": 0.8912065439672802,
      "val_jitter_neg_spec": 0.9273092369477912,
      "val_exterior_spec": 0.8602819996337667,
      "val_best_iou": 0.8297486395438516,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 35,
      "train_loss": 0.1526965437516012,
      "val_loss": 0.28537737056236273,
      "val_iou": 0.830072090628147,
      "boundary_margin_mae": 0.034432101994752884,
      "lr": 7.126720001679918e-05,
      "train_iou_t05": 0.9095074455897113,
      "train_pr_auc": 0.9800439515341203,
      "val_iou_t05": 0.830072090628147,
      "val_iou_calibrated": 0.830072090628147,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8346285618784294,
      "val_pr_auc": 0.9199967377918363,
      "val_accuracy": 0.9050996932515337,
      "val_boundary_margin_mae": 0.034432101994752884,
      "val_interior_recall": 0.9544039553195386,
      "val_bnd_pos_recall": 0.9050861033239888,
      "val_bnd_neg_spec": 0.899601593625498,
      "val_jitter_pos_recall": 0.8997955010224948,
      "val_jitter_neg_spec": 0.9180722891566265,
      "val_exterior_spec": 0.854788500274675,
      "val_best_iou": 0.830072090628147,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 36,
      "train_loss": 0.15019072856356844,
      "val_loss": 0.28801984383694135,
      "val_iou": 0.8323709385503031,
      "boundary_margin_mae": 0.03381139412522316,
      "lr": 6.63306437525477e-05,
      "train_iou_t05": 0.9107019562713669,
      "train_pr_auc": 0.9806715076626531,
      "val_iou_t05": 0.8323709385503031,
      "val_iou_calibrated": 0.8333905284831131,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8337768168966544,
      "val_pr_auc": 0.9239116643809645,
      "val_accuracy": 0.9067772239263804,
      "val_boundary_margin_mae": 0.03381139412522316,
      "val_interior_recall": 0.9670390038454495,
      "val_bnd_pos_recall": 0.8958750500600721,
      "val_bnd_neg_spec": 0.9123505976095617,
      "val_jitter_pos_recall": 0.898159509202454,
      "val_jitter_neg_spec": 0.9281124497991968,
      "val_exterior_spec": 0.8436183849111885,
      "val_best_iou": 0.8333905284831131,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 37,
      "train_loss": 0.14690864084003982,
      "val_loss": 0.28228586640460107,
      "val_iou": 0.8342403040510638,
      "boundary_margin_mae": 0.032760001718997955,
      "lr": 6.149049765891677e-05,
      "train_iou_t05": 0.9128831527999739,
      "train_pr_auc": 0.9810614626399767,
      "val_iou_t05": 0.8342403040510638,
      "val_iou_calibrated": 0.836013686911819,
      "val_val_threshold": 0.39999999999999997,
      "val_calib_best_iou": 0.8406748670212066,
      "val_pr_auc": 0.9236357906930784,
      "val_accuracy": 0.9080233895705522,
      "val_boundary_margin_mae": 0.032760001718997955,
      "val_interior_recall": 0.9657571873283282,
      "val_bnd_pos_recall": 0.9094913896676011,
      "val_bnd_neg_spec": 0.9087649402390439,
      "val_jitter_pos_recall": 0.9112474437627812,
      "val_jitter_neg_spec": 0.9216867469879518,
      "val_exterior_spec": 0.8419703351034609,
      "val_best_iou": 0.836013686911819,
      "val_best_threshold": 0.39999999999999997
    },
    {
      "epoch": 38,
      "train_loss": 0.1443316887216025,
      "val_loss": 0.28223848227534437,
      "val_iou": 0.8363384188626182,
      "boundary_margin_mae": 0.03283388167619705,
      "lr": 5.6760221440995395e-05,
      "train_iou_t05": 0.910664819944388,
      "train_pr_auc": 0.9811655139249744,
      "val_iou_t05": 0.8363384188626182,
      "val_iou_calibrated": 0.8363384188626182,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8376831116348848,
      "val_pr_auc": 0.9238808122080827,
      "val_accuracy": 0.9095092024539877,
      "val_boundary_margin_mae": 0.03283388167619705,
      "val_interior_recall": 0.9529390221571141,
      "val_bnd_pos_recall": 0.9010812975570685,
      "val_bnd_neg_spec": 0.9195219123505977,
      "val_jitter_pos_recall": 0.8973415132924335,
      "val_jitter_neg_spec": 0.9337349397590361,
      "val_exterior_spec": 0.8597326496978576,
      "val_best_iou": 0.8363384188626182,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 39,
      "train_loss": 0.1424131790653773,
      "val_loss": 0.2982236680244855,
      "val_iou": 0.8283887468029985,
      "boundary_margin_mae": 0.03583219647407532,
      "lr": 5.215296927257321e-05,
      "train_iou_t05": 0.9119194692288006,
      "train_pr_auc": 0.9816924236355618,
      "val_iou_t05": 0.8283887468029985,
      "val_iou_calibrated": 0.8273498414873766,
      "val_val_threshold": 0.5499999999999999,
      "val_calib_best_iou": 0.8275575500705703,
      "val_pr_auc": 0.9054275124989657,
      "val_accuracy": 0.9035180214723927,
      "val_boundary_margin_mae": 0.03583219647407532,
      "val_interior_recall": 0.9490935726057499,
      "val_bnd_pos_recall": 0.9058870644773729,
      "val_bnd_neg_spec": 0.8912350597609562,
      "val_jitter_pos_recall": 0.9042944785276074,
      "val_jitter_neg_spec": 0.9112449799196787,
      "val_exterior_spec": 0.8582677165354331,
      "val_best_iou": 0.8273498414873766,
      "val_best_threshold": 0.5499999999999999
    },
    {
      "epoch": 40,
      "train_loss": 0.14002864921642655,
      "val_loss": 0.2928155567171121,
      "val_iou": 0.829333333333262,
      "boundary_margin_mae": 0.03464958444237709,
      "lr": 4.768155321639744e-05,
      "train_iou_t05": 0.9182636804431035,
      "train_pr_auc": 0.9824044195627621,
      "val_iou_t05": 0.829333333333262,
      "val_iou_calibrated": 0.8302387267903799,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.831492400963306,
      "val_pr_auc": 0.9141270385278969,
      "val_accuracy": 0.9049079754601227,
      "val_boundary_margin_mae": 0.03464958444237709,
      "val_interior_recall": 0.9611792711957516,
      "val_bnd_pos_recall": 0.9018822587104526,
      "val_bnd_neg_spec": 0.8952191235059761,
      "val_jitter_pos_recall": 0.9006134969325154,
      "val_jitter_neg_spec": 0.9192771084337349,
      "val_exterior_spec": 0.8498443508514923,
      "val_best_iou": 0.8302387267903799,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 41,
      "train_loss": 0.13774233842832212,
      "val_loss": 0.29574787392662133,
      "val_iou": 0.8288590604026133,
      "boundary_margin_mae": 0.0352008230984211,
      "lr": 4.3358407595788475e-05,
      "train_iou_t05": 0.9179759704249264,
      "train_pr_auc": 0.9827368776299865,
      "val_iou_t05": 0.8288590604026133,
      "val_iou_calibrated": 0.8294082384173307,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8311170212765266,
      "val_pr_auc": 0.9097978162821098,
      "val_accuracy": 0.9046683282208589,
      "val_boundary_margin_mae": 0.0352008230984211,
      "val_interior_recall": 0.9628273210034792,
      "val_bnd_pos_recall": 0.893872647176612,
      "val_bnd_neg_spec": 0.9023904382470119,
      "val_jitter_pos_recall": 0.8977505112474438,
      "val_jitter_neg_spec": 0.9168674698795181,
      "val_exterior_spec": 0.8494781175608863,
      "val_best_iou": 0.8294082384173307,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 42,
      "train_loss": 0.1360835913321814,
      "val_loss": 0.2937467427473817,
      "val_iou": 0.8297157622738303,
      "boundary_margin_mae": 0.03511941060423851,
      "lr": 3.919555441669089e-05,
      "train_iou_t05": 0.9194817214250532,
      "train_pr_auc": 0.9827986743294819,
      "val_iou_t05": 0.8297157622738303,
      "val_iou_calibrated": 0.8297157622738303,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8319243324683324,
      "val_pr_auc": 0.906833630490359,
      "val_accuracy": 0.905243481595092,
      "val_boundary_margin_mae": 0.03511941060423851,
      "val_interior_recall": 0.9516572056399927,
      "val_bnd_pos_recall": 0.8986784140969163,
      "val_bnd_neg_spec": 0.9039840637450199,
      "val_jitter_pos_recall": 0.8965235173824131,
      "val_jitter_neg_spec": 0.9184738955823293,
      "val_exterior_spec": 0.8602819996337667,
      "val_best_iou": 0.8297157622738303,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 43,
      "train_loss": 0.1349767385539129,
      "val_loss": 0.29808846679704853,
      "val_iou": 0.8287240458343388,
      "boundary_margin_mae": 0.03526096045970917,
      "lr": 3.52045699363159e-05,
      "train_iou_t05": 0.9204256303490814,
      "train_pr_auc": 0.9832477906529858,
      "val_iou_t05": 0.8287240458343388,
      "val_iou_calibrated": 0.8287240458343388,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8308116693136478,
      "val_pr_auc": 0.9070273308751579,
      "val_accuracy": 0.9047162576687117,
      "val_boundary_margin_mae": 0.03526096045970917,
      "val_interior_recall": 0.9523896722212049,
      "val_bnd_pos_recall": 0.894673608329996,
      "val_bnd_neg_spec": 0.9059760956175299,
      "val_jitter_pos_recall": 0.8932515337423312,
      "val_jitter_neg_spec": 0.9200803212851406,
      "val_exterior_spec": 0.8591832997619484,
      "val_best_iou": 0.8287240458343388,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 44,
      "train_loss": 0.13330849978310574,
      "val_loss": 0.2958037107914349,
      "val_iou": 0.8294519954981534,
      "boundary_margin_mae": 0.03428668528795242,
      "lr": 3.139655247134254e-05,
      "train_iou_t05": 0.9186369958273252,
      "train_pr_auc": 0.9829231933657387,
      "val_iou_t05": 0.8294519954981534,
      "val_iou_calibrated": 0.8311934510985927,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8305212161709964,
      "val_pr_auc": 0.9129984212591101,
      "val_accuracy": 0.9055789877300614,
      "val_boundary_margin_mae": 0.03428668528795242,
      "val_interior_recall": 0.958981871452115,
      "val_bnd_pos_recall": 0.893872647176612,
      "val_bnd_neg_spec": 0.9147410358565737,
      "val_jitter_pos_recall": 0.8903885480572598,
      "val_jitter_neg_spec": 0.9265060240963855,
      "val_exterior_spec": 0.8525911005310383,
      "val_best_iou": 0.8311934510985927,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 45,
      "train_loss": 0.13165244119783986,
      "val_loss": 0.2973787847276369,
      "val_iou": 0.8314500129387706,
      "boundary_margin_mae": 0.034261077642440796,
      "lr": 2.7782091535198955e-05,
      "train_iou_t05": 0.9207599629284242,
      "train_pr_auc": 0.9834626046451139,
      "val_iou_t05": 0.8314500129387706,
      "val_iou_calibrated": 0.8316440474144621,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8324856166096195,
      "val_pr_auc": 0.9053352589554826,
      "val_accuracy": 0.9063458588957055,
      "val_boundary_margin_mae": 0.034261077642440796,
      "val_interior_recall": 0.9562351217725691,
      "val_bnd_pos_recall": 0.9030837004405287,
      "val_bnd_neg_spec": 0.9039840637450199,
      "val_jitter_pos_recall": 0.901840490797546,
      "val_jitter_neg_spec": 0.9180722891566265,
      "val_exterior_spec": 0.8546053836293719,
      "val_best_iou": 0.8316440474144621,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 46,
      "train_loss": 0.13059509136534217,
      "val_loss": 0.3045446179185344,
      "val_iou": 0.8285420063502249,
      "boundary_margin_mae": 0.03551153466105461,
      "lr": 2.4371238390246793e-05,
      "train_iou_t05": 0.9232192414428946,
      "train_pr_auc": 0.9833711595187102,
      "val_iou_t05": 0.8285420063502249,
      "val_iou_calibrated": 0.8285420063502249,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8301933977991972,
      "val_pr_auc": 0.8991939827676793,
      "val_accuracy": 0.9042369631901841,
      "val_boundary_margin_mae": 0.03551153466105461,
      "val_interior_recall": 0.9560520051272661,
      "val_bnd_pos_recall": 0.8966760112134562,
      "val_bnd_neg_spec": 0.8980079681274901,
      "val_jitter_pos_recall": 0.8977505112474438,
      "val_jitter_neg_spec": 0.9156626506024096,
      "val_exterior_spec": 0.8564365500824025,
      "val_best_iou": 0.8285420063502249,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 47,
      "train_loss": 0.1284950774817886,
      "val_loss": 0.3040054655919819,
      "val_iou": 0.8302195436934283,
      "boundary_margin_mae": 0.03484867513179779,
      "lr": 2.117347809676009e-05,
      "train_iou_t05": 0.9203887988889329,
      "train_pr_auc": 0.9836548185841019,
      "val_iou_t05": 0.8302195436934283,
      "val_iou_calibrated": 0.8309340188516854,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8324040941998142,
      "val_pr_auc": 0.9035471299575036,
      "val_accuracy": 0.9054831288343558,
      "val_boundary_margin_mae": 0.03484867513179779,
      "val_interior_recall": 0.9613623878410548,
      "val_bnd_pos_recall": 0.9014817781337605,
      "val_bnd_neg_spec": 0.9035856573705179,
      "val_jitter_pos_recall": 0.898159509202454,
      "val_jitter_neg_spec": 0.9172690763052209,
      "val_exterior_spec": 0.8500274674967955,
      "val_best_iou": 0.8309340188516854,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 48,
      "train_loss": 0.12819889961857273,
      "val_loss": 0.30266015373754906,
      "val_iou": 0.8295006006520654,
      "boundary_margin_mae": 0.034722208976745605,
      "lr": 1.81977031364249e-05,
      "train_iou_t05": 0.9232014804531752,
      "train_pr_auc": 0.9839549523532112,
      "val_iou_t05": 0.8295006006520654,
      "val_iou_calibrated": 0.8295006006520654,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8322768974144427,
      "val_pr_auc": 0.9009361274381305,
      "val_accuracy": 0.9047641871165644,
      "val_boundary_margin_mae": 0.034722208976745605,
      "val_interior_recall": 0.9551364219007508,
      "val_bnd_pos_recall": 0.9022827392871445,
      "val_bnd_neg_spec": 0.895617529880478,
      "val_jitter_pos_recall": 0.8989775051124744,
      "val_jitter_neg_spec": 0.9160642570281124,
      "val_exterior_spec": 0.8571690166636147,
      "val_best_iou": 0.8295006006520654,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 49,
      "train_loss": 0.12756486453910382,
      "val_loss": 0.30881743044016186,
      "val_iou": 0.8277137450541177,
      "boundary_margin_mae": 0.035501182079315186,
      "lr": 1.5452188683709577e-05,
      "train_iou_t05": 0.9226314570301314,
      "train_pr_auc": 0.9839615095945234,
      "val_iou_t05": 0.8277137450541177,
      "val_iou_calibrated": 0.8277012575925375,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8279301745635221,
      "val_pr_auc": 0.8991548754029537,
      "val_accuracy": 0.9039973159509203,
      "val_boundary_margin_mae": 0.035501182079315186,
      "val_interior_recall": 0.9620948544222669,
      "val_bnd_pos_recall": 0.8970764917901481,
      "val_bnd_neg_spec": 0.899203187250996,
      "val_jitter_pos_recall": 0.8920245398773006,
      "val_jitter_neg_spec": 0.9152610441767068,
      "val_exterior_spec": 0.8494781175608863,
      "val_best_iou": 0.8277012575925375,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 50,
      "train_loss": 0.12677905019319022,
      "val_loss": 0.30930770589897727,
      "val_iou": 0.828249849643369,
      "boundary_margin_mae": 0.03545474633574486,
      "lr": 1.2944569593871897e-05,
      "train_iou_t05": 0.9243230733624336,
      "train_pr_auc": 0.9840632353277867,
      "val_iou_t05": 0.828249849643369,
      "val_iou_calibrated": 0.828249849643369,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8298724043031582,
      "val_pr_auc": 0.898506675039404,
      "val_accuracy": 0.9041890337423313,
      "val_boundary_margin_mae": 0.03545474633574486,
      "val_interior_recall": 0.955868888481963,
      "val_bnd_pos_recall": 0.8958750500600721,
      "val_bnd_neg_spec": 0.9023904382470119,
      "val_jitter_pos_recall": 0.8928425357873211,
      "val_jitter_neg_spec": 0.9172690763052209,
      "val_exterior_spec": 0.8562534334370995,
      "val_best_iou": 0.828249849643369,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 51,
      "train_loss": 0.12629405535779012,
      "val_loss": 0.31095756765927185,
      "val_iou": 0.8286205414696323,
      "boundary_margin_mae": 0.03552113100886345,
      "lr": 1.068181917159557e-05,
      "train_iou_t05": 0.924947880472336,
      "train_pr_auc": 0.9841173259349488,
      "val_iou_t05": 0.8286205414696323,
      "val_iou_calibrated": 0.8286205414696323,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8285356695869146,
      "val_pr_auc": 0.8956345851467793,
      "val_accuracy": 0.9044286809815951,
      "val_boundary_margin_mae": 0.03552113100886345,
      "val_interior_recall": 0.9553195385460538,
      "val_bnd_pos_recall": 0.8966760112134562,
      "val_bnd_neg_spec": 0.8980079681274901,
      "val_jitter_pos_recall": 0.8936605316973415,
      "val_jitter_neg_spec": 0.9168674698795181,
      "val_exterior_spec": 0.8591832997619484,
      "val_best_iou": 0.8286205414696323,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 52,
      "train_loss": 0.12507556257527913,
      "val_loss": 0.3084468988370727,
      "val_iou": 0.8293269230768519,
      "boundary_margin_mae": 0.03517809137701988,
      "lr": 8.670229779297875e-06,
      "train_iou_t05": 0.9250173570930513,
      "train_pr_auc": 0.9840254957885592,
      "val_iou_t05": 0.8293269230768519,
      "val_iou_calibrated": 0.8293269230768519,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.831443814604799,
      "val_pr_auc": 0.8990836257502327,
      "val_accuracy": 0.9047162576687117,
      "val_boundary_margin_mae": 0.03517809137701988,
      "val_interior_recall": 0.9582494048709027,
      "val_bnd_pos_recall": 0.8986784140969163,
      "val_bnd_neg_spec": 0.9015936254980079,
      "val_jitter_pos_recall": 0.8928425357873211,
      "val_jitter_neg_spec": 0.9168674698795181,
      "val_exterior_spec": 0.855154733565281,
      "val_best_iou": 0.8293269230768519,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 53,
      "train_loss": 0.12485008871435088,
      "val_loss": 0.30929240392485285,
      "val_iou": 0.8289349265021208,
      "boundary_margin_mae": 0.03521474823355675,
      "lr": 6.915395339033419e-06,
      "train_iou_t05": 0.9269819193321911,
      "train_pr_auc": 0.9841871324166283,
      "val_iou_t05": 0.8289349265021208,
      "val_iou_calibrated": 0.8289349265021208,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8309529775327127,
      "val_pr_auc": 0.8966952202951753,
      "val_accuracy": 0.9046203987730062,
      "val_boundary_margin_mae": 0.03521474823355675,
      "val_interior_recall": 0.9544039553195386,
      "val_bnd_pos_recall": 0.8974769723668402,
      "val_bnd_neg_spec": 0.900398406374502,
      "val_jitter_pos_recall": 0.8957055214723927,
      "val_jitter_neg_spec": 0.9160642570281124,
      "val_exterior_spec": 0.8588170664713423,
      "val_best_iou": 0.8289349265021208,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 54,
      "train_loss": 0.12385223154246978,
      "val_loss": 0.3126983989996991,
      "val_iou": 0.8283018867923818,
      "boundary_margin_mae": 0.03536512702703476,
      "lr": 5.422195776653379e-06,
      "train_iou_t05": 0.9266712930832923,
      "train_pr_auc": 0.98422494744706,
      "val_iou_t05": 0.8283018867923818,
      "val_iou_calibrated": 0.8283018867923818,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8302924268932075,
      "val_pr_auc": 0.8942191849120074,
      "val_accuracy": 0.904045245398773,
      "val_boundary_margin_mae": 0.03536512702703476,
      "val_interior_recall": 0.9569675883537814,
      "val_bnd_pos_recall": 0.8986784140969163,
      "val_bnd_neg_spec": 0.898406374501992,
      "val_jitter_pos_recall": 0.8948875255623722,
      "val_jitter_neg_spec": 0.9132530120481928,
      "val_exterior_spec": 0.8560703167917963,
      "val_best_iou": 0.8283018867923818,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 55,
      "train_loss": 0.12399090539868009,
      "val_loss": 0.3139628938888883,
      "val_iou": 0.8271180615225273,
      "boundary_margin_mae": 0.03574320301413536,
      "lr": 4.194783451479122e-06,
      "train_iou_t05": 0.9260718424099823,
      "train_pr_auc": 0.9843997016143871,
      "val_iou_t05": 0.8271180615225273,
      "val_iou_calibrated": 0.8271180615225273,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8283544726301044,
      "val_pr_auc": 0.8934094331992627,
      "val_accuracy": 0.9035659509202454,
      "val_boundary_margin_mae": 0.03574320301413536,
      "val_interior_recall": 0.9536714887383263,
      "val_bnd_pos_recall": 0.8966760112134562,
      "val_bnd_neg_spec": 0.8988047808764941,
      "val_jitter_pos_recall": 0.8912065439672802,
      "val_jitter_neg_spec": 0.9136546184738956,
      "val_exterior_spec": 0.8597326496978576,
      "val_best_iou": 0.8271180615225273,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 56,
      "train_loss": 0.12428061577145236,
      "val_loss": 0.3123314382936812,
      "val_iou": 0.8290840415485595,
      "boundary_margin_mae": 0.03554025664925575,
      "lr": 3.236571609226764e-06,
      "train_iou_t05": 0.9272643039145407,
      "train_pr_auc": 0.9842967456538547,
      "val_iou_t05": 0.8290840415485595,
      "val_iou_calibrated": 0.8289035125202265,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8302904564314664,
      "val_pr_auc": 0.8944170471168709,
      "val_accuracy": 0.9045724693251533,
      "val_boundary_margin_mae": 0.03554025664925575,
      "val_interior_recall": 0.9628273210034792,
      "val_bnd_pos_recall": 0.9014817781337605,
      "val_bnd_neg_spec": 0.8972111553784861,
      "val_jitter_pos_recall": 0.8957055214723927,
      "val_jitter_neg_spec": 0.9136546184738956,
      "val_exterior_spec": 0.8489287676249772,
      "val_best_iou": 0.8289035125202265,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 57,
      "train_loss": 0.12339680091147807,
      "val_loss": 0.31283479674529546,
      "val_iou": 0.8286058105551978,
      "boundary_margin_mae": 0.03540441766381264,
      "lr": 2.5502248902935075e-06,
      "train_iou_t05": 0.926784059313965,
      "train_pr_auc": 0.9843053937656296,
      "val_iou_t05": 0.8286058105551978,
      "val_iou_calibrated": 0.8288843258041727,
      "val_val_threshold": 0.44999999999999996,
      "val_calib_best_iou": 0.8303549164657277,
      "val_pr_auc": 0.8954087534623233,
      "val_accuracy": 0.9044286809815951,
      "val_boundary_margin_mae": 0.03540441766381264,
      "val_interior_recall": 0.9622779710675701,
      "val_bnd_pos_recall": 0.8986784140969163,
      "val_bnd_neg_spec": 0.898406374501992,
      "val_jitter_pos_recall": 0.8952965235173824,
      "val_jitter_neg_spec": 0.914859437751004,
      "val_exterior_spec": 0.8502105841420985,
      "val_best_iou": 0.8288843258041727,
      "val_best_threshold": 0.44999999999999996
    },
    {
      "epoch": 58,
      "train_loss": 0.12350312373480124,
      "val_loss": 0.3131855557525908,
      "val_iou": 0.828551795224117,
      "boundary_margin_mae": 0.03555877506732941,
      "lr": 2.1376519198007782e-06,
      "train_iou_t05": 0.9274623406718592,
      "train_pr_auc": 0.9843492907601609,
      "val_iou_t05": 0.828551795224117,
      "val_iou_calibrated": 0.828551795224117,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8298831385642045,
      "val_pr_auc": 0.8945860617891284,
      "val_accuracy": 0.9043328220858896,
      "val_boundary_margin_mae": 0.03555877506732941,
      "val_interior_recall": 0.9556857718366599,
      "val_bnd_pos_recall": 0.8970764917901481,
      "val_bnd_neg_spec": 0.899203187250996,
      "val_jitter_pos_recall": 0.894478527607362,
      "val_jitter_neg_spec": 0.9156626506024096,
      "val_exterior_spec": 0.857901483244827,
      "val_best_iou": 0.828551795224117,
      "val_best_threshold": 0.49999999999999994
    },
    {
      "epoch": 59,
      "train_loss": 0.12366490295795611,
      "val_loss": 0.31366872726348954,
      "val_iou": 0.82889994851546,
      "boundary_margin_mae": 0.035625897347927094,
      "lr": 2.0000000000000003e-06,
      "train_iou_t05": 0.9270495599812582,
      "train_pr_auc": 0.984350774072981,
      "val_iou_t05": 0.82889994851546,
      "val_iou_calibrated": 0.82889994851546,
      "val_val_threshold": 0.49999999999999994,
      "val_calib_best_iou": 0.8302185883530093,
      "val_pr_auc": 0.8934884582419367,
      "val_accuracy": 0.9044286809815951,
      "val_boundary_margin_mae": 0.035625897347927094,
      "val_interior_recall": 0.9582494048709027,
      "val_bnd_pos_recall": 0.8978774529435323,
      "val_bnd_neg_spec": 0.8980079681274901,
      "val_jitter_pos_recall": 0.8936605316973415,
      "val_jitter_neg_spec": 0.9140562248995984,
      "val_exterior_spec": 0.8569859000183117,
      "val_best_iou": 0.82889994851546,
      "val_best_threshold": 0.49999999999999994
    }
  ],
  "val_metrics": {
    "mae": 0.6947223268736046,
    "mae_m": 0.027411582813596118,
    "mae_q": 0.061590589216620445,
    "spearman": 0.7581727785748865,
    "boundary_iou": 0.8029664924684783,
    "reach_accuracy": 0.8977734892103543,
    "n": 125398,
    "interior_recall": 0.9641091375206006,
    "bnd_pos_recall": 0.9275130156187424,
    "bnd_neg_spec": 0.9250996015936255,
    "jitter_pos_recall": 0.9276073619631902,
    "jitter_neg_spec": 0.9373493975903614,
    "exterior_spec": 0.8318989196117927,
    "iou": 0.844275731170663,
    "accuracy": 0.9134873466257669,
    "iou_t05": 0.844275731170663,
    "iou_calibrated": 0.8443206172309606,
    "val_threshold": 0.44999999999999996,
    "calib_best_iou": 0.8511364582465364,
    "pr_auc": 0.9264274027105357,
    "boundary_margin_mae": 0.028103552758693695,
    "best_iou": 0.8443206172309606,
    "best_threshold": 0.44999999999999996
  }
}
```

---

## 5. How to run smoke checks

```bash
cd ird_playground && source env.sh
python -m pytest tests/test_core.py -q
```

P1 smoke (synthetic manifold):

```python
from ird_playground.neural.model import NeuralIRD, NeuralIRDPoint
from ird_playground.traj.manifold import SyntheticVesselSkinManifold
from ird_playground.traj.p1_optimize import P1Config, optimize_p1_lambda_rail
net = NeuralIRD(NeuralIRDPoint(hidden=64, depth=2, use_physical_pe=False), device="cpu")
print(optimize_p1_lambda_rail(
    net, SyntheticVesselSkinManifold(),
    cfg=P1Config(n_ctrl=5, n_knots_eval=12, region_k=16, steps=5),
)["final_loss"])
```

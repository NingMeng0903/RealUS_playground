# Neural IRD v6 — 平台根因已定位并修复

Generated: 2026-07-20 17:08:54

## 结论

v5 不是「完全没学」：epoch 0 学完粗工作空间包络（IoU≈0.54），之后细边界无法继续学。

### Bug 1（最严重）：trusted boundary 筛选反了

`soft_neg <= 0.05` 在 7×7 邻域下等价于「局部最多 ~2 个 hit」→ **优先保留孤立盐粒点**，稳定连续边界反而被过滤。

v6 重建实测：

```
support_pos quantiles=[1, 1, 2, 3, 12]   # 中位数=2 → 孤立 hit
support_neg quantiles=[0, 0, 1, 2, 13]
trusted faces=34,205/400,000 (C+>=3 & C-=0); rejected=365,795
boundary capped 800k → 68,410
jitter 34k+34k
N=836,820 (was 1.9M)
```

约 63% 的旧训练数据（800k bnd + 400k jitter）依赖错误筛选。

### Bug 2：位置 PE 最高物理周期 ~10 cm，解析不了 1.5–3 cm 边界

v6：物理波长 `[0.48, 0.24, 0.12, 0.06, 0.03, 0.015]` m；`num_freqs_u=5`。

### 其它修复

- 删除「trusted 不足则退回全部 pair」fallback → RuntimeError
- 验证：固定 calib/test blocks + indices；分别报告 `iou@0.5` / `iou@cal` / `pr_auc`
- Phase A 训练 hard `reachable`，不用 `y_soft`
- Sampler：无放回 cycling

### 开训

```bash
cd ird_playground && source env.sh
python -m ird_playground.cli.train --config configs/train_config.yaml
```

期望：epoch 0 仍快速达粗包络；后续 `bnd_pos_recall` / `pr_auc` / `iou@cal` 应同步上升。

Phase A 进入 B 门槛：`PR-AUC>0.80`，`bnd_pos_recall>0.65`，`bnd_neg_spec>0.65`，`iou@cal>0.65`。

---

# Neural IRD — 完整诊断 & 代码归档

Generated: 2026-07-20 16:53:15

> 本文档**替换**此前所有 debug 内容。包含：根因分析、v4/v5 训练对比、仍存在的瓶颈、以及当前全部相关源码。

---

## 1. 结论（为什么「还是没有提升」）

### 1.1 最大根因（目标定义错误）

Capability map 是 **Monte Carlo 稀疏命中** 的二值场，不是 IK 验证后的真实可达场：

| 量 | 值 |
|---|---|
| 稀疏格 `M×642` | 417,201 × 642 ≈ **2.68×10⁸** |
| MC 正 bit | **4,814,538**（≈**1.80%**） |
| 每体素平均命中方向 | ≈ **11.5** / 642 |
| MC 采样量 | ~10⁷ 关节态 |
| 每 bin 期望命中 | ≪ 1 |

**`bit=0` 只表示「这次 MC 没碰巧落进该 (voxel, orient) bin」**，不表示 IK+碰撞后确定不可达。

旧 pipeline 做 `neg = np.nonzero(~bits)` → 网络学的是 **MC hit vs miss** 的盐粒场，不是真实 reachability。

### 1.2 v4 → v5 实际发生了什么

| 指标 | v4（错误标签） | v5（trusted 标签 + 修 batch） | 目标 |
|---|---|---|---|
| val IoU @0.5（日志 `val_iou`） | **0.23–0.29** | **0.54–0.56**（ep0 即 0.54，ep8 后平台） | ≥0.70 |
| best IoU（threshold sweep） | ~0.29 | **~0.559 @ 0.25–0.35** | ≥0.70 |
| PR-AUC | — | **~0.73** | — |
| bnd_pos_recall | **~5%** | **~35%** | 高 |
| bnd_neg_spec | ~95% | **~65%**（不再极端保守） | 高 |
| jitter 正率 | **3.7%** | **50%**（jitter_pos/neg 各半） | 50% |
| train loss | ~0.62 | **~0.58** | — |

**v5 相对 v4 有明显提升**（IoU 从 0.28→0.56，recall 从 5%→35%），但：

1. **epoch 8 后完全平台** — loss 仍降，IoU/PR-AUC 不动 → 不是 LR/epoch 问题，是**标签/任务上限**
2. **固定 0.5 阈值 IoU ~0.43**（wandb `val/iou`）— 概率整体偏低，需 threshold≈0.3
3. **bnd_pos_recall 仍只有 ~35%** — face-pair 负侧仍是「同 orient 邻格 bit=0」，大量仍是 MC 漏采假负
4. **未做 IK 假负率审计** — 近邻 bit=0 里有多少 IK 其实成功，未知
5. **仍无 hit_count/KDE map** — 网络无法学平滑密度场

若用户看的是 **固定 0.5 IoU ~0.43** 或 **与 0.70 目标的差距**，会感觉「完全没有提升」。

### 1.3 已排除的硬 bug

- **bit pack/unpack round-trip PASS** — writer `pack_bits_5dof` 与 reader `unpack_bits_5dof` 均为 little-endian OR；naive `np.packbits`（big-endian）约 **31.7%** bit 错位
- **FK ±Z 单样本** — 未跑（缺 `Robotic_Arm` 模块）

### 1.4 v5 标签合同（当前实现）

| 类型 | 规则 | cls_weight |
|---|---|---|
| 正 | exact MC hit；face 内侧；jitter_pos | 1 |
| 负 | soft≈0 的 bit=0；face 外侧；off-map | 1 |
| unknown | 近 MC hit 但本 bin 未命中 | **0**（不进 BCE）|

GT: `N=1,900,000`，`reach≈0.474`，`supervised=100%`，layers: int=300k, bnd±≈400k, jit±=200k, ext=400k

### 1.5 仍存在的结构性问题

1. **Boundary neg 仍基于 exact bit** — face 负侧 = 邻格同 orient `bit=0`，不是 IK 不可达
2. **Exterior neg = soft≈0 的 bit=0** — 仍可能是 MC 漏采
3. **Interior/bnd_pos = exact hit** — 正标签可信，但只占 1.8% 稀疏空间
4. **5-DoF 无 roll** — 对绕 u 的探头自旋不敏感（后续优化 α(s) 的缺口）
5. **MLP 容量不是瓶颈** — 256×5 + PE 足够学平滑场；瓶颈在 GT 语义

### 1.6 不要继续做的事

- 同标签加 epoch / 加深网络到 512×10
- focal loss（会放大 MC 漏采假负）
- hash grid / IPE / SIREN 堆模块
- 只调 pos_weight 而不改 GT

### 1.7 下一步（按优先级）

1. **IK 假负率审计** — 抽 5k–10k 个 bit=0 近邻点，多 seed IK，估 FN rate
2. **重建 map：hit_count + spatial/angular splatting** — 保存密度而非二值 bit
3. **训练目标改 soft density** — `y_soft = 1-exp(-τ·c)`，BCE on soft labels
4. **补 FK ±Z round-trip** — 确认 map 记录的是 TCP +Z
5. **roll bin 或 6D rotation** — 若探头非轴对称

---

## 2. 训练日志

### 2.1 v4 Phase-A（错误：~bits 当不可达）

```
epoch=0 train_loss=0.6400 val_loss=0.6181 val_iou=0.235 bnd_m_mae=0.591 lr=3.00e-04
epoch=1 train_loss=0.6311 val_loss=0.6162 val_iou=0.229 bnd_m_mae=0.593 lr=2.99e-04
epoch=2 train_loss=0.6293 val_loss=0.6156 val_iou=0.244 bnd_m_mae=0.600 lr=2.97e-04
epoch=3 train_loss=0.6282 val_loss=0.6141 val_iou=0.284 bnd_m_mae=0.606 lr=2.94e-04
epoch=4 train_loss=0.6274 val_loss=0.6134 val_iou=0.242 bnd_m_mae=0.598 lr=2.90e-04
epoch=5 train_loss=0.6268 val_loss=0.6129 val_iou=0.279 bnd_m_mae=0.608 lr=2.85e-04
epoch=6 train_loss=0.6265 val_loss=0.6125 val_iou=0.255 bnd_m_mae=0.605 lr=2.80e-04
epoch=7 train_loss=0.6253 val_loss=0.6103 val_iou=0.281 bnd_m_mae=0.622 lr=2.73e-04
epoch=8 train_loss=0.6230 val_loss=0.6080 val_iou=0.259 bnd_m_mae=0.627 lr=2.66e-04
epoch=9 train_loss=0.6217 val_loss=0.6079 val_iou=0.289 bnd_m_mae=0.627 lr=2.58e-04
epoch=10 train_loss=0.6209 val_loss=0.6089 val_iou=0.256 bnd_m_mae=0.617 lr=2.50e-04
epoch=11 train_loss=0.6202 val_loss=0.6073 val_iou=0.291 bnd_m_mae=0.623 lr=2.41e-04
epoch=12 train_loss=0.6198 val_loss=0.6064 val_iou=0.275 bnd_m_mae=0.621 lr=2.31e-04
epoch=13 train_loss=0.6195 val_loss=0.6072 val_iou=0.285 bnd_m_mae=0.623 lr=2.21e-04
epoch=14 train_loss=0.6193 val_loss=0.6061 val_iou=0.265 bnd_m_mae=0.621 lr=2.10e-04
epoch=15 train_loss=0.6188 val_loss=0.6062 val_iou=0.282 bnd_m_mae=0.621 lr=1.99e-04
epoch=16 train_loss=0.6186 val_loss=0.6057 val_iou=0.278 bnd_m_mae=0.622 lr=1.88e-04
epoch=17 train_loss=0.6182 val_loss=0.6057 val_iou=0.276 bnd_m_mae=0.618 lr=1.77e-04
epoch=18 train_loss=0.6178 val_loss=0.6063 val_iou=0.275 bnd_m_mae=0.620 lr=1.65e-04
```

wandb: `neural_ird_v4_cls_only` / run `wbnnrdfo`

### 2.2 v5 Phase-A（trusted labels，run `02gycww7`，中断于 ep18）

```
epoch=0 train_loss=0.6244 val_loss=0.5866 val_iou=0.542 best_iou=0.542@0.30 pr_auc=0.714 bnd_m_mae=0.148 lr=3.00e-04
epoch=1 train_loss=0.6039 val_loss=0.5801 val_iou=0.545 best_iou=0.545@0.35 pr_auc=0.716 bnd_m_mae=0.155 lr=2.99e-04
epoch=2 train_loss=0.6004 val_loss=0.5772 val_iou=0.548 best_iou=0.548@0.40 pr_auc=0.721 bnd_m_mae=0.162 lr=2.97e-04
epoch=3 train_loss=0.5955 val_loss=0.5709 val_iou=0.551 best_iou=0.551@0.35 pr_auc=0.723 bnd_m_mae=0.164 lr=2.94e-04
epoch=4 train_loss=0.5891 val_loss=0.5649 val_iou=0.555 best_iou=0.555@0.40 pr_auc=0.729 bnd_m_mae=0.159 lr=2.90e-04
epoch=5 train_loss=0.5868 val_loss=0.5620 val_iou=0.557 best_iou=0.557@0.30 pr_auc=0.730 bnd_m_mae=0.157 lr=2.85e-04
epoch=6 train_loss=0.5855 val_loss=0.5612 val_iou=0.557 best_iou=0.557@0.35 pr_auc=0.731 bnd_m_mae=0.152 lr=2.80e-04
epoch=7 train_loss=0.5849 val_loss=0.5607 val_iou=0.557 best_iou=0.557@0.35 pr_auc=0.730 bnd_m_mae=0.156 lr=2.73e-04
epoch=8 train_loss=0.5841 val_loss=0.5602 val_iou=0.558 best_iou=0.558@0.30 pr_auc=0.731 bnd_m_mae=0.159 lr=2.66e-04
epoch=9 train_loss=0.5833 val_loss=0.5606 val_iou=0.558 best_iou=0.558@0.35 pr_auc=0.731 bnd_m_mae=0.159 lr=2.58e-04
epoch=10 train_loss=0.5827 val_loss=0.5593 val_iou=0.558 best_iou=0.558@0.35 pr_auc=0.732 bnd_m_mae=0.156 lr=2.50e-04
epoch=11 train_loss=0.5822 val_loss=0.5592 val_iou=0.557 best_iou=0.557@0.25 pr_auc=0.734 bnd_m_mae=0.169 lr=2.41e-04
epoch=12 train_loss=0.5819 val_loss=0.5591 val_iou=0.558 best_iou=0.558@0.35 pr_auc=0.731 bnd_m_mae=0.168 lr=2.31e-04
epoch=13 train_loss=0.5814 val_loss=0.5590 val_iou=0.558 best_iou=0.558@0.35 pr_auc=0.726 bnd_m_mae=0.163 lr=2.21e-04
epoch=14 train_loss=0.5811 val_loss=0.5605 val_iou=0.558 best_iou=0.558@0.35 pr_auc=0.730 bnd_m_mae=0.167 lr=2.10e-04
epoch=15 train_loss=0.5802 val_loss=0.5580 val_iou=0.559 best_iou=0.559@0.35 pr_auc=0.731 bnd_m_mae=0.164 lr=2.00e-04
epoch=16 train_loss=0.5800 val_loss=0.5596 val_iou=0.558 best_iou=0.558@0.30 pr_auc=0.729 bnd_m_mae=0.164 lr=1.88e-04
epoch=17 train_loss=0.5794 val_loss=0.5588 val_iou=0.559 best_iou=0.559@0.25 pr_auc=0.732 bnd_m_mae=0.172 lr=1.77e-04
```

wandb: `neural_ird_v5_cls_trusted` / run `02gycww7`

**v5 epoch 摘要表：**

| epoch | train_loss | val_loss | val_iou(best) | best@thr | pr_auc | bnd_m_mae |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6244 | 0.5866 | 0.542 | 0.542@0.30 | 0.714 | 0.148 || 1 | 0.6039 | 0.5801 | 0.545 | 0.545@0.35 | 0.716 | 0.155 || 2 | 0.6004 | 0.5772 | 0.548 | 0.548@0.40 | 0.721 | 0.162 || 3 | 0.5955 | 0.5709 | 0.551 | 0.551@0.35 | 0.723 | 0.164 || 4 | 0.5891 | 0.5649 | 0.555 | 0.555@0.40 | 0.729 | 0.159 || 5 | 0.5868 | 0.5620 | 0.557 | 0.557@0.30 | 0.730 | 0.157 || 6 | 0.5855 | 0.5612 | 0.557 | 0.557@0.35 | 0.731 | 0.152 || 7 | 0.5849 | 0.5607 | 0.557 | 0.557@0.35 | 0.730 | 0.156 || 8 | 0.5841 | 0.5602 | 0.558 | 0.558@0.30 | 0.731 | 0.159 || 9 | 0.5833 | 0.5606 | 0.558 | 0.558@0.35 | 0.731 | 0.159 || 10 | 0.5827 | 0.5593 | 0.558 | 0.558@0.35 | 0.732 | 0.156 || 11 | 0.5822 | 0.5592 | 0.557 | 0.557@0.25 | 0.734 | 0.169 || 12 | 0.5819 | 0.5591 | 0.558 | 0.558@0.35 | 0.731 | 0.168 || 13 | 0.5814 | 0.5590 | 0.558 | 0.558@0.35 | 0.726 | 0.163 || 14 | 0.5811 | 0.5605 | 0.558 | 0.558@0.35 | 0.730 | 0.167 || 15 | 0.5802 | 0.5580 | 0.559 | 0.559@0.35 | 0.731 | 0.164 || 16 | 0.5800 | 0.5596 | 0.558 | 0.558@0.30 | 0.729 | 0.164 || 17 | 0.5794 | 0.5588 | 0.559 | 0.559@0.25 | 0.732 | 0.172 |
**观察：** ep0 即 val_iou=0.542；ep8 达 0.558 后至 ep17 几乎不变；loss 从 0.624→0.579 仍在下降。

---

## 3. 命令

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/ird_playground
source env.sh

# 重建 GT
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml

# Phase A cls-only
python -m ird_playground.cli.train --config configs/train_config.yaml
```

---

## 4. 完整源码


### `ird_playground/ird/export_gt.py`

```python
"""Build IRD GT v5 — MC-hit ≠ unreachable; soft / unknown / trusted-neg labels.

Contract:
  features = [p_base,tcp(3), u_base(3)]  natural 5-DoF
  y_soft ∈ [0,1] from local spatial×orient neighborhood of MC hits
  cls_weight = 0 on UNKNOWN (near hits but bit=0); only supervise trusted labels
  Label rules (SE(3)-proxy distance in voxel/orient neighborhood):
    exact hit          → y=1,   cls_weight=1
    soft coverage > τ  → y=y_soft, cls_weight=1   (optional soft positives)
    far from all hits  → y=0,   cls_weight=1       (trusted negative)
    near miss (bit=0)  → unknown, cls_weight=0     (NOT in BCE)
  margin: continuous face-pair interpolation only; margin_weight=0 elsewhere
  jitter: face-pair normal ±delta half/half (LAYER_JITTER_POS/NEG)
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
    # soft / unknown thresholds
    orient_knn: int = 7
    soft_tau: float = 0.05  # soft>tau counts as weak positive coverage
    unknown_soft_max: float = 0.25  # bit=0 & soft in (0, this] → unknown
    trusted_neg_soft_max: float = 1e-6  # soft≈0 → trusted negative


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

    def soft_at(ijk: np.ndarray, oids: np.ndarray) -> np.ndarray:
        """Local MC-hit coverage in 7-spatial × K-orient neighborhood (vectorized)."""
        ijk = np.asarray(ijk, dtype=np.int32)
        oids = np.asarray(oids, dtype=np.int32)
        n = oids.shape[0]
        o_nb = knn[oids]  # (n, K)
        acc = np.zeros(n, dtype=np.float64)
        cnt = np.zeros(n, dtype=np.float64)
        for dlt in spat:
            j = ijk + dlt
            inb = (
                (j[:, 0] >= 0) & (j[:, 0] < nx)
                & (j[:, 1] >= 0) & (j[:, 1] < ny)
                & (j[:, 2] >= 0) & (j[:, 2] < nz)
            )
            if not inb.any():
                continue
            keys = (
                np.clip(j[:, 0], 0, nx - 1).astype(np.int64) * (ny * nz)
                + np.clip(j[:, 1], 0, ny - 1).astype(np.int64) * nz
                + np.clip(j[:, 2], 0, nz - 1).astype(np.int64)
            )
            r = np.full(n, -1, dtype=np.int32)
            r[inb] = row_of[keys[inb]]
            ok = r >= 0
            if not ok.any():
                continue
            # mean over orient neighbors for valid rows (row-wise gather)
            sub = bits[r[ok][:, None], o_nb[ok]].mean(axis=1)
            acc[ok] += sub
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

    # --- Boundary face pairs ---
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
            assigned[fail] = True
        if assigned.all():
            break
    keep = assigned
    bnd_r, bnd_o, bnd_ijk_neg = bnd_r[keep], bnd_o[keep], bnd_ijk_neg[keep]
    print(f"[gt] face pairs kept={bnd_r.size:,}", flush=True)
    if bnd_r.size == 0:
        raise RuntimeError("no boundary face pairs found")

    # Classify neg side soft coverage (diagnostic). Face geometry is still trusted:
    # pos = exact MC hit, neg = adjacent cell with same orient bit=0.
    print("[gt] soft-coverage on boundary neg side…", flush=True)
    soft_neg = soft_at_batched(bnd_ijk_neg, bnd_o)
    trusted_neg_pair = soft_neg <= cfg.soft_tau  # soft≈0 → especially clean neg
    unknown_pair = (soft_neg > cfg.soft_tau) & (soft_neg <= cfg.unknown_soft_max)
    print(
        f"[gt] face neg soft: clean={trusted_neg_pair.mean():.3f} mid={unknown_pair.mean():.3f} "
        f"soft_mean={soft_neg.mean():.4f} (all face pairs kept for geometric margin)",
        flush=True,
    )
    # Prefer cleaner face pairs when available; otherwise use all
    trusted_idx = np.flatnonzero(trusted_neg_pair)
    if trusted_idx.size < max(1000, n_bnd // 20):
        trusted_idx = np.arange(bnd_r.size)
    print(f"[gt] boundary face pool={trusted_idx.size:,}", flush=True)

    print(f"[gt] boundary interpolate {n_bnd:,} (supervised face pairs)", flush=True)
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
        cw = np.ones(n, dtype=np.float32)  # only trusted pairs
        mw = np.ones(n, dtype=np.float32)
        layer = np.where(y >= 0.5, LAYER_BND_POS, LAYER_BND_NEG).astype(np.int32)
        rows_q = np.where(y >= 0.5, rows, -1)
        flush(ps, orients[oids], y, ys, cw, m, mw, layer, rows_q, oids)

    # --- Jitter from face normal (pos/neg half-half), NOT isotropic MC-noise ---
    n_jit = int(cfg.n_jitter)
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
        "label_kind": np.array([2], dtype=np.int32),  # 2 = soft/unknown/trusted
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

### `ird_playground/ird/capability_io.py`

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

### `ird_playground/ird/query_base.py`

```python
"""Query-time base pose from rail_y via full SE(3) composition + AD helpers."""

from __future__ import annotations

import numpy as np

from ird_playground.probe.se3 import features_from_delta_T, invert_T, se3_mul

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def trans_y(r: float) -> np.ndarray:
    """Homogeneous translation along +Y (rail axis)."""
    T = np.eye(4, dtype=np.float64)
    T[1, 3] = float(r)
    return T


def T_base_from_rail_y(
    rail_y: float,
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> np.ndarray:
    """T_base(r) = T_world_rail · Trans_y(r) · T_rail_base0."""
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
    """ΔT(r) = T_tcp^{-1} T_base(r)."""
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
    """Query network at ΔT(rail_y); returns scalar m,q,score."""
    dT = delta_T_from_tcp_and_rail(
        T_tcp, rail_y, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
    )
    ps = neural_ird.score(dT)
    return {"m": ps.m, "q": ps.q, "score": ps.score}


def _features_torch_from_delta_T(dT: "torch.Tensor") -> "torch.Tensor":
    """dT (4,4) → natural features (6,) = p_base,tcp + u_base."""
    R_delta = dT[:3, :3]
    t_delta = dT[:3, 3]
    R_base_tcp = R_delta.T
    p = -(R_base_tcp @ t_delta)
    u = R_base_tcp[:, 2]
    u = u / (u.norm().clamp_min(1e-6))
    return torch.cat([p, u], dim=0)


def score_vs_rail_y_torch(
    neural_ird,
    T_tcp: np.ndarray,
    rail_y: "torch.Tensor",
    *,
    T_world_rail: np.ndarray | None = None,
    T_rail_base0: np.ndarray | None = None,
) -> "torch.Tensor":
    """Differentiable score w.r.t. rail_y (scalar tensor)."""
    if torch is None:
        raise ImportError("torch required")
    Twr = np.eye(4) if T_world_rail is None else np.asarray(T_world_rail, dtype=np.float64)
    Trb = np.eye(4) if T_rail_base0 is None else np.asarray(T_rail_base0, dtype=np.float64)
    T_tcp = np.asarray(T_tcp, dtype=np.float64)
    device = neural_ird.device

    Twr_t = torch.as_tensor(Twr, dtype=torch.float32, device=device)
    Trb_t = torch.as_tensor(Trb, dtype=torch.float32, device=device)
    Ttcp_t = torch.as_tensor(T_tcp, dtype=torch.float32, device=device)

    # Trans_y(r)
    Ty = torch.eye(4, dtype=torch.float32, device=device)
    Ty = Ty.clone()
    Ty[1, 3] = rail_y
    T_base = Twr_t @ Ty @ Trb_t
    # invert T_tcp
    R = Ttcp_t[:3, :3]
    t = Ttcp_t[:3, 3]
    Ti = torch.eye(4, dtype=torch.float32, device=device)
    Ti = Ti.clone()
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    dT = Ti @ T_base
    feat = _features_torch_from_delta_T(dT).unsqueeze(0)
    _, _, _, score = neural_ird.model(feat)
    return score.squeeze()


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
    """Compare AD ∂score/∂rail_y to central finite differences."""
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
        T_tcp = mat4_from_Rt(complete_frame_from_tool_axis(u), p)

        r = torch.tensor(float(rail_y), dtype=torch.float32, device=neural_ird.device, requires_grad=True)
        s = score_vs_rail_y_torch(
            neural_ird, T_tcp, r, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )
        s.backward()
        g_ad = float(r.grad.item())

        sp = score_vs_rail_y(
            neural_ird, T_tcp, rail_y + eps, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )["score"]
        sm = score_vs_rail_y(
            neural_ird, T_tcp, rail_y - eps, T_world_rail=T_world_rail, T_rail_base0=T_rail_base0
        )["score"]
        g_fd = (sp - sm) / (2.0 * eps)
        denom = max(abs(g_fd), abs(g_ad), 1e-6)
        rels.append(abs(g_ad - g_fd) / denom)
        signs.append(1.0 if np.sign(g_ad) == np.sign(g_fd) or abs(g_fd) < 1e-8 else 0.0)

    return {
        "rail_ad_fd_rel": float(np.median(rels)),
        "rail_sign_agree": float(np.mean(signs)),
        "rail_n": float(n),
    }
```

### `ird_playground/probe/se3.py`

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

### `ird_playground/neural/model.py`

```python
"""Neural IRD v4: f_θ(p,u) → (reach_logit, margin, q). Natural 5-DoF + u PE."""

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


def positional_encoding(x: "torch.Tensor", num_freqs: int) -> "torch.Tensor":
    freqs = (2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)) * np.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


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

    Fourier PE on position (num_freqs) and tool axis (num_freqs_u).
    """

    def __init__(
        self,
        *,
        in_dim: int = 6,
        num_freqs: int = 6,
        num_freqs_u: int = 3,
        hidden: int = 256,
        depth: int = 5,
        tau_m: float = 1.0,
        lambda_q: float = 0.5,
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

    def normalize_xyz(self, features: "torch.Tensor") -> "torch.Tensor":
        p = features[..., :3]
        u = features[..., 3:6]
        u = u / (u.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
        p_n = 2.0 * (p - self.aabb_lo) / span - 1.0
        return torch.cat([p_n, u], dim=-1)

    def encode(self, features: "torch.Tensor") -> "torch.Tensor":
        x = self.normalize_xyz(features)
        p_enc = positional_encoding(x[..., :3], self.num_freqs)
        u_enc = positional_encoding(x[..., 3:6], self.num_freqs_u)
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
        score = -F.softplus(-margin / max(self.tau_m, 1e-6)) + self.lambda_q * q
        return reach_logit, margin, q, score

    def score_features(self, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        reach_logit, margin, q, score = self.forward(features)
        p_reach = torch.sigmoid(reach_logit)
        return {
            "reach_logit": reach_logit,
            "m": margin,
            "margin": margin,
            "q": q,
            "q_comfort": q,
            "score": score,
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
            num_freqs_u=int(cfg.get("num_freqs_u", 3)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 5)),
            tau_m=float(cfg.get("tau_m", 1.0)),
            lambda_q=float(cfg.get("lambda_q", 0.5)),
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
        cfg = model_cfg or {
            "in_dim": 6,
            "num_freqs": self.model.num_freqs,
            "num_freqs_u": self.model.num_freqs_u,
            "hidden": self.model.hidden,
            "depth": self.model.depth,
            "tau_m": self.model.tau_m,
            "lambda_q": self.model.lambda_q,
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

### `ird_playground/neural/train.py`

```python
"""Train Neural IRD v4: BCE + masked SmoothL1(margin) + SmoothL1(q|pos).

Difficulty-aware batches, block-split val, best-by-IoU checkpoints.
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
    num_freqs_u: int = 3
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
        num_freqs_u=int(model.get("num_freqs_u", 3)),
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
        mix_interior=float(mix.get("interior", 0.15)),
        mix_bnd_pos=float(mix.get("bnd_pos", 0.25)),
        mix_bnd_neg=float(mix.get("bnd_neg", 0.25)),
        mix_jitter_pos=float(mix.get("jitter_pos", mix.get("jitter", 0.20) / 2)),
        mix_jitter_neg=float(mix.get("jitter_neg", mix.get("jitter", 0.20) / 2)),
        mix_exterior=float(mix.get("exterior", 0.15)),
        val_eval_n=int(train.get("val_eval_n", 65536)),
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
    def __init__(self, arrays: dict, yk: str, mk: str, qk: str):
        self.x = torch.as_tensor(arrays["features"], dtype=torch.float32)
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


class DifficultyBatchSampler(Sampler if torch is not None else object):  # type: ignore[misc]
    """Fixed mix: interior / bnd+ / bnd- / jitter / exterior."""

    def __init__(self, layer: np.ndarray, batch_size: int, mix: dict[int, float], *, seed: int = 0, steps: int | None = None):
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
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
        # remap empty pools to nearest non-empty
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
        # fix rounding
        while sum(counts.values()) > self.batch_size:
            k = max(counts, key=counts.get)
            counts[k] -= 1
        while sum(counts.values()) < self.batch_size:
            k = max(weights, key=weights.get)
            counts[k] += 1
        self.counts = counts
        n = len(layer)
        self.steps = int(steps) if steps is not None else max(1, n // self.batch_size)

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            for lid, c in self.counts.items():
                pool = self.pools[lid]
                batch.append(self.rng.choice(pool, size=c, replace=True))
            yield np.concatenate(batch).tolist()


def _maybe_init_wandb(cfg: TrainConfig):
    if not cfg.wandb_enable:
        return None
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        mode=cfg.wandb_mode,
        name=cfg.wandb_run_name or "neural_ird_v4",
        tags=cfg.wandb_tags or ["neural_ird", "v4", "natural_pu"],
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


def _layer_metrics(y: np.ndarray, p: np.ndarray, layer: np.ndarray) -> dict[str, float]:
    out = {}
    names = {
        LAYER_INTERIOR: "interior",
        LAYER_BND_POS: "bnd_pos",
        LAYER_BND_NEG: "bnd_neg",
        LAYER_JITTER_POS: "jitter_pos",
        LAYER_JITTER_NEG: "jitter_neg",
        LAYER_EXTERIOR: "exterior",
    }
    pred = p >= 0.5
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
    # threshold-swept IoU + PR-AUC proxy
    thresholds = np.linspace(0.05, 0.95, 19)
    ious = []
    for t in thresholds:
        yp = p >= t
        inter_t = float(np.logical_and(gt, yp).sum())
        union_t = float(np.logical_or(gt, yp).sum()) + 1e-9
        ious.append(inter_t / union_t)
    best_i = int(np.argmax(ious))
    out["best_iou"] = float(ious[best_i])
    out["best_threshold"] = float(thresholds[best_i])
    # average precision approx via sorted scores
    order = np.argsort(-p)
    y_s = gt[order].astype(np.float64)
    if y_s.sum() > 0 and (~gt).sum() > 0:
        tp = np.cumsum(y_s)
        fp = np.cumsum(1.0 - y_s)
        prec = tp / np.maximum(tp + fp, 1.0)
        rec = tp / y_s.sum()
        # AP = ∫ P dR
        out["pr_auc"] = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    else:
        out["pr_auc"] = 0.0
    return out


def _eval_subset(net, arrays, cfg: TrainConfig, seed: int = 0) -> dict[str, float]:
    n = arrays["features"].shape[0]
    yk, mk, qk = _y_key(arrays), _m_key(arrays), _q_key(arrays)
    rng = np.random.default_rng(seed)
    # Prefer supervised labels only (cls_weight>0); unknowns must not enter IoU.
    cw_all = arrays.get("cls_weight")
    supervised = np.flatnonzero(cw_all > 0) if cw_all is not None else np.arange(n)
    if supervised.size == 0:
        supervised = np.arange(n)
    # stratified by layer within supervised pool
    if "layer_id" in arrays and supervised.size > cfg.val_eval_n:
        layer_all = arrays["layer_id"]
        picks = []
        per = max(1, cfg.val_eval_n // 6)
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
        idx = np.concatenate(picks) if picks else rng.choice(supervised, size=min(cfg.val_eval_n, supervised.size), replace=False)
    else:
        idx = rng.choice(supervised, size=min(cfg.val_eval_n, supervised.size), replace=False)

    feats = arrays["features"][idx]
    pred = net.score_features_np(feats)
    # hard reachability for IoU (not soft training target)
    y = arrays[yk][idx]
    layer = arrays["layer_id"][idx] if "layer_id" in arrays else np.zeros(len(idx), dtype=np.int32)
    metrics = _layer_metrics(y, pred["p_reach"], layer)
    mw = arrays["margin_weight"][idx] if "margin_weight" in arrays else np.ones(len(idx))
    mask = mw > 0
    if mask.any():
        metrics["boundary_margin_mae"] = float(
            np.mean(np.abs(pred["m"][mask] - arrays[mk][idx][mask]))
        )
    else:
        metrics["boundary_margin_mae"] = 0.0
    return metrics


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

    tr_ds = IRDTensorDataset(train, yk, mk, qk)
    va_ds = IRDTensorDataset(val, yk, mk, qk)
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

    model = NeuralIRDPoint(
        in_dim=6,
        num_freqs=cfg.num_freqs,
        num_freqs_u=cfg.num_freqs_u,
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
    best_iou, best_margin_mae = -1.0, float("inf")
    best_iou_state, best_margin_state = None, None
    global_step = 0
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def model_cfg():
        return {
            "in_dim": 6,
            "num_freqs": cfg.num_freqs,
            "num_freqs_u": cfg.num_freqs_u,
            "hidden": cfg.hidden,
            "depth": cfg.depth,
            "tau_m": cfg.tau_m,
            "lambda_q": cfg.lambda_q_score,
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
            val_m = _eval_subset(wrapper, val, cfg, seed=cfg.seed + epoch)
            val_iou = float(val_m.get("best_iou", val_m["iou"]))
            bmae = float(val_m.get("boundary_margin_mae", 0.0))

            row = {
                "epoch": epoch,
                "train_loss": tr_loss / max(n_tr, 1),
                "val_loss": va_loss / max(n_va, 1),
                "val_iou": val_iou,
                "boundary_margin_mae": bmae,
                "lr": float(opt.param_groups[0]["lr"]),
                **{f"val_{k}": v for k, v in val_m.items()},
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_iou={val_iou:.3f} "
                f"best_iou={float(val_m.get('best_iou', val_iou)):.3f}@"
                f"{float(val_m.get('best_threshold', 0.5)):.2f} "
                f"pr_auc={float(val_m.get('pr_auc', 0)):.3f} "
                f"bnd_m_mae={bmae:.3f} lr={row['lr']:.2e}"
            )
            if wb_run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": row["train_loss"],
                        "val/loss": row["val_loss"],
                        "val/iou": val_iou,
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
        metrics.update(_eval_subset(wrapper, val, cfg, seed=cfg.seed))
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

### `ird_playground/cli/build_ird_gt.py`

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
            "contract": "MC-hit=pos; far soft≈0=trusted neg; near-miss=unknown (cls_weight=0); natural(p,u); face-pair margin/jitter",
            "feature_kind": "natural_pu",
            "label_kind": "trusted_soft_unknown",
        },
    )
    print(f"wrote {out}  N={arrays['features'].shape[0]} dim={arrays['features'].shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `ird_playground/cli/train.py`

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

### `rm75_control/.../capability_map.py (pack_bits only)`

```python
def pack_bits_5dof(bool_matrix: np.ndarray) -> np.ndarray:
    """(M, n_orient) bool → (M, ceil(n_orient/8)) uint8 little-bit-endian.

    Bit ``k`` inside byte ``b`` corresponds to ``orient_idx = 8*b + k``.
    """
    if bool_matrix.dtype != np.bool_:
        bool_matrix = bool_matrix.astype(bool)
    m, n_orient = bool_matrix.shape
    n_bytes = (n_orient + 7) // 8
    padded = np.zeros((m, n_bytes * 8), dtype=bool)
    padded[:, :n_orient] = bool_matrix
    packed = np.zeros((m, n_bytes), dtype=np.uint8)
    for k in range(8):
        packed |= (padded[:, k::8].astype(np.uint8) << k)
    return packed


def unpack_bits_5dof(packed: np.ndarray, n_orient: int) -> np.ndarray:
    """Inverse of :func:`pack_bits_5dof`."""
    m, n_bytes = packed.shape
    out = np.zeros((m, n_bytes * 8), dtype=bool)
    for k in range(8):
        out[:, k::8] = ((packed >> k) & 1).astype(bool)
    return out[:, :n_orient]


def d_value_from_bitmask(packed: np.ndarray, n_orient: int) -> np.ndarray:
    """Reachability index D(x) = (# reachable orientations) / n_orient."""
    counts = np.zeros(packed.shape[0], dtype=np.int32)
    for k in range(8):
        counts += ((packed >> k) & 1).sum(axis=1).astype(np.int32)
    # trim last-byte padding
    if n_orient % 8 != 0:
        overshoot = (packed.shape[1] * 8) - n_orient
        # subtract padding bits (they are always 0 by construction of pack_bits_5dof)
        del overshoot  # kept as a comment marker; padding is zeros so no correction needed
    return (counts.astype(np.float32) / float(n_orient)).astype(np.float32)
```

### `configs/ird_gt_config.yaml`

```yaml
# IRD GT v5 — MC-hit positives; trusted far negatives; unknown near-misses excluded

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
```

### `configs/train_config.yaml`

```yaml
# train_config.yaml — Neural IRD v5 phase A: cls-only on trusted MC-hit labels
# Env: cd ird_playground && source env.sh

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6
  num_freqs_u: 3
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5

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
  boundary_iou_min: 0.70
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
  run_name: neural_ird_v5_cls_trusted
  tags: [neural_ird, v5, trusted_labels, cls_only]
```

### `configs/train_cls_only.yaml`

```yaml
# Alias of train_config.yaml — v5 cls-only on trusted MC-hit labels
# Prefer: python -m ird_playground.cli.train --config configs/train_config.yaml

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6
  num_freqs_u: 3
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5

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
  boundary_iou_min: 0.70
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
  run_name: neural_ird_v5_cls_trusted
  tags: [neural_ird, v5, trusted_labels, cls_only]
```

### `configs/train_phase_b.yaml`

```yaml
# Phase B: cls + boundary margin + q (init from best_iou.pt manually if needed)

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6
  num_freqs_u: 3
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5

training:
  seed: 42
  batch_size: 1024
  num_workers: 4
  torch_compile: false
  epochs: 60
  save_freq: 10
  learning_rate: 2.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 300
  min_lr_ratio: 0.01
  grad_clip_norm: 10.0
  log_every_steps: 10
  print_every_steps: 50
  hardneg_every: 0
  hardneg_frac: 0.0
  val_eval_n: 65536
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
  boundary_iou_min: 0.70
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
  run_name: neural_ird_v5_margin_q
  tags: [neural_ird, v5, trusted_labels, margin_q]
```

---

## 5. bit round-trip 验证脚本

```python
import numpy as np
from ird_playground.ird.capability_io import unpack_bits_5dof

def pack_bits_5dof(bool_matrix):
    m, n_orient = bool_matrix.shape
    n_bytes = (n_orient + 7) // 8
    padded = np.zeros((m, n_bytes * 8), dtype=bool)
    padded[:, :n_orient] = bool_matrix
    packed = np.zeros((m, n_bytes), dtype=np.uint8)
    for k in range(8):
        packed |= (padded[:, k::8].astype(np.uint8) << k)
    return packed

rng = np.random.default_rng(0)
bits = rng.random((100, 642)) > 0.8
packed = pack_bits_5dof(bits)
assert np.array_equal(bits, unpack_bits_5dof(packed, 642))
bad = np.packbits(bits.astype(np.uint8), axis=1)  # big-endian default
print("roundtrip OK; naive packbits mismatch:", float((bits != unpack_bits_5dof(bad, 642)).mean()))
```

---

*End of archive.*

# Neural IRD debug dump — training + GT production

**Date:** 2026-07-19  
**Run:** `wandb/run-20260719_213857-x65kb89k` (`neural_ird_mq_v2`)  
**Purpose:** third-party review of why ~29 epochs still underperform.

---

## 0. Verdict (short)

Training is **not diverging**, but it is **stuck on a high loss floor** with **weak reachability classification**. The dominant issues are **label/feature contract bugs and a harmful local loss**, not “need more epochs”.

| epoch | train | val |
|------:|------:|----:|
| 0 | 4.426 | 3.827 |
| 5 | 3.974 | 3.637 |
| 10 | 3.944 | 3.617 |
| 20 | 3.924 | 3.609 |
| 28 | 3.907 | 3.605 |

After ~epoch 5, val only moves **~0.03**. Step breakdown at epoch 28–29:

- `L_cls ≈ 0.59` (barely better than predicting majority / weak classifier)
- `L_m ≈ 2.9–3.2` (margin MSE stuck)
- `L_q ≈ 0.006` (comfort already easy)
- `L_local ≈ 5.5–8.0` (huge; with `λ_local=0.05` still adds ~0.3–0.4 every step)

`best.pt` on 50k random GT points:

- `boundary_iou ≈ 0.53`, `reach_accuracy ≈ 0.63`
- `mae_m ≈ 1.39`, pred `m` mean≈−0.26 vs GT pos mean≈+1.23 / neg≈−2.52

---

## 1. Root causes (ordered)

### A. Feature frame vs AABB mismatch (critical)

GT `features[:3]` are **ΔT translation** `t = −Rᵀ p` (TCP-inverse-base), built in `export_gt._batch_feat_from_p_u`.

But `aabb_lo/hi` saved into the NPZ / checkpoint come from **world voxel centres** `voxel_xyz ± margin`.

The network then does:

```text
p_n = 2 * (features[:3] − aabb_lo) / (aabb_hi − aabb_lo) − 1
```

So Fourier PE is applied after **wrong-frame normalization**. Empirically a large fraction of ΔT translations sit outside that world AABB, so PE sees saturated / distorted coords.

### B. Reachability signal in features is weak / entangled

On the 2M GT:

- `spearman(x|y|z, m) ≈ 0`
- `spearman(|t|, m) ≈ −0.23`, `spearman(|t|, y) ≈ −0.19`

Orientation (rot6D) carries much of the reachability bit, while margin labels are mostly a **function of map D(row)** attached to a sampled orientation — many different orientations at the same voxel share the same `m_gt`/`q`. That makes `(ΔT → m)` a **noisy, multi-valued** regression.

### C. `L_local` is still the wrong regularizer for this batching

`_spatial_local_pair` takes **within-batch** nearest neighbours in feature `xyz` with `σ=0.06 m`.

With `batch=1024` over a 2M workspace-scale cloud, in-batch NNs are usually **far**; when a rare close pair appears, their `m_gt` often differs a lot (esp. pos vs hard-neg). Log shows `L_local≈6–8` **late in training** — it never collapses. Val loss **does not include** `L_local` (eval path passes `local_pair=None`), so train/val are not comparable and the model is pressured by a term that does not match the val objective.

### D. Margin labels still have a hard floor

Even after log1p-D fix:

- many exterior points sit at `m≈−3` (`tanh` clip)
- trivial class-conditional mean MSE of `m` is still O(1)
- with `λ_margin=1.0`, `L_m≈3` means the net has not fitted the margin field; combined with (A)(B) it may be **underdetermined / mis-normalized**

### E. Classification head shares the margin logit

`p = σ(m)` for BCE while `m` is also regressed to a continuous margin in `[-3,3]`. Early/mid training fights: BCE wants large |m| for confident class, MSE wants calibrated magnitudes. IoU stuck ~0.53 is consistent with this conflict under bad PE.

### F. What is *not* the main issue

- LR / warmup: lr still ~2.4e-4 at epoch 29; not annealed to death
- q loss: already tiny
- “need 100 epochs”: val plateaued by epoch ~10–15

---

## 2. Suggested fixes for third party (do not implement here)

1. Store / normalize **ΔT-frame AABB** (from `features[:,:3]`), or stop AABB-normalizing and use fixed scale.
2. Either drop `L_local` until spatial pairs are true SE(3)-local neighbours from the map, or build neighbour indices offline.
3. Decouple classification logit from margin head (two heads), or use BCE on `σ(m/τ)` with stop-grad between tasks.
4. Rebuild GT so `m_gt` is consistent with the same ΔT sample (orientation-aware), not only voxel D.
5. Report val with the same loss terms as train; log IoU / mae_m each epoch (not only scalar loss).

---

## 3. Reproduce

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/ird_playground
source env.sh
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml
python -m ird_playground.cli.train --config configs/train_config.yaml
python -m ird_playground.cli.eval_point --checkpoint data/checkpoints/best.pt --config configs/train_config.yaml
```

Map used for GT: `rm75_control/data/reachability/rm75_6f_1p5cm_15deg_coll_probe`  
GT NPZ: `ird_playground/data/ird/gt_samples_1p5cm_probe.npz` (N=2_000_000)

---

## 4. Verbatim source dump

Below: full file contents (no omissions) for training + GT production stack.


## FILE: `ird_playground/configs/train_config.yaml`

```yaml
# train_config.yaml — Neural IRD point field f_θ(ΔT) → (m, q)
# Env: Among_US genesis — source ird_playground/env.sh

data:
  gt_npz: data/ird/gt_samples_1p5cm_probe.npz
  synthetic_n: 8192
  val_frac: 0.15

model:
  num_freqs: 6          # xyz Fourier only
  hidden: 256
  depth: 5
  tau_m: 1.0
  lambda_q: 0.5         # score = -softplus(-m/τ) + λq

training:
  seed: 42
  batch_size: 1024
  num_workers: 4
  torch_compile: false
  epochs: 100
  save_freq: 25
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 500
  min_lr_ratio: 0.01
  grad_clip_norm: 10.0
  log_every_steps: 10
  print_every_steps: 50
  hardneg_every: 20
  hardneg_frac: 0.02
  device: cuda

loss:
  lambda_cls: 1.0
  lambda_margin: 1.0
  lambda_q: 1.0
  # Spatial within-batch NN (||Δp|| < sigma_local_m), NOT random shuffle
  lambda_local: 0.05
  sigma_local_m: 0.06

io:
  # Written by cli.train only; inference: --checkpoint data/checkpoints/best.pt
  checkpoint: data/checkpoints/latest.pt
  checkpoint_dir: data/checkpoints
  report: data/reports/train_point.json

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
  run_name: neural_ird_mq_v2
  tags: [neural_ird, m_q, rm75, fixed_local]
```


## FILE: `ird_playground/configs/ird_gt_config.yaml`

```yaml
# Dense stratified IRD GT (~2M) from 1.5 cm horizontal-probe+shaft map.
# Layers: 35% interior / 40% boundary / 25% exterior.
# m_gt is a continuous truncated margin (log1p in D), not a strict SDF.

map_dir: ../rm75_control/data/reachability/rm75_6f_1p5cm_15deg_coll_probe
out: data/ird/gt_samples_1p5cm_probe.npz

sampling:
  n_interior: 700000
  n_boundary: 800000
  n_exterior: 500000
  max_orients_per_voxel: 28
  hard_negative_frac: 0.45
  hard_negative_radius_m: 0.06
  sigma_p_m: 0.03
  sigma_r_deg: 10.0
  # Prefer percentiles if hi≈d_max (exporter auto-retunes); these are fallbacks
  boundary_d_lo: 0.008
  boundary_d_hi: 0.020
  m_clip: 3.0
  bbox_margin_m: 0.20
  comfort_from: auto
  k_candidates: 4
  seed: 42
```


## FILE: `ird_playground/configs/probe_default.yaml`

```yaml
# Default ultrasound probe TCP relative to link7.
#
# Chain (body-fixed, right-multiply):
#   Trans_z(0.07) · Rot_y(+pi/2) · Trans_z(0.05)
# TCP origin ≈ (0.05, 0, 0.07) in link7; TCP +Z = link7 +X.

name: ultrasound_probe_default
parent_frame: link_7
child_frame: tcp

# Translation of the composed TCP origin in parent (link7), metres.
translation_m: [0.05, 0.0, 0.07]

# Orientation as quaternion xyzw of TCP axes in parent.
# Rot_y(+90°) maps parent +X → child +Z, parent +Z → child -X, +Y → +Y.
# SciPy Rotation.from_euler('y', 90, degrees=True).as_quat() → [0, 0.70710678, 0, 0.70710678]
quaternion_xyzw: [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]

# Equivalent Euler for URDF patch / RealMan tool offset (xyz order, radians).
euler_xyz_rad: [0.0, 1.5707963267948966, 0.0]
```


## FILE: `ird_playground/configs/region_config.yaml`

```yaml
# Query-side Region A extents (runtime only; NOT trained into the point field).

position_region:
  tangent_1_m: 0.020
  tangent_2_m: 0.010
  normal_m: 0.002

orientation_region:
  tilt_tangent_1_deg: 8.0
  tilt_tangent_2_deg: 5.0
  axial_roll_deg: 3.0

aggregation:
  name: mean_softmin
  lambda: 0.6
  tau: 0.10
  d_min: 0.30
  tau_c: 0.05

sampling:
  sobol: true
  num_samples_nlp: 32
  num_samples_eval: 512
  seed: 0
```


## FILE: `ird_playground/ird_playground/ird/export_gt.py`

```python
"""Build dense stratified IRD training samples from a CapabilityMap.

Layers (default): 35% reachable interior / 40% boundary band / 25% exterior.
Labels: reachable, m_gt (margin, not a strict SDF), q and optional factor channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class IrdGtConfig:
    n_interior: int = 700_000
    n_boundary: int = 800_000
    n_exterior: int = 500_000
    # legacy aliases used by older configs
    n_positive: int = 700_000
    n_negative: int = 500_000
    seed: int = 0
    comfort_from: str = "auto"
    bbox_margin_m: float = 0.20
    max_orients_per_voxel: int = 24
    hard_negative_frac: float = 0.45
    hard_negative_radius_m: float = 0.06
    sigma_p_m: float = 0.03
    sigma_r_deg: float = 10.0
    boundary_d_lo: float = 0.02
    boundary_d_hi: float = 0.08
    m_pos_scale: float = 2.0
    m_neg_scale: float = 2.0
    m_clip: float = 3.0
    w_manip: float = 0.5
    w_d: float = 0.5
    k_candidates: int = 4
    n_dof: int = 7


def margin_from_d(
    d: np.ndarray | float,
    *,
    d_ref: float,
    d_max: float,
    m_clip: float = 3.0,
) -> np.ndarray:
    """Continuous positive margin from capability D — avoids saturating all interiors.

    ``m = m_clip * log1p(d/d_ref) / log1p(d_max/d_ref)`` ∈ (0, m_clip].
    Not a strict SDF; monotonic in D.
    """
    d = np.asarray(d, dtype=np.float64)
    d_ref = max(float(d_ref), 1e-6)
    d_max = max(float(d_max), d_ref)
    num = np.log1p(np.maximum(d, 0.0) / d_ref)
    den = np.log1p(d_max / d_ref)
    m = float(m_clip) * (num / max(den, 1e-12))
    return np.clip(m, 0.0, float(m_clip)).astype(np.float64)


def margin_exterior(dist_m: np.ndarray | float, *, sigma_m: float, m_clip: float = 3.0) -> np.ndarray:
    """Negative margin from distance-to-reachable; tanh keeps it non-saturating."""
    dist = np.asarray(dist_m, dtype=np.float64)
    sig = max(float(sigma_m), 1e-6)
    return (-float(m_clip) * np.tanh(dist / sig)).astype(np.float64)


def _batch_feat_from_p_u(ps: np.ndarray, us: np.ndarray) -> np.ndarray:
    """ps (N,3), us (N,3) → features (N,9), vectorized frame build."""
    n = int(ps.shape[0])
    z = np.asarray(us, dtype=np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    a = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))
    flip = np.abs(z[:, 0]) >= 0.9
    a[flip] = np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    y = np.cross(z, x)
    # R columns = x,y,z; T_base_tcp; ΔT = T^{-1} = [R^T | -R^T t]
    # features = t_delta(3) + R_delta[:,0](3) + R_delta[:,1](3)
    # R_delta = R^T, t_delta = -R^T @ p
    Rt = np.stack([x, y, z], axis=1)  # (N,3,3) rows = x,y,z → actually want columns
    # stack as columns: R[..., :, 0] = x
    R = np.stack([x, y, z], axis=2)  # (N,3,3)
    Rt = np.transpose(R, (0, 2, 1))  # R^T
    t_delta = -np.einsum("nij,nj->ni", Rt, ps.astype(np.float64))
    r6 = np.concatenate([Rt[:, :, 0], Rt[:, :, 1]], axis=1)
    return np.concatenate([t_delta, r6], axis=1).astype(np.float32)


def _precompute_orient_table(cm, orients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row_ids, orient_ids) flattened over all reachable (voxel, orient) pairs."""
    from ird_playground.ird.capability_io import unpack_bits_5dof

    n_orient = orients.shape[0]
    if cm.roll is None:
        bits = unpack_bits_5dof(np.asarray(cm.bitmask), n_orient)
    else:
        bits = np.any(cm.bitmask, axis=-1)
    rows, oids = np.nonzero(bits)
    return rows.astype(np.int64), oids.astype(np.int64)


def _sample_from_table(
    rng: np.random.Generator,
    table_rows: np.ndarray,
    table_oids: np.ndarray,
    pool_idx: np.ndarray,
    orients: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n (row, u) from precomputed indices into the orient table."""
    if pool_idx.size == 0:
        rows = np.zeros(n, dtype=np.int64)
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return rows, v
    pick = rng.integers(0, pool_idx.shape[0], size=n)
    ti = pool_idx[pick]
    return table_rows[ti], orients[table_oids[ti]]


def export_ird_gt_from_capability_map(
    cm,
    cfg: IrdGtConfig | None = None,
    *,
    batch_size: int = 16384,
) -> dict[str, np.ndarray]:
    """Dense stratified GT with margin labels."""
    cfg = cfg or IrdGtConfig()
    rng = np.random.default_rng(cfg.seed)

    if cfg.n_interior or cfg.n_boundary or cfg.n_exterior:
        n_int = int(cfg.n_interior)
        n_bnd = int(cfg.n_boundary)
        n_ext = int(cfg.n_exterior)
    else:
        n_tot = int(cfg.n_positive + cfg.n_negative)
        n_int = int(round(0.35 * n_tot))
        n_bnd = int(round(0.40 * n_tot))
        n_ext = max(0, n_tot - n_int - n_bnd)

    orients = np.asarray(cm.orientations.vectors, dtype=np.float64)
    voxel_xyz = cm.grid.center_of(cm.voxel_ids)
    d_vals = np.asarray(cm.d_value, dtype=np.float64)
    M = int(cm.voxel_ids.shape[0])
    d_max = float(max(d_vals.max(), 1e-6))

    # Prefer percentile bands so interior is not only the D tip (old hi≈d_max
    # collapsed all interior m_gt → m_clip).
    d_lo = float(cfg.boundary_d_lo)
    d_hi = float(cfg.boundary_d_hi)
    if d_hi >= 0.9 * d_max:
        d_lo = float(np.percentile(d_vals, 35))
        d_hi = float(np.percentile(d_vals, 70))
        print(f"[gt] retuned boundary band to percentiles: lo={d_lo:.4f} hi={d_hi:.4f}", flush=True)

    interior_rows = np.flatnonzero(d_vals >= d_hi)
    boundary_rows = np.flatnonzero((d_vals >= d_lo) & (d_vals < d_hi))
    if interior_rows.size == 0:
        interior_rows = np.arange(M)
    if boundary_rows.size == 0:
        boundary_rows = interior_rows
    print(
        f"[gt] rows interior={interior_rows.size:,} boundary={boundary_rows.size:,} "
        f"d_max={d_max:.4f}",
        flush=True,
    )

    print(f"[gt] unpacking orientation table for {M} voxels…", flush=True)
    table_rows, table_oids = _precompute_orient_table(cm, orients)
    print(f"[gt] table size={table_rows.shape[0]:,}", flush=True)

    is_int = np.zeros(M, dtype=bool)
    is_bnd = np.zeros(M, dtype=bool)
    is_int[interior_rows] = True
    is_bnd[boundary_rows] = True
    pool_int = np.flatnonzero(is_int[table_rows])
    pool_bnd = np.flatnonzero(is_bnd[table_rows])
    if pool_int.size == 0:
        pool_int = np.arange(table_rows.shape[0])
    if pool_bnd.size == 0:
        pool_bnd = pool_int
    print(f"[gt] pool interior={pool_int.size:,} boundary={pool_bnd.size:,}", flush=True)

    d_n_all = np.clip(d_vals / d_max, 0.0, 1.0).astype(np.float32)
    if cm.mu_mean is not None:
        mu = np.asarray(cm.mu_mean, dtype=np.float64)
        qm_all = np.clip(mu / (np.abs(mu) + 1.0), 0.0, 1.0).astype(np.float32)
        bad = ~np.isfinite(mu)
        qm_all[bad] = d_n_all[bad]
    else:
        qm_all = d_n_all.copy()
    qj_all = d_n_all.copy()
    qs_all = d_n_all.copy()
    qn_all = d_n_all.copy()
    q_all = np.clip(cfg.w_manip * qm_all + cfg.w_d * d_n_all, 0.0, 1.0).astype(np.float32)

    chunks: list[dict[str, np.ndarray]] = []

    def _emit(
        ps: np.ndarray,
        us: np.ndarray,
        y: np.ndarray,
        m: np.ndarray,
        rows: np.ndarray | None,
        *,
        zero_q: bool = False,
    ) -> None:
        feats = _batch_feat_from_p_u(ps, us)
        if zero_q or rows is None:
            q = np.zeros(ps.shape[0], dtype=np.float32)
            qm = qj = qs = qn = q
        else:
            q = q_all[rows]
            qm = qm_all[rows]
            qj = qj_all[rows]
            qs = qs_all[rows]
            qn = qn_all[rows]
            q = np.where(y >= 0.5, q, 0.0).astype(np.float32)
            qm = np.where(y >= 0.5, qm, 0.0).astype(np.float32)
            qj = np.where(y >= 0.5, qj, 0.0).astype(np.float32)
            qs = np.where(y >= 0.5, qs, 0.0).astype(np.float32)
            qn = np.where(y >= 0.5, qn, 0.0).astype(np.float32)
        chunks.append(
            {
                "features": feats,
                "reachable": y.astype(np.float32),
                "m_gt": m.astype(np.float32),
                "q": q,
                "q_manip": qm,
                "q_joint": qj,
                "q_selfcol": qs,
                "q_nullspace": qn,
            }
        )

    def _layer_reachable(n_total: int, pool: np.ndarray, m_fn, label: str) -> None:
        for s in range(0, n_total, batch_size):
            n = min(batch_size, n_total - s)
            rows, us = _sample_from_table(rng, table_rows, table_oids, pool, orients, n)
            m = m_fn(rows)
            _emit(voxel_xyz[rows], us, np.ones(n, dtype=np.float32), m, rows)
            if (s // batch_size) % 5 == 0:
                print(f"[gt] {label} {s + n:,}/{n_total:,}", flush=True)

    d_ref = max(d_lo, 1e-4)

    def _m_reach(rows: np.ndarray) -> np.ndarray:
        return margin_from_d(d_vals[rows], d_ref=d_ref, d_max=d_max, m_clip=cfg.m_clip)

    print(f"[gt] interior {n_int:,}", flush=True)
    _layer_reachable(n_int, pool_int, _m_reach, "interior")

    n_bnd_vox = n_bnd // 2
    n_bnd_jit = n_bnd - n_bnd_vox
    print(f"[gt] boundary voxels {n_bnd_vox:,}", flush=True)
    # Boundary voxels: near-zero continuous margin around d_ref
    def _m_bnd(rows: np.ndarray) -> np.ndarray:
        # signed relative to band centre
        mid = 0.5 * (d_lo + d_hi)
        span = max(0.5 * (d_hi - d_lo), 1e-6)
        return np.clip((d_vals[rows] - mid) / span, -1.0, 1.0) * (0.35 * cfg.m_clip)

    _layer_reachable(n_bnd_vox, pool_bnd, _m_bnd, "boundary")

    print(f"[gt] boundary jitter {n_bnd_jit:,}", flush=True)
    for s in range(0, n_bnd_jit, batch_size):
        n = min(batch_size, n_bnd_jit - s)
        rows, us = _sample_from_table(rng, table_rows, table_oids, pool_int, orients, n)
        jitter = rng.normal(scale=cfg.sigma_p_m, size=(n, 3))
        ps = voxel_xyz[rows] + jitter
        dist_m = np.linalg.norm(jitter, axis=1)
        y = (dist_m < cfg.sigma_p_m).astype(np.float32)
        m_base = _m_reach(rows)
        # shrink / flip with radial distance in SE(3) σ-ball
        m = np.where(
            y >= 0.5,
            m_base * (1.0 - dist_m / max(cfg.sigma_p_m, 1e-6)),
            margin_exterior(dist_m, sigma_m=cfg.sigma_p_m, m_clip=cfg.m_clip),
        )
        _emit(ps, us, y, m, rows)
        if (s // batch_size) % 5 == 0:
            print(f"[gt] jitter {s + n:,}/{n_bnd_jit:,}", flush=True)

    # --- Exterior ---
    mins = voxel_xyz.min(axis=0) - cfg.bbox_margin_m
    maxs = voxel_xyz.max(axis=0) + cfg.bbox_margin_m
    n_hard = int(round(n_ext * float(cfg.hard_negative_frac)))
    n_unif = max(0, n_ext - n_hard)
    n_cent = min(voxel_xyz.shape[0], 80_000)
    cent_idx = rng.choice(voxel_xyz.shape[0], size=n_cent, replace=False)
    centers = voxel_xyz[cent_idx]

    t_unif = rng.uniform(mins, maxs, size=(n_unif, 3)) if n_unif else np.zeros((0, 3))
    pick = rng.integers(0, centers.shape[0], size=n_hard) if n_hard else np.zeros(0, dtype=int)
    t_hard = (
        centers[pick] + rng.normal(scale=cfg.hard_negative_radius_m, size=(n_hard, 3))
        if n_hard
        else np.zeros((0, 3))
    )
    t_neg = np.concatenate([t_unif, t_hard], axis=0) if n_ext > 0 else np.zeros((0, 3))
    v = rng.normal(size=(t_neg.shape[0], 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12

    print(f"[gt] exterior {t_neg.shape[0]:,}", flush=True)
    for s in range(0, t_neg.shape[0], batch_size):
        n = min(batch_size, t_neg.shape[0] - s)
        pts = t_neg[s : s + n]
        sub = centers[rng.choice(centers.shape[0], size=min(2048, centers.shape[0]), replace=False)]
        dmin = np.full(n, np.inf, dtype=np.float64)
        for c0 in range(0, sub.shape[0], 512):
            cblk = sub[c0 : c0 + 512]
            d2 = ((pts[:, None, :] - cblk[None, :, :]) ** 2).sum(axis=2)
            dmin = np.minimum(dmin, np.sqrt(d2.min(axis=1)))
        m = margin_exterior(dmin, sigma_m=cfg.hard_negative_radius_m, m_clip=cfg.m_clip)
        _emit(pts, v[s : s + n], np.zeros(n, dtype=np.float32), m, None, zero_q=True)
        if (s // batch_size) % 5 == 0:
            print(f"[gt] exterior {s + n:,}/{t_neg.shape[0]:,}", flush=True)

    if not chunks:
        raise RuntimeError("no IRD samples extracted from capability map")

    def _cat(key: str) -> np.ndarray:
        return np.concatenate([c[key] for c in chunks], axis=0)

    features = _cat("features")
    y = _cat("reachable")
    m_arr = _cat("m_gt")
    q_arr = _cat("q")
    perm = rng.permutation(features.shape[0])

    aabb_lo = (voxel_xyz.min(axis=0) - cfg.bbox_margin_m).astype(np.float32)
    aabb_hi = (voxel_xyz.max(axis=0) + cfg.bbox_margin_m).astype(np.float32)

    n = features.shape[0]
    k = int(cfg.k_candidates)
    q_best = np.zeros((n, cfg.n_dof), dtype=np.float32)
    q_candidates = np.zeros((n, k, cfg.n_dof), dtype=np.float32)

    print(f"[gt] stacking N={n:,}", flush=True)
    return {
        "features": features[perm],
        "reachable": y[perm],
        "p_reach": y[perm],
        "m_gt": m_arr[perm],
        "q": q_arr[perm],
        "q_comfort": q_arr[perm],
        "q_manip": _cat("q_manip")[perm],
        "q_joint": _cat("q_joint")[perm],
        "q_selfcol": _cat("q_selfcol")[perm],
        "q_nullspace": _cat("q_nullspace")[perm],
        "q_best": q_best[perm],
        "q_candidates": q_candidates[perm],
        "d": (y * q_arr)[perm],
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([cfg.sigma_p_m], dtype=np.float32),
        "sigma_r_deg": np.array([cfg.sigma_r_deg], dtype=np.float32),
    }


def save_ird_gt(path: str | Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if meta is not None:
        path.with_suffix(".yaml").write_text(
            yaml.safe_dump(meta, sort_keys=False), encoding="utf-8"
        )
    return path


def load_ird_gt(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: data[k] for k in data.files}


def make_synthetic_ird_gt(
    n: int = 4096,
    *,
    seed: int = 0,
    reach_radius: float = 0.6,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    r6 = np.tile(np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), (n, 1))
    features = np.concatenate([t, r6], axis=1)
    dist = np.linalg.norm(t, axis=1)
    y = (dist < reach_radius).astype(np.float32)
    # Continuous signed margin (not saturated ±const)
    m_gt = np.clip(
        (reach_radius - dist) / max(reach_radius, 1e-6) * 3.0,
        -3.0,
        3.0,
    ).astype(np.float32)
    q = np.clip(1.0 - dist / (reach_radius + 1e-6), 0.0, 1.0).astype(np.float32) * y
    aabb_lo = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    aabb_hi = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return {
        "features": features,
        "reachable": y,
        "p_reach": y,
        "m_gt": m_gt,
        "q": q,
        "q_comfort": q,
        "q_manip": q,
        "q_joint": q,
        "q_selfcol": q,
        "q_nullspace": q,
        "q_best": np.zeros((n, 7), dtype=np.float32),
        "q_candidates": np.zeros((n, 4, 7), dtype=np.float32),
        "d": y * q,
        "aabb_lo": aabb_lo,
        "aabb_hi": aabb_hi,
        "sigma_p_m": np.array([0.03], dtype=np.float32),
        "sigma_r_deg": np.array([10.0], dtype=np.float32),
    }
```


## FILE: `ird_playground/ird_playground/ird/capability_io.py`

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


## FILE: `ird_playground/ird_playground/ird/map_loader.py`

```python
"""Resolve capability-map directories."""

from __future__ import annotations

from pathlib import Path


def resolve_map_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        raise NotADirectoryError(p)
    if not (p / "manifest.yaml").exists():
        raise FileNotFoundError(f"missing manifest.yaml under {p}")
    return p
```


## FILE: `ird_playground/ird_playground/ird/query_base.py`

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
    """dT (4,4) torch → features (9,)."""
    t = dT[:3, 3]
    R = dT[:3, :3]
    r6 = torch.cat([R[:, 0], R[:, 1]], dim=0)
    return torch.cat([t, r6], dim=0)


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
    _, _, score = neural_ird.model(feat)
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


## FILE: `ird_playground/ird_playground/cli/build_ird_gt.py`

```python
"""Export IRD GT NPZ from a capability map (sampling from YAML)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ird_playground.ird.export_gt import IrdGtConfig, export_ird_gt_from_capability_map, save_ird_gt
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

    n_int = samp.get("n_interior")
    n_bnd = samp.get("n_boundary")
    n_ext = samp.get("n_exterior")
    n_pos = int(samp.get("n_positive", 700_000))
    n_neg = int(samp.get("n_negative", 500_000))
    if n_int is None and n_bnd is None and n_ext is None:
        n_tot = n_pos + n_neg
        n_int = int(round(0.35 * n_tot))
        n_bnd = int(round(0.40 * n_tot))
        n_ext = max(0, n_tot - n_int - n_bnd)
    else:
        n_int = int(n_int or 0)
        n_bnd = int(n_bnd or 0)
        n_ext = int(n_ext or 0)

    cfg = IrdGtConfig(
        n_interior=n_int,
        n_boundary=n_bnd,
        n_exterior=n_ext,
        n_positive=n_pos,
        n_negative=n_neg,
        max_orients_per_voxel=int(samp.get("max_orients_per_voxel", 24)),
        hard_negative_frac=float(samp.get("hard_negative_frac", 0.45)),
        hard_negative_radius_m=float(samp.get("hard_negative_radius_m", 0.06)),
        sigma_p_m=float(samp.get("sigma_p_m", 0.03)),
        sigma_r_deg=float(samp.get("sigma_r_deg", 10.0)),
        boundary_d_lo=float(samp.get("boundary_d_lo", 0.008)),
        boundary_d_hi=float(samp.get("boundary_d_hi", 0.020)),
        m_clip=float(samp.get("m_clip", 3.0)),
        bbox_margin_m=float(samp.get("bbox_margin_m", 0.20)),
        comfort_from=str(samp.get("comfort_from", "auto")),
        k_candidates=int(samp.get("k_candidates", 4)),
        seed=int(samp.get("seed", 0)),
    )
    return map_dir, out, cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("configs/ird_gt_config.yaml"))
    ap.add_argument("--map", type=Path, default=None, help="Override map_dir")
    ap.add_argument("--out", type=Path, default=None, help="Override out")
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
    save_ird_gt(
        out,
        arrays,
        meta={
            "map_dir": str(map_dir),
            "config": str(cfg_path),
            "n_interior": cfg.n_interior,
            "n_boundary": cfg.n_boundary,
            "n_exterior": cfg.n_exterior,
            "sigma_p_m": cfg.sigma_p_m,
            "sigma_r_deg": cfg.sigma_r_deg,
            "m_clip": cfg.m_clip,
            "boundary_d_lo": cfg.boundary_d_lo,
            "boundary_d_hi": cfg.boundary_d_hi,
            "max_orients_per_voxel": cfg.max_orients_per_voxel,
            "seed": cfg.seed,
            "n_total": int(arrays["features"].shape[0]),
            "stratification": "0.35_interior_0.40_boundary_0.25_exterior",
            "note": "m_gt = continuous log1p margin in D (not strict SDF); auto percentile band if hi≈d_max",
        },
    )
    print(f"wrote {out}  N={arrays['features'].shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```


## FILE: `ird_playground/ird_playground/cli/build_map.py`

```python
"""Patch URDF with probe TCP and optionally invoke rm75 reachability build (subprocess)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ird_playground.probe.transform import load_probe_yaml, patch_urdf_tcp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", type=Path, default=Path("configs/probe_default.yaml"))
    ap.add_argument(
        "--src-urdf",
        type=Path,
        default=None,
        help="Base 8-DOF URDF (default: rm75_control asset)",
    )
    ap.add_argument("--out-urdf", type=Path, default=Path("data/maps/RM75-probe.urdf"))
    ap.add_argument(
        "--reachability-config",
        type=Path,
        default=None,
        help="If set, run rm75 reachability build with patched URDF",
    )
    ap.add_argument("--output-map", type=Path, default=Path("data/maps/probe_capability"))
    ap.add_argument("--mc-samples", type=int, default=None, help="Override MC samples for quick builds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]  # ird_playground/
    rm75 = Path(__file__).resolve().parents[3] / "rm75_control"
    src = args.src_urdf or (
        rm75 / "rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.urdf"
    )
    probe = load_probe_yaml(args.probe if args.probe.is_absolute() else root / args.probe)
    out_urdf = args.out_urdf if args.out_urdf.is_absolute() else root / args.out_urdf
    patch_urdf_tcp(src, out_urdf, probe)
    print(f"patched URDF → {out_urdf}  probe={probe.name}")

    if args.reachability_config is None:
        return 0

    cfg_path = args.reachability_config
    if not cfg_path.is_absolute():
        # allow configs under rm75_control
        cand = rm75 / cfg_path
        cfg_path = cand if cand.exists() else root / args.reachability_config

    cmd = [
        sys.executable,
        "-m",
        "rm75_control.tools.reachability.build.cli",
        "--config",
        str(cfg_path),
        "--urdf",
        str(out_urdf),
        "--output",
        str(args.output_map if args.output_map.is_absolute() else root / args.output_map),
    ]
    if args.mc_samples is not None:
        cmd += ["--mc-samples", str(args.mc_samples)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(rm75))


if __name__ == "__main__":
    raise SystemExit(main())
```


## FILE: `ird_playground/ird_playground/neural/model.py`

```python
"""Neural IRD point field f_θ(ΔT) → (m margin logit, q comfort)."""

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


def positional_encoding_xyz(xyz: "torch.Tensor", num_freqs: int = 6) -> "torch.Tensor":
    """Fourier features on translation only; xyz shape (..., 3)."""
    freqs = (2.0 ** torch.arange(num_freqs, device=xyz.device, dtype=xyz.dtype)) * np.pi
    xb = xyz.unsqueeze(-1) * freqs
    return torch.cat([xyz, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


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
    """Generic point operator; no trajectory / body / rail inputs.

    Features: [px,py,pz, r1(3), r2(3)] — xyz should be AABB-normalized to [-1,1]
    before forward (or pass aabb to normalize inside).
    """

    def __init__(
        self,
        *,
        in_dim: int = 9,
        num_freqs: int = 6,
        hidden: int = 256,
        depth: int = 5,
        tau_m: float = 1.0,
        lambda_q: float = 0.5,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for NeuralIRDPoint")
        super().__init__()
        if in_dim != 9:
            raise ValueError("expected 9-D features (xyz + rot6D)")
        self.num_freqs = int(num_freqs)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.tau_m = float(tau_m)
        self.lambda_q = float(lambda_q)
        pe_xyz = 3 + 3 * 2 * self.num_freqs
        in_w = pe_xyz + 6
        self.stem = nn.Linear(in_w, hidden)
        self.blocks = nn.ModuleList([ResidualSiLUBlock(hidden) for _ in range(max(1, depth - 1))])
        self.head_m = nn.Linear(hidden, 1)
        self.head_q = nn.Linear(hidden, 1)
        self.register_buffer("aabb_lo", torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float32))
        self.register_buffer("aabb_hi", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))

    def set_aabb(self, lo: np.ndarray | "torch.Tensor", hi: np.ndarray | "torch.Tensor") -> None:
        self.aabb_lo.copy_(torch.as_tensor(lo, dtype=torch.float32).reshape(3))
        self.aabb_hi.copy_(torch.as_tensor(hi, dtype=torch.float32).reshape(3))

    def normalize_xyz(self, features: "torch.Tensor") -> "torch.Tensor":
        p = features[..., :3]
        r6 = features[..., 3:]
        span = (self.aabb_hi - self.aabb_lo).clamp_min(1e-6)
        p_n = 2.0 * (p - self.aabb_lo) / span - 1.0
        return torch.cat([p_n, r6], dim=-1)

    def encode(self, features: "torch.Tensor") -> "torch.Tensor":
        x = self.normalize_xyz(features)
        xyz = positional_encoding_xyz(x[..., :3], self.num_freqs)
        return torch.cat([xyz, x[..., 3:]], dim=-1)

    def forward(self, features: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Return m (logit/margin), q in [0,1], score = -softplus(-m/τ)+λq."""
        h = F.silu(self.stem(self.encode(features)))
        for block in self.blocks:
            h = block(h)
        m = self.head_m(h)
        q = torch.sigmoid(self.head_q(h))
        score = -F.softplus(-m / max(self.tau_m, 1e-6)) + self.lambda_q * q
        return m, q, score

    def score_features(self, features: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        m, q, score = self.forward(features)
        p_reach = torch.sigmoid(m)  # soft reachable probability for legacy/IoU
        return {
            "m": m,
            "q": q,
            "q_comfort": q,
            "score": score,
            "p_reach": p_reach,
            "d": score,  # legacy alias used by older callers
        }


@dataclass
class PointScore:
    m: float
    q: float
    score: float
    p_reach: float = 0.0
    q_comfort: float = 0.0
    d: float = 0.0


class NeuralIRD:
    """Production wrapper: score(delta_T) + region_score via Region A."""

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
        aabb = cfg.get("aabb")
        model = NeuralIRDPoint(
            in_dim=int(cfg.get("in_dim", 9)),
            num_freqs=int(cfg.get("num_freqs", 6)),
            hidden=int(cfg.get("hidden", 256)),
            depth=int(cfg.get("depth", 5)),
            tau_m=float(cfg.get("tau_m", 1.0)),
            lambda_q=float(cfg.get("lambda_q", 0.5)),
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
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
            "in_dim": 9,
            "num_freqs": self.model.num_freqs,
            "hidden": self.model.hidden,
            "depth": self.model.depth,
            "tau_m": self.model.tau_m,
            "lambda_q": self.model.lambda_q,
            "aabb": {
                "lo": self.model.aabb_lo.detach().cpu().numpy().tolist(),
                "hi": self.model.aabb_hi.detach().cpu().numpy().tolist(),
            },
        }
        payload = {
            "state_dict": self.model.state_dict(),
            "model_cfg": cfg,
            "meta": meta or {},
        }
        torch.save(payload, path)

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
        )

    def score_batch_delta_T(self, delta_Ts: np.ndarray) -> dict[str, np.ndarray]:
        from ird_playground.probe.se3 import batch_features_from_delta_T

        feats = batch_features_from_delta_T(delta_Ts)
        return self.score_features_np(feats)

    def region_score(self, **kwargs):
        from ird_playground.region.aggregate import region_score_a

        return region_score_a(self, **kwargs)
```


## FILE: `ird_playground/ird_playground/neural/train.py`

```python
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
    # loss weights
    lambda_cls: float = 1.0
    lambda_margin: float = 1.0
    lambda_q: float = 1.0
    lambda_local: float = 0.05
    sigma_local_m: float = 0.06
    hardneg_every: int = 20
    hardneg_frac: float = 0.02
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
        epochs=int(train.get("epochs", 200)),
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
        hardneg_every=int(train.get("hardneg_every", 20)),
        hardneg_frac=float(train.get("hardneg_frac", 0.02)),
        seed=int(train.get("seed", 42)),
        device=_normalize_device(train.get("device")),
        lambda_cls=float(loss.get("lambda_cls", 1.0)),
        lambda_margin=float(loss.get("lambda_margin", 1.0)),
        lambda_q=float(loss.get("lambda_q", 1.0)),
        lambda_local=float(loss.get("lambda_local", 0.05)),
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
    w = y
    L_q = ((q - q_gt) ** 2 * w).sum() / (w.sum() + 1e-6)
    L_local = m.new_tensor(0.0)
    if local_pair is not None:
        dm_pred, dm_gt, mask = local_pair
        if mask is not None and mask.any():
            L_local = torch.nn.functional.mse_loss(dm_pred[mask], dm_gt[mask])
        elif mask is None:
            L_local = torch.nn.functional.mse_loss(dm_pred, dm_gt)
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


def _spatial_local_pair(
    m: "torch.Tensor",
    m_gt: "torch.Tensor",
    xyz: "torch.Tensor",
    *,
    sigma_m: float,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"] | None:
    """Within-batch nearest neighbour in xyz; only pairs with ||Δp|| < sigma_m.

    Returns (Δm_pred, Δm_gt, mask). This is true local consistency — not a
    random shuffle of the batch.
    """
    m = m.squeeze(-1)
    B = m.shape[0]
    if B < 2:
        return None
    # (B,B) squared distances
    d2 = torch.cdist(xyz, xyz, p=2).pow(2)
    d2 = d2 + torch.eye(B, device=xyz.device, dtype=xyz.dtype) * 1e6
    nn = d2.argmin(dim=1)
    dist = torch.sqrt(d2.gather(1, nn.unsqueeze(1)).squeeze(1).clamp_min(0.0))
    mask = dist < float(sigma_m)
    dm_pred = m - m[nn]
    dm_gt = m_gt - m_gt[nn]
    return dm_pred, dm_gt, mask


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
            hard_idx = None
            if (
                cfg.hardneg_every > 0
                and epoch > 0
                and epoch % cfg.hardneg_every == 0
            ):
                src = model._orig_mod if hasattr(model, "_orig_mod") else model
                hard_idx = _mine_hard_negatives(
                    src, train["features"], train[yk], device, cfg.hardneg_frac
                )
                print(
                    f"[hardneg] epoch={epoch} mined {hard_idx.size} examples",
                    flush=True,
                )
            # Rebuild loader each epoch so hard-neg only applies on mining epochs
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
                local_pair = None
                if cfg.lambda_local > 0:
                    local_pair = _spatial_local_pair(
                        m, m_gt, x[:, :3], sigma_m=cfg.sigma_local_m
                    )
                loss, parts = _compute_loss(
                    m, q, y, m_gt, q_gt, cfg, local_pair=local_pair
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
                            "train/L_local": parts["L_local"],
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
                        f"q={parts['L_q']:.3f} loc={parts['L_local']:.3f} "
                        f"lr={opt.param_groups[0]['lr']:.2e}"
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
```


## FILE: `ird_playground/ird_playground/neural/metrics.py`

```python
"""Eval helpers: regression metrics + optimization-oriented P2 checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ird_playground.neural.train import differentiability_smoke, evaluate_point_field

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass
class PassThresholds:
    mae_max: float = 0.35
    spearman_min: float = 0.70
    boundary_iou_min: float = 0.50
    grad_cosine_min: float = 0.30
    ascent_improve_min: float = 0.40
    rail_ad_fd_rel_max: float = 0.25
    rail_sign_agree_min: float = 0.80
    region_improve_min: float = 0.40


def point_field_pass(metrics: dict[str, float], thr: PassThresholds | None = None) -> bool:
    thr = thr or PassThresholds()
    checks = [
        metrics.get("mae", 1e9) <= thr.mae_max,
        metrics.get("spearman", 0.0) >= thr.spearman_min,
        metrics.get("boundary_iou", 0.0) >= thr.boundary_iou_min,
    ]
    if "grad_cosine_median" in metrics:
        checks.append(metrics["grad_cosine_median"] >= thr.grad_cosine_min)
    if "ascent_improve_rate" in metrics:
        checks.append(metrics["ascent_improve_rate"] >= thr.ascent_improve_min)
    if "rail_ad_fd_rel" in metrics:
        checks.append(metrics["rail_ad_fd_rel"] <= thr.rail_ad_fd_rel_max)
    if "rail_sign_agree" in metrics:
        checks.append(metrics["rail_sign_agree"] >= thr.rail_sign_agree_min)
    if "region_improve_rate" in metrics:
        checks.append(metrics["region_improve_rate"] >= thr.region_improve_min)
    return all(checks)


def grad_cosine_vs_gt(
    net,
    arrays: dict[str, np.ndarray],
    *,
    n: int = 256,
    eps: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """Median cos(∇_xyz m_θ, ∇_xyz m_gt) via central differences on GT labels."""
    if torch is None:
        raise ImportError("torch required")
    rng = np.random.default_rng(seed)
    feats = arrays["features"]
    m_gt = arrays["m_gt"] if "m_gt" in arrays else arrays["d"]
    idx = rng.choice(feats.shape[0], size=min(n, feats.shape[0]), replace=False)
    cosines = []
    net.model.eval()
    for i in idx:
        f0 = feats[i].astype(np.float32).copy()
        # GT FD on xyz using nearest neighbours in batch as proxy: local linear
        # Use network-free FD of interpolated GT is hard; instead FD of m_gt via
        # finite difference of network GT surrogate: compare AD to FD of the net
        # against a GT directional proxy from m labels of nearby samples.
        x = torch.tensor(f0[None, :], dtype=torch.float32, device=net.device, requires_grad=True)
        m, _, _ = net.model(x)
        m.sum().backward()
        g_theta = x.grad[0, :3].detach().cpu().numpy()

        # GT gradient proxy: central FD on xyz using nearby samples' m_gt
        g_gt = np.zeros(3, dtype=np.float64)
        for ax in range(3):
            # find approximate ∂m/∂x via local regression on σ-ball neighbours
            dxyz = feats[:, :3] - f0[:3]
            dist = np.linalg.norm(dxyz, axis=1)
            nb = np.argsort(dist)[1:32]
            # least-squares fit m ≈ m0 + g·Δx
            A = dxyz[nb]
            b = m_gt[nb] - m_gt[i]
            if A.shape[0] >= 3:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
                g_gt = sol[:3]
                break
        else:
            # fallback: FD of network score (self-consistency) — skip
            continue
        n1 = np.linalg.norm(g_theta) + 1e-12
        n2 = np.linalg.norm(g_gt) + 1e-12
        cosines.append(float(np.dot(g_theta, g_gt) / (n1 * n2)))
    arr = np.asarray(cosines, dtype=np.float64) if cosines else np.array([0.0])
    return {
        "grad_cosine_median": float(np.median(arr)),
        "grad_cosine_mean": float(np.mean(arr)),
        "grad_cosine_n": float(arr.size),
    }


def ascent_gt_improve(
    net,
    arrays: dict[str, np.ndarray],
    *,
    n: int = 128,
    step: float = 0.01,
    seed: int = 0,
) -> dict[str, float]:
    """From unreachable points, take one ∇m ascent step; measure GT m improve rate."""
    if torch is None:
        raise ImportError("torch required")
    rng = np.random.default_rng(seed)
    feats = arrays["features"]
    y = arrays["reachable"] if "reachable" in arrays else arrays["p_reach"]
    m_gt = arrays["m_gt"] if "m_gt" in arrays else arrays["d"]
    unre = np.flatnonzero(y < 0.5)
    if unre.size == 0:
        return {"ascent_improve_rate": 1.0, "ascent_n": 0.0}
    pick = rng.choice(unre, size=min(n, unre.size), replace=False)
    improved = 0
    net.model.eval()
    for i in pick:
        f0 = feats[i].astype(np.float32).copy()
        x = torch.tensor(f0[None, :], dtype=torch.float32, device=net.device, requires_grad=True)
        m, _, _ = net.model(x)
        m.sum().backward()
        g = x.grad[0, :3].detach().cpu().numpy()
        g = g / (np.linalg.norm(g) + 1e-12)
        f1 = f0.copy()
        f1[:3] = f1[:3] + step * g
        # GT improve: nearest neighbour m_gt after move
        d0 = np.linalg.norm(feats[:, :3] - f0[:3], axis=1)
        d1 = np.linalg.norm(feats[:, :3] - f1[:3], axis=1)
        m0 = float(m_gt[i])
        m1 = float(m_gt[int(np.argmin(d1))])
        if m1 > m0 + 1e-4:
            improved += 1
    return {
        "ascent_improve_rate": float(improved / max(len(pick), 1)),
        "ascent_n": float(len(pick)),
    }


def rail_y_ad_vs_fd(
    net,
    *,
    n: int = 32,
    rail_y: float = 0.0,
    eps: float = 1e-3,
    seed: int = 0,
) -> dict[str, float]:
    """AD ∂score/∂rail_y vs central FD; also sign agreement."""
    from ird_playground.ird.query_base import rail_y_grad_ad_fd

    return rail_y_grad_ad_fd(net, n=n, rail_y=rail_y, eps=eps, seed=seed)


def region_softmin_improve(
    net,
    *,
    n: int = 24,
    seed: int = 0,
) -> dict[str, float]:
    """Perturb region centre along −∇ softmin(m); count softmin increases."""
    from ird_playground.probe.se3 import mat4_from_Rt
    from ird_playground.region.aggregate import region_score_a

    rng = np.random.default_rng(seed)
    improved = 0
    for _ in range(n):
        p = rng.uniform(-0.4, 0.4, size=3)
        T = mat4_from_Rt(np.eye(3), p)
        rs0 = region_score_a(net, T_mu=T, num_samples=16, seed=int(rng.integers(0, 1_000_000)))
        # finite-diff direction on translation via score
        if torch is None:
            break
        # nudge toward higher m_robust by small random search (smoke)
        best = rs0.m_robust
        for _k in range(4):
            dp = rng.normal(scale=0.01, size=3)
            T2 = mat4_from_Rt(np.eye(3), p + dp)
            rs1 = region_score_a(net, T_mu=T2, num_samples=16, seed=0)
            best = max(best, rs1.m_robust)
        if best > rs0.m_robust + 1e-4:
            improved += 1
    return {
        "region_improve_rate": float(improved / max(n, 1)),
        "region_n": float(n),
    }


def evaluate_optimization_suite(
    net,
    arrays: dict[str, np.ndarray],
    *,
    seed: int = 0,
) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(grad_cosine_vs_gt(net, arrays, seed=seed))
    out.update(ascent_gt_improve(net, arrays, seed=seed))
    out.update(rail_y_ad_vs_fd(net, seed=seed))
    out.update(region_softmin_improve(net, seed=seed))
    out["grad_norm"] = differentiability_smoke(net)
    return out


__all__ = [
    "PassThresholds",
    "ascent_gt_improve",
    "differentiability_smoke",
    "evaluate_optimization_suite",
    "evaluate_point_field",
    "grad_cosine_vs_gt",
    "point_field_pass",
    "rail_y_ad_vs_fd",
    "region_softmin_improve",
]
```


## FILE: `ird_playground/ird_playground/cli/train.py`

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


## FILE: `ird_playground/ird_playground/cli/eval_point.py`

```python
"""Evaluate point field vs GT + optimization-oriented P2 checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ird_playground.neural.metrics import (
    PassThresholds,
    evaluate_optimization_suite,
    point_field_pass,
)
from ird_playground.ird.export_gt import load_ird_gt, make_synthetic_ird_gt
from ird_playground.neural.model import NeuralIRD
from ird_playground.neural.train import evaluate_point_field, load_train_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/train_config.yaml"))
    ap.add_argument("--gt-npz", type=Path, default=None)
    ap.add_argument("--synthetic-n", type=int, default=2048)
    ap.add_argument("--skip-opt", action="store_true", help="Skip gradient/rail/region suite")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else root / args.config
    train_cfg = load_train_config(cfg_path, root=root) if cfg_path.exists() else None

    ckpt = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    net = NeuralIRD.load(ckpt)

    if args.gt_npz is not None:
        arrays = load_ird_gt(args.gt_npz if args.gt_npz.is_absolute() else root / args.gt_npz)
    elif train_cfg is not None and train_cfg.gt_npz:
        arrays = load_ird_gt(train_cfg.gt_npz)
    else:
        arrays = make_synthetic_ird_gt(args.synthetic_n, seed=1)

    metrics = evaluate_point_field(net, arrays)
    if not args.skip_opt:
        metrics.update(evaluate_optimization_suite(net, arrays, seed=0))

    thr = PassThresholds(
        mae_max=train_cfg.mae_max if train_cfg else 0.35,
        spearman_min=train_cfg.spearman_min if train_cfg else 0.70,
        boundary_iou_min=train_cfg.boundary_iou_min if train_cfg else 0.70,
        grad_cosine_min=train_cfg.grad_cosine_min if train_cfg else 0.30,
        ascent_improve_min=train_cfg.ascent_improve_min if train_cfg else 0.40,
        rail_ad_fd_rel_max=train_cfg.rail_ad_fd_rel_max if train_cfg else 0.25,
        rail_sign_agree_min=train_cfg.rail_sign_agree_min if train_cfg else 0.80,
        region_improve_min=train_cfg.region_improve_min if train_cfg else 0.40,
    )
    ok = point_field_pass(metrics, thr)
    metrics["pass"] = ok
    metrics["thresholds"] = {
        "mae_max": thr.mae_max,
        "spearman_min": thr.spearman_min,
        "boundary_iou_min": thr.boundary_iou_min,
        "grad_cosine_min": thr.grad_cosine_min,
        "ascent_improve_min": thr.ascent_improve_min,
        "rail_ad_fd_rel_max": thr.rail_ad_fd_rel_max,
        "rail_sign_agree_min": thr.rail_sign_agree_min,
        "region_improve_min": thr.region_improve_min,
    }
    report = root / "data/reports/eval_point.json"
    report.parent.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj):
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, (int, float)):
            return obj
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    payload = _jsonify(metrics)
    report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```


## FILE: `ird_playground/ird_playground/probe/se3.py`

```python
"""SE(3) helpers: ΔT features, Exp map, 6D rotation encoding."""

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
    """(9,) = translation(3) + rot6d(6)."""
    T = np.asarray(delta_T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([T[:3, 3], rot6d_from_R(T[:3, :3])], axis=0)


def batch_features_from_delta_T(delta_Ts: np.ndarray) -> np.ndarray:
    """(N,9) from (N,4,4)."""
    Ts = np.asarray(delta_Ts, dtype=np.float64)
    if Ts.ndim == 2:
        return features_from_delta_T(Ts)[None, :]
    out = np.empty((Ts.shape[0], 9), dtype=np.float64)
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
    # Pick a stable tangent.
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)
```


## FILE: `ird_playground/ird_playground/probe/transform.py`

```python
"""Parameterized link7 → TCP SE(3) probe transform."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ProbeTransform:
    """Rigid transform from parent (link7) to TCP."""

    name: str
    translation_m: np.ndarray  # (3,)
    quaternion_xyzw: np.ndarray  # (4,)
    parent_frame: str = "link_7"
    child_frame: str = "tcp"

    def rotation_matrix(self) -> np.ndarray:
        return Rotation.from_quat(self.quaternion_xyzw).as_matrix()

    def matrix(self) -> np.ndarray:
        """4×4 T_parent_tcp (TCP pose expressed in parent)."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation_matrix()
        T[:3, 3] = self.translation_m
        return T

    def pose6_xyz_rpy(self, *, euler_order: str = "xyz") -> np.ndarray:
        """[x,y,z,rx,ry,rz] for RealMan / Pinocchio tcp offset APIs."""
        if (
            euler_order == "xyz"
            and abs(self.quaternion_xyzw[0]) < 1e-9
            and abs(self.quaternion_xyzw[2]) < 1e-9
            and abs(abs(self.quaternion_xyzw[1]) - abs(self.quaternion_xyzw[3])) < 1e-6
        ):
            rpy = np.array([0.0, 0.5 * np.pi, 0.0])
        else:
            rpy = Rotation.from_quat(self.quaternion_xyzw).as_euler(euler_order, degrees=False)
        return np.concatenate([self.translation_m, rpy]).astype(np.float64)

    def urdf_xyz_rpy(self) -> tuple[str, str]:
        """Strings for `<origin xyz=... rpy=.../>` (URDF fixed-axis RPY = xyz)."""
        rpy = self.pose6_xyz_rpy(euler_order="xyz")[3:]
        xyz = " ".join(f"{v:.8f}" for v in self.translation_m)
        rpy_s = " ".join(f"{v:.8f}" for v in rpy)
        return xyz, rpy_s


def default_ultrasound_probe() -> ProbeTransform:
    """Trans_z(0.07)·Rot_y(+π/2)·Trans_z(0.05) composed origin/orientation."""
    Tz1 = np.eye(4)
    Tz1[2, 3] = 0.07
    Ry = np.eye(4)
    Ry[:3, :3] = Rotation.from_euler("y", 0.5 * np.pi).as_matrix()
    Tz2 = np.eye(4)
    Tz2[2, 3] = 0.05
    T = Tz1 @ Ry @ Tz2
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return ProbeTransform(
        name="ultrasound_probe_default",
        translation_m=T[:3, 3].copy(),
        quaternion_xyzw=quat.astype(np.float64),
    )


def load_probe_yaml(path: str | Path) -> ProbeTransform:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    t = np.asarray(raw["translation_m"], dtype=np.float64).reshape(3)
    q = np.asarray(raw["quaternion_xyzw"], dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
    return ProbeTransform(
        name=str(raw.get("name", Path(path).stem)),
        translation_m=t,
        quaternion_xyzw=q,
        parent_frame=str(raw.get("parent_frame", "link_7")),
        child_frame=str(raw.get("child_frame", "tcp")),
    )


def patch_urdf_tcp(
    src_urdf: str | Path,
    dst_urdf: str | Path,
    probe: ProbeTransform,
    *,
    joint_name: str = "link_7_to_tcp",
    add_probe_visual: bool = False,
    probe_length_m: float = 0.05,
    probe_radius_m: float = 0.012,
) -> Path:
    """Rewrite the fixed joint origin for ``joint_name`` and write ``dst_urdf``.

    If ``add_probe_visual``, replace an empty ``<link name="tcp" />`` with a short
    cylinder along TCP +Z so the horizontal mount is visible in PyVista/Genesis.
    """
    import re

    text = Path(src_urdf).read_text(encoding="utf-8")
    xyz, rpy = probe.urdf_xyz_rpy()
    pattern = re.compile(
        rf'(<joint\s+name="{re.escape(joint_name)}"[^>]*>\s*)'
        r"<origin\s+[^/]*/>",
        re.DOTALL,
    )
    repl = rf'\1<origin xyz="{xyz}" rpy="{rpy}" />'
    new_text, n = pattern.subn(repl, text, count=1)
    if n != 1:
        raise ValueError(f"could not patch joint {joint_name!r} in {src_urdf}")

    if add_probe_visual:
        half = 0.5 * float(probe_length_m)
        visual = (
            '<link name="tcp">\n'
            "    <visual>\n"
            f'      <origin xyz="0 0 {half:.6f}" rpy="0 0 0" />\n'
            "      <geometry>\n"
            f'        <cylinder length="{float(probe_length_m):.6f}" '
            f'radius="{float(probe_radius_m):.6f}" />\n'
            "      </geometry>\n"
            '      <material name="probe_cyan"><color rgba="0.2 0.75 0.85 1"/></material>\n'
            "    </visual>\n"
            "  </link>"
        )
        new_text2, n2 = re.subn(
            r'<link\s+name="tcp"\s*/>',
            visual,
            new_text,
            count=1,
        )
        if n2 != 1:
            # already has a tcp link body — leave geometry as-is
            pass
        else:
            new_text = new_text2

    dst = Path(dst_urdf)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(new_text, encoding="utf-8")
    return dst


def ensure_probe_visual_urdf(
    *,
    playground_root: str | Path,
    probe_yaml: str | Path | None = None,
    out_name: str = "RM75-probe.genesis.urdf",
) -> Path:
    """Genesis-mesh URDF with horizontal ultrasound TCP + cylinder glyph.

    Mesh ``filename`` entries are rewritten to absolute paths so the file can
    live under ``ird_playground/data/maps/`` without breaking PyVista.
    """
    import re

    root = Path(playground_root).resolve()
    rm75 = root.parent / "rm75_control"
    src = rm75 / "rm75_control/assets/robots/rm75_6f_8dof/RM75-6F-8dof.genesis.urdf"
    if not src.is_file():
        raise FileNotFoundError(src)
    yaml_path = Path(probe_yaml) if probe_yaml else root / "configs/probe_default.yaml"
    if not yaml_path.is_absolute():
        yaml_path = root / yaml_path
    probe = load_probe_yaml(yaml_path)
    out = root / "data/maps" / out_name
    patch_urdf_tcp(src, out, probe, add_probe_visual=True)

    mesh_root = src.parent
    text = out.read_text(encoding="utf-8")

    def _abs_mesh(m: re.Match[str]) -> str:
        rel = m.group(1)
        if Path(rel).is_absolute():
            return m.group(0)
        abs_p = (mesh_root / rel).resolve()
        return f'filename="{abs_p}"'

    text = re.sub(r'filename="([^"]+)"', _abs_mesh, text)
    out.write_text(text, encoding="utf-8")
    return out
```


## FILE: `ird_playground/ird_playground/region/aggregate.py`

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


## FILE: `ird_playground/tests/test_core.py`

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
    # TCP +Z should align with link7 +X
    assert np.allclose(R[:, 2], [1.0, 0.0, 0.0], atol=1e-6)


def test_probe_yaml_roundtrip(tmp_path):
    root = tmp_path / "probe.yaml"
    src = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "probe_default.yaml"
    )
    if not src.exists():
        pytest.skip("configs/probe_default.yaml missing")
    p = load_probe_yaml(src)
    assert p.name.startswith("ultrasound")


def test_delta_T_features_dim():
    R = complete_frame_from_tool_axis([0, 0, 1])
    T = mat4_from_Rt(R, [0.3, 0.1, 0.2])
    dT = delta_T_tcp_inv_base(T)
    f = features_from_delta_T(dT)
    assert f.shape == (9,)


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
    # legacy wrapper still exposes softmin < mean
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
        warmup_steps=0,
        lr=3e-3,
        hardneg_every=0,
        checkpoint=str(ckpt),
        seed=0,
        num_workers=0,
    )
    result = train_point_field(cfg)
    assert ckpt.exists()
    assert result["val_metrics"]["mae_m"] < 1.5
    assert result["val_metrics"]["boundary_iou"] > 0.3

    net = NeuralIRD.load(ckpt)
    g = differentiability_smoke(net)
    assert g >= 0.0

    T_mu = mat4_from_Rt(np.eye(3), np.array([0.2, 0.0, 0.1]))
    rs = region_score_a(net, T_mu=T_mu, num_samples=16, seed=0)
    assert np.isfinite(rs.score)
    assert np.isfinite(rs.m_robust)
    assert 0.0 <= rs.q_region <= 1.0
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
    assert cfg.hidden >= 64
    assert cfg.batch_size >= 1


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

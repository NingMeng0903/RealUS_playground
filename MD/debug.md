# Neural IRD RM75 — 第三方审查归档 (v6 GT + Phase A/B)

Generated: 2026-07-20 17:28:16

Repository: `/media/camp/EXT_DRIVE/RealUS_playground`  
Package: `ird_playground/`  
Map: `rm75_6f_1p5cm_15deg_coll_probe` (1.5 cm voxel, 642 orient, 5-DoF, collision+probe)

---

## 1. Executive summary

### 1.1 Problem & fix history

| Version | Root issue | Peak val IoU |
|---|---|---|
| v4 | `bit=0` treated as unreachable; jitter 96% neg | ~0.29 |
| v5 | Trusted boundary via `soft_neg<=0.05` **inverted** (isolated MC hits) | ~0.56 |
| **v6** | **C+≥3 & C-=0** half-neighborhoods; physical PE 48…1.5 cm; fixed val | **~0.844** |

MC map: 4.8M hits / 268M bins (1.8%). `bit=0` ≠ IK-unreachable.

### 1.2 Phase A (cls-only) — final metrics

Checkpoint: `ird_playground/data/checkpoints/best_iou.pt`  
Config: `configs/train_config.yaml`  
Wandb: `run-20260720_171010-yqv3lave` (`neural_ird_v6_stable_support`)

| Metric | Value |
|---|---|
| iou / iou@cal | **0.8438** |
| PR-AUC | **0.9300** |
| bnd_pos_recall | 0.920 |
| bnd_neg_spec | 0.929 |
| reach_accuracy | 0.896 |
| n (val full eval) | 125398 |

Peak iou@cal @ epoch 10 (of 40 epochs).

### 1.3 Phase B (cls + margin + q) — final metrics

Checkpoint: `ird_playground/data/checkpoints/phase_b_latest.pt`  
Config: `configs/train_phase_b.yaml` (λ_cls=1, λ_margin=0.25, λ_q=0.1)  
Wandb: `run-20260720_171633-062f6gy6` (`neural_ird_v6_margin_q`)  
**Note:** Phase B started **cold** (no init from Phase A best_iou.pt).

| Metric | Value |
|---|---|
| iou / iou@cal | **0.8443** |
| PR-AUC | **0.9264** |
| boundary_margin_mae | **0.0281** (~0.8 mm, σ_p=3 cm) |
| mae_q | 0.0616 |
| spearman(q) | 0.758 |
| bnd_pos_recall | 0.928 |
| bnd_neg_spec | 0.925 |

Peak iou@cal @ epoch 19.  
Best margin_mae @ epoch 13.

### 1.4 GT v6 contract

- **Positive:** exact MC hit; face interior; jitter_pos
- **Trusted boundary:** C+≥3 AND C-=0 on non-overlapping half-neighborhoods (6 spatial × 7 orient KNN)
- **Exterior:** soft≈0 bit=0 or off-map
- **Unknown:** not exported (no cls_weight=0 rows in NPZ)
- **Features:** natural 6-D `[p_base,tcp, u_base]` (5-DoF, no roll)
- **N=836,820** after stable-support filter

### 1.5 Known limits for downstream NLP

- Labels are MC stable-support, not full IK density
- Operator is 5-DoF; insensitive to probe roll α
- Region queries: not implemented (use MC multi-point average or future IPE)

---

## 2. GT build log (v6)

```
[gt] unpack bitmask M=417,201 n_orient=642
[gt] MC hits=4,814,538 (1.798% of 267,843,042 sparse bins)
[gt] stable-support filter (C+, C-)
[gt] support_pos quantiles=[1, 1, 2, 3, 12] support_neg quantiles=[0, 0, 1, 2, 13]
[gt] trusted faces=34,205/400,000 (C+>=3 & C-=0); rejected=365,795
[gt] boundary capped 800,000 → 68,410
[gt] N=836,820 reach=0.441 supervised=1.000
layers: int=300k bnd±≈34k×2 jit±=34k ext=400k
```

Command:
```bash
cd ird_playground && source env.sh
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml
```

---

## 3. Phase A training log

### 3.1 Commands

```bash
cd ird_playground && source env.sh
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml
python -m ird_playground.cli.train --config configs/train_config.yaml
```

### 3.2 Per-epoch table (from `data/reports/train_point.json`)

| epoch | train_loss | val_loss | iou@cal | pr_auc | bnd+ | bnd- | m_mae | train_iou |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4185 | 0.4889 | 0.7351 | 0.8553 | 0.929 | 0.890 | 0.1959 | 0.776 |
| 1 | 0.3262 | 0.4563 | 0.7560 | 0.8786 | 0.926 | 0.912 | 0.2572 | 0.806 |
| 2 | 0.3008 | 0.3947 | 0.7887 | 0.8941 | 0.923 | 0.920 | 0.2593 | 0.827 |
| 3 | 0.2778 | 0.3443 | 0.8076 | 0.9074 | 0.924 | 0.920 | 0.2614 | 0.847 |
| 4 | 0.2604 | 0.3081 | 0.8238 | 0.9176 | 0.922 | 0.925 | 0.2435 | 0.856 |
| 5 | 0.2477 | 0.2927 | 0.8312 | 0.9219 | 0.923 | 0.926 | 0.2699 | 0.862 |
| 6 | 0.2401 | 0.3029 | 0.8353 | 0.9259 | 0.922 | 0.929 | 0.4012 | 0.866 |
| 7 | 0.2319 | 0.2650 | 0.8398 | 0.9292 | 0.922 | 0.925 | 0.3921 | 0.870 |
| 8 | 0.2250 | 0.2734 | 0.8429 | 0.9283 | 0.926 | 0.928 | 0.4704 | 0.872 |
| 9 | 0.2196 | 0.2656 | 0.8423 | 0.9288 | 0.924 | 0.928 | 0.4354 | 0.877 |
| 10 | 0.2135 | 0.2723 | 0.8438 | 0.9300 | 0.920 | 0.929 | 0.4620 | 0.877 |
| 11 | 0.2099 | 0.2682 | 0.8436 | 0.9310 | 0.921 | 0.927 | 0.4221 | 0.878 |
| 12 | 0.2047 | 0.2688 | 0.8403 | 0.9300 | 0.916 | 0.922 | 0.3681 | 0.878 |
| 13 | 0.2002 | 0.2676 | 0.8427 | 0.9294 | 0.923 | 0.914 | 0.4379 | 0.883 |
| 14 | 0.1953 | 0.2680 | 0.8388 | 0.9304 | 0.909 | 0.921 | 0.4032 | 0.888 |
| 15 | 0.1897 | 0.2721 | 0.8385 | 0.9296 | 0.909 | 0.918 | 0.4058 | 0.889 |
| 16 | 0.1851 | 0.2707 | 0.8376 | 0.9305 | 0.905 | 0.923 | 0.4192 | 0.890 |
| 17 | 0.1802 | 0.2714 | 0.8391 | 0.9298 | 0.913 | 0.923 | 0.4417 | 0.891 |
| 18 | 0.1740 | 0.2685 | 0.8385 | 0.9290 | 0.915 | 0.918 | 0.4071 | 0.894 |
| 19 | 0.1690 | 0.2734 | 0.8353 | 0.9257 | 0.913 | 0.909 | 0.4357 | 0.898 |
| 20 | 0.1636 | 0.2781 | 0.8381 | 0.9281 | 0.913 | 0.914 | 0.4506 | 0.904 |
| 21 | 0.1571 | 0.2746 | 0.8356 | 0.9258 | 0.908 | 0.909 | 0.4413 | 0.903 |
| 22 | 0.1531 | 0.2810 | 0.8306 | 0.9213 | 0.913 | 0.900 | 0.4584 | 0.906 |
| 23 | 0.1476 | 0.2874 | 0.8324 | 0.9238 | 0.902 | 0.914 | 0.4674 | 0.908 |
| 24 | 0.1432 | 0.2878 | 0.8296 | 0.9178 | 0.903 | 0.904 | 0.4684 | 0.911 |
| 25 | 0.1395 | 0.2822 | 0.8347 | 0.9180 | 0.911 | 0.909 | 0.4782 | 0.915 |
| 26 | 0.1354 | 0.2875 | 0.8312 | 0.9158 | 0.901 | 0.906 | 0.4919 | 0.919 |
| 27 | 0.1321 | 0.2894 | 0.8358 | 0.9167 | 0.909 | 0.909 | 0.5149 | 0.921 |
| 28 | 0.1291 | 0.2909 | 0.8303 | 0.9125 | 0.903 | 0.902 | 0.5088 | 0.919 |
| 29 | 0.1265 | 0.2958 | 0.8320 | 0.9130 | 0.904 | 0.903 | 0.5291 | 0.921 |
| 30 | 0.1239 | 0.3013 | 0.8303 | 0.9080 | 0.905 | 0.896 | 0.5458 | 0.920 |
| 31 | 0.1221 | 0.3024 | 0.8300 | 0.9088 | 0.901 | 0.897 | 0.5504 | 0.923 |
| 32 | 0.1199 | 0.3031 | 0.8304 | 0.9085 | 0.899 | 0.902 | 0.5590 | 0.922 |
| 33 | 0.1190 | 0.3045 | 0.8288 | 0.9065 | 0.898 | 0.897 | 0.5596 | 0.923 |
| 34 | 0.1175 | 0.3051 | 0.8284 | 0.9071 | 0.897 | 0.899 | 0.5678 | 0.923 |
| 35 | 0.1164 | 0.3052 | 0.8295 | 0.9078 | 0.902 | 0.900 | 0.5692 | 0.924 |
| 36 | 0.1157 | 0.3080 | 0.8306 | 0.9070 | 0.902 | 0.896 | 0.5746 | 0.923 |
| 37 | 0.1159 | 0.3071 | 0.8305 | 0.9069 | 0.899 | 0.901 | 0.5734 | 0.924 |
| 38 | 0.1144 | 0.3083 | 0.8293 | 0.9065 | 0.900 | 0.899 | 0.5758 | 0.925 |
| 39 | 0.1148 | 0.3087 | 0.8294 | 0.9063 | 0.899 | 0.898 | 0.5773 | 0.924 |

### 3.3 Final val_metrics JSON (Phase A)

```json
{
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
```

### 3.4 Stdout epoch lines (wandb `run-20260720_171010-yqv3lave`)

```
epoch=0 train_loss=0.4185 val_loss=0.4889 iou@0.5=0.730 iou@cal=0.735@t=0.35 pr_auc=0.855 train_iou=0.776 bnd_pos_r=0.929 bnd_neg_s=0.890 lr=3.00e-04
epoch=1 train_loss=0.3262 val_loss=0.4563 iou@0.5=0.756 iou@cal=0.756@t=0.50 pr_auc=0.879 train_iou=0.806 bnd_pos_r=0.926 bnd_neg_s=0.912 lr=2.99e-04
epoch=2 train_loss=0.3008 val_loss=0.3947 iou@0.5=0.788 iou@cal=0.789@t=0.45 pr_auc=0.894 train_iou=0.827 bnd_pos_r=0.923 bnd_neg_s=0.920 lr=2.98e-04
epoch=3 train_loss=0.2778 val_loss=0.3443 iou@0.5=0.808 iou@cal=0.808@t=0.50 pr_auc=0.907 train_iou=0.847 bnd_pos_r=0.924 bnd_neg_s=0.920 lr=2.95e-04
epoch=4 train_loss=0.2604 val_loss=0.3081 iou@0.5=0.824 iou@cal=0.824@t=0.50 pr_auc=0.918 train_iou=0.856 bnd_pos_r=0.922 bnd_neg_s=0.925 lr=2.91e-04
epoch=5 train_loss=0.2477 val_loss=0.2927 iou@0.5=0.831 iou@cal=0.831@t=0.50 pr_auc=0.922 train_iou=0.862 bnd_pos_r=0.923 bnd_neg_s=0.926 lr=2.87e-04
epoch=6 train_loss=0.2401 val_loss=0.3029 iou@0.5=0.834 iou@cal=0.835@t=0.65 pr_auc=0.926 train_iou=0.866 bnd_pos_r=0.922 bnd_neg_s=0.929 lr=2.82e-04
epoch=7 train_loss=0.2319 val_loss=0.2650 iou@0.5=0.837 iou@cal=0.840@t=0.40 pr_auc=0.929 train_iou=0.870 bnd_pos_r=0.922 bnd_neg_s=0.925 lr=2.76e-04
epoch=8 train_loss=0.2250 val_loss=0.2734 iou@0.5=0.843 iou@cal=0.843@t=0.50 pr_auc=0.928 train_iou=0.872 bnd_pos_r=0.926 bnd_neg_s=0.928 lr=2.69e-04
epoch=9 train_loss=0.2196 val_loss=0.2656 iou@0.5=0.839 iou@cal=0.842@t=0.40 pr_auc=0.929 train_iou=0.877 bnd_pos_r=0.924 bnd_neg_s=0.928 lr=2.61e-04
epoch=10 train_loss=0.2135 val_loss=0.2723 iou@0.5=0.844 iou@cal=0.844@t=0.50 pr_auc=0.930 train_iou=0.877 bnd_pos_r=0.920 bnd_neg_s=0.929 lr=2.53e-04
epoch=11 train_loss=0.2099 val_loss=0.2682 iou@0.5=0.843 iou@cal=0.844@t=0.45 pr_auc=0.931 train_iou=0.878 bnd_pos_r=0.921 bnd_neg_s=0.927 lr=2.44e-04
epoch=12 train_loss=0.2047 val_loss=0.2688 iou@0.5=0.840 iou@cal=0.840@t=0.50 pr_auc=0.930 train_iou=0.878 bnd_pos_r=0.916 bnd_neg_s=0.922 lr=2.34e-04
epoch=13 train_loss=0.2002 val_loss=0.2676 iou@0.5=0.844 iou@cal=0.843@t=0.40 pr_auc=0.929 train_iou=0.883 bnd_pos_r=0.923 bnd_neg_s=0.914 lr=2.24e-04
epoch=14 train_loss=0.1953 val_loss=0.2680 iou@0.5=0.839 iou@cal=0.839@t=0.50 pr_auc=0.930 train_iou=0.888 bnd_pos_r=0.909 bnd_neg_s=0.921 lr=2.13e-04
epoch=15 train_loss=0.1897 val_loss=0.2721 iou@0.5=0.839 iou@cal=0.839@t=0.50 pr_auc=0.930 train_iou=0.889 bnd_pos_r=0.909 bnd_neg_s=0.918 lr=2.02e-04
epoch=16 train_loss=0.1851 val_loss=0.2707 iou@0.5=0.838 iou@cal=0.838@t=0.50 pr_auc=0.930 train_iou=0.890 bnd_pos_r=0.905 bnd_neg_s=0.923 lr=1.91e-04
epoch=17 train_loss=0.1802 val_loss=0.2714 iou@0.5=0.838 iou@cal=0.839@t=0.45 pr_auc=0.930 train_iou=0.891 bnd_pos_r=0.913 bnd_neg_s=0.923 lr=1.79e-04
epoch=18 train_loss=0.1740 val_loss=0.2685 iou@0.5=0.834 iou@cal=0.838@t=0.35 pr_auc=0.929 train_iou=0.894 bnd_pos_r=0.915 bnd_neg_s=0.918 lr=1.68e-04
epoch=19 train_loss=0.1690 val_loss=0.2734 iou@0.5=0.834 iou@cal=0.835@t=0.40 pr_auc=0.926 train_iou=0.898 bnd_pos_r=0.913 bnd_neg_s=0.909 lr=1.56e-04
epoch=20 train_loss=0.1636 val_loss=0.2781 iou@0.5=0.838 iou@cal=0.838@t=0.50 pr_auc=0.928 train_iou=0.904 bnd_pos_r=0.913 bnd_neg_s=0.914 lr=1.44e-04
epoch=21 train_loss=0.1571 val_loss=0.2746 iou@0.5=0.834 iou@cal=0.836@t=0.40 pr_auc=0.926 train_iou=0.903 bnd_pos_r=0.908 bnd_neg_s=0.909 lr=1.32e-04
epoch=22 train_loss=0.1531 val_loss=0.2810 iou@0.5=0.831 iou@cal=0.831@t=0.50 pr_auc=0.921 train_iou=0.906 bnd_pos_r=0.913 bnd_neg_s=0.900 lr=1.20e-04
epoch=23 train_loss=0.1476 val_loss=0.2874 iou@0.5=0.832 iou@cal=0.832@t=0.45 pr_auc=0.924 train_iou=0.908 bnd_pos_r=0.902 bnd_neg_s=0.914 lr=1.09e-04
epoch=24 train_loss=0.1432 val_loss=0.2878 iou@0.5=0.829 iou@cal=0.830@t=0.45 pr_auc=0.918 train_iou=0.911 bnd_pos_r=0.903 bnd_neg_s=0.904 lr=9.76e-05
epoch=25 train_loss=0.1395 val_loss=0.2822 iou@0.5=0.833 iou@cal=0.835@t=0.45 pr_auc=0.918 train_iou=0.915 bnd_pos_r=0.911 bnd_neg_s=0.909 lr=8.68e-05
epoch=26 train_loss=0.1354 val_loss=0.2875 iou@0.5=0.831 iou@cal=0.831@t=0.50 pr_auc=0.916 train_iou=0.919 bnd_pos_r=0.901 bnd_neg_s=0.906 lr=7.63e-05
epoch=27 train_loss=0.1321 val_loss=0.2894 iou@0.5=0.836 iou@cal=0.836@t=0.50 pr_auc=0.917 train_iou=0.921 bnd_pos_r=0.909 bnd_neg_s=0.909 lr=6.63e-05
epoch=28 train_loss=0.1291 val_loss=0.2909 iou@0.5=0.831 iou@cal=0.830@t=0.55 pr_auc=0.913 train_iou=0.919 bnd_pos_r=0.903 bnd_neg_s=0.902 lr=5.69e-05
epoch=29 train_loss=0.1265 val_loss=0.2958 iou@0.5=0.831 iou@cal=0.832@t=0.45 pr_auc=0.913 train_iou=0.921 bnd_pos_r=0.904 bnd_neg_s=0.903 lr=4.80e-05
epoch=30 train_loss=0.1239 val_loss=0.3013 iou@0.5=0.829 iou@cal=0.830@t=0.40 pr_auc=0.908 train_iou=0.920 bnd_pos_r=0.905 bnd_neg_s=0.896 lr=3.98e-05
epoch=31 train_loss=0.1221 val_loss=0.3024 iou@0.5=0.830 iou@cal=0.830@t=0.40 pr_auc=0.909 train_iou=0.923 bnd_pos_r=0.901 bnd_neg_s=0.897 lr=3.24e-05
epoch=32 train_loss=0.1199 val_loss=0.3031 iou@0.5=0.830 iou@cal=0.830@t=0.50 pr_auc=0.909 train_iou=0.922 bnd_pos_r=0.899 bnd_neg_s=0.902 lr=2.57e-05
epoch=33 train_loss=0.1190 val_loss=0.3045 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.907 train_iou=0.923 bnd_pos_r=0.898 bnd_neg_s=0.897 lr=1.98e-05
epoch=34 train_loss=0.1175 val_loss=0.3051 iou@0.5=0.828 iou@cal=0.828@t=0.50 pr_auc=0.907 train_iou=0.923 bnd_pos_r=0.897 bnd_neg_s=0.899 lr=1.47e-05
epoch=35 train_loss=0.1164 val_loss=0.3052 iou@0.5=0.829 iou@cal=0.830@t=0.45 pr_auc=0.908 train_iou=0.924 bnd_pos_r=0.902 bnd_neg_s=0.900 lr=1.05e-05
epoch=36 train_loss=0.1157 val_loss=0.3080 iou@0.5=0.829 iou@cal=0.831@t=0.40 pr_auc=0.907 train_iou=0.923 bnd_pos_r=0.902 bnd_neg_s=0.896 lr=7.25e-06
epoch=37 train_loss=0.1159 val_loss=0.3071 iou@0.5=0.829 iou@cal=0.830@t=0.45 pr_auc=0.907 train_iou=0.924 bnd_pos_r=0.899 bnd_neg_s=0.901 lr=4.90e-06
epoch=38 train_loss=0.1144 val_loss=0.3083 iou@0.5=0.828 iou@cal=0.829@t=0.45 pr_auc=0.906 train_iou=0.925 bnd_pos_r=0.900 bnd_neg_s=0.899 lr=3.47e-06
epoch=39 train_loss=0.1148 val_loss=0.3087 iou@0.5=0.828 iou@cal=0.829@t=0.45 pr_auc=0.906 train_iou=0.924 bnd_pos_r=0.899 bnd_neg_s=0.898 lr=3.00e-06
```

---

## 4. Phase B training log

### 4.1 Command

```bash
cd ird_playground && source env.sh
python -m ird_playground.cli.train --config configs/train_phase_b.yaml
```

### 4.2 Per-epoch table (from `data/reports/train_phase_b.json`)

| epoch | train_loss | val_loss | iou@cal | pr_auc | bnd+ | bnd- | m_mae | train_iou |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4327 | 0.4985 | 0.7339 | 0.8542 | 0.930 | 0.887 | 0.0429 | 0.777 |
| 1 | 0.3378 | 0.4755 | 0.7484 | 0.8732 | 0.925 | 0.913 | 0.0347 | 0.793 |
| 2 | 0.3167 | 0.4415 | 0.7663 | 0.8823 | 0.927 | 0.920 | 0.0325 | 0.811 |
| 3 | 0.3031 | 0.4119 | 0.7851 | 0.8908 | 0.928 | 0.920 | 0.0307 | 0.828 |
| 4 | 0.2871 | 0.3634 | 0.8040 | 0.9032 | 0.922 | 0.924 | 0.0294 | 0.844 |
| 5 | 0.2749 | 0.3476 | 0.8127 | 0.9074 | 0.919 | 0.929 | 0.0319 | 0.852 |
| 6 | 0.2678 | 0.3513 | 0.8169 | 0.9072 | 0.920 | 0.931 | 0.0294 | 0.855 |
| 7 | 0.2576 | 0.3030 | 0.8302 | 0.9177 | 0.928 | 0.925 | 0.0295 | 0.862 |
| 8 | 0.2480 | 0.2928 | 0.8330 | 0.9225 | 0.924 | 0.929 | 0.0293 | 0.864 |
| 9 | 0.2411 | 0.2866 | 0.8362 | 0.9251 | 0.923 | 0.929 | 0.0281 | 0.868 |
| 10 | 0.2342 | 0.2889 | 0.8387 | 0.9272 | 0.926 | 0.929 | 0.0295 | 0.869 |
| 11 | 0.2308 | 0.2812 | 0.8375 | 0.9268 | 0.926 | 0.931 | 0.0285 | 0.869 |
| 12 | 0.2260 | 0.2795 | 0.8391 | 0.9288 | 0.922 | 0.928 | 0.0285 | 0.869 |
| 13 | 0.2222 | 0.2848 | 0.8401 | 0.9263 | 0.924 | 0.929 | 0.0275 | 0.872 |
| 14 | 0.2184 | 0.2741 | 0.8414 | 0.9286 | 0.925 | 0.929 | 0.0295 | 0.873 |
| 15 | 0.2150 | 0.2862 | 0.8413 | 0.9272 | 0.925 | 0.931 | 0.0290 | 0.876 |
| 16 | 0.2126 | 0.2769 | 0.8436 | 0.9286 | 0.929 | 0.930 | 0.0280 | 0.878 |
| 17 | 0.2102 | 0.2770 | 0.8408 | 0.9275 | 0.923 | 0.929 | 0.0284 | 0.882 |
| 18 | 0.2065 | 0.2716 | 0.8412 | 0.9274 | 0.923 | 0.925 | 0.0293 | 0.881 |
| 19 | 0.2048 | 0.2794 | 0.8443 | 0.9264 | 0.928 | 0.925 | 0.0281 | 0.881 |
| 20 | 0.2022 | 0.2757 | 0.8418 | 0.9276 | 0.922 | 0.921 | 0.0296 | 0.884 |
| 21 | 0.1979 | 0.2769 | 0.8410 | 0.9270 | 0.915 | 0.926 | 0.0307 | 0.881 |
| 22 | 0.1963 | 0.2749 | 0.8436 | 0.9291 | 0.915 | 0.931 | 0.0298 | 0.885 |
| 23 | 0.1929 | 0.2717 | 0.8435 | 0.9277 | 0.916 | 0.929 | 0.0289 | 0.888 |
| 24 | 0.1899 | 0.2767 | 0.8407 | 0.9269 | 0.917 | 0.927 | 0.0295 | 0.888 |
| 25 | 0.1869 | 0.2713 | 0.8393 | 0.9279 | 0.913 | 0.918 | 0.0306 | 0.890 |
| 26 | 0.1834 | 0.2767 | 0.8414 | 0.9261 | 0.918 | 0.920 | 0.0301 | 0.894 |
| 27 | 0.1800 | 0.2719 | 0.8354 | 0.9265 | 0.905 | 0.916 | 0.0326 | 0.897 |
| 28 | 0.1771 | 0.2736 | 0.8355 | 0.9264 | 0.906 | 0.911 | 0.0314 | 0.896 |
| 29 | 0.1729 | 0.2756 | 0.8378 | 0.9262 | 0.905 | 0.920 | 0.0315 | 0.899 |
| 30 | 0.1691 | 0.2848 | 0.8296 | 0.9234 | 0.900 | 0.906 | 0.0333 | 0.901 |
| 31 | 0.1666 | 0.2769 | 0.8359 | 0.9260 | 0.905 | 0.914 | 0.0321 | 0.903 |
| 32 | 0.1624 | 0.2832 | 0.8320 | 0.9238 | 0.895 | 0.912 | 0.0333 | 0.904 |
| 33 | 0.1595 | 0.2852 | 0.8262 | 0.9201 | 0.895 | 0.896 | 0.0360 | 0.908 |
| 34 | 0.1557 | 0.2800 | 0.8297 | 0.9237 | 0.893 | 0.908 | 0.0342 | 0.909 |
| 35 | 0.1527 | 0.2854 | 0.8301 | 0.9200 | 0.905 | 0.900 | 0.0344 | 0.910 |
| 36 | 0.1502 | 0.2880 | 0.8334 | 0.9239 | 0.896 | 0.912 | 0.0338 | 0.911 |
| 37 | 0.1469 | 0.2823 | 0.8360 | 0.9236 | 0.909 | 0.909 | 0.0328 | 0.913 |
| 38 | 0.1443 | 0.2822 | 0.8363 | 0.9239 | 0.901 | 0.920 | 0.0328 | 0.911 |
| 39 | 0.1424 | 0.2982 | 0.8273 | 0.9054 | 0.906 | 0.891 | 0.0358 | 0.912 |
| 40 | 0.1400 | 0.2928 | 0.8302 | 0.9141 | 0.902 | 0.895 | 0.0346 | 0.918 |
| 41 | 0.1377 | 0.2957 | 0.8294 | 0.9098 | 0.894 | 0.902 | 0.0352 | 0.918 |
| 42 | 0.1361 | 0.2937 | 0.8297 | 0.9068 | 0.899 | 0.904 | 0.0351 | 0.919 |
| 43 | 0.1350 | 0.2981 | 0.8287 | 0.9070 | 0.895 | 0.906 | 0.0353 | 0.920 |
| 44 | 0.1333 | 0.2958 | 0.8312 | 0.9130 | 0.894 | 0.915 | 0.0343 | 0.919 |
| 45 | 0.1317 | 0.2974 | 0.8316 | 0.9053 | 0.903 | 0.904 | 0.0343 | 0.921 |
| 46 | 0.1306 | 0.3045 | 0.8285 | 0.8992 | 0.897 | 0.898 | 0.0355 | 0.923 |
| 47 | 0.1285 | 0.3040 | 0.8309 | 0.9035 | 0.901 | 0.904 | 0.0348 | 0.920 |
| 48 | 0.1282 | 0.3027 | 0.8295 | 0.9009 | 0.902 | 0.896 | 0.0347 | 0.923 |
| 49 | 0.1276 | 0.3088 | 0.8277 | 0.8992 | 0.897 | 0.899 | 0.0355 | 0.923 |
| 50 | 0.1268 | 0.3093 | 0.8282 | 0.8985 | 0.896 | 0.902 | 0.0355 | 0.924 |
| 51 | 0.1263 | 0.3110 | 0.8286 | 0.8956 | 0.897 | 0.898 | 0.0355 | 0.925 |
| 52 | 0.1251 | 0.3084 | 0.8293 | 0.8991 | 0.899 | 0.902 | 0.0352 | 0.925 |
| 53 | 0.1249 | 0.3093 | 0.8289 | 0.8967 | 0.897 | 0.900 | 0.0352 | 0.927 |
| 54 | 0.1239 | 0.3127 | 0.8283 | 0.8942 | 0.899 | 0.898 | 0.0354 | 0.927 |
| 55 | 0.1240 | 0.3140 | 0.8271 | 0.8934 | 0.897 | 0.899 | 0.0357 | 0.926 |
| 56 | 0.1243 | 0.3123 | 0.8289 | 0.8944 | 0.901 | 0.897 | 0.0355 | 0.927 |
| 57 | 0.1234 | 0.3128 | 0.8289 | 0.8954 | 0.899 | 0.898 | 0.0354 | 0.927 |
| 58 | 0.1235 | 0.3132 | 0.8286 | 0.8946 | 0.897 | 0.899 | 0.0356 | 0.927 |
| 59 | 0.1237 | 0.3137 | 0.8289 | 0.8935 | 0.898 | 0.898 | 0.0356 | 0.927 |

### 4.3 Final val_metrics JSON (Phase B)

```json
{
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
```

### 4.4 Stdout epoch lines (wandb `run-20260720_171633-062f6gy6`)

```
epoch=0 train_loss=0.4327 val_loss=0.4985 iou@0.5=0.726 iou@cal=0.734@t=0.30 pr_auc=0.854 train_iou=0.777 bnd_pos_r=0.930 bnd_neg_s=0.887 lr=2.00e-04
epoch=1 train_loss=0.3378 val_loss=0.4755 iou@0.5=0.748 iou@cal=0.748@t=0.50 pr_auc=0.873 train_iou=0.793 bnd_pos_r=0.925 bnd_neg_s=0.913 lr=2.00e-04
epoch=2 train_loss=0.3167 val_loss=0.4415 iou@0.5=0.766 iou@cal=0.766@t=0.45 pr_auc=0.882 train_iou=0.811 bnd_pos_r=0.927 bnd_neg_s=0.920 lr=1.99e-04
epoch=3 train_loss=0.3031 val_loss=0.4119 iou@0.5=0.786 iou@cal=0.785@t=0.45 pr_auc=0.891 train_iou=0.828 bnd_pos_r=0.928 bnd_neg_s=0.920 lr=1.98e-04
epoch=4 train_loss=0.2871 val_loss=0.3634 iou@0.5=0.803 iou@cal=0.804@t=0.45 pr_auc=0.903 train_iou=0.844 bnd_pos_r=0.922 bnd_neg_s=0.924 lr=1.97e-04
epoch=5 train_loss=0.2749 val_loss=0.3476 iou@0.5=0.813 iou@cal=0.813@t=0.50 pr_auc=0.907 train_iou=0.852 bnd_pos_r=0.919 bnd_neg_s=0.929 lr=1.96e-04
epoch=6 train_loss=0.2678 val_loss=0.3513 iou@0.5=0.814 iou@cal=0.817@t=0.65 pr_auc=0.907 train_iou=0.855 bnd_pos_r=0.920 bnd_neg_s=0.931 lr=1.94e-04
epoch=7 train_loss=0.2576 val_loss=0.3030 iou@0.5=0.830 iou@cal=0.830@t=0.45 pr_auc=0.918 train_iou=0.862 bnd_pos_r=0.928 bnd_neg_s=0.925 lr=1.92e-04
epoch=8 train_loss=0.2480 val_loss=0.2928 iou@0.5=0.834 iou@cal=0.833@t=0.55 pr_auc=0.923 train_iou=0.864 bnd_pos_r=0.924 bnd_neg_s=0.929 lr=1.90e-04
epoch=9 train_loss=0.2411 val_loss=0.2866 iou@0.5=0.833 iou@cal=0.836@t=0.45 pr_auc=0.925 train_iou=0.868 bnd_pos_r=0.923 bnd_neg_s=0.929 lr=1.88e-04
epoch=10 train_loss=0.2342 val_loss=0.2889 iou@0.5=0.840 iou@cal=0.839@t=0.45 pr_auc=0.927 train_iou=0.869 bnd_pos_r=0.926 bnd_neg_s=0.929 lr=1.85e-04
epoch=11 train_loss=0.2308 val_loss=0.2812 iou@0.5=0.837 iou@cal=0.837@t=0.50 pr_auc=0.927 train_iou=0.869 bnd_pos_r=0.926 bnd_neg_s=0.931 lr=1.82e-04
epoch=12 train_loss=0.2260 val_loss=0.2795 iou@0.5=0.839 iou@cal=0.839@t=0.50 pr_auc=0.929 train_iou=0.869 bnd_pos_r=0.922 bnd_neg_s=0.928 lr=1.79e-04
epoch=13 train_loss=0.2222 val_loss=0.2848 iou@0.5=0.842 iou@cal=0.840@t=0.55 pr_auc=0.926 train_iou=0.872 bnd_pos_r=0.924 bnd_neg_s=0.929 lr=1.76e-04
epoch=14 train_loss=0.2184 val_loss=0.2741 iou@0.5=0.837 iou@cal=0.841@t=0.40 pr_auc=0.929 train_iou=0.873 bnd_pos_r=0.925 bnd_neg_s=0.929 lr=1.72e-04
epoch=15 train_loss=0.2150 val_loss=0.2862 iou@0.5=0.841 iou@cal=0.841@t=0.50 pr_auc=0.927 train_iou=0.876 bnd_pos_r=0.925 bnd_neg_s=0.931 lr=1.68e-04
epoch=16 train_loss=0.2126 val_loss=0.2769 iou@0.5=0.843 iou@cal=0.844@t=0.45 pr_auc=0.929 train_iou=0.878 bnd_pos_r=0.929 bnd_neg_s=0.930 lr=1.65e-04
epoch=17 train_loss=0.2102 val_loss=0.2770 iou@0.5=0.841 iou@cal=0.841@t=0.50 pr_auc=0.928 train_iou=0.882 bnd_pos_r=0.923 bnd_neg_s=0.929 lr=1.60e-04
epoch=18 train_loss=0.2065 val_loss=0.2716 iou@0.5=0.839 iou@cal=0.841@t=0.40 pr_auc=0.927 train_iou=0.881 bnd_pos_r=0.923 bnd_neg_s=0.925 lr=1.56e-04
epoch=19 train_loss=0.2048 val_loss=0.2794 iou@0.5=0.844 iou@cal=0.844@t=0.45 pr_auc=0.926 train_iou=0.881 bnd_pos_r=0.928 bnd_neg_s=0.925 lr=1.52e-04
epoch=20 train_loss=0.2022 val_loss=0.2757 iou@0.5=0.842 iou@cal=0.842@t=0.45 pr_auc=0.928 train_iou=0.884 bnd_pos_r=0.922 bnd_neg_s=0.921 lr=1.47e-04
epoch=21 train_loss=0.1979 val_loss=0.2769 iou@0.5=0.841 iou@cal=0.841@t=0.50 pr_auc=0.927 train_iou=0.881 bnd_pos_r=0.915 bnd_neg_s=0.926 lr=1.43e-04
epoch=22 train_loss=0.1963 val_loss=0.2749 iou@0.5=0.844 iou@cal=0.844@t=0.50 pr_auc=0.929 train_iou=0.885 bnd_pos_r=0.915 bnd_neg_s=0.931 lr=1.38e-04
epoch=23 train_loss=0.1929 val_loss=0.2717 iou@0.5=0.840 iou@cal=0.844@t=0.40 pr_auc=0.928 train_iou=0.888 bnd_pos_r=0.916 bnd_neg_s=0.929 lr=1.33e-04
epoch=24 train_loss=0.1899 val_loss=0.2767 iou@0.5=0.842 iou@cal=0.841@t=0.40 pr_auc=0.927 train_iou=0.888 bnd_pos_r=0.917 bnd_neg_s=0.927 lr=1.28e-04
epoch=25 train_loss=0.1869 val_loss=0.2713 iou@0.5=0.839 iou@cal=0.839@t=0.45 pr_auc=0.928 train_iou=0.890 bnd_pos_r=0.913 bnd_neg_s=0.918 lr=1.23e-04
epoch=26 train_loss=0.1834 val_loss=0.2767 iou@0.5=0.841 iou@cal=0.841@t=0.45 pr_auc=0.926 train_iou=0.894 bnd_pos_r=0.918 bnd_neg_s=0.920 lr=1.18e-04
epoch=27 train_loss=0.1800 val_loss=0.2719 iou@0.5=0.835 iou@cal=0.835@t=0.50 pr_auc=0.927 train_iou=0.897 bnd_pos_r=0.905 bnd_neg_s=0.916 lr=1.13e-04
epoch=28 train_loss=0.1771 val_loss=0.2736 iou@0.5=0.836 iou@cal=0.836@t=0.45 pr_auc=0.926 train_iou=0.896 bnd_pos_r=0.906 bnd_neg_s=0.911 lr=1.07e-04
epoch=29 train_loss=0.1729 val_loss=0.2756 iou@0.5=0.837 iou@cal=0.838@t=0.40 pr_auc=0.926 train_iou=0.899 bnd_pos_r=0.905 bnd_neg_s=0.920 lr=1.02e-04
epoch=30 train_loss=0.1691 val_loss=0.2848 iou@0.5=0.831 iou@cal=0.830@t=0.55 pr_auc=0.923 train_iou=0.901 bnd_pos_r=0.900 bnd_neg_s=0.906 lr=9.69e-05
epoch=31 train_loss=0.1666 val_loss=0.2769 iou@0.5=0.835 iou@cal=0.836@t=0.45 pr_auc=0.926 train_iou=0.903 bnd_pos_r=0.905 bnd_neg_s=0.914 lr=9.17e-05
epoch=32 train_loss=0.1624 val_loss=0.2832 iou@0.5=0.831 iou@cal=0.832@t=0.45 pr_auc=0.924 train_iou=0.904 bnd_pos_r=0.895 bnd_neg_s=0.912 lr=8.65e-05
epoch=33 train_loss=0.1595 val_loss=0.2852 iou@0.5=0.826 iou@cal=0.826@t=0.50 pr_auc=0.920 train_iou=0.908 bnd_pos_r=0.895 bnd_neg_s=0.896 lr=8.14e-05
epoch=34 train_loss=0.1557 val_loss=0.2800 iou@0.5=0.830 iou@cal=0.830@t=0.50 pr_auc=0.924 train_iou=0.909 bnd_pos_r=0.893 bnd_neg_s=0.908 lr=7.63e-05
epoch=35 train_loss=0.1527 val_loss=0.2854 iou@0.5=0.830 iou@cal=0.830@t=0.50 pr_auc=0.920 train_iou=0.910 bnd_pos_r=0.905 bnd_neg_s=0.900 lr=7.13e-05
epoch=36 train_loss=0.1502 val_loss=0.2880 iou@0.5=0.832 iou@cal=0.833@t=0.45 pr_auc=0.924 train_iou=0.911 bnd_pos_r=0.896 bnd_neg_s=0.912 lr=6.63e-05
epoch=37 train_loss=0.1469 val_loss=0.2823 iou@0.5=0.834 iou@cal=0.836@t=0.40 pr_auc=0.924 train_iou=0.913 bnd_pos_r=0.909 bnd_neg_s=0.909 lr=6.15e-05
epoch=38 train_loss=0.1443 val_loss=0.2822 iou@0.5=0.836 iou@cal=0.836@t=0.50 pr_auc=0.924 train_iou=0.911 bnd_pos_r=0.901 bnd_neg_s=0.920 lr=5.68e-05
epoch=39 train_loss=0.1424 val_loss=0.2982 iou@0.5=0.828 iou@cal=0.827@t=0.55 pr_auc=0.905 train_iou=0.912 bnd_pos_r=0.906 bnd_neg_s=0.891 lr=5.22e-05
epoch=40 train_loss=0.1400 val_loss=0.2928 iou@0.5=0.829 iou@cal=0.830@t=0.45 pr_auc=0.914 train_iou=0.918 bnd_pos_r=0.902 bnd_neg_s=0.895 lr=4.77e-05
epoch=41 train_loss=0.1377 val_loss=0.2957 iou@0.5=0.829 iou@cal=0.829@t=0.45 pr_auc=0.910 train_iou=0.918 bnd_pos_r=0.894 bnd_neg_s=0.902 lr=4.34e-05
epoch=42 train_loss=0.1361 val_loss=0.2937 iou@0.5=0.830 iou@cal=0.830@t=0.50 pr_auc=0.907 train_iou=0.919 bnd_pos_r=0.899 bnd_neg_s=0.904 lr=3.92e-05
epoch=43 train_loss=0.1350 val_loss=0.2981 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.907 train_iou=0.920 bnd_pos_r=0.895 bnd_neg_s=0.906 lr=3.52e-05
epoch=44 train_loss=0.1333 val_loss=0.2958 iou@0.5=0.829 iou@cal=0.831@t=0.45 pr_auc=0.913 train_iou=0.919 bnd_pos_r=0.894 bnd_neg_s=0.915 lr=3.14e-05
epoch=45 train_loss=0.1317 val_loss=0.2974 iou@0.5=0.831 iou@cal=0.832@t=0.45 pr_auc=0.905 train_iou=0.921 bnd_pos_r=0.903 bnd_neg_s=0.904 lr=2.78e-05
epoch=46 train_loss=0.1306 val_loss=0.3045 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.899 train_iou=0.923 bnd_pos_r=0.897 bnd_neg_s=0.898 lr=2.44e-05
epoch=47 train_loss=0.1285 val_loss=0.3040 iou@0.5=0.830 iou@cal=0.831@t=0.45 pr_auc=0.904 train_iou=0.920 bnd_pos_r=0.901 bnd_neg_s=0.904 lr=2.12e-05
epoch=48 train_loss=0.1282 val_loss=0.3027 iou@0.5=0.830 iou@cal=0.830@t=0.50 pr_auc=0.901 train_iou=0.923 bnd_pos_r=0.902 bnd_neg_s=0.896 lr=1.82e-05
epoch=49 train_loss=0.1276 val_loss=0.3088 iou@0.5=0.828 iou@cal=0.828@t=0.45 pr_auc=0.899 train_iou=0.923 bnd_pos_r=0.897 bnd_neg_s=0.899 lr=1.55e-05
epoch=50 train_loss=0.1268 val_loss=0.3093 iou@0.5=0.828 iou@cal=0.828@t=0.50 pr_auc=0.899 train_iou=0.924 bnd_pos_r=0.896 bnd_neg_s=0.902 lr=1.29e-05
epoch=51 train_loss=0.1263 val_loss=0.3110 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.896 train_iou=0.925 bnd_pos_r=0.897 bnd_neg_s=0.898 lr=1.07e-05
epoch=52 train_loss=0.1251 val_loss=0.3084 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.899 train_iou=0.925 bnd_pos_r=0.899 bnd_neg_s=0.902 lr=8.67e-06
epoch=53 train_loss=0.1249 val_loss=0.3093 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.897 train_iou=0.927 bnd_pos_r=0.897 bnd_neg_s=0.900 lr=6.92e-06
epoch=54 train_loss=0.1239 val_loss=0.3127 iou@0.5=0.828 iou@cal=0.828@t=0.50 pr_auc=0.894 train_iou=0.927 bnd_pos_r=0.899 bnd_neg_s=0.898 lr=5.42e-06
epoch=55 train_loss=0.1240 val_loss=0.3140 iou@0.5=0.827 iou@cal=0.827@t=0.50 pr_auc=0.893 train_iou=0.926 bnd_pos_r=0.897 bnd_neg_s=0.899 lr=4.19e-06
epoch=56 train_loss=0.1243 val_loss=0.3123 iou@0.5=0.829 iou@cal=0.829@t=0.45 pr_auc=0.894 train_iou=0.927 bnd_pos_r=0.901 bnd_neg_s=0.897 lr=3.24e-06
epoch=57 train_loss=0.1234 val_loss=0.3128 iou@0.5=0.829 iou@cal=0.829@t=0.45 pr_auc=0.895 train_iou=0.927 bnd_pos_r=0.899 bnd_neg_s=0.898 lr=2.55e-06
epoch=58 train_loss=0.1235 val_loss=0.3132 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.895 train_iou=0.927 bnd_pos_r=0.897 bnd_neg_s=0.899 lr=2.14e-06
epoch=59 train_loss=0.1237 val_loss=0.3137 iou@0.5=0.829 iou@cal=0.829@t=0.50 pr_auc=0.893 train_iou=0.927 bnd_pos_r=0.898 bnd_neg_s=0.898 lr=2.00e-06
```

---

## 5. Architecture notes (reviewer)

**Model:** `NeuralIRDPoint` — 6-D in → reach_logit, margin, q, score  
**PE:** physical wavelengths [0.48, 0.24, 0.12, 0.06, 0.03, 0.015] m on p; Fourier on u (5 bands)  
**Loss:** BCE(reach, hard y, cls_weight) + λ_m SmoothL1(m|mw>0) + λ_q SmoothL1(q|pos)  
**Val:** block-split; fixed calib/test indices; iou@0.5 and iou@cal reported separately  
**Sampler:** CyclingLayerPool (no replace) + difficulty mix by layer_id

**Bit pack/unpack:** little-endian OR in `capability_map.py` — round-trip verified.

---

## 6. Complete source code


### `ird_playground/ird/export_gt.py`

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

### `ird_playground/neural/train.py`

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

### `configs/train_cls_only.yaml`

```yaml
# Alias of train_config.yaml — v6 cls-only on stable-support labels

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
  checkpoint: data/checkpoints/cls_only_latest.pt
  checkpoint_dir: data/checkpoints
  report: data/reports/train_cls_only.json

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

### `rm75_control/.../capability_map.py (pack_bits_5dof / unpack_bits_5dof only)`

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

---

## 7. Bit round-trip test

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
assert np.array_equal(bits, unpack_bits_5dof(pack_bits_5dof(bits), 642))
```

---

## 8. File index

| Path | Role |
|---|---|
| `ird_playground/ird/export_gt.py` | GT v6 export |
| `ird_playground/neural/train.py` | Training loop |
| `ird_playground/neural/model.py` | MLP + physical PE |
| `ird_playground/data/ird/gt_samples_1p5cm_probe.npz` | GT (N=836820) |
| `ird_playground/data/checkpoints/best_iou.pt` | Phase A best |
| `ird_playground/data/checkpoints/phase_b_latest.pt` | Phase B final |
| `ird_playground/data/reports/train_point.json` | Phase A history |
| `ird_playground/data/reports/train_phase_b.json` | Phase B history |

---

*End of third-party review archive.*

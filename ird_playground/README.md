# ird_playground

**Neural Inverse Reachability Distribution** — rail-locked **7-DoF** point field + query-side Region A.

## Architecture (v4 contract)

- **Features (6-D)**: natural \([p_{\mathrm{base,tcp}}(3),\,u_{\mathrm{base}}(3)]\). Recovered from \(\Delta T\) at query; **no fake roll**.
- **Heads**: \(f_\theta \to (\ell_{\mathrm{reach}},\, m_{\mathrm{margin}},\, q)\).
- **GT**: bitmask labels; **boundary-pair continuous margin** (no dilated EDT); `margin_weight=0` on far interior/exterior; difficulty layers.
- **Loss**: `BCEWithLogits` + **masked** `SmoothL1(m)` + `SmoothL1(q|pos)`; difficulty-aware batches; best by **val IoU**.
- **Query**: \(\Delta T(r)\) → \((p,u)\) still fully differentiable w.r.t. `rail_y`.

## Commands

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground/ird_playground
source env.sh

# Rebuild GT (v4)
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml

# Phase A: classification only (default train_config.yaml)
python -m ird_playground.cli.train --config configs/train_config.yaml

# Phase B: after val_iou ≥ 0.70
python -m ird_playground.cli.train --config configs/train_phase_b.yaml

python -m ird_playground.cli.eval_point --checkpoint data/checkpoints/best_iou.pt
```

Watch **`val_iou`**, layer recalls (`bnd_pos_recall`, `bnd_neg_spec`, `jitter_acc`), not total loss.

See [`../MD/debug.md`](../MD/debug.md) and [`../MD/todo.md`](../MD/todo.md).

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

# Continuous FK/multi-seed-IK labels. The smoke config validates the pipeline;
# the production config targets 1 mm SE(3) boundary bisection.
python -m ird_playground.cli.build_continuous_gt --config configs/continuous_gt_smoke.yaml

# Phase A: classification only (default train_config.yaml)
python -m ird_playground.cli.train --config configs/train_config.yaml

# Phase B: after val_iou ≥ 0.70
python -m ird_playground.cli.train --config configs/train_phase_b.yaml

python -m ird_playground.cli.eval_point --checkpoint data/checkpoints/best_iou.pt
```

Watch **`val_iou`**, layer recalls (`bnd_pos_recall`, `bnd_neg_spec`, `jitter_acc`), not total loss.

## Physical-accuracy gate

`eval_point` reports both learned-field metrics and the spatial resolution of the
GT source. A smooth neural interpolation is not evidence of sub-voxel physical
accuracy. The current `rm75_6f_1p5cm_15deg_coll_probe` source has a 15 mm grid,
so even an optimistic voxel-center boundary lower bound is 7.5 mm. It cannot
pass the 0.1 mm acceptance target; produce continuous FK/IK boundary labels or
at least a 0.2 mm grid before making that claim.

```bash
python -m ird_playground.cli.eval_point \
  --checkpoint data/checkpoints/phase_b/selected.pt \
  --config configs/train_phase_b.yaml \
  --target-position-error-mm 0.1

# Inspect cleanup first; deletion requires --apply.
python -m ird_playground.cli.prune_artifacts --root data
```

See [`../MD/debug.md`](../MD/debug.md) and [`../MD/todo.md`](../MD/todo.md).

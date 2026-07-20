# ird_playground

**Neural Inverse Reachability Distribution** — rail-locked **7-DoF** point field + query-side Region A.

## Architecture (v3 contract)

- **Features (6-D)**: \([t_\Delta(3),\,u(3)]\) — \(\Delta T\) translation + TCP tool axis (5-DoF map; no fake roll / rot6D).
- **Heads**: \(f_\theta \to (\ell_{\mathrm{reach}},\, m_{\mathrm{margin}},\, q)\). Classification, signed margin, and capability are **separate**.
- **GT**: bitmask-exact labels; jitter re-queries voxel+orient; margin = per-orient 3D EDT / \(\sigma_p\); \(y{=}1\Rightarrow m{>}0\), \(y{=}0\Rightarrow m{<}0\); AABB from `features[:,:3]`.
- **Loss**: `BCEWithLogits(\ell,y)` + `SmoothL1(m)` + `SmoothL1(q|pos)`; `lambda_local=0`, `hardneg_every=0`.
- **Query**: \(T_{\mathrm{base}}(\mathrm{rail}_y)=T_{\mathrm{rail}}\,\mathrm{Trans}_y(\mathrm{rail}_y)\,T_{\mathrm{base},0}\), then \(\Delta T(r)\).
- **Region A**: \(m_{\mathrm{robust}}=\mathrm{softmin}(m)\), \(q_{\mathrm{region}}=\mathrm{mean}(q)\).

## Layout

```
ird_playground/
  configs/          # YAML only
  data/             # gt, checkpoints, reports (gitignored bulky)
  ird_playground/
    probe/          # link7→TCP SE(3)
    ird/            # capability IO + GT export + rail SE(3) query
    neural/         # 3-head MLP + train + metrics
    region/         # query-side Region A
    viz/            # global IRD + capability compare
    cli/
  tests/
  env.sh
```

## Commands

```bash
cd ird_playground && source env.sh

# 1) Rebuild GT (required after contract change; delete old NPZ first)
python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml

# 2a) Phase-1: classification only (IoU should rise quickly)
python -m ird_playground.cli.train --config configs/train_cls_only.yaml

# 2b) Full: cls + margin + q
python -m ird_playground.cli.train --config configs/train_config.yaml

python -m ird_playground.cli.eval_point --checkpoint data/checkpoints/best.pt
```

### GT contract check (after export)

```python
from ird_playground.ird.export_gt import load_ird_gt, assert_gt_contract
a = load_ird_gt("data/ird/gt_samples_1p5cm_probe.npz")
assert_gt_contract(a)
print(a["features"].shape, float(a["reachable"].mean()))
```

### P2 pass targets

| Metric | Target |
|--------|--------|
| Reachable classification IoU | ≥ 0.70 |
| Margin MAE | ≤ 0.35 |
| Spearman on \(q\) (reachable) | ≥ 0.70 |
| Grad cosine vs GT (median) | ≥ 0.30 |
| Ascent GT improve rate | ≥ 0.40 |
| `rail_y` AD vs FD relative err | ≤ 0.25 |
| `rail_y` sign agree | ≥ 0.80 |
| Region softmin improve rate | ≥ 0.40 |

See [`../MD/todo.md`](../MD/todo.md) and [`../MD/debug.md`](../MD/debug.md).

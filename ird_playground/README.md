# ird_playground

**Neural Inverse Reachability Distribution** — rail-locked **7-DoF** point field + query-side Region A.

## Architecture (locked)

- **Train**: \(f_\theta(\Delta T)\to(m,q)\), \(\Delta T=T_{\mathrm{tcp}}^{-1}T_{\mathrm{base}}\). No rail / patient / trajectory in the net.
- **Query**: \(T_{\mathrm{base}}(\mathrm{rail}_y)=T_{\mathrm{rail}}\,\mathrm{Trans}_y(\mathrm{rail}_y)\,T_{\mathrm{base},0}\) (full SE(3)), then \(\Delta T(r)\).
- **Region A**: fixed Sobol; \(m_{\mathrm{robust}}=\mathrm{softmin}(m)\), \(q_{\mathrm{region}}=\mathrm{mean}(q)\). IPE is ablation only.
- Runtime robot is 7-DoF arm + 1-DoF rail; the capability map and network are **rail-locked 7-DoF**.

## Layout

```
ird_playground/
  configs/          # YAML only
  data/             # gt, checkpoints, reports (gitignored bulky)
  ird_playground/
    probe/          # link7→TCP SE(3)
    ird/            # capability IO + GT export + rail SE(3) query
    neural/         # MLP (m,q) + train + metrics
    region/         # query-side Region A
    viz/            # global IRD + capability compare
    cli/
  tests/
  env.sh
```

## Commands

```bash
cd ird_playground && source env.sh

python -m ird_playground.cli.build_ird_gt --config configs/ird_gt_config.yaml
python -m ird_playground.cli.train --config configs/train_config.yaml
python -m ird_playground.cli.eval_point --checkpoint data/checkpoints/best.pt

# Global IRD / capability figures (shared clim 0–18 for probe compare)
# see rm75_control/scripts/regen_probe_compare_figs.py
```

### P2 pass targets

| Metric | Target |
|--------|--------|
| Reachable classification IoU | ≥ 0.70 |
| Margin / score MAE | ≤ 0.35 |
| Spearman on \(q\) (reachable) | ≥ 0.70 |
| Grad cosine vs GT (median) | ≥ 0.30 |
| Ascent GT improve rate | ≥ 0.40 |
| `rail_y` AD vs FD relative err | ≤ 0.25 |
| `rail_y` sign agree | ≥ 0.80 |
| Region softmin improve rate | ≥ 0.40 |

See [`../MD/todo.md`](../MD/todo.md).

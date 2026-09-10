# Active 003 selected-frame audit

Five PNGs show one scan each, in three rows: saved contact-segment +0.5 s, lowest logged left-window quality while `h_ref_s > 0` whose frame is present in the saved segment, and end−0.5 s. Columns show the complete original grayscale image with window/ROI guides, the recomputed confidence map at fixed [0,1] display scale, and the top-22%-depth column confidence curve with threshold 0.8.

The exact `study_start.config.feature.config` is reused for each scan. No new flip, crop or enhancement is applied. Twelve of the fifteen selected frames were consumed online; every exact-frame recomputed quality matches online quality with maximum absolute difference **0**. The other three explicitly show no exact-frame match; no nearby online frame is substituted.

L_PtD illustrates the distinction between visible whole-image appearance and the configured statistic. At end−0.5 s (frame 2049012, t=35.233 s), qL=0.9325 and every top-ROI column inside L exceeds 0.8 (minimum 0.8776). Yet the outermost 4% of image width, excluded from the L window, has 28.24% full-depth gray<20 occupancy; the L window itself has 5.09%. The diagram also shows deeper/local dark appearance. This supports an ROI/coverage limitation, not a failed numerical extraction or proof of noncontact. Gray occupancy is descriptive and never labeled contact truth.

Full per-frame values, selected source IDs, online control IDs and config are in `metrics.json`. Reproduce from the repository root:

```
PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python MD/contact_qp/probe50_active003/visual/render_selected.py
```

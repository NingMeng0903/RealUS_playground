# Optimized analytic-target replay

This directory is separate from the immutable numerical-reference
`../source_snapshot` and `../results`. Its `cache` link reuses only the exact
image evidence and frozen recorded control inputs. No raw data is modified.

The final optimized snapshot pins:

- `qp.py`: `8bbddf12b3358908c719979b0ee7245159f85a86d7e139c05397e80bc64f93c4`
- `priority_allocation.py`: `9bcee00a5476553396757fb7f28179e5eff73583a4c0d572d8f55c25010c9767`

All other source hashes are in `source_snapshot/source_sha256.json`. Every
result records the actually loaded hashes and the local replay-script hash.
The shortcut returns the jointly feasible exact targets only when all hard
constraints and strict energy admission hold. Other cases retain the solver.

Use the same environment prefix as the reference README:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1`.
Four concurrent invocations of this directory's `replay.py` used disjoint
groups: `--groups 001 pei --cpu 21 --follow`, `--groups 003 --cpu 22 --follow`,
`--groups 007 --cpu 23 --follow`, and `--groups jiaqi --cpu 20 --follow`.

After all 49 finish, run this directory's `report.py --require-49`, then
`../compare_optimization.py`. `../OPTIMIZATION.md` and
`../optimization_comparison.json` report the observed numerical differences,
independent hard-constraint checks, admission differences, and timing changes.
No bit-exact equality assumption is imposed on the approximate numerical
reference. Frozen-image, loading-proxy, zero-publication-latency, and incomplete
realtime timing limitations remain in force.

# Reproduce the frozen-input replay

All inputs are read-only from `/media/camp/PEI_T7/icra 2027_contact/uncalibrated`.
The selected saved scans are all 49 files in `001` (5), `003` (12), `007` (12),
`jiaqi` (12), and `pei` (8). `001/RH_Per_S_DtP` uses attempt `002`; every other
saved scan uses attempt `001`. No abandoned attempts are substituted.

Run from the repository root. The commands below may run concurrently because
their scan groups are disjoint. Each process uses one BLAS thread and one CPU;
the examples avoid the declared control CPU 2 and native CPU 4.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python analysis_artifacts/confidence_cop_20260913/extract.py --groups 001 003 007 --cpu 23
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python analysis_artifacts/confidence_cop_20260913/extract.py --groups jiaqi pei --cpu 21
```

Extraction caches every exact control-used `feature.frame_seq == ultrasound/frame_index`
with a matching publisher identity. The stored JPEG is decoded directly; crop
and flip have already happened at the publisher. It uses the precise recorded
feature configuration, including the original near-depth ROI. Recomputed LCR
qualities are compared against the recording; the centroid is never invented
from the three qualities. Per-frame JPEG hashes and cache/source provenance are
saved. Completed frames resume without recomputing the random-walk map.

The replay uses the archived `source_snapshot/peirastic/contact_qp` modules,
whose hashes are recorded in `source_snapshot/source_sha256.json` and each
result. This makes the actually loaded implementation explicit even while the
working repository changes. To test a new implementation, make a new snapshot
and output directory instead of silently relabeling an existing result.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/rm75/bin/python analysis_artifacts/confidence_cop_20260913/replay.py --groups 001 003 007 --cpu 22 --follow
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/rm75/bin/python analysis_artifacts/confidence_cop_20260913/replay.py --groups jiaqi pei --cpu 20 --follow
```

After extraction finished, the final run used four disjoint replay shards for
more even CPU use: `--groups 001 003 --cpu 22`, `--groups 007 --cpu 23`,
`--groups jiaqi --cpu 20`, and `--groups pei --cpu 21`. Stop any overlapping
earlier replay processes before using this split. The compact control cache
was upgraded with `extract.py --prepare-only --cpu 19` to preserve publication
review clocks without recomputing feature frames.

`--follow` waits for each completed extraction manifest. It never reads a
partially completed scan cache. A result with different loaded source hashes
is recomputed. Both new modes run every recorded control tick, including
approach, with a distinct tank and command state for the whole attempt. The
recorded old candidate and successful final command models remain separately
available. Runtime and interpretation limits are detailed in `REPORT.md`.

```bash
PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/rm75/bin/python analysis_artifacts/confidence_cop_20260913/report.py --require-49
PYTHONNOUSERSITE=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python analysis_artifacts/confidence_cop_20260913/plot.py
```

The report command verifies 49 distinct completed paths, a single loaded
source revision, matching moving-command array lengths, actual budget
admission counts, tank bounds, and nonnegative admitted energy margins.
It also requires every result to match the current replay-script SHA-256 and
the exact-review-clock schema, preventing stale approximated-clock results
from entering the complete report.
`comparison.csv` has one row per scan and mode. `direction_conflicts.csv`
compares raw image requests to exactly one weak lateral window before dynamic
lag, with frame/tick counts, durations, and request magnitudes.

This is open-loop command analysis, not a physical simulation or a prediction
of how new commands would change the recorded images and force trajectories.

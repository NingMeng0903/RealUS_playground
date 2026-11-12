# Suffix-noise first-root diagnostic

Command

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python work/noise_headroom.py --n 24 --replicates 16 --workers 8
```

The first 24 archived cases were used as development cases. Their generated
anatomy, hidden stress parameters, physical-time dropout interval, persistent
ghost, prefix, and common probe were retained. Only the two suffix frames were
resampled. Each replicate used a new acquisition-noise bank, and the same bank
was shared across all three first views at the first stage and across all final
views at the terminal stage. Each forced first view updated the public belief
and then called the original `choose_real_tail`.

The disjoint replicate halves were used for per-case root selection and held-out
evaluation.

| forced first view | strict rate | correct-fresh rate | false-confirm rate | target-visible rate |
| --- | ---: | ---: | ---: | ---: |
| 2 | 61.46% | 61.46% | 10.94% | 74.48% |
| 3, archived center | 61.72% | 61.72% | 10.16% | 74.48% |
| 4 | 60.94% | 60.94% | 7.03% | 74.48% |

The center root had a 0.26 percentage-point pooled strict-rate advantage over
view 2 and 0.78 points over view 4. Per-case cross-fitted selection chose a side
root in only 1 of 24 cases. That case's train advantage disappeared on the
held-out half. The selected-root held-out strict rate was 63.02%, exactly equal
to the center-root held-out rate. No case had a positive held-out advantage.

This pilot finds no reproducible noise-robust first-root headroom in the
current wide-reserve observation model. The small one-case training preference
was noise-dependent. It should stop further scorer tuning on this intervention
unless a new observation model or a physically measured view-dependent signal
is introduced.

The full records are in `NOISE_HEADROOM_24x16.json`; compact values are in
`SUMMARY_24x16.json`. Motion legality is inherited from the frozen wide-reserve
preflight graph. Robot execution is not rerun here.

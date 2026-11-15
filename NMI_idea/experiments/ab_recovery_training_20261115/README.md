# Recovery training after a streaming leg scan

We collected paired recovery branches at the first failure or ambiguity checkpoint of a streaming ProDMP–MPC scan on the registered Among_US leg sections. The candidate menu includes advance, wait, left and right shifts, a shorter longitudinal scan, and revisit. The commands share a one-second decision block. All real observations update the model at 10 Hz, including during the probe and subsequent MPC continuation. See the [protocol](PROTOCOL.md) and [reproduction files](results/summary.json).

## Training and comparison

The phase-A teacher ran 192 independent simulated episodes and executed all feasible recovery options from its eligible checkpoints. It generated 1,435 segments, of which 944 with a correct local target return and no incorrect confident association were used for the ProDMP fit. Accepted segments comprise 602 advance, 60 wait, 58 left, 63 right, 74 short-scan, and 87 revisit examples. Left and right use the shared side-scan shape fit. Waiting has zero lateral shape. The legacy and retrained phase-A primitives run with the same advance-first feedback MPC.

The frozen retrained primitives generated 1,410 labelled action continuations from 235 eligible checkpoints in 320 separate training scenes. The small supervised B reads the public state, belief and candidate forecasts. It predicts correct completion, wrong acceptance, and local recovery after an opening action and the same live MPC tail. All outcomes, including failed actions, contribute training labels. A separate 96-scene development split chose waiting as the best fixed first option under its scoring rule. Eligible cases use a shared prefix, action menu and tail in all three comparison arms.

## Held-out outcomes

There are 192 independent shape-shifted scenes. Each scene is replayed at 15 degrees and under two opposite measured angle schedules using the same anatomy, observation noise and visibility tapes and the same entry state. Each angle schedule generates its own pre-probe history. The controller updates a shared 54-state structural, offset and sensitivity belief. All ProDMP and B fit examples were collected at 15 degrees.

| Held-out schedule | Old A | Retrained A | Live MPC recovery | Fixed first wait + MPC | Learned first action + MPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 60/192 | 57/192 | 143/192 | 104/192 | 136/192 |
| Flexion | 60/192 | 56/192 | 140/192 | 102/192 | 141/192 |
| Extension | 60/192 | 57/192 | 141/192 | 101/192 | 139/192 |

- **A tracking improved.** At fixed angle, mean absolute lateral error fell from 3.45 to 2.45 mm and the fraction of acquired frames with a correct centered return rose from 45.6% to 49.8%. The retrained path traveled 1.72 mm farther. Strict all-interval completion and wrong-acceptance counts provide no improvement for A.
- **Learned B beat the frozen fixed opening on all three schedules.** Its paired improvements were 16.67, 20.31 and 19.79 percentage points. B used 0.29–0.61 mm more mean reference-pose TCP travel and 0.13–0.20 s less mean simulated time than the fixed opening.
- **Learned B versus live MPC was inconclusive for completion.** Its changes were −3.65 pp [−8.33, 1.04], +0.52 pp [−3.65, 4.69] and −1.04 pp [−5.73, 3.65]. B traveled 1.50–2.04 mm less on average, but did not satisfy strict travel/time equivalence. Wrong acceptance went from 11 to 12 cases at fixed angle, 11 to 12 during flexion, and 10 to 11 during extension. These are paired case-bootstrap intervals. The repeated angle conditions are not additional independent subjects.

## Evidence and uncertainty

At fixed angle, 136/192 scenes reached an eligible checkpoint. B and live MPC chose different opening actions in 94/136. Acquiring a fresh correctly identified return in the local query interval at some point in the remaining episode changed from 117/136 to 120/136. Among 37 scenes with ambiguous downstream identity at the checkpoint, B reached the correct identity with posterior confidence of at least 0.95 after the opening action in 18/37, while live MPC did so in 22/37. All three planners returned zero confident certifications in 40 exact label-swap controls. The downstream identity posterior remains one half in that control.

At the same queried station and pre-action angle, fixed-angle B reduced target-position RMSE from 5.45 to 4.22 mm, versus 4.92 mm for the paired live MPC. The mean reduction in position covariance trace was 0.20 mm² for B and 0.66 mm² for live MPC. The nominal 90% marginal intervals had mean empirical axis coverage of only 11.8% for B and 9.2% for live MPC. Both started at 18.4% coverage and the median post-action interval width was zero on each axis. In this cohort the truth was within the full coordinate support for 96.3% of lateral queries and 72.1% of depth queries. The poorly resolved offset representation and posterior collapse limit the interpretation of variance. The uncertainty numbers are a diagnostic and cannot support a calibrated uncertainty-reduction claim.

The 66 independently resampled development checkpoints illustrate the action-selection limit. Selecting the best branch using a first tape yields +7.58 pp over live MPC on that tape. Repeating the selected action on a fresh suffix yields −1.52 pp [−9.09, 6.06] and takes 0.42 s longer on average. This diagnostic uses evaluator information in the first selection and does not enter the fitted policy, the fixed comparator or the held-out result.

The recovery-shape ablation preserves the originally selected action and planned endpoints. Of 63 selected side, short-scan, or revisit openings, 60 allowed an endpoint-only ProDMP replacement under the same motion constraints. Replacing the fitted lateral shape yielded one additional completion among these 60 paired cases and unchanged wrong acceptance and fresh local evidence. Three endpoint-only replacements failed the sampled motion gate. [Same-endpoint mechanism ablation](results/same_endpoint_ablation.json)

## Scope and reproduction

The observations are synthetic anonymous centroids on registered anatomy. The stress model introduces material-fixed clutter, view-dependent shadow and two competing vessel connections. The alternative connection rotates the pair around its midpoint and preserves the unlabeled set past the transition. The public likelihood marginalizes injective assignments including misses and field-of-view clipping. The evaluation generator includes temporal visibility and held-out geometry effects beyond this public likelihood.

The main experiment charges sampled Cartesian motion in the reference chart. The signed angle schedules change the hidden relative vessel displacement at each exposure. They evaluate online belief and policy transfer under that input. In the secondary RM75 replay, all 30 full fixed-angle paths from the first six scenes passed joint, FK, sampled rate and configured self-collision checks over 7,280 samples. The main observations and outcomes use the chart plant. The MPC probe preview uses the full mixture belief and B receives compact summary and forecast features. Representational differences remain a plausible explanation for their different opening choices.

From this directory, with NumPy, SciPy, scikit-learn and threadpoolctl installed, run:

```bash
python -B run.py --out /tmp/recovery-training-reproduction \
  --motion-count 192 --branch-count 320 --dev-count 96 --test-count 192 --alias-count 40 --jobs 12
```

`--jobs 1` runs serially. The case namespaces, source hashes, fit counts and paired summaries are recorded in [summary.json](results/summary.json). Gzipped JSON Lines store raw packets, trainee checkpoints, outcome labels and 20 ms chart trajectories. Archive packet and motion offsets are absolute within the complete episode. Suffix motion repeats the shared root as its first sample. Outcomes retain explicit evaluator-only role labels. The fitted ProDMP shapes and B are in [prodmp_weights.npz](results/prodmp_weights.npz) and [value_model.pkl](results/value_model.pkl). The RM75 audit uses the existing kinematic runtime and `lift.py --records results/test_fixed.jsonl.gz --out /tmp/rm75_lift.json --cases 6 --include-a`.

[ASTRA MAX theory and outcome review](results/ASTRA_REVIEW.md) · [SOL independent raw-record audit](results/SOL_AUDIT.md)

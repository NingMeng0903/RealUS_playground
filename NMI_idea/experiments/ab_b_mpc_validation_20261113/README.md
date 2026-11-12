# Independent review of B versus replanning MPC

This package records the attempted redesign of B for robotic ultrasound. The reference commit is `cdb1705a02f60382ffbe9e42f2ed420e27a28ddb`. The [original review request](../../B_REVIEW_REQUEST.md) and the [frozen moving-Y report](../ab_tro_y_continuation_20261112/REPORT.md) describe the original system and endpoint. New results are from development simulations. Neither redesign passed the promotion criteria, and the prospective confirmation split was kept unused.

[Download the reproduction bundle](b-validation-reproduction.zip) · [Reproduction instructions](reproduction/README.md) · [Full paired statistics](reproduction/results.json) · [Development protocol](reproduction/benchmark/development_protocol.md)

## Experiments and outcomes

A successful run needs a fresh, correctly associated image at each required target and completion of the corresponding checked motion within budget. In every planner comparison the strong MPC assimilates real masks and replans after each real acquisition. Comparison arms share their motion graph, estimator and verifier within a cell. Outcome labels are used by the evaluator. Sensor packets supplied to policies contain centroids and public pose data.

| Development condition | Strong MPC | Redesigned B |
| --- | ---: | ---: |
| Two-station daughter scheduling, nominal estimator, each stroke reserve | 14/24 | 14/24 |
| Two-station trunk scheduling, nominal estimator, each stroke reserve | 24/24 | 22/24 |
| Reference probe for daughter, shared fitted clutter likelihood | 12/32 | 13/32 |
| Reference probe for trunk, shared fitted clutter likelihood | 28/32 | 28/32 |

- **Historical replay.** The portable audit verified hashes, logged mechanics and all 768 terminal confidence calculations. In the original 128-case paired moving-Y comparison B and MPC both completed 76 cases with 16 wrong confirmations and identical paths. The headroom records separate the hindsight best legal action pair from first-action selection with the shared real tail planner. The suffix-noise repeat holds the latent case fixed and uses paired exogenous innovations across views. [Replay](reproduction/historical/legacy_headroom/FROZEN_REPLAY.json) · [Hindsight summary](reproduction/historical/legacy_headroom/SUMMARY_128.json) · [Noise summary](reproduction/historical/noise_headroom/SUMMARY_24x16.json)
- **Scheduling.** The 768 arm-condition rows comprise 24 independent cases. The five-slot planner branches on verifier status in its lookahead and assimilates actual masks online. The finite search uses per-slot marginal acceptance probabilities. Unpruned B had zero paired wins and nine losses versus MPC across all repeated conditions. The pruning variant returned the same decisions and reduced nodes by about 57% relative to unpruned B. The node count remained higher than both MPC controls. [Cases](reproduction/development_a/cases.jsonl) · [Paired statistics](reproduction/development_a/paired.json) · [Search](reproduction/benchmark/search.py)
- **Reference probe.** The 960 arm-condition rows comprise 32 independent cases. The two post-prefix slots allow a reference or near-target probe and an actual target acquisition. The B guard compares candidate gains and wrong-claim estimates across its supplied beliefs. The fitted-clutter daughter arm had one paired win and zero losses. It agreed with generic raw-mask feedback on all fitted routes. Its only additional success switched the view at the near-target station. The simultaneous paired turn interval for the positive cell extended outside ±2 degrees. Fitting the marginal clutter likelihood improved a held-apart point prediction score. The reference-deletion ensemble uses two evidence histories under a common likelihood. [Cases](reproduction/development_b/cases.jsonl) · [Paired statistics](reproduction/development_b/paired.json) · [Clutter fit](reproduction/benchmark/clutter_fit.json) · [Root audit](reproduction/development_b/reference_root_audit.json)
- **Repeatability on the new graph.** The noise diagnostic uses 16 additional development cases with eight paired acoustic replicates per first action. On the held half, public B and MPC had the same 32/64 correct daughter runs and 52/64 correct trunk runs. A per-case selector given labels from the other replicates gained two correct daughter runs in one case. All six trunk roots gave the same held-half outcomes. This diagnostic informs action opportunity. The independent unit is the case. [Noise repeat records](reproduction/development_b/reference_noise.json)

## Points for a reviewer to verify

1. Recompute outcomes, pairing and costs from the logs. Check that the branch, reserve and estimator conditions are paired by case and that any equivalence claim uses measured path, rotation and elapsed time.
2. Trace the separation between public evidence and evaluator labels. Inspect both actual observation assimilation and simulated first-observation branches.
3. Check the attribution to the B decision. Compare B with the shared-model MPC, generic feedback, fixed probes and evidence-model controls. Inspect the root audit to locate the single fitted daughter gain.
4. Assess the proposed next mechanism and its evidence requirements. The suggested extension searches paired target and persistent-alias hypotheses under the same feasible completion paths. Pruning uses an upper value after revealing the pair while retaining ambiguity within it. A decisive experiment must establish repeatable action-dependent evidence and remaining physical support before testing a success-rate claim.

The execution model uses sampled fixed-rail kinematics with ideal low-level tracking and synthetic centroids. Physical contact, moving B-mode phantom observations and deployment latency remain unmeasured. The two tasks reuse the same atlas. The historical and repeat-noise selectors using labels are evaluator diagnostics.

## Recompute the tables from this checkout

Run from the study directory with Python, NumPy and SciPy. This works directly on the committed logs and does not require the robot model.

```bash
review_tmp="$(mktemp -d)"
cd reproduction
python benchmark/summarize.py development_a/cases.jsonl --out "$review_tmp/paired_a.json"
python benchmark/summarize.py development_b/cases.jsonl --out "$review_tmp/paired_b.json"
diff -u development_a/paired.json "$review_tmp/paired_a.json"
diff -u development_b/paired.json "$review_tmp/paired_b.json"
python benchmark/reference_root_audit.py development_b/cases.jsonl \
  --settings development_b/settings.json --out "$review_tmp/reference_root_audit.json"
```

The [file hash manifest](reproduction/file_hashes.json) records the byte-identical expanded contents of the downloadable archive. Follow the [bundle instructions](reproduction/README.md) to rerun the sampled robot simulation. The [diagnostics directory](diagnostics/) preserves the smoke and preflight records outside the main tables.

# Active ultrasound reinspection after a shared MPC prescan

The held-out benchmark found a large effect of preserving all centroid candidates. It did not establish a cost- and risk-qualified benefit for a separate B policy over two-step MPC that already uses all candidates and replans on each actual observation. The test used a synthetic centroid sensor and sampled robot motion.

[Frozen protocol](PROTOCOL.md) · [Independent held-out review](independent/HELDOUT_REVIEW.md) · [Theory verdict and next experiment](theory/FINAL_VERDICT.md) · [Historical audit](independent/HISTORICAL_REVIEW.md) · [Related work](RELATED_WORK.md)

## Task and comparison

Each of the 512 independent scenes contains a sampled moving internal structure, a potentially persistent false return, and observation losses. The controller starts from the atlas and a population prior. It makes five common prescan acquisitions at fixed successive stations with two-step MPC for view selection and actual Bayesian updates. The tested policies share that executed history, the posterior at its exit, and the same 30 poses and 264 checked edges, including equivalent probe configurations from the inherited dIRD-ranked bank. They receive matching stochastic packets at the same station and acquisition clock on a given scene. The nearest control deliberately compresses the same underlying suffix packets.

Success on a Y case requires a correct fork interval and side, a fork estimate within 5 mm, and two consecutive new acquisitions at prescribed downstream sites, with both chosen returns from the true daughter. Single, parallel and disconnected-crossing scenes test rejection of a nonexistent offshoot. They do not test a new positive nonbranching inspection goal. Shape stress jointly changes the daughter and artifact geometries and increases nuisance variability. The benchmark does not generate B-mode, execute a U-Net, measure continuous lumen coverage, or simulate tissue contact. Its clock charges 1.60 s per capture including an allotted 0.10 s for decisions. Measured batched wall times are reported separately and do not advance the target motion.

## Held-out results

Each cohort includes 128 Y and 128 no-target cases. The table reports Y completions and erroneous affirmative declarations over all 256 cases. The seed-7 learned runs are shown. Other seeds reuse the held-out anatomies and disturbance tapes.

| Policy | Nominal Y | Held-shape Y | Nominal affirmative errors | Held-shape affirmative errors |
|---|---:|---:|---:|---:|
| Live two-step MPC, all returns | 51 | 38 | 19 | 19 |
| Longer belief rollout, public greedy tail | 56 | 34 | 18 | 26 |
| Checked continuation-route B | 58 | 31 | 17 | 19 |
| One-step with shared tail | 58 | 38 | 21 | 25 |
| Fisher-assisted one-step | 56 | 39 | 21 | 25 |
| Live two-step MPC, nearest suffix | 1 | 2 | 0 | 1 |
| Recurrent PPO initialized by MPC imitation | 0 | 0 | 0 | 0 |
| Feed-forward PPO initialized by MPC imitation | 0 | 0 | 0 | 0 |
| Discrete CQL | 2 | 0 | 0 | 1 |

On nominal Y cases, continuation-route B gained 5.47 percentage points relative to live MPC (paired 95% interval −0.78 to +12.50) while turning 4.45° more on average. The upper one-sided probability bound for a new erroneous affirmative declaration was 6.70%. Its held-shape difference was −5.47 points (−11.72 to 0). Neither it nor the longer rollout passed the joint no-worse or strict equivalence tests on realized TCP travel, rotation, rail travel and simulated elapsed time. The audit reports all errors, including false absence, separately from affirmative errors. The longer rollout uses a greedy tail and finite root/scenario budgets. It is an approximate public planner, with no guarantee of finding an optimal alternative.

On the nearest controller's actual paths, the two required daughter returns were available in full packets on 11 nominal and 13 held-shape Y cases. The compressed packets retained both returns on one and four cases, respectively. The closed-loop all-return/nearest comparison also changes executed routes, cost and error counts. It demonstrates consequential information loss under this specific nearest compression. The legacy fixed-Y study has an additional comparison against the original finer-grid sensor. [Earlier association benchmark](../ab_relational_b_20261113/README.md)

The shared prescan alone classifies Y presence correctly on 224/256 nominal and 205/256 shifted cases by posterior MAP. It establishes a useful prior for subsequent decisions. For full MPC, the nominal chain is 98/128 cases with the correct ordered stations, 67/128 with both actual true returns there, and 51/128 passing the joint endpoint. The shifted chain is 84, 55, and 38. A truth- and future-tape-assisted reachability check finds paths with the two required returns on 127/128 nominal and 126/128 shifted positive cases. This is a fixed-tape opportunity check; a public policy still has to choose and recognize those observations from the information it has when it moves.

The three initialized recurrent PPO seeds and three feed-forward seeds completed zero Y cases in both cohorts after 240,000 interactive decisions per run. The learner encodes a public particle-belief feature vector with an MLP and, for the recurrent arm, a GRU. It does not use a graph neural network or raw images. Most learned policies failed to execute the required pair. The recurrent clones managed 5/1/3 nominal and 3/1/1 shifted cases. CQL achieved 2/0/1 nominal and zero shifted. The XOR history check exercises the recurrent implementation separately. This is evidence about the tested policies and reward under this sensor and training budget. [Training audit](independent/TRAINING_AUDIT.json) · [Recurrent control](independent/memory_sanity/SUMMARY.json)

## Decision mechanism

The estimator weights each graph/geometry/artifact hypothesis by the likelihood of the complete unordered packet, including missed and duplicate-bin returns. The shared stop decision integrates the probability of choosing the correct identities in **both** new frames and of the fork falling inside the reported tolerance window. For hypothesis weights `b[h]`, label `l`, candidate slots `i,j` and position `c`, the joint probability is

```text
sum_h b[h] * I(label[h] == l) * P(i is daughter | h, first packet)
           * P(j is daughter | h, second packet) * I(abs(fork[h] - c) <= 5 mm).
```

For each pair of candidate slots, the selected `c` centers a maximum-weight 10 mm interval. This Bayes decision matches the tolerance endpoint. The approximation conditions on the simulator's persistent geometry and disturbance parameters before combining frame evidence. Under the model a repeated plausible false return does not become an independently drawn false structure. [Implementation review](theory/IMPLEMENTATION_REVIEW.md)

For a future image-based B, generate checked overlapping observation paths where credible anatomical explanations predict different registered image evidence. Score each path by the expected joint probability of a correct requested continuation after debiting its full cost. Update from the acquired evidence and compare stopping, another view, and recovery with the same acceptance rule. A bridge-aware live MPC and a fixed diagnostic sweep are the two decisive controls. They receive the same segmentations, poses, bank and budget. The new claim would require a prospective advantage of contingent selection on correct completion and error at matched realized cost. [Next experiment and acceptance criteria](theory/FINAL_VERDICT.md)

## Replay

The evaluated Python runtime was 3.10.20 with NumPy 2.2.6, SciPy 1.15.3, Numba 0.65.0 and PyTorch 2.12.0. Archived settings and freeze hashes identify the six cohort/seed configurations. `TRIAL_RECORDS.tar.xz` holds the original JSONL files for all paired, development and smoke runs. `TRAINING_RUNS.tar.xz` holds final and intermediate weights and detailed training curves, including the learning sanity checks. `DATA_MANIFEST.json` records released and extracted hashes. Unpack to a scratch copy of this directory to preserve the archive. The motion banks and atlas resolve against the review repository supplied with `--repo`.

```bash
repo_root="$(git rev-parse --show-toplevel)"
stage_dir="$repo_root/NMI_idea/experiments/ab_topology_reinspection_b_20261113"
audit_dir="$(mktemp -d)"
cp -a "$stage_dir" "$audit_dir/study"
study_dir="$audit_dir/study"
tar -xJf "$study_dir/TRIAL_RECORDS.tar.xz" -C "$study_dir"
tar -xJf "$study_dir/TRAINING_RUNS.tar.xz" -C "$study_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python "$study_dir/independent/audit_freeze.py" \
  --stage "$study_dir" --archive "$repo_root/NMI_idea/experiments" \
  --out "$audit_dir/freeze.json"

python "$study_dir/theory/final_audit.py" --stage "$study_dir" --repo "$repo_root" \
  --out "$audit_dir/theory.json"

python "$study_dir/independent/audit_trials.py" \
  --stage "$study_dir" --repo "$repo_root" --results "$study_dir/confirm_seed7" \
  --out "$audit_dir/nominal.json" --truth-seed 723113 --count 256 \
  --tape-namespace 723213 --split confirmation --baseline mpc --nearest-arm nearest \
  --prior-seed 723101 --prior-size 256

python "$study_dir/independent/audit_policy_replay.py" \
  --stage "$study_dir" --repo "$repo_root" --out "$audit_dir/policies.json" \
  --cohorts confirm_seed7 shape_seed7 confirm_seed29 shape_seed29 confirm_seed61 shape_seed61

python "$study_dir/run_topology.py" --repo "$repo_root" --out "$audit_dir/confirm" \
  --split confirm --n 256 --seed 7 --train-n 1536 --dev-n 96 --interactions 240000 \
  --checkpoints "$study_dir/fitted/seed7" --freeze "$study_dir/FREEZE_confirm_seed7.json" \
  --arms mpc rollout witness fisher nearest greedy prescan_only ppo mlp cql bc bc_mlp
```

For the shifted cohort use `--split shape`, case seed `723115`, tape namespace `723215` and `shape_holdout` in `audit_trials.py`. Seeds 29 and 61 replay only the five learned arms. The frozen benchmark replay uses the released weights. For exact fitting-source provenance, the training used `fitted/core_at_fit.py` in place of `core.py` and preceded the addition of `witness.py`. Restore that snapshot in a second scratch copy and remove `witness.py` there before running `fit_topology.py` with the recorded settings. The independent report reconstructs 11,264 arm-case records and 41,638 suffix packets. A separate dispatch replay reproduces 6,491 commands and declarations on every eighth case for each policy/configuration. [Dispatch check](independent/POLICY_REPLAY.json)

Earlier development outputs include alternate coverage and terminal rules. Their records retain original settings and are exploratory. The historical audit uses original raw files on the project archive in addition to the review Git tree. Its derived results are included. The current benchmark replay uses the review Git tree alone.

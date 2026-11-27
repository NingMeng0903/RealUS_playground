# Learned ProDMP with static artery–vein reinspection

This study reexecutes the [static two-vessel screen](../ab_static_two_vessel_b_20261113/README.md) with the earlier learned ProDMP. It preserves the same registered artery and vein, sensor, task, candidate endpoint poses, prescan policy and resource caps. It newly generates and checks the selected paths in the shared motion layer. The exact comparison finds a 6.27-point completion improvement over the original live MPC forecast family and a 0.00253-point improvement over a fixed first near-center view with the same updating and replanning. The fixed-first control already captures the useful near-retry effect.

[Frozen protocol](PROTOCOL.md) · [ASTRA MAX derivation](THEORY.md) · [Exact and replay results](verification/SUMMARY.json) · [Theory audit](verification/THEORY_AUDIT.md) · [Independent replay audit](verification/INDEPENDENT_AUDIT.json)

## Execution

Every run fits the original `LearnedProDMP.fit_default(seed=20261112)`. For a nonzero material displacement, both coordinates of the learned trajectory are mapped through the original 15° atlas to a sampled 3-D probe reference. Same-origin changes of view use the learned phase for the orientation trajectory. Every selected transition starts from the preceding realized joint state. Sequential IK from both endpoint witnesses supplies continuous seeds across the 7R arm's redundancy. A fixed-rail IK projection tracks the same reference at each sample. The original dIRD-selected joint endpoints are retained. The generator uses 1.29 s of transit and 0.21 s of dwell, and checks sampled FK, joint limits, joint rates and configured robot/probe self-collision. The initial forward-only solver produced abrupt joint closure at several goals. Its replacement with two-sided IK was fixed before evaluating policy outcomes.

All 63 motion candidates passed, including nine holds and 54 newly generated moves. The 4,851 samples had a minimum configured self-collision distance of 39.30 mm, maximum FK residual of 0.00396 mm and 0.00229°, and a maximum sampled joint-rate fraction of 0.5792. The nine nominal acoustic poses agree with realized FK endpoints within 0.00122 mm and 0.00057°. No archived intermediate samples, motion costs, or fallback edges are used to run these policies. The past bank supplies only the endpoint witnesses. [Recomputed candidate trajectories and checks](verification/MOTION.json.xz) · [Executed traces](verification/EXECUTED_MOTION.npz)

For exact finite planning, each invocation computes a new shared cost and feasibility matrix from its generated trajectories. The replay regenerates the selected trajectory from its actual start. After a small fully uncached subset, it may reuse a generated trajectory only for an exactly identical start and action in that same run. Start and goal configurations are verified for all replayed moves. Actual integrated TCP, orientation, rail and time costs are debited before a new acquisition. This is sampled ideal kinematics. It uses fixed endpoints and fitted mean weights. It does not optimize a continuous ProDMP parameter vector, measure physical tracking or contact, use images acquired in transit, or charge CPU latency as physical motion time.

## High-level ablation on the same learned ProDMP

| Controller | Correct accepted pair, exact | Mean TCP | Mean rotation | Mean modeled time |
|---|---:|---:|---:|---:|
| Live MPC, fixed-second forecast with actual replanning | 75.5760% | 42.97 mm | 50.91° | 6.790 s |
| B, contingent second-view forecast | 81.8484% | 43.90 mm | 44.55° | 6.202 s |
| Fixed first near-center view, same live MPC | 81.8459% | 43.90 mm | 46.05° | 6.202 s |
| Fixed first near +12° view, same live MPC | 81.8445% | 43.90 mm | 43.00° | 6.202 s |
| Two new images, no extra acquisition | 70.5594% | 42.23 mm | 43.24° | 5.763 s |

Across 512 paired replays, MPC completes 375 cases and B and fixed-center each complete 413. B has 59 wins and 21 losses against MPC, and identical completion and error outcomes to fixed-center on every tape. All tested arms produce zero wrong declarations in these replays. The first 32 cases regenerate and check every acquisition without a cache, totaling 495 syntheses across the shared prefixes and policy suffixes. The remaining 480 cases synthesize 34 distinct selected transitions and reuse an identical state/action result 7,257 times. Both ledgers agree exactly with the computed motion debits. [Independent audit](verification/INDEPENDENT_AUDIT.json)

B and fixed-center pass the predeclared four modeled mean-cost equivalence bands. Compared with fixed-center, B turns 1.50° less on average. Fixed +12° turns 1.55° less than B at a completion gap of 0.00391 points. This does not establish a robust efficiency gain. Against live MPC, B adds 0.93 mm mean TCP travel and saves 6.36° and 0.59 s. Its extra TCP travel is within the declared 2 mm upper margin. The changes in angle and time exceed the two-sided equivalence bands in the direction of savings. Exact wrong-confirmation probabilities for MPC, B and fixed-center are 0.000579%, 0.000506% and 0.001173% in the synthetic model. These modeled probabilities do not establish real error rates.

Define `F(a)` as the root score with a fixed second diagnostic and `G(a)` as the value with a shared, real-observation replanner after the first action. The measured modeled B increment is `max G − G(argmax F)`. MPC's raw fixed forecast is 70.5596%, but its realized expected value is 75.5760%, which is the actual baseline. The fixed-center arm has the same near retry, conditional far view and joint final declaration as B. It almost saturates the detection ceiling `p[1-(1-p)^2]=81.8496%` for at most two near attempts and one far attempt. The two-image arm yields 70.5594%. Its comparison with the other arms exposes the value of the optional near retry. In this class it is valuable to reserve the extra image for recovery. The decision problem provides negligible additional completion headroom over a fixed first view. [Estimand and finite task ceiling](verification/THEORY_AUDIT.md)

## Learned trajectory shape

A separate same-endpoint comparison replaces the fitted weights with the inherited endpoint-only ProDMP weights, retaining the solver, duration, dwell and checks. All 63 candidate motions pass in both variants. For the 36 moves between stations, the learned trajectories have maximum same-time FK displacement of 4.90–6.13 mm relative to endpoint-only, midpoint cross-track difference of 0.89–1.14 mm and 0.23–0.30 mm more TCP travel. This establishes that the fitted primitive changes the motion, with the same available end poses. Recomputing the exact decisions with the endpoint-only costs gives identical correct-completion and wrong-confirmation probabilities, capture counts, rail travel and time in all six policies. Learned shape adds 0.81 mm mean total TCP travel for B. The observed B-versus-MPC completion contrast is identical for the two ProDMP modes. The secondary comparison establishes shape use and feasible execution, with no incremental completion effect in this screen. [Matched trajectory shape results](verification/PRODMP_SHAPE_CHECK.json) · [Endpoint-only decision results](verification/endpoint_only/SUMMARY.json)

## Reproduce

Run this directory with the same offline RobotModel/CapabilityEvaluator runtime. The `--repo` checkout supplies the frozen experiments; the mounted `--robot-repo` supplies the physical probe configuration, URDF, collision assets and calibrated dIRD checkpoint used by the inherited bridge. The recorded runtime is Python 3.10 with NumPy, SciPy, PyTorch and the installed robot kinematics dependencies. The source hashes, weights, URDF and robot manifest identify the inputs. No robot connection is required.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
repo_root="$(git rev-parse --show-toplevel)"
scratch_dir="$(mktemp -d)"
robot_repo=/media/camp/EXT_DRIVE/RealUS_playground
python run.py --repo "$repo_root" --robot-repo "$robot_repo" \
  --out "$scratch_dir/run" --replay 512 --uncached 32
python audit.py --repo "$repo_root" --robot-repo "$robot_repo" --results "$scratch_dir/run"
python shape_check.py --repo "$repo_root" --robot-repo "$robot_repo" \
  --results "$scratch_dir/run" --out "$scratch_dir/run/PRODMP_SHAPE_CHECK.json"
python shape_ablation.py --repo "$repo_root" --learned "$scratch_dir/run" --out "$scratch_dir/endpoint_only"
```

The archived JSON and JSONL xz files contain the full generated candidates and paired trial records. The audit can read the compressed files directly. The exact and paired comparisons concern this one static anatomy, synthetic centroid observations and sampled execution. Results do not depend on new physiology or learned perception.

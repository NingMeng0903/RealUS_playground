# Learned recovery with CEM motion training

This study compares direct MPC with a learned first recovery decision, followed by the same complete MPC. Both run on the fixed-knee synthetic observation and registered chart benchmark. Each first decision begins after the same two initial MPC blocks. All legal options, the online association and evidence retraction, the sampled motion checks, and the segment feedback every 100 ms are shared. Both policies receive an open decision at the start and one at the first MPC continuation. After those decisions the normal active recovery logic applies.

## Discovery and training

CEM uses 48 new scenarios per task family. For each scenario it searches six requests per round for two rounds from each of three initial means, evaluated on four paired disturbances. It can access the simulated geometry during offline discovery. Eight other paired disturbances verify the nominated trajectories. A trajectory contributes to the fit after it beats the direct MPC in correct completion, completes at least two thirds of repeats, and incurs no additional new centered mistakes. The two-feature ProDMP fit takes nominal displacement and midpoint bow. The nominal fit conditions on the real initial position and velocity and preserves the specified terminal position and velocity.

Cluster the verified requests into six shared relative proposals using only the discovery scenarios. The same requests are available on every new case. The shared library also retains the prior MPC actions. At the first recovery decision, every arm attempts ProDMP rendering for the expanded options and uses the same admissible analytical waypoint when the fit fails. The original MPC advance and recovery skills are still available throughout. The continuation uses the existing MPC renderer, with analytical references for the expanded options, over the same request library. Every motion pays its realized chart travel, turning, time and acquisitions.

Train on 144 scenarios per family using ten common posterior rollouts and four common evaluator rollouts per feasible action. The posterior rollouts estimate the expected downstream value from public history. The evaluator rollouts use the simulated case and a separate disturbance bank. Train on the entire action table including failed and costly trajectories. Root attributes, association probabilities, public atlas positions, candidate path and a multimodal posterior enter the feature vector. Features are computed before executing the selected motion and use the same public model at test time. Case seeds and simulator labels appear only in generation, training outcomes and evaluation.

Learn six action effects relative to the incumbent MPC. These are correct completion, two mistake measures, travel, turning and simulated duration. Small gradient boosting models estimate each effect. A candidate may replace MPC if its predicted resource and error deltas are within the prior budgeted protocol's margins. Within that set it maximizes predicted completion with the prior small cost and final-error penalties. Keeping the incumbent has value and cost deltas of zero.

Development uses forty independent cases per family, with six evaluator repeats per candidate. For each learned arm select between posterior labels and a fixed 65/35 mixture of posterior and evaluator labels by the lowest realized mean cap violation, then the highest correct completion. Freeze the models, fitted primitive and candidate library by hash before holdout. The label ablation omits the six CEM-proposal types from training but has the full menu at evaluation. The ablation tests the contribution of those additional labeled actions under this training procedure.

## Held-out benchmark

Use 96 independent seeds per family with four paired disturbances per scenario. The families are local connection ambiguity and nonbranching nearby tracks. Each method receives the same public initial state and candidate library. The four arms are direct strong MPC, the complete learned B, the learned B with the label ablation, and an online twelve-world public rollout using the same fitted generator and predicted caps. B and the online rollout replace at most the first recovery action. Actual subsequent observations update the common belief and the complete MPC replans.

The primary endpoint is arrival at the prescribed scan end with six genuinely correct centered interval images in the active ledger. Report final wrong identity, new centered mistake, fresh confirmation at the queried interval, executed cost, and the actual renderer of each first choice. The success criterion is at least 5 percentage points improvement against direct MPC and paired 95% interval above zero. Compare paired intervals against the previous +2 mm, +2 degrees, +0.10 s, +1 acquisition and +2 percentage point error margins. Report both the symmetric cost interval and the upper-margin check. Also give the one-sided 97.5% upper limit on the fraction of paired scenarios with any added error across their repeats. Keep repeats paired at the scenario level in the bootstrap and report results by task family. The time report includes simulated duration and separately measured first-decision CPU time.

The discovery, training, development, and test namespaces are disjoint. The matched finite hypothesis model, fixed knee angle, synthetic centroid packets and chart motion checks define the scope of the evidence. Full robot contact, live plant evolution during computation, real ultrasound, and new anatomical models require their own measurements.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B run_adaptive_recovery.py discover --offset 30000 --cases 48 \
  --families connection parallel --replicas 8 --population 6 --iterations 2 \
  --search-samples 4 --jobs 20 --out results/learned_b_cem_v1/discovery
python -B run_learned_recovery.py --out results/learned_b_cem_v1 prepare \
  --discovery results/learned_b_cem_v1/discovery
python -B run_learned_recovery.py --out results/learned_b_cem_v1 collect \
  --split train --cases 144 --samples 10 --replicas 4 --jobs 24
python -B run_learned_recovery.py --out results/learned_b_cem_v1 collect \
  --split dev --cases 40 --samples 10 --replicas 6 --jobs 24
python -B run_learned_recovery.py --out results/learned_b_cem_v1 train
python -B run_learned_recovery.py --out results/learned_b_cem_v1 test \
  --cases 96 --samples 12 --replicas 4 --jobs 24
```

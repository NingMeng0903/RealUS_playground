# Stage-A recovery data: corrected protocol

## Status

The earlier `pilot.py` and `search_actions.py` tested four 1.5/2.5 mm
out-and-back sweeps. That pilot was stopped before the full run. It is a smoke
test only and **does not test meaningful exploration, recovery from a wrong
association, or an improved ProDMP**. Its data and fitted scorers are not
evidence for the proposed method. The previous `ab_recovery_training_20261115`
fit ProDMP segments after one locally clean target packet. It did not require
the episode to finish correctly before accepting a positive demonstration.

## Question and causal unit

Can an offline, privileged search discover feasible probe programs that improve
the probability of correct new-image coverage and completed continuation after
the *same public checkpoint*, when the program is executed with 10 Hz ultrasound
feedback and then handed to the *same full live MPC*? Cases, not suffix-noise
replicas or action branches, are the independent evaluation units.

The Stage-A objective is correct target identity in all required new images and
completed scan coverage. Reappearance of any vessel, one low-error frame,
posterior concentration, and local trajectory fit are intermediate diagnostics.
Wrong acceptance and executed TCP distance, angle, time, and acquisitions are
separate outcomes.

## Generate difficult checkpoints without handing over the answer

Run the current Stage-A MPC and, later, the current learned student. Store the
first *public* loss, ambiguity, or contradictory-evidence checkpoint, including
the full anonymous image packet, belief, chart pose, robot velocities, coverage,
and remaining resources. Keep episodes with no trigger in the denominator for
the closed-loop benchmark. Retain the hidden scene only in a separate evaluator
record.

Training scenes should independently cover:

1. Target loss with an action-dependent visible alternative window. Sample
   displacement on the scale of the field of view and the reachable 1-3 block
   region, not a fixed 1.5/2.5 mm pulse. Preserve the same hidden anatomy,
   shadow, clutter, and dropout process across action branches. Include a
   no-headroom control in which the target cannot become visible by moving.
2. A correct target and a neighboring distractor that can be confused at one
   view but can be separated by an accessible sequence of views. Include
   persistent false structures, merged/missing packets, and delayed recovery.
3. Different public geometric contexts in which longitudinal revisit,
   lateral window change, and continuing forward exchange rank. This is needed
   to test conditional action choice against a fixed diagnostic scan.

In the inherited two-vessel world, the structural hypotheses have the same
unordered cross-section again after the transition. A downstream lateral
sweep cannot recover connection identity there. Stratify checkpoints by
distance to the publicly predicted informative transition, because two
feasible backward blocks may reach it when the old single-block revisit
cannot. The actual hidden transition is evaluator-only.

All changes to the scene are made **before** the common prefix. Any injected
offset or error must be inside the public model's support, or the experiment
must be explicitly labeled a model-mismatch stress test. A forced four-frame
dropout that no action can overcome is a negative control, not a search example.
For wrong-association recovery, offline truth may oversample cases, but the
deployed trigger and correction must use public evidence only. The current
`Controller` permanently records the first accepted coverage and its strict
success metric permanently fails after any wrong acceptance. A separate
publicly revisable verification ledger is required before claiming that this
benchmark tests *corrected* mistaken identity; report both the old strict
success and final corrected success without silently changing either.

## Offline trajectory teacher

Use CEM over one to three feasible 1 s ProDMP blocks with continuous
longitudinal and lateral goals, including backtracking, longer lateral moves,
and optional early handoff. Each block starts at the actual end pose and
velocity of the previous block. Check the **executed** path against the same
speed, acceleration, chart, IK/contact proxy, travel, and time limits as MPC.
Record clipping/backoff and deduplicate physically identical paths. Do not add
a tilt variable until the acoustic model makes tilt causally affect the image.

For each candidate, replay every 10 Hz observation through the public Bayesian
filter and in-segment servoing. Stop or branch only on public packets and
posterior. Then run `Controller.finish(..., allow_recover=True)` with the same
remaining resources and real observation updates. Use the hidden scene only
to score final correct completion, wrong acceptance, and cost.

For search, hold the hidden anatomy fixed and use several independent suffix
disturbance replicas. Keep top diverse modes and negative examples, not a
single best tape. Choose candidates on search replicas; compare each selected
program with full live MPC on *fresh paired replicas*. Freeze the case list and
selection before held-out evaluation. The total eight-second budget remains
unchanged. If only one or two blocks fit, do not force a third block; a longer
budget may be studied only when it is granted to every comparator.

Two ceilings must be reported separately:

- **World oracle:** CEM can choose a different root action for each hidden
  world. This establishes whether the action family and simulator permit
  recovery. It is not deployable performance.
- **History-conditioned teacher:** all hidden worlds compatible with the same
  public history share the same root action. Later choices may branch on newly
  simulated public images. Rank programs by expected complete success over
  these worlds. This is the relevant deployable opportunity.

If the world oracle wins but the history-conditioned teacher does not, do not
clone the world oracle's unique direction into the actor: the label uses an
unobservable fact. A world-oracle trajectory may still supply a feasible
motion-shape proposal or a privileged critic target, with its provenance
marked.

## Training and fair evaluation

Fit ProDMP *search/recovery* shape from executed blocks in successful complete
episodes. Keep multiple modes and public context. Fit an outcome model or
actor on all candidate outcomes, including failed/wrong continuations; its
inputs are the public history, uncertain probe chart address, predicted image
footprint, anonymous packet locations, and action/resource context. Compare
without and with this position representation. Aggregate later student-induced
failure checkpoints for a second round, using training worlds only.

The full MPC and learned proposal receive the same public model, candidate
permissions, ProDMP, visual servoing, recovery and stop rules. Compare paired
complete cases and actual execution cost, not only a common cost ceiling.
Check at least target loss and identity ambiguity; include a no-branch or
non-fork task. Report calibration, wrong acceptance, and independent-seed
stability. If robust history-conditioned headroom is absent, stop before
ProDMP fitting and inspect the observation/action physics. If headroom exists
but the student fails, inspect public observability, representation and
imitation error rather than tuning an advantage threshold.

## References used to design data collection

- DART: disturbance injection near learner errors, CoRL 2017.
- DAgger: collect corrections on states visited by the learner, AISTATS 2011.
- iCEM: sample-efficient continuous trajectory search, CoRL 2020.
- Original project design: `NMI_idea/ideas/Anatomical_GPS_V9.4.tex`,
  section “训练：共享响应、技能提议与困难状态再采集”.

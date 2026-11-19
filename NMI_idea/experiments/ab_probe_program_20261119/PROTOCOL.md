# One-probe data and feedback-conditioned probe programs

The experiment uses the fixed-angle leg scan world and the released evidence
training cases at `b7a2121`. It asks whether targeted candidate supervision
improves the existing one-probe B and whether one additional observed
decision within the same scan limits improves completion. Both use the same
material surface, atlas and field-of-view features, ProDMP/analytic fallback,
segmentation-candidate likelihood, deferred candidate retention, and MPC.
No new tilt renderer, learning algorithm, or new motion template is required.

## Pre-execution hypotheses and contrasts

* A data improvement means better correct completion with the same number
  of labels and the same downstream controller. Primary data contrast is
  outcome-contrast nominations versus uniform nominations. Evidence
  nominations versus uniform, and all three versus frozen B, are separate
  comparisons. This matches labeled dose. The outcome-contrast method
  incurs extra offline screening, which is recorded explicitly.
* Conditional probe control is useful as a deployment choice when it
  improves correct completion over both the fully replanning MPC and
  the development-selected strongest fixed sequence. Its attribution
  additionally requires better outcomes than a same-first precommitment
  or same-first tape shuffle. Per-family and loss-context contrasts
  check generalization within the simulation. A positive 95% paired
  lower limit is required to claim reliable improvement for a contrast.
  Costs and mistaken confirmations are reported jointly with completion.
  The common physical limits apply, with no additional 2 mm or 0.1 s
  relative cutoffs.
* Repeat-one and child-trained isolate reusing the first B at a later state
  versus supervising that later state. The program and child-trained arms
  share the exact same second decision rule. Their difference is the
  continuation used to label the **first** decision. A matched eight-sample
  online rollout checks the selected hypothesis family with the same
  start requests and sample bank for one-step and conditional lookahead.
  Its second decision still uses the same fitted child model, so this
  diagnostic does not bound all potential two-decision policies.

## Data generation

128 released training roots are reconstructed after one or two common
scanning blocks. Each augmentation adds four feasible candidate records and
the same MPC reference per root, or 640 rows added to 8,122 historical rows.
Each label is the mean difference in the six existing outcomes from eight
paired posterior simulations. Both successful and failed requests are
retained. Final/interim identity, distance, turn and time stay in the data.
Candidate nominations are uniform sampling, the existing evidence
teacher, and four outcome-contrast nominations. Outcome contrast uses
four **separate** simulations per screened request to identify a good
completion and an inferior request of similar nominal path length when one
exists, filling the dose with evidence nominations and legal requests.
The final labels use the original seed bank, including the archived
replays wherever available. This checks learning from outcome contrast
within a fixed action space.

Train the child selector on 64 new starts with up to three first executions
at each start. Those executions are proposed by the fitted contrast
student, the evidence score, and random requests. Each branch obtains
synthetic observations from an actual feedback execution in a posterior
sampled world. At the resulting public state, compare each of the shared
second-stage grid requests with immediate MPC handoff using six fresh
posterior continuations. Refit the same six-output trainer on the
historical, contrast and child records. The successor state uses the
current belief, witness ledger, probe kinematics, timestamp and resources.
The public forecast averages over possible worlds. It does not make
per-hidden-world action labels.

Relabel the four contrast nominations and the same baseline at each of the
128 original starts by executing one probe, applying the frozen child
selector at the measured successor state (possibly handing off), then
executing common MPC. Use the original eight-sample bank. Match root rows
and features with the contrast one-step student. Preserve both rewards and
costs of intermediate failures. One- and two-probe arms use the existing
outcome predictions and selection objective. All arms can hand off at
their designated decision points. At a given scan frame the MPC tail has
the same recovery rights across arms, based on a single deadline two
blocks after the initial decision. Thereafter it returns to its common
triggered recovery rule. MPC always updates from the true synthetic
observations it receives and replans.

## Splits and tests

* Child training seeds start at 74,500,000. Fixed-program development starts
  at 74,600,000, eight roots per family × loss-context cell and three
  tapes per root. Enumerate legal stock and expanded first commands
  with a second grid command or handoff. Select best one-command and
  two-or-fewer-command policies independently by family, with completion,
  final wrong, travel, time, then turn as the ranking.
* Main holdout starts at 74,700,000. 32 roots per family × loss-context
  cell, eight common tapes per root, a total of 1,024 runs per arm.
  The scene seeds are the independent statistical units.
* Drifting-world check starts at 74,800,000, 12 roots per cell, again eight
  tapes each. Add a physical 1.8 mm lateral and 0.65 mm depth sinusoidal
  displacement, 3.0 s period, to the truth seen at every frame including the
  common prefix. Rerender mask and centroid probabilities. Keep the
  public estimator and policies frozen. Treat this as an out-of-model
  motion stress, not validation of different anatomy or real probe motion.

Each arm on a tape shares the same prefix, anatomy and time-indexed random
numbers. Different trajectories generate their own observations. Fixed
commands use current physical pose and standard safety/servo feedback.
On an unavailable fixed request, use normal MPC. Same-first precommitment
uses eight independent posterior replays to prespecify the second request
before any actual probe observation, with the first request from Program B.
Shuffle substitutes the Program B second request from another tape of the
same case and first request, and attempts it safely at the current pose.
The latter ablates agreement with the actual first-probe history and exit.
It keeps low-level feedback and the rest of the public scan intact.
Log intended and realized second commands, infeasibility, first-probe
exit velocities, and remaining distance to distinguish stopping from
constraint-driven fallback. These controls test combined dependence on
history and exit, including any resulting feasibility differences.

Primary outcome is correct completion with at least one newly captured
correct retained witness after the common start. Report all correct
completions, final and interim misidentifications, renewed query images,
and the strict centered metric as auxiliary outcomes. Actual simulated
TCP travel, accumulated change of direction, elapsed scan time,
acquisition count, and decision CPU are reported separately. Decision
CPU is for selector and candidate preparation at the designated decision
points (not all common MPC execution), and is not advanced on the
simulation clock. Physical caps are 67 mm and 8 s. Confidence intervals
use 8,000 bootstraps of whole paired case clusters. Other reported subgroup
intervals are diagnostic.

Verify legacy one-probe label equality and that direct handoff produces the
same result as the full MPC specified by the previous protocol. Replace
oracle truth fields in a child's scoring record while keeping its public
state to check unchanged features and predictions. Check initial
state/velocity continuity for every true execution. Freeze hashes and
case IDs before the two independent holdouts. An exact two-hypothesis
sanity check tests how a negative observation can reverse the preferred
action. It is separate from the simulator outcome.

## Reproduce

From this experiment directory with the released historical artifacts:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python run.py first single children child_fit program_labels program_fit development freeze test drift report \
  --workers 20 --out results/reproduction
```

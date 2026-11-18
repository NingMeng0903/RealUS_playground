# Backward training of B with paired continuation labels

The released experiment at b66151b used distinct second-stage training
but pooled one-step and two-step labels in the first-stage value
regression. Its 923/1,024 one-probe outcome-contrast student is the main
deployment reference. Its 908/1,024 learned two-probe policy is the
legacy two-stage reference. All published policies are **replayed** on
new case streams. Historical scores are identifiers for the models.

## Policy and training targets

Let the public scan history $h$ include the current posterior, observations
and witness ledger, measured probe state and remaining execution resources.
Let $a$ be one legal short probe motion. Starting from the same public
history and simulated world/tape, execute $a$ once using streaming
observations and within-segment tracking. Copy the attained feedback state,
history and resource totals to each continuation:

1. $Y_D$ completes the scan with the full common MPC immediately.
2. $Y_C$ lets a frozen successor policy select a legal second probe or
   immediately hand over. Complete with the same full MPC.

The target vector retains correct task completion, final wrong identity,
interim wrong identity, actual TCP travel, turn, and scan time. For both
head labels subtract the common replay of immediate MPC at the root.
The paired increment is the mean of $Y_C-Y_D$ on these stochastic paths.
A negative increment stays negative. If the successor hands over, the
two rollouts and labels coincide by construction.

Define the PSDP-style version in backward order:

* First generate actual successors by executing legal first motions from
  the earlier outcome-contrast and legacy program selectors, public
  evidence nomination and broad random candidate coverage. Each of 96
  new training scenes attempts four distinct legal moves using two
  simulated physical perturbation tapes. The first move uses streaming
  feedback. At the reached public exit each legal second
  option is labeled with eight independently resampled posterior futures
  followed by full MPC. Freeze and hash this child before root labeling.
* On **disjoint** 128 root-training scenes keep the paired outcome-contrast
  nomination rule, including weaker controls at similar path length.
  Also label relevant published selectors' requests and broaden feasible
  menu coverage. Use at most nine first requests per root and eight
  paired label tapes. Nomination uses separate screening tapes. No
  pair-root training data are supplied to the child. All train and test
  policies choose from public features and feasible trajectories.
* Fit the six existing gradient-boosted regressors to the **difference**
  between the two continuations. Combine this with the frozen 923
  direct head. At each root, explicitly offer each legal motion
  followed by direct MPC, and each legal motion followed by the
  conditional child. Offer immediate MPC too. The first action selected
  by MPC is allowed with an adaptive continuation as a separate
  action from immediate MPC. The child selects or stops from the actual
  first-probe observation history. The result is an at-most-two-probe
  policy with early handover.

This is a PSDP-style use of a frozen later policy to supervise an
earlier one. It uses paired outcome regression and the same learned
controller's action ranking (completion first, published utility for
ties). The old lexicographic selector is kept as an experimental control.
Final misidentification affects the secondary ranking.

## Controls and frozen comparisons

- Full MPC, released fixed best single retry, random-data B, evidence
  teacher B, previously published evidence B, and the 923 outcome-
  contrast B are the established references. Legacy two-probe B keeps
  both published heads.
- The contrast-child arm keeps the 923 first choice and then delegates
  to the freshly trained frozen child. The matched direct head
  uses only the same new root rows as the matched adaptive head.
  Both trained matched first selectors can execute the same child if
  they retain control. Their distinction is which continuation was
  supervised for the first decision.
- The matched mixed head has exactly the same first rows, features, model
  family and online child as the clean adaptive head. A fixed,
  seed-determined 7.3% of rows have adaptive labels, with direct labels
  on the other rows. The mixed residual arm applies the corresponding
  masking (direct rows give zero residual) and uses the same optional
  continuation selector. These controls match the legacy mixing
  ratio in the new design. Both heads see the same amount of data.
- The same-first direct control uses exactly the PSDP first command
  and then immediately returns to MPC. Same-first precommitment replaces
  child **only on trials where the root chose conditional continuation**.
  It chooses a fixed second command, or handoff, before executing the
  first, from the root's public posterior and eight hypothetical tapes.
  Plans without adaptive intention continue identically in both arms.
  The legacy precommitment does the same thing for the legacy program's
  first action with the same number of planning tapes. The matched
  controls share the actual first motion and exit and retain ordinary
  low-level feedback. They measure the value of the follow-up decision
  tied to that particular observation history and exit. Precommitment
  has a different CPU cost.

## Frozen held-out evaluation

- Training: 96 successor scene roots from 79,010,000 and 128 separate
  first-label roots from 79,110,000. Main holdout: 128 from 79,210,000.
  Drift holdout: 48 from 79,310,000. The latter uses the inherited test's
  physical lateral and depth sinusoidal displacements with the same
  estimator. In each group balance connection and non-fork tasks crossed
  with natural segmentation perturbations and a forced loss burst.
- Each held-out scene uses eight paired time-indexed perturbation tapes
  for each arm. All arms get the same initial MPC sweep, posterior,
  evidence memory, feasible requests, vessel model and ProDMP. Each
  trajectory generates its own causal observations on the shared tape.
  The full MPC replans after each executed block.
  Its initial expanded recovery allowance ends at the same absolute frame
  regardless of B's actions. Physical bounds are 67 mm, 8 s and the
  common trajectory and handoff feasibility tests.
- Primary outcome is correct scan completion with newly captured and
  retained correct evidence. Report both final and interim
  misidentification, executed simulated distance, turn and scan time,
  and decision CPU separately. CPU does not advance the simulation clock.
  Pointwise intervals use 8,000 paired bootstraps of whole scene clusters
  within the task/loss strata. A positive lower interval versus the frozen
  one-probe B is the completion evidence standard. The matched
  continuation arms test the role of supervision and
  observation-conditioned scheduling. All training uses separate cases.

Reproduce from this directory.

~~~sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_psdp.py successors child_fit pairs root_fit freeze test drift summary --workers 27
~~~

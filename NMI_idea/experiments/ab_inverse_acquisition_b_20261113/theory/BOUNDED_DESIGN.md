# Task-preserving inverse acquisition: bounded design and decision rules

Status: design recommendation, 2026-11-13. This note does not report a positive experimental result. It is based on the code at `review/b-mpc-validation-20261113`, starting from `ad966a6`, and on the three archived B studies. The implementation and outcome audit must determine which claims survive.

## Recommendation

Change the available acoustic action, rather than fitting another advantage gate on the exhausted action library. At an already-required scan site, propose a probe orientation that is expected either to produce a correct identity claim immediately or to distinguish posterior hypotheses that favor different executable continuations. The acquisition replaces a normal scan acquisition and preserves the task's ordered sites. No extra paid probe is introduced.

The first experiment should freeze a small proposal menu after the common anchor observation. The same menu, checked edges, estimator, identity rule, resources, and full-episode suffix controller belong to every arm. This is an initial test of observation-action synthesis. It is not repeated online continuous optimization, a population oracle, or a proof of improvement over MPC.

Two distinct effects must be measured:

1. **Generator effect:** Does shared MPC with the public inverse-generated menu improve on the same MPC with the legacy menu?
2. **Selector headroom:** With the generated menu already available to shared MPC, can a repeat-validated executable continuation improve on that MPC at matched realized cost?

Only the second effect justifies developing B as an additional continuation selector. If the first effect is positive and the second is absent, the contribution is the observation-action generator integrated with MPC. Naming that whole generator B is defensible; claiming that a B selector outperforms equally equipped MPC is not.

## What the failed experiments already covered

The full-B study already implemented complete episode execution, public-history options, two and three observations, common Bayesian updates, correlated episode nuisance, opposite views, alternate dIRD realizations, recovery, early handoff, and identical suffix MPC. The continuation study already expanded to every mechanically witnessed first command and many second commands. Its split-repeat, per-case cost-matched opportunity was approximately zero. The relational study already tested information from all returns and a paid observation. These are not missing ingredients to be reintroduced under a new name.

The fixed daughter continuation after two commands found a small scheduling defect: it advanced from an accepted site while MPC spent another capture there. That is evidence about a task-progress decision in the approximate controller, not evidence of a general information advantage. There was no corresponding trunk result.

The remaining plausible mechanisms are therefore distinguishable:

| Candidate cause | Test | Interpretation of a positive result |
| --- | --- | --- |
| Coarse action orientation | Expand geometry-derived, executable tilts; compare enriched and legacy MPC | Action generation can matter |
| Approximate controller objective or scenario quadrature | Same expanded actions; exact observation summation or continuation replay | A controller approximation matters |
| Decision-relevant posterior uncertainty | Conditional utility disagreement and repeat-validated prediction | There may be value in identifying which continuation is appropriate |
| Observation impossibility or redundant views | All legal kernels fail to distinguish utility-disagreeing hypotheses | More gating/learning cannot create information |
| Coarse or misspecified model | Predictive likelihood and claim calibration fail at proposed views | Apparent inverse benefit may be model exploitation |

## MPC, model-based B, supervised B, and RL

The present MPC is not a hand-written route rule. `control.mpc_action` enumerates feasible plans and optimizes a model-based objective. It uses a Bayesian posterior, joint probability mass for previous claims, a wrong-identity penalty, motion prices, a nominal terminal potential, ten first-observation quantiles, and a fixed two-capture plan during each optimization. It then executes the first action and replans after the real packet. “Full” describes execution through task termination; its optimization horizon is still two captures, and the second action is fixed within that optimization.

The proposed inverse generator is also model-based. It searches for an action from the predicted observation distribution and an explicit decision objective. It need not learn anything.

A fitted predictor of paired terminal completion differences is supervised learning: it regresses measured counterfactual outcomes from simulator data. The previous ridge B was of this kind. It is not RL merely because its labels came from trajectories. RL would train a sequential policy or value function against cumulative consequences; it may itself be model-based or model-free. Neither an RL label nor an inverse-model label establishes a scientific contribution. First establish useful executable headroom and public predictability.

## A tractable proposal menu

Use a one-dimensional tilt parameter at each required material site. Keep the acoustic origin and its registered robot target consistent. The bounded implementation now precomputes endpoint and edge checks on

`{-12, -9, -6, -3, 0, 3, 6, 9, 12}` degrees.

The legacy grid is `{-12, 0, 12}`. All six new tilts are within its angle envelope. A finite grid samples a continuous parameter; it is not a continuous optimum. A larger initial proposal with `+/-18` and `+/-24` degrees was rejected in favor of this bounded interpolation test. The original model has no empirical acoustic calibration validating these intermediate orientations; “interpolation” refers to the support of the legacy geometric experiment.

At the common anchor:

1. Obtain the public posterior `b` using the same anchor packet in all arms.
2. At each of the two required sites, select at most one nonlegacy tilt by the direct-claim score below and at most one different nonlegacy tilt by the decision-information score below.
3. Preserve all three legacy tilts. Do not replace a legacy action to make the new generator look useful.
4. Attach at most the two existing kinds of dIRD-ranked realization to a chosen acoustic pose, and admit only checked sampled motions. The realized TCP pose, acoustic pose, arrival clock, and integrated cost must agree.
5. Freeze this menu for the complete paired episode, including the MPC tail. The arm is not allowed to regenerate proposals from private case state or suffix outcomes.

Mechanical feasibility must be evaluated jointly. A dIRD margin or valid endpoint is not a complete route certificate. For a forced continuation, retain a checked full-completion witness under the common remaining resources. Distinguish this ideal-claim witness from the actual stochastic completion outcome.

If a tilt has no full witness, it is not a proposal. If the two objectives choose the same tilt, retain one; do not add an arbitrary second tilt just to fill a quota. Tie-breaking is fixed and public. A useful control is a fixed, belief-independent pair of new tilts selected without evaluation outcomes, using the identical menu cardinality and robot checks.

## Exact proposal objectives

Let `h` index the public model particles, `b_h` be their posterior weights, and `a` be an acquisition at a proposed tilt. The actual unknown case is never passed into these computations. Let

\[
L_a(o,h)=P(o\mid h,a),\qquad C_a(o,h)=P(o,\text{selected return is target}\mid h,a).
\]

These are the full public observation likelihood and target-joint mass from the shared sensor interface. Both include misses, multiple-return competition, bins, and the physical arrival clock. Define the public claim rule

\[
\alpha_a(o;b)=\mathbf 1\{o>0,\ b^\top C_a(o)\ge .87\,b^\top L_a(o)\}.
\]

The claim rule is not evaluated under a point-mass posterior for each particle. That would silently give a model-particle calculation clairvoyant identity decisions.

### Direct task-progress proposal

The first orientation maximizes

\[
S_{\rm claim}(a)=\sum_o\alpha_a(o;b)\left[b^\top C_a(o)-\lambda\,b^\top(L_a(o)-C_a(o))\right],\qquad\lambda=1.
\]

This is exactly the expected correct-claim reward minus wrong-claim penalty for an acquisition at a required site under the existing confidence rule. It is not an entropy measure and is not a prediction of full scan completion. It can select a better measurement without adding a capture. Mechanical constraints and actual full-episode outcomes still decide whether the benefit survives.

If costs are included in ranking, use the inherited MPC motion prices, frozen before outcomes. Do not fit a new success threshold or price on the new evaluation cohort. Keep raw scores and the effects of each cost term in the archive.

### Decision-information proposal

Choose a small, fixed set of reference continuations `c`: the three legacy-tilt acquisitions at the next required site, with a common nominal suffix. When proposing at the last required site, use the three legal last-site reacquisitions instead. Freeze the reference clock, reference robot state, remaining resources, and continuation availability before scoring candidate tilts. If a reference continuation is mechanically unavailable from a proposed exit, exclude it for that candidate and report the changing set; do not score the utility of an impossible action.

For the inexpensive first screen, make each particle utility `U_c(h)` from the actual public claim rule held fixed at the anchor belief. One fully specified version is

\[
\begin{aligned}
p^C_c(h)&=\sum_o\alpha_c(o;b)C_c(o,h),\\
p^W_c(h)&=\sum_o\alpha_c(o;b)[L_c(o,h)-C_c(o,h)],\\
U_c(h)&=(1-p^C_c(h)-p^W_c(h))\Phi_c^{\rm unresolved}
          +p^C_c(h)\Phi_c^{\rm advanced}-\lambda p^W_c(h).
\end{aligned}
\]

The `Phi` terms are the same resource-aware nominal suffix potentials used by the shared controller. In the final-site reference state they reduce to unresolved and completed task values. This utility is tied to useful future claims and legal progress, yet remains a surrogate: it freezes the reference acceptance rule and does not recompute a full adaptive suffix for every candidate observation.

For a fixed set of utilities, compute

\[
J(a)=\sum_o\max_c\sum_h b_hL_a(o,h)U_c(h)
      -\max_c\sum_h b_hU_c(h).
\]

This is the expected value of the sample for selecting among the reference continuations. No explicit posterior division is required. It measures whether the observation can change a useful decision, not whether it reduces every uncertain model parameter. It is cheap: with 256 particles, 99 observations, three reference continuations, and nine tilts, it is a small tensor contraction.

For fixed utilities and a normalized likelihood, `J >= 0` to numerical tolerance. A negative value indicates an implementation error or a comparison in which the reference utility/action set changed. Do not clip an appreciable negative value and continue. If all `U_c(h)` are ordered the same way for all posterior-supported `h`, `J=0` for every acquisition.

Important boundary: `J` does not pay the acquisition's resource cost, include its own progress or wrong identity, or account exactly for a changed exit robot configuration. It is a generator score only. Never report it as expected full-completion advantage. The complete forced-action-versus-enriched-MPC replay supplies that quantity. If the headroom screen is positive, `U_c` can later be replaced by paired full-suffix policy values under each model particle, with the public filter retained during each simulated suffix.

### Useful diagnostic bound, not a new theorem

For any fixed `U`,

\[
0\le J(a)\le \sum_hb_h\max_cU_c(h)-\max_c\sum_hb_hU_c(h).
\]

The upper bound is the perfect-information decision gap for this reference problem. It lets the implementation separate absence of consequential uncertainty from inability of the available observations to reveal that uncertainty. It does not bound the benefit of changing robot position, direct claim probability, or the entire task.

For two continuations, put `D_h=U_1(h)-U_0(h)`, `D_h^+=max(D_h,0)`, and `D_h^-=max(-D_h,0)`. Then

\[
J(a)=\min\{\sum_hb_hD_h^+,\sum_hb_hD_h^-\}
 -\sum_o\min\{\sum_hb_hL_a(o,h)D_h^+,\sum_hb_hL_a(o,h)D_h^-\}.
\]

This identity follows by subtracting the two-choice Bayes regrets before and after the observation. It makes the mechanism explicit: both meaningful continuation disagreement and observable separation are required. It is a standard value-of-information identity, not a new theorem for robotic ultrasound.

The current inexpensive implementation uses `U_c = p_c^C - p_c^W` without the `Phi` terms. That is a signed identity-claim utility, which remains a valid fixed decision utility in the `J` equation. It must be described as an identity-decision surrogate rather than a complete-scan value. The full-completion opportunity comparison is unchanged.

## Geometry and observation validity checks

The continuous orientation extension must reproject common world geometry. Do not interpolate already-quantized likelihood tables, linearly blend rotation matrices, or create an independent random anatomy/clutter field for each new tilt.

There is a specific legacy issue to address: `MapWorld._persistent_clutter` uses `(case.seed, site, view)` to determine whether its strong ghost exists. The displacement is coherent across views, but the existence switch is indexed by the discrete view ID. Assigning a fresh ID to every angle would create discontinuous, arbitrary opportunities for inverse search. A physical continuous model would use one case/site ghost anchor and a declared visibility rule shared across angles. If that changes the legacy-angle sensor model, run a legacy-menu control under the new kernel as well as the archived legacy reproduction. Never attribute a kernel change to action generation.

The current interpolation implementation instead preserves exact legacy-angle kernels and smoothly transports/interpolates their effective ghost means and visibility. This avoids fresh random draws from new view IDs, but is a declared view-conditioned synthetic ghost model. Whenever its endpoint regimes differ, the interpolated mean is not the projection of one rigid material ghost. Report this distinction. If gains concentrate on those transitions, a canonical single-anchor ghost control is needed before interpreting them as a physical sensing effect.

`KneeWorld.detection_probability` is currently a constant by identity-site status. It does not contain a calibrated orientation-dependent echogenicity response. Preserve that fact. New tilt effects can arise from geometry, aperture, anisotropic image noise, image-depth bias, return competition, and quantization. They cannot support a claim about acoustic anisotropy or visibility of a vessel wall unless those phenomena are added and independently validated.

Pure translation is mostly ineffective in this kernel. Translating the probe shifts both the centroid mean and its nominal-centered bin boundaries. Away from aperture clipping, these shifts cancel in the bin probabilities. Therefore do not add tangential offsets as though they necessarily increase observability. Tilts can change the projection of material displacement relative to depth calibration and unequal axis noise. Their value is still an empirical question, and the existing opposite extremes may already supply nearly all of it.

Minimum synthetic checks before outcomes:

- Recover the legacy `-12, 0, +12` projections from the declared transform convention to numerical tolerance.
- Preserve world anatomy, motion phase, calibration bias, and clutter anchors across all candidate poses for a case.
- Normalize every likelihood and maintain `0 <= C <= L`.
- Check smooth changes in continuous means before quantization. A change in a view ID must not change a random seed or latent object.
- Check actual FK endpoint against the acoustic origin and rotation, not only against a nominal target ID.
- Keep physical-clock/channel disturbance pairing. Different paths may intentionally meet different clock values; acquisition index pairing is not a substitute.
- Audit old-angle kernel change separately from new-angle effects.

The renderer and the online likelihood share structural assumptions. Fresh random cases test within-model generalization; they do not independently validate the renderer. A later real- or image-based validation must check prediction and identity calibration at proposed poses before deployment claims.

## Bounded experiment and stopping rule

### Phase 1: opportunity, before learning

Use fresh latent cases in daughter and single trunk, the same common anchor packet per paired case, and 24 independently resampled suffix tapes per case. Keep the prefix tape fixed. The final protocol uses 32 cases per task. This is a bounded development screen and is not enough to certify the historical 2% harmful-discordance criterion.

For every case run:

1. Legacy-menu MPC under the new continuous-consistent sensor kernel.
2. Inverse-menu MPC to task termination.
3. Every feasible first action in the inverse menu, followed by that exact inverse-menu MPC to task termination.

The forced action must carry the real exit belief, previous-claim mass, robot node, progress, physical clock, and remaining resources. A first action that equals MPC's own command is a duplicate. Previously completed or stopped cases remain in the denominator, with incumbent fallback when no alternative is available.

Use half the tapes to choose among mechanically legal continuations within the declared ranking-half cost constraints, evaluate the chosen action on the other half, and swap halves. Archive both selections. Report the full-denominator paired success difference, any wrong identity, harmful/favorable identity discordances, and actual TCP distance, rotation, rail, captures, duration, and measured CPU. Use case-clustered intervals; do not count 24 tapes from a case as 24 independent anatomical cases.

Keep the historical hard caps and realized-motion margins if comparison with the earlier studies is intended. Strict two-sided cost equivalence and one-sided no-worse cost are different outcomes; report both explicitly rather than treating faster duration as failure without context. A ranking-half predicted match is not a realized match. CPU time is additional to motion time; if the acoustic clock excludes planning, state that limitation.

**Stop before learning** if the fixed primary opportunity screen fails. The final protocol requires at least a three-point daughter completion gain, a positive case-paired lower confidence limit, no net wrong-identity increase, realized no-worse cost limits, and improvement over the fixed progress control. Report strict equivalence separately. A daughter-only gain does not support a generality claim; a practical trunk regression blocks one. A small positive point estimate with an interval spanning zero is an inconclusive development result, not evidence that a learning stage will recover a +5-point advantage. Do not use a sequence of gate changes to advance a failed screen.

If the generator improves legacy MPC but no continuation headroom remains above enriched MPC, stop selector work and retain only the generator result. If neither contrast improves, the action refinement has failed in this renderer. The next research step is independent evidence for a richer observation model or a different task need, not a renamed classifier on the same labels.

### Phase 2: predict and use, only after positive headroom

Freeze the generator and its mechanics. Use disjoint training, development, and new prospective case seeds. Predict the paired complete-scan difference for the legal continuations from public posterior/evidence, geometry, progress, and remaining resources. No case seed, latent anatomy, suffix tape, private return identity, or label-assisted oracle choice may be an online input.

For this initial anchor-only implementation, the entire public state is a deterministic function of `(task, anchor packet)`: the prior before that packet, robot state, progress, resources, and clock are fixed. There are at most 99 packet values per task. Therefore a public-packet action-value table with training-only shrinkage is a natural first predictor and a sufficient representation of this public state. A more elaborate network cannot recover private variation within the same packet. Cross-case prediction failure alongside positive within-case split-repeat headroom would diagnose unobservable case heterogeneity or insufficient sample size, rather than establish that another feature encoder is needed. If a later experiment adds richer public histories, retain full decision-utility signatures or validate any posterior compression.

The initial implementable selector can remain model-based: draw hypothetical cases from the public posterior and compare full continuations on paired model tapes. A learned surrogate is justified if it predicts those decisions accurately and improves the practical time ledger. RL should be introduced only for a demonstrated need for sequential policy learning, not to rename the proposal mechanism.

Compare against enriched MPC, the best training-only fixed continuation with fallback, a belief-independent matched-cardinality menu, and a direct-claim-only generator. The decision-information contribution requires improvement over the direct-claim-only design, not just over the old three-angle bank. Report both tasks and all assigned cases. The prospective test must freeze the predictor, selection rule, menu construction, identity threshold, and source hashes before outcomes are read.

## What can be claimed

Decision-region active sensing and value-of-information objectives are established. For example, [Javdani et al., AISTATS 2014](https://proceedings.mlr.press/v33/javdani14.html) explicitly studies choosing measurements for decisions rather than reducing all hypothesis uncertainty. [Golovin, Krause, and Ray](https://arxiv.org/abs/1010.3091) studies noisy, potentially correlated tests and their costs. The present setting does not automatically inherit those papers' guarantees: movement changes the available tests, progress, resources, and terminal outcome.

A credible contribution would be the concrete coupling of decision-conditioned observation synthesis, dIRD proposal ranking, checked full-route feasibility, and preserved task progress, with fresh evidence that the coupling matters beyond equally equipped replanning MPC. An inverse objective alone, ordinary rollout policy improvement, or a generic “uncertainty is decision relevant” proposition does not meet that standard.

The current evidence is synthetic nearest-centroid sensing on a nominal chart with sampled robot kinematics. It does not establish contact stability, real ultrasound image validity, continuous collision freedom between samples, or a real-time deployment loop. Those boundaries should be attached to the result itself rather than hidden in a later caveat.

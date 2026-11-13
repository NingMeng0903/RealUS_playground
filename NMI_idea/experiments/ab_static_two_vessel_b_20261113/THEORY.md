# Static two-vessel B mechanism screen

Theory review, 2026-11-13. This is a deliberately small decision-theory check, not a new anatomy, a clinical validation, or an efficacy result. Use one registered artery and one vein in the same coordinate system, a static scene, and the existing common MPC prescan. No pulsatility, flow, force, topology change, or learning is needed.

## 1. Question and likely failure mode

The question is whether valuing a future diagnostic choice **before its observation** changes completed arterial inspection beyond ordinary MPC that receives every new observation and replans. It is not whether two views contain more information than one, whether updating beats a frozen map, or whether hindsight can find a good path.

The initial proximal-thigh crop reportedly has approximately 12–18 mm artery–vein center separation and roughly 5 mm radii. Both lumens can therefore fit many ordinary transverse apertures. The implementation instead has a stronger physical connection available: apply the same archived static 15-degree SMPL-X joint transforms to both Among_US meshes, then use the existing dIRD/FK action bank at sites 3, 4, and 5, with center and +/-12-degree views and archived legal edges. The parent reports that the actual planes at these sites intersect one closed near-target lumen per vessel class. Freeze these physical assets before inspecting adaptive gain.

With only shared global translation uncertainty, one view containing both anonymous centroids generally identifies their correspondence to a known oriented pair. A strong fixed probe may already remove almost all ambiguity. This is a reason to measure and accept a null result, not to narrow the aperture or enlarge the registration error until B wins.

The planner needs the labeled registered templates, or an independently anchored arterial seed. New observations must contain only image geometry. If all the planner has is two unlabeled tubes with exchange-symmetric priors, geometry alone cannot identify the biological artery: swapping the role labels leaves the public history unchanged. Under equal probabilities the identity accuracy ceiling is one half.

## 2. Frozen physical and information contract

Use a crop chosen by anatomy coverage and available registration, before evaluating adaptive gain. Publish the crop, template transforms, nominal probe aperture/depth, sensor quantization, uncertainty support, candidate physical poses, trajectory cost table, and task rule. Candidate action addresses are physical probe coordinates, never artery/vein IDs or hidden-hypothesis addresses.

Start all arms at a byte-identical checkpoint: common acquired frames, posterior, probe pose, physical clock, and remaining resources. The prescan must actually update the same belief used later. Charge its cost, although it cancels in the paired contrast. Do not give B a deliberately less informative prescan or announce different task information.

Both live MPC and B may use every diagnostic action, repeat a view, stop, and select any feasible terminal continuation. They receive the same packets, use the same likelihood, and update and replan after every diagnostic packet. Both include the final arterial inspection in their lookahead. Fix ties by public action coordinates; report whether a claimed difference survives ties within numerical tolerance.

The smallest geometry uncertainty is a declared set of 16–48 shared rigid registration shifts applied to **both** vessels. Derive the shift scale from the registration assumption, rather than the desired outcome. Do not give artery and vein independent artificial displacements merely to create difficult cases. Add a shared calibration nuisance only if needed to represent the sensor. No role-specific noise or detection probability is permitted unless it is independently supported by observable geometry.

Keep the task positive: acquire two new qualifying frames on the named artery at sites 4 and 5, within the declared geometric tolerance and resource caps. The vein is a real distractor. A wrong-vessel frame is not successful completion, and abstention stays in the denominator. Existing prescan frames cannot satisfy freshness.

Use the concrete two/three-image contract: after the common site-4-to-site-3 prescan, a new site-4 packet is the near task frame, and a subsequent site-5 packet is the far task frame. One extra diagnostic at site 3 or a repeat near view may precede the far capture if the legal action graph and remaining cap permit it. A qualifying new site-4 diagnostic is already the near task frame, so it is not counted twice. A site-3 diagnostic cannot replace the required new site-4 frame. The public task state records which fresh near packet is eligible and the path must actually execute the far acquisition afterward.

Count physical captures literally. Two separate diagnostics followed by a separate two-frame terminal program would consume **four** new images. Calling the terminal pair one macro-action would not make that a two-image test. The contract above avoids this: information and useful task coverage may come from the same image.

## 3. Well-defined two-observation posterior

Let the finite hidden state be `x=(h,eta)`, where `h` is a shared registration hypothesis and `eta` contains any persistent pose/segmentation nuisance. The registered vessel surfaces are fixed. Let `b0(x)` be the posterior **after** the shared prescan. A packet is an unordered, quantized set of zero, one, or two image-derived returns. Centroid, radius, or a small mask summary is acceptable; role labels and stable simulator object IDs are not.

Precompute a normalized public categorical likelihood `L_a(y|x)` by rendering the same geometry and applying the declared quantization/noise model at physical pose `a`. Visibility follows the real aperture and depth gate. Marginalize the two possible vessel-to-return associations; randomly ordering returns must not change any posterior or action. Use identical role-independent detection/noise, conditional on visibility and geometry.

For a first view `a1`, packet `y1`, publicly selected second view `a2=pi(y1)`, and packet `y2`,

```text
b12(x) = b0(x) L_a1(y1|x) L_a2(y2|x)
         / sum_x' b0(x') L_a1(y1|x') L_a2(y2|x').                 (1)
```

The adaptive action adds no extra observation-likelihood factor: it is already a known function of the public history. Equation (1) assumes conditional independence given the **complete** static state. If calibration bias persists, retain it in `x` until after multiplying the two likelihoods. Marginalizing that bias separately for each frame incorrectly treats repeated evidence as independent. Track and fail clearly on a zero-support public packet; do not silently replace it with a favorable posterior.

This exact finite observation model is a valid declared mechanism model. It is not a calibrated B-mode or segmentation model unless calibration data support that statement. Quantization boundaries must be common physical bins, not hypothesis-specific codes that accidentally reveal which template generated the observation.

For the two-frame task, retain the fresh near packet and its association weights in the public task state. A posterior over registration alone does not preserve which observed near slot can be certified. If `r_N(x,i)` is the conditional probability that observed near slot `i` is arterial given state `x` and its near packet, then after a far packet `y_F`,

```text
P(N=i and F=j are arterial | public history)
  = sum_x b_before_far(x) r_N(x,i) L_F(y_F, arterial_slot=j | x)
    / sum_x b_before_far(x) L_F(y_F | x).                       (1a)
```

Here `b_before_far` already includes the near observation and any other diagnostic, and the association likelihood uses the same physical geometry. Equation (1a) assumes conditional independence of the two frame noises given complete `x`. Equivalently retain a joint belief over `x` and the near association. Do not multiply the two marginal arterial probabilities: shared registration creates dependence. If the final policy selects a near/far slot pair after seeing both frames, optimize this joint probability over the allowed pairs.

## 4. Task value and the exact B-versus-MPC comparison

Let `s` contain actual probe exit, remaining vector resources, and the eligible fresh near packet/association state. Let

```text
Phi(b,s) = max over feasible terminal policies t of E_b[u(x,t)] . (2)
```

For the smallest positive-only screen, use `u=1` exactly when the two new declared slots are both arterial and the physical task completes, and `u=0` otherwise. Maximize the joint probability in (1a); every wrong pair is a failure. This tests correct task selection, without claiming a confidence certificate or controlled wrong-confirmation risk. If such a certificate is required, predeclare the acceptance rule and use correct fresh arterial completion minus `lambda` times wrong-vessel confirmation, with abstention zero. A reward of `+1` for correct and `-lambda` for wrong implies the threshold `lambda/(1+lambda)` for a binary pair-validity claim. Also report completion and wrong-confirmation rates separately; a utility improvement is not automatically a completion improvement.

The terminal value must include the paid new target frames and actual continuation resources. A finite `success[t,x]`/`wrong[t,x]` table is valid if `t` denotes a fully specified physical program and declaration rule. It supplies an exact result only for that program class. If the first terminal image could change the second pose or final declaration, either include that public adaptation in the common terminal solver or explicitly label the fixed-program restriction. A truth-scored endpoint probability is not by itself an image-supported identity certificate.

In the concrete site-4/site-5 contract, `Phi` is the common one-step choice of a feasible far view, followed by the far observation and the common joint-pair decision. It is available only after an eligible **new** site-4 observation. For an initial site-3-only diagnostic, no immediate far completion is allowed. In particular, MPC may select a different far view after each possible first near packet when forecasting an immediate finish. Its forecast terminal solver is already conditional; only the choice of whether/where to acquire a second diagnostic is held fixed in its stochastic open-loop lookahead.

After first view `a` and packet `y`, write `b_y^a` for the update and `s_a` for the actual exit/resource state. The next option `c` includes immediate finish as well as every affordable second diagnostic followed by the same terminal solver. Define

```text
Q_a(y,c) = expected task utility of c from (b_y^a,s_a),
F(a)     = max_c sum_y p(y|b,a) Q_a(y,c),
G(a)     = sum_y p(y|b,a) max_c Q_a(y,c).                        (3)
```

Include action costs consistently, either as hard vector debits or a predeclared price in both values. Feasibility depends on the actual first exit and remaining resources. The geometry is static, so a diagnostic observation must not secretly change robot reachability.

Open-loop stochastic MPC selects `a_M in argmax F(a)`, executes it, observes the real packet, and replans the second option. Therefore its **executed** expected value is `G(a_M)`, not `F(a_M)`. Exact B chooses `a_B in argmax G(a)`. The incremental capacity is

```text
Delta_B = max_a G(a) - G(a_M).                                 (4)
```

The nonnegative Jensen gap `G(a)-F(a)` is not the B gain. It can be positive for every action while MPC and B choose the same first view. If their first actions agree, the shared one-step replanner, observation update, and tie rules make the two policies pathwise identical for a two-diagnostic horizon. An especially robust positive result has `max G - max_{a in argmax F} G(a) > epsilon`, so it does not depend on an arbitrary MPC tie.

If the purported MPC already optimizes the same contingent belief tree, it is the exact B policy. No independent B advantage exists in that case; only an approximation or computational-efficiency claim could remain.

For a third genuinely adaptive diagnostic, use the ordinary finite-horizon recursion

```text
V_k(b,s) = max{ Phi(b,s),
                 max_a sum_y p(y|b,a) V_{k-1}(b_y^a,s_a) },
V_0       = Phi.                                                (5)
```

Again evaluate live MPC by recursively following its chosen first action and then replanning; do not compare B against MPC's forecast fixed path. Two diagnostics are sufficient for the first screen. Add a third only if the declared scientific question requires that horizon, not after inspecting a disappointing result.

## 5. Exact computation and honest upper bounds

With tens of hypotheses, a few poses, and a finite packet alphabet, exhaustive sums and maxima are practical. Cache `L[a,x,y]`, physical edge costs, and terminal utility vectors. Propagate unnormalized branch weights `w_x=b_x L_a(y|x)`; task-value functions are positively homogeneous in these weights, so most branches need no posterior division. Omit exactly zero-mass branches. The two-diagnostic search has polynomial enumeration over two action/packet levels, rather than a learning problem. Verify the vectorized calculation against an independent direct enumeration on a tiny subproblem.

Compute these controls exactly from each shared checkpoint:

1. Immediate continuation with no extra diagnostic (`V0`).
2. Best single physical diagnostic, with full posterior-conditioned continuation (`V1`).
3. Best fixed two-diagnostic sequence, with the same final posterior/continuation. Selecting the best fixed pair separately at each shared checkpoint is a strong control; a globally fixed probe is weaker.
4. Live two-step stochastic MPC, evaluated by actual posterior replanning.
5. Exact two-diagnostic contingent B (`V2`).

Publish the finite-model `V2` as the **exact public-policy ceiling for the declared action, likelihood, terminal-program, and budget class**. It is not a ceiling over arbitrary ultrasound poses or unrestricted terminal adaptation. If a nonexact learned selector were later evaluated, it cannot exceed this ceiling in model expectation.

For a separate optimistic bound, grant the true registration before acting but still require fresh physical target completion through the same physical action/horizon class:

```text
U_PI(b,s) = sum_x b(x) V_known_state(x,s).                       (6)
```

`V_known_state` solves the same remaining physical task with state `x` revealed. It still allows all legal routes, task-qualifying diagnostic captures, repeat captures, and public adaptation to later segmentation failures. Restricting this oracle to an arbitrarily smaller direct-terminal library would not be a valid bound. Do not give it knowledge of future segmentation noise. If even this perfect-state bound barely exceeds MPC, further perception/planning capacity cannot materially help within the budget.

Bounds are in the quantity optimized. A utility ceiling is not a success-rate ceiling. For a simple optimistic success ceiling, separately solve the same tree with reward equal to correct completion only. This is also an upper bound for any more restrictive false-confirmation-constrained policy. A risk-constrained optimum requires the corresponding constrained policy calculation; it cannot be inferred by renaming a penalized utility.

## 6. Pre-test effect and capacity diagnostics

Inspect these before running a larger empirical cohort, and archive them for all considered checkpoints rather than only favorable examples:

- **Prescan headroom:** immediate task success/utility, posterior ambiguity, and `U_PI-V0`. An already resolved checkpoint has no useful reinspection question.
- **Physical observability:** for task-incompatible hypotheses, compare their public packet distributions at each reachable action. If every affordable likelihood is identical, no policy can distinguish those hypotheses.
- **Best-probe strength:** `V1-V0`, residual oracle gap after the best single view, and the best fixed two-view value. A uniformly useful full-width probe is a sufficient simpler method.
- **Structural complementarity:** the probability mass of first packets on which the optimal second action changes, and their task-value gaps. Report the actual geometry and visibility that cause the changes. Counts of switches or entropy reduction alone are insufficient.
- **Increment beyond live MPC:** exact equation (4), the first-action ranking, and the realized replanning value. If only `G-F` is positive, the advertised mechanism is not demonstrated.
- **Continuation cost:** actions remaining after each exit, exact expected capture count/travel/rotation/time, and the full cost distribution or relevant quantiles. An informative view can consume the ability to finish.
- **Capacity diagnosis:** compare actual B to exact `V2`, and `V2` to `U_PI`. Failure to approach `V2` is a policy/implementation issue; small `V2-MPC` is lack of incremental public-policy headroom; small `U_PI-MPC` is a task/budget ceiling.

For a declared target improvement of five percentage points, an exact optimistic success ceiling less than five points above MPC rules out that target in this model before sampling. A strictly positive but tiny gap should be reported as such. Do not lower the effect threshold after seeing it.

Linear-Gaussian translation estimation with state-independent view Jacobians and a quadratic uncertainty objective has an observation-independent covariance update. It cannot, by itself, generate evidence-dependent view rankings. In this experiment any adaptive gain must come from real nonlinear registered geometry, visibility/association ambiguity, or the physical task and resource state. Explicitly do not synthesize an observation symbol whose sole purpose is to reverse an action ranking.

## 7. Resource matching and execution claims

A common cap is necessary but does not mean equal actual cost. Debit every attempted move, rotation, dwell, image, and terminal frame, including failed or abandoned continuations. Carry the actual exit to the next edge. In this static screen, a checked geometric route table can establish exact **simulated** path cost; it does not establish robot contact execution.

Prefer a small mechanically/geometrically checked action menu whose competing routes naturally have equal resource vectors. If that is unavailable, compare exact success-versus-budget curves and report actual expected resource differences and distributions. Do not call the result cost-matched merely because all paths complete or because both use the same maximum horizon. Do not add artificial motion or conceal optional scans to repair a cost comparison after outcomes are known.

Exact enumeration over the declared finite probability model needs no Monte Carlo confidence interval. Model uncertainty remains. If independent generated tapes are also used as a replay check, compare paired cases and retain all failures/abstentions. Repeated shifts/noise tapes on one artery–vein pair are not independent patient anatomies.

## 8. Stages and conclusion rules

**Stage A — contract and numerical audit.** Check likelihood normalization, physically correct projection/FOV, pose and template-label invariance, anonymous-return permutation invariance, shared checkpoint equality, posterior calculation, actual cost debits, and exact enumeration consistency. Make sure the simulator has no hidden identity channel.

**Stage B — matched-model capacity screen.** Freeze geometry, support, probe width, sensor scale, action set, task utility, and budgets. Evaluate all finite-model states/observations exactly. This is the primary minimal experiment. If `Delta_B` is zero or below the predeclared useful size, report no useful headroom and stop this route. If B and MPC agree pathwise, no larger cohort or policy learner is justified by this model.

**Stage C — one small, separately identified robustness check, only after useful capacity.** It is worthwhile to test a modest prior-weight shift and a modest segmentation-noise shift **separately**, with the planner frozen. A prior shift changes truth weights over the same physical registration support; a noise shift changes the observation kernel while preserving anonymity and geometry. Do not simultaneously enlarge both and then attribute the result. Declare the shift before opening its outcomes. Exact reevaluation of the frozen policy under each shifted kernel is preferable to noisy sampling when available. An out-of-support geometry is a separate model-failure test, not the same-model capacity result.

The strongest permissible positive statement from this screen is: “For this one registered static artery–vein geometry and the specified finite sensor/action/budget model, a public-history contingent diagnostic policy has [measured] task value beyond equally equipped live stochastic MPC and the best fixed-probe control, with [reported] actual simulated resources.” This is a within-anatomy mechanism result.

If only `V1>V0`, the result supports reinspection with a good probe. If fixed two-view sensing matches B, it supports complementary views without contingent selection. If exact B exceeds the fixed sequence but matches live MPC, ordinary replanning supplies the benefit. If only same-model B is positive and modest shifts remove it, the mechanism is model-sensitive. None of these outcomes establishes a new general planning theory, realistic B-mode calibration, biological generalization, or a standalone T-RO contribution.

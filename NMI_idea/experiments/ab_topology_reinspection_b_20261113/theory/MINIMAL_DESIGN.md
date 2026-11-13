# Topology discovery followed by fresh target inspection

Design review, 2026-11-13. This is a proposed bounded experiment, not an efficacy result. The prior fixed-Y, relational, full-rollout and inverse-acquisition studies do not establish a reliable cost-matched B advantage. A new experiment is justified only if its hidden world actually contains consequential alternatives that the previous fixed graph excluded.

## Minimal world and public task

Sample the entire hidden anatomy once, before the common MPC prescan. The discrete graph alternatives should include a genuine Y, a single trunk with no qualifying daughter, parallel disconnected vessels, and a projection crossing of disconnected vessels. Randomize branch/crossing location, orientation, radius, separation, depth and curvature independently of the case/tape identifier. A graph assembled from fixed named daughter slots with noisy coordinates is still a known-topology task.

Represent each physical edge by a 3D tube around a smooth centerline in the material chart. A true Y uses a connected volume at its junction; a crossing uses tubes separated in depth. The difference must change an attainable ultrasound slice or short sweep. Giving the likelihood a topology-dependent symbol that the renderer does not produce is label leakage.

Specify the target using public semantics: starting from the vessel seeded by the common prescan, inspect the connected distal daughter that enters the positive-lateral distal region. Choose the target region independently of the hidden edge numbering. Include symmetric cases and apply random graph relabeling. A target such as "edge 2" or a simulator daughter ID is not an acceptable instruction. For worlds with no qualifying daughter, the correct terminal answer is "no qualifying daughter"; report these cases separately from positive-target inspection as well as in the full assigned cohort.

The common MPC prescan produces the same pose, physical time, raw observations and posterior for every arm in a paired case. Branch discovery and target inspection both start from this checkpoint. The evaluator credits target inspection only for newly executed post-checkpoint acquisitions. If the task is announced after prescan, announce the same public target region to every arm then.

## Observation and belief

The hidden state is `(G, theta, ghost, deformation, visibility_state)`: graph, continuous geometry, persistent acoustic artifact geometry, deformation state and correlated visibility. Graph vertex/edge labels are arbitrary internal coordinates. Store the belief on unlabeled embedded graphs, or explicitly verify invariance to random internal relabeling.

At each physically realized imaging pose, the renderer supplies a mask or an unordered set of candidate regions. A candidate may expose only image-derived attributes: centroid, area/radius, eccentricity, local axis, segmentation score and uncertainty. It must not expose physical identity, connectivity, truth-class flags or a stable simulator object ID. Randomize return order; if a network sorts candidates geometrically, repeat the permutation audit before sorting. Do not imply that a real U-Net was run if these are synthetic mask-derived candidates.

A minimal finite-set likelihood sums over partial injective associations `A` between physical visible components and observations:

`p(Y | x,a) = exp(-Lambda) * sum_A [product_matched pD_j f_j(y_i|x,a)] [product_missed (1-pD_j)] [product_unassigned kappa(y_i)]`.

Use one consistent finite-set/ordered-packet convention; its combinatorial factors must match the simulator. Each observation is explained at most once. Near the junction, a merged mask component is a possible observation state, not two independent detections at an identical position. With at most three vessels and one persistent ghost, exact assignment enumeration is practical. A joint likelihood for a short image sweep is also acceptable.

The ghost is sampled once per case and then rendered from physical pose. It can persist, curve and disappear with view. Independent Poisson clutter may be added, but must not replace the persistent artifact. Correlated dropout should be a latent Markov process or a fixed random field indexed by physical time and location. Treating repeated acquisitions of a single ghost as independent negative evidence against it creates artificial certainty.

For a bounded first implementation, use 64–256 graph/geometry particles spanning all four graph alternatives, with the same particles and update in every policy. Preserve low-probability alternatives or rejuvenate from a public-history posterior proposal; track effective sample size and graph extinction. Sampled continuous truth should not always equal one inference particle. A matched-model synthetic cohort is useful, but distinguish it from a stress cohort with withheld curve/ghost families and sensor misspecification.

## Physical action and completion

An action is a probe pose or a short scan segment in public material/world coordinates, plus permitted orientation and dwell time. It is never a hidden branch address. A compact lattice of longitudinal stations, lateral offsets and tilts is sufficient. All arms receive the same public action generator and mechanically checked motion graph. If proposals depend on the posterior, every arm may use that same proposal method.

Execute a complete continuation: move from the actual current exit, acquire at the actual realized pose, update the common belief, choose another action, and continue until a public stop decision or a common hard cap. Carry robot position, physical clock, contact state, remaining resources, prior evidence and task progress through each handoff. A correct predicted exit or a forced first action is not completed inspection.

For a positive-target case, terminal success is the conjunction of:

- the declared local graph connection is correct, including no false connection to a crossing/ghost;
- the declared target corresponds to the public target specification;
- fresh post-prescan evidence covers the requested downstream interval with a maximum permitted gap and belongs to the same connected physical daughter;
- the final inspected location/geometry and image-quality requirements pass;
- motion, acquisition, elapsed-time and safety caps are satisfied.

Use evaluator-only true correspondence to score those conditions. The controller computes its confidence by marginalizing the joint event under its posterior. Multiplying independent marginal confidences for links/frames is generally wrong. Wrong confident completion, wrong connection and abstention must remain separate outcomes. On no-target cases, evaluate a correct no-target declaration; do not credit spurious scanning as a daughter completion.

## What B can mean without claiming an invented algorithm

The minimal B is a computation-limited, task-conditioned policy that chooses a physical probe/continuation from the graph posterior, evaluating the complete fresh-inspection event. Its useful behavior would be to visit a view that distinguishes graph alternatives requiring different continuations, then execute the appropriate daughter scan. It may be implemented as scenario rollout, a recurrent policy or a learned surrogate; those are standard methods.

If a special frontier/diagnostic action generator is proposed, share it with MPC and report its contribution separately. If B and a long-horizon belief planner use the same model/action space/objective, a B advantage can only be a practical approximation, learning/generalization or computation advantage. It cannot exceed the optimal belief policy in principle. Do not describe a win over a short horizon as a new form of Bayesian decision making.

The potential research contribution is the complete ultrasound system contract: uncertain anatomical connectivity and candidate identity, material-chart evidence through deformation, physically executable observation proposals, and a fresh target-inspection certificate. It needs experiments showing which part changes completed scans. That integration is a candidate contribution, not yet a demonstrated T-RO-level novelty claim.

## Controls and bounded experiment

Use all of the following with identical post-prescan state, observations, actions and terminal objective:

1. Live belief MPC, replanning after every acquired packet and running to full task termination. Declare its horizon and terminal approximation.
2. Long-horizon belief search, using the same posterior and simulator and an explicit larger planning budget. The horizon must reach a discriminating view plus a complete target continuation on the designed diagnostic cases.
3. Recurrent PPO with the raw public history or an order-invariant candidate encoding, previous action, pose, resource state and public target. Train on whole sequential trajectories with legal-action masks; reset hidden state at episode boundaries and warm it with the common prescan. No hidden topology/identity features or privileged actor inputs.
4. B with the same public input/resources and a declared training/search budget.
5. At least one targeted mechanism ablation: collapse graph belief to its MAP graph, or remove the ability to revisit a topology-disambiguating view, while preserving the rest. A Fisher/geometry-uncertainty objective is a useful diagnostic control, not the only serious baseline.

Full live MPC and long belief search are different controls: running a short-horizon MPC for a whole episode does not make its foresight long. Conversely, the long control must update after actual observations; an open-loop long path is not equivalent.

Train recurrent PPO with at least three predeclared seeds if drawing conclusions about RL. Training must show learning curves and a functioning memory-dependent sanity task. A tiny failed PPO run is a weak baseline, not evidence against recurrent RL. Compare wall-clock inference and training cost in addition to physical cost. A privileged critic may be studied separately, but disclose it and keep actor inputs public.

If inexpensive, add a PPO variant that receives the same computed public graph posterior as B/MPC. The raw-history recurrent baseline measures the combined challenge of inference and action selection; the shared-belief variant better isolates policy learning. The posterior remains public evidence, not privileged true topology.

Freeze source hashes, distributions, policies, budgets, success definition and primary contrast before the held-out outcomes. Hold out at least a geometry/ghost *family* (for example, curved daughter paths and curved persistent artifacts after training on straight/piecewise-linear families), not only random seeds. Retain all graph types in the posterior support. A withheld topology that has zero inference support is a separate model-failure/abstention stress test, not a fair accuracy cohort for that fixed model.

Use a case-level latent seed and a policy-independent physical-time/space disturbance field. Replaying the same acquisition-index noise after different trajectories does not create a common physical world. Pair all policies on common hidden anatomy and disturbance fields, while preserving their different physical clocks. If runtime changes anatomy, include measured planning latency in that clock, or enforce the same planning-time allocation. Archive both the anatomy and tape keys privately for audit.

For a turn-sized screen, start with a small diagnostic cohort to verify that the graph pair is ambiguous in prescan, distinguishable by a reachable view, and still completable after that view. Then use a frozen independent cohort large enough to show paired discordances rather than only a few selected pictures. An example is 96–128 new cases balanced across graph families, with separate geometry-family stress cases. These numbers are a screen, not automatic statistical power for a small gain.

Report paired task completion, topology correctness, target-identity correctness, harmful confident completion, actual fresh coverage, captures, path length, rotation, elapsed time, planning time and constraint violations. Use case-clustered intervals if each anatomy has repeated tapes. Report every assigned case, with abstentions/fallbacks retained. A cost-matched claim requires both comparable hard caps and realized cost evidence. Plot success as a function of actual budget and compare at common budgets. Do not select policies after seeing realized cost or tune acceptance gates to manufacture equivalence.

## Falsifiable identifiability and feasibility results

These are task-specific audit propositions derived from standard decision theory; they are not claimed as a novel planning framework.

**1. Indistinguishability ceiling.** Let `G0,G1` be equally likely, require incompatible correct terminal declarations, and let `P0^pi,P1^pi` be the distributions of the complete *public* history under any admissible policy `pi`. Then

`P(correct completion | pi) <= (1 + TV(P0^pi, P1^pi))/2`.

Completion requires a correct declaration, so it cannot exceed optimal binary testing accuracy. If all actions taken have identical conditional observation laws in the two graphs, induction through the policy gives `P0^pi=P1^pi`; therefore success is at most one half. The policy's own randomization does not evade this bound. This establishes why a highly accurate local centerline/Fisher estimate can coexist with unresolved connectivity: conditional geometry information need not distinguish the discrete alternatives.

Construct a matched Y/crossing pair with identical common-prescan likelihoods, then numerically verify equality over that corridor. Add one physically reachable discriminating sweep. With increasingly low segmentation noise, full long-horizon search should approach correct topology plus completion if the remaining resources suffice. If it does not, first investigate belief support, likelihood, action reachability and task logic; training a gate does not solve the identifiability failure.

**2. Adaptive information budget.** For a horizon bounded by `N` acquisitions, the KL chain rule gives

`KL(P0^pi || P1^pi) = E_0 sum_t KL(p0(Y_t|H_t,a_t) || p1(Y_t|H_t,a_t))`.

Actions cancel because the policy uses the same public history. Conditional likelihoods must marginalize persistent ghost/deformation states correctly. Consequently, Pinsker gives the (possibly loose) ceiling

`P(correct completion) <= [1 + min(1, sqrt(KL(P0^pi||P1^pi)/2))]/2`.

If every affordable view has tiny conditional KL, no amount of policy capacity can produce a large discovery gain. Conversely, information outside the affordable view set is irrelevant to this task. Evaluate this on simple two-graph test pairs, not by pretending a finite Monte Carlo estimate is a rigorous upper bound on an unrestricted continuous model.

**3. Probe feasibility cap.** For a proposed physical probe `a`, let `c_reach(a)` be its actual cost and `C_min(h,a)` an optimistic lower bound on fresh target-coverage cost in hidden hypothesis `h` after it. Under a scalar remaining budget `B`, any policy that starts with that probe satisfies

`P(completion | start a) <= sum_h b(h) * 1[c_reach(a) + C_min(h,a) <= B]`.

Use vector resource feasibility for multiple hard caps. This is an optimistic cap because it grants correct topology/identity for free after the probe. It exposes diagnostic actions that consume the ability to finish; entropy/Fisher improvement alone cannot overcome that cap. It is an ordinary feasibility bound adapted to the fresh-coverage task, not a novel information criterion.

Together these tests distinguish three failure sources: the sensor cannot identify the graph, information is reachable but leaves insufficient continuation resources, or the approximate policy fails despite usable information and feasible completion. This is more informative than repeatedly modifying an advantage gate.

## Primary literature and claim boundaries

- [Pathak et al., DA-BSP, 2016](https://arxiv.org/abs/1606.05124) already plans under data-association ambiguity using mixture beliefs and expected future observations. Joint association and active disambiguation are not new here.
- [Shienman and Indelman, D2A-BSP, ICRA 2022](https://arxiv.org/abs/2202.04954) reduces hypothesis cost and derives performance bounds under computational constraints. A claim about distilling graph hypotheses with guarantees must compare against this work; distinguish a sensing budget from its computation/hypothesis budget.
- [Ranganathan and Dellaert, Bayesian partition models for topological mapping](https://ananth.in/Publications_files/Ranganathan04tr-b.pdf) represents uncertainty over topologies via measurement partitions and samples the posterior. Latent graph inference and correspondence uncertainty are established ideas.
- [Pleines et al., recurrent PPO, 2022](https://arxiv.org/abs/2205.11104) documents sequence arrangement, recurrent-state initialization and masking issues and studies generalization. A recurrent comparator requires an actual correct sequential training implementation.
- [Jiang et al., autonomous tubular screening](https://arxiv.org/abs/2011.00099) uses U-Net segmentation and online ultrasound feedback for tubular tracking and orientation. New work must go beyond tracking a selected tube.
- [Morales et al., BIFURC, IROS 2024](https://arxiv.org/abs/2409.06817) identifies vessel bifurcations for cannulation, with phantom and live-animal evaluation. Branch recognition alone is not an unoccupied application contribution.
- [Al-Zogbi et al., 2025](https://arxiv.org/abs/2503.06795) reports autonomous scanning of bifurcated femoral arteries using patient-derived phantoms and video segmentation. New synthetic results should be framed against this level of application evidence.

The claim supported by a positive first run would be narrow: in the specified parametric simulator, under stated model and motion assumptions, a frozen public-history policy improved correct graph-conditioned fresh inspection at the measured resources. It would not yet establish validated U-Net error physics, continuous contact/control safety, in-vivo performance, or a new general solution to belief-space planning.

# ProDMP execution with the static artery–vein decision test

Theory and proposed estimand, 2026-11-13. This document defines what the ProDMP version can test. It does not report its as-yet-unreviewed execution results.

## 1. What the preceding result established

At reviewed commit `125f2ec`, correct accepted completion was 75.71597304% for live stochastic MPC, 81.84842169% for B, and 81.84589205% for a fixed central first near view followed by the same live MPC. Thus B improved on that MPC forecast class but exceeded the strongest fixed-first control by only 0.00252964 percentage points, while adding 2.91797382 mm TCP travel. The useful mechanism was an optional near-frame retry; difficult artery–vein identity reasoning was not demonstrated.

Replacing archived trajectories with online ProDMP synthesis answers a necessary execution question: can these complete policies generate and check their actual motion from the current state? It does not, by itself, reverse the fixed-first result. The new comparison must separately identify the value of the high-level B decision and the capability of the shared motion generator.

## 2. The smallest shared execution contract

Keep the same registered static artery and vein, nine physical target poses at sites 3/4/5, anonymous geometry likelihood, common prescan, 0.87 joint-pair threshold, and two/three-new-image task. Do not introduce flow, pulsatility, forces, new anatomy, or learned high-level policies.

The currently selected implementation uses the existing `LearnedProDMP.fit_default(seed=20261112)` model, a 1.50 s acquisition slot including a 0.21 s stationary dwell, and the same archived IK witnesses as terminal joint configurations. Both vessels remain at the static 15-degree pose. The original 63 directed command pairs remain the declared admissible command class, but their archived trajectory samples, costs, and feasibility verdicts must not substitute for new ProDMP synthesis and checks. This is a finite endpoint/command class with newly synthesized continuous paths, not unrestricted continuous pose planning.

At each real decision, every controller receives the same kind of public state:

```text
S_k = (b_k, q_k, qdot_k, t_k, r_k, M_k),                       (1)
```

where `b_k` is the anatomical/registration posterior, `q_k` includes the rail and robot joints, `qdot_k` is measured or explicitly simulated velocity, `t_k` is the actual clock, `r_k` is remaining resource, and `M_k` contains fresh task-image records and association evidence. Begin the next primitive at this actual state. Do not silently substitute the previous intended goal for a residual exit error.

The common generator `Gamma` maps a physical command and the current robot state to a new reference:

```text
(reference, synthesis_record) = Gamma(q_k,qdot_k,a,settings).  (2)
```

Run the same joint-limit, speed, self-collision, endpoint, and budget checks for all arms. Execute the full selected path, including dwell; record actual states and the exit; acquire at the realized imaging pose; update; replan. A synthesis failure or attempted path failure remains in the assigned-case denominator. Charge consumed motion, captures, and planning effort. No archived-path fallback is permitted.

If simulated robot state is assigned exactly to each reference sample, call this ideal kinematic execution with sampled checks. It verifies continuous trajectory construction and its geometric constraints, but it does not establish feedback-tracking robustness. If a tracker is simulated, its integrated state, rather than the reference, supplies the next start and image pose. No force model is required for this distinction.

## 3. ProDMP derivation and what is actually optimized

ProDMP represents the solution of a movement-primitive differential equation using position/velocity basis functions, enabling conditioning of a trajectory distribution. This construction is established in [Li et al., ProDMP, IEEE RA-L 2023](https://doi.org/10.1109/LRA.2023.3248443). The following is an explicit critically damped normalization suitable for the present derivation, not a claim that every listed parameter is optimized in this experiment.

For movement duration `T_m=1.29 s`, phase `s=(t-t0)/T_m`, one coordinate `z`, endpoint attractor `g`, and forcing basis `phi_j`, write

```text
z''(s) + 2 lambda z'(s) + lambda^2 [z(s)-g] = phi(s)^T w.
z(0)=z0,                    z'(0)=T_m zdot0.                 (3)
```

The solution is affine in the weights:

```text
z(s) = h(s;z0,zdot0,g,T_m) + B(s)w,
h(s) = g + exp(-lambda s)
             [(1+lambda s)(z0-g) + s T_m zdot0],
B_j(s) = integral_0^s (s-u) exp[-lambda(s-u)] phi_j(u) du,
B'_j(s)= integral_0^s [1-lambda(s-u)] exp[-lambda(s-u)] phi_j(u) du.
                                                               (4)
```

Stack coordinates to obtain the joint or chart trajectory. Time derivatives are `zdot=z'/T_m` and `zddot=z''/T_m^2`. Exact finite-time endpoint/rest constraints require

```text
Cw=d,
C = [B(1); B'(1)],
d = [z_goal-h(1); T_m zdot_goal-h'(1)].                       (5)
```

Asymptotic goal attraction alone does not impose an exact endpoint at 1.29 s. Solve these constraints and report the residual. Appending a stationary dwell after a nonzero terminal velocity creates a discontinuity; verify terminal velocity before appending it. Reconditioning is also required when duration changes with nonzero starting velocity: the homogeneous solution contains `T_m zdot0`, so simple rescaling is not generally enough.

If the weight prior is Gaussian, conditioning on (5) gives, when the constraints are consistent,

```text
mu_c = mu + Sigma C^T (C Sigma C^T)^dagger (d-C mu),
Sigma_c = Sigma - Sigma C^T (C Sigma C^T)^dagger C Sigma.       (6)
```

A constrained least-squares/nullspace solution is another valid deterministic construction. Neither solution automatically enforces joint limits or collision avoidance throughout the path; those constraints still need checking. The trajectory covariance is `B(s) Sigma_c B(v)^T`. A fitted covariance describes demonstration variation, not automatically physical tracking error or calibrated collision risk.

The existing local `LearnedProDMP` is a learned **two-dimensional endpoint-normalized tangential-chart** primitive fitted to analytic demonstrations. Its longitudinal and cross-track shapes are meaningful only through an explicitly stated map into robot motion. Record that map:

- If the complete 2-D chart is mapped to a physical path and then to a continuous robot configuration, document the chart, orientation rule, and IK continuity.
- If only longitudinal mean progress `rho(s)` is used in `q(s)=q0+rho(s)(qg-q0)`, identify it as a learned ProDMP-derived scalar progress profile for joint interpolation. It is not full 2-D shape use, weight search, or optimization over IK branches.
- If the fixed fitted mean is reused after start/goal conditioning, this is online conditional synthesis. A probability distribution can be fitted without being sampled or optimized online.

To claim **online optimization over ProDMP parameters**, the planner must actually compare permissible `theta=(w,g,T_m,...)`, define their shared feasible set, evaluate their physical/task consequences, and choose among them. All controls need the same parameter set, optimization budget, initialization, and failure handling. The present minimum experiment does not require that larger claim. Fixed endpoint witnesses and duration plus a common fitted mean constitute a restricted generator `Gamma`, even when every trajectory is synthesized online.

**Implementation mapping now inspected.** The current generator uses both learned material-chart coordinates for advancing moves. It maps the learned chart through bilinear interpolation of the static surface strip, blends the small initial FK alignment residual, interpolates endpoint rotation using longitudinal progress, and interpolates rail position with that progress. At a fixed chart location it uses progress to schedule rotation. It generates forward and backward fixed-rail IK sequences from the two endpoint witnesses, blends those joint seeds using progress, and projects each blended seed to the original Cartesian reference. All resulting samples are checked. This construction uses the learned lateral excursion; it is not merely scalar joint interpolation. The bidirectional IK avoids the previously observed endpoint nullspace jump, but its adequacy is established by the common checks, not by a general branch-continuity theorem.

The generator accepts only the declared rest-state configurations and closes exactly to the fixed endpoint witness before dwell. Thus the present model is conditional synthesis at finitely many rest states, not recovery from arbitrary tracking errors or nonzero measured velocities. Equations (3)–(6) describe the primitive's general conditioning structure; after nonlinear surface mapping and sampled IK, exact continuous joint velocity/acceleration continuity is not implied by chart endpoint constraints. The implementation checks configuration closure and sampled rates and reports ideal kinematics.

## 4. Continuous execution, resources, and the observation kernel

Let `xi` be the realized trajectory and `e` the realized exposure pose. For a command `u` (a target `a`, or `(a,theta)` only if parameters are actually searched), use the joint transition kernel

```text
K(dz',dc,dy | z,h,u)
  = P_exec(dz',dc | z,h,u) L(dy | h,e(z')).                  (7)
```

`z` includes the physical state and clock; `h` is hidden static registration/anatomy nuisance. Execution failure is an explicit transition, not a missing trial. If execution is independent of `h`, its measured outcome does not itself reveal anatomy. If exposure pose is uncertain, marginalize that uncertainty in the same likelihood used by every arm.

The realized resource vector is obtained from the path, for example

```text
C_TCP = integral ||J_translation(q) qdot|| dt,
C_rot = integral ||omega(q,qdot)|| dt,
C_rail= integral |rail_dot| dt,
C_time= actual movement + settling + acquisition time,
C_img = number of actual captures.                           (8)
```

Report rotation in the declared units, plus planning time separately and end-to-end time if making a speed claim. A common 1.50 s slot or common cap is not proof of equal realized travel or computation. Sampling these integrals and constraints is a sampled validation, not a continuous-time collision certificate unless an additional bound covers between-sample behavior.

Crucially, evaluate `L` at the realized endpoint frame. A nominal sensor row remains valid only when the execution contract and its tolerance justify the same imaging pose. Otherwise resect/reproject the registered geometry at the actual pose, marginalize pose error, or apply the predeclared invalid-acquisition rule. Never capture a nominal image after missing the nominal pose.

Condition on the actual executed commands and measured exits. For two new packets, a schematic posterior is

```text
b_2(h) proportional to b_0(h)
       K(z_1,c_1,y_1 | z_0,h,u_1)
       K(z_2,c_2,y_2 | z_1,h,u_2(public history)).             (9)
```

Terms independent of `h` cancel. The adaptive choice of `u_2` adds no extra likelihood factor. Persistent pose/calibration nuisance belongs in `h`; do not marginalize it independently for each frame.

For the fresh pair, retain the near packet's per-state association probabilities `r_N(h,i)`. After the far packet, the joint slot probability is

```text
P(near i and far j are arterial | history)
 = sum_h b_before_far(h) r_N(h,i) L_F(y_F,arterial_slot=j | h,e_F)
   / sum_h b_before_far(h) L_F(y_F | h,e_F).                 (10)
```

Use the same maximum joint probability, .87 acceptance threshold, and truth scoring for every arm. Do not multiply marginal near/far probabilities. Retain the previous latest-near eligibility rule for the primary motion comparison, but keep its limitation visible: a repeat replaces the eligible near packet. A static task that permits either earlier qualifying near packet is a different, stronger task-state model and must not be introduced after examining the ProDMP outcomes.

## 5. Coupled MPC/B decision formula

Let `Phi(S)` be the shared optimal remaining far-view-and-declaration rule, including synthesis, execution, actual exposure, costs, failure, and equation (10). It is available only after the required new near image. Both controllers may select the far view conditional on the actual near packet, including when no second diagnostic is taken.

For root command `u`, let `Omega=(z',c,y)` be its public realized outcome and `Q(S,u,Omega,v)` the value of the common tail with next option `v`. The options include proceeding through `Phi`, a second near acquisition followed by `Phi`, and abstention. A fixed diagnostic option specifies a public target; its low-level ProDMP is still regenerated from `z'` by the shared generator. It must not use a stale predicted starting configuration. If that option becomes infeasible, the same predeclared failure/guard rule is included in `Q`.

```text
F(S,u) = sup_v E_Omega[ Q(S,u,Omega,v) ],
G(S,u) = E_Omega[ sup_v Q(S,u,Omega,v) ],
u_M    in argmax_u F(S,u),
u_B    in argmax_u G(S,u).                                   (11)
```

The immediate-finish option already contains the common conditional far-view optimization. B changes anticipation of the optional second diagnostic; it must not gain exclusive access to a better terminal solver.

After the real root outcome, MPC replans. Its executed model value is therefore `G(S,u_M)`, and the incremental B capacity is

```text
Delta_B(S) = sup_u G(S,u) - G(S,u_M).                         (12)
```

`G-F` is not this gain. It quantifies the value of conditional continuation for a specified root; MPC can realize that same continuation once the packet arrives. If the root commands and realized execution are equal, shared observations, exits, replanning, and tie rules imply identical subsequent policies for this two-diagnostic horizon.

For a general remaining horizon, the belief-state recursion is

```text
V_k(S) = max{Phi(S), sup_u E_K[V_(k-1)(Update(S,Omega))]},
V_0(S) = Phi(S).                                             (13)
```

All resource debits and failure states are inside the update. For genuinely continuous path/endpoint uncertainty the expectation is an integral and the action optimization may be continuous. A finite candidate enumeration is exact only over those candidates and the declared discrete model. A sampled continuous search is not an exact upper bound over all ProDMP parameters.

Within a correct common model, exact B includes the MPC policy and consequently has no lower optimal expected value. This is a policy-inclusion statement, not a guarantee about individual trials, approximate search, misspecified likelihoods, or real tracking. Nor does it guarantee that replacing the old executor with ProDMP improves success: the feasible path sets, costs, and actual exposure errors may change in either direction. If ProDMP reaches the same image poses with the same feasible high-level action sets, resources remain nonbinding, and the sensor is unchanged, motion synthesis alone cannot create new identity information or a new sensing advantage.

## 6. Nested ablation and attribution

Use these three primary arms with the **same ProDMP generator and actual execution**:

| Arm | Changed high-level component | Shared components |
|---|---|---|
| ProDMP + live MPC | Fixed-second forecast, actual replanning | All motion, likelihood, posterior, terminal decisions, checks and caps |
| ProDMP + B | Anticipates second-diagnostic branching | Same |
| ProDMP + fixed first near + live tail | First near view fixed independently of its prescan packet | Same |

Evaluate the three already declared fixed near-view types or freeze the earlier central winner prospectively. A first-view control is not a frozen tail: it keeps the ordinary posterior-conditioned retry and far choice. The primary B claim requires a useful gain over **both** live MPC and this strong control at comparable realized resources, with the same acceptance rule and reported wrong declarations.

A no-third-image control separately measures the value of permitting retry. A common MPC option containing the simple public rule “retry near if needed” is a useful stronger forecast control, but the fixed-first arm already answers the central practical question. Do not call a win over a forecast that omits this simple contingency a win over optimal belief-space MPC. A full belief-tree MPC using (13) is B under another name.

These are nested ablations of the high-level decision, with ProDMP held constant. They do not establish that ProDMP itself improves success over another equally capable executor. A causal motion-representation claim would need a separate matched executor contrast with the same high-level controller and task; historical comparison to `125f2ec` is descriptive when feasible sets, checks, or execution assumptions change.

If a motion-representation or interaction claim is needed, let `J_(H,E)` denote the same task metric for high-level policy `H` and executor `E`, where `E=L` is learned ProDMP and `E=E0` is endpoint-only ProDMP. A 2-by-2 comparison defines

```text
Delta_high^L = J_(B,L) - J_(M,L),
Delta_motion^H = J_(H,L) - J_(H,E0),
Interaction = [J_(B,L)-J_(M,L)] - [J_(B,E0)-J_(M,E0)].         (13a)
```

The high-level control could instead be fixed-first plus live tail when the practical B claim is the priority. A B/MPC-by-executor 2-by-2 design still needs the fixed-first control to address the preceding null result; the complete three-controller crossing has six cells. These extra cells are unnecessary for the smallest execution test. Three learned-ProDMP high-level arms plus a matched endpoint-only mechanics audit on identical commands can establish that the representation changes the generated trajectory and its checked resources. That mechanics audit cannot establish improved closed-loop task completion or a policy-by-executor interaction. A 4.91–6.10 mm same-time Cartesian reference difference, for example, proves nontrivial shape use; it is not by itself a task benefit.

For a prospectively strengthened MPC tie rule, define

```text
A_epsilon(S) = {u: F(S,u) >= max_v F(S,v)-epsilon}.           (13b)
```

Choose within this set by a public task-progress preference (a new required near frame), followed by a predeclared resource tie-break. This guarantees forecast regret at most `epsilon` and leaves the final .87 joint acceptance unchanged. It does not bound realized replanning regret by `epsilon`: writing `D(u)=G(u)-F(u)`, that regret also depends on differences in `D`. Freeze `epsilon` from numerical/planning resolution before evaluation; do not tune it on favorable outcomes. A post-primary calculation selecting the best `G` among epsilon-optimal `F` roots is a sensitivity/existence diagnostic, not an executed MPC policy. The fixed-first control is the stronger practical evidence when it already closes the gap.

## 7. Detection ceiling and stopping criterion

Under the retained terminal rule, at most two near attempts and exactly one far attempt can supply the qualifying pair. Intrinsic target detection is independent with `p=.84`. Even with free registration/slot identity, ideal field of view, and free feasible motion,

```text
P(success) <= p [1-(1-p)^2] = .818496.                        (14)
```

Proof: success requires a far detection and at least one of the two possible near detections. Additional geometric uncertainty, an acceptance threshold, or mechanical failure only removes successful outcomes. No independence between mechanical success and sensing is assumed in this bound; multiplication is only for the stipulated intrinsic detections.

The two-image ceiling is `p^2=.7056`; the maximum optional-near-retry increment is 11.2896 percentage points. The earlier fixed-first policy was already only 0.00370795 points below (14). If it remains near that value with ProDMP, no five-point incremental B success gain is possible in this class, regardless of motion-parameter capacity. ProDMP might still improve execution feasibility or resource use; that would be a different, measurable result.

Equation (14) is not a universal three-image bound. If far retry were admitted, an ideal policy that retries whichever required acquisition failed could attain `3p^2-2p^3=.931392`. That permission is outside the current terminal rule and is not needed for the minimum motion validation.

## 8. Exactness, caching, and review gates

The finite-state reduction survives only if the declared model reaches the fixed endpoint states with the declared dwell/velocity contract. If actual exit errors vary, future synthesis and sensing depend on those continuous errors and the old pose-index cost table is insufficient. Either carry that state through planning or state the ideal tracking approximation; do not call the unrestricted execution problem exactly solved.

Runtime caching of deterministic synthesis is acceptable when keyed by the full start configuration/velocity, goal witness, fitted model and weights, duration/dwell, mapping, and check settings. It is a computation optimization, not an archived path fallback. Quantizing measured exits or reusing a path whose start no longer matches changes the model and needs an explicit bound. State whether selected paths are freshly generated every time, generated once at first runtime use, or replayed from a valid cache.

A small uncached replay shared across arms can verify identical choices, path samples, exits, likelihoods, costs, and acceptance to the cached route. It checks caching; it does not replace the main execution ledger. Use equal cache initialization when comparing compute time, so one arm does not inherit another arm's warmed cache. Report synthesis and planning latency alongside physical motion time.

Review in this order:

1. **Execution:** confirm new weights/reference generation, explicit chart mapping, actual start continuity, endpoint/dwell residuals, sampled joint/rate/collision checks, actual exposure pose, and every attempted-path debit. Log all synthesis/tracking failures.
2. **Decision:** reconstruct the two-observation posterior and joint declaration, and compare B with `G(S,u_M)` rather than `F(S,u_M)`. Verify that equal root commands produce equal shared live tails.
3. **Attribution:** compare against fixed first near plus the same live tail, and compare success with actual motion/time/compute resources. A negligible fixed-control gap means no useful independent B contribution.
4. **Conclusion:** a successful path audit supports executable ProDMP-based synthesis in the declared kinematic model. A B win over both matched controls supports a high-level decision contribution in this anatomy/model. A tie supports the simpler controller. Failure of motion generation is an executor limitation; failure to exceed the exact restricted policy ceiling is not repaired by learner scaling.

No positive conclusion is guaranteed in advance. The smallest physically faithful test may confirm that ProDMP executes the task while the earlier conclusion about B remains unchanged.

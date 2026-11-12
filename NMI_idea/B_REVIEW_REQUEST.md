# Request for independent review of Anatomical GPS and active sensing for robotic ultrasound

**Review priority.** Please critique and revise the theoretical construction of B, then decide which other modules are necessary for a coherent T-RO paper. The proposed contribution is a research question. The latest closed-loop comparison did not establish a benefit for B over a strong baseline.

## What I want the robot to do

The robot starts from a broad, correlated prior over an internal vessel or structure bound to a deformable material atlas. The atlas supplies a stable address and a coarse prediction as the body moves. The robot obtains local ultrasound observations during the task, updates the uncertain internal map and current deformation, and acquires the requested view or scan. The setup should also handle brief target loss and ambiguous detections. A complete patient-specific internal scan is unnecessary as an initialization step.

I want to test whether **the value of a probe depends on which continuations will remain executable after it**, and whether this can translate into more fresh, correctly identified images and completed scans at comparable measured cost. A learns transferable local action-to-observation responses. B uses the observation model to plan what to probe, when to proceed, and what can be executed from the resulting state. In the latest test, the common strong baseline already assimilates observations and replans. B differs only in whether it anticipates those future revisions before choosing the first action.

## Current construction of B

Let \(h_t\) be the public evidence history, \(b_t\) the belief over persistent atlas parameters, transient deformation and identity, and \(x_t\) the robot and contact state. An executed probe \(a\) produces \(O=(y,x',\tau)\). Here \(y\) is the new observation, \(x'\) is the actual exit, and \(\tau\) is the elapsed time. A candidate continuation \(u\) must satisfy the shared execution checks. A two-step version is

\[
Q_B(a)=-\mathbb E[c(a,O)]+\mathbb E_O\!
\left[\max_{u\in\mathcal F(x',r-\tau)}
J_{\mathrm{fresh}}\!\left(u\mid b_t^{a,O},x',r-\tau\right)\right].
\]

The strong online MPC baseline optimizes the same terminal objective over a continuation selected before the next observation, with \(\max_u\mathbb E_O\) for the fixed future plan, and replans after each actual observation. Please define the treatment of infeasible branches and the available fallback in both expressions, particularly when the exit itself is uncertain. The expectation/maximum distinction is standard. We need a precise, ultrasound-specific contribution, such as a condition under which material-addressed evidence changes the ranking of checked continuations at an attainable exit. A possible extension is a stopping rule comparing a robust continuation with another probe or recovery. The latest test fixes the number of acquisitions and does not validate such a stopping rule.

## What we found

The latest protocol paired 128 synthetic moving-Y cases across three strategies and two cumulative lateral-stroke limits. All arms used the same 28-frame first sweep and common probe. Success required a **fresh, independently correct target-branch observation plus completion of the checked continuation**. The execution model was sampled fixed-rail kinematics with ideal low-level pose feedback. The observation generator used synthetic segmentation centroids, including misses and clutter.

| Strategy | Correct new image and completed continuation per reserve | Wrong confirmations |
| --- | ---: | ---: |
| Online material map + B | 76/128 | 16/128 |
| Online material map + strong online MPC | 76/128 | 16/128 |
| Frozen material map + B | 70/128 | 14/128 |

- B and MPC selected identical paths and claims, and made the same 16 observation-driven tail changes per condition. The realized costs passed the prespecified equivalence checks. No incremental benefit from anticipating those changes was observed.
- Online versus frozen B gained 4.69 percentage points (paired 95% interval 0 to 9.38), with two additional false confirmations. Both the gain criterion and the false-confirmation criterion failed. In the 96 cases with the daughter visible in the common probe, the corresponding counts were 56, 56 and 51.
- The two stroke limits permit different tails after a lateral first action. All methods took the centered first action, so both limits offered the same available tails in all realized trials. The bounded fixed-pose native preflight did not yield the required usable narrow exit. The main run supplies simulation evidence. Moving-phantom B-mode and physical contact are unverified.
- Earlier controlled two-window cases and fixed-pose native replays illustrate how new evidence or an exit can change a continuation. Those narrower mechanism results do not establish superiority over replanning MPC in the moving-Y task. Online material adaptation and a transient state have separate prediction evidence, which should be judged on those endpoints.

## Questions for the reviewer

1. **Revise B.** Propose the smallest defensible state, observation model, executable continuation set and objective or stopping rule. Identify the novel claim beyond standard belief-space MPC and a concrete counterexample to a strong replanning baseline. State assumptions for each guarantee. Say whether the present result suggests an inadequate theory, an inactive test intervention, or both, and what would discriminate those explanations.
2. **Assess module necessity.** For each component below, classify it as essential to the claim, shared implementation, optional ablation, or removable. Give the simplest fair replacement and the evidence that would change your classification.
3. **Audit the evidence and advise on T-RO claims.** Check target identity, the fresh-image criterion, access to hidden labels, feasibility and measured costs, and pairing by case. Identify the smallest meaningful validation and the claims that can be supported now. Please separate simulated mechanism, matched closed-loop benefit, and physical evidence.

| Component | Question and evidence |
| --- | --- |
| Harmonic material atlas and broad internal prior | Is persistent addressing necessary for reuse after motion, compared with a surface anchor or a local frame? Specify which identity claim and transport error must be tested. [Geometry and local response](experiments/ab_theory_recheck_20261110/FINAL_REPORT.txt) |
| Online material update and latent transient state | Separate map update, fast transient, association and pose sensing. Check whether the transient improves future predictions under matched observations and whether map updates improve the actual task at controlled false-confirmation risk. [Dynamic and matched evidence](experiments/ab_knee_matched_20261112/PAPER_CORE.txt) · [Map update evidence](experiments/ab_map_recheck_20261112/REPORT.txt) · [Revisit prediction data](experiments/ab_learned_acquisition_20261112/material/RESULTS.json) |
| Local response model and B | Compare geometry plus online residual/association with a learned model on held-out contexts. Give the conditions under which a probe has a decision-relevant observation and a reachable follow-up. [Study design and negative result](experiments/ab_tro_y_continuation_20261112/REPORT.md) |
| Equivariance and execution frames | Separate material identity, coordinate-change equivariance, and transfer across deformation. Compare explicit local contact frames with VN/e3nn only where learning adds independent value. [Transport and response results](experiments/ab_theory_recheck_20261110/transport_response_bridge/RESULTS.json) |
| ProDMP, dIRD and motion execution | Assess ProDMP as a short-segment representation against a simpler reference. Assess dIRD as a candidate prior or filter separately from actual state-dependent checks. [ProDMP and recovery](experiments/ab_prodmp_rl_20261112/REPORT.txt) · [Native exit evidence](experiments/ab_native_exit_effect_20261111/REPORT.txt) |
| CQL/RL and Fisher information | Check whether gains survive equal realized path and task budgets. Decide if learned selection or Fisher-directed data generation contributes independently to correct completion. [Matched RL](experiments/ab_knee_matched_20261112/PAPER_CORE.txt) · [Fisher and demonstrations](experiments/ab_fisher_prodmp_20261112/FISHER_PRODMP_REPORT.txt) |

## Where to inspect and reproduce

Repository **NingMeng0903/RealUS_playground**, under **NMI_idea/**. The review entry is this file. The main study is **NMI_idea/experiments/ab_tro_y_continuation_20261112/**.

- Manuscript starting point [V9.3](ideas/Anatomical_GPS_V9.3.tex), with [V6.1 posterior task risk](ideas/Anatomical_GPS_V6.1.tex), [V9.1 materials and geometry](ideas/Anatomical_GPS_V9.1.tex), and [V9.4 recovery and branches](ideas/Anatomical_GPS_V9.4.tex). [Earlier A/B research overview](AB_RESEARCH_OVERVIEW.md)
- [Frozen protocol](experiments/ab_tro_y_continuation_20261112/PROTOCOL.md) · [Report](experiments/ab_tro_y_continuation_20261112/REPORT.md) · [Astra max model audit](experiments/ab_tro_y_continuation_20261112/ASTRA_REVIEW.md)
- [128 common prefixes and probes](experiments/ab_tro_y_continuation_20261112/holdout/COMMON.jsonl.xz) · [768 complete rollout records](experiments/ab_tro_y_continuation_20261112/holdout/RUNS.jsonl.xz) · [Paired statistics](experiments/ab_tro_y_continuation_20261112/holdout/VERIFIED_SUMMARY.json) · [Offline 3D comparison](experiments/ab_tro_y_continuation_20261112/holdout/TRO_CONTINUATION_3D.html)
- [Full review instructions](experiments/ab_tro_y_continuation_20261112/REVIEW_REPRODUCE.md) · [Bounded native preflight](experiments/ab_tro_y_continuation_20261112/native_preflight/bounded/WITNESSES.json)

Please return a revised formulation of B, a module-by-module verdict, and the strongest T-RO claim you would allow on this evidence. The internal Astra max audit was an agent check. This request is for a separate review.

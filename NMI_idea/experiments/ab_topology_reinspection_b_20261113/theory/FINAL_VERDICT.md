# Final theory verdict

2026-11-13. Independently recomputed 11,264 stored policy-case records across the nominal and withheld-shape cohorts. Each cohort contains 256 anatomical cases, including 128 Y cases; repeated training seeds are not additional independent anatomies. Checked common prefixes/truth across arms and seeds, selected fresh frame identity, two-plane ordering, endpoint success, resource caps, and source hashes. Current execution sources match the recorded freeze. Details and independent paired bootstrap calculations are in `FINAL_AUDIT.json`; `final_audit.py` reproduces this record audit. This check does not regenerate the acoustic world or certify continuous robot execution.

## Decision

**Retain all-return belief updating as a supported component. Do not claim a reliable independent B advantage or adopt the learned selector as the principal method from this experiment.** The new task exposes consequential representation loss in nearest-return compression. It does not establish a new general topology-search algorithm.

| Policy | Nominal: all / Y | Withheld shape: all / Y |
|---|---:|---:|
| Live two-step MPC, all returns | 165/256 / 51/128 | 147/256 / 38/128 |
| Complete greedy-tail rollout | 176/256 / 56/128 | 141/256 / 34/128 |
| Witness policy | 174/256 / 58/128 | 140/256 / 31/128 |
| Greedy, all returns | 172/256 / 58/128 | 147/256 / 38/128 |
| Fisher-augmented greedy | 170/256 / 56/128 | 149/256 / 39/128 |
| Nearest-return MPC | 119/256 / 1/128 | 107/256 / 2/128 |

All-return MPC increases Y completion over nearest-return MPC by 39.06 percentage points nominally (50 paired wins, no losses; independent 95% interval [30.47, 47.66]) and 28.13 points on withheld shapes (37 wins, one loss; interval [20.31, 35.94]). This is a strong closed-loop representation result for this sensor/task. It is not a risk- and cost-matched dominance result: affirmative errors are 19 versus 0 nominally and 19 versus 1 on withheld shapes; all-return MPC also adds about 6.17/4.27 mm mean TCP travel and 27.41/25.65 mm rail travel, while finishing sooner on average.

The nominal pooled rollout gain over MPC is +4.30 points, with 95% interval [−0.39, 9.38]; the witness gain is +3.52 points, interval [−0.39, 7.42]. Both reverse direction on withheld shapes and fail the declared no-worse realized-cost criterion. These data do not demonstrate that deeper task search, witness selection or Fisher improves on equally equipped MPC. The greedy results also prevent attributing the all-return benefit specifically to MPC foresight.

Recurrent PPO completes zero Y cases for every training seed in both cohorts. The same-data no-GRU PPO also completes zero. CQL achieves 2/0/1 nominal Y completions across the three seeds and zero on withheld shapes; recurrent pure cloning achieves 5/1/3 and 3/1/1. This is a reproducible failure of these trained policies and budgets, not a theorem against RL. Lower motion or fewer affirmative errors obtained by declining positive inspection do not meet the user's objective.

The same-tape hindsight search finding a true fresh pair in all 48 development Y cases demonstrates physical/sensor opportunity on those tapes. It does not show that a public-history policy can identify the successful route in advance, certify its identity afterward, or repeat its result on new tapes. It is not attainable B performance and not a sufficient reason for further learner scaling.

## Claim that survives

“In a finite-family centroid simulator with uncertain offshoot existence, location and candidate identity, preserving all returns substantially improves correctly identified two-plane fresh offshoot inspection after a common prescan relative to nearest-return compression.” Attach the observed false-completion and movement tradeoff.

Do not expand this to general anatomical graph recovery, a measured benefit of topology search itself, real U-Net performance, full daughter coverage, or validated clinical/continuous-contact execution. The simulation implies connectivity through its centerline/visibility family; it does not render and test an independently observable lumen connection. No externally specified choice between two daughter identities is present. DA-BSP and prior topological mapping already cover the general belief-planning ideas.

## One bounded next experiment: measure the junction bridge

The next change should add an observable that can reveal physical connectivity, not another decision layer on the same five-way family classifier. Use the existing image/robot simulator with its segmentation front end and calibrate image/segmentation errors against available ultrasound clips.

1. **Paired physical alternatives.** Construct joined-Y and disconnected-overpass/parallel tube volumes that match the existing transverse prescan observations as closely as the renderer permits. Their difference lies in the junction region between the two candidate patches. Sample continuous geometry and random internal object labels; keep geometry labels inaccessible to policies. Verify the old observations' ambiguity empirically rather than declaring it by construction alone.
2. **A public diagnostic action.** From the posterior junction region, generate a mechanically checked short longitudinal/oblique sweep of at most five overlapping image planes connecting the seeded trunk patch to the candidate offshoot patch. Both policies see the same public poses and masks. A second oblique view may replace a plane when needed to break projection overlap. Only registered image evidence may support a connection; no simulator “connected” bit or hidden edge ID enters the observation.
3. **A new likelihood.** Retain segmentation probability masks, boundaries and pose/registration uncertainty across the overlapping patches. Compare a continuous lumen bridge with two separated lumen surfaces. Marginalize associations and registration instead of joining nearest centroids. A single 2D overlap is insufficient. This measurement is new relative to the current centroid proxy, but its general active-perception formulation is established.
4. **Actual task continuation.** After the diagnostic sweep, perform two new target-plane acquisitions on the publicly specified daughter and evaluate identity plus physical connection. Include a public goal such as the trunk-connected daughter entering a specified distal region. The five-plane diagnostic plus two inspection planes must fit a prechecked shared budget; failed motion feasibility stops the design before outcome testing. Charge every image plane and actual motion/planning time.

Use one bounded development set of 64 geometry pairs to freeze the renderer/error calibration, bridge likelihood, action menu, public planner and budgets, then one new confirmation set of 256 pairs. Treat the two counterfactual worlds from a geometry as a cluster. Do not retrain the segmentation model or change acceptance thresholds after confirmation is opened.

First use identical recorded diagnostic sweeps to compare full mask/bridge evidence with centroid compression of those **same images**. Require a positive paired improvement in held-out connection log score and correct connection decisions; report false joins. If the new observation cannot distinguish the paired physical alternatives, stop before policy learning.

Then compare four frozen complete policies: MPC with centroid compression; the same MPC with bridge evidence; a fixed diagnostic sweep followed by the same bridge-aware MPC; and a public-history contingent planner with the same bridge evidence, candidate moves and resources. All must execute the requested daughter inspection. This separates observation value, an always-useful probe, and state-dependent continuation value.

**Criterion for another learner:** the public contingent planner must improve completed correct target inspection over both equally equipped live MPC and the fixed-probe control, with a positive lower 95% case-clustered interval, no increase in affirmative errors beyond the predeclared safety margin, and the predeclared no-worse realized-resource margins. Its choices must be made from public evidence before the evaluation tape unfolds. Use the existing 2% harmful-discordance criterion if retaining that research claim; with 256 independent geometry pairs, even zero harmful pairs only gives about a 1.43% one-sided 97.5% upper bound. No positive criterion may be inferred from hindsight route selection.

If only the observation contrast passes, the contribution is bridge-aware sensing and MPC. If the contingent planner passes but is too slow for the enforced decision interval, train one fixed-budget distillation learner to reproduce it and measure retained task performance versus saved latency. If neither passes, stop this B route. This experiment tests a new physical source of connectivity evidence and a specific reason for learning without committing to unlimited policy tuning.

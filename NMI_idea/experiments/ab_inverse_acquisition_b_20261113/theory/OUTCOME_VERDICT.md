# Theory verdict: active tilt screen and aperture preflight

The bounded tilt continuation predictor must stop under the frozen protocol. The new menu was actively used, but it did not establish the required repeat-validated, no-worse-cost complete-task advantage over MPC with the same menu. The generator itself did not improve the corresponding coarse-angle MPC. This is a negative result for this mechanism and policy class, not a theorem that every inverse acquisition policy is impossible.

## Frozen screen

The source is `data/screen/summary.json`: 32 new latent cases per task, 24 suffix tapes per case, and paired case-level intervals. Repeated suffix tapes do not count as new anatomical cases. The selector uses private outcomes from one half of the tapes to select a continuation evaluated on the other half. It remains an achieved label-assisted diagnostic, not an online policy or a population upper bound.

| Contrast | Daughter completion change, points | Single-trunk change, points |
| --- | ---: | ---: |
| Cost-screened split-repeat continuation versus inverse-menu MPC | +0.91 [−0.26, +2.34] | +0.39 [−0.39, +1.30] |
| Hard-cap-only split-repeat continuation versus inverse-menu MPC | +4.04 [+0.78, +7.81] | +0.39 [−1.04, +1.82] |
| Inverse-menu MPC versus coarse-diagonal MPC | −0.65 [−2.73, +1.30] | −0.39 [−0.91, +0.13] |
| Inverse-menu MPC versus claim-only menu MPC | 0.00 [−3.26, +4.04] | −0.13 [−0.65, +0.26] |

The first row passes the realized no-worse cost test in both tasks, but fails the frozen three-point efficacy screen and does not improve on the fixed-progress control. The positive daughter hard-cap effect does not satisfy the actual cost criterion: mean added rotation is 2.90 degrees, and the simultaneous upper excess-cost limits are 2.76 mm TCP, 5.60 degrees, and 0.219 seconds, all above their respective margins. It must not be called a same-cost success.

Wrong-identity changes for the primary cost-screened diagnostic are −0.13 points in each task. The fixed-progress trunk control gains 1.69 completion points and adds 1.69 points of any-wrong cases. Its uncertain efficacy and risk tradeoff are not a B mechanism. This small cohort does not certify the historical 2% harmful-discordance requirement.

## Why this is informative

This was not another inactive gate. Inverse-menu MPC used intermediate tilts in 590/768 daughter suffix episodes and 490/768 trunk episodes. It made 1,113 and 799 new-angle captures, respectively. Every one of the 32 cases in each task had at least one repeat using a new tilt. Its public menu differed from the claim-only menu in 30/32 daughter cases and 32/32 trunk cases.

The decision-information score was positive in all 384 daughter scored poses and 383/384 trunk poses. All these scores had three mechanically reachable legacy reference views. The mean score was 0.00121 signed-identity-utility units for daughter and 0.00080 for trunk; the respective maximums were 0.01016 and 0.00430. These are small surrogate values, not complete-scan probabilities or bounds on full-task improvement. They show that the implementation did expose model-level decision distinctions. Those distinctions did not translate into the desired final outcome at matched practical cost.

The hierarchy of contrasts also matters. Expanding legacy-angle graph connectivity produced an uncertain daughter gain while substantially increasing rotation. Giving the controller new tilts produced no further completion gain. Giving a label-assisted selector the chance to override that enriched MPC helped mainly when it was allowed more movement. A larger network, another admission threshold, or rebranding the same two-stage objective as RL does not address these observations.

## Prior-only aperture check

`APERTURE_PREFLIGHT.json` and `aperture_preflight.py` test an alternative mechanism without opening a new outcome cohort or constructing new robot motions. They use all 256 public model particles at 1.5, 3.0, and 4.5 seconds and the existing projected lateral contacts at 0, +8, +16, and +24 mm. Each site's center-view ghost is reconstructed as one world-fixed anchor and projected to the other contacts; center raw detection probabilities are retained. This removes fresh view-ID ghost regimes from the comparison.

The check confirms the translation invariance of the current centered-bin observation model away from aperture clipping. With a fixed frame and unbounded aperture, shifting both means and nominal query coordinates changes component probabilities by at most 4.3e−15. The actual +8 and +16 mm surface frames rotate by less than 0.4 degrees and produce very small kernel changes.

The +24 mm contact creates a larger aperture change, but it does not establish a useful next experiment:

- The archived surface-projection feasibility flag is false at daughter site 4 and trunk sites 6 and 7, even though archived robot endpoint IK was found. Robot endpoint reachability is not a valid skin-contact certificate.
- At the one surface-supported daughter final site, +24 mm reduces the expected non-target return count, but improves the probability that the selected centroid is the target by only 0.15–0.29 points across the three clocks. Coarse-prior correct-accept changes are small and vary in sign.
- Trunk +24 mm lowers selected-target probability by roughly 1.9–3.2 points across sites and clocks. Its target fraction among expected detected returns also decreases.
- Endpoint chord-distance lower bounds fit under 65 mm, but they do not certify sampled paths, rail use, or realized episode cost.

This prior-only calculation is not a conditional-history oracle or a population performance bound. It is sufficient to reject a new offset-bank construction as the immediate next experiment: there is no credible shared daughter/trunk aperture-selection mechanism in these existing contacts.

## Temporal preflight: distinct mechanism, insufficient present signal

The existing world already contains time-varying lag and temporary losses shared across viewing angles. Fixed-time spatial changes cannot bypass a target that is absent in every view at that clock. A posterior over temporal state could, in principle, predict when the same required acquisition will become useful. This is a different mechanism from finer tilt selection and does not require inventing a new acoustic phenomenon.

The bounded prior-only temporal preflight has now run; see `temporal_preflight/SUMMARY.json` and `theory/TEMPORAL_VERDICT.md`. It uses the existing checked 1.5-second motion followed by a stationary hold of 0, 0.5, or 1.0 seconds before the same needed-site acquisition. It does not assume a faster robot trajectory or a longer sensor exposure. Six acquisitions and nine seconds remain separate caps. Public histories come from 128 model draws per task; the two checkpoints follow one and two shared MPC captures. Both keep all 128 histories in their denominators. Selection uses public joint-correct claim mass, the shared nominal suffix potential, and joint-cap completion witnesses.

The preflight provides too little modeled benefit to justify a new full timing cohort. Nominal-surrogate increments over no hold are 0.388 and 0.559 percentage-point units for daughter and 0.0389 and 0.0344 for trunk, at the two checkpoints. Mean selected extra holds are 0.594/0.445 seconds for daughter and 0.090/0.363 seconds for trunk. These are model-surrogate differences, not completed-scan gains. They do not establish matched elapsed cost or a benefit over time-aware MPC. The small signal, additional delay, and weak trunk signal give a no-go for expanding this timing design now.

Planning delay still has to enter the physical clock or be covered by an enforced, measured common timing allocation before any deployable temporal claim. The preflight excludes it and explicitly records that limitation. The result concerns retiming the current MPC's needed-site command with these three holds; it does not prove that every timing policy is ineffective. The earlier speed-derated 1.5/2.5-second graph sensitivity also did not establish a timing-specific continuation benefit.

Stop trying to obtain a general B contribution by further selector tuning on this centroid renderer. This does not redirect the objective to a different contribution. It identifies what a redesigned B now needs: independently supported observables that distinguish the competing anatomical explanations—such as image-derived wall/axis continuity or a calibrated flow cue—together with an action that changes their visibility. Such cues must be rendered or measured from physical structure and available to both policies; they must not be injected from private target labels. The failed all-return centroid study is not evidence that these richer cues have already been tested, and neither is it evidence that they will work.

## Precise conditions for restarting B

Keep inverse observation-action synthesis as the B research main line. The current evidence does not support claiming that contribution yet. A restart should satisfy the following sequence:

1. **Establish an action-dependent observation response.** Identify a measurable feature whose distribution changes with an executable action and helps distinguish alternatives that imply different useful continuations. Fit or validate that response on independent image/physical data, or a separately validated renderer. A new view ID, access to the simulator's target label, or an invented favorable return is not evidence of that response.
2. **Construct the inverse action with a completion certificate.** Generate a pose/timing action from the public model and history, preserve a required acquisition when possible, and retain a checked full route under the common resources. Explicitly separate sampled kinematics from any untested contact or imaging assumption.
3. **Establish headroom above equally equipped MPC.** Give the new measurement, candidate actions, state estimator, timing assumptions, and checked paths to the shared replanning MPC. Use new paired cases and independent-repeat selection to test complete-task opportunity with actual movement, elapsed time, planning cost, and wrong-identity accounting. A generator benefit belongs to the generator; a selector benefit requires the additional contrast above that enriched MPC.
4. **Show that public evidence predicts the opportunity.** Only after positive headroom, freeze a model-based selector or learned inverse/predictive model and validate on new cases. Demonstrate the claimed scope with a nonbranch task and relevant action/measurement ablations. Do not move the efficacy or cost rule after reading the new outcomes.

This sequence retains B as the intended contribution while making its missing evidence explicit. It does not claim that all inverse sensing, all timing policies, or all robotic-ultrasound B designs are impossible. No further tests are recommended within the current pilot families.

# Independent SOL review

The frozen 32-case screen does not establish a continuation advantage for the tested intermediate-tilt library at the measured costs. I recommend closing this design at the opportunity stage. The fixed progress control and the shared MPC menu controls were exercised. Forecast training and the fresh 256-case efficacy cohort were conditional on the screen and remained unopened.

## Paired outcome reconstruction

I reconstructed each path's fresh ordered claims, wrong-return type, capture count, TCP distance, accumulated turn, rail distance and acquisition clock. Both tasks reused the same 32 independently seeded latent cases and 24 suffix tapes per case. The intervals resample latent cases. The counts below refer to the 768 evaluated case–tape pairs within each task.

| Task | Cost-screened split selector correct / wrong | Augmented MPC correct / wrong | Correct completion difference, paired 95% interval | B-only / MPC-only wrong pairs |
| --- | ---: | ---: | ---: | ---: |
| Daughter | 196 / 71 | 189 / 72 | +0.91 points [−0.26, +2.34] | 5 / 6 |
| NONBRANCH | 664 / 41 | 661 / 42 | +0.39 points [−0.39, +1.30] | 6 / 7 |

The daughter split selector won 12 completion pairs and lost five. Its completion point difference against the fixed immediate-progress control was −0.26 points. For NONBRANCH, it won 10 completion pairs and lost seven, and its difference against the fixed control was −1.30 points. Daughter false-structure claims occurred on 62 selector and 63 MPC tapes. Each had 10 tapes with another-lumen claims, including one tape with both wrong types per arm. All NONBRANCH wrong claims were false-structure returns. Repeats support case-clustered uncertainty. The earlier exact 2% harmful-discordance criterion requires a suitably sized independent fresh cohort.

| Task | TCP difference in mm, simultaneous 95% interval | Turn difference in degrees, simultaneous 95% interval | Sampled-time difference in s, simultaneous 95% interval | Cost classification |
| --- | ---: | ---: | ---: | --- |
| Daughter | −0.096 [−0.473, +0.236] | −0.704 [−2.081, +0.203] | −0.002 [−0.035, +0.031] | No-worse within margins. Rotation exceeds the lower equivalence limit. |
| NONBRANCH | +0.123 [−0.376, +0.615] | +0.080 [−0.640, +0.804] | −0.029 [−0.084, +0.018] | Strict equivalence and no-worse. |

The hard-cap-only daughter selector gained +4.04 points [0.78, 7.81], with +1.57 mm TCP distance and +2.90° mean turn. Its upper cost intervals exceeded all three measured margins. Future-outcome selection on the same tape gave +15.36 points in daughter and +6.38 points in NONBRANCH within the tested library. The split procedure instead ranks on other tapes and measures attained selection. These estimates do not establish a population bound for executable policies.

The augmented MPC actually used an intermediate tilt in 590/768 daughter paths and 490/768 NONBRANCH paths. It had five fewer correct daughter completions and three fewer correct NONBRANCH completions than coarse diagonal MPC. The measured completion differences were −0.65 and −0.39 points, respectively, with intervals including zero. The claim-only and fixed-angle menu controls also yielded no repeat-validated positive generator effect. Added coarse diagonal edges alone gave +2.47 daughter points with an interval of [−0.52, +6.38], +7.56° mean turn and seven additional wrong cases on the repeated tapes. The pilot therefore gives no positive matched-cost result attributable to shared action generation or continuation selection.

Menu and controller CPU are charged to the logged ledger, including in unsuccessful episodes. For daughter, the mean charged CPU was 0.111 s per selected episode and 0.114 s per augmented MPC episode. The paired interval for the sum of sampled time and charged CPU was [−0.032, +0.022] s. NONBRANCH charged means were 0.067 and 0.071 s, with a combined interval [−0.078, +0.006] s. This reports measured processor time alongside simulated motion. Acoustic innovations were indexed by the motion clock, with instantaneous computation in the acoustic model. The selection itself used offline evaluator labels, so these figures describe the replayed continuation paths.

## Implementation and baseline checks

The acoustic audit independently checked 2,016 original view/clock/case combinations for exact packet, correct-return, detection and centroid-CDF parity. It reconstructed world-space lumen projections for 4,032 new view/clock/case combinations with a maximum discrepancy of 0.00000085 mm. Reassigning intermediate view IDs left the simulated kernels exactly unchanged. The numerical differences between new and old packet laws reflect the synthetic tilt and ghost interpolation.

All 60 intermediate endpoints passed independent dIRD proxy, endpoint FK and configured self-clearance checks. The largest endpoint discrepancy was 0.00134 mm and 0.000563°. Joint ranges and rates were recomputed for every one of the 2,028 added edges. An independent full FK and configured self-collision replay covered 173 edges, including all added coarse diagonals, and 13,148 samples. The smallest replayed self-clearance was 39.145 mm. Body/environment collision and maintained tissue contact remain outside these mechanics checks.

The expanded-bank smoke covered 168 complete paths, 779 capture packets, 48 proposal scores and 691 live MPC commands. In the screen I audited 16,056 full paths, 76,328 capture packets, 768 proposal witnesses, 285 forced first-command witnesses and 128 split-half choices. The independent tape regeneration recovered all 768 suffix tapes and their use in both tasks. Each contrast used the full assignment denominator, including failed and unchanged paths. I recomputed all 16 paired contrasts, including simultaneous cost intervals and both cost classifications.

On eight spread-out latent indices, both tasks and four repeats, an independent controller reconstruction recovered 16 anchor states, 192 proposal utilities, 2,868 MPC commands and 668 terminal public-state hashes. Every tested policy tail took the pinned full MPC with the real posterior, prior-claim mass, progress, exact graph node, acquisition clock and remaining resources. All 64 wrapped-original traces matched the compact original graph and archived full MPC command by command. The 16 expanded-smoke original traces also matched. The augmented direct MPC had the same public menu and likelihood as each forced intervention and retained access to the old actions.

The original bank and numerical expanded bank were checked by hash. All eight sources matched the hashes captured with the screen, including the exact offline builder snapshot at `motion/motion_at_screen.py`. A later builder import edit is preserved separately in `motion.py` and `motion/motion_postrun.diff`. The online runner loads the checked bank and does not import the builder. The saved bank and archived outcomes were unchanged.

## Decision

The tilt cohort establishes execution of the new views and a negative result on the prespecified continuation screen. The pilot remains too small to rule out every conditional benefit in other libraries. The geometry-only aperture preflight also found no supported next offset bank. Intermediate lateral offsets leave the relative-bin evidence nearly unchanged away from aperture clipping. The one supported wider final-daughter contact altered selected-target probability by at most 0.29 points in that preflight. This gives no reason to spend another outcome cohort on the tested shift mechanism. Retain the verified negative result and reserve any further B claim for a distinct observation mechanism with new validation.

The independent scripts in this directory accept the decompressed inherited and expanded motion banks. `audit_world.py` checks acoustic kernels, `audit_motion.py` replays mechanics, `audit_paths.py` reconstructs all executed outcomes and comparisons, and `audit_controller.py` verifies the original and augmented planner traces. The screen controller replay used `--first-cases 4 --extra-cases 7 15 23 31 --first-repeats 4`.

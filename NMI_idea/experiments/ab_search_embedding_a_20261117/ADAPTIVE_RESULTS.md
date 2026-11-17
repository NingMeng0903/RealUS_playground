# Recovery demonstrations and public observation selection

The [protocol](ADAPTIVE_PROTOCOL.md) defines the fixed-angle scene families, the common MPC prefix and continuation, and the paired analysis. Each verification tape begins at the saved post-prefix state, so its future observation noise is independent of the prefix. All competing policies on that tape share its hidden scene and disturbances. The primary and follow-up checks use separate, previously unused scene seeds. The underlying families and the shared observation model are known to the simulator and the controllers.

## Evidence and motion

For a historical image at time \(i\), the controller keeps the responsibility \(r_i(z)=P(A_i=\mathrm{target}\mid Z=z,Y_i,s_i,x_i)\) for its accepted candidate. The fixed hidden state \(Z\) includes the connection, registration offset, shadow lane and layout. If new packets change the belief \(b_t(Z)\), the old image is rechecked using \(\sum_z b_t(z)r_i(z)\). When the probability falls below 0.95, that interval again needs a fresh certificate. Historical mistakes remain in the legacy record. The point of making the shadow and clutter persistent in the model is to keep each hypothetical execution and its observations consistent with the same geometry.

CEM discovered one-second waypoint paths using 36 development scenes, six fresh tapes per scene and a full feedback MPC continuation. The lateral curve uses a quintic with the specified boundary position and speed plus \(64b\,t^3(1-t)^3\), where normalized time \(t\) is in \([0,1]\) and \(b\) is the midpoint displacement. That term preserves endpoint positions and velocities. Every candidate is checked in the same sampled chart limits and updated online from anonymous packets every 100 ms. The CEM archive has 1,296 requests and 1,246 distinct scored nominal paths. The archive retains outcomes for all scored paths and detailed examples for the MPC, the selected path, waiting, and a nearby poorer path where available.

| Development outcome | Shared MPC | Privileged continuous CEM |
| --- | ---: | ---: |
| Final correct completion | 72/216 | 154/216 |
| Legacy strict completion | 72/216 | 143/216 |
| Fresh correct confirmation at the queried interval | 115/216 | 196/216 |
| Final erroneous record | 15/216 | 0/216 |
| Mean executed reference TCP travel | 39.46 mm | 43.48 mm |
| Mean simulated duration | 6.69 s | 6.51 s |

The CEM comparison includes its wider waypoint space, privileged per-scene selection, and increased travel. The [development screen inside the shared menu](results/adaptive_same_menu_screen_v2/summary.json) also found scope for a better first choice. In the 24 ordinary development scenes, 47/144 MPC completions rose to 86/144 for the privileged shared-menu reference, at +3.90 mm mean TCP travel. Applying forecast resource limits reduced this result to 54/144 with −0.82 mm travel. That screen is a finite search, with disjoint selection and verification tapes, and is treated as a post-protocol diagnostic.

Two development scenes began with active mistaken certificates. For seed 6410001, a 6 mm longitudinal return and 6.81 mm requested lateral movement exposed new evidence. Within the first second the two wrong certificates' identity probabilities dropped below 0.95. Later image acquisition replaced both and completed the route on 6/6 fresh tapes. Seed 6410011 repaired at least one wrong bin on 6/6 tapes and finished on 5/6. Shared MPC repaired no bins in these twelve repeats. The legacy outcome still records the earlier errors. These are recoverability examples found by privileged search.

Twenty-one nominal motions passed the trained-motion selection rule. The ProDMP fit uses only decision-time nominal lateral position and velocity, with endpoint displacement and signed midpoint bow as features. The learned part does not use the later servo corrections. On the primary cases, 48 of 50 requested waypoint substitutions passed the sampled constraints, giving 144 executed fit replays on the three tapes. On these replays the nominal and fitted paths each completed 57/144 scans. They agreed in aggregate for the full 150 substitution trials (63/150 each), including six replays where the original motion was retained. The maximum discrepancy of the nominal lateral curves among accepted substitutions was 0.053 mm. Agreement describes this small shape family, with feedback shared.

An [equal-endpoint replay](WAYPOINT_SHAPE_AUDIT.md) subsequently tested whether the 21 admitted CEM examples depend on their interior lateral bow. On 252 new disturbances of the original development scenes, the full path and the zero-bow reference finished 219 and 218 scans respectively.

A [fixed-request replay of the old and CEM-fitted ProDMP](PRODMP_WEIGHT_SWAP.md) includes the old renderer's zero-residual option and a shared analytical waypoint fallback. The fit passes the motion checks for more of B's requests, with identical completion and identity outcomes under the common fallback.

## Primary closed-loop comparison

The controller scores a first segment from the public posterior by running it and the complete shared MPC on the same six hypothetical worlds as every alternative. It executes the selected segment on the actual scene. The adaptive variant scores all actions again using the **new** public packets after that segment. A first-only variant shares the first action and hands control back immediately. Each row below has 28 independent scenes and 84 paired suffix trials, with all confidence intervals resampled at the scene level.

| Ordinary cohort | Shared-menu MPC | Public first only | Public adaptive | Adaptive difference and paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Local connection ambiguity | 41/84 | 60/84 | 63/84 | +26.2 pp [10.7, 41.7] |
| Unbranched nearby tracks | 26/84 | 46/84 | 47/84 | +25.0 pp [11.9, 40.5] |
| Pooled | 67/168 | 106/168 | 110/168 | +25.6 pp [14.9, 36.3] |

MPC with its original action types finished 65/168. Providing the wider menu alone changed the result by two. The public teacher selected a different initial action than shared-menu MPC in 41/56 scenes. The second public choice varied across paired nuisance tapes of the same scene in eight scenes. The second intervention added four completions relative to the first-only arm, a gain of +2.38 pp with paired 95% interval [0.00, 5.36]. This establishes that the second choice was updated. The incremental completion benefit of that second choice remains unresolved.

The realized pooled differences for the adaptive teacher were +0.85 mm TCP [−1.48, 3.38], +0.63 degrees [−0.02, 1.38], and −0.29 simulated seconds [−0.51, −0.09]. The joint equivalence rule failed. The local connection cohort had higher completion and shorter travel and time (−3.34 mm and −0.67 s). For the unbranched cohort, the new correct checks used an additional +5.05 mm on average [1.45, 8.93]. The legacy strict outcome in the ordinary cohorts was 66/168 for shared MPC and 110/168 for the adaptive teacher. Final erroneous records were 0 versus 0, and new centered mistakes were 1 versus 0. Fresh query confirmation increased from 113/168 to 143/168.

In the stress cohort with a deliberately overconfident prior, shared MPC, first-only and adaptive teachers finished 17/84, 41/84 and 37/84. They retained 24, 10 and 10 wrong final records and incurred 24, 25 and 25 new centered mistakes. In the subset with a wrong pre-existing certificate at the common prefix (five worlds, 15 replays), MPC repaired no bin, while both public variants repaired at least one bin in 3/15. Only the first-only policy finished two of these 15. Recovery is observable, with substantial residual identity risk under this stress model. The exact label-swap control had zero confirmations, false identifications or completions for all policies.

[Post-hoc opening replay](results/adaptive_opening_replay_v2/summary.json) held the tail constant and allowed the public decision to use only the old option types as its first action. It completed 98/168, against 67/168 for the shared-menu MPC and 106/168 for the first-only teacher with the wider menu. In the unbranched cohort all the teacher's selected first actions already belonged to the original menu. This locates most of the benefit in selecting a first inspection by its effect on the completed scan. Endpoint/shape flexibility contributes to some connection cases. A general claim for newly curved trajectories is untested.

A subsequent [first-segment evidence replay](PROBE_EVIDENCE_REPLAY.md) clarifies when the selected motion contributes new identity evidence. It holds the already executed first path fixed and withholds its packets or fixes the structural belief weights until the shared handoff.

## Follow-up with predicted resource constraints

In an independent set of 56 ordinary scenes (168 paired trials), the two public variants scored the same candidates on the same twelve posterior tapes. One required the predicted motion, turn, time and mistake rates to be within the protocol's upper margins of the MPC incumbent before ranking. All arms then executed one choice and the same full MPC continuation.

| Cohort | Shared-menu MPC | Public without resource filter | Public with resource filter | Filtered difference and paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Local connection ambiguity | 46/84 | 57/84 | 57/84 | +13.1 pp [3.6, 25.0] |
| Unbranched nearby tracks | 16/84 | 52/84 | 18/84 | +2.4 pp [−2.4, 8.3] |
| Pooled | 62/168 | 109/168 | 75/168 | +7.7 pp [1.8, 14.3] |

The filtered teacher reduced observed pooled TCP travel by 1.74 mm [−2.80, −0.79] and scan duration by 0.17 s [−0.36, −0.01]. Rotation was −0.03 degrees [−0.18, 0.15] and acquisition count was −1.73 [−3.57, −0.06]. It passed the pooled upper-cost-margin tests. Its symmetric equivalence failed because of savings larger than the lower margins. Final/new mistaken cases were 1/168 versus 0/168; the case-bootstrap upper endpoint was +1.79 points. The binary per-scene sensitivity check for any added final error is much less precise. One out of 56 scenes yields a one-sided 97.5% binomial upper limit of 9.55%, and zero out of 56 in the primary comparison yields 6.38%. These counts are too small for a tight population risk guarantee.

The connection group repeated the success and resource benefit: −2.47 mm and −0.44 simulated seconds, with zero new errors. The filtered unbranched group did not pass the improvement, time or identity-risk gates by itself. The unrestricted unbranched method reacquired more fresh evidence but executed an additional 10.48 mm in this cohort. This motivates distinguishing access to any diagnostic observation from access to an equally cheap continuation. The current evidence supports a task-dependent selection claim, with fewer resources in the connection task. The results do not establish uniform improvement at matched execution cost across anatomical tasks.

The teacher uses finite rollout re-ranking with the shared belief model and MPC continuation. Timing is logged separately. A serial check on the first ordinary seed in each cohort took 2.43 and 3.71 s for the six-sample search over 15 candidates, while shared-menu MPC took about 5.6 ms on both seeds. Time in the tables measures the simulated scan. The study uses static template-derived synthetic centroid observations, a matched discrete likelihood and chart-based motion constraints. Student decision speed, physical ultrasound, contact and full-robot feasibility require separate evaluations.

## Reproduction and raw records

The [discovery](results/adaptive_discovery_v2/summary.json), [primary](results/adaptive_paired_v2/summary.json) and [resource comparison](results/adaptive_budgeted_v2/summary.json) directories contain the per-case `cases.jsonl.gz` files, as well as summaries. The first tape for each fully archived policy shows nominal and revised references, executed samples, public observations, Bayesian retractions and separate evaluator data. The [trace audit](results/ADAPTIVE_TRACE_AUDIT.json) reconstructs distances, rotation, decision continuity, active certificates and legacy outcomes for 751 archived executions, including 20 retractions. It also checks exact packet equality under a hidden label swap at 81 exposures.

```sh
export OPENBLAS_NUM_THREADS=1
python -B run_adaptive_recovery.py discover --evidence-mode legacy --cases 12 --families connection parallel wrong_prior \
  --replicas 6 --population 6 --iterations 2 --search-samples 3 --jobs 16 --out results/adaptive_discovery_v2
python -B run_adaptive_recovery.py paired --evidence-mode legacy --cases 28 --replicas 3 --public-samples 6 --jobs 18 \
  --fit results/adaptive_discovery_v2/prodmp_nominal_fit.npy --out results/adaptive_paired_v2
python -B same_menu_headroom.py results/adaptive_discovery_v2/cases.jsonl.gz \
  --jobs 12 --out results/adaptive_same_menu_screen_v2
python -B budgeted_recovery.py --cases 28 --samples 12 --replicas 3 --jobs 18 --out results/adaptive_budgeted_v2
python -B replay_adaptive_openings.py results/adaptive_paired_v2/cases.jsonl.gz --out results/adaptive_opening_replay_v2
python -B audit_adaptive_recovery.py results/adaptive_discovery_v2/cases.jsonl.gz \
  results/adaptive_paired_v2/cases.jsonl.gz results/adaptive_budgeted_v2/cases.jsonl.gz
```

The later resource check also records predicted turns and new mistakes for every candidate; the original public score and choice rule are unchanged by logging these additional fields.

The subsequent [trained end-to-end comparison](LEARNED_B_RESULTS.md) freezes a CEM-fitted motion generator and a learned recovery selector before running new paired cases against strong MPC.

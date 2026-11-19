# Independent theory and record audit

2026-11-13. **This completed screen supports a useful optional near-image retry, but no useful B advantage over a fixed sensible first view followed by ordinary live MPC. Archive this result and stop this route.** It does not establish two-perspective geometric disambiguation or transfer the earlier topology result.

## Result and attribution

Exact success probabilities in the frozen finite model are:

| Policy | Correct accepted fresh arterial pair |
|---|---:|
| Two-image controller | 70.55935115% |
| Live stochastic MPC | 75.71597304% |
| Fixed first near-center view + the same live MPC | 81.84589205% |
| Exact contingent B | 81.84842169% |

B exceeds executed MPC by 6.13244864 percentage points. This is not a comparison against a stale forecast: MPC itself improves from its 70.55955728% forecast to 75.71597304% through actual replanning, a 5.15641577-point increase. The reported root-tie calculation leaves a 3.84818303-point B gap at the code's `1e-12` tie tolerance. Thus the primary gap is real within this declared controller/task class and is not entirely a tie artifact.

The decisive control is fixed first near-center plus live MPC. B improves on it by only **0.00252964 percentage points**, while using **2.91797382 mm more TCP travel** on average, exceeding the declared +2 mm band. That control reproduces 99.95875% of B's improvement over MPC and 99.97759% of B's improvement over the two-image controller. These percentages are descriptive comparisons, not a causal decomposition. The fixed control fixes only the first view; it retains the common posterior updates, optional near retry, far-view selection, and final joint declaration.

B uses less motion, turning, time, and fewer frames than the particular MPC root forecast. Its failure of symmetric cost-equivalence bands against MPC reflects those savings, not a cost penalty. Against the strongest fixed-first control, however, the additional TCP travel is a material disadvantage for a negligible success increment. None of the primary comparisons should be described simply as equal realized cost.

## The retry ceiling and the latest-frame restriction

Intrinsic artery detection is independent between acquisitions with probability `p=.84`, the same for both vessel roles. Under the frozen class of at most two near attempts and **one** terminal far attempt, success requires a detected artery at the far station and at least one of the two near attempts. Even if registration, observed-slot identity, aperture, and route constraints were made favorable,

```text
P(success) <= p [1 - (1-p)^2] = .818496.
```

This grants no knowledge of future detection failures. It is an optimistic ceiling for this action class, not a universal three-image ceiling. A class permitting a retry of whichever near/far acquisition failed could reach `3p^2-2p^3=.931392` in the ideal dropout-only problem. That is outside the frozen protocol and was not added to the primary experiment.

The two-image detection ceiling is `p^2=.7056`; one optional near retry raises it by **11.2896 points**. The observed B-versus-two-image increment is **11.28907054 points**. The fixed-first control is only **0.00370795 points** below the generous detection ceiling; B is **0.00117831 points** below it. All remaining improvements from identity inference, aperture selection, or better policy together are therefore tiny in this model. The primary effect is acquisition recovery after a miss, not evidence of difficult semantic artery–vein identification.

The protocol's terminal pair uses the latest near frame. A forced second near capture therefore replaces the first eligible near packet even when that earlier packet was good. With free identity and full visibility, always taking two near frames but using only the last still forecasts `p^2`; retrying only after failure reaches `p^2(2-p)`. This explains why an apparently strong fixed-second forecast can undervalue beginning at the useful near station.

In a static anatomy, retaining either of two already acquired qualifying near frames is a natural stronger task-state baseline unless consecutive-frame eligibility is an independently required condition. Retention would let a fixed two-near sequence attain the same ideal detection ceiling. This is a transparent post-primary structural observation, not a changed primary result or a newly executed counterfactual policy. No additional sensitivity cohort is justified by the negligible gap to the frozen fixed-first control.

## Physical geometry and observation mechanism

The section extractor applies the same archived static 15-degree joint transforms to both registered Among_US vessel meshes. It takes actual mesh-plane intersections at the archived sites, chooses the unique local closed contour per class, and uses its polygon centroid. The maximum stored probe-origin discrepancy from the checked FK bank is 0.001218 mm. Source labels identify the reference templates; public packets marginalize the return assignments and contain no stable vessel IDs.

At each site, views 3, 7, and 8 have **the same origin and the same section-plane normal**. The +/-12-degree views rotate the axes and rectangular aperture within that same plane. They do not acquire independent anatomical perspectives. Nominal artery–vein centroid separations are 14.1476, 17.8514, and 20.8738 mm at sites 3, 4, and 5; this screen does not contain deliberately overlapping local lumens.

Prior-averaged artery detection probabilities, including aperture censoring, are:

| Station | Center | Minus 12 degrees | Plus 12 degrees |
|---|---:|---:|---:|
| 3 | .840000 | .773827 | .840000 |
| 4, near | .839946 | .467096 | .840000 |
| 5, far | .822326 | .142860 | .840000 |

The vein's probability is .84 at every pose. This is caused by geometric position within the aperture, not role-dependent intrinsic detection. Therefore orientation-dependent performance here is principally aperture/quantization behavior on the same cross-section. Centroid bins and independent synthetic dropout remain an explicitly assumed sensor, not rendered or calibrated B-mode evidence.

## Independent verification performed

The four frozen hashes in `SUMMARY.json` match the current geometry, protocol, and decision/evaluation sources. I independently reaggregated all 1,166 stored prescan branches, including the 558 above the numerical cutoff, reproducing B value, MPC forecast, MPC executed value, disagreement mass, and the numerical tie calculation. Omitted prescan probability is `2.608954214652854e-11`.

For all **3,072** saved policy replay paths, I reconstructed the posterior from public packet likelihoods, recomputed the final joint near/far slot decision using explicit per-hypothesis summation and the .87 pair threshold, and checked outcome truth, freshness/order, capture counts, legal edges, and resource caps. Summed motion ledger discrepancies were at most `1.42e-14`. The reconstructed replay has 57 B wins and 21 losses against MPC. Equal MPC/B root actions produced no differing live traces.

I also checked the two archived motion banks: their nine selected node joint configurations agree exactly; all 63 selected edge samples start and end at those node configurations; and inherited resource debits match the selected source edges exactly. This establishes consistency of the simulated bank and its recorded concatenations. It does not reexecute continuous robot contact or certify patient collision safety. I inspected the likelihood and decision equations and audited the stored exact results; I did not independently regenerate the entire geometry export or rerun all exact planning trees.

The final supported conclusion is narrow: **a good first near view and an optional retry nearly saturate the finite static acquisition model. Contingent B does not add useful value beyond that simple control.**

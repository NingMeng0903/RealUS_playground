# Two-window Fisher proposal and adaptive stopping

2026-11-11. Decisions below are specified before the new held-out acoustic cases. All motion is offline native QP simulation. Prior source is the corrected sibling `ab_closed_loop_reviewed_20261111`.

## Stage 1. Mechanical witnesses

Use both constrained fixtures with the same measured contact frame, geometry, initial native state and physical limits. Change only the nominal fork-plus local y from -3 mm to 0 mm. Keep x=-6 mm, positive tilt 0.075 rad and the same acquisition dwell. Check from the common start and after the same fork-minus prefix. Run both terminal continuations from each accepted old or shifted probe's actual exit. Save attempted routes and dIRD scores, original 2 mm/2 degree tracking gates, synthetic contact and sweep checks, q/qdot and controller-state linkage, checked elapsed times and measured acoustic poses. Keep rejections. A missing witness is a limited search result. Numerical jerk is logged under the existing experiment policy.

## Stage 2. Paired acoustic and task evaluation

- Both existing constrained layouts. Use held-out object seed 721503 and observation seed 721504, 128 objects. Pair on object and layout. Use common capture/visit noise keys across arms. Do not optimize candidates or scores on held-out results.
- No hard elapsed time deadline. A common finite tree allows up to two diagnostic decisions and up to two fresh terminal attempts. Keep the original acquisition dwell for each primitive. An adaptive policy can scan immediately, stop after one probe, or choose a second probe. Record cap occupancy, absolute time and wall-clock planning. This evaluates finite, interruptible probing, without claiming learned continuous dwell or indefinite-horizon behavior.
- The primary utility uses the previous planner's time preference, success probability minus 0.04 per simulated second of motion/dwell. Report success, wrong-window rate, and number of probes separately. If affordable, repeat with a predeclared lower time weight and a zero weight, with unchanged independent evaluation. These are sensitivity conditions, not opportunities to reselect the primary result.
- All policies receive only the public observation, measured pose, current checked exit and legal belief. The unchanged private renderer uses the realized pose in the true surface frame. The same independent new-packet target, center, axis and quality acceptance applies to all policies. After a public reject, use the common checked terminal continuation and retry policy. Never use the acceptance bit to decide whether to retry. Report any-success, first formal attempt success, false public stop, no scan and both types of elapsed time. Synthetic detected features are the endpoint.
- Wrong first window is scored independently from private mean quality and geometry at the two alternatives checked from the same formal-entry state. Use the old center, axis and quality gates. A wrong choice requires the selected alternative to fail and the other to pass. Report both good, both bad and unchecked alternatives separately. One private draw failing detection alone is not a wrong-window label.

## Comparators and attribution

- Original fixed fork-minus/fork-plus pair (the same original public order convention where applicable), followed by the shared terminal policy. If included, shifted fixed pair is named separately. The fixed arms do not skip the second probe when its checked prefix remains available.
- Observed-mixture logdet proposal followed by the same public stop/terminal decisions. State its probe scoring horizon and search budget. Verify it uses only visible likelihood with hidden target/clutter association marginalized.
- Original B continuation utility on the original primitive bank, adapted to replan after each diagnostic segment. This provides the historical-library comparator, with no hard time cutoff.
- Matched shared-bank task-only B and task-Fisher B, with the same number of candidate value evaluations. The common bank contains the shifted action only if independently motion-checked. Compare against an exhaustive same-bank B to separate any gain in candidate proposals from the changed bank or a search budget restriction.
- Task Fisher is a proposal score. Use local contrast between the two terminal designs to rank probe candidates. It is excluded from the final task utility. Give actual probe/terminal feasibility checks and the same checked recovery opportunities to all arms. Publish exact pre-orders, shortlists, value calls and chosen action. A match on all choices is a valid null result.
- Check quadrature convergence on public development histories before test or explicitly report a post-freeze sensitivity, retaining the original result. Candidate ordering changes near ties are not robust evidence. If a terminal retry is omitted from the predicted value, label the planner as truncated and test how that approximation changes the claim.

## Evidence and decision

The primary Fisher attribution is Fisher vs plain B at identical bank and value-evaluation budget. B on the full bank, original bank and fixed pairs identify changes due to the new geometry or adaptive stopping. Count first-probe and terminal choice differences. Pair all differences by object with both layouts retained in the same bootstrap cluster, 10,000 resamples. Do not treat motion cache reuse or two layouts as independent subjects. Compare the fixed-pair acquisition count alongside success and wrong-window rates. Keep positive, reversed and null paired examples. Do not claim that a raw Fisher increase or an already checked terminal is an improvement in closed-loop success.

## Frozen sensitivity and prediction implementation

Before any held-out run, add matched B shortlist sensitivity K=2 to distinguish
a tight value-evaluation budget from the primary K=4. Keep identical candidates,
two probe opportunities and terminal rules. Also evaluate a zero soft time
weight if computationally affordable. Treat the two variations as separate
sensitivity conditions, not replacements for the primary K=4, 0.04/s run.

Marginal-observation Fisher is evaluated at the current public response mean.
The covariance used in the rank-one score is the public mixture's total
covariance. Both association branches enter the scalar Fisher integral, and
missing contributes zero for w in this fixed geometry-support model. The
contrast is the two checked terminal H rows. Forecasts of future success use the
corrected mixture itself. Terminal retry continuation is integrated below the
public quality stop gate in component-CDF space. Direct adaptive integration
on all four public root terminals agreed within 0.000070 utility at order 16
and 0.000013 at order 32. Diagnostic branch integration has its own independent
convergence check and can affect policy choices.

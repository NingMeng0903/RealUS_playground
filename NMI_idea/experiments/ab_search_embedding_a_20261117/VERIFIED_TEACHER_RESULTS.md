# Held-out test of learning from verified online decisions

The verified-teacher student finished **330/768 paired scans** against **304/768 for strong MPC**, a gain of **3.39 percentage points** [1.30, 5.86]. It fell below the five-point minimum. The prior frozen learned B finished **349/768** on those same cases. The direct paired difference from prior B to the new student was **−2.47 percentage points** [−4.82, −0.26].

| First-decision policy | Connection | Unbranched | Pooled | Gain vs MPC, pp [95% CI] | Pooled first-decision CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong MPC | 188/384 | 116/384 | 304/768 | — | 37.3 ms |
| Prior learned B, frozen | 233/384 | 116/384 | 349/768 | +5.86 [3.13, 8.85] | 76.4 ms |
| Verified-teacher student | 214/384 | 116/384 | 330/768 | +3.39 [1.30, 5.86] | 69.8 ms |
| Online posterior rollout B | 236/384 | 123/384 | 359/768 | +7.16 [4.17, 10.55] | 27.86 s |
| Online rollout without extra CEM first choices | 230/384 | 123/384 | 353/768 | +6.38 [3.52, 9.64] | Shared forecasts |

The new student changed the first action in 14 connection scenarios and zero unbranched scenarios. Its gains come entirely from connection ambiguity, where it improved completion by 6.77 points [2.60, 11.46]. The prior model also gained in that family on the previous holdout and in this one, with a gain of 11.72 points [6.51, 17.45] here. In the new unbranched cases online B added seven net completions. The upper confidence endpoint for its extra simulated duration was +0.104 s, slightly above the +0.10 s margin there.

| Pooled measured outcome | MPC | Prior learned B | Verified-teacher student | Online rollout B |
| --- | ---: | ---: | ---: | ---: |
| Final wrong identities | 0/768 | 0/768 | 0/768 | 0/768 |
| New centered mistakes | 2/768 | 2/768 | 1/768 | 1/768 |
| Executed TCP path | 38.04 mm | 37.40 mm | 37.84 mm | 36.75 mm |
| Executed rotation | 7.141° | 7.007° | 7.072° | 7.012° |
| Simulated scan duration | 6.740 s | 6.505 s | 6.647 s | 6.529 s |
| Simulated scan plus first-decision CPU | 6.777 s | 6.582 s | 6.717 s | 34.390 s |

The verified student passed the upper execution cost margins. Its mean paired difference in simulated duration plus first-decision CPU was −0.060 s [−0.124, −0.007]. It added a mistake in zero of 192 scenarios, giving a 1.90% one-sided 97.5% upper bound on this scenario-level outcome. The prior student had an added mistake in one scenario despite equal pooled counts. Its sensitivity upper limit is 2.87%. In a [serial decision measurement](results/verified_teacher_b_v1/serial_decision_latency.json) on 48 evenly spaced cases, mean first-decision wall time was 14.5 ms for MPC, 30.0 ms for the prior student and 27.5 ms for the verified student. The corresponding 95th percentiles were 44.3, 59.8 and 57.4 ms. The scan is simulated and does not evolve during computation.

## Why the new training stays conservative

Training used 416 public root states and 1,453 nominated branches, each on twelve new paired disturbances. In the unbranched group the online choice improved completion in 22 of 272 training states, and six of those also passed every per-action cost and error check. Each of the other sixteen exceeded the observed mean duration margin alone. Including the verified challengers gave positive training choices in only 14 unbranched states, with two verified `short` candidates in the coverage-ahead stratum and one `cem_1` in the past-gap stratum. In connection the corresponding counts were 46 observed teacher improvements, 41 verified teacher improvements and 51 states with some verified alternative out of 144. The independently selected classifier retained the MPC action in every unbranched test case. This experiment tests direct supervision with strict observed action costs and quantifies the scarce positive support for that supervision.

Development compared the verified action classifier and the signed paired value model. The classifier delivered the higher correct completion with compliant mean costs. The value model changed many more actions, added 4.86 mm mean travel and lost 25 unbranched completions relative to MPC on development. The classifier and its training inputs were frozen before opening the test cases.

SOL MAX independently found that the new student won 14 and lost 33 tapes against the prior learned B. The 40 connection cases where the prior B switched and the new classifier stayed with MPC contributed 22 of the net losses before gains elsewhere. In the new unbranched evaluation, online B's seven extra completions came from four `advance` to `short` switches. Three of those four cases increased mean simulated duration by at least 0.5 s. The two learned selectors had the same correct completion as MPC in this family. The verified classifier ranked MPC first in all of its unbranched cases.

## Effect of the extra CEM recovery options

On the same online decisions, suppressing extra CEM first proposals while retaining the baseline action and common continuation reduced completion from 359 to 353. The increment was +0.78 percentage points [0.00, 1.82], all in connection scans. Mean executed travel decreased by 0.179 mm [−0.322, −0.061], rotation decreased by 0.056° [−0.118, −0.003], and identity mistake counts coincided. The simulated duration difference was −0.0065 s [−0.0286, +0.0130]. This comparison isolates extra first choices while both arms retain the same CEM-fitted ProDMP and continuation. The result supports small case-specific benefits under the shared execution margins.

The full planner won eight paired replays and lost two against the restricted planner. The net gain came from three cases, 17210018, 17210047 and 17210092. The first spent an additional 2.56 mm and 1.00 s relative to the restricted planner. The last saved 9.07 mm and 1.25 s. The execution equivalence comparison is pooled across the held-out cases.

The earlier label comparison tested a different effect. Both learned models could already execute the CEM options, and including their candidate labels in training changed 320/768 to 317/768. The direct interval [−2.86, +2.08] percentage points leaves an incremental learning benefit unresolved. The [per-case audit](TEACHER_CHOICE_AUDIT.md) identifies the specific unbranched decisions and the cost-prediction discrepancies.

## Reproduction records

See the [protocol](VERIFIED_TEACHER_PROTOCOL.md), [frozen manifest](results/verified_teacher_b_v1/freeze.json), [held-out statistics](results/verified_teacher_b_v1/test_summary.json), [raw paired cases](results/verified_teacher_b_v1/test.jsonl.gz) and [trace audit](results/verified_teacher_b_v1/trace_audit.json). The separate training and development archives preserve the paired outcomes used to select the model. Results use the fixed-pose, matched chart and synthetic packet benchmark.

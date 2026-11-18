# Optional probe control beyond the previous scan clock

This paired replay uses the frozen PSDP policies at ad7264e and the
same 128 nominal and 48 drifting held-out cases. The complete table
covers all 17 policies and controls in that release. The first 81
ultrasound packets use exactly the original disturbance arrays. Further
random packets are generated deterministically as queried, using the
same frame-indexed perturbations for every arm in a case and replica.
The moving vessel remains periodic across the previous endpoint. The
original rollout is replayed and checked against each archived
metric and leading/second action. The 67 mm travel budget and the base
kinematic and continuity tests remain in force.

## Conditions

- Original reproduces the 8-second stop.
- Extended removes the stop and leaves the original MPC decisions in
  place. When an already executed stationary command leaves coverage
  unchanged, and MPC requests another stationary command at the same
  pose and coverage, that rollout is marked **unresolved stationary**
  after the previous 8-second boundary. This observation of no progress
  does not rule out eventual new information during further waiting.
- Open removes the stop, keeps the existing recover-on-evidence
  condition available beyond its old upper-frame restriction, and
  uses a common recovery action if a repeated stationary command
  would leave the task pending. Among the existing lawful motions
  it first prefers the move arriving closest to the public pending
  scan interval (or finishing station if coverage is complete).
  The original MPC score ranks equal-distance commands. All methods
  use exactly this same fallback. A scan ends when the controller
  reaches its task-complete condition, has no valid command, or
  needs recovery and has no feasible repositioning movement.
  Elapsed time never triggers termination in this condition.

B and child weights are frozen. In particular, training still used
the old clock. The main intervention choices are made at the same
causal checkpoints as the original replay. Any recovery behavior
modified after B hands back belongs to the common MPC. The open
condition thus measures a no-clock scan pipeline with explicit progress
handling. The separate extended condition isolates the timer removal
up to stationary cases that remain unresolved.

## Control rates

The trajectory consists of successive one-second constrained ProDMP or
analytic blocks with 20 ms motion sampling. Position and velocity
match numerically at block transitions. Ultrasound supplies one packet
each 100 ms. The executor only installs a visual correction when the
latest target association and offset are actionable and the updated
suffix passes the checks. Every policy uses that same executor.
PSDP decides its first intervention at the fixed checkpoint after
the initial MPC sweep, then may decide once more based on the reached
observations. It can return to MPC at either decision. The application
still invokes the first PSDP selector at a fixed checkpoint, including
scans where the original MPC recovery trigger is false.

## Reproduction

The precommitted-second controls replay the commands recorded for the
original frozen planner. The planned second command is kept fixed before
the first scan returns, and still has to pass the same execution checks.
The open table varies execution duration and common recovery. It does
not recompute those plans with a longer forecast. Within-segment feedback
and the executed first segment are shared as in the original controls.

From this directory, with the original frozen experiment and the verified
eight-policy continuation archive in results/main:

~~~sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_open.py drift --complete-table --workers 20
python run_open.py test --complete-table --workers 27
python summarize_open.py --out results/full_table
~~~

The complete replay verifies and reuses the eight-policy continuation
records and runs missing action sequences. The capped scores and executed
first and second actions reproduce the original benchmark for every
added arm.

## Matched replay results

| Policy | Nominal / 1,024 at 8 s | Nominal extended | Nominal open | Drift / 384 at 8 s | Drift extended | Drift open |
|---|---:|---:|---:|---:|---:|---:|
| Full MPC | 718 | 721 | 723 | 283 | 288 | 289 |
| Best fixed single probe | 827 | 830 | 832 | 320 | 332 | 333 |
| Random training data + one probe | 891 | 895 | 897 | 339 | 347 | 347 |
| Evidence teacher nominations + one probe | 897 | 901 | 903 | 330 | 341 | 342 |
| Earlier released evidence B | 914 | 918 | 919 | 324 | 335 | 336 |
| Outcome-contrast data + one probe | 916 | 920 | 923 | 323 | 334 | 335 |
| Previous two-probe B | 919 | 928 | 929 | 326 | 341 | 342 |
| Previous two-probe first + frozen second | 940 | 944 | 944 | 332 | 337 | 337 |
| Outcome-contrast first + trained child | 934 | 943 | 943 | 332 | 343 | 343 |
| Matched direct labels, one probe | 907 | 911 | 915 | 322 | 332 | 332 |
| Matched direct labels + child | 927 | 942 | 945 | 331 | 343 | 343 |
| Matched adaptive labels + child | 931 | 940 | 941 | 318 | 335 | 341 |
| Matched mixed labels + child | 930 | 936 | 937 | 328 | 338 | 340 |
| PSDP mixed-residual control | 928 | 939 | 940 | 337 | 350 | 350 |
| Independent continuation labels + PSDP | 944 | 953 | 954 | 327 | 348 | 348 |
| Same PSDP first action, direct MPC | 854 | 858 | 860 | 301 | 310 | 310 |
| Same PSDP first action, frozen second | 960 | 963 | 964 | 320 | 332 | 332 |

Final/interim errors are trial counts. Motion entries are mean actual TCP
travel in mm, accumulated turn in degrees, and physical scan duration in
seconds. Planning compute is separate.

| Policy | Nominal final / interim wrong | Nominal mm / deg / s | Drift final / interim wrong | Drift mm / deg / s |
|---|---:|---:|---:|---:|
| Full MPC | 50 / 127 | 45.77 / 7.42 / 8.68 | 24 / 57 | 44.79 / 7.58 / 7.80 |
| Best fixed single probe | 23 / 195 | 49.70 / 9.39 / 8.51 | 20 / 71 | 48.58 / 9.48 / 6.97 |
| Random training data + one probe | 45 / 190 | 47.77 / 9.21 / 7.32 | 14 / 54 | 46.74 / 8.94 / 6.67 |
| Evidence teacher nominations + one probe | 29 / 178 | 47.98 / 9.14 / 7.35 | 15 / 56 | 47.29 / 9.06 / 6.78 |
| Earlier released evidence B | 29 / 170 | 46.82 / 9.05 / 7.21 | 20 / 58 | 47.30 / 9.20 / 6.84 |
| Outcome-contrast data + one probe | 44 / 176 | 47.09 / 9.20 / 6.94 | 16 / 53 | 47.05 / 9.01 / 6.79 |
| Previous two-probe B | 35 / 181 | 49.14 / 9.58 / 7.24 | 15 / 58 | 48.60 / 9.51 / 6.99 |
| Previous two-probe first + frozen second | 31 / 168 | 48.29 / 9.43 / 7.04 | 13 / 61 | 49.08 / 9.55 / 7.27 |
| Outcome-contrast first + trained child | 43 / 171 | 49.48 / 9.73 / 7.47 | 14 / 57 | 49.76 / 9.70 / 6.98 |
| Matched direct labels, one probe | 34 / 178 | 47.17 / 8.93 / 7.09 | 17 / 60 | 46.96 / 8.92 / 6.95 |
| Matched direct labels + child | 37 / 176 | 50.19 / 9.50 / 7.26 | 13 / 62 | 50.32 / 9.70 / 7.13 |
| Matched adaptive labels + child | 45 / 170 | 49.51 / 9.46 / 7.21 | 17 / 55 | 48.33 / 9.18 / 7.30 |
| Matched mixed labels + child | 39 / 173 | 50.89 / 9.83 / 7.38 | 14 / 59 | 50.65 / 9.57 / 7.58 |
| PSDP mixed-residual control | 29 / 166 | 50.25 / 9.76 / 7.41 | 15 / 57 | 48.94 / 9.38 / 7.02 |
| Independent continuation labels + PSDP | 31 / 163 | 50.21 / 9.65 / 7.28 | 13 / 55 | 50.12 / 9.52 / 7.28 |
| Same PSDP first action, direct MPC | 28 / 171 | 48.09 / 8.79 / 7.88 | 23 / 58 | 48.10 / 8.64 / 7.60 |
| Same PSDP first action, frozen second | 20 / 165 | 48.83 / 9.54 / 7.34 | 19 / 64 | 50.08 / 9.51 / 7.71 |

The capped rows independently reproduce all recorded metrics and the
two leading decisions for the eight evaluated policies on all 176
cases. Turning off the clock with original MPC let 21 additional
PSDP scans finish in the drifting group, compared with eight
additional random-data scans. The resulting counts were 348 and 347.
The shared stationary recovery condition gave the same correct
completion counts in that group. Both extended and open comparisons
yield +0.26 percentage points for PSDP with a case-paired 95% interval
of [−1.82, +2.60]. The previously observed 12-run difference was
sensitive to censoring at the original clock. A reliable PSDP advantage
over random-data B in the drifting group remains unestablished.

In the open drift comparison each of these two policies had 23
no-feasible-reposition stops. The PSDP policy required on average
3.38 mm more travel and approximately 0.61 s more scan time. Correct
completion does not require artificially precise centering. It uses the
same identity, coverage, trajectory endpoint and new retained image
test as the previous score.

With all 17 arms restored, the open nominal PSDP policy finished
954 versus 923 for the outcome-contrast one-probe B (+3.03 percentage
points [+0.20, +6.15]). It finished 10 fewer nominal scans than the
same-first frozen second command (954 versus 964, −0.98 points
[−2.64, +0.68]). In drift the conditional PSDP completed 16 more
than that particular frozen-command comparator (348 versus 332,
+4.17 points [+0.26, +9.11]). The drifting random-data B remained
close at 347. These comparisons evaluate fixed published head weights
and preserved precommitment choices under open execution.

The first PSDP selector was called at all 48 original drift roots and
made a different first motion from the MPC anchor at 45. At 22 of the
24 roots for which the ordinary evidence-based MPC trigger was false,
PSDP still changed the first motion. The numbers of executed B probe
blocks across the 384 tapes were 16 zero, 162 one and 206 two. Those
decisions are separate from 1,823 accepted within-block visual
corrections during B's probes and 5,634 during the subsequent common
MPC scan in open drift. The evidence-triggered entry of an entirely
event-driven B is a separate policy choice. This replay measures the
existing fixed-checkpoint entry.

The extended condition leaves cases unresolved when the original
control behavior remains stationary. The open condition applies the
documented shared recovery behavior, so its results include that
change. All results use a synthetic ultrasound stream, and the fitted
B models retain the distribution of the earlier capped training.
The raw archives include stop reasons, physical execution traces and
servo correction counts.

[Complete nominal cases](results/full_table/test.jsonl.gz) ·
[Complete drift cases](results/full_table/drift.jsonl.gz) ·
[Paired intervals and diagnostic telemetry](results/full_table/summary.json)

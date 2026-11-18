# Optional probe control beyond the previous scan clock

This paired replay uses the frozen PSDP policies at ad7264e and the
same 128 nominal and 48 drifting held-out cases. The first 81
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

From this directory, with the original frozen experiment present:

~~~sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python run_open.py drift --workers 20
python run_open.py test --workers 27
python summarize_open.py
~~~

## Matched replay results

| Policy | Nominal / 1,024 at 8 s | Nominal extended | Nominal open | Drift / 384 at 8 s | Drift extended | Drift open |
|---|---:|---:|---:|---:|---:|---:|
| Full MPC | 718 | 721 | 723 | 283 | 288 | 289 |
| Random training data + one probe | 891 | 895 | 897 | 339 | 347 | 347 |
| Evidence nominations + one probe | 897 | 901 | 903 | 330 | 341 | 342 |
| Outcome-contrast data + one probe | 916 | 920 | 923 | 323 | 334 | 335 |
| Previous two-probe B | 919 | 928 | 929 | 326 | 341 | 342 |
| Outcome-contrast first + child | 934 | 943 | 943 | 332 | 343 | 343 |
| Independent continuation labels + PSDP | 944 | 953 | 954 | 327 | 348 | 348 |
| Same PSDP first action, direct MPC | 854 | 858 | 860 | 301 | 310 | 310 |

The open condition has the following actual execution costs and identity outcomes.

| Split and policy | Final wrong | Interim wrong | TCP travel mm | Turn degrees | Scan duration s | Longest scan s |
|---|---:|---:|---:|---:|---:|---:|
| Nominal · Full MPC | 50 | 127 | 45.77 | 7.42 | 8.68 | 129.0 |
| Nominal · Random training data + one probe | 45 | 190 | 47.77 | 9.21 | 7.32 | 91.0 |
| Nominal · Evidence nominations + one probe | 29 | 178 | 47.98 | 9.14 | 7.35 | 91.0 |
| Nominal · Outcome-contrast data + one probe | 44 | 176 | 47.09 | 9.20 | 6.94 | 91.0 |
| Nominal · Previous two-probe B | 35 | 181 | 49.14 | 9.58 | 7.24 | 91.0 |
| Nominal · Outcome-contrast first + child | 43 | 171 | 49.48 | 9.73 | 7.47 | 93.0 |
| Nominal · Independent continuation labels + PSDP | 31 | 163 | 50.21 | 9.65 | 7.28 | 54.0 |
| Nominal · Same PSDP first action, direct MPC | 28 | 171 | 48.09 | 8.79 | 7.88 | 73.0 |
| Drift · Full MPC | 24 | 57 | 44.79 | 7.58 | 7.80 | 27.0 |
| Drift · Random training data + one probe | 14 | 54 | 46.74 | 8.94 | 6.67 | 23.0 |
| Drift · Evidence nominations + one probe | 15 | 56 | 47.29 | 9.06 | 6.78 | 30.0 |
| Drift · Outcome-contrast data + one probe | 16 | 53 | 47.05 | 9.01 | 6.79 | 27.0 |
| Drift · Previous two-probe B | 15 | 58 | 48.60 | 9.51 | 6.99 | 71.0 |
| Drift · Outcome-contrast first + child | 14 | 57 | 49.76 | 9.70 | 6.98 | 27.0 |
| Drift · Independent continuation labels + PSDP | 13 | 55 | 50.12 | 9.52 | 7.28 | 24.0 |
| Drift · Same PSDP first action, direct MPC | 23 | 58 | 48.10 | 8.64 | 7.60 | 28.0 |

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

[Nominal cases](results/main/test.jsonl.gz) ·
[Drift cases](results/main/drift.jsonl.gz) ·
[Paired intervals and diagnostic telemetry](results/main/summary.json)

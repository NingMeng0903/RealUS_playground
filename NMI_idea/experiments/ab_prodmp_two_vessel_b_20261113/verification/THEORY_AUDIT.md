# ProDMP theory and main-result audit

2026-11-13. **The learned ProDMP reference is genuinely used and the declared motion class passes its sampled checks. The high-level result still supplies no material B benefit over fixed near plus the same live replanner.** This review covers the completed main experiment, the stated mathematical diagnostics, and the saved secondary exact endpoint-only comparison.

| Common learned-ProDMP executor | Exact correct accepted completion | Paired correct / 512 |
|---|---:|---:|
| Live stochastic MPC | 75.57601090% | 375 |
| B | 81.84842169% | 413 |
| Fixed central first near + live tail | 81.84589205% | 413 |
| Fixed +12-degree first near + live tail | 81.84451372% | 413 |

The B–MPC contrast is **6.27241079 percentage points**, comparing against MPC's executed value after actual updating/replanning. MPC's fixed-tail forecast is 70.55955728%. B wins 59 paired cases and loses 21 against MPC. B, fixed center, and fixed +12 have identical success/failure outcomes on every paired case; these arms have no wrong declarations in that replay. Exact modeled wrong-declaration probabilities are nonzero and remain in the main summary.

B exceeds fixed center by only **0.00252964 points**. Their exact mean differences are −0.001773 mm TCP, −1.499270 degrees turning, equal rail, and −0.000120 s modeled time, all inside the declared equivalence bands. This removes the earlier archived-path travel penalty but does not create a useful completion effect. Fixed +12 uses about **1.552553 degrees less turning than B**, at essentially identical TCP/time and only 0.00390797 points less success. Consequently neither a broad success advantage nor general motion-efficiency dominance is supported. Against legacy MPC, B uses +0.930078 mm TCP, −6.364065 degrees turning, and −0.588273 s modeled time; report that tradeoff instead of calling all resources identical.

## Scope of the motion result

The learned two-dimensional chart's lateral excursion is mapped through the static material surface. Rotation and rail use its longitudinal progress. Bidirectional endpoint-seeded IK is blended and projected back to the Cartesian reference. Thus the implementation uses the shape, rather than merely renaming an archived joint path. The 63/63 accepted commands are newly synthesized; their samples start and close at the declared endpoint joint states, including the stationary dwell. Maximum recorded sampled position/rotation residuals are 0.003961 mm and 0.002286 degrees; maximum sampled joint-rate fraction is 0.579155.

The matched endpoint-only mechanics file also reports 63/63 accepted commands. On advancing transitions the recorded maximum same-time Cartesian reference difference is **4.9015–6.1274 mm**; the maximum absolute TCP-length difference over all commands is **0.29751 mm**. This establishes a nontrivial reference-shape change and a small path-length change. It does not establish improved closed-loop task completion. Such a claim requires the corresponding matched policy evaluation.

The current ProDMP weights, endpoints, duration, and admissible directed command class are fixed. This is conditional runtime synthesis and high-level command selection, **not search over ProDMP weights or unrestricted continuous motions**. Execution uses sampled ideal kinematics, fixed rest-state closure, and nominal-plane sensing within the declared endpoint tolerance. It is not a test of arbitrary residual starts, actuator tracking error, measured tissue contact, or in-motion ultrasound. Runtime caching is explicit: 32 uncached cases performed 495 fresh synthesis/check calls; the remaining replay generated 34 new paths and made 7,257 exact-state/action reuses. All reported forecast/execution cost residuals are zero.

## Secondary exact motion ablation

The subsequent saved endpoint-only comparison replaces the learned 2-D mean weights with the existing endpoint-only weights and uses the resulting checked cost table. Both representations accept 63/63 commands. I checked its three input hashes and independently reaggregated its 1,166 prescan records. Every stored root action and B/MPC forecast/executed value agrees exactly with the learned-ProDMP record. All six high-level policies have the same exact success, wrong-declaration, frame-count, modeled-time, and rail results between executors.

Accordingly the success interaction

```text
[J(B,learned)-J(MPC,learned)]
 - [J(B,endpoint-only)-J(MPC,endpoint-only)] = 0.
```

Learned shapes add 0.806450 mm mean total TCP travel for B, 0.790323 mm for MPC, and 0.806816 mm for fixed center; the turning differences are below 0.0003 degrees. **The learned shape changes motion but supplies no completion advantage in this test.** This is consistent with fixed final image poses, no in-motion sensing, and unchanged feasible decisions. It is a secondary exact model ablation, not an additional physical execution study or evidence against ProDMP in other tasks.

## Why the practical conclusion remains unchanged

With intrinsic independent target detection `p=.84`, at most two near attempts, and one terminal far attempt,

```text
P(success) <= p [1-(1-p)^2] = .818496.
```

This optimistic bound already grants free identity and ideal visibility/motion. The fixed-center control is only **0.00370795 percentage points** below it. ProDMP cannot create five points of sensing headroom when that sensor/task class remains fixed. The major effect remains optional near retry. The latest-near-only eligibility restriction also remains: a forced extra near capture replaces the earlier eligible packet, whereas conditional retry avoids replacing a good one.

A post-primary forecast-ranking sensitivity check gives the following optimistic B gap after selecting the best realized `G` among roots within `epsilon` of MPC's best forecast `F`:

| Forecast tolerance in success probability | Remaining B gap, percentage points |
|---|---:|
| `1e-12` | 3.8481834 |
| `1e-8` | 1.1661883 |
| `1e-6` | 0.000266665 |

This uses `G` to identify the favorable root and is **not an executed stronger MPC policy**. It shows sensitivity to very small forecast rankings. A prospectively specified tie rule that favors a required near image among epsilon-optimal `F` actions has forecast regret at most epsilon, but no automatic bound of epsilon on executed regret. Its final .87 joint acceptance need not change. The already executed fixed-first control is the decisive evidence here, so no posthoc tie tuning is necessary.

## Independent verification

I checked the seven main source hashes against the summary, independently reaggregated all 1,166 stored prescan branches, and reproduced B, MPC forecast, and MPC executed values. Omitted prescan probability is `2.60895e-11`. I recomputed TCP/rotation/rail/time from all 63 recorded motion traces with zero discrepancy, verified endpoint and dwell closure, and checked that all 34 archived executed paths exactly match their newly generated candidate traces.

For all **3,072** policy-case replay paths, I reconstructed the posterior and final joint near/far slot decision from public packets, verified the selected slots against truth, checked the .87 threshold, freshness/order, legal commands, capture counts and resource caps, and reaggregated the outcomes. Maximum cost-ledger discrepancy was `1.42e-14`. These checks reproduce the 59/21 MPC discordances and zero discordances against either fixed-center or fixed-plus12. They audit the recorded kinematic model and decision arithmetic; they do not constitute physical robot execution.

The supported result is therefore: **the static task can use these newly synthesized learned-ProDMP motions, while the simple fixed-near controller retains essentially all available task benefit.**

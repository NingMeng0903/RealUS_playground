# Session 005 recorded visual audit

Read-only analysis of successful attempts from `session.json`: L_DtP/003, C_DtP/002, S_DtP/001, L_PtD/002, C_PtD/004. The original audit was read-only; subsequent safe production changes add attribution diagnostics and the task-supply energy row. No hardware execution occurred. Reproduce with the Genesis Python environment, `analyze.py`, `edges.py`, and `work.py` in this directory. `metrics.json` contains all/first-half/late-half/last-quarter metrics; `edge_metrics.json` includes exact-frame edge selections; `work.json` contains work integrations.

## What happened in the late half

| Scan | Scan half duration | Differential request duration | Median L/C/R confidence | QP-minus-nominal wy integral | Published total wy integral | Measured relative tool-y rotation |
|---|---:|---:|---|---:|---:|---:|
| L DtP | 17.42 s | 0 s | .994/.994/.996 | 0° | +4.93° | +4.77° |
| C DtP | 17.70 s | 0 s | .994/.994/.996 | 0° | +8.70° | +8.36° |
| S DtP | 18.59 s | 3.27 s | .994/.995/.994 | +0.456° | +5.406° | +5.096° |
| L PtD | 18.53 s | 13.34 s | .786/.965/.997 | −0.937° | −10.313° | −9.184° |
| C PtD | 18.51 s | 10.60 s | .857/.968/.996 | −0.462° | −6.843° | −5.936° |

These integrals separate requested total command from nominal command. QP-minus-nominal includes every QP effect; it is not a visual-only counterfactual. Measured rotation includes baseline torque control and path motion. Net SO(3) tool-y rotation is also not mathematically identical to integrating body wy over a changing orientation.

Continuous permission was present on 100% of late-half cycles, with zero exhaustion, zero invalid-image cycles, and zero force-gate-zero cycles. For every requested and successfully published late-half sample, final wy had the requested direction. Maximum final/candidate wy discrepancy was below 0.00005°/s. The command was not lost in publication.

S DtP's force gate was partially reduced on 75.5% of late-half cycles; L PtD's on 11.8%. Neither scan was fully force-gated. Aperture constraints were active for 2.65% of requested S cycles and no requested L cycles. Remaining energy above reserve/obligations was at least 0.328 J (S) and 0.616 J (L), so neither energy exhaustion nor the former bounded episode explains the residual black band.

## Exact saved-image evidence

See `RH_Per_S_DtP_edges.png` and `RH_Per_L_PtD_edges.png`. Every displayed edge frame has an exact online frame-index match; recomputed L/C/R confidence equals the logged values exactly. Cyan marks the top-22%-depth boundary; green lines mark current lateral window boundaries. No image flip or enhancement was applied. The grayscale `<20` statistics are descriptive, not physical contact ground truth.

- S DtP +25.85 s, frame 129689: broad right dark band; qL/C/R=.994/.990/.796, differential request .165 mm/s, nominal→candidate wy +.665→+.940°/s.
- S DtP +27.32 s, frame 129733: broad right dark band persists; q=.993/.993/.896, absolute L/R difference .09748 below the .10 deadband, gate=1, request=0, nominal=candidate wy=0. This is a concrete visible-band/current-policy mismatch. The top-ROI confidence is lower on the right, but its broad window average falls inside the balance deadband and stays above c_min=.8.
- L PtD +23.99 s, frame 131952: broad left dark band; q=.543/.784/.982, request .755 mm/s, nominal→candidate wy −.896→−1.322°/s.
- L PtD +22.62 s, frame 131911: q=.656/.922/.998, request .538 mm/s, nominal→candidate wy −.964→−.990°/s. Baseline rotation already nearly pays the current total-velocity task.
- L PtD +33.92 s, frame 132250: narrower remaining left edge shading with q=.994/.997/.997, zero request and wy. The top-22% window is nearly uniformly high confidence despite visible shading below it.

The current statistic only directly averages top 22% depth and excludes the outermost 4% on both sides. Deeper image features can influence the random-walk solution, but they are not directly represented by the top-ROI average. Flat top-boundary confidence and broad lateral means can hide narrow or deeper defects. These recordings do not establish which visible dark regions are recoverable contact loss versus anatomy/attenuation; in particular, deep shadows under bright reflectors should not become angle commands simply because they are black.

## Total versus incremental visual target

The present differential term constrains `sign * differential_row @ V`, with a soft shortfall, so nominal torque rotation counts toward the visual request. In requested late-half cycles, nominal alone already covers the request in S 27.0%, L 30.4%, C PtD 39.3%. The median requested equivalent wy is .149/.678/.614°/s, whereas median absolute QP-minus-nominal wy is .112/.0587/.0227°/s. Current optimized total motion nearly meets the intended total-velocity target; persistent poor image quality does not imply a numerical failure.

An incremental objective was implemented temporarily and rejected after testing the actual torque-state interconnection. `TorqueTilt.commit_applied` stores total accepted omega; the next nominal admittance starts from that value. Adding the same visual increment to each next nominal therefore accumulates the visual correction. This risk is absent from independent single-state checks.

`recursion.json` uses actual TorqueTilt prepare/commit and QP over 600 steps at 5 ms, fresh identical confidence every 50 ms, engaged 4 N contact, zero slack and fixed ample energy. For zero torque, total-velocity fusion settles at .800°/s and 2.396° over 3 s; the incremental variant reaches the .22 rad/s rate cap (12.605°/s) and 37.042°. With aligned −.04 Nm torque, total fusion ends at 3.069°/s versus incremental at the cap. With opposing +.04 Nm torque, total ends at .771°/s versus incremental at the cap. This synthetic test isolates command-state feedback, not plant/image dynamics.

The final production decision is to retain total-velocity fusion. The optional incremental mode was removed entirely, its patch/tests archived under `REJECTED_increment_experiment.patch` and `increment_experiment_snapshots.json`. `increment_replay.json` remains evidence of the rejected single-state experiment and must not be interpreted as proving it better. Its script requires the archived experimental code. Distinct nominal/total/increment diagnostics remain for attribution, without changing the velocity objective.

For S frame 129733, the issue arises before optimization: .09748 imbalance falls below .10. A .03 initial engineering deadband is supported by 713 distinct feature frames from stable late L/C DtP sections: |R−L| p95=.00664, p99=.01250, maximum=.01658. All five early +2…8 s sections have maxima ≤.00559. .03 is a reasonable initial empirical setting, not a physical calibration or statistically unique optimum. Consecutive unique frames are still temporally correlated. Keep ROI and registration unchanged. A deep or narrow band with all three q≈1 remains invisible regardless of incremental versus total fusion or this deadband; anatomy and narrow-window noise matter before changing the image statistic.

With the declared `image_x_sign=-1`, image right maps to negative tool-x. Tool-y rotation gives endpoint normal velocity `vz−x*wy`; hence right-low confidence requests positive wy, which increases compression on the mapped right endpoint and decreases it on the opposite side. The algebra and mirrored tests are consistent. Actual image-to-TCP mapping is declared in configuration, not empirically certified by this audit. Delayed confidence improvement while the probe also translates cannot establish a causal benefit from rotation. Existing fusion is a reasonable shared velocity objective with force/aperture/slew/energy bounds, not proof of acoustic optimality.

## Approximate work accounting

`work.py` uses W=negative recorded raw control wrench and successfully published final V, common dt to the next control capped at path_done, and skips controls without successful publication. The integrated final net work differs from recorded logical epoch work by at most .000218 J across the five successful attempts. This is logical model arithmetic, not measured physical passivity.

For the proposed nominal-power allowance `S=max(0,−W·Vnominal)`, charge only excess outgoing power `max(−W·Vfinal−S,0)` and credit actual positive final power. This prevents allowance from filling a tank while the final command is zero. Applying that arithmetic to existing successful commands yields:

| Scan | Actual nominal supply used | Excess outgoing work | Largest cumulative drawdown | Peak excess 50 ms liability |
|---|---:|---:|---:|---:|
| L DtP | .39926 J | .02122 J | .02024 J | .000635 J |
| C DtP | .32806 J | .01811 J | .01748 J | .000353 J |
| S DtP | .39809 J | .02710 J | .02634 J | .000544 J |
| L PtD | .13775 J | .006292 J | .003057 J | .000251 J |
| C PtD | .08934 J | .002806 J | .001252 J | .000144 J |

This is a counterfactual ledger calculation on unchanged commands, not a replay of a changed controller. It sizes recorded excess demand, not all future scans or changed incremental-angle objectives. Whole-control outgoing angular work of final-minus-nominal is only .0000774 J (S) and .000104 J (L); corresponding total outgoing incremental work is .02771 J and .007377 J. Recorded excess demand is dominated by translation. P95 absolute raw y-torque is .0523 Nm (S) and .0462 Nm (L).

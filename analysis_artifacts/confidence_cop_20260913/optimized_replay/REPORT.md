# Frozen-input confidence/CoP replay: 49 saved scans

The replay covers 241,127 recorded control proposals, including 196,455 moving proposals. It recomputes 31,961 unique control-used JPEG frames by exact frame index and publisher identity; 0 requested frames are unavailable. Maximum difference between recomputed and recorded LCR quality is 0.

The result compares recorded old commands with two new command calculations on the same recorded inputs. New candidates are treated as successful ideal inner execution when the actual command-budget admission accepts them. Reservation and commit use the same historical proposal time, so publication has zero modeled compute latency. The online QP call has no real force-source deadline attached. Accepted therefore means candidate/model admission, not a guarantee that the live controller would send before its deadline or avoid stopping. It does not predict changed ultrasound images, forces, anatomy, scan duration, or physical stability. Recorded old energy settled actual final command models, so differences from new ideal-command tanks are descriptive rather than a controlled physical performance comparison.

Each new mode starts one .100 J tank per attempt with .150 J capacity and .050 J stopping reserve. Approach ticks remain in the tank history. Task authorization contains frozen loading Z and scan components only; the new confidence omega and explicit CoP normal companion do not enter the source directly. Recorded baseline Z is a historically mixed loading proxy, and unused requested-source authorization can fund other final motion. Therefore this does not prove the absence of all indirect cross-funding. Actual CommandBudget settlement, reservation, admission, and commit are used. No future recovery is credited. If a lease fault occurs, strict continuous coverage ends; the extra all-path geometry calculation explicitly starts a counterfactual segment with the same settled balance. Such a restart is not runtime recovery or a claim that hardware stopped.

The normal input is the recorded baseline Z command. Logs do not contain the independent previsual force-controller state, so this is a frozen loading proxy, not a reconstruction of the new force feedback trajectory. Recorded pose, wrench, path components, alpha preference, and proposal timing remain fixed. Proposal time is recovered from the logged exported task expiry minus the recorded QP certificate horizon; rejected reviews expose it directly as created_time_s. This is exact up to floating-point roundoff because the recorded active runtime supplies no external mechanical expiry. Proposals without a review use the explicitly approximate JSON emission time minus compute elapsed time; counts and control IDs are recorded in each summary. Source timestamp plus source age denotes earlier ingress and is not used as proposal time. Contact enabling uses moving progress and compressive Fz ≥ .8 N because the actual contact latch is not logged. The angle origin is the first recorded pose; the true contact-acquisition reset history is unavailable. No inner IK, rail adjustment, publication delay/rejection, or source-filter counterfactual is simulated. Hard QP limits are never relaxed to improve coverage.

Clock coverage across all attempted proposals: exact_exported_expiry_minus_horizon: 240,956, exact_rejected_created_time: 171.

Three moving proposals temporarily paused visual feedback because their recorded image was stale: 007/LH_Per_L_PtD control 1722 (302.406 ms image age), jiaqi/LH_Per_L_PtD control 2712 (300.703 ms), and pei/LH_Per_C_DtP control 2180 (304.267 ms). Each had feature=null and transient_stale status, with reference progress increasing on the next sample. They were genuine progressing samples, not endpoint convergence; feedback recovered about 8 ms later. These were one-tick visual-task pauses, not hard controller stops. Exact records are in moving_image_pauses.json.

The image task uses I=.051, D=.22, speed scale .002/.031 rad/s, image sign −1, centroid deadband .03, LCR threshold .8, and the existing 4→4.5 N force gate. Velocity .28 rad/s and angular acceleration 3 rad/s² are clamped by each recorded force-controller limit (typically 2 rad/s²). The measured-angle limit is 150°. Both lateral qualities meeting .8 makes the held image target zero; dynamics still decay from prior accepted state. CoP is −My/Fz, used only inside the declared ±25 mm face. It changes the final normal preference for the chosen angle, and is not a measured compliance center or a guarantee of constant force.

## Aggregate checks

- confidence_angular_v1: 196,455/196,455 moving commands accepted; 49/49 scans have full strict coverage; 0 explicitly segmented restarts; minimum tank 0.100000 J; mean replay compute 1.81 ms. These CPU timings are not a realtime execution certificate.
- confidence_cop_v1: 196,455/196,455 moving commands accepted; 49/49 scans have full strict coverage; 0 explicitly segmented restarts; minimum tank 0.100000 J; mean replay compute 1.81 ms. These CPU timings are not a realtime execution certificate.

## Timing evidence and limits

timing_risk.csv reports conservative count bounds for proposal source age plus replay compute exceeding 15 ms, and for replay compute exceeding 5 ms. Per-tick cost vectors were not retained, so a measured cost cannot honestly be paired with a particular historical source age. The lower/upper deadline counts use each scan’s observed minimum/maximum cost; a separately labeled scenario uses p95 cost for every tick. Compute-count bounds use the stored minimum, median, p95, and maximum. These are bounds on the observed replay calculation, not hardware stopping rates. Scheduling and other concurrent processes affect timings; replay Python/bookkeeping is included, while force-controller reconstruction, native IK, transport, and complete publication latency are absent. P95 is not a complete control-cycle budget.

- confidence_angular_v1: 15 ms source-age-plus-compute count bound 0–0 / 196,455; compute over 5 ms count bound 0–0. The constant-p95-cost deadline scenario flags 0 samples.
- confidence_cop_v1: 15 ms source-age-plus-compute count bound 0–2658 / 196,455; compute over 5 ms count bound 0–771. The constant-p95-cost deadline scenario flags 0 samples.

## All saved paths

Order in three-value cells: recorded old / angular / angular+CoP. Two-value cells: angular / angular+CoP. Weak-side speed is the signed geometric normal velocity averaged over L/R windows whose recorded quality is below .8; positive means compression in the declared face convention. It is not observed acoustic improvement. Angular error is against the dynamic image target; pairing residual is final Z minus recorded loading Z minus CoP×omega. Deferrals count all attempt proposals; the remaining command statistics use moving proposals and accepted commands. An em dash means no applicable samples.

| Saved scan | Moving ticks | Tank min J old/A/C | Weak speed mm/s old/A/C | Angular error p95 mrad/s A/C | CoP pair residual p95 mm/s | Deferrals A/C | Strict coverage % A/C | Compute p95 ms A/C | Direction conflicts / nondeadband unique-weak frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 001/RH_Per_C_DtP | 4838 | 0.098/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/0 |
| 001/RH_Per_C_PtD | 4845 | 0.100/0.100/0.100 | 1.196/1.014/1.040 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/386 |
| 001/RH_Per_L_DtP | 4878 | 0.100/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.81/1.81 | 0/0 |
| 001/RH_Per_L_PtD | 5027 | 0.100/0.100/0.100 | 0.847/0.756/0.779 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/473 |
| 001/RH_Per_S_DtP | 4857 | 0.098/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/0 |
| 003/LH_Per_C_DtP | 4126 | 0.100/0.100/0.100 | -0.170/-0.583/-0.586 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.84/1.83 | 0/183 |
| 003/LH_Per_C_PtD | 4582 | 0.099/0.100/0.100 | 1.330/1.174/1.164 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/312 |
| 003/LH_Per_L_DtP | 4044 | 0.100/0.100/0.100 | -0.173/-0.747/-0.748 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/44 |
| 003/LH_Per_L_PtD | 3918 | 0.100/0.100/0.100 | 0.939/0.947/0.946 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/204 |
| 003/LH_Per_S_DtP | 4801 | 0.100/0.100/0.100 | 0.532/0.023/0.046 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/213 |
| 003/LH_Per_S_PtD | 4294 | 0.097/0.100/0.100 | 1.677/1.344/1.353 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.83 | 0/174 |
| 003/RH_Per_C_DtP | 3862 | 0.100/0.100/0.100 | 0.153/-0.168/-0.166 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.83 | 0/167 |
| 003/RH_Per_C_PtD | 3615 | 0.100/0.100/0.100 | 1.073/0.800/0.800 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/141 |
| 003/RH_Per_L_DtP | 3951 | 0.096/0.100/0.100 | 0.020/-0.269/-0.271 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/98 |
| 003/RH_Per_L_PtD | 3682 | 0.100/0.100/0.100 | 0.659/0.580/0.586 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/162 |
| 003/RH_Per_S_DtP | 4202 | 0.100/0.100/0.100 | 0.240/-0.105/-0.094 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.84/1.84 | 0/227 |
| 003/RH_Per_S_PtD | 3920 | 0.100/0.100/0.100 | 1.233/0.916/0.930 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/168 |
| 007/LH_Per_C_DtP | 3827 | 0.100/0.100/0.100 | 0.087/-0.520/-0.517 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.87/1.87 | 0/136 |
| 007/LH_Per_C_PtD | 3308 | 0.100/0.100/0.100 | 0.934/0.881/0.887 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/55 |
| 007/LH_Per_L_DtP | 3325 | 0.100/0.100/0.100 | -0.116/-0.337/-0.338 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/32 |
| 007/LH_Per_L_PtD | 3327 | 0.100/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/0 |
| 007/LH_Per_S_DtP | 3418 | 0.099/0.100/0.100 | -0.254/-0.785/-0.785 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.87/1.87 | 0/8 |
| 007/LH_Per_S_PtD | 3460 | 0.100/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/0 |
| 007/RH_Per_C_DtP | 3595 | 0.100/0.100/0.100 | -0.058/-0.802/-0.795 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.87 | 0/35 |
| 007/RH_Per_C_PtD | 3539 | 0.100/0.100/0.100 | 1.564/1.259/1.265 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/29 |
| 007/RH_Per_L_DtP | 3581 | 0.100/0.100/0.100 | -0.364/-0.842/-0.843 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.86/1.86 | 0/4 |
| 007/RH_Per_L_PtD | 3549 | 0.100/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.87/1.87 | 0/0 |
| 007/RH_Per_S_DtP | 3809 | 0.100/0.100/0.100 | 0.282/-0.337/-0.319 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.87/1.87 | 0/148 |
| 007/RH_Per_S_PtD | 3782 | 0.100/0.100/0.100 | 2.559/1.653/1.688 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.88/1.88 | 0/95 |
| jiaqi/LH_Per_C_DtP | 4154 | 0.100/0.100/0.100 | -0.009/-0.347/-0.360 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/65 |
| jiaqi/LH_Per_C_PtD | 4712 | 0.091/0.100/0.100 | 1.179/1.056/1.057 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/252 |
| jiaqi/LH_Per_L_DtP | 4159 | 0.100/0.100/0.100 | -0.216/-0.623/-0.623 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/2 |
| jiaqi/LH_Per_L_PtD | 4147 | 0.099/0.100/0.100 | —/—/— | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.83 | 0/0 |
| jiaqi/LH_Per_S_DtP | 4334 | 0.100/0.100/0.100 | 0.408/0.040/0.045 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/67 |
| jiaqi/LH_Per_S_PtD | 4809 | 0.098/0.100/0.100 | 1.382/1.125/1.066 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/193 |
| jiaqi/RH_Per_C_DtP | 3949 | 0.095/0.100/0.100 | -0.177/-0.883/-0.881 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/256 |
| jiaqi/RH_Per_C_PtD | 4068 | 0.100/0.100/0.100 | 1.382/1.263/1.277 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/380 |
| jiaqi/RH_Per_L_DtP | 3906 | 0.100/0.100/0.100 | -0.208/-0.587/-0.584 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/49 |
| jiaqi/RH_Per_L_PtD | 3755 | 0.100/0.100/0.100 | 1.349/1.335/1.344 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/282 |
| jiaqi/RH_Per_S_DtP | 4335 | 0.087/0.100/0.100 | -0.014/-0.650/-0.645 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.83/1.83 | 0/285 |
| jiaqi/RH_Per_S_PtD | 4317 | 0.100/0.100/0.100 | 1.581/1.398/1.423 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.81/1.81 | 0/332 |
| pei/LH_Per_C_DtP | 3080 | 0.100/0.100/0.100 | 0.251/0.029/0.020 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/95 |
| pei/LH_Per_L_DtP | 3012 | 0.100/0.100/0.100 | 0.028/-0.001/-0.001 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.81/1.81 | 0/2 |
| pei/RH_Per_C_DtP | 4063 | 0.100/0.100/0.100 | 0.742/0.030/0.057 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/321 |
| pei/RH_Per_C_PtD | 3910 | 0.100/0.100/0.100 | 1.330/1.052/1.075 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.83 | 0/540 |
| pei/RH_Per_L_DtP | 3893 | 0.100/0.100/0.100 | 0.089/-0.243/-0.236 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/186 |
| pei/RH_Per_L_PtD | 3692 | 0.100/0.100/0.100 | 1.011/0.889/0.908 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.84/1.83 | 0/482 |
| pei/RH_Per_S_DtP | 4222 | 0.100/0.100/0.100 | 0.672/0.065/0.079 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.82/1.82 | 0/323 |
| pei/RH_Per_S_PtD | 3976 | 0.100/0.100/0.100 | 1.322/1.029/1.057 | 0.000/0.000 | 0.000 | 0/0 | 100.0/100.0 | 1.76/1.76 | 0/346 |

Direction conflicts compare the raw centroid request before M-D dynamics, acceleration limits, and force gating, only where exactly one lateral window is weak and the centroid lies outside the .03 deadband. With image sign −1, left-only weakness expects negative tool omega-Y and right-only weakness expects positive. A disagreement is a direction conflict between full-ROI mass and regional quality, not automatically an algorithm error: anatomical shadows and confidence outside the three windows can move the centroid. direction_conflicts.csv includes deduplicated frame counts, held weak/conflict time, signed mean request, mean and p95 absolute request, and conflict ticks with positive force gate. The CSV held_s columns sum recorded command_slew_dt_s at qualifying samples: they are sampled-time weights, not proven continuous frame-hold durations. These comparisons exclude dynamic lag as a cause of disagreement.

Full precision and additional costs, target achievement, reasons, and coverage are in comparison.csv, aggregate.json, results/<group>/<scan>/summary.json, and commands.npz. Each result records source hashes. Cache manifests identify every saved file, selected attempt, raw file, feature config, and feature-source hash; each cached frame records the exact raw index, JPEG SHA-256, normalized centroid, and LCR reproduction error.

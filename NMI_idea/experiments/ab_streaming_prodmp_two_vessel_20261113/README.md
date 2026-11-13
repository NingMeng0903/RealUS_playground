# Streaming ultrasound feedback with learned ProDMP

The `run.py atlas` entry evaluates the fixed material chart and its attached probe orientation on a supplied SMPL-X shape before streaming. The cross-shape study provides the transport equations, commands, and paired logs. [Cross-shape scan](CROSS_BETA.md)

Use `run.py atlas --motion ou` for the paired local vessel-motion benchmark. For one execution with the online dynamic map, add `--arm mpc_ou`. The dynamic study includes a short video from the saved closed-loop trace. [Dynamic residual and results](DYNAMIC_RESIDUAL.md)

This mechanism experiment uses the previous static 15° Among_US artery and vein and fixes longitudinal progression and probe orientation. It tests whether ultrasound can change the lateral motion before a segment ends, and whether learning from those corrections improves the result. The previous endpoint experiment captures after motion and dwell. Its ProDMP fits analytical curves generated without image observations. [Previous execution and result](../ab_prodmp_two_vessel_b_20261113/README.md)

## Demonstrations and execution

`SECTIONS.json` contains two vessel sections at 65 positions between stations 3 and 5. The simulator interpolates section centers along the scan and recomputes their image coordinates at the current probe position every 100 ms. A shared case-level offset and calibration bias persist across the scan. A Markov dropout state for each vessel makes misses persist across frames. The filter maintains the joint distribution of effective offset and both dropout states and marginalizes the assignments of the anonymous centroid packet. Simulator roles and physical centers are saved for audit and used by the evaluator. [Geometry provenance](SECTIONS.json) · [Sensor and association](sensor.py)

The visual-servo teacher runs 128 full simulated scans. A high-confidence arterial return supplies the lateral image error, which corrects the nominal lateral velocity under speed and acceleration limits. A missed or ambiguous return leaves the nominal feedforward velocity. The teacher logs pose, velocity, time, centroids, association probabilities and applied corrections. Its four one-second segments per scan produce 512 examples for fitting the integrated ProDMP shape. The fit uses the executed lateral positions and velocities and preserves each demonstration segment's endpoints. Test scenarios use a separate RNG seed. The eight-case integration smoke run uses the first eight scenarios from that seed, with a separate six-scan fit. [Demonstration generation](run.py) · [Saved demonstrations and fitted weights](verification/DEMONSTRATIONS.npz)

When a qualifying image arrives, the controller uses the estimated offset and measured lateral residual to update the goal. It reconditions the remaining trajectory from the realized lateral state $(x_k,v_k)$ without extending the segment deadline. For phase $s$ and remaining time $T$, the integrated basis satisfies $B(0)=B'(0)=0$. Thus

$$
x(s)=g+e^{-\lambda s}\big[(1+\lambda s)(x_k-g)+sT v_k\big]+B(s)w
$$

starts at $x_k$ with physical velocity $v_k$. The endpoint equations for $w$ fix the new goal and terminal velocity. The learned shape lies in their null space. Both controllers try the same bounded goal revision and retain a feasible existing suffix when the revision fails the sampled constraints. The controller executes the next 100 ms of the plan before the next update. [Reconditioning implementation](primitive.py) · [Integrated ProDMP basis](https://arxiv.org/abs/2210.01531)

## Paired comparison

All arms assimilate 41 pose-dependent frames over the same 4 s. They share the initial state, anatomy, dropout and measurement-noise tapes, and the 18 mm/s lateral speed and 100 mm/s² sampled acceleration bounds. The three arms vary the shape and when the current plan can change. Correct completion requires a new accepted arterial image within 3 mm of the image center in each of six spatial bins, confirmed by the physical lateral error and evaluator identity, and no accepted wrong-vessel image. The public declaration uses image measurements and a 0.95 arterial association threshold. A false completion is a public completion that fails the truth-based test.

| Controller | Correct completion | Wrong vessel | False completion | Mean physical lateral error | Mean TCP travel |
|---|---:|---:|---:|---:|---:|
| Learned ProDMP, adjust at segment boundaries | 174/256 | 0 | 17 | 0.962 mm | 30.050 mm |
| Endpoint-conditioned ProDMP, adjust within segment | 213/256 | 0 | 3 | 0.752 mm | 31.008 mm |
| Feedback-trained ProDMP, adjust within segment | 214/256 | 0 | 1 | 0.691 mm | 30.642 mm |

With the learned shape held fixed, letting images adjust the ongoing trajectory improves completion by **15.63 points** (40 wins, 0 losses, paired bootstrap 95% interval 11.33 to 19.92 points). TCP travel rises by 0.591 mm. The three groups have the same 6.483° of planned rotation. With the feedback schedule held fixed, the learned shape changes completion by **0.39 points** (2 wins, 1 loss, interval −0.78 to 1.56 points). It lowers mean physical lateral error by 0.061 mm and mean TCP travel by 0.366 mm. The learned streaming controller makes an average of 28.23 within-segment revisions that change the next frame's lateral pose by at least 1 µm. The measured lateral reference handoff residuals are 0 mm and at most $1.8\times10^{-15}$ mm/s. [Paired cases and intervals](verification/SUMMARY.json) · [Per-frame and per-tick traces](verification/TRACES.npz)

SOL independently reconstructed the outcomes and motion costs for all 768 arm-case records. A separate check lifted all 20 ms samples of the first six paired cases to fixed-rail RM75 kinematics, after separately positioning the start pose. All 18 paths passed the sampled FK, joint-limit, joint-rate and robot/probe self-collision checks. The joint-rate peak was 3.26% of the limit. ASTRA reviewed the causal controls, the conditioning equations and the difference between incorrect identity and false coverage. [Kinematic check](verification/KINEMATICS.json)

The evidence supports using images during the segment in this simulation. The fitted shape shows a small tracking and travel benefit. Its completion advantage is unresolved. The task uses one static registered template, an ideal chart-space plant, synthetic anonymous centroid packets, immediate frame availability, and one fixed high-level scan. Claims about actual B-mode, contact, tracking hardware, changing the scan direction, and the decision module B require their own evaluation.

## Reproduce

The benchmark ran from the repository root with Python 3.10, NumPy 2.2.6 and SciPy 1.15.3.

```sh
exp=NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113
OPENBLAS_NUM_THREADS=1 python "$exp/run.py" --out /tmp/two-vessel-streaming
python "$exp/check_kinematics.py" --traces /tmp/two-vessel-streaming/TRACES.npz --out /tmp/two-vessel-streaming/KINEMATICS.json
```

The second command uses the repository's RM75 capability runtime. The first command needs only the saved derived section table. `extract_sections.py` documents the optional regeneration from the local registered source mesh. The archived binaries contain only synthetic trajectories and observations and no raw meshes or ultrasound recordings.

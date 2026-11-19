# Why the fixed `cem_4` recovery succeeds

This audit replays the 128 fresh startup scenes and eight disturbances per scene from the [frozen comparison](STARTUP_B_WITHOUT_FORECAST_GATES.md). It varies only the first one-second request, then uses the same MPC with measured observation updates and replanning. The motion checks, within-segment ultrasound feedback, candidate image retention, delayed identity confirmation and absolute execution limits are shared. Correct completion requires a correct recorded target image from each of the six required scan intervals and reaching the scan endpoint. Centre error remains a separate metric.

Every opening scout ended at the same probe pose, with the target missing from all 11 scout frames. The synthetic shadow is strongest near the start and weakens with lateral displacement. The fixed request was selected on separate development scenes. Its name is inherited from the library. CEM and the budget-matched structured random search produced identical fitted demonstrations in the [source search experiment](SEARCH_EFFICIENCY.md).

## Isolate the recovery motion

`cem_4` requests 2.596 mm of longitudinal progress, +6.528 mm of lateral displacement and +1.095 mm of shaping at the interior waypoint. Its longitudinal position changes from 2.0 to 4.596 mm, inside the first required interval [2, 7) mm. Each motion variant uses the same fitted ProDMP and passes the same feasibility checks. The stationary control uses the shared `wait` request. The probe acquires 10 frames per second.

| First request | Frames in first interval during first second, mean | First interval target seen in first second | First interval target seen by termination | Correct completion | Final wrong identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed `cem_4` | 10.00 | 939/1024 | 1024/1024 | **1024/1024** | 0 |
| Same endpoints, zero interior shaping | 10.00 | 916/1024 | 1024/1024 | **1024/1024** | 0 |
| Zero lateral endpoint displacement, original shaping | 10.00 | 37/1024 | 971/1024 | 957/1024 | 2 |
| Advance 7.8 mm, original lateral motion request | 5.00 | 340/1024 | 860/1024 | 854/1024 | 2 |
| Reflected lateral motion, original progress | 10.00 | 890/1024 | 966/1024 | 952/1024 | 1 |
| Stationary wait | 10.00 | 21/1024 | 926/1024 | 901/1024 | 4 |
| Learned B, prioritizing correct completion | 9.82 | 764/1024 | 1021/1024 | 1021/1024 | 1 |
| Learned B, original utility with relative gates removed | 9.66 | 807/1024 | 1022/1024 | 1017/1024 | 1 |

The lateral endpoint and modest longitudinal progress work together. With the same lateral request and faster progress, 164 of 170 failed runs never acquired the first interval's target. Waiting in the shadow also yielded few early detections. The positive lateral direction had more total first-interval recaptures and completions than its reflection. Removing the interior shape reduced early captures by 23 and preserved final completion in every replay. The fixed request averaged 35.92 mm travel, 5.94° rotation and 6.053 s of simulated execution. The zero-shaping version averaged 35.78 mm, 5.94° and 6.047 s. The faster variant averaged 38.32 mm, 7.27° and 6.510 s.

Completion includes the shared downstream controller. For 638 of the fixed-request runs, the final first-interval target image came from that first second. Of those, 318 images were first confirmed later, after additional observations.

## The three failures of completion-first B

All three reached the endpoint. Each missed the physical target at every first-interval acquisition. Two finished with that interval unfilled and one kept a wrong-identity image there. Indices below refer to the zero-based paired disturbance tape.

| Scene and tape | First request selected by B | Difference in predicted completion over `cem_4` | Closed-loop consequence |
| --- | --- | ---: | --- |
| `48610029`, tape 4 | `cem_5` | +3.80 pp | At 0.5 s after the scout its lateral position had changed by +2.34 mm, versus +4.37 mm for `cem_4`. It left the interval 0.1 s earlier, with no target image. `cem_4` obtained the target during MPC continuation while still inside the interval. |
| `48610038`, tape 7 | `right` | +0.47 pp | It finished with a wrong-identity image in the first interval. |
| `48620071`, tape 7 | `advance` | +1.92 pp | It left the interval 0.7 s after the scout, versus 1.5 s for `cem_4`. MPC never recovered its missing target image. |

These predictions came from the saved B model outputs. The fixed action was admissible in all 128 scenes. Completion-first B selected a different first request in 98 of 128 scenes, with the three recorded losses among those choices. Its completion difference against fixed `cem_4` was −0.29 percentage points with a scene-paired 95% bootstrap interval of [−0.68, 0.00]. For the original-utility ungated B, two of seven failures missed the first target entirely. Five obtained the target there and still failed later, through other missing evidence, a final wrong identity, or running out of scan time before the endpoint.

The mechanism tested here is a lateral sweep that still acquires inside the initial blind interval. The fixed-action comparison saturates on these fresh scenes. The prior cohort had two fixed-request failures among 1024 runs. The audit is a post-hoc replay at one knee angle and probe starting pose using synthetic detections. Its success counts do not transfer directly to other blind-zone geometries or to real scans.

## Reproduce and check

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B diagnose_cem4_acquisition.py --out results/cem4_acquisition_audit_v1 --jobs 24
```

The replay checks the frozen source archive digest, admissible menus, execution continuity, and matching archived outcomes for the three original policies. The output includes actual first-block trajectories and later MPC choices for every B failure. All runs stayed within the shared 67 mm and 8 s limits. [Summary](results/cem4_acquisition_audit_v1/summary.json) · [Per-run records](results/cem4_acquisition_audit_v1/replay.jsonl.gz)

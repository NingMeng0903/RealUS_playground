# ProDMP fidelity at fixed startup recovery requests

Replay the same 128 fixed-knee startup scenes and eight disturbances per scene from the [startup comparison](STARTUP_B_WITHOUT_FORECAST_GATES.md). Freeze each first request and replace its fitted lateral ProDMP reference with the original analytic waypoint reference used for demonstration generation. Match start and goal position and velocity and the entire longitudinal trajectory. Keep the same within-segment ultrasound feedback, posterior updates, delayed identity confirmation and full replanning MPC continuation. Both trajectory generators pass the shared motion checks for every swapped request. The probe uses synthetic observations throughout.

| First-action choice | Deployed execution | Original analytic reference where a fitted reference was used | Paired changes in correct completion |
| --- | ---: | ---: | ---: |
| Fixed `cem_4`, all 128 scenes | 1024/1024 | 1024/1024 | 0 |
| Completion-first B, all 128 scenes | 1021/1024 | 1021/1024 | 0 |
| Completion-first B, 100 scenes with fitted first references | 799/800 | 799/800 | 0 |

No pair disagreed on first-interval target recapture, final wrong identity or new interim wrong identity. The maximum nominal lateral discrepancy for `cem_4` was 0.113 mm. The maximum among the 100 fitted B choices was 0.244 mm across motion samples and 0.238 mm at image acquisition times. The mean scan travel and duration of fitted versus analytic fixed `cem_4` were 35.918 versus 35.959 mm and 6.053 versus 6.051 s. The respective B means were 36.782 versus 36.871 mm and 6.219 versus 6.220 s. Simulated scan duration excludes computation.

The only failure among B's 800 fitted-reference trials was the `cem_5` request at scene `48610029`, tape 4. Its analytic version also missed the target throughout the first required interval. Its maximum nominal discrepancy from the fitted reference was 0.102 mm. The other two B failures used the original `right` and `advance` primitives, among the 28 scenes in which B selected an existing primitive. The reference swap retained those primitives in the full-policy comparison. The fixed `cem_4` completed all three paired trials with either generator. [First-interval acquisition audit](CEM4_ACQUISITION_AUDIT.md)

The 61 source demonstrations were qualified via closed-loop replays. The fit uses their nominal lateral positions and velocities before feedback, with requested endpoint displacement and interior deflection as inputs. The longitudinal schedule is specified separately by the action request. During execution the shared feedback controller can revise the lateral reference as observations arrive. This test supports reproduction of the task outcome for the tested nominal requests. The least-squares motion objective and the endpoint constraints do not imply a completion guarantee for all other requests, scan conditions or action selection policies. [Fitting implementation](adaptive_teacher.py) · [Execution implementation](adaptive_controller.py)

## Reproduction

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B verify_startup_prodmp_fidelity.py --out results/startup_prodmp_fidelity_v1 --jobs 24
```

The run checks the frozen archive and weight digests, the original feasible candidate menus, and all archived deployed outcomes. The replay stores the actual outcomes by scene and disturbance. [Summary](results/startup_prodmp_fidelity_v1/summary.json) · [Paired records](results/startup_prodmp_fidelity_v1/replay.jsonl.gz)

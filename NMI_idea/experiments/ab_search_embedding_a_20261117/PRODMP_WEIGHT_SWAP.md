# ProDMP comparison at the same requested endpoints

This replay tests the motion generated for a given inspection request. It uses the archived primary first-only and resource-limited decisions, restricting the comparison to requests for a continuous waypoint. The selected first action, public posterior, longitudinal path, nominal lateral endpoints and velocities, and one-second segment duration are fixed. A changed reference can produce different observations and feedback corrections. Every curve uses the same frame-wise feedback and the same full MPC continuation. Each replay uses the original test scene and its three archived disturbance tapes.

The old renderer uses the pre-CEM side-scan weights, as in the earlier continuous search. If those weights fail the motion checks, it also tries the old controller's zero-residual ProDMP at the requested endpoint. Shortening the move would change the request. The new renderer uses the fitted displacement and midpoint-bow model. All arms can fall back to the admitted analytical waypoint for the same request, and the replay logs whether that fallback was needed. The two fit families use different features and training procedures. This experiment compares those two implementations.

| Archived requests | Old ProDMP passes, including its own zero-residual option | CEM-fitted ProDMP passes | Old ProDMP with shared waypoint fallback | CEM-fitted ProDMP with shared waypoint fallback |
| --- | ---: | ---: | ---: | ---: |
| Primary, all 50 requests | 12/50 | 48/50 | 63/150 | 63/150 |
| Resource-limited, all 15 requests | 0/15 | 15/15 | 21/45 | 21/45 |

The old zero-residual option added no admitted requests. Both ProDMP renderers were admitted for ten primary requests. On those 30 paired replays, each finished 11 scans and the completion decisions agreed trial by trial. With the shared fallback, completion agreed in every primary and resource-limited replay. The recorded endpoint of fresh query confirmation, final wrong record and new centered mistake also agreed throughout. Mean executed chart travel differed by +0.132 mm in the primary requests and −0.016 mm in the resource-limited requests, with identical simulated duration in both sets.

The fitted ProDMP can render substantially more of the requested exploratory segments as parametrized here. Recovering the same outcome with the already admitted analytical waypoint clarifies where the evidence supports a contribution. The fit transfers the motion request into the ProDMP representation. These replays observe no incremental correct completion from replacing the first curve. They use subsets selected by the archived B decisions and a fixed-pose simulator, with sampled chart limits and synthetic centroid packets. Improvements from jointly changing the observation selector and the motion renderer require a separate held-out comparison.

[Paired summary](results/adaptive_prodmp_swap_v2/summary.json) · [Per-case replay records](results/adaptive_prodmp_swap_v2/cases.jsonl.gz)

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B audit_prodmp_weights.py \
  --primary results/adaptive_paired_v2/cases.jsonl.gz \
  --limited results/adaptive_budgeted_v2/cases.jsonl.gz \
  --fit results/adaptive_discovery_v2/prodmp_nominal_fit.npy \
  --out results/adaptive_prodmp_swap_v2 --jobs 10
```

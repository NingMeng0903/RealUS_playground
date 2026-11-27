# Equal-endpoint test of the discovered waypoint shapes

The motion discovery experiment admitted 21 example paths for the ProDMP fit. Each had better correct completion than the MPC incumbent when verified on six fresh tapes, at least four completions, and no increase in new centered mistakes. This criterion establishes a useful candidate intervention in those cases. The end position, intermediate shape and realized feedback all varied together in that comparison.

This retrospective audit holds each selected example's nominal start and end positions, velocities, and duration fixed. The endpoint-only arm sets the signed midpoint bow to zero in the same quintic reference. Both arms keep the same feedback, motion checks and live MPC continuation. Simulation charges all executed motion and recordings. Both arms execute on twelve new paired tapes per selected scene. The script also replays the original shared-menu MPC and the fitted ProDMP version where admissible.

| Reference | Correct completion | Mean executed TCP path | Mean simulated duration |
| --- | ---: | ---: | ---: |
| Direct shared-menu MPC | 71/252 | 39.69 mm | 6.85 s |
| CEM endpoint with zero bow | 218/252 | 47.38 mm | 6.80 s |
| Full CEM path | 219/252 | 47.33 mm | 6.72 s |

The full path adds +0.40 percentage points to the equal-endpoint result, with a paired scene-bootstrap 95% interval of [−2.78, +3.57]. It improves completion in three scenes, worsens it in three and ties in fifteen. Both produce 24/252 new centered mistakes and 0/252 wrong final records. Their mean TCP travel difference is −0.05 mm [−0.77, +0.59]. The ordinary connection and unbranched scenes comprise twelve of the selected examples. Their endpoint-only and full-path results are 130/144 and 134/144, with a paired 95% difference interval of [0.00, +6.94] percentage points.

The ProDMP replacement passes the sampled motion checks in 19/21 scenes. Within those scenes the original and fitted references give 197/228 and 199/228 completions. These paths come from the fit's own development worlds with fresh disturbances. The earlier 144 admissible substitutions were evaluated on independent scenes and each representation gave 57 completions there.

The CEM endpoints retain most of the average completion benefit among admitted examples. A path shape can change the outcome in individual scenes, so this result does not establish equivalence or make the interior waypoint useless. These are privileged selections, with approximately 7.7 mm more travel than direct MPC on the audited set. This comparison does not establish a deployable or cost-matched MPC improvement. All measurements use the fixed-pose, synthetic-centroid chart test. The endpoint matching applies to the nominal reference. Different frames during execution can still induce different motion corrections and handoff positions.

For further skill data collection, store the public conditions that suggest an informative scan, the evidence or reacquisition required to stop it, and the downstream task outcome. A proposed shape becomes a candidate for skill learning when it contributes repeatable value beyond its endpoint and transfers to held-out scene geometries under the same sensor and motion constraints.

[Summary](results/adaptive_waypoint_audit_v2/summary.json) · [Per-case replay records](results/adaptive_waypoint_audit_v2/cases.jsonl.gz)

```sh
OPENBLAS_NUM_THREADS=1 python -B audit_waypoint_shape.py \
  --archive results/adaptive_discovery_v2/cases.jsonl.gz \
  --fit results/adaptive_discovery_v2/prodmp_nominal_fit.npy \
  --out results/adaptive_waypoint_audit_v2 --jobs 12 --replicas 12
```

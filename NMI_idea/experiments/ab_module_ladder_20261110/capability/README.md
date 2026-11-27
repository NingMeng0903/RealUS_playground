# Reusable capability evaluator

`capability.py` loads the original calibrated dIRD `ReachabilitySDF`, the original
robot URDF/TCP contract, Pinocchio FK, and the configured 35 robot/probe collision
pairs. No production source is modified and no hardware/session API is created.

```python
from capability import CapabilityEvaluator
c = CapabilityEvaluator(device='cuda')  # CPU is also supported; threads = 1
T = c.fk(q_current)
proxy = c.score(T, q_current[0])
answer = c.evaluate_pose(T_target, q_current, exact=True, native=False)
# Actual offline native controller state, including complete continuation history:
executed = c.evaluate_path(T_short_path, q_current, exact=False, native=True)
q_out = executed['q_out']
state = executed['native']['exit_state']
```

Inputs are homogeneous `T_world_tcp` matrices, and raw URDF joint coordinates
`[rail_y metres, joint_1..joint_7 radians]`. `T_world_rail` defaults to identity.
A common SE(3) relabelling of the TCP and rail frame preserves the score. A body
move relative to the robot changes the score. Rotation gradients are left world
SO(3) tangent derivatives, in radians, and are distinct from translations.

Proxy decisions compare **raw score with `m_safe=0.00244944728910923`**; the
centered threshold alone is incorrect. Proxy score is neither metric clearance
nor a safety certificate. `gradient()` returns translation, SO(3), and rail
derivatives. `score_tensor()` preserves torch autograd and batches on GPU.

Static evaluation searches all 8 SRS branches, swivel angles every 15 degrees
plus the public seed swivel, at the specified rail value. It checks actual FK,
URDF limits, and every configured self-collision pair. A found `q_solution` is
kinematic evidence. A failed finite search is `search_not_found`, never an
infeasibility or no-path certificate. Path evaluation also records sampled joint
interpolation collision/limits checks, including departure; this interpolation
is not a claim of Cartesian tracking. External anatomy/environment collision,
contact and hardware dynamics are outside this evaluator's static scope.

Only native replay returns `q_out`. This is actual native-controller `q_send`
integrated by the offline ideal measured-q plant. Native replay uses the existing
reusable scan profile. Continuation can pass `native_options={'initial_state':
state}` with matching actual `q_out`. Pure rotation with zero translation needs
an explicitly timed reference, as required by the existing scan replay.

## Frozen public diagnostic

`frozen_public/protocol.json` and `candidates.json` were written before scoring or
robot-validity labels. They contain 86 deterministic body-relative TCP queries
from a previously public joint anchor and one fixed joint-space variation. No
anatomy, hidden task reward, preference table, or post-label candidate selection
is used. The protocol includes checkpoint, calibration, URDF, collision, and
source hashes. Run with the supplied interpreter and single-threaded numerical
libraries:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /media/camp/EXT_DRIVE/envs/genesis/bin/python run_diagnostic.py --reuse-frozen --native
```

53/86 endpoint queries found an actually checked robot-valid configuration; 33
remain unresolved by the finite search. Counts found at endpoint-query budgets:

| Ordering | 8 queries | 16 queries | 32 queries |
|---|---:|---:|---:|
| Original dIRD descending score | 8 | 16 | 32 |
| Geometry-only shortest relative displacement | 8 | 16 | 29 |
| Uniform frozen manifest order | 5 | 10 | 23 |
| Random, mean of seeds 0..31 | 5.21875 | 10.03125 | 20.09375 |

One endpoint query includes a complete fixed finite SRS search, not one SRS
candidate. `endpoint_checks.json` records per-query IK call counts, candidates,
robot checks, exact numerical outcomes, and runtime. All policies find a valid
endpoint on their first query except some random runs, so no deterministic
first-success gain is claimed. This is a robot-only endpoint retrieval result,
not evidence of task success or dynamic path feasibility.

Both fixed 2cm positive body-x/body-y native paths pass the existing 2mm/2degree
tracking gate, with maximum errors 0.101/0.100mm. Actual `q_out` and continuation
state are in `native_subset.json`. A tiny jerk margin residual of -5.05e-6 is
recorded and was not repaired. These two predetermined paths do not validate
native execution of all benchmark endpoints.

`tests.json` records finite-difference gradient max error 3.41e-8, common-frame
SE(3) score error 4.22e-15, batch/single error 5.33e-15, CPU/GPU error 3.55e-15,
invalid-rotation rejection, and body-relative score sensitivity. Runtime was
6.75s overall, including the 1.66s endpoint diagnostic and both native replays,
on this host. `run_summary.json` contains the exact executing source hashes.

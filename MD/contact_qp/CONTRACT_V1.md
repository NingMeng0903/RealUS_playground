# Contact QP contract v1 — frozen 2026-09-10

Baseline: `95038b5f33ff28ed38179482f626eacbb5822329`. This implementation follows the user's five-stage plan. No hardware motion is authorized by the implementation/test runners. Existing modified worktrees and reference repositories are outside this change.

## Task and evidence

Fixed target compression force 4 N; original ICRA force/torque law parameters are the nominal baseline. Required image windows are LEFT and RIGHT; CENTER is diagnostic. Unrepaired windows remain in the acquisition loss and are reported as gaps while scanning continues. Mechanical admission can stop progress. Image-direction and keep-margin rows are policies, not acoustic derivative certificates. No image Jacobian learner, patient stiffness estimator, lateral search or force-target adaptation.

Current source data: `/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact`, 68 raw complete scans, 12,588 decoded images. Raw/aligned H5 must not be aligned twice. The effective US delay 0.15196365053143765 s is a configured phase offset, not a certified hardware latency. Force remains current for control/energy.

## Frames and five different velocities

All public numerical inputs use SI units. Twists are `[vx,vy,vz,wx,wy,wz]`; wrenches are `[fx,fy,fz,mx,my,mz]`, paired at the same reference point. Baseline commands retain the existing TCP pivot. Contact geometry is specified by `T_tcp_face`, the aperture half-length, and image-x sign. Do not infer these from the collision mesh, CoP disk, phantom coverage width or software hflip.

1. `nominal_twist`: clamped baseline output, including the existing torque law.
2. `qp_twist`: Cartesian outer proposal, never called actual.
3. `sent_twist`: equivalent command reconstructed from FINAL software payloads; keep final arm/rail payloads, timestamps and sequence IDs alongside it.
4. `predicted_twist`: arm command plus current rail execution estimate, with its stated errors/horizon; not full measured velocity.
5. `measured_twist`: full measured arm/rail state derivative/FK, with validity and timestamps.

Controlled compression `force_n` is positive and separate from the signed environment-on-tool physical wrench used for power. No automatic conversion of a control torque into a physical port torque. Geometry and power signs default UNVERIFIED for hardware; explicit synthetic geometry/port bounds are allowed in simulation.

## Core interfaces and ownership

`peirastic/contact_qp/` is hardware/IPC/UI independent. It exposes validated versioned dataclasses for geometry, image observations, QP configuration/input/result and generic `TwistConstraints` (`lower <= A @ V <= upper`, current TCP, base components at the native boundary). Adapters own transport/time conversion. Array inputs must be copied/immutable enough that callers cannot mutate an admitted candidate.

`ContactObservation` carries frame/source identity, effective/received timestamps, `[left,center,right]` quality and validity, window version and registration version. Registration includes crop, hflip, image-x mapping, delay and calibration revision. Version changes invalidate old visual evidence and action-response history. Low quality is not invalid measurement.

Geometry maps each face point's velocity using `n.T @ (v + omega cross r)`. Only aligned geometry simplifies to `vn - x*omega`. Aperture/force-priority rows use the same calibrated normal/rocking representation. Physical variables remain `[vn, omega, alpha]`; slack and epigraph variables are algebraic only.

`T_tcp_face` maps face coordinates INTO TCP coordinates: `p_tcp = R_tcp_face @ p_face + r_tcp_face`. Face +z is the compression normal and face +y is `face_z cross face_x`. Positive omega is right-hand rotation about existing TCP +y, NOT inferred from image brightness. In TCP coordinates `H=[e_z_translation, e_y_rotation, b]`, where `b[2]=b[4]=0` and b contains permitted path PLUS positional feedback for this candidate. Thus `V_tcp=H@y`; both path feedforward and feedback are alpha-scaled. Window image right is `image_x_sign * face_x`. A stiffness center c is a physical FACE-x coordinate in [-a,a], independent of image flip. The robust added-velocity row is computed geometrically at `r_tcp_face + c*R_tcp_face[:,0]`: `g_c = n.T @ [I,-skew(r_c)] @ H[:,:2]`, applied to `[delta_vn,delta_omega]`. Do not substitute raw TCP coefficients into the simplified formula when the face is offset/rotated. Point kinematics stays exact for general R; image-plane invariance only applies when the rocking and image-plane axes align.

## State and publication semantics

Measurement filters/contact observations consume each valid fresh sample once, even if a candidate is rejected. Command-dependent integration has explicit prepare/commit/abort. Unchanged accepted nominal output must reproduce baseline history; changed accepted output uses explicit anti-windup/feedback. Aborted proposals do not advance command integrators. Measured tilt and integrated command tilt remain separate.

Arm and rail publication are NOT a physical atomic transaction. Track each device's publication facts and stop epoch. Partial/uncertain sending cannot roll physical history back. Logical success requires both transports to accept; this still does not prove measured motion.

Reference time is advanced only for a fully committed candidate with a bounded admitted interval (`max_step_s`), never by a later catch-up of rejected/coasted elapsed time. Actual acquisition coverage is reconstructed separately using measured path locations and aligned image observations. A partial/unknown publication freezes reference progress and requires trusted state before recovery. Existing supervisor deadline remains `4*T + 12 s`.

Precisely: `t_ref_next=t_ref+h_accepted*alpha_accepted`, `0<h_accepted<=max_step_s`; alpha=0 advances nothing. Final command/predicted twists must remain compatible with the candidate H subspace within the configured per-axis numerical command tolerance; extract the path coefficient from each FINAL mapping and commit `alpha_accepted=min(alpha_proposed, max(0,alpha_sent), max(0,alpha_predicted))`. Rank-deficient zero path gives zero progress. Excess off-subspace motion, reverse or over-proposed path coefficient beyond tolerance rejects ordinary publication. This is accepted reference progress, not proof of physical coverage. No ACK retry can advance it twice.

## QP and safety precedence

Fixed 4 N nominal target. LEFT/RIGHT signed-margin visual rows remain active after repair; gamma only multiplies deficit requests. CENTER never triggers repair. Alpha objective is normalized so its unconstrained preference lies in [0.25,1], but its HARD range is [0,1]. Stale/invalid image disables repair/history updates and uses maximal acquisition loss.

At both stiffness-center interval endpoints, impose the aperture added-velocity budget and, outside a reliable force-sign band, `sign(F-F0)*(delta_v-c*delta_omega) <= 0`. These constrain ordinary repair only. If they conflict with mechanical admission, discard the visual/relative-baseline attempt and request the existing mechanical stop/fault route; do not relax mechanical limits or silently claim the precision condition still holds.

Native receives generic linear Cartesian inequalities, not image semantics. It must enforce/recheck them for BOTH the commanded and rail-compensated predicted mapping. Recheck again after the last native AND Python payload modification, before the first transport publish. Unknown/expired certificates reject admission. Existing joint/collision/velocity/acceleration/jerk protections and watchdog thresholds are not weakened.

Immediately before EACH device send/commit, recheck candidate expiry and current stop epoch. Stop increments the epoch and fences queued software sends. If a send may already be in flight, record unknown/partial execution and retain bounded liability until trusted replacement/stop evidence; software fencing is not retroactive hardware cancellation. No ordinary recovery while old pending sends can still overtake the stop without a declared exposure bound.

## One energy ledger

One settled certified balance, earmarked outstanding reservations, and a stopping reserve; reservations are not a second energy source. Ordinary admission uses free balance after reservations and stopping reserve. Reserve worst PREFIX consumption over old-command persistence, mixed arm/rail execution and stopping tails, not merely end-of-interval work. Future returned energy is unavailable until conservatively settled.

Reject/abort is NOT a refund. Release an unexposed candidate only with explicit no-send evidence; old exposure remains reserved. Partial/uncertain publication retains liability. Each physical interval is settled once using conservative full-6D port work minus configured dissipation; release matching reservation in the same software ledger operation. Releasing unused reservation is not recharge. Never lower-clip the tank to hide overspend. Missing measurements retain liability; invalid bounds mark certification lost. Without finite verified exposure/stop/error bounds hardware is MONITOR/UNVERIFIED, never certified passive.

## Frozen faults and outcomes

Image unavailable: no visual repair, loss=max, unknown coverage. Visual slack: policy tradeoff, not proof of irreparability. Task constraint infeasible: abandon task attempt; mechanical stop/admission remains authoritative. Mechanical infeasible/expired certificate: reject new candidate and use existing stop/fault handling. Rejected/no-send: no new reference commit, old physical exposure persists. Partial send/unknown execution: record device facts, retain reservations, freeze logical progress, stop. Duplicate/stale ACK: cannot re-commit, re-settle or resurrect an earlier stop epoch. Final outcomes: complete, complete_with_gaps, aborted; coverage must not be inferred from reference endpoint alone.

## Stage gates

0: freeze this contract and acceptance JSON before implementation.
1: image/frame versioning, nominal transaction adapter and >=100,000-tick baseline equivalence; fix proxy slack/full measurement forwarding. Ultra review and relevant regression.
2: QP, finite-area numerical and image closed loops; quantify rocking restriction. Ablations: baseline, image-only, no aperture budget/cost, complete, gamma on/off, scan-speed matched, rocking-speed/budget matched. Ultra review and tests.
3: native generic constraints, final publication recheck, async publication and energy ledger fault injection. No fake rollback/refund/double settlement/progress. Ultra review and tests.
4: freeze all design parameters then independent seeded acceptance. Report failures honestly. Only after behavior evaluation remove obsolete code/config, rerun ALL related regression/build/performance checks, and audit again.

No held-out retuning. If statistical or quality acceptance fails, retain results and mark the feature NOT ACCEPTED/default disabled; do not relax thresholds, silently drop difficult cases, or present offline human correlations as new-controller efficacy.

The acceptance JSON freezes comparators, percentile paired-bootstrap method, uniform scenario weighting, per-scenario force gates, active-scan metric intervals, failed-run treatment, and physical synthetic image-coupling truth. Quality acceptance requires less missing path AND no loss of valid coverage per elapsed time against baseline and matched-motion controls. Timeout/missing/invalid pairs fail acceptance instead of being omitted. Test-only synthetic truth never becomes a claim about the human images.

Each scenario uses an independent deterministic RNG substream keyed by `(scenario_index, seed)`, so within-scenario paired bootstrap is valid; never reuse a common random stream across independently resampled scenario families. `full` has gamma fixed at 1 (consistency disabled), `full_consistency` has action-response gamma enabled; their other parameters are identical.

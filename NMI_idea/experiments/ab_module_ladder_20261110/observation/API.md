# Geometric observation and common A interface

This is an explicitly declared L0 continuous-polyline image-feature model. It is not ultrasound B-mode or evidence of clinical acquisition. The patient is persistent and independent of planned actions. Production `build_open_bake` is not used. Original `probe_plane_hit` is an audit reference; original `filter_obs_for_control` filters visible centroid/depth, with no tangent or out-of-plane channels supplied.

Import `observation` with the parent ladder directory on `sys.path`.

```python
from observation import *
particles = sample_prior(720281, 64)  # legal independent prior; use new approved seed per run
belief = Belief(particles, support='finite')  # exact frozen-catalogue Bayes; no SMC mutation
# Continuous independent-world exploration uses support='continuous' and reports degeneracy.
public, audit = render(private_world, actual_pose, key,
                       frame_id='unique/acquisition/frame', time_s=elapsed_s)
belief.update(public, actual_pose)  # never audit
history.append((public, actual_pose.copy()))
poses = candidate_actions(actual_pose, history, belief.particles, belief.weights)
response = predict_batch(belief.particles, poses)
```

`pose=[body_x_m,body_y_m,yaw_rad]`, surface z=0, Rz(yaw), image plane probe Y=0, u=probe X, depth=-probe Z. Body-to-robot/world transforms are supplied externally. Targets are clipped to x/y ±40mm and yaw ±0.8rad by common candidate generation; renderer evaluates the supplied actual pose and does not silently clip it.

`World` contains private persistent target and decoy polylines and nuisance appearance. For a **finite catalogue mechanism check**, the catalogue itself is the public prior and the hidden index selects one of its members. No selected index may enter the policy. This measures exact finite support inference/planning, not continuous patient generalization. For independent continuous worlds, `sample_worlds` and `sample_prior` must use independent random seeds.

`predict_batch(particles, poses)` returns:

- `means[P,A,2,5]`, ordered `(image_u_m,image_depth_m,visible_length_m,image_axis_rad,identity_score)`; source 0 is target hypothesis, source 1 decoy hypothesis. Model source indices are hypotheses, never actual sensor labels.
- `p_source[P,A,2]`, mutually exclusive observed-source return probabilities; `1-p_source.sum(-1)` is the real missing-image event. Both source supports derive from actual clipped persistent geometry.
- `p_visible[P,A]` and `contact[P,A]`.
- `geometric_length[P,A,2]` and `angle_audit[P,A,2]` are **hypothesized geometry**. They may be used in model-based planning; never substitute actual-world values. `particle_axis_plane_angle` returns these legal-hypothesis angles with NaN for no support.

`render_model(particle,pose)` returns the same fields with the P/A dimensions removed. `sample_model_observation(particle,pose,key,frame_id=...,time_s=...)` samples a legal public observation for a B branch. Branch observations must update against **all** persistent particles with `log_likelihood_batch` or a copied finite belief. Conditional follow-up actions are shared across particles compatible with that branch.

`log_likelihood_batch(public,particles,pose)->[P]` is the full public source-mixture likelihood, including misses and identity/measurement noise. Source probabilities of zero have exact zero likelihood. Identity and length clipping atoms are accounted for. Per-frame noise is independent conditional on persistent world/nuisances. Never count the same physical frame twice. `Belief.copy()` clones public history, weights and RNG state without seeing truth. `support='finite'` preserves catalogue support exactly; default `support='continuous'` resamples at ESS<.55N and performs full-public-history MH. Diagnostic raw ESS, severe degeneracy, max weight, unique geometries and MH acceptance must be reported. A uniform post-resample ESS does not establish posterior coverage. All-incompatible evidence raises explicitly rather than manufacturing a target measurement.

`render_block` produces exactly five new frames at 0,.125,.25,.375,.5s. `public_stop(frames)` checks all five unique frame IDs and actual timestamp offsets, measured connected projected length≥15mm, |u|≤2mm, identity≥.75 and contact. It cannot measure 3D out-of-plane angle. The precise observable proxy is a long connected projected image segment. A 2D image PCA axis in (u,depth) is supplied but is never mislabeled out-of-plane yaw.

`posterior_angle_probability(particles,weights,pose,public=last_frame)` optionally adds an explicitly estimated posterior risk gate, e.g. ≥.9 probability of ≤10° axis-plane angle. Weights must **already include** `last_frame`; this helper only uses its within-particle source association, avoiding duplicate evidence. Keep results of image-only and posterior-risk gates separate. `audit_stop(audits)` independently checks true selected source identity, true connected physical arclength≥15mm, true center≤2mm, true RMS tangent-to-plane angle≤10° and contact for those same five frames. A public stop terminates; audit failure earns no free retry.

The common servo candidates use observed u to produce a continuous centering correction along probe X; ±1.5mm normal and ±.06rad yaw probes; a local quality gradient from excited public pose history; a causal centroid-cloud line estimate only with ≥12mm span, anisotropy≥6 and ≤2.5mm orthogonal RMS; and four continuous posterior-mode template recovery targets. All methods receive these same candidates and reconstruction permissions. Candidate generation and observation have no robot ability or dIRD cost. Root owns actual motion generation/native C2 execution and capability scoring.

`predicted_public_utility` exposes analytic A response (`single_gate`,`block_gate`,`visible`,`per_particle_gate`). It integrates persistent-world five-frame probability as `sum(weights*q_world**5)`, not `(sum(weights*q_world))**5`. `continuous_servo` uses the same exact geometric response and common candidates as a baseline. No learned acoustic/response benefit is claimed.

Protocol was written before cohorts. Test seed 720380 and 256 independent test worlds remain unopened pending root approval. Development checks may use seed 720281. No formal cohort result is included here.

For fast B branching, precompute `prediction=predict_batch(particles,poses)` once. `sample_public_from_prediction(prediction,h,a,key)` and `sample_block_from_prediction(prediction,h,a,key,start_time=...)` sample legal model scenarios without rerendering geometry. `log_likelihood_from_prediction(public,prediction,a)` returns likelihoods against all P particles using the same fixed-action prediction. This avoids P×A repeated Python render loops. For a block, sum its five unique frame log likelihoods before Bayes normalization.

On missing frames, the shared candidate set also includes two yaw=0 crossing surveys at public lane summaries. These are important geometrically: an approximately parallel long-axis slice can miss a registration-shifted curve entirely, while a crossing slice turns lateral displacement into measurable image u. The same measured-u centering then places a legal public reconstructed point under the probe before yaw alignment. These are continuous template-derived poses, not patient truth or a sensor action table.

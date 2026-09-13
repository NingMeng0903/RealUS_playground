# Confidence–CoP experimental fusion, 2026-09-13

The existing `active_probe50_v8r3_tank.yaml` remains the default control.
The two new profiles are `active_probe50_confidence_tank.yaml` (image angular
task, geometric-center translation) and `active_probe50_confidence_cop_tank.yaml`
(same image task, CoP companion translation). They use identical dynamics,
limits, and tank settings. No robot was commanded during implementation.

The confidence mass-center idea is informed by [Welleweerd et al. (2020),
page 3](https://research.utwente.nl/files/247785100/Welleweerd2020automatic.pdf#page=3).
This implementation uses a normalized lateral centroid and the existing angular
M/D dynamics; it does not claim to reproduce that paper's complete control law.

## Mechanisms enabled

| Mechanism | Implementation / boundary |
|---|---|
| Full-column confidence direction | Existing near-depth ROI column confidence mass centroid, pixel-center coordinates normalized to image half-width. Transport policy `confidence_column_centroid_v1`; no L/C/R approximation. |
| Weak-side completion | Both lateral windows at or above 0.8 withdraw the angular target even if their values differ. L/C/R quality remains independent of centroid direction. |
| Angular response | Normalized centroid deadband uses existing 0.03; target scale is existing 0.002 m/s divided by the 0.031 m lateral-window lever. This is an engineering response scale, not an identified image Jacobian or paper gain. |
| Angular dynamics | Exact held-target `I*w_dot + D*w = D*w_target`; original I/D and speed/acceleration/angle limits, plus final publication jerk mechanism. One evidence update per frame; holding a frame holds a target, not a velocity increment. |
| Moment role | Original compensated moment continues to be observed. Its zero-moment angular drive and Coulomb deadband are bypassed in the new mode. Nonzero moment is allowed after image quality is adequate. |
| CoP preference | `cp=-My/Fz` on the registered centered tool-Z face. Positive compressive load above existing contact threshold and cp inside ±25 mm required; invalid cp falls back to zero with a reason. No ck estimate. |
| Allocation priorities | Shared hard mechanics, aperture correction envelope, and command energy; closest feasible progress no greater than quality alpha; closest full image angular target at that progress; closest `vF+cp*w` normal velocity. No CoP/visual objective blend weights. |
| Exact feasible-target fast path | If all three targets jointly satisfy the original rows and strict final-energy admission, all hierarchical objective errors are zero, so the allocator returns that exact optimum without numerical iterations. Otherwise it retains the bounded numerical stages and common deadline. Status remains REPAIR; backend avoidance is logged separately. |
| Component state | Visual dynamics and outer tool-Y acceleration bounds use the same successful final angular vector, re-expressed from base to current tool coordinates. Loading anti-windup removes the full CoP candidate preference, clips to the requested loading interval, and never feeds rail-compensated model Z into loading integration. Candidate/final pairing residuals are both logged. |
| Recovery | Existing `pause_visual`, source 15 ms admission, original 50 ms command lease, successful-interval clock, alpha slew, and genuine stop conditions remain. A stale image withdraws the target and decelerates; it does not restore zero-moment control. |
| Tank | Initial/capacity/reserve 0.100/0.150/0.050 J. Final full TCP command model checked and settled after dual publication. Existing native final power and rocking contracts are reused. |
| Versions / provenance | New controller capability `contact_qp.confidence_cop_components_v1`; preflight checks centroid worker version. Study source hashes include new policy/feature files and native/Python inner source; effective tilt configuration and proposal time recorded. Source hashes do not identify a running native binary by themselves. |

The original 4 N loading loop and measured-force supervisor remain. The hard
aperture envelope can conflict with a large force-loop retraction and an existing
acceleration bound. Such a genuinely infeasible command is not made sendable by
the new fusion. Existing short no-send recovery still requires a valid old lease.

## Energy interpretation

`loading_scan_v1` constructs two **requested task authorizations**, from pure
loading and scan twists before CoP pairing. It sums their positive outward-work
allowances. Angular velocity and companion translation do not enter this source.
The frozen authorization is explicit engineering supply, not recovered physical
energy. In particular it uses requested scan, not final achieved alpha: unused
task allowance can pay other final work; unused supply is discarded, not added
to the tank. Therefore “no visual source term” does not mean every visual action
must deplete the tank. Source availability and proportional source consumption
are logged separately for loading and scanning.

At fixed rotation cp minimizes load-weighted additional normal motion. It does
not ensure constant total force, replace ck, or prove physical passivity.
Filtered CoP and raw budget wrench can differ; the full final wrench–twist
calculation remains authoritative. A mechanical constraint may require a nonzero
closest-feasible rotation even when the image target is zero at fixed progress;
this deviation is logged, not described as a new visual request.

No tissue stiffness, ck, or image Jacobian identification is required. Joint
force-growth bounds, physical interval-work/error bounds, execution tails, and
physical passivity remain outside this implementation.

## Reproducible offline regression

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash analysis_artifacts/confidence_cop_20260913/run_regressions.sh
CONTACT_QP_CONFIG="$PWD/peirastic/config/contact_qp/active_probe50_confidence_cop_tank.yaml" \
  bash scripts/run_icra_tank.sh validate
```

The dataset replay script/report in this directory covers all 49 saved scans.
It is open-loop evaluation using recorded force/path/image inputs; it cannot
predict a new image after a different action or replace a same-path acquisition
comparison. Original final model measurements must not be relabeled as final
execution of a counterfactual command. See its coverage and restart labels.

## Complete acquisition commands

Restart the controller and confidence worker to load the new code. Run each
long-running process in its own terminal. Preserve the saved crop/registration.

Controller terminal:

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh controller
```

Ultrasound terminal (if not already running with the saved registered crop):

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh ultrasound
```

Confidence terminal:

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
CONTACT_QP_CONFIG="$PWD/peirastic/config/contact_qp/active_probe50_confidence_cop_tank.yaml" \
  bash scripts/run_icra_tank.sh confidence
```

Teaching gamepad terminal:

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh gamepad
```

Recording terminal (includes confidence preflight, 4 N ICRA profile, 5 mm/s, raw):

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
CONTACT_QP_CONFIG="$PWD/peirastic/config/contact_qp/active_probe50_confidence_cop_tank.yaml" \
  bash scripts/run_icra_tank.sh record
```

Equivalent existing recording entry:

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record \
  --force-profile icra \
  --speed-m-s 0.005 \
  --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_confidence_cop_tank.yaml
```

To compare image-angular without CoP, select `active_probe50_confidence_tank.yaml`
for confidence and recording. For the existing control, select
`active_probe50_v8r3_tank.yaml`. New mode startup must display
`angular_task=confidence_centroid`, `state=separate_loading_visual_scan`,
`supply=loading_scan_v1`, and the expected `cop_pairing=True/False`.

Keep the same teaching endpoints, scan speed, image settings and path when
comparing. Inspect side-edge image quality at matching measured TCP positions,
plus force, actual path speed, final angular changes and pairing deviations.
Include pei return paths, 003 reversal, jiaqi load, and already-good 007 paths.
Only a successful acquisition comparison supports promoting the experimental
profile to default; request completion alone is not evidence of edge repair.

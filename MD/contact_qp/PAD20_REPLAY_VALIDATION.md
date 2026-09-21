# Handheld 20-region replay and inspection

The handheld profile is `peirastic/config/contact_qp/pad_scan.yaml`. Its default
confidence stream uses 20 regions, a 145-column processing image and endpoint
`tcp://127.0.0.1:17362`. The recorded-scan profile and endpoint remain separate.

Run the independent confidence worker before enabling handheld contact:

```sh
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_outer.sh pad-validate
bash scripts/run_icra_outer.sh pad-confidence
# In another terminal:
bash scripts/run_icra_outer.sh pad-check
```

Restart the controller after updating the code. The ultrasound source must be
running for the new confidence worker to receive images. The original 10-region
scan worker cannot supply this handheld profile; an incompatible stream pauses
translation and disables visual repair rather than falling back to its raw Q.

The existing gamepad launch command remains unchanged:

```sh
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh gamepad --hybrid-law ultrapoc --analysis
```

A custom profile
must be passed consistently with `--pad-contact-config PATH` to the gamepad and
pad-confidence/check/validate commands. Validation and replay perform no device
or robot command I/O; `pad-check` only reads the feature stream.

## Reproduce the offline evidence report

Use an environment with numpy, scipy, cv2, h5py, PyYAML and matplotlib. On the
recording host, `/media/camp/EXT_DRIVE/envs/genesis/bin/python` has these packages.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/pad20_mpl
export PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src"
/media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.pad_contact_replay \
  apps/logs/pad_hybrid/20260921_181337 \
  apps/logs/pad_hybrid/20260921_181359 \
  apps/logs/pad_hybrid/20260921_182016 \
  apps/logs/pad_hybrid/20260921_182309 \
  --output /tmp/pad20_replay_verified --stride 1
```

Each `raw.h5` is opened read-only. The tool decodes its original JPEG and original
capture metadata, then runs `FeatureExtractor` at the configured width. The
10-region comparison aggregates the same original confidence map independently;
it never interpolates historical 10-element Q arrays. Labelled examples also run
a separate complete 10-region extraction for a limited runtime/payload comparison.
The raw 145-column near-field echo evidence is identical across the two layouts.
Finer region boundaries alone cannot improve the spatial resolution of that echo
detector; their expected benefit is finer confidence telemetry and conflict checks.

Output includes:

- `provenance.json`: resolved config, config/code hashes, source paths, original
  H5 sizes and control-log hashes. Each replay row includes the original JPEG hash.
- Per-session `frames.jsonl`: capture/effective/simulated control times, raw and
  predicted 10/20 Q25, 145-column echo and dark mask, sequential evidence requests,
  historical correlation IDs, publication status, runtime and payload measurements.
- `frame_XXXXXX.png`: original-image overlay with regions numbered 1 through 20,
  exact region boundaries, near-depth boundaries, raw/predicted Q25, and dark mask.
- `timeline.png`: raw Q25 and dark mask on effective image time; predicted Q25,
  recorded force, nominal/visual/allocated commands and alpha on control time.
- `cases.json`: qualitative image-review annotations with both region layouts'
  measurements. These annotations were chosen by viewing original frames, not by
  thresholding Q, and they are not physical contact ground truth.
- Per-session and combined `metrics.json`: measured CPU latency and packet sizes,
  30/20 Hz processing deadline exceed counts, synthetic/estimated image age, and a
  contiguous temporal split manifest covering every original frame.

`--stride 15` is a quicker visual inspection mode. It cannot validate consecutive
frame confirmation, pause duration or recovery duration. Full `--stride 1` feeds
every original frame sequentially through the pure evidence object. The tool uses
an in-memory receiver; it does not instantiate the mechanical controller.

## Time alignment and interpretation

The capture timestamp is `capture_monotonic_ns`; effective image time subtracts
the configured calibrated image delay exactly once. The original recorder receive
timestamp is retained separately. The historical worker's receive and publication
times are unavailable. Replay control/receive time is explicitly synthetic:
capture plus `--receive-delay-s` (default zero). CPU timing is measured locally;
the reported age after processing adds that offline cost and is an estimate, not
a measurement of the live publisher/network/control pipeline.

Historical logs often have null `frame_seq`. The report therefore calls their
relationship to the replayed images an **approximate delayed time correlation**.
It keeps both nearest control-time and nearest effective-image-time row IDs and
signed time offsets. Neither is proof that the historical controller consumed a
particular image. Historical `omega_cmd` is displayed as requested/allocated; it
is never inferred to be a published command. Publication remains unknown unless
an explicit published field is present.

The calibration/validation manifest assigns the first 60% and remaining 40% of
each recording to disjoint contiguous intervals. This is a retrospective split:
the listed examples had already been reviewed. No fitting occurs in the replay,
and the report does not claim an independent held-out detection score. Temporal
state carries through each session; there is no reset at the split boundary.

## Reviewed examples

| Recording | Original H5 frame row | Visual annotation |
| --- | ---: | --- |
| 181359 | 4757 | Residual left-edge dark wedge |
| 182016, human scan | 218 | Deep-only shadow; near field remains visible |
| 182016, human scan | 844 | Large right-side near-field gap |
| 182016, human scan | 998 | Broad near-field darkness |
| 182016, human scan | 1307 | Near field visible despite deeper shadow |
| 182309 | 2834 | Left-edge near-field wedge |
| 182309 | 3416 | Central dark column; direction ambiguous |

Numbers here are H5 row indices, not camera frame sequence IDs. The overlays show
both. These seven cases support a bounded inspection of localization and obvious
failure modes; the full recording is not densely annotated. A successful offline
replay does not demonstrate closed-loop repair, real-time scheduling, command
publication, stable physical contact or safety of a future robot trial.

## Recorded findings, 2026-09-21

The final stride-1 report is in `/tmp/pad20_replay_verified`. In the seven reviewed
examples, 20 regions did not independently fix detection: classifications match
the independently aggregated 10-region comparison. Two concrete limitations of
Q25 alone are visible in the original images and extracted measurements:

- The residual left wedges (181359 row 4757 and 182309 row 2834) have minimum
  20-region Q25 approximately .9885 and .9832, despite visible gaps. Their raw
  near-field masks identify 16 and 23 dark columns respectively.
- The deep-only shadow (human row 218) has all 20 Q25 values below .8 and minimum
  approximately .4785, while the near-field mask has zero dark columns and the
  image-only evidence classification is healthy.

The central ambiguous example (182309 row 3416) similarly has high minimum Q25
(.9849); the 29-column near-field mask produces an ambiguous classification and
does not select a direction. Human row 844 produces a unilateral mask of 73
columns on the right. Human row 998 is conservatively classified as bilateral
missing echo, **not** the exact visually assigned global-dark category; it does
not justify directional repair. Human row 1307 produces a healthy empty mask.

The direction confirmation and force/contact gates can still suppress a request
when the raw image diagnostic sees a gap. `raw_classification` and
`evidence_reason` are therefore separate fields. These results support retaining
full-width near-field evidence and inspecting finer confidence bins; they do not
establish improved detection accuracy from changing 10 regions to 20.

The report preserves the exact loaded configuration and evidence source.
`post_run_config_comparison.json` records subsequent mechanical-only configuration
changes (the normal allocation reserve and the naming of the 5 ms hold interval);
the feature, KF and evidence settings are identical. The seven reviewed overlays
were subsequently rerendered with the correct echo depth band, .04–.22, rather
than the confidence band, 0–.22. `overlay_render_provenance.json` records this
display-only correction; original decision and timing provenance is retained.

Across all four sessions, the run decoded and processed **11,218 original
frames**. Combined feature extraction plus evidence/KF cost had median 28.51 ms,
p95 30.04 ms and maximum 152.53 ms. 52 frames exceeded a 30 Hz period and four
exceeded a 20 Hz period. The maximum encoded payload was 6,511 bytes, below the
8,192-byte receiver envelope. The seven paired example extractions had median
10/20-region feature times 27.73/27.97 ms and median payloads 5,780/6,309 bytes.
These are host measurements with single-thread BLAS/OpenMP during concurrent
offline work, not a live scheduling guarantee. Estimated post-processing image
age had median 180.47 ms and maximum 304.49 ms, excluding live transport and
scheduling; the slowest frame can exceed the configured .3-second freshness gate.

The first temporal run exposed a recovery deadlock: baseline learning needed one
second of stable force before healthy raw images could release the pause. That
run allowed a positive alpha target for only two frames. Recovery now requires
.25 seconds of fresh healthy raw evidence independently of baseline readiness;
learning still requires one stable, healthy second and freezes during force
instability or missing echo. Confirmed recovery also resets the repair timer, so
a later defect gets its own .5-second repair interval.

The corrected replay allows a positive alpha target on **412 of 11,218 frames**:
241 healthy-progress and 171 limited-repair frames. Alpha remains zero on 10,806
frames. There are 2,839 force-unstable results (986 in the human recording), 1,604
recovery-confirming results, and many persistent or ambiguous defects. These are
responses to the old controller's recorded motions, not the motions the new
controller would produce. Replay uses recorded contact and force; it does not
reproduce live COP/angle/slack eligibility or physical repair. It therefore still
does not establish usability for continuous scanning or reduced bouncing.

There are three unlabelled 10/20 classification disagreements over the full
corpus and none among the seven reviewed cases; without labels for those frames,
this is not an accuracy gain. The complete counts are in `review_summary.json`.

## Mechanical implementation and verification

The handheld lifecycle prepares the existing 4 N / zero-My nominal controller,
allocates COP/visual/progress requests together, and commits only after successful
device sends. Nominal mechanical state and accepted final-command history remain
separate. Failure or partial send aborts the candidate and enters the stop chain.
Measured relative pose supplies both the mechanical and QP angle guards.

The outer normal target reserves .1 mm/s for IK realization error (14.9 mm/s
allocation cap); final normal speed remains limited to 15 mm/s and final My
rotation to .28 rad/s. Two constant physical speed rows enter the existing IK
interface, so the solver can satisfy these limits while coordinating joints.
They remain active across the existing scan smoothing tiers. No narrow dispatch
window, rail-thread wait, or full outer-QP constraint reinjection was introduced.
Final admission uses separate numerical tolerances of 1e-6 m/s and 1e-5 rad/s;
the guarded nominal rotation allows at most 1e-4 rad/s task residual, or a tighter
configured value. Custom QP residual settings cannot enlarge that allowance.

Filtered contact force controls the 4–4.5 N visual withdrawal. The fresh maximum
of raw and filtered force independently triggers the 6 N supervisory stop;
this is a measured trigger, not a guarantee that physical force never overshoots.
Preferred 2 rad/s² / base-jerk smoothing retains the scan fallback tiers when
mechanical feasibility conflicts. Final speed limits remain active at every tier.

**257 automated tests passed** across the targeted evidence/feature/KF/entry/
replay, lifecycle/visual/mechanical/handoff suites and recorded-pose IK tests.
The eight recorded-pose cases use both Python and native IK, two recorded startup
poses, seek and contact-repair requests, direction changes and cycle jitter. They
run offline with simulated sends; no hardware was commanded. The native worker
requires local IPC outside the restricted sandbox. Logging equivalence, stale or
incompatible frames, repeated frames, force withdrawal, COP failure, custom guard
tolerances and failed/partial sends are covered by the targeted suites.

An additional integrated offline check used real strict-20 image evidence, guide
and lifecycle allocation/commit with 20 Hz delayed healthy frames. Translation
resumed after raw recovery confirmation and ramped to the current 20 mm/s stick
request while baseline learning remained incomplete. This checks the former
deadlock through the outer path, without claiming physical closed-loop behavior.

Dense interval annotation and prospective held-out detector calibration are not
complete. The required mechanical/current/revised closed-loop comparison on
phantom, then human, remains to be performed; software tests and historical replay
cannot substitute for those measurements.

## Follow-up: immediate exit at the mechanical rotation guard

A subsequent live attempt reported `pad_final_review_rejected:final_mechanical_guard`.
The final review enforced the nominal mechanical rotation interval, but IK had
only received the broader physical angular-speed limit. A soft task residual
could therefore pass IK and then fail publication. The existing angular command
row now also intersects `[min(0, nominal My), max(0, nominal My)]` whenever the
mechanical candidate is guarded. This preserves permissible deceleration or
inward motion and remains active when scan smoothing falls back. Guard release
restores the ordinary physical angular bound. The final review and its numerical
tolerance are retained.

The gamepad now reports unexpected hybrid exits even without analysis recording,
including the observed status and message. If idle telemetry has already replaced
the fault text, it explicitly reports that the original reason is unavailable.
An installation acknowledgment that already contains an error or stop no longer
prints success merely because its installation sequence matches.

Read-only inspection of the failing run also found the legacy worker on port
17361 and no publisher on the new handheld port 17362. This is a separate issue:
missing images pause visual repair and translation; they do not explain the
mechanical-guard rejection. Use `pad-confidence` as shown above, rather than
passing the old `pad_contact.yaml` to the shared scan confidence launcher. Restart
the controller and gamepad to load these code changes.

Follow-up verification: 38 lifecycle tests, 10 recorded-pose Python/native IK
tests, and 81 gamepad failure/API tests passed (129 in this targeted run). The
new structural negative control applies a conflicting soft angular objective at
a saved pose: the former broad bounds produce `final_mechanical_guard`, while
the corrected row permits publication with rocking tier 4. The user's latest
pose was not captured; this is an offline structural reproduction, with simulated
sends, not a new physical closed-loop validation.

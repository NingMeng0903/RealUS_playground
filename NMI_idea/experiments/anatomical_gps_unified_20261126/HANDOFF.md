# Anatomical GPS Experiment Handoff & Gate Log

**Last updated**: 2026-11-27  
**Owner (gatekeeping & conclusions)**: GPT-6.2 Astra Max from Step 3 recovery onward. Previous entries retain their original Neon and Sol reviewer signatures. Sol/Luna implement and run experiments; Sol Max reviews independently.
**Experiment Root**: `NMI_idea/experiments/anatomical_gps_unified_20261126`  
**Python Environment**: `/media/camp/EXT_DRIVE/envs/genesis/bin/python` (GPU: RTX 4080 Laptop, Torch 2.12.0+cu126)

---

## Gate Status Overview

| Gate | Milestone | Status | Reviewer Audit | Raw Evidence Artifact | Decision Notes |
|---|---|---|---|---|---|
| **Gate 1** | Step 1 — Theory Foundations & Numerical Witnesses | **PASS** | Baseline verified | `theory/verify_theory_math.py` (6/6 passed) | Kirchhoff trace=0, Gauge err=0, banded GP err=1.332e-15, IPE attenuation, conformal bound=0.1997, V8.3/V8.4 err=5.551e-16. |
| **Gate 2** | Step 2 — `beta=0` Alignment, Material Chart & Dual Material GP (3-Axis Audit) | **PASS** | Round 3 PASS (`correct`, 0.99) | `results/step2_training_support.json` (`sha256: 1c7c4da0...`) | 26/26 pytest, 509/512 (99.4%) valid across 8 profiles at 25.6–53.5 mm median shifts; `2.12e-6 mm` slab continuity; 0 unknown/holdout leaks. |
| **Gate 3** | Step 3 — Topology-aware dynamic reader | **FIX_AND_REVERIFY — empirical FAIL** | Sol Max: Step 3R confirmation ACCEPT; Astra Max: retain v6, no gate release | `results/step3_blind_v6/`, `results/step3_blind_audit/station_identifiability.json`, `results/step3r_dev/rank_confirm.json` | All three original folds failed. Train-only proposal re-ranking did not improve whole-tape outcomes. No Blender evaluation. |
| **Gate 4** | Step 4 — Closed-loop scanner, fair baselines & asymmetric RL | **BLOCKED by Gate 3** | Not started | No Gate 4 result | Do not start RL, MPC comparisons or Blender evaluation on these checkpoints. |
| **Gate 5** | Step 5 — T-RO evidence synthesis | **BLOCKED by Gate 4** | Not started | No Gate 5 result | `TRO_EVIDENCE_SUMMARY.md` is a future deliverable, not evidence of a completed milestone. |
| **Gate 6** | Step 6 — IJRR embodied sufficiency experiment | **BLOCKED by Gates 4 and 5** | Not started | No Gate 6 result | No four-cell experiment or irreversible-cut sweep is authorized. |

---

## Evidence-Gap Tracker (Honest Reporting — Never Fabricate)

| Gap ID | Description | Affected Splits / Claims | Mitigation / Reporting Rule |
|---|---|---|---|
| `G1_NO_CLINICAL_STATION_LABELS` | `clinical_station_labels_available = false`. No clinician-annotated NYSORA clinical window labels exist in training or Blender holdout. | Step 2, Step 3, Step 4 station recognition | Evaluate topological role co-presence + verified junction-proximity windows (`SFJ`, `CFA_bif`, `CFV_bif`, popliteal band) and report explicitly as non-clinical topological surrogate. |
| `G2_CT_FOV_PROXIMAL_ONLY` | Real labelled CT (`medseg_right`) covers only 90 mm proximal groin/thigh (`y ∈ [-560, -480] mm`, 17 aligned 5-mm rows); 0 rows of `S5_POPLITEAL`. | `FAMILY_HELD_OUT_ATLASES`, CT coverage claims | Restrict CT-only cross-family station evaluation to proximal S1–S4; attribute S5 popliteal coverage transparently to anatomical atlases. |
| `G3_BODYPARTS3D_NO_DFA` | `bodyparts3d_right` and `bodyparts3d_left_mirrored` have 0 DFA mesh rows (`S3_ART_BIF` and `S4_VEN_BIF` = 0). | `LOAO_BODYPARTS` | Evaluate `LOAO_BODYPARTS` on S1/S2/S5 only; run full S1–S5 cross-atlas evaluation on `LOAO_OPEN3D`. |
| `G4_BLENDER_SINGLE_HOLDOUT_INFERRED_DFA` | Single independent holdout anatomy (`blender_holdout`); `R_DFA` has `support_source=2` (inferred arterial path) and 6 proximal arterial gaps (`[-545,-520]`), plus 10/147 bone-occluded legacy stations. | Zero-shot Blender holdout | Strictly isolated via `AGPS_EVAL_HOLDOUT_ALLOWED=1`; zero iteration/tuning on Blender; report direct roles (S1/S2/S5) separately from inferred `R_DFA` (S3/S4) and physically reachable `137/147` legacy stations. |
| `G5_PHYSICS_ECHO_SIMULATOR` | Evaluation uses label-blind physics echo kernel + SMPL-X kinematics, not raw B-mode images or closed-loop RM75 physical phantom trials. | T-RO system claims | State explicitly in `TRO_EVIDENCE_SUMMARY.md` limitations section. |

---

## Chronological Gate & Verification Log

### 2026-11-26 — Step 0: Plan Persisted & Baseline Recorded
- Persisted approved continuation plan to `PLAN.md` and initialized `HANDOFF.md`.
- Verified baseline status:
  - Gate 1 (`theory/verify_theory_math.py`): **PASS (6/6)**.
  - Baseline `tests/test_sim_and_firewall.py`: **12/12 passed** (requires re-verification after Step 2 `beta=0` & material-chart interface upgrades).
  - Baseline `sim/audit_training_support.py`: **FAIL (exit 2, 13 blockers)** — confirmed as First Blocker to fix in Step 2.
- Next action: Delegate Step 2 implementation (`sim/material_chart.py`, `sim/anatomy_assets.py`, `sim/smplx_dynamic_atlas.py`, `sim/metric_graph_gp.py`, `sim/audit_training_support.py`, theory remark, and tests) to Sol (`openai-codex/gpt-6.2-sol:xhigh`).
### 2026-11-26 — Gate 2 Round 1: FIX_AND_REVERIFY
- Sol (`step2_material_chart_and_dual_gp`) completed initial Step 2 implementation: 23/23 pytest passed, `audit_training_support --proposals 64 --seed 58 --require-reader-inputs` exited `0` (`58–64/64` valid across all 8 profiles, 0 `skin_unknown`/`bone_unknown`, `round-trip=2.33e-15`, `dworld_dtilde_xi=5.90e-7`, `beta=0 J_q=1.55e-8 mm/rad`).
- Reviewer (`gate2_reviewer_audit`, GPT-6.2 Sol max) reproduced both commands byte-for-byte (`sha256: 1241b43d...`) and returned **FIX_AND_REVERIFY** with 3 concrete defects:
  1. `[P1]` `sim/material_chart.py:214-222`: thin-corridor clamp backed `r_bone` inward past raw bone boundary at `(-173°, -880 mm, d=0.925)`, decoding into voxel occupancy `0.60 >= 0.55` (and 5 BP3D mirrored stress knots).
  2. `[P2]` `sim/material_chart.py:109-112` & `_TrainingValidity`: standalone geometry loaders reopened neutral skin file without verifying `_NEUTRAL_SMPLX_SURFACE_SHA256`.
  3. `[P2]` `sim/metric_graph_gp.py:1021-1045`: `audit_points()` and independent regression sampled straight knot chords rather than the exported material curve sampled by `audit_centers()`.
- Dispatched all 3 fixes to `step2_material_chart_and_dual_gp`; awaiting re-verification and Gate 2 Round 2 reviewer audit.

### 2026-11-26 — Gate 2 Round 2: FIX_AND_REVERIFY
- Sol (`step2_material_chart_and_dual_gp`) fixed all 3 Round 1 defects (`25/25` pytest passed; `audit_training_support` exited `0`, `61–64/64` valid across all 8 profiles, `0` unknown skin/bone/outside-skin/unsupported).
- Reviewer (`gate2_reviewer_audit`, GPT-6.2 Sol max) confirmed all 3 Round 1 fixes and byte-identical audit reproduction (`sha256: 9ddc4d53...`), and caught 1 new `[P1]` defect introduced by the discrete single-slab bone guard:
  1. `[P1]` `sim/material_chart.py:235-249`: `_bone_rays` evaluated a single discrete height slab, causing a **4.576 mm jump discontinuity** across `h = -856.5 ± 1e-6 mm` at `(θ=-42°, d=0.925)` (reproduced through `decode_knots_to_world` at Open3D knot 286 where `dworld_dx` predicted `1.58e-5 mm`).
- Dispatched continuous conservative vertex-max upper-envelope bone guard fix + slab-crossing regression test to `step2_material_chart_and_dual_gp`.

### 2026-11-26 — Gate 2 Round 3: PASS
- Sol (`step2_material_chart_and_dual_gp`) replaced the single-slab bone guard with a periodic $C^2$ conservative upper envelope consistent across `_boundaries`, `corridor_supported()`, and analytic Jacobians.
- Independent verification (Neon live run + Reviewer `gate2_reviewer_audit`, GPT-6.2 Sol max, `overall_correctness: correct`, `confidence: 0.99`, zero blocking findings):
  - **Tests**: `26/26` pytest passed (`theory/verify_theory_math.py`, `tests/test_sim_and_firewall.py`, `tests/test_material_chart.py`).
  - **3-Axis Audit**: `python -m sim.audit_training_support --out results/step2_training_support.json --proposals 64 --seed 58 --require-reader-inputs` exited `0`, byte-identical (`sha256: 1c7c4da0078256fc7f6649bcfbb0bfe0f00bf48f2fb42198e7e9c2ec0e4ade29`), `engineering_preflight.passed = true`, `zero_holdout_leakage = true`.
  - **Yield & Shifts**: `509/512` (`99.4%`) valid across all 8 `(source × difficulty)` profiles (`62–64/64` per profile), `0` `skin_unknown`/`bone_unknown`/`outside_skin`/`unsupported_material_points`, median 3D shifts `25.62–33.70 mm` (ordinary) and `42.70–53.46 mm` (stress); Cartesian baseline `0/64` on atlases.
  - **Continuity & Accuracy**: chart round-trip `2.331e-15`, chart Jacobian `5.899e-7`, `beta=0` SMPL-X pose Jacobian `1.545e-8 mm/rad` (`[-880, -475] mm`), slab crossing displacement `2.120e-6 mm` (down from `4.576 mm`), Open3D knot 286 latent linearization residual `3.06e-14 mm` (reviewer) / `6.758e-12 mm` (test sweep), 20,000/20,000 raw bone voxel exits bounded, both standalone loaders verify neutral-skin SHA-256.
- **Decision**: **Gate 2 PASS**. Advancing to Step 3 (`reader/`).

### 2026-11-27 — Step 3 recovery: historical reports are not evaluations of the current checkpoints

GPT-6.2 Astra Max inspected the three stored JSONs, the current reader code and the prior `step3_fix_and_verify_reader` delivery. Gate 2's historical PASS stands. The previous Sol delivery reported 39 passing tests; that is not a test count for the recovered worktree.

| Archived Full reader (seed 42) | F1 at 40 mm | Dynamic role F1 | Observed / all-slice RMSE (mm) | Prefix wrong→right / right→wrong | Station macro-F1 | Certified (matched / held) |
|---|---:|---:|---:|---|---:|---|
| `LOAO_OPEN3D` (S1–S5) | .7384 | .4818 | 22.93 / 33.09 | 23/115 · 17/67 | .5143 | 0/80 · 0/80 |
| `LOAO_BODYPARTS` (S1/S2/S5) | .4563 | .5073 | 20.11 / 32.44 | 57/271 · 26/108 | .7249 | 0/80 · 0/80 |
| `FAMILY_HELD_OUT_ATLASES` (proximal S1–S4) | .3292 | .3654 | 19.28 / 22.08 | 100/497 · 34/157 | .3773 | 0/80 · 0/80 |

Open3D's paired dynamic comparison against B2 passed (`p=.000213623`), but the absolute RMSE and identity thresholds failed. With no Full certificates the conditional violation rate is undefined, not zero. The reported 2.25 mm on correctly associated training singleton echoes is a diagnostic, not the held-atlas tracking result. The observed station labels are non-clinical topological surrogates.

| Checkpoint | SHA-256 in old report (prefix) | SHA-256 on recovery (prefix) |
|---|---|---|
| `reader_loao_open3d.pt` | `41ad65a4c59a` | `e8b1449d21f8` |
| `reader_loao_bodyparts.pt` | `52aadd95581c` | `49c79c8075c9` |
| `reader_family_ct.pt` | `d20d9e8b9b22` | `d7f021334e3f` |
| `reader_frozen.pt` | `65b17622cdb7` (prior agent's train-only delivery) | `f615aec87c0a` |

`reader/joint_association.py::joint_beam` is already imported by the factor graph, and `reader/split_prior.py::station_windows` already calls `_disjoint_windows`. Their existence is verified in code; improved F1, recovery of identity, and non-vacuous certification are **not** established by the archived JSONs. Do not repeat these as proposed fixes or silently attach the old numbers to the new bytes.

Recovery inspection (repository root; no Blender access):

```bash
E=NMI_idea/experiments/anatomical_gps_unified_20261126
jq '{protocol,checkpoint,gate}' "$E"/results/step3_reader_{loao,bodyparts,family_ct}.json
sha256sum "$E"/results/step3_reader_{loao,bodyparts,family_ct}.json "$E"/checkpoints/reader_{loao_open3d,loao_bodyparts,family_ct,frozen}.pt
```

Historical JSON SHA-256 (Open3D / BodyParts / CT): `c0a90ccf00e429df020116159b4dd1443be136d895299a77b4cd2b6ff168a5a2` / `7ae47a2f647b1761521e7284861bc5998a326fc2d60e8e3614d4f089fcc491fd` / `149c3ef4b1abf086c0f25ecdd7832780f21bc525e6fc27c3b8e6b4278d5e7639`. Preserve these reports. Use fresh output paths with `--checkpoint` (no refit) for the current worktree, record source and checkpoint digests, and read `gate.status` rather than interpreting process exit 0 as PASS.

### 2026-11-27 — Gate 3 recovery milestone: FIX_AND_REVERIFY

**Independent review first:** `ReaderRecoveryReview` (GPT-6.2 Sol Max) returned `FIX_AND_REVERIFY`; the experiment was not in Git HEAD, so there was no historical Git diff to reconstruct. Its audit used the actual implementation, tests, prior delivery and frozen reports. **Foreground:** GPT-6.2 Astra Max accepts that disposition after checking the report checkpoint digests, raw gate fields, contributor counts, station confusion and segment denominators. `ReaderRecovery` (Sol) ran each of the three non-Blender evaluations once using the current schema-5 training metadata, without refitting. The archived regression for this generation was **40/40** including **14 reader tests** (`results/step3_recovery/test_reader_and_tracking.log`); it was not rerun for recovery.

| Frozen replay | F1 at 40 mm | Dynamic ID F1 | Observed center RMSE (mm), measured/opportunities | All-slice RMSE (mm) | Prefix wrong→right / right→wrong | Station macro-F1 | Full certifications (matched; held) |
|---|---:|---:|---|---:|---|---:|---|
| Open3D | .6553 | .4537 | 21.13, 322/347 | 32.98 | 24/117; 7/65 | .4743 | 0/80; 0/80 |
| BodyParts | .4660 | .4484 | 19.58, 588/639 | 31.51 | 58/258; 26/121 | .5273 | 0/80; 0/80 |
| CT→atlases | .3337 | .3388 | 18.91, 990/1271 | 21.33 | 104/494; 40/160 | .3773 | 0/80; 0/80 |

Open3D: B0 40 mm F1 `.1105`; paired one-sided dynamic Full vs B2 `p=.001068`, but neither the absolute tracking nor ID threshold passes. Full p50/p95 `9.04/14.41 ms`. Open3D S2 and S3 had no correct station predictions. No conditional certification rate can be estimated; the one-sided upper bound reported with zero certified trials is 1. Matched (exchangeable) whole-segment miscoverage was `10,15,13 / 80`; held-source (shifted) miscoverage was `26,30,41 / 80` respectively. The two schema-5 `>=40%` acceptance checks are engineering diagnostics additional to the approved conditional-risk threshold, not a conformal theorem or a replacement for it.

**Scientifically material findings.** Same episode IDs/seeds do not imply identical observations: the schema-5 generator changed start heights and force/dwell schedules and used disjoint station anchors. The requested station selected training roles for steering, and the known offset recentered the public probe pose (`reader/split_prior.py::_aim`). These are acquisition side channels, not direct role IDs handed to the reader. Do not attribute the v4→v5 differences to the association beam. In v5, finite-only tracking RMSE could improve by omitting a prediction; replay rescored a frame under different association odds from online; B1 omitted partner uncertainty in 3+-lumen merges; the coarse search was skipped if the first frame was empty. Matched Open3D calibration put all 200 tapes in one stratum with `q=19.2957`; the ID-score ceiling was only `7.433844`, so every ID was admissible there and no segment could certify. The other stratum had no calibration tapes.

Reproduction of the *frozen schema-5 worktree*: all 15 source files in `results/step3_recovery/baseline_source.tar.gz` (`sha256: 56dd1d980dd466bd1b9dc3882c15317768f4094ef02a0eea11ba5ba71583298f`) match `input_sha256.txt`. `baseline_source_replay.txt` gives the isolated setup; `checkpoint_preflight.json` and `comparison.json` identify weights and comparisons. The retained reports are `results/step3_recovery/{loao_open3d,loao_bodyparts,family_held_out_atlases}.json`; never run the historical command against the corrective worktree or overwrite those reports.

```bash
E=NMI_idea/experiments/anatomical_gps_unified_20261126
P=/media/camp/EXT_DRIVE/envs/genesis/bin/python
R=$(mktemp -d)
unset AGPS_EVAL_HOLDOUT_ALLOWED
D=$(mktemp -d)
tar -xzf "$E/results/step3_recovery/baseline_source.tar.gz" -C "$D"
ln -s "$PWD/NMI_idea/experiments/ab_validation_20261110" "$D/NMI_idea/experiments/ab_validation_20261110"
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$D/$E:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$P" -m reader.eval_reader --split LOAO_OPEN3D --checkpoint "$E/checkpoints/reader_loao_open3d.pt" --seed 42 --out "$R/loao_open3d.json"
"$P" -m reader.eval_reader --split LOAO_BODYPARTS --checkpoint "$E/checkpoints/reader_loao_bodyparts.pt" --seed 42 --out "$R/loao_bodyparts.json"
"$P" -m reader.eval_reader --split FAMILY_HELD_OUT_ATLASES --checkpoint "$E/checkpoints/reader_family_ct.pt" --seed 42 --out "$R/family_held_out_atlases.json"
```

Next round: restore the approved (potentially overlapping) Gate-2 co-presence/junction surrogate and a station/offset-blind public scan law; make tracking opportunities fixed and replay use actual retained assignment histories; correct B1 merged observations and calibrate the same full contributor-set posterior used for readout. No selection of gates, hyperparameters or test tapes on held-source results. Prior checkpoints and reports are historical evidence, not compatible fitted models for a changed observation law.

### 2026-11-27 — Gate 3: blind acquisition, fixed tracking denominators and exact contributor sets

**Decision: `FIX_AND_REVERIFY` (empirical FAIL, no release).** Sol Max (`ReaderRecoveryReview`) reviewed the actual schema-5→6 changes against the archived source and closed all four reader/evaluator findings. Its initial code verdict still required one provenance fix; its final narrow verdict is **code ACCEPT, Gate 3 FAIL**. GPT-6.2 Astra Max checked the original three schema-6 reports, checkpoint references, per-tape tracking denominators, certification counts, raw station-window rows and the oracle calculation. Blender data were not evaluated and Step 4 RL was not started.

**What was tested.** Each fold trained on its own authorized sources, then scored all six methods on paired seed-42 tapes. Probe steering uses nominal training trunk geometry; the offset shifts anatomy only. The requested station selects a supported start height, not the probe steering roles; shared heights retain all Gate-2 surrogate labels. No new clinical labels or source coordinates went into inference. Retrospective estimates now marginalize the retained assignment histories (including NULL), not a second association search with different odds. Lost paths remain lost; this is not a continuous backward smoother. Exact-mask probabilities underlie both role readout and whole-segment verification; the segment score is a maximum, not a joint sequence likelihood.

| Schema 6 Full | 40 mm role F1 | Dynamic role F1 | Conditional world RMSE (mm); measured / rendered | Wrong→right; right→wrong | Station macro-F1 | Full certifications: matched; held |
|---|---:|---:|---|---|---:|---|
| `LOAO_OPEN3D` | .4421 | .4602 | 20.66; 322/347 | 8/109; 3/73 | .4100 | 0/80; 0/80 |
| `LOAO_BODYPARTS` | .4977 | .4410 | 19.72; 588/639 | 16/266; 6/113 | .5843 | 0/80; 0/80 |
| `FAMILY_HELD_OUT_ATLASES` | .4483 | .3546 | 17.65; 990/1271 | 26/469; 10/185 | .2411 | 0/80; 0/80 |

Primary tracking RMSE is **undefined on all three folds**: there are missing predictions in the fixed set of rendered contributors. None demonstrates ≤4 mm. On Open3D the Full vs B2 one-sided Wilcoxon `p=.000213623` on common measured centers does not repair this; conditional all-slice RMSE was 33.02 mm. At 40 mm B0 was .2863 but B1 was .6550 versus Full .4421. With the probe no longer recentered by the intervention, 120/527 finite slice centers rendered at 40 mm versus 456/527 at 0 mm. Capture is not the same as identifying the echoes that were captured. Retroactive regression 3/73 (4.1%) meets `<5%`, but correction 8/109 (7.3%) misses `≥60%`. Dynamic step latency p50/p95: 10.86/16.84 ms on Open3D.

Full abstained on every matched and held certification tape: the conditional violation rate and its 95% upper bound are **undefined**; superiority to B3 is not established. Matched marginal set miscoverage was 12/80, 8/80, 13/80 respectively; held-source miscoverage was 38/80, 45/80, 68/80. Only the matched source mixture is the exchangeable population for the rank claim.

**Why further tuning cannot resolve both station gates under this design.** Let `Z=(source, true start height)` and `Y` be the scheduled, single surrogate station label. Given `Z`, the public observation and control distribution does not depend on `Y`. For example, Open3D S2 and S3 share −510, −505, −500 and −495 mm. An oracle knowing `Z` can maximize each class's population F1 by sorting cells by `P(Y=i | Z)` and choosing the best prefix for `2 TP / (P(Y=i) + P(predict i))`. The average of these *independent* classwise optima is an upper bound; their positive sets need not form any single-label classifier.

| Evaluation population, balanced supported source/station pairs | Oracle accuracy | Upper bound on single-label population macro-F1 | Approved station threshold |
|---|---:|---:|---|
| Open3D, S1–S5 | .749020 | .752750 | .85 — above bound |
| BodyParts, S1/S2/S5 | .921569 | .920468 | No separate station threshold |
| CT→atlases, proximal S1–S4 | .784314 | .789430 | .80 — above bound |

`theory/station_identifiability.py` and `results/step3_blind_audit/station_identifiability.json` contain the assumptions, source/tag/height rows and joint masses. Luna (`StationOracle`) independently recomputed the bounds; Sol Max reviewed the derivation. This is **not** an upper bound on a realized six-tapes-per-class/source cohort, a clinical impossibility result, or an explanation for every role/tracking miss. Do not exploit finite-sample variation, disjoint the windows, disclose the requested tag, or tune further on these held reports. A reliably attainable station claim needs a separately approved observable endpoint (for example multi-label co-presence or independently annotated visible landmarks); it is not a retroactive amendment to this gate.

**Reproduction and isolation.** `results/step3_blind_v6/commands.log` records actual commands; `pytest.log`: **33 passed in 7.84 s**. `exits.log`: all three evaluator exits zero; **all three JSON `gate.status` fields are `fail`**. `source_signed_manifest.json` binds historical source/artifact digest values (run-local integrity, not external attestation). The saved JSONs/checkpoints predate a metadata-only fix: their provenance routine hashed the *loader's Python source* `sim/holdout_eval_assets.py`; it did not import the loader or evaluate Blender anatomy. Sol replaced recursive source discovery with an authorized positive list. The post-fix isolation regression passed **7/7 in 1.15 s** (`results/step3_blind_audit/manifest_fix/{commands.txt,test_reader_evaluation.log}`); Sol Max accepted the fix. No re-fit, new held metrics, or rewriting of the pre-fix reports was claimed for this metadata change.

From the repository root, with original Genesis environment and non-Blender assets, use a **fresh** output directory:

```bash
E=NMI_idea/experiments/anatomical_gps_unified_20261126
P=/media/camp/EXT_DRIVE/envs/genesis/bin/python
R=$(mktemp -d)
mkdir -p "$R/checkpoints"
unset AGPS_EVAL_HOLDOUT_ALLOWED
export PYTHONPATH="$E:." PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$P" -m pytest -q -s -p no:cacheprovider "$E"/tests/test_{reader_and_tracking,reader_evaluation,reader_history,reader_protocol,exact_set_calibration}.py
for s in LOAO_OPEN3D LOAO_BODYPARTS FAMILY_HELD_OUT_ATLASES; do
  "$P" -m reader.eval_reader --split "$s" --seed 42 --device cuda --checkpoint-out "$R/checkpoints/$s.pt" --out "$R/$s.json"
done
"$P" -m theory.station_identifiability --include-train --out "$R/station_identifiability.json"
```

### Step 3R milestone A — training-source diagnosis accepted; formal Gate 3 still FAIL

Sol implementation/experiments: `VisibleContinuation`, `TrackingInterventions`. **Independent GPT-6.2 Sol Max: ACCEPT for the diagnosis only. GPT-6.2 Astra Max: verified against the per-tape JSON and authorized one frozen repair comparison**, not an RL run. The historical Neon plan and gate criteria stand.

- Same frozen reader `results/step3_blind_v6/checkpoints/LOAO_OPEN3D.pt` (SHA-256 `88fbd7f90b358788ef8a77bd50a6521dba3697e5635265777e16a0742836dc21`); only BP right, BP mirrored left and MedSeg contributed tapes. The shared bank loader reads registered Open3D assets before the split restriction; no Open3D tape, fit example or prior contribution was used. No Blender evaluation. The new true pose, acquisition height and ghost flags are offline supervision, not reader inputs.
- `results/step3r_dev/visible.json`: 120 fit tapes (731901); 80 distinct dev tapes (731902). For the visible-role combination target, macro F1 **.795 with echoes vs .353 with pose alone**; S4 had only five positives. This does not recover the unidentifiable single scheduled label. Recounting `.dev.tapes[].funnel_joint`: **80 scheduled → 46 requested combinations visible → 19 exact identities → 18 also below 6 mm → 0 valid whole-segment certificates**. Only 19/46 visible tapes met the localization criterion by itself. The first verifier blocker was ambiguous/empty identity factors in 76 tapes, no lumen in 4. Conditional risk among accepted tapes is **undefined**, not zero.
- `results/step3r_dev/tracking.json`: 36 distinct eight-frame dev tapes (931901), **36/36** with matching replay SHA in all four arms; 10 no-lumen tapes retained. For real+tracker, true association+tracker, real+true pose, both-true respectively: whole-tape correct representable masks and all rendered-role errors <6 mm **8, 14, 7, 19 / 36**; finite world errors **520, 518, 516, 512 / 532**; conditional RMSE **9.83, 6.13, 5.94, 2.58 mm**. No complete RMSE. True pose is not B2; forced identities are not classifier achievements. True association given tracker gained six complete tapes but its paired RMSE CI crossed zero; true pose independently helped in-plane tracking, and forced association could worsen elevation with a noisy tracker.
- Repair selection: 193/205 nonempty tracker frames representable; all 193 passed the merge gate, 180 reached local top-k, 161 reached refinement, 154 survived pruning; only 70 retained the correct full prefix. Test **common-parent integrated evidence before proposal truncation**, not a wider merge gate or a lower certificate threshold. Frozen decision, populations and budgets: `PLAN.md` §4, “单项修正冻结”. No formal held evaluation authorized by this milestone.
- Checks: `results/step3r_dev/visible_tests.log` (9 passed), `tracking_tests.log` (21 passed, 1 held-source rendering test deselected); `tracking_run.log` preserves the initial empty-frame dtype failure. `tracking_tests_fix.log` (5 passed) and `tracking_run_fix.log` verify the fix without dropping empty frames. The two new repair test files passed before opening the confirmation cohort (`rank_tests.log`, 5 passed).

Reproduce from the repository root in a **new** output directory (recorded JSONs are exclusive-create):

```bash
E=NMI_idea/experiments/anatomical_gps_unified_20261126
P=/media/camp/EXT_DRIVE/envs/genesis/bin/python
unset AGPS_EVAL_HOLDOUT_ALLOWED
export PYTHONPATH="$E:." PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
D=$(mktemp -d)
"$P" -m pytest -q -p no:cacheprovider "$E/tests/test_visible_continuation.py" "$E/tests/test_tracking_interventions.py"
"$P" -m reader.dev_visible_continuation --cohort dev --out "$D/visible.json"
"$P" -m reader.dev_tracking_interventions --cohort dev --out "$D/tracking.json"
```

### Step 3R milestone B — locked paired repair did not improve continuation prerequisites

**Independent GPT-6.2 Sol Max: ACCEPT `retain_v6_only`. GPT-6.2 Astra Max: concur after recounting paired tapes and checking measured errors and timing.** Accepting the experiment is **not** accepting the candidate. The default reader, formal splits and thresholds are unchanged. No new held evaluation, Blender, MPC benchmark or RL run was started. Actual closed-loop `continuation_completed` remains unmeasured.

Sol implementation/experiments: `ProposalEvidence`, `PairedRepairAudit`. Only the proposal ranking changed (`reader/evidence_ranked_reader.py`); the candidate evaluates competing starts against their common parent prior before truncation. Same local top-8, at most 7 refinement proposals, at most 5 final branches; no source, requested station or oracle inputs. Raw comparison: `results/step3r_dev/rank_confirm.json`. Confirmed both arms used the same tape hashes on **80/80 station and 36/36 dynamic** tapes. `731903` and `931902` are independent seeds on the **same training anatomy sources**, not unseen anatomy.

| Fixed-denominator confirmation outcome | v6 | Evidence-ranked | Paired gains / losses |
|---|---:|---:|---:|
| Requested combination visible | 48/80 | 48/80 | 0 / 0 |
| Visible and all required identities correct | 18/80 | 17/80 | 0 / 1 |
| Also all required world errors <6 mm at first complete frame | 13/80 | 13/80 | 0 / 0 |
| Also entire 4-frame segment accepted without violation by old verifier | 0/80 | 0/80 | 0 / 0 |
| All dynamic masks representable/correct and all rendered centers <6 mm over 8 frames | 3/36 | 3/36 | 0 / 0 |

- Both verifiers' first rejection reasons: **78** ambiguous/empty identity factors, **2** no lumen. This is the same frozen verifier applied to two readers; neither 0/80 result estimates risk among accepted segments. No calibrated coverage is claimed for the candidate.
- Dynamic exact echo decisions increased **426→432/497**, but surviving correct prefixes decreased **125→121/198** (2 gains, 6 losses). Conditional world RMSE decreased **11.48→9.63 mm**, with **494/495** centers measured in both arms; complete RMSE undefined. All 12 no-lumen dynamic tapes stayed in the denominator. The one lost station identity was `step3r-visible-v1:731903:bodyparts3d_left_mirrored:S1_CFV_CFA:0`. Neither partial improvement translated into another whole-tape success.
- Median online `step` at stations: **12.405→14.191 ms** on 320 frames per arm, excluding rendering/reset/I/O; not the formal latency test. No downstream joint count increased and one decreased. Do not promote, recalibrate, or tune this variant on these confirmations.
- Highest-priority **next question, not a result**: can an allowed repeat view, compression or pulse dwell resolve the ambiguous *earlier* echo by the decision deadline? The existing certifier consumes online snapshots (`reader/eval_reader.py`, `_run` and `_risk_benchmark`); retrospective estimates are computed separately in the retroactive task. On new training-only tapes, compare online vs commit-time `relabel_history()` with **no post-commit frames**, unchanged whole-segment identity/geometry requirements and separate independent calibration for that decision rule. Measure corrected and newly wrong identities, <6 mm tracking and actual certificates before spending runs on PPO; keep both confirmation roots closed to tuning.

Verification after the final reader/runner edits: `results/step3r_dev/rank_tests.log` **5 passed**, `rank_confirm.log` records successful JSON write. Reproduce with the environment and new `D` from milestone A, using the archived `results/step3r_dev/visible.json` fit state:

```bash
"$P" -m pytest -q -p no:cacheprovider "$E/tests/test_evidence_ranked_reader.py" "$E/tests/test_dev_rank_comparison.py"
"$P" -m reader.dev_rank_comparison --out "$D/rank_confirm.json"
```

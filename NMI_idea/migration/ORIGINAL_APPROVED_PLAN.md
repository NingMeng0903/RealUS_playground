# Anatomical GPS Continuation Plan (T-RO First, Then IJRR)

## Context

This continuation task resumes `NMI_idea/experiments/anatomical_gps_unified_20261126` from an interrupted subagent run. The objective is to: (1) audit existing files, tests, and `results/step2_training_support.json` to separate verified work, items needing re-verification, and the first blocker; (2) write an updated `PLAN.md` and handoff record inside the experiment directory once approved, then fix and pass **Step 2** (3-axis review: original CT anatomical coverage, source-to-harmonic-atlas registration, and perturbed effective sample yield) without fabricating clinical station labels; (3) complete and gate-verify the **T-RO main line** — first passing the topology-aware Bayesian + Belief-IPE + SMPL-X dynamic tracking reader (**Step 3**), then training and benchmarking asymmetric privileged-critic active RL against same-reader, same-action-permission, same-time-budget strong MPC and fixed-recheck baselines (**Step 4**), reporting target capture (拍到目标), identity recognition (认对身份), and continuation completion (完成续扫) separately, plus an explicit theory-support vs. evidence-gap summary (**Step 5**); and (4) only after T-RO gates pass, execute the **IJRR embodied sufficiency gap extension** (**Step 6**). Foreground **Gemini 4 Neon xhigh** owns theory derivation, experimental design, raw-evidence gatekeeping at every milestone, and final conclusions; **Sol / Luna** subagents (`agent: task`) handle implementation, training, and test scripts; **Sol max reviewer** (`agent: reviewer`, `model` omitted) audits every milestone diff and verification run.

---

## Current Status Audit (Grounded in Files & Live Runs)

Environment verified on workstation: `/media/camp/EXT_DRIVE/envs/genesis/bin/python` (Python 3.10.20, PyTorch 2.12.0+cu126 on NVIDIA GeForce RTX 4080 Laptop GPU, NumPy, SciPy, Numba 0.65.0, trimesh 4.11.5, pytest 9.0.3). Note: system `dash` triggers a broken `pytest11` entry point unless `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is set.

### 1. 已完成 (Completed & Live-Verified)

| Artifact / Module | Verification Evidence | Reproducible Command |
|---|---|---|
| **Step 1 — Theory & Numerical Witnesses**: `theory/THEORY_FOUNDATIONS.md`, `theory/theorems_tro_ijrr.tex`, `theory/verify_theory_math.py` | **6/6 pytest PASSED** (`Kirchhoff trace=0`, `Gauge error=0`, `banded GP cov err=1.332e-15`, `IPE attenuation verified`, `split conformal rank bound=0.1997`, `V8.3/V8.4 max identity err=5.551e-16`). | `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider NMI_idea/experiments/anatomical_gps_unified_20261126/theory/verify_theory_math.py` |
| **Step 2a — Label-Blind Physics Ultrasound Kernel**: `sim/physics_ultrasound_kernel.py` | Unit-tested in `tests/test_sim_and_firewall.py`: byte-identical echoes under role/order permutations, world-anchored specular ghost extinguished under 45° re-window, paired CRN noise, force compression & 0.35 s dwell pulse, bone/fascia shadowing & merge. | Included in `tests/test_sim_and_firewall.py` command below. |
| **Step 2b — Holdout Firewall & Asset Separation**: `sim/anatomy_assets.py`, `sim/holdout_eval_assets.py`, `tests/test_sim_and_firewall.py` | **12/12 pytest PASSED**: holdout denied without `AGPS_EVAL_HOLDOUT_ALLOWED=1` and after revocation; training imports never touch holdout; Open3D DFA extracted (44 rows); Blender holdout preserves 6 proximal arterial gaps (`[-545,-520]`), separates `R_DFA` (`support_source=2`), and isolates 10 bone-occluded stations (`137/147` reachable). | `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_sim_and_firewall.py` |

### 2. 待复验 (Needs Re-verification After Interface Fixes)

| Item | Why Re-verification Is Required | Target Gate |
|---|---|---|
| `sim/smplx_dynamic_atlas.py` (`SMPLXLegKinematics`, `DynamicMotionProfile`) | Passed `2.012e-08 mm/rad` pose-Jacobian test ONLY on `manifest["betas"]` (`beta=[+0.75,+0.05,...]`) inside `shaft_roi_v2` (`y ∈ [-1209.5, -562.1]`). Must be re-verified at `beta=zeros(300)` (true training/holdout frame) and across groin+thigh+popliteal `y ∈ [-880, -475]` via the new material chart. | Gate 2 |
| `sim/metric_graph_gp.py` (`MetricGraphPrior`, `MetricGraphAnatomyGenerator`, `_TrainingValidity`) | `test_geometry_conditioning_and_independent_bone_proxy_check` passed only on `medseg_right` with `require_known=False`, which silently bypassed 199/280 `skin_unknown` points; all 8 `(source × difficulty)` profiles failed `fully_known_and_passing == 0/32` in `results/step2_training_support.json`. | Gate 2 |
| `results/step2_training_support.json` & `sim/audit_training_support.py` | Currently reports `reader_input_preflight.passed = false` (13 blockers). Must be re-run after separating engineering fixes from genuine anatomical coverage limits. | Gate 2 |

### 3. 第一个阻塞点 (First Blocker — 3-Axis Root-Cause Audit of Step 2)

Reproduce current failure state:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m sim.audit_training_support \
  --out /tmp/step2_audit_check.json --proposals 32 --seed 58 --require-reader-inputs
# Exits 2: reader_input_preflight.passed == false (13 blockers)
```

Decomposed into the 3 required Step 2 axes (every figure verified live against raw assets):

1. **Axis 1 — Original CT Anatomical Coverage (Genuine Data Scope Limits — Do NOT Fabricate)**:
   - `medseg_right` (real labelled CT: `MedSeg_Pelvic_Vasculature` / `groin_labels.nii.gz`) spans **only 90 mm** (31 native 3-mm slices; 17 aligned 5-mm stations `y ∈ [-560, -480] mm`, groin/proximal thigh only): supports `S1_CFV_CFA` (17), `S2_SFJ_MICKEY` (9), `S3_ART_BIF` (9), `S4_VEN_BIF` (7), and **0 rows of `S5_POPLITEAL` (`[-875, -825]`)**. `FAMILY_HELD_OUT_ATLASES` (CT-only train → atlases test) can therefore evaluate only proximal S1–S4.
   - `bodyparts3d_right` and `bodyparts3d_left_mirrored` have **zero DFA mesh** in `aligned_vessel_surfaces.npz` (all 24 femoral meshes verified; only `femoral artery`, `lateral circumflex femoral artery`, `deep femoral vein` exist), hence 0 rows for `S3_ART_BIF` and `S4_VEN_BIF`. `LOAO_BODYPARTS` evaluates S1/S2/S5; full S1–S5 cross-atlas evaluation runs on `LOAO_OPEN3D` (MedSeg supplies training DFA → Open3D holds 44 DFA rows) and `FULL_TO_BLENDER` (with Blender `R_DFA` flagged as `support_source=2` inferred arterial path).
   - **Evidence Gap**: `clinical_station_labels_available = false`. No clinician-annotated NYSORA clinical window labels exist in either training or Blender holdout; station evaluation uses topological role co-presence + verified junction-proximity windows (`SFJ`, `CFA_bif`, `CFV_bif`, popliteal band) and named vessel roles on Blender, explicitly reported as a non-clinical topological surrogate.

2. **Axis 2 — Source-to-Harmonic-Atlas & Skin/Bone Registration (3 Concrete Engineering Bugs)**:
   - **Bug 2.1 (48 mm SMPL-X `beta` Frame Mismatch + Open Groin Cut)**: `ct_geometry.npz`, `aligned_vessel_surfaces.npz`, `right_leg_open3dmodel_shared_field.npz`, `bonehub_right_leg_occupancy.npz`, `neutral_smplx_surface_mm.npz` (`sha256: 882b54e9bfefaa6d5329fcb056a81a3750b4b01e4de456f088d39eade337c6ba`), and Blender holdout are ALL registered in **`Male SMPL-X beta=0` neutral T-pose** (`max|v_beta0 - v_ns| = 6.1e-5 mm`; hip/knee/ankle Y = `-472.01 / -851.83 / -1286.74 mm`). In contrast, `_TrainingValidity` used `contact_multichart_right.npz`, built on **`manifest["betas"]`** (`beta=[+0.75,+0.05,...]`, 45.27 mm shorter leg: `-457.58 / -823.82 / -1241.47 mm`, `max|v_ref - v_ns| = 47.94 mm`) AND possessing open diagonal cuts at `y ∈ [-540.42, -462.82]` and `[-1244.01, -1233.94]`. This caused `199/280` `skin_unknown` on `medseg_right`, `323–383` `skin_unknown` on atlases, and `55` false `outside_skin` hits on nominal `open3dmodel_right`. Switching to closed `neutral_smplx_surface_mm.npz` (verified **0 boundary edges anywhere below `y = 0 mm`**) drops `skin_unknown` to **0** on all 4 sources and `outside_skin` to **0** on all 3 atlases (leaving only 2 real nominal MedSeg GSV knots at `y=-555,-560` protruding `2.65–5.67 mm` beyond `beta=0` skin due to the CT participant's wider anterior groin envelope).
   - **Bug 2.2 (Harmonic Chart Domain Gap)**: Sealed `shaft_roi_v2/atlas_right.npz` is in `manifest["betas"]` frame and stops at `y = -562.14 mm`, leaving **0/67** MedSeg CT knots and 0 proximal S1/S2/S3 groin stations inside it.
   - **Bug 2.3 (BodyParts3D Uncoupled Popliteal Bone Overlay)**: `open3dmodel_right` (shared-field registered) has only `1/654` nominal knot (`DFA` at `y=-710`) with `bone_occupancy >= 0.55` and `medseg_right` has `0/67`, whereas skin-only-aligned `bodyparts3d_right` / `bodyparts3d_left_mirrored` sit `10–15 mm` anteriorly inside the femoral condyles at `y ∈ [-875, -740]` (`23` and `29` thigh bone-hit knots; `73` and `78` full-leg).

3. **Axis 3 — Perturbed Effective Sample Yield (`0/32` across all 8 profiles)**:
   - Even against `beta=0` closed skin on `open3dmodel_right` thigh (`0` nominal outside-skin, `1` nominal bone hit), raw isotropic world-XYZ Gaussian shifts (`ORDINARY σ_s = 20 mm`, `STRESS σ_s = 38 mm`) yield **`0/32` (`BOTH_thigh_ok = 0/32`)**, because subcutaneous GSV sits only `1.8–8.4 mm` beneath skin and popliteal vessels sit `5–12 mm` from femoral condyles — isotropic 3D Cartesian noise pushes vessels into air or femur >95% of the time, and heavy rejection destroys Generation–Inference Duality (`KL = 0` breaks).
   - Meanwhile, measured real inter-source variability (`open3d` vs `bp3d_r` vs `medseg`) moves vessels `15–45 mm` (up to `67.2 mm` median for GSV!) **circumferentially/longitudinally inside the bone-to-skin soft-tissue annulus** while keeping depth beneath skin strictly positive (`+2` to `+68 mm`). Moving the Metric-Graph Whittle–Matérn GP prior directly into **unconstrained annular material coordinates $\tilde\xi = (\theta, h, \eta_d)$** (where $d = d_{\min} + (d_{\max}-d_{\min})\sigma(\eta_d) \in (0,1)$) solves Axis 3 cleanly while strengthening Theorem 1 (exact Material Gauge cancellation in $\tilde\xi$ + near-zero rejection $\Rightarrow$ Generation–Inference Duality holds empirically!).

---

## Approach (Ordered Execution Steps & Gatekeeping Protocol)

Every major step ends with a **mandatory two-layer gate**: (1) `agent: reviewer` (Sol max reviewer, omit `model`) inspects exact git diff, unit/integration tests, and reproducible commands; (2) **Foreground Neon xhigh** inspects raw JSON metrics, paired comparisons, ablations, and failure cases, records `PASS / FIX_AND_REVERIFY / INSUFFICIENT_EVIDENCE` in `HANDOFF.md`, and dispatches fixes to Sol before proceeding.

### Step 0 — Persist Approved Plan & Handoff Tracking (Neon xhigh, immediately upon approval)

- Create `NMI_idea/experiments/anatomical_gps_unified_20261126/PLAN.md` mirroring this approved plan (including Completed / Needs Re-verification / First Blocker tables, coordinate-fix rationales & evaluation-impact notes, and reproducible commands).
- Create `NMI_idea/experiments/anatomical_gps_unified_20261126/HANDOFF.md` with structured gate logs (Gate 1 PASSED; Gates 2–6 PENDING), exact command log, and evidence-gap tracker so any fresh agent can inspect state immediately.

### Step 2 (Unblock) — `beta=0` Alignment, Full-Leg Annular Material Chart, Dual Material-Space Metric-Graph GP & 3-Axis Audit

**Delegation**: `agent: task`, `model: openai-codex/gpt-6.2-sol:xhigh` for core `sim/` modules + tests; followed by `agent: reviewer` + Neon xhigh gate inspection.

1. **Canonical `beta=0` Skin & Coordinate Interface (`sim/metric_graph_gp.py`, `sim/smplx_dynamic_atlas.py`, `sim/anatomy_assets.py`)**:
   - In `sim/anatomy_assets.py`: add constants `NEUTRAL_SMPLX_SURFACE_PATH = Path("/media/camp/PEI_T7/SMPLX_Vascular_Probability_Cloud/neutral_smplx_surface_mm.npz")` and `_NEUTRAL_SMPLX_SURFACE_SHA256 = "882b54e9bfefaa6d5329fcb056a81a3750b4b01e4de456f088d39eade337c6ba"`; verify SHA-256 on load and record in `bank.provenance["neutral_smplx_surface_sha256"]`. Add `THIGH_NYSORA_Y_BOUNDS_MM = (-875.0, -480.0)` and `FULL_LEG_Y_BOUNDS_MM = (-1280.0, -480.0)`. Never modify raw `AnatomySource.centers_mm`.
   - In `sim/metric_graph_gp.py`: switch `_TrainingValidity` default `skin_path` from `_CONTACT_ATLAS` to `NEUTRAL_SMPLX_SURFACE_PATH` (handling `vertices` already in mm and `triangles` key alongside legacy `faces`/meters format so legacy `_CONTACT_ATLAS` remains testable for direct before/after comparison). Verify `0` boundary edges below `y = 0 mm` so every lower-body slice (`y ∈ [-1350, 0]`) has `skin_known == True`. Keep `bone_min_fraction = 0.55` on `bonehub_right_leg_occupancy.npz`.
   - In `sim/smplx_dynamic_atlas.py`: add `NEUTRAL_BETA = np.zeros(300, dtype=np.float64)` and `forward_pose_canonical(q_t, points_mm, *, beta=None)` (defaulting `beta` to `NEUTRAL_BETA` when `beta is None`) while preserving existing `forward_pose(beta, q_t, ...)` behavior (`beta=None` uses `manifest["betas"]` for sealed `shaft_roi_v2` compatibility). Add `pose_points_and_material_frame(beta, q_t, chart, xi)` returning `(phi_mm, G_material, J_q_mm_per_rad, J_xi_mm)` where `G_material` is the right-handed orthonormal frame from Gram-Schmidt on $(\partial\Phi/\partial h, \partial\Phi/\partial\theta, \partial\Phi/\partial d)$ and `J_q` is the exact analytic SMPL-X pose Jacobian.

2. **Full Right-Leg Annular Material Chart (`sim/material_chart.py`, new file — no existing equivalent covers `beta=0` groin `[-560, -480]` + thigh `[-875, -560]`)**:
   - Class `RightLegMaterialChart`:
     - Built deterministically from `NEUTRAL_SMPLX_SURFACE_PATH` (`beta=0` skin), `BONE_OCCUPANCY_PATH` (`bone_min_fraction=0.55`), and `SMPLX_MALE(beta=0)` right hip/knee/ankle joints (`[-56.70,-472.01,-3.89]`, `[-107.43,-851.83,-18.34]`, `[-90.27,-1286.74,-58.28]`).
     - Smooth $C^2$ cubic spline centerline $c_{\rm axis}(h) = (c_x(h), h, c_z(h))$ over $h \in [-1265, -475]\text{ mm}$ passing through hip joint, bone shaft centroids, knee joint, tibia centroids, ankle joint.
     - Precomputed table on $h$ (`5.0 mm` grid) × $\theta \in [-\pi, \pi)$ (`128` uniform bins):
       - $r_{\rm skin}(h, \theta)$: exact ray intersection from $c_{\rm axis}(h)$ along $(\cos\theta, 0, \sin\theta)$ to the closed `beta=0` skin horizontal contour, additionally capped at right-hemibody plane $x \le -2.0\text{ mm}$ for $h > -555\text{ mm}$ so medial pelvis rays stay single-valued inside the right groin/thigh.
       - $r_{\rm bone}(h, \theta)$: outer radial distance along the same ray of `bone_occupancy >= 0.55` plus `2.0 mm` safety clearance (minimum core radius `4.0 mm` where ray misses bone), clamped to ensure corridor thickness $r_{\rm skin} - r_{\rm bone} \ge 6.0\text{ mm}$.
       - Periodic bicubic `RectBivariateSpline` interpolation of $r_{\rm skin}(h,\theta)$ and $r_{\rm bone}(h,\theta)$ with conservative inner skin guard (`min(spline, raw_skin - 1.0 mm)`) and outer bone guard (`max(spline, raw_bone + 1.5 mm)` where corridor permits).
     - Exact forward & closed-form inverse APIs:
       - `xi_to_world(xi)` for $\xi = (\theta, h, d)$: $r = (1 - d) r_{\rm skin}(h,\theta) + d r_{\rm bone}(h,\theta)$, $X_0 = (c_x(h) + r\cos\theta, h, c_z(h) + r\sin\theta)$ (`d=0` skin, `d=1` bone, matching `Material_Atlas_Explanation.pdf`).
       - `world_to_xi(xyz_mm)` -> exact closed-form $(\theta, h, d)$ (`< 1e-12` round-trip error inside domain).
       - Unconstrained latent depth bijection `d_min = 0.04, d_max = 0.94`: `d_from_eta(eta) = d_min + (d_max - d_min) * sigmoid(eta)` and `eta_from_d(d)` (clipped to `[d_min + 1e-4, d_max - 1e-4]` with explicit projection residual logging via `encode_with_audit(xyz_mm)`).
       - Analytic Jacobian `dworld_dtilde_xi(tilde_xi)` returning `(..., 3, 3)` verified against central differences `< 1e-4`.
       - `nominal_reference_radius(h)` returning mean soft-tissue vessel radius $R_{\rm ref}(h)$ (for converting mm-equivalent tangential sigmas $\sigma_\theta = \sigma_{\rm arc} / R_{\rm ref}$).

3. **Material-Chart Generation–Inference Dual Metric-Graph GP (`sim/metric_graph_gp.py`)**:
   - Support two explicit parameterization modes on `MetricGraphPrior` and `MetricGraphAnatomyGenerator`:
     - `coordinate_mode="material"` (**default** for T-RO pipeline) and `coordinate_mode="cartesian_xyz"` (retained exclusively for Step 2/3 ablation comparison!).
     - `domain="thigh_nysora"` (`y ∈ [-875, -480]`, default for T-RO NYSORA S1–S5 + dynamic thigh training/eval) and `domain="full_leg"` (`y ∈ [-1280, -480]`, for legacy route audit).
   - In `coordinate_mode="material"`:
     - Encode nominal source knots into $\bar{\tilde\xi} = (\bar\theta, \bar h, \bar\eta_d)$ via `RightLegMaterialChart.encode_with_audit`, saving exact `nominal_projection_audit` (number of knots projected, median/max projection residual in mm — e.g., MedSeg's 2 distal GSV knots `2.65–5.67 mm` and BodyParts3D's popliteal condyle knots — NEVER modifying `AnatomySource` or holdout Blender).
     - Apply the **EXACT same sparse CSC Kirchhoff + Matérn-3/2 precision matrix $\Lambda_0$** (Theorem 1) to latent vector $x = (u_s, \delta_e, \Delta h_v)$ in unconstrained material coordinates $(\Delta s_\theta, \Delta h_{\rm drift}, \Delta\eta_d)$ where $\Delta\theta = \Delta s_\theta / R_{\rm ref}(\bar h)$:
       - `ORDINARY_MATERIAL`: `sigma_shared = (20.0 mm arc, 14.0 mm h, 0.45 eta)`, `sigma_relative = (4.5 mm arc, 4.0 mm h, 0.25 eta)`, `sigma_landmark_mm = 18.0 mm`, optional SMPL-X shape draw $\beta \sim \mathcal N(0, 0.4^2 I_{10})$ (zero for neutral-audit mode, active for dynamic/shape-randomized training).
       - `STRESS_MATERIAL`: `sigma_shared = (36.0 mm arc, 24.0 mm h, 0.70 eta)`, `sigma_relative = (7.0 mm arc, 6.0 mm h, 0.35 eta)`, `sigma_landmark_mm = 30.0 mm`.
       - `shift_sweep_material(offset_mm)`: deterministic circumferential arc shift $\Delta s_\theta = \text{offset\_mm}$ along the compartment (`0, 10, 20, 30, 40 mm` sweep stays 100% anatomically valid inside skin and outside bone!).
     - Decode $\tilde\xi = \bar{\tilde\xi} + L_{\rm knot} x \mapsto X_0(\tilde\xi)$ (and through $\Phi_{\beta, q_t}$ when moving/shaped), and STILL audit every sampled centerline independently against raw `neutral_smplx_surface_mm.npz` + `bonehub_right_leg_occupancy.npz` (`_TrainingValidity`)!
     - Record both `effective_valid_yield` AND `achieved_euclidean_rms_shift_mm` (median & p90 across knots) so we verify we achieved large 15–38+ mm 3D shifts WITHOUT trivial shrinking!
     - Update `theory/THEORY_FOUNDATIONS.md` and `theory/theorems_tro_ijrr.tex` with a concise 1-paragraph remark documenting the unconstrained material logit-depth chart $\tilde\xi = (\theta, h, \eta_d)$ preserving Theorem 1's sparse CSC precision & exact material gauge cancellation while achieving near-zero rejection.

4. **Upgrade 3-Axis Support Audit (`sim/audit_training_support.py`) & Tests (`tests/test_sim_and_firewall.py`, `tests/test_material_chart.py`)**:
   - `audit_training_support.py` outputs `results/step2_training_support.json` with 4 explicit top-level blocks:
     - `axis_1_original_ct_coverage`: per-source Y range, supported role rows, NYSORA role co-presence rows, split-level supported stations (`LOAO_OPEN3D`: S1–S5; `LOAO_BODYPARTS`: S1/S2/S5; `FAMILY_HELD_OUT_ATLASES`: S1–S4 proximal).
     - `axis_2_atlas_registration`: comparison of old `_CONTACT_ATLAS` (`manifest["betas"]`) vs canonical `neutral_smplx_surface_mm.npz` (`beta=0`), chart round-trip error, analytic Jacobian error, and per-source `nominal_projection_audit` (logging MedSeg 2 GSV knots + BodyParts3D popliteal condyle hits verbatim).
     - `axis_3_perturbed_sample_yield`: 64 proposals per `(source × difficulty)` comparing `cartesian_xyz` (old baseline) vs `material` (new dual GP) on `thigh_nysora`, reporting `fully_known_and_passing`, `skin_unknown`, `bone_unknown`, `outside_skin`, `in_excluded_bone`, and `achieved_euclidean_shift_mm_median_p90`.
     - `evidence_gaps`: explicit list (`no_independent_clinical_station_labels`, `ct_fov_proximal_only`, `bodyparts3d_lacks_dfa`), separated from `engineering_preflight.passed`.
   - `--require-reader-inputs` exits `0` iff `engineering_preflight.passed == true` (zero holdout leakage, `0` `skin_unknown`, `0` `bone_unknown`, chart round-trip `< 1e-6`, and material-GP `fully_known_and_passing >= 58/64` (`>= 90.6%`) across all 4 sources × both `ordinary` and `stress` with median Euclidean shift `>= 14 mm` on `ordinary` and `>= 25 mm` on `stress`).
   - **Gate 2 (Neon xhigh + Sol max reviewer)**: Run all pytest suites + `audit_training_support --proposals 64 --seed 58 --require-reader-inputs`. Inspect `results/step2_training_support.json` raw numbers; do NOT advance to Step 3 until Gate 2 passes!

---

### Step 3 — Topology-Aware Dynamic Bayesian Reader + Belief-IPE + Dynamic SMPL-X Tracking (T-RO Core Module 1 — MUST PASS BEFORE RL!)

**Delegation**: `agent: task`, `model: openai-codex/gpt-6.2-sol:xhigh` for `reader/` + training/eval pipeline; `openai-codex/gpt-6.2-sol:medium` for ablations & unit tests; `agent: reviewer` + Neon xhigh Gate 3 review.

1. **Module Architecture (`reader/` package inside `NMI_idea/experiments/anatomical_gps_unified_20261126/reader/`)**:
   - `reader/belief_ipe.py`: **Relative Material Belief-IPE** (exact Proposition 3 implementation):
     - Given posterior Gaussian $(\mu_x, \Sigma_x)$ over material latent state $x$ and graph landmarks/knots:
       - Compute per-vessel and pairwise co-centered material contrasts $\Delta\tilde\xi_{ij} = \tilde\xi_i - \tilde\xi_j$ with exact propagated covariance $\Sigma_{\Delta\tilde\xi_{ij}} = H_{ij} \Sigma_x H_{ij}^\top$ (including negative cross-covariance terms!).
       - Extract top-$K$ ($K=8$) **Kirchhoff Graph Laplacian spectral eigenfunctions** $\psi_k(v)$ + intrinsic arc-length progress $\tau_e$ + distance to nearest bifurcation vertex (`SFJ`, `CFA_bif`, `CFV_bif`).
       - Apply dyadic frequency bands $p \in \{2^0, \dots, 2^5\}$ (integer harmonics on periodic $\Delta\theta$) with exact **Gaussian uncertainty attenuation** $\exp(-\frac{1}{2} p^\top \Sigma_{\Delta\tilde\xi} p)[\sin(p^\top\mu), \cos(p^\top\mu)]$.
       - Unit-tested exact invariance (`< 1e-12`) to arbitrary common material shift $u_s(h)$ and SMPL-X pose $q_t$.
   - `reader/dynamic_factor_graph.py`: **Sparse Multi-Hypothesis Dynamic Factor-Graph Reader**:
     - State: exact sparse prior $(\Lambda_0, \eta_0)$ from `MetricGraphPrior(..., coordinate_mode="material")` + per-vessel 3D tissue OU state $z_t$ (`DynamicMotionProfile.ou_transition`).
     - At each frame $t$ with probe pose $T_t$, contact force $f_t$, dwell $d_t$, tracker pose $(\hat q_t, \Sigma_{q,t})$, and anonymous echoes $Y_t \in \mathbb R^{M\times 5}$ (from `PhysicsUltrasoundKernel`, strictly label-blind!):
       1. Predict vessel world centers $\hat c_r = \Phi_{\beta, \hat q_t}(X_0(\hat{\tilde\xi}_r))$ and Jacobians $H_r = \frac{\partial \Phi}{\partial X_0} \frac{\partial X_0}{\partial \tilde\xi} L_r$ and $J_{q,r} = \frac{\partial \Phi}{\partial q}$ in probe image plane $(x_{\rm lat}, z_{\rm dep})$.
       2. Compute **Theorem 2 dynamic pullback covariance** $R_{{\rm eff}, r} = R_{\rm US} + J_{{\rm img}, q, r} \Sigma_{q,t} J_{{\rm img}, q, r}^\top$ (plus motion-blur variance from `probe_pose.speed_mm_s`).
       3. Generate top-$K_{\rm hyp}$ ($K_{\rm hyp} = 8$) **multi-hypothesis data associations** $A_t$ matching anonymous echoes to candidate in-FOV vessel roles, explicit `NULL` (clutter/specular ghost), and `MERGED` pairs (when two lumens are within acoustic resolution `merge_distance_mm`), scoring geometric Mahalanobis distance + optional physical cues (`caliber`, `compression_ratio` when $f_t > f_{\rm nom}$, `pulse_amp` when `dwell >= 0.35 s`) using physically grounded likelihoods (never role-table cheats!).
       4. Maintain $M_{\rm branch} \le 6$ persistent **global hypothesis branches** (branching when association entropy exceeds threshold, pruning/merging equivalent branches) — enabling **retroactive hypothesis switching** when later cross-window / compression / bifurcation-traceback frames refute a ghost or swapped A/V assignment!
       5. Apply **correlation-discounted information update** $\bar R = R_{\rm eff} / \rho$ ($\rho \in (0, 1]$, default tuned ONLY on training dev split, e.g. $\rho = 0.25$ for correlated consecutive frames within same window and $\rho = 0.85$ for novel view angle $\Delta\alpha \ge 25^\circ$ or active force/dwell cue, per Marques ICRA 2024 & Theorem 2) to update sparse CSC $(\Lambda, \eta)$ via Gauss-Newton / iterated EIF (`splu` solve in $O(N)$).
       6. **Retroactive Frame Re-inference API (`relabel_history()`)**: re-evaluates EVERY past frame's `(frame, slot)` echo identity posterior under the updated multi-hypothesis global posterior — directly enabling wrong→right old-frame flipping!
   - `reader/station_classifier.py`: **Hybrid Topology-Bayesian + Calibrated Belief-IPE Scorer & Split-Conformal Segment Certifier**:
     - Light PyTorch MLP/Set-Transformer head (`~60k` params, trained on RTX 4080 ONLY on training-split simulated scans, zero holdout access) taking `Belief-IPE` features + factor-graph marginal posteriors + acoustic physical cue summaries -> outputs calibrated per-echo role probabilities and **NYSORA S1–S5 topological station probabilities** (strictly defined via role co-presence + junction-relative $h$ window).
     - **Calibration & Theorem 4(i) Whole-Segment Split-Conformal Certifier**:
       - Freeze reader weights on training split; fit temperature scaling on dev split; compute whole-segment max standardized geometric residual score $S_{\rm seg} = \max_{t \in \text{seg}} \| (y_t - \hat y_t) \|_{R_{\rm eff}^{-1}}$ (plus identity margin score) on $n_{\rm cal}$ exchangeable held-out dev calibration segments; set cutoff $q_{1-\alpha} = S_{(\lceil(n_{\rm cal}+1)(1-\alpha)\rceil)}$ at $\alpha = 0.20$. Certify segment iff entire joint conformal set passes geometric/identity verifier.
   - **Mandatory Reader Ablations & Baselines (`reader/baselines.py`)**:
     - `B0_Rigid27`: faithful reproduction of legacy 27-particle rigid-shift reader (`±10 mm` grid, narrow `2.35 mm` likelihood, undiscounted).
     - `B1_NoTopology_AbsXYZ`: independent per-vessel absolute XYZ Gaussian filter without shared/relative split or Kirchhoff junctions.
     - `B2_NoPullback`: `Full_AGPS_Reader` with $\Sigma_q \equiv 0$ (drops Theorem 2 $J_q \Sigma_q J_q^\top$).
     - `B3_NoDiscount`: `Full_AGPS_Reader` with $\rho \equiv 1.0$ (tests frame-correlation overconfidence & lock-in).
     - `B4_NoBeliefIPE`: factor graph without Belief-IPE neural calibration/scorer.
     - `Full_AGPS_Reader`: full model.

2. **Step 3 Evaluation Protocol & Gate 3 Pass Criteria (`reader/eval_reader.py` -> `results/step3_reader_eval.json`)**:
   - **Training/Dev Splits ONLY during tuning** (`LOAO_OPEN3D` full S1–S5, `LOAO_BODYPARTS` S1/S2/S5, `FAMILY_HELD_OUT_ATLASES` MedSeg→Atlases S1–S4 proximal). All hyperparameters ($\rho$, branch prune threshold, Belief-IPE head weights, temperature, conformal $q_{1-\alpha}$) are **FROZEN** before touching Blender holdout.
   - **Five Quantitative Benchmarks** (paired seeds across all 6 readers, 95% bootstrap CIs):
     1. **0–40 mm Offset Sweep** (`shift_sweep_material(0, 10, 20, 30, 40)` on `LOAO_OPEN3D`): `Full_AGPS_Reader` maintains vessel identity F1 `>= 0.85` at `40 mm` shift while `B0_Rigid27` collapses (`< 0.40`).
     2. **Dynamic SMPL-X Moving-Thigh Tracking** (continuous hip flexion `0–25°`, abduction `±10°`, axial rotation `±8°`, knee flexion `0–35°`, tracker latency `60 ms`, jitter `2°–4°`, cardiac pulsation, OU tissue lag): `Full_AGPS_Reader` achieves tracking RMSE `<= 4.0 mm`, identity F1 `>= 0.85`, and statistically significant improvement over `B2_NoPullback` (paired Wilcoxon $p < 0.01$).
     3. **Old-Frame Retroactive Error Correction**: on episodes seeded with specular ghost + initial ambiguity followed by 45° re-window / compression / bifurcation traceback, `Full_AGPS_Reader` flips `>= 60%` of initially misclassified `(frame, slot)` echoes from wrong→right with `< 5%` right→wrong regression (ending v15's `0/71` and beating `B3_NoDiscount`).
     4. **NYSORA S1–S5 Topological Role-Station Recognition** (on supported stations per split; non-clinical evidence gap noted): macro-F1 `>= 0.85` on `LOAO_OPEN3D` (S1–S5) and `>= 0.80` on `FAMILY_HELD_OUT_ATLASES` proximal (S1–S4).
     5. **Whole-Segment Split-Conformal Calibration (Theorem 4(i))**: held-out certified segment empirical violation rate `<= 0.20` (with 95% Clopper–Pearson upper bound reported on independent segment units), strictly outperforming uncalibrated `B3_NoDiscount`.
     6. **Locked Zero-Shot Blender Holdout Diagnostic** (run in an isolated subprocess with `AGPS_EVAL_HOLDOUT_ALLOWED=1` ONLY AFTER Gate 3 training/LOAO checks pass and weights are frozen; NO iteration back onto Blender!): report separately on `blender_nysora_thigh` (direct roles S1/S2/S5 vs. inferred `R_DFA` S3/S4) and `blender_legacy_routes` physically reachable `137/147` stations.
   - **Gate 3 (Neon xhigh + Sol max reviewer)**: NO RL training begins until Gate 3 is marked `PASS` in `HANDOFF.md`! If any metric misses gate (e.g., old-frame flip `< 60%` or `B2_NoPullback` shows no gap), Neon diagnoses root cause and returns targeted fixes to Sol for re-verification.

---

### Step 4 — Dynamic Closed-Loop Scanning Environment, Fair Baselines & Asymmetric Privileged-Critic Active RL (T-RO Core Module 2)

**Delegation**: `agent: task`, `model: openai-codex/gpt-6.2-sol:xhigh` for `env/` + `rl/`; `openai-codex/gpt-6.2-sol:medium` for fair MPC/fixed-recheck baselines & paired evaluator; `agent: reviewer` + Neon xhigh Gate 4 review.

1. **Closed-Loop Dynamic Active-Scanning Environment (`env/active_scan_env.py`)**:
   - Integrates frozen Gate-3 `Full_AGPS_Reader` + `SMPLXLegKinematics` (dynamic moving thigh!) + `PhysicsUltrasoundKernel` (world-anchored specular ghosts, bone/fascia acoustic shadows, force-dependent venous compression, dwell-gated arterial pulsatility, speed-dependent motion blur).
   - **Short-Segment Executable Action Space (IDENTICAL across ALL policies — RL, Strong MPC, Fixed Recheck!)**:
     - Discrete high-IG skill + bounded continuous parameter $\Delta u \in [-1, 1]^2$, executed by a deterministic short-segment ProDMP/kinematic trajectory generator respecting max probe speed (`35 mm/s`), tilt rate (`45°/s`), force limits (`[1.0, 5.5] N`), and exact wall-clock execution cost $\Delta t$:
       1. `TRACK_ADVANCE(dh, dlateral)`: advance along posterior MAP vessel centerline toward next required station at nominal force (`2.5 N`);
       2. `REWINDOW_TILT(tilt_deg, lateral_shift)`: acoustic re-windowing (`±30°..45°`) to clear bone shadow & extinguish world-anchored specular ghosts;
       3. `COMPRESSION_PROBE(delta_f)`: controlled compression pulse (`+1.5..+2.5 N`, `0.6 s`) to separate compressible vein from non-compressible artery;
       4. `DWELL_PULSE()`: brief low-speed stationary dwell (`0.40 s`) to observe arterial pulsatility;
       5. `TRACEBACK_JUNCTION(direction, dist_mm)`: track candidate lumen toward nearest topological junction (`SFJ` / `CFA_bif` / `CFV_bif`) for Kirchhoff topological disambiguation, then return;
       6. `CERTIFY_AND_NEXT()`: lock current station/segment certification (using Theorem 4(i) conformal certifier) and transition to next target station (enables Early-Stop!).
   - **Non-Trivial Physical Regimes (Explicitly Fixing Previous "Passive MPC Already Captures 100%" Flaw)**:
     - Evaluate across 3 stratified regimes (all under **strict time budget $T = 45\text{ s}$** for thigh S1→S5 continuation):
       - `R1_Standard_Dynamic`: `ordinary` material-GP anatomy + continuous SMPL-X thigh motion (`±12°` hip/knee) + tracker lag `60 ms` + jitter `2°` + light specular clutter.
       - `R2_Hard_Ambiguity_Shadow`: `stress` material-GP anatomy (`25–42 mm` shifts) + persistent world-anchored specular ghosts near CFA/CFV/SFJ + lateral femoral condyle acoustic shadow + elevated venous/arterial caliber overlap.
       - `R3_TightBudget_Dynamic`: `R2` + tighter time budget ($T = 32\text{ s}$) + higher tracker jitter (`4°`) + faster leg movement — where brute-force fixed recheck runs out of time and passive MPC misidentifies/loses track!
   - **Explicit Decomposed Episode Outcomes (Mandatory User Requirement)**:
     - Every episode logs 3 strictly separated primary binary/rate outcomes + efficiency/safety metrics:
       1. **`target_captured` (拍到目标)**: all required target vessels observed unoccluded inside probe FOV across target stations;
       2. **`identity_correct` (认对身份)**: final global MAP identity matches ground-truth vessel role at all required stations (plus NYSORA topological role-station accuracy and old-frame wrong→right flip rate);
       3. **`continuation_completed` (完成续扫)**: strict end-to-end success within time budget $T$: `target_captured` AND `identity_correct` AND continuous branch tracking (no wrong-branch handoff, tracking error `< 6.0 mm` along verified segment) AND conformal segment certified without violation;
       4. **Efficiency/Safety**: scan time (s), probe travel (mm), active-probe count, excess force integral ($\int (f - f_{\rm nom})_+ dt$), conformal false-certification rate.

2. **Mandatory Fair Baselines (`env/baselines.py`) — STRICT PARITY**:
   - All baselines use the **EXACT SAME frozen Gate-3 reader**, **EXACT SAME action primitives & execution kinematics**, **EXACT SAME time budget $T$**, and **PAIRED random seeds (common random numbers)** on every test anatomy/motion tape:
     - `MPC_Passive`: receding-horizon MPC tracking posterior MAP centerline (`TRACK_ADVANCE` + `CERTIFY_AND_NEXT` only — measures baseline without active probing).
     - `MPC_Strong_InfoGain`: **full-permission strong MPC** with 2-step lookahead over ALL 6 skills maximizing expected acquisition utility $C(\tau, b) = \mathbb E[\Delta\mathcal H(b) + U_{\rm cov} \mid b, \tau] - \lambda C_{\rm exec}(\tau)$ (matches `Material_Atlas_Explanation.pdf` Slide 11; $\lambda$ tuned on training dev split for highest `continuation_completed`).
     - `Fixed_Recheck_Schedule`: `MPC_Passive` + deterministic mandatory `REWINDOW_TILT` + `COMPRESSION_PROBE` at every station before advancing (measures brute-force rechecking cost vs. selective active RL).

3. **Asymmetric Privileged-Critic Active RL (`rl/asymmetric_active_rl.py`) — Fixing All 4 Previous RL Failure Modes**:
   - **Public Policy $\pi_\theta(a \mid h_t)$** (zero privileged inputs at deployment!): receives Belief-IPE relative material embedding + multi-hypothesis posterior entropy & top-2 margin + spatial hypothesis disagreement field $\tilde I(x)$ over candidate probe actions + per-station completion & conformal certification status + remaining time fraction $T_{\rm rem}/T$ + tracker velocity/jitter norm.
   - **Unbiased Asymmetric Privileged Critic $V_\phi(h_t, s^{\rm true}_t)$ / $Q_\phi(h_t, s^{\rm true}_t, a)$** (Baisero & Amato AAMAS 2022): during training ONLY, critic additionally receives true latent anatomy $x^{\rm true}$, true vessel identities, true ghost/shadow positions, and true current pose $q^{\rm true}_t$ — **completely replacing v9–v15's high-variance 2-tape noisy counterfactual teacher**.
   - **Decomposable Station-Level Potential-Based Reward (fixing $S = \prod Y_r$ credit collapse)**:
     - Per-step dense potential shaping on true-identity log-posterior improvement $\gamma \Phi(b_{t+1}, s^{\rm true}) - \Phi(b_t, s^{\rm true})$ (zero policy bias by Ng-Harada-Russell theorem!) + per-station verified capture & identity bonus + terminal `continuation_completed` bonus (`+5.0`) - explicit time cost (`-c_t Δt`) - travel cost (`-c_d Δd`) - excess force cost (`-c_f Δf_+ Δt`) - wrong-branch/false-certification penalty (`-3.0`).
   - **Training Recipe (RTX 4080, 3 independent training seeds: `seed ∈ {11, 22, 33}`)**:
     - Stage 4a (**AAWR Warm-Start**): collect 2,000 mixed episodes from `MPC_Strong_InfoGain` + $\epsilon$-exploratory active skills on training split (`LOAO_OPEN3D` train sources + material-GP draws); fit privileged critic $V_\phi(h_t, s^{\rm true})$ and run advantage-weighted regression updates to initialize $\pi_\theta$.
     - Stage 4b (**Online Asymmetric PPO Fine-Tuning**): 300k environment steps (~6,000 episodes) with GAE computed from privileged $V_\phi(h_t, s^{\rm true})$, clipped PPO surrogate (`clip=0.2`), adaptive KL anchor against warm-start to prevent collapse, entropy annealing, early checkpoint selection strictly on **training-dev split** (NEVER holdout!).
   - **Mandatory RL Ablations** (all 3 seeds, paired evaluation):
     - `AsymRL_AGPS` (full: Material-Chart GP + Belief-IPE + Topology + Dynamic Pullback + Privileged Critic);
     - `AsymRL_SymmetricCritic` (drops privileged critic → proves unbiased asymmetric critic removes training variance);
     - `AsymRL_NoBeliefIPE` (replaces Belief-IPE with raw coordinates → proves Proposition 3 embedding value);
     - `AsymRL_NoTopology` (uses `B1_NoTopology_AbsXYZ` reader → proves Theorem 1 topology value).

4. **Gate 4 Evaluation & Pass Criteria (`rl/eval_closed_loop.py` -> `results/step4_closed_loop_tro.json`)**:
   - Evaluate first on held-out training split **`LOAO_OPEN3D`** (200 paired episodes per regime `R1/R2/R3` × 3 seeds), THEN run **ONE locked zero-shot evaluation** on **Blender Holdout** (`AGPS_EVAL_HOLDOUT_ALLOWED=1`, isolated process, frozen policy & frozen reader, zero tuning on Blender!) covering both `blender_nysora_thigh` (direct S1/S2/S5 and inferred-DFA S3/S4 reported separately) and `blender_legacy_routes` physically reachable `137/147` subset.
   - **Gate 4 Pass Criteria (Neon xhigh + Sol max reviewer)**:
     1. Across `R2_Hard_Ambiguity_Shadow` and `R3_TightBudget_Dynamic` (on `LOAO_OPEN3D` and zero-shot Blender): `AsymRL_AGPS` achieves statistically significant improvement over `MPC_Passive` on **认对身份** (`+15 pp` min on R2/R3) and **完成续扫** (`+15 pp` min on R2/R3, paired McNemar $p < 0.01$);
     2. **Pareto dominance over BOTH strong baselines** (same reader, same actions, same budget $T$):
        - Beats `MPC_Strong_InfoGain` on **完成续扫** (`+5 pp` min on R2/R3) at equal or lower scan time;
        - Matches or beats `Fixed_Recheck_Schedule` on **完成续扫** while cutting scan time & probe travel by **`>= 20%`** (and strictly beating `Fixed_Recheck_Schedule` on `R3_TightBudget_Dynamic` where fixed rechecking times out!);
     3. All 3 ablations (`SymmetricCritic`, `NoBeliefIPE`, `NoTopology`) underperform `AsymRL_AGPS` on `R2/R3` `continuation_completed`, establishing causal attribution to each theoretical component;
     4. Conformal certification false-positive rate stays `<= 0.20` on held-out independent episodes.

---

### Step 5 — T-RO Evidence Synthesis & Manuscript Mapping (Neon xhigh)

- Write `NMI_idea/experiments/anatomical_gps_unified_20261126/TRO_EVIDENCE_SUMMARY.md` mapping EVERY theoretical claim to exact verified tables/JSON paths and reproducible commands:
  - **Supported Claims Table**: Theorem 1 (Material Gauge + Kirchhoff Sparse GP + near-zero rejection duality), Theorem 2 (Dynamic SMPL-X Pullback + Correlation Discount $\rho$), Proposition 3 (Relative Material Belief-IPE shift/pose invariance & RL gain), Theorem 4(i) (Whole-Segment Split-Conformal Risk Certification), plus Active RL vs. Strong MPC & Fixed Recheck Pareto dominance decomposed by `拍到目标 / 认对身份 / 完成续扫`.
  - **Explicit Evidence Gaps & Limitations Section** (mandatory honesty for T-RO):
    1. Single real CT participant (`medseg_right`, 90 mm proximal groin FOV only — S5 popliteal comes from anatomical atlases);
    2. Single independent holdout anatomy (`blender_holdout`, with `R_DFA` inferred via neutral arterial path rather than direct raw mesh);
    3. No clinician-annotated NYSORA clinical window labels (`clinical_station_labels_available = false`; station evaluation uses role co-presence + verified junction windows);
    4. Physics-kernel echo simulation (label-blind, specular ghosts, shadows, compression, pulsatility, SMPL-X motion), NOT yet raw B-mode ultrasound images or physical RM75 phantom closed-loop trials.

---

### Step 6 — IJRR Extension: Empirical Witness of the Embodied Sufficiency Gap (ONLY AFTER Gate 4 & Step 5 PASS!)

**Delegation**: `agent: task`, `model: openai-codex/gpt-6.2-sol:xhigh` for `ijrr/`; `agent: reviewer` + Neon xhigh Gate 6 review.

- Implement `ijrr/embodied_sufficiency_experiment.py` directly validating **Theorem 4(ii) (`V8.3` Physical Defect & `V8.4` Four-Cell Completion Contrast + Irreversible Cut)** on the Gate-3 reader + Gate-4 dynamic bifurcation simulator (`SFJ` / `CFA_bif` / `CFV_bif`):
  1. **Matched-Information Interventional Four-Cell Experiment**:
     - Balanced binary branch truth $\theta \in \{A, B\}$ (e.g., SFA vs. DFA at `CFA_bif`, or GSV vs. CFV at `SFJ`), early observation $Y$ from Gate-3 reader with measured identifiability $2\hat p - 1 = \mathrm{TV}(P_{Y|A}, P_{Y|B}) > 0$.
     - Intervene to establish physical exit mode $M \in \{A, B\}$ with identical remaining budget $T$ and matched control statistic, sweeping $T$ across the reconnection threshold $t_0 + d_0$ (where $t_0$ is direct branch completion time and $d_0$ is minimum physical backtracking + reconnection time to the bifurcation vertex).
     - Measure empirical cell means $\hat c_{AA}, \hat c_{BB}, \hat c_{AB}, \hat c_{BA}$, empirical contrast $\hat\Gamma_{\mathcal R}(T)$, and theoretical prediction $2(2\hat p - 1)(\hat s(T) - \hat r(T))$ with bootstrap 95% CIs across the $T$ sweep.
  2. **Irreversible Cut Bracket Verification**:
     - For deadlines $T \in [t_0, t_0 + d_0)$ (verified $\hat r(T) \approx 0$ — wrong-branch exit cannot physically reconnect before deadline!), verify empirical optimal value $\hat V^*_{\mathcal R}(T)$ sits inside Theorem 4(ii)'s bracket $(1-\alpha)\frac{1+\hat{\mathcal I}_{\rm in}(T)}{2} \le \hat V^*_{\mathcal R}(T) \le \frac{1+\hat{\mathcal I}_{\rm out}(T)}{2}$, and demonstrate that Gate-4's early active probing before the irreversible cut strictly lifts completion over late-commit / passive policies with statistically identical unprobed initial information.
  3. **Gate 6 (Neon xhigh + Sol max reviewer)**: Theoretical vs. empirical $\Gamma_{\mathcal R}(T)$ agreement within 95% CI across deadline sweep; irreversible cut bracket holds without filtering out timeouts; write `IJRR_EMBODIED_SUFFICIENCY_REPORT.md` cleanly separated from T-RO claims.

---

## Critical Files & Anchors

1. `NMI_idea/experiments/anatomical_gps_unified_20261126/sim/metric_graph_gp.py` (`_TrainingValidity` L629–696, `MetricGraphPrior` L400–593, `MetricGraphAnatomyGenerator` L781–923) — currently binds `_CONTACT_ATLAS` (`manifest["betas"]` open patch, causing 199–383 `skin_unknown` and 55 false `outside_skin`) and isotropic Cartesian XYZ noise (`0/32` yield). Anchor for Step 2 material-chart dual GP upgrade.
2. `NMI_idea/experiments/anatomical_gps_unified_20261126/sim/smplx_dynamic_atlas.py` (`SMPLXLegKinematics` L121–410, `DynamicMotionProfile` L437–552) — already computes exact 55-joint SMPL-X kinematics & analytic pose Jacobian $J_q$, currently defaults `beta=None` to `manifest["betas"]`; anchor for `beta=0` canonical transport & material frame $G_{\beta, q_t}$.
3. `NMI_idea/experiments/anatomical_gps_unified_20261126/sim/anatomy_assets.py` (`TrainingAnatomyBank.load` L182–360) — loads all 4 training sources + bone occupancy in `beta=0` SMPL-X; anchor for adding SHA-verified `NEUTRAL_SMPLX_SURFACE_PATH` and domain bounds without touching raw source centers.
4. `NMI_idea/experiments/anatomical_gps_unified_20261126/sim/physics_ultrasound_kernel.py` (`PhysicsUltrasoundKernel.render` L375–425) — label-blind echo renderer with world-anchored specular ghosts, compression, dwell pulse, shadowing, and CRN pairing; reused directly without modification by Step 3 reader eval, Step 4 RL env, and Step 6 IJRR witness.
5. `NMI_idea/experiments/anatomical_gps_unified_20261126/sim/holdout_eval_assets.py` (`load_blender_legacy_routes` L236–306, `load_blender_nysora_thigh` L308–410) — authorization-gated holdout loader (`AGPS_EVAL_HOLDOUT_ALLOWED=1`); strictly isolated for locked final zero-shot evaluation in Gates 3 & 4 only.

---

## Verification

All commands run from working directory `/media/camp/EXT_DRIVE/RealUS_playground` with:
- Python binary: `/media/camp/EXT_DRIVE/envs/genesis/bin/python`
- Pytest binary: `/media/camp/EXT_DRIVE/envs/genesis/bin/pytest`
- Common env vars: `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:.`

1. **Gate 2 (Step 2 Unblock — MUST PASS FIRST)**:
   - Unit + chart + firewall + theory regression:
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider \
       NMI_idea/experiments/anatomical_gps_unified_20261126/theory/verify_theory_math.py \
       NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_sim_and_firewall.py \
       NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_material_chart.py
     ```
     Expected: 100% pass; `RightLegMaterialChart` round-trip error `< 1e-10`, Jacobian error `< 1e-4`, `SMPLXLegKinematics` `beta=0` pose Jacobian error `< 1e-5 mm/rad`, holdout firewall enforced.
   - 3-axis training support & effective yield audit:
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/python -m sim.audit_training_support \
       --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step2_training_support.json \
       --proposals 64 --seed 58 --require-reader-inputs
     ```
     Expected: exits `0`; `engineering_preflight.passed == true`; `skin_unknown == 0` & `bone_unknown == 0` across all 4 sources on `thigh_nysora`; material-GP `fully_known_and_passing >= 58/64` (`>= 90.6%`) across all 8 `(source × difficulty)` profiles with median 3D shift `>= 14 mm` (`ordinary`) and `>= 25 mm` (`stress`); `evidence_gaps` explicitly lists `no_independent_clinical_station_labels`, `ct_fov_proximal_only`, `bodyparts3d_lacks_dfa`.

2. **Gate 3 (Step 3 Reader + Dynamic Tracking — MUST PASS BEFORE STEP 4 RL)**:
   - Unit tests (Belief-IPE invariance, dynamic pullback covariance, old-frame flip):
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
       PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider \
       NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_reader_and_tracking.py
     ```
   - Training/LOAO reader benchmark & ablations (NO holdout env var):
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/python -m reader.eval_reader \
       --split LOAO_OPEN3D --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step3_reader_loao.json
     ```
     Expected: 40 mm shift F1 `>= 0.85` (vs `B0_Rigid27 < 0.40`), dynamic tracking RMSE `<= 4.0 mm` & beats `B2_NoPullback` ($p < 0.01$), old-frame wrong→right flip `>= 60%` (`< 5%` right→wrong), conformal segment violation `<= 0.20`.
   - Locked zero-shot Blender holdout eval (isolated process ONLY after freezing reader):
     ```bash
     AGPS_EVAL_HOLDOUT_ALLOWED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/python -m reader.eval_reader \
       --split FULL_TO_BLENDER --checkpoint NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/reader_frozen.pt \
       --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step3_reader_blender_zeroshot.json
     ```

3. **Gate 4 (Step 4 Closed-Loop Active RL vs. Fair Strong MPC & Fixed Recheck)**:
   - Train 3 seeds on training split only (RTX 4080, NO holdout env var):
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/python -m rl.asymmetric_active_rl \
       --split LOAO_OPEN3D --seeds 11 22 33 --device cuda \
       --out-dir NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/rl_tro
     ```
   - Paired closed-loop evaluation on `LOAO_OPEN3D`, then locked zero-shot Blender holdout (`AGPS_EVAL_HOLDOUT_ALLOWED=1` in separate subprocess):
     ```bash
     PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
       /media/camp/EXT_DRIVE/envs/genesis/bin/python -m rl.eval_closed_loop \
       --split LOAO_OPEN3D --checkpoint-dir NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/rl_tro \
       --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step4_closed_loop_loao.json
     ```
     Expected: separately reports `target_captured`, `identity_correct`, `continuation_completed`, scan time, probe travel, and conformal risk across `MPC_Passive`, `MPC_Strong_InfoGain`, `Fixed_Recheck_Schedule`, `AsymRL_AGPS`, and ablations; `AsymRL_AGPS` achieves statistically significant Pareto dominance over both strong MPC and fixed recheck on R2/R3 (`>= +15 pp` vs passive, `>= +5 pp` vs strong MPC, `>= 20%` time/travel reduction vs fixed recheck).

4. **Gate 6 (Step 6 IJRR Extension — ONLY After Gate 4 + Step 5 Pass)**:
   ```bash
   PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
     /media/camp/EXT_DRIVE/envs/genesis/bin/python -m ijrr.embodied_sufficiency_experiment \
     --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step6_ijrr_embodied_sufficiency.json
   ```
   Expected: empirical 4-cell contrast $\hat\Gamma_{\mathcal R}(T)$ matches $2(2\hat p - 1)(\hat s(T) - \hat r(T))$ within 95% CI across deadline sweep; irreversible cut bracket holds on all retained histories including timeouts.

---

## Assumptions & Contingencies

- **Contingency A (Groin Pelvis-Merge Ray Geometry in `RightLegMaterialChart`)**: At `y ∈ [-540, -475]` where left/right legs merge into the pelvis, if any medial-posterior ray from $c_{\rm axis}(h)$ produces corridor thickness $r_{\rm skin} - r_{\rm bone} < 6.0\text{ mm}$ around the femoral neck/hip bone, smoothly shift $c_{\rm axis}(h)$ anteriorly by up to `8 mm` above `y = -530` toward the right hip-joint/anterior-thigh axis (keeping $c_{\rm axis}(h)$ strictly inside `bone_occupancy >= 0.55` or strictly posterior to all training groin vessels) and re-run round-trip + skin/bone validity tests until `fully_known_and_passing >= 90.6%` holds on all 4 sources.
- **Contingency B (Exact Material-Chart Linearization vs. Large Nonlinear Step in Step 3 Reader)**: If single-step EIF linearization of $X_0(\tilde\xi)$ degrades accuracy at `40 mm` circumferential shift (`Δθ ≈ 0.55 rad`) during Gate 3, run 3 Gauss-Newton re-linearization iterations per keyframe (iterated EIF / MAP solve on the sparse CSC graph, still `< 5 ms/frame`) and initialize the global multi-hypothesis filter with a 5-angle coarse circumferential bank (`θ_0 ∈ {-0.6, -0.3, 0, +0.3, +0.6} rad`) merged into the Max-Mixture hypotheses.
- **Contingency C (RL Exploration vs. Strong MPC Parity in Step 4)**: If `AsymRL_AGPS` ties `MPC_Strong_InfoGain` on `R1_Standard_Dynamic` (expected when standard episodes are already near-ceiling), do NOT inflate `R1` claims — evaluate Pareto separation primarily on `R2_Hard_Ambiguity_Shadow` and `R3_TightBudget_Dynamic` (where 2-step greedy lookahead underperforms multi-step topological traceback + selective early stopping), and if `AsymRL_AGPS` still trails `MPC_Strong_InfoGain` on R2/R3, return to Sol to inspect credit assignment, increase `TRACEBACK_JUNCTION` reward horizon / AAWR temperature $\beta_{\rm AWR}$, and re-verify on `LOAO_OPEN3D` before touching Blender holdout.
- **Contingency D (Blender Holdout Zero-Shot Performance Gap)**: Because Blender holdout is strictly zero-shot and never tuned on, if Gate 4 shows a gap on Blender despite passing `LOAO_OPEN3D`, log the exact failure breakdown (e.g., 6 proximal arterial gaps `[-545,-520]`, inferred `R_DFA` mismatch, or skin/contact offset) transparently in `TRO_EVIDENCE_SUMMARY.md` as an empirical generalization finding rather than post-hoc tuning on holdout.

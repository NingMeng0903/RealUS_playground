# Anatomical GPS Continuation Plan (T-RO First, Then IJRR)

**Experiment Directory**: `NMI_idea/experiments/anatomical_gps_unified_20261126`  
**Workstation Environment**: `/media/camp/EXT_DRIVE/envs/genesis/bin/python` (Python 3.10.20, PyTorch 2.12.0+cu126 on NVIDIA GeForce RTX 4080 Laptop GPU, NumPy, SciPy, Numba 0.65.0, trimesh 4.11.5, pytest 9.0.3; set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to avoid broken system `dash` `pytest11` plugin).  
**Original division of roles (through Gate 2)**: Foreground Gemini 4 Neon (XHigh) owned theory, design and gatekeeping; Sol / Luna implemented and ran experiments, with independent reviewer audits. See `HANDOFF.md` for the actual decisions and signatures.

**Continuation ownership (Step 3 onward):** GPT-6.2 Astra Max takes over Neon's pending design and raw-evidence decisions. Sol/Luna implement and run experiments; Sol Max independently reviews each milestone before Astra's decision in `HANDOFF.md`. Historical model attributions, split definitions and numerical gates are unchanged.

---

## 1. Current Status Audit (Baseline Before Step 2 Fixes)

### 1.1 已完成 (Completed & Live-Verified)

| Artifact / Module | Verification Evidence | Reproducible Command |
|---|---|---|
| **Step 1 — Theory & Numerical Witnesses**: `theory/THEORY_FOUNDATIONS.md`, `theory/theorems_tro_ijrr.tex`, `theory/verify_theory_math.py` | **6/6 pytest PASSED** (`Kirchhoff trace=0`, `Gauge error=0`, `banded GP cov err=1.332e-15`, `IPE attenuation verified`, `split conformal rank bound=0.1997`, `V8.3/V8.4 max identity err=5.551e-16`). | `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider NMI_idea/experiments/anatomical_gps_unified_20261126/theory/verify_theory_math.py` |
| **Step 2a — Label-Blind Physics Ultrasound Kernel**: `sim/physics_ultrasound_kernel.py` | Unit-tested in `tests/test_sim_and_firewall.py`: byte-identical echoes under role/order permutations, world-anchored specular ghost extinguished under 45° re-window, paired CRN noise, force compression & 0.35 s dwell pulse, bone/fascia shadowing & merge. | Included in `tests/test_sim_and_firewall.py` command below. |
| **Step 2b — Holdout Firewall & Asset Separation**: `sim/anatomy_assets.py`, `sim/holdout_eval_assets.py`, `tests/test_sim_and_firewall.py` | **12/12 pytest PASSED**: holdout denied without `AGPS_EVAL_HOLDOUT_ALLOWED=1` and after revocation; training imports never touch holdout; Open3D DFA extracted (44 rows); Blender holdout preserves 6 proximal arterial gaps (`[-545,-520]`), separates `R_DFA` (`support_source=2`), and isolates 10 bone-occluded stations (`137/147` reachable). | `PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_sim_and_firewall.py` |

### 1.2 待复验 (Needs Re-verification After Interface Fixes)

| Item | Why Re-verification Is Required | Target Gate |
|---|---|---|
| `sim/smplx_dynamic_atlas.py` (`SMPLXLegKinematics`, `DynamicMotionProfile`) | Passed `2.012e-08 mm/rad` pose-Jacobian test only on `manifest["betas"]` (`beta=[+0.75,+0.05,...]`) inside `shaft_roi_v2` (`y ∈ [-1209.5, -562.1]`). Must be re-verified at `beta=zeros(300)` (true training/holdout frame) and across groin+thigh+popliteal `y ∈ [-880, -475]` via `RightLegMaterialChart`. | Gate 2 |
| `sim/metric_graph_gp.py` (`MetricGraphPrior`, `MetricGraphAnatomyGenerator`, `_TrainingValidity`) | Previously passed on `medseg_right` with `require_known=False` silently bypassing 199/280 `skin_unknown` points; all 8 `(source × difficulty)` profiles failed `fully_known_and_passing == 0/32` in baseline `results/step2_training_support.json`. | Gate 2 |
| `results/step2_training_support.json` & `sim/audit_training_support.py` | Baseline exited 2 with `reader_input_preflight.passed = false` (13 blockers). Must be re-run separating engineering preflight from genuine anatomical coverage limits. | Gate 2 |

### 1.3 第一个阻塞点 (First Blocker — 3-Axis Root-Cause Diagnosis & Coordinate-Fix Rationale)

1. **Axis 1 — Original CT Anatomical Coverage (Genuine Data Scope Limits — Do NOT Fabricate)**:
   - `medseg_right` spans only **90 mm** (31 native 3-mm slices; 17 aligned 5-mm stations `y ∈ [-560, -480] mm`, groin/proximal thigh only): supports `S1_CFV_CFA` (17), `S2_SFJ_MICKEY` (9), `S3_ART_BIF` (9), `S4_VEN_BIF` (7), and **0 rows of `S5_POPLITEAL` (`[-875, -825]`)**. `FAMILY_HELD_OUT_ATLASES` therefore evaluates proximal S1–S4 only.
   - `bodyparts3d_right` and `bodyparts3d_left_mirrored` have **zero DFA mesh** in `aligned_vessel_surfaces.npz`, hence 0 rows for `S3_ART_BIF` and `S4_VEN_BIF`. `LOAO_BODYPARTS` evaluates S1/S2/S5; full S1–S5 cross-atlas evaluation runs on `LOAO_OPEN3D` and `FULL_TO_BLENDER` (with Blender `R_DFA` flagged `support_source=2`).
   - **Evidence Gap**: `clinical_station_labels_available = false`. Station evaluation uses topological role co-presence + verified junction-proximity windows (`SFJ`, `CFA_bif`, `CFV_bif`, popliteal band), explicitly reported as a non-clinical topological surrogate.
2. **Axis 2 — Source-to-Harmonic-Atlas & Skin/Bone Registration (Engineering Bugs Fixed in Step 2)**:
   - **Bug 2.1 (48 mm SMPL-X `beta` Frame Mismatch + Open Groin Cut)**: All training assets, `bonehub_right_leg_occupancy.npz`, `neutral_smplx_surface_mm.npz` (`sha256: 882b54e9bfefaa6d5329fcb056a81a3750b4b01e4de456f088d39eade337c6ba`), and Blender holdout live in **`Male SMPL-X beta=0` neutral T-pose** (`max|v_beta0 - v_ns| = 6.1e-5 mm`). Legacy `_TrainingValidity` used `contact_multichart_right.npz` (`manifest["betas"]`, 45.27 mm shorter leg, open groin cuts at `y ∈ [-540.42, -462.82]`). Switching to closed `neutral_smplx_surface_mm.npz` (0 boundary edges below `y = 0 mm`) eliminates all `skin_unknown` and all false `outside_skin` hits on atlases, leaving only real anatomical differences (2 nominal MedSeg GSV knots protruding `2.65–5.67 mm` beyond `beta=0` anterior skin).
   - **Bug 2.2 (Harmonic Chart Domain Gap)**: Sealed `shaft_roi_v2/atlas_right.npz` stops at `y = -562.14 mm`, excluding 100% of MedSeg CT knots (`[-560, -480]`). Replaced for the T-RO `beta=0` groin+thigh+full-leg domain by `sim/material_chart.py` (`RightLegMaterialChart`).
   - **Bug 2.3 (BodyParts3D Popliteal Bone Overlay)**: Skin-only-aligned BodyParts3D meshes sit `10–15 mm` anteriorly inside the femoral condyles at `y ∈ [-875, -740]`. Raw `AnatomySource` centers remain untouched; material encoding logs exact projection residuals transparently (`nominal_projection_audit`).
   - **Evaluation-Impact Note**: Raw `AnatomySource.centers_mm` and Blender holdout assets are NEVER modified. All perturbed samples are still audited independently against raw `neutral_smplx_surface_mm.npz` + `bonehub_right_leg_occupancy.npz`.
3. **Axis 3 — Perturbed Effective Sample Yield (`0/32` Baseline)**:
   - Isotropic 3D Cartesian noise (`20–38 mm`) pushes subcutaneous GSV (`1.8–8.4 mm` under skin) into air and popliteal vessels into femoral condyles (>95% rejection, breaking Generation–Inference Duality).
   - Moving the Metric-Graph Whittle–Matérn GP prior directly into unconstrained annular material coordinates $\tilde\xi = (\theta, h, \eta_d)$ ($d = d_{\min} + (d_{\max}-d_{\min})\sigma(\eta_d)$) preserves exact sparse CSC precision & Theorem 1 Material Gauge cancellation while achieving `>= 90.6%` valid yield at 15–38+ mm Euclidean shifts.

---

## 2. Step-by-Step Execution & Gate Protocol

Every milestone requires a two-layer gate: (1) `agent: reviewer` code/test/repro audit; (2) Foreground Neon (XHigh) raw-evidence audit logged in `HANDOFF.md` (`PASS / FIX_AND_REVERIFY / INSUFFICIENT_EVIDENCE`).

- **Step 0**: Persist `PLAN.md` and `HANDOFF.md`.
- **Step 2 (Gate 2 — Unblock)**: `beta=0` canonical alignment (`anatomy_assets.py`, `smplx_dynamic_atlas.py`), `sim/material_chart.py` (`RightLegMaterialChart`), dual material-space `MetricGraphPrior`/`MetricGraphAnatomyGenerator` (`metric_graph_gp.py`), theory remark (`THEORY_FOUNDATIONS.md`, `theorems_tro_ijrr.tex`), upgraded 3-axis `audit_training_support.py`, and tests (`test_material_chart.py`, `test_sim_and_firewall.py`).
  - Pass criteria: 100% pytest pass, chart round-trip `< 1e-10`, Jacobian `< 1e-4`, `beta=0` SMPL-X pose Jacobian `< 1e-5 mm/rad`, `audit_training_support --proposals 64 --seed 58 --require-reader-inputs` exits `0` (`skin_unknown=0`, `bone_unknown=0`, `fully_known_and_passing >= 58/64` on all 8 `(source × difficulty)` thigh profiles, median 3D shift `>= 14 mm` ordinary / `>= 25 mm` stress).
- **Step 3 (Gate 3 — T-RO Reader First, Before RL)**: `reader/` (`belief_ipe.py`, `dynamic_factor_graph.py`, `station_classifier.py`, `baselines.py`, `eval_reader.py`) + `tests/test_reader_and_tracking.py`.
  - Baselines: `B0_Rigid27`, `B1_NoTopology_AbsXYZ`, `B2_NoPullback`, `B3_NoDiscount`, `B4_NoBeliefIPE`, `Full_AGPS_Reader`.
  - Pass criteria on `LOAO_OPEN3D` (frozen before Blender): 40 mm shift F1 `>= 0.85` (vs `B0_Rigid27 < 0.40`); dynamic SMPL-X tracking RMSE `<= 4.0 mm`, F1 `>= 0.85`, beats `B2_NoPullback` (Wilcoxon $p < 0.01$); old-frame wrong→right flip `>= 60%` (`< 5%` right→wrong); NYSORA S1–S5 macro-F1 `>= 0.85` (`LOAO_OPEN3D`) & S1–S4 `>= 0.80` (`FAMILY_HELD_OUT_ATLASES`); split-conformal violation `<= 0.20`. Then run ONE locked zero-shot Blender holdout diagnostic (`AGPS_EVAL_HOLDOUT_ALLOWED=1`).
- **Step 4 (Gate 4 — T-RO Closed-Loop Active RL & Fair Baselines)**: `env/active_scan_env.py`, `env/baselines.py`, `rl/asymmetric_active_rl.py`, `rl/eval_closed_loop.py`.
  - Strict parity (same frozen Gate-3 reader, same 6 executable skills, same budget $T$, paired CRN seeds): `MPC_Passive`, `MPC_Strong_InfoGain`, `Fixed_Recheck_Schedule`, `AsymRL_AGPS`, plus ablations `AsymRL_SymmetricCritic`, `AsymRL_NoBeliefIPE`, `AsymRL_NoTopology`.
  - Separate primary metrics: `target_captured` (拍到目标), `identity_correct` (认对身份), `continuation_completed` (完成续扫), plus scan time, travel, excess force, conformal risk across `R1_Standard_Dynamic`, `R2_Hard_Ambiguity_Shadow`, `R3_TightBudget_Dynamic` (3 seeds `11, 22, 33` on `LOAO_OPEN3D`, then ONE locked zero-shot Blender eval).
  - Pass criteria: on R2/R3, `AsymRL_AGPS` beats `MPC_Passive` by `>= +15 pp` on `identity_correct` and `continuation_completed` (McNemar $p < 0.01$), beats `MPC_Strong_InfoGain` by `>= +5 pp` on `continuation_completed` at equal/lower time, matches/beats `Fixed_Recheck_Schedule` with `>= 20%` time/travel reduction (and beats it on R3), outperforms all 3 ablations, and keeps conformal false-cert `<= 0.20`.
- **Step 5 (T-RO Synthesis)**: Write `TRO_EVIDENCE_SUMMARY.md` mapping supported claims vs. explicit evidence gaps.
- **Step 6 (Gate 6 — IJRR Extension, ONLY After Gate 4 & Step 5 PASS)**: `ijrr/embodied_sufficiency_experiment.py` validating Theorem 4(ii) (`V8.3/V8.4` matched-information 4-cell contrast $\Gamma_{\mathcal R}(T)$ across deadline sweep + irreversible cut bracket without dropping timeouts) and `IJRR_EMBODIED_SUFFICIENCY_REPORT.md`.

---

## 3. Master Verification Commands (Working Dir: `/media/camp/EXT_DRIVE/RealUS_playground`)

```bash
# Gate 2: Tests + 3-Axis Support Audit
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider \
  NMI_idea/experiments/anatomical_gps_unified_20261126/theory/verify_theory_math.py \
  NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_sim_and_firewall.py \
  NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_material_chart.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m sim.audit_training_support \
  --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step2_training_support.json \
  --proposals 64 --seed 58 --require-reader-inputs

# Gate 3: Reader Tests + LOAO Eval + Locked Zero-Shot Blender
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -s -p no:cacheprovider \
  NMI_idea/experiments/anatomical_gps_unified_20261126/tests/test_reader_and_tracking.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m reader.eval_reader \
  --split LOAO_OPEN3D --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step3_reader_loao.json

AGPS_EVAL_HOLDOUT_ALLOWED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m reader.eval_reader \
  --split FULL_TO_BLENDER --checkpoint NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/reader_frozen.pt \
  --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step3_reader_blender_zeroshot.json

# Gate 4: Active RL Training + LOAO Eval + Locked Zero-Shot Blender
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m rl.asymmetric_active_rl \
  --split LOAO_OPEN3D --seeds 11 22 33 --device cuda \
  --out-dir NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/rl_tro

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m rl.eval_closed_loop \
  --split LOAO_OPEN3D --checkpoint-dir NMI_idea/experiments/anatomical_gps_unified_20261126/checkpoints/rl_tro \
  --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step4_closed_loop_loao.json

# Gate 6: IJRR Embodied Sufficiency Experiment (ONLY after Gate 4 + Step 5 PASS)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=NMI_idea/experiments/anatomical_gps_unified_20261126:. \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m ijrr.embodied_sufficiency_experiment \
  --out NMI_idea/experiments/anatomical_gps_unified_20261126/results/step6_ijrr_embodied_sufficiency.json
```

## 4. Step 3 revision proposal (draft)

**本节是提案，不是 Gate 变更。** [完整批准稿](local://anatomical-gps-tro-continuation-plan.md)、上文 Step 0–6 及原验收标准全部保留。已签署的结论见 [HANDOFF.md](HANDOFF.md)：Gate 1/2 PASS，原 Gate 3 实测未过。以下工作尚未启动；未经批准，不用新终点替换旧终点，不打开 Blender，不启动 RL。

### 3A　站点：先确定要从回声中识别什么

旧终点是“采集时被要求扫描的唯一站点”。同一来源、同一起始高度可以同时满足多个标签；例如 Open3D 的 S2 和 S3 共享 −510、−505、−500、−495 mm。固定来源、高度后，公开采集与回声的分布不随所要求的标签改变。[窗口逐行证据与上界](results/step3_blind_audit/station_identifiability.json)表明，即使知道来源和真实起始高度，单标签**总体**宏 F1 上界也只有 Open3D 0.752750、CT→atlases 0.789430，低于原门槛 0.85、0.80。这不是某一有限测试集的得分上限，也不是血管身份或定位误差的上限。

**推荐新终点：本段拍到了哪些站点所需的血管组合（多标签、非临床）。** 对每个预定采集段、每个有来源标注支持的站点，保留两个不同的真值：

| 真值 | 预注册判定规则 | 用途 |
|---|---|---|
| 窗口兼容 `C` | 至少一帧位于该来源原 Gate 2 角色共存／交汇窗口；S1、S5 的主干角色相同，仍靠各自原高度带区分。允许多个站点同时成立。 | 结构机会分母，不等于已经拍到。 |
| 可见组合 `V` | 至少在**同一兼容帧**，`NYSORA_STATIONS[k]` 的每个必需角色各贡献一个独立、可分辨的管腔回声。遮挡、未渲染或只能看到未分辨的合并回声，不能作为该组合完整可见的证据。 | 模型预测各站点 `V` 的概率。`C=1,V=0` 是“没拍全”，不等于解剖结构不存在。 |

- `C,V` 只由模拟器真值生成，在监督/计分侧使用；读者只收匿名回声、探头状态和跟踪器输入，不给请求标签、来源、真实高度或支持掩码。可见组合不证明临床标准切面，也不证明已沿分叉完成追踪。若目标必须是临床站点，另需盲法标注影像和独立解剖数据，不能用本提案代替。
- 不改变分折、探头动作、视野或窗口，不删模糊样本。Open3D 评 S1–S5；BodyParts 评 S1/S2/S5；CT→atlases 评近端 S1–S4，逐来源保留支持情况。不支持的来源/类别预先记为 `NA`；支持但没拍到的段留在评估中，仍可产生假阳性。
- 先在训练/开发集检查：只改请求标签而固定来源、高度、随机带，公开输入和新标签必须相同；列出每类 `V/C`、阳性段数和完全空段数。与**只看探头位姿/估计高度、不看回声**的基线配对比较；同高度重叠窗口单列，检验是否真的从回声获得信息。
- 拟以全部预定段上的逐类阳性 F1 再取宏平均；报告逐类精确率、召回率、支持数及按段配对的 95% 区间。必评类别无阳性，记 `INSUFFICIENT_EVIDENCE`，不剔除该类。建议保留 0.85/0.80 作为**新任务的候选目标**，不保证可达，也不与旧单标签 F1 直接比较。需先批准终点、冻结判定规则和样本量，再看留出折；旧终点与失败结果始终并列。

### 3B　读者：保留原跟踪任务，先分解误差

[Open3D 原始报告](results/step3_blind_v6/LOAO_OPEN3D.json)：347 个已渲染的帧–角色中只有 322 个有位置预测；这 322 个的条件 RMSE 仍为 20.66 mm，动态身份 F1 为 0.4602。B2 配对检验在共同有预测的中心上显著，不等于达到 4 mm。这里有实测失败，**不能由站点标签重叠解释，也不能靠修改站点评分消除**。

1. **固定误差分母。** 在同批训练/开发带上逐段记录：切片有多少有限真值、多少角色进入回声、多少得到有限预测、多少身份正确。分别报捕获率、预测覆盖、角色混淆／精确贡献集合、全机会 RMSE 与条件 RMSE。缺一个目标不把它算成零误差，不把“身份已经猜对”的子集充当主结果。
2. **同段做 2×2 配对诊断，不重新渲染。** 交叉“读者关联／真贡献集合（含 NULL、合并）”与“带噪跟踪姿态／真实姿态”；其余输入、训练先验和参数不变。真值仅用于训练来源的开发诊断，不输入正式读者；不在留出折选择权重，诊断结果不算门槛成绩。记录真角色是否进入候选、正确路径在哪帧被剪掉、图像平面和世界坐标的残差。真实姿态对照不等于 B2：B2 仍使用带噪姿态。
3. **按证据派修正。** 真关联下仍漏中心，先查训练支持与切片查询，不用测试中心填空；只替换关联就改善，查关联似然与路径剪枝；只替换姿态就改善，查延迟、姿态协方差与运动更新。两者有交互则记录交互，不强归因；双真值下仍大误差，再查几何、坐标映射及观测模型。每项改动在新的开发段上与未改版本配对，不按留出分数挑方案。旧帧另记真路径存活、错→对和对→错的原始计数。
4. **原跟踪、身份和认证门槛不变。** 40 mm 身份 F1 ≥0.85、B0 <0.40；动态完整机会 RMSE ≤4 mm、身份 F1 ≥0.85、对 B2 配对 Wilcoxon `p<0.01`；旧帧纠正 ≥60%、回退 <5%，并优于 B3 的纠正率；保留帧耗时中位数 ≤5 ms 的工程目标。冻结读者后在独立开发校准段拟合整段分位数，再冻结校准器。已认证段的违规率须 ≤0.20 且优于未校准 B3；同时报告接受数、违规数和单侧 95% Clopper–Pearson 上界。全弃权记 `INSUFFICIENT_EVIDENCE`，不是零风险。

### 提案获批后的顺序与完整路线

先做 **3A 标签支持审计**和 **3B 开发集配对诊断**；不以一个方向的改善抵消另一个方向的失败。开发判定后冻结读者、校准器、种子清单和预注册评分规则，再按三个原非 Blender 划分、六个读者成对评测一次。新增种子只算新采样，不算新解剖。Sol/Luna 每阶段一次交回复现命令、结果、失败项和原始路径；Sol Max 独立审查，Astra Max 复核原始证据并记入 `HANDOFF.md`。拟议新产物单独版本化，不覆盖 `results/step3_blind_v6/`。

| 完整里程碑 | 保留内容与启动条件 |
|---|---|
| Step 0–2 | 已留档；保留理论见证、材料图、三轴数据审计及历史 Gate 1/2 PASS，不重跑。 |
| Step 3 | 旧 Gate 3 保持未通过。只有显式批准的 Gate 3 修订版，其非 Blender 条件全部满足且权重冻结，才可做一次隔离的 Blender 零样本诊断并作放行决定；不回调参数。 |
| Step 4 | **未启动。** 仅在 Gate 3 签署 PASS 后进入动态闭环、同预算的被动／强信息增益 MPC、固定复查与三种子非对称 PPO；分别报告拍到、认对、完成续扫。 |
| Step 5 | Gate 4 通过后，完成 T-RO 理论主张、实证支持和证据缺口对照。 |
| Step 6 | Gate 4 和 Step 5 通过后，执行 IJRR 匹配信息四格干预及不可逆切割检验。 |

### Step 3R 执行登记：优先定位续扫失败（开发集，结果揭盲前固定）

按“开始”的指示执行上面的 3A/3B。沿用 Neon 原方案：未来 RL 的主要结果是同预算、同动作权限下的 `continuation_completed`，而非捕获率、奖励或分类 F1；最终仍须对照强 MPC 与固定复查。原 Step 0–6 全部保留。当前只开放训练来源诊断与基于开发证据的修正；旧 Gate 3 FAIL 不变。

- 主开发协议为 `LOAO_OPEN3D` 的三个**训练**来源（BodyParts3D 右/镜像左、MedSeg），不采样 Open3D 测试来源或 Blender。起点是 `results/step3_blind_v6/checkpoints/LOAO_OPEN3D.pt`。同折诊断不能充当另外两折的无泄漏验证；未来各折独立拟合。
- **站点和续扫前提：** 按支持的来源/请求站点配对均衡采样，每段 4 帧；731901 为独立拟合集（每对 12 段），731902 为诊断集（每对 8 段），731903 保留作修正后的配对确认（每对 8 段）。同一兼容帧须含每个必需角色的独立、非混叠、非混合伪影回声；真实高度取最近原生 5 mm 行，容差半格，等距取较低行，不跨未支持行补标签。固定 0.5 多标签决策阈值，对照只看公开位姿的模型；无真阳性的必评类别明示不可估。
- **同段计数、不乘边际率：** 全部预定段 → 请求组合拍全 → 首个拍全帧的必需回声精确身份全部正确 → 该帧必需中心均有预测且世界误差均 <6 mm → 整段认证且无违规。另报未按上述筛选的认证数、拒绝原因和接受后违规数。这是固定采集下的单站诊断，不是 RL 成功率，不代表跨站不断支、时限或动作选择。
- **动态读者：** 931901 为训练来源诊断集、931902 为独立配对确认集，每来源 12 段、每段 8 帧。同一带分别使用估计/真实贡献集合及跟踪/真实姿态，不改模拟、不扩展原先验支持；分别报告缺失、条件与完整分母误差、候选/保留路径及身份混淆。真值替换仅用于定位问题。
- 先用诊断集确定一个可检验的主要损失，再于保留开发种子上将冻结修正与旧版配对比较；确认集不用于选参。未具备非 Blender 放行证据不运行正式留出评测、Blender 或 PPO。新证据放在 `results/step3r_dev/`，不覆盖六版结果。

**单项修正冻结（打开 731903 / 931902 前）：** 931901 上实际关联、跟踪姿态的 205 个非空帧里，193 帧的完整贡献集合可表示并通过合并门，180 帧进入局部搜索，161 帧进入几何细化，154 帧最终保留；整段真实前缀仅在 70 帧仍存活。731902 上 46/80 拍全、19/80 认全、18/80 同时达到 <6 mm、0/80 通过整段认证。只检验一个可改因素：**在不可逆剪枝前，按同一父先验下的积分因子证据排名**，而不是先按不同搜索起点的临时打分截断。保留相同的局部搜索、合并门、非线性细化预算上限、最终束宽、输出头和旧认证器；不加真姿态、来源或站点特权，不以变更动作、阈值或校准来提高本次数字。原六版保持默认。

确认时旧版与候选在同一条已渲染 tape 上配对：731903 看固定 80 段的联合漏斗（含身份、<6 mm 和整段认证），931902 看固定 36 段的可表示正确前缀、固定分母的完整/条件误差、整段身份与 <6 mm 交集。报告配对得失、无观测、遗漏及实测推断耗时，路径存活或宏 F1 单独改善不构成放行。候选只在至少一个下游联合计数增加、上述下游计数和几何缺失不恶化时列为继续验证对象；否则保留六版。旧校准器用于同规则比较，不能给变更后的读者签发现成的覆盖保证。不得据确认集选第二个变体；正式 Gate 3 与 Step 4 门槛不变。

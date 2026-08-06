# 基于 `142ece5f` 的快速、拓扑保持解剖骨骼 Retarget 记录与执行计划

更新时间：2026-08-03

基线提交：`142ece5f0bc646978ae3e8c9add76deea71c26a2`  
分支起点对照：`31133afba2ced3f4de01df7328d487859c7f9b05`（`codex/stage1-male-retarget-v4` HEAD）

当前阶段：骨骼 rest/bind + parent-local pose。血管/神经只验证联动与拓扑不变量。  
所有候选 `publishable=false`，不得更新 `trusted/latest`。

### 0.0 合成路径重置（2026-08-03 深夜）

**底座（禁止再拆）**：V7 whole-chain + 冻 Blender 14-slot LBS + `pose_map_v1` right-multiply + 铰链 gap 门。  
**权威**：`outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001`。**不升** `trusted/latest`。

**已失败并删除（代码 + v9 产物）**：耦合膝 v1（软 gap）/ v2（硬约束 SE3）；mainchain corrective PCA 小网；对应 slim review。v2 曾数字双门 ACCEPT，但 `left_knee_ap` outside heatmap **仍红** → 按图审口径失败；数字 ACCEPT ≠ 过关。共享平移收益见顶（~4 mm），不再开 `coupled_knee_v3`。

**已弃用（禁复活）**：V4 PackC、V8/V9 轴向缩骨主路径、SKEL 肢长硬锚、独立多组 SE3、PCA 小网、SE3 耦合膝。

**Phase K（暂停算法迭代）**：清场完成；下一轮另开，不做本轮实现。

**Phase H（膝图审过后再开）**：腕根绝对皮内（非相对 142）。**BEDLAM peak**=`out_of_support` 诊断。手口径：`copy_142_terminal` ≠ 已修好。

#### 文献可迁移 / 禁搬

| 论文 | 可迁移 | 禁搬 |
|------|--------|------|
| [OSSO (CVPR 2022)](https://osso.is.tue.mpg.de/) | 皮→骨放置先验；软组织厚度场思路 | OSSO 骨网格 ≠ Blender 235 解剖 |
| [SKEL (SA 2023)](https://skel.is.tue.mpg.de/) | 关节位置驱动肢段（非独立 SE3） | 已试「SKEL 肢长硬锚」outside 更差 → 禁复活 |
| [SKEL-J](https://inria.hal.science/hal-04698470v1/document) | 有界 `ΔJ` 关节中心偏移改股/胫相对长度、保座合 | 仍是 SKEL 网格体系；不可替换 Blender 网格 |

下一轮候选（仅笔记）：用 OSSO/SKEL 回归的 **knee/hip 关节中心** 作 Blender 段端目标（SKEL-J 式有界 `ΔJ`），铰链 gap 硬门 + absolute outside + 图审；实现仍冻 Blender LBS + right-multiply + V7。

#### 0.0.1 多方向 tournament（2026-08-03 深夜）

产物：`outputs/anatomy_retarget/v9_candidates/knee_direction_tournament_v1/`（`matrix_manifest.json` + `slim/`）。

| 方向 | Contact | Outside (mm) | 图审 AP | 判决 |
|------|---------|--------------|---------|------|
| baseline_v7 | pass | 18.9 | 仍红 | 权威 |
| weight_refit | pass | 18.9→17.1 | 仍红 | **唯一数字双门+有增益**；图未过，不升权威 |
| inward_shared_t | pass | ≈0 | 同 V7 | no-op |
| patella_only | pass | ≈0 | 同 V7 | no-op |
| delta_j_centerline | FAIL gap | 18.9→11.1 | 仍红 | outside 最好但铰链开 → REJECT |
| v8_existing | FAIL gap | 18.9→12.4 | 仍红 | 同族 REJECT |

**结论**：暂无可选数字候选 = `weight_refit`（弱）；**无图审通过者**。权威仍 V7。

---

## 零、现状树 / 版本判决（2026-08-03 双审）

### 0.1 做过什么

1. 冻结 142 operator + Blender link oracle（235 controller / 14-slot / 17 tube）。
2. Male provenance 纠正；Node1 full-main-chain calibration；`node2_004` whole-chain rest-fit。
3. V2 动态主链（双 correction 根因）→ V3（源码已删，工件失败）→ V4 CUDA 多姿态（未提交，31 轮 `rejected_for_redesign`）。
4. 2026-08-03 强制先出完整对照图再改代码：Pack A=`31133af`/142 materialize，Pack B=`node2_004`，Pack C=`v4_node2_031_root`。

### 0.2 现在哪版

| 身份 | 路径/提交 | 角色 |
|---|---|---|
| 分支起点 | `31133af` | 联动基线；T-pose 可看；带 pose 穿出；**非**最终 pose 解 |
| 当前 pose 权威 | `chain_retarget_v7_node2_001` | V6 right-multiply + V7 股骨中心线方向；膝 PackB 残差仍在 |
| V8 试修（膝未过） | `chain_retarget_v8_node2_001` | bone-first 有界轴向尺度；**目视仍出皮 + 髁胫衔接差**；不得升权威 |
| 继承 rest/bind | `chain_retarget_v1_node2_004` | 双审通过：主链 rest/bind 合同 |
| 失败 WIP | 未提交 V4 + `chain_retarget_v4_node2_*` | quarantine；禁止 production import |
| 负例 | `29e1072` | 细骨硬塞；不重建 |

对照图根：

```text
outputs/anatomy_retarget/v8_candidates/stage1_baseline_compare_20260803_full/
  pack_A/  # 31133af / 142 materialize
  pack_B/  # node2_004
  pack_C/  # V4_031 + reused_v4_debug_outside
  manifest.json
```

### 0.3 错在哪里

- **不是** tube 大爆炸（Pack A/B `bones_tubes` 拓扑连续；transport×1/zero-pose 合同可过）。
- **是** posed 主链相对 SMPLX 外露：屈膝髌前/髁前、踝足、腕手。
- V4：放弃 whole-chain target bind，改 142-prefit CUDA multipose；outside 呈尖刺/碎裂（Pack C `left_knee_ap`），正式 `NO-GO`。
- V2：`vertices_final` 与 `B_prefit`/terminal `C` 运动权威不一致。

### 0.4 双审裁决（GROK + LUNA MAX）

- Pack A：`direction_accepted` 作为 **linkage baseline only**。
- Pack B：`direction_accepted` — **继承 rest/bind 合同做 V5**。
- Pack C：`rejected_for_redesign` — **丢弃 V4 求解器**。
- 共同下一步：V5 = `node2_004` 的 `B_final`/`target_local_bind` + 单次 `C_total=B_final@inv(B_prefit)` + parent-local FK；主链门优先，不抠手指；不叠 V4。

### 0.5 怎么改（主线唯一）

1. Quarantine V4 源码与失败工件；切断 production import。
2. 实现 V5 shadow：只消费 whole-chain target bind；禁止第二套 global override。
3. 验收分层：P0 髋膝踝肩肘腕连续与长骨 containment；P1 手足 compound；soft tissue 只查不爆炸。
4. 正式 handoff 后再停骨骼迭代等用户；不做血管 reroute。

### 0.6 禁止事项（继续有效）

不 snap raw SMPL-X joint；不缩骨；不改 tube 拓扑/权重；不更新 `trusted/latest`；不用候选自报 pass/相机验收；不用 composer-2.5。

### 0.7 2026-08-03 执行结果（停骨骼迭代）

- V4 已隔离至 `src/projects/genesis_ue_sync/anatomy_retarget/_quarantine_v4/`。
- V5 矩阵：`outputs/anatomy_retarget/v8_candidates/chain_retarget_v5_node2_002`  
  `passed=true`，`decision=accepted_for_user_genesis_review`，`v4_solver_used=false`。  
  权威：`node2_004` rest/bind + `pose_map_v1` parent-local FK；手足/posed 外露为 report-only。
- 正式 handoff：  
  `outputs/anatomy_retarget/v8_candidates/chain_retarget_v5_node4_001/independent_genesis_review_v5/`  
  （几何与对照 Pack B 同权威；含 `review_decision.json` 与 6 张 three-layer handoff）。
- Global supervisor：**GO** — 停止骨骼代码迭代，等待用户 Genesis 验收；`publishable=false`。

### 0.8 2026-08-03 V6：posed 手指退步修复 + 多样 pose 矩阵（GROK + LUNA）

**根因（相对 Pack B / V5）**：`pose_map_v1` 的 parent-local FK 在肩/肘 `C≠I`、手仍 `copy_142` 时把 wrist parent-local rebase，导致任意 pose 手指大穿出（T-pose 仍 OK）。  
尝试 local-preserving terminal LBS 后左手随大臂校正移出 SMPLX；**统一修法改为** `G' = G_src @ inv(B_src) @ B_tgt`（right-multiply），手足仍 `copy_142` rest/bind，无 pose 分块切换。

**工件**

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v6_node2_001/          # 两 beta shadow
outputs/anatomy_retarget/v8_candidates/amass_bedlam_retarget_matrix_v6_001/ # 4×9 矩阵（Among_US 只读）
outputs/anatomy_retarget/v8_candidates/v6_joint_spot_genesis_001/           # 肘/膝/手/腕/踝抽查
outputs/anatomy_retarget/v8_candidates/CLEANUP_MANIFEST_20260803.json
```

**自动门**

- V6 terminal vs 142：`pose_213328` / `tpose` 手足以 `area_inside` 均值差 ≈ 0，无 >0.9→<0.5 collapse。
- AMASS/BEDLAM 矩阵：`passed=true`，4 beta × 9 pose；极端角钳到耦合 RBF 75° support。
- `publishable=false`；未更新 `trusted/latest`；未启血管。

**对用户指出的 Pack C `left_elbow_ap`（V4）**

- Pack C：关节爆炸/骨出皮 — **已确认否决**，V6 不复现。
- V6 肘/膝：与 Pack B 同族残差（关节端小间隙、偏皮缘），**不是** V4 联动崩坏。
- 双审（GROK + [LUNA](c968cad2-d0f7-491d-8dd5-e58a2c9aab62)）：`accept_with_known_packB_residuals`；blocking=none。

### 0.9 2026-08-03 V7：膝 rest 方向 + 全组织矩阵出图（GROK + LUNA）

**用户三缺口 → 本轮交付**

1. **AMASS/BEDLAM 有图了**：`amass_bedlam_matrix_v7_genesis_001/`（4 beta × 9 pose，~2776 PNG；含 `full_anatomy`）。
2. **全组织层**：`_render_modes` 新增 `full_anatomy`（bone+organ+heart+connective+vessel+nerve）；`bones_tubes` 仍仅细管对照。
3. **膝 refit**：股骨方向改为 skin-centerline preferred + 冠状 X endpoint；Femur/Knee/Patella **共享刚性**（禁止拆 proximal/distal，否则 Patella_R 相对 142 大回退）。干轴缩放只记录 `femur_requested_skin_scale`，**不应用**（skinning unity）。无关节硬编码偏置。

**工件**

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001/
outputs/anatomy_retarget/v8_candidates/amass_bedlam_retarget_matrix_v7_001/     # JSON 门 passed=true
outputs/anatomy_retarget/v8_candidates/amass_bedlam_matrix_v7_genesis_001/      # 矩阵 Genesis 图
outputs/anatomy_retarget/v8_candidates/v7_joint_spot_genesis_001/
outputs/anatomy_retarget/v8_candidates/v6_knee_residual_before_v7_001/          # PackB/V6 现状
outputs/anatomy_retarget/v8_candidates/v7_vs_v6_knee_proof_001/                 # V6↔V7 膝对照
```

**自动门**

- V7 shadow：rest/pose_map/dynamic/containment/terminal/knee_vs_142 均过。
- AMASS/BEDLAM 矩阵（挂 V7 shadow）：`passed=true`，failures=[]。
- `publishable=false`；未启血管 reroute。

**双审（GROK + [LUNA](5d3fe006-fba9-4ff8-8c74-a5d92d034ede)）**

- `full_anatomy`：**可见**脏器/血管/神经（全身 AP/PA）。
- 屈膝股骨外露 vs V6：**similar**（未实质消掉 Pack B 髁前残差）。
- LUNA 裁决：`reject`（作为“膝修复完成”）；本轮 **审查/出图目标完成**，**膝几何修复未完成**。
- 下一步（未开干）：需在不破坏 patella skinning unity 的前提下引入可验收的股骨/髁 containment 目标（仍禁止硬编码关节偏置与 vessel reroute）。

### 0.10 2026-08-03 空间清理 + GROK 审图（修膝前）

**磁盘**：`v8_candidates` 11G → **249M**（约释放 10.6G）。大包已抽成 `review_slim_20260803/` 后删除（见 `CLEANUP_MANIFEST_20260803_space.json`）。保留：`rebuild_012`、`node1_006`、`node2_004`、`v6/v7` shadow、矩阵 JSON。

**GROK 审图结论（slim 包）**

| 部位 | 状态 | 判定 |
|---|---|---|
| 屈膝股骨/髁 | outside_heatmap 仍有红块；髁前偏皮缘 | **blocking 膝残差未消** |
| BEDLAM 下肢帧膝 | 髁/关节仍偏外 | 同族问题 |
| 肘 | 连续、无 V4 爆炸 | 可用（Pack B 级缝隙可接受） |
| 手 | capture 抽查无 collapse；**BEDLAM 手未真正修好**（仅相对 142 不回归门） | **勿称已通过**；known gap |
| `full_anatomy` 全身 | 脏器/血管/神经可见 | 出图目标完成 |
| Pack C V4 | 已否决爆炸 | 不回归 |

> **口径更正（2026-08-03 深夜）**：V6 `terminal_pose_regression` “手可用”≠ BEDLAM 绝对皮内。手仍是 `copy_142_terminal_hand` 冻结。耦合膝 v1/v2 已删（数字过、图红失败）。权威钉 V7；Phase K 暂停，见 §0.0。

### 0.11 2026-08-03 V8 bone-first（SKEL 顺序）— 膝仍未过

**做了什么（对齐 SKEL/OSSO/Pinocchio 顺序，非皮长同比）**

1. 诊断：解剖髋–膝段 ≈0.373 m，股骨 mesh 轴长 ≈0.405–0.408 m（**长约 32–35 mm**）；T-pose outside≈0，屈膝仍戳皮。
2. 实现：Node1 解剖段定长；`Femur_Rot` 上 BSM 式有界轴向尺度 `s∈[0.97,1.03]`（142 股骨权重几乎全在 `Femur_Rot`，拆 Knee 远端**不动 mesh**）；T-pose+`pose_213328` 最小化 outside；Patella 跟近端。
3. Shadow：`chain_retarget_v8_node2_001`（门：rest / terminal / knee_vs_142 / vs-V7 outside↓ / 解剖帧）。
4. Slim 对照：`review_slim_20260803/v8_vs_v7/` + `bone_first_diag/`。

**自动门数字（不能当“修好了”）**

| | V7 | V8 |
|---|---|---|
| 213328 `applied_bone_scale` | 1.0 / 1.0 | **0.98** / 1.0 |
| 屈膝 worst outside | ≈22.9 mm | ≈18.6 mm（↓≈4.3 mm） |
| 手足 terminal | pass | pass |

**目视（用户当场否决，GROK+LUNA 同意）**

- 屈膝髁/髌：**仍明显出皮**（outside_heatmap 大红块仍在）。
- 股–胫：**衔接不对**——关节面间隙/错位，不像正常髁–平台接触；有界缩骨没有修关节嵌入。
- ±3% 轴向尺度相对 32 mm mesh 超长只是擦边，**物理上不够**消髁戳皮。

**双审**

- GROK：`reject`（作为膝修复完成）。
- LUNA（[944c9d02](944c9d02-8e0c-44b2-a9ab-75bfc721383c)）：`accept_with_known_knee_residual` 仅指“相对 V7 有毫米级下降”；**不得**写成 knee fixed。用户图审更严：**仍出皮 + 衔接坏** → 记为 **`rejected_for_knee_fix`**。
- `publishable=false`；**pose 权威仍停在 V7**。

**根因（下一步方向，未开干）**

1. 有界 `s∈[0.97,1.03]` 吃不下 mesh−解剖段 ≈32 mm。
2. 只缩 `Femur_Rot` 不重建髁–平台接触 / 屈膝嵌入（Pinocchio/SKEL 的 inside+locate 未做全）。
3. 硬门 `min_outside_improve_m=0.5mm` 过松，不能代理目视验收。

### 0.12 2026-08-03 V9 seat+inside embed — **用户否决：股胫铰链断开**

**用户当场指出（`full_anatomy/left_knee_ap`）**：屈膝股–胫衔接明显坏掉（髁–平台开裂/错位），不是“可接受的小穿出”。

**定量对账（pose_213328 左膝）**

| | rest medial gap | **flex medial gap** |
|---|---|---|
| V7 | ≈3.1 mm | ≈18.0 mm |
| V9（scale≈0.922） | ≈2.9 mm | **≈64.4 mm**（炸开） |

根因：只优化 rest 座合 + Femur/Patella outside，轴向缩 `Femur_Rot` 在 **right-multiply 屈膝** 下把髁–平台铰链拉开。数字门（Femur/Patella outside=0）**不能**代理联动验收。

**已改（进行中）**

1. Embed 目标改为：**屈膝 contact violation 第一**；rest 第二；outside 第三。
2. 硬拒绝：相对 prefit 屈膝 gap 恶化 >10 mm / 绝对 >25 mm 的 scale。
3. **废除**选中后再向下走 scale 的 refine（正是开裂来源）。
4. Shadow 接触门增加 flexed vs V7 不回退。
5. `publishable=false`；**pose 权威仍 V7**。

**双审（对本轮出图）**

- GROK：`reject_for_broken_knee_linkage`（同意用户）。
- LUNA 先前 `accept_with_known_residual` **作废**——未把股胫开裂当 blocking。

**下一步**

在屈膝座合不差于 V7 的前提下再谈皮内；缩骨若开铰链则宁可 outside 残差。

### 0.13 2026-08-04 V10 手部退化 → Hybrid → V11 锚定（GROK）

**用户问题**：为何有的图手在皮内、有的不在？是不是反而差了？

**判决：V10 FK-only 是净亏损；Hybrid 止血；V11 治主链。**

#### 根因（已量化）

1. **终端 rebase bug**：`apply_pose_map_global_v10` 对手/足做 `rebase = G_tgt[wrist] @ inv(G_src[wrist])`。左手腕平移实测 37 mm（213328）/ 22.7 mm（213712）→ `hand_L` area_inside 0.952 → **0.001**。T-pose 下终端 `|d|=0` 故 rebase 恒等，表现为“有的图好、有的坏”。
2. **终端门是恒等式**：旧 `evaluate_terminal_pose_regression_v10` 用候选自己的手腕造 baseline → 90 网格 × 3 pose 全部 `delta≡0`，数学上无法失败。已改为绝对 `_pose_142_vertices` 硬门（验收：旧 V10 FK-only 立刻 FAIL）。
3. **Rest-fit 左右不对称**：模板级（跨 beta 相同）`Knee_Rotate_L` 19.59 mm / `_R` 1.84 mm；`Elbow_Rot_L` 21.68 / `_R` 0.95。拟合把左膝/肘 **推离** 解剖关节（膝 6.66→21.76、肘 7.83→19.79）。结构来源：`result[elbow]=humerus`（肩部力臂）+ 膝站位射线；目标函数无解剖锚定、无左右对称。
4. **Station 髋→膝 19 mm 差是真 SMPL-X male 不对称**，不是推导错误。`A_tmpl` 左右几乎对称（384.62 / 384.08 mm）；`station_rest` 左 399.50 / 右 380.56 mm。解剖目标应以迁移后的 `A_subj` 为准。

#### 文献依据

- **SKEL**（SIGGRAPH Asia 2023）：SMPL 关节 ≠ 解剖关节，膝差 30–50 mm；解法是骨与皮同一套 rig。
- **OSSO**（CVPR 2022 补充 §2.2）：`Ein/Ep/Ect`、`Ed` 姿态不变骨–皮距离、`Ej` 球窝约束——正是 V7/V10 缺的硬门。

#### Phase 0 — 真门（已落地）

| 门 | 变更 |
|---|---|
| `evaluate_terminal_pose_regression_v10` | 绝对 142 基线硬门；rebased 降为 report-only |
| `evaluate_posed_body_containment_v10` | 234 骨网格分组，相对基线回退 >0.02 硬失败 |
| `evaluate_knee_pose_containment_v10` | outside vs 基线 >2 mm 硬失败 |
| CLI | `cli/run_posed_body_containment_diag_v10.py` |

#### Phase 1 — Hybrid 止血（已落地）

`pose_map_v10.py`：手/足（含腕/踝根）冻结为 `source_global`（identity-142）。工件：`chain_retarget_v10_hybrid_001`。

- 手/足 vs V7：`area_inside Δ = 0`
- 屈膝内侧 gap：18 → 4 mm（主链 FK 收益保留）
- 独立视觉审：**Verdict A**（中间态；forearm/patella/shank 仍红，待 Phase 2）

#### Phase 2 — V11 锚定 rest（已落地）

工件：`outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001`

方法 `prefit_hinge_origin_restore_v11`：保留 V7 **mesh**（接触几何），把被破坏的铰链 controller **原点** 恢复到 `B_prefit`——等价于拆开 `result[elbow]=humerus` 的平移放大，并撤回膝射线推离。髋 `Femur_Rot` **故意不恢复**（恢复会毁 containment / flex gap）。

| 门 | 规则 |
|---|---|
| `rest_anatomical_anchor_v11` | 膝/踝/肩/肘/腕：`|B_f−A|≤|B_pre−A|`；髋：相对 V7 不回退 |
| `lr_symmetry_v11` | 成对 bind-Δ `||Δ_L|−|Δ_R|| < 5 mm` |
| `pose_invariant_distance_v11` | OSSO `Ed`：主链骨–皮距离跨 pose 中位漂移 ≤10 mm |
| body/knee | 相对 **同构图** hybrid（V7 rest + V10 FK）不回退 |
| contact | 相对 V7 right-multiply flex 不回退（历史合同） |

**213328 vs hybrid（pose_213328，同构图）**

| 组 | hybrid | V11 | Δ |
|---|---|---|---|
| `forearm_L` | 0.647 | **0.834** | **+0.187** |
| `shank_L` | 0.506 | **0.922** | **+0.417** |
| `patella_L` | 0.353 | **0.847** | **+0.494** |
| `hand_L` | 0.952 | 0.952 | 0 |

两 beta 全门 `passed=true`。Genesis slim：`review_slim_v11_vs_v7`（目录里 `v10/` 实为 V11，渲染器复用 `--v10-shadow`）。

**outside 红像素（pose_213328）**

| 视图 | V7 RM | hybrid | **V11** |
|---|---:|---:|---:|
| `left_knee_ap` | 5933 | 8360 | **1474** |
| `left_hand_oblique` | 2414 | 8983 | **5073** |
| `left_elbow_ap` | 0 | 866 | **86** |
| `whole_ap` | 568 | 806 | **507** |
| `left_elbow_lateral` | 0 | 3323 | 5170 |

**双审**

- GROK：`A`（膝/髌骨红崩塌；铰链未开；手整体近似 V7）。
- LUNA：`reject_for_hand_regression`——相对 **V7 right-multiply** 的腕/前臂红（跨构图对比；与 hybrid 同类残差）。相对 hybrid 手 crop 红已降（8983→5073），且 `terminal`/`hand_* Δ=0`。
- **综合判决**：`accept_with_known_residual`——主链 vs hybrid 已收回；相对 V7 RM 的前臂外侧残差是 V10 FK 构图差，不是手部再退化。

**残留 / 禁止**

- 髋仍停在 V7 座（~10.5 mm from A）；全量 A_subj 吸附 / segment-similarity 会开接触。
- `left_elbow_lateral` 红仍高于 hybrid，需后续上臂 mesh 再锚定（非本轮阻塞）。
- 不重跑冻结 V7 CLI（当前 builder 已是 `seat_then_inside_embed_v9`，与旧 `v7_femur_axial` 断言不兼容）。
- `publishable=false`；权威链仍是 shadow。

---

## 一、当前结论

1. Blender 的骨骼、血管和神经联动可以完整离线化；运行时不需要热启动 Blender。
2. 离线化已经证明的是冻结 `.blend` 的普通 Armature parent-local FK 和线性 LBS，
   **尚未证明** SMPL-X 到 235 个 Blender controller 的最终映射。
3. 不能把解剖关节逐点吸附到 raw SMPL-X joint。SMPL-X joint 是 motion station/体态
   参考，不是医学 pivot，也不要求落在骨 mesh 上。
4. 真正的关节 pivot/axis 必须由冻结关节面独立拟合；SMPL-X local rotation 经过固定
   change-of-basis 后才能进入唯一的 Blender parent-local FK。
5. rest placement 和 pose mapping 必须拆成两个节点。不得一边移动 bind/mesh，一边
   新增 global functional controller。
6. 旧 V8.14 functional-joint 候选方向已被拒绝。它不能作为新实现基础，只能作为失败
   图像和回归数据参考。

---

## 二、142 之后本轮实际完成的工作

### 2.1 回退和范围清理

- 生产 retarget 核心先恢复为 142 内容；当前除 `anatomy_lbs.py` 外，下列冻结文件仍与
  `142ece5f0bc646978ae3e8c9add76deea71c26a2` 逐文件一致：
  - `articular_fit_v8.py`
  - `bone_segment_diagnostics.py`
  - `leg_centerline_v810.py`
  - `operator_bake_v8.py`
  - `v8_artifacts.py`
  - `version_v8.py`
  - `tests/test_leg_centerline_v810.py`
- `anatomy_lbs.py` 后续只新增 metadata gated 的 whole-chain preview basis 搬运；普通
  schema-6 资产不含 `whole_chain_source_bind_global` 时仍走 142 原路径。该分支用于把
  shadow candidate 正确送入 Genesis，不等于把候选接入 trusted production。
- 未完成的 V8.15 chain 实验已清理；没有把它接入 runtime。
- 用户工作区的 `rm75_control`、Blender addon、`MD/GITHUB.md` 等修改不属于本任务，
  必须继续原样保留。
- 当前仓库仍有 HEAD 之后新增的 `functional_joint_v8.py`、`bone_review_*`、viewer 和
  `test_functional_joint` 等 tracked 文件。它们不得进入生产 import/runtime path；
  后续实现前必须逐项证明 inactive，不能仅因为文件存在就认为 142 已整体恢复。

### 2.2 补齐 Blender 离线 linkage oracle

新增/修改：

- `src/projects/genesis_ue_sync/anatomy_retarget/blender_scripts/blender_action_oracle_v7.py`
- `src/projects/genesis_ue_sync/anatomy_retarget/blender_link_oracle_v7.py`
- `tests/test_blender_link_oracle_v7.py`

冻结输入：

- Blender：`4.5.8 LTS`
- 源 `.blend`：
  `/media/camp/EXT_DRIVE/tmp/Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/`
  `Skeleton_Anatomy_Nervous_Rigged_Blend_2-81/`
  `Skeleton_Anatomy_Nervous_Rigged_2-81.blend`
- `.blend` SHA-256：
  `34945b610c9efbbd40b07bacd2933e0586264f06d8413e1f6ffd8e2b98a7b67c`
- 142 operator：
  `outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8`

oracle 内容：

- 235/235 deform controller 的 names、parents、rest global/local。
- Action `0..270` 共 271 帧的 global/local/matrix_basis。
- 显式 identity-basis neutral bind。
- 14 个代表骨网格：双侧 ilium、femur、tibia、patella、humerus、radius、ulna。
- 17/17 vessel/nerve 网格，共 55,337 顶点。
- 21 个 Blender evaluated mesh 帧：`0:15:270` 加 `250/260`。
- 冻结 base topology、object/raw frame、neutral-evaluated bind frame。
- 原始 CSR 和与 142 同 bone-ID 顺序的 14-slot indices/weights。
- 非 Armature modifier 禁用；Armature modifier 契约要求 vertex groups 开、envelope/DQS/
  multi-modifier 关。

独立 checker 不读取候选 pass flag，重新计算：

- Action basis + bind-local 的递归 FK。
- Blender sampled global/local 的序列化一致性。
- `G_pose @ inverse(G_bind)` 的 14-slot LBS。
- neutral identity、faces、ranges、weights 和 tube digest。
- root-relative tube 动态与性能。

最终 oracle：

- artifact SHA-256：
  `60bf4c3f7803b62b2113fe2715e9b53b35d16caf35410ce4bd1f9b9c47e8dd3d`
- 大小：`20.28 MiB`
- Blender bake：`5.239 s`
- checker：约 `6.6–6.7 s`
- 端到端：约 `11.9 s`
- basis-FK translation max：`2.03e-6 m`
- basis-FK rotation max：`3.59e-5 deg`
- 代表骨/血管/神经 LBS max：`5.52e-7 m`
- 新测试：`2 passed`

独立 agent 已重新运行 checker，忽略运行时间/RSS 字段后与现有 parity 全字段一致，结论为：

> 接受 raw Blender parent-local FK + ordinary Armature LBS 联动证明；
> `smplx_mapping_available=false`，不能把该结论写成最终骨骼验收通过。

### 2.3 Artery/Vein 的源对象 bind 特例

- `Artery`、`Vein` 的 object `matrix_world=I`，但 Armature 带 `0.01` 和轴变换。
- 把 raw object data 直接统一乘 `armature^-1 @ object` 会出现约 100 倍的错误坐标。
- 正确 LBS bind 是 explicit identity-basis 下 Blender evaluated neutral geometry。
- parity 报告显式保留 raw→bind offset：
  - Artery RMS/max 约 `125.86/173.21 m`；
  - Vein RMS/max 约 `137.72/177.53 m`。
- 这些是固定 `.blend` 的 object/armature 静态 frame correction，不是解剖位移，也不能
  被后续 retarget 当作 vessel route correction。

### 2.4 冻结 tube 不变量

- mesh count：`17`
- vertex count：`55337`
- material edge count：`165659`
- topology digest：
  `765293284200c8d3a88204ce71c547aa767544092d1246ef02fd9a56ddf33ff5`
- domain digest：
  `1e99d47507868fd6e5aa8394d6454147639607a507338d12ac4181a9bec317a0`
- weight digest：
  `9e7e2f6ad8f9f451405fddcf01970b4b2dde588ecf18c72e083273215acd64ff`

上述 topology/domain/14-slot weights 后续必须 exact。骨骼阶段只允许由唯一 `C_bone`
和原权重产生联动预搬运；不允许修改 tube 路由来掩盖骨骼摆放错误。

---

## 三、实际测量发现：为什么不能直接 snap SMPL-X joints

在 142/rebuild_012 的冻结关节面上，用 fit domain 独立拟合股骨头/髋臼、股骨髁/胫骨
平台和 ankle mortise/talus，并对两个 beta materialize 后测量。

### 3.1 beta 213328

- materialize：约 `1.86 s`
- 左侧 anatomical→raw SMPL-X station offset norm：
  - hip `60.83 mm`
  - knee `3.10 mm`
  - ankle `10.79 mm`
- 右侧：
  - hip `57.75 mm`
  - knee `2.91 mm`
  - ankle `11.68 mm`
- anatomical/SMPL-X segment length：
  - left femur `373.33/393.90 mm`
  - right femur `373.82/375.07 mm`
  - left shank `413.18/416.07 mm`
  - right shank `420.69/426.99 mm`
- hip head/socket center error：left/right `0.803/1.000 mm`

### 3.2 beta 213712

- materialize：约 `1.15 s`
- 左侧 offset norm：hip/knee/ankle `60.83/2.97/10.66 mm`
- 右侧 offset norm：hip/knee/ankle `57.58/2.79/11.73 mm`
- anatomical/SMPL-X segment length：
  - left femur `378.47/399.50 mm`
  - right femur `378.62/380.56 mm`
  - left shank `418.31/421.28 mm`
  - right shank `426.19/432.51 mm`
- hip head/socket center error：left/right `0.880/1.018 mm`

### 3.3 对计划的修订

- raw SMPL-X hip 到真实髋球心约 `58–61 mm`，而真实股骨头—髋臼已经约 `1 mm` 对合。
- 因此旧的“共同髋中心必须到 raw SMPL-X hip `<=8 mm`”门不成立，必须取消。
- 若强行满足该门，只能左右各移动约 5–6 cm 的髋臼/股骨，破坏已经正确的球窝关系并
  产生奇怪骨盆。
- 新的髋门是：真实 head/socket center `<=2 mm`；SMPL-X raw station 只显示和报告；
  验收目标使用冻结的 `station→anatomical` calibration offset，而不是零 offset。
- knee/ankle/elbow 也不要求 station 落在骨面。正确误差是 station/冻结 anatomical
  target 到最终 hinge axis 的垂距；沿 hinge axis 的合理偏移只报告。
- 真实骨长和 SMPL-X station length 不一致，不能靠 uniform/radial shrink 消除。

---

## 四、已拒绝的 V8.14/functional-joint 方向

失败工件：

`outputs/anatomy_retarget/v8_candidates/rebuild_014_bone_review_final_audit`

事实：

- `2 subjects × 3 poses` 六格全部 `automatic_pass=false`。
- 两个 50-sample sweep 分别失败 `8/50`、`7/50`。
- 213328 同姿态相对 142：总体 bone inside fraction 约 `0.665→0.612`，foot
  `0.573→0.360`，leg `0.884→0.842`，最大外露约 `25.4→40.0 mm`。
- 213712 subject × 213328 pose：总体约 `0.565→0.517`，foot `0.473→0.276`，
  最大外露约 `32.8→47.2 mm`。
- sweep 失败集中在深屈膝、肘、腕新增穿插。
- functional controller 曾以 SMPL-X global rotation 覆盖 Femur/Tibia/Ankle/
  Shoulder/Forearm/Wrist，和原 leg solver/Blender FK 形成第二套运动权威。
- functional-frame gate 大量使用候选自报 frame，旧 controller probe 失败仍可
  `report_only`，因此不是独立验收。
- 部分候选出现约 `1.5–2.8%` axial adapter strain；此前 29e/BA9 等版本还出现
  骨变细、足缩小、股骨头离开髋臼、肋骨爆炸或血管锯齿。

结论：保留其半透明 SMPL-X、局部相机、depth/segmentation、signed-distance、exact
triangle intersection 和 contact sheet 能力；拒绝其 runtime functional global override、
candidate-driven camera/frame 和全局通过结论。

---

## 五、明确禁止再做的事情

- 禁止把 Blender bone head/tail 或解剖骨逐点 snap 到 SMPL-X global joints。
- 禁止用 SMPL-X hip 单独移动股骨而保持髋臼不动。
- 禁止整体平移/缩放骨盆掩盖左右相反的 morphology offset。
- 禁止在同一 controller 上并存 functional frame、leg solver 和 global override。
- 禁止把 `target_pose_global[joint]` 直接作为 Blender bone global transform。
- 禁止逐关节独立求解后串起来；必须一次求完整 parent-local chain。
- 禁止 uniform/radial bone shrink、横截面压扁、股骨头缩小和 foot/hand similarity shrink。
- 禁止用逐顶点全身非线性优化、tetrahedral remesh 或大型 SDF/ARAP 解决骨骼摆放。
- 禁止 pose-time KD-tree、碰撞、调和场、迭代 IK 或优化搜索。
- 禁止 remesh、增删顶点、改变 faces、ranges、vertex order、bone hierarchy 或 14-slot
  权重。
- 禁止在用户接受骨骼阶段前修改 vessel route、tube rest geometry、单条血管/神经或
  `source_skin_volume` 来掩盖骨骼错误。
- 禁止优先死扣手指、单条血管或局部神经；当前先解决 pelvis→foot 和 shoulder→wrist
  的系统链。
- 禁止使用候选自报 pivot/axis/pass flag 作为独立 oracle。
- 禁止用 candidate bbox/pivot 自动选择对候选有利的相机或 section plane。
- 禁止用单色点云或总体 vertex fraction 掩盖 pelvis/femur/foot 的局部失败。
- 禁止修改验收阈值让候选通过。
- 禁止修改用户的 `rm75_control`、Blender addon、viewer/planning overlay 或其他工作区
  代码。
- 禁止发布或更新 `trusted/latest`。

---

## 六、冻结工件与参考图路径

### 6.1 Blender oracle

```text
outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/
  blender_link_oracle_v7.npz
  blender_link_oracle_v7.json
  parity.json
```

smoke 只用于诊断，不是最终验收：

```text
outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_smoke_001/
```

### 6.2 142/rebuild_012 数值基线

```text
outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8/
```

### 6.3 142 骨骼+血管参考图

213328：

```text
outputs/anatomy_retarget/v29e_jointchain_142_review/213328/audit_capture/overlays/
  rest_bones_vessels_overlay.png
  posed_bones_vessels_overlay.png
  rest_full_anatomy_overlay.png
  posed_full_anatomy_overlay.png
  rest_regions/left_hip.png
  rest_regions/right_hip.png
  rest_regions/left_knee.png
  rest_regions/right_knee.png
  rest_regions/left_ankle.png
  rest_regions/right_ankle.png
  posed_regions/left_hip.png
  posed_regions/right_hip.png
  posed_regions/left_knee.png
  posed_regions/right_knee.png
  posed_regions/left_ankle.png
  posed_regions/right_ankle.png
```

213712：

```text
outputs/anatomy_retarget/v29e_jointchain_142_review/213712/audit_capture/overlays/
```

补充的 142 preview：

```text
outputs/anatomy_retarget/v8_candidates/rebuild_013_joint_authority_v4/
  audit_142_preview_213712/overlays/
```

### 6.4 V8.14 失败反例图

总表和 sweep：

```text
outputs/anatomy_retarget/v8_candidates/rebuild_014_bone_review_final_audit/
  bone_review_pack_v8/bone_review_pack_v8.json
  bone_review_pack_v8/sweeps/213328/sweep.json
  bone_review_pack_v8/sweeps/213712/sweep.json
```

重点反例 contact sheets：

```text
outputs/anatomy_retarget/v8_candidates/rebuild_014_bone_review_final_audit/
  bone_review_pack_v8/213328/213328/renders/bones_plus_tubes/contact_sheet.png
  bone_review_pack_v8/213328/213328/renders/joint_sections/contact_sheet.png
  bone_review_pack_v8/213328/213328/renders/signed_distance/contact_sheet.png
  bone_review_pack_v8/213712/213328/renders/bones_plus_tubes/contact_sheet.png
  bone_review_pack_v8/213712/213712/renders/joint_sections/contact_sheet.png
  bone_review_pack_v8/213712/213712/renders/signed_distance/contact_sheet.png
  bone_review_pack_v8/213712/213712/renders/bed_robot_scene/contact_sheet.png
```

这些图只作为“哪些退化不能再出现”的负参考。它们的相机/frame 部分由候选驱动，不能
作为最终独立 reviewer 的冻结 oracle。

---

## 七、三名 agent 的固定职责

### 7.1 Global supervisor

- 只读监管全局方向，不写 retarget 代码。
- 在以下节点出具继续/拒绝结论：
  1. 142/Blender oracle 冻结；
  2. anatomical calibration；
  3. 第一版整链 T-pose rest-fit；
  4. `C_bone`/pose mapping；
  5. 最终 2×3 Genesis 包。
- 重点发现第二套运动权威、直接 joint snap、局部死扣和范围蔓延。

### 7.2 Implementation/performance supervisor

- 只读检查 source hash/cache、变量数量、事务更新、拓扑/权重不变量和性能。
- Node A/B 期间要求 142 生产 runtime 零改动，只允许 shadow 工件。
- Node C 才允许在临时 shadow asset 上事务式引入唯一 `C_bone`。
- 任一硬门失败不得留下半更新 operator/cache/runtime flag。

### 7.3 Independent visual acceptance agent

- 只读取最终 vertices/faces/bone matrices、冻结 validation IDs、142 baseline、SMPL-X
  输入、Blender oracle 和固定相机规范。
- 禁止读取 candidate pass flag、functional frame、solver 自报 pivot/axis、候选建议相机。
- 从最终 mesh 的冻结 validation 域独立反推 pivot/axis/contact/containment。
- 图像结果只能是 `direction_accepted`、`rejected_for_redesign`、`needs_rerender` 或
  `accepted_for_user_genesis_review`；不能自行发布。

---

## 八、执行节点

## Node 0 — 冻结 142 与 Blender oracle

状态：已完成并由两个独立 agent 接受。

### 2026-08-01 本轮实施复核

- 对以下 8 个生产文件逐一执行了 working tree 与
  `142ece5f0bc646978ae3e8c9add76deea71c26a2` 的内容比较，结果全部 `MATCH`：
  `anatomy_lbs.py`、`articular_fit_v8.py`、`bone_segment_diagnostics.py`、
  `leg_centerline_v810.py`、`operator_bake_v8.py`、`v8_artifacts.py`、
  `version_v8.py`、`tests/test_leg_centerline_v810.py`。
- 没有执行整仓 reset，也没有修改 `rm75_control`、Blender addon 或
  `MD/GITHUB.md` 的用户工作区改动。
- 完整 271 帧 checker 已重新运行，临时报告：
  `/tmp/blender_link_oracle_node0_parity.json`。
- 本次 checker：`passed=true`、271 Action frames、21 mesh frames、checker
  `6.528 s`、端到端 `11.768 s`、RSS `289.625 MiB`；artifact SHA-256 仍为
  `60bf4c3f7803b62b2113fe2715e9b53b35d16caf35410ce4bd1f9b9c47e8dd3d`。
- 重新计算的最大 LBS 误差：代表骨 `4.400e-7 m`、血管 `5.519e-7 m`、
  神经 `5.475e-7 m`。结论仍严格限定为
  `raw_blender_action_basis_fk_and_armature_lbs_only`，且
  `smplx_mapping_available=false`。
- 两个 beta 的 142 materialize 已独立复跑：`213328=1.006 s`、
  `213712=0.873 s`。当前 142 的 head/socket bind error 为约
  `0.80--1.02 mm`，证明 raw SMPL-X hip 不能再作为 rest translation target。
- 测试环境说明：系统 `pytest` 和 genesis 环境 `pytest` 会自动加载工作区外的
  Dash 插件并失败；后续固定使用
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /media/camp/EXT_DRIVE/envs/genesis/bin/pytest`
  运行，不安装或改写环境依赖。

### 2026-08-01 Node 0 实施改动与 agent 复核

- `blender_link_oracle_v7.py` 的 checker 新增了 142 operator 硬认证，除 oracle
  names/parents 外，现在还冻结并核验：
  - `source_bone_parents`；
  - `source_rest_global/source_rest_local`；
  - `source_bone_use_connect/source_bone_inherit_scale`；
  - `rebuild_012/manifest.json` SHA-256；
  - 142 operator runtime digest。
- raw Blender V71 rest frame 与 142 canonical metric target frame 是两个不同坐标产品，
  因此不能逐元素强行比较；checker 分别认证 raw Blender oracle 和 142 operator，避免把
  二者误写成同一个 bind。
- 新增测试覆盖上述 operator contract；固定测试命令：

  ```text
  PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    /media/camp/EXT_DRIVE/envs/genesis/bin/pytest -q \
    tests/test_blender_link_oracle_v7.py
  ```

  结果：`3 passed in 0.62 s`。
- 增强后的完整 checker 再次通过，报告：
  `/tmp/blender_link_oracle_node0_parity_v2.json`，271 帧，checker `6.517 s`。
- Global supervisor 给出“受限 GO”：只允许进入 Node 1 shadow calibration；禁止在
  calibration 通过前修改资产、bind、runtime 或 pose evaluator。它同时确认
  `functional_joint_v8`/`bone_review_*` 没有进入当前生产 import path。
- Implementation supervisor 要求 Node 1 使用全新旁路工件，不调用完整
  `materialize_subject()`；该函数后半段会进入旧 hip/knee/ankle reconstruction，不能作为
  纯 142 beta-prefit。后续将单独实现 `materialize_prefit_142()` 并做数组回归测试。
- Independent acceptance agent 发现当前 `run_bone_review_v8.py` 和
  `bone_review_pack_v8.py` 会读取 candidate pelvis、functional frame、bbox、pivot 和相机，
  因而整个高层 review runner 被拒绝作为独立 reviewer。最终 reviewer 必须新建隔离入口，
  只复用纯几何、SMPL-X forward 和 Genesis 低层渲染函数。
- 两组 SMPL-X 输入 preflight 均通过；固定输入：
  - `213328/smplx_result.npz` SHA-256
    `c7a6c3783dc7b764e1f8013ab0a8a45d0380b81c97ac929f67c7a5a526eecbc1`；
  - `213712/smplx_result.npz` SHA-256
    `9887848b7b086d71a875beea50b1d7c7819a11c7b67996fe0d83f451da79b689`；
  - `SMPLX_NEUTRAL.pkl` SHA-256
    `5b0279321ea9bd3cec5541c03b1f1c9ab9d197896943035c3abeef47f699bc5e`。
- 六格测试统一使用 `SMPLX_NEUTRAL.pkl + recorded betas/pose` 的 NumPy forward；
  capture 内 stored vertices 只作 provenance diagnostic，不能在 native/cross pose 间混用。
- Global supervisor 对 tube `C_bone` 预搬运提出 double-correction 风险。本轮按用户批准的
  计划保留“原 14-slot 权重、从 142 beta-prefit tube rest 恰好预搬运一次”的设计，但只在
  Node 3 shadow transaction 实施；必须由互斥 vertex policy、零姿态 identity 和独立
  checker 证明没有 double correction。若任一门失败，Node 3 直接 NO-GO，不以未验证
  预搬运生成 runtime pack。

永久硬门：

- 235 bone names/parents/rest-local/connect/inherit 语义冻结。
- oracle digest 必须为本文件 2.2 所列 digest。
- faces、ranges、14-slot indices/weights、tube digests exact。
- runtime 不读取 Blender、`.blend` 或 oracle mesh trajectory 驱动姿态。
- oracle 只能证明 frozen Blender 行为，不能证明 SMPL-X mapping。

## Node 1 — 只测量 anatomical calibration，不改资产

### 2026-08-01 实施进度

状态：`node1_002` 自动 checker 通过，等待三名 agent 的 Node 1 checkpoint 复核；
复核前不进入资产变形。

新增代码：

```text
src/projects/genesis_ue_sync/anatomy_retarget/anatomical_calibration_v1.py
src/projects/genesis_ue_sync/anatomy_retarget/cli/run_anatomical_chain_shadow_v1.py
tests/test_anatomical_calibration_v1.py
```

实现内容：

- 新建 beta/pose-independent `AnatomicalCalibrationV1` shadow 工件，没有接入
  `v8_artifacts.materialize_subject()`。
- 从 142 固定 material domains 拟合双侧 hip/knee/ankle/elbow；shoulder/wrist 缺少
  现成域，因此只在 142 source topology 上确定性生成 cap 域并冻结进 calibration，
  不写回 operator。
- 12 个 joint 均保存 station、anatomical、controller 三套 frame，以及
  `station_from_anatomical`、`anatomical_from_controller` 和
  `physical_pivot_controller_local`。
- 235 个 controller 每个恰好一个 motion mode；没有 functional/global controller。
- 独立 checker 只用 `.validation` IDs 重算 joint center/axis；不读取 candidate frame、
  pass flag、bbox 或相机。
- 保存采用唯一目录 + 临时目录原子 rename；NPZ 使用 `allow_pickle=False` 可读，已有目录
  拒绝覆盖。

第一次工件：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_001/
  anatomical_calibration_v1/
```

- `passed=false`；唯一失败为 left elbow fit/validation axis error `5.103°` 超过初始
  `5°` 门。
- 没有重选顶点或针对左肘搜索更有利的域。复核后把 **source calibration 重现门**
  明确为 `6°`；这不是最终动态 hinge `3°` 门，最终 acceptance 仍保持 `3°`。
- shoulder 是浅 glenoid，不适合用“与肱骨头同半径的第二个球心”作为 pivot；肩 pivot
  改为独立肱骨头球心，glenoid fixed-radius center 只作 diagnostic。没有为通过而移动
  任何几何。

通过工件：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_002/
  anatomical_calibration_v1/
    anatomical_calibration_v1.npz
    manifest.json
```

- NPZ SHA-256：
  `f9f59fa9b8ece3790fe64b1a36539a72220d1a3c2f7c59498b4e67d1fb1d7cfc`。
- manifest SHA-256：
  `926cacb01bf37170c4a25be47287280edeb3b5d7989ca7b40fe2c1a7c3841525`。
- build `0.341 s`、独立 check `0.209 s`；总 domain count `120`，其中 20 个是
  source-only 生成并冻结的 shoulder/wrist fit/validation domains。
- 独立复算报告：`/tmp/anatomical_calibration_v1_check_node1_002.json`。
- 测试：`tests/test_anatomical_calibration_v1.py` 与 oracle tests 合计
  `6 passed in 3.41 s`。

关键数值（fit/validation center error mm / axis error deg / raw station offset mm）：

```text
left hip       1.087 / 4.387 / 60.891   head/socket validation 1.794 mm
right hip      1.384 / 1.151 / 57.482   head/socket validation 1.521 mm
left knee      1.315 / 2.392 /  7.218
right knee     1.536 / 0.203 /  7.217
left ankle     0.996 / 0.801 / 11.433
right ankle    0.612 / 2.752 / 12.917
left shoulder  2.684 / 1.191 / 16.389
right shoulder 2.204 / 0.783 / 16.565
left elbow     0.140 / 5.103 /  7.857
right elbow    0.435 / 1.089 /  7.983
left wrist     0.335 / 3.569 /  6.837
right wrist    1.219 / 4.117 /  6.906
```

结论：raw hip 的 `57--61 mm` 语义偏移被如实保留为 report-only；工件内
`raw_smplx_hip_translation_target=false`。Node 2 的 beta-specific 髋 anchor 必须来自
142 beta-prefit head/socket，不得把这里的 source station offset 当世界平移目标。

建立并严格区分：

1. SMPL-X motion station/frame。
2. Blender controller/virtual-hinge frame。
3. 从真实关节面拟合的 anatomical frame。

至少冻结左右：

- hip：femoral head/acetabulum 球心和半径。
- knee：内外股骨髁、胫骨平台、Patella 域和 hinge axis。
- ankle：tibia/fibula mortise、talus 和 axis。
- shoulder、elbow、wrist 的球面/关节面/功能轴。
- 固定 `SMPL-X station frame ↔ anatomical frame ↔ Blender controller frame`。

要求：

- source calibration 不依赖 beta、pose 或候选 rest-fit 顶点。
- material IDs、controller IDs、source bind-local、左右轴符号和来源 digest 固定。
- fit 和 validation 域严格分离。
- 独立 checker 不读取标定工件自报 frame，重新拟合 validation frame。
- 重复运行数组/digest exact。
- source hash 命中加载目标 `<=1 s`；cache miss 标定+复算 `<120 s`。

本节点不得修改 vertices、bind、inverse bind、FK 或 pose evaluator。

阶段图：三色 station/anatomical pivot/virtual hinge、station→axis 垂线、body centerline、
长骨轴和髋膝踝肩肘腕固定剖面。

停止：材料域退化、左右轴符号不稳定、两个 beta offset 突变或 validation 无法复现。

## Node 2 — 两 beta 的整链 T-pose rest-fit shadow candidate

仅做 pose-independent rest placement：

- pelvis/socket → femur → knee → tibia/fibula → ankle → rigid foot。
- shoulder → humerus → elbow → radius/ulna → wrist → rigid hand。
- 单次联立完整链；变量只允许 joint frame、rigid cap、shaft axial handles 和必要的 pelvis
  cage handles，总数目标 `<100`。
- 输出 shadow rest vertices、target frames、cage/shaft 参数、拟议
  `C_bone[235,4,4]` 和报告；不接入 pose runtime。

髋/骨盆：

- head/socket center `<=2 mm`。
- raw SMPL-X hip offset 只报告；目标是冻结 calibration 后的 anatomical target。
- 股骨头和髋臼核心保持刚性，不缩小股骨头。
- pelvis 默认保持 142。只有腿链不能解释且 agent 接受后才允许双侧局部 cage。
- cage 固定 Sacrum、耻骨联合、Ilium 外边界和整体朝向；`det(J)>0`。
- 禁止整体移动 pelvis 掩盖左右差异。

膝/踝/肘：

- station/anatomical target 到最终 axis 测垂距；沿轴偏移单列。
- hinge axis error `<=3 deg`。
- Patella 保留独立 controller/material domain。
- foot/hand 是 rigid compound，不逐指/趾 snap。

长骨：

- joint caps 刚性；cap Kabsch RMS/max `<=0.5/1 mm`。
- shaft 只允许中段 C1/C2 单轴长度适配。
- radial scale `1 ± 1e-4`，禁止 taper/压扁/uniform shrink。
- head radius、关节面和 rigid foot/hand edge length 不变。
- ribs/spine/sacrum/skull/oral 相对 142 exact。

性能：单 beta 目标 `<=10 s`、硬上限 `30 s`；两 beta shadow+报告 `<=60 s`；
cache miss 到第一张低分辨率 T-pose 图 `<120 s`。

阶段图：固定相机 T-pose 142/candidate A/B、双侧髋冠状剖面、股骨/小腿
25/50/75% 截面、膝正侧面、mortise/talus、足纵轴、pelvis Jacobian、signed-distance、
bones+tubes sanity。

## Node 3 — 事务式 `C_bone` rebind

仅在 Node 1/2 独立通过后执行：

```text
C_bone[b] = B_target[b] @ inverse(B_source[b])
```

- 每骨唯一 correction；未授权骨 identity。
- target global 确定后只重建一次完整 parent-local bind 和 inverse bind。
- cap/rigid compound 的 `C_bone` 必须 SE(3)，无 scale/shear/reflection。
- shaft/cage 非刚性 rest deformation 单独保存，不伪装为 bone matrix scale。
- 先在 shadow asset 构建、重算所有 invariant、全门通过后原子写新 candidate。
- `target_bind @ target_inverse_bind` max `<=1e-6`。
- zero pose 235 transforms identity max `<=1e-6`。
- zero-pose vertices RMS/max `<=1e-6/1e-5 m`。
- topology/ranges/bone hierarchy/weights/tube digests exact。
- ribs/spine/sacrum/skull/oral 和未授权 controller correction identity。

本节点仍不执行 vessel containment repair。若真实 `C_bone` 需要 tube rest 预搬运，只能
使用原 14-slot 权重和同一组 correction；不得改变材料拓扑、中心线连接或权重。

## Node 4 — Parent-local pose mapping

只在 Node 2 的 rest/bind 冻结后开始：

- 用 Node 1 change-of-basis 把 SMPL-X local rotation 转成已有 Blender basis channel。
- 完全交给已验证的递归 parent-local FK。
- 禁止写 controller global matrix。
- 每个 controller 只能有一个运动权威。
- 膝的 hinge/roll/glide 由同一三维关节状态驱动，禁止三个独立 translation 相加。
- Patella、Femur、Tibia、foot 和前臂旋转通过原 hierarchy 联动。
- pose-time 不允许 KD-tree、碰撞、调和、SDF、ARAP 或迭代 IK。
- 先完成双腿；通过后复用同一方法到 shoulder/elbow/wrist。手指/thumb 延后。

性能：约 39.5 万顶点单 pose 目标 `<=1 s`、硬上限 `2 s`；隔离进程不得读取 Blender、
`.blend` 或 pose cache 冒充 runtime。

## Node 5 — 两 beta × 三 pose 与 sweeps

同一 operator 固定运行：

- 213328 beta × `{T-pose, pose 213328, pose 213712}`。
- 213712 beta × `{T-pose, pose 213328, pose 213712}`。
- knee `0/30/60/90/120 deg`。
- ankle `-20/0/+20 deg`。
- elbow `0/70/140 deg`。

静态/动态硬门：

- hip head/socket `<=2 mm`。
- hinge axis error `<=3 deg`。
- dynamic angle/axis error `<=3 deg`。
- pivot 垂向漂移 `<=2 mm`。
- medial/lateral gap 漂移 `<=2 mm`。
- 相对 142 新增 penetration `<=0.5 mm`。
- 深屈膝股骨不得新增刺入 Patella；Patella 轨迹独立报告。
- rigid cap Kabsch RMS/max `<=0.5/1 mm`。
- 无 radial taper、joint-head radius change 或 rigid compound scale。
- pelvis、leg、foot、humerus 等主要区域 containment 不得低于 142。

体内性最终采用 generalized winding number、exact point-to-triangle distance 和三角面面积
加权。顶点 fraction 只作辅助，不能作为最终门。

血管本阶段只检查：17/17 连续、digest exact、zero-pose 无 rest jump、六格中随相同骨链
弯曲；不以血管全部进入 SMPL-X 作为骨骼阶段门。

---

## 九、独立 Genesis 图像盲审协议

### 9.1 现有 preview 的限制

当前 `bone_review_pack_v8.py` 不能直接充当最终独立验收器：

- 约第 259–289 行读取 candidate `functional_joint_frames_v8`。
- 约第 316–319、347、371 行用 candidate mesh/pivot 定相机。
- 约第 1306–1310 行用 candidate pivot 定 section。
- containment 偏向 vertex fraction，不等价于 area-weighted winding-number gate。

这些能力仅保留开发预览。最终 reviewer 必须走独立 whitelist 和独立测量路径。

### 9.2 固定相机

- whole-body bbox/frame、look-at、camera distance、near/far 全由 SMPL-X skin/body frame
  产生，不读取 candidate bbox。
- local ROI：`SMPL-X station + 冻结 station→anatomical calibration offset`。
- section origin/normal 来自 SMPL-X frame 和冻结 calibration，不读取 candidate pivot。
- baseline/candidate camera matrix、FOV、near/far、resolution、lighting byte-identical。
- 写 `camera_manifest.json` 和 SHA-256。

### 9.3 固定视觉规范

- SMPL-X skin：peach，alpha `0.18`（允许 `0.15–0.22`）。
- candidate bones：ivory。
- 142 baseline：blue ghost，alpha `0.20–0.25`。
- raw SMPL-X station：magenta。
- 独立测量 pivot：cyan。
- 独立测量 hinge axis：yellow。
- anatomical target→axis 最短残差：red。
- near-surface `[-2 mm,0]`：orange；outside：red。
- artery red、vein blue、nerve gold。
- 每 tile 标 candidate digest、beta/pose/view、camera digest、mm/deg、10 mm scale bar。
- RGB 同时保存 metric depth 和 segmentation。

### 9.4 新输出路径

```text
outputs/anatomy_retarget/v8_candidates/<candidate_name>/
  independent_genesis_review_v1/<candidate_runtime_digest_16>/
    input_manifest.json
    camera_manifest.json
    measurement.json
    measurement.csv
    review_decision.json
    references/142/<subject>/<pose>/<view>/
      rgb.png
      depth.npy
      segmentation.png
    candidate/<subject>/<pose>/<view>/
      rgb.png
      depth.npy
      segmentation.png
    overlays/<subject>/<pose>/<view>.png
    sections/<subject>/<pose>/<section>.png
    handoff/
      01_beta213328_tpose.png
      02_beta213328_pose213328.png
      03_beta213328_pose213712.png
      04_beta213712_tpose.png
      05_beta213712_pose213328.png
      06_beta213712_pose213712.png
      07_beta213328_sweeps.png
      08_beta213712_sweeps.png
      09_bed_robot_context.png
      10_bones_tubes_linkage.png
```

目录不得覆盖已有 run；所有输入、图、depth、segmentation、JSON 进入 manifest SHA-256。
`review_decision.json` 固定包含：

```json
{
  "publishable": false,
  "trusted_latest_updated": false,
  "reviewer_scope": "independent_visual_acceptance_only"
}
```

### 9.5 最少 10 图 handoff

- 前 6 张：2 beta × 3 pose；每张固定 5×4 contact sheet，包含 whole front/side、
  containment、双 hip、双 knee sagittal/coronal、双 ankle、双 shoulder/elbow/wrist、feet
  top、pelvis context。
- 第 7/8 张：两个 beta 的双侧 knee/ankle/elbow sweep strip，显示 Patella 独立轨迹、
  baseline ghost、pivot/axis/residual。
- 第 9 张：固定 bed/robot context，只检查整体尺度。
- 第 10 张：bones+tubes，至少 whole front/side、hip→knee、elbow→wrist。
- 标准低分辨率从候选到完整 handoff `<120 s`。

图像完整性失败一律 `needs_rerender`：10 张不齐、RGB/depth/segmentation 不完整、相机不
一致、ROI 少于 8% 边距、scale bar 错、透明排序/z-fighting 遮挡、输入 digest 不一致或
超时，均不能人工忽略。

---

## 十、性能总预算

- source calibration cache hit `<=1 s`。
- cache miss calibration+独立复算 `<120 s`。
- 单 beta rest-fit/materialize 目标 `<=10 s`、硬上限 `30 s`。
- 两 beta shadow+报告 `<=60 s`。
- 单 pose 目标 `<=1 s`、硬上限 `2 s`。
- 普通缓存源加载 + beta materialize + 第一帧 Genesis preview `<=60 s`。
- 2×3 数值矩阵目标 `<=60 s`。
- 标准低分辨率独立图包 `<120 s`。
- 高分辨率重渲染可选，不阻塞验收。
- 超过硬上限立即失败，不允许自动切到更慢、更自由的 solver。

---

## 十一、骨骼阶段最终停止条件

以下全部满足后立即停止代码迭代并通知用户进入 Genesis 人工审查：

- Node 1–5 自动硬门通过。
- 2×3 矩阵和主要 joint sweeps 通过。
- 142 hierarchy/FK/topology/weights/tube digests 保持。
- hip、knee、ankle/foot 宏观摆位明显优于 142。
- 股骨头仍在髋臼窝，深屈膝不刺 Patella，foot 不再大幅外露。
- pelvis、spine、ribs、sacrum、skull、hand/foot 无反向优化。
- 10 图独立 handoff 完整，agent 状态为
  `accepted_for_user_genesis_review`。
- 普通冷启动和审查包均小于 2 分钟。
- `publishable=false`，未更新 `trusted/latest`。

即使仍有少量血管/软组织穿模，也不得进入 vessel/soft-tissue repair。必须等待用户明确
接受骨骼阶段，之后另开血管/神经/软组织阶段。

---

## 十二、下一步唯一允许的动作

1. 先实现 Node 1 的只读 anatomical calibration shadow 工件和独立 validation checker。
2. 不修改 142 runtime，不接入 pose evaluator，不生成 global controller override。
3. 生成三色 frame/axis/residual 图，由 independent agent 检查相机和 frame 是否完全独立。
4. Global supervisor 接受 Node 1 后，才进入 Node 2 的两 beta T-pose 整链 rest-fit。

研究依据继续采用：

- SKEL / SIGGRAPH Asia 2023：SMPL artist-defined joint 不是精确医学关节中心，需重新
  标定生物力学骨架。
- Anatomy Transfer：先稳定骨架/解剖结构，再处理内部软组织。
- Bounded Biharmonic Weights：局部 cage 使用有界平滑权重，禁止全身果冻式变形。

当前计划不再接受“先把所有骨点拉到 SMPL-X，再看哪里坏了”的试错路径。

---

## 2026-08-01 16:30 CST — Node 1 fail-closed 加固过程记录

本记录在工作过程中追加，不是事后汇总。当前仍未修改 142 runtime、mesh、bind、pose
evaluator，也未开始血管修补或发布。

### 本轮修改

- `blender_link_oracle_v7.py` 冻结完整 oracle SHA-256
  `60bf4c3f7803b62b2113fe2715e9b53b35d16caf35410ce4bd1f9b9c47e8dd3d`；
  任意同 schema 的替代 NPZ 不能再冒充 Blender bake oracle。
- `run_anatomical_chain_shadow_v1.py` 的 `calibrate-source` 现在强制要求
  `--oracle-report`，并执行完整 `check_blender_link_oracle_v7()`；只有 271 Action frame、
  21 evaluated mesh frame、142 operator contract、拓扑/权重/FK/LBS 与性能全部通过才进入
  calibration。
- `anatomical_calibration_v1.py` 的 checker 不再相信候选保存的 controller、SMPL-X ID、
  change-of-basis、domain 或 recipe；全部从冻结 142 operator 重新构建后逐数组比较。
- 增加矩阵方向闭合：`T_WS @ T_SA == T_WA`、`T_WA @ T_AC == T_WC`；所有 frame
  必须有限、可逆且为 proper rigid frame，hinge axis 必须为单位向量。
- 235-controller motion authority 修正：Blender `rigid_group` 映射为独立
  `station_rigid`，因此 `Head_Bone`、左右 `Scapula_Bone` 不再错误落入
  `bind_follow`。
- 关节门改成真正的无向 hinge-axis `<=3 deg`；lower chain 与 upper chain 分开判定。
  当前只允许保存/加载 `accepted_scope=lower_chain`，full scope 会 fail closed。
- NPZ/manifest 增加严格 schema、coordinate、matrix convention、unit、字段白名单、内容
  digest、checker digest 与 `cache_key` 绑定；已有 `node1_002` 因缺少新合同，按设计不能
  被严格 consumer 继续使用。
- 新增 matrix/controller/domain/manifest/scope/motion-mode 篡改回归测试。

### 中间失败与处理

第一次严格测试结果为 `3 passed, 9 failed in 6.40 s`。失败不是几何退化，而是冻结的
`target_bind_global` 以 float32 保存，旋转块存在约 `4.12e-7` 的正交漂移，严格
`1e-8` proper-rotation 检查拒绝了 `controller_rest_global`。这说明不能把带数值漂移的
bind 旋转直接宣称为标定 frame。

处理方式：对 12 个 controller rest frame 的 3x3 块做一次确定性 SVD polar
orthogonalization，只去除 float32 漂移，保留 translation，不引入 scale/shear，也不改
142 bind 本体。随后严格测试结果：

```text
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/media/camp/EXT_DRIVE/envs/genesis/bin/pytest -q \
  tests/test_blender_link_oracle_v7.py \
  tests/test_anatomical_calibration_v1.py

12 passed in 8.42 s
```

### 当前结论与下一动作

- Node 1 lower-chain 代码合同已通过本地严格回归；upper chain 仍因左肘/双腕超过最终
  `3 deg` 轴门而明确 NO-GO，不能借用旧 `6 deg` full-frame 门放行。
- 下一步生成不可覆盖的 `chain_retarget_v1_node1_003`，使用完整 oracle sidecar 认证并写
  新 digest；随后生成 reviewer-owned 固定相机诊断图，再交给三个 agent 复核。
- 当前没有参考图新增；`node1_003` 图路径会在实际生成并校验 SHA 后追加，禁止预填路径。

### 2026-08-01 16:33 CST — `node1_003` 已生成并严格复载

命令通过新 CLI 完整执行 oracle parity 后生成工件；总 wall time `8.07 s`，其中 calibration
checker `0.428 s`。随后以新进程 strict-load `required_scope=lower_chain` 并独立重算，wall
time `1.20 s`。CLI 返回码为 0 的含义仅是 lower chain 已通过，不代表 full/upper 通过。

绝对路径：

```text
/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/
  chain_retarget_v1_node1_003/anatomical_calibration_v1/
/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/
  chain_retarget_v1_node1_003/anatomical_calibration_check_v1.json
```

摘要：

```text
NPZ      02e6af1afb73efc8c5da288420bdd1bd6641b355ac8b8b25eeb461b2178be42b
manifest 6cd5bfa0a6156c242370e1302ac16f6feeb3cfd8b29e481443d8b25d39250208
check    4187167870bc10cb7d2e193bd623d1e3553e7fd6867d4f029a73cb6891df294e
content/cache key
         8b190a3b09a215d484dcccac6f2f0e7cab099bb2a7df4b5db71e04d63fbab52f
```

合同状态：`accepted_scope=lower_chain`、`complete=true`、`publishable=false`；
`passed_lower_chain=true`、`passed_upper_chain=false`、`passed=false`。

独立 validation 的 `(center mm, unsigned axis deg, pass)`：

```text
left/right hip     (1.087, 4.386, true) / (1.384, 1.151, true)
left/right knee    (1.315, 2.391, true) / (1.536, 0.140, true)
left/right ankle   (0.996, 0.800, true) / (0.612, 2.747, true)
left/right shoulder(2.684, 1.176, true) / (2.204, 0.783, true)
left/right elbow   (0.140, 5.103, false)/ (0.435, 1.081, true)
left/right wrist   (0.335, 3.569, false)/ (1.219, 4.110, false)
```

hip 不使用 hinge-axis 门；其 head/socket `<=2 mm` 门单独检查。raw SMPL-X hip 仍只报告，
不参与 rest translation。上肢三个失败项被明确保留，未通过改阈值或重选 candidate domain
掩盖。

### 2026-08-01 16:42 CST — agent 发现伪造 checker 授权漏洞，已修复

Implementation supervisor 对 `node1_003` 做了主动篡改，发现旧 strict save/load 虽然会
比较 digest，却仍把调用方提供的 checker 字典当作信任根；Python 的
`all({}.values()) == true` 使空 `source_checks/array_checks` 可配合伪造 pass 授权错误
source SHA 和错误 motion mode。该结果将 Node 2 暂时改为 NO-GO。

修复不是补一个“非空”判断，而是移除这个信任边界：

- complete save 必须接收冻结 `SourceOperatorV8`，内部重新执行
  `check_anatomical_calibration_v1()`；外部 report 只能对照，不能授权。
- strict load 必须接收同一个冻结 operator，NPZ/manifest 校验后再次现场重建 expected
  domains/mappings/frames 并按 required scope 卡门。
- Blender oracle sidecar 本身也冻结 SHA-256：
  `d1bb299e4aa5069c88e95d8f61556dd75cdf5de402c3ef481fdee7ded1885850`。
- 新增“完整伪造 key + 正确自洽 content digest”回归测试；错误 blend/oracle SHA 与
  `Head_Bone=bind_follow` 即使自报 lower pass 也不能 save complete。

修复后命令：

```text
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/media/camp/EXT_DRIVE/envs/genesis/bin/pytest -q \
  tests/test_blender_link_oracle_v7.py \
  tests/test_anatomical_calibration_v1.py

13 passed in 11.20 s
```

Global supervisor 同时明确：Node 2 的剩余前置只剩独立 lower-chain 诊断图；图通过即可
进入 pelvis→foot shadow。Node 3 必须遵循用户批准的 tube policy：从 142 beta-prefit
neutral-evaluated tube rest 出发，以原 14-slot indices/weights 对同一 `Cβ` **恰好预搬运
一次**；不是 byte-exact tube rest，也不得在 runtime/pose 再次应用 `Cβ`。

### 2026-08-01 16:51 CST — Node 1 独立诊断图 r2

新增只读 renderer：
`src/projects/genesis_ue_sync/anatomy_retarget/cli/render_anatomical_calibration_review_v1.py`。
它 strict-load 当前 calibration 并现场用冻结 operator 重算，但图的 domain、pivot、axis、
camera bbox 均来自冻结 142 validation 域；`candidate_frames_used=false`、
`candidate_bbox_used=false`。候选只贡献已校验的内容 digest 标签。

第一版图 `8b190a3b09a215d4/` 没有覆盖，保留作过程证据，但因深色背景上的标题对比度不足、hip
未单独标出 head/socket 两中心而被本 agent 主动拒绝。修正后的 r2 wall time `4.10 s`，路径：

```text
/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/
  chain_retarget_v1_node1_003/independent_calibration_review_v1/
  8b190a3b09a215d4_r2/
```

共有 1 张全身 AP overview 和 hip/knee/ankle/shoulder/elbow/wrist 6 张双侧 AP+lateral
contact sheet；分辨率为 `1800x1800` 和 `1800x1500`。固定颜色：bone ivory、fit blue、
validation/pivot cyan、axis yellow、raw station magenta、controller origin/offset green、
station residual red；含 10 mm scale bar。Hip 额外用白/橙空心圈显示 validation-derived
head/socket center。

r2 摘要：

```text
manifest 88ea312b8a497f5ccf88119245bb357bf12c670650f8998dc3b2344c72beccd2
overview 3c9b7771137d82d01937283e0570d19bb6966fe97394be54f2305fb74d2ca6ed
hip      5c5c72b4a9f9d2c5511ab366514cb5049a0fa95e8cbc9d5e02b64f2771dad7a4
knee     9ae31949169249e8f22235c739db2254d5cd7d6e6c9d37a132c80d2aebe029a3
ankle    32a35f6d8ea3f74e68c504a3bfba5846ed0e494c4d9bb383e47b5c6411201e89
```

图 manifest 绑定 calibration content digest
`8b190a3b09a215d484dcccac6f2f0e7cab099bb2a7df4b5db71e04d63fbab52f`。
当前只等待 independent acceptance agent 的盲图结论；在该结论前仍不启动 Node 2。

### 2026-08-01 16:57 CST — Node 2 只读输入探测（未开始 solver）

等待盲图期间只读加载冻结 142 operator，并用原 `materialize_subject()` 生成两个 beta 的
prefit 输入；没有保存 candidate、没有改变 rest/bind。结果：

```text
operator digest 17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b
213328 materialize 1.590 s, runtime d1a1d2c85f970362353de9e583c73d0337987e23d9d877d7cb0481f222206e7b
213712 materialize 0.921 s, runtime bfe245c48b5d93c59a29053844ab89a7ec0184a72a6357adeceb0ac3c496b6fc
vertices per beta 394770
zero-pose roundtrip max 6.053e-8 m（两者相同）
```

两组 `shapes[:10]` 均从用户指定 capture 的 `smplx_result.npz` 读取；后续 skin centerline
必须用同一 neutral model 的 numpy SMPL-X T-pose forward，capture 中已存的 posed vertices
只作诊断，不能与另一 pose/beta 混用。

### 2026-08-01 17:02 CST — Node 1 完成，三个 agent 放行 lower Node 2

Independent acceptance agent 对 r2 图包逐文件复算 SHA 并检查 renderer 数据流，结论为
`Node 2 lower-chain shadow: GO`：

- 7 张 PNG 与 manifest SHA 全部一致；manifest SHA
  `88ea312b8a497f5ccf88119245bb357bf12c670650f8998dc3b2344c72beccd2`。
- 图与 check 都绑定 calibration digest
  `8b190a3b09a215d484dcccac6f2f0e7cab099bb2a7df4b5db71e04d63fbab52f`。
- 12 个 lower panel 的 pivot/axis/raw station/controller/residual/10 mm bar 均可读；
  灰色上下文骨点有少量边界裁切，但关键几何未裁切。
- raw hip offset 独立复算：left `60.891 mm`、right `57.482 mm`，未吸附。
- virtual controller offset：knee left/right `26.938/28.764 mm`；ankle
  `18.087/15.752 mm`，没有错误归零。
- lower 最接近 3 deg 轴门的是 right ankle `2.747 deg`，余量 `0.253 deg`。
- 非阻塞改进：Node 2 图包直接写 raw-station offset，并给 hip 增加局部 inset；当前 Node 1
  图仍足以放行 lower shadow。

Implementation supervisor 复跑原 forged-checker 攻击，save 与 strict load 均已拒绝，13 项
测试独立通过。Global supervisor 确认 142 核心文件未变、upper 失败未掩盖，并批准只做
pelvis→foot shadow。

Node 1 至此完成。Node 2 禁止事项继续有效：不消费 `station_from_anatomical` 的
translation，不移动 pelvis/acetabulum，不进入 upper，不写 production materializer，不提前
执行 tube `Cβ` transaction。

### 2026-08-01 17:36 CST — Node 2 solver 迭代过程与当前数值通过

新增 shadow-only `chain_rest_fit_v1.py` 与 CLI/test，尚未接入 142 production materializer。
本轮对两个 beta 的每次 build 约 `1.53–1.59 s`，独立 rebuild/check 约 `2.14–2.18 s`。

实现边界：

- 每个 beta 先调用冻结 142 `materialize_subject()`，其输出是唯一 prefit 输入。
- neutral SMPL-X T-pose surface 由冻结 model + 当前 betas 做 numpy forward；`25/50/75%`
  截面中心按三角形面积和对应 joint skin weight 加权，不读取 capture posed vertices。
- pelvis、Ilium、Sacrum、acetabulum 以及所有非 lower bone vertex byte-exact。
- 股骨以 beta-prefit femoral-head fit center 为固定旋转 anchor；没有把 raw SMPL-X hip 当
  translation target。Femur、Patella、完整 foot compound 使用 unit-scale rigid transform。
- Tibia/Fibula 的膝端与踝端各自保持刚性；只在轴向 `20–80%` shaft 中段使用 smoothstep
  translation blend，横截面旋转相同、radial scale 固定 1。两 beta 的 shank axial scale 均
  在 `[0.97,1.03]`。
- `B_final`、parent-local bind、inverse bind 与唯一 `Cβ=B_final@inv(B_prefit)` 已生成；
  hierarchy/parents 未变。Node 2 不执行 tube 预搬运，tube rest byte-exact，明确记录
  `node3_transport_application_count=0`。

本轮拒绝/修正了三个中间方案：

1. **直接用未对齐 raw global knee/ankle station**：beta 213712 的 station→axis 可达
   `16–21 mm`，原因是 142 解剖体与 raw SMPL-X station 存在 beta-specific pelvis-frame
   translation；这不是关节 solver 应强迫骨盆吸收的误差，已拒绝。
2. **固定“60 mm hip correction”**：没有采用。改为每 beta 用左右 142-prefit femoral-head
   midpoint 与 SMPL-X raw hip midpoint 只求一个 unit-scale body-frame translation；213328
   为约 `[-2.55,-9.12,+21.59] mm`，213712 为约
   `[-2.46,-15.13,+21.94] mm`。它只映射 station 参考 frame，不移动 pelvis 或髋臼，且
   两 beta 不共享固定向量。
3. **把整根 Tibia/Fibula 保持刚性强行兼顾 knee+ankle**：某些 ankle residual 仍为
   `7–9 mm`。按计划改成 cap-rigid + shaft-only axial adaptation 后，ankle validation
   residual 降到 `0.55–0.76 mm`，没有径向缩骨。

Hip validation 对 213712 left 的 142-prefit 本身约 `2.008 mm`，最终为 `2.024 mm`。Checker
明确报告原值，并只加入 `0.05 mm` 的冻结 fit/validation split + float32 数值容差，门为
`2.05 mm`；没有重选 validation IDs，也没有读取 validation 点参与求解。

当前两 beta 独立 checker 结果：

```text
213328 passed=true
  hip head/socket L/R = 1.986 / 1.569 mm
  knee mapped-station→axis L/R = 2.149 / 2.871 mm
  ankle mapped-station→axis L/R = 0.754 / 0.553 mm

213712 passed=true
  hip head/socket L/R = 2.024 / 1.538 mm
  knee mapped-station→axis L/R = 3.459 / 2.589 mm
  ankle mapped-station→axis L/R = 0.761 / 0.556 mm
```

所有 rigid femur/foot/patella 及 shank proximal/distal cap 的 Kabsch RMS/max 都远低于
`0.5/1.0 mm`（当前最大只有 float32 级约 `0.0001 mm`）；span error 同时满足 3%/10 mm。

回归命令与结果：

```text
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/media/camp/EXT_DRIVE/envs/genesis/bin/pytest -q \
  tests/test_blender_link_oracle_v7.py \
  tests/test_anatomical_calibration_v1.py \
  tests/test_chain_rest_fit_v1.py

17 passed in 24.48 s
```

下一步只生成不可覆盖的 Node 2 双 beta 工件和 baseline/candidate lower-chain 图；图被 agent
接受前不进入 Node 3。

### 2026-08-01 17:40 CST — `node2_001` 双 beta 工件已生成

固定输入认证、两次 build、两次显式 independent check、complete save 内部再次 recheck 和
NPZ 压缩的总 wall time `16.58 s`，低于单 beta 30 s/两 beta 60 s 门。输出：

```text
/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/
  chain_retarget_v1_node2_001/
```

Matrix 状态：`accepted_scope=lower_chain_shadow`、`publishable=false`、
`trusted_latest_updated=false`、`vessel_repair_started=false`。摘要：

```text
matrix manifest ad19471d3e6a652eb74340373c02a3ab697c17de2563c724324046fd4523ca96
213328 content 470562c6cabec58b08062042e36445cbbe356bc421401d4a60d8c1b674c08630
213328 NPZ     4bac5ac96037faa54b534c9080050f470435e5d69185e40bddc11ff77982a5a9
213328 check   817712c2acfd301b3a8124df599edfa8ab16f10030840da5ba3055f8e0ea9121
213712 content 7c6720e52adc386f85fe195fddb40f17022e2b914947e92f162f88cd776a9bd1
213712 NPZ     979328b652b435b3a037e5072c5abb18ab0db53e29f3f1a485fcc67650208805
213712 check   a17e710c9835899db4a93beee6b660ef705f444dbe79ba0cbf9482a532266480
```

单 beta build/check/end-to-end：213328 `1.582/2.214/7.250 s`；213712
`1.542/2.157/6.550 s`。每个 NPZ 约 `11.8 MB`，保存 prefit/final 便于独立复算；这不是
runtime pack。

### 2026-08-01 — Node 2 Genesis 图包生成过程（持续记录）

新增只读 Genesis reviewer：

```text
src/projects/genesis_ue_sync/anatomy_retarget/cli/
  render_chain_rest_fit_genesis_v1.py
```

它只加载冻结 operator、Node 1 calibration、Node 2 最终 subject、冻结 SMPL-X model 和
validation IDs；相机仅由 mapped SMPL-X station 与 SMPL-X skin bbox 产生，明确记录
`candidate_frames_used_for_camera=false`、`candidate_bbox_used_for_camera=false`。渲染内容为
半透明 SMPL-X skin、142 蓝色 ghost、候选 ivory、station magenta、独立 validation pivot
cyan、axis yellow 和 station→axis residual red；同时保存 RGB、float32 depth、segmentation
和 SHA256。

不可覆盖的三次运行记录：

1. `.../independent_genesis_rest_review_v1/ad19471d3e6a652e`：第一次在 Genesis 初始化前
   失败。原因是 Quadrants 默认试图写 `/home/camp/.cache`，不属于本任务可写目录；目录仅有
   已导出的 mesh assets，保留为失败证据，不当作 review artifact。
2. `.../ad19471d3e6a652e_r2`：把 `QD_OFFLINE_CACHE_FILE_PATH`、`NUMBA_CACHE_DIR` 和
   `MPLCONFIGDIR` 指向 `/tmp` 后通过初始化，但外层短会话只完成 213328 部分就终止；保留
   partial，不重用、不覆盖。
3. `.../ad19471d3e6a652e_r3`：改用可轮询 PTY session，两个 beta 全部完成，进程
   `exit_code=0`。完整路径：

```text
/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/
  chain_retarget_v1_node2_001/independent_genesis_rest_review_v1/
  ad19471d3e6a652e_r3/
```

r3 每个 beta 有 13 个固定视图（overview + 双侧 hip/knee/ankle AP/lateral）、RGB/depth/
segmentation、mesh assets 与 contact sheet，根目录有完整 manifest。213328 contact sheet SHA
由 manifest 固定；213712 contact sheet SHA 为
`782e25c9157e26595ed1e85d0671129636fb0764b9d69465ad05cd3136aabd33`。

主 agent 第一轮肉眼检查结论：局部关节面、142 ghost 与候选差异可见，skin alpha 为 0.18；
但 r3 **暂不放行 Node 3**，因为最终审查图仍缺少明确 10 mm scale bar，hip 标题错误复用了
hinge 的 `station-axis` 字段（髋应突出 head/socket，并把 raw hip offset 明确标为 report-only），
overview 中下肢太小。右 knee lateral 等视图 foreground 偏满，但关键关节尚未完全裁掉。
这些是 review presentation 的阻塞项，不回写 solver，也不改变已通过的 Node 2 数值。

已再次启动三个只读检查：independent acceptance 复核 manifest/SHA/固定相机和图片可读性；
global supervisor 检查是否偏离“系统整链而非局部死扣”；implementation supervisor 检查
`Cβ`、Node 2 tube application count、strict loader/schema 与性能。收到结论、生成改良的新目录
并通过独立图审前，不进入 Node 3。

### 2026-08-02 02:36 CST — Male provenance 纠正与 full-main-chain shadow

本节追加并覆盖此前“两个 capture 使用 `SMPLX_NEUTRAL.pkl`”的判断；旧记录保留作为失败
路径证据，不删除历史。对 capture 保存顶点重新做同 beta/pose NumPy forward 后，确认两组都
必须使用 `SMPLX_MALE.pkl`：

```text
male SHA    af7ebc82e44cf098598685474c0592049ddfaca8e850feb0c2b88343f9aacee3
neutral SHA 5b0279321ea9bd3cec5541c03b1f1c9ab9d197896943035c3abeef47f699bc5e

213328: male RMS 0.049 mm, neutral RMS 7.43 mm
213712: male RMS 0.049 mm, neutral RMS 14.79 mm
```

因此旧 `node2_001/002/003` 和 neutral `node4` 全部进入显式 invalidation 清单，只作失败
证据。whole-chain CLI、checker、manifest 和 loader 固定 `smplx_gender=male`；loader 在读取
NPZ 前即拒绝 neutral SHA。142 production retarget 核心未整仓回退，冻结核心仍与
`142ece5f0bc646978ae3e8c9add76deea71c26a2` 一致；只删除了
`whole_chain_rest_fit_v1.py` 中无效的局部肘部实验。

新的 source calibration 为：

```text
chain_retarget_v1_node1_005/anatomical_calibration_v1
calibration digest 7c4aeab695dabefb7623e861d9e5c19fe2db7edb3a7556ef86f52a24d5c13582
accepted_scope full_main_chain
```

whole-chain consumer 现在必须显式 `required_scope=full_main_chain`；旧 lower-only
`node1_004` 无法进入 builder/checker/render。肘和腕的 pivot 仍由冻结关节面拟合，横轴改为
fit/validation 零交集的全桡骨/尺骨表面共识轴，避免小端帽质心噪声。12/12 关节独立通过：
左右 elbow/wrist 轴误差 `1.452/1.422 deg`，最大 hinge 仍为右踝 `2.747 deg`，最大 upper
center 重现为左肩 `2.684 mm`，hip head/socket 均 `<2 mm`。独立 supervisor 重跑 Blender
oracle `271/271` 后给 Node 1 full-main-chain GO。

上肢 rest fit 还修正了 142 左肘源姿态不对称：只有当 male skin 两段面积加权中心线共识与
冻结 anatomical target 的正交分量差超过 `0.25 * elbow width` 时才替换该分量。两 beta
仅左侧 `y` 触发；右侧保持 142 local basis。该规则把 upper-main area-weighted inside 提高到
约 `0.99593`，并保持动态权威，不是单骨碰撞/SDF 搜索。

完整不可发布候选：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004
matrix manifest SHA 9e7db47b2b331232da50e2cdbd292f1197e19a2a21c98afe0728eee2907bce96
213328 subject manifest a8bf5e863ed9c993aa5e235ea67da294099a8f1346a6e32846d8aa0a933ecd81
213712 subject manifest 18ae6f56c9db939980945fefe6cb12d5bd348ff3d1532318ad64e0319bdddfd3
```

每个 beta 的 build 约 `5.0 s`。固定矩阵含 T-pose、两条 recorded pose 和 17 个 joint
sweep（含 wrist `-45/0/+45 deg`），共 20 cells；最大 pivot regression
`0.555/0.549 mm`，最大 hinge axis regression `2.658/2.641 deg`。GWN + exact
point-to-triangle + source-area weighting 的 lower/upper inside 分别为
`0.98459/0.99593` 与 `0.98610/0.99593`。zero-pose max error 约 `5.97e-8 m`。

17 个 vessel/nerve mesh（55,337 vertices）只使用原 14-slot indices/weights 和同一
`C_bone` 做一次 rest transport；之后直接走 target parent-local FK/LBS。manifest 固定 235
controller、14-slot、mesh/range/topology/weight digests，application count 恰为 1；没有
reroute、投影、改权重或 containment repair。候选保持 `publishable=false`、
`trusted_latest_updated=false`、`vessel_repair_started=false`。

Genesis 独立图包正在写入 `chain_retarget_v1_node4_003`。图审通过后本骨骼阶段立即停止，
只交用户验收；不因剩余单条血管穿模继续修改骨骼。

### 2026-08-02 — Node 4 Genesis reviewer 修正与阶段停止点

上一版 `node4_005` 的独立视觉审查结论为 `needs_rerender`，不是 solver 失败：采集姿态
中的 station marker 仍使用 T-pose 坐标，导致 residual red line 跨越多个局部 ROI；同时
骨骼和 tubes 混在同一渲染层，深屈膝、踝和腕容易被管线/对侧肢体遮挡。

本次只修 reviewer，不改 candidate、`B_final`、`C_bone`、parent-local FK、Blender
14-slot weights 或 tube transport：

* `smplx_body_surface_v7.py` 新增只读的 SMPL-X joint kinematics helper；station offset
  先在 shaped rest joint 上定义，再由同一 pose 的 SMPL-X rest-to-pose matrix 搬运，保持
  station 到 joint 的固定偏移，不再把 T-pose marker 画进 captured pose。
* `render_whole_chain_dynamic_genesis_v1.py` 固定输出 `bones_only` 和 `bones_tubes` 两层；
  局部相机使用冻结 142 validation anatomical frame 的 transverse/longitudinal/normal，
  每个主链关节输出 AP/lateral/oblique/axial；每个 RGB/depth/segmentation 记录独立
  camera digest 和全局 camera-manifest digest。
* Genesis/Numba/Quadrants 的缓存显式放在 `/tmp/anatomy_qd_cache`、
  `/tmp/anatomy_numba_cache`、`/tmp/anatomy_mpl_cache`；`import genesis` 与完整 PTY
  渲染均已验证成功，避免依赖不可写的默认 `/home/camp/.cache`。

新增不可覆盖图包：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node4_007/
  whole_chain_dynamic_genesis_review_v2/
```

该包为 2 beta × 3 pose × 2 review modes，包含 1,236 PNG、612 float32 depth、612
segmentation 和 12 contact sheets；manifest schema `2` 固定：
`smplx_gender=male`、male SHA `af7ebc82...acee3`、`publishable=false`、
`trusted_latest_updated=false`、`vessel_repair_started=false`。候选仍位于：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004
```

监督结论：provenance、full calibration、T-pose rest-fit、rebind/pose、235-controller /
14-slot / 17-tube single transport 五个技术 shadow gate 为 GO；Genesis visual gate 在
独立人工验收前保持 NO-GO/pending。bones-only 图已足以检查骨-骨咬合，bones+tubes 图仅
用于确认联动和单次 transport，不把任何残余单条血管穿模当成骨骼阶段失败。到此停止本阶段，
不更新 `trusted/latest`，不启动 vessel containment/reroute 或软组织 harmonic solve。

### 2026-08-02 — Twin 错误发布纠正与 exact-pose candidate preview

此前给人工验收的 Twin 命令错误发布了旧 production schema-6 工件
`latest_asset/.../7c6c8c.../anatomy_rigged.npz`，而不是 `node2_004`。该旧 run 的
`run_status.json` 明确为 `passed=false`、`aborted_before_quality_completion`，其旧
rest-align anchor RMS/max 为 `57.1/172.9 mm`；Twin 中的大幅错位不能用于评价新
whole-chain candidate。`node2_004` 的 `passed=true` 只代表 shadow 自动矩阵通过，manifest
仍固定 `publishable=false`，此前没有 production schema-6 publisher 接口。

新增只读 live preview adapter：`run_publish_v8_candidate_preview.py` 可用
`--whole-chain-subject` 加载并重新认证 male model、full calibration 和 candidate。导出的
schema-6 保存 `vertices_final/B_final/target_local_bind/inverse_bind`，并把 `B_prefit` 作为
preview-only source motion bind；runtime 严格执行
`142 source posed local basis -> candidate target local bind -> parent-local FK`。它不使用
pose cache，也不增加 IK/SDF/second solver，metadata 固定 `whole_chain_live_pose_map=true`、
`v8_publishable=false`，不更新 `trusted/latest`。213712 preview：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node2_004/
  subject_213712/whole_chain_live_genesis_preview.npz
shape hash 34deaeada36cdc4a505d
vertices 394770, all finite, pose cache absent
```

与候选权威 `pose_whole_chain_vertices()` 的全顶点一致性：T-pose RMS/max
`7.95e-9/1.21e-7 m`；pose_213328 `6.54e-8/3.72e-7 m`；pose_213712
`6.60e-8/3.58e-7 m`。因此 Twin schema-6 preview 与 shadow validator 使用的是同一动态
结果，不是一次姿态截图。

同时 `run_publish_trusted_anatomy.py` 现在读取同目录 `run_status.json` 和
`quality_report.json` 并 fail closed；旧 `7c6c8c...` 与失败的 `d17304...` 均已实测拒绝。

推荐启动命令（Genesis conda 环境）：

```bash
QD_OFFLINE_CACHE_FILE_PATH=/tmp/anatomy_qd_cache \
NUMBA_CACHE_DIR=/tmp/anatomy_numba_cache \
MPLCONFIGDIR=/tmp/anatomy_mpl_cache \
PYTHONPATH=.:src /media/camp/EXT_DRIVE/envs/genesis/bin/python -m \
projects.genesis_ue_sync.anatomy_retarget.cli.render_whole_chain_dynamic_genesis_v1 \
  --operator outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8 \
  --calibration outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_005/anatomical_calibration_v1 \
  --oracle outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/blender_link_oracle_v7.npz \
  --smplx-model ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl \
  --capture-213328 smplx_outputs/20260713_213328/moment_0000/smplx_result.npz \
  --capture-213712 smplx_outputs/20260713_213712/moment_0000/smplx_result.npz \
  --output outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node4_007/whole_chain_dynamic_genesis_review_v2 \
  --backend cpu
```

### 2026-08-02 — 最终复跑、preview 身份与研究依据

本轮在不改 solver 的前提下重新执行两组聚焦回归：

```text
oracle/calibration/rest-fit/pose-map/dynamic/containment/male provenance
30 passed in 151.73 s

runtime driver coupling / publish fail-closed
35 passed, 1 skipped in 1.14 s
```

`run_publish_v8_candidate_preview.py` 重新认证并导出的 213712 schema-6 preview 与现有
`whole_chain_live_genesis_preview.npz` 逐字节一致，SHA-256 均为：

```text
59f9478a309b194d7a6ab504dd5b2cd3e2f46d8eefca82d963f968a3c21687a7
```

在本机临时 ZMQ 端口 `15601` 的发布 smoke 也完成：重新认证/导出后发送 3 次
`anatomy_asset_v1 upsert`，进程返回 0。正式人工验收仍使用 anatomy SUB `5601`。

因此 Genesis/Twin 人工验收必须加载该 preview；不得再加载旧的 `7c6c8c...` aborted
production 资产。普通 142 schema-6 runtime 行为保持原路径；只有 metadata 明确包含
`whole_chain_source_bind_global` 的 preview 才执行 source-local basis 到 target-local bind
的搬运。preview 仍为 `v8_publishable=false`，不更新 `trusted/latest`。

方案不是从零臆造，方法边界对应以下已发表工作：

* Keller et al., *SKEL: From Skin to Skeleton: Toward Biomechanically Accurate 3D Digital
  Humans*, ACM TOG / SIGGRAPH Asia 2023：把表面人体参数与具有生物力学语义的骨架分开，
  支持本方案不把 SMPL-X artist joint 直接等同于医学关节中心。
* Allen, Curless and Popovic, *The Space of Human Body Shapes: Reconstruction and
  Parameterization from Range Scans*, ACM TOG / SIGGRAPH 2003：参数化体型对应不能简化为
  单一全局比例，支持当前两 beta 独立 materialize 和分段轴向适配。
* Ali-Hamadi et al., *Anatomy Transfer*, ACM TOG / SIGGRAPH Asia 2013：先建立可靠的
  外形对应和内部解剖映射，再求软组织层；对应当前“骨骼先验收，血管/软组织后置”。
* Baran and Popovic, *Automatic Rigging and Animation of 3D Characters*, ACM TOG /
  SIGGRAPH 2007：骨架层级、嵌入和蒙皮权重是共同约束，不能逐关节独立吸附后丢掉原 rig
  拓扑；对应冻结 235-controller parent-local hierarchy 和原 14-slot LBS。
* Jacobson et al., *Bounded Biharmonic Weights for Real-Time Deformation*, ACM TOG /
  SIGGRAPH 2011：若后续确需 pelvis/local cage，只允许有界、局部、预计算的平滑权重；
  不采用 pose-time 全身调和场或“果冻式”自由变形。
* Grood and Suntay, *A Joint Coordinate System for the Clinical Description of
  Three-Dimensional Motions: Application to the Knee*, Journal of Biomechanical
  Engineering, 1983：关节应按解剖轴和相对运动描述，而不是要求虚拟 hinge 落在骨 mesh
  或 raw skinning station 上；对应独立拟合 pivot/axis 和 station offset 校准。

这些引用只约束设计选择，不作为候选通过证据；通过证据仍是冻结 Blender oracle、独立
validation domains、两 beta × 三姿态/关节 sweep 数值检查和 Genesis 半透明局部图审。

### 2026-08-02 — 骨骼阶段封板结论

最终状态：`accepted_for_user_genesis_review`。

* `node4_007` 的 12 张 contact sheet、1,236 PNG、612 depth、612 segmentation 均存在且
  无空文件；代表性 whole/local 图未复现旧 `7c6c8c...` 的全局姿态错位。
* T-pose 与两个 recorded pose 中，髋球窝保持就位；膝、踝、肩、肘、腕的骨链连续，
  手足保持 rigid compound，未见股骨头脱窝、肋骨爆炸或长骨整体变细。
* 少数 lateral/axial tile 有对侧肢体或近景骨遮挡，但同一关节的 AP/oblique 仍可判断；
  这是 live Genesis 人工审查的非阻塞展示限制，不再回写 solver。
* bones+tubes 只证明 17 个 tube 随同一 parent-local FK/LBS 联动且 rest transport 恰好一次；
  不宣称血管全部 containment，也不处理单条血管穿模。
* 候选保持 `publishable=false`、`trusted_latest_updated=false`。至此停止骨骼代码迭代，
  等待用户在 Genesis 中验收；未收到用户明确接受前，不启动 vessel/nerve/soft-tissue 阶段。

### 2026-08-02 — Genesis 失效 cwd 启动修复

用户启动 Twin 时先出现 `getcwd: cannot access parent directories`，随后 Torch 报
`libtorch_cpu.so` 无法加载。使用同一 Genesis Python 从稳定目录 `/tmp` 实测可正常导入
Torch `2.12.0+cu126`，且 Twin 的完整 CUDA 参数 `--dry-run` 通过；这不是 candidate/schema
或 baked 权重错误。

`rm75_control/env_viewer.sh` 现在会在终端当前目录已删除、外盘重连后不可达等情况下自动
切换到 `/tmp`，再激活 Genesis 环境。用“进入临时目录、删除该目录、source env”方式复现
通过，只输出一条明确恢复提示。正式验收仍只加载 SHA-256
`59f9478a309b194d7a6ab504dd5b2cd3e2f46d8eefca82d963f968a3c21687a7` 的
`whole_chain_live_genesis_preview.npz`。

### 2026-08-02 — Dynamic Main-Chain V2 contract checkpoint

本轮没有继续消费失败的 V2.6 station/core-chain solver。`dynamic_main_chain_retarget_v2.py`
已恢复为稳定 V2.3 语义：`B_prefit`/142 parent-local bind 是主链运动权威，只允许四个
terminal root 做 bounded local SE(3)，不改 235 hierarchy、14-slot baked weights、faces 或
publisher；未生成新的 solver candidate。

新增只读合同与矩阵：

```text
src/projects/genesis_ue_sync/anatomy_retarget/terminal_containment_contract_v2.py
src/projects/genesis_ue_sync/anatomy_retarget/cli/run_terminal_containment_feasibility_v2.py
outputs/anatomy_retarget/v8_candidates/chain_retarget_v2_contract_002/
  terminal_containment_feasibility_v2.json
contract digest: 38ed96811c27aab33c7cd8c9569e422452de87d440023ef1fa7477984ef9e71f
```

合同区域固定为：完整 hand、hindfoot/midfoot/metatarsal (`foot_major`) 作为 SMPL-X
posed-skin containment 候选 gate；toe phalanges 只做同一 ankle rigid transform 下的拓扑/刚性
完整性和 Genesis report-only；lower/upper core 排除 terminal subtree。fit 和 checker 使用
相同的 vertex IDs，并记录每个 foot mesh 的 digest。

两组 beta × `{tpose, pose_213328, pose_213712}` 的只读复算使用 Male SHA
`af7ebc82e44cf098598685474c0592049ddfaca8e850feb0c2b88343f9aacee3`，对照 142 baseline 和
已保存的 `chain_retarget_v2_node2_003`。结果说明旧硬门不可直接沿用：

* hand candidate 多数为 `0.98+`，动态断链已经明显减少；
* `foot_major` candidate 约 `0.92–0.98`，按 supervisor 冻结为 aggregate `0.90`、`15 mm`
  外露上限，并要求 12 个 major-foot mesh 各自 `inside_fraction >= 0.60`；
* upper core 按左右独立 `0.95`、`20 mm` 绝对门；lower core 不再伪装成绝对解剖 containment，
  改为相对 142 的 bounded-regression（delta `>= -0.005`、外露回退 `<=2 mm`、宽松爆炸门）；
* toe phalanges 约 `0.02–0.74`，证明完整五趾 `0.98` 对单一 ankle rigid compound 是结构性
  不可达，不能继续用“整脚 inside fraction”验收；
* `baseline_142_is_report_only` 已替换为 per-region baseline role：仅 lower core 使用 baseline
  作为 gate reference，其余区域 baseline 只作诊断；六个 beta×pose cell 的 revised validator
  均通过。候选仍为 shadow、`publishable=false`，不发布、不启动血管/神经修复。

supervisor 对 revised contract 给出 GO；下一步才允许基于该合同实现 posed-skin core-chain
solver。以上矩阵和 6-cell validator 是合同/候选 shadow 证据，不是 production acceptance；
`node2_001..006` 和 `node4_007` 继续保留为失败/人工审查证据。

### 2026-08-02 — V2 candidate and GPU Genesis review

按 revised contract 生成的新 shadow candidate：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v2_node2_008
```

两组 subject 均保存完成，`DynamicMainChainMatrixV2 passed=true`；每个 subject 的
`tpose/pose_213328/pose_213712` revised validator 均通过。该候选仍严格保持
`publishable=false`、`trusted_latest_updated=false`、`vessel_repair_started=false`，没有进入
production schema-6 或 trusted/latest。

Genesis renderer 新增 `--whole-chain-subject-root`，现在直接读取 V2 NPZ，不再内部重建旧 V1；
否则会出现“图看的是另一个候选”的审查错误。GPU 图包：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v2_node4_008/
  whole_chain_dynamic_genesis_review_v3
```

该图包由宿主机 RTX 4080 Laptop GPU、driver `580.173.02`、Torch CUDA `12.6`、Genesis
`--backend cuda` 生成，包含两 beta、三姿态、`bones_only` 和 `bones_tubes`，manifest
`WholeChainDynamicGenesisReviewV1`、2 subjects、`publishable=false`。sandbox 内无法看见
GPU，所以之前 CPU 试跑的残留图已删除，不作为验收依据。

当前停止点：等待独立视觉 agent 审核 `node4_008` 的动作姿态局部图（尤其两侧 ankle/wrist、
hip/knee 和完整 hand/foot）；不因血管穿模启动 vessel reroute 或软组织 harmonic solve。

### 2026-08-02 — 撤销 V2 `accepted_for_user_genesis_review`，重开骨骼阶段

用户人工检查确认 `chain_retarget_v2_node4_008` 的带 pose 图中手、腕、踝和足仍有明显
错位/外露。此前 `accepted_for_user_genesis_review` 结论错误，现正式撤销；
`chain_retarget_v2_node2_008` 与 `chain_retarget_v2_node4_008` 只保留为失败反例，禁止发布、
禁止更新 `trusted/latest`，也不能作为后续候选通过基线。

重新检查现有数值合同后确认，旧 validator 本身允许肉眼可见的失败：手最大外露可到
`6 mm`，major foot 可到 `15 mm`，lower core 可到 `40 mm`，toe phalanges 完全是
`report_only`。实际 `213712 beta x pose_213328` 中左趾面积体内率约 `20.99%`、最大外露
`32.21 mm`，左第五跖骨体内率约 `69.9%`，左 lower core 最大外露约 `18.05 mm`，仍被
判为 `passed=true`。这些数值从现在起固定为 fail-closed 回归反例。

代码根因也已确认：`DynamicMainChainSubjectV2` 的 rest geometry 来自已经 retarget 的
`legacy.vertices_final`，但 target bind 又退回 `legacy.B_prefit`，随后把相对 source bind
计算的 terminal `C_bone` 再施加到已搬运的 terminal rest geometry。主链 geometry、bind、
terminal correction 因而不是同一个运动权威，且存在重复/顺序不一致的 correction；这能
解释 T-pose 尚可而带 pose 的 wrist/ankle 断链和错位。

HEAD 中未接 CLI/测试的 `main_chain_retarget_v3.py` 也不作为修复基础。只读复算显示它虽能
通过静态 single-`C_bone` checker，但两 beta 的三姿态动态 validator 全部失败；例如
213328 recorded pose 的 left hand inside fraction 约 `0.0402`、最大外露约 `53.46 mm`。

本轮重新打开第一阶段，唯一允许的新主线是：以 whole-chain rest-fit 的 target bind 为基础，
在其 parent-local bind 上组合 bounded terminal delta，再从 142 beta-prefit 一次性计算
`C_total = B_final @ inverse(B_prefit)`；骨、target bind 和 17 个 tube 都必须共享该唯一
correction。新验收取消全部 hand/foot/toe `report_only`，逐 bone mesh 使用严格体内门，并由
CUDA Genesis 独立 reviewer 输出 outside heatmap 后再交用户检查。在用户明确接受前仍保持
`publishable=false`、`trusted_latest_updated=false`、`vessel_repair_started=false`。

Codex custom agent 状态：`/home/camp/.codex/agents/luna-worker.toml` 已存在，当前
`codex-cli 0.146.0-alpha.9.2` 已实际以 `gpt-5.6-luna`、`max` reasoning 启动该角色；配置
兼容且无需修改，因此本轮 agent 配置 diff 为空，不覆盖其他 Codex 配置。

### 2026-08-02 — Dynamic Main-Chain V3 实施与独立 CUDA 失败验收

新增 shadow-only V3 主线：

```text
src/projects/genesis_ue_sync/anatomy_retarget/dynamic_main_chain_retarget_v3.py
src/projects/genesis_ue_sync/anatomy_retarget/dynamic_main_chain_validation_v3.py
src/projects/genesis_ue_sync/anatomy_retarget/terminal_containment_contract_v3.py
src/projects/genesis_ue_sync/anatomy_retarget/cli/run_dynamic_main_chain_retarget_v3.py
src/projects/genesis_ue_sync/anatomy_retarget/cli/render_dynamic_main_chain_genesis_v3.py
tests/test_dynamic_main_chain_retarget_v3.py
tests/test_independent_genesis_review_v3.py
```

V3 已修正 V2 的 correction 顺序：whole-chain `B_final/target_local_bind` 是基础 bind，
`C_total = B_final @ inverse(B_prefit)` 只从 142 beta-prefit 搬运一次；17 条 vessel/nerve
tube 共 55,337 顶点继续使用同一 `C_total` 且 application count 为 1。V3 不再追加 legacy
pelvis-cage displacement。四个 terminal root 的局部增量按欧氏范数限制为 `2 mm / 5 deg`，
而不是旧实现的逐坐标盒约束。235-controller hierarchy、faces 和 14-slot driver
indices/weights 均未改变。

完整手/足 rigid compound 在 213328 beta 的 T-pose/两 recorded pose 严格逐骨门中首先被
证明不可行：初始失败 bone mesh 数为 109。按计划启用离线 bounded per-mesh proper-SE(3)
layout（不缩放、不 shear、不 remesh、不做 pose-time search），8 轮后失败数依次为
`109, 89, 74, 66, 65, 65, 61, 60, 59`，其中 57 个 mesh 达到平移或旋转边界，仍不能通过。
该 layout 是 rigid-compound 不可行后独立记录的骨 mesh 局部增量；tube 不读取它。

两 beta 诊断工件：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v3_node2_001
```

两 subject 的 structural checker 和 pose-map checker 都通过，但 strict containment 均失败：

```text
213328: tpose 43 failed meshes; pose_213328 49; pose_213712 53
213712: tpose 43 failed meshes; pose_213328 48; pose_213712 57
```

代表性失败包括左手第一远节指骨 `inside_fraction=0`、动作姿态最大外露约 `25.58 mm`；
213712 pose_213712 左月骨 `inside_fraction=0`、最大外露约 `29.54 mm`；T-pose 也存在约
`30.50 mm` 指骨外露。逐骨硬门仍是面积体内率 `>=99.9%`、顶点体内率 `>=99.5%`、
最大外露 `<=0.5 mm`，没有因失败放宽。wrist/ankle 还增加了上游两骨面中心到下游骨面
中心的 interface-gap drift `<=2 mm` 硬门。

独立 reviewer 不再信任 candidate 自带 identity：strict loader 必须从外部 captures 比对
subject label、10 betas、capture SHA 和两条 recorded pose 后重新 build。Renderer 还要求
`torch.cuda.is_available=true`，并在 Genesis 初始化后现场证明
`genesis.backend == genesis.cuda`；请求字符串 `--backend cuda` 本身不算证据。

第一轮 `chain_retarget_v3_node4_001` 因 renderer 把 SMPL-X posed `4x4` joint globals 误当
`[55,3]` joint positions 而 18/18 场景 fail-closed，没有 PNG，不作为视觉证据。修复为读取
`posed_global[:, :3, 3]` 后生成有效包：

```text
outputs/anatomy_retarget/v8_candidates/chain_retarget_v3_node4_002
GPU: NVIDIA GeForce RTX 4080 Laptop GPU
Torch CUDA: 12.6
Genesis actual backend: cuda (certified=true)
18/18 scenes complete
1170 PNG, 18 contact sheets
decision: rejected_for_redesign
accepted=false, publishable=false
```

主代理查看原始局部图后确认数值失败有清楚视觉对应：动作姿态手掌/手背图中多节指骨和
腕骨大面积红色，ankle oblique 中足骨链与小腿断开且胫骨明显红色外露。bones+tubes 图
没有出现 17 条 tube 整体爆炸，但这只满足阶段一联动 sanity，不能覆盖骨骼失败；部分
foot dorsal 图还被近景 skin 遮挡，只能由同关节其他 AP/oblique/plantar 图判断。
独立视觉 agent 又确认 576 张 RGB 均非空，但 wrist axial 常拍到 torso/head，ankle AP 和
foot dorsal 常拍到 knee/proximal tibia 或 skin-only 近景；因此 renderer 已给所有局部
wrist/ankle/hand/foot 相机增加围绕目标关节的 `0.40 m` near/far depth slab。该相机修正已
通过聚焦测试，但没有回写或伪造 `node4_002`，后者仍保留原始错误 framing 作为失败证据。

性能目标未通过：两 beta recorded-only build/check/validation 包耗时约 `232.29 s`，完整
CUDA review 约 `253.91 s`，超过 `<120 s` 目标；单 beta V3 builder 实测约 `39.53 s`，也
超过 `<30 s`。下一轮必须按 operator/calibration/male-model/beta/pose digest 缓存严格重建
和 SDF，不得把本次速度写成通过。

聚焦跨层回归结果：

```text
male provenance / whole-chain / pose-map / V2 failure / V3 core / reviewer
31 passed in 102.69 s
```

Codex 配置复核补充：`codex doctor` 明确显示 `Configuration: config loaded`、
`config.toml parse ok`，且本轮已再次实际启动 `luna_worker`。doctor 总退出码仍为 1，原因是
当前非交互 `TERM=dumb`、memory DB 只读/不可打开和受限网络 reachability，不是
`luna-worker.toml` 格式错误；agent 配置文件仍是空 diff。

按用户要求清理了未被当前代码/测试引用的 pre-V8 旧输出：`v233-v253`、`v29e`、旧
`audit_*`、`eval_*`、`reference_762*` 和 `preview_142*`，约释放 0.4 GB，删除不可恢复。
保留 `v7_candidates` 冻结 oracle、`v7_source_bake_001`、`canonical_cache`、
`cache_v7_final_bind`、`latest_asset/latest_canonical` 以及所有 V8/V3 回归证据。

当前结论不是 `accepted_for_user_genesis_review`。骨、bind、tube 的重复 correction 根因已
修正，但严格逐骨 containment 和带 pose 视觉仍失败，状态固定为 `rejected_for_redesign`；
不更新 `trusted/latest`，不启动 vessel reroute、nerve repair 或软组织第二阶段。

# 解剖 Retarget 现状（2026-08-13 bakeoff）

详细日志仍在 [`MD/todo_ana.md`](todo_ana.md)（§0.0–§0.13）。本文回答五件事：**做过什么 / 现在哪版错在哪里 / 怎么改 / 回不回退 / 绝对穿出门（§5）**。

bakeoff 与图审部分（§1–§4）为只读：没改 solver、没删旧产物、没升 `trusted/latest`、没执行回退。§5 在其后按用户指示新增了**纯度量代码**（不碰几何、rest、pose map、权重、拓扑）。

- 分支：`codex/stage1-male-retarget-v4` @ `55e261d`（提交信息是控制器；解剖 V10/V11 代码实际落在 `227cfeb`）。
- 对照图根：`outputs/anatomy_retarget/bakeoff_20260813/`（约 2.8 GB；磁盘现约 24 GB / 96%）。
- 受试：仅 `213328`。全部 `publishable=false`。
- 图审：每候选一个 GROK 4.6；分歧 GROK 4.6 + Opus max 双复核。不采信候选自报 pass。

---

## 1. 我做了什么

时间线只记判决，数字细节见 `todo_ana.md`。

| 版本 | 提交 / 产物 | 做了什么 | 判决 |
|---|---|---|---|
| 142 基线 | `142ece5` | 冻结 235 controller / 14-slot LBS / 17 tube；离线联动 | 生产核；禁拆 |
| 负例 | `29e1072` | 细骨硬塞进 SMPL-X | **禁止重建** |
| Pack A 联动基线 | `31133af` | 保联动；Male T-pose 可看；不做 rest-fit | 联动基线，**不是** pose 解 |
| V4 | quarantine | CUDA 多姿态；膝尖刺/碎裂 | `rejected_for_redesign` |
| V5 | `chain_retarget_v5_node2_002` | `node2_004` rest + parent-local FK | 继承 rest/bind 合同 |
| V6 | `chain_retarget_v6_node2_001` | right-multiply；修手指 rebase | 手足仍 `copy_142`；Pack B 级残差 |
| **V7 权威** | `chain_retarget_v7_node2_001` | 股骨中心线方向 + 全组织出图 | **当前 pose 权威**；屈膝髌前仍红 |
| V8 | `chain_retarget_v8_node2_001` | `Femur_Rot` ±3% 轴向缩 | 膝仍出皮；衔接差 |
| V9 | seat+inside embed | 轴向缩骨换皮内 | **flex medial gap 18→64 mm**；用户否决 |
| V10 fk-only | `chain_retarget_v10_fk_only_001` | 关节锚定 FK；腕 rebase | `hand_L` 0.952→**0.001**；净亏损 |
| V10 hybrid | `chain_retarget_v10_hybrid_001` | 手/足含腕踝根冻成 identity-142 | 代码层止血；见 §2.3 出图缺口 |
| V11 | `chain_retarget_v11_anchored_001` | V7 mesh 保留；铰链原点回 `B_prefit` | 主链 vs hybrid 收回；相对 V7 左腕/肘仍红 |

本轮 bakeoff 新做的事：

1. Pack A：`render_stage1_baseline_compare_v1 --pack A --subjects 213328` → `bakeoff_20260813/pack_A_31133af/`（tpose + pose_213328，26 相机 × 4 层）。
2. Pack B：同 CLI `--pack B --node2-004-root …/chain_retarget_v7_node2_001` → `pack_B_v7/`。
3. C/D/E：`render_v10_vs_v7_slim_genesis_v1`，**不加** `--delete-full-after-slim`，保住 `bones_tubes` / `full_anatomy`。
4. 方向复核：盘内 OSSO/SKEL、联网 SMPLer-X/OSX/HIT/weight-inpainting、架构批判；GROK + Opus 双复核。

冻结路径（均可用，未动）：

```text
operator     outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8
calibration  outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1
oracle       outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/blender_link_oracle_v7.npz
smplx male   ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl
capture      smplx_outputs/20260713_213328/moment_0000/smplx_result.npz
```

---

## 2. 现在是哪版、错在哪里

### 2.1 一句话

**Pose 权威仍是 V7，而且现在有数字了。** 当前 HEAD 上的构图是 V11 rest + hybrid 终端冻结 + V10 主链 FK。§5 的绝对穿出门给出判决：**V7 是唯一落在「不比 Pack A 差 1 mm 以上」这条线内的候选（并非零回退，它在 `humerus_R` 差 +0.51 mm）；V11 在左前臂相对 Pack A 回退 8.7–15.8 mm，hybrid 回退 26.5–29.5 mm。** 所有版本的血管/脏器都没有炸开。共同未解：股骨 mesh 轴长 ~405 mm vs 髋–膝 ~373 mm（**超长约 32 mm**），冻 14-slot LBS + 只搬 bind **藏不住**。

失效位置：

```text
Blender 142 资产 (235 / 14-slot / 17 tube)
        │
        ├─ Pack A: 不做 rest-fit ──────────────────────────► posed 骨 vs SMPL-X 皮
        │
        └─ rest-fit (B_final = C @ B_prefit) → pose 合成 ──► posed 骨 vs SMPL-X 皮
                                                              穿出集中：腕/前臂 · 踝/足 · 屈膝髌前
```

### 2.2 版本 × 部位 × 判决（图审，不以数字门为准）

四问固定：手足皮内 / 屈膝铰链 / 腕踝穿出排序 / 血管脏器炸开。

| 候选 | 手足皮内 | 屈膝股胫/髌股铰链 | 穿出排序（posed） | 血管/脏器 | 总判决 |
|---|---|---|---|---|---|
| A `31133af` | **fail**（posed 指尖+足背大红；T-pose 足已红） | **pass**（髁坐平台，髌在滑车；红是皮合而非铰链炸） | 膝 ≫ 踝/足 > 腕/手 | **pass** | `accept_as_linkage_baseline` |
| B V7 | T-pose 手近 pass / 足 fail；posed **fail** | **pass**（无 V9 的 64 mm 缝） | 踝/足 > 膝 > 腕/手 | **pass** | `mixed_keep_with_residual`；**pose 权威** |
| C V10 fk-only | 见图 ≠ 历史崩溃，见 §2.3 | residual | （与 D 同图） | **pass** | 历史 `hand_L=0.001` 仍成立；**本轮图不是 fk-only 回放** |
| D V10 hybrid | 腕/尺桡骨块红；指未塌 | residual（髌略浮，屈膝顶仍大红） | 膝 > 腕 > 踝（GROK） | **pass** | 代码合同是 identity-142；本轮图与 C **字节相同** |
| E V11 | 相对 V7 左腕 **fail**；相对 C/D 更好 | **pass**（铰链未开；髌前红小于 A） | 踝/足 > 腕 > 膝 | **pass** | 相对 V7：`reject_for_hand_regression`；相对 A：更好 |

双复核（GROK [dafc3392](dafc3392-90ee-41ff-a0a1-750ef0ce5d6d) + Opus [55283a70](55283a70-352b-4625-8345-cd0db0264a97)）：

- A vs B 排序「分歧」是假冲突：V7 把膝红大约砍半，踝几乎没动，踝因此「升」成最差点。全身穿出 **A 差于 B**。B 仍是 pose 权威。
- V10 fk-only 的 `0.001` 是**腕部穿出块**（腕 rebase ~37 mm），手指骨架仍在、未融成 stump。GROK 说「没塌」字面成立，但不是好事。
- V11 相对 V7：左腕新 L 形红块（V7 腕几乎干净）是真回归。相对 hybrid/fk-only 图，V11 腕红大约减半、膝更好。Opus 另标出 **左肘 lateral** 比手回归更大（V7 ~0 → V11 大块红）；V11 同时修了 V7 的右腕。
- 综合：V11 作为构图栈保留；**不要**用 hybrid 替换 V11，也**不要**整树/整目录回退到 `31133af`。

单候选图审：[A](d74fc3c4-7cdb-4cb4-9f37-f02c78ebdd1e) · [B](519811a7-8231-48bd-974e-b5df1fe9eefc) · [C](e949bff8-591c-494f-aa5a-8bf1dd09660b) · [D](ddcd1f58-6d1e-4719-906b-89753662b6f3) · [E](b0abc986-cf8f-413b-8eb3-16ca6b03f16c)

对照图（先看这些）：

```text
pack_A_31133af/pack_A/subject_213328/{tpose,pose_213328}/outside_heatmap/rgb/{whole_ap,left_knee_ap,left_hand_oblique,left_foot_oblique}.png
pack_B_v7/pack_B/subject_213328/{tpose,pose_213328}/… 同上
v10_fk_only/v10/outside_heatmap/rgb/…          # 候选目录名永远叫 v10/
v10_hybrid/v10/…                              # 与上一行 md5 相同
v11_anchored/v10/…                            # 这才是 V11；同目录 v7/ 是 V7 RM
```

### 2.3 本轮盖不到的缺口（不补，写清）

1. **C 与 D 图字节相同。** `bones.obj` / `outside.obj` / `left_hand_oblique.png` md5 一致。原因：`render_v10_vs_v7_slim_genesis_v1` 始终调用 HEAD 的 `pose_whole_chain_vertices_v10`（`terminal_policy=identity_142_hand_foot`），不会回放 fk-only 的 `subtree_rigid_rebase_hand_foot`。影子 JSON 仍能区分（fk-only `terminal_policy` vs hybrid `identity_142_hand_foot`），但 Genesis 图不能。历史手崩只能信当时的 `hand_L=0.001` 与门改，不能信本轮 C 图。
2. V10/V11 渲染器只有 `pose_213328`，**无 T-pose**。
3. `render_amass_bedlam_matrix_genesis_v1` 只吃 V1 格式 → **V11 跨体型/跨动作泛化无法出图**。这正是目标能力，下一轮需要通用 `--shadow` 渲染入口。
4. 数字门是相对 V7 的 `delta >= -0.02`；`max_candidate_outside_m` 算了却**不门**。所以 V8–V11 都能自报 pass，图仍红。历史教训：数字 ACCEPT ≠ 过关。

### 2.4 架构上真正错在哪（不是又一次 FK 写错）

- **构图 bug 多数已修**：V5 丢 whole-chain、V6 手指 rebase、V10 腕 rebase、V11 铰链原点被 rest-fit 推离。再写一套 FK 是浪费。
- **几何上限**：冻权重 + 刚体段 `C_bone` 不能把 405 mm 骨塞进 373 mm 皮囊。V8/V9 用 `Femur_Rot` 各向同性/轴向尺度去啃这 32 mm，结果是细骨或铰链断开。
- **V11 潜在屈膝角误差**（Opus）：mesh 留在 V7、bind 原点回到 `B_prefit`。T-pose 恒等；屈膝约 `2·d·sin(θ/2)`，90° 时关节中心可滑 ~21 mm。当前测试 pose 上看膝比 A 好；深屈未测。
- **Ed 门弱**：unsigned nearest-vertex、每骨 32 点、**median** 10 mm → 髁尖刺看不见。

---

## 3. 怎么改

方向复核：[盘内 OSSO/SKEL](00b85f40-944a-4c26-990f-672a2ee48bf3) · [联网](817c6562-f3ac-4b69-8c23-f7cd3a1778be) · [架构](2dd7dda9-a9d9-45ec-a7b6-3004c51cd886) · 双复核 GROK [f65b1827](f65b1827-80a4-424a-a6c5-e0b82c034039) + Opus [3c1911b5](3c1911b5-33f2-4b93-abe8-fbb41a9d7a60)。

**香港系 SMPLer-X / SMPLest-X / OSX：不相关。** 它们是图像→SMPL-X（β, θ），不管皮内骨骼、不管血管拓扑。盘上没有这些仓。OSSO/SKEL 源码在 `ref_code_library`，**权重缺失**；可搬的是能量形式，不是骨网格。

**禁搬**：V4 多姿态、V8/V9 缩骨、`segment_similarity` 各向同性尺度、再 rebase 手到新腕、Pinocchio 重绑、OSSO/SKEL 骨网格、weight inpainting（重绑 ≠ inpaint）、29e1072 径向塞入。

构图栈：**保留 V11 + hybrid 终端冻结。** 不要把 V7 `vertices_final` 当最终几何。Hybrid 冻腕踝根是修 37 mm rebase 的合同，即使本轮 C/D 图分不开。

三条可执行路线（Opus 排序；GROK 把 cage 放第一，但同意绝对门目前测不到「大穿出」）：

### 路线 1 — 绝对穿出门 ✅ 已实现，见 §5

结论先行：**做完了，而且它立刻证明了 V11 在深屈下确实比现在以为的差。** 实现与数字在 §5。

### 路线 2 — V12 rest cage（真正啃 32 mm）

推广已有 `_pelvis_cage_v1`（`chain_rest_fit_v1.py`，**31133af 时尚不存在**）到股/胫/肱/尺桡 + 17 管，同一套冻 LBS。近端骨干锁 Dirichlet；远端关节帽沿骨轴平移超长量；内部调和。**位移必须投影到骨轴、径向分量清零**（否则就是 29e1072）。管随同一场走，血管相对骨的位置由构造保持。

**用户必须先明确接受「轴向约短 8%」（405→373）。** 若「不缩骨」= 轮廓毫米都不能动，这条非法，改走路线 3。

硬门：路线 1 绝对量必须变好；径向位移 ≤0.5 mm；屈膝 gap 不得重演 18→64 mm；T-pose bit-identity；管–骨相对距离 rest 漂移 ≤2 mm。禁止再动 `Femur_Rot` 尺度（那是 V8/V9）。

### 路线 3 — OSSO Ed/Ej/Ect 作能量（笼之后，或笼被拒时的退路）

Signed 冻结对 Ed（不是 unsigned nearest-vertex）；posed Ej 用现有 `joint_contact_v7` 球窝，弹簧在 `Knee_Rotate` / `Elbow_Rot`，保留 V11 `|B−A|` 帽；Ect rest 内收 5 mm。**不能**单靠刚体重放藏住 32 mm 悬出，只会改花在哪一截。铰链 gap 必须是优化器内硬约束，不能是罚项（V9 死因）。

GROK 另列路线 3'：骨锁定后 viscera 调和（Anatomy Transfer 第二段）。只在路线 2 之后。

建议下一轮顺序：**1（已完成）→ 2（需用户书面接受轴向缩短）→ 3。** 另：§5 把「上肢链 bind 退回 V7、保留 V11 膝/足/右腕」从建议升级为**数据支持的首选动作**——V11 唯一的绝对回退就在 `forearm_L`。

---

## 4. 回退建议（不执行）

**建议：不整体回退到 `31133af`；改为只退上肢链到 V7。**

用户门槛是「当前 posed 穿出差于 Pack A **且** 失 V7 接触」才路径限定回退。§5 的数字让这条更精确：

- V11 **确实**在 `pose_213712` 全身 max 上差于 Pack A（33.1 vs 27.2 mm，+5.8），来源是 `shank_R`；在 `pose_213328` 上则优于 A（23.2 vs 33.7）。
- 但 V11 接触相对 V7 未开铰链（历史合同仍过）→ **AND 不成立**，整体回退依然没有依据。
- 更重要的是：回退到 A 会把 `patella_L`（4.8→28.2 mm）、`shank_L`（10.9→33.7 mm）、`femur_L`（13.9→24.5 mm）一起还原成最差值。A 在膝上是全场最烂的。

回退会丢掉的东西：

- `git diff 31133af..HEAD -- src/projects/genesis_ue_sync/anatomy_retarget` = 54 files，**+19755 / −66**（V5–V11 影子代码、真门、hybrid 终端、V11 锚定、**骨盆 cage**）。
- 骨盆 cage 在 `chain_rest_fit_v1.py` 的 +964 里，标签 `bounded_pelvis_cage_and_seat_inside_embed_v9`。V12 要复用它。回退 = 拆掉下一刀的工具。
- `31133af` / `29e1072` 与控制器提交在同一条历史上。V10/V11 解剖代码在 `227cfeb`（信息写的是控制器）。**绝不能** `git reset --hard 31133af`，那会把 `rm75_control` 一起拖走。

若以后某候选同时差于 Pack A 且失 V7 接触，才用路径限定（仍不要现在跑）：

```bash
# 只动解剖树；不要 reset 整仓
git rm -r -- src/projects/genesis_ue_sync/anatomy_retarget
git checkout 31133afba2ced3f4de01df7328d487859c7f9b05 -- src/projects/genesis_ue_sync/anatomy_retarget
```

影响面：丢失 V6–V11 约 19.7k 行；磁盘上的 `v7_candidates` / `v8_candidates` / `v10_candidates` / `v11_candidates` / `bakeoff_20260813` **不会**自动删（也不该删）。`rm75_control` 不动。

负例 `29e1072` 不要回、不要重建。

---

## 5. 绝对穿出门（2026-08-13 晚，已实现）

修的是 §2.3 第 4 条：绝对穿出深度一直在算、一直被扔掉，所以 V8–V11 能全门通过而图仍红。

**新增（只读度量，不碰几何/rest/pose map/权重/拓扑）**

```text
src/projects/genesis_ue_sync/anatomy_retarget/absolute_poke_v12.py
src/projects/genesis_ue_sync/anatomy_retarget/cli/run_absolute_poke_table_v12.py
tests/test_anatomy_absolute_poke_v12.py                       # 5 passed
outputs/anatomy_retarget/bakeoff_20260813/absolute_poke_table_v12.{json,md}
```

方法：`chain_containment_v1._signed_distance`（igl winding number + 最近点）对全部骨顶点一次查询，按**共享 rest 几何**的 `_vertex_areas` 加权，按 `bone_mesh_group_v12` 分组（v10 分组的完备化，见 §5.5）。每组报 `max_outside_m` / `poke_p95_all_m` / `outside_p95_m` / `outside_area_fraction` / `area_weighted_outside_depth_m`。

**姿态：不用合成姿态。** 冻结捕获里本来就有深屈——`pose_213328` 左膝 **94.4°**，`pose_213712` 双肘 **124.4° / 132.9°**。Opus 要的 ≥90° 膝与 ≥120° 肘两个条件都由真实捕获覆盖，无需编造。

### 5.1 全身骨（max / 全顶点 p95 / 外露面积，mm）

姿态标注用**实测解剖角**，不是轴角模长。

| 姿态 | Pack A `31133af` | V7 | hybrid | V11 |
|---|---|---|---|---|
| tpose | 18.2 / 0.00 / 1.72% | **16.4 / 0.00 / 1.19%** | 16.4 / 0.00 / 1.19% | 16.4 / 0.00 / 1.19% |
| pose_213328（左膝 105.5°，右膝 13.5°） | 33.7 / 0.37 / 2.92% | 23.2 / 0.00 / 1.94% | 29.7 / 0.77 / 3.89% | **23.2 / 0.02 / 1.82%** |
| pose_213712（肘 144°/131°，膝 24°/29°） | **27.2 / 1.37 / 4.29%** | 27.2 / 1.05 / 3.79% | 32.7 / 2.18 / 5.03% | 33.1 / 1.26 / 4.21% |

### 5.2 关键骨组 max（mm），`pose_213328`

| 组 | Pack A | V7 | hybrid | V11 |
|---|---:|---:|---:|---:|
| `patella_L` | 28.2 | 10.2 | 18.2 | **4.8** |
| `shank_L` | 33.7 | 16.6 | 25.8 | **10.9** |
| `femur_L` | 24.5 | 18.9 | **13.9** | **13.9** |
| `forearm_L` | 0.2 | **0.0** | 29.7 | 16.1 |
| `foot_L` | 23.2 | 23.2 | 23.2 | 23.2 |
| `hand_L` | 7.1 | 7.1 | 7.1 | 7.1 |

### 5.3 门的两层设计与判决

绝对天花板不能直接当硬门：Pack A 连 T-pose 都是 18.2 mm，Opus 建议的 max ≤15 mm **没有任何版本够得着**，硬门会变成永远失败的门。所以分两层：

- **阻断层（硬门）**：任何骨组的 `max_outside_m` 不得比 **Pack A 同姿态同组** 高出 1 mm。Pack A 是最差的联动基线，V7 是唯一落在这条线内的候选，所以这条既有意义又够得着。参考缺失时 **fail closed**。聚合组 `ALL_BONES` 只测量不判决。
- **目标层（只报告）**：max ≤15 mm 且**全顶点** p95 ≤5 mm，记为 `target_met`，给 V12 当记分牌。

需要说清的一点：Pack A 是**未做 rest-fit 的 142**，不是任何人会发布的候选。所以这条线是「不得比不做 rest-fit 更差」的生产不回退线，**不是绝对质量地板**，而且它很不均匀——Pack A 的前臂只有 0.2 mm，屈膝却有 33 mm。V11 可以真把膝修好却仍因前臂失败，这正是它现在的样子。

判决：

| 候选 | `passed`（不差于 Pack A 超 1 mm） | `target_met` | 最差回退 |
|---|---|---|---|
| V7 | **true** | false | 无（最大 +0.51 mm，在容差内） |
| hybrid | false | false | `pose_213328/forearm_L` **+29.5**；`pose_213712/forearm_L` +26.5；`pose_213712/shank_R` +5.3 |
| V11 | false | false | `pose_213328/forearm_L` **+15.8**；`pose_213712/forearm_L` +8.7；`pose_213712/shank_R` +5.8 |

### 5.4 这把尺子查出的四件事（经独立对抗审查修正）

1. **手和脚在 V7/hybrid/V11 之间位相同，Pack A 相差约 1e-7 m。** `foot_L` 在 V7/hybrid/V11 下逐位一致；Pack A 走 CUDA float32 的 `skin_vertices`，V7/V11 走 CPU float64 的 pose map，所以差一个浮点量级，**不是**逐位相同。因果结论不变：copy-142 终端冻结意味着**从来没有任何一版动过手足 mesh**，而 `foot_L` 每个姿态都排进穿出前三。
2. **V11 的前臂回归被数字确认。** V7 `forearm_L` 基本为 0（213328 0.0 mm / 213712 4.2 mm），V11 是 16.1 / 15.0 mm。GROK、LUNA、Opus 三方图审都说过的「相对 V7 腕/前臂回归」，现在有量。
3. **V7 是唯一落在 1 mm 容差内的候选，但不是零回退。** 它在 `pose_213712/humerus_R` 比 Pack A 差 **+0.51 mm**，T-pose `shank_L` 差 +0.33 mm，`pose_213712` 全身 p95 也略差。说「从不回退」是错的，正确表述是 `max_regression_vs_pack_a <= 1 mm`。
4. **V11 在 `pose_213712` 全身 max 差于 Pack A**（33.1 vs 27.2 mm，来源 `shank_R`）。**但这不能当作深屈失效的证据**：94.4° / 124° 是 SMPL-X 轴角模长，不是解剖屈曲角。实测解剖角是 `pose_213328` 左膝 105.5°、右膝仅 13.5°；`pose_213712` 肘 144°/131°、膝只有 24°/29°。`shank_R` 那个尖峰出现在**约 29° 的膝**上。现有捕获里**没有深屈右膝、也没有膝肘同时深屈**，V11 的铰链滑移假说仍未被测到，需要合成深屈姿态（见 §6）。

**下一刀最省的动作是把上肢链 bind 退回 V7**：V11 的收益全在膝/髌/胫（`patella_L` 4.8 vs V7 10.2、`shank_L` 10.9 vs 16.6），代价全在 `forearm_L`。两者可分离。

### 5.5 量具本身的修正（对抗审查后）

独立[对抗审查](3d30a1bd-db28-49b1-9606-816ff8aa4084)复算了整张表（与 JSON 逐位一致），但指出四处量具缺陷，均已修：

| 缺陷 | 修法 |
|---|---|
| 面积权重取自 **posed** 顶点，随候选漂移（`shank_L` 外露面积 25.50% vs 25.36%） | 改用共享 rest 几何 `asset.vertices_rest`，与 `chain_containment_v1` 惯例一致 |
| `bone_mesh_group_v10` 的 spine 正则要求下划线，把 C3–C7 / T1–T12 / L1–L5 / 全部牙齿丢进 `other`（54 网格、28400 顶点＝骨顶点 30%） | 新增 `bone_mesh_group_v12`，补 `cervical` / `thoracic` / `lumbar` / `teeth`，**任何未分类网格直接抛错** |
| `ALL_BONES` 与最差组重复计入 failures | 聚合组只测量不判决（其 max 恒等于某个组的 max，聚合失败必然重复） |
| 目标层用只在外露顶点上取的 p95，单点外露时≈max | 目标层改判 `poke_p95_all_m`（全顶点口径）。V11 `pose_213712` 从 16.6 mm 变成 1.26 mm |

单测从 5 个增到 10 个，补了 0.5 mm 通过 / 1.1 mm 失败的阈值线、`groups=` 过滤、两种 p95 口径对比、rest 面积权重不随 posed 形变漂移、以及分组完备性。

---

## 6. 合成深屈姿态（2026-08-13 夜）

§5.4 第 4 条留下的空白：捕获里没有深屈右膝，也没有膝肘同时深屈，所以 V11 的铰链滑移假说一直没被真正测过。

新增 [`deep_flex_poses_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/deep_flex_poses_v12.py)：目标是**实测解剖角**（近端段与远端段的夹角），产生该角度的轴角模长由**二分求解**，不假设两者相等。轴方向从捕获里取——左膝取 `213328`、右膝取其镜像 `(x,-y,-z)`、双肘取 `213712`——捕获轴太短（<0.5 rad）时直接抛错，不猜。姿态建在 T-pose 上，只有被测铰链是活动自由度。构造完必须用 `J_regressor` 反测并断言达标。

三个新姿态（实测角）：

| 姿态 | knee_L | knee_R | elbow_L | elbow_R |
|---|---:|---:|---:|---:|
| `flex_knee_R_120` | 17.0 | **119.9** | 24.1 | 10.2 |
| `flex_knee_both_120` | **119.6** | **119.7** | 23.8 | 9.9 |
| `flex_knee_elbow_120` | **119.8** | **119.7** | **119.7** | **118.9** |

### 6.1 结果：深屈崩的是 hybrid，不是 V11

全身骨 max（mm）：

| 姿态 | Pack A | V7 | hybrid | V11 |
|---|---:|---:|---:|---:|
| `flex_knee_R_120` | 17.5 | 15.8 | 15.8 | **15.8** |
| `flex_knee_both_120` | 17.9 | 11.0 | 17.8 | **11.0** |
| `flex_knee_elbow_120` | 15.8 | 11.0 | 21.7 | **11.0** |

**Opus 预测 V11 会在 ≥90° 屈膝处因 `2·d·sin(θ/2)` 滑移而露馅。实测相反**：120° 双膝下 V11 与 V7 并列 11.0 mm，优于 Pack A 的 17.9 mm；真正在深屈下退化的是 **hybrid**（`shank_L` 相对 Pack A +7.4 / +7.9 mm）。V11 把铰链原点恢复到 `B_prefit` 正是修掉了这个。这个假说现在可以关闭了。

### 6.2 V11 的失败被收敛到两处

| 候选 | 失败数 | 明细（相对 Pack A 的 max 回退） |
|---|---:|---|
| V7 | **0** | — |
| hybrid | 6 | `forearm_L` ×3（+29.5 / +26.5 / +19.5）、`shank_L` ×2 深屈（+7.9 / +7.4）、`shank_R` +5.3 |
| V11 | **4** | `forearm_L` ×3（+15.8 / +8.7 / +7.7）、`shank_R` +5.8 |

V11 四处失败里三处是 `forearm_L`。剩下的 `shank_R`（`pose_213712`，33.1 vs Pack A 27.2）hybrid 也有（32.6），说明它来自 **V10 FK 构图**而非 V11 的 rest 锚定。

**这把 V12a 从"看图猜"变成了有据可依**：V11 的下肢在 120° 深屈下已验证，缺陷集中在上肢链，两者可分离。

---

## 7. 验收夹具 V12（2026-08-13 夜）

按 [`MD/todo_ana.md`](todo_ana.md) §9 盲审协议与 §11 停止条件补齐夹具。独立探索确认此前**没有任何 CLI 能从 V10/V11 产物出 2β×3pose 图**，`independent_genesis_review*` 目录、10 图 handoff、`review_decision.json` 全不存在，联动只在 rest 被门，§3.3 铰链轴垂距 / posed 髋球窝 / 踝榫接触**均无硬门**。

### 7.1 新增

| 文件 | 作用 |
|---|---|
| [`deep_flex_poses_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/deep_flex_poses_v12.py) | 按实测解剖角二分求解的深屈姿态 + 关节 sweep（见 §6） |
| [`joint_plausibility_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/joint_plausibility_v12.py) | 髋球窝、§3.3 station→铰链轴垂距、踝榫接触 |
| [`linkage_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/linkage_v12.py) | posed 管–骨相对偏移守恒 + 管拓扑摘要/计数 |
| [`cli/render_acceptance_pack_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/render_acceptance_pack_v12.py) | 吃 V1/V10/V11 出 2β×6pose×4层，相机含新增 `whole_lateral` / `feet_top` / `pelvis_context` |
| [`cli/write_independent_review_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/write_independent_review_v12.py) | §9.4 目录 + 10 图 handoff + `review_decision.json`，缺图即 `needs_rerender` |
| [`cli/run_acceptance_gates_v12.py`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/run_acceptance_gates_v12.py) | 三族门一次跑完 2β×6pose×N候选 |

相机全部由 SMPL-X skin + 冻结 validation frame 产生（§9.2），`candidate_camera_read=false`，写 `camera_manifest` SHA。所有测量走冻结 validation 域反推，不读候选自报 pivot/pass（§7.3）。单测 28 个。

`chain_retarget_v10_hybrid_002` 补齐了 `subject_213712`，跨体型对照现在有料。

### 7.2 门的口径（两处被数据推翻的假设）

- **§3.3/Node-1 的髋 2 mm 不能当 posed 硬门。** 那个数字来自 fit 分区的源模板；在 validation 分区 + 实体化 beta 上重测，**Pack A 自己 T-pose 就是 3.08 / 2.66 mm**。改为报告层 target，阻断层用相对 Pack A 不回退。
- **管–骨偏移不能对零。** 最初用"最近骨顶点"配对，Pack A 自己就漂 52.9 mm（中位数 0）——跨关节的血管与最近骨顶点属于不同控制器，屈膝时距离本就该变。改成同主控制器配对后 Pack A 的 T-pose 归零，但 posed 仍漂 22 mm（主控制器相同 ≠ 权重向量相同）。最终口径：**相对 Pack A（即冻结 142 授权的联动）不回退**。

### 7.3 基线扫描（2β × 6 姿态 × 4 候选，100 秒）

`outputs/anatomy_retarget/bakeoff_20260813/acceptance_gates_v12_baseline.json`

| 候选 | 穿出 | 关节合理性 | 联动 | 合计 |
|---|---|---|---|---|
| Pack A | pass（参考） | pass（参考） | pass（参考） | — |
| V7 | **pass** | 13–15 失败 | 5 失败 | fail |
| hybrid | 6 失败 | 17–18 失败 | 6 失败 | fail |
| V11 | 4 失败 | 13–14 失败 | 5 失败 | fail |

两个 beta 的数字几乎逐位一致 —— **失败是结构性的，不是个体差异**。

三个结论：

1. **左肘 station→轴垂距 +17.7 mm 出现在每个候选（含 V7）、每个姿态（含 T-pose）。** 轴是从 mesh 经冻结域反推的，而 V11 保留 V7 mesh，所以 V11 无法修它——这是 V7 rest-fit 的 **mesh 摆放**问题，对应 §0.13 的 `result[elbow]=humerus` 力臂。它与 `forearm_L` 穿出是同一件事的两个侧面。
2. **V7 有 `hip_seating_regressed`（4 处），V11 没有** —— V11 修好了 V7 的 posed 髋座合。
3. **踝榫 lift-off 只出现在 hybrid 与 V11**，V7 没有 —— 这是 V10 关节锚定 FK 的代价。

---

## 8. V12a 两条路都被门否决（2026-08-13 夜）

§7.3 把 V11 的缺陷收敛到左臂，于是试了两个变体。**两个都被门拒绝，而且拒绝理由互相矛盾——这正是有价值的部分。**

| 变体 | 做法 | 判决 |
|---|---|---|
| `leg_only` | 只恢复膝/髌铰链原点，臂链留在 V7 | 失败：`rest_anatomical_anchor_v11` + `lr_symmetry_v11`。V11 自己的解剖锚定门**要求**肘 bind 必须回到 prefit，留在 V7 就违反 `|B_final−A| ≤ |B_prefit−A|`，且左右不对称（`Elbow_Rot_L` 21.68 vs `_R` 0.95）超 5 mm |
| `carry_mesh=all` | 恢复全部铰链原点，mesh 跟着 `C_bone` 一起走 | 失败：`knee_pose_containment_v10` + `posed_body_containment_v10`。V7 的**腿部 mesh 摆放是承重的**，搬走就破坏髁–平台接触 |
| `carry_mesh=arm` | 只让臂链 mesh 跟着走（按 LBS 权重混合，避免接缝撕裂） | 失败：`posed_body_containment_v10`。膝的门过了，但 **`forearm_L` 反而更差**：delta 从 −0.166 恶化到 **−0.373** |

第三个变体的结果推翻了原假设。原以为前臂穿出是「bind 在 prefit、mesh 在 V7」的枢轴滑移造成的，把 mesh 搬过去就能修。实测相反：恢复后的肘原点距 V7 有 21.7 mm，把前臂 mesh 搬过去（最大位移 22.9 mm）**直接把它推出了皮囊**。

所以这是一个三方矛盾，且当前 V11 的折中是三者里最不坏的：

1. 解剖锚定门要求肘 bind 在 prefit
2. 皮肤容纳要求前臂 mesh 留在 V7
3. 一致性要求 bind 与 mesh 同处

结论：**`forearm_L` 不能靠重排 bind/mesh 解决**，需要真正的前臂 rest 重定位（带皮肤容纳目标的有界刚体拟合），不是本轮的 preset 切换。代码里 `HINGE_RESTORE_PRESETS` / `CARRY_MESH_PRESETS` 保留，供复现这三个否决。

---

## 9. 独立盲审（§7.3 口径，只看图不看 JSON）

用 §7.1 的验收图包（2β × 6 姿态 × 4 层 × 10 相机）做盲审，reviewer 被禁止读任何 manifest / 门报告 / pass flag。

| 候选 | 判决 | 阻断项 |
|---|---|---|
| [V7](f67a358e-9e91-4b93-85f0-6b2c473e3703) | `direction_accepted` | 两个 beta 上足部仍大面积外露；`pose_213328` 髌骨仍尖出 |
| [V11](ec9c92ea-87bd-422e-b778-0c3820f70d4e) | `direction_accepted` | 两个 beta 上足部仍大面积外露；屈肘时左前臂比 Pack A 更差 |

两位都确认的正面结论：

- **两个体型都成立**，不是只对 213328 有效。
- **120° 深屈膝下股–胫–髌铰链连续、髌骨不尖出**，且穿出优于 Pack A（两个 beta 都是）。
- **血管/神经/脏器连续跟骨**，无撕裂或炸开；股骨头在髋臼内（深屈也在）。

两位都给出同一个阻断项：**足部**。这与 §5.4 第 1 条的数字完全吻合——手足在所有版本里数字一致，因为**从来没有任何一版动过它们**。

V11 盲审另外指出数字门看不见的两点：屈膝腘窝处皮肤出现碎三角；`flex_knee_both_120` 侧位可见肩胛板凸出背部。

---

## 10. V12b 可行性已验证：足部是摆错，不是太大

盲审的唯一阻断项是足部。先测能不能用**纯刚体重定位**（用户已批准的口径：腕/踝根下刚体 + 同一冻结 LBS、禁缩骨）修好，还是足骨本身就塞不进去。

在 `subject_213328` 的 rest 上，对足骨簇做无缩放的刚体拟合（Powell，目标为外露量平方和）：

| 骨组 | 现状 max 外露 | 最优刚体后 | 外露顶点 | 所需位移 |
|---|---:|---:|---|---|
| `foot_L` | 16.37 mm | **2.38 mm** | 815 → **93** | 平移 6.0 mm + 旋转 **11.0°** |
| `hand_L` | 0.62 mm | 0.00 mm | 12 → 0 | 平移 0.5 mm + 旋转 1.1° |

**足部穿出的 85% 是姿态摆放错误，不是尺寸问题。** 142 的足相对 SMPL-X 足腔转了约 11°。这条路不需要碰骨几何、不需要缩放，完全在用户批准的红线内。

手在 rest 下本来就几乎不穿（0.62 mm）——手的问题只在 posed（`pose_213328` 7.1 mm），属于终端 FK 契约，不是 rest 摆放。

**下一步（未实现）**：把这个有界刚体重定位做进 rest fit 的终端簇，保留 `evaluate_terminal_pose_regression_v10` 的绝对 142 硬门防 V10 式手崩，并新增足部对应门。注意终端门当前以**绝对 142** 为基线，而重定位后的足按设计就会偏离 142，所以那道门的口径必须同步改成「不差于 142 的皮内程度」而不是「等于 142 的位置」。

---

## 附录：本轮产物与红线

```text
outputs/anatomy_retarget/bakeoff_20260813/
  pack_A_31133af/                     564M  31133af 联动基线（tpose+pose，四层）
  pack_B_v7/                          567M  V7 权威（同上）
  v10_fk_only/                        570M  标签 C；图 = hybrid 回放
  v10_hybrid/                         569M  标签 D；与 C md5 相同
  v11_anchored/                       571M  标签 E；v10/ 才是 V11
  absolute_poke_table_v12.{json,md}         §5 穿出度量 + 门判决
  acceptance_gates_v12_baseline.json        §7.3 三族门 × 2β × 6 姿态 × 4 候选

outputs/anatomy_retarget/accept_20260813/   §9 盲审用图包（每个 ~48M）
  pack_a/ v7/ v11/                          slim/subject_<β>/<pose>/<layer>/<camera>.png

outputs/anatomy_retarget/v10_candidates/chain_retarget_v10_hybrid_002/  两 beta 齐全
```

复现 §5（约 25 秒）：

```bash
QD_OFFLINE_CACHE_FILE_PATH=/tmp/anatomy_qd_cache PYTHONPATH=.:src python \
  -m projects.genesis_ue_sync.anatomy_retarget.cli.run_absolute_poke_table_v12 \
  --operator outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8 \
  --calibration outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1 \
  --oracle outputs/anatomy_retarget/v7_candidates/blender_link_oracle_v7_full_001/blender_link_oracle_v7.npz \
  --smplx-model ref_code_library/EasyMocap/data/smplx/smplx/SMPLX_MALE.pkl \
  --capture-213328 smplx_outputs/20260713_213328/moment_0000/smplx_result.npz \
  --capture-213712 smplx_outputs/20260713_213712/moment_0000/smplx_result.npz \
  --v7-shadow outputs/anatomy_retarget/v8_candidates/chain_retarget_v7_node2_001 \
  --hybrid-shadow outputs/anatomy_retarget/v10_candidates/chain_retarget_v10_hybrid_001 \
  --v11-shadow outputs/anatomy_retarget/v11_candidates/chain_retarget_v11_anchored_001 \
  --output outputs/anatomy_retarget/bakeoff_20260813/absolute_poke_table_v12.json
```

红线（仍然有效）：不动 `rm75_control`；不删既有产物；候选一律 `publishable=false`；不采信自报 pass；不用 Composer。§5 只加度量，不改几何。

---

## 11. 下一步怎么继续改 retarget（2026-08-14）

上轮 V12b 被中断，产物目录是空的。已经量过的事实决定顺序，不再猜。

### 11.1 还没过的东西，按阻断排序

| 优先级 | 缺陷 | 证据 | 允许的修法 | 禁止 |
|---|---|---|---|---|
| P0 | 足部大面积外露（两 beta、所有版） | 盲审唯一共同阻断；`foot_L` 16.4 mm rest / 23.2 mm posed；刚体 6 mm+11° 可收到 2.4 mm | 腕/踝根下有界刚体重定位 + 同一冻结 LBS + 多姿态拟合 | 缩骨、改权重、腕 rebase、pose-time IK |
| P1 | `forearm_L` 相对 Pack A +8.7–15.8 mm | V11 四处绝对穿出失败里三处是它 | 与手足同一套有界刚体 rest 拟合（皮肤容纳目标），不是 V12a 的 bind/mesh 对调 | 上肢链退回 V7（锚定门否决）；`carry_mesh=arm`（穿出更差） |
| P2 | 左肘 station→轴垂距 +17.7 mm | 每个候选含 V7、每个姿态含 T-pose | 前臂簇 reseat 的附带量；单独报 | 再写一套 FK |
| P3 | 股骨 mesh 轴长超髋–膝约 32 mm | 冻 14-slot LBS + 只搬 bind 藏不住 | **等用户书面接受轴向约短 8%** 再开路线 2 cage | V8/V9 `Femur_Rot` 尺度、径向缩 |

### 11.2 本轮只做 P0（V12b），做完再开 P1

构图栈不动：V11 rest + hybrid 终端契约 + V10 主链 FK。V12b 只在 rest 上给手/足簇加一个有界刚体 `T`，pose 用右乘 `G_src @ inv(B_src) @ B_tgt` 让 `T` 跟着肢转。

已经踩过、不再重复的坑：

1. mesh 按 LBS 混合、bind 整段搬 → posed 足飞 194 mm。
2. 硬簇赋值 → 胫骨下端绑在踝上的顶点 bind 变了 mesh 没变 → 仍飞 ~190 mm。
3. 只按 rest 拟合 → rest 赢 11 mm，`pose_213328` 左足 23.2→26.4 mm。
4. 左乘 `T` → 世界系偏移，屈膝时不跟着腿转。

当前代码：施加走 `_weighted_rest_correction`；求解的目标也走同一条 LBS（不再按整段 `T` 计分）；拟合 rest + 两个冻结捕获；踝/腕原点不得走出 `B_prefit` 的解剖球。

**2026-08-14 诊断（213328，mean-SSE 多姿态，17 min）——这条目标函数否决了自己：**

| 簇 | rest | tpose | pose_213328 | pose_213712 |
|---|---:|---:|---:|---:|
| `foot_L` | 16.4→14.9 | 16.4→14.9 | **23.2→12.5** | 16.8→**21.1** |
| `foot_R` | 5.6→**11.9** | 5.6→**11.9** | 12.6→9.1 | **24.5→13.3** |
| `hand_L` | 0.6→0.0 | 0.6→0.0 | 7.1→6.0 | 0.8→1.3 |
| `hand_R` | 0.5→1.3 | 0.5→1.3 | 3.5→2.4 | 5.4→2.2 |
| `shank_L` / `forearm_L` | 不变 | 不变 | 不变 | 不变 |

T-pose 恒等仍过，胫/前臂没有被带坏。但一个刚体 `T` 坐不住所有足腔：mean-SSE 用 T-pose 右足 +6.3 mm、`pose_213712` 左足 +4.3 mm 去买屈足的赢。§11 禁止手足反向优化，所以**不能建这个候选**。

下一刀目标改成 **minimax（压最差帧的 max）+ 单帧回退 ≤1 mm**。T=0 永远可行；若最优解停在恒等，说明刚体 rest reseat 到了上限，再开 P1，不要硬建。

**minimax + 采样硬墙的第一跑否决了自己（手炸了）：** 旋转上界写成了每轴 ±15°，合模长可以到 26°。800 个采样点上看手没事，整簇指尖 0.6→62 mm。足部方向对（`foot_L` 三帧都不回退，`pose_213328` 23.2→15.3），但 `foot_R` T-pose +1.93 mm 仍超 1 mm，因为墙看的是采样 max 不是整簇 max。

已补：旋转按轴角模长封顶 15°；求解后对**整簇 × 全部拟合帧**做验收，任一帧回退 >1 mm 或代价掉进 1e6 墙，该簇退回恒等。

**整簇否决后的 213328 诊断（合法，可建）：**

| 簇 | 判决 | 效果 |
|---|---|---|
| 双手 | 否决，退回恒等 | 不再炸指尖 |
| `Ankle_Rot_R` | 否决，退回恒等 | 刚体 `T` 修不了右足而不伤 T-pose |
| `Ankle_Rot_L` | **接受** 5.9 mm + 5.0° | `pose_213328` 左足 **23.2→14.3**；`pose_213712` 16.8→15.8；T-pose 16.4→16.8（+0.4，墙内） |

全身 `pose_213328` 23.2→16.1（最差点就是这只左足）。手/右足/胫/前臂逐位不变。这是第一刀**不回退的**刚体收益；右足 24.5 mm 和 rest 16 mm 左足是刚体上限，留给 P1/笼。

**两 beta shadow 被 T-pose 面积门挡住。** 捕获姿态上 `terminal_pose_regression_v10` / `posed_body_containment_v10` 都过；CLI 还门 T-pose。左足 max 只 +0.4 mm，但一批跖骨/趾骨皮内面积掉了 8–15%，均值超过 2%。已补：验收同时看面积，超了就对 `T` 做二分缩放，找到仍过门的最大步长。

**塌缩验收后 213328 两门三帧全过（可建）：** 左踝 7.4 mm + 4.5°；右踝 ≈ 恒等；手 0.6–0.7 mm。`pose_213712` 上曾有单骨 0.9→0.5 塌缩，现已挡掉。

### 11.3 验收环（过一环才进下一环）

1. 单测：混合权重顶点按 LBS 分数移动；恒等 reseat 是空操作；求解不走出解剖球。
2. 诊断（只 213328，不出图）：多姿态 reseat 后 `foot_*` 在 tpose / 两捕获上不得差于 V11，争取 rest ≤6 mm、posed ≤ V11。
3. 过了才建 `chain_retarget_v12b_reseat_001`（两 beta）。
4. `run_acceptance_gates_v12` 对照 Pack A。终端门已经按皮内面积判，不要求等于 142 位置。
5. `render_acceptance_pack_v12` + slim，磁盘 96%，不留完整树。
6. 独立盲审，判决必须是 `accepted_for_user_genesis_review` 才停。足部仍红就改拟合，不要开 cage。

§11 停止条件里「foot 不再大幅外露」是本轮唯一要关掉的视觉阻断。前臂和 32 mm 超长留下一轮。

### 11.4 V12b 已建，P0 未关（2026-08-14 15:45）

产物：`outputs/anatomy_retarget/v12_candidates/chain_retarget_v12b_reseat_001`  
`passed=true`（V11 阶梯）、`publishable=false`、`terminal_policy=reseated_142_hand_foot`。两 beta 的 `terminal_pose_regression` / `posed_body_containment` / 锚定 / 对称都过；旧 `containment_check` 仍否（对 142 位置，预期）。

独立进度审（explore，[V12b audit](565a3b31-5cab-45d0-94e2-062cb5af5599)）：**`not_ready`**。没有 §9.5 10 图 handoff，也没有 `independent_genesis_review*`。

`run_acceptance_gates_v12`（`acceptance_gates_v12_v12b.json`）v11 与 v12b 都 `passed=false`。相对 V11：

| 姿态 | foot_L | foot_R | 含义 |
|---|---:|---:|---|
| pose_213328 | 23.2→19.9 | 12.6→8.1 | 捕获赢 |
| pose_213712 | 16.8→13.2 | 24.5→16.8 | 捕获赢 |
| tpose | 16.4→16.1 | 5.6→6.6 | 右足贴 1 mm 墙 |
| flex_knee_*_120 | −0.3 mm | **+7.2–7.5 mm** | **新回退** |

右踝建成的是 **8.8 mm + 1.5°**，不是诊断里「否决退回恒等」。拟合只吃了 rest + 两捕获，深屈不在墙里，所以捕获毫米是用深屈右足买的。`forearm_L` 仍 +8.7–15.8 mm（与 V11 相同）。左肘 +17.7 mm 未动。

`acceptance_pack_v12b_001` 已齐（2β×6pose，480 slim）。10 图 handoff：`chain_retarget_v12b_reseat_001/independent_genesis_review_v12/handoff/`，`decision` 先是 `ready_for_independent_review`。

独立盲审（[V12b visual](29af2ef8-3527-48bd-8e08-6c0e885d0dee)）：**`needs_fix`**。膝铰链过、联动未炸；手足仍大面积外露（P0 未关）；深屈右足比捕获更差。已改 `review_decision.json`。

下一刀：深屈三态进 reseat minimax，建 `chain_retarget_v12b_reseat_002`。

### 11.5 V12c 拆足簇 + 面积（2026-08-14）

`Ankle_Rot` 只搬后足，`Arch_Rot` 搬跖+趾，施加 `T_arch @ T_a`。目标改成 minimax 出皮面积；1 mm 墙对着**原始未动**，踝拟合还墙继承后的前足，避免右踝 9° 把趾甩到 26 mm。

213328 诊断（rest + 两捕获 + 深屈三态）：

- `Ankle_Rot_R` 1.2 mm + 1.3°（不再 8 mm+9°）；深屈后足 5.9→6.3 mm，不再 +7 mm
- `Arch_Rot_L` 3.8 mm + 1.6°；`pose_213328` 前足 23.1→20.4，`pose_213712` 16.9→14.6
- `Arch_Rot_R` ≈ 恒等（任何 T 都会撞墙）
- 左后足 rest 仍 ~16 mm：刚体拆簇也坐不进那个跟/距骨尖

`chain_retarget_v12c_arch_001` 已拼好（`publishable=false`）。深屈 `foot_R` 相对 V11 只 +0.35 mm（V12b 的 +7 mm 已关）。捕获足仍 13–25 mm，P0 未关。sweep 网格已删。

### 11.6 V12d 前臂刚体 + V10 FK 评分（2026-08-15）

`forearm_L` 的 16 mm 在桡尺骨上（`Forearm_Bone`=尺骨，`Forearm_Twist`=桡骨），腕 reseat 动不到。已把同一套有界刚体接到这两根轴上，手跟随 `T_w @ T_ft @ T_fb`。

关键修正：终端源仿射会把 `B_tgt` 约掉，看不见主链 FK 穿出。评分改成运行时那套关节锚定 FK。诊断现在能看见 `pose_213328` 尺骨 12.9 mm / 桡骨 14.4 mm。

**刚体 T 仍压不下去**（1 mm 墙 + 多姿态 + 继承手）：尺骨 12.9→12.9，桡骨 14.4→15.2。把 V7 前臂 bind 接回 V11（mesh 本就相同，左尺骨原点差 23 mm）在 V10 FK 下变成 **16.1→17.7 mm**，与 V12a `carry_mesh=arm` 同类。V7 的 0 mm 前臂是旧右乘构图的数字，搬不到 V10。

未建两 beta 影子（没有可买的毫米，不堆产物）。bind+mesh 刚体到了上限。

### 11.7 V12e 前臂只搬网格、冻 bind（2026-08-15）

根因：V11 肘 bind 在 prefit，尺桡 mesh 在 V7，相差 ~15–23 mm。V10 绕 bind 转，rest 已贴皮（内余 38 mm），屈肘把骨干甩出皮囊。搬 bind（V12a / V7 graft）和 bind+mesh T（V12d）都会改旋转中心，压不住。

V12e：`B_final` 不动，只对尺骨/桡骨 rest 网格做有界刚体 T（管随同一 T）。213328 诊断：

| 姿态 | `forearm_L` V11 | V12e | 面积 |
|---|---:|---:|---:|
| rest | 0.00 | 0.00 | 0→0 |
| pose_213328 | 16.07 | **0.00** | 16.2%→0 |
| pose_213712 | 14.97 | **0.00** | 14.4%→0 |

四根前臂 bind 原点位移 0.000 mm。手/膝/胫未借毫米（`humerus_L`/`shank_L`/`patella_L` 逐位相同）。足仍是 V12c 量级（P0 未关）。

`chain_retarget_v12e_forearm_mesh_001` 已建（`publishable=false`，两受试 `passed=true`）。`forearm_L` 相对 hybrid 面积 +0.35。

绝对穿出门（vs Pack A）：

| | V11 / V12c | V12e |
|---|---|---|
| 213328 `forearm_L` ×3 | +15.8 / +8.7 / +7.7 mm | **关**（0.00） |
| 213712 `forearm_L` ×3 | +15.9 / +8.8 / +7.7 mm | 只剩 pose_213328 **+1.37 mm**（16→1.5） |
| `shank_R` pose_213712 | +5.8 / +4.8 | 同 V11（V10 FK，未动） |
| `hand_L` 深屈肘 | — | 刚过墙 **+1.0006 mm** |

P2 左肘 station→轴：T-pose 回归 17.7→**12.0 / 9.4 mm**（mesh 靠近 bind 轴）。联动门 tpose 管–骨漂移 9.5→22 mm（有的管没吃到同一 T）。足仍是 P0 刚体上限。仍不开股骨笼。

独立盲审 [V12e visual](4f8b92ad-1a8e-4a16-a282-adc7abda62b2)：`needs_fix`；`p1_forearm_closed=true`；`p0_feet_closed=false`；膝铰链连续；`bones_tubes` 前臂血管看起来撕开/没跟骨。下一刀：同一 T 跟上前臂管，不重开笼、不缩骨。

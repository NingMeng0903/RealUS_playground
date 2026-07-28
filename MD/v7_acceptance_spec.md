# V7 解剖资产验收规范 (Acceptance Spec, schema_version = 7)

本文件是 V7 候选资产的唯一验收契约。独立盲审 agent 只依据本文件、下列源文件、
两条采集路径和候选目录做判断，不得使用任何“已经修了什么”的叙述、聊天上下文、
或候选自身报告里的 `pass` / `passed` 字段作为通过依据。

本文件描述**如何判定**，不描述当前状态。任何“已通过”的结论必须由盲审自己重算数组得出。

---

## 1. 受审对象

| 角色 | 路径 |
|---|---|
| SourceOperatorV7（一次离线 bake，pose/beta 无关） | `outputs/anatomy_retarget/v7_candidates/<candidate>/source_operator_v7.npz` |
| SubjectAssetV7（每个 beta 一个，pose 无关） | `outputs/anatomy_retarget/v7_candidates/<candidate>/subject_operator_<subject>.npz` |
| 冻结固定材料域 | `outputs/anatomy_retarget/v7_candidates/joint_rebuild_001/fixed_joint_domains_v7.json` |
| 冻结髌骨规范 oracle | `outputs/anatomy_retarget/v7_candidates/<candidate>/patella_oracle_v7.npz` |
| 验收矩阵报告 | `outputs/anatomy_retarget/v7_candidates/<candidate>/acceptance_matrix_v7.json` |
| 证据包 | `outputs/anatomy_retarget/v7_candidates/<candidate>/evidence/` |

运行入口只允许：

```
python -m projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v7 materialize-beta ...
python -m projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v7 apply-pose ...
python -m projects.genesis_ue_sync.anatomy_retarget.cli.run_anatomy_v7 diagnose-matrix ...
```

`materialize-beta` 与 `apply-pose` 运行时**不得**导入 Blender、读取 `.blend`、
或使用 pose cache。盲审必须在隐藏 Blender 与 `.blend` 的环境下复现一次。

---

## 2. 可继承的真值与禁止的真值

### 2.1 允许作为先验（inheritable priors）

- V71 隔离源导出（源 commit `15b6016`）：
  `outputs/anatomy_retarget/v7_source_bake_001/v71_operator_source_v6.npz`
  - 235 根骨骼 hierarchy、parent-local bind、pivot。
- V71 Blender Action 骨矩阵（271 帧）：
  `outputs/anatomy_retarget/v7_source_bake_001/blender_action_oracle_v7.npz`
  - 只允许使用 `bone_action_local` / `bone_action_global` / `bone_rest_local` /
    `bone_rest_global` 推导**联动响应斜率与枢轴/轴向**。
- 冻结固定材料域中的 vertex ID。
- Blender 稀疏蒙皮权重（authored driver_indices / driver_weights）。
- 两条采集的 SMPL-X 拟合结果（beta 与 pose 驱动输入）：
  - `smplx_outputs/20260713_213328/`
  - `smplx_outputs/20260713_213712/`

### 2.2 禁止作为几何真值（non-oracles）

1. **原始 Blender evaluated mesh 的髌骨深屈膝表面轨迹**
   （`mesh__Patella_L/R__vertices`）。它在约 90° 时把髌骨拉离滑车沟约
   97 mm（左）/ 44 mm（右），本身不是可接受接触状态。
   该数组只能用于计算 source penetration envelope 上界，不能当目标轨迹。
2. **旧 213712 refit 资产**。其髋臼在固定域恢复时需要局部最大约 43.5 mm 修正。
3. **候选自身生成的 spline / 轨迹**。任何“候选 vs 候选自己”的比较一律记为
   `available=false`，并按失败处理。
4. **`material_fit` 或任何 metadata 中写入的测量值**（例如
   `femoral_head_to_acetabulum_m`）。所有数值必须从最终顶点数组重算。

---

## 3. 证据规则（fail-closed）

R1. **固定域不可变**：所有关节探针只允许索引
`fixed_joint_domains_v7.json` 中已冻结的 vertex ID。禁止在拟合/refit 之后重新
选择最近点、重新拟合 landmark、或扩大/缩小域。
盲审必须先校验 `topology_digest` 与候选 `faces` 一致。

R2. **从最终表面重算**：髋、膝、髌股、刚性、血管全部从
`apply-pose` 输出的最终顶点（以及 T-pose 的 `vertices_rest`）重算。

R3. **禁止读取 pass 标记**：`build_report`、`correction_report`、
`joint_matrix_report` 中的任何布尔字段都不构成证据。

R4. **左右独立**：左右侧分别评估，不合并、不取平均。

R5. **结构独立**：Tibia、Fibula、Patella 必须作为独立结构评估，禁止合并成
“膝关节复合体”后取最有利值。

R6. **三门逻辑与**：`controller AND local_fk AND geometry`，再与
`vessel`、`compound` 门相与。任一 `available=false` 即失败。

R7. **oracle 必须来自磁盘**：髌股验收使用
`patella_oracle_v7.npz`，并校验其 `action_source_digest` 等于
`blender_action_oracle_v7.npz` 的 SHA-256、`topology_digest` 等于候选拓扑。

---

## 4. 硬门与阈值

阈值的可执行定义分别在：

- 关节：`JointContactThresholdsV7`（`src/projects/genesis_ue_sync/anatomy_retarget/joint_contact_v7.py`）
- 血管/神经：`VesselGateThresholdsV7`（`.../vessel_gates_v7.py`）
- 复合结构（肋骨/颅骨/脑/口腔）：`CompoundGateThresholdsV7`（`.../compound_gates_v7.py`）

以下为契约值；代码默认值与本表不一致时以**更严格者**为准，且必须在报告中列出实际使用值。

### 4.1 Controller 门（每姿态，左右独立）

从最终 posed 骨矩阵与最终表面重算，不读 metadata。

| 项 | 定义 | 限值 |
|---|---|---|
| `hip_{side}.translation_error_m` | 用髋固定域从最终表面拟合的髋臼球心，与股骨刚性 delta 作用于 bind 髋臼球心后的位置之差 | ≤ 0.001 m |
| `hip_{side}.direction_error_deg` | posed 股骨方向（股骨头球心 → 膝枢轴）与 SMPL-X posed 髋 → 膝方向的夹角 | ≤ 1.0° |
| `hip_{side}.axial_twist_deg` | 股骨绕自身长轴的残余扭转 | 只记录，不判定（见 §4.6-A） |
| `hip_{side}.rotation_error_deg` | 股骨刚性 delta 旋转 与 SMPL-X 髋 joint delta 旋转 的夹角 | 只记录，不判定（见 §4.6-A） |
| `knee_{side}.translation_error_m` | 膝枢轴在股骨局部帧中的漂移（posed 局部原点 vs bind 局部原点） | ≤ 0.001 m |
| `knee_{side}.rotation_error_deg` | 膝局部旋转轴 与 V71 Action 推导的铰链轴 的夹角 | ≤ 1.0° |

### 4.2 Local-FK 门（每姿态，6 条链，左右独立）

链：`{side}/Femur_Rot>Knee_Rotate`、`{side}/Knee_Rotate>Tibia_Bone`、
`{side}/Tibia_Bone>Patella_Rotate`。肘链见 4.5。

| 项 | 定义 | 限值 |
|---|---|---|
| `translation_error_m` | posed parent-local 平移 与 bind parent-local 平移之差（Tibia 允许的预烘焙 glide 上界即为该限值） | ≤ 0.001 m |
| `rotation_error_deg` | posed parent-local 旋转 与参考 parent-local 旋转的夹角。参考旋转 = bind 局部旋转 ∘ R(授权轴, 授权角)。授权轴与授权角只能来自 V71 Action（膝）或冻结 oracle（髌骨）；`bind_follow` 子骨的授权角为 0 | ≤ 1.0° |

补充记录（不单独判定但必须出现在报告）：`flexion_deg`、`authorized_angle_deg`、
`response_error_deg`、`off_axis_residual_deg`。这四项与 `available`、`reason`
必须逐链出现在 `items` 中，读者要能区分"不可用"与"已测且超限"。

`{side}/Femur_Rot>Knee_Rotate` 现记为 `available=false`（按 2.2.3 判失败）。
膝屈曲角由 leg IK 解出、不是从 drive 读入，V71 Action 只授权*响应律*（髌骨随
膝的斜率）和轴，并不能对任意 SMPL-X 姿态授权一个膝角。此前该链把候选自己测出的
on-axis 角当作"授权角"，参考旋转因此是被测量的函数，`rotation_error_deg` 退化为
off-axis 残差（运行时构造上就是 ~1e-6°），`response_error_deg` 更是硬编码 0.0。
`flexion_deg` 与 `off_axis_residual_deg` 仍照实记录，off-axis 扭转仍然可见。

### 4.3 Geometry 门（每姿态，左右独立）

| 项 | 限值 |
|---|---|
| 髋 `center_error_m`（最终股骨头球心 − 最终定半径髋臼球心） | ≤ 0.002 m |
| 髋 `center_drift_m` | ≤ 0.001 m |
| 髋 半径变化 | 相对 ≤ 2% 或绝对 ≤ 0.001 m |
| 髋 clearance 崩塌 `clearance_min_drop_m`（rest 最小间隙 − 最终最小间隙） | ≤ 0.001 m |
| 髋 抬离：最终 clearance `max_m` | ≤ rest `max_m` + 0.003 m |
| 髋 clearance median / q95 变化 | **仅记录，不判定**。非球股骨头在非球髋臼内刚性转动必然重分布 clearance；以该分布判定只能靠形变股骨满足（历史实现因此把股骨头顶点单独旋回 driver 姿态，sweep 中股骨边长比达 0.207–2.886）。同心度、半径、抬离与崩塌四项已封住脱位与形变。 |
| 膝 四个内外侧胫股 `min_m` | 0 ≤ gap ≤ 0.003 m |
| 膝 `gap_change_m` | ≤ 0.002 m |
| 股骨长度变化（头球心→髁中心） | ≤ 0.0005 m |
| 髌股 gap（髌骨域 → **整个** `{side}/femur` 冻结域，见 §4.6-B） | 0 ≤ gap ≤ 0.004 m |
| 髌股 `gap_drift_m` | ≤ 0.002 m |
| 髌骨轨迹 vs 冻结 oracle：`trajectory_rms_m` / `trajectory_max_m` / 方向误差 | ≤ 0.002 m / 0.003 m / 2.0° |
| Femur / Tibia / Patella 刚性边比 q01,q99 / min,max | ≥0.99, ≤1.01 / ≥0.98, ≤1.02 |
| 整体缩放 | 禁止；`whole_pelvis_scaled` 与任何 `scaled_structures` 必须为空/false，并由剖面图佐证 |

### 4.4 血管 / 神经门（每姿态）

| 项 | 限值 |
|---|---|
| `topology_digest` 与源模板一致 | 必须完全相等。参考摘要由调用方从 operator 的 pre-beta `template_asset` 算出并传入（`reference_faces_digest`）；未传入时该门记 `available=false` 判失败，禁止用候选自身的摘要充当参考 |
| 固定截面边长变化 | ≤ 5% |
| 中心线转折角新增量 `max_turn_increase_deg` | ≤ 5°。bin 由 rest 测地距离确定并原样用于 posed，故 rest/posed 逐样本配对相减；不比较两个分布的最大值（大 rest 折角会掩盖别处的真实折角） |
| 中心线 `q99_turn_increase_deg`（同样逐样本配对） | ≤ 3° |
| 中心线取心线方式 | 每个测地 bin 按图连通性分股，只取直径路径所在那一股；其余股各自成 branch 递归测量。分叉 bin 内各股仍通过交汇点相连，故分叉处样本的绝对折角不代表解剖，只有配对增量有效 |
| SMPL-X 体内比例 `inside_ratio` | ≥ 0.999 |
| 最大越界距离 `max_outside_m` | ≤ 0.005 m |
| 相对规范模板新增骨穿透 `added_penetration_m` | ≤ 0.001 m |

### 4.5 复合结构门

| 结构 | 项 | 限值 |
|---|---|---|
| 肘（优先） | 肱骨—尺骨、肱骨—桡骨固定域 gap | 0 ≤ gap ≤ 0.003 m |
| 肘 | `Shoulder_Rotate>Elbow_Rot`、`Elbow_Rot>Forearm_Bone`、`Forearm_Bone>Forearm_Twist` local-FK | 同 4.2 |
| 肋骨（逐根） | 两端连接距离相对 rest 的增量 | ≤ 0.002 m |
| 肋骨 | 刚性边比 | 同 4.3 |
| 颅骨 / 脑 | 脑顶点在颅骨内比例 / 最大越界 | 1.000 / ≤ 0.000 m |
| 口腔 / 舌头 | 源资产中是否存在同拓扑 Tongue mesh | 若不存在 → `tongue_present=false`，记为**发布阻断项**，禁止以“已保留”表述通过 |

肩、腕、踝：只在固定域诊断失败时才允许修改几何；诊断本身必须出现在报告。

肘链骨名按源 rig 实名：`Shoulder_Rotate_{L,R}` > `Elbow_Rot_{L,R}` >
`Forearm_Bone_{L,R}` > `Forearm_Twist_{L,R}`，分别承载 Humerus / Ulna / Radius
mesh。此前代码找的是 `Humerus_Rot` / `Ulna_Bone` / `Radius_Bone`——任何源 rig 里
都不存在，六条链全部报 "bone is absent"，而 `local_fk_arms` 又没有进入 cell 的
`failures` 与 `passed` 合取，等于测了又丢弃。现已并入判定：任一链
`available=false` 即该 cell 失败。

---

## 5. 验收矩阵

严格 2 beta × 3 pose，加一条合成扫掠：

| beta | pose |
|---|---|
| `213328` | `tpose`（零姿态）、`213328`（强屈膝，左膝约 94°）、`213712` |
| `213712` | `tpose`、`213328`、`213712` |

合成扫掠：`knee_sweep_0_120`，左右膝各自 0→120°，至少 13 个采样点，
用于髌股轨迹、膝 gap 与刚性门。

矩阵规则：

- 两个 beta 必须来自**同一个** SourceOperatorV7（`operator_digest` 相同）。
- 每个 cell 的 `reference_vertices` 使用该 beta 的 `vertices_rest`（T-pose），
  `final_vertices` 使用该 cell 的 `apply-pose` 输出。
- 213712 的结果**禁止**与旧 213712 refit 比较优劣。其正确性只能由：
  该 beta 的 SMPL-X 表面包含性、固定域几何门、刚性门、V71 局部联动门、剖面图共同判定。
- 任一 cell 失败 → 矩阵失败 → 不得发布。

---

## 4.6 测量定义修订（每条都附带被它替换掉的错误测量）

以下修订只改变**如何测量**，不放宽任何限值。每条都记录了修订前测到的错误数值，
盲审可以按这些数字复现旧口径并确认新口径更严或等严。

**A. 髋 controller：判定股骨方向，扭转只记录。**
股骨绕自身长轴的扭转对球窝髋关节是自由度，而膝的铰链轴方向由解剖决定。当 SMPL-X 拟合
把屈膝旋转放在与解剖铰链轴相差 53.3°（左）/ 85.4°（右）的轴上时，只有让股骨吸收这个差
（实测残余扭转 21°–54°）才能同时满足"绕解剖铰链屈膝"和"小腿落在皮肤内"。因此判定量是
`direction_error_deg`（限 1.0° 不变），`axial_twist_deg` 与
`rotation_error_deg` 必须出现在报告中但不判定。

**B. 髌股 gap 的目标域是整个股骨域，不是 rest 选出的滑车 patch。**
滑车 patch 是在 rest 状态选出的股骨参考面；生理上髌骨在深屈膝时必然滑离该 patch 而移到
髁上。要求对 patch 本身保持接触，只有"髌骨焊死在股骨上"才能满足。V71 源自身在其 Action
全程对**整个**股骨域维持约 1.4 mm。报告必须同时给出
`trochlea_patch_gap_min_m` / `trochlea_patch_gap_max_m` 作为对照。

**C. 每 cell 的髌骨轨迹：oracle 挂在候选自己的 posed 股骨上。**
per-cell 只有两帧（rest 与该 pose）。oracle 帧除髌骨外全部取候选自己的 posed 表面，
髌骨由 oracle 链在候选**posed 股骨**上求出。修订前把 oracle 挂在 bind 股骨、且滑车留在
rest，于是整条腿的运动（2–6 cm）和全局驱动平移（60 cm）都被记成髌骨误差。
多帧轨迹判定仍然只在 §5 的合成扫掠里做。

**D. 零运动时方向误差无定义。**
当候选与 oracle 的位移都小于 `patella_trajectory_rms_m` 时不存在可比较的方向，
`trajectory_direction_error_deg` 记 0 并置 `trajectory_direction_available=false`；
位置误差仍然判定。修订前该情形返回 `inf`，只会把零姿态判失败。

**E. 膝铰链枢轴由接触优化决定（离线一次）。**
只有铰链**轴向**继承 V71；枢轴位置是自由参数。股骨髁不是圆弧，穿过域质心的铰链会在屈膝
时抬起一侧间室：实测内侧间隙在 100° 时达 4.0 mm。在冻结域上、对 0–120° 授权范围求解垂直
于轴的两个自由度后，四个膝的最坏间隙降到 2.01–2.63 mm，最紧 0.87 mm。
`knee_pivot_optimization` 必须出现在 `articular_reconstruction` 报告中。

**F. 血管中心线用测地直径分箱 + 分支切分。**
按单一主轴分箱只对无分支管有效；本资产 17 个血管/神经组件中有分支树，分箱质心会在分支间
跳变。分箱一次性在 rest 上确定并复用到 posed（否则锯齿可以靠重新采样自己藏起来）。
修订后 `Lumbar_Nerves_L` 的 26.3° "锯齿"消失（0.001°），确认旧数值是分箱假象；
`Spinal_Cord`（+39.13°）与 `Sacral_Nerves_L`（+6.03°）仍然失败，是真实结果。

**G. SMPL-X 皮肤必须与该 cell 同 beta 同 pose。**
含在体内的判定要求皮肤与解剖在同一帧。用采集帧的皮肤去比零姿态的解剖，会把整具身体报成
离自己 0.991 m。皮肤由纯 numpy SMPL-X 前向（`smplx_body_surface_v7`，对采集拟合顶点复现
误差 0.367 mm；本机无 torch）按该 cell 的 beta 与 pose 生成，来源写进报告。

**H. 零姿态回归必须走完整运行时路径。**
`materialize-beta` 的 T-pose 往返检查原先只做骨骼蒙皮、跳过管状材料帧，因此漏掉了
运行时在零姿态把血管/神经拉回 operator rest 的 24.5 mm 突跳。往返检查现在与 `apply-pose`
走同一条调用，实测 1.2e-7 m。

---

## 6. 性能与确定性

在单进程隔离冷启动（无并行争用、无缓存命中、隐藏 Blender/`.blend`/pose cache）下测量：

| 项 | 限值 |
|---|---|
| `apply-pose` 冷启动（SubjectAsset 加载 + 39.5 万顶点求解） | ≤ 1.0 s |
| `materialize-beta` 冷启动 | 报告实测值；必须为隔离单进程数字，禁止使用并行或缓存命中值 |
| 同输入重复生成的顶点差 | ≤ 1e-6 m |

报告必须显式标注测量方式（isolated cold start），并给出 wall-clock 与进程隔离说明。

测量入口（每个进程只测一个 stage，并在 `bpy` 可导入时直接拒绝运行）：

```
python -m projects.genesis_ue_sync.anatomy_retarget.cli.run_v7_isolated_perf apply-pose --subject ... --pose ...
python -m projects.genesis_ue_sync.anatomy_retarget.cli.run_v7_isolated_perf materialize-beta --operator ... --betas-file ... --patella-oracle ...
```

确定性按**跨进程**判定：三次独立冷启动的 `vertex_checksum` 必须逐位相同，
而不是同一进程内重复调用。

---

## 7. 证据包

目录：`outputs/anatomy_retarget/v7_candidates/<candidate>/evidence/`

命名：`{operator8}_{beta}_{pose}_{view}.png`，其中 `operator8` 为
SourceOperatorV7 content digest 前 8 位。每张图必须有同名 `.json` sidecar，包含
`operator_digest`、`subject_digest`、`beta`、`pose_digest`、`asset_file_digest`、
`view`、生成命令。

必需视图：

1. `surface` 真实表面渲染（全身，前/侧）
2. `hip_section`、`knee_section`、`elbow_section` 解剖剖面
3. `hip_contact_heatmap` 股骨头—髋臼 contact/penetration 热图
4. `knee_condyle_heatmap` 内外侧股骨髁—平台热图
5. `patella_track` 髌骨滑车轨迹与轴线（含 oracle 对照曲线）
6. `vessel_centerline` 血管中心线曲率/锯齿图
7. `rib_connection` 肋骨两端连接图

---

## 8. 发布条件

全部满足才允许翻转 `publishable` 并更新 `latest`：

1. §5 矩阵全部 cell `passed=true`，且每个 cell 三门 + 血管门 + 复合门均 `available=true`。
2. `tongue_present=true`，或经全源搜索确认不存在并由用户显式接受该阻断项。
   现状：`asset.source_mesh_names`、`v71_operator_source_v6.npz`、`rig_inspect.json`
   三处全源搜索均无匹配 → `tongue_present=false`，等待用户显式接受。
2b. `skull_brain.publish_blocker=false`，或用户显式接受该源侧阻断项。
   现状：右侧大脑相对 `Upper_Skull` 外露 14.35 mm、`inside_ratio=0.9442`，
   在**beta 材质化之前**的 operator template 上完全相同（14.34 mm / 0.9442），
   `added_outside_m=0.000`。即这是源资产的作者侧事实，重定向未加剧，
   本流程不得为了过门去移动脑或颅骨。
3. §6 隔离性能与确定性达标。
4. §7 证据包完整且 sidecar 哈希与候选一致。
5. 独立盲审 agent 给出 `ACCEPT`。

任一项失败：保持 `publishable=false`，`latest` 不动，回到对应阶段迭代。
旧 V6 资产不得自动升级为 V7。

---

## 9. 盲审 agent 执行流程

盲审只接收：本文件、§2.1 源文件路径、两条采集路径、候选目录。**不接收**修改说明。

步骤：

1. 校验固定域 `topology_digest` 与候选 `faces`；校验 oracle 的
   `action_source_digest` 与 `blender_action_oracle_v7.npz` 实际 SHA-256。
2. 自行调用 `materialize-beta` / `apply-pose` 重新生成两个 beta × 三个 pose，
   在隐藏 Blender 与 `.blend` 的环境下确认可运行且确定性达标。
3. 自行用固定域重算 §4 全部数值，不复用候选报告中的数字。
4. 自行读取或重绘 §7 图像，确认剖面与热图与数值一致（例如：数值说髋臼未整体缩放，
   剖面必须支持该结论）。
5. 输出 `ACCEPT` 或 `REJECT` + 每条失败项的实测值、限值、复算命令。

若盲审无法独立复算某一项，该项按失败处理。

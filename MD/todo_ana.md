# 基于 `142ece5f` 的快速、拓扑保持解剖骨骼 Retarget 记录与执行计划

更新时间：2026-08-01

基线提交：`142ece5f0bc646978ae3e8c9add76deea71c26a2`

当前阶段：只解决骨骼 rest placement 和后续 parent-local pose mapping。血管/神经只验证
原有 Blender 权重联动与拓扑不变量，不做 reroute、投影、containment repair 或软组织
残差场。所有候选保持 `publishable=false`，不得更新 `trusted/latest`。

本文档是在 `MD/todo_ana.md` 被清空后重新建立的单一事实记录。它记录 142 之后本轮实际
做过的工作、失败方向、数值发现、不可再做的事情、工件/参考图路径、agent 监管规则和
下一阶段实施节点。

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

- 生产 retarget 核心已恢复为 142 内容，下列文件与
  `142ece5f0bc646978ae3e8c9add76deea71c26a2` 逐文件一致：
  - `anatomy_lbs.py`
  - `articular_fit_v8.py`
  - `bone_segment_diagnostics.py`
  - `leg_centerline_v810.py`
  - `operator_bake_v8.py`
  - `v8_artifacts.py`
  - `version_v8.py`
  - `tests/test_leg_centerline_v810.py`
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

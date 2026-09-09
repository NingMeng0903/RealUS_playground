# V15 bilateral collar 独立审查（2026-09-08）

本次只读审查针对两个已保存包：

- `outputs/anatomy_retarget/v15_bilateral_collar_213328_20260908_001/compiled`
- `outputs/anatomy_retarget/v15_bilateral_collar_213712_20260908_001/compiled`

没有修改 runtime/fit，也没有重新拟合。结论分成运行合同和解剖几何两层：**运行合同通过，解剖验收失败，不能发布。**

## 1. bilateral compile math

V15 的 collar response 本身做了明确、有限的改变：原 235 控制器保留，`Clavicle_Rot_L/R` 为 129/168，分别改用 SMPL-X J13/J14 的 `joint_local` 响应，`a=b=pivot`，只重烘焙这两行 coupling，其他 233 行保留。两个包均无 `pose_corrector.npz`，source operator runtime digest 均为 `17f5d4e0...d00972b`。原 faces、driver indices、14 槽 weights、235 层级和 source asset 均保持；neutral target rest 保持，J13/J14 probe 的完整 parent-local FK 回放成立。

有一个必须修正的兼容性 bug，位置在 [`compile_bilateral_collar_v15.py:49`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/compile_bilateral_collar_v15.py:49)：

```text
new_T = old_T · old_reference_Rᵀ · new_reference_R
```

这个式子只有在 `old_T` 的坐标基已经被认证为旧 motion/reference axes 时才成立。两个输入 V14 collar manifest 没有 `translation_transport`，加载后的 support report 因而回退为 `legacy_shape_reference_axes`。213328 上，107/129/168/171/175 行的保存 map 仍接近单位阵，而 `R_targetᵀR_reference` 的最大元素差约为 0.00570/0.00456/0.00594/0.00683/0.00521。V15 只是继承并乘了这个未认证的旧 map，不能把它写成坐标一致性已经修好。后续应显式重建 `T = R_targetᵀ J R_motion_reference`，或者旧基无法认证时确定性 fail-closed；补一个 provenance 字段不能代替数值重建。213712 的两个轴组接近，暂时把这个问题隐藏了。

[`bone_components_v15.py`](../src/projects/genesis_ue_sync/anatomy_retarget/bone_components_v15.py) 的 weighted-moment/SVD polar 实现通过了针对性代码测试，但它只是生成刚性骨边界 target。它没有接到 `CompiledAnatomyV14.apply_pose`、血管运输或固定软组织场，也不能消除已有骨相交，因此不能计入 production 解剖修复。

## 2. 已实际打开的 Genesis 图与可见事实

实际打开了两个体型的 T-pose、capture 和 AMASS walk 的 whole、right elbow、right wrist、left knee、left hip 代表图，并打开了 213328 walk 视频帧 `00000/00016/00032`。代表路径见机器报告 [`report.json`](../outputs/anatomy_retarget/v15_independent_recheck_20260908_002/report.json)。

可见事实如下：

- 两体型 T-pose，以及髋、膝比较格，candidate 与 V14 视觉上基本不变。
- AMASS walk 中 candidate 的右肩—肘—前臂—腕位置确实变化，红色血管随同一条链移动；213712 的 walk 图也能看到这一点。
- capture 图的右侧变化很小；前臂/腕仍靠近皮肤轮廓，不能从投影宣布清除穿出。
- 连续 walk 帧能看出时间上的跟随，但不能据渲染投影判断三维深度、骨—骨或管—骨三角面相交。

已有 21 帧 pair metrics 也没有通过：

| 体型/帧 | all bones 最大皮外（mm） | vessels 最大皮外（mm） |
|---|---:|---:|
| 213328 T-pose | 18.538 | 11.868 |
| 213328 自身 capture | 37.115 | 28.184 |
| 213328 互换 capture | 31.968 | 31.171 |
| 213328 walk 中点 | 25.500 | 18.291 |
| 213712 T-pose | 16.473 | 3.780 |
| 213712 自身 capture | 28.949 | 26.203 |
| 213712 互换 capture | 31.457 | 22.258 |
| 213712 walk 中点 | 18.299 | 8.294 |

两个 pair report 都是 `anatomical_passed=false`，每个评估帧都保留 `vessel_over_1mm_fail=true`。walk 中点 213712 的 vessels 从 V14 的 58.492 mm 降到 8.294 mm，说明本次右侧运动修正有效，但仍远未达到 1 mm 目标。

## 3. 未完成检查

1. 历史 `_001` translation map 的基底问题已在新 `_002` 路径重建并写入 provenance；新包仍需完成最终回放审查。
2. 刚性骨 boundary residual 尚未接入固定共享软组织/FEM 场，血管和软组织的生产运输尚未完成。
3. 尚未对 V15 做显式三维骨—骨、管—骨三角面穿入检查，特别是髋、膝、肘、腕。
4. 本轮只打开代表图，没有逐一打开两个体型全部生成的每个姿态/相机格。
5. 尚未证明任意新 beta 和广泛 theta；当前证据只是两个采集体型及有限存档/AMASS 帧，不能做 universal claim。

针对性测试已通过：`tests/test_collar_response_v15.py` 与 `tests/test_bone_components_v15.py` 共 **10 passed**。这只证明代码合同，不改变上述解剖失败结论。

## Worst-region foot follow-up

我实际打开了以下四张图：

- `outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/sitstand_sid4336_middle/comparison/left_foot_oblique.png`
- `outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/sitstand_sid4336_middle/smplx_skin/rgb/left_foot_oblique.png`
- `outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/capture_213712/comparison/right_foot_oblique.png`
- `outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/capture_213712/smplx_skin/rgb/right_foot_oblique.png`

这两处图审都显示脚部骨列及红色血管、黄色神经的长轴、位置与 SMPL-X 目标皮肤足部方向不一致，偏差贯穿踝、跖骨到趾端。这是足踝整链的几何问题，不能归结为肘部轻微出皮；该事实进一步确认当前候选不能通过解剖验收。图像是投影观察，仍不把它当作深度或三角面相交数值证明。

## Translation-map rebuild follow-up

我已只读复审 [`translation_rebuild_v15.py`](../src/projects/genesis_ue_sync/anatomy_retarget/translation_rebuild_v15.py) 和更新后的 [`compile_bilateral_collar_v15.py`](../src/projects/genesis_ue_sync/anatomy_retarget/cli/compile_bilateral_collar_v15.py)。原 `_001` 审查指出的旧 V14 translation basis 问题仍保留为历史问题；对新 `_002` 生成路径，当前实现已按正确顺序修复：

1. `_world_jacobians` 从保存的 map 解出世界 Jacobian `J`，即 `J = R_target · T · R_motionᵀ`，没有把旧 shape/reference map 直接当作 motion map。
2. 未标注 legacy 输入必须满足冻结 operator、calibration、`connected_left_arm_caps_v14`、一次 rest-field application、保存的 ArmMap 参数和 collar pivot 声明；未知 transport/method/shape、缺 calibration 或带旧 pose corrector 都确定性失败。
3. 新 compiler 保存 `T_new = R_new_targetᵀ · J · R_new_reference`，并写入 `translation_transport=motion_reference_axes` 及 rebuild provenance。

新 `_002` 报告的重建证据为：213328 的 rest/bind/reference 最大误差均为 `0`；213712 分别约为 `2.22e-16`、`2.78e-16`、`0`，均小于 `1e-7`。因此，对 translation-map 表示和基底转换本身，本轮未发现新的数学 bug；这不改变 `_001` 的解剖失败状态，也不代表新包已经通过骨—骨、管—骨或软组织验收。

## Translation `_002` Genesis 图审

我实际打开了新 translation 对照图：

- `outputs/anatomy_retarget/v15_translation_genesis_213328_20260908_001/capture_213712/comparison/left_elbow_lateral.png`
- `outputs/anatomy_retarget/v15_translation_genesis_213328_20260908_001/sitstand_sid4336_middle/comparison/left_foot_oblique.png`
- `outputs/anatomy_retarget/v15_translation_genesis_213712_20260908_001/sitstand_sid4336_middle/comparison/left_foot_oblique.png`

213328 的左肘新旧图基本相同，红色血管、黄色神经和骨端关系仍贴近皮肤边界，图上看不出足以宣称通过的改进。两个体型的坐站左足图中，candidate 与旧对照仍显示足踝骨列及红色血管、黄色神经沿斜向轴排列，和目标皮肤足部方向/位置不一致；translation 基底修复没有自动解决足踝整链几何。该图审确认 `_002` 仍不能称为解剖通过；这里仅记录渲染可见事实，不把投影当作三维穿入证明。
## V15/_002 组织运行与共享场有界审查

### 已确认的表示与运行事实

两个 `_002` source pack 都声明 331 个材料网格：`bone` 234、`organ` 69、`heart` 1、`vessel` 2、`nerve` 15、`connective_tissue` 10。原始 rig 的顶点、面和逐顶点 14 槽索引/权重覆盖完整资产（`394770` 个顶点、`782856` 个面）；这说明器官、心脏、血管和神经没有从基础 LBS 表示中漏掉，但只说明覆盖和联动来源完整。

源包另存了 18 个 `tube_coupling_v8.*` 字段，覆盖 55337 个管状材料顶点和 165659 条材料边。审计明确为 `strict_matrix_lbs_14slot_v8`，`runtime_collision=false`、`parallel_transport=false`、`cross_section_reconstruction=false`、`runtime_graph_solve=false`。因此它能认证冻结权重/拓扑运输，不能保证管径、截面、分叉连续性或管—骨无穿入。已有 `vessel_route_v8` 元数据仍记录皮外点和骨穿入残差，不能作为当前姿态场已经通过的证据。

V15 的 [`CompiledAnatomyV14.apply_pose`](../src/projects/genesis_ue_sync/anatomy_retarget/consistent_runtime_v14.py:404) 只求 source/target global transform 后执行 `_lbs`；它没有读取 source pack 的 tube 字段，也没有调用 `soft_follow`、器官体积修正或骨—皮约束。虽然 [`ResidentPoseEvaluatorV8`](../src/projects/genesis_ue_sync/anatomy_retarget/v8_artifacts.py:1000) 可在另一条 V8 API 中消费 tube pack，当前 V15 compiled `pose` 接口没有把这条路径接入。两个包的 `disable_soft_follow=true`，`soft_follow_*`、`soft_component_ids` 和 `source_mesh_follow_modes` 均为 `None`，所以器官/心脏也没有固定的区域体积跟随场；运行时实际是原始矩阵 LBS。

这留下三个明确缺口：关节附近矩阵 LBS 可剪切/丢体积，管状材料跨权重边界可能塌缩，所有材料均没有运行时皮肤内侧不等式或骨层约束。用户允许骨骼按图审轻微出皮，因此皮肤约束只能作用于软组织；不能用全皮肤 Dirichlet 把骨端推坏。

### 可复用接口和禁止原样复用项

- [`shape_volume._build_cage`](../src/projects/genesis_ue_sync/anatomy_retarget/shape_volume.py:82)、`_tet_stiffness`（141）、`_solve_harmonic_field`（426）和 `_field_jacobian_diagnostics`（339）可作为离线 ambient tet cage、FEM 平滑和正 Jacobian 检查的基础；[`_solve_harmonic_beta_basis`](../src/projects/genesis_ue_sync/anatomy_retarget/shape_volume.py:567) 可为体型编译生成固定 beta basis。
- [`source_skin_volume._barycentric_surface_map`](../src/projects/genesis_ue_sync/anatomy_retarget/source_skin_volume.py:358)、`_harmonic_step`（966）、`_incremental_harmonic_field`（1000）、`_jacobian_safe_harmonic_boundary_field`（1084）和 `_transport_sampled_material`（260）可复用固定材料坐标、离线 continuation 和域外拒绝；它们不应变成运行时 PDE 或最近点重绑定。
- [`soft_constraints.arap_volume_refine`](../src/projects/genesis_ue_sync/anatomy_retarget/soft_constraints.py:318) 只适合 watertight 器官 rest mesh 的离线保体积修正；开放血管/神经不能用 signed volume 验收。
- [`shape_volume.apply_material_bounded_soft_volume`](../src/projects/genesis_ue_sync/anatomy_retarget/shape_volume.py:1336) 不能原样作为生产场：其骨点和皮肤点都经 cKDTree 映射到最近 cage 节点，骨边界使用 `0.05` soft spring，并把场结果与 weighted target 混合且允许外点 fallback。这些条件不能提供精确骨端、连续皮内不等式或严格域内保证。

### 候选验收核对单

1. 编译时所有六类材料均有固定材料坐标；ambient cage 内插必须有正 Jacobian，域外查询确定性失败，禁止最近点/外推补洞。
2. 运行只求值已编译的 beta/pose basis；不得重绑定、PDE、KD-tree、碰撞迭代或每帧优化。T-pose、自身/互换 capture 和独立 AMASS holdout 需要直接 θ 回放。
3. 骨端和关节接合用独立骨组件/三角面检查；允许骨按图审轻微出皮，但骨—骨与管—骨穿入超过 0.5 mm 仍失败，不能用皮肤边界强行修骨。
4. 软组织必须满足 SMPL-X 皮肤内侧不等式；器官保持正体积和局部 Jacobian，血管/神经保持边长、半径/截面和分叉连续性。骨边界只约束可行的皮内段，出皮段裁分而不伪造闭合空腔。
5. Genesis 至少用两个 beta、T/动作和前后/侧/斜视角，同时保存深度/三角相交指标；投影图只能辅助图审，不能代替三维检查。

### 本轮实际图审边界

我打开了 `_002` pair Genesis 的 213328 capture whole/right-elbow 及 213712 walking whole/left-ankle 对照图：candidate 与旧版在 capture 多数区域接近，walk 右臂有可见位置变化，红色血管随链移动；足踝骨—血管—神经的方向问题仍可见，不能据此宣称解剖通过。另打开最新收到的 `v16_viscera_baseline` 213328 T-pose 和 sitstand context/viscera contact sheet；sitstand context 能看到器官，但 viscera 图被皮肤/遮罩大面积压暗，缺少可核对的前后和可靠深度，因此该批图不能作为器官 fit 验收证据，需 renderer 修正后重审。
### Clean `v16_viscera_baseline_genesis_002` 图审补充

我实际打开了两个体型的 T-pose 和 `sitstand_sid4336_middle` 的 `context`、`viscera`、`skin` contact sheet，以及 torso oblique 原图。两种体型的 context/viscera 图中，肺—肋笼、心脏—纵隔、肝—肋弓、肠管—骨盆的宏观相对位置连续，未从这些投影确认一个确定的脏器—骨骼冲突；肋骨覆盖肺缘属于预期包围关系，不能当作穿入结论。另一方面，图中没有同姿态 before/after 或可视化深度/三角相交标记，髋臼—股骨头、骨盆—肠管、肋骨—肺表面以及血管/神经—骨的深度关系仍不确定。

该批 manifest 标明 `solver_or_fit_run=false`、只有 `candidate` variant，输入是 `v16_baseline_geometry_20260908_001/*.npz`；所以它是渲染/材料选择检查，不能证明 V15 固定共享场已经接入。可执行的最小修正是：在编译阶段用共同 ambient tet cage 为 `organ/heart/vessel/nerve/connective_tissue` 持久化材料坐标和固定姿态残差 basis，并让 `CompiledAnatomyV14.apply_pose` 在基础 LBS 后只对这些 soft ranges 求值；骨组件保持独立刚体结果，皮肤仅对软组织施加内侧不等式。构建时所有 soft sample 必须在域内且 Jacobian 为正，骨/皮不可行时分割边界并 fail closed；不要把现有最近节点 soft spring 或外点 fallback 当作精确修正。
## Bounded colon/shared-radial candidate 审查

该候选可以作为受限实验：原 14 槽权重不变，运行只求值预烘焙的 `theta` scalar/field。最严重的可行性风险是，colon 的单一全局体积标量并不能确定一个局部可逆的三维位移；总 volume 恢复时，局部 Jacobian 仍可能变小或翻转，且可能把肠壁推入骨盆或皮肤。将同一个静态径向位移施加给相邻血管/神经也只能保持粗略共同运动，不能自动保持它们各自的分叉、管径、截面和骨距离。骨面 taper 到 0 只是位移边界，未提供软组织到骨面的非穿入约束，taper 邻域还可能产生高剪切。

因此本候选的最小 gate 不是“colon volume 变好”一个指标，而是对训练和 held-out 姿态同时检查所有共享 soft ranges：`min(det J)>0`、皮肤内侧距离、骨—软组织三角穿入、血管/神经边长与分叉残差；任一失败即候选失败。它不能被描述为已实现骨—皮双边界场，也不能因 held-out sitstand 的总体积恢复而发布。

图审补充：已打开 clean `v16_viscera_baseline_genesis_002` 的两个 beta（213328、213712）T-pose/sitstand `context`、`viscera`、`skin` contact sheet 及 torso oblique 原图。肺—肋笼、心脏—纵隔、肝—肋弓、肠管—骨盆的宏观相对位置未见一个可从投影确认的脏器—骨骼冲突；髋臼—股骨头、骨盆—肠管、肋骨—肺表面和血管/神经—骨深度仍需三角面指标。该 manifest 为 `solver_or_fit_run=false` 且只有 candidate variant，所以只能作为渲染检查。

### `visceral_volume_v16.py` 数学复核

实际用闭合四面体和两个包的 `Large_Intestine`（7 个 controller）将 `BakedVolumeV16.evaluate` 与直接 LBS 体积比对，误差约为 `1e-8` 相对量级；Cauchy—Binet 的 compact-controller 行列式索引/转置正确。将 transforms 的平移清零再运输 field，对“位移向量场”语义是正确的；在 field 为零的骨顶点上，额外修正也保持为零。

但 `requested = cbrt(1 / ratio) - 1` 只对同一中心的各向同性缩放精确。一般多 controller LBS 下 `after = before + amplitude * correction` 的真实体积是 amplitude 的三次式，不能由修正前 volume ratio 保证；必须计算修正后 volume 并 gate。静态 rest bone taper 也不等于 posed bone clearance，且代码没有 skin 约束。最后，`BakedVolumeV16.compile` 没有自行强制权重逐行归一化，`center` 的共平移消除会继承上游约 `2e-6` 容差；应在编译时归一化或 fail closed。上述问题不改变该实现作为 bounded candidate 的定位。

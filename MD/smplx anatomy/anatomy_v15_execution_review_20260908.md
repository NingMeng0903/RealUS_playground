# V15 双侧响应修正与真实 Genesis 二次检查 — 2026-09-08

**没有完成“任意 β、任意 θ 都有合理内部结构”的全部要求。** 本轮完成的是可复核的双侧锁骨响应修正、两个已编译体型的同输入比较，以及实际 Genesis 出图和走路回放。候选继续标记 `anatomical_passed=false`，没有提升为解剖通过版本。

## 直接查看

- [Genesis 对照检查页](../outputs/anatomy_retarget/v15_review_20260908_001/index.html)
- [最新固定 213328 包的走路视频：全身与足部](../outputs/anatomy_retarget/v15_translation_walk_video_213328_20260908_001/genesis_motion.mp4)
- [新候选独立代码与图审](anatomy_v15_bilateral_independent_review_20260908.md)
- [旧 V14 包的独立二次检查](anatomy_v15_independent_recheck_20260908.md)

视频为原生走路窗口每 8 帧取一帧，共 33 帧，按 12.5 fps 播放。主 agent 实际打开了首、中、末帧；不能把视频抽帧称作全部原生帧的解剖验收。

独立 agent 另完成了两个保存包各 1211 帧、合计 2422 帧的六段原生动作直接回放：全部 finite，0 rejected，0 exception；每包仅加载一次，没有重编译、裁角、Blender 或逐帧优化。耗时分别约 212.6 s / 274.6 s。这是完整原生窗口的**运行支持**检查，没有计算全帧表面距离，不能转为完整解剖通过。见[原生运行报告](../outputs/anatomy_retarget/v15_native_runtime_recheck_20260908_001/report.md)。

上述整段回放及下文 42 帧对照对应 `v15_bilateral_collar_*_001`。独立检查另发现 `_001` 继承的旧平移映射缺少坐标基认证，不能把求值成功写成平移坐标一致性已经通过。原报告与图片保留作为该版证据，数值修复必须另存候选并复测。

**最新 `_002` 已修复该兼容性问题。** `translation_rebuild_v15.py` 从保存的 ArmMap 参数重建原 rest 场，全量 rest/bind/reference 核对通过后才恢复世界 Jacobian 并生成新平移映射；未知旧格式拒绝迁移。213328 重建误差为 0，213712 约 3e-16。6 项新增测试通过，独立 agent 代码复审确认原坐标基问题已修复。

两个 `_002` 包又分别复测 21 帧，0 异常；相对 `_001` 的最大顶点变化为 0.2406 mm / 0.0581 mm。主 agent 实际打开新的肘、足部 Genesis 对照，以及新包走路视频的首、中、末帧：图中足部方向与位置的偏差仍在。最新血管最坏出皮为 38.923 mm / 29.911 mm，解剖结论仍失败。新包视频为 33 帧抽样视觉回放；没有把旧包的 2422 帧原生检查套用于新包。

- [_002 的 213328 复测](../outputs/anatomy_retarget/v15_translation_eval_213328_20260908_001/report.json)
- [_002 的 213712 复测](../outputs/anatomy_retarget/v15_translation_eval_213712_20260908_001/report.json)
- [_002 肘部实际对照](../outputs/anatomy_retarget/v15_translation_genesis_213328_20260908_001/capture_213712/comparison/left_elbow_lateral.png)
- [_002 足部实际对照](../outputs/anatomy_retarget/v15_translation_genesis_213712_20260908_001/sitstand_sid4336_middle/comparison/left_foot_oblique.png)

## 本轮实际修改

1. `collar_response_v15.py` 从源 rig 名称发现左右锁骨控制器，明确连接 J13/J14；恢复右侧锁骨支点及局部响应，整个右肩—肘—腕—手指子树继承一致运动。仅改变允许校准的响应与有效运动参考，源拓扑、逐顶点权重和控制器父子结构保留。
2. `compile_bilateral_collar_v15.py` 从原始 source asset 重建双侧响应，保存一致 reference/target bind 和旋转、平移运输。旧 rest 几何不变；在不同运动律下拟合过的 pose corrector 会被拒绝复用。实际保存两个候选包，重载后 3 个契约姿态逐元素一致。
3. `bone_components_v15.py` 实现由原 LBS 烘焙矩直接求独立骨组件刚体目标的模块，排除原分类为 bone 的椎间盘。**它尚未接入生产 pose，尚未形成骨—软组织共享场，也未解决骨间接触。** 不能将模块存在写成这一阶段完成。
4. 冻结六段 BABEL 标注的 AMASS 原生动作和来源 hash，增加同 β、同 θ、同 translation 的 before/candidate 比较 CLI。采集或 AMASS 的姿态只驱动当前包自己的 β，不移植动作文件的体型。

针对双侧响应、骨组件和冻结动作的 31 项测试通过。两个体型的原 `faces`、14 槽 `driver_indices/driver_weights`、mesh ranges 和 controller owner 在 before/after 包中逐元素一致。

本轮最新编译包（包含数值重建，仍为未通过候选）：

- [213328](../outputs/anatomy_retarget/v15_bilateral_collar_213328_20260908_002/compiled/manifest.json)
- [213712](../outputs/anatomy_retarget/v15_bilateral_collar_213712_20260908_002/compiled/manifest.json)

直接驱动示例（工作目录为仓库根，`PYTHONPATH=src`）：

```python
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    load_compiled_subject, pose,
)
compiled = load_compiled_subject(
    'outputs/anatomy_retarget/v15_bilateral_collar_213328_20260908_002/compiled'
)
# theta55: [55, 3] 轴角弧度；transl: [3] 米。
# 在动作循环外加载一次，此后每帧只调用 pose。
vertices = pose(compiled, theta55, transl)
```

这两个包继承各自已物化的 β/rest。**没有完成新 β 的全身自动骨长适配器。** 原 V8 路径的人工 β 范围限制与旧线性 anatomy 体型基底仍在；不能通过本轮两个包宣称这些问题已经消失。

## 实际图审发现

所有对照图均由现有 Genesis 渲染器读取保存几何生成，没有生成式图片、示意骨骼或图像修饰。左图是原左侧修正版，右图是本轮双侧响应候选。皮肤、相机、β、θ 相同，显示时对皮肤与内部模型等量去除 root 旋转。

| 实际打开的证据 | 可见结论及限制 |
|---|---|
| [213328 T-pose 全身](../outputs/anatomy_retarget/v15_pair_genesis_213328_20260908_001/tpose/comparison/whole_ap.png)、[213712 T-pose 全身](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/tpose/comparison/whole_ap.png) | rest 图基本相同；本轮没有用缩细骨骼取得外观改善。相同图不能证明原有骨面关系合理。 |
| [213328 互换采集动作](../outputs/anatomy_retarget/v15_pair_genesis_213328_20260908_001/capture_213712/comparison/whole_ap.png) | 双侧手臂和血管都随动作，但此类采集姿态的新旧差别很小，不能算此轮局部改善证据。 |
| [213712 走路全身](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/walk_sid8836_middle/comparison/whole_ap.png)、[右肘侧视](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/walk_sid8836_middle/comparison/right_elbow_lateral.png) | 新候选右臂整链向目标皮肤包络移动，骨与血管一起移动；不能只凭投影认定管—骨无穿入。 |
| [213712 喝水右腕](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_002/drink_sid3307_middle/comparison/right_wrist_lateral.png)、[右肘](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_002/drink_sid3307_middle/comparison/right_elbow_lateral.png) | 明显减少整条右臂相对目标的位移。腕部骨端、血管与皮肤的局部空间仍不能图审判定通过。 |
| [213712 髋前斜视](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/capture_213712/comparison/left_hip_oblique.png)、[髋后斜视](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/capture_213712/comparison/left_hip_posterior.png) | 本轮髋图无可见改善。股骨头被髋骨遮挡，不能把轮廓重合判成球窝关系通过。 |
| [213712 屈膝正视](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/capture_213328/comparison/left_knee_ap.png)、[侧视](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/capture_213328/comparison/left_knee_lateral.png) | 屈膝时骨与血管有连续跟随，新旧下肢关系基本相同；本轮未修复既有膝接合。 |
| [213712 坐起髋部](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_002/sitstand_sid4336_middle/comparison/left_hip_oblique.png)、[膝部](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_002/sitstand_sid4336_middle/comparison/left_knee_lateral.png)、[采集踝部](../outputs/anatomy_retarget/v15_pair_genesis_213712_20260908_001/capture_213712/comparison/left_ankle_oblique.png) | 下肢局部仍须修正与接触审查，不能将上肢整链改善推广为全身通过。 |

还实际查看了转身、伸手和抬臂全身对照，以及 213328 自身/互换采集、坐起全身图。局部相机前后各裁剪 160 mm；平切骨端、局部碎片外观必须先排除裁剪影响，不能直接称为骨折或拓扑断裂。

## 与图对应的量化补充

两个体型分别比较 T-pose、两采集动作，以及六段日常动作的首/中/末，共 21 帧/体型，42 帧均成功求值。每体型保存 9 个实际几何 NPZ；首末帧保留轻量指标。此处是抽查帧表面测量，不能称为全部原生帧的几何验证。

下表为血管最大皮外距离，单位 mm；仅用于量化相应图中的改善与残留问题。骨骼出皮按部位图审，不应用统一硬阈值。

| 体型 | 动作中点 | 修正前 | 修正后 |
|---|---|---:|---:|
| 213328 | 走路 | 58.20 | 18.29 |
| 213712 | 走路 | 58.49 | 8.29 |
| 213328 | 喝水 | 52.17 | 14.73 |
| 213712 | 喝水 | 51.95 | 11.64 |
| 213328 | 坐起 | 59.03 | 38.97 |
| 213712 | 坐起 | 61.63 | 29.93 |

两个体型的全部 42 个抽查帧仍有血管出皮超过 1 mm。每体型最坏值分别为 38.966 mm 与 29.930 mm，均出现在坐起中点。因此即使承认肘部等骨骼可以轻微出皮，也不能把当前结果算作软组织通过。

进一步按逐顶点距离和原权重定位：坐起的最坏血管点主要在左脚/左踝（两个体型一致），213328 使用另一采集动作时最坏点在右脚/右踝。已补出并实际打开 [213328 坐起左足斜视](../outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/sitstand_sid4336_middle/comparison/left_foot_oblique.png)及[同相机不透明目标皮肤](../outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/sitstand_sid4336_middle/smplx_skin/rgb/left_foot_oblique.png)，以及[采集动作右足](../outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/capture_213712/comparison/right_foot_oblique.png)及[目标皮肤](../outputs/anatomy_retarget/v15_worst_region_genesis_213328_20260908_001/capture_213712/smplx_skin/rgb/right_foot_oblique.png)。图中足骨与相连血管偏离目标脚的方向和轮廓，属于整段足部适配问题，不能以允许局部轻微骨出皮为由通过。最近 SMPL-X 关节仅用于选视角，不作为根因判定；控制器链还须单独审查。详见[顶点与权重定位](../outputs/anatomy_retarget/v15_vessel_failure_localization_20260908_001/report.json)。

- [213328 同输入完整报告](../outputs/anatomy_retarget/v15_pair_eval_213328_20260908_001/report.json)
- [213712 同输入完整报告](../outputs/anatomy_retarget/v15_pair_eval_213712_20260908_001/report.json)

## 对用户目标的明确回答

| 要求 | 当前状态 |
|---|---|
| 同一个人 bake 一次，换 θ 不重新 bake | 两个已保存候选具备此运行路径；新动作调用保存包求值，不运行 Blender 或逐帧优化。 |
| 保留原拓扑、权重、235 控制器联动结构 | 当前包保留；校准允许改变具体响应数值，不能称任意姿态下与原 Blender 坐标完全相同。 |
| 权重直接 bake 为 55 个 SMPL-X 关节就完全等价 | 未完成，也不能由当前证据推出。现在是 55 关节 θ 驱动保留的 235 控制器及保存响应，接口只需 θ。 |
| 新 β 自动有合理骨长、骨端与接合 | 未完成。通用骨层求解器、独立组件与最终 bind 尚未整合。 |
| 任意 θ 的髋球窝、膝肘腕踝都合理 | 未通过。原有穿插和骨端运动问题仍须独立修正。 |
| 骨与软组织在骨—皮之间共同仿形、保管径与分叉 | 未完成固定材料场和联合验收，当前仍有厘米级出皮。 |

后续修正顺序继续按[完整计划](anatomy_v15_general_beta_plan_20260908.md)：先求相邻骨端与独立骨组件的一致形状/运动，消除原始骨面相交，再构建骨—皮双边界共享运输场。每阶段以真实 Genesis 多视角、连续动作和独立检查确认；不能继续用骨骼单独位移而漏带相连血管，也不能仅删除 β 限制就宣称通用适配。

# V15 独立复核：V14 collar driver axes（2026-09-08）

本报告只读加载以下两个已保存包，并直接调用 `load_compiled_subject()` + `pose(compiled, theta55, transl)`。没有调用 `compile_subject`，没有 Blender、PDE、最近点重绑定或逐帧优化；没有修改 runtime/fit 代码。

## 结论

- **Operational：通过。** 两个包均能加载；235 控制器层级、原 14 槽 `driver_indices/driver_weights`、faces 和 source operator 身份均匹配。两个体型的 T pose、own/swap capture pose、held-out sitting/kicking 共 10 个 NPZ 回放全部逐元素一致；walking/sitting 四段共 144 帧的保存报告 SHA 也全部复现，0 个 unsupported frame。
- **Anatomical：失败。** 两个 manifest 本身标记 `anatomical_passed=false`。独立读取的全骨髋/膝/踝和全臂肘表面审计仍有 >0.5 mm 的表面穿入/接触候选；动作 skin outside 也达到厘米量级。Operational 通过不能转化为解剖通过或可发布。

## 结构与保存回放

| 检查 | 结果 |
|---|---|
| source operator | runtime digest `17f5d4e0bc328e85aef0d6dc6eba0e3fa8ca1ddd0a79f751ae259e129d00972b`；两个 embedded source pack 与 provenance 均匹配 |
| 控制器 | 两个包均为 235；parent 数组与 operator 完全一致，root parent = -1 |
| 原 14 槽权重/索引 | 两个包均为 `[394770,14]`，与 operator 逐元素一致；权重最大差 0，行和误差 `4.84e-8` |
| faces | `[782856,3]`，与 operator 完全一致；source skin faces/55-slot source skin LBS 也一致 |
| 归档 NPZ | 10/10 行 bit-exact，最大差 0 m |
| walking/sitting | 4×36 = 144/144 帧 hash 一致，全部 finite，unsupported = 0 |

输入包：

- [`213328 compiled`](../outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001/compiled/manifest.json)
- [`213712 compiled`](../outputs/anatomy_retarget/v14_collar_driver_axes_213712_20260908_001/compiled/manifest.json)
- [`source operator`](../outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8/manifest.json)

## 解剖表面失败证据

全骨 lower-chain 报告对两个体型的 T/own/swap 共 48 个 pair 做了三角面检查。最大 sampled signed depth 是采样下界：

| 区域 | 失败/总检查 | 最大 sampled depth | 最大 triangle contact pair |
|---|---:|---:|---:|
| 髋 `Ilium–Femur` | 12/12 | 5.048 mm | 92 |
| 膝 `Femur–Tibia` | 6/12 | 17.058 mm | 109 |
| 踝 `Tibia–Talus` | 12/12 | 7.294 mm | 100 |

全臂/肘 latest `driver_axes_arm_audit_..._002` 对每个体型 T、pose_213328、pose_213712 的 Humerus/Radius/Ulna 做 9 个 pair 检查：

- 213328：6/9 失败，最大 sampled depth 8.199 mm；pose_213712 全臂 skin outside 31.629 mm，9417 个顶点超过 1 mm。
- 213712：7/9 失败，最大 sampled depth 8.010 mm；pose_213712 全臂 skin outside 28.899 mm，8371 个顶点超过 1 mm。

既有 walking/sitting 回放的最大 skin outside（全骨/血管）也仍为厘米级：

- 213328 walking：全骨 61.076 mm，血管 61.690 mm；sitting：全骨 56.406 mm，血管 50.202 mm。
- 213712 walking：全骨 61.632 mm，血管 61.796 mm；sitting：全骨 47.848 mm，血管 45.988 mm。

这些报告明确区分了 signed depth 采样下界和连续表面证明；本次没有把投影图作为深度通过证据。

## Genesis 实际图审

本次用 `view_image` 打开了两个体型全身 T pose、左肘和左膝侧视，以及动作 contact sheet：

- [`213328 T 全身`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_tpose/comparison/whole_ap.png)
- [`213712 T 全身`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_tpose/comparison/whole_ap.png)
- [`213328 左肘侧视`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_elbow_lateral.png)
- [`213328 左膝侧视`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/comparison/left_knee_lateral.png)
- [`213328 pose contact sheet`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213328_pose_213328/candidate/contact_sheet.png)
- [`213712 pose contact sheet`](../outputs/anatomy_retarget/v14_driver_axes_genesis_20260908_001/subject_213712_pose_213712/candidate/contact_sheet.png)
- [`sitting frame 18`](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis/00018/contact_sheet.png)
- [`kicking frame 18`](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/genesis/00018/contact_sheet.png)

图中能看到整体骨骼/血管组件和关节邻域，但遮挡与投影无法判断三角面穿入或真实深度顺序；因此视觉结果只记为 **reviewed / no projection-only pass**。

## 可交付文件

- [`本次 JSON 报告`](../outputs/anatomy_retarget/v15_independent_recheck_20260908_001/report.json)
- [`本次 Markdown 报告`](anatomy_v15_independent_recheck_20260908.md)

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

1. 旧 translation map 的基底必须重建或确定性拒绝，补 provenance 后重新回放。
2. 刚性骨 boundary residual 尚未接入固定共享软组织/FEM 场，血管和软组织的生产运输尚未完成。
3. 尚未对 V15 做显式三维骨—骨、管—骨三角面穿入检查，特别是髋、膝、肘、腕。
4. 本轮只打开代表图，没有逐一打开两个体型全部生成的每个姿态/相机格。
5. 尚未证明任意新 beta 和广泛 theta；当前证据只是两个采集体型及有限存档/AMASS 帧，不能做 universal claim。

针对性测试已通过：`tests/test_collar_response_v15.py` 与 `tests/test_bone_components_v15.py` 共 **10 passed**。这只证明代码合同，不改变上述解剖失败结论。

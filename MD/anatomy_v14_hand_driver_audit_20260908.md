# V14 hand driver audit — 2026-09-08

这份记录是对 V14 arm-fit 包中 **raw142 source** 的只读几何审查。它回答的是“30 mm 级手部外穿来自哪里”，不修改 runtime、fit 或 renderer，也不把诊断用的根部平移当成运行时方案。

## 输入与复现方式

- 编译包：[v14 arm fit compiled](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_arm_fit_20260908_001/compiled)
- 输入帧：同目录中的 `subject_213328_tpose.npz`、`subject_213328_pose_213328.npz`、`subject_213328_pose_213712.npz`、`elbow_L_{30,60,90,120}.npz`，以及两个没有参与拟合的 `heldout_sitting.npz`、`heldout_kicking.npz`。
- 小型可复核输出：[metrics.json](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_hand_driver_audit_20260908_001/metrics.json)、[controller_metrics.npz](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_hand_driver_audit_20260908_001/controller_metrics.npz)。`metrics.json` 标记为 `analysis_only=true`、`anatomical_passed=false`、`publishable=false`。
- 三维复核使用现有 Genesis 输出：[sitting wrist lateral raw142](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_arm_fit_genesis_20260908_001/subject_213328_heldout_sitting/raw142/rgb/left_wrist_lateral.png)、[sitting wrist comparison](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_arm_fit_genesis_20260908_001/subject_213328_heldout_sitting/comparison/left_wrist_lateral.png)、[T-pose wrist comparison](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v14_arm_fit_genesis_20260908_001/subject_213328_tpose/comparison/left_wrist_lateral.png)。图中 T-pose 的手部基本落在目标区域；sitting 的 raw142 侧视图显示前臂、腕和手整体相对皮肤向外偏移。图审用于定位问题，数值以同一输入的几何测量为准。

对每一帧计算：

1. `G[i] = source_bone_posed_global(asset, pose)[i]`，即当前 235 控制器的 source global；
2. `F[i] = source_bone_driver_frames(asset, pose)[i]`，`D[i] = F[i] @ source_driver_coupling[i]`，即对应的直接目标 frame；
3. `T[i] = source_bone_skinning_transforms(asset, pose)[i] = G[i] @ inverse_bind[i]`，即真正作用于 mesh 的 deformation map；
4. 以 Wrist_Rotate_L 的后代 authored bone meshes 为手部集合，共 9,921 个顶点，和该 NPZ 的 SMPL-X skin 做 signed distance。正距离表示在皮肤外；
5. 对 133/134 计算 `S = inv(T[133]) @ T[134]`。这是动作中的相对 skinning 变形；T-pose 的 139 mm 骨间 rest 间距不混入这个量。

## A：30 mm 外穿首先是整条腕部原点漂移

### 共同平移证据

| pose | raw 手骨最大皮外 | raw Wrist_Rotate_L 到 SMPL-X joint 20 | `D[135]` 到 joint 20 | 仅把 raw 手部平移到 joint 20 后最大皮外 |
|---|---:|---:|---:|---:|
| T-pose | 0.620 mm | 0.114 mm | 0.114 mm | 0.641 mm |
| 213328 own | 7.138 mm | 7.359 mm | 0.114 mm | 2.240 mm |
| 213712 own | 0.796 mm | 1.209 mm | 0.114 mm | 1.085 mm |
| held-out sitting | **30.569 mm** | **48.110 mm** | 0.114 mm | **1.026 mm** |
| held-out kicking | **32.728 mm** | **46.427 mm** | 0.114 mm | **1.827 mm** |

这个“只平移”的结果是诊断，不是恢复独立 world-space wrist 的提案。它说明 held-out 的 30–33 mm 皮外距离在平移掉腕部共同偏差后只剩约 1–2 mm；因此主要故障是整条手的 origin path，不能归因于每根手指各自的旋转轴。

实际手指控制器也携带同一偏差。以 sitting 为例，Wrist 误差为 48.110 mm，各手指基节到末节均约 48.09–48.16 mm；kicking 为 46.40–46.46 mm。source controller 和 `D` 的旋转差在全部手部控制器上小于 `8.1e-14°`（双精度数值噪声），所以“位置整体错、方向基本一致”是直接测得的结果。

### 第一处可比较的链路分叉

`eff19`（130）是显式 `bind_follow` helper。其 frame 和 coupling 按 schema 有意跳过，当前 `D[130]` 是 identity 哨兵，不能拿它当 direct target 计算 origin 误差。source FK 的 helper 连续性本身是精确的：

```text
G[130] = G[129] @ target_bind_local[130]
continuity residual = 0.0 µm  (sitting and kicking)
```

因此表中的第一项从 129 跳到直接驱动的 131：

| source controller | held-out sitting `|G-D|` | held-out kicking `|G-D|` |
|---|---:|---:|
| 129 Clavicle_Rot_L | 0.266 mm | 0.041 mm |
| 131 Shoulder_Rotate_L | **48.582 mm** | **46.623 mm** |
| 132 Elbow_Rot_L | 48.559 mm | 46.612 mm |
| 133 Forearm_Bone_L | 48.559 mm | 46.612 mm |
| 134 Forearm_Twist_L | 48.191 mm | 46.514 mm |
| 135 Wrist_Rotate_L | 48.164 mm | 46.462 mm |

V14 的 full local FK 分支在 [`anatomy_lbs.py:1880`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/anatomy_lbs.py:1880) 以后用 fitted source-parent local translation 重建子骨；direct frame 则由 SMPL-X 16→18（131）和 18→20（132/133/134）提供目标 frame。129 的方向/原点已接近 direct target，但 130 的 helper path 把这个父系路径带到与目标 16 号肩部 frame 不同的位置，131 于是第一次出现约 46–49 mm 的 world origin 偏差。132–135 只是继承它，手指再继承 135。

这解释了为什么“给每一根手指加旋转补丁”不会解决问题，也解释了为什么恢复独立 world-space wrist motion 会破坏用户要求保留的联动关系。需要修的是 130/131 一致的 rest/local FK 与目标肩部路径；手腕仍应由这条链和 SMPL-X 参数驱动。

## B：手指映射和局部旋转不是 30 mm 首因

当前左手映射如下，全部是明确的 joint-local driver：

| source controller | SMPL-X joint |
|---|---:|
| Wrist_Rotate_L 135 | 20 |
| Thumb_Rot1/2/3 137/139/141 | 37/38/39 |
| Index_Rot1/2/3 143/145/147 | 25/26/27 |
| Middle_Rot1/2/3 149/151/153 | 28/29/30 |
| Ring_Rot1/2/3 155/157/159 | 34/35/36 |
| Pinky_Rot1/2/3 161/163/165 | 31/32/33 |

`D` frame 到对应 SMPL-X knuckle 的 direct origin 误差在 held-out 两帧相同：Wrist 0.114 mm；Thumb_Rot1 1.831 mm；其余手指控制器最大约 0.099 mm。Thumb_Rot1 的 1.831 mm 是 source driver contact/rest joint 与 SMPL-X thumb 37 的小偏移，量级远低于 46–48 mm 的腕部共同偏移。

把 `D[i]` 放回 source parent `G[parent]` 的局部 frame 后，sitting 的手部局部 translation 差为约 46.65–48.20 mm，kicking 为约 44.64–46.52 mm；局部 rotation 差仍小于 `2.4e-14°`。这说明 full local FK 保留了错误的父系原点路径，未显示出独立的 finger-axis mismatch。

rest 形状也不支持“需要先重编译手指才可解决 30 mm”的判断：fitted target bind controller 与 SMPL-X rest joint 的最大 origin 差约 0.114 mm，手指相邻 segment 的差小于 0.08 mm；T-pose 手骨最大皮外仅 0.620 mm，超过 1 mm 的点为 0。根部修正后的 held-out 残余为 1.026 mm（sitting）和 1.827 mm（kicking），这才是后续手部 shape/皮肤内约束可以处理的量级。这里没有重复引入另一个 beta source-cap 形状限制；该限制应在独立体型编译审查中处理。

## C：134 的前臂扭转输入，以及为什么 rest cap 通过不代表动作刚性

资产中 133/134 的输入不是独立 wrist world transform：

```text
133 Forearm_Bone_L   parent=132  mode=segment_root  a=18 b=20 blend=0
134 Forearm_Twist_L  parent=133  mode=twist         a=18 b=20 blend=0.77999997
```

在 [`anatomy_lbs.py:470`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/anatomy_lbs.py:470) 的 `_endpoint_segment_delta` 中，134 使用 `proximal_delta=joint_delta[18]`、`distal_delta=joint_delta[20]`，以 posed 18→20 轴投影出两端 frame 的部分 axial roll，再乘 `twist_alpha≈0.78`。因此它是 SMPL-X elbow/wrist endpoint frame 的 twist follower；代码没有单独的解剖 pronation 参数，也没有把 wrist 以 world-space 独立恢复。

### 纯肘探针与真实动作

| pose | `S` 旋转 | `S` 平移 | SMPL-X 18→20 相对旋转 | 轴向分量 / 正交分量 |
|---|---:|---:|---:|---:|
| elbow 30° | <0.00001° | <0.00002 mm | 0° | 0° / 0° |
| elbow 60° | <0.00001° | <0.00002 mm | 0° | 0° / 0° |
| elbow 90° | <0.00001° | ~0 mm | 0° | 0° / 0° |
| elbow 120° | <0.00001° | 0 mm | 0° | 0° / 0° |
| 213328 own | **30.752°** | **38.463 mm** | 42.730° | 39.692° / 15.824° |
| 213712 own | **14.744°** | **18.613 mm** | 24.660° | 18.096° / 16.752° |
| held-out sitting | **7.735°** | **9.785 mm** | 26.448° | 11.401° / 23.864° |
| held-out kicking | **1.648°** | **2.086 mm** | 32.236° | 3.798° / 32.012° |

纯 elbow probe 中 joint 20 没有相对 18 的旋转，所以即使端点轴从 rest 改变约 5.57°、36.19°、66.75°、97.14°，133 和 134 仍共同运动，`S` 近 identity。这把“肘屈曲”与“腕相对 18 的扭转/方向变化”分开了。

在 213328 own 中，`S` 的 30.752° 几乎全是沿当前 forearm axis 的分量，并随 18→20 的 39.692° axial component 增长；213712 own 也呈同方向的量级关系。sitting/kicking 的 SMPL-X 相对旋转主要是正交分量，134 只取出较小的轴向部分。这可以是对 forearm roll 的有意近似，但不能称为已经验证的解剖 pronation：输入是 endpoint frame 的轴投影，且同时受 SMPL-X wrist local rotation、elbow local rotation 和 fitted frame axes 影响。工程上应把 134 的大相对变形视为“需要动作中验证的 twist coupling”，不能静默把它归零，也不能把它替换成独立 wrist world motion。

### 133/134 权重使骨面会真实变形

原权重没有被改动，但 Radius/Ulna 并非各自完全单骨权重：

| mesh | 顶点数 | 同时受 133、134 影响 | 多正权重顶点 | 133/134 权重和 |
|---|---:|---:|---:|---:|
| Radius_L | 366 | **173** | 173 | 177.347 / 188.653 |
| Ulna_L | 527 | **200** | 200 | 287.970 / 239.030 |

所以 rest cap 的形状误差为 0，不能推出 posed bone cap 仍是刚体。用实际 `T` 反算 owner transform 时，213328 own 的 Radius/Ulna 最大残差分别为 11.83/7.87 mm，三角边相对长度 p99 分别约 3.45%/3.55%；213712 own 为 5.73/3.81 mm；held-out sitting 为 3.01/2.00 mm；kicking 为 0.64/0.43 mm。纯 elbow probes 的同一残差小于 0.06 mm。source NPZ 在这些子集上由当前 sparse LBS 重建的最大误差只有 0.13 µm，故上述变形不是读写误差，而是 133/134 相对 skinning map 与混合权重共同造成的动作变形。

这也是“bound cap 只在 rest 通过”与“动作骨面保持刚性”之间的区别：hand-driver root 漂移和 134 forearm coupling 是两个独立问题，前者先导致 30 mm 外穿，后者决定修好 root 后 Radius/Ulna 是否还会在 own pose 中折扭。

## 5°/5 mm 局部修正能否补掉腕部漂移

我对现有 arm corrector 的三个局部 controller `[131,132,135]` 做了一个独立矩阵 probe：每个 local rotation norm ≤5°、translation norm ≤5 mm，8 个确定性起点，目标只最小化 Wrist 135 到 direct target 的 origin error。它不是新的 runtime，也没有写入 fit 参数。

| pose | raw wrist error | translation-only lower bound | 受限 probe 最好 wrist residual | shoulder raw / 最好 |
|---|---:|---:|---:|---:|
| held-out sitting | 48.164 mm | 33.164 mm | 19.601 mm | 48.582 / 43.858 mm |
| held-out kicking | 46.462 mm | 31.462 mm | 18.105 mm | 46.623 / 42.294 mm |

SLSQP 两帧均达到 iteration limit，故这些是有界搜索的诊断值，不是通过的拟合结果；但 shoulder origin 的下界更直接：131 的 local translation 最多只能移动 5 mm，elbow/wrist correction 不能移动 131 的 origin，因此仍至少约 43.58/41.62 mm。结论是 ≤5°/5 mm 的 shoulder/elbow/wrist post-correction 无法单独修掉该源头偏差，必须先统一 130 helper、131 shoulder driver 与其 fitted local bind path。修好 root 后，再对 133/134 的 twist 和 Radius/Ulna 混合权重做骨—皮双边界验证。

## 审查结论与交接边界

1. 30+ mm hand failure 的首因是 130 bind-follow helper 之后的 131 source-parent path 与 SMPL-X shoulder target frame 的原点不一致；所有手指共同携带该偏差。
2. 现有 finger joint IDs 和轴向旋转没有显示出 30 mm 级错误；Thumb_Rot1 的约 1.83 mm contact offset 应单独记录。
3. 134 的输入明确来自 18→20 endpoint frame 和 `twist_alpha=0.78`。纯 elbow 运动不会触发它；own pose 中的 14.7–30.8°、18.6–38.5 mm 是动作中的相对 forearm skinning 变形，不能由 rest cap 结果掩盖。
4. Radius/Ulna 的 133/134 混合权重使动作骨面会变形；需要 root chain 统一后再做保骨端、骨—皮双边界和管线联动审查。
5. 本轮没有修改 runtime、fit、renderer 或权重，也没有提出恢复独立 world-space wrist motion。没有宣称 anatomical pass。


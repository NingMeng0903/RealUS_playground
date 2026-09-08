# 两个 capture 的髋、膝、肘关节指标审查

审查范围是已完成的 `v13_material_20260908_001` 矩阵中 subject `213328` 和
`213712` 的 `tpose`、`pose_213328`、`pose_213712` 三个姿态。原始记录见
[progress.json](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_material_20260908_001/progress.json)。本审查只读，没有改动 solver 或验证代码。

## 结论

当前 `joint_plausibility_v12` 的髋关节指标不是“股骨头中心到 SMPL-X 髋关节中心的距离”。它在冻结的 `femoral_head` 和 `acetabulum` 顶点域上分别拟合股骨头球和固定股骨头半径的髋臼球帽，然后报告两个拟合中心的欧氏距离（`center_error_m`）。因此它确实使用了股骨头—髋臼两侧的几何域，但它仍然只是球心同心度指标，不能单独证明关节面接触、间隙、穿插或脱臼。

膝和肘的 V12 指标更弱：它们把 SMPL-X 的膝/肘关节点当作 `station`，再计算该点到最终材料表面估计出的横向铰链轴的垂直距离，以及沿轴偏移。膝没有计算股骨髁—胫骨平台的表面间距；肘没有计算肱骨—桡骨/尺骨的关节面关系。因此这些值不能作为膝关节接触或肘关节脱位的证明。

## 代码实际测量的内容

### `joint_plausibility_v12`

实现位于 [`joint_plausibility_v12.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/joint_plausibility_v12.py:92)。

* 髋：在 `left/right/femoral_head.<partition>` 上调用最小二乘球拟合，在 `left/right/acetabulum.<partition>` 上调用固定股骨头半径的球帽拟合，然后返回 `||head.center - socket.center||`。髋路径没有使用传入的 `smplx_joints`；传入的 SMPL-X 关节只在铰链分支使用。
* 膝：由内外侧股骨髁和胫骨平台域的均值构造膝帧；横向轴是髁/平台中心的连线，指标是 SMPL-X joint 4/5 到这个轴的垂直距离。
* 肘：由 `elbow/<side>/humerus/radius/ulna` 域均值构造肘原点；横向轴使用独立的整段桡骨/尺骨 `forearm_axis` 域，指标是 SMPL-X joint 18/19 到该轴的垂直距离。
* 该函数的最近点间距只出现在踝关节的 `ankle_mortise` 分支；髋、膝、肘没有对应的最近点或三角形表面间距计算。

`compare_joint_plausibility_v12` 的默认规则也是相对规则：髋目标 2 mm、回归限值 1 mm，铰链轴回归限值 2 mm；它没有为膝/肘站点到轴的绝对值定义通过线。更关键的是，001 矩阵只调用了 `joint_plausibility_metrics_v12` 记录诊断值，没有调用这个 compare gate；该矩阵 cell 的 `passed` 由组织表面指标和 tube edge ratio 决定，不能解读为髋、膝、肘关节通过。

### `anatomical_calibration_v1`

校准实现位于 [`anatomical_calibration_v1.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/anatomical_calibration_v1.py:401)。它在 source rest 的冻结 `fit` 域上建立解剖原点和帧，再在不重选 ID 的 `validation` 域上复测。髋原点是股骨头球心与髋臼球帽球心的中点；膝原点由两侧髁/平台中心组合而成；肘原点是肱骨、桡骨、尺骨域均值。校准检查验证 fit→validation 的中心、帧、轴和宽度一致性，属于 rest 几何的重复性检查，不是 posed 关节接触检查。`build_report` 也明确 `method` 为 `frozen_source_material_frames_no_raw_smplx_snap`，没有把 raw SMPL-X 髋平移写回 source anatomy。

校准 artifact 的 checker report 显示相关 rest 检查如下；这些数值不能外推为两个 capture 在动作中的接触证据：

| joint | fit→validation center error | fit→validation frame / hinge-axis error | hip head/socket error |
|---|---:|---:|---:|
| left hip | 1.087 mm | 4.386° / — | 1.794 mm |
| right hip | 1.384 mm | 1.151° / — | 1.521 mm |
| left knee | 1.315 mm | 2.392° / 2.391° | — |
| right knee | 1.536 mm | 0.203° / 0.140° | — |
| left elbow | 0.140 mm | 1.452° / 1.452° | — |
| right elbow | 0.435 mm | 1.439° / 1.422° | — |

其中 checker 的 hip head/socket 限值是 2 mm，center/frame 限值是 10 mm/6°。这只能说明冻结 source 域的 fit/validation 复测通过，不能说明 candidate pose 中股骨头没有穿过髋臼，也不能说明膝、肘骨面仍接触。

## 001 三个姿态的实际数值

以下单位均为 mm。每个单元格写作 `target / raw`：`target` 是 001 的 candidate final mesh（selected mode 为 attachments），`raw` 是同一 pose 的 source runtime。髋列是 `center_error_m`；膝、肘列是 `station_to_axis_perpendicular_m`。这些是诊断值，不是外皮关节距离。

| subject | pose | left hip | right hip | left knee | right knee | left elbow | right elbow |
|---|---|---:|---:|---:|---:|---:|---:|
| 213328 | tpose | 3.083 / 3.085 | 2.648 / 2.659 | 3.253 / 2.562 | 3.603 / 2.943 | 19.556 / 7.596 | 2.784 / 8.279 |
| 213328 | pose_213328 | 3.270 / 3.269 | 2.607 / 2.616 | 26.129 / 25.664 | 18.189 / 20.181 | 21.425 / 13.845 | 2.994 / 9.136 |
| 213328 | pose_213712 | 3.201 / 3.202 | 2.465 / 2.475 | 17.498 / 18.868 | 30.481 / 32.453 | 24.474 / 10.162 | 5.127 / 9.116 |
| 213712 | tpose | 3.101 / 3.100 | 2.576 / 2.589 | 2.823 / 2.459 | 4.306 / 2.780 | 17.028 / 7.654 | 3.283 / 8.285 |
| 213712 | pose_213328 | 3.285 / 3.281 | 2.525 / 2.537 | 26.436 / 25.612 | 18.583 / 19.941 | 18.109 / 13.894 | 4.481 / 9.507 |
| 213712 | pose_213712 | 3.219 / 3.217 | 2.407 / 2.419 | 16.499 / 18.789 | 29.658 / 32.459 | 21.901 / 10.295 | 6.561 / 9.345 |

从这张表可以得出以下有限结论：

* 两个 subject、三个姿态的 candidate 髋中心误差都在约 2.4–3.3 mm；因此均超过 V12 记录的 2 mm target，但这不是脱臼证明，也不是 SMPL-X joint-center 误差。
* 膝/肘轴距在动作中可到约 16–30 mm。这说明 SMPL-X station 与材料估计轴存在明显偏移，但它没有告诉我们两个相对骨面之间是接触、分离还是互相穿透。
* raw 和 target 的数值相近，不能据此推断正确；它们共享同一类诊断定义，并且没有表面接触测量。

完整的 `station_along_axis_m` 也保存在同一 JSON 的 `joint_plausibility_target/raw.hinge_axis` 下。它是轴向投影偏移，不能替代垂直距离，更不能替代关节面间隙。

## 能证明和不能证明

| 问题 | 当前记录能证明 | 当前记录不能证明 |
|---|---|---|
| 髋 | 冻结股骨头域和髋臼域的球拟合中心分离量；source rest 域 fit/validation 的一致性 | 股骨头与髋臼实际表面是否接触、是否有穿透、最小/最大真实间隙、关节面方向、是否脱臼 |
| 膝 | SMPL-X 膝 station 到由冻结域重建的横向轴的距离 | 股骨髁—胫骨平台的接触/分离/穿透、半月板空间、膝关节面相对姿态 |
| 肘 | SMPL-X 肘 station 到由肱骨/桡骨/尺骨及 forearm-axis 域定义的轴的距离 | 肱骨小头/滑车与桡骨头/尺骨滑车切迹的接触、间隙、穿透或脱位 |
| 外皮 containment | 骨材料相对 SMPL-X 外皮的 signed distance | 内部关节面的关系；外皮距离不能替代内部关节面证据 |

旧的 [`joint_contact_v7.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/joint_contact_v7.py:1103) 中另有 `_hip_metrics`，会额外报告头/髋臼冻结域的点集最近距离、中心漂移、半径变化和最大分离；`_knee_metrics` 会报告髁—平台点集 clearance。但 001 的 V13 matrix 没有调用这些函数，V12 输出中也没有这些字段。即使以后启用，它们仍是冻结顶点采样的离散证据，需要三角形表面距离和相交测试才能把“没有脱臼/没有穿透”提升为更强结论。

## 髋部高亮的 source mesh 与 domain 范围

source operator 的 `source_mesh_names/source_vertex_ranges` 中没有名为 `Pelviscap` 或 `Acetabulum` 的独立 mesh。髋臼 cap 属于 `Ilium_L/R`，股骨头属于 `Femur_L/R`：

| source mesh | global vertex range `[start, stop)` | count | 用途 |
|---|---:|---:|---|
| `Femur_L` | `[63868, 64783)` | 915 | 左股骨整体；股骨头诊断域是其中的稀疏子集 |
| `Femur_R` | `[64783, 65698)` | 915 | 右股骨整体；股骨头诊断域是其中的稀疏子集 |
| `Ilium_L` | `[109898, 110847)` | 949 | 左髂骨/髋臼所在整体 mesh |
| `Ilium_R` | `[110847, 111796)` | 949 | 右髂骨/髋臼所在整体 mesh |

供局部高亮使用的冻结域不是连续 range，应优先使用 calibration 的 global vertex IDs：

| side | femoral head | acetabulum cap | broader pelvis domain |
|---|---:|---:|---:|
| left | 69 fit + 69 validation | 19 fit + 19 validation | 238 fit + 237 validation |
| right | 69 fit + 69 validation | 19 fit + 19 validation | 238 fit + 237 validation |

这些 ID 位于 [anatomical_calibration_v1.npz](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1/anatomical_calibration_v1.npz) 的 `domain_vertex_ids` 打包域中，也可通过以下方式提取；同一拓扑下可直接索引 candidate pose 顶点：

```python
from projects.genesis_ue_sync.anatomy_retarget.v8_artifacts import load_source_operator
from projects.genesis_ue_sync.anatomy_retarget.anatomical_calibration_v1 import (
    load_anatomical_calibration_v1,
)

operator = load_source_operator("outputs/anatomy_retarget/v8_candidates/rebuild_012/source_operator_v8")
calibration = load_anatomical_calibration_v1(
    "outputs/anatomy_retarget/v8_candidates/chain_retarget_v1_node1_006/anatomical_calibration_v1",
    operator=operator,
)
asset = operator.template_asset
names = [str(name) for name in asset.source_mesh_names]
ranges = {name: tuple(map(int, asset.source_vertex_ranges[i])) for i, name in enumerate(names)}

left_ilium_all = range(*ranges["Ilium_L"])
left_head = calibration.domains["left/femoral_head.validation"]
left_socket = calibration.domains["left/acetabulum.validation"]
left_pelvis = calibration.domains["left/pelvis.validation"]
```

`fit` 和 `validation` 域互不重叠；若渲染要突出实际参与 V12 指标的区域，用 `*.validation`，若要显示校准的全域支撑，再叠加 `pelvis.validation` 或 `Ilium_L` 整体范围。

## V13 冻结骨面局部三角诊断

进一步的实际查询由 [`audit_capture_articular_surfaces_v13.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/cli/audit_capture_articular_surfaces_v13.py) 完成，结果见 [`report.json`](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_capture_articular_surfaces_20260908_001/report.json) 和 [完整短表](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_capture_articular_surfaces_20260908_001/README.md)。它查询 `validation` 域顶点到对侧完整骨 mesh 三角面的点到三角距离；negative signed 表示查询点落在闭合对侧骨内部，阈值为 `0.1 mm`。这仍然是局部采样诊断，不是全三角面无相交证明。

六个 cell 全部完成，`failures=0`。12 个被检查的 source bone mesh 在每个 cell 都是 `watertight=true`、`winding_consistent=true`、`signed_available=true`；72 次质量记录的 signed volume 范围为 `3.554235e-05` 到 `4.200950e-04 m³`。每个 cell 的 source/candidate 查询骨段逐 mesh 精确相同，因此只计算一份；数值同时代表 V12e `source_vertices` 的 source baseline 和 V13 candidate 的骨段，不能解释为软组织 attachment 的改变量。

下表是 primary direction 的实际绝对值。髋的 `u` 为 unsigned `[closest/p05/median/p95/max] mm`，末尾为 `negative signed >0.1 mm` 计数；膝和肘列为各 primary cap 查询的 unsigned max 与 negative 计数（medial/lateral 或 radius/ulna 顺序）。所有更完整的 signed/unsigned 分位数、反向查询、domain IDs、face IDs 和质量字段都在 JSON 中。

| cell | left hip `u` / neg | right hip `u` / neg | left knee max / neg | right knee max / neg | left elbow max / neg | right elbow max / neg |
|---|---|---|---|---|---|---|
| 213328/tpose | 0.167/0.361/5.119/28.297/31.667 / 14 | 0.122/0.532/6.481/30.475/33.419 / 17 | 27.909/31.760 / 9/1 | 30.331/33.190 / 7/0 | 52.814/42.784 / 0/3 | 28.971/27.319 / 4/10 |
| 213328/pose_213328 | 0.084/0.426/5.686/28.724/32.309 / 11 | 0.031/0.351/5.954/31.421/32.462 / 12 | 19.266/20.607 / 9/11 | 28.125/32.763 / 12/1 | 63.839/50.403 / 0/0 | 28.547/26.278 / 2/8 |
| 213328/pose_213712 | 0.059/0.456/6.638/26.710/31.977 / 17 | 0.260/0.410/10.033/25.193/27.702 / 20 | 20.684/31.062 / 13/8 | 25.965/32.098 / 16/8 | 62.834/55.650 / 0/0 | 23.073/17.718 / 4/12 |
| 213712/tpose | 0.163/0.432/5.110/28.245/31.706 / 14 | 0.045/0.498/6.461/30.502/33.098 / 16 | 28.171/32.475 / 10/1 | 30.640/33.516 / 8/0 | 48.271/38.374 / 0/3 | 35.513/27.211 / 3/10 |
| 213712/pose_213328 | 0.009/0.443/5.631/28.620/32.424 / 11 | 0.048/0.402/5.965/31.331/32.629 / 12 | 17.980/20.453 / 9/11 | 28.328/32.965 / 13/1 | 59.026/45.423 / 0/0 | 34.684/26.144 / 7/6 |
| 213712/pose_213712 | 0.068/0.479/6.699/26.779/31.780 / 17 | 0.123/0.445/10.052/25.168/27.754 / 20 | 21.537/31.967 / 13/8 | 26.640/32.856 / 15/7 | 60.358/51.484 / 0/0 | 23.741/18.228 / 5/12 |

这些结果提供了轴距之外的局部骨面证据：例如 `213328/pose_213328` 左髋 `femoral_head.validation → Ilium_L` 的 signed 最小值为 `-2.930 mm`，11/69 个股骨头域点落入 Ilium；同一 cell 左肘肱骨域到桡骨/尺骨的 unsigned 最大距离为 `63.839/50.403 mm`，但该查询没有负 signed 点。它们分别支持“采样点有局部进入对侧闭合骨”和“采样域到对侧骨三角面的距离很大”的诊断；仍不能单独把整段骨面判定为全局穿透或脱臼。

## Raw 142 因果基线复核

为区分原始 Blender/source geometry 已有的问题和 V12e retarget 新增的问题，复用了同一 CLI，生成 [raw-only report](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_raw_articular_surfaces_20260908_001/report.json) 和 [raw-only short table](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_raw_articular_surfaces_20260908_001/README.md)。输入来自 001 旧 comparison 的 `raw_vertices`；faces、pose、skin、SMPL-X joints 和冻结 domain 完全不变，只把 source/candidate 两个 geometry 字段都替换成 raw geometry。四个 cell 全部完成，12 个骨 mesh 的质量检查仍全部可做 signed 查询。

最终 candidate 诊断的 queried bone 与 source baseline 逐 mesh 精确相同，所以这里 candidate 数值实际上就是 V12e source geometry；raw 行才是 142 source posed geometry。表中髋和膝的格式是 `最负 signed mm / negative signed < -0.1 mm 点数`；肘是 unsigned `closest/p05 mm`，顺序为 candidate → raw。

| cell | left hip candidate vs raw | right hip candidate vs raw | medial knee L candidate vs raw | medial knee R candidate vs raw | left elbow humerus→radius candidate→raw | left elbow humerus→ulna candidate→raw |
|---|---:|---:|---:|---:|---:|---:|
| 213328/tpose | -4.074/14 → -4.057/14 | -4.114/17 → -4.135/16 | -6.028/9 → -6.834/10 | -6.413/7 → -6.130/6 | 8.654/17.416 → 0.077/5.376 | 0.336/4.640 → 0.210/0.946 |
| 213328/pose_213328 | -2.930/11 → -3.001/11 | -4.367/12 → -4.301/12 | -12.902/9 → -13.188/28 | -8.025/12 → -7.699/11 | 12.890/19.084 → 0.147/3.818 | 2.542/8.424 → 0.113/0.802 |
| 213712/tpose | -3.992/14 → -3.979/13 | -4.036/16 → -4.045/15 | -6.091/10 → -6.755/10 | -6.516/8 → -6.209/6 | 6.511/12.160 → 0.077/5.351 | 0.004/4.709 → 0.209/0.941 |
| 213712/pose_213328 | -2.969/11 → -3.034/11 | -4.301/12 → -4.236/12 | -13.266/9 → -13.508/28 | -8.240/13 → -7.873/11 | 7.128/14.712 → 0.080/3.773 | 1.012/5.063 → 0.189/0.752 |

因果解释应保持克制：

* 髋头→髂骨的最负 signed 值和计数在 raw 与 V12e 间大体接近，不能据此说 retarget 新增了明显的髋穿入；两者都可能保留 source rest 的局部问题。
* 内侧膝的 raw 也有明显负 signed 点，尤其 `pose_213328` 左侧 raw 为 `-13.188 mm / 28` 点，说明膝局部进入并非全部由 V12e 新增。右膝在两个 pose 中 V12e 最负值/计数略差，属于需要继续看图和完整关节面关系的局部回归信号。
* 左肘是最清楚的 retarget 分离证据：在 `213328/pose_213328`，V12e 左肱骨域到桡骨的 closest/p05 为 `12.890/19.084 mm`，raw 为 `0.147/3.818 mm`；到尺骨为 `2.542/8.424 mm`，raw 为 `0.113/0.802 mm`。这支持“该姿态的左肘骨端间距主要在 retarget 后被拉开”的判断，也与 Genesis 图中可见的左肘错位相符。

最后，这里使用的髋 `femoral_head.validation` 域覆盖整个股骨头采样区，反向膝/肘 cap 也不是单一接触点；因此 `max unsigned` 是域中最远采样点到目标三角面的最大值，不能当作关节 gap。实际近接关系应优先看 closest/p05/median、signed 负值计数和对应 query 点/face，同时保持“顶点→triangle 局部采样”这个证据边界。

## V11 anchored → V12e forearm reseat 阶段因果审查

为把“V11 rest fit 已经造成的分离”和“V12e 前臂 mesh-only reseat 新增的分离”分开，新增了 [`audit_elbow_stage_v13.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/cli/audit_elbow_stage_v13.py)。它只审查 subject `213328` 的 `tpose` 与自身 `pose_213328`，两阶段都使用 `pose_whole_chain_vertices_v10`，并复用本报告的 frozen elbow validation 域和完整 `Radius_L/Ulna_L` 三角面。可直接交给 Genesis renderer 的 NPZ 在 [stage audit output](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_elbow_stage_audit_20260908_001/)；其中 `comparisons/*.npz` 的 `source_vertices=V11`、`candidate_vertices=V12e`，`skin_vertices/smplx_joints/pose` 由同一输入 cell 原样共享。

实际数值如下，单位为 mm；`negative` 是 signed distance 小于 `-0.1 mm` 的查询点数：

| pose | 方向 | V11 closest/p05 | V12e closest/p05 | V11 negative | V12e negative |
|---|---|---:|---:|---:|---:|
| tpose | 肱骨→桡骨 | 0.470 / 5.081 | 8.654 / 17.416 | 2 | 0 |
| tpose | 肱骨→尺骨 | 0.322 / 0.820 | 0.336 / 4.640 | 5 | 3 |
| pose_213328 | 肱骨→桡骨 | 6.930 / 11.432 | 12.890 / 19.084 | 0 | 0 |
| pose_213328 | 肱骨→尺骨 | 0.821 / 1.922 | 2.542 / 8.424 | 3 | 0 |

这组阶段对照支持以下有限因果结论：V11 rest fit 已经留下了局部肘面关系问题（例如 T-pose 桡骨 closest `0.470 mm`、2 个负 signed 点，尺骨 closest `0.322 mm`、5 个负 signed 点），所以不能把所有问题都归给 V12e；但 V12e reseat 对桡骨是主要新增分离来源，T-pose 的 closest/p05 增量为 `+8.184/+12.335 mm`，采集 pose 为 `+5.960/+7.652 mm`。尺骨增量较小但仍存在，采集 pose closest/p05 增量为 `+1.721/+6.502 mm`。negative 点数减少只能表示该局部采样的 inside 标记变化，不能解释为关节关系改善。

V12e 的左前臂刚体重座位移也单独从对应 mesh 顶点拟合记录。V11 的材料肘支点定义为 `elbow/left/{humerus,radius,ulna}.validation` 顶点均值；把它代入每个 V11→V12e 刚体变换后，支点位移和材料点位移为：

| mesh | V12e controller | rotation | 肘材料支点位移 | validation/mesh 点 p95 / max |
|---|---|---:|---:|---:|
| Radius_L | Forearm_Twist_L | 12.603° | 20.502 mm | 23.405 / 24.352 mm |
| Ulna_L | Forearm_Bone_L | 4.234° | 22.774 mm | 23.366 / 23.788 mm |

这里的 `20.502/22.774 mm` 是材料支点相对 V11 几何的实际变换结果，不是 `Elbow_Rot_L` 控制器 origin 的 FK 位移。逐 pose 的 elbow validation 域点位移在 JSON 的 `forearm_reseat.per_pose_domain_displacement` 中，T-pose 与自身采集 pose 均约为 `Radius_L 22.3 mm`、`Ulna_L 22.8 mm` 的中心位移。针对 `Elbow_Rot_L`、`Forearm_Bone_L`、`Forearm_Twist_L` 的 `B_prefit/B_final/C_bone/target_local_bind/inverse_bind` 字段均保持逐项相等；全局 bind digest 变差来自 V12e 同时重座的手/足末端字段，不能误读为左肘 bind 改动。完整证据见 [stage report.json](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_elbow_stage_audit_20260908_001/report.json) 和 [stage README](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_elbow_stage_audit_20260908_001/README.md)。

因此，左肘的主要新增分离发生在 V12e 的前臂 mesh-only reseat；V11 anchored rest fit 是已有基线问题，并非唯一来源。该结论仍限于两个 frozen validation 域的顶点→triangle 局部采样，不能升级为整段骨面无相交或生理脱臼判定。

## V12e → joint-lock 骨面复核

父流程已生成四个真实 pose 的 common fixed-pivot joint-lock NPZ。这里复用同一冻结骨面查询，输入为 [`audit_elbow_joint_lock_v13.py`](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/cli/audit_elbow_joint_lock_v13.py)，结果见 [joint-lock report](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_elbow_joint_lock_audit_20260908_001/report.json) 和 [joint-lock short table](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v13_elbow_joint_lock_audit_20260908_001/README.md)。每一行都是 source=`V12e_before_common_lock`、candidate=`common_Radius_Ulna_fixed_V11_elbow_pivot_rotation`；只审查左肱骨→桡骨/尺骨，单位为 mm。

| pose | 方向 | V12e closest/p05 | lock closest/p05 | V12e signed min | lock signed min | V12e negative | lock negative |
|---|---|---:|---:|---:|---:|---:|---:|
| tpose | 肱骨→桡骨 | 8.654 / 17.416 | 0.192 / 5.170 | +8.654 | -5.648 | 0 | 2 |
| tpose | 肱骨→尺骨 | 0.336 / 4.640 | 0.036 / 0.947 | -2.473 | -1.565 | 3 | 5 |
| pose_213328 | 肱骨→桡骨 | 12.890 / 19.084 | 6.866 / 11.502 | +12.890 | +6.866 | 0 | 0 |
| pose_213328 | 肱骨→尺骨 | 2.542 / 8.424 | 0.851 / 2.106 | +2.542 | -1.842 | 0 | 3 |
| heldout_sitting | 肱骨→桡骨 | 1.972 / 10.899 | 0.099 / 2.622 | +1.972 | -1.418 | 0 | 2 |
| heldout_sitting | 肱骨→尺骨 | 0.593 / 1.631 | 0.050 / 0.882 | -2.942 | -5.838 | 5 | 16 |
| heldout_kicking | 肱骨→桡骨 | 8.177 / 16.746 | 0.469 / 3.831 | +8.177 | -3.271 | 0 | 2 |
| heldout_kicking | 肱骨→尺骨 | 0.217 / 2.208 | 0.472 / 1.185 | -2.027 | -2.223 | 3 | 4 |

共同锁定旋转使无符号 closest/p05 在四个动作中大多下降，尤其桡骨；但 signed negative 点在 T-pose 和 heldout 中增加，说明“距离更近”同时伴随局部采样点进入对侧闭合骨，不能把这组改善称为关节面修复。`pose_213328` 桡骨仍有 `6.866/11.502 mm` 的 closest/p05，说明该动作的分离也没有被消除。

这项 lock 是骨-only 实验，软组织没有随 Radius/Ulna 共同变换。因此它不能用于证明血管、神经或器官联动保持。joint-lock report 的同一 forearm bone→skin containment 还显示 heldout 外皮回归：`heldout_sitting` 的最大穿出从 `21.697 mm` 增至 `30.882 mm`，`heldout_kicking` 从 `24.024 mm` 增至 `31.106 mm`。这两个外皮数字来自锁实验的骨穿出指标，不能用局部关节面距离下降抵消。当前证据应拒绝把 common lock 作为可接受生产方案。

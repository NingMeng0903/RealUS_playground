# V13 独立 RGB 图审（Genesis renderer）

日期：2026-09-08  
审查人：独立 evidence audit（read-only）

本审查只依据 Genesis runtime 生成的原始 RGB PNG，以及对应的 V13 comparison NPZ 的逐顶点差分。没有把 manifest 里的 `pass: true` 当作解剖学通过条件。

审查的两组原始图像位于：

- `outputs/anatomy_retarget/v13_genesis_review_20260908_002/flex_knee_elbow_120/`
- `outputs/anatomy_retarget/v13_genesis_review_20260908_002/heldout_sitting/`

每组均直接查看了 `before_v12e/rgb/*.png` 和 `candidate/rgb/*.png` 的 whole、双侧 elbow、双侧 knee、ankle、foot 图，以及 comparison 图。渲染 manifest 标注 `publishable: false`；图像里的每张技术渲染记录虽然有 `pass: true`，该字段只说明 RGB/depth/segmentation 产出，没有证明软组织在皮肤内或关节交接正确。

## 观察结果

### Flex knee + elbow 120°

- `whole_ap` 能看到完整骨架、双肘抬起和双膝深屈；candidate 的骨架轮廓与 before 基本重合。candidate 的金色神经在躯干、双上肢和下肢大面积可见，而 before 中多数被遮挡。
- 双侧 elbow lateral 图可看到 candidate 神经线贴近或跨过肩胛、肋骨和肱骨投影；局部 vessel 在若干视角中消失于骨后，单凭 RGB 不能把“看不见”判成没有或已正确包覆。
- `left_knee_ap/lateral`、`right_knee_ap/lateral` 在深屈姿态中被折叠的足/另一条腿前景遮挡；其中 lateral 图实际占据大面积的是足和交叉小腿。当前这些图不能证明膝关节的 hinge、骨端接触或软组织连续性。
- ankle/foot 图能看到 candidate 的金色线和红色线沿远端肢体走向，但线与暗色皮肤轮廓的间隙没有可量化的表面距离；因此不能据此宣称足踝包覆通过。

### Held-out sitting

- `whole_ap` 中 candidate 的骨架姿态仍保持坐姿，但 candidate 的软组织可见性显著少于 before：多数躯干神经和髋周血管消失，仅剩零散红色段。这个差异在左右并不对称。
- 左 knee AP/lateral 中，before 有髋周和大腿的血管/神经网络，candidate 多数区域只剩骨和少量红线；图中还混入手和另一侧肢体。右 knee 也有同样问题，不能作为闭合膝关节的独立证据。
- 左 ankle/foot candidate 仍可看到足背/足底附近的红线，但暗色皮肤包络与骨脚的边界关系不稳定；右 ankle/foot candidate 几乎只剩骨和极少数红色末端，暗色皮肤轮廓仍在骨脚旁边，视觉上存在明显的皮肤—骨脚错位风险。这是本批图像最严重的失败区域。
- sitting 的 elbow 局部图同样显示 before 与 candidate 的神经/血管可见性大幅变化；这更像近表面几何移动后的深度遮挡变化，不能作为联动关系已保持的证据。

## 逐顶点差分核对

比较文件：

`outputs/anatomy_retarget/v13_material_20260908_001/subjects/subject_213328/comparisons/{heldout_sitting,flex_knee_elbow_120}.npz`

两个文件中 bone 顶点逐位完全相同；vessel 和 nerve 顶点并不相同。candidate 相对 source 的欧氏移动如下（mm）：

| pose | tissue | mean | p95 | p99 | max | >1 mm |
|---|---:|---:|---:|---:|---:|---:|
| held-out sitting | vessel | 0.761 | 4.940 | 16.861 | 22.759 | 7.85% |
| held-out sitting | nerve | 0.590 | 3.461 | 14.396 | 19.827 | 10.90% |
| flex knee/elbow | vessel | 0.777 | 5.453 | 16.868 | 22.960 | 7.91% |
| flex knee/elbow | nerve | 0.615 | 3.698 | 14.469 | 19.900 | 10.39% |

**渲染证据修正（2026-09-08）**：上面关于“神经忽隐忽现由候选几何移动导致”的归因不能保留为结论。该批 `v13_material` NPZ 的 vessel/nerve 顶点差分是真实的，但 RGB 可见性还受到 Genesis 透明材质路径影响；仅凭 before/candidate 的可见性不能把差异归因到形变或联动改善。

独立 station probe 使用完全相同的 `skin.obj`、`baseline_bones.obj`、`baseline_vessels.obj`、`baseline_nerves.obj`、相同 `pelvis_ap` 相机和相同材质，在两个全新 Genesis runtime 中重复渲染。原 alpha（skin 0.18、vessel/nerve 0.92）两次 RGB 的 SHA-256 不同，`alpha092_a -> alpha092_b` 有 55,037 个变化像素、最大通道差 201、平均绝对差 9.401；红色像素由 110 变成 33,213。查看原 PNG 时，第一张几乎没有主动脉/大血管，第二张则显示完整红色主动脉和盆腔血管，几何和相机完全相同。换序同一透明材质也改变 23,807 个像素。相同几何下将 vessel/nerve alpha 改为 1.0 后，两次 RGB 只有 3 个像素出现 1 级差异，最大差 1，且两张都显示主动脉和神经网络。

probe 原图和 manifest：`outputs/anatomy_retarget/v13_alpha_probe_20260908_001/{alpha092_a,alpha092_b,alpha092_reordered,opaque_tubes_a,opaque_tubes_b}/rgb/pelvis_ap.png`。因此旧 `v13_genesis_review_20260908_002`/`003` 中“candidate 神经变可见”或“神经可见性证明几何联动变化”的说法应撤回。上表中的几何差分与独立截面中发现的骨—管路碰撞仍然有效，但必须和透明渲染可见性分开解释；透明 PNG 不能作为软组织包覆或联动通过证据。

源码核对支持该 probe：`GenesisPlatformRuntime.add_mesh_entity` 将 RGBA 传入 `gs.surfaces.Default(color=...)`（`src/projects/genesis_ue_sync/sim_platform/simulation/runtime.py:321-338`）；Genesis 的 vertex color 保留 alpha（`genesis/utils/mesh.py:235-239`），alpha 未满 255 时 Pyrender 设为 `BLEND`（`genesis/ext/pyrender/mesh.py:250-255`）。NVIDIA JIT 路径明确记录 `GL_BLEND` 会产生非确定性结果（`genesis/ext/pyrender/jit_render.py:515-521`），而透明实体仍按透明节点排序后绘制（`genesis/ext/pyrender/scene.py:602-623`）。这正解释了同几何的 RGB 差异；将内部 vessel/nerve 设为 opaque 消除了该证据污染，但 skin 仍为 alpha 0.18，所以后续审查仍需结合无遮挡几何/截面检查。

## 新 capture Genesis 包的独立复核

复核包：`outputs/anatomy_retarget/v13_capture_genesis_review_20260908_001/`。我直接查看了两个 subject 的 T-pose 与各自采集 pose 的 `candidate/rgb/whole_ap.png`、`smplx_skin/rgb/whole_ap.png`、`pelvis_ap`、左右 `hip_oblique`、左右 `knee_ap/lateral`、左右 `elbow_ap/lateral`，并对照了 comparison PNG。该包 manifest 标出 `internal_material_alpha: 1.0`、skin alpha 仍为 0.18，局部图使用关节点前后各 160 mm 的 near/far 裁剪。

### subject 213328

- **T-pose**：`whole_ap` 中骨架和红/黄内部组织大体落在 SMPL-X 皮肤轮廓内，能确认整体朝向和尺度可比；这张图不能确认皮肤截面内外距离。左右 `hip_oblique` 都出现股骨头与髋臼开口之间的明显投影间隙/偏置，存在髋关节未保持球窝关系的可疑证据。`hip_posterior` 被骨盆板遮住髋臼，不能用来补足 3D 接触结论。左右膝 AP/lateral 的股骨髁、胫骨平台大体相向，未看到明显整段脱离；血管/神经在膝后及侧方贴近骨面，仍没有无遮挡距离量测。左右 elbow lateral 的长骨和管路在视图边缘出现平整、锯齿状截断，这是 near/far 裁剪，不应解释为骨端断裂；关节中心仍有管路跨过骨面，2D 图不足以判断穿骨。
- **own pose**：`whole_ap` 正确显示单腿抬起/屈曲的采集姿态，内部骨架随姿态走向，未看到全身级别的明显错位。左 `hip_oblique` 仍能看到股骨头—髋臼的可疑间隙；右侧同类。左 `elbow_lateral` 的放大观察显示肱骨远端与近端尺/桡骨之间有可见关节缝和轻微端部偏置，红色管路紧贴并跨过该区域；这应作为待修复警报，不能仅凭这一投影定量为穿骨。左右膝在 AP/lateral 中总体保持屈曲链条，红管主要在后外侧，但局部相机和皮肤透明度不能证明关节间隙或组织包覆通过。

### subject 213712

- **T-pose**：`whole_ap`、骨盆和双侧膝肘的视觉模式与 213328 T-pose 基本一致。左右 `hip_oblique` 的股骨头—髋臼投影间隙仍可见，说明这不是单个采集 pose 才出现的现象；后方髋图仍被骨盆遮挡。膝关节轮廓大体对齐，但血管在骨后/骨侧的遮挡不允许判定 3D 关系。肘部 lateral 的平整截面仍主要是局部裁剪伪影，不能用来判骨端关系。
- **own pose**：`whole_ap` 显示双臂抬起的采集姿态，内部骨架随双臂抬高，未见全身姿态脱链。左右髋 oblique 仍有同样的球窝间隙可疑点。左右膝 AP/lateral 的骨端相向且没有明显整段脱开；管路在膝后/侧通过。左右 elbow 的弯曲关节局部可见骨端连续，但红色管路在关节附近贴/跨骨，且皮肤包络为透明显示，不能宣称软组织已完全在皮肤截面内。

四个复核 cell 中，before 与 candidate 的 bone 顶点逐位相同（用导出的 OBJ 加载后 `np.array_equal` 为真）；因此这些图不能被表述为“candidate 改善了骨骼位置”。candidate 变化主要在 vessel/nerve。另一个必须保留的限制是：局部 Genesis 图的平整边缘来自 manifest 明示的 ±160 mm 深度裁剪，任何骨端或管路端点都不能从裁剪边缘判断。opaque 内部材质解决了前述透明非确定性，但仍不能把有遮挡的 RGB 视图当作髋臼接触、肘膝 hinge、骨—管路无碰撞或软组织全包覆的证明。

## Forearm joint-lock stage 的独立复核

在 common rigid joint-lock 网格完成前，先查看了 `outputs/anatomy_retarget/v13_elbow_stage_genesis_review_20260908_001/` 的两组 stage comparison。这里右侧标签 `candidate` 实际是历史 V12e 独立 Radius_L/Ulna_L reseat，不是新的 joint-lock winner，不能把它当作最终候选。

- T-pose `left_elbow_ap` 两侧投影接近；`left_elbow_lateral` 的长骨和管路在远端被局部裁剪，不能据此判断骨端。
- own pose `left_elbow_ap` 和 `left_elbow_lateral` 中，V12e 右侧的 Radius/Ulna 近端相对肱骨远端出现更明显的间隙/偏置，红色管路在关节前后跨过骨面。该观察与“两个前臂网格独立 reseat 后交接不一致”相符，但仍是 RGB 投影警报，不是 3D 穿骨量测。
- 这组 stage 图没有验证“同一刚体、冻结肘心、无平移/缩放/弯曲”的新试验；必须等待 identity 与 common-lock winner 的独立图和 NPZ 指标，尤其要同时看 fit（T-pose、两 capture）及 held-out（sitting、kicking）。

## Common forearm joint-lock winner 的独立复核

新图包：`outputs/anatomy_retarget/v13_forearm_lock_genesis_20260908_001/`。我直接查看了 `subject_213328_joint_lock_best_{tpose,pose_213328,heldout_sitting,heldout_kicking}/comparison/{left_elbow_ap,left_elbow_lateral,whole_ap}.png`。winner 的搜索结果为总 rotvec `[0, 0, 5]` 度；图像上 own pose 的肘部投影比历史 V12e 更接近，但这只是同一相机下的 2D 投影，局部图仍有深度裁剪，不能证明 3D 骨端接触。T-pose lateral 的平整端仍是裁剪；sitting 和 kicking 的 whole/lateral 不能把软组织包覆或骨—管路无碰撞判为通过。

NPZ 复核：`subject_213328_joint_lock_best_{tpose,pose_213328,pose_213712,heldout_sitting,heldout_kicking}.npz` 中，source/candidate 的 vessel（28,360 顶点）和 nerve（26,977 顶点）均逐位 bit-exact；organ、heart、connective 也均未改变。变化只有 bone 的 893 个顶点。因此这次 common rigid lock 没有让血管或神经随前臂新骨位置运输，不能称为软组织联动已保持。

报告中的皮肤外侧结果也拒绝把数值 winner 当成普适通过：

| pose | V12e forearm max outside | common-lock max outside | V12e outside area | common-lock outside area |
|---|---:|---:|---:|---:|
| T-pose | 0.000 mm | 3.684 mm | 0.00% | 3.77% |
| 213328 | 0.000 mm | 0.000 mm | 0.00% | 0.00% |
| 213712 | 0.000 mm | 5.078 mm | 0.00% | 8.36% |
| held-out sitting | 21.697 mm | 30.882 mm | 41.29% | 46.17% |
| held-out kicking | 24.024 mm | 31.106 mm | 60.09% | 47.48% |

kicking 的外侧面积比例虽下降，最大外侧深度反而增大；sitting 的面积和最大深度都回退。故该 winner 不能满足跨 pose 的皮肤包覆要求。

### Pivot 来源更正

`evaluate_forearm_joint_lock_v13.py` 的实际 pivot 是 `_elbow_center(calibration, v11.vertices_final)`：对 `elbow/left/{humerus,radius,ulna}.validation` material 域顶点去重后求均值。report 的 pivot 为 `[0.4437127, 0.0335775, -0.0655374] m`。它不是 `B_final[Elbow_Rot_L,:3,3]`；后者为 `[0.4333662, 0.0123929, -0.0632102] m`，相差约 23.7 mm。代码注释中的 `Elbow_Rot_L origin` 与实际实现不一致，审查应以 material-domain mean 和 report 字段为准。

## 独立结论

本批 Genesis RGB 图审不通过当前 candidate。最明确的失败证据是坐姿右足/右踝的皮肤—骨脚错位和软组织消失，其次是深屈膝局部视图无法隔离膝关节，以及 candidate 前臂的骨—管路相交风险（该风险在独立截面图/几何检查中可见，RGB 遮挡图不能洗掉它）。

当前图像可以证明“渲染管线能在两个 pose 产生可对比的 whole/局部图”，不能证明用户要求的任意 pose 一次 retarget 后仍满足：骨端和窝的关系、关节 hinge、血管/神经与骨骼联动、所有软组织位于合理皮肤截面内。应继续把 candidate 视为 reject，直到有无遮挡较少的双侧膝/足踝证据和逐组织皮肤内外/骨碰撞证据。

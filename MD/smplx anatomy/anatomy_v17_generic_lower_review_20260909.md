# 通用下肢 retarget：实际实现与图审记录（2026-09-09）

状态：开发候选，尚未全身通过。用户最新优先级是相同算法适配新 β、一次编译后由 θ 直接驱动，先消除股骨顶出膝部和成片骨/软组织出皮；不为采集编号、某个 β 或测试动作添加专用分支。小范围合理骨突出由真实图审判断。

**最新统一入口：[真实 Genesis 图审页（374 张静态图、两段连续视频及失败报告）](/home/camp/anatomy_retarget_outputs/v17_coupled_review_20260909_001/index.html)。** 主 agent 和独立 agent 均实际查看，不以代码测试或日志代替模型判断。当前相关代码检查为 57 项通过，几何验收仍未通过。

## 当前进展：通用编译与固定姿态响应

本轮新增的 `coupled_leg_articulation_v17.py` 使用相同的 916 个规则姿态，为当前 β 离线拟合髋—膝—踝联动。采样由固定角度网格和固定 Sobol 组合产生，不读取采集编号、AMASS 帧或手工 β 表。24 个超出原 source knee/ankle 响应范数的训练候选被明确排除，没有裁角。

新的分阶段配置为 `--rest-fit-poses tpose --bake-articulation`：先拟合 rest/bind/骨干长度，再拟合局部姿态响应。`fit_leg_pose_v17` 固定当前股骨头点，通过髋、膝局部旋转修正下肢；踝局部反向旋转保留原来的足部全局方向。全部子树和所有接收原权重的骨、动脉、静脉、神经共同运动。离线拟合同时使用实际目标 β 的 SMPL-X 皮肤，对股骨、胫骨、腓骨和髌骨施加两轮骨面约束。

运行包保存原 235 控制器、原逐顶点权重及固定姿态响应系数。运行阶段只有 SMPL-X θ 驱动、插值、FK 和原 LBS，不访问 Blender、不查询最近骨/皮表面、不重新优化。这里烘焙的是“原权重＋辅助联动＋固定校正”的运行包；仅一张 55 骨线性权重表不能表达原来的全部非线性辅助关节响应。

早期独立轴相加的烘焙没有保存混合旋转效果，已经停止用于新编译。第一轮耦合 RBF 又因核过宽产生最高约 5.47° 的训练重建误差；在不改任何骨面拟合结果的情况下，把所有 β 共用的核宽修正后，固定采样矩阵条件数由约 1.4e8 降到约 3.7e4，三个体型的训练重建误差均低于 0.00003°。训练点重建正确仍不代表未见动作中的骨骼正确。

已保存的相同算法候选：

- [213328 运行包](/home/camp/anatomy_retarget_outputs/v17_coupled_baked_213328_20260909_002/compiled)
- [213712 运行包](/home/camp/anatomy_retarget_outputs/v17_coupled_baked_213712_20260909_002/compiled)
- [β=[1.5,0,…] 运行包](/home/camp/anatomy_retarget_outputs/v17_coupled_baked_beta_axis0_plus_20260909_002/compiled)
- [真实 Genesis 连续视频：213328 坐起](/home/camp/anatomy_retarget_outputs/v17_coupled_video_213328_sitstand_20260909_002/genesis_motion.mp4)
- [连续视频独立图审](/home/camp/anatomy_retarget_outputs/v17_coupled_video_213328_sitstand_20260909_002/visual_audit.json)
- [两体型独立全表面复测](/home/camp/anatomy_retarget_outputs/v17_coupled_independent_review_20260909_001/independent_review.md)

主 agent 实际打开了连续视频保存的 frame 425 五视角图，以及直接拟合试验的左膝、左足不透明图。独立 agent 实际查看视频的 353、413、425、569、785 帧及两体型测量。结果支持“明显改善”，不支持“已经通用通过”。

### 新候选的明确失败范围

下表为近似值，统一采用完整 `WINDING_NUMBER` 复测；这里的 before 是同一 β 的 neutral-rest、尚未施加新姿态校正的运行结果，不能与下方历史表的 canonical 列混用。

| 213328 姿态 | 12 个主要下肢骨最大出皮：before→baked | 动脉最大出皮：before→baked | 静脉最大出皮：before→baked |
|---|---:|---:|---:|
| 膝 90° | 11.79→7.70 mm | 14.80→6.40 mm | 5.20→2.90 mm |
| 髋 60°/膝 90° | 11.20→6.80 mm | 31.20→6.90 mm | 23.40→2.20 mm |
| 坐起 frame 422 | 30.20→7.80 mm | 46.00→15.10 mm | 42.30→8.70 mm |
| 站起 frame 792 | 11.90→15.00 mm | 13.00→13.40 mm | 5.10→6.90 mm |
| 213328 自身采集动作 | 35.98→27.03 mm | 38.70→21.00 mm | 33.10→14.90 mm |

表中保留回退项。frame 422 的主要骨已没有 >10 mm 点，但足骨仍有 823 个 >10 mm 点；213328 自身采集动作仍有 2,698 个下肢骨点 >10 mm。213712 使用同样算法后也有采集姿态和足部失败。因此不能把规则屈膝改善概括为“任意 θ 已解决”。

坐起 440/440 原始帧可直接回放，未二次编译；连续最大主要骨出皮仍约 17.33 mm、血管约 17.76 mm。新冻结的 BABEL 541 顺时针行走 290 帧中，259 帧能运行，31 帧因髋伸展略超过当前 +30° 烘焙范围被明确拒绝。这个普通动作暴露了支持域不足，不能静默裁成 30°，也不能把成功率写成 100%。[转弯验证报告](/home/camp/anatomy_retarget_outputs/v17_coupled_native_213712_turn_20260909_002/report.json)

独立检查排除了 frame 422/792 的 toe θ 遗漏：两帧的 toe 输入本来为零，踝与全部趾子树的旋转一致，位置残差相差仅约 0.00002 mm。余下足部问题来自共同踝位置残差和 rest 足部与目标皮肤的形状/位置不匹配，不应再给单独脚趾加测试帧补丁。

普通髋伸展的支持域不足随后通过**增加统一规则采样**修复：新协议包含髋 +60°、膝 −30°，合计 1,159 个姿态；两个采集 β 各复用原来 915 个非中立位离线 fit，仅新增 243 个规则 fit。旧包仍读取自身保存的边界，回放行为没有被代码常量静默改变。`_003` 包对 BABEL 541 的 290/290 帧均可直接求值，未裁角；这一成功只证明支持域问题已修复，不证明骨/软组织已经合理。[复测报告](/home/camp/anatomy_retarget_outputs/v17_coupled_native_213712_turn_20260909_003/report.json)

β=[1.5,0,…] 的同算法全表面检查仍失败：部分动脉最大出皮约 39 mm、静脉约 40 mm，部分神经在 T-pose 已出皮约 36 mm。它直接说明当前下肢 rest 拟合没有覆盖全身软组织的体型映射，不能宣传为任意 β 完成。[较大体型测量](/home/camp/anatomy_retarget_outputs/v17_coupled_measurement_beta_axis0_plus_20260909_002/lower_geometry_measurement_v17.md)

闭式两连杆试验也已拒收：它保持骨段长度并使多数踝点误差接近零，却把 213328 采集动作的主要骨最大出皮从 27.03 mm 推到 43.67 mm，213712 站起末段从 16.67 mm 推到 44.51 mm。实际 Genesis 图确认退步，`analytic_leg_closure_v17.py` 仅保留为未接入运行时的失败诊断。不能用关节点对齐代替骨面和组织位置验收。[拒收记录](/home/camp/anatomy_retarget_outputs/v17_analytic_leg_closure_review_20260909_001/review.md)

股骨头中心的独立排查没有发现厘米级脱窝漂移：头部表面域均值与近似球心相差 3.3–3.9 mm，球心距原 `Femur_Rot` bind 原点仅 0.3–0.4 mm；按近似球心测得所查动作中相对骨盆漂移约 0.03–0.34 mm。球拟合本身 RMS 约 0.9–1.0 mm，不能当作精确解剖真值。这些毫米级差异不能解释当前远端 10–40 mm 的出皮，暂不为它再添加局部补丁。[只读诊断](/home/camp/anatomy_retarget_outputs/v17_head_center_diagnosis_20260909.json)

### 按最新优先级保留的后续实现顺序

1. 先保证父链位置传递及普通动作的支持域，检验保存后的姿态响应，避免只检验每帧离线优化结果。
2. 完成足部整体 rest 适配及与踝端的接合，再处理上肢、骨盆/胸廓宽度与体型比例；使用同一骨图规则，不增加采集编号分支。
3. 大骨位置稳定后，以骨和皮肤双边界共同运输动静脉、神经和内脏。当前 V17 仅保留原有接收权重的联动，尚未完成全身软组织/内脏的通用 β 体积适配。
4. 同一个 β 编译一次，交叉回放采集动作、坐起、行走和转弯。少量合理骨面暴露按多视角图审处理；大块股骨、整片足骨或成束软组织出皮仍判失败。

姿态响应烘焙的研究依据可参照 Lewis 等的 [Pose Space Deformation](https://doi.org/10.1145/344779.344862)：在姿态参数空间中保存并插值局部校正。本文实现没有因此获得内部无穿透保证，仍以本模型的实际渲染和复测为准。

以下部分保留较早的 station-only 阶段记录，供追溯失败原因。

## 已实现的通用路径

`generic_lower_compile_v17.compile_lower_subject` 从当前十维 β 生成实际 SMPL-X 皮肤和关节，使用同一个 142 来源运动参考、同一套固定姿态和同一个目标函数，自动求解双侧膝/踝站点。输入接受十个有限 β，不使用旧 V8 的目标 β 白名单。V8 只在固定源原点建立一次参考。

`lower_chain_rest_v17.LowerChainRestMapV17` 将目标 rest、bind 位置、bind 方向、平移响应和局部旋转帧一起更新。股骨和胫腓骨只改变骨干轴向长度，允许范围仍为 ±10%；标定骨端保持刚体、截面粗细保持不变。整个踝足子树共同运输，所有混合权重的血管/神经顶点按原逐顶点权重受到同一组变形场影响。142 的面、顶点顺序、235 控制器、层级和逐顶点权重保持。

编译使用 T-pose、膝屈 30/60/90°、髋屈 30°+膝屈 45°、髋屈 60°+膝屈 90°。采集动作和 AMASS 录制不作为这个拟合器的输入。算法可接受新 β，不意味着当前下肢阶段已经完成胸廓、骨盆宽度、上肢和内脏体型适配。

保存包把 `target_betas` 与固定来源 `motion_reference_beta` 分开。运行只读取保存的 rest/bind、原权重和原响应，调用 `apply_pose(pose55, transl)`；不接触 Blender、不重新求解、不最近点重绑定。包对子运行包 manifest 与目标 β 文件校验哈希。

## 首轮失败及已修复的实际原因

首轮 `_001` 用最近三角面的法线判断内外，结果在本模型中错误地把深处皮内的点视为外部。例如 T-pose 的 `Lumbar_Nerves_L` 顶点 134807：近面法线距离为 +81.19 mm，广义 winding 判断约 −81.32 mm；膝屈 90° 的 `Femur_L` 顶点 64548：近面法线为 +43.22 mm，winding 约 −43.23 mm。T-pose 的 8,740 个拟合点中有 94 个符号分歧。

因此旧目标会把内部神经和骨点向错误方向移动。`_001` 不是可以发布的候选。不透明 Genesis 叠加确实显示右膝出现大块股骨、血管和神经出皮，不能用透明图上“链没有断开”代替出皮验收。

`_002` 改为真实三角面距离加全表面的 fast winding 内外分类。站点搜索范围由当前目标几何和参考骨长推导，修复原固定 ±60 mm 范围与初始化不一致的问题；不是裁剪 β。两个采集 β 与 β=0 使用相同 1,200 次优化预算，提前收敛者正常退出。213328 在预算耗尽时保存最佳候选，不能把它称作已收敛或已证明不可行。

## 真实渲染方式

`render_material_genesis_v13.py --opaque-skin-overlay` 在同一个 Genesis 场景中放入不透明灰色 SMPL-X 皮肤与不透明骨/血管/神经，按正常深度关系遮挡；另保留透明内部图和独立皮肤图。没有移动网格来制造可见性。

主 agent 已实际查看 `_002` 的右膝 90°、髋膝同时屈曲、坐姿左踝不透明对照。相对失败的 `_001`，右膝的大块股骨及成束血管出皮已消失于这些视角，只剩小块骨面；坐姿左踝原先大片可见胫骨已被皮肤遮挡。足部与部分血管仍有明显残余，不能宣布整个下肢通过。

- [首轮失败的不透明右膝图](/home/camp/anatomy_retarget_outputs/v17_lower_genesis_213328_opaque_overlay_20260909_001/knees_90/comparison_opaque_skin_overlay/right_knee_lateral.png)
- [修复后右膝图](/home/camp/anatomy_retarget_outputs/v17_lower_genesis_213328_20260909_002/knees_90/comparison_opaque_skin_overlay/right_knee_lateral.png)
- [修复后坐姿左踝图](/home/camp/anatomy_retarget_outputs/v17_lower_genesis_213328_20260909_002/validation_213328_sitstand_sid4336_middle/comparison_opaque_skin_overlay/left_ankle_oblique.png)

每张对照左侧 `canonical` 为固定来源经目标 root 对齐后的结果，右侧为该次候选。`_002` 图的左侧并不是 `_001` 候选，比较两次候选需看各自右侧。

## 独立测量与仍未解决的问题

213328 的 `_002` 已做完整表面量测，报告同时重新计算 canonical 与候选，采用相同 `WINDING_NUMBER` 方法，避免把方法差异算成改进。

| 姿态 | 12 个主要下肢骨最大出皮：来源→候选 | 动脉最大出皮：来源→候选 | 静脉最大出皮：来源→候选 |
|---|---:|---:|---:|
| T-pose | 16.81→16.14 mm | 11.57→4.47 mm | 8.98→3.27 mm |
| 膝 90° | 12.62→13.75 mm | 10.40→10.34 mm | 6.96→4.17 mm |
| 髋 60°/膝 90° | 12.59→10.26 mm | 33.59→13.35 mm | 30.53→6.73 mm |
| AMASS 坐起中间帧 | 16.82→9.73 mm | 37.49→21.59 mm | 33.02→18.63 mm |
| AMASS 行走中间帧 | 18.15→18.34 mm | 17.88→6.33 mm | 15.68→5.79 mm |

骨端冻结域跨上述五帧的最佳刚体残差最大约 0.00006 mm。它证明这些端区没有被拉细或拉坏，不证明关节位置已经正确。

不能只看 12 个大骨：纳入全部脚趾后，坐起中间帧仍有 3,005 个下肢骨点出皮超过 10 mm，最大 28.38 mm。足趾、跖骨和踝足方向必须继续检查。下肢集合中是否包含骨盆，以及全量脚趾范围，在各报告中明确给出，不能混用分母。

保存后的 213328 运行包已完整回放 BABEL 4336 对应坐起录制的 440 个原始帧，均能求值；全过程没有二次编译。fast winding 诊断找到 frame 422 足骨最大出皮 35.02 mm、血管 28.87 mm；主要大骨最差在 frame 792，为 15.71 mm。这是失败定位证据，不是连续动作通过证明。最差帧已导出供 Genesis 足侧/足底检查。

- [独立静态完整报告](/home/camp/anatomy_retarget_outputs/v17_lower_measurement_213328_20260909_002/lower_geometry_measurement_v17.md)
- [连续 440 帧运行报告](/home/camp/anatomy_retarget_outputs/v17_native_213328_sitstand_20260909_002/report.json)
- [冻结的录制与原始帧索引](/home/camp/anatomy_retarget_outputs/v17_frozen_motion_20260909_001/manifest.json)

SMPL-X 皮肤有 32 条边界边，winding 用于实用分类，不作为严格闭合体积证书。点距离也不替代三角面相交、管径或体积检查。当前重点是实际可见的大范围失败，不把所有细小接触升级为主线阻塞。

## 使用接口

```python
from projects.genesis_ue_sync.anatomy_retarget.generic_lower_compile_v17 import load_lower_subject
subject = load_lower_subject('/path/to/compiled')
vertices = subject.apply_pose(pose55, transl)
```

```bash
PYTHONPATH=src python -m projects.genesis_ue_sync.anatomy_retarget.cli.compile_lower_subject_v17 \
  --beta-values 0 0 0 0 0 0 0 0 0 0 \
  --reference /home/camp/anatomy_retarget_outputs/v17_canonical_reference_20260909_001 \
  --rest-fit-poses tpose --bake-articulation \
  --output /path/to/new_output
```

现阶段这是可复用的通用**下肢编译器**，不是已经完成的全身器官适配器。V16 内脏微调试验不混入当前运行包，其失败与限制保留在 [V16 记录](anatomy_v16_viscera_execution_20260908.md)。

研究与历史依据仍是 [Git 版本审计](anatomy_git_audit_20260908.md)、[SKEL](https://skel.is.tue.mpg.de/) 对皮肤参数化关节与解剖关节差异的分析，以及 [Anatomy Transfer](https://users.cs.utah.edu/~ladislav/alihamadi13anatomy/alihamadi13anatomy.html) 先骨层再骨—皮内部体积映射的方法。论文不会替代本模型的实际图审。

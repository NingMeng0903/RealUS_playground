# V14 一致 rest / bind / motion 实现与验收记录

当前工作基于工作区提交 `28f770c181bebad21ccacee8edae03916f35f1b4`，不改动其他项目的并行修改。本轮是执行记录；**尚未完成全身解剖验收，所有编译候选仍为 `publishable=false`。**

实际图像可在 [V14 Genesis 三维检查页](../outputs/anatomy_retarget/v14_review_20260908_001/index.html) 切换版本、体型、动作和相机。图像由仓库原有 Genesis 平台产生；没有使用 AI 出图替代几何验证。原始 [V13 检查页](../outputs/anatomy_retarget/v13_capture_genesis_review_20260908_001/index.html) 和 [Git 版本审计](anatomy_git_audit_20260908.md) 保留。

## 当前可复查的结果

最新完整六格候选是 `v14_collar_driver_axes_{213328,213712}_20260908_001`。它保留原始 142 几何和权重，使用此前已保存的保端 rest 参数，修正锁骨局部旋转、锁骨近端支点及 rest 改变后的旋转坐标转换。此次比较**没有重新拟合 rest，也没有叠加旧运动规则下拟合的姿态校正**。

| 体型 | T-pose 左臂最大皮外 | 自身动作 | 互换动作 | 原始模板骨端形状 gate |
|---|---:|---:|---:|---|
| 213328 | 0 mm | 11.625 mm | 31.629 mm | 通过 0.1 mm |
| 213712 | 0 mm | 28.899 mm | 10.648 mm | 通过 0.1 mm |

六个骨端相对原始模板的最大刚性拟合残差为 **0.006758 mm**，但两体型 T-pose 均仍有桡骨—尺骨 **83 对三角面交叉**，采样穿入下界约 **2.20 mm**。这是静态原始布局问题，不能因原模型也有就计为通过。自身/互换动作还出现肱骨与前臂的交叉。详细数据和实际打开的图像见 [当前独立审查](anatomy_v14_driver_axes_independent_review_20260908.md)。

这次运动修正对 213328 自身动作的左臂最大皮外误差为 **44.826 → 11.625 mm**；旧坐姿回归为 **27.080 → 9.600 mm**，旧踢腿回归为 **30.698 → 4.802 mm**。这些是相同 rest 几何、相同输入下的改善，仍未达到 1 mm 目标，也不是对原始 142 或 V7 的全面胜出结论。

## 已实现的运行接口

代码：[consistent_runtime_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/consistent_runtime_v14.py)。

```python
from projects.genesis_ue_sync.anatomy_retarget.consistent_runtime_v14 import (
    CompileConfigV14, compile_subject, load_compiled_subject, pose,
)

# 编译输入可以是完整 SourceOperatorV8 或已验证的 SubjectRuntimePackV8。
# source_operator 是已认证的完整 SourceOperatorV8。它同时提供原始几何，
# materialize 得到的 subject 只提供此 betas 对应的运动参考。
compiled = compile_subject(betas, source_operator,
    CompileConfigV14(shape_reference_operator=source_operator))
compiled.save(output_directory)

# 运行时仅加载保存的包与 SMPL-X 参数。
compiled = load_compiled_subject(output_directory)
vertices = pose(compiled, pose55, transl)
```

上例保留原始几何，还没有执行解剖拟合。默认空配置仍提供相对 materialized 来源的恒等映射，供旧实验回放；新的骨形适配使用显式 `shape_reference_operator`，避免继承上游 beta 顶点变形。实际左臂候选由下面的离线拟合 CLI 生成。

保存包嵌入完整原始 subject pack、235 个控制器、原拓扑、原 14 槽逐顶点权重和 CSR 权重、原 runtime coefficients、betas、目标 rest/bind、平移与旋转响应映射、可选固定姿态校正及哈希。加载校验体型、几何、权重、绑定和 provenance；拒绝裸 asset、旧 preview bind 和软组织单独跟随旁路。运行数组与 source/effective asset 使用独立的只读快照，避免调用方改写原 pack 后悄悄改变已编译运动。

运动核心使用以下统一组成：

```text
G0 = 保存的 source response(pose)，可含离线校准数值
D  = inverse(L0_bind) × global_to_local(G0)
mapped(D).R = transpose(Q) × D.R × Q
G1 = full_FK(L1_bind × mapped(D) × baked_local_correction(pose))
V1 = 原逐顶点权重的 LBS(rest1, G1 × inverse(bind1))
```

所有 235 控制器按父先于子的顺序执行，手腕、踝和其子树没有独立世界空间覆盖。`source_bone_posed_global` 的运动参考实际是 materialized asset 的 `target_bind_global`，不能误用其 authored `source_bind_global`。改变 rest 的同一场用于生成 bind 原点、局部刚体帧和辅助平移响应。

`driver_rotation_maps_v14` 在编译期投影到 proper SO(3)，计算 `Q_i = R_reference_i.T × R_root_map.T × R_target_i`，其中 `R_root_map = R_target_root × R_reference_root.T`。这使已改变 rest 轴向的控制器继续消费相同 SMPL-X 运动轴；整个模型统一刚体变换时 Q 为恒等，保留等变性。只共轭旋转，不能用完整 SE(3) 共轭把新支点再消去。新编译默认 `driver_axes`，旧包缺少 rotation_maps 时按原恒等值加载，保存结果不被重新解释。[独立代数消融](anatomy_v14_rotation_transport_probe_20260908.md) 使用 proper rotation 的相对旋转量测，非中性全局旋转等式误差低于 3.1e-9°。

平移基也已独立复核。正确式为 `T_i = R_target_i.T × J_field_i × R_motion_reference_i`，不能在右侧误用原始 shape bind 的轴。新编译已修正并记录 `translation_transport=motion_reference_axes`；旧包仍使用其保存的矩阵，保持历史精确回放。两个参考轴不同时的独立测试已通过。[实际代数审计](../outputs/anatomy_retarget/v14_translation_maps_coordinate_audit_20260908_001/report.json) 表明，此项在已查左臂动作中的影响低于 1.25e-8 mm，全身最多约 0.202 mm；它是应修正的一致性错误，但不是数十毫米皮外误差的主因。

完整 source pack 的实际保存/加载合同见 [runtime contract](../outputs/anatomy_retarget/v14_runtime_contract_20260908_001/report.json)：213328 的 T-pose 和自身动作 vertices / globals 保存前后均逐元素相等。V14 核心测试覆盖全局刚体等变性、手指子树跟随、混合权重、轴向变形和三角交叉漏检案例。这些只验证实现合同，不是解剖通过。

[motion_response_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/motion_response_v14.py) 将离线校准的 driver 声明、全局/局部/逆 bind 和 coupling 单独保存，禁止附带改变原权重、拓扑、层级或启用 preview 分支。[最新响应保存合同](../outputs/anatomy_retarget/v14_motion_response_contract_20260908_002/report.json) 在禁止运行时调用 coupling builder 的条件下，完成 T、两采集动作和旧坐姿的逐元素回放一致性检查。该合同使用的“只改 collar 响应”几何仍不合格，合同成功不等于候选被采用。

[后续独立响应审计](../outputs/anatomy_retarget/v14_response_review_20260908_001/review.json) 对最新两体型各 9 个已存姿态逐一验证：candidate vertices、controller globals、source globals 和 faces 全部 bitexact；禁止 compile/materialize/driver-map/coupling-builder/from_assets 的哨兵也全部通过。新响应保存内容签名包含几何、面、driver indices 和 weights，拒绝将响应应用于不同原始内容；旧响应只能通过包含完整已校验 source pack 的路径使用。加载时还检查非有限值、响应幅度、driver/frame 声明以及三绑定一致性。

最小直接驱动示例见 [pose_compiled_subject_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/cli/pose_compiled_subject_v14.py)。它只需要保存包与明确的 SMPL-X `pose55`，不加载 Blender 或外部 SMPL-X 模型；`--translation-key` 必须显式指定，避免误读采集系统的其他平移约定。

最新实际运行示例与输出见 [direct-drive report](../outputs/anatomy_retarget/v14_direct_drive_example_20260908_002/report.json)，其 394,770 个顶点与保存的采集动作候选逐元素一致。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/media/camp/EXT_DRIVE/envs/genesis/bin/python -m \
projects.genesis_ue_sync.anatomy_retarget.cli.pose_compiled_subject_v14 \
  --compiled outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001/compiled \
  --pose-file outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001/subject_213328_pose_213328.npz \
  --pose-key pose --output outputs/anatomy_retarget/my_direct_pose.npz
```

当前冻结膝响应的 SMPL-X 局部 rotvec 范数支持半径为 **2.268928 rad（130°）**，踝为 **1.308997 rad（75°）**；辅助平移幅度上限为 **16.2 mm**。这是现有响应的计算域，不是已经通过验证的解剖活动范围，也不能直接当作几何测得的膝屈曲角。超域明确报错，不裁角；当前 `anatomically_validated=false`、`universal_pose_claim=false`。

[SparseLBSV14](../src/projects/genesis_ue_sync/anatomy_retarget/sparse_lbs_v14.py) 在加载时预存非零槽索引，避免每帧为 5,526,780 个槽展开约 700 MB 的变换矩阵。实际正权重只有 578,790 项；求值后恢复原 14 槽位置并按原顺序求和，不修改原权重。两体型各 9 个存档姿态与原 LBS 的 **float64 输出全部逐元素一致**，保存的 float32 顶点也全部精确一致。单次 LBS 中位耗时在此机、单线程设置下由 1.362 / 1.206 s 降至 0.290 / 0.129 s；source motion 中位约 0.0083 s。此测量不是实时帧率保证，也不含渲染，见 [测速与精确回放合同](../outputs/anatomy_retarget/v14_sparse_lbs_contract_20260908_001/report.json)。

## 保骨端的轴向场

[axial_caps_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/axial_caps_v14.py) 用关节端平台区和中间平滑过渡调整骨干长度，限制总长度比例 `[0.9, 1.1]`，径向坐标不变。检查全程正 Jacobian，局部运动帧只接受 proper rigid rotation；不把 `s × R` 放入 bind。

左臂离线实现 [fit_consistent_arm_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/cli/fit_consistent_arm_v14.py) 同时求肘、腕站点，肩站点固定。桡骨与尺骨共享前臂场，腕与整个手子树共同更新；还允许前臂整体接合位置的受限调整。这保证增量轴向映射保端部形状，但**不能消除原来桡骨与尺骨彼此已经存在的交叉**。

新增可选 `--rest-roll`，以度为单位增加肱骨/前臂两项 rest 轴转动，范围各为 ±60°；轴向保端场之后绕目标段轴作刚体变换，整只手随前臂远端映射。旧 6/9 参数报告可以按零 roll 迁移，原有无 roll 调用仍保持其语义。真实 142 输入的 targeted tests 验证了几何/bind 同场、全手 FK、骨端形状、原权重和 compiler/scorer 一致性。增加自由度是修正模型表达能力，不能据此宣称已找到合格布局。

初始 fit 使用分层手骨采样；全量复测暴露漏掉的最坏手指点后，增加 `--full-hand-fit`，中性阶段使用全部 11,415 个左臂骨顶点。

## 已执行的候选与失败原因

下表最大穿出均指完整左臂骨集合，不是全身，也不是血管验收。

| 候选 | 输入与方法 | 已观察结果 | 判定 |
|---|---|---|---|
| 新冻结 V7 对照 | 两采集体型 × T / 自身 / 互换动作，原 V7 right-multiply 实现 | 左肘自身动作骨面间隙仍明显大于 raw142 | 仅保留对照 |
| `v14_arm_fit_20260908_001` | 213328，六站点参数、多姿态 fit | T 最大穿出 18.233 → 10.461 mm；坐姿 30.569 → 36.088 mm | 失败，关节重叠仍在 |
| `v14_arm_fit_20260908_002` | 加入前臂整体接合位置 | T 肱骨→尺骨 fit 采样穿入消除，最大皮外仍 9.549 mm | 失败，多姿态目标在静态与动态之间折中 |
| `v14_arm_restfit_20260908_004` | 先拟合中性骨层、目标皮内 1 mm | 完整左臂仍有 1 个手骨顶点穿出 1.137 mm | 失败，手部分层采样漏最坏点 |
| `v14_arm_restfit_20260908_005` | 213328，中性骨层全手骨 fit | T 全部左臂骨顶点无正皮外距离；肱骨→桡/尺 fit 深度在 0.5 mm 内 | 只改善中性阶段；未通过全部骨对和动作 |
| `v14_arm_restfit_213712_20260908_005` | 213712 独立编译及中性 fit | T 全部左臂骨顶点无正皮外距离；自身和互换动作均已导出 | 同样不能当全身通过 |
| `v14_arm_pose_fit_20260908_002` | rest4 + 7 个固定 fit 姿态，5° / 5 mm 初始响应界限 | 捕获动作和深弯肘有实际改善，仍存在骨/手部超差 | 失败候选，保留几何、固定校正和图审 |
| `v14_full_arm_graph_roll_fit_213328_20260908_001` | 新运动基、原始几何、两项 rest roll、12 条肩肘腕 FIT 接触边，单次有界 Powell | T 9.011 mm；自身 12.009 mm；互换 11.763 mm；120° 肘 7.393 mm | 失败；较前一个候选改善互换动作，但牺牲 T-pose，骨面交叉加重 |

各候选目录的 `report.json`、`compiled/` 和 `subject_*.npz` 可复算。优化器的 `success`、评分下降、导出成功和 `anatomical_passed` 分开记录。

最新 full-arm graph 通过显式 `--contact-graph full_arm` 启用，默认旧 `legacy_elbow` 保持兼容。它使用已冻结的 12 条肩/肘/腕 FIT 查询，纳入完整肩胛骨和舟骨目标，缺少任何 query/target lookup 时立即失败；两个尺骨↔舟骨方向只有排斥。代码同时拒绝 `.validation` 域进入这个评分函数。[独立代码审查](../outputs/anatomy_retarget/v14_full_arm_contact_graph_review_20260908_001/report.json) 提出的域后缀保护已补上并测试。

这次试验仅按预定预算运行一次（120 次 optimizer 调用，另有一次零参数基线），达到函数求值上限，不能据此断言约束不可行。scorer 与运行顶点的最大差异为 **6.01e-8 m**，实际 Genesis 已输出并由主 agent 打开 T、两采集动作和 120° 弯肘的肩/肘/腕图。[四格独立骨面审计](../outputs/anatomy_retarget/v14_full_arm_graph_roll_audit_20260908_001/report.json) 仍失败：桡尺骨各格 83 对交叉，T 的肱骨—桡骨/尺骨交叉分别为 51 / 154 对。候选保留供诊断，未替换检查页默认的两体型运动修正对照，也未使用新 AMASS 验证帧调参。

共同、可逆的空间变形会共同运输原来相交的体积，不能消除两骨固有的相交关系。这解释了当前“桡尺骨共用前臂场”的表达限制；下一步需要先对两骨分别求解满足近端与远端接合的相对 rest 布局，再在骨层通过后建立一致的组织材料场。不能把共同变形的保拓扑性质误作内部骨对无相交保证。

## 固定姿态校正

[pose_corrector_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/pose_corrector_v14.py) 保存紧支撑平滑 RBF；特征是局部关节旋转矩阵，避免同一旋转的轴角表示差异。中性校正强制精确为零；超出支持范围或幅度时抛出明确错误，不裁角或替换零值。

[arm_pose_fit_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/arm_pose_fit_v14.py) 与 [bake_arm_pose_corrector_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/cli/bake_arm_pose_corrector_v14.py) 是纯离线优化层。首轮用控制器 131 / 132 / 135，特征关节 `[13,16,18,20]`，中性、两采集动作和 30/60/90/120° 左肘扫描共 7 个节点。旧坐下/踢腿仅作为回归输出，不参与 fit。

腕部目标区分真实接触与排斥：桡骨→舟骨可以约束接合；尺骨→舟骨只检查穿入，不强行拉近。尺侧腕部有 TFCC 介导的关系，不能把所有腕骨当成直接相接；依据包括 [TFCC 组织学研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC1257203/)。也移除了舟骨采样到舟骨自身的无效查询。

## 独立检查及新发现

[独立左臂审查](anatomy_v14_arm_review_20260908.md) 实际打开了 T / 自身 / 坐姿的 Genesis 肘、腕图，并查询冻结 `validation` 域；fit 域与 validation 域不混用。

[新增独立 Genesis 图审](anatomy_v14_genesis_visual_review_20260908.md) 实际查看了两体型原始几何 T-pose、pose-fit 的自身/互换/120° 弯肘，以及连续坐姿第 0/18/35 帧。结论仍为肘骨端、腕部站点和管骨空间关系未通过。[原始几何总 cap 审计](../outputs/anatomy_retarget/v14_original_shape_audit_20260908_003/report.json) 已根据 `shape_reference_kind` 正式启用相对原始模板的 0.1 mm gate，最大 cap 残差 0.006635 mm；关节和动作 gate 仍独立判失败。

[surface_validation_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/surface_validation_v14.py) 增加全三角面相交、双向顶点与面心 signed distance、闭合与绕向检查。VTK 用两棵 OBB 树查询三角面接触，能发现没有内部顶点的薄面交叉；见 [VTK 原始文档](https://vtk.org/doc/nightly/html/classvtkCollisionDetectionFilter.html)。采样深度仍只是穿入深度下界，不能声称是连续空间证明。

动作中的 cap 形变已从仅诊断改为明确的 **0.1 mm 失败 gate**，大幅刚体转动/平移仍不受惩罚。新 [213328 审计](../outputs/anatomy_retarget/v14_driver_axes_arm_audit_213328_20260908_002/report.json) 与 [213712 审计](../outputs/anatomy_retarget/v14_driver_axes_arm_audit_213712_20260908_002/report.json) 使用同一几何重新测量，各新增 4 个动作 cap 失败：两动作的左右桡骨均超差，最大残差分别为 0.32772 / 0.31775 mm。这纠正了旧报告 `pose_rigidity_gate_applied=false` 的验收遗漏，不改变之前任何几何或图像。

已确认两项需要继续修正的结构问题：

1. 原始桡骨—尺骨局部已有约 2.208 mm 穿入。共同刚性搬动两骨会保留这个问题，不能靠换优化器解决。下一轮骨层模型需要允许有约束的骨间相对接合调整，同时维持端部形状、原权重与关联组织的材料映射。
2. V8 `materialize_subject` 的 `beta_vertex_basis` 在进入 V14 之前就改变了骨形。相对 operator.template_asset，213328 的左肘肱/桡/尺 validation cap 最大成对距离变化分别为 0.418408 / 0.443122 / 0.367493 mm；213712 为 0；合成 PC1 +1.5 为 2.375436 / 2.384439 / 2.045453 mm。已加入独立原始几何参考，原始全部组织与 bind 先做同一根刚体配准，再执行受限 rest 场；beta 对应的 materialized bind 仅作为运动参考。这样避免只恢复骨头而把血管留在旧位置。

新合同见 [original shape contract](../outputs/anatomy_retarget/v14_original_shape_contract_20260908_001/report.json)：两个采集体型和合成 PC1 +1.5 的原始骨端成对距离误差约 2e-9 m 以内，根刚体动作误差约 1.4e-7 m 以内，保存加载回放逐元素一致。该合同尚未拟合骨布局。随后 `v14_original_shape_restfit_213328_20260908_002` 完成原始几何下的 T-pose 左臂拟合，完整左臂没有正皮外距离；其肱骨 validation cap 相对原始几何的最大成对距离误差为 0.0068 mm。采集动作仍有最高 44.83 mm 左臂穿出，故候选仍失败。

3. [独立手部运动审查](anatomy_v14_hand_driver_audit_20260908.md) 将旧坐姿、踢腿中约 47–49 mm 的腕部漂移追溯到了肩站点。129 号 `Clavicle_Rot_L` 原映射没有正确消费局部 collar 旋转；其控制器原点又接近 SMPL-X J9，而真实锁骨近端靠近 J13。只补旋转但仍绕旧原点转，肩误差仍约 27–30 mm，不能作为完成修复。

已通过 `make_collar_pivot_response_asset_v14` 将 129 改为局部 J13 响应、在测得的 J13 附近建立转轴，并同时重建 global/local/inverse bind，只离线重算该控制器 coupling。旧坐姿、踢腿中 source shoulder 到 SMPL-X J16 的误差分别降至约 **0.182 / 0.139 mm**。原几何、原 source bind 与权重不改；移动的是虚拟控制器原点，不能误称锁骨网格被整体搬了 86 mm。锁骨—胸骨真实接合的 Genesis 图审未见明显新增断开，但完整肩部骨面对仍有交叉，局部管骨穿入也未全部改善，见 [锁骨独立审查](anatomy_v14_collar_pivot_independent_review_20260908.md) 和 [物理表面审计](../outputs/anatomy_retarget/v14_collar_surface_audit_20260908_001/report.json)。

4. 肩、肘、腕站点的平移并不充分表达骨的 rest 朝向。固定站点的 81 格肱骨/前臂轴向 roll 消融中，只按皮外选出的 `+60° / −15°` 将原始根配准几何的 T 最大皮外从 **15.746 降至 4.782 mm**；综合接触分数最优解却仍有 **16.290 mm** 皮外。说明 rest 旋转自由度有作用，但不能拿单一皮内指标直接选用。该消融不改参数范围、不作优化，所有候选都未通过，见 [roll 消融记录](../outputs/anatomy_retarget/v14_arm_rest_roll_probe_20260908_001/report.json)。

原始前臂两个混合蒙皮控制器在采集动作中也有显著相对 roll；因此还需要比较姿态中的骨端刚性，而不能只检查 T-pose。不得将原模型已有的变形和穿插标记成通过。

控制器 134 的有界响应扫描进一步把非刚性骨端变化定位到 wrist-driven twist：原 gain 0.78 下，两个采集动作的桡骨 cap 残差约 **0.329 / 0.159 mm**，120° 合成腕轴转动可达 **0.869 mm**。单纯肘屈伸没有同样的 cap 破坏。扫描不修改生产响应，也没有把 helper 清零当修复；完整记录见 [helper 134 诊断](../outputs/anatomy_retarget/v14_forearm_helper134_probe_20260908_007/report.json)。

下肢已补齐 [独立完整骨面审计](anatomy_v14_lower_chain_audit_20260908.md)：两体型 T/own/swap 六格，左右 `Ilium–Femur`、`Femur–Tibia`、`Tibia–Talus` 共 36 个必检骨对，另有 12 个 Sacrum–Femur 背景骨对。网格均闭合、绕向有效，36 个必检骨对中 **30 个失败**。213328 自身动作左膝采样穿入下界 **17.058 mm**；两体型 T 的左右髋约 **4.068 / 4.354 mm**，右踝约 **2.537 mm**。这些数据说明下肢也同时存在中性布局和姿态响应问题，不能将根帧一致、骨链连续或表面闭合当成接合正确。

## 连续动作与复现

[validation_motion_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/validation_motion_v14.py) 预先冻结另三段录制：`sitting2`、`kicking2`、`normal_walk2`，每段 36 帧，输出采样 12 Hz，保留原 SMPL-H 身体和手旋转、插入三个零面部关节，不把 AMASS shape 当 SMPL-X shape。输入路径、帧号和 SHA-256 见 [冻结清单](../outputs/anatomy_retarget/v14_frozen_validation_20260908_001/manifest.json)。

[replay_compiled_motion_v14.py](../src/projects/genesis_ue_sync/anatomy_retarget/cli/replay_compiled_motion_v14.py) 直接加载编译包，对全部原始帧运行 `pose`，生成 Genesis 视频与每帧皮外指标。诊断距离查询只测量，不参与重绑定或修正。超域帧保留错误卡片，不能冒充有效几何。视频显示时仅为直立观察，对皮肤和内部组织等量去除根旋转与平移。

已生成 [213328 连续 sitting2 Genesis 视频](../outputs/anatomy_retarget/v14_continuous_sitting_213328_20260908_002/genesis_motion.mp4)：固定 pose-fit2 包直接回放 36 帧，没有超域替代帧。主 agent 已实际查看中间帧的全身、肘和腕图，仍可见手腕与肩部问题；它是失败候选的视频证据，不能算通过。此片段从现在起继续只作验证，不参与后续拟合。

最新 collar + driver-axes 固定包的两体型三片段复测共 **216/216 帧有效求值、0 个超域或错误替代帧**。坐姿已分别输出 [213328 视频](../outputs/anatomy_retarget/v14_driver_axes_sitting_213328_20260908_001/genesis_motion.mp4) 和 [213712 视频](../outputs/anatomy_retarget/v14_driver_axes_sitting_213712_20260908_001/genesis_motion.mp4)；主 agent 实际打开两条视频对应的 0/18/35 帧，检查全身、左肘、左腕三个 Genesis 视角。动作保持连续，但局部管骨通道拥挤，腕侧血管贴近骨端；完整深度仍以独立测量为准。

| 体型 / 冻结片段 | 全骨最大皮外 | 左臂骨最大皮外 | 血管最大皮外 |
|---|---:|---:|---:|
| 213328 sitting2 | 56.406 mm | 13.290 mm | 50.202 mm |
| 213712 sitting2 | 47.848 mm | 10.769 mm | 45.988 mm |
| 213328 walking2 | 61.076 mm | 16.517 mm | 61.690 mm |
| 213712 walking2 | 61.632 mm | 11.159 mm | 61.796 mm |
| 213328 kicking2 | 47.841 mm | 20.155 mm | 48.278 mm |
| 213712 kicking2 | 51.223 mm | 15.905 mm | 51.670 mm |

每段的逐帧报告位于 `v14_driver_axes_{sitting,walking2,kicking2}_{213328,213712}_20260908_001/report.json`。walking2/kicking2 首次运行采用 metrics-only；坐姿同时渲染 Genesis。上表是全片段最大值，不是平均值；0 运行错误不等于解剖通过。所有新片段继续冻结为验证集，不用于后续拟合或挑参数。独立连续图审见 [复核记录](anatomy_v14_continuous_driver_axes_review_20260908.md)。

机器汇总见 [216 帧冻结验证报告](../outputs/anatomy_retarget/v14_motion_validation_summary_20260908_001/report.json)，记录各分组最坏帧、输入报告和编译包哈希。另已完成 [213328 踢腿下肢 Genesis 视频](../outputs/anatomy_retarget/v14_driver_axes_kicking_lower_genesis_213328_20260908_001/genesis_motion.mp4)，逐帧显示全身、髋、膝、踝四视图；主 agent 实际查看 0/18/35 帧。两次踢腿求值属于同一冻结运动，不能重复计入 216 帧总数。

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
QD_OFFLINE_CACHE_FILE_PATH=/tmp/anatomy_v13_genesis_qd \
/media/camp/EXT_DRIVE/envs/genesis/bin/python -m \
projects.genesis_ue_sync.anatomy_retarget.cli.replay_compiled_motion_v14 \
  --compiled outputs/anatomy_retarget/v14_collar_driver_axes_213328_20260908_001/compiled \
  --motion outputs/anatomy_retarget/v14_frozen_validation_20260908_001/sitting2.npz \
  --output outputs/anatomy_retarget/my_new_motion_review \
  --views whole_ap left_elbow_lateral left_wrist_ap
```

## 代码验证与交付索引

最新检查为 **65 passed**，没有 warning；覆盖一致 FK、原始形状参考、不同 shape/motion 轴的平移运输、固定响应保存、保端轴向场、RBF 支持范围、完整三角相交、稀疏 LBS 精确性、动作 cap gate 及冻结 FIT 域保护。CUDA 资产缓存现在显式复制只读数组，避免 PyTorch 将不可写 NumPy 数据当作可共享存储。

完整来源与结果入口见 [交付索引](../outputs/anatomy_retarget/v14_delivery_20260908_001/manifest.json)，测试原始输出也一并保存。该索引明确 `complete_user_plan=false`，不把可编译、可驱动、可渲染误标为解剖通过。检查页当前包括 **232 张 Genesis 对比图、4 条视频**。原始 rig/权重、骨长约束、运行回放与性能已有代码和证据；各失败候选均保留报告及图审。

## 仍未完成的计划阶段

全身骨层尚未通过；左臂仍有原始骨对重叠、辅助 twist 引起的动作骨端形变，以及静态接合与皮内空间的冲突。锁骨驱动/支点与旋转坐标已作可回放的代数修正，不能再把全部剩余误差笼统归咎于旧肩部转轴。上游体型骨形参考已分离并通过合同测试，但整体解剖通过仍需独立图审和表面检测。髋—膝—踝当前保留原始运动与新 V7 对照，没有宣称已完成新骨层求解。

按用户已批准的阶段顺序，**骨—皮双边界软组织体积场尚未进入最终拟合或验收**。当前血管和神经确实使用原权重随同一控制器链运输，但这不是管径、分叉、骨间不穿入或体积不翻转的证明；这些指标和材料坐标包仍需骨层通过之后实现与验证。当前包仅供可复现的研究与诊断，不具有“任意动作 / 任意体型已通过”的支持范围。

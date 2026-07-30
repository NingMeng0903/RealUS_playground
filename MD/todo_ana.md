结论：你的怀疑是对的。旧版髋关节定位和验收逻辑存在根本问题，V253 的“通过”不能成立。目前已完成真实 V71 基线重建、V7 资产/CLI 骨架、髋膝局部重建、完整父子 FK、单 operator 双 beta 材质化和运行时加速；但还没有达到最终发布条件，尤其是髌骨独立 oracle、血管完整门禁、肋骨/颅骨/舌头和独立盲审仍未完成。现有 latest 没有更新，全部 V7 资产仍是 publishable=false 候选。
一、目前确认的问题
1. V253 的髋关节通过值是无效自证
旧逻辑中的 femoral_head_to_acetabulum_m=0.0 不是从最终表面重新测量，而是 refit 后直接写入报告的值。
固定 V71 材料域后重新测量得到：
真正 V71 股骨头—髋臼中心误差约 0.61 mm。
后续 refit 版本同一区域偏差约 51–69 mm。
因此旧 locator 确实把错误位置当成了髋关节中心，或者在 refit 后重新选择了“最有利”的点。
V253 的膝关节验收还把 Tibia、Fibula、Patella 合并，并使用已经被 refit 的端点作为探头，导致错误结果可以重新定义自己的验收点。
2. 股骨过长、穿髌骨的主要原因已定位
196 以后多个相邻骨骼被独立锚定到 SMPL-X 全局 joint。以腿部为例：
Femur_Rot
  └─ Knee_Rotate
       └─ Tibia_Bone
            └─ Patella_Rotate
旧实现同时把 Knee、Tibia、Ankle 等节点独立拉向 SMPL-X joint，导致：
Femur_Rot → Knee_Rotate 的 Blender 局部位移不再守恒。
Knee 和 Tibia 可能被同一个 SMPL-X knee 重复驱动。
强屈膝时股骨远端看起来“变长”，刺入髌骨。
相邻骨骼用于血管蒙皮的 frame 在同一关节处可能相差几十厘米。
旧资产中部分 target_bone_head 和真实 bind origin 相差 25–34 mm。
这不是单纯的股骨 mesh 长度问题，而是骨骼父子 frame、端点和独立锚定同时不一致。
3. 原始 Blender/V71 也不能直接作为全部几何真值
V71 的层级、枢轴和局部 FK 是目前最可信的联动先验，但源 Blender 本身也存在几何问题：
静止姿态已有部分股骨—髂骨、股骨—胫骨面交叉。
深屈膝时有股骨—髌骨干涉。
左右髌骨响应不同。
原始 evaluated mesh Action 在深屈膝时会把髌骨明显拉离滑车沟。
直接测原始 Action：
左髌骨—滑车最近距离在约 90° 时扩大到约 97 mm，最大约 104 mm。
右侧扩大到约 44–49 mm。
因此不能把 Blender Action 的最终髌骨表面轨迹直接当验收 oracle。正确做法是：
骨矩阵、层级、枢轴、左右响应斜率继承 V71。
接触位置由固定材料域和几何 corridor 修正。
生成一次冻结的“V71 联动先验 + 最小接触修正”规范 oracle。
后续候选只能和该 oracle 比较，不能和自己生成的 spline 比较。
4. 213712 旧资产的髋臼已经严重变形
用 V71 固定髋臼域恢复时：
213328 局部最大修正约 5.3 mm。
213712 旧资产局部最大修正约 43.5 mm。
213712 髋臼定半径球拟合残差从约 5.76 mm 降至 1.19 mm。
这表明 213712 的旧 refit 不能作为几何真值。43.5 mm 修正虽然只影响固定髋臼核心及四圈过渡顶点，没有整体缩放骨盆，但仍必须通过剖面图和独立审查，不能只看球心指标。
二、已经完成的工作
1. 重建了真正的 V71 基线
已经从 15b6016... 的隔离源码重建，而不是使用后续伪装成 V71 的 v240_v71_full_fk。
导出了：
235 根骨骼的原始 hierarchy。
local/global bind。
271 帧 Blender Action 骨矩阵。
股骨、髂骨、胫骨、髌骨、肱骨、尺骨、桡骨的 evaluated mesh sweep。
三角化后的真实表面拓扑。
Blender 版本、源 commit、blend 哈希和拓扑哈希。
产物：
[V71 Source Asset](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_source_bake_001/v71_operator_source_v6.npz)
[Blender Action Oracle](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_source_bake_001/blender_action_oracle_v7.npz)
原导出器只保留原生三角面，导致大量 quad mesh 的 oracle faces 为空；现在已经改为 Blender loop_triangles，实际表面已完整导出。
2. 冻结了不可变关节材料域
已经固定：
左右股骨头。
左右髋臼。
内外侧股骨髁。
内外侧胫骨平台。
髌骨及髌骨后关节面。
股骨滑车沟。
Femur、Tibia、Patella 独立刚性域。
验收只能使用这些固定 vertex IDs，不允许在 refit 后重新找最近点。
产物：
[fixed_joint_domains_v7.json](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_candidates/joint_rebuild_001/fixed_joint_domains_v7.json)
3. 重建了完整 Blender 父子 FK
目前候选采用完整 fitted local FK：
父子局部 bind 平移来自当前 beta 修正后的 skeleton。
不再把相邻子骨骼独立锚定到 SMPL-X global joint。
修复了 stale target_bone_head/tail，端点重新与 bind origin 对齐。
SMPL-X 只提供动作驱动，不再覆盖真实骨骼连接位置。
股骨以髋臼球心为旋转枢轴。
Knee_Rotate 保持 rotation-only。
Tibia 使用最大 1 mm 的预烘焙 parent-local glide。
Patella 左右独立 spline，不与 Tibia 合并。
相关实现：
[joint_reconstruction_v7.py](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/joint_reconstruction_v7.py)
[anatomy_lbs.py](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/anatomy_lbs.py)
4. 髋关节不再整体缩放骨盆
当前实现：
用固定股骨头域拟合球面。
用固定髋臼域做定半径 socket fit。
将 V71 髋臼形状映射到当前 beta 的球头半径和骨盆方向。
只修改髋臼核心和四圈拓扑过渡。
股骨头、远端髁保持刚性。
报告明确记录 whole_pelvis_scaled=false。
5. 膝关节当前最终表面几何已明显改善
在同一 operator 的 2 beta × 3 pose 上，使用固定材料域从最终顶点重算，目前髋、胫股 gap、长骨刚性门可以通过。
当前四个内外侧胫股最小 gap 均处于 0–3 mm：
beta 213328：T-pose：2.52 / 1.48 / 0.99 / 1.64 mm
pose 213328：2.65 / 2.02 / 1.40 / 0.94 mm
pose 213712：2.24 / 2.30 / 2.28 / 2.09 mm

beta 213712：T-pose：1.11 / 1.15 / 0.90 / 1.46 mm
pose 213328：1.43 / 1.84 / 1.82 / 1.13 mm
pose 213712：1.32 / 2.97 / 1.86 / 1.70 mm

这里的股骨头中心误差是从最终股骨头/髋臼顶点重新拟合得到，不是 material-fit 自报值。
但当前矩阵中的髌骨轨迹比较仍临时使用候选自身轨迹，因此不能作为最终髌股验收结果；这部分尚未放行。
6. 建立了 V7 operator/subject/CLI
已经实现：
SourceOperatorV7
SubjectAssetV7
bake-template
materialize-beta
apply-pose
operator 中已有：
10 维 beta 顶点响应基。
SMPL-X joint beta basis。
235 根骨骼 bind twist basis。
头/中/尾内部 handle basis。
固定材料域。
关节 spline。
V71 socket 模板。
血管/神经运行时材料 frame。
完整来源和哈希。
相关代码：
[operator_bake_v7.py](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/operator_bake_v7.py)
[v7_artifacts.py](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/v7_artifacts.py)
[run_anatomy_v7.py](/media/camp/EXT_DRIVE/RealUS_playground/src/projects/genesis_ue_sync/anatomy_retarget/cli/run_anatomy_v7.py)
候选 operator：
[source_operator_v7.npz](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_candidates/rebuild_002/source_operator_v7.npz)
两个 beta 的 SubjectAsset 都来自这一个 operator，不依赖 Blender 或 pose cache：
[subject_operator_213328.npz](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_candidates/rebuild_002/subject_operator_213328.npz)
[subject_operator_213712.npz](/media/camp/EXT_DRIVE/RealUS_playground/outputs/anatomy_retarget/v7_candidates/rebuild_002/subject_operator_213712.npz)
7. 修复了重复关节修正
第一版 operator 使用“已经修正过关节的 beta 模板”作为中性几何，materialize-beta 又执行一次关节修正，导致同一 213328 beta 的骨骼最大偏差约 7.35 mm。
现在改为：
中性顶点来自未修正 beta 资产。
层级、bind、刚性权重、关节响应来自修正后资产。
每个 beta 只执行一次离线关节修正。
修复后，213328 从 operator 重新生成的结果与直接修正资产相比：
全局 RMS：约 0.0000586 mm
最大顶点差：约 0.00204 mm
重复修正已基本消除。
8. 血管运行时已改成离线材料 frame
当前做法：
离线将每条血管/神经分成固定材料 patch。
保存每个 patch 的局部坐标、驱动骨骼和 DQ 权重。
运行时只执行 DQ frame 求值和 gather。
不运行 SDF、ARAP、碰撞、重新投影或 Blender。
抽查中血管横截面固定边变化约 0.35–1.3%，低于 5% 门限。
但尚未完成：
中心线锯齿峰门。
全身 SMPL-X 内部比例。
最大越界。
骨穿透相对规范模板的新增深度。
拓扑哈希正式报告。
9. 完成了运行时完整性与速度优化
修复了真实大资产保存/加载时，由 schema-v6 local/world endpoint 浮点 round-trip 引起的 digest 不稳定。
现在：
operator/subject 嵌入 payload 有严格 blob SHA-256。
加载时先验证精确字节哈希。
不再在每次 pose 中重复遍历几十万数组计算语义 digest。
已验证真实 39.5 万顶点资产连续 round-trip 哈希稳定。
热路径可以跳过刚刚完成过的重复结构校验，但默认 API 仍保持验证开启。
已测到的隔离强屈膝冷路径：
SubjectAsset 加载：0.405 s
39.5 万顶点生成：0.420 s
合计：0.825 s
满足 ≤1 s。并行双进程争用时约 1.03–1.06 s，最终性能报告会明确使用隔离冷启动，不用并行值或缓存命中。
最新单修正 operator 的 materialize-beta 在并行双进程时为 11.1–11.7 s，超过门限；此前隔离测量为 7.6–8.8 s。最新版本仍需要重新做一次隔离冷启动测量，不能沿用旧数字。
10. Fail-closed 和发布隔离已经生效
候选只写 v7_candidates/rebuild_002。
所有 SubjectAsset 都是 publishable=false。
当前 latest 没有更新。
旧 V6 不能自动升级成 V7。
未通过报告不能进入可信发布。
目标测试目前已通过：
V7 artifact、固定材料域、驱动耦合、soft-follow 等定向测试：50 passed, 1 skipped。
单修正 operator 改造后 V7 artifact/CLI 测试再次通过：8 passed。
三、尚未解决或尚未形成有效证据的部分
1. 髌骨最终 oracle 尚未冻结
这是目前最重要的关节未完成项。
不能使用：
原始 Blender evaluated mesh 深屈膝轨迹，因为它本身会把髌骨拉开几十毫米。
当前候选自己生成的 spline 作为 oracle，因为那仍属于自证。
下一步要生成独立规范 oracle：
从 Action 骨矩阵提取左右独立的 Patella_Rotate 响应斜率。
保留 V71 knee/patella 枢轴和父子局部响应。
在固定滑车沟/髌骨关节面上执行一次离线接触优化。
同时满足 0–4 mm corridor 和 source penetration envelope。
冻结 spline、固定 IDs、输入 Action 哈希和输出哈希。
候选只能和冻结后的 oracle 比较。
2. controller/local-FK 门还没有完整接入最终矩阵
已有底层 fail-closed 诊断，但还需要从最终资产和真实 Action 重算：
Femur_Rot → Knee_Rotate
Knee_Rotate → Tibia_Bone
左右独立。
Patella 响应斜率。
髋、膝 controller pivot/axis。
肘关节局部链。
不能从 metadata 读取“pass”，必须从 posed global/local matrices 重算 SE(3) 误差。
3. 213712 新 operator 结果不能用旧 213712 refit 直接判优劣
新 operator 的 beta213712 和旧 V240/后续 refit 相比：
全局 RMS 约 5.28 mm
最大差约 86.1 mm
q95 约 12.35 mm
这不自动表示新 operator 错误，因为旧 213712 髋臼已经被证明严重变形。下一步必须使用：
213712 对应的 SMPL-X 表面。
固定材料域。
内部比例/越界。
骨骼刚性。
V71 局部联动。
剖面图。
来判断新结果是否真正 fit 到该 beta，而不是拿旧错误 refit 当真值。
4. 肋骨、颅骨、舌头还没有完成正式硬门
尤其是舌头：当前指定 source blend 的对象列表中没有找到明确的 Tongue mesh。配置中已经取消主动排除 Tongue，但还需要：
搜索其他源 blend/旧资产是否存在同拓扑舌头。
若有，恢复并建立上下颌/牙齿/口腔固定材料域。
若整个合法源资产中确实没有舌头，必须将此作为发布阻断项，不能伪造“舌头已保留”。
5. 视觉证据尚未生成
目前只有数值和候选资产，还需要输出：
真实表面渲染。
髋、膝、肘剖面。
股骨头—髋臼 contact/penetration heatmap。
内外侧股骨髁—平台热图。
髌骨滑车轨迹和轴线。
血管中心线曲率/锯齿图。
肋骨两端连接图。
每张图的 operator、beta、pose、asset hash。
四、下一步执行顺序
冻结规范髌骨 oracle
从 V71 Action 骨矩阵提取左右响应，结合固定滑车沟做一次独立接触优化，替换当前无效的 self-oracle。

完成真正的 controller/local-FK 重算
对每个姿态直接从最终 posed bone matrices 计算 pivot、axis 和父子 local SE(3) 误差。

重跑严格 2×3 矩阵
包括强屈膝 213328 pose 跨两个 beta，并加入 0–120° synthetic knee sweep。

扩展到肘、肩、腕、踝
优先处理肘部咬合；其余只有固定材料域诊断失败时才修改。

完成血管/神经正式门
拓扑哈希、半径、中心线锯齿、SMPL-X 内部比例、最大越界和新增骨穿透。

完成肋骨、颅骨、脑、口腔/舌头
对肋骨逐根检查两端连接；对颅骨/脑/舌头使用独立 compound 门。

最新代码重新测性能和隔离运行
单进程冷启动测 materialize-beta、apply-pose；隐藏 Blender、.blend 和 pose cache 后重复生成并检查 ≤1e-6 m 确定性。

生成全部渲染、剖面和热图证据。

启动一个全新的独立盲审 agent
只提供验收规范、源文件、两个采集路径和候选目录，不提供“已经修了什么”。审查必须自行重算数组和图片。

只有独立 ACCEPT 后才发布
任一门失败继续迭代；当前 latest 保持不动。
---

# 2026-07-28 rebuild_003 实测状态（全部数字自行重算，未读 metadata pass）

候选：`outputs/anatomy_retarget/v7_candidates/rebuild_003/`，`publishable=false`，`latest_asset` 未动。

## 已通过（2×3 矩阵，六个 cell 全部）
- controller / local-FK 三门：从最终 posed bone matrices 重算，全部 pass。
- 髋：`center_error≈2.3e-8 m`、`radius_change≈5.7e-10 m`，同心且半径守恒。
- 膝：四个胫股间隙在 0–3 mm 走廊内（剖面图 0.93–2.14 mm）。
- 刚性：股骨/胫骨/髌骨边长比 0.99989–1.00011。
- 肘、逐根肋骨两端连接：pass（`sternal_target_source=costal_cartilage+sternum`）。
- 血管 topology 哈希与固定截面：`radius_edge_ratio_max_abs_change=0.31%`（限 5%）。
- 确定性：重复 apply-pose 顶点最大差 `0.0 m`。

## 隔离性能（屏蔽 bpy/bmesh/mathutils 导入，单进程冷启动）
- `materialize-beta` 7.20 s/beta（一次性）。
- `apply-pose` 冷启动 0.524 s，热 0.505 s，394,770 顶点，subject 载入 0.400 s。达到 ≤1 s 目标。

## 仍然失败（发布阻断，未做任何门槛放宽）
1. `vessel.containment`（六个 cell）：骨架本身在 T-pose 就超出 SMPL-X 体表 69.6 mm，参考体表无效 → 属于源资产与 SMPL-X 体型的整体拟合缺陷，不是血管层问题。
2. `vessel.bone_penetration`（四个 posed cell）：rest 状态已有 90.04 mm 穿透，posing 再增加 16.9 mm，1328 个顶点增量 >0.5 mm。rest 基线属源缺陷，增量属 retarget 缺陷。
3. `vessel.centerline`（四个 posed cell）：最差 Spinal_Cord 67.1°（限 5°）。改用配对逐样本比较后灵敏度提高，真实折角暴露。
4. `compound.skull_brain`：`inside_ratio=0.944`、`max_outside=14.3 mm`，但 `added_outside_m=0.0` → 纯 rest 源缺陷（`blocker_kind=source_authoring`）。
5. `compound.oral_cavity`：源资产无舌头网格，按规范记为 publish blocker，不伪造保留。
6. synthetic knee sweep：`trajectory/left`、`trajectory/right`（髌骨 vs 冻结 oracle，rms 4.7–6.6 mm，方向误差 8.0–11.5°），以及每个 beta 一个间隙分室边际超限（3.07/3.10 mm vs 3.0 mm）。

## 本轮移除的自证路径
- 删除 `_apply_femoral_head_driver_attitude_v1`：它只把冻结的股骨头顶点绕头心旋回 SMPL-X driver 姿态，头/颈边界因此被拉伸，sweep 中股骨边长比达到 0.207–2.886（局部骨头变细/变粗的直接来源），却让髋 clearance 分布看起来没变。
- 髋门不再以 clearance 中位数/q95 变化判定：非球头在非球髋臼内刚性转动必然重分布 clearance，该判据只能靠形变骨头满足。改为判定同心度、半径守恒、抬离（max separation）和 clearance 崩塌（新增 `clearance_min_drop_m`，限 1 mm），分布量仅记录。
- 血管中心线改为按测地 bin 的连通分股取心线，并对 rest/posed 同一样本做配对差分，避免大 rest 折角掩盖真实新增折角。

---

# 独立盲审 REJECT 后的修正（本轮）

盲审独立复算矩阵，结果与提交的 `acceptance_matrix_v7.json` 逐字段一致（同样的
failures / thresholds / subjects / reason），`publishable=false` 成立。真正的问题
不在已失败的门，而在**通过的门里有五个按构造不可能失败**。以下是已修正的与已确认
但未修正的。

## 已修正：构造性自证门

1. **血管 topology 门自己比自己**。`vessel_gates_v7.py` 曾写
   `reference_digest = faces_digest` 再判 `faces_digest == reference_digest`，
   源模板从未被载入，报告却把两个相同摘要并排打印成"已核验一致"。现改为由调用方从
   operator 的 pre-beta `template_asset` 算出 `reference_faces_digest` 传入
   （新增 `vessel_topology_digest_v7`），未传入即 `available=false` 判失败。
   实测 template 与两个 subject 摘要均为
   `1f279c1cedbec0a02555fa3844ca8885e549463eb1dcdde3f46e8dae2df14975`，
   拓扑确实保持——但这次是真比出来的。

2. **膝 local-FK 拿候选自己当参考**。`_observe_local_link` 里
   `auth_angle = on_axis`，授权角就是候选自己测出的 on-axis 角，参考旋转因此是被测
   量的函数；`"response_error_deg": 0.0` 是硬编码字面量。剩下真正被判的只有 off-axis
   残差，而运行时构造上就把它压到 1e-7~2.6e-6°。按 §2.2.3 现记
   `available=false`，`flexion_deg` / `off_axis_residual_deg` 仍照实记录。

3. **肘 local-FK 测了又丢弃（潜在假 ACCEPT，本轮最危险的一条）**。`_ARM_LINK_SPECS`
   找的是 `Humerus_Rot` / `Ulna_Bone` / `Radius_Bone`，源 rig 里都不存在，六条链全报
   "bone is absent"；而 `local_fk_arms` 既不进 cell 的 `failures` 也不进 `passed`
   合取。也就是说血管与复合门一旦修好，矩阵会在六条必需链全部不可用的情况下报
   `passed=true`。现按源 rig 实名接上
   `Shoulder_Rotate > Elbow_Rot > Forearm_Bone > Forearm_Twist`，并并入判定。
   接上后立刻暴露真实缺陷：`Forearm_Bone>Forearm_Twist`（尺骨→桡骨，规范授权角为 0
   刚性）在 posed cell 实测 **30.75° 左 / 21.70° 右** 旋转误差。
   另外 `evaluate_local_fk_gate_v7` 此前只按 rotation 是否为 None 间接失败，现显式
   以 `available` 判定，并把 §4.2 要求的四项补充记录写进 `items`。

## 已修正：§6 性能报告形式不合规

旧 `runtime_perf_v7.json` 的 `determinism_max_vertex_delta_m: 0.0` 是同一进程内重复
调用得出的（§6 明文禁止），没有 `vertex_checksum`、没有 isolated cold start 标注、
没有进程隔离说明，`blender_blocked: true` 是硬编码字面量，`apply_pose.cold_seconds
= 0.524 s` 还把 §6 计入限值的 0.400 s 资产载入排除在外，`materialize_beta 7.198 s`
也复现不出来。`scripts/measure_v7_runtime_v7.py` 已重写为只驱动规范自己的入口
`cli.run_v7_isolated_perf`、每次测量一个全新进程，并新增按字节的 `vertex_digest`
（原 `vertex_checksum` 只是浮点求和，抵消性误差不会暴露）。

实测（三个独立冷启动进程）：
- `apply-pose` 冷启动 **0.751 / 0.754 / 0.754 s**（含 0.390 s 资产载入 + 0.365 s 求解，
  394,770 顶点），限 1.0 s → pass。与盲审独立测得的 0.783 s 同量级。
- 跨进程确定性：三次 `vertex_digest` 逐位相同
  （`f25f8015…`，`vertex_checksum = -326991.40036946005`，与盲审一致）→ pass。
- `materialize-beta` 隔离冷启动 **4.66 s**（盲审 4.93 s），此前报的 7.198 s 不成立。
- `blender_blocked` 改为读入口的 `bpy_importable` 探测结果。

## 已确认未修正：leg hinge 解的真实运行时缺陷（当前首要阻断）

盲审的机制判断是对的，我用 `scripts/probe_hinge_twist_conditioning_v7.py` 在真实
baked leg entry 上复现并定位到更具体的一层：

- 驱动轴与授权铰轴一致时，解是精确的：phi=0、theta 逐度跟随、ankle 误差 0。大扭转
  完全来自授权轴与 SMPL-X drive 的夹角（规范 §4.6-A 记的 53.3°/85.4°）。
- **保护性 fade 实际上是死代码。** 解里的 `flex_deg` 是股骨方向与 bone→ankle 段的
  夹角，而 bind 姿态下它本来就不是 0：左 **16.419°**、右 **10.316°**（生理股骨角）。
  fade 带是 `[5°, 15°]`，所以左腿在零驱动时 fade 就已全开，phi 从第一度起按全强度施加；
  右腿只被部分衰减。这个 guard 是按一个静息值已达 10–16° 的量校准的。
- 后果（左腿，53.3° 轴偏）：drive 0.5° → 股骨轴向扭转 2.92°；drive 5° → 28.52°；
  之后饱和在约 52.6°（≈轴偏本身）。且在 drive 0.5°–10° 区间 `theta` 一直被 clamp 到 0，
  ankle 落点差 3.07–24.17 mm——近伸直区解剖根本不跟随 drive。盲审在完整运行时上测到
  同一现象：0.5° 膝输入 → 股骨自轴扭转 −34.9°、滑车沟位移 63.6 mm，这正是髌骨轨迹
  4.7–6.6 mm RMS（限 2 mm）的来源。
- 更上游的疑点：授权铰轴本身与经股骨髁轴（transepicondylar，解剖屈膝轴）差
  **22.46–25.96°**（`fk_observation_v7.py` 已算出 `epicondylar_axis_error_deg`，但没有
  与任何限值比较，`joint_contact_v7.py` 里 `knee_axis_error_deg = 2.0` 在 `src/`、
  `tests/`、`scripts/` 中零引用）。所以"让股骨扭 52° 去迁就 drive 平面"很可能是在补偿
  一个本身就错的授权轴，而不是正确的解。修 leg IK 之前必须先定轴。

## 已确认未修正：其余盲审发现

- **烘焙期形变对门不可见**：门的基线是烘焙后的 rest，而关节重建相对纯 beta 预测最多
  移动 66.80 mm（左股骨头）；髋臼烘焙期边长比左 0.87970–0.87971、右
  0.87755–0.87756（min 等于 max 到五位小数，即 12.03%/12.24% 的纯均匀缩小），
  而 `build_report.articular_reconstruction` 仍报 `whole_pelvis_scaled: False`、
  `scaled_structures: []`，与 §4.3 冲突。所有运行时刚性数字都是相对这个已形变基线测的。
- **髋同心度是构造恒等式**：`fit_sphere_fixed_radius_v7(socket, radius=head_radius)`
  从四个不同初始球心出发都返回股骨头球心，`center_error_vs_head` 恒为 0.0000 mm。
  `hip_section` 图印 "centre offset=0.00 mm" 并把两球画成完全重合，因此它无法佐证
  §4.3。
- **刚性门是算术恒等式**：`_force_rigid_mesh_driver` 把整块 mesh 的权重压成单骨 1.0，
  单骨单位权重必然逐顶点同一刚性变换，边长只差 float32 舍入（实测最差 1.000111，
  限 ±0.01/±0.02）。这是权重矩阵的性质，不是解剖的性质。
- **复合门探针不是冻结域**：`fixed_joint_domains_v7.json` 只有 24 个腿部域，肘与肋端
  探针按候选自身 rest mesh 最近点逐 cell 重选，与 R1 冲突（虽是 rest 基准且确定性）。
- **24 根肋骨里 21 个胸骨端被自动放过**：`_rib_end_metrics` 返回
  `pass = not attached or increase <= limit`，rest 距离超阈的端点不受判定
  （已在报告里以 `ungated_sternal_ends` 披露，最大未判增量 0.385 mm，不改变结论，但
  §4.5 的逐根两端要求未被强制）。
- **证据图与门不同源**：`vessel_centerline` 图印 Δmax = 32.18°，同一 cell 同一组件门报
  67.09°，因为图自己跑了一套 centerline 采样而不是门的测地分箱实现。同一份发布里
  同名指标两个数。
- §5-F 记的中心线状态（"只剩 Spinal_Cord +39.13° 与 Sacral_Nerves_L +6.03° 失败"）与
  实际不符：17 个组件里 10 个失败，Spinal_Cord 为 +67.09°/+92.74°，
  Lumbar_Nerves_L 也在失败名单里。叙述把失败低估约 2 倍。

## 修正后的矩阵状态

`passed=false`、`publishable=false` 不变，但失败清单变得更诚实：新增
`local_fk.{left,right}/Femur_Rot>Knee_Rotate`（自证已拆除）与
`local_fk_arms` 六条链；`vessel.topology` 不再出现在失败里，因为它现在是真比出来的
通过。V7 项目测试 61 passed。

## 本轮产物更新
- `acceptance_matrix_v7.json`、`runtime_perf_v7.json` 已按修正后的门与 §6 形式重新生成
  （84 条 failures，`publishable=false`）。
- **证据包尚未重新生成**，`evidence_manifest_v7.json` 与 60 张图仍对应旧矩阵；且
  `vessel_centerline` 图本身与门不同源（32.18° vs 67.09°），需与图生成路径一并修。

---

# 2026-07-29 V8.10 腿部中心线、口腔显示与血管联动修复计划

## 一、当前判断

这轮 Genesis 人工反馈已经把优先级明确下来：

- 头骨基本可接受。下一版不再改头骨整体拟合，只处理口腔中穿过牙齿的舌状组织显示。
- 当前血管的拓扑、穿洞关系、靠近皮肤时的压缩感和骨骼跟随需要冻结。禁止重新跑会改变
  整条路径或把骨骼缩小给血管让路的全量优化。
- 骨盆的父子联动目前基本正确。最大问题是股骨干、胫骨/腓骨和踝足 compound 相对
  SMPL-X 人体中心线有横向或角度偏移。
- 默认不通过移动整个骨盆修腿。只有左右髋误差能被同一个很小的 pelvis SE(3) 解释时，
  才允许启用 pelvis correction。
- `HEAD=cb3d5de` 没有新的 anatomy 源码改动。当前实际候选仍以
  `outputs/anatomy_retarget/v8_candidates/rebuild_012` 和 Genesis 实测为准。
- `rebuild_012` 的版本标记为
  `unified-head-neck-vessel-coupled-v8.9 / coupled-contact-v8.2 /
  continuous-head-source-vessel-v8.4`。

当前实测说明腿部偏心不是单个参数没调好，而是“解剖接触点、人体中心线和真实骨长”
三套约束互不兼容：

- BA9 两个 subject 与 `rebuild_012` 都是 `394770` 顶点、`782856` 三角面，faces、
  mesh names 和 vertex ranges 完全一致。因此 BA9 可以提供同顶点的视觉中心线参考，
  但不能提供运行时 FK。
- 213712 上，BA9 左右股骨解剖轴与 SMPL-X hip-knee 方向约差
  `1.05° / 1.51°`；`rebuild_012` 约差 `8.44° / 8.83°`。足轴约从
  `3.30° / 5.32°` 回退到 `9.47° / 9.55°`。
- `rebuild_012` 股骨头/髋臼 compound 相对 SMPL-X hip 的左右误差不是共同平移：
  两个 beta 的 common norm 约 `20.3-20.6 mm`，differential norm 约
  `55.7-55.8 mm`，主要是左右相反的横向误差。移动整个骨盆不能修复。
- 当前左股骨解剖长度比 SMPL-X hip-knee station 短约 `25.8-26.5 mm`，右侧短约
  `6.4-7.2 mm`；小腿平台到 mortise 的长度也比 SMPL-X knee-ankle station 短约
  `12.6-15.3 mm`。
- 当前股骨 axial scale 约 `0.9992-1.0011`，但小腿 axial scale 已到
  `1.0278-1.0310`；L0 足部 similarity fit 仍使用 `0.95087 / 0.94028`。

所以“固定股骨头、精确落到原始 SMPL-X 膝/踝、只做 unit-scale SE(3)”不能同时成立。
V8.10 必须把视觉中心线修复与关节/bind frame 修复拆开：前者使用固定关节端面的
cross-section rigid transport，后者只在长度兼容时才允许 unit-scale SE(3)。禁止为了
让 raw SMPL-X endpoints 同时命中而再次缩放骨头。

## 二、当前关节和联动由什么决定

| 部位 | 当前权威 | 当前问题 | V8.10 权威 |
|---|---|---|---|
| SMPL-X 动作 | 55 点 `rest_joints` 和 pose axis-angle | 适合给动作状态和大方向，不是解剖骨最终位置 | 保留为动作和肢体方向输入 |
| 解剖 bind/FK | 235 根 V71 层级、`target_bind_global/local` | 多个后处理使用不同局部基准 | 仍是运行时唯一 parent-local FK 权威 |
| 骨盆 | beta bind、髋臼域、V71 层级 | 当前整体联动基本正确 | 默认冻结；只接受严格 common-mode correction |
| 髋 | 股骨头/髋臼固定域和 `Femur_Rot.target_bone_head` | socket 正确不等于股骨干在大腿中心 | socket 决定 pivot，SMPL-X/body station 决定大腿方向 |
| 股骨 | 股骨头到髁中心的 whole-femur affine | socket、真实骨长和 body centerline 不能由一个刚体同时命中 | 头心/髁固定接触；中段 cross-section transport；frame 仅在长度兼容时做 SE(3) |
| 膝/小腿 | 髁/平台 gap 搜索，踝端固定 | 只保证局部 gap，当前会把小腿拉长约 3% | 平台/mortise 固定接触；中段 transport；可行时才更新 rigid frame |
| 踝/足 | 胫腓距骨域和 Talus/Calcaneus/Metatarsal similarity | ankle、foot station 不统一，L0 已缩足 | mortise + 足纵轴统一成一个 rigid compound frame |
| 血管/神经 | V71 235 bone、14-slot 权重、strict matrix LBS | 当前视觉可保留；L0 tube rest pack 烘焙顺序错误 | 冻结拓扑/权重；仅真实 bone-frame correction 预搬运 route |

`ba9c5e41683e06b27319a2ca022916489db48698` 本身只改 publish CLI，不能作为算法源码。
旧 `v251/v252 v232l2` 资产只作为“长骨在肢体中心”的几何和截图参考。V71 继续提供真正
的 235 骨层级、parent-local FK 和 14-slot 权重，禁止复制 BA9 的旧运行时骨架。

## 三、V8.10 的硬约束

1. 所有新增 bone-frame correction 都是相对 `rebuild_012` L1 baseline 的 `SE(3)`：
   `det(R)=+1`，三个 singular value 均为 1，不允许 uniform、axial 或 radial scale。
   视觉中心线 transport 的每个截面也只允许平移和旋转，不允许改变截面尺度。
2. 不独立把相邻子骨骼锚到 SMPL-X global joint。只生成一条完整 parent-local 链。
3. SMPL-X joint 只提供动作状态、腿部方向和 beta 相关 station，不直接覆盖解剖 bind
   origin。
4. 当前血管/神经的 faces、source ranges、vertex IDs、material edges、14-slot indices
   和 weights 必须 byte-exact 不变。
5. 不重新调用 `bake_vessel_route_v8`，不重新做整条 vessel route，不允许缩骨换 clearance。
6. 口腔组织只做 Genesis draw exclusion，不从资产删除顶点，不影响蒙皮或拓扑。
7. `rebuild_013` 生成 Genesis preview 后立即暂停；人工确认前不做 trusted latest。

## 四、建立三套互不混淆的腿部 station

V8.10 必须同时保存“解剖接触 landmark”“SMPL-X driver station”和“人体截面中心线”。
只用骨头自身 landmark 会把当前偏心轴重新定义成正确轴；只用 SMPL-X joint 又会破坏
股骨头、髁、平台和踝 mortise 的真实连接。

### 4.1 解剖 landmark

每个 beta 从固定材料域计算并冻结：

- 髋：股骨头球心、髋臼球心。
- 股骨远端：内外侧髁中心、髁间轴和两髁中点。
- 膝：内外侧胫骨平台中心、平台中点和平台横轴。
- 踝：胫骨远端、腓骨远端和距骨域形成的 ankle mortise center/frame。
- 足：距骨中心、跟骨中心、第一/第二跖骨前足 station 和足纵轴。

这些 landmark 决定关节 pivot、内外侧方向、接触 gap 和 compound roll，不决定整条腿在
人体包络里的横向位置。

### 4.2 SMPL-X driver station 与长度兼容报告

快速版使用 beta-specific SMPL-X `rest_joints` 提供动作方向：

- thigh direction：`hip -> knee`。
- shank direction：`knee -> ankle`。
- foot direction：`ankle -> foot`。

这些 raw joint 坐标不直接覆盖解剖 bind origin。每段必须先报告：

- `anatomical_length_m`；
- `smplx_station_length_m`；
- `axial_residual_m`；
- unit-scale two-endpoint solve 的最小残差。

若 residual 超过门槛，该段禁止做“同时命中两端”的 bone-frame correction。允许继续做
固定端面的中心线 transport，因此长度不兼容不会让整个 `rebuild_013` 在生成 preview
之前直接终止。

### 4.3 人体截面中心线

对当前 beta 的 SMPL-X/body surface 冻结：

- 大腿 25%、50%、75% 三个截面中心。
- 小腿 25%、50%、75% 三个截面中心。
- 踝和前足截面中心。

BA9 与当前 topology 完全一致，因此可用相同 vertex bins 提取旧版股骨/胫骨中心线。
BA9 只提供“Genesis 曾经看起来居中”的方向和相对 offset 参考；最终 target station
由 body 截面、当前解剖端点和 BA9 reference 共同决定，不复制 BA9 的半径、骨长或 bind。

为避免只对一个 subject 有效，L0 operator 保存紧凑的 beta station basis 和冻结的 bin
定义，不保存某一个 beta 的最终 correction。

## 五、骨盆 common-mode 判定

默认 `pelvis_correction = identity`。不允许凭 Genesis 视觉把整个骨盆向某侧平移。

离线计算：

- 左右髋臼中心相对 body hip station 的误差 `e_L / e_R`。
- `common = (e_L + e_R) / 2`。
- `differential = (e_L - e_R) / 2`。
- 由左右髋臼和骶骨/骨盆中线 station 拟合一个 unit-scale pelvis SE(3)。

只有同时满足下列条件才允许产生 pelvis correction：

- 左右髋误差能由同一个刚体变换解释，拟合后每侧残差 `<=2 mm`。
- 左右 differential `<=2 mm`，不存在一侧向内、一侧向外的局部错误。
- correction 平移 `<=5 mm`、旋转 `<=2°`。
- 骶骨、脊柱、肋骨和主血管的相对连接无回归。

任一条件失败都保持骨盆不动，判定为左右 femur local frame 或 mesh centerline 问题。
当前两个 beta 的 differential 都约 `56 mm`，已经明确不满足 common-mode 门。因此
`rebuild_013` 固定 `pelvis_correction = identity`，只输出诊断；最早到
`rebuild_014` 且新数据通过上述门时才重新考虑 pelvis correction。

## 六、拆分中心线 transport 与 bone-frame correction

`rebuild_013` 的快速路径先把当前 `rebuild_012` L1 股骨/小腿 articular 结果作为尺寸和
接触基线，不重新从 BA9 或更早产品重拟合整条腿。足部是明确例外：从 clean foot product
保留原尺寸，只做 rigid fit。在 `materialize_subject()` 的当前髋、膝/踝 reconstruction
之后、`with_source_driver_coupling()` 和 `bake_tube_coupling_v8()` 之前，新增
`repair_leg_centerline_v8()`。

该 pass 输出两个互相独立的结果：

- `centerline_transport`：只改骨 mesh rest geometry，不改关节 pivot 或 bind。
- `bone_frame_corrections`：只有长度兼容时才产生的 unit-scale `C_bone`。

### 6.1 先做 unit-scale 可行性判定

每侧分别计算股骨头-髁、平台-mortise、mortise-前足的 source/target 长度和最小刚体残差。

- 两端残差 `<=3 mm` 才允许该段进入 bone-frame correction。
- 残差更大时，`C_bone = identity`，保留当前 V71 parent-local bind 和接触 pivot。
- 禁止在这里调用 axial/uniform/radial scale fallback。

当前两个 beta 的股骨和小腿 raw SMPL-X endpoint residual 已明显超门，预计它们走
centerline-only 路径；这不是整个候选失败。足部主要是 frame/roll 问题，可独立做 rigid
compound correction。

### 6.2 股骨和小腿的固定端面中心线 transport

股骨固定：

- 股骨头和股骨颈核心域；
- 内外侧髁和滑车关节域。

小腿固定：

- 内外侧胫骨平台；
- 胫骨远端、腓骨远端和 ankle mortise 关节域。

对两个固定端面之间的 mesh：

1. 使用冻结 topology/material bins 建立纵向坐标 `s`。
2. 从 body 25%/50%/75% 截面和 BA9 对应 bins 得到 target centerline。
3. 每个截面只计算平移和必要的刚体 roll，不改变截面半径或面积。
4. 用 C2 权重沿 `s` 插值，固定域权重严格为 0；不得移动股骨头、髁、平台或 mortise。
5. 左右腿独立求解，不共享固定 offset。

该 transport 只修改 `Femur`、`Tibia`、`Fibula` 的 rest mesh 顶点。由于 bind、pivot 和
FK 不变，当前骨盆联动、膝/踝 parent-local 链和血管 runtime frame 默认保持不变。它解决
的是“骨 mesh 不在腿中部”，不伪装成关节 retarget。

### 6.3 可行段的 bone-frame correction

若某段通过 6.1：

1. 用解剖长轴、内外侧横轴和 body station 构成 source/target 右手 frame。
2. 计算唯一 unit-scale `C_bone`。
3. 对完整 mesh、对应 bind frame 和其必须共同移动的子树应用同一个 correction。
4. 不允许 child 再接受第二个 SMPL-X global anchor。

股骨或小腿未通过长度门时，不允许为了“看起来更靠近 joint”移动它们的 pivot；只保留
6.2 的 mesh centerline transport。

### 6.4 踝和足

1. 以 ankle mortise 为 pivot。
2. 用距骨、跟骨和前足 station 构造 foot frame。
3. 对 `Ankle_Rot` 完整子树和全部足骨 mesh 使用一个 unit-scale SE(3)。
4. 删除 `reference_fit_v8.py` 当前 `_proper_similarity()` 对足部产生的
   `0.95087 / 0.94028` scale，改为 rigid Kabsch fit。
5. rigid fit 不能同时命中全部 controls 时保留足骨尺寸、报告 residual 并出 A/B preview，
   禁止通过缩足自动通过。

### 6.5 髌骨和完整 FK

- `Patella_Rotate` 保持 V71 的真实 parent-local driver 关系，不独立锚定 SMPL-X knee。
- 纯 centerline transport 不移动 Patella rest frame。
- 若父级存在真实 `C_bone`，Patella 和对应 response 随 parent frame 一次搬运。
- 所有 target global frame 更新完成后，只调用一次 `_global_to_local()` 重建完整 local
  chain，并重算 inverse bind。

## 七、correction 必须传播到全部运行时权威

纯 `centerline_transport` 只更新选中的 bone mesh rest vertices，以下运行时字段必须
bit-exact 不变：

- `target_bone_head/tail`；
- `target_rest_global/local`；
- `target_inverse_bind`；
- `source_driver_coupling`；
- coupled RBF coefficients。

只有 6.1 通过并实际生成 `C_bone = G_new @ inverse(G_old)` 时，才在一个事务内更新：

- bone mesh rest vertices；
- `target_bone_head/tail`；
- `target_rest_global`；
- 由新 global 一次重建的 `target_rest_local`；
- `target_inverse_bind`；
- compound metadata 中的 anatomical pivot；
- coupled RBF 的 parent-local translation coefficients。

当前 `reconstruct_knee_ankle_compounds_v8()` 只刷新 pivot，没有变换
`rbf_weights_parent_local_m` / `rbf_values_parent_local_m`。V8.10 必须根据旧、新 parent
frame 和该 response 所属 correction，把每个 translation vector 变换到新 parent-local
坐标。不能只改 pivot 后继续使用旧坐标系的 glide/roll 系数。

所有 geometry/frame correction 完成后只调用一次 `with_source_driver_coupling()`。
零姿态必须严格 identity，pose 时仍由 V71 235 骨 parent-local FK 和 SMPL-X driver
rotation 驱动。

## 八、血管/神经冻结和同 frame 跟随

当前冻结基线：

- tube vertex count：`55337`；
- material edge count：`165659`；
- mesh count：`17`；
- topology digest：
  `765293284200c8d3a88204ce71c547aa767544092d1246ef02fd9a56ddf33ff5`；
- domain digest：
  `1e99d47507868fd6e5aa8394d6454147639607a507338d12ac4181a9bec317a0`；
- weight digest：
  `9e7e2f6ad8f9f451405fddcf01970b4b2dde588ecf18c72e083273215acd64ff`。

这些 topology/domain/weight 数据必须 byte-exact 保持。纯 centerline transport 不改变
bone frame，因此 tube rest coordinates 和 rest/content digest 也应保持。只有真实
`C_bone` 影响到 tube vertex 时，rest/content digest 才允许确定性变化。

### 8.1 预搬运

以当前成功的 vessel/nerve route 为 rest 基线：

1. 只收集 `C_bone != identity` 的受影响 tube vertices；纯 mesh transport 不搬血管。
2. 对这些顶点读取原始 14-slot bone indices/weights，使用对应的 unit-scale `C_bone`
   对 rest route 做一次离线 weighted rigid transport。
3. 优先用现有 dual-quaternion helper 做这一次离线预搬运，避免多个旋转矩阵线性混合
   导致截面缩塌。
4. pose-time backend 不改，仍使用当前
   `strict_matrix_lbs_14slot_v8` 固定矩阵求值。
5. 不改变 faces、ranges、vertex IDs、edges、indices 或 weights。

若 DQ 预搬运后的 edge/cross-section 回归超过 5%，立即回退该腿 correction；不重跑整条
route，也不缩骨。

centerline-only 分支预期 route 顶点差为 0，只重新测新的 bone clearance。只有 correction
后出现局部 skin/bone clearance 回归时，才允许对受影响腿部分支做
一次小范围离线 residual。该 residual 不改变拓扑、权重和分支连接，运行时仍无 KD-tree、
graph solve 或 collision solve。

### 8.2 修复 stale L0 tube pack

`run_anatomy_v8.py` 当前在 unified compose、vessel route 和最终 template 形成之前，就对
`merged` 烘焙 L0 tube pack。`rebuild_012` 持久化 pack 因此与最终 template 不一致；
最终 55337 个 tube rest vertex 全部不匹配，RMS 约 `32.2 mm`、最大约 `117.8 mm`。

V8.10 必须改为：

1. 完成最终 L0 template；
2. 完成所有 L0 rigid reference correction；
3. 再从最终 `operator.template_asset` 烘焙 tube pack；
4. L1 beta 和 leg correction 完成后，再从最终 subject rest asset 烘焙 subject tube pack。

最终 template 只读重烘的正确 parent baseline rest digest 是
`50825b335838838e2cfd925a55e87f4646bf44e945ba47c9104e6f64c70cac00`。
新增测试必须证明 L0 和 L1 零姿态都通过 exact rest authentication。

当前 route audit 本身不是绝对通过状态，因此 `rebuild_013` 使用相对 non-regression：

- skin inside fraction `>=0.99696756`；
- skin maximum outside `<=3.7799 mm`；
- bone penetration maximum `<=3.8166 mm`；
- clearance violation count `<=436`；
- 每个 pose 的 tube material edge max change `<=5%`。

不能为了追绝对 gate 破坏这次已经确认可接受的视觉结果。更严格的绝对 route 优化只能
在 `rebuild_014` 人工确认后另行决定。

## 九、口腔中舌状组织的显示策略

实际 `rebuild_012` source manifest 没有独立 `Tongue`、舌肌或 muscle tissue mesh。
按 `tongue/lingual/gloss/muscle` 搜索只找到：

- `Sublingual_Ducts_L/R`；
- `Sublingual_Gland_L/R`。

这四项已经在 `hidden_mesh_names_v1` 中 draw-only 隐藏。因此 Genesis 中仍穿过牙齿的
“舌头组织”不能再靠搜索 `Tongue` 名称解决。牙区附近仍有 `Pharynx`、
`UNCUT_Digestive_Tract`、Submandibular duct/gland 等候选软组织，必须先做 isolate
preview 确认。

V8.10 建立 `oral_visibility_policy_v2`：

1. 保留当前四个 sublingual mesh 的隐藏。
2. 对牙齿相交的剩余软组织生成 isolate render 和 mesh/face 清单。
3. 若穿牙部分不是独立 mesh，冻结 `hidden_face_ids_v2`，只从 Genesis draw list 排除
   口腔内的对应 connected face domain。
4. 不删除资产顶点，不整块隐藏 `Pharynx` 或 `UNCUT_Digestive_Tract`，除非 isolate
   preview 证明整块就是需要移除的组织。
5. 保留 skull、Mandible、upper/lower teeth、Hyoid_Bone 和合法口腔结构。

旧 gate 把“没有合法 Tongue mesh”无条件记成 release blocker，这与当前明确的
`no_tongue_display` 产品策略冲突。V8.10 改为条件门：若策略要求显示舌头，原有
provenance/license gate 继续生效；若策略明确不显示舌头，则使用 oral visibility gate：

- `tongue_asset_present=false` 可以是合法状态；
- hidden face IDs 不得出现在 Genesis draw list；
- 32 个牙 mesh 的 `11384` 个 face 必须全部保留；
- Mandible 的 `4254` 个 face 必须全部保留；
- Hyoid 的 `448` 个 face 必须全部保留；
- closed-mouth preview 中不得再有软组织穿过上下牙。

## 十、rebuild_013：快速中心线修复候选

只做本轮用户可见问题，包含：

- 口腔 isolate preview 和 `oral_visibility_policy_v2`。
- L0 足部 similarity scale 改为 rigid fit。
- beta-specific anatomical、driver 和 body-section stations。
- common-mode pelvis 诊断，`pelvis_correction` 固定为 identity。
- 左右 femur/shank 固定端面的 cross-section rigid centerline transport。
- 足部 unit-scale rigid compound correction。
- 只有通过长度兼容门的段才同步更新 bind、parent-local FK、inverse bind 和 coupled RBF。
- 只有真实 bone-frame correction 才预搬运 vessel/nerve route；centerline-only 分支保持
  tube rest bit-exact。
- 修复 L0/L1 tube pack 烘焙顺序。
- 213328 和 213712 的 T-pose，以及 Genesis 当前重点 pose preview。
- 髋、膝、踝局部剖面和腿部中心线 overlay。

生成后立即暂停，人工重点检查：

- 股骨干是否穿过大腿中部。
- 股骨头是否仍完整落在髋臼窝。
- 膝屈曲时股骨是否新增刺入髌骨。
- 胫骨/腓骨是否穿过小腿中部。
- ankle mortise、距骨、跟骨和前足方向是否符合 SMPL-X station。
- 骨盆、骶骨和脊柱是否保持当前联动。
- 血管拓扑、穿洞、贴皮肤压缩和骨骼跟随是否保持。
- 口腔软组织是否完全不再穿牙，牙齿、下颌和舌骨是否完整。

## 十一、rebuild_014：人工确认后的严格验收

仅在 `rebuild_013` Genesis 人工确认后执行：

- 固化 station/correction 到 L0 SourceOperator。
- 两个 beta 分别生成 L1 SubjectRuntimePack。
- 跑 `2 beta x (T-pose + 213328 pose + 213712 pose)`。
- 跑髋、膝、踝 `0-120°` sweep 和复合三轴 pose。
- 回归躯干、肘、骶骨、脊柱、肋骨和主血管。
- 生成局部剖面、body centerline、bone axis、signed penetration 和 vessel regression 图。
- 重测 cold bake、hot bake、L1 miss/hit 和 pose latency。
- 启动新的独立 agent，只给规范、候选和输入，不给“已经修了什么”，做盲审。

## 十二、验收标准

- 股骨头心到髋臼中心误差 `<=2 mm`。
- 25%/50%/75% 截面 transport 后到其冻结 target station 的残差 `<=3 mm`，并且相对
  BA9 Genesis 可接受中心线不出现新增横向回归。
- 膝和踝表面 gap `0-3 mm`。
- 无新增股骨-髌骨、股骨髁-胫骨平台 signed penetration `>0.5 mm`。
- 所有 bone-frame correction scale 精确为 1。
- centerline transport 的固定关节域位移 `<=0.5 mm`，截面面积/半径变化 `<=1%`，
  local edge change q99 `<=3%`、max `<=5%`。
- 相对 `rebuild_012` L1 baseline 的股骨和小腿长度不变；足骨保持 clean foot product
  原尺寸。raw SMPL-X 长度残差只决定 frame correction 是否可用，不触发自动缩放。
- 完整 parent-local FK 保持，零姿态 skinning transform 为 identity。
- coupled RBF 的 pivot 和 parent-local translation coefficients 使用同一个新 frame。
- tube topology/domain/weight digest 与冻结基线完全一致。
- tube edge/cross-section 变化 `<=5%`，skin/clearance 不低于当前成功基线。
- oral hidden faces 不在 draw list，牙齿、Mandible、Hyoid face count 完全保持。
- `2x3` 矩阵证明同一个 operator 学到 beta-dependent 联动关系，不是某个 beta 的固定位置。
- `rebuild_013` 的 pelvis correction 必须为 identity。
- 任一 vessel 回归优先回退对应 leg correction；禁止重新缩骨或破坏 vessel topology。
- 独立盲审和 Genesis 人工确认都完成前，不更新 trusted latest。

## 十三、最小代码改动边界

- `reference_fit_v8.py`：足部 similarity 改为 rigid SE(3)，写入口腔 visibility metadata。
- `articular_fit_v8.py`：新增 station/length feasibility report、
  `repair_leg_centerline_v8()` 和条件式 RBF frame transport；保留当前 L1 articular
  结果作为快速版尺寸/接触基线。
- `v8_artifacts.py`：在当前 hip/knee-ankle reconstruction 后、driver coupling 和 tube
  bake 前调用 centerline/frame pass。
- `anatomy_lbs.py` / `coupled_joint_v8.py`：复用或新增明确的 parent-local response
  transport helper。
- `tube_frames_v8.py`：增加离线 tube rest correction 和 final-rest authentication。
- `cli/run_anatomy_v8.py`：把 L0 tube pack 烘焙移动到最终 operator template 之后。
- `anatomy_drawer.py`：支持 `hidden_face_ids_v2` 并验证 draw-list exclusion。
- `validation_matrix_v8.py` / `release_v8.py`：把强制 Tongue provenance blocker 改为
  oral visibility policy。
- 测试：新增 SE(3) feasibility、固定端面、截面尺度/edge、骨长、common-mode pelvis、
  station error、条件式 RBF frame transport、tube exact digest/rest auth、口腔 draw
  preservation 和 2x3 regression。

## 十四、长度不兼容后的约束修订：mixed-anchor projected chain

213328/213712 的真实资产复核证明，不能再把 raw SMPL-X 两端点同时命中作为
`rebuild_013` 的发布条件：

- 左股骨解剖长度比 SMPL-X hip-knee 短约 `26.56-27.26 mm`，右侧短约
  `7.81-8.63 mm`。
- 小腿解剖长度比 SMPL-X knee-ankle 短约 `12.86-15.86 mm`。
- 任意 unit-scale SE(3) 都保持长度，因此无法同时命中上述两端。
- 现有端点固定的逐顶点 rotation field 在真实股骨过渡区产生
  `208-214%` q99、`222-243%` 最大 edge strain，必须废弃。

`rebuild_013` 改用保长、全段刚体的 mixed-anchor projected chain：

1. 股骨使用 `distal_anchor`。固定当前髁/膝 station，取 SMPL-X
   `hip -> knee` 的方向，以真实股骨长度反投影股骨头。整根股骨只接受同一个
   unit-scale SE(3)，不再弯曲股骨干。
2. 胫骨和腓骨使用共享的 `proximal_anchor`。固定最终胫骨平台 station，取
   SMPL-X `knee -> ankle` 的方向，以真实小腿长度投影 ankle mortise。
3. 足部使用投影后的 mortise 作为 `proximal_anchor`，只做 rigid direction/roll fit，
   不使用 similarity scale。
4. pelvis root、Sacrum、Spine 和骨盆中线保持 identity。左右股骨头与髋臼的剩余冲突
   写入 `hip_station_unreachable_with_fixed_socket`，只能由后续 bilateral local
   socket/head station 修正处理，禁止移动整个骨盆。
5. 所有 global bind 更新结束后只重建一次完整 parent-local chain。运行时继续使用
   当前闭式 leg hinge solve：SMPL-X 提供 pose rotation/direction，解剖 local bind
   提供固定长度和 translation，因此不增加逐帧迭代或 IK 优化。

新的验收语义：

- `<=3 mm` 只约束股骨干、小腿和足部相对 body centerline 的径向误差。
- SMPL-X 与解剖骨长的轴向残差必须完整报告，但不触发缩骨、拉骨或整段失败。
- 股骨髁、胫骨平台和 foot mortise 的选定 anchor 漂移 `<=0.5 mm`。
- 全段 correction 必须满足 `det(R)=1`、scale `=1`，完整骨 mesh 的 edge length
  在数值精度内不变。
- 股骨头-髋臼接触误差继续单独报告。若与 body centerline 不能同时达到门限，
  `rebuild_013` 必须标记为未受信候选并等待 Genesis A/B 检查，不能伪造通过。
- 股骨禁止使用 `proximal_anchor`：真实数据会把膝自由端推到约 `55-57 mm` 误差。
- 小腿 projected free ankle 对 raw SMPL-X ankle 的预期残差约 `6.27-8.01 mm`；
  这是保长链的合法轴向/端点残差，不是 uniform scale 的理由。

BA9/v251 仅保留为 centerline audit 和 Genesis A/B 参考，不再提供 beta-linear
逐顶点 rotvec、bind、FK、权重或强制两端目标。血管 rest route、拓扑和 14-slot
权重保持不变；只运输 parent-local frame/RBF 坐标并在最终 rest 后重新认证 tube pack。

## 十五、长度与宽度都不兼容后的最终快速路径：morphology adapter + guide FK

`mixed-anchor projected chain` 仍不能同时满足髋臼接触和大腿视觉中心线。真实数据已经
证明这不是调一个 pivot 可以解决的问题：

- BA9 左右股骨头相对当前髋臼约有 `64-76 mm` 的分离；它能作为中心线参考，不能作为
  合法髋关节 rest pose。
- 当前股骨保持髋臼接触时，BA9 对应的五个 shaft station 横向差约从近端
  `50-58 mm` 逐步降到远端 `9-11 mm`。
- 股骨/小腿的 raw SMPL-X 长度残差分别约 `26 mm` 和 `13-15 mm`。因此“关节接触、
  BA9 中心线、原骨长、全段刚体”四项无法同时成立。

V8.10 不再把 raw SMPL-X 世界关节点当解剖端点，而采用两层模型：

1. **解剖 guide skeleton**：髋臼/股骨头、髁/平台、mortise 和足部 station 决定
   beta-specific `H/K/A/F` rest joints。每段保留自身解剖长度和 parent-local hierarchy。
2. **SMPL-X motion skeleton**：只提供 pelvis root motion 和各关节 local rotation。
   同一组 pose axis-angle 在 guide rest joints 上再跑一次 55-joint FK；禁止把 raw
   SMPL-X posed translation 复制回解剖链。

这使 Blender 与 SMPL-X 的宽度、腿长不一致变成冻结的 pelvis-local morphology offset，
而不是每帧 IK 误差。运行时仍由 V71 的 235 骨 FK、原 14-slot 权重和 coupled response
求值；新增的一次 55-joint guide FK 是固定矩阵运算，不做迭代。

离线 rest geometry 使用可选的 **cap-preserving axial adapter**：

- 只有用户明确要求 raw H/K/A/F station 同时命中时，才允许在 shaft 上吸收长度残差。
- 股骨头/颈核心、髁/滑车、胫骨平台和 ankle mortise 是刚性 cap；横截面尺度保持 1。
- 位移只沿段轴，用低峰值 C1/C2 profile 分布到 shaft，运行时 bind matrix 仍是
  `det(R)=1, scale=1`。
- 这不是 uniform bone shrink。股骨约 `26 mm` 的残差意味着约 `7%` 平均轴向适配，
  数学上不可消除；横截面、关节面和骨宽不得随之缩小。
- 以标量 `lambda` 选择满足 axial Jacobian、局部 edge strain 和血管回归门的最大幅度。
  未吸收的长度写入 `remaining_station_residual_m`，禁止静默兜底。

默认 `rebuild_013` 选择 **contact-first guide mode**：

- pelvis frame、Ilium、Sacrum、Spine 和当前髋臼 rest geometry 不动；
- 不采用把左右 Ilium 各自内移约 `60 mm` 的候选；
- 股骨头/髋臼与膝踝接触优先，BA9/body centerline 是带应变上限的软目标；
- 足部继续只做 rigid fit；
- vessel rest route、拓扑和权重保持不变，只在 bind frame 改变时运输局部坐标。

验收必须分别报告 radial centerline residual、raw SMPL-X axial residual、contact residual
和 adapter strain，不能再用一个 `<=3 mm` 数字假装四类约束可以同时满足。若 Genesis
仍要求完全复现 BA9 的近端中心线，则必须单独批准 pelvic-width morphology；这会改变
髂骨形状，不能伪装成 pelvis SE(3) 或“无缩放”修复。

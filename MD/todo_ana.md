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
# 五机躺卧 SMPL-X benchmark

## 动态后暂停图片的稳定性实测（2026-10-03）

用户随后明确：部署不考虑本次右腿遮挡情形，重点区分正常观测下的小幅晃动和发散。
因此遮挡组仅保留为压力诊断，不纳入当前部署稳定性判断。正常观测组在本次30秒
范围内表现为初始收敛后的小幅有界波动：最后10秒各body22关节偏离其时间均值的
P95为1.56mm、最大3.07mm；全身平均GT误差保持在30.76–31.32mm。5秒窗口平均
误差依次32.12、30.99、30.91、30.97、31.05、31.03mm，没有持续增长的趋势。
约31mm的位置偏差与毫米级的时序晃动应分开描述；本次并未验证任意长时间运行。
分窗结果见实验目录 `observed55_bounded_fluctuation.json`。

按用户要求，实际运行完整DWPose、三角化、HybrIK RGB先验和80步姿态拟合：先动态，
随后固定同一时刻五路原始JPEG，继续更新60次（采集时间每次增加0.5秒，暂停共30秒），
再恢复动作。每个场景内复用同一跟踪器，无暂停重置；初始化beta和所有跟踪权重不变。
这是按指定采集时钟离线处理，未启动UE发布/显示闭环，也未按墙钟睡眠控制到2Hz。

两组分别暂停在source55（25个身体点均有效）和source100（右髋、右膝等缺失）。
动态段分别4秒/8.5秒，恢复段分别4帧/3帧。每组静止60/60输出成功，图片路径和
SHA256保持相同，保存的67点三角化数组及置信度逐项完全相同。beta全程变化为零。
遮挡组恢复第一帧source105仍被质量门限拒绝；记录和视频均保留这次失败。

以下暂停位移均以最后一个动态输出为基准（它与随后的静态输入是同一组图片）：

| 指标 | source55观测完整 | source100右腿遮挡 |
|---|---:|---:|
| 暂停后右膝最大位移 | 6.14mm | 209.84mm |
| 暂停后骨盆最大位移 | 22.11mm | 24.26mm |
| 暂停后body22任一关节最大位移 | 40.29mm（spine1） | 209.84mm（右膝） |
| 最后10秒body22相邻更新位移P95 | 1.99mm | 2.52mm |
| 最后10秒body22端点净位移最大值 | 3.79mm | 15.28mm |
| 最后右膝GT位置误差 | 21.57mm | 75.47mm |

若以第一个freeze输出为基准，右膝最大位移分别2.50mm/192.32mm；这会漏掉暂停
后的第一次调整，因此不能和上表混用。遮挡组右膝GT误差从第一个freeze输出的
252.94mm降到75.47mm，主要是在纠正先前的错误姿态。正常组也有初期厘米级收敛，
随后进入毫米级波动。遮挡组虽然每步变化已较小，尾10秒右膝仍累计移动11.26mm。
本次没有观察到30秒内无界发散，但不能宣称静止输入会立即固定输出或所有关节均稳定。

与实现相符的解释是：每次都受当前观测约束，但仍以更新后的上一姿态作下一次的
优化起点和时间先验；有限步优化可以在同一观测下继续收敛，缺失的髋膝约束尤其弱。
“持续观测纠偏”不等于“所有部位都得到充分观测”。本轮只测试和记录，未修改跟踪算法。

工件目录：`outputs/realus_sync/tests/surface_motion_audit_20261003/freeze_tracking_20261003`。
`observed55_audit/`、`occluded100_audit/`各有完整summary和曲线；两组input_audit.json
记录静态输入一致性。`videos/observed55.mp4`、`videos/occluded100.mp4`以2fps展示
cam1和cam5，保留运动/暂停/恢复阶段，拒绝帧明确标记视频沿用上一估计，未当成新输出。
派生录制不含canonical姿态，gt_path只用于事后评价。评估入口为
`perception/apps/audit_smplx_freeze.py --sequence-dir ... --protocol-json ... --output-dir ...`。

## 当前验收重点：大关节贴合与持续更新

用户允许少数遮挡帧的局部腕部朝向偏差。优先检查躯干、骨盆、大腿、膝和小腿是否
贴合，以及关键点持续更新、丢失和恢复时是否出现异常错位或抽动。不能只以全身
平均误差或更平滑的曲线验收；应分别报告左右膝踝位置、膝屈曲几何角、相邻更新
相对真实运动的位移误差，并把低频更新和密集离线回放分开。

当前实现从上一后端接受姿态继续拟合，beta冻结。前端质量拒绝不进入后端，后端
step异常会恢复姿态、先验、根位置和时间戳；拒绝结果不发布。下游接收端的保持/
插值行为不在本次实测范围。当前尚无最终身体拟合残差、膝角或关节位移跳变的
硬接受门限，通过现有检查的错误观测仍可能影响下一帧。

注意：tracking YAML中的 `temporal_jump_reset_m`、`dropout_hold_frames` 和
`recovery_rms_hold_m` 属于另一套 `realtime_smpl_fitter`，当前 HybrIK 后端不读取
这些参数。腿部遮挡项是踝足世界朝向的软约束，不能称为膝关节恢复跳变拦截。

独立复核发现此前右腿遮挡压力段存在大关节错误，不能被下方全身均值改善掩盖。
按用户随后明确的部署范围，这些遮挡段不作为当前验收场景，仍保留记录：

| 部位 | 密集101帧均值/最大位置误差 mm | 2Hz有效20帧均值/最大位置误差 mm |
|---|---:|---:|
| 左膝 | 19.33 / 38.30 | 23.11 / 31.14 |
| 右膝 | 33.62 / 175.32（frame98） | 52.82 / 270.54（frame100） |
| 左踝 | 24.95 / 38.77 | 25.83 / 34.02 |
| 右踝 | 20.74 / 114.90（frame102） | 17.77 / 46.89 |

膝的几何屈曲角平均误差约3–5度，不代表其空间位置正确。密集输入右膝113→114
更新的位移向量相对真实运动偏差118.73mm，右踝102→103为127.30mm；2Hz右膝
110→115的偏差220.22mm。2Hz帧105被拒后，frame110右膝仍偏246.5mm。右腿
遮挡时的空间位置和恢复更新是当前主要未解决问题。GT全部用于事后评价。

对已有HybrIK网络UVD缓存做独立三角化验证：三视角15px以及22px（当前DWPose
gross门限）都不能覆盖最坏右膝帧98、100。因此本轮未启用RGB膝点替换，也未全局
放宽门限。保留当前模型/跟踪参数作对照，不能将“输出有限且未拒绝”当成准确性通过。

原始报告：`outputs/realus_sync/tests/surface_motion_audit_20261003/major_joint_audit.json`
和 `rgb_knee_fallback_audit.json`。问题帧图为同目录 `major_joint_worst_frame_98.jpg`。

按保存的右膝3D观测是否有效做事后分组（不按GT筛帧）：密集输入有效81帧的右膝
位置误差均值20.22mm、P95 26.20mm、最大38.95mm；无效20帧均值87.86mm。
2Hz有效17帧均值21.28mm、P95 30.31mm、最大47.04mm；无效3帧均值231.50mm。
相邻两次右膝都有观测时，更新位移相对真实运动误差均值分别6.47/6.99mm。
这支持右膝严重错误主要集中于无可靠3D观测的情况，不代表所有其他关节均有观测
或网格表面零误差。分组报告为同目录 `knee_observation_conditioned_accuracy.json`。

## 当前待改进基线：80步、观测几何检查与遮挡保护（2026-10-03）

保留已经验证的同人轮廓beta，跟踪期间不改变体型。DWPose负责多视角位置，完整
HybrIK-X RGB网络负责局部旋转先验；previous姿态初始化、80步、时间权重0.09、
RGB权重0.01继续使用。本轮重点修正异常观测和遮挡时的姿态恢复，没有换新模型。

修正了以下具体问题：

1. 三角化最终refit后，三视角核心点也必须通过gross重投影门限。原先右肘48.77/
   61.89像素残差仍能被当成高可信观测。现在拒绝不合格候选，保持核心点至少三视角。
2. 根据冻结beta的rest模型计算手部路径长度和可达范围，屏蔽离腕42–47厘米的MCP、
   51厘米指段等不可能观测。身体腕和手根按掌部观测支持选择；没有任何掌指支持的
   孤立腕点也不能强拉整只手。无腕锚点时不让低置信度点群牵引整个手臂。
3. 膝/踝缺失时，只给对应腿链的世界朝向增加保守时间约束，权重0.0005。约束覆盖
   上游膝踝传到脚掌的旋转；完整观测时该项为零，不对全身统一增加平滑。
4. DWPose骨盆/双髋不足时，可使用HybrIK-X输出的骨盆2D点，经真实标定至少三视角、
   15像素门限、正深度和最终refit验证，作为独立备用骨盆观测。它不会改写缺失的
   DWPose点；原始质量失败和root来源都保留在诊断中。只允许特定min_core_body25
   失败、至少12个身体点的情况延后检查，其他质量失败仍拒绝。

手部检查用同一冻结beta解码的101帧标准动作做独立评价：6767个合法标志点没有被
误剔除。GT只用于评价，未参与推理、筛点、网络裁剪或初始化。

完整原始RGB复跑 `onbed10s_stable_fresh_s80` 为101/101成功，10秒源动作：

| 指标 | 原80步 | 当前80步 |
|---|---:|---:|
| body22世界关节点平均误差 mm | 30.74 | 30.33 |
| hands30世界关节点平均误差 mm | 15.94 | 15.87 |

腕脚异常跳变也按真实运动核对，不能只看曲线更平滑：

| 源帧/部位 | 原估计转动 | 当前估计转动 | 真实转动 |
|---|---:|---:|---:|
| 25→26 右腕 | 64.83° | 23.33° | 22.65° |
| 45→46 右腕 | 44.48° | 7.41° | 5.60° |
| 96→97 左腕 | 44.77° | 15.81° | 11.06° |
| 104→105 右脚 | 42.82° | 11.23° | 0.84° |

整体右腕绝对朝向均值/P95由17.53°/48.94°改善到14.96°/41.59°；右脚由
20.00°/42.72°到19.55°/38.12°。仍有稳定但不准确的遮挡姿态：例如frame45右腕
绝对方向误差仍52.49°，不能称为“所有帧完全无扭曲”。颈部均值13.08°→13.17°，
没有明显改善。当前结果是位置精度基本保持、异常翻转显著减少。

当前完整moment计算平均575ms、P95 727ms。从同步RGB就绪计到重建完成，包含RGB
推理、融合和函数内输出写入，不含源图读取/归一化、冷启动、渲染和发布。视频按
源10Hz回放，不能当成10Hz实时处理。beta与原初始化逐项相同。

2Hz源采样另测 `onbed10s_stable_fresh_s80_2hz`：20/21成功；frame105的核心身体点/
有效肢段不足，cam2检测丢失后旧裁剪框也不能提供足够观测，因此保留拒绝。不能用
删去此帧后的平均误差直接和原21帧均值比较。相同20帧上，身体误差34.15→33.68mm，
手部18.20→19.78mm，手部平均退步1.58mm；低频下减少跳变不等于整体精度提高。
匹配帧评价见稳定性报告。当前仍不能保证所有遮挡场景无误差，也没有真实皮肤表面
PVE验证。

在下方80步命令的基础上启用：

```bash
--hybrik-observation-geometry-filter \
--hybrik-occlusion-rotation-weight 0.0005 \
--hybrik-rgb-root-fallback
```

建议使用新run-name。以上选项已接入实际recorded/live入口，原始默认路径保留。
当前结果目录为 `outputs/realus_sync/tests/surface_motion_audit_20261003`，最终比较
见 `stability_fix_comparison.json`；完整视频为 `onbed10s_stable_overlays/videos/montage.mp4`。
中间 `onbed10s_guarded_*` 是修复孤立错误腕锚点/骨盆备用观测前的诊断结果，不作为推荐。

## 60 / 80 / 100 步复核：常规配置选 80 步（2026-10-03）

同一床上10秒动作、同一101帧观测、固定轮廓beta和五路批量RGB先验，补跑80、100步。
三个配置都101/101成功。其余配置保持previous初始化、temporal_weight=0.09、
rgb_pose_weight=0.01。这里步数指每帧Adam姿态拟合次数，两套图像网络不重复运行。

| 步数 | body22 mm | hands30 mm | 后端均值 ms | 全流程分项和 ms |
|---|---:|---:|---:|---:|
| 60 | 31.42 | 16.29 | 163.90 | 522.82 |
| 80 | 30.74 | 15.94 | 207.54 | 566.42 |
| 100 | 30.60 | 15.93 | 257.63 | 616.58 |

上表是10Hz源输入离线逐帧处理；前端/RGB复用同一缓存计时，后端本次计时，不能当作
新测实时端到端。80→100步身体均值改善0.14mm、手部0.002mm，多约50ms。
右脚全局旋转P95依次为48.00°、42.72°、39.55°；同形状足部表面标志点三轴
平均方向误差P95为35.20°、33.93°、32.49°。左腕/头部等指标不随步数单调改善。
后者是方向诊断，不是真实皮肤法线或表面PVE。

另外从原始RGB运行相同片段的2Hz源采样，三种配置均21/21成功：

| 步数 | body22 mm | hands30 mm | 实测moment均值 ms | P95 ms |
|---|---:|---:|---:|---:|
| 60 | 37.70 | 19.72 | 497.30 | 631.15 |
| 80 | 34.75 | 18.40 | 538.87 | 648.82 |
| 100 | 32.77 | 18.17 | 594.69 | 739.01 |

低频时前后帧运动差更大，100步对身体位置仍有约2mm额外收益，不能将密集输入的
饱和结论直接推广。综合手部收益、局部朝向和耗时，常规配置选80步；60步保留作
延迟优先配置，100步用于更重视低频身体位置精度的场合。这是当前片段的工程折中，
不是所有人物/动作的最优值。80步moment计算均值约0.54秒、P95约0.65秒，尚未
稳定满足2Hz；计时从同步RGB就绪到重建完成，不含读取/归一化、冷启动、渲染和发布。

当前后处理是固定beta的SMPL-X姿态/根平移优化，目标同时含置信加权67点Huber误差、
RGB局部旋转约束和按源时间归一的上一帧旋转约束。previous模式每帧跳过解析IK，
不会反复套用静态twist。网格通过正常SMPL-X蒙皮生成，没有直接拉伸顶点。
多视角3D点约束位置，RGB补充绕骨轴旋转信息；遮挡和关键点语义误差仍会影响结果。
当前跟踪不含逐帧轮廓、显式关节限位、自碰撞或床接触损失，关节点准不等于表面完全准。

性能诊断（此前fresh101帧60步）：DWPose检测42.7ms、CPU三角化181.4ms、批量RGB
含融合129.4ms、后端177.8ms。后续提速应优先检查三角化和CPU/GPU重叠机会；
当前仍顺序执行，没有将潜在重叠收益计入任何实测结果。

常规运行沿用下方命令，将 `--hybrik-refine-steps 60` 改为
`--hybrik-refine-steps 80`，并使用新run-name。未改变历史CLI默认行为。
详见 `outputs/realus_sync/tests/surface_motion_audit_20261003/steps_60_80_100_comparison.json`。

## 前轮60步配置与验证记录（保留对照，2026-10-03）

前轮建议保留 DWPose。它提供经标定三角化的位置，HybrIK-X 完整 RGB 网络提供
关节局部旋转先验；共同驱动冻结 beta 的 SMPL-X。五视角 RGB 一次 batch，float32、
官方 flip test 保留；previous pose 初始化、60 步、temporal_weight=0.09、
rgb_pose_weight=0.01。不继续扫描更多步数或新增替代模型。

体型来源已逐项核验：新旧两个视频各101帧的10维 beta 完全相同，差值为0。
视频使用旧同人 `optimized5_neutral_silhouette20_final` 的单帧、五视角、20次轮廓
优化结果（DWPose 提示官方 SAM 2.1 Hiera Tiny mask）。不是本轮10秒片段新估计，
也不是 HybrIK-X 首帧 beta。原 baseline→轮廓初始化使 rest height 从1584.81变为
1656.80 mm，胸中段厚度从241.11变为227.23 mm；新 RGB pose 视频继续改善的是
姿态及扭转。此前提到的“骨长 refit 把人体压短”是未采用的独立实验。
详见 `beta_provenance_clarification.json`。

初始化原则是多相机共享一个 beta，由 RGB 体型候选提供起点、轮廓约束体表，
关键点帮助定位姿态；初始化完成后冻结。当前同人场景复用已验证的轮廓 beta。
改变人物时应重新初始化，不能把这份同人 beta 当成通用人体模型，也不把静态骨长
拟合当成最后一步强行覆盖轮廓结果。

同片101帧的速度/精度对比（前3行时间为独立分项之和，非实时流端到端）：

| 方案 | body22 mm | hands30 mm | 秒/五机组 |
|---|---:|---:|---:|
| 原RGB串行 + 300步 | 30.82 | 15.85 | 1.564 |
| RGB batch + 60步 | 31.42 | 16.29 | 0.523 |
| RGB batch + 30步 | 36.79 | 19.49 | 0.441 |
| 去掉DWPose点，RGB姿态融合，共享轮廓beta | 50.08 | 112.63 | 未测含检测器完整流程 |
| 去掉DWPose点，RGB姿态融合，首帧RGB beta | 56.23 | 122.74 | 未测含检测器完整流程 |

30步的右脚表面三轴方向 P95 从60步35.20°恶化到59.77°，不值得为约82ms采用。
60步 batch 的左右腕全局方向均误差13.56°/17.52°，左右脚13.97°/20.69°，颈13.03°。
纯 RGB 对照用真实标定三角化网络 pelvis UVD、外参变换根旋转、SO(3)多数融合、
旋转EMA与固定形状解码。两个版本101帧均成功，证明其能连续跟踪；它是直接融合
基线，不是对所有纯 HybrIK-X 多视角优化方法的穷举，也不证明 DWPose 理论上不可替代。
共享轮廓 beta 的版本仅移除 tracking 阶段的 DWPose 点，初始化仍复用轮廓身份；
RGB beta 版本不读原初始化。两者都沿用缓存检测/历史框，禁止DWPose点生成的bbox。

批推理实测平均115.45ms/五机组，对原串行500.73ms约4.34倍。官方数学/权重/crop
未改，但浮点差在少数解析IK手指分支被放大：全片 local rotation 平均差0.034°、
P95 0.105°、最大5.59°；XYZ最大4.09mm。因此没有只凭首帧宣称完全等价，以上
batch精度是重新跑完整跟踪所得。默认串行入口保留，可选 `cache_hybrik_rgb.py
--batch-views` 与实际链路 `--hybrik-rgb-pose-weight 0.01` 启用批量RGB先验。

结果目录仍为 `outputs/realus_sync/tests/surface_motion_audit_20261003`，汇总在
`sweetspot_comparison.json`。实际逐帧入口可用：

已从原始RGB运行实际入口 `onbed10s_sweetspot_fresh`，101/101成功，所有网格顶点
与对应batch缓存实验逐项完全一致，见 `sweetspot_fresh_parity.json`。完整moment
计算平均0.534秒、P95 0.667秒：前端0.225秒、RGB含传输与融合0.129秒、优化
0.178秒。范围从同步RGB已就绪至重建完成，含函数内小量输出写入，不含图像读取/
normalization、冷启动、后续渲染和发布；不是10Hz实时，也尚未保证稳定2Hz。
评估器已修正为优先读完整 `latency_s`，避免旧的 frontend+backend 相加漏算RGB。

另用 `--sample-stride 5` 验证源时间2Hz的实际处理：`onbed10s_sweetspot_2hz_source`
21/21成功，body22 37.70mm、hands30 19.72mm，平均计算0.497秒、P95 0.631秒。
这比10Hz输入逐帧离线处理更难，不能把31.42/16.29mm直接宣称为2Hz实时精度。
未继续调参；当前配置作为实用折中，低频和遮挡段仍有误差。

```bash
A=outputs/realus_sync/tests/surface_motion_audit_20261003
PYTHONPATH=.:src python -m projects.genesis_ue_sync.multiview_realtime.cli.run_delayed_smplx_sequence \
  --config "$A/onbed10s_tracking.yaml" --recording-dir "$A/onbed10s_recording" \
  --output-root "$A" --run-name onbed10s_sweetspot_new \
  --gender neutral --tracking-backend hybrik --initialization-npz "$A/onbed10s_silhouette_beta_init.npz" \
  --hybrik-device cuda --hybrik-refine-steps 60 --hybrik-pose-initialization previous \
  --hybrik-temporal-weight 0.09 --hybrik-rgb-pose-weight 0.01 \
  --triangulation-workers 3 --triangulation-pair-hypotheses 1 \
  --dwpose-temporal-bbox-fallback --disable-face-iteration --no-write-debug-images
```

## 表面与时序诊断（2026-10-03，新增实验）

工件目录：`outputs/realus_sync/tests/surface_motion_audit_20261003`。
用户反馈的五机截图对应旧主结果 frame45。以下诊断将静态体型和动态旋转分开测量，
GT 仅用于选取床上且在视野内的测试区间以及推理后的评分。

旧 75 帧的 beta 逐帧完全相同。相邻 body22 局部旋转的均值为 10.96°、P95 为
35.23°、最大为 105.35°；相同 canonical 动作的均值为 0.405°、P95 为 1.687°。
这说明明显的跳变来自估计，不是体型逐帧变化。原后端每帧重新运行解析 IK，
`phis` 始终取静态初始化；refinement 的微弱 6D 正则只锚定当帧 IK seed，没有
上一接受帧的旋转约束。稀疏关节不能唯一确定所有骨段的轴向扭转。

新增可选 `--hybrik-pose-initialization previous` 和 `--hybrik-temporal-weight`。
previous 模式使用上一接受姿态精修当前真实观测点，要求 refinement 步数大于零。
时序项比较旋转矩阵，并按采样时间间隔归一；拒绝帧不提交内部姿态或时间状态。
默认参数继续复现既有解析 IK 基线，便于独立对照。

同一组缓存 RGB 三角化、同一初始化 beta、60 步 refinement 的旧 75 帧对照：

| 方法 | body22 MPJPE mm | hands30 MPJPE mm | 相邻 body22 旋转 P95 | 局部旋转对 GT 均误差 |
|---|---:|---:|---:|---:|
| 每帧解析 IK，原结果 | 44.10 | 11.39 | 35.23° | 23.51° |
| previous，无新增时序项 | 44.74 | 14.11 | 16.37° | 29.56° |
| previous，时序权重 0.01 | 44.22 | 9.08 | 5.00° | 23.94° |

时序平稳性显著改善，但绝对 twist 未全面改善；例如左脚和右腕仍存在稳定但错误的
旋转。不能将平滑度改善写成全部表面扭曲已经解决。缓存重放只重新测量 backend；
summary 中与历史 frontend 耗时相加的结果是分项估计，不是本轮实际端到端测速。

`shape_audit.json` 在 rest pose 比较当前标准 neutral10 网格与 BEDLAM locked-head
neutral16 身份参考，使用同拓扑、骨盆平移对齐以及排除手臂的三个躯干截面。
这是模型空间体型诊断，不是 UE 渲染皮肤 PVE，也不是临床胸围测量。

| 截面（spine2→neck） | 当前前后厚度 mm | 身份参考 mm | 旧 RGB silhouette20 beta mm |
|---|---:|---:|---:|
| 25% | 255.62 | 219.4 | 230.38 |
| 50% | 244.50 | 220.0 | 227.23 |
| 75% | 202.76 | 190.4 | 192.84 |

当前主要是前后厚度偏大，宽度并非处处偏大，rest 身高反而短约 34 mm；整体缩放
不能解决该比例问题。当前躯干 rest 表面 RMS 差异约 14.47 mm；把身份前十维放入
标准模型再与 locked16 参考比较，该躯干差异约 1.09 mm，因此本例不能主要归因于
模型版本或少六个 beta。体型初始化只有 13 个骨长观测，缺少直接的胸廓表面约束。

旧 Small/Large RGB beta prior 使中段截面周长约为 1007/1030 mm，相比当前 973 mm
和身份参考 942 mm 更偏大。此前只用 MPJPE 判定 beta 优劣不足；应测轮廓、胸廓
截面与表面距离。旧 RGB silhouette20 候选中段周长约 949 mm，可作同人初始化
候选；它是先前独立 RGB 实验的估计，复用时必须记录来源，不能称为本轮新拟合。

参考：官方 [HybrIK-X](https://jeffli.site/HybrIK-X/) 用骨架解析求 swing、用图像网络
估计 twist；本工程的解析 IK 接口并未运行该网络。[EasyMocap](https://github.com/zju3dv/EasyMocap)
本身支持多视角骨架拟合 SMPL-X；骨架驱动属于常规路线，但需要形体、扭转和时间先验。

### 10 秒床上长片复测

正式输入为上述工件目录下的 `onbed10s_recording`：CMU 114_11 source 500:12:1700，
101 个不重复运动姿态，按真实源时间 10 Hz 播放为 10.1 秒；另有 15 个静态初始化
帧，总视频 11.6 秒。五机均为 1080p。动作包含床上坐撑、后躺、平躺和翻侧抬身，
没有离床、循环或慢放。仅实验专用相机放宽 FOV，并使用 UE 实际导出的 K；所有
116 组、580 张 RGB 验证通过，五视角 actual native55 全在画面内。机器人遮挡保留。
选片、标定、时钟和可见性证据见 `capture_manifest.json`。

原版与候选共享同一录制和观测质量门。baseline 从静态 RGB 重新估计初始化；
silhouette 候选仅替换同一人的 beta，姿态和位移取本次 baseline 初始化。
该 beta 来自以前的五视角 RGB 单帧轮廓拟合，并非本轮新训练或新拟合，更非 GT。
来源见 `onbed10s_silhouette_beta_init_provenance.json`。

新长片静态初始化的三个胸廓前后厚度为 254.0 / 241.1 / 198.3 mm，仍比身份参考
厚约 34.5 / 21.1 / 7.8 mm；轮廓 beta 的误差约 11.0 / 7.2 / 2.4 mm。宽度并非
整体偏大，详见独立的 `onbed10s_shape_audit.json`。增加重复静态帧没有补上缺失的
胸腔表面约束。

| 长片方法（均 101/101 输出） | body22 mm | hands30 mm | 旋转步长 P95 | 局部旋转对 GT 均误差 |
|---|---:|---:|---:|---:|
| 每帧解析 IK，原版 | 46.57 | 21.60 | 48.50° | 40.96° |
| previous，时序 0.01，原 beta | 42.52 | 20.46 | 16.08° | 47.12° |
| previous，时序 0.01，轮廓 beta | 40.81 | 21.00 | 16.58° | 48.49° |
| previous，时序 0.09，轮廓 beta | 36.84 | 17.44 | 10.81° | 46.01° |

候选时序 0.09 是本片上的探索参数，尚无独立验证集；10 Hz 下时序损失按 dt 归一。
本片 GT 的旋转步长 P95 为 8.66°。原版最大单步跳变 169.00°，候选仍有 58.47°。
候选降低关节误差和抖动，但局部旋转绝对误差比原版高，因此**仍不能认定表面扭曲
已解决**。长片末尾右膝有多帧无有效三角化观测；关节稀疏、遮挡以及缺少可靠扭转
约束会产生稳定的错误姿态。101/101 只表示通过当前质量门，不表示姿态正确。

`onbed10s_candidate_fresh` 已从 RGB 重新执行完整前端和后端，101 帧的所有网格
顶点与缓存候选完全一致，见 `fresh_replay_parity.json`。本轮实际测得候选计算耗时
均值 396 ms、P95 538 ms（不含初始化、文件写入；同时有 CPU 渲染任务），不能
据此声称稳定 2 Hz，更不是 10 Hz 实时重建。缓存分项耗时不能替代此次真实复测。

完整对照视频位于 `onbed10s_baseline_overlays/videos` 和
`onbed10s_candidate_overlays/videos`。这两组均无拒绝帧，输出 101 帧、10 fps，
保留完整 10.1 秒；当前 renderer 在其他含拒绝帧的序列上会跳帧，限制已记录于
`overlay_timing_audit.json`，不能将那种视频误称完整时间线。

两个版本的六路视频均完成逐帧解码验证。候选在末尾 113/114 帧无明显躯干整体
翻转，但腕掌、踝足和颈头仍有方向错误；右腕在 112 帧的单步旋转约 28.9°，参考
约 5.8°。详细检查见候选目录的 `visual_review.json`。原版没有身份切换，却仍在
98/103/105 等帧出现明显表面偏离，说明关联成功也不等于姿态正确。

后续方法应分两层改进：初始化用多个清晰姿态共享 beta，以多视角轮廓/边界约束
胸廓厚度并保留骨长项；跟踪固定 beta，联合真实图像/手掌脚面方向、姿态先验和
时间约束确定扭转。当前实验已证明单靠低 MPJPE 或加强平滑不够。旧 Small/Large
beta prior 在本人的截面测量更差，不能直接替代轮廓初始化。低质量肢体应单独降权
或标记不确定，不能因为全帧通过门槛就视作全身可靠。

足部映射审阅：native10/11 和 toe leaf55/56 共用 Body25 BigToe 19/22，并不将
两点压到同一点，因为各有不同的初始化 offset；不能直接判为左右或拓扑映射错误。
但同一锚点无法给两者提供独立方向信息，且固定的世界坐标 offset 不随肢体旋转。
previous 模式跳过该解析 IK seed，refinement 仍拟合全部 67 个观测点（包括足跟、
大小脚趾和手部点）。这些点有助于约束末端方向，但不能保证遮挡时可靠，也不能
唯一分配链上各关节的扭转。本轮不擅自改 foot→heel 映射。

本轮代码验证：21 项相关测试通过；覆盖旋转表示不变性、时间间隔归一、失败回滚、
源观测质量门保留和录制序列接口。新增行为通过可选参数启用，未将候选改成默认。

### 精度优先的局部扭曲复查

用户要求先保证准确，再考虑速度。固定长片输入、轮廓 beta 与时序权重 0.09，
仅将优化从 60 步增加到 300 步：body22 MPJPE 从 36.84 降至 35.49 mm，hands30
从 17.44 降至 17.39 mm，局部旋转对参考的平均误差却从 46.01 增至 55.80°；
后端平均计算时间从约 0.141 增至 0.719 秒。这是额外算力不足以解决可辨识性问题
的反例，不应把它作为新最优。结果为 `onbed10s_previous_silhouette_w9e2_steps300`。
局部旋转误差也不能直接等同于可见末端方向误差，因为父子局部旋转可能互相抵消；
需要同时检查掌面、脚面、头部的全局方向及皮肤形变。

实现复查确认：当前 67 点 loss 已经包含鼻、双眼、双耳、足跟、大小脚趾和双手
21 点；head/neck 可以优化。关闭 face iteration 只固定 jaw/eyes/expression。
重复加入这些位置点不会补充新信息。CPU 抽查 frame15/65/114 时，缓存 LBS 与
官方 forward_simple 顶点完全一致；不能归因于加速蒙皮计算错误。

### 完整官方 HybrIK-X RGB 实测（2026-10-03）

已从官方 README 的 Google Drive 下载 HybrIK-X HRNet-W48 + RLE 完整权重，
375,186,198 bytes，SHA256
`baec5f974de03c57201a0a6b72f862147c13e1d019dfb472ba810e401224fec6`。
缓存位于 `/home/camp/.cache/realus/hybrikx/hybrikx_rle_hrnet.pth`，仓库通过 symlink
使用。复用本机官方 SMPL-X NPZ/kid template，严格加载全部 1987 个 keys，
未用普通 SMPL 的 checkpoint 代替，未跳过不匹配参数。官方仓库：
<https://github.com/jeffffffli/HybrIK>；方法说明：<https://jeffli.site/HybrIK-X/>。

`perception/apps/cache_hybrik_rgb.py` 已运行完整 RGB forward，官方 256×256 crop、
float32、flip test 开启。床上长片 101 帧 × 5 机全部成功；431 个 crop 来自 detector，
74 个来自前端保存的 temporal_previous bbox，没有 GT crop 或 3D/GT 输入。
逐视角缓存保留 phi、55 个 local rotations、camera-relative xyz、sigma、beta 和计时，
见 `onbed10s_hybrik_rgb_cache/index.json`、`verification.json`。

RTX 4080 Laptop GPU 顺序处理每个视角，完整 RGB forward 平均 100.15 ms、P95
110.72 ms；每组五视角约 0.501 秒。该数值不含 cold init、图像解码、crop、磁盘写入，
不能称为整个系统延迟。新增 RGB 的混合跟踪 60 步分项耗时之和约 0.903 秒/组，
前端/RGB 计时来自缓存，非实时端到端复测。精度比较先保留 flip test，没有先降精度求快。

官方 RLE 网络 `pred_phi` 是 54 个非根关节的 cos/sin，可作为与多视角 DWPose
骨架互补的 twist 候选。提取 phi 仍需执行共享 HRNet backbone，不能把输出头小
误认为完整图像推理免费；flip test 还会增加一次 backbone 前向。其 `pred_sigma`
是关节 UVD 置信度，不是 twist 置信度；其 11 维 beta 含 kid 参数，不能直接覆盖
本工程固定的成人 10 维 beta。多视角 phi 融合需验证圆周角一致性和关节约定。

一个关键实现细节：官方 analytical IK 的 multi-child 分支对 native9（spine3）、
15（head）、20/21（wrists）使用子关节点 SVD 求方向，不直接使用这些关节的 phi。
因此“加 RGB twist 就能修腕掌/头颈”并不成立。本轮另外实现了 per-frame 完整 local
rotation prior：按共同 SMPL-X rest frame 做多视角 SO(3) medoid/多数一致性筛选，
阈值 45°，至少两视角；root、jaw/eyes 不参与。先验与原始 67 点和 SO(3) 时序项联合
优化；beta 冻结。跨视角一致性仅是筛选分数，不是置信概率，多个相机仍可能共同犯错。

同一 101 帧、同一轮廓 beta、同一时序权重 0.09 的受控结果：

| 初始化/附加信息 | 优化步数 | body22 mm | hands30 mm |
|---|---:|---:|---:|
| analytical IK，静态 phi | 60 | 41.17 | 20.57 |
| analytical IK，逐帧 RGB phi | 60 | 40.97 | 20.26 |
| previous，无 RGB pose prior | 60 | 36.84 | 17.44 |
| previous，RGB pose prior 0.001 | 60 | 36.12 | 16.97 |
| previous，RGB pose prior 0.01 | 60 | 31.50 | 16.21 |
| previous，RGB pose prior 0.01 | 300 | 30.82 | 15.85 |

phi-only 对右脚面平均三轴方向误差反而从 24.06° 增至 27.08°，不能据关节均值微降
就宣布成功。完整 pose prior 0.01、60 步则把左右踝的全局 native joint 旋转误差从
46.89°/27.48° 降至 12.57°/20.46°，左右脚从 50.64°/44.47° 降至 13.87°/20.75°，
左右腕从 16.56°/20.98° 降至 13.60°/17.59°，颈部从 17.13° 降至 13.07°。
头部 8.18°→7.99°，基本持平。该指标是关节链累计方向误差，不是皮肤表面 PVE。
Body22 逐帧旋转步长 P95 为 10.81°→10.19°，仍不能把平滑程度当成正确性。

从网格回归的掌面/脚面/头部三个轴另作检查：previous60 的左掌/右掌/左脚/右脚/头
为 10.88°/15.34°/18.61°/22.81°/6.46°；RGB pose prior 0.01、300 步为
9.01°/12.62°/10.16°/17.65°/6.77°。参考由固定相同 beta + canonical pose 生成，
不是 UE 原始皮肤表面真值；头部方向略有回退，右脚残差仍明显。GT 仅用于评估。
权重在本片上探索了 0.001/0.01，300 步也在本片验证；尚无独立动作片泛化结论。

beta 另作隔离实验：只取首个运动帧 15 的五视角 RGB beta 前10维中位数并冻结，
不使用全片或 GT 选择。胸部三个截面的前后厚度为 224.82/221.65/190.12 mm，
参考为 219.45/220.03/190.44 mm；但 rest height 高估 67.59 mm。相同 RGB pose
prior 0.01、60 步下，body22 从轮廓 beta 的 31.50 回退到 34.44 mm，hands30 从
16.21 回退到 16.75 mm。说明 RGB beta 改善胸厚并不等于整体体型更准，不直接替换。
详见 `rgb_beta_firstframe_audit.json` 与 `onbed10s_rgbpose_w1e2_s60_rgbbeta`。
另将该 RGB beta 作为 EasyMocap 静态骨长拟合的均值先验（reg_shapes=0、
init_shape=0.0005，沿用旧对照权重，未扫描），长骨最大中位残差从 74.26 降至
38.12 mm，但 rest height 从 1748.30 降到 1596.97 mm，低于参考约 83.74 mm；
胸厚为 226.86/217.59/182.52 mm。该结果没有解决整体比例，未替换轮廓 beta，
见 `rgb_beta_refit_audit.json`。弱骨架观测会把 shape 拉偏；后续初始化需要真实
多视角轮廓/体表证据与共享 beta 约束，不能只换 prior 或重复相同静态帧。

本轮用于视频复查的 300 步候选位于 `onbed10s_rgbpose_w1e2_s300_w9e2`。
其全局关节旋转误差（左腕/右腕/左踝/右踝/左脚/右脚/颈/头）为
12.54°/17.14°/12.52°/19.65°/14.02°/19.36°/12.85°/8.88°。
body22 局部旋转平均误差由 previous60 的 46.01° 降至 18.19°；旋转步长 P95
由 10.81° 变为 11.47°，因此收益主要在姿态正确性，不能宣称更平滑。
独立分项计算时间之和约 1.56 秒/组，后端约 0.817 秒；尚未优化实时速度。
`rgb_ablation_summary.json` 和 `rgb_ablation_comparison.png` 汇总受控结果。
尾部错误仍需保留：新候选右腕全局旋转最大误差 83.87°（frame29；旧候选最大
122.58°），右踝最大 50.33°（frame112；旧候选49.23°）。右腕 P95 从77.11°降至
48.69°，右踝 P95 却从42.50°变为44.04°。因此不能声称所有扭曲已消除，尤其是
末段缺右膝/足点与早段腕掌方向不可靠的帧。视频复查需包括这些失败帧。

两者的角色可拆分：DWPose 提供图像点和经标定三角化的世界坐标位置，RGB 网络
提供姿态/扭转先验，最终联合优化。本轮 A/B 支持该组合在当前床上片段的收益，
没有独立证明“DWPose 骨架一定更准”或“HybrIK-X 能替代 DWPose”；不同论文的
2D AP 与 3D MPJPE 不能直接比较。新方法目前为可选 benchmark 路径，未改生产默认。

新增 RGB loader/cache、圆周 phi 融合、SO(3) local pose 融合及后端参数检查。相关
42 项测试通过，包含视角分歧/非法矩阵、root/face 排除、缓存帧号、失败状态回滚与
原始观测门保留。复现实验：

```bash
A=outputs/realus_sync/tests/surface_motion_audit_20261003
python perception/apps/cache_hybrik_rgb.py \
  --recording-dir "$A/onbed10s_recording" \
  --observation-sequence-dir "$A/onbed10s_baseline" \
  --output-dir "$A/rgb_cache_new"
python perception/apps/run_hybrik_benchmark.py \
  --sequence-dir "$A/onbed10s_baseline" --recording-dir "$A/onbed10s_recording" \
  --init-npz "$A/onbed10s_silhouette_beta_init.npz" --gender neutral \
  --pose-initialization previous --refine-steps 300 --temporal-weight 0.09 \
  --rgb-cache-dir "$A/rgb_cache_new" --rgb-hint local-pose --rgb-pose-weight 0.01 \
  --output-dir "$A/rgb_pose_new"
```

本轮按用户要求删除 3,955 个废弃渲染输出和预览缓存，释放约 3.09 GiB，保留原始
RGB、模型、拟合 NPZ、评估摘要和主结果视频。逐文件清单为
`outputs/storage_cleanup/obsolete_outputs_20261003_deleted.json`。另外清理本轮淘汰的
出画长片诊断 JPEG，释放约 293 MiB，见 `capture_cleanup_manifest.json`。

完整 UE、Genesis、控制器和九窗口启动命令见 [COMMAND.md](COMMAND.md#phase-2--genesis--ue-当前链路)。所有测试使用当前床、轨道和 RM75 八自由度模型；仿真模式不启动真机控制器。

当前主 benchmark 是 outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2：五机、1920×1080、30 fps，前 15 组为静态初始化，随后 75 组为连续运动。动作源为 CMU 114/114_11_poses.npz，角色 7 使用 source frames 1000:4:1296；静态帧保持背部躺卧、双臂张开，运动段保留床轨和机械臂造成的真实遮挡。

calibration.json 提供五机 K 和 OpenCV 外参；joints_gt.jsonl 是 UE 实际应用的 55 个骨骼位置。GT 不参与重建，只在推理后评估。统计绝对和骨盆相对 body22 MPJPE、hands30 MPJPE、骨盆误差、拒绝帧比例。没有 UE 皮肤顶点 GT，因此不报告 PVE；in-frame 标签仅表示画面边界，不代表无遮挡。

每个终端先运行：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export BENCH=outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2
export REALUS_SCENE_SPEC="$REPO/configs/scenes/realus_bed_rail_5cam_scene.yaml"
export BENCH_SHAPE=/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions/motions_npz_training/it_4375_M_2400.npz
export BENCH_MODEL=/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam_clothing/support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz
```

重建新的 benchmark 时，先启动窗口 1、2、4、5、play。暂停窗口 6、7，避免实时动作覆盖严格录制请求。当前 clip 已完成录制；以下 role 7 参数定义其动作源窗口：

```bash
export REALUS_GT_LOG="$SESSION_DIR/lying_motion_gt.jsonl"
bash scripts/realus_fullflow.sh gt-log \
  --frame-start 1000 --frame-step 4 --frame-limit 75 \
  --human-shape-npz "$BENCH_SHAPE" --human-body-model "$BENCH_MODEL" \
  --no-track-subscribe --no-anatomy-asset-subscribe
bash scripts/realus_fullflow.sh benchmark-log \
  --canonical-jsonl "$REALUS_GT_LOG" \
  --output-jsonl "$SESSION_DIR/lying_benchmark_gt.jsonl" \
  --warmup-s 0.5 --fps 30
export BENCH="$SESSION_DIR/lying_benchmark_recording"
bash scripts/realus_fullflow.sh record \
  --canonical-jsonl "$SESSION_DIR/lying_benchmark_gt.jsonl" \
  --output-root "$BENCH" --session-id realus_lying_benchmark \
  --ack-connect tcp://127.0.0.1:17367 --keep-duplicate-motion-frames
bash scripts/realus_fullflow.sh video \
  --recording-dir "$BENCH" --output-dir "$BENCH/videos" --fps 30
bash scripts/realus_fullflow.sh annotations --recording-dir "$BENCH"
```

严格 recorder 每组先应用动作，再捕获五机 RGB 和实际骨骼，最后 ACK。录制结束恢复连续图传：

```bash
bash scripts/realus_fullflow.sh resume-preview
```

安装固定版本的官方参考仓库（保留已有目录，不擅自 checkout）：

```bash
bash scripts/install_mocap_benchmark_repos.sh
```

质量优先路径使用 neutral SMPL-X 的静态 RGB 初始化，随后冻结 beta，并运行
HybrIK-X analytical backend 60 步 pose refinement。五机序列使用
configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml，三路三角化 worker；
不要加入 VPoser 或 SAM2。最终 75 个运动组应逐组评估，不能用稀疏采样的吞吐
代替整段通过率。

```bash
bash scripts/realus_fullflow.sh sequence \
  --config configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml \
  --recording-dir "$BENCH" --run-name lying_hybrik_cuda \
  --output-root outputs/realus_sync/tests/pose \
  --gender neutral --fit-model smplx --tracking-backend hybrik \
  --hybrik-device cuda --hybrik-refine-steps 60 \
  --triangulation-workers 3 --triangulation-pair-hypotheses 1 \
  --dwpose-temporal-bbox-fallback \
  --warmup-moments 15 --sample-stride 1 --max-moments 75 \
  --disable-face-iteration --no-write-debug-images
export INIT=outputs/realus_sync/tests/pose/lying_hybrik_cuda/warmup_init/smplx_result.npz
```

骨架后端使用同一 RGB 初始化、DWPose 和多视角三角化。hybrik 接入官方
HybrIK-X 的解析 IK 层，初始化体型和 twist；没有运行其预训练 RGB 网络。
脸、眼、表情固定，beta 冻结。

三视角核心关节、重投影误差和射线角度门槛保持不变；本次测速显式设置
--triangulation-pair-hypotheses 1，并在完整 75 帧上检查通过率、误差和网格叠图；
缓存本身的精确一致性不代表减少候选假设一定不会损失精度。DLT 重复解缓存
在实际 67 点输入上与旧版逐项比对，XYZ、置信度、mask 和完整诊断结果完全一致。
CPU 端旧实现约 470 ms，缓存后约 211 ms；保存的输出保持 exact parity。
SMPL-X backend 的 fast-LBS parity 也已单独核对。GPU/优化器先在静态 RGB
初始化模型上热启动，再测运动帧，初始化耗时单列。

可选的 RGB 历史框关联在检测丢失时复用上一帧框；检测框排序变化时按 IoU
关联上一帧选中的人，避免把机械臂附近的误检测框当成患者。它只使用 RGB
检测结果，保留原有 2D/3D 质量门，默认关闭，本次命令显式开启
`--dwpose-temporal-bbox-fallback`；显式关闭使用
`--no-dwpose-temporal-bbox-fallback`。

在线 HybrIK 使用已成功估计的静态初始化 NPZ，发布到 5598；窗口 7 选择
`--human-source fitted` 后，Genesis 和 UE 都由重建结果驱动：

```bash
bash scripts/realus_fullflow.sh sequence \
  --config configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml \
  --duration-s 0 --interval-s 0.5 \
  --tracking-backend hybrik --gender neutral --fit-model smplx --initialization-npz "$INIT" \
  --hybrik-device cuda --hybrik-refine-steps 60 \
  --triangulation-workers 3 --triangulation-pair-hypotheses 1 \
  --dwpose-temporal-bbox-fallback \
  --no-write-debug-images --publish-genesis
```

在线模式缺少初始化文件时明确报错。录制 GT benchmark 时，保持 AMASS 为 UE
动作源，使用录好的 RGB 离线评估，避免估计动作覆盖 GT 后再评估自己。

MV4DHMR Small-672 和 CHI3D fine-tuned Large-672 的官方五机 pipeline：

```bash
bash scripts/realus_fullflow.sh mv4dhmr \
  --recording-dir "$BENCH" --size small --device cuda:0 \
  --download-checkpoint --warmup-frames 15 --motion-stride 15 \
  --include-vertices --output-jsonl outputs/realus_sync/benchmarks/small.jsonl
bash scripts/realus_fullflow.sh mv4dhmr \
  --recording-dir "$BENCH" --size large --device cuda:0 \
  --download-checkpoint --warmup-frames 15 --motion-stride 15 \
  --include-vertices --output-jsonl outputs/realus_sync/benchmarks/large.jsonl
```

MV4DHMR 原版逐帧预测 beta；15 帧仅热启动网络和 tracker，没有冒充固定体型实验。`--size large896` 单独指 Naver zero-shot 896 模型，不能和 Large-672 混用。公开 V100 结果不是本机测速。参考：[MV4DHMR 官方代码](https://github.com/Att100/MV4DHMR)、[HybrIK 官方代码](https://github.com/jeffffffli/HybrIK)。

重建结果必须投影成真实 SMPL-X 三角网格并覆盖到五机 RGB 上，再比较关节误差：

```bash
bash scripts/realus_fullflow.sh evaluate \
  --recording-dir "$BENCH" \
  --sequence-dir outputs/realus_sync/tests/pose/lying_hybrik_cuda \
  --method 'DWPose + initialized HybrIK-X CUDA' \
  --output outputs/realus_sync/benchmarks/hybrik_evaluation.json
bash scripts/realus_fullflow.sh evaluate \
  --recording-dir "$BENCH" --predictions outputs/realus_sync/benchmarks/large.jsonl \
  --method MV4DHMR-Large-672-CUDA \
  --output outputs/realus_sync/benchmarks/large_evaluation.json
bash scripts/realus_fullflow.sh overlays \
  --recording-dir "$BENCH" \
  --sequence-dir outputs/realus_sync/tests/pose/lying_hybrik_cuda \
  --frame-indices 15 45 --draw-gt-bones \
  --output-dir outputs/realus_sync/benchmarks/hybrik_overlays
```

每种 GPU 测速顺序运行，包含五机前处理、网络、三角化/融合及 SMPL-X 重建，GPU 同步后记录墙钟。冷初始化、文件写入和叠图耗时单列，不计入算法延迟；完整程序运行时间另存。`successful_output_hz` 指接受帧数除以全部尝试的计算耗时，是离线计算吞吐；不是源视频时间上的输出频率。2 Hz 输入采样只有 70% 通过时，新结果频率只有约 1.4 Hz，不能把计算吞吐超过 2 Hz 写成稳定追踪成功。

DWPose 网络、EasyMocap/HybrIK 的 SMPL-X 优化和 MV4DHMR 网络使用 CUDA；
图像处理和几何三角化仍有 CPU 部分，其墙钟耗时包含在上述端到端算法延迟中。
评估还单列 `successful_frame_latency`，避免很快的拒绝帧掩盖成功重建的实际耗时。
0.5 秒是静态输入时长，当前 15 帧 RGB 初始化的前端与拟合计算合计约 17.6 秒，
不是 0.5 秒内完成初始化。

SkelFormer 目前在[作者项目页](https://vdavoodnia.github.io/projects/skelformer/)未找到可下载的官方代码或 checkpoint，未用别的模型冒充它测速。

## 当前五机质量与速率状态（2026-10-03）

正式主结果使用
outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2，
15 组静态初始化加 75 组运动，配置为 neutral、HybrIK-X refine60 和
triangulation-workers 3。本机重新初始化 的 75/75 帧全部通过质量门：
body22 MPJPE 44.1030 mm，hands30 MPJPE 11.3938 mm，骨盆误差 64.9355 mm；
不做 Procrustes，也没有 UE 皮肤顶点 GT，不能报告 surface PVE。评估 JSON 为
outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_hybrik60_reg0005_evaluation.json。

平均算法耗时 461.953 ms，P95 536.806 ms，计算吞吐 2.1647 Hz；这是离线计算
吞吐，不是源视频时间上的 live FPS，也不能把 P95 超过 500 ms 写成稳定 2 Hz。
动态阶段的测量拆分约为 detector 42.152 ms、triangulation 239.426 ms、
HybrIK backend 178.128 ms。15 组静态输入的前端与拟合总初始化约 20.4 s，
不含 detector engine 预加载；初始化耗时不计入运动帧速率。

上一版 75 帧旧 beta 对照为 body22 48.5573 mm、hands30 13.2946 mm。显式
learned beta prior 在独立 60 帧检查中没有优于 zero-reg .0005；因此生产路径
继续使用 zero-reg .0005，不推荐把 Small/Large 的 RGB learned mu 当作固定体型
先验。该比较只涉及关节和速度，不能推出表面准确率。

渲染身份是 BEDLAM2 locked-head neutral16，识别端拟合的是 neutral10；两者
不能按名称中的 M 或 beta 维数互相冒充。root/pelvis 使用实测对齐修正；若
MidHip 缺失，只在左右两个实际髋点都通过可信度和跨度检查时恢复 hip-pair
根，否则拒绝该帧，不用合成的单侧根点。

最终 75 帧已全部生成真实三角网格叠图，共 375 张五机 JPEG 与 75 张拼图；
橙色为估计网格，绿色为实际 UE 骨骼。清单位于
outputs/realus_sync/tests/diagnostics_cameras/overlays/lying_open_arms5_hybrik60_reg0005/overlay_summary.json。
已检查首段、中段和最后一帧：动作整体跟随，躯干、腿部轮廓仍有局部差异，
不能用关节误差代替体型表面误差。该短片主要验证躺卧的手臂动作，未覆盖快速全身运动。
当前相关测试共 84 项通过；五路叠图与拼接 MP4 都已逐帧解码检查，共 75 帧、30 fps。

五路叠图视频及 montage.mp4 位于上述 overlay 目录的 videos/，75 帧、30 fps、
2.5 秒；video_manifest.json 对齐原始 source frames 15–89。原始五路视频包含
前 15 帧静态初始化，共 90 帧、3 秒，位于 benchmark 目录的 videos/。
重新导出全段叠图和视频：

```bash
bash scripts/realus_fullflow.sh overlays \
  --recording-dir outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2 \
  --sequence-dir outputs/realus_sync/tests/diagnostics_cameras/pose/lying_open_arms5_hybrik60_reg0005_cuda_v1 \
  --output-dir outputs/realus_sync/tests/diagnostics_cameras/overlays/lying_open_arms5_hybrik60_reg0005 \
  --draw-gt-bones --mesh-alpha 0.45 --export-videos --fps 30
```

### 历史四机对照（2026-10-02）

以下结果保留为历史四机参考，不是当前主 clip。

GPU 为 RTX 4080 Laptop 12 GB，Torch 2.12.0+cu126。下面比较相同的四个运动帧
15、30、45、60；两种 MV4DHMR、EasyMocap 和 HybrIK 均使用实际 CUDA。
误差是与 UE 实际 native55 骨骼比较的绝对世界坐标 MPJPE，不做 Procrustes 对齐。

| 方法 | 四机算法均值 | P95 | body22 MPJPE | hands30 MPJPE |
|---|---:|---:|---:|---:|
| DWPose + EasyMocap，固定 beta | 8497 ms | 9345 ms | 47.7 mm | 9.1 mm |
| MV4DHMR Small-672 | 203 ms | 273 ms | 259.6 mm | 217.7 mm |
| MV4DHMR Large-672，CHI3D fine-tuned | 363 ms | 429 ms | 188.5 mm | 141.3 mm |
| DWPose + 初始化 HybrIK-X，refine30 | 422 ms | 457 ms | 56.6 mm | 23.6 mm |
| DWPose + 初始化 HybrIK-X，refine40 | 464 ms | 499 ms | 57.5 mm | 19.9 mm |
| refine30 + 可选 RGB 框关联 | 409 ms | 440 ms | 56.6 mm | 23.6 mm |

完整十个运动样本中，Small 为 6/10、Large 为 7/10，原 HybrIK 为 6/10。
可选 RGB 框关联恢复第 120 帧，达到 7/10；成功帧平均 402 ms、P95 475 ms，
全七帧 body22 为 58.0 mm、hands30 为 24.6 mm。第 105、135、150 帧仍被
原三维核心关节/肢体一致性质量门拒绝。EasyMocap 安静环境复测只覆盖共同
四帧，不能把它写成全段 100% 通过。
refine40 在全部六个成功帧上的 P95 为 528 ms，超过 500 ms 预算；不能只用
共同四帧的 499 ms 得出全段达到 2 Hz 的结论。

MV4DHMR 当前躺卧重建在腿、手臂和根位置上明显偏离，速度通过但精度不足。
初始化 HybrIK-X 约快 20 倍，但手指精度低于 EasyMocap，仍需提高完整动作覆盖率。
此短片和十个样本不足以宣称持续稳定的 2 Hz；当前结论是骨架后端值得继续优化。

完整数值和各次原始结果路径见
[`benchmark_results.json`](../outputs/realus_sync/tests/benchmark_results.json)。
真实 SMPL-X 网格与 RGB、绿色 UE GT 骨骼的对比见
[四机 frame15 叠图](../outputs/realus_sync/tests/overlays/common/comparison_frame15_allcams_baseline_mv4small_mv4large_hybrik30_v2.jpg)、
[四机 frame45 叠图](../outputs/realus_sync/tests/overlays/common/comparison_frame45_allcams_baseline_mv4small_mv4large_hybrik30_v2.jpg)、
[恢复 frame120 的叠图](../outputs/realus_sync/tests/overlays/common/hybrik_temporal_bbox_refine30_quiet_frame120_v2/montages/frame_00000120.jpg)。

四个视频在 [`full_body_benchmark/videos`](../outputs/realus_sync/tests/full_body_benchmark/videos)，
标定与 GT 同目录保存。实际 RGB → CUDA Genesis → UE → 四机闭环和 live HybrIK
发布验证分别记录在 `runtime/actual_rgb_fitted_chain_v2/validation.json` 与
`pose/live_hybrik_smoke_isolated_refine30_quiet_v2/live_validation.json`。后者四组均
保留 native165、固定 beta、10475 顶点、20908 三角面及输入时间戳。

启动代码、当前场景和机器人配置已接入 RealUS；大型 BEDLAM UE 项目、AMASS、
SMPL-X 模型和 Genesis 库仍复用已有 Among_US 资产位置。尚未连接实体机器人验收。
可选解剖资产的原质量门未通过，未发布，不影响上述四机动作 benchmark。
文档所引用的命令已按当前五机配置更新；正式 75 帧结果以 evaluation JSON 和实际生成的 overlays 为准。
本次专用 UE、mux、scene publisher 和 GPU 测试进程已退出。

最初测试时盘仅剩约 0.5 GB；此前清理历史日志和缓存后的快照为 root 约
11 GB、EXT_DRIVE 约 4 GB，见 `outputs/storage_cleanup/summary.json`。
这轮再清理 51 份完成测试的日志（2,401,320 bytes），以及 14 份 UE 日志和
转码临时缓存（39,976,785 bytes）。清单见 `outputs/storage_cleanup` 中
`test_logs_20261003.json` 与 `additional_test_logs_20261003.json`。
完成新增动态复核后，root 约 7.2 GB、EXT_DRIVE 约 3.6 GB 可用。
三份旧版冗余大 JSON 已无损压缩为 `.json.gz`，
解压内容的 SHA256 与原文件一致，记录见 `tests/diagnostics_compression.json`。
原始四机数据、模型、GT 和最终结果保留；更长录像需要更多输出空间。

## 体型一致性与四/五机复核（2026-10-03）

这轮先修正渲染协议，再实际启动 UE 录下七个稀疏姿态：0 为前端预热，
15、45、105、120、135、150 为测量。它是布局诊断，不是新的连续录像，
也不能据此宣称整段稳定 2 Hz。旧 benchmark 的误差仍是对当时实际 UE 骨骼的
有效测量，但旧 UE 动作不完全等于原始 canonical 动作；不能混合新旧 GT。

已修正 UE 左右 collar 的父节点：应为 spine3（9），原来误设为 neck（12）。
旧第 120 帧因此使双臂/双手多转约 35.9°。新 UE 捕获的七帧，用原模型、原
体型、canonical 姿态做官方 CUDA forward，在只对齐骨盆平移后，55 关节
最大差 0.000283 mm、世界四元数最大差 0.000014°。
数据见 `outputs/realus_sync/tests/diagnostics_shape/corrected_ue_capture_parity.json`。
这验证骨架和渲染身份；没有 UE 网格顶点 GT，仍不宣称表面 PVE 为零。

当前 UE 的 `it_4375_M_2400` 身份实际是 BEDLAM2 locked-head neutral、16 维 β，
不是由名称中的 M 决定的 standard male。现有 RGB/EasyMocap/HybrIK 识别端拟合
neutral10；UE 固定 FBX 已烘焙另一套体型，canonical β 不会自动变形这个固定资产。
BEDLAM 并非只给任意 SMPL-X 加贴图。
模型版本、gender、完整 β、骨架和单位需要一致。
官方模型差异见 [BEDLAM2 FAQ](https://bedlam2.is.tuebingen.mpg.de/faq.html)。

保持当前 AMASS 动作选择、帧率和源平移，只让 Genesis 使用当前 UE 资产的体型：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export AVATAR_SHAPE=/media/camp/EXT_DRIVE/Among_US/dataset/raw/humans/bedlam2/motions/motions_npz_training/it_4375_M_2400.npz
export AVATAR_MODEL=/media/camp/EXT_DRIVE/Among_US/ref_code_library/bedlam_clothing/support_data/body_models/smplx_locked_head/SMPLX_NEUTRAL.npz
bash scripts/realus_fullflow.sh 7 \
  --human-shape-npz "$AVATAR_SHAPE" --human-body-model "$AVATAR_MODEL"
```

仍按原来的 1、2、4、5、play、6 启动 UE 链路，角色 7 默认仿真状态；
已运行真机 C++ 控制器时再给角色 7 增加 `--robot-state-source shm`。
`--human-shape-npz` 与 `--human-body-model` 必须成对提供；该覆盖只用于
AMASS 渲染，不用于 `--human-source fitted`，也不把已知 β 送进 RGB 重建。
实际 CUDA Genesis 八自由度场景已成功播放 1000、1004、1008 三帧，
canonical 保留全部 16 β、neutral 和 native165；日志为
`outputs/realus_sync/tests/diagnostics_shape/unified_genesis_gt.jsonl`。

此前的 β 初始化仅通过 EasyMocap optimizeShape 拟合 13 段 Body25 骨长，
不使用额外表面约束。15 帧静态 RGB 只平均测量，不增加独立体型约束。
当前 male10 骨长 Jacobian 条件数约 1614；正确 neutral16 仍有不可观方向。
直接把 shape 正则去掉，会得到极端 β，不能作为修正方法。
识别端 neutral10 与 BEDLAM2 neutral16 的渲染模型没有完全统一；显式 learned
beta prior 在 60 个运动帧上没有优于 zero-reg .0005，形状初始化也不能消除姿态误差。
若需要 UE 显示新估计出的任意体型，还需导入一次该体型的 subject mesh；
仅发布新 β 不会改变当前固定 BEDLAM 身体。统一缩放不能修正不同骨段的差异。

相机候选配置位于 `outputs/realus_sync/tests/diagnostics_cameras`：
`scene_candidate_4cam.yaml` 调整 cam1/cam3；`scene_candidate_5cam_foot.yaml`
再加床尾 cam5，位置 (1.6, 0, 1.4) m，look-at (-0.1, -0.1, 0.4) m。
配套 tracking/calibration YAML 一并生成。五机已实际 UE 录制，并从 capture
metadata 导出实际 K/外参；这些虚拟相机参数不能直接当成实体相机标定。
生产 scene/calibration 保持原配置，候选需通过误差和耗时检查后再采用。

```bash
# 在每个相关终端设置，让 UE 场景与 Genesis 使用同一候选。
export REALUS_SCENE_SPEC=outputs/realus_sync/tests/diagnostics_cameras/scene_candidate_5cam_foot.yaml
bash scripts/realus_fullflow.sh 3 --camera-names cam1 cam2 cam3 cam4 cam5
# 第 8 窗口切换匹配的五机重建配置。
bash scripts/realus_fullflow.sh 8 \
  --config outputs/realus_sync/tests/diagnostics_cameras/tracking_candidate_5cam_foot.yaml
```

布局重建测试固定复用此前从 RGB 得到的 neutral10 初始化，隔离相机影响；
不把已知渲染 β 用作识别输入。接电后 GPU 上限恢复 150 W，UE/Genesis
退出后依次测 CUDA，网络热启动与首轮五机 TensorRT 编译另计。
电池 10 W 下的运行仅保留为诊断，不能和接电测速比较。

新的通过率、成功帧耗时、共同成功帧误差、前端/后端耗时拆分和 MV4DHMR
复测见 `outputs/realus_sync/tests/diagnostics_cameras` 下的 evaluation/comparison
文件。各布局各自成功帧的均值不能直接互比。MV4DHMR 的 RGB 网络不使用
DWPose，因此它的误差不能归因于 DWPose 的三角化错误。
五机的真实 SMPL-X 网格叠图位于同目录 `overlays`。

## 当前生产复现命令

当前质量优先复现使用已选五机 clip 和 正式五相机配置。命令
从前 15 组静态 RGB fresh 初始化 neutral10 beta，再逐组处理 75 个运动组；
不使用 VPoser、SAM2 或 learned beta prior。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source env.sh
export BENCH=outputs/realus_sync/tests/diagnostics_cameras/lying_open_arms5_neutral16_v2
bash scripts/realus_fullflow.sh sequence \
  --config configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml \
  --recording-dir "$BENCH" \
  --output-root outputs/realus_sync/tests/pose \
  --run-name lying_open_arms5_hybrik60_reg0005_cuda_v1 \
  --tracking-backend hybrik --gender neutral --fit-model smplx \
  --hybrik-device cuda --hybrik-refine-steps 60 \
  --triangulation-workers 3 --triangulation-pair-hypotheses 1 \
  --dwpose-temporal-bbox-fallback \
  --warmup-moments 15 --sample-stride 1 --max-moments 75 \
  --disable-face-iteration --no-write-debug-images
export INIT=outputs/realus_sync/tests/pose/lying_open_arms5_hybrik60_reg0005_cuda_v1/warmup_init/smplx_result.npz
```

在线发布复用上面 warmup_init/smplx_result.npz，不重新拟合 beta：

```bash
bash scripts/realus_fullflow.sh sequence \
  --config configs/tracking/realus_ue_5cam_dwpose_easymocap.yaml \
  --tracking-backend hybrik --gender neutral \
  --initialization-npz "$INIT" \
  --hybrik-device cuda --hybrik-refine-steps 60 \
  --triangulation-workers 3 --interval-s 0.5 \
  --disable-face-iteration --no-write-debug-images --publish-genesis
```

该发布命令要求 UE mux 与 Genesis fitted 订阅场景已运行；interval-s 0.5 只是
采样请求，不承诺实际输出速度。五机配置显式指定，不改变原四机默认 scene。

## CPU/RGB 并行与五分钟完整链路验证（2026-10-03）

当前精度配置为固定 silhouette 初始化 beta、完整 HybrIK-X RGB float32/flip
五视角 batch、previous pose 初始化、temporal weight .09、RGB pose weight .01、
80 次拟合、几何过滤与原三角化质量门。新增 CPU 三角化与主线程 CUDA RGB
并行调度；no-debug 时在 worker 内压缩回传诊断，所有候选仍参与计算。
`--hybrik-overlap-rgb` 默认开启，需显式 `--triangulation-workers 3` 才能重叠。
`--no-hybrik-overlap-rgb` 可恢复顺序执行作对照。

同一 observed55 运动/暂停/恢复输入的 73 帧新对照：平均处理 650.646 →
530.989 ms（缩短 18.39%），P95 750.477 → 650.347 ms；73/73 全部接受。
三角化数值、姿态、beta、native55 关节及 10475 个顶点逐帧 bitwise 相同。
这是计算及 NPZ 写入耗时，不含 JPEG 解码、传输和 UE 显示。
原始对照见 `outputs/realus_sync/tests/tracking_speed_20261003/observed55_comparison.json`。

直播新增 `--live-frame-selection latest`（默认）：通过原 strict/hardware/
motion 门逐组取出完整同步组，选最新有效组；不独立拼各相机最新图片。
离线 `--recording-dir` 仍按 manifest 全帧处理。`--live-frame-selection fifo`
可复现旧策略。原 live CLI 至少保留 16 组，10 Hz 输入下造成约 1.6 s 旧帧等待。
108 秒 FIFO 对照的输入到 UE 骨骼应用平均 2408.395 ms，P95 2613.824 ms。

最新组方案实际运行了 300 秒：五机 JPEG 以 10 Hz 墙钟发布，床上 source
15–55 前进 4 秒、保持 20 秒、倒放 4 秒、保持 20 秒连续循环；GPU detector、
RGB 网络和 80 次拟合均每次真实执行，Genesis viewer 与 UE PIE 同时运行，
UE 限制 30 fps。3000 组输入得到 479/479 个接受结果（约 1.60 Hz）；五机
frame span 全部为 0，全部 479 个 frame ID 都有对应的 UE 55 骨骼应用记录。
输入到 UE 应用平均 944.459 ms、P95 1138.286 ms；五分钟逐分钟均值分别
954.813、950.063、934.375、929.267、954.101 ms，未观察到延迟累积。
这些时间含 JPEG 接收/解码、处理、发布与 UE 骨骼应用，不含实体相机曝光和
显示器呈现。本机没有实体五机连接，本次是录制 RGB 输入的软件完整链路验证。

479 帧对录制 GT 的 body22 MPJPE 均值 27.967 mm；左/右膝平均误差
25.170/21.809 mm，P95 32.856/24.963 mm。每个连续静止段分开统计，排除
前 10 秒调整和未完成的最后一个静止段：身体关节相邻更新位移 P95 1.804 mm，
双膝 P95 1.351 mm、最大 1.556 mm。全部 beta 相同且输出有限；无拒绝帧。
本轮未新增显示姿态插值。UE 的固定 BEDLAM 网格不会应用拟合 beta，因此
UE 表面不能用作 beta 准确性指标；上述精度来自真实拟合 NPZ 对录制骨骼 GT。

复现脚本与结果位于 `outputs/realus_sync/tests/tracking_speed_20261003`：
`live_env.sh`、`live_replay.py`、`run_live_latest.py`；后处理用
`summarize_live.py --live-dir .../live_latest --ue-session-dir ../ue_live_session`。
`live_latest/live_summary.json` 包含严格 frame ID/时钟范围关联、各分钟延迟、
独立静止段与 GT 精度；`live_latest/integrity.json` 记录输出完整性。
启动时先完成 UE prepare/PIE，再启动 canonical bridge，避免持续 UDP tick
占用冷启动命令轮询。本轮专用 UE/Genesis/bridge/publisher 均已退出；测试后
的重复 idle 结果已删除，canonical JSONL 无损压缩并校验 SHA256，见
`cleanup_manifest.json`。本轮相关 51 项测试通过。

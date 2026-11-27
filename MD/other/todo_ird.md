# RM75 IRD / 轨迹优化实施总账

更新时间：2026-08-02

## 结论

设计在理论和工程上可行，但结果必须分成“可微引导”和“最终可行性”两层：

- `SE(3)/Yaw_J1` 的固有商空间是 5D。它不是实现 8-DOF 规划的数学前提，但对本任务很有用：去掉任务中无用的共同 J1 yaw，提高样本效率并改善梯度质量。
- 生产网络保留 9D flange 嵌入。9D 是光滑、避免方位角分支并保留 probe roll 的工程表示，不是唯一合法坐标，也不是新的物理自由度。
- J1 的约 `+/-177.96 deg` 只作为任务范围内的近似对称。没有 seam 门槛、seam 修正或 fallback；最终 Pinocchio/SQP 始终使用真实关节限位。
- soft pose/rotation、外部世界约束、rail 和 7 个关节可以联合优化。成功定义为多初值中至少一个局部解通过所有硬验证，不宣称全局最优。
- IRD/neural field 只提供 warm start、早期裕度和梯度；它不是 IK、碰撞或硬可行性证书。

## 已完成实现

### 数据、监督和精度

- 新增五路 source-disjoint split：训练、模型选择、零点校准、安全阈值校准、最终测试；按 `source_pose_id` 防止泄漏。
- boundary stencil 保存真实二分得到的 SE(3) 边界姿态；canonical GT 从真实边界姿态生成 on-manifold 零点，不再平均正负 9D 特征。
- 默认 batch `4096` 中包含 `256` 个完整 8/9-row boundary group；全局补样按 source/label 分层，并保证每 epoch 覆盖训练边界组。
- 训练归一化只使用 train split；checkpoint 记录 dataset/split/sampler/metric/output-scale provenance，并拒绝不兼容旧 schema。
- 将经验 `boundary_slope_loss` 与 `generic_eikonal_loss` 分开。generic Eikonal 默认关闭；对归一化冗余 9D 梯度直接施加目标 1 是量纲错误，只有完成商空间 Jacobian 度量推导和测试后才允许消融。
- 增加独立 on-manifold zero bias 与 unreachable-only 单侧 safety threshold；`geometric_clearance` 和 `accepted` 分开，校准文件与 checkpoint/data hash 绑定。
- 当前冻结基线（不是新的高精度发布结果）：balanced accuracy `90.38%`，recall `91.94%`，specificity `88.82%`，位置/旋转方向一致率 `99.80%/99.58%`，position crossing P95 `0.953 mm`，rotation crossing P95 `0.0955 deg`。未达到 `>=95%` 前不声称高精度目标达成。

涉及文件：

- `ird_playground/ird_playground/ird/splits.py`
- `ird_playground/ird_playground/ird/canonical_gt.py`
- `ird_playground/ird_playground/ird/gpu_boundary_stencil.py`
- `ird_playground/ird_playground/neural/train_signed.py`
- `ird_playground/ird_playground/neural/signed_field.py`
- `ird_playground/ird_playground/calib/`
- `ird_playground/ird_playground/cli/eval_signed.py`

### 集合与轨迹算子

- 修正 lower-tail Rockafellar--Uryasev CVaR 为
  `max_t [ t - (1/alpha) E(t-X)_+ ]`，支持非整数尾部、ties、常数输入和有限梯度。
- SetQuery 返回自由权重、最佳候选、局部位置 offset、beta/roll；Trajectory operator 返回可恢复角度/psi 和权重。
- TCP、rail 标量和由 rail 重建的 J1 axis frame 使用同一个段内插值参数 `alpha`；不再沿用段起点 axis。
- 保持聚合顺序：场景不确定性 -> 自由角度 -> 段内路径 -> 全轨迹；世界障碍/患者/床/接触面继续独立于 quotient。

涉及文件：`region/set_query.py`、`region/trajectory_operator.py`。

### SRS、8-DOF SQP 和控制接口

- 新增 `ird_playground/ird_playground/optimization/trajectory_sqp.py`：
  `TrajectoryOptimizationProblem`、配置/结果类型、Pinocchio adapter、IRD rail warm starts、ProxSuite SQP、HPP-FCL witness gradients、世界约束、段内自适应验证、多初值选择和 fail-closed `validate=false`。
- 唯一机器人变量是 `q=[rail,j1,...,j7]`；task pose/rotation 是软代价，医学允许带、FK、真实 rail/joint limit、碰撞、世界边界和 KKT 是硬门槛。
- 默认验收：FK `<=0.5 mm / 1 deg`、缩放 KKT `<=1e-3`、无碰撞/边界/限位违规；失败不复制上一点、不替换姿态、不返回部分路径。
- TCP 弧长以不超过 `0.02 m/s` 生成初始时间；Ruckig 只能减速。输出 `T_tcp_ref`、Cartesian feedforward、`q_ref`、`qdot_ff`、rail reference、contact normal 给现有 QP-IK/力位混合控制器。v1 不把 torque/full dynamics 放入 SQP。
- 修复 RM75 SRS 契约：`RAIL_ORIGIN_Y + q_rail` 肩部坐标、probe45 flange TCP 转换、coaxial 兼容模式、URDF 常量/限位审计、`shoulder_y_from_q_rail`。
- 同步 `pose_ik.py`，使路径插值使用相同 alpha 的肩部 world-Y 和 live flange transform。

涉及文件：

- `ird_playground/ird_playground/optimization/`
- `rm75_control/rm75_control/kinematics/srs_ik.py`
- `rm75_control/rm75_control/control/joint_admittance_8dof/pose_ik.py`

### 文档

- 更新 README、signed IRD design 和中英文 operator mathematics：5D quotient/9D selected embedding、近似 J1 对称、当前 `90.38%` 基线、独立校准、CVaR、boundary slope、rail+7R SQP、硬/软约束、`0.02 m/s`/Ruckig、QP-IK 接口及“热图变化是重新查询景观，不是网络参数在线变化”。
- 增加本文件作为从原始设计到当前实现的变更总账。

## ellipse demo 状态

`ird_playground/experiments/ellipse_vessel_ird_demo.py` 现在把原始路径与最终路径分开：

- 原始 nearest/midline 路径可以失败，只作为诊断与 rail warm start；本次确实为 `0/81` exact IK。
- 位置优化始终通过 `ellipse_surface_tcp(theta, path_y)` 留在指定椭圆截面/扫描直线上，名义中轴指向血管。
- TaskCone 返回实际候选姿态、softmax 权重、best index 和 local rotvec；候选只在 `tip +/-20 deg`、`roll +/-20 deg` 内变化（参数来自此前 demo 配置，可改为 15 deg）。
- `select_constrained_task_path()` 用动态规划选择跨路点连续的 tip/roll 姿态。代价由归一化 IRD score deficit、角度变化和当前 clearance cliff 自动构造；突变处提高可达性优先级，连续段更重视贴近原姿态，不再为 Phase A/B 手填另一组权重。
- 最终 rail+7R QP-IK 必须全部路点收敛，再经 GT/Region-A/FK/碰撞审计；失败即不生成成功产物。图中蓝线是 `Validated projection`，不是未收敛行的复制。

应使用：

```bash
source rm75_control/env.sh
PYTHONPATH=ird_playground:$PYTHONPATH \
  python ird_playground/experiments/ellipse_vessel_ird_demo.py \
  --device cpu --out-dir data/reports/ellipse_vessel_ird_demo_projection
```

最新有效产物位于 `ird_playground/data/reports/ellipse_vessel_ird_demo_projection/`。本次结果：优化路径 `81/81` 可达，GT reachable fraction `1.0`，Region-A `1344/1344` 场景可达，最大 FK 误差约 `0.197 mm / 0.052 deg`，碰撞失败 `0`，rail 为 `0.271--0.430 m`。已生成全部 PNG、两个 MP4、`qpik_guidance.npz` 和 `summary.json`。

## 验证记录

- `python -m pytest ird_playground/tests -q`: `84 passed`（2026-08-02 最新全套回归）。
- `python -m pytest rm75_control/tests/test_srs_ik.py -q`: `52 passed`（SRS/tool/rail/path 回归）。
- `python -m compileall -q ird_playground/ird_playground ird_playground/experiments/ellipse_vessel_ird_demo.py rm75_control/rm75_control/kinematics/srs_ik.py`: passed。
- `git diff --check`: passed。
- English LaTeX 用 `pdflatex -halt-on-error` 生成 22 页 PDF；Chinese compilation requires the missing local `ctexart` package。
- operator regression：交接时记录 `18 passed`；precision/planner targeted suite：`35 passed`。
- 两个新 MP4 均实际解码为 `81` 帧、`1280x512x3`；不是空文件或仅容器头。

## 尚未声称完成的项目

### 2026-08-02 审计修正

- 自适应目标现在把段级 IRD cliff 提升为同长度的路点权重；安全 clearance 达标的连续路段自动投影回零 tip/roll 名义候选，避免追逐任意峰值。新增 `waypoint_cliff` 回归测试。
- 蓝色投影路径的硬验证已绑定同一 `q_optimized_8dof`：逐点收敛/有限值、真实 rail+关节限位、FK `<=0.5 mm/1 deg` 和碰撞必须全部通过，否则立即失败，不生成成功产物。
- 椭圆 demo 已强制 GPU-only。2026-08-02 在沙箱外执行 `source rm75_control/env.sh` 后，Torch `2.12.1+cu126` 可见 RTX 4080 Laptop GPU，`torch.cuda.is_available()=True`；CUDA smoke、静态图和视频重跑完成，没有 CPU fallback。
- 独立只读监督 Agent 已完成审计并报告上述缺口；此前两个监督启动因模型 404/容量失败。当前回归为 IRD `84 passed`、SRS `52 passed`，compile 与 `git diff --check` 通过；基础 ellipse 蓝色轨迹的 8DOF/FK/碰撞验证与障碍条件查询层的验证必须分别陈述，不能把后者当成全机器人硬验证。

### 查询后障碍引导架构（确定采用）

- IRD 网络和训练数据保持不变；先完成原始 `A` 聚合，再在查询层构造可微引导景观：
  `C_guided = C_IRD - lambda_obs * barrier(d_obs)`。
- `d_obs` 来自可替换的手动/动态障碍 ESDF 或解析距离；`barrier` 在安全距离内近似为零，接近障碍时平滑增大。`lambda_obs` 由距离和梯度突变自适应门控，避免再堆固定权重。
- 离线阶段对 TCP、rail 和任务角度联合优化；实时阶段只用组合场的 TCP 梯度做小步投影修正。IRD 不在线重训，场景变化只更新障碍查询。
- TCP 引导不是全机器人碰撞证明。肘部、腕部、probe、rail 等 link 的距离约束继续进入底层 QP/nullspace 的硬安全约束和最终 HPP-FCL 验证；这保留了通用性并处理 rail 移动导致的全链碰撞。
- 5 秒级离线预算应通过 batched IRD 查询、局部 ESDF 梯度和 warm-start SQP 控制；实时路径只做固定次数的小批量查询和投影。

### 聚合前的条件障碍引导（扩展设计）

- 为避免障碍信息只在 `A` 聚合后的局部邻域起作用，增加条件查询算子：对每个场景/自由角度/rail 候选先计算 `C_IRD(z, xi_k) - lambda_obs B(d_obs(z, xi_k, o))`，再将这一组条件分数送入原有 `A`（CVaR/softmax-best/轨迹聚合）。
- 该算子只改变查询时的候选排序和尾部风险，不改变 IRD 网络、训练集或离线权重；障碍特征保持世界系独立输入，因为它破坏 yaw 商对称。
- 聚合前条件引导与聚合后 TCP 梯度修正同时保留：前者避免安全候选被平均掉，后者提供 realtime 小步投影；全链碰撞仍由底层 QP/nullspace 硬约束和 HPP-FCL 验证负责。
- 实时预算采用固定数量候选、batched IRD/ESDF 查询和有限步更新；离线轨迹使用同一条件算子 warm-start SQP，不在线重训。
- 当前状态：通用 `TrajectoryOptimizationProblem` 已支持 `WorldConstraint`、有限差分/解析 Jacobian、段内硬验证和 fail-closed；新增 `obstacle_conditioned_ellipse_demo.py` 已在 GPU 生成聚合前全局候选可视化，并将选中姿态绑定到同一条 8DOF QP-IK/FK/真实限位/robot+probe 自碰撞审计。外部测试球目前作为 TCP signed-distance 硬门槛，不代表所有机器人 link 对世界障碍的 HPP-FCL 几何审计。
- 新增 `region/conditional_query.py`：冻结 IRD 的全局候选条件层。它在 `A` 前对 `[IRD clearance, obstacle signed distance, task mask, NEARST cost]` 做平滑 barrier、硬可行过滤和 lexicographic 选择；全阻塞时返回 `valid=false`。该节点新增两项算子测试时记录为 `71 passed`，当前全套回归已更新为 `84 passed`。
- 新增 `TaskConeReachability.query_conditioned()` 和 GPU 实验子目录 `data/reports/obstacle_conditioned_ellipse/`；2026-08-02 CUDA 重跑后，81 帧视频显示 76/81 个无障碍路点保持 `0°` 可达基线，仅中间 5 点以 `-12.5,-17.5,-20,-17.5,-15 deg` 的全轨迹弹簧偏移绕行。校准 target 为 `5.0`，选中路径最小 IRD clearance `12.495`、最小障碍 signed distance `2.104 mm`（要求 `>=2 mm`）。同一条蓝线以 `0.3 mm / 1 deg` QP-IK 阈值求出 81/81 rail+7R，最大 FK `0.298 mm / 0.962 deg`、自碰撞 `0`、rail `0.263--0.647 m`，查询层和机器人硬验证均 `valid=true`。
- 同路径 81 点 QP-IK tolerance benchmark：`0.1/0.2/0.3/0.5 mm` 分别约 `21.7/18.4/16.1/13.6 s`，全部 81/81 且零碰撞；采用 `0.3 mm` 作为准确度/时间甜点值。该基准是当前 Python continuation QP-IK 全路径耗时，不等同于最终优化器 5 秒预算，后续仍需并行候选/更好 warm start 加速。
- 障碍绕行的任务位置带不固定为几毫米：由医学扫描允许带/椭圆切面/接触法向共同定义，可按路点使用数组 tolerance。硬约束只要求落在允许带内；在此可行域内采用 lexicographic projection，最小化相对原始 NEARST 的位移、角度变化和 rail 变化。
- 障碍 barrier 只在其影响半径内激活；无障碍段的权重趋近零并保持 NEARST，障碍段选择最近的安全侧绕行。多障碍时先保证全局硬可行，再按总偏移和连续性排序，不使用一组固定手调绕行权重。
- IRD 的角色是全局候选生成器：障碍出现时不能只在 NEARST 邻域做局部梯度修正。应在全局 `SE(3)/Yaw` 候选、tip/roll、rail 和场景状态上做条件查询，构造候选可行集合 `F_obs`，再按硬任务约束过滤。
- 选择顺序为：`F_obs` 中先保留完整 FK/碰撞/限位/任务带可行者；再最大化 IRD clearance/条件聚合裕度；在裕度足够的解之间最小化到 NEARST 的 pose、rail 和曲率偏移。这样可跳到障碍另一侧的全局可行位置，同时无障碍区仍贴近 NEARST。

### 历史：反向移动球的 U-band 预测绕障演示（已被后文椭球版本取代）

- 新增 `ird_playground/experiments/moving_obstacle_u_band_demo.py`，输出独立目录
  `ird_playground/data/reports/moving_obstacle_u_band/`。冻结 checkpoint，不修改 IRD 参数；障碍是贴在椭圆柱皮肤外侧的 `12 mm` 球，沿 extrusion 参数从 `0.30 m` 反向慢移到 `0.10 m`，扫描轨迹同时从 `0.00 m` 正向到 `0.40 m`。
- 规划不再使用旧静态例子的五点一阶 DP。13 个 cubic B-spline 控制点生成 C2 surface-angle 曲线；先满足 IRD 与动态球硬可行，再等量聚合无量纲 NEARST、控制点一阶连续和二阶曲率项。因此允许绕行影响传播到前后无障碍段，而不是把非障碍路点锁死到 baseline。
- 所有位置由 `ellipse_surface_tcp(theta,path_y)` 生成，严格留在椭圆皮肤；标准 frame 的 `+Z` 指向内部血管。tip/roll 只从该标准 frame 采样一次，修复旧 obstacle demo 将已选 source 姿态再次套 TaskCone 的双重角度问题。最终绝对 tip/roll 最大 `14.917 deg / 9.098 deg`，均在 `+/-20 deg` 内。
- 球条件是 TCP 位置条件，对同一位置的 tip/roll 候选相同，所以本例中与 task-cone 聚合可交换；不能把这一结论泛化到 orientation-dependent probe 几何。后者必须显式在自由角聚合前计算各姿态的障碍距离。
- 预测蓝线最大 surface offset `14.185 deg`、RMS `7.221 deg`；最大相邻差 `0.684 deg`，最大二阶差 `0.135 deg`，不再有旧图 `15 deg/waypoint` 的折角。原 baseline 有 3 个预测冲突路点；以 `0.5 deg` 偏移为开始，蓝线提前 `39` 个路点、约 `9.77 s` 平滑绕行。
- 发布硬门槛为 TCP 对动态球 signed distance `>=3 mm`，优化使用 `4 mm` 规划缓冲。81 路点最小为 `3.996 mm`，12 次/段的同步插值共 1040 个样本最小为 `3.993 mm`。这是 TCP-to-sphere 的动态硬门槛，不是假装完成所有 link 对动态世界球的 HPP-FCL；全链世界障碍仍由底层 QP/nullspace，最终控制部署时需接全 link hard constraint。
- 最终 task-cone IRD 最小 `5.009 >= 5.0`。同一条蓝线通过 81/81 rail+7R QP-IK、真实限位、FK 和 robot+probe 自碰撞审计：最大 FK `0.295 mm / 1.000 deg`，自碰撞失败 `0`，rail `0.305--0.659 m`，`rail_ref == q_ref[:,0]`。
- 几何规划 GPU 实测约 `5.45 s`，当前 Python continuation QP-IK 另约 `14.41 s`。因此场规划已接近 5 s 级，但完整 pipeline 仍超过 5 s，不能声称总体预算达标。轨迹按 TCP 弧长以 `0.02 m/s` 上限计时，总扫描约 `20.05 s`；NPZ 保存 timestamps、`T_tcp_ref`、`q_ref[:,8]`、`qdot_ff`、rail、contact normal、球心轨迹、绝对 tip/roll、IRD 和距离裕度。
- 新 PNG/MP4 沿用 `u_band_cone_reachability.png` 的 3D ellipse shell、vessel 和 reachability 色图格式，并增加原始 IRD/动态条件 IRD 双视图、红色移动球、反向预测轨迹、橙色 baseline、蓝色整轨迹和 C2 offset inset。物理 TCP 留在皮肤；图中的橙/蓝线仅为防遮挡沿外法向 lift 的显示 overlay。
- 输出：`u_band_moving_obstacle.png`（1955x881）、`u_band_moving_obstacle.mp4`（81 帧、H.264、1280x592）、`moving_obstacle_guidance.npz`、`optimization_history.json`、`summary.json`。summary 和源 NPZ/checkpoint 都记录 SHA-256，`valid=true` 只在全部上述硬检查完成后写出。
- IRD 精度问题没有被这个演示“解决”：生产冻结基线仍为 balanced accuracy `90.38%`，`>=95%` 三 seed 发布目标尚未达到；方向一致率/crossing 指标较好，但 strict straddle 与 near-axis 仍弱。此前约 `93.45%` 只是一轮非发布消融，不能替代 production retrain。

- 没有运行耗时的 production retrain/paired-sampler 三 seed 消融，因此不能宣称 `>=95%` balanced accuracy。
- corrected Eikonal 尚未作为生产默认打开；若商空间 metric 推导连续两轮不能闭合，应继续关闭。
- 旧 `ellipse_vessel_ird_demo/` 目录仍是历史产物；发布应引用新的 `ellipse_vessel_ird_demo_projection/`。
- 独立只读 supervisor 的早期两次启动因模型 `404`/容量失败；后续 retry 已完成最终障碍图、81 帧视频、summary/NPZ、8DOF FK/限位/自碰撞、基础椭圆和 IRD 指标签核。审计边界：外部测试球目前只验证 TCP signed distance，尚未验证所有 robot link 对该球的 HPP-FCL 距离。

## 最终验收定义

发布前必须同时满足：无旧指标、无 fallback、五路 split 无泄漏、零 bias 与 false-accept 校准通过、算子 AD/FD 通过、至少一个多初值 SQP 解通过全部硬约束和 KKT、Ruckig 后速度/加速度不违规、在线 field P95 性能增幅不超过 5%，以及 ellipse 图/视频只来自有效结果。

## 2026-08-02 连续 Diffusion Guidance 地基（最新状态）

本节取代上面历史“反向移动球”中离散 TaskCone 姿态 DP、iterative QP-IK 和静态整带
热力图的旧发布链；旧数据只保留为历史对照。

### 连续可微核心

- 新增 `optimization/differentiable_energy.py`。未来 policy 变量固定为
  `controls[B,K,5] = [theta,beta_x,beta_y,roll,rail]`，用 clamped cubic B-spline
  连续解码。theta/tip/roll/rail 均为光滑有界参数化。
- spline 一阶/二阶项使用解析导数矩阵，并按控制点带宽 `K-1`、`(K-1)^2`
  归一化；连续时间支持 Gauss-Legendre 权重。basis 必须在 `[0,1]` 内、非负且逐行
  partition-of-unity，否则拒绝构造。
- guidance 的 `decode/forward` 无 `detach/no_grad/NumPy/argmax/argmin`。冻结 IRD
  参数不积累梯度，但 control、rail、姿态和 batched 动态障碍 context 保持梯度。
- raw IRD 与动态 SDF 是两个独立通道。改变障碍预测时 raw IRD 逐元素不变；图中 raw
  IRD 始终用同一色标，红色 alpha 点只表示动态不可用 mask，不再用 barrier 降低整片
  IRD 数值。
- reachability clearance `>5` 后梯度仍非零。`5` 只用于采样门槛和最终证书，不是停止点。
- solver helper 先最小化归一化 smooth minimax regret，再在第一阶段 rho 容差内最小化
  total regret。各项只使用任务允许范围、field output scale、安全距离和 spline 带宽归一化，
  没有分别手调 reachability/rule/smooth 权重。
- `sample_feasible` 仅表示固定连续时间采样点通过。最终仍做动态椭球密集段内插值、SRS
  whole-path lift、真实限位、FK、robot+probe 自碰撞及 Ruckig 速度/加速度重定时。

### SRS 与正式 moving 结果

- 新增 `optimization/srs_trajectory_dp.py`：`5 deg` 全局 psi 图、`1 deg` corridor
  refinement、二阶 whole-path DP、psi/rail/joint-step 边界和 fail-closed `lift_valid`。
  `lift_valid` 不是最终 `valid`；后者只由独立 hard certificate 产生。
- moving 发布链已移除离散 TaskCone 姿态 DP 和 iterative QP-IK。SRS-DP 只在连续
  guidance 计算图外生成 rail+7R joint lift。
- moving demo 只把首末两个 spline 控制点固定到 reference；其余全部内部控制点始终可微、
  可更新。不存在由 reference clearance 预先生成的 active mask。每个 optimizer step 都从
  当前 controls 重算 TCP、tip/roll、rail axis、raw IRD、角度条件聚合、椭球 SDF 和梯度。
- 动态障碍为沿运动方向对齐的椭球，半轴 `[20,12,12] mm`；最终 TCP 硬距离为 `3 mm`，
  optimizer 使用 `5 mm` 规划余量和 `2 mm` 光滑过渡。raw IRD、angle-conditioned IRD、
  SDF 与 smooth-min conditioned score 分开保存，障碍开关不改变 raw IRD。
- 障碍 score 使用紧凑尺度化
  `C_obs = C_safe * (d-3 mm)/(2 mm)`：在硬边界为 `0`，两毫米内升到 IRD 安全分 `5`，
  边缘梯度相对旧 `100(d-d_safe)` 提高约 `25x`，超过规划带后 smooth-min 自动回到
  angle-conditioned IRD，不再把红色低分区拖到几十毫米。
- 多初值只在完整采样硬可行集合中按 position/orientation/rail/continuity/curvature
  字典序选择。IRD 裕度只能在 projection epigraph 容差内继续提高，不能覆盖最近可行投影。
- 当前 hash-linked 正式结果：蓝线最大/RMS surface offset 为 `18.061/9.864 deg`，
  最大一阶/二阶差为 `1.060/0.285 deg`；最小 raw/angle-conditioned IRD 为
  `6.393/6.249`，精确动态椭球 waypoint/segment 裕度为 `6.196/6.189 mm`。
- 同一蓝线的 rail 为 `0.284--0.664 m`，最大绝对 tip/roll 为
  `18.502/2.953 deg`；SRS 最大关节步长/二阶差为 `3.325/1.500 deg`。独立硬审计
  最大 FK 误差约 `0.000889 mm / 0.000408 deg`，robot+probe 自碰撞失败 `0`。
- 最小 combined conditioned clearance 由旧线性障碍 score 的 `0.258` 提高到 `6.387`；
  相遇帧为 `8.13`，蓝点位于红色障碍区外。
- 30 次 warm request-to-certified-q P50/P95 为 `4.288/4.564 s`；分项 P95 为
  guidance `3.833 s`、SRS `0.729 s`、硬验证和重定时 `0.0106 s`。渲染与冷加载
  不计入该时间。

### 无 Learning 的 diffusion 修正

- 新增 `experiments/diffusion_guidance_foundation_demo.py`。没有训练 denoiser；对
  certified、负侧、正侧和 random-chart 初值加入 `sigma=0.05/0.15/0.30/0.50`
  噪声，每格 3 个 seed，用相同 `grad E` 与 noise-scaled trust region 固定修正 320 步。
- 从最新局部投影轨迹重算共 `48` 个样本，硬恢复 `37/48 = 77.08%`；不修改 TCP 的
  controller-nullspace task-space proxy 初始可行率为 `14.58%`。成功样本最差 raw IRD
  `6.294`，最差密集 segment obstacle margin `5.053 mm`。
- certified、负侧和 random-chart 均为 `12/12`；正侧仅 `1/12`。这明确说明非凸
  chart 仍有不可跨越吸引域，不得宣称全局最优、整个 chart 都可恢复或两侧都稳定成功。
- 这证明同一连续 energy 可以作为未来 diffusion sample guidance 地基，但不等于已经
  训练 diffusion policy；离散 SRS-DP 从未进入 guidance 计算图。

### 动态热力图可视化修正

- 视频不再一次性画满静态 40 cm U-band。每一帧都以当前蓝色 TCP 为中心重新查询
  `+/-0.04 m` 局部 longitudinal band，并使用当前 rail 重建 J1 axis；因此 raw IRD
  patch 随 TCP/rail 向扫描方向移动并改变颜色。
- 椭球沿相反方向移动；当前椭球的 signed-distance mask 在同一局部候选 patch 上重新
  计算。因此“查询窗口移动”和“障碍条件变化”是两种不同运动，conditioned landscape
  是二者在当前时刻的合成，不是整幅背景和球同速平移。
- 当前 TCP 使用抬高的蓝色点、白色外圈和物理接触点连接线，保证始终显示在热力图点云
  上方；物理保存的 TCP 仍严格位于椭圆皮肤，没有因显示 lift 修改数据。
- `u_band_moving_obstacle.mp4` 的左右面板分别显示 live raw IRD 与真正的 conditioned
  score。起/相遇/止帧 obstacle core/halo 点数为 `0/0`、`80/51`、`0/0`，因此红色
  SDF 层只在椭球进入当前移动查询带时出现。
- 另按基础 ellipse demo 的 surface-angle x rail 格式新增
  `region_ird_field_nearest_along_s.mp4`、`region_ird_field_optimized_along_s.mp4` 和
  `region_ird_field_conditioned_along_s.mp4`。nearest/optimized 颜色表示 raw aggregated
  IRD；conditioned 版颜色表示 smooth-min conditioned score，并在相遇帧覆盖深红
  `signed distance <3 mm` 核心和 `2 mm` 软 halo，箭头显示 angle-conditioned IRD 与
  SDF 合成 guidance 梯度。NEARST 视频明确显示 `s=0/0.5` clearance 约
  `-4.74/-0.11`，解释为什么早期仅靠原始最近投影并不可发布。

### 产物与残余风险

- `moving_obstacle_u_band/` 经媒体和数据审计后严格保留 `22` 个白名单文件：核心
  JSON/NPZ、U-band 主视频/静态图、nearest/optimized/conditioned/optimization-evolution
  场视频、ellipse/joint/controls 诊断和 diffusion foundation 产物。
- `artifact_manifest.json` 保存所有保留文件的 SHA-256、删除的 7 个可重建旧副本、
  6 个 MP4 的首/中/末帧解码结果以及 30 次 P95；目录与 manifest 白名单完全一致。
- production IRD balanced accuracy 仍为 `90.38%`，`>=95%` 三 seed 发布目标没有完成；
  field error 仍可能给出错误 guidance。本次没有重训 student，也没有用 `93.45%` 非发布
  消融替换生产指标。
- 当前 30 次 warm request-to-certified-q P50/P95 为 `4.288/4.564 s`；峰值显存和
  冷启动加载仍需独立 benchmark，因此该数字只代表 warm pipeline，不代表冷启动。
- TCP 动态椭球由规划层硬审计；all-link 动态世界碰撞仍由 controller QP/nullspace 处理。
  当前 robot+probe 检查是自碰撞，不代表所有 link 对移动椭球的 HPP-FCL 证书。

### 本轮验证

- `source rm75_control/env.sh` 后，IRD/规划回归 `91 passed`，SRS 回归 `52 passed`；
  targeted 可微/椭球测试 `17 passed`，compileall 与 `git diff --check` 通过。
- CUDA smoke 和所有正式 demo 使用 RTX 4080 Laptop GPU；没有 CPU fallback。
- 生产 IRD accuracy 仍为 `90.38%`，本轮没有执行三 seed production retrain，因此不能把
  规划与可视化改进表述为 field 已达到 `>=95%`。

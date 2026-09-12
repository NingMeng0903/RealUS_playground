# V8r3：confidence 差分融合与单一命令端口能量罐

当前验证入口见 [Session 005 反馈修正与完整命令](SESSION005_FEEDBACK_FIX_20260911.md)：持续视觉、图像丢帧暂停视觉、名义任务供能及 0.10/0.15/0.05 J 罐配置。本文保留此前单端口有限预算版本的推导与实验记录；下文的 episode、0.10 死区和单端口能量公式均不是当前试验配置。

本次从 `669beac6668f659aaa05f92c5425ca0235f0523b` 修改。原 force calibration 文件的用户改动保留；没有连接或启动机器人。论文逐项核查见 [端口审核](ENERGY_PORT_REVIEW_20260911.md)。本文区分控制策略、模型内证明和需要真机验证的性能。

2026-09-11 故障修订见 [启动/停止事故核查](STOP_INCIDENT_20260911.md)：已撤销独立串口停止进程，恢复 `669beac` 的地轨和内环代码；保留本版本能量罐及融合，只修正外环数值预条件和过期心跳的错误提示。此前异常后的长时间停止停滞尚未捕获现场栈，软件回归不表示该停滞已获真机验收。下面是用户手动启动命令，本次修订未自动启动设备；先在无人接触状态核验启动、停止，再进行仿体短轨迹。

## 1. 差分到底解决什么

当前 50 mm 探头、接触面中心 TCP、图像左侧为工具 +X 的配置下，左右窗口中心是 ±15.5 mm：

\[
r_L=v_n-0.0155\omega,\qquad r_R=v_n+0.0155\omega,
\qquad r_L-r_R=-0.031\omega.
\]

共同加压不能完成这个差分，α 也不能完成它。这是已知几何下的运动学结论；不是 confidence 对运动的导数模型。

新修订 `v8r3_confidence_balance` 使用原始左右质量差，而不只计算低于 0.8 的缺口：

\[
e_c=\operatorname{sgn}(c_R-c_L)
\frac{[|c_R-c_L|-\delta]_+}{1-\delta},\qquad \delta=0.10,
\]
\[
g_F=\operatorname{clip}\left(\frac{4.5-F}{0.5},0,1\right),\quad
q_D=k_{rep}g_F|e_c|,\quad
\operatorname{sgn}(e_c)(r_L-r_R)+v_s\xi_D\ge q_D.
\]

左右都过阈值但不平衡大于死区时，也允许有限微旋；小差异不主动追平。启用一致性消融时，共同倍率 `min(gamma_L,gamma_R)` 只缩小请求，不通过独立乘左右质量改变方向。当前真实配置仍不启用这个消融。

原力矩导纳给出名义角速度 `omega_tau`。QP 仍只求一个总角速度，相关代价为

\[
J=\tfrac12w_\omega((\omega-\omega_\tau)/\omega_s)^2
+\tfrac12w_D\xi_D^2
+\tfrac12w_\alpha(\alpha-\alpha_{pref})^2+J_{normal}+J_{windows}+J_{aperture}.
\]

新修订删除了 α 与差分松弛的交叉代价。名义转动已朝正确方向并足够时，不重复叠加图像转动；不够时，图像请求与力矩名义偏好在同一个 ω 上折中。共同加载仍由原窗口任务、法向名义偏差与孔径代价限制，不能被差分任务认作修复完成。

Welleweerd 的原文采用浅层 confidence 的加权质心做姿态方向反馈、均值做平移反馈；它没有力传感器，不能替本项目证明力矩融合增益最优。两窗口差是质心信息的简化代理。死区 0.10、4–4.5 N 门控和现有权重是明确的工程策略。

每个 episode 仍限制准许尝试时间 2 s 或累计实测角行程 3°。只有连续三个有效新帧同时满足左右质量达标且差异进入死区，才重新允许尝试。持续不平衡不会因“两侧都过 0.8”反复补额。这里包含名义角运动、延迟尾段仍可继续；不是实际额外转角的硬证书。

## 2. 为什么小旋转可能接近恒力，但不能保证严格恒力

在固定接触区域、局部法向增量刚度 `k(x)≥0` 的小运动模型中，令 `K_j=integral x^j k(x) dx`，则

\[
\dot F=K_0v_n-K_1\omega=K_0(v_n-c_k\omega),\qquad c_k=K_1/K_0.
\]

增量刚度近似左右对称时 `c_k≈0`，小幅纯旋转的一阶总力变化可以很小；不对称时，原法向反馈需补偿 `v_n≈c_k omega`。压力中心、零力矩位置和增量刚度中心不是同一概念。接触边缘开闭、曲率、黏弹性及反馈带宽都会影响实际误差。

旧 V7 的两端点力方向行在 `F>4.1 N` 时要求 `delta_vn≤−0.025|delta_omega|`。它要求新增运动在全孔径都不增加压入，强于“总力大致不变”，所以会阻挡左右重新分配接触。V8 已移除这组旧行，保留 4 N 名义目标、4–4.5 N 图像修复门控和原机械保护。4.5 N 是门控值，6 N 是最新有效 raw/filtered 控制力的停止阈值，都不是未来连续力峰值的严格上界。

## 3. 这次罐子证明哪个端口

输入符号明确为 `W_model=-W_control_raw_tcp`，力与力矩六轴同时反号。该 wrench 已在 TCP/tool 坐标，不能再重复移点。压入正 Z、压缩控制力为 +4 N 时，环境模型力为 −4 N，压入消耗余额。

唯一可花余额属于 **logical_final_model**：每次完整组合发布成功后，把该次最终载荷速度模型 `V_j` 和审核时的 `W_j` 作为同一坐标快照，在该逻辑区间保持。新测量只影响下次成功安装的区间，迟到测量不回写过去的能量。

\[
P_j=W_j^T V_j=f_j^Tv_j+\tau_j^T\omega_j,
\qquad E(t+\Delta t)=\min(E_{max},E(t)+P_j\Delta t).
\]

不使用 `F−4` 算功，不只记法向功，也不把名义虚拟阻尼再当成可回收能源。初版模型的误差、变化率和附加阻尼项固定为零，是 **held pair 模型的定义**，不是实测误差为零的宣称。实测端口另作不可花的诊断，不能给此余额充值。

对声明的最长逻辑持有时间 h，准入与最终复核均要求

\[
W_j^TV_j+\beta E_{avail}/h\ge0,\qquad0<\beta\le1,
\]

其中可用余额已扣除旧未结算责任和停止留额。先预留 `h[-P_j]_+`，不提前花费预计回收；已过去的区间按同一个功率精确分段结算。于是每个已覆盖前缀满足 `E_avail+tau P_j≥0`。区间拼接、容量上截断只丢能、无负余额下截断，得到这个**声明命令模型端口**的累计能量不等式。

5 ms 控制/参考进度步长与 h 分开。拒发新候选时旧逻辑命令继续扣账。只有完整发布成功才替换旧逻辑区间；部分发布、未知发送、时钟错误或租期到期转入故障并保留责任。逻辑租期结束不表示设备已经停止，停止留额也不等于已证明真实停止能量充足。

能量多时，该准入行更宽松，QP 有机会采用原任务所希望的修复；能量少时，耗能方向受到限制。罐子不决定图像修复方向，也不强迫增大压力。若某一拍力矩近零，模型中转动功也近零；这不说明随后接触力矩和真实转动耗能仍为零。

## 4. 命令端口与实际端口的边界

实际端口是同点、同时间的 `W_env^T V_meas`。它与 `W_j^T V_j` 的差含力测量误差、变化、速度跟踪、臂轨异步和停止尾段。Keemink、Lee 和 De Stefano 都明确讨论了这些区别。臂轨速度相消后某设备先停，会改变真实组合速度；这是执行与端口证明的边界，不是要求把图像语义放入内环 IK。

本次不修改内环 IK。开启的是有独立余额、真实参与 QP 与最终载荷准入的命令模型罐；`physical_w_checked=false`、`physical_certified=false` 继续如实保留。它不是原来 `observe` 模式的法向 TDPA，也不代表已经获得真实机器人连续端口的无源证书。若需要后者，须补足六轴受力符号实测、时间同步/误差界和实际执行/停止暴露界，再把这些界用于同一个物理端口账本。

历史 active/003 的离线记录可核对检测、指令与耗能量级，不能证明新策略使图像改善。当前 confidence 提取版本与浅层窗口没有改；浅层质量仍高的深部/边缘黑区，不能承诺由新差分自动修好。

## 5. 回放依据与预算数量级

[固定五状态、60 个策略/额度组合](../../analysis_artifacts/contact_qp_v8r3/five_state_energy_policy_check.json)保留旧力、图像、名义动作和实际 slew 步长。每个状态假设有完整的修复尝试额度，旧日志缺少精确角状态，故不是连续反事实闭环。V7 首先复现保存的旧命令。

| 状态 | V7 总角速度 / °/s | V8r2 / °/s | V8r3 / °/s | 解释 |
|---|---:|---:|---:|---|
| L DtP，1452，F=4.102 N | −0.339 | −0.481 | −0.685 | 显著左弱时增加差动倾斜，共同法向速度由 0.163 降至 0.102 mm/s |
| L PtD，7288，cL=0.828、cR=0.997 | 0 | 0 | −0.241 | 两侧达标但明显不平衡；α=1、共同法向速度仍为 0.304 mm/s |

表中使用无能量行的策略比较；这两个状态在 0.001 J 和 0.2 J 可用额度、50 ms 模型持有时间下也得到近似相同结果。所有 60 个组合中 56 个可解；极低 0.00001 J 下的四个不可行结果保留，不能为了展示成功而放松机械条件。低能量并不必然让每个速度分量都变小：QP 可以减少共同压入、同时转向模型回收方向。

[旧五条轨迹的命令模型功](../../analysis_artifacts/contact_qp_v8r3/archived_model_work.json)按成功发布的六维最终载荷模型与当次 raw wrench 配对计算，未积分未知末尾时段、未包含寻触和真实停止：

| 轨迹 | 净模型输出功 / J | 正输出功累计 / J |
|---|---:|---:|
| L DtP | 0.316 | 0.320 |
| C DtP | 0.325 | 0.327 |
| S DtP | 0.377 | 0.379 |
| L PtD | 0.083 | 0.093 |
| C PtD | 0.047 | 0.062 |

这解释了为何不能给完整六维扫描直接套用旧法向模块的 1 mJ。首个试验配置采用初始 0.6 J、容量 0.8 J、普通任务不可消耗留额 0.05 J、逻辑持有上限 50 ms。它们是明确的试验参数，不是经人体/设备证明的安全能量阈值。每次新任务初始化 E0 属于一次显式初始预算，不能把重复新建任务包装成无限长连续被动证明。

## 6. 冻结配置与实际测试

2026-09-11 Session 003 的后续修正与当前完整启动命令见 [confidence 接入和发布时序修正](SESSION003_FIX_20260911.md)。当前使用 `scripts/run_icra_tank.sh`，其 record 入口会先检查有效 confidence；下面保留早期手动入口作为历史记录。

配置：`peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml`。旧 `active_probe50.yaml` 仍是 V7，`active_probe50_v8.yaml` 仍是 V8r2，不能用它们的名字代替本次版本。

先离线检查配置，不连接设备：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python \
  -m peirastic.apps.contact_qp_run --validate-only \
  --config peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

报告应显示能量预算已启用、修订为 `v8r3_confidence_balance`、结算端口为 `logical_final_model`，物理认证仍为 false/unverified。controller 增加两个能力声明；旧进程不会接受本配置。由用户结束旧 controller，再启动一次：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh controller
```

confidence 工作进程若已经以相同窗口版本 `8b51133835031ba9ee2d` 运行，可以继续使用；未运行时在另一终端启动：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python \
  -m peirastic.apps.contact_qp_features \
  --feature-config peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

录制终端：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh check
bash run.sh record --force-profile icra --speed-m-s 0.005 --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

按用户当前要求省略 `--data-root`，使用录制脚本原默认目录 `/media/camp/yameng/icra 2027/uncalibrated/NNN/`。

首轮使用可重复的仿体/原短直线轨迹，完成一条后在下一条提示输入 `q`，保留成功与失败 attempts。比较同一路径的 baseline 与新版，检查四件事：

- 图像左右差出现时，`differential_request_m_s`、名义角速度、QP 角速度及实际角度分别如何变化。候选转动不等于实际转动，缺口减小才是声学收益。
- `energy.command_budget_enforced=true`、余额/预留变化与 `logical_command_energy` 事件是否连续；拒发旧区间继续扣账，部分发布不得授予完整进度。
- 4 N 附近的 RMSE、偏差、峰值与 >4.5 N 暴露，保留与原控制器的权衡结果，不能用 tank 非负代替力精度评价。
- `nonspendable_measured_port` 和 `port_alignment` 是否取得实际臂轨区间；用其估计命令与实测功的差，缺失区间单独报告，不补积分、不反馈充值。

遇到预算、时间或发布故障，按原停止链处理并保留日志；不要反复增大额度或重建任务掩盖失败。实际机械停止不依赖这个逻辑能量罐批准。

最终回归与源码冻结记录见[最终验收报告](../../analysis_artifacts/contact_qp_v8r3/FINAL_VALIDATION.md)；本次没有自动执行上述硬件命令。

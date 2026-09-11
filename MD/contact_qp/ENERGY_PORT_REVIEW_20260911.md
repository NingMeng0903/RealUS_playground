# Energy port review — 2026-09-11

独立核读范围：Secchi 2019、Lee 2024、Benzi 2021 三篇原始论文，以及审查开始时 HEAD `669beac6` 的 `port_constraint.py`、`energy.py`、`runtime_energy.py`、`port_alignment.py` 和 `realman8dof/modes/contact_active.py`。其余论文定位由 root 并行读取原文后提供，已汇入证据表；Welleweerd融合和 physical wrench 的采集链由并行专项审查负责。本次仅写分析；未修改生产代码、未连接或驱动硬件。第4节描述审查起点，第7节给出已确认可实施的最小方案，后续生产改动应另行验证。

## 1. 裁定

**可以准备开启一个明确标注物理界未认证的 command budget 实验，不能把当前实现直接称为已证明的 command-port passivity。** 单次 QP 工作下界本身有正确推导；缺口集中在运行时采用不同速度端口预留与结算、预留时间覆盖、以及平均功率预留与端点功率结算的不一致。它们可以做有限范围的修正，不需要等待所有机器人内部模型都已辨识才允许开启 tank。

当前已具备：单一余额、完整六维端口、最终 payload model 复核、禁止预测回收立即入账、发送前退款围栏、缺测不释放旧责任、重复区间拒绝、容量只上截断而不把负债抹零。这些机制应保留。

当前尚不能具备的表述包括：已证明真实机器人端口被动；实测账本加虚拟阻尼扣能就是机器人和代理完整物理储能；开启 tank 可保证6N以内；余额更多意味着必须更多加力或转动。

## 2. 论文证据及边界

| 文献及定位 | 可直接用于本项目的结果 | 不能直接移植的结论 |
|---|---|---|
| [Secchi 2019](</home/camp/Desktop/New Folder/FORCE CONTROLLER/secchi2019.pdf>), PDF pp.2–5，式(5)–(11)、(17)–(25) | 调制 tank 可过滤任意期望速度；储能导数等于所声明输入输出端口功率；优化误差在能量约束内最小化。离散积分 tank 状态会出现额外正能量项，必须处理。 | PDF p.2明确使用实际速度约等于指令速度的假设；没有替未知执行误差、异步发送、真实停止时长给出界。 |
| [Benzi 2021](</home/camp/Desktop/New Folder/FORCE CONTROLLER/benzi2021.pdf>), PDF pp.2–4，式(9)–(14)、(17)–(18) | 任务可以有 slack，能量约束没有用于放松它的独立 slack；完整任务速度使用同一个端口预算。 | 机器人采用速度控制/运动学模型，实际跟踪仍是前提；任务完成、恒力及峰值力不是 passivity 结论。 |
| [Lee et al. 2024](</home/camp/Desktop/New Folder/FORCE CONTROLLER/Lee et al. - 2024 - Bidirectional Energy Flow Modulation for Passive Admittance Control.pdf>), PDF pp.5–8，式(11)–(13)、(20)–(33)，及 p.14 Sec.VII-B | 代理与真实速度差造成不定号能量项；其完整证明明确包含真实/名义储能、控制结构混合参数变化、阻尼与额外反馈功率。 | 不能只借其 tank 名称而省略这些项。文中自己指出惯量估计误差形成未计入 tank 的不定号项，因而相应物理被动性不能保证。 |
| [Keemink et al. 2018](</home/camp/Desktop/New Folder/FORCE CONTROLLER/Keemink et al. - 2018 - Admittance control for physical human–robot interaction.pdf>), Sec.4.2、5；root原文核读 | 明确区分 virtual/apparent admittance，并讨论 velocity-loop bandwidth。 | 指令端口的被动性不能省略速度内环而直接成为实际端口被动性。 |
| [De Stefano et al. 2020](</home/camp/Desktop/New Folder/FORCE CONTROLLER/destefano2020.pdf>), Sec.III式(9)、Sec.IV及Fig.13；root原文核读 | 明确定义离散端口；`(F_e,V_s)` 与 `(F_e,V_s delayed)` 的能量可以不同。 | Euler离散误差与执行延迟能量分别处理，不能只修其中之一。 |
| [Heck et al. 2018](</home/camp/Desktop/New Folder/FORCE CONTROLLER/heck2018.pdf>), Sec.III–IV式(12)、(15)、(19)；root原文核读 | 储能/功率记账覆盖设备及通信通道。 | 不能用 passivity-layer 名称为未记录的延迟或通道能量担保。 |
| [A Perturbation-Robust Framework…](</home/camp/Desktop/New Folder/FORCE CONTROLLER/A_Perturbation-Robust_Framework_for_Admittance_Control_of_Robotic_Systems_With_High-Stiffness_Contacts_and_Heavy_Payload.pdf>), Samuel 2024，式(7)–(9)；root原文核读 | 即便高 inner gain，payload与delay残差仍存在；CDYOB依赖相应nominal T/C/R/Q。 | 本次外层预算不应自称已具备该补偿，不需要为开启明确的模型端口预算擅自部署整套内环模块。 |
| [Ma et al. 2015](</home/camp/Desktop/New Folder/FORCE CONTROLLER/ma2015.pdf>), root原文核读 | macro/mini midrange预滤与解耦是机构/控制分工的证据。 | 不是本项目tank端口证明的来源。 |
| [Welleweerd et al. 2020](</home/camp/Desktop/New Folder/FORCE CONTROLLER/Welleweerd2020automatic.pdf>), 并行融合专项审查 | 用于审查图像置信度与力/姿态任务融合。 | 本文不从置信度控制结果推断外层tank储能或真实接触力峰值保证。 |

Secchi/Benzi 的核心可按统一符号写为

\[
\dot E=W_{\rm env}^{T}V,\quad E\ge E_{\min}
\quad\Longrightarrow\quad
\int_0^tW_{\rm env}^{T}V\,d\tau\ge E_{\min}-E(0).
\]

这里 `W_env` 是环境施加给探头的 wrench，`V` 必须与其在同一个点、同一坐标系，且声明清楚是指令速度还是实际速度。环境输入功率为正时 tank 可回收，机器人向环境输出功率为正时 tank 消耗：

\[
P_{\rm in}=W_{\rm env}^{T}V,\qquad P_{\rm out}=-W_{\rm env}^{T}V.
\]

因此对于指向接触内部的正法向速度，环境反作用 wrench 的相应法向分量应为负。控制器使用的“+4N目标/正压紧标量”不是自动可用于功率的 wrench。现场符号链必须依据实际数据定义核对，不能根据变量名称或配置布尔值推断。

Lee 的关键式(13)留下 `(V_proxy−V_meas)^T W_env` 不定号项；其后用 `S=(1−α)S_n+αS_r`，并在 tank 中计入 `αdot(S_r−S_n)` 等项。**该文 α 是控制结构混合参数，与本项目的扫描进度 α 不同。** 论文 Sec.VII-B还列出遗漏项

\[
-\dot\alpha\,\frac{\lambda}{2}\dot q^T(M-\widehat M)\dot q.
\]

这些不是当前 QP 外层 tank 已自动覆盖的已知量。但也不意味着必须重写成 Lee 的控制器：若能直接对真实外端口建立每个时间前缀的可信工作下界，并通过可执行机制保持非负预算，外端口账本就能给出相应证书。现状尚未建立这些物理界。

## 3. QP 的单次工作下界：推导成立，前提必须明确

对应 [port_constraint.py](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/port_constraint.py:67)。设 hold 为 `T`，最终指令在该窗口使用固定向量 `v`；声明

\[
|W(t)-\widehat W|\le\epsilon_w+t\rho_w,\qquad
|V(t)-v|\le\eta,
\]

并设 `r=P_c V`、`|r_i|≤rbar_i`、`d_i≥0`，则

\[
d_i r_i^2\le d_i\bar r_i |r_i|,
\quad |P_cV|\le|P_cv|+|P_c|\eta.
\]

记 `a=d⊙rbar`、`q(t)=ε_w+tρ_w/2`，得到

\[
L(t;v)=t\left[\widehat W^Tv-q(t)^T|v|
-(|\widehat W|+q(t))^T\eta
-a^T\bigl(|P_cv|+|P_c|\eta\bigr)\right]
\]

以及

\[
\int_0^t\left[W^TV-\sum_i d_i(P_cV)_i^2\right]d\tau\ge L(t;v).
\]

现有 `velocity_rows()` 保证的正是 `|P_cv|≤rbar−|Pc|η`。QP 的绝对值辅助变量形成凸约束；全部六维 `v` 进入功率，扫描、法向和转动共用余额。

`L(t)` 关于 `t` 是凹二次函数，且 `L(0)=0`。因此

\[
L(T;v)+\beta E_{\rm avail}\ge0,\quad 0<\beta\le1
\]

足以推出 `E_avail+L(t;v)≥0` 对全部 `0≤t≤T` 成立。`aged()` 将源时间和求解/发送延迟累加到 wrench 误差，同时保留完整后续 hold，方向正确。这里 `tracking_error=0` 是理想跟踪模型的声明，不能解释为真实误差已验证为零。

直接更新能量状态 `E`，而不是 Euler 积分 `x_t` 后令 `E=x_t²/2`，可以避开 Secchi 式(17)讨论的离散额外造能项。现有 ledger 没有这类 tank 状态积分造能错误。仍需解决功率积分误差与指令持有时间问题，它们不是同一个误差。

## 4. 当前开启路径中的具体缺口

### 4.1 预留使用指令速度，结算使用实测速度，余额因而属于混合端口

[runtime_energy.py:282](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/runtime_energy.py:282) 用最终 `V_cmd` 计算预留；[runtime_energy.py:92](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/runtime_energy.py:92) 与 `observe_port()` 用实测/插值 `V_meas` 结算，并把其回收直接增加同一可花余额。

可复现：初始 `E=.1J`，`W=-1`，发布 `V_cmd=+1` 持续10ms，指令端口花费 `.01J`。若实测该区间 `V_meas=-1`，当前结算释放该命令预留并把余额变为 `.11J`。这是物理端口与指令端口不同，并非一次重复结算；但这个余额不能证明指令端口累计被动。

**最小修正方向：** 给余额明确且唯一的结算端口。如果目标是 command-port tank，余额由实际发布的最终指令时间序列及声明的 wrench 输入模型结算，实测外端口观测不得增加该余额。实测数据仍完整记录，但不是第二个可花 tank。若保留现有路径进行短期试验，准确名称应为“指令预算已启用、实测结算为估计、物理界未认证”，不能升级为 command-port passivity 证明。

### 4.2 预留的平均功率被当成逐时功率上界

`reserve()` 将 `max(0,−L(T))/T` 存为恒定 `power_upper_w`。这个数可覆盖本次模型的**前缀累计消耗**，却未必是每一时刻输出功率的上界。[energy.py:220](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/energy.py:220) 的 settlement 使用最大端点功率及 rate 裕量，又在第266行将它与上述平均值比较。

可复现：`W(t)=−1−10t`、`v=1`、`T=.01s`，且 `wrench_rate=10` 已正确声明。模型预留 `.0105J`，恒定标记为 `1.05W`；端点结算取 `1.1W×.01s=.011J`，产生 `execution_exceeded_reserved_power_envelope`。该轨迹没有违反已声明的 wrench rate。

**最小修正方向：** 统一准入、预留和结算的包络。简单实现可全部采用 hold 内的点态最坏输出功率，并让 QP 和预留使用同一较保守值；或者保留积分下界，但将 reservation 和 settlement 都改成前缀工作包络比较。不要仅删掉告警，也不要保持不匹配的 settlement 算法后把合法消耗超预留当真实硬件越界。

### 4.3 持有超期与停止过程没有接到预留覆盖

[contact_active.py:159](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/realman8dof/modes/contact_active.py:159) 使用 `dt=min(actual_dt,baseline.dt)` 作为下次 hold 模型；`reserve()` 仅登记 `[publication_now, publication_now+dt]` 的 `new_only` 段。下一次成功发送之前旧 payload 的持续作用、arm/rail 两设备可能的混合执行、以及停止过程，不会仅因这个区间到期自动消失。

可复现：预留10ms，实际正功区间15ms，后5ms无覆盖；ledger 记 `execution_exceeded_reserved_power_envelope`，但下一次 `reserve()` 仍返回 True。原因是 [energy.py:140](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/energy.py:140) 仅在 `bounds.verified` 为真时由该 certification 状态阻止普通预留，而 runtime 明确使用 unverified bounds。

`stopping_reserve_j` 当前只是从可用余额扣除的固定额度；[contact_active.py:332](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/realman8dof/modes/contact_active.py:332) 的 `record_stop()` 只 abort 和记日志，没有 reserve stop segment、记录实际终止时间或结算该尾段。不能声称固定扣除一个 stopping reserve 就已经保证停止能量。

**最小修正方向：** command 模型需要明确定义最终 payload 在发布、替换、拒绝、停止及超期时的有效时间线；记录并保留旧命令责任，普通指令缺少时间覆盖时转入既有停止流程。新增独立的 command-budget validity，不用“物理认证 false”作为一切错误都可继续的理由。停止动作本身不应被 QP 能量不足卡住，但其曝光必须记账。物理停止界未认证时仍标记未认证，不靠软件字段声称已经取得。

### 4.4 实测对齐是合理估计，但不能承担已认证回收

[port_alignment.py](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/port_alignment.py:1) 使用原始 arm/wrench 源时间和 raw rail bracket，缺失、逆序、过期时切断区间，不外推；这是正确取向。速度采用 `J(q_mid)Δq/Δt` 后转入 tool frame，wrench 在端点间插值。由此得到的是区间估计：不能保证相同时间戳之间存在的快速往返运动、姿态变化及 wrench 功率峰被界住。

[energy.py:41](/media/camp/EXT_DRIVE/RealUS_playground/peirastic/contact_qp/energy.py:41) 的端点误差展开 `|W|η+|V|ε+εη` 正确；最大端点功率加 `LΔt/2` 在真实功率 Lipschitz 界成立时是充分上界。当前 bounds 是 unverified，默认零误差/零 rate 不是从异步插值得来的保证。

缺测不结算、不退款可以阻止凭空回收，但持续缺测会把已过时责任永久留在预算中，最终抑制或停止运动。这是保守退化，不是余额泄漏的理由；选择 command 结算后不应再让 rail bracket 可用性决定 command 模型责任是否可结算。

### 4.5 D 是额外预算消耗，不是完整物理阻尼证明

`_dissipation_estimate()` 使用 `Δt∑d_i(Pc(V0+V1)/2)_i²`。对于区间内变化的真实速度，均值平方一般不是平方积分的上界；代码的 monitor 注释已经准确承认这一点。

外端口能量过滤可使用 `E_next=E+W_in−D_budget`，其中 `D_budget≥0` 只是主动丢弃部分可花余额。这保守于普通端口被动性，**不需要假装该数值一定对应实际机械阻尼**。但它不能同时被解释成：某真实/虚拟动力学中的阻尼已经耗散了这些能量，因此可给 tank 充值，或者它已经补齐 robot/proxy 的储能变化。要使用后两种物理解释，必须给出相应储能与功率互连方程，并排除重复记账。

对最小 command-port tank，`D_budget=0` 完全允许；保留正 D 也必须在三个阶段使用一致的端口、积分规则及含义。不能根据名义 admittance 配置中的 damping 数值，未经证明直接建立另一条可回收信用。

## 5. 下一次开启应达到的层级

| 层级 | 可用结论 | 本次建议 |
|---|---|---|
| 单次 QP 模型 | 在声明的 W/V/hold 误差条件内，每个前缀工作不超过预算 | 已有正确核心；保留 full-port hard constraint 和最终 payload 复核。 |
| 持续 command-port tank | 同一个声明 command 端口累计满足非负储能与工作界 | 修正唯一结算端口、预留/结算包络、完整命令时间覆盖；在软件和 fake device 中验证。 |
| 真机 command budget 实验 | 最终指令受到真实启用的预算约束，实测能量估计可评估表现 | 允许作为下一步；日志明确 enabled、结算端口/估计规则、physical uncertified；不宣称6N峰值保证。 |
| 真实外端口被动性 | `W_env,V_meas` 的累计真实工作有界 | 需要真实符号/参考点/量纲，跟踪与力误差界，全部执行及停止时间覆盖，可靠工作上/下界；或另一套完整机器人控制器储能证明。 |

这不是要求把整个系统变成某篇论文的控制结构。目标是先使已启用预算的含义和软件行为一致，再按证据提升物理结论。

融合层仍可把 confidence 修复目标放在 objective 中，把 tank 放在可行集中。固定状态下 `E2≥E1` 使可行集扩大；更大余额只是允许更接近期望任务，并不强迫公共加力或更大转角。低余额也不能用减小扫描 α 来伪造已经满足左右差动需求，但 α 对真实六维端口功率的影响必须计入同一预算。

## 6. 本次验证及下一轮最小回归

本次已在纯内存中复现第4.1、4.2、4.3的三个数值案例，输入分别明确区分指令/实测反向、合法力变化率和 hold 超期；未运行硬件。已有测试执行结果记录在本节末。

修正后必须覆盖的独立事实：

1. 同一个指令轨迹在不同实测跟踪误差下，command 账本结果不因观测端口不同而凭空回收；physical diagnostic 仍反映不同实测功率。
2. 合法 wrench ramp 的 QP、reservation、settlement 采用同一界；任意 hold 前缀预算非负。
3. 延迟发送、一次拒绝、两设备部分发送、旧命令继续、超期和停止尾段均有唯一时间归属与责任；重复结算及开始发送后的退款均拒绝。
4. `E>0` 时 QP 可以正常修复；接近耗尽时最终 payload 受限；缺测/失配按声明模式退化；不得静默关闭能量约束继续普通动作。
5. 六维 wrench/twist 的同点变换保持 `WᵀV`，包括 rail 对 TCP 速度的贡献；压入消耗、撤出回收的测试使用物理符号，不能把控制标量符号冒充环境 wrench。

重复运行一个只读 QP 100000次不能证明这些持续运行性质；它证明的是求解不私自修改余额，二者应分别报告。

验证使用实际 rm75 Python 环境及其 cmeel 路径；系统 Python 缺少 `proxsuite`/`pinocchio`，不是本项目的完整测试环境。本审查未安装或修改环境依赖。三个第4节反例使用纯numpy/运行时类，均已执行得到报告中的数值。后续实现的独立复核结果见第8节。

## 7. 已收敛的最小实施合同：logical final-model port

本方案可交付真正 `energy_constraint_enabled: true` 的实验配置，无需修改名义 force law、内环IK或假定真实硬件执行误差为零。

**端口定义。** 令一个模型 epoch 为

\[
\mathcal E_k=(\widehat W_k,V_{k,\rm final},t_k,t_{k,\rm expire},\text{common frame}_k).
\]

仅在完整组合发布被确认成功后，才安装这个 epoch。`W_k` 取本次 publication review 已获得且合乎源新鲜度规则的样本，`V_final` 是同次最后经过能量复核的完整六维最终payload模型。在该epoch内，两个向量同时冻结在其共同坐标快照；新样本更新下一次候选输入，不追溯或改变已激活epoch的 `W_k`。每个epoch可用不同共同坐标表示，标量 `W_k^T V_k` 在同时作正确坐标变换时不变；不跨epoch混乘旧frame的速度和新frame的wrench。

这是**最终发布模型的离散、held-pair端口**，不是实际TCP的采样重建。第一版可以明确设 `D_budget=0`、模型wrench/tracking error/rate为零；这些零定义此逻辑模型，不声明真实传感器和跟踪误差为零。

区间功率常数 `p_k=−W_k^TV_k`，因此

\[
E(t+\Delta t)=\min(E_{\max},E(t)-p_k\Delta t),
\quad R_k=\max(0,p_k)\,h_k.
\]

只回收已经闭合的模型区间，始终保留一个可花余额；正负六维功率先求和，再决定净消耗/回收。这个规则消除了第4.1的混合端口和第4.2的平均/端点包络冲突。

**事务顺序。**

1. `review(now)`：结算旧epoch到单调当前时刻；保留旧剩余责任；对新最终pair按显式 `max_command_interval_s` 预留。该参数是模型lease，不是声称已测得的硬件最大delay。QP的hold和final复核必须使用同一个lease，不能QP用5ms、发布预留用50ms而期望边界仍一致。
2. `publication_started`：围栏任何可能发送后的候选退款。
3. 双设备 `success(commit_now)`：先结算旧epoch到commit；结束旧logical epoch，释放其commit之后不再使用的模型尾段；安装新pair。新pair的expiry仍为review时确定的expiry，不在commit时免费向后平移。review到commit期间尚未激活的新模型尾段可按逻辑定义释放，但不得称为真实设备未曝光证明。
4. 明确 `no_send`：仅取消新候选；旧epoch继续。partial/unknown立即latch普通发布，保留未确认候选的模型预留，走原停止流程；禁止借超时/失败重置余额或重用command id。保留这笔pending预留不等于已为全部真实old/new/stop曝光取得工作上界。
5. `stop/expiry`：结算至有效lease内的逻辑终止时刻。若定义lease到期令该虚拟信号变为零，需要在声明中明说这只是模型端口结束；不等于设备自动停止。真实超期、部分执行及停止尾段作为 physical unknown exposure 记录，不能被模型尾段释放抹去。超过lease的新普通发布拒绝，直至显式新实验/恢复协议；不得静默延长已耗尽的模型授权。

在有效模型时间覆盖内，上述事务加 `E≥0` 保持

\[
\sum_k\int W_k^TV_k\,dt\ge-E(0).
\]

review到commit还必须有旧epoch覆盖；若commit已超过review证书/lease或旧epoch截止，不能将其记为一个正常已证明的success，需要按异常发布终止普通周期。不要靠发送成功回调事后把已失效的许可恢复为有效。

**模块分工。** 新 `command_budget.py` 管理唯一spendable余额、候选责任、epoch和command-valid状态；`contact_active.py` 只连接snapshot/review/started/success/abort/stop生命周期；`runtime_energy.py` 与aligner继续提供实测观测，但不得修改command余额。可复用底层ledger的算术/预留逻辑；结算入口应明确命名command model，不能把人工 `W_held,V_final` 包装成真实 `time_aligned PortInterval`。配置与日志明确 `settlement_port=logical_final_model`、`input_hold=per_successful_publication`、`physical_assurance=unverified`。

若以后要求W模型在每次异步measurement到达时都更新，则应先结算旧pair、在因果可用时刻切换W，并为仍在持续的旧V重新计算剩余责任；必须覆盖新候选成功前的持续时间。那是下一层扩展，不应在本次最小实现中暗中引入。

## 8. 后续实现的独立复核结果

MEDIUM执行者已按第7节实现 `CommandBudget`、active生命周期接线、明确schema与实测nonspendable日志。本审查再次只读diff和测试，未修改其生产代码。**未发现阻塞开启该logical模型预算的剩余问题**；该结论不升级物理端口认证。

确认到位的修正包括：QP采用同一50ms模型hold；review复核源新鲜度；实际组合发布成功时才安装不可变pair；新expiry不因commit延迟延后；旧epoch覆盖不足、原任务证书过期或partial/unknown均latch；已发后不退款；逻辑停止只结束模型输出，真实尾段明确unknown；模型工作及epoch事件定期drain进日志。Enabled分支没有第二个monitor tank，实测aligner输出 `nonspendable_measured_port`，其数据不修改command余额。

**STOP释放的仅是逻辑输出终止后不再使用的模型尾段。** `actual_tail=unknown` 是对真实停止/持续执行尚未结清的日志说明，不会自动建立持续、有限且可信的物理能量预留。保留的pending unknown只保留该候选既有的模型责任；它不能证明全部真实旧命令、新命令、两设备混合执行和停止过程的能量都已覆盖。停止后的真实尾段没有因logical余额释放而获得物理结算或物理有界性证明。

独立验证结果：

- 定向运行 `test_contact_qp_command_budget.py`、`test_contact_qp_command_budget_active.py`、`test_contact_qp_confidence_balance.py`、`test_contact_qp_port_alignment.py`：**53 passed**。
- 独立1000个随机六维W/V模型epoch，使用变化的事件间隔和review→commit延迟，再执行stop；余额与另一条独立累计的 `∫W_heldᵀV_final dt` 一致，误差小于 `1e−12J`；每次预留/提交均满足 `balance−reserved≥stopping_reserve`。
- 新融合覆盖双侧均超过 `.8` 的 `(.82,.97)` 情况，以及名义转动与图像方向相反时在同一ω变量上的折中；差动不能由α停止代付，tank可使无法资助的转动留下差动shortfall。

这里的50ms是本次模型lease参数，不是实测设备停止界，也不是接触力变化率保证。通过后可使用真正enabled配置执行下一轮既定fake publication/真机实验流程，并从日志分别评价command预算执行与实际机械表现。

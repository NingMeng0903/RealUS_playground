# Anatomical GPS 机器人超声第三方审查包

仓库 **NingMeng0903/RealUS_playground** · 分支 **review/b-mpc-validation-20261113** · 实验快照 **`f1dca81d1cd3da5232ba25947097013d7a5ea537`**

本文汇总从材料图册到主动复查的可复现实验。下文的相对链接指向同一 GitHub 快照中的源代码、训练数据、冻结模型和评估记录。设计总稿为 [Anatomical GPS V9.4](ideas/Anatomical_GPS_V9.4.tex)，早期独立审查入口为 [B_REVIEW_REQUEST](B_REVIEW_REQUEST.md)。

V9.4 的开放提议、e3nn 响应、dIRD 偏好和可变深度条件树属于更大的研究设计。当前可复现的主线使用解析几何与有限概率模型、ProDMP、采样可行性检查和共享 MPC。跨体型 A 有采样 IK/FK 结果，固定角 B 主实验使用参考图册上的运动代理。审查主张应对应具体比较和运行环境。

**给审查人的简要结论。** 目前最明确的机制结果来自段内超声反馈，以及同一观测流上保留匿名候选图、用新证据延后确认。材料图册在已验证的 SMPL-X 体型和姿态中支持同一地址与相对探头方向的复用。新版 ProDMP 更容易实现同一探查请求。主动复查在部分仿真条件能增加正确完成，收益取决于已有证据和复查动作权限。最新去掉相对路程、转角淘汰门槛后的敏感性复查中，强 MPC、公开信息教师、固定回访、旧学习 B 分别完成 **472、496、505、460 / 512**。由学习 B 实现稳定跨任务提升、真实 B-mode 身份确认及机器人闭环性能仍待实测。

## 1. 数据从哪里来，哪些部分需要训练

| 部件 | 输入和构造 | 拟合或更新 | 交付物 |
| --- | --- | --- | --- |
| 材料图册和粗先验 | Among_US 已配准腿部动脉、静脉作为名义几何。现有公开截面表含 65 个扫描站，各有两条血管的截面位置。参考 SMPL-X 绑定表皮和内部材料格点 | 图册离线求深度和绑定。当前 β、姿态 q 给出几何运输 | [截面表](experiments/ab_streaming_prodmp_two_vessel_20261113/SECTIONS.json) · [提取代码](experiments/ab_streaming_prodmp_two_vessel_20261113/extract_sections.py) · [绑定与跨体型协议](experiments/ab_streaming_prodmp_two_vessel_20261113/CROSS_BETA.md) |
| 仿真感知与在线后验 | 在实际仿真探头位姿生成匿名超声质心包。场景级解剖偏移和标定偏差保持一致，检测受视野、遮挡和失检影响 | 每帧贝叶斯更新。在线估计有效偏移、动态残差、身份概率，膝角实验额外估计响应斜率 | [原始传感器](experiments/ab_streaming_prodmp_two_vessel_20261113/sensor.py) · [动态滤波](experiments/ab_streaming_prodmp_two_vessel_20261113/residual.py) · [膝角滤波](experiments/ab_streaming_prodmp_two_vessel_20261113/pose_response.py) |
| A 第一遍扫描 | 规则视觉伺服教师实际跑 **128 个合成躯体场景**，每个 4 s，20 ms 运动采样、100 ms 取像。每例切出四段执行记录，共 **512 段** | 从执行过的横向位置和速度拟合 ProDMP，沿腿推进由任务提供。MPC 在线选动作，段内用新图像调节 | [教师与拟合](experiments/ab_streaming_prodmp_two_vessel_20261113/run.py) · [演示与权重](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/DEMONSTRATIONS.npz) · [样本结果](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/SUMMARY.json) |
| CEM 离线复查动作 | 从同一初扫后的状态出发，尝试回退、换窗和前进。搜索先得到名义请求，再用段内反馈执行并接回完整 MPC | 用独立扰动复验动作是否真的提高完整扫描结果。合格的名义轨迹拟合新的 ProDMP，聚类提取共享探查请求 | [CEM 与轨迹拟合](experiments/ab_search_embedding_a_20261117/adaptive_teacher.py) · [发现记录、权重](experiments/ab_search_embedding_a_20261117/results/learned_b_cem_v1/discovery/) |
| 公开信息教师与学习 B | 在共同初扫后，从已有观测、位置、完整后验和合法候选构造训练表。每个候选跑到最后，记录失败、错认、路程、转角和耗时 | 教师在线用公开后验做短段加完整续扫的 rollout。学生离线拟合相对 MPC 的候选结局。上线选择第一段，之后接回同一 MPC | [训练与测试](experiments/ab_search_embedding_a_20261117/run_learned_recovery.py) · [特征、回归、选择](experiments/ab_search_embedding_a_20261117/learned_recovery.py) · [当前模型和数据](experiments/ab_search_embedding_a_20261117/results/visible_atlas_v1/) |

原始 A 的传感器把粗先验写成参考截面上的离散有效偏移。横向网格为 −6 至 6 mm、间隔 2 mm，深度网格为 −2、0、2 mm。先验参考标准差为横向 3.5 mm、深度 1.5 mm，按高斯单元质量分配。仿真单例另带固定标定偏差，标准差为 0.55、0.30 mm。检测质心噪声标准差为 1.0、0.68 mm。每根血管各有一条清晰/失检马尔可夫链。滤波器维护两角色联合可见性和偏移。尺度和概率来自模拟器设定。 [传感器参数与似然](experiments/ab_streaming_prodmp_two_vessel_20261113/sensor.py)

后续 B 的固定膝角世界使用 **24 个离散假设**。两种身份/连接解释、三个横向偏移（−4、0、4 mm）、两个声影侧和两个布局共享同一参考几何。连接组只在局部过渡处提供鉴别线索，过渡后两种解释的无标签截面可再次一致。无分叉组依靠局部深度变化提供身份线索。观测包括动作依赖的声影、固定于材料的伪结构、均匀杂波及 6% 整包失检。主样本两组以不带身份的质心包进入滤波。固定角恢复的 24 假设 belief 与膝角响应实验中的 $w,z,k$ 联合滤波是两套具体模型。 [世界生成](experiments/ab_search_embedding_a_20261117/recovery_world.py) · [身份似然](experiments/ab_recovery_training_20261115/world.py)

## 2. 材料图册、血管残差和图像更新

### 2.1 固定材料地址

$\xi=(\theta,h,d)$ 中，$\theta$ 是模板的周期环向标签，$h$ 是沿腿的模板站点标签，$d$ 是无量纲向内深度。在参考模板上用中心线与平行运输截面建立 $(\theta,h)$。图深度满足 $d_\mathrm{skin}=0$、$d_\mathrm{inner}=1$。若邻接边权 $a_{ij}=\|X_i-X_j\|^{-2}$，则自由节点满足

$$
\sum_{j\sim i}a_{ij}(d_i-d_j)=0,\qquad
L_{FF}d_F=-L_{FB}d_B .
$$

皮肤点绑定原 SMPL-X 网格三角面及重心权重，体内四面体版本绑定单元 $K$ 及重心权重 $b$。变体型、姿态只运输顶点位置。同一材料标签与同一绑定关系一起迁移：

$$
\Phi_{\beta,q}(K,b)=\sum_{i\in K} b_i X_i(\beta,q),\qquad
\xi_{\beta,q}\!\left(\Phi_{\beta,q}(K,b)\right)=\xi_{\rm ref}(K,b).
$$

平滑版本构造周期张量积样条外壳 $S_{\beta,q}(h,\theta)$ 与核心样条 $C_{\beta,q}(h)$，用

$$
F_{\beta,q}(h,\theta,r)=(1-r)C_{\beta,q}(h)+r S_{\beta,q}(h,\theta)
$$

运输体积。令模板径向深度间隔为 $\Delta_j=d_j-d_{j+1}>0$。对 $\log \Delta_j$ 在 $(h,\theta)$ 上做样条、指数化并归一化，再在每条径向线上用共享节点斜率的三次 Hermite 拼接。相邻段割线斜率绝对值 $a,b>0$ 诱导同一节点的径向斜率

$$
\left.\partial_r d\right|_{r_j^-}
=\left.\partial_r d\right|_{r_j^+}
=-\frac{3ab}{a+b+\sqrt{(a-b)^2+ab}} .
$$

共享斜率和深度函数随 $(h,\theta)$ 平滑变化，给出联合 $C^1$ 深度。严格径向单调使 $r$ 可从 $d$ 唯一求得。全局反查使用另外的几何证书，在每个接受的体型、姿态上用区间界证明预条件映射 $\widetilde F$ 的强单调性 $(\widetilde F(b)-\widetilde F(a))^\top(b-a)\ge m\|b-a\|^2$，$m>0$。证书适用于已验收区域，材料地址运输描述模型对应。真实个体血管对应仍由观测验证。 [四面体图册与标架](experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_geometry.py) · [平滑构造和证书](experiments/ab_streaming_prodmp_two_vessel_20261113/SPLINE_ATLAS.md) · [样条实现](experiments/ab_streaming_prodmp_two_vessel_20261113/spline_atlas.py) · [独立证书代码](experiments/ab_streaming_prodmp_two_vessel_20261113/spline_certificate.py)

探头标定姿态以同一材料点处的局部标架 $G_0(\xi)$ 保存。局部切向和向内方向构造新标架，姿态恢复为

$$
\rho(\xi)=G_0(\xi)^\top R_0(\xi),\qquad R_{\beta,q}(\xi)=G_{\beta,q}(\xi)\rho(\xi).
$$

原有局部探头角度可迁移到新世界位姿。运动生成中的 $s,x$ 是参考模板扫描站点与参考横移毫米量，会转换成材料地址和实际探头目标。它们与图像测得的物理横向毫米在变体型时通过局部投影变化关联。 [位姿运输协议](experiments/ab_streaming_prodmp_two_vessel_20261113/CROSS_BETA.md) · [几何与投影](experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py)

### 2.2 在线概率血管图

可统一写成

$$
c_t(\xi)=\Phi_{\beta,q_t}(\xi)
+G_{\beta,q_t}(\xi)\left[w+g(\xi,u_t)+z_t\right],
\qquad z_{t+1}=\rho_z z_t+\epsilon_t,\quad
\rho_z=e^{-\Delta t/\tau}.
$$

$w$ 是参考条件下持续的有效偏差，可混合体表配准、解剖差异和稳定接触效应。$z_t$ 是短时、尚未解释的有效运动。OU 模型的创新方差为 $\sigma_z^2(1-\rho_z^2)$。已知驱动 $u_t$ 在本实验取膝角，名义 SMPL-X 运输 $\Phi$ 之外的额外相对滑移是

$$
g(\xi,u_{\rm ref})=0,\qquad
g(\xi,u_t)\approx J(\xi)(u_t-u_{\rm ref}),\qquad
J(\xi)=E(s(\xi))\,k\,e_{\rm lat}.
$$

已提交的膝角实验预先设定膝旁衰减 $E$、方向 $e_{\rm lat}$ 和斜率先验，用递推贝叶斯估计被试/病例斜率 $k$（mm/°）。同一角度复扫提高当前图的定位，跨角度复扫给 $k$ 提供辨识信息，下一次独立复扫保留 $w,k$ 联合后验并重置短时状态。通用 $J$ 可以扩成 $3\times m$，包含深度和多个外部驱动。这里以已提交、可复现的横向膝角实验作为证据，深度响应和心动扩展另行验证。 [动态与膝角模型](experiments/ab_streaming_prodmp_two_vessel_20261113/DYNAMIC_RESIDUAL.md) · [膝角实验与复扫协议](experiments/ab_streaming_prodmp_two_vessel_20261113/KNEE_RESPONSE.md)

实际帧使用当前标定探头切面，预测每条血管在切面内的对应位置 $\mu_t(\zeta,T_t)$。隐藏假设 $\zeta$ 按试验包含身份、几何、偏移、可见性或响应斜率。对于匿名包 $Y_t$，枚举合法关联 $\alpha_t$（包括漏检与杂波）并边缘化：

$$
\begin{aligned}
b_{t|t-1}(\zeta)&=\sum_{\zeta'}P(\zeta\mid\zeta')\,b_{t-1}(\zeta'),\\
L(Y_t\mid\zeta,T_t)&=\sum_{\alpha_t}P(Y_t,\alpha_t\mid\zeta,T_t),\\
b_t(\zeta)&=\frac{L(Y_t\mid\zeta,T_t)b_{t|t-1}(\zeta)}
{\sum_j L(Y_t\mid j,T_t)b_{t|t-1}(j)} .
\end{aligned}
$$

这样直接在成像空间比较预测与观测，合并不同观测时也保留各自探头位姿。该模型估计任务需要的有效位置和身份。单一超声视角通常无法唯一识别各种物理残差的来源。 [投影和更新循环](experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py) · [联合过滤实现](experiments/ab_streaming_prodmp_two_vessel_20261113/pose_response.py)

### 2.3 保留候选图和延后确认

以材料地址对应的扫描区间归档候选。保存采集帧 $f$、采集位姿 $T_f$、候选 $j$ 和该帧中每种假设下的条件身份责任 $r_{fj}(\zeta)$。新超声只更新一次后验，历史图随后重新评分：

$$
r_{fj}(\zeta)=P(j\text{ 为目标}\mid \zeta,Y_f,T_f),\qquad
p_{fj,t}=\sum_{\zeta} b_t(\zeta)\,r_{fj}(\zeta).
$$

区间以当前 $p_{fj,t}\ge0.95$ 的最佳图为确认图。后验翻转可以替换或撤销旧结论。区分实际拍摄、后续确认、曾经误认和终局误认。主验收要求到达扫描终点，六个区间都保留本次扫描实际取得且身份正确的图。历史照片可以在新证据到来后确认，±3 mm 居中作为单列质量指标。固定轨迹消融测量这一策略能多利用多少已经拍到的证据。 [控制器与验收](experiments/ab_search_embedding_a_20261117/adaptive_controller.py) · [新主验收](experiments/ab_search_embedding_a_20261117/VISIBLE_ATLAS_RECOVERY.md)

## 3. A 如何造示范、学习 ProDMP、执行闭环

每个 100 ms 时刻，教师先用概率关联选可信目标。令 $e_{\rm img}$ 为其横向误差，当前未达身份阈值时取零。接着执行带速度和加速度限制的伺服修正：

$$
v_{\rm cmd}=\operatorname{clip}(v_{\rm nominal}+5e_{\rm img},-v_{\max},v_{\max}).
$$

教师保存真实执行的 $x,v$、匿名候选、关联分数和修正量。把每次 4 s 执行切成四个 1 s 训练段。设 $\gamma\in[0,1]$ 是段相位，$T$ 是段长，$(x_0,v_0)$ 是接续点，$x_g,v_g$ 是指定终点。积分基写为

$$
\begin{aligned}
x(\gamma)&=x_g+e^{-\lambda\gamma}
 [(1+\lambda\gamma)(x_0-x_g)+\gamma T v_0]+\mathcal B(\gamma)a,\\
a&=A^\dagger
 \begin{bmatrix}x_g-H(1)\\Tv_g-H'(1)\end{bmatrix}+N\eta,\qquad
A=\begin{bmatrix}\mathcal B(1)\\\mathcal B'(1)\end{bmatrix},\quad AN=0 .
\end{aligned}
$$

$H$ 为第一式的齐次项。由于 $\mathcal B(0)=\mathcal B'(0)=0$，起点位置和物理速度接续到当前执行状态。零空间内的 $\eta$ 用段内位置与速度最小二乘拟合，保持终点条件。原 A 再用 $[1,x_g-x_0,v_0]$ 岭回归预测形状。中途有可信新图即重算剩余段，保持段尾时间和当前 $(x,v)$，通过离散速度、加速度和位姿约束后执行后续 100 ms。换图导致的反馈改动可在段内生效。 [真实教师代码](experiments/ab_streaming_prodmp_two_vessel_20261113/run.py#L73-L110) · [拟合和重条件化](experiments/ab_streaming_prodmp_two_vessel_20261113/primitive.py)

跨体型 A 的高层用两段候选前瞻。若 $H_\mathrm{img}(a;b)$ 为三个代表曝光的预测命中平均值，$L(a)$ 为该段 TCP 路程，则归纳其当前实现为

$$
S(a;b)=H_\mathrm{img}(a;b)-0.008L(a),\qquad
a_1^\star=\arg\max_{a_1\ \rm feasible}\Big[S(a_1;b_t)
+0.65\max_{a_2\ \rm feasible\mid a_1} S(a_2;b_{t+1|t})\Big].
$$

第二段使用已知转移传播的预测后验，并在第一段预测末状态枚举。实际执行第一段并反复读图，进入下一段时用真实后验重规划。材料路径映射到笛卡尔目标后逐步做 IK/FK 和限速等检查。后续 B 基准复用相同的“实际观测更新—重规划”原则，并使用独立的 6 区间检查任务得分和名义运动控制器。 [A 的两段 MPC 与采样检查](experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py#L209-L276) · [B 基准的闭环控制器](experiments/ab_search_embedding_a_20261117/adaptive_controller.py)

## 4. 复查数据如何生成，B 如何训练

### 4.1 离线动作发现与 ProDMP 拟合

1. 模拟器先抽取隐藏世界与扰动，再真正执行共同初扫。较早的试验从首次公开失检或混淆处截取状态。当前可见目标与 CEM 对照使用两个共同 MPC 段后的同一检查点。对一个场景的分支共享隐藏几何和扰动发生器，运动不同会产生不同观测。
2. CEM 对单段请求 $\theta=(\Delta s,\Delta x,b_{\rm bow})$ 采样，表示纵向位移、横向位移和中点弯量。初步用解析分段轨迹实现，校验采样速度、加速度、边界及总预算。重复实际路径去重。每轮以高终局效用的候选更新采样均值和尺度。历史协议还讨论多段分支方法，当前 CEM 训练结果来自单段发现后接回 MPC。
3. 离线发现可使用固定模拟真值算候选后果。候选执行时仍使用因果图像反馈，出段后接回同一 MPC 直到结束。用新的共用扰动复验正例。历史学习 B 数据的发现阶段是两组各 48 场景，共 96 场景。每场景三个初始分布，每个两轮、六次提案。搜索每候选四个共享扰动，用另外八个扰动复验。最终 **29 条**合格名义轨迹参与该版拟合。
4. 拟合使用**决策时的名义轨迹**上的位置、速度，经零空间拟合得到形状，再用 $[\Delta x,b_{\rm bow}]$ 最小二乘回归。在线按实际起始 $x,v$ 重条件化，仍采用实时反馈。合格请求聚类成六个共享候选，和原动作库一起给所有对照。新版选择前优先尝试 ProDMP 实现，同端点解析轨迹可作为共同备用。

候选搜索效用的历史实现是 $U= S-1.25W_{\rm final}-0.002L-0.02T$，$S$ 是完成指示，$W_{\rm final}$ 是终局错认，$T$ 是模拟执行时间。漏检、错认、耗时过高、未能覆盖的分支依然是决策学习的数据。短 1.5/2.5 mm 摆动的早期 pilot 仅做烟雾测试。 [修订造数协议](experiments/ab_search_embedding_a_20261117/PROTOCOL.md) · [实际发现与拟合代码](experiments/ab_search_embedding_a_20261117/adaptive_teacher.py) · [冻结发现摘要](experiments/ab_search_embedding_a_20261117/results/learned_b_cem_v1/discovery/summary.json)

### 4.2 公开教师、相对 MPC 标签和学生

令 $h$ 是当前可获得的图像历史、图册与机器人状态、后验和覆盖。令 $\pi_0$ 是完整强 MPC，$a_0$ 是其本轮选择。对每个合法候选先闭环执行第一段，再接回完整 $\pi_0$，形成六维结果向量 $M=(S,W_{\rm final},W_{\rm new},L,\Theta,T)$。每个候选和 MPC 用同一批后验世界与扰动配对：

$$
Q(h,a)=\mathbb E[M\mid h,\ a\to\pi_0],\qquad
\widehat{\Delta M}(h,a)=\frac1N\sum_{i=1}^N
 \left(M_i(a\to\pi_0)-M_i(a_0\to\pi_0)\right).
$$

公开教师使用 $b(\zeta\mid h)$ 采样，线上可用信息决定动作。特权 CEM、事后枚举和部分离线标签可访问模拟真值。后两者作为数据或机会诊断，真值字段留在生成器和评价器。可学习特征包含表面地址 $(h,\sin\theta,\cos\theta,d=0)$、候选路径采样点、成像范围外侧对应的表皮地址及原投影深度范围、24 维后验、候选关联和可见性、当前覆盖与任务预算。候选特征同时输入绝对值和相对 MPC 的差。

学生以六个小型直方图梯度提升回归器拟合 $\Delta M$。代码使用最大 125 次迭代、15 叶、最小叶 24 样本。在线保留 MPC 并设它的增量为零，筛出合规候选后按以下分数选第一段：

$$
s(h,a)=\widehat{\Delta S}-1.25\widehat{\Delta W}_{\rm final}
        -0.002\widehat{\Delta L}-0.02\widehat{\Delta T}.
$$

旧协议的相对预测上限含路程 +2 mm、转角 +2°、时间 +0.10 s、两类误认各 +2 个百分点。最新敏感性仅移除前两项淘汰门槛，保留路程软惩罚、时间/误认门槛和模拟器绝对总预算 **67 mm、8 s**。报告仍分列实走路程、转角、帧数、模拟时长与决策 CPU。验证统计按场景配对，在一个场景内汇总扰动重复后 bootstrap，四次重复不是四个独立解剖病例。 [特征与分数实现](experiments/ab_search_embedding_a_20261117/learned_recovery.py) · [成像范围和位置编码](experiments/ab_search_embedding_a_20261117/embedding.py) · [最新实验协议](experiments/ab_search_embedding_a_20261117/COMPONENT_VERIFICATION.md)

| 训练版本 | 造数及训练划分 | 代码与档案 |
| --- | --- | --- |
| 初版两阶段，固定角造数后做屈伸迁移 | 192 个 A 案例，1,435 段用于筛选，944 段拟合 ProDMP。B 由 320 个独立训练场景产生 235 个合格检查点、1,410 个动作后果。按预测完成/误认选首段 | [训练方法与结果](experiments/ab_recovery_training_20261115/README.md) · [训练及评估入口](experiments/ab_recovery_training_20261115/run.py) · [训练记录](experiments/ab_recovery_training_20261115/results/) |
| CEM 共享库及学习 B | 两组各 144/40/96 个训练/开发/测试场景。每个训练候选 10 次公开后验 rollout、4 次真实模拟病例复验。开发在纯后验与 65%/35% 混合标签中选定，测试每场景 4 条新扰动 | [协议](experiments/ab_search_embedding_a_20261117/LEARNED_B_PROTOCOL.md) · [统计](experiments/ab_search_embedding_a_20261117/LEARNED_B_RESULTS.md) · [逐例、训练数据、模型](experiments/ab_search_embedding_a_20261117/results/learned_b_cem_v1/) |
| 教师独立复验后蒸馏 | 288 个旧训练状态加 128 个新无分叉状态。只基于公开前缀分层采样。关键候选用 12 条新扰动复验，训练动作分类和带符号优势两类学生。开发选模后用新病例测试 | [协议](experiments/ab_search_embedding_a_20261117/VERIFIED_TEACHER_PROTOCOL.md) · [蒸馏代码](experiments/ab_search_embedding_a_20261117/distill_recovery.py) · [数据和模型](experiments/ab_search_embedding_a_20261117/results/verified_teacher_b_v1/) |
| 当前留图及材料地址版 | 两组各 96/32/96 个训练/开发/测试场景。每训练候选 8 次后验、4 次评价采样。开发各候选 8 次后验、6 条评价扰动。最终冻结纯后验标签模型。政策比较各测试场景 4 条扰动 | [协议与运行命令](experiments/ab_search_embedding_a_20261117/VISIBLE_ATLAS_RECOVERY.md) · [冻结摘要](experiments/ab_search_embedding_a_20261117/results/visible_atlas_v1/frozen_models.json) · [权重和逐例](experiments/ab_search_embedding_a_20261117/results/visible_atlas_v1/) |

这里的学习 B 是模型生成监督下的首段选择。历史 CQL、PPO 与完整条件策略是分开的实验分支。当前验证的固定角 B 在一段后接回完整 MPC。把一般理论写成 $\operatorname{E}_O[\max_{u\in\mathcal F(O)}J(u;O)]$ 与 $\max_u\operatorname{E}_O[J(u;O)]$ 的区别有助于陈述未来可行动作，但该交换属于标准观测分支规划。当前项目的可检验贡献应落在**材料地址上的可复查证据怎样改变首次探查选择并产生正确完成或耗时收益**。 [早期 B 理论与评价边界](B_REVIEW_REQUEST.md) · [完整 rollout 复核](experiments/ab_full_b_rollout_20261113/INDEPENDENT_REVIEW.md)

## 5. 结果与现阶段可支持的结论

每行对应独立协议，历史版本曾改变留图策略、覆盖条件和候选库，效果量应在行内配对解释。

| 问题 | 已核实的结果 | 可支持的结论 |
| --- | --- | --- |
| 跨 β/姿态及平滑图册 | 五个固定构型合计 **39/40** 完成，零错认。注册过的样条绑定替换四面体后，三种控制器的 **516 个配对结局逐例相同**，静态/IID/OU 各 **151/172、144/172、153/172**。不合格构型会被证书拒绝 | 在测试构型内保持材料表达、探头方向复用和对应任务结果。 [跨形状结果](experiments/ab_streaming_prodmp_two_vessel_20261113/CROSS_BETA.md) · [样条验收](experiments/ab_streaming_prodmp_two_vessel_20261113/SPLINE_ATLAS.md) |
| 段内反馈与原 A 的拟合 | 冻结学得形状时，段边界反馈 **174/256**，段内反馈 **214/256**，提高 **15.63 pp**。同为段内反馈时，端点版 **213/256**，反馈示范拟合版 **214/256**，物理横向误差从 **0.752 降至 0.691 mm** | 反馈时机是主要完成率收益。学习形状有小幅跟踪改善。 [输入与逐例结果](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/) |
| 残差 $z_t$ 与膝角 $g$ | 100 ms 有效横向预测 RMSE 静态/IID/OU **1.435/1.390/0.921 mm**。横向膝角斜率多角度复扫相对固定角重复的 RMSE **0.193→0.071 mm/°**，名义 90% 区间覆盖 **21/24**。有响应的闭环病例均为 **20/24** 完成，正确居中帧 **729→733 /984** | 仿真支持短时预测和带激励的参数辨识。任务成功率增益仍有不确定性。 [OU 状态结果](experiments/ab_streaming_prodmp_two_vessel_20261113/DYNAMIC_RESIDUAL.md) · [膝角结果](experiments/ab_streaming_prodmp_two_vessel_20261113/KNEE_RESPONSE.md) |
| 早期及历史主动探查结论 | 原移动 Y **76/128 对 76/128**。旧固定角学习 B **136/192 对强 MPC 143/192**。旧居中验收下，CEM 共享库的学习 B **317/768 对 292/768**，公开 rollout **365/768**。独立教师蒸馏轮中，蒸馏 B **330/768**、旧 B **349/768**、强 MPC **304/768**、公开 rollout **359/768** | 公开动作前瞻存在特定任务增益。学生对照跨版本和无分叉收益不稳定。 [移动 Y](experiments/ab_tro_y_continuation_20261112/REPORT.md) · [CEM 共享库结果](experiments/ab_search_embedding_a_20261117/LEARNED_B_RESULTS.md) · [蒸馏结果](experiments/ab_search_embedding_a_20261117/VERIFIED_TEACHER_RESULTS.md) |
| 延后确认的固定流消融 | 同一现有 MPC 轨迹和观测，非分叉组 **0/384→384/384**，连接组均 **384/384**。两臂最终误认零，曾误认为同一初扫的 4 条重复扰动。当前可见验收完整闭环下，强 MPC 和学习 B 都 **768/768** | 延后确认确实利用了此前会丢弃的早期候选。当前普通分布的完成率已到上限。 [同流消融](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/evidence_summary.json) · [当前留出结果](experiments/ab_search_embedding_a_20261117/results/visible_atlas_v1/test_summary.json) |
| CEM 或 ProDMP 的独立效应 | 加宽声影后等提议数下，合预测门槛的随机/CEM 路径 **205/275**，复验过门槛的开发场景两组均 **0/48**。留出随机与 CEM **465/512**，原库 **467/512**。同请求新版 ProDMP 直接实现 **96/96**，旧版 **85/96**，共同备用后解析/旧/新都 **359/384** | CEM 增加搜索中发现的合规路径，当前在线完成率增益未确立。新版 ProDMP 减少备用调用，其实现和训练配方同时改变。 [模块消融](experiments/ab_search_embedding_a_20261117/COMPONENT_VERIFICATION.md) · [CEM](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/discovery_summary.json) · [ProDMP](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/render_summary.json) |
| 真实超声 | 204 段现有录制，约 12.35 万帧，含对齐的探头位姿；已检查范围内缺少独立血管身份标注 | 真实流式回放输入具备，身份精度、纠错延迟与真实闭环待有标注数据验证。 [聚合输入审计](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/recorded_input_audit.json) |

### 最新压力测试及放宽相对代价后的读法

将仿真声影原有两条尺度加倍（2.6→5.2 mm、4.5→9 mm），测试 128 个独立种子、每例四组配对扰动。旧相对门槛下，强 MPC 和公开教师逐例同为 **472/512**，旧学习 B 为 **460/512**。随后按任务偏好，取消相对路程、转角的淘汰门槛。根据已保存的公开预测重选首动作、在同一扰动重放变化的动作，继续使用原强 MPC。固定回访由开发集按完成数选出，并受相同物理总预算。以下为在原留出集上的**事后敏感性分析**，独立新病例验证待做。

| 首段策略 | 连接歧义 | 无分叉 | 合计 | TCP mm / 转角 ° / 仿真 s | 新误认 / 终局误认 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 强 MPC | 216/256 | 256/256 | 472/512 | 34.98 / 6.13 / 4.74 | 1 / 0 |
| 公开信息教师，放宽相对运动门槛 | 240/256 | 256/256 | 496/512 | 35.24 / 6.31 / 4.52 | 2 / 0 |
| 固定先回访再接 MPC | 254/256 | 251/256 | 505/512 | 37.94 / 6.68 / 5.46 | 2 / 0 |
| 旧学习 B，放宽选择门槛 | 212/256 | 248/256 | 460/512 | 34.42 / 6.07 / 4.72 | 1 / 0 |

教师相对 MPC **+4.69 pp**（配对 95% 区间 **[1.56, 8.59]**），连接组增益为 **+9.38 pp**，整体相对 MPC 平均多走约 **0.26 mm**，模拟运动耗时少 **0.22 s**。固定回访多完成 33 次，平均多走约 2.97 mm，多花约 0.72 s。公开前瞻曾测得约 **23.95 s CPU** 每次首决策，强 MPC 约 **24.0 ms**，旧学习 B 约 **61.6 ms**。仿真在规划时暂停，效率结论应分开报告计算时延。冻结学习 B 的动作与结局在放宽后相同。

在 MPC 的 40 条失败扰动里，事后枚举显示每条都存在预算内可挽救的首动作。移除相对路程/转角门槛后，保留时间/误认限制时有 **311** 个成功替代分支，逐条最少额外路程的中位数为 **13.89 mm**。事后枚举根据真实回放结局逐条挑动作，此数目描述动作库的机会。只使用公开信息的教师在事后分析中净增加了 24 次，且较固定回访用更少模拟时间。这支持继续验证**何时回访**，并把固定回访纳入强对照。 [完整协议和局限](experiments/ab_search_embedding_a_20261117/COMPONENT_VERIFICATION.md) · [放宽门槛统计](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/relaxed_choice_summary.json) · [逐例](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/relaxed_choice_cases.jsonl.gz)

历史替代路线也保留。拓扑复查的探索策略在同模板组有正点估计，在形状留出组效果反转，当前强化学习试验的 PPO 阳性完成为零，选模效用与阳性检查目标存在错位。旧 CQL 在开发集匹配路程的协议中做到 **118/128**，前瞻规则 **125/128**。固定倾转量的 Fisher 排序定位 RMSE **3.6173 mm**，可见性优先 **3.5559 mm**。这些结果覆盖旧候选和任务定义。 [拓扑留出审计](experiments/ab_topology_reinspection_b_20261113/independent/HELDOUT_REVIEW.md) · [CQL 实际路程匹配](experiments/ab_knee_matched_20261112/PAPER_CORE.txt) · [Fisher](experiments/ab_fisher_prodmp_20261112/FISHER_PRODMP_REPORT.txt)

## 6. 第三方复核入口

| 要复核什么 | 主要入口 |
| --- | --- |
| 参考几何和图册重放 | [跨体型说明及命令](experiments/ab_streaming_prodmp_two_vessel_20261113/CROSS_BETA.md) · [四面体和注册样条核查](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/SPLINE/) |
| 段内视觉反馈、教师示范与检查 | [A README 和命令](experiments/ab_streaming_prodmp_two_vessel_20261113/README.md) · [示范 NPZ](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/DEMONSTRATIONS.npz) · [逐帧与逐时间步记录](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/TRACES.npz) |
| 几何/状态滤波与膝角 | [动态滤波数值与独立重放](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/DYNAMIC/) · [膝角后验及记录](experiments/ab_streaming_prodmp_two_vessel_20261113/verification/KNEE_RESPONSE/) |
| CEM 造数与复查学习 | [CEM 运行协议](experiments/ab_search_embedding_a_20261117/LEARNED_B_PROTOCOL.md) · [命令及冻结模型](experiments/ab_search_embedding_a_20261117/VISIBLE_ATLAS_RECOVERY.md) · [当前训练/开发/测试原始包](experiments/ab_search_embedding_a_20261117/results/visible_atlas_v1/) |
| 本轮消融与代价敏感性 | [统一评估入口](experiments/ab_search_embedding_a_20261117/verify_recovery_components.py) · [命令与各臂对照](experiments/ab_search_embedding_a_20261117/COMPONENT_VERIFICATION.md) · [全部记录](experiments/ab_search_embedding_a_20261117/results/component_verification_v1/) |
| 其他尝试与负结果 | [原移动 Y](experiments/ab_tro_y_continuation_20261112/REPORT.md) · [完整条件策略和 MPC 尾段](experiments/ab_full_b_rollout_20261113/INDEPENDENT_REVIEW.md) · [随机形状与 PPO](experiments/ab_topology_reinspection_b_20261113/independent/HELDOUT_REVIEW.md) · [CQL 等代价复验](experiments/ab_knee_matched_20261112/PAPER_CORE.txt) · [Fisher 与可见性对照](experiments/ab_fisher_prodmp_20261112/FISHER_PRODMP_REPORT.txt) |

快速复算用 Python 3.10.12、NumPy 2.2.6、SciPy 1.15.3、scikit-learn 1.7.2、joblib 1.5.2。已有归档可直接复验敏感性。原始 SMPL-X 资产和真实录制按各实验的授权数据路径使用。仓库的衍生截面表可直接跑固定角 A/B 公开归档。

```sh
git checkout f1dca81d1cd3da5232ba25947097013d7a5ea537
cd NMI_idea/experiments/ab_search_embedding_a_20261117
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m unittest -v test_qualified_evidence
python verify_recovery_components.py relaxed --out results/component_verification_v1 --jobs 12
```

第三方应优先核对四点：

1. 追踪训练、开发与测试种子和冻结散列，确认同一个场景下各臂共用前缀、扰动发生规则和续扫控制器。
2. 核对在线策略输入中没有隐藏身份或未来帧。检查证据是否按采集、确认、撤销三个时刻记录，以及 ±3 mm 是否只作为当前方法的辅助指标。
3. 复验固定回访是否能解释连接组收益。若短固定回访结果相近，比较随证据改变动作的策略能否节省扫描用时，同时维持完成和误认。
4. 在独立新种子以及少量有独立身份真值的真实超声序列上冻结验收。共享候选留图和成像模型，公平报告正确完成、曾经误认、终局误认、实际里程、转角和实时时延。先确认公开策略的排序有可重复收益，再决定是否重训 B。优先使用现有动作库，同时对照固定回访和感知条件触发的简单回访。

## 7. 主要参考论文及使用边界

| 论文 | 项目中用到的思想 |
| --- | --- |
| Pavlakos 等，*Expressive Body Capture: 3D Hands, Face, and Body From a Single Image*，CVPR 2019 | SMPL-X 提供一致的模板拓扑及体型、姿态的外部几何。内部材料图册、血管和超声验证由本项目另建。 [论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavlakos_Expressive_Body_Capture_3D_Hands_Face_and_Body_From_a_CVPR_2019_paper.html) |
| Joshi 等，*Harmonic Coordinates for Character Articulation*，SIGGRAPH 2007 | 调和坐标与边界控制的背景。项目特有的是环向/纵向模板标签、图调和径向深度以及固定材料绑定。 [原作者论文](https://graphics.pixar.com/library/HarmonicCoordinatesB) |
| Li 等，*ProDMPs: A Unified Perspective on Dynamic and Probabilistic Movement Primitives* | 积分基、端点条件化与在线轨迹重规划。本文实现集中于低维横向形状、边界空空间和实际视觉反馈。 [原文](https://arxiv.org/abs/2210.01531) |
| Laskey 等，*DART*，CoRL 2017；Ross 等，*DAgger*，AISTATS 2011；Hoque 等，*IntervenGen*，IROS 2024 | 训练覆盖教师/学生实际可能遇到的偏差和失败状态，保留可执行纠偏示范。当前实现采用仿真检查点分支收集与 CEM 复查，按论文区分噪声注入、在线聚合和人示范扩增。 [DART](https://proceedings.mlr.press/v78/laskey17a.html) · [DAgger](https://proceedings.mlr.press/v15/ross11a.html) · [IntervenGen](https://autolab.berkeley.edu/assets/publications/media/IROS24_2512_FI.pdf) |
| Pinneri 等，*Sample-efficient Cross-Entropy Method for Real-time Planning*，CoRL | 连续动作请求的交叉熵搜索思路。本仓库当前运行使用低维高斯提案的精英重估。 [iCEM 论文](https://proceedings.mlr.press/v155/pinneri21a.html) |
| Choudhury 等，*Learning to Gather Information via Imitation*，ICRA 2017 | 使用有模拟器信息的教师规划能力和长期任务后果监督公开信息学生，适合作为数据构造的先例。 [原文](https://arxiv.org/abs/1611.04180) |
| Pathak、Thomas、Indelman，*A Unified Framework for Data Association Aware Robust Belief Space Planning and Perception*，IJRR 2018 | 在信念演化中显式保留多关联和主动消歧的理论近邻。全候选关联本身已有先例。 [原文](https://journals.sagepub.com/doi/10.1177/0278364918759606) |
| Zhong、Fazeli、Berenson，*RUMI: Rummaging Using Mutual Information* | 主动感知可直接进入 MPC 的近邻。本文用身份正确的实际采图与整段完成作为方法验收。 [原文](https://arxiv.org/abs/2408.10450) |
| *VesNet-RL* 与 *BIFURC* | 前者为机器人超声标准纵切面学习、后者为血管分叉识别。可供比较感知、主动导航和连接辨识的实验设置。 [VesNet-RL](https://arxiv.org/abs/2205.06676) · [BIFURC](https://arxiv.org/abs/2409.06817) |
| Kumar 等，*Conservative Q-Learning for Offline Reinforcement Learning*，NeurIPS 2020 | 旧 CQL 补采实验的基线。当前 B 的主要学习结果来自监督候选结果回归。 [原文](https://arxiv.org/abs/2006.04779) |

# 材料图册上的贝叶斯机器人超声扫描

实现基线采用 RealUS_playground 审查分支的 68b60ed。两个深度漂移报告来自同一工作区的动膝实验。

目标是从粗解剖先验出发，依据扫描过程中实际采到的超声，找到指定血管、确认身份、交付所需区间的图像并完成续扫。图册保存跨姿态的地址，贝叶斯滤波维护多个血管解释，ProDMP 生成可接续的动作，B 在检查存在缺口时选择下一段复查。

### 1　先在一条参考腿上定义材料地址

$$
\xi=(\theta,h,d)\ \in\ S^1\times I_h\times[0,1].
$$

θ 是绕腿一周的周期标签，h 是髋、膝、踝骨轴上继承的纵向站位，d 是由皮肤向内增大的深度。参考右腿只取干段。用参考剖面的内部中心、以原站位 h 为参数拟合平滑中心线 C₀(h)，随后构建每一站的剖面标架。

$$
t_i=\frac{C_0^\prime(h_i)}{\|C_0^\prime(h_i)\|},\qquad e_{1,0}=\mathrm{unit}\!\left[(I-t_0t_0^\top)(p_{\rm pelvis}-p_{\rm hip})\right].
$$

$$
e_{1,i+1}=\mathrm{unit}\!\left[(I-t_{i+1}t_{i+1}^\top)Q_i e_{1,i}\right],\quad e_{2,i}=t_i\times e_{1,i}.
$$

Qᵢ 取把相邻切向对齐的最小旋转，因此不额外绕切线扭转。每站按同一套角度标签向皮肤发射射线，取唯一正向交点。保存命中皮肤三角面的顶点编号和重心权重。射线长度为 ℓ，r 是从中心向皮肤的半径比例。

$$
X(h,\theta,r)=C_0(h)+r\,\ell(h,\theta)\,[e_1(h)\cos\theta+e_2(h)\sin\theta].
$$

沿同一材料射线向外走，半径比例逐渐增大，深度标签从内壳的 1 降为皮肤的 0。

### 2　图调和方程定义深度

连接右腿参考体网格上的相邻节点。皮肤的 d 固定为 0，内壳固定为 1，中心轴在求解域以外。每个自由点的深度等于邻居深度的加权平均。

$$
a_{ij}=\frac{1}{\max(\|X_i-X_j\|^2,10^{-20})},\quad L=\mathrm{diag}\!\left(\sum_j a_{ij}\right)-[a_{ij}].
$$

$$
\sum_{j\sim i}a_{ij}(d_i-d_j)=0\quad\Longleftrightarrow\quad L_{FF}d_F=-L_{FB}d_B.
$$

F、B 在这一式分别表示自由和边界节点。它是离散 Dirichlet 能量最小化的驻点条件。当前实现的 θ、h 来自参考几何，调和求解给出 d 的节点值。

> 两个实验入口共用上述材料地址的设计。运动膝角实验检验血管动态预测与两段 MPC，最新复查 B 实验在固定膝角下检验身份恢复。二者的实现状态与验收条件见后页。

[构造、深度与源网格说明](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/SPLINE_ATLAS.md) · [跨体形运输](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_bodyshape_atlas_transfer_20261113/transport.py)

## 连续的内部深度与唯一地址

### 3　先平滑层间深度差，再连接相邻径向层

固定参考坐标 h、θ，按由内至外的半径排序 r₀,…,rⱼ。每层原始图深度之差 γⱼ 为正。在纵向 h 上做三次插值，在 θ 上做周期三次插值，作用于对数差值。再指数化和归一化，重建连续的层深函数。

$$
\gamma_j=d_j-d_{j+1},\quad b_j=\mathcal{I}_{h,\theta}(\log\gamma_j),\quad \widehat\gamma_j=\frac{e^{b_j}}{\sum_l e^{b_l}},\quad D_j=1-\sum_{l<j}\widehat\gamma_l.
$$

记 Δrⱼ 为层厚，νⱼ 为正割线幅值。两端节点的斜率幅值取最近的 ν，内部节点从两侧割线 a、b 算出同一个 mⱼ。由于向外深度降低，实际节点斜率为 −mⱼ。

$$
\delta_j=r_{j+1}-r_j,\quad \nu_j=\frac{\widehat\gamma_j}{\delta_j},\quad m_j=\frac{3ab}{a+b+\sqrt{(a-b)^2+ab}},\quad (a,b)=(\nu_{j-1},\nu_j).
$$

$$
t=\frac{r-r_j}{\delta_j},\quad D=D_j-\delta_j m_jt+\delta_j(2m_j+m_{j+1}-3\nu_j)t^2+\delta_j(2\nu_j-m_j-m_{j+1})t^3.
$$

这四个系数由两端的深度与径向斜率四个条件唯一决定。相邻两段共享同一个 Dⱼ(h,θ) 和同一个 −mⱼ(h,θ)。沿 h、θ 求导时，共享的 Dⱼ 也给出一致的两项切向导数，所以任意方向 v 的一阶导数 vᵀ∇D 连续。

$$
D_r(r_j^-)=D_r(r_j^+)=-m_j,\qquad \partial_vD=v_h D_h+v_\theta D_\theta+v_rD_r.
$$

−∂D/∂r 的三个二次 Bernstein 系数为 mⱼ、3νⱼ−mⱼ−mⱼ₊₁、mⱼ₊₁，均为正。因此沿一条射线可以唯一反求 r(h,θ,d)，同时保留皮肤和内壳边界。连续延拓复现模板图节点的深度。

### 4　另验三维体积有没有折叠

不同射线仍可能相交，需要独立检查三维映射 F。将参数写成 (h,r cosθ,r sinθ)，在凸实心参数柱上延拓。对每个待验几何固定可逆的预条件变换，以区间界检验归一化映射的雅可比对称部分。沿两点间线段积分即得全局单射。

$$
\frac{D\widetilde F+(D\widetilde F)^\top}{2}\succeq\mu I,\quad \mu>0\quad\Longrightarrow\quad (\widetilde F(y)-\widetilde F(x))^\top(y-x)\geq\mu\|y-x\|^2.
$$

> 五个具体右腿几何均通过连续域检查，最小证书下界为 0.08126。22,400 个反演往返测试最大误差为 2.0 nm，衡量的是数值地址反演。新 β、q 的几何适用范围和真实人体血管配准精度须分别验证。

径向标记 D 的方向导数定理与三维映射 F 的单射证书分开使用。三维点落在有效壳内且雅可比非奇异时，逆函数定理给出局部 C¹ 的物理位置到材料地址映射；跨角度缝口用圆周坐标表示。

[样条实现](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/spline_atlas.py) · [单射证书与配对闭环](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/SPLINE_ATLAS.md)

## 同一点、同一探头手法如何随人体移动

### 5　把材料标签固定在模板上

第 i 站、第 j 个角度的皮肤结点在模板上绑定到三角形 f，三个重心权重 λ 已知。更换 SMPL-X 体形 β、姿态 q 时，读取同样三个皮肤顶点的新位置。由此，同一皮肤标记随身体变形。

$$
S_{ij}(\beta,q)=\sum_{l=1}^3\lambda_l V_{f_l}(\beta,q),\qquad \sum_l\lambda_l=1.
$$

内部核心也要随体形与姿态移动。先将该站皮肤环的平均静息形状位移加给参考核心，再以该环的平均蒙皮权重和当前关节变换运送核心。用运送后的结点构建 C(h) 与周期皮肤曲面 S(h,θ)。以下 S* 表示规范参考环。

$$
\widehat C_i(\beta)=C_i^*+\frac{1}{n_\theta}\sum_j[S_{ij}(\beta,0)-S^*_{ij}],\quad C_i(\beta,q)=\sum_l w_{il}^{\rm skin}A_l(\beta,q)[\widehat C_i;1].
$$

$$
F_{\beta,q}(h,\theta,r)=C_{\beta,q}(h)+r[S_{\beta,q}(h,\theta)-C_{\beta,q}(h)].
$$

$$
\Phi_{\beta,q}(\theta,h,d)=F_{\beta,q}\!\left(h,\theta,D^{-1}_{h,\theta}(d)\right).
$$

参考动脉与静脉各有 65 个中心线采样点，在登记姿态反演一次得到材料地址。随后读取相同地址，从新几何得到三维位置。把一个地址绑到模型上能保留跨体形的区域对应；个体的真实血管在这个区域内仍有位置和身份不确定性。

### 6　保存探头相对于皮肤的旋转

$$
t=\mathrm{unit}(S_h),\quad n=\pm\mathrm{unit}(S_h\times S_\theta),\quad G^S=[t,\ n\times t,\ n].
$$

$$
\rho=(G^S_{\rm ref})^\top R_{\rm cal},\quad R_{\rm probe}^{\rm base}=R_{\rm atlas}^{\rm base}G^S_{\beta,q}\rho,\quad p_{\rm probe}^{\rm base}=T_{\rm atlas}^{\rm base}S_{\beta,q}.
$$

法向取朝向核心的符号。ρ 是同一材料点局部标架内保存的手法角度。右腿扫描用纵向站点 s 和模板横向毫米 x 表达请求，换体形后才换算目标位姿。令 λ₀ 为参考表面每弧度在横轴方向上的米数。

$$
h=h_0(s),\qquad \theta=\theta_0(s)+10^{-3}x/\lambda_0(s),\qquad d=0.
$$

> 样条是已有四面体图册的可选连续实现。两种图册在归档的 516 对模拟执行中逐例完成判断一致。运动学习可复用相对请求，运动学检查使用当前体形与机器人坐标。

符号 G 的实现层次值得留意。皮肤的探头旋转逐点由皮肤几何恢复。合成血管位移实验采用该扫描带标定的共用横向与深度方向，并未在每个真实器官内部测出完整的局部三维标架。

[体形运输](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_bodyshape_atlas_transfer_20261113/transport.py) · [探头标架与登记](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_geometry.py) · [样条验证](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/SPLINE_ATLAS.md)

## 血管的位置为什么是一个概率分布

### 7　已知几何、可重复响应与短时漂移

$$
c_t(\xi)=\Phi_{\beta,q_t}(\xi)+G_{\beta,q_t}(\xi)\,[\,w(\xi)+g(\xi,u_t)+z_t(\xi)\,].
$$

| 符号 | 物理含义 | 在本例中怎样获得 |
| --- | --- | --- |
| Φ、G | 名义三维位置与局部修正方向 | 由材料地址、形状与姿态计算。腿扫实现以扫描带标定的横向、深度方向施加位移。 |
| w | 参考条件下持续存在的偏移 | 例如同一位置总比模板深。稳定接触或配准偏差也可能进入 w。 |
| g | 能随已知输入重复出现的额外滑移 | u 是已知膝角。参考角下将响应设为零，拟合角度变化的响应系数。 |
| zₜ | 已有输入仍解释不了的时变偏移 | 用有时间相关性的状态预测，可包含短时滑移或接触变化。 |

$$
g(\xi,u_{\rm ref})=0,\quad g(\xi,u)\simeq J(\xi)(u-u_{\rm ref}),\quad J(\xi)=\left.\partial_u g(\xi,u)\right|_{u_{\rm ref}}.
$$

$$
g_{\rm knee}(s,u)=E(s)(u-u_{\rm ref})[\,k_{\rm lat},\,k_{\rm dep}\,]^\top,\quad \widehat J_{\rm dep}(s)=E(s)\,\mathbb{E}_{b_t}[k_{\rm dep}].
$$

两轴版用同一个预定的膝旁空间窗 E(s)，按超声更新个人斜率 k，单位为 mm/°。一般输入有 p 维时，三维位移的 J 可以是 3×p 矩阵。一个膝角对应一个多方向的向量斜率。

$$
z_{t+\Delta}=\alpha z_t+\eta_t,\quad\alpha=e^{-\Delta/\tau},\quad\eta_t\sim{\cal N}(0,(1-\alpha^2)\Sigma_z).
$$

### 8　哪些数据可以把 w、g 和 z 分开

对同一材料点、相似接触条件下的已关联观测，在局部方向上扣除图册预测得到残差 y。两次观测作差会消去共同 w，再比较随角度重复的变化。不同地点各扫一个角度容易把空间差异误认成角度响应。

$$
y_2-y_1=E(s)k(u_2-u_1)+(z_2-z_1)+(\epsilon_2-\epsilon_1).
$$

独立等方差噪声、每站另估截距时，斜率信息正比于 Σₛ E(s)² Σⱼ(uₛⱼ−ūₛ)²。真实序列需按漂移相关性加权。单角仍能更新当前总偏移；不同角度复扫同站并回参考角，才能逐渐学出可重复响应。

参考角单次或重复扫描，若斜率先验与偏移独立，关于 k 的直接似然是平的。多角度实验使用 5°、15°、25° 的同一位置复扫；对称角和重返参考角有助于区分持续漂移与可重复运动。

$$
\Sigma_{\rm pred}=L_\tau P_t L_\tau^\top+Q_\tau,\qquad L_\tau=[I,\ E(s)(u_{t+\tau}-u_{\rm ref})I,\ e^{-\tau/\tau_z}I].
$$

在固定局部线性模型内，Pₜ 是 (w,k,zₜ) 的联合后验协方差，Q 为新的过程噪声。计算位置预测区间要保留各项后验相关性。

> 图册更新滞后、按压与角度若同步，数据可能只辨识其合成效应。未来有心动或呼吸相位时，可将相对于参考相位的正弦、余弦差作为输入，重新检验辨识性。当前实证限于腿部合成响应。

[腿部参数驱动模型](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/KNEE_RESPONSE.md) · [实际位姿投影](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py)

## 用每一帧更新后验，并重审早期图像

### 9　把概率假设投影到实际探头切面

生成器先固定一例的动静脉、持续偏移和运动，再生成每帧可见性与测量噪声。核对和更新时使用标定探头在采集时刻的实际位姿 Tₜ。每个假设预测血管与这一切面的交点及其二维位置，和当前匿名候选集 Yₜ 比较。

$$
n_t^\top(c(s^*)-p_t)=0,\qquad \mu_t=\Pi_t(c(s^*)-p_t),\qquad Y_t=\{y_{t1},\ldots,y_{tn}\}.
$$

动膝实现的一动脉一静脉可见性会持续数帧；被检测时质心噪声横向约 1 mm、深度约 0.68 mm。固定膝角复查环境另含竞争结构、声影、配准偏差与失检。把这些误差写入观测过程，才能评估错误确认。

$$
b_t^-(X)=\int p(X\mid X^\prime,u_t)\,b_{t-1}(X^\prime)\,dX^\prime,\quad L_t(X)=\sum_{A_t}p(Y_t,A_t\mid X,T_t,u_t).
$$

$$
b_t(X)=\frac{L_t(X)b_t^-(X)}{\int L_t(X^\prime)b_t^-(X^\prime)dX^\prime},\qquad P(c_t(\xi)\in U\mid{\cal H}_t)=\int {\bf 1}_U(c_t(\xi;X))\,b_t(dX).
$$

X 在动膝实验包含 w、k、zₜ 和可见性 Vₜ。在只有双管、无额外杂波的条件模型中，若只看到一点 y，其似然为 ℓ动(y)m静＋ℓ静(y)m动；m 是相应管未检出的概率。看到两点则对两种指派求和。复查实验改用 40 个离散身份、几何及声影组合，选动作时只看公开后验。

早期两管先验举例：持续横向偏移有 7 个格点（−6 至 +6 mm），深度有 3 个（−2、0、2 mm），用尺度 3.5 和 1.5 mm 的离散正态质量初始化。再配动、静脉各自清晰或失检的四种状态。动膝模型另扩展角度斜率和漂移；最新固定膝角 B 使用独立的 40 假设模型。

### 10　拍摄时暂存，身份可以稍后确认

$$
r_{fj}(H)=P(I_{fj}=1\mid H,Y_f,T_f),\qquad p_{fj,t}=\sum_H b_t(H)\,r_{fj}(H).
$$

I 表示候选的目标身份。在固定结构假设 H 使历史帧关联条件充分的复查模型里，每个候选保存其在各假设下的关联概率。后续证据改变 bₜ 后，对原帧重新加权。达到 0.95 才确认，需要时可撤销。原帧只进入滤波一次，原采集时刻、位姿与后来的确认时刻分别记录。若关联还依赖过去的动态状态，则需归档相应的联合分布或做平滑。

早期拍到两根管腔 → 保存两者的候选 → 在上游取得区分线索 → 用新后验重审原帧。

> 动膝轻漂移实验中，24 个非零深度响应的斜率方向全部判对，RMSE 为 0.0387 mm/°；名义 90% 斜率区间覆盖 20/24。加入 g 将 0.5 s 深度预测误差从 0.772 降到 0.455 mm。另跑 36 例完整闭环，两组均完成 31/36；正确帧命中率差 +0.81 个百分点，95% 区间为 −0.34 至 +1.96。

深度两轴结果据本地 knee-depth-mild-drift/REPORT.md 与 knee-depth-closed-loop/REPORT.md 复核。下面的仓库链接提供已归档的流式扫描与证据机制。

[流式估计与完整闭环源码](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py) · [同观测流的证据消融](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/PROBE_EVIDENCE_REPLAY.md)

## ProDMP 学什么，换段时怎样接得上

### 11　第一批示范来自因果视觉伺服

沿既定纵向站点 s 运行扫描，每 0.1 s 读入一次匿名超声候选。若某候选的动脉概率达到 0.95，取它在图像横轴上的误差 e；否则该帧令 e 为零。参考扫描速度叠加纠偏，并以 0.02 s 步长作限速、限加速度积分，产生实际示范 x、v。

$$
v_{\rm cmd}=\mathrm{clip}(v_{\rm nominal}+5e,-18,18),\qquad v_{i+1}=v_i+\mathrm{clip}(v_{\rm cmd}-v_i,-2,2).
$$

$$
x_{i+1}=x_i+0.02(v_i+v_{i+1})/2.
$$

横向单位是模板毫米。128 次模拟扫描，每次 4 个一秒段，共 512 个段样本。先对每段求保端点的形状权重，再用 [1，末端横移，起始横向速度] 做岭回归预测形状。早期另一个恢复版本按推进、等待、侧移、短行、回访拟合技能。

### 12　积分基函数和带约束的最小二乘

令 σ=t/T 为段内相位，T 为时长，x₀、v₀ 是交接时的实测或模拟已执行状态，xₑ、vₑ 是要求的段末状态。令 λ=5，以临界阻尼系统的卷积得到四个基函数 Bⱼ。

$$
B_j(\sigma)=\int_0^\sigma(\sigma-v)e^{-\lambda(\sigma-v)}\phi_j(v)\,dv,\qquad B_j(0)=B_j^\prime(0)=0.
$$

$$
h(\sigma)=x_e+e^{-\lambda\sigma}[(1+\lambda\sigma)(x_0-x_e)+\sigma T v_0],\qquad x(\sigma)=h(\sigma)+B(\sigma)\omega.
$$

$$
C=\left[B(1)^\top,\ B^\prime(1)^\top\right]^\top,\quad d_c=[x_e-h(1),\,Tv_e-h^\prime(1)]^\top,\quad \omega=C^\dagger d_c+N\alpha,\quad CN=0.
$$

$$
\alpha^*=\arg\min_\alpha\sum_i\{(x_i-h_i-B_i\omega)^2+0.35^2(Tv_i-h_i^\prime-B_i^\prime\omega)^2\}.
$$

拟合所得自由形状 Nα 位于末端约束的零空间。换目标时重算特殊解，剩余段按时长平方缩放自由形状。新参考从当前 x、v 起步，到要求的端点位置和速度。速度连续来自这四个边界条件；加速度与整条机器人轨迹还要检查。

### 13　同一条探查请求可以有多种实现

恢复库的 [Δs，Δx，bow] 分别是纵向位移、横向位移与段中横向偏折。等预算搜索版只用 [Δx，bow] 回归四个横向自由形状权重；纵向由单独轨迹给定。这里拟合的是闭环验证过的名义参考，新的图像纠偏由执行时共享的反馈实现。

$$
W=(X^\top X+\Lambda)^{-1}X^\top Y,\quad \Lambda=\mathrm{diag}(0,0.1,0.1)\quad(\mathrm{servo\ demonstrations}).
$$

> 普通 ProDMP 理论允许对权重建分布。当前比较主要调用拟合权重生成确定性参考。机器人内部的解剖后验和运动基函数的权重是两个不同的随机建模对象。

[示范生成与回归](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/run.py) · [边界约束拟合](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/primitive.py) · [原始 ProDMP 论文](https://arxiv.org/abs/2210.01531)

## 两段前瞻、段内超声反馈与恢复示范

### 14　动膝腿扫 MPC 在可执行轨迹中选第一段

$$
j(a;b)=\frac{1}{3}\sum_{i=1}^3P_b(\mathrm{centered\ hit}\mid T_i^a)-0.008\,L(a).
$$

$$
a_1^*=\arg\max_{a_1\in{\cal A}_{\rm exec}}\left[j(a_1;b)+0.65\max_{a_2\in{\cal A}_{\rm exec}(x_1^\prime)}j(a_2;b_{\rm pred})\right].
$$

每段以三个代表性曝光点评价。L 是候选的基座系 TCP 路程。名义两段在收到未来图像前评分，仅执行第一段。执行中每 0.1 s 读实际切面，更新后验并尝试修改横向目标。重规划校验余下时间、位姿、逆运动学和限速；修改不可行则保留原可行后缀。

固定膝角复查实验另有共享 MPC。它用局部覆盖、可见性、推进、路程和两帧条件信息预览评分，每段结束后用最新后验和证据记录重选。普通技能可重新条件化。自由轨迹的解析与拟合实现共享下式修正剩余段，其中 τ 由 0 走到 1，δ 是限幅的横向目标修正。

$$
x^+(\tau)=x_{\rm old}(\tau)+\delta(10\tau^3-15\tau^4+6\tau^5),\quad \dot x^+(0)=\dot x_{\rm old}(0).
$$

动膝代码还把图像毫米换回模板毫米。若末端预测横向偏差为 μₑ，当前图像创新为 y−μₜ，Jₓ 是末端图像误差对模板横移的负导数，则目标按下式更新，Jₓ 不可靠时仍沿用原可行轨迹。

$$
x_e^+=x_{\rm nom}+\mathrm{clip}((\mu_e+0.28(y-\mu_t))/J_x,-7,7).
$$

### 15　用完整模拟检查筛选恢复轨迹

从真实运行的模拟初扫或失检状态出发。给同一目标请求生成纵向轨迹和横向平滑参考，在机器人与行程约束内执行。下式写出一秒段的横向参考，σ=t/T；实际初速通过第三项带入。

$$
S_5=10\sigma^3-15\sigma^4+6\sigma^5,\quad x=x_0+\Delta x\,S_5+T v_0(\sigma-6\sigma^3+8\sigma^4-3\sigma^5)+64b\,\sigma^3(1-\sigma)^3.
$$

b 即 bow，在中点贡献恰为 b。结构化随机从三个锚点附近按固定分布采样。CEM 用同样锚点和第一批样本，后续按每个锚点的精英请求重新估计均值与尺度。每个提议接回共同 MPC，以完整检查结果和执行代价评分。

$$
\mu^+=0.3\mu+0.7\,\bar a_{\rm elite},\qquad \sigma^+=0.3\sigma+0.7(\mathrm{sd}_{\rm elite}+0.25).
$$

每场两种方法各付 192 次闭环推演，另做独立复核。首批 48 次内，64/64 个开场失检场景都选中同一条首选；61 条首选通过另外 12 次扰动复验。两组用这 61 条名义轨迹拟合，权重逐字节相同。新病例两组均完成 438/512，最终各误认 1 次；同一前瞻选择器仅用原动作库完成 443/512。

> 较早的流式对照把“段内反馈”与“学得的形状”分开。固定形状后开放反馈，完成由 174 增至 214/256。固定反馈后仅换形状，完成由 213 增至 214/256。现有结果给反馈的增益比给轨迹形状的增益更清楚。

[两段规划与观测反馈](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_streaming_prodmp_two_vessel_20261113/atlas_scan.py) · [轨迹搜索、预算与复验](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/SEARCH_EFFICIENCY.md) · [恢复执行器](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/adaptive_controller.py)

## B 如何调度同一库中的动作

### 16　每个候选都要和当前血管后验一起评分

令 ℋ 表示截至选择时刻可用的全部历史。B 可用当前探头姿态与速度、实际成像范围、已采图和缺图的区间、各身份概率与剩余资源。对候选路径再编码它相对于这些历史信息的位置。

$$
e_{\rm skin}(\xi)=(h,\sin\theta,\cos\theta),\qquad e_a=\mathrm{enc}(\mathcal{H},\ \mathrm{path}(a),\ \mathrm{window}(a),\ b,\ \mathrm{ledger}).
$$

包括当前探头及候选上四个采样点的皮肤地址、名义声窗左右边界和图像深度界限。保留 40 个假设权重，其中最高的八个附有目标相对位置与预计可见性。另编码相对声窗的余量、覆盖、候选与本次 MPC 选择的差别。最新训练合计 457 个特征。

$$
\widehat P_{\rm visible}(a)=\sum_H b(H)\int P_{\rm detect}(a,H,\delta)\,p_{\rm reg}(\delta)\,d\delta.
$$

δ 是校准数据估计的配准误差，实际实现作数值加权。相邻材料地址提供粗位置关系，真正的识别置信还取决于声窗、遮挡与图像。独立视点试验的 Brier 分数由 0.1727 降到 0.1249；相对原编码的完成率仅增 0.68 个百分点，区间跨零。

### 17　回放每个可行动作，监督最终结局

$$
O=(S,W_{\rm final},W_{\rm interim},L,\Theta,T),\qquad \Delta_a(\mathcal{H})=\mathbb{E}[\,O(a\to\pi_0)-O(\pi_0)\mid\mathcal{H}\,].
$$

$$
\widehat\Delta_a=\frac{1}{M}\sum_{m=1}^M[\,O_m(a\to\pi_0)-O_m(\pi_0)\,].
$$

π₀ 为共享的完整重规划 MPC。对每个训练场景，各动作从同一初态、同一隐藏世界和相同外源扰动出发。动作改变探头位姿，各自收到相应的新图像。六次配对扰动构成最新训练均值。用全部成功与失败的候选，为六项结果各拟合一个梯度提升树回归器。

原训练集有 320 个状态。在共同 MPC 前缀上采集连续丢失 4–7 帧、恢复 1–3 帧后的状态，或实际回到旧区间的状态。两组各补 72 个状态，一组专选身份歧义，另一组按失检后恢复和回访等额取样，保留其自然的清晰与含糊比例。后者在主歧义测试集完成最多，前者在清晰和首次开场对照中更强。

$$
a_B=\arg\max^{\rm lex}_{a\in{\cal A}_{\rm exec}}(\widehat\Delta_{a,S},\ c^\top\widehat\Delta_a,\ {\bf 1}_{a=a_0}),\quad c=(1,-1.25,0,-0.002,0,-0.02).
$$

> 先按预测正确完成排序，完全同分再按效用和原 MPC 动作决胜。共同物理预算仍有效。B 只决定一次复查段；段内继续视觉反馈，随后继续原 MPC。可确认、撤销、停止均依赖共享证据流程。

[训练与留出协议](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/IDENTITY_RECOVERY_DATA_BENCHMARK.md) · [位置和声窗消融](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/WINDOW_ENCODING_BENCHMARK.md) · [实际选动作顺序](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/benchmark_window_encoding.py)

## 现有实验能把收益归给谁

### 18　最新身份歧义试验

96 个新模拟场景，每例 8 条共享扰动。下表比较同一动作库、段内反馈、证据保留和 MPC 尾段。B 为补充 72 个普通恢复／回访状态训练的版本。两组均使用 67 mm 和 8 s 的公共硬上限。

| 结果 | 强 MPC | 学习 B | 解释 |
| --- | --- | --- | --- |
| 正确完成 | 461/768 | 558/768 | +12.63 个百分点，病例配对 95% 区间 [5.47, 20.31]。 |
| 连接歧义 | 177/384 | 251/384 | +19.27 个百分点，[8.85, 30.73]。 |
| 无分叉相邻血管 | 284/384 | 307/384 | +5.99 个百分点，[−3.39, 16.15]。 |
| 终末／中途误认 | 19 / 61 | 26 / 99 | 均为出现相应错误的运行数。 |
| 实际均值 | 41.36 mm / 6.81 s | 48.73 mm / 7.12 s | 累计转角 7.77° 与 9.30°，决策 CPU 85 与 290 ms。 |
| 查询区新采后正确确认 | 481/768 | 607/768 | 检查点后在任意部位物理拍到目标，两组都是 768/768。 |

这表明恢复动作可以增加“对指定区间有用且认对”的证据。最终误认增量为 0.91 个百分点，95% 区间约 [−0.13, 2.73]。当前证据支持给定预算上限下的完成收益，实际路程更长，决策耗时未推进仿真时钟。另一开场协议中，经校准的 B 和开发集选定的分任务固定动作同为 950/1024。

### 19　两条可以追到观测和后续动作的证据

在连接模拟的 22 mm 之后，交换两种身份可得到相同可见性、几乎相同的无标签管腔位置。由此处开始的区域检查中，冻结 B 在 192/192 次都拍到所需区间并到终点，正确完成为零。重训 B 完成的 31 次，全部实际回访了 18.4 mm 上游。

另一早期配对研究固定首段实际轨迹，屏蔽这段的图像后，连接组的 B 优势由 11 次缩至 5 次。说明图像会贡献于收益；轨迹本身由反馈产生，出口、后续动作和随后新图像仍可贡献。

> 动膝 g 的 31/36、随机与 CEM 的 438/512、复查 B 的 558/768 是不同任务的结果。本次审阅的归档尚缺同一移动、变形病例上“连续图册＋个体角度响应＋完整 B”的联合验证，也缺真实 B-mode 闭环的同等证据。

[主结果和原始记录](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/results/identity_recovery_data_v1/summary.json) · [中途开场与上游回访](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/ARBITRARY_START_BENCHMARK.md) · [同运动、改证据的回放](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/PROBE_EVIDENCE_REPLAY.md)

## 特权模拟器怎样造出更值得学的数据

### 20　先找能区分解释、又来得及续扫的视点

把一个“视点” v 具体化为探头可到达的一小段运动及其声窗。给定同一公开历史和两个需要不同续行的解释，先验为 p₁、p₂。分别预测这段运动会产生的整组匿名观测分布 ℓ₁、ℓ₂，包含空帧、配准误差和持续声影。最优身份判断在观察后的期望错误为

$$
e(v)=\int \min[p_1\ell_1(y\mid v),p_2\ell_2(y\mid v)]\,dy,\qquad e(v)=\frac{1-\mathrm{TV}(\ell_1,\ell_2)}{2}\quad(p_1=p_2).
$$

在两种解释的差异区枚举或优化少量声窗，再从视点和可续扫出口反求能到达的起点、端点和时长。每条轨迹仍须正向运行共享反馈与完整 MPC，付清移动、回访和时间代价。能减少身份错误的视点，也可能因为离所需区间太远而降低完成率。

若两个解释对同一个可测特征给出同协方差高斯分布、先验各半，Bayes 误分率可以直接从两预测的马氏距离算出。这里 Φᴺ 是标准正态分布函数。由此可确定性地筛选可辨视角；遇到无标签点集、混合身份与失检，须回到整帧似然。

$$
\delta^2(v)=(\mu_1-\mu_2)^\top\Sigma^{-1}(\mu_1-\mu_2),\qquad e(v)=\Phi_{\rm N}(-\delta(v)/2).
$$

### 21　把“真值能赢”换算成“只看超声也能选对”

$$
Q(\mathcal{H},Z,a)=\mathbb{E}_\epsilon[U(a\to\pi_0)\mid\mathcal{H},Z],\quad Q_{\rm pub}(\mathcal{H},a)=\mathbb{E}_{Z\mid\mathcal{H}}Q(\mathcal{H},Z,a).
$$

$$
\mathbb{E}_{Z\mid\mathcal{H}}\max_a Q(\mathcal{H},Z,a)\ \geq\ \max_a Q_{\rm pub}(\mathcal{H},a)\ \geq\ Q_{\rm pub}(\mathcal{H},a_{\rm oldB}).
$$

以上用同一标量结局、同一可行动作与后续控制器。左边比中间多出的部分需要隐藏信息；中间比右边多出的部分才是当前 B 可通过选动作追回的余量。用真实世界结局监督公开历史上的动作价值，在训练分布匹配时成立。只模仿上帝视角的赢家会抹掉取证的价值。

例：两种身份等可能。左查／右查各只在一种身份下成功，知道真值的教师可以次次成功。机器人若照教师的动作频率选，只能成功一半。假设同预算下先取一份判别证据、再据观测选续行，两种身份均有 0.8 概率完成。合法的动作价值标签应当选第三种。

### 22　怎样优先标注难例

从真实初扫引出的决策历史出发，抽取多个与历史相容的隐藏世界。每个世界内各动作配同一扰动，再换世界和扰动确认。重点补有公开可预测差别、且稳定超过旧 B 的状态；同时保留普通、清晰、晚起步及无收益状态，按目标分布独立测试。

$$
\mathrm{Var}(\widehat\Delta\mid\mathcal{H})=\frac{\mathrm{Var}_{Z\mid\mathcal{H}}(\mathbb{E}_\epsilon[D\mid\mathcal{H},Z])}{L}+\frac{\mathbb{E}_{Z\mid\mathcal{H}}[\mathrm{Var}_\epsilon(D\mid\mathcal{H},Z)]}{LM}.
$$

上式中 D 为一次配对结局差，L 个独立后验世界，每个世界 M 条条件独立的扰动。只给同一解剖场景补噪声，降不了第一项。项目已有后验 rollout、分层估值和两时点反馈。新建议把视点的几何来源、任务余量和筛选参照明确地接入造数流程。

[按任务决策区分假设](https://proceedings.mlr.press/v33/javdani14.html) · [信息采集的特权模仿](https://arxiv.org/abs/1611.04180) · [部分可观测教师错位](https://proceedings.mlr.press/v139/warrington21a.html)

## 可以预测的主动探查收益

### 23　身份信息怎样转成正确续扫

设在可访问的探查段 a 之后收到观测 y、到达状态 x′、剩余资源为 r′。令 F(x′,r′) 为这个实际出口扫描机器人仍能完成的续行集合。用同一个物理模型和验收定义，给每条后续程序 v 评价身份正确的新采图与续扫概率。

$$
Q_{\rm cont}(\mathcal{H},a,y,v)=P(S=1\mid\mathcal{H},a,y,x^\prime,r^\prime,v).
$$

$$
Q_{\rm probe}(\mathcal{H},a)=\mathbb{E}_{y,x^\prime,r^\prime}\!\left[\max_{v\in{\cal F}(x^\prime,r^\prime)} Q_{\rm cont}(\mathcal{H},a,y,v)\right].
$$

对当前已有 B，真实运行在出口处采用共同 MPC；用它的续行选择代替上式的最大化即可得到 B 所要拟合的成功率。上式是检验条件策略价值的设计标尺。无可行续行的概率要计入失败，或进入大家都能用、照常扣费的恢复。

若两种观测都引向同一最优可行续行，此处因观测而改选的收益为零。即使不能改选，观察也可能直接补齐任务要求的新图。若某一观测要求改选且出口来得及执行，额外取证才可能通过“改选”提高完成。评估时记录是哪一条机制起作用。

若想证明“根据当前证据挑动作”本身有用，将公开状态上的最佳选择与开发集上统一选定的固定请求作比较。对不可行请求规定同一回退。令分数体现一致的资源要求；如下差值为零时，提高学生容量也不会创造条件选址收益。

$$
G_{\rm adapt}=\mathbb{E}_{\mathcal{H}}\max_a Q_{\rm pub}(\mathcal{H},a)-\max_a\mathbb{E}_{\mathcal{H}}Q_{\rm pub}(\mathcal{H},a)\ \geq\ 0.
$$

### 24　用旧 B 的逐动作矩阵确定下一批数据

| 36 个状态 × 6 条扰动 | 旧 B | 每状态事后挑一条 | 每次事后看未来再挑 |
| --- | --- | --- | --- |
| 定向失检后恢复 | 135/216 | 139/216 | 140/216 |
| 定向回访 | 180/216 | 198/216 | 198/216 |

这是对已保存矩阵的重新计数。只按正确完成比较，同一可行动作、同一 MPC 尾段。失检里 81 次旧 B 失败中的 76 次，全部已测动作都失败。回访还有 18 个结局可以改变。逐次事后上限允许预知未来，与可部署的公开收益仍有距离。

另一项已完成的“逆向找声窗”在两个必扫站点增补中间倾角。成本筛选后，分支和无分叉的增益仅 0.91 和 0.39 个百分点，两组区间均跨零。下一项应在确有上游身份线索的回访状态，检验反求位置与观测分支是否在相同资源下产生新证据。

> 更好的训练数据必须回答两件事。动作库中究竟有没有可验证的替代动作？以及机器人在执行前得到的公开证据，能不能判断哪一个值得选？两种情况的干预分别是改变可行探查，或改善决策监督。

[逐动作配对原始记录](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/results/identity_recovery_data_v1/targeted_train.jsonl.gz) · [原策略的逐例审计](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_search_embedding_a_20261117/results/identity_recovery_data_v1/previous_b_label_audit.json) · [早期逆向倾角负结果](https://github.com/NingMeng0903/RealUS_playground/blob/68b60ed661b06e3cdc35fd81593f3c2360a846c2/NMI_idea/experiments/ab_inverse_acquisition_b_20261113/README.md)

## 怎样判断应不应该换成 RL 或扩散

### 25　一次介入可以直接监督动作价值

$$
Q^*(\mathcal{H})=\max_{a\in{\cal A}}Q_{\rm pub}(\mathcal{H},a),\quad \|\widehat Q-Q_{\rm pub}\|_\infty\leq\varepsilon\ \Longrightarrow\ Q^*-Q_{\rm pub}(\mathcal{H},\hat a)\leq2\varepsilon.
$$

取估值最大的可行动作，分别控制最优动作和所选动作的估值误差即可得到该界。它假设同一可行集、准确标量目标和统一误差界。现有六个回归器未实测取得这样的界。相对于当前一次决策和固定 MPC 尾段，RL 没有额外的动作价值信息来自算法名称本身。

### 26　什么情况值得做新策略

| 候选路线 | 出现何种现象才值得投入 | 同资源下怎样检验 |
| --- | --- | --- |
| 监督 B | 现有动作在不同公开状态下各有胜负。 | 配对评估每个动作的完整后果，比较旧 B、固定动作、强 MPC 与公开后验规划。 |
| 多步规划／RL | 一次探查本身无明显收益，但所获证据决定第二步；B 的决策改变以后访问的状态。 | 先开放同样的两个决策点，比固定首段、提前规划观测分支、循环 RL。允许的停止和恢复一致。 |
| 条件扩散提议 | 公开状态和任务相同，却存在走法显著不同且都可行的高维动作；现有库遗漏可行解。 | 在同一条件下抽相同数量的请求或权重，统一 ProDMP、约束与 B；比结构化随机、几何提议和扩散的发现率与执行结果。 |

$$
Q(b,a)=\mathbb{E}[\,r+V(b^+)\mid b,a\,],\qquad \sum_a\pi(a\mid\mathcal{H})Q_{\rm pub}(\mathcal{H},a)\leq\max_a Q_{\rm pub}(\mathcal{H},a).
$$

第一式适用于允许 B 多次决策的有限预算任务，信念状态应包含证据账本、机器人状态及剩余资源。第二式说明在同一个已穷举的候选集上，抽样多个动作模式本身无法超越最佳公开动作。扩散的机会主要在扩大可执行动作集合或降低在线搜索成本。

### 27　最小的决定性检验

冻结任务、案例分层、动作权限和公开可用观测。等推演预算比较结构化随机、CEM、几何反求三种候选来源。分别用最佳固定流程、可更新的强 MPC、公开前瞻和学习 B 选择；先固定运动实现，随后固定请求替换解析与 ProDMP，保持反馈和共同尾段一致。

每例保存动作前的竞争解释及预测、执行后实际新增的证据、哪张图得到确认或被撤销、是否新采并完成指定续扫。按独立病例配对，报告终末与中途误认、实际运动和时间。用移动体模真实 B-mode 独立标注身份，另检验运动状态、视觉伺服与 B 的整合。

> 建议保留贝叶斯＋ProDMP＋监督 B。论文当前最可检验的主张是“在同一公开历史与可行资源下，预测哪次观察能破除任务相关的身份歧义，并兑现为正确续扫”。这项判断来自已有归档及原始论文，几何反求教师、RL 与扩散在这个系统上的新增收益均待测。

[AggreVaTe 与下游代价](https://arxiv.org/abs/1406.5979) · [循环 RL 对照](https://proceedings.mlr.press/v162/ni22a.html) · [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) · [审查快照](https://github.com/NingMeng0903/RealUS_playground/tree/68b60ed661b06e3cdc35fd81593f3c2360a846c2)

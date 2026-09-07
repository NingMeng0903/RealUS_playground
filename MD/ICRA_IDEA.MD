如果我的思路是吧力传感器的力作为任务层级送入qp然后设计能量约束qp共享 6 维罐；账本 E_{k+1} = E_k + Δt·W_Pᵀ V_{P,m}（用 T_n 延迟队列对齐）；先 shadow 记录 P_port、P_mis=(V_a−V_m)ᵀW 与假设约束的修正量。

已承诺能量 reserve：RM75 命令 28–40 ms 后才起作用，先用延迟队列 + T_n 算 ΔE_committed^UB（未来若干拍可能从端口取走的最大能量），只有 E_k − E_min ≥ ΔE_committed^UB 才允许继续压入；同理首触/过力用 F_future^UB = F_k + K_e·Δx_committed^UB（这是 10_tn、delay queue、stop_dx_ub 真正发挥价值的地方）。
确认符号后在最终 QPIK 加线性硬约束 W_Pᵀ J_P q̇ ≥ (E_min−E_k+ΔE_committed^UB)/Δt + ‖W_P‖ē_v，以及接触不变集约束 F_min ≤ F ≤ F_max（由 P2 的 h_±/K̂e 置信区间转成速度界，F_max 无 slack）；E_max≈F_safe·Δx_safe；删除 D+α 与 post-QP clip；F* 由罐单独预算 F*·v（到达目标 v=0 后不耗能，建立接触与跟随移动组织时耗能），罐见底降级而不是"不让"。Secchi 的 v≈v_d 假设在本平台 1–3 Hz 不成立，因此账本用实际速度、约束带速度误差与延迟队列鲁棒项。
QPIK 层级：硬安全 > 能量/接触不变集 > 法向力/hover 平移 （力也可以slack）> 接触点旋转 > 轨迹姿态 > nullspace/轨道回中（独立 slack）。我的想法是，之前按照ICRA_27的数据，说明在保持力去接触体表的话，稳定的kd最多只能527n/m不弹跳,所以力不能要求恒定，而是力在一个范围内稳定，而且力的修正幅度要能匹配硬件的能力，不行的话规划出来力的闭环也没有意义，所以应该吧力作为而约束放到QP,然后系统能量也是放大QP,可是在QP新建ENERGYTANK有意义吗。在QP里面产生加上ENERGYTANK的目的是什么？能证无源性吗，还是只是做一个调度的作用？能量/接触不变集 ，接触应该在外环直接作用到VCMD修正上比如结合图像的有显示区域知道接触面，然后结合力传感器数据只当往哪里动可以让图像全接触这个优化变量应该是探头tcp在垂直图像的旋转，这个是应该修正VCMD的（利用外环位置-速度闭环），然后VCMD传入QP传入，还是接触优化项直接传入？然后最重要的反而就是力进QP,现在的代码的修正建议，力本来就是进QP吗？怎么可以松弛力换取能追而而不是为了精确力超级大的反馈，反而带来滞后和不稳定？打死你力还是要证明通道有界限，比如目标是4N,那应该比如一定落到正负2n,开头撞一下不算，这个可有特殊方法优化。然还在这个范围里面qp可以松弛但有严格上下边界，还有类“力”jerk“的限制让可以松弛力换取能追而而不是为了精确力超级大的反馈。还有一个设计是我目前不是servoing,我不是时间系统，而是轨迹跟踪，也就是轨迹可以对时间归一化，成一个路径s 0-1,然后就而已定义一个联系到速度的扫描的速度缩放系数，这个系数也放到qp做松弛，也就是可以自动调速尽量保持稳定。还。还有最重要i的，我的cdyob/tdpa/popc这些论文都有给你，我应怎么设计力的qp.什么控制率能不弹跳，力又尽量柔顺，，手推追力快手感也好？还有就是hqp的任务层级怎么调度让总体的力控制最稳定，而且能跟上我识别出来的力特性，有一些叫conditonal filter或者设计l ya p lov函数让整个hqp稳定？仔细再找一波论文，结合
NingMeng0903/RealUS_playgroundmain的最新版本代码，前面的修改意见也是基于这个版本的2d9848cae211fab46dc38cb0bca0418b5bfc8031，还有github
NingMeng0903/ICRA_2027的辨识数据，仔细去在看一下论文和找需要的新的论文，分析我的修改意见是不是可行，我要一个icra 好论文 oral级别的idea和理论高度，甚至ijrr级别的，要鼓励多agent思考，看文章，左右讨论互相批评，然后要有审稿人agent时不时检查点评方向，最后收敛优化到可行，创新有意义的idea




# 总体判断

你的方向是**可行的，而且比“继续调 \(M,D\) 让固定导纳兼顾所有表面”高了一个理论层级**。真正有论文价值的核心不是简单地“把力送进 QP”或“在 QP 里加一个 energy tank”，而是：

$$
\boxed{
\text{在有输入延迟的黑盒速度机器人上，联合优化}
\begin{cases}
\text{患者端口能量}\\
\text{接触/力走廊}\\
\text{几何路径进度}\\
\text{图像接触与探头姿态}\\
\text{机器人运动约束}
\end{cases}}
$$

你当前 P4 已经抓住了共享六维罐、实际端口功率、已承诺能量和力不变集这几个关键点。

但还要修正四个核心概念：

1. 当前代码中**力并没有直接进入 QP**，只是先变成 \(V_{\rm force}\)，再作为普通 TCP 速度任务进入 QP。
2. 由于 RM75 有 \(28\sim40\,\mathrm{ms}\) 延迟，能量约束不能直接写成 \(W^\top J\dot q\ge b\)，而必须作用于 \(T_n\) 预测的未来实际速度。
3. “实际端口六维罐”与当前“只给 \(F^\star,v_r,\mathrm{DOB}\) 付费的 active-term tank”是两种证明架构，不能同时扣账，否则会重复计费。
4. 力不能作为当前时刻的等式任务 \(F=F^\star\)，因为 \(\dot q_k\) 不能瞬间改变 \(F_k\)；应当把**未来力、力变化率或力漏斗**变成 QP 的仿射约束。

我重新确认了仓库：`main` 当前确实是你指定的 `2d9848cae211fab46dc38cb0bca0418b5bfc8031`。

---

# 一、现在代码里的“力”到底有没有进入 QP

没有直接进入。

当前链路是：

$$
W_{\rm ext}
\xrightarrow{\text{FCE/controller}}
V_{\rm force}
$$

然后在 `HybridTffOuter.sample()` 中：

$$
V^\star
=
S V_{\rm position}
+
(I-S)V_{\rm force},
$$

代码对应：

```python
v_star = compose_tff(v_pos, fout.v_force, self.selection)
```

之后 QPIK 只接收到这个六维 `v_star`。

当前 QP1 是：

$$
x=
\begin{bmatrix}
\dot q\\w
\end{bmatrix},
\qquad
J_{\rm task}\dot q-w=V_{\rm cmd},
$$

并最小化六维 TCP 速度残差 \(w\)。QP2 再把 QP1 已实现的 TCP 速度锁住，优化姿态、轨道和 nullspace。它明确只消费一个已经生成好的 `v_cmd`，并不知道这个速度是由力、图像还是轨迹产生的。

所以现在虽然 QP 可以因为奇异、碰撞或速度盒而“松弛力控制产生的速度”，但它不知道：

* 当前力是多少；
* 允许的力区间是多少；
* 哪个残差是力误差；
* 继续压入会增加多少力；
* 为了保持力应该先降低轨迹速度还是牺牲姿态。

当前 `force_corridor.py` 也是在 QP 前对标量 Z 速度进行 clamp，并且代码自己明确区分：它想证明的是力集合不变性，不是无源性。

因此你要做的不是简单增加 force task weight，而是把 QP 从：

$$
\boxed{\text{追一个已经混合好的 }V_{\rm cmd}}
$$

改成：

$$
\boxed{\text{直接看见力、能量、路径进度和图像任务的交互 QP}}
$$

---

# 二、在 QP 中加入 energy tank 到底有什么意义

## 1. QP 不“产生”energy tank

energy tank 是一个位于 QP 外部、跨时间积累的动态状态：

$$
E_k\in[E_{\min},E_{\max}].
$$

QP 每一拍只根据当前 \(E_k\)，选择一个不会让未来能量跌破 \(E_{\min}\) 的最终命令。

也就是说：

$$
\text{Tank}=\text{动态储能/记账状态},
$$

$$
\text{QP}=\text{使用这个预算选择最接近名义行为的安全命令}.
$$

Secchi 的核心正是：在所有候选行为中求取“距离期望行为最近的被动实现”，而不是发现非无源后统一增加阻尼。

Benzi 则进一步把 energy-tank passivity row 和 CBF 约束放入同一个凸优化，并明确不给 passivity constraint 设置 slack，因为它代表最高优先级的鲁棒稳定条件。

---

## 2. 它什么时候只是调度，什么时候能证明无源

### 只是调度

如果你只是用 \(E_k\)：

* 改变 \(D\)；
* 改 task weight；
* 缩放 \(F^\star\)；
* 在能量低时令速度乘一个 \(\lambda\)；

但没有证明最终实际端口满足离散能量不等式，那么它只是 energy-aware scheduling。

### 可以成为无源证书

定义：

* \(W_{P,k}\)：环境作用在探头上的 wrench；
* \(V_{P,m,k}\)：同一点、同一坐标系、同一时刻的实际 twist；
* 正功率表示环境向机器人注入能量：

$$
p_k=W_{P,k}^{\top}V_{P,m,k}.
$$

以探头内压方向为正时，机器人继续压入，环境反力与速度反向，通常有 \(p_k<0\)，表示机器人对患者做功、罐能量减少。

考虑测量误差后，构造实际功率的保守下界：

$$
\underline p_k
=
\hat W_k^\top\hat V_k
-
\|\hat W_k\|\bar e_v
-
\bar e_W\bar V
-
\bar e_W\bar e_v.
$$

然后更新：

$$
E_{k+1}
=
\min
\left(
E_{\max},
E_k+\Delta t\,\underline p_k
\right).
$$

若最终控制器保证：

$$
E_k\ge E_{\min},\qquad\forall k,
$$

则：

$$
E_N-E_0
=
\sum_{k=0}^{N-1}\Delta t\,\underline p_k
\le
\sum_{k=0}^{N-1}\Delta t\,p_k,
$$

因此：

$$
\sum_{k=0}^{N-1}\Delta t\,W_{P,k}^\top V_{P,m,k}
\ge
E_{\min}-E_0.
$$

这就是离散时间端口无源性：

$$
\boxed{
\text{机器人从患者端输出的净能量，不超过初始允许能量 }E_0-E_{\min}.
}
$$

所以答案是：

> **QP 中的 energy tank 可以证明端口无源性，但只有当它约束的是最终实际端口行为，并且延迟、误差、命令队列和所有后处理都被纳入证明。**

---

## 3. 为什么 QP-tank 比 \(D+\alpha\) 更不影响手感

若暂时忽略其他约束，仅求：

$$
\min_V\frac12\|V-V_a\|^2
$$

满足：

$$
W^\top V\ge b_E,
$$

其闭式解是：

$$
V^\star
=
V_a+
\frac{
[b_E-W^\top V_a]_+
}{
\|W\|^2
}W.
$$

也就是说：

* 能量条件没有激活时：

  $$
  V^\star=V_a;
  $$

* 激活时，只沿当前 wrench 方向做最小修改；

* 不会把全部平移、旋转统一加大阻尼。

这正适合你的目标：

$$
\boxed{
\text{正常时维持低阻尼轻盈手感；只有产生危险端口功率时才干预。}
}
$$

这比当前 TDPA 长期钉住 \(\alpha_{\max}\)、全程增加阻尼合理得多。

---

# 三、你现在写的能量约束还要改成“延迟预测形式”

你当前写：

$$
W_P^\top J_P\dot q
\ge
\frac{
E_{\min}-E_k+\Delta E_{\rm committed}^{UB}
}{
\Delta t
}
+
\|W_P\|\bar e_v.
$$

这在理想的

$$
V_m=J\dot q
$$

条件下成立。但你的 RM75 明显不满足：

$$
\dot q_k
\not\rightarrow
V_{m,k}
$$

而是大约：

$$
\dot q_k
\rightarrow
V_{m,k+d},
\qquad d\approx6\sim8\text{ tick}.
$$

Secchi 和 Benzi 的原始框架都采用了低层速度控制器能较好实现期望速度的假设。 你的 \(1\sim3\,\mathrm{Hz}\) 实际速度误差已经说明不能直接使用这个假设。

---

## 1. 正确的延迟增广模型

将内部速度状态和待执行命令队列写成：

$$
\xi_k=
\begin{bmatrix}
x_{v,k}\\
u_{k-d}\\
\vdots\\
u_{k-1}
\end{bmatrix}.
$$

由 \(T_n\) 离散化后，可得到第一个受当前决策影响的未来速度：

$$
\hat V_{P,m,k+d+1|k}
=
a_k+B_kJ_P(q_k)\dot q_k.
$$

这里：

* \(a_k\) 由当前实际速度状态和队列中的旧命令决定；
* \(B_kJ_P\dot q_k\) 是当前新命令在延迟后产生的影响。

`10_tn` 的作用正是在这里：它验证空气中的

$$
T_n^0(s)
=
\frac{0.96e^{-0.028s}}{0.014s+1}
$$

能把未建模速度误差从约 \(8.2\,\mathrm{mm/s}\) 降到约 \(1.63\,\mathrm{mm/s}\)。它没有参与此前控制，也没有主动补偿；它适合用作未来实际速度预测、delay queue、committed motion 和 CDYOB 名义模型。

注意：

> 实际端口账本中的 \(W_k,V_{m,k}\) 应按照硬件时间戳插值到同一物理时刻，而不是用 \(T_n\) 把实际数据人为移到命令时刻。
> \(T_n\) 负责未来预测，不负责替代物理时间同步。

---

## 2. 已承诺能量应定义为能量“债务”

当前队列在未来 \(d\) 拍产生的速度已无法被本拍命令撤销。定义：

$$
R^E_k
=
\sup_{\theta\in\Theta}
\left[
-
\sum_{i=1}^{d}
\Delta t\,
\underline p_{k+i|k}^{\rm queue}
\right]_+.
$$

它表示即使从现在开始发出最安全的新命令，旧命令仍可能从罐中取走的最大能量。

真正可自由使用的能量是：

$$
E_k^{\rm free}
=
E_k-E_{\min}-R^E_k.
$$

于是本拍约束应写为：

$$
\boxed{
\underline p_{k+d+1|k}(\dot q_k)
\ge
-\frac{E_k^{\rm free}}{\Delta t}
}
$$

而不是直接用 \(W_k^\top J_k\dot q_k\)。

如果：

$$
E_k^{\rm free}<0,
$$

说明已经进入“新命令无法挽救”的状态，不能再靠当前 QP 修正，必须执行事先验证过的 backup/terminal policy。

你当前 `delay_safety_shield.py` 已经存在：

* delay queue；
* backup-to-terminal rollout；
* `stop_dx_ub`；
* 速度/位置误差 tube；
* terminal-domain flags。

它不需要全部推倒。最合理的做法是把它拆成：

$$
\begin{aligned}
&\texttt{DelayedPlantPredictor}\\
&\texttt{CommittedReserveEstimator}\\
&\texttt{ForceEnvelopeRowBuilder}\\
&\texttt{EnergyRowBuilder}
\end{aligned}
$$

然后把最终线性行送入 QPIK。当前 shield 自己也明确说明，只有 plant set、停止表和 terminal set 经硬件验证后才能声称证书。

最近的输入延迟 CBF 研究同样采用 predictor feedback 和延迟增广状态来获得安全集合前向不变性，并强调必须同时处理预测误差和可行性；这支持你“committed reserve”而不是普通一拍 CBF 的方向。([arXiv][1])

---

# 四、共享六维罐是对的，但要注意两件事

## 1. 为什么应该是一个共享六维罐

真正的接触功率是：

$$
p
=
F^\top v+M^\top\omega
=
W^\top V.
$$

所以从物理端口角度，一个标量共享罐是正确的，而且在固定刚体参考点之间变换时：

$$
W_P^\top V_P=W_T^\top V_T.
$$

这比六个彼此独立的轴向罐更符合实际总功率。

第一版应选择探头接触面中心的**固定刚体点 \(P_0\)** 作为端口。不要在证明版本中在线移动 \(P\)，否则当 \(P\) 相对探头移动时还会多出参考点运动项。

---

## 2. 共享罐可能产生跨任务“能量转移”

例如：

* 用户沿 X 推动产生的能量；
* 被用于资助绕 Z 的主动旋转；
* 或切向摩擦消耗全部能量，导致法向力任务突然降级。

全局无源性仍然成立，但临床行为可能不理想。

因此建议：

* 保留一个全局六维罐作为无源证书；
* 另外设置非证书性的子空间提款上限：

$$
p=p_n+p_t+p_r,
$$

$$
-\Delta t\,p_i
\le
\beta_i(E-E_{\min}),
\qquad
\sum_i\beta_i\le1.
$$

其中：

* \(p_n\)：法向接触；
* \(p_t\)：切向扫描及摩擦；
* \(p_r\)：旋转。

这样全局证明不变，但不会让某一自由度随意消耗整个罐。

---

## 3. \(E_{\max}\) 不只应该是 \(F_{\rm safe}\Delta x_{\rm safe}\)

单轴第一版可以使用：

$$
E_{\max}\approx F_{\rm safe}\Delta x_{\rm safe}.
$$

六维时至少应包含：

$$
E_{\max}
\lesssim
F_{\rm safe}\Delta x_{\rm safe}
+
M_{\rm safe}\Delta\theta_{\rm safe}.
$$

但 \(E_{\max}\) 只限制机器人可输出的总能量，不能替代瞬时力上限。很小的位移在很硬的表面上仍可能产生很大的力，因此：

$$
\boxed{
\text{energy constraint 与 force constraint 必须同时存在。}
}
$$

---

# 五、实际端口罐和 \(F^\star v\) 不能重复扣账

当前 `energy_tank.py` 是一个标量 Z 轴 active-term tank：

$$
P_{\rm active}
=
Dv_rv+u_{\rm DOB}v+F^\star v,
$$

由 \(Dv^2\) 充能，并用 \(\lambda\) 缩放主动项。

这与新方案中的：

$$
E_{k+1}=E_k+\Delta t\,W^\top V_m
$$

不是同一个罐。

必须二选一：

### 架构 A：实际患者端口罐——推荐作为论文主证书

* 账本使用实际 \(W^\top V_m\)；
* \(F^\star\)、CDYOB、图像控制只负责生成名义速度；
* 它们输出的能量最终已经体现在 \(W^\top V_m\) 中；
* **不要再单独减一次 \(F^\star v\)**。

可以记录：

$$
P_{F^\star}=F^{\star\top}V_a
$$

作为诊断，但不能再次扣账，否则重复计费。

### 架构 B：虚拟导纳储能 + active-term tank

需要构造：

$$
S_{\rm total}=H_a+E_t
$$

并严格推导：

$$
\Delta S_{\rm total}
\le
\Delta t\,W^\top V.
$$

这可以做，但在 RM75 的延迟和实际速度误差下更复杂，仍需要患者端口修正。

因此最终建议：

$$
\boxed{
\text{实际六维端口罐作为证书；active-term 功率只做分解遥测。}
}
$$

---

## 一个审稿人一定会问的问题：持续扫描摩擦从哪里取能量

建立静态 \(4\,\mathrm N\) 接触时，机器人只需要一次有限做功；达到目标后：

$$
V=0
\quad\Longrightarrow\quad
P=0.
$$

但沿皮肤持续扫描时存在摩擦，机器人会持续向患者/组织输出机械功。如果只有有限六维 tank，又不允许外部补能，那么：

$$
E_t\rightarrow E_{\min}
$$

之后路径必然减速或停止。

这是物理事实，不是控制器缺陷。不能同时要求：

* 患者单端口严格无源；
* 有限储能；
* 无限时间对抗摩擦继续扫描。

针对有限超声路径，最干净的处理是：

1. 在接触前初始化一个有限且有临床意义的任务能量预算；
2. 接触中不重置、不凭空补罐；
3. 能量不足时降低 \(\nu=\dot s\)；
4. 若要理论上允许持续供能，把路径规划器建模为第二个显式能量输入端口，证明多端口 dissipativity，而不能继续称作单端口无源。

这个问题反而使你的路径速度变量更有意义。

---

# 六、力应该怎样真正进入速度级 QP

## 1. 不能直接加 \(F=F^\star\)

在当前拍：

$$
F_k
$$

已经测定，\(\dot q_k\) 无法改变它。直接写：

$$
F_k=F^\star
$$

与优化变量无关，没有控制意义。

你真正需要的是延迟预测：

$$
\hat F_{k+d+1|k}
=
a_{F,k}+b_{F,k}^{\top}\dot q_k.
$$

这个模型可以来自：

* 在线 \(\hat K_e,\hat B_e\)；
* 局部 ARX；
* \(T_n\)+接触模型；
* 不确定区间回归。

因此所谓“force task Jacobian”实际上是：

$$
J_F
=
\frac{\partial\hat F_{k+d+1|k}}{\partial\dot q_k}
=
b_{F,k}^{\top}.
$$

这时力才真正成为 QP 可优化的任务。

已有的 force-constrained CBF 工作也正是利用接触模型，把交互力约束转换成关节速度上的 QP 约束，并对环境模型误差给出界。([arXiv][2])

---

## 2. 同一个力边界不能同时“有 slack”又“严格”

必须使用嵌套的三层力规范。

以目标：

$$
F^\star=4\,\mathrm N
$$

为例：

### 物理绝对安全边界：硬约束

$$
0\le F\le F_{\rm safe}.
$$

其中：

$$
\overline F_{k+d+1|k}(\dot q)
\le F_{\rm safe}
$$

永远无 slack。

### 接触保持边界：条件硬约束

确认进入稳定接触后：

$$
\underline F_{k+d+1|k}(\dot q)
\ge F_{\rm contact}>0.
$$

它保证不脱离。

但若患者移动速度超过机械臂能力，这个约束可能物理不可行。因此只能在已经验证的 viability kernel 内作为硬约束；超出时应进入 `LOSS/REACQUIRE`，不能假装仍可保证。

### 临床质量区间：软任务或可行域内硬约束

例如：

$$
2\,\mathrm N\le F\le6\,\mathrm N.
$$

写成：

$$
2-\delta_F^-
\le
\hat F_{k+d+1|k}
\le
6+\delta_F^+,
$$

$$
\delta_F^\pm\ge0.
$$

QP 优先最小化：

$$
(\delta_F^-)^2+(\delta_F^+)^2.
$$

最后再以较低优先级逼近中心：

$$
\min (\hat F-4)^2.
$$

所以：

$$
\boxed{
F_{\rm safe}\text{ 是硬墙，}
[2,6]\text{ 是临床走廊，}
4\text{ N 是偏好中心。}
}
$$

这样不会为了从 \(4.3\,\mathrm N\) 修到 \(4.0\,\mathrm N\)，输出一个超过硬件能力的巨大反馈速度。

---

# 七、怎样严格保证“首触之后最终落入 \(4\pm2\,\mathrm N\)”

最合适的是**收缩力漏斗**，而不是固定高增益力误差。

定义：

* \(t_{\rm touch}\)：首次触碰；
* \(t_c\)：接触确认且旧压入队列已清空；
* 保证从 \(t_c\) 开始。

构造：

$$
F_L(t)=F^\star-\rho(t),
\qquad
F_U(t)=F^\star+\rho(t),
$$

$$
\rho(t)
=
\rho_\infty+
\left(\rho_0-\rho_\infty\right)
e^{-\lambda_F(t-t_c)},
$$

其中：

$$
\rho_\infty=2\,\mathrm N,
$$

$$
\rho_0>
|F(t_c)-F^\star|.
$$

于是刚进入 capture 时，当前力一定在漏斗中；之后漏斗平滑收缩到：

$$
[2,6]\,\mathrm N.
$$

用 CBF 形式：

$$
h_L=F-F_L(t),
$$

$$
h_U=F_U(t)-F,
$$

要求：

$$
\dot h_L+\alpha_L h_L\ge0,
$$

$$
\dot h_U+\alpha_U h_U\ge0.
$$

经过延迟预测后，它们都是关于 \(\dot q\) 的仿射不等式。

只要：

* 模型误差界有效；
* QP 可行；
* 没有 slack；
* 最终发送命令没有再被修改；

就可以证明：

$$
F(t)\in[F_L(t),F_U(t)],
\qquad t\ge t_c,
$$

并最终得到：

$$
F(t)\in[2,6]\,\mathrm N.
$$

首次碰撞不能简单写成“不算”。它应由另一个约束保证：

$$
F_{\rm peak}^{UB}
=
F_{\rm touch}
+
\overline K_e
\Delta x_{\rm committed}^{UB}
+
\bar e_F
\le F_{\rm safe}.
$$

---

# 八、为什么“无源 + 接触力下界”才是真正的不弹跳证明

无源性本身不能保证不振荡，也不能保证不丢接触。

真正的无再碰撞结论应是：

1. 患者端口无源，控制器不会持续制造净能量；
2. 进入 capture 后：

   $$
   F\ge F_{\rm contact}>0
   $$

   前向不变；
3. 对无粘附的单边接触，正压缩力意味着接触间隙没有打开；
4. 因而不会发生：

   $$
   \text{接触}\rightarrow\text{脱离}\rightarrow\text{重新撞击}.
   $$

所以你的论文结论不能写成：

> “passivity guarantees no bouncing”。

应写成：

> “passivity bounds the energy exchange, while robust contact-set invariance eliminates separation and re-impact after capture.”

这是两个独立证书：

$$
\boxed{
\text{Energy certificate}
+
\text{Contact viability certificate}.
}
$$

你当前方案已经意识到这一区别。

---

# 九、“力 jerk”应该怎样设计

不要直接对原始 F/T 数据求二阶差分。那会把噪声放大得很严重。

应当限制**模型预测的力变化**。

一级 force slew：

$$
-r_F^-\Delta t
\le
\hat F_{k+d+1|k}-\hat F_{k+d|k}
\le
r_F^+\Delta t.
$$

二级 force curvature：

$$
-j_F\Delta t^2
\le
\hat F_{k+d+1|k}
-
2\hat F_{k+d|k}
+
\hat F_{k+d-1|k}
\le
j_F\Delta t^2.
$$

因为 \(\hat F^+\) 对 \(\dot q_k\) 是仿射的，这些仍可以是线性约束。

第一版建议只做：

$$
\boxed{
\text{关节/任务速度的 }a,j\text{ 限制}
+
\text{预测 force slew 限制}
}
$$

先不要上二阶 force jerk。收缩漏斗本身也已限制所要求的力变化速度。

---

# 十、把时间轨迹改成路径 \(s\in[0,1]\) 是一个很强的想法

这是你整个方案中最值得提升为论文主线的部分之一。

定义几何路径：

$$
X_d=X_d(s),
\qquad
s\in[0,1].
$$

增加 QP 决策变量：

$$
\nu_k=\dot s_k,
$$

状态更新：

$$
s_{k+1}=s_k+\Delta t\,\nu_k,
$$

并限制：

$$
0\le\nu_k\le\nu_{\max},
$$

$$
|\nu_k-\nu_{k-1}|
\le a_s\Delta t,
$$

$$
|\nu_k-2\nu_{k-1}+\nu_{k-2}|
\le j_s\Delta t^2.
$$

路径任务写为：

$$
S_t
\left[
J_P(q)\dot q
-
\mathcal T(s)\nu
+
K_xe_\perp
\right]
=
\delta_x,
$$

其中：

* \(\mathcal T(s)=\partial X_d/\partial s\) 是路径切向旋量；
* \(e_\perp\) 是横向偏离路径的误差；
* 法向自由度主要由力任务决定。

优化：

$$
\min
w_\nu(\nu-\nu_{\rm nominal})^2.
$$

这样当：

* 力接近上界；
* energy reserve 不足；
* 图像接触不好；
* 机械臂接近奇异；
* 患者正在移动；

QP 首先减小：

$$
\nu,
$$

而不是继续强行追一个固定时间参考，再产生大力反馈。

这本质上是一个**在线 path governor**。预测路径参数化早已用于在保持几何路径的同时满足运动约束，因此“在线调速”本身不新；你的新点应是把它与患者端口能量、延迟队列和接触力漏斗共同放进同一个 viability governor。([Eprints][3])

---

## 路径完成性的证明

可以给出条件式 liveness 结论：

若在某个安全区域内总存在：

$$
\nu_k\ge\underline\nu>0
$$

满足所有硬约束，则：

$$
s_k\rightarrow1
$$

并在有限时间内完成扫描。

若不存在正的可行 \(\nu\)，则控制器允许：

$$
\nu=0
$$

安全暂停，而不是违反力或能量约束。

这比“轨迹必须按时到达”更适合医学扫描。

---

# 十一、图像接触与探头 tilt 应该放在哪里

## 正确分层

### QP 外部：感知和估计

负责计算：

* 超声 confidence map；
* 接触区域左右/前后不均衡；
* 表面法向估计；
* 图像特征误差；
* 特征 Jacobian 或局部响应模型；
* 置信度和时间戳。

### 最终 QP 内部：控制任务和约束

不要先把这些全部合成为一个不可分解的 \(V_{\rm cmd}\)。

对于图像特征 \(y\)：

$$
\dot y=L_yV_P+d_y
=
L_yJ_P\dot q+d_y.
$$

可加入 CLF：

$$
e_y^\top L_yJ_P\dot q
\le
-c_y\|e_y\|^2+\delta_y.
$$

这样 QP 知道这是图像任务，可以在力、能量与图像任务冲突时按明确层级处理。

Chatelain 已经用在线 ultrasound confidence map 的矩构造图像质量控制任务，并与力和位置控制结合，因此“图像 confidence + force control”本身不能作为你的主要创新。([Technical University of Munich][4])

RUSOpt 则使用接触力和接触力学优化探头法向姿态，说明力信息确实可以帮助推断探头 normalization，但其方法是搜索型优化，不等价于你要做的实时延迟安全 QP。([arXiv][5])

---

# 十二、“略微朝皮肤内，不要朝外”不要用锐角/钝角描述

角度描述高度依赖法向定义。假设：

$$
n_{\rm in}
$$

指向皮肤内部，定义相对法向速度：

$$
v_n
=
n_{\rm in}^{\top}
\left(
V_P-V_{\rm surface}
\right).
$$

则：

* \(v_n>0\)：增加压入；
* \(v_n<0\)：卸载/离开表面。

对单边 Kelvin–Voigt 接触：

$$
F_n=K_e\delta+B_e\dot\delta,
$$

$$
\dot\delta=v_n.
$$

若 \(v_n<0\) 持续足够久，则：

$$
\delta\rightarrow0,
$$

从而发生接触丢失。之后由于 RM75 的延迟队列仍可能包含压入命令，控制器反向后又会重新接触，形成 loss–reimpact。

所以可以证明的是：

$$
\boxed{
\text{持续的 outward relative velocity 会耗尽压缩量并导致接触丢失。}
}
$$

不能普遍证明：

> “角度大于 \(90^\circ\) 加上摩擦就必然弹跳”。

摩擦会增加切向力和力矩耦合、边缘挤压以及 stick–slip，但不是上述接触丢失结论成立的必要条件。

---

## 更合适的 QP 约束

直接使用 force-dependent normal velocity cone：

$$
v_n
\ge
-\frac{
\gamma_-(F-F_{\min})
}{
\overline K_e
},
$$

$$
v_n
\le
\frac{
\gamma_+(F_{\max}-F)
}{
\overline K_e
}.
$$

含义：

* 靠近 \(F_{\min}\) 时，禁止快速向外；
* 靠近 \(F_{\max}\) 时，禁止继续压入；
* 在力区间中部，允许一定的双向柔顺。

也可以与路径速度耦合：

$$
-\kappa_{\rm out}\nu
\le
n_{\rm in}^{\top}J_P\dot q
\le
\kappa_{\rm in}\nu.
$$

它比“角度略小于 \(90^\circ\)”更容易定义、测量和证明。

探头方向本身则用：

$$
n_{\rm probe}^{\top}\hat n_{\rm surface}
\ge
\cos\theta_{\max}.
$$

这约束的是探头姿态；前一个约束的是运动方向，二者不能混成一个角度。

---

# 十三、能不能从扫描力数据粗略判断表面法向和角度趋势

可以判断趋势，但单靠腕部 F/T 不能唯一恢复精确几何法向。

在短窗口内可以拟合：

$$
\Delta F_n
\approx
g_x^\top\Delta x
+
g_\theta^\top\Delta\theta
+
B_e\Delta v_n.
$$

其中：

$$
\hat n_F
=
\frac{g_x}{\|g_x\|}
$$

表示使压缩力增加最快的**局部压缩梯度方向**。

由：

$$
g_x^\top X_d'(s)
$$

的符号可以定性判断：

* 当前路径前进是否倾向增加压缩；
* 是否倾向卸载；
* 调整 tilt 后力梯度是在变大还是变小。

但它不是严格的几何法向，因为还受到：

* 组织粘弹性；
* 患者自身运动；
* 探头接触面积变化；
* 摩擦；
* 力补偿误差；
* 延迟；

的共同影响。

建议融合三路：

$$
\hat n_s
=
\operatorname{Fuse}
\left(
n_{\rm depth},
n_{\rm force-gradient},
n_{\rm image-confidence}
\right),
$$

并输出一个法向置信锥：

$$
n_s\in\mathcal N_k.
$$

QP 对这个置信锥做鲁棒约束，而不是相信一根精确法向。近期未知表面跟随工作也在在线估计表面法向后定向调整导纳参数，说明这一估计—控制分层是合理的。([IEEE Xplore][6])

---

# 十四、CDYOB、TDPA、PO/PC 在最终系统中的角色

## 1. CDYOB：名义性能层

CDYOB 应放在安全 QP 之前：

$$
V_{\rm FCE}
\rightarrow
V_{\rm CDYOB}
\rightarrow
V_{\rm safe-QP}.
$$

其作用是：

* 尽量补偿内速度环；
* 减少 \(V_a-V_m\)；
* 提升导纳呈现；
* 改善低速摩擦和 \(1\sim3\,\mathrm{Hz}\) 的跟踪误差。

Samuel 的理想结果是：

$$
N_1=QC_n^{-1}AR_n^{-1},
$$

$$
N_2=QAP_n^{-1}T_n^{-1},
$$

并在 \(Q=1\) 的理想条件下得到：

$$
Y_{\rm CDYOB}=A.
$$



但 CDYOB：

* 不证明患者端口无源；
* 不保证力上界；
* 不撤销已在延迟队列中的运动；
* 不能替代首触保护。

而且 observer 的 \(V_i\) 必须使用**最终实际发送命令**，不能使用 QP 前的 nominal command。QP 干预较大时还要冻结或 anti-windup，否则 CDYOB 会把安全 QP 的修改误认为扰动，再反向补回来。

---

## 2. Lee 的 \(P_{\rm mis}\)：诊断和 proxy anti-windup

Lee 的关键不定项是：

$$
\dot S
\le
(V_a-V_m)^\top W.
$$

相位滞后可使机器人比代理压得更深，并在离开弹性环境时以更高速度回弹。

因此：

$$
P_{\rm mis}
=
(V_a-V_m)^\top W
$$

非常适合作为：

* shadow 指标；
* CDYOB 启停置信度；
* proxy anti-windup；
* 不确定性 tube 的在线膨胀量。

可以使用：

$$
\dot V_a
=
M_a^{-1}
\left(
W_d-W-D_aV_a
\right)
-
K_{\rm aw}(V_a-V_m).
$$

但 Lee 原方法依赖 nominal robot 和关节力矩输入，不能声称直接继承其多自由度无源定理。

---

## 3. TDPA/PO-PC：shadow 或 emergency fallback

最终系统不建议正常状态同时启用：

$$
D+\alpha
$$

和：

$$
\text{energy-QP}.
$$

否则会出现：

* 两层重复保守；
* 一层提高阻尼，另一层又投影速度；
* 手感不可解释；
* 符号错误或饱和后证书失效。

你当前 TDPA 代码本身在 \(\alpha\) 达到 clamp 时就把 `passivity_holds` 设为 false，因此它不适合继续作为主证书。

推荐：

* TDPA shadow：与实际六维账本交叉验证；
* QP predictor、timestamp 或 solver 失效时，进入保守 emergency mode；
* 正常运行由 actual-port tank-QP 主导。

De Stefano 的工作证明延迟和离散积分会导致能量不一致，并使用不同因果端口的 passivity controller 处理这些效应；它可作为你的延迟能量分析基础，但你的最终证明需要适配实际速度误差、命令队列和 QPIK。

---

## 4. Dimeas 指标：检测器，不是证书

Dimeas 的频域指标可以检测高频力振荡并触发增阻尼，优点是比位置振荡更容易检测。

在你的新系统中，更合适的用途是：

$$
I_{\rm osc}\uparrow
\Rightarrow
\begin{cases}
\nu_{\max}\downarrow\\
\bar e_v\uparrow\\
\overline K_e\uparrow\\
R_E^{UB}\uparrow\\
\text{CDYOB gain}\downarrow
\end{cases}
$$

即提高鲁棒性，而不是再次全局放大 \(M,D\)。

---

# 十五、推荐的最终 QP/HQP 数学形式

增加决策变量：

$$
z_k=
\begin{bmatrix}
\dot q_k\\
\nu_k\\
\delta_F^-\\
\delta_F^+\\
\delta_c\\
\delta_I\\
\delta_x
\end{bmatrix}.
$$

名义性能来自：

$$
V_a
=
V_{\rm FCE/CDYOB}.
$$

推荐使用如下分层结构。

## Level 0：绝对硬约束

无 slack：

$$
\dot q_{\min}\le\dot q\le\dot q_{\max},
$$

$$
a_{\min}\le
\frac{\dot q-\dot q_{k-1}}{\Delta t}
\le a_{\max},
$$

$$
j_{\min}
\le
\frac{\dot q-2\dot q_{k-1}+\dot q_{k-2}}{\Delta t^2}
\le j_{\max},
$$

以及：

* 关节/轨道边界；
* 自碰撞；
* \(F\le F_{\rm safe}\)；
* actual-port energy；
* delay queue/backup feasibility；
* 反馈过期 fail-safe。

---

## Level 1：接触可行性

条件硬约束：

$$
F\ge F_{\rm contact},
$$

$$
v_n\in[v_n^{\min}(F),v_n^{\max}(F)],
$$

$$
n_{\rm probe}^{\top}n_{\rm surface}
\ge\cos\theta_{\max}.
$$

如果患者移动导致该集合不可行，则：

* 首先令 \(\nu=0\)；
* 若仍不可行，进入 `LOSS/REACQUIRE`；
* 不能偷偷放松 \(F_{\rm safe}\)。

---

## Level 2：临床质量

最小化：

$$
\|\delta_F\|_{Q_F}^2
+
\|\delta_I\|_{Q_I}^2
+
\|J_P\dot q-V_a\|_{Q_A}^2.
$$

包括：

* 收缩力漏斗；
* image confidence；
* 左右/前后接触均匀；
* tilt；
* 接触点力矩；
* nominal low-admittance behavior。

超声 confidence、力和探头姿态已经分别被用于现有机器人超声控制，因此论文创新不能只写成这些目标的组合。([Technical University of Munich][4])

---

## Level 3：路径进度

在不恶化上层最优值的前提下：

$$
\min
w_\nu(\nu-\nu_{\rm nominal})^2
+
\|\delta_x\|_{Q_x}^2.
$$

因此：

$$
\text{力/能量受限}
\Rightarrow
\nu\downarrow,
$$

而不是力误差增益无限增大。

---

## Level 4：构型

* nullspace；
* manipulability；
* J4 comfort；
* rail mid-ranging；
* branch consistency。

---

## Hover 不应把平移严格放在旋转之前

对 6-DoF hover，平移与旋转应在同一归一化任务层：

$$
\widetilde V=
\begin{bmatrix}
v\\
\ell\omega
\end{bmatrix},
\qquad
\widetilde W=
\begin{bmatrix}
F\\
M/\ell
\end{bmatrix}.
$$

否则严格设定“平移优先于旋转”，用户同时推和扭时会再次感觉：

* 旋转被平移吃掉；
* 扭动需要更大力矩；
* 机械臂主动扭手。

---

# 十六、HQP 自身不能自动证明整个控制器稳定

Escande 的 HQP 可以保证严格优先级，并处理不同层级的等式与不等式；但其稳定性结论建立在每个任务控制律本身稳定、Jacobian 满足秩条件等假设上。

所以不能写：

> “因为使用 HQP，所以整个力控制系统稳定。”

正确证明分工应为：

| 性质         | 数学工具                                            |
| ---------- | ----------------------------------------------- |
| 患者端口能量有界   | dissipativity / energy tank                     |
| 力不超过上限     | robust CBF                                      |
| 接触不丢失      | contact-set CBF / viability                     |
| 力进入目标区间    | CLF 或收缩 funnel                                  |
| 图像/tilt 收敛 | CLF                                             |
| 路径不偏离      | transverse CLF                                  |
| 多任务不互相破坏   | HQP                                             |
| 有延迟时仍可实施   | predictor + committed reserve + terminal backup |

可定义混合状态机：

$$
\text{AIR}
\rightarrow
\text{TOUCH}
\rightarrow
\text{CAPTURE}
\rightarrow
\text{SCAN}
\rightarrow
\text{LOSS}.
$$

所有模式共享同一个 \(E_k\)，并要求切换时：

$$
E^+\le E^-,
$$

避免状态切换凭空补充能量。所谓 conditional filter 最好落实为：

* 有滞环的约束激活；
* minimum-intervention QP；
* predictor confidence；
* dwell time；
* 公共储能函数；

而不是动态切换一套完全不同的控制器。

近期 compliant hierarchical control 研究也在尝试统一 passivity 与任意等式/不等式层级，但它保留的是动力学/惯性结构，不直接解决你这里的黑盒速度延迟。([arXiv][7])

---

# 十七、当前代码的具体重构建议

## 1. `track.py`

当前：

```python
v_star = compose_tff(v_pos, fout.v_force, self.selection)
```

改为产生结构化对象，而不是不可分解的六维速度：

```python
InteractionTaskBundle(
    nominal_twist=...,
    path_pose=...,
    path_tangent=...,
    path_s=...,
    path_speed_ref=...,
    force_measurement=...,
    force_target=...,
    force_quality_band=...,
    force_hard_bounds=...,
    contact_normal=...,
    image_error=...,
    image_jacobian=...,
)
```

兼容模式仍可生成旧 `v_star`，但新 QP 应直接读取各任务。

---

## 2. `protocol.py`

增加：

* 固定接触端口点 \(P_0\)；
* \(W_P,V_{P,m}\)；
* \(\hat K_e^\pm,\hat B_e^\pm\)；
* \(F_{\rm contact},F_L,F_U,F_{\rm safe}\)；
* path \(s,\nu_{\rm ref},X_d'(s)\)；
* image feature/Jacobian/confidence；
* \(V_a,V_{\rm cdyob},V_{\rm final}\)；
* predictor domain；
* committed energy/force reserve。

---

## 3. `loop.py`

负责唯一的时间同步和动态状态：

* wrench/twist 同时刻插值；
* 共享六维账本；
* \(T_n\) delay queue；
* actual-port power；
* force predictor；
* backup terminal state；
* 所有 shadow 指标。

不要让每个 force controller 各自维护一套不同的时间队列。

---

## 4. `qp_builder.py`

当前变量只有：

$$
[\dot q,w_{\rm task},w_{\rm pref}].
$$

扩展为：

$$
[\dot q,\nu,w_{\rm task},
\delta_F,\delta_I,\delta_c,w_{\rm pref}].
$$

QP1 加入：

* actual-port energy row；
* force upper/lower rows；
* force slew；
* path/normal cone；
* image CLF；
* dedicated slacks。

QP2 继续做姿态、轨道、nullspace，但应锁定的是 QP1 已达到的临床任务最优值，而不只是完整 TCP 速度。

---

## 5. `force_corridor.py`

不要再直接 clamp 一个 pre-QP Z 速度。

改为输出：

$$
A_F\dot q\le b_F,
\qquad
A_F^-\dot q\ge b_F^-.
$$

也就是 force-envelope row builder。

---

## 6. `delay_safety_shield.py`

保留并抽取：

* delay queue；
* stop-distance table；
* backup rollout；
* terminal-set check；
* uncertainty tube。

删除它单独产生最终速度的职责，让 QPIK 成为唯一命令 owner。

---

## 7. `energy_tank.py`

当前 `ActiveTermTank` 保留为实验诊断类，新增：

```python
class PortEnergyLedger6D:
    update_actual_port(...)
    committed_reserve(...)
    build_qp_row(...)
```

不要在新 actual-port tank 中再次扣除 \(F^\star v\)。

---

## 8. `cdyob.py`

必须接收：

* 最终 `u_sent`；
* QP intervention norm；
* solver status；
* predictor confidence；
* actual velocity。

当：

$$
\|V_{\rm nominal}-V_{\rm final}\|
$$

较大时，冻结或回算 observer state。

---

# 十八、真正有 oral/IJRR 潜力的论文主线

单纯：

> energy tank + CBF + QP

不够新。Secchi 2019 和 Benzi 2021 已经完成了这一基本框架。 

单纯：

> force constraint + QP

也不够新，已有 force-constrained CBF 工作。([arXiv][2])

单纯：

> 超声力 + 姿态 + 图像 + QP

也已有相关优化控制框架。([arXiv][8])

2026 年 Piedra–Khatib 的系统更进一步，在力矩接口上直接控制接触力与力矩，并用声学耦合优化探头完整姿态；它显著降低了导纳方案在动态力跟踪中的相位滞后。因此你的论文必须突出“黑盒速度接口 + 延迟证书”，而不是与 torque-level explicit force controller 正面对称比较方法形式。([Sage Journals][9])

---

## 推荐题目

### ICRA 版本

**Delay-Aware Patient-Port Viability Governor for Force-Bounded Time-Free Robotic Ultrasound Scanning**

中文可以理解为：

> 面向黑盒速度机器人的延迟感知患者端口可行性调节器。

简称可以是：

$$
\boxed{\text{PPVG}}
$$

或者：

$$
\boxed{\text{DACE-QP}}
$$

Delay-Aware Contact-Energy QP。

---

## 核心创新不要写成控制器堆叠，而要写成四个定理

### 定理 1：延迟实际端口无源性

在：

* 实际 wrench/twist 时间同步；
* 速度模型误差有界；
* delay queue 已进入 committed reserve；
* backup terminal set 有效；
* QP 最终命令无后处理；

条件下，患者端口满足：

$$
\sum_{k=0}^{N-1}
\Delta t\,W_k^\top V_{m,k}
\ge
-E_{\rm available}.
$$

---

### 定理 2：接触无再碰撞

首次捕获后，若 robust force/contact CBF 可行，则：

$$
F_{\rm contact}\le F(t)\le F_{\rm safe},
$$

从而接触间隙不打开，不发生 separation/reimpact。

这比声称“无源所以不弹跳”严谨得多。

---

### 定理 3：可行域内无性能损失

定义：

$$
\mathcal V
=
\left\{
x:
V_{\rm nominal}
\text{ 满足所有硬约束}
\right\}.
$$

当：

$$
x\in\mathcal V,
$$

QP 的最优解严格满足：

$$
V_{\rm final}=V_{\rm nominal}.
$$

所以：

* CDYOB；
* 低阻尼 FCE；
* hover 轻盈手感；

在安全可行区内完全保留。

这就是你一直希望得到的：

$$
\boxed{
\text{passivity without performance degradation inside a certified viability region}.
}
$$

---

### 定理 4：安全路径进度

若持续存在：

$$
\nu\ge\underline\nu>0
$$

的可行解，则扫描有限时间完成；否则系统安全暂停，并在约束恢复后继续。

---

# 十九、三种角色检查后的最终收敛

## 性能设计者的结论

* 需要低 \(M,D\) 名义 FCE；
* CDYOB 尽量消除速度口失配；
* force target 不做高增益精确 equality；
* 保持轻盈手感。

## 安全理论视角的反对意见

* 低阻尼名义律不能直接发送；
* 命令端 tank 在 \(28\sim40\,\mathrm{ms}\) 延迟下不构成实际端口证明；
* 必须有 committed reserve、force upper bound 和 contact viability；
* 无源不等于不弹。

## 审稿人视角的批评

* “tank + CBF”已有；
* “力进 QP”已有；
* “超声图像 + 力 + 姿态”已有；
* “CDYOB”已有；
* “路径在线调速”已有。

## 收敛后的真正新点

$$
\boxed{
\begin{aligned}
&\text{黑盒速度接口的实际端口六维无源性}\\
+{}&\text{输入队列的 committed-energy reserve}\\
+{}&\text{接触捕获后的 robust force funnel}\\
+{}&\text{无分离/无再撞击证书}\\
+{}&\text{路径进度 }\nu\text{ 与力/能量/图像共同优化}
\end{aligned}}
$$

这个组合不是普通的模块堆叠，而是一个统一的：

$$
\textbf{delay-augmented viability governor}.
$$

---

# 二十、建议的实际实施顺序

## 第一阶段：不要马上做全六维理论

先锁轨道，只做 tool-Z：

1. actual wrench/velocity 时间同步；
2. \(T_n\) delay queue；
3. shadow committed force/energy reserve；
4. QP 内 hard \(F_{\rm safe}\)；
5. force funnel \([2,6]\,\mathrm N\)；
6. path progress \(\nu\)；
7. 统计 QP intervention。

这一步先证明：

$$
F_{\rm safe}\text{ 不违反},
\qquad
N_{\rm loss/reimpact}=0.
$$

## 第二阶段：激活 actual-port tank

比较：

* 无 reserve；
* 有 reserve；
* commanded-port；
* actual-port；
* TDPA；
* energy-QP。

## 第三阶段：CDYOB

将其作为 nominal performance enhancement，而不是证书。

## 第四阶段：图像和 tilt

先软任务，再做 confidence cone；不要依赖图像承担硬力安全。

## 第五阶段：共享六维端口

完成 \(6\times6\) FRF、旋转误差和固定端口点验证后，再从单轴扩展到：

$$
F^\top v+M^\top\omega.
$$

---

最终控制架构应是：

$$
\boxed{
\begin{aligned}
&\text{F/T + image + pose estimation}\\
\rightarrow{}&
\text{低阻尼 nominal admittance/CDYOB}\\
\rightarrow{}&
\text{delay predictor + queue reserve}\\
\rightarrow{}&
\text{final viability HQP}
\begin{cases}
\text{energy}\\
\text{force funnel}\\
\text{contact set}\\
\text{image/tilt}\\
\text{path progress}\\
\text{robot safety}
\end{cases}\\
\rightarrow{}&
\text{RM75}
\end{aligned}}
$$

你的修改思路整体正确，但最关键的提升是：**不要把 QP 看成新的力控制器，而应把它定义成一个作用于最终硬件命令的、延迟感知的患者端口可行性调节器。**这才是可以同时解释轻盈手感、力允许松弛、严格安全边界、无源性、无再碰撞和自动扫描调速的统一理论。

[1]: https://arxiv.org/abs/2603.24566 "https://arxiv.org/abs/2603.24566"
[2]: https://arxiv.org/abs/2405.04859 "https://arxiv.org/abs/2405.04859"
[3]: https://eprints.imtlucca.it/id/eprint/496 "https://eprints.imtlucca.it/id/eprint/496"
[4]: https://portal.fis.tum.de/en/publications/confidence-driven-control-of-an-ultrasound-probe/ "https://portal.fis.tum.de/en/publications/confidence-driven-control-of-an-ultrasound-probe/"
[5]: https://arxiv.org/abs/2310.03406 "https://arxiv.org/abs/2310.03406"
[6]: https://ieeexplore.ieee.org/document/10870963/ "https://ieeexplore.ieee.org/document/10870963/"
[7]: https://arxiv.org/abs/2403.08491 "https://arxiv.org/abs/2403.08491"
[8]: https://arxiv.org/abs/2203.02126 "https://arxiv.org/abs/2203.02126"
[9]: https://journals.sagepub.com/doi/10.1177/02783649261441635 "https://journals.sagepub.com/doi/10.1177/02783649261441635"

你说得对，**Tool-Z 力控制与 Hover 是两条完全不同的任务线**。上一条把它们混在一起了。

## 1. 你的当前任务不是 Hover

你的扫描控制应当是：

$$
\mathbf V_d=
\begin{bmatrix}
v_x^{\mathrm{path}}\\
v_y^{\mathrm{path}}\\
v_z^{\mathrm{force}}\\
\omega_x^{\mathrm{tilt}}\\
\omega_y^{\mathrm{tilt}}\\
\omega_z^{\mathrm{path}}
\end{bmatrix}.
$$

其中：

* Tool-Z：根据 \(F_z\) 做法向力控制；
* Tool-X/Y 平移：跟踪扫描路径；
* 两个 tilt 旋转轴：根据 \(\tau_x,\tau_y\)、超声图像接触质量或表面法向，缓慢修改姿态；
* 剩余旋转轴：跟踪规划姿态。

例如两个旋转方向可以生成一个受限的自适应参考：

$$
\boldsymbol\omega_{xy,d}
=
-K_\tau
\,\operatorname{sat}
\left(
\begin{bmatrix}
\tau_x\\
\tau_y
\end{bmatrix}
\right),
$$

或者先积分为姿态参考：

$$
\dot{\boldsymbol\theta}_{xy,d}
=
-K_\tau\boldsymbol\tau_{xy},
\qquad
\left\|
\boldsymbol\theta_{xy,d}
-
\boldsymbol\theta_{xy,\mathrm{path}}
\right\|
\le \theta_{\max}.
$$

这属于**受约束的 torque-guided tilt adaptation**，不是 Hover：

* 不是六维自由漂浮；
* 不是人推哪个方向就往哪个方向走；
* 姿态仍被扫描轨迹和角度边界约束；
* 只有 Tool-Z 是主要力闭环。

所以当前论文不需要 Hover 的 Poinsot 轴、固定握持中心、六维零力导纳、REST/ACTIVE/RELEASE 等内容。Hover 最多是另一篇工作或单独 demo。

### 能量端口也不必强行六维

若只证明法向通道：

$$
P_n=F_zv_z,
$$

用一个**标量法向能量账本**就够了。

若还要把两个 torque-guided tilt 包含在患者接触端口的无源性中，则使用三维接触功率：

$$
P_c
=
F_zv_z+\tau_x\omega_x+\tau_y\omega_y.
$$

这时是“1 个法向 + 2 个旋转”的共享罐，而不是完整六维罐。

---

# 2. CDYOB 不是唯一主线，也不应该成为你的论文主线

在这些论文中，CDYOB 确实是**最适合你黑盒速度接口的性能补偿器**：它使用指令速度、实测速度和 F/T 数据，在外部修改速度参考，不要求打开内部速度控制器。Samuel 的 PROB/CDYOB 正是为这种黑盒工业机器人设计的。

但它不应该成为你的核心创新，原因有两个：

1. CDYOB 已经是 Samuel 2024 的主要贡献；
2. CDYOB解决的是：

   $$
   \text{“如何让实际机器人更接近期望导纳？”}
   $$

   而你的核心问题是：

   $$
   \text{“延迟速度机器人怎样在扫描中保证力区间、接触不丢失和能量安全？”}
   $$

这两个问题不同。

你现在更合理的论文分工是：

| 模块                      | 作用                                   | 是否论文主线 |
| ----------------------- | ------------------------------------ | ------ |
| 名义 Tool-Z 导纳/FCE        | 产生舒服、快速的 \(v_{z,\mathrm{nom}}\)      | 基础控制律  |
| CDYOB                   | 改善 \(v_{\mathrm{nom}}\to v_m\) 的实现精度 | 可选性能增强 |
| \(T_n\) predictor       | 预测延迟后实际运动和已承诺压入                      | 核心使能模型 |
| Force corridor / funnel | 保证 \(F_z\) 落在允许区间                    | 核心证书   |
| Energy constraint       | 限制患者端口净输出能量                          | 核心证书   |
| 路径速度变量 \(\dot s\)       | 力控制困难时自动降速                           | 核心创新   |
| \(\tau_x,\tau_y\) tilt  | 改善探头接触姿态和图像质量                        | 应用任务   |

Secchi/Benzi 更接近你“能量约束进入 QP”的理论骨架；De Stefano 更接近延迟造成的能量不一致；Lee 解释 proxy 与真实机器人速度失配为何产能；Dimeas 是振荡检测器。CDYOB只负责其中的“名义性能恢复”。Secchi/Benzi 已经证明能量罐约束和 CBF 可以共同进入凸优化，所以你的创新必须落在**黑盒延迟、已承诺运动、力走廊和路径调速**上。 

你现有 P0–P4 方案中，把 P3 定为 CDYOB shadow、P4 定为能量与力约束 QP，这种分离本身是正确的。

而且当前 `RealUS_playground` 的 `cdyob.py` 自己也明确写着：它是 Tool-Z 的 **performance term**，不是 passivity certificate；第一版 \(N_2=0\)，采用不能拆分 \(C_n,R_n\) 时的等效 \(N_1\) 实现。

所以更准确的结论是：

$$
\boxed{
\text{CDYOB 是可选的名义性能层，不是论文安全主线。}
}
$$

---

# 3. `10_tn_observe.py` 识别的是 CDYOB 吗？

**不是识别整个 CDYOB。**

它做的是：

1. 从已有的 `03_chirp_gv.py` 或 `09_multisine_gv.py` 数据中，辨识黑盒速度内环；

2. 把该速度内环写成：

   $$
   G_v(s)
   =
   \Gamma_d(s)T_n(s);
   $$

3. 验证 Samuel 式 (10) 的第一项是否可以计算；

4. 不输出任何补偿命令。

仓库 README 明确写着：

* `10_tn_observe.py` 是 analyze-only；
* 只观察式 (10) 第一项；
* 不辨识 \(Q,N_1,N_2\)；
* 不启用 CDYOB。

脚本注释也明确给出：

$$
\hat P_{\mathrm{ert}}
=
Q\left[T_n^{-1}V_m-V_i\right]
+
N_1F_m
-
N_2V_m,
$$

而 `10` 只处理最前面的：

$$
T_n^{-1}V_m-V_i.
$$

它不设计 \(N_1,N_2\)，也不把估计结果写回命令。

因此它更准确的定位是：

$$
\boxed{
\text{CDYOB 所需的内环名义模型验证工具，而不是 CDYOB 辨识工具。}
}
$$

类似于：你给 MPC 辨识了一个电机模型，并不代表你已经辨识或实现了 MPC。

---

# 4. \(T_n\) 到底是什么

这里的“taon”应该就是 \(T_n\)，读作 **T-sub-n**。

它不是：

* 力矩 \(\tau\)；
* 也不是一个名为 \(\tau_n\) 的时间常数。

下标 \(n\) 表示：

$$
n=\text{nominal，名义模型}.
$$

Samuel 论文中内部系统是：

$$
C(s)=\text{内部速度控制器},
$$

$$
R(s)=\text{机器人机械动态},
$$

内部闭环速度传递函数为：

$$
T(s)
=
\frac{C(s)R(s)}
{1+C(s)R(s)}.
$$

\(T_n(s)\) 是 \(T(s)\) 的名义模型：

$$
T_n(s)\approx T(s).
$$

通信与计算纯延迟另外写成：

$$
\Gamma_d(s)=e^{-T_0s}.
$$

所以完整的指令速度到实测速度关系是：

$$
\boxed{
V_m(s)
\approx
\Gamma_d(s)T_n(s)V_i(s).
}
$$

Samuel 对 \(T_n\) 的定义就是“内部闭环动态 \(T=CR/(1+CR)\) 的 nominal model”；\(Q\) 则用来使 \(T_n^{-1}\) 的实现可实现并抑制速度测量噪声。

---

## 你现在辨识出的具体 \(T_n\)

`10_tn` 保存的数据是：

$$
T_0=0.028\ \mathrm{s},
$$

$$
T_p=0.014\ \mathrm{s},
$$

$$
K=0.96.
$$

因此：

$$
\Gamma_d(s)
=
e^{-0.028s},
$$

$$
T_n(s)
=
\frac{0.96}
{0.014s+1},
$$

完整速度植物是：

$$
\boxed{
G_v(s)
=
\frac{V_m(s)}{V_i(s)}
\approx
e^{-0.028s}
\frac{0.96}{0.014s+1}.
}
$$

记录中延迟是 6 个控制 tick，前向预测误差 RMSE 为：

$$
1.63\ \mathrm{mm/s}.
$$

同时 JSON 明确记录：

* `n1_n2_identified=false`；
* `cdyob_applied=false`；
* \(Q\) 也不是辨识出的植物参数。

一个需要纠正的细节是：这里的 \(1.63\,\mathrm{mm/s}\) 是

$$
V_m-\Gamma_dT_nV_i
$$

的**前向模型预测误差**，不是

$$
T_n^{-1}V_m-V_i
$$

经过 \(Q\) 后的残差大小。

---

# 5. 为什么需要 \(T_n^{-1}\)

已辨识：

$$
T_n(s)
=
\frac{K}{T_ps+1},
$$

所以：

$$
T_n^{-1}(s)
=
\frac{T_ps+1}{K}.
$$

时域中近似为：

$$
T_n^{-1}V_m
=
\frac{V_m+T_p\dot V_m}{K}.
$$

它的含义是：

> 根据机器人当前实际运动，反推出“名义情况下应该是什么速度指令产生了这个运动”。

然后与真正发送过的 \(V_i\) 比较：

$$
r
=
T_n^{-1}V_m-V_i.
$$

如果模型、时序和外部扰动都理想，残差应很小；若不小，可能来自：

* 摩擦；
* 饱和；
* 接触耦合；
* 未建模动态；
* 指令与实测速度不一致；
* 时间延迟没有正确对齐。

因为这里包含 \(\dot V_m\)，直接求导会放大噪声，所以 Samuel 使用：

$$
Q(s)
=
\frac{\omega_Q}{s+\omega_Q},
$$

计算：

$$
Q(s)T_n^{-1}(s)V_m.
$$

纯延迟：

$$
\Gamma_d^{-1}(s)
=
e^{+T_0s}
$$

不能实时实现，因为它要求提前知道未来。因此脚本和当前 CDYOB 都明确规定：

$$
\boxed{\Gamma_d\ \text{不反演。}}
$$

延迟只能通过：

* 时间戳对齐；
* 命令队列；
* predictor；
* 已承诺运动 reserve；

来处理。

---

# 6. \(T_n\) 对你的论文真正有什么用

即使完全不用 CDYOB，\(T_n\) 仍然非常重要。

## 用途一：计算已承诺压入距离

当你现在发出“停止压入”或“立即回撤”时，RM75 不会立刻执行。旧命令还会继续经过：

$$
T_0+T_p
$$

产生运动。

近似稳态速度命令为：

$$
v_i=20\ \mathrm{mm/s},
$$

则发送停止命令后的额外位移约为：

$$
\Delta x_{\mathrm{tail}}
\approx
K v_i(T_0+T_p).
$$

代入：

$$
\Delta x_{\mathrm{tail}}
\approx
0.96
\times
20
\times
(0.028+0.014)
\ \mathrm{mm},
$$

得到：

$$
\Delta x_{\mathrm{tail}}
\approx
0.806\ \mathrm{mm}.
$$

若环境刚度：

$$
K_e=2200\ \mathrm{N/m},
$$

那么即使这一拍已经命令停止，后续还可能增加：

$$
\Delta F
\approx
K_e\Delta x_{\mathrm{tail}}
\approx
1.77\ \mathrm N.
$$

这就是：

$$
F_{\mathrm{future}}^{UB}
=
F_k+
\bar K_e
\Delta x_{\mathrm{committed}}^{UB}
$$

为什么必须使用 \(T_n\) 和 delay queue。

---

## 用途二：让 QP 约束未来实际力，而不是当前命令

普通 QP 看到的是：

$$
v_{z,\mathrm{cmd}}
=
J_z\dot q.
$$

但患者真正感受到的是：

$$
v_{z,m}
=
\Gamma_dT_nJ_z\dot q.
$$

因此力走廊不能只写成：

$$
F_k+\hat K_e\Delta t\,J_z\dot q
\le F_{\max},
$$

而要基于延迟队列预测：

$$
\hat F_{k+h|k}
=
F_k+
\hat K_e
\int_{t_k}^{t_{k+h}}
\hat v_{z,m}(\tau)\,d\tau.
$$

\(T_n\) 就是把“QP 当前选择的命令”转换成“未来实际会发生的速度”的模型。

---

## 用途三：CDYOB 的性能补偿

CDYOB 使用：

$$
Q(T_n^{-1}V_m-V_i)
$$

估计速度内环没有实现的部分，再修正速度参考，使实际 apparent admittance 更接近目标：

$$
Y_{\mathrm{CDYOB}}\rightarrow A.
$$

Samuel 设计 \(N_1,N_2\) 后，在理想 \(Q=1\) 条件下得到：

$$
Y_{\mathrm{CDYOB}}=A.
$$



但这是性能恢复，不是力上限或无源性证明。

---

## 用途四：区分“控制器错了”还是“硬件没跟上”

有了 \(T_n\)，你可以分别记录：

$$
e_{\mathrm{plant}}
=
V_m-\Gamma_dT_nV_i,
$$

$$
e_{\mathrm{proxy}}
=
V_a-V_m.
$$

这样可以判断：

* 外层力律是否要求过激；
* QP 是否削减了命令；
* RM75 是否没有实现最终命令；
* 接触后植物是否偏离空中模型；
* 弹跳是力律产生，还是延迟队列造成。

---

# 7. \(T_n\) 目前还不是安全证书

现在的辨识只是一个点模型：

$$
K=0.96,\quad
T_0=28\,\mathrm{ms},\quad
T_p=14\,\mathrm{ms},
$$

加一个平均 RMSE。

它足够用于：

* CDYOB shadow；
* predictor 原型；
* 日志对齐；
* 名义仿真。

但要用于 QP 的硬安全约束，还必须得到不确定集合：

$$
K\in[\underline K,\overline K],
$$

$$
T_0\in[\underline T_0,\overline T_0],
$$

$$
T_p\in[\underline T_p,\overline T_p],
$$

以及有限时域速度误差界：

$$
|V_m-\hat V_m|
\le \bar e_v(h).
$$

然后使用最坏情况：

$$
\Delta x_{\mathrm{committed}}^{UB}
=
\sup_{\theta_T\in\Theta_T}
\int[\hat v_m(\tau;\theta_T)]_+d\tau.
$$

因此 `10_tn` 的下一步不是立即打开 CDYOB，而是：

1. 用不同速度、方向、姿态重复 `03/09`；
2. 用 `07_contact_gv` 在**力环关闭**情况下测试轻接触是否改变内环；
3. 得到 \(T_n\) 的不确定区间；
4. 先让它服务于 delay queue 和 force reserve；
5. CDYOB 最后作为性能增强做消融实验。

---

# 最终定位

你的当前论文结构应该收敛为：

$$
\boxed{
\begin{aligned}
&\text{Tool-Z 名义力控制}\\
+{}&\text{两个 torque-guided tilt 软任务}\\
+{}&T_n\text{ 延迟植物预测}\\
+{}&\text{未来力区间/接触不变约束}\\
+{}&\text{路径速度 }\dot s\text{ 自动调节}\\
+{}&\text{能量约束}
\end{aligned}}
$$

其中：

$$
\boxed{
T_n\text{ 是主线所需的预测模型，CDYOB只是它的一种用途。}
}
$$

CDYOB可以不启用，你的延迟力走廊 QP 仍然成立；但没有 \(T_n\) 或等价的延迟植物模型，就无法可靠计算“已经来不及撤回的未来压入和未来力”。

对，**你之前整套 ICRA 方案已经把“\(V_{\rm cmd}\) 追不上”当成核心前提了**。而且不是简单加一个误差项，而是把它拆成“延迟、旧命令、执行器动态、实际实现速度、残差包络”来处理。设计稿明确写的是：证明中心必须是**机器人实际能执行的未来运动**，不是当前 QP 给出的理想速度。

不过现在结合你刚发现的 Y 漂，我建议把这个假设再严格拆成 **两种不同的“追不上”**，这样和你的两层 QP 会非常干净。

### 第一种：QP 自己就实现不了 \(V_{\rm cmd}\)

你现在 QP1 本身是：

$$
J(q)\dot q-w=V_{\rm cmd}.
$$

因此

$$
V_{\rm qp}=J\dot q=V_{\rm cmd}+w.
$$

当奇异、关节限位、rail limit、collision CBF、a/jerk box 等发生时，

$$
w\neq0.
$$

这是**优化层的不可实现性**。

### 第二种：QP 算出来了，但硬件没有照着执行

即使

$$
w\approx0
$$

也只意味着：

$$
J\dot q_{\rm cmd}\approx V_{\rm cmd}.
$$

真实 RM75+rail 仍然是：

$$
\boxed{
V_{\rm actual}
\neq
J\dot q_{\rm cmd}
}
$$

因为有延迟、一阶响应、低速非线性、rail/arm 不同动态、hold tick 等。

这正是你方案里实验 11 要辨识的东西：从

$$
[u_z,\omega_\theta]
$$

到**实际接触点法向速度和实际角速度**的动态、延迟、交叉耦合和幅值依赖；如果 rail 混合导致响应模式变化，就要建立 mode-dependent contract，而不是硬平均成一个 LTI。

所以以后论文里的关系最好明确写成三级：

$$
\boxed{
V_{\rm task}
\rightarrow
V_{\rm qp}=J\dot q
\rightarrow
V_{\rm actual}
\rightarrow
F
}
$$

而不是写成：

$$
V_{\rm task}=V_{\rm actual}.
$$

---

## 这其实和你两层 QP 非常适合

你现在的两层结构可以基本保留：

$$
\boxed{
\text{QP1：task / safety / force viability}
}
$$

$$
\boxed{
\text{QP2：rail + posture + nullspace}
}
$$

但要做一个非常关键的调整。

### QP1 不再只保护“理想 Cartesian velocity”

你的第一层应该同时决定：

$$
z_1=
[\dot q,\;u_z,\;\omega_\theta,\;\rho,\;\epsilon,\ldots].
$$

并满足：

$$
J\dot q
=
\rho V_p
+
E_z u_z
+
E_\theta\omega_\theta
+
V_g
+
w.
$$

这只是**运动学命令层**。

真正的 force constraint 要走：

$$
u_{\rm send}
\longrightarrow
\hat V_{\rm actual,h}
\oplus\mathcal W_{v,h}
$$

再进入：

$$
\hat F_{h+1}
=
\mathcal F_F
(
\hat F_h,
\hat V_{\rm actual,h},
\rho_h
).
$$

最后才约束：

$$
\boxed{
F_L
\le
\hat F_h-r_{F,h}
}
$$

$$
\boxed{
\hat F_h+r_{F,h}
\le
F_U.
}
$$

你设计稿实际上已经这么要求了：实际速度是黑盒对**最终发送命令**的响应，不是运动学等式；如果保留 task slack，其产生的六维运动差异也必须进入预测，不能让 z 方向 slack 绕开力管道。

这点现在尤其重要。

---

# 但你的 QP2 有一个新发现后必须注意的问题

这和刚才的 Y 漂直接相关。

你当前 QP2 做的是：

> 锁定 QP1 已经得到的 Cartesian twist，然后在 nullspace 内重新分配 rail/arm。

在**纯运动学**意义上，这是完美的：

$$
J_r\dot q_r+J_a\dot q_a=V_{\rm qp}
$$

保持不变。

但是：

$$
\boxed{
\text{相同 }V_{\rm qp}
\not\Rightarrow
\text{相同 }V_{\rm actual}(t)
}
$$

因为：

```text
解 A:
rail +40 mm/s
arm  -40 mm/s

解 B:
rail   0 mm/s
arm    0 mm/s
```

瞬时运动学上，两者都可能：

$$
v_Y=0.
$$

但是实际硬件动态：

$$
G_r(s)\neq G_a(s).
$$

所以：

$$
G_r(s)(+40)+G_a(s)(-40)
\neq0
$$

特别是接触以后。

**这正是你现在 Y 滑问题给我们的一个重要提醒。**

所以 QP2 虽然不会破坏：

$$
J\dot q=V_{\rm protected},
$$

却可能破坏：

$$
V_{\rm actual}
\in
\hat{\mathcal V}_{\rm certified}.
$$

---

# 因此两层 QP 最终应该这样结合

我推荐：

$$
\boxed{
\text{QP1：Certified patient-port layer}
}
$$

负责：

$$
u_z,\omega_\theta,\rho
$$

以及任何**会显著改变实际执行动态**的自由度。

硬约束包括：

$$
\begin{aligned}
&\text{joint / collision / rail limits}\\
&\text{execution tube}\\
&F_L\le F_h\le F_U\\
&\text{committed queue / stop tail}\\
&\text{terminal viability}
\end{aligned}
$$

然后：

$$
\boxed{
\text{QP2：Performance/nullspace layer}
}
$$

只允许做：

* posture
* manipulability
* \(d^*\)
* \(\psi^*\)
* rail centering

但必须满足下面三个方案之一。

### 方案 A——最干净

QP1 已经决定 rail/arm 分配。

QP2 不再改变：

$$
\dot q_r
$$

以及会影响认证 execution dynamics 的分量。

然后 QP2 只用真正剩余 nullspace。

理论最容易写。

---

### 方案 B——更符合你现在的控制器

把**完整 QP1+QP2+rail allocator+send path**整体看成执行器。

也就是辨识：

$$
\boxed{
[u_z,\omega_\theta,\rho]
\rightarrow
V_{\rm actual}
}
$$

而不是辨识：

$$
\dot q\rightarrow V_{\rm actual}.
$$

那么：

```text
predictive force QP
       ↓
[uz, ωθ, ρ]
       ↓
现有 QP1
       ↓
现有 QP2 + rail allocator
       ↓
RM75 + rail
       ↓
Vactual
```

你实验 11 得到：

$$
x_{a,k+1}
=
A_a x_{a,k}
+
B_a u_{k-d}
+
w_a.
$$

于是整个两层 QP 是 **plant/executor 的一部分**。

这个方案对你当前软件改动最少。

而且你的设计稿已经允许这么做：如果 rail mixing / direction switching 显著，就建立**不同模式的执行合同**。

---

### 方案 C——QP2 仍任意分配，但全部吃进 uncertainty tube

即：

$$
V_{\rm actual}
\in
\hat G V_{\rm task}
\oplus
\mathcal W_{\rm rail/null}.
$$

理论可以，但如果 QP2 引起的 rail-arm cancellation 很大：

$$
\mathcal W
$$

会变得很肥。

那么 force corridor：

$$
[F_L,F_U]
$$

很快就不可行。

所以我最不喜欢这个方案。

---

# 你的实验 12 又正好补第二部分

实验 11解决：

$$
\boxed{\text{“命令以后实际速度怎么走？”}}
$$

实验 12解决：

$$
\boxed{\text{“我现在改命令了，以前发出去的东西还会造成什么？”}}
$$

也就是：

$$
\text{old queue}
\oplus
\text{candidate}
\oplus
\text{backup tail}.
$$

设计稿明确要求检查**整个前缀**，不是只检查新命令第一次能生效的那一拍。

所以你的完整链条其实已经非常清楚：

$$
\boxed{
\begin{array}{c}
\text{QP candidate}\\
\downarrow\\
\text{QP kinematic realizability}\\
\downarrow\\
\text{black-box execution contract}\\
\downarrow\\
\text{old queue + committed motion}\\
\downarrow\\
V_{\rm actual}\;\text{tube}\\
\downarrow\\
F/\tau\;\text{tube}\\
\downarrow\\
\text{force corridor admission}
\end{array}}
$$

这正是你原设计稿的主线：状态 \(\xi\) 里本来就包括 servo state、力、几何误差、相位、**已发送队列、历史发送速度/加速度**等，而不是只看当前 \(F,V\)。

---

## 所以我现在会把你的两层 QP 定义得更准确

不是：

$$
\text{QP1 = Cartesian tracking}
$$

$$
\text{QP2 = nullspace}
$$

而是：

$$
\boxed{
\text{QP1 = certified achievable task}
}
$$

决定：

> **机器人在自己的执行延迟和误差下，现在到底允许做多少。**

然后：

$$
\boxed{
\text{QP2 = best realization of that certified task}
}
$$

决定：

> **在不破坏 QP1 认证的前提下，用什么构型/rail 分配最好。**

这就和你的论文“**可实现力管道**”非常一致。

唯一需要比原稿再加强的一点就是：

$$
\boxed{
\text{QP2 的 nullspace 等价性不能只在 }J\dot q\text{ 层证明}
}
$$

还必须保证它不会让**实际 execution contract**失效。

而你这次“接触瞬间 Y 滑但没开力控制”的现象，恰好就是为什么这个区别必须写进论文。














面向未知接触环境的可执行接触可行性分层控制：用于黑盒速度接口机器人超声扫描
Executable Contact-Viability Hierarchical Control for Robotic Ultrasound With Unknown Contact Dynamics and Black-Box Velocity Actuation

Abstract— 机器人超声的低层控制通常被表述为“沿表面扫描，同时维持恒定法向力”。这一表述在静态、已知或高带宽力矩控制平台上是自然的，但对于仅开放速度接口、具有不可忽略输入延迟和有限闭环带宽的工业机械臂，它混淆了两个本质不同的问题：控制器“希望”产生什么接触修正，以及硬件“来得及”产生什么接触修正。更进一步，人体组织的刚度、黏弹性、迟滞、曲率和切向摩擦均随患者和扫描位置变化，因此依赖精确患者接触模型来预测未来力，会使安全证明建立在难以转移的辨识假设之上。

本文提出一种以可执行接触可行性为核心的两层分层二次规划框架。控制器只保留两个交互自由度：Tool-Z 法向运动和绕经过探头 TCP、垂直超声图像平面的单一旋转自由度；其他方向沿参数化几何路径运动。外层力控制器不再承担“必须精确保持 (F^\star)”的安全职责，而只生成柔顺的名义 Tool-Z 速度；超声图像接触特征与腕部接触力矩联合生成名义旋转速度。第一层 QP 联合优化最终机器人速度命令、扫描速度缩放和任务松弛，并通过一个非参数化的集合值接触执行契约拒绝那些在机器人输入延迟和已发送命令队列下可能使未来接触力离开允许区间的命令。第二层 QP 严格锁定第一层已经实现的任务速度和扫描进度，同时优化轨道分配、关节构型、奇异性恢复和命令平滑，并重新施加相同的患者端口硬约束，从而保证低优先级构型优化不能破坏第一层接触证书。

本文进一步分析一个可选的、基于实际患者端口机械功的能量预算。与组织刚度模型不同，该能量证书只依赖同步测得的实际 wrench–twist 和机器人自身的已承诺运动上界，因此可在未知组织环境中给出模型无关的净能量输出界。其作用与力走廊正交：力可行性限制瞬时接触风险，能量预算限制机器人向患者累计输出的机械能。核心理论结果包括：未来力区间的鲁棒前向不变性、递归执行可行性、第一层名义行为的条件透明性、第二层任务不劣化与患者端口安全继承，以及可选的实际端口耗散性。

Index Terms— Robotic ultrasound, force control, hierarchical quadratic programming, contact viability, input delay, admittance control, passivity, energy tank, acoustic contact, path-speed regulation.

研究问题、文献脉络与最终设计取舍
从“恒定力控制”重新定义问题
这篇论文最重要的第一句话不应该是“我们提出了一个新的力控制器”，而应该是：

机器人超声真正需要保证的不是每一时刻都精确等于某个牛顿值，而是在有限执行能力下维持有效、不过载、可恢复的声学接触，同时尽可能实现临床期望的力、图像和扫描速度。

设希望力为

[ F^\star=4~\mathrm N, ]

并假设某一具体实验任务定义允许接触范围

[ F_{\mathrm L}\le F\le F_{\mathrm U}, \qquad F_{\mathrm L}=2~\mathrm N,\quad F_{\mathrm U}=6~\mathrm N. \tag{1} ]

这里必须严格区分：

[ \boxed{ F_{\mathrm L},F_{\mathrm U}
\text{任务/安全规范} } ]

与

[ \boxed{ F^\star
\text{区间内部的性能偏好}. } ]

(2!-!6~\mathrm N) 不能在论文里被写成普适临床安全范围；它只是用于说明方法的任务规范。真正临床范围应由检查部位、探头尺寸、检查目的和伦理规范确定。

为什么这种区别重要？

考虑传统导纳：

[ M_a\dot v_a+D_av_a=F^\star-F. \tag{2} ]

冻结环境和速度内环后，一个典型法向闭环可近似为

[ L(p)= \frac{ K_eK_v e^{-T_dp} }{ p(M_ap+D_a)(1+\tau_vp) }, \tag{3} ]

其中 (p) 是拉普拉斯变量，(K_e) 是局部环境增量刚度，(K_v,\tau_v,T_d) 分别表示速度伺服增益、时间常数和延迟。

注意，特征方程

[ 1+L(p)=0 \tag{4} ]

中没有 (F^\star)。

因此：

把 (4~\mathrm N) 换成 (3~\mathrm N)，或者仅仅说“不再恒定力”，本身并不能改变闭环稳定性。

真正发生改变的是控制器的必须动作集合。

在精确力跟踪中，只要

[ F\neq F^\star, ]

控制器就认为存在必须立即消除的误差。

而在集合控制中：

[ F\in[F_{\mathrm L},F_{\mathrm U}] ]

时，精确中心误差只是一种偏好；如果为了把 (4.5~\mathrm N) 精确拉回 (4~\mathrm N) 所要求的运动已经超过硬件未来执行能力，QP 可以主动不执行全部修正。

换言之：

[ \boxed{ \text{松弛的是追 }F^\star\text{ 所要求的运动，} \quad \text{不是 }F_{\mathrm U}\text{ 本身。} } \tag{5} ]

这恰好对应 Keemink 等关于 admittance 中“低表观惯量/阻尼—刚性接触稳定性”权衡的分析，以及 Dimeas 和 Aspragathos 通过在线振荡检测增加耗散来改善稳定性的动机。二者都说明，提高阻尼可以改善接触鲁棒性，但同时损害自由运动或人工推动时的轻盈感 [Keemink et al., 2018; Dimeas & Aspragathos, 2016]。

为什么你的平台不能采用“命令速度约等于实际速度”的假设
你在 ICRA_2027 中已有的数据给出了非常明确的平台证据。

目前前面对仓库数据的核对得到：

[ G_v^z(p) \approx e^{-0.028p} \frac{0.96}{0.014p+1} \tag{6} ]

作为一次 Tool-Z 自由空间辨识的名义模型；另一组 multisine 数据给出的稳态增益约为 (0.882)，时间常数约为 (20~\mathrm{ms})，而纯延迟仍约 (28~\mathrm{ms})。在 (1!-!3~\mathrm{Hz}) 相关扫描频段，你已有记录中命令—实际速度误差 RMSE 约为 (5.58~\mathrm{mm/s})。10_tn 得到的约 (1.63~\mathrm{mm/s}) 是名义前向模型预测残差，而且其中 cdyob_applied=false，不能解释为 CDYOB 已经补偿后的误差。

这些结果有两个意义。

第一：

[ V_{\mathrm cmd,k}\neq V_{\mathrm actual,k}. ]

第二，更重要的是：

[ u_k \not\rightarrow V_{m,k} ]

而大致表现为：

[ u_k \rightarrow V_{m,k+d}, \qquad d\approx T_d/h. \tag{7} ]

也就是说，当传感器已经检测到过力而外层开始回撤时，机械臂可能仍在执行几十毫秒之前的压入命令。

De Stefano 等对具有通信/执行延迟的机器人交互系统给出了非常直接的能量解释：即使虚拟系统本身具有被动性质，延迟后的真实机器人也可能在碰撞中出现额外能量并产生不断增强的反弹。Lee 等的 BEFM 从另一角度指出，代理速度 (V_a) 与真实速度 (V_m) 之间的失配项

[ (V_a-V_m)^\top W_e \tag{8} ]

没有固定符号，因此“虚拟导纳被动”不能直接推出“真实物理端口被动” [De Stefano et al., 2020; Lee et al., 2024]。

Samuel 等针对黑盒工业机器人提出的 perturbation-robust/CDYOB 框架也正是从“期望导纳与真实内环响应不同”出发，通过名义闭环模型 (T_n)、实际速度和外力恢复期望 apparent admittance。它适合作为你的性能增强模块，但它解决的是

[ \text{实际运动更接近期望运动} ]

而不是

[ \text{未来力绝不越界}. ]

因此本文不把 CDYOB 放在核心理论中，而保留为可插拔的 nominal-controller baseline [Samuel et al., perturbation-robust admittance paper]。

为什么也不能要求精确辨识人体黏弹性
另一个极端是：

[ F_{k+h}
\mathcal M(K_e,B_e,\tau_e,\ldots) ]

并试图精确辨识每个患者、每个位置的 (K_e,B_e,\tau_e)。

这也不是一个理想的 IJRR 主线。

人体接触的：

局部刚度；
黏弹性衰减；
接触面积；
预条件效应；
摩擦；
深层骨性结构；
患者运动；
都会变化。你较新的 contact 数据中，短接触记录与空气模型拟合已经出现明显差异；这更适合作为“患者参数辨识不稳定，因此应避免依赖精确患者模型”的证据，而不是拿某个 contact fit 的单一 (K_e) 当作定理参数。

这里需要做一个理论上很重要的区分：

[ \boxed{ \text{不需要精确环境模型} \neq \text{可以对任意未知环境做无条件保证}. } ]

如果允许环境具有无限刚度、任意快自主运动，或者可以在下一采样周期无视机器人动作把力从 (4~\mathrm N) 跳到 (100~\mathrm N)，那么任何控制器都无法保证有限力区间。

因此严格安全至少必须假设某种有限的输入—输出响应类。

本文建议把这个最低要求从“组织参数辨识”改写为：

[ \boxed{ \textbf{集合值接触执行契约} } ]

即只需要知道：

在当前实际速度、当前力变化趋势、已发送命令队列和候选新命令下，未来力可能落在哪个保守集合中。

不要求这个集合来自 Kelvin–Voigt、Maxwell 或某个生物力学模型。

这正是本文和传统 model-based force MPC 最本质的区别。

最终只保留三个创新对象
经过上述取舍，本文不应该堆叠十几个模块。

真正值得保留的只有：

[ \boxed{ \begin{aligned} &\text{执行延迟感知的接触可行性约束},\ &\text{力区间 + 扫描速度联合让步},\ &\text{两层 QP 中患者端口约束的严格继承}. \end{aligned}} \tag{9} ]

图像—力矩旋转是临床意图输入；CDYOB 是性能模块；actual-port energy budget 是一个可选的第二证书，而不是论文标题中的主要创新。

这比“energy tank + CBF + CLF + DOB + CDYOB + CTF + HQP”更接近一篇有中心思想的论文。

最小模型：只辨识机器人，不辨识患者模型
坐标、状态与控制变量
设机器人与轨道合并后的广义位置为

[ q\in\mathbb R^{n}, \qquad n=8 ]

（一个 rail + 七个机械臂关节）。

定义 QP 真正发送给底层速度接口的命令：

[ u= \begin{bmatrix} u_r& u_1&\cdots&u_7 \end{bmatrix}^{!\top} \in\mathbb R^8. \tag{10} ]

这里刻意写 (u)，而不是 (\dot q)，因为：

[ \boxed{ u=\text{命令速度}, \qquad \dot q_m=\text{实际速度}. } ]

这对于黑盒速度接口非常重要。

探头 TCP twist 为：

[ V_T= \begin{bmatrix} v_T\\omega_T \end{bmatrix}
J(q)\dot q_m. \tag{11} ]

在 QP 的运动学层，命令对应的理想任务速度写为：

[ V_c=J(q)u. \tag{12} ]

但是患者真正看到的是 (V_T)，而不是 (V_c)。

本文只开放两个交互自由度。

定义：

[ c_z= \begin{bmatrix} z_T\0 \end{bmatrix}\in\mathbb R^6 \tag{13} ]

为 Tool-Z 平移方向。

定义：

[ c_\theta= \begin{bmatrix} 0\a_\theta \end{bmatrix}\in\mathbb R^6 \tag{14} ]

其中 (a_\theta) 是经过 TCP、垂直超声图像平面的旋转轴。

于是：

[ v_z=c_z^\top V_T, \qquad \omega_\theta=c_\theta^\top V_T. \tag{15} ]

其他四个 task-space 自由度主要承担扫描路径保持。

无时间路径与速度缩放
将原来的时间轨迹

[ X_d(t) ]

改写为几何路径：

[ X_d(s), \qquad s\in[0,1]. \tag{16} ]

其中 (s) 是路径 phase，而不是时间。

定义名义路径速率：

[ \nu_{\rm ref}(s)>0. ]

新增无量纲 QP 变量：

[ \alpha\in[0,1] \tag{17} ]

并令：

[ \dot s
\alpha\nu_{\rm ref}(s). \tag{18} ]

因此：

(\alpha=1)：按计划速度扫描；
(0<\alpha<1)：自动减速；
(\alpha=0)：保持当前位置，暂停路径推进。
路径状态更新：

[ s_{k+1}
s_k+h\alpha_k\nu_{\rm ref}(s_k), \tag{19} ]

其中 (h) 是 QP 控制周期。

注意，真正应该缩放的是沿路径的 feedforward progress，而不是保持横向路径误差的反馈。

设路径单位 phase 对应的 twist 为：

[ \Xi_d(s)
\frac{\partial X_d}{\partial s} \tag{20} ]

在适当的 (SE(3)) 局部表示下转换成 task-space twist，则：

[ V_{\rm path,ff}
\Xi_d(s)\nu_{\rm ref}(s). \tag{21} ]

最终路径请求写为：

[ V_{\rm path}
V_{\perp,\rm fb} + \alpha V_{\rm path,ff}, \tag{22} ]

其中 (V_{\perp,\rm fb}) 只负责把探头拉回几何路径附近，而不恢复固定墙钟时间。

这解决了一个传统 fixed-time trajectory 的结构问题：

[ \text{接触已经困难} \quad\text{但}\quad t\text{ 还在继续走}. ]

后者会产生越来越大的路径追赶误差，进一步与力控制争夺执行能力。

名义 Tool-Z 力控制器只负责“想怎么动”
保留一个简单、可解释的低阻尼导纳：

[ M_a\dot v_a+D_av_a=F^\star-F. \tag{23} ]

其中：

(M_a>0)：虚拟质量；
(D_a>0)：虚拟阻尼；
(F)：当前补偿后的 Tool-Z 压缩力；
(F^\star)：区间内期望力；
(v_a)：名义 Tool-Z 速度。
这条控制律不再承担硬安全证明。

它只回答：

如果不存在机器人延迟、安全约束和其他任务，我希望 Tool-Z 怎么动？

用隐式欧拉实现：

[ M_a \frac{v_{a,k+1}-v_{a,k}}{h} + D_av_{a,k+1}
F^\star_k-F_k. \tag{24} ]

可直接解得：

[ \boxed{ v_{a,k+1}
\frac{ M_av_{a,k} + h(F^\star_k-F_k) }{ M_a+hD_a }. } \tag{25} ]

所谓“隐式”只是因为耗散项使用 (v_{a,k+1})，不是 (v_{a,k})。

其一个优点是：

[ \begin{aligned} & \frac12M_av_{a,k+1}^2
\frac12M_av_{a,k}^2\ ={}& h(F^\star_k-F_k)v_{a,k+1} -hD_av_{a,k+1}^2\ &- \frac12M_a (v_{a,k+1}-v_{a,k})^2. \end{aligned} \tag{26} ]

最后两项非正，所以这个虚拟导纳本身的离散实现不会由于显式积分方式凭空加入一项数值能量。

但必须强调：

[ v_a\neq v_m. ]

因此式（26）不能证明真实患者端口被动，也不能证明不弹跳。

黑盒速度执行模型只需要辨识机器人一次
10_tn 的真正角色应该保留。

对于某个任务方向，使用：

[ G_v(p)
e^{-T_dp} \frac{K_v}{1+\tau_vp}. \tag{27} ]

它只描述机器人自身从命令到实际速度的响应，不描述人体。

一阶部分在零阶保持下可精确离散：

[ v_{m,k+1}
a_vv_{m,k} + b_vu_{k-d} + w_{v,k}, \tag{28} ]

其中：

[ a_v=e^{-h/\tau_v}, \qquad b_v=K_v(1-a_v), \tag{29} ]

(d) 是延迟拍数，(w_v) 是模型残差。

更重要的是保存命令队列：

[ Q_k
{ u_{k-d},\ldots,u_{k-1} }. \tag{30} ]

因为两个状态即使当前：

[ F_k=4~\mathrm N, \qquad v_{m,k}=0 ]

完全相同，只要一个 queue 里全是压入命令、另一个全是回撤命令，它们的未来风险完全不同。

因此定义执行状态：

[ \chi_k
\left( F_k,\dot F_k,V_{m,k},Q_k,q_k,s_k \right). \tag{31} ]

这就是论文里所谓 delay-augmented state。

注意：

(T_n) 是机器人/固件级辨识，一次辨识后可以复用。它不是患者级辨识。

核心变化：不用 (K_e,B_e,\tau_e)，改成集合值接触执行契约
这是我现在认为最值得做成论文核心的地方。

不要要求：

[ F_{k+i}
f(K_e,B_e,\tau_e,\ldots). ]

只要求存在一个经过验证的外包络：

[ F_{k+i} \in \mathcal F_{i} ( \chi_k,\zeta_k ), \tag{32} ]

其中：

[ \zeta_k= \begin{bmatrix} u_k\\alpha_k \end{bmatrix}. \tag{33} ]

(\mathcal F_i) 不需要是一个组织模型。它只需要是：

[ \boxed{ \text{真实未来力的保守外集合}. } ]

为了进入 QP，可以把上、下包络表示为分段仿射函数：

[ \overline F_{i|k}(\zeta)
\max_{r\in\mathcal R_i^+} \left[ a_{ir,k}^+ + b_{ir,k}^{+\top}\zeta \right], \tag{34} ]

[ \underline F_{i|k}(\zeta)
\min_{r\in\mathcal R_i^-} \left[ a_{ir,k}^- + b_{ir,k}^{-\top}\zeta \right]. \tag{35} ]

其中：

(i)：未来第 (i) 个采样时刻；
(r)：某个包络面；
(a_{ir,k}^\pm)：由当前实际力、力变化率、实际速度和旧 queue 决定的已知常数；
(b_{ir,k}^\pm)：当前新命令和路径速度对最坏未来力的外包络方向。
于是：

[ \overline F_{i|k}\le F_{\mathrm U} ]

等价于有限组线性行：

[ a_{ir,k}^+ + b_{ir,k}^{+\top}\zeta \le F_{\mathrm U}, \quad\forall r. \tag{36} ]

同理：

[ a_{ir,k}^- + b_{ir,k}^{-\top}\zeta \ge F_{\mathrm L}, \quad\forall r. \tag{37} ]

最终全部写成：

[ \boxed{ A_{F,k} \begin{bmatrix} u\\alpha \end{bmatrix} \le b_{F,k}. } \tag{38} ]

这就是新的 force viability rows。

关键在于：

[ \boxed{ \text{理论只依赖“包络覆盖真实系统”， 不依赖包络来自哪个生物力学模型。} } ]

你可以有三种实现：

第一种最保守，直接从大量接触实验建立：

[ (\chi,u,\alpha) \rightarrow \Delta F_{\max/\min} ]

的 outer table。

第二种稍微结构化，用机器人 (T_n+Q_k) 预测真实剩余运动，再只使用一两个单边接触响应上界，而不是完整 (K_e,B_e,\tau_e)。

第三种以后可以做在线 set-membership tightening：从一个通用保守 envelope 开始，只在新的数据能够确定性证明更紧时收缩包络。

论文理论对三种实现完全一致。

这比把组织刚度作为论文前提更可转移。

为什么一点环境假设都没有是不可能的
可以直接在论文给一个“不可能性命题”。

命题： 若允许环境在下一采样周期产生任意大的正力增量，且该增量与机器人输入无关，则不存在任何控制器能够保证有限的 (F_{\mathrm U})。

证明非常简单。对任意控制器 (\pi)，任意当前状态 (F_k<F_{\mathrm U})，选择一个允许的环境，使：

[ F_{k+1}
F_{\mathrm U}+1. ]

则无论控制器输出什么，都违反上界。

因此：

[ \boxed{ \text{安全保证不可避免地需要一个有限环境响应类；} } ]

创新不是消灭这个假设，而是把它降到最弱的、直接可验证的集合外包络，而不是精确患者参数。

首触与稳定扫描必须分开
对任意未知刚度表面，在非零接近速度和有限执行延迟下，不能证明首次接触峰值为零。

因此理论应该明确从 capture condition 开始。

定义：

[ \mathcal C_{\rm cap}
\left{ \chi: F_{\mathrm L} \le F\le F_{\mathrm U}, \quad \text{future envelope feasible} \right}. \tag{39} ]

接近阶段只需：

很低接近速度；
明确首触 force limit；
一旦触碰即停止路径推进；
等 queue/实际速度进入可认证状态后切换到扫描 QP。
论文中的“不弹跳”定理应该表述为：

在已经捕获接触后，系统不发生整体接触丢失与再撞击。

而不是宣称任何首次碰撞都完全无峰值。

核心算法：两层 QP 从“任务追踪”升级为“可执行任务 + 最佳实现”
名义任务如何组成
首先定义路径反馈：

[ V_{\perp,\rm fb}
-K_\perp e_\perp. \tag{40} ]

其中 (e_\perp) 是不包含 Tool-Z 力方向和被释放旋转方向的局部路径误差。

名义 Tool-Z 速度是式（25）的：

[ v_a. ]

名义面内旋转速度稍后定义为：

[ \omega_a. ]

于是：

[ V_0
V_{\perp,\rm fb} + c_zv_a + c_\theta\omega_a. \tag{41} ]

路径 feedforward 是：

[ V_{\rm ff}
V_{\rm path,ff}. ]

最终名义任务：

[ \boxed{ V_{\rm cmd}(\alpha)
V_0+\alpha V_{\rm ff}. } \tag{42} ]

注意这里没有再用一个不可解释的固定 TFF selection matrix 去硬拼“position/force axes”。

Tool-Z 和 (\theta) 是软交互 intent；其余路径方向由几何反馈维持。

第一层 QP 的变量到底是什么
定义：

[ \boxed{ x_1= \begin{bmatrix} u\ \alpha\ w \end{bmatrix}. } \tag{43} ]

维度：

[ u\in\mathbb R^8, \qquad \alpha\in\mathbb R, \qquad w\in\mathbb R^6. ]

其中：

(u)：最终候选 rail + joint velocity command；
(\alpha)：路径速度缩放；
(w)：六维 task-space velocity slack。
当前代码中 (w) 已经存在。它不需要改成“force slack”。

任务约束写为：

[ J_{\rm task}u + V_{\rm aff}
V_0
\alpha V_{\rm ff}
w. \tag{44} ]

这里：

(J_{\rm task})：现有第一层使用的 task Jacobian；
(V_{\rm aff})：不属于本拍控制变量但已经实际发生的运动，例如当前 rail actual-velocity contribution；
(V_0+\alpha V_{\rm ff})：希望的 TCP 任务；
(w)：做不到多少。
等价：

[ J_{\rm task}u
\alpha V_{\rm ff}
w
V_0-V_{\rm aff}. \tag{45} ]

这和你目前代码：

[ J_{\rm task}u-w=b_{\rm task} ]

完全同构，只多了一个 (\alpha) 列。

第一层优化什么
定义任务归一化矩阵：

[ S_w= \operatorname{diag} ( s_{v,1}^{-1},\ldots,s_{\omega,3}^{-1} ). \tag{46} ]

这是必要的，因为：

平移 residual 单位是 (\mathrm{m/s})；
旋转 residual 单位是 (\mathrm{rad/s})。
不要直接拿两个量数值相加。

第一层目标：

[ \boxed{ \min_{u,\alpha,w} \quad \frac12 w^\top S_w^\top Q_wS_ww + \frac{q_\alpha}{2}(1-\alpha)^2. } \tag{47} ]

其中：

[ Q_w
\operatorname{diag} ( q_{p,1},\ldots, q_{F}, q_\theta ). ]

建议的含义而不是固定数值是：

[ q_{\rm path,\perp}

q_F

q_\theta. \tag{48} ]

也就是说：

横向几何偏离最不希望发生；
精确力中心允许适度让步；
图像/力矩旋转最柔软。
但任何这些权重都不能放松硬安全约束。

路径速度项：

[ q_\alpha(1-\alpha)^2 ]

让系统尽量保持全速扫描。

因此：

[ \alpha\downarrow ]

不是外部规则触发，而是 QP 在接触和执行能力变紧时自然选择的结果。

第一层硬约束
首先保留你代码已有的：

[ u_{\min,k}\le u\le u_{\max,k} \tag{49} ]

其中已经可以合并：

velocity；
acceleration；
jerk；
joint wall；
rail wall。
加速度：

[ -a_{\max}h \le u-u_{k-1} \le a_{\max}h. \tag{50} ]

jerk：

[ -j_{\max}h^2 \le u-2u_{k-1}+u_{k-2} \le j_{\max}h^2. \tag{51} ]

碰撞 CBF：

[ \nabla d_\ell(q)^\top u \ge -\gamma_\ell \left( d_\ell-d_{\ell,\min} \right). \tag{52} ]

路径速度：

[ 0\le\alpha\le1. \tag{53} ]

最重要的是新加入：

[ \boxed{ A_{F,k} \begin{bmatrix} u\\alpha \end{bmatrix} \le b_{F,k}. } \tag{54} ]

它编码全部预测时刻：

[ F_{\mathrm L} \le F_{k+i} \le F_{\mathrm U}. \tag{55} ]

这个约束：

[ \boxed{\text{绝对没有 slack}.} ]

因此，如果名义导纳要求：

[ v_a=-30~\mathrm{mm/s} ]

但该动作根据实际 queue 和当前接触状态会导致未来：

[ F<2~\mathrm N, ]

QP 可以输出：

[ v_z^{\rm cmd}=-12~\mathrm{mm/s}. ]

那么：

[ w_z=18~\mathrm{mm/s} ]

正是：

“为了不离开真实力走廊，我没有完全执行追 (4~\mathrm N) 的名义控制器。”

这就是力松弛的数学定义。

不是：

[ F\le6+\delta_F. ]

为什么这就是最干净的 Conditional Filter
不再需要单独创造一套 conditional task filter。

如果名义任务本身满足全部硬约束，并且存在：

[ w=0,\qquad \alpha=1, ]

那么式（47）目标值为：

[ J_1=0. ]

因为目标是非负的，所以这就是全局最优。

因此：

[ \boxed{ \text{名义行为可行} \Rightarrow w^\star=0,\quad \alpha^\star=1. } \tag{56} ]

这意味着：

QP 在经过认证的正常工作区内完全透明。

一旦名义行为不可行，它才进行最小干预。

这比一个手工设计的 gate/filter 状态机在理论上干净得多。

为什么速度缩放真的有用
假设扫描继续前进会增加接触困难。

传统结构只有：

[ \alpha=1 ]

一种选择。

于是为了维持力，Tool-Z 必须提供更大修正。

现在 QP 可以比较两种代价：

[ \text{更大 }|w_z| ]

与

[ 1-\alpha. ]

例如可能选择：

[ \alpha=0.4 ]

而保留一个较温和的法向修正。

物理上它在做：

[ \boxed{ \text{把接触困难转换成时间，而不是转换成更高闭环增益。} } \tag{57} ]

这非常适合机器人超声，因为临床扫描通常关心的是：

[ \text{路径覆盖} ]

而不是必须在精确的绝对时间戳通过每个路径点。

不过应严谨说明：在完全平坦、切向运动与法向接触完全解耦的理想环境里，降低 (\alpha) 不一定改善一个独立的快速法向扰动。它主要帮助：

曲率变化；
组织空间异质性；
摩擦耦合；
机器人构型资源竞争；
扫描自身引起的接触变化。
因此不能宣传为万能“抗振荡参数”。

第一层 QP 的最终形式
[ \boxed{ \begin{aligned} \min_{u,\alpha,w}\quad& \frac12w^\top Q_w'w + \frac{q_\alpha}{2}(1-\alpha)^2 \ \text{s.t.}\quad& J_{\rm task}u -\alpha V_{\rm ff} -w
V_0-V_{\rm aff}, \ & u_{\min,k}\le u\le u_{\max,k}, \ & A_{\rm col,k}u\le b_{\rm col,k}, \ & A_{F,k} \begin{bmatrix} u\\alpha \end{bmatrix} \le b_{F,k}, \ & 0\le\alpha\le1. \end{aligned} } \tag{58} ]

其中：

[ Q_w'=S_w^\top Q_wS_w. ]

这就是论文的第一核心公式。

第二层 QP 的变量
第一层得到：

[ u_1^\star,\qquad \alpha_1^\star,\qquad w_1^\star. ]

第二层不再改变 (\alpha)。

定义：

[ \boxed{ x_2= \begin{bmatrix} u\ \epsilon \end{bmatrix}. } \tag{59} ]

其中 (\epsilon) 是构型偏好的 slack，例如 singularity recovery slack。

第二层首先锁什么
严格锁定第一层 task：

[ \boxed{ J_{\rm task}u
J_{\rm task}u_1^\star. } \tag{60} ]

而路径速度固定为：

[ \alpha=\alpha_1^\star. \tag{61} ]

由于第一层等式为：

[ J_{\rm task}u_1^\star
\alpha_1^\star V_{\rm ff} -w_1^\star
V_0-V_{\rm aff}, ]

第二层保持式（60）和式（61）后自然得到：

[ w_2=w_1^\star. \tag{62} ]

因此第二层不能用更好的 rail 姿态或 manipulability 去“购买”更多第一层任务误差。

为什么还必须重复患者接触约束
这是非常重要的一点。

不能只写：

[ J_{\rm task}u_2
J_{\rm task}u_1. ]

因为 arm 和 rail 的真实动态并不完全一样。

一个简单反例：

[ J= \begin{bmatrix} 1&1 \end{bmatrix}. ]

取：

[ \Delta u= \begin{bmatrix} a\-a \end{bmatrix}. ]

那么：

[ J\Delta u=0. ]

运动学 TCP 命令不变。

但若两个执行通道分别是：

[ G_1(p),\qquad G_2(p), ]

真实运动变化：

[ \Delta V_m
(G_1-G_2)a. \tag{63} ]

一般不为零。

因此：

[ \boxed{ \text{运动学 nullspace} \neq \text{真实执行动态 nullspace}. } ]

所以 QP2 必须再次施加：

[ \boxed{ A_{F,k} \begin{bmatrix} u\\alpha_1^\star \end{bmatrix} \le b_{F,k}. } \tag{64} ]

这就是第二层对患者安全证书的继承。

QP2 优化什么
定义：

[ \begin{aligned} J_2={}& \frac12 (u-u_{\rm pref})^\top R (u-u_{\rm pref}) \ & +\frac12 (u-u_{k-1})^\top S (u-u_{k-1}) \ & +\frac{q_r}{2} (e_r^\top u-v_{r,\rm ref})^2 \ & +\frac12 \epsilon^\top Q_\epsilon\epsilon. \end{aligned} \tag{65} ]

物理意义分别是：

[ u_{\rm pref} : \text{关节中心、manipulability 等构型偏好}; ]

[ R : \text{不同执行器的使用代价}; ]

[ S : \text{命令连续性}; ]

[ e_r^\top u : \text{rail 速度}; ]

[ v_{r,\rm ref} : \text{mid-ranging rail preference}; ]

[ \epsilon : \text{singularity / comfort 等软偏好 slack}. ]

于是：

[ \boxed{ \begin{aligned} \min_{u,\epsilon}\quad& J_2(u,\epsilon) \ \text{s.t.}\quad& J_{\rm task}u
J_{\rm task}u_1^\star, \ & u_{\min,k}\le u\le u_{\max,k}, \ & A_{\rm col,k}u\le b_{\rm col,k}, \ & A_{F,k} \begin{bmatrix} u\\alpha_1^\star \end{bmatrix} \le b_{F,k}, \ & A_{\rm pref,k}u+B_\epsilon\epsilon \ge b_{\rm pref,k}, \ & \epsilon\ge0. \end{aligned}} \tag{66} ]

这是第二核心公式。

nullspace 到底在哪里
理论上，式（60）的所有解都可以写成：

[ u
u_1^\star+N_Tz, \tag{67} ]

其中 (N_T) 的列张成：

[ \ker(J_{\rm task}). ]

因此 QP2 本质是在第一层任务的仿射 nullspace 中找最好构型。

但是代码不应该真的先投影再加速度。

原因是：

[ u_1^\star+N_Tz ]

仍可能违反：

joint box；
acceleration；
jerk；
collision；
force viability；
energy。
所以最干净的方法正是现有架构：

在完整 (u) 空间里优化，用式（60）硬锁 task。

damped nullspace 为什么不是严格保护
如果代码构造：

[ J_\lambda^#
J^\top(JJ^\top+\lambda^2I)^{-1}, \tag{68} ]

[ N_\lambda
I-J_\lambda^#J, \tag{69} ]

当：

[ \lambda>0 ]

时一般：

[ JN_\lambda\neq0. ]

因此 damped projection 只能用于构造：

[ u_{\rm pref}, ]

不能作为“低优先级绝不影响任务”的证明。

真正的证明来自 QP2 的 hard equality（60）。

singularity escape 应该只是 QP2 软偏好
定义：

[ \sigma(q)
\sigma_{\min}(J_a(q)). ]

一种 set-based 约束：

[ \nabla_q\sigma^\top u + \epsilon_\sigma \ge \gamma_\sigma ( \sigma_{\rm safe}-\sigma ), \tag{70} ]

其中：

(\sigma_{\rm safe})：希望恢复的最低 dexterity；
(\gamma_\sigma>0)：恢复速率；
(\epsilon_\sigma\ge0)：做不到时的 slack。
QP2 惩罚：

[ q_\sigma\epsilon_\sigma^2. ]

如果接触任务要求和 singularity escape 冲突：

[ \epsilon_\sigma ]

应增大。

而不是破坏第一层。

你当前代码此前核对过的 sigma_setbased, nullspace_task, manipulability_task, rail_allocator 正适合作为这一层的 building blocks；它们不是新的论文贡献。

与当前 RealUS_playground 的映射
本报告采用你最新指定版本对应的完整提交：

03a871ae17420ef41e5ad538f9ae7ad400a36d5b

你写的：

e5ad538f9ae7ad400a36d5b

是该 SHA 的后缀。

核心变量建议直接对应：

数学量	代码对象
(u)	现有 qdot / rail+arm velocity decision
(w)	现有 QP1 task slack
(J_{\rm task})	现有 J_task
(V_{\rm aff})	rail actual velocity contribution
(\alpha)	新增 path_speed_scale
(s)	新增/重构 path_phase
(A_F,b_F)	新增 force_viability_rows
(Q_k)	delay queue，已有 P4 shield 思想可复用
(T_n)	ICRA_2027 10_tn robot model
(u_{\rm pref})	现有 nullspace/manipulability/posture preference
(\epsilon_\sigma)	现有 singularity slack
(E_k)	可选 actual-port energy state

最重要的代码原则是：

[ \boxed{ \text{QP1 与 QP2 都调用同一个 } \texttt{force_viability_rows()}. } \tag{71} ]

而不是 QP1 安全、QP2 再随便重分配。

超声图像与接触力矩如何只用一个旋转自由度改善声学接触
为什么总力不够
超声探头是有限面积接触。

设探头表面压强为：

[ p(x,y). ]

总法向力：

[ F
\int_{\mathcal A} p(x,y),dA. \tag{72} ]

绕保留旋转轴的接触力矩：

[ \tau_\theta
\int_{\mathcal A} x,p(x,y),dA. \tag{73} ]

所以两个接触都可能满足：

[ F=4~\mathrm N, ]

但一个左右均匀，另一个所有压力集中在探头一侧。

因此：

[ \boxed{ \text{恒定净力不等于良好有限面积接触。} } ]

这也是机器人超声和很多点接触工业 force control 的重要区别。

在理想平面压力模型中：

[ x_{\rm CoP}
\frac{\tau_\theta}{F}. \tag{74} ]

它表示压力中心相对参考点的横向偏移。

严格说，单个力和力矩不能恢复完整压力分布，但至少给出了“一侧加载更重”的机械信息。

Piedra 等近期的 dynamic wrench control 工作同样强调了有限面积探头中 force 与 moment 联合调节的价值；Chatelain 等早期 confidence-driven ultrasound probe control 则说明 ultrasound confidence map 的空间矩可以用于自动调节探头姿态。这里本文的创新不应声称“首次使用图像或力矩”，而是：

把图像/力矩只作为软接触意图，而由同一个执行可行性 QP 判断该姿态动作是否真的允许。

先正确搬移 wrench
假设 F/T 传感器或 TCP wrench 表达在 (T) 点，希望得到探头面参考点 (P)。

设：

[ r_{TP} ]

为从 (T) 到 (P) 的位置向量。

则：

[ f_P=f_T, ]

[ \tau_P
\tau_T-r_{TP}\times f_T \tag{75} ]

（具体符号取决于 (r) 的方向定义，代码必须通过静态载荷检查确定）。

然后：

[ \tau_\theta
a_\theta^\top\tau_P. \tag{76} ]

过去如果把“与法向力垂直的力矩”一律过滤掉，就恰好丢掉了这里需要的有限面积接触信息。

图像给声学接触，力矩给机械接触
从超声近场图像构造一个左右接触不均衡标量。

例如置信图为：

[ C(x,z)\in[0,1]. ]

定义归一化第一矩：

[ e_I
\frac{ \int x C(x,z),dx,dz }{ r_I \int C(x,z),dx,dz+\varepsilon }. \tag{77} ]

其中 (r_I) 是图像半宽，因此 (e_I) 近似无量纲。

同样，把机械压力中心归一化：

[ e_\tau
\operatorname{sat} \left( \frac{\tau_\theta} { (F+\varepsilon_F)r_P } \right), \tag{78} ]

其中 (r_P) 是探头接触面特征半宽。

于是：

(e_I)：声学上哪一边接触差；
(e_\tau)：机械上压力中心偏向哪一边。
这两个信号具有互补性。

图像会受：

声影；
解剖结构；
图像处理噪声；
影响。

力矩则会受：

摩擦；
TCP reference；
工具补偿；
影响。

所以最好不要选其中一个为绝对真值。

不要再辨识一个复杂的“角度—图像—组织”模型
最简洁设计是只做一个符号校准。

定义图像置信权重：

[ 0\le c_I\le1. ]

当图像可信时 (c_I\to1)，图像差时 (c_I\to0)。

组合接触误差：

[ e_c
c_I e_I + (1-c_I)\lambda_\tau e_\tau. \tag{79} ]

其中 (\lambda_\tau) 只是两个无量纲信号之间的相对尺度。

名义旋转速度：

[ \boxed{ \omega_a
-k_\theta s_\theta e_c. } \tag{80} ]

其中：

(k_\theta>0)：最大响应尺度；
(s_\theta\in{-1,+1})：一次几何/小动作标定得到的旋转符号。
这没有患者刚度、黏弹性或角度响应模型。

如果担心复杂表面上图像映射符号发生变化，可以让 (s_\theta) 用小窗口 secant 估计：

[ \hat g_I
\frac{\Delta e_I}{\Delta\theta}, \tag{81} ]

只有当：

[ |\Delta\theta|>\theta_{\rm excite} ]

且图像置信度足够时才更新符号。

这仍然是一个低维局部梯度方向，不是生物力学辨识。

为什么不把图像直接做 QP 硬约束
因为：

[ e_I\neq0 ]

不一定意味着机械接触一定不好。

它可能来自组织纹理或声影。

因此本文建议：

[ \omega_a ]

只进入：

[ V_0 ]

成为 QP1 的软任务。

如果旋转会导致：

[ F_{\rm future}>F_{\mathrm U}, ]

QP通过：

[ w_\theta ]

自动减弱旋转。

所以结构是：

[ \boxed{ \text{image + moment} \rightarrow \omega_a \rightarrow V_{\rm cmd} \rightarrow \mathrm{QP1\ admission}. } \tag{82} ]

而不是：

[ \text{image controller} \rightarrow \text{直接改最终 }V_{\rm cmd} \rightarrow \text{QP 后再 clip}. ]

可以严格保证什么
不能仅凭一个 image feature 和一个 moment 严格证明：

“整个探头足迹每一个位置都完全接触。”

这在信息论上就不充分。

能够比较严谨地写：

在有限面积接触模型下，接触力矩提供压力中心偏移的机械观测；超声 confidence asymmetry 提供声学耦合不均衡观测。所提出的单轴接触调节器在两者符号一致时生成使两种不均衡同时减小的名义旋转方向，而患者端口 QP 保证该旋转不会牺牲已认证的法向力可行性。

这已经足够强，而且不会过度宣称。

(F^\star) 的识别放在哪里
完全可以把：

[ F^\star
\mathcal G(I,\text{view},\text{anatomy}) \tag{83} ]

作为独立高层模块。

Zakeri 等已经展示过图像/视图条件下调整期望交互力的思路；其他超声工作也使用 confidence 或组织变形信息调整目标力。

本文只定义接口：

[ F_{\mathrm L}+\Delta_F \le F^\star \le F_{\mathrm U}-\Delta_F. \tag{84} ]

高层模块无权修改：

[ F_{\mathrm L}, F_{\mathrm U}. ]

所以以后即使你有一个很好的“什么解剖视图需要多少力”的模型，也不用重写底层安全 QP。

Energy tank 是否值得保留：只有一种设计我认为值得
先给结论
如果 energy tank 只是：

(E) 低就增加 (D)；
(E) 低就把 (V_{\rm cmd}) 乘一个系数；
QP 后再 clip；
或者还用命令速度计算患者功率；
那么：

[ \boxed{\text{建议删掉。}} ]

它只会让论文更复杂。

但是如果做成：

[ \boxed{ \textbf{actual patient-port energy budget} + \textbf{delay committed-work reserve} } ]

那么我认为值得保留，因为它能够给出一个和患者刚度模型无关的漂亮定理。

这也是它和 force viability 的互补价值。

为什么 Lee 不需要辨识组织模型
Lee 等 BEFM 的关键不是预测一个 Kelvin–Voigt 环境。

它通过：

代理速度；
实际机器人速度；
实际外部 wrench；
直接判断代理和真实系统之间是否存在不良能量流。

因此它的核心思路本身就是 environment-model-independent。

不过 Lee 的实际控制接口和动力学结构与 RM75 不同，尤其其控制器可以直接作用于机器人动力学/力矩层，所以你不能把其 passivity proof 原样移植。

你可以继承的思想是：

[ \boxed{ \text{稳定性监督应尽量基于实际端口功率， 而不是患者参数。} } ]

患者实际端口
选择探头面上的固定刚体点 (P)。

定义环境作用于探头的 wrench：

[ W_P
\begin{bmatrix} f_P\ \tau_P \end{bmatrix}. \tag{85} ]

定义同一物理时刻该点的实际 twist：

[ V_{P,m}
\begin{bmatrix} v_{P,m}\ \omega_{P,m} \end{bmatrix}. \tag{86} ]

机械功率：

[ \boxed{ P_P
W_P^\top V_{P,m}. } \tag{87} ]

如果 (W_P) 定义为 environment-on-probe：

(P_P>0)：患者向机器人输入机械能；
(P_P<0)：机器人向患者输出机械能。
注意这里：

[ F^\top v+\tau^\top\omega ]

的每一项都是瓦特，所以完整六维空间功率天然量纲一致。

这也是为什么即使你只主动调 Tool-Z 和一个旋转轴，如果要声称完整患者端口被动，功率仍然应使用六维实际 wrench–twist。

扫描切向摩擦也是患者机械功的一部分。

能量账本
定义：

[ E_k\in[E_{\min},E_{\max}] ]

为当前允许的机械能预算。

每拍使用实际端口工作：

[ \mathcal W_{P,k}
\int_{t_k}^{t_{k+1}} W_P(t)^\top V_{P,m}(t),dt. \tag{88} ]

更新：

[ \boxed{ E_{k+1}
\min \left( E_{\max}, E_k+\underline{\mathcal W}_{P,k} \right). } \tag{89} ]

其中：

[ \underline{\mathcal W}{P,k} \le \mathcal W{P,k} ]

是考虑同步、噪声和积分误差后的保守下界。

非常重要：

[ \boxed{ \text{不能在 }E_{\min}\text{ 处强行 clip。} } ]

如果真实更新低于 (E_{\min})，证书已经失败，不能靠把软件变量写回 (E_{\min}) 修复物理事实。

上限 clip 是可以的，因为它只是丢弃多余储能。

为什么这里不再需要 (K_e)
真正漂亮的地方在已承诺能量 reserve。

假设控制器已经知道机器人旧命令在停止前最多还会产生多少绝对实际位移：

[ d_{j,k}^{\rm old}, \qquad j=1,\ldots,6. ]

其中前三个是米，后三个是弧度。

这些量来自：

robot-only (T_n)；
delay queue；
stop_dx_ub / stop-angle upper bound；
而不是人体模型。

再规定患者端口允许工作域中的 wrench hard envelope：

[ |W_{P,j}| \le \bar W_j. \tag{90} ]

这里 (\bar W_j) 是安全/操作规格，不是辨识组织刚度。

那么旧 queue 最坏还可能输出的机械功：

[ \boxed{ R_{E,k}^{\rm old}
\sum_{j=1}^{6} \bar W_jd_{j,k}^{\rm old}. } \tag{91} ]

因为：

[ \left| \int W_j,dq_j \right| \le \bar W_j \int|dq_j|. ]

这非常保守，但：

[ \boxed{\text{完全不需要患者刚度或黏弹参数。}} ]

这比：

[ F_{\rm future}=F+K_e\Delta x ]

更适合作为环境不确定时的能量 reserve。

当前候选命令也要预算
设当前候选：

[ \zeta= \begin{bmatrix} u\\alpha \end{bmatrix}. ]

通过 robot execution contract 可以得到候选动作在“下一次还能重新决策之前”造成的绝对端口运动上界。

假设表示为：

[ d_{\rm new} \le d_0+ |G_\zeta\zeta|. \tag{92} ]

引入 epigraph 变量：

[ \rho\ge0 ]

满足：

[ \rho \ge G_\zeta\zeta, ]

[ \rho \ge -G_\zeta\zeta. \tag{93} ]

则：

[ |G_\zeta\zeta|\le\rho. ]

候选最坏机械工作：

[ R_E^{\rm new}
\bar W^\top(d_0+\rho). \tag{94} ]

因此 QP 只需增加：

[ \boxed{ R_{E,k}^{\rm old} + \bar W^\top(d_0+\rho) \le E_k-E_{\min}. } \tag{95} ]

以及式（93）。

它全部是线性约束。

这比：

[ W^\top J u\ge b ]

更加鲁棒，因为它没有假设：

[ V_{\rm actual}=Ju. ]

也没有假设下一拍 wrench 保持当前值。

代价是更保守。

Energy-QP 的变量
如果启用 energy 模块，QP1 变量扩展为：

[ \boxed{ x_1= \begin{bmatrix} u\ \alpha\ w\ \rho \end{bmatrix}. } \tag{96} ]

(\rho) 没有独立控制意义，只是绝对值线性化的 auxiliary variable。

QP2 也必须重复同样 energy row。

它到底证明什么
由式（89）：

[ E_{k+1} \le E_k+ \mathcal W_{P,k}. ]

累计：

[ E_N-E_0 \le \sum_{k=0}^{N-1} \mathcal W_{P,k}. ]

又因为：

[ E_N\ge E_{\min}, ]

所以：

[ \boxed{ \sum_{k=0}^{N-1} \mathcal W_{P,k} \ge E_{\min}-E_0. } \tag{97} ]

因此机器人从患者端口净输出的机械功满足：

[ \boxed{ W_{\rm robot\rightarrow patient}^{\rm net} \le E_0-E_{\min}. } \tag{98} ]

这就是一个明确、漂亮、与组织刚度无关的端口能量定理。

Secchi/Ferraguti 和 Benzi/Secchi 已经证明 energy tank/passivity constraint 可以进入凸优化，因此“energy in QP”本身不是创新 [Secchi & Ferraguti, 2019; Benzi & Secchi, 2021]。

你的区别应写成：

现有优化式 passivity 方法通常在任务速度可被低层较好实现的假设下构造能量约束；本文对具有显著输入延迟的黑盒速度接口机器人直接使用实际患者端口功率，并把尚未执行的命令转化为 committed-work reserve，从而使端口能量证书作用于实际物理运动而非理想命令。

它不能证明什么
即使：

[ E_k\ge E_{\min} ]

始终成立，也不能推出：

[ F\le F_{\mathrm U}. ]

一个无限刚性的表面可以在极小位移下产生极大峰值力，但机械工作仍很小。

所以：

[ \boxed{ \text{force viability} \neq \text{passivity}. } ]

前者保护峰值/接触：

[ F_{\mathrm L}\le F\le F_{\mathrm U}. ]

后者保护累计机械能：

[ W_{\rm out}^{\rm net}\le E_{\rm budget}. ]

也不能写：

[ \text{passive}\Rightarrow\text{no bouncing}. ]

一个欠阻尼但被动的系统仍然可以振荡。

是否建议正文保留
我的最终建议是：

[ \boxed{ \textbf{主方法不依赖 energy； 正文保留一个 compact energy extension 和一个独立 theorem。} } ]

如果实验发现它几乎不激活、没有额外价值，就把它移到 appendix。

如果它在未知、黏弹或高刚度接触上能捕获 force corridor 没有直接观察到的危险能量行为，再把它升为主要贡献之一。

这样不会为了“看起来理论很多”而强行增加一个 tank。

理论结果：真正可以写进 IJRR 的几个定理
不可能性结果：未知环境不能无假设保证
命题 A： 若环境的未来接触力增量没有有限外包络，则任何因果控制器都无法保证有限双边力走廊。

这不是弱点，而是用来解释为什么本文不辨识组织模型，却仍然需要一个最弱 set-valued contract。

它帮助回答审稿人：

“既然你说环境未知，为什么还能证明？”

答案是：

未知的是模型结构和参数，不是允许产生任意无限响应。我们只假设真实系统属于经过验证的响应外集合。

条件透明性
设第一层名义任务存在一个可行解：

[ w=0,\qquad \alpha=1, ]

并满足所有硬约束。

因为式（47）的两项均非负，所以：

[ J_1=0. ]

不存在更小目标。

因此：

[ \boxed{ w_1^\star=0,\qquad \alpha_1^\star=1. } \tag{99} ]

即：

安全约束没有激活时，新方法完全不改变 nominal controller。

这正是你想要的“柔顺手感平时不被安全器破坏”。

Force corridor 前向不变性
假设集合外包络满足：

[ \underline F_{i|k}(\zeta_k) \le F_{k+i} \le \overline F_{i|k}(\zeta_k) \tag{100} ]

对真实系统始终成立。

QP1/QP2 强制：

[ \underline F_{i|k} \ge F_{\mathrm L}, ]

[ \overline F_{i|k} \le F_{\mathrm U}. ]

则：

[ F_{\mathrm L} \le F_{k+i} \le F_{\mathrm U} \tag{101} ]

对所保护预测 horizon 成立。

如果 horizon 末端状态进入一个经过验证的 backup invariant set：

[ \chi_{k+N} \in\mathcal K_B, \tag{102} ]

并且对任意：

[ \chi\in\mathcal K_B ]

存在 backup action 使：

[ \chi^+\in\mathcal K_B, \tag{103} ]

那么 receding-horizon 可行性递归成立。

因此从：

[ \chi_0\in\mathcal K ]

出发：

[ \boxed{ F_k\in[F_{\mathrm L},F_{\mathrm U}], \qquad\forall k. } \tag{104} ]

这比仅使用：

[ F_{k+1} ]

的一拍 CBF 更适合你的输入 delay。

正力下界与无整体再撞击
对于无粘附单边接触：

[ F\ge0,\qquad g\ge0,\qquad Fg=0, \tag{105} ]

其中 (g) 是接触间隙。

若：

[ F_{\mathrm L}>0 ]

且式（104）成立，则：

[ F_k>0 ]

所以：

[ g_k=0. ]

因此在模型有效域中：

[ \boxed{ \text{捕获后不发生整体 separation， 因此没有 separation–reimpact 型弹跳。} } \tag{106} ]

这里必须加“整体”。

它不保证：

某个探头边缘不瞬间卸载；
压力分布完全均匀；
力在走廊里一点都不振荡。
这些正是 image + moment 任务要改善但不作为 hard theorem 的部分。

QP2 对 QP1 不劣化
QP1：

[ J_{\rm task}u_1^\star -\alpha_1^\star V_{\rm ff} -w_1^\star
b. \tag{107} ]

QP2 强制：

[ J_{\rm task}u_2^\star
J_{\rm task}u_1^\star ]

和：

[ \alpha_2=\alpha_1^\star. ]

因此：

[ w_2=w_1^\star. ]

于是第一层目标中关于 task residual 的部分严格不变。

路径速度也不变。

所以：

[ \boxed{ J_1(u_2^\star,\alpha_1^\star)
J_1^\star. } \tag{108} ]

只要 QP2 重施所有患者硬约束，最终命令也仍满足：

[ \boxed{ \mathcal C_{\rm patient}(u_2^\star,\alpha_1^\star)=\text{true}. } \tag{109} ]

这就是：

[ \boxed{ \text{task non-degradation} + \text{certificate inheritance}. } ]

这是你两层 QP 最值得正式证明的结果。

QP2 为什么存在可行解
QP1 的：

[ u_1^\star ]

本身就满足：

task lock；
hardware；
collision；
force viability；
energy（若开启）。
因此 QP2 至少可以取：

[ u=u_1^\star. ]

对于 singularity 等 soft preference，只需要选择足够大的：

[ \epsilon. ]

所以 QP2 存在可行点。

因此：

[ \boxed{ \text{QP1 feasible} \Rightarrow \text{QP2 feasible} } \tag{110} ]

只要所有新增 QP2-only 约束确实具有 slack 或只是 cost。

这非常适合论文。

实际患者端口耗散性
如果启用上一节的 energy 约束，并且：

actual wrench/twist 同步有效；
energy increment lower bound 有效；
committed-work reserve 有效；
QP2 重复 energy constraint；
QP 后没有再修改命令；
则式（98）成立。

这是一个与 force theorem 独立的定理。

为什么没有必要再设计一个“大一统 Lyapunov 函数”
不需要为了“理论高度”强行构造：

[ V= V_F+V_I+V_\theta+V_q+V_s+E ]

然后声称所有项共同下降。

你的系统有：

moving path；
unknown environment；
changing image；
delayed actuator；
hierarchical inequality constraints。
要求这样一个标量函数每拍严格下降，既不自然，也可能逼出一堆没有物理意义的权重。

真正漂亮的理论分解是：

[ \boxed{ \begin{array}{ll} \text{force safety} & \rightarrow \text{robust viability/invariance}, \[1mm] \text{recursive execution ability} & \rightarrow \text{terminal recoverability}, \[1mm] \text{hierarchy} & \rightarrow \text{lexicographic non-degradation}, \[1mm] \text{patient energy} & \rightarrow \text{dissipativity/passivity}, \[1mm] \text{image/moment} & \rightarrow \text{soft contact-quality descent}. \end{array} } \tag{111} ]

每个数学工具只证明它真正能证明的性质。

这比“所有稳定性都塞进 Lyapunov”更像成熟的 IJRR 论文。

实现、实验与附录级推导：把论文真正落到现有代码上
最小执行架构
最终软件结构应当非常简单：

text
复制
F/T + robot state + ultrasound
            │
            ▼
      synchronized state χk
            │
     ┌──────┴────────┐
     │               │
     ▼               ▼
nominal intent   execution/contact
 generation      certificate builder
     │               │
     │          AF, bF
     │          [AE, bE optional]
     │               │
     └──────┬────────┘
            ▼
          QP1
      x1=[u, α, w]
            │
     u1*, α1*, w1*
            │
            ▼
          QP2
      x2=[u, ε]
   lock Jtask u
   repeat patient rows
            │
            ▼
        ufinal
            │
            ▼
           RM75
            │
            └────► delay queue
最关键的软件原则：

任何会影响最终命令的限制，必须在最终 QP 内部。

删除：

QP 后 force clip；
QP 后 TDPA 改速度；
QP 后图像 controller 再补一项；
QP 后独立 rail 修正。
否则：

[ u_{\rm sent} \neq u_{\rm certified}. ]

前面所有证明作废。

force_viability_rows() 应该输出什么
建议接口：

python
复制
rows = force_viability_rows(
    execution_state=chi_k,
    path_feedforward=V_path_ff,
    force_bounds=(F_lower, F_upper),
)
返回：

python
复制
A_force_u
A_force_alpha
b_force
margin_upper
margin_lower
certificate_valid
数学上：

[ A_{F,u}u+A_{F,\alpha}\alpha \le b_F. \tag{112} ]

这样 QP builder 完全不关心：

包络来自 (T_n+K_e)；
数据表；
set-membership；
backup rollout。
这让理论与具体辨识实现解耦。

如何获得这个 envelope 而不做患者级系统辨识
建议的实验顺序不是“给每个患者辨识一个 (K_e)”。

只做两类合同。

机器人执行合同一次完成：

[ u,Q_k \rightarrow V_{\rm actual}\text{ tube}, ]

[ u,Q_k \rightarrow \Delta x_{\rm stop}^{UB}, ]

[ u_\theta,Q_k \rightarrow \Delta\theta_{\rm stop}^{UB}. ]

这里使用：

03/09；
10_tn；
step stop tests；
各典型姿态重复；
即可。

这是机器人/firmware 性质。

接触恢复合同只要求直接回答：

[ \text{如果现在停止推进并启动 backup， 力最多还会上升多少？} ]

定义：

[ R_F^+(\chi)
\sup_{\rm allowed~contacts} \left( F_{\rm peak}^{\rm backup}-F_k \right). \tag{113} ]

同理：

[ R_F^-(\chi)
\sup \left( F_k-F_{\min}^{\rm backup} \right). \tag{114} ]

状态可以仅使用：

[ \chi= ( F, \dot F, v_m, Q ). ]

这比辨识：

[ K_e,B_e,\tau_e ]

简单得多。

你甚至可以把它做成单调 LUT / convex envelope：

[ R_F^+
R_F^+ ( F,\dot F^+,|v_m|,E_Q ). \tag{115} ]

这是一张“恢复能力图”，不是一个组织模型。

要在新患者上作严格 deterministic guarantee，仍然必须假设该患者位于认证 contact class 内；任何论文都逃不掉这一点。可以通过：

覆盖软、中、硬 phantom；
骨性结构模拟物；
多种预载；
多种扫描速度；
得到一个相当保守的 universal envelope。

新患者在线数据只允许：

[ \text{缩小保守性} ]

不能在没有证明的情况下扩展安全域。

为什么 K_e 可以从核心公式里删除
旧思路：

[ F_{\rm future}^{UB}
F_k+ K_e \Delta x_{\rm committed}^{UB}. \tag{116} ]

它隐含：

线性弹性；
没有 relaxation；
没有 hysteresis；
没有 tangential coupling；
(K_e) 在该窗口内不变。
新思路直接使用：

[ \boxed{ F_{\rm future}^{UB}
F_k+ R_F^+(\chi,u,\alpha). } \tag{117} ]

这样：

黏弹；
滞后；
摩擦；
robot delay；
queue；
都只需要被外包络覆盖。

这正是减少患者辨识工作的核心。

但为什么 T_n 仍然值得留
因为 T_n 不是环境模型。

它让你知道：

[ \boxed{ \text{这个命令什么时候会真正生效， 以及停止时还有多少运动没执行完。} } ]

如果未来换患者：

[ T_n ]

基本不需要重新辨识。

如果未来换更快 Tool-Z motor：

[ T_n ]

改变，其他论文框架不变。

这也自然支持你说的未来高速小行程执行器。

假设以后新增：

[ u_f ]

作为高速 Tool-Z actuator。

QP 变量只需扩展：

[ u
\begin{bmatrix} u_{\rm RM75}\u_f \end{bmatrix}. \tag{118} ]

该执行器有自己的快速 execution contract。

于是 QP 可以自动选择：

慢而大范围的 RM75；
快而小行程的 Z actuator。
而不是重新设计力控制理论。

这正体现：

[ \boxed{ \text{QP1 = certified achievable task}; \qquad \text{QP2 = best realization}. } ]

实验应该围绕“为什么这个核心思想有意义”设计
不要做二十个 controller combination。

最有说服力的主消融只有四组：

方法	延迟可执行约束	力走廊	路径自动调速	actual-port energy
Baseline admittance	否	否	否	否
Instantaneous corridor	否	是	否	否
Proposed viability HQP	是	是	是	否
Proposed + energy	是	是	是	是

CDYOB 放额外消融：

[ \text{Proposed nominal admittance} ]

vs

[ \text{Proposed + CDYOB nominal layer}. ]

这样能够证明：

提升究竟来自执行可行性，还是只是某个更好的 force controller。

主指标应该包括：

[ F_{\max}, \quad F_{\min}, \quad T_{\rm out\ of\ corridor}, ]

[ N_{\rm separation}, \quad N_{\rm reimpact}, ]

[ \operatorname{RMS}(F-F^\star), ]

[ \operatorname{RMS}(e_{\rm path}), ]

[ \operatorname{mean}(\alpha), ]

[ \text{scan completion time}, ]

[ \operatorname{RMS}(e_I), \quad \operatorname{RMS}(\tau_\theta), ]

以及：

[ \int P_{\rm out}^+dt ]

和 energy minimum（若启用）。

最重要的是同时报告：

[ |w| ]

因为它正是论文的新含义：

硬件无法安全实现的 nominal interaction request 被主动放弃了多少。

最值得展示的一张图
我认为 IJRR 稿件最关键的一张实验图应该是同一段困难接触下的六条时间曲线：

[ F(t), ]

[ v_{z,\rm nominal}(t), ]

[ v_{z,\rm QP}(t), ]

[ v_{z,\rm actual}(t), ]

[ \alpha(t), ]

[ R_F^+(t). ]

Baseline 中可以清楚看到：

[ F\text{ error} \rightarrow v_{\rm nominal}\text{ 大修正} \rightarrow V_{\rm actual}\text{ 滞后} \rightarrow F\text{ 继续越过}. ]

Proposed 中：

[ R_F^+\uparrow \rightarrow \alpha\downarrow \rightarrow w_z\neq0 \rightarrow \text{拒绝不可执行修正} \rightarrow F\text{ 留在 corridor}. ]

这比展示十张 Bode plot 更直接证明核心贡献。

审稿人最可能的质疑与应该提前回答的方式
“这是不是普通 predictive safety filter?”

部分思想当然与 reference governor、predictive safety filter、robust invariant set 有关系。不能声称这些数学工具是新发明。

你的贡献必须落在：

[ \boxed{ \text{黑盒速度执行延迟} + \text{非参数患者接触外包络} + \text{force-corridor relaxation} + \text{time-free scanning} + \text{two-stage HQP certificate inheritance}. } ]

特别是第二层真实安全继承问题，是许多纯运动学 HQP 不会处理的。

“未知环境怎么能有 deterministic guarantee?”

回答：

不假设精确环境模型；假设的是经过验证的集合值接触响应包络。没有任何有限响应假设时，有限力界在理论上不可能保证。本文给出了该不可能性，并把所需环境信息降到直接可验证的 one-sided recovery envelope。

“为什么不是把 force controller 增益调小?”

因为固定低增益会在所有区域降低性能。

本文满足：

[ \text{nominal safe} \Rightarrow w=0,\alpha=1. ]

只在实际可行域收缩时干预。

“energy tank 是否只是装饰?”

如果实验没有显示它捕获额外风险，放 appendix。

如果使用 actual-port ledger + committed-work reserve，则有式（98）的明确耗散定理，不是装饰。

“image + moment 是否真的保证声学接触?”

不要过度声称。

严格保证的是法向接触可行性。

image/moment 提供有限面积声学接触的低优先级优化方向。

附录应给出的矩阵形式
第一层变量：

[ x_1= \begin{bmatrix} u\\alpha\w \end{bmatrix}. ]

标准 QP：

[ \min_{x_1} \frac12x_1^\top H_1x_1 + g_1^\top x_1. \tag{119} ]

Hessian：

[ H_1= \begin{bmatrix} 0&0&0\ 0&q_\alpha&0\ 0&0&Q_w' \end{bmatrix}. \tag{120} ]

线性项：

[ g_1= \begin{bmatrix} 0\ -q_\alpha\ 0 \end{bmatrix}. \tag{121} ]

常数 (q_\alpha/2) 可以省略。

任务等式：

[ A_{\rm eq,1}
\begin{bmatrix} J_{\rm task}&-V_{\rm ff}&-I_6 \end{bmatrix}, \tag{122} ]

[ b_{\rm eq,1}
V_0-V_{\rm aff}. \tag{123} ]

Force rows：

[ A_{\rm force,1}
\begin{bmatrix} A_{F,u}&A_{F,\alpha}&0 \end{bmatrix}. \tag{124} ]

这最后一列必须为零：

[ \boxed{ w\text{ 不能买掉 force safety}. } ]

第二层的 task lock：

[ A_{\rm eq,2}
\begin{bmatrix} J_{\rm task}&0 \end{bmatrix}, \tag{125} ]

[ b_{\rm eq,2}
J_{\rm task}u_1^\star. \tag{126} ]

Force rows：

[ A_{\rm force,2}
\begin{bmatrix} A_{F,u}&0 \end{bmatrix}, \tag{127} ]

右端：

[ b_{\rm force,2}
b_F
A_{F,\alpha}\alpha_1^\star. \tag{128} ]

这几条足以让实现和理论完全一致。

附录应明确列出的最小实验合同
最终真正“必须补”的东西可以收缩到非常少。

机器人级，一次完成：

[ T_n^z, \quad T_n^\theta, ]

[ T_d^{UB}, ]

[ \Delta x_{\rm stop}^{UB}, \quad \Delta\theta_{\rm stop}^{UB}, ]

[ \bar e_V. ]

这些在换患者时不需要重做。

接触类级，不辨识患者模型：

[ R_F^+(\chi,u,\alpha), \qquad R_F^-(\chi,u,\alpha). ]

只需要证明它们是 future force excursion 的外包络。

能量扩展才需要：

[ \bar W_j, \quad \underline{\mathcal W}_{P,k}, \quad \text{wrench–twist timestamp bound}. ]

图像旋转：

只需：

图像接触指标定义；
一个旋转符号标定；
image latency；
TCP 到 probe-face reference point 的几何标定。
不需要辨识患者：

[ K_e,B_e,\tau_e. ]

核心参考文献脉络
本文理论关系应主要定位在以下文献链上，而不是把所有文章都写成方法组件。

Secchi 与 Ferraguti 的 energy-aware/passivity optimization 工作说明“最接近期望行为的被动实现”可以表述为凸优化；Benzi 与 Secchi 进一步说明 passivity 和 set-based/CBF 约束可以共存于优化控制中。它们支持 energy-QP 的数学形式，但其低层执行假设不能直接搬到 RM75。

Lee 等 2024 的 BEFM 最有价值的启示是：

[ V_a-V_m ]

这一代理—真实运动失配本身会破坏理想能量性质，而稳定性监督可以依赖实际 wrench 和运动，而不依赖人体参数。

De Stefano 等说明延迟必须进入机械能分析，因为虚拟系统的良好性质不能自动穿过通信和执行延迟。

Samuel 等的 perturbation-robust/CDYOB 工作支持对工业机器人黑盒内运动环辨识 (T_n)，但其作用应定位为 nominal performance recovery，而非 force-envelope safety certificate。

Keemink 等与 Dimeas–Aspragathos 解释了 admittance 中低阻尼良好手感与刚性接触鲁棒性之间的经典冲突，是本文“不再让 nominal force controller 独自负责所有安全”的直接动机。

Escande 等的 HQP/task-priority 理论支持第二层通过等式保持第一层最优任务，而不是用“超大权重”近似优先级。

Chatelain 等的 confidence-driven ultrasound probe control、Piedra 等的 force/moment probe orientation，以及 Zakeri 等 image-conditioned force target 工作说明图像、接触力矩和高层期望力都已有先例，因此本文不能把这些信号本身包装成创新；真正的贡献应是它们被如何安全地注入一个有执行延迟的 patient-port viability filter。

最终论文应该怎样一句话概括
我会把整个方法最终压缩成：

[ \boxed{ \begin{array}{c} \text{柔顺力/图像/路径只提出“希望做什么”}\ \downarrow\ \text{QP1 判断硬件和未知接触环境下“现在安全地做得了多少”}\ \downarrow\ \text{QP2 在不改变这个结果的情况下选择最佳 robot/rail realization} \end{array}} ]

更正式地：

[ \boxed{ \textbf{Nominal intent} \rightarrow \textbf{execution-contact admissibility} \rightarrow \textbf{hierarchical realization}. } \tag{129} ]

其中最核心的新理论对象不是 energy tank，也不是 CDYOB，而是：

[ \boxed{ A_{F,k} \begin{bmatrix} u\\alpha \end{bmatrix} \le b_{F,k} } ]

这个与具体患者模型解耦、但显式包含执行延迟和已发送命令历史的接触可行性契约。

如果这个 contract 能够从你的 ICRA_2027 数据和少量新增接触恢复实验中做成可靠、低维、保守而不过度保守的 convex envelope，那么这条线确实具有比“再设计一个导纳参数调度器”高得多的理论和工程价值。

它把你原本零散的几个好想法统一起来了：

[ \boxed{ \begin{aligned} \text{不要精确恒力} &\rightarrow F^\star\text{ 是软 intent},\ \text{力一定有界} &\rightarrow F\text{ corridor 是硬 viability constraint},\ \text{硬件跟不上} &\rightarrow w\text{ 吸收不可执行 task},\ \text{扫描可以慢} &\rightarrow \alpha\text{ 自动下降},\ \text{图像/力矩改善接触} &\rightarrow \omega_a\text{ 是低优先级 intent},\ \text{构型还要优化} &\rightarrow \mathrm{QP2}\text{ 锁任务并重复患者约束},\ \text{未知组织} &\rightarrow \text{只要求 set-valued response envelope},\ \text{还想有能量理论} &\rightarrow \text{actual-port energy + committed-work reserve}. \end{aligned}} \tag{130} ]

这比前面“给每个风险单独再加一个控制器”的版本更接近一篇真正有主张、有边界、有证明对象的 IJRR 论文。
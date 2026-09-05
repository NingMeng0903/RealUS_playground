如果我的思路是吧力传感器的力作为任务层级送入qp然后设计能量约束qp共享 6 维罐；账本 E_{k+1} = E_k + Δt·W_Pᵀ V_{P,m}（用 T_n 延迟队列对齐）；先 shadow 记录 P_port、P_mis=(V_a−V_m)ᵀW 与假设约束的修正量。

已承诺能量 reserve：RM75 命令 28–40 ms 后才起作用，先用延迟队列 + T_n 算 ΔE_committed^UB（未来若干拍可能从端口取走的最大能量），只有 E_k − E_min ≥ ΔE_committed^UB 才允许继续压入；同理首触/过力用 F_future^UB = F_k + K_e·Δx_committed^UB（这是 10_tn、delay queue、stop_dx_ub 真正发挥价值的地方）。
确认符号后在最终 QPIK 加线性硬约束 W_Pᵀ J_P q̇ ≥ (E_min−E_k+ΔE_committed^UB)/Δt + ‖W_P‖ē_v，以及接触不变集约束 F_min ≤ F ≤ F_max（由 P2 的 h_±/K̂e 置信区间转成速度界，F_max 无 slack）；E_max≈F_safe·Δx_safe；删除 D+α 与 post-QP clip；F* 由罐单独预算 F*·v（到达目标 v=0 后不耗能，建立接触与跟随移动组织时耗能），罐见底降级而不是"不让"。Secchi 的 v≈v_d 假设在本平台 1–3 Hz 不成立，因此账本用实际速度、约束带速度误差与延迟队列鲁棒项。
QPIK 层级：硬安全 > 能量/接触不变集 > 法向力/hover 平移 （力也可以slack）> 接触点旋转 > 轨迹姿态 > nullspace/轨道回中（独立 slack）。我的想法是，之前按照ICRA_27的数据，说明在保持力去接触体表的话，稳定的kd最多只能527n/m不弹跳,所以力不能要求恒定，而是力在一个范围内稳定，而且力的修正幅度要能匹配硬件的能力，不行的话规划出来力的闭环也没有意义，所以应该吧力作为而约束放到QP,然后系统能量也是放大QP,可是在QP新建ENERGYTANK有意义吗。在QP里面产生加上ENERGYTANK的目的是什么？能证无源性吗，还是只是做一个调度的作用？能量/接触不变集 ，接触应该在外环直接作用到VCMD修正上比如结合图像的有显示区域知道接触面，然后结合力传感器数据只当往哪里动可以让图像全接触这个优化变量应该是探头tcp在垂直图像的旋转，这个是应该修正VCMD的（利用外环位置-速度闭环），然后VCMD传入QP传入，还是接触优化项直接传入？然后最重要的反而就是力进QP,现在的代码的修正建议，力本来就是进QP吗？怎么可以松弛力换取能追而而不是为了精确力超级大的反馈，反而带来滞后和不稳定？打死你力还是要证明通道有界限，比如目标是4N,那应该比如一定落到正负2n,开头撞一下不算，这个可有特殊方法优化。然还在这个范围里面qp可以松弛但有严格上下边界，还有类“力”jerk“的限制让可以松弛力换取能追而而不是为了精确力超级大的反馈。还有一个设计是我目前不是servoing,我不是时间系统，而是轨迹跟踪，也就是轨迹可以对时间归一化，成一个路径s 0-1,然后就而已定义一个联系到速度的扫描的速度缩放系数，这个系数也放到qp做松弛，也就是可以自动调速尽量保持稳定。还有一个想法是，前面通超声图像和接触力数据修正了图像内旋转，西安市tilting,就是沿着超声探头接触线倾斜的自由度，如果轨迹是“压”向皮肤内或者在路径上“朝向”图像外面的，会导致导纳力的方向也不准，要不要也类似“辨识出”压”向皮肤内或者在路径上“朝向”图像外面的的方向作为约束，这个不要求大修正，而是极端情况修一下，而且至少能保证头和皮肤在行进方锐角/略小与90,而不是大于90,你可以从大于90 的在有摩擦的弹性物体上容易弹跳的角度顺便证明这个好吗？我这不用精确修，而是就是略小与90,而不是大于90能辨识，最好还能粗略从扫描力数据看出夹角大致大小吗，而且最好知道角度变大/变小的特性，就是定性调节。还有最重要i的，我的cdyob/tdpa/popc这些论文都有给你，我应怎么设计力的qp.什么控制率能不弹跳，力又尽量柔顺，，手推追力快手感也好？还有就是hqp的任务层级怎么调度让总体的力控制最稳定，而且能跟上我识别出来的力特性，有一些叫conditonal filter或者设计l ya p lov函数让整个hqp稳定？仔细再找一波论文，结合
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

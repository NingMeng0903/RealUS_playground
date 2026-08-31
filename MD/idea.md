可以，而且你这个系统其实比“直接在 task-space 砍 \(F\) 或 \(v\)”多了一个很有价值的自由度：**冗余自由度 + 一个低质量、高带宽的末端力轴**。但要把“零空间耗能”“多速率”“延迟”“500 Hz 无源证明”真正串起来，不能只是把几个 energy tank 叠在一起，需要把它做成一个明确的 **energy-routing / passivity layer**。

先给结论：

$$
\boxed{
\text{可以设计成：500 Hz 统一能量监督层}
+
\text{优先在内部/零空间耗散}
+
\text{末端力轴承担高频 passivation}
}
$$

并且可以做到：

$$
\boxed{
\text{arm 200 Hz、rail 20 Hz、force axis 500 Hz}
}
$$

仍然以 **500 Hz 的离散物理端口**为基准给出 passivity 证明。

但有两个条件非常关键：

$$
\boxed{
\text{500 Hz 层必须看到真实接触功率，而不是只看到 }v_{\rm cmd}
}
$$

以及

$$
\boxed{
\text{你声称“零空间耗散”的自由度必须真的能耗散物理能量，不能只是发一个 nullspace velocity command。}
}
$$

第二点对你的 velocity-controlled arm 尤其重要。

---

# 1. 你的“把能量丢到 nullspace”这个想法其实有直接文献基础

Ott、Artigas、Preusche 2011 就专门做过：

> **Subspace-oriented Energy Distribution for the Time Domain Passivity Approach**

他们的问题正是：

TDPA 发现需要耗掉例如：

$$
E_{\rm req}=0.1J
$$

但冗余机器人有很多自由度。

那这 \(0.1J\) 到底在哪里耗？

直接 task-space damping 会破坏主任务。

于是他们把机器人分成：

$$
\boxed{\text{task space}}
$$

和

$$
\boxed{\text{null space}}
$$

然后**优先把所需的耗散放到 nullspace**，只有 nullspace 不够时才动 task-space。

论文摘要说得非常直接：它在冗余机械臂的解耦子空间之间分配所需耗散，优先 nullspace，从而尽量不扰动主 task。([机器人研究所][1])

所以你的直觉：

$$
\boxed{
\text{“既然我有冗余自由度，为什么一定要砍 TCP 的力/速度？
能不能先在 nullspace 把能量耗掉？”}
}
$$

完全是有理论依据的。

---

# 2. 他们为什么能够把 task 和 nullspace 能量拆开？

对一个 \(n\)-DOF robot：

$$
\dot x=J(q)\dot q.
$$

如果 task 是 \(m\)-维，\(n>m\)，还有：

$$
r=n-m
$$

维 nullspace。

Ott 不是随便用：

$$
N=I-J^\#J
$$

就说“这是零空间”。

他使用 inertia metric 构造扩展坐标：

$$
\begin{bmatrix}
\dot x\\
v_N
\end{bmatrix}
=
\bar J(q)\dot q
=
\begin{bmatrix}
J\\
N
\end{bmatrix}\dot q.
$$

通过特殊选择 \(N\)，广义 inertia 变成 block diagonal：

$$
\boxed{
\Lambda=
\begin{bmatrix}
\Lambda_x&0\\
0&\Lambda_N
\end{bmatrix}.
}
$$

于是 kinetic energy 可以真正拆成：

$$
\boxed{
\frac12\dot q^TM\dot q
=
\frac12\dot x^T\Lambda_x\dot x
+
\frac12v_N^T\Lambda_Nv_N.
}
$$

也就是：

$$
\boxed{
E_{\rm kin}=E_{\rm task}+E_{\rm null}.
}
$$

这才使“优先在 nullspace 耗散”具有严格的能量意义，而不是几何上的感觉。([机器人研究所][1])

然后如果加入 nullspace damping：

$$
F_N=-D_Nv_N,
$$

它的功率：

$$
\boxed{
P_{D,N}
=
-v_N^TD_Nv_N
\le0.
}
$$

因此这是真正的耗散。

---

# 3. 但是这里马上出现你系统最大的区别

Ott 那类证明建立在**机器人动力学和 torque actuation**上。

你的 arm 是：

$$
\boxed{\text{velocity interface}}
$$

。

如果你给黑盒 arm 一个：

$$
\dot q_N=-k\nabla H_N
$$

的 nullspace velocity command，

你不能直接说：

$$
\boxed{
P_N=-v_N^TD_Nv_N
}
$$

因为你并不知道内部电机实际给了什么 torque。

事实上 velocity servo 完全可能：

$$
\boxed{\text{主动供电让机器人按照这个 nullspace velocity 运动。}}
$$

它甚至可能**注能**。

所以这是一个必须严格区分的地方：

$$
\boxed{
\text{nullspace velocity motion}
\neq
\text{nullspace physical dissipation}.
}
$$

因此，对你现在的 8-DoF velocity-controlled arm：

> 可以用 nullspace 来减少 task disturbance；
>
> 但不能因为“它在 nullspace 动”就把这部分记成 tank 的耗散收入。

---

# 4. Dietrich 2016 更进一步：nullspace projection 自己甚至会破坏 passivity

这篇特别值得你接下来读：

**Dietrich, Ott, Stramigioli — Passivation of Projection-Based Null Space Compliance Control Via Energy Tanks**

它指出传统 hierarchy：

$$
\tau
=
\tau_1+N^T\tau_2
$$

虽然能保持 task priority，但**projection 本身一般会破坏 passivity**。

他们最后加一个 energy tank，并把两个 hierarchy level 的 damping dissipated power 都送进 tank。([瑞斯大学][2])

他们的总 storage 是：

$$
\boxed{
S
=
\frac12\dot q^TM\dot q
+
V_1
+
V_2
+
E_{\rm tank}.
}
\tag{17}
$$

然后证明：

$$
\boxed{\tau_{\rm ext}\rightarrow\dot q}
$$

整体 passive。([瑞斯大学][2])

而且实验非常符合你想要的效果：

tank 有能量时，控制行为和高性能 classical hierarchy 基本一样；

tank 空时，才牺牲 subordinate task performance 来保持 passivity。([瑞斯大学][2])

---

# 5. 所以你的方向应该稍微改一下

不要说：

$$
\boxed{\text{“我要让 arm nullspace 给我耗能。”}}
$$

更严谨应该分成两个层次：

$$
\boxed{
\text{A. 能量应该优先在哪里被物理耗散？}
}
$$

和

$$
\boxed{
\text{B. 如果不得不修改控制动作，优先修改哪个子空间？}
}
$$

对于你的 velocity-controlled arm：

A 很难严格做到，因为 arm 内部 torque 不透明；

但 B 完全可以：

$$
\boxed{
\text{passivity correction 优先投影到 nullspace}
}
$$

以减少 TCP disturbance。

真正的**高可信物理耗散器**，反而应该是你准备加的：

$$
\boxed{\text{500 Hz、低运动质量的末端 force axis}}
$$

——前提是这个轴最好有 current/force/torque-level access。

这个东西理论上比 arm nullspace 更有价值。

---

# 6. 为什么这个 500 Hz 小力轴特别适合做“energy sink”

假设末端增加一个法向自由度：

$$
x_f
$$

其速度：

$$
v_f=\dot x_f.
$$

如果你能够在电机/力矩层实现：

$$
\boxed{
F_{d,f}=-b_fv_f
}
$$

那么它真实机械功率：

$$
\boxed{
P_{D,f}
=
F_{d,f}v_f
=
-b_fv_f^2
\le0.
}
$$

这是没有争议的耗散。

因此：

$$
\boxed{
E_{D,f}[k]
=
h\,b_fv_f^2[k]
}
$$

可以合法地给 tank 充值。

这和对黑盒 arm 发一个 nullspace velocity 完全不同。

---

# 7. 你甚至可以做出一个真正的“energy router”

我认为你这个系统最有研究价值的结构是：

$$
\boxed{
\text{一个 global tank}
}
$$

然后 tank 收入来自真正可证明的 dissipative elements：

$$
P_{\rm in,tank}
=
P_{D,f}
+
P_{D,N}
+\cdots
$$

其中 \(P_{D,N}\) 只有在你确实能够证明那条 nullspace damping 是 torque-level physical damping 时才能计入。

tank 支出则是：

$$
P_{\rm spend}
=
P_{\rm force\;active}
+
P_{\rm delay}
+
P_{\rm switching}
+
P_{\rm discrete}.
$$

可以概念写成：

$$
\boxed{
\dot T
=
P_{D,f}
+
P_{D,N}
-
P_{\rm act,F}
-
P_{\rm delay}
-
P_{\rm num}.
}
$$

只要求：

$$
\boxed{
T(t)\ge T_{\min}>0.
}
$$

那么你就可以允许 force controller 在短时间里保持高性能 active behavior。

这就和前面 Lee 的逻辑接上了。

---

# 8. 但有一个很重要的物理限制：nullspace 不能“制造免费能量”

假设系统现在完全静止：

$$
v_N=0.
$$

你不能说：

> “tank 没钱，那我自己在 nullspace 摇两下，产生一点 damping energy 给 tank 充值。”

因为为了先把 nullspace 摇起来，你的电机必须先注入：

$$
E_{\rm drive}.
$$

然后 damping 再消掉：

$$
E_{\rm dissipated}.
$$

如果你只记：

$$
E_{\rm dissipated}
$$

不记最初 motor 注入的能量，就是假账。

所以：

$$
\boxed{
\text{不能主动制造 nullspace motion 来“刷 tank”。}
}
$$

你只能回收：

$$
\boxed{\text{原本真实存在并被 damping 消耗的能量。}}
$$

---

# 9. 还有一个更深的问题：如果是“完美 nullspace”，怎么把 task energy 转过去？

如果：

$$
J\dot q_N=0,
$$

那么环境端口功率：

$$
F_{\rm ext}^TJ\dot q_N=0.
$$

这正意味着：

$$
\boxed{\text{完美 nullspace 不直接改变 TCP power。}}
$$

所以如果 task port 正在向环境输出：

$$
5W,
$$

一个与 task **完全解耦**的 nullspace damping，并不能神奇地直接从这个 \(5W\) 里把能量吸走。

它只能耗散 nullspace 自己已有的 kinetic energy。

这里 energy tank 的意义才出现：

你可以以前在 nullspace / damping 中真正耗散过：

$$
1J
$$

把它存到 tank。

之后 task controller 短暂 active：

$$
0.1J
$$

从 tank 支付。

于是 augmented system 仍然 passive。

所以是：

$$
\boxed{
\text{时间上的能量转移}
}
$$

而不一定是：

$$
\text{当前这一瞬间 task power 直接流进 nullspace}.
$$

---

# 10. 你这里真正最好的“当前瞬时能量吸收器”是末端小力轴

设法向 TCP 实际速度：

$$
\boxed{
v_{\rm tip}
=
v_{a}
+
v_r
+
v_f
}
$$

这里：

$$
v_a=\text{arm 对法向速度的实际贡献},
$$

$$
v_r=\text{rail 对法向速度的实际贡献},
$$

$$
v_f=\text{500 Hz local force axis}.
$$

真正物理接触功率：

$$
\boxed{
P_e
=
F_e v_{\rm tip}
=
F_e(v_a+v_r+v_f).
}
$$

现在 arm 因 delay 还在往下：

$$
v_a=+10\ {\rm mm/s}.
$$

rail 也有：

$$
v_r=+2\ {\rm mm/s}.
$$

但 500 Hz 小轴可以立即：

$$
v_f=-12\ {\rm mm/s}.
$$

于是：

$$
v_{\rm tip}\approx0.
$$

这就是真正直接切断：

$$
F_ev_{\rm tip}
$$

危险功率的办法。

这里不是“让 arm 立刻停下来”。

而是：

$$
\boxed{
\text{arm 来不及停，但 local actuator 先把它的运动在接触点抵消。}
}
$$

这非常适合你的问题。

---

# 11. 这比 Lee 的 torque-level bidirectional flow 更容易在你硬件上实现

Lee 是：

$$
\text{real robot跟不上 nominal}
$$

于是用 torque：

$$
\tau_c,\alpha
$$

重分配 real/nominal dynamics。

你没有 arm torque interface。

但如果末端有一个：

$$
\boxed{\text{500 Hz force/current controlled stage}}
$$

你就重新获得了一个真正高频、低惯量的物理 actuation port。

于是可以让：

$$
\boxed{
\text{slow massive arm}
}
$$

承担低频大行程，

$$
\boxed{
\text{fast light force axis}
}
$$

承担：

* contact transient；
* delay compensation；
* high-frequency energy absorption；
* force tracking correction。

这个结构比逼 7-DoF arm 自己在 200 Hz 下解决所有问题合理很多。

---

# 12. 你的 500 / 200 / 20 Hz 多速率问题其实也有非常直接的文献

这个我刚查到一个和你非常接近的：

**De Stefano et al., 2019 — Multi-rate Tracking Control for a Space Robot on a Controlled Satellite: A Passivity-Based Strategy**

他们就是：

$$
\boxed{\text{fast manipulator controller}
+
\text{slow actuated base controller}}
$$

并且指出：

$$
\boxed{\text{different sampling rates 本身会制造 virtual energy。}}
$$

然后用 PO/PC 做 passivation。([电子图书馆][3])

甚至实验是：

$$
\boxed{
\text{base}=250\,Hz,
\qquad
\text{manipulator}=1000\,Hz.
}
$$

([电子图书馆][3])

模拟中更夸张：

$$
T_m=1{\rm ms},
\qquad
T_b=0.3{\rm s}.
$$

也就是：

$$
1000Hz
\quad\text{vs}\quad
3.3Hz.
$$

而他们仍然基于 passivity 做了统一分析。([电子图书馆][3])

所以你的：

$$
500Hz/200Hz/20Hz
$$

并不是理论上无法处理的组合。

---

# 13. 这篇 multi-rate 论文有一个特别重要的技巧，几乎可以直接借给你

虽然 base controller 很慢，作者没有让 **energy observer 也跟着慢**。

他们明确写：

> slow base port 的 passivity observer 放在更快的系统里运行。

所以 PO 按快周期更新：

$$
\boxed{
E_{\rm obs,b}(k_m)
=
E_{\rm obs,b}(k_m-1)
+
F_B(k_m)\dot x_b(k_m)T_m
+
\cdots
}
\tag{23}
$$

而 base force command 只在慢周期更新。

两个慢周期之间：

$$
F_B
$$

hold 住。

最关键一句：

$$
\boxed{
\text{base velocity 必须能在快采样率 }T_m\text{ 下测到。}
}
$$

论文就是这么做的。([电子图书馆][3])

这跟你的系统高度对应。

---

# 14. 所以你的统一时钟应该是 500 Hz，而不是 20 或 200 Hz

定义：

$$
h_f=2{\rm ms}.
$$

500 Hz supervisor 每 2 ms 做一次：

$$
\boxed{
P_e[k]
=
W_e^T[k]V_{\rm tip,actual}[k].
}
$$

然后：

$$
\boxed{
E_e[k+1]
=
E_e[k]
+
h_fP_e[k].
}
$$

arm controller 仍然：

$$
200Hz
$$

更新。

rail：

$$
20Hz
$$

更新。

没有问题。

500 Hz PO 只要求：

$$
\boxed{\text{每 2 ms 能知道这些 subsystem 当前真正贡献的物理速度。}}
$$

---

# 15. 这里有一个必须解决的测量问题

如果 arm 的**控制命令**只有：

$$
200Hz
$$

没问题。

但如果 arm 的**真实状态反馈**也只有：

$$
200Hz,
$$

那么你在 500 Hz 并不知道：

$$
v_a[k].
$$

rail 如果连 encoder 都只：

$$
20Hz
$$

那更严重。

这时候你不能严格声称：

$$
\boxed{
P_e[k]=F_e[k]v_{\rm tip}[k]
}
$$

被 500 Hz 精确观测到了。

因此想要真正的 500 Hz physical-port proof，你至少需要以下二者之一：

$$
\boxed{
\text{A. 500 Hz actual TCP velocity measurement}
}
$$

例如高率 joint encoder、local encoder、外部 tracking；

或者：

$$
\boxed{
\text{B. 对两次状态之间运动的严格 upper bound}
}
$$

进行保守 energy accounting。

这点不能偷。

---

# 16. 500 Hz 力轴本身最好直接测 local displacement/velocity

例如：

$$
x_f,\qquad \dot x_f
$$

500 Hz 甚至 1 kHz。

然后如果 arm/rail 的高速真实 velocity 暂时不可得，可以写：

$$
v_{\rm tip}
=
\underbrace{\hat v_{a,r}}_{\text{估计}}
+
v_f
+
\underbrace{e_v}_{\text{未知}}.
$$

对未知误差给：

$$
|e_v|\le\bar e_v.
$$

于是 worst-case external power：

$$
P_e
\ge
F_e(\hat v_{a,r}+v_f)
-
|F_e|\bar e_v.
$$

所以用保守版本：

$$
\boxed{
P_{\rm safe}
=
F_e(\hat v_{a,r}+v_f)
-
|F_e|\bar e_v.
}
$$

做 tank update。

这样才能把“状态反馈没到 500 Hz”也纳入严格证明。

---

# 17. 然后可以做一个 500 Hz 的最小干预 QP

你的 nominal force controller 先给：

$$
v_f^\star.
$$

不要直接发。

500 Hz safety layer 解：

$$
\boxed{
\min_{v_f}
\quad
(v_f-v_f^\star)^2
}
$$

subject to：

$$
\boxed{
T_k
+
h_fF_e
\left(
v_a^{\rm actual}
+
v_r^{\rm actual}
+
v_f
\right)
\ge
T_{\min}.
}
$$

加上：

$$
|v_f|\le v_{f,\max},
$$

$$
|a_f|\le a_{f,\max},
$$

$$
x_{f,\min}\le x_f\le x_{f,\max}.
$$

这个 constraint 对 \(v_f\) 是线性的。

所以还是一个非常简单的 convex QP。

---

# 18. 这时候和 Secchi 2019 最大的区别出来了

Secchi 2019 是：

$$
F^Tv_{\rm command}.
$$

你应该是：

$$
\boxed{
F^T
\left(
v_{arm,\rm actual}
+
v_{rail,\rm actual}
+
v_f
\right).
}
$$

也就是**真实 physical contact port**。

因此不再要求：

$$
v_{\rm actual}=v_{\rm desired}.
$$

这正是你要解决的核心问题。

---

# 19. 而且不要让 QP 只最小化 velocity error

因为你真正想保的是 force：

$$
F_d.
$$

所以更好的 objective 是：

$$
\boxed{
\min_{v_f}
\quad
w_F
\left(
\hat F_{k+1}(v_f)-F_d
\right)^2
+
w_v(v_f-v_f^\star)^2
+
w_a(v_f-v_{f,k-1})^2
}
$$

subject to：

$$
\boxed{\text{passivity constraint}}.
$$

其中局部环境预测：

$$
\hat F_{k+1}
=
F_k
+
h_f\hat K_e
v_{\rm tip,k+1}
$$

可以只是短期线性模型。

所以：

$$
\boxed{
\text{force accuracy 是 objective，
passivity 是 hard constraint。}
}
$$

这比“无源了就直接把速度缩小”好很多。

---

# 20. 那 nullspace 怎么放进这个 QP？

这里要分两种。

如果是纯 velocity-controlled arm，可以让 PC correction **优先使用 kinematic nullspace**：

$$
\dot q
=
J^\#V_{\rm task}
+
N\nu.
$$

例如：

$$
\boxed{
\min_{\nu,\delta V}
\quad
w_N\|\nu\|^2
+
w_T\|\delta V\|^2
}
$$

其中：

$$
w_T\gg w_N.
$$

所以 optimizer 总是先改：

$$
\nu
$$

只有 nullspace 无法解决 constraint，才改：

$$
\delta V_{\rm task}.
$$

这是 Ott 的“nullspace first”的思想。

但注意：

$$
\boxed{\nu\text{ 本身不能给 tank 充值。}}
$$

它只是让 passivity correction 尽量不扰动 task。

---

# 21. 如果 force axis 是 torque/current-controlled，那就可以真正给 tank 充值

此时：

$$
P_{D,f}
=
b_fv_f^2.
$$

所以：

$$
\boxed{
T_{k+1}
=
T_k
+
h_fb_fv_f^2
-
E_{\rm spend,k}
}
$$

。

如果以后 arm 也能获得 torque-level nullspace damping：

$$
P_{D,N}
=
v_N^TD_Nv_N,
$$

就可以：

$$
\boxed{
T_{k+1}
=
T_k
+
h_f
\left(
v_N^TD_Nv_N+b_fv_f^2
\right)
-
E_{\rm spend,k}.
}
$$

这才是真正的“nullspace energy harvesting”。

---

# 22. 还有一个必须加的东西：delay reserve

假设现在 tank：

$$
T=0.05J.
$$

500 Hz 层发现危险，现在让 force axis反向。

但 arm 已经发出去的 200 Hz velocity command 还会继续执行几毫秒。

rail 的 command 甚至可能继续几十毫秒。

所以不能只要求：

$$
T>T_{\min}.
$$

需要：

$$
\boxed{
T
-
E_{\rm committed}
\ge
T_{\min}.
}
$$

其中：

$$
E_{\rm committed}
$$

表示：

> 已经发进 arm/rail pipeline，但尚未完成的动作，未来最坏还能向 contact port 输出多少能量。

可以写：

$$
\boxed{
E_{\rm commit}
=
E_{\rm arm,pipeline}
+
E_{\rm rail,pipeline}.
}
$$

比如有预测模型：

$$
\hat V_{a,k+i|k}
$$

则：

$$
\boxed{
E_{\rm arm,pipeline}
=
h_f
\sum_{i=1}^{N_d}
\max
\left(
0,
-W_e^T\hat V_{a,k+i|k}
\right).
}
$$

如果没有可靠模型，就用 worst-case velocity bound。

---

# 23. 这其实就是把 De Stefano 的 pure-delay 思路推广了

De Stefano 2020 假设：

$$
V_{\rm actual}(k)
=
V_s(k-\mu).
$$

你应该推广成：

$$
\boxed{
V_{\rm actual}
=
G(z)V_{\rm cmd}.
}
$$

其中：

$$
G(z)
$$

包含：

* pure delay；
* finite bandwidth；
* acceleration limit；
* sample/hold。

然后不是只估一个：

$$
\mu
$$

而是估：

$$
\boxed{
\text{future committed physical energy}.
}
$$

我认为这是你这个系统真正有研究贡献的位置。

---

# 24. 20 Hz rail 应该怎么处理？

rail 不应该承担高频 passivity recovery。

20 Hz 太慢。

它应该承担：

$$
\boxed{
\text{DC / very-low-frequency posture and workspace allocation}.
}
$$

也就是说：

$$
\text{rail}
\rightarrow
\text{慢速重心/可达性/构型}
$$

$$
\text{arm}
\rightarrow
\text{中频 TCP}
$$

$$
\text{force axis}
\rightarrow
\text{高频 contact/energy}.
$$

但是这只是**性能分频**。

passivity proof 不能写成：

> “20 Hz 是低频，500 Hz 是高频，所以一定无源。”

passivity 是：

$$
\boxed{\text{energy property}}
$$

不是 frequency-separation theorem。

所以 rail 的实际运动仍然必须进入：

$$
W_e^TV_{\rm actual}
$$

或者进入 conservative bound。

---

# 25. 200 Hz arm 也类似

arm 可以继续负责 nominal force/position task：

$$
V_a^\star.
$$

但 500 Hz layer 不要求 arm 立刻执行。

它只看：

$$
V_{a,\rm actual}.
$$

如果 arm 落后：

$$
V_a^\star-V_{a,\rm actual}
$$

很大，

500 Hz local stage 先补：

$$
v_f.
$$

这等价于：

$$
\boxed{
\text{force axis 作为 arm 的 high-frequency residual actuator}.
}
$$

这其实很接近你前面说的：

> arm 有大惯量、速度瞬态跟不上，那让小质量轴去补。

我认为这是正确方向。

---

# 26. 500 Hz 能不能“严格证明无源”？

要分两种。

### 离散时间 500 Hz passivity

可以。

你证明：

$$
\boxed{
\mathcal S[k+1]-\mathcal S[k]
\le
h_fW_e^T[k]V_{\rm tip}[k]
}
$$

对每个：

$$
k
$$

成立。

其中总 storage 可以是：

$$
\boxed{
\mathcal S
=
S_{\rm virtual}
+
S_f
+
T.
}
$$

如果 force reference 自己也是主动源，还必须把它作为另一个供能端口，或者由 tank 支付：

$$
\boxed{
\Delta\mathcal S
\le
h_fW_e^TV_{\rm tip}
+
E_{\rm ref,in}.
}
$$

只要所有内部 energy transfer cancel、所有数值误差和 delay debt 都记账，就可以形成严格的 sampled-data passivity proof。

---

# 27. 但是“500 Hz 离散无源”不自动等于连续时间每一瞬间都无源

在两个 2 ms sample 之间还发生物理运动。

所以：

$$
\boxed{\text{sampled passivity}}
$$

和：

$$
\boxed{\text{continuous physical passivity}}
$$

要区分。

Stramigioli、Secchi 等人在 2005 年专门研究了 continuous/discrete interconnection，构造了可以保持 passive 的 sampled-data interface，而且结论可以不依赖具体 sampling time。([瑞斯大学][4])

如果你论文里只证明：

$$
E[k]\ge0
$$

最好写：

$$
\boxed{\text{discrete-time passivity at the 500 Hz physical port}}
$$

而不要直接说：

$$
\text{continuous-time unconditional passivity}.
$$

除非你进一步处理 sampler/ZOH/intersample energy。

---

# 28. 500 / 200 / 20 还有个小技术点

500 Hz：

$$
2ms
$$

200 Hz：

$$
5ms.
$$

它们不是整数倍：

$$
5/2=2.5.
$$

所以不要写一个假装所有 controller 都每 \(N\) 个 tick 对齐的代码。

应该是：

$$
\boxed{\text{timestamped asynchronous sample-and-hold}}
$$

。

500 Hz energy observer 永远运行。

arm command 在：

$$
0,5,10,15\ldots ms
$$

更新。

force controller 在：

$$
0,2,4,6,\ldots ms
$$

更新。

rail：

$$
0,50,100,\ldots ms.
$$

能量积分使用真实：

$$
\Delta t_k
$$

即可：

$$
\boxed{
E_{k+1}
=
E_k
+
\Delta t_k W_k^TV_k.
}
$$

不要假定所有 event 共用同一个同步 index。

---

# 29. 真正适合你的 hierarchy

我会设计成：

$$
\boxed{
\textbf{Layer 1: nominal performance}
}
$$

200 Hz arm：

$$
\text{hybrid force/position + QPIK + nullspace}
$$

20 Hz rail：

$$
\text{workspace/posture allocation}
$$

500 Hz force axis：

$$
\text{fast force residual controller}.
$$

然后：

$$
\boxed{
\textbf{Layer 2: 500 Hz energy/passivity shell}
}
$$

每 2 ms：

$$
F_{\rm ext},\quad
V_{\rm tip,actual}
$$

计算：

$$
P_e.
$$

再维护：

$$
T,\quad E_{\rm commit}.
$$

然后按照优先级：

$$
\boxed{
\text{local force-axis correction}
\rightarrow
\text{nullspace correction}
\rightarrow
\text{arm task correction}
\rightarrow
\text{force-reference slowdown}
}
$$

尽量最后才牺牲 force tracking。

这其实就是 Franken 2011 的 two-layer 哲学：performance/transparency 和 passivity 分开，下层保证不产生“virtual energy”。

---

# 30. 最关键：这样是否真的能“不牺牲 force accuracy”？

不是绝对不牺牲。

没有任何方法可以同时对任意硬环境、任意 delay、有限 actuator 做到：

$$
\boxed{
\text{exact force tracking}
+
\text{exact trajectory}
+
\text{strict passivity}
}
$$

永远同时满足。

如果 force axis：

$$
v_{f,\max}
$$

已经不够抵消：

$$
v_a+v_r,
$$

energy constraint 与 force tracking constraint 本身就可能不可行。

所以必须有优先级。

但你的硬件设计可以把“需要牺牲 task”的概率大幅降低：

$$
\boxed{
\text{large-stroke slow rail}
+
\text{medium-bandwidth arm}
+
\text{small-mass high-bandwidth local force actuator}
+
\text{energy tank/nullspace routing}
}
$$

这比只有一台 200 Hz arm 强很多。

---

# 31. 还有一个特别重要的 force-control 理论问题

如果你要求：

$$
F=F_d
$$

而环境自己在运动，那么 constant-force controller 本身可能需要持续向环境输出能量。

所以若你想证明**整个系统对环境 one-port passive**，不能把：

$$
F_d
$$

当作“免费的 reference”。

要么把 desired force generator 当第二个 energy port：

$$
\boxed{
P_{\rm ref}
}
$$

要么让它从 tank 支付：

$$
\boxed{
T_{k+1}
=
T_k
-
E_{\rm force,active}
+\cdots
}
$$

这和 Lee 把：

$$
\dot p_{\rm des}^TF_o
$$

显式作为另一个 supply port 是同一个道理。

所以你的最终 theorem 最好不是含糊地说：

> “整个恒力控制器永远无源。”

而是很明确：

$$
\boxed{
\text{environment-facing mechanical port is passive,
provided all active reference energy is bounded by the tank.}
}
$$

这个说法理论上干净得多。

---

# 32. 我认为可以形成一个真正不错的论文方法

如果压缩成数学核心，我会把你要做的东西定义成：

$$
\boxed{
\textbf{Multi-rate Redundancy-Aware Passivity Layer}
}
$$

500 Hz global storage：

$$
\boxed{
\mathcal E[k+1]
=
\mathcal E[k]
+
hF_{\rm ext}^TV_{\rm TCP,actual}
+
E_{\rm diss,null}
+
E_{\rm diss,local}
-
E_{\rm active,ref}
-
E_{\rm num}
}
$$

同时保留：

$$
\boxed{
\mathcal E[k]-E_{\rm committed}[k]\ge E_{\min}.
}
$$

然后控制分配 QP：

$$
\boxed{
\min
\;
w_F\|F-F_d\|^2
+
w_f\|\delta u_f\|^2
+
w_N\|\delta\nu_N\|^2
+
w_T\|\delta V_{\rm task}\|^2
}
$$

其中：

$$
w_T\gg w_N,w_f
$$

使 task 最后才被动。

约束包括：

$$
\boxed{
\mathcal E_{k+1}\ge E_{\min}+E_{\rm committed}
}
$$

以及：

$$
J\dot q_N=0,
$$

$$
v_f,a_f,x_f
$$

limits，

arm/rail limits。

如果这个 QP feasible：

$$
\boxed{
\Delta\mathcal S
\le
hF_{\rm ext}^TV_{\rm TCP,actual}
}
$$

就可以成为你的 sampled-data passivity theorem。

---

我认为这里最有价值的不是“再加一个 energy tank”，而是这三个点合在一起：

$$
\boxed{
\text{① 真实端口 }F^TV_{\rm actual}
}
$$

$$
\boxed{
\text{② multi-rate / delayed actuation 的 committed-energy accounting}
}
$$

$$
\boxed{
\text{③ redundancy-aware energy routing：local force axis/nullspace first，task last}
}
$$

现有论文分别覆盖了其中的部分：Ott 2011 已经证明“优先 nullspace 耗散”这条路是合理的；De Stefano 2019 已经证明不同 controller rate 可以从 energy/passivity 角度统一处理，而且 slow controller 的 PO 可以放到 fast rate 上；De Stefano 2020 处理 delay/discretization；Dietrich 2016 说明 nullspace hierarchy 本身必须被 passivate；但把这些东西真正组合成**black-box velocity-controlled arm + slow rail + 500 Hz local force actuator + physical-port force control**，是一个明显不同、也更贴近你硬件的问题。

[1]: https://www.robotic.dlr.de/fileadmin/robotic/artigas/irosNullSpaceTDPC_final1.pdf "https://www.robotic.dlr.de/fileadmin/robotic/artigas/irosNullSpaceTDPC_final1.pdf"
[2]: https://ris.utwente.nl/ws/files/105395395/passivation.pdf "https://ris.utwente.nl/ws/files/105395395/passivation.pdf"
[3]: https://elib.dlr.de/127658/1/0_ral19_final.pdf "https://elib.dlr.de/127658/1/0_ral19_final.pdf"
[4]: https://ris.utwente.nl/ws/files/6785106/01492474.pdf "https://ris.utwente.nl/ws/files/6785106/01492474.pdf"

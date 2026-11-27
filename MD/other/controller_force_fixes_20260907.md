# 控制器 timeout、力矩引导与补偿修复

基线：`main` / `e75c5e32b24298a152e7500691ea0c2c987645e6`。
对照：`524b3eecda6370deb3efd70cbdf659da8a5e82d2`。

## 已确认的问题与修改

| 问题 | 证据与修复 |
|---|---|
| native 请求等待 | Python 忙轮询，C++ 仍定时 sleep 轮询，正常唤醒会受调度竞争及 timer slack 影响。改为 socketpair 双向通知，SHM 仍传负载。保留 20 ms 请求期限、故障锁存与显式 reset；不发送超时后的旧解。 |
| timeout 日志误导 | 未确认当前回复序号就读取 `solve_ms`，可能得到上一拍或尚在写的数据。超时当前求解时间改为 NaN，另报最后完成的求解时间、回复/请求序号和退出/超期原因。用户提供的 7.03/13.67 ms 不能证明当前请求已按时完成。 |
| nanobind 退出引用 | 原 daemon 未显式关闭 inner；signal handler 和 native client 还可保留 controller。增加确定的关闭路径、恢复信号处理器、解除反向强引用并释放本 controller 的 QP cache；stream 在 SDK 断开前停止。 |
| 力矩引导卡住 | 524b 的累计角度达到 0.52 rad 后停止；法向力误差达到 2.25 N 后增益可变为零。e75 则删除了整个 tilt 功能。恢复单轴 TCP-y 力矩导纳，删除这两种门控以及额外的 jerk 历史，保留速度/加速度与底层关节约束。 |
| monitor 非零/轴混合 | 原 monitor 把带工具偏置的 TCP 姿态用于 link_7 重力模型，参考点也与控制器不同。现在默认显示控制器实际 TCP 补偿 wrench；独立复算先还原 link_7 姿态，并复用在线 observer。`--frame` 同时指定坐标轴和力矩原点。 |
| 补偿坐标契约 | V2 在 link_7 拟合；原在线 observer 却仅给 sensor 原始值乘符号并使用 sensor 重力。现在所有模型与输入统一至 link_7，再移到 TCP。现有配置 sensor 与 link_7 同位同向，因此这项潜在错误不能独立解释本次全部现场现象。 |
| 标定文件与噪声权重 | 保存时明确参数坐标系与安装绑定并原子替换；不可估计的诊断值写 null，负载参数仍要求有限。静态噪声应对每个静态窗口去均值后池化，并只使用训练窗口，避免把真实姿态重力变化及留出集混入拟合权重。 |

没有用接触中的自动清零掩盖偏置，也没有覆盖仓库中的现场标定文件。

## 力矩引导的行为

只开放垂直成像面的工具 y 旋转轴，约定沿用现有 Legacy/FCE：驱动是期望 wrench 减去实测 wrench。

\[
M_\theta\dot\omega_y+D_\theta\omega_y+\tau_c\,\mathrm{Sgn}(\omega_y)
\ni \tau_y^\star-\hat\tau_y.
\]

使用离散隐式 Coulomb 更新，保持原参数：质量 0.04、阻尼 0.30、噪声/摩擦阈值 0.025 N·m、最大角速度 0.20 rad/s、最大角加速度 5 rad/s²。不会因累计转角耗尽而停止。扭矩翻转时先制动再反转，有限惯性不能让实际速度瞬间反向。

默认只在接触时引导。需要空中手动旋转时可明确设 `hybrid_motion.torque_tilt.contact_only: false`。显式 Z-only 任务仍保持其选轴；`peirastic.DEMO.hfpc` 已恢复 Z + y 旋转，新的 phantom 示例仍显式只控制 Z。

## 数据与理论边界

这次静态标定日志和残差的可复现核对见 [标定审计](force_compensation_audit_20260907.md)。即使 sensor/link_7 残差很小，移到较远 TCP 后，力残差也会贡献力矩残差；不能把这叫作数组串轴。

明确令 $r_{ST}^{S}$ 表示 sensor 原点指向 TCP 的向量，$R_{TS}$ 将 sensor 分量旋转至 TCP，则

\[
f^T=R_{TS}f^S,\qquad
\tau^T=R_{TS}\left(\tau^S-r_{ST}^{S}\times f^S\right).
\]

通过 TCP 作用且没有自由力偶的力应满足 $\tau^S=r_{ST}^{S}\times f^S$，移到 TCP 后力矩为零。坐标旋转本来会混合 y/z 分量；参考点平移本来会将力引入力矩。这与传感器物理串扰是不同问题。[Modern Robotics §3.4](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/)

静态重力/质心/偏置标定不等于任意动态惯性补偿，也不等于辨识任意 6×6 传感器解耦矩阵。相关原始来源：[Carlson 2019](https://arxiv.org/abs/1904.06158)、[Traversaro 等](https://arxiv.org/abs/1410.0885)。

用接触力矩调整超声探头姿态有文献依据，但零力矩不能独立证明全接触，也可能对应通过 TCP 的单点接触或离开表面：[Ning 等，2021](https://link.springer.com/article/10.1007/s11548-021-02462-6)。

## 运行与验证

最终联合回归 **176 passed，0 skipped**，包含真实 native 进程的 start/reset/step/stop、暂停/迟到/恢复、共享内存异常、并发请求、退出清理、源码哈希一致性，以及力矩、补偿、标定和 daemon 集成。完整 native C++ Release 构建通过；Python dry-run 正常退出，没有 nanobind 泄漏提示。

验证环境：Python 3.12、Pinocchio 4.1.0、ProxSuite 0.7.3、Ruckig 0.17.3。没有连接真机，也没有进行目标机器的长期负载测试。

更新 Python 文件后必须重编 native；源码哈希检查会拒绝旧二进制。在原有 rm75 环境和仓库根目录运行：

```bash
bash rm75_control/native/wbc_rt/build.sh
python -m peirastic.apps.run_controller --log-csv
```

另一个已加载相同环境的终端：

```bash
python rm75_control/apps/force_compensation/force_monitor.py --source shm --frame tcp
```

独立核对当前文件中的静态模型：

```bash
python rm75_control/apps/force_compensation/force_monitor.py --source shm --frame tcp --recompute
```

默认曲线来自控制器实际使用的值；`--recompute` 来自 monitor 自己的重算。monitor 可以重载本进程的模型，活动控制器不会被远程热替换；重新标定后需重启控制器使新参数生效。启动时会报告模型路径、来源、坐标系、质量与修订信息。

真实 native 挂起、进程退出、传感器/硬件故障仍应停止。普通 Linux 无法保证任意负载下绝无 deadline miss；本修复不构成无源性、全接触或真机不弹跳的证明。现场仍需用空中残差及正负小力矩确认本机模型和安装轴约定。

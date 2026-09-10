# 不均匀测量源时钟定向复核 v5

范围只包括 active 的源时间检查、观测 LP/HP 变步长实现及新能力预检，针对当前 `active_probe50.yaml` 声明的名义 5 ms、最大源间隔/年龄 15 ms。未连接设备、未重启服务，不涉及修改原 IK。**限定软件范围 PASS，无剩余阻断。** 本结论允许进入用户主动启动的实际实验，不等于已经验证非均匀采样下的真实力效果。

## 数学规格

固定名义间隔 (h_0=0.005\,s)，用旧数字 Butterworth 对应的预扭曲极点 (\Omega=2\tan(\pi f_c h_0)/h_0)。仅每次新源事件的实际间隔 (h_k) 改变；不可每帧重新预扭曲极点。45 Hz 一阶低通保留上一补偿原始输入 (u_{k-1}) 与上一滤波输出 (y_{k-1})：

\[
y_k=\frac{(2-\Omega h_k)y_{k-1}+\Omega h_k(u_{k-1}+u_k)}{2+\Omega h_k}.
\]

2.5 Hz 二阶高通可取固定连续实现

\[
A=\Omega\begin{bmatrix}0&1\\-1&-\sqrt2\end{bmatrix},\quad
B=\Omega\begin{bmatrix}0\\1\end{bmatrix},\quad C=[-1,-\sqrt2],\quad D=1,
\]

并按梯形法推进

\[
x_k=(I-h_k A/2)^{-1}\{(I+h_k A/2)x_{k-1}+h_kB(u_{k-1}+u_k)/2\},\quad y_k=Cx_k+u_k.
\]

恒定输入的稳态为 (x=[u,0]^T)，高通输出为零。固定 5 ms 时双线性变换与原数字滤波器传递函数一致；独立 4,000 点复合正弦、相同稳态初值数值验证最大差：LP `1.7764e-15`、HP `3.3751e-14`。这是数学方案核对，不能代替实际集成源码测试。

LP 切入须保留 `u_last/y_last`；切回原 DFII 滤波器的状态为 `zi=b1*u_last-a1*y_last`。不以最新 raw 重置滤波输出。重复源快照完全不推进滤波和观测统计；下一新源使用真实源时间差，不使用控制 tick 间隔。

## 性质与边界

连续极点稳定时，梯形离散的极点映射 μ=(1+hλ/2)/(1-hλ/2) 在任意正 h 下仍位于单位圆内；对这一固定连续系统，共同二次 Lyapunov 函数也保持每一步耗散。长期缺测仍必须由声明的间隔与年龄界拒绝，不能据此稳定性结论忽略缺测。

固定 5 ms 的滤波等价与非均匀采样下的新滤波语义是两种声明。后者不恢复未采集的 FT 连续信号，不保证原真实力效果或无超调；较长间隔可使 LP 离散极点为负。原 source 时间是 UDP 接收时间，不是假设的 FT 硬件采样时刻。

关闭要求：实际代码的默认旧模式路径保持；active 每个新源只消费一次，运行中源间隔/年龄超过声明上限、倒序和换 epoch 拒绝；进入与退出状态转换通过动态和恒 4 N 反例；命令积分仍按原控制时间，不借源时钟改变 4 N 或参考时钟。新可变源时钟能力须区别旧 `active_v1`，确保未重启的旧 daemon 在教学/prepare 前拒绝，不能仅本地配置解析成功就宣称服务已载入。

已停止、非实时编译后进入新 active epoch 时，共享 observer 的前一段历史可能已超过 15 ms。此处与运行中丢帧区分：首个新鲜且年龄不超过 15 ms 的真实样本，可明确以当前补偿 raw 稳态初始化 LP，并记录 epoch 重置；不能跨该缺口插值，也不能声称连续状态等价。连续历史不超过 15 ms 时保留 `u/y`，采用 observer 自己的真实时间差，不能由新 SourceClock 首样本默认 5 ms 覆盖。该例外仅允许新模式边界一次，运行中超限仍拒绝，不增加用户预热步骤。

## 源码关闭依据

- `SourceClock` 保留默认固定周期分支，新增显式 `variable_step_bilinear_v1`，分别检查真实新源间隔和当前样本年龄。重复时间戳返回 held，不递增样本数；倒序、epoch 改变、冲突 wall provenance 与超限仍拒绝。当前部署声明两项上限均为 15 ms。
- `VariableLowpass1` 与 `VariableHighpass2` 固定预扭曲极点，按实际新源时间差推进；没有每拍重新设计 Butterworth。HP 的代码采用与上文相似变换等价的速度/位置状态及闭式二阶梯形解。
- runner 在 phase 边界绑定 observer；每拍先执行 SourceClock 年龄/身份检查，再消费 observer。已有 observer 的同一时间戳不被再次处理；首个不同时间戳优先使用其自身 `t_s-last_t_s`，不使用新 clock 的默认名义间隔冒充该差值。
- 连续切入/退出保持动态 LP 输出状态；新 epoch 的历史 gap 只允许一次显式当前测量初始化，输出为当前补偿 raw，而不是 0。事件通过已有有界 sink 记录。运行中再次 gap 在修改 observer 水位与样本数之前拒绝。新 controller 首个测量按当前输入稳态初始化 HP，重复样本不推进其状态。
- 默认旧路径继续原 `lfilter` 与原控制时间；变步长只属于显式 active 路径。原 4 N、命令积分/参考提交语义和 IK 未因本修复改变。
- 新 daemon 在启动时导入实际滤波实现并发布 `contact_qp.source_timebase_bilinear_v1`。API 和原扫描入口均要求它以及 `contact_qp.active_v1`；原 Robot 构造和 attempt 的 prepare 前检查顺序保持。只提供旧 `active_v1` 的运行中服务明确拒绝，新本地 YAML 验证不能绕过服务版本检查。需用户在原控制器终端重新启动以载入新实现；本次没有代为启动。

## 验证

独立源码测试（rm75 环境）：variable timebase、two clock、capabilities、active、runtime config，**41 passed in 1.42 s**。外部原扫描入口独立窄复测（genesis）：新 capability 拒绝、构造前检查、默认不启用，**3 passed in 0.40 s**。首次误用 genesis 运行含实际 QP 求解的组合时，35 项通过、1 项因该环境没有 `proxsuite` 失败；随后按既有环境分工在 rm75 完整通过，未安装或更改依赖。

独立生产 helper 数值复核使用 4,000 点复合输入：固定 5 ms 对旧 `lfilter`，LP 最大差 `1.7764e-15`，HP `3.4195e-14`；循环使用 2.15/5/10.91 ms 对独立连续矩阵梯形 oracle，LP 最大差 0，HP `4.4409e-15`。动态 observer 切入/切出、进入时同 timestamp、首入 2.15/10.91 ms、一次 gap seed 后运行中 gap 拒绝、重复样本后 command abort 均有集成反例覆盖。

实现者的真实日志时间回放记录在 `source_timebase_v6_replay.json`：两条既有 shadow 日志共 13,738 条控制记录，13,633 个 fresh、105 个 held；源间隔 2.1516–10.9102 ms，最大记录年龄 10.0962 ms，声明 15 ms 上限下无拒绝。该证据只验证已有测量时序与滤波数值，不是未执行 active 的运动、力精度或贴合效果证据。

关闭源指纹 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `contact_qp/runtime_source.py` | `21388d61f17cccc7c46233d4eeeecce74214b75050bff70e99ed04472bc4d483` |
| `admittance_common/variable_step_filter.py` | `22b2975d0e71c1436a598f0b2c8073d4aac4f72c21a6352da6bdf155b98440fb` |
| `admittance_common/observer.py` | `fe902230ba0939720420bdcc85230c0ec082b8d2597a67e75788ae0d44e02efc` |
| `admittance_common/controller.py` | `b0ae8a5a224106624cfd17bcf1ec2cc3b33f34551dfe1510e635db066317b4b3` |
| `joint_admittance_8dof/loop.py` | `ac52277e7065f46980bed69371952a3ac9df10215ed60252b18cb583fb65c8db` |
| `modes/contact_active.py` | `e224eabceb78ace34d9d1d83ed66d4e071d8fe0482a03c8aed3b5ad68b9b18f1` |
| `core/capabilities.py` | `68500d5e5eec959cc1faff28f12c5b92f2b313cf203e39a66541d5d52e7f06bf` |
| `api/arm.py` | `4ffffbf5bf42254d98900751058d561329282b2e957f7d62ca9dbcd4614d5a9e` |
| `realman8dof/daemon.py` | `d69a640ab4e92b39b093b505cc95ea1274eb290438749764b7c640e45caa40e0` |
| `ICRA_YM/script/scan_robot.py` | `8cbfdc1c0f32d9b3e90d147e540a61c2ecfff82a78be9fab196b67778a07821b` |
| `test_contact_qp_variable_timebase.py` | `03e6f4b195b9aea465e5ba1c80187ad6b97136c511cfcd74692025cf853f80e5` |
| `contact_qp/active_probe50.yaml` | `ddef6c9cdc640d430d980ab77cf57fa4db2862ed8dba6fdcad20e2b277c50cf0` |

# 控制器纯冗余清理：2026-09-08

基线：`2828d6d2`。本次仅清理不参与控制结果的计算、缓存和重复格式化，未调整控制参数。

| 位置 | 删除或合并的内容 | 等价依据 |
| --- | --- | --- |
| `native/wbc_rt/src/inner.cpp` 的 `solve_hqp` | `slow_lo/slow_hi` 整段轨道边界预计算 | 结果只被 `(void)` 丢弃；SVD 和墙边界辅助函数只写局部变量，没有修改实际速度边界或 QP 约束。 |
| 同上及 `include/wbc_rt/inner.hpp` | `last_C_ / last_lo_ / last_hi_` 三份约束缓存 | 私有字段全仓库没有读取；每条求解退出路径原先都重复复制动态矩阵/向量。实际约束 `C/lo/hi` 保留。 |
| 同上 | `preview_horizon` 的计算和私有参数；未读取的 `conflict_j` | 前者原先只被 `(void)` 丢弃；后者改传 `nullptr`，区间冲突布尔判断不变。协议字段保留。 |
| `control/joint_admittance_8dof/loop.py` 的 `_TickLogger._write_impl` | 最多六次数组再次复制；同一组 twist 字符串重复格式化 | 格式化进程拥有 IPC 快照，后续只读取这些数组。保留生产端快照复制和跨行速度计算所需的历史复制。CSV 两组列均保留。 |
| `control/joint_admittance_8dof/wbc_rt/client.py` | 同一 native 输出快照的状态重复转换 | 复用同一个整数，提交与错误处理分支不变。 |

未调整力矩死区、增益、限幅、碰撞体、CBF 约束、QP 预算、超时策略、CPU 配置、日志队列或观测进程调度策略。

## 验证结果

- Native 已重新编译，源码与构建标识匹配。
- 修改前后 CSV：1,440 行 × 560 列序列化内容逐字节一致。覆盖 verbose JSON 开/关、只读数组、缺失值、允许补齐的短数组、非有限值和跨行发送历史。
- 修改前后 native：三个起始姿态，各运行轨道联动和锁定两种模式，共 720 个周期，104 项非计时协议输出逐项精确相等，未放宽数值容差。碰撞检测开启，活动碰撞对为 6–7 对。
- 上述回放包含 718 次成功和 2 次已有的 `publication_infeasible` 判定，前后完全一致；该验证并不声称消除了所有可能的停机原因。
- 离线比较固定输入采样时间，且仅在测试配置中设 `max_solve_ms=0`，避免运行速度导致可选 QP2 的预算分支不同。最初没有固定采样时间时，轨道观测器因时间戳不同而产生差异；固定后精确一致。生产配置未改。
- 仅排除九项实测计时字段：`solve_ms`、`qp1_solve_ms`、`qp2_solve_ms`、`assembly_ms`、`fallback_ms`、`kinematics_ms`、`collision_ms`、`qp_total_ms`、`ipc_wait_ms`。
- 日志、native facade/通知、碰撞运行、耦合执行、CBF/轨道 pin、HQP 轨道补偿、P0 轨道补偿和轨道预约相关回归：**78 passed**。
- `git diff --check` 通过。未连接或驱动实机。

这次减少的是无用计算和复制开销，没有用实机测量提速幅度。临时对比脚本、基线二进制、回放结果和构建/测试日志位于 `/tmp/rm75_redundancy_20260908/`。

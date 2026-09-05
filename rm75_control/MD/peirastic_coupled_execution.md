# PEIRASTIC 联动内环与会话 DOF

本次软件迁移保留慢 rail、快 ARM 和 mid-ranging PI。硬件误差界没有经过认证；执行延迟模型仅供离线分析与在线旁路观测。新 8DOF 实机数据才用于评价相对改善，07 的污染拟合不配置执行模型。

## 会话与调用

Window A 是 DOF 的唯一会话所有者，启动为 8，任务和其他客户端继承它。`arm.get_dof()` 返回 `(ret, effective_dof)`；`arm.set_dof(7|8, after_current=True, block=1)` 请求任务边界切换。ACK 与生效分开，快照记录请求、生效、序号和 pending 状态。持续 SERVO 的零输入不会结束任务；应显式停止或替换任务。等待超时保留 pending，不把 ACK 当作已经切换。

公开运动参数中的 `secondary` 已撤销，旧字段返回迁移错误。运动方法、单位、八维状态格式保留；七关节 MOVEJ 用实时 rail 坐标补齐，7DOF 拒绝 rail 位移目标。标定在准备运动前切入 7DOF，结束后在停止边界恢复原选择；恢复失败须保留并报告停止状态。控制 IPC 使用 ABI 2 的 `peirastic_ctl_v2` / `peirastic_payload_v2`，检查尺寸、magic 和版本；旧进程需要一起重启，不能混用布局。

ICRA 01–15 默认在实验入口选择一次 8DOF；`--dof 7` 是相同开关的对照。每次 collect 写入独立 run 目录，旧固定路径以 latest 链接兼容，来源、哈希和版本随结果保存。

## 内环方程与权限

沿当前 rail 平移雅可比的任务前馈为 `dot(Jr_linear, Vaccepted_linear) / dot(Jr_linear, Jr_linear)`。采用上一拍准入进度来生成本拍慢参考；本拍 HQP 会再次检查可行性。完整六维加权伪逆不再作为独立 rail 速度源。

沿本机 base-Y 的轨道，构型为 `d = y_TCP - q_rail`，参考由一个 `d_ref` 状态连接 planner 目标：

```
u_nominal = v_Y_accepted - d_ref_dot + alpha_posture * PI(d - d_ref) + u_recovery
```

参考推进和 PI 输出使用相同恢复权限。慢参考继续经过低通、速度、加速度、jerk 和墙约束。QP2 保留该参考作为偏好；候选命令继承基于最终提交历史的慢速度、加速度、jerk 与 ARM 补偿余量约束。名义参考的瞬时幅值不是硬上限：把候选区间截在制动与名义参考之间会删除扫描需要的联动解。主任务可以在上述慢约束内使用 rail，不能绕过带宽限制。

QP1 用单一标量 `0 <= alpha_task <= 1` 准入六维请求，六个独立 Cartesian slack 被固定为零。当前任务使用已发生的 rail 估计贡献：

```
J_arm * qdot_arm + J_rail * rail_actual_estimate = alpha_task * Vrequested
```

预检包含两组名义 ARM 补偿变量：一组对应下一次 rail 刷新的候选命令，另一组对应低刷新率 worker 暂时保持当前 rail 贡献的情况。两组都实现相同准入任务，并受关节位置、速度、加速度、jerk 和碰撞约束限制，避免进度依赖一次尚未发生的 rail 写入。QP2 固定 QP1 的进度和任务映射，继承全部约束，再优化慢参考和姿态。

最终输出必须再次通过约束检查。不可行进入明确制动并记录 `task_paused`/原因；制动期间不宣称原任务仍被精确执行。优化器的可行性只是上述运动学模型中的可行性，不是 rail/ARM 延迟和接触响应的证明。

## 命令历史与观测

总参考和 base-only 影子参考采用相同整形链。HQP 从整形参考向可行制动速度减速时，同一个准入比例作用于 base 和构型贡献：`post_committed = authority * (total_shaped - base_shaped)`，`base_committed = total_committed - post_committed`。比例限制为 0–1；主任务需要的额外 rail 贡献归于 base，外部制动/直接关节命令不归于 PI。两条整形链都回写各自最终提交历史，PI 用最终构型贡献加回 `d_ref_dot` 反算，避免前馈被减速的差额变成虚假的积分。这是非线性整形链的确定性贡献归属，不是物理测量。部分恢复权限同时缩放积分驱动与比较基准；墙约束停止误差积分，仍允许反算退饱和。

加速度记录始终使用限幅前保存的旧速度和最终提交速度的差商。后置改写、制动或限位即使覆盖平滑约束，也如实记录实际差商，不把历史重新限幅成一个虚假的平滑值。

日志的四层含义：

| 层 | 字段 |
| --- | --- |
| 外部要求 / 准入 | `task_requested_*`, `task_accepted_*`, `task_progress` |
| 瞬时运动学模型 | `task_model_*`, `rail_model_*`, `arm_model_*`, `rail_preview_residual` |
| 控制器命令贡献 | `rail_base_raw`, `rail_base_shaped`, `rail_base_committed`, `rail_commit_authority`, `rail_post_committed`, `rail_total_committed`, `rail_pi_xi`, `rail_d_ref`, `rail_ref_acceleration` |
| worker 接收 / 处理 / 写入 | `rail_command_rx_seq`, `rail_command_processed_seq`, `rail_command_written_seq`, `rail_drive_write_seq`, `rail_command_write_mono_s` |

`twist_achieved_*` 来自反馈关节速度与 rail 估计的雅可比映射，仍非外部 TCP 测量。worker 写入成功不等于设备实际运动完成。未写入的缓冲命令不会增加设备写入序号；最终 RPM 量化后的速度用于 worker 命令历史。

原有 CSV 列保持前缀顺序，新列追加；缺失观测采用 NaN/空值，不能用零冒充测量。Python/native IPC v7 的布局为输入 616 字节、输出 1440 字节，启动检查版本和尺寸。

可选 `inner.execution_model_path` 指向 schema 1 的 JSON，包含七个 ARM **位置目标**通道与一个 rail **实际写入速度命令**通道，每个通道具有 `delay_s`, `tau_s`, `gain`，并注明独立拟合数据 `provenance`。`mode` 只接受 `observe`；即使 `validated=true` 也不启用控制补偿。日志写入模型 SHA256 和预测 TCP 分量。默认不加载模型。

## 理论边界和验证

[Gayadeen–Heath 的 mid-ranging 工作](https://www.sciencedirect.com/science/article/pii/S147466701530330X)将设计要求表述为带宽、灵敏度和抗饱和协调；此处核对的是出版方摘要，计划中的会议 PDF 链接也位于摘要目录，不能据此声称逐式复现全文。[Nguyen–Gosselin 的宏微系统研究](https://journals.sagepub.com/doi/10.1177/09596518251324684)将执行器动态差异、冗余分配和 PI 结合；其简化解耦模型不能直接充当本机八轴接触误差保证。本机 `d` 方程按机构几何推导。

离线回归覆盖连续请求的进度、13 秒纯 Z seek 到小幅 chirp、前馈反转与 PI 贡献、限幅后历史、低刷新 rail、ARM 延迟/增益误差及停止尾巴。实机验收应保持 8DOF，在相同构型比较固定/缓变 d*、空气/轻接触、下压/回撤、rail 减速/反转；同时报告 TCP 横向误差、任务进度、构型恢复和 rail 平滑性，不能仅靠整体减少运动获得改善。

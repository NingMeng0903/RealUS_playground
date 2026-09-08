# dm_holdout `publication_infeasible` evidence

调查时间：2026-09-08 06:41 CST。调查范围限定为现有文件和目录元数据；没有连接实机、启动控制器、调用 `PeirasticArm`/`NativeWbcClient`、读写活动 SHM、复位或清理任何 SHM。证据目录是本次调查唯一新建/写入的位置。

## 结论

用户给出的诊断

```text
qp1=primal_infeasible qp2=not_run solve_ms=2.56142 cbf=2
rail(lo=0,hi=0,prev=8.47069e-12)
```

不能从当前磁盘证据逐字复现：`8.47069` 和 `2.56142` 不存在于当前工作区或 `/tmp` 的保存日志中；06:00 以后也没有 controller CSV/native 故障快照。当前可见的辨识记录在 06:10 完成 dm_holdout 动态采样，随后在 06:10:27 写出了两个辨识 sidecar。这不能证明运动流程成功：`run_hardware_campaign` 捕获 `CampaignAbort` 后仍拟合已采集的数据并写出结果。因此，下面对约束来源的判断是代码路径加诊断格式的推断，不能声称已经锁定到某一个碰撞几何对。

`rail(lo=0,hi=0,prev≈0)` 本身是预期的 **8 变量 QP 中冻结第 0 个 rail 变量**，不是切换成真正的 7-DOF 求解器，也不是区间反转。`payload_id` 策略在 Python 层显式设置 `LockedStyle.HOLD`；native 层在 rail locked 且上一拍速度接近零时把 rail velocity box 设为单点 0（`api.py:127-137`；`inner.cpp:905-918`）。`lo==hi` 不触发 native 的原始 P0 区间冲突检查；该检查只在 `lo > hi + 1e-12` 时返回 `kQpP0Conflict`（`inner.cpp:1012-1018`）。

在给出的 `qp1=primal_infeasible`、`qp2=not_run`、`cbf=2` 组合下，最有证据支持的分类是：冻结 rail 后，两个启用的硬 CBF 行与其余 7 个 arm 关节的硬速度/加速度/jerk/跟随 box 发生不可行，或 ProxQP 对这组硬约束报告了数值上的原始不可行。QP1 的 Cartesian task 有 6 个 slack（`inner.cpp:1052-1059`），所以任务等式偏差不是硬不可行的首要来源；QP1 失败后 QP2 按设计不运行（`inner.cpp:1107-1119`）。`cbf=2` 只表示两条 active CBF row，native 输出没有碰撞 pair 名称，当前证据不能进一步指认是哪一对 geometry 或哪一个 arm 关节。

## 当前辨识文件时间线

文件均位于 `rm75_control/data/force_compensation/logs/`：

| 文件 | 大小 | mtime（本地 CST） | SHA-256 |
|---|---:|---|---|
| `payload_id_v2.csv` | 21,022,720 B | 2026-09-08 06:10:23.557366 | `ababbe4c5d31c699f46383d8c359c0042986b21013634610bbc7e803bcee2c16` |
| `force_id_phi_v2.json` | 16,027 B | 2026-09-08 06:10:27.461246 | `bd9c7e42ef44f23c6e964fb3d5037e9bd3e52e15707c417cc948163c903d311a` |
| `force_id_phi.json` | 5,781 B | 2026-09-08 06:10:27.468246 | `a67a10c22b1a6366fc87ba3b52800684d0256b11edf6303f45c7783c4ccdb2af` |

`payload_id_v2.csv` 有 39,510 行、44 列；所有行 `snap_ok=1`。关键 phase 如下（时间戳是 `recv_wall_ns` 转换后的本地时间）：

| phase | 行数 | local seq | 起止时间 | `record_enable=1` 行数 |
|---|---:|---:|---|---:|
| `dr_x` | 2,329 | 36,732..39,062 | 06:09:35.628991..06:09:47.634193 | 2,262 |
| `dr_y` | 2,327 | 39,072..41,404 | 06:09:47.685261..06:09:59.691613 | 2,264 |
| `dr_z` | 2,329 | 41,413..43,744 | 06:09:59.739074..06:10:11.745899 | 2,263 |
| `dm_holdout` | 2,274 | 43,754..46,029 | 06:10:11.798642..06:10:23.503490 | 2,265 |

`dm_holdout` 首行是进入 phase 的未记录零命令；最后一条记录行 seq 46028 的 rail 为 `0.4000003051757813 m`，命令为

```text
v=[-4.59517e-7, 5.18013e-7, 3.92852e-7] m/s
w=[-2.71137e-6, 3.80052e-8, -3.22159e-6] rad/s
```

最后一行 seq 46029 是 `record_enable=0` 的零命令收尾。这些数据与用户粘贴的质量和一阶矩数值一致，应作为同一次辨识的关联证据。最后的零命令后没有保存收尾保持段；结合 `_stream_traj` 在此后调用 `_enter_servo(label="payload_id_hold_dm_holdout")`，故障很可能暴露在动态段结束到保持段的交接处。由于没有故障当拍的 controller/native 数据，不能确定是最后一个动态控制周期不可行，还是保持切换过程触发不可行。

`force_id_phi_v2.json` 的关键结果为 `schema_version=2`、`payload.mass_kg=0.5285286168626534`、`first_moment_kg_m=[0.004425981445230431,0.011865439549033394,0.02713950381563346]`、`delay_online_effective_s=8.33e-17`、`delay_ci95_s=0.00352449`。绑定工具是 `gripper2`，wrench semantics 是 `environment_on_tool`，`force_sign=[-1,-1,-1,-1,-1,-1]`。这些 sidecar 在 dm_holdout 记录结束约 4 s 后写出。

## 当前没有匹配的 controller/native 快照

`find` 检查 2026-09-08 06:00 以后工作区的文件，没有 controller CSV/native snapshot；只有辨识 sidecar、rail calibration 文件和其他 agent 当前正在编辑的源文件。`ps` 没有 identify/controller/wbc/genesis 进程。`/dev/shm` 中可见的 `peirastic7_*` 最后修改时间是 2026-09-03，`scan_replay_*` 最后修改时间是 2026-09-05，均不是 06:10 的活动辨识快照。当前没有可以安全地用来读取该次故障输入/输出的 native SHM 快照。

用户粘贴的精确数字 `8.47069`、`2.56142` 也没有出现在工作区或 `/tmp` 保存 artifact 中。因此不能从本地文件得到故障时的完整 `box_lo[8]`、`box_hi[8]`、CBF Jacobian/lower 或碰撞 pair 名称。

## 为什么 rail 单点不是“7-DOF box fault”

1. `force_id_v2.yaml:12-18` 对辨识设置了 `secondary: payload_id` 和 `qp_aux.collision: true`，其余二级偏好关闭。
2. `campaign.py:676-687` 明确保留持久 session；8-DOF 会继续是 8-DOF，`payload_id` 只锁 rail。
3. `api.py:127-137` 关闭 rail extension/centering/arm task/manipulability，并把当前 rail reference 装入 `LockedStyle.HOLD`。
4. `inner.cpp:905-918` 在上一拍 rail velocity 小于 lock epsilon 且 0 位于安全 box 时写入 `lo[0]=hi[0]=0`。这正是 `rail(lo=0,hi=0,prev=8.47e-12)` 的含义。
5. native QP 仍使用 `Vec8`/8 个速度变量；`protocol.hpp:88-97` 的 `kRailLocked` 是 rail mode，不是 DOF 数量切换。

所以不要通过给 rail 单点加一个任意小 epsilon 来“修复”：那会允许辨识过程意外移动 rail，也不能解决 arm CBF 可行性问题。

## QP/CBF 路径证据

- `inner.cpp:1020-1045` 先把 8 个 box 作为硬不等式放入 QP，再追加碰撞 CBF。`Collision::build_rows`（`inner.cpp:265-325`）从最近 geometry witness 计算 Jacobian，并用 `-gamma*(distance-d_safe)` 生成 CBF lower；native 没有把 pair 名称写入输出。
- `inner.cpp:1036-1043` 对每个 CBF row 计数，`n_cbf_active=2` 只能告诉我们有两行 active lower bound。
- `inner.cpp:1052-1059` 说明 QP1 的 task slack 存在；因此 Cartesian task 偏差可由 slack 承担，而 box/CBF 仍是硬约束。
- `inner.cpp:1076-1089` 对 ProxQP 结果按原始硬 rows 再认证；`inner.cpp:1107-1119` 在 QP1 失败时把 qdot 清零并将 QP2 设为 `not_run`。
- `protocol.hpp:186-229` 的输出已包含 rail box、bind stage、上一拍 qdot、完整 8 维 `box_lo/hi`、`qp1/qp2` status、CBF count、硬/等式 residual 和 pause reason；但不包含 CBF pair/J/lower。

因此，严格分类如下：

- **已证实**：rail 单点是 `payload_id` 的锁定行为；报告状态对应 native 的 QP1 failure/QP2 未运行；该报告中的 `cbf=2` 表示两条 CBF 行；没有证据显示 `lo>hi` 的原始 rail P0 区间冲突。
- **最可能**：两个硬 CBF 行和 arm 关节安全 box 在冻结 rail 后无共同 qdot，或者 ProxQP 在这组硬约束上的数值不可行。
- **未证实**：具体碰撞 pair、具体 arm axis、是否是实际穿透或几何模型误差；现有 native telemetry 无法回答这些。

## 旧 phantom 故障不可混用

旧文件 `rm75_control/apps/logs/peirastic/run_20260908_032609.csv` 的 mtime 是 03:26:54，故障位于其 `t_wall_s=17.9867` 的旧 `track_hybrid` 记录。该行也有 `qp1=primal_infeasible, qp2=not_run`，但 rail box 是 `[-0.00792427, +0.00806611]`、上一拍约 `7.573e-5`，不是本次报告的 `0,0,8.47e-12`；旧行还有 `n_cbf=4`。`/tmp/scan_trace_native*.log` 是 2026-09-05 的旧 replay。它们只能说明历史上存在同类 failure code，不能作为 06:10 dm_holdout 的输入/输出证据。

## 最小修复建议（供 root 决策）

1. 保留 rail locked HOLD；不要放宽 `rail=0`，也不要把 8-DOF 标签改成假 7-DOF。
2. 先加一次失败前后的完整诊断采样：`qpik_qp1_status`、`qpik_qp2_status`、`qpik_n_cbf_active`、`box_lo/box_hi`、`qdot_prev_used`、`rail_box_lo/hi`、`rail_bind_lo/hi`、`qpik_hard_residual_max`、`qpik_equality_residual_max`、`qpik_fallback_reason`、`task_pause_reason`。若仍只有 `cbf=2`，不要自动猜 pair。
3. 为 native 增加只读 CBF 诊断（geometry pair/slot、distance、Jacobian、lower、每行 residual）；或者在进入动态辨识前做离线/同一状态的 arm-only 可行性检查。这样可以把“碰撞约束冲突”和 ProxQP 数值故障分开。
4. 保留现有碰撞约束和停止逻辑。当前证据不足以支持关闭 CBF、扩大速度界或放宽故障判定。
5. `campaign.py:821-833` 的 `servo enter -6` 是 `cartesian_velocity` 返回错误后包装出的 `CampaignAbort`，属于下游表现。`run_dynamic` 在 `campaign.py:1125-1126` 打印 `dm_holdout` 后才进入 `_stream_traj`；`_stream_traj:1078-1080` 先安装动态 servo，记录结束后 `1094-1096` 发送零 tick 并再进入 hold servo。因此仅凭一行 `servo enter -6` 不能判断是 dm_holdout 动态 QP、dr_z 收尾，还是 hold phase 安装失败；必须有同一 wall/monotonic 时间戳的 controller row。


## 补充核对与本次修改

`last_recorded_pose.json` 保存了 CSV 最后一拍（seq 46029）的关节位置核对。使用当前控制器 URDF 和 0.3° 位置边距，最近的机械臂位置界在 J4，仍有 **15.276°** 余量。这个已记录姿态没有位置超限，不能据此推断故障当拍的速度/加速度/jerk、CBF 或数值可行性；也不能套用此前 phantom PTP 的 J4 越界结论。

同一末帧姿态用当前离线碰撞模型计算，40 mm 内只有 `link_6_0/link_8_0` 一对，距离约 39.152 mm。它不是故障当拍的 native 约束集合，不能与终端 `cbf=2` 等同，也不能用它排除故障当拍的 CBF 问题。

控制器现在在发出停止请求之后保存 `rm75_control/apps/logs/peirastic/faults/*.json`，包括拒绝发布的完整 JointIkStep、实测/命令 q8、位置界和边距、控制配置及 QP 状态。JSON 仍不含 native 未公开的每条 CBF Jacobian 和几何对，后续判断须保留这个限制。保存失败不改变停止结果。

辨识的 `servo enter` 报错现在包含请求阶段标签和控制器故障消息；中断后继续拟合时明确打印 `collection incomplete`。这些是诊断改进，不代表本次 QP 不可行的根因已修复。没有运行实机辨识或扫描。

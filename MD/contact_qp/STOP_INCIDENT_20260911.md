# 2026-09-11：恢复原地轨链路，修正外环能量 QP 数值失败

本次按用户要求撤销新增的独立地轨停止进程，保留 V8r3 confidence 差分融合和命令模型能量罐。此次软件修订没有启动 controller、扫描或发送设备命令。此前用户明确要求的单次 FA24=0 操作与此次软件验证分开：当时三次读回目标和测量转速均为零，不能作为后续运行的停止保证。

## 1. 三类问题分开处理

**新增 guard 导致的启动故障。** 独立进程与原地轨线程同时访问同一个透明串口桥；用户日志出现了期望 7 字节却收到多个 32 字节数据响应的情况，并在启动阶段触发过期心跳。这支持通信响应混杂的判断。离线假服务器没有覆盖真实串口桥的多连接行为；引入该进程的做法不成立，已完整撤销，未通过增大超时掩盖。

**开启能量行后出现的数值失败。** 原失败记录位于当时的 `/media/camp/yameng/icra 2027/uncalibrated/003/attempts/RH_Per_L_DtP/002/contact_qp.jsonl`。第 1487 个候选的 QP 返回 `PROXQP_MAX_ITER_REACHED`。当时命令模型余额约 0.593 J，可用额度约 0.543 J，不是能量耗尽。加入能量行后仍禁用数值预条件，导致这组数据在严格残差要求下不能正常结束。修正只改变数值预条件，并把其状态纳入 solver workspace 缓存键。

**异常后的长时间停滞。** `4850.778121180` 的外环 abort/账本日志至 `5405.600423014` 的上层异常 abort 之间相隔约 554.822 s。QP 已在这段间隔之前返回失败，不能说 QP 求解持续了九分钟。根据代码顺序，上层处理包含原 `_fault_stop` 和内环候选清理，但未取得现场栈，无法确认具体阻塞函数。恢复原代码、修正数值触发点不等于证明所有异常停止路径可靠；这一项保留未关闭。

此前 `does not advertise contact_qp.active_v1` 也可能把控制器心跳失效误报为版本旧。现在有正确能力声明但心跳过期时，明确报 `Controller unresponsive`，保持原 0.5 s 检查门限。

## 2. 本次保留与撤销的范围

- `run_controller.py`、地轨 bridge、LW100 drive/Modbus/registers、内环 `loop.py` 及 native 源码与 `669beac6668f659aaa05f92c5425ca0235f0523b` 一致。
- daemon 保留 `515f774c405c85e1a9b13456d2048c32b81016f3` 的 V8r3/tank 能力声明；相对 669 的差异仅为这些能力声明。新增独立停止进程及其接入、测试均已删除。
- 外环 `qp.py` 在存在能量约束时启用求解器预条件；不修改目标函数、机械/力/能量约束、残差容差、5 ms 周期或发布证书有效期。
- 保留 confidence 差分请求与力矩导纳在同一个总角速度中的折中、α 进度和 4 N 名义力目标；4–4.5 N 修复门控、6 N 测量停止阈值不变。
- `active_probe50_v8r3_tank.yaml` 继续 `energy_constraint_enabled: true`。0.6 J 初始余额、0.8 J 容量、0.05 J 留额、0.05 s 逻辑持有期限不变；只改正注释，明确逻辑到期不表示设备自动停止。
- 用户力标定、数据、其他工作区改动保留，未回滚整仓库。

## 3. 数值复现和回归

新增测试 `peirastic/tests/test_contact_qp_recorded_energy_failure.py` 使用失败候选的名义动作、前一已接受动作、力和步长。失败行缺少原始六维 wrench，因此使用上一条完整日志的原始 wrench；这是捕获数值触发条件的测试，不是逐字节完整回放，更不是闭环效果实验。

同一主机、相同输入和约束的一次比较见 [数值结果](../../analysis_artifacts/contact_qp_stop_20260911/numeric_trigger.json)：

| 可用额度 | 修正前 | 修正后 |
|---|---|---|
| 0.543 J | 4129 次迭代，失败，16.28 ms | 9 次迭代，成功，0.895 ms |
| 0.0001 J | 120 次迭代，成功 | 9 次迭代，成功 |
| 0 J | 7 次迭代，成功 | 7 次迭代，成功 |

所有接受的解都通过原能量约束。零额度仍可解表示选取模型非耗能方向，不是绕过能量条件。该时间是单用例软件耗时，不是整体 200 Hz 的长尾上界。

撤销后的完整相关回归：**408 passed，249.85 s**，包含两个 10 万周期的基线/透明性对照、能量、融合、执行提交、特征、原地轨停止及控制器清理测试。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source rm75_control/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m pytest peirastic/tests/test_contact*.py \
  peirastic/tests/test_controller_cleanup.py \
  rm75_control/tests/test_rail_servo_stop.py -q
```

native 构建成功，二进制/源码协议哈希一致；`run_controller --dry-run --no-panel` 返回 `dry-run bind ok`。这些检查不连接机器人。配置检查确认 `command_energy_budget_enabled=true`、`differential_repair_revision=v8r3_confidence_balance`、`command_authority=outer_qp_original_ik`。独立 ultra 审核未发现保留改动的代码阻断项，同时保留异常停止未验收的结论。

## 4. 上次并未接入有效 confidence

上述失败尝试有 1486 个已记录控制样本，`feature` 全为空、`image_compatible` 全为 false。因此不能用这次记录判断 confidence 倾角修复是否有效；图像缺失时走的是机械控制和未知质量的推进策略。日志本身无法进一步区分发布器未运行、窗口版本不兼容或未收到观测。

置信度发布器必须读取同一份 YAML，并取得有效超声流。有效流接入仍需新录制日志确认，不能仅凭发布器启动时打印的窗口版本判定接入成功。

## 5. 手动使用

完整 controller、置信度和录制命令见 [V8r3 使用说明](V8R3_TANK_TEST.md#6-冻结配置与实际测试)。手柄、超声 UI 沿用原命令 `bash run.sh gamepad`、`bash run.sh ultrasound`，本轮没有修改启动脚本。录制省略 `--data-root`，保存到原默认 `/media/camp/yameng/icra 2027/uncalibrated/NNN/`。

由于已发生臂停而轨未停，本次不能给出“真人连续运行已验收”的结论。先由用户在无人接触状态核验启动、停止；确认后做仿体短轨迹，并检查有效 feature、差分请求、实际转角、能量和停止记录。没有自动启动任何测试运动。

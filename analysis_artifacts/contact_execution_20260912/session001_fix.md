# /001 扫描中断修正

本次只修改软件、编译 native、运行离线测试。没有发送机器人或地轨命令，没有修改原始扫描数据。

## 实测定位

数据：`/media/camp/yameng/icra 2027/uncalibrated/001/attempts/RH_Per_L_DtP/002、003、004`。

三次扫描的最终拒绝均是 `energy_reservation_rejected`，当拍图像和力源有效。末拍外层候选与名义命令完全相同，已没有新增视觉修复；最终命令模型却额外产生地轨 base-Y 速度。

| 尝试 | 最终模型相对候选的 base-Y 增量 | 额外输出功率 | 剩余可用功率 |
|---|---:|---:|---:|
| 002 | 3.303 mm/s | 7.893 mW | 5.949 mW |
| 003 | 3.431 mm/s | 9.499 mW | 5.093 mW |
| 004 | 1.821 mm/s | 4.812 mW | 3.766 mW |

累计最终模型相对候选模型的额外输出约为 0.0530、0.0541、0.0528 J，与接近用尽最初 0.05 J 可支配能量一致。旧日志没有内层 qdot/观测重定位状态，不能精确分离其中虚算和真实未来命令差异的各自贡献，也不能证明修正后原轨迹必然完成。

首次 MOVEJ 失败发生在录制启动前，没有相应 raw.h5。代码核查发现这个 daemon 入口没有安装现有后台补偿力 observer，模式编译期间只有主循环发布力的链路会出现空窗。这是已修正的明确缺口，不冒称已测得该次原始传感器一直在线。

## 代码行为

1. 最终地轨速率改为 `step.qdot[0] + (published_rail - proposed_rail)/dt`。同一次 proposal 的两个位置已包含相同观测重定位，因此不再把观测重定位误算成命令位移。真实的最终限位改写仍计入。
2. 冻结功率约束进入 Python/native 的 QP1、QP2：`W_base @ J_full @ qdot >= -task_power - beta*available/hold`。使用完整命令 Jacobian，包括未来地轨命令列；没有换成测得地轨速度的任务模型。求解边界预留现有求解器认证容差，零力行不增加虚假的正下界。下游改写和最终账本仍严格复核。
3. Python 成功提交后的 jerk 历史使用最近两次已提交速度，去掉额外滞后一拍。native 未发送失败撤销 proposal 历史，保留测量重定位。
4. daemon 为 relay 安装独立、同模型的补偿力 observer；独立运动学 Data 已存在。后台持续消费不同的真实样本，任务已有 50 ms 发布所有权结束后接手。发布加锁并检查源时间顺序，恢复时旧任务样本不能覆盖新数据。没有把接收/发布时间冒充采样时间，也没有改变 record 的 100 ms 或发送的 15 ms 门槛。
5. 新增 `contact_qp.final_command_power_v2` 能力，native ABI 为 **v10，752/1496 字节**。旧二进制和旧 controller 必须重启/更新后使用。记录最终 qdot、地轨 proposal/发布位置及模型/执行地轨速度；最终拒绝同时记录当时能量并向采集端保留具体原因。

罐仍为 **0.100/0.150 J，0.050 J 留额**；名义任务供能、旧 epoch 责任、50 ms 期限与真实停止条件未改变。图像仍为 `pause_visual`，偶发过期撤去视觉请求，新鲜力任务继续。没有加入能量重置、自动续期、虚假补能或自动解除真实故障。

## 验证

- 初次全量：699 passed、5 failed。5 项失败来自测试夹具未提供新增最终模型函数；夹具补齐后全部复验通过，详见后两份日志。
- 最终 focused 回归：**281 passed**，native CTest **1/1**，配置验证通过。[日志](session001_focused_final.log)
- 另一次 Python 针对性复验：**62 passed**，覆盖上述 5 项、低能量及提交/撤销行为。[日志](session001_targeted_python.log)
- ABI 相关附加检查：**3 passed**。[日志](session001_protocol.log)
- 独立 ULTRA：无剩余阻断发现；独立 Python/relay/daemon **40 passed**、native **6 passed**。[审查](session001_ultra_review.md)
- 三组失败时刻的原始关节位置、旋转、力和预算作为固定回归输入，每组在两种 backend 上执行 **400 次连续提交/结算**，不重置罐；同时覆盖零力、真正不可行和拒绝回滚。[输入](session001_energy_failures.json)
- 300 ms 模式编译空窗的合成测试保持真实力时间戳，重复样本不重新消费；真正停流仍会变旧。后台补偿的离线额外计算中位 0.134 ms、P95 0.146 ms；这不是实时期限保证。[计时](relay_force_cost.json)

固定输入和合成测量不等同于原扫描的完整闭环重放。本轮没有新的实机扫描，不能据此宣称侧边贴合改善、物理被动性或任意机械约束下都能继续。

## 完整命令

离线复验（不启动硬件）：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/test_contact_execution.sh --full
```

结束旧的 controller 和 record 后，终端 A：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh controller
```

启动应看到 `force relay=continuous; independent mode-boundary compensation`。

终端 B，原命令保持兼容：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record \
  --force-profile icra \
  --speed-m-s 0.005 \
  --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

进入接触扫描时应出现：

```text
[CONTACT_QP] final_power=inner_qp1_qp2; velocity_model=rebased_command_delta_v2
```

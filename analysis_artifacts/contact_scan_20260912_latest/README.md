# 最新 /001 控制与图像审计（2026-09-12）

对象：`/media/camp/yameng/icra 2027/uncalibrated/001` 当前的五条完成扫描，L/C/S DtP、L/C PtD。这是本轮重新读取的数据，不能与之前同名目录中的失败 attempt 混用。原始 H5、JSONL 和用户改动均保留。

**结论：罐和 continuous confidence 已启用；本批 L_PtD 黑边不能归因于罐耗尽。视觉输出的最低要求被满足，并不证明声学贴合已修复。发现并修正了 confidence 工作者多等一个限频周期的性能问题，以及控制周期内同步打印的阻塞风险。**

## 1. 当前实现与能量收支

五条扫描记录的 21 个控制源文件指纹相同，且与本轮修改前 HEAD `76de62392bfec801136e7c32d4374ceafb71d63e` 对应文件逐一一致。记录中的较早 git baseline 不应取代逐文件指纹核验。本轮修改了 feature 工作者、重复计算和控制提示输出，需要重启相关进程才会加载。

L_PtD 从 0.100 J 开始，最终约 0.112236 J；累计正模型端口输入约 0.016110 J，罐的累计支出约 0.003875 J。模型负端口功约 0.137535 J，其中名义任务源支付约 0.133660 J，其余才从罐扣除。未用完的名义任务授权不会计成回充。这条扫描有实际的模型收支，没有永久空罐。

五条记录的完整流式统计见 [scan_counts.json](scan_counts.json)，以下回充/支出是逐 `logical_epoch_work` 结算的正/负罐增量，不把名义源支付重复计成回充：

| 扫描 | 最低 / 最高余额 J | 最终余额 J | 累计回充 / 支出 J | 视觉暂停次数 / 总时长 |
|---|---|---|---|---|
| L DtP | 0.09717 / 0.10973 | 0.10417 | 0.02694 / 0.02277 | 24 / 0.233 s |
| C DtP | 0.08822 / 0.10647 | 0.08916 | 0.01480 / 0.02564 | 19 / 0.142 s |
| S DtP | 0.09933 / 0.10835 | 0.10060 | 0.02588 / 0.02528 | 26 / 0.197 s |
| L PtD | 0.09994 / 0.11296 | 0.11224 | 0.01611 / 0.00387 | 18 / 0.155 s |
| C PtD | 0.09947 / 0.10724 | 0.10696 | 0.01414 / 0.00718 | 37 / 0.319 s |

全部最低余额高于 0.050 J 留额。暂停次数多，但这是包括接触建立在内的短暂停顿事件数，五条总计约 1.047 s；不能把每条 210–300 ms 的有效图像年龄当成每次视觉停用时长。

当前账本定义六轴冻结模型功率 `P = W_hat · V_final`，正号代表环境向机器人输入。名义任务源授权 `S = max(0, -W_hat · V_nominal)`。容量上限截断之前：

```text
E_dot = P + min(S, max(0, -P))

P > 0       : 回充 P
-S <= P <= 0: 名义源支付，罐余额不变
P < -S      : 从罐支出 -(P + S)
```

余额有 0.150 J 上限、0.050 J 留额；初始值仍为 0.100 J。六轴功率先相加再判定收支，不能把某个轴单独吸收的能量全部重复计入，同时漏掉其他轴的支出。正阻尼也不会自动生成一笔未经证明的充值。

这个账本在**冻结命令模型及显式名义任务源**下收支自洽；它不等于人体接触物理端口的被动性证明，不保证任意工况自动回满，也不单独保证振荡衰减。附件关于联合存储函数、实际执行功、延迟影响的限制仍适用。`c_k` 耦合和物理区间功证明尚未完成，不能拿 `c_p` 代替。

## 2. 地轨重新对齐为什么不再算作命令速度

旧计算将观测位置与上一规划目标的差，除以控制周期后混入命令速度。它包含观测重定位，不能直接解释为这一拍申请的速度。

现在从求解器的命令 `qdot` 出发，仅将同一 proposal 到实际发布目标的裁剪差 `/dt` 加回地轨速度。对于本拍地轨位置，等价于分清：

```text
proposal = rebased_position + proposed_increment
command_rate = proposed_increment / dt + (sent_target - proposal) / dt
```

这是按变量定义纠正混算，不是根据罐的曲线删掉耗能项。真实下发裁剪仍保留在最终模型，最终功率约束仍贯通 inner QP1、QP2 和发布复核。

**边界：**若真实地轨伺服为了追赶位置误差而运动，那部分物理功不能据此说成零；仍需要真实执行速度、力与时间对齐来核算。当前 `rebased_command_delta_v2` 只定义命令模型，测量端口仍是诊断，不能据此升级为物理被动性结论。

## 3. L_PtD 左侧黑边为什么仍在

原帧通过 `feature.frame_seq == ultrasound/frame_index` 精确匹配，见 [原帧图](L_PtD_exact_frames.png)、[控制与测量曲线](L_PtD_trace.png)、[逐点数据](L_PtD_frame_evidence.json)。图上的青线是当前 confidence 统计区域的深度下边界（图像前 22%）。

| 参考时间 | 左 / 中 / 右 confidence | 视觉最低差分请求 | 名义差分 | 最终发布模型差分 | 罐余额 |
|---|---|---|---|---|---|
| 17.200 s | 0.749 / 0.863 / 0.993 | 0.0544 mm/s | 0.8402 mm/s | 0.6794 mm/s | 0.11294 J |
| 22.461 s | 0.758 / 0.928 / 0.995 | 0.0316 mm/s | 1.3493 mm/s | 1.0874 mm/s | 0.11281 J |

17.2 s 时名义工具 y 角速度约 −1.55°/s，最终发布模型约 −1.26°/s；视觉 QP 的新增角速度约为数值零。因为修复项要求的是**总速度达到一个最低目标**，原名义力矩/姿态支路已经超过这个目标，QP 就没有理由再额外旋转。这符合当前实现，却不构成图像闭环改善的证据。

还有两个限制：

- 当前三个横向窗口只汇总浅层 confidence，不能将整幅深部黑边直接量化为实际接触缺口。浅层强回声、窗口平均和图像组织结构都可能让这个标量与人眼看到的侧边情况不同。
- 单侧 0.795、另一侧健康时，缺额只有约 0.005，小于现有 0.03 死区，因此允许零视觉请求；并不是只有达到 0.8 才完全停止新增修复。

实际 TCP 的测量姿态曲线也显示工具确实在转动，并非始终保持虚拟路径平面的姿态。但扫描位置同时变化，不能由这些移动中的图像证明当前方向就是改善声学耦合的正确梯度，更不能证明加大同方向旋转必然消除黑边。当前方向是带注册符号的左右缺额提示，尚不是标定后的 `d(confidence)/d(angle)`。

因此本轮没有依据黑边截图直接翻转方向、提高增益、改质量阈值或增加罐信用。要完善这一环，需验证同位置的小角度响应及执行延迟，建立能区分表面耦合与深部声影的质量目标，并避免把“超过最低运动目标”标成“贴合已改善”。一帧一次证据更新、continuous 请求和现有门控继续保留。

## 4. 图像约 210–303 ms 的原因及本轮修正

当前有效年龄是 `控制时刻 - (主机收帧时刻 - 登记成像延迟)`。登记延迟约 **151.964 ms**。所以显示 210 ms 不等于 Python 计算了 210 ms；它还包含登记延迟、采样相位、传输和等待。

发现旧工作者在一帧**计算及发布完成后，再等待 1/max_hz**；默认 20 Hz 就再等 50 ms。这使一次约 30–50 ms 的计算形成约 80–100 ms 的更新间隔。新帧在约 210 ms 时恢复，继续保持约 90 ms，就反复跨过 300 ms 的视觉暂停阈值。

本轮改为以**实际处理开始时刻**限频，间隔为 `max(处理耗时, 50 ms)`；等待时持续取最新帧，超时后没有累积补赶。另外复用已经得到的 window quality 和 unknown-column 结果，避免重复计算。

24 个真实 ROI 帧、72 组交错比较中，新旧 confidence map、观测、缓存特征及相同处理耗时字段下的序列化内容完全一致。离线提取耗时中位数从 28.883 ms 到 28.363 ms，主要收益来自消除额外等待。仅按这些耗时计算的调度模型从约 78.9 ms 到 50 ms；这**不是新实机帧率或端到端延迟实测**。

图像过期仍只撤去视觉请求，`pause_visual` 下新鲜力任务继续，未通过延长 300 ms 门限或篡改图像时间戳解决问题。偶发暂停可以继续扫描。频率改动后仍需观察实际 CPU 负载、图像间隔和力源年龄；增加特征吞吐不能保证其他计算不受影响。

## 5. 力恢复和回撤

`wrench_source_expired` 指当前候选使用的冻结力样本在发布复核前超过期限；它不等价于 TCP 读取很重或传感器已经断流。读取、观测器、外层求解、inner QP、地轨 proposal、复核及调度等待都消耗该样本的年龄。求解后的新 UDP 样本也不能换一个时间戳去冒充旧候选基于新测量。

五条扫描在推进阶段的力源入口年龄中位数约 2.12–2.25 ms，p95 约 4.49–4.62 ms。共有 7 次发布复核的 `wrench_source_expired`（L DtP 1、L PtD 3、C PtD 3），以同一 `control_id` 的原始 source 时间戳计算，复核时真实年龄为 **15.044–17.443 ms**。另有 C PtD 的 1 次 dispatch 拒绝。入口新鲜、最终复核超龄可以同时成立。不能把 `review_time - created_time` 的 5–12 ms 当成力样本年龄。

本轮将图像暂停、恢复和终点提示从控制拍内同步 `print(..., flush=True)` 改为现有有界后台打印队列。终端输出阻塞或队列满时不会把控制拍卡在 stdout；完整结构化事件仍照常进入记录。这消除了一个可避免的阻塞点，不能仅凭源码断言此前每次力超龄都由打印导致。

现有恢复逻辑继续：确定未发送的短时失效撤销本拍 proposal，沿原有效双设备命令期限等新测量重算；不续租、不补算路径进度、不重置罐，也不重新录制。真实超力、持续失联、原期限耗尽或部分发送依然需要故障处理。

这五条当前 attempt 的结果都是 completed。用户所贴 `reference complete; endpoint convergence` 后切换 `TRACK_CARTESIAN` 并回撤，是路径结束流程，不是图像暂停导致强制中断。FA24/Modbus 日志不能用这批成功结果证明地轨通信始终健康；本轮未连接或写地轨寄存器。

## 6. 验证及复跑

- 相关软件回归 **149 passed**，见 [regression.log](regression.log)：包括短时恢复、图像过期、满打印队列、最新帧调度、数值一致性及能量收支。
- 独立 **ULTRA** 复核无阻塞发现，独立执行其中 56 项通过；能量审查另外确认当前模型的收支式及其物理边界。
- 特征优化由 **HIGH** worker 实现；无硬件扫描、寄存器写入、自动复位或服务重启。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source rm75_control/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PYTHONPATH=.:rm75_control:src:$PYTHONPATH python -m pytest -q \
  peirastic/tests/test_contact_qp_execution_runtime.py \
  peirastic/tests/test_contact_qp_features.py \
  peirastic/tests/test_contact_qp_feature_pacing.py \
  peirastic/tests/test_contact_qp_live_confidence.py \
  peirastic/tests/test_contact_qp_transient_feedback.py \
  peirastic/tests/test_contact_qp_active.py \
  peirastic/tests/test_contact_qp_task_power.py \
  peirastic/tests/test_contact_qp_command_budget.py
```

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
OPENBLAS_NUM_THREADS=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python \
  analysis_artifacts/contact_scan_20260912_latest/plot_l_ptd.py
OPENBLAS_NUM_THREADS=1 /media/camp/EXT_DRIVE/envs/genesis/bin/python \
  analysis_artifacts/contact_scan_20260912_latest/summary_counts.py
PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:rm75_control:src /media/camp/EXT_DRIVE/envs/genesis/bin/python \
  analysis_artifacts/contact_scan_20260912_latest/feature_benchmark.py \
  --baseline 76de62392bfec801136e7c32d4374ceafb71d63e \
  --source '/media/camp/yameng/icra 2027/uncalibrated/001/attempts/RH_Per_L_PtD/001/raw.h5' \
  --frames 24 --repeats 3
```

## 7. 加载新代码的完整命令

结束当前扫描、退出对应旧进程后，在三个终端分别运行。已有 ultrasound 与 gamepad 入口不变。

控制器终端：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh controller
```

confidence 终端（必须重启这一进程，单独重启 controller 不会加载工作者修改）：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh confidence --max-hz 20
```

启动信息应有 `pacing=start_to_start_latest_v1`。采集终端：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh check
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record \
  --force-profile icra \
  --speed-m-s 0.005 \
  --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

控制器仍应打印 `visual=continuous`、`image_loss=pause_visual`、`tank=0.100/0.150 J`、`reserve=0.050 J`、`fusion=shared_total_velocity` 和 `velocity_model=rebased_command_delta_v2`。

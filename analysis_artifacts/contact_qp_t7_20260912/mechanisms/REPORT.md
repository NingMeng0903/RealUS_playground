# T7 energy tank 与 confidence 实录审核

对象：`/media/camp/PEI_T7/icra 2027_contact/uncalibrated` 的全部 49 份 `contact_qp.jsonl`，分别属于 yangming、yuhan_L、yuhan_R、zhongyao2。49 份对应 35 completed / 14 failed；48 份进入扫描。这里按文件夹名称区分数据，不根据 trial 中的 RH/LH 推断被试真实侧别。

**结论：tank 的结算/预留确实执行，confidence 确实影响控制和最终发布；但这组扫描没有逼近 tank reserve，不能把重采集归咎于 tank 耗尽。图像修复远未在全部扫描中完成，也不能把有修正命令直接当成图像改善的因果证据。**

## 数据完整性与范围

- 49 份配置完全相同，记录的源代码 SHA256 集也完全相同；读取期间无变化，无 malformed JSON，无 dropped records，全部有 recording_close。
- 共 246,483 个 control_sample、246,468 个成功 publication、739,419 个 logical_epoch_work。
- 按 session 共享时钟的 tracking_observed → path_done 定义扫描段；失败以最后控制样本作为扫描终点。总扫描段 1,158.149 s。
- 流式核账，不加载整个 4 GB 日志；只读原数据与代码，未控制硬件、修改生产代码或原始数据。`metrics.json` 带各日志 SHA256、逐 attempt 结果；`audit.py` 为可重跑脚本。

## Tank：执行了结算，但本批次没有触底或逼近能量约束

统一配置：initial = 0.100 J，capacity = 0.150 J，stopping reserve = 0.050 J；energy_constraint_enabled = true；task_power_source = nominal_command；settlement_port = logical_final_model；energy_assurance = two_port_command_model。

|指标|全部 49 日志|
|---|---:|
|最低 tank balance|0.087545476 J|
|最低 available（已扣 reserve 与预留）|0.035699744 J|
|最大预留|0.014300053 J|
|低于 reserve / available 接近 0 的事件|0 / 0|
|已提交命令最小约束余量|0.713994872 W|
|最终余额范围|0.087546585～0.150000000 J|
|发生额外 tank 扣款的 attempt|49 / 49|
|发生 capacity 溢出丢弃的 attempt|8 / 49|

最低 balance 在 yuhan_L/RH_Per_L_PtD/004；最低 available 在 yangming/LH_Per_L_PtD/001。前者仍高于 reserve 0.037545 J，后者仍有 0.035700 J 可用。

逐 work 验证以下模型结算：P_port = wrench·final_command；P_source = max(0, −wrench·nominal_command)；实际任务供能 = min(P_source, max(0, −P_port))；tank 增量 = (P_port + 实际任务供能) × dt，再按 capacity 截断。每个 work 的 port/source/tank/capacity 数值、每个 balance 和累计字段、commit 的功率、时间边界以及 reservation liability 均与复算相符（最大局部误差 0，跨 49 日志累计恒等式最大误差 1.94e−15 J）。没有 work 时间重叠，没有未提交/未发送 command 被计入 work。

49 个独立 tank 的累积账（不是一个连续 tank）：port work −3.400798 J；名义任务供能 +4.583059 J；正 port 回收 +1.597285 J；超过名义任务供能的额外扣款 0.415024 J；capacity 溢出丢弃 0.173665 J。末值相对各自初值合计 +1.008596 J。

因此“余额往往上升/没归零”并不表示 tank 没工作：nominal_command 为原本任务输出供能，tank 主要承担超出这份名义供能的部分，并接收正 port 回充。本组数据支持结算机制有效执行，**不支持它已在扫描中明显限制过任务或防止了某次能量失稳**。未做禁用 tank 的配对试验。

全部 completed 在结束时有 logical_output_stopped_actual_tail_unknown，14 份 failed 在后续结算时有 logical_epoch_expired_actual_tail_unknown；这是停止/命令过期后的尾部状态，不能当成“tank empty”。本审核观察到 publication review/retry 的原因是 wrench_source_expired / dispatch_source_expired，没有能量余额不足拒绝。

“two_port_command_model”是命令模型账本。日志声明 physical_port_assurance = unverified，physical_certified = false，实际执行尾部 unknown；不能把上述恒等式当成机械实际端口无源性证明。

## Confidence：有计算、有请求、有额外转动，但修复并不完全

统一视觉配置：continuous、image required、dropout_policy = pause_visual、balance_deadband = 0.03。图像随机游走 confidence 的 L/C/R 为上 22% 深度的窗口均值，侧窗为宽度 4–34% 与 66–96%；不是全深度图像质量或机械脱耦标签。

- 扫描期使用到 23,657 个按 source_id/frame_seq 去重的图像帧；其中 7,884 帧（33.3%）至少一侧 q < 0.8。
- 非零差分请求累计 796.010 s，占扫描 68.7%。请求且成功发布的 136,220 个周期中，133,557 个（98.05%）最终 wy 的方向与请求一致。
- 请求期间，已有名义总角速度已满足请求的周期占 39.2%。沿请求方向的额外 QP 修正持续 571.673 s；候选达到完整请求的周期约 41.5%。这是带软短缺惩罚的总速度请求，不能要求每个周期完全达到。
- 名义总速度包含此前已接受的视觉状态，因此 QP − nominal 只衡量这一步所有 QP 约束带来的差异，不能当成整个视觉控制的全部角度；已有名义满足请求时没有新增角速度是正常情况。反过来，已有名义反向且幅值较大时，QP 可能先减小反向速度，最终仍未转成请求方向。
- QP candidate 与最终 publication 也不是逐点相同。例如 yuhan_L/S_PtD/007 的 +23.089 s，名义 +2.470°/s → QP +0.852°/s → 发布 +1.362°/s；该时刻请求负向，QP 明显减小原正向运动，但最终尚未转负。该项是执行链路实录，不能称全部时刻都完全达到视觉目标。

### 同帧图像与命令例子

四个数据组各选 completed 中 QP−nominal 绝对积分最大的扫描，展示最差侧 q、末帧与最大沿请求修正，共 12 张图。保存 JPEG 的 exact frame_index 与控制 frame_seq 匹配；同帧复算 L/C/R 与在线日志误差均为 0。

|扫描 / 帧 / 扫描时间|qL/C/R|请求 mm/s|名义→QP→发布 wy °/s|意义|
|---|---|---:|---|---|
|yangming RH_S_DtP/001，242139，+29.096 s|.572/.812/.907|.630|−.402→−1.106→−1.097|左侧明显暗带，确实增加负向转动|
|yangming RH_S_DtP/001，241753，+16.213 s|.837/.635/.533|.564|+4.118→+4.118→+4.118|右侧暗带；已有名义正转动超过所需，QP不再额外加转|
|yuhan_L RH_S_PtD/007，346571，+11.982 s|.529/.704/.818|.534|−.775→−.972→−.972|左侧低 q 时增加负向转动|
|zhongyao2 LH_S_PtD/001，301981，+29.959 s|.487/.839/.945|.882|−1.252→−1.575→−1.575|低 q 产生较强请求，得到额外负向转动|

图像改善不能由“控制在工作”推出。四条代表扫描的末帧侧 q 仍分别低至 .660、.677、.730、.670。yangming S_DtP 的原图可见暗带先位于右侧，末尾转为左侧，仍未消除。35 条完成扫描按独立帧统计后半程相对前半程的侧 q < .8 比例：9 条下降、20 条上升、6 条近似不变（差异 ≤ .001 算不变）；10 条扫描末四分之一超过 90% 帧仍有一侧 q < .8。

这些前后变化来自不同空间位置、接触状态和名义力矩调节，**既不能证明视觉总体有益，也不能据此证明视觉导致变差**。支持的结论是当前阈值能探测并驱动修正，但不是所有扫描都恢复了浅层侧 confidence，更不能保证所有暗边或深部声影消失。完整逐扫描前半/后半/首末四分之一见 `visual_detail.json`。

## Image warning：本批记录实际验证了 pause_visual 的短暂停/恢复

扫描内共 5 段 transient_stale，分布于 3 个 attempt，累计 33.129 ms，最长 8.805 ms，全部恢复：

|attempt|扫描后时间 s|持续 ms|
|---|---:|---:|
|yuhan_L RH_C_DtP/001|18.155 / 23.753 / 24.154|6.076 / 5.721 / 5.447|
|yuhan_L RH_S_PtD/001|12.178|7.080|
|yuhan_L RH_S_PtD/007|3.819|8.805|

这 5 个周期都记录 fresh force，force age 为 1.48–3.08 ms；视觉 request = 0，但都有成功 publication。其实际行为与日志“visual paused; fresh-force task continues”一致。

如果把启动/seek 也纳入，全部 control_sample 中共 20 段非 ok 图像状态，全部恢复，累计 276.260 ms，最长 61.250 ms；其中 14 段发生在扫描前，1 段属于没有进入扫描的 attempt，其余为上述 5 段。启动 missing 和运行 transient_stale 不能混成相同告警数量。运行 image_feedback_unavailable / recovered 显式事件也是各 5 次。

## 产物

- `metrics.json`：49 日志配置、hash、完整能量核账、48 扫描视觉/请求/暂停统计及代表控制周期。
- `visual_detail.json`：全阶段缺图段、独立帧前后质量统计、12 张 JPEG 同帧复算结果。
- `*_timeline.png`：四个代表扫描的 tank、L/C/R confidence、视觉请求、名义/QP/发布 wy。
- `*_frames.png`：相应原始 JPEG 与 confidence map；青线为上 22% 深度，绿线为窗口边界。
- `*_balance.json`：49 份下采样 tank 曲线；精确核账来自完整 work 序列，未使用下采样曲线。

# 原 force law 的测量 / 控制推进接缝

2026-09-10，只读源码核对，供 high 实现和 ULTRA 验收。未修改 runtime、IK、native、rail worker，未运行硬件。本文补充 `RUNTIME_ADAPTER_MAP_V2.md` 中“重复源”的实现边界：**重复有效测量仍应执行一次新的控制周期；不能跳过整个 force compute、把 200 Hz 降成 100 Hz，也不能用 control tick 编造新测量。** baseline/shadow 保留原执行数值路径；新的显式 active 接缝缺省关闭。

## 身份、时间和已证限制

- `source_sample_id`：源身份/代际加 `snap.t_s`。relay seq 会在相同 `t_s` 下增加，不代表新测量。相同 wrench 数值也不代表重复帧；零变化是真实测量的正常情形。
- `control_step_id`：每次 prepare 的新事务序号，用于一个 pending、commit/abort 恰好一次。即使 source 未变，也必须能产生新的控制 proposal。
- `dt_control`：真实控制周期，现 `dt_actual` 经 compute 限幅得到 `dt_contact/dt_flow/dt_eff`；驱动导纳和计时。`dt_source`：两次新源接收时间之差，用于源差分/采样滤波。`feedback_velocity_valid` 与“本拍刚有新反馈”是不同概念。
- 现磁盘配置 `controller.yaml:32` 为 `realtime_push.cycle=1`，`state_relay.hz=200`；observer.from_yaml:338–341 也按 cycle 得 200 Hz。100 Hz source / 200 Hz control 是必须验收的异步场景，本文没有通过实机记录证明实际传感器就是 100 Hz。`snap.t_s` 是 UDP 主机接收时间，没有独立 F/T 硬件 acquisition timestamp，不能宣称内部 F/T 的每个采样都可辨识。

`NominalCommandTransaction.prepare:97`、`TorqueTilt.prepare:205` 当前把 `measurement_id` 当作每次 prepare 必须递增的唯一序号；`check_measurement_id:59` 对重复值抛错。直接传源 ID 会在正常重复拍报错；直接传 tick ID 会把重复源当成新观测。最小改动是在这两个 adapter 明确分开事务身份与源新鲜度，再把新鲜度/源间隔透传原 compute；不复制 AdmittanceController，不回滚已消费测量。现 `legacy.update:65` 没有透传这些参数，未知 kwargs 被吞掉，因此仅在 runner 加字段并不起作用。

## 原 compute 内准确边界

以下 `controller.py` 指 `rm75_control/rm75_control/control/admittance_common/controller.py`，行号为核对时源码。

| 位置 | 现有状态 / 混合职责 | 最小接缝 |
|---|---|---|
| runner `loop.py:6701` → `CompensatedForceObserver.update:202` | 每调用一次追加 `_pose_ring/_t_ring`、`_n_updates`、regressor history、因果 LPF；完全无源去重 | 新 active 源缓存：新源只调用一次 update，重复源复用输出；原 mode 调用次数保持原样。缓存本来消费过的最后源避免 hot-install 再消费。不能通过同源新 relay seq 绕过 |
| observer `update_leftover:124` | air/still 残差估计滤波；contact 上升冻结事件 | bias 观测只新源，用源时间间隔；接触冻结可随已有效的持续状态更新。缺源不补零 |
| compute `1400–1444` → `PhysicalContactTracker.update` | 持续高/低力定时、FREE/CONTACT/SUSPECT_LOSS/LOST 状态机；这里没有采样滤波或样本历史统计 | **仍每 control dt 用当前有效保持值推进**。状态转换自然只发一次 acquired/lost；每拍重算返回值，不能重复使用上拍带 acquired=True 的 result。持续计时属于时间推进，不是制造新测量 |
| compute `1450–1489` | `_contact_time_s`、episode detach/rearm 持续计时 | 每控制周期推进；否则 force ramp、0.1 s 确认/脱离阈值等按源速率错误变慢。只在真正 transition 触发一次事件 |
| compute `1502–1508` | `_v_tcp_z_prev/_a_tcp_z_actual` 由速度差分除 `dt_flow` | 仅新**速度来源**更新差分，用其测量时间间隔；重复保持上次 a。该源新鲜度不能直接用 wrench fresh 代替，rail/arm 速度时间另有 metadata |
| compute `1512–1563` | recontact hold/settle 定时、first/recontact press latch | 每 control dt，用当前有效状态继续；stale 判据仍实时检查 |
| compute `1565` → `_update_instability_index:2110` | raw force HP、DC、RMS 能量和 Dimeas 递推 | 只新 force 源消费一次；重复拍保持观测值。原 HP/递推时基必须同时处理，见下一节 |
| compute `1567–1599`、`_effective_desired_z:2084` | Ke-based mass/damping 调度、desired-force ramp、lateral chase hold | 每 control dt；使用最新观测量。不得随 source gate 冻结 force reference |
| compute `1603` → `EnvironmentStiffnessEstimator.update:291` | ΔF/Δx 观测、contact/impact window、命令位移积分、Ke decay、BD slew 混在同一个方法 | 不能对整个 update 加 `if fresh`；下面给出内部最小拆点 |
| compute `1637` → `ForceSpaceVelocityDamper.update_fdot:115` | `_f_prev` 与 force derivative LPF | 仅新 force 源更新并传 dt_source；重复保持 f_dot。否则重复拍拉向零，新帧又把 10 ms 差除以 5 ms |
| compute `1648–1770`、`caps:1742` | precontact impact/peak、确认 hold 计时、force corridor、预测上界 | 不产生样本历史的当前条件可以每拍重算；hold 消耗每 control dt，caps 用最新 K/F/fdot/实际运动信息每拍计算 |
| compute `1818–1863` → `_admittance_z:2150` | TDPA preview、damping dynamics、proactive reference、精确 ZOH 导纳 | 每控制周期计算；command state 由既有 nominal transaction 延后 commit。不是新源才计算 |
| `_admittance_z:2232` → `FastRetractGuard.update:86` | 3 点 raw median + fast LPF，以及 stop/rearm/hold 确认计时 | 方法继续每拍调用；仅 `fast_retract_guard.py:129 _update_fast_force` 用 fresh/dt_source，`161–211` timers 用 dt_control，stale 检查每拍保留 |
| compute `1878` → bidirectional_flow | 观测模式下的 proxy/历史/保持速度积分和诊断功 | 保留当前 wall-time推进；不能把每拍 `source_fresh=False` 直接传给其 `feedback_fresh`，那是有效性门（见下文），会改变行为 |
| compute `1931–2070` | force-point 积分、slew/clamp、shield 模型历史、x_adm/x_d、last command | 每控制周期 proposal；仅命令事务字段等实际发布后提交。拒绝不“撤销”物理状态或已消费测量 |
| compute `2072` → `TDPA.commit:115` | air bias 滤波与 `F × v_cmd × dt` 诊断积分混合 | `tdpa.py:135–137` bias 只 fresh/dt_source；`140–143` command-port 工作/泄漏按 control dt，仍属于诊断，不能称 measured physical work |

`feedback_fresh` 特别容易误用：`bidirectional_flow.py:620–630` 把 False 当作 stale，使正常保持帧变成无效反馈。原 runner outer 的 `loop.py:3139` 注释已经明确：`feedback_fresh_tick` 是每拍边沿，不是有效性 gate；维持最后有效速度并由 `feedback_age_s` 决定失效。新接口宜另命名 `measurement_fresh`，不要复用这个参数改变旧含义。

## 采样滤波时基不能只加 if

1. observer LPF 构造在 `observer.py:86–89`，`fs=cfg.poll_hz`、45 Hz、一阶；from_yaml 从 UDP cycle 推导 fs。若去重后真实源间隔是 10 ms，却保留 200 Hz 系数，物理截止频率会变化。新模式要明确声明/验证 source sample period，再配置采样算子；保留记录到的实际间隔与超出边界状态，不静默假定恒定 5 ms。旧 mode 保持原系数和路径。
2. Dimeas 的 `_init_hp_filter:542` 按 `1/self.dt` 设计二阶 Butterworth，`_is_energy_alpha=min(1,self.dt/0.2)`；`2110–2149` 还使用固定 `var_damping_dc_alpha` 和 `var_damping_lambda`。只 fresh-gate 会改变 Hz 和遗忘时标。需要同一个明确的源时间基准，不能每控制拍任意重建滤波器。若把原离散 decay 转为源间隔，可对标准 alpha 使用 `1-(1-alpha)^(dt_source/dt_nominal)`；具体滤波实现由 high/ULTRA 决定。
3. Dimeas 最后递推不是标准 EWMA，而是 `I_next = I_omega * I_rms + lambda * I_old`。只把 lambda 改成 `lambda^r` 而不改输入项会改变恒定输入下的增益。若保持常值驱动的原递推，输入项需乘 `(1-lambda^r)/(1-lambda)`，lambda=1 单独取极限。此处是数值时基核对，不建议顺便重新调 Dimeas。
4. FastRetractGuard 的 LPF 已显式使用 `1-exp(-2π fc dt)`，源 fresh 调用传 dt_source 即可保持时间常数；3 点 median 应代表 3 个独立源帧，不是 3 个 control ticks。重复源不能两次入窗口。

## Ke 方法内部最小分离

文件 `admittance_common/adaptive_ke.py`。现 `update:291` **没有 dt 参数**，全靠构造时 `self.dt`；`_slew_ke:227`、`_slew_damping:232`、`_normal_displacement_m:250` 等 helper 也使用它。可以在原方法加入可选 freshness/control_dt/source_dt 参数并对几个 helper 增加可选 dt；缺省仍走旧路径，无需复制算法。

必须先分清配置：`peirastic/configs/force.yaml:336` 实际是 `displacement_source: admittance`。`_normal_displacement_m:259` 的 `_x_adm += v_force_z * self.dt` 是历史命令驱动的位移代理，**不是实测 TCP 位移**。它在本拍新 nominal 计算之前使用既有 `v_force_z`，须每 control dt 继续积分；拒绝本拍 proposal 时也不能声称该代理对应真实执行。pose 分支才使用源 pose 与 contact reference 的几何位移。

- `328 _f_err_env`：峰值捕获可用最新样本，release 衰减按 control dt；否则同源重复 tick 会被误认为新观测，也会令时标错误。
- `332–353`：contact rising 初始化只在实际状态转换发生一次，保持 episode 语义；其时刻可能由持续确认计时触发，而不恰好落在新源帧。
- `355–365`：detached Ke decay、BD slew 是时间算子，应每 control dt。
- `369 _contact_ticks`：当前既作为 settle window，又用于 `378` 的 20-tick impact window；若只 fresh 增加，原 20×5 ms=100 ms 会变成 200 ms。最小兼容可保留每 control tick 窗口推进，并仅 gate 学习；若改成 elapsed time，要维持原 nominal duration，不能悄悄把“ticks”解释成测量帧数。
- `373 _normal_displacement_m`：admittance 分支每 control dt 累积；pose 分支在新 pose 到来时更新测量位移。源 force 与 pose 配对/age 必须保留。
- `374–399` impact ΔF/Δx 与 `404–421` incremental learning：只在新配对 source measurement 时计算；不能用旧 F 与新的虚拟 x 产生一次假的 `df=0, dx!=0` 软化学习。
- `428–439`：idle decay 按时间推进，但只持有当前可靠 contact/force gates，不能因缺测声称“空闲且稳定”；当前正式 force.yaml 的 `ke_idle_decay_s=0`，detach decay 仍为 1 s。
- `441–444`：`_last_f_z/_last_x/_have_prev` 只在真正新配对样本更新；否则下一帧差分基线被重复 tick 擦除。
- `446–448` BD critical target 与 slew 按 control dt；Ke per-second slew 在新学习发生时应使用观测时间跨度，而不是固定 5 ms。

这里不是“所有观察状态冻结、下一帧补一次 10 ms compute”；后者会改变命令轨迹、参考推进和接触时序。source 分支负责更新证据，control 分支持续演化受最新证据约束的原动态。

## TorqueTilt 与实测姿态

`peirastic/realman8dof/force/torque_tilt.py:325 update`：

- `350–371` 的 torque/CoP 观测只由新 wrench 改变；desired torque error 可随目标每拍重算。`_was_contact` 的 rising reset 根据持续状态转换只触发一次。
- `373–386` 的 `measured_pose_euler/n_world` 是 pose 观测，使用自身 pose 身份，不与 force freshness 强绑定。纯当前姿态/align、slack/angle 机械条件（`388–405`）每控制拍仍检查。
- `_update_stall:289` 混合 CoP 改善判断与持续 stall timer；改善证据由新 CoP 更新，timer 按 control dt。不能把“尚无新样本”直接当成新一次改善或卡住的独立证据。
- `419–438` 的 Kikuuwe 速度、a-limit、`theta_tilt += omega*dt` 必须每 control dt proposal/commit。**theta_tilt 是命令积分角，不是实测角**；`measured_pose_euler` 单独存在。新 active 的实测 rocking angle 应由真实 pose/标定得到，不拿 theta_tilt 冒充。
- `prepare:205/commit_applied:231` 只延迟 `_w/omega/theta/stuck` 命令状态；measured/contact observations 保留。用独立 control_step_id 管 pending，并让重复有效 source 正常进入下一控制周期。

## 可复用验收测试

以下为已存在的测试入口，本只读任务没有重跑。新增异步样本测试应使用这些真正的原实现，不用另一份 toy force controller。

| 现有文件 / 用例 | 作用 |
|---|---|
| `peirastic/tests/test_contact_qp_nominal.py:test_100000_ticks_effective_icra_nominal_equivalence` | 参数缺省原模式逐拍等价；不能仅用“最终位置接近”代替 |
| 同文件 `test_abort_preserves_measurements_and_prevents_windup`、`test_air_bias_filter_consumed_once_on_abort_and_modified_commit` | abort 不回滚测量、bias 不二次消费，命令状态不虚假积分 |
| 同文件 `test_duplicate_stale_pending_and_reset_do_not_reconsume` | 现旧单身份契约；新双身份路径必须另外测试重复 source 允许、重复 control_step 拒绝，不能直接删掉已有防重语义 |
| 同文件 `test_rejected_contact_episode_retains_reseed_without_command_integration` | 一次性 contact event 在 proposal reject 后不丢失，也不下拍重放 |
| `rm75_control/tests/test_contact_detection.py` 中 50 ms trough、reacquire、episode rearm 用例 | 100/200 Hz 调度下保持秒级确认语义、edge 次数 |
| `test_admittance_contact.py:test_engagement_force_ramp` | force reference 仍按物理时间推进 |
| `test_adaptive_ke.py`、`test_ke_first_impact.py` | ΔF/Δx/impact window；增加100Hz新样本与200Hz command积分分离检查 |
| `test_fast_retract_guard.py:test_single_raw_force_noise_spike_cannot_trigger_stop` 等 | 一帧脉冲不能因 relay duplicate 填满 median window；hold/rearm时长不变 |
| `test_force_barrier.py`、`test_tdpa.py` | 导数源Δt与诊断功dt分离；bias样本计数与积分时长分别验收 |
| `peirastic/tests/test_torque_tilt.py`、nominal measured-pose test | 每拍 w/theta/accel limits，观测 pose 与命令角分离 |

必要异步断言由 ULTRA 设计：1 s 内 source 100 次、control 200 次、filter/history 100 次、正常 proposal/publication 200 次；连续 contact/ramp/nominal elapsed 不减半；拒绝 proposal 后同源仍可下一 control step 准备且不重复消费；源回退、代际变化、过龄各显式处理；同源数值不一致不能以“新 tick”洗成有效。当前源值为零变化时仍是新样本这一点也必须覆盖。

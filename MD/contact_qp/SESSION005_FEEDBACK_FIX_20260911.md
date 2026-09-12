# Session 005：图像、反馈时序与能量核验

此页记录 2026-09-11 对 `/media/camp/yameng/icra 2027/uncalibrated/005` 的核验，以及本次修改的模型边界。原始采集文件未修改；软件测试和记录数据回放不能代替机器人/组织闭环验证。

## 记录直接支持的结论

12 次尝试中 5 次完成、7 次失败。已完成的 L、C、S、反向 L、反向 C 的最低罐余额分别约为 0.383、0.456、0.378、0.666、0.729 J，均高于 0.05 J 留额。S 和反向 L 后半段黑边出现时，持续视觉许可可用，没有 episode 耗尽，也没有罐耗尽。

| 尝试 | 首个可确定的异常 | 后续录制端显示 |
|---|---|---|
| L DtP / 002 | 图像年龄 304.505 ms 超过 300 ms | TCP target timeout |
| C DtP / 001 | 图像年龄 302.670 ms 超过 300 ms | TCP target timeout |
| L PtD / 001 | 发送审核时力源年龄 15.143 ms | Task ended before scan started |
| C PtD / 001 | 发送审核时力源年龄 15.947 ms | Stale or missing force feedback |
| C PtD / 003 | 最终发送前力源年龄 15.278 ms；上一审核为 14.940 ms | Task ended before scan started |

另外两次仅有准备阶段记录，没有扫描 QP 日志，不能据此断言是相同根因。终端的地轨 FA24/Modbus 信息缺少可对齐的时间戳，不能用它给每一次失败定因。

完整逐次证据见 [故障核验](../../analysis_artifacts/contact_qp_session005_20260911/failures/README.md)。

## 反馈时序的修改

- `feature.dropout_policy: pause_visual`：接触中已收到过有效且注册一致的图像之后，图像缺失、超过 300 ms 或左右必需窗口无效，就从 QP 输入中移除图像，暂停视觉任务，继续使用有效的新鲜力反馈。真正有效的新图像到达后自动恢复。持续没有图像不会仅因图像年龄结束扫描，日志明确标记质量降级；启动缺少有效图像、真正的未来时间戳或注册不匹配仍拒绝。质量低于 0.8 不等于窗口无效，仍可用于修复。
- 获取图像快照后再读取判断时间，避免计算过程中异步到达的新图像被旧时间戳误判为“来自未来”。
- 仅在力源过期且臂、轨均确定尚未发送时，撤销轨预留、外环候选和内环候选，保留上次已成功发布的指令，并等待严格更新的力样本重新计算。没有发送旧候选，也不改写源时间戳、续租或更新看门狗心跳。
- 重算受上次成功指令原有的 50 ms 有效期和原看门狗限制，不使用“最多重试一次”的任意计数门。相同旧力样本期间按原循环节拍等待。持续断流、源时间倒退、真实过长采样间隔、未知/部分发送及其他硬约束失败仍按原路径停止。

控制器会在视觉暂停/恢复时打印 `visual paused` / `visual resumed`，完整转换事件写入 QP 日志。力源重算看 `publication_fresh_retry`；它表示本候选没有发送，不能计为一次修复执行。实现和离线验证范围见 [反馈实现说明](../../analysis_artifacts/contact_qp_session005_20260911/failures/IMPLEMENTATION.md)。

录制脚本 `/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py` 的完成分支也已修改：`done_seq` 只说明任务已经终止，还须确认无错误的 `DONE` 或正常 `icra_wait` 交接，才记 `path_done` 并检查末端目标。否则直接报告控制器终止消息，避免后续的 `TCP target timeout` 掩盖中止。补丁经独立审查后应用，原文件保存在 `analysis_artifacts/contact_qp_session005_20260911/failures/scan_robot.py.before_apply`。

## 黑边和视觉融合

S 在约 27.32 s 的左右 confidence 为 0.993 和 0.896，差 0.0975 小于旧 0.10 死区，因此没有差分修正请求。反向 L 在约 23.99 s 的左右值为 0.543 和 0.982，名义角速度约 −0.896°/s，QP 输出约 −1.322°/s，确实下发了与符号约定一致的转动。S 和反向 L 后半程实际工具 y 轴转角分别约 +5.10°、−9.18°，不能解释为“完全没有输出”。

图像到工具方向仍采用已注册的 `image_x_sign: -1`。左右置信度差可以反映不对称声学耦合，但并不等于已辨识的图像质量对角度的梯度。解剖阴影、浅层/深层不一致、时延以及转动引起的压力重分布，都可能使瞬时差分方向不能改善整幅图像。当前质量使用顶部 22% 深度和 4%–96% 横向窗口，确有深层或边缘黑区未进入质量指标的记录；本次不静默改变 ROI 和注册标定。

死区试值取 0.03。依据是完成 L/C 的稳定高 confidence 后半程共 713 个不同帧中，左右差绝对值的 P99 为 0.0125、最大值为 0.0166；0.03 保留余量并能覆盖上述 0.0975 漏检。此样本并非独立噪声标定，数值是待实机核验的工程设置。视觉增益、力门、速度及角度边界沿用。

多约束 QP 中，名义力矩转动已满足视觉所需的总转动时，不额外增加角速度是合理行为。尤其 `TorqueTilt.commit_applied` 将最终执行的总角速度写入下一周期导纳状态，不能把同一个视觉速度请求每周期直接叠加到该状态上。必须通过多步递推而非单帧解判断融合方案。

实际 `TorqueTilt.prepare/commit_applied` 与 QP 联合运行 600 个 5 ms 步长、每 50 ms 更新同样的有效图像，零力矩时原总速度融合约稳定在 0.800°/s，而直接叠加方案达到 12.605°/s 上限；正反力矩案例也出现相同累加问题。因此撤回叠加实验，保留共享总角速度融合。此测试验证内部状态递推，不模拟组织和图像对转角的响应。见 [多步递推结果](../../analysis_artifacts/contact_qp_session005_20260911/visual/recursion.json)。

原帧、confidence 图和定量结果见 [视觉核验](../../analysis_artifacts/contact_qp_session005_20260911/visual/REPORT.md)。

## 能量模型：供能来源必须明确

提供的八篇论文已逐一核对，页码和公式见 [文献审计](../../analysis_artifacts/contact_qp_session005_20260911/papers/literature_energy_audit.md)。Secchi/Benzi 的外部端口罐本来就会在正输入功时回补；持续逆阻力扫描可以持续消耗有限初始能量。Lee 的阻尼回收依赖相联的机器人/名义储能、结构切换项及期望轨迹供能端口，不能单独摘取一个正的阻尼项加到现有账户。

本次采用明确声明的名义任务功率额度 `task_power_source: nominal_command`。每周期在当前视觉 QP 之前冻结名义指令 `V_nom` 及 `W = -raw_control_tcp`，实际成功发布的最终模型速度为 `V_final`：

```text
A = max(0, -W · V_nom)              # 授权的名义任务功率
P = W · V_final                    # 完整六轴端口功率
S_used = min(A, max(0, -P))         # 实际使用的任务供能
dE/dt = P + S_used
```

额度以内的实际输出由任务供能承担，超过额度的净输出消耗罐，正向模型端口输入功回补；未用额度丢弃，所以停着不动不能凭空充能。只对成功发布且仍有效的模型区间结算；重试、过期、未发送的候选不赚取新供能。账目分别记录完整端口功、使用的任务供能、罐变化和容量截断，满足 `E-E0 = port_work + source_used - capacity_discard`。

这改变了保证范围：它是有任务供能的双端口**指令模型**，不是外部机械端口的无源性证明，也不是 Lee 控制器的实现。名义额度来自当前 QP 前的受限指令，而非另求一次最终 IK 的无视觉反事实轨迹；名义导纳状态也包含以前执行过的视觉动作。因此额度不能称为严格隔离的“纯基础/纯视觉能量”，也不保证所有视觉运动都会花罐能量。最终硬约束与实际接触力保护仍有各自作用。

初始 0.10 J、容量 0.15 J、留额 0.05 J 是这一新模型下的试值。005 已完成记录按此供能规则估计的最大额外净支出约 0.02634 J，最差 50 ms 额外支出约 0.000635 J，因此初始可用 0.05 J 提供试验余量。不能把这个数值当作人体安全能量标定。

## 重新加载和运行

退出旧录制的重试提示，再在旧 controller 终端 Ctrl+C。长运行进程各占一个终端。已有正常运行且注册配置一致的 ultrasound、confidence、gamepad 可以继续使用；controller 必须重启，新配置的能力检查会拒绝不支持本次策略的旧进程。

接触任务启动时应打印：

```text
[CONTACT_QP] visual=continuous; image=required; image_loss=pause_visual; tank=0.100/0.150 J; reserve=0.050 J; task_power=nominal_command; fusion=shared_total_velocity
```

```bash
# 离线配置检查，不连接设备
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh validate
```

```bash
# 终端 A：重启控制器
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh controller
```

```bash
# 终端 B：超声采集（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh ultrasound
```

```bash
# 终端 C：confidence 发布（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh confidence
```

```bash
# 终端 D：示教手柄（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh gamepad
```

```bash
# 终端 E：先预检三个新鲜、注册一致的 confidence 帧，再扫描录制
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh record
```

录制等价于下列命令，但包装脚本还会先做 confidence 预检：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record \
  --force-profile icra \
  --speed-m-s 0.005 \
  --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

## 验证范围

代码实现使用 HIGH，独立审查使用 ULTRA。完整 contact/controller/rail 回归及录制完成分支测试共 **513 项通过**，报告为 [final_tests.xml](../../analysis_artifacts/contact_qp_session005_20260911/final_tests.xml)。ULTRA 独立复现并发现了“无效窗口假恢复”问题；修正后相关适配器、连续视觉、发送记账、录制分支和审查测试共 **80 项复测通过**，报告为 [final_feedback_tests.xml](../../analysis_artifacts/contact_qp_session005_20260911/final_feedback_tests.xml)。两组有重叠，不应相加当作独立测试总数。

[ULTRA 最终审查报告](../../analysis_artifacts/contact_qp_session005_20260911/review/ULTRA_REVIEW.md)无剩余阻断发现，另有独立 77 项测试及受审文件哈希记录。审查结论仅覆盖所述软件行为与模型记账。

离线配置验证、包装脚本语法、修改范围内的空白检查通过。多步 TorqueTilt 反馈、六轴随机能量账目、未发送候选撤销、旧指令有效期、看门狗/外部停止打断、图像暂停恢复均有离线测试。没有启动设备或模拟组织声学反馈，不能据此保证实际黑边已消除；准备阶段两次失败和未对齐的地轨通信问题也没有被软件测试定因。

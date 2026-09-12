# Session 006：运行效果与残留故障核验

数据源：`/media/camp/yameng/icra 2027/uncalibrated/006`。本次只读核验保存的 session、attempt、QP 日志和 raw 数据，没有启动设备或更改生产配置。快照已有六条 RH 完成文件，另有第一次 L DtP 扫描失败，以及两次 L PtD 准备失败。

[独立 ULTRA 核验](../../analysis_artifacts/contact_qp_session006_20260911/review.md)复核了主要故障结论、部分原始能量区间、重试上抬代码路径及 S 的最终模型偏差。剩余采样间隔问题尚未在本次诊断中修改。

## 第一次 L 中途失败

扫描约 10.1 s 后，control 2568 的发送前审核发现力源年龄 20.461 ms。新的 `publication_fresh_retry` 路径确实工作：撤销未发送候选，保留 command 2567，原有效期到 monotonic 13978.330998 s，没有续期或记作成功修正。

随后控制器终止原因为 `contact_qp_exception:ValueError:source interval exceeds declared maximum`。这属于 `SourceClock.observe` 的新旧源采样间隔检查，仍是 15 ms 上限；与发送时样本年龄检查是两个不同条件。原始 TCP 数据已出现约 15.707 ms 的间隔后恢复，控制器却没有继续发布补偿力，录制监督之后报 `Stale or missing force feedback`。停止日志是在停止流程后写入，不能把日志写入时刻当作首次检测时刻。

精度限制：15.707 ms 是保存的相邻 TCP 源间隔，不是直接记录的失败 `observe()` 入参。结合下一轮检查时间和已到达的新 TCP，它与实际间隔拒绝一致；记录存在序号遗漏，因此不把它当作精确失败源入参。明确记录的首个故障类别是采样间隔超限。

因此，上次未发送候选的重算修正已经执行，但没有覆盖后续采样间隔超过上限这条路径。失败时罐余额约 0.088 J，扫描图像一直有效，不能归因为罐耗尽或 confidence 停用。第二次 L 也有一次 dispatch 过期，随后成功重算并完成，证明重算不是只在软件测试里出现。

## 两次反向 L 准备失败与上抬

两次都发生在 `[MOVEJ]` 后、`[RECORD]` 和 `[SEEK]` 前，没有启动本次 contact QP；只有 failure.json，没有该准备阶段的 raw/QP。能够确认的是监督程序没有拿到 100 ms 内可用的补偿力反馈。不能从相同提示断言这两次也是采样间隔、IK 或地轨通信造成。

`scan_session.py` 遇错调用 `Robot.stop()`，后者清除 `air_verified` 等自动运动状态并发停止请求。下一次用户 Enter 后，`prepare()` 因尚未重新确认离开接触而执行 `withdraw()` 上抬 30 mm，再做 MOVEJ 起点准备。所以终端中重试后的上抬属于原重试准备流程，不是能量罐补能动作，也不是视觉修复动作。仅凭日志不能确认用户所指的所有实际上移都发生在这一分支。

## 新机制实际启用的证据

全部七份接触 QP 日志均加载 `continuous`、死区 0.03、`pause_visual`、`nominal_command` 和 0.10/0.15/0.05 J。扫描期间图像状态全部为 `ok`，没有触发丢图暂停，所以 006 不能声称已经实机验证暂停/恢复。

28 个选取的原始图像按相同 frame ID 重算 confidence，结果与控制日志完全一致。实例：

| 轨迹与扫描时间 | 左/右 confidence | 名义工具 y 角速度 | QP/最终发布角速度 |
|---|---|---:|---:|
| S DtP，24.31 s | 0.994 / 0.730 | +0.414°/s | +0.736°/s |
| L PtD，17.97 s | 0.642 / 0.998 | −0.917°/s | −1.195°/s |

这说明视觉信息确实进入控制并改变最终发布的转动，符号与声明的图像/工具映射一致。六条完成扫描的实际相对工具 y 转角约为 +8.18°、+8.08°、+8.21°、−9.19°、−8.94°、−9.63°；这些总转角包含名义力矩等动作，不能全部记作视觉贡献。C/S PtD 的多数请求本来已经由名义角速度满足，QP 接近名义是共享总角速度融合的预期行为。

不能把所有 QP 候选当作实际发布：S DtP 有 50 个扫描样本的最终模型工具 y 角速度与候选差值超过 0.001°/s，最大约 0.396°/s，主要位于 20.84–21.61 s。最终方向仍一致，其具体内环映射原因本次未定。本文的最终模型统计使用 publication，而非以候选替代；异常样本另存 [final_candidate_outliers.json](../../analysis_artifacts/contact_qp_session006_20260911/visual/final_candidate_outliers.json)。

## 能量支出、回补与边界

按每个成功发布区间的冻结 W、最终 V 和名义额度独立重算，逐步端口功、实际任务供能、罐变化、容量裁剪和累计账目均与日志一致，未发现中途重置或凭空补额。

| 完成轨迹 | 最低余额 / J | 最终余额 / J |
|---|---:|---:|
| L DtP | 0.070585 | 0.071818 |
| C DtP | 0.084290 | 0.085959 |
| S DtP | 0.087221 | 0.087551 |
| L PtD | 0.099997 | 0.100763 |
| C PtD | 0.099998 | 0.127093 |
| S PtD | 0.100000 | 0.135635 |

每条都存在正端口模型功回补；最小可用余额约 0.0204 J。能量约束实际参与 QP 和发送审核，但本次没有耗尽或成为主动限制，故只能确认记账与接入工作，不能说本次已经实机检验低罐降权表现。这里仍是有任务供能的双端口指令模型，不是机械端口无源性实测。

[六条轨迹的罐余额曲线](../../analysis_artifacts/contact_qp_session006_20260911/energy/completed_balance.png)将扫描开始对齐到 0 s；负时间为寻触/建立接触阶段。

## 图像是否改善

按不同图像帧计数，后半程至少一侧 confidence 低于 0.8 的比例，005→006：反向 L 为 55.7%→30.4%，反向 C 为 41.8%→0%；这两条支持用户“好了一点”的观感。S DtP 为约 0.26%→6.23%，不能说所有轨迹全面改善。路径、接触条件、时长不同，这不是控制策略的配对因果实验。

006 S 右侧低 confidence 段伴随修正后恢复，但机器人同时沿路径平移，不能单凭先后关系认定恢复完全由角度修复造成。仍可见 S 右侧、反向 L 左侧暗带，部分深层阴影没有进入顶部 22% 的质量指标。现有数据确认“接通并输出”，不能确认“所有黑边已修复”。

详细可复算文件：

- [故障时间线](../../analysis_artifacts/contact_qp_session006_20260911/failures/summary.json)
- [视觉运行量](../../analysis_artifacts/contact_qp_session006_20260911/visual/runtime.json)
- [005/006 描述性比较](../../analysis_artifacts/contact_qp_session006_20260911/visual/comparison005.json)
- [能量逐区间审计](../../analysis_artifacts/contact_qp_session006_20260911/energy/energy_audit.json)
- [S 原图与 confidence](../../analysis_artifacts/contact_qp_session006_20260911/visual/RH_Per_S_DtP_001.png)
- [反向 L 原图与 confidence](../../analysis_artifacts/contact_qp_session006_20260911/visual/RH_Per_L_PtD_003.png)

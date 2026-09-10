# 两时钟 Python 核心独立审核 V3

2026-09-10，ULTRA，只读生产代码，独立补充检查写于 `/tmp/test_contact_qp_two_clock_ultra.py`，已交代码 owner 保存。未运行硬件、留出实验或新的声学仿真。

**PASS，限定已交接的源时钟 / 原 force law 观察与命令推进接缝。** active runner 的自动绑定、实际发布和失败事务尚未交付，本结论不代替整链验收。语义以 [MEASUREMENT_CLOCK_CONTRACT_V3.md](MEASUREMENT_CLOCK_CONTRACT_V3.md) 为准。

## 数值及独立检查

- 独立运行 `test_contact_qp_two_clock.py` 与原 nominal 短测：20 passed / 2.87 s，100k 用例未重复。
- 独立补充五个反例全部通过：wrench held 时新速度仍更新、wrench fresh 时 held 速度不产生差分；Ke 命令位移代理每控制拍积分而 `_last_f/_last_x` 仅新样本更新；一帧力脉冲不会被 relay 重复填入 median；TDPA bias 只更新一次但两段命令功都记账；Dimeas 非零驱动使用 `lambda^r` 与同步输入增益，未仅改变衰减系数。
- 复核 [two_clock_default_equivalence.py](two_clock_default_equivalence.py) 的旧类加载：旧 controller、adaptive Ke、fast guard、TDPA、legacy、tilt 来自修改前归档，旧 helper 明确绑定给旧 controller；新路径使用当前实现。真实 ICRA force.yaml + scan profile / .010 normal / .010 seek / tool / 4 N 参数一致。已有 [two_clock_default_equivalence.log](two_clock_default_equivalence.log) 为 100000 ticks、max absolute output error=0，周期 reset 与状态对照均执行。
- 后加的 velocity freshness 与 observer 恢复周期接口在默认分支保留原运算；定向短测覆盖，因此未重复已有长测。

## 源码核对结论

`control_step_id` 管理每控制拍 prepare/commit/abort，源 ID/time 单独管理首次消费。同源 held proposal 可继续控制时间；重复 control ID 不能再积分。abort 保留已消费观察及真实接触转换，命令状态仍按既有 nominal transaction 恢复；没有把测量源到达频率用作参考推进频率。

observer 的 held 分支返回已保存结果，不 append regression/pose history。显式 source period 在绑定时设计滤波器；同周期保持状态，换周期使用已有 raw compensated residual 的稳态种子，恢复 None 时恢复原 observer 周期与系数并保留 watermark/history。原 force controller 的 HP 仅允许 fresh controller 改周期；active 创建新 nominal 时须在首拍前配置，不可对已运行 controller 偷换。HP 首个新观察稳态初始化，持续 4 N 不生成假欠力或 HP 脉冲。

力导数按真实 source dt，只新样本更新。DC/RMS/Dimeas 时间系数按声明源时基适配；Dimeas 的非归一化输入增益正确。Ke 的命令积分、接触窗口秒级长度、detach decay 与 damping slew 继续按 control dt；力/位移学习样本仅 fresh。fast guard 的 median/LPF 与 hold/rearm timers 已分开。TDPA 是原命令端口诊断，不把它当完整实测六维能量 ledger。

原 controller 的 `normal_sign * f_ext_z` 与 4 N 定义保持；CoP 观察与 tilt 命令角积分已区分；pose/velocity 不因 wrench held 被整体冻结。接触确认可在 held control tick 达到门槛，事件一次性处理，不因等待 fresh 丢掉该次 episode 初始化。

## active 集成仍必须保证

1. `SourceClock` 是入口验证器，observer / force helper 的 freshness 参数并非独立传输身份认证。active 应由同一已验证源快照生成 ID/fresh/age，held 输入来自缓存；不能把同 ID 的冲突内容、relay 新 seq 或相同数值的新包混淆。当前时间是 UDP host receive，不是独立 FT capture。
2. 源的正常有限保持不使控制步、导纳积分或参考时间减半；超过声明 age/cadence 范围显式处理。固定周期滤波在声明 jitter 内是近似，不证明任意丢帧下物理频响严格不变。
3. 模式进入 / 退出对共享 observer 的周期绑定要在真正序列化边界完成；active 100 Hz → 旧模式 200 Hz 返回不能遗留 active 系数，不能回滚已消费观察或把稳定力清零。核心恢复接口已测，自动绑定链尚未测。
4. 参考提交仍是完整发布成功后的 `alpha * h_ref`，不是每个 fresh 源一次。实际失败、部分发送、迟到事实及旧命令负债不能靠 nominal abort 撤销；后续 active 整链单独审核。
5. 两种透明性分别报告：baseline/shadow 保持旧观察调用语义；active 非约束 QP 相对同一新观察语义的 shared nominal 透明。不能把正确去重后的 100 Hz 观察强称等价于旧 200 Hz 重复滤波。

交接生产指纹全部经 `sha256sum --check` 通过，完整列表见 [two_clock_sources.sha256](two_clock_sources.sha256)。关键源：controller `a984d1b220a14c72b8050ad2917d75ac66a56f68666ae2a28fa87a4be807b90e`；observer `afa74f6cca3b1037dd5cbacc31207a5589c06b34b3412c0591d2be2a9182b6c3`；transaction `88776e0135ee2cf11f9fff778333785fbb6f35727fbbc63a0f8f0356894b5e83`；SourceClock `492ba7ff1b9b9944ea94c1c195d2730ad1a774fec31cda97bba5486bb52e1a69`。

# T7 QP 求解失败的离线数值复现（2026-09-12）

结论：这 8 次 `solver_status_or_residual` 不是现有证据支持的 tank 耗尽；可重建的约束集均可行。已复现两次与采集日志**迭代数完全一致**的数值失败。失败状态不能忽略或直接执行其输出，复现时有 NaN。

## 输入来源与复现边界

- 数据根目录：`/media/camp/PEI_T7/icra 2027_contact/uncalibrated`。
- `peirastic/contact_qp/qp.py` SHA256 为 `c687333ea3db2635e9756f8d3b64802fa074a9a3e283bcd5767cdcce612c2f6d`，与采集日志记录一致。
- 本地数值库 ProxSuite 0.7.3，使用 rm75 环境。数据中未记录库版本，所以不能证明采集时库版本完全一致。
- `qp_prepare_rejected` 保存 nominal、path、previous、force、dt、angle、quality，但**没有失败拍原始六维 wrench 和完整 energy snapshot**；它们用上一个成功 `control_sample` 的 raw wrench / available_j 近似，task_power 按该近似 wrench 和失败 nominal 重算。
- 顺序重放成功拍时使用它的已记录 nominal / previous / wrench / quality，available_j 为该拍 control_sample 中的 budget facts；未记录成功拍 measured_angle，因此置零。其作用只在远离当前运动的机械绝对角度边界。顺序重放不是闭环机器人回放，也不是所有输入的 bit-exact 重建。
- tilt 的速度/加速度/角度限制由当前 `build_contact_nominal()` 读取，force max limits 使用日志配置；没有完整重建采集时所有适配器状态。`repair_execution_enabled` 使用默认 True，真实初始 seek 可能 False；20 拍最小复现片段已在稳定 SCAN 内。失败观察的中间窗口 quality 置 .8，当前 QP 只取左右窗口。
- 重放实际图像 effective/received time，但 now 使用记录时刻；个别临界 300 ms 帧可能与原执行瞬间有效性不同。因此未复现的 6 个最终失败不能推断已无问题。

## 八个失败快照

所有快照 HiGHS LP 均返回可行（最大 LP 约束违反 ≤ 3.4e-16）。新建默认 ProxQP workspace 时 7/8 成功；case 06 仍失败。

|case|文件（相对数据根目录）|实测失败 iter / 总耗时|新 workspace 默认求解|
|---|---|---:|---|
|00|yuhan_L/attempts/RH_Per_S_DtP/001|19303 / 37.45 ms|成功，16 iter|
|01|yuhan_L/attempts/RH_Per_S_PtD/001|18707 / 34.93 ms|成功，18 iter|
|02|yuhan_L/attempts/RH_Per_S_PtD/004|440 / 3.99 ms|成功，13 iter|
|03|yuhan_L/attempts/RH_Per_S_PtD/005|770 / 4.41 ms|成功，78 iter|
|04|yuhan_L/attempts/RH_Per_S_PtD/006|246 / 3.42 ms|成功，13 iter|
|05|zhongyao2/attempts/LH_Per_L_PtD/001|207 / 2.79 ms|成功，18 iter|
|06|zhongyao2/attempts/RH_Per_S_DtP/001|18045 / 33.34 ms|失败，17914 iter / NaN|
|07|zhongyao2/attempts/RH_Per_S_DtP/002|19513 / 35.48 ms|成功，22 iter|

## 已确定的数值机制

`ContactQp._solve_numeric` 按 `(shape, precondition)` 复用 workspace。能量行存在时强制 Ruiz preconditioning；初始化计算预条件器，后续 `update` 未传 `update_preconditioner=True`。本地 API 默认该参数为 False。`NO_INITIAL_GUESS` 不代表清空预条件器和全部数值状态。

- case 00 顺序重放 3089 个此前成功的拍，0 次失败；接着失败拍得到 **19303 iter / 200 outer iterations / NaN dual residual、duality gap、objective**，与原日志 iter 完全一致。只保留此前 **20 拍**即可复现。此前 1、2、5、10 拍不能触发；新建 workspace 后该相同失败矩阵 16 iter 解出。
- case 06 顺序重放 4477 个此前成功的拍，0 次失败；接着失败拍得到 **18045 iter**，与原日志完全一致。即使新 workspace，也可产生 NaN / MAX_ITER_REACHED。因此仅清缓存仍不充分。
- 重新计算预条件器修复 case 00，却在 case 03 产生 770 iter 拒绝；case 06 仍失败。不能把“每拍 update_preconditioner=True”当成已验证的全量修复。
- `max_iter=200` 管外层迭代；`results.info.iter` 是包含内层的累计数，另有 `iter_ext=200`。19303 并非简单把最大迭代次数配置错成 20000。`max_iter_in=100` 加重最坏执行时间。
- 保存的 H 特征值范围通常 1–10；case 03 的两个 loading auxiliary 为零权重（半正定），其他案例并没有极端 Hessian 条件数。case 06 H 严格正定。因此不能把全部失败归因于 H 奇异，也不能只说“物理约束冲突”。

## 能量行是否导致不可行

对每个快照，去掉最后的共享能量行，以该行左边作为 LP 最小化目标，保留所有其他约束。八个快照的该行最小值仍比能量下界高 **124–234（归一化单位）**，证明在这些重建快照上，该行不限制其他约束允许的任何解。它仍会改变数值求解过程，但不存在 tank 余额导致的数学可行域耗尽。详细值见 `energy_row_redundancy.json`。

这不等于可以关闭 tank，也不等于其他时刻没有能量限制。

## 同一问题的数值替代实验

保留所有 H/g/C/l/u、能量行、视觉松弛、机械约束、1e-9 solver/gap tolerance、1e-8 feasibility tolerance：

- 新建 workspace，不预条件：8/8 求解成功，13–27 iter，最大原约束残差 3.30e-10。
- 新建 workspace，仍预条件，`rho=1e-3`：8/8 成功，13–32 iter，最大原约束残差 6.47e-10。
- 单纯关闭 duality gap 检查、替换无限界、移除零行，均不能解决 case 06。
- 单次离线 solve 耗时约 0.03–0.18 ms；不是实时 deadline 保证，也未测整条机器人控制链。

**建议的有限修复方向，而非本次已上线的修复：**

1. 先补拒绝日志：完整失败 QP 输入/energy snapshot、ProxSuite 版本、pri/dua/gap、iter_ext、numerical workspace/preconditioner 状态；日志提供稳定可重放样本。
2. 评估有明确计算上限的数值重试：清 workspace 并采用已回归的替代 rho/预条件方式；不能沿用 NaN iterate。保持既有完整约束、有限性、残差、能量 admissibility 和发布 deadline 检查。第一次已经花掉 30–40 ms 时，不能为了重试而延长有效期或绕过过期拒绝。
3. 需要补成功路径批量回放、以前 uncalibrated/003 的预条件回归、低能量/不可行输入、截止期超时拒绝等回归，之后才适合改生产求解器。不能全局关闭预条件器：此前已有相反方向的能量行数值回归。

本子任务没有修改生产代码，没有读取或发送机器人指令。

## 工件和运行

`replay.py` 生成八个失败矩阵 `case_00.npz` … `case_07.npz` 及 `replay.json`；`history.py` 生成全部八个 attempt 的三种数值策略顺序回放；`minimal_history.py` 定位 case 00 的最短测试片段范围。

`regression.py` 检查 20 拍历史触发 19303 iter，并验证两种替代策略的 16 个完整约束求解均成功，已运行 PASS；`case_00_history20.npz` 包含该片段实际调用数值求解器的矩阵序列，可构建独立于外部扫描盘的测试。

```bash
source rm75_control/env.sh
python analysis_artifacts/contact_qp_t7_20260912/solver/regression.py
```

主要机器可读结果：`replay.json`、`numeric_variants.json`、`history.json`、`minimal_history.json`、`energy_row_redundancy.json`、`regression.json`。

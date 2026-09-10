# 清理候选 v3：单能量账本与旧 native 认证分支

2026-09-10，只读调用图。范围为 `peirastic/contact_qp/energy.py`、`execution.py` 和已经归档的 `MD/contact_qp/experimental_native_v9/`。本次没有清理生产代码，也没有运行硬件。**当前无运行调用不等于可以马上删除**：high 正在接入 active，已明确确认将复用 `EnergyLedger/PortInterval`，不会复用旧 `ExecutionCoordinator` 的 native certificate/全速度 box 入口。

## 短表

| 对象 | 当前真实调用证据 | 主要软件行为验收后的处理候选 |
|---|---|---|
| `energy.py:134 EnergyLedger`、`19 PortBounds`、`101 PortInterval` | 当前生产入口尚无 import；`execution.py:16`、`tests/test_contact_qp_energy.py:6`、`tests/test_contact_qp_execution.py:6` 引用。high 已确认 active 将直接复用该单账本与实际测量区间结算 | **保留**。不能因当前运行引用为零删除；不要另造第二个 tank 替代 |
| `EnergyLedger.reserve:162`、`settle:228`、`mark_no_send:213`、`record_unfunded_exposure:202`、`covers:189`、`invalidate:157`、`snapshot:296`，以及 `ExposureSegment:52` | 账本内部完整依赖；旧 coordinator 用 reserve/covers/stop debt。测量 settle 暂由 tests 驱动，active 接口仍在完成 | **保留单账本结算核心**：余额/容量、普通与停止预留、拒绝不退款、缺测不释放负债、区间不重结算、真实恢复才入账、越界失效。不能只保留 balance 加减而删异常结算逻辑 |
| `energy.py:72 branch_exposure`、`85 mixed_device_branches` | 主树仅 `execution.py:16,142,161` 和 `test_contact_qp_energy.py:104–115` 调用 | 可在 active 的最终预留策略验收后，随旧 coordinator **移入独立研究/历史目录**；是否仍需要 helper 要在 high 完成后重新查。`ExposureSegment` 本身不是旧 native 专属，不随 helper 删除 |
| `execution.py:21 DeviceState`、`31 ExecutionBounds`、`56 FinalCommand`、`89 _Facts`、`97 Publication`、`121 ExecutionCoordinator` 整条链 | 当前主树只有 `tests/test_contact_qp_execution.py:7` import；其真实发送适配器仅存归档 `.../hw/contact_publication.py:12,54–90` | **可归档候选，不建议立即删除**。旧 review:144 要 native certificate、精确 final command 与严格 `accepted_alpha`；新 active 保留原 IK、用发布事实和实际残差监测，不应被迫依赖该旧入口。归档时连同对应 tests 保存 |
| `ExecutionCoordinator._exposure:134` 的 64 个全速度 box corners、`review:144` 的严格 H/证书/≤10 ms 限制、`before_send:175`/`commit:222` 的 native 式证书身份绑定 | 只由上述旧 coordinator 内部调用；新 active 实现代理明确不使用这一 admission 策略 | 随旧 coordinator 归档。不要把这里“已无 runtime 调用”解释为所有 stop/partial-send/late-reply 状态事实都不需要；新 adapter 仍须独立通过这些行为验收 |
| 已归档 `experimental_native_v9/source/.../wbc_rt/{protocol.py,cartesian.py,client.py}`、C++ Cartesian 扩展、`hw/contact_publication.py`、对应 rail 改动与 tests | 归档 protocol.py:11–12 为 v9/96 rows；归档 loop.py:6934 传约束，7137/7147 调 publish_contact_candidate；当前主树 protocol.py:11 / C++ protocol.hpp:12 都是 v8，当前 runner/构建无归档目录导入 | **已经退出运行链**，无需再“从内环删除一次”。保留 source/patch/manifest/performance 作为失败与撤回证据；移植包不包含它。不要重新启用它来满足新外环 |
| `test_contact_qp_energy.py` 与 `test_contact_qp_execution.py` | 前者验证核心资金/区间语义；后者目前只验证旧纯 Python coordinator，不能证明当前 arm/rail runner 已接入 | 账本 tests 保留；旧 coordinator tests 可随研究库归档。新 adapter 的故障行为测试先验收，再调整主套件范围；不得用删测试制造 PASS |

旁界核对：`port_constraint.py:14 PortEnergyConstraint` 是**只读预算/约束对象**，不是另一个结算 tank；`qp.py:19` 正常 import，`QpInput.energy:121`、`_energy_rows:282`、`solve:413/502/540` 使用。它不在本次删除候选内。生产 `contact_qp.py:22/49` 正在使用 ContactQp 做 shadow；不能误删整个 contact_qp 包或 `geometry.accepted_alpha`（后者另有实验调用），也不能凭名称含 certificate 删除原内环的机械约束检查。

## 可复现 rg 证据

在 workspace 根目录执行以下只读命令。本次结论基于源文件引用而非进程动态采样；high 并行接入后应重跑同样查询。

```bash
rg -n 'from .*contact_qp\.(energy|execution)|import .*contact_qp\.(energy|execution)|from \.(energy|execution) import' peirastic rm75_control -g '*.py'
```

主树仅有四行 import：

```text
peirastic/tests/test_contact_qp_energy.py:6:from peirastic.contact_qp.energy import (
peirastic/tests/test_contact_qp_execution.py:6:from peirastic.contact_qp.energy import EnergyLedger, PortBounds
peirastic/tests/test_contact_qp_execution.py:7:from peirastic.contact_qp.execution import DeviceState, ExecutionBounds, ExecutionCoordinator, FinalCommand
peirastic/contact_qp/execution.py:16:from .energy import EnergyLedger, branch_exposure, mixed_device_branches
```

再按符号检查直接使用，避免只看 import：

```bash
rg -n 'EnergyLedger|PortBounds|PortInterval|ExposureSegment|branch_exposure|mixed_device_branches|ExecutionCoordinator|ExecutionBounds|FinalCommand|DeviceState' peirastic rm75_control -g '*.py' -g '*.cpp' -g '*.hpp'
rg -n 'EnergyLedger|ExecutionCoordinator|contact_qp\.(energy|execution)' MD/contact_qp/experimental_native_v9/source
rg -n 'contact_publication|publish_contact_candidate|cartesian_constraints' MD/contact_qp/experimental_native_v9/source/rm75_control/rm75_control/control/joint_admittance_8dof/loop.py
rg -n 'WBC_VERSION|kVersion' rm75_control/rm75_control/control/joint_admittance_8dof/wbc_rt/protocol.py rm75_control/native/wbc_rt/include/wbc_rt/protocol.hpp
```

第一条结果全部在 energy/execution 自身及上述两个 tests。第二条显示归档 contact_publication.py:12 引用主树 execution，归档 tests/test_contact_publication.py:7–8 引用主树 energy/execution。最后一条确认当前两端都是 protocol 8。

这揭示一个清理前提：**v9 archive 的 `source/` 目前不是完全独立的 Python 依赖闭包**。它的 contact_publication.py 仍从主树 import execution/energy，而 manifest 没有收录这两个 Python 模块。若把 execution/helper 移出主树，需另存确切版本和依赖闭包（至少 energy/execution 及其 geometry/types 依赖、旧 tests）并记录新的归档说明；不能修改旧 manifest 的历史含义，也不能把“历史测试现已无法 import”误称为旧实现已经验收。原 archive README 已明确最后确认修复未回归、96-active-row 实时验收失败。

## 单账本必须保留的边界

`EnergyLedger` 的唯一结算余额是 `balance_j`；`reserved_j` 是未完成区间责任，`available_j=balance-reserved-stopping_reserve`，不能再初始化一个同初值 tank 给另一条功率分量使用。`settle` 只接受同参考点/标定版本、有效对齐的过去时间区间；`_settled` 防重复，重叠预留在一次物理扣账后分别释放；不把 abort/no-send 当成恢复能量。`PortBounds.verified=False` 必须保持 monitoring/unverified 标签，不通过默认零误差宣称真机认证。

本表不预先裁定 high 的最终字段/预留实现。当前 active 尚未导入的事实、未来明确复用的依赖，以及已撤回的旧发送策略，三者分别记录，等主要软件行为验收后再按实到调用清理。

核对 SHA-256：

```text
33d2031abc4dc1591b87da66dd940f67aa133567a5929f0fd3cf6f115a44ffbe  peirastic/contact_qp/energy.py
531394546f066a554d2d0a37b0abb73d414116694a510a95e5643df6db75c301  peirastic/contact_qp/execution.py
e3ccfd9348fd0380b8894e5083de50481cee3a3bf1fa969683e5fd89308f2368  peirastic/contact_qp/port_constraint.py
14175c3c2a095644bc9dc69aec7e747089ec948b3116a763b4cc8a5fe78831d8  MD/contact_qp/experimental_native_v9/manifest.json
```

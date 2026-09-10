# V8 最终定向审核

最终状态：**v8r2 限定软件接口 PASS，本次发现的必要阻断项已关闭。** 本文件只审差分分配、有限修复尝试、测量力门控/监督、既有六维能量接口和原执行事务；不把合成声学实验当真实声窗模型，不重审原 IK，也未连接硬件。此 PASS 不代表闭环实验、真实声学改善或物理峰值保证通过。

## 已核对的实现

`repair_policy.py` 的任务是总 rocking 单侧服务：`s·d + vscale·ξD ≥ qD`，其中 `d` 由完整几何窗行取差后只保留原 normal/rocking 控制分量。在当前实机几何下 `d=−.031ω`，共同 vn 与 α 系数为零。独立 `wD ξD²` 保留，因此减速不能在代数上支付差分任务；原名义已充分提供正确方向 rocking 时不强制重复追加。

α 耦合以仿射式的平方进入目标。独立检查 F=3.6/4/4.2/4.5/4.6 N 的构造 Hessian 为半正定，差分行的 vn/α 系数为零。v8 移除了旧相对名义的两端零退化力符号行；孔径、原速度/slew、角度及外部机械行保留。

原 +4 N 控制轴标量、名义控制器、内环 IK/native 与发布事务没有被本版公式替换。4.5 N 是测量门控和新增 loading 软代价，并无未来力预测认证。active 在新鲜测量的 filtered/raw 任一原标量达到 6 N 时，在准备名义 candidate 之前进入既有异常停止链；这不是连续时间峰值≤6 N 保证。

能量辅助变量从 v8 的 8 个优化变量之后开始，不与差分松弛或两个 loading 辅助量重叠。原单共享 `PortEnergyConstraint` 仍作用于完整 `V=Hy`；最终六维能量复核仍存在。没有另建 tank，没有将 auxiliary 当可支出的能量，也未将名义/路径功从能量行中剔除。

当前真实 v8 profile 明确 `physical_w_checked=false`、`energy_constraint_enabled=false`、`energy=null`。所以本轮真实数据不提供 tank 已启用或物理无源认证的证据。静态零余额例只应抑制确实净耗能的 rocking，回能或零功运动可保留。

另已核对交付报告 [V8_ALLOCATION_AND_TEST.md](V8_ALLOCATION_AND_TEST.md) 的条件前缀推导：在同一 hold 区间内，非负误差/rate 上界使 `lower_work(V,τ)≥τ·Plb(V,h)`；由准入行得 `Ea+lower_work≥Ea(1−βτ/h)≥0`。成立前提包括完整六维参考一致、声明的误差/阻尼速度界、持有时长以及已扣除停止/未结算暴露的 Ea。它没有把当前 energy-off 实机称为已受 tank 保护，也不能推出力峰值或声窗收敛。

## 发现及关闭的必要问题

**R1：关闭差分任务后仍出现虚假松弛。** F=4.6 N、nominal `[.005,0,−.003,0,−.01,0]`、quality `(.4,.8,1)`、机械行 `vx≤.001`，独立复现 `visual_active=false`、实际差分 shortfall=0，却求得 `ξD=.046153846`，α=.2 而原偏好=.85。原因是 `(α−αpref+.75ξD)²` 可用正 ξD 降低目标；仅给 ξD 正代价不能保证它为零。要求在差分请求为零时停用该耦合，恢复原 α 目标；包含压力关闭、未知图像、无左右差异及修复额度耗尽。

**已关闭：** `terms()` 仅在请求严格为正时替换 α 平方项；独立复核对应反例，新增测试验证辅助差分松弛为零。

**R2：持续声影会持续请求转动。** 冻结第一版 701/702 两个声影单元均达到实际机械角度边界后中止，声学缺失未减少；不能作为闭环通过证据，也不能用放宽角度阈值或后改结果消除。此反例揭示速度任务没有累计尝试范围，且原一拍命令角约束不覆盖执行延迟和停止尾。

用户/主代理已批准 r2 的有限尝试策略：累计许可时间 2 s 或实测角行程 3° 耗尽后关闭差分与坏窗口正修复；保留真正好窗口 keep、原 4 N 名义和原始图像损失。连续三个不同的新鲜双侧 good 帧才重新开放。不能累计 `candidate−nominal` 的角速度来声称总尝试有界，因为 nominal 每拍吸收已发布动作，差值趋零时真实总转动仍可持续。

**策略缺口已关闭：** `RepairEpisode` 累计过去许可窗口的真实 elapsed 和既有实测 rocking 角通道的绝对变化；不随力/图像暂停、命令拒绝、坏帧、版本变化或角原点改变返还已花额度。角原点重设不计为实际转动。该状态在 QP 内于发布前记许可，故日志准确标为 permission，而非成功执行功或角度。耗尽后坏窗口的 soft 行确实移除，并非仅把请求置零后继续惩罚回撤；good 窗口 keep 保留。健康重置要求同一有效版本下递增 frame_seq 和有效图像时间，重复帧不能凑满三个。

该预算是有限策略尝试，不是新的 energy tank，也不是包含延迟/停止尾的实际最大角保证。未经足够误差和停止暴露界支持，不能把此修改升级成严格物理角度/力证书。

**R3：真实入口许可缺少当前接触条件。** 初次 r2 将“无 contact_gate 或 gate 曾 started”视为修复可执行。独立调用真实 adapter，filtered/raw force 均为零、原 `controller.contact_present=False`、新鲜左低图，仍进入 episode 并产生约 .010 rad/s rocking。gate 启动后再失触也受此条件影响。

**已关闭：** 在原名义 `sample()` 更新接触状态后，入口采用 `controller.contact_present AND (gate不存在 OR gate.started)`；两个真实 adapter 回归覆盖无 gate 的空气状态及已有 episode 后失触暂停，检查未启动/未继续修复且不返额度。没有替换原接触检测算法。

同时升级现有 Python 能力字段为 `contact_qp.differential_repair_v8_r2`；新客户端会在准备动作前拒绝仅广告第一版 v8 的旧服务，测试检查未发出命令。未改 native 协议。

## 独立验证记录

第一版独立运行 `test_contact_qp_differential_repair.py`、`test_contact_qp_active.py`、`test_contact_qp_active_history.py`、`test_contact_qp_runtime_config.py`：**40 passed / 1.80 s**。另直接检查凸性、差分系数，以及上述 R1 反例。没有重复长 100k 或启动新的声学仿真。

最终 r2 独立运行上述四组并增加 `test_contact_qp_repair_episode.py`：**48 passed / 1.46 s**。记录：[ultra_r2_review.xml](../../analysis_artifacts/contact_qp_v8/ultra_r2_review.xml)。其中包含错误方向名义修正、左右镜像、足够正确名义不追加、零 tank 抑制耗能 rocking、raw 6 N 停止、原双设备成功提交/部分发布冻结、额度/图像去重和原接触入口反例。

已有第一版五个真实状态的离线反事实和 144 个六维能量静态快照属于单状态软件证据；闭环 701/702 不利结果原样保留。r2 独立闭环结果由对应冻结实验报告另行评价，不并入本次软件 PASS。真实 q 未检测到的黑带仍不会自动生成修复请求。

最终核对的 SHA256：

| 文件 | SHA256 |
|---|---|
| contact_qp/repair_episode.py | `7d590d80633e95ef06d128dd092e354041e2f9a61a6635a83cb6ed18ebb2550d` |
| contact_qp/repair_policy.py | `b451ebd6ac89b97c2b1d1724bd0c70fe9722d4c30f9b0d3b246159ee6466378a` |
| contact_qp/qp.py | `f0c60085f487271d9e1d9883a5192e0d4987e13b46b5e8e082e36b62e9443fb4` |
| realman8dof/modes/contact_active.py | `75f6c0b0739b17979c22303f83cebed5e634c70494a11bf1b7b4f437f86bdfed` |
| core/capabilities.py | `297dbe69839c403874b814f4cfd1fe8228f7d54a823d247ab38cfa6e9be19869` |
| realman8dof/daemon.py | `ecf97293bb8b6f301d9c1bb43116910f41e3bb6c98bff59e916397863ad43360` |
| config/contact_qp/active_probe50_v8.yaml | `80aacdb94ff291f4f06daad584aa8ad5f5c33ad920c968a2c607681d49d75713` |
| tests/test_contact_qp_repair_episode.py | `a53b674c55bb3435a9047b80447b6da9184e6fdcdabf96002193577289ba3ba5` |

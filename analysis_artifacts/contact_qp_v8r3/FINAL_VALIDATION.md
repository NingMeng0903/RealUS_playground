# V8r3 / logical_final_model 验收记录

起点 HEAD：`669beac6668f659aaa05f92c5425ca0235f0523b`。实现由 GPT-6 medium 完成，核心公式及最后实现由 GPT-6 ultra 独立审核。生产变更保存在工作区，未提交或改动用户的 force calibration 数据；没有启动机器人、地轨或真人扫描。

## 结论

新配置 `peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml` 的能量准入为 true，结算端口为 `logical_final_model`，图像分配修订为 `v8r3_confidence_balance`。在所声明的冻结 W/V 命令模型内，余额、预留、实际软件发布事件和分段功积分已闭合。图像差分、原力矩名义偏好和 α 的分配保留三物理变量，能量是独立完整六维硬准入。

这不是实际机器人连续端口的无源证书。`physical_w_checked=false`；软件 STOP 的逻辑尾段释放不表示物理停止，也不证明旧/新臂轨混合执行的真实尾段耗能已经有界。停止额度仍只是普通命令不能花的留额。该边界已写入配置、日志和[独立审核](../../MD/contact_qp/ENERGY_PORT_REVIEW_20260911.md)。

## 软件回归与失败关闭

| 运行 | 结果 | 证据 |
|---|---|---|
| `pytest -q peirastic/tests -k 'not 100000'` | 637 passed，2 failed；三项已有十万周期测试未在此重跑 | [原始 XML](peirastic_final.xml) |
| 配置 fixture 更新后独立复查 | 14 passed，包括新增三个缺失契约字段拒绝测试 | [配置 XML](config_regression.xml) |
| 相同 native IK 离线测试在沙盒外独立进程复查 | 1 passed | [native XML](native_final.xml) |
| 原法向力控/TDPA/观测等八组回归 | 46 passed | [原力控 XML](force_regression.xml) |
| medium 定向实现测试 | 99 passed，与主回归重叠，不另累计 | [命令与结果](fusion_medium_validation.json) |
| ultra 最终独立定向复查 | 53 passed，与主回归重叠；另有独立 1000 个随机发布区间功积分检查 | [审核第8节](../../MD/contact_qp/ENERGY_PORT_REVIEW_20260911.md) |

两个全回归失败都有明确关闭证据：native 测试在沙盒里子进程启动即退出（code 0），同一测试在沙盒外使用随机独立共享内存通过，未连接任何硬件；另一项配置测试的旧 fixture 开启能量却没有新版本强制要求的端口、符号和持有时间字段。只更新这个测试 fixture，未放松运行时验证，还逐一测试缺字段应拒绝。

全回归开始和最后文件 hash 比较，仅 `peirastic/tests/test_contact_qp_runtime_config.py` 发生该 fixture 修改，生产源码和配置未变。[开始 hash](regression_source_start.json)、[结束 hash](regression_source_end.json)与 `tested_source.zip` 保存此次主回归输入；不把后来修改的测试版本伪称为当时输入。

## 十万步持续账本验证

[脚本](command_budget_long_sequence.py)与[结果](command_budget_long_sequence.json)，seed=20260911。真实使用新 `CommandBudget` 的 snapshot、reserve、started、commit、no-send、stop API，独立使用 50 位 Decimal 计算六维功积分、容量弃能、余额和责任；不使用同一个账本方法作为验证 oracle。

- 100,000 步、594,120 个检查点，模拟时间 322.426 s。
- 94,118 次提交、5,882 次拒发；每步不同但未提交的 wrench 快照不能改写旧区间。
- 35,000 个正输入功候选未提前产生信用；拒发继续扣旧区间，预留与结算分开。
- 最大余额/可用额误差 `9.298e-16 J`，最大预留误差 `3.469e-18 J`。
- 每步 drain 后事件队列为空，最长缓存 8 个事件；源文件起止 SHA 一致。
- 实测端口不参与余额；实际硬件认证始终为 false。

34.57 s 是该离线测试的总墙钟耗时，包含高精度 oracle；不能当作完整真机 200 Hz 周期的实时性证明。

## 数据与效果边界

[历史模型功](archived_model_work.json)覆盖 active/003 的五条扫描，模型净功前缀约 0.047–0.378 J，支持为完整六维扫描选择焦耳额度的数量级，不支持把该数值当真实安全能量阈值。

[固定状态矩阵](five_state_energy_policy_check.json)共 60 个组合，56 个可解，极小预算的 4 个不可行结果保留。PtD 状态7288的两侧 confidence 均过0.8，但差超过0.10：旧r2没有图像旋转，新r3请求总角速度约−0.241°/s，共同法向速度不变。只证明命令分配变化，未执行这些反事实命令。

冷初始化、从0骤然输入4 N且固定姿态的附加测试在第11拍出现名义速度/机械加速/孔径相对约束冲突：[原始失败状态](profile_static_fixture_failure.json)。同条件旧r2（关闭能量）也失败，r3关闭图像也失败，不能删掉该结果或归因于能量耗尽。其用途是揭示输入/初始化条件的限制，不能包装成真实扫描效果验收。

[MEDIUM 的复现及实际 YAML 联调](profile_fixture_compare.json)保留这些失败对照；随后用 400 次生产名义 `update` 在 4 N、5 ms、零路径输入下使名义状态稳定，不修改内部状态、增益或约束。此时完整新 YAML、合成左右质量 0.82/0.97 和同一个真实 outer 的 review/started/双设备成功回调连续 **200 次通过**，罐余额从 0.6 J 变为约 0.508 J，没有故障锁存。脚本为 [profile_fixture_compare.py](profile_fixture_compare.py)。这是软件事务联调，不是物理仿真；上机保留原寻触/稳定后扫描的流程。

本轮没有新的实体闭环声窗效果、力统计零退化或完整设备实时性验收。实际测试步骤、模型公式和移植入口见[V8R3_TANK_TEST.md](../../MD/contact_qp/V8R3_TANK_TEST.md)。

# Active 命令接口定向复核 v7

**限定软件接口范围 PASS。** 本轮闭合了外环命令历史与最终发送模型混用、加速度时间与未来保持时间混用两项缺陷。没有改原 IK/native、机械参数、4 N 目标、力优先带或 10 ms 证书时限，没有连接或重启硬件。本结论不表示全部实机故障消失，也不表示实际力/贴合效果已经通过验收。

## 实际记录能说明什么

只读 `active_probe50/001/attempts/RH_Per_L_DtP/{002,003,004}/contact_qp.jsonl`，与 `active001_failure_evidence_v7.json` 核对：

| Attempt | 双设备软件发送成功记录 | 首个 active 拒绝 | 最后参考时钟 |
| --- | ---: | --- | ---: |
| 002 | 1 | 第二拍 relative baseline/mechanical conflict | 0 s |
| 003 | 946 | 第 947 拍 final review rejected | 0.620811 s |
| 004 | 3357 | 第 3358 拍 motion subspace conflict | 11.412007 s |

因此 active 命令链确实运行过；软件发送成功不等于物理运动完成。三个 attempt 均未启用能量约束，不能把本次失败归因于能量行。002/004 的失败 QP 输入未写入旧日志，不能声称精确重建了失败拍的所有系数。

004 最后成功拍的 outer command Y 为 4.604097 mm/s，最终 rail-compensated payload model Y 为 9.673986 mm/s。原适配器把后者存成下一拍 QP 的 `previous_twist`，但下一拍仍要求命令属于当前三变量子空间。以同一条记录的命令固定下一拍路径构造反例，混用模型时 `motion_subspace_conflict`，改用上一外环命令时可行。这证明接口缺陷及其可复现机制，不假称恢复了未记录的真实第 3358 拍。

003 旧记录中准备源到拒绝约 10.707 ms，outer log 到拒绝约 8.402 ms，与时效拒绝相容；但旧日志缺 `created_time/valid_until/review_time` 和失败 payload，无法单凭它排除其他拒绝分支。该后段长尾没有被本补丁宣称解决。外层随后出现 stale feedback、rail encoder/连接错误，须按独立时间线处理，不能倒推为更早的 QP 数学拒绝原因，也不能据此断言地轨一直健康。

## 命令域全链闭合

最终实现将成功获准的 `qp_twist` 同时提交给 nominal transaction 的法向/rocking 命令积分与完整 `last_v_cmd`，并保存为下一拍 outer command-slew 历史。`nominal_transaction` 中这些字段是命令积分和命令历史，不是实测状态；含 rail 补偿的最终模型不应改写它们。下一拍用 `R_current.T @ R_previous` 分别旋转上一命令的线速度和角速度，比较同一 TCP 轴系中的命令。

只分开 QP previous、仍把 model 提交 nominal 并不充分：例如上一 outer 法向命令 .001 m/s，model 为 .002 m/s；模型重基线后欠力 nominal 可按 .8 m/s²、5 ms 到 .006 m/s，而 outer slew 允许上限为 .005 m/s，力优先两端行又不允许低于新 nominal，仍会无解。最终代码及反例测试已关闭这一回流路径。

最终 payload model 继续用于原有最终能量复核、保守参考进度计算与发送事实记录；实测反馈按原测量入口独立消费。没有将外环命令或发送模型改称实际速度。这里的加速度行是 **外环命令变化率约束**，不构成实际机械加速度认证，原内环机械保护保持独立。

## 时间、拒发与部分发布

`QpInput.acceleration_dt_s` 显式表示命令变化的实际控制间隔，旧调用默认采用 `dt_s`；active 使用与原 nominal 相同的 actual control dt。未来 command hold、角度预测/能量保持仍使用原 `dt_s`，参考授予保持原上限，证书仍为 10 ms。6 ms nominal 的 .8 m/s² 法向变化为 .0048 m/s，不能误放进 5 ms 的 .004 m/s 框；新增测试确认只修时间口径，不放宽加速度参数或力优先行。

两个设备的软件发送成功后才提交命令历史；拒发、异常和部分发布继续走原 abort/stop，不授予失败候选参考进度，不回滚已经消费的测量。最终复核仍拒绝过期、倒时、非有限值和能量预留失败。新增拒绝日志写出具体原因与创建/截止/复核时间；失败 QP 写出 nominal/path/previous、force、actual/hold dt 与诊断，不以静默跳过检查来消除停止。

## 独立验证与范围限制

rm75 控制环境独立运行 `test_contact_qp_active_history.py`、`test_contact_qp_active.py`、`test_contact_qp_solver.py -k 'not 100000'`：**43 passed, 1 deselected in 1.38 s**，没有重复已运行的长测试。覆盖记录值构造反例、6 ms/5 ms 冲突、四周期相同测量但不同 normal model 残差的 nominal/outer 输出等价、坐标旋转、abort 保持、原逐设备失败分支和证书过期拒绝。

Root 另报告全量短回归有一项 `test_cartesian_ptp_handoff` native 子进程启动失败，正在独立复查，原失败 XML 保留；本限定 PASS 不将该故障抹去或宣称全套通过。原 force 相关 46 项通过。003 时效长尾和实际通信状态也仍需独立执行证据。

Root 后续补充关闭证据：同一个 native 单测在本轮受限沙箱内再次失败，独立 worker 环境通过；root 获工具批准后在沙箱外以相同命令复查 **1 passed / 1.38 s**，未修改源代码或测试。结果见 `native_handoff_host_recheck_active_v7.xml`，原失败 XML 保留。此环境相关测试问题已单列关闭，不改变上述对实机时效/通信的限制。

| 关闭文件 | SHA256 |
| --- | --- |
| `peirastic/realman8dof/modes/contact_active.py` | `5dc1c17a2a1dcab425d2dae8c1397acb8a131698597ac7e079391ee49a32898c` |
| `peirastic/contact_qp/qp.py` | `c9888391c435a88887435bdbf8322506b1a322af78123eed905fcb6f59cff043` |
| `peirastic/tests/test_contact_qp_active_history.py` | `88749dee90ba7bfb6b164afe6d87cb9331de2a2c0bb10b1db7e7daa83021047b` |
| `MD/contact_qp/active001_failure_evidence_v7.json` | `28de8053c8d7d871901822ba964c5ef53115f85a4704242d8d825f4eced83324` |

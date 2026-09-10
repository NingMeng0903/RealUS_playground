# 共享参考 A / 可选响应 B 独立审查（2026-09-10）

结论：**A、B 的本次接口/实现修复通过独立审查；不表示控制效果或完整能量外环通过验收。** 本轮没有修改实现、没有使用留出集或硬件，也没有继续 native 审查。名义事务 tiny-delta 问题和外环能量行不在此通过范围。

## A：共享参考与收尾

已核对当前反馈目标为已接受时钟的 `p(s)`；前馈是 `p(min(s+dt,T))−p(s)` 除以真实保持时间 `dt`，没有候选终点位置前视。末尾 `h_ref=min(dt,T−s)` 与 `h_exec=dt` 分开，成功发布后才提交 `α*h_ref`。所有比较使用同一 Forearm 平滑参考；MatchedMotion 只读取 scan 阶段，机械 stop_tail 不作为任务停顿回放。停止尾段仍计入实际运动、力、耗时和覆盖。

独立审查发现并关闭两个新的反例：

1. 持续 `α=0.41674` 时，末尾参考时钟渐近收敛后停在 `T−1 ULP`。修复前独立 `shadow/0/full` 回放因此在 24.25 s 超时，实际路径已达 60 mm，QP 没有失败。现在仅在正接受量推进后、剩余参考为机器精度范围时结束该时钟尾差；`α=0` 不推进。
2. 平滑曲线在 `T−10⁻⁵`、`T−10⁻⁶`、`T−10⁻⁷ s` 时，其完整参考位姿已逐元素等于终点，前馈为零；理想跟踪下 QP 因零路径给 `α=0`，单靠时钟条件仍会死等。新增 `reference_geometry_roundoff` 完成证据仅限末端 ramp，要求完整参考位姿精确相等；它不改接受时钟，仍须满足原实际距离条件并确认停止。普通零 α 暂停不能因此完成。

最终源码上的独立 `shadow/0/full` 回放为 `complete_with_gaps`、valid=true，实际前进 0.06 m、耗时 8.44 s、无 QP 失败或机械越界。接受参考为 `3.3999616408220077 s`，最后完成理由是 `reference_geometry_roundoff`；如实保留该值，没有伪造 `T` 进度。运行前后实验源码 SHA-256 相同。

HIGH 的 [11 个参考回放](reference_interval_replays.json) 覆盖原 8 个 full 中止和 3 个匹配对照；均有效完成。另有 [A-only shadow 回放](reference_shadow_a_only.json)，从旧 design_v2 源码快照加载旧 ResponseConsistency，用于隔离 B 的影响。这些是修复证据，不能从“完成了”推断图像或力指标改善。

## B：有实际动作才折扣无响应

已核对各窗口使用图像 effective 时刻之间的连续实测局部位移。累计达到 10 μm 且观察至少 0.08 s 后，正向动作的近零或反向质量响应不再被直接跳过。MotionHistory 对插值过零段计算真实卸载三角面积；显著卸载重置未完成正向证据，不会只累积压入而忽略中途退让。未到采集时刻的动作、重复帧、无效图像、版本变化和测量间断不能形成新响应证据。

这仍是明确的可选策略，默认 full 关闭。gamma 只折扣修复请求，不限制或恢复 alpha。因此通过此项不能声称 shadow 的持续慢扫已解决，不能把低 gamma 当作组织不可修复或学习到图像 Jacobian 的证明。合成纹理和图像响应假设的限制仍见 [DIAGNOSIS_V2.md](DIAGNOSIS_V2.md)。

## 独立验证与源码

在规定 rm75 Python 环境中执行 `pytest peirastic/tests/test_contact_qp_experiment.py peirastic/tests/test_contact_qp_plant_history.py -q`：**35 passed，0.85 s**。此外完成上述最后 shadow 闭环复验。未重复跑不相关 native 测试。

| 文件 | SHA-256 |
|---|---|
| peirastic/apps/contact_qp_experiment.py | 5b147ccdcaa1fec65f8c7418d1df4ed2e7885efc964914d62de17d8d1a31d5f5 |
| peirastic/contact_qp/history.py | dbaa4bc6a49644bd50cfaaaf753ac91d2d0ab185e3bee14eaa1a8c64758d8672 |
| peirastic/tests/test_contact_qp_experiment.py | 9c74336e0398a8d72762bcb9c33c3486bc74eaaaa9b7581c5bb318b20e18157c |
| peirastic/tests/test_contact_qp_plant_history.py | f5a4bfb495043256c95c94ebdc46a8e4ee0dffb51de64c123e89aa50238d06dd |

后续另需关闭 QP 故障分类：完整 6D 机械集合可行但当拍 H/α 子空间为空，应为 task/motion_subspace_conflict。独立反例已发送 root：`nominal=path=[0,−0.0001,0,0,0,0]`、`previous=[0,.01,0,0,0,0]`、`dt=.005`、默认机械配置。真实 Y 许可 `[.005,.015]` 有见证 `[0,.005,0,0,0,0]`，H 的 Y 范围却为 `[−.0001,0]`；当前有/无 interval diagnostics 两种路径均误报 mechanical_infeasible。该分类问题不是继续扩展 IK 的理由。

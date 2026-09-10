# 共用有限参考区间独立审核 V3

2026-09-10，ULTRA，只读生产源；未运行硬件或新的声学实验。

**PASS：`FiniteIntervalReference` 对原 `ForearmReference` 这类纯姿态参考的有限区间与时间计算。** 此结论不替代 active 发布事务及接触 gate 集成验收。

独立运行 `peirastic/tests/test_contact_qp_reference.py`：23 passed / 0.16 s。交接及复核 SHA-256 一致：

- `peirastic/contact_qp/reference.py`：`dd9e9c6d0804adc1476daa07ac5bfe14fba91a91b4e961fbd0558771d2bdc1e0`
- `peirastic/tests/test_contact_qp_reference.py`：`eabb7eaf900fab914431334bc16ac080950abdbb95f385403d46d6ac317d58d9`

核对结果：global origin 与 bounded local time 分离；反馈保持已接受 `p(s)`，候选前馈为有限姿态差除完整 `h_exec`；末端 `h_ref=min(h_exec,T-s)`，成功发布后才由调用方使用 `alpha*h_ref` 提交时间。旋转使用 `log(R1 R0^-1)/h_exec` 的 base/world 角速度，与原 MotionReference 约定一致，没有把 Euler 角差当角速度。零速启动、短尾区间、alpha=0、几何末端浮点相等和正向推进后的尾差 snap 与先前已通过的仿真有限参考语义一致。该模块不自动提交、没有设备入口，也不把几何 exhausted 当作实测到达或已停止。

active 集成的三个明确前提：

1. **包装底层纯 ForearmReference，不直接包装有状态 ContactGatedReference。** 后者 `sample` 推进自身 `_elapsed_s/_last_input_t_s`，且 `set_origin` 忽略传入 `t_s`。若直接作为本模块的 source，candidate 的前后两次取样会提前推进 gate，破坏单个 accepted clock。保留原接触/持续 4 N 规则，但其观察与定时只按一个已接受时钟推进；有限区间从底层纯参考取姿态。
2. candidate 与成功提交必须使用同一个 `t_s/h_exec`，不能执行时换 dt 却提交旧区间；部分发布、拒绝或超时不调用成功提交。该纯函数本身不保存 proposal 身份，exactly-once 仍由 active transaction 保证。
3. `MotionReference.valid` 必须在进入命令链时检查；`exhaustion_reason` 只是参考证据。完成仍要求原实测距离门槛及停止确认，不能因几何相等或 alpha=0 虚构实际覆盖/参考进度。

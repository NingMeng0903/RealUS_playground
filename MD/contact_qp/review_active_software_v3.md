# Active V3 最终关键软件审核

结论：**PASS，限已交付 active 外环及原发送链适配的软件 gate**。本轮定位的异常停止、部分 nominal prepare 清理、实际控制时间被截短及真实到达后的 DONE 问题均已关闭。允许进入用户主动启动的真实特性实验准备；没有自动连接或运行硬件。

独立验证：active / daemon / capability / runtime-energy / two-clock 组合 **61 passed / 1.07 s**；最后到达标记修复后 active **13 passed / 0.94 s**。另执行两个独立检查：9 ms 实际控制间隔保留在原 nominal 事务中、参考 grant 仍限定 5 ms；正常 IK 带 H 子空间外残差时只记录任务偏差，不因该残差拒发。没有重复 100k 或声学效果实验。

关键路径结论：

- 原 IK 继续封装；外环使用原工具轴 +4 N 标量，不用面法向投影替换目标。真实几何、图像版本、源周期须显式配置，不以合成值补齐。
- 控制 step 与源样本 ID 分开。重复源不重复观测，原命令动态按实际控制时间继续；进入/退出 active 经停止边界绑定/恢复共享 observer 周期。
- 最终 rail payload 调整进入完整六维 command-model 速度，能量开启时在发布时刻复核。H 与策略残差只记录，`task_certificate=false`、`physical_certified=false`。
- arm 与启用的 rail 都返回成功后才提交 nominal 和参考时钟。arm 未知、rail 失败及其他部分发布冻结参考并保留事实/负债。active 准备、求解和发布异常先调用原设备停止，再清理候选；日志或清理异常不应先于该停止。
- 几何到达 predicate 不授予完成。只有原 runner 的 arrival gate 同时通过速度与 rail settled 条件才设置 `arrival_confirmed`，daemon 才发 DONE/done_seq；不再按参考 T 提前切换 idle。

真实接口限度仍明确：当前生产 tracker 尚不提供 `port_time_aligned` 和端口校准版本，实测能量结算因此保持 MONITOR unavailable，不退款、不凭命令回补。默认命令预算关闭；单独开启候选预算不代表真实能量回补链已接通。此软件 PASS 不是完整 200 Hz 实时认证、物理无源保证、声窗收敛或零裕量力指标实验通过。合成声学效果不作为用户采集真实数据的前置条件。

最终审核 SHA-256：

- `peirastic/realman8dof/modes/contact_active.py`：`97aa40a37c27c9221dc01b4cd582ca6f53d01331f3b5b71709a34c0bd1763e8f`
- `rm75_control/rm75_control/control/joint_admittance_8dof/loop.py`：`9f72ae2dc7f655279a0a186ed0d26026ed38d2031b30ba369c87b5dc4215e0ab`
- `peirastic/realman8dof/daemon.py`：`94dcbe0d5e2603c879bb3058a243b7cd0f935280caaf13435c594502f466f829`
- `peirastic/tests/test_contact_qp_active.py`：`51944eece6012e9f6ff062e259d59d4a4846fbf4d5d32dccdac85286100af6d6`

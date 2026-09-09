# native_timeout 性能优化（2026-09-09）

已优化碰撞查询和控制线程的 CPU 绑定生命周期，并重新编译 native。
优化后共 21,000 拍离线测量全部 solved，没有 timeout/coast、故障锁存或丢日志。
原生计算与调度等待分别测量；所有测试使用模拟反馈、独立 UUID SHM，没有机器人通信。
详细数据保存在 [JSON](native_timeout_performance_20260909.json)。

用户故障中 `native_cpu_ms=3.24`、`native_runqueue_ms=48.35`，支持调度等待是
该次约 52 ms 超时的主要来源。优化计算降低每拍 CPU 需求，后台任务避开控制核
进一步降低排队。当前主机使用普通 Linux 调度，测得的最大值不是最坏延迟保证。

实际修改：

1. native 碰撞 broadphase 在原球包围体之后增加 OBB 分离轴筛选。本地盒仅初始化一次，
   每拍缓存各几何体的变换，用 6 个面法向和 9 个叉积方向计算保守间隔。
   只有间隔超过原激活阈值加 1e-9 m 时才跳过精确查询；无效盒或未知缩放回退。
   基线姿态的精确查询从 7 对减至 3 对，精确距离、witness、CBF 行和阈值不变。
   原球下界选择的空候选 fallback 也保留。
2. `run_joint_admittance_phases()` 退出时恢复进入前的线程 CPU mask，包括异常退出。
   原先 daemon 在首次运行后一直留在控制 CPU，后续 CSV 子进程可能继承该单核 mask，
   而其只收窄 affinity 的后台初始化无法把它移走。现在每次运行结束后回到原后台 mask。
3. 新增 `profile_native_contention.py`，以有限个独立计算子进程比较竞争同一核和
   移到后台核的差别，自动清理测试进程，并输出延迟分位数、最大值和完整状态。

20 ms 首次等待、50 ms 请求年龄上限及过期解拒绝规则保持原值。

同配置、同 CSV 初始姿态、每种负载 3,000 拍的前后对比：

| 负载 | 碰撞 P50 前→后 ms | native 总计算 P50 前→后 ms | 往返 P99 前→后 ms | 往返最大前→后 ms |
|---|---:|---:|---:|---:|
| 控制 | 1.485 → 0.479 | 1.638 → 0.643 | 4.911 → 4.321 | 7.368 → 6.271 |
| 加 CSV | 1.477 → 0.476 | 1.633 → 0.636 | 4.647 → 3.868 | 7.313 → 6.824 |
| 加 CSV 和绘图 | 1.483 → 0.485 | 1.632 → 0.650 | 4.756 → 4.093 | 7.793 → 6.253 |

上述前后各 9,000 拍全部 solved。混合负载下碰撞 P50 降低约 67%，native 总计算
P50 降低约 60%。这是固定初始姿态、理想反馈基准，不能当作完整标定轨迹的耗时。

优化后的 8 个计算子进程竞争测试，每种布局 6,000 拍：

| 负载放置 | 往返 P50 ms | P99 ms | 最大 ms | 超时 / coast |
|---|---:|---:|---:|---:|
| 8 个工作进程与 native 共用 CPU 8 | 3.826 | 12.855 | 19.707 | 0 / 0 |
| 相同工作进程移至 CPU 10/12 | 0.686 | 3.803 | 6.439 | 0 / 0 |

这组对比仅使用优化后的 binary，衡量后台任务 CPU 放置的影响，未测旧 binary 的
同核竞争。测试线程/native 使用 CPU 6/8，避开生产配置 CPU 2/4 及其 SMT 同胞。

离线复现：

```bash
source rm75_control/env.sh
RM75_OBSERVER_AVOID_CPUS=2,4,6,8 python \
  rm75_control/apps/joint_admittance_8dof/profile_controller_runtime.py \
  --config peirastic/configs/controller.yaml --control-cpu 6 --native-cpu 8 \
  --seed-csv rm75_control/apps/logs/peirastic/run_20260907_211243.csv \
  --ticks 3000 --output /tmp/native-runtime.json

RM75_OBSERVER_AVOID_CPUS=2,4,6,8 python \
  rm75_control/apps/joint_admittance_8dof/profile_native_contention.py \
  --config peirastic/configs/controller.yaml --control-cpu 6 --native-cpu 8 \
  --background-cpus 10,12 --workers 8 \
  --seed-csv rm75_control/apps/logs/peirastic/run_20260907_211243.csv \
  --ticks 6000 --output /tmp/native-contention.json
```

下次重启 Window A 会加载更新后的 Python 和 native；Window B/C 及仍在运行的旧相机、
Genesis 等程序也需重新启动，才能加载此前已有的单线程数值池和后台 CPU 设置。
CPU 2/4 的 SMT 同胞 3/5 也需要留给控制任务。离线计算任务可以使用
`python -m peirastic.apps.background -- 原命令及参数` 启动。

当前验证覆盖 CPU scope/时序 16 项、发送链及竞争工具 33 项、native 碰撞及通知
15 项 Python 回归，全部通过。额外的 CTest 对比实际 C++ OBB 筛选与 Coal 精确
盒间距离，覆盖随机姿态、重叠、阈值附近、平行/近乎平行和无效输入，1/1 通过。
构建后可用 `ctest --test-dir rm75_control/native/wbc_rt/build --output-on-failure`
复现数值检查。
实际机械臂网络通信、完整标定运动和整机极端负载未包含在此次基准中。

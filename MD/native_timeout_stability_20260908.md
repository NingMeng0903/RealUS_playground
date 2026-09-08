# native_timeout 资源争用修复（2026-09-08）

本次处理的是 `native_timeout`，不是点云坐标或 QP1 不可行。用户日志中，同一个请求先等待 20 ms，最终年龄达到 52.7 ms，超过既有 50 ms 上限，进入锁存停止。`last_completed_solve_ms=2.28` 属于上一条完成的请求，不能证明当前请求按时完成。没有发现 SHM 损坏证据。

## 现场只读证据

- Orbbec 发布器 PID 4174567 有 66 个线程，其中 31 个 Python/BLAS 工作线程长时间 runnable。2.008 秒采样累计使用 54.84 秒 CPU，约 27.3 核；其线程累计排队约 7.99 秒。
- 采样时系统约 98% 忙，run queue 36–47；机器为 32 逻辑 CPU / 24 物理核心。相机环境的 NumPy/OpenBLAS 和 OpenCV 默认线程数均为 32。
- 控制进程 PID 63856 绑定 CPU 2，native PID 64916 绑定 CPU 3；本机 CPU 2/3 是同一物理核心的 SMT 线程。两者同时受到相机线程竞争。
- 该次采样没有运行 Genesis，因此证实了相机和控制器的资源竞争，尚不能把某一次停机唯一归因于 Genesis。当前 Genesis 的 Quadrants 编译线程还需要单独限制。
- 正在运行的控制器虽然 cwd 是 `ICRA_2027`，其 `PYTHONPATH` 指向本工作区。未修改任何已有进程的调度、未重启服务、未发送机器人命令。

## 已修改

1. 控制线程保留 CPU 2，native 改为 CPU 4，分开物理核心。独立相机、扫描、Genesis、力监视器和日志进程避开这两个核心的全部 SMT 线程，即本机的 2/3/4/5。只调整调用进程自身，既有受限 affinity 不会被扩大。
2. 相关入口在 NumPy/Genesis 导入之前限制 BLAS、OpenMP、OpenCV 为单线程，设置被动等待；`QD_NUM_THREADS=1` 限制本机 Genesis 的 Quadrants CPU 和编译线程。观察进程 nice 至少为 10，控制器优先级不变。
3. `peirastic` 和 `viewer` 包改为按需导入公开对象，避免 `python -m` 在入口设置限制前提前加载 NumPy。可选 `threadpoolctl` 和已加载 OpenCV 的运行时设置覆盖晚调用情况。
4. 相机每帧的 `N×3 @ 3` 深度校正改为等价逐元素运算，保持原有标定系数、深度截断及 float32 输出。
5. STEP 请求在发布序号、通知 native **之前**建立请求计时。通知连接被关闭/重置时返回现有故障结果，避免异常穿出服务循环。20 ms 等待、50 ms 请求年龄上限、过期解拒绝及手动 RESET 语义均保留。
6. 超时日志增加 `native_cpu_ms`、`native_runqueue_ms`、`client_cpu_ms`、native 状态/等待点及 `observed_reply_seq`。这些是只读诊断，不接受迟到的候选命令。排队时间高说明 CPU 调度竞争；CPU 时间高则应进一步检查当前计算。进程状态和序号是打印时采样，不能单独重建整个超时过程。

`[ESTOP] qpik_fault:stop:native_timeout` 表示故障锁存。daemon 的既有逻辑保留服务、等待 RESET；扫描任务会因错误结束。这次未加入故障后自动续跑。

## 验证

相关控制/通知/线程配置/API/扫描流程回归以及深度标定测试通过。控制器 `--dry-run --no-panel` 和相机 `--dry-run` 均通过，Genesis 镜像入口 `--help` 通过。本次未修改 native C++ 或协议，无需为这些修改重新编译 native。

单独的深度校正测试使用 8000 点、30 FPS、30 帧，限制在 CPU 6/7/8/9，避免占用实机控制核心。旧环境请求 32 个 BLAS 线程，但该测试的 CPU mask 使实际池大小为 4：

| 深度校正 | 平均占用 CPU 核数 | 单帧 P99 |
|---|---:|---:|
| 原矩阵运算，默认多线程 | 3.004 | 0.540 ms |
| 原矩阵运算，限制为 1 线程 | 0.0071 | 0.252 ms |
| 逐元素运算，限制为 1 线程 | 0.0059 | 0.193 ms |

每帧实际运算很短，旧进程却累计消耗大量 CPU，符合 BLAS 工作线程空转的表现。该结果只覆盖这段合成点云运算，不是整个相机发布器的 CPU 预测。数据见 [camera benchmark](native_timeout_camera_bench_20260908.json)。

native 性能验证采用 `peirastic/configs/controller.yaml`，从现场 CSV 读取初始 q8，以理想反馈运行。独立 UUID SHM；测试线程/native 分别临时绑定 CPU 6/8，避免与正在运行的控制器竞争其指定核心。原有服务未重启加载补丁。

| 条件 | 拍数 / solved | native P99 / 最大 | 往返 P99 / 最大 | 整拍工作最大 |
|---|---:|---:|---:|---:|
| 控制器 | 1800 / 1800 | 2.233 / 2.574 ms | 4.820 / 6.238 ms | 6.547 ms |
| 加 CSV | 1800 / 1800 | 2.339 / 2.497 ms | 4.759 / 7.767 ms | 8.641 ms |
| 加 CSV、六幅离屏绘图 | 1800 / 1800 | 2.321 / 2.839 ms | 4.783 / 7.823 ms | 8.429 ms |

合计 5400 拍，无 timeout/coast、故障锁存或丢日志。仍有少量整拍超过 5 ms。测试不含真实硬件通信、接触、完整 Genesis GPU 场景；不能替代重启后实机与 Genesis 的长时间联调。数据见 [runtime profile](native_timeout_runtime_profile_20260908.json)。

离线测试复现（只创建测试进程，不连接机器人）：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source rm75_control/env.sh
RM75_OBSERVER_AVOID_CPUS=2,4,6,8 python \
  rm75_control/apps/joint_admittance_8dof/profile_controller_runtime.py \
  --config peirastic/configs/controller.yaml --control-cpu 6 --native-cpu 8 \
  --seed-csv rm75_control/apps/logs/peirastic/run_20260907_211243.csv \
  --ticks 1800 --output /tmp/native_timeout_runtime_profile.json
```

## 加载修改

相机发布器、控制器 A、Genesis 窗口都必须退出后重新启动；仅重启控制器不能改变已经运行的相机线程池。使用各自原来的环境和启动参数。控制器启动应显示：

```text
[STATE] scheduling control_cpu=2 native_cpu=4
```

启动命令仍为 `python -m peirastic.apps.run_controller`，需要 CSV 时加 `--log-csv`。相机入口仍为 `python perception/apps/run_orbbec_cloud_publisher.py`（camera_calib 环境）。若使用自定义 YAML 更改控制 CPU，观察进程可通过 `RM75_OBSERVER_AVOID_CPUS` 指定同一组 CPU，系统会同时排除其 SMT 线程。

# RealUS 共用时钟（无 ROS 依赖）

`realus_clock.py` 是所有接入节点共用的标准库模块。控制器、手柄、超声、RealSense、Orbbec 和 HDF5 录制器均已接入。新脚本在启动时调用 `get_clock()`，即可创建或接入同一时钟，不必先启动独立时钟服务。

## 生命周期

第一个节点在 `/dev/shm/realus_clock_<uid>_<namespace>_v1.json` 创建不可变基准；同时启动时使用文件锁选出唯一创建者。其他节点读取同一 `clock_id`、起点和单调时钟偏移。Linux 同机单调时钟提供持续走时，不广播周期 tick。

初始化时会用锁；每笔 `now_ns()` / `header()` 只读取系统单调时钟并做整数换算，不访问文件、不申请跨进程锁、不等待其他节点。第一个节点关闭或崩溃，不会停止时钟，也不会使其他节点的时间归零。基准在同一次开机、同一用户和 namespace 中一直有效。所有节点重新启动后仍接入同一基准；需要新实验时使用新的 namespace，不能在节点运行中删除/修改基准文件。

这只解决共用时钟，不会自动启动机器人、相机或替其他节点提供其必需的数据。无需安装 ROS、roscore、DDS、rclpy 或 rospy。

## 与 ROS 相同的时间头

字段采用 [ROS 2 Header](https://raw.githubusercontent.com/ros2/common_interfaces/rolling/std_msgs/msg/Header.msg) 和 [Time](https://raw.githubusercontent.com/ros2/rcl_interfaces/rolling/builtin_interfaces/msg/Time.msg) 的定义：

```json
{
  "header": {
    "stamp": {"sec": 1788832800, "nanosec": 123456789},
    "frame_id": "tcp"
  },
  "timestamp_ns": 1788832800123456789,
  "clock_id": "每个共用时钟唯一的32字符ID",
  "clock_domain": "realus_shared"
}
```

`sec` 是 int32 秒，`nanosec` 是 uint32 且在 `[0, 1000000000)` 内；`frame_id` 是坐标系字符串。`clock_id` 等扩展字段位于 header 外，不改变 ROS Header 结构。这里只兼容字段及时间语义，传输仍是项目自己的 SHM/ZMQ，不是 ROS 序列化协议。

## 起点

默认 `epoch`：第一个节点取系统时间作为基准，之后以单调时钟推进，系统校时不会让共用时钟突然倒退。不是录制器启动时从 0 开始；录制器仅加入已存在的钟。

若需要第一个节点启动时从 0 开始，在第一个节点所在终端设置：

```bash
export REALUS_CLOCK_NAMESPACE=experiment_001
export REALUS_CLOCK_MODE=elapsed
```

其他节点使用相同 `REALUS_CLOCK_NAMESPACE=experiment_001`；未指定 MODE 时自动继承已建立的模式。也可以在每个终端都设置相同的两个变量。`REALUS_CLOCK_MODE=epoch` 则明确要求系统时间模式，若现有钟模式不同会报错，防止混用。

默认 namespace 是 `default`，未设置任何变量的节点自然共用一个钟。重启实验需要归零时，换一个新 namespace，比如 `experiment_002`，所有节点一起使用它。

## 新脚本用法

先 `source /media/camp/EXT_DRIVE/RealUS_playground/env.sh`，或把工作区根目录加入 `PYTHONPATH`：

```python
from realus_clock import get_clock
import time

clock = get_clock()  # 首个节点创建，后续节点接入
sample_monotonic_ns = time.monotonic_ns()  # 在真正取到数据时采样
meta = clock.metadata("tcp", monotonic_ns=sample_monotonic_ns)
# meta 已包含 header、timestamp_ns、clock_id，可与数据一起发送。
```

ZMQ 图像和点云保持原 topic、multipart 结构，在 JSON 元数据中增加统一 header。旧 `source_time_ns` / `wall_time_ns` 仍保留来源语义，跨模态对齐请使用新 `header.stamp` 或 `timestamp_ns`，并检查 `clock_id` 相同。

SHM 不修改现有控制 payload 的 ABI，而是发布配对的 `<payload_name>_header_v1`。用 `HeaderReader` 读取标准 header，并核对数据序号：

```python
from realus_clock import HeaderReader
reader = HeaderReader("rm75_f_ext")  # 自动加入共用时钟
stamp = reader.read(source_seq=force_sequence, monotonic_ns=round(force_t_s * 1e9))
if stamp is not None:
    print(stamp["header"], stamp["clock_id"])
reader.close()  # 只关闭读端，不删除发布端数据或时钟
```

配对头包括 `rm75_state_header_v1`（位姿 world）、`rm75_f_ext_header_v1`（力 tcp）、`peirastic_ctl_v2_header_v1`（controller）、`peirastic_twist_header_v1`（tcp）及 `peirastic_motion_header_v1`（tcp）。定制名称/前缀同样适用。头与数据分开发送，因此读取时必须配对序号和来源单调时间；对不上就重读后续快照，不能把一个样本的头用于另一个样本。

## HDF5

录制器加入共用时钟，校验发布端 `clock_id` 和配对头，保存发布端时间。没有新 header 的旧发布端不会被静默当成已同步。

每个流都有 `timestamp_ns`，以及与 ROS Header 对应的层级：

```text
/tcp/header/stamp/sec        int32[N]
/tcp/header/stamp/nanosec    uint32[N]
/tcp/header/frame_id        string[N]
```

`/force`、`/ultrasound`、`/controller`、`/clock` 结构相同。根属性 `clock_id`、`clock_mode`、`shared_clock_json` 记录共用基准；原始单调/系统时间与设备时间继续保留用于诊断。自动接触裁剪也保留相同的时间头。

## 精度与边界

同一个钟不等于硬件同时曝光或同时采样，各笔仍记录各自的采样时刻。TCP/力使用机器人 UDP 在主机收到时的单调时间；超声使用成功读完 HDMI 帧、裁剪前的时间；RealSense 有可信 global/system 时间时映射设备采样时间，否则使用主机收到帧的时间并标明来源；Orbbec 使用原主机帧时间。设备原始时间一并保留，HDMI 内部延迟等仍需标定。

仅支持同一主机、同一次开机。不同机器需要另外的时钟同步机制，不能把远端 monotonic 直接接进来。不同 `clock_id` 会被录制器拒绝。内部 native QPIK 的 deadline/dt 仍使用原本的单调计时，不改控制算法。

诊断：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh clock
bash run.sh clock --watch
```

已运行的旧节点需要在合适时机重新启动一次，才能发布新增时间头。之后启动顺序不会影响共用时钟的建立和接入。

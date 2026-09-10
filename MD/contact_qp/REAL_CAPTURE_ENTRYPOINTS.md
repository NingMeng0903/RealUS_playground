# 真实特性采集入口核对

最新交付补充：baseline/shadow/active 的 Python 接入及服务能力预检现已完成软件验收，见 [ACTIVE_RUNTIME_MEDIUM_V1.md](ACTIVE_RUNTIME_MEDIUM_V1.md) 和 [review_active_software_v3.md](review_active_software_v3.md)。本文以下核对过程保留历史状态；当前真实配置仍未标定，未执行 `--execute` 或重启现有服务。默认 confidence 已升至明确版本 v2，旧统计属于 v1；见 [CONFIDENCE_FIX_V3.md](CONFIDENCE_FIX_V3.md)。

**当前接口状态：真实 H5 原 JPEG/最终 metadata → 生产 worker → inproc 特征接收 → 异步 JSONL 回归已 PASS。** 证据见 [`real_frame_interface_smoke.json`](real_frame_interface_smoke.json)，已在 metadata 与记录器修复后重新运行。初版漏查 publisher 最终覆盖 clock_domain 的失败保留于 [`real_frame_interface_pre_fix.json`](real_frame_interface_pre_fix.json)，第 5 节保留原因。此结论是接口回放通过，不是实时端点已接通或真实闭环效果通过；旧服务能力预检另行验收。

核对日期：2026-09-10。本文核对用户现有 `ICRA_YM/script` 的实际入口、环境和记录关联；只运行源码读取、导入和 `--help`，没有连接数据端点、启动机器人或操作已有控制器进程。优先由用户主动启动被动记录，以及原控制器执行下的 baseline/shadow 记录。新 active 外环仍需实测几何、图像轴向与物理 wrench 映射声明，本文不提供未验收的 active 运行命令。

## 1. 原入口的行为

`scan_robot.py` 是 `Robot` API 实现文件，本身没有扫描 CLI。入口是 `scan_session.py`，`run.sh record` 转发到该文件；`run.sh record-passive` 才转发到 `record.py`。

| 用户命令，均在 `/media/camp/EXT_DRIVE/ICRA_YM/script` 执行 | 实际入口与行为 | 是否可能发机器人动作 |
|---|---|---|
| `bash run.sh controller` | rm75 环境，`python -m peirastic.apps.run_controller`；启动原 SDK/native/外环服务 | 是，控制服务启动；不要与已有进程重复启动 |
| `bash run.sh gamepad` | genesis 环境，原 gamepad；默认 `--hold --force-profile icra --no-capture-y --no-vessel-b` | 是，用户输入可控制机器人 |
| `bash run.sh ultrasound --headless` | camera_calib 环境，`us_framegrab/scripts/run_ui.py`，读取相机并发布已保存裁剪 | 不发机器人动作；打开摄像设备与发布端点 |
| `bash run.sh check` | genesis 环境，`record.py --check` | 否；只读 SHM/ZMQ，不写 H5 |
| `bash run.sh record-passive --raw-only` | genesis 环境，`record.py` | 否；只读现有反馈/图像并写 H5 |
| `bash run.sh record --keep-raw` | genesis 环境，`scan_session.py`；两点示教、逐条 Enter 后自动扫描 | **是**；包含定位、寻触、扫描、退出 |
| `bash run.sh record --preview points.json` | 读取 `{"distal":[6 个数],"proximal":[6 个数]}` 规划预览 | 否；该分支在构造 Robot 之前返回 |
| `bash run.sh calibrate` | `delay_cal.py`；既有体模正弦按压标定 | **是**；不是被动时间检查 |
| `bash run.sh calibrate --from-h5 FILE` | 既有离线延迟拟合 | 否 |

来源：[`run.sh`](/media/camp/EXT_DRIVE/ICRA_YM/script/run.sh:13)、[`scan_session.py`](/media/camp/EXT_DRIVE/ICRA_YM/script/scan_session.py:271)、[`scan_robot.py`](/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py:308)。现有 [`README.md`](/media/camp/EXT_DRIVE/ICRA_YM/script/README.md:3) 明确保留 2026-09-08 事故后的“暂停人体自动采集”状态及未完成的实机验证。本文件是入口说明，不把只读核对或新增记录功能当作恢复人体自动采集的依据；没有执行 README 中的进程重启步骤。

## 2. 用户可复制的原记录命令

已有数据发布进程可复用，下面命令由用户在自己选择的采集时间执行。首先只读检查：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh check
```

纯被动记录完整原始数据，60 秒示例可自行改为其他时长；`--duration 0` 持续到 Ctrl-C：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record-passive --raw-only --duration 60 \
  --output-dir '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/raw'
```

此命令本身不产生激励或扫描；只有用户通过既有控制入口执行的动作才会被记录。`--raw-only` 保留建立接触、释放、异常和失败前后的数据，不做自动 4 N 接触裁剪。`--output-file` 可指定独占创建的精确文件路径，但需避免复用已存在文件。

原自动扫描与录制的准确命令如下，仅供用户了解和自行启动其原流程，行为包括机器人运动：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record --force-profile icra --speed-m-s 0.02 --keep-raw \
  --data-root '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/scans'
```

`--keep-raw` 让成功扫描也保留 `raw.h5`；否则原 `finish_trial` 会在接触裁剪成功后删除完整 raw。每个 attempt 的 recorder 子进程本来就使用 `record.py --raw-only --output-file <attempt>/raw.h5`，记录进程无需自己改为新 law。新目录示例用于把特性采集与既有数据区分，本文没有创建它们。

不要把 `--force-profile baseline` 与新记录的 `mode='baseline'` 混淆：前者选择原力矩参数配置（原手柄、扫描应一致），后者是在当前原控制行为上附加记录。若要复现此前 icra 参数，保留 `--force-profile icra`。这两个参数都不等价于 `law='contact_qp'`。

## 3. 原参数与新配置传入点

原扫描的关键参数由 [`scan_session.parse_args`](/media/camp/EXT_DRIVE/ICRA_YM/script/scan_session.py:238) 定义：

| 参数 | 实际默认值 / 限制 |
|---|---|
| `--data-root` | `/media/camp/yameng/icra 2027/uncalibrated`；与用户外置盘目录不同，命令示例已显式指定 |
| `--recorder-python` | `/media/camp/EXT_DRIVE/envs/genesis/bin/python`，可用 `ICRA_RECORD_PYTHON` 覆盖 |
| `--force-profile` | `icra`；仅允许 `icra, baseline` |
| `--speed-m-s` | 0.02；必须在 `(0, 0.02]` |
| `--curve-side` | +1；允许 ±1 |
| `--state-shm` / `--shm-prefix` | `rm75_state` / 空字符串；自定义 namespace 时各组件须一致 |
| `--us-endpoint` | `tcp://127.0.0.1:17359` |

原被动记录 [`record.parse_args`](/media/camp/EXT_DRIVE/ICRA_YM/script/record.py:385) 使用 `--output-dir`，默认 `/media/camp/EXT_DRIVE/ICRA_YM/data`；没有 `--data-root`。其他默认：`--poll-hz 500`、启动超时 20 s、流中断超时 5 s、源数据最大年龄 0.5 s、mode 对齐最大年龄 0.1 s、`--topic amongus_camera_frame_v1`、`--camera-name us_img`、`--force-axis 2 --force-sign 1`。后两个参数只定义 H5 的 `contact_force_n`，不会改变机器人 force law，也不能用来证明 physical wrench 的正负号。

真正的旧控制请求位于 [`Robot.scan`](/media/camp/EXT_DRIVE/ICRA_YM/script/scan_robot.py:325)：

```python
self.arm.hfpc(
    reference="icra_path", path_spec=spec, law="tff", force=4.0,
    force_axes=SCAN_FORCE_AXES, wait_for_contact=True,
    scan_contact_n=ENTRY_N, scan_contact_s=ENTRY_S,
    label="icra_scan", block=0,
)
```

其中 `SCAN_FORCE_AXES=[0,0,1,0,1,0]`，`ENTRY_N=4.0`、`ENTRY_S=0.1`；寻触前另显式设置原工具坐标力控和 10 mm/s 法向速度上限。`law="tff"` 是硬编码，**当前原扫描 CLI 没有 `--law` 或 `--contact-qp` 参数**；不能给 `run.sh record` 拼接尚不存在的参数。

实现代理 high 已确认新增唯一显式 API 参数为 `PeirasticArm.hfpc(contact_qp: dict | str | None)`，透传 `HfpcPayload`；`None` 保留原请求。baseline/shadow 使用 `law='tff'`，配置分别为 `{'mode':'baseline',...}` 或 `{'mode':'shadow',...}`。字符串表示 YAML 配置路径。应在上述同一个 `hfpc` 调用处传入，不另建原 force law、不改变 inner/native。**参数名是 `contact_qp`，不是 `extra`。**

high 随后确认实际新入口为 `peirastic/apps/contact_qp_run.py`（不是初拟的 contact_qp_record.py）：`--help/--validate-only` 不连接设备，显式 `--execute` 才通过旧 API 发请求。准确验证命令见下一段；不提供 active 执行示例。原 `ICRA_YM` 脚本在本次只读核对中没有改动。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=/media/camp/EXT_DRIVE/RealUS_playground:/media/camp/EXT_DRIVE/RealUS_playground/rm75_control:/media/camp/EXT_DRIVE/RealUS_playground/src \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_run \
  --config peirastic/config/contact_qp/real_study.yaml --validate-only
```

原配置默认 mode=baseline、feature_endpoint=null。该 CLI 的执行形态是在同一命令改用 `--execute --path-spec '<已有 taught icra_path_v1 spec 文件>'`，只向已运行 controller 提交 HFPC。这里的 path-spec 是 `peirastic.scan_path.ForearmReference` 接受的完整 spec，**不是** preview 的 `{distal,proximal}` 两点 JSON。新 CLI 不做原 scan_session 的示教、30 mm 离面定位、逐条 supervisor 与 H5 recorder 管理；不得把它当成 `run.sh record` 的整套安全/记录工作流替代。`--execute` 未由本任务执行。

此后已实际在 genesis 环境运行新 CLI 的 `--help` 与上述 `--validate-only`，均退出 0；返回 mode=baseline、configuration_valid=true、command_authority=unchanged_baseline、hardware_connected=false，同时明确列出 geometry/feature 的未标定项。这个 PASS 仅说明离线配置与入口可用，不代表实时帧已通过接入（第 5 节保留真实失败及回归状态）。

## 4. 图像 worker 环境与用户启动命令

2026-09-10 的无网络导入检查，两环境均为 Python 3.10.20：

| 模块 | rm75 | genesis |
|---|---|---|
| numpy | 2.2.6 | 2.2.6 |
| scipy | 1.15.3 | 1.15.3 |
| PyYAML | 6.0.3 | 6.0.3 |
| pyzmq | 初检缺失；root 随后仅补装 27.1.0，inproc PAIR 检查通过 | 27.1.0 |
| OpenCV / `cv2` | 缺失 | 4.13.0 |
| h5py | 缺失 | 3.16.0 |

pyzmq 补装证据见 [`runtime_dependency.json`](runtime_dependency.json)，未改变其他依赖或已有进程。rm75 现在可接收少量 confidence JSON；完整 JPEG 解码、confidence 处理与 H5 记录仍使用 genesis，避免让控制环境承担图像处理。`contact_qp_features` 的 `cv2/zmq` 是延迟导入，两个环境都能运行 `--help`，所以单凭 help 成功不能说明完整 worker 可运行。

用户可在独立终端启动被动特征处理，它只订阅 17359 图像并在 17361 发布特征，不接触机器人控制接口：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=/media/camp/EXT_DRIVE/RealUS_playground:/media/camp/EXT_DRIVE/RealUS_playground/rm75_control:/media/camp/EXT_DRIVE/RealUS_playground/src \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_features \
  --input tcp://127.0.0.1:17359 --output tcp://127.0.0.1:17361 \
  --calibration-version unverified
```

该示例保留 worker 当前延迟默认值 0.15196365053143765 s、`--image-x-sign 1`，并显式标记 `unverified`，适合观察记录，**不是本装置图像轴向或延迟已经验证的声明**。现有 worker 默认不写 H5/JSONL；特征会由新 controller sidecar 随 control sample 记录。重复启动同一个 17361 PUB 会端口冲突，应复用用户已启动的 worker。

如果只检查环境，把命令中的运行参数替换为 `--help` 即可。此次已实际验证：genesis `scan_session.py --help`、`record.py --help`，以及 rm75/genesis 两环境的 `python -m peirastic.apps.contact_qp_features --help` 均退出 0；同时独立导入实际依赖确认上表。没有实际运行 worker 循环或读取首帧。

## 5. 真实 H5 揭示最终 clock_domain 合并不兼容；时延只应用一次

**纠正初版漏查：不能只看 camera_frame_meta 的初始字典。** [`UsImagePublisher.send`](/media/camp/EXT_DRIVE/RealUS_playground/us_framegrab/src/us_framegrab/zmq_pub.py:191) 最后执行 `meta.update(self.clock.metadata(...))`；[`SharedClock.metadata`](/media/camp/EXT_DRIVE/RealUS_playground/realus_clock.py:131) 返回 `clock_domain='realus_shared'` 并覆盖初始 `host_monotonic`。源身份、frame_index、原始 capture_monotonic_ns 仍在，但修复前 `process_parts` 仅接受顶层 host_monotonic，**会拒绝真实现用 publisher 的帧**。原 `test_meta_template` 只检查初始模板，`test_send_uses_frame_read_wall_time_for_source` 未断言最终 clock_domain，不能作为整链兼容证明。

已用 `uncalibrated/chenwei/LH_Per_C_DtP.h5` 首帧的原始 JPEG 与原 metadata 离线重现 `ValueError: monotonic capture timestamp missing; cannot align control history`。帧 13931，publisher_instance_id=`79dd0712b7f0427fab1ead7ad2447e6a`，capture_monotonic_ns=`1048465350605926`，timestamp_ns=`1788954131312189033`，source_time_ns=`1788954131312189669`，clock_domain=realus_shared，timestamp_source=host_frame_read_complete。metadata SHA-256=`551a774fd7edf86edea4a05aa6dcda935e5ea981001cd91046bd6af9cc52997a`；JPEG SHA-256=`bdca19a34b17fb204c195c3d32b0433db97486a9e6169709051a75d806a571f9`。这次回放只读取 H5，没有订阅实时端点。

最小兼容解析已交 high：仍严格要求有效源身份/instance、非负整数 frame_index、正整数 capture_monotonic_ns；对 `realus_shared` 顶层只兼容已核对的 `realus.us_framegrab` schema_version=1、timestamp_source=host_frame_read_complete 契约，并核对 clock_id、header stamp 与 timestamp_ns 一致。读取 **capture_monotonic_ns 自身的 host-monotonic 语义**，不能把 shared/epoch `source_time_ns` 当作 monotonic。若将来显式提供 `capture_clock_domain`，可以独立验证该字段，不需要改顶层 shared clock 语义。帧实际没有 clock_version/boot_id；boot/host/clock schema 在 H5 root.shared_clock_json 中，不能凭空要求历史帧有这些键。实时同 clock/boot 验证可由接入端使用本机 clock description；离线回放不应拿今天 clock_id 拒绝历史帧。

修复后仍需用最终 publisher metadata 或上述真实 JPEG fixture 验证，而不仅是手造 host_monotonic 模板。用户首次主动采集应以日志有效 source/frame、接收计数与错误状态确认实际输入；不能仅凭进程存在判断成功。

原 publisher 的 `capture_monotonic_ns` 始终保留原始 read-completion 时间；配置 `time_offset_ns` 只加在 shared-clock `header.stamp/timestamp_ns/source_time_ns`。当前在线 feature worker 使用 **raw capture monotonic** 再减一次 `effective_delay_s`，因此即便 publisher 的 shared timestamp 已加 offset，也不能再把 offset 当成 raw capture 已对齐。离线已经对齐的 H5 若选择其已对齐时间轴作为输入，则使用 `already_aligned=True`，不再减 152 ms；现有实时 worker 没有 `--already-aligned` 参数。

## 6. JSONL sidecar 与原 H5 的可核查关联

原 H5 只有 `tcp/force/ultrasound/clock` 四个数据组；controller 状态用于录制就绪与 mode 判据，但没有作为独立组写入。因而候选、最终发送和发布状态需要新 sidecar，不能从旧 H5 凭空恢复。

| 数据 | 原 H5 确实保存的关联字段 | sidecar 要保存 / 核对的对应内容 |
|---|---|---|
| 会话 / 时钟 | root `clock_id`、`boot_id`、host、`shared_clock_json`、clock anchor/offset | study 开始时同 host/boot/clock namespace 信息、配置与版本指纹；sidecar 自己的 session_id 不能替代 publisher session |
| 原 TCP / 原始传感器 wrench | `/tcp/timestamp_mono_ns`、`source_time_s`、session_id、seq、pose_m_rad、q_deg、rail_m、wrench_raw_sensor | 源时间与来源身份；measured twist 来源/时间区间/有效标记。不能用处理时刻替代源时刻 |
| 控制补偿 wrench | `/force/timestamp_mono_ns`、source_time_s、session_id、seq、6D wrench_tcp | controller 这一拍实际使用的 filtered/unfiltered TCP wrench、各自源时间；以源 `t_s` 去重，不以 relay seq 去重 |
| 图像 | `/ultrasound/frame_index`、`timestamp_mono_ns`、timestamp_ns、完整 metadata_json | worker `source_id=原 source_id + ':' + publisher_instance_id` 与 `frame_seq=frame_index`，effective/received time，registration/window/calibration version |
| 单拍决策与发布 | 旧 H5 未保存 | 同 sidecar session 下的 `control_id` 关联 `control_sample → publication → shadow_candidate`；保留 sample/proposal 时刻和实际发布时刻，不按日志写入顺序当作执行顺序 |

`frame_id`（`world/tcp/us_prob`）是**坐标系名称**，不是图像帧唯一编号。图像精确 join 使用 `(publisher_instance_id, frame_index)`，source_id 也应一致；publisher 重启后 frame_index 会重置。H5 保留完整 metadata JSON，因此不需要修改上游 recorder 就能取到这组键。图像只做 latest-only 特征处理，少于 H5 原始帧数是可解释现象，不能用行号逐行拼接。

时间关联优先使用同一 boot 的 monotonic。`/force` 用原测量 `t_s` 转换的 timestamp，relay 重发不能造成重复消费。`/tcp` 的 robot UDP receive 时间和 rail 最新缓存可能不同；分析 full measured twist 必须保留 rail 时间/有效性，不能强行把它们视为完全同步。对 feature effective time 查找历史 W/V 时，保留所用样本的间隔与 stale/invalid 状态，不做无界最近邻匹配。控制力仍使用本拍最新有效 wrench，不能替换成图像有效时刻的历史力。

真实机械功率必须区分 physical W 和 control force。当前补偿配置声明 `environment_on_tool`，`f_ext_raw` 是未滤波的补偿外力候选，`f_ext` 是其因果滤波结果，runner 再转到活动 TCP；observer 返回的 `_signed` 尚未减 payload/gravity。不能把 `_signed` 当外部 W，不能把 `f_ext` 六维整体取负猜符号。sidecar 需记录完整 `[Fx,Fy,Fz,Mx,My,Mz]` 与同一点、同轴系的 `[Vx,Vy,Vz,Wx,Wy,Wz]`；预测 `J qdot`、执行模型估计与 measured twist 分开。若 physical wrench 的实际 TCP/标定绑定仍未核对，保存补偿候选和 `monitor_unavailable/unverified` 状态，不补零、也不声称能量认证通过。完整映射依据见 [`RUNTIME_ADAPTER_MAP_V2.md`](RUNTIME_ADAPTER_MAP_V2.md)。

sidecar 实现仍在并行完成；上表中的“要保存”是可移交的最少字段契约，不代表每项已经写入当前尚未定稿的生产模块。写盘丢行、异步候选丢弃、feature 不可用都应保留计数/原因，不能在后处理里删掉失败或用零填空。

## 7. 核对指纹

以下为本次读取的原入口和图像接口 SHA-256；实现代理并行修改的新 runtime 文件未列为冻结版本。

```text
2f8ba62795afc7e5712ee6dbb0fddac588d774e58d31cea7288ad5a1145d86ac  ICRA_YM/script/run.sh
4955829f28937418548702b2fbd07a7a2a706921c2bf4d7844941207786b9489  ICRA_YM/script/scan_robot.py
1c673b6a35d492118e78079b5d5a1a85f30155e7bb649ba81ad1cc3418853fe4  ICRA_YM/script/scan_session.py
aebe72d83ad7e4a2f4daa47b4a6731b3096ac2b2e62ffd594dd86c027b0e6264  ICRA_YM/script/record.py
17cf011b0570ce9971ebe04f4ce48f45743d7b49ffffca928ad8e13878ff6faf  ICRA_YM/script/scan_io.py
1f498e2e8524c200ac8cf1f17d5c7a9d33f0ba1c2593685e33975d1fcb326c9e  peirastic/apps/contact_qp_features.py
00dcf34682f5ec793ae0c2cc176b68cd1cd6056f141b9bddc1f1f286ca2ca36d  us_framegrab/src/us_framegrab/zmq_pub.py
```

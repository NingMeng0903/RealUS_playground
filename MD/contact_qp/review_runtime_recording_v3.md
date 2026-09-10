# Baseline / shadow 记录入口独立审核 V3

2026-09-10，ULTRA，只读生产代码；独立反例只在 `/tmp` 执行。范围是 `ContactStudyOuter`、后台记录/候选、HFPC payload / Proxy / daemon 绑定、离线 CLI 及真实 feature metadata 接口。active 测量接缝和主动发布尚未交付，不在本轮 verdict 内。未调用硬件、未订阅真实端口、未使用留出数据。

**最终 verdict：PASS，限定 baseline / shadow 记录与离线 CLI 子集。** 下列七类实现缺陷及修复边界已闭合；不以合成声学效果作为验收门槛。最终 [runtime_recording_sources.sha256](runtime_recording_sources.sha256) 全部文件校验一致。active、两时钟共享观察器改动和主动发布尚须另行审核。

**新增部署边界：本地 scoped PASS 不等于现有旧 daemon 已加载功能。** 旧进程能力预检尚待实现/验收，实际 `--execute` 部署链暂不能据此宣称 ready；没有连接或重启当前服务。

## 已确认成立的边界

- `ContactStudyOuter.sample` 调原 baseline sample 一次并返回原对象；shadow 独立线程只持有快照、QP 与 sink，没有 arm/rail/inner 发布入口。原 nominal 不在 shadow 重算。
- Proxy 显式参数保留 `0` / `False`。wrapped phase 的 on_tick 闭包捕获该 owner；生产 runner 的 on_tick 先于 daemon on_step 热切换。已有真实 installer AST 执行测试覆盖旧 owner 关闭与新 owner 绑定。
- CLI 缺省及 `--validate-only` 在导入/构造 PeirasticArm 之前返回。缺少真实几何仍可记录旧控制；不会用 synthetic geometry 兜底。active 目前明确拒启。
- 日志把 compensated physical-wrench candidate 与尚未确认的物理端口分开，`physical_port_status=monitor_unavailable`，没有把预测速度或控制符号冒充已验证六维实测功。
- root 的 [real_frame_interface_smoke.json](real_frame_interface_smoke.json) 使用原 H5 第 13931 帧的原 JPEG/最终 metadata，通过生产 `process_parts`、inproc `ConfidenceSubscriber` 和异步 JSONL。该旧帧以当前时间检查仍 stale；fixture received 时间明确仅供接口测试。这是接口证据，不是 live 接通或声学效果证据。旧失败保留在 `real_frame_interface_pre_fix.json`。

## 已闭合的问题与原反例

1. **结束原因在关闭后写入，实际丢失。** runner 在正常 break 后先 `phase.on_exit()`；包装 on_exit 已调用 sink.close。随后 finally 才 `record_stop(stop_reason)`，sink.emit 因 closed 返回 False。独立真实异步文件反例：模拟 `PARTIAL_ARM:rail_commit_failed` 分支的 on_exit → finally 顺序，文件只有 `study_start/control_sample/recording_close`，没有 stop 或失败原因。应在关闭前记录实际结束原因，保留发送失败/部分发送/未知事实，不能把 prepare 或求解成功当发布成功。验收须覆盖生产回调顺序及各失败出口，不能只直接调用 `record_stop` 测一次。

2. **热切换退出会阻塞控制线程。** daemon `_install_velocity` 在控制 on_step 内同步调用旧 phase.on_exit；其 close 会 join 后台线程。即使队列为空，writer 等待 `get(timeout=.05)`，独立三次空队列 close 实测为 44.199 / 44.287 / 44.273 ms，控制周期为 5 ms；I/O 堵塞还可能等到 2 s。控制回调应只请求关闭/排空，不等待线程或文件 I/O；最终执行已停止后再完成有界等待。验收应让后台写入故意阻塞，确认热切换仍立即返回、旧日志最终能够收尾、无线程回写命令。

3. **shadow 姿态使用了错误的角度量。** 初版把 `current_pose[4]` 的绝对 base Euler pitch 传入 QP 的 rocking-angle limit。它不是以本次安装/参考姿态定义的相对 tool rocking angle，受初始摆放和 Euler 分解影响。声明较小 rocking 限制时，静止但有初始 pitch 的正常姿态可能假报 mechanical infeasible；即使未触发，也使候选不对应真实限制。应使用已声明参考和旋转约定得到的实际相对量；信息不足则候选 unavailable，不能补绝对 Euler 或控制积分角。

4. **shadow 没有使用并验证声明的 feature policy。** 初版忽略 config.feature 的 c_min、窗口、registration/calibration 与 policy revision，直接使用 QpConfig 缺省值和任何时间有效 observation。独立检查声明 c_min=.8 时实际 solver c_min=.5；相反左右映射或其他 calibration 的 observation 也没有 adapter 层匹配。应从同一声明构建 QP 的阈值/窗口并验证实际帧版本；缺项或不匹配明确 unavailable。窗口仍 LEFT+RIGHT 必需、CENTER diagnostic，不通过降 loss/删除窗口掩盖缺项。

5. **畸形 metadata 可终止 feature worker。** 初版合法 JSON `[]` 或 shared envelope `header=null` / `stamp=[]` 会抛未捕获 AttributeError，而 main 只捕获若干解析错误。host_monotonic 分支的 `int(frame_index)` 还允许 bool/浮点身份被截断。应在解码图像之前验证顶层/嵌套 mapping 及严格非负整数 frame identity，坏帧统一作为输入拒绝、下一合法帧继续。high 已在复核期间修改此处；独立临时反例的两个畸形输入现均 ValueError，待最终测试/指纹确认。

6. **CLI execute 未复用原扫描 profile。** 原 `ICRA_YM/script/scan_robot.py:316–318` 设置 tool frame、normal speed=.010 m/s、seek=.010 m/s 与 `force_profile(icra)`；初版新 CLI 直接 hfpc，回落 force.yaml 的 seek=.030 m/s 及 tilt mass=.065/damping=.28，而原 ICRA tilt 是 .051/.22。这不满足已冻结的原扫描基线。应复用同一公开配置及 payload 语义，并用 fake arm 检查最终请求；无需也不得为该测试连接硬件。唯一原轴 4 N 标量的边界另见 [MEASUREMENT_CLOCK_CONTRACT_V3.md](MEASUREMENT_CLOCK_CONTRACT_V3.md)。

7. **原 inner proposal 与实际 rail publication 缺少区分。** 初版 JSONL 的 publication 仅取 `step.q_send/step.qdot`；生产发送前已有 `_wall_clock_rail_target` 修订及 coast 分支，实际 `rail_pub_m/qdot0_pub` 未进入该 sidecar。旧 CSV logger 虽有其中部分字段，但可关闭，不能依赖它使 JSONL 自动完整。应只转发现有发送局部事实，并明确原 inner proposal / 实际 rail target / model prediction 的区别；不改 IK、rail 求解或原发送逻辑。验收一次 wall-clock target 被修订的反例，确认候选值和实际发布值同时保留。

修复复核还暴露了同类边界，已直接反馈 high：stop_check / max_duration break 不能记为 `phase_complete`；控制回调 `close(wait=False)` 后，生产资源停止链仍需最终排空日志（测试额外手动 close 不能代替生产）；CLI stub 应复用真实方法或校验真实签名，曾漏掉不存在的 `max_normal_speed_m_s` 参数；genesis 真 OpenCV 对空 JPEG 抛 `cv2.error`，须按坏帧拒绝后继续，不是只验证 fake decoder 的返回 None。

## 最终复核证据

独立重新运行 runtime / features / observer / Proxy / scan_path 五组：**53 passed / 5.10 s**。随后仅修改 coast 的实际 target 日志语义，独立定向反例 **1 passed / 0.39 s**；未无依据重复全组。另以真实图像进程的 genesis Python / OpenCV 验证空 JPEG 变为 ValueError；独立真实 `ShadowSuggestions` 拒绝错误 calibration，匹配帧使用声明 c_min=.8，quality=.6 时 alpha_preferred=.8125，未回落缺省 .5。全部为软件或已记录数据接口验证。

修复后的生产链已核对：具体 stop / timeout / PARTIAL_ARM 原因先于 close；热切换 close 不 join；运行资源停止后统一 drain，atexit 路径有真实子进程 footer 回归；相对姿态基于参考旋转，不取绝对 pitch；feature policy 与实际版本共同验证；坏 metadata/空 JPEG 拒绝后保留工作进程；CLI 使用真实公开方法签名、原 ICRA profile、4 N / .1 s gate，保留原 gate 内 seek 保护。rail 的原 proposal、wall-clock 修订后的 target 与 coast hold 分开；coast 不伪报一个没有发布的 target。

最终关键源 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `modes/contact_recording.py` | `547efa2f485aa97ed3f5518b5c8af2f676f5bd7958398c2fc0ec2dd455aaf874` |
| `modes/contact_qp.py` | `de7c6711d8c5e7df544ad5bb62f991eb2ea57e7e6d7a1446995d95d083a271ac` |
| `apps/contact_qp_run.py` | `aa78126b3683c324fb03195e8f855ea1b01db063d8219fda265af4844812345e` |
| `apps/contact_qp_features.py` | `50a68739c9b48c64d90994bd9827f79a78c78b9a5491e6594b4393e760e62f28` |
| `joint_admittance_8dof/loop.py` | `a1a66fca3a46640a5169412bb27ad8d4ba428dab699eb9f28279a93ee5cd6316` |

## 后续验收限度

只读 HEAD 源码确认旧服务会静默降级：旧 `track._hybrid_force_law` 仅特判 fce，其他 law 走 Legacy；旧 `build_track_hybrid_phase` 不读取 contact_qp。故新客户端向旧 daemon 提交该字段/未知 law，可以得到普通 TFF 的 install ACK，却没有新日志/新 QP。现 IPC ABI v2、mode、ack/install seq 与客户端提供的 label 都不能证明 Python 服务已加载能力。

实际执行入口需要在 SET_MODE **之前**读取服务自身发布的 capability/version，并与当前服务实例及新鲜 status 绑定；缺失时明确不可用，不发送模式请求。不能用本地 import、磁盘代码存在或客户端 label 的回显替代服务能力。保持原 native 与现有控制 SHM 布局；可沿既有 companion metadata 方式提供小型只读能力信息。recording 与 active 能力分开，尚未实现的 active 不提前宣传。该部署预检不影响用户继续使用原机械采集入口，也不是要求代理替用户重启或另行审批。

本轮 scoped PASS 不证明完整真实 200 Hz loop 的时延，也不把软件发布返回视为物理执行确认；错误/缺失 footer 的文件仍须视为未完整收尾。baseline/shadow 旧路径逐元素等价与 active 对新两时钟 nominal 的透明性是不同声明；本轮通过不代替后续 active 事务、真实几何及实测时基审核。

这些软件缺陷修复不证明声窗收敛，也不削弱原零裕量力指标目标。用户可主动使用已验证的旧机械控制采集真实特性；无需以合成声学收益 PASS 作为采集入口的先决条件。

## Capability 与安装边界补充审核

随后交付的 `core/capabilities.py`、API 预检和 daemon 最小改动，经只读关键路径审核及独立运行 capability / daemon boundary 测试，**13 passed / 0.34 s**，未发现新的关键阻断项。服务广告绑定当前 control SHM 与 companion header 的设备/inode，客户端还检查新鲜单调 status；旧服务、错实例或未广告 active 能力均在 HFPC 模式请求前拒绝。当前服务只广告 `contact_qp.recording_v1`，本结论不授予 active 能力。

contact_qp 请求在运行回调中仅进入既有 pending/stop 路径，资源构建使用退出 runner 后的原外层 compile；原普通模式热切换保持。记录 drain 已移出 runner，并位于 daemon 的外部 STOP 协同停止之后。已有故障停止分支在 runner 内或 ESTOP 回调中先执行原设备停止。此修复关闭了已定位的 callback 内文件/线程/hash 构建及日志等待先于外部 STOP 的缺口，不宣称完整回路实时认证。

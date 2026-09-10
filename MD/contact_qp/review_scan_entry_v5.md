# 原扫描入口接入复核 v5

审查范围：`ICRA_YM/script/scan_session.py`、`scan_robot.py`、`scan_simulation.py` 及新增入口测试；以 `scan_entry_v5_before/` 为原流程对照。只读源代码、离线假机器人测试及既有模拟 H5；没有连接设备或控制服务。

## 当前结论

**限定软件入口范围 PASS，B1 已关闭，无剩余阻断。** 离线 validator 已补充实际生效 QP/能量参数检查，独立定向复测 **29 passed in 0.88 s**。本结论不代表真机效果或物理能量认证。

### B1 历史反例：坏配置可能在教学、prepare 之后才被拒绝（已关闭）

修复前独立离线反例：配置 `{"mode":"shadow","qp":{"force_target_n":5}}` 被 `scan_session.parse_args()` 接受，返回配置；同一参数随后构造 `QpConfig` 抛出 `ValueError: contact QP v1 requires the fixed 4 N target`。当时 `validate_study_config()` 没有检查 `qp`，也未覆盖 active 的能量开关、账本及约束参数。真正构造发生在扫描 HFPC 阶段，因此这类静态错误可以晚于机器人初始化、教学和 prepare。

关闭证据：`_validate_effective_tasks()` 按 shadow/active 实际覆盖顺序构造 `QpConfig`，复用账本、端口边界/约束和测量对齐参数类检查 active 能量配置，不创建 solver、线程、IPC 或机器人。active 原控制器必覆盖的速度/加速度/角度限制不误当用户额外限制；baseline 不解释根本不使用的 QP/能量参数。新增 force=5、未知 QP 字段、NaN 权重、负能量余额、坏约束形状、未知能量字段六类 `main()` 反例，真实 Robot 与 SimRobot 均设禁止构造 sentinel。独立 29 项定向测试通过，B1 关闭。

## 已通过的入口行为

- 不传 `--contact-qp-config` 时保持 `contact_qp=None`，原 `scan(spec, recorder)` 调用和 `law="tff"` 不增加新参数；旧监督、停止、失败后人工重试与保存逻辑保持。
- active 路由为 `law="contact_qp"`，原目标 `force=4.0`、`scan_contact_n=4.0`、`label="icra_scan"`、tool 控制轴、原 ICRA 配置和接触等待保持。active 选择其他 force profile 会在入口拒绝。
- Robot 构造中的只读服务能力检查早于 LiveSamples 与教学；每次 attempt 也在 prepare 前复核。active 要求 `contact_qp.active_v1`，baseline/shadow 要求 `contact_qp.recording_v1`，避免旧服务默默执行其他模式。
- 每次 attempt 使用独立目录和绝对 `contact_qp.jsonl` 路径；传入配置深复制；失败重试不覆盖前次路径，也不污染原参数。manifest 与完成 H5 保留对应 attempt 的配置和请求路径。
- `--simulate` 明示只执行虚拟采集流程，未执行真实 QP、未生成 QP 日志。独立检查既有 `scan_entry_v5_simulation/001/` 的 12 个 H5：全部保存 `simulated_workflow_only_no_qp`，12 个请求日志路径互不重复，没有伪造 JSONL。

## 独立验证

`genesis` Python，禁用外部 pytest 插件，运行 `test_scan_contact_qp.py`、`test_scan_session.py`、`peirastic/tests/test_scan_path.py`：**46 passed in 13.98 s**。测试使用注入的假机器人，未访问真实 IPC。上述 B1 为额外独立反例，原通过用例未覆盖该配置。

首次复核的 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `scan_session.py` | `3b1670298a21ffd3b5d1dbe04d449fad842218d07af1d2028a398ef388a8cc3d` |
| `scan_robot.py` | `611f00690cacba6da826534dc15d62e581a94e943118a29900b493dec946399c` |
| `scan_simulation.py` | `c9fd651b6eb6294a736be43be1b9f9db705b308bdfbe2a9d29e65f695682b0c6` |
| `tests/test_scan_contact_qp.py` | `4e1529c8610216827107539c94d7b0ab9976c311f1870ac7e03df208636f0dd0` |

B1 关闭时新增/更新文件 SHA256（上述三个生产扫描入口未改）：

| 文件 | SHA256 |
| --- | --- |
| `peirastic/contact_qp/runtime_config.py` | `9d78181c4ae0ae792ca2c1ddeec0fb7b5f090b2a5ad7081b61559e7c81fd925a` |
| `peirastic/tests/test_contact_qp_runtime_config.py` | `82c3b868fd02c9d3b242c702f2277d926023db2f03dda6ca4ba09447d81f080b` |
| `ICRA_YM/script/tests/test_scan_contact_qp.py` | `9079fb3ad78bae1fa76e85dcd16690036da3a83a02b09cf4f602ad567434d0f4` |

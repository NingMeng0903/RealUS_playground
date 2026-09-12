# 持续视觉修正：实机验证入口

本页保留 Session 004 的修订记录。当前配置、Session 005 数据核验和完整启动命令请使用 [Session 005 反馈修正](SESSION005_FEEDBACK_FIX_20260911.md)，其中图像过期策略、死区和能量供能模型已有更新。

当前 `active_probe50_v8r3_tank.yaml` 使用 `permission_mode: continuous`。接触期间，只要图像有效且力门允许，confidence 任务持续参与 QP；没有累计 2 s、累计 3° 的视觉许可耗尽，也不等待三个健康帧恢复。左右差在死区内时自然没有差分转动请求，名义力矩转动已满足请求时也不重复添加转动。

最终动作仍经过原速度、加速度、姿态及孔径约束、六轴能量 QP 和最终 IK/rail 载荷审核。缺失或过期图像、力反馈故障、6 N 监督停止和原设备停止链继续生效。旧配置省略 permission_mode 时保留 bounded_episode，便于复现实验；新配置要求运行中的控制器声明 continuous_visual_v1，防止未重启的旧进程继续使用有限许可。

## 参数

- 4 N 目标、4–4.5 N 修复力门、6 N 监督停止沿用。
- confidence 死区 0.10、差分权重 10、修复速度尺度 0.002 m/s 沿用；本次没有新增放宽倍率。
- 初始罐能量从 0.6 J 调至既有容量上限 0.8 J，容量仍为 0.8 J，普通任务不能花的留额仍为 0.05 J。依据是 004 首次 L 净模型支出约 0.55 J，原初始额度已几乎耗尽；增加的 0.2 J 用作本轮验证的工程余量，不保证所有轨迹完成，也不是人体安全能量标定。
- 图像质量改善不产生罐能量信用。每个新扫描任务仍按配置初始化罐，单次任务内不按图像/修复片段重置。

## 命令

先退出旧 record 的重试提示，并在旧 controller 终端 Ctrl+C 后重启。以下长运行命令分别使用独立终端；已有正常运行且配置一致的 ultrasound、confidence、gamepad 进程可以继续使用。

```bash
# 控制器：必须重启以加载本次修改
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh controller
```

```bash
# 超声采集（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh ultrasound
```

```bash
# confidence 发布（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh confidence
```

```bash
# 示教手柄（尚未运行时）
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh gamepad
```

```bash
# 预检 confidence，然后按 5 mm/s、ICRA 力参数录制并保留原始数据
bash /media/camp/EXT_DRIVE/RealUS_playground/scripts/run_icra_tank.sh record
```

开始接触任务时 controller 应显示：

```text
[CONTACT_QP] visual=continuous; image=required; tank=0.800/0.800 J; reserve=0.050 J
```

`record` 前的三个不同新帧检查用于确认发布链路，不要求 confidence 达到 0.8，也不参与扫描中的许可消耗/恢复。

## 核验与日志

完整 contact/controller/rail 软件回归 435 项通过；随后补充的运行适配器发布测试也通过，continuous 专项共 18 项通过。报告：

- `analysis_artifacts/contact_qp_session004_20260911/continuous_visual_tests.xml`
- `analysis_artifacts/contact_qp_session004_20260911/continuous_visual_focused_tests.xml`

旧 004 固定数据的许可回放：第一次 L 的 3267 个、第二次 L 的 625 个原先耗尽且有差分方向的周期恢复任务可用性。新策略在全部 21132 个记录控制样本上都没有 episode 耗尽。48 个抽取的旧输入代表点均可求解，保留其旧可用能量时均满足能量及导出的硬约束。数值检查缺少旧日志中的精确角状态，使用零相对角；没有重放机器人与组织的动力学、最终 IK/rail 发送或预测新的图像质量/完整任务耗能。

回放结果：`analysis_artifacts/contact_qp_session004_20260911/continuous_visual_replay.json`。本次没有启动或连接设备，没有修改原始采集数据。

新 `control_sample` 将 `repair_episode` 保存为 JSON 对象，`permission_mode=continuous`、`exhausted=false`，不再输出有限许可的剩余秒数/角度。`allocation_diagnostics` 增加 `repair_permission_mode` 和 `qp_delta_omega_y_rad_s`，后者是 QP 相对名义角速度的增量，包含 QP 其他约束效果，不能单独解释为已实现的纯视觉转角。实际发布值继续看 `publication.final_command_model_tool`。

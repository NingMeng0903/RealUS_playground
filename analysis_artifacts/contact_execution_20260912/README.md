# 连续推进、平顺旋转与短时恢复交付

2026-09-12。实现采用 HIGH，独立 ULTRA 审查见 [ultra_review.md](ultra_review.md)。本次仅修改软件、编译 native 并运行离线测试，没有操作机器人、地轨或采集新的人体扫描。

当前默认配置仍是 `peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml`，新增显式 `execution_policy: continuous_recovery_v1`。能量罐保持初始 **0.100 J / 容量 0.150 J / 留额 0.050 J**，原法向目标 4 N、视觉力门控 4–4.5 N、实际收到的原始/滤波 6 N 停止条件保留。

## 逐机制启用清单

| 机制 | 本轮状态及边界 |
|---|---|
| 实际参考时钟 | 成功发布按实际控制间隔和最终接受的 α 推进；未来命令预算的 5 ms 模型与参考间隔分开。拒绝、等待没有积分欠账；恢复时扣除失败计算占用的时间。 |
| 连续质量减速 | `α_target=1−0.75 loss`，范围 `[0.25,1]`；变化率取现有路径 ramp。5 mm/s 对应最低质量目标 1.25 mm/s。最终机械执行仍可进一步限速。 |
| 视觉弱侧目标 | `continuous` 不设修复倒计时；方向取左右低于 0.8 的缺额差，保留现有死区、视觉速度尺度、方向标定及力门控。两侧都健康时不再为数值不等增加视觉修复。 |
| 图像证据 | 相同图像不重复更新质量证据，不逐拍累加视觉角速度；图像超过 300 ms 继续使用既有 `pause_visual`，撤去新视觉请求。 |
| 持续漏边 | 记录弱侧区间、无改善标记、请求缺额、力门控、角度/速度/加速度/jerk 边界、能量边界及最终请求差额。全部为诊断，不因质量差启动新的整条重采流程。 |
| 最终旋转历史 | 成功发布的最终命令角速度、加速度、时间另存，比较前统一到当前工具坐标；外层名义历史仍保留外层命令，不写入地轨补偿后的速度。 |
| Native / Python 平顺性 | tool-y 区间进入两级内层 QP 与最终复核；jerk 上限由 `a_max/(I/D)` 得出。冲突按 jerk→加速度→任务角速度→原机械约束降级并记录，原关节/碰撞限制保留。 |
| 实际发布时间审计 | 另记录真实发布间隔、角加速度、jerk、实际层级与 `timing_limited`。求解后运输/调度抖动可能影响实际 jerk，**不宣称任意时序下严格满足 jerk 上限**。 |
| 两次数值求解 | 缓存 workspace 30×10；失败后同一问题用全新 Ruiz workspace、rho=1e-3、200×10。保留原容差、有限性和残差检查；在线不做 LP/Fourier 失败诊断。 |
| 未发送恢复 | 明确未发送时撤销 nominal、inner、rail proposals，等待新力样本；原已提交命令期限不变，参考和罐不重置、不续租、不补入能量。期限耗尽、真故障、部分/未知发送仍停止。 |
| Native 迟到结果 | 已撤销的迟到解必须丢弃，再用当前 frame、twist、旋转区间重新求解；原 native 50 ms 请求期限仍先检查，不能借 abort 续期。 |
| 力反馈间隔 | 发送仍要求当前样本 `<15 ms`。历史 15–50 ms 间隔仅在原双设备 lease 有效、同一时间域、前后水印一致时恢复；50 ms 从原命令期限派生。 |
| 恢复滤波 | 恢复间隔用版本化精确 FOH 传播低通状态，保留真实两个端点；普通间隔维持原滤波，高通按实际间隔消费一次。不能重建未收到的中间峰值。 |
| 力与故障发布 | 补偿力在 observer 后、QP 前发布；短时恢复显示 `[RECOVERING]` / `[RESUMED]`，终点显示 `[ENDPOINT]`。原故障通知由预创建线程处理，避免阻塞设备停止。 |
| 配置和记录 | 增加 Python 能力检查；native ABI **v9，692/1496 字节**及源/二进制哈希校验。记录 α 各阶段、实际间隔、求解尝试和完整失败输入；新版数值记录保留 ±inf、NaN、数组形状及类型，可离线重放。 |

质量 α、外层 α 和最终接受 α 分别记录；每次发布的 `progress_limited_by` 区分质量、外层约束、内层/最终模型、等待接触与终点收敛。原始实际 TCP/HDF5 记录继续保留，用于区分参考进度与实际移动；不能把参考速度或命令角速度冒充测得的机器人速度。

## 已执行验证

- 完整离线回归：**659 passed**，255.12 秒。native 编译成功，CTest **1/1 passed**；日志 [full_regression.log](full_regression.log)。
- 最新 T7 历史：**20,990 个重建输入，0 次拒绝**；[t7_replay.json](t7_replay.json)。历史没有记录的失败 wrench 使用前一条，成功样本未记录的角度取零，因此这是公开假设下的数值重建，不是精确原始或物理闭环回放。
- 最新诊断补充后：真实 active runtime / recording 回归 **28 passed**；独立 ULTRA 再核验 **47 passed**。这些与全量测试重叠，数量不可相加。
- 采集端既有回归：**43 passed**；真实 recorder 子进程与合成流握手 **1 passed**。本地套接字测试在允许 IPC 的环境中执行；数据均为合成数据。
- 覆盖旧 `/003` 约 0.543 J 数值反例、低/零能量、真正不可行、NaN、两次失败、截止耗尽、15/50 ms 边界、重复/倒退时间、未提交/过期 lease、部分发送、4.5 N / 30 ms 假峰、真实 6 N、150/179/200 Hz 参考速度和 final history / native 层级一致性。

## 可重跑命令

在正常终端运行以下离线命令，不启动硬件：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/test_contact_execution.sh --full
```

仅重跑本轮主要回归可使用 `bash scripts/test_contact_execution.sh focused`。

重跑原 T7 历史（需要原数据盘仍挂载）：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
source rm75_control/env.sh
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src:${PYTHONPATH:-}"
python scripts/replay_contact_execution.py --t7-history \
  --output analysis_artifacts/contact_execution_20260912/t7_replay.json
```

重放某个**新版**失败记录的精确矩阵：

```bash
python scripts/replay_contact_execution.py \
  --record '/完整路径/contact_qp.jsonl' \
  --output /tmp/contact_failure_replay.json
```

这里有意取消已过期的实时期限以检查数值问题；离线求解成功不意味着原截止时间内能够发布。

采集端回归使用实际 recorder 的 Python 环境：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:rm75_control:src \
/media/camp/EXT_DRIVE/envs/genesis/bin/python -m pytest \
  /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_air_handoff.py \
  /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_contact_qp.py \
  /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_session.py -q
```

## 启动和受控扫描

**必须重启 controller 才会加载新代码和 native。** 已在运行的 ultrasound / confidence / gamepad 可继续使用；它们的图像注册参数没有改变。若需完整启动，各长期进程放在独立终端：

```bash
# 终端 A：controller
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh controller
```

```bash
# 终端 B：ultrasound（未运行时）
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh ultrasound
```

```bash
# 终端 C：confidence（未运行时）
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh confidence
```

```bash
# 终端 D：gamepad（未运行时）
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh gamepad
```

```bash
# 终端 E：检查新鲜图像，然后按原操作交互采集
cd /media/camp/EXT_DRIVE/RealUS_playground
bash scripts/run_icra_tank.sh record
```

原入口也兼容，完整命令为：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record \
  --force-profile icra \
  --speed-m-s 0.005 \
  --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8r3_tank.yaml
```

进入 TRACK_HYBRID 时应看到 `execution=continuous_recovery_v1`、`clock=actual_success_interval`、`quality=deficit_only`、`solver=bounded_retry_v1`、`source_gap=lease_fresh_foh_v1` 和 `rocking=final_tool_y_v1`。旧 controller 会被配置能力检查挡在请求前。

下一次受控测试应使用相同起终点、路径形状和采集设置，结合实际 TCP 位置对齐侧边图像，分别比较最终发布与测得角速度变化、沿路径移动速度、弱侧持续区间及同位置侧边质量。终点收敛时间单列。请求数量、请求方向或较高 confidence 本身均不能替代侧边贴合效果；本次尚无新的物理扫描来得出改善比例。

没有实现 `c_k`、`v_z=v_F+c_k ω_y`、联合动态耗散/力增长约束，也没有把 `c_p` 当作 `c_k`。能量结算仍是原命令模型，**不宣称物理端口被动性证明**。

# 用现有扫描流程验证新外环

本轮已在原 `ICRA_YM/script` 增加 `--contact-qp-config`，无需绕过原示教、离面定位、寻触、扫描监督、退离和 H5 保存。默认不传该参数时保持原行为。本文件中的实际扫描命令由用户启动；本轮只做了离线检查和模拟流程。

## 1. 先记录原控制器与新检测

使用正常的单套 controller / gamepad / ultrasound 服务。若提示旧 controller 不支持 contact study，需要在停止原服务后按原 `bash run.sh controller` 入口启动本版本；不要并开第二套。新入口会在示教/prepare 前检查服务能力，不能因为新客户端可导入就认为旧服务支持新模式。

独立终端启动图像特征 worker；如果已经有相同配置的 worker，复用它：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_features \
  --feature-config peirastic/config/contact_qp/baseline_welleweerd.yaml
```

它从 `17359` 读取已有超声流，在 `17361` 发布摘要，不发机器人命令。当前 profile 的窗口版本是 `a3dd6a785d0fad60c1fc`；图像计算在此独立进程运行。

在扫描终端启动 **baseline 记录**：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh check
bash run.sh record --force-profile icra --speed-m-s 0.005 --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/baseline_welleweerd.yaml \
  --data-root '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/baseline'
```

这条 `record` 命令会产生原流程的机器人动作。目标力仍为 4 N，新 QP 此时不控制运动。5 mm/s 是首轮便于观察的扫描速度，后续对照保持相同设置。按原提示示教、逐条 Enter；首轮可先完成一条直线，在下一条提示时输入 `q` 结束，不必做满 12 条。

每次 attempt 自动保存 `attempts/<路径>/<次数>/contact_qp.jsonl` 的日志位置，并保留 raw H5。模式、完整配置和实际日志路径写入 session、attempt 及最终 H5 的 `scan_metadata_json`。不要给所有 attempt 指定同一个日志文件。

## 2. 根据这一轮检查实际图像和接口

用新录 H5 检查图像：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_preview \
  '/实际新录文件.h5' --frame 50 \
  --feature-config peirastic/config/contact_qp/baseline_welleweerd.yaml \
  --output /tmp/new_contact_check
```

检查红色低 confidence 区域是否落在肉眼可见的缺失声窗，橙色未知列是否确实缺少可用信号。0.8 是冻结的实验初值，不是论文给定常数；如在设计数据上改阈值，修改同一份 feature 配置并更新版本，再做新的独立验证，不能继续把旧结果当同版本验收。

baseline JSONL 中 `control_sample.feature` 应有持续更新的图像帧号、有效性、质量和配置版本，且日志无丢行。取其中实际 `registration_version`；它关联 publisher 身份、crop、hflip、延迟及标定。publisher 重启或配置变更后重新读取，不能照抄之前运行的值。

从同一 `wrench_source_id` 内、去重后的递增 `wrench_source_time_s` 检查力/机器人 ingress 周期与抖动。它不是超声帧率，也不是 5 ms 控制周期；不同 source epoch 不合并。旧 H5 或日志的统计值用于测量，不自动代替已验证的时限声明。

## 3. 填实际安装后运行 shadow / active

复制 baseline YAML 为自己的 study 配置，填入以下实际量。当前模板故意保留未标定状态，不能只把 `verified` 改为 true 来绕过检查。

| 配置 | 依据 |
|---|---|
| `geometry.half_length_m` | 有效声学孔径的半长，单位 m |
| `geometry.T_tcp_face`、`face_normal_convention` | 接触面到当前 TCP 的真实刚体外参及面法向约定 |
| `geometry.image_x_sign` | 图像左右与探头物理左右的核对结果 |
| geometry/feature 的 `calibration_version`、`verified` | 同一次已核对的安装配置；feature 图像方向必须一致 |
| `feature.config`、`window_version`、`registration_version`、`c_min` | 与实际 worker 完全一致；v3 的 c_min 必须等于预览阈值 |
| `source.period_s/jitter_fraction/max_age_s` | 当前力/机器人测量 ingress 的实测与许可范围 |
| `force_axis_monotonicity_confirmed` | 原工具 Z 轴 4 N 标量与接触加载方向的核对结果 |

若修改了 `feature.config`，可离线计算新的窗口 hash：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/rm75_control" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -c \
  "from peirastic.contact_qp.features import load_feature_config; print(load_feature_config('/绝对路径/study.yaml').window_version)"
```

先设 `mode: shadow`，让新 QP 只生成日志建议，运动仍用原 TFF。用第 1 节的同一 `record` 命令，将配置路径和输出目录分别换成自己的 shadow 文件与目录。确认建议把局部速度指向缺失一侧，修复后保护已改善窗口；`shadow_candidate` 若一直 unavailable，先按 reason 修复配置/图像接入，不进入 active。

再设 `mode: active`，先离线验证：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/rm75_control:$PWD/src" \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_run \
  --config /绝对路径/active.yaml --validate-only
```

验证通过后，由用户使用保留原监督流程的入口执行：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh record --force-profile icra --speed-m-s 0.005 --keep-raw \
  --contact-qp-config /绝对路径/active.yaml \
  --data-root '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active'
```

初次新外环动作验证沿用原现场验收流程，先在可重复的体模接触条件下检查左右映射和动作方向。首轮保持 `energy_constraint_enabled: false`，原机械保护照常工作；能量监测/预算另按已声明的真实端口配置启用，不将其与第一次声窗方向验证混为一次参数变化。

不要用 `contact_qp_run --execute` 替代上述完整流程：该低层入口仅提交一条已示教 HFPC，不管理原来的 prepare、supervisor 和 H5 记录。

## 4. 怎样判断外环实际有用

保持目标力、示教起终点、探头设置和扫描速度一致，交替做 baseline 与 active。先做几个独立配对扫描，保留正常贴合、单侧欠缺以及无法恢复的片段，不只保存成功例。

- 对齐图像有效时刻与实际累计局部运动，检查缺失一侧是否在真实修复动作之后改善，而不只看上一拍 ω 的符号。
- 用实测路径计算缺失长度、有效覆盖、恢复时间、扫描总时间和中止；不要用 α 或参考进度代替覆盖。
- 按独立扫描计算相对 4 N 的力 RMSE、绝对平均偏差和峰值绝对误差。既定零退化统计要求仍保留，少量试运行只能发现问题，不能自动满足该统计验收。
- 若 active 更慢，再做匹配实际扫描速度的 baseline；若 rocking 量不同，再匹配转动预算，以判断改善来自修复方向还是仅仅移动更慢。

完整路径 spec 已存于 `session.json` 的 `hands[RH/LH].paths`、`RH_paths.json/LH_paths.json` 和成功 H5 的 `scan_metadata_json.path`。曲线路径含随机种子，严格配对时使用同一份已保存 spec，不能重新随机生成曲线后称为同一路径。

## 本轮已验证的入口范围

新参数 `--help` 已实跑；默认无参数旧路径、baseline/shadow/active 路由、能力拒绝在动作前、唯一路径和 metadata 均有离线测试。完整 `--simulate --auto-enter` 成功保存 12 条模拟 H5 并核对 metadata；明确标为 `simulated_workflow_only_no_qp`，这只是流程验收，不是 QP 效果或实机验收。

# 超声接触离线审查

只读取原始扫描 H5，不调用机器人。检测结果是浅层持续暗带候选，不是机械脱离接触真值。默认参数用于 2026-09-09 这批固定图像裁剪；换成像深度、增益或探头时需要重新验证窗口与阈值。

完整运行：

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
MPLCONFIGDIR=/tmp/mpl-contact OPENBLAS_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/isaac_lab/bin/python \
  scripts/ultrasound_contact_audit.py \
  --input '/media/camp/PEI_T7/icra 2027/uncalibrated' \
  --output '/media/camp/PEI_T7/icra 2027/contact_analysis/20260910'

MPLCONFIGDIR=/tmp/mpl-contact OPENBLAS_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/isaac_lab/bin/python \
  scripts/ultrasound_admittance_audit.py \
  --input '/media/camp/PEI_T7/icra 2027/contact_analysis/20260910' \
  --output '/media/camp/PEI_T7/icra 2027/contact_analysis/20260910/admittance'
```

只重新计算候选和图，不重新解码全部 JPEG：

```bash
MPLCONFIGDIR=/tmp/mpl-contact \
  /media/camp/EXT_DRIVE/envs/isaac_lab/bin/python \
  scripts/ultrasound_contact_report.py \
  '/media/camp/PEI_T7/icra 2027/contact_analysis/20260910'
```

第一条命令可加 `--features-only` 只生成特征缓存。`--dark-threshold` 修改灰度阈值，默认 40。`--us-delay` 和 `--force-delay` 分别指定超声相对 TCP、力相对 TCP 的有效延迟，单位秒；默认采用原体模结果 0.15196365053143765 与 0.0006777471734364515。脚本拒绝已经对齐的 `calibrated` H5，防止重复加延迟。

输出包括：

- `index.html`：全部扫描的图表入口。
- `analysis_report.md`：本批数据结论、文献对照和融合 QP 设计说明，人工审阅文档，不由检测脚本自动改写。
- `events.csv`：候选区间、帧号、绝对超声时间、载荷与转速统计。
- `scan_summary.csv`、`person_summary.csv`：实际采样率与分组统计。
- `candidate_summary.csv`：65%、75%、85% 条带暗像素占比阈值的敏感性。
- `delay_sensitivity.csv`：US 有效延迟变化 ±20 ms 时，事件 My 低于阈值占比的变化。
- `人员/扫描/signals.npz`：原始高频时间、位姿、wrench、计算角速度与图像特征；不包含再次编码的 JPEG。
- `人员/扫描/frames.csv`：每帧图像特征与最近高频样本，有效性单独标记。
- `人员/扫描/timeline.png`、`event_XX.jpg`：同步曲线和候选前后帧。
- `admittance/`：假定接管门控开放的导纳预测，不能当作实测命令。

时间与字段：

- `time_s` 从对应原始 H5 第一帧超声起算。高频流已经按默认有效延迟移到图像时钟。
- `start_time_s` 与 `end_exclusive_time_s` 定义半开事件区间；`end_time_s` 是末帧时刻。
- `my_deadzone_fraction` 为兼容字段名，实际含义仅为事件中 `abs(My)<=0.025 Nm` 的高频样本比例，不是控制器门控日志。
- `omega_y_speed_near_limit_fraction` 仅表示实际速度接近名义限速，不证明命令饱和。
- `side` 依据最大暗带帧的图像半边；`both` 表示两边均有足够宽暗带。尚未转换为工具方向。
- `image_manual_review` 默认 `unreviewed`；重点看图判断另见分析报告与 visual_spotcheck，不会自动生成机械接触真值标签。
- `left_censored/right_censored` 表示文件边界截断，`aligned_frame_fraction` 表示该事件的高频数据对齐覆盖。

工程检查：

```bash
/media/camp/EXT_DRIVE/envs/isaac_lab/bin/python scripts/test_ultrasound_contact_audit.py
```

检查覆盖深部声影排除、浅层暗带、双边标记、短暂噪声排除、间断合并、最近邻等距选择。它们不证明检测器的接触识别准确率。

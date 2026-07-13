# 四相机 SMPL-X burst 验收（只使用新录制）

本验收不允许使用 `smplx_outputs` 内此前两次结果作调参或统计样本。

每组使用更新后的 RealSense 发布器（必须带 `--undistort`）和以下采集入口：

```bash
python -m projects.genesis_ue_sync.multiview_realtime.cli.run_offline_terminal8_capture \
  --config configs/tracking/realus_dwpose_easymocap.yaml \
  --capture-burst-s 0.5
```

入口会拒绝缺少 `image_geometry.undistorted=true` 和
`projection_distortion_model=zero` 的帧。每个新 run 保存四路 burst RGB、
`burst_sync_metadata.json`、`raw_simcc.npz`、逐帧 SimCC 候选/不确定度、逐关节
融合内点诊断及 EasyMocap 输出。

录制至少 20 个独立 burst，覆盖：躺姿、坐姿、伸腿、屈腿、单脚悬空、交叉腿、
脚部局部遮挡；另录制“两个清晰视角 + 一个错误/遮挡视角”和“仅单视角”脚点场景。

人工标注每组双脚的 6 个脚点（各相机可见处的 2D 真值），并在新录制上比较旧/新链路：

- 脚点缺失率、2D 重投影误差和跨帧抖动；
- 错误两视角接受率与错误第三视角剔除率；
- 两个清晰视角是否以 `observed_low_two_view` 保留；
- 单视角是否保持 `missing`，不被时序补造；
- `final_quality` 是否通过，且 `bed_penetrating_verts` 为零；
- `smpl_root_alignment` 必须是 `root_optimized_in_fit_no_posthoc_shift`。

只有 `fit_ok=true` 才会发布高精度 mesh。脚部、躯干、最终 SMPL-X 重投影或防穿透
任一质量门失败时，run 标记为 `degraded_skeleton_or_resample`，不向 Genesis 控制链路发布高精度 mesh。

# Probe50 直线超声视频

- `baseline_raw_gray_1x.mp4`、`shadow_raw_gray_1x.mp4`、`active_raw_gray_1x.mp4`：原 H5 灰度画面，实际时间 1 倍速，664×726。
- `comparison_elapsed_1x.mp4`：三列自各自片段起点按实际时间 1 倍速播放；已结束列清空并显示 END。
- `comparison_normalized_elapsed.mp4`：三列各自时间进度 0–100% 同步，**不是空间配准，也不是所有列 1 倍速**。
- `contact_sheet.png`：五个时间百分比的完整画面预览，只有预览缩小了尺寸。

输入为各次已有接触片段 H5；保留 H5 画面本身，不再裁剪、翻转、增强或改变灰度对比度。MP4 为 H.264 CRF18 播放副本，存在常规编码损失；原 H5 是定量分析来源。

按 `ultrasound/timestamp_ns` 以 30 fps 重采样，使用最近已出现源帧；不以帧数猜测时长，也不额外施加图像有效延迟。原片段时长依次为 29.583、29.648、30.601 秒。每次重新示教，不能作严格同路径或逐像素对照。

本环境无 ffprobe，使用 bundled ffmpeg 对五个 MP4 完整解码，全部返回 0；编码、解码日志及精确时长在本目录和 `manifest.json`。复现：genesis Python 执行 `export_videos.py`。

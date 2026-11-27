UE 默认柔光环境
================

公共场景加载器现在默认使用 `soft_studio`：深灰哑光地面、淡灰三段连续渐变天空、一次采集的天空补光和一盏主光。地面宽 2 km，在床附近的相机中呈现延伸到远处的平面。天空没有照片、云和太阳图案。

加载场景时自动应用，进入 PIE 后保留。RealUS 四/五相机场景及场景生成脚本均显式启用这个预设；未指定新字段的旧配置也默认启用。原地图的 HDRI 和后处理在当前加载实例中停用，原始地图文件不由此模块保存。

配置示例（颜色为线性 RGB）：

```yaml
environment:
  ue_environment_preset: soft_studio
  ue_ground_plane_color: [0.055, 0.060, 0.070, 1.0]
```

`ue_environment_preset: inherit` 使用地图原有环境。`ground_plane_color` 继续控制 Genesis 的地面，UE 使用单独的 `ue_ground_plane_color`。

实现位于 `src/projects/genesis_ue_sync/cli/render/unreal/ue_studio_environment.py`。两个材质首次生成到 UE 项目的 `/Game/Bedlam/Generated/RealUSStudio`，后续直接复用。天空光分辨率为 128，不逐帧采集；该预设关闭 Lumen、光追效果，使用普通阴影贴图和轻量屏幕空间环境遮蔽。远处使用非体积淡雾过渡，工作区内不加雾。离线序列生成也不再加载草地 HDRI 模板。

床上的 `GEN_human_anchor` 是人体初始化位置的调试球。它现在仅用于编辑器定位，设置 `Hidden in Game` 并关闭投影，不出现在 PIE 或渲染相机中。相机验证使用场景配置的 cam1–cam5 和原有实时采集插件，保留相机位置、内参及输出分辨率。

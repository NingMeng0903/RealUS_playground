# Confidence 提取复核与修正

用户指出旧预览右侧几乎全黑，复核确认这一问题真实存在。预览左半是 B-mode 原图，右半才是 confidence，并不是一幅超声图的左右两侧。69 张旧预览去掉强制边界行后，confidence 小于 0.01 的像素比例中位数为 91.85%（8 位预览量化值）；旧结果没有提供足够的声窗区分能力。证据保存在 `confidence_pre_fix_v3.json`。

## 修正依据

旧 `randomwalk_thesis_v1` 按论文正文公式实现，但没有与作者公开实现做数值对照。公开的 [CAMP B-mode MATLAB 实现](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/confMap.m) 在几个实际计算步骤上不同：输入 min-max 归一化、`1-exp(-alpha*depth)` 深度权重、边梯度两次归一化、横向与斜向同一 gamma，以及加性 `1e-5` 边权。分别见 [深度权重](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/attenuationWeighting.m) 和 [图 Laplacian](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/confidenceLaplacian.m)。论文正文与公开代码并不完全相同，因此新版本明确称为代码兼容版本，不声称二者逐项一致。

默认提取器现为 `randomwalk_camp_bmode_v2`，与该公开代码的 B-mode 路径对应。旧算法及历史窗口 hash 保留，可精确回放旧版本；新版本自动改变窗口配置身份。没有拉伸最终 confidence 值，也没有把黑区 mask 乘进 confidence。gamma 保持已有策略值 0.03，参考代码默认 0.05；本次没有据真人效果调它或修改质量阈值。

## 已验证与未解决的问题

21 项 feature 测试通过，包含独立稠密小图求解对照、镜像、边界、版本及黑区反例。另从旧预览分布的五个分位位置选取 5 张原始 H5 帧重新计算，而非按新结果择优展示。[对照图](confidence_v2_check/comparison.png) 的两版 confidence 共用固定 0–1 色标，窗口位置相同。

一张有明显左侧暗区的真实帧，左右窗口分数从旧版约 0.017 / 0.037，变为新版约 0.549 / 0.866；新版能呈现此次样本中的左右差异。这是图像特征证据，没有机械接触真值标签，也不是修复动作的闭环效果验证。

**修正仍不使 random-walk confidence 变成未接触黑区分割。** 左半常数灰度 80、右半全黑的解析反例中，新版右窗口分数仍约 0.878。随机游走描述到达源边界的概率，局部缺少强反射梯度的黑区不一定得低分；高分不能证明有回波或已接触。整幅空白图已有失效判定，也不能因此声称局部黑区全部已处理。

因此不能直接把旧仿真用的 `c_min=0.5` 当作真人接触阈值。真实配置仍保留未标定状态，旧 v1 的 12,588 帧分数及设计实验结果保留为历史证据，不能混入 v2 的阈值选择或新效果结论。黑区灰度占比仍可单独作为分析对照；本次没有暗中把它改名为 confidence，也没有增加一套接触控制模块。

复现脚本：[`scripts/compare_contact_confidence_v2.py`](/media/camp/EXT_DRIVE/RealUS_playground/scripts/compare_contact_confidence_v2.py)。所用原图索引、配置、分数和选择规则保存在 [`comparison.json`](confidence_v2_check/comparison.json)。

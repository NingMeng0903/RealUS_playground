# Confidence V2 关键复核

结论：**参考算法实现修正 PASS；黑区/脱离识别效果未获验证。** 独立运行 feature 测试 21 passed / 0.24 s，并核对固定 0..1 色标的五帧真实图与黑色半幅反例。

直接核对的 primary 源为 [confMap.m](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/confMap.m)、[attenuationWeighting.m](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/attenuationWeighting.m)、[confidenceLaplacian.m](https://github.com/TJKlein/Nakagami_Confidence_Maps/blob/master/confidenceLaplacian.m)。V2 在处理分辨率上匹配 B-mode 输入 minmax、`1-exp(-alpha*d)` 深度权重、含零对角的两次梯度归一化、横向及对角 gamma、`exp(-beta*cost)+1e-5`、顶边 1/底边 0 的图方程。项目 gamma=.03 与公开代码默认 .05 的差异已明示；这不是实机校准结果。

旧 V1 算法及版本 hash 保留，未把显示拉伸或黑区掩膜混入概率解。新旧阈值不具有互换依据。真实左侧大片黑区样例 V2 的左窗值仍为 0.5485；以 0.5 判断仍会漏检。左半常灰、右半纯黑的反例中，右窗仍为 0.8783。它们说明传播概率不等于接触占比或脱离标签，图变亮不能表述为接触检测已修好。当前未增加黑区检测模块，也未用这些帧调控制阈值。

审核 SHA-256：

- `peirastic/contact_qp/features.py`：`47ec030843bfd50f666752708399f768ff3ba4c29ee9044f2af343fb248e0692`
- `peirastic/tests/test_contact_qp_features.py`：`81d198d4c9c484f0cd55bb19b6a8f6220b626fbc65688bfe19c3a2bd5f559185`

图像与数值证据：`MD/contact_qp/confidence_v2_check/comparison.png`、`comparison.json`。本轮只核对算法与证据边界，没有连接机器或宣称声学收敛。

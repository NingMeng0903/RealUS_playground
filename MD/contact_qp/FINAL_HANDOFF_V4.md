# Welleweerd confidence 与外环能量修正交付

本轮按用户指定由 GPT-6 medium 编码，ultra 只检查关键能量证明和外环理论。没有启动设备、重启用户服务或执行真人扫描。原 4 N 力控、TCP 转动基线及内环 IK 保留。

## 现在直接检查图像

先看已经生成的 [真人左侧缺失帧预览](welleweerd_v4_check/real_4.png)：左/中/右浅层 confidence 为 **0.546 / 0.733 / 0.918**，阈值 0.8 将左侧判为低可信度。另一个局部左侧欠缺样本为 **0.725 / 0.879 / 0.896**；三个较好图像的左右窗均超过 0.93。五张图沿用之前固定选择，不根据这次检测结果重新挑选。完整数据在 [summary.json](welleweerd_v4_check/summary.json)。这些样本没有机械接触真值标签，不据此报告识别准确率。

复现命令如下；更换 H5、`--frame` 即可检查新录的数据，也可以把 H5 换成 PNG/JPEG。

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/rm75_control" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_preview \
  '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/uncalibrated/chenwei/RH_Per_C_PtD.h5' \
  --frame 95 --feature-config peirastic/config/contact_qp/welleweerd2020_features.json \
  --output /tmp/contact_check
```

输出 `/tmp/contact_check.png` 与 `.json`。图中红色为低 confidence，橙色为未知列；JSON 保留未修改的完整 confidence、每列曲线、区域、两种均值和像素重心。纯黑反例仍可能得到较高浅层 RW 值，所以另外标记无信号未知，禁止把它解释成良好贴合。

## 同一图像配置接到在线 worker

```bash
cd /media/camp/EXT_DRIVE/RealUS_playground
env PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/rm75_control" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m peirastic.apps.contact_qp_features \
  --feature-config peirastic/config/contact_qp/welleweerd2020_features.json
```

它订阅已有 `17359` 超声流，在 `17361` 发布特征，不连接或控制机器人。控制器的 `feature.config` 使用相同内容，`window_version` 使用该内容的 hash；标定版本改变后需要重新计算。当前未标定 profile hash 是 `a3dd6a785d0fad60c1fc`。Welleweerd profile 的预览阈值与控制器 `feature.c_min` 必须相同。

profile 对应论文的 Karamalis 随机游走、浅层均值及重心。高 100/宽 145、浅层 0–22%、阈值 0.8 是明确记录的实验初值；论文没有公布这些轴次/ROI/阈值的完整配置。没有训练模型，也没有声称 confidence 就是机械贴合概率。详见 [方法对应](WELLEWEERD_METHOD_V4.md)。

## 能量与 QP 审核结论

能量漏项已修复：同一个 tank 结算 **六维净输出功 + D 耗散**，QP 和结算共用 Pc、d。实测结算使用同次原始 W/臂位置与两侧夹持的地轨编码器样本，独立于命令提交。缺测不退款，重复物理区间不重复结算，负余额不截断补能。

[独立证明与理论审核](ENERGY_PROPERTY_REVIEW_V4.md) 给出了声明误差界下的整拍、每个前缀和累计关系：

`∫ WᵀV_actual dt ≥ −S_initial + ∫ D dt`。

该性质要求实际执行、功率、耗散和停止暴露的界成立，且预留/结算相容、账本非负。现在连接的是实测 MONITOR 估计，特别是平均速度平方不是真实耗散上界。软件不会把未验证误差界标为已认证。此证明不要求修改 IK，但不能把外环 `v_cmd` 等同于实际运动。

几何关系和孔径端点约束有明确推导；力符号条件只保证声明接触模型下的局部附加恢复方向；声窗修复/保持、α 偏好属于控制策略。它们不证明图像必定改善，也不证明全轨迹力误差零退化。后两项用你新采集的实际动作—图像和配对力数据验证，不再依靠合成声学世界调参。

## 控制器验证入口

`peirastic.apps.contact_qp_run --config <study.yaml>` 默认只做离线配置检查；显式 `--execute --path-spec <已示教路径>` 才向已有控制服务发起扫描。active 配置仍需真实孔径、安装外参、图像左右映射和测量时基，模板没有代填假标定。原始采集入口和 baseline/shadow/active 运行关系见 [REAL_CAPTURE_ENTRYPOINTS.md](REAL_CAPTURE_ENTRYPOINTS.md)。

源代码在 `peirastic/contact_qp/`，图像入口为 `peirastic/apps/contact_qp_features.py` 和 `contact_qp_preview.py`，硬件适配在 `peirastic/realman8dof/modes/contact_active.py`。当前 source overlay 和各文件 hash 见 `FINAL_SOURCE_MANIFEST_V4.json`；旧 V3 与失败实验记录保留。

## 最终验证

- 最后一次修改后，`peirastic/tests -k 'not 100000'`：**554 passed / 3 deselected，19.13 s**；JUnit 为 [peirastic_regression_v4.xml](peirastic_regression_v4.xml)。三个长序列在此前已通过，本轮没有改变名义动态方程，不重复运行。
- 原 force/admittance/Ke/guard/TDPA/observer 相关 **48 passed，11.27 s**。
- ultra 最后定向复核 **17 passed**，关闭已知 invalid rail 夹在有效点之间仍跨段结算的阻断项。无效点清空历史且不作为新锚点，之后从有效区间恢复；有效重复帧不误清空。
- worker 摘要消息检查 8192 字节接收上限；完整列曲线留在预览文件。v3 配置拒绝预览阈值与 QP 阈值不一致。
- 真实帧 → 特征 → QP 接口检查产生左修复、右保持，`alpha≈0.762`，硬约束残差为零。这里几何明确为仿真值，只检查接口与方向计算，不是声学闭环效果实验。
- 离线配置 CLI 通过；native Python/C++ 协议仍为 8，native/IK/rail 算法源未变，本轮没有重复构建未改动的 native。`git diff --check` 通过。
- 独立特征提取的 30 次本机测量：中位 **47.98 ms**、最大 **51.59 ms**，约每秒 20 帧；不是 5 ms 控制回调耗时，也不宣称处理每张 24 Hz 输入帧。worker 只取最新帧，控制线程只读摘要；完整硬件环实时性仍待实测。

当前结论是**核心软件修正及条件理论审核通过，可以交给用户验证**。完整 5 ms 硬件环尾延迟、真实检测/修复效果和零退化力指标仍由实际实验给出，未记录为已通过。

# 本轮修正交付

最终编码按用户最新要求使用 GPT-6 medium；关键软件验收由 ultra 独立完成。没有启动机器人、重启现有服务或运行真人扫描。

## 代码

- `peirastic/contact_qp/`：几何、左右声窗任务、三物理变量 QP、同一六维命令能量行、单账本、版本化图像特征与有限时间参考。
- `peirastic/realman8dof/modes/contact_active.py`：原 4 N 名义候选到原 IK 的适配、逐设备发布事实、最终能量复核、候选提交和进度管理。原 IK/native/rail 算法保持原版本。
- baseline/shadow 记录和独立图像 worker 保留；服务能力预检防止旧 daemon 静默执行普通 TFF。初始化移出 200 Hz 回调，日志等待在停止之后。
- 旧 native 扩展、旧 ExecutionCoordinator 及专用 helper/tests 已退出主链并保存可复现快照；其他模式仍使用的共享力控与保护保留。

## Confidence 的实际结论

用户指出的右图全黑问题已确认。默认提取从旧论文公式版本修正为与公开 CAMP B-mode 代码对应的 v2，旧算法及 hash 可回放。独立稠密小图对照和 5 张真实帧复核通过，显示使用固定 0–1 色标。

这修正了实现/参考不一致和本次样本中的严重分数压缩，**没有解决全部黑区/脱离识别**。明显左暗真实样本的新左窗仍约 0.549；纯黑半幅反例仍可能约 0.878。不能把 confidence 当接触概率，也不能照搬旧仿真的 0.5 阈值。详见 [修正说明](CONFIDENCE_FIX_V3.md) 与 [对照图](confidence_v2_check/comparison.png)。

## 最终验证

- 清理后的 `peirastic/tests`：**529 passed，3 个已运行的 100,000 拍长测试未重复执行**。
- 原 admittance / Ke / guard / TDPA / observer / force 接口相关测试：**48 passed**。
- ultra：active 软件 gate PASS；confidence 参考实现修正 PASS；实机声学效果未验收。
- 旧 coordinator 独立历史快照：26 passed；移除它不是删除失败以制造通过。
- 离线 CLI 默认配置验证通过，无设备连接。当前 Python/C++ native 协议均为 8，相关 IK/native/rail 源与基线版本一致。

原 100,000 拍默认名义对照误差为零；单账本 10,000 区间验证与性能记录、以前失败的声学设计实验均保留。当前没有以合成声学模型继续调参，没有使用留出集选择新参数。

## 尚需真实输入或实验

真实孔径/安装外参/图像左右、源时基与质量阈值仍需实际声明，模板没有填入假标定。用户原被动记录入口不依赖这些外环标定；active 需配置真实接口后由用户启动。

当前生产 tracker 未提供时间对齐的校准端口标记，实测能量结算保持 `monitor_unavailable`。默认关闭命令预算；显式打开只得到命令模型准入，保留的预留可能耗尽并停止，不等于实测能量闭环或物理无源认证。

全实时环的尾延迟、真实声窗收益以及零退化力指标，需要后续实际运行测量；不把上述软件 PASS 写成这些效果已经通过。运行入口见 [REAL_CAPTURE_ENTRYPOINTS.md](REAL_CAPTURE_ENTRYPOINTS.md)，真实特性判定流程见 [REAL_STUDY_PROTOCOL_V3.md](REAL_STUDY_PROTOCOL_V3.md)。

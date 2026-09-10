# Runtime energy V3 独立审核

历史审核范围说明：后续发现了同账本未扣 D 的问题，并在 V4 修复，同时新增实际端口区间接线。以下 V3 PASS 不可用作完整耗散/物理证明；当前结论见 [FINAL_HANDOFF_V4.md](FINAL_HANDOFF_V4.md)。

结论：**PASS，范围仅为独立 RuntimeEnergy 适配模块**。未发现阻止其进入 active 集成的关键正确性问题；此结论不包含尚未交付的整条发送链，也不是物理无源认证。

审核源：

- `peirastic/contact_qp/runtime_energy.py`：`08fb252adbf04e5ed2b9cfe050a08835024a84f28af4aacd47415f375657af1e`
- `peirastic/tests/test_contact_qp_runtime_energy.py`：`99ef1169cc72c1e22e8e9808b25c887e58a5cd56b2557247dc905e2609106256`

独立重跑专用测试 **32 passed / 0.05 s**，使用 rm75 Python，未连接硬件。已有 10k trace 与相关集成测试由实现者提供，本审核未重复该长测。

核查结果：完整六维端口先求净功；真实连续样本区间只结算一次。正常重复轮询保持锚点；无效、过期、错版本、乱序或缺测均阻断区间拼接，后续测量不会释放缺口中的旧负债。实测回能仅在完成的有效区间进入同一个余额，保留负余额，容量外回能不保留。命令预测回能不增加余额。

开启命令预算时，快照使用同一 ledger 的可用额，包含源年龄；最终完整六维 payload-model 速度在发布时刻重新扩张误差，并按最新余额复核。快照单次消费；仅新命令已证未曝光才可退款，发送开始后不能用拒发理由退款。MONITOR-only 不生成能量约束行，不因能量诊断阻止原机械停止。

必须保留的结论边界：实际结算采用端点净输出功率及声明误差/变化率的估计；缺少认证物理界时始终 `certified=false`、`physical_assurance=unverified`。通过候选约束只表示 command-model energy admissible，不能据此称完整任务证书、异步臂轨执行保证或声学收敛。

active 集成仍须兑现既定接口：真实 W 的符号、单位、参考点已核对；完整实测 V 与 W 的区间有效且同域；不得用命令/预测冒充 actual；最终实际 payload 复核后、任何设备发送前调用 `publication_started`；部分发送保留事实与负债；机械停止独立于预算许可。这里不要求重启旧内环或引入新的执行子系统。

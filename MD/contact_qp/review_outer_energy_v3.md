# 纯外环共享能量条件独立审查（2026-09-10）

结论：**本次纯数学/求解/最终输出复核模块 PASS。** 本结论不包含运行适配器的账本/发送事务，不包含实际几何、执行误差或声学效果认证。未接硬件、未使用留出集、未改变内环。

范围：`port_constraint.py`、`qp.py`、最终检查依赖的 `types.py`，及对应短测试。依据附件式（30）、[ARCHITECTURE_REVIEW_V2.md](ARCHITECTURE_REVIEW_V2.md) 与 [EFFECT_IMPROVEMENT_REVIEW_V3.md](EFFECT_IMPROVEMENT_REVIEW_V3.md)。

## 数学核对

物理变量仍为 `y=[vn,ω,α]`，`V=Hy`；测量误差/耗散的绝对值仅引入代数上图量，没有新增控制自由度。单条共享能量行等价于：

```
WmᵀV − (ε+hL/2)ᵀ|V| − (d⊙rbar)ᵀ|PcV|
       − (|Wm|+ε+hL/2)ᵀη − (d⊙rbar)ᵀ(|Pc|η) ≥ −βS/h
```

完整六维净功先求和；控制正压缩力与环境作用于探头的物理 wrench 区分。扫描/计划姿态通过 `H` 一同进入预算，有限预算可以联合改变 alpha 与法向/rocking，未变成 QP 后统一缩放或整条命令拒发。

对 `d_j>0`，同一个外环显式限制 `|Pc_j V|≤rbar_j−(|Pc|η)_j`，从而保证声明的实际速度上界；范围为负是任务能量不可行，不能假发零。`d_j=0` 不因该项添加无意义速度界。由实际 `|e|≤η`、wrench 误差界得到真实前缀功下界；减去耗散上界后对前缀时间为凹二次，端点约束与初始非负额度支持条件前缀结论。

绝对值辅助变量没有目标权重，其可不唯一；物理变量的严格凸目标保留。能量充分时比较的是“其余任务完全相同、仅移除能量条件”的问题。最终按真实输出的绝对值复核，预算对象只读，求解不扣账、不预支未来回能。

## 本轮发现并关闭的阻断

1. **前缀功溢出可能制造回能。** 原 `lower_work_j` 在有限大输入下返回 `+inf`，零前缀还可返回 NaN。独立反例：`W=full(6,1e308)`、`V=ones(6)`、`h=.005`。现非有限前缀功抛出明确异常，零时长返回零；不允许作为结算信用。
2. **最终发布时间不受完整检查，发布等待未进入能量界。** 原 final guard 接受 NaN、负无穷和早于求解的时间。现保存 `created_time_s`，拒绝非有限/倒时/过期检查；以 `ε←ε+age*L` 对发布时预算复核，tracking 常量随之扩大，随后执行 hold 保持完整长度。
3. **最终线性投影溢出可掩盖机械越界。** 一个无上下限的大系数行使 `A@V` 溢出，再由 NaN 污染最大违反量，原实现错误返回零。独立反例中全分量 10 的错误最终速度被接受。现 `TwistConstraints.violation` 遇非有限投影返回无穷，最终检查拒绝。

发布时间的独立物理反例也已关闭：`t0=1`、`Wm=0`、`L_y=1 N/s`、`h=.005`、`Vy=.02`、`S=2.5e−7 J`，创建时解为 alpha=1。若到 `1.004` 才发，合法 wrench `Wy(t)=−(t−1)` 在随后 hold 内做功 `−6.5e−7 J`，不能沿用原 `−2.5e−7 J` 下界。现最终检查拒绝，aged 前缀功重算恰为 `−6.5e−7 J`。

## 故障分类

完整 6D 原机械集合只在失败路径额外做可行性检查；H、progress、相对基线和能量条件不能冒充原机械边界。此前的 Y 反向路径反例现在为 `TASK_INFEASIBLE / motion_subspace_conflict`；原 6D 机械行本身冲突仍为 `MECHANICAL_INFEASIBLE`。诊断 LP 的数值失败为 `SOLVER_FAILED`，不被解释成已证明机械集合为空。成功控制拍没有新增 LP。

## 独立验证

在规定 rm75 Python 环境运行：

```
pytest peirastic/tests/test_contact_qp_port_constraint.py \
       peirastic/tests/test_contact_qp_solver.py \
       peirastic/tests/test_contact_qp_geometry.py -q -k 'not 100000'
```

结果 **74 passed，2 deselected，0.33 s**。另外独立复现并复验上述发布时间反例；用非正交 `Pc`、逐分量有界时变执行误差与 wrench 变化，在 10,000 个前缀上数值积分实际 `WᵀV−ρ`，实际值均不低于推导下界，声明接触速度界成立。

既有原/能量两项 100,000 拍透明性证据保留，本轮修复不改变普通 QP 目标或透明路径，未重复长跑。root 的 [孤立外环性能记录](outer_energy_performance_v1.json) 保留：各 1200 次，无能量/预算充分/能量活跃 p99 约 .525/.858/.696 ms，最大≤1.374 ms，无失败或超过5 ms；这是此前相邻源码的纯 solve 测量，不是完整运行链实时认证。

## 运行适配层仍必须保证的边界

- 输入 snapshot 的 wrench_error 已包含传感器源时间到求解时间的年龄；L 与 η 的声明适用范围覆盖发布等待及随后完整 hold，不能以当前误差样本替代未来界。
- S 是当时仍可用且已妥善预留的同一 tank 额度；重复调用纯求解器不会自动预约额度，不能把同一个快照拿来多次独立发布。旧命令、拒发、部分发送和停止尾段不能靠本模块退款或回滚。
- 实际 final 检查显式传入当前时间。省略 `now_s` 仅按创建时刻作静态复核，不检查现实中的等待时长。静态可行不等于实际已发布或已执行。
- `assurance='monitor'` 在当前纯模块中是保证等级标签，**不会自动关闭传入的命令预算约束**。纯旁路监测不应把候选送到设备；若只记录而不施加能量命令约束，应不传 active energy 对象。所有 assurance 当前均 `port_verified=False`，没有用版本字符串伪造实机认证。
- 实际功只按不重叠的真实时间区间结算；非有限功、缺测、负余额和界失效明确保留。端口参考点、单位和安装几何仍需真实标定。

这些是纯模块与运行适配器的职责接口，不是新增审批阶段，也不要求合成声学效果通过后才能采集真实特性。

## 已审源码

| 文件 | SHA-256 |
|---|---|
| peirastic/contact_qp/qp.py | d155d958f5bd0d5b3bd06218f9eef3bf2c2c2deb4b68a2a6103b1830e91335c1 |
| peirastic/contact_qp/port_constraint.py | e3ccfd9348fd0380b8894e5083de50481cee3a3bf1fa969683e5fd89308f2368 |
| peirastic/contact_qp/types.py | 9b7ee4727050c9514c5a9e045611d6d0a0c7f4c42dd754fe3481a095ec7fbe9e |
| peirastic/tests/test_contact_qp_port_constraint.py | bff339065049b3db85c20fdb6d320780a8701fcd3f8f3fa78582894c52c47f3f |
| peirastic/tests/test_contact_qp_solver.py | 810089efc37775bb2bae7257487c66955a91bcf1353525cb3dd7fe8e33fe1148 |

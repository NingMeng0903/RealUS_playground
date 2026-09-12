# 独立交叉复核：mechanisms 与 solver

复核者未编写 mechanisms/solver 分析；本次不复核自己先前的 failures 审计，不修改生产代码或硬件状态。本复核是独立交叉检查，未使用其他指定推理级别的审核服务。

## 通过的检查

阅读 `mechanisms/audit.py`、`visual_detail.py`、`REPORT.md`、`solver/replay.py`、`history.py`、`numeric_variants.py`，并对照 `command_budget.py`、`qp.py`、`repair_policy.py` 与控制入口。另直接重读两份原始日志，独立编写计算，没有调用 mechanisms 分析函数：

| 原始 attempt | work 事件 | 独立余额最大误差 | 同向发布 / 有请求的成功发布 | 独立复算扫描 stale 时长 ms |
|---|---:|---:|---:|---|
| yuhan_L/RH_Per_C_DtP/001 | 17,564 | 1.39e−15 | 2,399 / 2,618 | 6.076、5.721、5.447 |
| yuhan_L/RH_Per_S_PtD/007 | 17,523 | 4.72e−16 | 2,963 / 3,216 | 8.805 |

已提交 wrench 与对应控制样本的 `−physical_wrench_candidate_tool` 逐分量完全相同。独立以提交记录的 wrench、velocity、nominal 计算内积，再按每个 work 时间区间积分、补名义任务供能、裁剪 capacity，与账本吻合。直接从原始 qR−qL 判断差分请求方向，所有有请求周期与日志 sign 一致。细节保存在 `review_selected.json`。

全量指标相加得 133,557 / 136,220 = 98.045%，四舍五入 98.05%，与报告一致。扫描期 image pause 为 5 段 / 33.129 ms；全阶段是 20 段 / 276.260 ms，其余为启动/seek missing。报告已正确区分两者。本复核对两份原始日志中的四段扫描 stale 作了独立复算，其余全量计数检查了脚本和聚合结果，没有声称重新独立遍历全部原始日志。

## 结论必须保留的限制

1. 98.05% 衡量成功发布的最终命令 wy 与请求的符号一致，不是实际角度、目标完成率或图像改善率；也不等于 QP 增量均同向。统计按控制周期计数，涉及持续时间/角度积分的指标使用相邻控制记录时间近似，不是实测机械积分。
2. nominal task source 是显式模型供能。第二份日志 port work 为 −0.125981 J，却有 +0.142938 J 名义任务供能，最终 tank 为 0.116957 J。余额上升不能全称为机械回收能量。日志 `physical_certified=false`；核账成立不构成实际端口无源性证明。
3. 浅层窗口 confidence 是算法分数。请求随分数变化和 QP 命令变化支持控制机制参与；没有对照扫描，不能证明改善由视觉产生。前后扫描位置不同也不能证明视觉使图像变坏。mechanisms 报告已明确这些限制。
4. `QP−nominal` 也受其他 QP 约束影响，且 nominal 可包含前序视觉状态。它不能作为关闭视觉后的反事实差异或视觉总贡献。

## Solver 重构证据的审核

现有 8 个重构 QP 的线性可行性检查均通过；从全新 workspace 直接解时 7 个成功、1 个仍失败。已完成的 24 组历史回放显示：current 在 case 0 和 6 复现失败；fresh 仍失败 case 6；update_preconditioner 仍失败 case 6，且在 case 3 引入另一个终末失败。因此证据支持 solver 数值/工作区状态是重要排查方向，**不支持“重新初始化或更新预条件器已修好全部失败”**。

重构并非逐位还原：拒绝记录缺少该次完整 wrench / 能量快照，使用前一控制样本近似；成功周期使用零测量角度，拒绝周期才有实际记录角度；缺失中心窗口 quality 用 .8 补齐；tilt 参数由当前离线 law 构建；回放默认 repair execution enabled，实际 seek 受接触 gate 控制。available 取日志时点也有小的 reservation 差异。这些差异会改变约束或 solver 历史。重构的 LP 可行性只证明重构问题可行，不能作为原始在线矩阵全部可行的严格证明。

case 0 的历史回放复现 19,303 次迭代且此前样本无失败，比单个矩阵新建 solver 成功更有解释力；但最终故障诊断仍应表述为有较强数值证据的工作区/条件性问题，并为精确问题增加矩阵/solver 状态记录与离线回归。solver 作者已获知上述限制并确认会在报告保留。

## 复核结论

mechanisms 的账本计算和所抽查的控制方向、暂停统计通过交叉检查；当前报告的模型/物理、参与/效果限制充分。solver 的近似重构可帮助缩小问题范围，不能作为硬件修复已验证或所有失败原因已唯一定位的声明。

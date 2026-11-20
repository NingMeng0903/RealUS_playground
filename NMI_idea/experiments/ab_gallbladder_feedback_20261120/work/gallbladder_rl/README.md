# 胆囊：公开后验模型策略与循环 RL 对照

基于提交 `101325037dcb9c899dd4553245f5aa505a9caab9` 的新增实验入口。旧实验源码、旧模型配置与 `reference_results/` 保持不变。

**状态：已完成存档审计、实现和无解剖资产的程序检查；尚未完成新的胆囊 RL 训练或成功率评估。** 本次环境没有原实验要求的三份 NPZ。`results/archive_audit.json` 是旧结果复算；`interface_checks.json` 和 `ppo_smoke.json` 是合成输入下的代码检查，均不能作为新胆囊实验结果。

## 1. 旧结果能支持什么结论

| 留出数据 | 原固定站点 | 手写在途反馈 | 学习反馈 | PSDP 门控 |
|---|---:|---:|---:|---:|
| 主留出，40 解剖 × 2 噪声 | 45/80 | 54/80 | 48/80 | 49/80 |
| 独立复制，32 解剖 × 2 噪声 | 42/64 | 45/64 | 完整回报替换 45/64 | 未在该批重新比较 |

对应主留出的错误交付为 5、2、8、8 次。复制实验手写反馈错误交付为 6 次，固定站点为 5 次。脚本重新检查 CSV 与 JSON 的成功/误交计数、唯一配对键和完整配对网格，并按解剖病例聚类重采样。

源码限定了结论范围：

- `../gallbladder_feedback/engine.py::run_station` 只有最近六帧含合格的公开可见证据才调用局部选动作。全空观测时，外层模型仍可走站点，但局部学习器没有探索决策权。
- `policy.py` 只有 `continue / keep_view / center / fan_left / fan_right / recall` 六类候选；局部目标来自近期可见锚点。它不等于覆盖整个接触域的探索策略。
- `train.py` 拟合 ExtraTrees 局部结果排序器；`evaluate.py` 的 PSDP 只学习首站是否启用局部反馈，后段已冻结。
- `residual.py` 只替换首个手写改道点。校准选中不替换，说明这批数据没有证明该替换器有利，不能推导出循环 RL 没有价值。
- 旧读图遮蔽比较仍共享由新图产生的候选和触发条件，因此不是完全禁止新图的对照。

主留出手写策略的 26 次失败，存档汇总分解为 17 次从未有严格帧、5 次只在确认窗外有严格帧、4 次确认窗内有严格帧但未正确交付。这些 E/S 计数来自汇总 JSON，CSV 没有逐帧数据，审计不声称重新验证每帧。

## 2. 维护的策略和公平边界

| 名称 / `--arms` | 决策输入与行为 | 用途 |
|---|---|---|
| `legacy_hand` | 原连续解剖拟合与手写在途反馈，原执行接口 | 保留已有强基线；不可把它与新接口之差全归因于学习 |
| A0 / `model` | 公开先验后验、连续拟合、图像居中建议 | 在新公共执行器上的模型对照；不是解剖真值教师 |
| B / `rl` | 公开观测历史、位姿速度、后验、剩余预算 → GRU → 动作 | 学习闭环探索和停稳确认；不输入规划器建议 |
| A+B / `hybrid` | B 的全部输入，加 A0 的公开目标方向 | 检查模型建议能否降低学习难度；须单独训练 |
| `hybrid_frozen` | 同一个 A+B 检查点，但从共同出口冻结离散肝脏/胆囊后验 | 整环境后验冻结消融；新图和连续拟合仍更新 |

A0、B、A+B 共用短程动作尺度、速度/加速度/接触域检查、真实运动预算、刹停尾段和原确认器。确认器不以真值决定交付，真值只用于训练回报、训练用价值函数和离线结果评价。

`hybrid_frozen` 同时会影响模型建议、估长和确认；它衡量这个整体系统中离散后验更新的贡献，不能解释为仅隔离 actor 输入里的某几个数。原手写基线的执行接口不同，因此新评估同时保留它和共享执行器 A0。

## 3. 实际观测、动作和目标

### 公开观测

当前模拟器输出带噪分割读数与颈/体/底标记，**没有 B-mode 像素**。每个动作检查点的 507 维输入包括：

- 当前表面坐标位姿、实测速度、时间预算。
- 最近八帧的检测读数、标记、缺失掩码，以及每帧自己的实际采图位姿与时间。
- 64 个胆囊假设和 125 个肝脏平移假设的公开后验权重。
- 上一步动作、执行拒绝标志；A+B 另带模型方向及其有效标志。

`qualified`、`genuine`、真实器官坐标、噪声类别和病例编号不直接进入 actor 特征。已知噪声配置仍由原仿真后验更新器用于计算似然。GRU 在动作检查点更新，默认每三帧作一次移动决策；每个物理帧都执行原观测与后验更新。确认动作是一段不可在中间改道的驻留选项，GRU 不逐帧读取其全部中间帧，后验与确认器会读取它们。

### 动作与执行

策略输出两类动作：

1. `move`：四维有界增量，尺度为表面高度 12 mm、表面方位 5°、旋转 12°、扇扫 8°。动作允许在全空检测时发出。
2. `settle_confirm`：从实测位置和速度刹停，再调用原公开连续帧确认器。

默认新执行器采用从实测位置/速度连接的五次多项式，不要求 RL 输出 ProDMP 权重。规划只接纳能在预算内结束并保留确认时间的轨迹。默认仅执行前三帧便再次决策；新提议不可行时继续上一个已验证计划的停止尾段。它沿用原速度、加速度和接触域检查，不声称有 jerk 证书或真实接触力验证。

### 训练目标

设公开动作历史为 \(H_t\)，更新后的公开后验为 \(b_t\)，循环状态为 \(h_t\)。

\[
h_t=f_\theta(h_{t-1},o_t,b_t),\qquad
a_t\sim\pi_\theta(\cdot\mid h_t).
\]

默认有限物理预算内的目标为：

\[
J(\theta)=\mathbb E_\theta
\left[\mathbf 1\{\text{正确停稳交付}\}
-0.8\,\mathbf 1\{\text{错误交付}\}\right].
\]

超时记零；途中看见真值合格帧不自动加分。默认 `gamma=1`，避免按决策次数折扣偏好短的确认选项。默认不加路程和时间惩罚，评估单独报告它们；这与旧加权效用不同，是显式实验选择。若将 gamma 改成小于 1，现实现按决策步折扣，不是按物理秒折扣。

训练用价值函数同时读取公开循环状态和 17 维真实解剖特征：

\[
V_\phi\bigl(\operatorname{stopgrad}(h_t),z_t^{\mathrm{train}}\bigr).
\]

真值没有送入 actor，价值损失也不会通过循环特征反传。训练保存完整 episode 的公开序列，在每次 PPO 更新时重算前缀循环状态，MOVE 的概率包含类别项和带 tanh 变换的连续项；确认动作仅包含类别项。采用 PPO、近似价值函数与 GAE，并不声称继承某篇论文的严格无偏定理。

## 4. 在线造数与训练课程

旧 CSV 不作为新 RL 的状态转移数据。新数据来自策略在仿真里实际动作、重新取像、更新后验直至终局。

`--curriculum near_success` 仅用于训练：在新的训练解剖上离线寻找见证位姿，再实际执行接近、偏过、近失配三种路径，保留真正产生的历史和运动速度。没有见证或可行路径时明确记为回退到共同初扫出口，不删除病例。actor 不会收到见证坐标。

随后 `--curriculum prefix --initial-checkpoint ...` 从正常共同肋缘初扫出口继续训练。前一阶段及所有祖先的训练种子范围写入 checkpoint；评估拒绝与其中任何范围重叠的解剖。两种噪声用独立流重复采样，不能只保留一次噪声下的赢家。这里实现的是两阶段、三类近终点起点的课程，不是自动扩展起点分布的完整 Reverse Curriculum 算法。

**本版没有重新训练共同肋缘初扫本身。** `prefix` 测试从原初扫出口开始，允许该处无检测时主动探查；“第一次放上去就在骨头上”的完整冷启动必须另设起点和预算。这两类成功率不能混用。

## 5. 运行

以下命令从实验根目录执行，即 `NMI_idea/experiments/ab_gallbladder_feedback_20261120`。使用完整仓库，因为原仿真会引用其他实验的运动基元和表面图册代码。

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

路径沿用原 README。需配置原始文件，不能用随机 NPZ 替代：

```bash
export REALUS_GB_ANATOMY=/absolute/path/registered_torso.npz
export REALUS_GB_SMPLX=/absolute/path/SMPLX_MALE.npz
export REALUS_GB_ATLAS=/absolute/path/thorax_atlas.npz
python -m work.gallbladder_rl.env --preflight
```

只审计旧结果和检查代码接口，不需要这三份资产：

```bash
python -m work.gallbladder_rl.audit_archive
python -m work.gallbladder_rl.test_contracts
python -m work.gallbladder_rl.test_ppo
```

下面是一个**待运行的训练配置示例，不是已经验证足够的数据量**。运行量应根据独立验证集学习曲线调整，最终留出集不能用于调参。默认 trainer 的 32 episode 仅用于接口试跑，不能据此判断 RL 是否学会任务。

```bash
python -m work.gallbladder_rl.train_rl \
  --curriculum near_success --episodes 1024 --count 256 \
  --seed-start 58710000 --condition mixed --stream 20000 \
  --output work/gallbladder_rl/runs/b_near

python -m work.gallbladder_rl.train_rl \
  --curriculum prefix --episodes 4096 --count 512 \
  --seed-start 58711000 --condition mixed --stream 30000 \
  --initial-checkpoint work/gallbladder_rl/runs/b_near/checkpoint.pt \
  --output work/gallbladder_rl/runs/b_prefix
```

A+B 需在上述两个阶段均添加 `--planner-features`，使用新的输出目录，并让第二阶段初始化相应 A+B 检查点。不能把 B 检查点直接当 A+B，加载器会拒绝不同输入配置。

先用独立验证集（例如 `58718000` 开始）选择训练量和超参数。冻结之后的正式配对示例：

```bash
python -m work.gallbladder_rl.evaluate_rl \
  --arms legacy_hand model rl \
  --policy-checkpoint work/gallbladder_rl/runs/b_prefix/checkpoint.pt \
  --seed-start 58720000 --count 40 --stream 50000 \
  --output work/gallbladder_rl/runs/holdout_b
```

若已有独立训练的 A+B，添加 `--arms legacy_hand model rl hybrid hybrid_frozen` 和 `--hybrid-checkpoint <A+B检查点>`。每个解剖均运行 wide/stress，两档噪声在各策略间使用相同时间索引噪声流；每个策略独立重置世界。评估使用确定性 actor，绝不向 actor 传递 critic 特权向量。

输出包括每例动作/终局日志、配对 CSV、按解剖聚类的置信区间、训练谱系、源码与资产哈希。输出目录必须是新的，避免覆盖既有实验。

## 6. 未解决的建模问题与下一道检验

原显式后验只有器官/肝脏假设，没有联合肋位、声耦合和连续掉线隐变量；双器官都无检测时基本不更新。新 GRU 可以接收这段空观测和已走过的位置，是否学会避开坏窗必须实测，不能称为已经修好了遮挡后验。

本版保留肋缘初扫和同帧同腔/居中/长轴的严格终局评价，但局部居中与转轴可以交错。它没有显式的技能顺序自动机，不宣称已学出要求的全部流程顺序。若必须硬保证顺序，应另加带恢复边的阶段约束，并与无该约束共享动作权限做比较。

正式验收至少同时报告正确交付、误交、从未严格命中、确认窗外命中、确认窗内命中未交付、耗时和路程。新执行器的 timeout padding 也会成像，因此“确认窗外命中”包含等待和运动，不能全部称为“途中越过”。至少用多个训练随机种子检查方差；单个 PPO 检查点没有超过 A0 不足以判定这个策略类无效。

本版没有重训 PSDP。先检验 B/A+B 是否在无检测状态获得可重复的终局收益，再把冻结 B 当作尾策略做同出口 PSDP 训练，才有新的 T-RO/IJRR 对照对象。到那一步应额外比较同出口新图、冻结或遮蔽新图的策略，确保候选生成也没有偷偷共享新图。

## 7. 方法依据

- [Schulman et al., Proximal Policy Optimization Algorithms, 2017](https://arxiv.org/abs/1707.06347)：在线交互后优化策略；不能由此保证本任务成功率。
- [Florensa et al., Reverse Curriculum Generation for Reinforcement Learning, CoRL 2017](https://proceedings.mlr.press/v78/florensa17a.html)：从目标邻域逐渐扩展训练起点，缓解稀疏成功信号。本版借鉴起点课程思路，未完整复现自动课程生成。
- [Baisero & Amato, Unbiased Asymmetric Reinforcement Learning under Partial Observability, AAMAS 2022](https://arxiv.org/abs/2105.11674)：部分可观测时仅用真实状态构造 critic 存在理论问题，应考虑历史与状态。本文支持分开公开策略和训练特权信息的设计，不等于证明此近似 PPO 实现无偏。

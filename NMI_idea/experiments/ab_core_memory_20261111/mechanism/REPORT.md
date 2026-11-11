# Exact adaptive-horizon / memory mechanism

这是一个可复算的符号小模型，不是实际声学数据、图像、URDF 或机器人执行结果。`run_exact.py` 穷举四个隐藏状态 `(S,R)` 和每个二值声窗分支；没有 Monte-Carlo 估计，也没有读取旧 patch/旧结果。

运行：

```text
python run_exact.py
python -m unittest -v test_exact.py
```

两条命令分别重写同目录的 `RESULTS.json` 并运行 6 个小测试。

## 公共契约

`S,R` 独立均匀取 `{-1,+1}`；几何公开摘要对 L/R 对称。两个合法短探查各耗时 1、无直接收益、使用同一 `shared_symbolic_image_v1`：

- `probe_R` 观测 `R`；
- `probe_P` 观测 `P=S*R`；
- 每个声窗独立翻转，正确率 `a=0.96`。

终端 `L/R` 各耗时 3，只在选择的侧等于 `S` 时成功；`fallback` 也耗时 3，独立成功率 `.67`；预算为 5，最多一次终端。所有方法看到同一动作目录、图像标识、耗时、合法动作和 deadline。`h` 明确表示宏决策深度（终端算一个决策）；物理成本仍由 budget 强制。

后验使用

```text
P(s,r | D) ∝ 1/4 × ∏_i a^[y_i=t_i] (1-a)^[y_i≠t_i]
```

其中 `t_i` 是 `R` 或 `S*R`。脚本先做动态规划，再独立按隐藏状态和观测分支重放策略树，结果逐项比对。

## 主结果

| h | adaptive full Bayes | adaptive 忽略诊断 | fixed open-loop 选中 |
|---:|---:|---:|---:|
| 1 | `.6700`, 直接 fallback, time 3 | `.6700` | `n=0`, `.6700`, time 3 |
| 2 | `.6700`, 直接 fallback, time 3 | `.6700` | `n=0`, `.6700`, time 3 |
| 3 | `.9232`, `R→P→terminal`, time 5 | `.6700` | `n=2`, `.9232`, time 5 |
| 4 | `.9232`, `R→P→terminal`, time 5 | `.6700` | `n=2`, `.9232`, time 5 |

固定窗口逐项为：`n=0: .67/time3`、`n=1: .67/time4`、`n=2: .9232/time5`。`n=1` 不识别 `S`，且在与 `n=0` 同值时按共同 deadline 下的最短时间择优。忽略诊断的方法把两个声窗当作 `.5` 噪声，因而不付探查时间，直接选 `.67` fallback。

关键恒等式是

```text
P(ŷ_S=S | 两探针) = a² + (1-a)²
                  = .96² + .04² = .9232
```

单独看 `R` 或 `S*R` 时，另一个独立均匀符号把 `S` 边缘化掉，所以 `P(S=+1 | 一条声窗)=.5`。在有 `.67` fallback、探针准确率 `a≥.5` 时，两探针超过 fallback 的确定边界为

```text
a > 0.5 + sqrt((.67-.5)/2) = 0.7915475947...
```

因此 `.96` 与 `.67` 有 `.2532` 的明确分离；这不是调场景数值得到的胜利，而是该已给定二探针模型的公式边界。

纯几何 A 只读取几何等价公开摘要，主场景同样是 `.67` fallback。强 A/完整历史规划与 B 使用同一后验、动作、图像、时间和 deadline；在该机制层，它们的 `.9232` 是同模型公平上限/结果，不应解读成真实 A/B 系统差异。

## 固定前缀与记忆

前缀是 `probe_R (1)` 后三个强制、独立且不可辨识的中间步（各 `1`）；到目标局部只允许 `probe_P` 和终端，禁止旧 patch 重访。为了使 `R + 3 个中间步 + P + 终端` 合法，固定前缀协议采用其最小共同 deadline `8`；所有方法先付完全相同的前缀成本 `4`，公开机器人出口/time 相同。

| memory policy | 目标局部动作 | 保留状态（到 P 后） | exact success | elapsed |
|---|---|---|---:|---:|
| full history | `probe_P` | `R,middle1,middle2,middle3,P` | `.9232` | 8 |
| replay | `probe_P` | 与 full history 相同 | `.9232` | 8 |
| address cache | `probe_P` | 与 full history 相同 | `.9232` | 8 |
| window1 | 直接 fallback | 只有 `P`（丢旧 `R`） | `.6700` | 7 |
| window2 | 直接 fallback | `middle3,P`（丢旧 `R`） | `.6700` | 7 |
| ignore diagnostic | 直接 fallback | 诊断不进后验 | `.6700` | 7 |
| pure geometry A | 直接 fallback | 只读几何等价摘要 | `.6700` | 7 |

`replay` 与 `address_cache` 行在脚本中显式断言完全相同；window1/2 没有旧 patch 重访问预算，故丢掉 `R` 后不能补救。这个分离只说明持久观测状态的差异，不把合成声学 fixture 当作真实声学证据。

## 反例 / 边界

- 正确模型但 `a=.5`：h3 adaptive 为 `.67`，直接 fallback；没有信息。
- 已知几何足够：公开正确的 `S` 标签时，直接匹配终端为 `1.0/time3`，不需要探查。
- epoch reset：当前 `R_new` 与缓存的旧 `R` 独立；错误的 `old_R × current_P` 只有 `.5`，而当前 `R_new + P` 仍为 `.9232`。
- 模型失配：规划器假设 `.96`，实际声窗正确率 `.55`；它仍选择两探针，但 exact 成功率为 `.55²+.45²=.505 < .67`。这暴露了校准边界，而非奖励调参。
- 最小单探针负对照：预算 3，probe 耗 1，唯一可执行终端是 `L` 耗 2，无 fallback；adaptive 和固定一探针都为 `.5`，且任何 A/B 策略上限都是 `.5`。

全部数字、策略树、公共动作目录和分支校验均在 `RESULTS.json`；测试只检查该模型的公式、预算/单终端约束、记忆分离和反例。


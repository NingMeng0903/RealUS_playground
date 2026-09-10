# V8r2：主动倾斜、4–4.5 N 修复策略与有限尝试

交付配置：`peirastic/config/contact_qp/active_probe50_v8.yaml`，修订 `v8r2_bounded_episode`。代码由 medium 实现，最终限定审核见 [V8_FINAL_REVIEW.md](V8_FINAL_REVIEW.md)。开发、回放和仿真没有连接或启动机器人。

本版按用户的新要求改变原零退化策略：名义力控目标仍为 4 N，修复可以在 4–4.5 N 区间参与；不再使用超过 4.1 N 就禁止新增加载的旧端点力方向行。原零退化统计验收不再能代表这个新策略，必须重新报告力误差与图像收益的权衡。

## 分配公式与边界

物理变量仍只有 `y=[vn, omega, alpha]`，完整工具速度 `V=Hy`。左右局部法向运动行来自同一刚体几何，窗口欠缺为 `di=max(0,(cmin-ci)/cmin)`。定义：

\[
g_F=\operatorname{clip}((4.5-F)/0.5,0,1),\quad
s=\operatorname{sgn}(d_L-d_R),\quad q_D=k_{rep}g_F|d_L-d_R|.
\]

差分任务使用总 rocking 对局部速度差的贡献 `drock`，排除路径前馈的贡献。在当前探头标定下 `drock=-0.031*omega`（米/秒）；共同法向速度和 α 都不能替代这个差分。

\[
s\,d_{rock}+v_s\eta_D\ge q_D,\qquad \eta_D\ge0,
\]
\[
J_D=\tfrac12w_D\eta_D^2+
\tfrac12w_\alpha(\alpha-\alpha_{pref}+k_\eta\eta_D)^2.
\]

这里 `etaD` 是无量纲松弛量，`vs=0.002 m/s`。独立的第一项保证即使 α 降至零，倾斜不足仍有代价；第二项让尚未完成的倾斜请求影响推进偏好。请求为零时移除耦合项，恢复原 α 目标，避免制造虚假修复松弛量。已有足够且方向正确的名义倾斜不会被重复要求增加。两側同样欠缺时不凭空选择一个倾斜方向。

原左右修复/保持、有限孔径扰动预算与代价、机械速度/加速度/角度条件继续参与。新增端点正加载软代价随 F 接近 4.5 N 增大。QP 仍是凸二次目标和线性约束，图像松弛不能放宽机械或能量准入。

4.5 N 是**测量值驱动的软策略**：达到该值后关闭声学任务，使名义力控恢复动作不被视觉任务阻碍。6 N 是**有效最新 raw/filtered 原控制力标量任一达到阈值时的监督停止**。二者都不构成未来连续力峰值的数学上界；真实刚度、测量误差、执行延迟和停止暴露未经定界，不能宣称绝不超过 6 N。

## 持续缺失时如何退出

初版差分速度任务会对持续声影不断请求倾斜。在冻结的 701/702 仿真中，两次都到实际角度边界中止，声学缺口没有减少；[初版报告与失败记录](../../analysis_artifacts/contact_qp_v8/REPORT.md)完整保留。

V8r2 为同一缺失 episode 增加有限尝试：累计准许修复时间 2 秒，或 episode 中实测 rocking 角度总行程 3°，任一耗尽就停发该 episode 的差分和欠缺窗口修复任务。良好窗口保持、名义 4 N 力控、原始图像损失、α 推进及缺口记录继续工作。

时间计的是准许修复的过去实际间隔，不冒充成功执行时间；实测角度变化独立于命令事务累计，不能按 candidate−nominal 积分，因为名义状态会吸收已提交的外环命令。拒发、图像失效、新帧、窗口版本变化都不补额。只有连续三个不同且有效、双侧均达到阈值的图像才重新允许完整尝试；寻触前不启动尝试。原角度参考点重设只重新对齐测量，不能退回已用额度。

这是有限尝试策略，不是第二个机械能量罐，也不是声学不可修复证明。已在执行的命令尾段和名义控制仍可能继续运动，3°不是实际运动的硬边界；原一拍角度约束也没有因此取得延迟制动保证。

## 与单一能量罐的关系

新分配继续使用原完整六维 `Hy` 能量行，法向、rocking、扫描以及其余路径分量共同竞争同一可用能量；没有另设“修复能量来源”。在已经扣除停止预留和未结算暴露的额度 `Ea` 下，命令模型要求：

\[
P_{lb}(Hy)+\beta E_a/h\ge0,\qquad0<\beta\le1.
\]

`Plb` 包含统一 TCP/工具坐标下的环境 wrench、声明的 wrench/跟踪误差及阻尼项。在这些界成立、最终命令仍满足证书、声明持有时长内执行的前提下，`lower_work(V,tau) >= tau*Plb(V)`，故对所有 `0<=tau<=h`，`Ea+lower_work >= Ea*(1-beta*tau/h)>=0`。改变凸目标中的修复与 α 权重不会改变这个准入条件。能量准入启用时，最终命令仍经原发送前能量复核及预留/结算链处理。

这只说明声明模型内的全端口前缀能量性质，不能推出力峰值、声学收敛或未定界的臂轨异步执行无源性。独立测试覆盖六维非零 wrench、非零路径角速度、零/少量/充足额度；零额度确实抑制了耗能方向的倾斜，而非仅重复检查求解器自己的成功标志。

**当前真机配置仍为 `energy_constraint_enabled:false`、`physical_w_checked:false`。**真实六维物理端口语义尚未核对，因此未自动开启能量准入，也不能把下面的实际执行称为能量罐已经保护的试验。`--validate-only` 明确显示这一事实。

## 真实数据回放的解释

[固定五状态回放](../../analysis_artifacts/contact_qp_v8_r2/real_five_state_replay.json)使用 active/003 原始力、图像和名义命令。V7 解首先复现原保存命令。每个状态独立启动求解器，**假设该时刻有完整尝试额度**；旧日志不含精确实测角度，取远离原角限的零角度。因此这些是单状态分配比较，不是 V8r2 对整条真人轨迹的闭环预测。

| 原始状态 | F / N | V7 总 rocking / °/s | V8r2 总 rocking / °/s | 解释 |
|---|---:|---:|---:|---|
| L DtP，1452 | 4.102 | -0.339 | -0.481 | 原力方向行阻挡的修复现在可参与 |
| L DtP，1528 | 3.851 | -1.759 | -1.759 | V7 本拍解已满足差分，新任务未进一步改变它 |
| L PtD，5641 | 3.640 | -0.842 | -1.345 | 共同加载不再替代左右差分修复 |
| L PtD，5791 | 3.979 | -1.766 | -1.766 | V7 本拍解已满足差分，新任务未进一步改变它 |
| L PtD，7288 | 3.891 | 0 | 0 | 检测双侧已达阈值，不触发修复 |

这些指令没有执行，不证明图像已改善。当前 confidence 算法、浅层窗口、0.8 阈值及版本保持不变；之前可见深部/边缘黑区但浅层质量仍高的漏检也不会由本次分配修改自动解决。

## 用户执行

先在原 controller 终端结束旧进程，再由用户启动一次；不要同时运行两个 controller。新配置要求 `contact_qp.differential_repair_v8_r2` 能力，旧服务会在发扫描命令前被拒绝。

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh controller
```

confidence 工作进程可继续使用当前匹配版本；本版没有更换图像发布、crop 或 hflip。在另一个扫描终端执行：

```bash
cd /media/camp/EXT_DRIVE/ICRA_YM/script
bash run.sh check
bash run.sh record --force-profile icra --speed-m-s 0.005 --keep-raw \
  --contact-qp-config /media/camp/EXT_DRIVE/RealUS_playground/peirastic/config/contact_qp/active_probe50_v8.yaml \
  --data-root '/media/camp/EXT_DRIVE/ICRA_2027/icra 2027_contact/real_characterization/active_probe50_v8r2'
```

按原示教流程，首轮完成一条路径后在下一条提示输入 `q`，保留整个 session 和失败 attempts。比较实际倾斜、图像变化、力 RMSE/峰值/>4.5 N 曝光及完成状态；新日志 `repair_episode` 可区分力门控、额度耗尽与未寻触，不能只看 α 判断是否修复。

## 移植与验证产物

- `peirastic/contact_qp/repair_policy.py`：差分代价、力软门控与参数修订。
- `peirastic/contact_qp/repair_episode.py`：仅依赖有效观测、时间和实测角度的有限尝试状态。
- `peirastic/contact_qp/qp.py`：三物理变量、窗口行选择、原完整端口约束与诊断。
- `peirastic/realman8dof/modes/contact_active.py`：寻触/姿态参考接线、6 N 监督和日志；原名义核心、IK 与发送事务继续复用。
- [最终固定实验及能量检查](../../analysis_artifacts/contact_qp_v8_r2/REPORT.md)：新种子 801/802 的结果；旧 701/702 声影只作失败回归，单独保存。仿真中已经接触良好的左右曲面场景不能充当有欠缺时的修复效果证据。

实际执行/声学收益仍须由新录制验证。源码冻结包、配置、指标、测试记录及最终审核与本文件共同构成软件交付，不能替代硬件闭环验收。

## 最终验证结果

最后一次生产修改后，外环短回归 **602 passed**；一项 native 测试在受限沙箱启动失败，沙箱外相同代码离线复查 **1 passed**，两份记录均保留。原力控八组回归 **46 passed**。ultra 最终独立定向 **48 passed**，与总回归有重叠，不累计成独立样本数。三个已完成过的 100,000 周期长测试未在此次短回归重复；能量长序列在第一版针对回归中已通过，本轮没有改其账本实现。

新固定种子 801/802 共 20 个仿真单元全部完成：两策略各 8 次正常完成、2 次声影带缺口完成，无中止。最大采样力 4.18808 N，采样到的 >4.5 N 暴露为零。控制块 p99 为 1.99–2.75 ms，合计仍有 8 次 >5 ms；这是并行离线执行的局部耗时，不包含图像处理和真实 IK/设备执行，不能宣布整体 200 Hz 真机实时性已验收。

声影场景 V8r2 相对 V7 平均 RMSE 减少 0.04181 N，但峰值增加 0.03798 N；缺口均为 60 mm，**没有声学改善**。V8r2 含执行尾段的实际角行程约 3.14°。旧 701/702 声影四个独立回归运行全部带缺口完成，其中 V8r2 关闭了初版持续倾斜至中止的失败。健康/左右曲面/延迟场景原本已接触良好，两策略表现相同。

144 个静态全六维能量快照均通过准入和最坏前缀检查；另有左右两方向零额度相对充足额度的耗能倾斜抑制断言。该测试未将静态快照包装成真实能量结算。

主仿真期间仅真实 `contact_active.py` 的当前接触许可补丁发生变化，该适配器未参与 `run_case`；核心 QP、episode、名义、plant 和特征源码均未变。旧种子回归使用包含最后适配器的冻结包，前后无源码变化。最终单元回归和 ultra 审核针对最后适配器执行。

测试记录：[外环](../../analysis_artifacts/contact_qp_v8_r2/peirastic_final.xml)、[native 离线复查](../../analysis_artifacts/contact_qp_v8_r2/native_final.xml)、[原力控](../../analysis_artifacts/contact_qp_v8_r2/force_regression.xml)、[配置离线校验](../../analysis_artifacts/contact_qp_v8_r2/config_validation.json)。冻结实验源码见 `analysis_artifacts/contact_qp_v8_r2/regression_frozen_source.zip`；本轮修改与说明的增量包见同目录 `release_source.zip`，文件身份见 `release_manifest.json`。

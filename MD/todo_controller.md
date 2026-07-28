当前结论：问题不是单纯“阻尼没调好”，也不只是力补偿标定。已经修掉了接触状态、1 N 特殊退化和快速回撤延迟三个结构问题；但新的带延迟软硬切换仿真证明，6 Hz 力滤波 + 约 15 ms 运动延迟 仍会在硬表面产生明显相位滞后。这个剩余问题无法仅靠继续增加质量/阻尼解决。
所以目前版本还不是最终可上人体的版本。现有修改已经默认启用，但下一步的“带限相位提前 + 更平滑的主动参考”尚未完成。
一、目前确认的根因
1. 1 N 失联仍显示接触，主要不是标定问题
旧控制器只有一个 enter-only latch：
第一次 Fz > 0.8 N
→ 永久认为在接触
→ 直到控制器 reset
它同时承担两个互相冲突的作用：
恒力任务必须一直运行；
刚度估计器需要知道探头是否真的还压着表面。
所以探头真实失联后，扫描和恒力任务确实没有退出，但刚度估计器也错误地认为仍在接触，无法在下一次撞回人体时重新执行 stiff-first。
对 112604 的 1 N 日志重放后，新接触状态机识别到：
10 次确认失联；
9 次重新接触；
失联区间的补偿力约 0.19 N；
独立自由空间约 0.15 N。
这说明当时确实有真实卸载/失联。旧控制器显示“接触”主要是软件 latch 的含义错误，不是 0.15 N 偏置把它误判成 1 N 接触。
但是，如果未来空载补偿残差漂到 >0.35 N，标定误差仍可能阻止失联检测，因此标定依然重要。
2. 1 N 会额外触发错误的低刚度衰减
旧逻辑中，1 N 失联后：
Fdes ≈ 1 N
Fmeas ≈ 0.15 N
|Ferror| ≈ 0.85 N
这小于旧刚度估计的 1.2 N idle gate，于是控制器把“探头已经离开人体”错误理解成“稳定压在很软的组织上”，将估计刚度向 300 N/m 衰减。
5 N 失联时误差约 4.85 N，不会走相同分支。因此原来确实存在目标力相关的不一致，不需要专门写一个 if desired == 1N，需要修的是物理状态含义。
3. 快速回推时，6 Hz 力滞后让控制器多退了几十毫秒
112604 中：
raw force 的下降越过目标力，平均领先 filtered force 约 59 ms；
控制器仍根据落后的 filtered force 认为“过力”；
负向主动参考 v_r 继续回撤；
真实力已经掉到接近零，TCP 仍在退；
形成 gap；
随后高速重新撞上，raw 峰值约 8.49 N；
出现约 4.5–5.1 Hz、持续约 2 秒的弹跳；
Dimeas 最后已经达到 Is≈6.07、虚拟质量 5 kg，但动作已经发生。
因此问题不是 Dimeas 完全没工作，而是它属于振荡发生后的检测和增稳，无法撤销前面由滤波延迟造成的错误运动。
4. 单纯增加阻尼/质量不能同时保证跟手和稳定
新的严格仿真加入了：
200 Hz 控制；
二阶 6 Hz Butterworth；
15 ms 命令到实际 TCP 速度延迟；
单边接触和真实 separation gap；
300 → 2500 → 300 N/m 力连续刚度切换；
1 N/5 N；
表面 ±5/±10 mm/s 移动。
当前基线在硬段的代表结果：
场景	硬段 MAE	峰值	4–12 Hz 持续振荡
1 N / −10 mm/s	1.519 N	5.125 N	0.99 s
1 N / +10 mm/s	0.984 N	2.540 N	0.35 s
5 N / −10 mm/s	3.778 N	13.237 N	0.32 s
5 N / +10 mm/s	1.452 N	8.827 N	0.25 s

消融结果很明确：
5 N / −10 mm/s：

当前 6 Hz + 15 ms： MAE 3.778 N，峰值 13.24 N
去掉 6 Hz 滞后：    MAE 0.294 N，峰值 5.37 N
只去掉 15 ms：      MAE 1.666 N，峰值 8.73 N
直接给正确 Ke=2500： MAE 1.626 N，峰值 8.18 N
即使刚度估计完全正确，只要 6 Hz 相位滞后还在，也不能根治。
继续加阻尼、加质量可以降低部分峰值，但会重新造成你之前感觉到的“回推阻尼大、不跟手”，并且仍修不好 1 N 硬表面的真实失联。
二、已经完成的修改
1. 将“恒力任务”与“真实物理接触”分开
新增物理接触状态：
FREE
CONTACT
SUSPECT_LOSS
LOST
默认参数：
enter_n: 0.80
hard_enter_n: 1.50
exit_n: 0.35
enter_confirm_s: 0.010
exit_confirm_s: 0.100
作用：
恒力任务一旦开始，仍然永久保持到 controller reset；
失联不会结束扫描、不会重新启动目标力 ramp；
filtered force 连续低于 0.35 N 100 ms 才确认失联；
50 ms 的振荡低谷只进入 SUSPECT_LOSS，不会反复 stiff-first；
重新接触优先看 raw force，普通接触确认 10 ms；
明确的大冲击 >1.5 N 可立即认为重新接触。
因此这些阈值不会让任务“done”，只用于刚度估计器的内部重武装。
2. 修复 stiff-first 重新接触保护
确认失联后，刚度估计器会真正看到：
in_contact = false
重新接触时：
Ke ≥ 1500 N/m
并且有效阻尼的向上跳变现在同一个控制周期生效，不再被外层 0.1 s 阻尼平滑吃掉。
同时保留：
恒力任务 latch；
目标力 ramp 进度；
被动导纳速度 v_force_z；
Dimeas 状态。
只清除重新接触前已经过时的主动参考 v_r。
3. 禁止 1 N 失联时错误衰减 Ke
给刚度估计器增加了 allow_idle_decay。
只有满足以下条件才允许把刚度向软表面衰减：
physical state == CONTACT
并且实测承载力 >= 0.8 N
所以 1/2/5 N 现在使用完全相同的物理接触状态序列，不再因为目标力不同而走不同的 Ke 衰减路径。
4. 增加受限的快速回撤保护
当前 6 Hz filtered force 仍是被动导纳输入，没有把 noisy raw force 直接送进控制器。
新增通道：
raw compensated force
→ median-of-3
→ 20 Hz 一阶低通
→ 仅用于撤销已经过时的负向主动 v_r
它只在以下顺序发生时动作：
先确认处于过力回撤阶段
→ fast force 从目标高侧下降
→ 连续越过 Fdes - stop_margin
→ 但 6 Hz force 仍要求继续回撤
→ 清除负向 v_r，短暂 hold
它不能：
产生正向下压力；
清除被动导纳 v_force_z；
禁止被动过力逃生；
在传感器过期时工作。
传感器数据过期时采用 fail-open，并清掉快速滤波历史，避免恢复后的旧数据误触发。
最初这里曾错误地在 Fdes + 0.25 N 就停止回撤，导致 1 N 匀速移动表面 MAE 上升到约 0.407 N。现在已改成真正的低侧目标交叉，移动表面回归重新通过。
5. 增加完整日志
新增了以下字段：
force_task_latched
physical_contact_state
physical_contact_acquire_event
physical_contact_loss_event
physical_contact_reacquire_event
physical_contact_low_timer_s
physical_contact_high_timer_s

force_fast_z
retract_guard_armed
retract_fast_hold
retract_fast_stop_count
retract_fast_rearm_count
force_reference_fast_clear
以后可以直接判断：
是真实失联还是补偿偏置；
什么时候重新 stiff-first；
fast force 是否领先 filtered；
快速保护是否误触发；
Dimeas、Ke 阻尼和主动参考分别做了什么。
6. 默认配置已经启用
两个配置文件均已打开：
physical_contact
fast_retract_guard
仍然只有一个 legacy_symmetric 控制器，没有重新引入之前删除的两个 v2/legacy 运行模式，也没有改变 RealMan TCP 同步和 TCP-Z 方向逻辑。
三、当前测试状态
已有定向回归：
76 passed
覆盖：
1/2/5 N 相同接触状态；
50 ms 振荡低谷不重新 stiff-first；
100 ms 真实失联；
raw 提前重新接触；
Ke 同周期恢复至 1500；
快速保护不误伤稳态移动表面；
raw 单点噪声不触发；
传感器过期 fail-open；
正负主动参考；
软表面、硬表面和既有 Dimeas 回归；
CSV 字段与行列对齐。
但新增的严格软硬切换测试目前是：
2 passed，2 failed
失败是有意保留的，它证明当前结构还没有满足“带 6 Hz/15 ms 延迟时，软硬切换不弹”的最终目标，不能为了绿灯而放宽成无意义阈值。
关键文件：
[controller.py](/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/admittance_common/controller.py)
[contact_state.py](/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/admittance_common/contact_state.py)
[fast_retract_guard.py](/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/rm75_control/control/admittance_common/fast_retract_guard.py)
[软硬切换仿真](/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/tests/test_force_stiffness_transition.py)
[默认配置](/media/camp/EXT_DRIVE/RealUS_playground/rm75_control/configs/joint_admittance_8dof.yaml)
四、下一步计划
第一步：增加“带限相位提前”，不直接使用裸 raw 控制
候选结构：
F6   = 现有 6 Hz 控制力
F20  = median-3 + 20 Hz 的快速力
ΔF   = clip(F20 - F6, ±correction_limit)
Fctl = F6 + β ΔF
初始扫描范围：
β = 0.50–0.75
correction_limit = max(0.5 N, 0.5 Fdes)
原则：
保留 6 Hz 作为低频准确基线；
20 Hz 只补回一部分相位，不把 raw 噪声原样送进导纳；
数据过期立即退回纯 6 Hz；
接触失联判断仍使用原 filtered/raw 状态机；
过力硬保护仍看实测力；
日志同时记录 F6/F20/ΔF/Fctl。
初步扫描显示这是目前最有希望的方向。简单模型中，β≈0.75 已显著降低硬段峰值，但尚未完成“median、限幅、传感器噪声、1/5 N 全矩阵”的最终验证。
第二步：把主动参考从“高增益短泄漏”改成平滑速度学习
当前参数：
gain = 0.10
leak = 0.3 s
立即按 force-error reversal 清零
它虽然追赶快，但会在软表面产生明显 TCP 速度纹波；切到硬表面后，同样的速度纹波被高 Ke 放大成力振荡。
初步结果显示：
gain ≈ 0.02–0.04
leak ≈ 3–10 s
配合带限相位提前，可能同时做到：
10 mm/s 表面移动在约 0.25–0.5 s 内建立参考速度；
稳态不需要持续保留较大力误差；
不在每次小误差过零时清掉正确的表面速度；
真正方向反转仍由 fast crossing 和连续确认清除。
一个初步候选在 1 N 硬面仿真中达到：
hard MAE ≈ 0.10–0.14 N
peak ≈ 1.3 N
无持续 4–12 Hz 弹跳
但这个结果尚未完成 5 N、真实噪声以及保留合理 idle decay 的全矩阵验证，所以还没有写入默认参数。尤其不能简单关闭 idle decay，否则静态人体表面可能重新出现高阻尼手感。
第三步：重新跑完整验证矩阵
至少包括：
1/2/5 N；
表面 ±5/±10/±20 mm/s；
300→2500→300 N/m；
单独测试更硬的 20 kN/m 接触，但使用更保守速度包线；
15/30/60 ms 延迟；
150–200 Hz 循环抖动；
实测 raw-force 噪声注入；
Kelvin–Voigt 和标准线性固体软组织模型；
真实 gap、重新接触与力连续的刚度切换。
验收继续使用：
1 N MAE ≤ 0.20 N；
5 N MAE ≤ 0.50 N；
5 N 峰值 ≤ 6 N；
正反方向 MAE 比 ≤ 1.25；
不出现持续超过 0.3 s 的 4–12 Hz 弹跳；
关闭主动参考后仍稳定；
软表面不能因为稳定措施重新变得明显沉重。
第四步：重新做当前工具的力补偿标定
当前补偿模型的单轴空载残差约 0.09–0.19 N，辨识中的力残差约 0.2 N 量级。这对 5 N 影响不大，但已经接近 1 N 的精度预算。
另外当前模型质量约 0.335 kg，旧基线约 0.996 kg。这可能是工具确实更换了，也可能是当前标定对应的装配不同，不能直接手工改一个 bias 解决。
最终上机前应在当前完整工具、正确 Arm_Tip 下重新执行：
source env.sh
python apps/force_compensation/force_calibrate.py
标定期间探头不能接触外界。不会在控制器中增加自动空载校零，因为 LOST 状态不等于已知自由空间，人体仍可能轻触探头，自动校零反而会吞掉真实接触力。
五、发布顺序
完成带限相位提前和主动参考重整；
让严格软硬切换测试全部通过；
用旧硬件日志重放，确认没有新的误触发；
重新标定当前工具；
固定软材料假体测试；
移动软材料测试；
软→硬假体切换；
1 N、2 N、5 N 分级测试；
全部通过后才进行人体测试。
当前最准确的状态是：接触失联和重新撞击保护已经修正并默认启用；真正限制软硬切换稳定性的 6 Hz 相位滞后已经被定位，但其带限补偿还需要完成和验证。
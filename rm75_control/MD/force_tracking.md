# TCP-Z 恒力控制

当前只有一套可运行的力控制器：`2965fea` 的稳定导纳主干，加上不同目标
力之间一致的主动追赶修正。配置里不再提供控制器模式切换。

保留的控制结构：

- RealMan TCP 同步决定当前工具坐标系，力方向仍为 TCP-Z；
- TCP-X/Y 与姿态跟踪，TCP-Z 执行恒力导纳；
- 1 s 目标力建立、enter-only 接触锁存；
- stiff-first 环境刚度估计与临界阻尼；
- Dimeas 高频振荡指标与变惯量；
- 正反向统一的 TCP-Z 速度上限。

针对 1 N/5 N 不一致和快速换向顶手，只修改主动参考：

```text
e      = smooth_deadband(Fdes - Fext)
Fscale = max(0.20 N, 0.15 |Fdes|)
drive  = sat(e / Fscale, -1, 1)

e > 0（欠力、继续压入）:
    gate = 1                              (Is ≤ 0.20)
           1 - (Is-0.20)/(0.60-0.20)    (0.20 < Is < 0.60)
           0                              (Is ≥ 0.60)
    v̇r = 0.10 drive · gate - vr/0.30 s

e < 0（过力、释放接触）:
    v̇r = 0.10 drive - vr/0.30 s
```

两方向在无振荡的小误差区使用相同增益。Dimeas 只衰减会继续向接触注入
能量的欠力下压；过力回撤始终保留，但仍受归一化输入、`v_r` 上限、TCP-Z
速度上限、Dimeas 质量和临界阻尼约束。有效力误差换向时清除旧方向
`v_r`，但不清导纳速度。重新接触时清除任意符号的旧参考。

质量、Dimeas 力尺度和阻尼上限不随运行时目标力改变。

CSV 中用于调试力追赶的主要列：

- `fz`、`f_des_z_eff`：实测力与建立后的目标力；
- `v_force_z`、`v_r_z`：导纳速度与主动参考；
- `force_reference_scale_n`、`force_reference_drive`、
  `force_reference_gate_scale`、`force_reference_accel_m_s2`：归一化
  尺度、输入、Dimeas 衰减和实际参考加速度；
- `force_reference_reversal_reset`：本周期是否清除了反向旧参考；
- `mass_z_eff`：Dimeas 调整后的实时虚拟质量；
- `instability_idx`、`damping_ke_z`、`damping_dimeas_z`：抗弹跳各层；
- `twist_requested_vz`、`twist_achieved_vz`、`vz_achieved_tool`：请求与
  编码器估计的实际 TCP 速度；
- `dt_actual_s`、`sensor_age_s`：循环和传感器时序。

快速测试仍使用：

```bash
source env.sh
python apps/joint_admittance_8dof/run_joint_admittance.py

source env_viewer.sh
python apps/joint_admittance_8dof/run_with_twin.py

source env.sh
python apps/joint_admittance_8dof/d_sin_tool_y.py \
  --d-target joints --move-mode cartesian \
  --enable-force --desired-z 2.0 --scan-duration 3000 -v
```

人体测试前先在软硬材料假体上分别测试 1 N、2 N、5 N，并保留完整 CSV。

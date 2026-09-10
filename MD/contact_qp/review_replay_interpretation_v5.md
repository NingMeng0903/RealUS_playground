# 回放结论定向复核 v5

仅核对 `scripts/evaluate_contact_qp_replay.py`、`replay_v5/behavior_details.json`、`qp_main/summary.json` 和 `qp_force_sweep/summary.json` 的统计与解释；没有重跑图像或闭环实验。**限定解释范围未发现阻断。**

主回放为同批 12,588 个历史真实图像的独立第一拍 QP 诊断：假设 4 N、20 mm/s 路径速度、合成 identity TCP-face 与 25 mm 半孔径、0.16 s 图像年龄，上一速度等于声明 nominal，能量关闭。每帧重新构造 solver，不积分候选命令到后续图像；历史实测力仅审计字段，不驱动此诊断。它不是新控制器的真实执行记录、闭环修复结果或原控制器反事实。

`behavior_details` 给出 v3 左右窗均达到声明 0.8 阈值的 7,014 帧全部透明；4,877 左单低、112 右单低的 rocking 符号全部与声明合成几何一致。其余 585 帧为双低，与主汇总左低 5,462、右低 697 一致。这里的“达到阈值”不能改称已知贴合正常；这些数也不是检测准确率或实际修复成功率。

力扫描覆盖 **68 帧**、3 个假设力值、2 个 nominal 模式、4 个算法配置，共 1,632 次独立求解；不能误写成 12,588 帧均作了力扫描。4.3 N 下 v3 新增局部修复量和最优 rocking 接近零；`force_recovery` 中原示例 nominal 仍为 −0.0006 m/s。这是保守力优先行与偏离 nominal/修复代价共同选择的最优动作，不说明整个 rocking 可行区间等于零。反例为半孔径 .025 m、Δω=.01 rad/s、Δvn=−.00025 m/s：新增两端速度为 0 和 −.0005 m/s，满足该过力侧力优先与 .00075 m/s 孔径预算，但不提供额外压入修复。

复核输入 SHA256：

| 文件 | SHA256 |
| --- | --- |
| `evaluate_contact_qp_replay.py` | `bd495c8ebcdbcfc6260550e0ab0ff18acefc191b1ca9100fd817fe5998fa0cde` |
| `behavior_details.json` | `ea42ef65ada2e440307d1a9b9d4da10cee63c42307d2f754d5af089cbd6e8017` |
| `qp_main/summary.json` | `5ec4bc1067cddae779be586df9a247548d1f6a3e346a48c03560cc1ca4c95b74` |
| `qp_force_sweep/summary.json` | `94b7f3e54c6fd0590e54faf5739774581e8caa5d8a39af9f46120920b0dcb1ae` |

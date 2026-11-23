右腿扫查 v6
基线 ba287167 与两条具名路线的图谱仿真

结果见 REPORT.txt、validation_summary.json 和 analysis/。发布清单中的路径
以实验目录为参照。冻结原件的校验值见 analysis/verification.json。

复跑顺序

在 leg_reader_teacher_scale_20261123 父目录运行。保留同目录下 v1 至 v5 的源文件。
需要本地 v2 atlas_routes.npz、v4 select_reader 开发分片、v5 fit_revisit 和 reader_test
分片，以及冻结读图器清单和关联权重。图册和骨先验沿用父目录的说明。dIRD 关闭。

  export LEG_SCAN_PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
  exp=leg_scan_identity_state_20261123_v6
  out="$exp/run/confirmed"
  mkdir -p "$out"
  bash gpu_preflight.sh > "$out/gpu_preflight.json" 2> "$out/gpu_driver.txt"
  "$LEG_SCAN_PYTHON" "$exp/usability_fit.py" --out "$out"
  "$LEG_SCAN_PYTHON" "$exp/role_fusion.py"
  "$LEG_SCAN_PYTHON" "$exp/anchor_logic.py"
  "$LEG_SCAN_PYTHON" "$exp/reader_study.py" --out "$out" --workers 10
  "$LEG_SCAN_PYTHON" "$exp/checks_v6.py" --out "$out"
  "$LEG_SCAN_PYTHON" "$exp/cadence_audit.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/run_closed.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/decision_audit.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/link_evidence_audit.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/verify_v6.py" --out "$out"

读图状态和因果输入

每次查询读取截至当前已取得的完整公开前缀。身份头和关联向量随该前缀重算。缓存由
读图实例、前缀长度和最后一帧标识，重复查询直接复用。训练存档采用半精度特征，在线
使用相同精度编码。对象状态分别保存归属、六类条件角色、实体回波概率和局部质量。
不同站位与声窗的代表以局部回波可用性、测量质量及关联路径置信度加权。可靠反证仍可
降低原身份概率。独立校准的联合交付评分用来从选定轨迹挑视图。实体回波头以点回波
是否为物理候选作为监督目标。身份与局部可用性没有按独立事件相乘。真值匹配只在
同轨诊断中使用，三种诊断共享相同真值连边及强度。真值连边图的密度仍可不同于公开图。

执行与比较

用 v5 原始触发、选窗、反馈和续扫控制作为固定对照。所有臂终局使用开发集选择的同
一个 v6 因果读图器。初始状态冻结、逐候选动态和持久对象动态三臂只允许同一组前向
声窗动作，预算、触发前缀、材料锚点和回接时钟相同。额外两臂根据可观察的候选布局
预测可达站位的区分能力，并比较执行到目标和真实停止。完成依据为新获取的邻站或
跨窗关联证据、局部可分离结构、已可交付旧图、对象身份风险与继续取证的预期收益。
真实停止按实际姿态接回且无需补齐时钟。锚点后短回访使用反馈 ProDMP，其提前出口
使用原运动约束。实际回接先做两站公开身份引导，再用共同 MPC 扫到末站。

教师与学生

教师的两条初始搜索带和增采带与两条最终复验带分开。所有估值都是相对于同布局、
同带、同读图器的固定 v5 策略。最终验证的教师子集按事先固定的编号选定。开发中的
学生从有限技能菜单实际采样一个程序并扫完。监督型 CUDA 价值学生接收公开前缀特征，
学习软回报、严格交付和错支的配对均值，在独立布局上实际执行冻结策略。
冻结清单分别保存公开策略候选、身份增量和教师扩量的验收规则。

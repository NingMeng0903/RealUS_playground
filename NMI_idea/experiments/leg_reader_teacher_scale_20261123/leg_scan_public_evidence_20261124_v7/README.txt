右腿扫查 v7
基线 60a77233。股静脉与腹股沟至远端小腿两条路线分别验收。

复跑

在实验目录的父目录运行。使用可见 CUDA 的 LEG_SCAN_PYTHON，保留同目录的 v1 至 v6
源码、本机 v2 atlas_routes.npz、v4 开发分片、v5 关联数据及 v6 冻结读图器和轨迹。
诊断和拟合均使用这些只读输入。新产物写到 v7 的 run 目录，可用 --out 指定新的目录。
主试使用公开匿名候选、材料坐标和公共图谱。MedSeg 短段和 Blender 未进入本次实验，
dIRD 关闭。各路使用预先分配的图谱来源、SMPL-X 形体和扰动设置。

  exp=leg_scan_public_evidence_20261124_v7
  out="$exp/run"
  mkdir -p "$out"
  bash gpu_preflight.sh > "$out/gpu_preflight.json" 2> "$out/gpu_driver.txt"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  "$LEG_SCAN_PYTHON" "$exp/confirm_v6.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/readout_audit.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/association_audit.py" --out "$out/association_audit.json"
  "$LEG_SCAN_PYTHON" "$exp/bridge_v7.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/observation_v7.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/contracts_v7.py" --out "$out"
  "$LEG_SCAN_PYTHON" "$exp/cycles_v7.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/edge_score_replay_v7.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/student_mpc_v7.py" --out "$out" --workers 6
  "$LEG_SCAN_PYTHON" "$exp/student_mpc_v7.py" --out "$out" --score-only
  "$LEG_SCAN_PYTHON" "$exp/verify_v7.py" --out "$out"

扫描 worker 使用 spawn 和 CPU。学生模型、样本张量和 AdamW 更新在 CUDA 上运行。
分片提交文件保存数据和元数据哈希。学生检查点保存在各轮 student_wave*_checkpoint，
包括最佳验证模型、优化器和 RNG 快照。第二轮基于两批实际前缀的合并样本重新拟合。
确认、观测模型和闭环实验各有冻结清单，协议或源文件变化需要新结果目录。

指标

严格交付要求合法末站和六区目标正确；错支指整趟至少一次交付其它具名血管。
获取上限指六区是否各至少拍到一次目标。另记各区目标获取、拒交、伪回波和正确长度。
整趟联合成功要求旧候选终局正确、后缀规定区均正确且合法到达末站。救回、损失及
复验带置信区间均用同一布局及配对未来扰动计算。两条复验带先在布局内取均值。

学生试验为两次公开状态驱动的宏程序选择和监督动作价值循环。教师从学生实际到过
的第二个前缀起比较继续、补拍、观察和冻结参照。搜索带与实际带、测试复验带分离。
教师和学生共用严格收益下界和错支上界的放行条件，估计区间分别来自小样本符号
后验和开发布局的残差。学生每个决策点的参照为两步 v6 式公开引导，第二个决策结束
后接共同 MPC。验收同时配对普通 MPC。全程主确认用冻结 v6 完整动态锚点与普通 MPC。

REPORT.txt 保存结果和限制。分析下的 paired_compact.json 提供可复算的配对评分。
逐步分片、逐动作预测、候选级真值诊断与权重在本地 run 下。先前桥接代码修订期间
的预跑分片移存 pre_bridge_source_lock，和最终计数分开。

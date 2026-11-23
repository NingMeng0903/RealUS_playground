右腿远处锚点、实例关联和持久公共状态
实验 v5，基线 c4e4b321

结果见 REPORT.txt、validation_summary.json 和 analysis/。冻结摘要记录本机原件的
SHA-256。仓库里的 reader_settings.json 将本机路径写成实验父目录相对路径，重跑时
由脚本重新写出运行目录中的冻结文件。

复跑顺序

从本目录的父目录 leg_reader_teacher_scale_20261123 执行。图谱放在
leg_scan_target_routes_20261123_v2/atlas_routes.npz。使用 v4 的本地 run 归档，
包括 reader_freeze.json、先验权重、fit、calibrate、select_reader、reader_test
及 action_results.json。v4 冻结清单引用的本地权重路径须可读。调和图册和骨先验
沿用父目录 README.txt 所列的环境与资产路径。
  export LEG_SCAN_PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
  exp=leg_scan_anchor_state_20261123_v5
  mkdir -p "$exp/run"
  bash gpu_preflight.sh > "$exp/run/gpu_preflight.json" 2> "$exp/run/gpu_driver.txt"
  "$LEG_SCAN_PYTHON" "$exp/replay_v4.py" --workers 6
  "$LEG_SCAN_PYTHON" "$exp/ablate_v4_edges.py"
  "$LEG_SCAN_PYTHON" "$exp/study_data.py" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/mechanism_tests.py"
  "$LEG_SCAN_PYTHON" "$exp/fit_and_evaluate.py" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/mechanism_tests.py" --public-contract
  "$LEG_SCAN_PYTHON" "$exp/diagnose_fusion.py"
  "$LEG_SCAN_PYTHON" "$exp/run_anchor_study.py" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/verify_study.py"

上述入口使用现有 v1–v4 本机源文件、v4 冻结读图器和分片，以及只读公开图谱。保留
股静脉和全程目标路线。dIRD 关闭。CUDA 只训练关联头，仿真动作和公开图计算可在 CPU
并行进行。训练包括 1,152 个 v4 拟合布局及 1,536 个新回发布局。新测试的随机带独立。

取证和独立验收

前三个证据产物是 v4 逐例回放、删边重播和 v5 控制机制测试。回放保留旧池相同候选 ID、
提交状态、后缀状态和实际时钟。删边只从 v4 图中移除直接把错误赢家连到真目标的边，
保持其他对应关系。该诊断使用真值且限于离线。双轨机制测试让 C 节点通过 B 修改固定
的 A 节点，没有 A 到 C 直连。读取相同证据再次查询保持幂等。

关联训练和推断使用同一空间候选检索，允许时间相隔较远的跨窗回访。杂波、粘连、
邻近动脉和不同局部分量提供负对。先校准边、再匹配。同一轨迹内每帧最多一个候选。
一个来源及角色必须支持整条组轨迹的材料站位跨度。按视角去重并对多跳证据限幅，
最长 32 个匹配边。公开状态按帧和槽号一次入账，可以用新采到的证据更新旧候选。

同轨验证共用同一训练权重。比较只用逐候选概率、旧 guard、先校准再匹配及离线真值
对应诊断。校准与提交阈值都在测试采集之前冻结。代码与结果区分候选匹配阈值和报告
提交阈值。真值对应不进入动作、早停或在线读图。

diagnose_fusion.py 只重读冻结测试观测，把每区概率赢家的变化与拒识、跨区排序分开。
取证回放额外固定普通 MPC 的续扫姿态序列，并比较单独延后采样、延后并补入静止图像、
延后并补入复查图像。点回波和质量的哈希与读图后验分开记录。

闭环原型的两份假设由持续跟踪的公开位置和身份分布生成。动作预测显式比较可见性及
两假设下的观测差异。同步记录普通 MPC、逐候选、上一证据时刻和更新后的合法窗口
分数。主动程序从实际前缀向远侧已支持的拓扑站位走 3 至 6 站。根据后验可以提前结束
专门追踪，也可以在实际锚点合法短回访。各臂由同一公共 MPC 续扫。等时补拍与主动臂
在相同材料站、同一时钟进入续扫，允许因为实际取像程序而使用不同扇区。普通 MPC
按真实行程运行。结果同时给整趟严格和旧候选池正确、要求的后续区域取像并交付正确、
合法结束的联合指标。后续区域提交只由探查开始后从前进方向取得的候选满足。

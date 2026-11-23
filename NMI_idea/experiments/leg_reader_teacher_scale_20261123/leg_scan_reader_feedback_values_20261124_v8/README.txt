右腿闭环扫查 v8
基线 2f2c704e。股静脉至腘静脉、腹股沟至远端小腿各自独立评分。

复跑输入

在实验目录的父目录运行。保留 v1 至 v7 源码、本机 v2 atlas_routes.npz、v4
select_reader、v5 fit_revisit 和 v6 冻结读图器、归档轨迹。按原有材料坐标路线，
股静脉使用 BodyParts3D 与 Open3DModel 右腿，全程使用 Open3DModel 右腿。MedSeg
短段和 Blender 留出隔离，dIRD 关闭。传感器使用二维截面匿名质心回波。

  exp=leg_scan_reader_feedback_values_20261124_v8
  out="$exp/run"
  mkdir -p "$out"
  bash gpu_preflight.sh > "$out/gpu_preflight.json" 2> "$out/gpu_driver.txt"
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  "$LEG_SCAN_PYTHON" "$exp/capture_v8.py" --out "$out" --workers 10
  "$LEG_SCAN_PYTHON" "$exp/readout_v8.py" --out "$out" --workers 6
  "$LEG_SCAN_PYTHON" "$exp/edge_v8.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/cosine_panel_v8.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/reader_panel_v8.py" --out "$out" --workers 6
  "$LEG_SCAN_PYTHON" "$exp/false_links_v8.py" --out "$out" --workers 4
  "$LEG_SCAN_PYTHON" "$exp/evidence_v8.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/ambiguity_v8.py" --out "$out" --workers 6
  "$LEG_SCAN_PYTHON" "$exp/cycles_v8.py" --out "$out" --workers 12
  "$LEG_SCAN_PYTHON" "$exp/verify_v8.py" --out "$out"

先采固定终局轨迹和辅助复查快照。历史冻结轨迹上的对称读出实验独立进行。主关联用
直接实例监督。边阈值分开校准，同轨比较保留旧逐候选读出并逐病例匹配提交量。冻结
自主阈值的结果另报。公开对象身份、候选联合交付、赢家风险和正常提交阈值按开发
布局拆分，模型与阈值在新固定轨迹评分前锁定。边和学生训练使用 CUDA，外层几何
rollout 使用 spawn 和 CPU。

固定对象取证

一个布局先从匿名前缀选定两次回波的帧号与槽位，优先选择同一帧不同回波。
每个分支用实际扫描与相同未来扰动比较 A 偏好、B 偏好、双可见、原位补拍和普通
续扫。目标到达路径包含实际声窗与时钟。另记真实物理视场、独立候选、真检索通路、
评分阈值、匹配、身份融合和读出。A 或 B 为目标、两者都不是目标、同实例的重复表示
分别统计。隐含实例只进入离线诊断和回报。各臂以原始共同前缀的旧图库定义联合成功。

两阶段学生

第一阶段执行两站，第二阶段在新观测上选择完整宏程序并接公共 MPC。先在学生实际
到达的前缀上生成第二阶段标签，拟合并冻结第二阶段。再在另一批学生布局从共同起点
操纵第一阶段，并从每个实际到达前缀运行同一冻结第二阶段。未执行的候选也通过真实
分支生成反事实回报。搜索使用配对测量带，额外复验预算集中到较有希望的候选，
最终验收使用独立带。正确区数、长度、严格完成和错支进入软回报；最终严格交付仍为
合法末站的六区均正确。动作模型按布局均值拟合，校准以布局为单位。机理验收、最终
策略验收和学生训练各有独立布局、哈希和计数。每个阶段检查点包含最佳验证模型、
对应优化器和 RNG。报告与可复算配对评分见 REPORT.txt 和分析目录。

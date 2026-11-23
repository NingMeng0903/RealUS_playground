右腿读图器与动作教师更新，v3

路线

股静脉从 −595 至 −790 mm，沿股静脉至近端腘静脉。BodyParts3D 右腿和
Open3DModel 右腿提供连通路径。腹股沟至远端小腿从 −525 至 −1255 mm，沿股静脉、
腘静脉和胫后静脉，使用 Open3DModel 的连通路径。站位为中立形体材料站位。

读图验证

三组独立种子各采集 12,288 趟拟合、512 趟校准和 1,024 趟配对验证。拟合含普通
MPC 与真实执行换窗、短回访的轨迹。四个读图方法用相同验证帧。两个更新均从 v2
全公开 embedding 权重起跑，保留该网络的 4,296 个参数。数据量对照更新身份与跨视
角关联目标。新版另在各交付区全部已获得的候选中比较最强真目标与最强对手。校准按
路线匹配 v2 交付量。验证报告冻结阈值和逐趟等提交量两个结果。

教师验证

教师使用冻结的新版读图器。每个前缀比较直接继续、等时同窗补拍、双向换窗和短回访。
探查使用受限的逐帧反馈 ProDMP，回到合法主线姿态后，所有分支用同一公共 MPC 扫到
终点。同一布局和前缀的动作共用未来扰动。每个拟合前缀使用两条未来带，真值仅在
离线交付计分中使用。教师学习配对收益的条件期望、整趟救回和损失。开发组校准动作
选择。验证使用原始未来带与两条独立复验带。旧式特权搜索使用另外两条规划带作为
对照。主动 MPC 通过公开可见性预测选择同一可行动作菜单。dIRD 关闭，保留调和图册
和材料位置。保留旧图改对或改错、区间错支、去重正确毫米、路程和用时。

验收与下一轮

两条路线逐项计分。严格交付要求到达末站且六区均正确。完整配对结果以布局为单位。
读图器用等提交量的错交和冻结阈值下的交付判断进展。教师需要在等时补拍之上可复验
地增加严格交付，同时核对错支和相对普通 MPC 的整趟损失。满足两项后，下一轮用
冻结读图器和动作价值，在学生访问的失败前缀做小规模 PPO 和独立配对。由
next_experiment.json 记录各路线的决定。本轮包含合成图谱上的读图器与教师更新。

复跑

本机的 Python 3.10、PyTorch CUDA、NumPy、SciPy 和 Numba 用于本轮。仓库实验目录
附上运行所需的两版源代码快照。图谱、骨遮挡先验、调和图册和旧读图器检查点由本地
实验提供。由 reader_collection.json 和 reader_freeze.json 记录输入摘要及校验值。

  export LEG_SCAN_PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
  export RUN_DIR=/path/to/local_run
  mkdir -p "$RUN_DIR"
  bash gpu_preflight.sh > "$RUN_DIR/gpu_preflight.json" 2> "$RUN_DIR/gpu_driver.txt"
  "$LEG_SCAN_PYTHON" leg_reader_teacher_scale_20261123_v3/collect_scale.py \
      --out "$RUN_DIR" --atlas-pack /path/to/atlas_routes.npz --workers 16
  "$LEG_SCAN_PYTHON" leg_reader_teacher_scale_20261123_v3/run_scale.py \
      --out "$RUN_DIR" --atlas-pack /path/to/atlas_routes.npz --workers 16 \
      --prior-reader /path/to/full_public_embedding.pt \
      --prior-freeze /path/to/reader_freeze.json

调和图册和骨先验默认路径见锁定快照里的 material_assets.py 和 scan_kernel.py。
在其他工作站复跑时，提供与输入清单校验值一致的本地资产。固定图谱集合内的新合成
布局用于统计区间，独立人体或实机验证需另行评估。

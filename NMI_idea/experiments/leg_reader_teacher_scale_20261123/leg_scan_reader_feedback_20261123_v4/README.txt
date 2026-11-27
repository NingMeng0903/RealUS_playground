右腿血管公开关联、闭环反馈和动作复验
实验 v4，基于 22d1f9c8

几何与观测

保留股静脉路线 −595 至 −790 mm 和全程路线 −525 至 −1255 mm。使用具名连通图谱的
合成布局、同原材料坐标和运动约束。按图谱来源轴取最小距离，保留血管角色轴。短历史
分别记录采集帧、实际不同声窗和同姿态相关性。回波候选公开，实例及角色真值仅用于
监督和验收。dIRD 关闭。

读图器

以 v3 扩量身份与关联权重初始化 4,296 参数读图器。用 8,192 趟训练的实际路径，在
显式时间截断点提取匿名候选。加入第 3 区真实换窗和回访，使用只跨已支持相邻边的
源内局部实例关联监督。保留每段完整历史。读图头分别输出角色概率和关联向量。
公开关联图依据实际位置、声窗、关联向量和目标连通支持建边，一条轨迹在同帧最多一
个候选。利用不同视角的邻居重新估计旧候选身份。提交置信度拟合于 512 趟校准数据。
在额外 512 趟开发数据上按路线选择可用权重、推断方式和阈值。1,024 趟同轨测试只
在冻结后采集。分别比较逐趟等提交量、固定阈值、风险覆盖和旧图改对／改坏。

可执行复查

同一实际到达前缀上，继续、等时补拍、换窗和短回访共用布局、扰动带、起点和运动预算。
ProDMP 用实际经过的位置生成新回波，候选接同一公共 MPC 扫到终点。开发集部分案例
在实际到达的原窗开始一个混叠剖面。留出按预先定义的自然分布，无该干预条件。选择
使用两条规划带，复验另用正向实际带和两条独立带。分别报告普通 MPC 和等时补拍配对
结果。测实际动作角幅、轴向距离、最长连续正确毫米和总正确覆盖。较强搜索只能评估
已执行的菜单，报告是经验能力参照。

反馈与动作价值

同一个图关联读图器评估等级相同的复查。评分臂共用原 MPC 与 ProDMP 反馈，控制臂
用已取得的最近 14 帧公开后验更新接续规划权重和段内 ProDMP 反馈。两臂使用同一高层
动作计划。记录回波进入反馈和实际轨迹差异。两个参照下的预期严格收益及错支风险在
训练和推断用相同变换。仅当独立开发证据达到双参照收益条件，才纳入学习动作策略的
正式验收。开发续扫同时可用于实际 CUDA 参数化验证。后续的重复干预和停止学习由这
一轮的结果决定。

复跑

在包含本目录及 v1–v3 源码的实验父目录，使用本地 CUDA Python 和同版本只读资产。
所需图谱为 v2 的 atlas_routes.npz。权重为 v3 的 scaled_identity.pt，其 SHA-256
由 v3 reader_freeze.json 校验。调和图册和骨先验路径沿用 v3 说明。

  export LEG_SCAN_PYTHON=/media/camp/EXT_DRIVE/envs/genesis/bin/python
  export RUN_DIR=/path/to/new_run
  study=leg_scan_reader_feedback_20261123_v4
  mkdir -p "$RUN_DIR"
  bash gpu_preflight.sh > "$RUN_DIR/gpu_preflight.json" 2> "$RUN_DIR/gpu_driver.txt"
  "$LEG_SCAN_PYTHON" "$study/check_contracts.py" --atlas-pack /path/to/atlas_routes.npz \
      --out "$RUN_DIR/contracts.json"
  "$LEG_SCAN_PYTHON" "$study/collect_reader.py" --atlas-pack /path/to/atlas_routes.npz \
      --out "$RUN_DIR" --workers 16
  "$LEG_SCAN_PYTHON" "$study/run_study.py" --atlas-pack /path/to/atlas_routes.npz \
      --prior-reader /path/to/scaled_identity.pt --prior-manifest /path/to/reader_freeze.json \
      --out "$RUN_DIR" --workers 16
  "$LEG_SCAN_PYTHON" "$study/audit_graph_errors.py" --atlas-pack /path/to/atlas_routes.npz \
      --out "$RUN_DIR"
  "$LEG_SCAN_PYTHON" "$study/confirm_link_guard.py" --atlas-pack /path/to/atlas_routes.npz \
      --out "$RUN_DIR" --workers 6

源码与汇总保存在实验分支。合成几何、扫描数据和模型检查点保留在本机可核对的分片中。

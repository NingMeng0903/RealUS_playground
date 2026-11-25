# v15 右腿扫描联合读图与恢复

v15 保留完整的 177 维候选表示，在第 4–6 区共享观测到的对象轨迹、相对几何和病例上下文。新版后验参与补查选窗、短程追踪、普通续扫和最终交付。实验在本机超声几何仿真中运行，目标路线从腹股沟到远端小腿。

[实验分析](REPORT.md) · [实现说明](METHOD.md)

## 本机复现

在交接根目录运行。依赖同级 v13、v14 实验的冻结 `run`、前序代码和权重，以及公开图谱包。`--atlas` 可覆盖图谱路径。各阶段接受 `--out` 指定新的结果目录；改用新目录时，自检和各阶段需指向同一输出。全程使用 `spawn` 创建仿真进程，CUDA 训练单独运行。

```bash
set -euo pipefail
v15="$PWD/leg_scan_shared_belief_20261126_v15"
v15_python="${LEG_SCAN_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
mkdir -p "$v15/run"
bash gpu_preflight.sh > "$v15/run/gpu_preflight.json" 2> "$v15/run/gpu_driver.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$v15/run/numba-cache"
"$v15_python" "$v15/data_v15.py" --workers 16
"$v15_python" "$v15/run_v15.py" fit-reader
"$v15_python" "$v15/analysis/shape_replay.py"
"$v15_python" "$v15/run_v15.py" teach --workers 18
"$v15_python" "$v15/run_v15.py" fit-student
"$v15_python" "$v15/run_v15.py" develop --workers 18
"$v15_python" "$v15/run_v15.py" blind --workers 18
"$v15_python" "$v15/analysis/verify_v15.py"
```

`run/reader_data/` 保存训练观测及取证对，`run/shards/` 保存逐布局教师续扫和测试。两者均逐布局原子保存，可重开后继续。`run/models/` 保存读图推理权重、学生推理权重和 CUDA 学生优化器检查点。冻结文件记录图谱、源代码、自检、模型和随机种子。Git 发布包含实现、模型导出、冻结清单、汇总及压缩逐布局结果。

Git 导出也支持独立复算。在 v15 目录运行 `python analysis/verify_v15.py --outputs analysis --models models`。复算核对同轨候选、六区交付、错支、合法完成、取证改判和教师独立带收益。

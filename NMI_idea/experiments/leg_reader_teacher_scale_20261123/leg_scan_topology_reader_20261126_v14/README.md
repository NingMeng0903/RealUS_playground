# v14 关系图终局读图

本实验沿用 v13 的回波 embedding、冻结策略和探头轨迹。第 6 区的终局读图联合比较两个候选身份及未知，并在六区严格交付和具名错支上作同轨配对。方法和证据条件见 [DESIGN.md](DESIGN.md)。

在原本机交接根目录运行。需要同级 v13 实验及其冻结 `run`、前序代码和权重、已对齐的公开图谱包。`--out` 可以指定一个新的结果目录。CPU 重放在进程中并行，CUDA 只由训练阶段使用。特征和重放按分片保存，可重新执行相应阶段继续收集。

```bash
set -euo pipefail
v14="$PWD/leg_scan_topology_reader_20261126_v14"
v14_python=/media/camp/EXT_DRIVE/envs/genesis/bin/python
mkdir -p "$v14/run"
bash gpu_preflight.sh > "$v14/run/gpu_preflight.json" 2> "$v14/run/gpu_driver.txt"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$v14/run/numba-cache"
"$v14_python" "$v14/experiment_v14.py" smoke
"$v14_python" "$v14/experiment_v14.py" prepare --workers 16
"$v14_python" "$v14/experiment_v14.py" fit
"$v14_python" "$v14/experiment_v14.py" evaluate --workers 16
```

`--junctions` 接受经过核查的公开关系表。各条记录包含材料站索引 `station`、主线角色 `stem_role`、侧支角色 `side_role`、沿扫查方向的 `kind` 和 0–1 的置信权重 `weight`。`kind` 使用 `bifurcation` 或 `confluence`。本轮第 6 区采用冻结图谱，关系表为空。

输出包括 `protocol.json`、GPU 自检、开发集验证、原子保存的读图权重和优化器检查点、冻结清单及逐布局配对评分。公开分析包含选用的模型权重和配对指标。原始图谱及完整轨迹来自本机冻结 v13 输入。

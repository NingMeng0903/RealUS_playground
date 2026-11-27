# v16 右腿有限段风险配对仿真

冻结 v15 的读图器和学生。v16 从该学生实走的下肢历史出发，固定同一次复查和后续候选菜单，比较是否读到复查回波以及是否保留完整有限后验。控制器选择后直接执行一次局部非轴向超声取像，用隐藏静态血管几何和新回波标签评定结果。

[实验结果](REPORT.md) · [方法和验收定义](METHOD.md)

## 本机复现

使用交接根目录下原有的 v2 至 v15 源码、模型和图谱资产。v16 检查冻结 v15 权重和原图谱的哈希。盲测和剪切形变组使用此前未用于训练的种子。原始数据在本机只读使用。

```bash
cd leg_scan_segment_risk_20261126_v16
v16_python="${LEG_SCAN_PYTHON:-/media/camp/EXT_DRIVE/envs/genesis/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$PWD/run/numba"
"$v16_python" run_v16.py blind --out run/final --workers 16
"$v16_python" run_v16.py shear --out run/final --workers 16
"$v16_python" analysis/verify_v16.py --out run/final
"$v16_python" analysis/check_segment_math.py
```

每个布局和随机带支持按压缩记录续跑。导出版的逐布局结果和冻结清单位于 `analysis/`，使用 NumPy 可独立复算。

```bash
python analysis/verify_v16.py --out analysis --summary /tmp/v16_recount.json
sha256sum -c SHA256SUMS
```

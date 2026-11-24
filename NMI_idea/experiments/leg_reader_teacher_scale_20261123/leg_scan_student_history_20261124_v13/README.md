# v13 学生历史读图与统一续扫

本轮以冻结 v12 学生实际完成的多次补查轨迹为读图训练输入。候选以帧号和槽位关联取证前后，排序训练使用部署时的 v10 残差分数。动作标签使用同一版本的续扫学生和终局读图。冻结验收比较同轨读图更新、学习选区、同菜单几何策略与旧候选评分锁定。

[验收报告](REPORT.md) · [采集与训练方法](METHOD.md)

从本机交接根目录运行。需要同级 v10、v11、v12 的源码及已冻结本地权重、现有 atlas 投影包。采集和反事实由 CPU worker 执行，读图器和学生在 CUDA 更新。运行目录保存原子分片、权重、优化器、RNG 与冻结哈希。

```bash
src_v13="$PWD/leg_scan_student_history_20261124_v13"
out_v13="$src_v13/reproduction"
py_v13=/media/camp/EXT_DRIVE/envs/genesis/bin/python
mkdir -p "$out_v13"
LEG_SCAN_PYTHON="$py_v13" bash gpu_preflight.sh >"$out_v13/gpu_preflight.json" 2>"$out_v13/gpu_driver.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$py_v13" "$src_v13/collect_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/reader_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/run_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/verify_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/matched_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/scale_v13.py" --out "$out_v13"
"$py_v13" "$src_v13/verify_v13.py" --out "$out_v13" --supplemental
```

Git 的 `analysis/` 包含训练清单、独立配对结果及哈希，解压后的 JSON 直接对应运行清单中的 SHA-256。候选历史分片与模型权重存于本机 `run/`。

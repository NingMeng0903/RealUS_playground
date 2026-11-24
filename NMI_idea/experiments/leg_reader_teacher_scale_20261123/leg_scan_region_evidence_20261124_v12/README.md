# v12 右腿超声分区取证与续扫

v12 将第 4 至 6 区的可执行取证程序纳入同一个动作菜单。每次取证后，从已采帧更新跨窗对象关联和具名身份，再由学生决定继续取证、走下一段或交付。全程优先检验第 5 区小腿静脉与竞争动脉的辨识。股静脉全路线沿用 v10 冻结行为。

[冻结验收](REPORT.md) · [方法、先例和贡献边界](METHOD.md)

## 复跑

从本机交接根目录运行。依赖原有 atlas 投影包、v10 冻结读图与学生、v11 冻结读图与动作权重。所有仿真在 CPU worker 上执行，读图与学生优化在 CUDA 上执行。原 v11 数据、图谱和权重用作只读输入。

```bash
src_v12="$PWD/leg_scan_region_evidence_20261124_v12"
out_v12="$src_v12/reproduction"
py_v12=/media/camp/EXT_DRIVE/envs/genesis/bin/python
mkdir -p "$out_v12"
LEG_SCAN_PYTHON="$py_v12" bash gpu_preflight.sh >"$out_v12/gpu_preflight.json" 2>"$out_v12/gpu_driver.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$py_v12" "$src_v12/replay_v11.py" --out "$out_v12"
"$py_v12" "$src_v12/data_v12.py" --out "$out_v12"
"$py_v12" "$src_v12/fit_v12.py" --out "$out_v12"
"$py_v12" "$src_v12/data_mix_v12.py" --out "$out_v12"
"$py_v12" "$src_v12/run_v12.py" --out "$out_v12"
"$py_v12" "$src_v12/verify_v12.py" --out "$out_v12"
"$py_v12" "$src_v12/fixed_evidence_v12.py" --out "$out_v12"
```

每个布局的分片含原始观测、实际动作、配对结果及哈希校验。CUDA 检查点保存优化器和 RNG，位于本机 `run/models/`。Git 上的 `analysis/` 含冻结清单、逐布局结果与复算结果。

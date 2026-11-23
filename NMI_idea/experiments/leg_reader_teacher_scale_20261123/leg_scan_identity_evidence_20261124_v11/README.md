# v11 右腿超声身份取证

沿用 v10 的定向恢复学生。股静脉保留冻结的策略和读图器。全程在第 4–6 区对旧候选提出身份查询，再实走同一材料站位的等时补拍或异窗取证。原始帧键保留到终局，可直接测量新证据是否让旧候选重新交付。时间和路程按实际轨迹记录，全程采用 v10 的运动和采样接口。

[配对验收](REPORT.md) · [论文方法与贡献边界](RELATED_WORK.md) · [主试逐布局数据](analysis/paired_outcomes.json) · [几何换窗确认配对](analysis/geometry_confirmation_paired_outcomes.json)

## 代码和输入

| 入口 | 用途 |
|---|---|
| `run_v11.py`、`pipeline.py` | 分片采集、教师标签、开发选择、冻结验收 |
| `evidence.py`、`evidence_model.py` | 材料候选地址、真实轨迹取像、身份先验与似然、任务收益 |
| `mixed_reader.py` | 混合旧轨迹、困难病例和实际恢复序列的 CUDA 监督训练 |
| `confirm_geometry.py`、`verify_v11.py` | 新布局确认、数据与权重复核、公开观测复算 |

在本机交接根目录运行。保留同级 v2 拓扑包、v4 读图训练归档、v9 reader bundle、v10 冻结读图与学生权重，以及各版本导入的原有源码。归档读取和权重的指纹见运行清单。重跑首先执行实际 CUDA 前向、反向、优化器更新和同步。

```bash
src_v11="$PWD/leg_scan_identity_evidence_20261124_v11"
out_v11="$src_v11/reproduction"
py_v11=/media/camp/EXT_DRIVE/envs/genesis/bin/python
mkdir -p "$out_v11"
LEG_SCAN_PYTHON="$py_v11" bash gpu_preflight.sh >"$out_v11/gpu_preflight.json" 2>"$out_v11/gpu_driver.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$py_v11" "$src_v11/run_v11.py" --out "$out_v11" --workers 14
"$py_v11" "$src_v11/confirm_geometry.py" --out "$out_v11" --workers 14
"$py_v11" "$src_v11/verify_v11.py" --out "$out_v11"
```

Git 附全量验收逐布局评分、训练与数据清单。带超声观测的采集分片、PDF、模型、优化器检查点和运行日志保存在本地 `run/`。`analysis/protocol.json`、`analysis/test_freeze.json` 与 `analysis/geometry_confirmation_freeze.json` 固定版本、划分和扰动。

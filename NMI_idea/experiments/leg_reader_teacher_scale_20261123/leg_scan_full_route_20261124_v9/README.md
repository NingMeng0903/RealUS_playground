# v9 右腿全程闭环训练

股静脉路线沿股静脉到腘静脉。全程路线从腹股沟的股静脉标注段经腘静脉到远端胫后静脉。两条路线分别验收。供试环境是图谱衍生解剖和匿名质心回波的接触几何仿真。本轮关闭 dIRD。

- `fit_reader_v9.py` 在已有仿真轨迹上联合训练共享编码器、实例关联、历史融合、区域排序与拒交。独立校准区域提交阈值。
- `run_v9.py` 在当前冻结读图条件下，以实际 MPC 轨迹初始化学生。从起点到末站最多每五站交回控制权，执行两组匹配布局的多轮 PPO。教师组对部分学生实际到访前缀做配对续扫，并训练公共动作收益头。
- `bc_ablation_v9.py` 用冻结的行为克隆初始策略补跑同一份配对验收，检验 PPO 更新比初始探索分布多带来多少收益。
- `verify_v9.py` 从提交分片复算实走完成数，重放保存的行为概率与价值，核对训练及部署哈希，并按两条路线独立复算六组配对结果。

[结果与失效分析](REPORT.md) · [理论推导和机制实测](THEORY.md) · [论文依据](PAPERS.md)

## 本机复跑

在交接根目录运行以下命令。新试验使用独立的输出目录。读图训练访问 v4 的 `run/fit` 与 `run/select_reader`、v5 的 `run/fit_revisit` 和 v8 的 `run/reader_test`。配对运行还需本机 v6 存档的旧读图权重和 v2 的具名 `atlas_routes.npz`。这些文件及仿真旧依赖随本机交接保存。远端提交中的源码和分析可用于审阅，复跑需要这些本机输入。

```bash
py_v9=/media/camp/EXT_DRIVE/envs/genesis/bin/python
src_v9="$PWD/leg_scan_full_route_20261124_v9"
out_v9="$src_v9/reproduction"
mkdir -p "$out_v9"
LEG_SCAN_PYTHON="$py_v9" bash gpu_preflight.sh >"$out_v9/gpu_preflight.json" 2>"$out_v9/gpu_driver.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$py_v9" "$src_v9/fit_reader_v9.py" \
  --out "$out_v9" --epochs 5 --batch 24 --workers 6
"$py_v9" "$src_v9/run_v9.py" \
  --out "$out_v9" --workers 12 --rounds 12 --batch 96 --dev-layouts 48 --test-layouts 160
"$py_v9" "$src_v9/bc_ablation_v9.py" --out "$out_v9"
"$py_v9" "$src_v9/verify_v9.py" --out "$out_v9" --with-bc
```

读图器可以用 `--resume` 从本机保存的检查点恢复。全程训练命令以相同参数重启会验证冻结协议和分片后继续。运行目录中的权重、优化器、随机数状态、训练分片、精确路线以及 CUDA 日志均可本机审计。远端分析目录含冻结配置、逐布局评分和复算结果。

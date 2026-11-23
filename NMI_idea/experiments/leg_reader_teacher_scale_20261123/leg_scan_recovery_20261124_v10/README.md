# v10 右腿分区恢复

常规段使用已有 MPC。学生依据每区目标支持、竞争身份和材料位置选择复查目标、技能、声窗与探查幅度。复查结束后学生可以继续在该区取像、返回另一区或前进。所有移动按接触几何与真实时钟采样。v10 取消 v9 的全程时间、路程和复查次数配额，单列实际成本。本机离散化观测带有固定长度，验证记录最大时钟。

训练先由强教师在真实前缀搜索两步恢复程序。在两个共同未来带上评价候选，用第三条带验证已选程序。学生先用成对动作价值训练，再实走，使用自己取得的困难轨迹重训读图器和解剖角色头。读图器更新后重算教师标签，并监督补拍后的继续或离开。随后对学生实际访问的前缀用同一冻结学生复验，结合实走回报更新残差策略。股静脉和全程分开统计。

[验收报告](REPORT.md) · [理论与 v9 失败复核](THEORY.md)

从本机交接根目录复跑。需要 v2 具名图谱和本机 v9 导出的新读图器。

```bash
src_v10="$PWD/leg_scan_recovery_20261124_v10"
out_v10="$src_v10/reproduction"
py_v10=/media/camp/EXT_DRIVE/envs/genesis/bin/python
mkdir -p "$out_v10"
LEG_SCAN_PYTHON="$py_v10" bash gpu_preflight.sh >"$out_v10/gpu_preflight.json" 2>"$out_v10/gpu_driver.txt"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
"$py_v10" "$src_v10/run_recovery.py" --out "$out_v10" --workers 14
"$py_v10" "$src_v10/verify_recovery.py" --out "$out_v10"
```

冻结协议、数据分片、优化器检查点、实际动作轨迹与部署选择均保存在本地运行目录。Git 附读图、教师和学生的代码及配对分析。

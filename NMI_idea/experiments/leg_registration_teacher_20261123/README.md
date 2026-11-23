# 胫后静脉配准、联合留图与前瞻动作教师

实验从同一段已执行的右腿仿真前缀出发。验收包含入口旧图、实际取得的下游新图，以及连续三个 5 mm 区间的因果判读。所有对照使用共同提交阈值 0.45。特权教师可以在仿真内使用真实个体布局规划，交付由公开读图器和离线真值核对。

## 实验结果

| 独立的 96 趟闭环，使用同一个新配准读图器 | 双图都留对 | 连续三段都留对 |
| --- | ---: | ---: |
| 等时固定路线 | 57/96 | 40/96 |
| 公开历史规划 | 60/96 | 42/96 |
| 特权前瞻教师 | 56/96 | 40/96 |

教师实际做了 19 次入口探查。遮图复验中，恢复探查帧纠正 2 趟旧图，改坏 0 趟。另一批固定的教师路径上，换成新配准读者救回 4 趟，双图从 60/96 到 64/96。跨批次的双图收益仍不稳定。配对区间、错误计数和消融汇总见 [results.json](results.json)。

## 复跑

需要 Python 3.10。使用 [已交付的复跑包](https://chatgpt.com/library?file_id=libfile_fde3_f-2295ac965b048191ac661426415ac324_0d0b152ec7a08191b61f6c19d4af0a3e&account_id=personal) 安装冻结权重、图谱和骨遮挡先验。安装脚本逐项核对 SHA-256。需要重读历史轨迹或核对已保存的动作时，增加 `--with-records`。

```bash
cd NMI_idea/experiments/leg_registration_teacher_20261123
python -m pip install -r requirements.txt
python prepare_inputs.py /path/to/leg_registration_teacher_replay.zip --with-records

export LEG_SCAN_HANDOFF="$PWD/outputs/leg_scan_cli_handoff"
export LEG_BONE_PRIOR="$PWD/assets/bonehub_right_leg_occupancy.npz"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

study=work/leg_registration_teacher_20261123
python "$study/motion_eval.py" --split test --count 2 --workers 1 --draws 2 \
    --reader registered --modes fixed public verified_teacher --name smoke
python "$study/audit_followup.py" --name smoke --workers 1
```

复跑 96 趟闭环时，将 `--count` 设为 96，规划复验设为 `--draws 6`。原版双支读者的控制对照用 `--seed 604558341 --reader pair`，新配准闭环用 `--seed 177545318 --reader registered`。每个新运行用独立的 `--name`。重训入口包括 `fit_reader.py`、`fit_ranker.py`、`fit_album.py` 和 `fit_hypotheses.py`。时序模型按训练、开发和测试各生成一次独立缓存后，运行 `group_fit.py CACHE` 和 `group_eval.py --cache CACHE`。各路径在实验目录下保留原复跑结构。

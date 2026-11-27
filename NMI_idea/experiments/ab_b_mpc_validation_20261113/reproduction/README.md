# B 验证复现包

本包固定参考仓库 `NingMeng0903/RealUS_playground` 提交 `cdb1705a02f60382ffbe9e42f2ed420e27a28ddb`。开发代码位于 `benchmark/`。源仓库的实验、数据和控制代码保持原样。实验使用合成质心和采样固定滑轨关节执行模型。控制器在线更新图册与身份信念，并依据真实新包重新规划。

## 文件

- `development_a/` 包含两站新证据续扫的逐例记录、输入配置和配对统计。宽窄行程及不同观测判读共用同一病例。
- `development_b/` 包含参考探查对照的逐例记录、输入配置和配对统计。每例包含名义、拟合杂波、参考站信念三种条件。
- `historical/` 包含 768 条归档记录的独立重算结果，以及固定候选的事后成功上限和重采样诊断。
- `benchmark/clutter_fit.json` 记录拟合与验证种子、参数及简单的逐点校准诊断。它估计每个声窗类别的边缘杂波率和密度。
- `benchmark/development_protocol.md` 记录开发比较、停止条件和晋级标准。

## 重跑

在解压目录旁的指定位置放置原项目运行环境和机械资产，并在解压目录中检出 `RealUS_playground`。审查复算只需要固定版本、Python、NumPy 和 SciPy。新增闭环需要项目原运行环境中的 Torch、机器人模型和 dIRD 资产，以及 Numba。仓库中原有的机械资产加载代码使用 `/media/camp/EXT_DRIVE/RealUS_playground`。在其他机器复现时需提供等效的原项目资产与环境，配置和已完成的结果可以独立查看。

```bash
git clone --filter=blob:none --no-checkout https://github.com/NingMeng0903/RealUS_playground.git
git -C RealUS_playground sparse-checkout init --cone
git -C RealUS_playground sparse-checkout set NMI_idea rm75_control/rm75_control/assets/robots/rm75_6f_8dof rm75_control/rm75_control/control/joint_admittance_8dof rm75_control/rm75_control/kinematics
git -C RealUS_playground checkout --detach cdb1705a02f60382ffbe9e42f2ed420e27a28ddb

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python RealUS_playground/NMI_idea/experiments/ab_tro_y_continuation_20261112/audit_replay.py > frozen_replay.json
python legacy_headroom.py --n 128
python noise_headroom.py --n 24 --replicates 16 --workers 8

# 以下三条用带项目依赖的 Python 运行
python benchmark/benchmark.py --out rerun_a --split development --n 24 --workers 12 --preflight-cache rerun_preflight.npz --estimators nominal block
python benchmark/challenge.py --out rerun_b --split development --n 32 --workers 12 --audit-roots
python benchmark/reference_noise.py --out rerun_reference_noise.json

python benchmark/summarize.py rerun_a/cases.jsonl --out rerun_a/paired.json
python benchmark/summarize.py rerun_b/cases.jsonl --out rerun_b/paired.json
```

## 验收边界

调度试验要求两个指定站都出现新的正确身份图像，且通过两站运动与预算检查。搜索预测用接受与拒绝的状态和各时刻边缘概率。实测新包到达后重新运行共同滤波器。剪枝上界只针对声明的有限调度模型。

参考探查要求两段经过检查的实际模拟运动和终点的一张新正确身份图像。所有方法使用相同的质心生成器，其中持续伪结构也有定位抖动。参考信念省略被检结构邻域的一组历史观测。参考挑战的跨模型最小优势是模型评分。留视窗与参考信念是对相关证据的敏感性检查。拟合后的边缘杂波模型仍缺少跨帧联合噪声标定。

样本按独立病例配对。同一病例的重复采集、行程和估计器条件用于相应对照。预检图上的事后最优根动作使用评价标签，只用于量化机会。开发结果均保留，确认性种子未运行。物理接触、移动体模实拍、独立解剖几何和规划墙钟在运动期间的后果仍需实测。

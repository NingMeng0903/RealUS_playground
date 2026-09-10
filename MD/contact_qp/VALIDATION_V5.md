# V5 最终验证与移植增量

2026-09-10。本轮代码使用 GPT-6 medium；入口关键检查及回放证据解释使用 GPT-6 ultra。没有启动或重启硬件服务，没有执行真实扫描。

## 最后一次代码修改后的测试

| 检查 | 结果 | 范围 |
|---|---|---|
| `peirastic/tests -k 'not 100000'` | **571 passed / 3 deselected，21.99 s** | 包含新回放和配置检查；三个此前已通过的长序列不重复执行 |
| runtime config/runtime + 新旧 scan entry + scan path | **80 passed，14.20 s** | medium 完成最终组合；与上一行有重叠，不累加总数 |
| ultra 定向关闭 B1 | **29 passed，0.88 s** | 独立检查参数校验、构造前拦截，属于重叠定向复测 |
| 原 `run.sh record --help` | 通过 | 新参数可发现 |
| `run.sh record --simulate --auto-enter --contact-qp-config ...` | 12 条模拟 H5 | 配置 metadata 和 attempt 唯一路径；明确不执行 QP，不生成假的 QP 动作日志 |
| 全量同图特征 | 68 / 68 段，12,588 / 12,588 帧 | 行数、配置、源数据身份和输出 SHA 校验通过 |
| 全量 QP 静态决策 | 50,352 成功 | 4 N、合成几何、独立第一拍；非物理修复验证 |
| 68 代表帧力工况 | 1,632 成功 | 3.7/4.0/4.3 N × 两种名义输入 × 四组 |

JUnit：[peirastic_regression_v5.xml](peirastic_regression_v5.xml)。审核：[入口](review_scan_entry_v5.md)、[回放解释](review_replay_interpretation_v5.md)。实际效果、硬件 5 ms 全链实时性和物理无源认证仍不由上述软件通过替代。

全套短回归命令，工作目录为 RealUS_playground：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages" \
  LD_LIBRARY_PATH=/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix/lib \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/rm75/bin/python -m pytest -q \
  peirastic/tests -k 'not 100000' \
  --junitxml=MD/contact_qp/peirastic_regression_v5.xml
```

入口组合命令：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$PWD:$PWD/rm75_control:/media/camp/EXT_DRIVE/ICRA_YM/script:/media/camp/EXT_DRIVE/ICRA_YM/script/tests" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m pytest -q \
  peirastic/tests/test_contact_qp_runtime_config.py \
  peirastic/tests/test_contact_qp_runtime.py \
  /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_contact_qp.py \
  /media/camp/EXT_DRIVE/ICRA_YM/script/tests/test_scan_session.py \
  peirastic/tests/test_scan_path.py
```

## 本轮代码增量

| 位置 | 改动与用途 |
|---|---|
| `scripts/extract_contact_replay_features.py` | 冻结清单同 JPEG 的新旧特征批处理，核对缓存身份 |
| `scripts/evaluate_contact_qp_replay.py` | 真实图像输入的静态 QP 工况与原始逐次诊断 |
| `peirastic/contact_qp/runtime_config.py` | 生效 QP/energy 参数的离线检查；错误在机器人初始化前拒绝 |
| `peirastic/config/contact_qp/baseline_welleweerd.yaml` | 原控制器 + 新检测记录的入口配置，未冒充真实安装已标定 |
| `ICRA_YM/script/scan_session.py` | 可选参数、入口验证、每 attempt 独立日志与配置 metadata |
| `ICRA_YM/script/scan_robot.py` | 服务能力前置检查，原扫描函数的 baseline/shadow/active 路由 |
| `ICRA_YM/script/scan_simulation.py` | 接口兼容与明确的虚拟流程标记 |

相应测试分别在 `peirastic/tests/` 与 `ICRA_YM/script/tests/test_scan_contact_qp.py`。V4 的 QP 求解器、confidence 核心、能量账本、原 IK/native/rail 算法没有本轮行为修改。

移植在 V4 核心之上应用本轮增量；外部扫描入口属于 `ICRA_YM/script`，不能只复制 `peirastic/` 后以为旧 `record` 已支持新参数。`run.sh` 自身无需改动，原转发行为已接受新增参数。原三个扫描文件备份在 [scan_entry_v5_before](scan_entry_v5_before/)。

源码 SHA 与相对 V4 的变化列表保存在 [source_manifest_v5.json](source_manifest_v5.json)；[source_delta_v5.zip](source_delta_v5.zip) 包含本轮源码增量及外部入口，外部文件放在 `ICRA_YM/script/` 前缀下。已有无关脏 worktree 和参考库未改动。

结果报告与真机步骤分别见 [REPLAY_AND_MACHINE_V5.md](REPLAY_AND_MACHINE_V5.md) 和 [HARDWARE_TEST_RUNBOOK_V5.md](HARDWARE_TEST_RUNBOOK_V5.md)。

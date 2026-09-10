# Contact QP：从这里开始

当前方向：保持原 IK 封装，用同一个三变量外环输出 `v_cmd`；软件、几何和条件能量关系做独立验收，声窗动作效果由用户主动采集的真实闭环数据判断。不会根据合成图像结果持续调参或自动启动真人扫描。

| 需要查看的内容 | 文档 |
|---|---|
| 最新：50 mm 实际探头 active 执行命令与交付前检查 | [ACTIVE_PROBE50_RUN.md](ACTIVE_PROBE50_RUN.md) |
| 真实测量时基修复与限定 ultra 审核 | [SOURCE_TIMEBASE_V6.md](SOURCE_TIMEBASE_V6.md)、[审核](review_irregular_source_v5.md) |
| 最新：同批 68 段 / 12,588 帧新旧检测与 QP 决策结果 | [REPLAY_AND_MACHINE_V5.md](REPLAY_AND_MACHINE_V5.md) |
| 最新：使用原 run.sh record 跑 baseline / shadow / active | [HARDWARE_TEST_RUNBOOK_V5.md](HARDWARE_TEST_RUNBOOK_V5.md) |
| 最新入口修复、回归与源码身份 | [VALIDATION_V5.md](VALIDATION_V5.md) |
| V4 控制核心交付与能量证据 | [FINAL_HANDOFF_V4.md](FINAL_HANDOFF_V4.md) |
| 用户指定 Welleweerd 2020 的方法对应与配置边界 | [WELLEWEERD_METHOD_V4.md](WELLEWEERD_METHOD_V4.md) |
| 真人左侧缺失帧：原图、confidence、低可信度分区 | [预览图](welleweerd_v4_check/real_4.png) |
| 耗散漏扣修复与真实端口区间结算接线 | [ENERGY_D_ALIGNMENT_MEDIUM_V1.md](ENERGY_D_ALIGNMENT_MEDIUM_V1.md) |
| 用户指出的 confidence 全黑问题与修正对照 | [CONFIDENCE_FIX_V3.md](CONFIDENCE_FIX_V3.md) |
| 已接通的 active 软件链与剩余真机接口限制 | [ACTIVE_RUNTIME_MEDIUM_V1.md](ACTIVE_RUNTIME_MEDIUM_V1.md) |
| active 最终关键软件审核 | [review_active_software_v3.md](review_active_software_v3.md) |
| 哪些已经实现、哪些仍待验收 | [STATUS.md](STATUS.md) |
| 外环、IK、物理端口的边界；此前架构修正 | [ARCHITECTURE_REVIEW_V2.md](ARCHITECTURE_REVIEW_V2.md) |
| 仿真能说明什么；真实特性应怎样记录 | [EFFECT_IMPROVEMENT_REVIEW_V3.md](EFFECT_IMPROVEMENT_REVIEW_V3.md) |
| 最新真实特性记录与效果判定流程 | [REAL_STUDY_PROTOCOL_V3.md](REAL_STUDY_PROTOCOL_V3.md) |
| 原机器/被动记录入口、环境与 H5 关联 | [REAL_CAPTURE_ENTRYPOINTS.md](REAL_CAPTURE_ENTRYPOINTS.md) |
| 模块职责与移植依赖 | [PORTING.md](PORTING.md) |
| 运行适配的源码位置及最小改动 | [RUNTIME_ADAPTER_MAP_V2.md](RUNTIME_ADAPTER_MAP_V2.md) |
| 共享外环能量的 ultra 审核与问题关闭 | [review_outer_energy_v3.md](review_outer_energy_v3.md) |

`CONTRACT_V1.md` 和 `acceptance_v1.json` 保存原冻结契约与统计要求；后续版本文档明确列出修订，不覆盖旧证据。`design_v1/`、`design_v2/` 是指定合成模型下的历史实验与失败记录，不能当作真人声学效果。`human_audit/` 是既有真人记录的离线分析，也不能代替新控制器的闭环因果验证。

`experimental_native_v9/` 保存已撤出主链的 native 扩展，不属于新外环的移植依赖。具体代码和运行入口是否已就绪，以当前状态文件和对应的软件审核为准。

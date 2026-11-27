# RealUS / Anatomical GPS：给第三方 agent 的完整接手说明（2026-11-27）

请作为机器人超声实验的独立接手人，从下面的 Git 分支工作。目标不是把代理奖励或局部 F1 做漂亮，而是提高**同一时间预算内、没有错支交接、真正完成续扫**的概率。不要连接机器人，不要打开 Blender 留出，不要在确认集上调参。以下“已测”来自存档，不表示所有历史版本已在新机器复跑。旧手稿与旧交接里的“下一步”不是现在的放行指令。

## 1. 先定位：不是只克隆 `main`

```bash
git clone --filter=blob:none --branch handoff/realus-nmi-step3r-20261127 https://github.com/NingMeng0903/RealUS_playground.git
cd RealUS_playground
E=NMI_idea/experiments/anatomical_gps_unified_20261126
L=NMI_idea/experiments/leg_reader_teacher_scale_20261123
```

需要对原仓库有读取权限，不需要分享密钥。记录 `git rev-parse HEAD`。阅读顺序：

1. `$E/HANDOFF.md`：当前门、审查署名、正负结果、复现命令。`$E/PLAN.md`：现行协议和 Step 3R 预登记。`NMI_idea/migration/ORIGINAL_APPROVED_PLAN.md`：完整旧版总计划（T-RO → IJRR，含历史状态；不覆盖后来的实验记录）。
2. `$E/results/step3r_dev/{visible,tracking,rank_confirm}.json`：当前研究的原始逐段证据。`$E/results/step3_blind_v6/`：原来三个正式非 Blender 划分。`$E/results/step3_recovery/`：较旧 checkpoint/源码的恢复、对照和局限；不要用它覆盖 v6。
3. `NMI_idea/AB_RESEARCH_OVERVIEW.md`、`MD/NMI_IDEA.MD`、`NMI_idea/ideas/`：手稿与 A/B 的演化。`NMI_idea/migration/LEGACY_AGENT_PROMPT_20261126.md` 是**本轮 unified 实验之前**的详尽第三方说明（含逐版理论、腿部实验、文献索引及旧工作站路径）；其历史目录和链接不是“文件都随 Git 交付”的承诺。
4. `README.md`、`MD/other/COMMAND.md`、`rm75_control/MD/` 和 `ird_playground/README.md`：整个平台。原始研究分支/提交在下表；历史结果冲突时以对应冻结报告、逐例数据和统计单位为准，不以最近一段聊天为准。

## 2. 整个平台：哪些东西后来成为 Anatomical GPS 的前提

|层|迭代发现与边界|证据入口|
|---|---|---|
|术野、床、机器人坐标|相机内参 → AprilTag 多相机相对外参 → 手眼/地面/床角统一到世界系。`genesis_bundle.yaml` 是跨包接口；一台机器的标定不等于另一台机器已校准。|`camera_calibration/README.md`、`camera_calibration/calibration_results/`、`us_framegrab/`|
|观测、人体与解剖|RealSense/ZMQ、DWPose/EasyMocap → SMPL-X 姿态 → 材质绑定/LBS → Genesis/UE。分离坐标约定、配准、追踪队列及延迟。下肢 V17 仍有特定姿态出皮/体型 β 适配问题；渲染对齐并非临床定位准确。|`perception/`、`src/bridge/`、`src/projects/genesis_ue_sync/anatomy_retarget/`、`MD/smplx anatomy/anatomy_v17_generic_lower_review_20260909.md`|
|可达性、轨道与接触控制|Neural IRD 是可达裕量的代理，不是碰撞或安全证书，仍须完整位姿 IK/碰撞检查。RM75＋轨道由导纳扩展到慢轨/快臂、原生 WBC/QPIK、力补偿和 Peirastic 会话。控制器、仿体记录、离线回放和自由空间仿真不能充当本研究的实机闭环成功率。|`ird_playground/README.md`、`rm75_control/native/wbc_rt/`、`rm75_control/MD/peirastic_coupled_execution.md`、`peirastic/`、`MD/contact_qp/CONTRACT_V1.md`|
|多条工程历史|主工作区原 tip `3122a77d`；其未发布祖先含超出 GitHub 单文件上限的生成物，**没有直接推整条脏分支**。保留在本迁移中的是当前一方源码，以及 18 个提交的源码/文档邮件补丁和 3 个独立控制工作树的源码差异，并非把互斥控制器拼成一个程序。|`NMI_idea/migration/platform-history/`；已发布 `codex/controller-timeout-wrench-tilt-fix`、`codex/stage1-male-retarget-v4`|

若要复原历史差异，只在**新的临时工作树**操作。`8e85-live-homotopy.mbox` 的基点是 `a64c1f73f10fe6981c8d4e4f0394fbcdc67cb22b`（按序 `git am`）；`e85-hard-contact.patch` 和 `qpik-dbb-repair.patch` 各自独立应用到 `c1362750cd0a329a0aac05e351d35ff8c54d0c2c`；`peirastic-session-dof.patch` 应用到 `13f1c27f01b2d05f6a3874167e0940f666c9d91a`。`backup/force-control-baseline` 对应 `03a871ae`。不要在迁移 tip 上叠加这些补丁，也不要向硬件发送轨迹。

## 3. NMI 手稿、A/B 设计与实验不是同一套版本号

- **论文 V3.1–V4.1：** 从复用探头位姿转为复用“任务记忆”：材质地址、成像关系、连续可执行走廊和局部修复。一个地址可以对应多种机器人实现。**V4.2–V4.5：** 获取信息的动作也改变身体/机器人状态；看清目标不等于有时间或能力继续。
- **V5.1–V7.3：** 用未来的观测及执行查询定义后验该保留什么；证据必须对应到产生它的实现、出口和后继任务能力。有限技能库搜不到路径，不是全局不可达证明。**V8.1–V8.4：** 分清对象事实、请求和新完成记录；旧图可帮助定位，不能代替这次要求的新取像；短动作程序同时产生新证据、实际出口与耗时。
- **V8.5.x：** 把“提前释放旧证据能否节省后续代价”做成可验实验。V8.5.2 的 5,440 条模拟链来自 136 个配置单位，不是 5,440 个对象。部分预算有提前释放的收益，压力条件不稳定；关节执行见证为自由空间，不是接触或真实 B-mode。见 `analysis/v852-evidence-20261108` 的 `NMI_idea/experiments/v852/2026-11-08/third_party_review/`。
- **V9.1–V9.4 论文稿：** 患者后验的分型运输 → 未见对象的首次合格获取 → 相关先验、解析切片与条件观测响应 → 带退回/恢复的短技能提议。e3nn、KMP、ProDMP 是待比较或可替换的实现，不因为写进公式就证明优于简单基线。**论文 V9.x 不等于腿部策略 v9。**
- **11 月 A/B：** A 做局部观测/响应决策；B 加上对未来观察分支和后续任务价值的决策。不是 V8.5.2 验证包里另称 A/B/C 的三个问题。读/遮同一首探包、相同实际出口，在一个机制性截止上是 120/256 对 78/256；主截止各法却同为 78/256。移动 Y 续扫的 B 与强在线 MPC **同为 76/128 正确、16/128 误认**，逐例同动作。能证明“信息有时改变后续选择”，不能证明 B 普遍胜过强控制。见 `ab_closed_loop_20261111/ASTRA_CONCLUSIONS.txt`、`ab_tro_y_continuation_20261112/{REPORT.md,ASTRA_INDEPENDENT_STATS.json}`（均在 `NMI_idea/experiments/` 下）。

## 4. RL 和读者的真实历史：哪些训练过，为什么没有形成稳定优势

下表的 `/N` 是该实验的布局、轨迹或片段分母，**不是独立病人数**；各行不能直接纵比。表中的“交付”遵循对应旧协议，不能改称 unified 的 `continuation_completed`。

|实验|已测结果，及它不能证明的事|原始入口|
|---|---|---|
|膝局部 Y / ProDMP 离线 RL|膝部 256 个留出例，规则/FQI/CQL/离线 SAC 为 239/240/244/213；CQL 对规则 +1.95 pp，区间跨零，采集更耗时。另一批 ProDMP 离线 RL 对规则 84/96 对 82/96（4 得 2 失，p=.688），并多走路、多补采。局部改善不是全腿同预算优势。|`ab_knee_latent_rl_20261112/REPORT.txt`、`ab_prodmp_rl_20261112/REPORT.txt`|
|胆囊|反馈版 80 趟固定/手写反馈/学习排序/PSDP 为 45/54/48/49；独立复制上手写反馈优势不稳定。另一个 recurrent PPO 分支具备实现与合成 smoke，但缺 torso、SMPL-X、thorax 资产，**新解剖 rollouts=0，医学成功率未测**。|`ab_gallbladder_feedback_20261120/REPORT.md`、`work/gallbladder_rl/results/archive_audit.json`（相对该实验）|
|胫后注册、腿 v1–v3|注册实验同一读者的等时固定/公开历史规划/特权教师双图为 57/60/56，各 /96。旧 v1 实际跑过纯 PPO 和教师 PPO。v2 全腿四臂 0/24，但普通 MPC 在 **144/144 个区域**拍到真候选；逐区认错或弃权令整趟失败。v3 全程等交付量用较宽读图后再排序是 31→37→32/512，不能靠候选数量冒充识别能力。|`leg_registration_teacher_20261123/results.json`、`$L/leg_closed_loop_20261122_v1/`、`$L/leg_scan_target_routes_20261123_v2/{REPORT.txt,fixed_pilot/results.json}`、`$L/REPORT.txt`|
|腿 v4–v8|逐帧图关联会连错、并把错误向后传。v4 全程逐候选→图关联 105→76/512；在固定高层的另一对照中，复查轨迹改变却没有改变后继选窗。后续分别审计锚点、身份状态、公开证据和动作价值。|`$L/leg_scan_reader_feedback_20261123_v4/REPORT.txt`；v5–v8 各自的 `REPORT.txt`|
|腿 v9–v13|v9 **实际在线 PPO 2,304 趟**。两条独立复验带上，仅替换读者的 MPC 比旧读者 +9.81 pp；教师学生相对**同读者 MPC** 为 −3.74 pp，95% 区间 [−11.68,+3.74]。在同带择优再评分的增益 +.190，交叉带仅 +.038。v9 已有训练时真值 critic，并非后来首次提出。v13 学生原带 78/108，两复验带 67、71，对照 71、77；原带 30 个失败均曾拍到六区真目标。|`$L/leg_scan_full_route_20261124_v9/{REPORT.md,analysis/full_route_results.json}`、v10 恢复、v11 身份、v12 区域证据、`$L/leg_scan_student_history_20261124_v13/REPORT.md`|
|腿 v14–v16|v14 同轨只换末端 R6 读者：三带严格交付 78/67/71→79/68/71（各 /108），具名错支总数仍 50。v15 学生对 v13 的具名错支 63→46/324，但相对同接口冻结参照，两复验带严格交付 −2.31 pp，区间跨零；对 v13 原带平均耗时 94.9→155.4 s。71 是取证对数，**不是** 71 个初始错误。v16 是冻结 v15 的局部段风险实验，**不是新 RL**：自然组完整后验读包/遮包为 30/216 对 39/216，25/101 个模型内认证动作实际几何违规；该认证不是 split-conformal。|`$L/leg_scan_topology_reader_20261126_v14/`、`leg_scan_shared_belief_20261126_v15/`、`leg_scan_segment_risk_20261126_v16/`（后两项同样在 `$L`；各含报告和 analysis）|

原远端分支锚点（在 GitHub 上按分支名访问；或 `git show <commit>:<路径>`）：

|系列|分支|固定提交前缀|
|---|---|---|
|V8.5.2 证据，后续理论修订|`analysis/v852-evidence-20261108`|`8cbd9b2e`|
|B/MPC 独审|`review/b-mpc-validation-20261113`|`36dd120c`|
|胆囊反馈；胆囊 belief-RL|`experiment/gallbladder-feedback-20261120`；`experiment/gallbladder-belief-rl-20261120`|`10132503`；`cbb5e9a0`|
|腿注册；扩量教师；读者反馈|`experiment/leg-registration-teacher-20261123`；`experiment/leg-reader-teacher-scale-20261123`；`experiment/leg-reader-feedback-20261123`|`eea9fae4`；`22d1f9c8`；`2a5bbd5a`|
|腿 v14；v15；v16|`experiment/leg-topology-reader-v14-20261126`；`experiment/leg-shared-belief-v15-20261126`；`experiment/leg-segment-risk-v16-20261126`|`24b17882`；`e8837c83`；`5e82732a`|

## 5. 当前统一版：Gate 2 PASS，Gate 3 FAIL，Step 4 未运行

**Gate 2。** 将原本不一致的 SMPL-X β、皮肤/骨走廊与髋膝图谱统一到 `beta=0`，从世界 XYZ 加噪改为骨与皮肤之间的材质坐标扰动。509/512 个扰动在独立检查下有效，保持了有意义的位移。物理核只给匿名回声。只有一个约 90 mm 近端 CT 参与者，不覆盖腘窝 S5；BodyParts3D 缺 DFA；无临床 NYSORA 人工站点标签。不能填造解剖或把多颗随机种子当成多位患者。见 `$E/results/step2_training_support.json`。

**Gate 3 分开看两个问题。** (1) 原任务给重叠窗口强制一个请求标签，按原采样律即使知道来源和真实高度，Open3D 的总体宏 F1 上界约 .753、CT→图谱约 .789，低于 .85/.80；不是说有限样本或所有临床任务都有这个上界。改评“同一帧确实看到了哪些角色组合”后，在独立**训练来源**诊断中，回声头 .795、纯位姿 .353；不是原标签 Gate 通过。(2) 即使拍全仍有真关联、位姿和几何缺失。v6 正式三折失败；Open3D 40 mm 身份 F1 .442，动态身份 F1 .460，缺失 25/347 个几何机会，条件 RMSE 20.66 mm，完整 RMSE 不可估。原划分和门槛仍在 `$E/PLAN.md`。

**一次锁定的修正已经被否决。** v6 → 将候选按同一父先验的积分证据提前排序；不变更观测、束宽及校准器。731903 站点 80 段：拍全 48→48，精确认全 18→17，同时 <6 mm 13→13，再通过整段旧核验 0→0。两版各 78 段首个失败原因是身份候选非唯一或为空，2 段无管腔。931902 动态 36 段，整段身份全对且所有渲染中心 <6 mm **3→3**；两臂均只测得 494/495 个中心，条件误差 11.48→9.63 mm 不能抵消整段未改善。Sol Max 独审接受的是负结果，Astra Max 逐段复核；默认保留 v6。0/80 意味着没有可估的“已接受段上的风险”，也**不是 RL 成功率为零**。

## 6. 值得继续检验的想法（不是已证实的改进）

1. **先证明动作确实能挽回一次交付。** 当前认证读取逐帧当时的在线快照，后续证据另走 `relabel_history()`；只提高单帧 F1 不一定改变早期错误。用**新的训练来源**轨迹配对一次允许的换窗、压迫或低力停留复查与等时对照；比较即时快照和截止时刻的历史重判。不得用提交后的帧；两套决定规则分别独立校准，整段标准不降。保留空段、漏中心、纠正和反向改坏的分母；报告是否多出正确认证，而不是仅报后验更自信。731903/931902 保持封存，不拿来选变体。
2. **不要让 RL 为不可观察状态或没有用的动作菜单买单。** 已观察到：拍到 ≠ 认对；降低局部 RMSE ≠ 完成全程；同带选教师再打分偏乐观；允许更多补拍常增加时间与路程。待检验而非已证实的原因包括稀疏的多站乘积回报、观测对称性、信号在训练/执行分布间不稳定。先比实际复查、等时补拍和直接继续，再决定值得让策略学什么。
3. **Gate 3 通过后才比较控制策略。** 冻结同一读者；强信息增益 MPC、固定复查和 RL 用相同可执行技能、时间预算及配对噪声；被动 MPC 的权限差异单列。公开 actor 禁用真值；训练专用的历史＋真值 critic、按站信用分配和潜势塑形要分别消融，不能拿旧 v9 的特权 critic 当作新方案的因果证据。分别报告 `target_captured`、`identity_correct`、`continuation_completed`、时间、路程、错支和错误认证。最后一项成功事件要求限时内拍全、认全、无错支、沿段误差 <6 mm、整段有效认证。T-RO 先行，IJRR 的四格具身充分性/不可逆截止实验在后；不得跳门。

## 7. 什么在 Git，什么必须另外取得

- **迁移树内：** 第一方源码、测试、配置、论文/技术文档；保留 `main` 已有论文，不把另一工作树中消失的旧稿当删除指令；补入实验分支的源码、报告及选定配对证据、本机以前交付的腿部源码/报告与 v1/v2 关键收据；本轮统一版原始 JSON、引用的测试日志和旧源码归档。`NMI_idea/migration/WORKSPACE_FILES.sha256` 对应工作区清单；`IMPORTED_GIT_BLOBS.tsv` 每行是原提交、原 Git blob 和路径；`RECOVERED_FILES.sha256` 对应另行补入的旧交接和总计划。不是所有历史版本的同名文件同时成为默认实现；查原样版本用原分支。Git 能公开给已有权限的协作者，不代表仓库已设为公开。
- **不在本次上传：** 原始 CT/医疗数据、受许可的 SMPL-X 模型、现场图像/视频、用户会话与凭证、整套外置盘、环境、编译缓存、大量逐动作渲染/日志。按本仓库 `MD/GITHUB.md` 的规则不提交本地 `*.pt`；**SHA 列表不是权重文件**。7 个旧/current reader checkpoint 的 SHA 在 `NMI_idea/migration/LOCAL_CHECKPOINTS.sha256`；当前训练来源 LOAO_OPEN3D v6 是 `88fbd7f90b358788ef8a77bd50a6521dba3697e5635265777e16a0742836dc21`。向数据/模型所有者通过获准渠道取回并校验，或按该版本流程重新拟合并记新 SHA，不得伪称同一 checkpoint。
- **复跑核对入口：** `$E/sim/anatomy_assets.py` 列出 `ct_geometry.npz`、`medseg_native_graph.json`、`ct_reference_topology.json.gz`、注册表面、骨占据场和中性皮肤；`$E/sim/smplx_dynamic_atlas.py` 指向外部 `Among_US` 的 `SMPLX_MALE.npz`、`source_manifest.json` 及 `shaft_roi_v2` 的密封图谱/节点附着。克隆后路径不自动迁移；不要用空数组、不同 β 或另一个患者顶替。统一实验是合成匿名回声证据，不是临床 B-mode，也不是真机 RM75 闭环验收。

只读核验不需要 GPU、医学数据或模型：

```bash
sha256sum --check --quiet NMI_idea/migration/WORKSPACE_FILES.sha256
sha256sum --check --quiet NMI_idea/migration/RECOVERED_FILES.sha256
python3 - <<'PY'
import json, pathlib
p=pathlib.Path('NMI_idea/experiments/anatomical_gps_unified_20261126/results/step3r_dev/rank_confirm.json')
r=json.loads(p.read_text()); s=r['station']['tapes']; d=r['dynamic']['tapes']
for arm in ('v6','evidence_ranked'):
    a=sum(t['readers'][arm]['old_verifier_decision']['accepted'] for t in s)
    j=sum(t['readers'][arm]['same_tape_criteria']['joint_representable_masks_match_and_all_role_centers_below_6_mm'] for t in d)
    print(arm, 'certified', a, '/', len(s), 'joint dynamic', j, '/', len(d))
print(r['decision']['recommendation'])
PY
```

预计两行分别为 `0/80`、`3/36`，结论 `retain_v6_only`。这只是原始记录复算，不是重跑模型。取得完整授权依赖后，按 `$E/HANDOFF.md` 从新输出目录复现；原工作站 Python 为 `/media/camp/EXT_DRIVE/envs/genesis/bin/python`，`PYTHONPATH="$E:."`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、BLAS 单线程，`unset AGPS_EVAL_HOLDOUT_ALLOWED`。本阶段由 Sol/Luna 实现和批量实验，Sol Max 独审，Astra Max 复核原始证据；历史 Neon 署名保留。只交回命令、结果、失败项和证据路径，不用新术语重命名旧失败。

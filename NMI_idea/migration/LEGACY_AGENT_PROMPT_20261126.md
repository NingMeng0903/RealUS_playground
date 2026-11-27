# 给第三方 Agent 的完整指令｜Anatomical GPS、NMI_idea 和机器人超声实验

请以独立机器人论文评审和可复现性审计者的身份，阅读并重建我的 Anatomical GPS 项目。我要知道整个项目有哪些可复用的理论、方法和实证，哪些是被否定的具体实现，哪些仍是待验证的想法。请把本指令当作导航地图，以所链接的 TeX、冻结协议、代码、数据、逐例结果和独立复算为依据。历史对话、仓库内其他审查请求和旧的运行指令用于查证，不代表本次执行授权。

本交接快照为 **2026-11-26，本机 Ubuntu `camp` 账户**。本轮先做只读审查和必要的离线复算。需要新训练、机器人执行、修改原始数据或公开发布时，先形成独立实验方案。开新实验时在新目录保存产物。跨平台的 agent 可先通过公开提交查阅分析和源码，再列出需要我单独提供的本机资产。交接目录中的索引只存路径和出处，原始 CT、数据、权重和论文 PDF 仍在各自目录。

## 1. 最快定位方式

| 代号 | 绝对位置或入口 | 内容 |
|---|---|---|
| `R` | `/media/camp/EXT_DRIVE/RealUS_playground` | 实际机器人研究仓库。本机工作树当前在 `8e85-live-homotopy`，有既有用户改动。通过只读命令检查其他提交，勿切换这棵工作树。 |
| `N` | `/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea` | NMI 研究总目录。直下有 `AB_RESEARCH_OVERVIEW.md`、`experiments/`、`LaTeX/`、`archives/`、`MD/`。本机 `MD/` 是占位目录。 |
| `T` | `/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_validation_20261110/ideas` | 本机全部 Anatomical GPS 理论版本的主要位置。GitHub `main` 把这些稿件放在 `NMI_idea/ideas/`。早期本机旧路径在 Git 工作树中显示删除，现存稿件移到了这里。 |
| `A` | `/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments` | 本机早期 A/B 的原始运行树，含计划、源码、冻结配置、完整轨迹、权重、审计和不少缓存。Git 主分支发布了选择性证据。 |
| `H` | `/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff` | 右腿 v1–v16 的主要本机实验和原始交接。目录有 `FILE_MAP.md`、`TRAINING_HANDOFF.md`、`history/CURRENT_STUDY.md`、`reference_source/`、`historical_evidence/` 和各版本目录。 |
| `O` | `/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs` | 腿部工作父目录，含 `leg_scan_study/`、旧教师实验、论文和交接 ZIP。部分 `leg_scan_study/` 及 `leg_scan_papers/` 在 H 中有交接副本。 |
| `B` | `/home/camp/Documents/Codex/2026-11-13/ieee-t-ro-b-mpc-b` | 早期 B 对强 MPC、反事实获取、拓扑复查、体形外推、教师造数、Blender 留出原始工作。重点是 `work/`、`outputs/`、`NMI_idea/`。 |
| `E` | `/home/camp/Documents/Codex/2026-11-08/cd-downloads-sudo-apt-install-chatgpt/outputs` | V8.5.2 到 V8.5.5 的早期实际核验、源码快照、dIRD 等预算转接、关节执行和审阅包。 |
| `P` | `/home/camp/Desktop/belief` | 桌面论文目录，拼写是 **`belief`**。29 篇 PDF 和两份来源 JSON。某个 PDF 的文件名含真实换行符。见文献章节及 `paper_inventory.tsv`。 |
| 论文交接包 | `/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/leg_scan_papers` | 三份补充 PDF 和开放原文清单，`README.md` 记录版本、正式出处及当时无本地副本的近邻论文。 |
| 主论文 Overleaf | `/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/LaTeX/Anatomical_Geometry_Online_Belief_Overleaf` | `main.tex`、`references.bib`、`main.bbl`、`figures/`、`compile.sh` 和 `LOCAL_BUILD.txt`。整体用 XeLaTeX。压缩包在 `/media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/archives/Anatomical_Geometry_Online_Belief_Overleaf.zip`。独立版本 TeX 与这套 Overleaf 工程分别引用。 |
| 本交接与索引 | `/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/third_party_agent_handoff_20261126` | 本文 `ANATOMICAL_GPS_AGENT_PROMPT.md`。`COPY_TO_AGENT.sh` 输出完整本文。`SOURCE_ENTRY_POINTS.md` 定位主证据。`theory_formula_index.tsv` 是 TeX 标签，`tex_display_math_index.tsv` 是展示公式原文与行号。`file_inventory.tsv.gz` 是已声明本机树的完整文件级索引。`git_inventory.tsv` 和 `published_entry_points.tsv` 保存固定远端版和报告。`archive_members.tsv.gz` 定位旧 ZIP 内的文件。 |

本机另有 NMI 工作副本，用于定位当年没进主工作树的分支。B 下的 `NMI_idea/`、`work/RealUS_b_teacher_fit/NMI_idea/`、`work/RealUS_playground/NMI_idea/`、`work/RealUS_probe_program/NMI_idea/`、`work/RealUS_teacher_gen/NMI_idea/` 和 `work/topology-release.XgOBqB/NMI_idea/` 是历史副本。早期 V8.5.2 发布暂存为 `/home/camp/Documents/Codex/2026-11-08/cd-downloads-sudo-apt-install-chatgpt/work/publish_v852/stage/NMI_idea`。胆囊另有 `/home/camp/Documents/Codex/2026-11-20/new-chat/work/gallbladder_belief_rl/NMI_idea` 和 `/home/camp/Documents/Codex/2026-11-20/new-chat-3/work/RealUS_playground/NMI_idea`。优先用各发布分支的加锁 commit 和报告复原当时状态，副本里的同名实验不能按多次独立试验计数。

### 1.1 已核实的 GitHub 仓库、分支和固定提交

- [公开仓库 RealUS_playground](https://github.com/NingMeng0903/RealUS_playground) · [主分支的 NMI_idea](https://github.com/NingMeng0903/RealUS_playground/tree/main/NMI_idea) · [main 固定提交 cdb1705a](https://github.com/NingMeng0903/RealUS_playground/commit/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb)。主线检查入口是 [早期研究总览](https://github.com/NingMeng0903/RealUS_playground/blob/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/AB_RESEARCH_OVERVIEW.md)、[理论稿目录](https://github.com/NingMeng0903/RealUS_playground/tree/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/ideas) 和 [B_REVIEW_REQUEST.md](https://github.com/NingMeng0903/RealUS_playground/blob/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/B_REVIEW_REQUEST.md)。
- [V8.5.2 审阅分支](https://github.com/NingMeng0903/RealUS_playground/tree/analysis/v852-evidence-20261108) · [固定证据目录 8cbd9b2e](https://github.com/NingMeng0903/RealUS_playground/tree/8cbd9b2ee65cff0c4017affc1281d47941549c9d/NMI_idea/experiments/v852/2026-11-08/third_party_review)。带源代码快照、`MANIFEST.json`、`KEY_METRICS.json`、训练记录和复验。
- [B 对 MPC 审阅分支](https://github.com/NingMeng0903/RealUS_playground/tree/review/b-mpc-validation-20261113) · [固定 git 36dd120c 的实验目录](https://github.com/NingMeng0903/RealUS_playground/tree/36dd120c1261c8ff44ed61ada8f8ce115e5628f7/NMI_idea/experiments)。内容涵盖移动 Y 之后的逆向获取、拓扑复查、恢复、教师提名和 PSDP。
- [配准教师，原始胫后静脉实验](https://github.com/NingMeng0903/RealUS_playground/tree/eea9fae44bfac02234bdd407a758cb210731daaa/NMI_idea/experiments/leg_registration_teacher_20261123)。分支 `experiment/leg-registration-teacher-20261123`。
- [扩量读图与教师 v3 固定目录](https://github.com/NingMeng0903/RealUS_playground/tree/22d1f9c8c1e048f64ba6546c166c0b748e9d3893/NMI_idea/experiments/leg_reader_teacher_scale_20261123)。分支 `experiment/leg-reader-teacher-scale-20261123`，也含 v1 与 v2 的入库子目录。
- [v4–v13 延续分支](https://github.com/NingMeng0903/RealUS_playground/tree/experiment/leg-reader-feedback-20261123) · [固定 v13 总目录](https://github.com/NingMeng0903/RealUS_playground/tree/2a5bbd5ad92dd00f3f3d3642279d6c198525a75f/NMI_idea/experiments/leg_reader_teacher_scale_20261123)。版本段落各给自己的精确提交。
- [v14 分支](https://github.com/NingMeng0903/RealUS_playground/tree/experiment/leg-topology-reader-v14-20261126) · [v15 分支](https://github.com/NingMeng0903/RealUS_playground/tree/experiment/leg-shared-belief-v15-20261126) · [v16 分支](https://github.com/NingMeng0903/RealUS_playground/tree/experiment/leg-segment-risk-v16-20261126)。三者均保持 `NMI_idea/experiments/leg_reader_teacher_scale_20261123/<版本目录>` 的发布路径。
- [胆囊 belief-RL 分支，cbb5e9a0](https://github.com/NingMeng0903/RealUS_playground/tree/cbb5e9a05a836c5fbb1ebfb8af646d4927811c07) · [胆囊反馈分支，10132503](https://github.com/NingMeng0903/RealUS_playground/tree/101325037dcb9c899dd4553245f5aa505a9caab9/NMI_idea/experiments/ab_gallbladder_feedback_20261120)。其他器官的结果须保留各自任务定义，不能并入腿部六区指标。

`git_inventory.tsv` 有完整 SHA、分支和内容入口，包括验证旧分支前后差的关键提交。远端入口已核对，挂载磁盘上的大型原始资产、矩阵、训练分片和逐时钟轨迹只有相应本机生产目录可用。公开分支的校验或压缩分析包仍可做很多逐布局复算。

### 1.2 挂载资产、环境和早期扫描输入

以下是查证覆盖和完整复跑时才读取的本机输入。路径的角色可能仅为历史对照。

| 资产 | 路径和角色 |
|---|---|
| 血管概率云 | `/media/camp/PEI_T7/SMPLX_Vascular_Probability_Cloud`。对齐曲面 `aligned_vessel_surfaces.npz`，Open3DModel 更新曲面投影 `right_leg_joint/shared_field_test/right_leg_open3dmodel_shared_field.npz`，骨声影参考 `right_leg_joint/bonehub_right_leg_occupancy.npz`。|
| 原近端 CT | `/media/camp/PEI_T7/Leg_Vascular_Reference_Extraction/medseg_ct/groin_labels.nii.gz`。只读。标签 23、39 的原生 31 层连续，3 mm 采样，首末层中心跨度 90 mm。已配准 5 mm 网格在 −480 至 −560 mm 的 17 个节点连续。不能作为大腿远端或全腿路线标注。|
| Blender 留出 | `/media/camp/PEI_T7/1_team meeting/atlas_ppt_build/assets/blender_topology_holdout`。`geometry.npz`、`summary.json`、`transfer_checks.json` 和导出的原扫查。源 Blender 骨与中心线资产在 `/media/camp/PEI_T7/SMPLX_Vascular_Retarget_Audit`。Blender 旧面板只有一个独立解剖。新噪声带重复同一个体。|
| 图谱打包及拓扑 | `H/leg_scan_study/topology/README.md`、`ct_geometry.npz`、`medseg_ct_route.npz`、`blender_holdout_geometry_links.npz`。训练、近端原标签、只评估留出各用各的资产。|
| 原 216 条和 Blender 160 条 | `B/work/leg_reference_transfer/blender_holdout/{scan.py,geometry.py,export_dataset.py}`。同目录 `student_transfer.py`、`continuous_geometry.py`、`training_audit.py`、`exit_continuation.py`。|
| SMPL-X 与执行 | `/media/camp/EXT_DRIVE/Among_US/ref_code_library/InteractVLM/data/body_models/smplx/SMPLX_MALE.npz`。原机器人、运动与 dIRD 在 `R/rm75_control`、`R/ird_playground/ird_playground/ird`、`/media/camp/EXT_DRIVE/Among_US/DMP_playground`。早期探查基函数见 `B/work/RealUS_b_teacher_fit/NMI_idea/experiments/ab_leg_bifurcation_20261112/learned_prodmp.py`。|
| 本机 Python 与设备自检 | `/media/camp/EXT_DRIVE/envs/genesis/bin/python`，`H/gpu_preflight.sh`。历史 GPU 报 RTX 4080 Laptop、PyTorch 2.12.0+cu126、驱动 580.178.04。设备、数值和依赖以各版本 `gpu_preflight.json` 与本机实际环境为准。`/tmp/right-leg-render-deps` 是历史附加依赖目录，未必长期存在。|

## 2. 先掌握核心对象，再看版本

把问题说成一句话：**机器人怎样在身体变形、血管身份不明、不同探头末态有不同续扫能力时，保存可复用的对象证据，并选择能完成后续检查的动作？** 推荐读稿顺序是 V9.1→V6.1→V8.2→V8.3→V8.4→V9.3→V9.4，再对照 v10、v14、v15、v16。它们分别处理坐标与成像几何、执行风险、检查接口、统计充分性、程序后果、首次搜索、恢复，以及具体仿真结果。

### 2.1 身体地址、几何类型、执行位姿

设 `xi=(theta,h,d)` 为人体材料地址，`Psi_t(xi)` 为当前物理位置，`F=D Psi_t`。`theta` 是运输的周向角，`h` 是平滑轴向站位，`d` 是参考皮肤与内壳之间图 Laplace 的离散 Dirichlet 场。稿件写法可能交换 `h` 与 `theta` 顺序。实现要核对数组实际索引。深度 `d=1` 对应内壳，身体中心线在所验证环域之外。可观测血管内移、组织变化和配准误差留给个体后验。生产审计覆盖 `h∈[0.14,0.94]` 和七个离散膝角，不代表全身全域。

\[
v'=Fv,\qquad n'\propto F^{-\top}n,\qquad \Sigma'\simeq F\Sigma F^\top.
\]

切向和旧平面法向的运输不同。旧短轴法向和当前重建短轴同向，当且仅当 `F^T F t ∥ t`，其中 `t` 是参考血管轴。若要求所有切向都成立，就需要 `F^T F=λI`。剪切反例展示地址对应成立时切面仍会错。协方差式在确定仿射下精确，一般形变下是一阶近似。刚体共变换的等变性、跨姿态地址对应和跨个体学习是三项独立验证。

探头标架有图内横向 `r_x`、出平面 `r_y` 和深度 `r_z`。V6.1 的短轴完整几何残差为

\[
h_z(q)=\big[r_x^\top(c_z-o),\ r_y^\top(c_z-o),\ r_x^\top t_z,\ r_z^\top t_z\big]^\top.
\]

局部切向标架用可观察的表面接触点、法向和有向任务方向构造。给定实际标架 `F_hat_t`，局部运动参考 `U(t)` 变为 `T_ref=F_hat_t U(t)`，移动标架的速度还要包含 `dot(F_hat_t)` 项。ProDMP 用积分基和边界条件生成短段，常见写法 `p_C(t)=phi_bc(t)+Phi(t)w`。dIRD 查询或排序允许实现。连续 IK、接触、速度、碰撞和动态续接仍由独立运动验收负责。主腿部 v2–v16 多数实验 dIRD 关闭，v10 另存虚拟排序运行。见 `T/Anatomical_GPS_V3.1.tex`、`T/Anatomical_GPS_V9.1.tex`、`T/Anatomical_GPS_V6.1.tex`，以及 `A/ab_validation_20261110/repair_20261110/audit/FINAL_HARMONIC_AUDIT.txt`。

### 2.2 局部观测、后验、持久记忆和拓扑先验

A 表示学一个可迁移局部动作响应。解析几何给出已知项，来自实际探头动作的观察用来学习响应残差。早期可复现的标量近似是

\[
Y_a=g_{\mathrm{geo}}(\Psi_t(\xi),a)+H_\xi(a)w_\xi+\epsilon_a,\qquad H_\xi(a)=[1,s_a].
\]

`s_a` 是探头相对当前标架的有向小倾角，`w=(b,beta)` 包含共享偏差和斜率。两个不同倾角的差分能消去共同偏差。任务比较若取 `d=H_u−H_v`，新观测对差值均值的线性高斯更新系数为 `d C H_a^T/(H_a C H_a^T+R_a)`。系数为零时，观测信息不能改变这个方向的估计。B 使用当前后验推演接下来可执行的多个程序。

静态、关联已知、条件独立的高斯子类中，唯一事件集合 `E_xi` 的累积后验以自然参数保存：

\[
(\Lambda,\eta)_\xi=(\Lambda^-,\eta^-)_\xi+
\sum_{i\in E_\xi}(H_i^\top R_i^{-1}H_i,\ H_i^\top R_i^{-1}(Y_i-g_i)),\qquad
C=\Lambda^{-1},\quad m=C\eta.
\]

旧图应保存材料地址、帧键/候选键、实际位姿、时间、观测与标定版本、关联解释。相关证据使用联合模型或显式折扣。发生重配准、接触模式改变、暂态或身份改判时，更新、替换或重新读相应证据。V8.2 把 `Update(证据→对象后验)`、`Query(对象后验→当前检查要求)` 和 `Realize(任务意图→合法机器人实现)` 分开。实际「完成」还需记录合格图像和覆盖。旧短轴证据可以帮助新长轴定位，新长轴仍需要自身的合格图像。

动态原型用 `X_t(xi)=g(xi,u_t)+H(xi,u_t)w_s+L(xi,u_t)z_t` 区分长期个体偏移 `w_s` 和随运动衰减或变化的暂态 `z_t`。同膝角同角速度可以对应不同内部历史。配对结果应检验是否真的提高位置预测或交付，不能由状态模型存在直接推出。见 `A/ab_knee_matched_20261112/PAPER_CORE.txt`。

腿部公开先验使用材料站、源图谱、几何支持、同源延续和血管角色。邻接距离本身不生成连接。公共估计器在 3 个图谱来源各 9 条有限几何假设上更新，代码近似为 `w_t(k) ∝ [(1−eta) w_(t−1)(k)+eta w_0(k)] × L(o_t|k,x_t)^kappa`。不同声窗、观测质量和重复度改变更新强度。该几何权重、v15 对象图的角色 softmax、动作的整趟价值是不同量。估计器的公开输入是匿名点回波、位姿和历史，隐藏标签只参与仿真、训练监督和验收。见 `H/leg_closed_loop_20261122_v1/src/{motion.py,scan_kernel.py}`、`H/leg_scan_target_routes_20261123_v2/{atlas_projection.py,route_rollouts.py}`。

### 2.3 哪些后验和动作后果必须一起记

V8.3 的严格理论固定一次信息实验。证据 `y` 出来以后可选择一个真实可达模式 `m`，联合分布 `Q_pi(y,m|theta)=Q(y|theta)π(m|y)`，耦合 `mu_(y,m)=p_y π(m|y)`。普通统计信息可以相同，完成价值 `sum_(y,m) mu_(y,m) c_R(y,m)` 依赖证据与构型的配对。给定可达支持与固定边缘时，所有允许耦合的价值相等，等价于价值表在可变支撑上的加性分解 `c_R(y,m)=alpha_y+beta_m`。交替环对比提供可检验判据。多个请求所需的最小线性关联维数可写为 `rank(C^T D)`，`C` 取可变耦合差空间的基，`D` 列出请求能力表。结果有明确有限模型和可达性前提。见 `T/Anatomical_GPS_V8.3.tex` 的 `eq:joint`、`thm:separable`、`thm:cycle`、`eq:rank`。

V8.4 的任务模型直接学习一个合法短程序的联合后果：

\[
K_a(dw',dz,dx',d\tau\mid w,x).
\]

`w` 是对象状态，`z` 是新公开观测，`x'` 是实际出口，`tau` 是程序时间。更新用同一核预测 `o=(z,x',tau)` 并求 `b'(dw')=Pr(dw'|b,x,a,o)`。后续价值在实际 `x'` 的可行集合求，不在名义计划端点求。一个有限递推是

\[
V_{R,h}(b,x,r)=\max\left\{ g_R(x),\ \max_{a\in\mathcal A(x)}\int_{\tau\le r}
V_{R,h-1}(b'_a,x',r-\tau)\,\bar K_a(dz,dx',d\tau\mid b,x)\right\}.
\]

理论稿里 `r` 表示声明任务截止或资源。腿部后期为了优先正确率，没有新增时长或里程配额，仍保留实际接触、速度、可达性和报告的时间/距离。旧有限记忆实验和 v16 还有各自协议。对执行核的逐步误差，核验 V8.4 的总变差累积界 `delta_H`、对最优值和固定策略值的条件保证。V8.5.2 对条件核已知而对象混合未知的有限子类，给出 `span{c_(q,π)} ⊆ range(G^T)` 的任务辨识条件。一般模型的策略置信需要共同可部署策略及独立校准。V8.5.3/4 区分「压缩无法同时预测不同条件后果」与「这些历史没有共同获胜策略」，前者不能推出后者。见 `T/Anatomical_GPS_V8.4.tex`、`T/Anatomical_GPS_V8.5.2.tex` 和 `T/V8.5.4_execution_sufficiency_20261108/`。

B 的早期动作价值可以写成 `−E[c(a,O)] + E_O[max_{u∈F(x',r−tau)} J_fresh(u|b_after,x',r−tau)]`。强在线 MPC 在初始评估时用一个预先选的尾段，真实看到新观测后仍会重规划。一般的 `E max` 与 `max E` 关系已有先例。项目需要检验的是，**材料地址下的新增证据是否实质改变实际可达尾段的排序和最终检查**。如果公共候选共用同一最优尾段，或者本轮各种策略都离开到同一出口，观察信息或出口边缘变化可能对任务没有可测增益。见 [B 审阅请求](https://github.com/NingMeng0903/RealUS_playground/blob/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/B_REVIEW_REQUEST.md)。

### 2.4 概率切面判据及其适用边界

V6.1 在后验下计算运动变化的任务失配。若 `A(z)` 是任务残差对候选实现的局部导数，二阶量 `G_t=E[A^T W A|D_t]` 给出 `E[||A(z)delta_q||_W²|D_t]=delta_q^T G_t delta_q`。实现验收要含名义残差、偏心、有限段曲率或 Taylor 余项，并明确几何后验、校准和物理约束。v16 真正用的是有限支撑下逐假设整段通过权重，未知质量不计为通过，模型门 0.80。V6.1 的矩与 Markov 形式在 v16 作为额外诊断，未通过本轮所设认证门，控制器用的是有限支撑形式。详见 v16 条目。这个区分决定你能否直接把 TeX 中的保证用于仿真结果。

## 3. NMI 理论稿逐版读法

以下文件都以 `T` 为前缀。除 V8.5.3/4 明示子目录外，每个版本的原稿就是 `T/Anatomical_GPS_<版本号>.tex`。公开同版固定目录见 [cdb1705a 的 NMI_idea/ideas](https://github.com/NingMeng0903/RealUS_playground/tree/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/ideas)。公式标签和绝对文件在 `SOURCE_ENTRY_POINTS.md`，全部 `label` 的行号在 `theory_formula_index.tsv`。稿号 **V6.1** 与腿部实验 **v6** 是两个编号系统，其他相同数字也一样。

| 稿件 | 主要新增设计与公式入口 | 可用证据和待审查的限制 |
|---|---|---|
| **V3.1** | 从不变材料地址到允许短轴视角族。把内部目标中心线、皮肤交点、探头姿态和 IK 联成隐式提升，检查有限走廊再用样条 SQP/QP 给整段参考。看 `eq:metriccovector`、`eq:viewmanifold`、`eq:lift` 和有限盒定理。 | 材料、入射和度量的类型区分可从数学核查。可行结论依赖横截性、局部条件数、约束和有限走廊证书。采样点 IK 成功不认证点间连续运动。 |
| **V3.2** | 维护可执行记忆。先运输旧解码器，再计算剩余任务缺陷，分块估计需重算部分。`eq:repair`、`eq:secondorder`、`eq:influence`。 | 选择性保留要求依赖关系完整、端点和走廊连续。缓存节省和减少重新规划的实效要由同一任务跨形变比较。 |
| **V4.1** | 沿几何变化运输整个任务等价实现族。按物理代价选横向提升，dIRD 引导允许的冗余运动，切锥和对偶表示局部方向障碍。`eq:horizontal`、`eq:activecone`、`eq:barrier`。 | 运输依赖路径、正则性、速度和可行域边界。相同终点几何可能产生不同出口。稿中的实施级假设要与原生轨迹分别验。 |
| **V4.2 原稿** | 从 SMPL-X 到材料体积坐标和姿态纤维，研究任务驱动的物理实验编译。在短时静态可复位的线性高斯模型中比较充分统计量、实验模拟和半正定设计。 | 物理动作产生的实验族与统计后处理权限要分开。只能按所声明的可复位、局部响应和未来管假设推论。 |
| **V4.2_revised** | 迁移材料任务及允许实现。响应 `J=H+Theta N`，若 `HB=I`，则 `JB=I+Theta NB`。任务零空间与已有校准可消去部分暴露，导出首次校正最小最大误差和精确任务响应所需探查方向数。看 `thm:cal-rank`、`eq:cal-kstar`。 | 证明依赖指定不确定集、任务度量和消息权限。谱条件不代替物理输入、接触和持续扫描的约束。 |
| **V4.3 原稿** | 用材料切空间及 SO(3) 姿态残差学习 GP forcing。二阶几何生成器沿材料进度推进，并拼上局部反馈校准、任务暴露和失锁恢复。`eq:v43-space-generator`、`eq:v43-common-tangent`。 | GP/DMP 是技能表示工具。稿件中的统一几何与 forcing 并不意味着后续腿部所有策略都训练过这个 GP，或学过连续 DMP 参数。 |
| **V4.3_revised** | 细化带几何误差的暴露 `JB=I+Theta NB+E_geo B`、数据条件暴露 `D_Q`、最小致秩亏半径和带正推进的二阶局部恢复。闭谱范数球定理使用 `delta ||D_star||₂<1`。 | 需核对半径和满秩结论所用的局部线性类。近似误差和能力余量须传到连续轨迹。 |
| **V4.4 原稿** | 单参数仿射响应下，把不确定模型端点、任务多面体顶点和物理输入能力组合为精确校准预算。零结构暴露仍可能因运动容量或推进要求需要真实探查。扩展到多通道和有限加速度回位。`thm:v44:scalar-exact`、`eq:v44:scalar-exact-frontier`。 | 有强解析性，也有窄模型条件。该资源定理不能直接填到无新增预算的腿部策略上。 |
| **V4.4_revised** | 把材料图册引入共同数据空间，用不同可行物理实现对同一未知局部目标取证。dIRD 讨论后验模型共用同一个物理输入的能力。看 `eq:quantifier`、`eq:stack`、`eq:dual`。 | 强条件是对所有相容模型用同一输入。逐模型存在输入不能推出可在未知患者上选同一个动作。有限滑轨满秩仍可能无有效整段路径。 |
| **V4.5** | 状态扩展到材料任务、数据一致世界、连续实现分量和合格覆盖。显式三轴反例、带所有观察结果的原语合同和观察前视距离/反扫资源条件。`prop:counter`、`eq:state`、`eq:primitive`。 | 保存几何或瞬时 Jacobian 不足以表征所有后继。有限原语没找到见证时记未知。当前腿仿真难以单独证明理论中所有原语权限。 |
| **V5.1** | 由完整观测与执行查询族自动编译双线性概率表示。选择共同条件坐标，保留高维条件高斯部分，给出类内最小非高斯条件维数。保存唯一事件以支持历史重放和新增查询。看 `thm:compiler`、`eq:bilinear`、`eq:generators`。 | 最小性是声明表示类内的结论。独立局部耦合可强制维数上升。腿部 v15 的图融合采用工程后验，不自动继承这个完备性。 |
| **V6.1** | 后验条件下的任务允许实现、完整二阶失配和有限段验收。允许协同改变接触和姿态维持同一成像要求。`eq:vesselrule`、`eq:gram` 及有限/连续风险公式是 v16 读码起点。 | 数学命题依赖正确的后验和余项。v16 验证执行链与有限混合判定，盲测给出负结果和未校准的实际几何违反率。 |
| **V7.1** | 选择关联的交互基元，在共同未知参数下保留选择证据、持久模式和规范得分。两模式不交叠剩余空间内的提取条件，另有可设计协方差的 LMI。`eq:sel-contract`、`eq:coupling`、`eq:selection-lmi`。 | 边界是有限/规范高斯实验及物理控制族。相同信息矩阵可以留下不同可用后继，例子是结构性动机。 |
| **V7.2** | 一般两模式的随机核自由度约简到剩余空间交集。正测度卷积和边缘相容刻画。标量选择时用短接算子，矩阵可比时条件锐利。`thm:general`、`eq:short`。 | 三维反例说明普通投影会错。精确刻画、可计算子类、连续物理扩展须分开评价。 |
| **V7.3** | 加上有界参数域的严格后继间隙、材料图册、真实切片几何、有限鲁棒执行管和因果分支树。用共享当前任务来分离后继完成率。`eq:finish`、`eq:physicalcapability`、`eq:sliceimplicit73`。 | 有连续解析反例和有限证书接口。需要独立物理检查才能把能力间隙落到真实机器人。 |
| **V8.1** | 身体中心感知动作系统稿。联合材料任务、观测、目标估计、Genesis 技能、dIRD 与 UltraPoC 外层接触/成像 QP。`eq:sharedstate`、`eq:sa`、`eq:la`、名义速度和约束。 | 早期完整架构和实现规划。各子系统在不同实验中上线，勿把集成草图当完整人体系统结果。 |
| **V8.2** | `Update`、`Query`、`Realize` 分工。对象证据状态、独立检查要求、可交付图像账本和新鲜度。多任务共用对象认识并在各自允许实现上取得新图。`eq:updateop`、`eq:queryop`、`eq:realizeop`、`eq:taskspec`。 | 很适合组织论文正文。现有腿部六区、补查和旧图重读属于同一路线任务内。短轴→长轴和跨器官真实迁移各需独立验证。 |
| **V8.3** | 固定信息实验与可行模式的耦合。加性分解和环判据、最小关联秩、物理受限细化证书、可计算的连续例子。看第 2 节公式。 | 已有数学原型和局部原生回放。对大规模腿部自然数据里的依赖强度，要单独做证据×出口干预。 |
| **V8.4** | 将手设能力表改为可学习的短程序联合核 `K_a`，统一后验、可执行续行、期限完成预测、已采证据释放和旧知识复用。`eq:kernel`、`eq:bayes`、`eq:dp`、`thm:uniform`。 | 对同一核跨请求、跨预算的理论误差界有明确前提。后期腿部教师以采样估计当前策略价值，任务核的跨请求泛化仍是研究命题。 |
| **V8.5 恢复稿** | 有限对象和短程序接口，探针分布的行空间辨识、稳健/乐观界和可审计的有限核频数。 | TeX 明确为恢复版，附新写有限合成核验。无法取回的旧数字未沿用。不要把恢复稿数字与后来 V8.5.2–5 的关节实验混用。 |
| **V8.5.2** | 冻结本地短程序模型预测未训练请求链的期限完成率。区分总体训练关联与当前对象后验，有限历史策略混合 LP、同时置信集、物理抽象误差。`eq:rowspace`、`thm:ident`。 | 有专门第三方审阅包和负结果。真实传感与强基线的交互效率没有被该稿全面验证。 |
| **V8.5.3** | 文件 `V8.5.4_execution_sufficiency_20261108/INPUT_Anatomical_GPS_V8.5.3.tex`。从局部训练预测转向证据条件化完成可行域，支持集递推、共同策略和冲突。`eq:feas`、`eq:decision_suff`。 | 原 Genesis 版本只有一个受训练入口支持的完整续程，截止边缘预测亦失准。该反例直接约束证据选支的论断。 |
| **V8.5.4** | 同一子目录的 `Anatomical_GPS_V8.5.4_experimental_revision.{tex,pdf}`。预测相冲突与共同获胜策略相冲突分开，讨论可实现划分和冲突超图。 | 新增 Genesis 自由空间关节执行和 0°/30°/60° SMPL-X 特征运输。配准和机器人回放与真实接触、B-mode、人体证据分开。 |
| **V8.5.5 证据系列** | 主要在 `E/v855_*`。同对象下试 E0 中断/早晚证据，E1 跨姿态或体形短段传输，dIRD 入口，E2/E3 局部转接和新请求链。`E/v855_transfer_review_20261108/REPORT.txt` 最适合先读。 | E1 报告 material 6/6、rigid anchor 4/4、world frozen 0/6，material 6 次仅 4 种独立配置。同标架 canonical 是数值等价检查。简化横向反馈+DLS+Genesis PD 的范围内，有短段迁移证据。E0/E2/E3 是有限配置和程序见证。 |
| **V9.1** | 规范患者概率 Belief、分型几何运输、身份关联和局部内部漂移。e3nn 提议、任务查询、dIRD 允许实现。`eq:transport`、`prop:plane`、`eq:compatibility`、`eq:latent`、`eq:totalcov`。 | 类型化运输的命题与已通过的有限区域审计可独立引用。e3nn 在腿部主线没有独立收益实证。 |
| **V9.2** | 任务收缩到未见对象的首次获取。公开粗 ROI，局部图像、e3nn 提议和有限后验，目标尚未确认时允许搜索，确认后以 V6.1 检查任务自由度。`eq:firstpassage`、`eq:terminal`、`eq:freedom`。 | 需要单列首次合格新图的达成、失锁和未见解剖。旧图与完整背靠背扫描不能替代首次获取。 |
| **V9.3** | A/B 的主要开发稿。相关宽先验、解析切片几何+条件残差、检出/视角质量、实际运动后的可微任务能量。`eq:prior`、`eq:lift`、`eq:hit`、`eq:kernel`。 | 强在线 Jacobian、条件响应学习和纯解析均有早期实验。宽先验和采样支持有限。具体 A/B 增量按冻结协议看，强实时重规划可能获得同样结果。 |
| **V9.4** | 技能作为可扩展的提议先验。嵌套短样条、广支持候选、暂时失锁、观察分支、退回和恢复，物理合同保留。`eq:spline`、`eq:nesting`、`eq:distequiv`。 | v10、v12、v15 有定向复查和恢复实证，但使用有限程序。不要把已做离散菜单写成任意连续拓扑探索和完整 e3nn 策略。 |

## 4. 早期 NMI A/B 实验与重要的正反证据

本机目录均以 `A/` 为前缀。以下逐项给出入口。旧目录常有 smoke、重复调参和大批量文件，由附录索引定位冻结版本。A 是解析几何加可估计的局部响应，并可用强公开反馈重新规划。B 额外预见探查后的公开新观测和真实出口对任务续行的影响。A/B 指决策权限，不能以早期弱 A 替代后来的强在线 MPC。

| 本机目录 | 看哪些文件 | 实验内容、结果和局限 |
|---|---|---|
| `ab_validation_20261110` | `PLAN.txt`、`AB_ROADMAP.txt`、`FINAL_REPORT.txt`、`audit/FINAL_AUDIT.txt`、`repair_20261110/FINAL_REPORT.txt` | 做 A 响应、B 消融、实际 dIRD、DMP/KMP、仿真伺服、e3nn 类型与数学例子。响应变化组解析误差 0.08888、学习残差 0.04840、在线 Jacobian 0.04773。精确解析负对照学习无额外增益。有限 B 对照优于简化臂，和同候选穷举值一致，去 dIRD 代理仍可持平。早期有隐藏目标读取、深度查错和跟踪缺陷，须读修复后结果。 |
| `ab_theory_recheck_20261110` | `FINAL_REPORT.txt`、`EVIDENCE_LEDGER.txt`、`REFOCUSED_ROUTE.txt`、`RESULT_UNITS.txt` | 重核材料运输、响应可识别性及 e3nn/VN 与普通本地网络。有限粒子若没有真实患者支持，即使重加权也会失效。有独立持出和缺支持反例。 |
| `ab_module_ladder_20261110` | `FINAL_REPORT.txt`、`MODULE_STATUS.txt`、`planning/run_experiment.py` | 用模块阶梯找哪个替换实际改变结果。对照真后验、信息选择、任务价值和实际可行动作，记录开闭哪些模块。留意有限动作/观察模型不能说明完整机器人优势。 |
| `ab_core_memory_20261111` | `FINAL_REPORT.txt`、`REPLAY_PROTOCOL.txt`、`mechanism/REPORT.md`、`revisit/RESULTS.json` | 理论式三件套、缓存等价和复访。稳定响应下缓存等于完整重放，相对四事件窗口少 1.64 次探查、新请求少 0.53 秒，70 对 68。复位后更新/重探 65/128，对盲旧缓存 45/128。有限双窗同信息例不同出口有 0.74 对 0.50。声学和复访仍是合成。 |
| `ab_live_surface_20261111` | `GEOMETRY_PROTOCOL.txt`、`REPORT.txt`、`run_geometry.py`、`run_continuation.py` | 从材料锚点重建实测表面标架，64 地址×七姿态。合成传感修正高度 p95 5.709→0.066 mm、法向 13.91→0.58°。同变换下等变另验。真实接触传感缺席。 |
| `ab_closed_loop_20261111` | `PROTOCOL.txt`、`REPRODUCE.txt`、`ASTRA_CONCLUSIONS.txt`、`ASTRA_INDEPENDENT_AUDIT.txt`、`closed_loop_reviewed_run/` | 新声学包读取×预测真实出口的 2×2。128 合成对象各两镜像记录。主 3.9 s 时 B/强 A/成对回退都 78/256。固定首段的机制时限，遮包 78→读包 120/256。5.85 s 为 B 136、强 A 88、完整成对 133。7.8 s B 153、强 A 159、成对 155。说明条件下使用新证据有价值，同时报告强基线追平。 |
| `ab_closed_loop_reviewed_20261111` | `ASSOCIATION_CORRECTION.txt`、`PROTOCOL.txt`、`run_closed_loop.py`、`run_open_matched.py` | 上一行的关联校正或复跑入口。`closed_loop_frozen_run` 和 `closed_loop_reviewed_run` 使用同一批场景，前者是校正前归档。读取首包和揭示新的实际可达出口是两个干预。 |
| `ab_native_exit_effect_20261111` | `REPORT.txt`、`NATIVE_DWELL_CASE.txt`、`PHANTOM_HANDOFF_RECHECK.txt`、`dwell_sensitivity/`、`late_native_witness/` | 同一起点、不同右侧停留导致几乎相同 TCP 出口，但后续同一左动作因全状态不同通过或失败。0.30 s 探后峰值 1.557 mm 通过，0.60 s 探后 2.211 mm 未通过。解释为理想低层回放下执行见证，短停留的图像信息性未测。 |
| `ab_executable_skills_20261111` | `REPORT.txt`、`ASTRA_DESIGN_REVIEW.md` | 合法反馈程序组合，枚举首包到续扫映射。真实出口对名义出口正确同为 632/640，前者少采 532 帧、少回旧窗 83 次。强 A 正确 631/640。支持名义与实际出口区别，收益以合成短任务效率为主。 |
| `ab_adaptive_stopping_20261111` | `DESIGN_PROTOCOL.txt`、`RUN_STATE.txt`、`MID_REVIEW.txt`、`RUNTIME_AUDIT.txt`、`NATIVE_PILOT_AUDIT.txt`、`study_v3/` | 学观测条件下何时探、何时正式取证，逐步查真实位姿和回撤。区分正式研究、smoke 和驱动 dry run。达到最大诊断段数不可表成自主最优停止。 |
| `ab_continuing_rollout_20261111` | `DESIGN_PROTOCOL.txt`、`RELEASE_REVIEW.txt`、`RUN_STATE.txt`、`study_v4_closed_loop_20261111/` | 多段继续/探查使用共同强 A 续行，对照 B 的前瞻覆盖决策。`RELEASE_REVIEW` 是实现审查；疗效结论须读相应冻结运行。|
| `ab_fisher_adaptive_20261111` | `PROTOCOL.md`、`ASTRA_AUDIT.txt`、`native_outputs/` | 固定动作树做 Fisher 提名、原候选与共享候选的拆分；含 3 mm 平移和 native 见证。成本权重与两段截断来自本轮协议。Fisher 提升代理不能替代任务闭环。 |
| `ab_observation_stress_20261111` | `FINAL_REPORT.txt`、`SENSOR_CONTRACT.txt`、`run_stress.py`、`dense_followup/run_followup.py` | 攻击条件是漏检、杂波、遮挡、有限候选与后验错设。由稠密候选覆盖和同信息对照鉴别“看不到”与“选错”。正反例均附假设。 |
| `ab_soft_exploration_20261111` | `ASTRA_STRESS_PROTOCOL.txt`、实际结果见索引 | 公开软模式、长段失锁、回撤以及不同重试。协议记录完成/错停/未决和恢复时刻，不能把行政检查点所报未决直接视为终局失败。 |
| `ab_fisher_continuation_20261112` | `FISHER_CONTINUATION_REPORT.txt`、`RECHECK_READOUT.txt` | 运动可续行与 Fisher 反复取证，同可行候选比较。读原始对照成本、完成曲线和提名是否只带来更多帧。 |
| `ab_fisher_prodmp_20261112` | `FISHER_PRODMP_REPORT.txt`、`outcome_teacher/` | Fisher 导向与 ProDMP 示教生成。优先核查等采样、等路程和后验收益，关注选择数据的学习效应与更大动作菜单的效应。 |
| `ab_leg_bifurcation_20261112` | `LONG_LEG_REPORT.txt`、`ASTRA_AUDIT.txt`、`learned_prodmp.py` | 早期长段分叉图册和可执行 ProDMP 候选。供腿部回访底层代码复用。本轮验收范围及其声学、关节边检查与后来的六区定义不同。 |
| `ab_leg_dynamic_20261112` | `DYNAMIC_LEG_REPORT.txt` | 动态膝姿态和暂态内部形变，当前角度与历史滞后分开。报告预测和合格身份各自的结果。 |
| `ab_knee_latent_rl_20261112` | `PROTOCOL.txt`、`REPORT.txt` | 28 站有限 Y，隐藏暂态、匿名质心。512 训练例、3,072 候选序列、9,216 已执行转移。离线 SAC、CQL 或学习对照按正式报告名称引用。256 留出，规则 239/256 的严格分支通过已很高；策略回报和身份确认门均特殊。 |
| `ab_knee_matched_20261112` | `PAPER_CORE.txt`、目录内冻结和配对记录 | 同观测/动作/成本比较移动地图、暂态与补采。重点是同姿态不同内部历史的机制，和换位置/换时刻是否改善局部可识别性。 |
| `ab_map_recheck_20261112` | `PROTOCOL.txt`、`REPORT.txt` | 冻结材料参数边缘 vs 在线，同一 28 帧首扫后各三帧。常规 112→116/128，压力 65→65/128，误确认 55→58。两种地图各自的控制路径和同轨读图分别评价。 |
| `ab_learned_acquisition_20261112` | `PROTOCOL.txt`、`REPORT.txt`、`material/RESULTS.json` | 局部响应与新观测能否提高后续获取，复访的长期个体参数需与短期状态分开。训练集和持出、预测改进与动作改进分别报告。 |
| `ab_live_anchor_firstscan_20261112` | `REPORT.txt`、`anchor_replay.py` | 低频延迟体形与较快局部接触测量，比较真实位置估计。32 次膝动和 9 个材料站是离线几何批次。实际接触与第一张真实超声未证（报告用合成输入）。 |
| `ab_motion_recovery_20261112` | `ASTRA_RECOVERY_DESIGN.txt`、`EXECUTION_CONDITIONED_PROBING_THEORY_20261112.txt`、`B_GLOBAL_LOCAL_ATLAS_SYNTHESIS_20261112.txt` | 丢失、回撤、局部重捕获、全局地址的设计。看合法备选集合是否包含救回动作，理论路线不可自动当作已闭环成功。 |
| `ab_b_refinement_20261112` | `REFINEMENT_REPORT.txt` | 48 标签合成主试，共用强 A 首扫和实际出口。把修改评分、锁定续行、观测容许改选分别对照，记录路程与错见证。可检验 `E max` 的有限候选机制。 |
| `ab_prodmp_rl_20261112` | `REPORT.txt`、`IDENTITY_AND_DYNAMICS_REPORT.txt` | 规则首扫、从真实前缀学有限续行。旧 96 例规则 82、RL 84。另 48 例规则 43、后验改选 44、配对回报学习 43。身份失败多，Fisher/均匀扩量未建立提升。共享合法路径与估计器。 |
| **`ab_tro_y_continuation_20261112`** | `PROTOCOL.md`、`REPORT.md`、`ASTRA_REVIEW.md`、远端 `REVIEW_REPRODUCE.md`、`holdout/{COMMON.jsonl.xz,RUNS.jsonl.xz,VERIFIED_SUMMARY.json}` | 128 个合成移动 Y，各臂共 28 帧首扫。成功要新独立正确支图并完成可续行。在线地图+B 与强在线 MPC 同 **76/128**，错确认均 16，冻结地图+B 为 70/128、14 错。三方法实际都先选中窗，横向储备条件在已选动作上没有产生不同尾段集合。该实验没有给 B 相对强 MPC 的增益。 [报告与逐例证据](https://github.com/NingMeng0903/RealUS_playground/tree/cdb1705a02f60382ffbe9e42f2ed420e27a28ddb/NMI_idea/experiments/ab_tro_y_continuation_20261112) |

复跑 `ab_closed_loop_20261111` 时优先按照 `REPRODUCE.txt` 在校正目录重放，跟踪 `case_id=5` 的 `same_fork_minus_I0` 与 `same_fork_minus_I1`。共同首段 1.225 s、同一读数 0.811537、同一出口。遮包选择左窗失败，读包选择右窗通过。`case 41` 是读了证据改去错误窗口的反例。保存共同位姿、q/qd、耗时、包、实际出口和真实末段验收。将磁盘轨迹缓存视作确定回放。

### 4.1 后续 B 分支、跨对象和历史教学

- **B 对强 MPC**。本机 `B/work/b_validation/{README.md,benchmark.py,challenge.py,search.py}`、`B/outputs/b-validation-results.json`。该结果的 `decision` 是没有候选通过开发集，未启动确认集。发布在 `ab_b_mpc_validation_20261113`。
- **逆向获取与任务分拆**。本机 `B/work/inverse_b/{PROTOCOL.md,PROTOCOL_FREEZE.json,bench.py,independent/}` 和 `B/outputs/b-inverse-acquisition-results.json`。32 独立病例 ×24 后缀带，两类任务。daughter 的拆分成本筛选/扩展候选相差约 +0.91 个百分点，病例聚类区间跨零。预测器在进一步训练前止于该证据。本地 JSON 记录 [固定发布](https://github.com/NingMeng0903/RealUS_playground/tree/e028eb4b91ad9bba99f834012d9dafb50724aace/NMI_idea/experiments/ab_inverse_acquisition_b_20261113)。
- **Y 拓扑复查**。本机 `B/work/topology_b/{README.md,PROTOCOL.md,fit_topology.py,imitation.py,suffix.py,learner.py}` 和 `B/outputs/b-topology-reinspection-results.json`。每个 cohort 128 有 Y、128 无目标。全回波强 MPC 在 nominal/held-shape 的 Y 完成为 51/128、38/128。Checked continuation B 为 58/128、31/128。RNN PPO 与前馈 PPO 的 Y 完成都为 0，CQL 2/0。nearest 压缩可丢正目标观测。名义与持出形状结果反向，混合两者会掩盖迁移失败。
- **教师提名与 PSDP**。审阅分支有 `ab_teacher_b_training_20261118`、`ab_teacher_generation_20261118`、`ab_probe_program_20261119`、`ab_psdp_b_20261119`、`ab_psdp_open_duration_20261119`。关键提交 `3f2db45041b97495a5c94784dfd6a593a7ba4c50` 是等 dose 教师数据重训，`ad7264e17488069ea450a543d65e79d6121c6cbd` 是成对 continuation 标签 PSDP，`45615cbe5978a38969d1d0f93d9320bc3380c78c` 和 `36dd120c1261c8ff44ed61ada8f8ce115e5628f7` 是打开截止的冻结复放和完整控制表。PSDP 先训练续行 child，再给根动作估计共同实际出口后的直接 MPC 和自适应 child 价值差。开放时长复用原策略和原采样前缀，仍有 67 mm 行程与既有运动检查，不能解读为无约束新训练。 [PSDP 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/ad7264e17488069ea450a543d65e79d6121c6cbd/NMI_idea/experiments/ab_psdp_b_20261119/REPORT.md) · [开放时长固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/36dd120c1261c8ff44ed61ada8f8ce115e5628f7/NMI_idea/experiments/ab_psdp_open_duration_20261119/REPORT.md)
- **PSDP 是另一项可写的学习证据。** 开放扫描时钟条件下，Full MPC nominal/drift 为 723/1,024、289/384，独立 continuation 标签+PSDP 为 954/1,024、348/384。保持 PSDP 首动作、交给直接 MPC 为 860/1,024、310/384。同首动作、固定第二段则为 964/1,024、332/384。nominal 里固定第二段更多，drift 里动态 child 更多。这组配对把“找到较强的第一动作”和“学到观察条件化的第二动作”分开。主要样本单位仍是 128 nominal、48 drifting 布局，1,024/384 是各布局重复记录。这是早期有限探查任务，独立于后来 v10 的腿部六区主线。 [完整对照与成本](https://github.com/NingMeng0903/RealUS_playground/blob/36dd120c1261c8ff44ed61ada8f8ce115e5628f7/NMI_idea/experiments/ab_psdp_open_duration_20261119/REPORT.md)
- **V8.5.5 转接和共预算**。`E/v855_transfer_review_20261108/{REPORT.txt,INTERPRETATION_CORRECTION_20261108.txt,manifest.json}`、`E/v855_handoff_equal_budget_confirmation/{summary.json,frozen_source/dird_probe.py}`、`E/v855_composable_confirmation/summary.json`、`E/v855_dird_entry_transfer/summary.json`、`E/v854_theory_revision/Anatomical_GPS_V8.5.4_experimental_revision.tex`。把冻前预测、相同测试调用、独立对象数和各态重复分别核对。部分等预算预测模型共用同一个测试动作，说明预测迁移，不能算学习策略做出了更好的行为。
- **胆囊作为独立任务的相邻证据**。实际本机报告在 `/home/camp/Documents/Codex/2026-11-20/new-chat-3/work/RealUS_playground/NMI_idea/experiments/ab_gallbladder_feedback_20261120/REPORT.md`。40 解剖两档噪声 80 记录。固定站点正确交付 45/80，手写在途反馈 54/80、误交 5→2，多峰固定站点 19/80，学生访问数据学习 48/80、误交 8。动作在运动中发现好图后改变路线，最后仍要求停稳时独立新图。支持主动动作能把「曾看到」转成「停驻可交付」的机制。它的解剖、任务、模拟器和期限独立于腿部，不是共享策略已经跨器官迁移的证据。 [胆囊固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/101325037dcb9c899dd4553245f5aa505a9caab9/NMI_idea/experiments/ab_gallbladder_feedback_20261120/REPORT.md)

## 5. 腿部任务和三个模块的实际训练合同

**股静脉路线**在中立材料站 −595 到 −790 mm，40 站。具名目标沿股静脉接到腘静脉，BodyParts3D 右腿和 Open3DModel 右腿有同源支持。BodyParts3D 左镜像缺相应连接证据。**全程路线**为 −525 到 −1255 mm，147 站，股静脉→腘静脉→胫后静脉。只有 Open3DModel 右腿覆盖整条同源路线。BodyParts3D 的腘静脉至胫后静脉有约 44 mm 缺口，在覆盖审计保留，未拿来填全程。资料含同名另一分量作为竞争对象。没有独立股总静脉标签时明确标为缺乏，勿凭图合成具名汇入口。每条路线各定义六区任务，不是六条血管，也不能合报一个观察性成功率。MedSeg 近端短段退出 v2 后试验，新样本全是图谱衍生仿真。见 `H/leg_scan_target_routes_20261123_v2/{REPORT.txt,atlas_coverage_audit.py,atlas_projection.py}`。

**合成造数**。选择 SMPL-X 体形、源图谱、种子和普通/压力档。`AtlasLayoutFactory` 按源内支持的每条角色路径加入局部平滑偏移，关联图的空缺保持未知。v13、v15 增加局部动静脉间距和声窗变体。v16 的形变组让相对布局与连边不变，加入局部轴向剪切并审计骨体表。合成布局每趟固定。实际探头姿态和时钟决定声影、视野、掉检、噪声、伪点和混叠。真值供回波生成、监督和验收。候选的公开评分与部署动作按已采匿名回波运行，置换隐藏标签应逐项不改变公开 logits、动作菜单或执行。独立扰动使用正式版本记录的种子和随机带。 [基础代码](https://github.com/NingMeng0903/RealUS_playground/tree/22d1f9c8c1e048f64ba6546c166c0b748e9d3893/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_closed_loop_20261122_v1/src)

**读图器**。保留候选级匿名回波、位姿/材料地址、图谱和历史。早期公共契约 48 数值+48 缺失掩码 = 96。后续加八类角色、学习式关联 embedding、相对几何和对象轨迹。v13/v15 的 177 维是完整公开输入，包含构造项与学习项。v15 的对象图跨声窗去重，融合 R4–R6 的病例上下文，在区域内对不同对象与 unknown 打分。保留局部候选路径降低错传播，实际部署目标是候选交付排序和拒交。`s = s_v13 + 6 tanh(delta/2)` 是 v15 的 rank 残差。原型关系带支持度，绝对位置和拓扑均参与。固定旧图用 `(frame,slot)` 锁定同一候选，检验其身份是否更新。旧候选池换了赢家、旧候选自身身份改了、新图参加竞争是三种不同结果。

**教师**。同一个公开前缀、解剖、实际状态和未来带上，执行候选 `a` 与参照 `a_ref` 的可行运动并收取带噪回波。用同一冻结读图器和同一冻结学生续扫到任务终点。标签是 `Delta Q_pi(H,a)=mean_k[U(a→pi,H,omega_k)−U(a_ref→pi,H,omega_k)]`。v15 的回报向量分别是严格交付、后三正确区比例、具名错支标记和后三伪点/混叠比例，聚合权重 `[1,.36,−.38,−.18]`。额外独立未来带复验搜索选择。真值只用于训练后果和评分。强公共 MPC 对照可以在公开模型里规划，不能试跑未选动作的隐藏真实布局再当成自己从公开信息获得的结果。相似公开状态内不同隐藏布局可需相反动作，要学期望和主动增加可辨信息。

**学生**。早期学继续/探查与可行参数。v9 真正 on-policy PPO，先 BC 再在线实走，公开 GRU actor，训练 critic 可用额外真值。回报 `r_t=aux_{t+1}−aux_t+2.50 1_done·strict_final`，标准 clipping/GAE、熵项和 critic 回归，教师组用实际配对效果做辅助梯度。v10 主要是对明确定义的恢复目标做监督初始化，试了残差 PPO 但正式冻结监督版本。v12–v15 主要对候选宏程序监督学习，使用学生访问前缀重标并在完整开发扫查选部署。程序可以在中途换窗、短追踪、回访已扫区、重算剩余区域，常规推进仍复用基控制器。v16 冻结 v15 模型，只换局部动作判据，CPU 完成。论文可单列反事实取证训练贡献。v15 的读图取证对和后续教师标签有各自数据分割，虽然复用同一设计和公共接口，不能把所有标签写成由同一批行同时产生、或写成已实现三模块端到端联合 PPO。

## 6. 腿部版本、关键实验、实际结论和缺陷

绝对路径把 `H/` 替换为 `/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff/`。有些远端小型结果在 `analysis/`、根目录汇总或复算文件，本机完整中间产物在 `run/`。以附录源目录和 Git 提交树为准。下列报告可以直接从链接打开，不必先克隆大仓库。

### 6.0 历史起点：216 条训练、一个 Blender 留出及独立配准教师

- `H/history/CURRENT_STUDY.md` 是前序事实、已放弃的接口和运行要求的摘要。`H/history/prior_sessions/` 是长期对话归档。`H/historical_evidence/` 和 `H/reference_source/` 为只读历史副本。`O/Leg_Scan_CLI_Handoff.zip`、`O/Leg_Scan_Review_Evidence.zip` 是当时打包，不含所有后来的 v4–v16。
- 早期三图谱×四路线×两档扰动共 216 次，6,426 帧。6,140 个候选评价里正例约 4.153%，用 ExtraTrees。Blender 160 条为五路线×两档噪声×16 带，只有一个独立解剖。MPC、学习换窗、再加半速分别正确留 34、35、42/960 区，严格整趟均 0/160。较差的 8 mm 先验门和旧质量通道含隐信息，需要读 `H/leg_scan_study/diagnostics/` 审计。公开选择、标签评分、覆盖上限分别统计。
- `O/leg_registration_teacher_20261123/` 有 `.bundle` 和 replay ZIP。固定 96 趟下两张图正确交付，等时固定 57、公开历史规划 60、特权教师 56；三臂 96/96 旧/新候选池都见过目标。两张图中的错图 43、44、46。教师备选模拟用真值贪心续扫、实际执行再搜索，存在续扫不一致和旧图评分时钟差异。几何粗筛和隐藏真值的免探查规则后来移除。 [eea9fae4 实验与配置](https://github.com/NingMeng0903/RealUS_playground/tree/eea9fae44bfac02234bdd407a758cb210731daaa/NMI_idea/experiments/leg_registration_teacher_20261123)
- 同一父目录有 `leg_disambiguation_learning_20261123`、`leg_follow_condition_20261123`、`leg_privileged_teacher_20261123`、`leg_student_learnability_20261123`、`leg_teacher_reader_rl_20261123`、`leg_teacher_rebuild_20261123`、`leg_teacher_stage1_20261123`、`leg_teacher_verification_20261123`、`leg_topology_feedback_20261123`。大多数保留 ZIP。文本入口包括 `leg_follow_condition_report.md`、`leg_teacher_reader_results.md`、`leg_scan_request_benchmark.md`。用 `unzip -l` 检查内容，解压时用新目录。这些是小试或来源归档，勿和六区盲测重复合计。
- **早期主动消歧与可学性**。`O/leg_disambiguation_learning_20261123/leg_disambiguation_learning.zip` 内 `README.md`。两批固定候选中，共用段内排序的规则门控使全程严格 30→40/96。高层学习门控未占优。`O/leg_student_learnability_20261123/leg_student_learnability.zip` 的 `audit.json`、`stop_audit.json`。旧真值教师 730 mm 全部沿 MPC，195 mm 36 次实际探查中 32 次重新拍到目标。另同轨轻量读图强行提交 0→14/32，却增加错交，严拒答门下全不提交。可见更高的先验采集上限、可用读图和合理退出是三个独立问题。
- **区分能拍到与能交付**。`O/leg_teacher_rebuild_20261123/leg_scan_request_benchmark.zip` 中的新请求验收，48/48 教师轨迹六区捕获、oracle 48/48，而固定公开 reader 严格 25/48。`O/leg_teacher_reader_rl_20261123/leg_reader_bottleneck_check_20261123.zip` 的 `archive_components.json`、`fresh_components.json`、`summary.json`。局部胫后静脉诊断归档 128/128 捕获，双图只 75/128，53 个失败里 52 个至少错交一张，37 趟错交伴行动脉。`leg_reader_distillation_check_20261123.zip` 将入口已拍候选锁定，再比入口判读和看到后续 15 mm 后回判。样本来自相同基础图谱，不是独立患者。
- **主动拓扑小试和续扫计分**。`O/leg_topology_feedback_20261123/leg_topology_feedback.zip` 比直接继续、等时等待不读、同窗等时补拍、真走拓扑锚点再返回。64 场景×两带，每趟一次分叉闭环，关联假设含伴行动脉和伪回波。`O/leg_follow_condition_20261123/leg_follow_condition_report.md` 定义当前位置图、接回后新图和因果连续三个 5 mm 子区。这里是针对局部胫后静脉的实验，另有教师阶段一、验证、注册 replay 作为造数和标签溯源。

### 6.1 v1｜初始 GPU 闭环、合成解剖、ProDMP 和旧版大规模 PPO

本机 `H/leg_closed_loop_20261122_v1/`。看 `src/run_online.py`、`src/policy.py`、`src/teacher.py`、`src/online_reader.py`、`src/synthetic_anatomy.py`、`src/windows.py`、`src/material_motion.py`、`src/scan_kernel.py`，以及 `training/seed_276382173/` 和 `diagnostics/teacher_signal_summary.json`。环境保留材料坐标、实际帧时钟、探查形状与段内反馈，可模拟换窗及回访。合成形变有共享节点、局部扰动与连通检查机制。旧归档纳入 `medseg_labelled_trunk`，任务混杂和全程覆盖是重定义实验的原因。`training/seed_276382173/teacher_ppo/complete.json` 记录教师 PPO **30,720 完成**，早期纯 PPO 历史存 **25,088** 趟。旧版 30,720 不是后续具名全程增强、v15 或 v16 的训练数。训练合格比例与独立验证是不同量。 [v1 入库目录](https://github.com/NingMeng0903/RealUS_playground/tree/22d1f9c8c1e048f64ba6546c166c0b748e9d3893/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_closed_loop_20261122_v1)

### 6.2 v2｜重定具名路线，分解覆盖、身份、动作，比较停止开关

本机 `H/leg_scan_target_routes_20261123_v2/`。`fixed_pilot/{protocol.json,results.json,reader_ceiling_breakdown.json,teacher_probe_audit.json}` 是核心，程序 `run_target_pilot.py`、`public_embedding.py`、`teacher_continuation.py`。每路线 24 组。股静脉普通 MPC、主动 MPC、纯/教师 PPO 分别 10、6、10、10/24。全程四类严交均 0/24。全程 MPC 的 144 区全部捕获真候选，93 区正确交付、45 区拒识、6 区认错，无门槛第一候选仅 8/24 六区全对。加入公开材料、图谱和跨窗历史的读图后，股静脉同交量 10→13/24、全程未改善；股静脉等时补拍和教师均 13/24，直接继续 14/24。固定场景中少量早停不支持整程优势。dIRD 关闭。论文取证在 `literature/local_review.txt` 与 `literature/papers.json`。 完整报告和固定小试逐例在本机；公开提交可读该版接入后续实验的五个源码。 [v2 固定公开源码](https://github.com/NingMeng0903/RealUS_playground/tree/22d1f9c8c1e048f64ba6546c166c0b748e9d3893/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_target_routes_20261123_v2)

### 6.3 v3｜大数据重训身份/关联，公开动作价值与等时补拍

本机 `H/leg_reader_teacher_scale_20261123_v3/`。`REPORT.txt`、`run/`，源码 `collect_scale.py`、`scale_worlds.py`、`scaled_reader.py`、`scale_readout_eval.py`、`scale_teacher.py`、`scaled_value.py`、`scale_teacher_eval.py`。12,288 拟合扫查、6,181,845 匿名候选。两个读图训练臂各 33,040 次 CUDA 更新。教师 2,048 训练布局，训练/校准/验证的完整续扫合计 57,396 条。这里统计的是反事实完整续扫，有别于在线学生从起点采的回合。

同提交量时，股静脉错交区 132→53→49，严格 240→280→284/512。全程错交区 212→126→226，严格 31→37→32/512，具名错支 45→50→127。三臂依次为 v2、扩量身份关联、再加排序。独立冻结阈值下扩量对照可得 102/512，排序 75/512，记住覆盖率不同。全程普通 MPC、等时补拍、更新教师、允许直接继续教师为 59、51、53、66/384。最后的增量集中在直接继续的组，复验不稳定，错支 93→102。最小复现确认 `online_reader._new_geometry` 对 `[候选,来源,角色]` 的轴归约错，导致“目标距离”实际上跨角色取最小。训练关联头还未参与部署概率推断。 [v3 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/22d1f9c8c1e048f64ba6546c166c0b748e9d3893/NMI_idea/experiments/leg_reader_teacher_scale_20261123/REPORT.txt)

### 6.4 v4｜修图谱轴，关联参与推断，真实 ProDMP 与续扫复验

本机 `H/leg_scan_reader_feedback_20261123_v4/`。`REPORT.txt`，全数据 `run/{reader_results.json,action_results.json,link_guard_results.json,feedback_results.json,validation_summary.json}`。`public_features.py`、`public_graph.py`、`readout_state.py`、`active_trial.py`、`action_eval.py`、`guarded_association.py`。8,192 扫查、24,576 历史、约 802 万候选。学到的关联头进入图；候选新证据能重读旧图，控制反馈能改变连续轨迹。全程同网络逐候选→图推断，等交量错交 130→230、严格 105→76/512。图使 110 个赢家改对、264 个改错，其中 217 个错误赢家直接连真目标。归一化让单条弱边也可能承担固定 65% 混合。后补 guard 大幅压伤，回到近逐候选表现，不能视作获得额外增益，且后裁边会丢失正确候补。全程特权复查对等时补拍严格 40 对 32/256、原始带救回 8/无损失；对普通 MPC 40 对 44、救回 21/损失 25。看错支与独立带。旧指标 “preserving suffix” 只核对整趟二值不降和任意错支不增，不能直接说后缀各区全正确。 [v4 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/c4e4b321d68c011d2c6e5d80b03390e39c8ee5ea/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_reader_feedback_20261123_v4/REPORT.txt)

### 6.5 v5｜持久关联、多跳、公开前向锚点与时间扰动机制

本机 `H/leg_scan_anchor_state_20261123_v5/`。`object_tracks.py`、`anchor_policy.py`、`forward_anchor.py`、`train_links.py`、`replay_v4.py`、`run_anchor_study.py`。`run/anchor_results.json` 和 `run/reader_results.json` 是主要本机记录，远端还有 `analysis/` 的诊断。匹配前边评估、长时回访、杂波负对、同源路径一致、多跳关联入账。4,794 CUDA 更新只改关联头，旧主骨干和身份头冻结。全程 96 新布局实际带 MPC→公开锚点 43→46，复验差 +7.81 点，区间 [2.60,13.02]。相对等时补拍 +11.98 点。特权流和股静脉未确证。旧时钟取证选 22 个局部改善和 25 个整趟损失，交集只有 2。用固定 MPC 路径延迟采样重现 28 个选中后缀失败区，仅适用于该错误病例集。原图对八类分数直接平均会把清楚回波打低，真实对应也会被伪回波反超。首次缓存有历史依赖，按帧与按段入账终态不同。自然选窗由关联概率造成的单独动作改变是 0，自然早停为 0。固定站位与受控机制都不能充当自然拓扑锚点。 [v5 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/ba2871673325b714d877b8004acef0f58139b959/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_anchor_state_20261123_v5/REPORT.txt)

### 6.6 v6｜因果重读一致、身份与可用性分开、动态锚点和真实早停

本机 `H/leg_scan_identity_state_20261123_v6/`。`causal_reader.py`、`role_fusion.py`、`anchor_logic.py`、`live_programs.py`、`link_evidence_audit.py`、`paired_value.py`、`run_closed.py`。历史顺序固定，改变批次输入的终态一致。动态目标包括可辨识位置和短回访，部分股静脉自然早停平均省约 0.413 s，复验省 0.380 s。全程动态锚点 48/96 与 MPC 持平，复验 +6.77 点 [1.04,12.50]，原始错支 21→25。当轮预注册主臂是锁定 v5，动态程序结果只支持下一批独立确认。76 次再次见到同一实例，oracle 连通 64，公共连通 9。对象优先级容易偏向物理概率极低却条件目标高的伪回波。训练小样本和置信门控失配，教师全部退回参照，学生全程 42/96、比冻结 v5 低 3。 [v6 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/60a77233b60686cfeed377757503c1fa95f8ff6f/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_identity_state_20261123_v6/REPORT.txt)

### 6.7 v7｜独立动态确认，关联阶段追责和小批学生访问

本机 `H/leg_scan_public_evidence_20261124_v7/`。`run/{paired_compact.json,confirmation_results.json,student_results.json}`。源码 `association_audit.py`、`edge_score_replay_v7.py`、`readout_audit.py`、`observation_v7.py`、`value_v7.py`、`cycles_v7.py`。独立全程主带基准 44→50/96，R5 58→71，R6 78→73，原始错支 24→22，复验未过预定置信和错支上界。关联阶段 76 再见→64 有由检索支持的真通路→9 边过阈。旧边头重罚角色冲突，强同实例（即便 cosine=1）也可能被拒绝，旧 .90 阈值未真正独立校准。对象 A/B/other 样本失衡，84 次分支 83 次 unresolved，不能据此说观测 intrinsically 无辨识力。oracle 图重拟合候选交付从 6→62/96，公共图的对称重训到下一版补。动作两阶段共享错误标签，教师 3/6 条未来带需通过过强局部风险门，学生用单扰动残差低分位去约束期望收益。完成了真实两轮学生前缀访问与聚合，政策仍保守。 [v7 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/2f2c704eb0085f78e638d41cd2467c436713b695/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_public_evidence_20261124_v7/REPORT.txt)

### 6.8 v8｜公共/真值图对称，边头重训，两阶段动作标签修正

本机 `H/leg_scan_reader_feedback_values_20261124_v8/`。`run/{readout_symmetry_results.json,edge_training_results.json,evidence_results.json,paired_outcomes.json,student_test_predictions.json}`，以及 `reader_v8.py`、`edge_v8.py`、`motion_v8.py`、`cycles_v8.py`。同轨原公共图重拟合 35→34/96，oracle 图 6→62/96。去除角色一致性特征提高困难正例通过率，也让假边大量入场。72 新固定轨迹，逐候选 41/72，新图加对象监督 30/72。最终仍使用逐候选部署。实际学生独立 48 布局、两个决策点共 96 次全 `continue`，全程 15/32 等于 MPC。第二阶段有多类只 7 校准布局，被 8 的覆盖门整体禁用。局部一次短偏离不支持整趟修多个错误区。已实走的五种程序在另一 48 趟里后见择优上限 28/48，但把该选择固定切独立带为 21 和 20，MPC 三带都 22。上限是菜单内 oracle 诊断。 [v8 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/0e498eb2f4702f2efed997389cc8e09917f3dc47/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_reader_feedback_values_20261124_v8/REPORT.txt)

### 6.9 v9｜重训完整读图，真正完整在线 recurrent PPO，因果实走对照

本机 `H/leg_scan_full_route_20261124_v9/`。`fit_reader_v9.py`、`reader_v9.py`、`scan_v9.py`、`policy_v9.py`、`teacher_v9.py`、`run_v9.py`、`PAPERS.md`、`THEORY.md`，结果 `analysis/{full_route_results.json,paired_outcomes.json,independent_freeze.json,bc_paired_results.json}`。读图 9,249 拟合布局、9,335,033 候选行，更新完整共享骨干、关联、历史融合、身份和排序。128 趟行为克隆初始化，纯 PPO 与教师 PPO 各 1,152 趟完整在线扫描，每组 192 CUDA 更新。原始带全程旧 MPC 46/107、同轨新 reader MPC 55/107、纯 PPO+新 reader 57/107、BC 采样+新 reader 60/107、教师 PPO+新 reader 61/107。教师 PPO 错支 12/107，旧 MPC 为 30。独立复验未确证比同 reader MPC 更强。教师学生 46 个失败中，40 六区见过目标、6 新增漏拍，后六个跨三带重复漏 R3。多次偏离真实发生，但采样 BC 已接近，贪心开发检查点接近 MPC，读图和控制的改善不能全归给 PPO。 [v9 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/f115a663ec260779bc6a7dcb9b17e88a1354dd71/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_full_route_20261124_v9/REPORT.md)

### 6.10 v10｜目前最强控制证据，转向分区定向恢复

本机 `H/leg_scan_recovery_20261124_v10/`。`recovery_task.py` 按材料地址选择已扫区域及可执行补查，`recovery_motion.py`、`recovery_reader.py`、`recovery_teacher.py`、`recovery_policy.py`、`run_recovery.py`。原数据入口 `analysis/{reader_manifest.json,strong_training_summary.json,strong_composition_summary.json,development_results.json,independent_freeze.json,paired_outcomes.json,results.json,dird_virtual_check.txt}`。在原读图基础用 132 拟合布局约 41.7 万候选更新八角色头和共享表征。教师在 168 前缀和 54 实际补拍后前缀监督学生。192 残差 PPO 的三个开发轮均逊于监督初始化，正式冻结的是**教师监督学生**。动作可以从末站回到此前材料区，实测合法回接。dIRD 虚拟核对单列，主试关闭。

共用 v10 读图。股静脉原始 **26→48/54**、救回 22/损失 0、R6 漏拍 19→0、具名错支 0。两带按布局复验 +33.33 点 [23.15,44.44]。全程 **58→72/106**、救回 17/损失 3、原始错支 19→15，复验 +7.08 点 [1.89,12.74]，共用读图的复验错支差为 0。相对更旧 v9 reader MPC 的整套系统，复验具名错支却上升 +6.60 点。全程时长 33.45→43.70 s。剩余 34/106 已覆盖仍失败。公开目标高置信可能把错误血管标成已完成。早期参照先验和不确定性条件也制约教师是否主动查询真正完整起点的前缀。 [v10 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/4c3859b1024f2a9a890fb83f98df86c77c65314a/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_recovery_20261124_v10/REPORT.md)

### 6.11 v11｜高置信错支挖掘，A/B/未知身份取证与几何换窗基线

本机 `H/leg_scan_identity_evidence_20261124_v11/`。`evidence.py`、`evidence_model.py`、`mixed_reader.py`、`run_v11.py`、`confirm_geometry.py`，本机 `run/`、远端 `analysis/{paired_outcomes.json,geometry_confirmation_paired_outcomes.json,teacher_results.json,evidence_training.json}`。两批 80+96 全程，合并描述原始 v10 100/176、重训同轨 94、等时补拍 102、几何换窗 111、学习价值 100。原始相对补拍救回 10/坏 1，四条复验合计仅净增 1。同轨混合 reader 两批均坏。350 个教师 query 有 146 个 A/B 同真目标，37 个 A 坏 B 对，实际 “仅支持 B” 事件大部分并非可纠错。末区补查常选错区域，几何学生 65 个失败中 28 只错 R5，其中 22 未查 R5。最终系统是 v10 扫完后一次后补查，旧图 `resolve` 未驱动实际后续续扫。旧图固定改正为 0。 [v11 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/6a3f59805558932338980556cd1df6bd55c0c5c9/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_identity_evidence_20261124_v11/REPORT.md)

### 6.12 v12｜联合选区与程序，途中取证再回接，闭环验证

本机 `H/leg_scan_region_evidence_20261124_v12/`。`closed_scan.py`、`track_state.py`、`fit_v12.py`、`data_v12.py`、`region_policy.py`，`analysis/{results.json,test_paired.json.gz,fixed_identity_results.json,teacher_summary.json,reader_data_summary.json,v11_cross_origina*}`。同 108 新布局，v11 几何+v10 读图 64/65/66，联合学生 72/70/66，同学生轨迹改用 v10 读图 75/69/74。原带错支 31、24、19。联合学生相对几何救回 22/损失 14，两复验净 +2.31 点 [−4.63,9.26]，使用旧读图的同轨臂相对冻结 v10 有更强收益。真实执行 626 次，424 次在末站前，97 趟多区。更好取像主要在 R4、R6，时长 48.9→112.9 s。新读图的训练多是一次换窗，部署约 5.8 程序/趟，R6 多次历史覆盖薄。教师标签混用 132 个 v10 续扫和 48 个 r0 续扫，后续 v13 统一。固定旧对象证据更新和闭环身份反馈的独立配对未确证，平均表现可持平或反向。 [v12 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/373a4a6abec0c1d47980984c23d1b20bf71a34d1/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_region_evidence_20261124_v12/REPORT.md)

### 6.13 v13｜学生真实历史扩量、重算同续扫标签和同菜单时间参考

本机 `H/leg_scan_student_history_20261124_v13/`。`collect_v13.py`、`trajectory_v13.py`、`reader_v13.py`、`policy_v13.py`、`matched_v13.py`、`scale_v13.py`、`analysis/{results.json,test_paired.json.gz,matched_geometry_results.json,scale_confirmation_results.json,teacher_summary.json}`。冻结 v12 学生在 **1,024 个新合成布局**实际运行，统一冻结参考续扫、终局读图重标 **156 前缀**，加了失败编辑取证，末区读图在同轨验证。108 盲测布局，v12 取像+v10 读图 74/71/77，时间接近同菜单几何 74/67/77，v13 78/67/71。相对强参照原始救回 16/损失 12、错支 19→13、116.3→98.7 s，独立复验差 −4.63 点 [−11.57,3.02]。R6 补查 73→166 次而正确 95→90。30 个失败全覆盖，17 个只错 R6。256/512/1024 的固定更新次数曲线未持续上升。锁定旧候选分数对照支持把单候选改判和额外候选入池分开。 [v13 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/2a5bbd5ad92dd00f3f3d3642279d6c198525a75f/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_student_history_20261124_v13/REPORT.md)

### 6.14 v14｜冻结 embedding，第 6 区 A/B/unknown 的相对拓扑读图

本机 `H/leg_scan_topology_reader_20261126_v14/`。`DESIGN.md`、`relations_v14.py`、`fit_v14.py`、`experiment_v14.py`、`run/{results.json,paired.json.gz,freeze.json}`，远端 `analysis/{topology_source_audit.json,relative_invariance.json,smoke.json,verification.json}`。同轨同候选固定前五区，只换 R6。关系图共享 A/B 打分，交换顺序时 logits 对调，unknown 保持对称。绝对 embedding 贡献限幅保留，加入相对分布、同源延续、分离趋势与有证据的分叉汇入。两个约 4,826 参数头 CUDA 训练，1,024 拟合、64 校准、72 开发。三个源在这个末区的可靠具名跨角色分叉/汇入标注为 0，规则接入通过具名合成例验证，真实 R6 主要有连通延续和几何关系。正常三带全程 v13 78/67/71 → 图 79/68/71，具名错误三带合计持平。共同平移压力 39→44/108，救回 5/损失 0，错支 13→14。扭转拉伸 44→45/107、区间跨零，新增动脉误交；相对几何消融为 46。因此能主张受控共同平移下的特定帮助，不能把任意非刚性形变或具名拓扑直接说成已经解决。 [v14 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/24b17882d6e2e92e46fc3f3da465db6db2a71ebd/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_topology_reader_20261126_v14/REPORT.md)

### 6.15 v15｜保留全 embedding，R4–R6 共享身份，后验进入选窗追踪和续扫

本机 `H/leg_scan_shared_belief_20261126_v15/`。`METHOD.md`、`identity_v15.py`、`reader_fit_v15.py`、`control_v15.py`、`teacher_v15.py`、`data_v15.py`、`run_v15.py`。权重与冻结到具体 `run/` 和 `analysis/{reader_freeze.json,student_freeze.json,teacher_freeze.json,test_freeze.json}`。可直接复算 `analysis/{blind_paired.json.gz,teacher_pairs.json.gz,reader_evidence_pairs.json,results.json,verification.json,verify_v15.py}`。网络保留 177 输入，近端视角、病例内原型和短轨迹一起打分。追踪从确见位置出发，按公开角色和同源图谱权重加站间位移。每新宏步和跨区刷新身份，逐站更新几何。CUDA 读图 396 步，256/48/72 拟合/校准/开发，真实取证 375 对。教师 144 个实际前缀、776 备选，同一个冻结 v13 权重搭新版读图和控制续扫。备选的正负零效用都保留。独立第三带教师相对继续严格 90→94、错支 31→24。训练 6,286 参数的价值学生 225 CUDA 步。

同一 108 盲测布局三带，冻结 v13 67/75/65，v15 同轨终局 65/74/65，新控制+冻结策略 66/79/76，新控制+GPU 学生 67/71/79。**新学生相对 v13 三带具名错支 63→46，差 −5.25 点，布局区间 [−9.57,−0.62]**。同新版接口冻结参照错支 59，对新学生 46，严格交付的两条复验则仍偏向冻结权重。冻结参照加新在线控制相对同读图固定轨迹，两复验 +7.41 点 [0.93,13.89]，是整个在线控制/动作变化的对照。原带 v13 与新学生严格均 67。读图-only 的同轨部署不占优。71 对固定旧赢家在 0.5 阈值下 0 改对、0 改坏，不能宣称持久记忆已稳定改对旧图。新学生原带用时 155.4 s，v13 为 94.9 s。 [v15 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/e8837c832c2d6abda2151092c787377b81aed770/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_shared_belief_20261126_v15/REPORT.md)

### 6.16 v16｜V6.1 有限段风险接入实际探头动作，核心盲测负结果

本机 `H/leg_scan_segment_risk_20261126_v16/`。`METHOD.md`、`segment_v16.py`、`physics_v16.py`、`variants_v16.py`、`run_v16.py`。`analysis/{protocol.json,math_check.json,check_segment_math.py,blind_paired.json.gz,shear_paired.json.gz,results.json,verify_v16.py}`。共固定同一前缀、A/B 旧地址、共同实走探查和候选菜单。遮包 vs 读包，均值 vs 完整后验有限混合，四臂配对。每段保持接触点，围绕实际探头横轴转 4°，7 个采样拍。三维成像以当前平面相交隐藏中心线，再生成带噪匿名回波。几何残差尺度 4 mm、3 mm、sin(10°)，含端点线性化和旋转 Taylor 余项。独立几何和身份分别验收，联合指标要求该新图属于指定血管且整个短轴段都合法，不等于全程六区。自然 108×2=216，剪切 54×2=108。剪切 43/54 真正过几何审计，其他保留原布局。共 714 实际局部动作，未训练新 GPU 模型，也未加运动预算。

自然组联合合格为 **遮包均值 43、遮包完整 39、读包均值 39、读包完整 30/216**。读包完整相对遮包完整导致 81/216 改动作，救回 2/损失 11，差 −4.17 点 [−7.87,−0.93]。读包完整相对读包均值净少 9，区间也偏负。剪切两完整后验臂同 11/108。读包完整模型认证 101 个，真几何违反 25 个；剪切认证组 17/26 违反。失败中另有真切面合格、拍到目标但最终误交的情况。`v16_blind_00060` 是读证据改变动作后峰值误差从 1.050→0.913 的正例，`v16_blind_00006` 是读包后的反例。没有观察到模型经新证据首次打开认证并额外救回的病例。结论是实证了后验→动作通路，也否定了当前读包+完整混合选择器在自然盲测的增益和实际风险校准。理论中的条件命题和应用效果分别给结论。 [v16 固定报告](https://github.com/NingMeng0903/RealUS_playground/blob/5e82732afd83712a2b15a0c387551dfe4ec0f563/NMI_idea/experiments/leg_reader_teacher_scale_20261123/leg_scan_segment_risk_20261126_v16/REPORT.md)

### 6.17 大腿截面、血管概率云与旧标注审计

本机 `H/leg_scan_landmark_audit_20261125/`。`anatomy_audit.json`、`five_probability_sections.{json,png}`、`groin_probability_slices.{json,png}`、`ct_scan_landmarks.png`、`thigh_view_v13.json`、`v13_replay.json`。把解剖图中类似“米奇”组合的候选分别核对，股动/静脉、大隐静脉、股深支、腘段等是否在原始源标签、聚合概率图、仿真匿名对象和实时读图中分别有支持。注意截图在不同探头朝向下可镜像，分叉/汇入要沿站位看连续路径。统计“进入视野”的血管候选和“在概率云中有独立的命名通道”是不同检查。考虑副隐静脉、穿通静脉、膝降动脉、旋股支等近邻时，以原始命名源及 `anatomy_audit` 为准，勿从占据峰或邻近主干推导命名标注。早期 MedSeg、BodyParts3D、Open3DModel 和其他图谱各自覆盖不等。独立 landmark 审计属于解释和可视化，不能替代多对象辨认实验。

具体起点可查该 JSON 的 `native_ct.layers_with_named_vessels` 和 `native_voxel_contact_by_label`。本机原生近端 CT 标注通道 CFA=21、CFV=23、GSV=27、DFV=29、SFA=35、DFA=36、FV=39。三个近端典型位置中，CFA/CFV 常并排，报告的 GSV/CFV 接触主要在原生层 14–17，CFA 与 SFA、DFA 的交界是原生层 15/16。由这些信息可以在这例 CT 核对分叉，而远端累计图谱需要各自查 `five_probability_sections.json` 和轨迹视野。不能把原生 CT 的层编号对接为全腿统一标注。

## 7. 桌面 belief 论文与技术对照

**本机目录 `/home/camp/Desktop/belief` 实际有 29 篇 PDF**。`sources_20261117.json`、`sources_20261120.json` 保存大部分原文链接、页数和校验。附录 `paper_inventory.tsv` 逐篇列名称、绝对路径、文件指纹、原文网址和用途。这里的比较是给正式相关工作审读的起点。优先读方法、观测模型、动作权限、损失、数据划分和终局指标。论文里的观察特征、训练真值权限与我们的系统必须逐项对齐。

| 桌面 PDF 文件 | 方法内容 | 对本项目的作用和边界 |
|---|---|---|
| `2005_Diffeomorphic_Mappings.pdf` | LDDMM 用速度场和变分度量构建平滑大形变。 | 材料运输和有限形变的对照。SMPL-X 绑定、离散深度场及内部血管对应没有包入 LDDMM 保证。 |
| `2011_DAgger.pdf` | 学生 roll-in 在自己的状态分布采教师标签，数据聚合。 | 支持学生失败前缀训练。当前需要先执行候选并得续扫后果，标签不局限于老师选中的一个动作。 |
| `2017_Choudhury_IPP.pdf` | 特权信息规划者为有限资源主动信息采集给动作/后果，学习能从公开状态选的策略。 | 与动作教师训练很接近。比较 oracle score、环境后验和终局图像交付是否一致，重点核查同信息下的可继承性。 |
| `2017_DART.pdf` | 对教师演示施加与当前学生误差有关的扰动并让教师恢复。 | 解释为什么应按学生的配准、失检和运动残差造难例，并评估自然失败分布。噪声强度和种类要有来源。 |
| `2017_Reverse_Curriculum.pdf` | 从目标附近采起始状态，构造可学的反向课程，再向更远起点扩展。 | 可组织单区失败→多区→整趟。真实历史、位姿速度和可行尾段须从同一前缀复原。 |
| `2018_Dense_Object_Nets.pdf` | 用自监督稠密描述子找跨视角物体点对应，服务操控。 | 提供跨窗实例对齐的 embedding 先例。局部图像配对与具名解剖身份、漏检和整趟交付还需图谱和任务模型。 |
| `2018_Gaussian_Process_Morphable_Models.pdf` | GP 和低秩近似给出形状形变的统计分布。 | 用于同体形下异血管布局、相关先验和不确定性。当前几何造数只用有限图谱、平滑扰动和有限关系，不声称已拟合人群 GP。 |
| `2019_ChauffeurNet.pdf` | 示教结合偏离/扰动轨迹以学习纠偏和鲁棒驾驶。 | 可比较成功轨迹和学生失败编辑样本的价值。偏离要实际执行后评分，勿把合成错误图当可恢复行为。 |
| `2020_Popovic_IPP.pdf` | 航空地形监测的信息路径规划，使用地图及多尺度感知做更新和在线规划。 | 代表信息或覆盖目标强基线。我们的终点是正确血管和合格图像，动作的信息量可能与交付排序不同。 |
| `2021_A2D.pdf` | 优化特权专家对学生实际可利用的信息和预期回报，再不对称模仿。 | 最贴近教师可继承性。我们的 teacher 以冻结 reader+续扫产生监督，不必直接学一套全知专家神经策略。 |
| `2021_MipNeRF.pdf` | 通过锥台积分和 multiscale encoding 表示有限像素射线的区域支持。 | 有助思考有限超声厚度和声束投影。当前模拟是点回波和有限三维相交，没有实现辐射场像素生成。 |
| `2021_iCEM.pdf` | 给交叉熵规划加入时间相关噪声、候选保留和重用，减少迭代成本。 | 可作连续短程序候选生成 / MPC 复用。与控制收益比较时保持相同 observation、执行和计算资源。 |
| `2022_COS_POMDP.pdf` | 用对象间相关性在 POMDP 搜索中借助非目标对象。 | 对相对位置和邻近结构先验直接相关。超声还要把匿名相邻候选关联到同一真实实例，并验证错误身份。 |
| `2022_ProDMPs.pdf` | 统一动态与概率原语的时间积分基，可给段的边界和观测做条件化。 | 短段执行/重规划工具。可公平与简单样条或反馈原语比较，运动表示与身份正确率之间需实走消融。 |
| `2022_VesNet_RL.pdf` | 分割、运动/历史状态与仿真奖励，用循环 actor-critic 学标准血管长轴探头导航。 | 很近的仿真到真实结构化观测基线。任务是局部标准视图，六区具名身份错支和前后同对象复查是另一终点。 |
| `2023_GPOMCP.pdf` | 在部分未知环境的目标搜索中使用生成式 POMDP 搜索和估计的结构。 | 可参考扩充未知环境/可见性观测分支。强公共规划只使用公开模型，不得在搜索中直接读取隐藏实际布局。 |
| `2023_MP3.pdf` | 以运动原语参数化策略，重规划并保持轨迹段的反馈衔接。 | 可比较短段决策、重规划频率和学习参数。体表接触和身份证据需另外纳入任务合同。 |
| `2023_Reference_Based_POMDP.pdf` | 参考策略辅助部分可观测规划。 | 可作为 MPC/规则参考与学习策略共同规划的对照。要对齐允许行动及观察权限，优先比整趟完成。 |
| `2024_Geometry_Aware_DMP.pdf` | 在具有几何结构的空间上统一 DMP 的状态、误差和运输。 | 有力的几何运动先例。本文的额外要素在材料地址、解剖身份和真实物理切面验证，而不是仅将笛卡尔动作改成局部坐标。 |
| `2024_IntervenGen.pdf` | 实跑当前策略触发失败，把少量恢复动作变换到新构型，复走并验证后聚合。 | 与失败编辑和定向恢复最接近。比较其人工恢复示教、训练只保留成功干预，与我们的仿真配对后果和正负零监督。 |
| `2024_RACP.pdf` | 面向多模态预测的风险感知条件规划。 | 对比在多个未来模式上维护风险和备选路线。我们的模式包括血管身份和实际成像完成，不从 MAP 一条预测替代整段风险。 |
| `2024_RUMI.pdf` | 以互信息选择交互扭动物体，减少遮挡下形状或对象不确定性。 | 有用的信息论对照。实验可分联合交付和不确定性变化，单独减少熵不足以解释整段扫描成功。 |
| `2024_TCE.pdf` | 对时间相关的 episode 运动技能做 step-based 更新。 | 可参考宏动作时间归因。已有 v9 PPO、v10 残差 PPO 和 v15 教师价值学习仍按各自实际算法引用。 |
| `2025_IntraSweep_Ultrasound_Embedding.pdf` | 同次 sweep 用追踪到的相近/远离探头位置构造检索监督，加入拒匹配。 | 训练超声 embedding 的最近方法之一。近位姿配对不自动提供跨形变、跨窗同一血管实例和具名身份。 |
| `2025_SkillMimicGen.pdf` | 以技能及动作模板自动生成可执行示教，覆盖多配置和转接。 | 适合比较可执行恢复和数据生成。用户系统的目标还包括补证后的身份修订和整程交付。 |
| `A unified framework for data association\naware robust belief space planning and\nperception.pdf` | DA-BSP 在候选动作的未来观测中传播多峰关联后验。 | 最接近信念层。应比较未来缺测、混叠与候选关联，而非只比较目标能否进入视野。 [作者原文](https://indelman.github.io/ANPL-Website/Publications/Pathak18ijrr.pdf) |
| `Learning over Subgoals for Efficient Navigation of.pdf` | 学未知区域子目标属性并做长时域期望代价规划。 | 可能的高层程序价值目标。子目标需同时含可辨图像和合法出口。 [CoRL 原文](https://proceedings.mlr.press/v87/stein18a.html) |
| `Real-World Reinforcement Learning of Active Perception Behaviors.pdf` | AAWR 用训练期特权传感估优势，从粗策略和少量示教学真实机器人主动感知。 | 强实机主动感知先例。比较特权 critic、策略可见信息、状态访问及真实硬件，与训练 reader 修订旧对象的机制不同。 [NeurIPS 原文](https://papers.nips.cc/paper_files/paper/2025/hash/aa1b1a959c80086cba61d0fd66de412f-Abstract-Conference.html) |
| `Towards-Map-Agnostic-Policies-for-Adaptive-Informative-Path-Planning.pdf` | 让自适应信息路径策略共用不同地图表示。 | 检验地图配置或形体变化下的迁移叙事。我们的目标身份、反事实续扫和合格图像验收须保留。 [RA-L 作者原文](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/rueckin2025ral.pdf) |

程序读取含换行文件名时用 `Path.glob('*.pdf')` 或 `find ... -print0`，附录 TSV 用字面 `\n` 表示实际换行。不要把 README 中“下载候选”误算成桌面现有 PDF。专项补充包有 Jiang 单管自主超声、Goel Bayesian optimization、自主扫描 Residual RL 三份。其 `README.md` 另索引 Al-Zogbi 股动脉模体、Bi gaze-guided、Morales bifurcation。已经深读的内容优先查 `[H]/leg_scan_full_route_20261124_v9/PAPERS.md`、`[H]/leg_scan_identity_evidence_20261124_v11/RELATED_WORK.md`、`[H]/leg_scan_target_routes_20261123_v2/literature/local_review.txt`。以上逐项研究方向来自本机文献和已有精读笔记，本清单不主张重新复现全部论文的方法。

### 7.1 最需要逐方法对照的补充原文

| 原文 | 比对技术问题 |
|---|---|
| [Senden 等：Multi-Hypothesis Tracking in a Graph-Based World Model](https://doi.org/10.1109/LRA.2023.3300282) | 先被动扫描，旧测量关联存成多个对象假设，再按假设预测缺失目标的位置主动补看。与“多假设对象图、补证、重新解释旧观测”高度接近。需要对照任务是否允许旧图成为当前交付、材料地址随形变如何引用、实际探头末态怎样影响后续任务。 |
| [LOLS](https://proceedings.mlr.press/v37/changb15.pdf) · [Deeply AggreVaTeD](https://proceedings.mlr.press/v70/sun17d/sun17d.pdf) · [SEAL](https://arxiv.org/pdf/2112.01001) | LOLS 用 roll-in/roll-out 代价，AggreVaTeD 用专家 cost-to-go，SEAL 对动作采集和感知学习分别训练。给三块可训练闭环找各自强先例。我们的 candidate 反事实必须实际带噪取像，冻结真实续扫，能给看过的对象追踪身份变化。 |
| [LSP-AIG](https://arxiv.org/pdf/2403.03269) | 以揭示未知地图对未来任务的价值训练长时域取证。可作反事实信息价值先例。参照值要接图像交付、实际出口和身份歧义。 |
| [Anatomy-Anchored Learning](https://arxiv.org/abs/2610.32525) · [SAMe](https://arxiv.org/abs/2604.25646) | 初始解剖接触固定锚与条件流式策略，以及从人体外图到器官和探头初始化。二者让“身体相关坐标”或“从人体先验选探头初态”本身不足以成为论文唯一增量。比持久内部证据、跨窗关联、变形后的切面重建和后果学习。 |
| [SonoGraph-WM](https://arxiv.org/abs/2609.32837) · [SonoAtlasWorld](https://arxiv.org/html/2611.26867v1) | 前者预测未来图与探头位姿并频繁再规划，验证图基于标注衍生观测。后者用持续 CT 资产生成动作条件因果超声，训练数据扩大后未见解剖关键切面成功提高。正文须对比真实观测权限、匿名身份、不确定性、任务成功定义和造数规模。 |
| [UltraBot](https://pmc.ncbi.nlm.nih.gov/articles/PMC12375115/) · [Automatic robotic Doppler sonography of leg arteries](https://doi.org/10.1007/s11548-024-03235-7) · [DopUS-Net](https://doi.org/10.1109/TASE.2023.3277331) | 超声强基线要包含正常推进、独立阶段判定、Doppler 或时序跟踪和失检恢复。动脉 Doppler 实际实验与静脉匿名点仿真不可直接比百分比，但提供更真实动作和观察的对照。 |
| [GLFS/融合后的校准](https://motion.cs.illinois.edu/papers/ICRA2024_Marques_OverconfidenceSemanticMapping.pdf) · [Neural Solver for MOT](https://arxiv.org/pdf/1912.07515) · [ByteTrack](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php) | 和 v4–v8 错关联、融合失配相关。边真值、对象后验、赢家的拒识及最终交付分别校准。弱回波可帮助接轨，证据强度与是否连边分权处理。 |

## 8. 请第三方给出的审查报告

1. **复原代码和证据链。** 按「理论命题 → 原始 TeX/公式 → 代码中的实现状态 → 冻结面板 → 配对结果」给表。把材料地址、相对拓扑、分型概率、状态融合、反事实教师、RL、ProDMP 和 dIRD 分行。写明各自启用的版本与最强对照，按需要给 commit 固定链接或本机绝对路径与行号。
2. **重算至少四个结论。** v10 共用读图器的两路线整趟收益；v14 同轨常规、同移、扭转的净增与错支；v15 同轨读图、同新版控制的冻参照和更新学生、三带错支净降；v16 四臂自然/剪切联合验收、真实几何违反率和模型认证。区分严格交付、采到目标、错支趟数、伪回波错交、同一候选改判和旧候选池换赢家。报告分母、分布和救回/损失，避免把随机带当独立解剖。
3. **核对训练权限、数据和算法。** 读图、教师、学生各自实际训练哪些权重，老师怎样模拟及验收，PPO/BC/协同监督各出现在哪个面板。对 v9/v10/v15 分开。读图训练与 v15 teacher 数据并非同一行数据，也未共享端到端梯度。复算时若只读预聚合，标注汇总核对；若重新读逐例，标注重算；若实际重放，给配置、环境、哈希、失败口径。
4. **用强对照组织创新。** 用 V9.1 的几何类型、V8.2 的对象证据接口、V8.3/4 的证据—真实出口耦合、v10 的分区恢复、v15 的后验驱动错支下降和 v16 的盲测负结果来判断合适主张。PPO 和 GRU 是学习器，三部件联合组织的反事实训练是独立方法候选。核实哪一环能证明效果、哪一环只有接通、哪一环支撑数学命题。对照理论先例和实证先例，指出强重合时具体重合的假设、算法和验收。
5. **明确哪里不好。** 必须写入 A/B 移动 Y 对强 MPC 持平，早期长段 Blender 解剖只有一份，v4 错关联传播，v7/v8 门控和可继承样本，v9 BC 与 PPO 控制增益难分，v10 的高置信误认与额外采集，v11 的单次误分配，v12–v13 的增益与复验差，v14 的具名拓扑标签缺口，v15 固定旧赢家纠错为零和耗时，v16 读包选择坏及认证校准不足。任何更精确表述以相应逐例结果为准。
6. **提出最小的决定性对照。** 若要宣称「反事实训练优于启发式」，可固定 reader、展开范围、合法菜单和学生容量，以当前学生实走及同噪声带比较动作监督标签。若要宣称「跨窗证据改变身份并改善后续」，固定同一次取证、候选、旧 ID 和真实出口，只干预可读新证据及融合规则。若要宣称「证据与出口联合优于分开」，保持证据分布和出口分布，作包含实际可达见证的配对干预。先确定各试验证明的命题，避免继续优化一项边指标就宣称整条链成立。优先准确交付，不新增任意时长路程门来制造差异，实际运动、时间和图像均报告。
7. **给文章组织建议。** 以 T-RO/IJRR 读者可复核的命题给 150–250 字摘要思路、三项贡献、四张关键图/表各负责的证明、和证据限制。把实际人体切面、离线原生轨迹、体模、质心仿真分开。已知脚本输出和空缺结果都写清楚，无须把每个技术一样力度推上首页。讲清最强新对象是如何从局部证据与物理末态推出整段检查可执行性，并用定向恢复、错支、实际退出案例支持。

### 最小只读检查命令

在同一 Ubuntu 主机使用以下变量即可复核路径。外部机器通过 GitHub 固定链接阅读，或根据自身挂载重新设变量。此代码块用于定位文件，执行只读。

```bash
leg_handoff='/home/camp/Documents/Codex/2026-11-22/new-chat-3/outputs/leg_scan_cli_handoff'
research_repo='/media/camp/EXT_DRIVE/RealUS_playground'
idea_tex="$research_repo/NMI_idea/experiments/ab_validation_20261110/ideas"
belief_papers='/home/camp/Desktop/belief'
handoff_bundle="$leg_handoff/third_party_agent_handoff_20261126"

git ls-remote --heads https://github.com/NingMeng0903/RealUS_playground.git
rg --files "$idea_tex" -g '*.tex' -g '*.pdf'
rg -n 'prop:plane|eq:compatibility|thm:separable|eq:kernel|eq:rowspace|eq:gram' "$idea_tex" -g '*.tex'
find "$belief_papers" -maxdepth 1 -name '*.pdf' -print0 | xargs -0 -r -n 1 printf '%q\n'
rg --files "$leg_handoff" -g 'REPORT.md' -g 'REPORT.txt' -g 'METHOD.md' -g 'DESIGN.md'
gzip -dc -- "$handoff_bundle/file_inventory.tsv.gz" | rg 'v16|V9\.1|belief' | head -100
```

公开最短阅读顺序是固定 `cdb1705a` 的 `B_REVIEW_REQUEST.md` 和 V9.1/V8.3，随后 v10 `REPORT.md`、v15 `METHOD.md` 与 `REPORT.md`、v16 `METHOD.md` 与 `REPORT.md`。本机有条件时先查 `SOURCE_ENTRY_POINTS.md` 定位精确公式和复算函数，再读压缩逐例。所有新判断须有来源和证据级别。




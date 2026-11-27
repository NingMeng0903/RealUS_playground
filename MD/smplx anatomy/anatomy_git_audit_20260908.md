# Anatomy Git 独立审计（2026-09-08）

范围：只读检查 Git 对象、解剖目录 diff、函数实现与既有日志；本报告没有重新 bake 或回放历史候选。唯一新增文件为本文。**提交说明常写控制器，不能用说明判断解剖版本；候选版本也不与提交一一对应。**

证据标记：**代码确认**＝从指定提交直接读取；**本日资产检查**＝本会话对冻结 operator 的真实数组检查；**历史记录**＝引用 [todo_ana.md](todo_ana.md) 或 [20260813 状态记录](todo_ana_status_20260813.md)，不是本日复算的指标。

## 1. 提交与实际实现

| 提交 / 日期 | 解剖代码中实际发生的事 | 证据边界 |
|---|---|---|
| `142ece5f0bc646978ae3e8c9add76deea71c26a2` / 07-30 | 当前冻结 operator 的来源基线；235 controller、14-slot Blender LBS、SMPL-X→controller 映射与 coupled joint 响应 | 联动成立不等于内部几何完全位于皮内；脚部残差在提交说明和后续记录中均存在 |
| `29e1072c6d0b27005b140cb356b28ab618331354` / 07-30 | **该提交自身只改两个文件**：`articular_fit_v8.py`、`cli/pose_runtime_v8.py`，104 插入/38 删除；包含 coupled calibration 处理和 EasyMocap root/pelvis 平移补偿 | “骨头细”是当时现象记录，不能据提交标题断定这两处新增了缩骨算法 |
| `142ece5f → 29e1072c` 累计差 | 19 文件，6052 插入/267 删除；含 V8.11 soft-volume 保护、foot station transport、head/FK/tube corrective 扩展 | 必须区分区间差与 `29e1072c^ → 29e1072c` 单次差 |
| `31133afba2ced3f4de01df7328d487859c7f9b05` / 08-02 | 已恢复主要 142 生产核，另有 `anatomical_calibration_v1`、whole-chain rest-fit、`pose_map_v1`、V2，以及 metadata-gated live preview | Pack A 是 142 materialize 的对照语义；不是该提交下所有新 solver 都可用 |
| `2b2ff3c09a87a6a880742055728ea2cc45c09fad` / 08-03 | 加入 `_quarantine_v4`；V5 明确继承 node2_004 whole-chain，校验唯一 `C_total` | 单一 correction 修正了合同，不自动解决穿出；V4 图失败属于历史记录 |
| `45718ba8` / 08-03 | 同一个提交加入 V6、V7 CLI；`pose_map_v1.apply_pose_map_global` 从 parent-local FK 改成 global right-multiply；V7 改股骨中心线方向 | V6/V7 是不同候选，而非两个可单独 checkout 的完整提交快照 |
| `c1362750` / 08-03 | V8 增加 `_axial_scaled_pivot_rotation`、`_select_bone_first_femur_scale`，范围 0.97–1.03 | 这里可以从代码直接确认轴向缩放；不是径向或各向同性缩放 |
| `4caafa08` / 08-03 | V9 增加 `_select_femur_embed_v9`，股骨尺度范围0.88–1.05，加入方向、座合/inside 搜索与接触量 | 屈膝 medial gap 18→64 mm 是历史失败记录，未于本日重算 |
| `94c24c48e91dfc02df5c24958c21d34f25da0989` / 08-06 | 一次加入 `pose_map_v10`、`anchored_rest_fit_v11`、V10/V11 CLI/门/工件 | 提交中的 V10 已是 hybrid 根冻结版本；不能用它回放历史 fk-only |
| `3edf1869` / 08-13 | 新增 V12 absolute poke、joint/linkage、deep-flex、验收/渲染入口 | 属于度量与验收新增，不能把新门通过反推旧几何正确 |
| `3a1715b1 → 8a9e32f7 → 88ee1fbf → 05a6c34e` / 08-13–14 | 加入并迭代 `terminal_reseat_v12`：有界终端 reseat、右乘终端运动、深屈拟合、Arch 拆簇 | 名称主要仍写 V11/V12b，必须同时读报告的 terminal policy 和源实现 |
| `383a40b8` / 08-15 | 加入前臂 mesh-only reseat 与 V10 FK 评分；同时加入按整 mesh owner 选 follower 的错误 | 是 V12e 前臂骨改善但血管漏跟的直接代码来源 |

历史日志中的 `227cfeb7cdef54a6db4448b8712d9a38e818f8d4` 与当前祖先 `94c24c48` 的 **anatomy 子树完全一致**，Git tree ID 均为 `d3e203fd574f6747e9cbe0f709172bf1167179e8`。它们是不同提交历史中的相同解剖代码，日志没有指向另一种 V11。类似地，`2644c671` 与 `383a40b8` 的 anatomy tree 均为 `43a66c99172602ddb0a98862c5aa807a6b29c602`。

## 2. 五个故障的可核查因果链

### A. “29 细骨”：现象成立，精确首因尚不能由该提交确认

**代码确认：** `142ece5f:source_skin_volume.py:apply_source_skin_volume_registration` 已有 `mapped = query + delta` 的全材料调和场，并报告 `all_material_volume_field_applied_before_bone_fit`。同处明确把 radial-section shrink 标为 disabled。全材料场一般不能保证骨形刚性：`det(J)>0` 只说明不翻转，不能推出 `JᵀJ=I` 或骨干半径不变。

然而 `29e1072c` 树新增的 V8.11 selective bake 恰恰调用 `preserve_protected_material=True, rebind_source_rig=False`，把非 soft 顶点恢复为源值；不能把这一保护分支写成“新增径向缩骨”。**历史记录**把 29 候选称作“细骨硬塞”，但缺少当时所看资产 SHA、完整 build 配置和逐点历史回放，本文不把某个体积场调用认定为已经复证的唯一首因。真正代码可确认的后续缩骨是上表 V8/V9 轴向尺度。

### B. V2 geometry/bind 分离与重复 correction 风险

**代码确认，`31133afb:dynamic_main_chain_retarget_v2.py:build_dynamic_main_chain_retarget_v2`：**

```text
rest = legacy.vertices_final             # 已按旧 whole-chain 搬运
bind = legacy.B_prefit                   # bind 却退回搬运前
base_target_global/local = bind/source_local
C_bone = FK(target_local) @ inv(bind)
rest[terminal] = C_bone[root] @ rest[terminal]
rest_transport[desc] = C_bone[desc] @ legacy.C_bone[desc]
tubes = LBS(legacy.vertices_prefit, rest_transport)
```

骨网格、主链 bind、管采用不同的组合起点；不是仅凭 T-pose identity 可以认证的同一映射。**不要简化成“所有点一定乘了两次相同 C”**：直接看到的是 mesh 已经搬运、bind 被重置，以及 terminal/bone/tube 组合次序不同；具体重复量由原 C 决定。

V5 的直接修正是继承 node2_004 的完整 `B_final`，要求 `C_total = B_final @ inv(B_prefit) = C_bone`，shape 必须为235×4×4；runtime 直接委托同一 `pose_whole_chain_vertices`。历史日志撤销 V2 验收并记录手足断链；本报告没有重算其毫米值。

### C. V6/V7 right-multiply 修了终端，却保留源 pivot

**代码确认，`45718ba8:pose_map_v1.py:apply_pose_map_global`：**

```text
旧：D_local = inv(L_source_rest) @ L_source_pose
    G_target = FK(L_target_rest @ D_local)
新：G_target = G_source @ inv(B_source) @ B_target
蒙皮：G_target @ inv(B_target) = G_source @ inv(B_source)
```

终端若 `B_target=B_source`，运动精确保持 copy-142，避免 ancestor 改动而手指 rest/bind 仍旧时的 rebase 错位。但 target bind 在蒙皮矩阵中约掉了；搬 target pivot 本身不会改变单控制器的旋转中心。V7 的 rest 几何与腿方向变化仍然有效，不能误写为“所有 rest-fit 都被约掉”。

### D. V10 target FK 与 V11 mesh/bind 不一致

**代码确认，`94c24c48:pose_map_v10.py:apply_pose_map_global_v10`：** 主链恢复 `FK(L_target_rest @ D_source)`，膝/肘围绕 target bind 运动；手、足连同 wrist/ankle 根整体用 `G_source`，避免根权重仍被新 FK 拉动。提交中已有此 hybrid 策略；历史 fk-only 的37 mm腕 rebase、`hand_L≈0.001`仅见代码注释/工件/日志，不能拿这个提交的渲染自称独立复现 fk-only。

**代码确认，`anchored_rest_fit_v11.py:build_anchored_rest_fit_v11`：** 对选定铰链把 `B_final[:3,3]` 恢复为 `B_prefit[:3,3]`，但 `vertices_final` 留在 V7。随后重算 inverse bind，因此 zero-pose 恒等仍可过。运动时骨形与轴不相容会被放大；纯旋转中心偏移 d 的位移量可达 `2d·sin(θ/2)`，这是几何推导，不是本日复算的 V11 误差。历史状态记录确认左前臂穿出回退，同时膝/髌有收益。

V12b 的 `8a9e32f7` 把终端绝对 `G_source` 更新为 `G_source @ inv(B_source) @ B_target`，让新增 reseat 随肢体运动；直接左乘世界偏移不等价。V12d/e 的 `_v10_skinning_transforms` 也将拟合评分改为同一 V10 FK，避免只按 source affine 计分看不到前臂真实穿出。

### E. V12e 前臂血管漏跟：已经直接确认的代码错误

**代码确认，`383a40b8:terminal_reseat_v12.py`：** `_cluster_follower_ids` 按 `source_mesh_controller_bones` 的整 mesh owner 选择 bone/vessel/nerve；`apply_terminal_reseat_v12` 的 mesh-only 分支仅对这些 `vertex_ids` 施加刚体 T，bind 不动。

**本日资产检查：** 冻结资产中 Artery、Vein 整管 owner 是 `Head_Bone`，Cervical_Nerves 左右 owner 是对应肩 controller；因此它们不会被选入前臂 follower。但真实14-slot权重中，动脉每侧有295–347个、静脉448–456个、颈神经246–268个顶点受到前臂 controller 影响，最大权重约0.8–0.985。这里的计数按每个 controller 分别统计，同一顶点可出现在两个计数中。

所以“同管 owner”不能替代逐顶点联动。应按原 frozen 权重构造一次共享 correction/材料绑定，并用 held-out pose、管拓扑、管骨相对运动共同验收。本文只确认旧故障，不提前认证正在实现的新候选。

## 3. 防止历史复现再次串版本

1. 同时记录代码 commit、operator/subject SHA、pose composition、terminal policy、实际 runtime 函数；只写 V7/V11 文件夹名不够。
2. 保存候选时保留构建时的数学语义。V10 renderer 曾用最新 hybrid runtime 回放 fk-only 目录，历史记录指出图字节相同；这是回放身份缺陷，不是两个算法等价的证据。
3. `31133afb` 之后的 142 普通资产与 whole-chain preview 是两个入口；`anatomy_lbs.source_bone_posed_global` 仅在 metadata 含 `whole_chain_source_bind_global` 时走新 preview basis/FK。不能将 preview 结果归因到普通142 runtime。
4. 本日没有 checkout、reset、修改历史 solver 或重新发布资产；历史数值始终标为引用，新增验证结果另存新候选报告。

可重复的审计入口：

```bash
git show --stat 29e1072c -- src/projects/genesis_ue_sync/anatomy_retarget
git diff --stat 142ece5f 29e1072c -- src/projects/genesis_ue_sync/anatomy_retarget
git log --all --format='%h %cs %s' -- src/projects/genesis_ue_sync/anatomy_retarget/pose_map_v10.py
git show 31133afb:src/projects/genesis_ue_sync/anatomy_retarget/dynamic_main_chain_retarget_v2.py
git show 383a40b8:src/projects/genesis_ue_sync/anatomy_retarget/terminal_reseat_v12.py
git rev-parse 94c24c48:src/projects/genesis_ue_sync/anatomy_retarget 227cfeb7:src/projects/genesis_ue_sync/anatomy_retarget
```

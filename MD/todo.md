# Neural IRD / 可达算子 — 全期 TODO

顶层包：[`ird_playground/`](../ird_playground/)（与 `rm75_control/` 同级）。

## 定案（P1/P2）

**构建并训练锁定导轨的 7-DoF 通用 Neural IRD。** 导轨作查询/轨迹优化变量，经

\[
T_{\mathrm{base}}(\mathrm{rail}_y)=T_{\mathrm{rail}}\,\mathrm{Trans}_y(\mathrm{rail}_y)\,T_{\mathrm{base},0}
\]

完整 SE(3) 组合进入算子，**不增加网络输入自由度**。主输出可达裕量 \(m\) 与舒适质量 \(q\)；区域查询采用固定 Sobol/QMC 扰动与 softmin(\(m\))+mean(\(q\))；IPE 仅 ablation。

| 层 | 内容 | 训练？ |
|----|------|--------|
| 底层点场 | \(f_\theta(\Delta T)\to(m,q)\)，\(\Delta T=T_{\mathrm{tcp}}^{-1}T_{\mathrm{base}}\) | 是 — 全工作空间通用；无 rail / 轨迹 / 人体 |
| 区域 A | 固定 Sobol；\(m_{\mathrm{robust}}=\mathrm{softmin}(m)\)，\(q_{\mathrm{region}}=\mathrm{mean}(q)\) | 否 |
| 区域 B | 蒸馏加速头（标签=A） | 二期+ |
| 导轨 | 查询层 `rail_y` → \(T_{\mathrm{base}}(r)\) → \(\Delta T(r)\) | 否 |

整机运行：7-DoF arm + 1-DoF rail = 8-DoF；能力图与网络均为 **rail locked** 的 7-DoF。

---

## P0 — 包骨架与文档

- [x] 本文件 `MD/todo.md`
- [x] 根 README 登记 `ird_playground/`
- [x] `ird_playground/`：`env.sh` / `pyproject.toml` / `README.md` / 包目录
- [x] 默认探头 SE(3)（link7 → Trans_z(7cm)·Rot_y(+90°)·Trans_z(5cm)）

## P1 — 离散 CapabilityMap + 加密 IRD GT

- [x] 参数化探头注入建图；横装探头 + 探头杆碰撞 URDF
- [x] **1.5cm** 横装+杆碰撞能力图（`rm75_6f_1p5cm_15deg_coll_probe`），rail locked
- [x] 加密 GT：~2M；分层 35% 可达内 / 40% 边界 / 25% 不可达
- [x] GT 字段：`m_gt`, `q`, 分解因子；尽量 `q_best` / top-K；SE(3) 度量 \(\sigma_p=3\mathrm{cm},\,\sigma_R=10^\circ\)
- [x] CLI：`build_map` / `build_ird_gt`

## P2a — 通用 Neural IRD 点场（裕量 + 舒适度）

- [x] \(f_\theta(\Delta T)\to(m,q)\)：\(m\) logit/裕量，\(q\in[0,1]\)；点代价 \(\mathrm{softplus}(-m/\tau)-\lambda q\)
- [x] 输入：归一化 xyz + rot6D；Fourier **主要在 xyz**（\(L_p=6\)）
- [x] 网络：4–6 层 residual MLP，hidden 256，SiLU/Softplus
- [x] 损失：BCE(m) + margin 回归 + 可达上 q + 局部差分一致性；难例挖掘
- [x] 训练不含：人体、血管、\(s\)、**rail**、患者、特定轨迹
- [x] **过关**：分类 IoU / q 回归 / Spearman **+** 梯度余弦 / 上升改善 / `rail_y` AD·FD / 区域 softmin 改善
- [x] CLI：`train` / `eval_point`（需扩展验收）

## P2b — 查询侧区域 A

- [x] 各向异性片区 + 固定 Sobol
- [x] 聚合改为 softmin(\(m\))+mean(\(q\))；antithetic 配对
- [x] API：`T_base_from_rail_y` 完整 SE(3)；`score` / `region_score` 输出 \(m,q\)
- [x] **不做**：IPE 默认、局部人体网、manipulability 球主表示、rail 入网

## P3 — 假轨迹 NLP

- [ ] \(C_{\mathrm{IRD}}(s)=\mathrm{softplus}(-m_{\mathrm{robust}}/\tau)-\lambda q_{\mathrm{region}}\)
- [ ] 沿轨迹 softmin（木桶）；输出 `rail_y(s), q(s)`
- [ ] 固定 Sobol K=16–32，勿每步重随机

## P4 — 体表 / 纤维 / 血管

- [ ] 接入 `leg_volume_coordinates` / 血管 GT
- [ ] 纤维丛朝向约束进 \(R_{\mathrm{task}}(s)\)
- [ ] 偏离仍贴肤

## P5 — 对接 rm75_control

- [ ] 导出轨迹 → `MotionReferenceSource`
- [ ] `phase_hybrid_track`；DMP 式伺服可经 hybrid（`pose_d`+`vel_ff`）
- [ ] 不能直接灌关节角进 hybrid

## 后期

- [ ] 可选区域头：对片区直接密集造 GT，不必从 A 蒸馏
- [ ] IPE / hash-grid / SIREN 仅研究对照
- [ ] GPU 批 FK（碰撞仍 CPU）；端到端 CUDA 碰撞不做

## 明确不做（一期）

- 为某条腿/某条轨迹单独训练网络
- 把 `rail_y` / 患者 / 血管作为网络输入
- Mip-NeRF IPE 作为默认区域算子
- manipulability 单独决定可达性
- 跨包深层 import（文件/子进程对接 reachability）

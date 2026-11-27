# V16 独立图审与 guarded-field 复核（2026-09-09）

本记录只覆盖我实际打开的 Genesis 图、`guarded_visceral_field_v16.py` 的有界审查和小型回归测试。没有改运行时、fit 或渲染实现，也没有把投影观察当成三维碰撞证明。

## 当前结论

运行合同层面，guard fixture 的 5 个测试通过；新失败 rest clip 的 renderer manifest 也显示三路 context depth 有效、focus mesh 明确导出。解剖层面仍失败/待复核：V16 共享场和本次 clip 都没有证明骨—软组织、管—骨或三角面相交已经清除，`guard_004` 最终图审仍待生成。

## 两个 beta 的既有 baseline 图审

我实际打开过 213328 和 213712 的 clean baseline Genesis：两种体型各自的 T-pose 与 `sitstand_sid4336_middle` 的 `context`、`viscera`、`skin` contact sheet。图中肺—肋笼、心脏—纵隔、肝—肋弓和肠管—骨盆的宏观相对位置连续；从这些投影没有确认确定的脏器—骨骼冲突。肋骨覆盖肺缘不能单凭颜色/遮挡标为穿入。缺少同姿态 before/after 深度和三角相交标记，因此髋臼—股骨头、骨盆—肠管、肋骨—肺以及血管/神经—骨的深度关系仍不确定。

实际图目录为：

```text
/home/camp/anatomy_retarget_outputs/v16_viscera_baseline_genesis_20260908_002/
  213328_tpose/candidate/{context,viscera,skin}/contact_sheet.png
  213328_sitstand_sid4336_middle/candidate/{context,viscera,skin}/contact_sheet.png
  213712_tpose/candidate/{context,viscera,skin}/contact_sheet.png
  213712_sitstand_sid4336_middle/candidate/{context,viscera,skin}/contact_sheet.png
```

这批 manifest 标记 `solver_or_fit_run=false` 且只有 candidate variant，所以只能证明材料选择和渲染可读性，不能证明固定共享场已经接入生产 pose。

## Colon bounded candidate 图和数值边界

我实际打开过 `v16_colon_focus_genesis_20260908_003` 两个 beta 的 sit-stand context 对照图（anterior、left、oblique、posterior）。candidate 的紫色 colon loop 相对 `v15_base` 有小而连续的变化；这些视图没有显示一个可以确定的新增骨—colon 碰撞，但也没有足够深度信息证明 clearance。该候选的 native audit 记录 213328/213712 各 1211 帧 finite，Cauchy–Binet polynomial 相对直接 colon SparseLBS 体积误差约 `9.8e-9`；after-volume 仍只到约 `0.8095…1.0101` 的范围，且存在 amplitude saturation。它是数值范围实验，不是解剖通过；体积恢复不能替代正 Jacobian、皮肤内侧、骨距离、管径和分叉检查。

```text
/home/camp/anatomy_retarget_outputs/v16_colon_focus_genesis_20260908_003/
  213328_sitstand_sid4336_middle/comparison/context/torso_{anterior,left,oblique,posterior}.png
  213712_sitstand_sid4336_middle/comparison/context/torso_{anterior,left,oblique,posterior}.png
```

## Guard solver 代码审查和 5 个测试

当前 guard 的可核对行为是：

- skin 输入先用 `_closed_oriented` 检查闭合且方向一致；开口 skin 在 fit 入口 `ValueError` fail-closed。
- 非骨顶点只在 Wendland 支持域内为 active；inactive nonbone 顶点在 checker 中必须与 `current` 精确相等。
- 骨 signed-distance floor 使用 `min(max(initial,current), 0.5 mm) - 0.02 mm`；skin ceiling 使用 `max(min(initial,current), -1 mm) + 0.02 mm`。
- 骨顶点保持 bit-exact；rest edge budget 以 initial 为累计基准，同时保留 current 的既有位移；每轴还有步长界限。
- checker 重新计算实际闭合表面的 signed distance，而不是只相信 QP 的线性化结果。

运行的命令及结果：

```text
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /media/camp/EXT_DRIVE/envs/genesis/bin/python -m pytest -p no:dash -q \
  tests/test_guarded_visceral_field_v16.py
5 passed, 2 warnings
```

回归覆盖：closed bone 外侧目标、骨不可移动、累计 edge/path budget、initial/current clearance floor、inactive 顶点改动拒绝和 open skin 拒绝。实现仍把 `triangle_crossings_tested` 标为 false；AABB 选点加点 SDF/边界约束不能覆盖“顶点均在外、三角形边/面穿过骨”的 counterexample，也不能保证未进入 active band 的细小邻近结构。V15 实际资产还有 open bone mesh 被记录为 skipped 的风险，这不应写成全骨覆盖通过。

## 新失败 rest clip：实际图审

本轮实际打开了：

```text
/home/camp/anatomy_retarget_outputs/v16_neighbor_failure_genesis_213328_20260908_005/
  213328_tpose/comparison/context/torso_right.png
  213328_tpose/comparison/context/torso_left.png
  213328_tpose/comparison/context/torso_posterior.png
```

三张图都是左侧 `v15_base`、右侧 `candidate` 的同相机对照。`torso_right` 中灰色 L1 骨性结构、纵向橙色组织和紫色躯干局部轮廓在两栏均可见，candidate 相对 base 有小幅位置/轮廓变化，但没有从这一侧确认间隙已经恢复。`torso_left` 中 L1 灰色结构仍贴近紫色躯干边界，candidate 与 base 的宏观形状接近；下缘橙色组织的遮挡使深度关系不确定。`torso_posterior` 中灰色 L1 与纵向橙色结构在屏幕上重叠/相邻，两栏均存在这一现象；这是 screen-space overlap，不能据此宣布三角面相交，也不能据此宣布无相交。

该 clip 的 manifest 给出的是显示事实：x 轴 offset `-9.0 mm`、保留 `x >= plane_origin[x]`、uncapped、`input_geometry_modified=false`，focus mesh 为 `L1`、`Anterior_Longitudinal_Ligament` 和 `Diaphragm`，三路 context depth 的 valid fraction 均为 `1.0`。offset 和“保留最深韧带点”来自运行记录/数值选择，不是从 RGB 像素反推的毫米测量；本 manifest 本身没有给出骨—韧带最小三维距离或三角相交数值。该图因此确认了失败区域值得继续测量，但不能替代独立三维指标。

## 待完成

`guard_004` 的最终 Genesis 图审尚未收到；需要在同一 beta/姿态/相机下检查 candidate 与 base，并同时读取三维 signed distance、triangle crossing、正 Jacobian 和软组织—皮肤指标。当前 evidence 足以保留候选为失败/待审状态，不能宣称任意 beta 或任意 theta 已通过。

# Blender 内部解剖 → SMPL-X：论文与当前实现的对应

检索日期：2026-09-08。本文区分论文已有结论、仓库证据和本轮工程假设；不是论文方法的完整复现。Git 证据见 [独立版本审计](anatomy_git_audit_20260908.md)。用户本轮已明确允许受限轴向调整，须保留关节端形状、骨干粗细和相连组织关系。

## 直接相关的研究

| 论文/原始来源 | 论文实际解决什么 | 对本项目的意义与边界 |
|---|---|---|
| Dicko 等，**Anatomy Transfer**，SIGGRAPH Asia 2013，[作者项目页](https://users.cs.utah.edu/~ladislav/alihamadi13anatomy/alihamadi13anatomy.html) | 先转移骨层，再以骨层与扣除脂肪厚度后的皮肤作为边界，用调和变形转移肌肉、脏器等内部结构。 | 骨与皮之间的体积场最贴近“截面仿形”。但仅凭该项目摘要不能推导出任意姿态、无翻转或源 Blender 联动的保证。需要独立关节、Jacobian、动作检查。 |
| Bauer 等，**Anatomical augmented reality with 3D commodity tracking and image-space alignment**，Computers & Graphics 2017，[出版社](https://www.sciencedirect.com/science/article/pii/S0097849317301747)、[作者机构记录](https://www.timc.fr/publication/hal-01633632v1?equipe=GMCAO) | 将通用内部解剖个体化，并随 Kinect 运动显示；登记中加入解剖约束，运行中用运动过滤/仿真追踪骨骼。 | 强调长骨、骨端和关节关系需要一起约束，不能将骨块独立塞进皮内。其运行时含仿真，不符合本任务“只用参数直接驱动”的严格目标，不能照搬或宣称相同保证。HAL 全文访问受阻；记录基于出版社摘要与可检索作者稿内容。 |
| Keller 等，**OSSO: Obtaining Skeletal Shape from Outside**，CVPR 2022，[作者项目页](https://osso.is.tue.mpg.de/) | 从体表推断骨架形状的统计先验。 | 可以帮助估计骨布局；不会自动替换本源资产的235控制器、17管和其关系。仓库旧方向中把骨长先验当硬目标已经失败，不能再次将统计骨架硬套进去。 |
| Keller 等，**From Skin to Skeleton (SKEL)**，SIGGRAPH Asia 2023，[作者项目页和代码](https://skel.is.tue.mpg.de/) | 用运动序列优化生物力学骨架，再学习皮肤到关节与骨姿态的回归，并重新参数化SMPL。 | 关节中心与运动自由度需要正确，不是简单把55个SMPL-X关节当骨关节。SKEL使用自己的SMPL/生物力学骨架；不等于保留本Blender拓扑并从SMPL-X参数直接驱动的现成插件。 |
| Dakri 等，**On predicting 3D bone locations inside the human body (SKEL-J)**，MICCAI 2024，[作者项目页](https://3dbones.is.tue.mpg.de/)、[会议页](https://papers.miccai.org/miccai-2024/573-Paper0588.html) | 从三维MRI配对骨/皮数据出发，增加皮内骨位置的可调自由度并学习回归。 | 支持把“骨的内部位置”与“外皮形状”分开建模。它并未证明本Blender关节面接触或任意动作绝对包含；本轮有界关节站位属于工程假设，需重新验证。 |
| Keller 等，**HIT: Estimating Internal Human Implicit Tissues from the Body Surface**，CVPR 2024，[官方代码](https://github.com/MarilynKeller/HIT)、[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Keller_HIT_Estimating_Internal_Human_Implicit_Tissues_from_the_Body_Surface_CVPR_2024_paper.pdf) | 给定SMPL形状/姿态及空间点，预测脂肪、瘦组织、骨和空腔等类别，并学习体积变形。 | 可作组织厚度/分布先验，不能提供当前每根血管的对应、分叉拓扑或每个脏器的可靠分割。若marching cubes重建，拓扑也不再是源Blender拓扑。 |
| Cao & Mukai，**Skeleton-Aware Skin Weight Transfer for Helper Joint Rigs**，Eurographics 2024 Short Papers，[作者全文](https://mukai-lab.org/content/EG2024ShortCao.pdf) | 结合骨段投影、骨到皮射线和guide weight相似性建立对应，插值转移权重与辅助关节。 | 最贴近“已有Blender辅助骨联动应保留”。论文§3直接复制helper控制器，未自动适配控制器语义；§5承认复杂分叉关节和高细节几何仍困难。它是皮权重转移，不是血管包含算法；本轮材料附着只借鉴骨/空间关系同时约束的思想。 |
| Mukai & Taketomi，**Skeleton Subspace Skin Penetration Removal**，Eurographics 2026 Short Papers，[作者项目页](https://mukai-lab.org/publications/eg2026short-ssmik/)、[作者全文](https://mukai-lab.org/content/EG2026ShortMukai.pdf) | 将穿透顶点按变形梯度聚类，再做多目标IK改变骨姿态以去除碰撞。 | 可参考优化问题的降维方式；它会在运行时调整姿态，与“输入SMPL-X姿态不改、无需每帧重求解”的目标不同，不能作为本轮直接运行方案。 |

## 由论文和代码共同约束的方案

1. **保留运动来源。** 源235控制器及14-slot权重继续记录Blender的原始动作。将它压成55关节的普通LBS一般不能无损保存patella/tibia/ankle辅助响应。当前冻结资产的膝响应支撑130°、踝75°，超界会报错；不能裁角再称任意动作通过。
2. **骨与bind必须属于同一运动系统。** 关节端使用耦合约束，允许骨干内部轴向适应体型；必须实测骨端形状、截面粗细、关节间隙、穿出和方向。把405mm网格轴长直接设成373mm髋膝距离并不充分：两者不是同一解剖长度定义。
3. **软组织映射需要骨与皮两类边界。** 仅骨面附着能表达联动，不能保证处于皮内；仅皮肤harmonic变形也不能独立证明骨关节正确。应把骨作为内部边界，皮肤减去合理厚度作为外边界，检查无翻转、血管半径/边长和跨关节过渡。
4. **一次编译的含义。** 每一体型离线确定rest骨、bind、材料坐标和可复用响应；运行只计算SMPL-X参数所对应的控制器变换与固定坐标插值。固定坐标不代表所有姿态都一定包含，必须公布验证域和失败姿态。

## 本轮正在实际验证的范围

- 修复V12e按整个血管mesh owner漏掉前臂顶点的错误，以逐顶点源权重一次搬运；修正离线采样种子跨进程不确定的问题。
- 对照固定骨三角形的材料附着：保留三角形重心坐标和完整三维局部偏移，在源已pose组织上加目标骨相对源骨的位移，保留原有组织运动残差。它是**联动实验**，尚未完成骨—皮双边界体积求解。
- 在用户允许轴向调整后，对共同膝站位场做可行性测试；不会仅靠更低穿出分数接受断开关节的候选。
- 两个采集beta、两个明确合成的SMPL-X beta；T-pose、两采集动作、120°合成深屈、五个未裁剪AMASS动作；所有未支持姿态与失败指标都保留。
- 比较相机来自SMPL-X实际关节点，截面直接计算三角形和平面的交线。旧候选相机provenance错误已修，不把历史PNG重标成新结果。

没有任何上述论文或本轮小规模动作矩阵能证明“任何动作、任何人体体型、全内部组织绝不穿透”。本项目的可交付结果必须是带明确支持域、实测失败项和可复现资产的运行包；最终是否满足用户目标，以独立图审与绝对几何门为准。

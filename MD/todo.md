我沿着“数据生成 → RM4D编码 → 神经场训练/评估 → 区域查询 → 圆柱轨迹优化 → SMPL-X血管/骨骼重定向 → 调和坐标 → 体表投影 → DMP/QP-IK输出”的调用链检查了代码，也对照了四篇论文。
先给出最重要的结论：
问题	结论
内部血管投影是否真正用了你的调和场	主血管投影路径没有真正使用。它仍然是几何最近投影、候选点打分和动态规划连续性；调和投影只在少量共享分叉锚点上用到。
IRD神经算子是否正确	方法思想成立，而且局部区域查询确实达到毫秒级；但当前RM4D规范化存在一个必须修复的J1轴原点错误，训练/验证还有分组泄漏，因此目前不能把现有模型当作已验证正确的RM4D-IRD。
能否直接做人体上的联合优化	可以建立一个统一、几乎处处可微的问题；LBS公式本身可微。但是你当前的调和查询、投影、候选选择和LBS实现包含NumPy、KD-tree、argmin、DP等离散操作，不能直接端到端反传，需要换成可微解码器或固定预绑定关系。

1. 血管到体表的投影没有真正走调和坐标主路径
决定性证据在 [`project_vessel_centerlines_to_skin` (line 746)](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/projection.py:746)。
即使检测到了 harmonic volume，主分支在 [projection.py (line 779)](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/projection.py:779) 仍然调用：
_continuous_skin_projection(...)
harmonic/non-harmonic 两个分支的实际差别主要只是候选数，harmonic 分支没有把内部血管点的 (theta,h,d) 传给体表解码器。
_continuous_skin_projection 的实际行为是：
用 _piecewise_station 重新计算几何纵向坐标 h；
用 _theta_for_points 根据腿轴重新计算几何角度 theta；
在体表搜索候选点；
使用法线距离、前后方向、theta/h差异等代价；
用动态规划约束相邻血管点的投影连续性。
所以它是一个“几何先验 + 最近候选 + 序列连续性”的投影器，不是沿你的调和场流线从内部走到 d=0 的投影器。
代码里确实存在 [`_harmonic_skin_projection` (line 358)](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/projection.py:358)，它大致执行：
内部点 → query_atlas_coordinates → 保留(theta,h) → 设置d=0 → 体表refine
但它主要通过 _project_single_point 服务于共享分叉/连接锚点，并没有用于普通血管中心线的逐点投影。
此外：
DMP默认直接读取预先投影好的体表血管，而不是运行时重新联合投影，见 [pipeline.json (line 29)](/media/camp/EXT_DRIVE/Among_US/DMP_playground/configs/pipeline.json:29)。
投影CLI若发现已有NPZ，默认更倾向于 remap 旧投影，而不是重新从内部血管计算。
我比较了现有NPZ：827个血管点的 xi 完全相同，体表坐标最多变化约8.52 mm。这说明后续主要是在新皮肤上重新映射旧材料坐标，而不是重新求解投影方向。
打包脚本还存在一个来源NPZ和目标NPZ指向同一生产文件的循环依赖/来源记录问题，干净环境第一次生成可能失败。
你的调和坐标系实际做了什么
你的设计本质上是一个腿部材料坐标图：
\[
\xi=(\theta,h,d)
\]再加上局部材料框架相对旋转：
\[
\rho=F_{\mathrm{can}}^\top R_{\mathrm{can}}
\]这套抽象的意义是：
h：沿腿纵向的材料位置；
theta：绕腿的环向位置；
d：从皮肤到内部结构的径向/层次位置；
rho：探头或解剖局部框架相对于材料框架的姿态；
通过固定的SMPL-X附着和LBS，在T-pose、任意pose、笛卡尔世界空间之间保持对应关系。
它更准确的名称是“拓扑/材料感知坐标系”，并不是三个分量都严格为调和函数。当前实现中：
表面 h 确实通过cotangent Laplace–Beltrami Dirichlet问题求解；
基础版本的 theta 主要是绕中轴的解析角度；
基础 d 是内外表面距离比；
production layered版本只对 d 求了图拉普拉斯场；
但运行时 [`interpolate_volume_field` (line 1092)](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/harmonic.py:1092) 又根据距离重新计算 d，没有使用烘焙的 vol_d。
因此，当前production运行时甚至没有真正使用烘焙出来的调和 d。
还有一个数学问题：体积 theta 不能直接把包裹的弧度值当普通标量解拉普拉斯方程，否则在 \(0/2\pi\) 接缝处会出现错误插值。应改为同时传播：
\[
c_\theta=\cos\theta,\qquad s_\theta=\sin\theta
\]查询后再用 atan2 恢复角度，或者显式切开接缝。经典调和坐标本身适合做平滑体积坐标，但其“不变性”仍依赖固定拓扑、固定附着关系且变形不发生翻折。Joshi等人的调和坐标工作可作为这一部分的理论参照。
2. IRD方向是对的，但当前模型需要先修一个关键错误
你的模型实际上是什么
[`SignedReachabilityField`](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/ird_playground/neural/signed_field.py) 是一个输入RM4D规范坐标、输出标量可达分数的神经隐式场。严格来说它不是“神经算子”：
它没有学习“函数到函数”的映射；
它学习的是有限维坐标到标量场 \(f(c)\) 的映射；
每个构型的梯度不是单独存储的，而是通过自动微分实时计算：
\[
\nabla_c f(c)
\]更准确的名称是：
Differentiable neural implicit signed reachability field with robust task-set aggregation

区域算子 [region/operator.py (line 19)](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/ird_playground/region/operator.py:19) 在 nominal pose 周围采样64个固定Sobol场景，当前范围大致为：
切向位置 ±4 mm；
横向位置 ±3 mm；
法向位置 ±2 mm；
姿态锥约3°；
使用soft-min形成鲁棒区域得分。
这个做法是合理的。它把Vahrenkamp对轨迹多个任务姿态取最小可达性的思想推广成可微的soft-min；Vahrenkamp的逆可达图和轨迹交集方法可见用户提供的 [Vahrenkamp 2013 PDF](/home/camp/Desktop/New Folder/Vahrenkamp et al. - 2013 - Robot placement based on reachability inversion.pdf)。
Murooka 2025也验证了“学习一个可微可达函数，将其作为SQP/QP不等式约束”的总体路线。他们同时指出普通分类器在远离边界时可能出现梯度接近零的问题，并认为高维signed-distance表示值得进一步研究。你的边界有符号监督正好是在解决这个问题。
必须先修复的J1轴原点错误
RM4D规范化在 [canonical.py (line 31)](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/ird_playground/ird/canonical.py:31) 使用：
\[
[p_z,\;u_z,\;\sqrt{p_x^2+p_y^2},\;p_xu_x+p_yu_y,\;p_xu_y-p_yu_x]
\]它假设绕Z轴旋转时，旋转轴经过坐标原点。
但你的真实J1轴原点不是世界/rail坐标原点，而是：
\[
o_{J1}=[0,\,-0.4,\,0.2405]\ {\rm m}
\]当前训练特征直接使用原始FK位置，没有先减去这个轴原点。
我固定 \(q_2\ldots q_7\)，只扫描 \(q_1\) 后测得：
当前后三个规范特征的变化范围约为0.671、0.757、0.758 m；
先做 \(p_{\rm rel}=p-o_{J1}\) 后，变化降低到约 \(10^{-7}\) m。
这意味着当前特征并没有真正消掉J1旋转自由度。现有单元测试只测试“人为绕原点旋转”，所以没有发现这个运动学问题。
正确修复方式是：
明确定义物理J1轴坐标系 \(T_{\rm axis}\)；
对位置和工具轴都先变换到该轴坐标系：
\[
p_a=R_a^\top(p-o_a),\qquad u_a=R_a^\top u
\]再计算RM4D特征；
重新生成训练数据、训练并评估；
增加“固定其他关节、扫描真实q1轨道”的不变性测试。
原始RM4D也明确依赖基座关节完整轴向旋转和末端腕轴完整roll的假设。Rudorfer RM4D使用这一结构把SE(3)压缩到四维。你的J1范围约为±177.96°，不是严格360°；J7接近完整旋转。因此即使修正原点，J1关节限位附近仍要做代表元可实现性验证。
当前“signed field”还不是物理距离场
边界标签在 [gpu_boundary_stencil.py (line 135)](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/ird_playground/ird/gpu_boundary_stencil.py:135) 中将：
平移距离除以6 mm；
旋转距离除以3°；
混成一个无量纲值。production训练的Eikonal权重又是0。
所以输出应解释为“有符号可达裕度代理”，不能解释为：
离不可达边界还有多少毫米；
严格的SDF；
安全证书。
可以借鉴机器人神经SDF工作中对距离标定和梯度约束的处理，例如 RDF，至少增加：
单位明确的加权SE(3)度量；
Eikonal/局部Lipschitz正则；
holdout边界上的值、法向和距离标定；
保守阈值或conformal calibration。
验证集存在来源泄漏
训练划分正确地按 boundary_id 对边界样本分组，但所有全局样本的 boundary_id=-1，随后被逐行随机划分，见 [train_signed.py (line 117)](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/ird_playground/neural/train_signed.py:117)。
我的复核结果是：
validation中74,882个全局样本；
38,500个不同的 source_pose_id；
这些source ID有100%也出现在training中。
也就是说，同一个源构型附近的扰动样本会同时出现在训练和验证中。需要按 source_pose_id 分组，最好再留一套按工作空间块、关节轨道或不同障碍物配置划分的外部测试集。
它能否处理“不完整task空间”
概念上能，但当前API只实现了一种特定形式：
一个完整nominal SE(3)姿态 + 固定位置盒 + 固定姿态锥 + 对所有场景soft-min。

它还不是任意的“位置小区域 + 大致角度范围 + 部分自由度未指定”查询接口。
这里必须区分三种语义：
角度是优化器可以选择的自由变量：对角度范围应取 max/softmax 或直接把角度作为决策变量；
角度是必须对所有误差都可达的不确定性：取 min/softmin；
角度服从概率分布：取期望、分位数或CVaR。
可以写成：
\[
C(\mathcal T)=
\max_{\eta\in\mathcal D_{\rm free}}
\operatorname{CVaR/softmin}_{\delta\in\mathcal U}
f\!\left(c(T(\eta,\delta))\right)
\]当前RegionA把所有采样都当成“必须同时满足”的鲁棒不确定性。如果把可自由选择的扫描角也放入soft-min，会不必要地过度保守。
现有结果能支持什么结论
现有报告中比较可靠的结果包括：
独立水平探头切片：29,791点，准确率约99.17%，FPR约0.81%，FNR约1.43%；
当前冻结发布基线 balanced accuracy 为90.38%；
位置/旋转梯度方向一致性为99.80%/99.58%；
±1 mm严格边界两侧正确跨越率只有53.6%；该旧记录不应与当前 frozen baseline 混用。
已成功跨越的样本中，位置边界误差p95约0.928 mm、旋转约0.601°。
最后一个p95是以“已经严格跨越”为条件的，不能表述成全部工作空间都达到亚毫米边界精度。
本机CPU单线程、模型预热后的实测为：
单姿态模型：约0.10 ms；
64场景RegionA前向：约0.76 ms；
单姿态RegionA前向+梯度：约2.16 ms；
81点整条轨迹前向：约45 ms；
81点整条轨迹前向+梯度：约100 ms。
所以可以声称：
局部区域鲁棒可达查询及优化方向达到毫秒级。

目前还不能声称：
完整轨迹的多轮全局重优化已经在控制周期内实时完成。

需要在目标GPU/控制器上测完整优化迭代数、FK、碰撞、人体变形和QP-IK的端到端延迟。
另外，[README.md](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/README.md)仍描述旧的6D多头分类模型，与当前5D标量场不一致，论文和文档前应同步修正。
3. 人体扫描应改成“联合投影—可达—遮挡—轨迹”问题
你的新想法比“先生成体表GT，再围绕GT局部修正”更合理。最近投影只应该是初始化/先验，不应先冻结成GT。
建议在归一化血管弧长 \(s\in[0,1]\) 上，用少量B-spline控制点优化：
\[
z(s)=
[\cos\theta(s),\sin\theta(s),h(s),\rho(s),r_{\rm rail}(s)]
\]必要时再加入 \(q(s)\) 或关节样条控制点。
其中：
\(v^{can}(s)\)：内部静脉中心线；
\(x^{can}(s)=D_{\rm skin}(\theta,h,d=0)\)：候选皮肤接触点；
\(\rho(s)\)：探头roll/扫描模式；
\(r_{\rm rail}(s)\)：导轨位置；
通过SMPL-X/LBS得到世界空间 \(v^w(s),x^w(s)\)。
探头声束轴可以由：
\[
b(s)=\frac{v^w(s)-x^w(s)}
{\|v^w(s)-x^w(s)\|}
\]初始化，再允许在小角锥内优化。这样血管在视野中心和投影方向不是后处理，而是直接进入探头姿态参数化。
一个适合你的联合目标是：
\[
\begin{aligned}
\min_z\quad
&\lambda_m L_{\rm material}
+\lambda_c L_{\rm center/FOV}
+\lambda_d L_{\rm depth}\\
&+\lambda_b L_{\rm bone}
+\lambda_r L_{\rm IRD}
+\lambda_J L_{\rm task\ compatibility}\\
&+\lambda_s L_{\rm smooth}
+\lambda_q L_{\rm IK/joint}
\end{aligned}
\]各项具体含义如下。
调和/最近投影先验
保持候选点靠近内部血管对应的 \((\theta_v,h_v)\)，但允许可达性和骨骼遮挡把它推向更好的邻近位置：
\[
L_{\rm material}
=
\|\,[\cos\theta,\sin\theta,h]
-[\cos\theta_v,\sin\theta_v,h_v]\,\|^2
\]再加声束长度或皮肤距离项，而不是把最近点设为硬GT。
血管中心和扫描截面
对血管点在探头图像坐标中的横向、升降向偏差直接惩罚，并限制焦深。纵切和横切应使用不同的显式约束：纵切：血管切向应位于成像平面内；
横切：成像平面法向/平面轴与血管切向满足所需正交关系。

不要只通过一个通用姿态误差间接表达，否则容易得到“可达但成像模式错误”的解。
骨骼声学遮挡
当前骨骼OBJ/点云在导出链中存在，但主要用于可视化；DMP里的visibility是探头切片/FOV可见性，不是从皮肤到血管的骨骼声影判断。相关骨骼导出可见 [run_export_vessel_segments.py (line 1940)](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/cli/run_export_vessel_segments.py:1940)。
建议把股骨等转换成封闭mesh SDF \(\phi_b(x)\)，沿探头到血管的声束线段或声束锥采样：

\[
y_{ij}=x_i+\lambda_j(v_i-x_i)
\]\[
L_{\rm bone}
=
\sum_{i,j}\operatorname{softplus}
(\delta_b-\phi_b(y_{ij}))^2
\]仅用稀疏点云最近距离不能可靠判断“骨头是否位于声束前方”，也不能区分内外。机器人超声中已经有显式把声影纳入轨迹规划的研究，可参考Acoustic Shadowing Aware Robotic Ultrasound。
区域鲁棒IRD
对每个候选探头姿态，将人体注册误差、血管误差、皮肤法向误差和探头姿态锥作为场景：
\[
C_i=-\tau\log\frac1K
\sum_k\exp[-f(c(T_i(\delta_k)))/\tau]
\]约束：
\[
C_i\ge m_{\rm safe}
\]但它只能作为平滑可达先验，最终仍需精确IK和碰撞检查。
Chiu方向任务兼容性
单一IRD只回答“综合上是否可达”，不能表达超声扫描最关心的方向性：沿血管切向要有良好的速度能力；
沿皮肤法向要有接触力/顺应控制能力；
roll和侧向应保持图像中心；
应避免关节限位和奇异位形。

这正是 [Chiu 1988](/home/camp/Desktop/New Folder/Chiu - 1988 - Task Compatibility of Manipulator Postures.pdf) 的任务兼容性思想。可以在 \(q_{\rm ref}(s)\) 上加入加权Jacobian方向指标，而不是只用通用manipulability标量。速度椭球与力椭球是对偶的，因此“切向运动好”和“法向施力好”需要分别评价。
平滑和连续性
在材料坐标上对 \((\cos\theta,\sin\theta,h)\) 做一阶、二阶平滑；对探头方向使用SO(3)测地误差；对rail和关节加入速度、加速度、jerk以及限位裕度。这样可以避免theta接缝，也比直接对世界空间点做欧氏平滑更稳定。
圆柱demo当前实际做了什么
圆柱demo在 [cylinder_region_ird_demo.py (line 57)](/media/camp/EXT_DRIVE/RealUS_playground/ird_playground/experiments/cylinder_region_ird_demo.py:57) 中通过探头坐标系构造，已经把“切面/血管中心关系”硬编码进参数化。
实际损失并不是三个，而是六类：
IRD；
theta跟踪；
一阶连续性；
曲率；
rail居中；
基座侧向正则。
它优化的是9个控制点、81个采样点的任务轨迹。随后才独立执行多初值IK，并以前一个关节解为种子做贪心连续化。因此：
q_ref不是与任务轨迹一起联合优化的；
它不是严格的全局关节轨迹最优；
GT约束主要是theta回到零，并非完整SE(3) GT约束。
不过结果是不错的工程验证：
初始精确可达45/81，优化后81/81；
21×64鲁棒场景全部通过精确IK/碰撞检查；
最大表面角偏移约4.06°；
rail最大约0.120 m；
\(q_{\rm ref}\) 最小关节裕度比例约0.0515；
FK位置误差最大约0.199 mm；
未发现碰撞。
这说明“低维轨迹参数 + 区域IRD + 后验IK连续化”路线可用，但人体版本需要把投影、骨遮挡和扫描方向一起放进优化变量。
LBS会不会破坏可微性
数学上的LBS不会。固定权重时：
\[
x_i^w=
\sum_k w_{ik}G_k(\beta,\Theta)
\begin{bmatrix}x_i^{can}\\1\end{bmatrix}
\]对canonical点、姿态参数和关节变换都是可微的。SMPL本身也是围绕可微蒙皮构建的。
但你现在的 [`lbs_bridge.py`](/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/lbs_bridge.py) 和投影链不能直接自动微分，因为包含：
NumPy数组转换；
SciPy插值；
KD-tree最近邻；
argmin和硬候选选择；
三角形/四面体硬切换；
barycentric clipping；
动态规划路径选择；
SVD极分解和行列式分支。
建议这样改：
在T-pose中一次性预绑定血管、骨骼和体表：保存tet/face ID、barycentric权重、LBS权重。
用Torch/JAX实现固定附着关系下的LBS。
体表解码采用局部光滑patch，或在当前active face内用barycentric坐标；跨face时使用top-k soft assignment或trust-region更新active set。
探头姿态尽量由体表偏导、血管切向和声束方向构造，避免每次对混合LBS矩阵做SVD。
骨骼使用可微SDF。
在外层使用continuation/trust region；每轮后仍运行精确IK、机器人碰撞和声束遮挡验证。
所以你可以把它写成一个统一的可微目标，但“全局可微”不等于“全局凸”或“一次迭代得到全局最优”。这是高度非凸问题，合理做法是一个统一目标、多个初始化、逐步增加损失权重，并做最终离散验证。
建议的实施优先级
修正RM4D到真实J1轴坐标系，重新生成数据和训练；这是最高优先级。
按 source_pose_id 重做数据划分，并建立真正独立的测试集。
把普通血管点的投影改成真正的 (theta,h,d_v)\rightarrow(theta,h,0)，修复theta接缝和runtime忽略harmonic d的问题。
将最近投影降级为初始化/软先验，加入骨SDF声束遮挡。
实现Torch材料坐标解码和LBS。
将RegionA推广为明确区分“自由扫描变量”和“不确定变量”的partial-task算子。
加入Chiu方向任务兼容性和联合/内层可微IK。
最终给QP-IK输出除 s,T,rail,q_ref 外，还应包含 qdot/qddot或样条、IRD分位裕度、骨遮挡裕度、方向兼容性、精确IK/碰撞有效标志和不确定性参数。
测试方面，IRD现有8项、腿部坐标7项、anatomy retarget 3项均通过；DMP为49项通过、1项失败，失败是预烘焙patch顶点数1245超过测试中旧的1200阈值，更像数据/阈值漂移，不是上述数学链路错误。本次审阅约21分14秒，未修改任何代码。

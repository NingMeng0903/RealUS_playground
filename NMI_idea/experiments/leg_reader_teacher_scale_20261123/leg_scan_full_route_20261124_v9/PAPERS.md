# v9 全程扫查的论文依据

## Reader 的候选排序、历史关联和拒答

- **超声状态的几何信号。** Bi 等的 VesNet-RL 用 U-Net 二值 vessel mask、近期帧、动作和面积变化构造观测，由 LSTM A2C 学习寻面。训练分数是模拟投影中心线距离分数和当前切片面积分数的加权和（0.2、0.8），奖励主要来自分数增量，另有失去血管及到达目标的离散奖励。推理用最小外接矩形的填充、尺寸稳定和视野跨度条件终止。任务覆盖单一血管的纵切面搜索，验证包括凝胶 phantom 和志愿者颈动脉体积回放。对 v9，沿用局部形状、尺度、相对位姿与动作历史作为观测，单独测量全路线目标、错认和停止行为。 [VesNet-RL，§III.A–C、§IV](https://arxiv.org/abs/2205.06676)
- **原始材料与方向特征。** 候选内保留可观测的原始材料统计、形状及尺度，另加相对扫描几何与低阶角向谐波。圆周谐波相位随旋转变化，模长保持不变，因此可在同一局部坐标系下同时描述形状和方向。轴向特征若无头尾，可用 `sin(2θ), cos(2θ)` 表示。缺失或近圆候选保留方向有效性掩码。以同一候选池比较几何、原始材料、谐波及组合对候选召回和首选正确率的增益。Zahn–Roskies 的原始方法是闭合曲线切向角函数的 Fourier 描述，这里借用其表示思想。 [Fourier Descriptors for Plane Closed Curves](https://www.cs.jhu.edu/~misha/Papers/Zahn72.pdf)
- **跨窗证据。** Chen 等在同一超声 sweep 中用训练期跟踪到的探头平移距离采正例，以帧嵌入点积排序，训练以双向交叉熵、位置距离辅助损失及可学习的未匹配项进行。实验主要检验帧检索与定位，而且省略探头旋转。Sermanet 等用同步跨视角 triplet，将外观相似但功能相异的近时帧作为难负例。v9 可沿用相同实体跨位置和角度的正例，加入同站相似异实体、遮挡后候选消失、重复视角和无匹配的难例。历史 edge 输入使用两端候选表示及可观测的相对几何，实际检索和训练采样共用几何门。 [Intra-Sweep Ultrasound Retrieval，§2–3](https://arxiv.org/abs/2412.07741) · [Time-Contrastive Networks](https://research.google/pubs/time-contrastive-networks-self-supervised-learning-from-video/)
- **匹配与最终首选。** SuperGlue 将经过帧内和帧间注意力的描述子点积作为候选边分数，追加未匹配行列，再用 Sinkhorn 求部分一对一分配。其分配分数包含候选集合的竞争关系。DA-BSP 用多假设后验评估未来动作能否消除关联歧义。v9 可给历史边加可学习的未匹配项，汇总不同视角支持，并让首选分数对比未匹配项后执行 `选第一名 / 拒答`。检测重复视角是否压过独立证据，必要时加去重，并评估首选与次选差值能否改善决策。跟踪连通或多跳可用于聚合证据，最终首选在实际推理链上校准。 [SuperGlue，§3](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf) · [DA-BSP，§3–4](https://doi.org/10.1177/0278364918759606)
- **评分与校准。** 温度缩放拟合保留集上的 log-loss，正温度保留同一候选集内的排序。选择性分类在有标签的校准集上定阈值，以接受后的错误率与覆盖率衡量结果。分别测候选池召回、首选排序正确率、最终首选的错认率与覆盖率，并加上跨窗/相似异实体分层。最终 gate 以独立布局上的实际首选结果校准，可用首选正确 logit 和竞争差值。同一布局的帧、边和噪声带相关，应按布局汇总及重采样。选择性分类论文的有限样本保证依赖其 i.i.d. 抽样假设。 [Guo 等，§4](https://proceedings.mlr.press/v70/guo17a.html) · [Geifman 与 El-Yaniv，§3](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)

## 全路线策略与未来噪声 teacher

- **循环 actor 与特权 critic。** Pinto 等用 DDPG，actor 接 RGB-D，critic 接模拟器完整状态。Baisero 与 Amato 在部分可观测设定下推导了 `V(h_t, x_t)`，其中 `h_t` 为观测和动作历史，`x_t` 为潜在状态。v9 actor 用递归记忆、当前 reader 输出、路线进度、动作反馈和公开目标条件。训练 critic 同时接 actor 历史表示和模拟器的真实状态或进度，按完整路线回报训练。用完整未见布局回放评估，并检查 actor 的输入和导出依赖。 [Pinto 等，§III](https://m.roboticsproceedings.org/rss14/p08.pdf) · [Baisero 与 Amato，§5](https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p44.pdf)
- **全程目标与执行层。** Active Neural SLAM 从 RGB 与传感位姿更新地图和姿态，RL global policy 根据覆盖面积增量打分并选择长程目标，解析规划器转成短程目标，监督训练的 GRU local policy 执行。v9 可复用扫查动作与运动约束，让全路线 actor 在同一回报下处理前进、回看、补扫和终止。有效覆盖按新确认的路线事件累计，分别记录 strict 完成、wrong、预算和停止。只靠局部首选质量或重复观测奖励，会把好读的局部动作排在有利全路线完成的动作之前。 [Active Neural SLAM，§3、附录 B](https://arxiv.org/abs/2004.05155)
- **teacher 值由哪条 suffix 产生。** Deeply AggreVaTeD 在当前策略分布的状态上最小化专家 `Q*(s,a)`，查询后的回报由专家续跑定义。LOLS 用当前策略 roll-in，改变某一步动作并以到终局的损失相对最优分支打分，默认按整条 suffix 抽取参考或当前策略。v9 要估计的是相对于同一学生的单步偏离效应。对每个真实到访状态复制环境、已见观测、reader 账本和 actor 隐状态，固定一份学生权重。每条未来噪声带在基准首步及每个候选分支共用外生随机源，首步之后各分支均运行该学生直至同一结束条件。记录分支的实际动作与终局，检验分支中出现的恢复或错认是否来自学生自身的续跑。 [Deeply AggreVaTeD，§3–4](https://proceedings.mlr.press/v70/sun17d.html) · [LOLS，§2.2、§3.1](https://proceedings.mlr.press/v37/changb15.html)
- **teacher 比较和采样。** 标签以同噪声带下的终局差值计算，分别估计 strict、wrong 和成本差，并保存样本量、不确定性和学生版本。可观测状态与动作掩码先确定候选，然后才做未来分支评分。各分支根据自己接收到的图像与历史重算后续动作。同一个布局可分阶段增加复现次数用于筛选候选，跨布局判断则以完整布局为抽样单位。A2D 说明拥有真状态的专家会在观测歧义处产生学生无法稳定模仿的动作，支持监督以学生实际可见信息和续跑结果为条件。 [Robust Asymmetric Learning in POMDPs，§3–5](https://proceedings.mlr.press/v139/warrington21a.html)
- **离线与在线交替。** 可先从全路线离线序列训练 reader、动作先验与 critic，再迭代「冻结学生采全路线 → 标注此版本到访状态的配对分支 → 学生更新 → 用更新后的版本重新采集及标注」。保留离线序列作为辅助监督或回放，分别记录真实在线扫查、teacher 分支和训练更新。RLPD 在 off-policy SAC 中将在线与离线样本等量混采，说明离线数据可贯穿在线阶段。使用 PPO 时，策略梯度用当前采样批次与对应行为 log-prob，离线及 teacher 标签进独立的监督项。在循环模型更新时重放所需观测前缀，使当前参数重建隐状态。 [RLPD，§3–4](https://proceedings.mlr.press/v202/ball23a/ball23a.pdf) · [PPO，§3、§5](https://arxiv.org/abs/1707.06347) · [R2D2，§3](https://openreview.net/pdf/387fb2fcee8f74c53cf707a9856f40c458f33933.pdf)

## 原文 PDF

本机已有并复用的原文位于 `/home/camp/Desktop/belief/`，包括 `2022_VesNet_RL.pdf`、`2025_IntraSweep_Ultrasound_Embedding.pdf`、`2021_A2D.pdf` 和 `A unified framework for data association\naware robust belief space planning and\nperception.pdf`。最后一个文件名含实际换行符。

补齐的原文保存在本次运行的 [run/papers](run/papers/) 下。每份文件通过 `pdfinfo` 和文本提取检查。

| 原文 | 本机文件 | 页数 |
| --- | --- | ---: |
| Chang 等，LOLS (2015) | [PDF](run/papers/2015_LOLS_Chang.pdf) | 9 |
| Sun 等，Deeply AggreVaTeD (2017) | [PDF](run/papers/2017_Deeply_AggreVaTeD_Sun.pdf) | 10 |
| Geifman 与 El-Yaniv，Selective Classification (2017) | [PDF](run/papers/2017_Selective_Classification_Geifman_ElYaniv.pdf) | 10 |
| Pinto 等，Asymmetric Actor Critic (2018) | [PDF](run/papers/2018_Asymmetric_Actor_Critic_Pinto.pdf) | 10 |
| Chaplot 等，Active Neural SLAM (2020) | [PDF](run/papers/2020_Active_Neural_SLAM_Chaplot.pdf) | 18 |
| Sarlin 等，SuperGlue (2020) | [PDF](run/papers/2020_SuperGlue_Sarlin.pdf) | 10 |
| Baisero 与 Amato，Unbiased Asymmetric RL (2022) | [PDF](run/papers/2022_Unbiased_Asymmetric_AC_Baisero_Amato.pdf) | 9 |
| Ball 等，RLPD (2023) | [PDF](run/papers/2023_RLPD_Ball.pdf) | 18 |

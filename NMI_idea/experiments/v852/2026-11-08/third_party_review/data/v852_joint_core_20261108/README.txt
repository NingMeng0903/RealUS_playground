关节执行版 A/B/C 探索结果（2026-11-08）

所有实验为当前8DOF URDF的Genesis位置PD实际推进，图像特征由实测q的FK生成切片观察；每tick没有位置teleport。
这是自由空间6D DLS/有限SRS转场程序，不是现有接触QP/力控的实机验证。床/人体环境碰撞、真实B-mode质量和接触均未验证。
机器人flange-to-TCP来自当前合同；image-to-TCP仍是示例Rx(pi)、零原点偏移。所有摆放是公开几何的虚拟registration。

一、原全腿范围的边界
public_20_placement_screen.json保存20个预声明anchor/wrist摆放：每个目标独立检查rail=(.15,.30,.45,.60,.75)、SRS8分支、10度swivel网格，dIRD只排序，FK/限位/自碰撞独立核查。
未找到同时覆盖原near(.10-.25)/fresh(.14-.21)与远端A(.62-.76)的摆放。
rail_endpoint_supplement.json追加同20摆放的rail=0/.8端点，仍无全目标支持。
结论仅限该虚拟registration、示例探头坐标变换和有限候选网格，绝不是实际硬件全局不可达证明。
full_leg_boundary_v2实际执行6次续程尝试，0完成，安全停止/拒绝全部保留。历史near在该范围不支持，其0值是无观测占位，因此这批不是有效信息时机机制检验；零预测误差也是零完成的退化结果。

二、局部可达版本 v2
固定同一已检查摆放，将near/probe/f全部明确改为(.56,.60)，目标仍为A(.62,.76)。它检验局部证据续程，不冒称原跨近端U/远端V关联或全腿返回。
从2个独立0度配置采集joint-only局部程序，再以2个新60度配置测试react/probe两树×early/late/hide，共12条完整续程。
所有预测先写盘再执行；same packet hash、fresh capture-time、配置身份均核验。
严格完成6/12；每种条件完成2/4。early平均20.790s，late/hide20.245s；本版本没有显示保留旧信息减少成本。
点预测使用公共Cartesian转场近似，未条件化实际8DOF入口，4个预算上的平均CDF误差0.5625；不能据此把核心idea判假，也不能宣称真实窗口预测成功。
derived_joint_entry_exit_records.json从逐tick轨迹导出实际q/qdot入口出口，明确未用于v2预测，供后续训练使用。

三、入口核版本 v3
entry_conditioned_v3固定pose60、同一形状与公共摆放，去除v2的跨姿态迁移；2个新训练对象、3个公开关节入口、A-/A+/slowProbe，实测18个独立单程序，17个通过。
每个局部核同时保存实际q/qdot入口出口、完整转场+服务时间、public_success、independent_success、safe_exit；完整测试链终点参与拟合次数为0。
模型按q/qdot入口找整条记录，并跨程序保持同一对象，推进sampled exit_q/qdot；不再添加Cartesian转场时间。
入口支持阈值预声明rail6cm、arm .35rad，速度rail .03m/s、arm .12rad/s；超出者贡献[0,1]未知质量。这不是有限样本置信保证。
用含fixed_minus/fixed_plus/react/probe/wait的10棵树，在40s期限按完成下界、支持质量、时长预选；所有树、预测与预选先冻结。
2个新测试对象×early/late/hide共6条实际预选续程全部严格完成，所有条件均在测试前选fixed_plus。
实际两个配置耗时19.88/21.56s；三个信息条件每个对象的q_cmd轨迹逐项相同，表明模型选择忽略旧证据，B/C收益为0。
四预算CDF平均绝对误差0.125，但T=20时预测完成率1、实测0.5，窗口边缘差50个百分点；T>=40时这批均完成。
因v3同时匹配了姿态并加入入口核，不能把相对v2的改善单独归因于q条件化。
模型在入口覆盖内仍低估了耗时，2训练配置+单次局部记录不能支持窄的真实完成保证；应增加独立校准与执行噪声样本。

四、必须保留的解释
6或12条运行来自2个独立隐藏配置，重复树/信息条件不增加独立对象数。
旧证据准备是单独实测的局部程序；续程从共同实际关节入口重置开始。没有声称已连续执行整个历史准备到新请求的全程路径，准备成本另存。
历史包是从实际采集特征形成的同一derived packet，canonical历史时间用于gate；原始准备帧capture时间保存在preparation记录，旧包从不计fresh。
v1_development_findings.json与旧目录保留最初success/status不一致及异常；这些数据不计v2/v3正式结果。
v2/v3是开发验证。未展示强基线数据效率优势，未展示正的B/C收益，未证明NMI主张。
但已经实际检验出：当前全腿程序/摆放覆盖缺口；Cartesian转场近似对joint窗口不可靠；入口核能组合出可执行续程；无信息收益时预选策略可以合法忽略旧证据；截止边缘的时间误差仍需校准。

五、文件
entry_conditioned_v3/{frozen_protocol,source_freeze,local_training,prospective_predictions,prediction_freeze,outcomes,summary,execution_contract_checks,window_edge_metrics}.json
local_continuation_v2/保存对应v2文件与逐tickNPZ。
full_leg_boundary_v2/保存全腿失败。
joint_completion_window_v2_v3.png为描述性CDF图，图中未计算2配置不足以支持的精确置信带。

源码入口：work/leg_program_bench/run_joint_core.py、run_joint_core_entry.py。

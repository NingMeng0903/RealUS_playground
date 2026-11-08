V8.5.2 第三方独立分析材料（2026-11-08）

先读 FINDINGS_AND_NEXT_ITERATION_zh.txt，再看 KEY_METRICS.json。
结论：几何模拟基础档可预测、部分预算有证据时机收益；尚无稳定数据效率优势。Genesis入口核版20秒预测100%、实测50%，不能用40秒全部完成掩盖窗口误差。

目录
manuscript/：本轮依据的V8.5.2原稿。
reports/combined/：两次主运行的完整数值、中文报告、主要图；main/replicate保留单次结果。
data/：四个前瞻几何运行；有限数学核验；Genesis关节路径与joint v2/v3，包括阴性/失败。大JSON/JSONL为gzip，原始冻结哈希可在解压后复验。
source/：当前可运行分析与实验Python模块及测试。各run/source_snapshot才是准确生成该run的版本；后续修复可能不同。
MANIFEST.json：上传文件SHA256、原始解压内容SHA256、大小与排除项。
prepare_analysis.py：校验文件及解压分析数据（Python标准库即可）。

快速分析
python prepare_analysis.py --out /tmp/v852_analysis
PYTHONPATH=source python -m leg_program_bench.report_core_suite --combine-input /tmp/v852_analysis/data/v852_core_suite_20261108 /tmp/v852_analysis/data/v852_core_suite_20261108_replicate --output /tmp/v852_analysis/recomputed_report
分析图表依赖numpy/matplotlib，完整环境见source/environment.json。重新运行几何/机器人实验另需用户自有SMPL-X/atlas与RealUS机器人依赖；这些资产不随包发布。

必须保留的解释边界
1. Cartesian模拟秒、Genesis自由空间模拟秒与实机时间不同；没有真实B-mode、接触或硬件结果。
2. 主两个run共96个测试配置、3840条反事实程序链；两种matched-budget补充共40个新测试配置、1600条链，总5440条，不是5440名独立对象。
3. matched-budget复用既有局部训练数据，不能重复计算为新增采样交互。balanced版本固定60度，UV×请求完整平衡；不要混合不同设计的学习曲线。
4. lower/upper是经验bootstrap/coupling场景包络，不是95%置信集。数学有限模型结果与真实过程保证分开。
5. 主m±是几何参考参数，不是实测dIRD关节构型。joint版本另有实际q/qdot与时钟。
6. 旧主run名为joint_nominal的方法实际上是robust alias，见method_label_audit.json；不作为Bayes基线。后续源码dispatch已修，matched-budget点模型比较使用修正版本。
7. 早期一次audit真值影响重试的错误已修；这些无效完整链不随包上传。有效run换了测试种子，并有private-label非干预回归。单程序数据仅标签字段重整，有原始hash来源。
8. 全腿joint v2历史本身unsupported，只是装置/程序库压力测试；其零完成/零预测误差不是有效的信息时机证据。
9. v3只有2个独立测试对象，6条续程；三信息条件选中同一fixed_plus，q_cmd完全相同，B/C为null。v2→v3同时改变入口与pose覆盖，不是单因素消融。
10. 不支持“人体/胸部多器官已泛化”“所有控制器都不能恢复”“已有严格真实完成保证”。

建议第三方重点检查
A. 聚合完成率曲线的校准，尤其窗口边缘，不用单个Bernoulli样本|p-y|直接诊断失配。
B. 同已采证据的早/晚差异是否被事前预测；权限、缓存、fresh与恢复是否公平。
C. 同一最低完成要求下的成本，同时报告实际完成率overshoot；匹配数据预算与优化目标后是否仍有优势。
D. 稀疏局部数据、持续噪声、q/qdot入口与出口覆盖是否足以支持组合；bootstrap是否过度自信。
E. 固定有限树库的限制；它不是任意合法控制器能力上界。

所有材料保留负结果。此包供独立审阅，不替代独立复现实验。

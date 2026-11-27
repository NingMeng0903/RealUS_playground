from pathlib import Path
import csv,json
import numpy as np
from run_motion_validation import provenance,stats
P=Path(__file__).resolve().parent
m=json.loads((P/'metrics.json').read_text())
# Derive the missing-only comparison from saved samples, truncating at either
# world's first detection rather than treating later unequal histories as equal.
for method,x in m['servo']['missing_observation_counterfactual'].items():
    plus=np.load(P/'controlled_servo'/f'{method}_not_detected_depth_plus120mm_lateral_plus10mm_gain1.npz')
    minus=np.load(P/'controlled_servo'/f'{method}_not_detected_depth_plus120mm_lateral_minus10mm_gain1.npz')
    detected=np.flatnonzero((plus['visible']>0)|(minus['visible']>0))
    prefix=int(detected[0]) if len(detected) else len(plus['visible'])
    x['identical_missing_prefix_steps']=prefix
    x['identical_missing_prefix_gain1_path_difference_m']=stats(np.linalg.norm(plus['p_cmd'][:prefix]-minus['p_cmd'][:prefix],axis=1))
    x['gain1_hidden_world_path_difference_m']=stats(np.linalg.norm(plus['p_cmd']-minus['p_cmd'],axis=1))
    x.pop('same_public_missing_history_gain1_hidden_world_path_difference_m',None)
    x['inference']='Nominal motion while missing is allowed. Hidden-world path comparisons concern only the common missing prefix before either detects; stock additionally consumes simulator geometric image_u even when invisible. This violates a missing-only sensor contract, not the internal simulator observation contract.'
m['provenance']=provenance()
(P/'metrics.json').write_text(json.dumps(m,indent=2))
s=m['servo'];g=m['production_geometry'];d=json.loads((P/'depth_contract.json').read_text())
rows=s['rows'];r={x['id']:x for x in rows}
lines=['E1 实际运动与生产坐标验证报告','日期：2026-11-10','',
'结论：真实本地 DMP/KMP primitive 与完整因果重建伺服均已执行；在同一合成血管目标上，未启用 Module C 的局部横向反馈明显降低图像横向误差。共同 SE(3) 变换通过。规范注册 3D 在相同对应/附件信息下保持精确，调和表面地址有小量但非零解码误差，不能宣称表示本身优于规范注册 3D。',
'生产部署门槛尚未通过：现有 Ruckig 接入使本测试失去可见目标；stock 主要生产伺服函数因未定义变量崩溃；stock 生产体坐标查询严重违反皮肤 d=0 边界；缺失检测时仍消费模拟器的几何 u。结果不支持把系统称为已部署、已物理验证或已完成首次合格超声成像。','',
'一、实际执行层级与信息契约',
'1. adapter.py：真实导入 DMP_playground.dmp_honest.dmp.fit_discrete_dmp/DiscreteDMP.rollout 和 kmp_honest.kmp.fit_phase_kmp/PhaseKMP.predict。候选 demonstration 是调用方给定 start/goal/局部法线构造的正弦偏移候选；这只是实际 primitive 路径生成，不是完整伺服或机器人可执行保证。DMP 固定至少161点积分后采样，避免查询采样预算改变积分稳定性。端点拟合误差保留并报告，不强行覆盖。',
'2. run_motion_validation.py 完整调用真实 run_honest_kmp_servo；DMP 主路径由真实 DMP prior/bake 生成，源码 run_honest_dmp_servo 也复用该同一分区伺服函数。包含当前位姿探头平面观测、因果历史点重建切线、接触表面投影和可选真实 Ruckig Module C。',
'3. 生产 atlas 实际加载：右腿31785皮肤顶点/63104三角面；186624体顶点/1058184四面体。实际调用 SurfaceAtlasRefiner、query_atlas_coordinates、距离比深度算子、SMPL-X关节变换和 LBS；没有替换成圆柱几何。',
'4. primary production 是源码中功能最多的配置候选，不能据此认定是实际部署过或质量最强。stock 函数调用首先保留失败；仅在 owned harness 中补模块全局变量别名后另记结果，未修改原源文件。',
'5. 合法公开输入：测试表面 mesh/对应关系、身体位姿几何、法线、0/30度合法训练 anatomy ensemble、公共候选 start/goal、已取得的可见图像横向/深度观测和历史。test 60度血管曲线只给环境构造与评价；honest loop 的 gt_tangent 用于记录指标，在线 roll 使用已可见历史点因果重建。',
'6. 特权/边界：scheduled synthetic vessel 的深度由测试 anatomy 构造；这属于环境生成。stock honest 观测函数在 visible=False 时仍返回几何 u/depth 且控制消费 u，超过 missing-only 真实传感器契约。primary 函数更直接使用 bent 真值 rho/roll、vessel tangent path、VesselOracle 和 nominal replay，故不能作为未知目标获取时的因果无真值基线。','',
'补充范围：build_prebake 的 crop 由已知vessel-centered footprint定义；0/30度合法prior由已知解剖扫描路径提取。完整伺服实验是已准备路径附近的继续扫描/居中测试，不能称为仅从粗人体区域、无目标路径先验启动的首次成像检验。','',
'二、公平的完整伺服对照',
'原 stock run_dmp_benchmark.py 分别基于 DMP/KMP bake 生成 scheduled vessel，两方法各自有不同世界目标。因此其成功运行只作现有管线回归，不能据此比较方法。新测试将 KMP 固定世界血管曲线逐点复制给 DMP；两个方法共享曲线、表面、可见条件、局部伺服与相同步长。合法 prior knees=[0,30]，持出 pose=60度。',
'以下 u 是合成探头平面横向误差。first_synthetic_centered_time_s 仅指 visible 且 |u|<=3mm 的首帧诊断，绝非合格超声/身份确认成功。全部计入启动观察；没有学习 A/B 响应模型。',
'method/scenario/gain/ModuleC : mean|u|mm / p95mm / visible fraction / max translation step mm']
for x in rows:
    lines.append(f"{x['id']}: {x['u_abs_mm']['mean']:.6f} / {x['u_abs_mm']['p95']:.6f} / {x['visible_fraction']:.6f} / {1000*x['translation_step_m']['max']:.6f}")
lines += ['',
'无 Module C 名义完整伺服：KMP 16.992->1.515mm，DMP16.697->1.677mm；±10mm固定目标错配也可纠偏。该单个合成 pose 的机械事实不能升级为独立对象上的统计疗效；147帧不是147个独立样本。',
'Module C 未解决部署问题：使用现有 _closed_loop_module_c_cfg、真实 Ruckig，KMP/DMP 可见比例均降至0.0272，均未达到上述合成居中阈值；保留该失败，不替换为无平滑版本冒充执行控制器。确切根因/闭环延迟重设计尚未验证。','',
'三、缺失观察反事实',
'只看走了多远不能判定泄漏；无检测时执行预定 nominal scan 是允许的。测试先比较相同全不可见世界 gain1/gain0，然后构造两个隐藏横向偏移±10mm的深目标世界，比较相同公共 missing 历史前缀内动作是否依赖隐藏几何。']
for method,x in s['missing_observation_counterfactual'].items():
    z=x['identical_missing_prefix_gain1_path_difference_m']
    lines.append(f"{method}: 两世界可见帧总数={x['two_hidden_offsets_visible_counts']}；共同缺失前缀={x['identical_missing_prefix_steps']}步；前缀gain1路径差均值={1000*z['mean']:.6f}mm/max={1000*z['max']:.6f}mm；gain0路径差max={1000*x['same_public_missing_history_gain0_hidden_world_path_difference_m']['max']:.6f}mm。")
lines += ['KMP 两世界全147步不可见；DMP只在前90步保证相同缺失观测，后续其中一世界出现检测，后续差异不能用作相同观测反事实。源码明确未门控横向 u，因此结论是 stock 消费模拟器全几何 u，违反 missing-only 数据接口；这不是其内部 richer simulator-observation 契约下的数学矛盾。','',
'四、几何、身份和生产深度',
f"真实 primitive 共同 SE(3) 最大位置误差={max(x['position_se3_max_m'] for x in m['primitive']['rows']):.3g}m；完整 honest KMP loop 位置最大误差={s['full_honest_loop_se3']['position_max_m']:.3g}m，旋转元素最大误差={s['full_honest_loop_se3']['rotation_max']:.3g}。",
f"生产表面96随机三角面/barycentric点，无 reference_points 真点提示：调和 surface decode 均值={1000*g['surface_decode_error_m']['mean']:.6f}mm，p95={1000*g['surface_decode_error_m']['p95']:.6f}mm，max={1000*g['surface_decode_error_m']['max']:.6f}mm。",
'公平规范注册3D：双方共享相同 mesh 对应面、barycentric系数、生产stencil派生SMPL-X权重、60度膝关节已求值变换。3D baseline 保留规范点并使用相同附件运输，误差为0（按构造代数精确）；调和额外经历地址解码，运输最大误差约3.004mm。因此此条件下未证明调和优于规范注册3D。',
f"LBS反变换最大位置误差={g['exact_lbs_inverse_position_error_m']['max']:.3g}m；共同SE3运输max={g['common_se3_transport_error_m']['max']:.3g}m。错误theta地址+0.6rad平均移位={1000*g['wrong_body_address_0_6rad_displacement_m']['mean']:.3f}mm，说明地址身份必须正确，坐标表示不会自动解决错身份。",
f"stock生产深度：16个真皮肤顶点存储d=0，但运行时d均值={d['runtime_abs_d_mean']:.6f}、max={d['runtime_abs_d_max']:.6f}。同一个距离比算子传入真实skin数组则为0。原因：atlas.py把 harmonic_vertices前skin_count节点当皮肤；production layered顺序不符合此假设；这些假皮肤与真皮肤采样点距离最高0.622m。且runtime d重新计算解析距离比，不插值baked harmonic_d，本来就是不同深度场。",
f"depth_adapter.py source-free修正只把真实atlas.skin_vertices传给原距离比算子，保留原h/theta查询。修正后皮肤dmax={d['corrected_adapter_skin_d_max']:.3g}，h/theta变化max={d['corrected_adapter_h_theta_unchanged_max']:.3g}，共同SE3深度误差max={d['corrected_adapter_depth_common_se3_max']:.3g}。这是皮肤绑定修正，不代表已使解析深度等于layered baked深度，也没有解决所有体坐标契约。",
'生产表面连续曲线80点h=0.1..0.9的步长max约11.1mm；这是特定采样曲线诊断，不是全域单射/连续性定理。','',
'五、primary production 执行状态',
json.dumps(s['primary_production'],ensure_ascii=False,indent=2),
'修正仅为 owned harness 给 module global use_scheduled_vessel 赋现有 scheduled_observation_truth_enabled 的值，下一次 stock 调用仍会失败。此结果拥有更多特权信息且另有自身环境生成，不能与共享世界的 honest loop 性能直接排序。','',
'六、复现与交付',
'所有命令在 /media/camp/EXT_DRIVE/Among_US 为工作目录，或运行绝对路径 harness（它自动切换）。无需GPU、torch或机器人设备。未更改原仓库源文件，已有dirty source/status原样保留。',
'环境：'+json.dumps({k:m['provenance'][k] for k in ['python','version','packages','threads']},ensure_ascii=False),
'主命令：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python /media/camp/EXT_DRIVE/RealUS_playground/NMI_idea/experiments/ab_validation_20261110/motion/run_motion_validation.py',
'source tests：PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest DMP_playground.dmp_honest.test_dmp DMP_playground.kmp_honest.test_kmp tests.test_leg_volume_coordinates -v；23 tests=22pass+1skip（旧latest_canonical不可用）。',
'stock full CLI：PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m DMP_playground.dmp_honest.run_dmp_benchmark --config DMP_playground/configs/smplx_recovered_20261108.json --output <owned motion>/full_servo；成功。默认default.json尝试失败，旧canonical/centerline缺失已记录。',
'adapter API：generate_candidates(start_xyz,goal_xyz,normal_xyz=(0,0,1),n=41,rotation=None,frame="railbase",duration_s=1.,lateral_offsets_m=(-.015,0,.015),methods=("dmp","kmp")) -> list dict。T_tcp[W,4,4]/xyz_m/rotations/normal_xyz/phase；单位m，caller声明frame，提供rotation则常量保留；生成wall_s/path_length/max_step/start_error/goal_error/max_speed均返回。不能把该wall_s当真实机器人执行时长。',
'metrics.json：全部结构化指标、包版本、原源文件/实际资产/本次harness SHA256、git HEAD、既有dirty status、随机种子20261110。candidate_paths.json：完整候选位姿。production_geometry_samples.npz：对应、解码、运输、权重/变换和连续曲线。depth_contract.json：stock/corrected逐样本。controlled_servo/*.npz：逐帧p/R/u/depth/visible/phase/环境目标。servo_metrics.csv：每方法情景聚合。logs/：实际命令输出、stock失败traceback、unit tests。','',
'七、审计定位（源根 /media/camp/EXT_DRIVE/Among_US）',
'DMP_playground/kmp_honest/servo.py:226-230 visible观测判定；:289横向image_u未按visible门控；:304-308真实ModuleC接口；gt_tangent仅metrics，:350 estimator.ingest按visible门控。',
'DMP_playground/run_trajectory_extraction_demo.py:1149-1153 bent GT rho；:1157-1159 bent replay roll；:1182-1190 vessel tangent/normal/nominal oracle输入；:1222未定义use_scheduled_vessel。',
'src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates/atlas.py:482假设前skin_count个体节点是skin；harmonic.py:1064-1068取这些节点；:1112重算解析距离比d。',
'DMP_playground/kmp_honest/imaging.py:187由合法prior和测试公开patch bake；:188-191 test anatomy只用于环境生成scheduled depth。',
'范围限制：这是实际源模块的CPU计算/合成观察检验，不是实际B-mode、真实机器人/探头接触、力控制、独立受试者或临床系统验证。未检测真实身份/错误停止，无条件反应学习，未在本目录内完成dIRD联合伺服。A/B集成由parent core使用本adapter另测；不能从本报告推断B已优于A。','']
(P/'REPORT.txt').write_text('\n'.join(lines))
with (P/'servo_metrics.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['id','method','scenario','gain','module_c','u_mean_mm','u_p95_mm','visible_fraction','first_synthetic_centered_s','wall_s','path_length_m','max_step_m'])
    for x in rows:w.writerow([x['id'],x['method'],x['scenario'],x['gain'],x['use_module_c'],x['u_abs_mm']['mean'],x['u_abs_mm']['p95'],x['visible_fraction'],x['first_synthetic_centered_time_s'],x['wall_s'],x['path_length_m'],x['translation_step_m']['max']])
print(P/'REPORT.txt')

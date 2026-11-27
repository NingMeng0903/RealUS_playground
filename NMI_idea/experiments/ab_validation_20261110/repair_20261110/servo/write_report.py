from pathlib import Path
import csv,hashlib,json,subprocess
P=Path(__file__).resolve().parent
SOURCE=Path('/media/camp/EXT_DRIVE/Among_US')
m=json.loads((P/'metrics.json').read_text())
lines=['伺服与生产入口修复报告','2026-11-10','',
'已修复：honest KMP/DMP完整局部伺服在不可见时消费模拟器隐藏u/depth/confidence的问题；生产主入口 use_scheduled_vessel 未定义崩溃。未更改 production harmonic、dIRD 或 ModuleCSmoother 源码。',
'初始检查：本次拥有的两个tracked源文件 kmp_honest/servo.py 和 run_trajectory_extraction_demo.py 在开始时无既有diff；整个 dmp_honest 目录为预先存在的untracked内容，原样保留。其他并行工作者脏改动不回退。','',
'一、源码变化',
'1. kmp_honest/servo.py 的 filter_obs_for_control：visible=False时删除image_u/depth/out-of-plane/slice_margin，并将gamma、gamma_pos、gamma_tangent全部置0。完整loop实际调用该过滤器；缺失时不重建/摄入测量，横向反馈由visible && gamma>0显式门控。0只用于门控分支的算术占位，不是检测到中心。',
'2. 结果增加 public_image_u_m/public_image_depth_m，missing为NaN，measured_vessel缺失也为NaN；旧image_*结果保留模拟器评价几何以兼容现有plot/metric接口，函数说明明确其评价用途。此次NPZ将原始量命名truth_image_*，公开量单列，避免真实0与缺失混淆。',
'3. dmp_honest/servo.py复用同一真实honest loop，因此无需改动该已有包装文件；回归分别调用KMP/DMP实际入口。',
'4. run_trajectory_extraction_demo.py 内部局部初始化 use_scheduled_vessel=scheduled_observation_truth_enabled(servo_cfg)，并复用于启动选项；不再依赖harness全局变量别名。scheduled on/off两条真实入口均执行成功。ClosedLoopServoResult追加可选公开测量字段，现有构造方式兼容。','',
'二、缺失观测因果回归',
'同一公开prior/表面/nominal路径，两个隐藏世界只有深目标横向±10mm不同；额外深度120mm，整个147步均missing。KMP与DMP各自两世界的p_cmd、r_cmd和控制输入逐元素一致，最大差全部0；confidence全0，public图像数组全NaN。几何truth_image_u明显不同，证明确实改变隐藏目标而非重复世界。允许缺失时执行nominal扫描；不把移动距离本身判为泄漏。',
'另有完整loop回归：相同5步可见观测前缀后，分别给予不同隐藏u/depth但相同missing历史，动作仍逐元素一致；真实可见偏移仍产生反馈，真实可见u=0保持为有效测量。',
json.dumps(m['missing'],ensure_ascii=False,indent=2),'',
'三、实际关闭Ruckig与整个Module C的区别',
'causal_no_ruckig_config.json：enable:true/use_ruckig:false，确实关闭Ruckig，但保留现有fallback位置混合与旋转步限。',
'causal_module_c_bypass_config.json：enable:false/use_ruckig:false，旁路整个Module C。不是仅关闭Ruckig。',
'真实调用均use_module_c=True，由配置决定。用构造器计数监测实际Ruckig：on为1，off/bypass为0。原可开启路径保留，未全局关闭Ruckig。ModuleCSmoother.enable:false的现有旁路和controller enable判断已经生效，无需新增旁路源码。',
'共同固定世界目标/相同0、30度合法prior/60度持出pose；以下truth_geometry_u为包含不可见帧的几何评价误差，不能解释为所有帧都有传感器测量。public_visible_u只在可见帧求平均，可能受可见选择影响。',
'id | visible fraction | truth mean|u| mm | visible-only mean|u| mm | Ruckig constructor calls']
for r in m['rows']:
    lines.append(f"{r['id']} | {r['visible_fraction']:.6f} | {r['truth_geometry_u_abs_mean_mm']:.6f} | {r['public_visible_u_abs_mean_mm']} | {r['ruckig_constructor_calls']}")
lines += ['',
'关闭Ruckig且保留fallback：KMP/DMP可见率均78.91%，几何平均|u|分别10.482/10.981mm，明显好于Ruckig-on的2.72%，但未恢复完整跟踪。旁路整个Module C：两者均100%可见，误差1.515/1.677mm。故恢复上一轮100%可见率需要本实验中的完整Module C旁路，不能写成只关闭Ruckig即可。Ruckig及fallback闭环动态的根因、时序/速度适配仍未解决。','',
'四、生产主入口范围',
'真实 run_closed_loop_servo_on_bent 在 scheduled True/False、Ruckig-off fallback/ModuleC-bypass四组合均完成，无全局别名，所有off配置实际Ruckig构造次数0；回归测试覆盖两种scheduled开关和enable:false合并配置。',
'该主入口仍是oracle-assisted参考：bent GT rho/roll、vessel tangent path、VesselOracle、测试anatomy nominal replay会进入控制。此次只修复崩溃和验证配置生效，不把它改称未知目标因果基线；它仍不能用于证明missing-only传感器下的盲获取。所有primary结果单列。',
'honest实验也有强先验：vessel-centered预制crop、已知扫描路径演示和测试身体表面；属于预制路径附近的继续扫描/居中，不是从宽人体区域无目标路径启动的首次成像。','',
'五、验证与复现',
'22 tests全部通过：4项新增完整loop因果回归、2项新增实际主入口/配置回归、16项已有KMP/DMP算子回归。日志logs/regressions.log。',
'命令（Among_US工作目录）：PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m unittest DMP_playground.kmp_honest.test_servo_causality DMP_playground.tests.test_servo_entrypoint DMP_playground.kmp_honest.test_kmp DMP_playground.dmp_honest.test_dmp -v',
'完整实验：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python '+str(P/'run_repaired_servo.py'),
'环境：'+json.dumps(m['provenance'],ensure_ascii=False,indent=2),
'metrics.json记录全部情景有效配置、wall time、构造器计数和因果差值；逐帧NPZ记录p/R、phase、visibility、public测量、truth评价量、confidence、控制输入和重建点。sources.patch记录tracked源差异；source_manifest.json记录新增回归测试及执行文件SHA256。',
'执行层级：只执行CPU探头位姿路径模拟和几何接触代理；没有joint关节轨迹层、dIRD可行性、机器人动作、真实力/接触或真实B-mode。此次修复不等价于物理执行控制器验证。','']
(P/'REPORT.txt').write_text('\n'.join(lines))
with (P/'metrics.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['id','visible_fraction','truth_mean_u_mm','visible_only_mean_u_mm','ruckig_constructor_calls','wall_s'])
    for r in m['rows']:w.writerow([r['id'],r['visible_fraction'],r['truth_geometry_u_abs_mean_mm'],r['public_visible_u_abs_mean_mm'],r['ruckig_constructor_calls'],r['wall_s']])
files=[SOURCE/'DMP_playground/kmp_honest/servo.py',SOURCE/'DMP_playground/run_trajectory_extraction_demo.py',SOURCE/'DMP_playground/dmp_honest/servo.py',SOURCE/'DMP_playground/kmp_honest/test_servo_causality.py',SOURCE/'DMP_playground/tests/test_servo_entrypoint.py']+list(P.glob('*.py'))+list(P.glob('*config.json'))
(P/'source_manifest.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
diff=subprocess.check_output(['git','diff','--','DMP_playground/kmp_honest/servo.py','DMP_playground/run_trajectory_extraction_demo.py'],cwd=SOURCE,text=True)
(P/'sources.patch').write_text(diff)
print(P/'REPORT.txt')

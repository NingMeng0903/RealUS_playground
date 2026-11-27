from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
R=H.parent/'harmonic'
read=lambda name:json.loads((H/name).read_text())
assets={s:R/'shaft_roi_v2'/f'atlas_{s}.npz' for s in ['left','right']}
A={s:read(f'shaft_roi_v2_{s}_audit.json') for s in assets};C={s:read(f'shaft_roi_v2_{s}_contract.json') for s in assets};M={s:read(f'shaft_roi_v2_{s}_material_overlap.json') for s in assets}
summary={'verdict':'通过所声明 shaft ROI、固定 LBS 模型和7个离散姿态的独立数值门；不等于整腿/任意姿态/物理超声验证','domain_h':[.14,.94],'coordinate_domain':'shaft_roi','inference_scope':['原生产Loop皮肤本侧筛选后的shaft子域','角度0/15/30/45/60/75/90度同一固定SMPL-X权重','数值mesh/材料chart检验；非真实接触或B-mode'],'sides':{}}
for s in assets:
 a,c,m=A[s],C[s],M[s];summary['sides'][s]={'asset_sha256':a['asset_sha256'],'physical_raw_overlap':a['canonical_overlap']['positive_interior_overlap_pairs'],'canonical_volume':a['canonical_volume'],'physical_pose':{p:{'relative_inversion_count':v['pose_det_ratio']['nonpositive_count'],'minimum_relative_volume':v['pose_det_ratio']['min'],'positive_volume_overlaps':v['overlap']['positive_interior_overlap_pairs'],'minimum_corner_scaled_jacobian':v['corner_scaled_jacobian']['min']} for p,v in a['poses'].items()},'material_overlap':m['base']['positive_interior_overlap_pairs'],'material_periodic_overlap':m['periodic_three_copies']['positive_interior_overlap_pairs'],'full_boundary_self_intersections':c['complete_tet_boundary_cgal']['cgal_intersecting_face_pairs'],'full_boundary_topology':c['complete_tet_boundary_topology'],'surface_binding':c['surface_binding'],'weight_lineage':c['surface_weight_lineage'],'volume_weight_rule':c['volume_weight_rule'],'inverse_roundtrip':c['production_inverse_roundtrip'],'inverse_seam':c['periodic_inverse_seam'],'outside_inverse_all_rejected':all(x['inverse_rejected'] for x in c['outside_inverse_domain']),'outside_forward_all_rejected':all(x['rejected'] for x in c['outside_forward_domain']),'executed_volume_refine_source_sha256':c['volume_refine_source_sha256']}
(H/'FINAL_METRICS.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
s='''独立调和坐标几何审计报告（2026-11-10，已封存）

结论
shaft_roi_v2 两腿在明确的 h∈[.14,.94] 新子域内通过本轮数值门。
这不是旧完整腿部域修复完成，也不是任意SMPL-X姿态、实机接触或物理
超声验证。旧资产/失败cap/有限SRS/QP诊断均保留，未覆盖或回退生产修改。
本审计不恢复旧B机械增益结论；192边连续QP图没有运行。

检查方法与冻结范围
PROTOCOL.txt记录阈值/样本及随后用户指定的7姿态扩展。每tet全量有向
体积；完整AABB BVH+tet SAT正体积相交，物理空间容差1e-10m；材料空间
以所在angular cell正确unwrap θ，θ/(2π),h,d做同一全量检测，容差1e-10。
再检查θ±2π完整三副本。边界使用CGAL完整self-intersection检测。
SAT核5个独立几何对照（分离、正确共面接触、同侧折叠、包含、重合）通过。
所有方法是有限浮点资产检查，不是数学上对所有形状/姿态的保证。

旧资产根因与基线
两腿旧全体积均非可注入：主体tet采用负定向，但left有2552个、right有
169个相反定向，分别全部集中在过膝station-cell33/32。ring法向突跳
16.72877°/10.55966°与骨轴折角一致；generator分段骨轴tangent跨膝不连续，
随后直接连接相邻ring六tet，未作全域Jacobian/重叠验收。
旧0度identity-transport全量重叠对left347794/right7529；其中无共享vertex
315233/4572。30度相对rest新增反转1606/2208，60度3938/4162。
旧体节点W是固定k6同腿皮肤近邻代理；其运输结果不是实测体权重证据。

旧皮肤本体不是完全干净：CGAL两腿各289对自交；native/base两腿各37对。
right的37/37对均跨本侧/对侧LBS主导face，right chart986faces中78片由
left-leg权重主导；源码跨中线15mm几何mask把对侧groin皮肤混入单腿域。
按face平均本侧hip/knee/ankle/foot权重≥对侧、保留真实xyz的独立原型消交。
上游重烘clean Butterfly L3两腿CGAL0，原480native节点位移0；但其新增
面并非原物理Loop整面，dense采样距native面mean约.62mm/max2.573mm。
故主v2改用筛净的原生产Loop皮肤和原CSR附件，v1 Butterfly候选仅保留对照。

旧完整皮肤两腿均无端盖：boundary edge496/464，均2闭环，内部winding
无冲突但不watertight。72真实皮肤平面×108ray，left120/right133射线有
多个正交点，另96/69条无正交点；不能用最大hit假造单半径星形管。

v2全部物理与材料检查
每腿62208节点、322056tet。实际写盘float32重载后全正、0退化、0量化
翻转；全部raw物理tet正体积相交0。完整tet边界含内管及端环，不应与
仅有外侧开放物理skin patch混淆；完整边界watertight、nonmanifold0、
winding冲突0、CGAL自交0。体内节点不属于边界，未使用边界vertex计数
不是丢失体节点。

每腿7姿态中，所有tet相对rest反转0、全物理BVH/SAT相交0。以下为最小
有向体积比 det_pose/det_rest（同一固定W，角度为local-X knee轴）：
角度      left          right
'''
for angle in ['0','15','30','45','60','75','90']:
 s+=f'{int(angle):>3}    {A["left"]["poses"][angle]["pose_det_ratio"]["min"]:.9f}   {A["right"]["poses"][angle]["pose_det_ratio"]["min"]:.9f}\n'
s+='''
材料坐标空间每腿322056tet全量冲突0；θ±2π三副本966168tet仍0。
局部材料Jacobian全为同一负号，核心r=0不在本环状域内，θ奇点未冒充有效地址。
同一cell只有两个θ节点值，实际sin/cos角度插值是该cell线性θ的单调重参数化；
全量材料检测与1024独立inverse样本共同支持本有限chart无检测到的多值。
不把它推广为任意非刚性变形的等变/单射定理。

皮肤、权重、反解与域外门
'''
for side in ['left','right']:
 c=C[side];b=c['surface_binding'];i=c['production_inverse_roundtrip'];s+=f'{side}: 7776个outer节点源三角bary最大位置误差 {b["maximum_vertex_barycentric_error_m"]:.9g}m；\n  15336面中心到真实旧Loop源skin mean {b["outer_face_centroids_to_actual_source_triangles"]["mean_m"]*1000:.6f}mm，max {b["outer_face_centroids_to_actual_source_triangles"]["max_m"]*1000:.6f}mm。\n  1024随机tet内部点（seed41109，不给inverse真实point reference）max roundtrip {i["max_position_error_m"]:.9g}m；\n  θ=-2π/0/2π/4π周期inverse最大位置差 {c["periodic_inverse_seam"]["max_position_difference_m"]:.9g}m。\n'
s+='''原production CSR→source owned W独立重构差为0；outer bary W误差≤1.73e-7。
体W为outer与逐站core环平均W的固定径向凸组合；全部非负、rowsum误差
≤1.20e-7，并在全部姿态原样复用。此体权重仍是明确的SMPL-X插值模型，
不是独立测得的组织形变。旧/新体权重扩展规则不同，不能把运输改进全部
归因单独网格修复。
旧inverse会对域外ξ静默clip；失败记录保留于v1 contract。生产strict修复后，
每腿4个ξ域外探针（h上下界、d上下界）均由inverse自身拒绝，两个轴向
域外p也均由forward拒绝。原坐标域之外不得默认最近点续用。
皮肤距离是全部节点/面中心的有限采样结果，并非完整Hausdorff证明。

整腿端盖/sliver失败与未通过项
直接fan cap、直接PSLG、cap-only MeshFix均曾失败。独立native right仅cap
MeshFix可保全部原skin xyz/faces且watertight，但仍2个proper交，未接受。
只删cap4重复face仍7boundary edge/3nonmanifold；3次cap-only2-2 flip后
仍1nonmanifold；Steiner双向微抬的失败均保留。反向重复skin/cap面抵消版
虽TetGen成功，但删去4个skin耳面、端部bbox缩约6mm，明确不作全保形接受。
CGAL self-union备选也会将2.2634e-4m²旧skin内部化，未冒称原skin全保留。

worker的完整clean-smooth+MeshFix TetGen原型报告全正tet，但最小det仅
right7.3175e-22、left1.3396e-20（低于本审计1e-18退化门）；“全正”不能
代替质量、序列化与姿态验证。该完整腿域未获本独立审计通过。主v2采用
显式缩小shaft ROI来避开未解决近端/远端域，不把该域外范围静默删除。

复现与封存
run_final_audit.sh重跑v2物理7pose、材料/周期全量、CGAL边界、附件/反解。
FINAL_METRICS.json为结构化汇总；每腿*_audit.json、*_contract.json、
*_material_overlap.json含完整计数/样本/时长；*.npz保留切片和固定W。
SEALED_MANIFEST.json记录本目录脚本/指标、v2四个NPZ及执行源文件SHA256。
生产generator、资产、伺服与dIRD源码均未由本独立审查者编辑。
'''
(H/'REPORT.txt').write_text(s)
inputs=list(assets.values())+[R/'shaft_roi_v2'/f'attachments_{x}.npz' for x in assets]
source=Path('/media/camp/EXT_DRIVE/Among_US/src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates')
inputs += [source/x for x in ['atlas.py','harmonic.py','volume_refine.py','shaft_rebuild.py']]
files=[p for p in H.iterdir() if p.is_file() and p.name!='SEALED_MANIFEST.json' and p.suffix in ['.py','.cpp','.txt','.json','.sh']]+inputs
manifest={'scope':'independent geometry audit; no production edits','sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'asset_hashes_match_audited':{side:hashlib.sha256(assets[side].read_bytes()).hexdigest()==A[side]['asset_sha256'] for side in assets},'inverse_source_matches_executed':{side:hashlib.sha256((source/'volume_refine.py').read_bytes()).hexdigest()==C[side]['volume_refine_source_sha256'] for side in assets}}
(H/'SEALED_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'assets':manifest['asset_hashes_match_audited'],'inverse_source':manifest['inverse_source_matches_executed']},indent=2))

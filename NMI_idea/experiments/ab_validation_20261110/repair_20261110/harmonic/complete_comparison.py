from pathlib import Path
import sys,json,numpy as np,trimesh
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
from validate_harmonic_repair import REPO,load_leg_volume_atlas,query_atlas_coordinates,nearest_core
r=json.loads((OUT/'production_metrics.json').read_text());old=json.loads((OUT/'before_metrics.json').read_text())
x=np.asarray(r['right']['runtime_distance_ratio']['original_16_skin_xi']);before=np.asarray(old['skin_xi']);diff=x-before;diff[:,0]=(diff[:,0]+np.pi)%(2*np.pi)-np.pi
comparison=dict(original_16_skin_depth_abs_mean_before=old['skin_depth_abs_mean'],original_16_skin_depth_abs_max_before=old['skin_depth_abs_max'],original_16_skin_depth_abs_mean_after=float(abs(x[:,2]).mean()),original_16_skin_depth_abs_max_after=float(abs(x[:,2]).max()),original_16_h_theta_abs_change_max=np.max(abs(diff[:,:2]),axis=0).tolist(),before_96_node_depth_error_mean=old['node_depth_abs_error_mean'],before_96_node_depth_error_max=old['node_coordinate_abs_error_max'][2],after_explicit_baked_accepted_sampled_node_depth_error_max=r['right']['baked_native_tetrahedra']['node_coordinate_abs_error_max'][2])
(OUT/'before_after_comparison.json').write_text(json.dumps(comparison,indent=2))
a=load_leg_volume_atlas(REPO/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_left.npz');mesh=trimesh.Trimesh(vertices=a.skin_vertices,faces=a.skin_faces,process=False)
for record in r['left']['baked_native_tetrahedra']['overlap_evidence']:
 p=np.asarray(record['query_point_m'],np.float32)[None]
 q=query_atlas_coordinates(a,p)[0][0]
 skin=trimesh.proximity.closest_point_naive(mesh,p)[1][0];core=nearest_core(p,a.core_points)[1][0]
 record['default_distance_ratio_xi']=q.tolist();record['independent_distance_ratio']=float(skin/(skin+core))
 assert abs(record['independent_distance_ratio']-q[2])<1.e-6
(OUT/'production_metrics.json').write_text(json.dumps(r,indent=2));(OUT/'left_metrics.json').write_text(json.dumps(r['left'],indent=2));(OUT/'left_overlap_evidence.json').write_text(json.dumps(r['left']['baked_native_tetrahedra']['overlap_evidence'],indent=2))
print(json.dumps(comparison,indent=2));print(json.dumps(r['left']['baked_native_tetrahedra']['overlap_evidence'],indent=2))

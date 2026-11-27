from pathlib import Path
import sys,json,time,numpy as np
ROOT=Path('/media/camp/EXT_DRIVE/Among_US');BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas,tet_determinants
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import _compute_vertex_normals
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
canonical=np.load(ROOT/'outputs/anatomy_retarget/recovered_smplx_20261108/smpl_canonical_weights.npz')
plan=dict(thickness_candidates_m=[.0001,.0005,.001,.002],layers=3,selection='largest thickness passing every canonical and 0..90 step15 signed-volume gate on both sides; later global overlap/boundary/material gates mandatory',scope='contact collar local depth, not medial-depth volume',geometry='inward area-averaged vertex normal offsets of identical true owned skin',weight_model='same fixed original normalized surface W on each column at every depth')
(BASE/'frozen_contact_collar_plan.json').write_text(json.dumps(plan,indent=2))
def collar_nodes_tets(m,thickness):
 n=_compute_vertex_normals(m.vertices,m.faces).astype(float);n/=np.linalg.norm(n,axis=1)[:,None];levels=np.array([0.,.5,1.])*thickness;nodes=(m.vertices[None].astype(float)-levels[:,None,None]*n[None]).reshape(-1,3);nv=len(m.vertices);sf=np.sort(m.faces,axis=1);p=np.argsort(m.faces,axis=1);odd=((p[:,0]>p[:,1]).astype(int)+(p[:,0]>p[:,2])+(p[:,1]>p[:,2]))%2==1;tets=[]
 for layer in range(2):
  a,b,c=(sf+layer*nv).T;A,B,C=(sf+(layer+1)*nv).T;t=np.stack([np.column_stack([a,b,c,C]),np.column_stack([a,b,C,B]),np.column_stack([a,A,B,C])],1);t[odd,:,0],t[odd,:,1]=t[odd,:,1].copy(),t[odd,:,0].copy();tets.append(t.reshape(-1,4))
 t=np.vstack(tets).astype(np.int32);t[:,[0,1]]=t[:,[1,0]]
 return nodes,t,n
rows=[]
for side in ['right','left']:
 m=SurfaceMaterialAtlas.load(BASE/f'contact_multichart_{side}.npz');w=np.tile(m.weights,(3,1))
 for thickness in plan['thickness_candidates_m']:
  started=time.perf_counter();v,t,n=collar_nodes_tets(m,thickness);det=tet_determinants(v,t);row=dict(side=side,thickness_m=thickness,nodes=len(v),tets=len(t),canonical_nonpositive=int(sum(det<=0)),canonical_det_min=float(det.min()),pose_rows=[])
  if row['canonical_nonpositive']==0:
   for angle in range(0,91,15):
    pose=np.zeros((55,3));pose[5 if side=='right' else 4,0]=np.deg2rad(angle);tf=joint_global_transforms(pose_axis_angle=pose.ravel(),rest_joints=canonical['rest_joints'],parents=canonical['parents'])@canonical['inverse_bind'];pv=np.einsum('nj,jab,nb->na',w,tf[:,:3],np.column_stack([v,np.ones(len(v))]));d=tet_determinants(pv,t);row['pose_rows'].append(dict(knee_deg=angle,nonpositive=int(sum(d<=0)),det_ratio_min=float((d/det).min())))
  row['passed_local_geometry']=row['canonical_nonpositive']==0 and all(x['nonpositive']==0 for x in row['pose_rows']);row['wall_s']=time.perf_counter()-started;rows.append(row);(BASE/'contact_collar_screen.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
passed=[x for x in plan['thickness_candidates_m'] if all(next(r for r in rows if r['side']==side and r['thickness_m']==x)['passed_local_geometry'] for side in ['right','left'])]
(BASE/'contact_collar_selected.json').write_text(json.dumps(dict(selected_thickness_m=max(passed) if passed else None,selection_status='local signed-volume gates only, not release certification'),indent=2))

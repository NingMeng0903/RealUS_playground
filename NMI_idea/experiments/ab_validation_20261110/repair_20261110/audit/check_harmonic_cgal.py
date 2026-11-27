"""Audit-only exact-predicate boundary self-intersection subset (raw and 90deg)."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path[:0]=['/media/camp/EXT_DRIVE/Among_US','/media/camp/EXT_DRIVE/Among_US/src']
from igl.copyleft.cgal import remesh_self_intersections
from DMP_playground.core_algorithms.patch_binding import load_canonical_bundle
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import blend_lbs_transforms
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'harmonic/shaft_roi_v2';canon=load_canonical_bundle('/media/camp/EXT_DRIVE/Among_US/outputs/anatomy_retarget/recovered_smplx_20261108');out={}
for side in ['left','right']:
 d=np.load(ROOT/f'atlas_{side}.npz');v=d['harmonic_vertices'];t=d['harmonic_tets'];att=np.load(ROOT/f'attachments_{side}.npz');faces=np.concatenate([t[:,ix] for ix in [[1,2,3],[0,3,2],[0,1,3],[0,2,1]]]);_,first,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True,return_index=True);f=faces[first[counts==1]];rec={'boundary_faces':len(f),'interior_nonmanifold_faces':int(sum(counts>2))}
 for deg in ['raw',90]:
  if deg=='raw':pv=v
  else:
   pose=np.zeros((len(canon['joint_names']),3),np.float32);pose[list(map(str,canon['joint_names'])).index(side+'_knee'),0]=np.deg2rad(deg);G=joint_global_transforms(pose_axis_angle=pose.ravel(),rest_joints=canon['rest_joints'],parents=canon['parents'])@canon['inverse_bind'];B=blend_lbs_transforms(att['volume_lbs_weights'],G);pv=np.einsum('nij,nj->ni',B,np.column_stack((v,np.ones(len(v),np.float32))))[:,:3].astype(np.float32)
  start=time.perf_counter();r=remesh_self_intersections(pv.astype(float),f.astype(np.int64),detect_only=True,first_only=False,cutoff=10000000);pairs=np.asarray(r[2]);rec[str(deg)]={'intersecting_face_pairs':len(pairs),'pairs':pairs[:32].tolist(),'wall_s':time.perf_counter()-start}
 out[side]=rec;print(side,rec,flush=True)
(HERE/'harmonic_cgal_final.json').write_text(json.dumps(out,indent=2))

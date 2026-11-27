"""Fresh public API d=0/1, endpoint, exact face-center and source-lineage audit."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
sys.path[:0]=['/media/camp/EXT_DRIVE/Among_US','/media/camp/EXT_DRIVE/Among_US/src']
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
from DMP_playground.run_volume_coordinate_query import query_and_transport
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent/'harmonic';A=ROOT/'shaft_roi_v2';out={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for side in ['left','right']:
 a=load_leg_volume_atlas(A/f'atlas_{side}.npz');m=VolumeTetRefiner.from_atlas(a);rng=np.random.default_rng(770311);lo,hi=float(a.harmonic_h.min()),float(a.harmonic_h.max())
 theta=rng.uniform(-4*np.pi,4*np.pi,1024);h=rng.uniform(lo,hi,1024);depth=np.repeat([0.,1.],512);xi=np.column_stack([theta,h,depth]);xi=np.vstack([xi,[[th,hh,dd] for th in [0.,1.2,2*np.pi] for hh in [lo,hi] for dd in [0.,1.]]]);start=time.perf_counter()
 p=m.xi_to_p(xi);xx=query_atlas_coordinates(a,p)[0];target=xi.astype(np.float32);err=xx-target;err[:,0]=np.angle(np.exp(1j*err[:,0]));rec={'asset_sha256':sha(A/f'atlas_{side}.npz'),'boundary_inverse_count':len(xi),'physical_output_dtype':str(p.dtype),'xi_roundtrip_max_error':np.max(abs(err),axis=0).tolist(),'wall_s':time.perf_counter()-start}
 centers=a.skin_vertices[a.skin_faces].astype(float).mean(1);cx=query_atlas_coordinates(a,centers)[0];rec['all_skin_face_centers']={'count':len(centers),'accepted':len(cx),'max_abs_d':float(abs(cx[:,2]).max())}
 old=np.load(ROOT/'shaft_roi_v2_before_boundary_fix'/f'atlas_{side}.npz',allow_pickle=True);new=np.load(A/f'atlas_{side}.npz',allow_pickle=True)
 rec['volume_arrays_bitwise_unchanged']={k:bool(np.array_equal(old[k],new[k])) for k in ['harmonic_vertices','harmonic_tets','harmonic_h','harmonic_theta','harmonic_d','volume_points','volume_xi','skin_vertices']}
 rec['attachments_bytes_unchanged']=sha(ROOT/'shaft_roi_v2_before_boundary_fix'/f'attachments_{side}.npz')==sha(A/f'attachments_{side}.npz')
 # Do not smuggle a geometric core through the valid inner-shell boundary.
 try:m.p_to_xi(a.core_points[len(a.core_points)//2]);rec['centerline_rejected']=False
 except ValueError:rec['centerline_rejected']=True
 out[side]=rec;print(side,'boundary and lineage complete',flush=True)
config=Path('/media/camp/EXT_DRIVE/Among_US/DMP_playground/configs/shaft_roi_v2_20261110.json')
out['actual_entrypoint_boundary']=[query_and_transport(config,xi=np.array([1.23,.73,d]),knee_deg=90.) for d in [0.,1.]]
(HERE/'harmonic_boundary_extended_final.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

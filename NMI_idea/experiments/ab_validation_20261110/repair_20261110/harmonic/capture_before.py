from pathlib import Path
import sys,json,time,numpy as np
REPO=Path('/media/camp/EXT_DRIVE/Among_US'); sys.path.insert(0,str(REPO/'src'))
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas,query_atlas_coordinates
out=Path(__file__).resolve().parent
a=load_leg_volume_atlas(REPO/'dataset/processed/anatomy_retarget/leg_volume_coordinates/production/atlas/atlas_right.npz')
ids=np.linspace(0,len(a.skin_vertices)-1,16,dtype=int)
nids=np.random.default_rng(20261110).choice(len(a.harmonic_vertices),96,replace=False)
q=np.concatenate([a.skin_vertices[ids],a.harmonic_vertices[nids]])
t=time.perf_counter();x,_=query_atlas_coordinates(a,q)
e=x[16:]-np.column_stack([a.harmonic_theta[nids],a.harmonic_h[nids],a.harmonic_d[nids]]);e[:,0]=(e[:,0]+np.pi)%(2*np.pi)-np.pi
r=dict(skin_ids=ids.tolist(),node_ids=nids.tolist(),skin_depth_abs_mean=float(abs(x[:16,2]).mean()),skin_depth_abs_max=float(abs(x[:16,2]).max()),node_coordinate_abs_error_max=np.max(abs(e),axis=0).tolist(),node_depth_abs_error_mean=float(abs(e[:,2]).mean()),wall_s=time.perf_counter()-t,skin_xi=x[:16].tolist())
(out/'before_metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

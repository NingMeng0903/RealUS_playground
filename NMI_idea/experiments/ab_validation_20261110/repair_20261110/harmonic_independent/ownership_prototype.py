from pathlib import Path
import json,numpy as np
from audit import sha
H=Path(__file__).resolve().parent
B=Path('/media/camp/EXT_DRIVE/Among_US/dataset/processed/anatomy_retarget/leg_volume_coordinates/bake/base_atlas')
C=Path('/media/camp/EXT_DRIVE/Among_US/outputs/anatomy_retarget/recovered_smplx_20261108/smpl_canonical_weights.npz')
c=np.load(C,allow_pickle=True);names=[str(n) for n in c['joint_names']];left=[i for i,n in enumerate(names) if n in ['left_hip','left_knee','left_ankle','left_foot']];right=[i for i,n in enumerate(names) if n in ['right_hip','right_knee','right_ankle','right_foot']]
rows=[]
for side in ['left','right']:
 p=B/f'atlas_{side}.npz';a=np.load(p,allow_pickle=True);D={k:a[k] for k in a.files};w=c['lbs_weights'][a['full_vertex_indices']];L=w[:,left].sum(1)[a['skin_faces']].mean(1);R=w[:,right].sum(1)[a['skin_faces']].mean(1);keep=L>=R if side=='left' else R>=L;D['skin_faces']=a['skin_faces'][keep];np.savez_compressed(H/f'ownership_{side}_skin.npz',**D);rows.append({'side':side,'input':str(p),'sha256':sha(p),'original_faces':len(keep),'kept_faces':int(sum(keep)),'removed_faces':np.flatnonzero(~keep).tolist(),'rule':'same leg chain LBS mean weight >= opposite leg chain weight per face; no vertex displacement; prototype only'})
(H/'ownership_rule_results.json').write_text(json.dumps(rows,indent=2))

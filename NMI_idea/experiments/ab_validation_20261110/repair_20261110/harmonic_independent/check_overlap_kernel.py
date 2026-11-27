from pathlib import Path
import json,subprocess,numpy as np
from audit import binfile,lp_witness
H=Path(__file__).resolve().parent
A=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]],float)
cases=[('face_touch_opposite_sides',np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,-1.]]),0),('same_side_face_fold',np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.5]]),1),('separated',A+[2,0,0],0),('strict_containment',A*.1+[.1,.1,.1],1),('identical',A.copy(),1)]
rows=[]
for name,B,expected in cases:
 V=np.vstack((A,B));T=np.array([[0,1,2,3],[4,5,6,7]],np.uint32);p=H/('check_'+name+'.bin');binfile(p,V,T,np.zeros(8));out=json.loads(subprocess.check_output([str(H/'tet_overlap'),str(p)],text=True));p.unlink();assert out['positive_interior_overlap_pairs']==expected,(name,out)
 if expected:
  w=lp_witness(V,T,[0,1]);assert w['success'] and w['interior_ball_radius_m']>1e-8
 rows.append({'case':name,'overlap':out['positive_interior_overlap_pairs']})
(H/'kernel_checks.json').write_text(json.dumps(rows,indent=2));print(rows)

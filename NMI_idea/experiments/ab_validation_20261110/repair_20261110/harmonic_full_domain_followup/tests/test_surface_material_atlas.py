from pathlib import Path
from types import SimpleNamespace
import sys,numpy as np,pytest
sys.path.insert(0,'/media/camp/EXT_DRIVE/Among_US/src')
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas

def mesh():
 vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,.1],[-1.,0.,0.],[0.,-1.,-.1]])
 faces=np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]],np.int32)
 a=SimpleNamespace(skin_vertices=vertices,skin_faces=faces,side='right',metadata={})
 return SurfaceMaterialAtlas.from_surface(a,np.ones((5,1)))

def test_face_interior_transition_and_fixed_pose_transport():
 m=mesh();b=np.array([.2,.3,.5]);face=0;chart,uv=m.address_from_face(face,b);p=m.decode(chart,uv)[0];other=next(int(x) for x in m.face_charts[face] if x!=chart);q,J=m.transition(chart,uv,other);np.testing.assert_allclose(m.decode(other,q)[0],p,atol=1.e-14)
 tf=np.eye(4)[None];tf[0,:3,3]=[.3,-.2,.7];posed=m.posed_vertices(tf)
 np.testing.assert_allclose(m.decode(other,q,vertices=posed)[0],p+[.3,-.2,.7],atol=1.e-14)
 back,B=m.transition(other,q,chart);np.testing.assert_allclose(back,uv,atol=1.e-14);np.testing.assert_allclose(B@J,np.eye(2),atol=1.e-14)

def test_shared_vertex_and_edges_do_not_false_reject_transition():
 m=mesh();center=0;target=3
 # Source decode may choose face0; target3 only covers faces1/2. The common
 # physical vertex is nevertheless a legal chart overlap.
 q,J=m.transition(center,[0.,0.],target);np.testing.assert_allclose(m.decode(target,q)[0],m.vertices[0],atol=1.e-14)
 assert len(m.transition_options(center,[0.,0.],target))==2
 for face in range(4):
  for bary in [[1.,0.,0.],[.5,.5,0.],[0.,1.,0.]]:
   source=int(m.face_charts[face,0]);target=int(m.face_charts[face,1]);s,u=m.address_from_face(face,bary,source);v,_=m.transition(s,u,target)
   np.testing.assert_allclose(m.decode(s,u)[0],m.decode(target,v)[0],atol=1.e-14)

def test_strict_surface_query_and_explicit_projection():
 m=mesh();p=np.array([[.2,.3,.03]]);a=m.query(p)[0];np.testing.assert_allclose(m.decode(a['chart_id'],a['coordinate_uv_m'])[0],p[0],atol=1.e-14)
 with pytest.raises(ValueError):m.query([[.2,.3,.04]])
 projected=m.query([[.2,.3,.04]],project=True)[0];assert projected['projection_distance_m']>0 and projected['projected']
 with pytest.raises(ValueError):m.decode(0,[100.,100.])
 with pytest.raises(ValueError):m.query([[np.nan,0.,0.]])


def test_single_triangle_boundary_fans_and_save_load(tmp_path):
 a=SimpleNamespace(skin_vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),skin_faces=np.array([[0,1,2]],np.int32),side='left',metadata={})
 m=SurfaceMaterialAtlas.from_surface(a,np.ones((3,1)));path=tmp_path/'surface.npz';m.save(path);n=SurfaceMaterialAtlas.load(path)
 for bary in [[1.,0.,0.],[0.,.5,.5],[.2,.3,.5]]:
  c,u=n.address_from_face(0,bary);np.testing.assert_allclose(n.decode(c,u)[0],np.array(bary)@a.skin_vertices,atol=1.e-14)

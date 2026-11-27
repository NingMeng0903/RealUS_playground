"""Independent bounded check of sealed d=0 multichart assets only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;REPAIR=HERE.parents[1];BASE=REPAIR/'harmonic_full_domain_followup'
sys.path.insert(0,'/media/camp/EXT_DRIVE/Among_US/src')
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.shaft_rebuild import SurfaceMaterialAtlas
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
from igl.copyleft.cgal import remesh_self_intersections

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((BASE/'manifest.json').read_text());checks=[]
def visit(x):
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x:
            checks.append({'path':x['path'],'matches':sha(x['path'])==x['sha256']})
        for y in x.values():visit(y)
    elif isinstance(x,list):
        for y in x:visit(y)
visit(manifest);assert all(x['matches'] for x in checks)
canonical=np.load('/media/camp/EXT_DRIVE/Among_US/outputs/anatomy_retarget/recovered_smplx_20261108/smpl_canonical_weights.npz')
out={'scope':'d=0 full retained skin, no deep volume or control-loop integration claim','manifest_entries_checked':len(checks),'manifest_all_match':True,'sides':{}}
rng=np.random.default_rng(619935)
for side in ['right','left']:
    m=SurfaceMaterialAtlas.load(BASE/f'contact_multichart_{side}.npz')
    source=load_leg_volume_atlas(REPAIR/f'harmonic/ownership_smooth/owned_smooth_{side}_skin.npz')
    w=np.load(REPAIR/f'harmonic/ownership_smooth/{side}_surface_lbs_weights.npz')['weights'].astype(float);w=w/w.sum(1)[:,None]
    identity={'vertices':np.array_equal(m.vertices,source.skin_vertices),'faces':np.array_equal(m.faces,source.skin_faces),'normalized_fixed_weights':np.array_equal(m.weights,w)}
    assert all(identity.values())
    faces=rng.choice(len(m.faces),512,replace=False);bary=rng.dirichlet(np.ones(3),len(faces));bary[:64]=[1.,0.,0.];bary[64:128]=[.5,.5,0.]
    addrs=[m.address_from_face(int(fi),b) for fi,b in zip(faces,bary)];points=np.einsum('ni,nij->nj',bary,m.vertices[m.faces[faces]].astype(float))
    roundtrip=max(np.linalg.norm(m.decode(cid,uv)[0]-p) for (cid,uv),p in zip(addrs,points))
    transition_error=cycle_error=0.;edge_options=0
    for fi,b,(cid,uv),p in zip(faces,bary,addrs,points):
        target=next(int(c) for c in m.face_charts[fi] if c!=cid)
        opts=m.transition_options(cid,uv,target);edge_options+=int(len(opts)>1)
        for op in opts:
            transition_error=max(transition_error,float(np.linalg.norm(m.decode(target,op['coordinate_uv_m'])[0]-p)))
            back=m.transition_options(target,op['coordinate_uv_m'],cid)
            matching=[o for o in back if o['source_face_id']==op['source_face_id']]
            assert matching
            cycle_error=max(cycle_error,float(np.max(abs(matching[0]['jacobian']@op['jacobian']-np.eye(2)))))
    queries=m.query(points);query_error=max(np.linalg.norm(m.decode(a['chart_id'],a['coordinate_uv_m'])[0]-p) for a,p in zip(queries,points))
    bad=[]
    for name,call in [('physical_outside',lambda:m.query([[100.,100.,100.]])),('uv_outside',lambda:m.decode(0,[100.,100.])),('NaN',lambda:m.query([[np.nan,0.,0.]])),('bad_chart',lambda:m.decode(-1,[0.,0.]))]:
        try:call()
        except ValueError:bad.append(name)
    assert len(bad)==4
    projected=m.query([[100.,100.,100.]],project=True)[0];assert projected['projected'] and projected['projection_distance_m']>1
    alluv=m.chart_uv;uvmat=np.moveaxis(alluv[:,1:]-alluv[:,:1],1,2);uvdet=np.linalg.det(uvmat)
    canonical_faces=m.vertices[m.faces].astype(float);canonical_double_area=np.linalg.norm(np.cross(canonical_faces[:,1]-canonical_faces[:,0],canonical_faces[:,2]-canonical_faces[:,0]),axis=1)
    poses=[]
    for degree in range(0,91,15):
        pose=np.zeros((55,3));pose[5 if side=='right' else 4,0]=np.deg2rad(degree)
        tf=joint_global_transforms(pose_axis_angle=pose.ravel(),rest_joints=canonical['rest_joints'],parents=canonical['parents'])@canonical['inverse_bind']
        pv=m.posed_vertices(tf);tri=pv[m.faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        tgt=np.einsum('ni,nij->nj',bary,pv[m.faces[faces]])
        derr=max(np.linalg.norm(m.decode(cid,uv,vertices=pv)[0]-p) for (cid,uv),p in zip(addrs,tgt))
        q=m.query(tgt[:64],vertices=pv);qerr=max(np.linalg.norm(m.decode(x['chart_id'],x['coordinate_uv_m'],vertices=pv)[0]-p) for x,p in zip(q,tgt[:64]))
        common=np.eye(4);common[:3,:3]=Rotation.from_rotvec([.3,-.15,.2]).as_matrix();common[:3,3]=[.2,.1,-.3]
        moved=m.posed_vertices(common[None]@tf);expected=pv@common[:3,:3].T+common[:3,3]
        se3=float(np.max(abs(moved-expected)));assert area.min()>0 and derr<1e-12 and qerr<1e-12 and se3<1e-12
        intersections=None
        if degree==90:
            intersections=len(remesh_self_intersections(pv.astype(float),m.faces.astype(np.int64),detect_only=True,first_only=False,cutoff=10000000)[2]);assert intersections==0
        poses.append({'knee_degrees':degree,'minimum_double_triangle_area_m2':float(area.min()),'minimum_area_ratio':float((area/canonical_double_area).min()),'decode_max_m':float(derr),'query_vertices_max_m':float(qerr),'common_SE3_max_abs':se3,'independently_rerun_CGAL_self_intersections':intersections})
    templates=np.load(BASE/f'{side}_template_prior_chart_addresses.npz');decoded=np.asarray([m.decode(int(c),uv)[0] for c,uv in zip(templates['chart_ids'],templates['coordinates_uv_m'])]);terr=float(np.linalg.norm(decoded-templates['canonical_reprojected_points'],axis=1).max());dist=np.linalg.norm(decoded-templates['source_skin_projected_points'],axis=1)
    q=m.query(templates['source_skin_projected_points'],project=True);reproj=np.asarray([m.decode(x['chart_id'],x['coordinate_uv_m'])[0] for x in q]);rerr=float(np.linalg.norm(reproj-decoded,axis=1).max())
    assert terr<1e-12 and rerr<1e-12
    out['sides'][side]={'identity':identity,'vertices':len(m.vertices),'faces':len(m.faces),'charts':len(m.chart_centers),'local_material_positive_area':bool(uvdet.min()>0),'minimum_uv_determinant_m2':float(uvdet.min()),'all_faces_have_three_charts':bool(np.all(m.face_charts>=0)),'random_and_boundary_samples':512,'roundtrip_max_m':float(roundtrip),'public_query_max_m':float(query_error),'transition_max_m':transition_error,'transition_inverse_jacobian_max_abs':cycle_error,'samples_exposing_multiple_face_derivatives':edge_options,'strict_rejections':bad,'poses':poses,'template_footprint_count':len(decoded),'template_decode_error_m':terr,'template_fresh_reprojection_error_m':rerr,'template_max_projection_distance_m':float(dist.max()),'deep_points_preserved_separately':bool('source_original_vessel_points' in templates.files)}
    print(side,'passed',flush=True)
(HERE/'contact_surface_independent.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

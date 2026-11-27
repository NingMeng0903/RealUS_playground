"""Independent bounded shaft ROI audit. Writes only to this audit directory."""
from pathlib import Path
from dataclasses import replace
import hashlib, json, struct, subprocess, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
REPAIR = HERE.parent
ASSETS = REPAIR / 'harmonic' / 'shaft_roi_v2'
SOURCE = Path('/media/camp/EXT_DRIVE/Among_US')
sys.path[:0] = [str(SOURCE), str(SOURCE/'src')]
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.atlas import load_leg_volume_atlas, query_atlas_coordinates
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.volume_refine import VolumeTetRefiner
from projects.genesis_ue_sync.anatomy_retarget.anatomy_lbs import joint_global_transforms
from projects.genesis_ue_sync.anatomy_retarget.leg_volume_coordinates.lbs_bridge import blend_lbs_transforms
from DMP_playground.core_algorithms.patch_binding import load_canonical_bundle
from DMP_playground.run_volume_coordinate_query import query_and_transport

tag = sys.argv[1] if len(sys.argv)>1 else 'fresh'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dets(v,t):
    e=np.asarray(v,dtype=np.float64)[t[:,1:]]-np.asarray(v,dtype=np.float64)[t[:,0,None]]
    return np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))
kernel = REPAIR/'harmonic_independent/tet_overlap'
def overlap(label,v,t):
    p=HERE/f'harmonic_{tag}_{label}.bin'
    v=np.ascontiguousarray(v,dtype='<f8');t=np.ascontiguousarray(t,dtype='<u4')
    with p.open('wb') as f:
        f.write(struct.pack('<QQ',len(v),len(t)));f.write(v.tobytes());f.write(t.tobytes());f.write(np.zeros(len(t),'<i4').tobytes())
    r=subprocess.run([str(kernel),str(p)],check=True,text=True,capture_output=True)
    result=json.loads(r.stdout);result['binary_input_sha256']=sha(p)
    p.unlink()
    return result
def wrapped_error(a,b):
    d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    d[:,0]=np.angle(np.exp(1j*d[:,0]))
    return np.max(abs(d),axis=0).tolist()
def rejected(fn):
    try: return dict(rejected=False,returned=np.asarray(fn()).tolist())
    except (ValueError,IndexError) as e:return dict(rejected=True,type=type(e).__name__,message=str(e))

manifest=json.loads((ASSETS/'asset_hashes.json').read_text())
canon=load_canonical_bundle(SOURCE/'outputs/anatomy_retarget/recovered_smplx_20261108')
out=dict(tag=tag,seed=594201,scope='finite offline fixed-LBS geometry and public point API; not servo, contact, orientation or ultrasound',
         source_hashes={str(SOURCE/'src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates'/n):sha(SOURCE/'src/projects/genesis_ue_sync/anatomy_retarget/leg_volume_coordinates'/n) for n in ['atlas.py','harmonic.py','shaft_rebuild.py','volume_refine.py','lbs_bridge.py']},
         kernel_sha256=sha(kernel),kernel_source_sha256=sha(kernel.with_suffix('.cpp')),sides={})
for side in ['left','right']:
    start=time.perf_counter();a=load_leg_volume_atlas(ASSETS/f'atlas_{side}.npz');att=np.load(ASSETS/f'attachments_{side}.npz')
    v=a.harmonic_vertices;t=a.harmonic_tets;W=att['volume_lbs_weights'];grid=np.arange(len(v)).reshape(tuple(att['grid_shape']))
    rec=dict(asset_hashes={n:dict(actual=sha(ASSETS/n),expected=manifest[n],match=sha(ASSETS/n)==manifest[n]) for n in [f'atlas_{side}.npz',f'attachments_{side}.npz']},metadata=a.metadata)
    D=dets(v,t);rec['raw']=dict(tets=len(t),nonpositive=int(sum(D<=0)),det_min=float(D.min()),overlap=overlap(side+'_raw',v,t))
    theta=a.harmonic_theta[t].astype(float);theta=theta[:,:1]+np.angle(np.exp(1j*(theta-theta[:,:1])))
    theta-=np.floor(theta.mean(1)/(2*np.pi))[:,None]*(2*np.pi)
    C=np.stack((theta/(2*np.pi),a.harmonic_h[t],a.harmonic_d[t]),axis=2).reshape(-1,3)
    CT=np.arange(len(C),dtype=np.uint32).reshape(-1,4)
    Cm=np.vstack((C-[1,0,0],C,C+[1,0,0]));Tm=np.vstack((CT,CT+len(C),CT+2*len(C)))
    rec['material']=dict(periodic_three_copy_overlap=overlap(side+'_material_periodic',Cm,Tm),
        max_theta_span=float(np.ptp(theta,axis=1).max()),exact_two_angular_values=bool(np.all(np.sum(np.diff(np.sort(theta,axis=1),axis=1)!=0,axis=1)==1)),
        determinant_min=float(dets(C,CT).min()),determinant_max=float(dets(C,CT).max()))
    del Cm,Tm,C,CT
    mapper=VolumeTetRefiner.from_atlas(a);rng=np.random.default_rng(594201);ids=rng.integers(0,len(t),1024);b=rng.dirichlet([2,2,2,2],len(ids));p=np.einsum('ni,nij->nj',b,v[t[ids]].astype(float))
    q0=time.perf_counter();xi=mapper.p_to_xi(p);q1=time.perf_counter();back=mapper.xi_to_p(xi);q2=time.perf_counter()
    rec['roundtrip']=dict(samples=len(p),reference_supplied=False,max_error_m=float(np.linalg.norm(back-p,axis=1).max()),forward_wall_s=q1-q0,inverse_wall_s=q2-q1,second_forward_coordinate_error=wrapped_error(mapper.p_to_xi(back),xi))
    q0=time.perf_counter()
    for point in p[:100]:query_atlas_coordinates(a,point)
    rec['warm_public_scalar_query_ms']=(time.perf_counter()-q0)*10
    lo,hi=float(a.harmonic_h.min()),float(a.harmonic_h.max())
    rec['outside_inverse']=[rejected(lambda x=x:mapper.xi_to_p(x)) for x in [[1,lo-.001,.5],[1,hi+.001,.5],[1,.5,-.001],[1,.5,1.001],[float('nan'),.5,.5]]]
    rec['outside_forward']=[rejected(lambda x=x:mapper.p_to_xi(x)) for x in [[5,5,5],a.core_points[len(a.core_points)//2],v[grid[0,4,0]]-.01*att['frames'][0,:,2],v[grid[-1,4,0]]+.01*att['frames'][-1,:,2]]]
    seam=np.array([[x,.5,.5] for x in [-2*np.pi,0,2*np.pi,4*np.pi]])
    seamP=mapper.xi_to_p(seam);rec['periodic_seam_max_m']=float(np.linalg.norm(seamP-seamP[1],axis=1).max())
    depth=a.harmonic_d.reshape(tuple(att['grid_shape']))
    rec['boundary_nodes']=dict(outer_max_abs_d=float(abs(depth[:,-1]).max()),inner_max_abs_d_minus1=float(abs(depth[:,0]-1).max()),max_radial_depth_difference=float(np.diff(depth,axis=1).max()),min_h_station_difference=float(np.diff(a.harmonic_h.reshape(tuple(att['grid_shape'])),axis=0).min()))
    rec['pose']={}
    for angle in [0,15,30,45,60,75,90]:
        pose=np.zeros((len(canon['joint_names']),3),np.float32);j=list(map(str,canon['joint_names'])).index(side+'_knee');pose[j,0]=np.deg2rad(angle)
        G=joint_global_transforms(pose_axis_angle=pose.ravel(),rest_joints=canon['rest_joints'],parents=canon['parents'])@canon['inverse_bind']
        B=blend_lbs_transforms(W,G)
        pv=np.einsum('nij,nj->ni',B,np.column_stack((v,np.ones(len(v),np.float32))))[:,:3].astype(np.float32)
        PD=dets(pv,t);pa=replace(a,harmonic_vertices=pv,skin_vertices=pv[grid[:,-1].ravel()])
        pp=np.einsum('ni,nij->nj',b[:64],pv[t[ids[:64]]].astype(float));pxi=query_atlas_coordinates(pa,pp)[0]
        rec['pose'][str(angle)]=dict(nonpositive=int(sum(PD<=0)),min_det_ratio=float((PD/D).min()),overlap=overlap(f'{side}_pose{angle}',pv,t),transport_error=wrapped_error(pxi,xi[:64]))
    # Common SE(3): independent fixed axis-angle Rodrigues construction.
    axis=np.array([.2,-.6,.7]);axis/=np.linalg.norm(axis);ang=.73;K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R=np.eye(3)+np.sin(ang)*K+(1-np.cos(ang))*(K@K);tr=np.array([.12,.21,-.08]);vv=(v@R.T+tr).astype(np.float32)
    ra=replace(a,harmonic_vertices=vv,skin_vertices=vv[grid[:,-1].ravel()],core_points=(a.core_points@R.T+tr).astype(np.float32))
    rp=np.einsum('ni,nij->nj',b[:64],vv[t[ids[:64]]].astype(float));rxi=query_atlas_coordinates(ra,rp)[0]
    rec['common_SE3_material_error']=wrapped_error(rxi,xi[:64]);rec['wall_s']=time.perf_counter()-start
    out['sides'][side]=rec
    (HERE/f'harmonic_{tag}_results.json').write_text(json.dumps(out,indent=2))
    print(side,'finished',rec['wall_s'],'seconds',flush=True)
config=SOURCE/'DMP_playground/configs/shaft_roi_v2_20261110.json'
out['entrypoint']=query_and_transport(config,xi=np.array([2.2,.63,.37]),knee_deg=90.)
out['entrypoint_outside']=rejected(lambda:query_and_transport(config,xi=np.array([1.,.95,.5])))
out['entrypoint_hash']=sha(SOURCE/'DMP_playground/run_volume_coordinate_query.py')
(HERE/f'harmonic_{tag}_results.json').write_text(json.dumps(out,indent=2))
print('complete',flush=True)

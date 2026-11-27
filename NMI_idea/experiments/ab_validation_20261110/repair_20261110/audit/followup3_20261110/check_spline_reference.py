"""Independent mathematical checks of actual frozen scan path references."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad
from scipy.spatial.transform import Rotation,Slerp

HERE=Path(__file__).resolve().parent;REPAIR=HERE.parents[1]
ROOT=Path('/media/camp/EXT_DRIVE/RealUS_playground')
sys.path.insert(0,str(ROOT/'rm75_control'))
from rm75_control.control.joint_admittance_8dof import scan_reference as source
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maxabs(x):return float(np.max(np.abs(x)))
inputs=np.load(REPAIR/'audit/phase2_native_subset_inputs.npz');sourcefile=Path(source.__file__)
(HERE/'source_snapshot'/sourcefile.name).write_bytes(sourcefile.read_bytes())
out={'source_sha256':sha(sourcefile),'input_sha256':sha(REPAIR/'audit/phase2_native_subset_inputs.npz'),'scope':'production reference mathematics and geometry only; no native run','paths':{}}
for path in (0,1):
    T=inputs[f'branch4_path{path}_T'];P=T[:,:3,3];R=Rotation.from_matrix(T[:,:3,:3])
    ref=source.WorldPoseSplineReference(P,rpy=R.as_euler('xyz'),speed_m_s=.02,ramp_s=.8)
    knots=ref.knots;duration=ref.duration_s();node_times=[0.]+[brentq(lambda t:ref.parameter_state(t)[0]-u,0,duration,xtol=1e-14) for u in knots[1:-1]]+[duration]
    times=np.unique(np.r_[np.linspace(1e-4,duration-1e-4,1025),node_times[1:-1],.8,duration-.8]);h=1e-6
    maxfd=np.zeros(6);wrongbody=0.;peak_lin=peak_ang=0.
    for t in times:
        minus,mid,plus=ref.sample(t-h),ref.sample(t),ref.sample(t+h)
        rm,rp=Rotation.from_euler('xyz',minus.pose_d[3:]),Rotation.from_euler('xyz',plus.pose_d[3:])
        worldomega=(rp*rm.inv()).as_rotvec()/(2*h);bodyomega=(rm.inv()*rp).as_rotvec()/(2*h)
        fd=np.r_[(plus.pose_d[:3]-minus.pose_d[:3])/(2*h),worldomega]
        maxfd=np.maximum(maxfd,abs(fd-mid.vel_ff));wrongbody=max(wrongbody,maxabs(bodyomega-mid.vel_ff[3:]))
        peak_lin=max(peak_lin,float(np.linalg.norm(mid.vel_ff[:3])));peak_ang=max(peak_ang,float(np.linalg.norm(mid.vel_ff[3:])))
    assert maxfd.max()<2e-8 and peak_lin<=.02+1e-12
    # Independent interval optimization of ||dp/du|| avoids the implementation's quartic-root method.
    extrema=[]
    for lo,hi in zip(knots[:-1],knots[1:]):
        grid=np.linspace(lo,hi,65);values=np.linalg.norm(ref.position_spline(grid,1),axis=1);candidates=[float(values[0]),float(values[-1])]
        for k in range(1,64):
            if values[k]>=values[k-1] and values[k]>=values[k+1]:
                optimum=minimize_scalar(lambda u:-float(np.linalg.norm(ref.position_spline(u,1))),bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-15})
                candidates.append(-float(optimum.fun))
        extrema.append(max(candidates))
    independent_max=max(extrema);assert abs(independent_max-ref.maximum_parameter_speed)<1e-9
    arc_segments=np.array([quad(lambda u:float(np.linalg.norm(ref.position_spline(u,1))),lo,hi,epsabs=1e-13,epsrel=1e-13)[0] for lo,hi in zip(knots[:-1],knots[1:])])
    ug=np.linspace(0,knots[-1],32769);smooth=ref.position_spline(ug);edges=np.diff(P,axis=0);length=np.linalg.norm(edges,axis=1)
    alpha=np.clip(np.einsum('nkj,kj->nk',smooth[:,None]-P[:-1],edges)/(length**2),0,1)
    projected=P[:-1]+alpha[:,:,None]*edges;nearest=np.linalg.norm(smooth[:,None]-projected,axis=2).min(1)
    seg=np.clip(np.searchsorted(knots,ug,side='right')-1,0,len(knots)-2);fraction=(ug-knots[seg])/length[seg]
    linear=P[seg]+fraction[:,None]*edges[seg];samelinear=np.linalg.norm(smooth-linear,axis=1)
    oldrot=Slerp(knots,R)(ug);rotdev=(oldrot.inv()*ref.rotation_spline(ug)).magnitude()
    nonzero=np.linalg.norm(ref.position_spline(ug,1),axis=1)
    transforms=Rotation.from_rotvec([.37,-.23,.41]);tr=np.array([.12,.27,-.16])
    moved=source.WorldPoseSplineReference(transforms.apply(P)+tr,rpy=(transforms*R).as_euler('xyz'),speed_m_s=.02,ramp_s=.8)
    se3p=se3r=se3v=0.
    for t in np.linspace(0,duration,129):
        x,y=ref.sample(t),moved.sample(t)
        se3p=max(se3p,maxabs(y.pose_d[:3]-(transforms.apply(x.pose_d[:3])+tr)))
        se3r=max(se3r,float((Rotation.from_euler('xyz',y.pose_d[3:]).inv()*transforms*Rotation.from_euler('xyz',x.pose_d[3:])).magnitude()))
        se3v=max(se3v,maxabs(y.vel_ff-np.r_[transforms.apply(x.vel_ff[:3]),transforms.apply(x.vel_ff[3:])]))
    assert max(se3p,se3r,se3v)<1e-10
    node_p=maxabs(ref.position_spline(knots)-P);node_r=float((R.inv()*ref.rotation_spline(knots)).magnitude().max())
    continuity={}
    for order in (1,2):
        pu=[];wu=[]
        for u in knots[1:-1]:
            dh=1e-10
            pu.append(maxabs(ref.position_spline(u-dh,order)-ref.position_spline(u+dh,order)))
            wu.append(maxabs(ref.rotation_spline(u-dh,order)-ref.rotation_spline(u+dh,order)))
        continuity[str(order)]={'position_parameter_derivative_left_right_max':max(pu),'local_rotation_parameter_derivative_left_right_max':max(wu),'offset_parameter_m':1e-10}
    for t in [-1,0,duration,duration+1]:assert np.array_equal(ref.sample(t).vel_ff,np.zeros(6))
    out['paths'][str(path)]={'duration_s':duration,'original_chord_length_m':float(knots[-1]),'spline_arc_length_m':float(arc_segments.sum()),'spline_arc_segments_m':arc_segments.tolist(),'node_times_s':node_times,'effective_ramp_s':ref.effective_ramp_s,'maximum_parameter_speed_implementation':ref.maximum_parameter_speed,'maximum_parameter_speed_independent_optimization':independent_max,'parameter_rate':ref.parameter_rate,'translation_peak_upper_bound_m_s':ref.parameter_rate*independent_max,'sampled_translation_peak_m_s':peak_lin,'sampled_angular_peak_rad_s':peak_ang,'all_six_FF_finite_difference_max_abs':maxfd.tolist(),'wrong_body_frame_omega_would_have_max_error_rad_s':wrongbody,'node_position_max_abs_m':node_p,'node_orientation_max_rad':node_r,'minimum_spatial_parameter_speed':float(nonzero.min()),'dense_geometry_samples':len(ug),'max_spline_to_old_polyline_distance_m':float(nearest.max()),'max_same_chord_parameter_position_difference_m':float(samelinear.max()),'max_same_chord_parameter_orientation_difference_deg':float(np.degrees(rotdev.max())),'SE3_position_max_abs_m':se3p,'SE3_rotation_max_rad':se3r,'SE3_FF_max_abs':se3v,'left_right_parameter_derivative_checks':continuity,'geometry_note':'Spline changes the candidate between exact pose nodes. Directed distance is a dense sampled value, not an interval-certified Hausdorff bound.'}
    print(path,json.dumps(out['paths'][str(path)]),flush=True)
(HERE/'spline_reference_independent.json').write_text(json.dumps(out,indent=2)+'\n')

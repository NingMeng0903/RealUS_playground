"""Analytic 6D velocity, interpolation, endpoint and frame contracts."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rm75_control.control.joint_admittance_8dof.scan_reference import WorldPoseSplineReference


def make_reference(**kwargs):
    points=np.array([[.4,.1,.2],[.41,.13,.205],[.405,.16,.21],[.42,.20,.2]])
    angles=np.array([[.2,-.3,.1],[.3,-.2,.15],[.4,.0,.2],[.35,.2,.25]])
    return WorldPoseSplineReference(points,rpy=angles,**kwargs)


def test_all_six_velocity_components_match_actual_pose_derivative():
    ref=make_reference()
    h=1e-6
    for t in np.linspace(.001,ref.duration_s()-.001,301):
        a,b,m=ref.sample(t-h),ref.sample(t+h),ref.sample(t)
        R0=Rotation.from_euler('xyz',a.pose_d[3:]).as_matrix()
        R1=Rotation.from_euler('xyz',b.pose_d[3:]).as_matrix()
        fd=np.r_[(b.pose_d[:3]-a.pose_d[:3])/(2*h),Rotation.from_matrix(R1@R0.T).as_rotvec()/(2*h)]
        np.testing.assert_allclose(m.vel_ff,fd,atol=2e-8,rtol=2e-6)


def test_original_pose_nodes_are_exact_and_spline_derivatives_are_continuous():
    ref=make_reference()
    np.testing.assert_allclose(ref.position_spline(ref.knots),ref.points,atol=1e-14)
    err=(Rotation.from_euler('xyz',ref.rpy).inv()*ref.rotation_spline(ref.knots)).magnitude()
    assert err.max()<1e-13
    for u in ref.knots[1:-1]:
        for order in (1,2):
            np.testing.assert_allclose(ref.position_spline(u-1e-10,order),ref.position_spline(u+1e-10,order),atol=2e-5)
            np.testing.assert_allclose(ref.rotation_spline(u-1e-10,order),ref.rotation_spline(u+1e-10,order),atol=2e-5)


def test_endpoint_hold_and_translation_speed_cap():
    ref=make_reference()
    for t in (-1.,0.,ref.duration_s(),ref.duration_s()+1):
        np.testing.assert_array_equal(ref.sample(t).vel_ff,np.zeros(6))
    samples=[ref.sample(t) for t in np.linspace(0.,ref.duration_s(),1001)]
    assert max(np.linalg.norm(x.vel_ff[:3]) for x in samples)<=.02+1e-12
    np.testing.assert_allclose(samples[-1].pose_d[:3],ref.points[-1],atol=1e-14)


def test_full_world_transform_rotates_both_geometric_velocity_blocks():
    ref=make_reference()
    g=Rotation.from_rotvec([.4,-.2,.3]); translation=np.array([.2,-.1,.7])
    moved=WorldPoseSplineReference(g.apply(ref.points)+translation,
        rpy=(g*Rotation.from_euler('xyz',ref.rpy)).as_euler('xyz'))
    assert moved.duration_s()==pytest.approx(ref.duration_s(),abs=1e-12)
    for t in np.linspace(0.,ref.duration_s(),51):
        a,b=ref.sample(t),moved.sample(t)
        np.testing.assert_allclose(b.pose_d[:3],g.apply(a.pose_d[:3])+translation,atol=2e-13)
        np.testing.assert_allclose(b.vel_ff[:3],g.apply(a.vel_ff[:3]),atol=2e-12)
        np.testing.assert_allclose(b.vel_ff[3:],g.apply(a.vel_ff[3:]),atol=2e-12)


def test_short_scan_has_no_overlapping_time_ramps_and_anchor_is_only_time():
    ref=WorldPoseSplineReference([[0,0,0],[.001,0,0]],rpy=np.zeros((2,3)),ramp_s=.8)
    assert ref.cruise_s==0.
    expected=[ref.sample(t) for t in np.linspace(0.,ref.duration_s(),11)]
    ref.set_origin(np.ones(6),t_s=10.)
    for t,a in zip(np.linspace(0.,ref.duration_s(),11),expected):
        b=ref.sample(t+10.)
        np.testing.assert_allclose(a.pose_d,b.pose_d,atol=1e-13)
        np.testing.assert_allclose(a.vel_ff,b.vel_ff,atol=1e-13)


@pytest.mark.parametrize('kwargs',[dict(speed_m_s=0.),dict(ramp_s=-1.),dict(soft_start=False)])
def test_invalid_scan_time_law_rejected(kwargs):
    with pytest.raises(ValueError):
        make_reference(**kwargs)

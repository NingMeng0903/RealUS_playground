"""Offline regression for the phantom's controller/viewer frame mix-up."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from peirastic.DEMO.phathom_scanning.cloud import cloud_in_rail_base, resolve_urdf
from peirastic.DEMO.phathom_scanning.s_scan import load_detect_pose, _lift_target
from rm75_control.control.joint_admittance_8dof.model import RobotKinematics
from rm75_control.control.joint_admittance_8dof.reference import WorldPolylineReference
from rm75_control.control.joint_admittance_8dof.viewer.orbbec_cloud import load_T_link7_cam
from peirastic.realman8dof.modes.cartesian import resolve_pose_q


@pytest.mark.parametrize("rail", [0.2, 0.533258, 0.8])
def test_cloud_uses_controller_fk_with_no_viewer_pedestal(rail):
    kin = RobotKinematics()
    q = np.asarray(load_detect_pose())
    q[0] = rail
    pose = kin.frame_pose(q, "link_7")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    transform = transform @ load_T_link7_cam()
    points = np.array([[0.02, -0.03, 0.45], [-0.1, 0.06, 0.6]])
    expected = points @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(cloud_in_rail_base(points, q), expected, atol=1e-9)
    assert resolve_urdf() == kin.urdf_path


def test_logged_standoff_after_frame_correction_has_valid_ptp_ik():
    kin = RobotKinematics()
    q = np.asarray(load_detect_pose())
    # Last standoff reference in run_20260908_034408.csv. The old cloud chain
    # added the viewer's 0.020 + 0.256 m pedestal; corrected motion moves down.
    goal = np.array([0.225456, 0.195403, 0.484475 - 0.276,
                     3.130747, -0.008726, -3.111275])
    ctx = SimpleNamespace(kin=kin, inner=SimpleNamespace(q_cmd=q), euler_order="xyz")
    result = resolve_pose_q(ctx, goal, require_path=False)
    actual = kin.fk_pose(result)
    assert goal[2] < kin.fk_pose(q)[2] - 0.15
    np.testing.assert_allclose(actual[:3], goal[:3], atol=0.001)
    error = Rotation.from_euler("xyz", actual[3:]).inv() * Rotation.from_euler("xyz", goal[3:])
    assert np.degrees(error.magnitude()) < 1.0


def test_lift_is_one_cm_from_measured_tcp_including_force_indentation():
    live = np.array([0.25, 0.21, 0.158, 3.05, -0.06, -3.1])
    target = _lift_target(live, 0.01)
    z_tool = Rotation.from_euler("xyz", live[3:]).as_matrix()[:, 2]
    np.testing.assert_allclose(target[:3] - live[:3], -0.01 * z_tool)
    np.testing.assert_allclose(target[3:], live[3:])
    assert target[2] > live[2]


def test_polyline_endpoint_has_no_forward_velocity():
    reference = WorldPolylineReference(
        np.array([[0.2, 0.2, 0.16], [0.2, 0.2, 0.17]]), speed_m_s=0.01)
    reference.set_origin(np.zeros(6), t_s=20)
    assert reference.sample(20.7).vel_ff[2] > 0
    sample = reference.sample(20 + reference.duration_s() + 0.1)
    np.testing.assert_allclose(sample.pose_d[:3], [0.2, 0.2, 0.17])
    np.testing.assert_allclose(sample.vel_ff, np.zeros(6))

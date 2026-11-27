"""Factory contracts without native processes or device access."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from scan_replay import ScanReplayBridge,QPReplayBridge
from rm75_control.control.joint_admittance_8dof.scan_reference import WorldPoseSplineReference


def factory():
    bridge=ScanReplayBridge.__new__(ScanReplayBridge)
    bridge.reference_options=dict(ramp_s=.8)
    return bridge


def test_stationary_pose_is_valid_hold():
    pose=np.tile(np.eye(4),(3,1,1))
    ref=factory().make_pose_reference(pose,.02,SimpleNamespace(euler_order='xyz'))
    np.testing.assert_array_equal(ref.sample(.5).vel_ff,np.zeros(6))


def test_internal_pure_rotation_cannot_be_silently_erased():
    poses=np.tile(np.eye(4),(3,1,1))
    poses[1,:3,:3]=Rotation.from_rotvec([0,0,.1]).as_matrix()
    with pytest.raises(ValueError,match='pure rotation'):
        factory().make_pose_reference(poses,.02,SimpleNamespace(euler_order='xyz'))


def test_scan_uses_local_reference_factory():
    poses=np.tile(np.eye(4),(3,1,1));poses[:,0,3]=[0.,.01,.02]
    ref=factory().make_pose_reference(poses,.02,SimpleNamespace(euler_order='xyz'))
    assert isinstance(ref,WorldPoseSplineReference)


def test_unsupported_profile_reference_is_rejected(tmp_path,monkeypatch):
    def no_runtime(self,*args,**kwargs):
        self.raw=dict(qpik=dict(hard_limits={}),inner=dict(qp={}))
    monkeypatch.setattr(QPReplayBridge,'__init__',no_runtime)
    source=dict(schema='offline_scan_tracking_v1',controller_overrides=dict(
        command_lead_arm_deg=20.,task_weight=[100]*3+[50]*3),reference=dict(kind='typo'))
    path=tmp_path/'profile.yaml';path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError,match='reference kind'):
        ScanReplayBridge(path)


def test_short_path_reports_no_middle_instead_of_empty_maximum():
    from run_scan_profile import middle_summary
    t=np.linspace(0.,.5,11)
    summary=middle_summary(np.zeros(11),np.zeros(11),t,.5,.8)
    assert summary==dict(samples=0,position_max_mm=None,angle_max_deg=None)

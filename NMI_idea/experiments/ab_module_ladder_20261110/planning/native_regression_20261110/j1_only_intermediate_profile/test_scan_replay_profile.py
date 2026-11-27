"""Factory contracts without native processes or device access."""
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
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


def test_scan_uses_model_j1_range_without_relaxing_motion_limits(monkeypatch):
    raw=dict(qpik=dict(hard_limits=dict(v_scale=.8,a_max_arm_rad_s2=3.)),
             inner=dict(qp=dict(j_max_arm_rad_s3=300.,j_max_rail_m_s3=320.,
                               branch_barrier=dict(j1_overfold_abs_rad=np.deg2rad(140.)))))
    def no_runtime(self,*args,**kwargs):
        self.raw=deepcopy(raw)
        self.assets=SimpleNamespace(kin=SimpleNamespace(
            q_lower=np.array([0.,-3.106]),q_upper=np.array([.8,3.106])))
    monkeypatch.setattr(QPReplayBridge,'__init__',no_runtime)
    bridge=ScanReplayBridge()
    assert bridge.raw['inner']['qp']['branch_barrier']['j1_overfold_abs_rad']==3.106
    assert bridge.raw['qpik']['hard_limits']['v_scale']==.8
    assert bridge.raw['qpik']['hard_limits']['a_max_arm_rad_s2']==3.
    assert bridge.raw['inner']['qp']['j_max_arm_rad_s3']==300.
    assert bridge.raw['inner']['qp']['j_max_rail_m_s3']==320.


def test_archived_profile_retains_its_configured_j1_policy(tmp_path,monkeypatch):
    def no_runtime(self,*args,**kwargs):
        self.raw=dict(qpik=dict(hard_limits={}),inner=dict(qp=dict(
            branch_barrier=dict(j1_overfold_abs_rad=np.deg2rad(140.)))))
    monkeypatch.setattr(QPReplayBridge,'__init__',no_runtime)
    profile=yaml.safe_load((Path(__file__).parent/'scan_tracking_profile.yaml').read_text())
    del profile['controller_overrides']['j1_overfold_limit']
    path=tmp_path/'legacy.yaml';path.write_text(yaml.safe_dump(profile))
    bridge=ScanReplayBridge(path)
    assert bridge.raw['inner']['qp']['branch_barrier']['j1_overfold_abs_rad']==np.deg2rad(140.)

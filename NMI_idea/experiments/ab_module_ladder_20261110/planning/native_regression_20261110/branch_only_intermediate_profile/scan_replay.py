"""Reusable offline scan profile; no process-global reference replacement."""
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'dird'))
from qp_replay import QPReplayBridge
from common import sha256
from rm75_control.control.joint_admittance_8dof import scan_reference


class ScanReplayBridge(QPReplayBridge):
    def __init__(self, profile_path=HERE/'scan_tracking_profile.yaml'):
        super().__init__('native','urdf',native_clock='deterministic')
        self.profile_path=Path(profile_path)
        self.profile=yaml.safe_load(self.profile_path.read_text())
        if self.profile.get('schema')!='offline_scan_tracking_v1':
            raise ValueError('unsupported scan profile')
        overrides=self.profile['controller_overrides']
        self.raw['qpik']['hard_limits']['command_lead_arm_deg']=float(overrides['command_lead_arm_deg'])
        self.raw['inner']['qp']['task_weight']=list(overrides['task_weight'])
        j1_policy=overrides.get('j1_overfold_limit','configured')
        if j1_policy not in ('configured','urdf'):
            raise ValueError('unsupported J1 overfold limit policy')
        if j1_policy=='urdf':
            # The generic posture preference used a140degree J1 wall even
            # when a legal scan starts beyond it. Keep the model joint limits
            # and their normal velocity damping, without that extra wall.
            kin=self.assets.kin
            limit=min(abs(float(kin.q_lower[1])),abs(float(kin.q_upper[1])))
            if not np.isfinite(limit) or limit<=0.:
                raise ValueError('J1 requires finite two-sided URDF limits')
            self.raw['inner']['qp'].setdefault('branch_barrier',{})['j1_overfold_abs_rad']=limit
        extra_limits=overrides.get('extra_posture_limits',True)
        if not isinstance(extra_limits,bool):
            raise ValueError('extra_posture_limits must be boolean')
        if not extra_limits:
            # Extra branch/sign/overfold policy is separate from real URDF,
            # speed/acceleration/jerk and collision constraints. The scan actor
            # requests only the latter plus normal secondary nullspace motion.
            self.raw['inner']['qp'].setdefault('branch_barrier',{})['enabled']=False
        self.reference_options=self.profile['reference']
        if self.reference_options.get('kind')!='WorldPoseSplineReference':
            raise ValueError('unsupported scan reference kind')

    def make_pose_reference(self,tcp,speed_m_s,cfg):
        angles=Rotation.from_matrix(tcp[:,:3,:3]).as_euler(cfg.euler_order)
        chord=np.linalg.norm(np.diff(tcp[:,:3,3],axis=0),axis=1)
        if not len(chord) or np.all(chord<=1e-10):
            # A persistent observation hold is stationary, not a new scan.
            rotation_gap=Rotation.from_matrix(tcp[0,:3,:3].T@tcp[:,:3,:3]).magnitude()
            if np.any(rotation_gap>1e-9):
                raise ValueError('pure rotation needs an explicitly timed reference')
            return super().make_pose_reference(tcp,0.,cfg)
        return scan_reference.WorldPoseSplineReference(tcp[:,:3,3],rpy=angles,
            speed_m_s=speed_m_s,ramp_s=float(self.reference_options['ramp_s']),
            euler_order=cfg.euler_order)

    def _run_pose_path(self,*args,**kwargs):
        out=super()._run_pose_path(*args,**kwargs)
        out['actual_reference_kind']='WorldPoseSplineReference (stationary observations use a hold)'
        out['actual_start_stop_ramp_s']=float(self.reference_options['ramp_s'])
        out['profile_sha256']=sha256(self.profile_path)
        return out

    def _state_fingerprint(self):
        out=super()._state_fingerprint()
        out.update(scan_profile_sha256=sha256(self.profile_path),
            scan_reference_sha256=sha256(scan_reference.__file__))
        return out

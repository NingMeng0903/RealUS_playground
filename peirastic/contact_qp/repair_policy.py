"""Versioned differential repair policy; no acoustic or force prediction claim.

Only the physical normal/rocking components enter the differential request.
The independent shortfall cost cannot be paid by common loading or alpha.
Energy admission is deliberately absent here: the QP's existing full-port row
continues to constrain every resulting six-dimensional command.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DifferentialRepairConfig:
    revision: str = "v8r2_bounded_episode"
    balance_deadband: float = 0.10
    max_permission_s: float = 2.0
    max_angle_travel_rad: float = math.radians(3.)
    healthy_frames: int = 3
    differential_weight: float = 10.0
    residual_progress_gain: float = 0.75
    loading_weight: float = 10.0
    nominal_force_n: float = 4.0
    soft_force_n: float = 4.5
    hard_stop_n: float = 6.0

    def __post_init__(self):
        if self.revision not in ('v8r2_bounded_episode','v8r3_confidence_balance'):raise ValueError('unsupported differential repair revision')
        if isinstance(self.balance_deadband,(bool,np.bool_)) or not math.isfinite(self.balance_deadband) or not 0 <= self.balance_deadband < 1:
            raise ValueError('balance_deadband must be finite in [0, 1)')
        from .repair_episode import RepairEpisode
        RepairEpisode(max_permission_s=self.max_permission_s,max_angle_travel_rad=self.max_angle_travel_rad,healthy_frames=self.healthy_frames)
        for name in ('differential_weight','residual_progress_gain','loading_weight'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not math.isfinite(value) or value<=0:
                raise ValueError(name+' must be finite and positive')
        if (self.nominal_force_n,self.soft_force_n,self.hard_stop_n)!=(4.,4.5,6.):
            raise ValueError('v8 policy fixes nominal/soft/supervisory force at 4/4.5/6 N')

    def force_gate(self,force_n):
        """Measured scalar scheduling, not an upper bound on future force."""
        if not math.isfinite(force_n):raise ValueError('finite measured force required')
        return float(np.clip((self.soft_force_n-force_n)/(self.soft_force_n-self.nominal_force_n),0.,1.))

    def confidence_imbalance(self,quality,gamma):
        """Coarse shallow-confidence direction cue, not an acoustic derivative.

        A common consistency multiplier cannot reverse the image direction.
        The deadband is an engineering setting requiring image calibration.
        """
        difference=float(quality[1]-quality[0])
        error=math.copysign(max(abs(difference)-self.balance_deadband,0.),difference)/(1.-self.balance_deadband)
        return error*float(np.min(gamma))

    def terms(self,*,scaled_basis,visual_rows,endpoint_rows,nominal_twist,deficits,
              repair_speed,normal_scale,force_n,alpha_preferred,progress_weight,
              differential_imbalance=None):
        """Eight normalized variables: base five, differential slack, two loads.

        Output matrices are additive costs and upper-bound inequalities. Added
        slacks are nonnegative. Differential slack is dimensionless; loading
        slacks represent endpoint speed divided by normal_scale.
        """
        gate=self.force_gate(force_n)
        imbalance=float(deficits[0]-deficits[1]) if differential_imbalance is None else float(differential_imbalance)
        sign=float(np.sign(imbalance))
        requested=repair_speed*gate*abs(imbalance)
        # Remove b(alpha): total normal/rocking contribution, not its nominal
        # increment. Common translation cancels for a calibrated flat aperture.
        row=np.asarray(visual_rows[0]-visual_rows[1],dtype=float).copy()
        row[[0,1,3,5]]=0.
        a=np.zeros(8);a[:3]=sign*(row @ scaled_basis)/normal_scale;a[2]=0.
        a[5]=1.
        rows=[-a];upper=[-requested/normal_scale]
        h=np.zeros((8,8));g=np.zeros(8)
        h[5,5]=self.differential_weight
        # Historical r2 couples alpha to residual. R3 keeps alpha independent,
        # so stopping path progress cannot manufacture a visual shortfall.
        if requested>0. and self.revision=='v8r2_bounded_episode':
            alpha=np.zeros(8);alpha[2]=1.;alpha[5]=self.residual_progress_gain
            h+=progress_weight*np.outer(alpha,alpha);h[2,2]-=progress_weight
            g-=progress_weight*alpha_preferred*alpha;g[2]+=progress_weight*alpha_preferred
        for i,endpoint in enumerate(endpoint_rows):
            load=np.zeros(8);load[:3]=(endpoint @ scaled_basis)/normal_scale
            load[6+i]=-1.
            rows.append(load);upper.append(float(endpoint @ nominal_twist)/normal_scale)
            h[6+i,6+i]=self.loading_weight*(1.-gate)
        return h,g,np.asarray(rows),np.asarray(upper),dict(
            repair_force_gate=gate,differential_sign=sign,differential_request_m_s=requested,
            differential_row=row,force_limit_assurance='measured_policy_not_prediction',
            soft_force_n=self.soft_force_n,hard_force_supervisor_n=self.hard_stop_n)

"""Visual task availability, independent of command transactions.

Continuous mode uses only current image/contact/force availability; it has no
attempt expiry or healthy-frame reset. Bounded mode retains its original finite
attempt bookkeeping. Neither mode admits energy or certifies physical angles.
"""
import math


class RepairEpisode:
    def __init__(self,*,max_permission_s=2.,max_angle_travel_rad=math.radians(3),healthy_frames=3,
                 permission_mode='bounded_episode'):
        if permission_mode not in ('bounded_episode','continuous'):
            raise ValueError('unsupported repair permission mode')
        self.permission_mode=permission_mode
        if not math.isfinite(max_permission_s) or max_permission_s<=0:
            raise ValueError('positive repair permission duration required')
        if not math.isfinite(max_angle_travel_rad) or max_angle_travel_rad<=0:
            raise ValueError('positive measured repair travel budget required')
        if type(healthy_frames) is not int or healthy_frames<1:raise ValueError('positive healthy-frame count required')
        self.max_permission_s=float(max_permission_s);self.max_angle_travel_rad=float(max_angle_travel_rad)
        self.healthy_frames=healthy_frames
        self.armed=False;self.permission_elapsed_s=0.;self.angle_travel_rad=0.
        self.last_time=None;self.last_angle=None;self.previous_permission=False
        self.last_frame=None;self.last_image_time=None;self.version=None;self.healthy_count=0;self.resets=0

    def update(self,*,now_s,measured_angle,observation,image_valid,c_min,force_gate,execution_enabled,angle_reference_reset=False,balance_deadband=None):
        now=float(now_s);angle=float(measured_angle)
        if not math.isfinite(now) or not math.isfinite(angle):
            self.healthy_count=0
            raise ValueError('finite repair measurement time/angle required')
        if self.last_time is not None and now==self.last_time and self.last_angle!=angle and not angle_reference_reset:
            self.healthy_count=0
            raise ValueError('conflicting repair angle at repeated measurement time')
        if self.last_time is not None and now<self.last_time:
            self.healthy_count=0
            raise ValueError('repair measurement clock reversed')
        if self.permission_mode=='continuous':
            self.last_time=now;self.last_angle=angle
            allowed=bool(image_valid and observation is not None and execution_enabled and force_gate>0.)
            reason=('contact_execution_not_enabled' if not execution_enabled else
                    'image_unavailable' if not image_valid or observation is None else
                    'force_gate_paused' if force_gate<=0. else 'continuous_visual_feedback')
            return dict(permission_mode='continuous',repair_allowed=allowed,armed=allowed,
                exhausted=False,reason=reason,angle_reference_reset=bool(angle_reference_reset),
                permission_elapsed_s=None,measured_angle_travel_rad=None,
                remaining_permission_s=None,remaining_angle_travel_rad=None,
                healthy_confirmation_count=0,healthy_resets=0,
                assurance='task_availability_not_execution_or_angle_certificate')
        if self.armed:
            if self.last_time is not None and self.previous_permission:
                self.permission_elapsed_s+=now-self.last_time
            if self.last_angle is not None and not angle_reference_reset:self.angle_travel_rad+=abs(angle-self.last_angle)
        self.last_time=now;self.last_angle=angle
        good=False;reason='image_unavailable'
        if image_valid and observation is not None:
            version=(observation.source_id,observation.version)
            frame=(version,observation.frame_seq)
            changed=self.version is not None and version!=self.version
            if changed:self.healthy_count=0
            self.version=version
            new_frame=(self.last_frame is None or changed or (observation.frame_seq>self.last_frame[1] and observation.effective_time_s>self.last_image_time))
            good=all(observation.quality[i]>=c_min for i in (0,2))
            if balance_deadband is not None:
                good=good and abs(float(observation.quality[2]-observation.quality[0]))<=balance_deadband
            if not good:self.healthy_count=0
            elif new_frame:self.healthy_count=min(self.healthy_frames,self.healthy_count+1)
            elif frame!=self.last_frame:self.healthy_count=0
            if new_frame:
                self.last_frame=frame;self.last_image_time=observation.effective_time_s
            if good and self.healthy_count>=self.healthy_frames and self.armed:
                self.armed=False;self.permission_elapsed_s=0.;self.angle_travel_rad=0.
                self.previous_permission=False;self.healthy_count=0;self.resets+=1
                reason='three_distinct_healthy_frames_reset'
            elif good:reason='awaiting_healthy_confirmation'
            else:reason='bad_window_or_imbalanced' if balance_deadband is not None else 'bad_window'
        else:self.healthy_count=0
        if image_valid and not good and execution_enabled and force_gate>0. and not self.armed:
            self.armed=True
        exhausted=(self.permission_elapsed_s>=self.max_permission_s-1e-12 or
                   self.angle_travel_rad>=self.max_angle_travel_rad-1e-12)
        allowed=bool(self.armed and image_valid and not good and execution_enabled and force_gate>0. and not exhausted)
        self.previous_permission=allowed
        if exhausted:
            reason='permission_time_exhausted' if self.permission_elapsed_s>=self.max_permission_s-1e-12 else 'measured_angle_travel_exhausted'
        elif not execution_enabled:reason='contact_execution_not_enabled'
        elif force_gate<=0.:reason='force_gate_paused'
        return dict(permission_mode='bounded_episode',angle_reference_reset=bool(angle_reference_reset),repair_allowed=allowed,armed=self.armed,exhausted=exhausted,reason=reason,
            permission_elapsed_s=self.permission_elapsed_s,measured_angle_travel_rad=self.angle_travel_rad,
            remaining_permission_s=max(0.,self.max_permission_s-self.permission_elapsed_s),
            remaining_angle_travel_rad=max(0.,self.max_angle_travel_rad-self.angle_travel_rad),
            healthy_confirmation_count=self.healthy_count,healthy_resets=self.resets,
            assurance='measurement_policy_not_execution_or_angle_certificate')

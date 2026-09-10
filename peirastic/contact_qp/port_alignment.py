"""Measured-only asynchronous arm/wrench/rail interval estimates.

No extrapolation, command velocities, or rail predictors. W and arm positions
share the original UDP source time; raw rail samples must bracket both ends.
Piecewise linear interpolation is an estimate, never a physical error bound.
"""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .energy import PortInterval
from .types import vector, positive


class MeasuredPortAligner:
    def __init__(self, *, calibration_version, max_source_interval_s=.02,
                 max_rail_interval_s=.02, max_wait_s=.05, max_samples=64,
                 euler_order='xyz'):
        self.calibration_version=str(calibration_version)
        if not self.calibration_version:raise ValueError('port calibration version required')
        self.max_source_interval_s=positive(max_source_interval_s,'source interval')
        self.max_rail_interval_s=positive(max_rail_interval_s,'rail interval')
        self.max_wait_s=positive(max_wait_s,'alignment wait')
        if type(max_samples) is not int or max_samples<3:raise ValueError('bounded history requires at least three samples')
        self.max_samples=max_samples;self.euler_order=euler_order
        self.rail=deque();self.pending=deque();self.previous=None;self.epoch=None
        self.last_now=-math.inf;self.events=[]

    def _gap(self,reason):
        self.events.append(dict(event='port_alignment_missing',reason=reason,certified=False))

    def _rail_fence(self, reason):
        self.rail.clear();self.pending.clear();self.previous=None
        self._gap(reason)

    def _rail_sample(self,feedback,now):
        if feedback is None or not bool(getattr(feedback,'valid',False)):
            self._rail_fence('missing_or_invalid_rail_sample')
            return
        try:
            t=float(feedback.sample_mono_s);q=float(feedback.position_m);seq=int(feedback.motion_seq)
            if not all(math.isfinite(x) for x in (t,q)) or t<=0 or not 0<=now-t<=self.max_wait_s:
                raise ValueError('rail time')
            if self.rail:
                old=self.rail[-1]
                if t==old[0] and seq==old[2] and q==old[1]:return
                if t<=old[0] or seq<=old[2]:
                    self.rail.clear();self.pending.clear();self.previous=None
                    raise ValueError('rail order or identity')
            self.rail.append((t,q,seq))
            while len(self.rail)>self.max_samples:self.rail.popleft()
        except (AttributeError,TypeError,ValueError,OverflowError):
            self._rail_fence('invalid_rail_sample')

    def _bracket(self,t):
        for left,right in zip(self.rail,list(self.rail)[1:]):
            if left[0]<=t<=right[0] and right[0]-left[0]<=self.max_rail_interval_s+1e-12:
                weight=(t-left[0])/(right[0]-left[0])
                return (1-weight)*left[1]+weight*right[1],dict(
                    source_ids=[left[2],right[2]],source_times_s=[left[0],right[0]],
                    weights=[1-weight,weight])
        return None

    def update(self, *, source_id, source_t_s, arm_q_rad, wrench_tcp,
               source_valid, rail_feedback, kin, now_s):
        """Return completed (PortInterval, provenance) pairs, each at most once."""
        now=float(now_s)
        if not math.isfinite(now) or now<self.last_now:
            self.pending.clear();self.previous=None;self._gap('clock_invalid');return []
        self.last_now=now
        epoch=(str(source_id),self.calibration_version)
        if epoch!=self.epoch:
            if self.epoch is not None:self._gap('source_or_calibration_epoch_changed')
            self.rail.clear();self.pending.clear();self.previous=None;self.epoch=epoch
        self._rail_sample(rail_feedback,now)
        if not self.rail:return []  # a known rail gap cannot seed a new common-port anchor
        try:
            t=float(source_t_s)
            if not source_valid or not source_id or not math.isfinite(t) or t<=0 or not 0<=now-t<=self.max_wait_s:
                raise ValueError('source invalid')
            q=vector(arm_q_rad,(7,));w=vector(wrench_tcp,(6,))
            current=(t,q,w,(str(source_id),t))
            old=self.previous
            if old is not None and t==old[0]:
                if not np.array_equal(q,old[1]) or not np.array_equal(w,old[2]):
                    raise ValueError('duplicate source changed')
            elif old is not None and t<old[0]:raise ValueError('source reversed')
            else:
                if old is not None:
                    if t-old[0]<=self.max_source_interval_s+1e-12:self.pending.append((old,current))
                    else:self._gap('source_measurement_gap')
                self.previous=current
            while len(self.pending)>self.max_samples:
                self.pending.popleft();self._gap('pending_history_overflow')
        except (TypeError,ValueError,OverflowError):
            self.previous=None;self.pending.clear();self._gap('invalid_source_sample');return []
        ready=[]
        while self.pending:
            first,last=self.pending[0];t0,t1=first[0],last[0]
            if now-t0>self.max_wait_s:
                self.pending.popleft();self._gap('rail_bracket_timeout');continue
            if self._bracket(t0) is None or self._bracket(t1) is None:break
            grid=[t0]+[r[0] for r in self.rail if t0<r[0]<t1]+[t1]
            pieces=[]
            try:
                for start,end in zip(grid[:-1],grid[1:]):
                    r0,r1=self._bracket(start),self._bracket(end)
                    if r0 is None or r1 is None:raise ValueError('rail bracket gap')
                    # Ensure the entire subinterval has one measured rail support.
                    if not any(a[0]<=start and b[0]>=end and b[0]-a[0]<=self.max_rail_interval_s+1e-12
                               for a,b in zip(self.rail,list(self.rail)[1:])):
                        raise ValueError('rail interpolation gap')
                    u0=(start-t0)/(t1-t0);u1=(end-t0)/(t1-t0)
                    delta=(last[1]-first[1]+np.pi)%(2*np.pi)-np.pi
                    q0=np.r_[r0[0],first[1]+u0*delta];q1=np.r_[r1[0],first[1]+u1*delta]
                    midpoint=.5*(q0+q1)
                    base=vector(np.asarray(kin.jacobian(midpoint)) @ ((q1-q0)/(end-start)),(6,))
                    rotation=Rotation.from_euler(self.euler_order,np.asarray(kin.fk_pose(midpoint))[3:6]).as_matrix()
                    velocity=np.r_[rotation.T @ base[:3],rotation.T @ base[3:]]
                    # W inputs are compensated at the same attached TCP in its
                    # body axes, not at different world-origin wrench points.
                    w0=(1-u0)*first[2]+u0*last[2];w1=(1-u1)*first[2]+u1*last[2]
                    interval=PortInterval(start,end,w0,w1,velocity,velocity,
                        calibration_version=self.calibration_version,valid=True,time_aligned=True)
                    provenance=dict(alignment_method='measured_bracket_interpolation',estimated=True,certified=False,
                        source_ids=[first[3],last[3]],source_times_s=[t0,t1],
                        source_endpoint_weights=[[1-u0,u0],[1-u1,u1]],rail_start=r0[1],rail_end=r1[1],
                        q_start=q0.tolist(),q_end=q1.tolist(),velocity_method='J_midpoint_measured_position_difference',
                        wrench_reference='compensated_environment_on_tool_at_attached_tcp_body_axes')
                    pieces.append((interval,provenance))
            except (TypeError,ValueError,OverflowError,RuntimeError):
                self._gap('kinematic_or_bracket_interval_invalid')
            else:ready.extend(pieces)
            self.pending.popleft()
        return ready

    def drain_events(self):
        events=self.events;self.events=[];return events

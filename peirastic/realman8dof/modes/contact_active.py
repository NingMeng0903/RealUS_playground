"""Publication-owned contact QP around the original ICRA TFF and IK interfaces.

Only the outer command changes. Final IK task residuals are recorded, while the
six-dimensional command energy budget is rechecked on the final payload model.
Neither successful transport nor this model establishes physical certification.
"""
from __future__ import annotations
from dataclasses import replace
import time
import uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput
from peirastic.contact_qp.types import ProbeGeometry,ContactStatus,REQUIRED_WINDOWS
from peirastic.contact_qp.geometry import motion_basis
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.runtime_source import SourceClock,SourceGapContext
from peirastic.contact_qp.execution import EXECUTION_POLICY,ProposalDeferred,QualityProgress,QualityIntervals
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from peirastic.contact_qp.runtime_config import validate_study_config,calibrated_geometry,source_settings
from peirastic.contact_qp.energy import EnergyLedger,PortBounds
from peirastic.contact_qp.runtime_energy import RuntimeEnergy
from .contact_recording import ContactRecordSink,FeatureReceiver,clock_metadata,study_fingerprints

_LATEST_IMAGE=object()

class _PrepareLaw:
    """Choose the existing law's transaction entry without copying its math."""
    def __init__(self,law):self.law=law;self.context={}
    def __getattr__(self,name):return getattr(self.law,name)
    def update(self,**kwargs):return self.law.prepare(**kwargs,**self.context)


class ContactQpOuter:
    contact_study_enabled=True
    contact_qp_enabled=True
    owns_reference_clock=True

    def __init__(self,baseline,config,*,sink=None,feature_receiver=None):
        validate_study_config(config)
        if config.get('mode')!='active':raise ValueError('active outer requires mode=active')
        self._continuous_execution=config.get('execution_policy')==EXECUTION_POLICY
        source=source_settings(config)
        self.source_clock=SourceClock(**source)
        if config.get('force_axis_monotonicity_confirmed') is not True:
            raise ValueError('declare the measured relation between original force axis and face loading')
        self.baseline=baseline;self.config=dict(config);self.controller=baseline.controller
        if self.controller.cfg.control_frame!='tool' or baseline.desired_force[2]!=4.:
            raise ValueError('active ICRA outer retains original +4 N tool-Z task')
        self.controller.configure_source_period(self.source_clock.period_s,
            variable_dt=self.source_clock.timebase=='variable_step_bilinear_v1')
        self.nominal=baseline.force_law
        self._preview=_PrepareLaw(self.nominal);baseline.force_law=self._preview
        self.contact_gate=getattr(baseline,'contact_gate',None)
        raw_reference=baseline.position.reference
        if self.contact_gate is not None:
            raw_reference=self.contact_gate.reference
            # The active adapter drives this gate once. Position uses an
            # explicit candidate override, so it never samples the gate twice.
            del baseline.contact_gate
        self.reference=FiniteIntervalReference(raw_reference,baseline.dt)
        self._quality_progress=(QualityProgress(float(getattr(raw_reference,"ramp",baseline.dt)),
            config["feature"]["c_min"]) if self._continuous_execution else None)
        geometry,_=calibrated_geometry(config)
        if config['geometry']['face_normal_convention']=='outward':
            # Preserve face X and convert the declared outward +Z into the
            # solver's into-contact +Z with a proper rigid rotation.
            geometry=replace(geometry,T_tcp_face=geometry.T_tcp_face @ np.diag([1.,-1.,-1.,1.]))
        self.geometry=geometry
        feature=config['feature'];self.feature_config=FeatureConfig(**feature['config'])
        self.registration_version=feature['registration_version']
        self._image_dropout_grace_s=float(feature.get('dropout_grace_s',0.))
        if not np.isfinite(self._image_dropout_grace_s) or self._image_dropout_grace_s<0:
            raise ValueError('image dropout grace must be finite and nonnegative')
        self._image_dropout_policy=feature.get('dropout_policy','stop')
        if self._image_dropout_policy not in ('stop','pause_visual'):
            raise ValueError('image dropout policy must be stop or pause_visual')
        self._image_seen_in_contact=False;self._image_unavailable_since_s=None
        self._image_last_notice_s=float('-inf');self._image_notice_active=False
        settings=dict(config.get('qp') or {})
        vmax=np.asarray(self.controller.cfg.max_velocity,dtype=float).copy()
        vmax[2]=min(vmax[2],self.controller.cfg.max_vz_tool_m_s)
        vmax[4]=min(vmax[4],self.nominal.tilt.cfg.vmax_rad_s)
        acceleration=np.asarray(self.controller.cfg.max_acceleration,dtype=float).copy()
        acceleration[4]=min(acceleration[4],self.nominal.tilt.cfg.a_max)
        settings.update(c_min=feature['c_min'],quality_policy_version=feature['quality_policy_version'],
                        lateral_windows=self.feature_config.lateral_windows,max_velocity=vmax,
                        max_acceleration=acceleration,angle_limit_rad=self.nominal.tilt.cfg.theta_max_rad)
        self.solver=ContactQp(QpConfig(**settings))
        tilt=self.nominal.tilt.cfg
        self._rocking=(RockingSmoothing(vmax[4],acceleration[4],tilt.mass,tilt.damping)
            if self._continuous_execution else None)
        self._pending_rocking=None;self._rocking_review_facts={}
        energy=config.get('energy') or {}
        enabled=config.get('energy_constraint_enabled',False)
        if type(enabled) is not bool:raise ValueError('energy_constraint_enabled must be boolean')
        self.energy=None
        self.command_budget=None
        if enabled and energy:
            from peirastic.contact_qp.command_budget import CommandBudget
            self.command_budget=CommandBudget(energy['initial_j'],energy['capacity_j'],energy['stopping_reserve_j'],
                settlement_port=energy.get('settlement_port'),wrench_convention=energy.get('wrench_convention'),
                max_command_interval_s=energy.get('max_command_interval_s'),constraint=energy.get('constraint'),
                task_power_source=energy.get('task_power_source','none'))
        elif energy:
            bounds=PortBounds(**dict(energy.get('measurement_bounds') or {}),verified=False)
            ledger=EnergyLedger(energy['initial_j'],energy['capacity_j'],energy['stopping_reserve_j'],bounds)
            self.energy=RuntimeEnergy(ledger,command_budget_enforced=False,
                                      max_measurement_age_s=energy.get('max_measurement_age_s',self.source_clock.max_age_s))
        elif enabled:raise ValueError('enabled command energy requires an explicit single-tank configuration')
        self._energy_parameters=dict(energy.get('constraint') or {})
        self._port_aligner=None
        if self.command_budget is not None:
            from peirastic.contact_qp.port_alignment import MeasuredPortAligner
            bounds=PortBounds(**dict(energy.get('measurement_bounds') or {}),verified=False)
            self._port_aligner=MeasuredPortAligner(calibration_version=bounds.calibration_version,
                max_source_interval_s=bounds.max_sample_interval_s,
                max_rail_interval_s=energy.get('max_rail_interval_s',bounds.max_sample_interval_s),
                max_wait_s=energy.get('max_measurement_age_s',self.source_clock.max_age_s),
                euler_order=self.controller.cfg.euler_order)
        if self.energy is not None:
            from peirastic.contact_qp.port_alignment import MeasuredPortAligner
            self.energy.bind_dissipation(self._energy_parameters)
            self._port_aligner=MeasuredPortAligner(
                calibration_version=self.energy.ledger.bounds.calibration_version,
                max_source_interval_s=self.energy.ledger.bounds.max_sample_interval_s,
                max_rail_interval_s=energy.get('max_rail_interval_s',self.energy.ledger.bounds.max_sample_interval_s),
                max_wait_s=self.energy.max_measurement_age_s,euler_order=self.controller.cfg.euler_order)
        self._physical_w_checked=config.get('physical_w_checked') is True
        self.sink=sink or ContactRecordSink(config.get('log_path') or
            Path('apps/logs/contact_qp')/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl'))
        self._quality_intervals=(QualityIntervals(self.solver.config.c_min,
            self.solver.config.differential_repair.balance_deadband,
            self.solver.config.max_image_age_s,self.sink.emit) if self._continuous_execution else None)
        self._endpoint_started_s=None;self._deferred_source_t_s=None
        self._reference_resume_after_s=None
        self.features=feature_receiver or FeatureReceiver(config['feature_endpoint'])
        self.reference_time_s=0.;self._control_time_s=0.;self._control_id=0;self.arrival_confirmed=False
        self._source_step=None;self._source_prepared=False;self._angle_origin=None
        self._last_velocity_time=None;self._previous=np.zeros(6);self._previous_rotation=None;self.pending_result=None
        self._pending_id=None;self._nominal_pending=False;self._reserved_id=None;self._stop_recorded=False
        self._dispatch_time_s=None;self._review_time_s=None
        self._publication_rejection_reason=None
        self._publication_retry_source_t_s=None;self._publication_retry_count=0
        self.last_path_twist=np.zeros(6);self.last_feedback_twist=np.zeros(6)
        self.sink.emit('study_start',mode='active',command_authority='outer_qp_original_ik',
            physical_port_assurance='unverified',task_certificate=False,config=config,
            **clock_metadata(),**study_fingerprints(baseline,config))
        if self.solver.config.differential_repair.permission_mode=='continuous':
            print(f"[CONTACT_QP] visual=continuous; image=required; "
                  f"image_loss={self._image_dropout_policy}; "
                  f"tank={energy['initial_j']:.3f}/{energy['capacity_j']:.3f} J; "
                  f"reserve={energy['stopping_reserve_j']:.3f} J; "
                  f"task_power={energy.get('task_power_source','none')}; "
                  f"fusion=shared_total_velocity",flush=True)
        if self._continuous_execution:
            print(f"[CONTACT_QP] execution={EXECUTION_POLICY}; clock=actual_success_interval; "
                  f"quality=deficit_only; alpha=0.25..1 slew=path_ramp; solver=bounded_retry_v1; "
                  f"source_gap={self.source_clock.gap_policy}; rocking={self._rocking.policy}; "
                  f"jerk={self._rocking.jerk_limit:.6g} rad/s^3",flush=True)

    def __getattr__(self,name):return getattr(self.baseline,name)
    @property
    def publication_owner(self):return self
    @property
    def pending_id(self):return self._pending_id

    def set_origin(self,pose0,*,t_s=None):
        self.baseline.set_origin(pose0,t_s=t_s)
        self.reference.set_origin(pose0,t_s=t_s or 0.)
        self.reference_time_s=self.reference.origin_s;self._control_time_s=self.reference_time_s
        self._angle_origin=Rotation.from_euler(self.controller.cfg.euler_order,np.asarray(pose0)[3:]).as_matrix()

    def begin_hybrid_episode(self,applied_twist_base,current_pose):
        self.baseline.begin_hybrid_episode(applied_twist_base,current_pose)
        if self._rocking is not None:
            self._rocking.seed(np.asarray(applied_twist_base)[3:],time.monotonic())

    def prepare_source(self,source_id,source_t_s,wall_time_ns,*,now_s):
        previous=self.source_clock.last
        context=None
        budget=self.command_budget
        if self._continuous_execution and previous is not None:
            # A currently old sample is never supplied to either filter. The
            # runner may wait inside the unchanged old dual-device lease.
            if (source_id==previous.source_id and np.isfinite(source_t_s) and
                    source_t_s>=previous.source_t_s and np.isfinite(now_s) and
                    type(wall_time_ns) is int and wall_time_ns>=0 and
                    (source_t_s!=previous.source_t_s or wall_time_ns==previous.source_wall_time_ns) and
                    now_s-source_t_s>=self.source_clock.max_age_s):
                self._publication_rejection_reason='awaiting_fresh_force'
                self._deferred_source_t_s=source_t_s
                raise ProposalDeferred('awaiting_fresh_force')
            if (source_id==previous.source_id and source_t_s-previous.source_t_s>
                    self.source_clock.max_interval_s and budget is not None and
                    budget.active is not None and not budget.latched_reason and
                    budget.pending is None and not budget.started):
                old=budget.active
                context=SourceGapContext(source_id,previous.source_t_s,source_t_s,
                    old.command_id,old.committed_s,old.expires_s,now_s,budget.max_command_interval_s)
        self._source_step=self.source_clock.observe(source_id,source_t_s,wall_time_ns,
            now_s=now_s,gap_context=context)
        self._source_prepared=True
        if self._source_step.gap_recovered:
            self.sink.emit('source_gap_recovered',source=self._source_step.__dict__,
                           filter_policy='exact_foh_gap_v1',reference_catch_up=False)
        return self._source_step

    def rocking_constraints(self):
        if self._rocking is None:return {}
        axis,bounds,facts=self._rocking.preview(self.pending_rotation_base_tcp,
            time.monotonic(),self.pending_actual_dt)
        self._pending_rocking=(axis,bounds,facts)
        return dict(rocking_axis_base=axis,rocking_bounds=bounds)

    def observe_measured_port(self, snap, raw_rail_feedback, wrench_tcp, kin, *, now_s, source_id):
        """Measurement-only ingress, independent of the current command proposal."""
        if self._port_aligner is None:return
        command_budget=self.__dict__.get('command_budget')
        if command_budget is not None:wrench_tcp=command_budget.wrench_from_control(wrench_tcp)
        intervals=self._port_aligner.update(source_id=source_id,
            source_t_s=getattr(snap,'t_s',float('nan')),
            arm_q_rad=np.deg2rad(np.asarray(getattr(snap,'q_deg',[]),dtype=float)),
            wrench_tcp=wrench_tcp,source_valid=bool(getattr(snap,'ok',False)),
            rail_feedback=raw_rail_feedback,kin=kin,now_s=now_s)
        for interval,provenance in intervals:
            if command_budget is not None:
                self.sink.emit('nonspendable_measured_port',start_s=interval.start_s,end_s=interval.end_s,
                    wrench_start=interval.wrench_start,wrench_end=interval.wrench_end,
                    velocity_start=interval.velocity_start,velocity_end=interval.velocity_end,
                    provenance=provenance,settlement_port='diagnostic_only',
                    physical_w_checked=self._physical_w_checked,physical_certified=False,
                    wrench_convention=command_budget.wrench_convention,budget_balance_changed=False)
            else:
                self.energy.observe_interval(interval,now_s=now_s,
                    physical_w_checked=self._physical_w_checked,provenance=provenance)
        for event in self._port_aligner.drain_events():
            self.sink.emit('port_alignment',facts=event)
        if self.energy is None:return
        events=self.energy.drain_events()
        if events['runtime'] or events['ledger']:
            self.sink.emit('energy_measurement',**events)

    def _drain_command_energy(self):
        if self.command_budget is not None:
            events=self.command_budget.drain_events()
            if events:self.sink.emit('logical_command_energy',events=events,**self.command_budget.facts)

    def _image_observation(self,now,*,contact_enabled,observation=_LATEST_IMAGE):
        """Unavailable feedback removes the visual task without reusing its image."""
        if observation is _LATEST_IMAGE:observation=self.features.observation
        expected=(self.registration_version,self.feature_config.window_version,self.feature_config.calibration_version)
        age=None if observation is None else now-observation.effective_time_s
        reason=('missing' if observation is None else
                'version_mismatch' if observation.version!=expected else
                'future_or_invalid_time' if not (np.isfinite(age) and
                    np.isfinite(observation.received_time_s) and observation.received_time_s<=now and age>=0) else
                'stale' if age>self.solver.config.max_image_age_s else
                'invalid_required_windows' if not observation.valid[list(REQUIRED_WINDOWS)].all() else 'ok')
        if reason=='ok':
            if contact_enabled and observation.fresh(now,self.solver.config.max_image_age_s):
                self._image_seen_in_contact=True
            if self._image_unavailable_since_s is not None:
                self.sink.emit('image_feedback_recovered',effective_age_s=age,
                    unavailable_s=now-self._image_unavailable_since_s)
                if self._image_notice_active:
                    print(f"[CONTACT_QP] visual resumed; image age={age*1000:.1f} ms",flush=True)
                    self._image_notice_active=False
            self._image_unavailable_since_s=None
            return observation,reason
        required=self.config['feature'].get('required',False) and contact_enabled
        if required:
            eligible=(reason in ('missing','stale','invalid_required_windows') and self._image_seen_in_contact
                      and (self._image_dropout_policy=='pause_visual' or self._image_dropout_grace_s>0))
            if eligible:
                if self._image_unavailable_since_s is None:
                    self._image_unavailable_since_s=now
                    self.sink.emit('image_feedback_unavailable',reason=reason,effective_age_s=age,
                        grace_s=self._image_dropout_grace_s,policy=self._image_dropout_policy,
                        visual_task_enabled=False,scan_quality_degraded=True)
                    if now-self._image_last_notice_s>=1.:
                        age_label='missing' if age is None else f'{age*1000:.1f} ms'
                        print(f"[CONTACT_QP] visual paused; {reason}; image age={age_label}; fresh-force task continues",flush=True)
                        self._image_last_notice_s=now;self._image_notice_active=True
                elapsed=now-self._image_unavailable_since_s
                if elapsed>=0 and (self._image_dropout_policy=='pause_visual' or elapsed<self._image_dropout_grace_s):
                    return None,'transient_'+reason
            self.sink.emit('required_image_feedback_rejected',reason=reason,effective_age_s=age,
                unavailable_s=None if self._image_unavailable_since_s is None else now-self._image_unavailable_since_s,
                expected_version=expected,received_feature=None if observation is None else observation.to_dict(),
                receiver_error=getattr(self.features,'error',None))
            raise RuntimeError('confidence feedback '+reason+'; expected registration '+self.registration_version+
                ', received '+('none' if observation is None else observation.registration_version)+
                '; run scripts/run_icra_tank.sh check')
        return None,reason

    def retry_unsent_publication(self):
        """Allow fresh recomputation inside an unchanged, committed old lease.

        The runner calls this only after aborting rail, outer and inner proposals.
        No deadline, watchdog heartbeat, source timestamp or command is renewed.
        """
        budget=self.command_budget
        retry_reasons={'wrench_source_expired','dispatch_source_expired'}
        if self._continuous_execution:retry_reasons.update(('awaiting_fresh_force','solver_deadline_exceeded','solver_attempts_exhausted','native_timeout'))
        if (self._publication_rejection_reason not in retry_reasons
                or self._nominal_pending
                or self._dispatch_time_s is not None or budget is None
                or budget.pending is not None or budget.started):
            return False
        now=time.monotonic()
        if not np.isfinite(now) or (budget.last_time_s is not None and now<=budget.last_time_s):
            return False
        if not budget.advance(now) or budget.active is None or now>=budget.active.expires_s:
            return False
        self._publication_retry_count+=1
        self._publication_retry_source_t_s=(self._deferred_source_t_s if self._deferred_source_t_s is not None
            else self._source_step.source_t_s)
        self._deferred_source_t_s=None
        self.sink.emit('publication_fresh_retry',reason=self._publication_rejection_reason,
            rejected_source_time_s=self._publication_retry_source_t_s,
            retained_command_id=budget.active.command_id,retained_expiry_s=budget.active.expires_s,
            retry_count=self._publication_retry_count,definitely_not_sent=True)
        self._drain_command_energy()
        return True

    def waiting_for_retry_source(self,source_t_s,*,now_s):
        """No new proposal or watchdog heartbeat until the rejected source changes."""
        watermark=self._publication_retry_source_t_s
        if watermark is None:return False
        budget=self.command_budget
        if budget is None or budget.active is None or budget.latched_reason:
            raise RuntimeError('fresh publication retry has no valid committed lease')
        if not np.isfinite(now_s) or now_s<budget.last_time_s:
            raise RuntimeError('fresh publication retry clock invalid')
        if now_s>budget.last_time_s and not budget.advance(now_s):
            raise RuntimeError('fresh publication retry lease expired: '+str(budget.latched_reason))
        if budget.active is None or now_s>=budget.active.expires_s:
            raise RuntimeError('fresh publication retry lease expired')
        if not np.isfinite(source_t_s) or source_t_s<watermark or source_t_s>now_s:
            raise RuntimeError('fresh publication retry source reversed or invalid')
        self._drain_command_energy()
        waiting=source_t_s==watermark
        if waiting and self._continuous_execution:self._reference_resume_after_s=now_s
        return waiting

    def sample(self,t_s,current_pose,f_ext,*,contact=None,f_ext_raw=None,dt_actual=None,
               sensor_age_s=None,feedback_age_s=None,feedback_fresh_tick=None,
               feedback_velocity_valid=None,v_tcp_z_actual=None,slack_norm=None,
               measured_twist_base=None,measured_twist_valid=None,measured_twist_fresh=None,
               measured_twist_metadata=None,measurement_time_s=None,measurement_id=None,
               wrench_source_time_s=None,wrench_source_wall_time_ns=None,wrench_source_id=None):
        if self._nominal_pending:raise RuntimeError('previous outer proposal remains unresolved')
        # Read the immutable receiver snapshot before its decision timestamp.
        # A frame arriving during nominal computation belongs to the next tick.
        image_snapshot=self.features.observation
        now=time.monotonic();actual_dt=float(self.baseline.dt if dt_actual is None else dt_actual)
        if not np.isfinite(actual_dt) or actual_dt <= 0: raise ValueError("invalid control interval")
        dt=min(actual_dt,float(self.baseline.dt))  # future command-budget hold model
        reference_dt=actual_dt if self._continuous_execution else dt
        if self._continuous_execution and self._reference_resume_after_s is not None:
            reference_dt=min(reference_dt,max(0.,now-self._reference_resume_after_s))
            self._reference_resume_after_s=None
        if not self._source_prepared:
            self.prepare_source(wrench_source_id,wrench_source_time_s,wrench_source_wall_time_ns,now_s=now)
        source=self._source_step;self._source_prepared=False
        source_expired=now-source.source_t_s>=self.source_clock.max_age_s
        if source_expired and not self._continuous_execution:raise RuntimeError('source hold age expired')
        if self._publication_retry_source_t_s is not None:
            if not source.fresh or source.source_t_s<=self._publication_retry_source_t_s:
                raise RuntimeError('fresh publication retry requires a newer force source')
            self._publication_retry_source_t_s=None
        self._control_id+=1;self._pending_id=self._control_id;self._control_time_s+=actual_dt
        self._dispatch_time_s=None;self._review_time_s=None
        self._publication_rejection_reason=None
        self.pending_dt=dt
        self.pending_reference_dt=reference_dt;self.pending_actual_dt=actual_dt
        rotation=Rotation.from_euler(self.controller.cfg.euler_order,np.asarray(current_pose)[3:]).as_matrix()
        self.pending_rotation_base_tcp=rotation.copy()
        # This history is an accepted OUTER command, not the rail-compensated
        # payload model or measured motion. Compare components in one frame.
        previous=self._previous.copy()
        if self._previous_rotation is not None:
            old_to_current=rotation.T @ self._previous_rotation
            previous=np.r_[old_to_current @ previous[:3],old_to_current @ previous[3:]]
        metadata=dict(measured_twist_metadata or {})
        velocity=None
        if measured_twist_valid and measured_twist_base is not None:
            velocity=np.r_[rotation.T @ np.asarray(measured_twist_base)[:3],rotation.T @ np.asarray(measured_twist_base)[3:]]
        velocity_fresh=bool(measured_twist_fresh and measurement_time_s is not None and
                            (self._last_velocity_time is None or measurement_time_s>self._last_velocity_time))
        velocity_dt=None if self._last_velocity_time is None or not velocity_fresh else measurement_time_s-self._last_velocity_time
        if velocity_fresh:self._last_velocity_time=measurement_time_s
        if self.contact_gate is not None:
            self.contact_gate.guard_approach(current_pose)
            if source.fresh:self.contact_gate.observe_force(float(f_ext[2]),source.source_t_s,valid=True)
            self.contact_gate.sample(self.reference_time_s)
        ref,h_ref=self.reference.candidate(self.reference_time_s,reference_dt if reference_dt>0 else dt)
        if reference_dt<=0:h_ref=0.
        if self.contact_gate is not None and not self.contact_gate.started:
            ref=replace(ref,vel_ff=np.zeros(6));h_ref=0.
        self.pending_h_ref=h_ref
        self.baseline.position.set_reference_override(ref)
        self._preview.context=dict(control_step_id=self._control_id,source_sample_id=source.sample_id,
            source_t_s=source.source_t_s,measurement_fresh=source.fresh,source_dt_s=source.source_dt_s,
            velocity_measurement_fresh=velocity_fresh,velocity_dt_s=velocity_dt)
        try:
            if (self.solver.config.allocation_policy=='differential_repair_v8' and source.fresh):
                original_scalar=float(f_ext[2])
                raw_scalar=original_scalar if f_ext_raw is None else float(np.asarray(f_ext_raw)[2])
                if max(original_scalar,raw_scalar)>=self.solver.config.differential_repair.hard_stop_n:
                    self.sink.emit('force_supervisor_stop',control_id=self._control_id,
                        filtered_original_force_n=original_scalar,raw_original_force_n=raw_scalar,
                        threshold_n=6.,source=source.__dict__,
                        assurance='fresh_measurement_stop_not_inter_sample_peak_guarantee')
                    raise RuntimeError('v8 fresh force supervisor reached 6 N; original stop chain required')
            nominal=np.asarray(self.baseline.sample(self._control_time_s,current_pose,f_ext,contact=contact,
                f_ext_raw=f_ext_raw,dt_actual=actual_dt,sensor_age_s=source.age_s,feedback_age_s=feedback_age_s,
                feedback_fresh_tick=feedback_fresh_tick,feedback_velocity_valid=velocity is not None,
                v_tcp_z_actual=None if velocity is None else float(velocity[2]),slack_norm=slack_norm))
            self._nominal_pending=True
            # Consume the accepted source's real LP/HP measurement once even
            # if computation used up its remaining send age. Commands roll
            # back; measurement history does not.
            if self.nominal.tilt.needs_normal_retract:
                raise RuntimeError('mechanical recovery requested; leave ordinary visual QP')
            if bool(getattr(self.controller,'shield_uncertified_brake',False)):
                raise RuntimeError('uncertified_brake')
            if source_expired:
                self._publication_rejection_reason='wrench_source_expired'
                raise ProposalDeferred('wrench_source_expired')
            angle_reference_reset=bool(self.controller.physical_contact_acquire_event or self._angle_origin is None)
            if angle_reference_reset:
                self._angle_origin=rotation.copy()
            measured_angle=float(Rotation.from_matrix(self._angle_origin.T @ rotation).as_rotvec()[1])
            path=nominal.copy();path[[2,4]]=0.;self.pending_basis=motion_basis(path)
            observation,image_reason=self._image_observation(now,
                contact_enabled=self.contact_gate is None or self.contact_gate.started,
                observation=image_snapshot)
            image_valid=observation is not None
            self._pending_alpha_target=None;self._pending_alpha_preferred=None
            if self._quality_progress is not None:
                self._pending_alpha_target,self._pending_alpha_preferred=self._quality_progress.preview(observation,actual_dt)
            self.pending_wrench_environment=None if f_ext_raw is None else np.asarray(f_ext_raw,dtype=float).copy()
            energy_snapshot=None
            if self.command_budget is not None:
                energy_snapshot=self.command_budget.snapshot(now_s=now,wrench_control_raw=f_ext_raw,
                    rotation_base_tcp=self.pending_rotation_base_tcp,
                    **({'nominal_twist_tool':nominal} if self.config.get('energy',{}).get('task_power_source','none')=='nominal_command' else {}))
            elif self.energy is not None:
                # Completed measured intervals enter through observe_measured_port;
                # this mixed-time diagnostic twist is never relabeled as aligned.
                energy_snapshot=self.energy.snapshot(now_s=now,hold_s=dt,
                    wrench_environment=self.pending_wrench_environment,source_t_s=source.source_t_s,
                    **self._energy_parameters)
            deadline=source.source_t_s+self.source_clock.max_age_s
            if self.command_budget is not None and self.command_budget.active is not None:
                deadline=min(deadline,self.command_budget.active.expires_s)
            self.pending_result=self.solver.solve(QpInput(self.geometry,nominal,path,
                float(np.sign(self.baseline.desired_force[2])*f_ext[2]),dt,now,
                observation=observation,previous_twist=previous,measured_angle=measured_angle,energy=energy_snapshot,
                acceleration_dt_s=actual_dt,
                repair_execution_enabled=bool(self.controller.contact_present) and (self.contact_gate is None or bool(self.contact_gate.started)),
                repair_angle_reference_reset=angle_reference_reset,alpha_preferred=self._pending_alpha_preferred),
                **(dict(deadline_s=deadline,online=True) if self._continuous_execution else {}))
            if self.pending_result.qp_twist is None:
                self.sink.emit('qp_prepare_rejected',control_id=self._control_id,
                    nominal_twist_tool=nominal,path_twist_tool=path,previous_twist_tool=previous,
                    force_n=float(f_ext[2]),control_actual_dt_s=actual_dt,command_hold_model_s=dt,
                    measured_angle_rad=measured_angle,diagnostics=dict(self.pending_result.diagnostics))
                if self._continuous_execution and self.pending_result.status==ContactStatus.DEFERRED:
                    self._publication_rejection_reason=str(self.pending_result.diagnostics.get('reason'))
                    raise ProposalDeferred(self._publication_rejection_reason)
                raise RuntimeError('outer QP: '+str(self.pending_result.status.value)+': '+str(self.pending_result.diagnostics.get('reason')))
            if self._quality_intervals is not None:
                diagnostic=self.pending_result.diagnostics
                reasons=[]
                request=float(diagnostic.get('differential_request_m_s',0.))
                if request<=0:reasons.append('request_zero_or_deadband')
                if diagnostic.get('differential_shortfall_m_s',0.)>self.solver.config.feasibility_tolerance:
                    reasons.append('outer_repair_shortfall')
                reasons.extend('outer_'+str(label) for label in diagnostic.get('active_hard_rows',())
                    if 'omega' in str(label) or 'angle' in str(label) or 'acceleration_4' in str(label) or 'velocity_4' in str(label))
                if diagnostic.get('energy_margin_power_w',float('inf'))<=self.solver.config.feasibility_tolerance:
                    reasons.append('command_energy_limit')
                if diagnostic.get('repair_force_gate',1.)<1.:reasons.append('force_gate')
                if self.pending_result.alpha+1e-8<self._pending_alpha_preferred:reasons.append('outer_mechanical_or_energy_progress_limit')
                self._quality_intervals.observe(observation,now,reasons)
            self.last_path_twist=self.pending_result.alpha*self.baseline.last_path_twist
            self.last_feedback_twist=self.pending_result.alpha*self.baseline.last_feedback_twist
            self.sink.emit('control_sample',control_id=self._control_id,reference_s=self.reference_time_s,
                nominal_twist_tool=nominal,candidate_twist_tool=self.pending_result.qp_twist,
                previous_outer_command_tool=previous,command_slew_dt_s=actual_dt,
                rotation_base_tcp=rotation,
                repair_episode=(dict(self.pending_result.diagnostics['repair_episode'])
                    if 'repair_episode' in self.pending_result.diagnostics else None),
                allocation_diagnostics={key:value for key,value in self.pending_result.diagnostics.items()
                    if key.startswith("differential_") or key in ("repair_force_gate","visual_window_active",
                        "repair_permission_mode","qp_delta_omega_y_rad_s")},
                alpha=self.pending_result.alpha,alpha_target=self._pending_alpha_target,
                alpha_preferred=self._pending_alpha_preferred,h_ref_s=h_ref,command_hold_model_s=dt,
                reference_interval_s=reference_dt,numeric_attempts=self.pending_result.diagnostics.get("numeric_attempts"),
                compute_elapsed_s=time.monotonic()-now,cycle_target_overrun=time.monotonic()-now>self.baseline.dt,
                control_actual_dt_s=actual_dt,source=source.__dict__,
                feature=None if observation is None else observation.to_dict(),image_compatible=image_valid,
                image_feedback_status=image_reason,
                control_wrench_tool=f_ext,physical_wrench_candidate_tool=f_ext_raw,
                measured_twist_base=measured_twist_base,measured_twist_metadata=metadata,
                measured_twist_valid=measured_twist_valid,measurement_time_s=measurement_time_s,
                energy=(self.command_budget.facts if self.command_budget is not None else
                        None if self.energy is None else self.energy.facts),physical_certified=False)
            self._drain_command_energy()
            return self.pending_result.qp_twist.copy()
        except Exception:
            self.publication_abort('outer_prepare_failed',definitely_not_sent=True)
            raise

    def publication_review(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer publication')
        final=np.asarray(final_tool,dtype=float).reshape(6)
        result=self.pending_result
        def rejected(reason):
            self._publication_rejection_reason=reason
            self.sink.emit('publication_review_rejected',control_id=candidate_id,reason=reason,
                review_time_s=now_s,created_time_s=result.created_time_s,
                valid_until_s=result.hard_constraints.valid_until_s,
                final_command_model_tool=final,facts=facts or {})
            return False
        if not np.isfinite(final).all() or not np.isfinite(now_s):return rejected('nonfinite_payload_or_time')
        if now_s<result.created_time_s:return rejected('review_time_reversed')
        if self._rocking is not None:
            self._rocking_review_facts=dict(facts or {})
            rocking_tolerance=float(self._rocking_review_facts.get("rocking_tolerance_rad_s",self.solver.config.feasibility_tolerance))
            if not np.isfinite(rocking_tolerance) or rocking_tolerance<0:return rejected("invalid_rocking_tolerance")
            if not self._rocking.review(final,self._rocking_review_facts,rocking_tolerance):
                return rejected('final_rocking_interval_violation')
            self.sink.emit('final_rocking_review',control_id=candidate_id,
                final_omega_y_rad_s=float(final[4]),axis_base=self._pending_rocking[0] if self._pending_rocking else None,
                bounds=self._pending_rocking[1] if self._pending_rocking else None,
                history=self._pending_rocking[2] if self._pending_rocking else None,**self._rocking_review_facts)
            if self._quality_intervals is not None and self._pending_rocking is not None:
                tier=int(self._rocking_review_facts['rocking_policy_tier'])
                names=('rotation_speed_limit','rotation_acceleration_limit','rotation_jerk_limit')
                for name,pair in zip(names[:max(0,4-tier)],self._pending_rocking[1].reshape(3,2)):
                    if min(abs(final[4]-pair[0]),abs(final[4]-pair[1]))<=rocking_tolerance:
                        self._quality_intervals.add_reasons([name])
            if self._quality_intervals is not None and self._rocking_review_facts.get('rocking_limited'):
                self._quality_intervals.add_reasons(['rotation_smoothing_limited_by_mechanics'])
        if self.command_budget is not None:
            # Original IK owns the final mechanical constraints; its payload
            # does not carry the outer QP's exported task certificate. Admission
            # uses the actual force-source deadline and the final-model budget.
            if not 0<=now_s-self._source_step.source_t_s<self.source_clock.max_age_s:return rejected('wrench_source_expired')
            if not self.command_budget.reserve(candidate_id,result.energy_certificate,final,now_s=now_s):return rejected('energy_reservation_rejected')
            self._reserved_id=candidate_id
            self._drain_command_energy()
        elif now_s>=result.hard_constraints.valid_until_s:return rejected('certificate_expired')
        self._review_time_s=float(now_s)
        self.sink.emit('publication_review',control_id=candidate_id,final_command_model_tool=final,
            task_hard_violation=result.hard_constraints.violation(final),task_certificate=False,
            physical_certified=False,review_time_s=now_s,
            dispatch_valid_until_s=self._source_step.source_t_s+self.source_clock.max_age_s,
            exported_task_valid_until_s=result.hard_constraints.valid_until_s,facts=facts or {})
        return True

    def publication_started(self,candidate_id):
        if candidate_id!=self._pending_id:raise RuntimeError('orphan publication start')
        if self.command_budget is not None:
            # Last gate before the first device send. A late review/logging tick
            # must be rejected while the rail reservation is still abortable.
            now=time.monotonic()
            pending=self.command_budget.pending
            valid=(self._review_time_s is not None and self._review_time_s<=now and
                self._dispatch_time_s is None and self._reserved_id==candidate_id and
                pending is not None and pending.command_id==candidate_id and now<pending.expires_s and
                self.pending_result.created_time_s<=now)
            source_age=now-self._source_step.source_t_s
            reason=('missing_review_or_invalid_dispatch' if not valid else
                    'dispatch_source_expired' if source_age>=self.source_clock.max_age_s else
                    'dispatch_source_future' if not source_age>=0 else None)
            if reason is not None:
                self._publication_rejection_reason=reason
                self.sink.emit('publication_dispatch_rejected',control_id=candidate_id,
                    dispatch_time_s=now,reason=reason,
                    source_time_s=self._source_step.source_t_s,
                    dispatch_valid_until_s=self._source_step.source_t_s+self.source_clock.max_age_s)
                return False
            self.command_budget.publication_started(candidate_id)
            self._dispatch_time_s=now
        return True

    def publication_commit(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer commit')
        final=np.asarray(final_tool,dtype=float).reshape(6)
        self.sink.emit('publication_transport_result',control_id=candidate_id,publication_time_s=now_s,
            dispatch_time_s=self._dispatch_time_s,review_time_s=self._review_time_s,
            final_command_model_tool=final,facts=facts or {},logical_commit=False)
        if self.command_budget is not None:
            if self._dispatch_time_s is None or not np.isfinite(now_s) or now_s<self._dispatch_time_s:
                self.command_budget.fail('commit_without_valid_dispatch',now_s=now_s)
                raise ValueError('commit without valid dispatch')
            # Account successful transport against the reserved held-pair epoch.
            # The force was fresh at dispatch; crossing an unrelated 10 ms
            # outer-task diagnostic deadline after sending cannot undo a send.
            self.command_budget.commit(candidate_id,final,now_s=now_s,
                rotation_base_tcp=self.pending_rotation_base_tcp,
                dual_success=bool(facts and facts.get('arm')=='sent' and facts.get('rail') in ('sent','disabled')))
            self._drain_command_energy()
        path=self.pending_basis[:,2];denom=float(path @ path)
        alpha=0. if denom<=1e-28 else float(np.clip(path @ final/denom,0.,self.pending_result.alpha))
        # These are command integrators. The rail-compensated payload model
        # is not a newly requested outer action and must not rebase them.
        accepted_outer=self.pending_result.qp_twist
        force=np.zeros(6);force[[2,4]]=accepted_outer[[2,4]]
        self.nominal.commit_applied(force,final_full_twist=accepted_outer)
        timely = self.command_budget is not None or (np.isfinite(now_s) and
            self.pending_result.created_time_s <= now_s < self.pending_result.hard_constraints.valid_until_s)
        if not timely: alpha=0.
        if self.pending_h_ref>0:
            self.reference_time_s=self.reference.commit_time(self.reference_time_s,alpha,self.pending_reference_dt)
            if self.contact_gate is not None:
                self.contact_gate._elapsed_s=self.reference.local_time(self.reference_time_s)
                self.contact_gate._last_input_t_s=self.reference_time_s
        # Commit only on dual-device publication success. Final-model residuals
        # remain in energy/progress accounting and logs; they cannot become
        # nominal or outer command-slew history in a different space.
        if self._quality_progress is not None:
            self._quality_progress.commit(self._pending_alpha_preferred)
            if self._rocking.time_s is None:self._rocking.seed(np.zeros(3),now_s-self.pending_actual_dt)
            audit=self._rocking.publication_audit(final,self.pending_rotation_base_tcp,now_s,
                self.pending_actual_dt,int(self._rocking_review_facts['rocking_policy_tier']),
                float(self._rocking_review_facts.get('rocking_tolerance_rad_s',self.solver.config.feasibility_tolerance)))
            self.sink.emit('final_rocking_publication',control_id=candidate_id,publication_time_s=now_s,
                final_omega_y_rad_s=float(final[4]),selected_policy_tier=self._rocking_review_facts['rocking_policy_tier'],**audit)
            if audit['timing_limited']:self._quality_intervals.add_reasons(['rotation_smoothing_limited_by_timing'])
            self._rocking.commit(final,self.pending_rotation_base_tcp,now_s)
            reasons=[]
            diag=self.pending_result.diagnostics
            request=float(diag.get('differential_request_m_s',0.))
            row=diag.get('differential_row')
            if request>0 and row is not None:
                achieved=float(diag.get('differential_sign',0.))*float(np.asarray(row)@final)
                if achieved+self.solver.config.feasibility_tolerance<request:reasons.append('final_repair_shortfall')
            if alpha+1e-8<self.pending_result.alpha:reasons.append('inner_progress_limit')
            self._quality_intervals.add_reasons(reasons)
            if self._endpoint_started_s is None and self.reference.exhaustion_reason(self.reference_time_s):
                self._endpoint_started_s=now_s
                self.sink.emit('reference_complete',time_s=now_s,reference_s=self.reference_time_s,phase='endpoint_convergence')
                print('[CONTACT_QP] reference complete; endpoint convergence.',flush=True)
        self._previous=self.pending_result.qp_twist.copy()
        self._previous_rotation=self.pending_rotation_base_tcp.copy()
        self._nominal_pending=False;self._reserved_id=None
        self._publication_retry_count=0;self._publication_retry_source_t_s=None
        quality_alpha=(self._pending_alpha_preferred if self._pending_alpha_preferred is not None
                       else self.pending_result.diagnostics.get('alpha_preferred',1.))
        tolerance=self.solver.config.feasibility_tolerance
        progress_limit=('endpoint_convergence' if self.reference.exhaustion_reason(self.reference_time_s) else
                        'waiting_for_contact' if self.pending_h_ref<=0 else
                        'inner_or_final_model' if alpha+tolerance<self.pending_result.alpha else
                        'outer_constraints' if self.pending_result.alpha+tolerance<quality_alpha else
                        'quality' if quality_alpha<1.-tolerance else 'none')
        self.sink.emit('publication',control_id=candidate_id,success=True,accepted_alpha=alpha,
            progress_limited_by=progress_limit,
            reference_s=self.reference_time_s,reference_interval_s=self.pending_reference_dt,
            alpha_target=self._pending_alpha_target,alpha_preferred=self._pending_alpha_preferred,
            final_command_model_tool=final,
            task_certificate=False,physical_certified=False,facts=facts or {})
        self._pending_id=None;self._dispatch_time_s=None;self._review_time_s=None
        self._publication_rejection_reason=None

    def publication_abort(self,reason,*,definitely_not_sent=False,facts=None):
        # A delegated sample can raise after preparing only one nominal part.
        # Retire each existing proposal without rolling back consumed measurements.
        if self.nominal.z_law._transaction.pending is not None:
            self.nominal.z_law.abort()
        if self.nominal.tilt._pending_command is not None:
            self.nominal.tilt.abort()
        self._nominal_pending=False
        if self.command_budget is not None:
            self.command_budget.reject_new_only(self._reserved_id,definitely_not_sent=definitely_not_sent,now_s=time.monotonic())
            self._reserved_id=None
        self.sink.emit('publication_aborted',control_id=self._pending_id,reason=reason,
                       reference_s=self.reference_time_s,facts=facts or {})
        self._drain_command_energy()
        if self._continuous_execution and definitely_not_sent:self._reference_resume_after_s=time.monotonic()

    def arrived(self,pose):
        if not self.reference.exhaustion_reason(self.reference_time_s):return False
        target=self.reference.reference.sample(self.reference.duration_s).pose_d
        rotation=Rotation.from_euler(self.controller.cfg.euler_order,np.asarray(pose)[3:]).as_matrix()
        return bool(np.linalg.norm((rotation.T @ (np.asarray(pose)[:3]-target[:3]))*self.baseline.selection[:3])<=1e-4)

    def record_stop(self,reason):
        if self._stop_recorded:return
        self._stop_recorded=True
        if self._quality_intervals is not None:self._quality_intervals.close(time.monotonic())
        self.publication_abort(reason)
        if self.command_budget is not None:self.command_budget.stop(now_s=time.monotonic())
        self._drain_command_energy()
        self.sink.emit('stop',reason=reason,reference_s=self.reference_time_s,physical_execution='unconfirmed')

    def close(self,*,wait=True):
        self.features.close(wait=wait);self.sink.close(wait=wait)

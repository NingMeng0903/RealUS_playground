"""Publication-owned contact QP around the original ICRA TFF and IK interfaces.

Only the outer command changes. Final IK task residuals are recorded, while the
six-dimensional command energy budget is rechecked on the final payload model.
Neither successful transport nor this model establishes physical certification.
"""
from __future__ import annotations
from dataclasses import replace
import math
import time
import uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rm75_control.control.joint_admittance_8dof.loop import _rt_print
from peirastic.contact_qp.qp import ContactQp,QpConfig,QpInput
from peirastic.contact_qp.types import ProbeGeometry,ContactStatus,REQUIRED_WINDOWS,TwistConstraints,REGION_FEATURE_VERSION
from peirastic.contact_qp.geometry import (motion_basis,twist_tcp_to_face,wrench_tcp_to_face,
    affine_contact_motion,contact_cop_from_wrench)
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.runtime_source import SourceClock,SourceGapContext
from peirastic.contact_qp.execution import EXECUTION_POLICY,ProposalDeferred,QualityProgress,QualityIntervals
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from peirastic.contact_qp.region_visual import region_visual_request
from peirastic.contact_qp.confidence_fusion import (POLICIES, FEATURE_VERSION, TASK_SOURCE,
    ConfidenceAngularTask, cop_preference, task_components, accepted_loading)
from peirastic.contact_qp.runtime_config import validate_study_config,calibrated_geometry,source_settings
from peirastic.contact_qp.energy import EnergyLedger,PortBounds
from peirastic.contact_qp.runtime_energy import RuntimeEnergy
from .contact_recording import ContactRecordSink,FeatureReceiver,clock_metadata,study_fingerprints


def _reference_ramp_s(reference):
    obj=reference
    seen=set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        ramp=getattr(obj,'ramp',None)
        if ramp is not None:
            try:
                value=float(ramp)
            except (TypeError,ValueError):
                value=float('nan')
            if math.isfinite(value) and value>=0.:
                return value
        obj=getattr(obj,'reference',None)
    return 0.

_LATEST_IMAGE=object()

class _PrepareLaw:
    """Choose the existing law's transaction entry without copying its math."""
    def __init__(self,law):self.law=law;self.context={};self.geometry=None;self.last_output=None
    def __getattr__(self,name):return getattr(self.law,name)
    def contact_pose(self,pose,euler_order):
        pose=np.asarray(pose,dtype=float)
        rotation=Rotation.from_euler(euler_order,pose[3:]).as_matrix()
        face=self.geometry.T_tcp_face
        return np.r_[pose[:3]+rotation @ face[:3,3],
                     Rotation.from_matrix(rotation @ face[:3,:3]).as_euler(euler_order)]
    def reset(self,**kwargs):
        if self.geometry is not None:
            kwargs['pose']=self.contact_pose(kwargs['pose'],self.law.controller.cfg.euler_order)
            kwargs['f_ext']=wrench_tcp_to_face(kwargs['f_ext'],self.geometry)
        return self.law.reset(**kwargs)
    def update(self,**kwargs):
        if self.geometry is not None:
            kwargs['pose']=self.contact_pose(kwargs['pose'],kwargs.get('euler_order') or 'xyz')
            kwargs['f_ext']=wrench_tcp_to_face(kwargs['f_ext'],self.geometry)
            if kwargs.get('f_ext_raw') is not None:
                kwargs['f_ext_raw']=wrench_tcp_to_face(kwargs['f_ext_raw'],self.geometry)
            path=twist_tcp_to_face(self.geometry) @ np.asarray(kwargs['path_twist'])
            path[[2,4]]=0.;kwargs['path_twist']=path
            # f_des is the fixed [0,0,4,0,0,0] task in C; it is a task
            # definition, not a physical wrench attached to the old TCP.
        self.last_output=self.law.prepare(**kwargs,**self.context)
        return self.last_output


class ContactQpOuter:
    contact_study_enabled=True
    contact_qp_enabled=True
    owns_reference_clock=True

    def __init__(self,baseline,config,*,sink=None,feature_receiver=None):
        validate_study_config(config)
        if config.get('mode')!='active':raise ValueError('active outer requires mode=active')
        self._continuous_execution=config.get('execution_policy')==EXECUTION_POLICY
        self._delayed=(config.get('qp') or {}).get('allocation_policy')=='delay_kf_cop_v1'
        source=source_settings(config)
        self.source_clock=SourceClock(**source)
        if config.get('force_axis_monotonicity_confirmed') is not True:
            raise ValueError('declare the measured relation between original force axis and face loading')
        self.baseline=baseline;self.config=dict(config);self.controller=baseline.controller
        if self.controller.cfg.control_frame!='tool' or baseline.desired_force[2]!=4.:
            raise ValueError('active ICRA outer retains original +4 N tool-Z task')
        from peirastic.contact_qp.runtime_config import motion_settings
        motion=motion_settings(config)
        if motion:
            # This controller belongs to the new phase; set the profile before
            # its first prepare and before constructing any outer hard rows.
            self.controller.cfg.max_velocity=np.asarray(self.controller.cfg.max_velocity,dtype=float).copy()
            self.controller.cfg.max_velocity[2]=motion['normal_max_m_s']
            self.controller.cfg.max_vz_tool_m_s=motion['normal_max_m_s']
            self.controller.cfg.force_barrier.v_seek_free_m_s=motion['seek_m_s']
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
        if self._delayed:
            self._preview.geometry=geometry
            # The contact selector is applied after transforming the full path
            # to C. The legacy TCP selector must not discard screw components.
            self.baseline.mask_force_from_path=False
            if not np.allclose(geometry.T_tcp_face,np.eye(4),rtol=0,atol=1e-12):
                self.baseline.position.cfg.track_axes=np.ones(6)
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
        if motion:
            _rt_print(f"[CONTACT_QP] normal_max={vmax[2]*1000:.1f} mm/s; "
                      f"air_seek={self.controller._v_air_seek()*1000:.1f} mm/s; "
                      f"force_age_limit={self.source_clock.max_age_s*1000:.0f} ms; "
                      f"candidate_lifetime={self.solver.config.certificate_horizon_s*1000:.0f} ms")
        tilt=self.nominal.tilt.cfg
        self._image_owned=self.solver.config.allocation_policy in POLICIES
        self._command_lease=None;self._image_kf=None;self._pending_prediction=None
        if self._delayed:
            from peirastic.contact_qp.command_lease import CommandLease
            from peirastic.contact_qp.delayed_kf import DelayConfidenceKF,KfConfig
            self._command_lease=CommandLease(config['command']['max_interval_s'])
            self._image_kf=DelayConfidenceKF(KfConfig(**config['image_kf']))
        self._visual_task=None
        if self._image_owned:
            from peirastic.contact_qp.geometry import window_rows
            rows=window_rows(self.geometry,self.feature_config.lateral_windows)
            self._visual_task=ConfidenceAngularTask(mass=tilt.mass,damping=tilt.damping,
                repair_speed_m_s=self.solver.config.repair_speed_m_s,lever_m=abs(rows[0,4]-rows[2,4]),
                image_x_sign=self.geometry.image_x_sign,deadband=self.solver.config.differential_repair.balance_deadband,
                c_min=self.solver.config.c_min,max_velocity=vmax[4],max_acceleration=acceleration[4])
        self._pending_fusion={}
        self._rocking=(RockingSmoothing(vmax[4],acceleration[4],tilt.mass,tilt.damping)
            if self._continuous_execution else None)
        self._pending_rocking=None;self._rocking_review_facts={}
        self._rocking_export_id=None
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
            self.solver.config.max_image_age_s,self.sink.emit,
            region_count=self.feature_config.region_count if self._delayed else None) if self._continuous_execution else None)
        self._endpoint_started_s=None;self._deferred_source_t_s=None
        self._reference_resume_after_s=None
        self.features=feature_receiver or FeatureReceiver(config['feature_endpoint'],retain_oos=self._delayed)
        self.reference_time_s=0.;self._control_time_s=0.;self._control_id=0;self.arrival_confirmed=False
        self._source_step=None;self._source_prepared=False;self._source_epoch_reset=False;self._angle_origin=None
        self._last_velocity_time=None;self._previous=np.zeros(6);self._previous_rotation=None;self.pending_result=None
        self._previous_final=np.zeros(6);self._pending_previous_final=np.zeros(6)
        self._pending_id=None;self._nominal_pending=False;self._reserved_id=None;self._stop_recorded=False
        self._dispatch_time_s=None;self._review_time_s=None
        self._publication_rejection_reason=None
        self._publication_retry_source_t_s=None;self._publication_retry_count=0
        self._lease_expiry_retries=0
        self._lease_expiry_retry_limit=3
        self._recovery_started_s=None
        self.last_path_twist=np.zeros(6);self.last_feedback_twist=np.zeros(6)
        self.sink.emit('study_start',mode='active',command_authority='outer_qp_original_ik',
            physical_port_assurance='unverified',task_certificate=False,config=config,
            **clock_metadata(),**study_fingerprints(baseline,config))
        if self._delayed:
            print(f"[CONTACT_QP] fusion=delay_kf_cop_v1; mechanical=force_torque; "
                  f"visual={self.feature_config.region_count}_region_KF; energy=disabled; command_lease={self._command_lease.max_command_interval_s:.3f}s; "
                  "cop_pairing=coupled_normal_task_v1; feedback=affine; alpha=feedforward_only",flush=True)
        elif self.solver.config.differential_repair.permission_mode=='continuous':
            print(f"[CONTACT_QP] visual=continuous; image=required; "
                  f"image_loss={self._image_dropout_policy}; "
                  f"tank={energy['initial_j']:.3f}/{energy['capacity_j']:.3f} J; "
                  f"reserve={energy['stopping_reserve_j']:.3f} J; "
                  f"task_power={energy.get('task_power_source','none')}; "
                  f"fusion={self.solver.config.allocation_policy if self._image_owned else 'shared_total_velocity'}",flush=True)
        if self._image_owned:
            print(f"[CONTACT_QP] angular_task=confidence_centroid; feature={FEATURE_VERSION}; "
                  f"cop_pairing={self.solver.config.allocation_policy=='confidence_cop_v1'}; "
                  "state=separate_loading_visual_scan; supply=loading_scan_v1; experimental=true",flush=True)
        if self._continuous_execution:
            print(f"[CONTACT_QP] execution={EXECUTION_POLICY}; clock=actual_success_interval; "
                  f"quality=deficit_only; alpha=0.25..1 slew=path_ramp; solver=bounded_retry_v1; "
                  f"source_gap={self.source_clock.gap_policy}; rocking={self._rocking.policy}; "
                  f"jerk={self._rocking.jerk_limit:.6g} rad/s^3",flush=True)
        if self.command_budget is not None:
            print("[CONTACT_QP] final_power=inner_qp1_qp2; velocity_model=rebased_command_delta_v2",flush=True)

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
        self.source_clock.begin_epoch()
        self._source_epoch_reset=True

    def begin_hybrid_episode(self,applied_twist_base,current_pose):
        self.baseline.begin_hybrid_episode(applied_twist_base,current_pose)
        if self._delayed:
            rotation=Rotation.from_euler(self.controller.cfg.euler_order,np.asarray(current_pose)[3:]).as_matrix()
            applied=np.asarray(applied_twist_base)
            self._previous=np.r_[rotation.T @ applied[:3],rotation.T @ applied[3:]]
            self._previous_final=self._previous.copy()
            self._previous_rotation=rotation.copy()
        if self._visual_task is not None:
            self._visual_task.omega_base=np.asarray(applied_twist_base)[3:].copy()
        if self._rocking is not None:
            self._rocking.seed(np.asarray(applied_twist_base)[3:],time.monotonic())
        self.source_clock.begin_epoch()
        self._source_epoch_reset=True

    def take_source_epoch_reset(self):
        flag=self._source_epoch_reset
        self._source_epoch_reset=False
        return flag

    def prepare_source(self,source_id,source_t_s,wall_time_ns,*,now_s):
        previous=self.source_clock.last
        context=None
        budget=self.command_budget or self._command_lease
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
            if (source_id==previous.source_id and self.source_clock.max_interval_s is not None
                    and source_t_s-previous.source_t_s>self.source_clock.max_interval_s):
                old=None if budget is None else budget.active
                live=(old is not None and not budget.latched_reason and budget.pending is None
                      and not budget.started and old.committed_s<source_t_s<=now_s<old.expires_s)
                if live:
                    context=SourceGapContext(source_id,previous.source_t_s,source_t_s,
                        old.command_id,old.committed_s,old.expires_s,now_s,budget.max_command_interval_s)
                elif self._continuous_execution:
                    self.source_clock.begin_epoch()
                    self._source_epoch_reset=True
                    self.sink.emit('source_epoch_reset',previous_source_t_s=previous.source_t_s,
                                   source_t_s=source_t_s,reason='unleased_or_expired_gap')
        self._source_step=self.source_clock.observe(source_id,source_t_s,wall_time_ns,
            now_s=now_s,gap_context=context)
        self._source_prepared=True
        if self._source_step.gap_recovered:
            self.sink.emit('source_gap_recovered',source=self._source_step.__dict__,
                           filter_policy='exact_foh_gap_v1',reference_catch_up=False)
        return self._source_step

    def rocking_constraints(self):
        if self._rocking is None:return {}
        rotation=self.pending_rotation_base_tcp
        if self._delayed:rotation=rotation @ self.geometry.T_tcp_face[:3,:3]
        axis,bounds,facts=self._rocking.preview(rotation,
            time.monotonic(),self.pending_actual_dt)
        self._pending_rocking=(axis,bounds,facts)
        self._rocking_export_id=self._pending_id
        return dict(rocking_axis_base=axis,rocking_bounds=bounds)

    def _final_command_constraints(self):
        """Review-only slew envelope re-anchored to the last accepted final.

        Inner IK no longer receives these rows. Publication review uses them
        with a one-tick acceleration allowance.
        """
        hard=self.pending_result.hard_constraints
        if not self._delayed:return hard
        lower,upper=hard.lower.copy(),hard.upper.copy()
        delta=self.solver.config.max_acceleration*max(self.pending_actual_dt,self.pending_dt)
        identity=np.eye(6)
        for index,label in enumerate(hard.labels):
            for axis in range(6):
                if label==f'acceleration_{axis}' and np.array_equal(hard.A[index],identity[axis]):
                    lower[index]=self._pending_previous_final[axis]-delta[axis]
                    upper[index]=self._pending_previous_final[axis]+delta[axis]
                    break
        return replace(hard,lower=lower,upper=upper)

    def command_power_constraints(self):
        """Admit the future command model in BOTH inner QPs before sending.

        Available energy is conservative until the original active lease ends:
        old outward work reduces balance and old reservation by the same amount.
        The final reservation still checks the current balance without tolerance.
        """
        if self.command_budget is None:return {}
        if not self._nominal_pending:raise RuntimeError('power row requires pending outer proposal')
        bound=self.pending_result.energy_certificate
        if bound is None:raise RuntimeError('missing frozen command power bound')
        rotation=self.pending_rotation_base_tcp
        wrench=np.r_[rotation @ bound.wrench_environment[:3],rotation @ bound.wrench_environment[3:]]
        minimum=-(bound.task_power_w+bound.beta*bound.available_j/bound.hold_s)
        self.sink.emit('inner_command_power_constraint',control_id=self.pending_id,
            wrench_base=wrench,minimum_power_w=minimum,available_j=bound.available_j,
            task_power_w=bound.task_power_w,hold_s=bound.hold_s,
            velocity_model='rebased_command_delta_v2',physical_certified=False)
        return dict(command_power_wrench_base=wrench,command_power_min_w=minimum)

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
                'centroid_version_mismatch' if self._image_owned and observation.confidence_feature_version!=FEATURE_VERSION else
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
                    _rt_print(f"[CONTACT_QP] visual resumed; image age={age*1000:.1f} ms")
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
                        _rt_print(f"[CONTACT_QP] visual paused; {reason}; image age={age_label}; fresh-force task continues")
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

    def _delayed_observation(self,now,batch,snapshot):
        """Consume transport observations once before QP transactions begin."""
        expected=(self.registration_version,self.feature_config.window_version,self.feature_config.calibration_version)
        for obs in batch:
            compatible=(obs.version==expected and self._region_layout_compatible(obs))
            accepted=self._image_kf.ingest(obs,now) if compatible else False
            replayed=self._image_kf.predict(now)
            self.sink.emit('image_kf_measurement',accepted=accepted,
                reason=self._image_kf.last_reason if compatible else 'registration_or_feature_mismatch',
                feature=obs.to_dict(),latest_replayed_innovation=self._image_kf.last_innovation,
                latest_replayed_innovation_covariance=self._image_kf.last_innovation_covariance,
                latest_replayed_frame_seq=replayed.frame_seq,
                latest_replayed_effective_time_s=replayed.effective_time_s)
        prediction=self._image_kf.predict(now)
        if snapshot is not None and (snapshot.version!=expected or not self._region_layout_compatible(snapshot)):
            prediction=replace(prediction,valid=False,reason='registration_or_feature_mismatch')
        self._pending_prediction=prediction
        return snapshot if prediction.valid else None,prediction.reason

    def _region_layout_compatible(self,observation):
        """The production branch requires all N channels; LR is diagnostic only."""
        return bool(observation.region_feature_version==REGION_FEATURE_VERSION
            and observation.region_layout_version==self.feature_config.region_layout_version
            and observation.region_confidence is not None
            and np.shape(observation.region_confidence)==(self.feature_config.region_count,)
            and np.array_equal(observation.region_edges,self.feature_config.region_edges))

    def retry_unsent_publication(self):
        """Allow fresh recomputation inside an unchanged, committed old lease.

        The runner calls this only after aborting rail, outer and inner proposals.
        No deadline, watchdog heartbeat, source timestamp or command is renewed.
        """
        budget=self.command_budget or self._command_lease
        retry_reasons={'wrench_source_expired','dispatch_source_expired','dispatch_certificate_expired','certificate_expired'}
        if self._continuous_execution:retry_reasons.update(('awaiting_fresh_force','solver_deadline_exceeded','solver_attempts_exhausted','native_timeout'))
        if (self._continuous_execution and self._publication_retry_source_t_s is None
                and 'lease expired' in str(self._publication_rejection_reason or '')):
            self._publication_rejection_reason=None
            return True
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

    def _solver_retry_reason(self):
        return self._publication_rejection_reason in (
            'solver_attempts_exhausted','solver_deadline_exceeded')

    def _forgive_solver_retry_lease(self,budget,now_s):
        """Drop a dead solver-retry lease so the next tick can reserve a new one."""
        self._lease_expiry_retries+=1
        if budget is not None:
            budget.latched_reason=None
            budget.active=None
            budget.pending=None
            budget.started=False
            budget.last_time_s=float(now_s)
        self._publication_retry_source_t_s=None
        self._publication_rejection_reason=None
        self.sink.emit('publication_lease_forgiven',retry_count=self._lease_expiry_retries,
            limit=self._lease_expiry_retry_limit)

    def waiting_for_retry_source(self,source_t_s,*,now_s):
        """No new proposal or watchdog heartbeat until the rejected source changes."""
        watermark=self._publication_retry_source_t_s
        if watermark is None:return False
        budget=self.command_budget or self._command_lease
        def expire(reason):
            text=str(reason or '')
            if (self._continuous_execution and self._solver_retry_reason()
                    and self._lease_expiry_retries<self._lease_expiry_retry_limit):
                self._forgive_solver_retry_lease(budget,now_s)
                raise ProposalDeferred('fresh publication retry lease expired: '+text)
            raise RuntimeError('fresh publication retry lease expired: '+text)
        if budget is None or budget.active is None or budget.latched_reason:
            if (budget is not None and str(budget.latched_reason)=='committed_lease_expired'
                    and self._continuous_execution and self._solver_retry_reason()
                    and self._lease_expiry_retries<self._lease_expiry_retry_limit):
                expire(budget.latched_reason)
            raise RuntimeError('fresh publication retry has no valid committed lease')
        if not np.isfinite(now_s) or now_s<budget.last_time_s:
            raise RuntimeError('fresh publication retry clock invalid')
        if now_s>budget.last_time_s and not budget.advance(now_s):
            expire(budget.latched_reason)
        if budget.active is None or now_s>=budget.active.expires_s:
            expire(budget.latched_reason or '')
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
        image_batch=()
        if self._delayed:
            drain=getattr(self.features,'drain_observations',None)
            image_batch=drain() if drain is not None else (() if image_snapshot is None else (image_snapshot,))
        now=time.monotonic();actual_dt=float(self.baseline.dt if dt_actual is None else dt_actual)
        contact_force=(float(wrench_tcp_to_face(f_ext,self.geometry)[2]) if self._delayed else float(f_ext[2]))
        contact_force_raw=(contact_force if f_ext_raw is None else
            float(wrench_tcp_to_face(f_ext_raw,self.geometry)[2]) if self._delayed else float(np.asarray(f_ext_raw)[2]))
        if self._delayed:
            delayed_observation,delayed_image_reason=self._delayed_observation(now,image_batch,image_snapshot)
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
        previous_final=self._previous_final.copy()
        if self._previous_rotation is not None:
            old_to_current=rotation.T @ self._previous_rotation
            previous=np.r_[old_to_current @ previous[:3],old_to_current @ previous[3:]]
            previous_final=np.r_[old_to_current @ previous_final[:3],old_to_current @ previous_final[3:]]
        self._pending_previous_final=previous_final
        if self._image_owned:
            # The image dynamic target and its hard outer Y slew must share
            # the same successful final angular history. Translational history
            # remains an outer command: rail compensation never enters loading.
            previous[4]=float((rotation.T @ self._visual_task.omega_base)[1])
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
            if source.fresh:self.contact_gate.observe_force(contact_force,source.source_t_s,valid=True)
            self.contact_gate.sample(self.reference_time_s)
        ref,h_ref=self.reference.candidate(self.reference_time_s,reference_dt if reference_dt>0 else dt)
        if reference_dt<=0:h_ref=0.
        if self.contact_gate is not None and not self.contact_gate.started:
            ref=replace(ref,pose_d=np.asarray(current_pose,dtype=float).reshape(6).copy(),
                        vel_ff=np.zeros(6));h_ref=0.
        self.pending_h_ref=h_ref
        self.baseline.position.set_reference_override(ref)
        self._preview.context=dict(control_step_id=self._control_id,source_sample_id=source.sample_id,
            source_t_s=source.source_t_s,measurement_fresh=source.fresh,source_dt_s=source.source_dt_s,
            velocity_measurement_fresh=velocity_fresh,velocity_dt_s=velocity_dt)
        try:
            if (self.solver.config.allocation_policy in ('differential_repair_v8','delay_kf_cop_v1',*POLICIES) and source.fresh):
                original_scalar=float(f_ext[2])
                raw_scalar=original_scalar if f_ext_raw is None else float(np.asarray(f_ext_raw)[2])
                supervisor_values=(original_scalar,raw_scalar,contact_force,contact_force_raw) if self._delayed else (original_scalar,raw_scalar)
                if max(supervisor_values)>=self.solver.config.differential_repair.hard_stop_n:
                    self.sink.emit('force_supervisor_stop',control_id=self._control_id,
                        filtered_original_force_n=original_scalar,raw_original_force_n=raw_scalar,
                        filtered_contact_force_n=contact_force,raw_contact_force_n=contact_force_raw,
                        primary_force_frame='contact_face' if self._delayed else 'tcp_tool',
                        additional_stop_frame='tcp_tool' if self._delayed else None,
                        threshold_n=6.,source=source.__dict__,
                        assurance='fresh_measurement_stop_not_inter_sample_peak_guarantee')
                    raise RuntimeError('v8 fresh force supervisor reached 6 N; original stop chain required')
            self._pending_fusion={}
            if self._image_owned:
                observation,image_reason=self._image_observation(now,
                    contact_enabled=self.contact_gate is None or self.contact_gate.started,observation=image_snapshot)
                enabled=bool(self.controller.contact_present) and (self.contact_gate is None or self.contact_gate.started)
                omega,self._pending_fusion=self._visual_task.preview(observation,rotation_base_tcp=rotation,
                    dt_s=actual_dt,force_gate=self.solver.config.differential_repair.force_gate(float(f_ext[2])),enabled=enabled)
                # Re-express the successful angular state before the original
                # tilt transaction's hard caps/slew. Its torque drive is bypassed.
                self.nominal.tilt._w=self.nominal.tilt.omega_y=float((rotation.T @ self._visual_task.omega_base)[1])
                self._preview.context['visual_velocity_target_rad_s']=omega
            nominal=np.asarray(self.baseline.sample(self._control_time_s,current_pose,f_ext,contact=contact,
                f_ext_raw=f_ext_raw,dt_actual=actual_dt,sensor_age_s=source.age_s,feedback_age_s=feedback_age_s,
                feedback_fresh_tick=feedback_fresh_tick,feedback_velocity_valid=velocity is not None,
                v_tcp_z_actual=None if velocity is None else float((twist_tcp_to_face(self.geometry) @ velocity)[2]) if self._delayed else float(velocity[2]),slack_norm=slack_norm))
            self._nominal_pending=True
            if self._delayed:
                tf=twist_tcp_to_face(self.geometry)
                feedback=tf @ np.asarray(self.baseline.last_feedback_twist)
                feedforward=tf @ np.asarray(self.baseline.last_path_twist)
                self._pending_mechanical_contact=feedback+feedforward
                self._pending_mechanical_contact[[2,4]]=np.asarray(self._preview.last_output.v_force)[[2,4]]
                nominal=np.linalg.solve(tf,self._pending_mechanical_contact)
            self._pending_mechanical_tcp=nominal.copy()
            # Consume the accepted source's real LP/HP measurement once even
            # if computation used up its remaining send age. Commands roll
            # back; measurement history does not.
            if self.nominal.tilt.needs_normal_retract:
                timeout=float(self.nominal.tilt.cfg.recovery_timeout_s)
                if self._recovery_started_s is None:
                    self._recovery_started_s=now
                    self.sink.emit('mechanical_recovery',control_id=self._control_id,
                        tilt_stop_reason=self.nominal.tilt.tilt_stop_reason,
                        tilt_frozen=bool(self.nominal.tilt.tilt_frozen),
                        tilt_capped=bool(self.nominal.tilt.tilt_capped),
                        tilt_stalled=bool(self.nominal.tilt.tilt_stalled),
                        on_tube=bool(self.nominal.tilt.on_tube),
                        cop_r=self.nominal.tilt.cop_r)
                elif now-self._recovery_started_s>timeout:
                    raise RuntimeError('mechanical recovery requested; leave ordinary visual QP')
            elif self._recovery_started_s is not None:
                self.sink.emit('mechanical_recovery_cleared',control_id=self._control_id)
                self._recovery_started_s=None
            if bool(getattr(self.controller,'shield_uncertified_brake',False)):
                raise RuntimeError('uncertified_brake')
            if source_expired:
                self._publication_rejection_reason='wrench_source_expired'
                raise ProposalDeferred('wrench_source_expired')
            angle_reference_reset=bool(self.controller.physical_contact_acquire_event or self._angle_origin is None)
            if angle_reference_reset:
                self._angle_origin=rotation.copy()
            relative_rotvec=Rotation.from_matrix(self._angle_origin.T @ rotation).as_rotvec()
            measured_angle=float(relative_rotvec @ self.geometry.T_tcp_face[:3,1]) if self._delayed else float(relative_rotvec[1])
            path=nominal.copy();path[[2,4]]=0.;self.pending_basis=motion_basis(path)
            if self._delayed:
                observation,image_reason=delayed_observation,delayed_image_reason
            elif not self._image_owned:
                observation,image_reason=self._image_observation(now,
                    contact_enabled=self.contact_gate is None or self.contact_gate.started,
                    observation=image_snapshot)
            image_valid=observation is not None
            fusion_input={};supply_input={}
            if self._image_owned:
                loading,scan=task_components(nominal)
                cp,cp_reason=cop_preference(f_ext,self.geometry,
                    contact=bool(self.controller.contact_present),minimum_force_n=self.nominal.tilt.cfg.contact_n)
                if self.solver.config.allocation_policy=='confidence_angular_v1':cp=0.;cp_reason='angular_only_control'
                fusion_input=dict(visual_omega_target_rad_s=float(nominal[4]),
                    loading_velocity_m_s=float(loading[2]),cop_m=cp)
                supply_input=dict(loading_twist_tool=loading,scan_twist_tool=scan)
                self._pending_fusion.update(cop_preference_m=cp,cop_reason=cp_reason,
                    loading_requested_tool=loading,scan_requested_tool=scan,
                    visual_after_nominal_limits_rad_s=float(nominal[4]))
            self._pending_alpha_target=None;self._pending_alpha_preferred=None
            if self._quality_progress is not None and not self._delayed:
                self._pending_alpha_target,self._pending_alpha_preferred=self._quality_progress.preview(observation,actual_dt)
            mechanical_rows=TwistConstraints()
            acceleration_dt=actual_dt
            if self._delayed:
                tf=twist_tcp_to_face(self.geometry)
                nominal_contact=tf @ nominal
                ff_contact=tf @ np.asarray(self.baseline.last_path_twist)
                fb_contact=tf @ np.asarray(self.baseline.last_feedback_twist)
                self.pending_offset,self.pending_basis=affine_contact_motion(self.geometry,fb_contact,ff_contact)
                region_mask=(None if image_snapshot is None or not self._region_layout_compatible(image_snapshot)
                             else image_snapshot.region_valid)
                region_task=region_visual_request(self._pending_prediction.quality_raw,region_mask,
                    self.feature_config.region_edges,self.geometry,c_min=self.solver.config.c_min,
                    repair_speed_m_s=self.solver.config.repair_speed_m_s,
                    evidence_valid=self._pending_prediction.valid)
                request=region_task.request_rad_s;alpha_des=region_task.alpha_preferred
                if self._quality_intervals is not None:
                    request,alpha_des=self._quality_intervals.apply_escalation(
                        request,alpha_des,repair_speed_m_s=self.solver.config.repair_speed_m_s)
                self._pending_alpha_target=self._pending_alpha_preferred=alpha_des
                # Existing control wrench is tool-on-environment. Convert all
                # six axes together, then rotate/shift once to the contact face.
                env_contact=wrench_tcp_to_face(-np.asarray(f_ext,dtype=float),self.geometry)
                cp,cp_reason=contact_cop_from_wrench(env_contact,min_force_n=self.solver.config.cop_min_force_n,
                    max_abs_m=min(self.geometry.half_length_m,self.solver.config.cop_max_m))
                fusion_input=dict(mechanical_normal_m_s=float(nominal_contact[2]),
                    mechanical_omega_rad_s=float(nominal_contact[4]),path_feedback_contact=fb_contact,
                    path_feedforward_contact=ff_contact,visual_request_rad_s=request,
                    visual_task_valid=region_task.valid,
                    visual_gamma=self.solver.config.differential_repair.force_gate(contact_force),cop_m=cp)
                self._pending_fusion=dict(prediction=self._pending_prediction.to_dict(),
                    region_task=region_task.to_dict(),region_feature_version=REGION_FEATURE_VERSION,
                    region_layout_version=self.feature_config.region_layout_version,
                    region_count=self.feature_config.region_count,
                    visual_request_rad_s=request,cop_preference_m=cp,cop_reason=cp_reason,
                    environment_wrench_contact=env_contact,wrench_boundary='environment_contact = transform(-control_tcp)',
                    mechanical_nominal_contact=nominal_contact,force_task_frame='contact_face',
                    filtered_contact_force_n=contact_force,raw_contact_force_n=contact_force_raw,
                    additional_stop_frame='tcp_tool')
                if self._rocking is not None:
                    axis,bounds,history=self._rocking.preview(rotation @ self.geometry.T_tcp_face[:3,:3],now,actual_dt)
                    self._pending_rocking=(axis,bounds,history)
                    low=float(np.max(bounds[::2]));high=float(np.min(bounds[1::2]))
                    if low>high:raise RuntimeError('published rocking acceleration/jerk intersection empty')
                    mechanical_rows=TwistConstraints(tf[[4]], [low], [high],valid_until_s=now+self.config['command']['max_interval_s'],
                        labels=('published_contact_rocking_speed_acc_jerk',))
                    # Rocking elapsed_s is only the contact-Y publication interval.
                    # Six-axis command slew stays on the control step: a 0.6 ms
                    # rocking gap after retract/MoveJ made seek QP empty.
            self.pending_wrench_environment=None if f_ext_raw is None else np.asarray(f_ext_raw,dtype=float).copy()
            energy_snapshot=None
            if self.command_budget is not None:
                energy_snapshot=self.command_budget.snapshot(now_s=now,wrench_control_raw=f_ext_raw,
                    rotation_base_tcp=self.pending_rotation_base_tcp,
                    **({'nominal_twist_tool':nominal} if self.config.get('energy',{}).get('task_power_source','none')=='nominal_command' else supply_input))
            elif self.energy is not None:
                # Completed measured intervals enter through observe_measured_port;
                # this mixed-time diagnostic twist is never relabeled as aligned.
                energy_snapshot=self.energy.snapshot(now_s=now,hold_s=dt,
                    wrench_environment=self.pending_wrench_environment,source_t_s=source.source_t_s,
                    **self._energy_parameters)
            if self._recovery_started_s is not None:
                self._pending_alpha_preferred=0.
                self._pending_alpha_target=0.
            deadline=source.source_t_s+self.source_clock.max_age_s
            if self.command_budget is not None and self.command_budget.active is not None:
                deadline=min(deadline,self.command_budget.active.expires_s)
            if self._command_lease is not None and self._command_lease.active is not None:
                deadline=min(deadline,self._command_lease.active.expires_s)
            self.pending_result=self.solver.solve(QpInput(self.geometry,nominal,path,
                float(-env_contact[2]) if self._delayed else float(np.sign(self.baseline.desired_force[2])*f_ext[2]),dt,now,
                observation=observation,previous_twist=previous,measured_angle=measured_angle,energy=energy_snapshot,
                acceleration_dt_s=acceleration_dt,mechanical=mechanical_rows,
                repair_execution_enabled=bool(self.controller.contact_present) and (self.contact_gate is None or bool(self.contact_gate.started)),
                repair_angle_reference_reset=angle_reference_reset,alpha_preferred=self._pending_alpha_preferred,**fusion_input),
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
                if self._delayed:
                    reasons.append(self._pending_fusion['region_task']['reason'])
                    if diagnostic.get('visual_shortfall_rad_s',0.)>self.solver.config.feasibility_tolerance:
                        reasons.append('outer_regional_visual_shortfall')
                elif self._image_owned:
                    reasons.extend(diagnostic.get('allocation_reason_codes',()))
                    reasons.append(self._pending_fusion['visual_reason'])
                    if self._pending_fusion['visual_force_gate']<1.:reasons.append('force_gate')
                    if self._pending_fusion['cop_reason'] not in ('valid','angular_only_control'):
                        reasons.append('cop_'+self._pending_fusion['cop_reason'])
                elif request<=0:reasons.append('request_zero_or_deadband')
                if diagnostic.get('differential_shortfall_m_s',0.)>self.solver.config.feasibility_tolerance:
                    reasons.append('outer_repair_shortfall')
                reasons.extend('outer_'+str(label) for label in diagnostic.get('active_hard_rows',())
                    if 'omega' in str(label) or 'angle' in str(label) or 'acceleration_4' in str(label) or 'velocity_4' in str(label))
                if diagnostic.get('energy_margin_power_w',float('inf'))<=self.solver.config.feasibility_tolerance:
                    reasons.append('command_energy_limit')
                if diagnostic.get('repair_force_gate',1.)<1.:reasons.append('force_gate')
                if self.pending_result.alpha+1e-8<self._pending_alpha_preferred:
                    reasons.append('outer_mechanical_progress_limit' if self._delayed else 'outer_mechanical_or_energy_progress_limit')
                self._quality_intervals.observe(observation,now,reasons)
            self.last_path_twist=self.pending_result.alpha*(self.pending_basis[:,2] if self._delayed else self.baseline.last_path_twist)
            self.last_feedback_twist=self.pending_offset.copy() if self._delayed else self.pending_result.alpha*self.baseline.last_feedback_twist
            self.sink.emit('control_sample',control_id=self._control_id,reference_s=self.reference_time_s,
                proposal_time_s=now,
                nominal_twist_tool=nominal,candidate_twist_tool=self.pending_result.qp_twist,
                previous_outer_command_tool=previous,previous_final_command_tool=previous_final,
                command_slew_dt_s=actual_dt,
                rotation_base_tcp=rotation,
                angular_slew_history='final_published_contact' if self._delayed else 'final_published_tool_y' if self._image_owned else 'outer_candidate',
                fusion_components=self._pending_fusion if self._image_owned or self._delayed else None,
                repair_episode=(dict(self.pending_result.diagnostics['repair_episode'])
                    if 'repair_episode' in self.pending_result.diagnostics else None),
                allocation_diagnostics={key:value for key,value in self.pending_result.diagnostics.items()
                    if key.startswith("differential_") or key in ("repair_force_gate","visual_window_active",
                        "repair_permission_mode","qp_delta_omega_y_rad_s") or self._image_owned or self._delayed},
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
                        None if self.energy is None else self.energy.facts),physical_certified=False,
                tilt_stop_reason=self.nominal.tilt.tilt_stop_reason,
                tilt_frozen=bool(self.nominal.tilt.tilt_frozen),
                tilt_capped=bool(self.nominal.tilt.tilt_capped),
                tilt_stalled=bool(self.nominal.tilt.tilt_stalled),
                tilt_on_tube=bool(self.nominal.tilt.on_tube),
                mechanical_recovery_active=self._recovery_started_s is not None)
            self._drain_command_energy()
            return self.pending_result.qp_twist.copy()
        except Exception:
            self.publication_abort('outer_prepare_failed',definitely_not_sent=True)
            raise

    def publication_review(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer publication')
        final=np.asarray(final_tool,dtype=float).reshape(6)
        result=self.pending_result
        mechanical_review={}
        def rejected(reason):
            self._publication_rejection_reason=reason
            self.sink.emit('publication_review_rejected',control_id=candidate_id,reason=reason,
                review_time_s=now_s,created_time_s=result.created_time_s,
                valid_until_s=result.hard_constraints.valid_until_s,
                final_command_model_tool=final,facts=facts or {},mechanical_review=mechanical_review,
                energy=None if self.command_budget is None else self.command_budget.facts)
            return False
        if not np.isfinite(final).all() or not np.isfinite(now_s):return rejected('nonfinite_payload_or_time')
        if now_s<result.created_time_s:return rejected('review_time_reversed')
        if self._rocking is not None:
            self._rocking_review_facts=dict(facts or {})
            rocking_tolerance=float(self._rocking_review_facts.get("rocking_tolerance_rad_s",self.solver.config.feasibility_tolerance))
            if not np.isfinite(rocking_tolerance) or rocking_tolerance<0:return rejected("invalid_rocking_tolerance")
            rocking_final=twist_tcp_to_face(self.geometry) @ final if self._delayed else final
            if not self._rocking.review(rocking_final,self._rocking_review_facts,rocking_tolerance):
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
        if self._delayed:
            if not 0<=now_s-self._source_step.source_t_s<self.source_clock.max_age_s:return rejected('wrench_source_expired')
            if now_s>=result.hard_constraints.valid_until_s:return rejected('certificate_expired')
            # The inner IK receives a later preview from rocking_constraints().
            # Validate that exact exported envelope and its accepted tier, then
            # supersede ONLY the old proposal-time rocking row. Applying both
            # envelopes rejected valid second SEEK commands in session 008.
            refreshed_rocking=(self._rocking is not None and self._pending_rocking is not None
                               and self._rocking_export_id==candidate_id)
            if refreshed_rocking:
                tier=int(self._rocking_review_facts['rocking_policy_tier'])
                if tier in (1,2,3):
                    pairs=self._pending_rocking[1].reshape(3,2)[:4-tier]
                    expected=np.array([np.max(pairs[:,0]),np.min(pairs[:,1])])
                    reported=np.array([self._rocking_review_facts['rocking_lower_rad_s'],
                                       self._rocking_review_facts['rocking_upper_rad_s']])
                    if not np.allclose(expected,reported,rtol=0.,atol=1e-12):
                        return rejected('final_rocking_certificate_mismatch')
            command_hard=self._final_command_constraints()
            mechanical=np.array([not label.startswith(('affine_motion_subspace_','progress_range'))
                for label in command_hard.labels],dtype=bool)
            superseded=[]
            if refreshed_rocking:
                contact_y=twist_tcp_to_face(self.geometry)[4]
                for index,label in enumerate(command_hard.labels):
                    if (label=='published_contact_rocking_speed_acc_jerk' and
                            np.allclose(command_hard.A[index],contact_y,rtol=0.,atol=1e-12)):
                        mechanical[index]=False;superseded.append(label)
            values=command_hard.A[mechanical] @ final
            lo=command_hard.lower[mechanical];hi=command_hard.upper[mechanical]
            slew_dt=max(self.pending_actual_dt,self.pending_dt)
            allowance=np.abs(command_hard.A[mechanical]) @ (self.solver.config.max_acceleration*slew_dt)
            excess=np.maximum(lo-values,values-hi)
            violation_i=np.maximum(0.,excess)
            violation=float(np.max(violation_i,initial=0.))
            labels=np.asarray(command_hard.labels)[mechanical]
            failures=excess>self.solver.config.feasibility_tolerance
            mechanical_review=dict(superseded_proposal_rows=superseded,max_violation=violation,
                allowance=[float(x) for x in allowance],
                violated_rows=[dict(label=str(label),value=float(value),lower=float(low),upper=float(high))
                              for label,value,low,high in zip(labels[failures],values[failures],lo[failures],hi[failures])])
            if np.any(violation_i>allowance):return rejected('final_mechanical_interval_violation')
            if (np.any(violation_i>self.solver.config.feasibility_tolerance)
                    and self._quality_intervals is not None):
                self._quality_intervals.add_reasons(['final_outside_outer_envelope'])
            if not self._command_lease.reserve(candidate_id,created_s=result.created_time_s,now_s=now_s):
                return rejected('command_lease_reservation_rejected')
            self._reserved_id=candidate_id
        elif self.command_budget is not None:
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
            mechanical_review=mechanical_review,
            task_hard_violation=result.hard_constraints.violation(final),task_certificate=False,
            physical_certified=False,review_time_s=now_s,
            dispatch_valid_until_s=self._source_step.source_t_s+self.source_clock.max_age_s,
            exported_task_valid_until_s=result.hard_constraints.valid_until_s,facts=facts or {})
        return True

    def _accepted_progress_alpha(self,projected,denom):
        """Keep the stop-ramp clock moving when the path column has vanished."""
        outer=0. if self.pending_result is None else float(self.pending_result.alpha)
        if denom<=1e-28:
            return outer if self.pending_h_ref>0 else 0.
        clipped=float(np.clip(projected,0.,outer))
        if clipped>self.solver.config.feasibility_tolerance:
            return clipped
        if self.pending_h_ref<=0:
            return 0.
        adapter=self.reference
        local=float(adapter.local_time(self.reference_time_s))
        ramp=_reference_ramp_s(adapter)
        if ramp>0. and local>=float(adapter.duration_s)-ramp:
            return outer
        return 0.

    def publication_started(self,candidate_id):
        if candidate_id!=self._pending_id:raise RuntimeError('orphan publication start')
        budget=self.command_budget or self._command_lease
        if budget is not None:
            # Last gate before the first device send. A late review/logging tick
            # must be rejected while the rail reservation is still abortable.
            now=time.monotonic()
            pending=budget.pending
            valid=(self._review_time_s is not None and self._review_time_s<=now and
                self._dispatch_time_s is None and self._reserved_id==candidate_id and
                pending is not None and pending.command_id==candidate_id and now<pending.expires_s and
                self.pending_result.created_time_s<=now)
            source_age=now-self._source_step.source_t_s
            reason=('missing_review_or_invalid_dispatch' if not valid else
                    'dispatch_certificate_expired' if self._delayed and now>=self.pending_result.hard_constraints.valid_until_s else
                    'dispatch_source_expired' if source_age>=self.source_clock.max_age_s else
                    'dispatch_source_future' if not source_age>=0 else None)
            if reason is not None:
                self._publication_rejection_reason=reason
                self.sink.emit('publication_dispatch_rejected',control_id=candidate_id,
                    dispatch_time_s=now,reason=reason,
                    source_time_s=self._source_step.source_t_s,
                    dispatch_valid_until_s=self._source_step.source_t_s+self.source_clock.max_age_s)
                return False
            if self._delayed:
                if not budget.publication_started(candidate_id,now_s=now):
                    self._publication_rejection_reason='command_lease_dispatch_rejected'
                    return False
            else:budget.publication_started(candidate_id)
            self._dispatch_time_s=now
        return True

    def publication_commit(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer commit')
        self._lease_expiry_retries=0
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
        if self._delayed:
            if self._dispatch_time_s is None:
                raise RuntimeError('commit without valid dispatch')
            self._command_lease.commit(candidate_id,now_s=now_s,
                dual_success=bool(facts and facts.get('arm')=='sent' and facts.get('rail') in ('sent','disabled')))
        path=self.pending_basis[:,2];denom=float(path @ path)
        projected=0. if denom<=1e-28 else float(path @ final/denom)
        alpha=self._accepted_progress_alpha(projected,denom)
        if self._delayed:
            tolerance=self.solver.config.subspace_tolerance
            coefficients=np.linalg.lstsq(self.pending_basis/tolerance[:,None],
                (final-self.pending_offset)/tolerance,rcond=None)[0]
            projected=0. if denom<=1e-28 else float(coefficients[2])
            alpha=self._accepted_progress_alpha(projected,denom)
            self.sink.emit('affine_final_realisation',control_id=candidate_id,
                realised_coefficients=coefficients,affine_offset_tcp=self.pending_offset,
                affine_residual_tool=final-self.pending_offset-self.pending_basis @ coefficients,
                final_contact_twist=twist_tcp_to_face(self.geometry) @ final,
                candidate_contact_twist=twist_tcp_to_face(self.geometry) @ self.pending_result.qp_twist)
        # These are command integrators. The rail-compensated payload model
        # is not a newly requested outer action and must not rebase them.
        accepted_outer=self.pending_result.qp_twist
        force=np.zeros(6);force[[2,4]]=accepted_outer[[2,4]]
        if self._delayed:
            # The independent mechanical proposal advances only after success.
            # Neither QP visual/CoP increments nor rail compensation becomes its
            # next admittance drive. Publication history below remains separate.
            mechanical=self._pending_mechanical_contact
            force[:]=0.;force[[2,4]]=mechanical[[2,4]]
            self.nominal.commit_applied(force,final_full_twist=mechanical,
                accepted_normal_z=float(mechanical[2]))
            accepted_omega=float((twist_tcp_to_face(self.geometry)@final)[4])
            self.nominal.tilt.note_executed_omega(accepted_omega,self.pending_actual_dt)
        elif self._image_owned:
            requested_z=float(self._pending_fusion['loading_requested_tool'][2])
            cp=float(self._pending_fusion['cop_preference_m'])
            loading_z=accepted_loading(accepted_outer,requested_z,cp)
            force[2]=loading_z;force[4]=final[4]
            pure=alpha*path.copy();pure[2]=loading_z
            self.nominal.commit_applied(force,final_full_twist=pure,accepted_normal_z=loading_z)
            self._visual_task.commit(final,self.pending_rotation_base_tcp)
            self.sink.emit('fusion_publication',control_id=candidate_id,
                loading_accepted_outer_z_m_s=loading_z,
                geometric_candidate_z_m_s=float(accepted_outer[2]-loading_z),
                visual_final_omega_rad_s=float(final[4]),cop_preference_m=cp,
                cop_candidate_residual_m_s=float(accepted_outer[2]-requested_z-cp*accepted_outer[4]),
                cop_final_model_residual_m_s=float(final[2]-requested_z-cp*final[4]),
                final_minus_candidate_tool=final-accepted_outer,
                final_model_is_measurement=False,loading_state_uses_rail_model=False)
        else:
            self.nominal.commit_applied(force,final_full_twist=accepted_outer)
        timely = self.command_budget is not None or self._delayed or (np.isfinite(now_s) and
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
            rocking_final=twist_tcp_to_face(self.geometry) @ final if self._delayed else final
            rocking_rotation=(self.pending_rotation_base_tcp @ self.geometry.T_tcp_face[:3,:3]
                              if self._delayed else self.pending_rotation_base_tcp)
            audit=self._rocking.publication_audit(rocking_final,rocking_rotation,now_s,
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
            if self._delayed:
                target=float(diag['visual_request_gated_rad_s'])
                achieved=float((twist_tcp_to_face(self.geometry) @ final)[4])
                shortfall=max(abs(target)-np.sign(target)*achieved,0.) if diag['visual_rows_active'] else 0.
                if shortfall>self.solver.config.feasibility_tolerance:reasons.append('final_regional_visual_shortfall')
                self.sink.emit('final_regional_visual_task',control_id=candidate_id,
                    region_count=self.feature_config.region_count,region_layout_version=self.feature_config.region_layout_version,
                    gated_request_rad_s=target,final_contact_omega_rad_s=achieved,
                    shortfall_rad_s=shortfall,diagnostic_only=True)
            if self._image_owned:
                target=float(diag['visual_omega_target_rad_s'])
                if abs(final[4]-target)>self.solver.config.feasibility_tolerance:
                    reasons.append('final_visual_target_deviation')
            if alpha+1e-8<self.pending_result.alpha:reasons.append('inner_progress_limit')
            self._quality_intervals.add_reasons(reasons)
            if self._endpoint_started_s is None and self.reference.exhaustion_reason(self.reference_time_s):
                self._endpoint_started_s=now_s
                self.sink.emit('reference_complete',time_s=now_s,reference_s=self.reference_time_s,phase='endpoint_convergence')
                _rt_print('[CONTACT_QP] reference complete; endpoint convergence.')
        self._previous=self.pending_result.qp_twist.copy()
        self._previous_final=final.copy()
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
        if self._command_lease is not None:
            self._command_lease.reject_new_only(definitely_not_sent=definitely_not_sent,now_s=time.monotonic())
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
        if self._command_lease is not None:self._command_lease.stop(now_s=time.monotonic())
        self._drain_command_energy()
        self.sink.emit('stop',reason=reason,reference_s=self.reference_time_s,physical_execution='unconfirmed')

    def close(self,*,wait=True):
        self.features.close(wait=wait);self.sink.close(wait=wait)

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
from peirastic.contact_qp.types import ProbeGeometry,ContactStatus
from peirastic.contact_qp.geometry import motion_basis
from peirastic.contact_qp.features import FeatureConfig
from peirastic.contact_qp.reference import FiniteIntervalReference
from peirastic.contact_qp.runtime_source import SourceClock
from peirastic.contact_qp.runtime_config import validate_study_config,calibrated_geometry
from peirastic.contact_qp.energy import EnergyLedger,PortBounds
from peirastic.contact_qp.runtime_energy import RuntimeEnergy
from .contact_recording import ContactRecordSink,FeatureReceiver,clock_metadata,study_fingerprints


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
        source=config.get('source') or {}
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
        geometry,_=calibrated_geometry(config)
        if config['geometry']['face_normal_convention']=='outward':
            # Preserve face X and convert the declared outward +Z into the
            # solver's into-contact +Z with a proper rigid rotation.
            geometry=replace(geometry,T_tcp_face=geometry.T_tcp_face @ np.diag([1.,-1.,-1.,1.]))
        self.geometry=geometry
        feature=config['feature'];self.feature_config=FeatureConfig(**feature['config'])
        self.registration_version=feature['registration_version']
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
        energy=config.get('energy') or {}
        enabled=config.get('energy_constraint_enabled',False)
        if type(enabled) is not bool:raise ValueError('energy_constraint_enabled must be boolean')
        if enabled and config.get('physical_w_checked') is not True:
            raise ValueError('command energy needs checked environment-on-tool wrench semantics')
        self.energy=None
        if energy:
            bounds=PortBounds(**dict(energy.get('measurement_bounds') or {}),verified=False)
            ledger=EnergyLedger(energy['initial_j'],energy['capacity_j'],energy['stopping_reserve_j'],bounds)
            self.energy=RuntimeEnergy(ledger,command_budget_enforced=enabled,
                                      max_measurement_age_s=energy.get('max_measurement_age_s',self.source_clock.max_age_s))
        elif enabled:raise ValueError('enabled command energy requires an explicit single-tank configuration')
        self._energy_parameters=dict(energy.get('constraint') or {})
        self._port_aligner=None
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
        self.features=feature_receiver or FeatureReceiver(config['feature_endpoint'])
        self.reference_time_s=0.;self._control_time_s=0.;self._control_id=0;self.arrival_confirmed=False
        self._source_step=None;self._source_prepared=False;self._angle_origin=None
        self._last_velocity_time=None;self._previous=np.zeros(6);self._previous_rotation=None;self.pending_result=None
        self._pending_id=None;self._nominal_pending=False;self._reserved_id=None;self._stop_recorded=False
        self.last_path_twist=np.zeros(6);self.last_feedback_twist=np.zeros(6)
        self.sink.emit('study_start',mode='active',command_authority='outer_qp_original_ik',
            physical_port_assurance='unverified',task_certificate=False,config=config,
            **clock_metadata(),**study_fingerprints(baseline,config))

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

    def prepare_source(self,source_id,source_t_s,wall_time_ns,*,now_s):
        self._source_step=self.source_clock.observe(source_id,source_t_s,wall_time_ns,now_s=now_s)
        self._source_prepared=True
        return self._source_step

    def observe_measured_port(self, snap, raw_rail_feedback, wrench_tcp, kin, *, now_s, source_id):
        """Measurement-only ingress, independent of the current command proposal."""
        if self.energy is None:return
        intervals=self._port_aligner.update(source_id=source_id,
            source_t_s=getattr(snap,'t_s',float('nan')),
            arm_q_rad=np.deg2rad(np.asarray(getattr(snap,'q_deg',[]),dtype=float)),
            wrench_tcp=wrench_tcp,source_valid=bool(getattr(snap,'ok',False)),
            rail_feedback=raw_rail_feedback,kin=kin,now_s=now_s)
        for interval,provenance in intervals:
            self.energy.observe_interval(interval,now_s=now_s,
                physical_w_checked=self._physical_w_checked,provenance=provenance)
        for event in self._port_aligner.drain_events():
            self.sink.emit('port_alignment',facts=event)
        events=self.energy.drain_events()
        if events['runtime'] or events['ledger']:
            self.sink.emit('energy_measurement',**events)

    def sample(self,t_s,current_pose,f_ext,*,contact=None,f_ext_raw=None,dt_actual=None,
               sensor_age_s=None,feedback_age_s=None,feedback_fresh_tick=None,
               feedback_velocity_valid=None,v_tcp_z_actual=None,slack_norm=None,
               measured_twist_base=None,measured_twist_valid=None,measured_twist_fresh=None,
               measured_twist_metadata=None,measurement_time_s=None,measurement_id=None,
               wrench_source_time_s=None,wrench_source_wall_time_ns=None,wrench_source_id=None):
        if self._nominal_pending:raise RuntimeError('previous outer proposal remains unresolved')
        now=time.monotonic();actual_dt=float(self.baseline.dt if dt_actual is None else dt_actual)
        if not np.isfinite(actual_dt) or actual_dt <= 0: raise ValueError("invalid control interval")
        dt=min(actual_dt,float(self.baseline.dt))  # next command hold/reference grant model only
        if not self._source_prepared:
            self.prepare_source(wrench_source_id,wrench_source_time_s,wrench_source_wall_time_ns,now_s=now)
        source=self._source_step;self._source_prepared=False
        if now-source.source_t_s>self.source_clock.max_age_s:raise RuntimeError('source hold age expired')
        self._control_id+=1;self._pending_id=self._control_id;self._control_time_s+=actual_dt
        self.pending_dt=dt
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
        ref,h_ref=self.reference.candidate(self.reference_time_s,dt)
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
            if self.nominal.tilt.needs_normal_retract:
                raise RuntimeError('mechanical recovery requested; leave ordinary visual QP')
            angle_reference_reset=bool(self.controller.physical_contact_acquire_event or self._angle_origin is None)
            if angle_reference_reset:
                self._angle_origin=rotation.copy()
            measured_angle=float(Rotation.from_matrix(self._angle_origin.T @ rotation).as_rotvec()[1])
            path=nominal.copy();path[[2,4]]=0.;self.pending_basis=motion_basis(path)
            observation=self.features.observation
            expected=(self.registration_version,self.feature_config.window_version,self.feature_config.calibration_version)
            image_valid=observation is not None and observation.version==expected
            if not image_valid:observation=None
            self.pending_wrench_environment=None if f_ext_raw is None else np.asarray(f_ext_raw,dtype=float).copy()
            energy_snapshot=None
            if self.energy is not None:
                # Completed measured intervals enter through observe_measured_port;
                # this mixed-time diagnostic twist is never relabeled as aligned.
                energy_snapshot=self.energy.snapshot(now_s=now,hold_s=dt,
                    wrench_environment=self.pending_wrench_environment,source_t_s=source.source_t_s,
                    **self._energy_parameters)
            self.pending_result=self.solver.solve(QpInput(self.geometry,nominal,path,
                float(np.sign(self.baseline.desired_force[2])*f_ext[2]),dt,now,
                observation=observation,previous_twist=previous,measured_angle=measured_angle,energy=energy_snapshot,
                acceleration_dt_s=actual_dt,
                repair_execution_enabled=bool(self.controller.contact_present) and (self.contact_gate is None or bool(self.contact_gate.started)),
                repair_angle_reference_reset=angle_reference_reset))
            if self.pending_result.qp_twist is None:
                self.sink.emit('qp_prepare_rejected',control_id=self._control_id,
                    nominal_twist_tool=nominal,path_twist_tool=path,previous_twist_tool=previous,
                    force_n=float(f_ext[2]),control_actual_dt_s=actual_dt,command_hold_model_s=dt,
                    measured_angle_rad=measured_angle,diagnostics=dict(self.pending_result.diagnostics))
                raise RuntimeError('outer QP: '+str(self.pending_result.status.value)+': '+str(self.pending_result.diagnostics.get('reason')))
            self.last_path_twist=self.pending_result.alpha*self.baseline.last_path_twist
            self.last_feedback_twist=self.pending_result.alpha*self.baseline.last_feedback_twist
            self.sink.emit('control_sample',control_id=self._control_id,reference_s=self.reference_time_s,
                nominal_twist_tool=nominal,candidate_twist_tool=self.pending_result.qp_twist,
                previous_outer_command_tool=previous,command_slew_dt_s=actual_dt,
                repair_episode=self.pending_result.diagnostics.get("repair_episode"),
                allocation_diagnostics={key:value for key,value in self.pending_result.diagnostics.items()
                    if key.startswith("differential_") or key in ("repair_force_gate","visual_window_active")},
                alpha=self.pending_result.alpha,h_ref_s=h_ref,command_hold_model_s=dt,
                control_actual_dt_s=actual_dt,source=source.__dict__,
                feature=None if observation is None else observation.to_dict(),image_compatible=image_valid,
                control_wrench_tool=f_ext,physical_wrench_candidate_tool=f_ext_raw,
                measured_twist_base=measured_twist_base,measured_twist_metadata=metadata,
                measured_twist_valid=measured_twist_valid,measurement_time_s=measurement_time_s,
                energy=None if self.energy is None else self.energy.facts,physical_certified=False)
            return self.pending_result.qp_twist.copy()
        except Exception:
            self.publication_abort('outer_prepare_failed',definitely_not_sent=True)
            raise

    def publication_review(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer publication')
        final=np.asarray(final_tool,dtype=float).reshape(6)
        result=self.pending_result
        def rejected(reason):
            self.sink.emit('publication_review_rejected',control_id=candidate_id,reason=reason,
                review_time_s=now_s,created_time_s=result.created_time_s,
                valid_until_s=result.hard_constraints.valid_until_s,
                final_command_model_tool=final,facts=facts or {})
            return False
        if not np.isfinite(final).all() or not np.isfinite(now_s):return rejected('nonfinite_payload_or_time')
        if now_s<result.created_time_s:return rejected('review_time_reversed')
        if now_s>=result.hard_constraints.valid_until_s:return rejected('certificate_expired')
        if self.energy is not None and self.energy.command_budget_enforced:
            if not self.energy.reserve(candidate_id,result.energy_certificate,final,now_s=now_s):return rejected('energy_reservation_rejected')
            self._reserved_id=candidate_id
        self.sink.emit('publication_review',control_id=candidate_id,final_command_model_tool=final,
            task_hard_violation=result.hard_constraints.violation(final),task_certificate=False,
            physical_certified=False,facts=facts or {})
        return True

    def publication_started(self,candidate_id):
        if candidate_id!=self._pending_id:raise RuntimeError('orphan publication start')
        if self._reserved_id is not None:self.energy.publication_started(candidate_id)

    def publication_commit(self,candidate_id,final_tool,*,now_s,facts=None):
        if candidate_id!=self._pending_id or not self._nominal_pending:raise RuntimeError('orphan outer commit')
        final=np.asarray(final_tool,dtype=float).reshape(6)
        path=self.pending_basis[:,2];denom=float(path @ path)
        alpha=0. if denom<=1e-28 else float(np.clip(path @ final/denom,0.,self.pending_result.alpha))
        # These are command integrators. The rail-compensated payload model
        # is not a newly requested outer action and must not rebase them.
        accepted_outer=self.pending_result.qp_twist
        force=np.zeros(6);force[[2,4]]=accepted_outer[[2,4]]
        self.nominal.commit_applied(force,final_full_twist=accepted_outer)
        timely = np.isfinite(now_s) and self.pending_result.created_time_s <= now_s < self.pending_result.hard_constraints.valid_until_s
        if not timely: alpha=0.
        if self.pending_h_ref>0:
            self.reference_time_s=self.reference.commit_time(self.reference_time_s,alpha,self.pending_dt)
            if self.contact_gate is not None:
                self.contact_gate._elapsed_s=self.reference.local_time(self.reference_time_s)
                self.contact_gate._last_input_t_s=self.reference_time_s
        # Commit only on dual-device publication success. Final-model residuals
        # remain in energy/progress accounting and logs; they cannot become
        # nominal or outer command-slew history in a different space.
        self._previous=self.pending_result.qp_twist.copy()
        self._previous_rotation=self.pending_rotation_base_tcp.copy()
        self._nominal_pending=False;self._reserved_id=None
        self.sink.emit('publication',control_id=candidate_id,success=True,accepted_alpha=alpha,
            reference_s=self.reference_time_s,final_command_model_tool=final,
            task_certificate=False,physical_certified=False,facts=facts or {})

    def publication_abort(self,reason,*,definitely_not_sent=False,facts=None):
        # A delegated sample can raise after preparing only one nominal part.
        # Retire each existing proposal without rolling back consumed measurements.
        if self.nominal.z_law._transaction.pending is not None:
            self.nominal.z_law.abort()
        if self.nominal.tilt._pending_command is not None:
            self.nominal.tilt.abort()
        self._nominal_pending=False
        if self._reserved_id is not None:
            self.energy.reject_new_only(self._reserved_id,definitely_not_sent=definitely_not_sent)
            self._reserved_id=None
        self.sink.emit('publication_aborted',control_id=self._pending_id,reason=reason,
                       reference_s=self.reference_time_s,facts=facts or {})

    def arrived(self,pose):
        if not self.reference.exhaustion_reason(self.reference_time_s):return False
        target=self.reference.reference.sample(self.reference.duration_s).pose_d
        rotation=Rotation.from_euler(self.controller.cfg.euler_order,np.asarray(pose)[3:]).as_matrix()
        return bool(np.linalg.norm((rotation.T @ (np.asarray(pose)[:3]-target[:3]))*self.baseline.selection[:3])<=1e-4)

    def record_stop(self,reason):
        if self._stop_recorded:return
        self._stop_recorded=True
        self.publication_abort(reason)
        self.sink.emit('stop',reason=reason,reference_s=self.reference_time_s,physical_execution='unconfirmed')

    def close(self,*,wait=True):
        self.features.close(wait=wait);self.sink.close(wait=wait)

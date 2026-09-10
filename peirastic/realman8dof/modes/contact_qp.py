"""Opt-in recording/shadow adapter around the unchanged ICRA TFF outer.

Shadow suggestions run on a separate worker and are never returned to IK.
Unconfigured geometry or image registration is recorded explicitly; synthetic
plant geometry is never substituted for a physical probe.
"""
from __future__ import annotations
from dataclasses import fields
import inspect
import json
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Event,Thread
import time
import uuid
import numpy as np
from .contact_recording import ContactRecordSink,FeatureReceiver,clock_metadata,study_fingerprints


class ShadowSuggestions:
    def __init__(self,config,sink):
        from peirastic.contact_qp.qp import QpConfig,ContactQp
        from peirastic.contact_qp.types import ProbeGeometry
        self.sink=sink;self.config=config;self.queue=Queue(maxsize=1);self.dropped=0
        self._finish=Event();self.geometry=None;self.unavailable='geometry_not_declared'
        raw=config.get('geometry')
        if raw is not None:
            required=('half_length_m','T_tcp_face','image_x_sign','calibration_version')
            if all(raw.get(key) is not None for key in required):
                self.geometry=ProbeGeometry(**{k:v for k,v in raw.items() if k in ProbeGeometry.__dataclass_fields__})
                self.unavailable=''
            else:self.unavailable='geometry_fields_missing'
        qp=dict(config.get('qp') or {})
        qp.setdefault('quality_policy_version','unverified_real_study_policy')
        self.feature_config=None;self.registration_version=None
        feature=config.get('feature') or {}
        if all(feature.get(k) is not None for k in ('config','window_version','registration_version','c_min','quality_policy_version')):
            from peirastic.contact_qp.features import FeatureConfig
            self.feature_config=FeatureConfig(**feature['config'])
            if self.feature_config.window_version!=feature['window_version']:
                raise ValueError('feature window version does not match declared worker configuration')
            self.registration_version=feature['registration_version']
            qp.update(c_min=feature['c_min'],quality_policy_version=feature['quality_policy_version'],
                      lateral_windows=self.feature_config.lateral_windows)
        self.qp_config=QpConfig(**qp);self.solver=ContactQp(self.qp_config)
        self._thread=Thread(target=self._run,name='contact-study-shadow',daemon=True)
        self._thread.start()

    def submit(self,record):
        try:self.queue.put_nowait(record)
        except Full:
            try:self.queue.get_nowait()
            except Empty:pass
            self.dropped+=1
            try:self.queue.put_nowait(record)
            except Full:self.dropped+=1

    def _run(self):
        from peirastic.contact_qp.qp import QpInput
        while not self._finish.is_set():
            try:item=self.queue.get(timeout=.05)
            except Empty:continue
            output=dict(control_id=item['control_id'],sample_monotonic_s=item['now_s'],
                        assurance='unverified',shadow_queue_dropped=self.dropped,
                        geometry_calibration=None if self.geometry is None else self.geometry.calibration_version,
                        quality_policy_version=self.qp_config.quality_policy_version)
            try:
                if self.geometry is None:
                    output.update(status='unavailable',reason=self.unavailable)
                elif self.feature_config is None:
                    output.update(status='unavailable',reason='feature_policy_not_declared')
                elif (item['observation'] is None or item['observation'].version !=
                      (self.registration_version,self.feature_config.window_version,self.feature_config.calibration_version)
                      or self.feature_config.calibration_version!=self.geometry.calibration_version
                      or self.feature_config.image_x_sign!=self.geometry.image_x_sign):
                    output.update(status='unavailable',reason='feature_registration_mismatch_or_missing')
                elif item['measured_angle'] is None:
                    output.update(status='unavailable',reason='relative_angle_unavailable')
                else:
                    result=self.solver.solve(QpInput(self.geometry,item['nominal'],item['path'],item['force_n'],
                        item['dt_s'],item['now_s'],observation=item['observation'],previous_twist=item['previous'],
                        measured_angle=item['measured_angle']))
                    output.update(status=result.status.value,reason=result.diagnostics.get('reason',''),
                                  candidate_twist_tool=result.qp_twist,alpha=result.alpha,
                                  slack=result.slack,diagnostics=dict(result.diagnostics))
                self.sink.emit('shadow_candidate',**output)
            except Exception as exc:
                output.update(status='unavailable',reason=str(exc))
                self.sink.emit('shadow_candidate',**output)

    def close(self,*,wait=True):
        self._finish.set()
        if wait:self._thread.join(timeout=.5)


class ContactStudyOuter:
    """Delegate the exact old sample once; record independent proposed/sent facts."""
    contact_study_enabled=True
    def __init__(self,baseline,config,*,sink=None,feature_receiver=None):
        self.baseline=baseline;self.config=dict(config);self.mode=str(config.get('mode','baseline'))
        if self.mode not in ('baseline','shadow'):
            raise ValueError('recording outer mode must be baseline or shadow')
        path=config.get('log_path')
        if path is None:
            path=Path('apps/logs/contact_qp')/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl')
        self.sink=sink or ContactRecordSink(path)
        self.features=feature_receiver
        if self.features is None and config.get('feature_endpoint'):
            self.features=FeatureReceiver(config['feature_endpoint'])
        self.shadow=ShadowSuggestions(config,self.sink) if self.mode=='shadow' else None
        self._sample_keys=set(inspect.signature(baseline.sample).parameters)
        self._control_id=0;self._previous=np.zeros(6);self._last_record_id=None;self._closed=False
        self._shadow_source=None
        self._relative_rotation_origin=None;self._stop_recorded=False
        self.sink.emit('study_start',mode=self.mode,config=config,
                       command_authority='unchanged_baseline',physical_port_assurance='unverified',
                       **clock_metadata(),**study_fingerprints(baseline,config))

    def __getattr__(self,name):return getattr(self.baseline,name)

    def sample(self,t_s,current_pose,f_ext,*,contact=None,f_ext_raw=None,dt_actual=None,
               sensor_age_s=None,feedback_age_s=None,feedback_fresh_tick=None,
               feedback_velocity_valid=None,v_tcp_z_actual=None,slack_norm=None,
               measured_twist_base=None,measured_twist_valid=None,measured_twist_fresh=None,
               measured_twist_metadata=None,measurement_time_s=None,measurement_id=None,
               wrench_source_time_s=None,wrench_source_wall_time_ns=None,wrench_source_id=None):
        kwargs=dict(contact=contact,f_ext_raw=f_ext_raw,dt_actual=dt_actual,sensor_age_s=sensor_age_s,
                    feedback_age_s=feedback_age_s,feedback_fresh_tick=feedback_fresh_tick,
                    feedback_velocity_valid=feedback_velocity_valid,v_tcp_z_actual=v_tcp_z_actual,slack_norm=slack_norm)
        nominal=self.baseline.sample(t_s,current_pose,f_ext,**{k:v for k,v in kwargs.items() if k in self._sample_keys})
        now=time.monotonic();self._control_id+=1;self._last_record_id=self._control_id
        observation=None if self.features is None else self.features.observation
        feature_fields=None if observation is None else observation.to_dict()
        source_key=(wrench_source_id,wrench_source_time_s)
        source_fresh=(wrench_source_id is not None and wrench_source_time_s is not None
                      and np.isfinite(wrench_source_time_s) and
                      (self._shadow_source is None or wrench_source_id!=self._shadow_source[0]
                       or wrench_source_time_s>self._shadow_source[1]))
        self.sink.emit('control_sample',control_id=self._control_id,mode=self.mode,reference_s=t_s,
            pose_tcp_base=current_pose,control_wrench_tool=f_ext,raw_control_wrench_tool=f_ext_raw,
            physical_wrench_candidate_tool=f_ext_raw,physical_wrench_tool=None,
            physical_port_status="monitor_unavailable",physical_port_reason="installation_and_timing_unverified",
            nominal_twist_tool=nominal,path_feedforward_tool=self.baseline.last_path_twist,
            feedback_twist_tool=self.baseline.last_feedback_twist,dt_execution_s=dt_actual,
            wrench_source_time_s=wrench_source_time_s,wrench_source_wall_time_ns=wrench_source_wall_time_ns,
            wrench_source_id=wrench_source_id,measurement_id=measurement_id,
            analysis_source_fresh=source_fresh,
            measured_twist_base=measured_twist_base,measured_twist_valid=measured_twist_valid,
            measured_twist_fresh=measured_twist_fresh,measurement_time_s=measurement_time_s,
            measured_twist_metadata=measured_twist_metadata,feature=feature_fields,
            feature_age_s=None if observation is None else now-observation.effective_time_s,
            feature_error=None if self.features is None else self.features.error)
        if source_fresh:self._shadow_source=source_key
        if self.shadow is not None and source_fresh:
            # Full position contribution includes accepted-clock feedback.
            path=np.asarray(nominal,dtype=float).copy();path[[2,4]]=0.
            from scipy.spatial.transform import Rotation
            rotation=Rotation.from_euler(self.baseline.controller.cfg.euler_order,np.asarray(current_pose)[3:]).as_matrix()
            if self._relative_rotation_origin is None or self.baseline.controller.physical_contact_acquire_event:
                self._relative_rotation_origin=rotation.copy()
            angle=float(Rotation.from_matrix(self._relative_rotation_origin.T @ rotation).as_rotvec()[1])
            self.shadow.submit(dict(control_id=self._control_id,now_s=now,nominal=np.asarray(nominal).copy(),path=path,
                force_n=float(f_ext[2]),dt_s=float(dt_actual or self.baseline.dt),observation=observation,
                previous=self._previous.copy(),measured_angle=angle))
        self._previous=np.asarray(nominal).copy()
        return nominal

    def record_publication(self,t_ref,step,q_meas,*,kinematics=None):
        fields={name:getattr(step,name,None) for name in ('q_send','qdot','valid','reason','v_tcp','v_tcp_estimated',
            'arm_send_mono_ns','rail_target_publish_mono_ns','rail_fa24_write_mono_ns','rail_encoder_sample_mono_ns',
            'physical_saturated','slack_norm','dt_wall_s','rail_exec_vel_m_s')}
        fields.update(rail_target_published_m=getattr(step,'study_rail_target_m',None),
            rail_velocity_published_m_s=getattr(step,'study_rail_velocity_m_s',None),
            rail_coast=getattr(step,'study_rail_coast',None),
            rail_target_modified=getattr(step,'study_rail_target_modified',None),
            q_send_semantics='inner_proposal_before_rail_wall_clock_adjustment')
        if fields['rail_coast']:
            fields['rail_target_published_m']=None
            fields['rail_action']='hold_current_target_unavailable'
        else:fields['rail_action']='target_publication'
        if kinematics is not None:
            try:fields['command_model_twist_base']=kinematics.jacobian(q_meas) @ step.qdot
            except Exception:fields['command_model_twist_base']=None
        self.sink.emit('publication',control_id=self._last_record_id,reference_s=t_ref,q_measured=q_meas,
                       publication_fact='legacy_transport_calls_returned',**fields)

    def record_stop(self,reason):
        if self._stop_recorded:return
        self._stop_recorded=True
        self.sink.emit('stop',control_id=self._last_record_id,reason=reason,
                       physical_execution='not_inferred_from_software_stop')

    def close(self,*,wait=True):
        self._closed=True
        if self.shadow is not None:self.shadow.close(wait=wait)
        if self.features is not None:self.features.close(wait=wait)
        self.sink.close(wait=wait)


def study_config(payload):
    raw=dict(payload or {}).get('contact_qp')
    if raw is None:return None
    if isinstance(raw,str):
        import yaml
        raw=yaml.safe_load(Path(raw).expanduser().read_text())
    if not isinstance(raw,dict):raise ValueError('contact_qp requires a configuration mapping or YAML path')
    if raw.get('mode','baseline') not in ('baseline','shadow','active'):
        raise ValueError('contact_qp mode must be baseline, shadow, or active')
    return dict(raw)


def wrap_study_phase(phase,payload,context):
    config=study_config(payload)
    if config is None:return phase
    if str(payload.get('reference'))!='icra_path':
        raise ValueError('contact_qp recording currently requires reference=icra_path')
    active = config.get('mode') == 'active'
    if active:
        from .contact_active import ContactQpOuter
        wrapped = ContactQpOuter(phase.outer, config)
    else:
        wrapped = ContactStudyOuter(phase.outer, config)
    phase.outer=wrapped
    if active:
        phase.wait_until = wrapped.arrived
        phase.max_duration_s = 4.0 * wrapped.reference.duration_s + 12.0
    previous_tick,previous_exit=phase.on_tick,phase.on_exit
    def on_tick(t_ref,step,q_meas):
        if not active:
            wrapped.record_publication(t_ref,step,q_meas,kinematics=context.inner.kin)
        if previous_tick is not None:previous_tick(t_ref,step,q_meas)
    def on_exit():
        try:
            wrapped.record_stop('phase_exit_or_replaced')
            if previous_exit is not None:previous_exit()
        finally:wrapped.close(wait=False)
    phase.on_tick,phase.on_exit=on_tick,on_exit
    return phase

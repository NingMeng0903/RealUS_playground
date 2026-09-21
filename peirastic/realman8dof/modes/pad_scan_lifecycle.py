"""Opt-in publication bookkeeping for the existing scan-style handheld TFF.

The nominal force law, allocation proposal, accepted IK command and measured
motion are separate channels. Only successful transport advances command state.
"""
from dataclasses import fields
import time

import numpy as np
from scipy.spatial.transform import Rotation

from peirastic.contact_qp.geometry import contact_cop_from_wrench, twist_tcp_to_face, wrench_tcp_to_face
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing
from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
from peirastic.realman8dof.modes.pad_scan_config import pad_visual_force_gate


def _pad_lifecycle_gamma(settings, cfg, force_n):
    start = settings.get('force_gate_start_n')
    end = settings.get('force_gate_end_n')
    if start is not None and end is not None:
        return pad_visual_force_gate(force_n, start, end)
    return cfg.differential_repair.force_gate(force_n)


class PadScanLifecycle:
    policy = 'pad_scan_v1'

    def __init__(self, baseline, config, *, clock=time.monotonic):
        self.baseline, self.config, self.clock = baseline, config, clock
        self.settings = dict(config.get('pad') or {})
        ceilings = dict(normal_max_m_s=.015, angular_max_rad_s=.28,
            angular_acceleration_rad_s2=2., max_visual_omega_rad_s=.18,
            cop_min_force_n=.8, cop_max_m=.025, hard_stop_n=6.)
        for name, ceiling in ceilings.items():
            value = float(self.settings.get(name, ceiling))
            if not np.isfinite(value) or value <= 0 or (
                    value < ceiling if name == 'cop_min_force_n' else value > ceiling):
                raise ValueError('pad scan setting exceeds approved bound: '+name)
            self.settings[name] = value
        margin = float(self.settings.get('normal_allocation_margin_m_s', .0001))
        if not np.isfinite(margin) or not 0 < margin < self.settings['normal_max_m_s']:
            raise ValueError('pad scan normal allocation margin must lie within normal speed cap')
        self.settings['normal_allocation_margin_m_s'] = margin
        hold = float(self.settings.get('control_hold_s', .005))
        if not np.isfinite(hold) or not np.isclose(hold, baseline.dt, rtol=0., atol=1e-12):
            raise ValueError('pad scan control hold must match baseline control period')
        self.settings['control_hold_s'] = hold
        self.force_law, self.controller = baseline.force_law, baseline.controller
        self.guide = baseline.pad_visual
        if not getattr(self.guide, 'progress', False):
            raise ValueError('pad scan requires the scan-style progress allocator')
        self.geometry = self.guide.geometry
        if (self.controller.cfg.control_frame != 'tool' or
                not np.allclose(baseline.desired_force, [0., 0., 4., 0., 0., 0.]) or
                not np.array_equal(baseline.selection, [1., 1., 0., 1., 0., 1.])):
            raise ValueError('pad scan requires tool-frame 4 N / My=0 TFF')
        if not np.allclose(self.geometry.T_tcp_face, np.eye(4), atol=1e-12):
            raise ValueError('pad scan requires coincident calibrated tool/contact frames')
        self.transform = twist_tcp_to_face(self.geometry)
        cfg, tilt = self.controller.cfg, self.force_law.tilt.cfg
        # Limits are local to this opt-in controller; M and D stay unchanged.
        cfg.max_velocity = np.asarray(cfg.max_velocity, dtype=float).copy()
        cfg.max_velocity[2] = min(cfg.max_velocity[2], self.settings['normal_max_m_s'])
        cfg.max_vz_tool_m_s = min(cfg.max_vz_tool_m_s, self.settings['normal_max_m_s'])
        vmax = cfg.max_velocity.copy()
        vmax[2], vmax[4] = min(vmax[2], cfg.max_vz_tool_m_s), min(vmax[4], tilt.vmax_rad_s, self.settings['angular_max_rad_s'])
        self.final_max_velocity = vmax.copy()
        # Leave room for IK tracking residual while preserving the physical
        # publication cap and the nominal force law's existing 15 mm/s limit.
        if margin >= vmax[2]:
            raise ValueError('pad scan normal allocation margin exceeds controller speed cap')
        vmax[2] -= margin
        accel = np.asarray(cfg.max_acceleration, dtype=float).copy()
        accel[4] = min(accel[4], tilt.a_max, self.settings['angular_acceleration_rad_s2'])
        allowed = {f.name for f in fields(QpConfig)}
        raw = {k: v for k, v in (config.get('qp') or {}).items() if k in allowed}
        for name in ('escalation_omega_rad_s', 'visual_hold_omega_rad_s', 'visual_hold_omega_s'):
            if raw.get(name) == 0.:
                raw[name] = None
        raw.update(allocation_policy='delay_kf_cop_v1', solver_policy='bounded_retry_v1',
            enable_aperture=False, max_velocity=vmax, max_acceleration=accel,
            angle_limit_rad=tilt.theta_max_rad, cop_min_force_n=self.settings['cop_min_force_n'],
            cop_max_m=self.settings['cop_max_m'], max_visual_omega_rad_s=self.settings['max_visual_omega_rad_s'],
            c_min=config['feature']['c_min'], quality_policy_version=config['feature']['quality_policy_version'],
            visual_rocking_jerk_scale=1.)
        self.solver = ContactQp(QpConfig(**raw))
        self.rocking = RockingSmoothing(vmax[4], accel[4], tilt.mass, tilt.damping)
        self.record_sink = None
        self._pending = None
        self._nominal_pending = False
        self._control_id = 0
        self._faulted = False
        self._last_source = None
        self._last_source_time = None
        self._last_final = np.zeros(6)
        self._last_rotation = np.eye(3)
        self._angle_origin = None
        self.last_review_reason = None
        self.max_age_s = float(self.settings.get('max_feedback_age_s', .05))
        self.command_max_age_s = float(self.settings.get('command_max_age_s', .05))
        # Match numerical IK feasibility, without changing physical limits.
        self.final_normal_tolerance_m_s = 1e-6
        self.final_omega_tolerance_rad_s = 1e-5
        # A guarded zero/nominal task permits at most the established 1e-4
        # residual, while preserving any tighter configured task tolerance.
        self.mechanical_guard_tolerance_rad_s = min(float(self.solver.config.subspace_tolerance[4]), 1e-4)

    def _emit(self, event, **fields):
        if self.record_sink is not None:
            self.record_sink.emit(event, record_kind='pad_hybrid', control_id=self._control_id, **fields)

    def configuration_snapshot(self):
        from dataclasses import asdict
        import hashlib
        from pathlib import Path
        from .contact_recording import study_fingerprints
        snapshot = study_fingerprints(self.baseline, self.config)
        directory = Path(__file__).resolve().parent
        for name in ('track.py', 'pad_scan_lifecycle.py', 'pad_scan_config.py',
                     'pad_visual.py', 'pad_contact_evidence.py'):
            snapshot['source_sha256']['realman8dof/modes/'+name] = hashlib.sha256((directory/name).read_bytes()).hexdigest()
        return dict(snapshot, publication_policy=self.policy,
                    rocking_policy='scan_tiers_1_2_3_4', rocking_jerk_scale=1.,
                    effective_qp_config=asdict(self.solver.config), effective_pad_settings=self.settings,
                    final_max_velocity=self.final_max_velocity,
                    normal_allocation_max_m_s=float(self.solver.config.max_velocity[2]),
                    final_normal_tolerance_m_s=self.final_normal_tolerance_m_s,
                    mechanical_guard_tolerance_rad_s=self.mechanical_guard_tolerance_rad_s,
                    final_omega_tolerance_rad_s=self.final_omega_tolerance_rad_s)

    def reset(self, pose, *, applied_twist_base=None):
        self.abort('episode_reset', definitely_not_sent=True, latch=False)
        rotation = rotation_from_pose(pose, euler_order=self.baseline._euler_order())
        applied = np.zeros(6) if applied_twist_base is None else np.asarray(applied_twist_base, dtype=float).reshape(6)
        if not np.isfinite(applied).all():
            raise ValueError('finite pad handoff twist required')
        self._last_rotation = rotation.copy()
        self._last_final = np.r_[rotation.T @ applied[:3], rotation.T @ applied[3:]]
        self.baseline._pad_previous_twist = self._last_final.copy()
        self.controller.begin_hybrid_episode(self._last_final)
        self.force_law.tilt._w = self.force_law.tilt.omega_y = float(self._last_final[4])
        self.rocking.seed(applied[3:], self.clock())
        self._angle_origin = None
        self._last_source = self._last_source_time = None
        self._faulted = False
        self._emit('pad_handoff', applied_twist_base=applied, previous_tool=self._last_final)

    @property
    def pending_rotation_base_tcp(self):
        return self._last_rotation if self._pending is None else self._pending['rotation']

    def sample(self, callback, t_s, pose, wrench, kwargs, **metadata):
        if self._faulted:
            raise RuntimeError('pad scan latched after failed publication; re-enter mode')
        if self._pending is not None:
            raise RuntimeError('unresolved pad scan candidate')
        now = self.clock()
        actual_dt = self.baseline.dt if kwargs['dt_actual'] is None else float(kwargs['dt_actual'])
        ages = np.asarray([kwargs['sensor_age_s'], kwargs['feedback_age_s']], dtype=float)
        wrench = np.asarray(wrench, dtype=float).reshape(6)
        raw = wrench if kwargs['f_ext_raw'] is None else np.asarray(kwargs['f_ext_raw'], dtype=float).reshape(6)
        self._control_id += 1
        if (not np.isfinite(ages).all() or np.any(ages < 0) or np.max(ages) > self.max_age_s or
                not np.isfinite(actual_dt) or not 0 < actual_dt <= self.command_max_age_s or
                not np.isfinite(wrench).all() or not np.isfinite(raw).all()):
            self._emit('pad_input_rejected', reason='invalid_or_stale_feedback', ages_s=ages, dt_actual_s=actual_dt)
            raise RuntimeError('pad scan invalid or stale force/pose feedback')
        source_time = metadata.get('wrench_source_time_s')
        source = (metadata.get('wrench_source_id'), source_time)
        if source_time is not None and (not np.isfinite(source_time) or not 0 <= now-source_time <= self.max_age_s or
                self._last_source_time is not None and source_time < self._last_source_time):
            raise RuntimeError('pad scan invalid or stale wrench source')
        fresh = source != self._last_source if source_time is not None else kwargs['feedback_fresh_tick'] is not False
        source_dt = (source_time-self._last_source_time if fresh and source_time is not None and
                     self._last_source_time is not None else self.baseline.dt)
        forces = [float(wrench[2]), float(raw[2]),
                  float(-wrench_tcp_to_face(-wrench, self.geometry)[2]),
                  float(-wrench_tcp_to_face(-raw, self.geometry)[2])]
        self.force_n = forces[2]
        self.supervisory_force_n = max(forces)
        if fresh and self.supervisory_force_n >= self.settings['hard_stop_n']:
            self._emit('pad_input_rejected', reason='force_stop', force_n=self.supervisory_force_n,
                       force_channels_n=forces, threshold_n=self.settings['hard_stop_n'])
            self._faulted = True
            raise RuntimeError('pad scan supervisory force stop (6 N)')
        if fresh:
            self._last_source, self._last_source_time = source, source_time
        rotation = rotation_from_pose(pose, euler_order=self.baseline._euler_order())
        contact = kwargs['contact']
        if contact is None:
            contact = bool(getattr(self.controller, 'contact_present', False))
        if not contact:
            self._angle_origin = None
        elif self._angle_origin is None:
            self._angle_origin = rotation.copy()
        theta = (0. if self._angle_origin is None else
                 float(Rotation.from_matrix(self._angle_origin.T @ rotation).as_rotvec()[1]))
        cop, _ = contact_cop_from_wrench(wrench_tcp_to_face(-wrench, self.geometry),
            min_force_n=self.solver.config.cop_min_force_n,
            max_abs_m=min(self.geometry.half_length_m, self.solver.config.cop_max_m))
        tilt_cfg = self.force_law.tilt.cfg
        slack = kwargs['slack_norm']
        preview_contact = bool(contact and cop is not None and
            abs(theta) < tilt_cfg.theta_max_rad and
            (tilt_cfg.align_min <= 0 or abs(rotation[2, 2]) >= tilt_cfg.align_min) and
            (slack is None or not np.isfinite(slack) or slack <= tilt_cfg.slack_freeze) and
            not self.force_law.tilt.tilt_stalled)
        # The same measured relative-pose angle feeds the nominal guard and QP.
        self.force_law.tilt.theta_tilt = theta
        previous_base = np.r_[self._last_rotation @ self._last_final[:3], self._last_rotation @ self._last_final[3:]]
        previous = np.r_[rotation.T @ previous_base[:3], rotation.T @ previous_base[3:]]
        if self.rocking.time_s is None:
            self.rocking.seed(previous_base[3:], now-self.baseline.dt)
        self._pending = dict(state='preparing', now=now, rotation=rotation, theta=theta,
            previous=previous, actual_dt=actual_dt, source_dt=max(float(source_dt), 1e-4),
            source_deadline=(now-max(ages) if source_time is None else min(source_time, now-max(ages)))+self.max_age_s,
            fresh=fresh, preview_contact=preview_contact,
            metadata=dict(metadata, sensor_age_s=ages[0], feedback_age_s=ages[1],
                measurement_fresh=fresh, dt_actual_s=actual_dt, command_hold_s=self.baseline.dt,
                filtered_contact_force_n=self.force_n, supervisory_force_n=self.supervisory_force_n,
                measured_vz_tool_m_s=kwargs['v_tcp_z_actual']))
        kwargs = dict(kwargs, dt_actual=self.baseline.dt)
        try:
            return callback(t_s, pose, wrench, **kwargs)
        except Exception as exc:
            self._emit('pad_proposal_failed', reason=type(exc).__name__, detail=str(exc))
            self.abort('proposal_failed:'+str(exc), definitely_not_sent=True)
            raise

    def prepare_force(self, **kwargs):
        p = self._pending
        result = self.force_law.prepare(measurement_id=self._control_id,
            measurement_fresh=p['fresh'], source_dt_s=p['source_dt'], **kwargs)
        self._nominal_pending = True
        p['nominal'] = np.asarray(result.v_force, dtype=float).copy()
        p['nominal_normal'] = float(result.v_force_z)
        return result

    def allocate(self, pose, f_ext, fout, path_raw, guide, contact, dt_s):
        p, cfg = self._pending, self.solver.config
        env = wrench_tcp_to_face(-np.asarray(f_ext, dtype=float), self.geometry)
        cop, cop_reason = contact_cop_from_wrench(env, min_force_n=cfg.cop_min_force_n,
            max_abs_m=min(self.geometry.half_length_m, cfg.cop_max_m))
        tilt = self.force_law.tilt
        guarded = bool(tilt.tilt_frozen or tilt.tilt_stalled or tilt.tilt_capped or not tilt.engaged)
        visual_valid = bool(contact and guide.evidence_valid and cop is not None and not guarded)
        request = float(np.clip(getattr(guide, 'omega_target', 0.) or 0.,
            -cfg.max_visual_omega_rad_s, cfg.max_visual_omega_rad_s)) if visual_valid else 0.
        alpha = float(np.clip(guide.last_alpha, 0., 1.)) if visual_valid else 0.
        feedback = np.asarray(self.baseline.position.last_feedback_twist, dtype=float).copy()*self.baseline.selection
        moving = np.asarray(path_raw, dtype=float).copy()*self.baseline.selection
        feedback[[2, 4]], moving[[2, 4]] = 0., 0.
        if alpha == 0.:
            feedback[:], moving[:] = 0., 0.
        if np.linalg.norm(moving[:2]) < min(cfg.subspace_tolerance[:2]):
            moving[:2] = 0.
        previous = p['previous'].copy()
        # Preserve the existing immediate handheld XY path; command history
        # still supplies current-frame normal and angular acceleration state.
        previous = np.where(self.baseline.selection > .5, feedback+alpha*moving, previous)
        mechanical = p['nominal']
        now = self.clock()
        result = self.solver.solve(QpInput(geometry=self.geometry, nominal_twist=mechanical,
            path_twist=moving, force_n=self.force_n, dt_s=self.baseline.dt, now_s=now,
            previous_twist=previous, measured_angle=p['theta'],
            acceleration_dt_s=max(self.baseline.dt, now-self.rocking.time_s),
            alpha_preferred=alpha, mechanical_normal_m_s=float((self.transform@mechanical)[2]),
            mechanical_omega_rad_s=float((self.transform@mechanical)[4]),
            path_feedback_contact=self.transform@feedback, path_feedforward_contact=self.transform@moving,
            visual_request_rad_s=request, visual_task_valid=visual_valid,
            visual_gamma=_pad_lifecycle_gamma(self.settings, cfg, self.force_n), cop_m=cop,
            repair_execution_enabled=bool(contact)), online=True,
            deadline_s=time.monotonic()+min(.004, self.baseline.dt))
        self.baseline.last_facts = dict(guide.last_facts or {}, cop_m=cop, cop_valid=cop is not None,
            cop_reason=cop_reason, alpha_target=alpha, visual_request_rad_s=request,
            visual_task_valid=visual_valid, mechanical_guarded=guarded,
            measured_theta_rad=p['theta'], visual_policy='pad_scan_delay_kf_cop_v1',
            allocation_success=result.qp_twist is not None, allocation_status=str(result.status),
            normal_shortfall=(None if result.qp_twist is None else float(mechanical[2]-result.qp_twist[2])),
            allocation_diagnostics=dict(result.diagnostics))
        # Full column evidence is emitted once per image by the guide logger.
        image_refs = {'source_id', 'frame_seq', 'effective_time_s', 'image_age_s', 'visual_reason'}
        allocation_facts = {key: value for key, value in self.baseline.last_facts.items()
                            if key not in guide.last_facts or key in image_refs}
        self._emit('pad_allocation', nominal_tool=mechanical, candidate_tool=result.qp_twist,
                   facts=allocation_facts, **p['metadata'])
        if result.qp_twist is None:
            raise RuntimeError('pad scan QP failed: '+str(result.diagnostics.get('reason', result.status)))
        candidate = np.asarray(result.qp_twist, dtype=float).reshape(6)
        self.baseline.last_pad_alpha = float(result.alpha)
        self.baseline.last_path_twist = feedback+float(result.alpha)*moving
        p.update(state='prepared', candidate=candidate.copy(), result=result,
                 guarded=guarded, path_paused=alpha == 0.)
        return candidate

    def command_constraints(self):
        p = self._pending
        if p is None or p['state'] != 'prepared':
            raise RuntimeError('pad native constraints require fresh candidate')
        axis, bounds, history = self.rocking.preview(p['rotation'], self.clock(), self.baseline.dt, jerk_scale=1.)
        p['rocking_preview'] = history
        # These physical speed bounds stay active through every native rocking
        # tier. They do not certify the outer allocation or its soft objectives.
        rows = np.zeros((2, 6))
        rows[0, :3] = p['rotation'][:, 2]
        rows[1, 3:] = p['rotation'][:, 1]
        caps = self.final_max_velocity[[2, 4]].copy()
        lower, upper = -caps, caps.copy()
        if p['guarded']:
            # Admit the same mechanical interval that publication reviews.
            # This row survives rocking relaxation, including tier four.
            lower[1] = max(lower[1], min(0., p['nominal'][4]))
            upper[1] = min(upper[1], max(0., p['nominal'][4]))
        self._emit('pad_rocking_constraints', bounds=bounds, axis_base=axis, history=history,
                   command_twist_lower=lower, command_twist_upper=upper,
                   mechanical_guarded=p['guarded'])
        return dict(rocking_axis_base=axis, rocking_bounds=bounds,
                    command_twist_rows_base=rows,
                    command_twist_lower=lower, command_twist_upper=upper)

    def review(self, final_tool, *, now_s, facts=None):
        p = self._pending
        final = np.asarray(final_tool, dtype=float).reshape(6)
        facts = dict(facts or {})
        reason = None
        if p is None or p['state'] != 'prepared':
            reason = 'missing_candidate'
        elif not p['now'] <= now_s < min(p['now']+self.command_max_age_s, p['source_deadline']):
            reason = 'stale_candidate'
        elif not np.isfinite(final).all():
            reason = 'nonfinite_final'
        elif np.any(np.abs(final[[2, 4]]) > self.final_max_velocity[[2, 4]]+
                    [self.final_normal_tolerance_m_s, self.final_omega_tolerance_rad_s]):
            reason = 'final_hard_velocity_limit'
        elif abs(p['theta']+final[4]*self.baseline.dt) > self.solver.config.angle_limit_rad+1e-8:
            reason = 'final_measured_angle_limit'
        elif not self.rocking.review(final, facts, 1e-5):
            reason = 'final_native_rocking_interval'
        elif p['guarded'] and not (min(0., p['nominal'][4])-self.mechanical_guard_tolerance_rad_s <= final[4] <=
                                   max(0., p['nominal'][4])+self.mechanical_guard_tolerance_rad_s):
            reason = 'final_mechanical_guard'
        self.last_review_reason = reason
        self._emit('pad_publication_review', accepted=reason is None, reason=reason,
                   final_tool=final, final_is_measurement=False, facts=facts, review_time_s=now_s)
        if reason is not None:
            return False
        p.update(state='reviewed', reviewed=final.copy(), review_facts=facts)
        return True

    def started(self):
        p, now = self._pending, self.clock()
        if p is None or p['state'] != 'reviewed' or now >= min(p['source_deadline'], p['now']+self.command_max_age_s):
            return False
        p['state'] = 'sending'
        return True

    def arm_dispatch_time(self, now_s):
        p = self._pending
        if (p is None or p['state'] != 'sending' or not p['now'] <= now_s <
                min(p['source_deadline'], p['now']+self.command_max_age_s) or now_s <= self.rocking.time_s):
            return False
        p['publication_time_s'] = float(now_s)
        return True

    def commit(self, final_tool, *, now_s, facts=None):
        p = self._pending
        final = np.asarray(final_tool, dtype=float).reshape(6)
        if (p is None or p['state'] != 'sending' or 'publication_time_s' not in p or
                not np.array_equal(final, p['reviewed'])):
            raise RuntimeError('pad commit requires exact reviewed and dispatched command')
        stamp = p['publication_time_s']
        audit = self.rocking.publication_audit(final, p['rotation'], stamp, self.baseline.dt,
            int(p['review_facts']['rocking_policy_tier']), 1e-5)
        nominal = p['nominal']
        self.force_law.commit_applied(nominal, final_full_twist=nominal,
                                     accepted_normal_z=p['nominal_normal'])
        self._nominal_pending = False
        self.force_law.tilt.note_executed_omega(float(final[4]), p['actual_dt'])
        self.rocking.commit(final, p['rotation'], stamp)
        self._last_final, self._last_rotation = final.copy(), p['rotation'].copy()
        self.baseline._pad_previous_twist = final.copy()
        self.guide.commit(final, p['rotation'])
        self._pending = None
        self._emit('pad_publication_commit', transaction='committed', nominal_tool=nominal,
            candidate_tool=p['candidate'], final_tool=final, final_is_measurement=False,
            final_minus_candidate_tool=final-p['candidate'], path_request_paused=p['path_paused'],
            publication_time_s=stamp, transport_completion_time_s=now_s, rocking_audit=audit,
            facts=facts, **p['metadata'])

    def abort(self, reason, *, definitely_not_sent=False, facts=None, latch=None):
        pending = self._pending
        try:
            if self._nominal_pending:
                self.force_law.abort()
        finally:
            self._nominal_pending = False
            self._pending = None
            if pending is not None:
                self._faulted = bool(not definitely_not_sent if latch is None else latch)
                self._emit('pad_publication_abort', transaction='aborted', reason=reason,
                    definitely_not_sent=definitely_not_sent, facts=facts, latch=self._faulted)

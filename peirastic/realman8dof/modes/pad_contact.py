"""Transactional gamepad contact allocation, independent of scan trajectories.

The image specifies a TOTAL rocking request. Only this adapter composes normal,
rocking and tangential progress; the runner commits it after successful sends.
"""
import hashlib
from pathlib import Path
import time

import numpy as np

from peirastic.contact_qp.geometry import constraints_to_base, contact_cop_from_wrench, twist_tcp_to_face, wrench_tcp_to_face
from peirastic.contact_qp.qp import ContactQp, QpConfig, QpInput
from peirastic.contact_qp.repair_policy import DifferentialRepairConfig
from peirastic.contact_qp.rocking_smoothing import RockingSmoothing, published_rocking_rows
from peirastic.contact_qp.runtime_config import calibrated_geometry
from peirastic.contact_qp.types import TwistConstraints
from peirastic.realman8dof.force.torque_tilt import rotation_from_pose
from .pad_contact_config import load_pad_contact_config
from .pad_publication import dispatch_rocking_bounds


class PadContactOuter:
    def __init__(self, baseline, config=None, *, evidence=None, clock=time.monotonic):
        self.baseline = baseline
        self.config = load_pad_contact_config(config)
        self.settings = self.config['pad']
        self.controller = baseline.controller
        self.force_law = baseline.force_law
        if (self.controller.cfg.control_frame != 'tool' or not np.isclose(baseline.desired_force[2], 4.)
                or not np.isclose(baseline.desired_force[4], 0.)):
            raise ValueError('pad contact requires the existing tool-frame 4 N force task')
        if not np.array_equal(np.asarray(baseline.selection), [1., 1., 0., 1., 0., 1.]):
            raise ValueError('pad contact owns tool Z and tool My')
        self.geometry, _ = calibrated_geometry(self.config)
        # The nominal torque task is tool My. A rotated installation needs an
        # independently calibrated mechanical task; do not silently relabel it.
        if self.config['geometry']['face_normal_convention'] != 'into_contact' or not np.allclose(
                self.geometry.T_tcp_face, np.eye(4), atol=1e-12):
            raise ValueError('pad contact requires coincident calibrated tool/contact frames')
        self.transform = twist_tcp_to_face(self.geometry)
        self.clock = clock
        cfg = self.controller.cfg
        cfg.max_velocity = np.asarray(cfg.max_velocity, dtype=float).copy()
        cfg.max_velocity[2] = min(cfg.max_velocity[2], self.settings['normal_max_m_s'])
        cfg.max_vz_tool_m_s = min(cfg.max_vz_tool_m_s, self.settings['normal_max_m_s'])
        tilt = self.force_law.tilt.cfg
        vmax = cfg.max_velocity.copy()
        vmax[2] = min(vmax[2], cfg.max_vz_tool_m_s)
        vmax[4] = min(vmax[4], tilt.vmax_rad_s, self.settings['angular_max_rad_s'])
        accel = np.asarray(cfg.max_acceleration, dtype=float).copy()
        accel[4] = min(accel[4], tilt.a_max, self.settings['angular_acceleration_rad_s2'])
        self.solver = ContactQp(QpConfig(allocation_policy='delay_kf_cop_v1',
            solver_policy='bounded_retry_v1', enable_aperture=False,
            max_velocity=vmax, max_acceleration=accel, angle_limit_rad=tilt.theta_max_rad,
            max_step_s=self.settings['command_max_age_s'],
            certificate_horizon_s=self.settings['command_max_age_s'],
            cop_min_force_n=self.settings['cop_min_force_n'], cop_max_m=self.settings['cop_max_m'],
            max_visual_omega_rad_s=self.settings['max_visual_omega_rad_s']))
        self.rocking = RockingSmoothing(vmax[4], accel[4], tilt.mass, tilt.damping)
        if evidence is None:
            from .pad_contact_evidence import PadContactEvidence
            evidence = PadContactEvidence(self.config)
        self.pad_visual = evidence
        self.command_lifecycle = self
        self.record_sink = None
        self._pending = None
        self._control_id = 0
        self._last_source = None
        self._last_source_time = None
        self._last_final = np.zeros(6)
        self._last_rotation = np.eye(3)
        self._last_alpha = 1.
        self._last_commit_s = None
        self._faulted = False
        self._force_supervisor_since = None
        self.last_facts = {}
        self.last_review_reason = None
        self._record_measurement = {}

    def __getattr__(self, name):
        return getattr(self.baseline, name)

    def _emit(self, event, **fields):
        if self.record_sink is not None:
            self.record_sink.emit(event, record_kind='pad_hybrid', control_id=self._control_id, **fields)

    def configuration_snapshot(self):
        from .contact_recording import study_fingerprints
        info = study_fingerprints(self.baseline, self.config)
        root = Path(__file__).resolve().parents[2]
        for relative in ('realman8dof/modes/pad_contact.py', 'realman8dof/modes/pad_contact_config.py',
                         'realman8dof/modes/pad_contact_evidence.py', 'realman8dof/modes/pad_visual.py',
                         'realman8dof/modes/pad_publication.py', 'realman8dof/daemon.py'):
            path = root / relative
            if path.is_file():
                info['source_sha256'][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for relative in ('joint_admittance_8dof/command_lifecycle.py', 'joint_admittance_8dof/hw/rail_servo.py'):
            path = root.parent/'rm75_control/rm75_control/control'/relative
            info['source_sha256']['rm75_control/'+relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = 'rm75_control/rm75_control/hw/lw100/modbus_rtu_tcp.py'
        info['source_sha256'][relative] = hashlib.sha256((root.parent/relative).read_bytes()).hexdigest()
        return info

    def set_origin(self, pose0, *, t_s=None):
        self.abort('set_origin', definitely_not_sent=True)
        self.baseline.set_origin(pose0, t_s=t_s)
        self._last_final[:] = 0.
        self._last_rotation = rotation_from_pose(pose0)
        self._last_commit_s = None
        self._last_alpha = 1.
        self._faulted = False
        self._force_supervisor_since = None
        # The previous mode does not supply its accepted publication time.
        # Seeding at the observed handoff is conservative; never fabricate an
        # earlier send to gain extra acceleration/jerk room on the first tick.
        self.rocking.seed(np.zeros(3), self.clock())

    def begin_hybrid_episode(self, applied_twist_base, current_pose):
        self.set_origin(current_pose)
        rotation = rotation_from_pose(current_pose)
        applied = np.asarray(applied_twist_base, dtype=float).reshape(6)
        self._last_final = np.r_[rotation.T @ applied[:3], rotation.T @ applied[3:]]
        self.rocking.seed(applied[3:], self.clock())
        self._emit('pad_handoff', applied_twist_base=applied, previous_tool=self._last_final,
                   pose=current_pose, handoff_time_s=self.rocking.time_s)

    @property
    def pending_rotation_base_tcp(self):
        return self._pending['rotation'] if self._pending is not None else self._last_rotation

    def _dispatch_times(self, now, source_deadline):
        earliest = max(now, self.rocking.time_s+self.settings['minimum_publication_period_s'])
        latest = min(now+self.settings['command_max_age_s'], source_deadline)-1e-6
        if earliest >= latest:
            raise RuntimeError('pad feedback expires before minimum publication period')
        return earliest, latest

    def sample(self, t_s, current_pose, f_ext, *, contact=None, f_ext_raw=None,
               dt_actual=None, sensor_age_s=None, feedback_age_s=None,
               feedback_fresh_tick=None, feedback_velocity_valid=None, v_tcp_z_actual=None,
               slack_norm=None, wrench_source_time_s=None, wrench_source_id=None,
               wrench_source_wall_time_ns=None, measured_twist_base=None,
               measured_twist_valid=False, measured_twist_fresh=False,
               measurement_time_s=None, measured_twist_metadata=None):
        if self._faulted:
            raise RuntimeError('pad contact is latched after a failed command; re-enter the mode')
        if self._pending is not None:
            raise RuntimeError('previous pad proposal has not been committed or aborted')
        now = self.clock()
        actual_dt = float(self.baseline.dt if dt_actual is None else dt_actual)
        if not np.isfinite(actual_dt) or actual_dt <= 0:
            raise ValueError('invalid pad control interval')
        if actual_dt > self.settings['command_max_age_s']:
            self._emit('pad_input_rejected', reason='control_interval_expired', dt_actual_s=actual_dt,
                       maximum_s=self.settings['command_max_age_s'])
            raise RuntimeError(f'pad control interval expired ({actual_dt:.6f} s)')
        # Scan Ultra hold: force, COP and image fusion command a 5 ms
        # increment. A stretched wall period must not retune those dynamics.
        dt = min(actual_dt, float(self.baseline.dt))
        ages = np.asarray([sensor_age_s, feedback_age_s], dtype=float)
        if not np.isfinite(ages).all() or np.any(ages < 0) or np.max(ages) > self.settings['max_feedback_age_s']:
            raise RuntimeError('pad force/pose feedback unavailable or stale')
        wrench = np.asarray(f_ext, dtype=float).reshape(6)
        if not np.isfinite(wrench).all():
            raise ValueError('pad needs a finite compensated wrench')
        self._supervise_force(float(wrench[2]), now)
        rotation = rotation_from_pose(current_pose, euler_order=self.baseline._euler_order())
        raw = np.asarray(self.baseline.position.sample(t_s, current_pose, wrench), dtype=float).reshape(6)
        path = self.transform @ raw
        path[[2, 4]] = 0.
        # A filtered released stick can leave nanometre/s tails. Exporting
        # progress=alpha*tail turns a harmless zero into a near-zero-width
        # constraint with a huge inverse path scale in the inner solver.
        if np.linalg.norm(path[:2]) < float(np.min(self.solver.config.subspace_tolerance[:2])):
            path[:2] = 0.
        self._control_id += 1
        source = (wrench_source_id, wrench_source_time_s)
        if wrench_source_time_s is not None and (not np.isfinite(wrench_source_time_s)
                or self._last_source_time is not None and float(wrench_source_time_s) < self._last_source_time):
            raise RuntimeError('pad wrench source time regressed or is invalid')
        fresh = source != self._last_source if wrench_source_time_s is not None else feedback_fresh_tick is not False
        source_dt = (float(wrench_source_time_s) - self._last_source_time
                     if fresh and wrench_source_time_s is not None and self._last_source_time is not None else dt)
        if fresh and wrench_source_time_s is not None:
            self._last_source_time = float(wrench_source_time_s)
        self._last_source = source
        self._record_measurement = dict(sensor_age_s=sensor_age_s, feedback_age_s=feedback_age_s,
            dt_actual_s=actual_dt, command_hold_s=dt, source_time_s=wrench_source_time_s,
            measurement_fresh=fresh, measured_vz_tool_m_s=v_tcp_z_actual,
            feedback_velocity_valid=feedback_velocity_valid,
            measured_twist_base=measured_twist_base, measured_twist_valid=measured_twist_valid,
            measured_twist_fresh=measured_twist_fresh, measurement_time_s=measurement_time_s,
            measured_twist_metadata=measured_twist_metadata)
        # The nominal command state is speculative; all exceptions after this
        # point abort it. Measurement/contact observations are consumed once.
        out = self.force_law.prepare(measurement_id=self._control_id, dt_s=dt, dt_actual=dt,
            pose=current_pose, f_ext=wrench, f_des=self.baseline.desired_force,
            path_twist=np.linalg.solve(self.transform, path), path_twist_raw=raw,
            contact=contact, f_ext_raw=f_ext_raw, sensor_age_s=sensor_age_s,
            feedback_age_s=feedback_age_s, measurement_fresh=fresh, source_dt_s=max(source_dt, 1e-4),
            v_tcp_z_actual=v_tcp_z_actual if feedback_velocity_valid is not False else None,
            slack_norm=slack_norm, euler_order=self.baseline._euler_order())
        self._pending = dict(state='prepared', now=now, dt=dt, rotation=rotation,
                             source_deadline=now + self.settings['max_feedback_age_s'] - float(np.max(ages)))
        try:
            present = bool(contact if contact is not None else self.controller.contact_present)
            nominal = self.transform @ np.asarray(out.v_force, dtype=float)
            if self.force_law.tilt.needs_normal_retract:
                raise RuntimeError('pad mechanical normal recovery requires exiting automatic contact')
            cp, cp_reason = contact_cop_from_wrench(wrench_tcp_to_face(-wrench, self.geometry),
                min_force_n=self.settings['cop_min_force_n'], max_abs_m=self.settings['cop_max_m'])
            visual = self.pad_visual.update(now_s=now, dt_s=dt, contact_present=present,
                force_n=float(wrench[2]), stick_speed_m_s=float(np.linalg.norm(path[:2])))
            telemetry = dict(out.telemetry or {})
            guarded = (not present or not self.force_law.tilt.engaged or self.force_law.tilt.tilt_frozen
                       or self.force_law.tilt.tilt_stalled)
            visual_valid = bool(visual.valid and cp is not None and not guarded)
            alpha_target = float(visual.alpha_target)
            if present and (cp is None or guarded):
                alpha_target = 0.
            rate = 1. / self.settings['alpha_ramp_down_s' if alpha_target < self._last_alpha else 'alpha_ramp_up_s']
            alpha = float(self._last_alpha + np.clip(alpha_target-self._last_alpha, -rate*dt, rate*dt))
            previous = np.r_[rotation.T @ self._last_rotation @ self._last_final[:3],
                             rotation.T @ self._last_rotation @ self._last_final[3:]]
            earliest, latest = self._dispatch_times(now, self._pending['source_deadline'])
            axis, bounds = dispatch_rocking_bounds(self.rocking, rotation, earliest, latest,
                minimum_period_s=self.settings['minimum_publication_period_s'])
            rock_rows, _ = published_rocking_rows(self.transform[4], bounds,
                now_s=now, horizon_s=self.settings['command_max_age_s'], tier=1)
            if rock_rows is None:
                raise RuntimeError('pad final angular acceleration/jerk interval is infeasible')
            self._pending.update(dispatch_earliest_s=earliest, dispatch_latest_s=latest)
            rows, low, high, labels = list(rock_rows.A), list(rock_rows.lower), list(rock_rows.upper), list(rock_rows.labels)
            published_dt = earliest-self.rocking.time_s
            if published_dt <= 0:
                raise RuntimeError('pad command publication clock did not advance')
            # Shared scan allocation floors its own slew window at dt_s. The
            # pad supplies additional non-relaxable rows for the actual gap.
            rows.extend(np.eye(6))
            low.extend(previous-self.solver.config.max_acceleration*published_dt)
            high.extend(previous+self.solver.config.max_acceleration*published_dt)
            labels.extend('pad_published_acceleration_'+str(i) for i in range(6))
            # The shared scan row uses the control period. Pad dispatch may
            # occur later, and angle history commits that actual interval.
            # Certify both endpoints so the arm boundary need only check time.
            theta = self.force_law.tilt.theta_tilt
            angle_limit = self.solver.config.angle_limit_rad
            angular_gaps = (published_dt, latest-self.rocking.time_s)
            rows.append(self.transform[4])
            low.append(max((-angle_limit-theta)/gap for gap in angular_gaps))
            high.append(min((angle_limit-theta)/gap for gap in angular_gaps))
            labels.append('pad_dispatch_angle_limit')
            guard_bounds = None
            if guarded:
                # Contact/freeze withdraws visual repair and requests nominal
                # braking. Mechanical and publication loops have different
                # slew limits: include ONLY the braking tail forced by the
                # final jerk constraints instead of conflicting with them.
                brake = float(np.clip(nominal[4], max(bounds[::2]), min(bounds[1::2])))
                guard_bounds = (min(0., brake), max(0., brake))
                rows.append(self.transform[4]); low.append(guard_bounds[0]); high.append(guard_bounds[1])
                labels.append('pad_mechanical_rotation_guard')
            # Current manual demand may leave the prior motion subspace on a
            # release/reversal. Retain only the finite braking tail needed by
            # published acceleration limits, never a position reference.
            delta = self.solver.config.max_acceleration*published_dt
            fixed = np.zeros(6)
            fixed[[3, 5]] = np.clip(path[[3, 5]], previous[[3, 5]]-delta[[3, 5]],
                                    previous[[3, 5]]+delta[[3, 5]])
            fixed[[3, 5]] = np.clip(fixed[[3, 5]], -self.solver.config.max_velocity[[3, 5]],
                                    self.solver.config.max_velocity[[3, 5]])
            moving = path.copy(); moving[[3, 5]] = 0.
            progress_lo, progress_hi = 0., alpha
            for i in (0, 1):
                lower, upper = previous[i]-delta[i], previous[i]+delta[i]
                if abs(moving[i]) < 1e-12:
                    if lower > 1e-12 or upper < -1e-12:
                        progress_hi = -1.
                else:
                    limits = sorted((lower/moving[i], upper/moving[i]))
                    progress_lo, progress_hi = max(progress_lo, limits[0]), min(progress_hi, limits[1])
            braking_translation = progress_lo > progress_hi+1e-12
            if braking_translation:
                fixed[:2] = np.clip(alpha*moving[:2], previous[:2]-delta[:2], previous[:2]+delta[:2])
                moving[:2] = 0.
            path_norm2 = float(np.dot(moving[:2], moving[:2]))
            if path_norm2 > 1e-16:
                # Preferred progress alone is a soft objective in the shared
                # scan allocator. The handheld pause is a hard upper bound.
                rows.append(moving[:2] @ self.transform[:2] / np.sqrt(path_norm2))
                low.append(0.); high.append(alpha*np.sqrt(path_norm2))
                labels.append('pad_progress_cap')
            mechanical = TwistConstraints(np.asarray(rows), low, high, valid_until_s=now+self.settings['command_max_age_s'], labels=tuple(labels))
            # Manual rotations are independent; alpha governs only tangential
            # translation. There is no accumulated position reference.
            data = QpInput(geometry=self.geometry, nominal_twist=np.linalg.solve(self.transform, nominal+fixed+alpha*moving),
                path_twist=np.linalg.solve(self.transform, moving), force_n=float(wrench[2]),
                dt_s=dt, acceleration_dt_s=published_dt,
                now_s=now, previous_twist=previous, measured_angle=self.force_law.tilt.theta_tilt,
                mechanical=mechanical, mechanical_normal_m_s=float(nominal[2]), mechanical_omega_rad_s=float(nominal[4]),
                path_feedback_contact=fixed, path_feedforward_contact=moving, alpha_preferred=alpha,
                visual_request_rad_s=float(visual.omega_request_rad_s), visual_task_valid=visual_valid,
                visual_gamma=DifferentialRepairConfig().force_gate(float(wrench[2])), cop_m=cp)
            result = self.solver.solve(data, interval_diagnostics=False,
                                       deadline_s=time.monotonic()+min(.004, dt))
            if not result.success:
                self._emit('pad_allocation_failed', diagnostics=dict(result.diagnostics),
                    previous_tool=previous, mechanical_nominal_contact=nominal,
                    requested_path=path, guarded=guarded, guard_bounds=guard_bounds,
                    force_tool=wrench, pose=current_pose, contact=present,
                    dt_actual_s=actual_dt, command_hold_s=dt,
                    dispatch_earliest_s=earliest, dispatch_latest_s=latest,
                    rocking_bounds=bounds)
                raise RuntimeError('pad contact allocation failed: '+str(result.diagnostics.get('reason', result.status)))
            self._pending.update(result=result, nominal=nominal, cop=cp, alpha=float(result.alpha),
                visual_valid=visual_valid, guarded=guarded, guard_bounds=guard_bounds, previous=previous, mechanical=mechanical,
                bounds=bounds, axis=axis, path=moving, fixed=fixed, theta=self.force_law.tilt.theta_tilt)
            self.last_contact_present = present
            self.last_pad_alpha = float(result.alpha)
            self.last_path_twist_raw = raw.copy()
            self.last_path_twist = np.linalg.solve(self.transform, fixed+result.alpha*moving)
            self.last_feedback_twist = np.zeros(6)
            self.last_vel_ff = self.last_path_twist.copy()
            self.last_pose_d = np.asarray(current_pose).copy()
            self.last_omega_y = float(nominal[4])
            self.last_tau_y = float(wrench[4])
            self.last_tau_error_y = -float(wrench[4])
            self.last_facts = dict(visual.facts, alpha_target=alpha_target, alpha=self.last_pad_alpha,
                cop_m=cp, cop_valid=cp is not None, cop_reason=cp_reason,
                visual_request_rad_s=float(visual.omega_request_rad_s), visual_task_valid=visual_valid,
                mechanical_nominal_contact=nominal.tolist(), allocated_tool=result.qp_twist.tolist(),
                braking_translation=braking_translation,
                guarded_braking=guarded, guarded_omega_bounds=guard_bounds,
                visual_reason=visual.reason, visual_policy='pad_contact_v1',
                dispatch_earliest_s=earliest, dispatch_latest_s=latest,
                rocking_bounds=bounds.tolist(),
                allocation_diagnostics=dict(result.diagnostics), nominal_telemetry=telemetry,
                **self._record_measurement)
            self._emit('pad_allocation', **self.last_facts)
            return result.qp_twist.copy()
        except BaseException as exc:
            self.abort('sample_failed', definitely_not_sent=True,
                       facts={'exception': type(exc).__name__, 'detail': str(exc)})
            raise

    def dispatch_wait_s(self, now_s):
        if self._pending is None:
            raise RuntimeError('no prepared pad command')
        return max(0., self._pending['dispatch_earliest_s']-float(now_s))

    def command_constraints(self):
        if self._pending is None:
            raise RuntimeError('no prepared pad command')
        p = self._pending
        rows = constraints_to_base(p['result'].hard_constraints, p['rotation'])
        # Inner/native transport admits at most 16 Cartesian rows. Intersect
        # parallel rows (speed, acceleration, angle and jerk share directions)
        # rather than dropping any of their constraints.
        merged = {}
        for a, low, high in zip(rows.A, rows.lower, rows.upper):
            norm = float(np.linalg.norm(a))
            if norm < 1e-12:
                if not low <= 0. <= high:
                    raise RuntimeError('infeasible zero Cartesian constraint')
                continue
            a, low, high = a/norm, low/norm, high/norm
            if a[np.flatnonzero(np.abs(a)>1e-12)[0]] < 0:
                a, low, high = -a, -high, -low
            key = tuple(np.round(a, 12))
            old = merged.get(key, (a, -np.inf, np.inf))
            merged[key] = (a, max(low, old[1]), min(high, old[2]))
        if len(merged) > 16:
            raise RuntimeError('pad Cartesian constraints exceed inner row capacity')
        values = list(merged.values())
        return dict(rocking_axis_base=p['axis'], rocking_bounds=p['bounds'],
                    command_twist_rows_base=np.array([v[0] for v in values]),
                    command_twist_lower=np.array([v[1] for v in values]),
                    command_twist_upper=np.array([v[2] for v in values]))

    def review(self, final_tool, *, now_s, facts=None):
        p = self._pending
        if p is None or p['state'] != 'prepared':
            return False
        final = np.asarray(final_tool, dtype=float).reshape(6)
        reason = None
        if not np.isfinite(final).all():
            reason = 'nonfinite_final_command'
        elif not p['dispatch_earliest_s'] <= now_s <= p['dispatch_latest_s']:
            reason = 'outside_dispatch_window'
        elif now_s < p['now'] or now_s-p['now'] >= self.settings['command_max_age_s'] or now_s >= p['source_deadline']:
            reason = 'expired_command_or_feedback'
        else:
            cfg = self.solver.config
            tolerance = 1e-5
            elapsed = now_s-self.rocking.time_s
            if np.any(np.abs(final) > cfg.max_velocity+tolerance):
                reason = 'final_speed_limit'
            elif np.any(np.abs(final-p['previous']) > cfg.max_acceleration*elapsed+tolerance):
                reason = 'final_acceleration_limit'
            elif abs(p['theta']+(self.transform@final)[4]*elapsed) > cfg.angle_limit_rad+tolerance:
                reason = 'final_angle_limit'
            else:
                # Arm admission may occur after this review. Check the FULL
                # certified interval at the same tolerance as instantaneous
                # slew; the generic 1e-5 Cartesian tolerance can otherwise
                # admit a command outside an interior jerk extremum.
                certified_low, certified_high = max(p['bounds'][::2]), min(p['bounds'][1::2])
                if not certified_low-1e-8 <= final[4] <= certified_high+1e-8:
                    reason = 'final_dispatch_rocking_limit'
                _, bounds, _ = self.rocking.preview(p['rotation'], now_s, 1e-9)
                low, high = np.max(bounds[::2]), np.min(bounds[1::2])
                if not low-1e-8 <= final[4] <= high+1e-8:
                    reason = 'final_rocking_slew_limit'
                if p['guarded'] and not p['guard_bounds'][0]-tolerance <= final[4] <= p['guard_bounds'][1]+tolerance:
                    reason = 'final_mechanical_guard'
                face = self.transform @ final
                path = p['path'][:2]
                norm2 = float(path @ path)
                tangent = face[:2]-p['fixed'][:2]
                realised = 0. if norm2 < 1e-16 else float(tangent @ path/norm2)
                residual = tangent-realised*path
                if (np.max(np.abs(residual)) > tolerance or
                        realised*np.sqrt(norm2) < -tolerance or
                        (realised-p['alpha'])*np.sqrt(norm2) > tolerance):
                    reason = 'final_progress_limit'
                p['accepted_alpha'] = float(np.clip(realised, 0., p['alpha']))
                certificate = p['result'].hard_constraints
                values = certificate.A @ final
                row_tolerance = tolerance*np.linalg.norm(certificate.A, axis=1)
                if (np.any(values < certificate.lower-row_tolerance) or
                        np.any(values > certificate.upper+row_tolerance)):
                    reason = 'final_allocation_constraint'
        self._emit('pad_publication_review', accepted=reason is None, reason=reason,
                   final_tool=final, facts=facts, final_is_measurement=False,
                   review_time_s=now_s, dispatch_earliest_s=p['dispatch_earliest_s'],
                   dispatch_latest_s=p['dispatch_latest_s'])
        self.last_review_reason = reason
        if reason is not None:
            return False
        p.update(state='reviewed', reviewed=final.copy(), review_time=now_s)
        return True

    def started(self):
        p = self._pending
        if p is None or p['state'] != 'reviewed':
            return False
        now = self.clock()
        if now-p['now'] >= self.settings['command_max_age_s'] or now >= p['source_deadline']:
            return False
        # Recheck at dispatch: proposal/review timing cannot stand in for the
        # timestamp of the command that is actually published. Completion of
        # the transport calls is recorded separately and never used as a hold.
        p['state'] = 'prepared'
        if not self.review(p['reviewed'], now_s=now):
            return False
        p['publication_time_s'] = now
        p['state'] = 'sending'
        return True

    def arm_dispatch_time(self, now_s):
        """Bind history to the runner's final arm-call admission timestamp.

        The already-reviewed command is certified throughout this window, so
        no logging or second optimizer is needed at the transport boundary.
        """
        p = self._pending
        if (p is None or p['state'] != 'sending' or
                not p['dispatch_earliest_s'] <= now_s <= p['dispatch_latest_s'] or
                now_s >= p['source_deadline'] or
                now_s-p['now'] >= self.settings['command_max_age_s']):
            return False
        p['publication_time_s'] = float(now_s)
        return True

    def commit(self, final_tool, *, now_s, facts=None):
        p = self._pending
        final = np.asarray(final_tool, dtype=float).reshape(6)
        if p is None or p['state'] != 'sending' or not np.array_equal(final, p['reviewed']):
            raise RuntimeError('pad commit requires the exact reviewed and dispatched command')
        completion_time = now_s
        now_s = p['publication_time_s']
        face = self.transform @ final
        geometric = 0. if p['cop'] is None else p['cop']*(face[4]-p['nominal'][4])
        loading = float(face[2]-geometric)
        pure = final.copy()
        pure -= np.linalg.solve(self.transform, np.array([0.,0.,geometric,0.,0.,0.]))
        force = pure.copy()
        force[[0,1,3,5]] = 0.
        # Angle history covers the complete published rotation, not only the
        # My contribution or the candidate before inverse kinematics.
        elapsed = now_s-self.rocking.time_s
        self.force_law.commit_applied(force, final_full_twist=pure, accepted_normal_z=loading,
                                     accepted_angular_dt_s=elapsed)
        publication_audit = self.rocking.publication_audit(final, p['rotation'], now_s,
                                                         1e-9, 1, 1e-8)
        self.rocking.commit(final, p['rotation'], now_s)
        audit = dict(loading_accepted_m_s=loading, cop_geometric_m_s=geometric,
                     cop_pairing_residual_m_s=face[2]-p['nominal'][2]-geometric)
        self._last_final, self._last_rotation = final.copy(), p['rotation'].copy()
        self._last_alpha, self._last_commit_s = p['accepted_alpha'], now_s
        self._pending = None
        self._emit('pad_publication_commit', final_tool=final, final_is_measurement=False,
                   transaction='committed', publication_time_s=now_s,
                   transport_completion_time_s=completion_time, rocking_audit=publication_audit,
                   facts=facts, **audit, **self._record_measurement)

    def abort(self, reason, *, definitely_not_sent=False, facts=None, latch=None):
        if self._pending is None:
            return
        try:
            self.force_law.abort()
        finally:
            self._pending = None
            if latch is None:
                latch = not any(token in str(reason) for token in
                                ('native_timeout_coast', 'publication_infeasible'))
            self._faulted = bool(latch)
            self._emit('pad_publication_abort', reason=reason, definitely_not_sent=definitely_not_sent,
                       transaction='aborted', facts=facts, latch=self._faulted)

    def _supervise_force(self, force_n, now):
        """Keep the 4 N loop running through a brief 6 N overshoot.

        Recorded 17:23 exits aborted the mode on the first sample above 6 N
        while the command was already retracting. That leaves the probe loaded.
        Persist at 6 N, latch immediately at the higher bound.
        """
        latch_n = float(self.settings['hard_stop_latch_n'])
        persist_s = float(self.settings['hard_stop_persist_s'])
        if force_n >= latch_n:
            self._emit('pad_input_rejected', reason='force_latch', force_n=force_n,
                       latch_n=latch_n)
            raise RuntimeError(f'pad supervisory force stop ({latch_n:g} N)')
        if force_n >= self.settings['hard_stop_n']:
            if self._force_supervisor_since is None:
                self._force_supervisor_since = now
            held = now - self._force_supervisor_since
            self._emit('pad_force_supervisor', force_n=force_n, persist_s=held,
                       persist_limit_s=persist_s)
            if held >= persist_s:
                raise RuntimeError('pad supervisory force stop (6 N)')
            return
        self._force_supervisor_since = None

    def close(self, *, wait=False):
        self.abort('phase_exit')
        self.pad_visual.close(wait=wait)
